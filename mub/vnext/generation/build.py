from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Any

from mub.vnext.contracts import MemUpdateTask, Split, TaskFamily
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.generation.family_b import generate_family_b_cores
from mub.vnext.generation.family_c import generate_family_c_cores
from mub.vnext.generation.family_d import generate_family_d_cores
from mub.vnext.generation.render import (
    _RenderedTask as _RenderEnvelope,
    _RenderPlan,
    _expected_render_plan,
    _render_core_unvalidated,
    _render_envelope_issues,
)
from mub.vnext.generation.splits import (
    SplitAssignmentResult,
    _validate_split_assignment_result,
    assign_splits,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import validate_gold_replay, validate_task

_COMPILED_SNAPSHOT_SEAL_TOKEN = object()
_EXPECTED_CORE_COUNT = 480
_EXPECTED_TASK_COUNT = 1440
_EXPECTED_SPLIT_COUNTS = {
    Split.TRAIN: 1008,
    Split.DEV: 144,
    Split.TEST: 288,
}
_EXPECTED_FAMILY_COUNT = 360
_EXPECTED_CORES_PER_FAMILY = 120
_EXPECTED_CORE_SPLIT_COUNTS = {
    Split.TRAIN: 336,
    Split.DEV: 48,
    Split.TEST: 96,
}
_EXPECTED_FAMILY_CORE_SPLIT_COUNTS = {
    Split.TRAIN: 84,
    Split.DEV: 12,
    Split.TEST: 24,
}
_SPLIT_ORDER = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
_FAMILY_ORDER = {
    TaskFamily.REPEATED_SAME_SLOT.value: 0,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: 1,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: 2,
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: 3,
}


def _compiled_snapshot_seal(snapshot: CompiledPilotTasks) -> str:
    if not isinstance(snapshot.split_assignment, SplitAssignmentResult):
        raise ValueError("split_assignment must be a SplitAssignmentResult")
    if type(snapshot.tasks_jsonl) is not bytes:
        raise ValueError("tasks_jsonl must be exact bytes")
    for field_name in (
        "config_sha256",
        "code_revision",
        "compiler_version",
        "generator_name",
    ):
        value = getattr(snapshot, field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(f"{field_name} must be a nonblank exact string")
    digest = hashlib.sha256()
    values = (
        canonical_json_bytes(snapshot.split_assignment),
        snapshot.config_sha256.encode("utf-8"),
        snapshot.code_revision.encode("utf-8"),
        snapshot.compiler_version.encode("utf-8"),
        snapshot.generator_name.encode("utf-8"),
        snapshot.tasks_jsonl,
    )
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledPilotTasks:
    split_assignment: SplitAssignmentResult
    config_sha256: str
    code_revision: str
    compiler_version: str
    generator_name: str
    tasks_jsonl: bytes
    _compile_issues: InitVar[tuple[str, ...]] = ()
    _seal_token: InitVar[object | None] = None
    _snapshot_seal: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
        _compile_issues: tuple[str, ...],
        _seal_token: object | None,
    ) -> None:
        _validate_compiled_snapshot(self, _compile_issues)
        if _seal_token is _COMPILED_SNAPSHOT_SEAL_TOKEN:
            object.__setattr__(self, "_snapshot_seal", _compiled_snapshot_seal(self))

    @staticmethod
    def parse_tasks_jsonl(tasks_jsonl: bytes) -> tuple[MemUpdateTask, ...]:
        return _parse_tasks_jsonl(tasks_jsonl)

    @staticmethod
    def validated_task_set(
        tasks_jsonl: bytes,
        *,
        config_sha256: str,
        code_revision: str,
        compiler_version: str,
        generator_name: str,
        seed: int,
    ) -> tuple[MemUpdateTask, ...]:
        tasks = _parse_tasks_jsonl(tasks_jsonl)
        _raise_task_set_issues(
            tasks,
            config_sha256=config_sha256,
            code_revision=code_revision,
            compiler_version=compiler_version,
            generator_name=generator_name,
            seed=seed,
        )
        return tasks

    @property
    def tasks(self) -> tuple[MemUpdateTask, ...]:
        return self.parse_tasks_jsonl(self.tasks_jsonl)

    def validated_tasks(self) -> tuple[MemUpdateTask, ...]:
        return _validated_snapshot_tasks(self)

    def verify_authenticated_snapshot(self) -> None:
        """Reject ordinary replacement/serialization tampering.

        This process-local seal is not cryptographic authentication against code
        that intentionally introspects private module state or recomputes it.
        """
        try:
            current_seal = _compiled_snapshot_seal(self)
            if (
                type(self._snapshot_seal) is not str
                or self._snapshot_seal != current_seal
            ):
                raise ValueError("seal mismatch")
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "compiled snapshot is not an authenticated compiler output"
            ) from None

    def authenticated_clone(self) -> CompiledPilotTasks:
        """Return a sealed copy of this unchanged authenticated snapshot."""
        self.verify_authenticated_snapshot()
        return type(self)(
            split_assignment=self.split_assignment,
            config_sha256=self.config_sha256,
            code_revision=self.code_revision,
            compiler_version=self.compiler_version,
            generator_name=self.generator_name,
            tasks_jsonl=self.tasks_jsonl,
            _seal_token=_COMPILED_SNAPSHOT_SEAL_TOKEN,
        )

    def validate_snapshot_binding(
        self,
        tasks: tuple[MemUpdateTask, ...],
    ) -> None:
        issues = _snapshot_consistency_issues(self, tasks)
        if issues:
            raise ValueError(_render_bounded_diagnostics(issues))


@dataclass(frozen=True, slots=True)
class _CompiledRender:
    core: SemanticCore
    split: Split
    surface_variant: int
    envelope: _RenderEnvelope
    expected_plan: _RenderPlan

    @property
    def task(self) -> MemUpdateTask:
        return self.envelope.task


def _task_sort_key(task: MemUpdateTask) -> tuple[int, int, str, int]:
    return (
        _SPLIT_ORDER[task.metadata.split],
        _FAMILY_ORDER[task.task_family],
        task.metadata.split_key.semantic_core_id,
        task.metadata.extra["surface_variant"],
    )


def _canonical_jsonl(tasks: tuple[MemUpdateTask, ...]) -> bytes:
    return b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)


def _parse_tasks_jsonl(tasks_jsonl: bytes) -> tuple[MemUpdateTask, ...]:
    if type(tasks_jsonl) is not bytes:
        raise TypeError("tasks_jsonl must be exact bytes")
    if tasks_jsonl.startswith(b"\xef\xbb\xbf"):
        raise ValueError("canonical tasks JSONL must not contain a UTF-8 BOM")
    if b"\r" in tasks_jsonl:
        raise ValueError("canonical tasks JSONL must use LF-only framing")
    if not tasks_jsonl.endswith(b"\n"):
        raise ValueError("canonical tasks JSONL must end with a final LF")
    rows = tasks_jsonl[:-1].split(b"\n")
    if any(not row for row in rows):
        raise ValueError("canonical tasks JSONL must not contain blank rows")
    parsed = tuple(MemUpdateTask.model_validate_json(row) for row in rows)
    if _canonical_jsonl(parsed) != tasks_jsonl:
        raise ValueError("canonical tasks JSONL bytes are not exact canonical encoding")
    return parsed


def _linked_ids(task: MemUpdateTask):
    yield "task", task.task_id
    yield "source", task.source.source_id
    for event in task.events:
        yield "event", event.event_id
    for action in task.gold.actions:
        yield "action", action.action_id
    for query in task.queries:
        yield "query", query.query_id


def _validated_snapshot_tasks(
    snapshot: CompiledPilotTasks,
    compile_issues: tuple[str, ...] = (),
) -> tuple[MemUpdateTask, ...]:
    tasks = _parse_tasks_jsonl(snapshot.tasks_jsonl)
    expected_seed = getattr(
        getattr(snapshot.split_assignment, "split_balance", None),
        "seed",
        -1,
    )
    issues = _task_set_consistency_issues(
        tasks,
        config_sha256=snapshot.config_sha256,
        code_revision=snapshot.code_revision,
        compiler_version=snapshot.compiler_version,
        generator_name=snapshot.generator_name,
        seed=expected_seed,
    )
    issues.extend(_snapshot_consistency_issues(snapshot, tasks))
    issues.extend(compile_issues)
    if issues:
        raise ValueError(_render_bounded_diagnostics(issues))
    return tasks


def _validate_compiled_snapshot(
    snapshot: CompiledPilotTasks,
    compile_issues: tuple[str, ...],
) -> None:
    _validated_snapshot_tasks(snapshot, compile_issues)


def _linkage_issues(rendered: tuple[_CompiledRender, ...]) -> list[str]:
    issues: list[str] = []
    for row_number, record in enumerate(rendered, start=1):
        try:
            integrity_issues = _render_envelope_issues(
                record.envelope,
                record.expected_plan,
            )
        except Exception as exc:
            integrity_issues = (
                f"envelope verification exception={type(exc).__name__}: {exc}",
            )
        for detail in integrity_issues:
            issues.append(
                f"stage=render_receipt code=envelope_integrity "
                f"core={record.core.core_id} row={row_number} "
                f"task={record.task.task_id!r} variant={record.surface_variant} "
                f"detail={detail}"
            )
    return issues


def _diagnostic_token(message: str, name: str, default: str) -> str:
    prefix = f"{name}="
    for token in message.split():
        if token.startswith(prefix):
            return token[len(prefix) :].split("@", 1)[0].strip("':,")
    return default


def _render_bounded_diagnostics(issues: list[str]) -> str:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    counts: Counter[tuple[str, str, str]] = Counter()
    for message in issues:
        key = (
            _diagnostic_token(message, "stage", "snapshot"),
            _diagnostic_token(message, "code", "consistency"),
            _diagnostic_token(message, "core", "global"),
        )
        counts[key] += 1
        sample = message[:512]
        if sample in groups[key]:
            continue
        group_samples = groups[key]
        # Preserve the first root-cause labels and terminal consistency labels.
        if len(group_samples) < 3:
            group_samples.append(sample)
        elif len(group_samples) < 5:
            group_samples.append(sample)
        else:
            group_samples[-2] = group_samples[-1]
            group_samples[-1] = sample

    lines = [
        f"compiled Pilot snapshot failed: total_evidence={len(issues)} "
        f"groups={len(counts)}"
    ]
    omitted_groups = 0
    for key in sorted(counts):
        stage, code, core_id = key
        group_lines = [
            f"group stage={stage} code={code} core={core_id} count={counts[key]}"
        ]
        group_lines.extend(f"  sample={sample}" for sample in groups[key])
        candidate = "\n".join((*lines, *group_lines))
        if len(candidate.encode("utf-8")) > 60_000:
            omitted_groups += 1
            continue
        lines.extend(group_lines)
    if omitted_groups:
        lines.append(f"diagnostics_truncated omitted_groups={omitted_groups}")
    rendered = "\n".join(lines)
    if len(rendered.encode("utf-8")) > 65_536:
        raise AssertionError("bounded diagnostics exceeded 64 KiB")
    return rendered


def _validation_issues(tasks: tuple[MemUpdateTask, ...]) -> list[str]:
    issues: list[str] = []
    validators = (
        ("task", validate_task),
        ("gold_replay", validate_gold_replay),
    )
    for task in tasks:
        core_id = task.metadata.split_key.semantic_core_id
        for stage, validator in validators:
            try:
                report = validator(task)
            except Exception as exc:
                issues.append(
                    f"stage=validation_{stage} code=validator_exception "
                    f"core={core_id} task={task.task_id!r} "
                    f"exception={type(exc).__name__}: {exc}"
                )
                continue
            for issue in report.issues:
                if issue.severity != "error":
                    continue
                issues.append(
                    f"stage=validation_{stage} code={issue.code} core={core_id} "
                    f"task={task.task_id!r} path={issue.path}: {issue.message}"
                )
    return issues


def _task_set_consistency_issues(
    tasks: tuple[MemUpdateTask, ...],
    *,
    config_sha256: str,
    code_revision: str,
    compiler_version: str,
    generator_name: str,
    seed: int,
) -> list[str]:
    issues = _artifact_issues(tasks)
    issues.extend(_validation_issues(tasks))
    variants_by_core: dict[str, set[Any]] = defaultdict(set)
    hashes_by_core: dict[str, set[str]] = defaultdict(set)
    core_records: dict[str, tuple[str, Any, Split]] = {}
    core_by_semantic_hash: dict[str, str] = {}
    linked_ids: dict[str, str] = {}

    for row_number, task in enumerate(tasks, start=1):
        core_id = task.metadata.split_key.semantic_core_id
        generator = task.source.generator
        if (
            task.metadata.generation_config_hash != config_sha256
            or task.metadata.compiler_version != compiler_version
            or generator is None
            or generator.config_sha256 != config_sha256
            or generator.code_revision != code_revision
            or generator.compiler_version != compiler_version
            or generator.generator_name != generator_name
            or generator.seed != seed
        ):
            issues.append(
                f"stage=task_set code=generator_provenance core={core_id} "
                f"row={row_number} task={task.task_id!r} "
                "disagrees with generator provenance"
            )

        core_record = (task.task_family, task.difficulty, task.metadata.split)
        previous_record = core_records.setdefault(core_id, core_record)
        if previous_record != core_record:
            issues.append(
                f"stage=task_set code=core_metadata core={core_id} "
                "variants disagree on family, difficulty, or split"
            )
        variant = task.metadata.extra.get("surface_variant")
        if type(variant) is not int:
            issues.append(
                f"stage=task_set code=invalid_surface_variant core={core_id} "
                f"row={row_number} task={task.task_id!r} "
                "surface_variant must be an exact built-in integer"
            )
        else:
            variants_by_core[core_id].add(variant)
        try:
            semantic_hash = semantic_task_hash(task)
            hashes_by_core[core_id].add(semantic_hash)
            first_core_id = core_by_semantic_hash.setdefault(semantic_hash, core_id)
            if first_core_id != core_id:
                issues.append(
                    f"stage=semantic_hash code=cross_core_collision core={core_id} "
                    f"hash={semantic_hash} first_core={first_core_id}"
                )
        except Exception as exc:
            issues.append(
                f"stage=semantic_hash code=hash_exception core={core_id} "
                f"row={row_number} task={task.task_id!r} "
                f"exception={type(exc).__name__}: {exc}"
            )
        for id_kind, linked_id in _linked_ids(task):
            location = f"row={row_number}:{id_kind}"
            first_location = linked_ids.get(linked_id)
            if first_location is not None:
                issues.append(
                    f"stage=task_set code=duplicate_{id_kind}_id core={core_id} "
                    f"duplicate linked ID={linked_id!r} "
                    f"first={first_location} duplicate={location}"
                )
            else:
                linked_ids[linked_id] = location

    if len(core_records) != _EXPECTED_CORE_COUNT:
        issues.append(
            f"stage=task_set code=core_count core=global "
            f"expected={_EXPECTED_CORE_COUNT} observed={len(core_records)}"
        )
    if set(variants_by_core) != set(core_records):
        issues.append(
            "stage=task_set code=core_coverage core=global "
            "variant coverage disagrees with semantic-core records"
        )
    for core_id in sorted(variants_by_core):
        if variants_by_core[core_id] != {0, 1, 2}:
            issues.append(
                f"stage=task_set code=surface_variants core={core_id} "
                f"observed={sorted(variants_by_core[core_id], key=repr)!r}"
            )
        if len(hashes_by_core[core_id]) != 1:
            issues.append(
                f"stage=semantic_hash code=variant_mismatch core={core_id} "
                "variants must share one semantic hash"
            )

    core_split_counts = Counter(record[2] for record in core_records.values())
    if core_split_counts != Counter(_EXPECTED_CORE_SPLIT_COUNTS):
        issues.append(
            f"stage=task_set code=core_split_quotas core=global "
            f"expected={_EXPECTED_CORE_SPLIT_COUNTS!r} "
            f"observed={dict(core_split_counts)!r}"
        )
    for family in _FAMILY_ORDER:
        family_records = [
            record for record in core_records.values() if record[0] == family
        ]
        if len(family_records) != _EXPECTED_CORES_PER_FAMILY:
            issues.append(
                f"stage=task_set code=family_core_count core=global "
                f"family={family} expected={_EXPECTED_CORES_PER_FAMILY} "
                f"observed={len(family_records)}"
            )
        family_splits = Counter(record[2] for record in family_records)
        if family_splits != Counter(_EXPECTED_FAMILY_CORE_SPLIT_COUNTS):
            issues.append(
                f"stage=task_set code=family_split_quotas core=global "
                f"family={family} expected={_EXPECTED_FAMILY_CORE_SPLIT_COUNTS!r} "
                f"observed={dict(family_splits)!r}"
            )
    return issues


def _raise_task_set_issues(
    tasks: tuple[MemUpdateTask, ...],
    *,
    config_sha256: str,
    code_revision: str,
    compiler_version: str,
    generator_name: str,
    seed: int,
) -> None:
    issues = _task_set_consistency_issues(
        tasks,
        config_sha256=config_sha256,
        code_revision=code_revision,
        compiler_version=compiler_version,
        generator_name=generator_name,
        seed=seed,
    )
    if issues:
        raise ValueError(_render_bounded_diagnostics(issues))


def _snapshot_consistency_issues(
    snapshot: CompiledPilotTasks,
    tasks: tuple[MemUpdateTask, ...],
) -> list[str]:
    issues: list[str] = []
    if not isinstance(snapshot.split_assignment, SplitAssignmentResult):
        issues.append("split_assignment must be a SplitAssignmentResult")
        return issues
    try:
        _validate_split_assignment_result(snapshot.split_assignment)
    except (TypeError, ValueError) as exc:
        detail = str(exc)
        prefix = "inconsistent split assignment: "
        details = (
            detail[len(prefix) :].split("; ")
            if detail.startswith(prefix)
            else [detail]
        )
        for item in details:
            if item:
                issues.append(
                    f"stage=split_assignment code=inconsistent core=global "
                    f"detail={item}"
                )
    if (
        type(snapshot.config_sha256) is not str
        or len(snapshot.config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in snapshot.config_sha256
        )
    ):
        issues.append("config_sha256 must be a lowercase SHA-256 digest")
    for field_name in ("code_revision", "compiler_version", "generator_name"):
        value = getattr(snapshot, field_name)
        if type(value) is not str or not value.strip():
            issues.append(f"{field_name} must be a nonblank string")

    assignments = snapshot.split_assignment.assignments
    assignment_by_core = {
        assignment.semantic_core_id: assignment for assignment in assignments
    }
    if len(assignments) != _EXPECTED_CORE_COUNT or len(assignment_by_core) != len(
        assignments
    ):
        issues.append("compiled snapshot must contain 480 unique split assignments")

    task_core_ids = set()
    for row_number, task in enumerate(tasks, start=1):
        core_id = task.metadata.split_key.semantic_core_id
        task_core_ids.add(core_id)
        assignment = assignment_by_core.get(core_id)
        if assignment is None:
            issues.append(
                f"stage=snapshot code=missing_assignment core={core_id} "
                f"row={row_number} task={task.task_id!r} has no split assignment"
            )
            continue
        expected_fields = (
            ("split", assignment.split, task.metadata.split),
            ("task_family", assignment.task_family.value, task.task_family),
            ("difficulty", assignment.difficulty, task.difficulty),
        )
        for field_name, expected, observed in expected_fields:
            if expected != observed:
                issues.append(
                    f"stage=snapshot code=field_{field_name} core={core_id} "
                    f"row={row_number} task={task.task_id!r} "
                    f"expected={expected!r} observed={observed!r}"
                )
    if task_core_ids != set(assignment_by_core):
        issues.append(
            "stage=snapshot code=core_coverage core=global "
            "snapshot core coverage disagrees with split assignments"
        )
    return issues


def _artifact_issues(tasks: tuple[MemUpdateTask, ...]) -> list[str]:
    issues: list[str] = []
    if len(tasks) != _EXPECTED_TASK_COUNT:
        issues.append(
            f"stage=snapshot code=task_count core=global "
            f"task count expected={_EXPECTED_TASK_COUNT} observed={len(tasks)}"
        )

    try:
        split_counts = Counter(task.metadata.split for task in tasks)
    except Exception as exc:
        issues.append(
            f"stage=snapshot code=split_count_exception core=global "
            f"exception={type(exc).__name__}: {exc}"
        )
    else:
        if split_counts != Counter(_EXPECTED_SPLIT_COUNTS):
            issues.append(
                f"stage=snapshot code=split_counts core=global "
                f"split counts expected={_EXPECTED_SPLIT_COUNTS!r} "
                f"observed={dict(split_counts)!r}"
            )

    expected_family_counts = Counter(
        {family: _EXPECTED_FAMILY_COUNT for family in _FAMILY_ORDER}
    )
    try:
        family_counts = Counter(task.task_family for task in tasks)
    except Exception as exc:
        issues.append(
            f"stage=snapshot code=family_count_exception core=global "
            f"exception={type(exc).__name__}: {exc}"
        )
    else:
        if family_counts != expected_family_counts:
            issues.append(
                f"stage=snapshot code=family_counts core=global "
                f"family counts expected={dict(expected_family_counts)!r} "
                f"observed={dict(family_counts)!r}"
            )

    try:
        observed_keys = tuple(_task_sort_key(task) for task in tasks)
        expected_keys = tuple(sorted(observed_keys))
    except Exception as exc:
        issues.append(
            f"stage=snapshot code=canonical_order_exception core=global "
            f"exception={type(exc).__name__}: {exc}"
        )
    else:
        if observed_keys != expected_keys:
            issues.append(
                "stage=snapshot code=canonical_order core=global "
                "tasks are not in canonical order"
            )
    return issues


def _validate_fixed_config(config: PilotConfig) -> None:
    if config.surface_variants_per_core != 3:
        raise ValueError("Pilot compilation requires surface_variants_per_core == 3")
    if config.total_semantic_cores != _EXPECTED_CORE_COUNT:
        raise ValueError("Pilot compilation requires total_semantic_cores == 480")
    if config.total_tasks != _EXPECTED_TASK_COUNT:
        raise ValueError("Pilot compilation requires total_tasks == 1440")
    expected_split_tasks = {
        split.value: count for split, count in _EXPECTED_SPLIT_COUNTS.items()
    }
    if config.expected_split_tasks != expected_split_tasks:
        raise ValueError(
            "Pilot compilation requires split task counts "
            "{train: 1008, dev: 144, test: 288}"
        )


def _render_requested_task(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
) -> _CompiledRender:
    plan = _expected_render_plan(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
    )
    envelope = _render_core_unvalidated(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        plan=plan,
    )
    return _CompiledRender(
        core=core,
        split=split,
        surface_variant=surface_variant,
        envelope=envelope,
        expected_plan=plan,
    )


def compile_pilot_tasks(
    config: PilotConfig,
    *,
    code_revision: str,
) -> CompiledPilotTasks:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if type(code_revision) is not str:
        raise TypeError("code_revision must be a string")
    if not code_revision.strip():
        raise ValueError("code_revision must not be blank")
    _validate_fixed_config(config)

    context = GenerationContext(config=config, code_revision=code_revision)
    immutable_config = context.config
    cores = (
        *generate_family_a_cores(immutable_config),
        *generate_family_b_cores(immutable_config),
        *generate_family_c_cores(immutable_config),
        *generate_family_d_cores(immutable_config),
    )
    if len(cores) != _EXPECTED_CORE_COUNT:
        raise ValueError(
            "Pilot compilation requires exactly 480 semantic cores; "
            f"observed {len(cores)}"
        )

    split_assignment = assign_splits(cores, immutable_config.seed)
    split_by_core = {
        assignment.semantic_core_id: assignment.split
        for assignment in split_assignment.assignments
    }
    ordered_cores = sorted(
        cores,
        key=lambda core: (
            _SPLIT_ORDER[split_by_core[core.core_id]],
            _FAMILY_ORDER[core.task_family.value],
            core.core_id,
        ),
    )
    rendered = tuple(
        _render_requested_task(
            core,
            split=split_by_core[core.core_id],
            surface_variant=surface_variant,
            context=context,
        )
        for core in ordered_cores
        for surface_variant in range(3)
    )
    tasks = tuple(record.task for record in rendered)

    compile_issues = tuple(_linkage_issues(rendered))
    tasks_jsonl = _canonical_jsonl(tasks)
    compiled = CompiledPilotTasks(
        split_assignment=split_assignment,
        config_sha256=context.config_sha256,
        code_revision=context.code_revision,
        compiler_version=context.compiler_version,
        generator_name=context.generator_name,
        tasks_jsonl=tasks_jsonl,
        _compile_issues=compile_issues,
        _seal_token=_COMPILED_SNAPSHOT_SEAL_TOKEN,
    )
    if compiled.tasks != tasks:
        raise ValueError("canonical tasks JSONL changed records during round trip")
    return compiled


__all__ = ["CompiledPilotTasks", "compile_pilot_tasks"]
