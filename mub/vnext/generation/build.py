from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts import MemUpdateTask, Split, TaskFamily
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.generation.family_b import generate_family_b_cores
from mub.vnext.generation.family_c import generate_family_c_cores
from mub.vnext.generation.family_d import generate_family_d_cores
from mub.vnext.generation.render import render_core
from mub.vnext.generation.splits import SplitAssignmentResult, assign_splits
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import validate_gold_replay, validate_task

_EXPECTED_CORE_COUNT = 480
_EXPECTED_TASK_COUNT = 1440
_EXPECTED_SPLIT_COUNTS = {
    Split.TRAIN: 1008,
    Split.DEV: 144,
    Split.TEST: 288,
}
_EXPECTED_FAMILY_COUNT = 360
_SPLIT_ORDER = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
_FAMILY_ORDER = {
    TaskFamily.REPEATED_SAME_SLOT.value: 0,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: 1,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: 2,
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: 3,
}


@dataclass(frozen=True, slots=True)
class CompiledPilotTasks:
    split_assignment: SplitAssignmentResult
    config_sha256: str
    tasks_jsonl: bytes

    def __post_init__(self) -> None:
        _validate_compiled_snapshot(self)

    @property
    def tasks(self) -> tuple[MemUpdateTask, ...]:
        return _parse_tasks_jsonl(self.tasks_jsonl)


@dataclass(frozen=True, slots=True)
class _RenderedTask:
    core: SemanticCore
    split: Split
    surface_variant: int
    task: MemUpdateTask


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


def _verify_round_trip(
    tasks: tuple[MemUpdateTask, ...],
    tasks_jsonl: bytes,
) -> None:
    reparsed = _parse_tasks_jsonl(tasks_jsonl)
    if len(reparsed) != len(tasks):
        raise ValueError("canonical tasks JSONL row count changed during serialization")
    for row_number, (task, parsed) in enumerate(
        zip(tasks, reparsed, strict=True), start=1
    ):
        if parsed != task:
            raise ValueError(
                f"canonical tasks JSONL row {row_number} changed its task record"
            )
    if reparsed != tasks or _canonical_jsonl(reparsed) != tasks_jsonl:
        raise ValueError("canonical tasks JSONL failed exact record/byte round trip")


def _linked_ids(task: MemUpdateTask):
    yield "task", task.task_id
    yield "source", task.source.source_id
    for event in task.events:
        yield "event", event.event_id
    for action in task.gold.actions:
        yield "action", action.action_id
    for query in task.queries:
        yield "query", query.query_id


def _validate_compiled_snapshot(snapshot: CompiledPilotTasks) -> None:
    if not isinstance(snapshot.split_assignment, SplitAssignmentResult):
        raise TypeError("split_assignment must be a SplitAssignmentResult")
    if (
        type(snapshot.config_sha256) is not str
        or len(snapshot.config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in snapshot.config_sha256)
    ):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")

    tasks = _parse_tasks_jsonl(snapshot.tasks_jsonl)
    if len(tasks) != _EXPECTED_TASK_COUNT:
        raise ValueError("compiled snapshot must contain exactly 1440 task rows")
    if tasks != tuple(sorted(tasks, key=_task_sort_key)):
        raise ValueError("compiled snapshot tasks are not in canonical order")
    if Counter(task.metadata.split for task in tasks) != Counter(
        _EXPECTED_SPLIT_COUNTS
    ):
        raise ValueError("compiled snapshot task split counts are inconsistent")
    expected_family_counts = Counter(
        {family: _EXPECTED_FAMILY_COUNT for family in _FAMILY_ORDER}
    )
    if Counter(task.task_family for task in tasks) != expected_family_counts:
        raise ValueError("compiled snapshot task family counts are inconsistent")

    assignments = snapshot.split_assignment.assignments
    assignment_by_core = {
        assignment.semantic_core_id: assignment for assignment in assignments
    }
    if len(assignments) != _EXPECTED_CORE_COUNT or len(assignment_by_core) != len(
        assignments
    ):
        raise ValueError("compiled snapshot must contain 480 unique split assignments")

    variants_by_core: dict[str, set[Any]] = defaultdict(set)
    hashes_by_core: dict[str, set[str]] = defaultdict(set)
    linked_ids: set[str] = set()
    for task in tasks:
        core_id = task.metadata.split_key.semantic_core_id
        assignment = assignment_by_core.get(core_id)
        if assignment is None:
            raise ValueError(
                f"compiled snapshot task {task.task_id!r} has no split assignment"
            )
        if (
            task.metadata.split != assignment.split
            or task.task_family != assignment.task_family
            or task.difficulty != assignment.difficulty
        ):
            raise ValueError(
                f"compiled snapshot task {task.task_id!r} disagrees with its split assignment"
            )
        if (
            task.metadata.generation_config_hash != snapshot.config_sha256
            or task.source.generator is None
            or task.source.generator.config_sha256 != snapshot.config_sha256
        ):
            raise ValueError(
                f"compiled snapshot task {task.task_id!r} disagrees with config_sha256"
            )
        variants_by_core[core_id].add(task.metadata.extra.get("surface_variant"))
        hashes_by_core[core_id].add(semantic_task_hash(task))
        for _, linked_id in _linked_ids(task):
            if linked_id in linked_ids:
                raise ValueError(
                    f"compiled snapshot contains duplicate linked ID {linked_id!r}"
                )
            linked_ids.add(linked_id)

    if set(variants_by_core) != set(assignment_by_core):
        raise ValueError("compiled snapshot core coverage disagrees with split assignments")
    if any(variants != {0, 1, 2} for variants in variants_by_core.values()):
        raise ValueError("compiled snapshot cores must each contain variants {0,1,2}")
    if any(len(hashes) != 1 for hashes in hashes_by_core.values()):
        raise ValueError("compiled snapshot core variants must share one semantic hash")


def _linkage_issues(rendered: tuple[_RenderedTask, ...]) -> list[str]:
    issues: list[str] = []
    variants_by_core: dict[str, list[Any]] = defaultdict(list)
    hashes_by_core: dict[str, list[str]] = defaultdict(list)
    linked_ids: dict[str, str] = {}

    for row_number, record in enumerate(rendered, start=1):
        task = record.task
        expected = (
            record.core.core_id,
            record.split,
            record.core.task_family.value,
            record.core.difficulty,
            record.surface_variant,
        )
        observed = (
            task.metadata.split_key.semantic_core_id,
            task.metadata.split,
            task.task_family,
            task.difficulty,
            task.metadata.extra.get("surface_variant"),
        )
        field_names = (
            "semantic_core_id",
            "split",
            "task_family",
            "difficulty",
            "surface_variant",
        )
        for field_name, expected_value, observed_value in zip(
            field_names, expected, observed, strict=True
        ):
            if (
                type(observed_value) is not type(expected_value)
                or observed_value != expected_value
            ):
                issues.append(
                    f"linkage row={row_number} task={task.task_id!r} field={field_name} "
                    f"expected={expected_value!r} observed={observed_value!r}"
                )

        variants_by_core[record.core.core_id].append(observed[-1])
        hashes_by_core[record.core.core_id].append(semantic_task_hash(task))
        for id_kind, linked_id in _linked_ids(task):
            location = f"row={row_number}:{id_kind}"
            first_location = linked_ids.get(linked_id)
            if first_location is not None:
                issues.append(
                    f"duplicate linked ID {linked_id!r}: first={first_location} "
                    f"duplicate={location}"
                )
            else:
                linked_ids[linked_id] = location

    for core_id in sorted(variants_by_core):
        variants = variants_by_core[core_id]
        if len(variants) != 3 or set(variants) != {0, 1, 2}:
            issues.append(
                f"core {core_id!r} must have exactly surface variants [0, 1, 2]; "
                f"observed={variants!r}"
            )
        hashes = hashes_by_core[core_id]
        if len(set(hashes)) != 1:
            issues.append(
                f"core {core_id!r} surface variants do not share one semantic hash: "
                f"observed={hashes!r}"
            )
    return issues


def _validation_issues(tasks: tuple[MemUpdateTask, ...]) -> list[str]:
    issues: list[str] = []
    validators = (
        ("task", validate_task),
        ("gold_replay", validate_gold_replay),
    )
    for task in tasks:
        for stage, validator in validators:
            try:
                report = validator(task)
            except Exception as exc:
                issues.append(
                    f"validation stage={stage} task={task.task_id!r} "
                    f"validator_exception={type(exc).__name__}: {exc}"
                )
                continue
            for issue in report.issues:
                if issue.severity != "error":
                    continue
                issues.append(
                    f"validation stage={stage} task={task.task_id!r} "
                    f"issue={issue.code}@{issue.path}: {issue.message}"
                )
    return issues


def _validate_fixed_config(config: PilotConfig) -> None:
    if config.surface_variants_per_core != 3:
        raise ValueError("Pilot compilation requires surface_variants_per_core == 3")
    if config.total_semantic_cores != _EXPECTED_CORE_COUNT:
        raise ValueError("Pilot compilation requires total_semantic_cores == 480")
    if config.total_tasks != _EXPECTED_TASK_COUNT:
        raise ValueError("Pilot compilation requires total_tasks == 1440")


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
        _RenderedTask(
            core=core,
            split=split_by_core[core.core_id],
            surface_variant=surface_variant,
            task=render_core(
                core,
                split=split_by_core[core.core_id],
                surface_variant=surface_variant,
                context=context,
            ),
        )
        for core in ordered_cores
        for surface_variant in range(3)
    )
    tasks = tuple(record.task for record in rendered)

    if len(tasks) != _EXPECTED_TASK_COUNT:
        raise ValueError(
            "Pilot compilation requires exactly 1440 tasks; "
            f"observed {len(tasks)}"
        )
    split_counts = Counter(task.metadata.split for task in tasks)
    if split_counts != Counter(_EXPECTED_SPLIT_COUNTS):
        raise ValueError(
            f"Pilot task split counts must be {_EXPECTED_SPLIT_COUNTS}; "
            f"observed {dict(split_counts)}"
        )
    family_counts = Counter(task.task_family for task in tasks)
    expected_family_counts = Counter(
        {family: _EXPECTED_FAMILY_COUNT for family in _FAMILY_ORDER}
    )
    if family_counts != expected_family_counts:
        raise ValueError(
            "Pilot task family counts must be 360 each; "
            f"observed {dict(family_counts)}"
        )
    if tasks != tuple(sorted(tasks, key=_task_sort_key)):
        raise ValueError("Pilot tasks are not in canonical order")

    issues = _linkage_issues(rendered)
    issues.extend(_validation_issues(tasks))
    if issues:
        raise ValueError("Pilot task compilation failed:\n" + "\n".join(issues))

    tasks_jsonl = _canonical_jsonl(tasks)
    _verify_round_trip(tasks, tasks_jsonl)
    return CompiledPilotTasks(
        split_assignment=split_assignment,
        config_sha256=context.config_sha256,
        tasks_jsonl=tasks_jsonl,
    )


__all__ = ["CompiledPilotTasks", "compile_pilot_tasks"]
