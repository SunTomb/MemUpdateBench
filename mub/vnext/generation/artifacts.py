from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Final

from mub.vnext.contracts import (
    ArtifactRef,
    Difficulty,
    MemUpdateTask,
    Split,
    TaskFamily,
    TaskManifest,
)
from mub.vnext.generation.build import CompiledPilotTasks
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.splits import (
    CoreSplitAssignment,
    SplitAssignmentResult,
    SplitBalanceReport,
    _ranking_sha256_from_material,
    _stratum_sort_key,
    _validate_split_assignment_result,
)
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.validation import ValidationReport, validate_splits
from mub.vnext.validation.split import FAMILY_STRATIFICATION_AXES
from mub.vnext.version import COMPILER_VERSION

_TASKS_PATH: Final = "tasks.jsonl"
_CONFIG_PATH: Final = "generation_config.json"
_SPLIT_BALANCE_PATH: Final = "split_balance.json"
_TASK_MANIFEST_PATH: Final = "task_manifest.json"
_VALIDATION_REPORT_PATH: Final = "validation_report.json"
_JSON_MEDIA_TYPE: Final = "application/json"
_JSONL_MEDIA_TYPE: Final = "application/x-ndjson"
_CREATED_AT: Final = "deterministic-generation-provenance"
_STANDARD_SPLITS: Final = (Split.TRAIN, Split.DEV, Split.TEST)
_PILOT_FAMILIES: Final = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)


@dataclass(frozen=True, slots=True)
class InMemoryPilotArtifact:
    path: str
    content: bytes
    media_type: str
    record_count: int
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("content must be exact bytes")
        digest = hashlib.sha256(self.content).hexdigest()
        ref = ArtifactRef(
            path=self.path,
            sha256=digest,
            media_type=self.media_type,
            record_count=self.record_count,
        )
        object.__setattr__(self, "path", ref.path)
        object.__setattr__(self, "media_type", ref.media_type)
        object.__setattr__(self, "record_count", ref.record_count)
        object.__setattr__(self, "_sha256", digest)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            path=self.path,
            sha256=self._sha256,
            media_type=self.media_type,
            record_count=self.record_count,
        )


@dataclass(frozen=True, slots=True)
class PilotArtifactBundle:
    resolved_config: PilotConfig
    split_balance_report: SplitBalanceReport
    task_manifest: TaskManifest
    validation_report: ValidationReport
    artifacts: tuple[InMemoryPilotArtifact, ...]

    def __post_init__(self) -> None:
        _validate_public_bundle(self)

    @property
    def tasks_jsonl(self) -> bytes:
        return self.artifacts[0].content

    @property
    def resolved_config_bytes(self) -> bytes:
        return self.artifacts[1].content

    @property
    def config_sha256(self) -> str:
        return self.artifacts[1].ref.sha256

    @property
    def split_balance_bytes(self) -> bytes:
        return self.artifacts[2].content

    @property
    def task_manifest_bytes(self) -> bytes:
        return self.artifacts[3].content

    @property
    def validation_report_bytes(self) -> bytes:
        return self.artifacts[4].content


def _require_bundle_contracts(bundle: PilotArtifactBundle) -> None:
    expected = (
        ("resolved_config", bundle.resolved_config, PilotConfig),
        ("split_balance_report", bundle.split_balance_report, SplitBalanceReport),
        ("task_manifest", bundle.task_manifest, TaskManifest),
        ("validation_report", bundle.validation_report, ValidationReport),
    )
    for field_name, value, contract in expected:
        if not isinstance(value, contract):
            raise TypeError(f"{field_name} must be a {contract.__name__} contract")
    if type(bundle.artifacts) is not tuple:
        raise TypeError("artifacts must be an immutable tuple")
    if any(not isinstance(item, InMemoryPilotArtifact) for item in bundle.artifacts):
        raise TypeError("artifacts must contain InMemoryPilotArtifact records")


def _validate_artifact_envelopes(
    artifacts: tuple[InMemoryPilotArtifact, ...],
) -> None:
    expected = (
        (_TASKS_PATH, _JSONL_MEDIA_TYPE, None),
        (_CONFIG_PATH, _JSON_MEDIA_TYPE, 1),
        (_SPLIT_BALANCE_PATH, _JSON_MEDIA_TYPE, 1),
        (_TASK_MANIFEST_PATH, _JSON_MEDIA_TYPE, 1),
        (_VALIDATION_REPORT_PATH, _JSON_MEDIA_TYPE, 1),
    )
    if len(artifacts) != len(expected):
        raise ValueError("bundle must contain exactly five canonical artifacts")
    for index, (artifact, (path, media_type, record_count)) in enumerate(
        zip(artifacts, expected, strict=True)
    ):
        label = "task artifact" if index == 0 else f"artifact {path}"
        if artifact.path != path or artifact.media_type != media_type:
            raise ValueError(f"{label} path or media type is not canonical")
        if record_count is not None and artifact.record_count != record_count:
            raise ValueError(f"{label} record count is not canonical")
        actual_hash = hashlib.sha256(artifact.content).hexdigest()
        if artifact.ref.sha256 != actual_hash:
            raise ValueError(f"{label} hash does not match its content")


def _task_provenance(tasks: tuple[MemUpdateTask, ...]):
    records = set()
    for task in tasks:
        generator = task.source.generator
        if generator is None:
            raise ValueError("task artifact contains a task without generator provenance")
        record = (
            generator.generator_name,
            generator.compiler_version,
            generator.code_revision,
            generator.config_sha256,
            generator.seed,
        )
        if (
            task.metadata.generation_config_hash != generator.config_sha256
            or task.metadata.compiler_version != generator.compiler_version
        ):
            raise ValueError("task artifact generator provenance is internally inconsistent")
        records.add(record)
    if len(records) != 1:
        raise ValueError("task artifact must have one exact generator provenance binding")
    return next(iter(records))


def _strata_material_key(
    family: TaskFamily,
    difficulty: Difficulty,
    strata: tuple[tuple[str, object], ...],
) -> bytes:
    return json.dumps(
        {
            "task_family": family.value,
            "difficulty": difficulty.value,
            "strata": dict(strata),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_split_balance(
    tasks: tuple[MemUpdateTask, ...],
    config: PilotConfig,
    report: SplitBalanceReport,
) -> None:
    core_records = {}
    variant_counts = Counter()
    for task in tasks:
        try:
            family = TaskFamily(task.task_family)
            axes = FAMILY_STRATIFICATION_AXES[task.task_family]
            strata = tuple(
                (axis, task.metadata.resolved_profile[axis]) for axis in axes
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("split balance cannot resolve canonical task strata") from exc
        core_id = task.metadata.split_key.semantic_core_id
        record = (family, task.difficulty, strata, task.metadata.split)
        previous = core_records.setdefault(core_id, record)
        if previous != record:
            raise ValueError("split balance found inconsistent variants within one core")
        variant_counts[core_id] += 1

    if len(tasks) != config.total_tasks or len(core_records) != config.total_semantic_cores:
        raise ValueError("split balance task or semantic-core total disagrees with config")
    if any(
        count != config.surface_variants_per_core
        for count in variant_counts.values()
    ):
        raise ValueError("split balance core variant count disagrees with config")
    family_core_counts = Counter(record[0] for record in core_records.values())
    if any(family_core_counts[family] != config.cores_per_family for family in _PILOT_FAMILIES):
        raise ValueError("split balance per-family core total disagrees with config")

    core_counts = Counter(record[3] for record in core_records.values())
    task_counts = Counter(task.metadata.split for task in tasks)
    expected_core_counts = {
        split.value: core_counts[split] for split in _STANDARD_SPLITS
    }
    expected_task_counts = {
        split.value: task_counts[split] for split in _STANDARD_SPLITS
    }
    if report.seed != config.seed:
        raise ValueError("split balance seed disagrees with config")
    if dict(report.core_counts) != expected_core_counts:
        raise ValueError("split balance core counts disagree with task artifact")
    if dict(report.projected_task_counts) != expected_task_counts:
        raise ValueError("split balance task counts disagree with task artifact")

    observed = Counter(
        (family, difficulty, strata, split)
        for family, difficulty, strata, split in core_records.values()
    )
    totals = Counter(
        (family, difficulty, strata)
        for family, difficulty, strata, _ in core_records.values()
    )
    expected_keys = {
        (*base, split)
        for base in totals
        for split in _STANDARD_SPLITS
    }
    reported = {}
    reported_order = []
    for cell in report.cells:
        key = (
            cell.task_family,
            cell.difficulty,
            tuple(cell.strata.items()),
            cell.split,
        )
        if key in reported:
            raise ValueError("split balance contains a duplicate cell")
        reported[key] = cell
        reported_order.append(key)
    if set(reported) != expected_keys:
        raise ValueError("split balance cells do not exactly cover task strata")

    family_order = {family: index for index, family in enumerate(_PILOT_FAMILIES)}
    split_order = {split: index for index, split in enumerate(_STANDARD_SPLITS)}
    expected_order = sorted(
        expected_keys,
        key=lambda key: (
            family_order[key[0]],
            _strata_material_key(key[0], key[1], key[2]),
            split_order[key[3]],
        ),
    )
    if reported_order != expected_order:
        raise ValueError("split balance cells are not in canonical order")

    quotas = {
        split: int(config.cores_per_family * ratio)
        for split, ratio in {
            Split.TRAIN: config.splits.train,
            Split.DEV: config.splits.dev,
            Split.TEST: config.splits.test,
        }.items()
    }
    for key, cell in reported.items():
        base = key[:3]
        total = totals[base]
        count = observed[key]
        expected_count = float(
            total * quotas[key[3]] / config.cores_per_family
        )
        if (
            cell.total != total
            or cell.observed != count
            or cell.expected != expected_count
            or cell.deviation != float(count - expected_count)
        ):
            raise ValueError("split balance cell values disagree with task artifact")

    assignments = [
        CoreSplitAssignment(
            semantic_core_id=core_id,
            task_family=family,
            difficulty=difficulty,
            strata=dict(strata),
            split=split,
            ranking_sha256=_ranking_sha256_from_material(
                seed=config.seed,
                task_family=family,
                difficulty=difficulty,
                strata=dict(strata),
                semantic_core_id=core_id,
            ),
        )
        for core_id, (family, difficulty, strata, split) in core_records.items()
    ]
    difficulty_order = {
        difficulty: index
        for index, difficulty in enumerate(
            (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        )
    }
    assignments.sort(
        key=lambda assignment: (
            family_order[assignment.task_family],
            difficulty_order[assignment.difficulty],
            _stratum_sort_key(
                assignment.task_family,
                assignment.difficulty,
                assignment.strata,
            ),
            assignment.semantic_core_id,
        )
    )
    try:
        _validate_split_assignment_result(
            SplitAssignmentResult(
                assignments=tuple(assignments),
                split_balance=report,
            )
        )
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("\n", " ")[:512]
        raise ValueError(
            f"split assignment disagrees with deterministic ranking: {detail}"
        ) from exc


def _validate_public_bundle(bundle: PilotArtifactBundle) -> None:
    _require_bundle_contracts(bundle)
    _validate_artifact_envelopes(bundle.artifacts)

    try:
        context = GenerationContext(
            config=bundle.resolved_config,
            code_revision=bundle.task_manifest.code_revision,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("bundle config or revision binding is invalid") from exc
    resolved_config = context.config.model_copy(deep=True)
    object.__setattr__(bundle, "resolved_config", resolved_config)

    typed_payloads = (
        canonical_json_bytes(resolved_config),
        canonical_json_bytes(bundle.split_balance_report),
        canonical_json_bytes(bundle.task_manifest),
        canonical_json_bytes(bundle.validation_report),
    )
    if tuple(item.content for item in bundle.artifacts[1:]) != typed_payloads:
        raise ValueError("typed bundle records disagree with canonical artifact bytes")

    try:
        tasks = CompiledPilotTasks.validated_task_set(
            bundle.tasks_jsonl,
            config_sha256=context.config_sha256,
            code_revision=bundle.task_manifest.code_revision,
            compiler_version=COMPILER_VERSION,
            generator_name=context.generator_name,
            seed=resolved_config.seed,
        )
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("\n", " ")[:512]
        raise ValueError(f"Pilot task set is invalid: {detail}") from exc
    if bundle.artifacts[0].record_count != len(tasks):
        raise ValueError("task artifact record count disagrees with canonical JSONL")

    (
        generator_name,
        compiler_version,
        code_revision,
        config_sha256,
        seed,
    ) = _task_provenance(tasks)
    if (
        compiler_version != COMPILER_VERSION
        or code_revision != bundle.task_manifest.code_revision
        or config_sha256 != bundle.artifacts[1].ref.sha256
        or config_sha256 != sha256_model(resolved_config)
        or seed != resolved_config.seed
    ):
        raise ValueError("task artifact provenance disagrees with config or manifest")

    expected_manifest = _build_manifest(
        tasks=tasks,
        compiler_versions={generator_name: compiler_version},
        code_revision=code_revision,
        config=resolved_config,
        config_ref=bundle.artifacts[1].ref,
        tasks_ref=bundle.artifacts[0].ref,
    )
    if canonical_json_bytes(bundle.task_manifest) != canonical_json_bytes(
        expected_manifest
    ):
        raise ValueError("task manifest is not the exact canonical bundle manifest")

    _validate_split_balance(tasks, resolved_config, bundle.split_balance_report)

    recomputed = validate_splits(tasks, task_manifest=bundle.task_manifest)
    if not recomputed.valid or recomputed.issues:
        codes = ", ".join(issue.code for issue in recomputed.issues[:8])
        raise ValueError(
            "bundle failed split validation" + (f": {codes}" if codes else "")
        )
    if bundle.validation_report != recomputed:
        raise ValueError("validation report does not equal recomputed split validation")
    if not bundle.validation_report.valid or bundle.validation_report.issues:
        raise ValueError("validation report must be valid with zero issues")


def _artifact(
    path: str,
    content: bytes,
    *,
    media_type: str,
    record_count: int,
) -> InMemoryPilotArtifact:
    return InMemoryPilotArtifact(
        path=path,
        content=content,
        media_type=media_type,
        record_count=record_count,
    )


def _validated_compiled_tasks(
    compiled: CompiledPilotTasks,
    context: GenerationContext,
):
    if type(compiled) is not CompiledPilotTasks:
        raise TypeError("compiled must be a CompiledPilotTasks")
    if compiled.compiler_version != COMPILER_VERSION:
        raise ValueError(
            "compiled snapshot compiler version does not equal the current compiler version"
        )
    if compiled.generator_name != context.generator_name:
        raise ValueError("compiled snapshot generator does not equal the current generator")
    try:
        return compiled.validated_tasks()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"compiled snapshot is invalid: {exc}") from exc


def _split_policy_version(tasks) -> str:
    versions = {task.metadata.split_key.split_policy_version for task in tasks}
    if len(versions) != 1:
        raise ValueError("compiled snapshot does not have one split policy version")
    version = next(iter(versions))
    if type(version) is not str or not version.strip():
        raise ValueError("compiled snapshot split policy version is invalid")
    return version


def _manifest_statistics(tasks):
    split_counts = Counter()
    family_difficulty_counts = Counter()
    semantic_core_ids = {split: set() for split in Split}
    strata_counts = Counter()
    for task in tasks:
        split = task.metadata.split
        stratum = (
            task.task_family,
            task.difficulty.value,
            str(task.metadata.resolved_profile["update_depth_bucket"]),
        )
        split_counts[split] += 1
        family_difficulty_counts[f"{stratum[0]}|{stratum[1]}"] += 1
        semantic_core_ids[split].add(task.metadata.split_key.semantic_core_id)
        strata_counts[(*stratum, split.value)] += 1

    required_keys = tuple(sorted({key[:3] for key in strata_counts}))
    required = [
        {
            "task_family": family,
            "difficulty": difficulty,
            "update_depth_bucket": bucket,
        }
        for family, difficulty, bucket in required_keys
    ]
    deviations = [
        {
            "task_family": family,
            "difficulty": difficulty,
            "update_depth_bucket": bucket,
            "split": split.value,
            "observed_count": 0,
            "rationale": "deterministic grouped split produced an exact zero-count small cell",
        }
        for family, difficulty, bucket in required_keys
        for split in _STANDARD_SPLITS
        if strata_counts[(family, difficulty, bucket, split.value)] == 0
    ]
    return (
        {split.value: split_counts[split] for split in Split},
        {
            key: family_difficulty_counts[key]
            for key in sorted(family_difficulty_counts)
        },
        {
            split.value: len(semantic_core_ids[split])
            for split in Split
        },
        required,
        deviations,
    )


def _build_manifest(
    *,
    tasks,
    compiler_versions: dict[str, str],
    code_revision: str,
    config: PilotConfig,
    config_ref: ArtifactRef,
    tasks_ref: ArtifactRef,
) -> TaskManifest:
    (
        split_counts,
        family_difficulty_counts,
        semantic_core_counts,
        required_strata,
        small_cell_deviations,
    ) = _manifest_statistics(tasks)
    task_hashes = {
        task.task_id: sha256_model(task)
        for task in sorted(tasks, key=lambda item: item.task_id)
    }
    return TaskManifest(
        data_release_id=config.release_id,
        split_policy_version=_split_policy_version(tasks),
        task_schema_version=config.schema_version,
        compiler_versions=compiler_versions,
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(config_ref,),
        split_counts=split_counts,
        family_difficulty_counts=family_difficulty_counts,
        semantic_core_counts=semantic_core_counts,
        task_file_paths_and_hashes=(tasks_ref,),
        leakage_check_summary={
            "task_hashes": task_hashes,
            "required_minimum_strata": required_strata,
            "small_cell_deviations": small_cell_deviations,
        },
        human_audit_artifacts=(),
        created_at=_CREATED_AT,
        code_revision=code_revision,
    )


def build_pilot_artifact_bundle(
    compiled: CompiledPilotTasks,
    config: PilotConfig,
) -> PilotArtifactBundle:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if type(compiled) is not CompiledPilotTasks:
        raise TypeError("compiled must be a CompiledPilotTasks")
    context = GenerationContext(config=config, code_revision=compiled.code_revision)
    if context.config_sha256 != compiled.config_sha256:
        raise ValueError("config does not match compiled snapshot config binding")
    tasks = _validated_compiled_tasks(compiled, context)

    resolved_config = context.config.model_copy(deep=True)
    resolved_config_bytes = canonical_json_bytes(resolved_config)
    if hashlib.sha256(resolved_config_bytes).hexdigest() != compiled.config_sha256:
        raise ValueError("resolved config serialization disagrees with compiled snapshot")

    tasks_artifact = _artifact(
        _TASKS_PATH,
        compiled.tasks_jsonl,
        media_type=_JSONL_MEDIA_TYPE,
        record_count=len(tasks),
    )
    config_artifact = _artifact(
        _CONFIG_PATH,
        resolved_config_bytes,
        media_type=_JSON_MEDIA_TYPE,
        record_count=1,
    )
    split_balance_report = SplitBalanceReport.model_validate(
        compiled.split_assignment.split_balance.model_dump(mode="python")
    )
    split_balance_bytes = canonical_json_bytes(split_balance_report)
    split_balance_artifact = _artifact(
        _SPLIT_BALANCE_PATH,
        split_balance_bytes,
        media_type=_JSON_MEDIA_TYPE,
        record_count=1,
    )

    task_manifest = _build_manifest(
        tasks=tasks,
        compiler_versions={compiled.generator_name: compiled.compiler_version},
        code_revision=compiled.code_revision,
        config=resolved_config,
        config_ref=config_artifact.ref,
        tasks_ref=tasks_artifact.ref,
    )
    task_manifest_bytes = canonical_json_bytes(task_manifest)
    task_manifest_artifact = _artifact(
        _TASK_MANIFEST_PATH,
        task_manifest_bytes,
        media_type=_JSON_MEDIA_TYPE,
        record_count=1,
    )

    validation_report = validate_splits(tasks, task_manifest=task_manifest)
    if not validation_report.valid or validation_report.issues:
        issue_codes = ", ".join(issue.code for issue in validation_report.issues[:8])
        raise ValueError(
            "Pilot artifact bundle failed split validation"
            + (f": {issue_codes}" if issue_codes else "")
        )
    validation_report_bytes = canonical_json_bytes(validation_report)
    validation_report_artifact = _artifact(
        _VALIDATION_REPORT_PATH,
        validation_report_bytes,
        media_type=_JSON_MEDIA_TYPE,
        record_count=1,
    )

    return PilotArtifactBundle(
        resolved_config=resolved_config,
        split_balance_report=split_balance_report,
        task_manifest=task_manifest,
        validation_report=validation_report,
        artifacts=(
            tasks_artifact,
            config_artifact,
            split_balance_artifact,
            task_manifest_artifact,
            validation_report_artifact,
        ),
    )


__all__ = [
    "InMemoryPilotArtifact",
    "PilotArtifactBundle",
    "build_pilot_artifact_bundle",
]
