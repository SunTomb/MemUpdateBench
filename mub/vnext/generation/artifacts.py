from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Final

from mub.vnext.contracts import ArtifactRef, Split, TaskManifest
from mub.vnext.generation.build import CompiledPilotTasks
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.splits import SplitBalanceReport
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.validation import ValidationReport, validate_splits
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
        if type(self.artifacts) is not tuple:
            raise TypeError("artifacts must be an immutable tuple")
        if any(not isinstance(item, InMemoryPilotArtifact) for item in self.artifacts):
            raise TypeError("artifacts must contain InMemoryPilotArtifact records")
        paths = tuple(item.path for item in self.artifacts)
        expected_paths = (
            _TASKS_PATH,
            _CONFIG_PATH,
            _SPLIT_BALANCE_PATH,
            _TASK_MANIFEST_PATH,
            _VALIDATION_REPORT_PATH,
        )
        if paths != expected_paths:
            raise ValueError("bundle artifacts are not in canonical path order")
        expected_payloads = (
            canonical_json_bytes(self.resolved_config),
            canonical_json_bytes(self.split_balance_report),
            canonical_json_bytes(self.task_manifest),
            canonical_json_bytes(self.validation_report),
        )
        if tuple(item.content for item in self.artifacts[1:]) != expected_payloads:
            raise ValueError("typed bundle records disagree with canonical artifact bytes")

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
    compiled: CompiledPilotTasks,
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
        compiler_versions={compiled.generator_name: compiled.compiler_version},
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
        code_revision=compiled.code_revision,
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
        compiled=compiled,
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
