from __future__ import annotations

import hashlib
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from mub.vnext.contracts import ArtifactRef, Split
from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.generation.artifacts import InMemoryPilotArtifact
from mub.vnext.generation.core_build import (
    CompiledCoreSnapshot,
    _generated_cores,
    _validate_snapshot,
)
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core_hard_suite import CoreHardSuiteManifest, build_core_hard_suite
from mub.vnext.io import canonical_json_bytes, sha256_model

_JSON = "application/json"
_JSONL = "application/x-ndjson"
_PATHS: Final = (
    "tasks.jsonl",
    "semantic_cores.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "core-hard-v1.json",
    "validation_report.json",
)


def _validate_core_artifact_tree(root: Path) -> tuple[Path, ...]:
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != set(_PATHS) or len(entries) != len(
        _PATHS
    ):
        raise ValueError("Core candidate must contain exactly seven artifacts")
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if (
            entry.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & 0x400
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise ValueError("Core candidate artifacts must be regular files")
    return tuple(root / path for path in _PATHS)


_VALIDATION_CHECKS: Final = (
    "canonical_artifact_bytes",
    "exact_four_surfaces",
    "family_and_split_quotas",
    "group_leakage_zero",
    "semantic_equivalence",
    "v3_replay",
    "normative_evidence_evaluation",
    "hard_suite_authentication",
    "trusted_source_config_and_revision",
)


class CoreSplitBalance(ImmutableContractModel):
    family_core_counts: dict[str, int]
    split_core_counts: dict[str, int]
    split_task_counts: dict[str, int]
    total_semantic_cores: int
    total_tasks: int


class CoreValidationReport(ImmutableContractModel):
    valid: bool
    semantic_core_count: int
    task_count: int
    split_core_counts: dict[str, int]
    split_task_counts: dict[str, int]
    family_core_counts: dict[str, int]
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoreArtifactBundle:
    resolved_config: CoreConfig
    snapshot: CompiledCoreSnapshot
    task_manifest: TaskManifestV3
    hard_suite: CoreHardSuiteManifest
    validation_report: CoreValidationReport
    artifacts: tuple[InMemoryPilotArtifact, ...]

    def artifact(self, path: str) -> InMemoryPilotArtifact:
        return next(item for item in self.artifacts if item.path == path)


def _jsonl(models, *, key) -> bytes:
    return b"".join(
        canonical_json_bytes(model) + b"\n" for model in sorted(models, key=key)
    )


def _artifact(path: str, content: bytes, media_type: str, count: int):
    return InMemoryPilotArtifact(
        path=path,
        content=content,
        media_type=media_type,
        record_count=count,
    )


def _manifest(
    snapshot: CompiledCoreSnapshot,
    config: CoreConfig,
    *,
    task_ref: ArtifactRef,
    core_ref: ArtifactRef,
    config_ref: ArtifactRef,
) -> TaskManifestV3:
    tasks = snapshot.tasks
    split_counts = Counter(task.metadata.split.value for task in tasks)
    family_difficulty = Counter(
        f"{task.task_family}|{task.difficulty.value}" for task in tasks
    )
    semantic_core_counts = Counter(
        assignment.split.value for assignment in snapshot.assignments
    )
    revisions = {task.source.generator.code_revision for task in tasks}
    generator_versions = {
        (
            task.source.generator.generator_name,
            task.source.generator.compiler_version,
        )
        for task in tasks
    }
    split_versions = {task.metadata.split_key.split_policy_version for task in tasks}
    if (
        len(revisions) != 1
        or len(generator_versions) != 1
        or len(split_versions) != 1
    ):
        raise ValueError("Core snapshot must have one code, generator/compiler, and split version")
    generator_name, compiler_version = next(iter(generator_versions))
    return TaskManifestV3(
        data_release_id=config.release_id,
        split_policy_version=next(iter(split_versions)),
        compiler_versions={generator_name: compiler_version},
        source_manifest_paths_and_hashes=(core_ref,),
        generation_configs_and_hashes=(config_ref,),
        split_counts={split.value: split_counts[split.value] for split in (Split.TRAIN, Split.DEV, Split.TEST)},
        family_difficulty_counts=dict(sorted(family_difficulty.items())),
        semantic_core_counts={split.value: semantic_core_counts[split.value] for split in (Split.TRAIN, Split.DEV, Split.TEST)},
        task_file_paths_and_hashes=(task_ref,),
        leakage_check_summary={
            "group_first": True,
            "surface_variants_per_core": 4,
            "cross_split_overlap_count": 0,
        },
        human_audit_artifacts=(),
        created_at="deterministic-core-candidate",
        code_revision=next(iter(revisions)),
        task_record_hashes={task.task_id: sha256_model(task) for task in sorted(tasks, key=lambda item: item.task_id)},
    )


def build_core_artifact_bundle(
    snapshot: CompiledCoreSnapshot,
    config: CoreConfig,
) -> CoreArtifactBundle:
    if type(snapshot) is not CompiledCoreSnapshot:
        raise TypeError("snapshot must be a CompiledCoreSnapshot")
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    if {core.core_id for core in snapshot.semantic_cores} != {
        assignment.semantic_core_id for assignment in snapshot.assignments
    }:
        raise ValueError("snapshot semantic cores and assignments differ")
    canonical_by_id = {core.core_id: core for core in _generated_cores(config)}
    if any(
        canonical_by_id.get(core.core_id) != core for core in snapshot.semantic_cores
    ):
        raise ValueError("snapshot semantic core payload is not canonical")
    _validate_snapshot(snapshot, config, tuple(canonical_by_id.values()))

    task_bytes = _jsonl(snapshot.tasks, key=lambda task: task.task_id)
    core_bytes = _jsonl(snapshot.semantic_cores, key=lambda core: core.core_id)
    config_bytes = canonical_json_bytes(config)
    task_artifact = _artifact(_PATHS[0], task_bytes, _JSONL, len(snapshot.tasks))
    core_artifact = _artifact(_PATHS[1], core_bytes, _JSONL, len(snapshot.semantic_cores))
    config_artifact = _artifact(_PATHS[2], config_bytes, _JSON, 1)

    split_balance = CoreSplitBalance(
        family_core_counts=dict(snapshot.family_core_counts),
        split_core_counts=dict(snapshot.core_counts),
        split_task_counts=dict(snapshot.task_counts),
        total_semantic_cores=len(snapshot.semantic_cores),
        total_tasks=len(snapshot.tasks),
    )
    split_artifact = _artifact(_PATHS[3], canonical_json_bytes(split_balance), _JSON, 1)
    task_manifest = _manifest(
        snapshot,
        config,
        task_ref=task_artifact.ref,
        core_ref=core_artifact.ref,
        config_ref=config_artifact.ref,
    )
    manifest_bytes = canonical_json_bytes(task_manifest)
    manifest_artifact = _artifact(_PATHS[4], manifest_bytes, _JSON, 1)

    per_family = 20 if len(snapshot.semantic_cores) == config.total_semantic_cores else min(
        2, min(dict(snapshot.family_core_counts).values())
    )
    hard_suite = build_core_hard_suite(
        snapshot,
        source_task_manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        per_family=per_family,
    )
    hard_artifact = _artifact(_PATHS[5], canonical_json_bytes(hard_suite), _JSON, 1)
    validation_report = CoreValidationReport(
        valid=True,
        semantic_core_count=len(snapshot.semantic_cores),
        task_count=len(snapshot.tasks),
        split_core_counts=dict(snapshot.core_counts),
        split_task_counts=dict(snapshot.task_counts),
        family_core_counts=dict(snapshot.family_core_counts),
        checks=_VALIDATION_CHECKS,
    )
    report_artifact = _artifact(
        _PATHS[6], canonical_json_bytes(validation_report), _JSON, 1
    )
    artifacts = (
        task_artifact,
        core_artifact,
        config_artifact,
        split_artifact,
        manifest_artifact,
        hard_artifact,
        report_artifact,
    )
    if tuple(item.path for item in artifacts) != _PATHS:
        raise AssertionError("Core artifact paths are not canonical")
    return CoreArtifactBundle(
        resolved_config=config.model_copy(deep=True),
        snapshot=snapshot,
        task_manifest=task_manifest,
        hard_suite=hard_suite,
        validation_report=validation_report,
        artifacts=artifacts,
    )


__all__ = [
    "CoreArtifactBundle",
    "CoreSplitBalance",
    "CoreValidationReport",
    "build_core_artifact_bundle",
]
