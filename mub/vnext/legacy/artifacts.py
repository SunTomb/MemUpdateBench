from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    freeze_mapping,
)
from mub.vnext.contracts.enums import Split
from mub.vnext.contracts.manifest import TaskManifest
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io.canonical import canonical_json_bytes, sha256_model
from mub.vnext.legacy.dataset import compile_legacy_episode
from mub.vnext.legacy.loaders import _parse_dataset
from mub.vnext.legacy.validation import _validate_trusted_legacy_task_semantics
from mub.vnext.profiles import hard_profile, resolve_profile
from mub.vnext.validation.issues import ValidationReport


LEGACY_SCHEMA_VERSION = "1.0.0"
LEGACY_ANALYSIS_MANIFEST_VERSION = "1.0.0"
LEGACY_CLI_COMPILER_VERSION = "vnext-phase0-cli-1.0.0"
LEGACY_CLI_CODE_REVISION = "legacy-compatibility-import"
LegacyAnalysisKind = Literal["conflict", "dose", "stale-removal", "api", "ledger"]


class LegacyAnalysisManifest(ImmutableContractModel):
    schema_version: Literal["1.0.0"] = LEGACY_SCHEMA_VERSION
    legacy_analysis_manifest_version: Literal["1.0.0"] = LEGACY_ANALYSIS_MANIFEST_VERSION
    analysis_kind: LegacyAnalysisKind
    compiler_version: Literal["vnext-phase0-cli-1.0.0"]
    compatibility_only: Literal[True]
    source_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    output_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    row_counts: FrozenNonnegativeIntMap
    warnings: tuple[str, ...]
    caveats: tuple[str, ...]
    code_revision: Literal["legacy-compatibility-import"]

    @field_validator("row_counts")
    @classmethod
    def _freeze_row_counts(cls, value: Mapping[str, int]):
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _validate_artifact_linkage(self):
        source_paths = [artifact.path for artifact in self.source_artifacts]
        output_paths = [artifact.path for artifact in self.output_artifacts]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("source artifact paths must be unique")
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("output artifact paths must be unique")
        if set(source_paths) & set(output_paths):
            raise ValueError("source and output artifact paths must be disjoint")
        if any(
            not artifact.media_type.strip() or artifact.record_count is None
            for artifact in (*self.source_artifacts, *self.output_artifacts)
        ):
            raise ValueError("legacy artifacts require media_type and record_count")
        expected_counts = {
            Path(artifact.path).name: artifact.record_count
            for artifact in self.output_artifacts
        }
        if len(expected_counts) != len(self.output_artifacts):
            raise ValueError("output artifact basenames must be unique")
        if dict(self.row_counts) != expected_counts:
            raise ValueError("row_counts must exactly match output artifacts")
        return self


    @field_validator("warnings", "caveats")
    @classmethod
    def _validate_messages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("manifest messages must be nonblank exact strings")
        if len(values) != len(set(values)):
            raise ValueError("manifest messages must be unique")
        return values


def with_legacy_index_profile(task: MemUpdateTask, index: int) -> MemUpdateTask:
    payload = task.model_dump(mode="python")
    extra = payload["metadata"]["extra"]
    if "legacy_example_index" in extra:
        raise ValueError("compiler output unexpectedly contains legacy_example_index")
    extra["legacy_example_index"] = index
    update_depth = extra.get("num_target_updates")
    if type(update_depth) is not int or update_depth <= 0:
        raise ValueError("compiled task lacks a valid num_target_updates linkage")
    noop_count = sum(
        action["operation"] == "NOOP" for action in payload["gold"]["actions"]
    )
    payload["metadata"]["resolved_profile"] = resolve_profile(
        hard_profile(task.task_family),
        {
            "update_depth": update_depth,
            "active_object_count": len(task.target_objects),
            "stale_count": max(update_depth - 1, 0),
            "context_length": len(task.events),
            "cross_slot_interleaving": 0.0,
            "noop_density": noop_count / len(task.events),
        },
    )
    return MemUpdateTask.model_validate(payload)


def build_expected_legacy_task_manifest(
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
) -> TaskManifest:
    resolved_tasks_path = _require_regular_file(tasks_path, "compiled tasks")
    return _build_expected_legacy_task_manifest_snapshot(
        tasks,
        tasks_path=resolved_tasks_path,
        tasks_bytes=resolved_tasks_path.read_bytes(),
    )


def _build_expected_legacy_task_manifest_snapshot(
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
    tasks_bytes: bytes,
) -> TaskManifest:
    manifest, source_path, source_bytes, source_signature = (
        _build_expected_legacy_task_manifest_snapshot_bound(
            tasks,
            tasks_path=tasks_path,
            tasks_bytes=tasks_bytes,
        )
    )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    return manifest


def _build_expected_legacy_task_manifest_snapshot_bound(
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
    tasks_bytes: bytes,
) -> tuple[TaskManifest, Path, bytes, tuple[int, int, int, int]]:
    if not tasks:
        raise ValueError("compiled task artifact must contain at least one task")
    resolved_tasks_path = tasks_path.resolve(strict=False)
    source_path, source_hash, split, legacy_phase = _declared_task_source(tasks)
    source_bytes, rows, source_signature = _read_legacy_source_snapshot(
        source_path,
        source_hash,
    )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    try:
        _authenticate_tasks_against_rows(
            tasks,
            rows=rows,
            source_path=source_path,
            source_hash=source_hash,
            split=split,
            legacy_phase=legacy_phase,
        )
    except Exception:
        _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
        raise
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    family_difficulty = Counter(
        f"{task.task_family}|{task.difficulty.value}" for task in tasks
    )
    split_counts = {
        item.value: sum(task.metadata.split == item for task in tasks)
        for item in Split
    }
    semantic_counts = {
        item.value: len(
            {
                task.metadata.split_key.semantic_core_id
                for task in tasks
                if task.metadata.split == item
            }
        )
        for item in Split
    }
    required_strata = [
        {
            "task_family": family,
            "difficulty": difficulty,
            "update_depth_bucket": bucket,
        }
        for family, difficulty, bucket in sorted(
            {
                (
                    task.task_family,
                    task.difficulty.value,
                    str(task.metadata.resolved_profile["update_depth_bucket"]),
                )
                for task in tasks
            }
        )
    ]
    tasks_hash = hashlib.sha256(tasks_bytes).hexdigest()
    manifest = TaskManifest(
        data_release_id=_stable_identity(
            "legacy_release",
            {
                "source_sha256": source_hash,
                "split": split.value,
                "legacy_phase": legacy_phase,
            },
        ),
        split_policy_version="vnext-phase0-legacy-v1",
        compiler_versions={"vnext_compile_legacy": LEGACY_CLI_COMPILER_VERSION},
        source_manifest_paths_and_hashes=(
            ArtifactRef(
                path=str(source_path),
                sha256=source_hash,
                media_type="application/json",
                record_count=len(tasks),
            ),
        ),
        generation_configs_and_hashes=(),
        split_counts=split_counts,
        family_difficulty_counts=dict(sorted(family_difficulty.items())),
        semantic_core_counts=semantic_counts,
        task_file_paths_and_hashes=(
            ArtifactRef(
                path=str(resolved_tasks_path),
                sha256=tasks_hash,
                media_type="application/x-ndjson",
                record_count=len(tasks),
            ),
        ),
        leakage_check_summary={
            "compatibility_only": True,
            "warnings": [
                "legacy_compatibility_only",
                "legacy_source_split_preserved",
            ],
            "row_counts": {"tasks.jsonl": len(tasks)},
            "input_hashes": {str(source_path): source_hash},
            "output_hashes": {str(resolved_tasks_path): tasks_hash},
            "task_hashes": {
                task.task_id: sha256_model(task)
                for task in sorted(tasks, key=lambda item: item.task_id)
            },
            "required_minimum_strata": required_strata,
            "small_cell_deviations": [],
        },
        human_audit_artifacts=(),
        created_at="legacy-authenticated-source",
        code_revision=LEGACY_CLI_CODE_REVISION,
    )
    return manifest, source_path, source_bytes, source_signature


def _authenticate_legacy_task_manifest_snapshot_bound(
    manifest: TaskManifest,
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
    tasks_bytes: bytes,
) -> tuple[TaskManifest, Path, bytes, tuple[int, int, int, int]]:
    expected, source_path, source_bytes, source_signature = (
        _build_expected_legacy_task_manifest_snapshot_bound(
            tasks,
            tasks_path=tasks_path,
            tasks_bytes=tasks_bytes,
        )
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise ValueError(
            "TaskManifest does not exactly match authenticated deterministic compilation"
        )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    return expected, source_path, source_bytes, source_signature


def _authenticate_legacy_task_manifest_snapshot(
    manifest: TaskManifest,
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
    tasks_bytes: bytes,
) -> TaskManifest:
    expected, source_path, source_bytes, source_signature = (
        _authenticate_legacy_task_manifest_snapshot_bound(
            manifest,
            tasks,
            tasks_path=tasks_path,
            tasks_bytes=tasks_bytes,
        )
    )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    return expected


def authenticate_legacy_task_manifest(
    manifest: TaskManifest,
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
) -> TaskManifest:
    resolved_path = _require_regular_file(tasks_path, "compiled tasks")
    return _authenticate_legacy_task_manifest_snapshot(
        manifest,
        tasks,
        tasks_path=resolved_path,
        tasks_bytes=resolved_path.read_bytes(),
    )


def _authenticate_and_validate_legacy_tasks(
    manifest: TaskManifest,
    tasks: list[MemUpdateTask],
    *,
    tasks_path: Path,
) -> tuple[TaskManifest, tuple[ValidationReport, ...]]:
    """Authenticate one task-file snapshot, then validate those exact task models."""
    resolved_path = _require_regular_file(tasks_path, "compiled tasks")
    snapshot = resolved_path.read_bytes()
    canonical_snapshot = b"".join(
        canonical_json_bytes(task) + b"\n" for task in tasks
    )
    if snapshot != canonical_snapshot:
        raise ValueError(
            "compiled task snapshot does not equal supplied canonical task bytes"
        )
    authenticated, source_path, source_bytes, source_signature = (
        _authenticate_legacy_task_manifest_snapshot_bound(
            manifest,
            tasks,
            tasks_path=resolved_path,
            tasks_bytes=snapshot,
        )
    )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    reports = tuple(
        _validate_trusted_legacy_task_semantics(task) for task in tasks
    )
    _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
    return authenticated, reports


def _authenticate_tasks_against_rows(
    tasks: list[MemUpdateTask],
    *,
    rows: list[dict[str, object]],
    source_path: Path,
    source_hash: str,
    split: Split,
    legacy_phase: str,
) -> None:
    if len(rows) != len(tasks):
        raise ValueError("legacy dataset source record count does not match tasks")
    for index, (row, supplied) in enumerate(zip(rows, tasks, strict=True)):
        expected = with_legacy_index_profile(
            compile_legacy_episode(
                row,
                source_path=source_path,
                source_sha256=source_hash,
                split=split,
                example_index=index,
                legacy_phase=legacy_phase,
            ),
            index,
        )
        if canonical_json_bytes(supplied) != canonical_json_bytes(expected):
            raise ValueError(
                f"compiled task at legacy index {index} does not match source compilation"
            )


def _declared_task_source(
    tasks: list[MemUpdateTask],
) -> tuple[Path, str, Split, str]:
    identities: set[tuple[str, str, Split, str]] = set()
    for task in tasks:
        provenance = task.metadata.legacy_provenance
        if provenance is None:
            raise ValueError("compiled legacy task lacks LegacyProvenance")
        if provenance.legacy_split_id != task.metadata.split.value:
            raise ValueError("task split and LegacyProvenance split differ")
        if task.source.raw_hash != provenance.source_artifact_hash:
            raise ValueError("task source and LegacyProvenance hashes differ")
        identities.add(
            (
                provenance.source_artifact_path,
                provenance.source_artifact_hash,
                task.metadata.split,
                provenance.legacy_phase,
            )
        )
    if len(identities) != 1:
        raise ValueError("compiled tasks must share one authenticated source/split/phase")
    source_text, declared_hash, split, legacy_phase = identities.pop()
    source_path = _require_regular_file(Path(source_text), "legacy dataset source")
    return source_path, declared_hash, split, legacy_phase


def _stat_signature(result: os.stat_result) -> tuple[int, int, int, int]:
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _source_changed(path: Path) -> RuntimeError:
    return RuntimeError(f"legacy dataset source changed during authentication: {path}")


def _read_legacy_source_snapshot(
    source_path: Path,
    declared_hash: str,
) -> tuple[bytes, list[dict[str, object]], tuple[int, int, int, int]]:
    try:
        initial_stat = source_path.stat()
        with source_path.open("rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            source_bytes = handle.read()
            descriptor_after = os.fstat(handle.fileno())
        after_read_stat = source_path.stat()
    except OSError as exc:
        raise _source_changed(source_path) from exc
    signatures = {
        _stat_signature(initial_stat),
        _stat_signature(descriptor_before),
        _stat_signature(descriptor_after),
        _stat_signature(after_read_stat),
    }
    source_signature = _stat_signature(initial_stat)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if len(signatures) != 1 or source_hash != declared_hash:
        raise _source_changed(source_path)
    try:
        rows = _parse_dataset(source_bytes, source_path)
    except Exception:
        _verify_legacy_source_snapshot(source_path, source_bytes, source_signature)
        raise
    return source_bytes, rows, source_signature


def _verify_legacy_source_snapshot(
    source_path: Path,
    source_bytes: bytes,
    source_signature: tuple[int, int, int, int],
) -> None:
    try:
        before_hash_stat = source_path.stat()
        current_hash = _sha256_file(source_path)
        final_stat = source_path.stat()
    except OSError as exc:
        raise _source_changed(source_path) from exc
    if (
        _stat_signature(before_hash_stat) != source_signature
        or _stat_signature(final_stat) != source_signature
        or current_hash != hashlib.sha256(source_bytes).hexdigest()
    ):
        raise _source_changed(source_path)


def _require_regular_file(path: Path, label: str) -> Path:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_identity(prefix: str, material: dict[str, object]) -> str:
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "LEGACY_ANALYSIS_MANIFEST_VERSION",
    "LEGACY_CLI_CODE_REVISION",
    "LEGACY_CLI_COMPILER_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "LegacyAnalysisManifest",
    "authenticate_legacy_task_manifest",
    "build_expected_legacy_task_manifest",
    "with_legacy_index_profile",
]
