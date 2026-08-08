"""Authoritative current-root validation receipt for Core audit readiness."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, field_validator, model_validator

from mub.vnext.contracts.common import ArtifactRef, ImmutableContractModel
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.generation.core_artifacts import _PATHS, _validate_core_artifact_tree
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation.core_release import (
    _assert_tracked_core_sources_clean,
    _trusted_code_revision,
    validate_core_release,
)


CORE_CANDIDATE_RECEIPT_VERSION = "core-candidate-validation-v1"
_MAX_ARTIFACT_BYTES = {
    "tasks.jsonl": 1_073_741_824,
    "semantic_cores.jsonl": 268_435_456,
    "generation_config.json": 67_108_864,
    "split_balance.json": 67_108_864,
    "task_manifest.json": 268_435_456,
    "core-hard-v1.json": 67_108_864,
    "validation_report.json": 67_108_864,
}
_VERIFIED_RECEIPT_SNAPSHOTS: dict[str, tuple[str, str, str]] = {}


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
    )


@dataclass(frozen=True)
class _CoreCandidateTreeSnapshot:
    candidate_dir: Path
    directory_identity: tuple[int, int, int, int, int]
    artifacts: tuple[ArtifactRef, ...]
    artifact_identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...]
    artifact_bytes: tuple[tuple[str, bytes], ...]
    root_digest: str

    def content(self, name: str) -> bytes:
        return dict(self.artifact_bytes)[name]


def core_candidate_root_digest(artifacts: tuple[ArtifactRef, ...]) -> str:
    payload = [artifact.model_dump(mode="json") for artifact in artifacts]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guarded_candidate_tree_snapshot(root: Path) -> _CoreCandidateTreeSnapshot:
    """Reject unsafe boundaries before bounded reads, then freeze all seven files."""
    root = Path(root)
    root_metadata = root.stat(follow_symlinks=False)
    if (
        root.is_symlink()
        or getattr(root_metadata, "st_file_attributes", 0) & 0x400
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise ValueError("Core candidate root must be a real directory")
    paths = _validate_core_artifact_tree(root)
    if tuple(path.name for path in paths) != tuple(_PATHS):
        raise ValueError("Core candidate artifact order is not canonical")
    directory_identity = _file_identity(root_metadata)
    artifacts = []
    identities = []
    contents = []
    for path in paths:
        before = path.stat(follow_symlinks=False)
        limit = _MAX_ARTIFACT_BYTES[path.name]
        if before.st_size > limit:
            raise ValueError(f"{path.name} exceeds the bounded artifact size limit")
        identity = _file_identity(before)
        with path.open("rb", buffering=0) as handle:
            content = handle.read(before.st_size + 1)
        after = path.stat(follow_symlinks=False)
        if len(content) != before.st_size or _file_identity(after) != identity:
            raise ValueError(f"{path.name} changed during bounded snapshot read")
        artifacts.append(
            ArtifactRef(
                path=path.name,
                sha256=hashlib.sha256(content).hexdigest(),
                media_type=(
                    "application/x-ndjson"
                    if path.suffix == ".jsonl"
                    else "application/json"
                ),
                record_count=None,
            )
        )
        identities.append((path.name, identity))
        contents.append((path.name, content))
    after_paths = _validate_core_artifact_tree(root)
    after_root = root.stat(follow_symlinks=False)
    if (
        tuple(path.name for path in after_paths) != tuple(_PATHS)
        or _file_identity(after_root) != directory_identity
        or any(
            _file_identity(path.stat(follow_symlinks=False)) != identity
            for path, (_, identity) in zip(after_paths, identities)
        )
    ):
        raise ValueError("Core candidate tree changed during complete snapshot")
    artifact_tuple = tuple(artifacts)
    return _CoreCandidateTreeSnapshot(
        candidate_dir=root.resolve(strict=True),
        directory_identity=directory_identity,
        artifacts=artifact_tuple,
        artifact_identities=tuple(identities),
        artifact_bytes=tuple(contents),
        root_digest=core_candidate_root_digest(artifact_tuple),
    )


def _stable_tracked_revision() -> str:
    before = _trusted_code_revision()
    _assert_tracked_core_sources_clean()
    after = _trusted_code_revision()
    if before != after:
        raise ValueError("trusted audit-tooling revision changed during cleanliness check")
    return before


def _authoritatively_validate_candidate_tree(
    root: Path, *, expected_full: bool
) -> _CoreCandidateTreeSnapshot:
    before = _guarded_candidate_tree_snapshot(root)
    revision_before = _stable_tracked_revision()
    report = validate_core_release(root, expected_full=expected_full)
    if not report.valid:
        raise ValueError("Core candidate validation did not return a valid report")
    after = _guarded_candidate_tree_snapshot(root)
    revision_after = _stable_tracked_revision()
    if before != after or revision_before != revision_after:
        raise ValueError("Core candidate tree or trusted revision changed during validation")
    return after


class CoreCandidateValidationReceipt(ImmutableContractModel):
    model_config = ConfigDict(strict=True)

    receipt_version: Literal[CORE_CANDIDATE_RECEIPT_VERSION] = (
        CORE_CANDIDATE_RECEIPT_VERSION
    )
    candidate_dir: str
    expected_full: bool
    source_task_manifest_hash: str
    tasks_artifact_hash: str
    task_count: int
    code_revision: str
    trusted_audit_tooling_revision: str
    candidate_artifacts: tuple[ArtifactRef, ...]
    candidate_root_digest: str
    receipt_hash: str

    @field_validator(
        "candidate_dir", "code_revision", "trusted_audit_tooling_revision"
    )
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator(
        "source_task_manifest_hash",
        "tasks_artifact_hash",
        "candidate_root_digest",
        "receipt_hash",
    )
    @classmethod
    def _sha256(cls, value: str, info) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{info.field_name} must be lowercase sha256")
        return value

    @field_validator("candidate_artifacts", mode="before")
    @classmethod
    def _artifact_tuple(cls, value):
        if type(value) not in (list, tuple):
            raise ValueError("candidate_artifacts must be a list or tuple")
        return tuple(
            item if type(item) is ArtifactRef else ArtifactRef.model_validate(item)
            for item in value
        )

    @model_validator(mode="after")
    def _bindings(self) -> CoreCandidateValidationReceipt:
        if tuple(item.path for item in self.candidate_artifacts) != tuple(_PATHS):
            raise ValueError("candidate receipt must bind the exact seven artifacts")
        if self.candidate_root_digest != core_candidate_root_digest(
            self.candidate_artifacts
        ):
            raise ValueError("candidate root digest mismatch")
        hashes = {item.path: item.sha256 for item in self.candidate_artifacts}
        if (
            hashes["task_manifest.json"] != self.source_task_manifest_hash
            or hashes["tasks.jsonl"] != self.tasks_artifact_hash
        ):
            raise ValueError("candidate receipt primary artifact hashes disagree")
        if self.code_revision != self.trusted_audit_tooling_revision:
            raise ValueError(
                "candidate generation revision must equal trusted audit-tooling revision"
            )
        if self.receipt_hash != core_candidate_receipt_hash(self):
            raise ValueError("candidate validation receipt hash mismatch")
        return self


def core_candidate_receipt_hash(receipt) -> str:
    if isinstance(receipt, CoreCandidateValidationReceipt):
        payload = receipt.model_dump(mode="json", exclude={"receipt_hash"})
    else:
        payload = dict(receipt)
        payload.pop("receipt_hash", None)
        if "candidate_artifacts" in payload:
            payload["candidate_artifacts"] = [
                item.model_dump(mode="json")
                if isinstance(item, ArtifactRef)
                else item
                for item in payload["candidate_artifacts"]
            ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_core_candidate_validation_receipt(
    *,
    candidate_dir: Path,
    manifest_bytes: bytes,
    tasks_bytes: bytes,
    manifest: TaskManifestV3,
    expected_full: bool,
    candidate_snapshot: _CoreCandidateTreeSnapshot | None = None,
) -> CoreCandidateValidationReceipt:
    snapshot = candidate_snapshot or _guarded_candidate_tree_snapshot(candidate_dir)
    if (
        snapshot.content("task_manifest.json") != manifest_bytes
        or snapshot.content("tasks.jsonl") != tasks_bytes
    ):
        raise ValueError("receipt inputs do not match the complete candidate snapshot")
    tooling_revision = _stable_tracked_revision()
    if manifest.code_revision != tooling_revision:
        raise ValueError(
            "candidate generation revision differs from trusted audit-tooling revision"
        )
    payload = {
        "receipt_version": CORE_CANDIDATE_RECEIPT_VERSION,
        "candidate_dir": str(snapshot.candidate_dir),
        "expected_full": expected_full,
        "source_task_manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
        "tasks_artifact_hash": hashlib.sha256(tasks_bytes).hexdigest(),
        "task_count": sum(manifest.split_counts.values()),
        "code_revision": manifest.code_revision,
        "trusted_audit_tooling_revision": tooling_revision,
        "candidate_artifacts": snapshot.artifacts,
        "candidate_root_digest": snapshot.root_digest,
    }
    return CoreCandidateValidationReceipt(
        **payload,
        receipt_hash=core_candidate_receipt_hash(payload),
    )


def _snapshot_matches_receipt(
    receipt: CoreCandidateValidationReceipt,
    snapshot: _CoreCandidateTreeSnapshot,
    tooling_revision: str,
) -> bool:
    try:
        manifest_bytes = snapshot.content("task_manifest.json")
        manifest = TaskManifestV3.model_validate_json(manifest_bytes)
        return bool(
            str(snapshot.candidate_dir) == receipt.candidate_dir
            and snapshot.artifacts == receipt.candidate_artifacts
            and snapshot.root_digest == receipt.candidate_root_digest
            and canonical_json_bytes(manifest) == manifest_bytes
            and sum(manifest.split_counts.values()) == receipt.task_count
            and manifest.code_revision == receipt.code_revision
            and tooling_revision == receipt.trusted_audit_tooling_revision
            and receipt.code_revision == receipt.trusted_audit_tooling_revision
            and receipt.receipt_hash == core_candidate_receipt_hash(receipt)
        )
    except Exception:
        return False


def _register_validated_core_candidate_receipt(
    receipt: CoreCandidateValidationReceipt,
    *,
    validated_snapshot: _CoreCandidateTreeSnapshot,
) -> None:
    """Register only core_stage's stable, authoritatively validated snapshot."""
    tooling_revision = _stable_tracked_revision()
    if not _snapshot_matches_receipt(receipt, validated_snapshot, tooling_revision):
        raise ValueError("validated candidate tree does not match its receipt")
    _VERIFIED_RECEIPT_SNAPSHOTS[receipt.receipt_hash] = (
        receipt.candidate_dir,
        receipt.candidate_root_digest,
        tooling_revision,
    )


def verify_core_candidate_validation_receipt(
    receipt: CoreCandidateValidationReceipt,
    *,
    trusted_candidate_root: Path,
) -> bool:
    """Recheck the caller-trusted complete tree and tracked-clean state."""
    if type(receipt) is not CoreCandidateValidationReceipt:
        return False
    try:
        root = Path(trusted_candidate_root).resolve(strict=True)
        if str(root) != receipt.candidate_dir:
            return False
        before = _guarded_candidate_tree_snapshot(root)
        revision_before = _stable_tracked_revision()
        if not _snapshot_matches_receipt(receipt, before, revision_before):
            return False
        expected_cache = (
            receipt.candidate_dir,
            receipt.candidate_root_digest,
            revision_before,
        )
        if _VERIFIED_RECEIPT_SNAPSHOTS.get(receipt.receipt_hash) != expected_cache:
            validated = _authoritatively_validate_candidate_tree(
                root, expected_full=receipt.expected_full
            )
            if validated != before:
                return False
        after = _guarded_candidate_tree_snapshot(root)
        revision_after = _stable_tracked_revision()
        if (
            before != after
            or revision_before != revision_after
            or not _snapshot_matches_receipt(receipt, after, revision_after)
        ):
            return False
        _VERIFIED_RECEIPT_SNAPSHOTS[receipt.receipt_hash] = expected_cache
        return True
    except Exception:
        return False


__all__ = [
    "CORE_CANDIDATE_RECEIPT_VERSION",
    "CoreCandidateValidationReceipt",
    "build_core_candidate_validation_receipt",
    "core_candidate_receipt_hash",
    "core_candidate_root_digest",
    "verify_core_candidate_validation_receipt",
]
