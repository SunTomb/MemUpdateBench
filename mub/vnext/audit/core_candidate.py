"""Authoritative current-root validation receipt for Core audit readiness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation.core_release import validate_core_release


CORE_CANDIDATE_RECEIPT_VERSION = "core-candidate-validation-v1"


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
    receipt_hash: str

    @field_validator("candidate_dir", "code_revision")
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator(
        "source_task_manifest_hash", "tasks_artifact_hash", "receipt_hash"
    )
    @classmethod
    def _sha256(cls, value: str, info) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{info.field_name} must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def _hash_binding(self) -> CoreCandidateValidationReceipt:
        if self.receipt_hash != core_candidate_receipt_hash(self):
            raise ValueError("candidate validation receipt hash mismatch")
        return self


def core_candidate_receipt_hash(receipt) -> str:
    if isinstance(receipt, CoreCandidateValidationReceipt):
        payload = receipt.model_dump(mode="json", exclude={"receipt_hash"})
    else:
        payload = dict(receipt)
        payload.pop("receipt_hash", None)
    encoded = __import__("json").dumps(
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
) -> CoreCandidateValidationReceipt:
    payload = {
        "receipt_version": CORE_CANDIDATE_RECEIPT_VERSION,
        "candidate_dir": str(Path(candidate_dir).resolve(strict=True)),
        "expected_full": expected_full,
        "source_task_manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
        "tasks_artifact_hash": hashlib.sha256(tasks_bytes).hexdigest(),
        "task_count": sum(manifest.split_counts.values()),
        "code_revision": manifest.code_revision,
    }
    return CoreCandidateValidationReceipt(
        **payload,
        receipt_hash=core_candidate_receipt_hash(payload),
    )


def verify_core_candidate_validation_receipt(
    receipt: CoreCandidateValidationReceipt,
) -> bool:
    """Revalidate the candidate against the current root and exact receipt bytes."""
    if type(receipt) is not CoreCandidateValidationReceipt:
        return False
    try:
        root = Path(receipt.candidate_dir)
        report = validate_core_release(root, expected_full=receipt.expected_full)
        if not report.valid:
            return False
        manifest_bytes = (root / "task_manifest.json").read_bytes()
        tasks_bytes = (root / "tasks.jsonl").read_bytes()
        manifest = TaskManifestV3.model_validate_json(manifest_bytes)
        return bool(
            canonical_json_bytes(manifest) == manifest_bytes
            and hashlib.sha256(manifest_bytes).hexdigest()
            == receipt.source_task_manifest_hash
            and hashlib.sha256(tasks_bytes).hexdigest()
            == receipt.tasks_artifact_hash
            and sum(manifest.split_counts.values()) == receipt.task_count
            and manifest.code_revision == receipt.code_revision
            and receipt.receipt_hash == core_candidate_receipt_hash(receipt)
        )
    except Exception:
        return False


__all__ = [
    "CORE_CANDIDATE_RECEIPT_VERSION",
    "CoreCandidateValidationReceipt",
    "build_core_candidate_validation_receipt",
    "core_candidate_receipt_hash",
    "verify_core_candidate_validation_receipt",
]
