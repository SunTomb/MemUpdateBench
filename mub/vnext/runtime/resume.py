from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.io.jsonl import read_models


@dataclass(frozen=True)
class ResumeDecision:
    action: str
    reason: str
    record: TaskRunRecord | None = None


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def compute_run_identity(
    *,
    task_manifest_hash: str,
    adapter_info: AdapterInfo | Mapping[str, Any],
    adapter_capabilities: AdapterCapabilities | Mapping[str, Any],
    runtime_config: Any,
    retrieval_policy: str,
    answer_mode: str,
    prompt_config: Mapping[str, Any] | None = None,
    decoding_config: Mapping[str, Any] | None = None,
    schema_version: str = "2.0.0",
    compiler_version: str = "unknown",
    profile_version: str = "unknown",
    output_dir: str | None = None,
    **extra: Any,
) -> str:
    """Hash execution identity; output locations are deliberately excluded."""
    config_payload = runtime_config.identity_payload() if hasattr(runtime_config, "identity_payload") else _plain(runtime_config)
    payload = {
        "task_manifest_hash": task_manifest_hash,
        "adapter_info": _plain(adapter_info),
        "adapter_capabilities": _plain(adapter_capabilities),
        "runtime_config": config_payload,
        "retrieval_policy": retrieval_policy,
        "answer_mode": answer_mode,
        "prompt_config": _plain(prompt_config or {}),
        "decoding_config": _plain(decoding_config or {}),
        "schema_version": schema_version,
        "compiler_version": compiler_version,
        "profile_version": profile_version,
        "extra": _plain(extra),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_metadata(row: TaskRunRecord, key: str) -> Any:
    for event in row.system_events:
        if isinstance(event, dict) and event.get("event") == "runtime_identity":
            return event.get(key)
    return None


class ResumeIndex:
    """One-pass index over task-run JSONL with strict identity and ID checks."""

    def __init__(
        self,
        records: Iterable[TaskRunRecord],
        *,
        expected_task_ids: Iterable[str],
        run_identity: str | None = None,
        expected_task_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.expected_task_ids = tuple(expected_task_ids)
        if len(self.expected_task_ids) != len(set(self.expected_task_ids)):
            raise ValueError("duplicate expected task IDs")
        self.run_identity = run_identity
        self.expected_task_hashes = dict(expected_task_hashes or {})
        self.records: dict[str, TaskRunRecord] = {}
        for row in records:
            if row.task_id in self.records:
                raise ValueError(f"duplicate task ID in existing JSONL: {row.task_id}")
            if run_identity is not None and _event_metadata(row, "run_identity") != run_identity:
                raise ValueError(f"identity mismatch for task {row.task_id}")
            self.records[row.task_id] = row

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        expected_task_ids: Iterable[str],
        run_identity: str | None = None,
        expected_task_hashes: Mapping[str, str] | None = None,
    ) -> "ResumeIndex":
        rows = read_models(path, TaskRunRecord, id_field="task_id")
        return cls(rows, expected_task_ids=expected_task_ids, run_identity=run_identity, expected_task_hashes=expected_task_hashes)

    def task_hash(self, task_id: str) -> str | None:
        row = self.records.get(task_id)
        return _event_metadata(row, "task_hash") if row else None

    def decide(
        self,
        task_id: str,
        *,
        task_hash: str | None = None,
        retry_failed: bool = False,
        retry_partial: bool = False,
        retry_not_supported: bool = False,
    ) -> ResumeDecision:
        row = self.records.get(task_id)
        if row is None:
            return ResumeDecision("execute", "missing")
        expected_hash = task_hash if task_hash is not None else self.expected_task_hashes.get(task_id)
        if expected_hash is not None and self.task_hash(task_id) != expected_hash:
            return ResumeDecision("retry", "task_hash_mismatch", row)
        if row.completion_status is CompletionStatus.COMPLETED:
            return ResumeDecision("skip", "matching_completed", row)
        if row.completion_status is CompletionStatus.FAILED:
            return ResumeDecision("retry" if retry_failed else "reject", "failed_retry_enabled" if retry_failed else "failed_requires_retry_failed", row)
        if row.completion_status is CompletionStatus.PARTIAL:
            return ResumeDecision("retry" if retry_partial else "reject", "partial_retry_enabled" if retry_partial else "partial_requires_retry_partial", row)
        if row.completion_status is CompletionStatus.NOT_SUPPORTED:
            return ResumeDecision("retry" if retry_not_supported else "reject", "not_supported_retry_enabled" if retry_not_supported else "not_supported_requires_retry_not_supported", row)
        return ResumeDecision("reject", "unknown_status", row)

    @property
    def missing_task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for task_id in self.expected_task_ids if task_id not in self.records)

    def require_complete(self) -> None:
        missing = self.missing_task_ids
        if missing:
            raise ValueError(f"missing expected task IDs: {', '.join(missing)}")


__all__ = ["ResumeDecision", "ResumeIndex", "compute_run_identity"]
