from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts import (
    AdapterActionLog,
    AdapterCapabilities,
    AdapterInfo,
    AnswerDisposition,
    AnswerResult,
    MemoryEntryRecord,
    MemoryEvent,
    MemoryObjectKey,
    MemoryQuery,
    MemUpdateTask,
    Operation,
    ResetResult,
)
from mub.vnext.adapters.retrieval import apply_retrieval_policy


_NOT_SUPPORTED = {"code": "not_supported", "reason": "capability_unavailable"}
_CLOSED = {"code": "closed", "reason": "adapter_closed"}


def _configuration_hash(adapter_id: str, config: dict[str, Any]) -> str:
    payload = json.dumps({"adapter_id": adapter_id, "config": config}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(text: str) -> Any:
    value = text.strip().rstrip(".").strip()
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _parse_part(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text[0] in "\"'" and text[-1:] == text[0]:
        return text[1:-1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, str) else (None if parsed is None else str(parsed))


def _atomic_key(text: str) -> MemoryObjectKey | None:
    match = re.search(r"object\s*\((.*?)\)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    fields: dict[str, str | None] = {}
    for name, raw in re.findall(r"(object_type|namespace|entity|attribute|subkey)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,;]+)", match.group(1), re.IGNORECASE):
        fields[name.lower()] = _parse_part(raw)
    if not all(fields.get(name) for name in ("namespace", "entity", "attribute")):
        return None
    return MemoryObjectKey(
        object_type=fields.get("object_type") or "slot",
        namespace=fields["namespace"] or "default",
        entity=fields["entity"] or "",
        attribute=fields["attribute"] or "",
        subkey=fields.get("subkey"),
    )


@dataclass(frozen=True)
class ParsedEvent:
    operation: Operation | None
    object_key: MemoryObjectKey | None
    value: Any
    format_valid: bool
    error: str | None = None


def parse_event_text(event: MemoryEvent) -> ParsedEvent:
    """Parse constrained-slot syntax plus the canonical atomic object rendering."""
    text = event.raw_text.strip() or event.normalized_text.strip()
    try:
        from mub.manager.memory_manager import MemoryManager

        parsed = MemoryManager.parse_constrained_slot_operation(text)
    except Exception:
        parsed = {"operation": "INVALID", "entity": "", "attribute": "", "value": ""}
    operation = parsed.get("operation")
    if operation in {"ADD", "UPDATE"}:
        return ParsedEvent(
            Operation(operation),
            MemoryObjectKey(object_type="slot", entity=parsed["entity"], attribute=parsed["attribute"]),
            _json_value(parsed.get("value", "")),
            True,
        )
    if operation == "NOOP" or re.match(r"^(?:NOOP|No memory object changes|Keep memory unchanged)\.?$", text, re.I):
        return ParsedEvent(Operation.NOOP, None, None, True)

    key = _atomic_key(text)
    if key is not None:
        lowered = text.lower()
        if re.search(r"\b(?:forget|erase|remove|delete)\b", lowered):
            return ParsedEvent(Operation.DELETE, key, None, True)
        if re.search(r"\b(?:update|change|correct|revise)\b", lowered):
            op = Operation.UPDATE
        elif re.search(r"\b(?:add|create)\b", lowered):
            op = Operation.ADD
        else:
            op = None
        value_match = (
            re.search(r"with\s+value\s+(.+?)(?:\.|$)", text, re.I | re.S)
            or re.search(r"(?:as|to)\s+(.+?)\s+to\s+memory(?:\.|$)", text, re.I | re.S)
            or re.search(r"(?:as|to|:)\s*(.+?)(?:\.|$)", text, re.I | re.S)
        )
        return ParsedEvent(op, key, _json_value(value_match.group(1)) if value_match else None, op is not None)

    metadata_op = event.metadata.get("operation")
    metadata_key = event.metadata.get("object_key") or event.metadata.get("target_object_key")
    if metadata_op in {item.value for item in Operation} and isinstance(metadata_key, dict):
        try:
            return ParsedEvent(
                Operation(metadata_op),
                MemoryObjectKey.model_validate(metadata_key),
                event.metadata.get("value"),
                True,
            )
        except Exception:
            pass
    return ParsedEvent(None, None, None, False, "invalid_action_format")


def _copy_task(task: MemUpdateTask) -> MemUpdateTask:
    if not isinstance(task, MemUpdateTask):
        raise TypeError("task must be a MemUpdateTask")
    return MemUpdateTask.model_validate(task.model_dump(mode="python"))


def _entry_content(key: MemoryObjectKey | None, value: Any, fallback: str) -> str:
    if key is None:
        return fallback
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{key.canonical_id} = {rendered}"


class BaseBuiltinAdapter:
    """Shared lifecycle, canonical entry, and answer behavior for Pilot adapters."""

    adapter_id = "builtin"
    adapter_version = "1.0.0"
    system_name = "mub_vnext_builtin"
    system_version = "1.0.0"
    capabilities_config: dict[str, bool] = {}

    def __init__(self, *, retrieval_policy: str = "normal_topk") -> None:
        if retrieval_policy not in {"normal_topk", "latest_per_object"}:
            raise ValueError(f"unknown retrieval policy: {retrieval_policy}")
        self.retrieval_policy = retrieval_policy
        self._entries: list[MemoryEntryRecord] = []
        self._state: dict[str, MemoryEntryRecord] = {}
        self._history: dict[str, list[Any]] = {}
        self._actions: list[dict[str, Any]] = []
        self._namespace: str | None = None
        self._closed = False
        self._ready = True
        self._startup_error: dict[str, Any] | None = None

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            system_name=self.system_name,
            system_version=self.system_version,
            configuration_hash=_configuration_hash(self.adapter_id, self._info_config()),
            extractor_id=None,
            extractor_version=None,
        )

    def _info_config(self) -> dict[str, Any]:
        return {"retrieval_policy": self.retrieval_policy}

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities.model_validate(dict(self.capabilities_config))

    def capability_bitset(self) -> int:
        bits = 0
        for index, field_name in enumerate(AdapterCapabilities.model_fields):
            if field_name == "extractor_version":
                continue
            if bool(getattr(self.capabilities(), field_name)):
                bits |= 1 << index
        return bits

    @property
    def capability_bits(self) -> int:
        return self.capability_bitset()

    def reset(self, namespace: str, config: dict) -> ResetResult:
        if type(namespace) is not str or not namespace.strip():
            raise TypeError("namespace must be a nonblank string")
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        self._namespace = namespace
        self._entries.clear()
        self._state.clear()
        self._history.clear()
        self._actions.clear()
        self._closed = False
        if not self._ready:
            return ResetResult(success=False, namespace=namespace, error=self._startup_error or _NOT_SUPPORTED)
        return ResetResult(success=True, namespace=namespace)

    def _error_log(self, event: MemoryEvent, error: dict[str, Any]) -> AdapterActionLog:
        return AdapterActionLog(event_id=event.event_id, raw_action=event.raw_text, error=error)

    def _ensure_ready(self, event: MemoryEvent) -> AdapterActionLog | None:
        if self._closed:
            return self._error_log(event, _CLOSED)
        if not self._ready:
            return self._error_log(event, self._startup_error or _NOT_SUPPORTED)
        if self._namespace is None:
            return self._error_log(event, {"code": "not_reset", "reason": "reset_required"})
        return None

    def _append_entry(self, event: MemoryEvent, parsed: ParsedEvent, *, entry_id: str | None = None) -> MemoryEntryRecord:
        key = parsed.object_key
        version = len(self._history.get(key.canonical_id, [])) if key else len(self._entries)
        record = MemoryEntryRecord(
            entry_id=entry_id or f"{self.adapter_id}:{self._namespace}:{event.sequence_index}:{len(self._entries)}",
            content=_entry_content(key, parsed.value, event.raw_text),
            object_key_candidate=key,
            value_candidate=parsed.value,
            created_at=event.timestamp or f"{event.sequence_index:020d}",
            updated_at=event.timestamp or f"{event.sequence_index:020d}",
            source_event_ids=[event.event_id],
            version_index=version,
            raw_metadata={
                "adapter_order": event.sequence_index,
                "sequence_index": event.sequence_index,
                "operation": parsed.operation.value if parsed.operation else None,
            },
        )
        self._entries.append(record)
        return record

    def export_entries(self) -> list[MemoryEntryRecord]:
        return [MemoryEntryRecord.model_validate(entry.model_dump(mode="python")) for entry in self._entries]

    def export_raw_state(self) -> dict[str, Any]:
        return {
            "namespace": self._namespace,
            "entries": [entry.model_dump(mode="json") for entry in self._entries],
            "state_by_object": {key: entry.value_candidate for key, entry in self._state.items()},
            "history": {key: list(values) for key, values in self._history.items()},
            "action_trace": list(self._actions),
        }

    def retrieve(self, query: MemoryQuery, k: int):
        if not self._ready:
            from mub.vnext.contracts import RetrievalResult
            return RetrievalResult(query_id=query.query_id, error=self._startup_error or _NOT_SUPPORTED)
        if self._namespace is None:
            from mub.vnext.contracts import RetrievalResult
            return RetrievalResult(query_id=query.query_id, error={"code": "not_reset", "reason": "reset_required"})
        return apply_retrieval_policy(self.retrieval_policy, self.export_entries(), query, k)

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        if self._closed:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error=_CLOSED)
        if not self._ready:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error=self._startup_error or _NOT_SUPPORTED)
        if self._namespace is None:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error={"code": "not_reset", "reason": "reset_required"})
        if mode not in {"slot_direct", "slot_prompt", "native_answer"}:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error={"code": "not_supported", "mode": mode})
        if len(query.target_object_keys) != 1:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error={"code": "not_supported", "reason": "multi_object_answer"})
        entry = self._state.get(query.target_object_keys[0].canonical_id)
        if entry is None:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error={"code": "missing_value"})
        value = entry.value_candidate
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        return AnswerResult(query_id=query.query_id, raw_output=raw, value=value)

    def close(self) -> None:
        self._closed = True

    def export_action_trace(self) -> list[dict[str, Any]]:
        return list(self._actions)


class ReferenceAdapter(BaseBuiltinAdapter):
    adapter_id = "reference"
    system_name = "oracle_smoke_only"
    capabilities_config = {
        "supports_isolated_reset": True, "supports_event_ingest": True,
        "supports_add": True, "supports_update": True, "supports_noop": True,
        "supports_delete": True, "supports_native_answer": True,
        "exports_entries": True, "exports_raw_state": True,
        "exports_source_event_ids": True, "exports_timestamps_or_order": True,
        "exports_object_keys": True, "exports_values": True,
        "exports_retrieval_ids": True, "exports_retrieval_scores": True,
        "exports_action_trace": True,
    }

    def __init__(self, task: MemUpdateTask, *, retrieval_policy: str = "normal_topk") -> None:
        super().__init__(retrieval_policy=retrieval_policy)
        self._task = _copy_task(task)
        self._gold_actions = {action.action_id: action for action in self._task.gold.actions}

    def expected_answer(self, query: MemoryQuery) -> Any:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        if query.query_id in self._task.gold.gold_answers:
            return self._task.gold.gold_answers[query.query_id]
        canonical = self._task.gold.canonical_answers.get(query.query_id)
        if canonical is None:
            raise KeyError(f"unknown query ID: {query.query_id}")
        return canonical.value

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        if self._closed:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error=_CLOSED)
        if mode not in {"slot_direct", "slot_prompt", "native_answer"}:
            return AnswerResult(query_id=query.query_id, raw_output="", disposition=AnswerDisposition.UNAVAILABLE, error={"code": "not_supported", "mode": mode})
        if query.query_id in self._task.gold.gold_answers:
            value = self._task.gold.gold_answers[query.query_id]
            raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            return AnswerResult(query_id=query.query_id, raw_output=raw, value=value)
        canonical = self._task.gold.canonical_answers.get(query.query_id)
        if canonical is not None and canonical.disposition is AnswerDisposition.ABSTAINED:
            return AnswerResult(query_id=query.query_id, raw_output="ABSTAIN", disposition=AnswerDisposition.ABSTAINED, error={"reason": canonical.abstention_reason})
        if canonical is not None:
            value = canonical.value
            raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            return AnswerResult(query_id=query.query_id, raw_output=raw, value=value)
        return super().answer(query, mode)

    def gold_state(self) -> dict[str, Any]:
        return dict(self._task.gold.final_state)

    def gold_history(self) -> dict[str, list[Any]]:
        return {key: list(values) for key, values in self._task.gold.version_history.items()}

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        error = self._ensure_ready(event)
        if error:
            return error
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        event_actions = sorted(
            (self._gold_actions[action_id] for action_id in event.gold_action_ids),
            key=lambda action: self._task.gold.action_sequence.index(action.action_id),
        )
        affected: list[str] = []
        effective: Operation | None = None
        for action in event_actions:
            effective = action.operation
            for key in action.target_object_keys:
                object_id = key.canonical_id
                affected.append(object_id)
                self._history.setdefault(object_id, [])
                if action.operation in {Operation.ADD, Operation.UPDATE}:
                    self._history[object_id].append(action.value)
                    parsed = ParsedEvent(action.operation, key, action.value, True)
                    existing = self._state.get(object_id)
                    if existing is None:
                        entry = self._append_entry(event, parsed, entry_id=f"reference:{self._namespace}:{object_id}")
                    else:
                        entry = existing.model_copy(update={
                            "content": _entry_content(key, action.value, event.raw_text),
                            "value_candidate": action.value,
                            "updated_at": event.timestamp or f"{event.sequence_index:020d}",
                            "source_event_ids": [*existing.source_event_ids, event.event_id],
                            "version_index": len(self._history[object_id]) - 1,
                            "raw_metadata": {**existing.raw_metadata, "adapter_order": event.sequence_index, "sequence_index": event.sequence_index},
                        })
                        self._entries[self._entries.index(existing)] = entry
                    self._state[object_id] = entry
                elif action.operation == Operation.DELETE:
                    self._state.pop(object_id, None)
                    self._entries[:] = [entry for entry in self._entries if entry.object_key_candidate is None or entry.object_key_candidate.canonical_id != object_id]
        self._actions.append({"event_id": event.event_id, "operation": effective.value if effective else None, "target_object_keys": affected, "source_event_ids": [event.event_id]})
        return AdapterActionLog(event_id=event.event_id, requested_operation=effective, effective_operation=effective, affected_entry_ids=affected, raw_action=event.raw_text)


__all__ = ["BaseBuiltinAdapter", "ParsedEvent", "ReferenceAdapter", "parse_event_text"]
