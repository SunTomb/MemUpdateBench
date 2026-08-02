from __future__ import annotations

from mub.vnext.contracts import AdapterActionLog, MemoryEntryRecord, MemoryEvent, Operation
from mub.vnext.adapters.reference import BaseBuiltinAdapter, ParsedEvent, _action_payload, parse_event_text, _entry_content


class ExactCrudAdapter(BaseBuiltinAdapter):
    adapter_id = "exact_crud"
    system_name = "exact_object_crud"
    capabilities_config = {
        "supports_isolated_reset": True, "supports_event_ingest": True,
        "supports_add": True, "supports_update": True, "supports_noop": True,
        "supports_delete": True, "exports_entries": True, "exports_raw_state": True,
        "exports_source_event_ids": True, "exports_timestamps_or_order": True,
        "exports_object_keys": True, "exports_values": True,
        "exports_retrieval_ids": True, "exports_retrieval_scores": True,
        "exports_action_trace": True,
    }

    def _write_current(self, event: MemoryEvent, parsed: ParsedEvent) -> MemoryEntryRecord:
        assert parsed.object_key is not None
        object_id = parsed.object_key.canonical_id
        old = self._state.get(object_id)
        source_ids = [*(old.source_event_ids if old else []), event.event_id]
        version = len(self._history.get(object_id, [])) - 1
        entry = MemoryEntryRecord(
            entry_id=f"{self.adapter_id}:{self._namespace}:{object_id}",
            content=_entry_content(parsed.object_key, parsed.value, event.raw_text),
            object_key_candidate=parsed.object_key,
            value_candidate=parsed.value,
            created_at=old.created_at if old else event.timestamp or f"{event.sequence_index:020d}",
            updated_at=event.timestamp or f"{event.sequence_index:020d}",
            source_event_ids=source_ids,
            version_index=version,
            raw_metadata={
                "adapter_order": event.sequence_index,
                "sequence_index": event.sequence_index,
                "operation": parsed.operation.value if parsed.operation else None,
            },
        )
        if old is None:
            self._entries.append(entry)
        else:
            self._entries[self._entries.index(old)] = entry
        self._state[object_id] = entry
        return entry

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        error = self._ensure_ready(event)
        if error:
            return error
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        parsed = parse_event_text(event)
        if not parsed.format_valid:
            return AdapterActionLog(event_id=event.event_id, raw_action=_action_payload(event, parsed), error={"code": "invalid_action_format", "reason": parsed.error})
        if parsed.operation is Operation.NOOP:
            self._actions.append({"event_id": event.event_id, "operation": "NOOP", "target_object_keys": []})
            return AdapterActionLog(event_id=event.event_id, requested_operation=Operation.NOOP, effective_operation=Operation.NOOP, raw_action=_action_payload(event, parsed))
        if parsed.object_key is None:
            return AdapterActionLog(event_id=event.event_id, raw_action=_action_payload(event, parsed), error={"code": "invalid_action_format", "reason": "object_key_required"})
        object_id = parsed.object_key.canonical_id
        if parsed.operation in {Operation.ADD, Operation.UPDATE}:
            self._history.setdefault(object_id, []).append(parsed.value)
        if parsed.operation is Operation.DELETE:
            old = self._state.pop(object_id, None)
            if old is not None:
                self._entries.remove(old)
            affected = [old.entry_id] if old else []
        else:
            entry = self._write_current(event, parsed)
            affected = [entry.entry_id]
        self._actions.append({"event_id": event.event_id, "operation": parsed.operation.value, "target_object_keys": [object_id], "entry_ids": affected})
        return AdapterActionLog(event_id=event.event_id, requested_operation=parsed.operation, effective_operation=parsed.operation, affected_entry_ids=affected, raw_action=_action_payload(event, parsed))


__all__ = ["ExactCrudAdapter"]
