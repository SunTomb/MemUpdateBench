from __future__ import annotations

from mub.vnext.contracts import AdapterActionLog, MemoryEvent, Operation
from mub.vnext.adapters.reference import BaseBuiltinAdapter, _action_payload, parse_event_text


class RawAppendAdapter(BaseBuiltinAdapter):
    adapter_id = "raw_add"
    system_name = "raw_append"
    capabilities_config = {
        "supports_isolated_reset": True, "supports_event_ingest": True,
        "supports_add": True, "supports_update": True, "supports_noop": True,
        "supports_delete": False, "exports_entries": True, "exports_raw_state": True,
        "exports_source_event_ids": True, "exports_timestamps_or_order": True,
        "exports_object_keys": True, "exports_values": True,
        "exports_retrieval_ids": True, "exports_retrieval_scores": True,
        "exports_action_trace": True,
    }

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        error = self._ensure_ready(event)
        if error:
            return error
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        parsed = parse_event_text(event)
        if not parsed.format_valid:
            return AdapterActionLog(
                event_id=event.event_id,
                raw_action=_action_payload(event, parsed),
                error={"code": "invalid_action_format", "reason": parsed.error},
            )
        if parsed.operation is Operation.DELETE:
            return AdapterActionLog(
                event_id=event.event_id,
                requested_operation=Operation.DELETE,
                raw_action=_action_payload(event, parsed),
                error={"code": "not_supported", "reason": "capability_unavailable"},
            )
        if parsed.operation is Operation.NOOP:
            self._actions.append({"event_id": event.event_id, "operation": "NOOP", "target_object_keys": []})
            return AdapterActionLog(
                event_id=event.event_id,
                requested_operation=Operation.NOOP,
                effective_operation=Operation.NOOP,
                raw_action=_action_payload(event, parsed),
            )
        entry = self._append_entry(event, parsed)
        if parsed.operation in {Operation.ADD, Operation.UPDATE} and parsed.object_key is not None:
            self._history.setdefault(parsed.object_key.canonical_id, []).append(parsed.value)
            self._state[parsed.object_key.canonical_id] = entry
        self._actions.append({"event_id": event.event_id, "operation": parsed.operation.value, "entry_id": entry.entry_id})
        return AdapterActionLog(
            event_id=event.event_id,
            requested_operation=parsed.operation,
            effective_operation=parsed.operation,
            affected_entry_ids=[entry.entry_id],
            raw_action=_action_payload(event, parsed),
        )


__all__ = ["RawAppendAdapter"]
