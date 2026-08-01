from __future__ import annotations

import json
from typing import Any, ClassVar

from mub.vnext.adapters.exact_crud import ExactCrudAdapter
from mub.vnext.adapters.reference import ParsedEvent, parse_event_text
from mub.vnext.contracts import (
    AdapterActionLog,
    AnswerDisposition,
    AnswerResult,
    MemoryEntryRecord,
    MemoryEvent,
    MemoryObjectKey,
    MemoryQuery,
    MemUpdateTask,
    Operation,
    RetrievalResult,
)
from mub.vnext.failure import canonicalize_failure_flags, primary_failure


_CONTROL_SYSTEM_NAME = "mub_vnext_corrupted_control"


def _failure_payload(flags: tuple[str, ...] | list[str] | set[str]) -> dict[str, Any]:
    canonical = canonicalize_failure_flags(flags)
    return {
        "failure_flags": list(canonical),
        "primary_failure": primary_failure(canonical),
    }


def _json_output(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _copy_task(task: MemUpdateTask | None) -> MemUpdateTask | None:
    if task is None:
        return None
    if not isinstance(task, MemUpdateTask):
        raise TypeError("task must be a MemUpdateTask")
    return MemUpdateTask.model_validate(task.model_dump(mode="python"))


class CorruptedControlAdapter(ExactCrudAdapter):
    """Deterministic in-memory controls used only to sanity-check the scorer.

    The controls deliberately alter one observable layer.  Failure labels are
    emitted as canonical observation metadata in action traces, retrieval
    results, and answer results; the shared failure module remains the only
    failure taxonomy.
    """

    control_id: ClassVar[str] = "control/base"
    expected_failure_flags: ClassVar[tuple[str, ...]] = ()
    smoke_control: ClassVar[bool] = True
    leaderboard_eligible: ClassVar[bool] = False
    adapter_version = "1.0.0"
    system_name = _CONTROL_SYSTEM_NAME
    capabilities_config = {
        "supports_isolated_reset": True,
        "supports_event_ingest": True,
        "supports_add": True,
        "supports_update": True,
        "supports_noop": True,
        "supports_delete": True,
        "exports_entries": True,
        "exports_raw_state": True,
        "exports_source_event_ids": True,
        "exports_timestamps_or_order": True,
        "exports_object_keys": True,
        "exports_values": True,
        "exports_retrieval_ids": True,
        "exports_retrieval_scores": True,
        "exports_action_trace": True,
    }

    def __init__(self, task: MemUpdateTask | None = None, *, retrieval_policy: str = "normal_topk") -> None:
        super().__init__(retrieval_policy=retrieval_policy)
        self._task = _copy_task(task)
        self._observed_flags: set[str] = set()

    def _info_config(self) -> dict[str, Any]:
        return {
            **super()._info_config(),
            "control_id": self.control_id,
            "smoke_control": self.smoke_control,
            "leaderboard_eligible": self.leaderboard_eligible,
            "expected_failure_flags": list(self.expected_failure_flags),
        }

    def control_metadata(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "smoke_control": self.smoke_control,
            "leaderboard_eligible": self.leaderboard_eligible,
            "expected_failure_flags": list(self.expected_failure_flags),
        }

    def observed_failure_flags(self) -> tuple[str, ...]:
        return canonicalize_failure_flags(self._observed_flags)

    def canonical_output(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible snapshot of adapter identity and state."""
        return {
            "adapter_info": self.adapter_info().model_dump(mode="json"),
            "capabilities": self.capabilities().model_dump(mode="json"),
            "entries": [entry.model_dump(mode="json") for entry in self.export_entries()],
            "raw_state": self.export_raw_state(),
        }

    def reset(self, namespace: str, config: dict) -> Any:
        result = super().reset(namespace, config)
        self._observed_flags.clear()
        return result

    def _record(self, event: MemoryEvent, *, flags: tuple[str, ...] = (), **fields: Any) -> None:
        canonical = canonicalize_failure_flags(flags)
        self._observed_flags.update(canonical)
        record = {"event_id": event.event_id, **fields}
        if canonical:
            record.update(_failure_payload(canonical))
        self._actions.append(record)

    def _invalid(self, event: MemoryEvent, *, raw_action: Any = None) -> AdapterActionLog:
        flags = ("invalid_action_format",)
        self._record(event, flags=flags, operation=None, target_object_keys=[])
        payload = _failure_payload(flags)
        return AdapterActionLog(
            event_id=event.event_id,
            raw_action=event.raw_text if raw_action is None else raw_action,
            error={"code": "invalid_action_format", **payload},
        )

    def _expected_action(self, event: MemoryEvent):
        if self._task is None:
            return None
        actions = [action for action in self._task.gold.actions if action.action_id in event.gold_action_ids]
        if not actions:
            return None
        order = {action_id: index for index, action_id in enumerate(self._task.gold.action_sequence)}
        return min(actions, key=lambda action: order[action.action_id])

    def _parse_or_gold(self, event: MemoryEvent) -> ParsedEvent:
        parsed = parse_event_text(event)
        if parsed.format_valid:
            return parsed
        action = self._expected_action(event)
        if action is None:
            return parsed
        key = action.target_object_keys[0] if action.target_object_keys else None
        return ParsedEvent(action.operation, key, action.value, True)

    def _fallback_key(self, parsed: ParsedEvent) -> MemoryObjectKey | None:
        if parsed.object_key is not None:
            return parsed.object_key
        if self._task is not None and self._task.target_objects:
            return self._task.target_objects[0]
        return None

    def _fallback_value(self, key: MemoryObjectKey | None, parsed: ParsedEvent) -> Any:
        if parsed.value is not None:
            return parsed.value
        if key is not None and self._task is not None:
            return self._task.gold.final_state.get(key.canonical_id)
        return None

    def _apply_exact(
        self,
        event: MemoryEvent,
        parsed: ParsedEvent,
        *,
        flags: tuple[str, ...] = (),
    ) -> AdapterActionLog:
        key = parsed.object_key
        if parsed.operation is Operation.NOOP:
            self._record(event, flags=flags, operation="NOOP", target_object_keys=[])
            return AdapterActionLog(
                event_id=event.event_id,
                requested_operation=Operation.NOOP,
                effective_operation=Operation.NOOP,
                raw_action=event.raw_text,
            )
        if key is None or parsed.operation is None:
            return self._invalid(event)
        object_id = key.canonical_id
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
        canonical_flags = canonicalize_failure_flags(flags)
        self._record(
            event,
            flags=canonical_flags,
            operation=parsed.operation.value,
            target_object_keys=[object_id],
            entry_ids=affected,
        )
        return AdapterActionLog(
            event_id=event.event_id,
            requested_operation=parsed.operation,
            effective_operation=parsed.operation,
            affected_entry_ids=affected,
            raw_action=event.raw_text,
        )

    def _apply_append(self, event: MemoryEvent, parsed: ParsedEvent, *, flags: tuple[str, ...] = ()) -> AdapterActionLog:
        if parsed.operation is Operation.NOOP:
            self._record(event, flags=flags, operation="NOOP", target_object_keys=[])
            return AdapterActionLog(
                event_id=event.event_id,
                requested_operation=Operation.NOOP,
                effective_operation=Operation.NOOP,
                raw_action=event.raw_text,
            )
        if parsed.operation not in {Operation.ADD, Operation.UPDATE} or parsed.object_key is None:
            return self._invalid(event)
        entry = self._append_entry(event, parsed)
        object_id = parsed.object_key.canonical_id
        self._history.setdefault(object_id, []).append(parsed.value)
        self._state[object_id] = entry
        self._record(
            event,
            flags=flags,
            operation=parsed.operation.value,
            target_object_keys=[object_id],
            entry_ids=[entry.entry_id],
        )
        return AdapterActionLog(
            event_id=event.event_id,
            requested_operation=parsed.operation,
            effective_operation=parsed.operation,
            affected_entry_ids=[entry.entry_id],
            raw_action=event.raw_text,
        )

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        return self._apply_exact(event, self._parse_or_gold(event))

    def _wrong_key(self, key: MemoryObjectKey, dimension: str) -> MemoryObjectKey:
        candidates = [] if self._task is None else list(self._task.target_objects)
        for candidate in sorted(candidates, key=lambda item: item.canonical_id):
            if dimension == "entity" and candidate.entity != key.entity and candidate.attribute == key.attribute:
                return candidate
            if dimension == "attribute" and candidate.attribute != key.attribute and candidate.entity == key.entity:
                return candidate
        replacement = f"{getattr(key, dimension)}__corrupted_{dimension}"
        return key.model_copy(update={dimension: replacement})

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        return super().answer(query, mode)


class AlwaysAddAdapter(CorruptedControlAdapter):
    control_id = "control/always_add"
    adapter_id = control_id
    expected_failure_flags = ("false_write",)

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        parsed = self._parse_or_gold(event)
        if not parsed.format_valid:
            return self._invalid(event)
        key = self._fallback_key(parsed)
        if key is None:
            return self._invalid(event)
        action = self._expected_action(event)
        expected_operation = action.operation if action is not None else parsed.operation
        value = self._fallback_value(key, parsed)
        forced = ParsedEvent(Operation.ADD, key, value, True)
        flags = ("false_write",) if expected_operation is not Operation.ADD else ()
        return self._apply_append(event, forced, flags=flags)


class AlwaysNoopAdapter(CorruptedControlAdapter):
    control_id = "control/always_noop"
    adapter_id = control_id
    expected_failure_flags = ("missed_update",)

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        parsed = self._parse_or_gold(event)
        if not parsed.format_valid:
            return self._invalid(event)
        action = self._expected_action(event)
        expected_operation = action.operation if action is not None else parsed.operation
        flags = ("missed_update",) if expected_operation in {Operation.ADD, Operation.UPDATE} else ()
        return self._apply_exact(event, ParsedEvent(Operation.NOOP, None, None, True), flags=flags)


class StaleValueCopierAdapter(CorruptedControlAdapter):
    control_id = "control/stale_value_copier"
    adapter_id = control_id
    expected_failure_flags = ("stale_copied",)

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        result = super().answer(query, mode)
        if result.disposition is not AnswerDisposition.ANSWERED or len(query.target_object_keys) != 1:
            return result
        object_id = query.target_object_keys[0].canonical_id
        history = self._history.get(object_id, [])
        if len(history) < 2:
            return result
        stale = history[0]
        flags = ("stale_copied",)
        self._observed_flags.update(flags)
        return AnswerResult(
            query_id=query.query_id,
            raw_output=_json_output(stale),
            value=stale,
            error=_failure_payload(flags),
        )


class WrongEntityWriterAdapter(CorruptedControlAdapter):
    control_id = "control/wrong_entity_writer"
    adapter_id = control_id
    expected_failure_flags = ("wrong_entity",)

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        parsed = self._parse_or_gold(event)
        if not parsed.format_valid:
            return self._invalid(event)
        if parsed.operation not in {Operation.ADD, Operation.UPDATE, Operation.DELETE} or parsed.object_key is None:
            return self._apply_exact(event, parsed)
        key = self._wrong_key(parsed.object_key, "entity")
        corrupted = ParsedEvent(parsed.operation, key, parsed.value, True)
        return self._apply_exact(event, corrupted, flags=("wrong_entity",))


class WrongAttributeWriterAdapter(CorruptedControlAdapter):
    control_id = "control/wrong_attribute_writer"
    adapter_id = control_id
    expected_failure_flags = ("wrong_attribute",)

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        parsed = self._parse_or_gold(event)
        if not parsed.format_valid:
            return self._invalid(event)
        if parsed.operation not in {Operation.ADD, Operation.UPDATE, Operation.DELETE} or parsed.object_key is None:
            return self._apply_exact(event, parsed)
        key = self._wrong_key(parsed.object_key, "attribute")
        corrupted = ParsedEvent(parsed.operation, key, parsed.value, True)
        return self._apply_exact(event, corrupted, flags=("wrong_attribute",))


class InvalidFormatterAdapter(CorruptedControlAdapter):
    control_id = "control/invalid_formatter"
    adapter_id = control_id
    expected_failure_flags = ("invalid_action_format",)

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        error = self._ensure_ready(event)
        if error:
            return error
        return self._invalid(event, raw_action="<invalid action format>")


class CurrentNotRetrievedAdapter(CorruptedControlAdapter):
    control_id = "control/current_not_retrieved"
    adapter_id = control_id
    expected_failure_flags = ("current_not_retrieved",)

    def retrieve(self, query: MemoryQuery, k: int) -> RetrievalResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        result = super().retrieve(query, k)
        if result.error is not None:
            return result
        flags = ("current_not_retrieved",)
        self._observed_flags.update(flags)
        metadata = result.raw_result if isinstance(result.raw_result, dict) else {"base_result": result.raw_result}
        metadata = {**metadata, **_failure_payload(flags)}
        return RetrievalResult(query_id=query.query_id, entries=[], scores=[], raw_result=metadata)


class GoldRetrievedWrongAnswerAdapter(CorruptedControlAdapter):
    control_id = "control/gold_retrieved_wrong_answer"
    adapter_id = control_id
    expected_failure_flags = ("gold_retrieved_wrong_answer",)

    @staticmethod
    def _wrong_value(value: Any) -> Any:
        if isinstance(value, str):
            return f"__corrupted_wrong_answer__:{value}"
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return value + 1
        if value is None:
            return "__corrupted_wrong_answer__"
        if isinstance(value, list):
            return ["__corrupted_wrong_answer__"]
        if isinstance(value, dict):
            return {**value, "__corrupted_wrong_answer__": True}
        return "__corrupted_wrong_answer__"

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        result = super().answer(query, mode)
        if result.disposition is not AnswerDisposition.ANSWERED:
            return result
        wrong = self._wrong_value(result.value)
        flags = ("gold_retrieved_wrong_answer",)
        self._observed_flags.update(flags)
        return AnswerResult(
            query_id=query.query_id,
            raw_output=_json_output(wrong),
            value=wrong,
            error=_failure_payload(flags),
        )


CONTROL_ADAPTERS = {
    AlwaysAddAdapter.control_id: AlwaysAddAdapter,
    AlwaysNoopAdapter.control_id: AlwaysNoopAdapter,
    StaleValueCopierAdapter.control_id: StaleValueCopierAdapter,
    WrongEntityWriterAdapter.control_id: WrongEntityWriterAdapter,
    WrongAttributeWriterAdapter.control_id: WrongAttributeWriterAdapter,
    InvalidFormatterAdapter.control_id: InvalidFormatterAdapter,
    CurrentNotRetrievedAdapter.control_id: CurrentNotRetrievedAdapter,
    GoldRetrievedWrongAnswerAdapter.control_id: GoldRetrievedWrongAnswerAdapter,
}
CORRUPTED_CONTROLS = CONTROL_ADAPTERS


def build_corrupted_adapter(control_id: str, **kwargs: Any) -> CorruptedControlAdapter:
    try:
        adapter_type = CONTROL_ADAPTERS[control_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown corrupted control: {control_id!r}") from exc
    return adapter_type(**kwargs)


# Descriptive aliases keep the public names readable without introducing new
# control identities or another failure taxonomy.
AlwaysAddControlAdapter = AlwaysAddAdapter
AlwaysNoopControlAdapter = AlwaysNoopAdapter
StaleValueCopierControlAdapter = StaleValueCopierAdapter
WrongEntityWriterControlAdapter = WrongEntityWriterAdapter
WrongAttributeWriterControlAdapter = WrongAttributeWriterAdapter
InvalidFormatterControlAdapter = InvalidFormatterAdapter
CurrentNotRetrievedControlAdapter = CurrentNotRetrievedAdapter
GoldRetrievedWrongAnswerControlAdapter = GoldRetrievedWrongAnswerAdapter


__all__ = [
    "AlwaysAddAdapter",
    "AlwaysAddControlAdapter",
    "AlwaysNoopAdapter",
    "AlwaysNoopControlAdapter",
    "CONTROL_ADAPTERS",
    "CORRUPTED_CONTROLS",
    "CorruptedControlAdapter",
    "CurrentNotRetrievedAdapter",
    "CurrentNotRetrievedControlAdapter",
    "GoldRetrievedWrongAnswerAdapter",
    "GoldRetrievedWrongAnswerControlAdapter",
    "InvalidFormatterAdapter",
    "InvalidFormatterControlAdapter",
    "StaleValueCopierAdapter",
    "StaleValueCopierControlAdapter",
    "WrongAttributeWriterAdapter",
    "WrongAttributeWriterControlAdapter",
    "WrongEntityWriterAdapter",
    "WrongEntityWriterControlAdapter",
    "build_corrupted_adapter",
]
