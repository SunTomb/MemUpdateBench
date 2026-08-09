from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3
from mub.vnext.contracts.v3.common import MemoryObjectKeyV3
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, MemoryEventV3


_HISTORICAL_QUERY_TYPES = frozenset({
    QueryTypeV3.PREVIOUS,
    QueryTypeV3.POINT_IN_TIME,
    QueryTypeV3.TRANSITION,
    QueryTypeV3.ORDERED_HISTORY,
})
_HISTORICAL_SELECTOR_KINDS = frozenset({
    "previous",
    "exact_version",
    "event_anchor",
    "logical_time_anchor",
    "transition",
    "ordered_history",
})
_OPERATION_CAPABILITIES = {
    Operation.ADD: "supports_add",
    Operation.UPDATE: "supports_update",
    Operation.NOOP: "supports_noop",
    Operation.DELETE: "supports_delete",
}
_MUTATION_PATTERN = re.compile(r"^(Add|Update) (.+) with value (.+)\.$")
_DELETE_PATTERN = re.compile(r"^Delete (.+)\.$")
_DELETE_METADATA_PATTERN = re.compile(r"\[([^\[\]]+)\]\s*$")


class VisibleActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class ObservedActionV3:
    action_id: str
    event_id: str
    operation: Operation
    scope: ActionScope | None
    target_object_keys: tuple[MemoryObjectKeyV3, ...]
    value: Any
    effective_at: str | None


@dataclass(frozen=True)
class CoreTaskSupportV3:
    runtime_support: Mapping[str, bool]
    operation_support: Mapping[str, bool]
    query_support: Mapping[str, bool]
    metric_support: Mapping[str, bool]
    missing_capabilities: tuple[str, ...]

    @property
    def terminal_supported(self) -> bool:
        return (
            all(self.runtime_support.values())
            and all(self.operation_support.values())
            and all(self.query_support.values())
        )


def _parse_targets(
    rendered: str,
    target_objects: tuple[MemoryObjectKeyV3, ...],
) -> tuple[MemoryObjectKeyV3, ...]:
    by_identity = {key.canonical_id: key for key in target_objects}
    ids = tuple(part.strip() for part in rendered.split(","))
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise VisibleActionParseError("visible action has invalid target enumeration")
    try:
        return tuple(by_identity[item] for item in ids)
    except KeyError as exc:
        raise VisibleActionParseError(
            f"visible action targets undeclared object {exc.args[0]!r}"
        ) from exc


def _delete_metadata(event: MemoryEventV3) -> dict[str, str]:
    match = _DELETE_METADATA_PATTERN.search(event.raw_text)
    if match is None:
        raise VisibleActionParseError("visible DELETE lacks typed scope metadata")
    metadata: dict[str, str] = {}
    for field in match.group(1).split(";"):
        key, separator, value = field.strip().partition("=")
        if not separator or not key or not value or key in metadata:
            raise VisibleActionParseError("visible DELETE metadata is malformed")
        metadata[key] = value
    required = {"scope", "enumerated_targets", "event_logical_time", "effective_at"}
    if not required <= metadata.keys():
        raise VisibleActionParseError("visible DELETE metadata is incomplete")
    if event.timestamp != metadata["event_logical_time"]:
        raise VisibleActionParseError("visible DELETE logical time is not event-bound")
    return metadata


def parse_visible_action_v3(
    event: MemoryEventV3,
    target_objects: tuple[MemoryObjectKeyV3, ...],
) -> ObservedActionV3:
    if not isinstance(event, MemoryEventV3):
        event = MemoryEventV3.model_validate(event)
    targets = tuple(target_objects)
    normalized = event.normalized_text.strip()
    action_id = f"observed_action:{event.event_id}"

    if normalized == "No memory object changes.":
        return ObservedActionV3(
            action_id=action_id,
            event_id=event.event_id,
            operation=Operation.NOOP,
            scope=None,
            target_object_keys=(),
            value=None,
            effective_at=event.timestamp,
        )

    mutation = _MUTATION_PATTERN.fullmatch(normalized)
    if mutation is not None:
        operation = Operation.ADD if mutation.group(1) == "Add" else Operation.UPDATE
        parsed_targets = _parse_targets(mutation.group(2), targets)
        try:
            value = json.loads(mutation.group(3))
        except (TypeError, ValueError) as exc:
            raise VisibleActionParseError("visible mutation value is not canonical JSON") from exc
        if value is None:
            raise VisibleActionParseError("visible mutation value cannot be null")
        return ObservedActionV3(
            action_id=action_id,
            event_id=event.event_id,
            operation=operation,
            scope=ActionScope.OBJECT,
            target_object_keys=parsed_targets,
            value=value,
            effective_at=event.timestamp,
        )

    deletion = _DELETE_PATTERN.fullmatch(normalized)
    if deletion is not None:
        parsed_targets = _parse_targets(deletion.group(1), targets)
        metadata = _delete_metadata(event)
        try:
            scope = ActionScope(metadata["scope"])
        except ValueError as exc:
            raise VisibleActionParseError("visible DELETE scope is unknown") from exc
        enumerated = tuple(part.strip() for part in metadata["enumerated_targets"].split(","))
        if enumerated != tuple(key.canonical_id for key in parsed_targets):
            raise VisibleActionParseError("visible DELETE target bindings disagree")
        return ObservedActionV3(
            action_id=action_id,
            event_id=event.event_id,
            operation=Operation.DELETE,
            scope=scope,
            target_object_keys=parsed_targets,
            value=None,
            effective_at=metadata["effective_at"],
        )

    raise VisibleActionParseError("visible event does not contain a supported action surface")


def resolve_task_support_v3(
    task: MemUpdateTaskV3,
    capabilities: AdapterCapabilitiesV3,
    *,
    allow_append_only_observation: bool = False,
    answer_mode: str = "slot_direct",
) -> CoreTaskSupportV3:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("task must be a MemUpdateTaskV3")
    if not isinstance(capabilities, AdapterCapabilitiesV3):
        capabilities = AdapterCapabilitiesV3.model_validate(capabilities)

    if answer_mode not in {"slot_direct", "slot_prompt", "native_answer"}:
        raise ValueError(f"unknown answer mode: {answer_mode}")

    missing: set[str] = set()
    runtime_support = {
        "supports_isolated_reset": bool(capabilities.supports_isolated_reset),
        "supports_event_ingest": bool(capabilities.supports_event_ingest),
        "supports_native_answer": (
            bool(capabilities.supports_native_answer)
            if answer_mode == "native_answer"
            else True
        ),
    }
    for capability, supported in runtime_support.items():
        if not supported:
            missing.add(capability)
    observed_actions: list[ObservedActionV3] = []
    operation_support: dict[str, bool] = {}
    for event in task.events:
        try:
            observed_actions.append(parse_visible_action_v3(event, task.target_objects))
        except VisibleActionParseError:
            operation_support[f"PARSE:{event.event_id}"] = False
            missing.add("visible_action_parse")

    for operation in dict.fromkeys(action.operation for action in observed_actions):
        capability = _OPERATION_CAPABILITIES[operation]
        supported = bool(getattr(capabilities, capability))
        if operation is Operation.DELETE and allow_append_only_observation:
            supported = True
        operation_support[operation.value] = supported
        if not supported:
            missing.add(capability)

    if any(action.scope is ActionScope.TTL for action in observed_actions):
        ttl_supported = capabilities.supports_ttl or allow_append_only_observation
        operation_support["TTL"] = bool(ttl_supported)
        if not ttl_supported:
            missing.add("supports_ttl")
    if any(
        action.operation is Operation.DELETE
        and action.scope not in {None, ActionScope.OBJECT, ActionScope.TTL}
        for action in observed_actions
    ):
        scoped_supported = capabilities.supports_scoped_delete or allow_append_only_observation
        operation_support["SCOPED_DELETE"] = bool(scoped_supported)
        if not scoped_supported:
            missing.add("supports_scoped_delete")

    query_support: dict[str, bool] = {}
    for query in task.queries:
        supported = True
        if (
            query.query_type in _HISTORICAL_QUERY_TYPES
            or query.selector.kind in _HISTORICAL_SELECTOR_KINDS
        ):
            supported = capabilities.supports_historical_query
            if not supported:
                missing.add("supports_historical_query")
        if query.query_type in {
            QueryTypeV3.MULTI_OBJECT_CURRENT,
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        } or len(query.target_object_keys) > 1:
            supported = supported and capabilities.supports_multi_object_query
            if not capabilities.supports_multi_object_query:
                missing.add("supports_multi_object_query")
        query_support[query.query_id] = bool(supported)

    metric_support = {
        "exports_entries": capabilities.exports_entries,
        "exports_raw_state": capabilities.exports_raw_state,
        "exports_version_history": capabilities.exports_version_history,
        "exports_evidence_linkage": capabilities.exports_evidence_linkage,
        "exports_action_trace": capabilities.exports_action_trace,
    }
    return CoreTaskSupportV3(
        runtime_support=MappingProxyType(runtime_support),
        operation_support=MappingProxyType(operation_support),
        query_support=MappingProxyType(query_support),
        metric_support=MappingProxyType(metric_support),
        missing_capabilities=tuple(sorted(missing)),
    )


__all__ = [
    "CoreTaskSupportV3",
    "ObservedActionV3",
    "VisibleActionParseError",
    "parse_visible_action_v3",
    "resolve_task_support_v3",
]
