from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts.common import MemoryObjectKey, thaw_json
from mub.vnext.contracts.enums import (
    AnswerSchema,
    Difficulty,
    EventRole,
    Operation,
    QueryType,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.v3.common import typed_json_equal
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    EventAnchorSelector,
    ExactVersionSelector,
    LogicalTimeAnchorSelector,
    MemUpdateTaskV3,
    OrderedHistorySelector,
    PreviousSelector,
    SelectorV3,
    TransitionSelector,
    VersionHistoryEntry,
    resolve_selector_version_indices_v3,
)
from mub.vnext.generation.core import CoreEvent, GenerationContext, SemanticCore
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.identity import core_id, stable_id, task_id
from mub.vnext.generation.render import _answer_schema
from mub.vnext.validation.replay_v3 import (
    evaluate_evidence_v3,
    replay_task_v3,
    resolve_query_v3,
)


FAMILY_F_MICRO_PROFILE_ID = "family_f_diagnostic_micro_v1"
FAMILY_F_QUERY_TEMPLATE = (
    "Resolve the typed version selector for $targets. "
    "Return only the requested result."
)
FAMILY_F_SELECTOR_KINDS = (
    "current",
    "previous",
    "exact_version",
    "event_anchor",
    "logical_time_anchor",
    "transition",
    "ordered_history",
)
_VERSION_COUNT = 4
_EVENT_ANCHOR_NAME = "version_event_2"


@dataclass(frozen=True, slots=True)
class _TrajectorySpec:
    namespace: str
    entity: str
    attribute: str
    subkey: str | None
    values: tuple[str, ...]
    logical_times: tuple[str, ...]
    difficulty: Difficulty


_TRAJECTORIES = (
    _TrajectorySpec(
        namespace="family_f_notifications",
        entity="synthetic_user_0",
        attribute="preferred_channel",
        subkey=None,
        values=("email", "sms", "push", "voice"),
        logical_times=("00000010", "00000020", "00000030", "00000040"),
        difficulty=Difficulty.EASY,
    ),
    _TrajectorySpec(
        namespace="family_f_projects",
        entity="synthetic_project_1",
        attribute="delivery_status",
        subkey="primary",
        values=("planned", "active", "paused", "complete"),
        logical_times=("00000110", "00000120", "00000130", "00000140"),
        difficulty=Difficulty.MEDIUM,
    ),
    _TrajectorySpec(
        namespace="family_f_reservations",
        entity="synthetic_booking_2",
        attribute="assigned_room",
        subkey=None,
        values=("room_a", "room_b", "room_c", "room_d"),
        logical_times=("00000210", "00000220", "00000230", "00000240"),
        difficulty=Difficulty.HARD,
    ),
)


@dataclass(frozen=True, slots=True)
class _SelectorSpec:
    selector: SelectorV3
    core_query_type: QueryType
    requested_version_distance: int


_SELECTOR_SPECS = (
    _SelectorSpec(CurrentSelector(), QueryType.CURRENT_STATE, 0),
    _SelectorSpec(PreviousSelector(), QueryType.HISTORICAL_STATE, 1),
    _SelectorSpec(ExactVersionSelector(version_index=1), QueryType.HISTORICAL_STATE, 2),
    _SelectorSpec(EventAnchorSelector(event_id=_EVENT_ANCHOR_NAME), QueryType.HISTORICAL_STATE, 1),
    _SelectorSpec(LogicalTimeAnchorSelector(logical_time="00000025"), QueryType.HISTORICAL_STATE, 2),
    _SelectorSpec(TransitionSelector(from_version_index=1, to_version_index=3), QueryType.TRANSITION, 2),
    _SelectorSpec(OrderedHistorySelector(start_version_index=0, end_version_index=3), QueryType.HISTORICAL_STATE, 3),
)


@dataclass(frozen=True, slots=True)
class CompiledFamilyFMicroPilot:
    profile_id: str
    cores: tuple[SemanticCore, ...]
    tasks: tuple[MemUpdateTaskV3, ...]


def _identity(key: Any) -> tuple[str, str, str, str | None]:
    if isinstance(key, dict):
        return key["namespace"], key["entity"], key["attribute"], key.get("subkey")
    return key.namespace, key.entity, key.attribute, key.subkey


def _key_payload(key: MemoryObjectKey) -> dict[str, str | None]:
    return {
        "namespace": key.namespace,
        "entity": key.entity,
        "attribute": key.attribute,
        "subkey": key.subkey,
    }


def _trajectory_payload(spec: _TrajectorySpec) -> dict[str, Any]:
    return {
        "object_key": {
            "namespace": spec.namespace,
            "entity": spec.entity,
            "attribute": spec.attribute,
            "subkey": spec.subkey,
        },
        "values": list(spec.values),
        "logical_times": list(spec.logical_times),
    }


def _trajectory_identifier(spec: _TrajectorySpec) -> str:
    return stable_id("trajectory", _trajectory_payload(spec))


def _selector_payload(selector: SelectorV3) -> dict[str, Any]:
    return selector.model_dump(mode="json")


def _core_identifier(spec: _TrajectorySpec, selector: SelectorV3) -> str:
    return core_id(
        TaskFamily.CURRENT_HISTORICAL_QUERY.value,
        {
            **_trajectory_payload(spec),
            "selector": _selector_payload(selector),
        },
    )


@dataclass(frozen=True, slots=True)
class FamilyFSelectorResolution:
    selected_indices: tuple[int, ...]
    core_query_type: QueryType
    task_query_type: QueryTypeV3
    answer_schema: AnswerSchema
    answer: Any
    requested_version_distance: int


def _selector_query_types(selector: SelectorV3) -> tuple[QueryType, QueryTypeV3]:
    if isinstance(selector, CurrentSelector):
        return QueryType.CURRENT_STATE, QueryTypeV3.CURRENT
    if isinstance(selector, PreviousSelector):
        return QueryType.HISTORICAL_STATE, QueryTypeV3.PREVIOUS
    if isinstance(
        selector,
        (ExactVersionSelector, EventAnchorSelector, LogicalTimeAnchorSelector),
    ):
        return QueryType.HISTORICAL_STATE, QueryTypeV3.POINT_IN_TIME
    if isinstance(selector, TransitionSelector):
        return QueryType.TRANSITION, QueryTypeV3.TRANSITION
    if isinstance(selector, OrderedHistorySelector):
        return QueryType.HISTORICAL_STATE, QueryTypeV3.ORDERED_HISTORY
    raise ValueError("Family F requires one of the seven typed selectors")


def resolve_family_f_selector(
    selector: SelectorV3,
    entries,
    event_position: dict[str, int],
    event_times: dict[str, str],
    horizon: str | None,
) -> FamilyFSelectorResolution:
    selected_indices = resolve_selector_version_indices_v3(
        selector,
        entries,
        event_position,
        event_times,
        horizon,
    )
    if not selected_indices:
        raise ValueError("Family F typed selector does not resolve a version")
    by_index = {entry.version_index: entry for entry in entries}
    selected = tuple(by_index[index] for index in selected_indices)
    core_query_type, task_query_type = _selector_query_types(selector)
    if isinstance(selector, TransitionSelector):
        answer = {"from": selected[0].value, "to": selected[1].value}
        schema = AnswerSchema.OBJECT
        distance = selector.to_version_index - selector.from_version_index
    elif isinstance(selector, OrderedHistorySelector):
        answer = [entry.value for entry in selected]
        schema = AnswerSchema.LIST
        distance = selected_indices[-1] - selected_indices[0]
    else:
        answer = thaw_json(selected[-1].value)
        schema = _answer_schema(answer)
        distance = len(entries) - 1 - selected_indices[-1]
    return FamilyFSelectorResolution(
        selected_indices=selected_indices,
        core_query_type=core_query_type,
        task_query_type=task_query_type,
        answer_schema=schema,
        answer=answer,
        requested_version_distance=distance,
    )


def _event_ledger(events):
    event_anchors = tuple(event.metadata.get("event_anchor") for event in events)
    entries = tuple(
        VersionHistoryEntry(
            version_index=index,
            status="present",
            value=event.value,
            valid_from_event_id=event_anchors[index],
            valid_until_event_id=(
                event_anchors[index + 1]
                if index + 1 < len(event_anchors)
                else None
            ),
            logical_time=event.metadata.get("logical_time"),
            source_event_ids=(event_anchors[index],),
        )
        for index, event in enumerate(events)
    )
    event_position = {
        event_anchor: index for index, event_anchor in enumerate(event_anchors)
    }
    event_times = {
        event_anchor: entry.logical_time
        for event_anchor, entry in zip(event_anchors, entries)
    }
    return entries, event_position, event_times


def resolve_family_f_core_selector(core: SemanticCore) -> FamilyFSelectorResolution:
    if core.query_selector is None:
        raise ValueError("Family F core requires a typed selector")
    entries, event_position, event_times = _event_ledger(core.events)
    return resolve_family_f_selector(
        core.query_selector,
        entries,
        event_position,
        event_times,
        entries[-1].logical_time,
    )


def bind_family_f_core_selector(
    core: SemanticCore,
    rendered_event_ids,
) -> tuple[
    SelectorV3,
    FamilyFSelectorResolution,
    tuple[VersionHistoryEntry, ...],
    tuple[VersionHistoryEntry, ...],
]:
    if len(rendered_event_ids) != len(core.events):
        raise ValueError("Family F rendered event IDs must align with semantic events")
    semantic_anchor_to_rendered = {
        event.metadata["event_anchor"]: rendered_event_ids[index]
        for index, event in enumerate(core.events)
    }
    selector = core.query_selector
    if isinstance(selector, EventAnchorSelector):
        selector = EventAnchorSelector(
            event_id=semantic_anchor_to_rendered[selector.event_id]
        )
    entries = tuple(
        VersionHistoryEntry(
            version_index=index,
            status="present",
            value=event.value,
            valid_from_event_id=rendered_event_ids[index],
            valid_until_event_id=(
                rendered_event_ids[index + 1]
                if index + 1 < len(rendered_event_ids)
                else None
            ),
            logical_time=event.metadata["logical_time"],
            source_event_ids=(rendered_event_ids[index],),
        )
        for index, event in enumerate(core.events)
    )
    event_position = {
        event_id: index for index, event_id in enumerate(rendered_event_ids)
    }
    event_times = {
        event_id: core.events[index].metadata["logical_time"]
        for index, event_id in enumerate(rendered_event_ids)
    }
    resolution = resolve_family_f_selector(
        selector,
        entries,
        event_position,
        event_times,
        entries[-1].logical_time,
    )
    by_index = {entry.version_index: entry for entry in entries}
    selected = tuple(
        by_index[index] for index in resolution.selected_indices
    )
    return selector, resolution, selected, entries


def _profile(query_type: QueryType, distance: int) -> dict[str, Any]:
    return {
        "update_depth": _VERSION_COUNT,
        "active_object_count": 1,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": 0.0,
        "cross_slot_interleaving": 0.0,
        "stale_count": _VERSION_COUNT - 1,
        "context_length": _VERSION_COUNT,
        "context_order": "chronological",
        "version_metadata": "version_event_logical_time",
        "source_naturalness": "synthetic",
        "query_type": query_type.value,
        "requested_version_distance": distance,
    }


def _build_core(trajectory_index: int, selector_index: int) -> SemanticCore:
    trajectory = _TRAJECTORIES[trajectory_index]
    selector_spec = _SELECTOR_SPECS[selector_index]
    key = MemoryObjectKey(
        object_type="slot",
        namespace=trajectory.namespace,
        entity=trajectory.entity,
        attribute=trajectory.attribute,
        subkey=trajectory.subkey,
    )
    events = [
        CoreEvent(
            operation=Operation.ADD if version_index == 0 else Operation.UPDATE,
            object_keys=[key],
            value=value,
            role=(
                EventRole.LATEST_GOLD
                if version_index == _VERSION_COUNT - 1
                else EventRole.HISTORICAL_SUPPORT
            ),
            metadata={
                "event_anchor": f"version_event_{version_index}",
                "logical_time": trajectory.logical_times[version_index],
            },
        )
        for version_index, value in enumerate(trajectory.values)
    ]
    selector = selector_spec.selector.model_copy()
    if isinstance(selector, LogicalTimeAnchorSelector):
        selector = LogicalTimeAnchorSelector(
            logical_time=f"{int(trajectory.logical_times[1]) + 5:08d}"
        )
    identifier = _core_identifier(trajectory, selector)
    entries, event_position, event_times = _event_ledger(events)
    resolution = resolve_family_f_selector(
        selector,
        entries,
        event_position,
        event_times,
        entries[-1].logical_time,
    )
    distance = selector_spec.requested_version_distance
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.CURRENT_HISTORICAL_QUERY,
        difficulty=trajectory.difficulty,
        core_index=trajectory_index * len(_SELECTOR_SPECS) + selector_index,
        trajectory_id=_trajectory_identifier(trajectory),
        events=events,
        query_targets=[key],
        query_type=selector_spec.core_query_type,
        query_selector=selector,
        expected_answer=resolution.answer,
        profile=_profile(selector_spec.core_query_type, distance),
        stratification={
            "query_type": selector_spec.core_query_type.value,
            "requested_version_distance": distance,
        },
    )


def _match_trajectory(core: SemanticCore) -> tuple[int, _TrajectorySpec]:
    if len(core.query_targets) != 1:
        raise ValueError("Family F core requires exactly one query target")
    identity = _identity(core.query_targets[0])
    values = tuple(event.value for event in core.events)
    logical_times = tuple(event.metadata.get("logical_time") for event in core.events)
    matches = [
        (index, spec)
        for index, spec in enumerate(_TRAJECTORIES)
        if identity == (spec.namespace, spec.entity, spec.attribute, spec.subkey)
        and values == spec.values
        and logical_times == spec.logical_times
    ]
    if len(matches) != 1:
        raise ValueError("Family F trajectory is not canonical")
    return matches[0]


def _match_selector(selector: SelectorV3 | None, trajectory: _TrajectorySpec) -> tuple[int, _SelectorSpec]:
    if selector is None:
        raise ValueError("Family F core requires a typed selector")
    for index, spec in enumerate(_SELECTOR_SPECS):
        expected = spec.selector
        if isinstance(expected, LogicalTimeAnchorSelector):
            expected = LogicalTimeAnchorSelector(
                logical_time=f"{int(trajectory.logical_times[1]) + 5:08d}"
            )
        if selector == expected:
            return index, spec
    raise ValueError("Family F selector is not canonical")


def validate_family_f_core(core: SemanticCore) -> None:
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if core.task_family is not TaskFamily.CURRENT_HISTORICAL_QUERY:
        raise ValueError("Family F validator requires current_historical_query")
    if len(core.query_targets) != 1:
        raise ValueError("Family F core requires exactly one query target")
    if len(core.events) < 4:
        raise ValueError("Family F trajectory requires at least four present versions")
    expected_operations = (Operation.ADD,) + (Operation.UPDATE,) * (len(core.events) - 1)
    if tuple(event.operation for event in core.events) != expected_operations:
        raise ValueError("Family F trajectory must contain ADD followed only by UPDATE")
    if any(event.value is None for event in core.events):
        raise ValueError("Family F versions must all be present")
    target_identity = _identity(core.query_targets[0])
    if any(
        len(event.object_keys) != 1
        or _identity(event.object_keys[0]) != target_identity
        for event in core.events
    ):
        raise ValueError("Family F events must use one exact four-part object identity")
    logical_times = tuple(event.metadata.get("logical_time") for event in core.events)
    event_anchors = tuple(event.metadata.get("event_anchor") for event in core.events)
    if any(
        type(value) is not str or len(value) != 8 or not value.isdecimal()
        for value in logical_times
    ):
        raise ValueError("Family F requires canonical logical timestamps")
    if logical_times != tuple(sorted(logical_times)) or len(set(logical_times)) != len(logical_times):
        raise ValueError("Family F logical timestamps must be strictly increasing")
    if any(type(value) is not str or not value.strip() for value in event_anchors):
        raise ValueError("Family F versions require named event anchors")
    if len(set(event_anchors)) != len(event_anchors):
        raise ValueError("Family F event anchors must be unique")
    if any(set(event.metadata) != {"event_anchor", "logical_time"} for event in core.events):
        raise ValueError("Family F event metadata must contain only anchor and logical time")
    expected_roles = (EventRole.HISTORICAL_SUPPORT,) * (len(core.events) - 1) + (
        EventRole.LATEST_GOLD,
    )
    if tuple(event.role for event in core.events) != expected_roles:
        raise ValueError("Family F historical and current event roles are invalid")
    resolution = resolve_family_f_core_selector(core)
    if core.query_type is not resolution.core_query_type:
        raise ValueError("Family F selector/query mismatch")
    if thaw_json(core.expected_answer) != resolution.answer:
        raise ValueError("Family F selector answer or history order is invalid")
    if core.profile.get("query_type") != resolution.core_query_type.value:
        raise ValueError("Family F profile query type does not match typed selector")
    if core.profile.get("requested_version_distance") != resolution.requested_version_distance:
        raise ValueError("Family F profile distance does not match typed selector")
    if core.stratification.get("query_type") != resolution.core_query_type.value:
        raise ValueError("Family F stratification query type does not match typed selector")
    if core.stratification.get("requested_version_distance") != resolution.requested_version_distance:
        raise ValueError("Family F stratification distance does not match typed selector")


def validate_family_f_micro_core(core: SemanticCore) -> None:
    validate_family_f_core(core)
    if not isinstance(core, SemanticCore):
        raise TypeError("core must be a SemanticCore")
    if core.task_family is not TaskFamily.CURRENT_HISTORICAL_QUERY:
        raise ValueError("Family F validator requires current_historical_query")
    trajectory_index, trajectory = _match_trajectory(core)
    selector_index, selector_spec = _match_selector(core.query_selector, trajectory)
    if len(core.events) < 4 or len(core.events) != _VERSION_COUNT:
        raise ValueError("Family F trajectory requires exactly four present versions")
    expected_operations = (Operation.ADD,) + (Operation.UPDATE,) * (_VERSION_COUNT - 1)
    if tuple(event.operation for event in core.events) != expected_operations:
        raise ValueError("Family F trajectory must contain ADD followed only by UPDATE")
    if any(event.value is None for event in core.events):
        raise ValueError("Family F versions must all be present")
    if len({event.value for event in core.events}) != len(core.events):
        raise ValueError("Family F version values must be distinct")
    target_identity = _identity(core.query_targets[0])
    if any(len(event.object_keys) != 1 or _identity(event.object_keys[0]) != target_identity for event in core.events):
        raise ValueError("Family F events must use one exact four-part object identity")
    if any(event.operation is Operation.DELETE for event in core.events):
        raise ValueError("Family F forbids deletion-history interaction")
    expected_roles = (EventRole.HISTORICAL_SUPPORT,) * (_VERSION_COUNT - 1) + (EventRole.LATEST_GOLD,)
    if tuple(event.role for event in core.events) != expected_roles:
        raise ValueError("Family F historical and current event roles are invalid")
    logical_times = tuple(event.metadata.get("logical_time") for event in core.events)
    if any(type(value) is not str or len(value) != 8 or not value.isdecimal() for value in logical_times):
        raise ValueError("Family F requires canonical logical timestamps")
    if tuple(sorted(logical_times)) != logical_times or len(set(logical_times)) != len(logical_times):
        raise ValueError("Family F logical timestamps must be strictly increasing")
    if any(set(event.metadata) != {"event_anchor", "logical_time"} for event in core.events):
        raise ValueError("Family F event metadata must be canonical")
    if core.query_type is not selector_spec.core_query_type:
        raise ValueError("Family F selector/query mismatch")
    expected_answer = resolve_family_f_core_selector(core).answer
    if thaw_json(core.expected_answer) != expected_answer:
        raise ValueError("Family F selector answer or history order is invalid")
    expected_profile = _profile(
        selector_spec.core_query_type,
        selector_spec.requested_version_distance,
    )
    if dict(core.profile) != expected_profile:
        raise ValueError("Family F profile metadata is not canonical")
    expected_stratification = {
        "query_type": selector_spec.core_query_type.value,
        "requested_version_distance": selector_spec.requested_version_distance,
    }
    if dict(core.stratification) != expected_stratification:
        raise ValueError("Family F stratification metadata is not canonical")
    expected_index = trajectory_index * len(_SELECTOR_SPECS) + selector_index
    if core.core_index != expected_index:
        raise ValueError("Family F core index is not canonical")
    if core.trajectory_id != _trajectory_identifier(trajectory):
        raise ValueError("Family F trajectory identifier is not canonical")
    if core.core_id != _core_identifier(trajectory, core.query_selector):
        raise ValueError("Family F semantic core identifier is not canonical")


def generate_core_family_f_cores(config: CoreConfig) -> list[SemanticCore]:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    cores = [
        _build_core(trajectory_index, selector_index)
        for trajectory_index in range(len(_TRAJECTORIES))
        for selector_index in range(len(_SELECTOR_SPECS))
    ]
    for core in cores:
        validate_family_f_micro_core(core)
    counts = Counter(core.trajectory_id for core in cores)
    selector_counts = Counter(core.query_selector.kind for core in cores)
    if len(cores) != 21 or counts != Counter({trajectory_id: 7 for trajectory_id in counts}):
        raise ValueError("Family F micro-pilot requires exactly three seven-core trajectories")
    if selector_counts != Counter({kind: 3 for kind in FAMILY_F_SELECTOR_KINDS}):
        raise ValueError("Family F micro-pilot requires exactly three of every typed selector")
    return cores


def _validate_canonical_task_selector(selector, entries) -> None:
    if isinstance(selector, (CurrentSelector, PreviousSelector)):
        return
    if isinstance(selector, ExactVersionSelector):
        if selector.version_index != 1:
            raise ValueError("Family F exact-version selector is not canonical")
        return
    if isinstance(selector, EventAnchorSelector):
        if selector.event_id != entries[2].source_event_ids[0]:
            raise ValueError("Family F event selector is not canonical")
        return
    if isinstance(selector, LogicalTimeAnchorSelector):
        expected = f"{int(entries[1].logical_time) + 5:08d}"
        if selector.logical_time != expected:
            raise ValueError("Family F logical timestamp selector is not canonical")
        return
    if isinstance(selector, TransitionSelector):
        if (selector.from_version_index, selector.to_version_index) != (1, 3):
            raise ValueError("Family F transition endpoints are not canonical")
        return
    if isinstance(selector, OrderedHistorySelector):
        if (selector.start_version_index, selector.end_version_index) != (0, 3):
            raise ValueError("Family F ordered-history selector is not canonical")
        return
    raise ValueError("Family F task has an unsupported selector")


def family_f_query_tokens(selector, selected, entries) -> tuple[str, ...]:
    tokens = [f"selector={selector.kind}"]
    if isinstance(selector, TransitionSelector):
        first, second = selected
        tokens.extend((
            f"from_version_index={selector.from_version_index}",
            f"to_version_index={selector.to_version_index}",
            f"from_event_id={first.source_event_ids[0]}",
            f"to_event_id={second.source_event_ids[0]}",
            f"from_logical_time={first.logical_time}",
            f"to_logical_time={second.logical_time}",
        ))
    elif isinstance(selector, OrderedHistorySelector):
        tokens.extend((
            "history_order=oldest_to_newest",
            f"start_version_index={selector.start_version_index}",
            f"end_version_index={selector.end_version_index}",
        ))
        for entry in entries:
            tokens.extend((
                f"version_index={entry.version_index}",
                f"event_id={entry.source_event_ids[0]}",
                f"logical_time={entry.logical_time}",
            ))
    else:
        entry = selected[-1]
        tokens.extend((
            f"version_index={entry.version_index}",
            f"event_id={entry.source_event_ids[0]}",
            f"logical_time={entry.logical_time}",
        ))
        if isinstance(selector, LogicalTimeAnchorSelector):
            tokens.extend(("at_or_before", f"logical_time={selector.logical_time}"))
        if isinstance(selector, EventAnchorSelector):
            tokens.append(f"event_id={selector.event_id}")
    return tuple(dict.fromkeys(tokens))


def _task_semantic_binding(task, query, entries) -> tuple[_TrajectorySpec, SelectorV3]:
    task_identity = _identity(task.version_history[0].object_key)
    values = tuple(entry.value for entry in entries)
    logical_times = tuple(entry.logical_time for entry in entries)
    matches = [
        spec
        for spec in _TRAJECTORIES
        if task_identity == (spec.namespace, spec.entity, spec.attribute, spec.subkey)
        and values == spec.values
        and logical_times == spec.logical_times
    ]
    if len(matches) != 1:
        raise ValueError("Family F task trajectory is not canonical")
    trajectory = matches[0]
    semantic_selector = query.selector
    if isinstance(semantic_selector, EventAnchorSelector):
        semantic_selector = EventAnchorSelector(event_id=_EVENT_ANCHOR_NAME)
    return trajectory, semantic_selector


def validate_family_f_task(task: MemUpdateTaskV3) -> None:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("task must be a MemUpdateTaskV3")
    if task.task_family != TaskFamily.CURRENT_HISTORICAL_QUERY.value:
        raise ValueError("Family F task validator requires current_historical_query")
    if len(task.queries) != 1 or len(task.gold_evidence) != 1 or len(task.version_history) != 1:
        raise ValueError("Family F task requires one query, evidence row, and version ledger")
    ledger = task.version_history[0]
    entries = ledger.entries
    if len(entries) < 4:
        raise ValueError("Family F task requires at least four versions")
    if any(entry.status.value != "present" for entry in entries):
        raise ValueError("Family F version ledger cannot contain deletion history")
    expected_operations = (Operation.ADD,) + (Operation.UPDATE,) * (len(entries) - 1)
    if tuple(action.operation for action in task.actions) != expected_operations:
        raise ValueError("Family F action trajectory is invalid")
    if any(
        len(action.target_object_keys) != 1
        or _identity(action.target_object_keys[0]) != _identity(ledger.object_key)
        for action in task.actions
    ):
        raise ValueError("Family F actions must preserve exact four-part identity")
    logical_times = tuple(entry.logical_time for entry in entries)
    if (
        any(value is None for value in logical_times)
        or logical_times != tuple(sorted(logical_times))
        or len(set(logical_times)) != len(logical_times)
    ):
        raise ValueError("Family F ledger logical times must be strictly increasing")
    if any(
        entry.source_event_ids != (task.events[index].event_id,)
        or entry.valid_from_event_id != task.events[index].event_id
        or task.actions[index].event_id != task.events[index].event_id
        for index, entry in enumerate(entries)
    ):
        raise ValueError("Family F version ledger must be exactly source-event linked")
    replay = replay_task_v3(task)
    if replay.issues:
        raise ValueError(f"Family F task replay failed: {replay.issues[0].code}")
    if replay.horizon_logical_time != logical_times[-1]:
        raise ValueError("Family F ledger versions must all be horizon-active")

    query = task.queries[0]
    evidence = task.gold_evidence[0]
    event_position = {event.event_id: event.sequence_index for event in task.events}
    event_times = {
        event.event_id: event.timestamp
        for event in task.events
        if event.timestamp is not None
    }
    selector_resolution = resolve_family_f_selector(
        query.selector,
        entries,
        event_position,
        event_times,
        replay.horizon_logical_time,
    )
    by_index = {entry.version_index: entry for entry in entries}
    selected = tuple(
        by_index[index] for index in selector_resolution.selected_indices
    )
    if query.query_type is not selector_resolution.task_query_type:
        raise ValueError("Family F selector/query mismatch")
    canonical_resolution = resolve_query_v3(query, replay, task.events)
    if (
        canonical_resolution.issues
        or not typed_json_equal(canonical_resolution.answer, evidence.answer)
        or not typed_json_equal(selector_resolution.answer, evidence.answer)
    ):
        raise ValueError("Family F selector gold answer is not reproducible")
    evaluated_evidence = evaluate_evidence_v3(
        evidence,
        replay,
        query=query,
        events=task.events,
    )
    if evaluated_evidence.issues:
        raise ValueError("Family F derivation evidence is not reproducible")
    selected_events = tuple(
        dict.fromkeys(
            event_id for entry in selected for event_id in entry.source_event_ids
        )
    )
    if tuple(evidence.supporting_event_ids) != selected_events:
        raise ValueError("Family F gold evidence must exactly cover selected historical support")
    if tuple(_identity(key) for key in evidence.supporting_object_keys) != (
        _identity(ledger.object_key),
    ):
        raise ValueError("Family F gold evidence must use exact object identity")
    if query.answer_schema is not selector_resolution.answer_schema:
        raise ValueError("Family F answer schema does not match selector")
    query_tokens = family_f_query_tokens(query.selector, selected, entries)
    exact_anchor_suffix = " [" + "; ".join(query_tokens) + "]"
    if (
        not query.text.endswith(exact_anchor_suffix)
        or query.text.count(" [") != 1
        or query.text.count("selector=") != 1
    ):
        raise ValueError(
            "Family F visible selector anchors are incomplete, hidden, or contradictory"
        )
    event_by_id = {event.event_id: event for event in task.events}
    for entry in entries:
        event_id = entry.source_event_ids[0]
        required = (
            f"version_index={entry.version_index}",
            f"event_id={event_id}",
            f"logical_time={entry.logical_time}",
        )
        raw_text = event_by_id[event_id].raw_text
        exact_anchor_suffix = " [" + "; ".join(required) + "]"
        if (
            not raw_text.endswith(exact_anchor_suffix)
            or raw_text.count("version_index=") != 1
            or raw_text.count("event_id=") != 1
            or raw_text.count("logical_time=") != 1
        ):
            raise ValueError("Family F visible version ledger anchors are incomplete or contradictory")
    stratification = task.metadata.extra.get("stratification", {})
    if (
        stratification.get("query_type")
        != selector_resolution.core_query_type.value
        or stratification.get("requested_version_distance")
        != selector_resolution.requested_version_distance
    ):
        raise ValueError("Family F self-declared metadata does not match typed selector")
    expected_version_group = stable_id(
        "version_group", {"trajectory_id": task.metadata.split_key.trajectory_id}
    )
    if task.metadata.split_key.version_group_id != expected_version_group:
        raise ValueError("Family F version group is not trajectory-derived")


def validate_family_f_micro_task(
    task: MemUpdateTaskV3,
    core: SemanticCore | None = None,
) -> None:
    validate_family_f_task(task)
    if task.metadata.split is not Split.EVALUATION_ONLY:
        raise ValueError("Family F micro-pilot tasks must be diagnostic evaluation_only")
    entries = task.version_history[0].entries
    if len(entries) != _VERSION_COUNT:
        raise ValueError("Family F micro-pilot requires exactly four versions")
    query = task.queries[0]
    evidence = task.gold_evidence[0]
    _validate_canonical_task_selector(query.selector, entries)
    trajectory, semantic_selector = _task_semantic_binding(task, query, entries)
    selector_index, _ = _match_selector(semantic_selector, trajectory)
    expected_core_id = _core_identifier(trajectory, semantic_selector)
    expected_trajectory_id = _trajectory_identifier(trajectory)
    surface_variant = task.metadata.extra.get("surface_variant")
    expected_core_index = (
        _TRAJECTORIES.index(trajectory) * len(_SELECTOR_SPECS) + selector_index
    )
    if (
        type(surface_variant) is not int
        or task.task_id != task_id(expected_core_id, surface_variant)
        or task.metadata.split_key.semantic_core_id != expected_core_id
        or task.metadata.split_key.trajectory_id != expected_trajectory_id
        or task.metadata.extra.get("semantic_core_id") != expected_core_id
        or task.metadata.extra.get("core_index") != expected_core_index
        or task.source.provenance.get("semantic_core_id") != expected_core_id
        or task.source.provenance.get("trajectory_id") != expected_trajectory_id
    ):
        raise ValueError("Family F task semantic core binding is not canonical")

    from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
    from mub.vnext.generation.render import (
        _normalized_event_text,
        _render_event_text,
        _render_query_text,
    )

    canonical_core = _build_core(_TRAJECTORIES.index(trajectory), selector_index)
    surface_templates = CORE_SURFACE_CATALOG_V1.template_sets[surface_variant]
    expected_surface_template = surface_templates[0]
    operation_templates = {
        Operation.ADD: surface_templates[1],
        Operation.UPDATE: surface_templates[2],
        Operation.DELETE: surface_templates[3],
        Operation.NOOP: surface_templates[4],
    }
    if (
        task.difficulty is not canonical_core.difficulty
        or task.metadata.profile_name is not canonical_core.difficulty
        or len(task.events) != len(canonical_core.events)
        or len(task.actions) != len(canonical_core.events)
        or len(task.version_history[0].entries) != len(canonical_core.events)
        or task.metadata.extra.get("surface_template")
        != expected_surface_template
        or task.source.provenance.get("surface_template")
        != expected_surface_template
        or task.source.provenance.get("surface_variant") != surface_variant
    ):
        raise ValueError("Family F micro-pilot task/core projection is not canonical")
    for version_index, (task_event, task_action, core_event) in enumerate(zip(
        task.events,
        task.actions,
        canonical_core.events,
    )):
        renderer_metadata = task_event.metadata.get("__surface_renderer__")
        expected_raw_text = _render_event_text(
            core_event,
            operation_templates,
        ) + (
            " ["
            f"version_index={version_index}; "
            f"event_id={task_event.event_id}; "
            f"logical_time={core_event.metadata['logical_time']}"
            "]"
        )
        if (
            task_event.timestamp != core_event.metadata["logical_time"]
            or task_event.role is not core_event.role
            or task_event.raw_text != expected_raw_text
            or task_event.normalized_text != _normalized_event_text(core_event)
            or task_event.speaker
            != CORE_SURFACE_CATALOG_V1.speakers[surface_variant]
            or not isinstance(renderer_metadata, Mapping)
            or renderer_metadata.get("surface_variant") != surface_variant
            or renderer_metadata.get("surface_template")
            != expected_surface_template
            or task_event.metadata.get("event_anchor")
            != core_event.metadata["event_anchor"]
            or task_event.metadata.get("logical_time")
            != core_event.metadata["logical_time"]
            or task_action.operation is not core_event.operation
            or task_action.effective_at != core_event.metadata["logical_time"]
            or len(task_action.target_object_keys) != 1
            or _identity(task_action.target_object_keys[0])
            != _identity(core_event.object_keys[0])
            or not typed_json_equal(task_action.value, core_event.value)
        ):
            raise ValueError(
                "Family F micro-pilot event/action core projection is not canonical"
            )
    bound_selector, _, selected_entries, bound_entries = (
        bind_family_f_core_selector(
            canonical_core,
            tuple(event.event_id for event in task.events),
        )
    )
    expected_query_text = _render_query_text(
        canonical_core,
        FAMILY_F_QUERY_TEMPLATE,
    ) + " [" + "; ".join(
        family_f_query_tokens(
            bound_selector,
            selected_entries,
            bound_entries,
        )
    ) + "]"
    if task.queries[0].text != expected_query_text:
        raise ValueError(
            "Family F micro-pilot query surface projection is not canonical"
        )
    if core is not None:
        validate_family_f_micro_core(core)
        if task.metadata.split_key.semantic_core_id != core.core_id:
            raise ValueError("Family F compiler task/core binding mismatch")
        if task.metadata.split_key.trajectory_id != core.trajectory_id:
            raise ValueError("Family F compiler trajectory binding mismatch")
        if evidence.answer != core.expected_answer:
            raise ValueError("Family F compiler answer differs from semantic core")
        expected_selector = core.query_selector
        if isinstance(expected_selector, EventAnchorSelector):
            anchor_index = next(
                index
                for index, event in enumerate(core.events)
                if event.metadata["event_anchor"] == expected_selector.event_id
            )
            expected_selector = EventAnchorSelector(
                event_id=task.events[anchor_index].event_id
            )
        if query.selector != expected_selector:
            raise ValueError("Family F compiler typed selector differs from semantic core")


def _validate_compiled(tasks: list[MemUpdateTaskV3], cores: list[SemanticCore]) -> None:
    if len(cores) != 21:
        raise ValueError("Family F micro-pilot requires exactly 21 semantic cores")
    if len(tasks) != 84:
        raise ValueError("Family F micro-pilot requires exactly 84 tasks")
    core_by_id = {core.core_id: core for core in cores}
    if len(core_by_id) != 21:
        raise ValueError("Family F semantic core IDs must be unique")
    by_core: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    by_trajectory: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    for task in tasks:
        core_identifier = task.metadata.split_key.semantic_core_id
        if core_identifier not in core_by_id:
            raise ValueError("Family F compiled task references an unknown core")
        validate_family_f_micro_task(task, core_by_id[core_identifier])
        by_core[core_identifier].append(task)
        by_trajectory[task.metadata.split_key.trajectory_id].append(task)
    if set(by_core) != set(core_by_id):
        raise ValueError("Family F compiled tasks do not cover exact semantic cores")
    for variants in by_core.values():
        if len(variants) != 4 or {task.metadata.extra["surface_variant"] for task in variants} != {0, 1, 2, 3}:
            raise ValueError("Family F core requires four surface variants")
        if len({task.semantic_hash for task in variants}) != 1:
            raise ValueError("Family F four surfaces are not semantically equivalent")
    if Counter(len(group) for group in by_trajectory.values()) != Counter({28: 3}):
        raise ValueError("Family F trajectories require seven cores and four surfaces each")
    for group in by_trajectory.values():
        if len({task.metadata.split_key.version_group_id for task in group}) != 1:
            raise ValueError("Family F trajectory must share one version group")
        if {task.metadata.split for task in group} != {Split.EVALUATION_ONLY}:
            raise ValueError("Family F trajectory must share one diagnostic split")


def compile_family_f_micro_pilot(
    config: CoreConfig,
    *,
    code_revision: str,
) -> CompiledFamilyFMicroPilot:
    cores = generate_core_family_f_cores(config)
    context = GenerationContext(
        config=config,
        code_revision=code_revision,
        generator_name="memupdatebench_vnext_core_family_f_micro",
    )
    tasks = [
        render_core_v3(
            core,
            split=Split.EVALUATION_ONLY,
            surface_variant=surface_variant,
            context=context,
        )
        for core in cores
        for surface_variant in range(4)
    ]
    _validate_compiled(tasks, cores)
    return CompiledFamilyFMicroPilot(
        profile_id=FAMILY_F_MICRO_PROFILE_ID,
        cores=tuple(cores),
        tasks=tuple(tasks),
    )


__all__ = [
    "CompiledFamilyFMicroPilot",
    "FAMILY_F_MICRO_PROFILE_ID",
    "FAMILY_F_SELECTOR_KINDS",
    "FamilyFSelectorResolution",
    "bind_family_f_core_selector",
    "compile_family_f_micro_pilot",
    "family_f_query_tokens",
    "generate_core_family_f_cores",
    "resolve_family_f_core_selector",
    "resolve_family_f_selector",
    "validate_family_f_core",
    "validate_family_f_micro_core",
    "validate_family_f_micro_task",
    "validate_family_f_task",
]
