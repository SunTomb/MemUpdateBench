from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from mub.vnext.contracts.common import MemoryObjectKey, thaw_json
from mub.vnext.contracts.enums import ActionScope, Difficulty, EventRole, Operation, Split, TaskFamily
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.core import CoreEvent, GenerationContext, SemanticCore
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.identity import core_id, trajectory_id


FAMILY_E_LIFECYCLE_CELLS = (
    "explicit_object_or_attribute_deletion",
    "entity_wide_deletion",
    "namespace_privacy_wipe",
    "correction_versus_deletion_hard_negative",
    "logical_ttl_expiry",
    "post_delete_similar_retrieval",
    "delete_then_relearn",
    "scoped_delete_protected_collateral",
)
FAMILY_E_DELETE_SCOPES_BY_CELL = MappingProxyType({
    "explicit_object_or_attribute_deletion": (
        ActionScope.OBJECT,
        ActionScope.ATTRIBUTE,
    ),
    "entity_wide_deletion": (ActionScope.ENTITY,),
    "namespace_privacy_wipe": (ActionScope.NAMESPACE,),
    "correction_versus_deletion_hard_negative": (ActionScope.OBJECT,),
    "logical_ttl_expiry": (ActionScope.TTL,),
    "post_delete_similar_retrieval": (ActionScope.OBJECT,),
    "delete_then_relearn": (ActionScope.OBJECT,),
    "scoped_delete_protected_collateral": (ActionScope.ATTRIBUTE,),
})
FAMILY_E_MICRO_PROFILE_ID = "family_e_diagnostic_micro_v1"
_EXAMPLES_PER_CELL = 3
_FAMILY_E_POSITION_PADDING = MappingProxyType({
    "early": (0, 7),
    "middle": (3, 4),
    "final": (7, 0),
})


@dataclass(frozen=True, slots=True)
class _FamilyECellShape:
    operations: tuple[Operation, ...]
    lifecycles: tuple[str, ...]
    query_target_count: int
    delete_target_count: int
    unique_object_count: int
    protected_count: int
    final_present_count: int
    peak_active_object_count: int

    @property
    def roles(self) -> tuple[EventRole, ...]:
        return tuple(
            EventRole.DELETION
            if operation is Operation.DELETE
            else EventRole.NEUTRAL
            if operation is Operation.NOOP
            else EventRole.LATEST_GOLD
            for operation in self.operations
        )


_FAMILY_E_CELL_SHAPES = MappingProxyType({
    ("explicit_object_or_attribute_deletion", ActionScope.OBJECT): _FamilyECellShape(
        operations=(Operation.ADD, Operation.DELETE),
        lifecycles=("seed", "explicit_delete"),
        query_target_count=1,
        delete_target_count=1,
        unique_object_count=1,
        protected_count=0,
        final_present_count=0,
        peak_active_object_count=1,
    ),
    ("explicit_object_or_attribute_deletion", ActionScope.ATTRIBUTE): _FamilyECellShape(
        operations=(Operation.ADD, Operation.ADD, Operation.DELETE),
        lifecycles=("seed", "seed", "explicit_delete"),
        query_target_count=2,
        delete_target_count=2,
        unique_object_count=2,
        protected_count=0,
        final_present_count=0,
        peak_active_object_count=2,
    ),
    ("entity_wide_deletion", ActionScope.ENTITY): _FamilyECellShape(
        operations=(Operation.ADD, Operation.ADD, Operation.DELETE),
        lifecycles=("seed", "seed", "entity_wide_delete"),
        query_target_count=2,
        delete_target_count=2,
        unique_object_count=2,
        protected_count=0,
        final_present_count=0,
        peak_active_object_count=2,
    ),
    ("namespace_privacy_wipe", ActionScope.NAMESPACE): _FamilyECellShape(
        operations=(Operation.ADD, Operation.ADD, Operation.ADD, Operation.DELETE),
        lifecycles=(
            "synthetic_private_seed",
            "synthetic_private_seed",
            "synthetic_private_seed",
            "namespace_privacy_wipe",
        ),
        query_target_count=3,
        delete_target_count=3,
        unique_object_count=3,
        protected_count=0,
        final_present_count=0,
        peak_active_object_count=3,
    ),
    ("correction_versus_deletion_hard_negative", ActionScope.OBJECT): _FamilyECellShape(
        operations=(Operation.ADD, Operation.UPDATE),
        lifecycles=("seed", "correction_not_deletion"),
        query_target_count=1,
        delete_target_count=0,
        unique_object_count=1,
        protected_count=0,
        final_present_count=1,
        peak_active_object_count=1,
    ),
    ("logical_ttl_expiry", ActionScope.TTL): _FamilyECellShape(
        operations=(Operation.ADD, Operation.DELETE, Operation.NOOP),
        lifecycles=("ttl_seed", "ttl_schedule", "ttl_boundary_probe"),
        query_target_count=1,
        delete_target_count=1,
        unique_object_count=1,
        protected_count=0,
        final_present_count=0,
        peak_active_object_count=1,
    ),
    ("post_delete_similar_retrieval", ActionScope.OBJECT): _FamilyECellShape(
        operations=(Operation.ADD, Operation.ADD, Operation.DELETE),
        lifecycles=("seed", "similar_protected", "delete_before_similar_probe"),
        query_target_count=2,
        delete_target_count=1,
        unique_object_count=2,
        protected_count=1,
        final_present_count=1,
        peak_active_object_count=2,
    ),
    ("delete_then_relearn", ActionScope.OBJECT): _FamilyECellShape(
        operations=(Operation.ADD, Operation.DELETE, Operation.ADD),
        lifecycles=("seed", "delete", "relearn"),
        query_target_count=1,
        delete_target_count=1,
        unique_object_count=1,
        protected_count=0,
        final_present_count=1,
        peak_active_object_count=1,
    ),
    ("scoped_delete_protected_collateral", ActionScope.ATTRIBUTE): _FamilyECellShape(
        operations=(Operation.ADD, Operation.ADD, Operation.ADD, Operation.DELETE),
        lifecycles=("seed", "seed", "protected_seed", "scoped_delete"),
        query_target_count=3,
        delete_target_count=2,
        unique_object_count=3,
        protected_count=1,
        final_present_count=1,
        peak_active_object_count=3,
    ),
})


@dataclass(frozen=True, slots=True)
class CompiledFamilyEMicroPilot:
    profile_id: str
    cores: tuple[SemanticCore, ...]
    tasks: tuple[MemUpdateTaskV3, ...]


def _key(namespace: str, entity: str, attribute: str, subkey: str | None = None) -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=subkey,
    )


def _identity(key: MemoryObjectKey) -> tuple[str, str, str, str | None]:
    return key.namespace, key.entity, key.attribute, key.subkey


def _key_payload(key: MemoryObjectKey) -> dict[str, str | None]:
    return {
        "namespace": key.namespace,
        "entity": key.entity,
        "attribute": key.attribute,
        "subkey": key.subkey,
    }


def _family_e_event_identity_payload(event: CoreEvent) -> dict:
    return {
        "operation": event.operation.value,
        "object_keys": [_key_payload(key) for key in event.object_keys],
        "value": event.value,
        "role": event.role.value,
        "metadata": thaw_json(event.metadata),
    }


def _family_e_core_identifier(
    cell: str,
    example: int,
    events,
    query_targets,
) -> str:
    payload = {
        "family": TaskFamily.DELETION_FORGETTING.value,
        "cell": cell,
        "example": example,
        "events": [_family_e_event_identity_payload(event) for event in events],
        "query_targets": [_key_payload(key) for key in query_targets],
    }
    return core_id(TaskFamily.DELETION_FORGETTING.value, payload)


def _event(
    operation: Operation,
    keys: list[MemoryObjectKey],
    value: str | None,
    lifecycle: str,
    logical_time: int,
    *,
    scope: ActionScope = ActionScope.OBJECT,
    protected: bool = False,
    effective_at: int | None = None,
) -> CoreEvent:
    metadata: dict[str, Any] = {
        "lifecycle": lifecycle,
        "action_scope": scope.value,
        "logical_time": f"{logical_time:08d}",
    }
    metadata["effective_at"] = f"{(logical_time if effective_at is None else effective_at):08d}"
    if protected:
        metadata["protected_collateral"] = True
    return CoreEvent(
        operation=operation,
        object_keys=keys,
        value=value,
        role=EventRole.DELETION if operation is Operation.DELETE else EventRole.LATEST_GOLD,
        metadata=metadata,
    )


def _profile(
    scope: ActionScope,
    relearning: str,
    event_count: int,
    active_object_count: int,
    noop_density: float,
) -> dict[str, Any]:
    return {
        "update_depth": 1,
        "active_object_count": active_object_count,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": noop_density,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": event_count,
        "context_order": "chronological",
        "version_metadata": "logical_time",
        "source_naturalness": "synthetic",
        "deletion_scope": scope.value,
        "relearning_condition": relearning,
    }


def _mixed_absence_answer(keys: list[MemoryObjectKey], absent: set[str]) -> dict[str, bool]:
    return {key.canonical_id: key.canonical_id in absent for key in keys}


def _peak_active_object_count(events: list[CoreEvent]) -> int:
    active: set[tuple[str, str, str, str | None]] = set()
    peak = 0
    for event in events:
        identities = {_identity(key) for key in event.object_keys}
        if event.operation in {Operation.ADD, Operation.UPDATE}:
            active.update(identities)
        elif event.operation is Operation.DELETE:
            active.difference_update(identities)
        peak = max(peak, len(active))
    return peak


def _scope_for_cell(cell: str, example: int) -> ActionScope:
    permitted = FAMILY_E_DELETE_SCOPES_BY_CELL[cell]
    if len(permitted) == 1:
        return permitted[0]
    return permitted[0] if example == 0 else permitted[1]


def _build_cell_core(
    cell: str,
    example: int,
    core_index: int,
    *,
    difficulty: Difficulty | None = None,
    deletion_position: str | None = None,
    scope_override: ActionScope | None = None,
) -> SemanticCore:
    namespace = f"family_e_ns_{core_index:02d}"
    entity = f"synthetic_subject_{example}"
    primary = _key(namespace, entity, "contact_channel")
    old = f"synthetic_value_{core_index:02d}_old"
    new = f"synthetic_value_{core_index:02d}_new"
    query_targets: list[MemoryObjectKey]
    expected_answer: Any
    relearning = "none"
    scope = scope_override or _scope_for_cell(cell, example)

    if cell == "explicit_object_or_attribute_deletion":
        if scope is ActionScope.OBJECT:
            targets = [primary]
        else:
            targets = [
                _key(namespace, entity, "contact_channel", "primary"),
                _key(namespace, entity, "contact_channel", "backup"),
            ]
        events = [
            *[_event(Operation.ADD, [key], f"synthetic_value_{core_index:02d}_{i}", "seed", 10 + i) for i, key in enumerate(targets)],
            _event(Operation.DELETE, targets, None, "explicit_delete", 20, scope=scope),
        ]
        query_targets = targets
        expected_answer = None
    elif cell == "entity_wide_deletion":
        targets = [primary, _key(namespace, entity, "preferred_locale")]
        events = [
            _event(Operation.ADD, [targets[0]], old, "seed", 10),
            _event(Operation.ADD, [targets[1]], f"synthetic_locale_{example}", "seed", 11),
            _event(Operation.DELETE, targets, None, "entity_wide_delete", 20, scope=scope),
        ]
        query_targets = targets
        expected_answer = None
    elif cell == "namespace_privacy_wipe":
        targets = [
            primary,
            _key(namespace, f"synthetic_subject_{example}_peer", "contact_channel"),
            _key(namespace, entity, "privacy_note"),
        ]
        events = [
            *[
                _event(
                    Operation.ADD,
                    [key],
                    f"synthetic_private_{core_index:02d}_{i}",
                    "synthetic_private_seed",
                    10 + i,
                )
                for i, key in enumerate(targets)
            ],
            _event(Operation.DELETE, targets, None, "namespace_privacy_wipe", 20, scope=scope),
        ]
        query_targets = targets
        expected_answer = None
    elif cell == "correction_versus_deletion_hard_negative":
        events = [
            _event(Operation.ADD, [primary], old, "seed", 10),
            _event(Operation.UPDATE, [primary], new, "correction_not_deletion", 20),
        ]
        query_targets = [primary]
        expected_answer = new
        relearning = "correction_hard_negative"
    elif cell == "logical_ttl_expiry":
        expiry = 20 + example
        events = [
            _event(Operation.ADD, [primary], old, "ttl_seed", 10),
            _event(Operation.DELETE, [primary], None, "ttl_schedule", 11, scope=scope, effective_at=expiry),
            CoreEvent(
                operation=Operation.NOOP,
                object_keys=[],
                value=None,
                role=EventRole.NEUTRAL,
                metadata={
                    "lifecycle": "ttl_boundary_probe",
                    "action_scope": ActionScope.OBJECT.value,
                    "logical_time": f"{expiry:08d}",
                    "effective_at": f"{expiry:08d}",
                    "surface_statement": "The deterministic logical TTL boundary is now reached.",
                },
            ),
        ]
        query_targets = [primary]
        expected_answer = None
    elif cell == "post_delete_similar_retrieval":
        similar = _key(namespace, f"synthetic_subject_{example}_similar", "contact_channel")
        events = [
            _event(Operation.ADD, [primary], old, "seed", 10),
            _event(Operation.ADD, [similar], new, "similar_protected", 11, protected=True),
            _event(Operation.DELETE, [primary], None, "delete_before_similar_probe", 20, scope=scope),
        ]
        query_targets = [primary, similar]
        expected_answer = _mixed_absence_answer(query_targets, {primary.canonical_id})
    elif cell == "delete_then_relearn":
        events = [
            _event(Operation.ADD, [primary], old, "seed", 10),
            _event(Operation.DELETE, [primary], None, "delete", 20, scope=scope),
            _event(Operation.ADD, [primary], new, "relearn", 30),
        ]
        query_targets = [primary]
        expected_answer = new
        relearning = "new_value_after_delete"
    elif cell == "scoped_delete_protected_collateral":
        targets = [
            _key(namespace, entity, "contact_channel", "primary"),
            _key(namespace, entity, "contact_channel", "backup"),
        ]
        collateral = _key(namespace, entity, "contact_channel_note")
        events = [
            _event(Operation.ADD, [targets[0]], old, "seed", 10),
            _event(Operation.ADD, [targets[1]], new, "seed", 11),
            _event(Operation.ADD, [collateral], f"synthetic_protected_{example}", "protected_seed", 12, protected=True),
            _event(Operation.DELETE, targets, None, "scoped_delete", 20, scope=scope),
        ]
        query_targets = [*targets, collateral]
        expected_answer = _mixed_absence_answer(query_targets, {key.canonical_id for key in targets})
    else:
        raise ValueError(f"unsupported Family E lifecycle cell {cell}")

    if deletion_position is not None:
        if cell == "correction_versus_deletion_hard_negative":
            if deletion_position != "not_applicable":
                raise ValueError("Family E correction hard negative has no deletion position")
        else:
            if deletion_position not in _FAMILY_E_POSITION_PADDING:
                raise ValueError("Family E deletion position must be early, middle, or final")
            before_count, after_count = _FAMILY_E_POSITION_PADDING[deletion_position]

            def padding_event(logical_time: int) -> CoreEvent:
                return CoreEvent(
                    operation=Operation.NOOP,
                    object_keys=[],
                    value=None,
                    role=EventRole.NEUTRAL,
                    metadata={
                        "lifecycle": "position_padding",
                        "action_scope": ActionScope.OBJECT.value,
                        "logical_time": f"{logical_time:08d}",
                        "effective_at": f"{logical_time:08d}",
                    },
                )

            events = [
                *(padding_event(1 + index) for index in range(before_count)),
                *events,
                *(padding_event(90 + index) for index in range(after_count)),
            ]

    identifier = _family_e_core_identifier(
        cell,
        example,
        events,
        query_targets,
    )
    stratification: dict[str, str | int | float | bool] = {
        "lifecycle_cell": cell,
        "cell_example": example,
        "deletion_scope": scope.value,
        "relearning_condition": relearning,
        "operation_signature": ",".join(
            event.operation.value
            for event in events
            if event.metadata.get("lifecycle") != "position_padding"
        ),
    }
    if deletion_position is not None:
        stratification["deletion_position"] = deletion_position
    if cell == "logical_ttl_expiry":
        expiry_at = next(event.metadata["effective_at"] for event in events if event.operation is Operation.DELETE)
        stratification["ttl_expiry_at"] = expiry_at
        stratification["query_logical_time"] = expiry_at
    if cell == "delete_then_relearn":
        stratification["forgotten_value"] = old
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.DELETION_FORGETTING,
        difficulty=difficulty or (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[example],
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, f"{cell}:{example}"),
        events=events,
        query_targets=query_targets,
        expected_answer=expected_answer,
        profile=_profile(
            scope,
            relearning,
            len(events),
            _peak_active_object_count(events),
            sum(event.operation is Operation.NOOP for event in events) / len(events),
        ),
        stratification=stratification,
    )


def _scope_matches(scope: ActionScope, selector: MemoryObjectKey, candidate: MemoryObjectKey) -> bool:
    if scope in {ActionScope.OBJECT, ActionScope.TTL}:
        return _identity(candidate) == _identity(selector)
    if scope is ActionScope.ATTRIBUTE:
        return (
            candidate.namespace,
            candidate.entity,
            candidate.attribute,
        ) == (selector.namespace, selector.entity, selector.attribute)
    if scope is ActionScope.ENTITY:
        return (candidate.namespace, candidate.entity) == (selector.namespace, selector.entity)
    if scope is ActionScope.NAMESPACE:
        return candidate.namespace == selector.namespace
    return False


def validate_family_e_core(core: SemanticCore) -> None:
    if core.task_family is not TaskFamily.DELETION_FORGETTING:
        raise ValueError("Family E validator requires deletion_forgetting")
    cell = core.stratification.get("lifecycle_cell")
    if cell not in FAMILY_E_LIFECYCLE_CELLS:
        raise ValueError("Family E lifecycle cell is not approved")
    permitted_scopes = FAMILY_E_DELETE_SCOPES_BY_CELL[cell]
    try:
        profile_scope = ActionScope(core.profile["deletion_scope"])
        stratification_scope = ActionScope(core.stratification["deletion_scope"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Family E deletion scope is invalid") from exc
    if (
        profile_scope is not stratification_scope
        or profile_scope not in permitted_scopes
    ):
        raise ValueError("Family E deletion scope is not permitted for its lifecycle cell")
    if core.profile.get("relearning_condition") != core.stratification.get("relearning_condition"):
        raise ValueError("Family E relearning profile mismatch")
    expected_delete_scope = stratification_scope

    state: dict[tuple[str, str, str, str | None], Any] = {}
    known: dict[tuple[str, str, str, str | None], MemoryObjectKey] = {}
    protected: set[tuple[str, str, str, str | None]] = set()
    delete_events: list[CoreEvent] = []
    padding_events: list[CoreEvent] = []
    lifecycle_events: list[CoreEvent] = []
    previous_logical_time: str | None = None
    for event in core.events:
        if event.metadata.get("lifecycle") == "position_padding":
            padding_events.append(event)
        else:
            lifecycle_events.append(event)
        try:
            scope = ActionScope(event.metadata["action_scope"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Family E event requires a valid action scope") from exc
        logical_time = event.metadata.get("logical_time")
        effective_at = event.metadata.get("effective_at")
        if (
            type(logical_time) is not str
            or len(logical_time) != 8
            or not logical_time.isdecimal()
            or type(effective_at) is not str
            or len(effective_at) != 8
            or not effective_at.isdecimal()
            or (
                previous_logical_time is not None
                and logical_time <= previous_logical_time
            )
        ):
            raise ValueError("Family E events require strictly increasing canonical logical times")
        previous_logical_time = logical_time
        if event.operation is Operation.DELETE:
            if scope is not expected_delete_scope:
                raise ValueError("Family E deletion scope does not match its lifecycle profile")
        elif scope is not ActionScope.OBJECT:
            raise ValueError("Family E non-delete events require object scope")
        if scope is not ActionScope.TTL and effective_at != logical_time:
            raise ValueError("Family E effective logical time must equal its event logical time")
        if event.metadata.get("protected_collateral") is True:
            protected.update(_identity(key) for key in event.object_keys)
        if event.operation in {Operation.ADD, Operation.UPDATE}:
            for key in event.object_keys:
                identity = _identity(key)
                known[identity] = key
                state[identity] = event.value
        elif event.operation is Operation.DELETE:
            delete_events.append(event)
            selector = event.object_keys[0]
            exact_scope = {
                identity
                for identity, key in known.items()
                if identity in state and _scope_matches(scope, selector, key)
            }
            enumerated = {_identity(key) for key in event.object_keys}
            if enumerated & protected:
                raise ValueError("Family E DELETE must preserve protected collateral")
            if enumerated != exact_scope:
                raise ValueError("Family E DELETE must enumerate exact scope targets")
            for identity in enumerated:
                state.pop(identity, None)

    deletion_position = core.stratification.get("deletion_position")
    if deletion_position is None:
        if padding_events:
            raise ValueError("Family E position padding requires declared deletion position")
    elif cell == "correction_versus_deletion_hard_negative":
        if deletion_position != "not_applicable" or padding_events:
            raise ValueError("Family E correction hard negative deletion position is invalid")
    else:
        if deletion_position not in _FAMILY_E_POSITION_PADDING:
            raise ValueError("Family E deletion position is invalid")
        before_count, after_count = _FAMILY_E_POSITION_PADDING[deletion_position]
        first_lifecycle = next(
            index
            for index, event in enumerate(core.events)
            if event.metadata.get("lifecycle") != "position_padding"
        )
        last_lifecycle = max(
            index
            for index, event in enumerate(core.events)
            if event.metadata.get("lifecycle") != "position_padding"
        )
        if (
            first_lifecycle != before_count
            or len(core.events) - last_lifecycle - 1 != after_count
            or len(padding_events) != before_count + after_count
            or any(
                event.operation is not Operation.NOOP
                or event.role is not EventRole.NEUTRAL
                or event.object_keys
                or event.value is not None
                for event in padding_events
            )
        ):
            raise ValueError("Family E deletion position padding is invalid")

    shape = _FAMILY_E_CELL_SHAPES[(cell, expected_delete_scope)]
    observed_operations = tuple(event.operation for event in lifecycle_events)
    observed_roles = tuple(event.role for event in lifecycle_events)
    observed_lifecycles = tuple(
        str(event.metadata.get("lifecycle")) for event in lifecycle_events
    )
    query_identities = {_identity(key) for key in core.query_targets}
    delete_target_count = sum(len(event.object_keys) for event in delete_events)
    structural_shape = (
        observed_operations == shape.operations
        and observed_roles == shape.roles
        and observed_lifecycles == shape.lifecycles
        and len(query_identities) == shape.query_target_count
        and query_identities == set(known)
        and delete_target_count == shape.delete_target_count
        and len(known) == shape.unique_object_count
        and len(protected) == shape.protected_count
        and len(state) == shape.final_present_count
        and _peak_active_object_count(list(core.events))
        == shape.peak_active_object_count
    )
    derived_signature = ",".join(operation.value for operation in observed_operations)
    if (
        not structural_shape
        or core.stratification.get("operation_signature") != derived_signature
    ):
        raise ValueError("Family E lifecycle structural shape is invalid")
    if core.profile.get("active_object_count") != shape.peak_active_object_count:
        raise ValueError("Family E profile active_object_count is stale")
    if core.profile.get("context_length") != len(core.events):
        raise ValueError("Family E profile context_length is stale")

    if cell == "correction_versus_deletion_hard_negative":
        if delete_events or not any(event.operation is Operation.UPDATE for event in core.events):
            raise ValueError("correction hard negative must use UPDATE rather than DELETE")
    elif not delete_events:
        raise ValueError("Family E lifecycle cell requires DELETE")

    if cell == "logical_ttl_expiry":
        if len(delete_events) != 1:
            raise ValueError("Family E TTL cell requires exactly one TTL deletion")
        probes = [
            event
            for event in core.events
            if event.metadata.get("lifecycle") == "ttl_boundary_probe"
        ]
        if len(probes) != 1:
            raise ValueError("Family E TTL boundary requires exactly one probe")
        delete = delete_events[0]
        probe = probes[0]
        expiry = core.stratification.get("ttl_expiry_at")
        schedule_time = delete.metadata.get("logical_time")
        query_time = core.stratification.get("query_logical_time")
        probe_time = probe.metadata.get("logical_time")
        times = (expiry, schedule_time, query_time, probe_time)
        if any(type(value) is not str or len(value) != 8 or not value.isdecimal() for value in times):
            raise ValueError("Family E TTL boundary requires canonical logical times")
        if (
            delete.metadata.get("effective_at") != expiry
            or probe.metadata.get("effective_at") != expiry
            or probe_time != expiry
            or query_time != expiry
            or schedule_time >= expiry
        ):
            raise ValueError("Family E TTL boundary must equal the observable query boundary")
        if delete.metadata.get("action_scope") != ActionScope.TTL.value:
            raise ValueError("Family E TTL deletion requires ttl scope")

    if protected - set(state):
        raise ValueError("Family E protected collateral was deleted")

    if cell == "delete_then_relearn":
        forgotten = core.stratification.get("forgotten_value")
        relearn_events = [event for event in core.events if event.metadata.get("lifecycle") == "relearn"]
        if len(relearn_events) != 1 or relearn_events[0].value == forgotten:
            raise ValueError("Family E relearn must establish a new value")
        target = _identity(core.query_targets[0])
        if state.get(target) != core.expected_answer or state.get(target) == forgotten:
            raise ValueError("Family E forgotten value must not remain current after relearn")
        delete_position = next(index for index, event in enumerate(core.events) if event.operation is Operation.DELETE)
        if any(event.value == forgotten for event in core.events[delete_position + 1 :] if event.operation in {Operation.ADD, Operation.UPDATE}):
            raise ValueError("Family E forgotten value retention is forbidden")


def generate_core_family_e_cores(
    config: CoreConfig,
    *,
    profile: Literal["micro", "full"] = "micro",
) -> list[SemanticCore]:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    if profile == "micro":
        cores = [
            _build_cell_core(cell, example, cell_index * _EXAMPLES_PER_CELL + example)
            for cell_index, cell in enumerate(FAMILY_E_LIFECYCLE_CELLS)
            for example in range(_EXAMPLES_PER_CELL)
        ]
        expected_counts = Counter({cell: 3 for cell in FAMILY_E_LIFECYCLE_CELLS})
        expected_total = 24
    elif profile == "full":
        schedule = config.families.deletion_forgetting.schedule
        per_cell = schedule.cores_per_lifecycle_cell
        difficulties = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        positions = tuple(config.families.deletion_forgetting.deletion_positions)
        cores = []
        for cell_index, cell in enumerate(FAMILY_E_LIFECYCLE_CELLS):
            permitted_scopes = FAMILY_E_DELETE_SCOPES_BY_CELL[cell]
            for example in range(per_cell):
                position = (
                    "not_applicable"
                    if cell == "correction_versus_deletion_hard_negative"
                    else positions[
                        example // (per_cell // len(positions))
                    ]
                )
                cores.append(
                    _build_cell_core(
                        cell,
                        example,
                        cell_index * per_cell + example,
                        difficulty=difficulties[example % len(difficulties)],
                        deletion_position=position,
                        scope_override=permitted_scopes[example % len(permitted_scopes)],
                    )
                )
        expected_counts = Counter(
            {cell: per_cell for cell in FAMILY_E_LIFECYCLE_CELLS}
        )
        expected_total = config.families.deletion_forgetting.semantic_core_count
    else:
        raise ValueError("Family E profile must be micro or full")

    for core in cores:
        validate_family_e_core(core)
    counts = Counter(core.stratification["lifecycle_cell"] for core in cores)
    if len(cores) != expected_total or counts != expected_counts:
        raise ValueError(f"Family E {profile} schedule counts are invalid")
    return cores


def _validate_compiled(tasks: list[MemUpdateTaskV3], cores: list[SemanticCore]) -> None:
    if len(cores) != 24:
        raise ValueError("Family E micro-pilot requires exactly 24 semantic cores")
    if len(tasks) != 96:
        raise ValueError("Family E micro-pilot requires exactly 96 tasks")
    by_core: dict[str, list[MemUpdateTaskV3]] = {}
    for task in tasks:
        by_core.setdefault(task.metadata.split_key.semantic_core_id, []).append(task)
    if set(by_core) != {core.core_id for core in cores}:
        raise ValueError("Family E compiled tasks do not cover exact semantic cores")
    for variants in by_core.values():
        surface_ids = {task.metadata.extra["surface_variant"] for task in variants}
        if len(variants) != 4 or surface_ids != {0, 1, 2, 3}:
            raise ValueError("Family E core requires four surface variants")
        if len({task.semantic_hash for task in variants}) != 1:
            raise ValueError("Family E four surface variants are not semantically equivalent")
        for task in variants:
            from mub.vnext.validation.replay_v3 import replay_task_v3

            if replay_task_v3(task).issues:
                raise ValueError("Family E rendered task failed exact v3 replay")


def compile_family_e_micro_pilot(
    config: CoreConfig,
    *,
    code_revision: str,
) -> CompiledFamilyEMicroPilot:
    cores = generate_core_family_e_cores(config)
    if len(cores) != 24:
        raise ValueError("Family E micro-pilot requires exactly 24 semantic cores")
    for core in cores:
        validate_family_e_core(core)
    context = GenerationContext(
        config=config,
        code_revision=code_revision,
        generator_name="memupdatebench_vnext_core_family_e_micro",
    )
    tasks = [
        render_core_v3(
            core,
            split=Split.TEST,
            surface_variant=surface_variant,
            context=context,
        )
        for core in cores
        for surface_variant in range(4)
    ]
    _validate_compiled(tasks, cores)
    return CompiledFamilyEMicroPilot(
        profile_id=FAMILY_E_MICRO_PROFILE_ID,
        cores=tuple(cores),
        tasks=tuple(tasks),
    )


__all__ = [
    "CompiledFamilyEMicroPilot",
    "FAMILY_E_DELETE_SCOPES_BY_CELL",
    "FAMILY_E_LIFECYCLE_CELLS",
    "FAMILY_E_MICRO_PROFILE_ID",
    "compile_family_e_micro_pilot",
    "generate_core_family_e_cores",
    "validate_family_e_core",
]
