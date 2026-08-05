from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts.common import MemoryObjectKey
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
FAMILY_E_MICRO_PROFILE_ID = "family_e_diagnostic_micro_v1"
_EXAMPLES_PER_CELL = 3


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


def _build_cell_core(cell: str, example: int, core_index: int) -> SemanticCore:
    namespace = f"family_e_ns_{core_index:02d}"
    entity = f"synthetic_subject_{example}"
    primary = _key(namespace, entity, "contact_channel")
    old = f"synthetic_value_{core_index:02d}_old"
    new = f"synthetic_value_{core_index:02d}_new"
    query_targets: list[MemoryObjectKey]
    expected_answer: Any
    relearning = "none"
    scope = ActionScope.OBJECT

    if cell == "explicit_object_or_attribute_deletion":
        if example == 0:
            targets = [primary]
            scope = ActionScope.OBJECT
        else:
            targets = [
                _key(namespace, entity, "contact_channel", "primary"),
                _key(namespace, entity, "contact_channel", "backup"),
            ]
            scope = ActionScope.ATTRIBUTE
        events = [
            *[_event(Operation.ADD, [key], f"synthetic_value_{core_index:02d}_{i}", "seed", 10 + i) for i, key in enumerate(targets)],
            _event(Operation.DELETE, targets, None, "explicit_delete", 20, scope=scope),
        ]
        query_targets = targets
        expected_answer = None
    elif cell == "entity_wide_deletion":
        targets = [primary, _key(namespace, entity, "preferred_locale")]
        scope = ActionScope.ENTITY
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
        scope = ActionScope.NAMESPACE
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
            _event(Operation.DELETE, [primary], None, "ttl_schedule", 11, scope=ActionScope.TTL, effective_at=expiry),
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
        scope = ActionScope.TTL
    elif cell == "post_delete_similar_retrieval":
        similar = _key(namespace, f"synthetic_subject_{example}_similar", "contact_channel")
        events = [
            _event(Operation.ADD, [primary], old, "seed", 10),
            _event(Operation.ADD, [similar], new, "similar_protected", 11, protected=True),
            _event(Operation.DELETE, [primary], None, "delete_before_similar_probe", 20),
        ]
        query_targets = [primary, similar]
        expected_answer = _mixed_absence_answer(query_targets, {primary.canonical_id})
    elif cell == "delete_then_relearn":
        events = [
            _event(Operation.ADD, [primary], old, "seed", 10),
            _event(Operation.DELETE, [primary], None, "delete", 20),
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
        scope = ActionScope.ATTRIBUTE
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

    payload = {
        "family": TaskFamily.DELETION_FORGETTING.value,
        "cell": cell,
        "example": example,
        "events": [event.model_dump(mode="json") for event in events],
        "query_targets": [_key_payload(key) for key in query_targets],
    }
    identifier = core_id(TaskFamily.DELETION_FORGETTING.value, payload)
    stratification: dict[str, str | int | float | bool] = {
        "lifecycle_cell": cell,
        "cell_example": example,
        "deletion_scope": scope.value,
        "relearning_condition": relearning,
        "operation_signature": ",".join(event.operation.value for event in events),
    }
    if cell == "logical_ttl_expiry":
        expiry_at = next(event.metadata["effective_at"] for event in events if event.operation is Operation.DELETE)
        stratification["ttl_expiry_at"] = expiry_at
        stratification["query_logical_time"] = expiry_at
    if cell == "delete_then_relearn":
        stratification["forgotten_value"] = old
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.DELETION_FORGETTING,
        difficulty=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[example],
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
    if core.profile.get("deletion_scope") != core.stratification.get("deletion_scope"):
        raise ValueError("Family E deletion scope profile mismatch")
    if core.profile.get("relearning_condition") != core.stratification.get("relearning_condition"):
        raise ValueError("Family E relearning profile mismatch")

    state: dict[tuple[str, str, str, str | None], Any] = {}
    known: dict[tuple[str, str, str, str | None], MemoryObjectKey] = {}
    protected: set[tuple[str, str, str, str | None]] = set()
    delete_events: list[CoreEvent] = []
    for event in core.events:
        try:
            scope = ActionScope(event.metadata["action_scope"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Family E event requires a valid action scope") from exc
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


def generate_core_family_e_cores(config: CoreConfig) -> list[SemanticCore]:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    cores = [
        _build_cell_core(cell, example, cell_index * _EXAMPLES_PER_CELL + example)
        for cell_index, cell in enumerate(FAMILY_E_LIFECYCLE_CELLS)
        for example in range(_EXAMPLES_PER_CELL)
    ]
    for core in cores:
        validate_family_e_core(core)
    counts = Counter(core.stratification["lifecycle_cell"] for core in cores)
    if len(cores) != 24 or counts != Counter({cell: 3 for cell in FAMILY_E_LIFECYCLE_CELLS}):
        raise ValueError("Family E micro-pilot requires exactly 24 cores and three per lifecycle cell")
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
    "FAMILY_E_LIFECYCLE_CELLS",
    "FAMILY_E_MICRO_PROFILE_ID",
    "compile_family_e_micro_pilot",
    "generate_core_family_e_cores",
    "validate_family_e_core",
]
