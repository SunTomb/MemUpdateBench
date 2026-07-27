from __future__ import annotations

from itertools import product

from mub.vnext.contracts import Difficulty, EventRole, MemoryObjectKey, Operation, TaskFamily
from mub.vnext.generation.catalogs import (
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    VALUES,
    select_conflicting_values,
)
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id


_FAMILY_NAME = TaskFamily.REPEATED_SAME_SLOT.value
_DEPTHS = (1, 4, 16)
_DIFFICULTIES = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_AMBIGUITY = {
    Difficulty.EASY: ("none", "none"),
    Difficulty.MEDIUM: ("moderate", "moderate"),
    Difficulty.HARD: ("high", "high"),
}
_VERSION_METADATA = {
    Difficulty.EASY: "latest_outdated",
    Difficulty.MEDIUM: "event_index",
    Difficulty.HARD: "none",
}


def _canonical_axis_order(config: PilotConfig) -> tuple[tuple[str, str, str], ...]:
    candidates = tuple(product(NAMESPACES, RELATION_QUALIFIED_ENTITIES, CANONICAL_ATTRIBUTES))
    return tuple(
        sorted(
            candidates,
            key=lambda axis: stable_id(
                "family_a_axis",
                {"seed": config.seed, "namespace": axis[0], "entity": axis[1], "attribute": axis[2]},
            ),
        )
    )


def _validate_config(config: PilotConfig) -> None:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if config.cores_per_family != 120:
        raise ValueError("Family A requires cores_per_family=120")
    family = config.families.repeated_same_slot_update
    if not family.enabled:
        raise ValueError("Family A must be enabled")
    if tuple(sorted(family.update_depths)) != _DEPTHS:
        raise ValueError("Family A update_depths must be exactly [1, 4, 16]")
    if set(family.difficulties) != set(_DIFFICULTIES):
        raise ValueError("Family A difficulties must include easy, medium, and hard")


def _key(namespace: str, entity: str, attribute: str) -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=None,
    )


def _same_name_keys(target: MemoryObjectKey, count: int, axis_index: int) -> tuple[MemoryObjectKey, ...]:
    group = next(group for group in SAME_NAME_ENTITIES if target.entity in group)
    alternatives = tuple(entity for entity in group if entity != target.entity)
    return tuple(
        _key(
            NAMESPACES[(axis_index + offset + 1) % len(NAMESPACES)],
            alternatives[offset % len(alternatives)],
            target.attribute,
        )
        for offset in range(count)
    )


def _other_attribute_keys(target: MemoryObjectKey, count: int) -> tuple[MemoryObjectKey, ...]:
    alternatives = tuple(attribute for attribute in CANONICAL_ATTRIBUTES if attribute != target.attribute)
    return tuple(_key(target.namespace, target.entity, alternatives[index]) for index in range(count))


def _value_for_axis(config: PilotConfig, axis_index: int) -> str:
    return VALUES[(config.seed + axis_index) % len(VALUES)]


def _target_events(
    config: PilotConfig,
    core_index: int,
    axis_index: int,
    target: MemoryObjectKey,
    depth: int,
) -> tuple[CoreEvent, ...]:
    final_value = _value_for_axis(config, axis_index)
    stale_values = select_conflicting_values(
        VALUES,
        final_value,
        depth,
        {
            "family": _FAMILY_NAME,
            "seed": config.seed,
            "core_index": core_index,
            "axis_index": axis_index,
            "target": target.canonical_id,
        },
    )
    versions = (*stale_values, final_value)
    events = [
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[target],
            value=versions[0],
            role=EventRole.STALE_SAME_SLOT,
            metadata={"version_index": 0, "version_metadata": "stale"},
        )
    ]
    for version_index, value in enumerate(versions[1:], start=1):
        is_latest = version_index == len(versions) - 1
        events.append(
            CoreEvent(
                operation=Operation.UPDATE,
                object_keys=[target],
                value=value,
                role=EventRole.LATEST_GOLD if is_latest else EventRole.STALE_SAME_SLOT,
                metadata={
                    "version_index": version_index,
                    "version_metadata": "latest" if is_latest else "stale",
                },
            )
        )
    return tuple(events)


def _distractor_events(
    config: PilotConfig,
    core_index: int,
    axis_index: int,
    target: MemoryObjectKey,
    difficulty: Difficulty,
) -> tuple[CoreEvent, ...]:
    family = config.families.repeated_same_slot_update
    configured_same_name = getattr(family.same_name_distractors, difficulty.value)
    configured_other_attribute = getattr(family.same_entity_other_attribute, difficulty.value)
    same_name_keys = _same_name_keys(target, configured_same_name, axis_index)
    other_attribute_keys = _other_attribute_keys(target, configured_other_attribute)
    all_keys = (*same_name_keys, *other_attribute_keys)
    values = select_conflicting_values(
        VALUES,
        _value_for_axis(config, axis_index),
        len(all_keys),
        {
            "family": _FAMILY_NAME,
            "seed": config.seed,
            "core_index": core_index,
            "axis_index": axis_index,
            "target": target.canonical_id,
            "role": "distractor",
        },
    )
    events: list[CoreEvent] = []
    distractor_specs = tuple(
        (key, EventRole.SAME_NAME_OTHER_ENTITY, "same_name")
        for key in same_name_keys
    ) + tuple(
        (key, EventRole.SAME_ENTITY_OTHER_ATTRIBUTE, "other_attribute")
        for key in other_attribute_keys
    )
    for index, ((key, role, kind), value) in enumerate(zip(distractor_specs, values)):
        events.append(
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[key],
                value=value,
                role=role,
                metadata={"distractor_index": index, "distractor_kind": kind},
            )
        )
    return tuple(events)


def _noop_events(config: PilotConfig, target: MemoryObjectKey, difficulty: Difficulty) -> tuple[CoreEvent, ...]:
    count = getattr(config.families.repeated_same_slot_update.noop_near_miss, difficulty.value)
    return tuple(
        CoreEvent(
            operation=Operation.NOOP,
            object_keys=[],
            value=None,
            role=EventRole.NOOP_NEAR_MISS,
            metadata={
                "surface_statement": (
                    f"Near miss {index + 1}: the record mentions "
                    f"{target.entity} without changing memory."
                ),
                "near_miss_index": index,
            },
        )
        for index in range(count)
    )


def _build_core(
    config: PilotConfig,
    core_index: int,
    axis_index: int,
    axis: tuple[str, str, str],
    depth: int,
    difficulty: Difficulty,
    axis_product_size: int,
    depth_allocation_count: int,
    difficulty_allocation_count: int,
    cell_allocation_count: int,
) -> SemanticCore:
    target = _key(*axis)
    target_events = _target_events(config, core_index, axis_index, target, depth)
    distractor_events = _distractor_events(config, core_index, axis_index, target, difficulty)
    noop_events = _noop_events(config, target, difficulty)
    events = (*target_events, *distractor_events, *noop_events)
    same_name_count = sum(event.role is EventRole.SAME_NAME_OTHER_ENTITY for event in events)
    other_attribute_count = sum(event.role is EventRole.SAME_ENTITY_OTHER_ATTRIBUTE for event in events)
    noop_count = sum(event.operation is Operation.NOOP for event in events)
    stale_count = sum(event.role is EventRole.STALE_SAME_SLOT for event in events)
    num_events = len(events)
    entity_ambiguity, attribute_ambiguity = _AMBIGUITY[difficulty]
    final_value = target_events[-1].value
    semantic_payload = {
        "family": _FAMILY_NAME,
        "seed": config.seed,
        "core_index": core_index,
        "axis_index": axis_index,
        "target": {"namespace": target.namespace, "entity": target.entity, "attribute": target.attribute, "subkey": target.subkey},
        "update_depth": depth,
        "difficulty": difficulty.value,
        "same_name_distractor_count": same_name_count,
        "same_entity_other_attribute_count": other_attribute_count,
        "noop_count": noop_count,
        "target_values": [event.value for event in target_events],
    }
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    profile = {
        "update_depth": depth,
        "stale_count": stale_count,
        "active_object_count": 1 + same_name_count + other_attribute_count,
        "noop_density": noop_count / num_events,
        "entity_ambiguity": entity_ambiguity,
        "attribute_ambiguity": attribute_ambiguity,
        "context_length": num_events,
        "query_type": "current_state",
        "version_metadata": _VERSION_METADATA[difficulty],
    }
    stratification = {
        "num_events": num_events,
        "num_target_updates": depth,
        "same_name_distractor_count": same_name_count,
        "same_entity_other_attribute_count": other_attribute_count,
        "noop_count": noop_count,
        "stale_same_slot_count": stale_count,
        "stale_count": stale_count,
        "axis_product_index": axis_index,
        "axis_product_size": axis_product_size,
        "depth_allocation_count": depth_allocation_count,
        "difficulty_allocation_count": difficulty_allocation_count,
        "depth_difficulty_cell_count": cell_allocation_count,
    }
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=difficulty,
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, f"family_a_{core_index:03d}"),
        events=list(events),
        query_targets=[target],
        expected_answer=final_value,
        profile=profile,
        stratification=stratification,
    )


def _allocation_counts(core_count: int) -> tuple[dict[int, int], dict[Difficulty, int], dict[tuple[int, Difficulty], int]]:
    depth_counts = {depth: 0 for depth in _DEPTHS}
    difficulty_counts = {difficulty: 0 for difficulty in _DIFFICULTIES}
    cell_counts = {
        (depth, difficulty): 0
        for depth in _DEPTHS
        for difficulty in _DIFFICULTIES
    }
    for core_index in range(core_count):
        depth = _DEPTHS[core_index % len(_DEPTHS)]
        difficulty = _DIFFICULTIES[(core_index // len(_DEPTHS)) % len(_DIFFICULTIES)]
        depth_counts[depth] += 1
        difficulty_counts[difficulty] += 1
        cell_counts[(depth, difficulty)] += 1
    return depth_counts, difficulty_counts, cell_counts


def generate_family_a_cores(config: PilotConfig) -> list[SemanticCore]:
    """Generate the deterministic 120-core repeated same-slot Family A."""
    _validate_config(config)
    axes = _canonical_axis_order(config)
    depth_counts, difficulty_counts, cell_counts = _allocation_counts(config.cores_per_family)
    cores: list[SemanticCore] = []
    for core_index in range(config.cores_per_family):
        depth = _DEPTHS[core_index % len(_DEPTHS)]
        difficulty = _DIFFICULTIES[(core_index // len(_DEPTHS)) % len(_DIFFICULTIES)]
        axis_index = core_index % len(axes)
        cores.append(
            _build_core(
                config,
                core_index,
                axis_index,
                axes[axis_index],
                depth,
                difficulty,
                len(axes),
                depth_counts[depth],
                difficulty_counts[difficulty],
                cell_counts[(depth, difficulty)],
            )
        )
    return cores


__all__ = ["generate_family_a_cores"]
