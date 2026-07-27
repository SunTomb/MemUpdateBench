from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from itertools import product

from mub.vnext.contracts import Difficulty, EventRole, MemoryObjectKey, Operation, TaskFamily
from mub.vnext.generation.catalogs import (
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    VALUES,
    select_conflicting_values,
)
from mub.vnext.generation.config import InterleavedMultiSlotUpdateConfig, PilotConfig
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id


_FAMILY_NAME = TaskFamily.INTERLEAVED_MULTI_SLOT.value
_DEPTHS = (1, 4, 16)
_DIFFICULTIES = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_PATTERNS = ("round_robin", "burst", "adversarial_adjacent")
_VERSION_METADATA = {
    Difficulty.EASY: "latest_outdated",
    Difficulty.MEDIUM: "event_index",
    Difficulty.HARD: "none",
}


def _validate_config(config: PilotConfig) -> InterleavedMultiSlotUpdateConfig:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if config.cores_per_family != 120:
        raise ValueError("Family B requires cores_per_family=120")
    family = config.families.interleaved_multi_slot_update
    if not family.enabled:
        raise ValueError("Family B must be enabled")
    if tuple(sorted(family.update_depths)) != _DEPTHS:
        raise ValueError("Family B update_depths must be exactly [1, 4, 16]")
    if len(family.difficulties) != len(_DIFFICULTIES) or set(
        family.difficulties
    ) != set(_DIFFICULTIES):
        raise ValueError(
            "Family B difficulties must include easy, medium, and hard exactly once"
        )
    if len(family.interleaving_patterns) != len(_PATTERNS) or set(
        family.interleaving_patterns
    ) != set(_PATTERNS):
        raise ValueError(
            "Family B interleaving_patterns must include round_robin, burst, and "
            "adversarial_adjacent"
        )
    expected_active_counts = {Difficulty.EASY: 2, Difficulty.MEDIUM: 4, Difficulty.HARD: 8}
    expected_densities = {Difficulty.EASY: 0.0, Difficulty.MEDIUM: 0.25, Difficulty.HARD: 0.5}
    for difficulty in _DIFFICULTIES:
        if getattr(family.active_object_counts, difficulty.value) != expected_active_counts[difficulty]:
            raise ValueError("Family B active_object_counts must be easy=2, medium=4, hard=8")
        if getattr(family.cross_slot_distractor_density, difficulty.value) != expected_densities[difficulty]:
            raise ValueError(
                "Family B cross_slot_distractor_density must be easy=0.0, "
                "medium=0.25, hard=0.50"
            )
    return family


def _key(namespace: str, entity: str, attribute: str) -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=None,
    )


def _canonical_axis_order(config: PilotConfig) -> tuple[tuple[str, str, str], ...]:
    candidates = tuple(
        product(NAMESPACES, RELATION_QUALIFIED_ENTITIES, CANONICAL_ATTRIBUTES)
    )
    return tuple(
        sorted(
            candidates,
            key=lambda axis: stable_id(
                "family_b_axis",
                {
                    "seed": config.seed,
                    "namespace": axis[0],
                    "entity": axis[1],
                    "attribute": axis[2],
                },
            ),
        )
    )


def _active_keys(
    axis: tuple[str, str, str],
    active_object_count: int,
) -> tuple[MemoryObjectKey, ...]:
    namespace, entity, target_attribute = axis
    other_attributes = tuple(
        attribute
        for attribute in CANONICAL_ATTRIBUTES
        if attribute != target_attribute
    )
    if active_object_count - 1 > len(other_attributes):
        raise ValueError("insufficient canonical attributes for distinct active slots")
    return (
        _key(namespace, entity, target_attribute),
        *(
            _key(namespace, entity, attribute)
            for attribute in other_attributes[: active_object_count - 1]
        ),
    )


def _ordered_final_values(
    config: PilotConfig,
    group_index: int,
    active_object_count: int,
) -> tuple[str, ...]:
    ordered = tuple(
        sorted(
            VALUES,
            key=lambda value: stable_id(
                "family_b_final_value",
                {"seed": config.seed, "group_index": group_index, "value": value},
            ),
        )
    )
    return ordered[:active_object_count]


def _target_trajectory(
    config: PilotConfig,
    group_index: int,
    target: MemoryObjectKey,
    depth: int,
    final_value: str,
) -> tuple[CoreEvent, ...]:
    stale_values = select_conflicting_values(
        VALUES,
        final_value,
        depth,
        {
            "family": _FAMILY_NAME,
            "seed": config.seed,
            "group_index": group_index,
            "target": target.canonical_id,
            "role": "target",
        },
    )
    versions = (*stale_values, final_value)
    events = []
    for version_index, value in enumerate(versions):
        is_latest = version_index == len(versions) - 1
        events.append(
            CoreEvent(
                operation=Operation.ADD if version_index == 0 else Operation.UPDATE,
                object_keys=[target],
                value=value,
                role=EventRole.LATEST_GOLD if is_latest else EventRole.STALE_SAME_SLOT,
                metadata={
                    "slot_index": 0,
                    "version_index": version_index,
                    "version_metadata": "latest" if is_latest else "stale",
                    "target_relation": "target",
                },
            )
        )
    return tuple(events)


def _cross_slot_distractor_count(base_event_count: int, density: float) -> int:
    scaled = Decimal(base_event_count) * Decimal(str(density))
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _distribute_updates(count: int, slot_count: int) -> tuple[int, ...]:
    allocations = [0] * slot_count
    for index in range(count):
        allocations[index % slot_count] += 1
    return tuple(allocations)


def _non_target_trajectory(
    config: PilotConfig,
    group_index: int,
    slot_index: int,
    key: MemoryObjectKey,
    update_count: int,
    final_value: str,
) -> tuple[CoreEvent, ...]:
    stale_values = select_conflicting_values(
        VALUES,
        final_value,
        update_count,
        {
            "family": _FAMILY_NAME,
            "seed": config.seed,
            "group_index": group_index,
            "slot_index": slot_index,
            "target": key.canonical_id,
            "role": "cross_slot",
        },
    )
    versions = (*stale_values, final_value)
    return tuple(
        CoreEvent(
            operation=Operation.ADD if version_index == 0 else Operation.UPDATE,
            object_keys=[key],
            value=value,
            role=EventRole.SAME_ENTITY_OTHER_ATTRIBUTE,
            metadata={
                "slot_index": slot_index,
                "version_index": version_index,
                "version_metadata": (
                    "latest" if version_index == len(versions) - 1 else "stale"
                ),
                "distractor_kind": "cross_slot",
                "target_relation": "same_entity_other_attribute",
            },
        )
        for version_index, value in enumerate(versions)
    )


def _build_trajectories(
    config: PilotConfig,
    group_index: int,
    axis: tuple[str, str, str],
    depth: int,
    active_object_count: int,
    cross_slot_distractor_count: int,
) -> tuple[tuple[CoreEvent, ...], ...]:
    keys = _active_keys(axis, active_object_count)
    final_values = _ordered_final_values(config, group_index, active_object_count)
    target = _target_trajectory(
        config,
        group_index,
        keys[0],
        depth,
        final_values[0],
    )
    allocations = _distribute_updates(
        cross_slot_distractor_count,
        active_object_count - 1,
    )
    non_targets = tuple(
        _non_target_trajectory(
            config,
            group_index,
            slot_index,
            key,
            allocations[slot_index - 1],
            final_values[slot_index],
        )
        for slot_index, key in enumerate(keys[1:], start=1)
    )
    return (target, *non_targets)


def _interleave(
    trajectories: tuple[tuple[CoreEvent, ...], ...],
    pattern: str,
) -> tuple[CoreEvent, ...]:
    target = trajectories[0]
    non_targets = trajectories[1:]
    if pattern == "burst":
        return tuple(event for trajectory in trajectories for event in trajectory)
    if pattern == "round_robin":
        ordered_trajectories = (*non_targets, target)
        max_length = max(len(trajectory) for trajectory in ordered_trajectories)
        return tuple(
            trajectory[version_index]
            for version_index in range(max_length)
            for trajectory in ordered_trajectories
            if version_index < len(trajectory)
        )
    if pattern == "adversarial_adjacent":
        distractors = tuple(
            event for trajectory in non_targets for event in trajectory
        )
        return (*target[:-1], *distractors, target[-1])
    raise ValueError(f"unsupported interleaving pattern: {pattern}")


def _allocation_counts(
    core_count: int,
    depths: tuple[int, ...],
    difficulties: tuple[Difficulty, ...],
    patterns: tuple[str, ...],
) -> Counter[tuple[int, Difficulty, str]]:
    counts: Counter[tuple[int, Difficulty, str]] = Counter()
    for core_index in range(core_count):
        group_index = core_index // len(patterns)
        depth = depths[group_index % len(depths)]
        difficulty = difficulties[
            (group_index // len(depths)) % len(difficulties)
        ]
        counts[(depth, difficulty, patterns[core_index % len(patterns)])] += 1
    return counts


def _build_core(
    config: PilotConfig,
    core_index: int,
    group_index: int,
    axis_index: int,
    axis_product_size: int,
    depth: int,
    difficulty: Difficulty,
    pattern: str,
    active_object_count: int,
    density: float,
    base_event_count: int,
    distractor_count: int,
    trajectories: tuple[tuple[CoreEvent, ...], ...],
    allocation_cell_count: int,
) -> SemanticCore:
    events = _interleave(trajectories, pattern)
    target = trajectories[0][0].object_keys[0]
    final_value = trajectories[0][-1].value
    num_events = len(events)
    semantic_payload = {
        "family": _FAMILY_NAME,
        "seed": config.seed,
        "core_index": core_index,
        "group_index": group_index,
        "axis_index": axis_index,
        "target": {
            "namespace": target.namespace,
            "entity": target.entity,
            "attribute": target.attribute,
            "subkey": target.subkey,
        },
        "update_depth": depth,
        "difficulty": difficulty.value,
        "active_object_count": active_object_count,
        "cross_slot_distractor_count": distractor_count,
        "interleaving_pattern": pattern,
        "events": [
            {
                "operation": event.operation.value,
                "object_keys": [key.canonical_id for key in event.object_keys],
                "value": event.value,
                "role": event.role.value,
            }
            for event in events
        ],
    }
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    profile = {
        "update_depth": depth,
        "stale_count": depth,
        "active_object_count": active_object_count,
        "noop_density": 0.0,
        "cross_slot_interleaving": density,
        "context_length": num_events,
        "query_type": "current_state",
        "version_metadata": _VERSION_METADATA[difficulty],
    }
    stratification = {
        "num_events": num_events,
        "num_target_updates": depth,
        "active_object_count": active_object_count,
        "cross_slot_distractor_count": distractor_count,
        "cross_slot_distractor_density": density,
        "realized_cross_slot_distractor_density": distractor_count / base_event_count,
        "base_event_count": base_event_count,
        "interleaving_pattern": pattern,
        "update_depth": depth,
        "stale_count": depth,
        "noop_count": 0,
        "axis_product_index": axis_index,
        "axis_product_size": axis_product_size,
        "pattern_group_index": group_index,
        "allocation_cell_count": allocation_cell_count,
    }
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.INTERLEAVED_MULTI_SLOT,
        difficulty=difficulty,
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, f"family_b_{core_index:03d}"),
        events=list(events),
        query_targets=[target],
        expected_answer=final_value,
        profile=profile,
        stratification=stratification,
    )


def generate_family_b_cores(config: PilotConfig) -> list[SemanticCore]:
    """Generate the deterministic 120-core interleaved multi-slot Family B."""
    family = _validate_config(config)
    depths = tuple(family.update_depths)
    difficulties = tuple(family.difficulties)
    patterns = tuple(family.interleaving_patterns)
    axes = _canonical_axis_order(config)
    allocation_counts = _allocation_counts(
        config.cores_per_family,
        depths,
        difficulties,
        patterns,
    )
    cores = []
    trajectory_cache: dict[
        int,
        tuple[int, int, tuple[tuple[CoreEvent, ...], ...]],
    ] = {}
    for core_index in range(config.cores_per_family):
        group_index = core_index // len(patterns)
        depth = depths[group_index % len(depths)]
        difficulty = difficulties[
            (group_index // len(depths)) % len(difficulties)
        ]
        pattern = patterns[core_index % len(patterns)]
        axis_index = group_index % len(axes)
        active_object_count = getattr(
            family.active_object_counts,
            difficulty.value,
        )
        density = getattr(
            family.cross_slot_distractor_density,
            difficulty.value,
        )
        cached = trajectory_cache.get(group_index)
        if cached is None:
            base_event_count = active_object_count + depth
            distractor_count = _cross_slot_distractor_count(
                base_event_count,
                density,
            )
            trajectories = _build_trajectories(
                config,
                group_index,
                axes[axis_index],
                depth,
                active_object_count,
                distractor_count,
            )
            cached = (base_event_count, distractor_count, trajectories)
            trajectory_cache[group_index] = cached
        base_event_count, distractor_count, trajectories = cached
        cores.append(
            _build_core(
                config,
                core_index,
                group_index,
                axis_index,
                len(axes),
                depth,
                difficulty,
                pattern,
                active_object_count,
                density,
                base_event_count,
                distractor_count,
                trajectories,
                allocation_counts[(depth, difficulty, pattern)],
            )
        )
    return cores


__all__ = ["generate_family_b_cores"]
