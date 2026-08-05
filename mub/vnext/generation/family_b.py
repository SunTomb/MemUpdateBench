from __future__ import annotations

from collections import Counter
from itertools import product

from mub.vnext.contracts import Difficulty, EventRole, MemoryObjectKey, Operation, TaskFamily
from mub.vnext.generation.catalogs import (
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    select_conflicting_values,
    values_for_attribute,
)
from mub.vnext.generation.config import InterleavedMultiSlotUpdateConfig, PilotConfig
from mub.vnext.generation.core_config import (
    CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS,
    CORE_FAMILY_B_DEPTHS,
    CORE_FAMILY_B_INTERLEAVING_PATTERNS,
    CoreConfig,
)
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id
from mub.vnext.generation.family_b_schedule import (
    INTERLEAVING_PATTERNS,
    canonical_cross_slot_update_count,
    canonical_interleaving_schedule,
)


_FAMILY_NAME = TaskFamily.INTERLEAVED_MULTI_SLOT.value
_DEPTHS = (1, 4, 16)
_DIFFICULTIES = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_PATTERNS = INTERLEAVING_PATTERNS


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
    keys: tuple[MemoryObjectKey, ...],
) -> tuple[str, ...]:
    return tuple(
        min(
            values_for_attribute(key.attribute),
            key=lambda value: stable_id(
                "family_b_final_value",
                {
                    "seed": config.seed,
                    "group_index": group_index,
                    "target": key.canonical_id,
                    "value": value,
                },
            ),
        )
        for key in keys
    )


def _target_trajectory(
    config: PilotConfig,
    group_index: int,
    target: MemoryObjectKey,
    depth: int,
    final_value: str,
) -> tuple[CoreEvent, ...]:
    stale_values = select_conflicting_values(
        values_for_attribute(target.attribute),
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
        values_for_attribute(key.attribute),
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
                "distractor_kind": (
                    "active_non_target" if version_index == 0 else "cross_slot"
                ),
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
    final_values = _ordered_final_values(config, group_index, keys)
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
    schedule = canonical_interleaving_schedule(
        tuple(len(trajectory) for trajectory in trajectories),
        pattern,
    )
    return tuple(
        trajectories[slot_index][version_index]
        for slot_index, version_index in schedule
    )


def _balanced_group_cells(
    group_count: int,
    depths: tuple[int, ...],
    difficulties: tuple[Difficulty, ...],
) -> tuple[tuple[int, Difficulty], ...]:
    cells = tuple(product(depths, difficulties))
    repeats, remainder = divmod(group_count, len(cells))
    assignments = list(cells) * repeats
    cell_counts = Counter(assignments)
    depth_counts = Counter(depth for depth, _ in assignments)
    difficulty_counts = Counter(difficulty for _, difficulty in assignments)
    depth_base, depth_remainder = divmod(group_count, len(depths))
    difficulty_base, difficulty_remainder = divmod(
        group_count,
        len(difficulties),
    )
    depth_targets = {
        depth: depth_base + (index < depth_remainder)
        for index, depth in enumerate(depths)
    }
    difficulty_targets = {
        difficulty: difficulty_base + (index < difficulty_remainder)
        for index, difficulty in enumerate(difficulties)
    }
    depth_order = {depth: index for index, depth in enumerate(depths)}
    difficulty_order = {
        difficulty: index for index, difficulty in enumerate(difficulties)
    }
    for _ in range(remainder):
        candidates = tuple(
            cell
            for cell in cells
            if depth_counts[cell[0]] < depth_targets[cell[0]]
            and difficulty_counts[cell[1]] < difficulty_targets[cell[1]]
        )
        if not candidates:
            raise ValueError("unable to balance Family B allocation cells")
        depth, difficulty = min(
            candidates,
            key=lambda cell: (
                cell_counts[cell],
                -(depth_targets[cell[0]] - depth_counts[cell[0]]),
                -(
                    difficulty_targets[cell[1]]
                    - difficulty_counts[cell[1]]
                ),
                depth_order[cell[0]],
                difficulty_order[cell[1]],
            ),
        )
        assignments.append((depth, difficulty))
        cell_counts[(depth, difficulty)] += 1
        depth_counts[depth] += 1
        difficulty_counts[difficulty] += 1
    return tuple(assignments)


def _allocation_counts(
    group_cells: tuple[tuple[int, Difficulty], ...],
    patterns: tuple[str, ...],
) -> Counter[tuple[int, Difficulty, str]]:
    return Counter(
        (depth, difficulty, pattern)
        for depth, difficulty in group_cells
        for pattern in patterns
    )


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
    allocation_cell_ideal: float,
    difficulty_allocation_count: int,
    difficulty_allocation_ideal: float,
    semantic_profile: str | None = None,
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
    if semantic_profile is not None:
        semantic_payload["profile"] = semantic_profile
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    profile = {
        "update_depth": depth,
        "stale_count": depth,
        "active_object_count": active_object_count,
        "noop_density": 0.0,
        "cross_slot_interleaving": density,
        "context_length": num_events,
        "context_order": "chronological",
        "query_type": "current_state",
        "version_metadata": "event_index",
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
        "allocation_cell_ideal": allocation_cell_ideal,
        "allocation_cell_deviation": allocation_cell_count - allocation_cell_ideal,
        "difficulty_allocation_count": difficulty_allocation_count,
        "difficulty_allocation_ideal": difficulty_allocation_ideal,
        "difficulty_allocation_deviation": (
            difficulty_allocation_count - difficulty_allocation_ideal
        ),
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
    group_count, incomplete_group = divmod(
        config.cores_per_family,
        len(patterns),
    )
    if incomplete_group:
        raise ValueError("Family B cores must form complete interleaving groups")
    group_cells = _balanced_group_cells(
        group_count,
        depths,
        difficulties,
    )
    allocation_counts = _allocation_counts(group_cells, patterns)
    difficulty_allocation_counts = Counter(
        difficulty
        for _, difficulty in group_cells
        for _ in patterns
    )
    allocation_cell_ideal = config.cores_per_family / (
        len(depths) * len(difficulties) * len(patterns)
    )
    difficulty_allocation_ideal = config.cores_per_family / len(difficulties)
    cores = []
    trajectory_cache: dict[
        int,
        tuple[int, int, tuple[tuple[CoreEvent, ...], ...]],
    ] = {}
    for core_index in range(config.cores_per_family):
        group_index = core_index // len(patterns)
        depth, difficulty = group_cells[group_index]
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
            distractor_count = canonical_cross_slot_update_count(
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
                allocation_cell_ideal,
                difficulty_allocation_counts[difficulty],
                difficulty_allocation_ideal,
            )
        )
    return cores


def _core_active_keys(
    config: CoreConfig,
    axis: tuple[str, str, str],
    active_object_count: int,
) -> tuple[MemoryObjectKey, ...]:
    namespace, target_entity, target_attribute = axis
    same_entity_count = min(active_object_count, len(CANONICAL_ATTRIBUTES))
    keys = list(_active_keys(axis, same_entity_count))
    if len(keys) == active_object_count:
        return tuple(keys)
    candidates = tuple(
        _key(namespace, entity, attribute)
        for entity in RELATION_QUALIFIED_ENTITIES
        if entity != target_entity
        for attribute in CANONICAL_ATTRIBUTES
    )
    target = _key(namespace, target_entity, target_attribute)
    ordered = sorted(
        candidates,
        key=lambda key: stable_id(
            "core_family_b_active_object",
            {
                "seed": config.seed,
                "target": target.canonical_id,
                "candidate": key.canonical_id,
            },
        ),
    )
    keys.extend(ordered[: active_object_count - len(keys)])
    if len({key.canonical_id for key in keys}) != active_object_count:
        raise ValueError(
            "Core Family B active objects must have distinct exact identities"
        )
    return tuple(keys)


def _ordered_core_final_values(
    config: CoreConfig,
    group_index: int,
    keys: tuple[MemoryObjectKey, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for key in keys:
        ordered = sorted(
            values_for_attribute(key.attribute),
            key=lambda value: stable_id(
                "core_family_b_final_value",
                {
                    "seed": config.seed,
                    "group_index": group_index,
                    "target": key.canonical_id,
                    "value": value,
                },
            ),
        )
        value = next((candidate for candidate in ordered if candidate not in selected), None)
        if value is None:
            raise ValueError("insufficient distinct values for Core Family B replay")
        selected.append(value)
    return tuple(selected)


def _build_core_family_b_trajectories(
    config: CoreConfig,
    group_index: int,
    axis: tuple[str, str, str],
    depth: int,
    active_object_count: int,
    cross_slot_distractor_count: int,
) -> tuple[tuple[CoreEvent, ...], ...]:
    keys = _core_active_keys(config, axis, active_object_count)
    final_values = _ordered_core_final_values(config, group_index, keys)
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


def _core_family_b_difficulty(active_object_count: int) -> Difficulty:
    if active_object_count == 2:
        return Difficulty.EASY
    if active_object_count == 4:
        return Difficulty.MEDIUM
    return Difficulty.HARD


def generate_core_family_b_cores(config: CoreConfig) -> list[SemanticCore]:
    """Generate the deterministic 480-core Core interleaved multi-slot Family B."""
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    family = config.families.interleaved_multi_slot_update
    depths = tuple(family.update_depths)
    active_counts = tuple(family.active_object_counts)
    patterns = tuple(family.interleaving_patterns)
    if depths != CORE_FAMILY_B_DEPTHS:
        raise ValueError("Core Family B update_depths do not match the approved axes")
    if active_counts != CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS:
        raise ValueError(
            "Core Family B active_object_counts do not match the approved axes"
        )
    if patterns != CORE_FAMILY_B_INTERLEAVING_PATTERNS:
        raise ValueError("Core Family B patterns do not match the approved axes")

    axes = _canonical_axis_order(config)
    schedule = family.schedule
    depths = tuple(family.update_depths)
    active_counts = tuple(family.active_object_counts)
    patterns = tuple(family.interleaving_patterns)
    groups_per_stratum = schedule.cores_per_active_object_count // len(patterns)
    if schedule.cores_per_active_object_count != groups_per_stratum * len(patterns):
        raise ValueError("Core Family B active-object schedule is not pattern-aligned")
    if family.semantic_core_count != schedule.cores_per_active_object_count * len(active_counts):
        raise ValueError("Core Family B schedule does not match semantic_core_count")
    density_by_count = {2: 0.0, 4: 0.25, 8: 0.5, 12: 0.5}
    cell_counts = Counter(
        (
            active_count,
            depths[
                (group_in_stratum + pattern_index + active_stratum_index)
                % len(depths)
            ],
            pattern,
        )
        for active_stratum_index, active_count in enumerate(active_counts)
        for group_in_stratum in range(groups_per_stratum)
        for pattern_index, pattern in enumerate(patterns)
    )
    difficulty_counts = Counter(
        _core_family_b_difficulty(active_count)
        for active_count in active_counts
        for _ in range(schedule.cores_per_active_object_count)
    )
    cores: list[SemanticCore] = []
    for active_stratum_index, active_object_count in enumerate(active_counts):
        density = density_by_count[active_object_count]
        difficulty = _core_family_b_difficulty(active_object_count)
        for group_in_stratum in range(groups_per_stratum):
            group_index = active_stratum_index * groups_per_stratum + group_in_stratum
            axis_index = group_index % len(axes)
            for pattern_index, pattern in enumerate(patterns):
                depth = depths[
                    (group_in_stratum + pattern_index + active_stratum_index)
                    % len(depths)
                ]
                base_event_count = active_object_count + depth
                distractor_count = canonical_cross_slot_update_count(
                    base_event_count,
                    density,
                )
                trajectories = _build_core_family_b_trajectories(
                    config,
                    group_index,
                    axes[axis_index],
                    depth,
                    active_object_count,
                    distractor_count,
                )
                core_index = (
                    active_stratum_index * schedule.cores_per_active_object_count
                    + group_in_stratum * len(patterns)
                    + pattern_index
                )
                core = _build_core(
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
                    cell_counts[(active_object_count, depth, pattern)],
                    schedule.cores_per_active_object_count
                    / (len(depths) * len(patterns)),
                    difficulty_counts[difficulty],
                    family.semantic_core_count / len(_DIFFICULTIES),
                    semantic_profile="core",
                )
                stratification = dict(core.stratification)
                stratification.update(
                    {
                        "active_object_stratum_count": schedule.cores_per_active_object_count,
                        "active_object_depth_pattern_cell_count": cell_counts[
                            (active_object_count, depth, pattern)
                        ],
                    }
                )
                cores.append(
                    core.model_copy(update={"stratification": stratification})
                )
    return cores


__all__ = ["generate_core_family_b_cores", "generate_family_b_cores"]
