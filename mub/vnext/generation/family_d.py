from __future__ import annotations

from itertools import product

from mub.vnext.contracts import Difficulty, EventRole, MemoryObjectKey, Operation, TaskFamily
from mub.vnext.generation.catalogs import (
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    VALUES,
)
from mub.vnext.generation.config import NoopWriteDisciplineConfig, PilotConfig
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id


_FAMILY_NAME = TaskFamily.NOOP_WRITE_DISCIPLINE.value
_DENSITIES = (0.25, 0.50, 0.75)
_TRAPS = (
    "semantic_near_miss",
    "duplicate_current",
    "other_entity_correction",
    "other_attribute_correction",
)
_DENSITY_DIFFICULTY = {
    0.25: Difficulty.EASY,
    0.50: Difficulty.MEDIUM,
    0.75: Difficulty.HARD,
}
_EVENT_COUNT = 8


def _validate_config(config: PilotConfig) -> NoopWriteDisciplineConfig:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if config.cores_per_family != 120:
        raise ValueError("Family D requires cores_per_family=120")
    family = config.families.noop_write_discipline
    if not family.enabled:
        raise ValueError("Family D must be enabled")
    if len(family.noop_densities) != len(_DENSITIES) or set(
        family.noop_densities
    ) != set(_DENSITIES):
        raise ValueError("Family D noop_densities must be exactly 0.25, 0.50, and 0.75")
    if len(family.trap_types) != len(_TRAPS) or set(family.trap_types) != set(_TRAPS):
        raise ValueError("Family D trap_types must include each reviewed trap exactly once")
    expected_difficulties = set(_DENSITY_DIFFICULTY.values())
    if len(family.difficulties) != len(expected_difficulties) or set(
        family.difficulties
    ) != expected_difficulties:
        raise ValueError("Family D difficulties must include easy, medium, and hard exactly once")
    return family


def _key(namespace: str, entity: str, attribute: str) -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=None,
    )


def _identity_payload(key: MemoryObjectKey) -> dict[str, str | None]:
    return {
        "namespace": key.namespace,
        "entity": key.entity,
        "attribute": key.attribute,
        "subkey": key.subkey,
    }


def _canonical_axis_order(config: PilotConfig) -> tuple[tuple[str, str, str], ...]:
    axes = tuple(product(NAMESPACES, RELATION_QUALIFIED_ENTITIES, CANONICAL_ATTRIBUTES))
    return tuple(
        sorted(
            axes,
            key=lambda axis: stable_id(
                "family_d_axis",
                {
                    "seed": config.seed,
                    "namespace": axis[0],
                    "entity": axis[1],
                    "attribute": axis[2],
                },
            ),
        )
    )


def _current_value(config: PilotConfig, target: MemoryObjectKey) -> str:
    return min(
        VALUES,
        key=lambda value: stable_id(
            "family_d_current_value",
            {"seed": config.seed, "target": _identity_payload(target), "value": value},
        ),
    )


def _different_value(
    config: PilotConfig,
    target: MemoryObjectKey,
    key: MemoryObjectKey,
    purpose: str,
) -> str:
    current_value = _current_value(config, target)
    return min(
        (value for value in VALUES if value != current_value),
        key=lambda value: stable_id(
            "family_d_non_target_value",
            {
                "seed": config.seed,
                "target": _identity_payload(target),
                "key": _identity_payload(key),
                "purpose": purpose,
                "value": value,
            },
        ),
    )


def _other_entity_key(target: MemoryObjectKey) -> MemoryObjectKey:
    group = next(group for group in SAME_NAME_ENTITIES if target.entity in group)
    entity = next(candidate for candidate in group if candidate != target.entity)
    return _key(target.namespace, entity, target.attribute)


def _other_attribute_key(target: MemoryObjectKey) -> MemoryObjectKey:
    attribute = next(
        candidate for candidate in CANONICAL_ATTRIBUTES if candidate != target.attribute
    )
    return _key(target.namespace, target.entity, attribute)


def _trap_event(
    config: PilotConfig,
    trap_type: str,
    target: MemoryObjectKey,
    current_value: str,
) -> CoreEvent:
    if trap_type == "semantic_near_miss":
        return CoreEvent(
            operation=Operation.NOOP,
            object_keys=[],
            value=None,
            role=EventRole.NOOP_NEAR_MISS,
            metadata={
                "trap_type": trap_type,
                "surface_statement": (
                    f"A hypothetical note mentions {target.entity}.{target.attribute}, "
                    "but it does not assert a current-state change."
                ),
            },
        )
    if trap_type == "duplicate_current":
        return CoreEvent(
            operation=Operation.NOOP,
            object_keys=[],
            value=None,
            role=EventRole.NOOP_NEAR_MISS,
            metadata={
                "trap_type": trap_type,
                "allow_accepted_answer_ambiguity": True,
                "surface_statement": (
                    f"{target.entity}.{target.attribute} remains exactly {current_value}; "
                    "this repeats the exact current target value."
                ),
            },
        )
    if trap_type == "other_entity_correction":
        key = _other_entity_key(target)
        return CoreEvent(
            operation=Operation.ADD,
            object_keys=[key],
            value=_different_value(config, target, key, trap_type),
            role=EventRole.SAME_NAME_OTHER_ENTITY,
            metadata={"trap_type": trap_type},
        )
    if trap_type == "other_attribute_correction":
        key = _other_attribute_key(target)
        return CoreEvent(
            operation=Operation.ADD,
            object_keys=[key],
            value=_different_value(config, target, key, trap_type),
            role=EventRole.SAME_ENTITY_OTHER_ATTRIBUTE,
            metadata={"trap_type": trap_type},
        )
    raise ValueError(f"unsupported Family D trap type: {trap_type}")


def _filler_write_events(
    config: PilotConfig,
    target: MemoryObjectKey,
    excluded_keys: tuple[MemoryObjectKey, ...],
    count: int,
) -> tuple[CoreEvent, ...]:
    excluded_ids = {target.canonical_id, *(key.canonical_id for key in excluded_keys)}
    candidates = tuple(
        sorted(
            (
                _key(namespace, entity, attribute)
                for namespace, entity, attribute in product(
                    NAMESPACES,
                    RELATION_QUALIFIED_ENTITIES,
                    CANONICAL_ATTRIBUTES,
                )
                if _key(namespace, entity, attribute).canonical_id not in excluded_ids
            ),
            key=lambda key: stable_id(
                "family_d_filler_key",
                {
                    "seed": config.seed,
                    "target": _identity_payload(target),
                    "candidate": _identity_payload(key),
                },
            ),
        )
    )
    selected = candidates[:count]
    return tuple(
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[key],
            value=_different_value(config, target, key, "ordinary_write"),
            role=EventRole.NEUTRAL,
            metadata={"event_kind": "ordinary_write"},
        )
        for key in selected
    )


def _filler_noop_events(
    target: MemoryObjectKey,
    count: int,
) -> tuple[CoreEvent, ...]:
    return tuple(
        CoreEvent(
            operation=Operation.NOOP,
            object_keys=[],
            value=None,
            role=EventRole.NEUTRAL,
            metadata={
                "event_kind": "ordinary_noop",
                "surface_statement": (
                    f"Background note {index + 1} is related to {target.entity} but "
                    "does not direct any memory change."
                ),
            },
        )
        for index in range(count)
    )


def _build_core(
    config: PilotConfig,
    core_index: int,
    target: MemoryObjectKey,
    density: float,
    trap_type: str,
) -> SemanticCore:
    current_value = _current_value(config, target)
    target_event = CoreEvent(
        operation=Operation.ADD,
        object_keys=[target],
        value=current_value,
        role=EventRole.LATEST_GOLD,
        metadata={"event_kind": "target_initialization"},
    )
    trap_event = _trap_event(config, trap_type, target, current_value)
    noop_count = int(_EVENT_COUNT * density)
    true_write_count = _EVENT_COUNT - noop_count
    trap_is_noop = trap_event.operation is Operation.NOOP
    filler_write_count = true_write_count - 1 - (0 if trap_is_noop else 1)
    filler_noop_count = noop_count - (1 if trap_is_noop else 0)
    filler_writes = _filler_write_events(
        config,
        target,
        tuple(trap_event.object_keys),
        filler_write_count,
    )
    filler_noops = _filler_noop_events(target, filler_noop_count)
    events = (target_event, trap_event, *filler_writes, *filler_noops)
    duplicate_current_count = int(trap_type == "duplicate_current")
    difficulty = _DENSITY_DIFFICULTY[density]
    semantic_payload = {
        "family": _FAMILY_NAME,
        "noop_density": density,
        "trap_type": trap_type,
        "target": _identity_payload(target),
        "events": [
            {
                "operation": event.operation.value,
                "object_keys": [_identity_payload(key) for key in event.object_keys],
                "value": event.value,
                "role": event.role.value,
                "metadata": dict(event.metadata),
            }
            for event in events
        ],
    }
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    profile = {
        "update_depth": 1,
        "active_object_count": true_write_count,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": density,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": _EVENT_COUNT,
        "context_order": "chronological",
        "version_metadata": "none",
        "query_type": "current_state",
        "source_naturalness": "synthetic",
        "write_trap_type": trap_type,
        "duplicate_current_condition": bool(duplicate_current_count),
    }
    stratification = {
        "num_events": _EVENT_COUNT,
        "true_write_count": true_write_count,
        "num_target_updates": 0,
        "noop_count": noop_count,
        "duplicate_current_count": duplicate_current_count,
        "trap_type": trap_type,
        "configured_noop_density": density,
        "observed_noop_density": noop_count / _EVENT_COUNT,
    }
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE,
        difficulty=difficulty,
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, f"family_d_{core_index:03d}"),
        events=list(events),
        query_targets=[target],
        expected_answer=current_value,
        profile=profile,
        stratification=stratification,
    )


def generate_family_d_cores(config: PilotConfig) -> list[SemanticCore]:
    """Generate the deterministic 120-core NOOP/write-discipline Family D."""
    family = _validate_config(config)
    cells = tuple(product(family.noop_densities, family.trap_types))
    examples_per_cell, remainder = divmod(config.cores_per_family, len(cells))
    if remainder or examples_per_cell != 10:
        raise ValueError("Family D requires exactly ten cores per density/trap cell")
    axes = _canonical_axis_order(config)
    cores = []
    for cell_index, (density, trap_type) in enumerate(cells):
        for example_index in range(examples_per_cell):
            core_index = cell_index * examples_per_cell + example_index
            target = _key(*axes[core_index % len(axes)])
            cores.append(
                _build_core(
                    config,
                    core_index,
                    target,
                    density,
                    trap_type,
                )
            )
    return cores


__all__ = ["generate_family_d_cores"]
