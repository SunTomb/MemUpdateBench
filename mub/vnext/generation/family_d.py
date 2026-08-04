from __future__ import annotations

from itertools import product

from mub.vnext.contracts import Difficulty, EventRole, MemoryObjectKey, Operation, TaskFamily
from mub.vnext.generation.catalogs import (
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    values_for_attribute,
)
from mub.vnext.generation.config import NoopWriteDisciplineConfig, PilotConfig
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id


_FAMILY_NAME = TaskFamily.NOOP_WRITE_DISCIPLINE.value
_DENSITIES = (0.25, 0.50, 0.75)
_TRAPS = (
    "semantic_near_miss",
    "duplicate_current",
    "other_entity_correction",
    "other_attribute_correction",
)
_CORE_TRAPS = (
    "transient",
    "hypothetical",
    "negated",
    "uncertain",
    "semantic_near_miss",
    "duplicate_current",
    "unsupported_inference",
)
_DENSITY_DIFFICULTY = {
    0.25: Difficulty.EASY,
    0.50: Difficulty.MEDIUM,
    0.75: Difficulty.HARD,
}
_EVENT_COUNT = 12


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


def _validate_core_config(config: CoreConfig) -> NoopWriteDisciplineConfig:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    family = config.families.noop_write_discipline
    if family.semantic_core_count != 420:
        raise ValueError("Core Family D requires semantic_core_count=420")
    if not family.enabled:
        raise ValueError("Core Family D must be enabled")
    if len(family.noop_densities) != len(_DENSITIES) or set(
        family.noop_densities
    ) != set(_DENSITIES):
        raise ValueError(
            "Core Family D noop_densities must be exactly 0.25, 0.50, and 0.75"
        )
    expected_difficulties = set(_DENSITY_DIFFICULTY.values())
    if len(family.difficulties) != len(expected_difficulties) or set(
        family.difficulties
    ) != expected_difficulties:
        raise ValueError(
            "Core Family D difficulties must include easy, medium, and hard exactly once"
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


def _identity_payload(key: MemoryObjectKey) -> dict[str, str | None]:
    return {
        "namespace": key.namespace,
        "entity": key.entity,
        "attribute": key.attribute,
        "subkey": key.subkey,
    }


def family_d_semantic_near_miss_statement(entity: str, attribute: str) -> str:
    return (
        f"A note refers to {entity.replace('_', ' ')}'s {attribute.replace('_', ' ')}, "
        "but does not state that it has changed."
    )


def family_d_duplicate_current_statement(
    entity: str,
    attribute: str,
    current_value: str,
) -> str:
    return (
        f"The record confirms that {entity.replace('_', ' ')}'s "
        f"{attribute.replace('_', ' ')} is still {current_value}."
    )


def _core_trap_statement(
    trap_type: str,
    target: MemoryObjectKey,
    current_value: str,
) -> str:
    entity = target.entity.replace("_", " ")
    attribute = target.attribute.replace("_", " ")
    if trap_type == "transient":
        return (
            f"A temporary note mentions {entity}'s {attribute}, but explicitly says "
            "the transient information must not be stored."
        )
    if trap_type == "hypothetical":
        return (
            f"A hypothetical asks what would happen if {entity}'s {attribute} "
            "changed, without asserting that any change occurred."
        )
    if trap_type == "negated":
        return (
            f"The note explicitly denies that {entity}'s {attribute} changed and "
            "provides no replacement fact."
        )
    if trap_type == "uncertain":
        return (
            f"The note is uncertain whether {entity}'s {attribute} changed and "
            "contains no confirmed value to store."
        )
    if trap_type == "semantic_near_miss":
        return family_d_semantic_near_miss_statement(target.entity, target.attribute)
    if trap_type == "duplicate_current":
        return family_d_duplicate_current_statement(
            target.entity,
            target.attribute,
            current_value,
        )
    if trap_type == "unsupported_inference":
        return (
            f"A contextual remark mentions {entity}'s {attribute}, but provides no "
            "stated fact that supports inferring a different value."
        )
    raise ValueError(f"unsupported Core Family D NOOP trap type: {trap_type}")


def family_d_independent_noop_statement(entity: str, note_number: int) -> str:
    statements = (
        "A routine status note concerns {entity}, but contains no new fact to save.",
        "A brief update mentions {entity} without changing any stored detail.",
        "A general note about {entity} provides no information that needs to be remembered.",
        "An informational message refers to {entity} but does not revise a fact.",
        "A status update names {entity} while leaving all recorded details unchanged.",
        "A passing mention of {entity} does not add or correct any information.",
        "The note concerns {entity}, but it gives no instruction to change a record.",
        "A routine message refers to {entity} and introduces no fact to store.",
        "An unrelated update mentions {entity} without altering any saved detail.",
    )
    return statements[note_number % len(statements)].format(entity=entity.replace("_", " "))


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


def _balanced_attribute_order(config: PilotConfig) -> tuple[str, ...]:
    return tuple(
        sorted(
            CANONICAL_ATTRIBUTES,
            key=lambda attribute: stable_id(
                "family_d_attribute_order",
                {"seed": config.seed, "attribute": attribute},
            ),
        )
    )


def _target_for_example(
    config: PilotConfig,
    axes: tuple[tuple[str, str, str], ...],
    core_index: int,
    example_index: int,
    trap_type: str,
) -> MemoryObjectKey:
    eligible = axes
    if trap_type == "other_attribute_correction":
        correction_attribute = _balanced_attribute_order(config)[
            example_index % len(CANONICAL_ATTRIBUTES)
        ]
        eligible = tuple(axis for axis in axes if axis[2] != correction_attribute)
    return _key(*eligible[core_index % len(eligible)])


def _current_value(config: PilotConfig, target: MemoryObjectKey) -> str:
    return min(
        values_for_attribute(target.attribute),
        key=lambda value: stable_id(
            "family_d_current_value",
            {"seed": config.seed, "target": _identity_payload(target), "value": value},
        ),
    )


def _non_target_values(
    config: PilotConfig,
    target: MemoryObjectKey,
    key: MemoryObjectKey,
    purpose: str,
    count: int,
) -> tuple[str, ...]:
    current_value = _current_value(config, target)
    candidates = values_for_attribute(key.attribute)
    if key.attribute == target.attribute:
        candidates = tuple(value for value in candidates if value != current_value)
    ordered = tuple(
        sorted(
            candidates,
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
    )
    return ordered[:count]


def _other_entity_key(
    config: PilotConfig,
    target: MemoryObjectKey,
    example_index: int,
) -> MemoryObjectKey:
    group = next(group for group in SAME_NAME_ENTITIES if target.entity in group)
    alternatives = tuple(
        sorted(
            (entity for entity in group if entity != target.entity),
            key=lambda entity: stable_id(
                "family_d_other_entity",
                {
                    "seed": config.seed,
                    "target": _identity_payload(target),
                    "entity": entity,
                },
            ),
        )
    )
    return _key(
        target.namespace,
        alternatives[example_index % len(alternatives)],
        target.attribute,
    )


def _other_attribute_key(
    config: PilotConfig,
    target: MemoryObjectKey,
    example_index: int,
) -> MemoryObjectKey:
    ordered = _balanced_attribute_order(config)
    attribute = ordered[example_index % len(ordered)]
    if attribute == target.attribute:
        attribute = next(candidate for candidate in ordered if candidate != target.attribute)
    return _key(target.namespace, target.entity, attribute)


def _noop_trap_event(
    trap_type: str,
    target: MemoryObjectKey,
    current_value: str,
    *,
    core_metadata: bool = False,
) -> CoreEvent:
    if trap_type not in _CORE_TRAPS:
        raise ValueError(f"unsupported Family D NOOP trap type: {trap_type}")

    if core_metadata:
        metadata = {
            "trap_type": trap_type,
            "lifecycle": "trap_noop",
            "semantic_effect": "noop",
            "review_status": "reviewed",
            "wording_style": "deterministic_reviewed_v1",
            "referenced_object_identity": _identity_payload(target),
            "surface_statement": _core_trap_statement(
                trap_type,
                target,
                current_value,
            ),
        }
    elif trap_type == "semantic_near_miss":
        metadata = {
            "trap_type": trap_type,
            "lifecycle": "trap_noop",
            "surface_statement": family_d_semantic_near_miss_statement(
                target.entity,
                target.attribute,
            ),
        }
    elif trap_type == "duplicate_current":
        metadata = {
            "trap_type": trap_type,
            "lifecycle": "trap_noop",
            "allow_accepted_answer_ambiguity": True,
            "surface_statement": family_d_duplicate_current_statement(
                target.entity,
                target.attribute,
                current_value,
            ),
        }
    else:
        raise ValueError(f"unsupported Pilot Family D NOOP trap type: {trap_type}")

    if trap_type == "duplicate_current":
        metadata["allow_accepted_answer_ambiguity"] = True
    return CoreEvent(
        operation=Operation.NOOP,
        object_keys=[],
        value=None,
        role=EventRole.NOOP_NEAR_MISS,
        metadata=metadata,
    )


def _correction_events(
    config: PilotConfig,
    trap_type: str,
    target: MemoryObjectKey,
    example_index: int,
) -> tuple[MemoryObjectKey, CoreEvent, CoreEvent]:
    if trap_type == "other_entity_correction":
        key = _other_entity_key(config, target, example_index)
        role = EventRole.SAME_NAME_OTHER_ENTITY
    elif trap_type == "other_attribute_correction":
        key = _other_attribute_key(config, target, example_index)
        role = EventRole.SAME_ENTITY_OTHER_ATTRIBUTE
    else:
        raise ValueError(f"unsupported Family D correction trap type: {trap_type}")
    before_value, after_value = _non_target_values(
        config,
        target,
        key,
        trap_type,
        2,
    )
    setup = CoreEvent(
        operation=Operation.ADD,
        object_keys=[key],
        value=before_value,
        role=role,
        metadata={"lifecycle": "correction_before"},
    )
    correction = CoreEvent(
        operation=Operation.UPDATE,
        object_keys=[key],
        value=after_value,
        role=role,
        metadata={
            "trap_type": trap_type,
            "lifecycle": "correction_after",
        },
    )
    return key, setup, correction


def _filler_write_events(
    config: PilotConfig,
    target: MemoryObjectKey,
    excluded_keys: tuple[MemoryObjectKey, ...],
    example_index: int,
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
                    "example_variant": example_index,
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
            value=_non_target_values(
                config,
                target,
                key,
                "independent_current",
                1,
            )[0],
            role=EventRole.NEUTRAL,
            metadata={"lifecycle": "independent_current"},
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
                "lifecycle": "independent_noop",
                "surface_statement": family_d_independent_noop_statement(
                    target.entity,
                    index + 1,
                ),
            },
        )
        for index in range(count)
    )


def _schedule_events(
    config: PilotConfig,
    target: MemoryObjectKey,
    density: float,
    trap_type: str,
    example_index: int,
    event_specs: tuple[tuple[str, CoreEvent, frozenset[str]], ...],
) -> tuple[CoreEvent, ...]:
    remaining = {tag: (event, dependencies) for tag, event, dependencies in event_specs}
    emitted: list[str] = []
    events: list[CoreEvent] = []
    while remaining:
        ready = tuple(
            tag
            for tag, (_, dependencies) in remaining.items()
            if dependencies.issubset(emitted)
        )
        if not ready:
            raise ValueError("Family D event dependencies contain a cycle")
        selected = min(
            ready,
            key=lambda tag: stable_id(
                "family_d_event_order",
                {
                    "seed": config.seed,
                    "target": _identity_payload(target),
                    "density": density,
                    "trap_type": trap_type,
                    "example_variant": example_index,
                    "emitted": emitted,
                    "event_tag": tag,
                },
            ),
        )
        event, _ = remaining.pop(selected)
        emitted.append(selected)
        events.append(event)
    return tuple(events)


def _semantic_payload(
    events: tuple[CoreEvent, ...] | list[CoreEvent],
    target: MemoryObjectKey,
    density: float,
    trap_type: str,
) -> dict[str, object]:
    return {
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
                "lifecycle": event.metadata["lifecycle"],
            }
            for event in events
        ],
    }


def _build_core(
    config: PilotConfig | CoreConfig,
    core_index: int,
    example_index: int,
    target: MemoryObjectKey,
    density: float,
    trap_type: str,
    *,
    core_mode: bool = False,
) -> SemanticCore:
    current_value = _current_value(config, target)
    target_event = CoreEvent(
        operation=Operation.ADD,
        object_keys=[target],
        value=current_value,
        role=EventRole.LATEST_GOLD,
        metadata={"lifecycle": "target_current"},
    )
    requested_noop_count = int(_EVENT_COUNT * density)
    requested_write_count = _EVENT_COUNT - requested_noop_count
    event_specs: list[tuple[str, CoreEvent, frozenset[str]]] = [
        ("target", target_event, frozenset())
    ]
    excluded_keys: tuple[MemoryObjectKey, ...] = ()
    if core_mode or trap_type in {"semantic_near_miss", "duplicate_current"}:
        trap_event = _noop_trap_event(
            trap_type,
            target,
            current_value,
            core_metadata=core_mode,
        )
        event_specs.append(("trap", trap_event, frozenset({"target"})))
        filler_write_count = requested_write_count - 1
        filler_noop_count = requested_noop_count - 1
    else:
        correction_key, setup_event, trap_event = _correction_events(
            config,
            trap_type,
            target,
            example_index,
        )
        excluded_keys = (correction_key,)
        event_specs.extend(
            (
                ("correction_setup", setup_event, frozenset()),
                (
                    "trap",
                    trap_event,
                    frozenset({"target", "correction_setup"}),
                ),
            )
        )
        filler_write_count = requested_write_count - 3
        filler_noop_count = requested_noop_count
    filler_writes = _filler_write_events(
        config,
        target,
        excluded_keys,
        example_index,
        filler_write_count,
    )
    filler_noops = _filler_noop_events(target, filler_noop_count)
    event_specs.extend(
        (f"write_{index:02d}", event, frozenset())
        for index, event in enumerate(filler_writes)
    )
    event_specs.extend(
        (f"noop_{index:02d}", event, frozenset())
        for index, event in enumerate(filler_noops)
    )
    events = _schedule_events(
        config,
        target,
        density,
        trap_type,
        example_index,
        tuple(event_specs),
    )

    num_events = len(events)
    noop_count = sum(event.operation is Operation.NOOP for event in events)
    true_write_count = sum(event.operation is not Operation.NOOP for event in events)
    num_target_updates = sum(
        event.operation is Operation.UPDATE and target in event.object_keys
        for event in events
    )
    duplicate_current_count = sum(
        event.metadata.get("trap_type") == "duplicate_current"
        for event in events
    )
    active_object_count = len(
        {
            key.canonical_id
            for event in events
            if event.operation is not Operation.NOOP
            for key in event.object_keys
        }
    )
    observed_density = noop_count / num_events
    difficulty = _DENSITY_DIFFICULTY[density]
    semantic_payload = _semantic_payload(events, target, density, trap_type)
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    trajectory_variant = stable_id(
        "family_d_trajectory_variant",
        {
            "trap_type": trap_type,
            "events": semantic_payload["events"],
        },
    )
    profile = {
        "update_depth": 1,
        "active_object_count": active_object_count,
        "entity_ambiguity": "none",
        "attribute_ambiguity": "none",
        "noop_density": observed_density,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": num_events,
        "context_order": "chronological",
        "version_metadata": "none",
        "query_type": "current_state",
        "source_naturalness": "synthetic",
        "write_trap_type": trap_type,
        "duplicate_current_condition": bool(duplicate_current_count),
    }
    stratification = {
        "num_events": num_events,
        "true_write_count": true_write_count,
        "num_target_updates": num_target_updates,
        "noop_count": noop_count,
        "duplicate_current_count": duplicate_current_count,
        "trap_type": trap_type,
        "configured_noop_density": density,
        "observed_noop_density": observed_density,
        "trap_position": events.index(trap_event),
        "operation_signature": ",".join(event.operation.value for event in events),
    }
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE,
        difficulty=difficulty,
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, trajectory_variant),
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
            target = _target_for_example(
                config,
                axes,
                core_index,
                example_index,
                trap_type,
            )
            cores.append(
                _build_core(
                    config,
                    core_index,
                    example_index,
                    target,
                    density,
                    trap_type,
                )
            )
    return cores


def generate_core_family_d_cores(config: CoreConfig) -> list[SemanticCore]:
    """Generate the deterministic 420-core Core NOOP/write-discipline grid."""
    family = _validate_core_config(config)
    cells = tuple(product(_CORE_TRAPS, family.noop_densities))
    examples_per_cell, remainder = divmod(family.semantic_core_count, len(cells))
    if remainder or examples_per_cell != 20:
        raise ValueError("Core Family D requires exactly 20 cores per trap/density cell")

    axes = _canonical_axis_order(config)
    cores = []
    for cell_index, (trap_type, density) in enumerate(cells):
        for example_index in range(examples_per_cell):
            core_index = cell_index * examples_per_cell + example_index
            target = _target_for_example(
                config,
                axes,
                core_index,
                example_index,
                trap_type,
            )
            cores.append(
                _build_core(
                    config,
                    core_index,
                    example_index,
                    target,
                    density,
                    trap_type,
                    core_mode=True,
                )
            )
    return cores


__all__ = [
    "family_d_duplicate_current_statement",
    "family_d_independent_noop_statement",
    "family_d_semantic_near_miss_statement",
    "generate_core_family_d_cores",
    "generate_family_d_cores",
]
