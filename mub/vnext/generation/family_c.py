from __future__ import annotations

from itertools import product

from mub.vnext.contracts import (
    AnswerDisposition,
    CanonicalAnswer,
    Difficulty,
    EventRole,
    MemoryObjectKey,
    Operation,
    QueryType,
    ReferenceCandidate,
    ReferenceResolutionStatus,
    SurfaceReference,
    TaskFamily,
)
from mub.vnext.generation.catalogs import (
    ALIAS_MAPPINGS,
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    VALUES,
)
from mub.vnext.generation.config import EntityAttributeGroundingConfig, PilotConfig
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, stable_id, trajectory_id


_FAMILY_NAME = TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value
_ENTITY_CONDITIONS = ("distinct", "same_name", "alias", "namespace_collision")
_ATTRIBUTE_CONDITIONS = ("exact", "paraphrase", "near_name")

_ATTRIBUTE_PARAPHRASE_MAPPINGS = (
    ("home_town", "city"),
    ("workplace", "employer"),
    ("preferred_colour", "favorite_color"),
    ("telephone", "phone_number"),
    ("usual_coffee_shop", "preferred_cafe"),
    ("project_identifier", "project_code"),
    ("delivery_location", "shipping_address"),
    ("time_zone", "timezone"),
)

_ATTRIBUTE_NEAR_NAMES = (
    ("city_code", "city"),
    ("employer_code", "employer"),
    ("favorite_color_code", "favorite_color"),
    ("phone_number_type", "phone_number"),
    ("preferred_cafe_rating", "preferred_cafe"),
    ("project_code_owner", "project_code"),
    ("shipping_address_note", "shipping_address"),
    ("timezone_offset_note", "timezone"),
)

_ENTITY_DIFFICULTY = {
    "distinct": 0,
    "alias": 1,
    "same_name": 2,
    "namespace_collision": 2,
}
_ATTRIBUTE_DIFFICULTY = {"exact": 0, "paraphrase": 1, "near_name": 2}
_DIFFICULTIES = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_ENTITY_AMBIGUITY = {
    "distinct": "none",
    "alias": "moderate",
    "same_name": "high",
    "namespace_collision": "high",
}
_ATTRIBUTE_AMBIGUITY = {
    "exact": "none",
    "paraphrase": "moderate",
    "near_name": "high",
}


def _validate_config(config: PilotConfig) -> EntityAttributeGroundingConfig:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if config.cores_per_family != 120:
        raise ValueError("Family C requires cores_per_family=120")
    family = config.families.entity_attribute_grounding
    if not family.enabled:
        raise ValueError("Family C must be enabled")
    if len(family.entity_conditions) != len(_ENTITY_CONDITIONS) or set(
        family.entity_conditions
    ) != set(_ENTITY_CONDITIONS):
        raise ValueError(
            "Family C entity_conditions must include distinct, same_name, alias, "
            "and namespace_collision exactly once"
        )
    if len(family.attribute_conditions) != len(_ATTRIBUTE_CONDITIONS) or set(
        family.attribute_conditions
    ) != set(_ATTRIBUTE_CONDITIONS):
        raise ValueError(
            "Family C attribute_conditions must include exact, paraphrase, and "
            "near_name exactly once"
        )
    if set(family.difficulties) != set(_DIFFICULTIES):
        raise ValueError(
            "Family C difficulties must include easy, medium, and hard"
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


def _ordered_indices(label: str, config: PilotConfig, cell_index: int, size: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(size),
            key=lambda index: stable_id(
                label,
                {"seed": config.seed, "cell_index": cell_index, "index": index},
            ),
        )
    )


def _different_bare_name_entity(entity: str, ordered_entities: tuple[str, ...]) -> str:
    bare_name = entity.rsplit("_", 1)[-1]
    return next(
        candidate
        for candidate in ordered_entities
        if candidate.rsplit("_", 1)[-1] != bare_name
    )


def _entity_spec(
    config: PilotConfig,
    cell_index: int,
    example_index: int,
    entity_condition: str,
) -> tuple[str, tuple[tuple[str, str], ...], str, str, str]:
    namespace_order = _ordered_indices(
        "family_c_namespace", config, cell_index + example_index, len(NAMESPACES)
    )
    namespaces = tuple(NAMESPACES[index] for index in namespace_order)
    entity_order = _ordered_indices(
        "family_c_entity", config, cell_index + example_index, len(RELATION_QUALIFIED_ENTITIES)
    )
    entities = tuple(RELATION_QUALIFIED_ENTITIES[index] for index in entity_order)

    if entity_condition == "same_name":
        group_index = (cell_index + example_index) % len(SAME_NAME_ENTITIES)
        group = SAME_NAME_ENTITIES[group_index]
        pair_offset = (config.seed + cell_index + example_index) % len(group)
        first = group[pair_offset]
        second = group[(pair_offset + 1) % len(group)]
        bare_name = first.rsplit("_", 1)[-1]
        return (
            bare_name,
            ((namespaces[0], first), (namespaces[0], second)),
            f"same_name_group_v1:{bare_name}",
            f"unqualified_with_shared_namespace:{namespaces[0]}",
            "unqualified_same_name",
        )

    if entity_condition == "alias":
        mapping_index = (cell_index + example_index) % len(ALIAS_MAPPINGS)
        alias, canonical_entity = ALIAS_MAPPINGS[mapping_index]
        other = _different_bare_name_entity(canonical_entity, entities)
        return (
            alias,
            ((namespaces[0], canonical_entity), (namespaces[0], other)),
            f"reviewed_alias_v1:{alias}->{canonical_entity}",
            f"unqualified_with_shared_namespace:{namespaces[0]}",
            "reviewed_alias_map",
        )

    first = entities[0]
    if entity_condition == "namespace_collision":
        return (
            first,
            ((namespaces[0], first), (namespaces[1], first)),
            f"namespace_collision_v1:{first}:{namespaces[0]}|{namespaces[1]}",
            f"unqualified:{first}@{namespaces[0]}|{namespaces[1]}",
            "unqualified_namespace",
        )

    second = _different_bare_name_entity(first, entities[1:])
    return (
        f"{namespaces[0]}:{first}",
        ((namespaces[0], first), (namespaces[0], second)),
        f"exact_entity_v1:{first}",
        f"qualified:{namespaces[0]}",
        "exact_qualified_entity",
    )


def _attribute_spec(
    cell_index: int,
    example_index: int,
    attribute_condition: str,
) -> tuple[str, str, str, str, str]:
    attribute_index = (cell_index * 10 + example_index) % len(CANONICAL_ATTRIBUTES)
    canonical_attribute = CANONICAL_ATTRIBUTES[attribute_index]
    if attribute_condition == "paraphrase":
        surface, mapped_attribute = _ATTRIBUTE_PARAPHRASE_MAPPINGS[attribute_index]
        if mapped_attribute != canonical_attribute:
            raise ValueError("reviewed attribute paraphrase catalog is misaligned")
        return (
            surface,
            canonical_attribute,
            f"reviewed_attribute_paraphrase_v1:{surface}->{canonical_attribute}",
            f"reviewed_match:{surface}->{canonical_attribute}",
            "reviewed_attribute_paraphrase",
        )
    if attribute_condition == "near_name":
        surface, neighboring_attribute = _ATTRIBUTE_NEAR_NAMES[attribute_index]
        if neighboring_attribute != canonical_attribute:
            raise ValueError("reviewed near-name catalog is misaligned")
        return (
            surface,
            canonical_attribute,
            f"near_name_nonmatch_v1:{surface}!{canonical_attribute}",
            f"noncanonical_attribute:{surface}!={canonical_attribute}",
            "near_name_nonmatch",
        )
    return (
        canonical_attribute,
        canonical_attribute,
        f"exact_attribute_v1:{canonical_attribute}",
        f"reviewed_match:{canonical_attribute}->{canonical_attribute}",
        "exact_attribute",
    )


def _resolution(
    entity_condition: str,
    attribute_condition: str,
) -> tuple[ReferenceResolutionStatus, AnswerDisposition]:
    if attribute_condition == "near_name":
        return ReferenceResolutionStatus.NO_MATCH, AnswerDisposition.ABSTAINED
    if entity_condition in {"same_name", "namespace_collision"}:
        return ReferenceResolutionStatus.AMBIGUOUS, AnswerDisposition.ABSTAINED
    return ReferenceResolutionStatus.UNIQUE, AnswerDisposition.ANSWERED


def _difficulty(entity_condition: str, attribute_condition: str) -> Difficulty:
    level = max(
        _ENTITY_DIFFICULTY[entity_condition],
        _ATTRIBUTE_DIFFICULTY[attribute_condition],
    )
    return _DIFFICULTIES[level]


def _candidate_values(config: PilotConfig, core_index: int) -> tuple[str, str]:
    ordered = tuple(
        sorted(
            VALUES,
            key=lambda value: stable_id(
                "family_c_value",
                {"seed": config.seed, "core_index": core_index, "value": value},
            ),
        )
    )
    return ordered[0], ordered[1]


def _build_core(
    config: PilotConfig,
    core_index: int,
    cell_index: int,
    example_index: int,
    entity_condition: str,
    attribute_condition: str,
) -> SemanticCore:
    (
        entity_surface,
        entity_candidates,
        entity_mapping_id,
        namespace_evidence,
        entity_evidence_kind,
    ) = _entity_spec(
        config,
        cell_index,
        example_index,
        entity_condition,
    )
    (
        attribute_surface,
        canonical_attribute,
        attribute_mapping_id,
        near_name_evidence,
        attribute_evidence_kind,
    ) = _attribute_spec(cell_index, example_index, attribute_condition)
    keys = tuple(
        _key(namespace, entity, canonical_attribute)
        for namespace, entity in entity_candidates
    )
    values = _candidate_values(config, core_index)
    candidate_ids = tuple(
        stable_id(
            "candidate",
            {
                "family": _FAMILY_NAME,
                "core_index": core_index,
                "candidate_index": candidate_index,
                "namespace": key.namespace,
                "entity": key.entity,
                "attribute": key.attribute,
                "subkey": key.subkey,
            },
        )
        for candidate_index, key in enumerate(keys)
    )
    candidates = tuple(
        ReferenceCandidate(
            candidate_id=candidate_id,
            object_key=key,
            evidence=(
                f"event_candidate={candidate_index}; namespace={key.namespace}; "
                f"entity={key.entity}; attribute={key.attribute}"
            ),
            source_anchors=[],
        )
        for candidate_index, (candidate_id, key) in enumerate(
            zip(candidate_ids, keys)
        )
    )
    events = tuple(
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[key],
            value=value,
            role=EventRole.LATEST_GOLD,
            metadata={"candidate_index": candidate_index},
        )
        for candidate_index, (key, value) in enumerate(zip(keys, values))
    )

    status, disposition = _resolution(entity_condition, attribute_condition)
    linked_candidate_ids = (
        [candidate_ids[0]]
        if status is ReferenceResolutionStatus.UNIQUE
        else list(candidate_ids)
        if status is ReferenceResolutionStatus.AMBIGUOUS
        else []
    )
    condition_kind = (
        "attribute_paraphrase"
        if entity_condition == "distinct" and attribute_condition == "paraphrase"
        else entity_condition
    )
    surface_text = f"{entity_surface}.{attribute_surface}"
    reference = SurfaceReference(
        reference_id=stable_id(
            "reference",
            {"family": _FAMILY_NAME, "core_index": core_index},
        ),
        surface_text=surface_text,
        normalized_text=surface_text.casefold(),
        condition_kind=condition_kind,
        evidence_kind=f"{entity_evidence_kind}+{attribute_evidence_kind}",
        candidate_ids=linked_candidate_ids,
    )
    canonical = (
        CanonicalAnswer(
            disposition=AnswerDisposition.ANSWERED,
            resolution_status=ReferenceResolutionStatus.UNIQUE,
            selected_candidate_ids=[candidate_ids[0]],
            value=values[0],
        )
        if status is ReferenceResolutionStatus.UNIQUE
        else CanonicalAnswer(
            disposition=AnswerDisposition.ABSTAINED,
            resolution_status=status,
            selected_candidate_ids=[],
            abstention_reason=(
                "reference matches multiple exact memory objects"
                if status is ReferenceResolutionStatus.AMBIGUOUS
                else "reference attribute has no reviewed canonical match"
            ),
            value=None,
        )
    )
    difficulty = _difficulty(entity_condition, attribute_condition)
    linked_candidate_indices = [
        candidate_ids.index(candidate_id) for candidate_id in linked_candidate_ids
    ]
    selected_candidate_indices = [
        candidate_ids.index(candidate_id)
        for candidate_id in canonical.selected_candidate_ids
    ]
    semantic_payload = {
        "family": _FAMILY_NAME,
        "entity_condition": entity_condition,
        "attribute_condition": attribute_condition,
        "entity_mapping_id": entity_mapping_id,
        "attribute_mapping_id": attribute_mapping_id,
        "events": [
            {
                "operation": event.operation.value,
                "object_keys": [
                    _identity_payload(key) for key in event.object_keys
                ],
                "value": event.value,
                "role": event.role.value,
                "metadata": dict(event.metadata),
            }
            for event in events
        ],
        "candidate_identities": [
            _identity_payload(candidate.object_key) for candidate in candidates
        ],
        "reference_graph": {
            "condition_kind": reference.condition_kind,
            "evidence_kind": reference.evidence_kind,
            "linked_candidate_indices": linked_candidate_indices,
        },
        "canonical_answer": {
            "disposition": canonical.disposition.value,
            "resolution_status": canonical.resolution_status.value,
            "selected_candidate_indices": selected_candidate_indices,
            "value": canonical.value,
        },
    }
    identifier = core_id(_FAMILY_NAME, semantic_payload)
    profile = {
        "update_depth": 1,
        "active_object_count": len(keys),
        "entity_ambiguity": _ENTITY_AMBIGUITY[entity_condition],
        "attribute_ambiguity": _ATTRIBUTE_AMBIGUITY[attribute_condition],
        "noop_density": 0.0,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": len(events),
        "context_order": "chronological",
        "version_metadata": "none",
        "query_type": QueryType.UNRESOLVED_REFERENCE.value,
        "source_naturalness": "mixed_template",
        "alias_namespace_condition": entity_condition,
    }
    stratification = {
        "entity_condition": entity_condition,
        "attribute_condition": attribute_condition,
        "resolution_status": status.value,
        "answer_disposition": disposition.value,
        "candidate_count": len(candidates),
        "entity_mapping_id": entity_mapping_id,
        "attribute_mapping_id": attribute_mapping_id,
        "namespace_evidence": namespace_evidence,
        "near_name_evidence": near_name_evidence,
        "difficulty": difficulty.value,
        "cell_index": cell_index,
        "cell_example_index": example_index,
        "cell_count": 10,
        "num_events": len(events),
        "num_target_updates": 0,
        "noop_count": 0,
    }
    return SemanticCore(
        core_id=identifier,
        task_family=TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
        difficulty=difficulty,
        core_index=core_index,
        trajectory_id=trajectory_id(identifier, f"family_c_{core_index:03d}"),
        events=list(events),
        query_targets=list(keys),
        query_type=QueryType.UNRESOLVED_REFERENCE,
        reference_candidates=list(candidates),
        surface_references=[reference],
        canonical_answer=canonical,
        expected_answer=None,
        profile=profile,
        stratification=stratification,
    )


def generate_family_c_cores(config: PilotConfig) -> list[SemanticCore]:
    """Generate the deterministic 120-core entity/attribute grounding Family C."""
    family = _validate_config(config)
    cells = tuple(product(family.entity_conditions, family.attribute_conditions))
    expected_count = len(cells) * 10
    if expected_count != config.cores_per_family:
        raise ValueError("Family C requires exactly ten cores per condition cell")

    cores = []
    for cell_index, (entity_condition, attribute_condition) in enumerate(cells):
        for example_index in range(10):
            core_index = cell_index * 10 + example_index
            cores.append(
                _build_core(
                    config,
                    core_index,
                    cell_index,
                    example_index,
                    entity_condition,
                    attribute_condition,
                )
            )
    return cores


__all__ = ["generate_family_c_cores"]
