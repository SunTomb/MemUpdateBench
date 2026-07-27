from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import ContractModel, MemoryObjectKey
from mub.vnext.contracts.enums import Difficulty, EventRole, Operation, TaskFamily
from mub.vnext.io import canonical_json_bytes

from mub.vnext.generation import (
    ALIAS_MAPPINGS,
    CANONICAL_ATTRIBUTES,
    CoreEvent,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    SURFACE_TEMPLATE_SETS,
    SemanticCore,
    VALUES,
    action_id,
    core_id,
    event_id,
    paraphrase_group_id,
    query_id,
    select_conflicting_values,
    source_id,
    stable_id,
    task_id,
    trajectory_id,
)


def test_stable_id_is_canonical_and_repeatable() -> None:
    payload = {"family": "A", "axes": {"depth": 4, "hard": True}}
    reordered = {"axes": {"hard": True, "depth": 4}, "family": "A"}

    first = stable_id("semantic_core", payload)

    assert first == stable_id("semantic_core", payload)
    assert first == stable_id("semantic_core", reordered)
    assert re.fullmatch(r"semantic_core_[0-9a-f]{16}", first)
    assert stable_id("semantic_core", {**payload, "family": "B"}) != first


@pytest.mark.parametrize(
    "prefix",
    ["", "Core", "core-id", "core id", "_core", "core_", "core__id", "1core"],
)
def test_stable_id_rejects_malformed_prefixes(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        stable_id(prefix, {"value": 1})


@pytest.mark.parametrize("prefix", [None, 7, True])
def test_stable_id_rejects_non_string_prefixes(prefix: object) -> None:
    with pytest.raises(TypeError, match="prefix"):
        stable_id(prefix, {"value": 1})  # type: ignore[arg-type]


def test_explicit_id_helpers_use_distinct_namespaces() -> None:
    semantic_core = core_id("repeated_same_slot_update", {"depth": 4})
    task = task_id(semantic_core, 2)
    event = event_id(task, 3)
    action = action_id(task, 3, 1)
    query = query_id(task, 0)
    source = source_id("personal", 5, {"document": "alpha"})
    trajectory = trajectory_id(semantic_core, 0)
    paraphrase = paraphrase_group_id(semantic_core, "query_answer")

    ids = {
        "core": semantic_core,
        "task": task,
        "event": event,
        "action": action,
        "query": query,
        "source": source,
        "trajectory": trajectory,
        "paraphrase_group": paraphrase,
    }

    assert len(set(ids.values())) == len(ids)
    for prefix, identifier in ids.items():
        assert re.fullmatch(rf"{prefix}_[0-9a-f]{{16}}", identifier)


def test_id_helpers_are_repeatable_and_change_with_meaningful_inputs() -> None:
    semantic_core = core_id("entity_attribute_grounding", {"condition": "alias"})
    task = task_id(semantic_core, 1)

    assert core_id(
        "entity_attribute_grounding", {"condition": "alias", "depth": 4}
    ) == core_id(
        "entity_attribute_grounding", {"depth": 4, "condition": "alias"}
    )
    assert core_id(
        "entity_attribute_grounding", {"condition": "alias"}
    ) != core_id("entity_attribute_grounding", {"condition": "same_name"})
    assert task == task_id(semantic_core, 1)
    assert task != task_id(semantic_core, 2)
    assert event_id(task, 0) != event_id(task, 1)
    assert action_id(task, 0, 0) != action_id(task, 0, 1)
    assert query_id(task, 0) != query_id(task, 1)
    assert source_id("personal", 0, {"x": 1}) != source_id(
        "work", 0, {"x": 1}
    )
    assert trajectory_id(semantic_core, 0) != trajectory_id(semantic_core, 1)
    assert paraphrase_group_id(
        semantic_core, "event"
    ) != paraphrase_group_id(semantic_core, "query")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: task_id("core_abcd", -1), "surface_variant_index"),
        (lambda: event_id("task_abcd", True), "event_index"),
        (lambda: action_id("task_abcd", 0, -1), "action_index"),
        (lambda: query_id("task_abcd", -1), "query_index"),
        (lambda: source_id("personal", -1, {}), "source_index"),
        (lambda: trajectory_id("core_abcd", -1), "trajectory_variant"),
    ],
)
def test_id_helpers_reject_invalid_indices(call: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        call()  # type: ignore[operator]


def test_catalogs_are_immutable_unique_and_preserve_distinctions() -> None:
    catalogs = (
        NAMESPACES,
        RELATION_QUALIFIED_ENTITIES,
        SAME_NAME_ENTITIES,
        ALIAS_MAPPINGS,
        CANONICAL_ATTRIBUTES,
        VALUES,
        SURFACE_TEMPLATE_SETS,
    )
    assert all(isinstance(catalog, tuple) for catalog in catalogs)

    assert len(NAMESPACES) == len(set(NAMESPACES))
    assert len(RELATION_QUALIFIED_ENTITIES) == len(set(RELATION_QUALIFIED_ENTITIES))
    assert len(CANONICAL_ATTRIBUTES) == len(set(CANONICAL_ATTRIBUTES))
    assert len(VALUES) == len(set(VALUES))

    same_name_members = [entity for group in SAME_NAME_ENTITIES for entity in group]
    alias_names = [alias for alias, _ in ALIAS_MAPPINGS]
    assert len(same_name_members) == len(set(same_name_members))
    assert set(same_name_members) <= set(RELATION_QUALIFIED_ENTITIES)
    assert len(alias_names) == len(set(alias_names))
    assert all(target in RELATION_QUALIFIED_ENTITIES for _, target in ALIAS_MAPPINGS)
    assert set(alias_names).isdisjoint(RELATION_QUALIFIED_ENTITIES)
    assert set(NAMESPACES).isdisjoint(RELATION_QUALIFIED_ENTITIES)
    assert set(CANONICAL_ATTRIBUTES).isdisjoint(RELATION_QUALIFIED_ENTITIES)

    with pytest.raises(TypeError):
        NAMESPACES[0] = "changed"  # type: ignore[index]


def test_surface_catalog_has_exactly_three_immutable_variants() -> None:
    assert len(SURFACE_TEMPLATE_SETS) == 3
    assert all(isinstance(template_set, tuple) for template_set in SURFACE_TEMPLATE_SETS)
    variant_ids = [template_set[0] for template_set in SURFACE_TEMPLATE_SETS]
    assert len(variant_ids) == len(set(variant_ids))
    assert all("{entity}" in template_set[1] for template_set in SURFACE_TEMPLATE_SETS)
    assert all("{attribute}" in template_set[1] for template_set in SURFACE_TEMPLATE_SETS)
    assert all("{value}" in template_set[1] for template_set in SURFACE_TEMPLATE_SETS)

    with pytest.raises(TypeError):
        SURFACE_TEMPLATE_SETS[0][0] = "changed"  # type: ignore[index]


def test_conflicting_value_selection_excludes_current_and_is_deterministic() -> None:
    values = ("red", "blue", "green", "blue", "gold")
    seed = {"core": "abc", "event": 2}

    selected = select_conflicting_values(values, "blue", 3, seed)

    assert selected == select_conflicting_values(tuple(reversed(values)), "blue", 3, seed)
    assert selected == select_conflicting_values(values, "blue", 3, seed)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert "blue" not in selected
    assert set(selected) <= {"red", "green", "gold"}
    assert selected != select_conflicting_values(
        values, "blue", 3, {"core": "different", "event": 2}
    )


def test_conflicting_value_selection_zero_count_is_explicit() -> None:
    assert select_conflicting_values(VALUES, VALUES[0], 0, {"seed": 1}) == ()


@pytest.mark.parametrize("count", [-1, 1.0, True])
def test_conflicting_value_selection_rejects_invalid_count(count: object) -> None:
    with pytest.raises((TypeError, ValueError), match="count"):
        select_conflicting_values(VALUES, VALUES[0], count, {})  # type: ignore[arg-type]


def test_conflicting_value_selection_raises_for_insufficient_distinct_pool() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        select_conflicting_values(("current", "current", "other"), "current", 2, {})


CORE_ID = "core_0123456789abcdef"
TRAJECTORY_ID = "trajectory_fedcba9876543210"


def _location_key(*, object_type: str = "slot") -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type=object_type,
        namespace="default",
        entity="friend:alex",
        attribute="location",
        subkey=None,
    )


def _representative_core() -> SemanticCore:
    target = _location_key()
    return SemanticCore(
        core_id=CORE_ID,
        task_family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        core_index=0,
        trajectory_id=TRAJECTORY_ID,
        events=[
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[target],
                value="Dalian",
                role=EventRole.STALE_SAME_SLOT,
                metadata={"sequence": 0},
            ),
            CoreEvent(
                operation=Operation.UPDATE,
                object_keys=[target],
                value="Qingdao",
                role=EventRole.LATEST_GOLD,
                metadata={"sequence": 1},
            ),
        ],
        query_targets=[target],
        expected_answer="Qingdao",
        profile={"update_depth": 1},
        stratification={"update_depth": 1},
    )


def test_semantic_core_accepts_representative_same_slot_update() -> None:
    core = _representative_core()

    assert isinstance(core, ContractModel)
    assert core.task_family is TaskFamily.REPEATED_SAME_SLOT
    assert core.difficulty is Difficulty.EASY
    assert [event.operation for event in core.events] == [
        Operation.ADD,
        Operation.UPDATE,
    ]
    assert [event.value for event in core.events] == ["Dalian", "Qingdao"]
    assert [event.role for event in core.events] == [
        EventRole.STALE_SAME_SLOT,
        EventRole.LATEST_GOLD,
    ]
    assert core.query_targets[0].canonical_id == "default|friend:alex|location|"
    assert core.expected_answer == "Qingdao"
    assert core.profile["update_depth"] == 1


@pytest.mark.parametrize(
    ("field_name", "identifier"),
    [
        ("core_id", ""),
        ("core_id", "core-0123456789abcdef"),
        ("core_id", "core_0123456789abcdeg"),
        ("core_id", "task_0123456789abcdef"),
        ("trajectory_id", "trajectory_0123"),
        ("trajectory_id", "Trajectory_0123456789abcdef"),
        ("trajectory_id", "core_0123456789abcdef"),
    ],
)
def test_semantic_core_rejects_malformed_stable_ids(
    field_name: str,
    identifier: str,
) -> None:
    data = _representative_core().model_dump(mode="python")
    data[field_name] = identifier

    with pytest.raises(ValidationError, match=field_name):
        SemanticCore.model_validate(data)


def test_semantic_core_rejects_negative_or_coerced_core_index() -> None:
    data = _representative_core().model_dump(mode="python")
    for invalid in (-1, "0", 0.0, True):
        with pytest.raises(ValidationError, match="core_index"):
            SemanticCore.model_validate({**data, "core_index": invalid})


def test_semantic_core_requires_events_and_query_targets() -> None:
    data = _representative_core().model_dump(mode="python")

    with pytest.raises(ValidationError, match="events"):
        SemanticCore.model_validate({**data, "events": []})
    with pytest.raises(ValidationError, match="query_targets"):
        SemanticCore.model_validate({**data, "query_targets": []})


def test_core_event_requires_targets_for_mutations_but_allows_targetless_noop() -> None:
    with pytest.raises(ValidationError, match="targets"):
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[],
            value="Dalian",
            role=EventRole.STALE_SAME_SLOT,
        )

    noop = CoreEvent(
        operation=Operation.NOOP,
        object_keys=[],
        value=None,
        role=EventRole.NOOP_NEAR_MISS,
    )
    assert noop.object_keys == ()
    assert noop.value is None


def test_core_event_rejects_duplicate_exact_object_keys_ignoring_object_type() -> None:
    with pytest.raises(ValidationError, match="duplicate.*object_keys"):
        CoreEvent(
            operation=Operation.UPDATE,
            object_keys=[_location_key(), _location_key(object_type="profile")],
            value="Qingdao",
            role=EventRole.LATEST_GOLD,
        )


def test_semantic_core_rejects_duplicate_query_identity_ignoring_object_type() -> None:
    data = _representative_core().model_dump(mode="python")
    data["query_targets"] = [
        _location_key(),
        _location_key(object_type="profile"),
    ]

    with pytest.raises(ValidationError, match="duplicate.*query_targets"):
        SemanticCore.model_validate(data)


@pytest.mark.parametrize(
    ("operation", "targets", "value", "message"),
    [
        (Operation.NOOP, [_location_key()], None, "NOOP"),
        (Operation.NOOP, [], "ignored", "NOOP"),
        (Operation.ADD, [], "Dalian", "ADD"),
        (Operation.ADD, [_location_key()], None, "ADD"),
        (Operation.UPDATE, [], "Qingdao", "UPDATE"),
        (Operation.UPDATE, [_location_key()], None, "UPDATE"),
        (Operation.DELETE, [], None, "DELETE"),
        (Operation.DELETE, [_location_key()], "Qingdao", "DELETE"),
    ],
)
def test_core_event_enforces_operation_value_legality(
    operation: Operation,
    targets: list[MemoryObjectKey],
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        CoreEvent(
            operation=operation,
            object_keys=targets,
            value=value,
            role=EventRole.NEUTRAL,
        )


def test_core_event_accepts_authoritative_delete_shape() -> None:
    event = CoreEvent(
        operation=Operation.DELETE,
        object_keys=[_location_key()],
        value=None,
        role=EventRole.DELETION,
    )
    assert event.operation is Operation.DELETE
    assert event.value is None


def test_event_roles_are_explicit_and_preserve_stale_duplicate_distinction() -> None:
    common = {
        "operation": Operation.UPDATE,
        "object_keys": [_location_key()],
        "value": "Qingdao",
    }
    stale = CoreEvent(**common, role=EventRole.STALE_SAME_SLOT)
    duplicate = CoreEvent(**common, role=EventRole.DUPLICATE_CURRENT)

    assert stale.role is EventRole.STALE_SAME_SLOT
    assert duplicate.role is EventRole.DUPLICATE_CURRENT
    assert stale.role is not duplicate.role


def test_semantic_core_rejects_blank_stratification_keys() -> None:
    data = _representative_core().model_dump(mode="python")
    with pytest.raises(ValidationError, match="stratification.*blank"):
        SemanticCore.model_validate({**data, "stratification": {"  ": 1}})


def test_core_records_reject_non_json_and_nonfinite_values() -> None:
    with pytest.raises(ValidationError, match="finite JSON"):
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[_location_key()],
            value=float("nan"),
            role=EventRole.LATEST_GOLD,
        )

    data = _representative_core().model_dump(mode="python")
    with pytest.raises(ValidationError, match="JSON"):
        SemanticCore.model_validate({**data, "profile": {"bad": {1, 2}}})
    with pytest.raises(ValidationError, match="finite"):
        SemanticCore.model_validate(
            {**data, "stratification": {"dose": float("inf")}}
        )


def test_core_records_are_strict_frozen_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match="operation"):
        CoreEvent(
            operation="ADD",  # type: ignore[arg-type]
            object_keys=[_location_key()],
            value="Dalian",
            role=EventRole.STALE_SAME_SLOT,
        )

    event_data = _representative_core().events[0].model_dump(mode="python")
    with pytest.raises(ValidationError, match="extra"):
        CoreEvent.model_validate({**event_data, "extra": True})

    core = _representative_core()
    with pytest.raises(ValidationError, match="frozen"):
        core.core_index = 1
    core_data = core.model_dump(mode="python")
    with pytest.raises(ValidationError, match="extra"):
        SemanticCore.model_validate({**core_data, "extra": True})


def test_core_records_isolate_and_freeze_nested_mutable_inputs() -> None:
    source_key = _location_key()
    object_keys = [source_key]
    event_value = {"history": ["Dalian"]}
    event_metadata = {"nested": {"tags": ["old"]}}
    event = CoreEvent(
        operation=Operation.ADD,
        object_keys=object_keys,
        value=event_value,
        role=EventRole.STALE_SAME_SLOT,
        metadata=event_metadata,
    )
    events = [event]
    query_targets = [source_key]
    profile = {"update_depth": 1, "axes": {"values": [1]}}
    stratification = {"update_depth": 1}
    core = SemanticCore(
        core_id=CORE_ID,
        task_family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        core_index=0,
        trajectory_id=TRAJECTORY_ID,
        events=events,
        query_targets=query_targets,
        expected_answer={"city": "Dalian"},
        profile=profile,
        stratification=stratification,
    )

    object_keys.clear()
    event_value["history"].append("Qingdao")
    event_metadata["nested"]["tags"].append("new")
    events.clear()
    query_targets.clear()
    profile["axes"]["values"].append(2)
    stratification["update_depth"] = 99
    source_key.entity = "friend:changed"

    assert len(core.events) == 1
    assert len(core.query_targets) == 1
    assert core.query_targets[0].entity == "friend:alex"
    assert core.events[0].value["history"] == ("Dalian",)
    assert core.events[0].metadata["nested"]["tags"] == ("old",)
    assert core.profile["axes"]["values"] == (1,)
    assert core.stratification["update_depth"] == 1

    with pytest.raises(AttributeError):
        core.events.append(event)
    with pytest.raises(TypeError):
        core.profile["new"] = True
    with pytest.raises(ValidationError, match="frozen"):
        core.query_targets[0].entity = "friend:changed"


def test_core_model_dumps_are_canonical_safe_and_deterministic() -> None:
    core = _representative_core()
    dumped = core.model_dump(mode="python")
    rebuilt = SemanticCore.model_validate(dumped)

    assert isinstance(dumped["events"], list)
    assert isinstance(dumped["query_targets"], list)
    assert isinstance(dumped["profile"], dict)
    assert isinstance(dumped["stratification"], dict)
    assert canonical_json_bytes(core) == canonical_json_bytes(rebuilt)
    assert canonical_json_bytes(core) == canonical_json_bytes(core)
