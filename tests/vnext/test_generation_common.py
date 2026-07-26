from __future__ import annotations

import re

import pytest

from mub.vnext.generation import (
    ALIAS_MAPPINGS,
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    SURFACE_TEMPLATE_SETS,
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
