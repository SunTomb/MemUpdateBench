from __future__ import annotations

import re

import pytest
from pydantic import RootModel, ValidationError

from mub.vnext.contracts.common import ContractModel, MemoryObjectKey
from mub.vnext.contracts.enums import (
    Difficulty,
    EventRole,
    Operation,
    SourceType,
    Split,
    TaskFamily,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import validate_gold_replay, validate_task
from mub.vnext.validation.replay import replay_actions

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
    render_core as exported_render_core,
    select_conflicting_values,
    source_id,
    stable_id,
    task_id,
    trajectory_id,
)
from mub.vnext.generation.render import render_core


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


class _GoldProjection(RootModel[object]):
    pass


def test_render_core_is_exported_from_generation_package() -> None:
    assert exported_render_core is render_core


def _normalized_gold_bytes(task: object) -> bytes:
    payload = task.gold.model_dump(mode="json")
    event_indices = {
        event.event_id: index for index, event in enumerate(task.events)
    }
    action_indices = {
        action_identifier: index
        for index, action_identifier in enumerate(task.gold.action_sequence)
    }
    query_indices = {
        query.query_id: index for index, query in enumerate(task.queries)
    }

    for action in payload["actions"]:
        action["action_id"] = f"action[{action_indices[action['action_id']]}]"
        action["event_id"] = f"event[{event_indices[action['event_id']]}]"
    payload["action_sequence"] = [
        f"action[{action_indices[action_identifier]}]"
        for action_identifier in payload["action_sequence"]
    ]
    payload["gold_source_event_ids"] = [
        f"event[{event_indices[event_identifier]}]"
        for event_identifier in payload["gold_source_event_ids"]
    ]
    payload["gold_answers"] = {
        f"query[{query_indices[query_identifier]}]": answer
        for query_identifier, answer in payload["gold_answers"].items()
    }
    payload["acceptable_answers"] = {
        f"query[{query_indices[query_identifier]}]": answers
        for query_identifier, answers in payload["acceptable_answers"].items()
    }
    return canonical_json_bytes(_GoldProjection(root=payload))


def test_render_core_produces_valid_semantically_equivalent_surface_variants() -> None:
    core = _representative_core()
    tasks = [
        render_core(core, split=Split.TEST, surface_variant=variant)
        for variant in range(3)
    ]

    assert len({task.task_id for task in tasks}) == 3
    assert len({task.source.source_id for task in tasks}) == 3
    assert all(
        len({task.events[index].event_id for task in tasks}) == 3
        for index in range(len(core.events))
    )
    assert all(
        len({task.gold.actions[index].action_id for task in tasks}) == 3
        for index in range(len(core.events))
    )
    assert len({task.queries[0].query_id for task in tasks}) == 3

    split_keys = [task.metadata.split_key for task in tasks]
    assert {key.semantic_core_id for key in split_keys} == {core.core_id}
    assert {key.trajectory_id for key in split_keys} == {core.trajectory_id}
    assert len({key.source_group_id for key in split_keys}) == 1
    assert len({key.paraphrase_group_id for key in split_keys}) == 1
    assert len({key.source_document_id for key in split_keys}) == 1
    assert len({key.version_group_id for key in split_keys}) == 1
    assert all(task.metadata.split is Split.TEST for task in tasks)

    assert len({_normalized_gold_bytes(task) for task in tasks}) == 1
    assert len({semantic_task_hash(task) for task in tasks}) == 1
    assert all(
        replay_actions(task.gold.actions).final_state == task.gold.final_state
        for task in tasks
    )
    assert all(
        replay_actions(task.gold.actions).model_dump(mode="json")["version_history"]
        == task.gold.model_dump(mode="json")["version_history"]
        for task in tasks
    )
    assert all(task.gold.gold_answers[task.queries[0].query_id] == "Qingdao" for task in tasks)

    assert len({tuple(event.raw_text for event in task.events) for task in tasks}) == 3
    assert len({tuple(event.speaker for event in task.events) for task in tasks}) == 3
    assert len({task.queries[0].text for task in tasks}) == 3


def test_render_core_preserves_event_action_query_semantics_and_linkage() -> None:
    core = _representative_core()
    task = render_core(core, split=Split.DEV, surface_variant=1)

    assert task.task_family == core.task_family.value
    assert task.difficulty is core.difficulty
    assert [event.sequence_index for event in task.events] == [0, 1]
    assert [event.role for event in task.events] == [
        EventRole.STALE_SAME_SLOT,
        EventRole.LATEST_GOLD,
    ]
    assert [action.operation for action in task.gold.actions] == [
        Operation.ADD,
        Operation.UPDATE,
    ]
    assert [action.value for action in task.gold.actions] == ["Dalian", "Qingdao"]
    assert [
        action.target_object_keys[0].canonical_id for action in task.gold.actions
    ] == ["default|friend:alex|location|"] * 2
    assert [event.metadata["sequence"] for event in task.events] == [0, 1]
    assert task.queries[0].target_object_keys[0].canonical_id == (
        "default|friend:alex|location|"
    )

    for event, action in zip(task.events, task.gold.actions):
        assert event.gold_action_ids == [action.action_id]
        assert action.event_id == event.event_id
    assert task.gold.action_sequence == [
        action.action_id for action in task.gold.actions
    ]
    assert task.gold.gold_source_event_ids == [task.events[1].event_id]
    assert set(task.gold.gold_answers) == {task.queries[0].query_id}
    assert set(task.gold.acceptable_answers) == {task.queries[0].query_id}

    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
    replay = replay_actions(task.gold.actions)
    assert replay.final_state == {
        "default|friend:alex|location|": "Qingdao"
    }
    assert replay.version_history == {
        "default|friend:alex|location|": ("Dalian", "Qingdao")
    }


def test_render_core_builds_deterministic_ids_source_provenance_and_profile() -> None:
    core = _representative_core()
    variant = 2
    task = render_core(core, split=Split.TRAIN, surface_variant=variant)
    expected_task_id = task_id(core.core_id, variant)

    assert task.task_id == expected_task_id
    assert task.source.source_id == source_id(
        "vnext_pilot",
        core.core_index,
        {"semantic_core_id": core.core_id, "surface_variant": variant},
    )
    assert [event.event_id for event in task.events] == [
        event_id(expected_task_id, index) for index in range(len(core.events))
    ]
    assert [action.action_id for action in task.gold.actions] == [
        action_id(expected_task_id, index, 0) for index in range(len(core.events))
    ]
    assert task.queries[0].query_id == query_id(expected_task_id, 0)

    source = task.source
    assert source.source_type is SourceType.SYNTHETIC
    assert source.source_uri == f"memory://{source.source_id}"
    assert source.license_or_privacy == "synthetic_redistributable"
    assert re.fullmatch(r"[0-9a-f]{64}", source.raw_hash or "")
    assert re.fullmatch(r"[0-9a-f]{64}", source.normalized_hash)
    assert source.provenance["redistributable"] is True
    assert source.provenance["semantic_core_id"] == core.core_id
    assert source.provenance["surface_variant"] == variant
    assert source.provenance["source_group_id"] == (
        task.metadata.split_key.source_group_id
    )
    assert source.provenance["source_document_id"] == (
        task.metadata.split_key.source_document_id
    )
    assert source.provenance["version_group_id"] == (
        task.metadata.split_key.version_group_id
    )
    assert source.generator is not None
    assert source.generator.generator_name == "memupdatebench_vnext_pilot"
    assert source.generator.seed == core.core_index
    assert source.generator.config_sha256 == task.metadata.generation_config_hash
    assert source.generator.code_revision == "vnext-pilot-task-2c"
    assert source.generator.compiler_version == task.metadata.compiler_version

    resolved = task.metadata.resolved_profile
    assert resolved["task_family"] == core.task_family.value
    assert resolved["difficulty"] == core.difficulty.value
    assert resolved["profile_name"] == core.difficulty.value
    assert resolved["profile_version"] == "1.0.0"
    assert resolved["update_depth"] == 1
    assert resolved["update_depth_bucket"] == "1"
    assert task.metadata.extra["semantic_core_id"] == core.core_id
    assert task.metadata.extra["core_index"] == core.core_index
    assert task.metadata.extra["surface_variant"] == variant
    assert task.metadata.extra["stratification"] == {"update_depth": 1}


def test_render_core_is_byte_deterministic() -> None:
    core = _representative_core()

    first = render_core(core, split=Split.TEST, surface_variant=0)
    second = render_core(core, split=Split.TEST, surface_variant=0)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)


@pytest.mark.parametrize("variant", [True, False, 0.0, "0", None])
def test_render_core_rejects_non_integer_surface_variants(variant: object) -> None:
    with pytest.raises(TypeError, match="surface_variant must be an integer"):
        render_core(
            _representative_core(),
            split=Split.TEST,
            surface_variant=variant,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("variant", [-1, 3, 4])
def test_render_core_rejects_out_of_range_surface_variants(variant: int) -> None:
    with pytest.raises(ValueError, match="surface_variant must be one of 0, 1, 2"):
        render_core(_representative_core(), split=Split.TEST, surface_variant=variant)


@pytest.mark.parametrize("split", ["test", None, True, 1])
def test_render_core_rejects_noncanonical_splits(split: object) -> None:
    with pytest.raises(TypeError, match="split must be a Split"):
        render_core(
            _representative_core(),
            split=split,  # type: ignore[arg-type]
            surface_variant=0,
        )


def test_render_core_rejects_non_semantic_core_input() -> None:
    with pytest.raises(TypeError, match="core must be a SemanticCore"):
        render_core({}, split=Split.TEST, surface_variant=0)  # type: ignore[arg-type]


def test_render_core_preserves_duplicate_current_role_when_present() -> None:
    payload = _representative_core().model_dump(mode="python")
    payload["events"][1]["role"] = EventRole.DUPLICATE_CURRENT
    core = SemanticCore.model_validate(payload)

    task = render_core(core, split=Split.TEST, surface_variant=0)

    assert task.events[1].role is EventRole.DUPLICATE_CURRENT
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid


def test_render_core_does_not_leak_mutable_aliases() -> None:
    core_payload = _representative_core().model_dump(mode="python")
    core_payload["events"][0]["metadata"] = {
        "sequence": 0,
        "nested": {"tags": ["stale"]},
    }
    core = SemanticCore.model_validate(core_payload)
    first = render_core(core, split=Split.TEST, surface_variant=0)

    first.events[0].metadata["nested"]["tags"].append("changed")
    first.target_objects[0].entity = "friend:changed"
    first.gold.actions[0].target_object_keys[0].entity = "friend:changed-again"

    repeated = render_core(core, split=Split.TEST, surface_variant=0)
    assert repeated.events[0].metadata["nested"]["tags"] == ["stale"]
    assert repeated.target_objects[0].entity == "friend:alex"
    assert repeated.gold.actions[0].target_object_keys[0].entity == "friend:alex"
    assert core.events[0].metadata["nested"]["tags"] == ("stale",)
    assert core.query_targets[0].entity == "friend:alex"


def test_render_core_raises_clear_error_for_unreplayable_core() -> None:
    payload = _representative_core().model_dump(mode="python")
    payload["events"] = [payload["events"][1]]
    core = SemanticCore.model_validate(payload)

    with pytest.raises(ValueError, match="gold replay failed.*UPDATE requires present"):
        render_core(core, split=Split.TEST, surface_variant=0)


def test_render_core_raises_clear_error_for_wrong_core_expected_answer() -> None:
    payload = _representative_core().model_dump(mode="python")
    payload["expected_answer"] = "Dalian"
    core = SemanticCore.model_validate(payload)

    with pytest.raises(ValueError, match="expected_answer.*replayed query answer"):
        render_core(core, split=Split.TEST, surface_variant=0)


def test_render_core_resolves_registered_family_specific_profile_parameters() -> None:
    target = _location_key()
    core = SemanticCore(
        core_id="core_1111111111111111",
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE,
        difficulty=Difficulty.EASY,
        core_index=1,
        trajectory_id="trajectory_1111111111111111",
        events=[
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[target],
                value="Dalian",
                role=EventRole.LATEST_GOLD,
            ),
            CoreEvent(
                operation=Operation.NOOP,
                object_keys=[],
                value=None,
                role=EventRole.NOOP_NEAR_MISS,
            ),
        ],
        query_targets=[target],
        expected_answer="Dalian",
        profile={
            "update_depth": 1,
            "noop_density": 0.0,
            "write_trap_type": "semantic_near_miss",
        },
        stratification={"update_depth": 1},
    )

    task = render_core(core, split=Split.TEST, surface_variant=0)

    assert task.metadata.resolved_profile["write_trap_type"] == (
        "semantic_near_miss"
    )
    assert task.metadata.resolved_profile["noop_density"] == 0.0
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
def test_core_event_model_copy_revalidates_and_freezes_updates() -> None:
    event = _representative_core().events[0]

    with pytest.raises(ValidationError, match="ADD"):
        event.model_copy(update={"value": None})

    source_metadata = {"nested": {"values": [1]}}
    copied = event.model_copy(update={"metadata": source_metadata})
    source_metadata["nested"]["values"].append(2)

    assert copied.metadata["nested"]["values"] == (1,)
    with pytest.raises(TypeError):
        copied.metadata["new"] = True


def test_semantic_core_model_copy_revalidates_and_freezes_updates() -> None:
    core = _representative_core()

    with pytest.raises(ValidationError, match="core_index"):
        core.model_copy(update={"core_index": -1})

    source_profile = {"update_depth": 2, "axes": {"depths": [1, 2]}}
    copied = core.model_copy(update={"profile": source_profile})
    source_profile["axes"]["depths"].append(4)

    assert copied.profile["axes"]["depths"] == (1, 2)
    with pytest.raises(TypeError):
        copied.profile["new"] = True


def test_core_model_copy_preserves_normal_shallow_and_deep_copy_behavior() -> None:
    core = _representative_core()

    shallow = core.model_copy()
    deep = core.model_copy(deep=True)

    assert shallow == core
    assert deep == core
    assert shallow.events is core.events
    assert deep.events is not core.events
