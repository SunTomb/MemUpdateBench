from __future__ import annotations

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import AnswerSchema, Difficulty, EvaluationMode, MemoryObjectKey, Operation, ActionScope
from mub.vnext.contracts.task import GoldAction, MemUpdateTask


def _undeclared_key() -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="friend:alex",
        attribute="office",
    )


def _same_slot_different_object_type() -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="friend:alex",
        attribute="location",
    )


def test_fixed_task_json_round_trip_preserves_design_fields_and_action_order(make_task) -> None:
    task = make_task()

    round_tripped = MemUpdateTask.model_validate_json(task.model_dump_json())

    assert round_tripped == task
    assert round_tripped.task_family == "repeated_same_slot_update"
    assert round_tripped.metadata.split_key.source_group_id == "source_group_0001"
    assert round_tripped.metadata.split_key.trajectory_id == "trajectory_0001"
    assert round_tripped.target_objects[0].object_type == "slot"
    assert round_tripped.gold.action_sequence == ["action_0", "action_1"]
    assert round_tripped.gold.actions[1].operation == Operation.UPDATE
    assert round_tripped.gold.gold_answers == {"query_0": "Qingdao"}
    assert round_tripped.gold.acceptable_answers == {"query_0": ["Qingdao"]}


def test_future_task_family_string_is_accepted(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["task_family"] = "future_family_not_in_helper_enum"

    task = MemUpdateTask.model_validate(data)

    assert task.task_family == "future_family_not_in_helper_enum"


def test_blank_task_family_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["task_family"] = "   "

    with pytest.raises(ValidationError, match="task_family must not be blank"):
        MemUpdateTask.model_validate(data)


def test_inline_gold_query_field_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["queries"][0]["expected_answer"] = "Qingdao"

    with pytest.raises(ValidationError) as exc_info:
        MemUpdateTask.model_validate(data)

    message = str(exc_info.value)
    assert "expected_answer" in message
    assert "Extra inputs are not permitted" in message


def test_duplicated_action_sequence_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["action_sequence"] = ["action_0", "action_0"]

    with pytest.raises(ValidationError, match="action_sequence must contain every action exactly once"):
        MemUpdateTask.model_validate(data)


def test_missing_event_action_reference_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["events"][0]["gold_action_ids"] = ["missing"]

    with pytest.raises(ValidationError, match="missing gold action"):
        MemUpdateTask.model_validate(data)


def test_missing_gold_source_event_reference_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["gold_source_event_ids"] = ["missing_event"]

    with pytest.raises(ValidationError, match="gold_source_event_ids references missing event"):
        MemUpdateTask.model_validate(data)


def test_undeclared_query_target_object_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["queries"][0]["target_object_keys"] = [_undeclared_key().model_dump(mode="json")]

    with pytest.raises(ValidationError, match="query query_0 targets undeclared object"):
        MemUpdateTask.model_validate(data)


def test_action_target_reference_ignores_object_type(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["actions"][0]["target_object_keys"] = [_same_slot_different_object_type().model_dump(mode="json")]

    task = MemUpdateTask.model_validate(data)

    assert task.gold.actions[0].target_object_keys[0].object_type == "profile"


def test_query_target_reference_ignores_object_type(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["queries"][0]["target_object_keys"] = [_same_slot_different_object_type().model_dump(mode="json")]

    task = MemUpdateTask.model_validate(data)

    assert task.queries[0].target_object_keys[0].object_type == "profile"


def test_expected_present_reference_ignores_object_type(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["expected_present_objects"] = [_same_slot_different_object_type().model_dump(mode="json")]

    task = MemUpdateTask.model_validate(data)

    assert task.gold.expected_present_objects[0].object_type == "profile"


def test_expected_absent_reference_ignores_object_type(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["expected_absent_objects"] = [_same_slot_different_object_type().model_dump(mode="json")]

    task = MemUpdateTask.model_validate(data)

    assert task.gold.expected_absent_objects[0].object_type == "profile"


def test_undeclared_action_target_object_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["actions"][0]["target_object_keys"] = [_undeclared_key().model_dump(mode="json")]

    with pytest.raises(ValidationError, match="action action_0 targets undeclared object"):
        MemUpdateTask.model_validate(data)


def test_undeclared_expected_present_object_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["expected_present_objects"] = [_undeclared_key().model_dump(mode="json")]

    with pytest.raises(ValidationError, match="expected_present_objects contains undeclared object"):
        MemUpdateTask.model_validate(data)


def test_undeclared_expected_absent_object_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["expected_absent_objects"] = [_undeclared_key().model_dump(mode="json")]

    with pytest.raises(ValidationError, match="expected_absent_objects contains undeclared object"):
        MemUpdateTask.model_validate(data)


def test_missing_gold_answer_mapping_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["gold_answers"] = {}

    with pytest.raises(ValidationError, match="gold_answers must contain every query ID"):
        MemUpdateTask.model_validate(data)


def test_missing_acceptable_answer_mapping_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["acceptable_answers"] = {}

    with pytest.raises(ValidationError, match="acceptable_answers must contain every query ID"):
        MemUpdateTask.model_validate(data)


def test_unknown_query_answer_key_is_rejected(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["gold"]["gold_answers"]["unknown_query"] = "Qingdao"

    with pytest.raises(ValidationError, match="gold_answers contains unknown query ID"):
        MemUpdateTask.model_validate(data)


def test_canonical_query_and_profile_values(make_task) -> None:
    task = make_task()

    assert task.queries[0].answer_schema == AnswerSchema.STRING
    assert task.queries[0].evaluation_mode == EvaluationMode.RETRIEVED_PROMPT
    assert task.metadata.profile_name == Difficulty.EASY


def test_gold_action_valid_shapes_are_accepted(make_task) -> None:
    key = make_task().target_objects[0]

    valid_actions = [
        GoldAction(
            action_id="noop",
            event_id="event_0",
            operation=Operation.NOOP,
            scope=ActionScope.ATTRIBUTE,
        ),
        GoldAction(
            action_id="add",
            event_id="event_0",
            operation=Operation.ADD,
            scope=ActionScope.ATTRIBUTE,
            target_object_keys=[key],
            value="Dalian",
        ),
        GoldAction(
            action_id="update",
            event_id="event_1",
            operation=Operation.UPDATE,
            scope=ActionScope.ATTRIBUTE,
            target_object_keys=[key],
            value="Qingdao",
        ),
        GoldAction(
            action_id="delete",
            event_id="event_1",
            operation=Operation.DELETE,
            scope=ActionScope.ATTRIBUTE,
            target_object_keys=[key],
        ),
    ]

    assert [action.operation for action in valid_actions] == [
        Operation.NOOP,
        Operation.ADD,
        Operation.UPDATE,
        Operation.DELETE,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation": Operation.NOOP, "target_object_keys": "target"},
        {"operation": Operation.NOOP, "value": "Dalian"},
        {"operation": Operation.ADD},
        {"operation": Operation.UPDATE},
        {"operation": Operation.DELETE},
    ],
)
def test_gold_action_invalid_shapes_are_rejected(make_task, kwargs) -> None:
    key = make_task().target_objects[0]
    action_kwargs = {
        "action_id": "action_shape",
        "event_id": "event_0",
        "scope": ActionScope.ATTRIBUTE,
        **kwargs,
    }
    if action_kwargs.get("target_object_keys") == "target":
        action_kwargs["target_object_keys"] = [key]

    with pytest.raises(ValidationError):
        GoldAction(**action_kwargs)


def test_top_level_legacy_num_updates_is_forbidden(make_task) -> None:
    data = make_task().model_dump(mode="json")
    data["num_updates"] = 2

    with pytest.raises(ValidationError) as exc_info:
        MemUpdateTask.model_validate(data)

    message = str(exc_info.value)
    assert "num_updates" in message
    assert "Extra inputs are not permitted" in message
