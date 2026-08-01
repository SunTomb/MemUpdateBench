from __future__ import annotations

import pytest

from mub.vnext.adapters.corrupted import (
    CONTROL_ADAPTERS,
    AlwaysAddAdapter,
    AlwaysNoopAdapter,
    CurrentNotRetrievedAdapter,
    GoldRetrievedWrongAnswerAdapter,
    InvalidFormatterAdapter,
    StaleValueCopierAdapter,
    WrongAttributeWriterAdapter,
    WrongEntityWriterAdapter,
    build_corrupted_adapter,
)
from mub.vnext.failure import FailureFlag


EXPECTED = {
    "control/always_add": {FailureFlag.FALSE_WRITE.value},
    "control/always_noop": {FailureFlag.MISSED_UPDATE.value},
    "control/stale_value_copier": {FailureFlag.STALE_COPIED.value},
    "control/wrong_entity_writer": {FailureFlag.WRONG_ENTITY.value},
    "control/wrong_attribute_writer": {FailureFlag.WRONG_ATTRIBUTE.value},
    "control/invalid_formatter": {FailureFlag.INVALID_ACTION_FORMAT.value},
    "control/current_not_retrieved": {FailureFlag.CURRENT_NOT_RETRIEVED.value},
    "control/gold_retrieved_wrong_answer": {FailureFlag.GOLD_RETRIEVED_WRONG_ANSWER.value},
}
UNRELATED = {
    "wrong_operation",
    "wrong_entity",
    "wrong_attribute",
    "invalid_action_format",
    "missed_update",
    "false_write",
    "current_not_retrieved",
    "stale_copied",
    "gold_retrieved_wrong_answer",
}


def _flags_from_trace(adapter) -> set[str]:
    return {
        flag
        for record in adapter.export_action_trace()
        for flag in record.get("failure_flags", [])
    }


def _run(adapter, task):
    assert adapter.reset("control-test", {}).success
    logs = [adapter.ingest_event(event) for event in task.events]
    return logs


@pytest.mark.parametrize(
    ("adapter_id", "expected"),
    sorted(EXPECTED.items()),
)
def test_each_control_emits_only_its_expected_failure_signal(make_task, adapter_id, expected):
    task = make_task()
    adapter = build_corrupted_adapter(adapter_id, task=task)
    _run(adapter, task)

    if adapter_id == "control/current_not_retrieved":
        result = adapter.retrieve(task.queries[0], 8)
        observed = set(result.raw_result["failure_flags"])
    elif adapter_id == "control/stale_value_copier":
        observed = set(adapter.answer(task.queries[0], "slot_direct").error["failure_flags"])
    elif adapter_id == "control/gold_retrieved_wrong_answer":
        observed = set(adapter.answer(task.queries[0], "slot_direct").error["failure_flags"])
    elif adapter_id == "control/invalid_formatter":
        observed = _flags_from_trace(adapter)
    else:
        observed = _flags_from_trace(adapter)

    assert observed == expected
    assert not (observed & (UNRELATED - expected))


def test_controls_have_stable_smoke_metadata_and_canonical_info(make_task):
    task = make_task()
    left = build_corrupted_adapter("control/always_add", task=task)
    right = build_corrupted_adapter("control/always_add", task=task)

    assert set(CONTROL_ADAPTERS) == set(EXPECTED)
    assert left.smoke_control is True
    assert left.leaderboard_eligible is False
    assert left.control_metadata() == {
        "control_id": "control/always_add",
        "smoke_control": True,
        "leaderboard_eligible": False,
        "expected_failure_flags": ["false_write"],
    }
    assert left.adapter_info().model_dump(mode="json") == right.adapter_info().model_dump(mode="json")
    assert left.capabilities().model_dump(mode="json") == right.capabilities().model_dump(mode="json")


def test_controls_reset_isolated_state_and_trace(make_task):
    task = make_task()
    adapter = AlwaysAddAdapter(task=task)
    _run(adapter, task)
    assert adapter.export_entries()
    assert adapter.export_action_trace()

    assert adapter.reset("new-namespace", {}).success
    assert adapter.export_entries() == []
    assert adapter.export_action_trace() == []
    assert adapter.export_raw_state()["state_by_object"] == {}


def test_control_answers_preserve_expected_layer_behavior(make_task):
    task = make_task()

    stale = StaleValueCopierAdapter(task=task)
    _run(stale, task)
    stale_answer = stale.answer(task.queries[0], "slot_direct")
    assert stale_answer.value == "Dalian"
    assert stale_answer.error["failure_flags"] == ["stale_copied"]

    wrong = GoldRetrievedWrongAnswerAdapter(task=task)
    _run(wrong, task)
    retrieval = wrong.retrieve(task.queries[0], 8)
    assert retrieval.entries and retrieval.entries[-1].value_candidate == "Qingdao"
    wrong_answer = wrong.answer(task.queries[0], "slot_direct")
    assert wrong_answer.value != task.gold.gold_answers[task.queries[0].query_id]
    assert wrong_answer.error["failure_flags"] == ["gold_retrieved_wrong_answer"]

    hidden = CurrentNotRetrievedAdapter(task=task)
    _run(hidden, task)
    retrieval = hidden.retrieve(task.queries[0], 8)
    assert retrieval.entries == []
    assert retrieval.raw_result["failure_flags"] == ["current_not_retrieved"]


def test_malformed_and_unsupported_inputs_fail_closed_without_mutation(make_task):
    task = make_task()
    adapter = AlwaysAddAdapter(task=task)
    assert adapter.reset("safe", {}).success
    before = adapter.export_raw_state()

    with pytest.raises(TypeError):
        adapter.ingest_event(object())
    with pytest.raises(TypeError):
        adapter.retrieve(object(), 1)
    assert adapter.export_raw_state() == before

    malformed = task.events[0].model_copy(update={"raw_text": "not an action", "normalized_text": "not an action", "gold_action_ids": []})
    log = adapter.ingest_event(malformed)
    assert log.error is not None
    assert adapter.export_entries() == []
    assert adapter.export_raw_state()["state_by_object"] == before["state_by_object"]


def test_invalid_formatter_never_writes_on_valid_or_malformed_events(make_task):
    task = make_task()
    adapter = InvalidFormatterAdapter(task=task)
    logs = _run(adapter, task)
    assert all(log.error["code"] == "invalid_action_format" for log in logs)
    assert adapter.export_entries() == []
    assert _flags_from_trace(adapter) == {"invalid_action_format"}
