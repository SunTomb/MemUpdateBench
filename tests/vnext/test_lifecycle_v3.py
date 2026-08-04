from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from mub.vnext.contracts.v3.common import MemoryObjectKeyV3
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.scoring.lifecycle_v3 import (
    EntryLifecycleStatusV3,
    TargetLifecycleClassifierV3,
)
from mub.vnext.validation.replay_v3 import (
    ReplayLedgerV3,
    ReplayVersionV3,
    replay_task_v3,
)
from tests.vnext.test_v3_replay_scoring import payload


def _task_with_present_values(*, first, second, current):
    changed = deepcopy(payload())
    for index, value in ((0, first), (1, second), (3, current)):
        changed["actions"][index]["value"] = value
        changed["version_history"][0]["entries"][index]["value"] = value
    changed["gold_evidence"][0]["answer"] = [first, second, None, current]
    return MemUpdateTaskV3.model_validate(changed)


def _relearn_then_update_task():
    changed = deepcopy(payload())
    changed["actions"][0]["value"] = "x"
    changed["actions"][1]["operation"] = "DELETE"
    changed["actions"][1].pop("value")
    changed["actions"][2]["operation"] = "ADD"
    changed["actions"][2]["value"] = "x"
    changed["actions"][3]["operation"] = "UPDATE"
    changed["actions"][3]["value"] = "y"
    statuses_and_values = (("present", "x"), ("tombstone", None), ("present", "x"), ("present", "y"))
    for entry, (status, value) in zip(
        changed["version_history"][0]["entries"], statuses_and_values
    ):
        entry["status"] = status
        if status == "present":
            entry["value"] = value
        else:
            entry.pop("value", None)
    changed["gold_evidence"][0]["answer"] = ["x", None, "x", "y"]
    return MemUpdateTaskV3.model_validate(changed)


def _replace_replay(replay, **updates):
    fields = {name: getattr(replay, name) for name in type(replay).model_fields}
    fields.update(updates)
    return type(replay)(**fields)


def _classifier(task, *, horizon=None):
    replay = replay_task_v3(task)
    assert replay.issues == ()
    if horizon is not None:
        replay = _replace_replay(replay, horizon_logical_time=horizon)
    return TargetLifecycleClassifierV3.for_query(task.queries[0], replay), replay


def _entry(task, entry_id, value, *, version_index=None, source_event_ids=(), key=None):
    return MemoryEntryRecordV3(
        entry_id=entry_id,
        content=entry_id,
        object_key_candidate=key or task.target_objects[0],
        value_candidate=value,
        version_index=version_index,
        source_event_ids=source_event_ids,
    )


def test_lifecycle_status_is_immutable():
    status = EntryLifecycleStatusV3(obsolete=False, stale=False, forgotten=False)
    with pytest.raises(FrozenInstanceError):
        status.stale = True


def test_lifecycle_classifier_isolates_unrelated_ledgers_and_missing_keys():
    task = _task_with_present_values(first="target-old", second="target-middle", current="target-current")
    _, replay = _classifier(task)
    unrelated_key = MemoryObjectKeyV3(
        object_type="slot", namespace="n", entity="other", attribute="a", subkey=None
    )
    unrelated_versions = (
        ReplayVersionV3(
            object_key=unrelated_key,
            version_index=0,
            status="present",
            value="unrelated-old",
            source_action_id="other-a0",
            source_event_ids=("other-e0",),
            logical_time="000",
        ),
        ReplayVersionV3(
            object_key=unrelated_key,
            version_index=1,
            status="present",
            value="unrelated-current",
            source_action_id="other-a1",
            source_event_ids=("other-e1",),
            logical_time="001",
        ),
    )
    replay = _replace_replay(
        replay,
        ledgers=replay.ledgers
        + (ReplayLedgerV3(object_key=unrelated_key, versions=unrelated_versions),),
    )
    classifier = TargetLifecycleClassifierV3.for_query(task.queries[0], replay)

    assert classifier.classify_entry(
        _entry(task, "unrelated", "unrelated-old", version_index=0, key=unrelated_key)
    ) == EntryLifecycleStatusV3(obsolete=False, stale=False, forgotten=False)
    assert classifier.is_stale_value("unrelated-old") is False
    assert classifier.is_forgotten_value("unrelated-old") is False

    missing_key = MemoryEntryRecordV3(entry_id="missing", content="missing", value_candidate="target-old")
    assert classifier.classify_entry(missing_key) == EntryLifecycleStatusV3(
        obsolete=None, stale=None, forgotten=None
    )


def test_lifecycle_classifier_relearned_value_is_obsolete_but_not_stale_or_forgotten():
    task = _task_with_present_values(first="x", second="middle", current="x")
    classifier, _ = _classifier(task)

    status = classifier.classify_entry(_entry(task, "old-x", "x", version_index=0))

    assert status == EntryLifecycleStatusV3(obsolete=True, stale=False, forgotten=False)
    assert classifier.is_stale_value("x") is False
    assert classifier.is_forgotten_value("x") is False


def test_lifecycle_value_predicate_does_not_forget_a_relearned_incarnation():
    task = _relearn_then_update_task()
    classifier, _ = _classifier(task)

    old_status = classifier.classify_entry(
        _entry(task, "old-x-before-delete", "x", version_index=0)
    )

    assert old_status == EntryLifecycleStatusV3(
        obsolete=True, stale=True, forgotten=True
    )
    assert classifier.is_stale_value("x") is True
    assert classifier.is_forgotten_value("x") is False


def test_lifecycle_classifier_marks_stale_value_forgotten_after_active_tombstone():
    task = _task_with_present_values(first="x", second="middle", current="y")
    classifier, _ = _classifier(task)

    status = classifier.classify_entry(_entry(task, "old-x", "x", version_index=0))

    assert status == EntryLifecycleStatusV3(obsolete=True, stale=True, forgotten=True)
    assert classifier.is_stale_value("x") is True
    assert classifier.is_forgotten_value("x") is True


def test_lifecycle_classifier_excludes_future_tombstone_from_horizon():
    task = _task_with_present_values(first="x", second="y", current="z")
    classifier, _ = _classifier(task, horizon="001")

    status = classifier.classify_entry(_entry(task, "old-x", "x", version_index=0))

    assert status == EntryLifecycleStatusV3(obsolete=True, stale=True, forgotten=False)
    assert classifier.is_stale_value("x") is True
    assert classifier.is_forgotten_value("x") is False


def test_lifecycle_classifier_uses_typed_equality_without_hashing_nested_values():
    nested_int = {"nested": [1, {"leaf": ["x"]}]}
    same_nested_int = {"nested": [1, {"leaf": ["x"]}]}
    nested_bool = {"nested": [True, {"leaf": ["x"]}]}

    relearned_task = _task_with_present_values(
        first=nested_int, second={"middle": [0]}, current=same_nested_int
    )
    relearned, _ = _classifier(relearned_task)
    assert relearned.is_stale_value(nested_int) is False
    assert relearned.classify_entry(
        _entry(relearned_task, "nested-relearn", nested_int, version_index=0)
    ).stale is False

    changed_task = _task_with_present_values(
        first=nested_int, second={"middle": [0]}, current=nested_bool
    )
    changed, _ = _classifier(changed_task)
    assert changed.is_stale_value(nested_int) is True
    assert changed.classify_entry(
        _entry(changed_task, "nested-changed", nested_int, version_index=0)
    ).stale is True


def test_lifecycle_classifier_marks_ambiguous_or_inconsistent_provenance_indeterminate():
    task = _task_with_present_values(first="x", second="x", current="y")
    classifier, _ = _classifier(task)

    ambiguous = _entry(task, "ambiguous", "x", source_event_ids=("e0", "e1"))
    inconsistent = _entry(
        task, "inconsistent", "x", version_index=0, source_event_ids=("e1",)
    )

    indeterminate = EntryLifecycleStatusV3(obsolete=None, stale=None, forgotten=None)
    assert classifier.classify_entry(ambiguous) == indeterminate
    assert classifier.classify_entry(inconsistent) == indeterminate


def test_lifecycle_classifier_marks_target_missing_from_replay_indeterminate():
    task = _task_with_present_values(first="x", second="middle", current="y")
    replay = _replace_replay(replay_task_v3(task), ledgers=())
    classifier = TargetLifecycleClassifierV3.for_query(task.queries[0], replay)

    assert classifier.classify_entry(
        _entry(task, "unknown-target", "x", version_index=0)
    ) == EntryLifecycleStatusV3(obsolete=None, stale=None, forgotten=None)
    assert classifier.is_stale_value("x") is False
    assert classifier.is_forgotten_value("x") is False
