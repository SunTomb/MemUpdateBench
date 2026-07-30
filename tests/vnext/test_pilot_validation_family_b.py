from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytest

from mub.vnext.contracts import EventRole, MemoryObjectKey, Operation, Split, TaskFamily
from mub.vnext.contracts.task import GoldAction, MemUpdateTask
from mub.vnext.generation import (
    GenerationContext,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.validation import (
    replay_actions,
    validate_distractors,
    validate_family_b_task,
    validate_family_c_task,
    validate_pilot_task,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def family_b_tasks(config):
    context = GenerationContext(config=config, code_revision="family-b-validation-test")
    return [
        render_core(core, split=Split.TEST, surface_variant=variant, context=context)
        for core in generate_family_b_cores(config)
        for variant in range(3)
    ]


def _render_first(config, generator, family: str):
    context = GenerationContext(config=config, code_revision=f"{family}-fallback-test")
    return render_core(
        generator(config)[0],
        split=Split.TEST,
        surface_variant=0,
        context=context,
    )


def _codes(report):
    return {issue.code for issue in report.issues}


def test_validate_family_b_task_accepts_all_surfaces_and_explicit_dispatches(family_b_tasks):
    assert len(family_b_tasks) == 360
    assert {task.metadata.extra["surface_variant"] for task in family_b_tasks} == {0, 1, 2}
    for task in family_b_tasks:
        direct = validate_family_b_task(task)
        explicit = validate_pilot_task(task)
        assert direct.valid, (task.task_id, direct.issues)
        assert direct.issues == ()
        assert explicit == direct


def test_validate_family_b_task_is_inapplicable_to_other_exact_families(config):
    for generator, family in (
        (generate_family_a_cores, "a"),
        (generate_family_c_cores, "c"),
        (generate_family_d_cores, "d"),
    ):
        task = _render_first(config, generator, family)
        report = validate_family_b_task(task)
        assert _codes(report) == {"family_b_inapplicable_task_family"}


def test_family_c_explicit_validation_dispatches_strict_validator(config):
    task = _render_first(config, generate_family_c_cores, "c")
    assert validate_pilot_task(task) == validate_family_c_task(task)


def test_validate_family_b_task_rejects_non_task_and_malformed_family():
    invalid = validate_family_b_task(object())
    assert _codes(invalid) == {"family_b_invalid_task_type"}

    malformed = MemUpdateTask.model_construct(
        task_family=TaskFamily.INTERLEAVED_MULTI_SLOT.value
    )
    direct = validate_family_b_task(malformed)
    explicit = validate_pilot_task(malformed)
    assert "family_b_malformed_record" in _codes(direct)
    assert explicit == direct


class HostileFamilyString(str):
    __hash__ = str.__hash__

    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.override_access_count = 0
        return instance

    def _fail(self):
        self.override_access_count += 1
        raise RuntimeError("hostile family override executed")

    def __eq__(self, other):
        return self._fail()

    def __str__(self):
        return self._fail()

    def strip(self, *args, **kwargs):
        return self._fail()


class ExplosiveSequence(Sequence):
    def __init__(self):
        self.iteration_count = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError(f"explosive item {index}")

    def __iter__(self):
        self.iteration_count += 1
        raise RuntimeError("explosive iteration")


def _payload(task):
    return task.model_dump(mode="json")


def _family_b_task(family_b_tasks, *, pattern="round_robin", difficulty="hard", depth=4, variant=0):
    return next(
        task
        for task in family_b_tasks
        if task.metadata.extra["stratification"]["interleaving_pattern"] == pattern
        and task.difficulty.value == difficulty
        and task.metadata.resolved_profile["update_depth"] == depth
        and task.metadata.extra["surface_variant"] == variant
    )


def _action_for_event(payload, event):
    action_id = event["gold_action_ids"][0]
    return next(
        action for action in payload["gold"]["actions"] if action["action_id"] == action_id
    )


def _canonical_id(key_payload):
    return MemoryObjectKey.model_validate(key_payload).canonical_id


def _rewrite_replay(payload):
    action_by_id = {
        action["action_id"]: action for action in payload["gold"]["actions"]
    }
    payload["gold"]["action_sequence"] = [
        action_id
        for event in payload["events"]
        for action_id in event["gold_action_ids"]
    ]
    ordered_actions = [
        GoldAction.model_validate(action_by_id[action_id])
        for action_id in payload["gold"]["action_sequence"]
    ]
    replay = replay_actions(ordered_actions).model_dump(mode="json")
    payload["gold"]["final_state"] = replay["final_state"]
    payload["gold"]["version_history"] = replay["version_history"]


def _reorder_events(payload, ordered_events):
    payload["events"] = ordered_events
    for index, event in enumerate(payload["events"]):
        event["sequence_index"] = index
        event["source_anchor"] = {"event_index": index}
    _rewrite_replay(payload)


def _replace_identity_everywhere(payload, old_key, new_key):
    old_id = _canonical_id(old_key)
    new_id = _canonical_id(new_key)
    for action in payload["gold"]["actions"]:
        action["target_object_keys"] = [
            deepcopy(new_key) if key == old_key else key
            for key in action["target_object_keys"]
        ]
    for field in ("target_objects",):
        payload[field] = [deepcopy(new_key) if key == old_key else key for key in payload[field]]
    payload["gold"]["expected_present_objects"] = [
        deepcopy(new_key) if key == old_key else key
        for key in payload["gold"]["expected_present_objects"]
    ]
    if old_id in payload["gold"]["final_state"]:
        payload["gold"]["final_state"][new_id] = payload["gold"]["final_state"].pop(old_id)
    if old_id in payload["gold"]["version_history"]:
        payload["gold"]["version_history"][new_id] = payload["gold"]["version_history"].pop(old_id)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("operation", "family_b_target_chain_corruption"),
        ("role", "family_b_target_chain_corruption"),
        ("version", "family_b_target_chain_corruption"),
        ("value", "family_b_target_value_corruption"),
    ),
)
def test_family_b_rejects_target_chain_corruption(family_b_tasks, mutation, expected_code):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    target_events = [
        event for event in payload["events"] if event["metadata"]["slot_index"] == 0
    ]
    event = target_events[1]
    action = _action_for_event(payload, event)
    if mutation == "operation":
        action["operation"] = Operation.ADD.value
    elif mutation == "role":
        event["role"] = EventRole.NEUTRAL.value
    elif mutation == "version":
        event["metadata"]["version_index"] = 9
    else:
        final_event = target_events[-1]
        final_action = _action_for_event(payload, final_event)
        final_action["value"] = _action_for_event(payload, target_events[0])["value"]
        _rewrite_replay(payload)
        query_id = payload["queries"][0]["query_id"]
        payload["gold"]["gold_answers"][query_id] = final_action["value"]
        payload["gold"]["acceptable_answers"][query_id] = final_action["value"]
    corrupted = MemUpdateTask.model_validate(payload)
    assert expected_code in _codes(validate_family_b_task(corrupted))


@pytest.mark.parametrize("destination_slot", (0, 2))
def test_family_b_rejects_coordinated_non_target_retargeting(
    family_b_tasks, destination_slot
):
    task = _family_b_task(family_b_tasks, pattern="round_robin")
    payload = _payload(task)
    source_event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 1
        and event["metadata"]["version_index"] == 1
    )
    destination_event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == destination_slot
    )
    source_action = _action_for_event(payload, source_event)
    destination_action = _action_for_event(payload, destination_event)
    source_action["target_object_keys"] = deepcopy(destination_action["target_object_keys"])
    _rewrite_replay(payload)
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_non_target_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


@pytest.mark.parametrize("gold_field", ("final_state", "version_history"))
def test_family_b_rejects_one_corrupted_non_target_gold_field(
    family_b_tasks, gold_field
):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    target_id = _canonical_id(payload["queries"][0]["target_object_keys"][0])
    non_target_id = next(
        object_id for object_id in payload["gold"][gold_field] if object_id != target_id
    )
    if gold_field == "final_state":
        payload["gold"][gold_field][non_target_id] = "corrupted-current"
    else:
        payload["gold"][gold_field][non_target_id][-1] = "corrupted-history"
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_non_target_state_history_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_collapsed_active_identities(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    target_objects = list(task.target_objects)
    target_objects[2] = target_objects[1]
    expected_present = list(task.gold.expected_present_objects)
    expected_present[2] = expected_present[1]
    corrupted_gold = type(task.gold).model_construct(
        **{**task.gold.__dict__, "expected_present_objects": expected_present}
    )
    corrupted = MemUpdateTask.model_construct(
        **{
            **task.__dict__,
            "target_objects": target_objects,
            "gold": corrupted_gold,
        }
    )
    assert "family_b_active_identity_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_unreviewed_attribute_geometry(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    slot_event = next(
        event for event in payload["events"] if event["metadata"]["slot_index"] == 1
    )
    old_key = deepcopy(_action_for_event(payload, slot_event)["target_object_keys"][0])
    new_key = deepcopy(old_key)
    new_key["attribute"] = "unreviewed_attribute"
    _replace_identity_everywhere(payload, old_key, new_key)
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_active_identity_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_non_target_current_equal_to_target_gold(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    query_id = payload["queries"][0]["query_id"]
    target_gold = payload["gold"]["gold_answers"][query_id]
    non_target_final = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 1
        and event["metadata"]["version_metadata"] == "latest"
    )
    _action_for_event(payload, non_target_final)["value"] = target_gold
    _rewrite_replay(payload)
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_non_target_current_gold_collision" in _codes(
        validate_family_b_task(corrupted)
    )


@pytest.mark.parametrize("change", ("missing", "additional"))
def test_family_b_rejects_missing_or_additional_declared_slot(family_b_tasks, change):
    task = _family_b_task(family_b_tasks)
    if change == "missing":
        corrupted = MemUpdateTask.model_construct(
            **{**task.__dict__, "target_objects": list(task.target_objects[:-1])}
        )
    else:
        payload = _payload(task)
        extra = deepcopy(payload["target_objects"][-1])
        extra["attribute"] = "additional_slot"
        payload["target_objects"].append(extra)
        corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_active_identity_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("version_index", 7),
        ("version_metadata", "stale"),
        ("distractor_kind", "active_non_target"),
        ("target_relation", "target"),
    ),
)
def test_family_b_rejects_non_target_version_metadata_corruption(
    family_b_tasks, field, value
):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 1
        and event["metadata"]["version_index"] == 1
    )
    event["metadata"][field] = value
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_non_target_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_cross_slot_expected_effect_claim(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 1
        and event["metadata"]["version_index"] == 1
    )
    _action_for_event(payload, event)["expected_effect"] = {
        "mutated_slot": "another"
    }
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_non_target_corruption" in _codes(
        validate_family_b_task(corrupted)
    )


@pytest.mark.parametrize("pattern", ("round_robin", "burst", "adversarial_adjacent"))
def test_family_b_rejects_coordinated_pattern_order_mutation(family_b_tasks, pattern):
    task = _family_b_task(family_b_tasks, pattern=pattern)
    payload = _payload(task)
    events = payload["events"]
    if pattern == "round_robin":
        first = next(i for i, event in enumerate(events) if event["metadata"]["slot_index"] == 1)
        second = next(i for i, event in enumerate(events) if event["metadata"]["slot_index"] == 2)
        events[first], events[second] = events[second], events[first]
    else:
        slot_one = [event for event in events if event["metadata"]["slot_index"] == 1]
        slot_two = [event for event in events if event["metadata"]["slot_index"] == 2]
        first = min(events.index(event) for event in slot_one)
        selected = {id(event) for event in (*slot_one, *slot_two)}
        remainder = [event for event in events if id(event) not in selected]
        payload["events"] = remainder[:first] + slot_two + slot_one + remainder[first:]
    _reorder_events(payload, payload["events"])
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_interleaving_pattern_mismatch" in _codes(
        validate_family_b_task(corrupted)
    )


@pytest.mark.parametrize(
    ("location", "field", "value"),
    (
        ("stratification", "cross_slot_distractor_count", 0),
        ("stratification", "cross_slot_distractor_density", 0.25),
        ("stratification", "base_event_count", 999),
        ("stratification", "num_events", 999),
        ("stratification", "allocation_cell_deviation", 999.0),
        ("stratification", "difficulty_allocation_count", "39"),
        ("stratification", "axis_product_size", 0),
        ("profile", "cross_slot_interleaving", 0.25),
        ("profile", "context_order", "reverse_chronological"),
        ("profile", "active_object_count", 2),
        ("profile", "interleaving_pattern", "burst"),
        ("stratification", "num_updates", 4),
    ),
)
def test_family_b_rejects_density_count_and_profile_rewrites(
    family_b_tasks, location, field, value
):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    target = (
        payload["metadata"]["extra"]["stratification"]
        if location == "stratification"
        else payload["metadata"]["resolved_profile"]
    )
    target[field] = value
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_counter_profile_mismatch" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_noop_injection(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 1
        and event["metadata"]["version_metadata"] == "latest"
    )
    action = _action_for_event(payload, event)
    action["operation"] = Operation.NOOP.value
    action["target_object_keys"] = []
    action["value"] = None
    event["role"] = EventRole.NOOP_NEAR_MISS.value
    _rewrite_replay(payload)
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_noop_forbidden" in _codes(validate_family_b_task(corrupted))


def test_family_b_rejects_multiple_current_answers(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    query_id = payload["queries"][0]["query_id"]
    payload["gold"]["acceptable_answers"][query_id] = [
        payload["gold"]["gold_answers"][query_id],
        "alternate",
    ]
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_b_multiple_current_answers" in _codes(
        validate_family_b_task(corrupted)
    )


def test_family_b_rejects_target_key_and_count_rewrites(family_b_tasks):
    task = _family_b_task(family_b_tasks, pattern="round_robin")
    payload = _payload(task)
    target_events = [
        event for event in payload["events"] if event["metadata"]["slot_index"] == 0
    ]
    non_target_event = next(
        event for event in payload["events"] if event["metadata"]["slot_index"] == 1
    )
    final_event = target_events[-1]
    _action_for_event(payload, final_event)["target_object_keys"] = deepcopy(
        _action_for_event(payload, non_target_event)["target_object_keys"]
    )
    _rewrite_replay(payload)
    query_id = payload["queries"][0]["query_id"]
    target_id = _canonical_id(payload["queries"][0]["target_object_keys"][0])
    current = payload["gold"]["final_state"][target_id]
    payload["gold"]["gold_answers"][query_id] = current
    payload["gold"]["acceptable_answers"][query_id] = current
    payload["gold"]["gold_source_event_ids"] = [target_events[-2]["event_id"]]
    retargeted = MemUpdateTask.model_validate(payload)
    assert "family_b_target_chain_corruption" in _codes(
        validate_family_b_task(retargeted)
    )

    payload = _payload(task)
    final_event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 0
        and event["metadata"]["version_metadata"] == "latest"
    )
    final_action_id = final_event["gold_action_ids"][0]
    payload["events"].remove(final_event)
    payload["gold"]["actions"] = [
        action
        for action in payload["gold"]["actions"]
        if action["action_id"] != final_action_id
    ]
    _reorder_events(payload, payload["events"])
    target_events = [
        event for event in payload["events"] if event["metadata"]["slot_index"] == 0
    ]
    query_id = payload["queries"][0]["query_id"]
    target_id = _canonical_id(payload["queries"][0]["target_object_keys"][0])
    current = payload["gold"]["final_state"][target_id]
    payload["gold"]["gold_answers"][query_id] = current
    payload["gold"]["acceptable_answers"][query_id] = current
    payload["gold"]["gold_source_event_ids"] = [target_events[-1]["event_id"]]
    shortened = MemUpdateTask.model_validate(payload)
    assert "family_b_target_chain_corruption" in _codes(
        validate_family_b_task(shortened)
    )


def test_family_b_rejects_missing_non_target_trajectory(family_b_tasks):
    task = _family_b_task(family_b_tasks, pattern="burst")
    payload = _payload(task)
    removed_events = [
        event for event in payload["events"] if event["metadata"]["slot_index"] == 7
    ]
    removed_action_ids = {
        action_id for event in removed_events for action_id in event["gold_action_ids"]
    }
    removed_key = deepcopy(_action_for_event(payload, removed_events[0])["target_object_keys"][0])
    payload["events"] = [event for event in payload["events"] if event not in removed_events]
    payload["gold"]["actions"] = [
        action
        for action in payload["gold"]["actions"]
        if action["action_id"] not in removed_action_ids
    ]
    payload["target_objects"] = [key for key in payload["target_objects"] if key != removed_key]
    payload["gold"]["expected_present_objects"] = [
        key for key in payload["gold"]["expected_present_objects"] if key != removed_key
    ]
    _reorder_events(payload, payload["events"])
    corrupted = MemUpdateTask.model_validate(payload)
    codes = _codes(validate_family_b_task(corrupted))
    assert "family_b_active_identity_corruption" in codes
    assert "family_b_non_target_corruption" in codes


def test_family_b_allocation_metadata_is_task_local_not_corpus_policy(family_b_tasks):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    stratification = payload["metadata"]["extra"]["stratification"]
    payload["metadata"]["extra"]["core_index"] = 999
    stratification["axis_product_index"] = 5
    stratification["axis_product_size"] = 6
    stratification["pattern_group_index"] = 100
    stratification["allocation_cell_count"] = 7
    stratification["allocation_cell_ideal"] = 6.5
    stratification["allocation_cell_deviation"] = 0.5
    stratification["difficulty_allocation_count"] = 11
    stratification["difficulty_allocation_ideal"] = 10.5
    stratification["difficulty_allocation_deviation"] = 0.5
    rewritten = MemUpdateTask.model_validate(payload)
    report = validate_family_b_task(rewritten)
    assert report.valid, report.issues


def test_family_b_structured_distractor_overlap_policy_preserves_real_leaks(
    family_b_tasks
):
    intentional = next(
        task
        for task in family_b_tasks
        if "distractor_text_contains_accepted_answer"
        in _codes(validate_distractors(task))
    )
    specialized = validate_distractors(
        intentional,
        allow_superseded_non_target_answer_overlap=True,
    )
    assert "distractor_text_contains_accepted_answer" not in _codes(specialized)
    assert validate_family_b_task(intentional).valid

    payload = _payload(intentional)
    query_id = payload["queries"][0]["query_id"]
    target_gold = payload["gold"]["gold_answers"][query_id]
    leaking_event = next(
        event
        for event in payload["events"]
        if event["role"] == EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
        and _action_for_event(payload, event)["value"] != target_gold
    )
    leaking_event["raw_text"] += f" Unrelated leak: {target_gold}."
    leaking_event["normalized_text"] += f" Unrelated leak: {target_gold}."
    leaking_event["metadata"]["allow_accepted_answer_ambiguity"] = True
    leaking = MemUpdateTask.model_validate(payload)
    assert "distractor_text_contains_accepted_answer" in _codes(
        validate_distractors(leaking)
    )
    report = validate_family_b_task(leaking)
    assert "distractor_text_contains_accepted_answer" in _codes(report)


def test_family_b_forged_role_and_history_cannot_authorize_answer_leak(
    family_b_tasks,
):
    task = _family_b_task(family_b_tasks)
    payload = _payload(task)
    query_id = payload["queries"][0]["query_id"]
    target_gold = payload["gold"]["gold_answers"][query_id]
    target_id = _canonical_id(payload["queries"][0]["target_object_keys"][0])
    target_event = next(
        event
        for event in payload["events"]
        if event["metadata"]["slot_index"] == 0
        and event["metadata"]["version_index"] == 0
    )
    target_action = _action_for_event(payload, target_event)
    target_action["value"] = target_gold
    payload["gold"]["version_history"][target_id][0] = target_gold
    target_event["role"] = EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
    target_event["metadata"]["allow_accepted_answer_ambiguity"] = True
    target_event["raw_text"] += f" Forged current answer: {target_gold}."
    target_event["normalized_text"] += f" Forged current answer: {target_gold}."
    forged = MemUpdateTask.model_validate(payload)

    specialized = validate_distractors(
        forged,
        allow_superseded_non_target_answer_overlap=True,
    )
    assert "distractor_text_contains_accepted_answer" in _codes(specialized)
    report = validate_family_b_task(forged)
    assert "distractor_text_contains_accepted_answer" in _codes(report)
    assert "family_b_target_chain_corruption" in _codes(report)


def test_family_b_preflight_is_hostile_safe_bounded_and_dispatch_equivalent(
    family_b_tasks
):
    task = _family_b_task(family_b_tasks)
    hostile = HostileFamilyString(TaskFamily.INTERLEAVED_MULTI_SLOT.value)
    hostile_task = MemUpdateTask.model_construct(**{**task.__dict__, "task_family": hostile})
    direct = validate_family_b_task(hostile_task)
    explicit = validate_pilot_task(hostile_task)
    assert direct == explicit
    assert "family_b_invalid_field_type" in _codes(direct)
    assert hostile.override_access_count == 0
    assert len(direct.issues) <= 128

    explosive = ExplosiveSequence()
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "events": explosive})
    first = validate_family_b_task(malformed)
    second = validate_family_b_task(malformed)
    assert first == second
    assert "family_b_malformed_collection" in _codes(first)
    assert explosive.iteration_count == 0
    assert len(first.issues) <= 128
