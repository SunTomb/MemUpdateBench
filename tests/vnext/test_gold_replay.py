from __future__ import annotations

from copy import deepcopy

import pytest

from mub.vnext.contracts import (
    ActionScope,
    AnswerDisposition,
    AnswerSchema,
    CanonicalAnswer,
    EvaluationMode,
    EventRole,
    GoldAction,
    MemoryEvent,
    MemoryObjectKey,
    MemoryQuery,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
)
from mub.vnext.validation import (
    ReplayResult,
    replay_actions,
    validate_distractors,
    validate_gold_replay,
    validate_task_semantics,
)


def _replace(model, **changes):
    data = {name: getattr(model, name) for name in type(model).model_fields}
    data.update(changes)
    return type(model).model_construct(**data)


def _codes(report):
    return [issue.code for issue in report.issues]


def _unresolved_task(make_task, status="ambiguous"):
    data = make_task().model_dump(mode="json")
    second_key = {
        **data["target_objects"][0],
        "entity": "colleague:alex",
    }
    data["target_objects"].append(second_key)
    linked_ids = {
        "unique": ["candidate_friend"],
        "ambiguous": ["candidate_friend", "candidate_colleague"],
        "no_match": [],
    }[status]
    data["queries"][0].update(
        query_type="unresolved_reference",
        target_object_keys=[],
        reference_candidates=[
            {
                "candidate_id": "candidate_friend",
                "object_key": data["target_objects"][0],
                "evidence": None,
                "source_anchors": [],
            },
            {
                "candidate_id": "candidate_colleague",
                "object_key": second_key,
                "evidence": None,
                "source_anchors": [],
            },
        ],
        surface_references=[
            {
                "reference_id": "reference_alex",
                "surface_text": "Alex",
                "normalized_text": "alex",
                "condition_kind": "same_surface_name",
                "evidence_kind": "query_span",
                "candidate_ids": linked_ids,
            }
        ],
    )
    data["gold"]["gold_answers"] = {}
    data["gold"]["acceptable_answers"] = {}
    if status == "unique":
        canonical = {
            "disposition": "answered",
            "resolution_status": "unique",
            "selected_candidate_ids": ["candidate_friend"],
            "abstention_reason": None,
            "value": "Qingdao",
        }
    else:
        canonical = {
            "disposition": "abstained",
            "resolution_status": status,
            "selected_candidate_ids": [],
            "abstention_reason": "not uniquely resolvable",
            "value": None,
        }
    data["gold"]["canonical_answers"] = {"query_0": canonical}
    return type(make_task()).model_validate(data)


def _action(action_id, event_id, operation, targets, value=None):
    return GoldAction.model_construct(
        action_id=action_id,
        event_id=event_id,
        operation=operation,
        scope=ActionScope.ATTRIBUTE,
        target_object_keys=targets,
        value=value,
        effective_at=None,
        expected_effect={},
    )


def _event(event_id, index, action_ids, role, text="event", metadata=None):
    return MemoryEvent(
        event_id=event_id,
        sequence_index=index,
        timestamp=None,
        raw_text=text,
        normalized_text=text,
        speaker=None,
        gold_action_ids=action_ids,
        role=role,
        metadata=metadata or {},
    )



def test_replay_result_is_recursively_immutable_and_json_round_trips(make_object_key):
    key = make_object_key(); value = {"nested": [1]}
    result = replay_actions([_action("a", "e", Operation.ADD, [key], value)])
    with pytest.raises((TypeError, AttributeError)):
        result.final_state[key.canonical_id]["nested"] += (2,)
    with pytest.raises(Exception):
        result.mutation_count = 3
    dumped = result.model_dump(mode="json")
    assert dumped["final_state"][key.canonical_id] == {"nested": [1]}
    assert dumped["version_history"][key.canonical_id] == [{"nested": [1]}]
    assert ReplayResult.model_validate(dumped) == result


def test_replay_add_update_and_object_type_insensitive_identity(make_object_key):
    key = make_object_key()
    alternate = _replace(key, object_type="record")
    result = replay_actions(
        [
            _action("a0", "e0", Operation.ADD, [key], "Dalian"),
            _action("a1", "e1", Operation.UPDATE, [alternate], "Qingdao"),
        ]
    )
    assert result == ReplayResult(
        final_state={key.canonical_id: "Qingdao"},
        version_history={key.canonical_id: ["Dalian", "Qingdao"]},
        mutation_count=2,
    )


def test_replay_noop_does_not_mutate():
    result = replay_actions([_action("noop", "e0", Operation.NOOP, [], None)])
    assert result == ReplayResult(final_state={}, version_history={}, mutation_count=0)


def test_duplicate_current_update_appends_history_and_counts_mutation(make_object_key):
    key = make_object_key()
    result = replay_actions(
        [
            _action("a0", "e0", Operation.ADD, [key], "same"),
            _action("a1", "e1", Operation.UPDATE, [key], "same"),
        ]
    )
    assert result.version_history[key.canonical_id] == ("same", "same")
    assert result.mutation_count == 2


def test_delete_removes_exact_targets_and_preserves_history_and_other_state(make_object_key):
    first = make_object_key()
    second = MemoryObjectKey(object_type="slot", namespace="default", entity="friend:sam", attribute="location")
    result = replay_actions(
        [
            _action("a0", "e0", Operation.ADD, [first, second], {"city": "Dalian"}),
            _action("a1", "e1", Operation.DELETE, [first]),
        ]
    )
    assert result.final_state == {second.canonical_id: {"city": "Dalian"}}
    assert result.model_dump(mode="json")["version_history"] == {
        first.canonical_id: [{"city": "Dalian"}],
        second.canonical_id: [{"city": "Dalian"}],
    }
    assert result.mutation_count == 3


@pytest.mark.parametrize(
    "actions, message",
    [
        (lambda key: [_action("a0", "e0", Operation.UPDATE, [key], "x")], "action[0] a0"),
        (lambda key: [_action("a0", "e0", Operation.ADD, [key], "x"), _action("a1", "e1", Operation.ADD, [key], "y")], "action[1] a1"),
        (lambda key: [_action("a0", "e0", Operation.DELETE, [key])], "action[0] a0"),
        (lambda key: [_action("a0", "e0", Operation.ADD, [key, _replace(key, object_type="other")], "x")], "duplicate target"),
        (lambda key: [_action("a0", "e0", Operation.NOOP, [key], None)], "NOOP"),
        (lambda key: [_action("a0", "e0", Operation.UPDATE, [key], None)], "non-null"),
        (lambda key: [_action("a0", "e0", Operation.DELETE, [key], "x")], "null"),
    ],
)
def test_replay_rejects_illegal_transitions_and_shapes(make_object_key, actions, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        replay_actions(actions(make_object_key()))


def test_replay_has_no_input_or_result_aliasing(make_object_key):
    key = make_object_key()
    value = {"nested": ["a"]}
    action = _action("a0", "e0", Operation.ADD, [key], value)
    result = replay_actions([action])
    action.value["nested"].append("input-change")
    assert result.model_dump(mode="json")["final_state"][key.canonical_id] == {"nested": ["a"]}
    with pytest.raises((AttributeError, TypeError)):
        result.final_state[key.canonical_id]["nested"] += ("result-change",)
    assert result.model_dump(mode="json")["version_history"][key.canonical_id] == [{"nested": ["a"]}]
    assert action.value == {"nested": ["a", "input-change"]}


def test_gold_replay_uses_action_sequence_not_storage_order(make_task):
    task = make_task()
    reordered = _replace(task, gold=_replace(task.gold, actions=list(reversed(task.gold.actions))))
    assert validate_gold_replay(reordered).valid is True


@pytest.mark.parametrize("status", ["unique", "ambiguous", "no_match"])
def test_gold_replay_accepts_explicit_unresolved_reference_outcomes(make_task, status):
    assert validate_gold_replay(_unresolved_task(make_task, status)).valid is True


def test_unresolved_reference_replay_is_not_inferred_from_final_state_or_delete(make_task):
    task = _unresolved_task(make_task, "unique")
    key = task.target_objects[0]
    delete = _action("action_1", "event_1", Operation.DELETE, [key])
    gold = _replace(
        task.gold,
        actions=[task.gold.actions[0], delete],
        final_state={},
        version_history={key.canonical_id: ["Dalian"]},
        expected_present_objects=[],
        expected_absent_objects=[key],
    )

    report = validate_gold_replay(_replace(task, gold=gold))

    assert report.valid is True
    assert "current_query_target_absent" not in _codes(report)
    assert "unresolved_query_semantics" not in _codes(report)


def test_semantic_validation_reports_unresolved_defects_exactly_once(make_task):
    task = _unresolved_task(make_task)
    guessed = CanonicalAnswer.model_construct(
        disposition=AnswerDisposition.ANSWERED,
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        selected_candidate_ids=["candidate_friend"],
        abstention_reason=None,
        value="Qingdao",
    )
    gold = _replace(
        task.gold,
        gold_answers={"query_0": None},
        canonical_answers={"query_0": guessed},
    )

    codes = _codes(validate_task_semantics(_replace(task, gold=gold)))

    assert codes.count("unresolved_raw_answer") == 1
    assert codes.count("canonical_answer_status_disposition_mismatch") == 1
    assert codes.count("guessed_ambiguous_candidate") == 1


def test_semantic_validation_rejects_ordinary_query_abstention_once(make_task):
    task = make_task()
    abstention = CanonicalAnswer(
        disposition=AnswerDisposition.ABSTAINED,
        resolution_status=ReferenceResolutionStatus.NO_MATCH,
        abstention_reason="not found",
    )
    gold = _replace(task.gold, canonical_answers={"query_0": abstention})

    assert _codes(
        validate_task_semantics(_replace(task, gold=gold))
    ).count("ordinary_query_canonical_answer") == 1


def test_gold_replay_reports_replay_errors_without_escaping(make_task):
    task = make_task()
    sequence = ["action_1", "action_0"]
    malformed = _replace(task, gold=_replace(task.gold, action_sequence=sequence))
    report = validate_gold_replay(malformed)
    assert report.valid is False
    assert "gold_replay_error" in _codes(report)
    assert report.issues[0].path == "gold.actions[1]"


def test_gold_replay_reports_final_history_and_expectation_mismatches(make_task):
    task = make_task()
    key = task.target_objects[0]
    gold = _replace(
        task.gold,
        final_state={key.canonical_id: "wrong"},
        version_history={key.canonical_id: ["wrong"]},
        expected_present_objects=[],
        expected_absent_objects=[key],
    )
    codes = _codes(validate_gold_replay(_replace(task, gold=gold)))
    assert "replay_final_state_mismatch" in codes
    assert "replay_version_history_mismatch" in codes
    assert "expected_absent_replay_present" in codes


def test_expected_present_missing_is_reported(make_task):
    task = make_task()
    key = task.target_objects[0]
    actions = [task.gold.actions[0], _action("action_1", "event_1", Operation.DELETE, [key])]
    gold = _replace(task.gold, actions=actions, final_state={}, expected_present_objects=[key], expected_absent_objects=[])
    assert "expected_present_replay_missing" in _codes(validate_gold_replay(_replace(task, gold=gold)))


def test_historical_query_resolves_zero_based_version(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_type=QueryType.HISTORICAL_STATE, metadata={"version_index": 0})
    gold = _replace(task.gold, gold_answers={"query_0": "Dalian"}, acceptable_answers={"query_0": ["Dalian"]})
    assert validate_gold_replay(_replace(task, queries=[query], gold=gold)).valid is True


def test_historical_query_reports_out_of_range_and_wrong_canonical_answer(make_task):
    task = make_task()
    out_of_range = _replace(task.queries[0], query_type=QueryType.HISTORICAL_STATE, metadata={"version_index": 9})
    assert "historical_version_out_of_range" in _codes(validate_gold_replay(_replace(task, queries=[out_of_range])))
    wrong = _replace(task.queries[0], query_type=QueryType.HISTORICAL_STATE, metadata={"version_index": 0})
    gold = _replace(task.gold, gold_answers={"query_0": "Qingdao"}, acceptable_answers={"query_0": ["Dalian", "Qingdao"]})
    assert "historical_gold_answer_mismatch" in _codes(validate_gold_replay(_replace(task, queries=[wrong], gold=gold)))


def test_current_single_target_answer_must_equal_replayed_state(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_answers={"query_0": "wrong"}, acceptable_answers={"query_0": ["wrong"]})
    assert "current_gold_answer_mismatch" in _codes(validate_gold_replay(_replace(task, gold=gold)))


def test_gold_replay_and_distractor_validation_do_not_mutate_task(make_task):
    task = make_task()
    before = deepcopy(task.model_dump(mode="python"))
    validate_gold_replay(task)
    validate_distractors(task)
    assert task.model_dump(mode="python") == before


def test_semantic_validators_contain_unhashable_constructed_ids(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_id=[])
    malformed = _replace(task, queries=[query])
    replay_report = validate_gold_replay(malformed)
    distractor_report = validate_distractors(malformed)
    assert "malformed_gold_replay_structure" in _codes(replay_report)
    assert "malformed_distractor_structure" in _codes(distractor_report)


def test_factory_stale_event_is_later_superseded(make_task):
    assert validate_distractors(make_task()).valid is True


def test_stale_role_that_remains_current_is_invalid(make_task):
    task = make_task()
    events = [task.events[0]]
    actions = [task.gold.actions[0]]
    key = task.target_objects[0]
    gold = _replace(
        task.gold,
        actions=actions,
        action_sequence=["action_0"],
        final_state={key.canonical_id: "Dalian"},
        version_history={key.canonical_id: ["Dalian"]},
        gold_source_event_ids=["event_0"],
        gold_answers={"query_0": "Dalian"},
        acceptable_answers={"query_0": ["Dalian"]},
    )
    assert "stale_not_superseded" in _codes(validate_distractors(_replace(task, events=events, gold=gold)))


def test_stale_value_must_differ_from_later_value(make_task):
    task = make_task()
    second = _replace(task.gold.actions[1], value="Dalian")
    gold = _replace(task.gold, actions=[task.gold.actions[0], second], final_state={task.target_objects[0].canonical_id: "Dalian"}, version_history={task.target_objects[0].canonical_id: ["Dalian", "Dalian"]}, gold_answers={"query_0": "Dalian"}, acceptable_answers={"query_0": ["Dalian"]})
    assert "stale_value_not_obsolete" in _codes(validate_distractors(_replace(task, gold=gold)))


def test_duplicate_current_is_judged_at_event_time_not_as_stale(make_task):
    task = make_task()
    key = task.target_objects[0]
    events = [
        _event("e0", 0, ["a0"], EventRole.NEUTRAL, "Dalian"),
        _event("e1", 1, ["a1"], EventRole.DUPLICATE_CURRENT, "Dalian again"),
        _event("e2", 2, ["a2"], EventRole.LATEST_GOLD, "Qingdao"),
    ]
    actions = [
        _action("a0", "e0", Operation.ADD, [key], "Dalian"),
        _action("a1", "e1", Operation.UPDATE, [key], "Dalian"),
        _action("a2", "e2", Operation.UPDATE, [key], "Qingdao"),
    ]
    gold = _replace(task.gold, actions=actions, action_sequence=["a0", "a1", "a2"], final_state={key.canonical_id: "Qingdao"}, version_history={key.canonical_id: ["Dalian", "Dalian", "Qingdao"]}, gold_source_event_ids=["e2"])
    codes = _codes(validate_distractors(_replace(task, events=events, gold=gold)))
    assert "duplicate_current_value_mismatch" not in codes
    assert "stale_value_not_obsolete" not in codes


def test_duplicate_current_value_must_match_immediately_preceding_current(make_task):
    task = make_task()
    events = [task.events[0], _replace(task.events[1], role=EventRole.DUPLICATE_CURRENT)]
    assert "duplicate_current_value_mismatch" in _codes(validate_distractors(_replace(task, events=events)))


@pytest.mark.parametrize("role", [EventRole.SAME_ENTITY_OTHER_ATTRIBUTE, EventRole.SAME_NAME_OTHER_ENTITY, EventRole.NOOP_NEAR_MISS, EventRole.NEUTRAL])
def test_distractor_action_must_not_establish_accepted_current_answer(make_task, role):
    task = make_task()
    events = [task.events[0], _replace(task.events[1], role=role)]
    assert "distractor_establishes_accepted_answer" in _codes(validate_distractors(_replace(task, events=events)))


def test_simple_distractor_text_leak_rejects_forged_metadata_waiver(make_task):
    task = make_task()
    event = _replace(task.events[0], raw_text="Maybe Qingdao", normalized_text="maybe qingdao", role=EventRole.NEUTRAL)
    report = validate_distractors(_replace(task, events=[event, task.events[1]]))
    assert "distractor_text_contains_accepted_answer" in _codes(report)
    forged = _replace(
        event,
        metadata={
            "allow_accepted_answer_ambiguity": True,
            "compatibility_rule": "non_target_accepted_answer_text_overlap_v1",
            "legacy_role": EventRole.NEUTRAL.value,
        },
    )
    forged_codes = _codes(validate_distractors(_replace(task, events=[forged, task.events[1]])))
    assert "distractor_text_contains_accepted_answer" in forged_codes


def test_canonical_answer_must_be_present_in_acceptable_support(make_task):
    task = make_task()
    gold = _replace(task.gold, acceptable_answers={"query_0": ["alias only"]})
    assert "canonical_answer_not_acceptable" in _codes(validate_distractors(_replace(task, gold=gold)))


def test_current_answer_requires_one_replayed_target_support(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_answers={"query_0": "wrong"}, acceptable_answers={"query_0": ["wrong"]})
    assert "current_answer_not_supported" in _codes(
        validate_distractors(_replace(task, gold=gold))
    )


def test_harmless_string_aliases_do_not_create_unique_support_error(make_task):
    task = make_task()
    gold = _replace(task.gold, acceptable_answers={"query_0": ["Qingdao", " qingdao "]})
    assert "ambiguous_current_answer_support" not in _codes(validate_distractors(_replace(task, gold=gold)))


def test_multiple_queried_current_values_with_accepted_support_are_ambiguous(make_task):
    task = make_task()
    first = task.target_objects[0]
    second = MemoryObjectKey(object_type="slot", namespace="default", entity="friend:sam", attribute="location")
    action = _action("action_2", "event_2", Operation.ADD, [second], "Dalian")
    event = _event("event_2", 2, ["action_2"], EventRole.LATEST_GOLD, "Sam is in Dalian")
    query = _replace(task.queries[0], target_object_keys=[first, second])
    gold = _replace(
        task.gold,
        actions=[*task.gold.actions, action],
        action_sequence=[*task.gold.action_sequence, "action_2"],
        final_state={first.canonical_id: "Qingdao", second.canonical_id: "Dalian"},
        version_history={**task.gold.version_history, second.canonical_id: ["Dalian"]},
        expected_present_objects=[first, second],
        gold_answers={"query_0": "Qingdao"},
        acceptable_answers={"query_0": ["Qingdao", "Dalian"]},
    )
    malformed = _replace(task, events=[*task.events, event], target_objects=[first, second], queries=[query], gold=gold)
    assert "ambiguous_current_answer_support" in _codes(validate_distractors(malformed))


def test_distractor_validator_reports_malformed_answer_maps(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_answers={"unknown": "x"}, acceptable_answers={"unknown": ["x"]})
    codes = _codes(validate_distractors(_replace(task, gold=gold)))
    assert "missing_query_gold_answer" in codes
    assert "missing_query_acceptable_answers" in codes
    assert "unknown_gold_answer_query" in codes
    assert "unknown_acceptable_answer_query" in codes


def test_distractor_validator_contains_malformed_targets_in_report(make_task):
    task = make_task()
    malformed_key = MemoryObjectKey.model_construct(
        object_type="slot", namespace="default", entity=None, attribute="location", subkey=None
    )
    later = _replace(task.gold.actions[1], target_object_keys=[malformed_key])
    malformed = _replace(task, gold=_replace(task.gold, actions=[task.gold.actions[0], later]))
    report = validate_distractors(malformed)
    assert report.valid is False
    assert "malformed_distractor_target" in _codes(report)


@pytest.mark.parametrize("role", [EventRole.STALE_SAME_SLOT, EventRole.DUPLICATE_CURRENT])
def test_stale_and_duplicate_roles_require_a_write_action(make_task, role):
    task = make_task()
    event = _replace(task.events[0], role=role, gold_action_ids=[])
    malformed = _replace(task, events=[event, task.events[1]])
    expected = "stale_role_without_write" if role == EventRole.STALE_SAME_SLOT else "duplicate_current_role_without_write"
    assert expected in _codes(validate_distractors(malformed))


def test_same_value_on_two_queried_targets_is_still_ambiguous_support(make_task):
    task = make_task()
    first = task.target_objects[0]
    second = MemoryObjectKey(object_type="slot", namespace="default", entity="friend:sam", attribute="location")
    action = _action("action_2", "event_2", Operation.ADD, [second], "Qingdao")
    event = _event("event_2", 2, ["action_2"], EventRole.LATEST_GOLD, "Sam is in Qingdao")
    query = _replace(task.queries[0], target_object_keys=[first, second])
    gold = _replace(
        task.gold,
        actions=[*task.gold.actions, action],
        action_sequence=[*task.gold.action_sequence, "action_2"],
        final_state={first.canonical_id: "Qingdao", second.canonical_id: "Qingdao"},
        version_history={**task.gold.version_history, second.canonical_id: ["Qingdao"]},
        expected_present_objects=[first, second],
    )
    malformed = _replace(task, events=[*task.events, event], target_objects=[first, second], queries=[query], gold=gold)
    assert "ambiguous_current_answer_support" in _codes(validate_distractors(malformed))


def test_repeated_reference_to_same_query_target_is_not_two_supports(make_task):
    task = make_task()
    key = task.target_objects[0]
    alternate = _replace(key, object_type="record")
    query = _replace(task.queries[0], target_object_keys=[key, alternate])
    assert "ambiguous_current_answer_support" not in _codes(
        validate_distractors(_replace(task, queries=[query]))
    )


def test_distractor_string_leak_matching_is_case_insensitive(make_task):
    task = make_task()
    event = _replace(task.events[0], raw_text="maybe qingdao", normalized_text="maybe qingdao", role=EventRole.NEUTRAL)
    assert "distractor_text_contains_accepted_answer" in _codes(
        validate_distractors(_replace(task, events=[event, task.events[1]]))
    )




@pytest.mark.parametrize(
    ("schema", "old_value", "current_value"),
    [
        (AnswerSchema.LIST, ["old"], ["current"]),
        (AnswerSchema.OBJECT, {"value": "old"}, {"value": "current"}),
    ],
)
def test_single_target_current_structured_value_is_not_aggregated(make_task, schema, old_value, current_value):
    task = make_task(); key = task.target_objects[0]
    actions = [_replace(task.gold.actions[0], value=old_value), _replace(task.gold.actions[1], value=current_value)]
    query = _replace(task.queries[0], answer_schema=schema)
    gold = _replace(task.gold, actions=actions, final_state={key.canonical_id: current_value}, version_history={key.canonical_id: [old_value, current_value]}, gold_answers={"query_0": current_value}, acceptable_answers={"query_0": current_value})
    structured = _replace(task, queries=[query], gold=gold)
    assert validate_gold_replay(structured).valid is True
    assert validate_distractors(structured).valid is True


@pytest.mark.parametrize("schema", [AnswerSchema.LIST, AnswerSchema.OBJECT])
def test_multi_target_current_values_still_use_schema_aggregation(make_task, schema):
    task = make_task(); first = task.target_objects[0]
    second = MemoryObjectKey(object_type="slot", namespace="default", entity="friend:sam", attribute="location")
    add_second = _action("action_2", "event_2", Operation.ADD, [second], "Dalian")
    event_second = _event("event_2", 2, ["action_2"], EventRole.LATEST_GOLD, "Sam is in Dalian")
    query = _replace(task.queries[0], target_object_keys=[first, second], answer_schema=schema)
    answer = ["Qingdao", "Dalian"] if schema == AnswerSchema.LIST else {first.canonical_id: "Qingdao", second.canonical_id: "Dalian"}
    gold = _replace(task.gold, actions=[*task.gold.actions, add_second], action_sequence=[*task.gold.action_sequence, "action_2"], final_state={first.canonical_id: "Qingdao", second.canonical_id: "Dalian"}, version_history={**task.gold.version_history, second.canonical_id: ["Dalian"]}, expected_present_objects=[first, second], gold_answers={"query_0": answer}, acceptable_answers={"query_0": answer})
    multi = _replace(task, events=[*task.events, event_second], target_objects=[first, second], queries=[query], gold=gold)
    assert validate_gold_replay(multi).valid is True
    assert validate_distractors(multi).valid is True



def test_malformed_current_query_target_keeps_specific_issue_and_continues(make_task):
    task = make_task()
    malformed_key = MemoryObjectKey.model_construct(
        object_type="slot", namespace="default", entity=None, attribute="location", subkey=None
    )
    malformed_query = _replace(task.queries[0], target_object_keys=[malformed_key])
    later_query = _replace(task.queries[0], query_id="query_1")
    gold = _replace(
        task.gold,
        gold_answers={"query_0": "Qingdao", "query_1": "wrong"},
        acceptable_answers={"query_0": ["Qingdao"], "query_1": ["wrong"]},
    )
    malformed = _replace(task, queries=[malformed_query, later_query], gold=gold)
    before = deepcopy(malformed.model_dump(mode="python"))
    report = validate_distractors(malformed)
    assert "queries[0].target_object_keys[0]" in [
        issue.path for issue in report.issues if issue.code == "malformed_distractor_target"
    ]
    assert "current_answer_not_supported" in _codes(report)
    assert "malformed_distractor_structure" not in _codes(report)
    assert malformed.model_dump(mode="python") == before


def test_replay_rejects_whitespace_and_nonstring_identity_parts_with_context():
    bad_keys = [
        MemoryObjectKey.model_construct(object_type=" ", namespace="default", entity="x", attribute="a", subkey=None),
        MemoryObjectKey.model_construct(object_type="slot", namespace=" ", entity="x", attribute="a", subkey=None),
        MemoryObjectKey.model_construct(object_type="slot", namespace="default", entity=1, attribute="a", subkey=None),
    ]
    for key in bad_keys:
        with pytest.raises(ValueError, match=r"action\[0\] bad"):
            replay_actions([_action("bad", "e0", Operation.ADD, [key], "x")])


def test_duplicate_current_uses_action_sequence_not_event_storage_order(make_task):
    task = make_task()
    key = task.target_objects[0]
    events = [
        _event("e_dup", 0, ["a_dup"], EventRole.DUPLICATE_CURRENT, "same"),
        _event("e_add", 1, ["a_add"], EventRole.NEUTRAL, "same"),
    ]
    actions = [
        _action("a_dup", "e_dup", Operation.UPDATE, [key], "same"),
        _action("a_add", "e_add", Operation.ADD, [key], "same"),
    ]
    gold = _replace(task.gold, actions=actions, action_sequence=["a_add", "a_dup"], final_state={key.canonical_id: "same"}, version_history={key.canonical_id: ["same", "same"]}, gold_answers={"query_0": "same"}, acceptable_answers={"query_0": ["same"]})
    assert "duplicate_current_value_mismatch" not in _codes(validate_distractors(_replace(task, events=events, gold=gold)))


def test_distractor_action_path_uses_storage_index_when_order_differs(make_task):
    task = make_task()
    bad = MemoryObjectKey.model_construct(object_type="slot", namespace="default", entity=None, attribute="x", subkey=None)
    stored = [_replace(task.gold.actions[1], target_object_keys=[bad]), task.gold.actions[0]]
    report = validate_distractors(_replace(task, gold=_replace(task.gold, actions=stored)))
    assert "gold.actions[0].target_object_keys[0]" in [i.path for i in report.issues if i.code == "malformed_distractor_target"]


@pytest.mark.parametrize(
    ("query_type", "schema", "metadata", "answer"),
    [
        (QueryType.CURRENT_STATE, AnswerSchema.STRING, {}, "Qingdao"),
        (QueryType.HISTORICAL_STATE, AnswerSchema.STRING, {"version_index": 0}, "Dalian"),
        (QueryType.MULTI_OBJECT, AnswerSchema.LIST, {}, ["Qingdao"]),
        (QueryType.TRANSITION, AnswerSchema.OBJECT, {"from_version_index": 0, "to_version_index": 1}, {"from": "Dalian", "to": "Qingdao"}),
    ],
)
def test_query_types_have_canonical_resolution(make_task, query_type, schema, metadata, answer):
    task = make_task()
    query = _replace(task.queries[0], query_type=query_type, answer_schema=schema, metadata=metadata)
    gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": answer})
    assert validate_distractors(_replace(task, queries=[query], gold=gold)).valid is True


def test_deletion_compliance_has_canonical_absence_resolution(make_task):
    task = make_task(); key = task.target_objects[0]
    delete = _action("action_1", "event_1", Operation.DELETE, [key])
    events = [_replace(task.events[0], role=EventRole.HISTORICAL_SUPPORT), task.events[1]]
    query = _replace(task.queries[0], query_type=QueryType.DELETION_COMPLIANCE, answer_schema=AnswerSchema.BOOLEAN)
    gold = _replace(task.gold, actions=[task.gold.actions[0], delete], final_state={}, version_history={key.canonical_id: ["Dalian"]}, expected_present_objects=[], expected_absent_objects=[key], gold_answers={"query_0": True}, acceptable_answers={"query_0": [True]})
    assert validate_distractors(_replace(task, events=events, queries=[query], gold=gold)).valid is True


def test_historical_distractor_support_is_ambiguous(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_type=QueryType.HISTORICAL_STATE, metadata={"version_index": 0})
    event = _replace(task.events[0], role=EventRole.NEUTRAL)
    gold = _replace(task.gold, gold_answers={"query_0": "Dalian"}, acceptable_answers={"query_0": ["Dalian"]})
    assert "distractor_establishes_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))



def test_multi_object_distractor_alias_is_ambiguous(make_task):
    task = make_task(); key = task.target_objects[0]
    alias = ["Dalian"]
    actions = [_replace(task.gold.actions[0], value=alias), task.gold.actions[1]]
    event = _replace(task.events[0], role=EventRole.NEUTRAL)
    query = _replace(task.queries[0], query_type=QueryType.MULTI_OBJECT, answer_schema=AnswerSchema.LIST)
    canonical = ["Qingdao"]
    gold = _replace(task.gold, actions=actions, version_history={key.canonical_id: [alias, "Qingdao"]}, gold_answers={"query_0": canonical}, acceptable_answers={"query_0": [canonical, alias]})
    assert "distractor_establishes_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))


def test_deletion_distractor_value_is_ambiguous(make_task):
    task = make_task(); key = task.target_objects[0]
    add = _replace(task.gold.actions[0], value=True)
    delete = _action("action_1", "event_1", Operation.DELETE, [key])
    event = _replace(task.events[0], role=EventRole.NEUTRAL)
    query = _replace(task.queries[0], query_type=QueryType.DELETION_COMPLIANCE, answer_schema=AnswerSchema.BOOLEAN)
    gold = _replace(task.gold, actions=[add, delete], final_state={}, version_history={key.canonical_id: [True]}, expected_present_objects=[], expected_absent_objects=[key], gold_answers={"query_0": True}, acceptable_answers={"query_0": [True]})
    assert "distractor_establishes_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))


def test_transition_distractor_alias_is_ambiguous(make_task):
    task = make_task(); key = task.target_objects[0]
    alias = {"from": "old", "to": "wrong"}
    actions = [_replace(task.gold.actions[0], value=alias), task.gold.actions[1]]
    event = _replace(task.events[0], role=EventRole.NEUTRAL)
    query = _replace(task.queries[0], query_type=QueryType.TRANSITION, answer_schema=AnswerSchema.OBJECT, metadata={"from_version_index": 0, "to_version_index": 1})
    canonical = {"from": alias, "to": "Qingdao"}
    gold = _replace(task.gold, actions=actions, version_history={key.canonical_id: [alias, "Qingdao"]}, gold_answers={"query_0": canonical}, acceptable_answers={"query_0": [canonical, alias]})
    assert "distractor_establishes_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))


def test_unresolvable_transition_is_not_silently_accepted(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_type=QueryType.TRANSITION, answer_schema=AnswerSchema.OBJECT, metadata={})
    answer = {"from": "Dalian", "to": "Qingdao"}
    gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": answer})
    assert "unresolved_query_semantics" in _codes(validate_distractors(_replace(task, queries=[query], gold=gold)))


def test_transition_accepts_explicit_destination_version_index(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_type=QueryType.TRANSITION, answer_schema=AnswerSchema.OBJECT, metadata={"version_index": 1})
    answer = {"from": "Dalian", "to": "Qingdao"}
    gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": answer})
    assert validate_distractors(_replace(task, queries=[query], gold=gold)).valid is True




def test_standalone_semantic_validators_reject_empty_duplicate_targets_and_malformed_maps(make_task):
    task = make_task(); key = task.target_objects[0]
    duplicate = _replace(key, object_type="other")
    empty = _replace(task.queries[0], query_type=QueryType.TRANSITION, target_object_keys=[], metadata={"version_index": 1})
    duplicated = _replace(task.queries[0], target_object_keys=[key, duplicate])
    for query, expected_path in ((empty, "queries[0].target_object_keys"), (duplicated, "queries[0].target_object_keys[1]")):
        malformed = _replace(task, queries=[query])
        for validator in (validate_gold_replay, validate_distractors):
            report = validator(malformed)
            assert expected_path in [issue.path for issue in report.issues]
    for field in ("final_state", "version_history"):
        gold = _replace(task.gold, **{field: []})
        malformed = _replace(task, gold=gold)
        for validator in (validate_gold_replay, validate_distractors):
            report = validator(malformed)
            assert f"gold.{field}" in [issue.path for issue in report.issues]
            assert report.valid is False


def test_numeric_leakage_compares_complete_numeric_literals(make_task):
    task = make_task(); query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    gold = _replace(task.gold, gold_answers={"query_0": 42}, acceptable_answers={"query_0": [42]})
    for text in ("142", "42.5", "-42"):
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        assert "distractor_text_contains_accepted_answer" not in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))
    for text in ("42", "42.0", "4.2e1", "420e-1"):
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        assert "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))



def test_standalone_answer_maps_are_guarded_without_queries_or_mutation(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_answers=[], acceptable_answers=None)
    malformed = _replace(task, queries=[], gold=gold)
    before_answers = deepcopy(malformed.gold.gold_answers)
    before_acceptable = deepcopy(malformed.gold.acceptable_answers)
    for validator in (validate_gold_replay, validate_distractors):
        report = validator(malformed)
        assert [issue.path for issue in report.issues if issue.code in {"malformed_gold_answers", "malformed_acceptable_answers"}] == [
            "gold.gold_answers",
            "gold.acceptable_answers",
        ]
        assert report.valid is False
    composed = validate_task_semantics(malformed)
    assert [issue.path for issue in composed.issues if issue.code in {"malformed_gold_answers", "malformed_acceptable_answers"}].count("gold.gold_answers") >= 1
    assert [issue.path for issue in composed.issues if issue.code in {"malformed_gold_answers", "malformed_acceptable_answers"}].count("gold.acceptable_answers") >= 1
    assert malformed.gold.gold_answers == before_answers
    assert malformed.gold.acceptable_answers == before_acceptable


def test_grouped_numeric_leakage_uses_complete_tokens(make_task):
    task = make_task(); query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    def codes_for(answer, text):
        gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": [answer]})
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        return _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))
    assert "distractor_text_contains_accepted_answer" not in codes_for(42, "42,000")
    assert "distractor_text_contains_accepted_answer" in codes_for(42000, "value: 42,000.")
    assert "distractor_text_contains_accepted_answer" in codes_for(42000.5, "- ignore; +42,000.5!")
    assert "distractor_text_contains_accepted_answer" in codes_for(-42000, "-42,000")
    for malformed in ("4,20", "42,00"):
        assert "distractor_text_contains_accepted_answer" not in codes_for(20, malformed)



def test_numeric_like_spans_are_consumed_once_before_strict_parsing(make_task):
    task = make_task(); query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    def leaked(answer, text):
        gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": [answer]})
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        return "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))
    for text in ("42,,000", "x42,000", "42,000x", "id42", "42kg", "42,00", "4,20", "42e"):
        assert leaked(42, text) is False
    assert leaked(0, "x42,000") is False
    for text in ("42,000", "value: 42,000.", "+42,000", "42,000.5", "4.2e4"):
        expected = 42000.5 if ".5" in text else 42000
        assert leaked(expected, text) is True
    assert leaked(42, "bad 42,,000 then valid 42") is True



def test_numeric_scanner_consumes_connected_malformed_syntax_once(make_task):
    task = make_task(); query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    def leaked(answer, text):
        gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": [answer]})
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        return "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))
    for text in ("42..000", "42...", "42,,000", "42e", "42e+", "x42", "42x"):
        assert leaked(42, text) is False
    for text in ("+-42", "--42", "-+42", "x-42", "-42x"):
        assert leaked(-42, text) is False
    for text in ("42.", "(42)", "value=42", "42.0", "4.2e1"):
        assert leaked(42, text) is True
    assert leaked(42.5, "42.5.") is True
    assert leaked(42, "42.5.") is False
    assert leaked(42, "bad 42..000 then 42.") is True



def test_single_trailing_sentence_comma_is_normalized_after_maximal_scan(make_task):
    task = make_task(); query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    def leaked(answer, text):
        gold = _replace(task.gold, gold_answers={"query_0": answer}, acceptable_answers={"query_0": [answer]})
        event = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text=text, normalized_text=text)
        return "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[event, task.events[1]], queries=[query], gold=gold)))
    for text in ("42, then", "42,", "(42,)"):
        assert leaked(42, text) is True
    for text in ("42,000, people", "42,000,"):
        assert leaked(42000, text) is True
    assert leaked(-42, "-42,") is True
    assert leaked(0.5, "0.5,") is True
    assert leaked(0.5, "5e-1,") is True
    for text in ("42,,", "42,,000,", "42,00,", "42,,,"):
        assert leaked(42, text) is False
    assert leaked(42, "x42,") is False
    assert leaked(42, "bad 42,, then valid 42,") is True


def test_stale_terminal_delete_supersedes_and_duplicate_delete_is_invalid(make_task):
    task = make_task(); key = task.target_objects[0]
    delete = _action("action_1", "event_1", Operation.DELETE, [key])
    gold = _replace(task.gold, actions=[task.gold.actions[0], delete], final_state={}, version_history={key.canonical_id: ["Dalian"]}, expected_present_objects=[], expected_absent_objects=[key])
    deleted = _replace(task, gold=gold)
    codes = _codes(validate_distractors(deleted))
    assert "stale_not_superseded" not in codes
    assert "stale_value_not_obsolete" not in codes
    duplicate_event = _replace(task.events[1], role=EventRole.DUPLICATE_CURRENT)
    assert "invalid_duplicate_current_action" in _codes(validate_distractors(_replace(deleted, events=[task.events[0], duplicate_event])))


def test_typed_text_leakage_and_numeric_boundaries(make_task):
    task = make_task()
    query = _replace(task.queries[0], answer_schema=AnswerSchema.NUMBER)
    gold = _replace(task.gold, gold_answers={"query_0": 42}, acceptable_answers={"query_0": [42]})
    safe = _replace(task.events[0], role=EventRole.NEUTRAL, raw_text="value 142", normalized_text="value 142")
    assert "distractor_text_contains_accepted_answer" not in _codes(validate_distractors(_replace(task, events=[safe, task.events[1]], queries=[query], gold=gold)))
    leaking = _replace(safe, raw_text="value 42", normalized_text="value 42")
    assert "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[leaking, task.events[1]], queries=[query], gold=gold)))
    object_query = _replace(query, answer_schema=AnswerSchema.OBJECT)
    object_gold = _replace(task.gold, gold_answers={"query_0": {"nested": "needle"}}, acceptable_answers={"query_0": {"nested": "needle"}})
    nested = _replace(safe, raw_text="a needle here", normalized_text="a needle here")
    assert "distractor_text_contains_accepted_answer" in _codes(validate_distractors(_replace(task, events=[nested, task.events[1]], queries=[object_query], gold=object_gold)))


def test_composed_validation_order_purity_and_duplicate_behavior(make_task):
    task = make_task()
    before = deepcopy(task.model_dump(mode="python"))
    malformed = _replace(task, schema_version="bad", gold=_replace(task.gold, final_state={}))
    report = validate_task_semantics(malformed)
    assert report.issues[0].code == "unsupported_schema_version"
    assert [i.code for i in report.issues].count("replay_final_state_mismatch") == 1
    assert malformed.model_dump(mode="python") != before
    snapshot = deepcopy(malformed.model_dump(mode="python"))
    assert validate_task_semantics(malformed) == report
    assert malformed.model_dump(mode="python") == snapshot
