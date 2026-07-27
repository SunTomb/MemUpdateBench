from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import (
    AnswerDisposition,
    AnswerSchema,
    CanonicalAnswer,
    Difficulty,
    MemoryObjectKey,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
    TaskFamily,
)
from mub.vnext.validation import ValidationIssue, ValidationReport, validate_task
from mub.vnext.version import SCHEMA_VERSION


def _replace(model, **changes):
    data = {name: getattr(model, name) for name in type(model).model_fields}
    data.update(changes)
    return type(model).model_construct(**data)


def _codes(report):
    return [issue.code for issue in report.issues]


def _paths(report, code):
    return [issue.path for issue in report.issues if issue.code == code]


def _unresolved_task(make_task, status="ambiguous"):
    data = make_task().model_dump(mode="json")
    second_key = {
        **data["target_objects"][0],
        "object_type": "profile",
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
                "evidence": "friend-qualified Alex",
                "source_anchors": [],
            },
            {
                "candidate_id": "candidate_colleague",
                "object_key": second_key,
                "evidence": "colleague-qualified Alex",
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
            "abstention_reason": "reference is not uniquely resolvable",
            "value": None,
        }
    data["gold"]["canonical_answers"] = {"query_0": canonical}
    return type(make_task()).model_validate(data)


def test_validation_report_enforces_error_consistency_and_strict_bool():
    warning = ValidationIssue(code="warning", message="warning", path="task", severity="warning")
    assert ValidationReport(valid=True, issues=[warning]).valid is True
    error = ValidationIssue(code="error", message="error", path="task", severity="error")
    with pytest.raises(ValidationError):
        ValidationReport(valid=True, issues=[error])
    with pytest.raises(ValidationError):
        ValidationReport(valid=1, issues=[])



def test_validation_artifacts_are_immutable_and_round_trip():
    issue = ValidationIssue(code="w", message="warning", path="x", severity="warning")
    report = ValidationReport(valid=True, issues=[issue])
    with pytest.raises(ValidationError):
        issue.code = "changed"
    with pytest.raises(ValidationError):
        report.valid = False
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        report.issues += (issue,)
    with pytest.raises(ValidationError):
        report.model_copy(update={"valid": False})
    dumped = report.model_dump(mode="json")
    assert dumped["issues"] == [{"code": "w", "message": "warning", "path": "x", "severity": "warning"}]
    assert ValidationReport.model_validate(dumped) == report


def test_valid_task_has_exactly_empty_valid_report(make_task):
    assert validate_task(make_task()) == ValidationReport(valid=True, issues=[])


@pytest.mark.parametrize("status", ["unique", "ambiguous", "no_match"])
def test_unresolved_reference_structural_validation_accepts_explicit_gold(
    make_task, status
):
    assert validate_task(_unresolved_task(make_task, status)) == ValidationReport(
        valid=True, issues=[]
    )


def test_unresolved_reference_structural_validation_reports_linkage_defects(make_task):
    task = _unresolved_task(make_task)
    candidates = task.queries[0].reference_candidates
    duplicate_candidate = _replace(
        candidates[1], candidate_id=candidates[0].candidate_id
    )
    surface = _replace(
        task.queries[0].surface_references[0],
        candidate_ids=[candidates[0].candidate_id, candidates[0].candidate_id, "missing"],
    )
    query = _replace(
        task.queries[0],
        reference_candidates=[candidates[0], duplicate_candidate],
        surface_references=[surface],
    )

    report = validate_task(_replace(task, queries=[query]))

    assert "duplicate_reference_candidate_id" in _codes(report)
    assert "duplicate_surface_candidate_id" in _codes(report)
    assert "unknown_surface_candidate_id" in _codes(report)


def test_unresolved_reference_structural_validation_rejects_typed_gold_mismatches(
    make_task,
):
    task = _unresolved_task(make_task)
    canonical = CanonicalAnswer.model_construct(
        disposition=AnswerDisposition.ANSWERED,
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        selected_candidate_ids=["candidate_friend"],
        abstention_reason=None,
        value="Qingdao",
    )
    gold = _replace(
        task.gold,
        gold_answers={"query_0": None},
        canonical_answers={"query_0": canonical},
    )

    codes = _codes(validate_task(_replace(task, gold=gold)))

    assert "unresolved_raw_answer" in codes
    assert "canonical_answer_status_disposition_mismatch" in codes
    assert "guessed_ambiguous_candidate" in codes


def test_abstained_ambiguous_guess_is_reported_exactly_once(make_task):
    task = _unresolved_task(make_task)
    canonical = CanonicalAnswer.model_construct(
        disposition=AnswerDisposition.ABSTAINED,
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        selected_candidate_ids=["candidate_friend"],
        abstention_reason="reference is not uniquely resolvable",
        value=None,
    )
    malformed = _replace(
        task,
        gold=_replace(
            task.gold,
            canonical_answers={"query_0": canonical},
        ),
    )

    assert _codes(validate_task(malformed)).count("guessed_ambiguous_candidate") == 1


def test_unresolved_reference_structural_validation_rejects_missing_and_ordinary_canonical_gold(
    make_task,
):
    unresolved = _unresolved_task(make_task)
    assert "missing_canonical_answer" in _codes(
        validate_task(_replace(unresolved, gold=_replace(unresolved.gold, canonical_answers={})))
    )

    ordinary = make_task()
    abstention = CanonicalAnswer(
        disposition=AnswerDisposition.ABSTAINED,
        resolution_status=ReferenceResolutionStatus.NO_MATCH,
        abstention_reason="not found",
    )
    report = validate_task(
        _replace(
            ordinary,
            gold=_replace(ordinary.gold, canonical_answers={"query_0": abstention}),
        )
    )
    assert "ordinary_query_canonical_answer" in _codes(report)


def test_validation_is_deterministic_and_does_not_mutate(make_task):
    task = make_task()
    malformed = _replace(task, schema_version="unsupported", task_id=" ")
    before = deepcopy(malformed.model_dump(mode="python"))
    first = validate_task(malformed)
    assert first == validate_task(malformed)
    assert malformed.model_dump(mode="python") == before
    assert _codes(first)[:2] == ["unsupported_schema_version", "blank_task_id"]


def test_unsupported_schema_version_is_reported(make_task):
    report = validate_task(_replace(make_task(), schema_version=SCHEMA_VERSION + "-future"))
    assert _paths(report, "unsupported_schema_version") == ["schema_version"]


@pytest.mark.parametrize(
    ("task_builder", "code", "path"),
    [
        (lambda task: _replace(task, events=[task.events[0], _replace(task.events[1], event_id=task.events[0].event_id)]), "duplicate_event_id", "events[1].event_id"),
        (lambda task: _replace(task, gold=_replace(task.gold, actions=[task.gold.actions[0], _replace(task.gold.actions[1], action_id=task.gold.actions[0].action_id)])), "duplicate_action_id", "gold.actions[1].action_id"),
        (lambda task: _replace(task, queries=[task.queries[0], _replace(task.queries[0])]), "duplicate_query_id", "queries[1].query_id"),
    ],
)
def test_duplicate_ids_are_reported(make_task, task_builder, code, path):
    assert path in _paths(validate_task(task_builder(make_task())), code)


def test_blank_ids_in_all_domains_are_reported(make_task):
    task = make_task()
    malformed = _replace(
        task,
        task_id=" ",
        source=_replace(task.source, source_id=" "),
        events=[_replace(task.events[0], event_id=""), task.events[1]],
        queries=[_replace(task.queries[0], query_id=" ")],
        gold=_replace(task.gold, actions=[_replace(task.gold.actions[0], action_id=" ", event_id=""), task.gold.actions[1]]),
    )
    codes = _codes(validate_task(malformed))
    for expected in ("blank_task_id", "blank_source_id", "blank_event_id", "blank_action_id", "blank_action_event_id", "blank_query_id"):
        assert expected in codes


def test_event_indices_must_be_strict_ordered_and_contiguous(make_task):
    task = make_task()
    reordered = _replace(task, events=[_replace(task.events[0], sequence_index=1), _replace(task.events[1], sequence_index=0)])
    assert _paths(validate_task(reordered), "invalid_event_sequence") == ["events"]
    boolean_index = _replace(task, events=[task.events[0], _replace(task.events[1], sequence_index=True)])
    assert _paths(validate_task(boolean_index), "invalid_event_sequence") == ["events"]


def test_action_event_references_and_ownership_are_bidirectional(make_task):
    task = make_task()
    malformed = _replace(
        task,
        events=[task.events[0], _replace(task.events[1], gold_action_ids=["action_0", "missing_action"])],
        gold=_replace(task.gold, actions=[_replace(task.gold.actions[0], event_id="missing"), task.gold.actions[1]]),
    )
    codes = _codes(validate_task(malformed))
    assert "missing_action_event" in codes
    assert "missing_event_action" in codes
    assert "action_event_ownership_mismatch" in codes


def test_action_sequence_must_cover_every_action_exactly_once(make_task):
    task = make_task()
    malformed = _replace(task, gold=_replace(task.gold, action_sequence=["action_0", "action_0", "missing"]))
    codes = _codes(validate_task(malformed))
    assert "duplicate_action_sequence_id" in codes
    assert "unknown_action_sequence_id" in codes
    assert "missing_action_sequence_id" in codes


def test_gold_source_and_source_anchor_references_are_checked(make_task):
    task = make_task()
    event = _replace(task.events[0], source_anchor={"source_id": "other_source", "event_id": "missing_event", "event_ids": ["event_1", "missing_2"]})
    malformed = _replace(task, events=[event, task.events[1]], gold=_replace(task.gold, gold_source_event_ids=["missing_gold_event"]))
    codes = _codes(validate_task(malformed))
    assert "source_anchor_source_mismatch" in codes
    assert codes.count("source_anchor_missing_event") == 2
    assert "missing_gold_source_event" in codes


def test_duplicate_gold_source_event_references_are_reported(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_source_event_ids=["event_1", "event_1"])
    assert "duplicate_gold_source_event_id" in _codes(validate_task(_replace(task, gold=gold)))


@pytest.mark.parametrize(
    ("family", "profile", "missing"),
    [
        (TaskFamily.REPEATED_SAME_SLOT.value, {}, "update_depth"),
        (TaskFamily.INTERLEAVED_MULTI_SLOT.value, {"update_depth": 2}, "active_object_count"),
        (TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value, {"entity_ambiguity": 1}, "attribute_ambiguity"),
        (TaskFamily.NOOP_WRITE_DISCIPLINE.value, {"noop_density": 0.5}, "write_trap_type"),
        (TaskFamily.DELETION_FORGETTING.value, {"deletion_scope": "attribute"}, "relearning_condition"),
        (TaskFamily.CURRENT_HISTORICAL_QUERY.value, {"query_type": "historical_state"}, "requested_version_distance"),
        (TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value, {"reasoning_depth": 2}, "active_object_count"),
        (TaskFamily.REALISTIC_SOURCE_UPDATE.value, {"source_type": "dialogue"}, "provenance_class"),
    ],
)
def test_known_families_require_resolved_profile_keys(make_task, family, profile, missing):
    task = make_task()
    malformed = _replace(task, task_family=family, metadata=_replace(task.metadata, resolved_profile=profile))
    assert f"metadata.resolved_profile.{missing}" in _paths(validate_task(malformed), "missing_family_profile_key")



def test_top_level_constructed_discriminators_are_validated(make_task):
    task = make_task()
    malformed = _replace(
        task,
        task_family=" ",
        difficulty="bogus",
        source=_replace(task.source, source_type="bogus"),
        metadata=_replace(task.metadata, split="bogus"),
    )
    report = validate_task(malformed)
    assert [(issue.code, issue.path) for issue in report.issues if issue.code.startswith("invalid_")] == [
        ("invalid_task_family", "task_family"),
        ("invalid_difficulty", "difficulty"),
        ("invalid_source_type", "source.source_type"),
        ("invalid_split", "metadata.split"),
    ]



def test_profile_name_must_match_top_level_difficulty(make_task):
    task = make_task()
    mismatched = _replace(task, difficulty=Difficulty.HARD)
    report = validate_task(mismatched)
    assert [(issue.code, issue.path) for issue in report.issues if issue.code == "profile_difficulty_mismatch"] == [
        ("profile_difficulty_mismatch", "metadata.profile_name")
    ]
    assert "profile_difficulty_mismatch" not in _codes(validate_task(task))


def test_unknown_future_family_is_schema_compatible(make_task):
    task = make_task()
    codes = _codes(validate_task(_replace(task, task_family="future_family", metadata=_replace(task.metadata, resolved_profile={}))))
    assert "unknown_task_family" not in codes
    assert "missing_family_profile_key" not in codes


@pytest.mark.parametrize(
    ("operation", "targets", "value", "expected_code"),
    [
        (Operation.NOOP, "one", "value", "invalid_noop_shape"),
        (Operation.ADD, "none", "value", "invalid_write_shape"),
        (Operation.UPDATE, "one", None, "invalid_write_shape"),
        (Operation.DELETE, "one", "value", "invalid_delete_shape"),
        (Operation.DELETE, "none", None, "invalid_delete_shape"),
    ],
)
def test_malformed_action_shapes_are_collected(make_task, operation, targets, value, expected_code):
    task = make_task()
    target_list = [task.target_objects[0]] if targets == "one" else []
    action = _replace(task.gold.actions[0], operation=operation, target_object_keys=target_list, value=value)
    malformed = _replace(task, gold=_replace(task.gold, actions=[action, task.gold.actions[1]]))
    assert expected_code in _codes(validate_task(malformed))


def test_duplicate_action_targets_use_canonical_identity_not_object_type(make_task):
    task = make_task()
    key = task.target_objects[0]
    action = _replace(task.gold.actions[0], target_object_keys=[key, _replace(key, object_type="different")])
    malformed = _replace(task, gold=_replace(task.gold, actions=[action, task.gold.actions[1]]))
    assert "duplicate_action_target" in _codes(validate_task(malformed))


@pytest.mark.parametrize(
    ("schema", "bad_value"),
    [
        (AnswerSchema.STRING, 1),
        (AnswerSchema.NUMBER, True),
        (AnswerSchema.NUMBER, float("inf")),
        (AnswerSchema.BOOLEAN, 1),
        (AnswerSchema.LIST, {"not": "list"}),
        (AnswerSchema.OBJECT, ["not", "object"]),
    ],
)
def test_gold_and_acceptable_answers_match_query_schema(make_task, schema, bad_value):
    task = make_task()
    query = _replace(task.queries[0], answer_schema=schema)
    gold = _replace(task.gold, gold_answers={"query_0": bad_value}, acceptable_answers={"query_0": bad_value})
    report = validate_task(_replace(task, queries=[query], gold=gold))
    assert "invalid_gold_answer_schema" in _codes(report)
    assert "invalid_acceptable_answer_schema" in _codes(report)


def test_answer_maps_must_exactly_cover_queries(make_task):
    task = make_task()
    gold = _replace(task.gold, gold_answers={"unknown": "x"}, acceptable_answers={})
    codes = _codes(validate_task(_replace(task, gold=gold)))
    assert "missing_gold_answer" in codes
    assert "unknown_gold_answer_query" in codes
    assert "missing_acceptable_answer" in codes


def test_current_and_historical_queries_require_targets_and_strict_version_index(make_task):
    task = make_task()
    current = _replace(task.queries[0], target_object_keys=[])
    historical = _replace(task.queries[0], query_id="query_h", query_type=QueryType.HISTORICAL_STATE, target_object_keys=[], metadata={"version_index": True})
    gold = _replace(task.gold, gold_answers={"query_0": "Qingdao", "query_h": "Dalian"}, acceptable_answers={"query_0": ["Qingdao"], "query_h": ["Dalian"]})
    codes = _codes(validate_task(_replace(task, queries=[current, historical], gold=gold)))
    assert "missing_current_query_target" in codes
    assert "missing_historical_query_target" in codes
    assert "invalid_historical_version_index" in codes


def test_deletion_compliance_target_must_be_declared_or_expected_absent(make_task):
    task = make_task()
    unknown = MemoryObjectKey(object_type="slot", namespace="default", entity="alex", attribute="deleted")
    query = _replace(task.queries[0], query_type=QueryType.DELETION_COMPLIANCE, target_object_keys=[unknown])
    assert "undeclared_deletion_query_target" in _codes(validate_task(_replace(task, queries=[query])))


def test_all_object_references_and_state_keys_must_be_declared(make_task):
    task = make_task()
    unknown = MemoryObjectKey(object_type="slot", namespace="default", entity="other", attribute="location")
    action = _replace(task.gold.actions[0], target_object_keys=[unknown])
    query = _replace(task.queries[0], target_object_keys=[unknown])
    gold = _replace(task.gold, actions=[action, task.gold.actions[1]], expected_present_objects=[unknown], expected_absent_objects=[unknown], final_state={unknown.canonical_id: "x"}, version_history={unknown.canonical_id: ["x"]})
    codes = _codes(validate_task(_replace(task, queries=[query], gold=gold)))
    for expected in ("undeclared_action_target", "undeclared_query_target", "undeclared_expected_present_target", "undeclared_final_state_key", "undeclared_version_history_key"):
        assert expected in codes


def test_expected_sets_cannot_overlap_and_absent_cannot_be_in_final_state(make_task):
    task = make_task()
    key = task.target_objects[0]
    codes = _codes(validate_task(_replace(task, gold=_replace(task.gold, expected_absent_objects=[key]))))
    assert "expected_presence_overlap" in codes
    assert "expected_absent_in_final_state" in codes


def test_object_identity_resolution_ignores_object_type(make_task):
    task = make_task()
    alternate = _replace(task.target_objects[0], object_type="record")
    action0 = _replace(task.gold.actions[0], target_object_keys=[alternate])
    query = _replace(task.queries[0], target_object_keys=[alternate])
    gold = _replace(task.gold, actions=[action0, task.gold.actions[1]], expected_present_objects=[alternate])
    codes = _codes(validate_task(_replace(task, queries=[query], gold=gold)))
    assert not any(code.startswith("undeclared_") for code in codes)


def test_malformed_model_construct_collections_are_reported(make_task):
    task = make_task()
    action = _replace(task.gold.actions[0], operation=Operation.NOOP, target_object_keys="not-a-list", value=None)
    query = _replace(task.queries[0], target_object_keys="not-a-list")
    malformed = _replace(task, queries=[query], gold=_replace(task.gold, actions=[action, task.gold.actions[1]]))
    codes = _codes(validate_task(malformed))
    assert "malformed_action_targets" in codes
    assert "malformed_query_targets" in codes



def test_empty_acceptable_support_is_invalid(make_task):
    task = make_task()
    gold = _replace(task.gold, acceptable_answers={"query_0": []})
    assert "invalid_acceptable_answer_schema" in _codes(validate_task(_replace(task, gold=gold)))


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        (lambda t: _replace(t, events={}), "malformed_events", "events"),
        (lambda t: _replace(t, target_objects={}), "malformed_target_objects", "target_objects"),
        (lambda t: _replace(t, queries={}), "malformed_queries", "queries"),
        (lambda t: _replace(t, gold=_replace(t.gold, actions={})), "malformed_gold_actions", "gold.actions"),
        (lambda t: _replace(t, gold=_replace(t.gold, action_sequence={})), "malformed_action_sequence", "gold.action_sequence"),
        (lambda t: _replace(t, gold=_replace(t.gold, expected_present_objects={})), "malformed_expected_present_objects", "gold.expected_present_objects"),
        (lambda t: _replace(t, gold=_replace(t.gold, expected_absent_objects={})), "malformed_expected_absent_objects", "gold.expected_absent_objects"),
        (lambda t: _replace(t, gold=_replace(t.gold, gold_source_event_ids={})), "malformed_gold_source_event_ids", "gold.gold_source_event_ids"),
        (lambda t: _replace(t, gold=_replace(t.gold, final_state=[])), "malformed_final_state", "gold.final_state"),
        (lambda t: _replace(t, gold=_replace(t.gold, version_history=[])), "malformed_version_history", "gold.version_history"),
        (lambda t: _replace(t, gold=_replace(t.gold, gold_answers=[])), "malformed_gold_answers", "gold.gold_answers"),
        (lambda t: _replace(t, gold=_replace(t.gold, acceptable_answers=[])), "malformed_acceptable_answers", "gold.acceptable_answers"),
        (lambda t: _replace(t, metadata=_replace(t.metadata, resolved_profile=[])), "malformed_resolved_profile", "metadata.resolved_profile"),
        (lambda t: _replace(t, events=[_replace(t.events[0], source_anchor=[]), t.events[1]]), "malformed_source_anchor", "events[0].source_anchor"),
        (lambda t: _replace(t, events=[_replace(t.events[0], gold_action_ids={}), t.events[1]]), "malformed_event_action_ids", "events[0].gold_action_ids"),
    ],
)
def test_malformed_containers_have_field_specific_issues(make_task, mutate, code, path):
    report = validate_task(mutate(make_task()))
    assert report.valid is False
    assert path in _paths(report, code)


@pytest.mark.parametrize("source_id", [None, "", " ", "wrong"])
def test_present_source_anchor_id_must_be_nonblank_exact_match(make_task, source_id):
    task = make_task()
    event = _replace(task.events[0], source_anchor={"source_id": source_id})
    assert "source_anchor_source_mismatch" in _codes(validate_task(_replace(task, events=[event, task.events[1]])))


def test_deletion_query_may_resolve_from_expected_absent_only(make_task):
    task = make_task()
    absent = MemoryObjectKey(object_type="slot", namespace="default", entity="gone", attribute="location")
    query = _replace(task.queries[0], query_type=QueryType.DELETION_COMPLIANCE, target_object_keys=[absent], answer_schema=AnswerSchema.BOOLEAN)
    gold = _replace(task.gold, expected_absent_objects=[absent], gold_answers={"query_0": True}, acceptable_answers={"query_0": [True]})
    codes = _codes(validate_task(_replace(task, queries=[query], gold=gold)))
    assert "undeclared_deletion_query_target" not in codes
    assert "undeclared_expected_absent_target" not in codes



def test_constructed_closed_enums_targets_and_duplicate_query_targets_are_rejected(make_task):
    task = make_task(); key = task.target_objects[0]
    action = _replace(task.gold.actions[0], operation="BOGUS", scope="bogus")
    event = _replace(task.events[0], role="bogus")
    duplicate = _replace(key, object_type="other")
    query = _replace(task.queries[0], query_type="bogus", answer_schema="bogus", evaluation_mode="bogus", target_object_keys=[key, duplicate])
    report = validate_task(_replace(task, events=[event, task.events[1]], queries=[query], gold=_replace(task.gold, actions=[action, task.gold.actions[1]])))
    for code in ("invalid_action_operation", "invalid_action_scope", "invalid_event_role", "invalid_query_type", "invalid_answer_schema", "invalid_evaluation_mode", "duplicate_query_target"):
        assert code in _codes(report)
    transition = _replace(task.queries[0], query_type=QueryType.TRANSITION, target_object_keys=[])
    assert "missing_query_target" in _codes(validate_task(_replace(task, queries=[transition])))


def test_unhashable_ids_do_not_stop_later_independent_collection(make_task):
    task = make_task()
    query = _replace(task.queries[0], query_id=[])
    malformed = _replace(task, schema_version="bad", task_id=" ", queries=[query], metadata=_replace(task.metadata, resolved_profile={}))
    codes = _codes(validate_task(malformed))
    assert "unsupported_schema_version" in codes
    assert "blank_task_id" in codes
    assert "blank_query_id" in codes
    assert "missing_family_profile_key" in codes
    assert "malformed_task_structure" not in codes
