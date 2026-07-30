from __future__ import annotations

import json
from enum import Enum

import pytest
from pydantic import ValidationError

from mub.vnext.audit.sample import (
    AuditDecision,
    AuditDecisionTemplate,
    AuditGateReport,
    AuditSelection,
    audit_decision_template,
    evaluate_audit_gate,
)
from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.io.jsonl import read_models, write_models


def selection(audit_id: str = "audit-1", **updates) -> AuditSelection:
    payload = {
        "audit_id": audit_id,
        "task_id": "task-1",
        "family": TaskFamily.REPEATED_SAME_SLOT,
        "difficulty": Difficulty.EASY,
        "split": Split.DEV,
        "covered_conditions": ["surface:direct", "order:chronological"],
        "selection_reason": "covers the baseline condition",
    }
    payload.update(updates)
    return AuditSelection(**payload)


def decision(audit_id: str = "audit-1", **updates) -> AuditDecision:
    payload = {
        "audit_id": audit_id,
        "reviewer": "reviewer-1",
        "verdict": "pass",
        "answer_unique": True,
        "actions_correct": True,
        "roles_correct": True,
        "surface_natural": True,
        "notes": "reviewed",
    }
    payload.update(updates)
    return AuditDecision(**payload)


def test_records_are_frozen_with_exact_fields_and_strict_values() -> None:
    assert set(AuditSelection.model_fields) == {
        "audit_id",
        "task_id",
        "family",
        "difficulty",
        "split",
        "covered_conditions",
        "selection_reason",
    }
    assert set(AuditDecision.model_fields) == {
        "audit_id",
        "reviewer",
        "verdict",
        "answer_unique",
        "actions_correct",
        "roles_correct",
        "surface_natural",
        "notes",
    }
    selected = selection()
    reviewed = decision()
    with pytest.raises((TypeError, ValidationError)):
        selected.audit_id = "other"
    with pytest.raises((TypeError, ValidationError)):
        reviewed.verdict = "block"
    with pytest.raises((TypeError, AttributeError)):
        selected.covered_conditions.append("new")
    with pytest.raises(TypeError):
        list.append(selected.covered_conditions, "new")
    assert isinstance(selected.covered_conditions, tuple)
    assert selected.covered_conditions == ("order:chronological", "surface:direct")
    with pytest.raises(ValidationError):
        selection(covered_conditions=["order:chronological", "order:chronological"])
    with pytest.raises(ValidationError):
        decision(answer_unique=1)
    with pytest.raises(ValidationError):
        selection(audit_id=" ", selection_reason="ok")
    with pytest.raises(ValidationError):
        decision(reviewer=" ")


def test_blank_template_is_not_a_decision_and_never_release_ready() -> None:
    template = audit_decision_template("audit-1")
    assert isinstance(template, AuditDecisionTemplate)
    assert not isinstance(template, AuditDecision)
    assert template.audit_id == "audit-1"
    assert template.reviewer is None
    assert template.verdict is None
    assert template.answer_unique is None
    assert template.actions_correct is None
    assert template.roles_correct is None
    assert template.surface_natural is None
    assert template.notes is None
    assert template.release_ready is False
    with pytest.raises(ValidationError):
        AuditDecision.model_validate(template.model_dump(mode="python"))
    with pytest.raises(ValidationError):
        AuditDecisionTemplate(audit_id="audit-1", reviewer="fabricated")
    report = evaluate_audit_gate([selection()], [template])
    assert report.release_ready is False
    assert report.missing_audit_ids == ("audit-1",)
    assert report.malformed_decision_ids == ("audit-1",)


def test_gate_report_readiness_requires_normalized_decision_evidence() -> None:
    with pytest.raises(ValidationError):
        AuditGateReport(selected_audit_ids=("audit-1",), release_ready=True)

    with pytest.raises(ValidationError):
        AuditGateReport(
            selected_audit_ids=("audit-1",),
            passed_audit_ids=("audit-1",),
            release_ready=True,
        )

    report = AuditGateReport(
        selected_audit_ids=("audit-1",),
        decision_evidence=(decision("audit-1"),),
    )
    assert report.release_ready is True
    assert report.passed_audit_ids == ("audit-1",)
    assert report.model_dump()["release_ready"] is True
    assert report.model_dump()["decision_evidence"][0]["audit_id"] == "audit-1"
    with pytest.raises(ValidationError):
        AuditGateReport(selected_audit_ids=("audit-2", "audit-1"))
    with pytest.raises(ValidationError):
        AuditGateReport(selected_audit_ids=("audit-1", "audit-1"))


@pytest.mark.parametrize(
    "evidence",
    (
        (),
        (decision("audit-1"), decision("audit-1")),
        (decision("audit-1"), decision("foreign")),
        (decision("audit-1", verdict="block"),),
        (decision("audit-1", answer_unique=False),),
    ),
)
def test_gate_report_requires_exactly_one_complete_pass_per_selection(evidence) -> None:
    report = AuditGateReport(
        selected_audit_ids=("audit-1",),
        decision_evidence=evidence,
    )

    assert report.release_ready is False


def test_gate_report_rejects_hostile_constructed_decision_evidence() -> None:
    malformed = AuditDecision.model_construct(
        audit_id="audit-1",
        reviewer=" ",
        verdict="pass",
        answer_unique=True,
        actions_correct=True,
        roles_correct=True,
        surface_natural=True,
        notes="fabricated",
    )

    with pytest.raises(ValidationError):
        AuditGateReport(
            selected_audit_ids=("audit-1",),
            decision_evidence=(malformed,),
        )


def test_selection_jsonl_round_trip_accepts_canonical_enum_strings(tmp_path) -> None:
    path = tmp_path / "selections.jsonl"
    selected = selection()

    write_models(path, [selected], id_field="audit_id")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["family"] == TaskFamily.REPEATED_SAME_SLOT.value
    assert payload["difficulty"] == Difficulty.EASY.value
    assert payload["split"] == Split.DEV.value
    assert payload["covered_conditions"] == ["order:chronological", "surface:direct"]

    loaded = list(read_models(path, AuditSelection, id_field="audit_id"))
    assert loaded == [selected]
    assert loaded[0].family is TaskFamily.REPEATED_SAME_SLOT
    assert loaded[0].difficulty is Difficulty.EASY
    assert loaded[0].split is Split.DEV
    assert isinstance(loaded[0].covered_conditions, tuple)


def test_literal_index_prefix_is_reported_as_foreign_id() -> None:
    report = evaluate_audit_gate(
        [selection("audit-1")],
        [decision("<index:foreign>")],
    )

    assert report.foreign_audit_ids == ("<index:foreign>",)
    assert report.missing_audit_ids == ("audit-1",)


@pytest.mark.parametrize(
    ("updates", "expected_field"),
    (
        ({"verdict": "block"}, "non_pass_audit_ids"),
        ({"verdict": "needs_revision"}, "non_pass_audit_ids"),
        ({"answer_unique": False}, "failed_check_audit_ids"),
        ({"actions_correct": False}, "failed_check_audit_ids"),
        ({"roles_correct": False}, "failed_check_audit_ids"),
        ({"surface_natural": False}, "failed_check_audit_ids"),
    ),
)
def test_each_terminal_decision_failure_blocks_release(updates, expected_field) -> None:
    report = evaluate_audit_gate([selection()], [decision(**updates)])

    assert report.release_ready is False
    assert getattr(report, expected_field) == ("audit-1",)


def test_gate_accepts_one_all_pass_decision_and_is_deterministic() -> None:
    selections = [selection("audit-2"), selection("audit-1")]
    decisions = [decision("audit-1"), decision("audit-2")]

    report = evaluate_audit_gate(selections, decisions)

    assert isinstance(report, AuditGateReport)
    assert report.release_ready is True
    assert report.missing_audit_ids == ()
    assert report.duplicate_audit_ids == ()
    assert report.foreign_audit_ids == ()
    assert report.malformed_decision_ids == ()
    assert report.failed_check_audit_ids == ()
    assert report.non_pass_audit_ids == ()
    assert report.selected_audit_ids == ("audit-1", "audit-2")
    assert report == evaluate_audit_gate(tuple(reversed(selections)), tuple(reversed(decisions)))


def test_gate_reports_missing_duplicate_foreign_and_failed_decisions() -> None:
    report = evaluate_audit_gate(
        [selection("audit-1"), selection("audit-2")],
        [
            decision("audit-1", verdict="block"),
            decision("audit-1", verdict="needs_revision", actions_correct=False),
            decision("foreign"),
        ],
    )

    assert report.release_ready is False
    assert report.missing_audit_ids == ("audit-2",)
    assert report.duplicate_audit_ids == ("audit-1",)
    assert report.foreign_audit_ids == ("foreign",)
    assert report.non_pass_audit_ids == ("audit-1",)
    assert report.failed_check_audit_ids == ("audit-1",)


def test_gate_rejects_malformed_model_construct_and_wrong_iterables() -> None:
    malformed = AuditDecision.model_construct(
        audit_id="audit-1",
        reviewer="reviewer-1",
        verdict="pass",
        answer_unique=True,
        actions_correct=True,
        roles_correct=True,
        surface_natural=[],
        notes="reviewed",
    )
    cyclic = []
    cyclic.append(cyclic)
    malformed_selection = AuditSelection.model_construct(
        audit_id="audit-selection",
        task_id="task-1",
        family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        split=Split.DEV,
        covered_conditions=cyclic,
        selection_reason="reason",
    )

    report = evaluate_audit_gate(
        [selection("audit-1")],
        [malformed, cyclic],
    )

    assert report.release_ready is False
    assert report.malformed_decision_ids == ("audit-1", "<index:1>")
    selection_report = evaluate_audit_gate([malformed_selection], [])
    assert selection_report.release_ready is False
    assert selection_report.malformed_selection_ids == ("audit-selection",)
    with pytest.raises(TypeError):
        evaluate_audit_gate(iter([selection("audit-1")]), [])
    with pytest.raises(TypeError):
        evaluate_audit_gate([selection("audit-1")], {decision("audit-1")})


def test_template_and_records_reject_wrong_iterable_condition_payloads() -> None:
    with pytest.raises(ValidationError):
        selection(covered_conditions=(condition for condition in ["one"]))
    with pytest.raises(ValidationError):
        selection(covered_conditions=["ok", 3])
    with pytest.raises(ValidationError):
        audit_decision_template(" ")


class _UnexpectedEnum(Enum):
    VALUE = "easy"


def test_exact_enum_types_are_not_coerced() -> None:
    selected = selection(family=TaskFamily.REPEATED_SAME_SLOT.value)
    assert selected.family is TaskFamily.REPEATED_SAME_SLOT
    with pytest.raises(ValidationError):
        selection(family="not-a-task-family")
    with pytest.raises(ValidationError):
        selection(family=_UnexpectedEnum.VALUE)
