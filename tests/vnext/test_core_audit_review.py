from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from mub.vnext.audit.core import select_core_audit_sample
from mub.vnext.audit.core_review import (
    CoreAuditChecks,
    CoreAuditDecision,
    CoreAuditDecisionTemplate,
    applicable_core_audit_checks,
    core_audit_decision_templates,
    evaluate_core_audit_gate,
)
from mub.vnext.contracts import TaskFamily
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import sha256_model


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def selection_package():
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=40, code_revision=revision)
    manifest = build_core_artifact_bundle(snapshot, config).task_manifest
    return select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )


def _checks(family: TaskFamily, *, failed: str | None = None) -> CoreAuditChecks:
    payload = {}
    for field in CoreAuditChecks.model_fields:
        payload[field] = (
            "not_applicable"
            if field not in applicable_core_audit_checks(family)
            else "pass"
        )
    if failed is not None:
        payload[failed] = "fail"
    return CoreAuditChecks(**payload)


def _decision(package, selected, role: str, *, reviewer: str | None = None, **updates):
    payload = {
        "audit_id": selected.audit_id,
        "task_id": selected.task_id,
        "task_hash": selected.task_hash,
        "source_task_manifest_hash": package.source_task_manifest_hash,
        "selection_hash": package.selection_hash,
        "reviewer_id": reviewer or f"reviewer-{role}",
        "reviewer_role": role,
        "review_record_id": f"record-{selected.audit_id}-{role}",
        "independent_review_attestation": True,
        "verdict": "pass",
        "checks": _checks(selected.family),
        "task_specific_observation": f"Checked exact task {selected.task_id} as {role}",
        "notes": "",
    }
    payload.update(updates)
    return CoreAuditDecision(**payload)


def _all_passing_decisions(package):
    decisions = []
    for selected in package.selections:
        decisions.append(_decision(package, selected, "primary"))
        if selected.family in {
            TaskFamily.DELETION_FORGETTING,
            TaskFamily.CURRENT_HISTORICAL_QUERY,
            TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
        }:
            decisions.append(
                _decision(
                    package,
                    selected,
                    "secondary",
                    reviewer="reviewer-secondary",
                )
            )
    return decisions


def test_templates_are_blank_role_complete_and_never_release_ready(
    selection_package,
) -> None:
    templates = core_audit_decision_templates(selection_package)

    assert len(templates) == 320
    assert all(type(item) is CoreAuditDecisionTemplate for item in templates)
    assert all(item.release_ready is False for item in templates)
    assert Counter(item.reviewer_role for item in templates) == {
        "primary": 224,
        "secondary": 96,
    }
    with pytest.raises(ValidationError):
        CoreAuditDecision.model_validate(templates[0].model_dump(mode="python"))
    report = evaluate_core_audit_gate(selection_package, templates, ())
    assert report.release_ready is False
    assert len(report.missing_review_roles) == 320


def test_all_independent_terminal_passes_open_gate_and_report_agreement(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)

    report = evaluate_core_audit_gate(selection_package, decisions, ())

    assert report.release_ready is True
    assert len(report.terminal_pass_audit_ids) == 224
    assert report.required_adjudication_ids == ()
    assert report.raw_agreement == 1.0
    assert report.cohens_kappa is None
    assert report.agreement_item_count > 96


def test_disagreement_and_nonpass_require_terminal_adjudication(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected_e = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.DELETION_FORGETTING
    )
    selected_a = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.REPEATED_SAME_SLOT
    )
    decisions = [
        item
        for item in decisions
        if not (
            (item.audit_id == selected_e.audit_id and item.reviewer_role == "secondary")
            or (item.audit_id == selected_a.audit_id and item.reviewer_role == "primary")
        )
    ]
    decisions.extend(
        [
            _decision(
                selection_package,
                selected_e,
                "secondary",
                reviewer="reviewer-secondary",
                verdict="needs_revision",
                checks=_checks(selected_e.family, failed="selector_history_evidence_correct"),
            ),
            _decision(
                selection_package,
                selected_a,
                "primary",
                verdict="block",
                checks=_checks(selected_a.family, failed="surface_natural"),
            ),
        ]
    )

    unresolved = evaluate_core_audit_gate(selection_package, decisions, ())
    assert unresolved.release_ready is False
    assert unresolved.required_adjudication_ids == tuple(
        sorted((selected_a.audit_id, selected_e.audit_id))
    )
    adjudications = [
        _decision(
            selection_package,
            selected,
            "adjudicator",
            reviewer="reviewer-adjudicator",
        )
        for selected in (selected_a, selected_e)
    ]
    resolved = evaluate_core_audit_gate(
        selection_package, decisions, adjudications
    )
    assert resolved.release_ready is True
    assert set(resolved.adjudicated_audit_ids) == {
        selected_a.audit_id,
        selected_e.audit_id,
    }


@pytest.mark.parametrize(
    "corruption",
    (
        "duplicate_role",
        "unknown_id",
        "binding_hash",
        "non_independent",
        "copied_record",
        "copied_observation",
    ),
)
def test_gate_fails_closed_on_review_evidence_corruption(
    selection_package,
    corruption: str,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected_e = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.DELETION_FORGETTING
    )
    primary_index = next(
        index
        for index, item in enumerate(decisions)
        if item.audit_id == selected_e.audit_id and item.reviewer_role == "primary"
    )
    secondary_index = next(
        index
        for index, item in enumerate(decisions)
        if item.audit_id == selected_e.audit_id and item.reviewer_role == "secondary"
    )
    if corruption == "duplicate_role":
        decisions.append(decisions[primary_index].model_copy(update={"review_record_id": "duplicate-role-record"}))
    elif corruption == "unknown_id":
        decisions.append(decisions[primary_index].model_copy(update={"audit_id": "unknown-audit", "review_record_id": "unknown-record"}))
    elif corruption == "binding_hash":
        decisions[primary_index] = decisions[primary_index].model_copy(update={"selection_hash": "f" * 64})
    elif corruption == "non_independent":
        decisions[secondary_index] = decisions[secondary_index].model_copy(update={"reviewer_id": decisions[primary_index].reviewer_id})
    elif corruption == "copied_record":
        other = 0 if primary_index != 0 else 1
        decisions[primary_index] = decisions[primary_index].model_copy(update={"review_record_id": decisions[other].review_record_id})
    else:
        other = 0 if primary_index != 0 else 1
        decisions[primary_index] = decisions[primary_index].model_copy(update={"task_specific_observation": decisions[other].task_specific_observation})

    report = evaluate_core_audit_gate(selection_package, decisions, ())

    assert report.release_ready is False
    assert report.issues


def test_family_aware_applicability_rejects_fake_not_applicable_values(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.CURRENT_HISTORICAL_QUERY
    )
    index = next(
        index
        for index, item in enumerate(decisions)
        if item.audit_id == selected.audit_id and item.reviewer_role == "primary"
    )
    fake = _checks(selected.family).model_copy(
        update={"selector_history_evidence_correct": "not_applicable"}
    )
    decisions[index] = decisions[index].model_copy(update={"checks": fake})

    report = evaluate_core_audit_gate(selection_package, decisions, ())

    assert report.release_ready is False
    assert selected.audit_id in report.invalid_applicability_audit_ids
