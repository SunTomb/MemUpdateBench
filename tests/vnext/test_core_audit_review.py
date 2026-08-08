from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import subprocess

import pytest
from pydantic import ValidationError

import mub.vnext.audit.core as core_audit
import mub.vnext.audit.core_review as core_review
import mub.vnext.audit.core_stage as core_stage
from mub.vnext.audit.core_candidate import (
    CoreCandidateValidationReceipt,
    core_candidate_receipt_hash,
)
from mub.vnext.audit.core import (
    core_audit_review_context_hash,
    select_core_audit_sample,
)
from mub.vnext.audit.core_review import (
    CoreAuditChecks,
    CoreAuditDecision,
    CoreAuditDecisionTemplate,
    CoreAuditGateReport,
    applicable_core_audit_checks,
    core_audit_adjudication_templates,
    core_audit_decision_templates,
    evaluate_core_audit_gate,
)
from mub.vnext.contracts import TaskFamily
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import canonical_json_bytes, sha256_model


ROOT = Path(__file__).resolve().parents[2]
_TASKS_BY_SELECTION_HASH = {}
_MANIFEST_BY_SELECTION_HASH = {}
_RECEIPT_BY_SELECTION_HASH = {}
_BOUNDED_PACKAGE = None
_BOUNDED_MANIFEST = None


def _test_receipt(manifest, *, expected_full: bool):
    payload = {
        "receipt_version": "core-candidate-validation-v1",
        "candidate_dir": str(ROOT),
        "expected_full": expected_full,
        "source_task_manifest_hash": sha256_model(manifest),
        "tasks_artifact_hash": "a" * 64,
        "task_count": sum(manifest.split_counts.values()),
        "code_revision": manifest.code_revision,
        "trusted_audit_tooling_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
    }
    return CoreCandidateValidationReceipt(
        **payload,
        receipt_hash=core_candidate_receipt_hash(payload),
    )


def _promote_to_full_manifest_and_package(manifest, package):
    manifest_payload = manifest.model_dump(mode="python")
    hashes = dict(manifest.task_record_hashes)
    missing = 12_000 - len(hashes)
    hashes.update(
        {
            f"full-padding-task-{index:05d}": hashlib.sha256(
                f"full-padding-hash-{index}".encode("ascii")
            ).hexdigest()
            for index in range(missing)
        }
    )
    manifest_payload["task_record_hashes"] = hashes
    manifest_payload["split_counts"] = {
        "train": 8_400,
        "dev": 1_200,
        "test": 2_400,
    }
    manifest_payload["code_revision"] = (
        "1a2d2f8c80b64f08221056b35cf64cc3390d62a8"
    )
    task_refs = list(manifest.task_file_paths_and_hashes)
    task_refs[0] = task_refs[0].model_copy(update={"record_count": 12_000})
    manifest_payload["task_file_paths_and_hashes"] = tuple(task_refs)
    full_manifest = type(manifest).model_validate(manifest_payload)
    manifest_hash = sha256_model(full_manifest)
    config_hash = package.selector_config_hash
    rows = []
    for row in package.selections:
        pending = row.model_copy(update={"audit_id": "pending"})
        rows.append(
            pending.model_copy(
                update={
                    "audit_id": core_audit._audit_id(
                        pending,
                        source_task_manifest_hash=manifest_hash,
                        selector_config_hash=config_hash,
                    )
                }
            )
        )
    payload = object.__getattribute__(package, "__dict__").copy()
    payload.update(
        {
            "source_task_manifest_hash": manifest_hash,
            "selections": tuple(rows),
        }
    )
    payload["selection_hash"] = core_audit.core_audit_selection_hash(payload)
    return full_manifest, type(package).model_validate(payload)


@pytest.fixture(autouse=True)
def accept_test_candidate_receipts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        core_review,
        "verify_core_candidate_validation_receipt",
        lambda receipt: receipt in _RECEIPT_BY_SELECTION_HASH.values(),
    )


@pytest.fixture(scope="module")
def selection_package():
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=40, code_revision=revision)
    manifest = build_core_artifact_bundle(snapshot, config).task_manifest
    bounded_package = select_core_audit_sample(
        snapshot.tasks,
        manifest,
        source_task_manifest_hash=sha256_model(manifest),
    )
    full_manifest, package = _promote_to_full_manifest_and_package(
        manifest, bounded_package
    )
    global _BOUNDED_PACKAGE, _BOUNDED_MANIFEST
    _BOUNDED_PACKAGE = bounded_package
    _BOUNDED_MANIFEST = manifest
    task_map = {task.task_id: task for task in snapshot.tasks}
    _TASKS_BY_SELECTION_HASH[package.selection_hash] = task_map
    _TASKS_BY_SELECTION_HASH[bounded_package.selection_hash] = task_map
    _MANIFEST_BY_SELECTION_HASH[package.selection_hash] = full_manifest
    _MANIFEST_BY_SELECTION_HASH[bounded_package.selection_hash] = manifest
    _RECEIPT_BY_SELECTION_HASH[package.selection_hash] = _test_receipt(
        full_manifest, expected_full=True
    )
    _RECEIPT_BY_SELECTION_HASH[bounded_package.selection_hash] = _test_receipt(
        manifest, expected_full=False
    )
    return package


def _context(package):
    by_id = _TASKS_BY_SELECTION_HASH[package.selection_hash]
    selected = tuple(by_id[item.task_id] for item in package.selections)
    surfaces = tuple(
        by_id[variant.task_id]
        for item in package.selections
        for variant in item.surface_variants
    )
    return selected, surfaces


def _gate(package, decisions, adjudications):
    selected, surfaces = _context(package)
    return evaluate_core_audit_gate(
        package,
        decisions,
        adjudications,
        source_task_manifest=_MANIFEST_BY_SELECTION_HASH[package.selection_hash],
        candidate_validation_receipt=_RECEIPT_BY_SELECTION_HASH.get(
            package.selection_hash
        ),
        selected_tasks=selected,
        surface_context_tasks=surfaces,
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
        "review_context_hash": core_audit_review_context_hash(package),
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
    adjudication_templates = core_audit_adjudication_templates(
        selection_package,
        (item.audit_id for item in selection_package.selections),
    )
    assert len(adjudication_templates) == 224
    assert {item.reviewer_role for item in adjudication_templates} == {"adjudicator"}
    with pytest.raises(ValueError, match="unknown"):
        core_audit_adjudication_templates(selection_package, ("unknown-audit",))
    with pytest.raises(ValidationError):
        CoreAuditDecision.model_validate(templates[0].model_dump(mode="python"))
    report = _gate(selection_package, templates, ())
    assert report.release_ready is False
    assert len(report.missing_review_roles) == 320


def test_fully_passing_bounded_candidate_is_structurally_test_only(
    selection_package,
) -> None:
    del selection_package
    bounded = _BOUNDED_PACKAGE
    assert bounded is not None
    report = _gate(bounded, _all_passing_decisions(bounded), ())

    assert len(report.terminal_pass_audit_ids) == 224
    assert report.candidate_scope == "bounded_test"
    assert report.full_candidate is False
    assert report.release_ready is False


def test_all_independent_terminal_passes_open_gate_and_report_agreement(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)

    report = _gate(selection_package, decisions, ())

    assert report.release_ready is True
    assert report.candidate_scope == "full"
    assert report.full_candidate is True
    assert len(report.terminal_pass_audit_ids) == 224
    assert report.required_adjudication_ids == ()
    assert report.raw_agreement == 1.0
    assert report.cohens_kappa is None
    assert report.agreement_item_count > 96


def test_rejected_extra_evidence_remains_bound_after_issue_summary_tampering(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    rejected = decisions[0].model_copy(
        update={
            "audit_id": "foreign-extra-audit",
            "review_record_id": "foreign-extra-record",
        }
    )
    report = _gate(selection_package, (*decisions, rejected), ())
    restored = CoreAuditGateReport.model_validate_json(canonical_json_bytes(report))

    assert not restored.release_ready
    assert any(not observation.accepted for observation in restored.input_observations)
    tampered = restored.validated_replace(
        unknown_audit_ids=(),
        issues=(),
    )
    assert tampered.unknown_audit_ids == ()
    assert tampered.issues == ()
    assert tampered.release_ready is False


def test_gate_report_round_trips_typed_evidence_and_rejects_fabricated_readiness(
    selection_package,
) -> None:
    report = _gate(
        selection_package, _all_passing_decisions(selection_package), ()
    )

    restored = CoreAuditGateReport.model_validate_json(canonical_json_bytes(report))
    assert restored.release_ready is True
    assert all(type(item) is CoreAuditDecision for item in restored.decision_evidence)
    with pytest.raises(ValidationError, match="accepted input observations"):
        report.model_copy(
            update={
                "decision_evidence": (),
                "adjudication_evidence": (),
                "terminal_pass_audit_ids": tuple(
                    item.audit_id for item in selection_package.selections
                ),
                "issues": (),
            }
        )


def test_reviewer_ids_must_use_one_canonical_offline_identity(selection_package) -> None:
    selected = selection_package.selections[0]
    with pytest.raises(ValidationError, match="canonical"):
        _decision(selection_package, selected, "primary", reviewer="alice ")
    with pytest.raises(ValidationError, match="canonical"):
        _decision(selection_package, selected, "primary", reviewer="Alice")
    with pytest.raises(ValidationError, match="letters or digits"):
        _decision(
            selection_package,
            selected,
            "primary",
            task_specific_observation="✅🔥",
        )


def test_count_only_full_public_api_without_current_root_receipt_is_nonready(
    selection_package,
) -> None:
    selected, surfaces = _context(selection_package)
    report = evaluate_core_audit_gate(
        selection_package,
        _all_passing_decisions(selection_package),
        (),
        source_task_manifest=_MANIFEST_BY_SELECTION_HASH[
            selection_package.selection_hash
        ],
        selected_tasks=selected,
        surface_context_tasks=surfaces,
    )

    assert sum(report.source_task_manifest.split_counts.values()) == 12_000
    assert report.source_task_manifest.code_revision == (
        "1a2d2f8c80b64f08221056b35cf64cc3390d62a8"
    )
    assert len(report.terminal_pass_audit_ids) == 224
    assert report.candidate_validation_receipt is None
    assert report.candidate_scope == "unattested"
    assert report.full_candidate is False
    assert report.release_ready is False


def test_gate_api_requires_the_exact_authenticated_source_manifest(
    selection_package,
) -> None:
    selected, surfaces = _context(selection_package)
    manifest = _MANIFEST_BY_SELECTION_HASH[selection_package.selection_hash]
    fabricated = manifest.model_copy(
        update={"data_release_id": "fabricated-core-release"}
    )
    with pytest.raises(ValueError, match="source task manifest hash"):
        evaluate_core_audit_gate(
            selection_package,
            _all_passing_decisions(selection_package),
            (),
            source_task_manifest=fabricated,
            selected_tasks=selected,
            surface_context_tasks=surfaces,
        )


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

    unresolved = _gate(selection_package, decisions, ())
    assert unresolved.release_ready is False
    assert unresolved.required_adjudication_ids == tuple(
        sorted((selected_a.audit_id, selected_e.audit_id))
    )
    failed_adjudication = _decision(
        selection_package,
        selected_a,
        "adjudicator",
        reviewer="reviewer-adjudicator",
        verdict="block",
        checks=_checks(selected_a.family, failed="surface_natural"),
    )
    remediation_report = _gate(
        selection_package, decisions, (failed_adjudication,)
    )
    remediation = next(
        item
        for item in remediation_report.remediations
        if item.audit_id == selected_a.audit_id
    )
    assert remediation.required_action == "regenerate_reselect"
    assert remediation.generator_stratum.startswith(selected_a.family.value)
    assert remediation.template_stratum == selected_a.surface_id
    assert selected_a.audit_id in remediation_report.adjudicated_audit_ids
    assert selected_a.audit_id not in remediation_report.unresolved_adjudication_ids
    remediation_templates = core_stage._adjudication_templates_for_report(
        selection_package, remediation_report
    )
    assert selected_a.audit_id not in {
        item.audit_id for item in remediation_templates
    }
    assert selected_e.audit_id in {item.audit_id for item in remediation_templates}

    adjudications = [
        _decision(
            selection_package,
            selected,
            "adjudicator",
            reviewer="reviewer-adjudicator",
        )
        for selected in (selected_a, selected_e)
    ]
    resolved = _gate(
        selection_package, decisions, adjudications
    )
    assert resolved.release_ready is True
    assert set(resolved.adjudicated_audit_ids) == {
        selected_a.audit_id,
        selected_e.audit_id,
    }
    assert resolved.unresolved_adjudication_ids == ()
    assert core_stage._adjudication_templates_for_report(
        selection_package, resolved
    ) == ()


@pytest.mark.parametrize("ignorable", ("͏", "️", "\U000e0100"))
def test_observation_fingerprint_removes_default_ignorables_not_all_combining_marks(
    ignorable: str,
) -> None:
    assert core_review._normalized_observation("verified") == (
        core_review._normalized_observation(f"ver{ignorable}ified")
    )
    assert core_review._normalized_observation("cafe") != (
        core_review._normalized_observation("café")
    )


@pytest.mark.parametrize("different_reviewers", (False, True))
def test_copy_fingerprint_detects_cross_audit_reuse_regardless_of_reviewer(
    selection_package,
    different_reviewers: bool,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected = tuple(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.REPEATED_SAME_SLOT
    )[:2]
    indices = [
        next(
            index
            for index, decision in enumerate(decisions)
            if decision.audit_id == item.audit_id
        )
        for item in selected
    ]
    observations = (
        "Verified stale-update trajectory 17 against every gold event. ✅",
        "Verified stale update trajectory 17 against every gold event! 🔥",
    )
    for position, (index, observation) in enumerate(zip(indices, observations)):
        updates = {"task_specific_observation": observation}
        if different_reviewers:
            updates["reviewer_id"] = f"cross-audit-reviewer-{position}"
        decisions[index] = decisions[index].model_copy(update=updates)

    report = _gate(selection_package, decisions, ())

    assert set(report.copied_observation_audit_ids) >= {
        item.audit_id for item in selected
    }
    assert not ({item.audit_id for item in selected} & set(report.terminal_pass_audit_ids))
    assert not report.release_ready


def test_distinct_substantive_observations_pass_despite_empty_optional_notes(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)

    report = _gate(selection_package, decisions, ())

    assert all(decision.notes == "" for decision in decisions)
    assert report.copied_observation_audit_ids == ()
    assert report.release_ready


def test_copy_fingerprint_crosses_reviewer_roles_and_unicode_evasions(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.DELETION_FORGETTING
    )
    indices = {
        item.reviewer_role: index
        for index, item in enumerate(decisions)
        if item.audit_id == selected.audit_id
    }
    decisions[indices["primary"]] = decisions[indices["primary"]].model_copy(
        update={
            "task_specific_observation": (
                f"Checked exact task {selected.task_id} — independently."
            )
        }
    )
    decisions[indices["secondary"]] = decisions[indices["secondary"]].model_copy(
        update={
            "task_specific_observation": (
                f"Checked exact task {selected.task_id}​ - independently!"
            )
        }
    )

    report = _gate(selection_package, decisions, ())

    assert selected.audit_id in report.copied_observation_audit_ids
    assert selected.audit_id not in report.terminal_pass_audit_ids
    assert len(report.terminal_pass_audit_ids) == 223


def test_copied_reviewer_adjudicator_observation_is_human_input_not_remediation(
    selection_package,
) -> None:
    decisions = _all_passing_decisions(selection_package)
    selected = next(
        item
        for item in selection_package.selections
        if item.family is TaskFamily.REPEATED_SAME_SLOT
    )
    primary_index = next(
        index for index, item in enumerate(decisions) if item.audit_id == selected.audit_id
    )
    decisions[primary_index] = decisions[primary_index].model_copy(
        update={"verdict": "block", "checks": _checks(selected.family, failed="surface_natural")}
    )
    adjudicator = _decision(
        selection_package,
        selected,
        "adjudicator",
        reviewer="reviewer-adjudicator",
        task_specific_observation=decisions[primary_index].task_specific_observation,
    )

    report = _gate(selection_package, decisions, (adjudicator,))

    assert selected.audit_id in report.copied_observation_audit_ids
    assert selected.audit_id in report.unresolved_adjudication_ids
    assert not report.remediations


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
        decisions[primary_index] = decisions[primary_index].model_copy(
            update={
                "task_specific_observation": decisions[
                    secondary_index
                ].task_specific_observation
            }
        )

    report = _gate(selection_package, decisions, ())

    assert report.release_ready is False
    assert report.issues
    if corruption != "unknown_id":
        assert selected_e.audit_id not in report.terminal_pass_audit_ids
        assert report.agreement_item_count < 672


def test_decision_model_does_not_expose_family_blind_terminal_pass(
    selection_package,
) -> None:
    selected = selection_package.selections[0]
    decision = _decision(selection_package, selected, "primary")
    assert not hasattr(decision, "terminal_pass")
    assert not hasattr(decision.checks, "all_applicable_pass")


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

    report = _gate(selection_package, decisions, ())

    assert report.release_ready is False
    assert selected.audit_id in report.invalid_applicability_audit_ids
    assert selected.audit_id not in report.terminal_pass_audit_ids
    assert not report.remediations
