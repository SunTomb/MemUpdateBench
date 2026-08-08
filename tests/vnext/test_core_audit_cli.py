from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

import mub.vnext.audit.core_stage as core_stage
import scripts.vnext_gate_core_audit as gate_cli


ROOT = Path(__file__).resolve().parents[2]


def test_core_audit_clis_run_from_project_root_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for script in (
        "vnext_prepare_core_audit.py",
        "vnext_gate_core_audit.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
    prepare_help = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vnext_prepare_core_audit.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    assert "--candidate-dir" in prepare_help
    gate_help = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vnext_gate_core_audit.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    assert "--allow-bounded" not in gate_help


def test_missing_required_decisions_fail_but_optional_adjudications_may_be_absent(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        core_stage._read_review_rows(missing, required=True)
    assert core_stage._read_review_rows(missing, required=False) == ()


def test_gate_cli_payload_emits_verification_binding_hashes() -> None:
    report = SimpleNamespace(
        terminal_pass_audit_ids=("audit",),
        required_adjudication_ids=(),
        unresolved_adjudication_ids=(),
        remediations=(),
        raw_agreement=1.0,
        cohens_kappa=None,
        issues=("candidate_verification_required:explicit_current_root_verification",),
    )
    attestation = SimpleNamespace(
        attestation_hash="a" * 64,
        structural_report_hash="b" * 64,
        candidate_receipt_hash="c" * 64,
        candidate_root_digest="d" * 64,
        audit_evidence_hash="e" * 64,
        source_task_manifest_hash="f" * 64,
        selection_hash="0" * 64,
        review_context_hash="9" * 64,
        candidate_generation_revision="1" * 40,
        trusted_audit_tooling_revision="1" * 40,
        candidate_scope_at_verification="full",
        full_candidate_at_verification=True,
        release_ready_at_verification=True,
    )
    verified = SimpleNamespace(report=report, attestation=attestation)

    payload = gate_cli._gate_status_payload(verified)

    assert payload["release_ready_at_verification"] is True
    assert payload["candidate_scope_at_verification"] == "full"
    assert payload["attestation_hash"] == "a" * 64
    assert payload["structural_report_hash"] == "b" * 64
    assert payload["candidate_receipt_hash"] == "c" * 64
    assert payload["candidate_root_digest"] == "d" * 64
    assert payload["audit_evidence_hash"] == "e" * 64
    assert payload["source_task_manifest_hash"] == "f" * 64
    assert payload["selection_hash"] == "0" * 64
    assert payload["review_context_hash"] == "9" * 64
    assert payload["candidate_generation_revision"] == "1" * 40
    assert payload["trusted_audit_tooling_revision"] == "1" * 40


def test_review_jsonl_loader_rejects_noncanonical_and_duplicate_key_rows(
    tmp_path,
) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_bytes(b'{ "audit_id":"audit" }\n')
    with pytest.raises(ValueError, match="canonical"):
        core_stage._read_review_rows(path)
    path.write_bytes(b'{"audit_id":"audit","audit_id":"audit"}\n')
    with pytest.raises(ValueError, match="canonical"):
        core_stage._read_review_rows(path)
