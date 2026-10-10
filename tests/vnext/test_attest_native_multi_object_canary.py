from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import vnext_attest_native_multi_object_canary as attestor


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/vnext/native_multi_object_canary_tang3_evidence_20261010_v3"
PREPARATION = ROOT / "results/vnext/native_multi_object_canary_preparation_20261010_v1"
PREPARATION_SHA = "cb50151dceba0256f1962f00e51bfbd86c0f8383fa895d434e0f73f56ba578cd"


def test_attest_v10_binds_direct_observation_without_granting_admission():
    report = attestor.attest(EVIDENCE, PREPARATION, PREPARATION_SHA)
    assert report["status"] == "BOUNDED_CANARY_ATTESTED"
    assert report["native_admission"] == "NOT_GRANTED"
    assert report["scientific_release_allowed"] is False
    assert report["independent_signer"] is False
    assert report["counts"] == {
        "cases": 8, "observed_cases": 8, "mutation_attempts": 50,
        "mutation_completions": 50, "planned_mutation_operations": 50,
        "retrieval_attempts": 8, "retrieval_completions": 8,
        "reset_attempts": 48, "setup_seed_attempts": 16,
    }
    assert report["provider_witnesses"]["query_points"] == 8
    assert report["cleanup"]["all_case_clients_closed"] is True
    assert report["cleanup"]["all_case_collections_deleted"] is True
    assert report["execution"] == {"model_loads": 0, "generations": 0, "gpu_calls": 0, "paid_provider_calls": 0}


def test_attest_rejects_preparation_pin_drift():
    with pytest.raises(ValueError, match="preparation index"):
        attestor.attest(EVIDENCE, PREPARATION, "0" * 64)


def test_attest_rejects_tampered_evidence_without_mutating_source(tmp_path):
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    summary = json.loads((evidence / "summary.json").read_bytes())
    summary["counts"]["mutation_attempts"] = 49
    (evidence / "summary.json").write_bytes(attestor.canonical(summary))
    with pytest.raises(ValueError, match="summary|mutation"):
        attestor.attest(evidence, PREPARATION, PREPARATION_SHA)


def test_publish_attestation_is_no_replace_and_reopenable(tmp_path):
    report = attestor.attest(EVIDENCE, PREPARATION, PREPARATION_SHA)
    output = tmp_path / "attestation"
    index = attestor.publish(report, output)
    assert index["status"] == "BOUNDED_CANARY_ATTESTED"
    assert json.loads((output / "attestation.json").read_bytes()) == report
    with pytest.raises(FileExistsError):
        attestor.publish(report, output)


def test_attest_rejects_evidence_index_hash_tamper(tmp_path):
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    raw = (evidence / "observations.json").read_bytes()
    (evidence / "observations.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="artifact hash"):
        attestor.attest(evidence, PREPARATION, PREPARATION_SHA)
