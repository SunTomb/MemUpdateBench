from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

from mub.vnext.release.task14_review import build_task14_structural_report_v1
from mub.vnext.release.task14_sources import Task14SourcePathsV1, load_task14_sources_v1


REPOSITORY = Path(__file__).resolve().parents[2]
REMOTE_STAGE = "/NAS/yesh/MemUpdateBench/results/vnext/.mub-task13-stage-1a791f4cbfdd471aa6a8bd45ab6432d4"


def loaded_sources():
    return load_task14_sources_v1(
        Task14SourcePathsV1(
            core_root=REPOSITORY / "data/vnext/core/v3",
            evidence_root=REPOSITORY / "results/vnext/core_task14_evidence",
            task13_root=REPOSITORY / "results/vnext/core_task13_bc82566_v1",
            task13_audit_path=REPOSITORY / "results/vnext/core_task13_bc82566_v1_audit.json",
            repository_root=REPOSITORY,
            remote_task13_staging_path=REMOTE_STAGE,
        )
    )


def report_for(loaded, review_id="approval-test"):
    return build_task14_structural_report_v1(
        loaded,
        review_id=review_id,
        trusted_source_revision="test-revision",
        trusted_source_tree_sha256="a" * 64,
    )


def test_core_audit_attestation_tamper_is_not_approved() -> None:
    loaded = loaded_sources()
    payloads = dict(loaded.payloads)
    role = "core/audit/gate_verification_attestation.json"
    value = json.loads(payloads[role])
    value["release_ready_at_verification"] = False
    payloads[role] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    forged = replace(loaded, payloads=MappingProxyType(payloads))
    report = report_for(forged, "core-tamper")
    assert report.status == "NOT_APPROVED"
    assert any(item.check_id == "core_human_audit_verified" and not item.passed for item in report.checks)


def test_task11_unqualified_model_is_not_approved() -> None:
    loaded = loaded_sources()
    payloads = dict(loaded.json_payloads)
    qualification = dict(payloads["task11/qualification_report.json"])
    qualification["status"] = "blocked"
    payloads["task11/qualification_report.json"] = qualification
    forged = replace(loaded, json_payloads=MappingProxyType(payloads))
    report = report_for(forged, "task11-tamper")
    assert report.status == "NOT_APPROVED"
    assert any(item.check_id == "task11_models_qualified" and not item.passed for item in report.checks)


def test_structural_report_never_persists_final_approved_status() -> None:
    report = report_for(loaded_sources())
    payload = report.model_dump(mode="json")
    assert payload["status"] == "READY_FOR_VERIFICATION"
    assert payload["status"] != "FINAL_APPROVED"
