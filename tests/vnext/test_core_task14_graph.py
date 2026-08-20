from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from mub.vnext.release.task14_review import (
    TASK14_REQUIRED_EXCLUSIONS,
    build_task14_evidence_graph_v1,
    build_task14_structural_report_v1,
)
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


def test_real_evidence_graph_is_closed_and_scope_safe() -> None:
    loaded = loaded_sources()
    graph = build_task14_evidence_graph_v1(loaded)
    assert len(graph.nodes) == 22
    assert len(graph.edges) == 22
    by_kind = {item.evidence_kind: item for item in graph.nodes}
    assert by_kind["mem0_admission"].accuracy_evidence is False
    assert by_kind["prompted_answer_matrix"].accuracy_evidence is True
    assert by_kind["clustered_statistics"].accuracy_evidence is True


def test_real_structural_report_is_ready_with_explicit_exclusions() -> None:
    report = build_task14_structural_report_v1(
        loaded_sources(),
        review_id="core-task14-test",
        trusted_source_revision="test-revision",
        trusted_source_tree_sha256="a" * 64,
    )
    assert report.status == "READY_FOR_VERIFICATION"
    assert all(item.passed for item in report.checks)
    assert not report.findings
    assert report.exclusions == TASK14_REQUIRED_EXCLUSIONS
    assert {item.evidence_kind for item in report.exclusions} == {
        "pilot_deterministic",
        "fake_offline",
        "slot_direct",
        "mem0_admission",
        "api_probe",
        "remote_nfs_staging",
    }


def test_mem0_admission_cannot_be_rebound_as_accuracy() -> None:
    graph = build_task14_evidence_graph_v1(loaded_sources())
    mem0 = next(item for item in graph.nodes if item.evidence_kind == "mem0_admission")
    from pydantic import ValidationError
    import pytest

    forged = mem0.validated_replace(accuracy_evidence=True)
    nodes = tuple(forged if item.node_id == mem0.node_id else item for item in graph.nodes)
    with pytest.raises(ValidationError, match="accuracy"):
        type(graph)(nodes=nodes, edges=graph.edges)


def test_task13_unsupported_policy_tamper_derives_not_approved() -> None:
    loaded = loaded_sources()
    payloads = dict(loaded.json_payloads)
    audit = dict(payloads["task13_audit/core_task13_bc82566_v1_audit.json"])
    metric_status = dict(audit["metric_status"])
    metric_status["unsupported"] = {}
    audit["metric_status"] = metric_status
    payloads["task13_audit/core_task13_bc82566_v1_audit.json"] = audit
    forged = replace(loaded, json_payloads=MappingProxyType(payloads))
    report = build_task14_structural_report_v1(
        forged,
        review_id="tampered",
        trusted_source_revision="test-revision",
        trusted_source_tree_sha256="a" * 64,
    )
    assert report.status == "NOT_APPROVED"
    assert "task13_unsupported_policy" in {
        item.check_id for item in report.checks if not item.passed
    }


def test_task12_incomplete_matrix_derives_not_approved() -> None:
    loaded = loaded_sources()
    payloads = dict(loaded.json_payloads)
    audit = dict(payloads["task12/matrix_integrity_audit.json"])
    audit["run_count"] = 17
    payloads["task12/matrix_integrity_audit.json"] = audit
    forged = replace(loaded, json_payloads=MappingProxyType(payloads))
    report = build_task14_structural_report_v1(
        forged,
        review_id="tampered",
        trusted_source_revision="test-revision",
        trusted_source_tree_sha256="a" * 64,
    )
    assert report.status == "NOT_APPROVED"
    assert any(item.finding_id == "failed:task12_real_matrix_complete" for item in report.findings)
