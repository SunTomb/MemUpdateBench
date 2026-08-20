from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

from mub.vnext.audit.core_candidate import core_candidate_root_digest
from mub.vnext.audit.core_review import (
    CoreAuditGateReport,
    CoreAuditVerificationAttestation,
    verify_core_audit_verification_attestation,
)
from mub.vnext.release.task14_contracts import (
    Task14CheckV1,
    Task14EvidenceEdgeV1,
    Task14EvidenceGraphV1,
    Task14EvidenceNodeV1,
    Task14ExclusionV1,
    Task14FindingV1,
    Task14StructuralReportV1,
)
from mub.vnext.release.task14_sources import (
    TASK14_TASK9_IMPLEMENTATION_REVISION,
    Task14LoadedSourcesV1,
)
from mub.vnext.statistics.task13_v3 import verify_task13_artifact_root_v3


TASK14_REQUIRED_EXCLUSIONS = (
    Task14ExclusionV1(
        exclusion_id="exclude_pilot_deterministic_accuracy",
        evidence_kind="pilot_deterministic",
        reason="Pilot slot_direct outputs are bounded engineering evidence, not prompted-answer accuracy.",
    ),
    Task14ExclusionV1(
        exclusion_id="exclude_task12_fake_offline",
        evidence_kind="fake_offline",
        reason="Fake-offline Task 12 runs are engineering regression evidence only.",
    ),
    Task14ExclusionV1(
        exclusion_id="exclude_slot_direct_science",
        evidence_kind="slot_direct",
        reason="Deterministic slot_direct checkpoints are not prompted-answer scientific evidence.",
    ),
    Task14ExclusionV1(
        exclusion_id="exclude_mem0_accuracy",
        evidence_kind="mem0_admission",
        reason="Mem0 admission and NOT_SUPPORTED canaries are capability evidence, not accuracy.",
    ),
    Task14ExclusionV1(
        exclusion_id="exclude_api_probe_external_memory",
        evidence_kind="api_probe",
        reason="P8.4 API answer probes are not external-memory-system evidence.",
    ),
    Task14ExclusionV1(
        exclusion_id="exclude_nfs_staging_final",
        evidence_kind="remote_nfs_staging",
        reason="The verified remote Task 13 staging root is not a published final root.",
    ),
)
TASK14_REQUIRED_NODE_IDS = frozenset(
    {
        "core_release", "core_candidate", "core_candidate_root", "core_audit",
        "task9_implementation", "task9_provenance", "task10_report", "task10_decision",
        "task11_qualification", "task11_provenance", "task12_manifest", "task12_summary",
        "task12_audit", "task13_index", "task13_receipt", "task13_statistics",
        "task13_contrasts", "task13_case_records", "task13_cases", "task13_claims",
        "task13_audit", "task13_nfs_staging",
    }
)


def _node(
    loaded: Task14LoadedSourcesV1,
    *,
    node_id: str,
    role: str,
    evidence_kind: str,
    accuracy: bool = False,
    scope: str = "bounded_core",
) -> Task14EvidenceNodeV1:
    return Task14EvidenceNodeV1(
        node_id=node_id,
        evidence_kind=evidence_kind,
        artifact=loaded.artifacts[role],
        accuracy_evidence=accuracy,
        scope=scope,
    )


def _edge(
    nodes: Mapping[str, Task14EvidenceNodeV1],
    source: str,
    target: str,
    edge_type: str,
) -> Task14EvidenceEdgeV1:
    return Task14EvidenceEdgeV1(
        source_node_id=source,
        target_node_id=target,
        edge_type=edge_type,
        source_sha256=nodes[source].artifact.sha256,
        target_sha256=nodes[target].artifact.sha256,
    )


def build_task14_evidence_graph_v1(
    loaded: Task14LoadedSourcesV1,
) -> Task14EvidenceGraphV1:
    values = (
        _node(loaded, node_id="core_release", role="core/task_release_manifest.json", evidence_kind="core_task_release"),
        _node(loaded, node_id="core_candidate", role="core/audit/gate_report.json", evidence_kind="core_candidate_receipt"),
        _node(loaded, node_id="core_candidate_root", role="core/task_release_manifest.json", evidence_kind="core_candidate_root_digest"),
        _node(loaded, node_id="core_audit", role="core/audit/gate_verification_attestation.json", evidence_kind="core_human_audit"),
        _node(loaded, node_id="task9_implementation", role="task9/model_provenance.json", evidence_kind="task9_implementation", scope=f"revision:{TASK14_TASK9_IMPLEMENTATION_REVISION}"),
        _node(loaded, node_id="task9_provenance", role="task9/model_provenance.json", evidence_kind="task9_engineering"),
        _node(loaded, node_id="task10_report", role="task10/external_admission_report.json", evidence_kind="mem0_admission"),
        _node(loaded, node_id="task10_decision", role="task10/admission_decision.json", evidence_kind="mem0_admission"),
        _node(loaded, node_id="task11_qualification", role="task11/qualification_report.json", evidence_kind="answer_model_qualification"),
        _node(loaded, node_id="task11_provenance", role="task11/mistral_snapshot_provenance.json", evidence_kind="answer_model_provenance"),
        _node(loaded, node_id="task12_manifest", role="task12/matrix_bundle_manifest.json", evidence_kind="prompted_answer_matrix", accuracy=True),
        _node(loaded, node_id="task12_summary", role="task12/matrix_run_summary.json", evidence_kind="prompted_answer_matrix", accuracy=True),
        _node(loaded, node_id="task12_audit", role="task12/matrix_integrity_audit.json", evidence_kind="prompted_answer_matrix", accuracy=True),
        _node(loaded, node_id="task13_index", role="task13/task13_artifact_index.json", evidence_kind="clustered_statistics", accuracy=True),
        _node(loaded, node_id="task13_receipt", role="task13/statistics_receipt.json", evidence_kind="clustered_statistics", accuracy=True),
        _node(loaded, node_id="task13_statistics", role="task13/cell_statistics.jsonl", evidence_kind="clustered_statistics", accuracy=True),
        _node(loaded, node_id="task13_contrasts", role="task13/paired_contrasts.jsonl", evidence_kind="paired_contrasts", accuracy=True),
        _node(loaded, node_id="task13_case_records", role="task13/cases.jsonl", evidence_kind="verified_cases", accuracy=True),
        _node(loaded, node_id="task13_cases", role="task13/case_index.json", evidence_kind="verified_cases", accuracy=True),
        _node(loaded, node_id="task13_claims", role="task13/claim_ledger.jsonl", evidence_kind="claim_ledger", accuracy=True),
        _node(loaded, node_id="task13_audit", role="task13_audit/core_task13_bc82566_v1_audit.json", evidence_kind="task13_independent_audit", accuracy=True),
        Task14EvidenceNodeV1(
            node_id="task13_nfs_staging",
            evidence_kind="remote_nfs_staging",
            artifact=loaded.artifacts["task13_audit/core_task13_bc82566_v1_audit.json"].validated_replace(
                root_kind="remote_nfs_staging",
                source_location=loaded.paths.remote_task13_staging_path,
            ),
            accuracy_evidence=False,
            scope="verified_staging_evidence_not_published_final",
        ),
    )
    nodes = {item.node_id: item for item in values}
    edges = (
        _edge(nodes, "core_candidate", "core_release", "depends_on"),
        _edge(nodes, "core_candidate_root", "core_candidate", "authenticates"),
        _edge(nodes, "core_audit", "core_candidate", "authenticates"),
        _edge(nodes, "task9_implementation", "core_release", "depends_on"),
        _edge(nodes, "task9_provenance", "task9_implementation", "authenticates"),
        _edge(nodes, "task10_report", "task9_provenance", "depends_on"),
        _edge(nodes, "task10_decision", "task10_report", "authenticates"),
        _edge(nodes, "task11_provenance", "task11_qualification", "authenticates"),
        _edge(nodes, "task12_manifest", "core_release", "depends_on"),
        _edge(nodes, "task12_manifest", "task10_decision", "depends_on"),
        _edge(nodes, "task12_manifest", "task11_provenance", "depends_on"),
        _edge(nodes, "task12_summary", "task12_manifest", "derived_from"),
        _edge(nodes, "task12_audit", "task12_summary", "authenticates"),
        _edge(nodes, "task13_index", "task12_audit", "derived_from"),
        _edge(nodes, "task13_receipt", "task13_index", "authenticates"),
        _edge(nodes, "task13_statistics", "task13_receipt", "derived_from"),
        _edge(nodes, "task13_contrasts", "task13_receipt", "derived_from"),
        _edge(nodes, "task13_case_records", "task13_receipt", "derived_from"),
        _edge(nodes, "task13_cases", "task13_case_records", "authenticates"),
        _edge(nodes, "task13_claims", "task13_receipt", "derived_from"),
        _edge(nodes, "task13_audit", "task13_index", "authenticates"),
        _edge(nodes, "task13_nfs_staging", "task13_index", "excludes"),
    )
    return Task14EvidenceGraphV1(nodes=values, edges=edges)


def _core_audit_ready(loaded: Task14LoadedSourcesV1) -> bool:
    try:
        report = CoreAuditGateReport.model_validate_json(
            loaded.payloads["core/audit/gate_report.json"]
        )
        attestation = CoreAuditVerificationAttestation.model_validate_json(
            loaded.payloads["core/audit/gate_verification_attestation.json"]
        )
        if not verify_core_audit_verification_attestation(attestation, report=report):
            return False
        receipt = report.candidate_validation_receipt
        if receipt is None or not attestation.release_ready_at_verification:
            return False
        core_snapshot = next(
            item for item in loaded.root_snapshots if item.root_id == "immutable_core"
        )
        current = {item.relative_path: item.sha256 for item in core_snapshot.entries}
        if any(
            current.get(f"candidate/{artifact.path}") != artifact.sha256
            for artifact in receipt.candidate_artifacts
        ):
            return False
        return bool(
            receipt.expected_full
            and receipt.task_count == 12000
            and core_candidate_root_digest(receipt.candidate_artifacts)
            == receipt.candidate_root_digest
        )
    except Exception:
        return False


def _task13_closure_ready(loaded: Task14LoadedSourcesV1) -> bool:
    try:
        result = verify_task13_artifact_root_v3(loaded.paths.task13_root)
        return result.artifact_index_sha256 == loaded.artifacts[
            "task13/task13_artifact_index.json"
        ].sha256
    except Exception:
        return False


def _task13_policy_ready(loaded: Task14LoadedSourcesV1) -> bool:
    audit = loaded.json_payloads["task13_audit/core_task13_bc82566_v1_audit.json"]
    if not isinstance(audit, dict):
        return False
    unsupported = audit.get("metric_status", {}).get("unsupported", {})
    numeric = audit.get("metric_status", {}).get("numeric", {})
    return bool(
        set(unsupported)
        == {
            "answer_scores.gold_retrieved_wrong_answer",
            "retrieval_scores.stale_count_in_context",
            "retrieval_scores.stale_exposure_rate",
        }
        and all(value == 18 for value in unsupported.values())
        and set(unsupported).isdisjoint(numeric)
    )


def _check(check_id: str, passed: bool, detail: str, *nodes: str) -> Task14CheckV1:
    return Task14CheckV1(
        check_id=check_id,
        passed=passed,
        detail=detail,
        evidence_node_ids=tuple(nodes),
    )


def build_task14_structural_report_v1(
    loaded: Task14LoadedSourcesV1,
    *,
    review_id: str,
    trusted_source_revision: str,
    trusted_source_tree_sha256: str,
) -> Task14StructuralReportV1:
    if type(loaded) is not Task14LoadedSourcesV1:
        raise TypeError("Task 14 structural review requires loaded sources")
    graph = build_task14_evidence_graph_v1(loaded)
    release = loaded.json_payloads["core/task_release_manifest.json"]
    task10_decision = loaded.json_payloads["task10/admission_decision.json"]
    task10_report = loaded.json_payloads["task10/external_admission_report.json"]
    task10_rows = loaded.json_payloads["task10/canary_terminal_rows.json"]
    task11 = loaded.json_payloads["task11/qualification_report.json"]
    task12 = loaded.json_payloads["task12/matrix_integrity_audit.json"]
    task13_audit = loaded.json_payloads["task13_audit/core_task13_bc82566_v1_audit.json"]

    checks = (
        _check("core_release_final", isinstance(release, dict) and release.get("release_status") == "FINAL_APPROVED", "Immutable Core task release is FINAL_APPROVED.", "core_release"),
        _check("core_human_audit_verified", _core_audit_ready(loaded), "Core human-audit report and current candidate root reverify.", "core_audit", "core_release"),
        _check("task9_implementation_bound", any(item.node_id == "task9_implementation" and item.scope == f"revision:{TASK14_TASK9_IMPLEMENTATION_REVISION}" for item in graph.nodes), "Task 9 implementation revision is an ancestor of the trusted Task 14 source.", "task9_implementation"),
        _check("task9_engineering_bound", loaded.artifacts["task9/model_provenance.json"].sha256 == "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e", "Task 9 provenance is exact and engineering-only.", "task9_provenance"),
        _check("task10_mem0_admitted", isinstance(task10_decision, dict) and task10_decision.get("status") == "admitted" and task10_decision.get("reasons") == ["admitted_mem0_primary"] and isinstance(task10_report, dict) and len(task10_report.get("gates", ())) == 14 and all(item.get("status") == "pass" for item in task10_report.get("gates", ())), "Mem0 is admitted with 14/14 PASS as capability evidence only; fallback is not authorized.", "task10_report", "task10_decision"),
        _check("task10_canaries_explicit_unsupported", isinstance(task10_rows, dict) and len(task10_rows.get("terminal_rows", ())) == 128 and all(item.get("completion_status") == "not_supported" for item in task10_rows.get("terminal_rows", ())), "All 128 Mem0 canary rows are explicit NOT_SUPPORTED, not zero scores.", "task10_report"),
        _check("task11_models_qualified", isinstance(task11, dict) and task11.get("status") in {"qualified", "verified", "pass"}, "Task 11 model provenance and qualification are bound.", "task11_qualification", "task11_provenance"),
        _check("task12_real_matrix_complete", isinstance(task12, dict) and task12.get("status") == "verified" and task12.get("run_count") == 18 and task12.get("total_task_rows") == 1440 and task12.get("total_score_rows") == 1440 and task12.get("failed_or_partial_rows") == 0, "Task 12 real prompted-answer matrix is complete.", "task12_manifest", "task12_summary", "task12_audit"),
        _check("task13_artifacts_closed", _task13_closure_ready(loaded), "Task 13 exact eight-artifact final root reopens and verifies.", "task13_index", "task13_receipt", "task13_cases", "task13_claims"),
        _check("task13_unsupported_policy", _task13_policy_ready(loaded), "Task 13 unsupported/null metrics remain typed and nonnumeric.", "task13_audit"),
        _check("task13_matrix_case_rejoin", isinstance(task13_audit, dict) and task13_audit.get("matrix_case_rejoin") == {"status": "verified", "runs": 18, "cases": 57, "observations": 1440}, "Task 13 cases rejoin 18 Task 12 runs and 1,440 observations.", "task13_case_records", "task13_cases", "task13_audit"),
        _check("task13_local_remote_boundary", isinstance(task13_audit, dict) and task13_audit.get("remote_final_root_absent") is True and task13_audit.get("local_output_root") == "results/vnext/core_task13_bc82566_v1" and ".mub-task13-stage-" in loaded.paths.remote_task13_staging_path, "Local published final and remote NFS staging evidence remain distinct.", "task13_audit", "task13_nfs_staging"),
        _check("evidence_graph_closed", {item.node_id for item in graph.nodes} == TASK14_REQUIRED_NODE_IDS and len(graph.edges) == 22, "All required upstream evidence nodes and edges are closed; Task 14 output bindings are carried by manifest/index to avoid self-hash cycles.", *(item.node_id for item in graph.nodes)),
    )
    findings: tuple[Task14FindingV1, ...] = ()
    if not all(item.passed for item in checks):
        findings = tuple(
            Task14FindingV1(
                finding_id=f"failed:{item.check_id}",
                severity="critical",
                detail=item.detail,
                evidence_node_ids=item.evidence_node_ids,
            )
            for item in checks
            if not item.passed
        )
    return Task14StructuralReportV1(
        report_id=f"{review_id}:structural",
        review_id=review_id,
        trusted_source_revision=trusted_source_revision,
        trusted_source_tree_sha256=trusted_source_tree_sha256,
        graph=graph,
        checks=checks,
        findings=findings,
        exclusions=TASK14_REQUIRED_EXCLUSIONS,
    )


__all__ = [
    "TASK14_REQUIRED_EXCLUSIONS",
    "build_task14_evidence_graph_v1",
    "build_task14_structural_report_v1",
]
