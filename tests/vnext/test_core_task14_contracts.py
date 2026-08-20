from __future__ import annotations

import pytest
from pydantic import ValidationError

from mub.vnext.release.task14_contracts import (
    TASK14_ARTIFACT_PATHS,
    TASK14_ALLOWED_EDGE_TYPES,
    Task14ArtifactRefV1,
    Task14AttestationV1,
    Task14CheckV1,
    Task14EvidenceEdgeV1,
    Task14EvidenceGraphV1,
    Task14EvidenceNodeV1,
    Task14RootIndexV1,
    Task14RootManifestV1,
    Task14RootSnapshotV1,
    Task14StructuralReportV1,
    VerifiedCoreFinalRelease,
    task14_attestation_file_hash_v1,
    task14_attestation_hash_v1,
    task14_graph_hash_v1,
    task14_index_hash_v1,
    task14_manifest_hash_v1,
    task14_report_hash_v1,
)


H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64
H4 = "d" * 64


def ref(path: str, digest: str = H) -> Task14ArtifactRefV1:
    return Task14ArtifactRefV1(
        artifact_id=path.removesuffix(".json"),
        path=path,
        sha256=digest,
        media_type="application/json",
    )


def node(node_id: str, kind: str = "task13", digest: str = H) -> Task14EvidenceNodeV1:
    return Task14EvidenceNodeV1(
        node_id=node_id,
        evidence_kind=kind,
        artifact=ref(f"{node_id}.json", digest),
        accuracy_evidence=False,
    )


def graph(*nodes: Task14EvidenceNodeV1) -> Task14EvidenceGraphV1:
    return Task14EvidenceGraphV1(nodes=nodes, edges=())


def check(check_id: str = "root_matches", passed: bool = True) -> Task14CheckV1:
    return Task14CheckV1(check_id=check_id, passed=passed, detail="checked")


def report(*, passed: bool = True) -> Task14StructuralReportV1:
    g = graph(node("core"))
    return Task14StructuralReportV1(
        report_id="review-1",
        review_id="review-1",
        trusted_source_revision="e" * 40,
        trusted_source_tree_sha256=H,
        graph=g,
        checks=(check(passed=passed),),
        findings=(),
        exclusions=(),
    )


def attestation_for(value: Task14StructuralReportV1) -> Task14AttestationV1:
    payload = {
        "report_sha256": task14_report_hash_v1(value),
        "graph_sha256": task14_graph_hash_v1(value.graph),
        "trusted_source_revision": value.trusted_source_revision,
        "trusted_source_tree_sha256": value.trusted_source_tree_sha256,
        "source_snapshot_sha256": H2,
        "final_approval_at_verification": value.status == "READY_FOR_VERIFICATION",
    }
    return Task14AttestationV1(
        **payload,
        attestation_sha256=task14_attestation_hash_v1(payload),
    )


def manifest_for(
    value: Task14StructuralReportV1,
    attestation: Task14AttestationV1,
) -> Task14RootManifestV1:
    return Task14RootManifestV1(
        artifacts=(
            ref(TASK14_ARTIFACT_PATHS[0], task14_report_hash_v1(value)),
            ref(TASK14_ARTIFACT_PATHS[1], task14_graph_hash_v1(value.graph)),
            ref(TASK14_ARTIFACT_PATHS[2], task14_attestation_file_hash_v1(attestation)),
        ),
    )


def index_for(
    value: Task14StructuralReportV1,
    attestation: Task14AttestationV1,
) -> Task14RootIndexV1:
    manifest = manifest_for(value, attestation)
    return Task14RootIndexV1(
        artifacts=manifest.artifacts
        + (ref(TASK14_ARTIFACT_PATHS[3], task14_manifest_hash_v1(manifest)),),
    )


def test_hashes_are_strict_lowercase_and_contracts_are_frozen() -> None:
    with pytest.raises(ValidationError):
        ref("bad.json", "A" * 64)
    with pytest.raises(ValidationError):
        Task14ArtifactRefV1(
            artifact_id="x",
            path="x.json",
            sha256=1,
            media_type="application/json",
        )
    artifact = ref("x.json")
    with pytest.raises((ValidationError, TypeError)):
        artifact.path = "other.json"


def test_root_snapshot_requires_canonical_unique_entries() -> None:
    entry = {"relative_path": "a.json", "byte_count": 1, "sha256": H}
    snapshot = Task14RootSnapshotV1(
        root_id="core",
        root_path="data/vnext/core/v3",
        filesystem_identity="volume:1",
        entries=(entry,),
        tree_sha256=H2,
    )
    assert snapshot.entries[0].relative_path == "a.json"
    with pytest.raises(ValidationError):
        Task14RootSnapshotV1(
            root_id="core",
            root_path="data/vnext/core/v3",
            filesystem_identity="volume:1",
            entries=(entry, entry),
            tree_sha256=H2,
        )


def test_graph_rejects_duplicate_foreign_invalid_edges_and_forbidden_accuracy() -> None:
    left = node("left")
    right = node("right", digest=H2)
    edge = Task14EvidenceEdgeV1(
        source_node_id="left",
        target_node_id="right",
        edge_type="depends_on",
        source_sha256=H,
        target_sha256=H2,
    )
    valid = Task14EvidenceGraphV1(nodes=(left, right), edges=(edge,))
    assert valid.edges[0].edge_type in TASK14_ALLOWED_EDGE_TYPES

    with pytest.raises(ValidationError, match="duplicate"):
        Task14EvidenceGraphV1(nodes=(left, left), edges=())
    with pytest.raises(ValidationError, match="foreign"):
        Task14EvidenceGraphV1(
            nodes=(left, right),
            edges=(
                Task14EvidenceEdgeV1(
                    source_node_id="left",
                    target_node_id="missing",
                    edge_type="depends_on",
                    source_sha256=H,
                    target_sha256=H2,
                ),
            ),
        )
    with pytest.raises(ValidationError):
        Task14EvidenceGraphV1(
            nodes=(left, right),
            edges=(
                Task14EvidenceEdgeV1(
                    source_node_id="left",
                    target_node_id="right",
                    edge_type="not_allowed",
                    source_sha256=H,
                    target_sha256=H2,
                ),
            ),
        )

    with pytest.raises(ValidationError, match="acyclic"):
        Task14EvidenceGraphV1(
            nodes=(left, right),
            edges=(
                edge,
                Task14EvidenceEdgeV1(
                    source_node_id="right",
                    target_node_id="left",
                    edge_type="depends_on",
                    source_sha256=H2,
                    target_sha256=H,
                ),
            ),
        )

    forbidden = Task14EvidenceNodeV1(
        node_id="mem0",
        evidence_kind="mem0_admission",
        artifact=ref("mem0.json", H3),
        accuracy_evidence=True,
    )
    with pytest.raises(ValidationError, match="accuracy"):
        Task14EvidenceGraphV1(nodes=(forbidden,), edges=())

    fake = Task14EvidenceNodeV1(
        node_id="fake",
        evidence_kind="fake_offline",
        artifact=ref("fake.json", H3),
        accuracy_evidence=True,
    )
    direct = Task14EvidenceNodeV1(
        node_id="direct",
        evidence_kind="slot_direct",
        artifact=ref("direct.json", H4),
        accuracy_evidence=True,
    )
    for bad in (fake, direct):
        with pytest.raises(ValidationError, match="accuracy"):
            Task14EvidenceGraphV1(nodes=(bad,), edges=())


def test_report_status_is_derived_and_cannot_be_caller_falsified() -> None:
    ready = report(passed=True)
    blocked = report(passed=False)
    assert ready.status == "READY_FOR_VERIFICATION"
    assert blocked.status == "NOT_APPROVED"
    with pytest.raises(ValidationError):
        Task14StructuralReportV1(
            report_id="review-1",
            review_id="review-1",
            trusted_source_revision="e" * 40,
            graph=ready.graph,
            checks=(check(),),
            findings=(),
            exclusions=(),
            status="READY_FOR_VERIFICATION",
        )


def test_attestation_hash_self_verifies_and_tampering_fails() -> None:
    value = report()
    attestation = attestation_for(value)
    assert attestation.attestation_sha256 == task14_attestation_hash_v1(attestation)
    payload = attestation.model_dump(mode="python")
    payload["attestation_sha256"] = H4
    with pytest.raises(ValidationError, match="attestation_sha256"):
        Task14AttestationV1(**payload)


def test_manifest_and_index_have_exact_acyclic_order() -> None:
    value = report()
    attestation = attestation_for(value)
    manifest = manifest_for(value, attestation)
    assert tuple(item.path for item in manifest.artifacts) == TASK14_ARTIFACT_PATHS[:3]
    index = index_for(value, attestation)
    assert tuple(item.path for item in index.artifacts) == TASK14_ARTIFACT_PATHS[:4]
    changed_ref = index.artifacts[3].validated_replace(sha256=H4)
    changed_index = index.validated_replace(
        artifacts=(*index.artifacts[:3], changed_ref)
    )
    assert task14_index_hash_v1(index) != task14_index_hash_v1(changed_index)

    with pytest.raises(ValidationError):
        Task14RootManifestV1(artifacts=manifest.artifacts[:2])
    with pytest.raises(ValidationError):
        Task14RootIndexV1(artifacts=index.artifacts + (ref("core_final_root_index.json", H4),))
    with pytest.raises(ValidationError):
        Task14RootIndexV1(artifacts=(index.artifacts[1], *index.artifacts[1:]))


def test_verified_wrapper_cannot_be_constructed_or_minted_without_current_sources() -> None:
    value = report()
    attestation = attestation_for(value)
    manifest = manifest_for(value, attestation)
    index = index_for(value, attestation)
    with pytest.raises(TypeError, match="current-source"):
        VerifiedCoreFinalRelease(value, attestation, manifest, index)
