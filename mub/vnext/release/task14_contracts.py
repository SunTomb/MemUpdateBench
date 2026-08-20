from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, computed_field, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.io import canonical_json_bytes, sha256_model


SHA256_PATTERN = r"^[0-9a-f]{64}$"
TASK14_ARTIFACT_PATHS = (
    "core_final_review_report.json",
    "core_final_evidence_graph.json",
    "core_final_verification_attestation.json",
    "core_final_root_manifest.json",
    "core_final_root_index.json",
)
TASK14_ALLOWED_EDGE_TYPES = (
    "depends_on",
    "authenticates",
    "derived_from",
    "qualifies",
    "excludes",
)
_FORBIDDEN_ACCURACY_KINDS = {
    "mem0_admission",
    "fake_offline",
    "slot_direct",
    "pilot_deterministic",
    "api_probe",
}
_VERIFIED_RELEASE_TOKEN = object()


class Task14ArtifactRefV1(ImmutableContractModel):
    artifact_id: StrictStr = Field(min_length=1)
    path: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    media_type: StrictStr = Field(min_length=1)
    byte_count: StrictInt | None = Field(default=None, ge=0)
    record_count: StrictInt | None = Field(default=None, ge=0)
    root_kind: Literal[
        "immutable_local",
        "published_local",
        "remote_evidence",
        "remote_nfs_staging",
        "task14_output",
    ] = "task14_output"
    source_location: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def _relative_safe_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or normalized in {".", ".."} or ".." in normalized.split("/"):
            raise ValueError("artifact path must be a safe relative path")
        return normalized


class Task14RootEntryV1(ImmutableContractModel):
    relative_path: StrictStr = Field(min_length=1)
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or normalized in {".", ".."} or ".." in normalized.split("/"):
            raise ValueError("root entry path must be safe and relative")
        return normalized


class Task14RootSnapshotV1(ImmutableContractModel):
    root_id: StrictStr = Field(min_length=1)
    root_path: StrictStr = Field(min_length=1)
    filesystem_identity: StrictStr = Field(min_length=1)
    entries: tuple[Task14RootEntryV1, ...]
    tree_sha256: StrictStr = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _canonical_entries(self) -> "Task14RootSnapshotV1":
        paths = tuple(item.relative_path for item in self.entries)
        if len(set(paths)) != len(paths):
            raise ValueError("root snapshot contains duplicate entries")
        if paths != tuple(sorted(paths, key=lambda value: value.encode("utf-8"))):
            raise ValueError("root snapshot entries are not canonical")
        return self


class Task14EvidenceNodeV1(ImmutableContractModel):
    node_id: StrictStr = Field(min_length=1)
    evidence_kind: StrictStr = Field(min_length=1)
    artifact: Task14ArtifactRefV1
    accuracy_evidence: StrictBool = False
    scope: StrictStr = "bounded_core"


class Task14EvidenceEdgeV1(ImmutableContractModel):
    source_node_id: StrictStr = Field(min_length=1)
    target_node_id: StrictStr = Field(min_length=1)
    edge_type: Literal[
        "depends_on", "authenticates", "derived_from", "qualifies", "excludes"
    ]
    source_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    target_sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class Task14EvidenceGraphV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task14-evidence-graph.v1"] = (
        "memupdatebench.core-task14-evidence-graph.v1"
    )
    nodes: tuple[Task14EvidenceNodeV1, ...]
    edges: tuple[Task14EvidenceEdgeV1, ...]

    @model_validator(mode="after")
    def _closed_graph(self) -> "Task14EvidenceGraphV1":
        by_id: dict[str, Task14EvidenceNodeV1] = {}
        for item in self.nodes:
            if item.node_id in by_id:
                raise ValueError("evidence graph contains duplicate node IDs")
            if item.accuracy_evidence and item.evidence_kind in _FORBIDDEN_ACCURACY_KINDS:
                raise ValueError(f"{item.evidence_kind} cannot be accuracy evidence")
            by_id[item.node_id] = item
        edge_keys: set[tuple[str, str, str]] = set()
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in by_id}
        for edge in self.edges:
            if edge.source_node_id not in by_id or edge.target_node_id not in by_id:
                raise ValueError("evidence graph edge references a foreign node")
            if edge.source_node_id == edge.target_node_id:
                raise ValueError("evidence graph cannot contain self-cycles")
            adjacency[edge.source_node_id].add(edge.target_node_id)
            key = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            if key in edge_keys:
                raise ValueError("evidence graph contains duplicate edges")
            edge_keys.add(key)
            if edge.source_sha256 != by_id[edge.source_node_id].artifact.sha256:
                raise ValueError("evidence edge source hash mismatch")
            if edge.target_sha256 != by_id[edge.target_node_id].artifact.sha256:
                raise ValueError("evidence edge target hash mismatch")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("evidence graph must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in adjacency[node_id]:
                visit(target)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in adjacency:
            visit(node_id)
        return self


class Task14CheckV1(ImmutableContractModel):
    check_id: StrictStr = Field(min_length=1)
    passed: StrictBool
    detail: StrictStr = Field(min_length=1)
    evidence_node_ids: tuple[StrictStr, ...] = ()


class Task14FindingV1(ImmutableContractModel):
    finding_id: StrictStr = Field(min_length=1)
    severity: Literal["critical", "important", "minor"]
    detail: StrictStr = Field(min_length=1)
    evidence_node_ids: tuple[StrictStr, ...] = ()


class Task14ExclusionV1(ImmutableContractModel):
    exclusion_id: StrictStr = Field(min_length=1)
    evidence_kind: StrictStr = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)


class Task14StructuralReportV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task14-review.v1"] = (
        "memupdatebench.core-task14-review.v1"
    )
    report_id: StrictStr = Field(min_length=1)
    review_id: StrictStr = Field(min_length=1)
    trusted_source_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    trusted_source_tree_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    graph: Task14EvidenceGraphV1
    checks: tuple[Task14CheckV1, ...]
    findings: tuple[Task14FindingV1, ...]
    exclusions: tuple[Task14ExclusionV1, ...]

    @model_validator(mode="after")
    def _canonical_report(self) -> "Task14StructuralReportV1":
        check_ids = tuple(item.check_id for item in self.checks)
        finding_ids = tuple(item.finding_id for item in self.findings)
        exclusion_ids = tuple(item.exclusion_id for item in self.exclusions)
        for values, label in (
            (check_ids, "checks"),
            (finding_ids, "findings"),
            (exclusion_ids, "exclusions"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"report contains duplicate {label}")
        graph_ids = {item.node_id for item in self.graph.nodes}
        for item in (*self.checks, *self.findings):
            if any(value not in graph_ids for value in item.evidence_node_ids):
                raise ValueError("report references foreign evidence nodes")
        return self

    @computed_field(return_type=Literal["NOT_APPROVED", "READY_FOR_VERIFICATION"])
    @property
    def status(self) -> Literal["NOT_APPROVED", "READY_FOR_VERIFICATION"]:
        ready = bool(self.checks) and all(item.passed for item in self.checks) and not self.findings
        return "READY_FOR_VERIFICATION" if ready else "NOT_APPROVED"


class Task14AttestationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task14-attestation.v1"] = (
        "memupdatebench.core-task14-attestation.v1"
    )
    report_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    graph_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    trusted_source_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    trusted_source_tree_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    source_snapshot_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    final_approval_at_verification: StrictBool
    attestation_sha256: StrictStr = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _self_hash(self) -> "Task14AttestationV1":
        if self.attestation_sha256 != task14_attestation_hash_v1(self):
            raise ValueError("attestation_sha256 mismatch")
        return self


class Task14RootManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task14-manifest.v1"] = (
        "memupdatebench.core-task14-manifest.v1"
    )
    artifacts: tuple[Task14ArtifactRefV1, ...]

    @model_validator(mode="after")
    def _exact_manifest(self) -> "Task14RootManifestV1":
        paths = tuple(item.path for item in self.artifacts)
        if paths != TASK14_ARTIFACT_PATHS[:3]:
            raise ValueError("Task 14 manifest must bind the exact first three artifacts")
        return self


class Task14RootIndexV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task14-index.v1"] = (
        "memupdatebench.core-task14-index.v1"
    )
    artifacts: tuple[Task14ArtifactRefV1, ...]

    @model_validator(mode="after")
    def _exact_index(self) -> "Task14RootIndexV1":
        paths = tuple(item.path for item in self.artifacts)
        if paths != TASK14_ARTIFACT_PATHS[:4]:
            raise ValueError("Task 14 index must bind the exact preceding four artifacts")
        if TASK14_ARTIFACT_PATHS[4] in paths:
            raise ValueError("Task 14 index cannot self-bind")
        return self


def task14_canonical_bytes_v1(value: Any, *, exclude: set[str] | None = None) -> bytes:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(
            mode="json",
            exclude=exclude or set(),
            exclude_computed_fields=False,
        )
    else:
        payload = dict(value)
        for field in exclude or set():
            payload.pop(field, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_payload(value: Any, *, exclude: set[str] | None = None) -> str:
    return hashlib.sha256(task14_canonical_bytes_v1(value, exclude=exclude)).hexdigest()


def task14_graph_hash_v1(value: Task14EvidenceGraphV1) -> str:
    return _hash_payload(value)


def task14_report_hash_v1(value: Task14StructuralReportV1) -> str:
    return _hash_payload(value)


def task14_attestation_hash_v1(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return _hash_payload(value, exclude={"attestation_sha256"})
    payload = dict(value)
    payload.setdefault("schema_version", "memupdatebench.core-task14-attestation.v1")
    return _hash_payload(payload, exclude={"attestation_sha256"})


def task14_attestation_file_hash_v1(value: "Task14AttestationV1") -> str:
    return _hash_payload(value)


def task14_manifest_hash_v1(value: Task14RootManifestV1) -> str:
    return _hash_payload(value)


def task14_index_hash_v1(value: Task14RootIndexV1) -> str:
    return _hash_payload(value)


@dataclass(frozen=True)
class VerifiedCoreFinalRelease:
    report: Task14StructuralReportV1
    attestation: Task14AttestationV1
    manifest: Task14RootManifestV1
    index: Task14RootIndexV1
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED_RELEASE_TOKEN:
            raise TypeError("VerifiedCoreFinalRelease requires explicit verification")

    @property
    def final_approved(self) -> bool:
        return bool(
            self.report.status == "READY_FOR_VERIFICATION"
            and self.attestation.final_approval_at_verification
        )


def verify_task14_release_v1(
    report: Task14StructuralReportV1,
    attestation: Task14AttestationV1,
    manifest: Task14RootManifestV1,
    index: Task14RootIndexV1,
) -> VerifiedCoreFinalRelease:
    if not all(
        type(value) is expected
        for value, expected in (
            (report, Task14StructuralReportV1),
            (attestation, Task14AttestationV1),
            (manifest, Task14RootManifestV1),
            (index, Task14RootIndexV1),
        )
    ):
        raise TypeError("Task 14 release verification requires exact contract types")
    from mub.vnext.release.task14_review import (
        TASK14_REQUIRED_CHECK_IDS,
        TASK14_REQUIRED_EDGE_TRIPLES,
        TASK14_REQUIRED_EXCLUSIONS,
        TASK14_REQUIRED_NODE_IDS,
    )

    if {item.check_id for item in report.checks} != TASK14_REQUIRED_CHECK_IDS:
        raise ValueError("Task 14 report lacks required checks")
    if not all(item.passed for item in report.checks) or report.findings:
        raise ValueError("Task 14 report is not structurally ready")
    if {item.node_id for item in report.graph.nodes} != TASK14_REQUIRED_NODE_IDS:
        raise ValueError("Task 14 graph lacks required evidence nodes")
    if {
        (item.source_node_id, item.target_node_id, item.edge_type)
        for item in report.graph.edges
    } != TASK14_REQUIRED_EDGE_TRIPLES:
        raise ValueError("Task 14 graph lacks required evidence edges")
    if report.exclusions != TASK14_REQUIRED_EXCLUSIONS:
        raise ValueError("Task 14 report lacks required exclusions")
    if attestation.report_sha256 != task14_report_hash_v1(report):
        raise ValueError("Task 14 attestation report hash mismatch")
    if attestation.graph_sha256 != task14_graph_hash_v1(report.graph):
        raise ValueError("Task 14 attestation graph hash mismatch")
    if attestation.trusted_source_revision != report.trusted_source_revision:
        raise ValueError("Task 14 trusted revision mismatch")
    if attestation.trusted_source_tree_sha256 != report.trusted_source_tree_sha256:
        raise ValueError("Task 14 trusted source tree mismatch")
    expected_approval = report.status == "READY_FOR_VERIFICATION"
    if attestation.final_approval_at_verification != expected_approval:
        raise ValueError("Task 14 approval attestation mismatch")
    expected_manifest = (
        task14_report_hash_v1(report),
        task14_graph_hash_v1(report.graph),
        task14_attestation_file_hash_v1(attestation),
    )
    if tuple(item.sha256 for item in manifest.artifacts) != expected_manifest:
        raise ValueError("Task 14 manifest hash chain mismatch")
    if index.artifacts[:3] != manifest.artifacts:
        raise ValueError("Task 14 index does not bind the manifest artifacts")
    if index.artifacts[3].sha256 != task14_manifest_hash_v1(manifest):
        raise ValueError("Task 14 index manifest hash mismatch")
    return VerifiedCoreFinalRelease(
        report=report,
        attestation=attestation,
        manifest=manifest,
        index=index,
        _token=_VERIFIED_RELEASE_TOKEN,
    )


# Short aliases used by downstream review/publish modules.
Task14ArtifactRef = Task14ArtifactRefV1
Task14RootSnapshot = Task14RootSnapshotV1
Task14EvidenceNode = Task14EvidenceNodeV1
Task14EvidenceEdge = Task14EvidenceEdgeV1
Task14EvidenceGraph = Task14EvidenceGraphV1
Task14Check = Task14CheckV1
Task14Finding = Task14FindingV1
Task14Exclusion = Task14ExclusionV1
Task14StructuralReport = Task14StructuralReportV1
Task14Attestation = Task14AttestationV1
Task14RootManifest = Task14RootManifestV1
Task14RootIndex = Task14RootIndexV1


__all__ = [
    "TASK14_ALLOWED_EDGE_TYPES",
    "TASK14_ARTIFACT_PATHS",
    "Task14ArtifactRefV1",
    "Task14AttestationV1",
    "Task14CheckV1",
    "Task14EvidenceEdgeV1",
    "Task14EvidenceGraphV1",
    "Task14EvidenceNodeV1",
    "Task14ExclusionV1",
    "Task14FindingV1",
    "Task14RootEntryV1",
    "Task14RootIndexV1",
    "Task14RootManifestV1",
    "Task14RootSnapshotV1",
    "Task14StructuralReportV1",
    "VerifiedCoreFinalRelease",
    "task14_attestation_file_hash_v1",
    "task14_attestation_hash_v1",
    "task14_canonical_bytes_v1",
    "task14_graph_hash_v1",
    "task14_index_hash_v1",
    "task14_manifest_hash_v1",
    "task14_report_hash_v1",
    "verify_task14_release_v1",
]
