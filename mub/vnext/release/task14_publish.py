from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import uuid
from types import MappingProxyType
from typing import Mapping

from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.release.task14_contracts import (
    TASK14_ARTIFACT_PATHS,
    Task14ArtifactRefV1,
    Task14AttestationV1,
    Task14RootIndexV1,
    Task14RootManifestV1,
    Task14StructuralReportV1,
    VerifiedCoreFinalRelease,
    task14_attestation_hash_v1,
    task14_canonical_bytes_v1,
    task14_graph_hash_v1,
    task14_report_hash_v1,
    _verify_task14_release_current_v1,
)
from mub.vnext.release.task14_review import build_task14_structural_report_v1
from mub.vnext.release.task14_sources import (
    Task14LoadedSourcesV1,
    _source_locations,
    revalidate_task14_sources_v1,
)
from mub.vnext.statistics.task13_v3 import (
    _directory_commit_noreplace_v3,
    _fsync_parent_directory_v3,
    current_clean_task13_runtime_v3,
)


@dataclass(frozen=True)
class Task14PublicationV1:
    report: Task14StructuralReportV1
    attestation: Task14AttestationV1
    manifest: Task14RootManifestV1
    index: Task14RootIndexV1
    artifact_bytes: Mapping[str, bytes]


@dataclass(frozen=True)
class Task14PublicationResultV1:
    output_root: Path
    verified_release: VerifiedCoreFinalRelease
    index_sha256: str
    attestation_sha256: str

    @property
    def final_approved(self) -> bool:
        return self.verified_release.final_approved


def _artifact(path: str, raw: bytes) -> Task14ArtifactRefV1:
    return Task14ArtifactRefV1(
        artifact_id=path.removesuffix(".json"),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        media_type="application/json",
        byte_count=len(raw),
        record_count=1,
        root_kind="task14_output",
    )


def _runtime_matches(
    loaded: Task14LoadedSourcesV1,
    revision: str,
    tree_sha256: str,
) -> bool:
    try:
        runtime = current_clean_task13_runtime_v3(loaded.paths.repository_root)
        return bool(
            runtime.runtime_revision == revision
            and runtime.runtime_tree_sha256 == tree_sha256
        )
    except Exception:
        return False


def build_task14_publication_v1(
    loaded: Task14LoadedSourcesV1,
    *,
    review_id: str,
    trusted_source_revision: str,
    trusted_source_tree_sha256: str,
) -> Task14PublicationV1:
    if not revalidate_task14_sources_v1(loaded):
        raise RuntimeError("Task 14 source snapshot changed before attestation")
    if not _runtime_matches(
        loaded, trusted_source_revision, trusted_source_tree_sha256
    ):
        raise RuntimeError("Task 14 trusted runtime binding mismatch")
    report = build_task14_structural_report_v1(
        loaded,
        review_id=review_id,
        trusted_source_revision=trusted_source_revision,
        trusted_source_tree_sha256=trusted_source_tree_sha256,
    )
    graph = report.graph
    attestation_payload = {
        "report_sha256": task14_report_hash_v1(report),
        "graph_sha256": task14_graph_hash_v1(graph),
        "trusted_source_revision": trusted_source_revision,
        "trusted_source_tree_sha256": trusted_source_tree_sha256,
        "source_snapshot_sha256": loaded.aggregate_snapshot_sha256,
        "final_approval_at_verification": report.status == "READY_FOR_VERIFICATION",
    }
    attestation = Task14AttestationV1(
        **attestation_payload,
        attestation_sha256=task14_attestation_hash_v1(attestation_payload),
    )
    report_bytes = task14_canonical_bytes_v1(report)
    graph_bytes = task14_canonical_bytes_v1(graph)
    attestation_bytes = task14_canonical_bytes_v1(attestation)
    manifest = Task14RootManifestV1(
        artifacts=(
            _artifact(TASK14_ARTIFACT_PATHS[0], report_bytes),
            _artifact(TASK14_ARTIFACT_PATHS[1], graph_bytes),
            _artifact(TASK14_ARTIFACT_PATHS[2], attestation_bytes),
        )
    )
    manifest_bytes = task14_canonical_bytes_v1(manifest)
    index = Task14RootIndexV1(
        artifacts=manifest.artifacts
        + (_artifact(TASK14_ARTIFACT_PATHS[3], manifest_bytes),)
    )
    index_bytes = task14_canonical_bytes_v1(index)
    _verify_task14_release_current_v1(report, attestation, manifest, index)
    payloads = MappingProxyType(
        {
            TASK14_ARTIFACT_PATHS[0]: report_bytes,
            TASK14_ARTIFACT_PATHS[1]: graph_bytes,
            TASK14_ARTIFACT_PATHS[2]: attestation_bytes,
            TASK14_ARTIFACT_PATHS[3]: manifest_bytes,
            TASK14_ARTIFACT_PATHS[4]: index_bytes,
        }
    )
    return Task14PublicationV1(report, attestation, manifest, index, payloads)


def _regular_single_link(path: Path) -> bool:
    metadata = path.lstat()
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and not (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        and getattr(metadata, "st_nlink", 1) == 1
    )


def _parse_exact_model(path: Path, model_type):
    raw = path.read_bytes()
    if model_type is Task14StructuralReportV1:
        import json

        payload = json.loads(raw)
        persisted_status = payload.pop("status", None)
        model = model_type.model_validate(payload)
        if persisted_status != model.status:
            raise ValueError("Task 14 persisted structural status is not derived")
    else:
        model = model_type.model_validate_json(raw)
    if task14_canonical_bytes_v1(model) != raw:
        raise ValueError(f"Task 14 artifact is not canonical: {path.name}")
    return model


def _validate_required_review_semantics(report: Task14StructuralReportV1) -> None:
    from mub.vnext.release.task14_review import (
        TASK14_REQUIRED_CHECK_IDS,
        TASK14_REQUIRED_EDGE_TRIPLES,
        TASK14_REQUIRED_EXCLUSIONS,
        TASK14_REQUIRED_NODE_IDS,
    )

    if {item.check_id for item in report.checks} != TASK14_REQUIRED_CHECK_IDS:
        raise ValueError("Task 14 final report lacks the exact required checks")
    failed_checks = {item.check_id for item in report.checks if not item.passed}
    finding_checks = {
        item.finding_id.removeprefix("failed:")
        for item in report.findings
        if item.finding_id.startswith("failed:")
    }
    if report.status == "READY_FOR_VERIFICATION":
        if failed_checks or report.findings:
            raise ValueError("Task 14 ready report contains failed checks or findings")
    elif not failed_checks or finding_checks != failed_checks:
        raise ValueError("Task 14 NOT_APPROVED report does not bind every failed check")
    if {item.node_id for item in report.graph.nodes} != TASK14_REQUIRED_NODE_IDS:
        raise ValueError("Task 14 final evidence graph lacks required nodes")
    if {
        (item.source_node_id, item.target_node_id, item.edge_type)
        for item in report.graph.edges
    } != TASK14_REQUIRED_EDGE_TRIPLES:
        raise ValueError("Task 14 final evidence graph lacks required edges")
    if report.exclusions != TASK14_REQUIRED_EXCLUSIONS:
        raise ValueError("Task 14 final report lacks required exclusions")


def _reject_reparse_components(path: Path) -> None:
    selected = path if path.is_absolute() else Path.cwd() / path
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Task 14 path contains a reparse component: {current}")


def verify_task14_root_v1(
    root: Path,
    *,
    loaded_sources: Task14LoadedSourcesV1,
) -> VerifiedCoreFinalRelease:
    selected = Path(root)
    _reject_reparse_components(selected)
    if selected.is_symlink() or (
        selected.exists()
        and getattr(selected.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError("Task 14 final root may not be a link or reparse point")
    checked = selected.resolve(strict=True)
    if checked.is_symlink() or not checked.is_dir():
        raise ValueError("Task 14 final root must be a real directory")
    entries = tuple(checked.iterdir())
    if tuple(sorted(item.name for item in entries)) != tuple(sorted(TASK14_ARTIFACT_PATHS)):
        raise ValueError("Task 14 final root must contain exactly five artifacts")
    if any(not _regular_single_link(item) for item in entries):
        raise ValueError("Task 14 final root contains an unsafe artifact")
    report = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[0], Task14StructuralReportV1)
    expected_report = build_task14_structural_report_v1(
        loaded_sources,
        review_id=report.review_id,
        trusted_source_revision=report.trusted_source_revision,
        trusted_source_tree_sha256=report.trusted_source_tree_sha256,
    )
    if report != expected_report:
        raise ValueError("Task 14 final report differs from current source-derived review")
    _validate_required_review_semantics(report)
    from mub.vnext.release.task14_contracts import Task14EvidenceGraphV1

    graph = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[1], Task14EvidenceGraphV1)
    if graph != report.graph:
        raise ValueError("Task 14 graph artifact differs from report graph")
    attestation = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[2], Task14AttestationV1)
    manifest = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[3], Task14RootManifestV1)
    index = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[4], Task14RootIndexV1)
    actual_refs = tuple(
        _artifact(path, (checked / path).read_bytes())
        for path in TASK14_ARTIFACT_PATHS[:4]
    )
    if index.artifacts != actual_refs:
        raise ValueError("Task 14 index does not bind final artifact bytes and metadata")
    verified = _verify_task14_release_current_v1(report, attestation, manifest, index)
    if not revalidate_task14_sources_v1(loaded_sources):
        raise RuntimeError("Task 14 current source roots no longer match attestation")
    if attestation.source_snapshot_sha256 != loaded_sources.aggregate_snapshot_sha256:
        raise RuntimeError("Task 14 attestation source snapshot mismatch")
    if not _runtime_matches(
        loaded_sources,
        attestation.trusted_source_revision,
        attestation.trusted_source_tree_sha256,
    ):
        raise RuntimeError("Task 14 attested runtime no longer matches current source")
    return verified


def _validate_staged_bytes(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise ValueError("Task 14 staged artifact bytes changed")


def _assert_output_safe(
    output_root: Path,
    loaded: Task14LoadedSourcesV1,
) -> tuple[Path, Path, tuple[int, int]]:
    requested = Path(output_root)
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    _reject_reparse_components(requested)
    if requested.is_symlink() or (
        requested.exists()
        and getattr(requested.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError("Task 14 output root may not be a link or reparse point")
    output = requested.resolve(strict=False)
    if output.exists():
        raise FileExistsError("Task 14 output root already exists")
    parent = output.parent.resolve(strict=True)
    protected = (
        Path(loaded.paths.core_root).resolve(strict=True),
        Path(loaded.paths.evidence_root).resolve(strict=True),
        Path(loaded.paths.task13_root).resolve(strict=True),
        Path(loaded.paths.task13_audit_path).resolve(strict=True),
        Path(loaded.paths.repository_root).resolve(strict=True),
    )
    output_key = os.path.normcase(str(output))
    for item in protected:
        item_key = os.path.normcase(str(item))
        try:
            common = os.path.commonpath((output_key, item_key))
        except ValueError:
            continue
        if common in {output_key, item_key}:
            raise ValueError("Task 14 output root overlaps a source root")
    parent_metadata = parent.stat()
    return output, parent, (parent_metadata.st_dev, parent_metadata.st_ino)


def _cleanup_owned_staging(
    staging: Path,
    staging_identity: tuple[int, int],
    expected: Mapping[str, bytes],
) -> None:
    if not staging.exists():
        return
    metadata = staging.stat()
    if (metadata.st_dev, metadata.st_ino) != staging_identity or staging.is_symlink():
        return
    entries = tuple(staging.iterdir())
    for entry in entries:
        expected_bytes = expected.get(entry.name)
        if (
            expected_bytes is None
            or not _regular_single_link(entry)
            or entry.read_bytes() != expected_bytes
        ):
            return
    for entry in entries:
        entry.unlink()
    staging.rmdir()


def publish_task14_review_v1(
    loaded: Task14LoadedSourcesV1,
    *,
    review_id: str,
    trusted_source_revision: str,
    trusted_source_tree_sha256: str,
    output_root: Path,
) -> Task14PublicationResultV1:
    output, parent, parent_identity = _assert_output_safe(output_root, loaded)
    publication = build_task14_publication_v1(
        loaded,
        review_id=review_id,
        trusted_source_revision=trusted_source_revision,
        trusted_source_tree_sha256=trusted_source_tree_sha256,
    )
    staging = parent / f".mub-task14-stage-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    staging_identity = staging.stat().st_dev, staging.stat().st_ino
    committed = False
    try:
        destinations = {
            staging / name: raw for name, raw in publication.artifact_bytes.items()
        }
        validators = {
            path: (lambda selected, expected=raw: _validate_staged_bytes(selected, expected))
            for path, raw in destinations.items()
        }

        def pre_publish() -> None:
            if not revalidate_task14_sources_v1(loaded):
                raise RuntimeError("Task 14 source changed during publication")
            if not _runtime_matches(
                loaded, trusted_source_revision, trusted_source_tree_sha256
            ):
                raise RuntimeError("Task 14 runtime changed during publication")

        publish_files_atomically(
            destinations,
            overwrite=False,
            source_paths=tuple(_source_locations(loaded.paths).values()),
            validators=validators,
            pre_publish=pre_publish,
        )
        staged = verify_task14_root_v1(staging, loaded_sources=loaded)
        if staged.index != publication.index:
            raise ValueError("Task 14 staged publication differs from computed index")
        current_parent = parent.stat()
        current_staging = staging.stat()
        if (
            (current_parent.st_dev, current_parent.st_ino) != parent_identity
            or (current_staging.st_dev, current_staging.st_ino) != staging_identity
            or parent.is_symlink()
            or staging.is_symlink()
        ):
            raise RuntimeError("Task 14 parent or staging identity changed before commit")
        _directory_commit_noreplace_v3(staging, output)
        committed = True
        final_metadata = output.stat()
        if (final_metadata.st_dev, final_metadata.st_ino) != staging_identity:
            raise RuntimeError("Task 14 final root identity differs from owned staging")
        _fsync_parent_directory_v3(parent)
        try:
            verified = verify_task14_root_v1(output, loaded_sources=loaded)
        except Exception as exc:
            raise RuntimeError(
                "Task 14 final root committed but post-commit verification failed; root preserved"
            ) from exc
        if verified.index != publication.index:
            raise ValueError("Task 14 reopened final root differs from staged index")
        return Task14PublicationResultV1(
            output_root=output,
            verified_release=verified,
            index_sha256=hashlib.sha256(
                publication.artifact_bytes[TASK14_ARTIFACT_PATHS[4]]
            ).hexdigest(),
            attestation_sha256=publication.attestation.attestation_sha256,
        )
    finally:
        if not committed:
            _cleanup_owned_staging(
                staging, staging_identity, publication.artifact_bytes
            )


__all__ = [
    "Task14PublicationResultV1",
    "Task14PublicationV1",
    "build_task14_publication_v1",
    "publish_task14_review_v1",
    "verify_task14_root_v1",
]
