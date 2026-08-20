from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
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
    task14_attestation_file_hash_v1,
    task14_attestation_hash_v1,
    task14_canonical_bytes_v1,
    task14_graph_hash_v1,
    task14_index_hash_v1,
    task14_manifest_hash_v1,
    task14_report_hash_v1,
    verify_task14_release_v1,
)
from mub.vnext.release.task14_review import build_task14_structural_report_v1
from mub.vnext.release.task14_sources import (
    Task14LoadedSourcesV1,
    revalidate_task14_sources_v1,
)
from mub.vnext.statistics.task13_v3 import (
    _directory_commit_noreplace_v3,
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
    verify_task14_release_v1(report, attestation, manifest, index)
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


def verify_task14_root_v1(root: Path) -> VerifiedCoreFinalRelease:
    checked = Path(root).resolve(strict=True)
    if checked.is_symlink() or not checked.is_dir():
        raise ValueError("Task 14 final root must be a real directory")
    entries = tuple(checked.iterdir())
    if tuple(sorted(item.name for item in entries)) != tuple(sorted(TASK14_ARTIFACT_PATHS)):
        raise ValueError("Task 14 final root must contain exactly five artifacts")
    if any(not _regular_single_link(item) for item in entries):
        raise ValueError("Task 14 final root contains an unsafe artifact")
    report = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[0], Task14StructuralReportV1)
    from mub.vnext.release.task14_contracts import Task14EvidenceGraphV1

    graph = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[1], Task14EvidenceGraphV1)
    if graph != report.graph:
        raise ValueError("Task 14 graph artifact differs from report graph")
    attestation = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[2], Task14AttestationV1)
    manifest = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[3], Task14RootManifestV1)
    index = _parse_exact_model(checked / TASK14_ARTIFACT_PATHS[4], Task14RootIndexV1)
    actual = tuple(
        hashlib.sha256((checked / path).read_bytes()).hexdigest()
        for path in TASK14_ARTIFACT_PATHS[:4]
    )
    if tuple(item.sha256 for item in index.artifacts) != actual:
        raise ValueError("Task 14 index does not bind final artifact bytes")
    return verify_task14_release_v1(report, attestation, manifest, index)


def _validate_staged_bytes(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise ValueError("Task 14 staged artifact bytes changed")


def _assert_output_safe(output_root: Path, loaded: Task14LoadedSourcesV1) -> tuple[Path, Path]:
    output = Path(output_root)
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError("Task 14 output root already exists")
    parent = output.parent.resolve(strict=True)
    protected = (
        Path(loaded.paths.core_root).resolve(strict=True),
        Path(loaded.paths.evidence_root).resolve(strict=True),
        Path(loaded.paths.task13_root).resolve(strict=True),
        Path(loaded.paths.task13_audit_path).resolve(strict=True),
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
    return output, parent


def publish_task14_review_v1(
    loaded: Task14LoadedSourcesV1,
    *,
    review_id: str,
    trusted_source_revision: str,
    trusted_source_tree_sha256: str,
    output_root: Path,
) -> Task14PublicationResultV1:
    output, parent = _assert_output_safe(output_root, loaded)
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
            validators=validators,
            pre_publish=pre_publish,
        )
        staged = verify_task14_root_v1(staging)
        if staged.index != publication.index:
            raise ValueError("Task 14 staged publication differs from computed index")
        if not revalidate_task14_sources_v1(loaded):
            raise RuntimeError("Task 14 source changed before directory commit")
        if not _runtime_matches(
            loaded, trusted_source_revision, trusted_source_tree_sha256
        ):
            raise RuntimeError("Task 14 runtime changed before directory commit")
        _directory_commit_noreplace_v3(staging, output)
        committed = True
        verified = verify_task14_root_v1(output)
        if not revalidate_task14_sources_v1(loaded) or not _runtime_matches(
            loaded, trusted_source_revision, trusted_source_tree_sha256
        ):
            raise RuntimeError("Task 14 current roots changed after publication")
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
        if not committed and staging.exists():
            metadata = staging.stat()
            if (metadata.st_dev, metadata.st_ino) == staging_identity:
                names = tuple(sorted(item.name for item in staging.iterdir()))
                if set(names) <= set(TASK14_ARTIFACT_PATHS):
                    shutil.rmtree(staging)


__all__ = [
    "Task14PublicationResultV1",
    "Task14PublicationV1",
    "build_task14_publication_v1",
    "publish_task14_review_v1",
    "verify_task14_root_v1",
]
