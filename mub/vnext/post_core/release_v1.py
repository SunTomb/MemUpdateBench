from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import uuid

from mub.vnext.post_core.contracts_v1 import (
    ArtifactIndexV1,
    ModelCandidateV1,
    POST_CORE_ARTIFACT_ORDER,
    ReleaseManifestV1,
    canonical_bytes,
)
from mub.vnext.external.canaries_v3 import CoreTaskReleaseManifestV1
from mub.vnext.release.task14_contracts import (
    TASK14_ARTIFACT_PATHS,
    Task14AttestationV1,
    Task14EvidenceGraphV1,
    Task14RootIndexV1,
    Task14RootManifestV1,
    Task14StructuralReportV1,
    _verify_task14_release_current_v1,
    task14_canonical_bytes_v1,
)
from mub.vnext.post_core.model_registry_v1 import (
    build_initial_model_registry_v1,
    validate_model_registry_v1,
)
from mub.vnext.post_core.planning_v1 import (
    ExecutionPlanV1,
    build_phase0_execution_plan_v1,
)
from mub.vnext.post_core.provenance_v1 import (
    ProvenanceRecordV1,
    sha256_file,
    validate_secret_free,
)
from mub.vnext.post_core.qualification_v1 import (
    CapabilityProbeReportV1,
    QualificationReportV1,
    qualify_registry_offline_v1,
)


POST_CORE_INDEX_PATH = "post_core_artifact_index.json"
POST_CORE_ARTIFACTS = (*POST_CORE_ARTIFACT_ORDER, POST_CORE_INDEX_PATH)
EXPECTED_REGISTRY_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)
SHA256_PATTERN = set("0123456789abcdef")
EXPECTED_CORE_MANIFEST_SHA256 = "dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b"
EXPECTED_TASK14_INDEX_SHA256 = "2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035"

# Typed CLI outcomes.  Success-with-pending is intentionally a successful exit:
# Phase 0 is a metadata release, and pending identity is an expected result.
EXIT_SUCCESS_WITH_PENDING = 0
EXIT_SUCCESS = EXIT_SUCCESS_WITH_PENDING
EXIT_BLOCKED = 10
EXIT_USAGE = 11
EXIT_CONTRACT_USAGE = EXIT_USAGE
EXIT_STALE_SOURCE = 12
EXIT_PUBLICATION = 13
EXIT_UNTRUSTED_RUNTIME = 14


class PostCoreReleaseError(RuntimeError):
    """Base error for source-bound post-Core publication failures."""


class StaleSourceError(PostCoreReleaseError):
    """An authenticated source changed or failed its frozen hash check."""


class UnsafePathError(PostCoreReleaseError, ValueError):
    """A source, staging, or output path is unsafe for publication."""


class CommittedPostCoreReleaseError(PostCoreReleaseError):
    """The output directory was committed but could not be verified."""

    def __init__(self, committed_root: Path, message: str, *, recovery: str | None = None) -> None:
        self.committed_root = Path(committed_root)
        self.recovery = recovery or "preserve the committed root and inspect its artifacts before retrying"
        super().__init__(f"{message}; committed root preserved at {self.committed_root}; recovery: {self.recovery}")


@dataclass(frozen=True)
class _SourceSnapshot:
    path: Path
    identity: tuple[int, int]
    byte_count: int
    sha256: str
    raw: bytes
    siblings: tuple["_SourceSnapshot", ...] = ()


@dataclass(frozen=True)
class PostCoreReleaseConfigV1:
    schema_version: str
    release_id: str
    phase: int
    network_allowed: bool
    core_manifest_sha256: str
    core_task14_index_sha256: str
    registry_keys: tuple[str, ...]
    config_sha256: str
    config_raw: bytes
    config_path: Path | None = None
    source_snapshot: _SourceSnapshot | None = None


@dataclass(frozen=True)
class _OwnedMember:
    name: str
    identity: tuple[int, int]
    byte_count: int
    sha256: str


class _RegistryInput(dict[str, ModelCandidateV1]):
    def __init__(self, values: Mapping[str, ModelCandidateV1], source: _SourceSnapshot) -> None:
        super().__init__(values)
        self.source_snapshot = source


@dataclass(frozen=True)
class PostCorePublicationV1:
    release_id: str
    output_root: Path | None
    artifact_bytes: Mapping[str, bytes]
    index_sha256: str
    pending_count: int
    provider_calls: int
    model_loads: int
    network_calls: int
    executable_call_count: int


def _canonical_mapping_bytes(value: Any) -> bytes:
    return canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= SHA256_PATTERN


def _regular_single_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and not (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        and getattr(metadata, "st_nlink", 1) == 1
    )


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_reparse_components(path: Path) -> None:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    selected = Path(os.path.normpath(str(selected)))
    anchor = Path(selected.anchor) if selected.anchor else Path.cwd()
    current = anchor
    parts = selected.parts[1:] if selected.anchor else selected.parts
    for part in parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise UnsafePathError(f"path contains a reparse component: {current}")


def _absolute_path(path: Path, *, require_exists: bool = False) -> Path:
    selected = Path(path)
    _reject_reparse_components(selected)
    if require_exists:
        try:
            selected = selected.resolve(strict=True)
        except OSError as exc:
            raise UnsafePathError(f"path does not exist: {path}") from exc
    else:
        selected = Path(os.path.abspath(os.path.normpath(str(selected))))
    _reject_reparse_components(selected)
    return selected


def _read_source(path: Path, label: str) -> _SourceSnapshot:
    selected = _absolute_path(path, require_exists=True)
    if not _regular_single_link(selected):
        raise UnsafePathError(f"{label} must be a regular single-link file")
    before = selected.stat()
    raw = selected.read_bytes()
    after = selected.stat()
    identity = (after.st_dev, after.st_ino)
    if (before.st_dev, before.st_ino) != identity or before.st_size != after.st_size:
        raise StaleSourceError(f"{label} changed while being read")
    return _SourceSnapshot(selected, identity, len(raw), _sha256(raw), raw)


def _revalidate_source(snapshot: _SourceSnapshot, label: str) -> None:
    selected = snapshot.path
    if not _regular_single_link(selected):
        raise StaleSourceError(f"{label} is no longer a safe regular file")
    metadata = selected.stat()
    if (metadata.st_dev, metadata.st_ino) != snapshot.identity:
        raise StaleSourceError(f"{label} identity changed")
    raw = selected.read_bytes()
    if len(raw) != snapshot.byte_count or _sha256(raw) != snapshot.sha256:
        raise StaleSourceError(f"{label} bytes changed")
    for sibling in snapshot.siblings:
        _revalidate_source(sibling, f"Task 14 sibling {sibling.path.name}")


def _validate_core_manifest(raw: bytes) -> None:
    if _sha256(raw) != EXPECTED_CORE_MANIFEST_SHA256:
        raise StaleSourceError("Core source manifest hash is not the immutable approved release")
    try:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping) or canonical_bytes(payload) != raw:
            raise ValueError("Core source manifest is not canonical")
        manifest = CoreTaskReleaseManifestV1.model_validate(payload)
        without_hash = dict(payload)
        without_hash.pop("release_manifest_hash", None)
        if manifest.release_manifest_hash != _sha256(canonical_bytes(without_hash)):
            raise ValueError("Core source manifest self-hash is invalid")
        if manifest.release_stage != "task_release" or manifest.release_status != "FINAL_APPROVED":
            raise ValueError("Core source manifest release identity is not final-approved")
    except Exception as exc:
        if isinstance(exc, StaleSourceError):
            raise
        raise StaleSourceError("Core source manifest schema or release identity is invalid") from exc


def _validate_task14_index_and_siblings(index_path: Path, raw: bytes) -> None:
    if _sha256(raw) != EXPECTED_TASK14_INDEX_SHA256:
        raise StaleSourceError("Task 14 index hash is not the immutable approved release")
    try:
        index = Task14RootIndexV1.model_validate_json(raw)
        if task14_canonical_bytes_v1(index) != raw:
            raise ValueError("Task 14 index is not canonical")
        root = index_path.parent
        refs = {item.path: item for item in index.artifacts}
        expected = TASK14_ARTIFACT_PATHS[:4]
        if tuple(refs) != expected:
            raise ValueError("Task 14 index artifact order is not exact")
        report_payload = json.loads((root / expected[0]).read_bytes())
        persisted_status = report_payload.pop("status", None)
        report = Task14StructuralReportV1.model_validate(report_payload)
        if persisted_status != report.status or task14_canonical_bytes_v1(report) != (root / expected[0]).read_bytes():
            raise ValueError("Task 14 review report is not canonical")
        graph = Task14EvidenceGraphV1.model_validate_json((root / expected[1]).read_bytes())
        attestation = Task14AttestationV1.model_validate_json((root / expected[2]).read_bytes())
        manifest = Task14RootManifestV1.model_validate_json((root / expected[3]).read_bytes())
        models = (report, graph, attestation, manifest)
        for model, name in zip(models, expected, strict=True):
            if task14_canonical_bytes_v1(model) != (root / name).read_bytes():
                raise ValueError(f"Task 14 sibling is not canonical: {name}")
        _verify_task14_release_current_v1(report, attestation, manifest, index)
        for item in index.artifacts:
            sibling = root / item.path
            if not _regular_single_link(sibling) or item.byte_count != sibling.stat().st_size or item.sha256 != _sha256(sibling.read_bytes()):
                raise ValueError(f"Task 14 sibling hash binding is invalid: {item.path}")
    except Exception as exc:
        if isinstance(exc, StaleSourceError):
            raise
        raise StaleSourceError("Task 14 index schema, sibling set, or hash chain is invalid") from exc


def _capture_task14_siblings(index_path: Path) -> tuple[_SourceSnapshot, ...]:
    return tuple(
        _read_source(index_path.parent / name, f"Task 14 sibling {name}")
        for name in TASK14_ARTIFACT_PATHS[:4]
    )


def _source_pair(core_manifest_path: Path, task14_index_path: Path, expected_core_sha256: str, expected_task14_sha256: str) -> tuple[_SourceSnapshot, _SourceSnapshot]:
    if expected_core_sha256 != EXPECTED_CORE_MANIFEST_SHA256 or expected_task14_sha256 != EXPECTED_TASK14_INDEX_SHA256:
        raise StaleSourceError("caller-selected source hashes are not permitted")
    core = _read_source(core_manifest_path, "Core source manifest")
    task14 = _read_source(task14_index_path, "Task 14 index")
    if core.identity == task14.identity:
        raise UnsafePathError("Core source manifest and Task 14 index alias each other")
    _validate_core_manifest(core.raw)
    _validate_task14_index_and_siblings(task14.path, task14.raw)
    task14 = replace(task14, siblings=_capture_task14_siblings(task14.path))
    return core, task14


def _load_config_payload(
    raw: bytes,
    path: Path,
    *,
    source_snapshot: _SourceSnapshot | None = None,
) -> PostCoreReleaseConfigV1:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid post-Core config JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("post-Core config must be a JSON object")
    if canonical_bytes(payload) != raw:
        raise ValueError("post-Core config must use canonical JSON bytes")
    required = {
        "schema_version", "release_id", "phase", "network_allowed",
        "core_manifest_sha256", "core_task14_index_sha256", "registry_keys",
    }
    if set(payload) != required:
        raise ValueError("post-Core config has an unexpected or missing field")
    keys = payload["registry_keys"]
    if not isinstance(keys, list) or tuple(keys) != EXPECTED_REGISTRY_KEYS:
        raise ValueError("post-Core config registry order is not frozen")
    values = {key: payload[key] for key in required}
    if values["schema_version"] != "memupdatebench.post-core.config.v1":
        raise ValueError("post-Core config schema version mismatch")
    if values["release_id"] != "memupdatebench.post-core.phase0.v1":
        raise ValueError("post-Core release ID mismatch")
    if type(values["phase"]) is not int or values["phase"] != 0:
        raise ValueError("post-Core release phase must be zero")
    if type(values["network_allowed"]) is not bool or values["network_allowed"] is not False:
        raise ValueError("post-Core Phase 0 must forbid network access")
    if values["core_manifest_sha256"] != EXPECTED_CORE_MANIFEST_SHA256:
        raise ValueError("post-Core Core manifest hash is not frozen")
    if values["core_task14_index_sha256"] != EXPECTED_TASK14_INDEX_SHA256:
        raise ValueError("post-Core Task 14 hash is not frozen")
    for label in ("core_manifest_sha256", "core_task14_index_sha256"):
        if not isinstance(values[label], str) or not _is_sha256(values[label]):
            raise ValueError(f"post-Core {label} must be lowercase SHA-256")
    validate_secret_free(payload)
    return PostCoreReleaseConfigV1(
        schema_version=values["schema_version"],
        release_id=values["release_id"],
        phase=values["phase"],
        network_allowed=values["network_allowed"],
        core_manifest_sha256=values["core_manifest_sha256"],
        core_task14_index_sha256=values["core_task14_index_sha256"],
        registry_keys=tuple(keys),
        config_sha256=_sha256(raw),
        config_raw=bytes(raw),
        config_path=None if str(path) == "<mapping>" else path,
        source_snapshot=source_snapshot,
    )


def load_post_core_config_v1(path: Path) -> PostCoreReleaseConfigV1:
    source = _read_source(Path(path), "post-Core config")
    if not _regular_single_link(source.path):
        raise UnsafePathError("post-Core config must be a regular file")
    return _load_config_payload(
        source.raw,
        source.path,
        source_snapshot=source,
    )


def _coerce_config(config: PostCoreReleaseConfigV1 | Path | Mapping[str, Any]) -> PostCoreReleaseConfigV1:
    if isinstance(config, PostCoreReleaseConfigV1):
        if config.core_manifest_sha256 != EXPECTED_CORE_MANIFEST_SHA256 or config.core_task14_index_sha256 != EXPECTED_TASK14_INDEX_SHA256:
            raise ValueError("post-Core config source hashes are not frozen")
        if canonical_bytes(json.loads(config.config_raw)) != config.config_raw or _sha256(config.config_raw) != config.config_sha256:
            raise ValueError("post-Core config bytes are not canonical or hash-bound")
        if config.source_snapshot is not None and (
            config.source_snapshot.path != config.config_path
            or config.source_snapshot.raw != config.config_raw
            or config.source_snapshot.sha256 != config.config_sha256
            or config.source_snapshot.byte_count != len(config.config_raw)
        ):
            raise ValueError("post-Core config source snapshot does not match config bytes")
        return config
    if isinstance(config, (str, Path)):
        return load_post_core_config_v1(Path(config))
    raw = _canonical_mapping_bytes(config)
    return _load_config_payload(raw, Path("<mapping>"))


def load_post_core_registry_v1(path: Path, config: PostCoreReleaseConfigV1 | None = None) -> Mapping[str, ModelCandidateV1]:
    source = _read_source(Path(path), "model registry input")
    try:
        payload = json.loads(source.raw)
        if not isinstance(payload, Mapping) or canonical_bytes(payload) != source.raw:
            raise ValueError("model registry input is not canonical")
        if set(payload) != {"schema_version", "release_id", "registry_keys", "candidates"}:
            raise ValueError("model registry input has unexpected or missing fields")
        if payload["schema_version"] != "memupdatebench.post-core.model-registry.v1":
            raise ValueError("model registry input schema version mismatch")
        if config is not None and payload["release_id"] != config.release_id:
            raise ValueError("model registry input release binding differs")
        candidates = payload["candidates"]
        keys = tuple(payload["registry_keys"])
        if not isinstance(candidates, list) or keys != EXPECTED_REGISTRY_KEYS:
            raise ValueError("model registry input keys are not frozen")
        registry = {item["registry_key"]: ModelCandidateV1.model_validate(item) for item in candidates}
        if tuple(registry) != EXPECTED_REGISTRY_KEYS:
            raise ValueError("model registry input candidate order differs")
        validate_model_registry_v1(registry)
        if config is not None and tuple(registry) != config.registry_keys:
            raise ValueError("model registry input differs from config")
        validate_secret_free(payload, read_environment=False)
        return _RegistryInput(registry, source)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("invalid model registry input") from exc


def _registry_payload(config: PostCoreReleaseConfigV1, registry: Mapping[str, ModelCandidateV1]) -> dict[str, Any]:
    validate_model_registry_v1(registry)
    if tuple(registry) != config.registry_keys:
        raise ValueError("registry keys differ from the release config")
    if any(candidate.identity is not None for candidate in registry.values()):
        raise ValueError("Phase 0 registry identities must remain pending")
    candidates = [candidate.model_dump(mode="json") for candidate in registry.values()]
    payload = {
        "schema_version": "memupdatebench.post-core.model-registry.v1",
        "release_id": config.release_id,
        "registry_keys": list(config.registry_keys),
        "candidates": candidates,
    }
    validate_secret_free(payload, read_environment=False)
    return payload


def _coerce_registry(registry: Mapping[str, ModelCandidateV1] | Path | None, config: PostCoreReleaseConfigV1) -> Mapping[str, ModelCandidateV1]:
    if registry is None:
        return build_initial_model_registry_v1()
    if isinstance(registry, (str, Path)):
        return load_post_core_registry_v1(Path(registry), config)
    validate_model_registry_v1(registry)
    return registry


def _deterministic_provenance_bytes(
    config: PostCoreReleaseConfigV1,
    registry: Mapping[str, ModelCandidateV1],
    registry_payload: Mapping[str, Any],
) -> bytes:
    rows: list[bytes] = []
    for key, candidate_payload in zip(config.registry_keys, registry_payload["candidates"]):
        candidate_raw = _canonical_mapping_bytes(candidate_payload)
        candidate = registry[key]
        row = ProvenanceRecordV1(
            registry_key=key,
            identity_status=candidate.state.value,
            evidence_type="pending_intent_registry",
            artifact_sha256=_sha256(candidate_raw),
            byte_count=len(candidate_raw),
            source_location=f"registry://{key}",
            credential_env_var=candidate.credential_env_var,
            git_revision=None,
            runtime={},
        )
        raw = canonical_bytes(row)
        validate_secret_free(row.model_dump(mode="json"), read_environment=False)
        rows.append(raw)
    return b"".join(item + b"\n" for item in rows)


def _provenance_bytes(
    config: PostCoreReleaseConfigV1,
    registry: Mapping[str, ModelCandidateV1],
    registry_payload: Mapping[str, Any],
    provided: Path | None,
    *,
    source_snapshot: _SourceSnapshot | None = None,
) -> bytes:
    expected = _deterministic_provenance_bytes(config, registry, registry_payload)
    if provided is None:
        return expected
    source = source_snapshot or _read_source(provided, "provenance input")
    _validate_provenance_bytes(source.raw, registry)
    if source.raw != expected:
        raise ValueError(
            "provided provenance must exactly match deterministic pending-intent provenance"
        )
    return source.raw


def _validate_provenance_bytes(raw: bytes, registry: Mapping[str, ModelCandidateV1]) -> tuple[ProvenanceRecordV1, ...]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("provenance JSONL must be nonempty and LF-terminated")
    rows: list[ProvenanceRecordV1] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise ValueError("provenance JSONL contains an empty row")
        row = ProvenanceRecordV1.model_validate_json(line)
        if canonical_bytes(row) != line:
            raise ValueError("provenance JSONL is not canonical")
        validate_secret_free(row.model_dump(mode="json"), read_environment=False)
        rows.append(row)
    if tuple(row.registry_key for row in rows) != tuple(registry):
        raise ValueError("provenance JSONL registry order mismatch")
    return tuple(rows)


def _build_artifacts(
    config: PostCoreReleaseConfigV1,
    core: _SourceSnapshot,
    task14: _SourceSnapshot,
    registry: Mapping[str, ModelCandidateV1],
    provenance_path: Path | None,
    *,
    provenance_snapshot: _SourceSnapshot | None = None,
) -> Mapping[str, bytes]:
    registry_payload = _registry_payload(config, registry)
    registry_bytes = _canonical_mapping_bytes(registry_payload)
    qualification, probes = qualify_registry_offline_v1(registry)
    plan = build_phase0_execution_plan_v1(registry)
    provenance = _provenance_bytes(
        config,
        registry,
        registry_payload,
        provenance_path,
        source_snapshot=provenance_snapshot,
    )
    manifest = ReleaseManifestV1(
        release_id=config.release_id,
        schema_version="memupdatebench.post-core.release.v1",
        artifact_order=POST_CORE_ARTIFACT_ORDER,
        source_hashes={
            "config": config.config_sha256,
            "core_source_manifest": core.sha256,
            "core_task14_index": task14.sha256,
            "model_registry": _sha256(registry_bytes),
            "provenance": _sha256(provenance),
        },
    )
    artifacts: dict[str, bytes] = {
        "post_core_release_manifest.json": canonical_bytes(manifest),
        "model_registry.json": registry_bytes,
        "provenance.jsonl": provenance,
        "qualification_report.json": canonical_bytes(qualification),
        "capability_probe_report.json": canonical_bytes(probes),
        "execution_plan.json": canonical_bytes(plan),
    }
    for raw in artifacts.values():
        validate_secret_free(json.loads(raw) if raw.endswith(b"}") else raw.decode("utf-8"), read_environment=False)
    index = ArtifactIndexV1(
        release_id=config.release_id,
        artifacts=tuple({"path": name, "sha256": _sha256(artifacts[name])} for name in POST_CORE_ARTIFACT_ORDER),
    )
    artifacts[POST_CORE_INDEX_PATH] = canonical_bytes(index, exclude={"canonical_hash"})
    validate_secret_free(index.model_dump(mode="json"), read_environment=False)
    return MappingProxyType(dict((name, artifacts[name]) for name in POST_CORE_ARTIFACTS))


def _publication_from_artifacts(config: PostCoreReleaseConfigV1, artifacts: Mapping[str, bytes], output_root: Path | None = None) -> PostCorePublicationV1:
    registry_payload = json.loads(artifacts["model_registry.json"])
    capability = CapabilityProbeReportV1.model_validate_json(artifacts["capability_probe_report.json"])
    plan = ExecutionPlanV1.model_validate_json(artifacts["execution_plan.json"])
    index_sha = _sha256(artifacts[POST_CORE_INDEX_PATH])
    return PostCorePublicationV1(
        release_id=config.release_id,
        output_root=output_root,
        artifact_bytes=MappingProxyType(dict(artifacts)),
        index_sha256=index_sha,
        pending_count=sum(1 for candidate in registry_payload["candidates"] if candidate["state"].startswith("PENDING")),
        provider_calls=capability.provider_calls,
        model_loads=capability.model_loads,
        network_calls=capability.network_calls,
        executable_call_count=plan.executable_call_count,
    )


def build_post_core_release_v1(
    config: PostCoreReleaseConfigV1 | Path | Mapping[str, Any],
    core_manifest_path: Path,
    task14_index_path: Path,
    *,
    registry: Mapping[str, ModelCandidateV1] | Path | None = None,
    provenance_path: Path | None = None,
) -> PostCorePublicationV1:
    config = _coerce_config(config)
    core, task14 = _source_pair(core_manifest_path, task14_index_path, config.core_manifest_sha256, config.core_task14_index_sha256)
    registry = _coerce_registry(registry, config)
    artifacts = _build_artifacts(config, core, task14, registry, provenance_path)
    return _publication_from_artifacts(config, artifacts)


def _revalidate_config_snapshot(config: PostCoreReleaseConfigV1) -> None:
    if config.source_snapshot is not None:
        _revalidate_source(config.source_snapshot, "post-Core config")


def _source_snapshot_for_optional(path: Path | None, label: str) -> _SourceSnapshot | None:
    return _read_source(path, label) if path is not None else None


def _revalidate_optional(snapshot: _SourceSnapshot | None, label: str) -> None:
    if snapshot is not None:
        _revalidate_source(snapshot, label)


def _flatten_snapshots(sources: Sequence[_SourceSnapshot]) -> tuple[_SourceSnapshot, ...]:
    flattened: list[_SourceSnapshot] = []
    for source in sources:
        flattened.append(source)
        flattened.extend(_flatten_snapshots(source.siblings))
    return tuple(flattened)


def _assert_output_safe(output_root: Path, sources: Sequence[_SourceSnapshot]) -> tuple[Path, Path, tuple[int, int]]:
    requested = Path(output_root)
    _reject_reparse_components(requested)
    output = Path(os.path.abspath(os.path.normpath(str(requested))))
    output_key = os.path.normcase(str(output))
    for source in _flatten_snapshots(sources):
        source_key = os.path.normcase(str(source.path))
        if output_key == source_key or output in source.path.parents or source.path in output.parents:
            raise UnsafePathError("post-Core output root overlaps a source")
    if output.exists() or output.is_symlink() or _is_reparse(output):
        raise FileExistsError("post-Core output root already exists or is unsafe")
    parent = output.parent
    parent = _absolute_path(parent, require_exists=True)
    if not stat.S_ISDIR(parent.stat().st_mode) or _is_reparse(parent):
        raise UnsafePathError("post-Core output parent must be a real directory")
    parent_stat = parent.stat()
    return output, parent, (parent_stat.st_dev, parent_stat.st_ino)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL), getattr(errno, "EOPNOTSUPP", errno.EINVAL)}:
                raise
    finally:
        os.close(descriptor)


def _directory_commit_noreplace(staging: Path, final_root: Path) -> None:
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        move_file_ex = kernel.MoveFileExW
        move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(str(staging), str(final_root), 0x00000008):
            error = kernel.GetLastError()
            if error in {80, 183}:
                raise FileExistsError("post-Core output root appeared during commit")
            raise OSError(error, "post-Core no-replace directory commit failed")
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("no safe no-replace directory commit primitive is available")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(final_root), 1)
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("post-Core output root appeared during commit")
        raise OSError(error, "post-Core no-replace directory commit failed")


def _capture_owned_members(staging: Path) -> tuple[_OwnedMember, ...]:
    observed = tuple(sorted(item.name for item in staging.iterdir()))
    if observed != tuple(sorted(POST_CORE_ARTIFACTS)):
        raise PostCoreReleaseError("staged ownership set is incomplete")
    members: list[_OwnedMember] = []
    for name in POST_CORE_ARTIFACTS:
        item = staging / name
        if not _regular_single_link(item):
            raise UnsafePathError("staged artifact has an unsafe type")
        metadata = item.stat()
        members.append(_OwnedMember(name, (metadata.st_dev, metadata.st_ino), metadata.st_size, sha256_file(item)))
    return tuple(members)


def _owned_matches(staging: Path, member: _OwnedMember) -> bool:
    item = staging / member.name
    if not _regular_single_link(item):
        return False
    metadata = item.stat()
    return (metadata.st_dev, metadata.st_ino) == member.identity and metadata.st_size == member.byte_count and sha256_file(item) == member.sha256


def _cleanup_owned_staging(staging: Path, identity: tuple[int, int], ownership: tuple[_OwnedMember, ...] | None) -> None:
    try:
        metadata = staging.lstat()
    except OSError:
        return
    if _is_reparse(staging) or not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != identity:
        return
    if ownership is None:
        if not tuple(staging.iterdir()):
            staging.rmdir()
        return
    if tuple(sorted(item.name for item in staging.iterdir())) != tuple(sorted(member.name for member in ownership)):
        return
    if any(not _owned_matches(staging, member) for member in ownership):
        return
    for member in ownership:
        (staging / member.name).unlink()
    _fsync_directory(staging.parent)
    staging.rmdir()
    _fsync_directory(staging.parent)


def _write_staged(staging: Path, artifacts: Mapping[str, bytes]) -> None:
    for name in POST_CORE_ARTIFACTS:
        path = staging / name
        raw = artifacts[name]
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if not _regular_single_link(path) or path.read_bytes() != raw:
            raise PostCoreReleaseError("staged artifact bytes changed")
    _fsync_directory(staging)


def verify_post_core_release_v1(
    root: Path,
    config: PostCoreReleaseConfigV1 | Path | Mapping[str, Any],
    core_manifest_path: Path,
    task14_index_path: Path,
    *,
    registry: Mapping[str, ModelCandidateV1] | Path | None = None,
    provenance_path: Path | None = None,
    _provenance_snapshot: _SourceSnapshot | None = None,
) -> PostCorePublicationV1:
    config = _coerce_config(config)
    registry_value = _coerce_registry(registry, config)
    registry_snapshot = getattr(registry_value, "source_snapshot", None)
    provenance_snapshot = _provenance_snapshot or _source_snapshot_for_optional(
        provenance_path, "provenance input"
    )
    checked = _absolute_path(Path(root), require_exists=True)
    if _is_reparse(checked) or not checked.is_dir():
        raise UnsafePathError("post-Core output root must be a real directory")
    entries = tuple(checked.iterdir())
    if tuple(sorted(item.name for item in entries)) != tuple(sorted(POST_CORE_ARTIFACTS)):
        raise PostCoreReleaseError("post-Core output root must contain exactly seven artifacts")
    if any(not _regular_single_link(item) for item in entries):
        raise UnsafePathError("post-Core output contains an unsafe artifact")
    core, task14 = _source_pair(core_manifest_path, task14_index_path, config.core_manifest_sha256, config.core_task14_index_sha256)
    actual = MappingProxyType({name: (checked / name).read_bytes() for name in POST_CORE_ARTIFACTS})
    try:
        manifest = ReleaseManifestV1.model_validate_json(actual[POST_CORE_ARTIFACT_ORDER[0]])
        if canonical_bytes(manifest) != actual[POST_CORE_ARTIFACT_ORDER[0]]:
            raise PostCoreReleaseError("release manifest is not canonical")
        if manifest.release_id != config.release_id or manifest.source_hashes != {
            "config": config.config_sha256,
            "core_source_manifest": core.sha256,
            "core_task14_index": task14.sha256,
            "model_registry": _sha256(actual["model_registry.json"]),
            "provenance": _sha256(actual["provenance.jsonl"]),
        }:
            raise StaleSourceError("release manifest source binding differs")
        registry_payload = json.loads(actual["model_registry.json"])
        if not isinstance(registry_payload, Mapping) or canonical_bytes(registry_payload) != actual["model_registry.json"]:
            raise PostCoreReleaseError("model registry is not canonical")
        if tuple(registry_payload.get("registry_keys", ())) != config.registry_keys:
            raise PostCoreReleaseError("model registry key order differs")
        parsed_registry = {item["registry_key"]: ModelCandidateV1.model_validate(item) for item in registry_payload["candidates"]}
        if tuple(parsed_registry) != config.registry_keys:
            raise PostCoreReleaseError("model registry candidate order differs")
        validate_model_registry_v1(parsed_registry)
        _validate_provenance_bytes(actual["provenance.jsonl"], parsed_registry)
        qualification = QualificationReportV1.model_validate_json(actual["qualification_report.json"])
        probes = CapabilityProbeReportV1.model_validate_json(actual["capability_probe_report.json"])
        plan = ExecutionPlanV1.model_validate_json(actual["execution_plan.json"])
        for model, name in ((qualification, "qualification_report.json"), (probes, "capability_probe_report.json"), (plan, "execution_plan.json")):
            if canonical_bytes(model) != actual[name]:
                raise PostCoreReleaseError(f"{name} is not canonical")
        if probes.network_allowed is not False or probes.provider_calls != 0 or probes.model_loads != 0 or plan.network_allowed is not False or plan.executable_call_count != 0:
            raise PostCoreReleaseError("Phase 0 execution boundary is not closed")
        index = ArtifactIndexV1.model_validate_json(actual[POST_CORE_INDEX_PATH])
        if canonical_bytes(index, exclude={"canonical_hash"}) != actual[POST_CORE_INDEX_PATH] or index.release_id != config.release_id:
            raise PostCoreReleaseError("artifact index is not canonical or release-bound")
        if tuple(item.path for item in index.artifacts) != POST_CORE_ARTIFACT_ORDER:
            raise PostCoreReleaseError("artifact index order is not exact")
        for item in index.artifacts:
            if item.sha256 != _sha256(actual[item.path]):
                raise PostCoreReleaseError("artifact index hash does not match artifact bytes")
        expected = _build_artifacts(
            config,
            core,
            task14,
            registry_value,
            provenance_path,
            provenance_snapshot=provenance_snapshot,
        )
        if dict(actual) != dict(expected):
            raise PostCoreReleaseError("reopened post-Core artifacts differ from deterministic build")
    except PostCoreReleaseError:
        raise
    except Exception as exc:
        raise PostCoreReleaseError(f"post-Core artifact verification failed: {exc}") from exc
    _revalidate_config_snapshot(config)
    _revalidate_source(core, "Core source manifest")
    _revalidate_source(task14, "Task 14 index")
    _revalidate_optional(registry_snapshot, "model registry input")
    _revalidate_optional(provenance_snapshot, "provenance input")
    return _publication_from_artifacts(config, actual, checked)


def publish_post_core_release_v1(
    config: PostCoreReleaseConfigV1 | Path | Mapping[str, Any],
    core_manifest_path: Path,
    task14_index_path: Path,
    output_root: Path,
    *,
    registry: Mapping[str, ModelCandidateV1] | Path | None = None,
    provenance_path: Path | None = None,
    before_commit: Callable[[], None] | None = None,
) -> PostCorePublicationV1:
    config = _coerce_config(config)
    core, task14 = _source_pair(core_manifest_path, task14_index_path, config.core_manifest_sha256, config.core_task14_index_sha256)
    registry_value = _coerce_registry(registry, config)
    registry_snapshot = getattr(registry_value, "source_snapshot", None)
    config_snapshot = config.source_snapshot
    provenance_snapshot = _source_snapshot_for_optional(provenance_path, "provenance input")
    source_inputs = tuple(item for item in (core, task14, config_snapshot, registry_snapshot, provenance_snapshot) if item is not None)
    output, parent, parent_identity = _assert_output_safe(output_root, source_inputs)
    artifacts = _build_artifacts(
        config,
        core,
        task14,
        registry_value,
        provenance_path,
        provenance_snapshot=provenance_snapshot,
    )
    publication = _publication_from_artifacts(config, artifacts)
    staging = parent / f".mub-post-core-stage-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    staging_identity = (staging.stat().st_dev, staging.stat().st_ino)
    if staging_identity[0] != parent_identity[0]:
        staging.rmdir()
        raise PostCoreReleaseError("staging directory is not on the output parent filesystem")
    ownership: tuple[_OwnedMember, ...] | None = None
    committed = False
    try:
        _write_staged(staging, publication.artifact_bytes)
        verify_post_core_release_v1(
            staging,
            config,
            core.path,
            task14.path,
            registry=registry_value,
            provenance_path=provenance_path,
            _provenance_snapshot=provenance_snapshot,
        )
        ownership = _capture_owned_members(staging)
        if before_commit is not None:
            before_commit()
        _revalidate_source(core, "Core source manifest")
        _revalidate_source(task14, "Task 14 index")
        _revalidate_optional(config_snapshot, "post-Core config")
        _revalidate_optional(registry_snapshot, "model registry input")
        _revalidate_optional(provenance_snapshot, "provenance input")
        current_parent = parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity or _is_reparse(parent):
            raise PostCoreReleaseError("output parent identity changed before commit")
        if _is_reparse(staging) or (staging.stat().st_dev, staging.stat().st_ino) != staging_identity:
            raise PostCoreReleaseError("staging identity changed before commit")
        _directory_commit_noreplace(staging, output)
        committed = True
        final_metadata = output.stat()
        if (final_metadata.st_dev, final_metadata.st_ino) != staging_identity:
            raise PostCoreReleaseError("committed output identity differs from owned staging")
        _fsync_directory(parent)
        reopened = verify_post_core_release_v1(
            output,
            config,
            core.path,
            task14.path,
            registry=registry_value,
            provenance_path=provenance_path,
            _provenance_snapshot=provenance_snapshot,
        )
        _revalidate_config_snapshot(config)
        _revalidate_source(core, "Core source manifest")
        _revalidate_source(task14, "Task 14 index")
        _revalidate_optional(registry_snapshot, "model registry input")
        _revalidate_optional(provenance_snapshot, "provenance input")
        if reopened.artifact_bytes != publication.artifact_bytes:
            raise PostCoreReleaseError("reopened output differs from staged publication")
        return reopened
    except CommittedPostCoreReleaseError:
        raise
    except StaleSourceError as exc:
        if committed:
            raise CommittedPostCoreReleaseError(output, str(exc)) from exc
        raise
    except (FileExistsError, UnsafePathError):
        raise
    except PostCoreReleaseError as exc:
        if committed:
            raise CommittedPostCoreReleaseError(output, str(exc)) from exc
        raise
    except OSError as exc:
        if committed:
            raise CommittedPostCoreReleaseError(output, str(exc)) from exc
        raise PostCoreReleaseError(f"post-Core publication failed: {exc}") from exc
    finally:
        if not committed:
            _cleanup_owned_staging(staging, staging_identity, ownership)


__all__ = [
    "EXPECTED_REGISTRY_KEYS",
    "EXPECTED_CORE_MANIFEST_SHA256",
    "EXPECTED_TASK14_INDEX_SHA256",
    "EXIT_BLOCKED",
    "EXIT_CONTRACT_USAGE",
    "EXIT_PUBLICATION",
    "EXIT_STALE_SOURCE",
    "EXIT_SUCCESS",
    "EXIT_SUCCESS_WITH_PENDING",
    "EXIT_UNTRUSTED_RUNTIME",
    "EXIT_USAGE",
    "POST_CORE_ARTIFACTS",
    "POST_CORE_INDEX_PATH",
    "PostCorePublicationV1",
    "PostCoreReleaseConfigV1",
    "PostCoreReleaseError",
    "CommittedPostCoreReleaseError",
    "StaleSourceError",
    "UnsafePathError",
    "build_post_core_release_v1",
    "load_post_core_config_v1",
    "load_post_core_registry_v1",
    "publish_post_core_release_v1",
    "verify_post_core_release_v1",
]
