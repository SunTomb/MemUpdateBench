from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.identity_v1 import IdentityEvidenceBundleV1
from mub.vnext.post_core.provenance_v1 import validate_secret_free
from mub.vnext.post_core.qualification_decisions_v1 import derive_qualification_decisions_v1
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilitySmokePlanV1,
    QualificationArtifactIndexV1,
    QualificationDecisionBundleV1,
    QualificationReleaseManifestV1,
    QualificationStatus,
    QualificationValidationReceiptV1,
    SourceBindingBundleV1,
    SourceBindingV1,
    QUALIFICATION_ARTIFACT_ORDER,
    QUALIFICATION_INDEX_PATH,
    ProviderCapabilityAttestationV1,
    OpenRuntimeReceiptV1,
)
from mub.vnext.post_core.qualification_validation_v1 import (
    load_canonical_jsonl_v1,
    validate_provider_attestations_v1,
    validate_qualification_secret_free,
    validate_runtime_receipts_v1,
)


RELEASE_ID = "memupdatebench.post-core.qualification.v1"
BASE_COMMIT = "a56857431023d2af1a392c75c5575316a916c174"
REQUIRED_SOURCE_IDS = (
    "core_manifest",
    "handoff_source",
    "identity_evidence",
    "open_snapshot_audit_receipt",
    "open_snapshot_closure_receipt",
    "phase0_index",
    "qwen_load_receipt",
    "task14_index",
    "workflow_source",
)
_SHA256 = set("0123456789abcdef")
_METRIC_KEYS = frozenset({
    "em", "f1", "accuracy", "state_accuracy", "stale_copied", "stale_same_slot",
    "memory_size", "benchmark_metric", "metric_name", "metric_value",
})


@dataclass(frozen=True)
class _SourceSnapshot:
    source_id: str
    path: Path
    identity: tuple[int, int]
    byte_count: int
    sha256: str
    raw: bytes


@dataclass(frozen=True)
class QualificationReleaseConfigV1:
    schema_version: str
    release_id: str
    base_commit: str
    registry_keys: tuple[str, ...]
    base_attempts_per_role: int
    escalation_attempts_per_role: int
    max_retries: int
    publisher_network_allowed: bool
    scientific_execution_allowed: bool
    required_source_sha256: Mapping[str, str]
    config_sha256: str
    config_raw: bytes
    config_path: Path | None = None
    source_snapshot: _SourceSnapshot | None = None


@dataclass(frozen=True)
class QualificationPublicationV1:
    release_id: str
    output_root: Path | None
    artifact_bytes: Mapping[str, bytes]
    index_sha256: str
    provider_calls: int = 0
    model_loads: int = 0
    network_calls: int = 0
    credential_reads: int = 0
    benchmark_generations: int = 0


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _absolute(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return Path(os.path.abspath(os.path.normpath(str(selected))))


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
    selected = _absolute(path)
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise ValueError("source path contains a link or reparse component")


def _regular_single_link(metadata: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and not (getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        and getattr(metadata, "st_nlink", 1) == 1
    )


def _read_source(source_id: str, path: Path) -> _SourceSnapshot:
    selected = _absolute(path)
    _reject_reparse_components(selected)
    try:
        before = selected.lstat()
    except OSError as exc:
        raise ValueError(f"source {source_id} does not exist") from exc
    if _is_reparse(selected) or not _regular_single_link(before):
        raise ValueError(f"source {source_id} must be a regular single-link file")
    try:
        descriptor = os.open(selected, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or opened.st_size != before.st_size:
                raise ValueError(f"source {source_id} changed while being read")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = selected.lstat()
    except OSError as exc:
        raise ValueError(f"source {source_id} could not be read safely") from exc
    raw = b"".join(chunks)
    if (
        not _regular_single_link(after_open)
        or not _regular_single_link(after)
        or (after_open.st_dev, after_open.st_ino) != (before.st_dev, before.st_ino)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after_open.st_size != before.st_size
        or after.st_size != before.st_size
        or len(raw) != before.st_size
    ):
        raise ValueError(f"source {source_id} changed while being read")
    return _SourceSnapshot(source_id, selected, (before.st_dev, before.st_ino), len(raw), _sha256(raw), raw)


def _revalidate_source(snapshot: _SourceSnapshot) -> None:
    current = _read_source(snapshot.source_id, snapshot.path)
    if (current.identity, current.byte_count, current.sha256) != (
        snapshot.identity, snapshot.byte_count, snapshot.sha256
    ):
        raise ValueError(f"source {snapshot.source_id} changed after validation")


def _load_canonical_json(path: Path, label: str) -> tuple[Any, bytes]:
    source = _read_source(label, path)
    try:
        payload = json.loads(source.raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    validate_qualification_secret_free(payload)
    if canonical_bytes(payload) != source.raw:
        raise ValueError(f"{label} is not canonical JSON")
    return payload, source.raw


def _validate_config_payload(
    payload: Any,
    raw: bytes,
    path: Path | None,
    *,
    source_snapshot: _SourceSnapshot | None = None,
) -> QualificationReleaseConfigV1:
    if not isinstance(payload, Mapping):
        raise ValueError("qualification config must be a JSON object")
    validate_qualification_secret_free(payload)
    expected_fields = {
        "base_attempts_per_role", "base_commit", "escalation_attempts_per_role", "max_retries",
        "publisher_network_allowed", "registry_keys", "release_id", "required_source_sha256",
        "schema_version", "scientific_execution_allowed",
    }
    if set(payload) != expected_fields:
        raise ValueError("qualification config has unexpected or missing fields")
    if payload["schema_version"] != "memupdatebench.post-core.qualification-config.v1":
        raise ValueError("qualification config schema version mismatch")
    if payload["release_id"] != RELEASE_ID:
        raise ValueError("qualification release ID mismatch")
    if payload["base_commit"] != BASE_COMMIT:
        raise ValueError("qualification base commit mismatch")
    if type(payload["base_attempts_per_role"]) is not int or payload["base_attempts_per_role"] != 8:
        raise ValueError("qualification base attempts must be eight")
    if type(payload["escalation_attempts_per_role"]) is not int or payload["escalation_attempts_per_role"] != 8:
        raise ValueError("qualification escalation attempts must be eight")
    if type(payload["max_retries"]) is not int or payload["max_retries"] != 0:
        raise ValueError("qualification retries must be zero")
    if payload["publisher_network_allowed"] is not False or type(payload["publisher_network_allowed"]) is not bool:
        raise ValueError("publisher network access must be disabled")
    if payload["scientific_execution_allowed"] is not False or type(payload["scientific_execution_allowed"]) is not bool:
        raise ValueError("scientific execution must be disabled")
    keys = payload["registry_keys"]
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("qualification registry tuple is invalid")
    required = payload["required_source_sha256"]
    if not isinstance(required, Mapping) or tuple(required) != REQUIRED_SOURCE_IDS:
        raise ValueError("qualification source hash mapping is incomplete or out of order")
    if any(not _is_sha256(value) for value in required.values()):
        raise ValueError("qualification source hashes must be lowercase SHA-256")
    if canonical_bytes(payload) != raw:
        raise ValueError("qualification config must use canonical JSON bytes")
    return QualificationReleaseConfigV1(
        schema_version=payload["schema_version"], release_id=payload["release_id"], base_commit=payload["base_commit"],
        registry_keys=tuple(keys), base_attempts_per_role=payload["base_attempts_per_role"],
        escalation_attempts_per_role=payload["escalation_attempts_per_role"], max_retries=payload["max_retries"],
        publisher_network_allowed=False, scientific_execution_allowed=False,
        required_source_sha256=MappingProxyType(dict(required)), config_sha256=_sha256(raw), config_raw=bytes(raw),
        config_path=path, source_snapshot=source_snapshot,
    )


def load_qualification_release_config_v1(path_or_mapping: Path | str | Mapping[str, Any]) -> QualificationReleaseConfigV1:
    if isinstance(path_or_mapping, (str, Path)):
        selected = _absolute(Path(path_or_mapping))
        source = _read_source("qualification config", selected)
        try:
            payload = json.loads(source.raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("qualification config is not valid JSON") from exc
        return _validate_config_payload(payload, source.raw, selected, source_snapshot=source)
    raw = canonical_bytes(path_or_mapping)
    return _validate_config_payload(path_or_mapping, raw, None)


def _coerce_config(config: QualificationReleaseConfigV1 | Path | Mapping[str, Any]) -> QualificationReleaseConfigV1:
    if isinstance(config, QualificationReleaseConfigV1):
        return _validate_config_payload(
            json.loads(config.config_raw), config.config_raw, config.config_path,
            source_snapshot=config.source_snapshot,
        )
    if isinstance(config, (str, Path)):
        return load_qualification_release_config_v1(Path(config))
    raw = canonical_bytes(config)
    return _validate_config_payload(config, raw, None)


def _source_map(
    source_paths: Mapping[str, Path] | None,
    explicit: Mapping[str, Path | None],
) -> dict[str, Path]:
    result = dict(source_paths or {})
    result.update({key: value for key, value in explicit.items() if value is not None})
    if set(result) != set(REQUIRED_SOURCE_IDS):
        raise ValueError("qualification source paths must contain the exact nine source IDs")
    return {key: Path(result[key]) for key in REQUIRED_SOURCE_IDS}


def _source_bindings(config: QualificationReleaseConfigV1, paths: Mapping[str, Path]) -> tuple[SourceBindingBundleV1, tuple[_SourceSnapshot, ...]]:
    snapshots = tuple(_read_source(source_id, paths[source_id]) for source_id in REQUIRED_SOURCE_IDS)
    for snapshot in snapshots:
        expected = config.required_source_sha256[snapshot.source_id]
        if snapshot.sha256 != expected:
            raise ValueError(f"source {snapshot.source_id} hash differs from the frozen config")
    bindings = tuple(
        SourceBindingV1(source_id=s.source_id, evidence_class=("source_blob" if s.source_id in {"workflow_source", "handoff_source"} else "authenticated_receipt"), sha256=s.sha256, required=True, byte_count=s.byte_count)
        for s in snapshots
    )
    return SourceBindingBundleV1(release_id=config.release_id, sources=bindings), snapshots


def _jsonl_bytes(rows: Sequence[Any], label: str) -> bytes:
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in rows))
    if not raw:
        raise ValueError(f"{label} must be nonempty")
    return raw


def _load_provider_rows(path: Path) -> tuple[tuple[ProviderCapabilityAttestationV1, ...], bytes]:
    return load_canonical_jsonl_v1(path, ProviderCapabilityAttestationV1, label="provider attestations")


def _load_runtime_rows(path: Path) -> tuple[tuple[OpenRuntimeReceiptV1, ...], bytes]:
    return load_canonical_jsonl_v1(path, OpenRuntimeReceiptV1, label="runtime receipts")


_SAFE_COUNTER_KEYS = frozenset({
    "provider_calls_during_publication", "model_loads_during_publication",
    "network_calls_during_publication", "credential_reads_during_publication",
    "benchmark_generations",
})


def _secret_scan_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _secret_scan_payload(item)
            for key, item in value.items()
            if str(key) not in _SAFE_COUNTER_KEYS
        }
    if isinstance(value, (tuple, list)):
        return tuple(_secret_scan_payload(item) for item in value)
    return value


def _reject_metrics(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _METRIC_KEYS:
                raise ValueError("qualification artifacts may not contain benchmark metric fields")
            _reject_metrics(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_metrics(item)


def _validate_smoke_plan(config: QualificationReleaseConfigV1, plan: CapabilitySmokePlanV1) -> None:
    if plan.release_id != config.release_id or tuple(plan.registry_keys) != config.registry_keys:
        raise ValueError("qualification smoke plan release or registry binding differs")
    if plan.base_attempts_per_role != 8 or plan.escalation_attempts_per_role != 8 or plan.max_retries != 0 or plan.authorized is not False:
        raise ValueError("qualification smoke plan budget differs from frozen config")
    if any(item.authorized or item.executable or item.budget.max_retries != 0 for item in plan.attempts):
        raise ValueError("qualification smoke plan may not authorize execution")


def _decision_bundle(config: QualificationReleaseConfigV1, decisions: QualificationDecisionBundleV1 | Sequence[Any]) -> QualificationDecisionBundleV1:
    bundle = decisions if isinstance(decisions, QualificationDecisionBundleV1) else QualificationDecisionBundleV1(release_id=config.release_id, decisions=tuple(decisions))
    if bundle.release_id != config.release_id or not bundle.decisions:
        raise ValueError("qualification decision bundle binding is invalid")
    if any(item.scientific_status != "NOT_RUN" for item in bundle.decisions):
        raise ValueError("scientific qualification execution must remain NOT_RUN")
    return bundle


def _validation_receipt(config: QualificationReleaseConfigV1, receipt: QualificationValidationReceiptV1 | None, decisions: QualificationDecisionBundleV1, source_count: int) -> QualificationValidationReceiptV1:
    counts: dict[str, int] = {status.value: 0 for status in QualificationStatus}
    for decision in decisions.decisions:
        counts[decision.status.value] += 1
    if receipt is None:
        receipt = QualificationValidationReceiptV1(release_id=config.release_id, status="SUCCESS_WITH_BLOCKERS", source_count=source_count, decision_counts=counts)
    if receipt.release_id != config.release_id or receipt.source_count != source_count:
        raise ValueError("qualification validation receipt binding differs")
    if any(getattr(receipt, field) != 0 for field in ("provider_calls_during_publication", "model_loads_during_publication", "network_calls_during_publication", "credential_reads_during_publication", "benchmark_generations")):
        raise ValueError("qualification publication counters must remain zero")
    return receipt


def build_qualification_release_v1(
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    *,
    source_paths: Mapping[str, Path] | None = None,
    core_manifest_path: Path | None = None,
    handoff_source_path: Path | None = None,
    identity_evidence_path: Path | None = None,
    open_snapshot_audit_receipt_path: Path | None = None,
    open_snapshot_closure_receipt_path: Path | None = None,
    phase0_index_path: Path | None = None,
    qwen_load_receipt_path: Path | None = None,
    task14_index_path: Path | None = None,
    workflow_source_path: Path | None = None,
    provider_attestations: Sequence[ProviderCapabilityAttestationV1] | None = None,
    provider_attestations_path: Path | None = None,
    runtime_receipts: Sequence[OpenRuntimeReceiptV1] | None = None,
    runtime_receipts_path: Path | None = None,
    smoke_plan: CapabilitySmokePlanV1 | None = None,
    decision_bundle: QualificationDecisionBundleV1 | None = None,
    decisions: Sequence[Any] | None = None,
    validation_receipt: QualificationValidationReceiptV1 | None = None,
    identity_bundle: IdentityEvidenceBundleV1 | None = None,
) -> QualificationPublicationV1:
    config = _coerce_config(config)
    paths = _source_map(source_paths, {
        "core_manifest": core_manifest_path, "handoff_source": handoff_source_path,
        "identity_evidence": identity_evidence_path, "open_snapshot_audit_receipt": open_snapshot_audit_receipt_path,
        "open_snapshot_closure_receipt": open_snapshot_closure_receipt_path, "phase0_index": phase0_index_path,
        "qwen_load_receipt": qwen_load_receipt_path, "task14_index": task14_index_path, "workflow_source": workflow_source_path,
    })
    source_bundle, snapshots = _source_bindings(config, paths)
    if provider_attestations is None:
        if provider_attestations_path is None:
            raise ValueError("provider attestations must be supplied as typed rows or a JSONL source")
        provider_attestations, provider_raw = _load_provider_rows(Path(provider_attestations_path))
    else:
        provider_raw = _jsonl_bytes(provider_attestations, "provider attestations")
    providers = validate_provider_attestations_v1(provider_attestations)
    if runtime_receipts is None:
        if runtime_receipts_path is None:
            raise ValueError("runtime receipts must be supplied as typed rows or a JSONL source")
        runtime_receipts, runtime_raw = _load_runtime_rows(Path(runtime_receipts_path))
    else:
        runtime_raw = _jsonl_bytes(runtime_receipts, "runtime receipts")
    runtimes = validate_runtime_receipts_v1(runtime_receipts)
    if smoke_plan is None:
        raise ValueError("capability smoke plan must be supplied as typed input")
    _validate_smoke_plan(config, smoke_plan)
    if decision_bundle is None:
        if decisions is not None:
            decision_bundle = _decision_bundle(config, decisions)
        elif identity_bundle is not None:
            decision_bundle = QualificationDecisionBundleV1(release_id=config.release_id, decisions=derive_qualification_decisions_v1(identity_bundle, providers, runtimes))
        else:
            raise ValueError("qualification decisions must be supplied as typed input")
    decision_bundle = _decision_bundle(config, decision_bundle)
    receipt = _validation_receipt(config, validation_receipt, decision_bundle, len(source_bundle.sources))
    manifest = QualificationReleaseManifestV1(release_id=config.release_id, base_commit=config.base_commit, artifact_order=QUALIFICATION_ARTIFACT_ORDER, source_hashes=config.required_source_sha256)
    artifacts: dict[str, bytes] = {
        "qualification_release_manifest.json": canonical_bytes(manifest),
        "source_bindings.json": canonical_bytes(source_bundle),
        "provider_capability_attestations.jsonl": provider_raw,
        "open_runtime_receipts.jsonl": runtime_raw,
        "capability_smoke_plan.json": canonical_bytes(smoke_plan),
        "qualification_decisions.json": canonical_bytes(decision_bundle),
        "qualification_validation_receipt.json": canonical_bytes(receipt),
    }
    for name, raw in artifacts.items():
        if name.endswith(".jsonl"):
            for line in raw.splitlines():
                _reject_metrics(json.loads(line))
        else:
            payload = json.loads(raw)
            validate_qualification_secret_free(_secret_scan_payload(payload))
            _reject_metrics(payload)
    index = QualificationArtifactIndexV1(release_id=config.release_id, artifacts=tuple({"path": name, "sha256": _sha256(artifacts[name])} for name in QUALIFICATION_ARTIFACT_ORDER))
    artifacts[QUALIFICATION_INDEX_PATH] = canonical_bytes(index, exclude={"canonical_hash"})
    validate_qualification_secret_free(_secret_scan_payload(index.model_dump(mode="json")))
    _reject_metrics(json.loads(artifacts[QUALIFICATION_INDEX_PATH]))
    return QualificationPublicationV1(config.release_id, None, MappingProxyType(dict(artifacts)), _sha256(artifacts[QUALIFICATION_INDEX_PATH]))


def verify_qualification_artifact_bytes_v1(publication: QualificationPublicationV1) -> QualificationPublicationV1:
    if tuple(publication.artifact_bytes) != tuple(QUALIFICATION_ARTIFACTS):
        raise ValueError("qualification artifact set/order mismatch")
    index = QualificationArtifactIndexV1.model_validate_json(publication.artifact_bytes[QUALIFICATION_INDEX_PATH])
    if canonical_bytes(index, exclude={"canonical_hash"}) != publication.artifact_bytes[QUALIFICATION_INDEX_PATH]:
        raise ValueError("qualification artifact index is not canonical")
    for binding in index.artifacts:
        raw = publication.artifact_bytes[binding.path]
        if _sha256(raw) != binding.sha256:
            raise ValueError(f"qualification artifact hash mismatch: {binding.path}")
    return publication


class QualificationReleaseError(RuntimeError):
    """Base class for qualification publication errors."""


class UnsafeQualificationPathError(QualificationReleaseError, ValueError):
    """A source, staging, or output path is unsafe."""


class NoReplacePrimitiveUnavailableError(QualificationReleaseError):
    """The host does not expose an atomic no-replace rename primitive."""


class CommittedQualificationReleaseError(QualificationReleaseError):
    """The output root was committed but failed post-commit verification."""

    def __init__(self, committed_root: Path, message: str) -> None:
        self.committed_root = Path(committed_root)
        super().__init__(f"{message}; committed root preserved at {self.committed_root}")


@dataclass(frozen=True)
class _StageMember:
    path: Path
    identity: tuple[int, int]
    byte_count: int
    sha256: str


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _capture_member(path: Path) -> _StageMember:
    metadata = path.lstat()
    if not _regular_single_link(metadata):
        raise UnsafeQualificationPathError(f"publication member is not a regular single-link file: {path.name}")
    raw = path.read_bytes()
    after = path.lstat()
    if (
        not _regular_single_link(after)
        or (metadata.st_dev, metadata.st_ino) != (after.st_dev, after.st_ino)
        or metadata.st_size != after.st_size
        or len(raw) != metadata.st_size
    ):
        raise QualificationReleaseError(f"publication member changed while captured: {path.name}")
    return _StageMember(path, (metadata.st_dev, metadata.st_ino), metadata.st_size, _sha256(raw))


def _capture_stage(stage: Path) -> tuple[_StageMember, ...]:
    if _is_reparse(stage) or not stage.is_dir():
        raise UnsafeQualificationPathError("qualification staging directory is unsafe")
    names = tuple(item.name for item in stage.iterdir())
    if set(names) != set(QUALIFICATION_ARTIFACTS):
        raise QualificationReleaseError("qualification staging artifact set mismatch")
    return tuple(_capture_member(stage / name) for name in QUALIFICATION_ARTIFACTS)


def _stage_matches(members: Sequence[_StageMember]) -> bool:
    try:
        for item in members:
            current = _capture_member(item.path)
            if (current.identity, current.byte_count, current.sha256) != (item.identity, item.byte_count, item.sha256):
                return False
        return True
    except (OSError, ValueError, QualificationReleaseError):
        return False


def _path_overlap(left: Path, right: Path) -> bool:
    left = _absolute(left)
    right = _absolute(right)
    try:
        return os.path.commonpath((str(left), str(right))) in {str(left), str(right)}
    except ValueError:
        return False


def _prepare_output_root(output_root: Path, snapshots: Sequence[_SourceSnapshot], config: QualificationReleaseConfigV1) -> Path:
    selected = _absolute(output_root)
    _reject_reparse_components(selected.parent)
    if any(_path_overlap(selected, snapshot.path) for snapshot in snapshots):
        raise UnsafeQualificationPathError("qualification output overlaps a frozen source")
    if config.config_path is not None and _path_overlap(selected, config.config_path):
        raise UnsafeQualificationPathError("qualification output overlaps the release config")
    if selected.exists() or selected.is_symlink() or _is_reparse(selected):
        raise UnsafeQualificationPathError("qualification output root must be absent")
    if not selected.parent.exists() or not selected.parent.is_dir() or _is_reparse(selected.parent):
        raise UnsafeQualificationPathError("qualification output parent must be a safe directory")
    return selected


def _write_stage(stage: Path, artifacts: Mapping[str, bytes]) -> tuple[_StageMember, ...]:
    stage.mkdir()
    for name in QUALIFICATION_ARTIFACTS:
        target = stage / name
        try:
            with target.open("xb") as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise QualificationReleaseError(f"duplicate qualification staging member: {name}") from exc
    _fsync_directory(stage)
    return _capture_stage(stage)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        try:
            move_file = ctypes.windll.kernel32.MoveFileExW
            move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
            move_file.restype = ctypes.c_int
        except (AttributeError, NameError) as exc:
            raise NoReplacePrimitiveUnavailableError("Windows MoveFileExW is unavailable") from exc
        if destination.exists() or destination.is_symlink():
            raise UnsafeQualificationPathError("qualification output root appeared before commit")
        # The destination is checked absent immediately before MoveFileExW; the
        # WRITE_THROUGH flag makes the successful same-volume move durable.
        if not move_file(str(source), str(destination), 0x00000008):
            raise QualificationReleaseError("Windows MoveFileExW no-replace commit failed")
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
    except (AttributeError, OSError) as exc:
        raise NoReplacePrimitiveUnavailableError("POSIX renameat2 is unavailable") from exc
    # Linux exposes renameat2 through a raw syscall; unsupported architectures
    # fail closed rather than falling back to a replace-capable operation.
    syscall_number = {
        "x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276,
    }.get(platform.machine().lower())
    if syscall_number is None:
        raise NoReplacePrimitiveUnavailableError("POSIX renameat2 syscall number is unavailable on this architecture")
    syscall.restype = ctypes.c_long
    result = syscall(syscall_number, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == getattr(errno, "EEXIST", 17):
            raise UnsafeQualificationPathError("qualification output root appeared before commit")
        if error in {getattr(errno, "ENOSYS", 38), getattr(errno, "EINVAL", 22)}:
            raise NoReplacePrimitiveUnavailableError("POSIX renameat2 no-replace is unavailable")
        raise QualificationReleaseError(f"POSIX no-replace commit failed: errno {error}")


def _read_published_root(root: Path, expected: Mapping[str, bytes]) -> Mapping[str, bytes]:
    if _is_reparse(root) or not root.is_dir():
        raise UnsafeQualificationPathError("committed qualification root is unsafe")
    if set(item.name for item in root.iterdir()) != set(QUALIFICATION_ARTIFACTS):
        raise QualificationReleaseError("committed qualification artifact set/order mismatch")
    actual: dict[str, bytes] = {}
    for name in QUALIFICATION_ARTIFACTS:
        path = root / name
        member = _capture_member(path)
        raw = path.read_bytes()
        if raw != expected[name] or member.sha256 != _sha256(expected[name]):
            raise QualificationReleaseError(f"committed qualification artifact mismatch: {name}")
        actual[name] = raw
    publication = QualificationPublicationV1(
        RELEASE_ID, root, MappingProxyType(actual), _sha256(actual[QUALIFICATION_INDEX_PATH])
    )
    return verify_qualification_artifact_bytes_v1(publication).artifact_bytes


def _publication_with_root(publication: QualificationPublicationV1, root: Path) -> QualificationPublicationV1:
    return QualificationPublicationV1(
        publication.release_id, root, publication.artifact_bytes, publication.index_sha256,
        publication.provider_calls, publication.model_loads, publication.network_calls,
        publication.credential_reads, publication.benchmark_generations,
    )


def publish_qualification_release_v1(
    output_root: Path,
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    *,
    before_commit: Any | None = None,
    **inputs: Any,
) -> QualificationPublicationV1:
    config_obj = _coerce_config(config)
    paths = _source_map(inputs.get("source_paths"), {
        "core_manifest": inputs.get("core_manifest_path"), "handoff_source": inputs.get("handoff_source_path"),
        "identity_evidence": inputs.get("identity_evidence_path"), "open_snapshot_audit_receipt": inputs.get("open_snapshot_audit_receipt_path"),
        "open_snapshot_closure_receipt": inputs.get("open_snapshot_closure_receipt_path"), "phase0_index": inputs.get("phase0_index_path"),
        "qwen_load_receipt": inputs.get("qwen_load_receipt_path"), "task14_index": inputs.get("task14_index_path"),
        "workflow_source": inputs.get("workflow_source_path"),
    })
    _, snapshots = _source_bindings(config_obj, paths)
    source_snapshots = (*snapshots, *((config_obj.source_snapshot,) if config_obj.source_snapshot is not None else ()))
    output = _prepare_output_root(Path(output_root), source_snapshots, config_obj)
    publication = build_qualification_release_v1(config_obj, **inputs)
    parent = output.parent
    stage = parent / f".mub-post-core-qualification-stage-{uuid.uuid4().hex}"
    owned: tuple[_StageMember, ...] = ()
    committed = False
    try:
        owned = _write_stage(stage, publication.artifact_bytes)
        if before_commit is not None:
            before_commit(stage)
        for snapshot in source_snapshots:
            _revalidate_source(snapshot)
        if not _stage_matches(owned):
            raise QualificationReleaseError("qualification staging bytes changed before commit")
        _rename_noreplace(stage, output)
        committed = True
        _fsync_directory(parent)
        try:
            _read_published_root(output, publication.artifact_bytes)
        except Exception as exc:
            raise CommittedQualificationReleaseError(output, "committed qualification release failed reopening") from exc
        return _publication_with_root(publication, output)
    except CommittedQualificationReleaseError:
        raise
    except Exception:
        if not committed and stage.exists() and _stage_matches(owned):
            import shutil
            shutil.rmtree(stage)
        raise


def verify_qualification_release_v1(
    root: Path,
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    **inputs: Any,
) -> QualificationPublicationV1:
    config_obj = _coerce_config(config)
    paths = _source_map(inputs.get("source_paths"), {
        "core_manifest": inputs.get("core_manifest_path"), "handoff_source": inputs.get("handoff_source_path"),
        "identity_evidence": inputs.get("identity_evidence_path"), "open_snapshot_audit_receipt": inputs.get("open_snapshot_audit_receipt_path"),
        "open_snapshot_closure_receipt": inputs.get("open_snapshot_closure_receipt_path"), "phase0_index": inputs.get("phase0_index_path"),
        "qwen_load_receipt": inputs.get("qwen_load_receipt_path"), "task14_index": inputs.get("task14_index_path"),
        "workflow_source": inputs.get("workflow_source_path"),
    })
    _, snapshots = _source_bindings(config_obj, paths)
    root = _absolute(Path(root))
    if not root.exists() or _is_reparse(root) or not root.is_dir():
        raise UnsafeQualificationPathError("qualification root is absent or unsafe")
    for snapshot in (*snapshots, *((config_obj.source_snapshot,) if config_obj.source_snapshot is not None else ())):
        _revalidate_source(snapshot)
    expected = build_qualification_release_v1(config_obj, **inputs)
    try:
        _read_published_root(root, expected.artifact_bytes)
    except Exception as exc:
        raise CommittedQualificationReleaseError(root, "qualification release verification failed") from exc
    return _publication_with_root(expected, root)


QUALIFICATION_ARTIFACTS = (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)


__all__ = [
    "BASE_COMMIT", "QUALIFICATION_ARTIFACTS", "QUALIFICATION_ARTIFACT_ORDER", "QualificationPublicationV1",
    "QualificationReleaseConfigV1", "QualificationReleaseError", "UnsafeQualificationPathError",
    "NoReplacePrimitiveUnavailableError", "CommittedQualificationReleaseError",
    "build_qualification_release_v1", "load_qualification_release_config_v1",
    "publish_qualification_release_v1", "verify_qualification_release_v1", "verify_qualification_artifact_bytes_v1",
]
