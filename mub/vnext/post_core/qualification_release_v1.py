from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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


def _validate_config_payload(payload: Any, raw: bytes, path: Path | None) -> QualificationReleaseConfigV1:
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
        required_source_sha256=MappingProxyType(dict(required)), config_sha256=_sha256(raw), config_raw=bytes(raw), config_path=path,
    )


def load_qualification_release_config_v1(path: Path) -> QualificationReleaseConfigV1:
    selected = _absolute(Path(path))
    payload, raw = _load_canonical_json(selected, "qualification config")
    return _validate_config_payload(payload, raw, selected)


def _coerce_config(config: QualificationReleaseConfigV1 | Path | Mapping[str, Any]) -> QualificationReleaseConfigV1:
    if isinstance(config, QualificationReleaseConfigV1):
        return _validate_config_payload(json.loads(config.config_raw), config.config_raw, config.config_path)
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
    if tuple(result) != REQUIRED_SOURCE_IDS:
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


QUALIFICATION_ARTIFACTS = (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)

__all__ = [
    "BASE_COMMIT", "QUALIFICATION_ARTIFACTS", "QUALIFICATION_ARTIFACT_ORDER", "QualificationPublicationV1",
    "QualificationReleaseConfigV1", "build_qualification_release_v1", "load_qualification_release_config_v1",
    "verify_qualification_artifact_bytes_v1",
]
