from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.identity_v1 import IdentityEvidenceBundleV1
from mub.vnext.post_core.provenance_v1 import validate_secret_free
from mub.vnext.post_core import (
    qualification_decisions_v1,
    qualification_planning_v1,
    qualification_receipts_v1,
    qualification_validation_v1,
)
from mub.vnext.post_core.qualification_planning_v1 import (
    CapabilitySmokePlanConfigV1,
    _canonical_fixture_bundle_bytes,
    _canonical_parser_contract_bytes,
    _EXPECTED_REGISTRY_KEYS,
    build_capability_budget_v1,
    build_capability_fixtures_v1,
    build_capability_smoke_plan_v1,
)
from mub.vnext.post_core.qualification_decisions_v1 import derive_qualification_decisions_v1
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityBudgetV1,
    CapabilityFixtureV1,
    CapabilitySmokePlanV1,
    DecisionScope,
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
    ProviderSetupEventV1,
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
_SOURCE_SECRET_VALUE = re.compile(
    r"(?:Authorization[ \t]*:[ \t]*(?:Basic|Bearer)[ \t]+\S+|"
    r"Bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|"
    r"(?:OPENAI|ANTHROPIC|GEMINI|GOOGLE|XAI)_API_KEY[ \t]*=[ \t]*[^\s#]+|"
    r"(?:ACCESS|AUTH|REFRESH)_TOKEN[ \t]*=[ \t]*[^\s#]+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----)",
    re.IGNORECASE,
)
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


_SOURCE_PATH_ALIASES = {
    "core_manifest": "core_manifest_path",
    "handoff_source": "handoff_source_path",
    "identity_evidence": "identity_evidence_path",
    "open_snapshot_audit_receipt": "open_snapshot_audit_receipt_path",
    "open_snapshot_closure_receipt": "open_snapshot_closure_receipt_path",
    "phase0_index": "phase0_index_path",
    "qwen_load_receipt": "qwen_load_receipt_path",
    "task14_index": "task14_index_path",
    "workflow_source": "workflow_source_path",
}
_ALLOWED_INPUT_KEYS = frozenset({
    "source_paths", *tuple(_SOURCE_PATH_ALIASES.values()),
    "provider_attestations", "provider_attestations_path",
    "runtime_receipts", "runtime_receipts_path",
    "identity_bundle", "capability_fixtures", "capability_budget",
    "smoke_plan", "decision_bundle", "decisions", "validation_receipt",
})


@dataclass(frozen=True)
class _SourceSnapshot:
    source_id: str
    path: Path
    identity: tuple[int, int]
    byte_count: int
    sha256: str
    stable_times: tuple[int, int]
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


@dataclass(frozen=True)
class _ResolvedQualificationInputs:
    config: QualificationReleaseConfigV1
    source_bundle: SourceBindingBundleV1
    source_snapshots: tuple[_SourceSnapshot, ...]
    providers: tuple[ProviderCapabilityAttestationV1, ...]
    provider_raw: bytes
    provider_snapshot: _SourceSnapshot | None
    runtimes: tuple[OpenRuntimeReceiptV1, ...]
    runtime_raw: bytes
    runtime_snapshot: _SourceSnapshot | None
    identity_bundle: IdentityEvidenceBundleV1 | None
    capability_fixtures: Sequence[CapabilityFixtureV1] | None
    capability_budget: CapabilityBudgetV1 | None

    @property
    def snapshots(self) -> tuple[_SourceSnapshot, ...]:
        return (
            *self.source_snapshots,
            *((self.config.source_snapshot,) if self.config.source_snapshot is not None else ()),
            *((self.provider_snapshot,) if self.provider_snapshot is not None else ()),
            *((self.runtime_snapshot,) if self.runtime_snapshot is not None else ()),
        )


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


def _stable_times(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_mtime_ns, metadata.st_ctime_ns


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
            if (
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size != before.st_size
            ):
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
        or _stable_times(after_open) != _stable_times(opened)
        or _stable_times(after) != _stable_times(before)
        or len(raw) != before.st_size
    ):
        raise ValueError(f"source {source_id} changed while being read")
    try:
        source_text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source {source_id} must use UTF-8 text in qualification v1") from exc
    if _SOURCE_SECRET_VALUE.search(source_text):
        raise ValueError(f"source {source_id} contains secret or credential material")
    return _SourceSnapshot(
        source_id,
        selected,
        (before.st_dev, before.st_ino),
        len(raw),
        _sha256(raw),
        _stable_times(before),
        raw,
    )


def _revalidate_source(snapshot: _SourceSnapshot) -> None:
    current = _read_source(snapshot.source_id, snapshot.path)
    if (current.identity, current.byte_count, current.sha256, current.stable_times) != (
        snapshot.identity, snapshot.byte_count, snapshot.sha256, snapshot.stable_times
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
    if (
        not isinstance(keys, list)
        or tuple(keys) != _EXPECTED_REGISTRY_KEYS
    ):
        raise ValueError("qualification registry tuple/order differs from the frozen planner registry")
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
        if config.config_path is not None and config.source_snapshot is None:
            raise ValueError("file-backed qualification config must retain a matching source snapshot")
        if config.source_snapshot is not None and (
            config.source_snapshot.path != config.config_path
            or config.source_snapshot.raw != config.config_raw
            or config.source_snapshot.sha256 != config.config_sha256
            or config.source_snapshot.byte_count != len(config.config_raw)
        ):
            raise ValueError("qualification config source snapshot does not match canonical config bytes")
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


def _source_binding(
    source_id: str,
    raw: bytes,
    *,
    evidence_class: str,
    required: bool = True,
) -> SourceBindingV1:
    return SourceBindingV1(
        source_id=source_id,
        evidence_class=evidence_class,
        sha256=_sha256(raw),
        required=required,
        byte_count=len(raw),
    )


def _implementation_source_paths() -> dict[str, Path]:
    modules = {
        "qualification_receipts": qualification_receipts_v1,
        "qualification_validation": qualification_validation_v1,
        "qualification_decisions": qualification_decisions_v1,
        "qualification_planner": qualification_planning_v1,
        "qualification_release": None,
    }
    paths: dict[str, Path] = {}
    for source_id, module in modules.items():
        module_file = __file__ if module is None else getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ValueError(f"implementation source path is unavailable: {source_id}")
        paths[source_id] = Path(module_file)
    repository_root = Path(__file__).resolve().parents[3]
    paths["capability_smoke_runner"] = repository_root / "scripts" / "vnext_run_post_core_capability_smoke.py"
    return paths


def _source_bindings(config: QualificationReleaseConfigV1, paths: Mapping[str, Path]) -> tuple[SourceBindingBundleV1, tuple[_SourceSnapshot, ...]]:
    snapshots = tuple(_read_source(source_id, paths[source_id]) for source_id in REQUIRED_SOURCE_IDS)
    for snapshot in snapshots:
        expected = config.required_source_sha256[snapshot.source_id]
        if snapshot.sha256 != expected:
            raise ValueError(f"source {snapshot.source_id} hash differs from the frozen config")
    implementation_snapshots = tuple(
        _read_source(source_id, path)
        for source_id, path in _implementation_source_paths().items()
    )
    bindings = (
        *(
            SourceBindingV1(
                source_id=s.source_id,
                evidence_class=("source_blob" if s.source_id in {"workflow_source", "handoff_source"} else "authenticated_receipt"),
                sha256=s.sha256,
                required=True,
                byte_count=s.byte_count,
            )
            for s in snapshots
        ),
        _source_binding("qualification_config", config.config_raw, evidence_class="qualification_config"),
        _source_binding("capability_fixtures", _canonical_fixture_bundle_bytes(), evidence_class="capability_fixture_bundle"),
        _source_binding("capability_parser_contract", _canonical_parser_contract_bytes(), evidence_class="capability_parser_contract"),
        *(
            SourceBindingV1(
                source_id=s.source_id,
                evidence_class=(
                    "planner_source"
                    if s.source_id == "qualification_planner"
                    else "qualification_implementation_source"
                ),
                sha256=s.sha256,
                required=True,
                byte_count=s.byte_count,
            )
            for s in implementation_snapshots
        ),
    )
    return SourceBindingBundleV1(release_id=config.release_id, sources=bindings), (*snapshots, *implementation_snapshots)


def _jsonl_bytes(rows: Sequence[Any], label: str) -> bytes:
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in rows))
    if not raw:
        raise ValueError(f"{label} must be nonempty")
    return raw


def _rows_from_snapshot(
    snapshot: _SourceSnapshot,
    model_type: type[Any],
    *,
    label: str,
) -> tuple[tuple[Any, ...], bytes]:
    raw = snapshot.raw
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} JSONL must be nonempty and LF-terminated")
    rows: list[Any] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise ValueError(f"{label} JSONL contains an empty row")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} JSONL contains invalid JSON") from exc
        validate_qualification_secret_free(payload)
        try:
            row = model_type.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"{label} JSONL row does not satisfy its contract") from exc
        if canonical_bytes(row) != line:
            raise ValueError(f"{label} JSONL is not canonical")
        rows.append(row)
    return tuple(rows), raw


def _load_provider_rows(path: Path) -> tuple[tuple[ProviderCapabilityAttestationV1, ...], bytes, _SourceSnapshot]:
    snapshot = _read_source("provider attestations", path)
    raw = snapshot.raw
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("provider attestations JSONL must be nonempty and LF-terminated")
    providers: list[ProviderCapabilityAttestationV1] = []
    setup_events: list[ProviderSetupEventV1] = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise ValueError("provider attestations JSONL contains an empty row")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("provider attestations JSONL contains invalid JSON") from exc
        validate_qualification_secret_free(payload)
        try:
            if payload.get("schema_version") == ProviderSetupEventV1.model_fields["schema_version"].default:
                row = ProviderSetupEventV1.model_validate(payload)
                setup_events.append(row)
            else:
                row = ProviderCapabilityAttestationV1.model_validate(payload)
                providers.append(row)
        except Exception as exc:
            raise ValueError("provider attestations JSONL row does not satisfy its contract") from exc
        if canonical_bytes(row) != line:
            raise ValueError("provider attestations JSONL is not canonical")
    if len(setup_events) > 1:
        raise ValueError("provider attestations JSONL contains multiple setup events")
    return tuple(providers), raw, snapshot


def _load_runtime_rows(path: Path) -> tuple[tuple[OpenRuntimeReceiptV1, ...], bytes, _SourceSnapshot]:
    snapshot = _read_source("runtime receipts", path)
    rows, raw = _rows_from_snapshot(snapshot, OpenRuntimeReceiptV1, label="runtime receipts")
    return rows, raw, snapshot


def _resolve_inputs(
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> _ResolvedQualificationInputs:
    unknown = set(inputs) - _ALLOWED_INPUT_KEYS
    if unknown:
        raise TypeError(f"unknown qualification release input: {sorted(unknown)[0]}")
    source_mapping = inputs.get("source_paths")
    if source_mapping is not None:
        if not isinstance(source_mapping, Mapping):
            raise TypeError("source_paths must be a mapping")
        for source_id, alias in _SOURCE_PATH_ALIASES.items():
            if alias in inputs and source_id in source_mapping and inputs[alias] is not None:
                if _absolute(Path(source_mapping[source_id])) != _absolute(Path(inputs[alias])):
                    raise ValueError(f"conflicting source path forms for {source_id}")
    resolved_config = _coerce_config(config)
    paths = _source_map(inputs.get("source_paths"), {
        "core_manifest": inputs.get("core_manifest_path"), "handoff_source": inputs.get("handoff_source_path"),
        "identity_evidence": inputs.get("identity_evidence_path"), "open_snapshot_audit_receipt": inputs.get("open_snapshot_audit_receipt_path"),
        "open_snapshot_closure_receipt": inputs.get("open_snapshot_closure_receipt_path"), "phase0_index": inputs.get("phase0_index_path"),
        "qwen_load_receipt": inputs.get("qwen_load_receipt_path"), "task14_index": inputs.get("task14_index_path"),
        "workflow_source": inputs.get("workflow_source_path"),
    })
    source_bundle, source_snapshots = _source_bindings(resolved_config, paths)
    provider_path = inputs.get("provider_attestations_path")
    if provider_path is None:
        raise ValueError("provider attestations path is required for source-bound release evidence")
    provider_rows, provider_raw, provider_snapshot = _load_provider_rows(Path(provider_path))
    provider_input = inputs.get("provider_attestations")
    if provider_input is not None and _jsonl_bytes(provider_input, "provider attestations") != provider_raw:
        raise ValueError("provider attestations typed rows differ from the source-bound JSONL")

    runtime_path = inputs.get("runtime_receipts_path")
    if runtime_path is None:
        raise ValueError("runtime receipts path is required for source-bound release evidence")
    runtime_rows, runtime_raw, runtime_snapshot = _load_runtime_rows(Path(runtime_path))
    runtime_input = inputs.get("runtime_receipts")
    if runtime_input is not None and _jsonl_bytes(runtime_input, "runtime receipts") != runtime_raw:
        raise ValueError("runtime receipts typed rows differ from the source-bound JSONL")

    identity_snapshot = next(
        snapshot for snapshot in source_snapshots if snapshot.source_id == "identity_evidence"
    )
    try:
        source_identity_bundle = IdentityEvidenceBundleV1.model_validate_json(identity_snapshot.raw)
    except Exception as exc:
        raise ValueError("identity evidence source does not satisfy IdentityEvidenceBundleV1") from exc
    if canonical_bytes(source_identity_bundle) != identity_snapshot.raw:
        raise ValueError("identity evidence source is not canonical")
    compatibility_identity = inputs.get("identity_bundle")
    if compatibility_identity is not None:
        if not isinstance(compatibility_identity, IdentityEvidenceBundleV1):
            raise ValueError("identity bundle compatibility input must use IdentityEvidenceBundleV1")
        if (
            compatibility_identity != source_identity_bundle
            or canonical_bytes(compatibility_identity) != identity_snapshot.raw
        ):
            raise ValueError("identity bundle compatibility input differs from source-bound identity evidence")

    source_bundle = SourceBindingBundleV1(
        release_id=resolved_config.release_id,
        sources=(
            *source_bundle.sources,
            SourceBindingV1(
                source_id="provider_attestations",
                evidence_class="provider_attestation_jsonl",
                sha256=provider_snapshot.sha256,
                required=True,
                byte_count=provider_snapshot.byte_count,
            ),
            SourceBindingV1(
                source_id="runtime_receipts",
                evidence_class="runtime_receipt_jsonl",
                sha256=runtime_snapshot.sha256,
                required=True,
                byte_count=runtime_snapshot.byte_count,
            ),
        ),
    )
    return _ResolvedQualificationInputs(
        config=resolved_config,
        source_bundle=source_bundle,
        source_snapshots=source_snapshots,
        providers=validate_provider_attestations_v1(provider_rows),
        provider_raw=provider_raw,
        provider_snapshot=provider_snapshot,
        runtimes=validate_runtime_receipts_v1(runtime_rows),
        runtime_raw=runtime_raw,
        runtime_snapshot=runtime_snapshot,
        identity_bundle=source_identity_bundle,
        capability_fixtures=inputs.get("capability_fixtures"),
        capability_budget=inputs.get("capability_budget"),
    )


def _revalidate_resolved(resolved: _ResolvedQualificationInputs) -> None:
    for snapshot in resolved.snapshots:
        _revalidate_source(snapshot)


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


def _build_expected_smoke_plan(
    config: QualificationReleaseConfigV1,
    fixtures: Sequence[CapabilityFixtureV1],
    budget: CapabilityBudgetV1,
) -> CapabilitySmokePlanV1:
    return build_capability_smoke_plan_v1(
        CapabilitySmokePlanConfigV1(
            release_id=config.release_id,
            registry_keys=config.registry_keys,
            budget=budget,
            base_attempts_per_role=config.base_attempts_per_role,
            escalation_attempts_per_role=config.escalation_attempts_per_role,
            max_retries=config.max_retries,
            authorized=False,
        ),
        fixtures,
    )


def _derived_decision_bundle(
    config: QualificationReleaseConfigV1,
    identity_bundle: IdentityEvidenceBundleV1 | None,
    providers: Sequence[ProviderCapabilityAttestationV1],
    runtimes: Sequence[OpenRuntimeReceiptV1],
) -> QualificationDecisionBundleV1:
    if not isinstance(identity_bundle, IdentityEvidenceBundleV1):
        raise ValueError("identity bundle is required to derive qualification decisions")
    decisions = derive_qualification_decisions_v1(identity_bundle, providers, runtimes)
    if len(decisions) != len(config.registry_keys) * 4:
        raise ValueError("derived qualification decision count is not exact")
    coordinates = tuple((item.registry_key, item.scope) for item in decisions)
    expected_coordinates = tuple(
        (key, scope)
        for key in config.registry_keys
        for scope in DecisionScope
    )
    if coordinates != expected_coordinates or len(set(coordinates)) != len(coordinates):
        raise ValueError("derived qualification decision coordinates/order are not exact")
    if any(
        item.scope is DecisionScope.BENCHMARK_ADMISSION
        and item.status is not QualificationStatus.BLOCKED
        for item in decisions
    ):
        raise ValueError("benchmark admission decisions must remain BLOCKED")
    return QualificationDecisionBundleV1(release_id=config.release_id, decisions=decisions)


def _validate_evidence_binding_ids(
    source_bundle: SourceBindingBundleV1,
    providers: Sequence[ProviderCapabilityAttestationV1],
    runtimes: Sequence[OpenRuntimeReceiptV1],
    decisions: QualificationDecisionBundleV1,
) -> None:
    source_ids = {source.source_id for source in source_bundle.sources}
    for row in (*providers, *runtimes):
        if any(source_id not in source_ids for source_id in row.source_binding_ids):
            raise ValueError("provider/runtime evidence binding refers to an unknown source")
    for decision in decisions.decisions:
        if any(source_id not in source_ids for source_id in decision.evidence_binding_ids):
            raise ValueError("decision evidence binding refers to an unknown source")


def _validation_receipt(
    config: QualificationReleaseConfigV1,
    decisions: QualificationDecisionBundleV1,
    source_count: int,
) -> QualificationValidationReceiptV1:
    counts: dict[str, int] = {status.value: 0 for status in QualificationStatus}
    for decision in decisions.decisions:
        counts[decision.status.value] += 1
    status = "SUCCESS_WITH_BLOCKERS" if counts[QualificationStatus.BLOCKED.value] or counts[QualificationStatus.UNSUPPORTED.value] else "SUCCESS"
    return QualificationValidationReceiptV1(
        release_id=config.release_id,
        status=status,
        source_count=source_count,
        decision_counts=counts,
    )


def _build_from_resolved(
    resolved: _ResolvedQualificationInputs,
    *,
    smoke_plan: CapabilitySmokePlanV1 | None = None,
    decision_bundle: QualificationDecisionBundleV1 | None = None,
    decisions: Sequence[Any] | None = None,
    validation_receipt: QualificationValidationReceiptV1 | None = None,
) -> QualificationPublicationV1:
    config = resolved.config
    source_bundle = resolved.source_bundle
    providers = resolved.providers
    runtimes = resolved.runtimes
    provider_raw = resolved.provider_raw
    runtime_raw = resolved.runtime_raw
    capability_fixtures = build_capability_fixtures_v1()
    capability_budget = build_capability_budget_v1()
    compatibility_fixtures = resolved.capability_fixtures
    compatibility_budget = resolved.capability_budget
    if compatibility_fixtures is not None or compatibility_budget is not None:
        if compatibility_fixtures is None or type(compatibility_budget) is not CapabilityBudgetV1:
            raise ValueError("capability fixture and budget compatibility inputs must be supplied together")
        _build_expected_smoke_plan(config, compatibility_fixtures, compatibility_budget)
    expected_smoke_plan = _build_expected_smoke_plan(config, capability_fixtures, capability_budget)
    if smoke_plan is not None and smoke_plan != expected_smoke_plan:
        raise ValueError("caller smoke plan differs from the canonical derived plan")
    derived_bundle = _derived_decision_bundle(config, resolved.identity_bundle, providers, runtimes)
    _validate_evidence_binding_ids(source_bundle, providers, runtimes, derived_bundle)
    if decision_bundle is not None and decision_bundle != derived_bundle:
        raise ValueError("caller decision bundle differs from derived qualification decisions")
    if decisions is not None and tuple(decisions) != derived_bundle.decisions:
        raise ValueError("caller decisions differ from derived qualification decisions")
    receipt = _validation_receipt(config, derived_bundle, len(source_bundle.sources))
    if validation_receipt is not None and validation_receipt != receipt:
        raise ValueError("caller validation receipt differs from derived validation receipt")
    manifest = QualificationReleaseManifestV1(
        release_id=config.release_id,
        base_commit=config.base_commit,
        artifact_order=QUALIFICATION_ARTIFACT_ORDER,
        source_hashes={source.source_id: source.sha256 for source in source_bundle.sources},
        source_byte_counts={source.source_id: source.byte_count for source in source_bundle.sources},
    )
    artifacts: dict[str, bytes] = {
        "qualification_release_manifest.json": canonical_bytes(manifest),
        "source_bindings.json": canonical_bytes(source_bundle),
        "provider_capability_attestations.jsonl": provider_raw,
        "open_runtime_receipts.jsonl": runtime_raw,
        "capability_smoke_plan.json": canonical_bytes(expected_smoke_plan),
        "qualification_decisions.json": canonical_bytes(derived_bundle),
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
    _revalidate_resolved(resolved)
    return QualificationPublicationV1(config.release_id, None, MappingProxyType(dict(artifacts)), _sha256(artifacts[QUALIFICATION_INDEX_PATH]))


def build_qualification_release_v1(
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    **inputs: Any,
) -> QualificationPublicationV1:
    resolved = _resolve_inputs(config, inputs)
    publication = _build_from_resolved(
        resolved,
        smoke_plan=inputs.get("smoke_plan"),
        decision_bundle=inputs.get("decision_bundle"),
        decisions=inputs.get("decisions"),
        validation_receipt=inputs.get("validation_receipt"),
    )
    _revalidate_resolved(resolved)
    return publication


def _verify_qualification_artifact_bytes_v1(publication: QualificationPublicationV1) -> QualificationPublicationV1:
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
    raw: bytes


@dataclass(frozen=True)
class _StageOwnership:
    identity: tuple[int, int]
    members: tuple[_StageMember, ...]


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            unsupported = {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL), getattr(errno, "EOPNOTSUPP", errno.EINVAL)}
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _read_member(path: Path, label: str) -> _StageMember:
    selected = _absolute(path)
    _reject_reparse_components(selected)
    before = selected.lstat()
    if _is_reparse(selected) or not _regular_single_link(before):
        raise UnsafeQualificationPathError(f"{label} is not a regular single-link file")
    descriptor = os.open(selected, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not _regular_single_link(opened) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise QualificationReleaseError(f"{label} changed while opened")
        expected_size = opened.st_size
        chunks: list[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _reject_reparse_components(selected)
    after = selected.lstat()
    if (
        len(raw) != expected_size
        or not _regular_single_link(after_open)
        or not _regular_single_link(after)
        or (after_open.st_dev, after_open.st_ino) != (before.st_dev, before.st_ino)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after_open.st_size != expected_size
        or after.st_size != expected_size
    ):
        raise QualificationReleaseError(f"{label} changed while being captured")
    return _StageMember(selected, (before.st_dev, before.st_ino), expected_size, _sha256(raw), raw)


def _capture_member(path: Path) -> _StageMember:
    return _read_member(path, f"publication member {path.name}")


def _capture_stage(stage: Path) -> tuple[_StageMember, ...]:
    if _is_reparse(stage) or not stage.is_dir():
        raise UnsafeQualificationPathError("qualification staging directory is unsafe")
    names = tuple(item.name for item in stage.iterdir())
    if set(names) != set(QUALIFICATION_ARTIFACTS):
        raise QualificationReleaseError("qualification staging artifact set mismatch")
    return tuple(_capture_member(stage / name) for name in QUALIFICATION_ARTIFACTS)


def _stage_matches(members: Sequence[_StageMember]) -> bool:
    try:
        if set(item.path.name for item in members) != set(QUALIFICATION_ARTIFACTS):
            return False
        stage = members[0].path.parent if members else None
        if stage is None or set(item.name for item in stage.iterdir()) != set(QUALIFICATION_ARTIFACTS):
            return False
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


def _prepare_output_root(
    output_root: Path,
    snapshots: Sequence[_SourceSnapshot],
    config: QualificationReleaseConfigV1,
) -> tuple[Path, Path, tuple[int, int]]:
    selected = _absolute(output_root)
    _reject_reparse_components(selected.parent)
    if any(_path_overlap(selected, snapshot.path) for snapshot in snapshots):
        raise UnsafeQualificationPathError("qualification output overlaps a frozen source")
    if config.config_path is not None and _path_overlap(selected, config.config_path):
        raise UnsafeQualificationPathError("qualification output overlaps the release config")
    if selected.is_symlink() or _is_reparse(selected):
        raise UnsafeQualificationPathError("qualification output root is unsafe")
    if selected.exists():
        raise FileExistsError("qualification output root already exists")
    parent = _absolute(selected.parent)
    if not parent.exists() or not parent.is_dir() or _is_reparse(parent):
        raise UnsafeQualificationPathError("qualification output parent must be a safe directory")
    parent_metadata = parent.stat()
    return selected, parent, (parent_metadata.st_dev, parent_metadata.st_ino)


def _write_stage(
    stage: Path,
    artifacts: Mapping[str, bytes],
    tracked: list[_StageMember],
) -> tuple[_StageMember, ...]:
    if _is_reparse(stage) or not stage.is_dir():
        raise UnsafeQualificationPathError("qualification staging directory is unsafe")
    for name in QUALIFICATION_ARTIFACTS:
        target = stage / name
        try:
            with target.open("xb") as stream:
                stream.write(artifacts[name])
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise QualificationReleaseError(f"duplicate qualification staging member: {name}") from exc
        tracked.append(_capture_member(target))
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
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            renameat2.restype = ctypes.c_int
            result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        else:
            if platform.system() != "Linux":
                raise NoReplacePrimitiveUnavailableError(
                    "renameat2 syscall fallback is supported only on Linux"
                )
            syscall = getattr(libc, "syscall", None)
            if syscall is None:
                raise NoReplacePrimitiveUnavailableError("POSIX renameat2 is unavailable")
            syscall_number = {
                "x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276,
            }.get(platform.machine().lower())
            if syscall_number is None:
                raise NoReplacePrimitiveUnavailableError("POSIX renameat2 syscall number is unavailable on this architecture")
            syscall.restype = ctypes.c_long
            result = syscall(syscall_number, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    except NoReplacePrimitiveUnavailableError:
        raise
    except (AttributeError, OSError, TypeError) as exc:
        raise NoReplacePrimitiveUnavailableError("POSIX renameat2 is unavailable") from exc
    if result != 0:
        error = ctypes.get_errno()
        if error == getattr(errno, "EEXIST", 17):
            raise UnsafeQualificationPathError("qualification output root appeared before commit")
        if error in {getattr(errno, "ENOSYS", 38), getattr(errno, "EINVAL", 22)}:
            raise NoReplacePrimitiveUnavailableError("POSIX renameat2 no-replace is unavailable")
        raise QualificationReleaseError(f"POSIX no-replace commit failed: errno {error}")


def _capture_all_artifacts(root: Path, expected: Mapping[str, bytes]) -> Mapping[str, bytes]:
    _reject_reparse_components(root)
    if _is_reparse(root) or not root.is_dir():
        raise UnsafeQualificationPathError("committed qualification root is unsafe")
    if set(item.name for item in root.iterdir()) != set(QUALIFICATION_ARTIFACTS):
        raise QualificationReleaseError("committed qualification artifact set/order mismatch")
    opened: list[tuple[str, Path, os.stat_result, int, os.stat_result]] = []
    try:
        # Keep every member descriptor open before reading any bytes, so the
        # capture observes one coherent, no-follow member set.
        for name in QUALIFICATION_ARTIFACTS:
            path = root / name
            before = path.lstat()
            if _is_reparse(path) or not _regular_single_link(before):
                raise UnsafeQualificationPathError(f"committed qualification artifact is unsafe: {name}")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
            opened_stat = os.fstat(descriptor)
            if (
                not _regular_single_link(opened_stat)
                or (opened_stat.st_dev, opened_stat.st_ino) != (before.st_dev, before.st_ino)
            ):
                os.close(descriptor)
                raise QualificationReleaseError(f"committed qualification artifact changed while opened: {name}")
            opened.append((name, path, before, descriptor, opened_stat))
        captured: list[tuple[str, Path, os.stat_result, int, os.stat_result, bytes]] = []
        for name, path, before, descriptor, opened_stat in opened:
            remaining = opened_stat.st_size + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            captured.append((name, path, before, descriptor, opened_stat, b"".join(chunks)))
        if set(item.name for item in root.iterdir()) != set(QUALIFICATION_ARTIFACTS):
            raise QualificationReleaseError("committed qualification artifact set changed during capture")
        actual: dict[str, bytes] = {}
        for name, path, before, descriptor, opened_stat, raw in captured:
            after_open = os.fstat(descriptor)
            _reject_reparse_components(path)
            after = path.lstat()
            if (
                len(raw) != opened_stat.st_size
                or raw != expected[name]
                or _sha256(raw) != _sha256(expected[name])
                or not _regular_single_link(after_open)
                or not _regular_single_link(after)
                or (after_open.st_dev, after_open.st_ino) != (before.st_dev, before.st_ino)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or after_open.st_size != opened_stat.st_size
                or after.st_size != opened_stat.st_size
                or (after_open.st_mtime_ns, after_open.st_ctime_ns) != (opened_stat.st_mtime_ns, opened_stat.st_ctime_ns)
                or (after.st_mtime_ns, after.st_ctime_ns) != (before.st_mtime_ns, before.st_ctime_ns)
            ):
                raise QualificationReleaseError(f"committed qualification artifact changed during collective capture: {name}")
            actual[name] = raw
        return MappingProxyType(actual)
    finally:
        for _, _, _, descriptor, _ in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_published_root(root: Path, expected: Mapping[str, bytes]) -> Mapping[str, bytes]:
    actual = _capture_all_artifacts(root, expected)
    publication = QualificationPublicationV1(
        RELEASE_ID, root, actual, _sha256(actual[QUALIFICATION_INDEX_PATH])
    )
    return _verify_qualification_artifact_bytes_v1(publication).artifact_bytes


def _cleanup_verified_stage(
    stage: Path,
    stage_identity: tuple[int, int],
    tracked: Sequence[_StageMember],
) -> None:
    if not tracked:
        return
    try:
        metadata = stage.lstat()
        if (
            _is_reparse(stage)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != stage_identity
            or set(item.name for item in stage.iterdir()) != {item.path.name for item in tracked}
        ):
            return
        for item in tracked:
            current = _capture_member(item.path)
            if (current.identity, current.byte_count, current.sha256) != (
                item.identity, item.byte_count, item.sha256
            ):
                return
        for item in tracked:
            item.path.unlink()
        _fsync_directory(stage.parent)
        stage.rmdir()
        _fsync_directory(stage.parent)
    except (OSError, ValueError, QualificationReleaseError, UnsafeQualificationPathError):
        return


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
    resolved = _resolve_inputs(config, inputs)
    output, parent, parent_identity = _prepare_output_root(
        Path(output_root), resolved.snapshots, resolved.config
    )
    publication = _build_from_resolved(
        resolved,
        smoke_plan=inputs.get("smoke_plan"),
        decision_bundle=inputs.get("decision_bundle"),
        decisions=inputs.get("decisions"),
        validation_receipt=inputs.get("validation_receipt"),
    )
    stage = parent / f".mub-post-core-qualification-stage-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    stage_metadata = stage.stat()
    stage_identity = (stage_metadata.st_dev, stage_metadata.st_ino)
    if stage_identity[0] != parent_identity[0] or _is_reparse(stage):
        raise QualificationReleaseError("qualification staging directory is unsafe or crosses filesystems")
    owned: tuple[_StageMember, ...] | None = None
    tracked: list[_StageMember] = []
    committed = False
    try:
        owned = _write_stage(stage, publication.artifact_bytes, tracked)
        if before_commit is not None:
            before_commit()
        _revalidate_resolved(resolved)
        if not _stage_matches(owned):
            raise QualificationReleaseError("qualification staging bytes changed before commit")
        current_parent = parent.stat()
        if _is_reparse(parent) or (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise QualificationReleaseError("qualification output parent identity changed before commit")
        current_stage = stage.stat()
        if _is_reparse(stage) or (current_stage.st_dev, current_stage.st_ino) != stage_identity:
            raise QualificationReleaseError("qualification staging identity changed before commit")
        _rename_noreplace(stage, output)
        committed = True
        final_metadata = output.stat()
        if _is_reparse(output) or (final_metadata.st_dev, final_metadata.st_ino) != stage_identity:
            raise QualificationReleaseError("committed qualification output identity differs from staging")
        try:
            _fsync_directory(parent)
            _read_published_root(output, publication.artifact_bytes)
            _revalidate_resolved(resolved)
            _read_published_root(output, publication.artifact_bytes)
            _revalidate_resolved(resolved)
        except Exception as exc:
            raise CommittedQualificationReleaseError(output, "committed qualification release failed verification") from exc
        return _publication_with_root(publication, output)
    except CommittedQualificationReleaseError:
        raise
    except Exception as exc:
        if committed:
            raise CommittedQualificationReleaseError(output, "committed qualification release failed verification") from exc
        if stage.exists():
            _cleanup_verified_stage(
                stage,
                stage_identity,
                owned if owned is not None else tracked,
            )
        raise


def verify_qualification_release_v1(
    root: Path,
    config: QualificationReleaseConfigV1 | Path | Mapping[str, Any],
    **inputs: Any,
) -> QualificationPublicationV1:
    resolved = _resolve_inputs(config, inputs)
    _reject_reparse_components(Path(root))
    root = _absolute(Path(root))
    if not root.exists() or _is_reparse(root) or not root.is_dir():
        raise UnsafeQualificationPathError("qualification root is absent or unsafe")
    _revalidate_resolved(resolved)
    expected = _build_from_resolved(
        resolved,
        smoke_plan=inputs.get("smoke_plan"),
        decision_bundle=inputs.get("decision_bundle"),
        decisions=inputs.get("decisions"),
        validation_receipt=inputs.get("validation_receipt"),
    )
    try:
        _read_published_root(root, expected.artifact_bytes)
        _revalidate_resolved(resolved)
        _read_published_root(root, expected.artifact_bytes)
        _revalidate_resolved(resolved)
    except Exception as exc:
        raise CommittedQualificationReleaseError(root, "qualification release verification failed") from exc
    return _publication_with_root(expected, root)


QUALIFICATION_ARTIFACTS = (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)


__all__ = [
    "BASE_COMMIT", "QUALIFICATION_ARTIFACTS", "QUALIFICATION_ARTIFACT_ORDER", "QualificationPublicationV1",
    "QualificationReleaseConfigV1", "QualificationReleaseError", "UnsafeQualificationPathError",
    "NoReplacePrimitiveUnavailableError", "CommittedQualificationReleaseError",
    "build_qualification_release_v1", "load_qualification_release_config_v1",
    "publish_qualification_release_v1", "verify_qualification_release_v1",
]
