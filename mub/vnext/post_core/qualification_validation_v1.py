from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from mub.vnext.post_core.contracts_v1 import canonical_bytes, canonical_hash
from mub.vnext.post_core.qualification_planning_v1 import (
    CapabilitySmokePlanConfigV1,
    _EXPECTED_REGISTRY_KEYS,
    _EXPECTED_RELEASE_ID,
    build_capability_budget_v1,
    build_capability_fixtures_v1,
    build_capability_smoke_plan_v1,
)
from mub.vnext.post_core.provenance_v1 import validate_secret_free
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAnomalyReceiptV1,
    CapabilityAttemptPlanV1,
    CapabilityAttemptReceiptV1,
    CapabilitySmokePlanV1,
    ExecutionAuthorizationV1,
    GateStatus,
    OpenRuntimeReceiptV1,
    ProviderCapabilityAttestationV1,
    RuntimeManifestV1,
)


_EXPECTED_CAPABILITY_RESPONSE_MODELS = {
    "claude_sonnet_4_6": "claude-sonnet-4-6",
    "claude_opus_4_8": "claude-opus-4-8",
    "gemini_3_6_flash": "Gemini 3.6 Flash (Low)",
    "grok_4_5": "grok-4.5",
    "gpt_5_5": "gpt-5.5",
}
_EXPECTED_PROVIDER_ROWS = (
    ("claude_sonnet_4_6", "claude-sonnet-4-6", 2),
    ("claude_opus_4_8", "claude-opus-4-8", 2),
    ("gemini_3_6_flash", "Gemini 3.6 Flash (Low)", 2),
    ("grok_4_5", "grok-4.5", 2),
    ("gpt_5_5", "gpt-5.5", 4),
)
_EXPECTED_RUNTIME_ROWS = (
    (
        "qwen35_9b_bf16",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "transformers",
        "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db",
        ("open_snapshot_closure_receipt", "qwen_load_receipt"),
    ),
    (
        "meta_muse_glimmer_30b_int4",
        "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
        "llama.cpp",
        "55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f",
        ("open_snapshot_closure_receipt",),
    ),
    (
        "meta_muse_glimmer_30b_bf16",
        "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "transformers",
        "7a90420d22f8c98737f15bc31473bbe8a3579ee95f9bf2237172679709877782",
        ("open_snapshot_closure_receipt",),
    ),
)
_RUNTIME_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_EVIDENCE_FIELDS = (
    "prompt_fixture_sha256",
    "parser_sha256",
    "chat_template_sha256",
    "output_projection_sha256",
    "generated_token_count",
    "peak_memory_bytes",
)
_QWEN_PACKAGE_FIELDS = (
    "python_version",
    "torch_version",
    "transformers_version",
    "accelerate_version",
    "cuda_version",
    "driver_version",
)
_EXPECTED_SOURCE_BINDINGS = ("workflow_source", "handoff_source")
_EXPECTED_IDENTITY_METADATA = {
    "claude_sonnet_4_6": ("claude-sonnet-4-6", None, None),
    "claude_opus_4_8": ("claude-opus-4-8", None, None),
    "gemini_3_6_flash": ("gemini-3.6-flash", "Low", None),
    "grok_4_5": (None, None, "explicitly mutable transfer alias"),
    "gpt_5_5": (None, None, "unverified official upstream identity"),
}
_GPT_OBSERVATION_IDS = (
    "LOCAL_INITIAL_SSE",
    "LOCAL_EXPLICIT_FALSE_SSE",
    "TANG2_PREFIX_SSE",
    "TANG2_POSTFIX_JSON",
)
_GPT_RESPONSE_FORMATS = ("SSE", "SSE", "SSE", "ANTHROPIC_MESSAGE_JSON")
_URL_KEYS = frozenset({"endpoint", "endpoint_url", "source_url"})
_SENSITIVE_IDENTIFIER_ATOMS = frozenset(
    {"auth", "authorization", "bearer", "credential", "password", "secret", "token"}
)
_SENSITIVE_IDENTIFIER_COMPOUNDS = frozenset(
    {
        "api_key", "x_api_key", "private_key", "access_token", "refresh_token", "auth_token",
        "x_auth_token", "x_access_token", "x_api_token", "x_goog_api_key",
        "x_amz_security_token", "openai_api_key", "anthropic_api_key", "gemini_api_key",
        "google_api_key", "xai_api_key",
    }
)
_SENSITIVE_COMPACT_IDENTIFIERS = frozenset(
    {item.replace("_", "") for item in _SENSITIVE_IDENTIFIER_COMPOUNDS}
    | {"apikey", "xapikey", "awsaccesskeyid", "awssecretaccesskey", "gcpserviceaccountkey"}
)
_SENSITIVE_PART_SEQUENCES = (
    ("private", "key"),
    ("access", "key"),
    ("secret", "access", "key"),
    ("service", "account", "key"),
)
_CREDENTIAL_PROVIDER_PREFIXES = frozenset({"aws", "gcp", "google", "azure", "amz"})
_QUERY_GENERIC_SENSITIVE_PARTS = frozenset({"key", "private", "sig", "signature"})
_PRIVATE_KEY_BLOCK = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^\s*([^=]+?)\s*=", re.DOTALL)
_IDENTIFIER_SHAPED = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SAFE_CONTRACT_KEYS = frozenset({
    "registry_key",
    "generated_token_count",
    "authorization_attestation_sha256",
    "escalation_anomaly_receipt_sha256",
})


def _normalized_key(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[-\s]+", "_", text).lower()


def _contains_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    return any(parts[index : index + len(sequence)] == sequence for index in range(len(parts)))


def _is_provider_prefixed_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    for index in range(len(parts) - len(sequence) + 1):
        if parts[index : index + len(sequence)] == sequence and any(
            part in _CREDENTIAL_PROVIDER_PREFIXES for part in parts[:index]
        ):
            return True
    return False


def _is_sensitive_identifier(value: object, *, query_key: bool = False) -> bool:
    normalized = _normalized_key(value)
    if normalized == "credential_env_var" or normalized in _SAFE_CONTRACT_KEYS:
        return False
    parts = tuple(part for part in normalized.split("_") if part)
    compact = normalized.replace("_", "")
    if (
        normalized in _SENSITIVE_IDENTIFIER_ATOMS
        or normalized in _SENSITIVE_IDENTIFIER_COMPOUNDS
        or compact in _SENSITIVE_COMPACT_IDENTIFIERS
        or any(part in _SENSITIVE_IDENTIFIER_ATOMS for part in parts)
        or _contains_sequence(parts, ("private", "key"))
        or any(
            _is_provider_prefixed_sequence(parts, sequence)
            for sequence in _SENSITIVE_PART_SEQUENCES[1:]
        )
    ):
        return True
    return query_key and any(part in _QUERY_GENERIC_SENSITIVE_PARTS for part in parts)


def _validate_host(host: str) -> None:
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("endpoint/source URL host is invalid") from exc
    labels = ascii_host.lower().split(".")
    if not labels or len(ascii_host.encode("ascii")) > 253 or any(not _HOST_LABEL.fullmatch(label) for label in labels):
        raise ValueError("endpoint/source URL host is invalid")


def _strict_unquote(value: str) -> str:
    try:
        return unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("URL query contains invalid percent encoding") from exc


def _scan_decoded_query_value(value: str) -> None:
    if _PRIVATE_KEY_BLOCK.search(value):
        raise ValueError("URL query contains sensitive material")
    assignment = _ASSIGNMENT.match(value)
    if assignment and _is_sensitive_identifier(assignment.group(1)):
        raise ValueError("URL query contains sensitive material")


def _validate_url(value: object, key: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an HTTPS URL")
    if any(char.isspace() or ord(char) < 32 for char in value) or _BAD_PERCENT_ESCAPE.search(value):
        raise ValueError(f"{key} must be a well-formed HTTPS URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{key} must be a well-formed HTTPS URL") from exc
    if parsed.scheme != "https" or not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"{key} must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError(f"{key} may not include userinfo")
    _validate_host(parsed.hostname)
    if parsed.fragment:
        raise ValueError(f"{key} may not include a fragment")
    for field in parsed.query.split("&"):
        raw_query_key, separator, raw_query_value = field.partition("=")
        query_key = _strict_unquote(raw_query_key)
        query_value = _strict_unquote(raw_query_value) if separator else ""
        if _is_sensitive_identifier(query_key, query_key=True):
            raise ValueError(f"{key} query contains a credential-like key")
        _scan_decoded_query_value(query_value)


def _post_scan(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in _URL_KEYS:
                _validate_url(item, normalized)
            elif normalized == "credential_env_var":
                continue
            elif _is_sensitive_identifier(key):
                raise ValueError("credential-like key rejected")
            _post_scan(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _post_scan(item)
    elif isinstance(value, str):
        if _PRIVATE_KEY_BLOCK.search(value):
            raise ValueError("private key block rejected")
        if _IDENTIFIER_SHAPED.fullmatch(value) and _is_sensitive_identifier(value):
            raise ValueError("credential-like identifier rejected")
        assignment = _ASSIGNMENT.match(value)
        if assignment and _is_sensitive_identifier(assignment.group(1)):
            raise ValueError("credential-like assignment rejected")


def validate_qualification_secret_free(value: Any) -> None:
    validate_secret_free(value, read_environment=False)
    _post_scan(value)


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


def _absolute_path(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return Path(os.path.abspath(os.path.normpath(str(selected))))


def _reject_reparse_components(path: Path) -> None:
    selected = _absolute_path(path)
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
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


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stable_times(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_mtime_ns, metadata.st_ctime_ns


def _read_fd_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular_single_link(path: Path, label: str) -> bytes:
    selected = _absolute_path(path)
    _reject_reparse_components(selected)
    try:
        before_path = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if _is_reparse(selected) or not _regular_single_link(before_path):
        raise ValueError(f"{label} must be a regular single-link file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(selected, flags)
        before_descriptor = os.fstat(descriptor)
        if (
            not _regular_single_link(before_descriptor)
            or _identity(before_descriptor) != _identity(before_path)
            or before_descriptor.st_size != before_path.st_size
        ):
            raise ValueError(f"{label} changed while being read")
        raw = _read_fd_all(descriptor)
        after_descriptor = os.fstat(descriptor)
        _reject_reparse_components(selected)
        after_path = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if (
        _is_reparse(selected)
        or not _regular_single_link(after_descriptor)
        or not _regular_single_link(after_path)
        or _identity(after_descriptor) != _identity(before_descriptor)
        or _identity(after_path) != _identity(before_descriptor)
        or after_descriptor.st_size != before_descriptor.st_size
        or after_path.st_size != before_path.st_size
        or _stable_times(after_descriptor) != _stable_times(before_descriptor)
        or _stable_times(after_path) != _stable_times(before_path)
        or len(raw) != before_descriptor.st_size
    ):
        raise ValueError(f"{label} changed while being read")
    return raw


def load_canonical_jsonl_v1(path: Path, model_type: type[Any], *, label: str) -> tuple[tuple[Any, ...], bytes]:
    raw = _read_regular_single_link(Path(path), label)
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
        validate_qualification_secret_free(row.model_dump(mode="json"))
        rows.append(row)
    result = tuple(rows)
    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in result))
    return result, raw



def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plan_payload_without_computed_call_ids(payload: object) -> object:
    if not isinstance(payload, Mapping):
        return payload
    normalized = dict(payload)
    attempts = normalized.get("attempts")
    if isinstance(attempts, list):
        normalized["attempts"] = [
            {key: value for key, value in row.items() if key != "call_id"}
            if isinstance(row, Mapping)
            else row
            for row in attempts
        ]
    return normalized


def _load_canonical_json_model_v1(path: Path, model_type: type[Any], *, label: str) -> Any:
    raw = _read_regular_single_link(Path(path), label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} contains invalid JSON") from exc
    validate_qualification_secret_free(payload)
    try:
        model = model_type.model_validate(
            _plan_payload_without_computed_call_ids(payload)
            if model_type is CapabilitySmokePlanV1
            else payload
        )
    except Exception as exc:
        raise ValueError(f"{label} does not satisfy its contract") from exc
    if canonical_bytes(model) != raw:
        raise ValueError(f"{label} is not canonical")
    validate_qualification_secret_free(model.model_dump(mode="json"))
    return model


def validate_canonical_capability_smoke_plan_v1(
    plan: CapabilitySmokePlanV1,
) -> CapabilitySmokePlanV1:
    if type(plan) is not CapabilitySmokePlanV1:
        raise TypeError("capability smoke plan must use CapabilitySmokePlanV1")
    expected = build_capability_smoke_plan_v1(
        CapabilitySmokePlanConfigV1(
            release_id=_EXPECTED_RELEASE_ID,
            registry_keys=_EXPECTED_REGISTRY_KEYS,
            budget=build_capability_budget_v1(),
        ),
        build_capability_fixtures_v1(),
    )
    if canonical_bytes(plan) != canonical_bytes(expected):
        raise ValueError("capability smoke plan differs from the canonical planner-derived plan")
    return plan


def load_capability_smoke_plan_v1(path: Path) -> CapabilitySmokePlanV1:
    return validate_canonical_capability_smoke_plan_v1(
        _load_canonical_json_model_v1(path, CapabilitySmokePlanV1, label="capability smoke plan")
    )


def _coerce_capability_smoke_plan_v1(plan_raw: object) -> CapabilitySmokePlanV1:
    if isinstance(plan_raw, CapabilitySmokePlanV1):
        plan = plan_raw
    elif isinstance(plan_raw, (bytes, bytearray)):
        try:
            payload = json.loads(bytes(plan_raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("capability smoke plan contains invalid JSON") from exc
        validate_qualification_secret_free(payload)
        try:
            plan = CapabilitySmokePlanV1.model_validate(
                _plan_payload_without_computed_call_ids(payload)
            )
        except Exception as exc:
            raise ValueError("capability smoke plan does not satisfy its contract") from exc
        if canonical_bytes(plan) != bytes(plan_raw):
            raise ValueError("capability smoke plan is not canonical")
    else:
        raise TypeError("plan_raw must be CapabilitySmokePlanV1 or canonical JSON bytes")
    validate_qualification_secret_free(plan.model_dump(mode="json"))
    return plan



def load_capability_anomaly_receipt_v1(
    path: Path, plan_raw: CapabilitySmokePlanV1 | bytes | bytearray
) -> tuple[CapabilityAnomalyReceiptV1, bytes]:
    plan = _coerce_capability_smoke_plan_v1(plan_raw)
    raw = _read_regular_single_link(Path(path), "capability anomaly receipt")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("capability anomaly receipt contains invalid JSON") from exc
    validate_qualification_secret_free(payload)
    try:
        receipt = CapabilityAnomalyReceiptV1.model_validate(payload)
    except Exception as exc:
        raise ValueError("capability anomaly receipt does not satisfy its contract") from exc
    if canonical_bytes(receipt) != raw:
        raise ValueError("capability anomaly receipt is not canonical")
    _require(receipt.release_id == plan.release_id, "anomaly receipt release ID mismatch")
    _require(receipt.plan_sha256 == canonical_hash(plan), "anomaly receipt plan hash mismatch")
    base_call_ids = {
        attempt.call_id for attempt in plan.attempts if attempt.phase is AttemptPhase.BASE
    }
    _require(
        set(receipt.base_call_ids).issubset(base_call_ids),
        "anomaly receipt base call IDs must be base plan calls",
    )
    validate_qualification_secret_free(receipt.model_dump(mode="json"))
    return receipt, raw


def validate_escalation_anomaly_receipt_v1(
    receipt: CapabilityAnomalyReceiptV1,
    plan_raw: CapabilitySmokePlanV1 | bytes | bytearray,
    selected_attempts: Sequence[CapabilityAttemptPlanV1],
) -> CapabilityAnomalyReceiptV1:
    plan = _coerce_capability_smoke_plan_v1(plan_raw)
    selected = tuple(selected_attempts)
    _require(
        all(isinstance(attempt, CapabilityAttemptPlanV1) for attempt in selected),
        "selected attempts must use CapabilityAttemptPlanV1",
    )
    escalation_keys = {
        attempt.registry_key for attempt in selected if attempt.phase is AttemptPhase.ESCALATION
    }
    _require(escalation_keys, "anomaly receipt requires selected escalation attempts")
    required_base_call_ids = {
        attempt.call_id
        for attempt in plan.attempts
        if attempt.phase is AttemptPhase.BASE and attempt.registry_key in escalation_keys
    }
    _require(
        required_base_call_ids.issubset(receipt.base_call_ids),
        "anomaly receipt lacks base coverage for selected escalation roles",
    )
    return receipt


def validate_escalation_anomaly_evidence_v1(
    receipt: CapabilityAnomalyReceiptV1,
    base_receipts: Sequence[CapabilityAttemptReceiptV1],
    base_receipts_raw: bytes,
    plan_raw: CapabilitySmokePlanV1 | bytes | bytearray,
    selected_attempts: Sequence[CapabilityAttemptPlanV1],
) -> CapabilityAnomalyReceiptV1:
    plan = _coerce_capability_smoke_plan_v1(plan_raw)
    _require(receipt.release_id == plan.release_id, "anomaly receipt release ID mismatch")
    _require(receipt.plan_sha256 == canonical_hash(plan), "anomaly receipt plan hash mismatch")
    rows = tuple(base_receipts)
    _require(isinstance(base_receipts_raw, bytes), "base receipt bytes must be supplied")
    expected_base_receipts_raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    _require(
        base_receipts_raw == expected_base_receipts_raw,
        "base receipt bytes do not match supplied receipt rows",
    )
    _require(
        hashlib.sha256(base_receipts_raw).hexdigest() == receipt.base_receipts_sha256,
        "anomaly receipt base receipt hash mismatch",
    )
    _require(rows, "base receipts must be nonempty")
    _require(
        all(isinstance(row, CapabilityAttemptReceiptV1) for row in rows),
        "base receipts must use CapabilityAttemptReceiptV1",
    )
    receipt_ids = tuple(row.call_id for row in rows)
    _require(len(receipt_ids) == len(set(receipt_ids)), "base receipt call IDs must be unique")
    _require(
        set(receipt_ids) == set(receipt.base_call_ids),
        "base receipts must exactly cover anomaly base call IDs",
    )
    attempts_by_id = {attempt.call_id: attempt for attempt in plan.attempts}
    base_attempts: list[CapabilityAttemptPlanV1] = []
    for row in rows:
        attempt = attempts_by_id.get(row.call_id)
        _require(attempt is not None, "base receipt call ID is not in plan")
        base_attempts.append(attempt)
    validate_capability_attempt_receipts_v1(tuple(base_attempts), rows)
    for row in rows:
        attempt = attempts_by_id.get(row.call_id)
        _require(attempt is not None, "base receipt call ID is not in plan")
        _require(attempt.phase is AttemptPhase.BASE, "base receipt must bind a BASE plan attempt")
        _require(row.registry_key == attempt.registry_key, "base receipt registry mismatch")
        _require(row.retry_count == 0, "base receipt retries must be zero")
    selected = tuple(selected_attempts)
    _require(
        all(isinstance(attempt, CapabilityAttemptPlanV1) for attempt in selected),
        "selected attempts must use CapabilityAttemptPlanV1",
    )
    attempts_by_id = {attempt.call_id: attempt for attempt in plan.attempts}
    _require(bool(selected), "anomaly evidence requires selected escalation attempts")
    for selected_attempt in selected:
        plan_attempt = attempts_by_id.get(selected_attempt.call_id)
        _require(
            plan_attempt is not None
            and plan_attempt.phase is AttemptPhase.ESCALATION
            and plan_attempt == selected_attempt,
            "selected escalation attempt must exactly exist in the plan",
        )
    escalation_keys = {
        attempt.registry_key for attempt in selected if attempt.phase is AttemptPhase.ESCALATION
    }
    _require(escalation_keys, "anomaly evidence requires selected escalation attempts")
    required_base_ids = {
        attempt.call_id for attempt in plan.attempts
        if attempt.phase is AttemptPhase.BASE and attempt.registry_key in escalation_keys
    }
    _require(
        required_base_ids.issubset(receipt.base_call_ids),
        "anomaly receipt lacks complete base receipts for selected escalation roles",
    )
    receipt_by_id = {row.call_id: row for row in rows}
    anomalous_rows = tuple(receipt_by_id[call_id] for call_id in receipt.anomalous_call_ids)
    _require(
        all(row.status is GateStatus.FAIL for row in anomalous_rows),
        "anomalous base receipts must be FAIL",
    )
    checks = {
        "PARSER": lambda value: value == "PARSER_MISMATCH",
        "FORMAT": lambda value: value is not None and "FORMAT" in value,
        "STABILITY": lambda value: value is not None and "STABILITY" in value,
    }
    matches_declared_type = lambda row: any(
        checks[anomaly_type](row.error_class) for anomaly_type in receipt.anomaly_types
    )
    _require(
        all(matches_declared_type(row) for row in anomalous_rows),
        "every anomalous base receipt must match a declared anomaly type",
    )
    for registry_key in escalation_keys:
        _require(
            any(
                row.registry_key == registry_key and matches_declared_type(row)
                for row in anomalous_rows
            ),
            "each selected escalation role requires a matching anomaly type",
        )
    for anomaly_type in receipt.anomaly_types:
        _require(
            any(checks[anomaly_type](row.error_class) for row in anomalous_rows),
            "anomaly type does not match an anomalous receipt error class",
        )
    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in rows))
    return receipt


def load_execution_authorization_v1(
    path: Path, plan_raw: CapabilitySmokePlanV1 | bytes | bytearray
) -> ExecutionAuthorizationV1:
    plan = _coerce_capability_smoke_plan_v1(plan_raw)
    authorization = _load_canonical_json_model_v1(
        path, ExecutionAuthorizationV1, label="execution authorization"
    )
    _require(authorization.release_id == plan.release_id, "authorization release ID mismatch")
    _require(authorization.plan_sha256 == canonical_hash(plan), "authorization plan hash mismatch")
    _require(authorization.max_calls >= len(authorization.authorized_call_ids), "authorization max calls is below selected calls")
    _require(authorization.max_calls <= len(plan.attempts), "authorization max calls exceeds plan calls")
    plan_call_ids = {attempt.call_id for attempt in plan.attempts}
    _require(
        set(authorization.authorized_call_ids).issubset(plan_call_ids),
        "authorization contains an unknown call ID",
    )
    selected = tuple(
        attempt for attempt in plan.attempts if attempt.call_id in set(authorization.authorized_call_ids)
    )
    has_escalation = any(attempt.phase is AttemptPhase.ESCALATION for attempt in selected)
    _require(
        (has_escalation and authorization.escalation_anomaly_receipt_sha256 is not None)
        or (not has_escalation and authorization.escalation_anomaly_receipt_sha256 is None),
        "authorization escalation anomaly receipt does not match selected phases",
    )
    return authorization


def validate_capability_attempt_receipts_v1(
    selected_attempts: Sequence[CapabilityAttemptPlanV1],
    receipts: Sequence[CapabilityAttemptReceiptV1],
) -> tuple[CapabilityAttemptReceiptV1, ...]:
    _require(not isinstance(selected_attempts, (str, bytes)), "selected attempts must be a sequence")
    _require(not isinstance(receipts, (str, bytes)), "capability receipts must be a sequence")
    selected = tuple(selected_attempts)
    result = tuple(receipts)
    _require(selected, "selected attempts must be nonempty")
    _require(len(selected) == len(result), "capability receipt count mismatch")
    _require(
        all(isinstance(attempt, CapabilityAttemptPlanV1) for attempt in selected),
        "selected attempts must use CapabilityAttemptPlanV1",
    )
    _require(
        all(isinstance(receipt, CapabilityAttemptReceiptV1) for receipt in result),
        "capability receipts must use CapabilityAttemptReceiptV1",
    )
    selected_call_ids = tuple(attempt.call_id for attempt in selected)
    _require(len(selected_call_ids) == len(set(selected_call_ids)), "selected call IDs must be unique")
    _require(
        tuple(receipt.call_id for receipt in result) == selected_call_ids,
        "capability receipt call IDs or order mismatch",
    )
    validate_qualification_secret_free(tuple(receipt.model_dump(mode="json") for receipt in result))
    for attempt, receipt in zip(selected, result):
        _require(receipt.registry_key == attempt.registry_key, "capability receipt registry mismatch")
        _require(receipt.retry_count == 0, "capability receipt retries must be zero")
        if receipt.status is GateStatus.PASS:
            _require(receipt.error_class is None, "PASS capability receipt cannot carry an error")
            _require(receipt.redacted_response_sha256 is not None, "PASS capability receipt requires a response hash")
            if attempt.runtime_or_endpoint_class == "api_transfer_station":
                _require(
                    receipt.response_format in {"ANTHROPIC_MESSAGE_JSON", "SSE"},
                    "closed PASS receipt response format mismatch",
                )
                _require(
                    receipt.response_model == _EXPECTED_CAPABILITY_RESPONSE_MODELS.get(attempt.registry_key),
                    "closed PASS receipt response model mismatch",
                )
                _require(
                    receipt.stop_reason is not None and receipt.stop_reason.strip(),
                    "closed PASS receipt requires a nonblank stop reason",
                )
                _require(receipt.usage_present is not None, "closed PASS receipt requires usage evidence")
            else:
                _require(
                    receipt.response_format == "LOCAL_TEXT",
                    "local PASS receipt response format mismatch",
                )
                _require(receipt.response_model is None, "local PASS receipt must not carry a response model")
                _require(receipt.latency_ms is not None, "local PASS receipt requires latency")
        elif receipt.status in {GateStatus.FAIL, GateStatus.BLOCKED}:
            _require(receipt.error_class is not None and receipt.error_class.strip(), "non-PASS capability receipt requires error class")
            _require(receipt.redacted_response_sha256 is None, "non-PASS capability receipt cannot carry a response hash")
        else:
            raise ValueError("capability receipt status must be PASS, FAIL, or BLOCKED")
    return result


def validate_provider_attestations_v1(
    rows: Sequence[ProviderCapabilityAttestationV1],
) -> tuple[ProviderCapabilityAttestationV1, ...]:
    _require(not isinstance(rows, (str, bytes)), "provider attestations must be a sequence")
    result = tuple(rows)
    _require(len(result) == len(_EXPECTED_PROVIDER_ROWS), "provider attestation count mismatch")
    _require(
        all(isinstance(row, ProviderCapabilityAttestationV1) for row in result),
        "provider attestations must use ProviderCapabilityAttestationV1",
    )
    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in result))

    observation_ids: set[str] = set()
    for row, (registry_key, request_name, expected_calls) in zip(result, _EXPECTED_PROVIDER_ROWS):
        _require(row.registry_key == registry_key, "provider attestation registry order mismatch")
        _require(row.request_name == request_name, "provider request name mismatch")
        _require(row.evidence_class == "connectivity_interface_attestation", "provider evidence class mismatch")
        _require(row.provider_call_count == expected_calls, "provider call count mismatch")
        _require(row.retry_count == 0, "provider retry count must be zero")
        _require(row.benchmark_generation_count == 0, "benchmark generation count must be zero")
        _require(row.raw_response_persisted is False, "raw provider responses may not be persisted")
        _require(row.source_binding_ids == _EXPECTED_SOURCE_BINDINGS, "provider source bindings mismatch")
        _require(len(row.observations) == expected_calls, "provider observation count mismatch")
        _require(
            sum(item.provider_call_count for item in row.observations) == expected_calls,
            "provider observation call total mismatch",
        )
        _require(
            (row.canonical_model_identity, row.reasoning_tier, row.identity_caveat)
            == _EXPECTED_IDENTITY_METADATA[registry_key],
            "provider identity metadata mismatch",
        )
        for observation in row.observations:
            _require(bool(observation.observation_id), "provider observation ID must be nonempty")
            _require(observation.observation_id not in observation_ids, "provider observation IDs must be globally unique")
            observation_ids.add(observation.observation_id)
            _require(observation.provider_call_count == 1, "each observation must record one provider call")
            _require(observation.retry_count == 0, "observation retry count must be zero")
            _require(observation.http_status == 200, "observation HTTP status must be 200")
            _require(observation.exact_ok is True, "observation exact result must be true")
            _require(observation.stop_reason == "end_turn", "observation must end normally")
            _require(observation.usage_present is True, "observation usage must be retained")
            _require(observation.response_model == row.request_name, "observation response model mismatch")

        if registry_key != "gpt_5_5":
            _require(
                tuple(item.location for item in row.observations) == ("LOCAL", "TANG2"),
                "non-GPT observations must cover LOCAL then TANG2",
            )
            _require(
                all(item.response_format == "ANTHROPIC_MESSAGE_JSON" for item in row.observations),
                "non-GPT observations must use Anthropic message JSON",
            )
        else:
            _require(
                tuple(item.observation_id for item in row.observations) == _GPT_OBSERVATION_IDS,
                "GPT observation ordering mismatch",
            )
            _require(
                tuple(item.response_format for item in row.observations) == _GPT_RESPONSE_FORMATS,
                "GPT response format sequence mismatch",
            )
            _require(
                tuple(item.location for item in row.observations) == ("LOCAL", "LOCAL", "TANG2", "TANG2"),
                "GPT observation locations mismatch",
            )

    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in result))
    return result


def _gate_value(value: object) -> object:
    return getattr(value, "value", value)


def _runtime_payloads(rows: object) -> tuple[object, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return (rows,)
    payloads: list[object] = []
    for row in rows:
        if isinstance(row, OpenRuntimeReceiptV1):
            payloads.append(row.model_dump(mode="json"))
        else:
            payloads.append(row)
    return tuple(payloads)


def validate_runtime_receipts_v1(
    rows: Sequence[OpenRuntimeReceiptV1],
) -> tuple[OpenRuntimeReceiptV1, ...]:
    validate_qualification_secret_free(_runtime_payloads(rows))
    _require(not isinstance(rows, (str, bytes)), "runtime receipts must be a sequence")
    result = tuple(rows)
    _require(len(result) == len(_EXPECTED_RUNTIME_ROWS), "runtime receipt count mismatch")
    _require(
        all(isinstance(row, OpenRuntimeReceiptV1) for row in result),
        "runtime receipts must use OpenRuntimeReceiptV1",
    )

    for row, (registry_key, revision, engine, snapshot_tree_sha256, source_binding_ids) in zip(
        result, _EXPECTED_RUNTIME_ROWS
    ):
        _require(row.registry_key == registry_key, "runtime receipt registry order mismatch")
        _require(row.revision == revision, "runtime receipt revision mismatch")
        _require(
            row.snapshot_tree_sha256 == snapshot_tree_sha256,
            "runtime snapshot tree identity mismatch",
        )
        _require(
            isinstance(row.source_binding_ids, tuple) and bool(row.source_binding_ids),
            "runtime source bindings must be a nonempty tuple",
        )
        _require(
            all(isinstance(item, str) for item in row.source_binding_ids),
            "runtime source bindings must be strings",
        )
        _require(
            row.source_binding_ids == source_binding_ids,
            "runtime source bindings mismatch",
        )
        _require(
            len(row.source_binding_ids) == len(set(row.source_binding_ids)),
            "runtime source bindings must be unique",
        )
        _require(isinstance(row.runtime, RuntimeManifestV1), "runtime manifest type mismatch")
        _require(row.runtime.engine == engine, "runtime engine mismatch")

        statuses = (
            row.load_status,
            row.generation_status,
            row.determinism_status,
            row.unload_status,
        )
        status_values = tuple(_gate_value(status) for status in statuses)
        _require(
            all(isinstance(status, str) and status in {item.value for item in GateStatus} for status in status_values),
            "runtime gate status invalid",
        )
        _require(
            GateStatus.UNSUPPORTED.value not in status_values,
            "UNSUPPORTED runtime gates require a future typed incompatibility proof contract",
        )
        if registry_key == "meta_muse_glimmer_30b_bf16":
            _require(
                _gate_value(row.load_status) != GateStatus.UNSUPPORTED.value,
                "Muse BF16 resource block must use BLOCKED",
            )
        if registry_key == "qwen35_9b_bf16" and _gate_value(row.load_status) == GateStatus.PASS.value:
            _require(
                all(getattr(row.runtime, field) is not None for field in _QWEN_PACKAGE_FIELDS),
                "Qwen transformers runtime package metadata incomplete",
            )

        if registry_key == "meta_muse_glimmer_30b_int4":
            _require(row.runtime.engine == "llama.cpp", "Muse GGUF must use llama.cpp")
            _require(
                isinstance(row.runtime.engine_commit, str)
                and _RUNTIME_GIT_SHA.fullmatch(row.runtime.engine_commit) is not None,
                "Muse GGUF engine commit must be a lowercase git SHA",
            )
            _require(
                isinstance(row.runtime.binary_sha256, str)
                and _RUNTIME_SHA256.fullmatch(row.runtime.binary_sha256) is not None,
                "Muse GGUF binary hash is required",
            )
            _require(
                isinstance(row.runtime.build_options_sha256, str)
                and _RUNTIME_SHA256.fullmatch(row.runtime.build_options_sha256) is not None,
                "Muse GGUF build hash is required",
            )
            _require(row.speculative_decoding == "off", "Muse GGUF speculative decoding must be off")

        generation_pass = _gate_value(row.generation_status) == GateStatus.PASS.value
        determinism_pass = _gate_value(row.determinism_status) == GateStatus.PASS.value
        if generation_pass:
            _require(
                _gate_value(row.load_status) == GateStatus.PASS.value
                and _gate_value(row.unload_status) == GateStatus.PASS.value,
                "generation requires load and unload PASS",
            )
            _require(determinism_pass, "generation requires determinism PASS")
            _require(
                all(getattr(row, field) is not None for field in _RUNTIME_EVIDENCE_FIELDS[:5]),
                "generation evidence is incomplete",
            )
            _require(
                row.peak_memory_bytes is not None and row.peak_memory_bytes > 0,
                "generation requires positive peak memory evidence",
            )
        else:
            _require(
                all(getattr(row, field) is None for field in _RUNTIME_EVIDENCE_FIELDS),
                "non-PASS generation cannot carry measurements or evidence",
            )
        if not determinism_pass:
            _require(
                all(getattr(row, field) is None for field in _RUNTIME_EVIDENCE_FIELDS),
                "non-PASS determinism cannot carry measurements or evidence",
            )
        if determinism_pass:
            _require(generation_pass, "determinism requires generation PASS")
        if _gate_value(row.load_status) == GateStatus.BLOCKED:
            _require(
                all(_gate_value(status) != GateStatus.PASS.value for status in statuses[1:]),
                "blocked load cannot have downstream PASS gates",
            )
            _require(
                all(getattr(row, field) is None for field in _RUNTIME_EVIDENCE_FIELDS),
                "blocked load cannot carry measurements",
            )
        if any(_gate_value(status) in {GateStatus.FAIL.value, GateStatus.BLOCKED.value, GateStatus.UNSUPPORTED.value} for status in statuses):
            _require(bool(row.blocked_reasons), "failed or blocked runtime gate requires a reason")

    validate_qualification_secret_free(tuple(row.model_dump(mode="json") for row in result))
    return result


__all__ = [
    "load_capability_anomaly_receipt_v1",
    "load_canonical_jsonl_v1",
    "load_execution_authorization_v1",
    "validate_capability_attempt_receipts_v1",
    "validate_canonical_capability_smoke_plan_v1",
    "validate_escalation_anomaly_evidence_v1",
    "validate_escalation_anomaly_receipt_v1",
    "validate_provider_attestations_v1",
    "validate_qualification_secret_free",
    "validate_runtime_receipts_v1",
]
