from __future__ import annotations

from collections.abc import Mapping, Sequence
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.provenance_v1 import validate_secret_free
from mub.vnext.post_core.qualification_receipts_v1 import ProviderCapabilityAttestationV1


_EXPECTED_PROVIDER_ROWS = (
    ("claude_sonnet_4_6", "claude-sonnet-4-6", 2),
    ("claude_opus_4_8", "claude-opus-4-8", 2),
    ("gemini_3_6_flash", "Gemini 3.6 Flash (Low)", 2),
    ("grok_4_5", "grok-4.5", 2),
    ("gpt_5_5", "gpt-5.5", 4),
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
_SAFE_CONTRACT_KEYS = frozenset({"registry_key"})


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


__all__ = [
    "load_canonical_jsonl_v1",
    "validate_provider_attestations_v1",
    "validate_qualification_secret_free",
]
