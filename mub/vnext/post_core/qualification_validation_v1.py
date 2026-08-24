from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import parse_qsl, urlsplit

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
_GPT_OBSERVATION_IDS = (
    "LOCAL_INITIAL_SSE",
    "LOCAL_EXPLICIT_FALSE_SSE",
    "TANG2_PREFIX_SSE",
    "TANG2_POSTFIX_JSON",
)
_GPT_RESPONSE_FORMATS = ("SSE", "SSE", "SSE", "ANTHROPIC_MESSAGE_JSON")
_URL_KEYS = frozenset({"endpoint", "endpoint_url", "source_url"})
_SENSITIVE_QUERY_KEY = re.compile(
    r"credential|token|authorization|password|secret|private[-_ ]?key|api[-_ ]?key|bearer|auth",
    re.IGNORECASE,
)
_PRIVATE_KEY_BLOCK = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE)


def _normalized_key(value: object) -> str:
    return re.sub(r"[-\s]+", "_", str(value).strip()).lower()


def _validate_url(value: object, key: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an HTTPS URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{key} must be a well-formed HTTPS URL") from exc
    if parsed.scheme != "https" or not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"{key} must be an HTTPS URL")
    if "@" in parsed.netloc:
        raise ValueError(f"{key} may not include userinfo")
    if parsed.fragment:
        raise ValueError(f"{key} may not include a fragment")
    for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        if _SENSITIVE_QUERY_KEY.search(query_key):
            raise ValueError(f"{key} query contains a credential-like key")


def _scan_urls(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in _URL_KEYS:
                _validate_url(item, normalized)
            _scan_urls(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan_urls(item)
    elif isinstance(value, str) and _PRIVATE_KEY_BLOCK.search(value):
        raise ValueError("private key block rejected")


def validate_qualification_secret_free(value: Any) -> None:
    validate_secret_free(value, read_environment=False)
    _scan_urls(value)


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


def _read_regular_single_link(path: Path, label: str) -> bytes:
    selected = _absolute_path(path)
    _reject_reparse_components(selected)
    try:
        before = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if _is_reparse(selected) or not _regular_single_link(before):
        raise ValueError(f"{label} must be a regular single-link file")
    try:
        with selected.open("rb") as handle:
            raw = handle.read()
        after = selected.lstat()
    except OSError as exc:
        raise ValueError(f"{label} could not be read safely") from exc
    if _is_reparse(selected) or not _regular_single_link(after):
        raise ValueError(f"{label} changed while being read")
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or len(raw) != before.st_size
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
        _require(
            row.source_binding_ids == _EXPECTED_SOURCE_BINDINGS,
            "provider source bindings must be the exact workflow/handoff pair",
        )
        _require(len(row.observations) == expected_calls, "provider observation count mismatch")
        _require(
            sum(item.provider_call_count for item in row.observations) == expected_calls,
            "provider observation call total mismatch",
        )
        for observation in row.observations:
            _require(bool(observation.observation_id), "provider observation ID must be nonempty")
            _require(observation.observation_id not in observation_ids, "provider observation IDs must be globally unique")
            observation_ids.add(observation.observation_id)
            _require(observation.provider_call_count == 1, "each observation must record one provider call")
            _require(observation.retry_count == 0, "observation retry count must be zero")
            _require(observation.http_status == 200, "observation HTTP status must be 200")
            _require(observation.exact_ok is True, "observation exact result must be true")
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
            _require(
                all(item.stop_reason == "end_turn" for item in row.observations),
                "non-GPT observations must end normally",
            )

        if registry_key == "claude_sonnet_4_6":
            _require(row.canonical_model_identity == "claude-sonnet-4-6", "Claude Sonnet identity mismatch")
        elif registry_key == "claude_opus_4_8":
            _require(row.canonical_model_identity == "claude-opus-4-8", "Claude Opus identity mismatch")
        elif registry_key == "gemini_3_6_flash":
            _require(
                row.canonical_model_identity == "gemini-3.6-flash"
                and row.request_name == "Gemini 3.6 Flash (Low)"
                and row.reasoning_tier == "Low",
                "Gemini request, canonical identity, and reasoning tier must be exact",
            )
        elif registry_key == "grok_4_5":
            _require(row.canonical_model_identity == "grok-4.5", "Grok canonical identity mismatch")
            _require(
                isinstance(row.identity_caveat, str) and "mutable alias" in row.identity_caveat.lower(),
                "Grok identity caveat must say mutable alias",
            )
        else:
            _require(row.canonical_model_identity is None, "GPT canonical identity must remain null")
            _require(
                isinstance(row.identity_caveat, str) and "unverified" in row.identity_caveat.lower(),
                "GPT identity caveat must state unverified status",
            )
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
