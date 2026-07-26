from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from mub.vnext.contracts.common import (
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
)
from mub.vnext.legacy.caveats import LEGACY_CAVEATS, legacy_namespace
from mub.vnext.legacy.loaders import (
    _parse_csv,
    _parse_json,
    parse_legacy_bool,
)


_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_NONNEGATIVE_INTEGER_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_PROMPT_SHA_RE = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})\Z")
_PRIVATE_PATH_SEGMENTS = frozenset({".private", "private", "secret", "secrets"})
_WIN32_RESERVED_NAME_RE = re.compile(
    r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z",
    re.IGNORECASE,
)
_SENSITIVE_KEYS = frozenset(
    {
        "accesstoken",
        "accesskey",
        "apikey",
        "apitoken",
        "authenticationkey",
        "authenticationtoken",
        "authkey",
        "authorization",
        "authtoken",
        "bearerkey",
        "bearertoken",
        "clientsecret",
        "cookie",
        "idtoken",
        "key",
        "password",
        "privatekey",
        "refreshkey",
        "refreshtoken",
        "secret",
        "sessionkey",
        "sessiontoken",
        "setcookie",
        "token",
        "xapikey",
    }
)


def _normalized_key(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        unicodedata.normalize("NFKC", value).casefold(),
    )


def _is_sensitive_key(normalized: str) -> bool:
    if normalized in _SENSITIVE_KEYS:
        return True
    return normalized.endswith(
        (
            "apikey",
            "apitoken",
            "authenticationtoken",
            "accesstoken",
            "accesskey",
            "sessiontoken",
            "sessionkey",
            "refreshtoken",
            "refreshkey",
            "bearertoken",
            "bearerkey",
            "authkey",
            "authenticationkey",
            "clientsecret",
            "privatekey",
            "password",
            "secret",
            "cookie",
            "authorization",
        )
    )


def _scan_payload_for_credentials(value: Any, path: Path) -> None:
    node_count = 0

    def visit(item: Any, field: str, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > 1_000_000:
            _fail(path, field, "payload node budget exceeded")
        if depth > 64:
            _fail(path, field, "payload nesting exceeds maximum depth 64")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                _fail(path, field, "must be finite")
            return
        if type(item) is str:
            if _contains_surrogate(item):
                _fail(path, field, "must contain only Unicode scalar values")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                visit(child, f"{field}[{index}]", depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                key = _string(key, path, f"{field}.key")
                if _is_sensitive_key(_normalized_key(key)):
                    _fail(
                        path,
                        f"{field}.{key}",
                        "sensitive credential fields are not importable",
                    )
                visit(child, f"{field}.{key}", depth + 1)
            return
        _fail(path, field, "must contain exact built-in JSON containers and scalars")

    visit(value, "payload", 0)


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _fail(path: Path, field: str, message: str) -> None:
    raise ValueError(f"{path} field={field}: {message}")


def _string(
    value: Any,
    path: Path,
    field: str,
    *,
    nonblank: bool = True,
) -> str:
    if type(value) is not str:
        _fail(path, field, "must be an exact built-in string")
    if nonblank and not value.strip():
        _fail(path, field, "must be non-blank")
    if _contains_surrogate(value):
        _fail(path, field, "must contain only Unicode scalar values")
    return value


def _optional_string(
    row: dict[str, Any], path: Path, field: str, *, preserve_blank: bool = False
) -> str | None:
    if field not in row or row[field] is None:
        return None
    if row[field] == "":
        return "" if preserve_blank else None
    return _string(row[field], path, field, nonblank=not preserve_blank)


def _integer(
    value: Any,
    path: Path,
    field: str,
    *,
    positive: bool = False,
) -> int:
    if type(value) is int:
        result = value
    elif type(value) is str and _NONNEGATIVE_INTEGER_RE.fullmatch(value):
        result = int(value)
    else:
        _fail(path, field, "must be an exact non-negative integer")
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        _fail(path, field, f"must be an exact {qualifier} integer")
    return result


def _optional_integer(
    row: dict[str, Any], path: Path, field: str
) -> int | None:
    if field not in row or row[field] is None or row[field] == "":
        return None
    return _integer(row[field], path, field)


def _number(value: Any, path: Path, field: str) -> float:
    if type(value) in {int, float} and type(value) is not bool:
        result = float(value)
    elif type(value) is str and _JSON_NUMBER_RE.fullmatch(value):
        result = float(value)
    else:
        _fail(path, field, "must be an exact JSON number")
    if not math.isfinite(result):
        _fail(path, field, "must be finite")
    return result


def _optional_number(
    row: dict[str, Any], path: Path, field: str, *, rate: bool = False
) -> float | None:
    if field not in row or row[field] is None or row[field] == "":
        return None
    result = _number(row[field], path, field)
    if rate and not 0.0 <= result <= 1.0:
        _fail(path, field, "must be in [0, 1]")
    return result


def _optional_bounded_number(
    row: dict[str, Any],
    path: Path,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    value = _optional_number(row, path, field)
    if value is not None and not minimum <= value <= maximum:
        _fail(path, field, f"must be in [{minimum:g}, {maximum:g}]")
    return value


def _optional_rate_or_bool(
    row: dict[str, Any], path: Path, field: str
) -> float | None:
    if field not in row or row[field] is None or row[field] == "":
        return None
    value = row[field]
    if type(value) is bool:
        return float(value)
    if type(value) is str:
        try:
            parsed = parse_legacy_bool(value)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None and value.lower() in {"true", "false"}:
            return float(parsed)
    return _optional_number(row, path, field, rate=True)


def _has_c0_or_del(value: str) -> bool:
    return any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value)


def _raw_response_reference(
    row: dict[str, Any], path: Path
) -> tuple[str | None, str | None]:
    raw_path = _optional_string(row, path, "raw_response_path")
    raw_hash = _optional_string(row, path, "raw_response_sha256")
    if raw_path is not None and _has_c0_or_del(raw_path):
        _fail(path, "raw_response_path", "must not contain a control character")
    if (raw_path is None) != (raw_hash is None):
        _fail(path, "raw_response_path/raw_response_sha256", "must be supplied together")
    if raw_hash is not None and re.fullmatch(SHA256_PATTERN, raw_hash) is None:
        _fail(path, "raw_response_sha256", "must be an exact lowercase SHA-256 digest")
    return raw_path, raw_hash


def _stat_signature(result: os.stat_result) -> tuple[int, int, int, int]:
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _path_stat(path: Path) -> os.stat_result:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Legacy mechanism artifact does not exist: {path}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise IsADirectoryError(f"Legacy mechanism artifact is not a regular file: {path}")
    return result


def _load_payload(path: Path) -> tuple[Any, str, str]:
    if type(path) is not type(Path()):
        raise TypeError("path must be an exact concrete pathlib.Path")
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    parser = {".csv": _parse_csv, ".json": _parse_json}.get(suffix)
    if parser is None:
        _fail(source_path, "source_path", "must have .csv or .json suffix")

    path_before = _path_stat(source_path)
    parse_error: Exception | None = None
    payload: Any = None
    try:
        with source_path.open("rb") as source:
            descriptor_before = os.fstat(source.fileno())
            path_after_open = _path_stat(source_path)
            if descriptor_before.st_size > _MAX_SOURCE_BYTES:
                _fail(
                    source_path,
                    "source_path",
                    f"source byte cap {_MAX_SOURCE_BYTES} exceeded",
                )
            raw = source.read(_MAX_SOURCE_BYTES + 1)
            if len(raw) > _MAX_SOURCE_BYTES:
                _fail(
                    source_path,
                    "source_path",
                    f"source byte cap {_MAX_SOURCE_BYTES} exceeded on read",
                )
            descriptor_after_read = os.fstat(source.fileno())
            try:
                payload = parser(raw, source_path)
            except Exception as exc:
                parse_error = exc
            descriptor_after_parse = os.fstat(source.fileno())
            path_after_parse = _path_stat(source_path)
    except OSError as exc:
        raise RuntimeError(
            f"Legacy mechanism artifact changed during import: {source_path}"
        ) from exc

    signatures = {
        _stat_signature(path_before),
        _stat_signature(path_after_open),
        _stat_signature(descriptor_before),
        _stat_signature(descriptor_after_read),
        _stat_signature(descriptor_after_parse),
        _stat_signature(path_after_parse),
    }
    if len(signatures) != 1:
        raise RuntimeError(
            f"Legacy mechanism artifact changed during import: {source_path}"
        ) from parse_error
    if parse_error is not None:
        raise parse_error
    source_hash = hashlib.sha256(raw).hexdigest()
    return payload, str(source_path), source_hash


def _validate_row_container(row: Any, path: Path, field: str) -> dict[str, Any]:
    if type(row) is not dict:
        _fail(path, field, "must be an exact built-in object")
    for key, value in row.items():
        key = _string(key, path, f"{field}.key")
        normalized = _normalized_key(key)
        if normalized not in {
            "rawresponse",
            "rawresponsepath",
            "rawresponsesha256",
        } and _is_sensitive_key(normalized):
            _fail(path, f"{field}.{key}", "sensitive credential fields are not importable")
        if type(value) not in {str, int, float, bool, type(None)}:
            _fail(path, f"{field}.{key}", "mechanism rows may contain only exact JSON scalars")
        if type(value) is float and not math.isfinite(value):
            _fail(path, f"{field}.{key}", "must be finite")
        if type(value) is str and _contains_surrogate(value):
            _fail(path, f"{field}.{key}", "must contain only Unicode scalar values")
    return row


def _flat_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if type(payload) is list:
        rows = payload
    elif type(payload) is dict and type(payload.get("rows")) is list:
        rows = payload["rows"]
    else:
        _fail(path, "payload", "must be an array of rows or an object with a rows array")
    if not rows:
        _fail(path, "payload", "must contain at least one row")
    return [
        _validate_row_container(row, path, f"rows[{index}]")
        for index, row in enumerate(rows)
    ]


def _config_hash(kind: str, material: dict[str, Any]) -> str:
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(f"legacy-{kind}-config-v1".encode("ascii") + b"\0" + encoded).hexdigest()


def _prompt_sha(row: dict[str, Any], path: Path) -> str | None:
    declared = _optional_string(row, path, "prompt_sha256")
    prompt: str | None = None
    if "prompt" in row:
        raw_prompt = row["prompt"]
        if (
            type(raw_prompt) is not str
            or not raw_prompt.strip()
            or _contains_surrogate(raw_prompt)
        ):
            _fail(
                path,
                "prompt",
                "must be an exact built-in non-blank Unicode scalar string",
            )
        prompt = raw_prompt
    if declared is not None:
        if _PROMPT_SHA_RE.fullmatch(declared) is None:
            _fail(path, "prompt_sha256", "must be an exact lowercase 16- or 64-hex digest")
        if prompt is not None:
            full_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            authenticated = full_digest if len(declared) == 64 else full_digest[:16]
            if declared != authenticated:
                _fail(
                    path,
                    "prompt_sha256",
                    "does not authenticate inline prompt",
                )
        return declared
    if prompt is None:
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _sample_count(row: dict[str, Any], path: Path) -> int:
    return _integer(row.get("n", 1), path, "n", positive=True)


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


class MechanismAnalysisCell(ImmutableContractModel):
    source_path: str
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    legacy_namespace: str
    surface_condition: str
    sample_count: int = Field(gt=0, strict=True)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str | None = None
    raw_response: str | None = None
    raw_response_path: str | None = None
    raw_response_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    caveats: tuple[str, ...] = ()
    compatibility_analysis_only: Literal[True] = True

    @model_validator(mode="after")
    def _validate_common_provenance(self):
        if not self.source_path or _contains_surrogate(self.source_path):
            raise ValueError("source_path must be a non-blank Unicode scalar string")
        if not self.surface_condition or _contains_surrogate(self.surface_condition):
            raise ValueError("surface_condition must be a non-blank Unicode scalar string")
        if self.prompt_sha256 is not None and _PROMPT_SHA_RE.fullmatch(self.prompt_sha256) is None:
            raise ValueError("prompt_sha256 must be an exact lowercase 16- or 64-hex digest")
        if (self.raw_response_path is None) != (self.raw_response_sha256 is None):
            raise ValueError("raw_response_path and raw_response_sha256 must be supplied together")
        if self.raw_response_path is not None:
            raw_path = self.raw_response_path
            if _has_c0_or_del(raw_path):
                raise ValueError(
                    "raw_response_path must not contain a control character"
                )
            pure_path = PurePosixPath(raw_path)
            unsafe_path = (
                "\\" in raw_path
                or raw_path.startswith("/")
                or re.match(r"^[A-Za-z]:", raw_path) is not None
                or any(character in raw_path for character in "*?[]{}")
                or pure_path.is_absolute()
                or any(
                    part in {"", ".", ".."} or ":" in part
                    for part in pure_path.parts
                )
            )
            win32_ambiguous = any(
                part.endswith((".", " "))
                or _WIN32_RESERVED_NAME_RE.fullmatch(part) is not None
                for part in pure_path.parts
            )
            private_path = any(
                part.casefold() in _PRIVATE_PATH_SEGMENTS
                for part in pure_path.parts
            )
            if unsafe_path:
                raise ValueError(
                    "raw_response_path must be a safe repository-relative POSIX path"
                )
            if win32_ambiguous:
                raise ValueError(
                    "raw_response_path contains a Win32-ambiguous segment"
                )
            if private_path:
                raise ValueError("private raw-response paths are not importable")
        if any(not caveat or _contains_surrogate(caveat) for caveat in self.caveats):
            raise ValueError("caveats must contain non-blank Unicode scalar strings")
        return self


class ConflictProbeCell(MechanismAnalysisCell):
    legacy_namespace: Literal["legacy_p83"]
    distractor_count: int | None = Field(default=None, ge=0, strict=True)
    em: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    f1: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    value_em: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    answer_value_present: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    stale_value_copied: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    em_drop_from_final_only: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        strict=True,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def _validate_config_identity(self):
        if self.raw_response is not None:
            raise ValueError("P8.3 cells cannot preserve inline raw_response")
        expected = _config_hash(
            "p83-conflict",
            {
                "condition": self.surface_condition,
                "distractor_count": self.distractor_count,
            },
        )
        if self.config_sha256 != expected:
            raise ValueError("config_sha256 does not match conflict probe identity")
        return self


class SyntheticDoseCell(MechanismAnalysisCell):
    legacy_namespace: Literal["legacy_p83"]
    value_policy: str
    context_order: str
    context_annotation: str
    stale_count: int = Field(ge=0, strict=True)
    em: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    f1: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    answer_value_present: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_writer_identity(self):
        if self.raw_response is not None:
            raise ValueError("P8.3 cells cannot preserve inline raw_response")
        derived = (
            f"{self.value_policy}_stale{self.stale_count}_"
            f"{self.context_order}_{self.context_annotation}"
        )
        if self.surface_condition != derived:
            raise ValueError("surface_condition does not match writer-derived identity")
        expected = _config_hash(
            "p83-synthetic-dose",
            {
                "value_policy": self.value_policy,
                "context_order": self.context_order,
                "context_annotation": self.context_annotation,
                "stale_count": self.stale_count,
            },
        )
        if self.config_sha256 != expected:
            raise ValueError("config_sha256 does not match synthetic dose identity")
        return self


class StaleRemovalTraceCell(MechanismAnalysisCell):
    legacy_namespace: Literal["legacy_p83"]
    intervention: str
    gold_in_context_rate: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    stale_count_avg: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    entry_count_avg: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    removed_count_avg: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    original_em_avg: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    original_f1_avg: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    original_score_semantics: Literal["trace_composition_not_answer_rerun"]

    @model_validator(mode="after")
    def _validate_config_identity(self):
        if self.raw_response is not None:
            raise ValueError("P8.3 cells cannot preserve inline raw_response")
        if self.surface_condition != self.intervention:
            raise ValueError("surface_condition must equal intervention")
        expected = _config_hash(
            "p83-stale-removal", {"intervention": self.intervention}
        )
        if self.config_sha256 != expected:
            raise ValueError("config_sha256 does not match stale-removal identity")
        return self


ApiCellStatus = Literal[
    "completed",
    "format_caveat",
    "pending",
    "model_unavailable",
    "capacity_failed",
]


class ApiProbeCell(MechanismAnalysisCell):
    legacy_namespace: Literal["legacy_p84"]
    model: str
    raw_response_present: StrictBool
    stale_count: int | None = Field(default=None, ge=0, strict=True)
    status: ApiCellStatus
    is_completed: StrictBool
    em: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)
    stale_copied: float | None = Field(default=None, ge=0, le=1, strict=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_completion_state(self):
        expected_config = _config_hash(
            "p84-api-probe",
            {
                "model": self.model,
                "condition": self.surface_condition,
                "stale_count": self.stale_count,
                "raw_response_present": self.raw_response_present,
            },
        )
        if self.config_sha256 != expected_config:
            raise ValueError("config_sha256 does not match API probe identity")
        capacity_evidence = self.status == "capacity_failed" or any(
            caveat in {"capacity_exhausted", "capacity_failed", "resource_exhausted"}
            for caveat in self.caveats
        )
        if capacity_evidence and self.status != "capacity_failed":
            raise ValueError("capacity_failed evidence cannot become a completed cell")
        if not self.raw_response_present and self.raw_response is not None:
            raise ValueError(
                "raw_response must be None when raw_response_present is false"
            )
        if (
            self.status == "completed"
            and self.raw_response_present
            and (
                self.raw_response is None
                or not self.raw_response.strip()
            )
        ):
            raise ValueError(
                "completed API cells require a nonblank explicit raw_response"
            )
        if self.is_completed != (self.status == "completed"):
            raise ValueError("is_completed must be true exactly for completed status")
        if self.status == "completed" and (self.em is None or self.stale_copied is None):
            raise ValueError("completed API cells require em and stale_copied")
        return self


def import_conflict_probe(path: Path) -> list[ConflictProbeCell]:
    payload, source_path, source_hash = _load_payload(path)
    _scan_payload_for_credentials(payload, Path(path))
    source = Path(path)
    rows = _flat_rows(payload, source)
    cells: list[ConflictProbeCell] = []
    for row in rows:
        condition = _string(row.get("condition"), source, "condition")
        distractor_count = _optional_integer(row, source, "distractor_count")
        raw_path, raw_hash = _raw_response_reference(row, source)
        cells.append(
            ConflictProbeCell(
                source_path=source_path,
                source_sha256=source_hash,
                legacy_namespace=legacy_namespace("p83"),
                surface_condition=condition,
                sample_count=_sample_count(row, source),
                config_sha256=_config_hash(
                    "p83-conflict",
                    {"condition": condition, "distractor_count": distractor_count},
                ),
                prompt_sha256=_prompt_sha(row, source),
                raw_response=None,
                raw_response_path=raw_path,
                raw_response_sha256=raw_hash,
                caveats=("p83_order_metadata", LEGACY_CAVEATS["p83_order_metadata"]),
                distractor_count=distractor_count,
                em=_optional_number(row, source, "em", rate=True),
                f1=_optional_number(row, source, "f1", rate=True),
                value_em=_optional_rate_or_bool(row, source, "value_em"),
                answer_value_present=_optional_rate_or_bool(row, source, "answer_value_present"),
                stale_value_copied=_optional_rate_or_bool(row, source, "stale_value_copied"),
                em_drop_from_final_only=_optional_bounded_number(
                    row,
                    source,
                    "em_drop_from_final_only",
                    minimum=-1.0,
                    maximum=1.0,
                ),
            )
        )
    return cells


def import_synthetic_dose(path: Path) -> list[SyntheticDoseCell]:
    payload, source_path, source_hash = _load_payload(path)
    _scan_payload_for_credentials(payload, Path(path))
    source = Path(path)
    rows = _flat_rows(payload, source)
    cells: list[SyntheticDoseCell] = []
    for row in rows:
        value_policy = _string(row.get("value_policy"), source, "value_policy")
        context_order = _string(row.get("context_order"), source, "context_order")
        annotation = _string(row.get("context_annotation"), source, "context_annotation")
        stale_count = _integer(row.get("stale_count"), source, "stale_count")
        generated_condition = f"{value_policy}_stale{stale_count}_{context_order}_{annotation}"
        if "condition" not in row:
            condition = generated_condition
        else:
            condition = _string(row["condition"], source, "condition")
            if condition != generated_condition:
                _fail(
                    source,
                    "condition",
                    f"contradicts writer-derived condition {generated_condition!r}",
                )
        config = {
            "value_policy": value_policy,
            "context_order": context_order,
            "context_annotation": annotation,
            "stale_count": stale_count,
        }
        raw_path, raw_hash = _raw_response_reference(row, source)
        cells.append(
            SyntheticDoseCell(
                source_path=source_path,
                source_sha256=source_hash,
                legacy_namespace=legacy_namespace("p83"),
                surface_condition=condition,
                sample_count=_sample_count(row, source),
                config_sha256=_config_hash("p83-synthetic-dose", config),
                prompt_sha256=_prompt_sha(row, source),
                raw_response=None,
                raw_response_path=raw_path,
                raw_response_sha256=raw_hash,
                caveats=("p83_order_metadata", LEGACY_CAVEATS["p83_order_metadata"]),
                value_policy=value_policy,
                context_order=context_order,
                context_annotation=annotation,
                stale_count=stale_count,
                em=_optional_number(row, source, "em", rate=True),
                f1=_optional_number(row, source, "f1", rate=True),
                answer_value_present=_optional_rate_or_bool(row, source, "answer_value_present"),
            )
        )
    return cells


def import_stale_removal_trace(path: Path) -> list[StaleRemovalTraceCell]:
    payload, source_path, source_hash = _load_payload(path)
    _scan_payload_for_credentials(payload, Path(path))
    source = Path(path)
    rows = _flat_rows(payload, source)
    cells: list[StaleRemovalTraceCell] = []
    for row in rows:
        intervention = _string(row.get("intervention"), source, "intervention")
        raw_path, raw_hash = _raw_response_reference(row, source)
        cells.append(
            StaleRemovalTraceCell(
                source_path=source_path,
                source_sha256=source_hash,
                legacy_namespace=legacy_namespace("p83"),
                surface_condition=intervention,
                sample_count=_sample_count(row, source),
                config_sha256=_config_hash("p83-stale-removal", {"intervention": intervention}),
                prompt_sha256=_prompt_sha(row, source),
                raw_response=None,
                raw_response_path=raw_path,
                raw_response_sha256=raw_hash,
                caveats=(
                    "p83_order_metadata",
                    "original_scores_are_trace_composition_not_answer_rerun",
                ),
                intervention=intervention,
                gold_in_context_rate=_optional_number(row, source, "gold_in_context_rate", rate=True),
                stale_count_avg=_optional_number(row, source, "stale_count_avg"),
                entry_count_avg=_optional_number(row, source, "entry_count_avg"),
                removed_count_avg=_optional_number(row, source, "removed_count_avg"),
                original_em_avg=_optional_number(row, source, "original_em_avg", rate=True),
                original_f1_avg=_optional_number(row, source, "original_f1_avg", rate=True),
                original_score_semantics="trace_composition_not_answer_rerun",
            )
        )
    return cells


def _api_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if type(payload) is list or (type(payload) is dict and type(payload.get("rows")) is list):
        return _flat_rows(payload, path)
    if type(payload) is not dict:
        _fail(path, "payload", "API probe payload must be rows or a summary object")
    model = _string(payload.get("model"), path, "model")
    rows: list[dict[str, Any]] = []
    grouped = payload.get("by_condition_and_stale_count")
    if type(grouped) is dict:
        for condition, stale_map in grouped.items():
            condition = _string(condition, path, "by_condition_and_stale_count.condition")
            if type(stale_map) is not dict:
                _fail(path, f"by_condition_and_stale_count.{condition}", "must be an object")
            for stale_count, metrics in stale_map.items():
                stale_count = _string(stale_count, path, "stale_count")
                metrics = _validate_row_container(metrics, path, f"{condition}.{stale_count}")
                reserved = sorted(_NESTED_API_RESERVED_FIELDS.intersection(metrics))
                if reserved:
                    _fail(
                        path,
                        f"{condition}.{stale_count}",
                        f"reserved nested metric key {reserved[0]!r}",
                    )
                rows.append({"model": model, "condition": condition, "stale_count": stale_count, **metrics})
    elif type(payload.get("by_condition")) is dict:
        for condition, metrics in payload["by_condition"].items():
            condition = _string(condition, path, "by_condition.condition")
            metrics = _validate_row_container(metrics, path, f"by_condition.{condition}")
            reserved = sorted(_NESTED_API_RESERVED_FIELDS.intersection(metrics))
            if reserved:
                _fail(
                    path,
                    f"by_condition.{condition}",
                    f"reserved nested metric key {reserved[0]!r}",
                )
            rows.append({"model": model, "condition": condition, **metrics})
    else:
        _fail(path, "payload", "API summary requires by_condition or by_condition_and_stale_count")
    if not rows:
        _fail(path, "payload", "must contain at least one API cell")
    return rows


_CLEAN_API_MODELS = frozenset(
    {
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gemini-3.1-flash-lite-preview",
    }
)
_NESTED_API_RESERVED_FIELDS = frozenset(
    {
        "model",
        "condition",
        "stale_count",
        "source_path",
        "source_sha256",
        "legacy_namespace",
        "surface_condition",
        "sample_count",
        "config_sha256",
        "prompt_sha256",
        "raw_response",
        "raw_response_present",
        "raw_response_path",
        "raw_response_sha256",
        "caveats",
        "compatibility_analysis_only",
    }
)
_STATUS_CATEGORY_BY_VALUE: dict[str, ApiCellStatus] = {
    "clean": "completed",
    "completed": "completed",
    "complete": "completed",
    "success": "completed",
    "succeeded": "completed",
    "capacity_failed": "capacity_failed",
    "capacity_exhausted": "capacity_failed",
    "resource_exhausted": "capacity_failed",
    "pending": "pending",
    "queued": "pending",
    "running": "pending",
    "model_unavailable": "model_unavailable",
    "unavailable": "model_unavailable",
    "unsupported": "model_unavailable",
    "not_supported": "model_unavailable",
    "format_caveat": "format_caveat",
    "empty_truncated_response_caveat": "format_caveat",
    "empty_response": "format_caveat",
    "truncated_response": "format_caveat",
}


def _declared_api_status(
    row: dict[str, Any], path: Path
) -> ApiCellStatus | None:
    declarations: list[tuple[str, ApiCellStatus]] = []
    for field in ("row_status", "status"):
        value = _optional_string(row, path, field)
        if value is None:
            continue
        normalized = value.strip().lower()
        category = _STATUS_CATEGORY_BY_VALUE.get(normalized)
        if category is None:
            _fail(path, field, f"unsupported API probe status {value!r}")
        declarations.append((field, category))
    if not declarations:
        return None
    first_field, first_category = declarations[0]
    for field, category in declarations[1:]:
        if category != first_category:
            _fail(
                path,
                "row_status/status",
                f"alias conflict between {first_field} and {field}",
            )
    return first_category


def _api_status(
    row: dict[str, Any],
    path: Path,
    model: str,
    *,
    raw_response_present: bool,
    raw_response: str | None,
) -> tuple[ApiCellStatus, tuple[str, ...]]:
    caveats = ["p84_answer_layer_only", LEGACY_CAVEATS["p84_answer_layer_only"]]
    declared_caveat = _optional_string(row, path, "caveat")
    if declared_caveat is not None:
        caveats.append(declared_caveat)
    capacity = False
    if "capacity_failed" in row and row["capacity_failed"] not in {None, ""}:
        try:
            capacity = parse_legacy_bool(row["capacity_failed"]) is True
        except (TypeError, ValueError) as exc:
            _fail(path, "capacity_failed", f"invalid completion state: {exc}")
    normalized_status = _declared_api_status(row, path)
    if capacity or normalized_status == "capacity_failed":
        caveats.append(declared_caveat or "capacity_failed")
        return "capacity_failed", _dedupe(caveats)
    if normalized_status == "pending":
        caveats.append("pending")
        return "pending", _dedupe(caveats)
    if normalized_status == "model_unavailable":
        caveats.append("model_unavailable")
        return "model_unavailable", _dedupe(caveats)
    format_evidence = (
        raw_response_present
        and (raw_response is None or not raw_response.strip())
    ) or normalized_status == "format_caveat" or declared_caveat in {
        "empty_or_truncated_response",
        "empty_response",
        "truncated_response",
        "format_caveat",
    }
    if model == "gemini-2.5-flash":
        format_evidence = True
        caveats.append("empty_or_truncated_response")
    if format_evidence:
        caveats.append(declared_caveat or "empty_or_truncated_response")
        return "format_caveat", _dedupe(caveats)
    if normalized_status not in {None, "completed"}:
        _fail(path, "row_status/status", f"unsupported normalized API status {normalized_status!r}")
    has_metrics = all(row.get(field) not in {None, ""} for field in ("em", "stale_copied"))
    if not has_metrics:
        caveats.append("pending_missing_completion_evidence")
        return "pending", _dedupe(caveats)
    return "completed", _dedupe(caveats)


def import_api_probe(path: Path) -> list[ApiProbeCell]:
    payload, source_path, source_hash = _load_payload(path)
    _scan_payload_for_credentials(payload, Path(path))
    source = Path(path)
    rows = _api_rows(payload, source)
    cells: list[ApiProbeCell] = []
    for row in rows:
        row = _validate_row_container(row, source, "api_row")
        model = _string(row.get("model"), source, "model")
        condition = _string(row.get("condition"), source, "condition")
        stale_count = _optional_integer(row, source, "stale_count")
        raw_response_present = "raw_response" in row
        raw_response = _optional_string(
            row, source, "raw_response", preserve_blank=True
        )
        status, caveats = _api_status(
            row,
            source,
            model,
            raw_response_present=raw_response_present,
            raw_response=raw_response,
        )
        raw_path, raw_hash = _raw_response_reference(row, source)
        config = {
            "model": model,
            "condition": condition,
            "stale_count": stale_count,
            "raw_response_present": raw_response_present,
        }
        cells.append(
            ApiProbeCell(
                source_path=source_path,
                source_sha256=source_hash,
                legacy_namespace=legacy_namespace("p84"),
                surface_condition=condition,
                sample_count=_sample_count(row, source),
                config_sha256=_config_hash("p84-api-probe", config),
                prompt_sha256=_prompt_sha(row, source),
                raw_response=raw_response,
                raw_response_path=raw_path,
                raw_response_sha256=raw_hash,
                caveats=caveats,
                model=model,
                raw_response_present=raw_response_present,
                stale_count=stale_count,
                status=status,
                is_completed=status == "completed",
                em=_optional_number(row, source, "em", rate=True),
                stale_copied=_optional_number(row, source, "stale_copied", rate=True),
            )
        )
    return cells


def select_clean_api_cells(cells: list[ApiProbeCell]) -> list[ApiProbeCell]:
    if type(cells) is not list:
        raise TypeError("cells must be an exact built-in list")
    selected: list[ApiProbeCell] = []
    for index, cell in enumerate(cells):
        if type(cell) is not ApiProbeCell:
            raise TypeError(f"cells[{index}] must be an ApiProbeCell")
        stable_model = cell.model in _CLEAN_API_MODELS
        response_evidence_clean = (
            not cell.raw_response_present
            or (
                cell.raw_response is not None
                and bool(cell.raw_response.strip())
            )
        )
        if (
            stable_model
            and response_evidence_clean
            and cell.status == "completed"
            and cell.em is not None
            and cell.stale_copied is not None
        ):
            selected.append(cell)
    return selected


__all__ = [
    "ApiProbeCell",
    "ConflictProbeCell",
    "StaleRemovalTraceCell",
    "SyntheticDoseCell",
    "import_api_probe",
    "import_conflict_probe",
    "import_stale_removal_trace",
    "import_synthetic_dose",
    "select_clean_api_cells",
]
