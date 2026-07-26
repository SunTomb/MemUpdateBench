from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")


class _DuplicateJSONKeyError(ValueError):
    pass


class _InvalidJSONNumberError(ValueError):
    pass


def _strict_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _InvalidJSONNumberError("non-finite JSON number")
    return value


def _reject_json_constant(token: str) -> Any:
    raise _InvalidJSONNumberError(f"non-finite JSON constant {token}")


def _path_text(path: Path) -> str:
    return str(path)


def _require_regular_file(path: Path) -> os.stat_result:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Legacy artifact does not exist: {_path_text(path)}") from exc
    except OSError as exc:
        raise OSError(f"Cannot inspect legacy artifact {_path_text(path)}: {exc}") from exc

    if stat.S_ISDIR(result.st_mode):
        raise IsADirectoryError(f"Legacy artifact path is a directory: {_path_text(path)}")
    if not stat.S_ISREG(result.st_mode):
        raise OSError(f"Legacy artifact is not a regular file: {_path_text(path)}")
    return result


def _signature(result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _changed_error(path: Path) -> RuntimeError:
    return RuntimeError(f"Legacy artifact changed while being loaded: {_path_text(path)}")


def _load_stable(path: Path, parser: Callable[[bytes, Path], _T]) -> _T:
    source_path = Path(path)
    initial_stat = _require_regular_file(source_path)
    pre_digest = _sha256_file(source_path)
    before_read_stat = _require_regular_file(source_path)

    try:
        with source_path.open("rb") as source:
            descriptor_before = os.fstat(source.fileno())
            raw = source.read()
            descriptor_after = os.fstat(source.fileno())
    except OSError as exc:
        raise _changed_error(source_path) from exc

    raw_digest = hashlib.sha256(raw).hexdigest()
    parse_error: Exception | None = None
    parsed: _T | None = None
    try:
        parsed = parser(raw, source_path)
    except Exception as exc:
        parse_error = exc

    try:
        after_parse_stat = _require_regular_file(source_path)
        post_digest = _sha256_file(source_path)
        final_stat = _require_regular_file(source_path)
    except (OSError, FileNotFoundError) as exc:
        raise _changed_error(source_path) from (parse_error or exc)

    signatures = {
        _signature(initial_stat),
        _signature(before_read_stat),
        _signature(descriptor_before),
        _signature(descriptor_after),
        _signature(after_parse_stat),
        _signature(final_stat),
    }
    if len(signatures) != 1 or not (pre_digest == raw_digest == post_digest):
        raise _changed_error(source_path) from parse_error

    if parse_error is not None:
        raise parse_error
    assert parsed is not None
    return parsed


def _decode_utf8(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Legacy artifact is not valid UTF-8: {_path_text(path)}") from exc


def _duplicate_key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(_duplicate_key_fingerprint(key))
        result[key] = value
    return result


def _parse_json(raw: bytes, path: Path) -> Any:
    text = _decode_utf8(raw, path)
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_strict_json_float,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJSONKeyError as exc:
        raise ValueError(
            "Legacy artifact contains duplicate JSON key "
            f"<redacted:{exc.args[0]}>: {_path_text(path)}"
        ) from exc
    except _InvalidJSONNumberError as exc:
        raise ValueError(f"Legacy artifact contains invalid JSON number: {_path_text(path)}") from exc
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"Legacy artifact contains invalid JSON: {_path_text(path)}") from exc


def _parse_dataset(raw: bytes, path: Path) -> list[dict[str, Any]]:
    payload = _parse_json(raw, path)
    if not isinstance(payload, list):
        raise ValueError(
            f"EvoMemory dataset must have a top-level JSON array: {_path_text(path)}"
        )
    if not payload:
        raise ValueError(f"EvoMemory dataset must not be empty: {_path_text(path)}")
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(
                f"EvoMemory dataset record at index {index} must be an object: "
                f"{_path_text(path)}"
            )
    return payload


def _parse_object(raw: bytes, path: Path) -> dict[str, Any]:
    payload = _parse_json(raw, path)
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy artifact must have a top-level JSON object: {_path_text(path)}")
    return payload


def _parse_csv(raw: bytes, path: Path) -> list[dict[str, str]]:
    text = _decode_utf8(raw, path)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"Legacy CSV must not be empty: {_path_text(path)}") from exc
    except csv.Error as exc:
        raise ValueError(f"Legacy artifact contains invalid CSV: {_path_text(path)}") from exc

    if not header:
        raise ValueError(f"Legacy CSV has a blank CSV header: {_path_text(path)}")
    for index, name in enumerate(header):
        if not name.strip():
            raise ValueError(
                f"Legacy CSV has a blank CSV header at column {index + 1}: {_path_text(path)}"
            )
    if len(set(header)) != len(header):
        raise ValueError(f"Legacy CSV has a duplicate CSV header: {_path_text(path)}")

    rows: list[dict[str, str]] = []
    try:
        for row_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise ValueError(
                    f"Legacy CSV row {row_number} has {len(values)} fields; "
                    f"expected {len(header)}: {_path_text(path)}"
                )
            rows.append(dict(zip(header, values, strict=True)))
    except csv.Error as exc:
        raise ValueError(f"Legacy artifact contains invalid CSV: {_path_text(path)}") from exc

    if not rows:
        raise ValueError(
            f"Legacy CSV must contain at least one data row: {_path_text(path)}"
        )
    return rows


def load_evomemory_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty legacy dataset without changing or enriching its records."""

    return _load_stable(path, _parse_dataset)


def load_evomemory_results(path: Path) -> dict[str, Any]:
    """Load a legacy EvoMemory result object without backfilling optional fields."""

    return _load_stable(path, _parse_object)


def load_json_summary(path: Path) -> dict[str, Any]:
    """Load a legacy JSON summary object while preserving its raw field semantics."""

    return _load_stable(path, _parse_object)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a non-empty CSV table as exact strings.

    Record terminators are consumed by ``csv.reader``; embedded quoted newlines are
    preserved, headers are not trimmed, and empty cells remain empty strings.
    """

    return _load_stable(path, _parse_csv)


def parse_legacy_bool(value: str | bool | int | float | None) -> bool | None:
    """Parse exact ASCII legacy boolean spellings without coercion.

    ASCII spaces, tabs, CR, LF, vertical tabs, and form feeds are trimmed.
    Unicode whitespace and non-ASCII confusables are not accepted.
    """

    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        if value == 0 or value == 1:
            return bool(value)
        raise ValueError("Unsupported legacy boolean integer value")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Unsupported legacy boolean float value")
        if value == 0.0:
            if math.copysign(1.0, value) < 0:
                raise ValueError("Unsupported legacy boolean float value")
            return False
        if value == 1.0:
            return True
        raise ValueError("Unsupported legacy boolean float value")
    if type(value) is str:
        normalized = value.strip(" \t\r\n\v\f")
        if not normalized:
            return None
        if any(ord(character) > 127 for character in normalized):
            raise ValueError("Unsupported legacy boolean string")
        normalized = normalized.lower()
        if normalized in {"true", "1", "1.0"}:
            return True
        if normalized in {"false", "0", "0.0"}:
            return False
        raise ValueError("Unsupported legacy boolean string")
    raise TypeError("Unsupported legacy boolean type")
