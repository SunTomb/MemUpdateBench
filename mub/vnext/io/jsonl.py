from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from mub.vnext.io.canonical import canonical_json_bytes

ModelT = TypeVar("ModelT", bound=BaseModel)
ScalarId = str | int | float | bool


def write_models(
    path: str | PathLike[str],
    models: Iterable[BaseModel],
    *,
    id_field: str,
) -> None:
    """Stream models to binary-LF JSONL, flushing after every completed row.

    Validation occurs incrementally. If a duplicate, invalid ID, or serialization
    failure is encountered late, the exception is raised before that row is
    written; all earlier flushed rows intentionally remain in the destination as
    the valid protocol prefix.
    """
    output_path = Path(path)
    seen: dict[tuple[type, ScalarId], int] = {}
    with output_path.open("wb") as handle:
        for row_number, model in enumerate(models, start=1):
            record_id = _model_id(
                model,
                id_field=id_field,
                context=f"{output_path}: row {row_number}",
            )
            identity = (type(record_id), record_id)
            first_row = seen.get(identity)
            if first_row is not None:
                raise ValueError(
                    f"{output_path}: row {row_number}: duplicate ID {record_id!r} "
                    f"for {id_field!r}; first seen at row {first_row}"
                )
            seen[identity] = row_number
            try:
                serialized = canonical_json_bytes(model)
            except (PydanticSerializationError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{output_path}: row {row_number}: serialization failed: {exc}"
                ) from exc
            handle.write(serialized)
            handle.write(b"\n")
            handle.flush()


def read_models(
    path: str | PathLike[str],
    model_type: type[ModelT],
    *,
    id_field: str,
) -> Iterator[ModelT]:
    """Lazily stream and validate one model per JSONL line without skipping errors."""
    input_path = Path(path)
    seen: dict[tuple[type, ScalarId], int] = {}
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            context = f"{input_path}: line {line_number}"
            if not raw_line.strip():
                raise ValueError(f"{context}: blank JSONL row is not allowed")
            try:
                text = raw_line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{context}: row is not valid strict UTF-8: {exc}") from exc
            try:
                payload = json.loads(
                    text,
                    parse_constant=_reject_nonfinite_constant,
                    parse_float=_parse_finite_float,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: malformed JSON: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"{context}: malformed JSON: {exc}") from exc
            try:
                model = model_type.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{context}: model validation failed: {exc}") from exc

            record_id = _model_id(model, id_field=id_field, context=context)
            identity = (type(record_id), record_id)
            first_line = seen.get(identity)
            if first_line is not None:
                raise ValueError(
                    f"{context}: duplicate ID {record_id!r} for {id_field!r}; "
                    f"first seen at line {first_line}"
                )
            seen[identity] = line_number
            yield model


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")
    return parsed


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _model_id(model: BaseModel, *, id_field: str, context: str) -> ScalarId:
    if id_field not in type(model).model_fields:
        raise ValueError(f"{context}: missing ID field {id_field!r}")
    value: Any = getattr(model, id_field)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{context}: ID field {id_field!r} must not be blank")
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{context}: ID field {id_field!r} must be a nonblank scalar")


__all__ = ["read_models", "write_models"]
