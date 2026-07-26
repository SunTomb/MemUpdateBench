from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel

from mub.vnext.io.atomic import publish_files_atomically

from mub.vnext.contracts.manifest import RunManifest, TaskManifest
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.score import ScoreRecord
from mub.vnext.contracts.task import MemUpdateTask

DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"

_TOP_LEVEL_SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("mem_update_task.schema.json", MemUpdateTask),
    ("task_run_record.schema.json", TaskRunRecord),
    ("score_record.schema.json", ScoreRecord),
    ("task_manifest.schema.json", TaskManifest),
    ("run_manifest.schema.json", RunManifest),
)
TOP_LEVEL_SCHEMA_MODELS = _TOP_LEVEL_SCHEMA_MODELS


def export_schemas(output_dir: str | Path) -> tuple[Path, ...]:
    """Atomically export the five deterministic vNext Draft 2020-12 schemas."""
    registry = _validated_registry(_TOP_LEVEL_SCHEMA_MODELS)
    destination = Path(output_dir)
    resolved_destination = destination.resolve(strict=False)
    exports: list[tuple[Path, type[BaseModel]]] = []
    for filename, model_type in registry:
        path = destination / filename
        if path.resolve(strict=False).parent != resolved_destination:
            raise ValueError(f"schema filename {filename!r} escapes the output directory")
        exports.append((path, model_type))

    destination.mkdir(parents=True, exist_ok=True)
    payloads: dict[Path, bytes] = {}
    validators = {}
    for path, model_type in exports:
        schema = model_type.model_json_schema(mode="serialization")
        schema["$schema"] = DRAFT_2020_12_URI
        schema["title"] = model_type.__name__
        content = _schema_bytes(schema)
        payloads[path] = content
        validators[path] = _schema_validator(content)
    publish_files_atomically(
        payloads,
        overwrite=True,
        validators=validators,
    )
    return tuple(path for path, _ in exports)


def _validated_registry(
    registry: tuple[tuple[str, type[BaseModel]], ...],
) -> tuple[tuple[str, type[BaseModel]], ...]:
    seen: set[str] = set()
    seen_casefolded: dict[str, str] = {}
    for filename, model_type in registry:
        is_plain_basename = (
            isinstance(filename, str)
            and bool(filename)
            and Path(filename).name == filename
            and not Path(filename).is_absolute()
            and not PurePosixPath(filename).is_absolute()
            and not PureWindowsPath(filename).is_absolute()
            and "/" not in filename
            and "\\" not in filename
            and filename.endswith(".schema.json")
        )
        if not is_plain_basename:
            raise ValueError(
                f"schema filename {filename!r} must be a plain .schema.json basename"
            )
        if filename in seen:
            raise ValueError(f"schema filename {filename!r} is duplicated")
        casefolded = filename.casefold()
        colliding_filename = seen_casefolded.get(casefolded)
        if colliding_filename is not None:
            raise ValueError(
                f"schema filename {filename!r} collides case-insensitively with "
                f"{colliding_filename!r}"
            )
        if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
            raise TypeError(f"schema filename {filename!r} has an invalid model type")
        seen.add(filename)
        seen_casefolded[casefolded] = filename
    return registry


def _schema_validator(expected: bytes):
    def validate(path: Path) -> None:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if type(payload) is not dict or raw != expected:
            raise ValueError("staged schema is not the expected canonical JSON")

    return validate


def _schema_bytes(schema: dict) -> bytes:
    return (
        json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


__all__ = ["DRAFT_2020_12_URI", "TOP_LEVEL_SCHEMA_MODELS", "export_schemas"]
