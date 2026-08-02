from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from mub.vnext.contracts.manifest import RunManifest, TaskManifest
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.score import ScoreRecord
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3

ModelT = TypeVar("ModelT", bound=BaseModel)

_REGISTRIES: dict[str, dict[str, type[BaseModel]]] = {
    "task": {"2.0.0": MemUpdateTask, "3.0.0": MemUpdateTaskV3},
    "runtime": {"2.0.0": TaskRunRecord, "3.0.0": TaskRunRecordV3},
    "score": {"2.0.0": ScoreRecord, "3.0.0": ScoreRecordV3},
    "task_manifest": {"2.0.0": TaskManifest, "3.0.0": TaskManifestV3},
    "run_manifest": {"2.0.0": RunManifest, "3.0.0": RunManifestV3},
}


def parse_versioned_payload(kind: str, payload: Mapping[str, Any]) -> BaseModel:
    if type(payload) is not dict:
        raise TypeError("versioned payload must be an exact built-in dictionary")
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown top-level contract kind {kind!r}")
    version = payload.get("schema_version")
    if type(version) is not str:
        raise ValueError("schema_version is required and must be an exact built-in string")
    model = _REGISTRIES[kind].get(version)
    if model is None:
        raise ValueError(f"unsupported {kind} schema_version {version!r}")
    return model.model_validate(payload)


def parse_versioned_task(payload: Mapping[str, Any]) -> MemUpdateTask | MemUpdateTaskV3:
    return parse_versioned_payload("task", payload)  # type: ignore[return-value]


def parse_versioned_runtime_record(payload: Mapping[str, Any]) -> TaskRunRecord | TaskRunRecordV3:
    return parse_versioned_payload("runtime", payload)  # type: ignore[return-value]


def parse_versioned_score_record(payload: Mapping[str, Any]) -> ScoreRecord | ScoreRecordV3:
    return parse_versioned_payload("score", payload)  # type: ignore[return-value]


def parse_versioned_task_manifest(payload: Mapping[str, Any]) -> TaskManifest | TaskManifestV3:
    return parse_versioned_payload("task_manifest", payload)  # type: ignore[return-value]


def parse_versioned_run_manifest(payload: Mapping[str, Any]) -> RunManifest | RunManifestV3:
    return parse_versioned_payload("run_manifest", payload)  # type: ignore[return-value]


__all__ = ["parse_versioned_payload", "parse_versioned_run_manifest", "parse_versioned_runtime_record", "parse_versioned_score_record", "parse_versioned_task", "parse_versioned_task_manifest"]
