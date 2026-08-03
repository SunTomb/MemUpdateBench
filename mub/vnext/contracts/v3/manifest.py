from typing import Literal

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import SHA256_PATTERN, FrozenStringMap, freeze_mapping
from mub.vnext.contracts.manifest import RunManifest, TaskManifest
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.contracts.v3.score import ScorerConfigV3
from mub.vnext.contracts.v3.version import (
    METRIC_REGISTRY_VERSION_V3,
    PROFILE_VERSION_V3,
    RUN_MANIFEST_VERSION_V3,
    RUNTIME_RECORD_VERSION_V3,
    SCHEMA_VERSION_V3,
    SCORER_VERSION_V3,
    TASK_MANIFEST_VERSION_V3,
)


def _validated_record_hashes(value, label):
    if not isinstance(value, dict) and not hasattr(value, "items"):
        raise ValueError(f"{label} must be a mapping")
    copied = dict(value)
    if any(type(key) is not str or not key.strip() for key in copied):
        raise ValueError(f"{label} keys must be nonblank task IDs")
    if any(type(item) is not str or len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in copied.values()):
        raise ValueError(f"{label} values must be lowercase sha256")
    return freeze_mapping(copied)


class TaskManifestV3(TaskManifest):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    task_manifest_version: Literal[TASK_MANIFEST_VERSION_V3] = TASK_MANIFEST_VERSION_V3
    task_schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    data_release_id: StrictIdentifier
    task_record_hashes: FrozenStringMap = Field(
        json_schema_extra={"additionalProperties": {"type": "string", "pattern": SHA256_PATTERN}}
    )

    @field_validator("task_record_hashes")
    @classmethod
    def _task_hashes(cls, value):
        return _validated_record_hashes(value, "task_record_hashes")

    @model_validator(mode="after")
    def _task_hash_coverage(self):
        expected = sum(self.split_counts.values())
        if len(self.task_record_hashes) != expected:
            raise ValueError("task_record_hashes must cover every task exactly")
        return self


class RunManifestV3(RunManifest):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    run_manifest_version: Literal[RUN_MANIFEST_VERSION_V3] = RUN_MANIFEST_VERSION_V3
    run_id: StrictIdentifier
    task_schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    runtime_record_version: Literal[RUNTIME_RECORD_VERSION_V3] = RUNTIME_RECORD_VERSION_V3
    scorer_version: Literal[SCORER_VERSION_V3] = SCORER_VERSION_V3
    metric_registry_version: Literal[METRIC_REGISTRY_VERSION_V3] = METRIC_REGISTRY_VERSION_V3
    profile_version: Literal[PROFILE_VERSION_V3] = PROFILE_VERSION_V3
    scorer_config: ScorerConfigV3
    adapter_info: AdapterInfoV3
    adapter_capabilities: AdapterCapabilitiesV3
    run_record_hashes: FrozenStringMap = Field(
        json_schema_extra={"additionalProperties": {"type": "string", "pattern": SHA256_PATTERN}}
    )

    @field_validator("run_record_hashes")
    @classmethod
    def _run_hashes(cls, value):
        return _validated_record_hashes(value, "run_record_hashes")

    @model_validator(mode="after")
    def _run_hash_coverage(self):
        if len(self.run_record_hashes) != self.expected_task_count:
            raise ValueError("run_record_hashes must cover expected tasks exactly")
        return self


__all__ = ["RunManifestV3", "TaskManifestV3"]
