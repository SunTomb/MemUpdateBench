from typing import Literal

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


class TaskManifestV3(TaskManifest):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    task_manifest_version: Literal[TASK_MANIFEST_VERSION_V3] = TASK_MANIFEST_VERSION_V3
    task_schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    data_release_id: StrictIdentifier


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


__all__ = ["RunManifestV3", "TaskManifestV3"]
