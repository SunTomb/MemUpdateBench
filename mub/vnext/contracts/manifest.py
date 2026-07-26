from __future__ import annotations

import re

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import (
    ArtifactRef,
    FrozenJsonObject,
    FrozenNonnegativeIntMap,
    FrozenStringMap,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
    StrictNonnegativeInt,
    freeze_json,
    freeze_mapping,
)
from mub.vnext.contracts.score import METRIC_FIELD_PATHS
from mub.vnext.version import (
    METRIC_REGISTRY_VERSION,
    PRIMARY_FAILURE_PRECEDENCE_VERSION,
    PROFILE_VERSION,
    RUN_MANIFEST_VERSION,
    RUNTIME_RECORD_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TASK_MANIFEST_VERSION,
)

VALUE_NORMALIZATION_PROFILE = "typed_exact_v1"
ANSWER_NORMALIZATION_PROFILE = "normalized_exact_v1"


class TaskManifest(ImmutableContractModel):
    schema_version: str = SCHEMA_VERSION
    task_manifest_version: str = TASK_MANIFEST_VERSION
    data_release_id: str
    split_policy_version: str
    task_schema_version: str = SCHEMA_VERSION
    compiler_versions: FrozenStringMap
    source_manifest_paths_and_hashes: tuple[ArtifactRef, ...]
    generation_configs_and_hashes: tuple[ArtifactRef, ...]
    split_counts: FrozenNonnegativeIntMap
    family_difficulty_counts: FrozenNonnegativeIntMap
    semantic_core_counts: FrozenNonnegativeIntMap
    task_file_paths_and_hashes: tuple[ArtifactRef, ...]
    leakage_check_summary: FrozenJsonObject
    human_audit_artifacts: tuple[ArtifactRef, ...]
    created_at: str
    code_revision: str

    @field_validator(
        "compiler_versions",
        "split_counts",
        "family_difficulty_counts",
        "semantic_core_counts",
    )
    @classmethod
    def _freeze_validated_maps(cls, value):
        return freeze_mapping(value)

    @field_validator("leakage_check_summary")
    @classmethod
    def _freeze_leakage_summary(cls, value):
        return freeze_json(value)


class RunManifest(ImmutableContractModel):
    schema_version: str = SCHEMA_VERSION
    run_manifest_version: str = RUN_MANIFEST_VERSION
    run_id: str
    timestamp: str
    code_revision: str
    dirty_state: StrictBool
    task_manifest: ArtifactRef
    task_schema_version: str = SCHEMA_VERSION
    runtime_record_version: str = RUNTIME_RECORD_VERSION
    scorer_version: str = SCORER_VERSION
    metric_registry_version: str = METRIC_REGISTRY_VERSION
    profile_version: str = PROFILE_VERSION
    adapter_info: AdapterInfo
    adapter_capabilities: AdapterCapabilities
    capability_verification_artifact: ArtifactRef | None
    model_name: str | None
    provider: str | None
    model_revision: str | None
    prompt_config: FrozenJsonObject
    decoding_config: FrozenJsonObject
    seed_information: FrozenJsonObject
    action_parser_version: str
    answer_parser_version: str
    memory_entry_extractor_version: str
    object_value_extractor_config_hash: str = Field(pattern=SHA256_PATTERN)
    redaction_policy_version: str
    environment_summary: FrozenJsonObject
    package_summary: FrozenJsonObject
    expected_task_count: StrictNonnegativeInt
    completed_task_count: StrictNonnegativeInt
    failed_task_count: StrictNonnegativeInt
    not_supported_task_count: StrictNonnegativeInt
    raw_provider_response_artifacts: tuple[ArtifactRef, ...]
    raw_adapter_state_artifacts: tuple[ArtifactRef, ...]
    normalized_runtime_artifacts: tuple[ArtifactRef, ...]
    score_artifacts: tuple[ArtifactRef, ...]
    native_vs_extracted_field_summary: FrozenJsonObject

    @field_validator(
        "prompt_config",
        "decoding_config",
        "seed_information",
        "environment_summary",
        "package_summary",
        "native_vs_extracted_field_summary",
    )
    @classmethod
    def _freeze_json_maps(cls, value):
        return freeze_json(value)

    @model_validator(mode="after")
    def _validate_task_counts(self):
        observed = (
            self.completed_task_count
            + self.failed_task_count
            + self.not_supported_task_count
        )
        if observed != self.expected_task_count:
            raise ValueError(
                "completed_task_count + failed_task_count + "
                "not_supported_task_count must equal expected_task_count"
            )
        return self


class ScorerConfig(ImmutableContractModel):
    scorer_version: str = SCORER_VERSION
    metric_registry_version: str = METRIC_REGISTRY_VERSION
    value_normalization_profile: str
    answer_normalization_profile: str
    primary_failure_precedence_version: str = PRIMARY_FAILURE_PRECEDENCE_VERSION
    requested_metric_fields: tuple[str, ...] = Field(
        default=(),
        description="Empty list means all registered metrics.",
    )
    legacy_compatibility_mode: str | None = None
    strict_capability_check: StrictBool = True

    @field_validator(
        "scorer_version",
        "metric_registry_version",
        "primary_failure_precedence_version",
    )
    @classmethod
    def _validate_current_versions(cls, value: str, info) -> str:
        expected = {
            "scorer_version": SCORER_VERSION,
            "metric_registry_version": METRIC_REGISTRY_VERSION,
            "primary_failure_precedence_version": PRIMARY_FAILURE_PRECEDENCE_VERSION,
        }[info.field_name]
        if value != expected:
            raise ValueError(f"{info.field_name} must equal current version {expected}")
        return value

    @field_validator(
        "value_normalization_profile",
        "answer_normalization_profile",
        mode="before",
    )
    @classmethod
    def _validate_current_normalization_profiles(cls, value, info) -> str:
        expected = {
            "value_normalization_profile": VALUE_NORMALIZATION_PROFILE,
            "answer_normalization_profile": ANSWER_NORMALIZATION_PROFILE,
        }[info.field_name]
        if type(value) is not str or value != expected:
            raise ValueError(f"{info.field_name} must equal current profile {expected}")
        return value

    @field_validator("legacy_compatibility_mode")
    @classmethod
    def _validate_legacy_namespace(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"legacy_[a-z0-9_]+", value):
            raise ValueError(
                "legacy_compatibility_mode must name an exact legacy metric namespace"
            )
        return value

    @field_validator("requested_metric_fields", mode="before")
    @classmethod
    def _canonicalize_requested_metric_fields(cls, value):
        if type(value) not in {tuple, list, set, frozenset}:
            raise ValueError("requested_metric_fields must be a tuple-like semantic set")
        supplied = tuple(value)
        if any(type(field) is not str for field in supplied):
            raise ValueError("requested_metric_fields values must be exact built-in strings")
        if len(supplied) != len(set(supplied)):
            raise ValueError("requested_metric_fields must be unique")
        return tuple(sorted(supplied))

    @field_validator("requested_metric_fields")
    @classmethod
    def _validate_requested_metric_fields(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if len(fields) != len(set(fields)):
            raise ValueError("requested_metric_fields must be unique")
        malformed = [
            field
            for field in fields
            if not re.fullmatch(r"[a-z][a-z0-9_]*_scores\.[a-z][a-z0-9_]*", field)
        ]
        if malformed:
            raise ValueError(f"malformed requested metric fields: {sorted(malformed)}")
        unknown = set(fields) - METRIC_FIELD_PATHS
        if unknown:
            raise ValueError(f"unknown requested metric fields: {sorted(unknown)}")
        return fields


__all__ = ["RunManifest", "ScorerConfig", "TaskManifest"]
