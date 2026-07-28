from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, PlainSerializer, field_serializer, field_validator, model_validator

from mub.vnext.contracts.common import (
    FrozenDict,
    FrozenJsonObject,
    ImmutableContractModel,
    MetricFieldSupport,
    StrictBool,
    StrictNonnegativeFloat,
    StrictNonnegativeInt,
    freeze_json,
    freeze_mapping,
    thaw_json,
)
from mub.vnext.contracts.enums import CompletionStatus, Difficulty
from mub.vnext.failure import FailureFlag, canonicalize_failure_flags, primary_failure
from mub.vnext.version import SCHEMA_VERSION, SCORER_VERSION

StrictOptionalBool = StrictBool | None
StrictOptionalRate = Annotated[
    float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
] | None
StrictOptionalCount = StrictNonnegativeInt | None
StrictOptionalNonnegativeFloat = StrictNonnegativeFloat | None
FrozenMetricSupportMap = Annotated[
    Mapping[str, MetricFieldSupport],
    PlainSerializer(
        thaw_json,
        return_type=dict[str, MetricFieldSupport],
        when_used="always",
    ),
]


class ProtocolScores(ImmutableContractModel):
    action_parse_valid: StrictOptionalBool = None
    answer_parse_valid: StrictOptionalBool = None
    execution_success_rate: StrictOptionalRate = None
    unsupported_operation_rate: StrictOptionalRate = None
    fallback_rate: StrictOptionalRate = None


class ActionScores(ImmutableContractModel):
    operation_accuracy: StrictOptionalRate = None
    full_action_exact_match: StrictOptionalRate = None
    object_key_accuracy: StrictOptionalRate = None
    entity_accuracy: StrictOptionalRate = None
    attribute_accuracy: StrictOptionalRate = None
    value_accuracy: StrictOptionalRate = None
    false_write_rate: StrictOptionalRate = None
    missed_write_rate: StrictOptionalRate = None
    wrong_object_write_rate: StrictOptionalRate = None


class StateScores(ImmutableContractModel):
    final_state_accuracy: StrictOptionalRate = None
    state_precision: StrictOptionalRate = None
    state_recall: StrictOptionalRate = None
    state_f1: StrictOptionalRate = None
    state_resolve_rate: StrictOptionalRate = None
    collateral_corruption_rate: StrictOptionalRate = None
    expected_absence_accuracy: StrictOptionalRate = None


class StoreScores(ImmutableContractModel):
    obsolete_version_count: StrictOptionalCount = None
    stale_conflicting_value_count: StrictOptionalCount = None
    duplicate_current_count: StrictOptionalCount = None
    final_memory_size: StrictOptionalCount = None
    compaction_ratio: StrictOptionalNonnegativeFloat = None
    write_amplification: StrictOptionalNonnegativeFloat = None


class RetrievalScores(ImmutableContractModel):
    current_recall_at_k: StrictOptionalRate = None
    current_mrr: StrictOptionalRate = None
    stale_exposure_rate: StrictOptionalRate = None
    stale_count_in_context: StrictOptionalCount = None
    distractor_exposure_rate: StrictOptionalRate = None


class AnswerScores(ImmutableContractModel):
    exact_match: StrictOptionalRate = None
    normalized_match: StrictOptionalRate = None
    token_f1: StrictOptionalRate = None
    structured_field_accuracy: StrictOptionalRate = None
    reference_resolution_accuracy: StrictOptionalRate = None
    stale_copied: StrictOptionalRate = None
    distractor_copied: StrictOptionalRate = None
    gold_retrieved_wrong_answer: StrictOptionalRate = None
    answer_state_consistency: StrictOptionalRate = None


class SystemScores(ImmutableContractModel):
    ingest_latency_ms: StrictOptionalNonnegativeFloat = None
    retrieval_latency_ms: StrictOptionalNonnegativeFloat = None
    answer_latency_ms: StrictOptionalNonnegativeFloat = None
    token_usage: StrictOptionalCount = None
    api_cost: StrictOptionalNonnegativeFloat = None
    error_rate: StrictOptionalRate = None


class AuditScores(ImmutableContractModel):
    action_trace_available: StrictOptionalBool = None
    state_export_available: StrictOptionalBool = None
    retrieval_trace_available: StrictOptionalBool = None
    source_provenance_coverage: StrictOptionalRate = None
    manifest_completeness: StrictOptionalRate = None


SCORE_LAYER_TYPES = {
    "protocol_scores": ProtocolScores,
    "action_scores": ActionScores,
    "state_scores": StateScores,
    "store_scores": StoreScores,
    "retrieval_scores": RetrievalScores,
    "answer_scores": AnswerScores,
    "system_scores": SystemScores,
    "audit_scores": AuditScores,
}
METRIC_FIELD_PATHS = frozenset(
    f"{layer_name}.{field_name}"
    for layer_name, layer_type in SCORE_LAYER_TYPES.items()
    for field_name in layer_type.model_fields
)


class ScoreRecord(ImmutableContractModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    scorer_version: Literal[SCORER_VERSION] = SCORER_VERSION
    task_id: str
    run_id: str
    adapter_id: str
    task_family: str
    difficulty: Difficulty
    completion_status: CompletionStatus
    supported_metric_fields: FrozenMetricSupportMap = Field(
        default_factory=FrozenDict
    )
    protocol_scores: ProtocolScores
    action_scores: ActionScores
    state_scores: StateScores
    store_scores: StoreScores
    retrieval_scores: RetrievalScores
    answer_scores: AnswerScores
    system_scores: SystemScores
    audit_scores: AuditScores
    failure_flags: tuple[FailureFlag, ...] = ()
    primary_failure: str | None = None
    legacy_metrics: FrozenJsonObject = Field(default_factory=FrozenDict)

    @field_validator("failure_flags", mode="before")
    @classmethod
    def _canonicalize_failure_flags(cls, value):
        return canonicalize_failure_flags(value)

    @field_serializer("failure_flags", when_used="always")
    def _serialize_failure_flags(self, value):
        return tuple(
            flag.value if isinstance(flag, FailureFlag) else flag
            for flag in value
        )

    @field_validator("primary_failure", mode="before")
    @classmethod
    def _validate_primary_failure_label(cls, value):
        if value is None:
            return None
        if type(value) is not str:
            raise TypeError("primary_failure must be an exact built-in string")
        if value != "correct" and value not in {flag.value for flag in FailureFlag}:
            raise ValueError("unknown primary_failure label")
        return value

    @field_validator("supported_metric_fields")
    @classmethod
    def _freeze_supported_metric_fields(cls, value):
        return freeze_mapping(value)

    @field_validator("legacy_metrics")
    @classmethod
    def _freeze_legacy_metrics(cls, value):
        return freeze_json(value)

    @model_validator(mode="after")
    def _validate_metric_support_map(self):
        null_paths = {
            f"{layer_name}.{field_name}"
            for layer_name in SCORE_LAYER_TYPES
            for field_name, value in getattr(self, layer_name)
            if value is None
        }
        support_paths = set(self.supported_metric_fields)
        unknown_paths = support_paths - METRIC_FIELD_PATHS
        if unknown_paths:
            raise ValueError(f"unknown metric support paths: {sorted(unknown_paths)}")
        nonnull_paths = support_paths - null_paths
        if nonnull_paths:
            raise ValueError(f"support entries for non-null metrics: {sorted(nonnull_paths)}")
        missing_paths = null_paths - support_paths
        if missing_paths:
            raise ValueError(f"null metric fields missing support entries: {sorted(missing_paths)}")
        if self.primary_failure is not None:
            expected_primary = primary_failure(flag.value for flag in self.failure_flags)
            if self.primary_failure != expected_primary:
                raise ValueError(
                    f"primary_failure must equal {expected_primary!r} for failure_flags"
                )
        return self


__all__ = [
    "ActionScores",
    "AnswerScores",
    "AuditScores",
    "ProtocolScores",
    "RetrievalScores",
    "ScoreRecord",
    "StateScores",
    "StoreScores",
    "SystemScores",
]
