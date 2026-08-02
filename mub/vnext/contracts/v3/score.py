from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, PlainSerializer, field_validator, model_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, MetricFieldSupport, StrictBool, StrictNonnegativeFloat, StrictNonnegativeInt, freeze_mapping, thaw_json
from mub.vnext.contracts.enums import CompletionStatus, Difficulty
from mub.vnext.contracts.score import ActionScores, AnswerScores, AuditScores, ProtocolScores, RetrievalScores, StateScores, StoreScores, SystemScores
from mub.vnext.failure import FailureFlag
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.contracts.v3.enums import FailureFlagV3
from mub.vnext.contracts.v3.version import METRIC_REGISTRY_VERSION_V3, SCHEMA_VERSION_V3, SCORER_VERSION_V3

StrictOptionalBool = StrictBool | None
StrictOptionalRate = Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)] | None
StrictOptionalCount = StrictNonnegativeInt | None
StrictOptionalNonnegativeFloat = StrictNonnegativeFloat | None
FrozenMetricSupportMap = Annotated[Mapping[str, MetricFieldSupport], PlainSerializer(thaw_json, return_type=dict[str, MetricFieldSupport], when_used="always")]


class DeletionScoresV3(ImmutableContractModel):
    deletion_accuracy: StrictOptionalRate = None
    delete_scope_accuracy: StrictOptionalRate = None
    collateral_damage_rate: StrictOptionalRate = None
    ttl_compliance_rate: StrictOptionalRate = None
    relearn_accuracy: StrictOptionalRate = None
    forgotten_exposure_rate: StrictOptionalRate = None
    forgotten_value_leakage_rate: StrictOptionalRate = None


class HistoricalScoresV3(ImmutableContractModel):
    current_state_accuracy: StrictOptionalRate = None
    previous_state_accuracy: StrictOptionalRate = None
    point_in_time_accuracy: StrictOptionalRate = None
    transition_accuracy: StrictOptionalRate = None
    ordered_history_accuracy: StrictOptionalRate = None
    version_confusion_rate: StrictOptionalRate = None
    historical_support_recall: StrictOptionalRate = None
    historical_distance_accuracy: StrictOptionalRate = None


class SynthesisScoresV3(ImmutableContractModel):
    multi_hop_accuracy: StrictOptionalRate = None
    multi_object_accuracy: StrictOptionalRate = None
    evidence_precision: StrictOptionalRate = None
    evidence_recall: StrictOptionalRate = None
    evidence_f1: StrictOptionalRate = None
    reasoning_support_accuracy: StrictOptionalRate = None
    stale_propagation_rate: StrictOptionalRate = None


CORE_SCORE_LAYER_TYPES = {
    "protocol_scores": ProtocolScores,
    "action_scores": ActionScores,
    "state_scores": StateScores,
    "store_scores": StoreScores,
    "retrieval_scores": RetrievalScores,
    "answer_scores": AnswerScores,
    "system_scores": SystemScores,
    "audit_scores": AuditScores,
    "deletion_scores": DeletionScoresV3,
    "historical_scores": HistoricalScoresV3,
    "synthesis_scores": SynthesisScoresV3,
}
CORE_METRIC_FIELD_PATHS = frozenset(
    f"{layer}.{field}" for layer, model in CORE_SCORE_LAYER_TYPES.items() for field in model.model_fields
)


class ScoreRecordV3(ImmutableContractModel):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    scorer_version: Literal[SCORER_VERSION_V3] = SCORER_VERSION_V3
    metric_registry_version: Literal[METRIC_REGISTRY_VERSION_V3] = METRIC_REGISTRY_VERSION_V3
    task_id: StrictIdentifier
    run_id: StrictIdentifier
    adapter_id: StrictIdentifier
    task_family: StrictIdentifier
    difficulty: Difficulty
    completion_status: CompletionStatus
    supported_metric_fields: FrozenMetricSupportMap = Field(default_factory=FrozenDict)
    protocol_scores: ProtocolScores
    action_scores: ActionScores
    state_scores: StateScores
    store_scores: StoreScores
    retrieval_scores: RetrievalScores
    answer_scores: AnswerScores
    system_scores: SystemScores
    audit_scores: AuditScores
    deletion_scores: DeletionScoresV3
    historical_scores: HistoricalScoresV3
    synthesis_scores: SynthesisScoresV3
    failure_flags: tuple[FailureFlag | FailureFlagV3, ...] = ()
    primary_failure: str | None = None

    @classmethod
    def empty(cls, **values):
        for field, model in CORE_SCORE_LAYER_TYPES.items():
            values.setdefault(field, model())
        return cls(**values)

    @field_validator("supported_metric_fields")
    @classmethod
    def _freeze_support(cls, value):
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _complete_support(self):
        null_paths = {
            f"{layer}.{field}"
            for layer in CORE_SCORE_LAYER_TYPES
            for field, value in getattr(self, layer)
            if value is None
        }
        support_paths = set(self.supported_metric_fields)
        unknown = support_paths - CORE_METRIC_FIELD_PATHS
        if unknown:
            raise ValueError(f"unknown metric support paths: {sorted(unknown)}")
        nonnull = support_paths - null_paths
        if nonnull:
            raise ValueError(f"support entries for non-null metrics: {sorted(nonnull)}")
        missing = null_paths - support_paths
        if missing:
            raise ValueError(f"null metric fields missing support entries: {sorted(missing)}")
        if self.primary_failure is not None and self.primary_failure not in {str(flag) for flag in self.failure_flags}:
            raise ValueError("primary_failure must be one of failure_flags")
        return self


__all__ = ["CORE_METRIC_FIELD_PATHS", "CORE_SCORE_LAYER_TYPES", "DeletionScoresV3", "HistoricalScoresV3", "ScoreRecordV3", "SynthesisScoresV3"]
