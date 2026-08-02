from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, PlainSerializer, field_serializer, field_validator, model_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, MetricFieldSupport, StrictBool, StrictNonnegativeFloat, StrictNonnegativeInt, freeze_mapping, thaw_json
from mub.vnext.contracts.enums import CompletionStatus, Difficulty
from mub.vnext.contracts.score import ActionScores, AnswerScores, AuditScores, ProtocolScores, RetrievalScores, StateScores, StoreScores, SystemScores
from mub.vnext.failure import FAILURE_FLAGS, PRIMARY_FAILURE_PRECEDENCE, FailureFlag
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.contracts.v3.enums import FailureFlagV3
from mub.vnext.contracts.v3.version import METRIC_REGISTRY_VERSION_V3, PRIMARY_FAILURE_PRECEDENCE_VERSION_V3, SCHEMA_VERSION_V3, SCORER_VERSION_V3

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


V3_FAILURE_FLAGS = FAILURE_FLAGS + tuple(flag.value for flag in FailureFlagV3)
V3_PRIMARY_FAILURE_PRECEDENCE = (
    "system_exception", "invalid_action_format", "unsupported_action",
    "wrong_operation", "wrong_entity", "wrong_attribute", "wrong_value", "false_write", "missed_update",
    "wrong_delete_scope", "collateral_mutation", "ttl_violation", "forgotten_value_exposed",
    "version_confusion", "evidence_linkage_error", "stale_propagation",
    "collateral_corruption", "deletion_failure", "current_state_missing", "stale_retained",
    "current_not_retrieved", "stale_retrieved", "distractor_retrieved",
    "wrong_reference_guess", "unjustified_abstention", "stale_copied", "distractor_copied",
    "gold_retrieved_wrong_answer", "answer_format_only",
)


def _canonicalize_v3_flags(value) -> tuple[str, ...]:
    if value is None or type(value) in {str, bytes} or isinstance(value, Mapping):
        raise ValueError("failure_flags must be a non-string iterable")
    supplied = []
    for flag in value:
        if isinstance(flag, (FailureFlag, FailureFlagV3)):
            flag = flag.value
        if type(flag) is not str:
            raise ValueError("failure flags must be exact built-in strings")
        supplied.append(flag)
    unknown = set(supplied) - set(V3_FAILURE_FLAGS)
    if unknown:
        raise ValueError(f"unknown failure flags: {sorted(unknown)}")
    present = set(supplied)
    return tuple(flag for flag in V3_FAILURE_FLAGS if flag in present)


def _v3_primary(flags) -> str:
    present = {flag.value if isinstance(flag, (FailureFlag, FailureFlagV3)) else flag for flag in flags}
    return next((flag for flag in V3_PRIMARY_FAILURE_PRECEDENCE if flag in present), "correct")


class ScorerConfigV3(ImmutableContractModel):
    scorer_version: Literal[SCORER_VERSION_V3] = SCORER_VERSION_V3
    metric_registry_version: Literal[METRIC_REGISTRY_VERSION_V3] = METRIC_REGISTRY_VERSION_V3
    primary_failure_precedence_version: Literal[PRIMARY_FAILURE_PRECEDENCE_VERSION_V3] = PRIMARY_FAILURE_PRECEDENCE_VERSION_V3
    value_normalization_profile: Literal["typed_exact_v1"] = "typed_exact_v1"
    answer_normalization_profile: Literal["normalized_exact_v1"] = "normalized_exact_v1"
    requested_metric_fields: tuple[str, ...] = ()
    strict_capability_check: StrictBool = True

    @field_validator("requested_metric_fields", mode="before")
    @classmethod
    def _canonical_metrics(cls, value):
        if type(value) not in {list, tuple, set, frozenset}:
            raise ValueError("requested_metric_fields must be tuple-like")
        if any(type(item) is not str for item in value):
            raise ValueError("metric paths must be exact built-in strings")
        supplied = tuple(value)
        if len(supplied) != len(set(supplied)):
            raise ValueError("requested_metric_fields must be unique")
        unknown = set(supplied) - CORE_METRIC_FIELD_PATHS
        if unknown:
            raise ValueError(f"unknown requested metric fields: {sorted(unknown)}")
        return tuple(sorted(supplied))

    @property
    def configuration_hash(self) -> str:
        payload = self.model_dump(mode="json")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class ScoreRecordV3(ImmutableContractModel):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    scorer_version: Literal[SCORER_VERSION_V3] = SCORER_VERSION_V3
    metric_registry_version: Literal[METRIC_REGISTRY_VERSION_V3] = METRIC_REGISTRY_VERSION_V3
    primary_failure_precedence_version: Literal[PRIMARY_FAILURE_PRECEDENCE_VERSION_V3] = PRIMARY_FAILURE_PRECEDENCE_VERSION_V3
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

    @field_validator("failure_flags", mode="before")
    @classmethod
    def _canonicalize_flags(cls, value):
        return _canonicalize_v3_flags(value)

    @field_serializer("failure_flags", when_used="always")
    def _serialize_flags(self, value):
        return tuple(flag.value if isinstance(flag, (FailureFlag, FailureFlagV3)) else flag for flag in value)

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
        expected_primary = _v3_primary(self.failure_flags)
        if self.primary_failure is None:
            object.__setattr__(self, "primary_failure", expected_primary)
        elif self.primary_failure != expected_primary:
            raise ValueError(f"primary_failure must equal {expected_primary!r}")
        return self


__all__ = ["CORE_METRIC_FIELD_PATHS", "CORE_SCORE_LAYER_TYPES", "DeletionScoresV3", "HistoricalScoresV3", "ScoreRecordV3", "ScorerConfigV3", "SynthesisScoresV3", "V3_FAILURE_FLAGS", "V3_PRIMARY_FAILURE_PRECEDENCE"]
