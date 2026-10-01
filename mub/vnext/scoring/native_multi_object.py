from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import Field, field_serializer, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool, freeze_mapping
from mub.vnext.contracts.enums import Operation, StringEnum
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, StrictFiniteFloat, object_identity, typed_json_equal
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.native_multi_object import (
    NativeAtomicityV3,
    NativeMultiObjectMutationResultV3,
    NativeProviderEntryObservationV3,
    NativeRetrievalStatusV3,
    NativeRetrievalTraceV3,
)
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.scoring.scorer_v3 import _normalized, _structured_accuracy, _token_f1


class NativeMetricStatusV3(StringEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    RUNTIME_FAILED = "runtime_failed"
    MISSING_ARTIFACT = "missing_artifact"


class NativeMetricSupportV3(ImmutableContractModel):
    status: NativeMetricStatusV3
    reason: str = Field(strict=True, min_length=1)
    detail: str | None = None


_NATIVE_METRIC_FIELDS = (
    "final_state_accuracy",
    "mutation_outcome_accuracy",
    "all_target_retrieval_coverage",
    "provider_order_preserved",
    "provider_id_coverage",
    "provider_score_validity",
    "source_event_linkage_accuracy",
    "atomicity_accuracy",
    "delete_collateral_rate",
    "answer_exact_match",
    "answer_normalized_match",
    "answer_token_f1",
    "answer_structured_field_accuracy",
)

_MISSING_GOLD_ANSWER = object()


class NativeMultiObjectScoreV3(ImmutableContractModel):
    status: NativeMetricStatusV3
    status_detail: str | None = None
    final_state_accuracy: float | None = Field(default=None, ge=0, le=1, strict=True)
    mutation_outcome_accuracy: float | None = Field(default=None, ge=0, le=1, strict=True)
    all_target_retrieval_coverage: float | None = Field(default=None, ge=0, le=1, strict=True)
    provider_order_preserved: float | None = Field(default=None, ge=0, le=1, strict=True)
    provider_id_coverage: float | None = Field(default=None, ge=0, le=1, strict=True)
    provider_score_validity: float | None = Field(default=None, ge=0, le=1, strict=True)
    source_event_linkage_accuracy: float | None = Field(default=None, ge=0, le=1, strict=True)
    atomicity_accuracy: float | None = Field(default=None, ge=0, le=1, strict=True)
    delete_collateral_rate: float | None = Field(default=None, ge=0, le=1, strict=True)
    answer_exact_match: float | None = Field(default=None, ge=0, le=1, strict=True)
    answer_normalized_match: float | None = Field(default=None, ge=0, le=1, strict=True)
    answer_token_f1: float | None = Field(default=None, ge=0, le=1, strict=True)
    answer_structured_field_accuracy: float | None = Field(default=None, ge=0, le=1, strict=True)
    metric_support: Mapping[str, NativeMetricSupportV3]

    @field_validator("metric_support")
    @classmethod
    def _freeze_support(cls, value):
        return freeze_mapping(value)

    @field_serializer("metric_support")
    def _serialize_support(self, value: Mapping[str, NativeMetricSupportV3]) -> dict[str, Any]:
        return {str(key): item for key, item in value.items()}

    @model_validator(mode="after")
    def _support_is_complete(self) -> Self:
        if set(self.metric_support) != {f"native_scores.{field}" for field in _NATIVE_METRIC_FIELDS}:
            raise ValueError("native metric support must cover every native score field")
        if self.status is not NativeMetricStatusV3.SUPPORTED:
            if any(getattr(self, field) is not None for field in _NATIVE_METRIC_FIELDS):
                raise ValueError("unsupported native rows must keep every metric null")
            expected = self.status.value
            if any(item.status is not self.status or item.reason != expected for item in self.metric_support.values()):
                raise ValueError("native unsupported rows require typed null support reasons")
        else:
            for field in _NATIVE_METRIC_FIELDS:
                support = self.metric_support[f"native_scores.{field}"]
                if getattr(self, field) is None:
                    if support.status is not NativeMetricStatusV3.MISSING_ARTIFACT:
                        raise ValueError("null native metrics require missing_artifact support")
                elif support.status is not NativeMetricStatusV3.SUPPORTED or support.reason != "supported":
                    raise ValueError("non-null native metrics require supported support")
        return self

    def metric_values(self) -> tuple[float | None, ...]:
        return tuple(getattr(self, field) for field in _NATIVE_METRIC_FIELDS)

    @property
    def collateral_damage_rate(self) -> float | None:
        return self.delete_collateral_rate

    @property
    def provider_order_accuracy(self) -> float | None:
        return self.provider_order_preserved

    @property
    def provider_entry_id_coverage(self) -> float | None:
        return self.provider_id_coverage

    @property
    def provider_score_accuracy(self) -> float | None:
        return self.provider_score_validity


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _same(left: Any, right: Any) -> bool:
    return typed_json_equal(left, right)


def _state_map(state: Mapping | None) -> dict[str, Any] | None:
    if state is None:
        return None
    result = {}
    for key, value in state.items():
        canonical_id = key.canonical_id if hasattr(key, "canonical_id") else key
        if type(canonical_id) is not str:
            raise ValueError("native state keys must be canonical object IDs or object keys")
        if canonical_id in result:
            raise ValueError("native state keys contain a canonical-ID collision")
        result[canonical_id] = value
    return result


def _state_accuracy(expected_state: Mapping[str, Any] | None, observed_state: Mapping[str, Any] | None) -> float | None:
    expected_mapped = _state_map(expected_state)
    observed_mapped = _state_map(observed_state)
    if expected_mapped is None or observed_mapped is None:
        return None
    if set(expected_mapped) != set(observed_mapped):
        return float(False)
    return float(all(_same(observed_mapped[key], expected_mapped[key]) for key in expected_mapped))


def _normalize_provider_observations(observations: Any) -> tuple[NativeProviderEntryObservationV3, ...] | None:
    if observations is None or isinstance(observations, bool):
        return None
    if isinstance(observations, Mapping):
        if any(name in observations for name in ("entries", "provider_entries", "observations")):
            observations = next(
                observations[name]
                for name in ("entries", "provider_entries", "observations")
                if name in observations
            )
        elif all(name in observations for name in ("provider_entry_id", "order_index", "score")):
            observations = (observations,)
        else:
            ids = observations.get("provider_entry_ids", observations.get("entry_ids"))
            orders = observations.get("order_indices", observations.get("orders"))
            scores = observations.get("scores")
            if ids is None or orders is None or scores is None:
                return None
            try:
                observations = tuple(
                    {"provider_entry_id": entry_id, "order_index": order, "score": score}
                    for entry_id, order, score in zip(ids, orders, scores, strict=True)
                )
            except (TypeError, ValueError):
                return None
    if not isinstance(observations, (list, tuple)):
        return None
    try:
        normalized = tuple(
            item if type(item) is NativeProviderEntryObservationV3
            else NativeProviderEntryObservationV3.model_validate(item, strict=True)
            for item in observations
        )
    except Exception:
        return None
    ids = tuple(item.provider_entry_id for item in normalized)
    if len(ids) != len(set(ids)):
        return None
    return normalized


def _unsupported_score(status: NativeMetricStatusV3, detail: str | None) -> NativeMultiObjectScoreV3:
    support = {
        f"native_scores.{field}": NativeMetricSupportV3(
            status=status, reason=status.value, detail=detail,
        )
        for field in _NATIVE_METRIC_FIELDS
    }
    return NativeMultiObjectScoreV3(
        status=status,
        status_detail=detail,
        metric_support=support,
    )


def score_native_multi_object(
    *,
    expected_state: Mapping[str, Any] | None = None,
    observed_state: Mapping[str, Any] | None = None,
    target_object_keys: tuple[FrozenMemoryObjectKey, ...] = (),
    expected_outcome_statuses: Mapping[str, ExecutionStatusV3] | None = None,
    mutation_result: NativeMultiObjectMutationResultV3 | None = None,
    retrieval_trace: NativeRetrievalTraceV3 | None = None,
    expected_source_event_ids: Mapping[str, tuple[str, ...]] | None = None,
    expected_atomicity: NativeAtomicityV3 | None = None,
    answer_prediction: AnswerPredictionV3 | None = None,
    gold_answer: Any = _MISSING_GOLD_ANSWER,
    provider_observations: Any = None,
    provider_observation_witness: Any = None,
    status: NativeMetricStatusV3 = NativeMetricStatusV3.SUPPORTED,
    status_detail: str | None = None,
) -> NativeMultiObjectScoreV3:
    """Score native manager observations without filling unsupported cells with zero."""
    status = NativeMetricStatusV3(status)
    if status is not NativeMetricStatusV3.SUPPORTED:
        return _unsupported_score(status, status_detail)

    values: dict[str, float | None] = {field: None for field in _NATIVE_METRIC_FIELDS}
    expected_state_mapped = _state_map(expected_state)
    observed_state_mapped = _state_map(observed_state)
    if expected_state_mapped is not None and observed_state_mapped is not None:
        values["final_state_accuracy"] = _state_accuracy(expected_state_mapped, observed_state_mapped)

    target_ids = tuple(object_identity(key) for key in target_object_keys)
    target_canonical_ids = tuple(key.canonical_id for key in target_object_keys)
    if expected_outcome_statuses is not None and mutation_result is not None:
        outcomes = {item.object_key.canonical_id: item for item in mutation_result.outcomes}
        expected_ids = set(expected_outcome_statuses)
        if expected_ids != set(outcomes):
            values["mutation_outcome_accuracy"] = 0.0
        else:
            observations = [
                float(outcomes[canonical_id].status is ExecutionStatusV3(expected_status))
                for canonical_id, expected_status in expected_outcome_statuses.items()
            ]
            values["mutation_outcome_accuracy"] = _mean(observations) if observations else 1.0

    normalized_provider_observations = _normalize_provider_observations(
        provider_observations if provider_observations is not None else provider_observation_witness
    )

    if retrieval_trace is not None and retrieval_trace.status not in {
        NativeRetrievalStatusV3.FAILED,
        NativeRetrievalStatusV3.UNSUPPORTED,
    }:
        entry_ids = [entry.provider_entry_id for entry in retrieval_trace.entries]
        entry_id_set = set(entry_ids)
        retrieved_targets = {object_identity(entry.object_key) for entry in retrieval_trace.entries}
        values["all_target_retrieval_coverage"] = (
            1.0 if not target_ids else sum(identity in retrieved_targets for identity in target_ids) / len(target_ids)
        )
        if normalized_provider_observations is not None:
            trace_ids = tuple(entry.provider_entry_id for entry in retrieval_trace.entries)
            observed_ids = tuple(item.provider_entry_id for item in normalized_provider_observations)
            trace_orders = tuple(entry.order_index for entry in retrieval_trace.entries)
            observed_orders = tuple(item.order_index for item in normalized_provider_observations)
            trace_scores = tuple(entry.score for entry in retrieval_trace.entries)
            observed_scores = tuple(item.score for item in normalized_provider_observations)
            values["provider_order_preserved"] = float(
                len(trace_ids) == len(observed_ids)
                and tuple(zip(trace_ids, trace_orders)) == tuple(zip(observed_ids, observed_orders))
            )
            values["provider_id_coverage"] = float(
                len(trace_ids) == len(observed_ids) and set(trace_ids) == set(observed_ids)
            )
            values["provider_score_validity"] = float(
                len(trace_ids) == len(observed_ids)
                and tuple(zip(trace_ids, trace_scores)) == tuple(zip(observed_ids, observed_scores))
            )
        if expected_source_event_ids is not None:
            by_object = {entry.object_key.canonical_id: entry for entry in retrieval_trace.entries}
            linkage = []
            for canonical_id, expected_events in expected_source_event_ids.items():
                entry = by_object.get(canonical_id)
                linkage.append(float(entry is not None and tuple(entry.source_event_ids) == tuple(expected_events)))
            values["source_event_linkage_accuracy"] = _mean(linkage)

    if expected_atomicity is not None and mutation_result is not None:
        values["atomicity_accuracy"] = float(mutation_result.atomicity is expected_atomicity)

    if expected_state_mapped is not None and observed_state_mapped is not None:
        protected = [canonical_id for canonical_id in expected_state_mapped if canonical_id not in target_canonical_ids]
        unexpected = [canonical_id for canonical_id in observed_state_mapped if canonical_id not in expected_state_mapped]
        collateral = tuple(dict.fromkeys((*protected, *unexpected)))
        if collateral:
            values["delete_collateral_rate"] = _mean([
                float(
                    canonical_id not in observed_state_mapped
                    or canonical_id not in expected_state_mapped
                    or not _same(observed_state_mapped[canonical_id], expected_state_mapped[canonical_id])
                )
                for canonical_id in collateral
            ])
        elif target_object_keys:
            values["delete_collateral_rate"] = 0.0

    if answer_prediction is not None and gold_answer is not _MISSING_GOLD_ANSWER:
        valid = answer_prediction.format_valid and answer_prediction.disposition.value == "answered"
        values["answer_exact_match"] = float(valid and _same(answer_prediction.parsed_answer, gold_answer))
        values["answer_normalized_match"] = float(valid and _same(_normalized(answer_prediction.parsed_answer), _normalized(gold_answer)))
        values["answer_token_f1"] = _token_f1(answer_prediction.parsed_answer, gold_answer) if valid else 0.0
        values["answer_structured_field_accuracy"] = _structured_accuracy(answer_prediction.parsed_answer, gold_answer) if valid else 0.0

    support = {}
    for field in _NATIVE_METRIC_FIELDS:
        metric = values[field]
        support[f"native_scores.{field}"] = NativeMetricSupportV3(
            status=NativeMetricStatusV3.SUPPORTED if metric is not None else NativeMetricStatusV3.MISSING_ARTIFACT,
            reason="supported" if metric is not None else "missing_artifact",
            detail=None if metric is not None else "required native observation was not supplied",
        )
    return NativeMultiObjectScoreV3(
        status=status,
        status_detail=status_detail,
        metric_support=support,
        **values,
    )


def score_native_multi_object_v3(**kwargs) -> NativeMultiObjectScoreV3:
    return score_native_multi_object(**kwargs)


__all__ = [
    "NativeMetricStatusV3",
    "NativeMetricSupportV3",
    "NativeMultiObjectScoreV3",
    "score_native_multi_object",
    "score_native_multi_object_v3",
]
