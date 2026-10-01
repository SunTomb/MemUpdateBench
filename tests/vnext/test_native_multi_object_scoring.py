from __future__ import annotations

import pytest

from mub.vnext.contracts.enums import AnswerDisposition, Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.native_multi_object import (
    NativeAtomicityV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationResultV3,
    NativeObjectOutcomeV3,
    NativeRetrievalEntryV3,
    NativeRetrievalRequestV3,
    NativeRetrievalTraceV3,
)
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.scoring.native_multi_object import (
    NativeMetricStatusV3,
    NativeMultiObjectScoreV3,
    score_native_multi_object,
)


def key(namespace: str, subkey: str):
    return FrozenMemoryObjectKey(
        object_type="slot", namespace=namespace, entity="entity",
        attribute="attribute", subkey=subkey,
    )


def prov():
    return {"kind": NativeDataProvenanceV3.PROVIDER_NATIVE, "provider_owned": True}


def test_native_scoring_reports_state_mutation_retrieval_provenance_atomicity_and_answer_independently():
    first = key("ns-a", "one")
    second = key("ns-b", "two")
    outcomes = (
        NativeObjectOutcomeV3(
            event_id="event-1", object_key=first, operation=Operation.UPDATE,
            status=ExecutionStatusV3.EXECUTED, provider_entry_id="p-1",
            source_event_ids=("event-1",), write_count=1, provenance=prov(),
        ),
        NativeObjectOutcomeV3(
            event_id="event-1", object_key=second, operation=Operation.UPDATE,
            status=ExecutionStatusV3.EXECUTED, provider_entry_id="p-2",
            source_event_ids=("event-1",), write_count=1, provenance=prov(),
        ),
    )
    mutation = NativeMultiObjectMutationResultV3(
        request_id="request-event-1", event_id="event-1", operation=Operation.UPDATE,
        outcomes=outcomes, atomicity=NativeAtomicityV3.ATOMIC, provenance=prov(),
    )
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=2, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-2", object_key=second, provider_payload={"value": "v2"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=2, order_index=1, score=0.8, source_event_ids=("event-1",), provenance=prov(),
            ),
        ), provenance=prov(),
    )
    answer = AnswerPredictionV3(
        query_id="q-1", raw_output='["v1", "v2"]', disposition=AnswerDisposition.ANSWERED,
        parsed_answer=("v1", "v2"), format_valid=True,
    )
    score = score_native_multi_object(
        expected_state={first.canonical_id: "v1", second.canonical_id: "v2"},
        observed_state={first.canonical_id: "v1", second.canonical_id: "v2"},
        expected_outcome_statuses={first.canonical_id: ExecutionStatusV3.EXECUTED, second.canonical_id: ExecutionStatusV3.EXECUTED},
        mutation_result=mutation,
        target_object_keys=(first, second),
        retrieval_trace=trace,
        expected_source_event_ids={first.canonical_id: ("event-1",), second.canonical_id: ("event-1",)},
        expected_atomicity=NativeAtomicityV3.ATOMIC,
        answer_prediction=answer,
        gold_answer=("v1", "v2"),
    )
    assert isinstance(score, NativeMultiObjectScoreV3)
    assert score.status is NativeMetricStatusV3.SUPPORTED
    assert score.final_state_accuracy == 1.0
    assert score.mutation_outcome_accuracy == 1.0
    assert score.all_target_retrieval_coverage == 1.0
    assert score.provider_order_preserved is None
    assert score.provider_id_coverage is None
    assert score.provider_score_validity is None
    assert score.metric_support["native_scores.provider_order_preserved"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.metric_support["native_scores.provider_id_coverage"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.metric_support["native_scores.provider_score_validity"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.source_event_linkage_accuracy == 1.0
    assert score.atomicity_accuracy == 1.0
    assert score.answer_exact_match == 1.0
    assert score.answer_normalized_match == 1.0
    assert score.answer_token_f1 == 1.0


def test_native_scoring_detects_delete_collateral_and_missing_target_without_zeroing_unsupported():
    target = key("ns", "target")
    collateral = key("ns", "collateral")
    score = score_native_multi_object(
        expected_state={target.canonical_id: None, collateral.canonical_id: "keep"},
        observed_state={target.canonical_id: None},
        target_object_keys=(target,),
        status=NativeMetricStatusV3.SUPPORTED,
    )
    assert score.final_state_accuracy == 0.0
    assert score.delete_collateral_rate == 1.0

    unsupported = score_native_multi_object(
        expected_state={target.canonical_id: "v"}, observed_state={},
        target_object_keys=(target,), status=NativeMetricStatusV3.UNSUPPORTED,
        status_detail="multi-object query unsupported",
    )
    assert unsupported.status is NativeMetricStatusV3.UNSUPPORTED
    assert unsupported.final_state_accuracy is None
    assert unsupported.answer_exact_match is None
    assert unsupported.metric_support["native_scores.final_state_accuracy"].reason == "unsupported"


def test_native_scoring_keeps_unavailable_and_runtime_failed_metrics_null_with_typed_reasons():
    for status, reason in (
        (NativeMetricStatusV3.UNAVAILABLE, "unavailable"),
        (NativeMetricStatusV3.RUNTIME_FAILED, "runtime_failed"),
    ):
        score = score_native_multi_object(
            expected_state={}, observed_state={}, target_object_keys=(), status=status,
        )
        assert all(value is None for value in score.metric_values())
        assert all(item.reason == reason for item in score.metric_support.values())


def test_native_scoring_uses_explicit_provider_entry_observations():
    first = key("ns-a", "one")
    second = key("ns-b", "two")
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=2, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-2", object_key=second, provider_payload={"value": "v2"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=2, order_index=1, score=0.8, source_event_ids=("event-1",), provenance=prov(),
            ),
        ), provenance=prov(),
    )
    observations = (
        {"provider_entry_id": "p-2", "order_index": 0, "score": 0.9},
        {"provider_entry_id": "p-1", "order_index": 1, "score": 0.8},
    )
    score = score_native_multi_object(
        target_object_keys=(first, second), retrieval_trace=trace,
        provider_observations=observations,
    )
    assert score.provider_order_preserved == 1.0
    assert score.provider_id_coverage == 1.0
    assert score.provider_score_validity == 1.0

    mismatched = score_native_multi_object(
        target_object_keys=(first, second), retrieval_trace=trace,
        provider_observations=(
            {"provider_entry_id": "p-2", "order_index": 0, "score": 0.7},
            {"provider_entry_id": "p-1", "order_index": 1, "score": 0.8},
        ),
    )
    assert mismatched.provider_order_preserved == 1.0
    assert mismatched.provider_id_coverage == 1.0
    assert mismatched.provider_score_validity == 0.0


def test_native_scoring_does_not_treat_bare_provider_witness_as_observation():
    first = key("ns-a", "one")
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=1, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
        ), provenance=prov(),
    )
    score = score_native_multi_object(
        target_object_keys=(first,), retrieval_trace=trace,
        provider_observation_witness=True,
    )
    assert score.provider_order_preserved is None
    assert score.provider_id_coverage is None
    assert score.provider_score_validity is None


def test_native_scoring_rejects_canonical_id_collisions_in_state_maps():
    object_key = key("ns", "one")
    colliding_state = {object_key: "object-key-value", object_key.canonical_id: "string-value"}
    with pytest.raises(ValueError, match="canonical"):
        score_native_multi_object(expected_state=colliding_state, observed_state={})


def test_native_scoring_scores_explicit_null_gold_answer():
    answer = AnswerPredictionV3.model_construct(
        query_id="q-1", raw_output="null", disposition=AnswerDisposition.ANSWERED,
        parsed_answer=None, format_valid=True,
    )
    score = score_native_multi_object(answer_prediction=answer, gold_answer=None)
    assert score.answer_exact_match == 1.0
    assert score.answer_normalized_match == 1.0
    assert score.answer_token_f1 == 1.0
    assert score.answer_structured_field_accuracy == 1.0


def test_native_scoring_penalizes_unexpected_mutation_outcomes():
    first = key("ns-a", "one")
    second = key("ns-b", "two")
    outcomes = tuple(
        NativeObjectOutcomeV3(
            event_id="event-1", object_key=object_key, operation=Operation.UPDATE,
            status=ExecutionStatusV3.EXECUTED, provider_entry_id=f"p-{index}",
            source_event_ids=("event-1",), write_count=1, provenance=prov(),
        )
        for index, object_key in enumerate((first, second), start=1)
    )
    mutation = NativeMultiObjectMutationResultV3(
        request_id="request-event-1", event_id="event-1", operation=Operation.UPDATE,
        outcomes=outcomes, atomicity=NativeAtomicityV3.ATOMIC, provenance=prov(),
    )
    score = score_native_multi_object(
        expected_outcome_statuses={first.canonical_id: ExecutionStatusV3.EXECUTED},
        mutation_result=mutation,
    )
    assert score.mutation_outcome_accuracy == 0.0

def test_native_scoring_provider_metrics_null_without_witness():
    first = key("ns-a", "one")
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=1, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
        ),
        provenance=prov(),
    )
    score = score_native_multi_object(
        expected_state={first.canonical_id: "v1"},
        observed_state={first.canonical_id: "v1"},
        target_object_keys=(first,),
        retrieval_trace=trace,
        provider_observation_witness=None,
    )
    assert score.all_target_retrieval_coverage == 1.0
    assert score.provider_order_preserved is None
    assert score.provider_id_coverage is None
    assert score.provider_score_validity is None
    assert score.metric_support["native_scores.provider_order_preserved"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.metric_support["native_scores.provider_id_coverage"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.metric_support["native_scores.provider_score_validity"].status is NativeMetricStatusV3.MISSING_ARTIFACT


def test_native_scoring_handles_none_state_map_without_crash():
    score = score_native_multi_object(
        expected_state=None,
        observed_state=None,
        target_object_keys=(),
    )
    assert score.final_state_accuracy is None
    assert score.delete_collateral_rate is None
    assert score.metric_support["native_scores.final_state_accuracy"].status is NativeMetricStatusV3.MISSING_ARTIFACT
    assert score.metric_support["native_scores.delete_collateral_rate"].status is NativeMetricStatusV3.MISSING_ARTIFACT


def test_native_scoring_metric_support_model_dump_json_mapping():
    score = score_native_multi_object(
        expected_state={},
        observed_state={},
        target_object_keys=(),
    )
    dumped = score.model_dump(mode="json")
    assert isinstance(dumped["metric_support"], dict)
    assert isinstance(dumped["metric_support"]["native_scores.final_state_accuracy"], dict)
    assert dumped["metric_support"]["native_scores.final_state_accuracy"]["status"] == "supported"


@pytest.mark.parametrize("invalid_witness", [
    {},
    {"random": "data"},
    {"direct_provider_observed": True},
    {"direct_provider_observed": "true"},
    {"kind": "observed", "provider_owned": "true", "direct_provider_observed": True},
    "truthy_string",
    1,
    1.0,
    False,
])
def test_native_scoring_rejects_malformed_empty_or_non_boolean_witness(invalid_witness):
    first = key("ns-a", "one")
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=1, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
        ),
        provenance=prov(),
    )
    score = score_native_multi_object(
        expected_state={first.canonical_id: "v1"},
        observed_state={first.canonical_id: "v1"},
        target_object_keys=(first,),
        retrieval_trace=trace,
        provider_observation_witness=invalid_witness,
    )
    assert score.provider_order_preserved is None
    assert score.provider_id_coverage is None
    assert score.provider_score_validity is None
    assert score.metric_support["native_scores.provider_order_preserved"].status is NativeMetricStatusV3.MISSING_ARTIFACT


@pytest.mark.parametrize("valid_witness", [
    True,
    {"kind": "observed", "provider_owned": True, "direct_provider_observed": True},
    {"kind": "provider_native", "provider_owned": True, "direct_provider_observed": True},
])
def test_native_scoring_legacy_witness_never_enables_metrics(valid_witness):
    first = key("ns-a", "one")
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=1, namespace="run"),
        entries=(
            NativeRetrievalEntryV3(
                provider_entry_id="p-1", object_key=first, provider_payload={"value": "v1"},
                rank=1, order_index=0, score=0.9, source_event_ids=("event-1",), provenance=prov(),
            ),
        ),
        provenance=prov(),
    )
    score = score_native_multi_object(
        expected_state={first.canonical_id: "v1"},
        observed_state={first.canonical_id: "v1"},
        target_object_keys=(first,),
        retrieval_trace=trace,
        provider_observation_witness=valid_witness,
    )
    assert score.provider_order_preserved is None
    assert score.provider_id_coverage is None
    assert score.provider_score_validity is None
