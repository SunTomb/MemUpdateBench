from __future__ import annotations

import pytest

from mub.vnext.contracts import (
    AnswerDisposition,
    AnswerSchema,
    CompletionStatus,
    EvaluationMode,
    MemoryObjectKey,
    QueryType,
    ReferenceCandidate,
    ReferenceResolutionStatus,
    SurfaceReference,
    SupportReason,
    TaskFamily,
)
from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.manifest import ScorerConfig
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.task import CanonicalAnswer, MemUpdateTask, MemoryQuery
from mub.vnext.scoring.registry import METRIC_REGISTRY
from mub.vnext.scoring.scorer import score_task


_METRIC = "answer_scores.reference_resolution_accuracy"


def _capabilities(**overrides) -> AdapterCapabilities:
    return AdapterCapabilities(**overrides)


def _config() -> ScorerConfig:
    return ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=(_METRIC,),
    )


def _complete_actions(run: TaskRunRecord) -> dict:
    payload = run.model_dump(mode="python")
    key = payload["parsed_actions"][0]["target_object_key"]
    payload["parsed_actions"].insert(
        0,
        {
            "event_id": "event_0",
            "operation": "ADD",
            "target_object_key": key,
            "value": "Dalian",
            "format_valid": True,
            "execution_status": "succeeded",
            "fallback_used": False,
            "error_flags": [],
            "raw_output": "ADD friend:alex.location = Dalian",
            "latency_ms": 1.0,
        },
    )
    return payload


def _reference_task(
    base: MemUpdateTask,
    status: ReferenceResolutionStatus,
    *,
    keep_ordinary: bool = False,
) -> MemUpdateTask:
    first_key = base.target_objects[0]
    second_key = MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="colleague:alex",
        attribute=first_key.attribute,
        subkey=None,
    )
    query_id = "query_reference" if keep_ordinary else "query_0"
    candidate_ids = ["candidate_current"]
    candidates = [
        ReferenceCandidate(candidate_id="candidate_current", object_key=first_key)
    ]
    target_objects = list(base.target_objects)
    if status is ReferenceResolutionStatus.AMBIGUOUS:
        candidate_ids.append("candidate_other")
        candidates.append(
            ReferenceCandidate(candidate_id="candidate_other", object_key=second_key)
        )
        target_objects.append(second_key)
    referenced_ids = [] if status is ReferenceResolutionStatus.NO_MATCH else candidate_ids
    query = MemoryQuery(
        query_id=query_id,
        query_type=QueryType.UNRESOLVED_REFERENCE,
        text="Where does Alex live?",
        target_object_keys=[],
        reference_candidates=candidates,
        surface_references=[
            SurfaceReference(
                reference_id="reference_alex",
                surface_text="Alex",
                normalized_text="alex",
                candidate_ids=referenced_ids,
            )
        ],
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    canonical = (
        CanonicalAnswer(
            disposition=AnswerDisposition.ANSWERED,
            resolution_status=ReferenceResolutionStatus.UNIQUE,
            selected_candidate_ids=["candidate_current"],
            abstention_reason=None,
            value="Qingdao",
        )
        if status is ReferenceResolutionStatus.UNIQUE
        else CanonicalAnswer(
            disposition=AnswerDisposition.ABSTAINED,
            resolution_status=status,
            selected_candidate_ids=[],
            abstention_reason=status.value,
            value=None,
        )
    )
    gold_update = {
        "canonical_answers": {query_id: canonical},
    }
    if not keep_ordinary:
        gold_update.update({"gold_answers": {}, "acceptable_answers": {}})
    return base.model_copy(
        update={
            "task_family": TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value,
            "target_objects": target_objects,
            "queries": [*base.queries, query] if keep_ordinary else [query],
            "gold": base.gold.model_copy(update=gold_update),
        }
    )


def _reference_run(
    base: TaskRunRecord,
    disposition: AnswerDisposition,
    *,
    value=None,
    keep_ordinary: bool = False,
    status: CompletionStatus = CompletionStatus.COMPLETED,
) -> TaskRunRecord:
    payload = _complete_actions(base)
    query_id = "query_reference" if keep_ordinary else "query_0"
    prediction = {
        "query_id": query_id,
        "raw_output": "ABSTAIN" if disposition is AnswerDisposition.ABSTAINED else str(value or ""),
        "disposition": disposition.value,
        "parsed_answer": value if disposition is AnswerDisposition.ANSWERED else None,
        "cited_event_ids": [],
        "cited_entry_ids": [],
        "format_valid": disposition is not AnswerDisposition.UNAVAILABLE,
        "error_flags": [],
        "latency_ms": 1.0,
        "usage": {},
    }
    payload["answer_predictions"] = (
        [*payload["answer_predictions"], prediction]
        if keep_ordinary
        else [prediction]
    )
    payload["completion_status"] = status.value
    if status is not CompletionStatus.COMPLETED:
        payload["exceptions"] = [{"type": "runtime_error", "message": "failed"}]
    return TaskRunRecord.model_validate(payload)


def test_reference_resolution_metric_is_family_c_only_without_new_capabilities() -> None:
    definition = METRIC_REGISTRY[_METRIC]

    assert definition.applicable_task_families == (
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value,
    )
    assert definition.required_adapter_capabilities == ()
    assert "disposition" in definition.numerator_definition.casefold()
    assert "unavailable" in definition.unsupported_value_policy.casefold()


@pytest.mark.parametrize(
    "status",
    [ReferenceResolutionStatus.AMBIGUOUS, ReferenceResolutionStatus.NO_MATCH],
)
def test_ambiguous_and_no_match_score_correct_explicit_abstention(
    status, make_task, make_task_run
) -> None:
    score = score_task(
        _reference_task(make_task(), status),
        _reference_run(make_task_run(), AnswerDisposition.ABSTAINED),
        _capabilities(),
        _config(),
    )

    assert score.answer_scores.reference_resolution_accuracy == 1.0
    assert "wrong_reference_guess" not in score.failure_flags


def test_answering_abstention_gold_is_wrong_reference_guess(
    make_task, make_task_run
) -> None:
    score = score_task(
        _reference_task(make_task(), ReferenceResolutionStatus.AMBIGUOUS),
        _reference_run(
            make_task_run(),
            AnswerDisposition.ANSWERED,
            value="Qingdao",
        ),
        _capabilities(),
        _config(),
    )

    assert score.answer_scores.reference_resolution_accuracy == 0.0
    assert "wrong_reference_guess" in score.failure_flags
    assert score.primary_failure == "wrong_reference_guess"


def test_unique_reference_requires_explicit_answered_exact_value(
    make_task, make_task_run
) -> None:
    score = score_task(
        _reference_task(make_task(), ReferenceResolutionStatus.UNIQUE),
        _reference_run(
            make_task_run(),
            AnswerDisposition.ANSWERED,
            value="Qingdao",
        ),
        _capabilities(),
        _config(),
    )

    assert score.answer_scores.reference_resolution_accuracy == 1.0
    assert score.failure_flags == ()


def test_abstaining_on_unique_reference_is_unjustified_abstention(
    make_task, make_task_run
) -> None:
    score = score_task(
        _reference_task(make_task(), ReferenceResolutionStatus.UNIQUE),
        _reference_run(make_task_run(), AnswerDisposition.ABSTAINED),
        _capabilities(),
        _config(),
    )

    assert score.answer_scores.reference_resolution_accuracy == 0.0
    assert "unjustified_abstention" in score.failure_flags
    assert score.primary_failure == "unjustified_abstention"


def test_unavailable_and_missing_reference_predictions_remain_missing_artifacts(
    make_task, make_task_run
) -> None:
    task = _reference_task(make_task(), ReferenceResolutionStatus.AMBIGUOUS)
    unavailable = score_task(
        task,
        _reference_run(make_task_run(), AnswerDisposition.UNAVAILABLE),
        _capabilities(),
        _config(),
    )
    missing_payload = _complete_actions(make_task_run())
    missing_payload["answer_predictions"] = []
    missing = score_task(
        task,
        TaskRunRecord.model_validate(missing_payload),
        _capabilities(),
        _config(),
    )

    for score in (unavailable, missing):
        assert score.answer_scores.reference_resolution_accuracy is None
        assert score.supported_metric_fields[_METRIC].reason is SupportReason.MISSING_ARTIFACT
        assert "wrong_reference_guess" not in score.failure_flags
        assert "unjustified_abstention" not in score.failure_flags
        assert score.primary_failure is None


def test_runtime_failure_preserves_runtime_failed_support_for_reference_metric(
    make_task, make_task_run
) -> None:
    score = score_task(
        _reference_task(make_task(), ReferenceResolutionStatus.UNIQUE),
        _reference_run(
            make_task_run(),
            AnswerDisposition.UNAVAILABLE,
            status=CompletionStatus.PARTIAL,
        ),
        _capabilities(),
        _config(),
    )

    assert score.answer_scores.reference_resolution_accuracy is None
    assert score.supported_metric_fields[_METRIC].reason is SupportReason.RUNTIME_FAILED
    assert score.primary_failure == "system_exception"


def test_unresolved_rows_are_excluded_from_ordinary_answer_denominators(
    make_task, make_task_run
) -> None:
    task = _reference_task(
        make_task(),
        ReferenceResolutionStatus.AMBIGUOUS,
        keep_ordinary=True,
    )
    run = _reference_run(
        make_task_run(),
        AnswerDisposition.ABSTAINED,
        keep_ordinary=True,
    )
    config = ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=(
            "answer_scores.exact_match",
            "answer_scores.normalized_match",
            "answer_scores.token_f1",
            "answer_scores.stale_copied",
            "answer_scores.distractor_copied",
            _METRIC,
        ),
    )

    score = score_task(
        task,
        run,
        _capabilities(exports_retrieval_ids=True),
        config,
    )

    assert score.answer_scores.exact_match == 1.0
    assert score.answer_scores.normalized_match == 1.0
    assert score.answer_scores.token_f1 == 1.0
    assert score.answer_scores.stale_copied is None
    assert score.supported_metric_fields[
        "answer_scores.stale_copied"
    ].reason is SupportReason.NOT_APPLICABLE
    assert score.answer_scores.distractor_copied == 0.0
    assert score.answer_scores.reference_resolution_accuracy == 1.0


def test_reference_metric_is_not_applicable_to_ordinary_queries(
    make_task, make_task_run
) -> None:
    config = ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=(_METRIC, "answer_scores.exact_match"),
    )
    score = score_task(
        make_task(),
        _complete_run(make_task_run()),
        _capabilities(),
        config,
    )

    assert score.answer_scores.reference_resolution_accuracy is None
    assert score.supported_metric_fields[_METRIC].reason is SupportReason.NOT_APPLICABLE
    assert score.answer_scores.exact_match == 1.0


def _complete_run(run: TaskRunRecord) -> TaskRunRecord:
    return TaskRunRecord.model_validate(_complete_actions(run))
