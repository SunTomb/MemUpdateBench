from __future__ import annotations

from mub.vnext.contracts import AnswerDisposition
from mub.vnext.contracts.adapter import AnswerResult
from mub.vnext.runtime.engine import normalize_answer_result
from mub.vnext.runtime.run import normalize_answer_results


def test_runtime_normalizes_only_explicit_answer_dispositions() -> None:
    answered = normalize_answer_result(
        AnswerResult(
            query_id="answered",
            raw_output="Qingdao",
            disposition=AnswerDisposition.ANSWERED,
            value="Qingdao",
        )
    )
    abstained = normalize_answer_result(
        AnswerResult(
            query_id="abstained",
            raw_output="I cannot resolve that reference.",
            disposition=AnswerDisposition.ABSTAINED,
        )
    )
    unavailable = normalize_answer_result(
        AnswerResult(
            query_id="unavailable",
            raw_output="",
            disposition=AnswerDisposition.UNAVAILABLE,
            error={"type": "adapter_error"},
        )
    )

    assert answered.disposition is AnswerDisposition.ANSWERED
    assert answered.parsed_answer == "Qingdao"
    assert answered.format_valid is True
    assert abstained.disposition is AnswerDisposition.ABSTAINED
    assert abstained.parsed_answer is None
    assert abstained.format_valid is True
    assert unavailable.disposition is AnswerDisposition.UNAVAILABLE
    assert unavailable.parsed_answer is None
    assert unavailable.format_valid is False


def test_runtime_does_not_infer_abstention_from_none_or_free_form_text() -> None:
    for raw_output in (
        "I do not know.",
        "NOOP",
        "The target is absent.",
        "DELETE default|friend:alex|location|~",
    ):
        result = AnswerResult(
            query_id=f"query_{len(raw_output)}",
            raw_output=raw_output,
            disposition=AnswerDisposition.ANSWERED,
            value=None,
            usage={},
            cost=None,
            latency_ms=None,
            error=None,
        )
        prediction = normalize_answer_result(
            result,
            parsed_answer=None,
            format_valid=False,
        )
        assert prediction.disposition is AnswerDisposition.ANSWERED
        assert prediction.parsed_answer is None
        assert prediction.format_valid is False


def test_legacy_none_answer_payload_is_not_reinterpreted_as_abstention() -> None:
    result = AnswerResult.model_validate(
        {
            "query_id": "legacy_none",
            "raw_output": "I do not know.",
            "value": None,
        }
    )

    prediction = normalize_answer_result(
        result,
        parsed_answer=None,
        format_valid=False,
    )

    assert result.disposition is AnswerDisposition.ANSWERED
    assert prediction.disposition is AnswerDisposition.ANSWERED
    assert prediction.parsed_answer is None
    assert prediction.format_valid is False


def test_explicit_unavailable_and_abstained_none_answers_remain_distinct() -> None:
    abstained = normalize_answer_result(
        AnswerResult(
            query_id="abstained",
            raw_output="I cannot resolve that reference.",
            disposition=AnswerDisposition.ABSTAINED,
        )
    )
    unavailable = normalize_answer_result(
        AnswerResult(
            query_id="unavailable",
            raw_output="",
            disposition=AnswerDisposition.UNAVAILABLE,
        )
    )

    assert abstained.parsed_answer is unavailable.parsed_answer is None
    assert abstained.disposition is AnswerDisposition.ABSTAINED
    assert unavailable.disposition is AnswerDisposition.UNAVAILABLE
    assert abstained.format_valid is True
    assert unavailable.format_valid is False


def test_runtime_propagates_parser_artifacts_without_reinterpreting_missing_rows() -> None:
    results = [
        AnswerResult(query_id="query_present", raw_output="Qingdao"),
    ]

    predictions = normalize_answer_results(
        results,
        parsed_answers={"query_present": "Qingdao", "query_missing": None},
        format_validity={"query_present": True, "query_missing": False},
        error_flags={"query_present": ("normalized",), "query_missing": ("missing",)},
    )

    assert [prediction.query_id for prediction in predictions] == ["query_present"]
    assert predictions[0].parsed_answer == "Qingdao"
    assert predictions[0].error_flags == ["normalized"]
