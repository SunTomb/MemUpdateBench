from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import JsonValue

from mub.vnext.contracts.adapter import AnswerResult
from mub.vnext.contracts.runtime import AnswerPrediction
from mub.vnext.runtime.engine import normalize_answer_result


def normalize_answer_results(
    results: Iterable[AnswerResult],
    *,
    parsed_answers: Mapping[str, JsonValue | None] | None = None,
    format_validity: Mapping[str, bool] | None = None,
    error_flags: Mapping[str, Iterable[str]] | None = None,
) -> list[AnswerPrediction]:
    parsed_answers = parsed_answers or {}
    format_validity = format_validity or {}
    error_flags = error_flags or {}
    predictions: list[AnswerPrediction] = []
    seen: set[str] = set()
    for result in results:
        if result.query_id in seen:
            raise ValueError(f"duplicate answer result query_id: {result.query_id}")
        seen.add(result.query_id)
        predictions.append(
            normalize_answer_result(
                result,
                parsed_answer=parsed_answers.get(result.query_id),
                format_valid=format_validity.get(result.query_id),
                error_flags=error_flags.get(result.query_id, ()),
            )
        )
    return predictions


__all__ = ["normalize_answer_results"]
