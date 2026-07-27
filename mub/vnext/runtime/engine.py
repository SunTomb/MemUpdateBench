from __future__ import annotations

from collections.abc import Iterable

from pydantic import JsonValue

from mub.vnext.contracts.adapter import AnswerResult
from mub.vnext.contracts.enums import AnswerDisposition
from mub.vnext.contracts.runtime import AnswerPrediction


def normalize_answer_result(
    result: AnswerResult,
    *,
    parsed_answer: JsonValue | None = None,
    format_valid: bool | None = None,
    cited_event_ids: Iterable[str] = (),
    cited_entry_ids: Iterable[str] = (),
    error_flags: Iterable[str] = (),
) -> AnswerPrediction:
    result = AnswerResult.model_validate(result.model_dump(mode="python"))
    disposition = result.disposition
    normalized_value = (
        result.value if result.value is not None else parsed_answer
    ) if disposition is AnswerDisposition.ANSWERED else None
    if disposition is AnswerDisposition.ABSTAINED:
        normalized_format_valid = True
    elif disposition is AnswerDisposition.UNAVAILABLE:
        normalized_format_valid = False
    else:
        normalized_format_valid = (
            normalized_value is not None if format_valid is None else format_valid
        )
    return AnswerPrediction(
        query_id=result.query_id,
        raw_output=result.raw_output,
        disposition=disposition,
        parsed_answer=normalized_value,
        cited_event_ids=list(cited_event_ids),
        cited_entry_ids=list(cited_entry_ids),
        format_valid=normalized_format_valid,
        error_flags=list(error_flags),
        latency_ms=result.latency_ms,
        usage=dict(result.usage),
    )


__all__ = ["normalize_answer_result"]
