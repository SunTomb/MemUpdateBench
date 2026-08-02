from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, MemoryObjectKey, StrictBool, StrictNonnegativeFloat
from mub.vnext.contracts.enums import ActionScope, AnswerDisposition, CompletionStatus, Operation
from mub.vnext.contracts.runtime import MemorySnapshot, ParserExtractorProvenance, RetrievalTrace
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.version import RUNTIME_RECORD_VERSION_V3, SCHEMA_VERSION_V3


class ParsedManagerActionV3(ImmutableContractModel):
    event_id: str = Field(strict=True, min_length=1)
    operation: Operation | None = None
    observed_scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKey, ...] = ()
    value: JsonValue | None = None
    format_valid: StrictBool
    execution_status: ExecutionStatusV3
    fallback_used: StrictBool
    error_flags: tuple[str, ...] = ()
    raw_output: str = Field(strict=True)
    latency_ms: StrictNonnegativeFloat | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if self.operation == Operation.NOOP:
            if self.observed_scope is not None or self.target_object_keys or self.value is not None:
                raise ValueError("NOOP cannot carry scope, targets, or value")
        elif self.operation is not None:
            if self.observed_scope is None or not self.target_object_keys:
                raise ValueError("observed operations require scope and target objects")
        if len({key.canonical_id for key in self.target_object_keys}) != len(self.target_object_keys):
            raise ValueError("target object identities must be unique")
        if self.execution_status == ExecutionStatusV3.EXECUTED and not self.format_valid:
            raise ValueError("executed actions must be format-valid")
        return self


class AnswerPredictionV3(ImmutableContractModel):
    query_id: str = Field(strict=True, min_length=1)
    raw_output: str = Field(strict=True)
    disposition: AnswerDisposition = AnswerDisposition.ANSWERED
    parsed_answer: JsonValue | None = None
    cited_event_ids: tuple[str, ...] = ()
    cited_entry_ids: tuple[str, ...] = ()
    cited_object_keys: tuple[MemoryObjectKey, ...] = ()
    cited_derivation_step_ids: tuple[str, ...] = ()
    format_valid: StrictBool
    error_flags: tuple[str, ...] = ()
    latency_ms: StrictNonnegativeFloat | None = None
    usage: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prediction(self) -> Self:
        if self.disposition != AnswerDisposition.ANSWERED and self.parsed_answer is not None:
            raise ValueError("non-answered predictions cannot carry parsed_answer")
        if self.disposition == AnswerDisposition.ANSWERED and self.format_valid and self.parsed_answer is None:
            raise ValueError("format-valid answered predictions require parsed_answer")
        for label, values in (("event citations", self.cited_event_ids), ("entry citations", self.cited_entry_ids), ("derivation citations", self.cited_derivation_step_ids)):
            if any(type(value) is not str or not value.strip() for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{label} must be nonblank and unique")
        if len({key.canonical_id for key in self.cited_object_keys}) != len(self.cited_object_keys):
            raise ValueError("cited object identities must be unique")
        if any(type(value) is not int or isinstance(value, bool) or value < 0 for value in self.usage.values()):
            raise ValueError("usage counts must be exact nonnegative integers")
        return self


class TaskRunRecordV3(ImmutableContractModel):
    schema_version: Literal[SCHEMA_VERSION_V3] = SCHEMA_VERSION_V3
    runtime_record_version: Literal[RUNTIME_RECORD_VERSION_V3] = RUNTIME_RECORD_VERSION_V3
    task_id: str = Field(strict=True, min_length=1)
    adapter_id: str = Field(strict=True, min_length=1)
    run_id: str = Field(strict=True, min_length=1)
    parsed_actions: tuple[ParsedManagerActionV3, ...] = ()
    memory_snapshots: tuple[MemorySnapshot, ...] = ()
    retrieval_traces: tuple[RetrievalTrace, ...] = ()
    answer_predictions: tuple[AnswerPredictionV3, ...] = ()
    system_events: tuple[dict[str, JsonValue], ...] = ()
    parser_extractor_provenance: ParserExtractorProvenance
    exceptions: tuple[dict[str, JsonValue], ...] = ()
    completion_status: CompletionStatus

    @model_validator(mode="after")
    def _status_semantics(self) -> Self:
        if self.completion_status == CompletionStatus.COMPLETED and self.exceptions:
            raise ValueError("completed records cannot carry exceptions")
        for values, label in ((self.parsed_actions, "action event IDs"), (self.answer_predictions, "prediction query IDs")):
            ids = [item.event_id if hasattr(item, "event_id") else item.query_id for item in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label}")
        return self


__all__ = ["AnswerPredictionV3", "ParsedManagerActionV3", "TaskRunRecordV3"]
