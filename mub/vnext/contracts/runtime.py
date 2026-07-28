from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from mub.vnext.contracts.common import (
    ContractModel,
    MemoryObjectKey,
    SHA256_PATTERN,
    StrictBool,
    StrictNonnegativeFloat,
    StrictNumericScore,
)
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, Operation
from mub.vnext.version import RUNTIME_RECORD_VERSION, SCHEMA_VERSION

NonnegativeStrictInt = Annotated[int, Field(ge=0, strict=True)]
PositiveStrictInt = Annotated[int, Field(gt=0, strict=True)]


class ParsedManagerAction(ContractModel):
    event_id: str
    operation: Operation | None = None
    target_object_key: MemoryObjectKey | None = None
    value: JsonValue | None = None
    format_valid: StrictBool
    execution_status: str
    fallback_used: StrictBool
    error_flags: list[str] = Field(default_factory=list)
    raw_output: str
    latency_ms: StrictNonnegativeFloat | None = None


class MemoryEntryRecord(ContractModel):
    entry_id: str
    content: str
    object_key_candidate: MemoryObjectKey | None = None
    value_candidate: JsonValue | None = None
    created_at: str | None = None
    updated_at: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    version_index: NonnegativeStrictInt | None = None
    raw_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class MemorySnapshot(ContractModel):
    after_event_id: str | None = None
    entries: list[MemoryEntryRecord] = Field(default_factory=list)
    state_by_object: dict[str, JsonValue] = Field(default_factory=dict)
    store_size: NonnegativeStrictInt
    raw_adapter_state: JsonValue | None = None
    snapshot_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class RetrievalTrace(ContractModel):
    query_id: str
    retrieved_entries: list[MemoryEntryRecord] = Field(default_factory=list)
    scores: list[StrictNumericScore] = Field(default_factory=list)
    ranks: list[PositiveStrictInt] = Field(default_factory=list)
    gold_in_context: StrictBool | None = None
    stale_in_context: StrictBool | None = None
    distractor_in_context: StrictBool | None = None
    retrieval_policy: str | None = None
    context_order: str | None = None
    version_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_trace_lengths(self):
        if self.scores and len(self.scores) != len(self.retrieved_entries):
            raise ValueError("scores length must match retrieved_entries length")
        if self.ranks and len(self.ranks) != len(self.retrieved_entries):
            raise ValueError("ranks length must match retrieved_entries length")
        return self


class AnswerPrediction(ContractModel):
    query_id: str
    raw_output: str
    disposition: AnswerDisposition = AnswerDisposition.ANSWERED
    parsed_answer: JsonValue | None = None
    cited_event_ids: list[str] = Field(default_factory=list)
    cited_entry_ids: list[str] = Field(default_factory=list)
    format_valid: StrictBool
    error_flags: list[str] = Field(default_factory=list)
    latency_ms: StrictNonnegativeFloat | None = None
    usage: dict[str, NonnegativeStrictInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_answer_disposition(self):
        if self.disposition != AnswerDisposition.ANSWERED and self.parsed_answer is not None:
            raise ValueError("non-answered predictions cannot carry parsed_answer")
        if self.disposition == AnswerDisposition.ANSWERED and self.format_valid and self.parsed_answer is None:
            raise ValueError("answered predictions with valid format require parsed_answer")
        _validate_nonnegative_usage(self.usage)
        return self


class ParserExtractorProvenance(ContractModel):
    action_parser_version: str
    answer_parser_version: str
    memory_entry_extractor_version: str
    object_value_extractor_config_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    redaction_policy_version: str
    raw_provider_artifact_path: str | None = None
    raw_provider_artifact_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    raw_adapter_state_path: str | None = None
    raw_adapter_state_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class TaskRunRecord(ContractModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    runtime_record_version: Literal[RUNTIME_RECORD_VERSION] = RUNTIME_RECORD_VERSION
    task_id: str
    adapter_id: str
    run_id: str
    parsed_actions: list[ParsedManagerAction] = Field(default_factory=list)
    memory_snapshots: list[MemorySnapshot] = Field(default_factory=list)
    retrieval_traces: list[RetrievalTrace] = Field(default_factory=list)
    answer_predictions: list[AnswerPrediction] = Field(default_factory=list)
    system_events: list[dict[str, JsonValue]] = Field(default_factory=list)
    parser_extractor_provenance: ParserExtractorProvenance
    exceptions: list[dict[str, JsonValue]] = Field(default_factory=list)
    completion_status: CompletionStatus


def _validate_nonnegative_usage(usage: dict[str, NonnegativeStrictInt]) -> None:
    for key, value in usage.items():
        if value < 0:
            raise ValueError(f"usage count {key} must be nonnegative")


__all__ = [
    "AnswerPrediction",
    "MemoryEntryRecord",
    "MemorySnapshot",
    "ParsedManagerAction",
    "ParserExtractorProvenance",
    "RetrievalTrace",
    "TaskRunRecord",
]
