from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool, StrictNonnegativeFloat
from mub.vnext.contracts.enums import ActionScope, AnswerDisposition, CompletionStatus, Operation
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, FrozenJsonValue, FrozenUsageMap, MemoryObjectKeyV3, StrictFiniteFloat, StrictIdentifier, StrictPositiveInt, validate_action_coherence
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.version import RUNTIME_RECORD_VERSION_V3, SCHEMA_VERSION_V3


class MemoryEntryRecordV3(ImmutableContractModel):
    entry_id: StrictIdentifier
    content: str = Field(strict=True)
    object_key_candidate: MemoryObjectKeyV3 | None = None
    value_candidate: FrozenJsonValue | None = None
    created_at: str | None = Field(default=None, strict=True)
    updated_at: str | None = Field(default=None, strict=True)
    source_event_ids: tuple[StrictIdentifier, ...] = ()
    version_index: int | None = Field(default=None, strict=True, ge=0)
    raw_metadata: FrozenJsonObjectV3 = Field(default_factory=dict)


class MemorySnapshotV3(ImmutableContractModel):
    after_event_id: StrictIdentifier | None = None
    entries: tuple[MemoryEntryRecordV3, ...] = ()
    state_by_object: FrozenJsonObjectV3 = Field(default_factory=dict)
    store_size: int = Field(strict=True, ge=0)
    raw_adapter_state: FrozenJsonValue | None = None
    snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)


class RetrievalTraceV3(ImmutableContractModel):
    query_id: StrictIdentifier
    retrieved_entries: tuple[MemoryEntryRecordV3, ...] = ()
    scores: tuple[StrictFiniteFloat, ...] = ()
    ranks: tuple[StrictPositiveInt, ...] = ()
    gold_in_context: StrictBool | None = None
    stale_in_context: StrictBool | None = None
    distractor_in_context: StrictBool | None = None
    retrieval_policy: str | None = None
    context_order: str | None = None
    version_metadata: FrozenJsonObjectV3 = Field(default_factory=dict)
    prompt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        if self.scores and len(self.scores) != len(self.retrieved_entries):
            raise ValueError("scores length must match retrieved_entries length")
        if self.ranks and len(self.ranks) != len(self.retrieved_entries):
            raise ValueError("ranks length must match retrieved_entries length")
        if any(type(score) is not float for score in self.scores):
            raise ValueError("scores must be exact finite floats")
        if any(type(rank) is not int or isinstance(rank, bool) or rank <= 0 for rank in self.ranks):
            raise ValueError("ranks must be exact positive integers")
        return self


class ParserExtractorProvenanceV3(ImmutableContractModel):
    action_parser_version: str = Field(strict=True, min_length=1)
    answer_parser_version: str = Field(strict=True, min_length=1)
    memory_entry_extractor_version: str = Field(strict=True, min_length=1)
    object_value_extractor_config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)
    redaction_policy_version: str = Field(strict=True, min_length=1)
    raw_provider_artifact_path: str | None = None
    raw_provider_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)
    raw_adapter_state_path: str | None = None
    raw_adapter_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", strict=True)


class ParsedManagerActionV3(ImmutableContractModel):
    action_id: StrictIdentifier
    event_id: StrictIdentifier
    operation: Operation | None = None
    observed_scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    value: FrozenJsonValue | None = None
    format_valid: StrictBool
    execution_status: ExecutionStatusV3
    fallback_used: StrictBool
    error_flags: tuple[str, ...] = ()
    raw_output: str = Field(strict=True)
    latency_ms: StrictNonnegativeFloat | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        validate_action_coherence(
            operation=self.operation,
            scope=self.observed_scope,
            targets=self.target_object_keys,
            value=self.value,
            executed=self.execution_status == ExecutionStatusV3.EXECUTED,
        )
        if self.execution_status == ExecutionStatusV3.EXECUTED and not self.format_valid:
            raise ValueError("executed actions must be format-valid")
        return self


class AnswerPredictionV3(ImmutableContractModel):
    query_id: StrictIdentifier
    raw_output: str = Field(strict=True)
    disposition: AnswerDisposition = AnswerDisposition.ANSWERED
    parsed_answer: FrozenJsonValue | None = None
    cited_event_ids: tuple[StrictIdentifier, ...] = ()
    cited_entry_ids: tuple[StrictIdentifier, ...] = ()
    cited_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    cited_derivation_step_ids: tuple[StrictIdentifier, ...] = ()
    format_valid: StrictBool
    error_flags: tuple[str, ...] = ()
    latency_ms: StrictNonnegativeFloat | None = None
    usage: FrozenUsageMap = Field(default_factory=dict)

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
    task_id: StrictIdentifier
    adapter_id: StrictIdentifier
    run_id: StrictIdentifier
    parsed_actions: tuple[ParsedManagerActionV3, ...] = ()
    memory_snapshots: tuple[MemorySnapshotV3, ...] = ()
    retrieval_traces: tuple[RetrievalTraceV3, ...] = ()
    answer_predictions: tuple[AnswerPredictionV3, ...] = ()
    system_events: tuple[FrozenJsonObjectV3, ...] = ()
    parser_extractor_provenance: ParserExtractorProvenanceV3
    exceptions: tuple[FrozenJsonObjectV3, ...] = ()
    completion_status: CompletionStatus

    @model_validator(mode="after")
    def _status_semantics(self) -> Self:
        if self.completion_status == CompletionStatus.COMPLETED and self.exceptions:
            raise ValueError("completed records cannot carry exceptions")
        action_ids = [action.action_id for action in self.parsed_actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("duplicate action IDs")
        prediction_ids = [prediction.query_id for prediction in self.answer_predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("duplicate prediction query IDs")
        return self


__all__ = ["AnswerPredictionV3", "MemoryEntryRecordV3", "MemorySnapshotV3", "ParsedManagerActionV3", "ParserExtractorProvenanceV3", "RetrievalTraceV3", "TaskRunRecordV3"]
