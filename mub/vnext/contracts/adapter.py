from typing import Any, Protocol

from pydantic import Field, JsonValue, model_validator

from mub.vnext.contracts.common import (
    ContractModel,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
    StrictNonnegativeFloat,
    StrictNumericScore,
)
from mub.vnext.contracts.enums import AnswerDisposition, Operation
from mub.vnext.contracts.runtime import MemoryEntryRecord, NonnegativeStrictInt
from mub.vnext.contracts.task import MemoryEvent, MemoryQuery


class AdapterInfo(ImmutableContractModel):
    adapter_id: str
    adapter_version: str
    system_name: str
    system_version: str
    sdk_version: str | None = None
    configuration_hash: str = Field(pattern=SHA256_PATTERN)
    extractor_id: str | None = None
    extractor_version: str | None = None


class AdapterCapabilities(ImmutableContractModel):
    supports_isolated_reset: StrictBool = False
    supports_event_ingest: StrictBool = False
    supports_add: StrictBool = False
    supports_update: StrictBool = False
    supports_noop: StrictBool = False
    supports_delete: StrictBool = False
    supports_ttl: StrictBool = False
    supports_native_answer: StrictBool = False
    exports_entries: StrictBool = False
    exports_raw_state: StrictBool = False
    exports_source_event_ids: StrictBool = False
    exports_timestamps_or_order: StrictBool = False
    exports_object_keys: StrictBool = False
    exports_values: StrictBool = False
    exports_retrieval_ids: StrictBool = False
    exports_retrieval_scores: StrictBool = False
    exports_action_trace: StrictBool = False
    reports_latency: StrictBool = False
    reports_token_usage: StrictBool = False
    reports_cost: StrictBool = False
    requires_evaluation_extractor: StrictBool = False
    extractor_version: str | None = None

    def presentation_level(self, state_transition_linkage_available: bool = False) -> int | None:
        has_l1 = self.exports_retrieval_ids or self.exports_entries
        has_l2 = (
            self.supports_isolated_reset
            and self.exports_entries
            and (self.requires_evaluation_extractor or (self.exports_object_keys and self.exports_values))
        )
        if has_l2 and self.exports_action_trace and state_transition_linkage_available:
            return 3
        if has_l2:
            return 2
        if has_l1:
            return 1
        if self.supports_native_answer:
            return 0
        return None


class ResetResult(ContractModel):
    success: bool
    namespace: str
    error: JsonValue | None = None


class AdapterActionLog(ContractModel):
    event_id: str
    requested_operation: Operation | None = None
    effective_operation: Operation | None = None
    affected_entry_ids: list[str] = Field(default_factory=list)
    raw_action: Any = None
    latency_ms: StrictNonnegativeFloat | None = None
    error: JsonValue | None = None


class RetrievalResult(ContractModel):
    query_id: str
    entries: list[MemoryEntryRecord] = Field(default_factory=list)
    scores: list[StrictNumericScore] = Field(default_factory=list)
    raw_result: Any = None
    latency_ms: StrictNonnegativeFloat | None = None
    error: JsonValue | None = None

    @model_validator(mode="after")
    def _validate_score_length(self):
        if self.scores and len(self.scores) != len(self.entries):
            raise ValueError("scores length must match entries length")
        return self


class AnswerResult(ContractModel):
    query_id: str
    raw_output: str
    disposition: AnswerDisposition = AnswerDisposition.ANSWERED
    usage: dict[str, NonnegativeStrictInt] = Field(default_factory=dict)
    cost: StrictNonnegativeFloat | None = None
    latency_ms: StrictNonnegativeFloat | None = None
    error: JsonValue | None = None

    @model_validator(mode="after")
    def _validate_usage_counts(self):
        for key, value in self.usage.items():
            if value < 0:
                raise ValueError(f"usage count {key} must be nonnegative")
        return self


class MemoryAdapter(Protocol):
    def adapter_info(self) -> AdapterInfo:
        raise NotImplementedError

    def capabilities(self) -> AdapterCapabilities:
        raise NotImplementedError

    def reset(self, namespace: str, config: dict) -> ResetResult:
        raise NotImplementedError

    def ingest_event(self, event: MemoryEvent) -> AdapterActionLog:
        raise NotImplementedError

    def export_entries(self) -> list[MemoryEntryRecord]:
        raise NotImplementedError

    def export_raw_state(self) -> object:
        raise NotImplementedError

    def retrieve(self, query: MemoryQuery, k: int) -> RetrievalResult:
        raise NotImplementedError

    def answer(self, query: MemoryQuery, mode: str) -> AnswerResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


__all__ = [
    "AdapterActionLog",
    "AdapterCapabilities",
    "AdapterInfo",
    "AnswerResult",
    "MemoryAdapter",
    "ResetResult",
    "RetrievalResult",
]
