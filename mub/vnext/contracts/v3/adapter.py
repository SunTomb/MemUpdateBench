from __future__ import annotations

from typing import Protocol

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import ImmutableContractModel, StrictBool
from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.common import FrozenJsonValue, MemoryObjectKeyV3, validate_action_coherence
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.contracts.v3.task import MemoryEventV3, MemoryQueryV3


class AdapterInfoV3(AdapterInfo):
    pass


class AdapterCapabilitiesV3(AdapterCapabilities):
    supports_scoped_delete: StrictBool = False
    supports_historical_query: StrictBool = False
    exports_version_history: StrictBool = False
    supports_multi_object_query: StrictBool = False
    exports_evidence_linkage: StrictBool = False

    def core_capability_requirements(self) -> dict[str, bool]:
        return {
            "scoped_delete": self.supports_scoped_delete,
            "historical_query": self.supports_historical_query,
            "version_history": self.exports_version_history,
            "multi_object_query": self.supports_multi_object_query,
            "evidence_linkage": self.exports_evidence_linkage,
        }


class AdapterActionResultV3(ImmutableContractModel):
    event_id: str = Field(strict=True, min_length=1)
    requested_operation: Operation | None = None
    effective_operation: Operation | None = None
    observed_scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    value: FrozenJsonValue | None = None
    affected_entry_ids: tuple[str, ...] = ()
    raw_result: FrozenJsonValue | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if any(type(item) is not str or not item.strip() for item in self.affected_entry_ids):
            raise ValueError("affected entry IDs must be nonblank strings")
        if len(self.affected_entry_ids) != len(set(self.affected_entry_ids)):
            raise ValueError("affected entry IDs must be unique")
        if self.effective_operation is not None and self.requested_operation is None:
            raise ValueError("effective operation requires requested operation")
        operations = tuple(
            operation
            for operation in (self.requested_operation, self.effective_operation)
            if operation is not None
        )
        if not operations:
            validate_action_coherence(operation=None, scope=self.observed_scope, targets=self.target_object_keys, value=self.value)
        for operation in operations:
            validate_action_coherence(operation=operation, scope=self.observed_scope, targets=self.target_object_keys, value=self.value)
        return self


class AdapterAnswerResultV3(ImmutableContractModel):
    prediction: AnswerPredictionV3
    raw_result: FrozenJsonValue | None = None


class MemoryAdapterV3(Protocol):
    def adapter_info(self) -> AdapterInfoV3: ...
    def capabilities(self) -> AdapterCapabilitiesV3: ...
    def reset(self, namespace: str, config: dict) -> object: ...
    def ingest_event(self, event: MemoryEventV3) -> AdapterActionResultV3: ...
    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3: ...
    def close(self) -> None: ...


__all__ = ["AdapterActionResultV3", "AdapterAnswerResultV3", "AdapterCapabilitiesV3", "AdapterInfoV3", "MemoryAdapterV3"]
