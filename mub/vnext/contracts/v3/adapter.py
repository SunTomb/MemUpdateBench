from __future__ import annotations

from typing import Protocol

from pydantic import JsonValue

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import ImmutableContractModel, MemoryObjectKey, StrictBool
from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.contracts.v3.task import MemoryQueryV3
from mub.vnext.contracts.task import MemoryEvent


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
    event_id: str
    requested_operation: Operation | None = None
    effective_operation: Operation | None = None
    observed_scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKey, ...] = ()
    affected_entry_ids: tuple[str, ...] = ()
    raw_result: JsonValue | None = None


class AdapterAnswerResultV3(ImmutableContractModel):
    prediction: AnswerPredictionV3
    raw_result: JsonValue | None = None


class MemoryAdapterV3(Protocol):
    def adapter_info(self) -> AdapterInfoV3: ...
    def capabilities(self) -> AdapterCapabilitiesV3: ...
    def reset(self, namespace: str, config: dict) -> object: ...
    def ingest_event(self, event: MemoryEvent) -> AdapterActionResultV3: ...
    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3: ...
    def close(self) -> None: ...


__all__ = ["AdapterActionResultV3", "AdapterAnswerResultV3", "AdapterCapabilitiesV3", "AdapterInfoV3", "MemoryAdapterV3"]
