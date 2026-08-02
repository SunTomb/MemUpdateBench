from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import ImmutableContractModel, StrictBool
from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, FrozenJsonValue, MemoryObjectKeyV3, StrictIdentifier, StrictPositiveInt, object_identity, validate_action_coherence
from mub.vnext.contracts.v3.enums import LedgerEntryStatus
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import MemoryEventV3, MemoryQueryV3


class AdapterInfoV3(AdapterInfo):
    adapter_id: StrictIdentifier
    adapter_version: StrictIdentifier
    system_name: StrictIdentifier
    system_version: StrictIdentifier
    sdk_version: StrictIdentifier | None = None
    extractor_id: StrictIdentifier | None = None
    extractor_version: StrictIdentifier | None = None


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
    event_id: StrictIdentifier
    requested_operation: Operation | None = None
    effective_operation: Operation | None = None
    observed_scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    value: FrozenJsonValue | None = None
    affected_entry_ids: tuple[StrictIdentifier, ...] = ()
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


class ResetRequestV3(ImmutableContractModel):
    namespace: StrictIdentifier
    config: FrozenJsonObjectV3 = Field(default_factory=dict)


class RetrievalRequestV3(ImmutableContractModel):
    query: MemoryQueryV3
    k: StrictPositiveInt
    filters: FrozenJsonObjectV3 = Field(default_factory=dict)
    options: FrozenJsonObjectV3 = Field(default_factory=dict)


class VersionHistoryExportRequestV3(ImmutableContractModel):
    namespace: StrictIdentifier
    object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    filters: FrozenJsonObjectV3 = Field(default_factory=dict)
    options: FrozenJsonObjectV3 = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        identities = [object_identity(key) for key in self.object_keys]
        if len(identities) != len(set(identities)):
            raise ValueError("history request object keys must be unique")
        return self


class ExportedEventAnchorV3(ImmutableContractModel):
    event_id: StrictIdentifier
    sequence_index: int = Field(strict=True, ge=0)
    logical_time: StrictIdentifier | None = None

    @field_validator("sequence_index", mode="before")
    @classmethod
    def _exact_sequence_index(cls, value):
        if type(value) is not int:
            raise ValueError("sequence_index must be an exact built-in integer")
        return value


class ExportedVersionRecordV3(ImmutableContractModel):
    version_index: int = Field(strict=True, ge=0)
    status: LedgerEntryStatus
    value: FrozenJsonValue | None = None
    valid_from: ExportedEventAnchorV3 | None = None
    valid_until: ExportedEventAnchorV3 | None = None
    logical_time: StrictIdentifier | None = None
    source_anchors: tuple[ExportedEventAnchorV3, ...] = Field(min_length=1)

    @field_validator("version_index", mode="before")
    @classmethod
    def _exact_index(cls, value):
        if type(value) is not int:
            raise ValueError("version_index must be an exact built-in integer")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status == LedgerEntryStatus.PRESENT and self.value is None:
            raise ValueError("present exported versions require value")
        if self.status == LedgerEntryStatus.TOMBSTONE and self.value is not None:
            raise ValueError("tombstone exported versions cannot carry value")
        if (self.valid_from is None) != (self.valid_until is None):
            raise ValueError("valid_from and valid_until must be supplied together")
        if self.valid_from is None and self.logical_time is None:
            raise ValueError("exported versions require an event interval or logical-time anchor")
        if self.valid_from is not None and self.valid_from.sequence_index >= self.valid_until.sequence_index:
            raise ValueError("validity interval must be strictly chronological")
        source_keys = [(anchor.sequence_index, anchor.event_id) for anchor in self.source_anchors]
        if source_keys != sorted(source_keys) or len(source_keys) != len(set(source_keys)):
            raise ValueError("source anchors must be ordered and unique")
        source_logical_times = [anchor.logical_time for anchor in self.source_anchors if anchor.logical_time is not None]
        if source_logical_times != sorted(source_logical_times):
            raise ValueError("source anchor logical times must be nondecreasing")
        if self.valid_from is not None and any(
            anchor.sequence_index < self.valid_from.sequence_index
            or anchor.sequence_index >= self.valid_until.sequence_index
            for anchor in self.source_anchors
        ):
            raise ValueError("source anchors must belong to the validity interval")
        return self


class ObjectVersionHistoryV3(ImmutableContractModel):
    object_key: MemoryObjectKeyV3
    versions: tuple[ExportedVersionRecordV3, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous(self) -> Self:
        if tuple(version.version_index for version in self.versions) != tuple(range(len(self.versions))):
            raise ValueError("exported version history must start at zero and be contiguous")
        event_positions: dict[str, int] = {}
        for version in self.versions:
            for anchor in (version.valid_from, version.valid_until, *version.source_anchors):
                if anchor is None:
                    continue
                previous_position = event_positions.setdefault(anchor.event_id, anchor.sequence_index)
                if previous_position != anchor.sequence_index:
                    raise ValueError("event anchors must use one consistent sequence_index")
        for previous, current in zip(self.versions, self.versions[1:]):
            if (previous.valid_until is None) != (current.valid_from is None):
                raise ValueError("adjacent histories cannot mix event and logical-time bounds")
            if previous.valid_until is not None and (
                previous.valid_until.event_id != current.valid_from.event_id
                or previous.valid_until.sequence_index != current.valid_from.sequence_index
            ):
                raise ValueError("adjacent validity intervals must be continuous and nonoverlapping")
            previous_logical_time = previous.logical_time or (previous.valid_from.logical_time if previous.valid_from is not None else None)
            current_logical_time = current.logical_time or (current.valid_from.logical_time if current.valid_from is not None else None)
            if previous_logical_time is not None and current_logical_time is not None and previous_logical_time > current_logical_time:
                raise ValueError("logical time must be nondecreasing")
        return self


class VersionHistoryExportResultV3(ImmutableContractModel):
    histories: tuple[ObjectVersionHistoryV3, ...] = ()

    @model_validator(mode="after")
    def _unique_histories(self) -> Self:
        identities = [object_identity(history.object_key) for history in self.histories]
        if len(identities) != len(set(identities)):
            raise ValueError("exported histories must have unique object identities")
        return self


class ResetResultV3(ImmutableContractModel):
    success: StrictBool
    namespace: StrictIdentifier
    error: FrozenJsonValue | None = None


class ExportEntriesResultV3(ImmutableContractModel):
    entries: tuple[MemoryEntryRecordV3, ...] = ()


class ExportStateResultV3(ImmutableContractModel):
    raw_state: FrozenJsonValue


class RetrievalResultV3(ImmutableContractModel):
    trace: RetrievalTraceV3


@runtime_checkable
class MemoryAdapterV3(Protocol):
    def adapter_info(self) -> AdapterInfoV3: ...
    def capabilities(self) -> AdapterCapabilitiesV3: ...
    def reset(self, request: ResetRequestV3) -> ResetResultV3: ...
    def ingest_event(self, event: MemoryEventV3) -> AdapterActionResultV3: ...
    def export_entries(self) -> ExportEntriesResultV3: ...
    def export_raw_state(self) -> ExportStateResultV3: ...
    def export_version_history(self, request: VersionHistoryExportRequestV3) -> VersionHistoryExportResultV3: ...
    def retrieve(self, request: RetrievalRequestV3) -> RetrievalResultV3: ...
    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3: ...
    def close(self) -> None: ...


__all__ = [
    "AdapterActionResultV3", "AdapterAnswerResultV3", "AdapterCapabilitiesV3", "AdapterInfoV3",
    "ExportEntriesResultV3", "ExportStateResultV3", "ExportedEventAnchorV3", "ExportedVersionRecordV3", "MemoryAdapterV3",
    "ObjectVersionHistoryV3", "ResetRequestV3", "ResetResultV3", "RetrievalRequestV3", "RetrievalResultV3",
    "VersionHistoryExportRequestV3", "VersionHistoryExportResultV3",
]
