from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.adapter import AdapterCapabilities, AdapterInfo
from mub.vnext.contracts.common import ImmutableContractModel, StrictBool
from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, FrozenJsonValue, MemoryObjectKeyV3, StrictIdentifier, StrictPositiveInt, object_identity, validate_action_coherence
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, ParsedManagerActionV3, RetrievalTraceV3
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


class AdapterActionPayloadV3(ImmutableContractModel):
    operation: Operation | None = None
    scope: ActionScope | None = None
    target_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    value: FrozenJsonValue | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        validate_action_coherence(
            operation=self.operation,
            scope=self.scope,
            targets=self.target_object_keys,
            value=self.value,
        )
        return self


class AdapterActionResultV3(ImmutableContractModel):
    event_id: StrictIdentifier
    requested_action: AdapterActionPayloadV3 = Field(default_factory=AdapterActionPayloadV3)
    effective_action: AdapterActionPayloadV3 = Field(default_factory=AdapterActionPayloadV3)
    execution_status: ExecutionStatusV3
    reason: StrictIdentifier | None = None
    error: FrozenJsonValue | None = None
    affected_entry_ids: tuple[StrictIdentifier, ...] = ()
    raw_result: FrozenJsonValue | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.affected_entry_ids) != len(set(self.affected_entry_ids)):
            raise ValueError("affected entry IDs must be unique")
        requested = self.requested_action.operation
        effective = self.effective_action.operation
        mutation_ops = {Operation.ADD, Operation.UPDATE, Operation.DELETE}
        if effective in {None, Operation.NOOP} and self.affected_entry_ids:
            raise ValueError("effective None/NOOP actions cannot carry affected_entry_ids")
        if self.execution_status == ExecutionStatusV3.EXECUTED:
            if requested is None:
                raise ValueError("executed actions require a nonempty requested_action")
            if effective is None:
                raise ValueError("executed actions require an effective action")
            if effective == Operation.NOOP and requested != Operation.NOOP:
                raise ValueError("mutation-to-NOOP outcomes require status=no_effect")
            if effective in mutation_ops and not self.affected_entry_ids:
                raise ValueError("executed mutations require affected_entry_ids")
            if self.reason is not None or self.error is not None:
                raise ValueError("executed actions cannot carry reason or error")
        elif self.execution_status == ExecutionStatusV3.NO_EFFECT:
            if requested not in mutation_ops or effective != Operation.NOOP or self.affected_entry_ids:
                raise ValueError("no_effect requires requested mutation, effective NOOP, and no affected entries")
            if self.reason is None or self.error is not None:
                raise ValueError("no_effect requires reason and cannot carry error")
        elif self.execution_status in {ExecutionStatusV3.REJECTED, ExecutionStatusV3.NOT_SUPPORTED}:
            if effective is not None or self.affected_entry_ids or self.reason is None:
                raise ValueError("rejected/not_supported actions require reason and no effective action")
            if self.error is not None:
                raise ValueError("rejected/not_supported actions cannot carry error")
        elif self.execution_status == ExecutionStatusV3.FAILED:
            if effective is not None or self.affected_entry_ids or self.error is None:
                raise ValueError("failed actions require error and no effective action")
        return self

    def to_parsed_manager_action(
        self,
        *,
        raw_output: str,
        format_valid: bool,
        fallback_used: bool,
    ) -> ParsedManagerActionV3:
        return ParsedManagerActionV3(
            event_id=self.event_id,
            operation=self.requested_action.operation,
            observed_scope=self.requested_action.scope,
            target_object_keys=self.requested_action.target_object_keys,
            value=self.requested_action.value,
            format_valid=format_valid,
            execution_status=self.execution_status,
            fallback_used=fallback_used,
            error_flags=(() if self.reason is None else (self.reason,)),
            raw_output=raw_output,
        )


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


def _effective_version_logical_time(version) -> str | None:
    record_time = version.logical_time
    anchor_time = (
        version.valid_from.logical_time
        if version.valid_from is not None
        else None
    )
    if record_time is not None and anchor_time is not None and record_time != anchor_time:
        raise ValueError("record logical_time must equal valid_from logical_time")
    return record_time if record_time is not None else anchor_time


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
        if self.valid_from is None and self.valid_until is not None:
            raise ValueError("valid_until cannot be supplied without valid_from")
        if self.valid_from is None and self.logical_time is None:
            raise ValueError("exported versions require an event interval or logical-time anchor")
        if self.valid_from is not None and self.valid_until is not None and self.valid_from.sequence_index >= self.valid_until.sequence_index:
            raise ValueError("validity interval must be strictly chronological")
        effective_logical_time = _effective_version_logical_time(self)
        if (
            effective_logical_time is not None
            and self.valid_until is not None
            and self.valid_until.logical_time is not None
            and effective_logical_time > self.valid_until.logical_time
        ):
            raise ValueError("validity logical-time bounds must be ordered")
        source_keys = [(anchor.sequence_index, anchor.event_id) for anchor in self.source_anchors]
        if source_keys != sorted(source_keys) or len(source_keys) != len(set(source_keys)):
            raise ValueError("source anchors must be ordered and unique")
        source_logical_times = [anchor.logical_time for anchor in self.source_anchors if anchor.logical_time is not None]
        if source_logical_times != sorted(source_logical_times):
            raise ValueError("source anchor logical times must be nondecreasing")
        if self.valid_from is not None and any(
            anchor.sequence_index < self.valid_from.sequence_index
            or (self.valid_until is not None and anchor.sequence_index >= self.valid_until.sequence_index)
            for anchor in self.source_anchors
        ):
            raise ValueError("source anchors must belong to the validity interval")
        logical_lower = effective_logical_time
        logical_upper = self.valid_until.logical_time if self.valid_until is not None else None
        if any(
            anchor.logical_time is not None
            and (
                (logical_lower is not None and anchor.logical_time < logical_lower)
                or (logical_upper is not None and anchor.logical_time > logical_upper)
            )
            for anchor in self.source_anchors
        ):
            raise ValueError("source anchors must belong to the logical-time interval")
        return self


class ObjectVersionHistoryV3(ImmutableContractModel):
    object_key: MemoryObjectKeyV3
    versions: tuple[ExportedVersionRecordV3, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous(self) -> Self:
        if tuple(version.version_index for version in self.versions) != tuple(range(len(self.versions))):
            raise ValueError("exported version history must start at zero and be contiguous")
        for version in self.versions[:-1]:
            if version.valid_from is not None and version.valid_until is None:
                raise ValueError("only the final exported version may have an open event interval")
        for version in self.versions:
            if version.valid_from is None:
                if any(anchor.logical_time is None for anchor in version.source_anchors):
                    raise ValueError("logical-only history source anchors require logical_time")
                if any(anchor.logical_time < version.logical_time for anchor in version.source_anchors):
                    raise ValueError("logical-only source anchors cannot precede their version logical_time")
        event_positions: dict[str, int] = {}
        position_events: dict[int, str] = {}
        for version in self.versions:
            for anchor in (version.valid_from, version.valid_until, *version.source_anchors):
                if anchor is None:
                    continue
                previous_position = event_positions.setdefault(anchor.event_id, anchor.sequence_index)
                if previous_position != anchor.sequence_index:
                    raise ValueError("event anchors must use one consistent sequence_index")
                previous_event = position_events.setdefault(anchor.sequence_index, anchor.event_id)
                if previous_event != anchor.event_id:
                    raise ValueError("each sequence_index must identify exactly one event_id")
        for previous, current in zip(self.versions, self.versions[1:]):
            if (previous.valid_until is None) != (current.valid_from is None):
                raise ValueError("adjacent histories cannot mix event and logical-time bounds")
            if previous.valid_until is not None and (
                previous.valid_until.event_id != current.valid_from.event_id
                or previous.valid_until.sequence_index != current.valid_from.sequence_index
            ):
                raise ValueError("adjacent validity intervals must be continuous and nonoverlapping")
            previous_logical_time = _effective_version_logical_time(previous)
            current_logical_time = _effective_version_logical_time(current)
            if previous_logical_time is not None and current_logical_time is not None:
                logical_only = previous.valid_from is None and current.valid_from is None
                if previous_logical_time > current_logical_time or (logical_only and previous_logical_time == current_logical_time):
                    raise ValueError("logical-only histories require strictly increasing logical time")
                if logical_only:
                    if any(anchor.logical_time >= current_logical_time for anchor in previous.source_anchors):
                        raise ValueError("logical-only source anchors must stay within the half-open version interval")
                    if max(anchor.sequence_index for anchor in previous.source_anchors) >= min(anchor.sequence_index for anchor in current.source_anchors):
                        raise ValueError("logical-only source sequence ranges must strictly precede the next version")
        return self


class VersionHistoryExportResultV3(ImmutableContractModel):
    histories: tuple[ObjectVersionHistoryV3, ...] = ()

    @model_validator(mode="after")
    def _unique_histories(self) -> Self:
        identities = [object_identity(history.object_key) for history in self.histories]
        if len(identities) != len(set(identities)):
            raise ValueError("exported histories must have unique object identities")
        event_bindings: dict[str, tuple[int, str | None]] = {}
        sequence_bindings: dict[int, str] = {}
        for history in self.histories:
            for version in history.versions:
                for anchor in (version.valid_from, version.valid_until, *version.source_anchors):
                    if anchor is None:
                        continue
                    binding = (anchor.sequence_index, anchor.logical_time)
                    previous_binding = event_bindings.setdefault(anchor.event_id, binding)
                    if previous_binding != binding:
                        raise ValueError("global event anchors must use one consistent sequence_index and logical_time")
                    previous_event = sequence_bindings.setdefault(anchor.sequence_index, anchor.event_id)
                    if previous_event != anchor.event_id:
                        raise ValueError("global sequence_index values must identify exactly one event_id")
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
    request: RetrievalRequestV3
    trace: RetrievalTraceV3

    @model_validator(mode="after")
    def _bind_request(self) -> Self:
        if self.trace.query_id != self.request.query.query_id:
            raise ValueError("retrieval trace query_id must match bound request query")
        return self


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
    "AdapterActionPayloadV3", "AdapterActionResultV3", "AdapterAnswerResultV3", "AdapterCapabilitiesV3", "AdapterInfoV3",
    "ExportEntriesResultV3", "ExportStateResultV3", "ExportedEventAnchorV3", "ExportedVersionRecordV3", "MemoryAdapterV3",
    "ObjectVersionHistoryV3", "ResetRequestV3", "ResetResultV3", "RetrievalRequestV3", "RetrievalResultV3",
    "VersionHistoryExportRequestV3", "VersionHistoryExportResultV3",
]
