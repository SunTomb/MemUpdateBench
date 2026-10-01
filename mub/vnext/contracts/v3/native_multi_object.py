from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN, StrictBool, StrictNonnegativeInt
from mub.vnext.contracts.enums import Operation, StringEnum
from mub.vnext.contracts.v3.common import (
    FrozenJsonObjectV3,
    FrozenJsonValue,
    MemoryObjectKeyV3,
    StrictFiniteFloat,
    StrictIdentifier,
    StrictNonnegativeInt,
    StrictPositiveInt,
    object_identity,
)
from mub.vnext.contracts.v3.enums import ExecutionStatusV3


class NativeDataProvenanceV3(StringEnum):
    PROVIDER_NATIVE = "provider_native"
    EXTRACTED = "extracted"
    MIXED = "mixed"
    DECLARED = "declared"
    OBSERVED = "observed"


class NativeAtomicityV3(StringEnum):
    ATOMIC = "atomic"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class NativeRetrievalStatusV3(StringEnum):
    RETURNED = "returned"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class NativeProvenanceV3(ImmutableContractModel):
    kind: NativeDataProvenanceV3
    provider_owned: StrictBool
    extractor_id: StrictIdentifier | None = None
    extractor_version: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.kind is NativeDataProvenanceV3.PROVIDER_NATIVE and not self.provider_owned:
            raise ValueError("provider_native provenance must be provider-owned")
        if self.kind is NativeDataProvenanceV3.EXTRACTED and self.provider_owned:
            raise ValueError("extracted provenance cannot be provider-owned")
        if self.kind is NativeDataProvenanceV3.MIXED and self.extractor_id is None:
            raise ValueError("mixed provenance requires an extractor ID")
        if self.extractor_id is not None and self.extractor_version is None:
            raise ValueError("extractor provenance requires an extractor version")
        return self


class NativeSurfaceIdentityV3(ImmutableContractModel):
    candidate_id: StrictIdentifier
    provider_name: StrictIdentifier
    provider_version: StrictIdentifier
    surface_version: StrictIdentifier
    source_revision: StrictIdentifier
    runtime_revision: StrictIdentifier
    configuration_hash: str = Field(pattern=SHA256_PATTERN, strict=True)
    source_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, strict=True)
    runtime_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, strict=True)


class NativeObjectMutationV3(ImmutableContractModel):
    object_key: MemoryObjectKeyV3 = Field(validation_alias=AliasChoices("object_key", "key"))
    value: FrozenJsonValue | None = Field(default=None, validation_alias=AliasChoices("value", "payload"))

    @property
    def value_supplied(self) -> bool:
        return "value" in self.model_fields_set


class NativeMultiObjectMutationRequestV3(ImmutableContractModel):
    request_id: StrictIdentifier = Field(validation_alias=AliasChoices("request_id", "mutation_id"))
    event_id: StrictIdentifier
    sequence_index: StrictNonnegativeInt
    operation: Operation
    objects: tuple[NativeObjectMutationV3, ...] = Field(
        default=(), validation_alias=AliasChoices("objects", "object_mutations", "mutations")
    )
    source_event_ids: tuple[StrictIdentifier, ...] = Field(
        default=(), validation_alias=AliasChoices("source_event_ids", "source_events")
    )
    provenance: NativeProvenanceV3 = Field(validation_alias=AliasChoices("provenance", "data_provenance"))

    @model_validator(mode="before")
    @classmethod
    def _legacy_object_aliases(cls, data):
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        target_keys = values.pop("target_object_keys", None)
        has_object_values = "values" in values
        object_values = values.pop("values", None)
        if target_keys is None:
            return values
        supplied_objects = next((values[name] for name in ("objects", "object_mutations", "mutations") if name in values), None)
        if supplied_objects not in (None, (), []):
            raise ValueError("native target_object_keys cannot be combined with objects")
        if not has_object_values:
            values["objects"] = tuple({"object_key": key} for key in target_keys)
            return values
        object_values = tuple(object_values)
        if len(object_values) != len(target_keys):
            raise ValueError("native values must align with target_object_keys")
        values["objects"] = tuple({"object_key": key, "value": value} for key, value in zip(target_keys, object_values))
        return values

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        objects = self.objects
        identities = [object_identity(item.object_key) for item in objects]
        if len(identities) != len(set(identities)):
            raise ValueError("native mutation object identities must be unique")
        if self.operation is Operation.NOOP:
            if self.objects:
                raise ValueError("native NOOP must have zero target objects")
        elif not self.objects:
            raise ValueError("native mutation requires at least one object")
        if self.operation in {Operation.ADD, Operation.UPDATE} and any(not item.value_supplied for item in self.objects):
            raise ValueError("native ADD/UPDATE requires a value for every object")
        if self.operation is Operation.DELETE and any(item.value_supplied for item in self.objects):
            raise ValueError("native DELETE cannot carry values")
        source_ids = self.source_event_ids or (self.event_id,)
        if len(source_ids) != len(set(source_ids)) or self.event_id not in source_ids:
            raise ValueError("native mutation source events must be unique and include event_id")
        object.__setattr__(self, "source_event_ids", source_ids)
        return self


class NativeObjectOutcomeV3(ImmutableContractModel):
    event_id: StrictIdentifier
    object_key: MemoryObjectKeyV3 = Field(validation_alias=AliasChoices("object_key", "key"))
    operation: Operation
    status: ExecutionStatusV3 = Field(validation_alias=AliasChoices("status", "execution_status"))
    provider_entry_id: StrictIdentifier | None = Field(default=None, validation_alias=AliasChoices("provider_entry_id", "provider_id", "entry_id"))
    source_event_ids: tuple[StrictIdentifier, ...] = Field(default=(), validation_alias=AliasChoices("source_event_ids", "source_events"))
    write_count: StrictNonnegativeInt = 0
    reason: StrictIdentifier | None = None
    error: FrozenJsonValue | None = None
    provenance: NativeProvenanceV3

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        source_ids = self.source_event_ids or (self.event_id,)
        if len(source_ids) != len(set(source_ids)) or self.event_id not in source_ids:
            raise ValueError("native object outcome source events must be unique and include event_id")
        object.__setattr__(self, "source_event_ids", source_ids)
        if self.status is ExecutionStatusV3.EXECUTED:
            if self.reason is not None or self.error is not None:
                raise ValueError("executed native outcomes cannot carry reason or error")
            if self.operation is Operation.NOOP:
                if self.provider_entry_id is not None or self.write_count != 0:
                    raise ValueError("native NOOP outcomes require zero writes and no provider entry")
            elif self.provider_entry_id is None or self.write_count < 1:
                raise ValueError("executed native mutations require provider entry ID and a write")
        elif self.status is ExecutionStatusV3.NO_EFFECT:
            if self.provider_entry_id is not None or self.write_count != 0 or self.reason is None or self.error is not None:
                raise ValueError("native no-effect outcomes require reason and zero writes")
        elif self.status in {ExecutionStatusV3.REJECTED, ExecutionStatusV3.NOT_SUPPORTED}:
            if self.provider_entry_id is not None or self.write_count != 0 or self.reason is None or self.error is not None:
                raise ValueError("rejected native outcomes require reason and zero writes")
        elif self.status is ExecutionStatusV3.FAILED:
            if self.provider_entry_id is not None or self.write_count != 0 or self.error is None:
                raise ValueError("failed native outcomes require error and zero writes")
        return self


class NativeMultiObjectMutationResultV3(ImmutableContractModel):
    request_id: StrictIdentifier = Field(validation_alias=AliasChoices("request_id", "mutation_id"))
    event_id: StrictIdentifier
    operation: Operation
    outcomes: tuple[NativeObjectOutcomeV3, ...] = Field(
        default=(), validation_alias=AliasChoices("outcomes", "object_outcomes", "per_object_outcomes")
    )
    atomicity: NativeAtomicityV3
    provider_operation_id: StrictIdentifier | None = Field(default=None, validation_alias=AliasChoices("provider_operation_id", "provider_request_id"))
    provenance: NativeProvenanceV3 = Field(validation_alias=AliasChoices("provenance", "data_provenance"))

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.operation is Operation.NOOP and self.outcomes:
            raise ValueError("native NOOP results must have zero object outcomes")
        if self.operation is not Operation.NOOP and not self.outcomes:
            raise ValueError("native mutation results require per-object outcomes")
        identities = [object_identity(item.object_key) for item in self.outcomes]
        if len(identities) != len(set(identities)):
            raise ValueError("native result object identities must be unique")
        provider_ids = [item.provider_entry_id for item in self.outcomes if item.provider_entry_id is not None]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("native provider entry IDs must be unique per result")
        if any(item.event_id != self.event_id or item.operation is not self.operation for item in self.outcomes):
            raise ValueError("native outcomes must link result event and operation")
        statuses = {item.status for item in self.outcomes}
        if self.atomicity is NativeAtomicityV3.PARTIAL and len(statuses) < 2:
            raise ValueError("partial atomicity requires mixed per-object statuses")
        if self.atomicity is NativeAtomicityV3.ATOMIC and len(statuses) > 1:
            raise ValueError("atomicity=atomic cannot contain mixed statuses")
        return self


class NativeRetrievalRequestV3(ImmutableContractModel):
    query_id: StrictIdentifier
    query_text: str = Field(strict=True, min_length=1, validation_alias=AliasChoices("query_text", "text"))
    k: StrictPositiveInt
    namespace: StrictIdentifier = Field(validation_alias=AliasChoices("namespace", "runtime_namespace"))
    filters: FrozenJsonObjectV3 = Field(default_factory=dict)
    options: FrozenJsonObjectV3 = Field(default_factory=dict)

    @field_validator("query_text")
    @classmethod
    def _nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("native retrieval query text must not be blank")
        return value


class NativeRetrievalEntryV3(ImmutableContractModel):
    provider_entry_id: StrictIdentifier = Field(validation_alias=AliasChoices("provider_entry_id", "provider_id", "entry_id"))
    object_key: MemoryObjectKeyV3 = Field(validation_alias=AliasChoices("object_key", "key"))
    provider_payload: FrozenJsonValue = Field(validation_alias=AliasChoices("provider_payload", "payload"))
    rank: StrictPositiveInt
    order_index: StrictNonnegativeInt = Field(validation_alias=AliasChoices("order_index", "order", "position"))
    score: StrictFiniteFloat
    source_event_ids: tuple[StrictIdentifier, ...] = Field(min_length=1, validation_alias=AliasChoices("source_event_ids", "source_events"))
    provenance: NativeProvenanceV3 = Field(validation_alias=AliasChoices("provenance", "data_provenance"))

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("native retrieval source event IDs must be unique")
        return self

    @property
    def entry_id(self) -> str:
        return self.provider_entry_id

    @property
    def payload(self):
        return self.provider_payload


class NativeProviderEntryObservationV3(ImmutableContractModel):
    """Raw provider-owned retrieval fields used to audit a normalized trace."""

    provider_entry_id: StrictIdentifier = Field(validation_alias=AliasChoices("provider_entry_id", "provider_id", "entry_id"))
    order_index: StrictNonnegativeInt = Field(validation_alias=AliasChoices("order_index", "order", "position"))
    score: StrictFiniteFloat


class NativeRetrievalTraceV3(ImmutableContractModel):
    query: NativeRetrievalRequestV3
    entries: tuple[NativeRetrievalEntryV3, ...] = Field(
        default=(), validation_alias=AliasChoices("entries", "retrieved_entries")
    )
    status: NativeRetrievalStatusV3 = NativeRetrievalStatusV3.RETURNED
    provenance: NativeProvenanceV3 = Field(validation_alias=AliasChoices("provenance", "data_provenance"))

    @model_validator(mode="before")
    @classmethod
    def _query_aliases(cls, data):
        if isinstance(data, Mapping) and "query" not in data and "query_id" in data:
            values = dict(data)
            query_fields = {name: values.pop(name) for name in (
                "query_id", "query_text", "text", "k", "namespace", "runtime_namespace", "filters", "options"
            ) if name in values}
            values["query"] = query_fields
            return values
        return data

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.status is NativeRetrievalStatusV3.EMPTY and self.entries:
            raise ValueError("empty native retrieval cannot carry entries")
        if self.status in {NativeRetrievalStatusV3.FAILED, NativeRetrievalStatusV3.UNSUPPORTED} and self.entries:
            raise ValueError("failed/unsupported native retrieval cannot carry entries")
        if len(self.entries) > self.query.k:
            raise ValueError("native retrieval cannot exceed requested k")
        ids = [entry.provider_entry_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("native retrieval provider entry IDs must be unique")
        ranks = tuple(entry.rank for entry in self.entries)
        orders = tuple(entry.order_index for entry in self.entries)
        if ranks != tuple(range(1, len(self.entries) + 1)):
            raise ValueError("native retrieval ranks must preserve provider order")
        if orders != tuple(range(len(self.entries))):
            raise ValueError("native retrieval order indices must preserve provider order")
        return self

    @property
    def query_id(self) -> str:
        return self.query.query_id

    @property
    def retrieved_entries(self) -> tuple[NativeRetrievalEntryV3, ...]:
        return self.entries

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(entry.rank for entry in self.entries)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(entry.score for entry in self.entries)


class NativeResetResultV3(ImmutableContractModel):
    namespace: StrictIdentifier
    success: StrictBool
    provider_operation_id: StrictIdentifier | None = None
    provenance: NativeProvenanceV3
    error: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.success and self.error is not None:
            raise ValueError("successful native reset cannot carry error")
        if not self.success and self.error is None:
            raise ValueError("failed native reset requires error")
        return self


class NativeCapabilityDeclarationV3(ImmutableContractModel):
    direct_provider_crud: StrictBool = False
    provider_owned_collection: StrictBool = False
    supports_multi_object_mutation: StrictBool = False
    supports_multi_object_retrieval: StrictBool = False
    supports_add: StrictBool = False
    supports_update: StrictBool = False
    supports_noop: StrictBool = False
    supports_delete: StrictBool = False
    supports_atomic_mutation: StrictBool = False
    supports_isolated_reset: StrictBool = False
    exports_provider_entry_ids: StrictBool = False
    exports_provider_payloads: StrictBool = False
    exports_provider_order: StrictBool = False
    exports_provider_scores: StrictBool = False
    exports_source_event_linkage: StrictBool = False
    visible_only: StrictBool = False
    uses_local_authoritative_store: StrictBool = False
    uses_local_reranking: StrictBool = False
    uses_local_deduplication: StrictBool = False
    uses_local_filtering: StrictBool = False
    hidden_target_injection: StrictBool = False
    supports_native_answer: Literal[False] = False

    @model_validator(mode="after")
    def _native_boundary(self) -> Self:
        forbidden = {
            "uses_local_authoritative_store": self.uses_local_authoritative_store,
            "uses_local_reranking": self.uses_local_reranking,
            "uses_local_deduplication": self.uses_local_deduplication,
            "uses_local_filtering": self.uses_local_filtering,
            "hidden_target_injection": self.hidden_target_injection,
        }
        active = [name for name, enabled in forbidden.items() if enabled]
        if active:
            labels = {
                "uses_local_authoritative_store": "local authoritative store",
                "uses_local_reranking": "local reranking",
                "uses_local_deduplication": "local deduplication",
                "uses_local_filtering": "local filtering",
                "hidden_target_injection": "hidden target injection",
            }
            raise ValueError("native boundary forbids " + ", ".join(labels[name] for name in active))
        if self.provider_owned_collection and not self.direct_provider_crud:
            raise ValueError("provider-owned collection requires direct provider CRUD")
        if self.direct_provider_crud and not self.provider_owned_collection:
            raise ValueError("direct provider CRUD requires a provider-owned collection")
        if self.supports_multi_object_mutation and not self.direct_provider_crud:
            raise ValueError("multi-object mutation requires direct provider CRUD")
        if self.supports_multi_object_retrieval and not self.direct_provider_crud:
            raise ValueError("multi-object retrieval requires direct provider CRUD")
        if self.supports_atomic_mutation and not self.supports_multi_object_mutation:
            raise ValueError("atomic mutation requires multi-object mutation support")
        if any(getattr(self, name) for name in ("supports_add", "supports_update", "supports_noop", "supports_delete")) and not self.supports_multi_object_mutation:
            raise ValueError("per-operation support requires multi-object mutation support")
        return self


# Short aliases are intentionally additive; the longer names remain canonical in JSON.
NativeMutationRequestV3 = NativeMultiObjectMutationRequestV3
NativeMutationResultV3 = NativeMultiObjectMutationResultV3
NativeObjectOutcomeStatusV3 = ExecutionStatusV3
NativeProviderEntryObservation = NativeProviderEntryObservationV3
NativeRetrievalEntry = NativeRetrievalEntryV3
NativeRetrievalTrace = NativeRetrievalTraceV3
NativeAtomicity = NativeAtomicityV3
NativeCapabilityDeclaration = NativeCapabilityDeclarationV3
NativeDataProvenance = NativeDataProvenanceV3
NativeMutationRequest = NativeMultiObjectMutationRequestV3
NativeMutationResult = NativeMultiObjectMutationResultV3
NativeObjectMutation = NativeObjectMutationV3
NativeObjectOutcome = NativeObjectOutcomeV3
NativeProvenance = NativeProvenanceV3
NativeResetResult = NativeResetResultV3
NativeRetrievalEntryV3Alias = NativeRetrievalEntryV3
NativeRetrievalRequest = NativeRetrievalRequestV3
NativeRetrievalTraceV3Alias = NativeRetrievalTraceV3
NativeSurfaceIdentity = NativeSurfaceIdentityV3


__all__ = [
    "NativeAtomicity",
    "NativeAtomicityV3",
    "NativeCapabilityDeclaration",
    "NativeCapabilityDeclarationV3",
    "NativeDataProvenance",
    "NativeDataProvenanceV3",
    "NativeMultiObjectMutationRequestV3",
    "NativeMultiObjectMutationResultV3",
    "NativeMutationRequest",
    "NativeMutationRequestV3",
    "NativeMutationResult",
    "NativeMutationResultV3",
    "NativeObjectMutation",
    "NativeObjectMutationV3",
    "NativeObjectOutcome",
    "NativeObjectOutcomeStatusV3",
    "NativeObjectOutcomeV3",
    "NativeProviderEntryObservation",
    "NativeProviderEntryObservationV3",
    "NativeProvenance",
    "NativeProvenanceV3",
    "NativeResetResult",
    "NativeResetResultV3",
    "NativeRetrievalEntry",
    "NativeRetrievalEntryV3",
    "NativeRetrievalEntryV3Alias",
    "NativeRetrievalRequest",
    "NativeRetrievalRequestV3",
    "NativeRetrievalStatusV3",
    "NativeRetrievalTrace",
    "NativeRetrievalTraceV3",
    "NativeRetrievalTraceV3Alias",
    "NativeSurfaceIdentity",
    "NativeSurfaceIdentityV3",
]
