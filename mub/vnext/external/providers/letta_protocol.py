from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.contracts.v3.common import StrictFiniteFloat, StrictIdentifier

StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]


class LettaWorkerHealthV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    package_name: Literal["letta"]
    package_version: Literal["0.16.8"]
    source_commit: Literal["1131535716e8a31c9a437f8695e25ac98f203a24"]
    license_id: Literal["Apache-2.0"]
    configuration_hash: StrictSha256


class LettaWorkerCloseResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    closed: Literal[True] = True


class LettaWorkerResetResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    namespace: StrictIdentifier
    success: Literal[True] = True


class LettaWorkerEntryV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    entry_id: StrictIdentifier
    canonical_object_id: StrictIdentifier
    content: str = Field(strict=True)
    value: JsonValue
    created_at: str | None = Field(default=None, strict=True)
    updated_at: str | None = Field(default=None, strict=True)
    source_event_ids: tuple[StrictIdentifier, ...] = ()
    sequence_index: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _unique_source_ids(self) -> Self:
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("Letta entry source event IDs must be unique")
        return self


class LettaWorkerEntryListV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    entries: tuple[LettaWorkerEntryV1, ...] = ()

    @model_validator(mode="after")
    def _unique_entries(self) -> Self:
        ids = tuple(entry.entry_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("Letta entry export IDs must be unique")
        return self


class LettaWorkerMutationResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    event_id: StrictIdentifier
    effective_operation: Literal["add", "update", "delete", "noop"]
    entry_id: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.effective_operation == "noop" and self.entry_id is not None:
            raise ValueError("Letta NOOP cannot identify an entry")
        if self.effective_operation != "noop" and self.entry_id is None:
            raise ValueError("Letta mutation requires a stable entry ID")
        return self


class LettaWorkerRetrievalResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.worker.v1"] = (
        "memupdatebench.external.letta.worker.v1"
    )
    query_id: StrictIdentifier
    entries: tuple[LettaWorkerEntryV1, ...] = ()
    scores: tuple[StrictFiniteFloat, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.entries) != len(self.scores):
            raise ValueError("Letta retrieval scores must match entries")
        return self


__all__ = [
    "LettaWorkerCloseResultV1",
    "LettaWorkerEntryListV1",
    "LettaWorkerEntryV1",
    "LettaWorkerHealthV1",
    "LettaWorkerMutationResultV1",
    "LettaWorkerResetResultV1",
    "LettaWorkerRetrievalResultV1",
]
