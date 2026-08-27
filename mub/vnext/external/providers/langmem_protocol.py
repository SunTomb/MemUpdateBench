from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.contracts.v3.common import StrictFiniteFloat, StrictIdentifier

StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]


class LangMemWorkerHealthV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    package_name: Literal["langmem"]
    package_version: Literal["0.0.30"]
    source_commit: Literal["29cbe41e58528f92e9efa773c12e15c47be3808c"]
    license_id: Literal["MIT"]
    configuration_hash: StrictSha256


class LangMemWorkerCloseResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    closed: Literal[True] = True


class LangMemWorkerResetResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    namespace: StrictIdentifier
    success: Literal[True] = True


class LangMemWorkerEntryV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
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
            raise ValueError("LangMem entry source event IDs must be unique")
        return self


class LangMemWorkerEntryListV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    entries: tuple[LangMemWorkerEntryV1, ...] = ()

    @model_validator(mode="after")
    def _unique_entries(self) -> Self:
        ids = tuple(entry.entry_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("LangMem entry export IDs must be unique")
        return self


class LangMemWorkerMutationResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    event_id: StrictIdentifier
    effective_operation: Literal["add", "update", "delete", "noop"]
    entry_id: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.effective_operation == "noop" and self.entry_id is not None:
            raise ValueError("LangMem NOOP cannot identify an entry")
        if self.effective_operation != "noop" and self.entry_id is None:
            raise ValueError("LangMem mutation requires a stable entry ID")
        return self


class LangMemWorkerRetrievalResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.worker.v1"] = (
        "memupdatebench.external.langmem.worker.v1"
    )
    query_id: StrictIdentifier
    entries: tuple[LangMemWorkerEntryV1, ...] = ()
    scores: tuple[StrictFiniteFloat, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.entries) != len(self.scores):
            raise ValueError("LangMem retrieval scores must match entries")
        return self


__all__ = [
    "LangMemWorkerCloseResultV1",
    "LangMemWorkerEntryListV1",
    "LangMemWorkerEntryV1",
    "LangMemWorkerHealthV1",
    "LangMemWorkerMutationResultV1",
    "LangMemWorkerResetResultV1",
    "LangMemWorkerRetrievalResultV1",
]
