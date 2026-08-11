from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictBool,
)
from mub.vnext.contracts.v3.common import (
    FrozenJsonObjectV3,
    StrictFiniteFloat,
    StrictIdentifier,
)

StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]


class Mem0WorkerHealthV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    package_name: Literal["mem0ai"]
    package_version: Literal["2.0.17"]
    collection_name: StrictIdentifier
    configuration_hash: StrictSha256


class Mem0WorkerCloseResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    closed: Literal[True] = True


class Mem0WorkerResetResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    namespace: StrictIdentifier
    success: StrictBool


class Mem0WorkerEntryV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    entry_id: StrictIdentifier
    content: str = Field(strict=True)
    created_at: str | None = Field(default=None, strict=True)
    updated_at: str | None = Field(default=None, strict=True)
    source_event_ids: tuple[StrictIdentifier, ...] = ()
    native_metadata: FrozenJsonObjectV3 = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_sources(self) -> Self:
        if len(self.source_event_ids) != len(set(self.source_event_ids)):
            raise ValueError("Mem0 entry source event IDs must be unique")
        return self


class Mem0WorkerEntryListV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    entries: tuple[Mem0WorkerEntryV1, ...] = ()

    @model_validator(mode="after")
    def _unique_entries(self) -> Self:
        ids = tuple(entry.entry_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("Mem0 entry export IDs must be unique")
        return self


class Mem0WorkerIngestResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    event_id: StrictIdentifier
    effective_operation: Literal["add", "noop"]
    affected_entry_ids: tuple[StrictIdentifier, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.affected_entry_ids) != len(set(self.affected_entry_ids)):
            raise ValueError("Mem0 affected entry IDs must be unique")
        if self.effective_operation == "add" and not self.affected_entry_ids:
            raise ValueError("Mem0 effective add requires affected entry IDs")
        if self.effective_operation == "noop" and self.affected_entry_ids:
            raise ValueError("Mem0 effective noop cannot affect entries")
        return self


class Mem0WorkerRetrievalResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.mem0.worker.v1"] = (
        "memupdatebench.external.mem0.worker.v1"
    )
    query_id: StrictIdentifier
    entries: tuple[Mem0WorkerEntryV1, ...] = ()
    scores: tuple[StrictFiniteFloat, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if len(self.entries) != len(self.scores):
            raise ValueError("Mem0 retrieval scores must match entries")
        ids = tuple(entry.entry_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("Mem0 retrieval entry IDs must be unique")
        return self


__all__ = [
    "Mem0WorkerCloseResultV1",
    "Mem0WorkerEntryListV1",
    "Mem0WorkerEntryV1",
    "Mem0WorkerHealthV1",
    "Mem0WorkerIngestResultV1",
    "Mem0WorkerResetResultV1",
    "Mem0WorkerRetrievalResultV1",
]
