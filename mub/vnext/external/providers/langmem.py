from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.io import canonical_json_bytes

LANGMEM_PACKAGE_VERSION = "0.0.30"
LANGMEM_SOURCE_COMMIT = "29cbe41e58528f92e9efa773c12e15c47be3808c"
LANGMEM_PROVIDER_CONTRACT_VERSION = "memupdatebench.external.langmem.v1"
LANGMEM_INSTALLED_CONTENT_SHA256 = (
    "3e6d6cf4d81a1cd77ebc2b98b68b17043841b34239a2499c3a91a3cdc601847a"
)
LANGMEM_INSTALLED_CONTENT_FILE_COUNT = 27
StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]
StrictCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$", strict=True)]

_LANGMEM_PACKAGE_LOCK = (
    "langmem",
    "0.0.30",
    "https://github.com/langchain-ai/langmem",
    "29cbe41e58528f92e9efa773c12e15c47be3808c",
    "langmem-0.0.30-py3-none-any.whl",
    "https://files.pythonhosted.org/packages/ae/08/"
    "c7bc95456f6e02819e9fed56aa01578c3b8ee1a47b520994efc37e9febcc/"
    "langmem-0.0.30-py3-none-any.whl",
    "142f040014493eebd67e1055c0642f9ab38868b5b1fde5c8f2d39add57f4ba5b",
    67122,
    "MIT",
    ">=3.10",
)


class LangMemPackageProvenanceV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.v1"] = (
        LANGMEM_PROVIDER_CONTRACT_VERSION
    )
    package_name: Literal["langmem"]
    package_version: Literal["0.0.30"]
    repository_url: StrictIdentifier
    source_commit: StrictCommit
    wheel_filename: Literal["langmem-0.0.30-py3-none-any.whl"]
    wheel_url: StrictIdentifier
    wheel_sha256: StrictSha256
    wheel_size_bytes: Literal[67122]
    license_id: Literal["MIT"]
    python_requires: Literal[">=3.10"]

    @model_validator(mode="after")
    def _frozen_release(self) -> Self:
        if _package_lock_tuple(self) != _LANGMEM_PACKAGE_LOCK:
            raise ValueError("frozen LangMem package provenance does not match")
        return self


class LangMemAdapterConfigurationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.langmem.v1"] = (
        LANGMEM_PROVIDER_CONTRACT_VERSION
    )
    run_id: StrictIdentifier
    candidate_id: Literal["langmem_0_0_30_profile"] = (
        "langmem_0_0_30_profile"
    )
    package_provenance: LangMemPackageProvenanceV1
    mode: Literal["profile_single_record"] = "profile_single_record"
    collection_mode_supported: Literal[False] = False
    store_backend: Literal["langgraph_in_memory_store"] = (
        "langgraph_in_memory_store"
    )
    namespace_root: Literal["memupdatebench"] = "memupdatebench"
    native_operations: tuple[
        Literal["create", "update", "delete", "search"], ...
    ] = ("create", "update", "delete", "search")
    llm_required: Literal[False] = False
    network_required: Literal[False] = False
    credentials_required: Literal[False] = False

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        package = validate_langmem_package_provenance(self.package_provenance)
        object.__setattr__(self, "package_provenance", package)
        if self.native_operations != ("create", "update", "delete", "search"):
            raise ValueError("LangMem native operations must be exact and ordered")
        return self


def _package_lock_tuple(
    provenance: LangMemPackageProvenanceV1,
) -> tuple[object, ...]:
    return (
        provenance.package_name,
        provenance.package_version,
        provenance.repository_url,
        provenance.source_commit,
        provenance.wheel_filename,
        provenance.wheel_url,
        provenance.wheel_sha256,
        provenance.wheel_size_bytes,
        provenance.license_id,
        provenance.python_requires,
    )


def _revalidate_exact(value: object, expected_type: type, label: str):
    if type(value) is not expected_type:
        raise ValueError(f"{label} trust-boundary requires exact type")
    try:
        payload = {
            field_name: value.__dict__[field_name]
            for field_name in expected_type.model_fields
        }
        return expected_type.model_validate(payload, strict=True)
    except Exception:
        raise ValueError(f"{label} trust-boundary validation failed") from None


def fixed_langmem_package_provenance() -> LangMemPackageProvenanceV1:
    return LangMemPackageProvenanceV1(
        package_name=_LANGMEM_PACKAGE_LOCK[0],
        package_version=_LANGMEM_PACKAGE_LOCK[1],
        repository_url=_LANGMEM_PACKAGE_LOCK[2],
        source_commit=_LANGMEM_PACKAGE_LOCK[3],
        wheel_filename=_LANGMEM_PACKAGE_LOCK[4],
        wheel_url=_LANGMEM_PACKAGE_LOCK[5],
        wheel_sha256=_LANGMEM_PACKAGE_LOCK[6],
        wheel_size_bytes=_LANGMEM_PACKAGE_LOCK[7],
        license_id=_LANGMEM_PACKAGE_LOCK[8],
        python_requires=_LANGMEM_PACKAGE_LOCK[9],
    )


def validate_langmem_package_provenance(
    value: LangMemPackageProvenanceV1,
) -> LangMemPackageProvenanceV1:
    return _revalidate_exact(
        value,
        LangMemPackageProvenanceV1,
        "frozen LangMem package",
    )


def build_langmem_adapter_configuration(
    *,
    run_id: str,
) -> LangMemAdapterConfigurationV1:
    return LangMemAdapterConfigurationV1(
        run_id=run_id,
        package_provenance=fixed_langmem_package_provenance(),
    )


def compute_langmem_configuration_hash(
    configuration: LangMemAdapterConfigurationV1,
) -> str:
    validated = _revalidate_exact(
        configuration,
        LangMemAdapterConfigurationV1,
        "LangMem adapter configuration",
    )
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


__all__ = [
    "LANGMEM_INSTALLED_CONTENT_FILE_COUNT",
    "LANGMEM_INSTALLED_CONTENT_SHA256",
    "LANGMEM_PACKAGE_VERSION",
    "LANGMEM_PROVIDER_CONTRACT_VERSION",
    "LANGMEM_SOURCE_COMMIT",
    "LangMemAdapterConfigurationV1",
    "LangMemPackageProvenanceV1",
    "build_langmem_adapter_configuration",
    "compute_langmem_configuration_hash",
    "fixed_langmem_package_provenance",
    "validate_langmem_package_provenance",
]
