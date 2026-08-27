from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.io import canonical_json_bytes

LETTA_PACKAGE_VERSION = "0.16.8"
LETTA_SOURCE_COMMIT = "1131535716e8a31c9a437f8695e25ac98f203a24"
LETTA_PROVIDER_CONTRACT_VERSION = "memupdatebench.external.letta.v1"
StrictSha256 = Annotated[str, Field(pattern=SHA256_PATTERN, strict=True)]
StrictCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$", strict=True)]

_LETTA_PACKAGE_LOCK = (
    "letta",
    "0.16.8",
    "0.16.8",
    "https://github.com/letta-ai/letta",
    "https://github.com/letta-ai/letta/releases/tag/0.16.8",
    "1131535716e8a31c9a437f8695e25ac98f203a24",
    "letta-0.16.8-py3-none-any.whl",
    "https://files.pythonhosted.org/packages/10/20/"
    "eedf6bd8b55e97edf9cbcaea8f575b83157737d6483ddba6b304babc0a4a/"
    "letta-0.16.8-py3-none-any.whl",
    "2d200cd13212c9232650c7efa16c0a4682ce5e89af8a0e53f3fbd09fc34091d3",
    1541384,
    "Apache-2.0",
    ">=3.11, <3.14",
)


class LettaPackageProvenanceV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.v1"] = (
        LETTA_PROVIDER_CONTRACT_VERSION
    )
    package_name: Literal["letta"]
    package_version: Literal["0.16.8"]
    release_tag: Literal["0.16.8"]
    repository_url: StrictIdentifier
    release_url: StrictIdentifier
    source_commit: StrictCommit
    wheel_filename: Literal["letta-0.16.8-py3-none-any.whl"]
    wheel_url: StrictIdentifier
    wheel_sha256: StrictSha256
    wheel_size_bytes: Literal[1541384]
    license_id: Literal["Apache-2.0"]
    python_requires: Literal[">=3.11, <3.14"]

    @model_validator(mode="after")
    def _frozen_release(self) -> Self:
        if _package_lock_tuple(self) != _LETTA_PACKAGE_LOCK:
            raise ValueError("frozen Letta package provenance does not match")
        return self


class LettaAdapterConfigurationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.letta.v1"] = (
        LETTA_PROVIDER_CONTRACT_VERSION
    )
    run_id: StrictIdentifier
    candidate_id: Literal["letta_0_16_8_block_profile"] = (
        "letta_0_16_8_block_profile"
    )
    package_provenance: LettaPackageProvenanceV1
    mode: Literal["direct_block_profile"] = "direct_block_profile"
    namespace_root: Literal["memupdatebench"] = "memupdatebench"
    block_label: Literal["memupdatebench_profile"] = "memupdatebench_profile"
    native_operations: tuple[Literal["create", "update", "delete", "search"], ...] = (
        "create",
        "update",
        "delete",
        "search",
    )
    llm_required: Literal[False] = False
    network_required: Literal[False] = False
    credentials_required: Literal[False] = False

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        package = validate_letta_package_provenance(self.package_provenance)
        object.__setattr__(self, "package_provenance", package)
        if self.native_operations != ("create", "update", "delete", "search"):
            raise ValueError("Letta native operations must be exact and ordered")
        return self


def _package_lock_tuple(
    provenance: LettaPackageProvenanceV1,
) -> tuple[object, ...]:
    return (
        provenance.package_name,
        provenance.package_version,
        provenance.release_tag,
        provenance.repository_url,
        provenance.release_url,
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


def fixed_letta_package_provenance() -> LettaPackageProvenanceV1:
    return LettaPackageProvenanceV1(
        package_name=_LETTA_PACKAGE_LOCK[0],
        package_version=_LETTA_PACKAGE_LOCK[1],
        release_tag=_LETTA_PACKAGE_LOCK[2],
        repository_url=_LETTA_PACKAGE_LOCK[3],
        release_url=_LETTA_PACKAGE_LOCK[4],
        source_commit=_LETTA_PACKAGE_LOCK[5],
        wheel_filename=_LETTA_PACKAGE_LOCK[6],
        wheel_url=_LETTA_PACKAGE_LOCK[7],
        wheel_sha256=_LETTA_PACKAGE_LOCK[8],
        wheel_size_bytes=_LETTA_PACKAGE_LOCK[9],
        license_id=_LETTA_PACKAGE_LOCK[10],
        python_requires=_LETTA_PACKAGE_LOCK[11],
    )


def validate_letta_package_provenance(
    value: LettaPackageProvenanceV1,
) -> LettaPackageProvenanceV1:
    return _revalidate_exact(value, LettaPackageProvenanceV1, "frozen Letta package")


def build_letta_adapter_configuration(*, run_id: str) -> LettaAdapterConfigurationV1:
    return LettaAdapterConfigurationV1(
        run_id=run_id,
        package_provenance=fixed_letta_package_provenance(),
    )


def compute_letta_configuration_hash(
    configuration: LettaAdapterConfigurationV1,
) -> str:
    validated = _revalidate_exact(
        configuration,
        LettaAdapterConfigurationV1,
        "Letta adapter configuration",
    )
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


__all__ = [
    "LETTA_PACKAGE_VERSION",
    "LETTA_PROVIDER_CONTRACT_VERSION",
    "LETTA_SOURCE_COMMIT",
    "LettaAdapterConfigurationV1",
    "LettaPackageProvenanceV1",
    "build_letta_adapter_configuration",
    "compute_letta_configuration_hash",
    "fixed_letta_package_provenance",
    "validate_letta_package_provenance",
]
