from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import (
    ArtifactRef,
    ImmutableContractModel,
    SHA256_PATTERN,
    StrictNonnegativeInt,
)
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.external.registry import validate_artifact_provenance


StrictSha256 = Annotated[str, Field(strict=True, pattern=SHA256_PATTERN)]
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMMUTABLE_CORE_ROOT = _PROJECT_ROOT / "data" / "vnext" / "core" / "v3"


class RawPayloadLicenseStatus(str, Enum):
    PRIVATE = "private"
    LICENSE_UNCERTAIN = "license_uncertain"
    REDISTRIBUTABLE = "redistributable"


class PrivateRawArtifactRefV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.external.private_raw_ref.v1"] = (
        "memupdatebench.external.private_raw_ref.v1"
    )
    sha256: StrictSha256
    size_bytes: StrictNonnegativeInt
    media_type: str = Field(strict=True, min_length=1)
    storage_class: Literal["private_raw"] = "private_raw"
    license_status: RawPayloadLicenseStatus = RawPayloadLicenseStatus.PRIVATE


class NormalizedArtifactRefV1(ArtifactRef):
    private_raw_hashes: tuple[StrictSha256, ...] = ()
    redaction_version: StrictIdentifier
    storage_class: Literal["redistributable_normalized"] = (
        "redistributable_normalized"
    )

    @model_validator(mode="after")
    def _portable_and_unique(self) -> Self:
        validate_artifact_provenance(
            ArtifactRef(
                path=self.path,
                sha256=self.sha256,
                media_type=self.media_type,
                record_count=self.record_count,
            )
        )
        if not self.private_raw_hashes:
            raise ValueError(
                "normalized artifacts require at least one private raw hash"
            )
        if len(self.private_raw_hashes) != len(set(self.private_raw_hashes)):
            raise ValueError("private raw hashes must be unique")
        return self


@dataclass(frozen=True)
class ExternalArtifactRootsV1:
    private_raw_root: Path
    normalized_root: Path
    private_raw_identity: tuple[int, int]
    normalized_identity: tuple[int, int]


def _is_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def assert_no_reparse_components(path: str | Path) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and _is_reparse_point(current):
            raise ValueError("external artifact root contains a reparse point")


def _identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_external_artifact_roots(
    private_raw_root: str | Path,
    normalized_root: str | Path,
) -> ExternalArtifactRootsV1:
    private_path = Path(private_raw_root).absolute()
    normalized_path = Path(normalized_root).absolute()
    for path in (private_path, normalized_path):
        assert_no_reparse_components(path)
        if not path.is_dir() or _is_reparse_point(path):
            raise ValueError("external artifact roots must be real directories")
    private_resolved = private_path.resolve(strict=True)
    normalized_resolved = normalized_path.resolve(strict=True)
    if (
        _contains(private_resolved, normalized_resolved)
        or _contains(normalized_resolved, private_resolved)
    ):
        raise ValueError("private and normalized artifact roots must not overlap")
    if _IMMUTABLE_CORE_ROOT.exists():
        immutable_root = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
        if any(
            _contains(left, right)
            for left, right in (
                (immutable_root, private_resolved),
                (private_resolved, immutable_root),
                (immutable_root, normalized_resolved),
                (normalized_resolved, immutable_root),
            )
        ):
            raise ValueError(
                "external artifact roots must be outside immutable Core"
            )
    return ExternalArtifactRootsV1(
        private_raw_root=private_resolved,
        normalized_root=normalized_resolved,
        private_raw_identity=_identity(private_resolved),
        normalized_identity=_identity(normalized_resolved),
    )


__all__ = [
    "ExternalArtifactRootsV1",
    "NormalizedArtifactRefV1",
    "PrivateRawArtifactRefV1",
    "RawPayloadLicenseStatus",
    "assert_no_reparse_components",
    "validate_external_artifact_roots",
]
