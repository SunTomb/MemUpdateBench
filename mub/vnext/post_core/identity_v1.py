from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal, Mapping

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import (
    ArtifactIndexV1,
    CandidateIdentityState,
    SHA256_PATTERN,
    canonical_bytes,
)
from mub.vnext.post_core.provenance_v1 import validate_secret_free


EXPECTED_IDENTITY_EVIDENCE_SHA256 = (
    "9e3780ed3d4303bda7bbd27865df89fcb384041da64af56107c8c5b7abf0a4f0"
)
EXPECTED_PHASE0_INDEX_SHA256 = (
    "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd"
)
EXPECTED_IDENTITY_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)


class IdentityArtifactEvidenceV1(ImmutableContractModel):
    role: Literal[
        "weight_shard",
        "tokenizer",
        "quantized_target",
        "vision_projector",
        "speculative_drafter",
    ]
    path: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class IdentityEvidenceRecordV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.identity-record.v1"] = (
        "memupdatebench.post-core.identity-record.v1"
    )
    registry_key: StrictStr
    state: CandidateIdentityState
    evidence_class: Literal[
        "official_pinned_repository",
        "official_pinned_api_id",
        "official_stable_api_id_with_response_version",
        "official_mutable_api_alias",
        "not_verified_in_official_catalog",
    ]
    official_owner: StrictStr | None = None
    official_model_id: StrictStr | None = None
    revision: StrictStr | None = None
    license_id: StrictStr | None = None
    architecture: StrictStr | None = None
    parameter_count: StrictInt | None = Field(default=None, gt=0)
    base_model_id: StrictStr | None = None
    base_revision: StrictStr | None = None
    quantization_id: StrictStr | None = None
    identifier_stability: Literal[
        "pinned_snapshot", "stable_version", "mutable_alias", "unverified"
    ]
    mutable_identifier: StrictBool | None = None
    response_identity_field: Literal["model", "modelVersion"] | None = None
    source_urls: tuple[StrictStr, ...]
    artifacts: tuple[IdentityArtifactEvidenceV1, ...] = ()
    blockers: tuple[StrictStr, ...]

    @field_validator("source_urls")
    @classmethod
    def _official_https_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("identity evidence requires official source URLs")
        for url in value:
            if not url.startswith("https://") or any(marker in url for marker in ("?", "#", "@")):
                raise ValueError("identity source URL must be plain HTTPS without query, fragment, or userinfo")
        if len(set(value)) != len(value):
            raise ValueError("identity source URLs must be unique")
        return value

    @model_validator(mode="after")
    def _evidence_boundary(self) -> "IdentityEvidenceRecordV1":
        if self.state is CandidateIdentityState.QUALIFIED:
            raise ValueError("document identity evidence cannot qualify a model")
        if self.evidence_class == "not_verified_in_official_catalog":
            if self.state is not CandidateIdentityState.PENDING_OFFICIAL_IDENTITY:
                raise ValueError("unverified identity must remain pending official identity")
            if any(
                value is not None
                for value in (
                    self.official_model_id,
                    self.revision,
                    self.license_id,
                    self.architecture,
                    self.parameter_count,
                    self.response_identity_field,
                )
            ) or self.artifacts:
                raise ValueError("unverified identity cannot carry invented model facts")
        elif self.evidence_class == "official_pinned_repository":
            if self.state is not CandidateIdentityState.PENDING_LOCAL_SNAPSHOT:
                raise ValueError("pinned open repository must wait for a local snapshot")
            if not all(
                (
                    self.official_owner,
                    self.official_model_id,
                    self.revision,
                    self.license_id,
                    self.architecture,
                    self.artifacts,
                )
            ):
                raise ValueError("pinned repository evidence is incomplete")
            if self.identifier_stability != "pinned_snapshot" or self.mutable_identifier is not False:
                raise ValueError("pinned repository identity cannot be mutable")
        elif self.evidence_class in {
            "official_pinned_api_id",
            "official_stable_api_id_with_response_version",
        }:
            if self.state is not CandidateIdentityState.READY_FOR_PROVIDER_PREFLIGHT:
                raise ValueError("verified closed identity must enter only provider preflight")
            if not self.official_owner or not self.official_model_id or not self.response_identity_field:
                raise ValueError("closed identity evidence is incomplete")
            if self.artifacts or self.license_id or self.architecture or self.parameter_count:
                raise ValueError("closed identity evidence cannot invent open-weight facts")
        elif self.evidence_class == "official_mutable_api_alias":
            if self.state is not CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION:
                raise ValueError("mutable API alias must remain pending provider qualification")
            if not self.official_model_id or self.mutable_identifier is not True or self.revision is not None:
                raise ValueError("mutable API alias evidence is inconsistent")
        return self


class IdentityEvidenceBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.identity-evidence.v1"] = (
        "memupdatebench.post-core.identity-evidence.v1"
    )
    evidence_id: Literal["memupdatebench.post-core.identity-evidence.2026-08-21.v1"]
    retrieved_on: Literal["2026-08-21"]
    phase0_index_sha256: Literal[
        "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd"
    ]
    records: tuple[IdentityEvidenceRecordV1, ...]

    @model_validator(mode="after")
    def _frozen_candidate_evidence(self) -> "IdentityEvidenceBundleV1":
        if tuple(row.registry_key for row in self.records) != EXPECTED_IDENTITY_KEYS:
            raise ValueError("identity evidence candidate order differs from frozen Phase 0 registry")
        rows = {row.registry_key: row for row in self.records}
        expected_open = {
            "qwen35_9b_bf16": (
                "Qwen/Qwen3.5-9B",
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "Qwen3_5ForConditionalGeneration",
            ),
            "meta_muse_glimmer_30b_int4": (
                "meta-models/Muse-Glimmer-30B-GGUF",
                "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
                "MuseGlimmerForConditionalGeneration",
            ),
            "meta_muse_glimmer_30b_bf16": (
                "meta-models/Muse-Glimmer-30B",
                "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
                "MuseGlimmerForConditionalGeneration",
            ),
        }
        for key, (model_id, revision, architecture) in expected_open.items():
            row = rows[key]
            if (row.official_model_id, row.revision, row.architecture, row.license_id) != (
                model_id,
                revision,
                architecture,
                "Apache-2.0",
            ):
                raise ValueError(f"{key} official repository evidence differs")
        expected_ready = {
            "claude_sonnet_4_6": ("Anthropic", "claude-sonnet-4-6", "model"),
            "claude_opus_4_8": ("Anthropic", "claude-opus-4-8", "model"),
            "gemini_3_6_flash": ("Google", "gemini-3.6-flash", "modelVersion"),
        }
        for key, expected in expected_ready.items():
            row = rows[key]
            if (row.official_owner, row.official_model_id, row.response_identity_field) != expected:
                raise ValueError(f"{key} provider identity evidence differs")
        grok = rows["grok_4_5"]
        if (
            grok.official_model_id != "grok-4.5"
            or grok.state is not CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION
            or grok.identifier_stability != "mutable_alias"
        ):
            raise ValueError("Grok 4.5 must remain pending until a dated provider identity is verified")
        gpt = rows["gpt_5_5"]
        if (
            gpt.state is not CandidateIdentityState.PENDING_OFFICIAL_IDENTITY
            or gpt.official_model_id is not None
            or gpt.evidence_class != "not_verified_in_official_catalog"
        ):
            raise ValueError("GPT-5.5 is not verified in the official provider catalog")
        validate_secret_free(self.model_dump(mode="json"), read_environment=False)
        return self


class IdentityEvidenceReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.identity-receipt.v1"] = (
        "memupdatebench.post-core.identity-receipt.v1"
    )
    evidence_id: StrictStr
    evidence_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    phase0_index_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    candidate_count: Literal[8] = 8
    state_counts: Mapping[StrictStr, StrictInt]
    provider_calls: Literal[0] = 0
    model_loads: Literal[0] = 0
    network_calls: Literal[0] = 0
    executable_calls: Literal[0] = 0



def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_reparse_components(path: Path) -> None:
    selected = Path(path)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    selected = Path(os.path.normpath(str(selected)))
    current = Path(selected.anchor)
    for part in selected.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise ValueError(f"identity source path contains a link or reparse component: {current}")


def _canonical_model_file(
    path: Path,
    model_type,
    label: str,
    *,
    exclude: set[str] | None = None,
):
    _reject_reparse_components(Path(path))
    selected = Path(path).resolve(strict=True)
    if _is_reparse(selected) or not selected.is_file() or selected.stat().st_nlink != 1:
        raise ValueError(f"{label} must be a regular single-link file")
    raw = selected.read_bytes()
    model = model_type.model_validate_json(raw)
    if canonical_bytes(model, exclude=exclude) != raw:
        raise ValueError(f"{label} must use canonical JSON bytes")
    return model, raw


def load_identity_evidence_v1(
    evidence_path: Path,
    phase0_index_path: Path,
) -> IdentityEvidenceBundleV1:
    index, index_raw = _canonical_model_file(
        phase0_index_path,
        ArtifactIndexV1,
        "post-Core Phase 0 artifact index",
        exclude={"canonical_hash"},
    )
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    if index_sha256 != EXPECTED_PHASE0_INDEX_SHA256:
        raise ValueError("identity evidence requires the authoritative Phase 0 index")
    if len(index.artifacts) != 6:
        raise ValueError("Phase 0 index closure is incomplete")
    bundle, evidence_raw = _canonical_model_file(
        evidence_path, IdentityEvidenceBundleV1, "official identity evidence"
    )
    if hashlib.sha256(evidence_raw).hexdigest() != EXPECTED_IDENTITY_EVIDENCE_SHA256:
        raise ValueError("identity evidence differs from the authoritative official identity evidence")
    if bundle.phase0_index_sha256 != index_sha256:
        raise ValueError("identity evidence Phase 0 binding differs")
    return bundle


def build_identity_evidence_receipt_v1(
    bundle: IdentityEvidenceBundleV1,
    evidence_path: Path,
) -> IdentityEvidenceReceiptV1:
    raw = Path(evidence_path).resolve(strict=True).read_bytes()
    if canonical_bytes(bundle) != raw:
        raise ValueError("identity receipt source bytes differ from validated evidence")
    counts: dict[str, int] = {}
    for row in bundle.records:
        counts[row.state.value] = counts.get(row.state.value, 0) + 1
    return IdentityEvidenceReceiptV1(
        evidence_id=bundle.evidence_id,
        evidence_sha256=hashlib.sha256(raw).hexdigest(),
        phase0_index_sha256=bundle.phase0_index_sha256,
        state_counts=dict(sorted(counts.items())),
    )


__all__ = [
    "EXPECTED_IDENTITY_EVIDENCE_SHA256",
    "EXPECTED_IDENTITY_KEYS",
    "EXPECTED_PHASE0_INDEX_SHA256",
    "IdentityArtifactEvidenceV1",
    "IdentityEvidenceBundleV1",
    "IdentityEvidenceReceiptV1",
    "IdentityEvidenceRecordV1",
    "build_identity_evidence_receipt_v1",
    "load_identity_evidence_v1",
]
