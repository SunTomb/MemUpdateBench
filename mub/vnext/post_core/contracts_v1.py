from __future__ import annotations

from decimal import Decimal
from enum import Enum
import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import Field, StrictBool, StrictInt, StrictStr, computed_field, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"
POST_CORE_ARTIFACT_ORDER = (
    "post_core_release_manifest.json",
    "model_registry.json",
    "provenance.jsonl",
    "qualification_report.json",
    "capability_probe_report.json",
    "execution_plan.json",
)


class CandidateIdentityState(str, Enum):
    PENDING_OFFICIAL_IDENTITY = "PENDING_OFFICIAL_IDENTITY"
    PENDING_LOCAL_SNAPSHOT = "PENDING_LOCAL_SNAPSHOT"
    PENDING_PROVIDER_QUALIFICATION = "PENDING_PROVIDER_QUALIFICATION"
    READY_FOR_OFFLINE_PREFLIGHT = "READY_FOR_OFFLINE_PREFLIGHT"
    READY_FOR_PROVIDER_PREFLIGHT = "READY_FOR_PROVIDER_PREFLIGHT"
    QUALIFIED = "QUALIFIED"
    BLOCKED = "BLOCKED"


class ModelIdentityV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.model-identity.v1"] = "memupdatebench.post-core.model-identity.v1"
    official_model_id: StrictStr
    revision: StrictStr
    license_id: StrictStr
    architecture: StrictStr
    weights_uri: StrictStr | None = None
    tokenizer_identity: StrictStr | None = None
    endpoint: StrictStr | None = None
    resolved_upstream_identity: StrictStr | None = None


class ModelCandidateV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.model-candidate.v1"] = "memupdatebench.post-core.model-candidate.v1"
    registry_key: StrictStr = Field(pattern=r"^[a-z0-9_]+$")
    role: Literal[
        "modern_open_anchor", "large_open_anchor", "quantization_control",
        "closed_full", "closed_expensive_hard", "closed_proposed",
    ]
    state: CandidateIdentityState
    identity: ModelIdentityV1 | None
    scopes: tuple[Literal["full", "hard_subset", "k16_subset", "none"], ...]
    credential_env_var: StrictStr | None = None
    blocked_reasons: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _identity_state(self) -> "ModelCandidateV1":
        pending = self.state in {
            CandidateIdentityState.PENDING_OFFICIAL_IDENTITY,
            CandidateIdentityState.PENDING_LOCAL_SNAPSHOT,
            CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION,
        }
        if pending and self.identity is not None:
            raise ValueError("pending identity fields must remain null")
        if self.state in {
            CandidateIdentityState.READY_FOR_OFFLINE_PREFLIGHT,
            CandidateIdentityState.READY_FOR_PROVIDER_PREFLIGHT,
            CandidateIdentityState.QUALIFIED,
        } and self.identity is None:
            raise ValueError("ready/qualified candidate requires authenticated identity")
        if self.state is CandidateIdentityState.BLOCKED and not self.blocked_reasons:
            raise ValueError("blocked candidate requires reasons")
        return self


class QuantizationSpecV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.quantization.v1"] = "memupdatebench.post-core.quantization.v1"
    mode: Literal["none", "bf16", "int4"]
    bits: StrictInt | None = Field(default=None, ge=1, le=16)
    group_size: StrictInt | None = Field(default=None, ge=1)
    compute_dtype: Literal["bf16", "fp16", "fp32"]
    checkpoint_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    method: StrictStr | None = None
    calibration_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _static_quantization(self) -> "QuantizationSpecV1":
        if self.mode == "int4" and (
            self.bits != 4 or self.group_size is None or self.checkpoint_sha256 is None
        ):
            raise ValueError("int4 requires a static checkpoint, four bits, and group size")
        if self.mode != "int4" and any(
            value is not None for value in (self.bits, self.group_size, self.checkpoint_sha256, self.calibration_sha256)
        ):
            raise ValueError("non-int4 modes cannot carry quantization checkpoint fields")
        return self


class SpeculativeDecodingSpecV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.speculative.v1"] = "memupdatebench.post-core.speculative.v1"
    mode: Literal["off", "on"] = "off"
    draft_model_key: StrictStr | None = None
    parity_receipt_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _parity_required(self) -> "SpeculativeDecodingSpecV1":
        if self.mode == "on" and (not self.draft_model_key or not self.parity_receipt_sha256):
            raise ValueError("speculative decoding requires a parity receipt and draft model")
        if self.mode == "off" and (self.draft_model_key or self.parity_receipt_sha256):
            raise ValueError("speculative-off cannot carry draft/parity fields")
        return self


class MatrixCellV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.matrix-cell.v1"] = "memupdatebench.post-core.matrix-cell.v1"
    model_key: StrictStr
    context_order: Literal["chronological", "reverse_chronological"]
    context_annotation: Literal["none", "latest_outdated_label"]
    retrieval_k: Literal[4, 8, 16]
    precision: Literal["bf16", "int4", "api_managed"]
    quantization: Literal["none", "int4"]
    speculative_mode: Literal["off", "on"]
    repetition: StrictInt = Field(ge=1)
    seed: StrictInt = Field(ge=0)
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)

    @computed_field(return_type=str)
    @property
    def call_id(self) -> str:
        return canonical_hash(
            self,
            exclude={"call_id"},
        )


class CallBudgetV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.call-budget.v1"] = "memupdatebench.post-core.call-budget.v1"
    max_calls: StrictInt = Field(ge=0)
    max_prompt_tokens: StrictInt = Field(ge=0)
    max_output_tokens: StrictInt = Field(ge=0)
    estimated_cost: Decimal = Field(ge=Decimal("0"))
    hard_max_cost: Decimal = Field(ge=Decimal("0"))
    price_version: StrictStr
    max_retries: StrictInt = Field(ge=0)
    timeout_seconds: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _cost_bound(self) -> "CallBudgetV1":
        if not self.estimated_cost.is_finite() or not self.hard_max_cost.is_finite():
            raise ValueError("costs must be finite")
        if self.estimated_cost > self.hard_max_cost:
            raise ValueError("estimated cost exceeds hard maximum")
        return self


class CallPlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.call-plan.v1"] = "memupdatebench.post-core.call-plan.v1"
    call_id: StrictStr = Field(pattern=SHA256_PATTERN)
    model_key: StrictStr
    state: CandidateIdentityState
    executable: StrictBool
    matrix_cell: MatrixCellV1
    budget: CallBudgetV1

    @model_validator(mode="after")
    def _execution_boundary(self) -> "CallPlanV1":
        if self.call_id != self.matrix_cell.call_id or self.model_key != self.matrix_cell.model_key:
            raise ValueError("call plan identity mismatch")
        if self.executable and self.state is not CandidateIdentityState.QUALIFIED:
            raise ValueError("pending or blocked candidates cannot be executable")
        return self


class ReleaseManifestV1(ImmutableContractModel):
    release_id: StrictStr
    schema_version: Literal["memupdatebench.post-core.release.v1"]
    artifact_order: tuple[StrictStr, ...]
    source_hashes: Mapping[StrictStr, StrictStr]

    @model_validator(mode="after")
    def _artifact_order(self) -> "ReleaseManifestV1":
        if self.artifact_order != POST_CORE_ARTIFACT_ORDER:
            raise ValueError("post-Core manifest artifact order mismatch")
        if not self.source_hashes or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.source_hashes.values()
        ):
            raise ValueError("source hashes must be lowercase SHA-256")
        return self


class ArtifactBindingV1(ImmutableContractModel):
    path: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class ArtifactIndexV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.artifact-index.v1"] = "memupdatebench.post-core.artifact-index.v1"
    release_id: StrictStr
    artifacts: tuple[ArtifactBindingV1, ...]

    @model_validator(mode="after")
    def _exact_index(self) -> "ArtifactIndexV1":
        paths = tuple(item.path for item in self.artifacts)
        if paths != POST_CORE_ARTIFACT_ORDER:
            raise ValueError("post-Core index must bind the exact preceding six artifacts")
        return self

    @computed_field(return_type=str)
    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self, exclude={"canonical_hash"})


def canonical_bytes(value: Any, *, exclude: set[str] | None = None) -> bytes:
    payload = value.model_dump(
        mode="json",
        exclude=exclude or set(),
        exclude_computed_fields=False,
    ) if hasattr(value, "model_dump") else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: Any, *, exclude: set[str] | None = None) -> str:
    return hashlib.sha256(canonical_bytes(value, exclude=exclude)).hexdigest()


__all__ = [
    "ArtifactIndexV1", "CallBudgetV1", "CallPlanV1", "CandidateIdentityState",
    "MatrixCellV1", "ModelCandidateV1", "ModelIdentityV1", "POST_CORE_ARTIFACT_ORDER",
    "QuantizationSpecV1", "ReleaseManifestV1", "SpeculativeDecodingSpecV1",
    "canonical_bytes", "canonical_hash",
]
