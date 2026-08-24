from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Mapping

from pydantic import (
    Field,
    PlainSerializer,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    field_validator,
    model_validator,
)

from mub.vnext.contracts.common import (
    FrozenNonnegativeIntMap,
    FrozenStringMap,
    ImmutableContractModel,
    freeze_mapping,
    thaw_json,
)
from mub.vnext.post_core.contracts_v1 import SHA256_PATTERN, canonical_hash


QUALIFICATION_ARTIFACT_ORDER = (
    "qualification_release_manifest.json",
    "source_bindings.json",
    "provider_capability_attestations.jsonl",
    "open_runtime_receipts.jsonl",
    "capability_smoke_plan.json",
    "qualification_decisions.json",
    "qualification_validation_receipt.json",
)
QUALIFICATION_INDEX_PATH = "qualification_artifact_index.json"
QUALIFICATION_ARTIFACTS = (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)

FrozenDecimalMap = Annotated[
    Mapping[str, Decimal],
    PlainSerializer(thaw_json, return_type=dict[str, Decimal], when_used="always"),
]


def _require_exact_int_literal(value: object) -> object:
    if type(value) is not int:
        raise ValueError("numeric literal fields require an exact int input")
    return value


def _require_exact_bool_literal(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("boolean literal fields require an exact bool input")
    return value


class QualificationStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


class DecisionScope(str, Enum):
    STORAGE_INPUT = "STORAGE_INPUT"
    SHORT_GENERATION_GATE = "SHORT_GENERATION_GATE"
    CAPABILITY_SMOKE = "CAPABILITY_SMOKE"
    BENCHMARK_ADMISSION = "BENCHMARK_ADMISSION"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


class AttemptPhase(str, Enum):
    BASE = "BASE"
    ESCALATION = "ESCALATION"


class SourceBindingV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-source-binding.v1"] = (
        "memupdatebench.post-core.qualification-source-binding.v1"
    )
    source_id: StrictStr
    evidence_class: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    required: StrictBool
    byte_count: StrictInt | None = Field(default=None, ge=0)


class SourceBindingBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-source-bindings.v1"] = (
        "memupdatebench.post-core.qualification-source-bindings.v1"
    )
    release_id: StrictStr
    sources: tuple[SourceBindingV1, ...]


class CapabilityBudgetV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-budget.v1"] = (
        "memupdatebench.post-core.capability-budget.v1"
    )
    max_calls: Literal[1] = 1
    input_token_cap: StrictInt = Field(gt=0)
    output_token_cap: StrictInt = Field(gt=0)
    estimated_cost_usd: Decimal = Field(ge=Decimal("0"))
    hard_max_cost_usd: Decimal = Field(ge=Decimal("0"))
    price_version: StrictStr
    max_retries: Literal[0] = 0
    timeout_seconds: StrictInt = Field(gt=0)

    @field_validator("max_calls", "max_retries", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @model_validator(mode="after")
    def _cost_bound(self) -> "CapabilityBudgetV1":
        if not self.estimated_cost_usd.is_finite() or not self.hard_max_cost_usd.is_finite():
            raise ValueError("capability costs must be finite")
        if self.estimated_cost_usd > self.hard_max_cost_usd:
            raise ValueError("estimated capability cost exceeds hard maximum")
        return self


class CapabilityAttemptPlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-attempt-plan.v1"] = (
        "memupdatebench.post-core.capability-attempt-plan.v1"
    )
    release_id: StrictStr
    registry_key: StrictStr
    fixture_id: StrictStr
    phase: AttemptPhase
    repetition: StrictInt = Field(ge=1, le=2)
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    parser_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    runtime_or_endpoint_class: StrictStr
    budget: CapabilityBudgetV1
    authorized: StrictBool = False
    executable: StrictBool = False

    @model_validator(mode="after")
    def _authorization_boundary(self) -> "CapabilityAttemptPlanV1":
        if self.executable and not self.authorized:
            raise ValueError("executable capability attempt requires authorization")
        return self

    @computed_field(return_type=str)
    @property
    def call_id(self) -> str:
        return canonical_hash(self, exclude={"call_id"})


class ArtifactBindingV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-artifact-binding.v1"] = (
        "memupdatebench.post-core.qualification-artifact-binding.v1"
    )
    path: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)


class QualificationArtifactIndexV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-artifact-index.v1"] = (
        "memupdatebench.post-core.qualification-artifact-index.v1"
    )
    release_id: StrictStr
    artifacts: tuple[ArtifactBindingV1, ...]

    @model_validator(mode="after")
    def _exact_index(self) -> "QualificationArtifactIndexV1":
        if tuple(item.path for item in self.artifacts) != QUALIFICATION_ARTIFACT_ORDER:
            raise ValueError("qualification index must bind the exact preceding artifacts")
        return self

    @computed_field(return_type=str)
    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self, exclude={"canonical_hash"})


class ProviderObservationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-observation.v1"] = (
        "memupdatebench.post-core.provider-observation.v1"
    )
    location: Literal["LOCAL", "TANG2"]
    observation_id: StrictStr
    provider_call_count: Literal[1] = 1
    retry_count: Literal[0] = 0
    http_status: Literal[200] = 200
    response_format: Literal["JSON", "SSE"]
    response_model: StrictStr
    exact_ok: Literal[True] = True
    end_turn: StrictBool | None = None
    usage: FrozenNonnegativeIntMap | None = None

    @field_validator("provider_call_count", "retry_count", "http_status", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @field_validator("exact_ok", mode="before")
    @classmethod
    def _strict_bool_literals(cls, value: object) -> object:
        return _require_exact_bool_literal(value)

    @field_validator("usage")
    @classmethod
    def _freeze_usage(cls, value: FrozenNonnegativeIntMap | None) -> FrozenNonnegativeIntMap | None:
        return None if value is None else freeze_mapping(value)


class ProviderCapabilityAttestationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-capability-attestation.v1"] = (
        "memupdatebench.post-core.provider-capability-attestation.v1"
    )
    evidence_class: Literal["connectivity_interface_attestation"] = (
        "connectivity_interface_attestation"
    )
    registry_key: StrictStr
    requested_model: StrictStr
    canonical_model: StrictStr
    capability_tier: StrictStr
    caveat: StrictStr | None = None
    observations: tuple[ProviderObservationV1, ...]
    observation_count: StrictInt = Field(gt=0)
    provider_call_count: StrictInt = Field(gt=0)
    raw_response_persisted: Literal[False] = False
    source_bindings: tuple[SourceBindingV1, ...]

    @model_validator(mode="after")
    def _observation_counts(self) -> "ProviderCapabilityAttestationV1":
        if self.observation_count != len(self.observations):
            raise ValueError("provider observation count does not match observations")
        if self.provider_call_count != sum(item.provider_call_count for item in self.observations):
            raise ValueError("provider call count does not match observations")
        return self

    @field_validator("raw_response_persisted", mode="before")
    @classmethod
    def _strict_bool_literals(cls, value: object) -> object:
        return _require_exact_bool_literal(value)


class ProviderSetupEventV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-setup-event.v1"] = (
        "memupdatebench.post-core.provider-setup-event.v1"
    )
    registry_key: StrictStr
    event_type: Literal["PRE_PROVIDER_SETUP"] = "PRE_PROVIDER_SETUP"
    status: Literal["FAILED"] = "FAILED"
    provider_call_count: Literal[0] = 0
    detail: StrictStr | None = None
    source_binding_ids: tuple[StrictStr, ...] = ()

    @field_validator("provider_call_count", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)


class RuntimeManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.runtime-manifest.v1"] = (
        "memupdatebench.post-core.runtime-manifest.v1"
    )
    engine: Literal["transformers", "llama.cpp"]
    engine_version: StrictStr
    engine_commit: StrictStr | None = None
    binary_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    package_versions: FrozenStringMap | None = None
    runtime_version: StrictStr | None = None
    device: StrictStr | None = None
    context_window: StrictInt | None = Field(default=None, gt=0)
    output_token_cap: StrictInt | None = Field(default=None, gt=0)
    build: StrictStr | None = None

    @field_validator("package_versions")
    @classmethod
    def _freeze_package_versions(cls, value: FrozenStringMap | None) -> FrozenStringMap | None:
        return None if value is None else freeze_mapping(value)


class OpenRuntimeReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.open-runtime-receipt.v1"] = (
        "memupdatebench.post-core.open-runtime-receipt.v1"
    )
    registry_key: StrictStr
    revision: StrictStr
    tree_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    runtime: RuntimeManifestV1
    speculative_decoding: Literal["off"] = "off"
    storage_input_status: GateStatus
    short_generation_status: GateStatus
    capability_smoke_status: GateStatus
    benchmark_admission_status: GateStatus
    tokenizer_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    measurements: FrozenDecimalMap | None = None
    blockers: tuple[StrictStr, ...] = ()
    source_binding_ids: tuple[StrictStr, ...]

    @field_validator("measurements")
    @classmethod
    def _freeze_measurements(cls, value: FrozenDecimalMap | None) -> FrozenDecimalMap | None:
        return None if value is None else freeze_mapping(value)


class CapabilityFixtureV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-fixture.v1"] = (
        "memupdatebench.post-core.capability-fixture.v1"
    )
    fixture_id: StrictStr
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    parser_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    expected_response_format: Literal["JSON", "SSE"]


class CapabilitySmokePlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-smoke-plan.v1"] = (
        "memupdatebench.post-core.capability-smoke-plan.v1"
    )
    release_id: StrictStr
    base_attempts_per_role: Literal[8] = 8
    escalation_attempts_per_role: Literal[8] = 8
    max_retries: Literal[0] = 0
    authorized: Literal[False] = False
    attempts: tuple[CapabilityAttemptPlanV1, ...]

    @field_validator(
        "base_attempts_per_role", "escalation_attempts_per_role", "max_retries", mode="before"
    )
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @field_validator("authorized", mode="before")
    @classmethod
    def _strict_bool_literals(cls, value: object) -> object:
        return _require_exact_bool_literal(value)

    @model_validator(mode="after")
    def _attempt_shape(self) -> "CapabilitySmokePlanV1":
        base = sum(item.phase is AttemptPhase.BASE for item in self.attempts)
        escalation = sum(item.phase is AttemptPhase.ESCALATION for item in self.attempts)
        if base != self.base_attempts_per_role or escalation != self.escalation_attempts_per_role:
            raise ValueError("capability smoke plan must contain eight base and eight escalation attempts")
        if any(item.authorized or item.executable or item.budget.max_retries != 0 for item in self.attempts):
            raise ValueError("qualification smoke plan cannot authorize execution or retries")
        return self


class CapabilityAttemptReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-attempt-receipt.v1"] = (
        "memupdatebench.post-core.capability-attempt-receipt.v1"
    )
    call_id: StrictStr = Field(pattern=SHA256_PATTERN)
    registry_key: StrictStr
    fixture_id: StrictStr
    phase: AttemptPhase
    gate_status: GateStatus
    retry_count: Literal[0] = 0
    provider_call_count: StrictInt | None = Field(default=None, ge=0)
    response_model: StrictStr | None = None
    exact_ok: StrictBool | None = None
    blocker: StrictStr | None = None
    source_binding_ids: tuple[StrictStr, ...] = ()

    @field_validator("retry_count", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)


class QualificationDecisionV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-decision.v1"] = (
        "memupdatebench.post-core.qualification-decision.v1"
    )
    registry_key: StrictStr
    scope: DecisionScope
    status: QualificationStatus
    gate_status: GateStatus
    reason: StrictStr
    source_binding_ids: tuple[StrictStr, ...] = ()


class QualificationDecisionBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-decisions.v1"] = (
        "memupdatebench.post-core.qualification-decisions.v1"
    )
    release_id: StrictStr
    decisions: tuple[QualificationDecisionV1, ...]


class QualificationReleaseManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-release.v1"] = (
        "memupdatebench.post-core.qualification-release.v1"
    )
    release_id: StrictStr
    artifact_order: tuple[StrictStr, ...]
    required_source_sha256: FrozenStringMap

    @field_validator("required_source_sha256")
    @classmethod
    def _freeze_required_source_sha256(cls, value: FrozenStringMap) -> FrozenStringMap:
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _artifact_order(self) -> "QualificationReleaseManifestV1":
        if self.artifact_order != QUALIFICATION_ARTIFACT_ORDER:
            raise ValueError("qualification manifest artifact order mismatch")
        if not self.required_source_sha256 or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.required_source_sha256.values()
        ):
            raise ValueError("qualification manifest source hashes must be lowercase SHA-256")
        return self


class QualificationValidationReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-validation-receipt.v1"] = (
        "memupdatebench.post-core.qualification-validation-receipt.v1"
    )
    release_id: StrictStr
    status: Literal["SUCCESS_WITH_BLOCKERS", "SUCCESS"]
    source_count: StrictInt = Field(ge=0)
    decision_count: StrictInt = Field(ge=0)
    decision_counts: FrozenNonnegativeIntMap | None = None
    provider_calls: Literal[0] = 0
    model_loads: Literal[0] = 0
    network_calls: Literal[0] = 0
    executable_calls: Literal[0] = 0
    published_provider_attestations: Literal[0] = 0
    published_open_runtime_receipts: Literal[0] = 0
    published_capability_attempt_receipts: Literal[0] = 0

    @field_validator("decision_counts")
    @classmethod
    def _freeze_decision_counts(
        cls, value: FrozenNonnegativeIntMap | None
    ) -> FrozenNonnegativeIntMap | None:
        return None if value is None else freeze_mapping(value)

    @field_validator(
        "provider_calls",
        "model_loads",
        "network_calls",
        "executable_calls",
        "published_provider_attestations",
        "published_open_runtime_receipts",
        "published_capability_attempt_receipts",
        mode="before",
    )
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)


__all__ = [
    "ArtifactBindingV1",
    "AttemptPhase",
    "CapabilityAttemptPlanV1",
    "CapabilityAttemptReceiptV1",
    "CapabilityBudgetV1",
    "CapabilityFixtureV1",
    "CapabilitySmokePlanV1",
    "DecisionScope",
    "FrozenDecimalMap",
    "GateStatus",
    "OpenRuntimeReceiptV1",
    "ProviderCapabilityAttestationV1",
    "ProviderObservationV1",
    "ProviderSetupEventV1",
    "QUALIFICATION_ARTIFACT_ORDER",
    "QUALIFICATION_ARTIFACTS",
    "QUALIFICATION_INDEX_PATH",
    "QualificationArtifactIndexV1",
    "QualificationDecisionBundleV1",
    "QualificationDecisionV1",
    "QualificationReleaseManifestV1",
    "QualificationStatus",
    "QualificationValidationReceiptV1",
    "RuntimeManifestV1",
    "SourceBindingBundleV1",
    "SourceBindingV1",
]
