from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, computed_field, field_validator, model_validator

from mub.vnext.contracts.common import (
    FrozenNonnegativeIntMap,
    FrozenStringMap,
    ImmutableContractModel,
    freeze_mapping,
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
    schema_version: Literal["memupdatebench.post-core.qualification-source.v1"] = (
        "memupdatebench.post-core.qualification-source.v1"
    )
    source_id: StrictStr
    evidence_class: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    required: StrictBool
    byte_count: StrictInt | None = Field(default=None, ge=0)


class SourceBindingBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-sources.v1"] = (
        "memupdatebench.post-core.qualification-sources.v1"
    )
    release_id: StrictStr
    sources: tuple[SourceBindingV1, ...]


class CapabilityBudgetV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-budget.v1"] = (
        "memupdatebench.post-core.qualification-budget.v1"
    )
    max_calls: Literal[1] = 1
    max_prompt_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)
    estimated_cost: Decimal = Field(ge=Decimal("0"))
    hard_max_cost: Decimal = Field(ge=Decimal("0"))
    price_version: StrictStr
    max_retries: Literal[0] = 0
    timeout_seconds: StrictInt = Field(gt=0)

    @field_validator("max_calls", "max_retries", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @model_validator(mode="after")
    def _cost_bound(self) -> "CapabilityBudgetV1":
        if not self.estimated_cost.is_finite() or not self.hard_max_cost.is_finite():
            raise ValueError("capability costs must be finite")
        if self.estimated_cost > self.hard_max_cost:
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
    schema_version: Literal["memupdatebench.post-core.qualification-index.v1"] = (
        "memupdatebench.post-core.qualification-index.v1"
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
    response_format: Literal["ANTHROPIC_MESSAGE_JSON", "SSE"]
    response_model: StrictStr
    exact_ok: Literal[True] = True
    stop_reason: Literal["end_turn"] | None = None
    usage_present: StrictBool | None = None

    @field_validator("provider_call_count", "retry_count", "http_status", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @field_validator("exact_ok", mode="before")
    @classmethod
    def _strict_bool_literals(cls, value: object) -> object:
        return _require_exact_bool_literal(value)


class ProviderCapabilityAttestationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-attestation.v1"] = (
        "memupdatebench.post-core.provider-attestation.v1"
    )
    registry_key: StrictStr
    evidence_class: Literal["connectivity_interface_attestation"] = (
        "connectivity_interface_attestation"
    )
    request_name: StrictStr
    canonical_model_identity: StrictStr | None = None
    reasoning_tier: StrictStr | None = None
    identity_caveat: StrictStr | None = None
    observations: tuple[ProviderObservationV1, ...]
    provider_call_count: StrictInt = Field(ge=0)
    retry_count: Literal[0] = 0
    benchmark_generation_count: Literal[0] = 0
    raw_response_persisted: Literal[False] = False
    source_binding_ids: tuple[StrictStr, ...]

    @field_validator("retry_count", "benchmark_generation_count", mode="before")
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)

    @field_validator("raw_response_persisted", mode="before")
    @classmethod
    def _strict_bool_literals(cls, value: object) -> object:
        return _require_exact_bool_literal(value)

    @model_validator(mode="after")
    def _observation_count(self) -> "ProviderCapabilityAttestationV1":
        if self.provider_call_count != sum(item.provider_call_count for item in self.observations):
            raise ValueError("provider call count does not match observations")
        return self


class ProviderSetupEventV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-setup-event.v1"] = (
        "memupdatebench.post-core.provider-setup-event.v1"
    )
    event_id: StrictStr
    stage: Literal["PRE_PROVIDER_SETUP"] = "PRE_PROVIDER_SETUP"
    status: Literal["FAILED"] = "FAILED"
    provider_call_count: Literal[0] = 0
    reason_class: StrictStr
    source_binding_ids: tuple[StrictStr, ...]

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
    python_version: StrictStr | None = None
    torch_version: StrictStr | None = None
    transformers_version: StrictStr | None = None
    accelerate_version: StrictStr | None = None
    cuda_version: StrictStr | None = None
    driver_version: StrictStr | None = None
    device_name: StrictStr
    context_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)
    build_options_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)


class OpenRuntimeReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.open-runtime-receipt.v1"] = (
        "memupdatebench.post-core.open-runtime-receipt.v1"
    )
    registry_key: StrictStr
    revision: StrictStr
    snapshot_tree_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    runtime: RuntimeManifestV1
    speculative_decoding: Literal["off"] = "off"
    load_status: GateStatus
    generation_status: GateStatus
    determinism_status: GateStatus
    unload_status: GateStatus
    prompt_fixture_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    parser_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    chat_template_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    output_projection_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    generated_token_count: StrictInt | None = Field(default=None, ge=0)
    peak_memory_bytes: StrictInt | None = Field(default=None, ge=0)
    blocked_reasons: tuple[StrictStr, ...] = ()
    source_binding_ids: tuple[StrictStr, ...]


class CapabilityFixtureV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-fixture.v1"] = (
        "memupdatebench.post-core.capability-fixture.v1"
    )
    fixture_id: StrictStr
    category: Literal["EXACT_OUTPUT", "CHAT_TEMPLATE_PARSER"]
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    parser_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    max_prompt_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)


class CapabilitySmokePlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-smoke-plan.v1"] = (
        "memupdatebench.post-core.capability-smoke-plan.v1"
    )
    release_id: StrictStr
    registry_keys: tuple[StrictStr, ...]
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
        if not self.registry_keys or len(self.registry_keys) != len(set(self.registry_keys)):
            raise ValueError("qualification smoke plan registry keys must be nonempty and unique")
        if len(self.attempts) != 16 * len(self.registry_keys):
            raise ValueError("qualification smoke plan has an unexpected attempt count")
        if {item.registry_key for item in self.attempts} != set(self.registry_keys):
            raise ValueError("qualification smoke plan attempts must reference exactly listed keys")
        coordinates = tuple(
            (item.registry_key, item.fixture_id, item.phase, item.repetition)
            for item in self.attempts
        )
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("qualification smoke plan attempt coordinates must be unique")
        call_ids = tuple(item.call_id for item in self.attempts)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("qualification smoke plan call IDs must be unique")
        for key in self.registry_keys:
            base_count = sum(
                item.registry_key == key and item.phase is AttemptPhase.BASE
                for item in self.attempts
            )
            escalation_count = sum(
                item.registry_key == key and item.phase is AttemptPhase.ESCALATION
                for item in self.attempts
            )
            if (
                base_count != self.base_attempts_per_role
                or escalation_count != self.escalation_attempts_per_role
            ):
                raise ValueError("qualification smoke plan must have eight attempts per phase and role")
        if any(item.authorized or item.executable or item.budget.max_retries != 0 for item in self.attempts):
            raise ValueError("qualification smoke plan cannot authorize execution or retries")
        return self


class CapabilityAnomalyReceiptV1(ImmutableContractModel):
    schema_version: Literal["qualification-capability-anomaly-receipt.v1"] = (
        "qualification-capability-anomaly-receipt.v1"
    )
    release_id: StrictStr
    plan_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    base_receipts_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    base_call_ids: tuple[StrictStr, ...]
    anomalous_call_ids: tuple[StrictStr, ...]
    anomaly_types: tuple[Literal["PARSER", "FORMAT", "STABILITY"], ...]
    summary_class: StrictStr

    @field_validator("summary_class")
    @classmethod
    def _nonblank_summary_class(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("summary class must be nonblank")
        return value

    @model_validator(mode="after")
    def _anomaly_shape(self) -> "CapabilityAnomalyReceiptV1":
        for name, values in (
            ("base call IDs", self.base_call_ids),
            ("anomalous call IDs", self.anomalous_call_ids),
            ("anomaly types", self.anomaly_types),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(
            len(call_id) != 64 or any(char not in "0123456789abcdef" for char in call_id)
            for call_id in (*self.base_call_ids, *self.anomalous_call_ids)
        ):
            raise ValueError("anomaly receipt call IDs must be lowercase SHA-256")
        if not set(self.anomalous_call_ids).issubset(self.base_call_ids):
            raise ValueError("anomalous call IDs must be a subset of base call IDs")
        return self


class ExecutionAuthorizationV1(ImmutableContractModel):
    schema_version: Literal["qualification-execution-authorization.v1"] = (
        "qualification-execution-authorization.v1"
    )
    release_id: StrictStr
    plan_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    scope: Literal[DecisionScope.CAPABILITY_SMOKE]
    authorized_call_ids: tuple[StrictStr, ...]
    max_calls: StrictInt = Field(gt=0)
    issued_at: StrictStr
    issuer: StrictStr
    authorization_attestation_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    adapter_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    escalation_anomaly_receipt_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("issued_at")
    @classmethod
    def _utc_timestamp(cls, value: StrictStr) -> StrictStr:
        from datetime import datetime

        if not value.strip() or not value.endswith("Z"):
            raise ValueError("issued_at must be a nonblank UTC timestamp")
        try:
            datetime.fromisoformat(f"{value[:-1]}+00:00")
        except ValueError as exc:
            raise ValueError("issued_at must be an ISO8601-ish UTC timestamp") from exc
        return value

    @field_validator("issuer")
    @classmethod
    def _nonblank_issuer(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("issuer must be nonblank")
        return value

    @model_validator(mode="after")
    def _authorized_call_shape(self) -> "ExecutionAuthorizationV1":
        if not self.authorized_call_ids:
            raise ValueError("authorized call IDs must be nonempty")
        if len(self.authorized_call_ids) != len(set(self.authorized_call_ids)):
            raise ValueError("authorized call IDs must be unique")
        if any(
            len(call_id) != 64 or any(char not in "0123456789abcdef" for char in call_id)
            for call_id in self.authorized_call_ids
        ):
            raise ValueError("authorized call IDs must be lowercase SHA-256")
        return self


class CapabilityAdapterResultV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-adapter-result.v1"] = (
        "memupdatebench.post-core.capability-adapter-result.v1"
    )
    call_id: StrictStr = Field(pattern=SHA256_PATTERN)
    registry_key: StrictStr
    response_projection: StrictStr | None = None
    response_model: StrictStr | None = None
    response_format: Literal["ANTHROPIC_MESSAGE_JSON", "SSE", "LOCAL_TEXT"] | None = None
    stop_reason: StrictStr | None = None
    usage_present: StrictBool | None = None
    latency_ms: StrictInt | None = Field(default=None, ge=0)
    error_class: Literal[
        "ADAPTER_TIMEOUT",
        "ADAPTER_FAILURE",
        "FORMAT_ERROR",
        "STABILITY_MISMATCH",
    ] | None = None


class CapabilityAttemptReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-attempt-receipt.v1"] = (
        "memupdatebench.post-core.capability-attempt-receipt.v1"
    )
    call_id: StrictStr = Field(pattern=SHA256_PATTERN)
    registry_key: StrictStr
    status: GateStatus
    retry_count: Literal[0] = 0
    response_model: StrictStr | None = None
    response_format: Literal["ANTHROPIC_MESSAGE_JSON", "SSE", "LOCAL_TEXT"] | None = None
    stop_reason: StrictStr | None = None
    usage_present: StrictBool | None = None
    latency_ms: StrictInt | None = Field(default=None, ge=0)
    redacted_response_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    error_class: StrictStr | None = None

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
    reasons: tuple[StrictStr, ...]
    evidence_binding_ids: tuple[StrictStr, ...]
    scientific_status: Literal["NOT_RUN"] = "NOT_RUN"


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
    base_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_order: tuple[StrictStr, ...]
    source_hashes: FrozenStringMap

    @field_validator("source_hashes")
    @classmethod
    def _freeze_source_hashes(cls, value: FrozenStringMap) -> FrozenStringMap:
        return freeze_mapping(value)

    @model_validator(mode="after")
    def _artifact_order(self) -> "QualificationReleaseManifestV1":
        if self.artifact_order != QUALIFICATION_ARTIFACT_ORDER:
            raise ValueError("qualification manifest artifact order mismatch")
        if not self.source_hashes or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.source_hashes.values()
        ):
            raise ValueError("qualification manifest source hashes must be lowercase SHA-256")
        return self


class QualificationValidationReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-validation.v1"] = (
        "memupdatebench.post-core.qualification-validation.v1"
    )
    release_id: StrictStr
    status: Literal["SUCCESS_WITH_BLOCKERS", "SUCCESS"]
    source_count: StrictInt = Field(gt=0)
    decision_counts: FrozenNonnegativeIntMap
    provider_calls_during_publication: Literal[0] = 0
    model_loads_during_publication: Literal[0] = 0
    network_calls_during_publication: Literal[0] = 0
    credential_reads_during_publication: Literal[0] = 0
    benchmark_generations: Literal[0] = 0

    @field_validator("decision_counts")
    @classmethod
    def _freeze_decision_counts(cls, value: FrozenNonnegativeIntMap) -> FrozenNonnegativeIntMap:
        return freeze_mapping(value)

    @field_validator(
        "provider_calls_during_publication",
        "model_loads_during_publication",
        "network_calls_during_publication",
        "credential_reads_during_publication",
        "benchmark_generations",
        mode="before",
    )
    @classmethod
    def _strict_numeric_literals(cls, value: object) -> object:
        return _require_exact_int_literal(value)


__all__ = [
    "ArtifactBindingV1",
    "AttemptPhase",
    "CapabilityAnomalyReceiptV1",
    "CapabilityAdapterResultV1",
    "CapabilityAttemptPlanV1",
    "CapabilityAttemptReceiptV1",
    "CapabilityBudgetV1",
    "CapabilityFixtureV1",
    "CapabilitySmokePlanV1",
    "DecisionScope",
    "ExecutionAuthorizationV1",
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
