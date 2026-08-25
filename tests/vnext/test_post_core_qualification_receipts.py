from decimal import Decimal
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import canonical_bytes, canonical_hash
from mub.vnext.post_core.qualification_receipts_v1 import (
    ArtifactBindingV1,
    AttemptPhase,
    CapabilityAdapterResultV1,
    CapabilityAttemptPlanV1,
    CapabilityAttemptReceiptV1,
    CapabilityBudgetV1,
    CapabilityFixtureV1,
    CapabilitySmokePlanV1,
    DecisionScope,
    GateStatus,
    ProviderCapabilityAttestationV1,
    ProviderObservationV1,
    ProviderSetupEventV1,
    QualificationArtifactIndexV1,
    QualificationDecisionV1,
    QualificationReleaseManifestV1,
    QualificationStatus,
    QualificationValidationReceiptV1,
    RuntimeManifestV1,
    SourceBindingV1,
    SourceBindingBundleV1,
    QUALIFICATION_ARTIFACT_ORDER,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _binding() -> SourceBindingV1:
    return SourceBindingV1(
        source_id="source",
        evidence_class="fixture",
        sha256=HASH_A,
        required=True,
    )


def _budget(**overrides: object) -> CapabilityBudgetV1:
    payload: dict[str, object] = {
        "max_calls": 1,
        "max_prompt_tokens": 1,
        "max_output_tokens": 1,
        "estimated_cost": Decimal("0.01"),
        "hard_max_cost": Decimal("0.02"),
        "price_version": "v1",
        "max_retries": 0,
        "timeout_seconds": 1,
    }
    payload.update(overrides)
    return CapabilityBudgetV1(**payload)


def _attempt(
    registry_key: str = "role-a",
    fixture_id: str = "fixture",
    phase: AttemptPhase = AttemptPhase.BASE,
    repetition: int = 1,
    **overrides: object,
) -> CapabilityAttemptPlanV1:
    payload: dict[str, object] = {
        "release_id": "release",
        "registry_key": registry_key,
        "fixture_id": fixture_id,
        "phase": phase,
        "repetition": repetition,
        "prompt_sha256": HASH_B,
        "parser_sha256": HASH_C,
        "runtime_or_endpoint_class": "offline",
        "budget": _budget(),
    }
    payload.update(overrides)
    return CapabilityAttemptPlanV1(**payload)


def _attempts_for_role(registry_key: str) -> tuple[CapabilityAttemptPlanV1, ...]:
    return tuple(
        _attempt(
            registry_key=registry_key,
            fixture_id=f"{phase.value.lower()}-{index}",
            phase=phase,
            repetition=(index % 2) + 1,
        )
        for phase in (AttemptPhase.BASE, AttemptPhase.ESCALATION)
        for index in range(8)
    )


def test_source_binding_preserves_none_byte_count_and_models_are_frozen() -> None:
    binding = _binding()

    assert binding.byte_count is None
    assert binding.schema_version == "memupdatebench.post-core.qualification-source.v1"
    assert SourceBindingBundleV1.model_fields["schema_version"].default == (
        "memupdatebench.post-core.qualification-sources.v1"
    )
    with pytest.raises(ValidationError):
        binding.source_id = "changed"


def test_capability_budget_uses_planned_names_and_strict_single_call_bounds() -> None:
    budget = _budget()

    assert budget.max_calls == 1
    assert budget.max_retries == 0
    assert budget.estimated_cost.is_finite()
    assert budget.estimated_cost <= budget.hard_max_cost
    with pytest.raises(ValidationError):
        _budget(max_calls=True)
    with pytest.raises(ValidationError):
        _budget(max_retries=False)
    with pytest.raises(ValidationError):
        _budget(estimated_cost=Decimal("NaN"))


def test_capability_attempt_defaults_to_non_executable_and_hashes_call_id() -> None:
    attempt = _attempt()

    assert attempt.authorized is False
    assert attempt.executable is False
    assert attempt.call_id == canonical_hash(attempt, exclude={"call_id"})
    with pytest.raises(ValidationError):
        _attempt(executable=True)


def test_qualification_statuses_and_decision_scopes_are_distinct() -> None:
    assert {QualificationStatus.READY, QualificationStatus.BLOCKED, QualificationStatus.UNSUPPORTED} == set(QualificationStatus)
    assert {
        DecisionScope.STORAGE_INPUT,
        DecisionScope.SHORT_GENERATION_GATE,
        DecisionScope.CAPABILITY_SMOKE,
        DecisionScope.BENCHMARK_ADMISSION,
    } == set(DecisionScope)


def test_qualification_artifact_index_has_exact_order_no_self_hash_and_planned_schema() -> None:
    artifacts = [
        {"path": path, "sha256": f"{index:064x}"}
        for index, path in enumerate(QUALIFICATION_ARTIFACT_ORDER, start=1)
    ]
    index = QualificationArtifactIndexV1(release_id="release", artifacts=artifacts)

    assert index.schema_version == "memupdatebench.post-core.qualification-index.v1"
    assert tuple(binding.path for binding in index.artifacts) == QUALIFICATION_ARTIFACT_ORDER
    assert index.canonical_hash == canonical_hash(index, exclude={"canonical_hash"})
    assert index.artifacts[0].schema_version == (
        "memupdatebench.post-core.qualification-artifact-binding.v1"
    )
    with pytest.raises(ValidationError):
        QualificationArtifactIndexV1(release_id="release", artifacts=list(reversed(artifacts)))


def test_provider_observation_uses_planned_interface_and_strict_literals() -> None:
    observation = ProviderObservationV1(
        location="LOCAL",
        observation_id="observation",
        response_format="ANTHROPIC_MESSAGE_JSON",
        response_model="model",
        stop_reason="end_turn",
        usage_present=True,
    )

    assert observation.provider_call_count == 1
    assert observation.retry_count == 0
    assert observation.http_status == 200
    assert observation.exact_ok is True
    assert observation.model_dump(mode="json")["stop_reason"] == "end_turn"
    for field, invalid in (
        ("provider_call_count", True),
        ("retry_count", False),
        ("http_status", True),
        ("exact_ok", 1),
    ):
        with pytest.raises(ValidationError):
            ProviderObservationV1(
                location="LOCAL",
                observation_id="observation",
                response_format="SSE",
                response_model="model",
                **{field: invalid},
            )


def test_remaining_literal_fields_reject_cross_type_inputs() -> None:
    observation = ProviderObservationV1(
        location="LOCAL",
        observation_id="observation",
        response_format="SSE",
        response_model="model",
    )
    for field, invalid in (
        ("retry_count", True),
        ("benchmark_generation_count", False),
        ("raw_response_persisted", 0),
    ):
        with pytest.raises(ValidationError):
            ProviderCapabilityAttestationV1(
                registry_key="role",
                request_name="request",
                observations=(observation,),
                provider_call_count=1,
                source_binding_ids=("source",),
                **{field: invalid},
            )
    with pytest.raises(ValidationError):
        ProviderSetupEventV1(
            event_id="setup",
            reason_class="offline",
            source_binding_ids=("source",),
            provider_call_count=False,
        )
    attempts = _attempts_for_role("role")
    for field, invalid in (
        ("base_attempts_per_role", True),
        ("escalation_attempts_per_role", True),
        ("max_retries", False),
        ("authorized", 0),
    ):
        with pytest.raises(ValidationError):
            CapabilitySmokePlanV1(
                release_id="release", registry_keys=("role",), attempts=attempts, **{field: invalid}
            )


def test_source_hash_and_decision_count_maps_are_frozen_and_canonical() -> None:
    manifest = QualificationReleaseManifestV1(
        release_id="release",
        base_commit="a" * 40,
        artifact_order=QUALIFICATION_ARTIFACT_ORDER,
        source_hashes={"source": HASH_A},
        source_byte_counts={"source": 1},
    )
    receipt = QualificationValidationReceiptV1(
        release_id="release",
        status="SUCCESS",
        source_count=1,
        decision_counts={"READY": 1},
    )

    for model, field, key, replacement in (
        (manifest, "source_hashes", "source", HASH_B),
        (manifest, "source_byte_counts", "source", 2),
        (receipt, "decision_counts", "READY", 2),
    ):
        before = canonical_bytes(model)
        before_json = model.model_dump(mode="json")
        with pytest.raises(TypeError):
            getattr(model, field)[key] = replacement
        assert canonical_bytes(model) == before
        assert model.model_dump(mode="json") == before_json


def test_smoke_plan_validates_two_roles_and_rejects_duplicate_coordinates_and_calls() -> None:
    attempts = _attempts_for_role("role-a") + _attempts_for_role("role-b")
    plan = CapabilitySmokePlanV1(
        release_id="release",
        registry_keys=("role-a", "role-b"),
        attempts=attempts,
    )

    assert len(plan.attempts) == 32
    for role in plan.registry_keys:
        assert sum(item.registry_key == role and item.phase is AttemptPhase.BASE for item in plan.attempts) == 8
        assert sum(item.registry_key == role and item.phase is AttemptPhase.ESCALATION for item in plan.attempts) == 8
    duplicate_coordinate = list(attempts)
    duplicate_coordinate[-1] = _attempt(
        registry_key="role-b",
        fixture_id="escalation-0",
        phase=AttemptPhase.ESCALATION,
        repetition=1,
        prompt_sha256=HASH_D,
    )
    with pytest.raises(ValidationError):
        CapabilitySmokePlanV1(
            release_id="release", registry_keys=("role-a", "role-b"), attempts=duplicate_coordinate
        )
    duplicate_call = list(attempts)
    duplicate_call[-1] = duplicate_call[-2]
    with pytest.raises(ValidationError):
        CapabilitySmokePlanV1(
            release_id="release", registry_keys=("role-a", "role-b"), attempts=duplicate_call
        )


def test_planned_fixture_attempt_receipt_and_decision_interfaces() -> None:
    fixture = CapabilityFixtureV1(
        fixture_id="fixture",
        category="EXACT_OUTPUT",
        prompt_sha256=HASH_A,
        parser_sha256=HASH_B,
        max_prompt_tokens=1,
        max_output_tokens=1,
    )
    receipt = CapabilityAttemptReceiptV1(
        call_id=HASH_C,
        registry_key="role",
        status=GateStatus.NOT_RUN,
        response_format="LOCAL_TEXT",
    )
    decision = QualificationDecisionV1(
        registry_key="role",
        scope=DecisionScope.CAPABILITY_SMOKE,
        status=QualificationStatus.BLOCKED,
        reasons=("not run",),
        evidence_binding_ids=("source",),
    )

    assert fixture.category == "EXACT_OUTPUT"
    assert receipt.retry_count == 0
    assert decision.scientific_status == "NOT_RUN"
    with pytest.raises(ValidationError):
        CapabilityAttemptReceiptV1(
            call_id=HASH_C, registry_key="role", status=GateStatus.NOT_RUN, retry_count=False
        )


def test_runtime_requires_explicit_reproducibility_fields() -> None:
    payload = {
        "engine": "transformers",
        "engine_version": "v1",
        "device_name": "A40",
        "context_tokens": 1,
        "max_output_tokens": 1,
        "trust_remote_code": False,
        "compute_dtype": "bf16",
        "attention_implementation": "eager",
        "seed": 0,
        "sampling_mode": "greedy",
        "timeout_seconds": 60,
        "engine_args_sha256": HASH_A,
    }
    assert RuntimeManifestV1(**payload).device_name == "A40"
    for required_field in (
        "device_name",
        "context_tokens",
        "max_output_tokens",
        "trust_remote_code",
        "compute_dtype",
        "attention_implementation",
        "seed",
        "sampling_mode",
        "timeout_seconds",
        "engine_args_sha256",
    ):
        incomplete = dict(payload)
        incomplete.pop(required_field)
        with pytest.raises(ValidationError):
            RuntimeManifestV1(**incomplete)


@pytest.mark.parametrize("field,value", [("engine_version", " "), ("device_name", ""), ("attention_implementation", "\t")])
def test_runtime_rejects_blank_reproducibility_strings(field: str, value: str) -> None:
    payload = {
        "engine": "transformers",
        "engine_version": "v1",
        "device_name": "A40",
        "context_tokens": 1,
        "max_output_tokens": 1,
        "trust_remote_code": False,
        "compute_dtype": "bf16",
        "attention_implementation": "eager",
        "seed": 0,
        "sampling_mode": "greedy",
        "timeout_seconds": 60,
        "engine_args_sha256": HASH_A,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        RuntimeManifestV1(**payload)


def test_attempt_receipt_accepts_non_end_turn_local_stop_reason() -> None:
    receipt = CapabilityAttemptReceiptV1(
        call_id=HASH_A,
        registry_key="role",
        status=GateStatus.NOT_RUN,
        stop_reason="local_length_limit",
    )

    assert receipt.stop_reason == "local_length_limit"


    receipt = QualificationValidationReceiptV1(
        release_id="release",
        status="SUCCESS_WITH_BLOCKERS",
        source_count=1,
        decision_counts={"BLOCKED": 1},
    )

    assert receipt.provider_calls_during_publication == 0
    for field in (
        "provider_calls_during_publication",
        "model_loads_during_publication",
        "network_calls_during_publication",
        "credential_reads_during_publication",
        "benchmark_generations",
    ):
        with pytest.raises(ValidationError):
            QualificationValidationReceiptV1(
                release_id="release",
                status="SUCCESS",
                source_count=1,
                decision_counts={"READY": 1},
                **{field: False},
            )


def test_production_config_is_canonical_and_has_complete_frozen_payload() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "vnext"
        / "post_core"
        / "qualification_release_v1.json"
    )
    raw = config_path.read_bytes()
    payload = json.loads(raw)

    assert raw == canonical_bytes(payload)
    assert payload == {
        "base_attempts_per_role": 8,
        "base_commit": "a56857431023d2af1a392c75c5575316a916c174",
        "escalation_attempts_per_role": 8,
        "max_retries": 0,
        "publisher_network_allowed": False,
        "registry_keys": [
            "qwen35_9b_bf16",
            "meta_muse_glimmer_30b_int4",
            "meta_muse_glimmer_30b_bf16",
            "claude_sonnet_4_6",
            "claude_opus_4_8",
            "gemini_3_6_flash",
            "grok_4_5",
            "gpt_5_5",
        ],
        "release_id": "memupdatebench.post-core.qualification.v1",
        "required_source_sha256": {
            "core_manifest": "dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b",
            "handoff_source": "4c1424bd2da72e9ed1042f091256fc55484c2f04cfdc0f6a0b4cf731eb5519a2",
            "identity_evidence": "9e3780ed3d4303bda7bbd27865df89fcb384041da64af56107c8c5b7abf0a4f0",
            "open_snapshot_audit_receipt": "0b146bd8dc04e334d899801f4746bee0ae69635f1ace3f4c92ada8f32819940",
            "open_snapshot_closure_receipt": "77a69e02a8b092b7e1bf5e89ff9a5f69b449c89a1c2cd319f9c48edd3e2f4645",
            "phase0_index": "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd",
            "qwen_load_receipt": "fd4e47d75d86efdbe9add3cc469017b9aef23bb05bc4d03b74877bfbe289f6b7",
            "task14_index": "2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035",
            "workflow_source": "b2dc80c6dc30b74aff597cdeb83044056fb24efe7a260b39a004aa5d2f4905cb",
        },
        "schema_version": "memupdatebench.post-core.qualification-config.v1",
        "scientific_execution_allowed": False,
    }


def test_capability_adapter_result_is_immutable_and_cannot_declare_a_verdict() -> None:
    result = CapabilityAdapterResultV1(
        call_id=HASH_A,
        registry_key="role",
        request_sha256=HASH_B,
        provider_call_count=1,
        retry_count=0,
        response_projection="READY",
        response_format="LOCAL_TEXT",
        latency_ms=1,
    )

    assert result.schema_version == "memupdatebench.post-core.capability-adapter-result.v1"
    assert "CapabilityAdapterResultV1" in __import__(
        "mub.vnext.post_core.qualification_receipts_v1", fromlist=["__all__"]
    ).__all__
    assert tuple(CapabilityAdapterResultV1.model_fields) == (
        "schema_version", "call_id", "registry_key", "request_sha256", "provider_call_count", "retry_count",
        "response_projection", "response_model", "response_format", "stop_reason", "usage_present", "latency_ms",
        "error_class",
    )
    with pytest.raises(ValidationError):
        CapabilityAdapterResultV1.model_validate({**result.model_dump(mode="json"), "status": "PASS"})
    with pytest.raises(ValidationError):
        CapabilityAdapterResultV1.model_validate({**result.model_dump(mode="json"), "redacted_response_sha256": HASH_B})
    with pytest.raises(ValidationError):
        result.response_projection = "ACK"


def test_adapter_result_requires_exact_request_hash_and_single_dispatched_call_attestation() -> None:
    result = CapabilityAdapterResultV1(
        call_id=HASH_A,
        registry_key="role",
        request_sha256=HASH_B,
        provider_call_count=1,
        retry_count=0,
        response_projection="READY",
        response_format="LOCAL_TEXT",
        latency_ms=1,
    )

    assert result.request_sha256 == HASH_B
    assert result.provider_call_count == 1
    assert result.retry_count == 0
    for field, value in (
        ("request_sha256", "A" * 64),
        ("provider_call_count", 0),
        ("provider_call_count", True),
        ("retry_count", 1),
        ("retry_count", False),
    ):
        payload = result.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError):
            CapabilityAdapterResultV1.model_validate(payload)
