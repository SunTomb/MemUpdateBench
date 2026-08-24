from decimal import Decimal
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import canonical_bytes, canonical_hash
from mub.vnext.post_core.qualification_receipts_v1 import (
    ArtifactBindingV1,
    AttemptPhase,
    CapabilityAttemptPlanV1,
    CapabilityBudgetV1,
    DecisionScope,
    QualificationArtifactIndexV1,
    QualificationStatus,
    SourceBindingV1,
    QUALIFICATION_ARTIFACT_ORDER,
)


def _binding() -> SourceBindingV1:
    return SourceBindingV1(
        source_id="source",
        evidence_class="fixture",
        sha256="a" * 64,
        required=True,
    )


def _budget() -> CapabilityBudgetV1:
    return CapabilityBudgetV1(
        max_calls=1,
        input_token_cap=1,
        output_token_cap=1,
        estimated_cost_usd=Decimal("0.01"),
        hard_max_cost_usd=Decimal("0.02"),
        price_version="v1",
        max_retries=0,
        timeout_seconds=1,
    )


def _attempt(**overrides: object) -> CapabilityAttemptPlanV1:
    payload: dict[str, object] = {
        "release_id": "release",
        "registry_key": "model",
        "fixture_id": "fixture",
        "phase": AttemptPhase.BASE,
        "repetition": 1,
        "prompt_sha256": "b" * 64,
        "parser_sha256": "c" * 64,
        "runtime_or_endpoint_class": "offline",
        "budget": _budget(),
    }
    payload.update(overrides)
    return CapabilityAttemptPlanV1(**payload)


def test_source_binding_preserves_none_byte_count_and_models_are_frozen() -> None:
    binding = _binding()

    assert binding.byte_count is None
    with pytest.raises(ValidationError):
        binding.source_id = "changed"


def test_artifact_binding_uses_the_qualification_schema_version() -> None:
    binding = ArtifactBindingV1(path="artifact.json", sha256="d" * 64)

    assert (
        binding.schema_version
        == "memupdatebench.post-core.qualification-artifact-binding.v1"
    )


def test_capability_budget_is_single_call_no_retry_and_cost_bounded() -> None:
    budget = _budget()

    assert budget.max_calls == 1
    assert budget.max_retries == 0
    assert budget.estimated_cost_usd.is_finite()
    assert budget.estimated_cost_usd <= budget.hard_max_cost_usd
    with pytest.raises(ValidationError):
        CapabilityBudgetV1(
            **{**budget.model_dump(), "estimated_cost_usd": Decimal("NaN")}
        )


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


def test_qualification_artifact_index_has_exact_order_and_no_self_hash() -> None:
    artifacts = [
        {"path": path, "sha256": f"{index:064x}"}
        for index, path in enumerate(QUALIFICATION_ARTIFACT_ORDER, start=1)
    ]
    index = QualificationArtifactIndexV1(release_id="release", artifacts=artifacts)

    assert tuple(binding.path for binding in index.artifacts) == QUALIFICATION_ARTIFACT_ORDER
    assert index.canonical_hash == canonical_hash(index, exclude={"canonical_hash"})
    with pytest.raises(ValidationError):
        QualificationArtifactIndexV1(
            release_id="release", artifacts=list(reversed(artifacts))
        )


def test_production_config_is_canonical_and_has_nine_frozen_sources() -> None:
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
