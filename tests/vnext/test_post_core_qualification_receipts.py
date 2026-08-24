from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import canonical_bytes, canonical_hash
from mub.vnext.post_core.qualification_receipts_v1 import (
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
    import json

    payload = json.loads(raw)

    assert raw == canonical_bytes(payload)
    assert len(payload["required_source_sha256"]) == 9
    assert set(payload["required_source_sha256"]) == {
        "core_manifest",
        "handoff_source",
        "identity_evidence",
        "open_snapshot_audit_receipt",
        "open_snapshot_closure_receipt",
        "phase0_index",
        "qwen_load_receipt",
        "task14_index",
        "workflow_source",
    }
