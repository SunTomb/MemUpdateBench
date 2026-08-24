from __future__ import annotations

from pathlib import Path

import pytest

from mub.vnext.post_core.identity_v1 import IdentityEvidenceBundleV1
from mub.vnext.post_core.qualification_receipts_v1 import (
    DecisionScope,
    QualificationStatus,
)
from mub.vnext.post_core.qualification_validation_v1 import (
    validate_provider_attestations_v1,
    validate_runtime_receipts_v1,
)
from tests.vnext.qualification_fixtures import open_runtime_receipts, provider_attestations


IDENTITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "vnext"
    / "post_core"
    / "official_identity_evidence_v1.json"
)
EXPECTED_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)
EXPECTED_SCOPES = (
    DecisionScope.STORAGE_INPUT,
    DecisionScope.SHORT_GENERATION_GATE,
    DecisionScope.CAPABILITY_SMOKE,
    DecisionScope.BENCHMARK_ADMISSION,
)


def identity_bundle() -> IdentityEvidenceBundleV1:
    return IdentityEvidenceBundleV1.model_validate_json(IDENTITY_PATH.read_bytes())


def decisions_by_key_scope():
    from mub.vnext.post_core.qualification_decisions_v1 import derive_qualification_decisions_v1

    return derive_qualification_decisions_v1(
        identity_bundle(), provider_attestations(), open_runtime_receipts()
    )


def test_public_surface_exposes_only_the_derivation_function() -> None:
    import mub.vnext.post_core.qualification_decisions_v1 as decisions

    assert decisions.__all__ == ["derive_qualification_decisions_v1"]


def test_sonnet_capability_is_ready_but_benchmark_admission_is_blocked() -> None:
    decisions = decisions_by_key_scope()
    sonnet = {row.scope: row for row in decisions if row.registry_key == "claude_sonnet_4_6"}

    assert sonnet[DecisionScope.CAPABILITY_SMOKE].status is QualificationStatus.READY
    assert sonnet[DecisionScope.BENCHMARK_ADMISSION].status is QualificationStatus.BLOCKED
    assert sonnet[DecisionScope.BENCHMARK_ADMISSION].scientific_status == "NOT_RUN"


def test_grok_and_gpt_capability_can_be_ready_but_identity_caveat_blocks_benchmark() -> None:
    decisions = decisions_by_key_scope()

    for key in ("grok_4_5", "gpt_5_5"):
        selected = {row.scope: row for row in decisions if row.registry_key == key}
        assert selected[DecisionScope.CAPABILITY_SMOKE].status is QualificationStatus.READY
        benchmark = selected[DecisionScope.BENCHMARK_ADMISSION]
        assert benchmark.status is QualificationStatus.BLOCKED
        assert "identity caveat" in " ".join(benchmark.reasons).lower()


def test_qwen_load_only_receipt_allows_storage_but_blocks_short_and_smoke() -> None:
    decisions = decisions_by_key_scope()
    qwen = {row.scope: row for row in decisions if row.registry_key == "qwen35_9b_bf16"}

    assert qwen[DecisionScope.STORAGE_INPUT].status is QualificationStatus.READY
    assert qwen[DecisionScope.SHORT_GENERATION_GATE].status is QualificationStatus.BLOCKED
    assert qwen[DecisionScope.CAPABILITY_SMOKE].status is QualificationStatus.BLOCKED


def test_muse_bf16_resource_block_keeps_storage_identity_but_blocks_execution_gates() -> None:
    decisions = decisions_by_key_scope()
    muse = {
        row.scope: row
        for row in decisions
        if row.registry_key == "meta_muse_glimmer_30b_bf16"
    }

    assert muse[DecisionScope.STORAGE_INPUT].status is QualificationStatus.READY
    assert muse[DecisionScope.SHORT_GENERATION_GATE].status is QualificationStatus.BLOCKED
    assert muse[DecisionScope.CAPABILITY_SMOKE].status is QualificationStatus.BLOCKED
    assert "resource" in " ".join(muse[DecisionScope.SHORT_GENERATION_GATE].reasons).lower()


def test_output_follows_frozen_identity_order_and_fixed_scope_order() -> None:
    decisions = decisions_by_key_scope()

    assert len(decisions) == len(EXPECTED_KEYS) * len(EXPECTED_SCOPES)
    assert tuple(row.registry_key for row in decisions[::4]) == EXPECTED_KEYS
    assert tuple(row.scope for row in decisions[:4]) == EXPECTED_SCOPES
    for index, key in enumerate(EXPECTED_KEYS):
        assert tuple(row.scope for row in decisions[index * 4 : (index + 1) * 4]) == EXPECTED_SCOPES
        assert all(row.registry_key == key for row in decisions[index * 4 : (index + 1) * 4])
    assert isinstance(decisions, tuple)


def test_decisions_have_no_metrics_or_benchmark_fields_and_scientific_status_is_not_run() -> None:
    decisions = decisions_by_key_scope()

    assert all(row.scientific_status == "NOT_RUN" for row in decisions)
    fields = set(type(decisions[0]).model_fields)
    assert fields == {
        "schema_version",
        "registry_key",
        "scope",
        "status",
        "reasons",
        "evidence_binding_ids",
        "scientific_status",
    }
    assert all(row.status is not QualificationStatus.READY or row.scope is not DecisionScope.BENCHMARK_ADMISSION for row in decisions)
    assert all(
        row.scope is not DecisionScope.BENCHMARK_ADMISSION
        or row.status is QualificationStatus.BLOCKED
        for row in decisions
    )


def test_provider_and_runtime_inputs_are_validated_before_derivation() -> None:
    from mub.vnext.post_core.qualification_decisions_v1 import derive_qualification_decisions_v1

    with pytest.raises(ValueError, match="provider"):
        derive_qualification_decisions_v1(identity_bundle(), (provider_attestations()[0],), open_runtime_receipts())
    with pytest.raises(ValueError, match="runtime"):
        derive_qualification_decisions_v1(identity_bundle(), provider_attestations(), (open_runtime_receipts()[0],))
    with pytest.raises(ValueError, match="ProviderCapabilityAttestationV1"):
        derive_qualification_decisions_v1(identity_bundle(), (object(),) * 5, open_runtime_receipts())
    assert validate_provider_attestations_v1(provider_attestations())
    assert validate_runtime_receipts_v1(open_runtime_receipts())


def test_derivation_does_not_access_environment_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import mub.vnext.post_core.qualification_decisions_v1 as decisions

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("qualification derivation must remain offline")

    monkeypatch.setattr("socket.create_connection", fail)
    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("subprocess.run", fail)
    monkeypatch.setattr("os.getenv", fail)
    assert decisions.derive_qualification_decisions_v1(
        identity_bundle(), provider_attestations(), open_runtime_receipts()
    )
