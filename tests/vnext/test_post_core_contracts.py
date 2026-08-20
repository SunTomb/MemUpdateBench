from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import (
    ArtifactIndexV1,
    CandidateIdentityState,
    CallBudgetV1,
    CallPlanV1,
    MatrixCellV1,
    ModelCandidateV1,
    ModelIdentityV1,
    QuantizationSpecV1,
    ReleaseManifestV1,
    SpeculativeDecodingSpecV1,
    canonical_hash,
)

_SHA = "a" * 64


def _identity(**changes):
    data = dict(
        official_model_id="official/model",
        revision="r1",
        license_id="apache-2.0",
        architecture="CausalLM",
        weights_uri="https://example.invalid/model",
        tokenizer_identity="tok-r1",
        endpoint=None,
        resolved_upstream_identity="official/model@r1",
    )
    data.update(changes)
    return ModelIdentityV1(**data)


def test_pending_identity_requires_all_identity_fields_null() -> None:
    with pytest.raises(ValidationError, match="pending identity fields"):
        ModelCandidateV1(
            registry_key="candidate",
            role="modern_open_anchor",
            state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY,
            identity=_identity(),
            scopes=("full",),
        )


def test_contracts_are_strict_frozen_and_canonical_hash_is_stable() -> None:
    candidate = ModelCandidateV1(
        registry_key="candidate",
        role="modern_open_anchor",
        state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY,
        identity=None,
        scopes=("full",),
    )
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        candidate.registry_key = "changed"
    assert canonical_hash(candidate) == canonical_hash(candidate.model_copy())
    with pytest.raises(ValidationError):
        ModelCandidateV1.model_validate(
            {**candidate.model_dump(), "unexpected": 1}, strict=True
        )


def test_quantization_and_speculative_factors_are_explicit() -> None:
    quant = QuantizationSpecV1(
        mode="int4", bits=4, group_size=128, compute_dtype="bf16", checkpoint_sha256=_SHA
    )
    assert quant.mode == "int4"
    with pytest.raises(ValidationError, match="static"):
        QuantizationSpecV1(mode="int4", bits=4, group_size=128, compute_dtype="bf16")
    with pytest.raises(ValidationError, match="parity"):
        SpeculativeDecodingSpecV1(mode="on")


def test_matrix_call_budget_and_release_reject_pending_execution() -> None:
    cell = MatrixCellV1(
        model_key="candidate",
        context_order="chronological",
        context_annotation="none",
        retrieval_k=16,
        precision="bf16",
        quantization="none",
        speculative_mode="off",
        repetition=1,
        seed=0,
        prompt_sha256=_SHA,
    )
    budget = CallBudgetV1(
        max_calls=1,
        max_prompt_tokens=100,
        max_output_tokens=20,
        estimated_cost=Decimal("0"),
        hard_max_cost=Decimal("1"),
        price_version="unpriced",
        max_retries=0,
        timeout_seconds=30,
    )
    with pytest.raises(ValidationError, match="pending"):
        CallPlanV1(
            call_id=cell.call_id,
            model_key="candidate",
            state=CandidateIdentityState.PENDING_OFFICIAL_IDENTITY,
            executable=True,
            matrix_cell=cell,
            budget=budget,
        )


def test_release_artifact_order_and_index_is_non_self_hashing() -> None:
    manifest = ReleaseManifestV1(
        release_id="memupdatebench.post-core.release.v1",
        schema_version="memupdatebench.post-core.release.v1",
        artifact_order=(
            "post_core_release_manifest.json",
            "model_registry.json",
            "provenance.jsonl",
            "qualification_report.json",
            "capability_probe_report.json",
            "execution_plan.json",
        ),
        source_hashes={"core_task14": _SHA},
    )
    index = ArtifactIndexV1(
        release_id=manifest.release_id,
        artifacts=tuple(
            {"path": name, "sha256": _SHA}
            for name in manifest.artifact_order
        ),
    )
    assert "post_core_artifact_index.json" not in manifest.artifact_order
    assert index.canonical_hash == canonical_hash(index, exclude={"canonical_hash"})
