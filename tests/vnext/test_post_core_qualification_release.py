from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.qualification_receipts_v1 import (
    AttemptPhase,
    CapabilityAttemptPlanV1,
    CapabilityBudgetV1,
    CapabilitySmokePlanV1,
    DecisionScope,
    QualificationDecisionBundleV1,
    QualificationDecisionV1,
    QualificationStatus,
    QUALIFICATION_ARTIFACT_ORDER,
    QUALIFICATION_INDEX_PATH,
)
from tests.vnext.qualification_fixtures import open_runtime_receipts, provider_attestations
from mub.vnext.post_core.qualification_release_v1 import (
    BASE_COMMIT,
    QUALIFICATION_ARTIFACTS,
    build_qualification_release_v1,
    load_qualification_release_config_v1,
    verify_qualification_artifact_bytes_v1,
)


def _source_inputs(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    source_ids = (
        "core_manifest", "handoff_source", "identity_evidence", "open_snapshot_audit_receipt",
        "open_snapshot_closure_receipt", "phase0_index", "qwen_load_receipt", "task14_index", "workflow_source",
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for index, source_id in enumerate(source_ids):
        path = tmp_path / f"{source_id}.blob"
        raw = f"source-{index}".encode()
        path.write_bytes(raw)
        paths[source_id] = path
        hashes[source_id] = hashlib.sha256(raw).hexdigest()
    payload = {
        "base_attempts_per_role": 8,
        "base_commit": BASE_COMMIT,
        "escalation_attempts_per_role": 8,
        "max_retries": 0,
        "publisher_network_allowed": False,
        "registry_keys": [
            "qwen35_9b_bf16", "meta_muse_glimmer_30b_int4", "meta_muse_glimmer_30b_bf16",
            "claude_sonnet_4_6", "claude_opus_4_8", "gemini_3_6_flash", "grok_4_5", "gpt_5_5",
        ],
        "release_id": "memupdatebench.post-core.qualification.v1",
        "required_source_sha256": hashes,
        "schema_version": "memupdatebench.post-core.qualification-config.v1",
        "scientific_execution_allowed": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_bytes(payload))
    return paths, config_path


def _smoke_plan(release_id: str, keys: tuple[str, ...]) -> CapabilitySmokePlanV1:
    budget = CapabilityBudgetV1(
        max_calls=1, max_prompt_tokens=8, max_output_tokens=8,
        estimated_cost=Decimal("0"), hard_max_cost=Decimal("0"), price_version="unpriced",
        max_retries=0, timeout_seconds=1,
    )
    attempts = []
    for key in keys:
        for phase in (AttemptPhase.BASE, AttemptPhase.ESCALATION):
            for index in range(8):
                attempts.append(CapabilityAttemptPlanV1(
                    release_id=release_id, registry_key=key, fixture_id=f"{phase.value}-{index}",
                    phase=phase, repetition=(index % 2) + 1, prompt_sha256="a" * 64,
                    parser_sha256="b" * 64, runtime_or_endpoint_class="offline", budget=budget,
                ))
    return CapabilitySmokePlanV1(release_id=release_id, registry_keys=keys, attempts=tuple(attempts))


def _decisions(release_id: str, keys: tuple[str, ...]) -> QualificationDecisionBundleV1:
    return QualificationDecisionBundleV1(
        release_id=release_id,
        decisions=tuple(
            QualificationDecisionV1(
                registry_key=key, scope=scope, status=QualificationStatus.BLOCKED,
                reasons=("not run",), evidence_binding_ids=("core_manifest",),
            )
            for key in keys for scope in DecisionScope
        ),
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    paths, config_path = _source_inputs(tmp_path)
    config = load_qualification_release_config_v1(config_path)
    return {
        "config": config, "source_paths": paths,
        "provider_attestations": provider_attestations(), "runtime_receipts": open_runtime_receipts(),
        "smoke_plan": _smoke_plan(config.release_id, config.registry_keys),
        "decision_bundle": _decisions(config.release_id, config.registry_keys),
    }


def test_builder_emits_exact_deterministic_artifacts_and_zero_counters(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    left = build_qualification_release_v1(**inputs)
    right = build_qualification_release_v1(**inputs)
    assert tuple(left.artifact_bytes) == QUALIFICATION_ARTIFACTS
    assert tuple(left.artifact_bytes) == (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)
    assert dict(left.artifact_bytes) == dict(right.artifact_bytes)
    assert left.index_sha256 == right.index_sha256
    assert (left.provider_calls, left.model_loads, left.network_calls, left.credential_reads, left.benchmark_generations) == (0, 0, 0, 0, 0)
    verify_qualification_artifact_bytes_v1(left)
    assert "stale_copied" not in left.artifact_bytes["qualification_validation_receipt.json"].decode()


def test_builder_rejects_source_substitution_and_missing_source(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    paths = dict(inputs["source_paths"])
    paths["workflow_source"] = replacement
    with pytest.raises(ValueError, match="workflow_source"):
        build_qualification_release_v1(**{**inputs, "source_paths": paths})
    paths.pop("workflow_source")
    with pytest.raises(ValueError, match="exact nine"):
        build_qualification_release_v1(**{**inputs, "source_paths": paths})


def test_builder_rejects_secret_like_and_metric_fields_before_exposing_bytes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bad = dict(inputs)
    bad["decision_bundle"] = _decisions(inputs["config"].release_id, inputs["config"].registry_keys).model_copy(update={"decisions": ()})
    with pytest.raises(ValueError):
        build_qualification_release_v1(**bad)
