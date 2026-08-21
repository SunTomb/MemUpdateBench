from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mub.vnext.post_core.contracts_v1 import CandidateIdentityState, canonical_bytes
from mub.vnext.post_core.identity_v1 import (
    EXPECTED_IDENTITY_EVIDENCE_SHA256,
    EXPECTED_IDENTITY_KEYS,
    IdentityEvidenceBundleV1,
    IdentityEvidenceReceiptV1,
    build_identity_evidence_receipt_v1,
    load_identity_evidence_v1,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "configs/vnext/post_core/official_identity_evidence_v1.json"
PHASE0_INDEX = (
    ROOT
    / "tests/vnext/fixtures/post_core/phase0_0745fc9_clean_v1/"
    "post_core_artifact_index.json"
)


def _load() -> IdentityEvidenceBundleV1:
    return load_identity_evidence_v1(EVIDENCE, PHASE0_INDEX)


def test_identity_evidence_has_exact_candidate_order_and_phase0_binding() -> None:
    bundle = _load()
    assert tuple(row.registry_key for row in bundle.records) == EXPECTED_IDENTITY_KEYS
    assert bundle.phase0_index_sha256 == (
        "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd"
    )
    assert bundle.retrieved_on == "2026-08-21"
    assert canonical_bytes(bundle) == EVIDENCE.read_bytes()
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        EXPECTED_IDENTITY_EVIDENCE_SHA256
    )
    assert hashlib.sha256(PHASE0_INDEX.read_bytes()).hexdigest() == (
        "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd"
    )


def test_open_model_evidence_is_pinned_and_waits_for_local_snapshots() -> None:
    records = {row.registry_key: row for row in _load().records}
    qwen = records["qwen35_9b_bf16"]
    assert qwen.state is CandidateIdentityState.PENDING_LOCAL_SNAPSHOT
    assert qwen.official_model_id == "Qwen/Qwen3.5-9B"
    assert qwen.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert qwen.license_id == "Apache-2.0"
    assert qwen.architecture == "Qwen3_5ForConditionalGeneration"
    assert qwen.parameter_count == 9_653_104_368
    assert len(qwen.artifacts) >= 4

    glimmer = records["meta_muse_glimmer_30b_int4"]
    assert glimmer.state is CandidateIdentityState.PENDING_LOCAL_SNAPSHOT
    assert glimmer.official_model_id == "meta-models/Muse-Glimmer-30B-GGUF"
    assert glimmer.revision == "70bf1b61ac09f91b24d39038091b41c582bc5d7a"
    assert glimmer.base_model_id == "meta-models/Muse-Glimmer-30B"
    assert glimmer.base_revision == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    target = next(item for item in glimmer.artifacts if item.role == "quantized_target")
    assert target.path == "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf"
    assert target.sha256 == "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38"


def test_closed_identity_states_preserve_fixed_version_boundary() -> None:
    records = {row.registry_key: row for row in _load().records}
    for key, model_id in (
        ("claude_sonnet_4_6", "claude-sonnet-4-6"),
        ("claude_opus_4_8", "claude-opus-4-8"),
        ("gemini_3_6_flash", "gemini-3.6-flash"),
    ):
        assert records[key].state is CandidateIdentityState.READY_FOR_PROVIDER_PREFLIGHT
        assert records[key].official_model_id == model_id
        assert records[key].response_identity_field in {"model", "modelVersion"}

    grok = records["grok_4_5"]
    assert grok.state is CandidateIdentityState.PENDING_PROVIDER_QUALIFICATION
    assert grok.official_model_id == "grok-4.5"
    assert grok.mutable_identifier is True
    assert grok.revision is None

    gpt = records["gpt_5_5"]
    assert gpt.state is CandidateIdentityState.PENDING_OFFICIAL_IDENTITY
    assert gpt.official_model_id is None
    assert gpt.evidence_class == "not_verified_in_official_catalog"


def test_identity_receipt_is_deterministic_and_executes_nothing() -> None:
    bundle = _load()
    left = build_identity_evidence_receipt_v1(bundle, EVIDENCE)
    right = build_identity_evidence_receipt_v1(bundle, EVIDENCE)
    assert left == right
    assert isinstance(left, IdentityEvidenceReceiptV1)
    assert left.evidence_sha256 == hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    assert left.candidate_count == 8
    assert left.provider_calls == 0
    assert left.model_loads == 0
    assert left.network_calls == 0
    assert left.executable_calls == 0
    assert left.state_counts == {
        "PENDING_LOCAL_SNAPSHOT": 3,
        "PENDING_OFFICIAL_IDENTITY": 1,
        "PENDING_PROVIDER_QUALIFICATION": 1,
        "READY_FOR_PROVIDER_PREFLIGHT": 3,
    }


def test_identity_contract_rejects_mutable_or_unverified_promotions(tmp_path: Path) -> None:
    payload = json.loads(EVIDENCE.read_bytes())
    records = {row["registry_key"]: row for row in payload["records"]}

    records["grok_4_5"]["state"] = "READY_FOR_PROVIDER_PREFLIGHT"
    bad_grok = tmp_path / "grok.json"
    bad_grok.write_bytes(canonical_bytes(payload))
    with pytest.raises(ValueError, match="mutable API alias"):
        load_identity_evidence_v1(bad_grok, PHASE0_INDEX)

    payload = json.loads(EVIDENCE.read_bytes())
    records = {row["registry_key"]: row for row in payload["records"]}
    records["gpt_5_5"]["official_model_id"] = "gpt-5.5"
    records["gpt_5_5"]["state"] = "READY_FOR_PROVIDER_PREFLIGHT"
    bad_gpt = tmp_path / "gpt.json"
    bad_gpt.write_bytes(canonical_bytes(payload))
    with pytest.raises(ValueError, match="unverified identity"):
        load_identity_evidence_v1(bad_gpt, PHASE0_INDEX)


def test_identity_loader_rejects_any_evidence_byte_substitution(tmp_path: Path) -> None:
    payload = json.loads(EVIDENCE.read_bytes())
    payload["records"][0]["source_urls"][0] = "https://example.com/untrusted"
    substituted = tmp_path / "substituted.json"
    substituted.write_bytes(canonical_bytes(payload))
    with pytest.raises(ValueError, match="authoritative official identity evidence"):
        load_identity_evidence_v1(substituted, PHASE0_INDEX)


def test_identity_loader_rejects_lexical_symlink_sources(tmp_path: Path) -> None:
    linked = tmp_path / "identity-link.json"
    try:
        os.symlink(EVIDENCE, linked)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="link|reparse"):
        load_identity_evidence_v1(linked, PHASE0_INDEX)


def test_identity_validation_cli_is_no_execution_and_secret_free() -> None:
    script = ROOT / "scripts/vnext_validate_post_core_identities.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert help_run.returncode == 0
    lowered = help_run.stdout.lower()
    for forbidden in (
        "--allow-network",
        "--api-key",
        "--token",
        "--download",
        "--model-load",
        "--provider",
        "--endpoint",
    ):
        assert forbidden not in lowered

    index = PHASE0_INDEX
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--evidence",
            str(EVIDENCE),
            "--phase0-index",
            str(index),
            "--execute",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    receipt = IdentityEvidenceReceiptV1.model_validate_json(run.stdout)
    assert canonical_bytes(receipt).decode("utf-8") + "\n" == run.stdout
    assert receipt.candidate_count == 8
    assert receipt.provider_calls == 0
    assert receipt.model_loads == 0
    assert receipt.network_calls == 0
    assert receipt.executable_calls == 0
