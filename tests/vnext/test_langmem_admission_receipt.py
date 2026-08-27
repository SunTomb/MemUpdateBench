from __future__ import annotations

import json
from pathlib import Path

from scripts.vnext_admit_langmem_profile import build_admission_receipt


def _preflight() -> dict:
    return {
        "schema_version": "memupdatebench.external.langmem.preflight.v1",
        "candidate_id": "langmem_0_0_30_profile",
        "mode": "profile_single_record",
        "identity": {
            "package_name": "langmem",
            "package_version": "0.0.30",
            "source_repository": "langchain-ai/langmem",
            "source_commit": "29cbe41e58528f92e9efa773c12e15c47be3808c",
            "license_id": "MIT",
        },
        "namespace_reset_probe": {"passed": True},
        "lifecycle": {"passed": True},
        "execution_boundary": {
            "llm_used": False,
            "api_used": False,
            "gpu_used": False,
            "network_credential_inputs": False,
        },
        "unsupported": {"collection_mode": True},
        "capabilities": {"supports_multi_object_query": False},
        "passed": True,
        "outcome": "pass",
        "blockers": [],
    }


def test_langmem_admission_receipt_passes_only_complete_profile_preflight() -> None:
    receipt = build_admission_receipt(_preflight())

    assert receipt["outcome"] == "pass"
    assert receipt["admitted"] is True
    assert receipt["admission_scope"] == "profile_single_record_only"
    assert receipt["reasons"] == []


def test_langmem_admission_receipt_blocks_license_identity_mismatch() -> None:
    preflight = _preflight()
    preflight["identity"]["license_id"] = "Apache-2.0"

    receipt = build_admission_receipt(preflight)

    assert receipt["outcome"] == "blocked"
    assert receipt["admitted"] is False
    assert "frozen_package_identity_mismatch" in receipt["reasons"]


def test_langmem_admission_reader_accepts_canonical_preflight(
    tmp_path: Path,
) -> None:
    from scripts.vnext_admit_langmem_profile import _read_canonical_json

    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(_preflight(), separators=(",", ":"), sort_keys=True))

    assert _read_canonical_json(preflight) == _preflight()
