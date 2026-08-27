from __future__ import annotations

from scripts.vnext_admit_letta_profile import build_admission_receipt


def _blocked_preflight() -> dict:
    return {
        "schema_version": "memupdatebench.external.letta.preflight.v1",
        "candidate_id": "letta_0_16_8_block_profile",
        "mode": "direct_block_profile",
        "identity": {
            "package_name": "letta",
            "package_version": "0.16.8",
            "source_repository": "letta-ai/letta",
            "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24",
            "license_id": "Apache-2.0",
        },
        "package_preflight": {"identity_verified": False},
        "namespace_reset_probe": {"passed": False, "trials": []},
        "lifecycle": {"passed": False},
        "execution_boundary": {
            "llm_used": False,
            "api_used": False,
            "gpu_used": False,
            "network_credential_inputs": False,
        },
        "unsupported": {
            "passage_memory": True,
            "agent_mode": True,
            "native_answer": True,
        },
        "capabilities": {"supports_multi_object_query": False},
        "passed": False,
        "outcome": "blocked",
        "blockers": ["letta_package_not_installed"],
    }


def test_letta_admission_receipt_keeps_unverified_runtime_blocked() -> None:
    receipt = build_admission_receipt(_blocked_preflight())

    assert receipt["admitted"] is False
    assert receipt["outcome"] == "blocked"
    assert receipt["admission_scope"] == "direct_block_profile_only"
    assert "runtime_preflight_incomplete" in receipt["reasons"]
    assert receipt["gates"]["frozen_package_identity"] == "pass"


def test_letta_admission_receipt_rejects_untruthful_unsupported_surface() -> None:
    preflight = _blocked_preflight()
    preflight["unsupported"]["agent_mode"] = False

    receipt = build_admission_receipt(preflight)

    assert "unsupported_surface_not_explicit" in receipt["reasons"]
