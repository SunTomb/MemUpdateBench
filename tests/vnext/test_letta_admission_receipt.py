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


def test_runtime_admission_requires_all_runtime_gates() -> None:
    from scripts.vnext_admit_letta_runtime import build_admission_receipt

    preflight = {
        "schema_version": "memupdatebench.external.letta.preflight.v2",
        "candidate_id": "letta_0_16_8_profile",
        "mode": "profile_single_record_runtime",
        "identity": _blocked_preflight()["identity"],
        "official_health": {"passed": True, "source_binding": "verified"},
        "runtime": {"loopback": True, "database_isolated": True},
        "namespace_reset_probe": {"passed": True},
        "lifecycle": {"passed": True},
        "clean_close": {"passed": True},
        "security": {"secret_scan_passed": True, "raw_logs_recorded": False},
        "boundary": {"llm_used": False, "api_used": False, "gpu_used": False, "network_credential_inputs": False},
        "unsupported": {"multi_object_query": True, "native_answer": True, "historical_query": True, "version_history_export": True, "scoped_delete": True},
        "passed": True, "outcome": "pass",
    }
    receipt = build_admission_receipt(preflight)
    assert receipt["admitted"] is True
    assert receipt["schema_version"] == "memupdatebench.external.letta.admission.v2"

    preflight["runtime"]["database_isolated"] = False
    blocked = build_admission_receipt(preflight)
    assert blocked["admitted"] is False
    assert "database_isolation_failed" in blocked["reasons"]
