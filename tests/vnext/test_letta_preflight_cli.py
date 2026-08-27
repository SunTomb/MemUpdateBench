from __future__ import annotations

from scripts.vnext_preflight_letta import run_preflight


def test_letta_preflight_is_metadata_only_and_fails_closed_without_verified_runtime() -> None:
    payload = run_preflight(run_prefix="letta-test-preflight")

    assert payload["schema_version"] == "memupdatebench.external.letta.preflight.v1"
    assert payload["identity"]["package_version"] == "0.16.8"
    assert payload["identity"]["license_id"] == "Apache-2.0"
    assert payload["execution_boundary"] == {
        "llm_used": False,
        "api_used": False,
        "gpu_used": False,
        "network_credential_inputs": False,
    }
    assert payload["unsupported"]["passage_memory"] is True
    assert payload["unsupported"]["agent_mode"] is True
    assert payload["unsupported"]["native_answer"] is True
    assert payload["package_preflight"]["identity_verified"] is False
    assert payload["outcome"] == "blocked"
    assert payload["passed"] is False
