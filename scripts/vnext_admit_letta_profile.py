from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import scan_for_secrets

_EXPECTED_IDENTITY = {
    "package_name": "letta",
    "package_version": "0.16.8",
    "source_repository": "letta-ai/letta",
    "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24",
    "license_id": "Apache-2.0",
}


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def build_admission_receipt(preflight: dict) -> dict:
    if not isinstance(preflight, dict):
        raise ValueError("Letta preflight must be a JSON object")
    identity_matches = preflight.get("identity") == _EXPECTED_IDENTITY
    runtime_complete = (
        preflight.get("schema_version") == "memupdatebench.external.letta.preflight.v1"
        and preflight.get("candidate_id") == "letta_0_16_8_block_profile"
        and preflight.get("mode") == "direct_block_profile"
        and preflight.get("package_preflight", {}).get("identity_verified") is True
        and preflight.get("namespace_reset_probe", {}).get("passed") is True
        and preflight.get("lifecycle", {}).get("passed") is True
        and preflight.get("passed") is True
        and preflight.get("outcome") == "pass"
    )
    boundary = preflight.get("execution_boundary", {})
    boundary_complete = all(
        boundary.get(name) is False
        for name in ("llm_used", "api_used", "gpu_used", "network_credential_inputs")
    )
    unsupported = preflight.get("unsupported", {})
    unsupported_explicit = (
        unsupported.get("passage_memory") is True
        and unsupported.get("agent_mode") is True
        and unsupported.get("native_answer") is True
        and preflight.get("capabilities", {}).get("supports_multi_object_query") is False
    )
    reasons: list[str] = []
    if not identity_matches:
        reasons.append("frozen_package_identity_mismatch")
    if not runtime_complete:
        reasons.append("runtime_preflight_incomplete")
    if not boundary_complete:
        reasons.append("forbidden_execution_boundary")
    if not unsupported_explicit:
        reasons.append("unsupported_surface_not_explicit")
    admitted = not reasons
    return {
        "schema_version": "memupdatebench.external.letta.admission.v1",
        "candidate_id": "letta_0_16_8_block_profile",
        "admission_scope": "direct_block_profile_only",
        "identity": _EXPECTED_IDENTITY,
        "preflight_schema_version": preflight.get("schema_version"),
        "gates": {
            "frozen_package_identity": "pass" if identity_matches else "blocked",
            "runtime_preflight": "pass" if runtime_complete else "blocked",
            "no_llm_api_gpu_credentials": "pass" if boundary_complete else "blocked",
            "unsupported_surface": "pass" if unsupported_explicit else "blocked",
        },
        "admitted": admitted,
        "outcome": "pass" if admitted else "blocked",
        "reasons": reasons,
    }


def _read_canonical_json(path: Path) -> dict:
    if not path.is_absolute():
        raise ValueError("Letta preflight path must be absolute")
    assert_no_reparse_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("Letta preflight path must be a real file")
    try:
        value = json.loads(resolved.read_bytes())
    except Exception:
        raise ValueError("Letta preflight is invalid JSON") from None
    if not isinstance(value, dict) or _canonical_object_bytes(value) != resolved.read_bytes():
        raise ValueError("Letta preflight must be canonical JSON")
    return value


def _publish_no_replace(path: Path, receipt: dict) -> None:
    if not path.is_absolute():
        raise ValueError("Letta receipt output must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("Letta receipt output parent must be a real directory")
    if scan_for_secrets(receipt):
        raise ValueError("Letta receipt failed security scan")
    with path.open("xb") as handle:
        handle.write(_canonical_object_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a no-replace Letta block-profile admission receipt.")
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    receipt = build_admission_receipt(_read_canonical_json(Path(arguments.preflight)))
    _publish_no_replace(Path(arguments.output), receipt)
    return 0 if receipt["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
