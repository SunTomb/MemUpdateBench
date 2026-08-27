from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import scan_for_secrets

SCHEMA_VERSION = "memupdatebench.external.letta.admission.v2"
_EXPECTED_IDENTITY = {"package_name": "letta", "package_version": "0.16.8", "source_repository": "letta-ai/letta", "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24", "license_id": "Apache-2.0"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def build_admission_receipt(preflight: dict) -> dict:
    if not isinstance(preflight, dict):
        raise ValueError("Letta runtime preflight must be a JSON object")
    identity = preflight.get("identity") == _EXPECTED_IDENTITY
    gates = {
        "source_binding": preflight.get("official_health", {}).get("source_binding") == "verified",
        "official_health": preflight.get("official_health", {}).get("passed") is True,
        "loopback": preflight.get("runtime", {}).get("loopback") is True,
        "database_isolation": preflight.get("runtime", {}).get("database_isolated") is True,
        "namespace_reset": preflight.get("namespace_reset_probe", {}).get("passed") is True,
        "lifecycle": preflight.get("lifecycle", {}).get("passed") is True,
        "clean_close": preflight.get("clean_close", {}).get("passed") is True,
        "security": preflight.get("security", {}).get("secret_scan_passed") is True and preflight.get("security", {}).get("raw_logs_recorded") is False,
        "boundary": all(preflight.get("boundary", {}).get(name) is False for name in ("llm_used", "api_used", "gpu_used", "network_credential_inputs")),
        "profile_boundary": all(preflight.get("unsupported", {}).get(name) is True for name in ("multi_object_query", "native_answer", "historical_query", "version_history_export", "scoped_delete")),
        "preflight_passed": preflight.get("passed") is True and preflight.get("outcome") == "pass",
    }
    reasons = list(preflight.get("blockers", ())) if isinstance(preflight.get("blockers", ()), list) else []
    if preflight.get("schema_version") != "memupdatebench.external.letta.preflight.v2" or preflight.get("candidate_id") != "letta_0_16_8_profile" or preflight.get("mode") != "profile_single_record_runtime":
        reasons.append("runtime_preflight_schema_mismatch")
    if not identity:
        reasons.append("frozen_package_identity_mismatch")
    gate_codes = {"source_binding": "source_binding_unverified", "official_health": "official_health_failed", "loopback": "loopback_required", "database_isolation": "database_isolation_failed", "namespace_reset": "namespace_reset_probe_failed", "lifecycle": "lifecycle_assertions_failed", "clean_close": "clean_close_failed", "security": "security_boundary_failed", "boundary": "forbidden_execution_boundary", "profile_boundary": "profile_scope_not_truthful", "preflight_passed": "runtime_preflight_incomplete"}
    for name, passed in gates.items():
        if not passed:
            reasons.append(gate_codes[name])
    reasons = list(dict.fromkeys(str(reason) for reason in reasons if reason))
    admitted = not reasons
    return {"schema_version": SCHEMA_VERSION, "candidate_id": "letta_0_16_8_profile", "admission_scope": "profile_single_record_runtime", "identity": _EXPECTED_IDENTITY, "preflight_schema_version": preflight.get("schema_version"), "gates": {name: "pass" if passed else "blocked" for name, passed in gates.items()}, "admitted": admitted, "outcome": "pass" if admitted else "blocked", "reasons": reasons}


def _read_json(path: Path) -> dict:
    if not path.is_absolute():
        raise ValueError("Letta runtime preflight path must be absolute")
    assert_no_reparse_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("Letta runtime preflight path must be a real file")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except Exception:
        raise ValueError("Letta runtime preflight is invalid JSON") from None
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise ValueError("Letta runtime preflight must be canonical JSON")
    return value


def _publish(path: Path, receipt: dict) -> None:
    if not path.is_absolute():
        raise ValueError("Letta runtime receipt output must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink() or scan_for_secrets(receipt):
        raise ValueError("Letta runtime receipt output is unsafe")
    with path.open("xb") as handle:
        handle.write(_canonical(receipt)); handle.flush(); os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish additive Letta runtime admission receipt.")
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_admission_receipt(_read_json(Path(args.preflight)))
    _publish(Path(args.output), receipt)
    return 0 if receipt["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
