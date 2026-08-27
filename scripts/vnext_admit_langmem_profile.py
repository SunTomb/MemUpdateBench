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


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


_EXPECTED_IDENTITY = {
    "package_name": "langmem",
    "package_version": "0.0.30",
    "source_repository": "langchain-ai/langmem",
    "source_commit": "29cbe41e58528f92e9efa773c12e15c47be3808c",
    "license_id": "MIT",
}


def build_admission_receipt(preflight: dict) -> dict:
    if not isinstance(preflight, dict):
        raise ValueError("LangMem preflight must be a JSON object")
    identity = preflight.get("identity")
    identity_matches = identity == _EXPECTED_IDENTITY
    runtime_complete = (
        preflight.get("schema_version")
        == "memupdatebench.external.langmem.preflight.v1"
        and preflight.get("candidate_id") == "langmem_0_0_30_profile"
        and preflight.get("mode") == "profile_single_record"
        and preflight.get("passed") is True
        and preflight.get("outcome") == "pass"
        and preflight.get("namespace_reset_probe", {}).get("passed") is True
        and preflight.get("lifecycle", {}).get("passed") is True
    )
    boundary = preflight.get("execution_boundary", {})
    boundary_complete = (
        boundary.get("llm_used") is False
        and boundary.get("api_used") is False
        and boundary.get("gpu_used") is False
        and boundary.get("network_credential_inputs") is False
    )
    profile_boundary = (
        preflight.get("unsupported", {}).get("collection_mode") is True
        and preflight.get("capabilities", {}).get("supports_multi_object_query")
        is False
    )
    reasons: list[str] = []
    if not identity_matches:
        reasons.append("frozen_package_identity_mismatch")
    if not runtime_complete:
        reasons.append("runtime_preflight_incomplete")
    if not boundary_complete:
        reasons.append("forbidden_execution_boundary")
    if not profile_boundary:
        reasons.append("profile_scope_not_truthful")
    admitted = not reasons
    return {
        "schema_version": "memupdatebench.external.langmem.admission.v1",
        "candidate_id": "langmem_0_0_30_profile",
        "admission_scope": "profile_single_record_only",
        "identity": _EXPECTED_IDENTITY,
        "preflight_schema_version": preflight.get("schema_version"),
        "gates": {
            "frozen_package_identity": "pass" if identity_matches else "blocked",
            "runtime_preflight": "pass" if runtime_complete else "blocked",
            "no_llm_api_gpu_credentials": "pass" if boundary_complete else "blocked",
            "profile_scope_truthfulness": "pass" if profile_boundary else "blocked",
        },
        "admitted": admitted,
        "outcome": "pass" if admitted else "blocked",
        "reasons": reasons,
    }


def _read_canonical_json(path: Path) -> dict:
    if not path.is_absolute():
        raise ValueError("LangMem preflight path must be absolute")
    assert_no_reparse_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("LangMem preflight path must be a real file")
    raw = resolved.read_bytes()
    try:
        import json

        value = json.loads(raw)
    except Exception:
        raise ValueError("LangMem preflight is invalid JSON") from None
    if _canonical_object_bytes(value) != raw:
        raise ValueError("LangMem preflight must be canonical JSON")
    if not isinstance(value, dict):
        raise ValueError("LangMem preflight must be a JSON object")
    return value


def _publish_no_replace(path: Path, receipt: dict) -> None:
    if not path.is_absolute():
        raise ValueError("LangMem receipt output must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("LangMem receipt output parent must be a real directory")
    if scan_for_secrets(receipt):
        raise ValueError("LangMem receipt failed security scan")
    raw = _canonical_object_bytes(receipt)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a no-replace LangMem profile admission receipt."
    )
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    receipt = build_admission_receipt(_read_canonical_json(Path(arguments.preflight)))
    _publish_no_replace(Path(arguments.output), receipt)
    return 0 if receipt["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
