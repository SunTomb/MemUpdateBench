from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IMMUTABLE_CORE_ROOT = PROJECT_ROOT / "data" / "vnext" / "core" / "v3"

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.letta import build_letta_adapter_configuration, compute_letta_configuration_hash
from mub.vnext.external.security import redact_sensitive_text, scan_for_secrets
from mub.vnext.external.workers.letta_worker import inspect_local_letta_package


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _blocked_capabilities() -> dict[str, bool]:
    return {
        "supports_isolated_reset": False,
        "supports_event_ingest": False,
        "supports_add": False,
        "supports_update": False,
        "supports_noop": False,
        "supports_delete": False,
        "exports_entries": False,
        "exports_values": False,
        "exports_retrieval_ids": False,
        "exports_retrieval_scores": False,
        "supports_native_answer": False,
        "supports_multi_object_query": False,
    }


def run_preflight(*, run_prefix: str) -> dict:
    configuration = build_letta_adapter_configuration(run_id=run_prefix)
    package_preflight = inspect_local_letta_package()
    blocker = package_preflight.get("blocker")
    if type(blocker) is not str or not blocker:
        blocker = "local_source_commit_and_server_bootstrap_unverified"
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
        "configuration_hash": compute_letta_configuration_hash(configuration),
        "package_preflight": package_preflight,
        "capabilities": _blocked_capabilities(),
        "namespace_reset_probe": {
            "attempted": False,
            "passed": False,
            "trials": [],
            "blocker": "real_runtime_not_verified",
        },
        "lifecycle": {
            "attempted": False,
            "passed": False,
            "blocker": "real_runtime_not_verified",
        },
        "unsupported": {
            "passage_memory": True,
            "agent_mode": True,
            "native_answer": True,
            "multi_object_query": True,
            "scoped_delete": True,
            "historical_query": True,
            "version_history_export": True,
            "remote_server": True,
        },
        "execution_boundary": {
            "llm_used": False,
            "api_used": False,
            "gpu_used": False,
            "network_credential_inputs": False,
        },
        "outcome": "blocked",
        "blockers": [blocker],
        "passed": False,
    }


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_evidence(path: Path, payload: dict) -> None:
    if not path.is_absolute():
        raise ValueError("Letta preflight output path must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("Letta preflight output parent must be a real directory")
    if _IMMUTABLE_CORE_ROOT.exists():
        immutable = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
        if _contains(immutable, parent) or _contains(parent, immutable):
            raise ValueError("Letta preflight output must be outside immutable Core")
    if scan_for_secrets(payload):
        raise ValueError("Letta preflight evidence failed security scan")
    with path.open("xb") as handle:
        handle.write(_canonical_object_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a metadata-only, fail-closed Letta 0.16.8 direct-block preflight.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-prefix", default="letta-0-16-8-preflight")
    arguments = parser.parse_args(argv)
    try:
        payload = run_preflight(run_prefix=arguments.run_prefix)
    except Exception as exc:
        payload = {
            "schema_version": "memupdatebench.external.letta.preflight.v1",
            "candidate_id": "letta_0_16_8_block_profile",
            "outcome": "blocked",
            "passed": False,
            "blockers": ["metadata_preflight_failed"],
            "error_type": type(exc).__name__,
            "error_message": redact_sensitive_text(str(exc)),
        }
        if scan_for_secrets(payload):
            payload["error_message"] = "preflight failure details redacted"
    _write_evidence(Path(arguments.output), payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
