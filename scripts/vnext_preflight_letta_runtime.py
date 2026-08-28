from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.contracts.enums import AnswerSchema, EvaluationMode, EventRole
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryEventV3, MemoryQueryV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.bridge import JsonlSubprocessBridge
from mub.vnext.external.probe_v3 import run_namespace_reset_probe
from mub.vnext.external.providers.letta import build_letta_adapter_configuration, compute_letta_configuration_hash
from mub.vnext.external.providers.letta_adapter import LettaExternalAdapterV3
from mub.vnext.external.security import build_worker_environment, redact_sensitive_text, scan_for_secrets
from mub.vnext.external.workers.letta_rest import (
    LETTA_NATIVE_API_BASE_URL_ENV,
    LettaRestDependencyUnavailable,
    _valid_loopback_base_url,
)
from mub.vnext.io import canonical_json_bytes

SCHEMA_VERSION = "memupdatebench.external.letta.preflight.v2"
CANDIDATE_ID = "letta_0_16_8_profile"


def _key() -> FrozenMemoryObjectKey:
    return FrozenMemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city", subkey=None)


def _event(event_id: str, index: int, text: str) -> MemoryEventV3:
    return MemoryEventV3(event_id=event_id, sequence_index=index, timestamp="2026-08-27T00:00:00Z", raw_text=text, normalized_text=text, speaker="user", role=EventRole.LATEST_GOLD)


def _query() -> MemoryQueryV3:
    return MemoryQueryV3(query_id="letta-runtime-preflight-query", query_type=QueryTypeV3.CURRENT, text="Where does Alice live?", selector=CurrentSelector(), target_object_keys=(_key(),), answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.STATE_DIRECT)


def _validate_timeout(value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("preflight timeout must be a finite positive float")
    return value


def build_letta_worker_command(*, python_executable: str | Path, worker_command: tuple[str, ...], configuration_json: str) -> tuple[str, ...]:
    executable = Path(python_executable)
    if not executable.is_absolute() or not str(executable):
        raise ValueError("Letta worker executable must be absolute")
    if type(worker_command) is not tuple or not worker_command or any(type(part) is not str or not part for part in worker_command):
        raise ValueError("Letta worker command must be a nonempty tuple")
    if type(configuration_json) is not str or not configuration_json:
        raise ValueError("Letta worker configuration must be canonical JSON")
    command = (str(executable), *worker_command, "--configuration-json", configuration_json)
    if scan_for_secrets(command):
        raise ValueError("Letta worker command security scan failed")
    return command


def build_letta_preflight_worker_environment(source_environment: Mapping[str, str], *, project_root: str | Path) -> Mapping[str, str]:
    project = str(Path(project_root).resolve(strict=True))
    source = dict(source_environment)
    source.update({"PYTHONPATH": project, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"})
    if LETTA_NATIVE_API_BASE_URL_ENV in source:
        try:
            source[LETTA_NATIVE_API_BASE_URL_ENV] = _valid_loopback_base_url(
                source[LETTA_NATIVE_API_BASE_URL_ENV]
            )
        except LettaRestDependencyUnavailable as exc:
            raise ValueError(
                "LETTA_NATIVE_API_BASE_URL must be a loopback HTTP(S) URL without credentials, query, or fragment"
            ) from exc
    allowed_order = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONIOENCODING", "HF_HUB_OFFLINE", LETTA_NATIVE_API_BASE_URL_ENV)
    return build_worker_environment(source, allowed_names=tuple(name for name in allowed_order if name in source), required_names=("PATH", "PYTHONPATH", "PYTHONIOENCODING"))


def _run_lifecycle(adapter, namespace: str) -> dict:
    reset = adapter.reset(ResetRequestV3(namespace=namespace))
    add = adapter.ingest_event(_event("add", 0, 'Add default|alice|city| with value "Paris".'))
    update = adapter.ingest_event(_event("update", 1, 'Update default|alice|city| with value "Lyon".'))
    noop = adapter.ingest_event(_event("noop", 2, "No memory object changes."))
    before = adapter.export_entries()
    retrieval = adapter.retrieve(RetrievalRequestV3(query=_query(), k=1))
    answer = adapter.answer(_query(), "slot_direct")
    delete = adapter.ingest_event(_event("delete", 3, "Delete default|alice|city| [scope=object; enumerated_targets=default|alice|city|; event_logical_time=2026-08-27T00:00:00Z; effective_at=2026-08-27T00:00:00Z]."))
    after = adapter.export_entries()
    stable_id = add.affected_entry_ids[0] if add.affected_entry_ids else None
    passed = bool(reset.success and add.execution_status.value == "executed" and update.execution_status.value == "executed" and noop.execution_status.value == "executed" and delete.execution_status.value == "executed" and stable_id and add.affected_entry_ids == update.affected_entry_ids == delete.affected_entry_ids and len(before.entries) == 1 and before.entries[0].value_candidate == "Lyon" and retrieval.trace.retrieved_entries and retrieval.trace.retrieved_entries[0].entry_id == stable_id and answer.prediction.parsed_answer == "Lyon" and after.entries == ())
    return {"status": "PASS" if passed else "BLOCKED", "reset": reset.model_dump(mode="json"), "add": add.model_dump(mode="json"), "update": update.model_dump(mode="json"), "noop": noop.model_dump(mode="json"), "export_latest": before.model_dump(mode="json"), "retrieval": retrieval.model_dump(mode="json"), "slot_direct": answer.model_dump(mode="json") | {"passed": answer.prediction.parsed_answer == "Lyon"}, "delete": delete.model_dump(mode="json"), "export_after_delete": after.model_dump(mode="json"), "export_after_delete_empty": not after.entries, "stable_entry_id": stable_id, "passed": passed}


def _not_run(blocker: str) -> dict:
    return {"status": "NOT_RUN", "passed": False, "blocker": blocker}


def run_preflight(*, python_executable: str | Path, worker_command: tuple[str, ...], server_identity: str, database_identity: str, run_prefix: str, timeout_seconds: float = 30.0, database_isolation_verified: bool = False, bridge_factory: Callable[..., object] | None = None, adapter_factory: Callable[..., object] | None = None) -> dict:
    timeout_seconds = _validate_timeout(timeout_seconds)
    if type(server_identity) is not str or not server_identity.strip() or type(database_identity) is not str or not database_identity.strip():
        raise ValueError("Letta runtime identities must be nonblank strings")
    if type(database_isolation_verified) is not bool:
        raise ValueError("database isolation verification must be an exact boolean")
    configuration = build_letta_adapter_configuration(run_id=run_prefix)
    config_json = canonical_json_bytes(configuration).decode("utf-8")
    blockers: list[str] = []
    bridge = None
    adapter = None
    reset_probe = _not_run("runtime_not_started")
    lifecycle = _not_run("runtime_not_started")
    close = {"status": "NOT_RUN", "passed": False}
    try:
        environment = build_letta_preflight_worker_environment(os.environ, project_root=PROJECT_ROOT)
        command = build_letta_worker_command(python_executable=python_executable, worker_command=worker_command, configuration_json=config_json)
        factory = bridge_factory or (lambda **kwargs: JsonlSubprocessBridge(**kwargs))
        bridge = factory(command=command, cwd=PROJECT_ROOT, environment=environment, timeout_seconds=timeout_seconds)
        make_adapter = adapter_factory or (lambda **kwargs: LettaExternalAdapterV3(**kwargs))
        adapter = make_adapter(bridge=bridge, configuration=configuration, target_objects=(_key(),))
        reset_probe_model = run_namespace_reset_probe(adapter, candidate_id=CANDIDATE_ID, run_prefix=run_prefix)
        reset_probe = reset_probe_model.model_dump(mode="json")
        lifecycle = _run_lifecycle(adapter, f"{run_prefix}-lifecycle")
        if not reset_probe_model.passed:
            blockers.append("namespace_reset_probe_failed")
        if not lifecycle["passed"]:
            blockers.append("lifecycle_assertions_failed")
        close = {"status": "PASS", "passed": True}
    except Exception as exc:
        blockers.append("runtime_bridge_unavailable" if bridge is None else "runtime_preflight_failed")
        if not isinstance(exc, (ValueError, TypeError)):
            blockers.append("official_health_or_runtime_unavailable")
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                close = {"status": "BLOCKED", "passed": False, "blocker": "clean_close_failed"}
                blockers.append("clean_close_failed")
        elif bridge is not None:
            bridge.close()
    official_runtime = type(adapter) is LettaExternalAdapterV3
    if adapter is not None and not official_runtime:
        blockers.append("official_adapter_boundary_unverified")
    blockers = list(dict.fromkeys(blockers))
    passed = not blockers and official_runtime and reset_probe.get("passed") is True and lifecycle.get("passed") is True and close.get("passed") is True
    payload = {"schema_version": SCHEMA_VERSION, "candidate_id": CANDIDATE_ID, "mode": "profile_single_record_runtime", "identity": {"package_name": "letta", "package_version": "0.16.8", "source_repository": "letta-ai/letta", "source_commit": "1131535716e8a31c9a437f8695e25ac98f203a24", "license_id": "Apache-2.0"}, "configuration_hash": compute_letta_configuration_hash(configuration), "runtime": {"server_identity": server_identity, "database_identity": database_identity, "loopback": server_identity.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")), "database_isolated": database_isolation_verified}, "official_health": {"passed": official_runtime, "source_binding": "verified" if official_runtime else "blocked"}, "namespace_reset_probe": reset_probe, "lifecycle": lifecycle, "clean_close": close, "security": {"secret_scan_passed": True, "raw_logs_recorded": False}, "boundary": {"llm_used": False, "api_used": False, "gpu_used": False, "network_credential_inputs": False}, "unsupported": {"multi_object_query": True, "native_answer": True, "historical_query": True, "version_history_export": True, "scoped_delete": True}, "metrics": None, "outcome": "pass" if passed else "blocked", "blockers": blockers, "passed": passed}
    if scan_for_secrets(payload):
        raise ValueError("Letta runtime preflight evidence failed security scan")
    return payload


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _write_evidence(path: Path, payload: dict) -> None:
    if not path.is_absolute():
        raise ValueError("Letta runtime preflight output path must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink() or scan_for_secrets(payload):
        raise ValueError("Letta runtime preflight output is unsafe")
    with path.open("xb") as handle:
        handle.write(_canonical_object_bytes(payload)); handle.flush(); os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run additive Letta runtime preflight.")
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--worker-command", nargs="+", required=True)
    parser.add_argument("--server-identity", required=True)
    parser.add_argument("--database-identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-prefix", default="letta-runtime-preflight")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--database-isolation-verified", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_preflight(python_executable=args.python_executable, worker_command=tuple(args.worker_command), server_identity=args.server_identity, database_identity=args.database_identity, run_prefix=args.run_prefix, timeout_seconds=args.timeout_seconds, database_isolation_verified=args.database_isolation_verified)
    except Exception as exc:
        payload = {"schema_version": SCHEMA_VERSION, "candidate_id": CANDIDATE_ID, "outcome": "blocked", "passed": False, "blockers": ["preflight_failed"], "metrics": None, "error_type": type(exc).__name__, "error_message": redact_sensitive_text(str(exc))}
    _write_evidence(Path(args.output), payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
