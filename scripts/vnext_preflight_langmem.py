from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

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
from mub.vnext.external.providers.langmem import (
    LANGMEM_PACKAGE_VERSION,
    LANGMEM_SOURCE_COMMIT,
    build_langmem_adapter_configuration,
)
from mub.vnext.external.providers.langmem_adapter import LangMemExternalAdapterV3
from mub.vnext.external.security import (
    build_worker_environment,
    redact_sensitive_text,
    scan_for_secrets,
)
from mub.vnext.io import canonical_json_bytes


def _canonical_object_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_langmem_preflight_worker_environment(
    source_environment: Mapping[str, str], *, project_root: str | Path
) -> Mapping[str, str]:
    project = str(Path(project_root).resolve(strict=True))
    source = dict(source_environment)
    source.update(
        {
            "PYTHONPATH": project,
            "PYTHONIOENCODING": "utf-8",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_ENDPOINT": "",
            "LANGSMITH_ENDPOINT": "",
        }
    )
    allowed_order = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONIOENCODING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_ENDPOINT",
    )
    allowed = tuple(name for name in allowed_order if name in source)
    return build_worker_environment(
        source,
        allowed_names=allowed,
        required_names=("PATH", "PYTHONPATH", "PYTHONIOENCODING"),
    )


def build_langmem_worker_command(
    *, python_executable: str | Path, configuration_json: str
) -> tuple[str, ...]:
    executable = Path(python_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("LangMem worker executable must be an absolute file")
    if type(configuration_json) is not str or not configuration_json:
        raise ValueError("LangMem worker configuration must be canonical JSON")
    return (
        str(executable),
        "-m",
        "mub.vnext.external.workers.langmem_worker",
        "--configuration-json",
        configuration_json,
    )


def _key() -> FrozenMemoryObjectKey:
    return FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
        subkey=None,
    )


def _event(event_id: str, index: int, text: str) -> MemoryEventV3:
    return MemoryEventV3(
        event_id=event_id,
        sequence_index=index,
        timestamp="2026-08-26T00:00:00Z",
        raw_text=text,
        normalized_text=text,
        speaker="user",
        role=EventRole.LATEST_GOLD,
    )


def _query() -> MemoryQueryV3:
    return MemoryQueryV3(
        query_id="langmem-preflight-query",
        query_type=QueryTypeV3.CURRENT,
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.STATE_DIRECT,
    )


def _validate_timeout_seconds(value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("preflight timeout must be a finite positive float")
    return value


def _run_lifecycle(adapter: LangMemExternalAdapterV3, namespace: str) -> dict:
    reset = adapter.reset(ResetRequestV3(namespace=namespace))
    add = adapter.ingest_event(
        _event(
            "langmem-preflight-add",
            0,
            'Add default|alice|city| with value "Paris".',
        )
    )
    update = adapter.ingest_event(
        _event(
            "langmem-preflight-update",
            1,
            'Update default|alice|city| with value "Lyon".',
        )
    )
    noop = adapter.ingest_event(
        _event("langmem-preflight-noop", 2, "No memory object changes.")
    )
    entries = adapter.export_entries()
    retrieval = adapter.retrieve(RetrievalRequestV3(query=_query(), k=1))
    answer = adapter.answer(_query(), "slot_direct")
    delete = adapter.ingest_event(
        _event(
            "langmem-preflight-delete",
            3,
            "Delete default|alice|city| [scope=object; "
            "enumerated_targets=default|alice|city|; "
            "event_logical_time=2026-08-26T00:00:00Z; "
            "effective_at=2026-08-26T00:00:00Z].",
        )
    )
    after_delete = adapter.export_entries()
    passed = (
        reset.success
        and add.execution_status.value == "executed"
        and update.execution_status.value == "executed"
        and noop.execution_status.value == "executed"
        and delete.execution_status.value == "executed"
        and add.affected_entry_ids == update.affected_entry_ids
        and len(entries.entries) == 1
        and entries.entries[0].value_candidate == "Lyon"
        and len(retrieval.trace.retrieved_entries) == 1
        and answer.prediction.parsed_answer == "Lyon"
        and not after_delete.entries
    )
    return {
        "reset": reset.model_dump(mode="json"),
        "add": add.model_dump(mode="json"),
        "update": update.model_dump(mode="json"),
        "noop": noop.model_dump(mode="json"),
        "entries_before_delete": entries.model_dump(mode="json"),
        "retrieval": retrieval.model_dump(mode="json"),
        "answer": answer.model_dump(mode="json"),
        "delete": delete.model_dump(mode="json"),
        "entries_after_delete": after_delete.model_dump(mode="json"),
        "passed": passed,
    }


def run_preflight(
    *, python_executable: str | Path, run_prefix: str, timeout_seconds: float
) -> dict:
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)
    configuration = build_langmem_adapter_configuration(run_id=run_prefix)
    configuration_json = canonical_json_bytes(configuration).decode("utf-8")
    environment = build_langmem_preflight_worker_environment(
        os.environ, project_root=PROJECT_ROOT
    )
    bridge = JsonlSubprocessBridge(
        command=build_langmem_worker_command(
            python_executable=python_executable,
            configuration_json=configuration_json,
        ),
        cwd=PROJECT_ROOT,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    adapter: LangMemExternalAdapterV3 | None = None
    try:
        adapter = LangMemExternalAdapterV3(
            bridge=bridge,
            configuration=configuration,
            target_objects=(_key(),),
        )
        reset_probe = run_namespace_reset_probe(
            adapter,
            candidate_id="langmem_0_0_30_profile",
            run_prefix=run_prefix,
        )
        lifecycle = _run_lifecycle(adapter, f"{run_prefix}-lifecycle")
        payload = {
            "schema_version": "memupdatebench.external.langmem.preflight.v1",
            "candidate_id": "langmem_0_0_30_profile",
            "mode": "profile_single_record",
            "identity": {
                "package_name": "langmem",
                "package_version": LANGMEM_PACKAGE_VERSION,
                "source_repository": "langchain-ai/langmem",
                "source_commit": LANGMEM_SOURCE_COMMIT,
                "license_id": "MIT",
            },
            "adapter_info": adapter.adapter_info().model_dump(mode="json"),
            "capabilities": adapter.capabilities().model_dump(mode="json"),
            "namespace_reset_probe": reset_probe.model_dump(mode="json"),
            "lifecycle": lifecycle,
            "unsupported": {
                "collection_mode": True,
                "multi_object_query": True,
                "scoped_delete": True,
                "historical_query": True,
                "native_answer": True,
                "llm_enrichment": True,
                "semantic_embedding_search": True,
            },
            "execution_boundary": {
                "llm_used": False,
                "api_used": False,
                "gpu_used": False,
                "network_credential_inputs": False,
                "search_policy": "native_inmemory_then_deterministic_local",
            },
            "outcome": "pass" if reset_probe.passed and lifecycle["passed"] else "blocked",
            "blockers": [] if reset_probe.passed and lifecycle["passed"] else ["preflight_check_failed"],
            "passed": reset_probe.passed and lifecycle["passed"],
        }
        if scan_for_secrets(payload):
            raise ValueError("LangMem preflight evidence failed security scan")
        return payload
    finally:
        if adapter is not None:
            adapter.close()
        else:
            bridge.close()


def _write_evidence(path: Path, payload: dict) -> None:
    if not path.is_absolute():
        raise ValueError("preflight output path must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("preflight output parent must be a real directory")
    if scan_for_secrets(payload):
        raise ValueError("preflight evidence failed security scan")
    raw = _canonical_object_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated LangMem 0.0.30 profile admission preflight."
    )
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-prefix", default="langmem-0-0-30-preflight")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output)
    try:
        payload = run_preflight(
            python_executable=arguments.python_executable,
            run_prefix=arguments.run_prefix,
            timeout_seconds=arguments.timeout_seconds,
        )
    except Exception as exc:
        payload = {
            "schema_version": "memupdatebench.external.langmem.preflight.v1",
            "candidate_id": "langmem_0_0_30_profile",
            "outcome": "blocked",
            "passed": False,
            "blockers": ["identity_or_runtime_preflight_failed"],
            "error_type": type(exc).__name__,
            "error_message": redact_sensitive_text(str(exc)),
        }
        if scan_for_secrets(payload):
            payload["error_message"] = "preflight failure details redacted"
    _write_evidence(output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
