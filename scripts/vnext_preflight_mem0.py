from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IMMUTABLE_CORE_ROOT = PROJECT_ROOT / "data" / "vnext" / "core" / "v3"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.contracts.enums import (
    AnswerSchema,
    EvaluationMode,
    EventRole,
)
from mub.vnext.contracts.v3.adapter import (
    ResetRequestV3,
    RetrievalRequestV3,
)
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    MemoryEventV3,
    MemoryQueryV3,
)
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.bridge import JsonlSubprocessBridge
from mub.vnext.external.probe_v3 import run_namespace_reset_probe
from mub.vnext.external.providers.mem0_adapter import Mem0ExternalAdapterV3
from mub.vnext.external.security import (
    build_worker_environment,
    redact_sensitive_text,
    scan_for_secrets,
)
from mub.vnext.external.workers.mem0_worker import (
    load_mem0_worker_configuration,
)


def build_mem0_preflight_worker_environment(
    source_environment: Mapping[str, str],
    *,
    project_root: str | Path,
) -> Mapping[str, str]:
    project = str(Path(project_root).resolve(strict=True))
    source = dict(source_environment)
    source.update(
        {
            "PYTHONPATH": project,
            "PYTHONIOENCODING": "utf-8",
            "MEM0_TELEMETRY": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    allowed_order = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONPATH",
        "PYTHONIOENCODING",
        "MEM0_TELEMETRY",
    )
    allowed = tuple(name for name in allowed_order if name in source)
    required = (
        "PATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "PYTHONPATH",
        "PYTHONIOENCODING",
        "MEM0_TELEMETRY",
    )
    return build_worker_environment(
        source,
        allowed_names=allowed,
        required_names=required,
    )


def build_mem0_worker_command(
    *,
    python_executable: str | Path,
    worker_configuration_path: str | Path,
) -> tuple[str, ...]:
    executable = Path(python_executable)
    configuration = Path(worker_configuration_path)
    if not executable.is_absolute() or not configuration.is_absolute():
        raise ValueError("Mem0 worker command paths must be absolute")
    return (
        str(executable),
        "-m",
        "mub.vnext.external.workers.mem0_worker",
        "--worker-configuration",
        str(configuration),
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_evidence(path: Path, payload: dict) -> None:
    if not path.is_absolute():
        raise ValueError("preflight output path must be absolute")
    assert_no_reparse_components(path)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("preflight output parent must be a real directory")
    if _IMMUTABLE_CORE_ROOT.exists():
        immutable = _IMMUTABLE_CORE_ROOT.resolve(strict=True)
        if _contains(immutable, parent) or _contains(parent, immutable):
            raise ValueError("preflight output must be outside immutable Core")
    if scan_for_secrets(payload):
        raise ValueError("preflight evidence failed security scan")
    raw = _canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _key() -> FrozenMemoryObjectKey:
    return FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
        subkey=None,
    )


def _event() -> MemoryEventV3:
    text = 'Add default|alice|city| with value "Paris".'
    return MemoryEventV3(
        event_id="mem0-preflight-event-1",
        sequence_index=0,
        timestamp="2026-08-11T00:00:00Z",
        raw_text=text,
        normalized_text=text,
        speaker="user",
        role=EventRole.LATEST_GOLD,
    )


def _query() -> MemoryQueryV3:
    return MemoryQueryV3(
        query_id="mem0-preflight-query-1",
        query_type=QueryTypeV3.CURRENT,
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.STATE_DIRECT,
    )


def _integration_passed(
    *,
    reset_result,
    action_result,
    entries_result,
    retrieval_result,
    answer_result,
) -> bool:
    expected_entry_text = _event().raw_text
    if len(entries_result.entries) != 1:
        return False
    entry = entries_result.entries[0]
    retrieved_ids = {
        retrieved.entry_id
        for retrieved in retrieval_result.trace.retrieved_entries
    }
    return (
        reset_result.success
        and action_result.execution_status.value == "executed"
        and entry.content == expected_entry_text
        and entry.object_key_candidate == _key()
        and entry.value_candidate == "Paris"
        and retrieval_result.trace.query_id == _query().query_id
        and entry.entry_id in retrieved_ids
        and answer_result.prediction.parsed_answer == "Paris"
    )


def _validate_timeout_seconds(value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("preflight timeout must be a finite positive float")
    return value


def run_preflight(
    *,
    worker_configuration_path: Path,
    run_prefix: str,
    timeout_seconds: float,
) -> dict:
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)
    worker_configuration = load_mem0_worker_configuration(
        worker_configuration_path
    )
    public_configuration = worker_configuration.public_configuration
    environment = build_mem0_preflight_worker_environment(
        os.environ,
        project_root=PROJECT_ROOT,
    )
    bridge = JsonlSubprocessBridge(
        command=build_mem0_worker_command(
            python_executable=sys.executable,
            worker_configuration_path=worker_configuration_path,
        ),
        cwd=PROJECT_ROOT,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    adapter: Mem0ExternalAdapterV3 | None = None
    try:
        adapter = Mem0ExternalAdapterV3(
            bridge=bridge,
            configuration=public_configuration,
            target_objects=(_key(),),
        )
        reset_probe = run_namespace_reset_probe(
            adapter,
            candidate_id="mem0_oss",
            run_prefix=run_prefix,
        )
        namespace = f"{run_prefix}-integration"
        reset_result = adapter.reset(ResetRequestV3(namespace=namespace))
        action_result = adapter.ingest_event(_event())
        entries_result = adapter.export_entries()
        retrieval_result = adapter.retrieve(
            RetrievalRequestV3(query=_query(), k=5)
        )
        answer_result = adapter.answer(_query(), "slot_direct")
        integration_passed = _integration_passed(
            reset_result=reset_result,
            action_result=action_result,
            entries_result=entries_result,
            retrieval_result=retrieval_result,
            answer_result=answer_result,
        )
        payload = {
            "schema_version": "memupdatebench.external.mem0.preflight.v1",
            "candidate_id": "mem0_oss",
            "run_prefix": run_prefix,
            "configuration_hash": adapter.adapter_info().configuration_hash,
            "adapter_info": adapter.adapter_info().model_dump(mode="json"),
            "capabilities": adapter.capabilities().model_dump(mode="json"),
            "namespace_reset_probe": reset_probe.model_dump(mode="json"),
            "integration": {
                "reset": reset_result.model_dump(mode="json"),
                "action": action_result.model_dump(mode="json"),
                "entries": entries_result.model_dump(mode="json"),
                "retrieval": retrieval_result.model_dump(mode="json"),
                "answer": answer_result.model_dump(mode="json"),
                "passed": integration_passed,
            },
            "passed": reset_probe.passed and integration_passed,
        }
        return payload
    finally:
        if adapter is not None:
            adapter.close()
        else:
            bridge.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated genuine Mem0 OSS Task 10 preflight."
    )
    parser.add_argument("--worker-configuration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run-prefix",
        default="task10-mem0-real-preflight",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = Path(arguments.output)
    try:
        payload = run_preflight(
            worker_configuration_path=Path(arguments.worker_configuration),
            run_prefix=arguments.run_prefix,
            timeout_seconds=arguments.timeout_seconds,
        )
    except Exception as exc:
        message = redact_sensitive_text(str(exc))
        failure = {
            "schema_version": "memupdatebench.external.mem0.preflight.v1",
            "candidate_id": "mem0_oss",
            "run_prefix": arguments.run_prefix,
            "passed": False,
            "error_type": type(exc).__name__,
            "error_message": message,
        }
        if scan_for_secrets(failure):
            failure["error_message"] = "preflight failure details redacted"
        _write_evidence(output, failure)
        return 1
    _write_evidence(output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
