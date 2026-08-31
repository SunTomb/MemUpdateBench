from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import redact_sensitive_text, scan_for_secrets
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.external.providers.langmem import (
    LANGMEM_PACKAGE_VERSION,
    LANGMEM_SOURCE_COMMIT,
    build_langmem_adapter_configuration,
    compute_langmem_configuration_hash,
)
from scripts import vnext_plan_main_track_factorial as planner

QUALIFICATION_SCHEMA_VERSION = (
    "memupdatebench.external.langgraph-store-custom-adapter.qualification.v1"
)
QUALIFICATION_INDEX_SCHEMA_VERSION = f"{QUALIFICATION_SCHEMA_VERSION}.artifact-index"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "vnext"
    / "main_track_langgraph_custom_qualification_v1"
)
_LANGGRAPH_MANAGER_ID = "langgraph_store_custom_adapter"
_LANGGRAPH_ADAPTER_VERSION = "memupdatebench-langmem-adapter-v1"
_LANGGRAPH_SYSTEM_NAME = "langgraph_in_memory_store"
_LANGGRAPH_SYSTEM_VERSION = "0.0.30"
_LANGGRAPH_BACKEND = "langgraph_in_memory_store"
_LANGGRAPH_SOURCE_FILES = (
    "mub/vnext/external/providers/langmem.py",
    "mub/vnext/external/providers/langmem_adapter.py",
    "mub/vnext/external/providers/langmem_protocol.py",
    "mub/vnext/external/workers/langmem_worker.py",
    "scripts/vnext_run_main_track_factorial_manager.py",
)
_EXECUTION_BOUNDARY = {
    "provider_calls": 0,
    "model_loads": 0,
    "database_accesses": 0,
    "network_calls": 0,
    "gpu_calls": 0,
    "executable_calls": 0,
    "remote_operations": 0,
}
_CONFIG_HASH = compute_langmem_configuration_hash(
    build_langmem_adapter_configuration(run_id="langgraph-production-manifest")
)


def _canonical_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _runtime_identity_for_executable(executable: Path) -> dict[str, Any]:
    candidate = Path(executable)
    if not candidate.is_absolute():
        raise ValueError("LangGraph qualification Python executable must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("LangGraph qualification Python executable is unavailable") from exc
    if not resolved.is_file():
        raise ValueError("LangGraph qualification Python executable must be a regular file")
    current = Path(sys.executable).resolve(strict=False)
    same_interpreter = resolved == current
    return {
        "python_executable": str(resolved),
        "executable_sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "python_version": sys.version if same_interpreter else None,
        "implementation": platform.python_implementation() if same_interpreter else None,
        "platform": platform.platform() if same_interpreter else None,
        "packages": None,
    }


def _metadata_files_sha256(distribution: importlib.metadata.Distribution) -> str:
    files = distribution.files
    if files is None:
        raise RuntimeError("installed distribution does not expose file metadata")
    records: list[tuple[str, bytes]] = []
    for relative in files:
        relative_path = Path(str(relative))
        target = Path(distribution.locate_file(relative))
        if not target.is_file():
            raise RuntimeError(
                f"installed distribution file is unavailable: {relative_path.as_posix()}"
            )
        records.append((relative_path.as_posix(), target.read_bytes()))
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for relative, content in records:
        relative_raw = relative.encode("utf-8")
        digest.update(len(relative_raw).to_bytes(8, "big"))
        digest.update(relative_raw)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _child_distribution_binding(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    metadata_path = Path(distribution._path).resolve(strict=True)
    if not metadata_path.is_dir():
        raise RuntimeError(f"installed distribution metadata is unavailable: {name}")
    return {
        "status": "AVAILABLE",
        "version": distribution.version,
        "metadata_path": str(metadata_path),
        "metadata_files_sha256": _metadata_files_sha256(distribution),
    }


def _child_runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise RuntimeError("child Python executable is not a regular file")
    return {
        "python_executable": str(executable),
        "executable_sha256": _sha256_file(executable),
        "size_bytes": executable.stat().st_size,
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _child_distribution_binding(name)
            for name in ("langmem", "langgraph")
        },
    }


def _validate_runtime_identity(
    value: Any, *, ready: bool, label: str = "LangGraph qualification runtime"
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity is missing")
    required = (
        "python_executable",
        "executable_sha256",
        "size_bytes",
        "python_version",
        "implementation",
        "platform",
        "packages",
    )
    if any(key not in value for key in required):
        raise ValueError(f"{label} identity is incomplete")
    executable = value["python_executable"]
    executable_hash = value["executable_sha256"]
    size_bytes = value["size_bytes"]
    if type(executable) is not str or not executable or not Path(executable).is_absolute():
        raise ValueError(f"{label} executable path is invalid")
    if type(executable_hash) is not str or len(executable_hash) != 64 or any(
        char not in "0123456789abcdef" for char in executable_hash
    ):
        raise ValueError(f"{label} executable_sha256 is invalid")
    if type(size_bytes) is not int or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise ValueError(f"{label} size_bytes is invalid")
    for field in ("python_version", "implementation", "platform"):
        if ready and (type(value[field]) is not str or not value[field]):
            raise ValueError(f"{label} {field} is missing")
    packages = value["packages"]
    if ready:
        if not isinstance(packages, Mapping) or set(packages) != {"langmem", "langgraph"}:
            raise ValueError(f"{label} package distributions are incomplete")
        for package_name in ("langmem", "langgraph"):
            package = packages[package_name]
            if not isinstance(package, Mapping):
                raise ValueError(f"{label} package distribution is invalid")
            if package.get("status") != "AVAILABLE":
                raise ValueError(f"{label} package distribution is unavailable")
            if type(package.get("version")) is not str or not package["version"]:
                raise ValueError(f"{label} package distribution version is missing")
            metadata_path = package.get("metadata_path")
            if type(metadata_path) is not str or not metadata_path or not Path(metadata_path).is_absolute():
                raise ValueError(f"{label} package metadata path is invalid")
            metadata_hash = package.get("metadata_files_sha256")
            if type(metadata_hash) is not str or len(metadata_hash) != 64 or any(
                char not in "0123456789abcdef" for char in metadata_hash
            ):
                raise ValueError(f"{label} package distribution hash is invalid")
    elif packages is not None and not isinstance(packages, Mapping):
        raise ValueError(f"{label} package distributions are invalid")
    return dict(value)


def _source_bindings(project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for relative in _LANGGRAPH_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"LangGraph adapter source file is unavailable: {relative}")
        result.append({"path": relative, "sha256": _sha256_file(path)})
    return result


def _base_payload(
    *,
    packages: Mapping[str, Mapping[str, Any]],
    runtime_identity: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, str]],
    lifecycle: Mapping[str, Any],
    configuration_hash: str,
    qualification_status: str,
    outcome: str,
    blockers: Sequence[str],
    reason: str | None,
) -> dict[str, Any]:
    if qualification_status not in {"READY", "BLOCKED"}:
        raise ValueError("qualification_status must be READY or BLOCKED")
    if outcome not in {"PASS", "BLOCKED"}:
        raise ValueError("qualification outcome must be PASS or BLOCKED")
    payload: dict[str, Any] = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "qualification_status": qualification_status,
        "outcome": outcome,
        "evidence_class": "capability_runtime_qualification",
        "scientific_evidence": False,
        "manager_id": _LANGGRAPH_MANAGER_ID,
        "adapter_id": _LANGGRAPH_MANAGER_ID,
        "adapter_version": _LANGGRAPH_ADAPTER_VERSION,
        "system_name": _LANGGRAPH_SYSTEM_NAME,
        "system_version": _LANGGRAPH_SYSTEM_VERSION,
        "backend": _LANGGRAPH_BACKEND,
        "implementation_boundary": planner.LANGGRAPH_IMPLEMENTATION_BOUNDARY,
        "package_identity": {
            "langmem_source_repository": "langchain-ai/langmem",
            "langmem_source_commit": LANGMEM_SOURCE_COMMIT,
            "langmem_license_id": "MIT",
        },
        "packages": {name: dict(value) for name, value in sorted(packages.items())},
        "runtime_identity": dict(runtime_identity),
        "adapter_source_files": [dict(item) for item in source_bindings],
        "official_backend": {
            "class": "OfficialLangMemBackendV1",
            "module": "mub.vnext.external.workers.langmem_worker",
            "configuration_hash": configuration_hash,
        },
        "configuration_hash": configuration_hash,
        "store_class": "langgraph.store.memory.InMemoryStore",
        "capabilities": dict(planner._PROFILE_CAPABILITIES),
        "execution_boundary": dict(_EXECUTION_BOUNDARY),
        "lifecycle": dict(lifecycle),
        "preflight": {
            "command": f'"{runtime_identity["python_executable"]}" scripts/vnext_qualify_langgraph_custom.py',
            "outcome": outcome,
        },
        "blockers": list(blockers),
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def _with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["qualification_hash"] = _sha256_bytes(
        _canonical_json_bytes({key: value for key, value in result.items() if key != "qualification_hash"})
    )
    return result


def build_ready_payload(
    probe: Mapping[str, Any], *, source_bindings: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    if not isinstance(probe, Mapping):
        raise ValueError("LangGraph qualification probe must be an object")
    runtime_identity = _validate_runtime_identity(
        probe.get("runtime_identity"), ready=True
    )
    packages = runtime_identity["packages"]
    if not isinstance(packages, Mapping):
        raise ValueError("LangGraph qualification package distributions are missing")
    lifecycle = probe.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("LangGraph qualification probe is incomplete")
    for package_name in ("langmem", "langgraph"):
        package = packages.get(package_name)
        if not isinstance(package, Mapping) or package.get("status") != "AVAILABLE":
            raise ValueError("LangGraph qualification packages are unavailable")
        if type(package.get("version")) is not str or not package["version"]:
            raise ValueError("LangGraph qualification package version is missing")
        if package_name == "langmem" and package["version"] != LANGMEM_PACKAGE_VERSION:
            raise ValueError("LangMem qualification package version is invalid")
    if probe.get("store_class") != "langgraph.store.memory.InMemoryStore":
        raise ValueError("LangGraph qualification store identity is invalid")
    if probe.get("configuration_hash") != _CONFIG_HASH:
        raise ValueError("LangGraph qualification configuration hash is invalid")
    if lifecycle.get("passed") is not True:
        raise ValueError("LangGraph qualification lifecycle did not pass")
    payload = _base_payload(
        packages=packages,
        runtime_identity=runtime_identity,
        source_bindings=source_bindings,
        lifecycle=lifecycle,
        configuration_hash=_CONFIG_HASH,
        qualification_status="READY",
        outcome="PASS",
        blockers=(),
        reason=None,
    )
    return _with_hash(payload)


def build_blocked_payload(
    *,
    reason_code: str,
    reason: str,
    source_bindings: Sequence[Mapping[str, str]],
    runtime_identity: Mapping[str, Any],
    packages: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if type(reason_code) is not str or not reason_code:
        raise ValueError("blocked qualification reason_code is required")
    if type(reason) is not str or not reason:
        raise ValueError("blocked qualification reason is required")
    requested_runtime_identity = _validate_runtime_identity(
        runtime_identity, ready=False
    )
    observed_packages = packages or {
        "langmem": {"status": "UNAVAILABLE", "version": None},
        "langgraph": {"status": "UNAVAILABLE", "version": None},
    }
    payload = _base_payload(
        packages=observed_packages,
        runtime_identity=requested_runtime_identity,
        source_bindings=source_bindings,
        lifecycle={"passed": False, "operations": []},
        configuration_hash=_CONFIG_HASH,
        qualification_status="BLOCKED",
        outcome="BLOCKED",
        blockers=(reason_code,),
        reason=reason,
    )
    return _with_hash(payload)


def _child_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONIOENCODING": "utf-8",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_ENDPOINT": "",
            "LANGSMITH_ENDPOINT": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _run_child_probe(
    *,
    python_executable: str | Path,
    timeout_seconds: float = 30.0,
    expected_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executable = Path(python_executable)
    requested_identity = dict(
        expected_runtime_identity or _runtime_identity_for_executable(executable)
    )
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("LangGraph qualification Python executable must be absolute")
    expected_json = json.dumps(
        requested_identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    code = (
        "import json; "
        "from scripts.vnext_qualify_langgraph_custom import run_local_probe; "
        f"expected = json.loads({expected_json!r}); "
        "print('MUB_LANGGRAPH_QUALIFICATION=' + json.dumps(run_local_probe("
        "expected_runtime_identity=expected), "
        "ensure_ascii=True, sort_keys=True, separators=(',', ':')))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", code],
            cwd=PROJECT_ROOT,
            env=_child_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("LangGraph local qualification probe could not run") from exc
    marker = "MUB_LANGGRAPH_QUALIFICATION="
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            try:
                value = json.loads(line[len(marker) :])
            except json.JSONDecodeError as exc:
                raise RuntimeError("LangGraph local qualification probe returned invalid JSON") from exc
            if completed.returncode != 0 or not isinstance(value, dict):
                raise RuntimeError("LangGraph local qualification probe failed")
            runtime_identity = value.get("runtime_identity")
            if not isinstance(runtime_identity, Mapping):
                raise RuntimeError("LangGraph local qualification probe omitted runtime identity")
            for field in ("python_executable", "executable_sha256", "size_bytes"):
                if runtime_identity.get(field) != requested_identity.get(field):
                    raise RuntimeError(
                        "LangGraph local qualification probe executable identity mismatch"
                    )
            return value
    raise RuntimeError("LangGraph local qualification probe returned no result")


def run_local_probe(
    *, expected_runtime_identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Run only local imports and the in-memory adapter lifecycle in the child Python."""
    runtime_identity = _child_runtime_identity()
    if expected_runtime_identity is not None:
        for field in ("python_executable", "executable_sha256", "size_bytes"):
            if runtime_identity.get(field) != expected_runtime_identity.get(field):
                raise RuntimeError(
                    "child executable identity does not match requested executable"
                )
    import importlib.metadata

    from langgraph.store.memory import InMemoryStore

    from mub.vnext.contracts.enums import AnswerSchema, EvaluationMode, EventRole
    from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3
    from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
    from mub.vnext.contracts.v3.enums import QueryTypeV3
    from mub.vnext.contracts.v3.task import CurrentSelector, MemoryEventV3, MemoryQueryV3
    from mub.vnext.external.providers.langmem import build_langmem_adapter_configuration
    from mub.vnext.external.providers.langmem_adapter import LangMemExternalAdapterV3
    from mub.vnext.external.workers.langmem_worker import (
        LangMemWorkerServiceV1,
        OfficialLangMemBackendV1,
    )

    class Bridge:
        def __init__(self, service: LangMemWorkerServiceV1) -> None:
            self.service = service
            self.closed = False

        def request(self, request: Any) -> Any:
            if self.closed:
                raise RuntimeError("qualification bridge is closed")
            return self.service.handle(request)

        def close(self) -> None:
            self.closed = True

    configuration = build_langmem_adapter_configuration(
        run_id="langgraph-production-manifest"
    )
    backend = OfficialLangMemBackendV1(configuration)
    if not isinstance(backend._store, InMemoryStore):
        raise RuntimeError("OfficialLangMemBackendV1 did not use InMemoryStore")
    bridge = Bridge(LangMemWorkerServiceV1(backend))
    key = FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
        subkey=None,
    )
    adapter = LangMemExternalAdapterV3(
        bridge=bridge, configuration=configuration, target_objects=(key,)
    )
    try:
        namespace = "langgraph-qualification-lifecycle"
        reset = adapter.reset(ResetRequestV3(namespace=namespace))
        def event(event_id: str, sequence_index: int, text: str) -> MemoryEventV3:
            return MemoryEventV3(
                event_id=event_id,
                sequence_index=sequence_index,
                timestamp="2026-08-30T00:00:00Z",
                raw_text=text,
                normalized_text=text,
                speaker="user",
                role=EventRole.LATEST_GOLD,
            )

        add = adapter.ingest_event(
            event("qualification-add", 0, 'Add default|alice|city| with value "Paris".')
        )
        update = adapter.ingest_event(
            event("qualification-update", 1, 'Update default|alice|city| with value "Lyon".')
        )
        noop = adapter.ingest_event(event("qualification-noop", 2, "No memory object changes."))
        query = MemoryQueryV3(
            query_id="langgraph-qualification-query",
            query_type=QueryTypeV3.CURRENT,
            text="Where does Alice live?",
            selector=CurrentSelector(),
            target_object_keys=(key,),
            answer_schema=AnswerSchema.STRING,
            evaluation_mode=EvaluationMode.STATE_DIRECT,
        )
        retrieved = adapter.retrieve(RetrievalRequestV3(query=query, k=1))
        exported = adapter.export_entries()
        answer = adapter.answer(query, "slot_direct")
        reset_after = adapter.reset(ResetRequestV3(namespace=namespace))
        lifecycle = {
            "passed": (
                reset.success
                and add.execution_status.value == "executed"
                and update.execution_status.value == "executed"
                and noop.execution_status.value == "executed"
                and add.affected_entry_ids == update.affected_entry_ids
                and len(exported.entries) == 1
                and exported.entries[0].value_candidate == "Lyon"
                and len(retrieved.trace.retrieved_entries) == 1
                and answer.prediction.parsed_answer == "Lyon"
                and reset_after.success
                and not adapter.export_entries().entries
            ),
            "operations": ["reset", "add", "update", "noop", "retrieve", "export", "reset"],
        }
    finally:
        adapter.close()
    return {
        "runtime_identity": runtime_identity,
        "packages": runtime_identity["packages"],
        "store_class": "langgraph.store.memory.InMemoryStore",
        "configuration_hash": compute_langmem_configuration_hash(configuration),
        "lifecycle": lifecycle,
    }


def qualify(
    *, python_executable: str | Path, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    source_bindings = _source_bindings()
    requested_runtime_identity = _runtime_identity_for_executable(Path(python_executable))
    try:
        probe = _run_child_probe(
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            expected_runtime_identity=requested_runtime_identity,
        )
        return build_ready_payload(probe, source_bindings=source_bindings)
    except Exception as exc:
        return build_blocked_payload(
            reason_code="langgraph_dependency_or_runtime_unavailable",
            reason=redact_sensitive_text(type(exc).__name__),
            source_bindings=source_bindings,
            runtime_identity=requested_runtime_identity,
        )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if scan_for_secrets(payload):
        raise ValueError("LangGraph qualification failed security scan")
    if payload.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("LangGraph qualification schema mismatch")
    qualification_hash = payload.get("qualification_hash")
    expected = _sha256_bytes(
        _canonical_json_bytes({key: value for key, value in payload.items() if key != "qualification_hash"})
    )
    if qualification_hash != expected:
        raise ValueError("LangGraph qualification hash mismatch")


def publish_qualification(
    output_root: str | Path, payload: Mapping[str, Any]
) -> dict[str, Path]:
    output = Path(output_root)
    if not output.is_absolute():
        raise ValueError("LangGraph qualification output root must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("LangGraph qualification output parent must exist")
    assert_no_reparse_components(output)
    _validate_payload(payload)
    qualification_raw = _canonical_json_bytes(payload)
    qualification_hash = _sha256_bytes(qualification_raw)
    index = {
        "schema_version": QUALIFICATION_INDEX_SCHEMA_VERSION,
        "qualification_status": payload["qualification_status"],
        "outcome": payload["outcome"],
        "evidence_class": payload["evidence_class"],
        "qualification_sha256": qualification_hash,
        "artifacts": {
            "qualification.json": {
                "sha256": qualification_hash,
                "bytes": len(qualification_raw),
                "record_count": 1,
            }
        },
    }
    index_raw = _canonical_json_bytes(index)
    source_paths = tuple(PROJECT_ROOT / relative for relative in _LANGGRAPH_SOURCE_FILES)
    publish_files_atomically(
        {
            output / "qualification.json": qualification_raw,
            output / "artifact_index.json": index_raw,
        },
        overwrite=False,
        source_paths=source_paths,
    )
    return {
        "qualification_path": output / "qualification.json",
        "artifact_index_path": output / "artifact_index.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local no-model/no-network LangGraph custom adapter qualification."
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    payload = qualify(
        python_executable=arguments.python_executable,
        timeout_seconds=arguments.timeout_seconds,
    )
    paths = publish_qualification(arguments.output_root, payload)
    print(
        json.dumps(
            {
                "status": payload["qualification_status"],
                "outcome": payload["outcome"],
                "qualification_sha256": _sha256_file(paths["qualification_path"]),
                "qualification_path": str(paths["qualification_path"]),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["qualification_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
