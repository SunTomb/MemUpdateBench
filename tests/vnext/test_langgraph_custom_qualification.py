from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import pytest

from scripts import vnext_run_main_track_factorial_manager as runner
from scripts.vnext_qualify_langgraph_custom import (
    QUALIFICATION_SCHEMA_VERSION,
    build_blocked_payload,
    build_ready_payload,
    publish_qualification,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CONFIGURATION_HASH = runner._LANGMEM_CONFIGURATION_HASH


def _source_bindings() -> list[dict[str, str]]:
    return runner._langgraph_adapter_source_bindings()


def _runtime() -> dict:
    executable = Path(sys.executable).resolve()
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    return {
        "python_executable": str(executable),
        "executable_sha256": executable_sha256,
        "size_bytes": executable.stat().st_size,
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            "langmem": {
                "status": "AVAILABLE",
                "version": "0.0.30",
                "metadata_path": "C:/Python314/Lib/site-packages/langmem-0.0.30.dist-info",
                "metadata_files_sha256": "a" * 64,
            },
            "langgraph": {
                "status": "AVAILABLE",
                "version": "1.2.11",
                "metadata_path": "C:/Python314/Lib/site-packages/langgraph-1.2.11.dist-info",
                "metadata_files_sha256": "b" * 64,
            },
        },
    }


def _probe() -> dict:
    package_metadata = {
        "langmem": {
            "status": "AVAILABLE",
            "version": "0.0.30",
        },
        "langgraph": {
            "status": "AVAILABLE",
            "version": "1.2.11",
        },
    }
    return {
        "runtime_identity": _runtime(),
        "packages": package_metadata,
        "store_class": "langgraph.store.memory.InMemoryStore",
        "configuration_hash": EXPECTED_CONFIGURATION_HASH,
        "lifecycle": {
            "passed": True,
            "operations": ["reset", "add", "update", "noop", "retrieve", "export", "reset"],
        },
    }


def _rehash(payload: dict) -> dict:
    result = dict(payload)
    result["qualification_hash"] = hashlib.sha256(
        runner.canonical_json_bytes(
            {key: value for key, value in result.items() if key != "qualification_hash"}
        )
    ).hexdigest()
    return result


def test_ready_payload_binds_manifest_identity_sources_and_zero_boundary() -> None:
    payload = build_ready_payload(_probe(), source_bindings=_source_bindings())

    assert payload["schema_version"] == QUALIFICATION_SCHEMA_VERSION
    assert payload["qualification_status"] == "READY"
    assert payload["evidence_class"] == "capability_runtime_qualification"
    assert payload["scientific_evidence"] is False
    assert payload["manager_id"] == "langgraph_store_custom_adapter"
    assert payload["adapter_version"] == "memupdatebench-langmem-adapter-v1"
    assert payload["system_name"] == "langgraph_in_memory_store"
    assert payload["configuration_hash"] == EXPECTED_CONFIGURATION_HASH
    assert payload["adapter_source_files"] == _source_bindings()
    assert payload["execution_boundary"] == {
        "provider_calls": 0,
        "model_loads": 0,
        "database_accesses": 0,
        "network_calls": 0,
        "gpu_calls": 0,
        "executable_calls": 0,
        "remote_operations": 0,
    }
    assert payload["capabilities"] == runner.planner._PROFILE_CAPABILITIES
    assert payload["qualification_hash"] == hashlib.sha256(
        runner.canonical_json_bytes({key: value for key, value in payload.items() if key != "qualification_hash"})
    ).hexdigest()


def test_ready_payload_requires_runtime_identity() -> None:
    probe = _probe()
    probe.pop("runtime_identity")

    with pytest.raises(ValueError, match="runtime identity"):
        build_ready_payload(probe, source_bindings=_source_bindings())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("executable_sha256", "0" * 64, "executable_sha256"),
        ("python_executable", "C:/tampered/python.exe", "python_executable"),
    ],
)
def test_runner_rejects_tampered_runtime_executable_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = build_ready_payload(_probe(), source_bindings=_source_bindings())
    payload["runtime_identity"][field] = value
    tampered_path = _write_payload(tmp_path, _rehash(payload))

    with pytest.raises(runner.RuntimeIdentityError, match=message):
        runner._load_langgraph_qualification(
            tampered_path, EXPECTED_CONFIGURATION_HASH
        )


def test_runner_rejects_tampered_runtime_distribution_hash(tmp_path: Path) -> None:
    payload = build_ready_payload(_probe(), source_bindings=_source_bindings())
    payload["runtime_identity"]["packages"]["langmem"]["metadata_files_sha256"] = "0" * 64
    tampered_path = _write_payload(tmp_path, _rehash(payload))

    with pytest.raises(runner.RuntimeIdentityError, match="distribution"):
        runner._load_langgraph_qualification(
            tampered_path, EXPECTED_CONFIGURATION_HASH
        )


def test_blocked_payload_is_typed_and_never_ready(tmp_path: Path) -> None:
    payload = build_blocked_payload(
        reason_code="langgraph_dependency_unavailable",
        reason="langgraph is not installed",
        source_bindings=_source_bindings(),
        runtime_identity=_runtime(),
    )

    assert payload["outcome"] == "BLOCKED"
    assert payload["blockers"] == ["langgraph_dependency_unavailable"]
    assert payload["packages"]["langgraph"]["status"] == "UNAVAILABLE"
    with pytest.raises(runner.RuntimeIdentityError, match="qualification identity mismatch"):
        runner._load_langgraph_qualification(
            _write_payload(tmp_path, payload), EXPECTED_CONFIGURATION_HASH
        )


def _write_payload(directory: Path, payload: dict) -> Path:
    path = directory / ".langgraph-qualification-test.json"
    path.write_bytes(runner.canonical_json_bytes(payload))
    return path


def test_publish_writes_canonical_no_replace_qualification_and_index(tmp_path: Path) -> None:
    payload = build_ready_payload(_probe(), source_bindings=_source_bindings())
    result = publish_qualification(tmp_path / "qualification", payload)

    qualification_path = result["qualification_path"]
    index_path = result["artifact_index_path"]
    assert qualification_path == tmp_path / "qualification" / "qualification.json"
    assert index_path == tmp_path / "qualification" / "artifact_index.json"
    raw = qualification_path.read_bytes()
    assert json.loads(raw) == payload
    assert runner.canonical_json_bytes(json.loads(raw)) == raw
    index = json.loads(index_path.read_bytes())
    assert index["schema_version"] == f"{QUALIFICATION_SCHEMA_VERSION}.artifact-index"
    assert index["artifacts"]["qualification.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        publish_qualification(tmp_path / "qualification", payload)


def test_runner_accepts_ready_payload_and_rejects_source_tampering(tmp_path: Path) -> None:
    payload = build_ready_payload(_probe(), source_bindings=_source_bindings())
    path = _write_payload(tmp_path, payload)

    loaded = runner._load_langgraph_qualification(path, EXPECTED_CONFIGURATION_HASH)
    assert loaded["qualification_status"] == "READY"

    tampered = dict(payload)
    tampered["adapter_source_files"] = [dict(item, sha256="0" * 64) for item in _source_bindings()]
    tampered["qualification_hash"] = hashlib.sha256(
        runner.canonical_json_bytes({key: value for key, value in tampered.items() if key != "qualification_hash"})
    ).hexdigest()
    tampered_path = _write_payload(tmp_path, tampered)
    with pytest.raises(runner.RuntimeIdentityError, match="source"):
        runner._load_langgraph_qualification(tampered_path, EXPECTED_CONFIGURATION_HASH)
