from __future__ import annotations

import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "vnext_preflight_mem0.py"


def test_mem0_preflight_cli_help_imports_no_optional_sdk() -> None:
    result = subprocess.run(
        (sys.executable, str(SCRIPT), "--help"),
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--worker-configuration" in result.stdout
    assert "--output" in result.stdout
    assert "Traceback" not in result.stderr


def test_mem0_preflight_worker_environment_is_offline_and_secret_free() -> None:
    module = runpy.run_path(str(SCRIPT))
    source = dict(os.environ)
    source.update(
        {
            "CUDA_VISIBLE_DEVICES": "2",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_CACHE": "/approved/cache",
            "PYTHONPATH": os.pathsep.join(("/untrusted/shadow", "/other")),
            "OPENAI_API_KEY": "must-not-cross",
            "UNRELATED_SECRET": "must-not-cross",
        }
    )
    environment = module["build_mem0_preflight_worker_environment"](
        source,
        project_root=PROJECT_ROOT,
    )
    assert environment["MEM0_TELEMETRY"] == "false"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["PYTHONPATH"] == str(PROJECT_ROOT)
    assert "/untrusted/shadow" not in environment["PYTHONPATH"]
    assert "OPENAI_API_KEY" not in environment
    assert "UNRELATED_SECRET" not in environment


def test_mem0_preflight_preserves_isolated_python_executable_path() -> None:
    module = runpy.run_path(str(SCRIPT))
    executable = PROJECT_ROOT / "external" / "venv" / "bin" / "python"
    configuration = PROJECT_ROOT / "external" / "worker-config.json"
    command = module["build_mem0_worker_command"](
        python_executable=executable,
        worker_configuration_path=configuration,
    )
    assert command[0] == str(executable)
    assert command[-1] == str(configuration)


def test_mem0_preflight_rejects_immutable_core_output(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT))
    immutable = tmp_path / "data" / "vnext" / "core" / "v3"
    immutable.mkdir(parents=True)
    write_evidence = module["_write_evidence"]
    write_evidence.__globals__["_IMMUTABLE_CORE_ROOT"] = immutable

    with pytest.raises(ValueError, match="immutable Core"):
        write_evidence(
            immutable / "preflight.json",
            {"passed": True},
        )


def test_mem0_preflight_integration_requires_exact_entry_and_retrieval() -> None:
    module = runpy.run_path(str(SCRIPT))
    key = module["_key"]()
    expected_text = module["_event"]().raw_text
    entry = SimpleNamespace(
        entry_id="native-entry",
        content=expected_text,
        object_key_candidate=key,
        value_candidate="Paris",
    )
    retrieval = SimpleNamespace(
        trace=SimpleNamespace(
            query_id=module["_query"]().query_id,
            retrieved_entries=(entry,),
        )
    )
    answer = SimpleNamespace(
        prediction=SimpleNamespace(parsed_answer="Paris")
    )
    reset = SimpleNamespace(success=True)
    action = SimpleNamespace(execution_status=SimpleNamespace(value="executed"))
    entries = SimpleNamespace(entries=(entry,))

    assert module["_integration_passed"](
        reset_result=reset,
        action_result=action,
        entries_result=entries,
        retrieval_result=retrieval,
        answer_result=answer,
    )

    changed = SimpleNamespace(**{**entry.__dict__, "content": f" {expected_text} "})
    changed_entries = SimpleNamespace(entries=(changed,))
    assert not module["_integration_passed"](
        reset_result=reset,
        action_result=action,
        entries_result=changed_entries,
        retrieval_result=retrieval,
        answer_result=answer,
    )

    unrelated = SimpleNamespace(**{**entry.__dict__, "entry_id": "unrelated"})
    unrelated_retrieval = SimpleNamespace(
        trace=SimpleNamespace(
            query_id=module["_query"]().query_id,
            retrieved_entries=(unrelated,),
        )
    )
    assert not module["_integration_passed"](
        reset_result=reset,
        action_result=action,
        entries_result=entries,
        retrieval_result=unrelated_retrieval,
        answer_result=answer,
    )


def test_mem0_preflight_rejects_nonfinite_timeout() -> None:
    module = runpy.run_path(str(SCRIPT))
    for value in (float("inf"), float("nan"), 0.0):
        with pytest.raises(ValueError, match="finite positive"):
            module["_validate_timeout_seconds"](value)
