from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mub.vnext.contracts import CompletionStatus
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.io.jsonl import write_models
from mub.vnext.runtime.resume import ResumeIndex, compute_run_identity
from mub.vnext.runtime.run import run_tasks
from tests.vnext.factories import build_task, build_task_run
from tests.vnext.test_runtime_engine import FakeAdapter, config


def _write_rows(path: Path, rows: list[Any]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
            handle.flush()


def _identity(**overrides: Any) -> str:
    data = {
        "task_manifest_hash": "a" * 64,
        "adapter_info": {"adapter_id": "fake", "adapter_version": "1"},
        "adapter_capabilities": {"supports_event_ingest": True},
        "runtime_config": {"retrieval_k": 3},
        "retrieval_policy": "normal_topk",
        "answer_mode": "slot_direct",
        "prompt_config": {"version": "p1"},
        "decoding_config": {"temperature": 0.0},
        "schema_version": "2.0.0",
        "compiler_version": "c1",
        "profile_version": "p1",
    }
    data.update(overrides)
    return compute_run_identity(**data)


def test_resume_index_decides_by_status_and_hash() -> None:
    task = build_task()
    completed = build_task_run(status=CompletionStatus.COMPLETED).model_copy(
        update={"system_events": [{"event": "runtime_identity", "run_identity": _identity(), "task_hash": "h1"}]}
    )
    failed = completed.model_copy(update={"task_id": "failed", "completion_status": CompletionStatus.FAILED})
    partial = completed.model_copy(update={"task_id": "partial", "completion_status": CompletionStatus.PARTIAL})
    unsupported = completed.model_copy(update={"task_id": "unsupported", "completion_status": CompletionStatus.NOT_SUPPORTED})
    path = Path("resume-index-test.jsonl")
    try:
        _write_rows(path, [completed, failed, partial, unsupported])
        index = ResumeIndex.from_jsonl(
            path,
            expected_task_ids=[completed.task_id, "failed", "partial", "unsupported", "missing"],
            run_identity=_identity(),
            expected_task_hashes={completed.task_id: "h1", "failed": "h1", "partial": "h1", "unsupported": "h1", "missing": "h1"},
        )
        assert index.decide(completed.task_id).action == "skip"
        assert index.decide("failed").action == "reject"
        assert index.decide("failed", retry_failed=True).action == "retry"
        assert index.decide("partial").action == "reject"
        assert index.decide("partial", retry_partial=True).action == "retry"
        assert index.decide("unsupported").action == "reject"
        assert index.decide("unsupported", retry_not_supported=True).action == "retry"
        assert index.decide(completed.task_id, task_hash="different").action == "retry"
        with pytest.raises(ValueError, match="missing"):
            index.require_complete()
    finally:
        path.unlink(missing_ok=True)


def test_resume_index_rejects_duplicate_ids_and_identity_mismatch(tmp_path: Path) -> None:
    row = build_task_run().model_copy(
        update={"system_events": [{"event": "runtime_identity", "run_identity": _identity(), "task_hash": "h1"}]}
    )
    duplicate = tmp_path / "duplicate.jsonl"
    _write_rows(duplicate, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        ResumeIndex.from_jsonl(duplicate, expected_task_ids=[row.task_id], run_identity=_identity())

    mismatch = tmp_path / "mismatch.jsonl"
    mismatched = row.model_copy(update={"system_events": [{"event": "runtime_identity", "run_identity": _identity(answer_mode="native_answer")}]})
    _write_rows(mismatch, [mismatched])
    with pytest.raises(ValueError, match="identity"):
        ResumeIndex.from_jsonl(mismatch, expected_task_ids=[row.task_id], run_identity=_identity())


def test_run_tasks_flushes_sidecar_and_finalizes_manifest(tmp_path: Path) -> None:
    task = build_task()
    output = tmp_path / "run"

    result = run_tasks(
        [task],
        adapter_factory=lambda item: FakeAdapter(item),
        run_config=config(),
        output_dir=output,
        task_manifest_hash="b" * 64,
    )

    assert result.manifest_path == output / "run_manifest.json"
    assert result.manifest_path.exists()
    rows = (output / "task_runs.jsonl").read_bytes().splitlines()
    assert len(rows) == 1
    assert (output / "progress.json").exists()
    progress = json.loads((output / "progress.json").read_text())
    assert progress["expected_ids"] == [task.task_id]
    assert progress["completed_ids"] == [task.task_id]
    assert result.manifest.completed_task_count == 1


def test_run_tasks_keeps_partial_progress_without_false_manifest(tmp_path: Path) -> None:
    task = build_task()
    output = tmp_path / "interrupted"
    calls = 0

    def factory(item):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeAdapter(item)
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run_tasks([task, task.model_copy(update={"task_id": "task-2"})], adapter_factory=factory, run_config=config(), output_dir=output, task_manifest_hash="b" * 64)
    assert (output / "progress.json").exists()
    assert not (output / "run_manifest.json").exists()


def test_run_identity_does_not_include_output_directory() -> None:
    assert _identity() == _identity(output_dir="one")


def test_runtime_cli_help_is_local_and_declares_protocol_flags() -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "vnext_run_pilot.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=True)
    assert "--tasks" in result.stdout
    assert "--task-manifest" in result.stdout
    assert "--adapter" in result.stdout
    assert "--retry-failed" in result.stdout
    assert "api" in result.stdout.lower()
