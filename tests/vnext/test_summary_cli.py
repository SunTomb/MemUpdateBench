from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.contracts import ArtifactRef, CompletionStatus, TaskManifest
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash, sha256_model
from mub.vnext.io.jsonl import write_models
from scripts.vnext_summarize_pilot import main


def _manifest(task, task_path: Path) -> TaskManifest:
    return TaskManifest(
        data_release_id="fixture-release",
        split_policy_version="split-v1",
        compiler_versions={"compiler": "1"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={"test": 1},
        family_difficulty_counts={"repeated_same_slot/easy": 1},
        semantic_core_counts={"semantic_core_0001": 1},
        task_file_paths_and_hashes=(ArtifactRef(path=task_path.name, sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(), media_type="application/jsonl", record_count=1),),
        leakage_check_summary={"task_hashes": {task.task_id: sha256_model(task)}, "semantic_task_hashes": {task.task_id: semantic_task_hash(task)}},
        human_audit_artifacts=(),
        created_at="2026-07-20T00:00:00Z",
        code_revision="fixed-test-revision",
    )


def _run_with_identity(run, task):
    payload = run.model_dump(mode="python")
    payload["system_events"] = [
        {"event": "runtime_identity", "run_identity": "fixture-runtime", "task_hash": semantic_task_hash(task)},
    ]
    return type(run).model_validate(payload)


def _write_fixture(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest):
    task = make_task()
    run = _run_with_identity(make_task_run(), task)
    score = make_score_record()
    tasks_path = tmp_path / "tasks.jsonl"
    runs_path = tmp_path / "task_runs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_models(tasks_path, [task], id_field="task_id")
    write_models(runs_path, [run], id_field="task_id")
    write_models(scores_path, [score], id_field="task_id")
    task_manifest = _manifest(task, tasks_path)
    task_manifest_path = tmp_path / "task_manifest.json"
    task_manifest_path.write_bytes(canonical_json_bytes(task_manifest) + b"\n")
    run_manifest = make_run_manifest(
        native_vs_extracted_field_summary={"runtime_identity": "fixture-runtime"},
        task_manifest={"path": task_manifest_path.name, "sha256": hashlib.sha256(task_manifest_path.read_bytes()).hexdigest(), "media_type": "application/json"},
        score_artifacts=[{"path": scores_path.name, "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(), "media_type": "application/jsonl", "record_count": 1}],
        normalized_runtime_artifacts=[{"path": runs_path.name, "sha256": hashlib.sha256(runs_path.read_bytes()).hexdigest(), "media_type": "application/jsonl", "record_count": 1}],
    )
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_bytes(canonical_json_bytes(run_manifest) + b"\n")
    return tasks_path, task_manifest_path, runs_path, scores_path, run_manifest_path


def _run_cli(tmp_path, fixture, policy):
    tasks, task_manifest, runs, scores, run_manifest = fixture
    output = tmp_path / f"out-{policy}"
    assert main([
        "--tasks", str(tasks), "--task-manifest", str(task_manifest),
        "--task-runs", str(runs), "--scores", str(scores),
        "--run-manifest", str(run_manifest), "--output-dir", str(output),
        "--case-policy", policy,
    ]) == 0
    return output


def test_summary_cli_writes_all_required_outputs_and_policies(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest):
    fixture = _write_fixture(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest)
    for policy in ("all", "failures", "stratified"):
        output = _run_cli(tmp_path, fixture, policy)
        assert {p.name for p in output.iterdir()} >= {
            "summary.json", "summary.csv", "failure_breakdown.json",
            "capability_coverage.json", "cases.jsonl", "artifact_index.json",
        }
        assert json.loads((output / "summary.json").read_text())["leaderboard"]["eligible"] is True


def test_summary_cli_is_deterministic_and_index_hashes_outputs(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest):
    fixture = _write_fixture(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest)
    first = _run_cli(tmp_path, fixture, "all")
    second = tmp_path / "second"
    tasks, manifest, runs, scores, run_manifest = fixture
    assert main(["--tasks", str(tasks), "--task-manifest", str(manifest), "--task-runs", str(runs), "--scores", str(scores), "--run-manifest", str(run_manifest), "--output-dir", str(second), "--case-policy", "all"]) == 0
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    index = json.loads((first / "artifact_index.json").read_text())
    for name, digest in index["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest


def test_summary_cli_refuses_incomplete_and_raw_legacy_without_mixing_outputs(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest):
    fixture = _write_fixture(tmp_path, make_task, make_task_run, make_score_record, make_run_manifest)
    tasks, task_manifest, runs, scores, run_manifest = fixture
    output = tmp_path / "atomic"
    output.mkdir()
    (output / "summary.json").write_text("old")
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_bytes(b"")
    assert main(["--tasks", str(tasks), "--task-manifest", str(task_manifest), "--task-runs", str(incomplete), "--scores", str(scores), "--run-manifest", str(run_manifest), "--output-dir", str(output), "--case-policy", "all"]) == 2
    assert (output / "summary.json").read_text() == "old"
    raw = tmp_path / "legacy.json"
    raw.write_text(json.dumps({"task_id": "legacy", "metrics": {"exact_match": 1.0}}))
    assert main(["--tasks", str(tasks), "--task-manifest", str(task_manifest), "--task-runs", str(runs), "--scores", str(raw), "--run-manifest", str(run_manifest), "--output-dir", str(output), "--case-policy", "all"]) == 2
    assert (output / "summary.json").read_text() == "old"


def test_summary_cli_rejects_task_swapped_completion_statuses_without_mixing_outputs(
    tmp_path, capsys, make_task, make_task_run, make_score_record, make_run_manifest
):
    first_task = make_task()
    second_task = first_task.model_copy(update={"task_id": "task_repeated_same_slot_easy_0002"})
    tasks = (first_task, second_task)
    first_run = _run_with_identity(make_task_run(), first_task)
    second_run = make_task_run(status=CompletionStatus.FAILED, exception_type="timeout").model_copy(
        update={"task_id": second_task.task_id}
    )
    second_run = _run_with_identity(second_run, second_task)
    scores = (
        make_score_record(completion_status=CompletionStatus.FAILED),
        make_score_record(task_id=second_task.task_id, completion_status=CompletionStatus.COMPLETED),
    )

    tasks_path = tmp_path / "tasks.jsonl"
    runs_path = tmp_path / "task_runs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    write_models(tasks_path, tasks, id_field="task_id")
    write_models(runs_path, (first_run, second_run), id_field="task_id")
    write_models(scores_path, scores, id_field="task_id")
    task_manifest = TaskManifest(
        data_release_id="fixture-release",
        split_policy_version="split-v1",
        compiler_versions={"compiler": "1"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={"test": 2},
        family_difficulty_counts={"repeated_same_slot/easy": 2},
        semantic_core_counts={"semantic_core_0001": 2},
        task_file_paths_and_hashes=(ArtifactRef(
            path=tasks_path.name,
            sha256=hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
            media_type="application/jsonl",
            record_count=2,
        ),),
        leakage_check_summary={
            "task_hashes": {task.task_id: sha256_model(task) for task in tasks},
            "semantic_task_hashes": {task.task_id: semantic_task_hash(task) for task in tasks},
        },
        human_audit_artifacts=(),
        created_at="2026-07-20T00:00:00Z",
        code_revision="fixed-test-revision",
    )
    task_manifest_path = tmp_path / "task_manifest.json"
    task_manifest_path.write_bytes(canonical_json_bytes(task_manifest) + b"\n")
    run_manifest = make_run_manifest(
        expected=2,
        completed=1,
        failed=1,
        native_vs_extracted_field_summary={"runtime_identity": "fixture-runtime"},
        task_manifest={
            "path": task_manifest_path.name,
            "sha256": hashlib.sha256(task_manifest_path.read_bytes()).hexdigest(),
            "media_type": "application/json",
        },
        score_artifacts=[{
            "path": scores_path.name,
            "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
            "media_type": "application/jsonl",
            "record_count": 2,
        }],
        normalized_runtime_artifacts=[{
            "path": runs_path.name,
            "sha256": hashlib.sha256(runs_path.read_bytes()).hexdigest(),
            "media_type": "application/jsonl",
            "record_count": 2,
        }],
    )
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_bytes(canonical_json_bytes(run_manifest) + b"\n")
    output = tmp_path / "out"
    output.mkdir()
    (output / "summary.json").write_text("old")

    assert main([
        "--tasks", str(tasks_path), "--task-manifest", str(task_manifest_path),
        "--task-runs", str(runs_path), "--scores", str(scores_path),
        "--run-manifest", str(run_manifest_path), "--output-dir", str(output),
        "--case-policy", "all",
    ]) == 2
    assert capsys.readouterr().err == "vNext Pilot summary failed: score/run record mismatch\n"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {"summary.json": b"old"}
