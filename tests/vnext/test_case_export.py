from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.audit.cases import export_case
from mub.vnext.contracts import ArtifactRef, TaskManifest
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash, sha256_model
from mub.vnext.io.jsonl import write_models
from mub.vnext.scoring.pilot import authenticate_pilot_files


def _task_manifest(task, task_path: Path) -> TaskManifest:
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


def _runtime_identity(run):
    payload = run.model_dump(mode="python")
    payload["system_events"] = [
        {"event": "runtime_identity", "run_identity": "fixture-runtime", "task_hash": ""},
    ]
    return type(run).model_validate(payload)


def test_export_case_joins_artifacts_without_recomputing_metrics(make_task, make_task_run, make_score_record, make_run_manifest, tmp_path):
    task = make_task()
    run = _runtime_identity(make_task_run())
    run_payload = run.model_dump(mode="python")
    run_payload["system_events"][0]["task_hash"] = semantic_task_hash(task)
    run = type(run).model_validate(run_payload)
    score = make_score_record()
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_bytes(canonical_json_bytes(task) + b"\n")
    task_manifest = _task_manifest(task, task_path)
    run_manifest = make_run_manifest(native_vs_extracted_field_summary={"runtime_identity": "fixture-runtime"}, normalized_runtime_artifacts=[{"path": "tasks.jsonl", "sha256": "b" * 64, "media_type": "application/jsonl", "record_count": 1}])
    case = export_case(task, run, score, task_manifest, run_manifest, task_artifact_hash=task_manifest.task_file_paths_and_hashes[0].sha256, run_artifact_hash="b" * 64)
    assert case["task"]["task_id"] == task.task_id
    assert case["timeline"][0]["role"] == "stale_same_slot"
    assert case["gold_actions"] and case["predicted_actions"]
    assert case["snapshots"][0]["state_by_object"]["default|friend:alex|location|"] == "Qingdao"
    assert case["metrics"]["state_scores"]["final_state_accuracy"] == 1.0
    assert case["failure_flags"] == []
    assert case["primary_failure"] is None
    assert case["artifacts"]["task_hash"] == task_manifest.task_file_paths_and_hashes[0].sha256
    assert case["source"]["normalized_hash"] == task.source.normalized_hash


def test_export_case_preserves_missing_capabilities_without_empty_trace(make_task, make_task_run, make_score_record, make_run_manifest):
    task = make_task()
    payload = make_task_run().model_dump(mode="python")
    payload["memory_snapshots"] = []
    payload["retrieval_traces"] = []
    run = type(make_task_run()).model_validate(payload)
    score = make_score_record(audit_scores={"action_trace_available": False, "state_export_available": False, "retrieval_trace_available": False})
    capabilities = type(make_run_manifest().adapter_capabilities).model_fields
    manifest = make_run_manifest(adapter_capabilities={name: False for name in capabilities if name != "extractor_version"} | {"extractor_version": None}, native_vs_extracted_field_summary={"runtime_identity": "fixture-runtime"})
    case = export_case(task, run, score, None, manifest)
    assert case["capabilities"]["state_export"] is False
    assert case["snapshots"] is None
    assert case["retrieval"] is None
    assert case["metrics"]["audit_scores"]["state_export_available"] is False


def test_export_case_redacts_private_source_text_but_retains_hashes_and_anchors(make_task, make_task_run, make_score_record, make_run_manifest):
    task = make_task()
    payload = task.model_dump(mode="python")
    payload["source"]["provenance"]["redistributable"] = False
    private_task = type(task).model_validate(payload)
    case = export_case(private_task, make_task_run(), make_score_record(), None, make_run_manifest())
    assert "raw_text" not in case["source"]
    assert case["source"]["redacted"] is True
    assert case["source"]["raw_hash"] == private_task.source.raw_hash
    assert case["source"]["anchors"] == [event.source_anchor for event in private_task.events]


def test_export_case_rejects_id_or_hash_mismatch(make_task, make_task_run, make_score_record, make_run_manifest, tmp_path):
    task = make_task()
    run_payload = make_task_run().model_dump(mode="python")
    run_payload["task_id"] = "other"
    with pytest.raises(ValueError, match="task ID"):
        export_case(task, type(make_task_run()).model_validate(run_payload), make_score_record(), None, make_run_manifest())
    score_payload = make_score_record().model_dump(mode="python")
    score_payload["task_id"] = "other"
    with pytest.raises(ValueError, match="task ID"):
        export_case(task, make_task_run(), type(make_score_record()).model_validate(score_payload), None, make_run_manifest())
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_bytes(canonical_json_bytes(task) + b"\n")
    task_manifest = _task_manifest(task, task_path)
    with pytest.raises(ValueError, match="hash"):
        export_case(task, make_task_run(), make_score_record(), task_manifest, make_run_manifest(), task_artifact_hash="0" * 64)
