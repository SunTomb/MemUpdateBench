from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.contracts import ArtifactRef, TaskManifest
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, SupportReason
from mub.vnext.contracts.manifest import RunManifest
from mub.vnext.contracts.runtime import AnswerPrediction, TaskRunRecord
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash, sha256_model
from mub.vnext.io.jsonl import write_models
from mub.vnext.scoring.aggregate import aggregate_scores
from mub.vnext.scoring.pilot import (
    authenticate_pilot_files,
    score_pilot_records,
)


def _task_manifest(task, task_path: str = "tasks.jsonl", task_hash: str | None = None):
    return TaskManifest(
        data_release_id="pilot-test",
        split_policy_version="test-v1",
        compiler_versions={"compiler": "test"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={"test": 1},
        family_difficulty_counts={"family|easy": 1},
        semantic_core_counts={"core": 1},
        task_file_paths_and_hashes=(ArtifactRef(
            path=task_path,
            sha256=task_hash or "a" * 64,
            media_type="application/jsonl",
            record_count=1,
        ),),
        leakage_check_summary={"task_hashes": {task.task_id: sha256_model(task)}},
        human_audit_artifacts=(),
        created_at="2026-08-01T00:00:00Z",
        code_revision="test",
    )


def _manifest(task, *, task_manifest_hash="b" * 64, runtime_hash="c" * 64, system_name="fixture_memory_system"):
    return RunManifest(
        run_id="run_fixture_0001",
        timestamp="2026-08-01T00:00:00Z",
        code_revision="test",
        dirty_state=False,
        task_manifest=ArtifactRef(path="task_manifest.json", sha256=task_manifest_hash, media_type="application/json"),
        adapter_info={
            "adapter_id": "adapter_fixture", "adapter_version": "1.0", "system_name": system_name,
            "system_version": "1.0", "configuration_hash": "1" * 64,
        },
        adapter_capabilities={
            "supports_isolated_reset": True, "supports_event_ingest": True, "supports_add": True,
            "supports_update": True, "supports_noop": True, "exports_entries": True,
            "exports_raw_state": True, "exports_source_event_ids": True, "exports_timestamps_or_order": True,
            "exports_object_keys": True, "exports_values": True, "exports_retrieval_ids": True,
            "exports_retrieval_scores": True, "exports_action_trace": True,
        },
        capability_verification_artifact=None,
        model_name=None, provider=None, model_revision=None,
        prompt_config={}, decoding_config={}, seed_information={},
        action_parser_version="action-parser-v1", answer_parser_version="answer-parser-v1",
        memory_entry_extractor_version="entry-extractor-v1", object_value_extractor_config_hash="2" * 64,
        redaction_policy_version="redaction-v1", environment_summary={}, package_summary={},
        expected_task_count=1, completed_task_count=1, failed_task_count=0, not_supported_task_count=0,
        raw_provider_response_artifacts=(), raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(ArtifactRef(path="runs.jsonl", sha256=runtime_hash, media_type="application/jsonl", record_count=1),),
        score_artifacts=(), native_vs_extracted_field_summary={},
    )


def test_perfect_and_failure_decomposition(make_task, make_task_run):
    task = make_task()
    perfect = make_task_run()
    manifest = _manifest(task)
    score = score_pilot_records([task], [perfect], _task_manifest(task), manifest)[0]
    assert score.state_scores.final_state_accuracy == 1.0
    assert score.answer_scores.exact_match == 1.0
    assert "missed_update" in score.failure_flags

    payload = perfect.model_dump(mode="python")
    payload["parsed_actions"][1 if len(payload["parsed_actions"]) > 1 else 0]["operation"] = "NOOP"
    payload["memory_snapshots"][0]["state_by_object"] = {}
    stale = TaskRunRecord.model_validate(payload)
    failed = score_pilot_records([task], [stale], _task_manifest(task), manifest)[0]
    assert failed.state_scores.final_state_accuracy == 0.0
    assert "missed_update" in failed.failure_flags
    assert failed.primary_failure == "wrong_operation"


def test_family_c_dispositions_are_not_collapsed(make_task, make_task_run):
    task = make_task()
    task_payload = task.model_dump(mode="python")
    query = task_payload["queries"][0]
    key = task_payload["target_objects"][0]
    query.update({
        "query_type": "unresolved_reference",
        "reference_candidates": [{"candidate_id": "candidate_0", "object_key": key}],
        "surface_references": [{
            "reference_id": "reference_0", "surface_text": "it", "normalized_text": "it",
            "candidate_ids": ["candidate_0"],
        }],
    })
    task_payload["task_family"] = "entity_attribute_grounding"
    task_payload["gold"]["gold_answers"] = {}
    task_payload["gold"]["acceptable_answers"] = {}
    task_payload["gold"]["canonical_answers"] = {
        "query_0": {
            "disposition": "answered", "resolution_status": "unique",
            "selected_candidate_ids": ["candidate_0"], "value": "Qingdao",
        }
    }
    task = type(task).model_validate(task_payload)
    run_payload = make_task_run().model_dump(mode="python")
    run_payload["answer_predictions"][0]["disposition"] = AnswerDisposition.ABSTAINED
    run_payload["answer_predictions"][0]["parsed_answer"] = None
    abstained = TaskRunRecord.model_validate(run_payload)
    score = score_pilot_records([task], [abstained], _task_manifest(task), _manifest(task))[0]
    assert score.answer_scores.reference_resolution_accuracy == 0.0
    run_payload["answer_predictions"][0]["disposition"] = AnswerDisposition.UNAVAILABLE
    unavailable = TaskRunRecord.model_validate(run_payload)
    unavailable_score = score_pilot_records([task], [unavailable], _task_manifest(task), _manifest(task))[0]
    assert unavailable_score.answer_scores.reference_resolution_accuracy is None
    assert unavailable_score.supported_metric_fields["answer_scores.reference_resolution_accuracy"].reason is SupportReason.MISSING_ARTIFACT


def test_aggregate_groups_are_explicit_and_deterministic(make_task, make_score_record, make_run_manifest):
    task = make_task()
    first = make_score_record(task_id=task.task_id, run_id="run_fixture_0001")
    second = make_score_record(task_id="task-2", run_id="run_fixture_0001", difficulty="hard")
    # The second fixture is only used as a score row; aggregation must retain the task manifest join.
    task2 = task.model_copy(update={"task_id": "task-2", "difficulty": "hard"})
    manifest = make_run_manifest(expected=2, completed=2)
    summary_a = aggregate_scores([first, second], [task, task2], manifest)
    summary_b = aggregate_scores([first, second], [task, task2], manifest)
    assert summary_a == summary_b
    assert summary_a["counts"]["expected"] == 2
    groups = summary_a["leaderboard"]["rows"]
    assert any(row["group_by"] == "overall" for row in groups)
    assert any(row["group_by"] == "family" for row in groups)
    metric = next(row for row in groups if row["group_by"] == "overall")["micro"]["action_scores.operation_accuracy"]
    assert set(metric) == {"numerator", "denominator", "value"}


def test_authentication_rejects_missing_duplicate_and_hash_mismatch(tmp_path: Path, make_task, make_task_run):
    task = make_task()
    tasks_path = tmp_path / "tasks.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    manifest_path = tmp_path / "task_manifest.json"
    run_manifest_path = tmp_path / "run_manifest.json"
    write_models(tasks_path, [task], id_field="task_id")
    task_manifest = _task_manifest(task, task_hash=hashlib.sha256(tasks_path.read_bytes()).hexdigest())
    manifest_path.write_bytes(canonical_json_bytes(task_manifest))
    run = make_task_run()
    write_models(runs_path, [run], id_field="task_id")
    run_manifest = _manifest(task, task_manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), runtime_hash=hashlib.sha256(runs_path.read_bytes()).hexdigest())
    run_manifest_path.write_bytes(canonical_json_bytes(run_manifest))
    bundle = authenticate_pilot_files(tasks_path, manifest_path, runs_path, run_manifest_path)
    assert bundle.tasks[0].task_id == task.task_id

    run_manifest_path.write_bytes(canonical_json_bytes(run_manifest.model_copy(update={"task_manifest": {**run_manifest.task_manifest.model_dump(mode="python"), "sha256": "d" * 64}})))
    with pytest.raises(ValueError, match="hash"):
        authenticate_pilot_files(tasks_path, manifest_path, runs_path, run_manifest_path)
