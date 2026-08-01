from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mub.vnext.contracts.manifest import RunManifest, TaskManifest
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.score import ScoreRecord
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io.canonical import semantic_task_hash, sha256_model


def export_case(
    task: MemUpdateTask,
    run: TaskRunRecord,
    score: ScoreRecord,
    task_manifest: TaskManifest | None,
    run_manifest: RunManifest,
    *,
    task_artifact_hash: str | None = None,
    run_artifact_hash: str | None = None,
    score_artifact_hash: str | None = None,
) -> dict[str, Any]:
    """Join authenticated canonical artifacts into one auditable case.

    This function is deliberately a projection: metric values, support reasons,
    and failure labels are copied from ``score`` and never recalculated.
    """
    if run.task_id != task.task_id or score.task_id != task.task_id:
        raise ValueError("task ID mismatch across case artifacts")
    if run.run_id != score.run_id or run.adapter_id != score.adapter_id:
        raise ValueError("run or adapter ID mismatch across case artifacts")
    if run_manifest.run_id != run.run_id or run_manifest.adapter_info.adapter_id != run.adapter_id:
        raise ValueError("run manifest identity mismatch")
    if score.task_family != task.task_family or score.difficulty != task.difficulty:
        raise ValueError("task metadata mismatch in score")

    task_hash = sha256_model(task)
    if task_manifest is not None:
        declared = task_manifest.leakage_check_summary.get("task_hashes", {})
        if not isinstance(declared, Mapping) or declared.get(task.task_id) != task_hash:
            raise ValueError("task hash mismatch in task manifest")
        refs = task_manifest.task_file_paths_and_hashes
        if task_artifact_hash is not None and task_artifact_hash not in {ref.sha256 for ref in refs}:
            raise ValueError("task artifact hash mismatch")
    run_hash = sha256_model(run)
    if run_artifact_hash is not None:
        refs = run_manifest.normalized_runtime_artifacts
        if run_artifact_hash not in {ref.sha256 for ref in refs}:
            raise ValueError("run artifact hash mismatch")
    if score_artifact_hash is not None:
        if score_artifact_hash not in {ref.sha256 for ref in run_manifest.score_artifacts}:
            raise ValueError("score artifact hash mismatch")

    private = task.source.provenance.get("redistributable") is False
    source = {
        "source_id": task.source.source_id,
        "source_type": task.source.source_type,
        "source_uri": task.source.source_uri if not private else None,
        "license_or_privacy": task.source.license_or_privacy,
        "raw_hash": task.source.raw_hash,
        "normalized_hash": task.source.normalized_hash,
        "normalization_version": task.source.normalization_version,
        "anchors": [event.source_anchor for event in task.events],
        "redacted": private,
    }
    timeline = []
    for event in task.events:
        row = event.model_dump(mode="json")
        if private:
            row.pop("raw_text", None)
            row.pop("normalized_text", None)
        timeline.append(row)

    capabilities = {
        "action_trace": score.audit_scores.action_trace_available is not False and bool(run_manifest.adapter_capabilities.exports_action_trace),
        "state_export": score.audit_scores.state_export_available is not False and bool(run_manifest.adapter_capabilities.exports_entries and run_manifest.adapter_capabilities.exports_values),
        "retrieval_trace": score.audit_scores.retrieval_trace_available is not False and bool(run_manifest.adapter_capabilities.exports_retrieval_ids),
        "answer_output": bool(run.answer_predictions),
    }
    action_trace = [item.model_dump(mode="json") for item in run.parsed_actions]
    snapshots = [item.model_dump(mode="json") for item in run.memory_snapshots]
    retrieval = [item.model_dump(mode="json") for item in run.retrieval_traces]
    answers = [item.model_dump(mode="json") for item in run.answer_predictions]
    return {
        "schema_version": task.schema_version,
        "case_id": f"{run.run_id}:{task.task_id}",
        "task": {
            "task_id": task.task_id,
            "task_family": task.task_family,
            "difficulty": task.difficulty,
            "target_objects": [key.model_dump(mode="json") for key in task.target_objects],
            "queries": [query.model_dump(mode="json") for query in task.queries],
            "metadata": task.metadata.model_dump(mode="json"),
        },
        "timeline": timeline,
        "gold_actions": [action.model_dump(mode="json") for action in task.gold.actions],
        "predicted_actions": action_trace if capabilities["action_trace"] else None,
        "snapshots": snapshots if capabilities["state_export"] else None,
        "final_state": snapshots[-1]["state_by_object"] if snapshots and capabilities["state_export"] else None,
        "retrieval": retrieval if capabilities["retrieval_trace"] else None,
        "retrieved_context_ids": [entry["entry_id"] for trace in retrieval for entry in trace["retrieved_entries"]] if capabilities["retrieval_trace"] else None,
        "answer_output": answers if capabilities["answer_output"] else None,
        "metrics": {name: getattr(score, name).model_dump(mode="json") for name in type(score).model_fields if name.endswith("_scores")},
        "metric_support": {key: value.model_dump(mode="json") for key, value in score.supported_metric_fields.items()},
        "failure_flags": [flag.value if hasattr(flag, "value") else flag for flag in score.failure_flags],
        "primary_failure": score.primary_failure,
        "capabilities": capabilities,
        "artifacts": {
            "task_hash": task_artifact_hash or task_hash,
            "semantic_task_hash": semantic_task_hash(task),
            "run_hash": run_artifact_hash or run_hash,
            "score_hash": score_artifact_hash,
            "task_manifest_hash": None,
            "run_manifest_hash": sha256_model(run_manifest),
        },
        "source": source,
    }


__all__ = ["export_case"]
