from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.common import MetricFieldSupport
from mub.vnext.contracts.enums import CompletionStatus, SupportReason
from mub.vnext.contracts.manifest import (
    ANSWER_NORMALIZATION_PROFILE,
    VALUE_NORMALIZATION_PROFILE,
    RunManifest,
    ScorerConfig,
    TaskManifest,
)
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.score import SCORE_LAYER_TYPES, ScoreRecord
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash, sha256_model
from mub.vnext.io.jsonl import read_models
from mub.vnext.runtime.resume import _event_metadata
from mub.vnext.scoring.scorer import _compute_audit, _compute_system, score_task
from mub.vnext.version import RUNTIME_RECORD_VERSION, SCHEMA_VERSION


@dataclass(frozen=True)
class PilotInputBundle:
    tasks: tuple[MemUpdateTask, ...]
    task_manifest: TaskManifest
    runs: tuple[TaskRunRecord, ...]
    run_manifest: RunManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_for(path: Path, artifacts, label: str):
    exact = [artifact for artifact in artifacts if Path(artifact.path) == path]
    if not exact:
        same_name = [artifact for artifact in artifacts if Path(artifact.path).name == path.name]
        if len(same_name) == 1:
            exact = same_name
    if len(exact) != 1:
        raise ValueError(f"{label} is not uniquely referenced by its manifest")
    return exact[0]


def _validate_task_set(tasks: Sequence[MemUpdateTask], manifest: TaskManifest) -> dict[str, MemUpdateTask]:
    by_id: dict[str, MemUpdateTask] = {}
    for task in tasks:
        if task.schema_version != SCHEMA_VERSION:
            raise ValueError(f"task {task.task_id} has an unauthenticated schema version")
        if task.task_id in by_id:
            raise ValueError(f"duplicate task ID: {task.task_id}")
        by_id[task.task_id] = task
    declared = manifest.leakage_check_summary.get("task_hashes")
    if not isinstance(declared, Mapping):
        raise ValueError("task manifest lacks authenticated task_hashes")
    if set(declared) != set(by_id):
        raise ValueError("task IDs do not match the task manifest")
    for task_id, task in by_id.items():
        expected = declared[task_id]
        if expected != sha256_model(task):
            raise ValueError(f"task hash mismatch for {task_id}")
    return by_id


def _validate_run_set(
    tasks: dict[str, MemUpdateTask],
    runs: Sequence[TaskRunRecord],
    manifest: RunManifest,
) -> dict[str, TaskRunRecord]:
    if manifest.schema_version != SCHEMA_VERSION or manifest.runtime_record_version != RUNTIME_RECORD_VERSION:
        raise ValueError("run manifest schema/runtime version is not current")
    if manifest.expected_task_count != len(tasks):
        raise ValueError("run manifest expected task count does not match task manifest")
    by_id: dict[str, TaskRunRecord] = {}
    for run in runs:
        if run.schema_version != SCHEMA_VERSION or run.runtime_record_version != RUNTIME_RECORD_VERSION:
            raise ValueError(f"run {run.run_id} has an unauthenticated schema version")
        if run.task_id in by_id:
            raise ValueError(f"duplicate task ID in run records: {run.task_id}")
        if run.task_id not in tasks:
            raise ValueError(f"unexpected task ID in run records: {run.task_id}")
        if run.run_id != manifest.run_id:
            raise ValueError(f"run identity mismatch for task {run.task_id}")
        if run.adapter_id != manifest.adapter_info.adapter_id:
            raise ValueError(f"adapter identity mismatch for task {run.task_id}")
        provenance = run.parser_extractor_provenance
        for field, expected in (
            ("action_parser_version", manifest.action_parser_version),
            ("answer_parser_version", manifest.answer_parser_version),
            ("memory_entry_extractor_version", manifest.memory_entry_extractor_version),
            ("redaction_policy_version", manifest.redaction_policy_version),
        ):
            if getattr(provenance, field) != expected:
                raise ValueError(f"{field} mismatch for task {run.task_id}")
        runtime_identity = _event_metadata(run, "run_identity")
        declared_identity = manifest.native_vs_extracted_field_summary.get("runtime_identity")
        if runtime_identity is not None and runtime_identity != declared_identity:
            raise ValueError(f"runtime identity mismatch for task {run.task_id}")
        declared_task_hash = _event_metadata(run, "task_hash")
        if declared_task_hash is not None and declared_task_hash != semantic_task_hash(tasks[run.task_id]):
            raise ValueError(f"runtime task hash mismatch for {run.task_id}")
        by_id[run.task_id] = run
    if set(by_id) != set(tasks):
        missing = sorted(set(tasks) - set(by_id))
        raise ValueError(f"missing task IDs in run records: {missing}")
    completed = sum(run.completion_status is CompletionStatus.COMPLETED for run in by_id.values())
    failed = sum(run.completion_status in {CompletionStatus.FAILED, CompletionStatus.PARTIAL} for run in by_id.values())
    unsupported = sum(run.completion_status is CompletionStatus.NOT_SUPPORTED for run in by_id.values())
    if (completed, failed, unsupported) != (
        manifest.completed_task_count,
        manifest.failed_task_count,
        manifest.not_supported_task_count,
    ):
        raise ValueError("run manifest status counts do not match normalized run records")
    return by_id


def authenticate_pilot_files(
    tasks_path: str | Path,
    task_manifest_path: str | Path,
    run_records_path: str | Path,
    run_manifest_path: str | Path,
) -> PilotInputBundle:
    tasks_path, task_manifest_path = Path(tasks_path), Path(task_manifest_path)
    run_records_path, run_manifest_path = Path(run_records_path), Path(run_manifest_path)
    task_manifest = TaskManifest.model_validate_json(task_manifest_path.read_bytes())
    run_manifest = RunManifest.model_validate_json(run_manifest_path.read_bytes())
    if run_manifest.task_manifest.sha256 != _sha256_file(task_manifest_path):
        raise ValueError("run manifest task-manifest hash mismatch")
    task_ref = _artifact_for(tasks_path, task_manifest.task_file_paths_and_hashes, "task file")
    if task_ref.sha256 != _sha256_file(tasks_path):
        raise ValueError("task file hash mismatch")
    run_ref = _artifact_for(run_records_path, run_manifest.normalized_runtime_artifacts, "run-record file")
    if run_ref.sha256 != _sha256_file(run_records_path):
        raise ValueError("run-record file hash mismatch")
    tasks = tuple(read_models(tasks_path, MemUpdateTask, id_field="task_id"))
    runs = tuple(read_models(run_records_path, TaskRunRecord, id_field="task_id"))
    task_by_id = _validate_task_set(tasks, task_manifest)
    _validate_run_set(task_by_id, runs, run_manifest)
    if task_ref.record_count is not None and task_ref.record_count != len(tasks):
        raise ValueError("task manifest record_count mismatch")
    if run_ref.record_count is not None and run_ref.record_count != len(runs):
        raise ValueError("run manifest record_count mismatch")
    return PilotInputBundle(tasks, task_manifest, runs, run_manifest)


def _default_config() -> ScorerConfig:
    return ScorerConfig(
        value_normalization_profile=VALUE_NORMALIZATION_PROFILE,
        answer_normalization_profile=ANSWER_NORMALIZATION_PROFILE,
        strict_capability_check=False,
    )


def _fix_zero_write_metric(task: MemUpdateTask, score: ScoreRecord) -> ScoreRecord:
    required_writes = sum(action.operation.value != "NOOP" for action in task.gold.actions)
    if required_writes:
        return score
    if score.store_scores.write_amplification is None:
        return score
    payload = score.model_dump(mode="python")
    payload["store_scores"]["write_amplification"] = None
    payload["supported_metric_fields"]["store_scores.write_amplification"] = MetricFieldSupport(
        reason=SupportReason.NOT_APPLICABLE,
        null_policy="exclude_from_aggregation",
        detail="Task has no required mutating gold actions; write amplification is not applicable.",
    )
    return ScoreRecord.model_validate(payload)


def score_pilot_records(
    tasks: Iterable[MemUpdateTask],
    runs: Iterable[TaskRunRecord],
    task_manifest: TaskManifest,
    run_manifest: RunManifest,
    *,
    config: ScorerConfig | None = None,
) -> tuple[ScoreRecord, ...]:
    task_list = tuple(tasks)
    run_list = tuple(runs)
    task_by_id = _validate_task_set(task_list, task_manifest)
    run_by_id = _validate_run_set(task_by_id, run_list, run_manifest)
    capabilities = AdapterCapabilities.model_validate(run_manifest.adapter_capabilities.model_dump(mode="python"))
    scorer_config = config or _default_config()
    if scorer_config.strict_capability_check:
        scorer_config = scorer_config.model_copy(update={"strict_capability_check": False})
    scores = []
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        run = run_by_id[task_id]
        score = _fix_zero_write_metric(task, score_task(task, run, capabilities, scorer_config))
        disposition_counts = {}
        for answer in run.answer_predictions:
            disposition_counts[answer.disposition.value] = disposition_counts.get(answer.disposition.value, 0) + 1
        if disposition_counts:
            payload = score.model_dump(mode="python")
            legacy = dict(payload.get("legacy_metrics", {}))
            legacy["pilot_answer_dispositions"] = disposition_counts
            payload["legacy_metrics"] = legacy
            score = ScoreRecord.model_validate(payload)
        scores.append(score)
    return tuple(scores)


def _score_layers(task, run, capabilities, config=None):
    return score_task(task, run, capabilities, config or _default_config())


def score_actions(task, run, capabilities, config=None):
    return _score_layers(task, run, capabilities, config).action_scores


def score_state(task, run, capabilities, config=None):
    return _score_layers(task, run, capabilities, config).state_scores


def score_store(task, run, capabilities, config=None):
    return _score_layers(task, run, capabilities, config).store_scores


def score_retrieval(task, run, capabilities, config=None):
    return _score_layers(task, run, capabilities, config).retrieval_scores


def score_answers(task, run, capabilities, config=None):
    return _score_layers(task, run, capabilities, config).answer_scores


def score_system(run, capabilities):
    values = {}
    for field in SCORE_LAYER_TYPES["system_scores"].model_fields:
        value, _ = _compute_system(f"system_scores.{field}", run)
        values[field] = value
    from mub.vnext.contracts.score import SystemScores
    return SystemScores(**values)


def score_audit(run, capabilities):
    values = {}
    for field in SCORE_LAYER_TYPES["audit_scores"].model_fields:
        value, _ = _compute_audit(f"audit_scores.{field}", run, capabilities)
        values[field] = value
    from mub.vnext.contracts.score import AuditScores
    return AuditScores(**values)


def publish_scores(
    output_dir: str | Path,
    scores: Sequence[ScoreRecord],
    summary: dict,
    run_manifest: RunManifest,
) -> RunManifest:
    import json

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_path = output / "scores.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "run_manifest.json"
    score_bytes = b"".join(canonical_json_bytes(score) + b"\n" for score in scores)
    summary_bytes = json.dumps(
        summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    score_artifacts = (
        {
            "path": "scores.jsonl",
            "sha256": hashlib.sha256(score_bytes).hexdigest(),
            "media_type": "application/jsonl",
            "record_count": len(scores),
        },
        {
            "path": "summary.json",
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "media_type": "application/json",
            "record_count": 1,
        },
    )
    payload = run_manifest.model_dump(mode="python")
    payload["score_artifacts"] = score_artifacts
    updated_manifest = RunManifest.model_validate(payload)
    manifest_bytes = canonical_json_bytes(updated_manifest) + b"\n"
    publish_files_atomically(
        {score_path: score_bytes, summary_path: summary_bytes, manifest_path: manifest_bytes},
        overwrite=True,
    )
    return updated_manifest


load_pilot_inputs = authenticate_pilot_files
score_pilot = score_pilot_records


__all__ = [
    "PilotInputBundle", "authenticate_pilot_files", "load_pilot_inputs", "publish_scores", "score_actions",
    "score_answers", "score_audit", "score_pilot", "score_pilot_records", "score_retrieval", "score_state",
    "score_store", "score_system",
]
