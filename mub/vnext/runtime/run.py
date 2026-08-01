from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mub.vnext.contracts import AdapterCapabilities, AdapterInfo, ArtifactRef, RunManifest
from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.runtime import AnswerPrediction, TaskRunRecord
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash
from mub.vnext.io.jsonl import read_models
from mub.vnext.runtime.engine import RuntimeConfig, execute_task
from mub.vnext.runtime.resume import ResumeIndex, compute_run_identity


@dataclass(frozen=True)
class RunResult:
    run_identity: str
    rows: tuple[TaskRunRecord, ...]
    manifest: RunManifest | None
    manifest_path: Path
    progress_path: Path
    task_runs_path: Path


def normalize_answer_results(
    results,
    *,
    parsed_answers: Mapping[str, Any] | None = None,
    format_validity: Mapping[str, bool] | None = None,
    error_flags: Mapping[str, Iterable[str]] | None = None,
) -> list[AnswerPrediction]:
    from mub.vnext.runtime.engine import normalize_answer_result
    parsed_answers = parsed_answers or {}
    format_validity = format_validity or {}
    error_flags = error_flags or {}
    predictions: list[AnswerPrediction] = []
    seen: set[str] = set()
    for result in results:
        if result.query_id in seen:
            raise ValueError(f"duplicate answer result query_id: {result.query_id}")
        seen.add(result.query_id)
        predictions.append(normalize_answer_result(result, parsed_answer=parsed_answers.get(result.query_id), format_valid=format_validity.get(result.query_id), error_flags=error_flags.get(result.query_id, ())))
    return predictions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, media_type: str, record_count: int | None = None, root: Path | None = None, sha256: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        path=str(path.relative_to(root)) if root is not None and path.is_relative_to(root) else str(path),
        sha256=sha256 or _sha256_file(path),
        media_type=media_type,
        record_count=record_count,
    )


def _write_progress(path: Path, *, run_identity: str, expected_ids: list[str], rows: Iterable[TaskRunRecord]) -> None:
    status_ids = {status.value: [] for status in CompletionStatus}
    for row in rows:
        status_ids[row.completion_status.value].append(row.task_id)
    payload = {
        "run_identity": run_identity,
        "expected_ids": expected_ids,
        "expected_task_ids": expected_ids,
        "completed_ids": sorted(status_ids[CompletionStatus.COMPLETED.value]),
        "failed_ids": sorted(status_ids[CompletionStatus.FAILED.value]),
        "partial_ids": sorted(status_ids[CompletionStatus.PARTIAL.value]),
        "not_supported_ids": sorted(status_ids[CompletionStatus.NOT_SUPPORTED.value]),
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()


def _write_rows(path: Path, rows: Iterable[TaskRunRecord]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")
            handle.flush()


def _validate_task_runs_file(path: Path, expected_ids: list[str], expected_rows: Mapping[str, TaskRunRecord]) -> None:
    rows = list(read_models(path, TaskRunRecord, id_field="task_id"))
    if [row.task_id for row in rows] != expected_ids:
        raise ValueError("task-runs file IDs do not match expected task IDs")
    for row in rows:
        if row != expected_rows[row.task_id]:
            raise ValueError(f"task-runs file row mismatch for {row.task_id}")


def _adapter_identity(adapter: Any) -> tuple[AdapterInfo, AdapterCapabilities]:
    info = adapter.adapter_info()
    capabilities = adapter.capabilities()
    return (
        info if isinstance(info, AdapterInfo) else AdapterInfo.model_validate(info),
        capabilities if isinstance(capabilities, AdapterCapabilities) else AdapterCapabilities.model_validate(capabilities),
    )


def _finalize_manifest(
    *,
    output_dir: Path,
    run_config: RuntimeConfig,
    run_identity: str,
    task_manifest_hash: str,
    adapter_info: AdapterInfo,
    adapter_capabilities: AdapterCapabilities,
    expected_ids: list[str],
    rows: list[TaskRunRecord],
) -> RunManifest:
    task_runs_path = output_dir / "task_runs.jsonl"
    completed = sum(row.completion_status is CompletionStatus.COMPLETED for row in rows)
    unsupported = sum(row.completion_status is CompletionStatus.NOT_SUPPORTED for row in rows)
    failed = sum(row.completion_status in {CompletionStatus.FAILED, CompletionStatus.PARTIAL} for row in rows)
    if len(rows) != len(expected_ids) or {row.task_id for row in rows} != set(expected_ids):
        raise ValueError("cannot finalize run with missing or unexpected task IDs")
    if completed + failed + unsupported != len(expected_ids):
        raise ValueError("run status counts do not match expected task count")
    artifact = _artifact(task_runs_path, media_type="application/jsonl", record_count=len(rows), root=output_dir)
    return RunManifest(
        run_id=run_config.run_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        code_revision=run_config.compiler_version,
        dirty_state=False,
        task_manifest=ArtifactRef(path="task_manifest.json", sha256=task_manifest_hash, media_type="application/json"),
        adapter_info=adapter_info,
        adapter_capabilities=adapter_capabilities,
        capability_verification_artifact=None,
        model_name=None,
        provider=None,
        model_revision=None,
        prompt_config=run_config.prompt_config,
        decoding_config=run_config.decoding_config,
        seed_information={},
        action_parser_version=run_config.action_parser_version,
        answer_parser_version=run_config.answer_parser_version,
        memory_entry_extractor_version=run_config.memory_entry_extractor_version,
        object_value_extractor_config_hash=run_config.object_value_extractor_config_hash,
        redaction_policy_version=run_config.redaction_policy_version,
        environment_summary={"runtime": "mub.vnext.runtime"},
        package_summary={},
        expected_task_count=len(expected_ids),
        completed_task_count=completed,
        failed_task_count=failed,
        not_supported_task_count=unsupported,
        raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(artifact,),
        score_artifacts=(),
        native_vs_extracted_field_summary={"runtime_identity": run_identity},
    )


def run_tasks(
    tasks: Iterable[MemUpdateTask],
    adapter_factory: Callable[[MemUpdateTask], Any],
    run_config: RuntimeConfig,
    output_dir: str | Path,
    *,
    task_manifest_hash: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    retry_partial: bool = False,
    retry_not_supported: bool = False,
) -> RunResult:
    task_list = list(tasks)
    if not isinstance(run_config, RuntimeConfig):
        if hasattr(run_config, "model_dump"):
            run_config = RuntimeConfig(**run_config.model_dump(mode="python"))
        else:
            run_config = RuntimeConfig(**dict(run_config))
    if not task_list:
        raise ValueError("at least one task is required")
    task_ids = [task.task_id for task in task_list]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_runs_path = output / "task_runs.jsonl"
    progress_path = output / "progress.json"
    manifest_path = output / "run_manifest.json"
    if not resume:
        manifest_path.unlink(missing_ok=True)
    task_hash = task_manifest_hash or hashlib.sha256("".join(semantic_task_hash(task) for task in task_list).encode()).hexdigest()

    identity_adapter = adapter_factory(task_list[0])
    adapter_info, adapter_capabilities = _adapter_identity(identity_adapter)
    try:
        identity_adapter.close()
    except Exception:
        pass
    run_identity = compute_run_identity(
        task_manifest_hash=task_hash,
        adapter_info=adapter_info,
        adapter_capabilities=adapter_capabilities,
        runtime_config=run_config,
        retrieval_policy=run_config.retrieval_policy,
        answer_mode=run_config.answer_mode,
        prompt_config=run_config.prompt_config,
        decoding_config=run_config.decoding_config,
        schema_version=run_config.schema_version,
        compiler_version=run_config.compiler_version,
        profile_version=run_config.profile_version,
        output_dir=str(output),
    )
    run_config = replace(run_config, run_identity=run_identity)
    expected_hashes = {task.task_id: semantic_task_hash(task) for task in task_list}
    existing: dict[str, TaskRunRecord] = {}
    if resume and task_runs_path.exists():
        index = ResumeIndex.from_jsonl(task_runs_path, expected_task_ids=task_ids, run_identity=run_identity, expected_task_hashes=expected_hashes)
        existing.update(index.records)
    elif task_runs_path.exists():
        task_runs_path.unlink()

    decisions = {
        task.task_id: ResumeIndex(existing.values(), expected_task_ids=task_ids, run_identity=run_identity, expected_task_hashes=expected_hashes).decide(
            task.task_id,
            task_hash=expected_hashes[task.task_id],
            retry_failed=retry_failed,
            retry_partial=retry_partial,
            retry_not_supported=retry_not_supported,
        )
        for task in task_list
    }
    retried_ids = {task_id for task_id, decision in decisions.items() if decision.action == "retry"}
    rejected = [decision for decision in decisions.values() if decision.action == "reject"]
    if rejected:
        raise ValueError("resume rejected task: " + rejected[0].reason)
    if retried_ids:
        existing = {task_id: row for task_id, row in existing.items() if task_id not in retried_ids}
        _write_rows(task_runs_path, existing.values())
    elif not task_runs_path.exists():
        task_runs_path.touch()

    rows_by_id = dict(existing)
    _write_progress(progress_path, run_identity=run_identity, expected_ids=task_ids, rows=rows_by_id.values())
    try:
        for task in task_list:
            decision = decisions[task.task_id]
            if decision.action == "skip":
                continue
            adapter = adapter_factory(task)
            row = execute_task(task, adapter, run_config)
            rows_by_id[task.task_id] = row
            with task_runs_path.open("ab") as handle:
                handle.write(canonical_json_bytes(row))
                handle.write(b"\n")
                handle.flush()
            _write_progress(progress_path, run_identity=run_identity, expected_ids=task_ids, rows=rows_by_id.values())
    except BaseException:
        _write_progress(progress_path, run_identity=run_identity, expected_ids=task_ids, rows=rows_by_id.values())
        raise

    if set(rows_by_id) != set(task_ids):
        _write_progress(progress_path, run_identity=run_identity, expected_ids=task_ids, rows=rows_by_id.values())
        return RunResult(run_identity, tuple(rows_by_id.values()), None, manifest_path, progress_path, task_runs_path)
    rows = [rows_by_id[task_id] for task_id in task_ids]
    _validate_task_runs_file(task_runs_path, task_ids, rows_by_id)
    manifest = _finalize_manifest(output_dir=output, run_config=run_config, run_identity=run_identity, task_manifest_hash=task_hash, adapter_info=adapter_info, adapter_capabilities=adapter_capabilities, expected_ids=task_ids, rows=rows)
    with manifest_path.open("wb") as handle:
        handle.write(canonical_json_bytes(manifest))
        handle.write(b"\n")
        handle.flush()
    return RunResult(run_identity, tuple(rows), manifest, manifest_path, progress_path, task_runs_path)


run = run_tasks

__all__ = ["RunResult", "normalize_answer_results", "run", "run_tasks"]
