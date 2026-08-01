from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mub.vnext.audit.cases import export_case
from mub.vnext.contracts.manifest import RunManifest, TaskManifest
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.contracts.score import SCORE_LAYER_TYPES, ScoreRecord
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.io.jsonl import read_models
from mub.vnext.scoring.aggregate import aggregate_scores
from mub.vnext.scoring.pilot import authenticate_pilot_files

MAX_CASES = 256
_OUTPUTS = (
    "summary.json",
    "summary.csv",
    "failure_breakdown.json",
    "capability_coverage.json",
    "cases.jsonl",
    "artifact_index.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize authenticated MemUpdateBench vNext Pilot results.")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--task-runs", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-policy", choices=("all", "failures", "stratified"), default="stratified")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(path: Path, refs, label: str) -> str:
    matches = [ref for ref in refs if Path(ref.path) == path or Path(ref.path).name == path.name]
    if len(matches) != 1:
        raise ValueError(f"{label} is not uniquely referenced by its manifest")
    digest = _sha256_file(path)
    if matches[0].sha256 != digest:
        raise ValueError(f"{label} hash mismatch")
    return digest


def _load_scores(
    path: Path,
    tasks: set[str],
    manifest: RunManifest,
    runs: tuple[TaskRunRecord, ...],
) -> tuple[ScoreRecord, ...]:
    # Typed JSONL is intentional: this rejects P6/P8 legacy result JSON.
    scores = tuple(read_models(path, ScoreRecord, id_field="task_id"))
    if not scores:
        raise ValueError("score set is empty")
    if {score.task_id for score in scores} != tasks:
        raise ValueError("score task set is incomplete or unexpected")
    runs_by_task = {run.task_id: run for run in runs}
    for score in scores:
        run = runs_by_task.get(score.task_id)
        if run is None or (
            score.run_id,
            score.adapter_id,
            score.completion_status,
        ) != (
            run.run_id,
            run.adapter_id,
            run.completion_status,
        ):
            raise ValueError("score/run record mismatch")
        if score.run_id != manifest.run_id or score.adapter_id != manifest.adapter_info.adapter_id:
            raise ValueError("score/run record mismatch")
    return tuple(sorted(scores, key=lambda item: item.task_id))


def _select_cases(scores: tuple[ScoreRecord, ...], tasks: dict[str, Any], policy: str) -> tuple[ScoreRecord, ...]:
    ordered = tuple(sorted(scores, key=lambda score: score.task_id))
    if policy == "all":
        return ordered[:MAX_CASES]
    if policy == "failures":
        return tuple(score for score in ordered if score.failure_flags)[:MAX_CASES]
    cells: dict[tuple[str, str, str], list[ScoreRecord]] = {}
    for score in ordered:
        task = tasks[score.task_id]
        cell = (task.task_family, task.difficulty.value, score.adapter_id)
        cells.setdefault(cell, []).append(score)
    selected: dict[str, ScoreRecord] = {}
    for cell in sorted(cells):
        rows = cells[cell]
        correct = next((row for row in rows if not row.failure_flags), None)
        failure = next((row for row in rows if row.failure_flags), None)
        for row in (correct, failure):
            if row is not None:
                selected[row.task_id] = row
        if correct is None and failure is None and rows:
            selected[rows[0].task_id] = rows[0]
    # Small cells are represented by one correct and one failure when available;
    # the fixed maximum keeps audit bundles reviewable and reproducible.
    return tuple(selected[key] for key in sorted(selected)[:MAX_CASES])


def _capability_coverage(scores: tuple[ScoreRecord, ...], manifest: RunManifest) -> dict[str, Any]:
    fields = {path for score in scores for path in score.supported_metric_fields}
    reasons = Counter(
        support.reason.value
        for score in scores
        for support in score.supported_metric_fields.values()
    )
    audit = {}
    for field in SCORE_LAYER_TYPES["audit_scores"].model_fields:
        path = f"audit_scores.{field}"
        audit[field] = {
            "supported": sum(getattr(score.audit_scores, field) is not None for score in scores),
            "total": len(scores),
            "support_reason": next((score.supported_metric_fields[path].reason.value for score in scores if path in score.supported_metric_fields), None),
        }
    return {
        "run_id": manifest.run_id,
        "adapter_id": manifest.adapter_info.adapter_id,
        "system_name": manifest.adapter_info.system_name,
        "adapter_capabilities": manifest.adapter_capabilities.model_dump(mode="json"),
        "supported_metric_field_count": len(fields),
        "unsupported_metric_reasons": dict(sorted(reasons.items())),
        "audit_fields": audit,
    }


def _failure_breakdown(scores: tuple[ScoreRecord, ...]) -> dict[str, Any]:
    flags = Counter(flag.value if hasattr(flag, "value") else flag for score in scores for flag in score.failure_flags)
    primary = Counter(score.primary_failure for score in scores if score.primary_failure)
    return {"task_count": len(scores), "failure_flags": dict(sorted(flags.items())), "primary_failures": dict(sorted(primary.items()))}


def _csv_bytes(scores: tuple[ScoreRecord, ...], tasks: dict[str, Any]) -> bytes:
    rows: list[list[str]] = [["task_id", "task_family", "difficulty", "method", "completion_status", "primary_failure", "failure_flags"]]
    for score in scores:
        task = tasks[score.task_id]
        rows.append([
            score.task_id, task.task_family, task.difficulty.value, score.adapter_id,
            score.completion_status.value, score.primary_failure or "",
            ";".join(flag.value if hasattr(flag, "value") else flag for flag in score.failure_flags),
        ])
    output = __import__("io").StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")


def summarize_pilot(
    tasks_path: Path,
    task_manifest_path: Path,
    runs_path: Path,
    scores_path: Path,
    run_manifest_path: Path,
    output_dir: Path,
    case_policy: str = "stratified",
) -> None:
    bundle = authenticate_pilot_files(tasks_path, task_manifest_path, runs_path, run_manifest_path)
    tasks = {task.task_id: task for task in bundle.tasks}
    score_hash = _artifact_hash(scores_path, bundle.run_manifest.score_artifacts, "score file")
    scores = _load_scores(scores_path, set(tasks), bundle.run_manifest, bundle.runs)
    task_hash = _artifact_hash(tasks_path, bundle.task_manifest.task_file_paths_and_hashes, "task file")
    run_hash = _artifact_hash(runs_path, bundle.run_manifest.normalized_runtime_artifacts, "run-record file")
    aggregate = aggregate_scores(scores, bundle.tasks, bundle.run_manifest)
    selected = _select_cases(scores, tasks, case_policy)
    cases = []
    for score in selected:
        run = next(item for item in bundle.runs if item.task_id == score.task_id)
        case = export_case(task=tasks[score.task_id], run=run, score=score, task_manifest=bundle.task_manifest, run_manifest=bundle.run_manifest, task_artifact_hash=task_hash, run_artifact_hash=run_hash, score_artifact_hash=score_hash)
        case["artifacts"]["task_manifest_hash"] = _sha256_file(task_manifest_path)
        cases.append(case)
    summary = {
        **aggregate,
        "input_artifacts": {"tasks": task_hash, "task_manifest": _sha256_file(task_manifest_path), "task_runs": run_hash, "scores": score_hash, "run_manifest": _sha256_file(run_manifest_path)},
        "case_policy": {"name": case_policy, "selected": len(cases), "max_cases": MAX_CASES, "stratification": "one correct and one failure per available family/difficulty/method cell"},
    }
    failure_bytes = json.dumps(_failure_breakdown(scores), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    capability_bytes = json.dumps(_capability_coverage(scores, bundle.run_manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    summary_bytes = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    csv_bytes = _csv_bytes(scores, tasks)
    cases_bytes = b"".join(canonical_json_bytes_from_dict(case) + b"\n" for case in cases)
    payloads = {
        output_dir / "summary.json": summary_bytes,
        output_dir / "summary.csv": csv_bytes,
        output_dir / "failure_breakdown.json": failure_bytes,
        output_dir / "capability_coverage.json": capability_bytes,
        output_dir / "cases.jsonl": cases_bytes,
    }
    file_hashes = {path.name: hashlib.sha256(data).hexdigest() for path, data in payloads.items()}
    index = {"schema_version": bundle.run_manifest.schema_version, "run_id": bundle.run_manifest.run_id, "files": dict(sorted(file_hashes.items())), "inputs": summary["input_artifacts"]}
    index_bytes = json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    payloads[output_dir / "artifact_index.json"] = index_bytes
    publish_files_atomically(payloads, overwrite=True)


def canonical_json_bytes_from_dict(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summarize_pilot(args.tasks, args.task_manifest, args.task_runs, args.scores, args.run_manifest, args.output_dir, args.case_policy)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"vNext Pilot summary failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
