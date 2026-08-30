from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, Split
from mub.vnext.contracts.v3.runtime import TaskRunRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.post_core_artifacts import (
    POST_CORE_ARTIFACT_NAMES,
    validate_post_core_artifact_tree,
)
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.jsonl import read_models
from mub.vnext.contracts.v3.common import typed_json_equal
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_tasks_v3


_DEFAULT_CANDIDATE_ROOT = _PROJECT_ROOT / "data" / "vnext" / "main_track_v1_independence_v1"
_DIAGNOSTIC_FILENAME = "oracle_diagnostic.json"
_EXPECTED_TEST_TASK_COUNT = 720


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    root: Path
    tasks: tuple[MemUpdateTaskV3, ...]
    test_tasks: tuple[MemUpdateTaskV3, ...]
    artifact_hashes: dict[str, str]
    validation_report: dict[str, Any]
    family_counts: dict[str, int]
    domain_counts: dict[str, int]
    attribute_counts: dict[str, int]
    language_counts: dict[str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(root / name)
        for name in POST_CORE_ARTIFACT_NAMES
    }


def _axis_counts(tasks: Sequence[MemUpdateTaskV3], field: str) -> dict[str, int]:
    values = []
    for task in tasks:
        value = task.task_family if field == "family" else task.metadata.extra.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"test task axis {field} must be a nonblank string")
        values.append(value)
    return dict(sorted(Counter(values).items()))


def select_test_tasks(tasks: Sequence[MemUpdateTaskV3]) -> tuple[MemUpdateTaskV3, ...]:
    selected = tuple(task for task in tasks if task.metadata.split is Split.TEST)
    if len(selected) != _EXPECTED_TEST_TASK_COUNT:
        raise ValueError(
            f"candidate must contain exactly {_EXPECTED_TEST_TASK_COUNT} test tasks; got {len(selected)}"
        )
    if len({task.task_id for task in selected}) != len(selected):
        raise ValueError("candidate test task IDs must be unique")
    return selected


def load_candidate(root: str | Path) -> CandidateSnapshot:
    candidate_root = Path(root).resolve(strict=True)
    validation_report = validate_post_core_artifact_tree(candidate_root)
    if validation_report.get("valid") is not True:
        raise ValueError("candidate validation report is not valid")
    if validation_report.get("review_status") != "NOT_STARTED":
        raise ValueError("candidate review_status must be NOT_STARTED")

    task_path = candidate_root / "tasks.jsonl"
    tasks = tuple(read_models(task_path, MemUpdateTaskV3, id_field="task_id"))
    test_tasks = select_test_tasks(tasks)
    return CandidateSnapshot(
        root=candidate_root,
        tasks=tasks,
        test_tasks=test_tasks,
        artifact_hashes=_artifact_hashes(candidate_root),
        validation_report=validation_report,
        family_counts=_axis_counts(test_tasks, "family"),
        domain_counts=_axis_counts(test_tasks, "domain"),
        attribute_counts=_axis_counts(test_tasks, "attribute"),
        language_counts=_axis_counts(test_tasks, "language"),
    )


def _assert_candidate_unchanged(candidate: CandidateSnapshot) -> None:
    current = _artifact_hashes(candidate.root)
    if current != candidate.artifact_hashes:
        raise ValueError("candidate artifact bytes or hashes changed during oracle publication")


def _runtime_task(task: MemUpdateTaskV3) -> MemUpdateTaskV3:
    """Bind the candidate's action surface to the v3 parser without editing it."""
    actions = {action.action_id: action for action in task.actions}
    events = []
    for event in task.events:
        if len(event.gold_action_ids) != 1:
            raise ValueError("oracle requires exactly one action per event")
        action = actions[event.gold_action_ids[0]]
        if action.operation.value == "NOOP":
            normalized = "No memory object changes."
        elif action.operation.value in {"ADD", "UPDATE"}:
            targets = ",".join(key.canonical_id for key in action.target_object_keys)
            value = json.dumps(
                action.value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            normalized = f"{action.operation.value.title()} {targets} with value {value}."
        else:
            raise ValueError(f"oracle does not support candidate operation {action.operation.value}")
        events.append(event.model_copy(update={"normalized_text": normalized}))
    return task.model_copy(update={"events": tuple(events)})


def execute_reference_oracle(
    tasks: Sequence[MemUpdateTaskV3],
    *,
    run_id: str = "main-track-v1-reference-oracle",
) -> tuple[TaskRunRecordV3, ...]:
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("oracle requires at least one task")
    runtime_tasks = tuple(_runtime_task(task) for task in task_list)
    run_config = RuntimeConfigV3(
        run_id=run_id,
        retrieval_policy="normal_topk",
        answer_mode="slot_direct",
        retrieval_k=10,
        capture_snapshots=True,
    )
    return execute_tasks_v3(
        runtime_tasks,
        adapter_factory=lambda task: ReferenceAdapterV3(
            task,
            retrieval_policy="normal_topk",
        ),
        run_config=run_config,
    )


def verify_task_record(
    task: MemUpdateTaskV3,
    record: TaskRunRecordV3,
) -> tuple[str, ...]:
    issues: list[str] = []
    if record.task_id != task.task_id:
        issues.append("task_id_mismatch")
    if record.completion_status is not CompletionStatus.COMPLETED:
        issues.append(f"completion_status:{record.completion_status.value}")
    if len(record.parsed_actions) != len(task.events):
        issues.append("incomplete_action_coverage")
    if len(record.retrieval_traces) != len(task.queries):
        issues.append("incomplete_retrieval_coverage")
    query_ids = tuple(query.query_id for query in task.queries)
    prediction_by_id = {prediction.query_id: prediction for prediction in record.answer_predictions}
    retrieval_ids = tuple(trace.query_id for trace in record.retrieval_traces)
    if tuple(prediction_by_id) != query_ids:
        issues.append("query_answer_coverage_mismatch")
    if retrieval_ids != query_ids:
        issues.append("query_retrieval_coverage_mismatch")

    evidence_by_id = {evidence.query_id: evidence for evidence in task.gold_evidence}
    for query_id in query_ids:
        prediction = prediction_by_id.get(query_id)
        evidence = evidence_by_id[query_id]
        if prediction is None:
            continue
        expected_disposition = evidence.disposition or AnswerDisposition.ANSWERED
        if prediction.disposition is not expected_disposition:
            issues.append(f"{query_id}:disposition_mismatch")
            continue
        if expected_disposition is AnswerDisposition.ABSTAINED:
            if prediction.parsed_answer is not None:
                issues.append(f"{query_id}:abstention_has_answer")
        elif not typed_json_equal(prediction.parsed_answer, evidence.answer):
            issues.append(f"{query_id}:typed_answer_mismatch")
    return tuple(issues)


def build_oracle_diagnostic(
    candidate: CandidateSnapshot,
    records: Sequence[TaskRunRecordV3],
) -> dict[str, Any]:
    record_list = tuple(records)
    if tuple(record.task_id for record in record_list) != tuple(task.task_id for task in candidate.test_tasks):
        raise ValueError("oracle records must cover selected test tasks exactly and in order")
    failures = {
        task.task_id: verify_task_record(task, record)
        for task, record in zip(candidate.test_tasks, record_list, strict=True)
    }
    failed_task_ids = sorted(task_id for task_id, issues in failures.items() if issues)
    return {
        "schema_version": "memupdatebench.main-track-v1-oracle-diagnostic.v1",
        "evidence_class": "dataset_oracle_diagnostic",
        "review_status": "NOT_STARTED",
        "claim_boundary": "deterministic_reference_replay_only; not model or external-system evidence",
        "candidate_artifact_hashes": dict(candidate.artifact_hashes),
        "test_task_count": len(candidate.test_tasks),
        "family_counts": dict(candidate.family_counts),
        "domain_counts": dict(candidate.domain_counts),
        "attribute_counts": dict(candidate.attribute_counts),
        "language_counts": dict(candidate.language_counts),
        "pass_count": len(candidate.test_tasks) - len(failed_task_ids),
        "fail_count": len(failed_task_ids),
        "failed_task_ids": failed_task_ids,
        "runtime": {
            "adapter_id": "reference",
            "answer_mode": "slot_direct",
            "retrieval_policy": "normal_topk",
            "executor": "execute_task_v3",
        },
    }


def _canonical_diagnostic_bytes(diagnostic: dict[str, Any]) -> bytes:
    return json.dumps(
        diagnostic,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_staged_diagnostic(staged: Path, expected: bytes) -> None:
    if staged.read_bytes() != expected:
        raise ValueError("staged oracle diagnostic bytes changed")


def publish_oracle_diagnostic(
    candidate: CandidateSnapshot,
    diagnostic: dict[str, Any],
    output_root: str | Path,
    *,
    before_publish: Callable[[], None] | None = None,
) -> Path:
    output = Path(output_root)
    resolved_output = output.resolve(strict=False)
    if resolved_output == candidate.root or candidate.root in resolved_output.parents:
        raise ValueError("oracle output must not overlap candidate root")
    if output.is_symlink():
        raise ValueError("oracle output root must not be a symlink")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("oracle output root must be empty for no-replace publication")

    _assert_candidate_unchanged(candidate)
    destination = output / _DIAGNOSTIC_FILENAME
    expected = _canonical_diagnostic_bytes(diagnostic)

    def guard() -> None:
        if before_publish is not None:
            before_publish()
        _assert_candidate_unchanged(candidate)

    output_preexisted = output.exists()
    try:
        publish_files_atomically(
            {destination: expected},
            overwrite=False,
            source_paths=tuple(candidate.root / name for name in POST_CORE_ARTIFACT_NAMES),
            validators={destination: lambda staged: _validate_staged_diagnostic(staged, expected)},
            pre_publish=guard,
        )
    except BaseException:
        if not output_preexisted and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise
    if destination.read_bytes() != expected:
        raise RuntimeError("published oracle diagnostic bytes differ from canonical payload")
    return destination


def run_main_track_oracle(
    candidate_root: str | Path = _DEFAULT_CANDIDATE_ROOT,
    output_root: str | Path | None = None,
    *,
    run_id: str = "main-track-v1-reference-oracle",
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if output_root is None:
        raise ValueError("output_root is required")
    output = Path(output_root)
    if output.exists() and output.is_dir() and any(output.iterdir()):
        raise FileExistsError("oracle output root must be empty for no-replace publication")
    candidate = load_candidate(candidate_root)
    records = execute_reference_oracle(candidate.test_tasks, run_id=run_id)
    diagnostic = build_oracle_diagnostic(candidate, records)
    publish_oracle_diagnostic(candidate, diagnostic, output, before_publish=before_publish)
    return diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic vNext main-track dataset oracle without model or network calls"
    )
    parser.add_argument("--candidate-root", type=Path, default=_DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="main-track-v1-reference-oracle")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        diagnostic = run_main_track_oracle(
            args.candidate_root,
            args.output_root,
            run_id=args.run_id,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"main-track oracle failed: {exc}", file=sys.stderr)
        return 2
    print(_canonical_diagnostic_bytes(diagnostic).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
