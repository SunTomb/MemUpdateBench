from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.v3.common import typed_json_equal
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, Operation, Split
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, TaskRunRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, QueryGoldEvidenceV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.generation.post_core_artifacts import POST_CORE_ARTIFACT_NAMES, validate_post_core_artifact_tree
from mub.vnext.io import read_models, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.runtime.answer_model_v3 import render_visible_prompt_v3
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_tasks_v3


RELEASE_ID = "memupdatebench.main-track.qwen35-answer-baseline.v1"
SCHEMA_VERSION = "memupdatebench.main-track.qwen35-answer-baseline.v1"
ROW_SCHEMA_VERSION = "memupdatebench.main-track.qwen35-answer-baseline.row.v1"
CANARY_SCOPE = "canary32"
TEST_SCOPE = "test720"
EXPECTED_TEST_COUNT = 720
CANARY_COUNT = 32
RETRIEVAL_K = 16
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE_SHA256 = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
MODEL_RUNTIME_RECEIPT_SHA256 = "5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f"
AUDIT_ATTESTATION_DEFAULT = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


@dataclass(frozen=True, slots=True)
class AuditAttestation:
    path: Path
    sha256: str
    review_status: str
    completed_packet_sha256: str
    source_packet_sha256: str
    selection_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    root: Path
    tasks: tuple[MemUpdateTaskV3, ...]
    test_tasks: tuple[MemUpdateTaskV3, ...]
    artifact_hashes: dict[str, str]

    def assert_unchanged(self) -> None:
        observed = _artifact_hashes(self.root)
        if observed != self.artifact_hashes:
            raise ValueError("candidate artifact hashes changed during answer-baseline publication")


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {name: _sha256_file(root / name) for name in POST_CORE_ARTIFACT_NAMES}


def _axis_counts(tasks: Sequence[MemUpdateTaskV3], axis: str) -> dict[str, int]:
    values = []
    for task in tasks:
        if axis == "family":
            value = task.task_family
        elif axis == "language":
            value = task.metadata.extra.get("language")
        else:
            value = task.metadata.extra.get(axis)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"task axis {axis} must be a nonblank string")
        values.append(value)
    return dict(sorted(Counter(values).items()))


def validate_audit_attestation(path: str | Path, candidate: CandidateSnapshot) -> AuditAttestation:
    attestation_path = Path(path).resolve(strict=True)
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise ValueError("audit attestation must be a regular file")
    raw = attestation_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("audit attestation is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("audit attestation must be an object")
    if _canonical_bytes(value) != raw:
        raise ValueError("audit attestation must use canonical JSON")
    if value.get("review_status") != "PASS":
        raise ValueError("audit attestation review_status must be PASS")
    if value.get("benchmark_release_eligible") is not True:
        raise ValueError("audit attestation benchmark_release_eligible must be true")
    for field in ("completed_packet_sha256", "source_packet_sha256", "selection_artifact_sha256"):
        if not isinstance(value.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise ValueError(f"audit attestation {field} is missing or invalid")
    declared = value.get("candidate_artifact_hashes")
    if declared != candidate.artifact_hashes:
        raise ValueError("audit attestation candidate artifact hashes do not match candidate")
    return AuditAttestation(
        path=attestation_path,
        sha256=_sha256(raw),
        review_status="PASS",
        completed_packet_sha256=value["completed_packet_sha256"],
        source_packet_sha256=value["source_packet_sha256"],
        selection_artifact_sha256=value["selection_artifact_sha256"],
    )


def load_candidate(root: str | Path) -> CandidateSnapshot:
    candidate_root = Path(root).resolve(strict=True)
    report = validate_post_core_artifact_tree(candidate_root)
    if report.get("valid") is not True:
        raise ValueError("candidate validation report is not valid")
    if report.get("review_status") != "NOT_STARTED":
        raise ValueError("candidate review_status must be NOT_STARTED")
    tasks = tuple(read_models(candidate_root / "tasks.jsonl", MemUpdateTaskV3, id_field="task_id"))
    test_tasks = tuple(task for task in tasks if task.metadata.split is Split.TEST)
    if len(test_tasks) != EXPECTED_TEST_COUNT:
        raise ValueError(f"candidate must contain exactly {EXPECTED_TEST_COUNT} test tasks")
    if len({task.task_id for task in test_tasks}) != len(test_tasks):
        raise ValueError("candidate test task IDs must be unique")
    return CandidateSnapshot(candidate_root, tasks, test_tasks, _artifact_hashes(candidate_root))


def select_tasks(tasks: Sequence[MemUpdateTaskV3] | CandidateSnapshot, scope: str) -> tuple[MemUpdateTaskV3, ...]:
    if scope not in {CANARY_SCOPE, TEST_SCOPE}:
        raise ValueError(f"unknown scope: {scope}")
    source = tasks.test_tasks if isinstance(tasks, CandidateSnapshot) else tuple(
        task for task in tasks if task.metadata.split is Split.TEST
    )
    if len(source) != EXPECTED_TEST_COUNT:
        raise ValueError(f"scope selection requires exactly {EXPECTED_TEST_COUNT} test tasks")
    return source[:CANARY_COUNT] if scope == CANARY_SCOPE else source


def _normal_form(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, (list, tuple)):
        return tuple(_normal_form(item) for item in value)
    if isinstance(value, dict):
        return {key: _normal_form(item) for key, item in value.items()}
    return value


def _answer_f1(predicted: Any, gold: Any, *, correct: bool) -> float:
    if correct:
        return 1.0
    if not isinstance(predicted, str) or not isinstance(gold, str):
        return 0.0
    predicted_tokens = predicted.casefold().split()
    gold_tokens = gold.casefold().split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_prediction(query, prediction: AnswerPredictionV3, gold: QueryGoldEvidenceV3) -> dict[str, Any]:
    expected = gold.disposition or AnswerDisposition.ANSWERED
    if expected is AnswerDisposition.ABSTAINED:
        correct = prediction.disposition is AnswerDisposition.ABSTAINED and prediction.format_valid
        outcome = "CORRECT_ABSTENTION" if correct else "WRONG_ABSTENTION"
        return {
            "expected_disposition": expected.value,
            "answer_outcome": outcome,
            "exact_match": correct,
            "normalized_match": correct,
            "typed_match": correct,
            "typed_exact_match": correct,
            "answer_f1": 1.0 if correct else 0.0,
        }
    if prediction.disposition is not AnswerDisposition.ANSWERED:
        return {
            "expected_disposition": expected.value,
            "answer_outcome": "UNAVAILABLE",
            "exact_match": False,
            "normalized_match": False,
            "typed_match": False,
            "typed_exact_match": False,
            "answer_f1": 0.0,
        }
    if not prediction.format_valid:
        return {
            "expected_disposition": expected.value,
            "answer_outcome": "FORMAT_INVALID",
            "exact_match": False,
            "normalized_match": False,
            "typed_match": False,
            "typed_exact_match": False,
            "answer_f1": 0.0,
        }
    typed = typed_json_equal(prediction.parsed_answer, gold.answer)
    normalized = _normal_form(prediction.parsed_answer) == _normal_form(gold.answer)
    return {
        "expected_disposition": expected.value,
        "answer_outcome": "CORRECT" if typed else "WRONG",
        "exact_match": typed,
        "normalized_match": normalized,
        "typed_match": typed,
        "typed_exact_match": typed,
        "answer_f1": _answer_f1(prediction.parsed_answer, gold.answer, correct=typed),
    }


def _runtime_task(task: MemUpdateTaskV3) -> MemUpdateTaskV3:
    actions = {action.action_id: action for action in task.actions}
    events = []
    for event in task.events:
        if len(event.gold_action_ids) != 1:
            raise ValueError("answer baseline requires exactly one action per event")
        action = actions[event.gold_action_ids[0]]
        if action.operation is Operation.NOOP:
            normalized = "No memory object changes."
        elif action.operation in {Operation.ADD, Operation.UPDATE}:
            targets = ",".join(key.canonical_id for key in action.target_object_keys)
            value = json.dumps(action.value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
            normalized = f"{action.operation.value.title()} {targets} with value {value}."
        else:
            raise ValueError(f"answer baseline does not support operation {action.operation.value}")
        events.append(event.model_copy(update={"normalized_text": normalized}))
    return task.model_copy(update={"events": tuple(events)})


def execute_reference_task(
    task: MemUpdateTaskV3,
    prompted_answer_model,
    *,
    run_id: str = "main-track-qwen35-answer-baseline",
) -> TaskRunRecordV3:
    runtime_task = _runtime_task(task)
    records = execute_tasks_v3(
        (runtime_task,),
        lambda item: ReferenceAdapterV3(item, retrieval_policy="normal_topk"),
        RuntimeConfigV3(
            run_id=run_id,
            retrieval_policy="normal_topk",
            answer_mode="slot_prompt",
            retrieval_k=RETRIEVAL_K,
            capture_snapshots=False,
        ),
        prompted_answer_model=prompted_answer_model,
    )
    return records[0]


class _RedactedHashMap(dict):
    """Expose the historical hash key to callers without serializing raw-output wording."""

    def __getitem__(self, key):
        aliases = {
            "raw_output_sha256": "model_output_sha256",
            "rendered_prompt_sha256": "visible_prompt_sha256",
            "rendered_chat_prompt_sha256": "chat_prompt_sha256",
        }
        actual = aliases.get(key, key)
        return super().__getitem__(actual)


def _metadata(model: Any) -> tuple[dict[str, Any], dict[str, str]]:
    raw = getattr(model, "last_answer_metadata", {})
    if not isinstance(raw, dict):
        raw = {}
    hashes = _RedactedHashMap()
    for key, value in raw.items():
        if str(key).endswith("_sha256") and type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value):
            safe_key = {
                "raw_output_sha256": "model_output_sha256",
                "rendered_prompt_sha256": "visible_prompt_sha256",
                "rendered_chat_prompt_sha256": "chat_prompt_sha256",
            }.get(str(key), str(key))
            hashes[safe_key] = value
    safe = _RedactedHashMap(hashes)
    for key in ("generated_tokens", "latency_ms"):
        value = raw.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            safe[key] = value
    return safe, hashes


def _model_binding(model_snapshot: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    binding = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tree_sha256": MODEL_TREE_SHA256,
        "snapshot": None if model_snapshot is None else str(Path(model_snapshot).resolve()),
        "runtime_receipt_sha256": MODEL_RUNTIME_RECEIPT_SHA256,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 64, "seed": 0},
    }
    binding.update(overrides)
    return binding


def build_row(
    task: MemUpdateTaskV3,
    record: TaskRunRecordV3,
    *,
    model: Any,
    candidate: CandidateSnapshot,
    model_binding: dict[str, Any] | None = None,
    runtime_binding: dict[str, Any] | None = None,
    audit_attestation: AuditAttestation | None = None,
) -> dict[str, Any]:
    trace = record.retrieval_traces[0] if record.retrieval_traces else None
    prediction = record.answer_predictions[0] if record.answer_predictions else None
    gold = task.gold_evidence[0]
    metadata, metadata_hashes = _metadata(model)
    fields: dict[str, Any] = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "task_id": task.task_id,
        "core_id": task.metadata.split_key.semantic_core_id,
        "semantic_core_id": task.metadata.split_key.semantic_core_id,
        "family": task.task_family,
        "domain": task.metadata.extra.get("domain"),
        "attribute": task.metadata.extra.get("attribute"),
        "language": task.metadata.extra.get("language"),
        "split": task.metadata.split.value,
        "status": "PASS" if record.completion_status is CompletionStatus.COMPLETED and trace and prediction else "FAIL",
        "completion_status": record.completion_status.value,
        "expected_disposition": (gold.disposition or AnswerDisposition.ANSWERED).value,
        "gold_answer": gold.answer,
        "answer_disposition": None if prediction is None else prediction.disposition.value,
        "answer_format_valid": None if prediction is None else prediction.format_valid,
        "parsed_answer": None if prediction is None else prediction.parsed_answer,
        "answer_error_flags": [] if prediction is None else list(prediction.error_flags),
        "answer_outcome": None,
        "exact_match": None,
        "normalized_match": None,
        "typed_match": None,
        "typed_exact_match": None,
        "answer_f1": None,
        "retrieval_trace_sha256": None if trace is None else _sha256(_canonical_bytes(trace)),
        "visible_prompt_sha256": None if trace is None else trace.prompt_hash,
        "answer_output_sha256": None if prediction is None else _sha256(prediction.raw_output.encode("utf-8")),
        "qwen_metadata": metadata,
        "qwen_metadata_hashes": metadata_hashes,
        "task_sha256": sha256_model(task),
        "candidate_artifact_hashes": dict(candidate.artifact_hashes),
        "model_binding": _model_binding(**(model_binding or {})),
        "runtime_binding": runtime_binding or {
            "adapter_id": "reference",
            "answer_mode": "slot_prompt",
            "retrieval_policy": "normal_topk",
            "retrieval_k": RETRIEVAL_K,
            "executor": "execute_tasks_v3",
        },
        "evidence_class": "answer_layer_reference_state",
        "audit_attestation_sha256": None if audit_attestation is None else audit_attestation.sha256,
        "audit_review_status": None if audit_attestation is None else audit_attestation.review_status,
        "error_sha256": None,
    }
    if prediction is not None:
        fields.update(score_prediction(task.queries[0], prediction, gold))
    elif record.exceptions:
        fields["error_sha256"] = _sha256(_canonical_bytes(record.exceptions))
    if scan_for_secrets(fields):
        raise ValueError("answer baseline row failed secret scan")
    return fields


def _axis_counts_rows(rows: Sequence[dict[str, Any]], axis: str) -> dict[str, int]:
    return dict(sorted(Counter(row[axis] for row in rows).items()))


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    scope: str,
    candidate: CandidateSnapshot,
    rows_sha256: str | None = None,
    model_binding: dict[str, Any] | None = None,
    audit_attestation: AuditAttestation | None = None,
) -> dict[str, Any]:
    expected = CANARY_COUNT if scope == CANARY_SCOPE else EXPECTED_TEST_COUNT
    if scope not in {CANARY_SCOPE, TEST_SCOPE} or not rows or len(rows) > expected:
        raise ValueError("summary rows do not match scope")
    answers = [row for row in rows if row.get("answer_outcome") is not None]
    outcome_names = ("CORRECT", "WRONG", "FORMAT_INVALID", "UNAVAILABLE", "CORRECT_ABSTENTION", "WRONG_ABSTENTION")
    outcomes = {name: sum(row.get("answer_outcome") == name for row in answers) for name in outcome_names}
    denominators = {
        "attempted": len(rows),
        "evaluable": len(answers),
        "answerable": sum(row.get("expected_disposition") == AnswerDisposition.ANSWERED.value for row in answers),
        "abstention": sum(row.get("expected_disposition") == AnswerDisposition.ABSTAINED.value for row in answers),
    }
    metrics = {}
    for name in ("exact_match", "normalized_match", "typed_match", "typed_exact_match", "answer_f1"):
        values = [row[name] for row in answers if row.get(name) is not None]
        metrics[name] = sum(float(value) for value in values) / len(values) if values else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "scope": scope,
        "evidence_class": "answer_layer_reference_state",
        "claim_boundary": "fixed ReferenceAdapterV3 state/retrieval plus offline Qwen prompted answering only; no external-system or extraction evidence",
        "rows": len(rows),
        "attempted_denominator": denominators["attempted"],
        "evaluable_denominator": denominators["evaluable"],
        "answerable_denominator": denominators["answerable"],
        "abstention_denominator": denominators["abstention"],
        "metrics": metrics,
        "exact_match": metrics["exact_match"],
        "normalized_match": metrics["normalized_match"],
        "typed_answer_match": metrics["typed_match"],
        "answer_em": metrics["exact_match"],
        "answer_normalized_em": metrics["normalized_match"],
        "answer_typed_em": metrics["typed_match"],
        "answer_f1": metrics["answer_f1"],
        "answer_outcome_counts": outcomes,
        "by_family": _axis_counts_rows(rows, "family"),
        "by_domain": _axis_counts_rows(rows, "domain"),
        "by_language": _axis_counts_rows(rows, "language"),
        "family_counts": _axis_counts_rows(rows, "family"),
        "domain_counts": _axis_counts_rows(rows, "domain"),
        "attribute_counts": _axis_counts_rows(rows, "attribute"),
        "language_counts": _axis_counts_rows(rows, "language"),
        "candidate_artifact_hashes": dict(candidate.artifact_hashes),
        "audit_attestation_sha256": None if audit_attestation is None else audit_attestation.sha256,
        "audit_review_status": None if audit_attestation is None else audit_attestation.review_status,
        "model_binding": _model_binding(**(model_binding or {})),
        "runtime_binding": {"adapter_id": "reference", "answer_mode": "slot_prompt", "retrieval_policy": "normal_topk", "retrieval_k": RETRIEVAL_K, "provider_calls": 0, "api_calls": 0, "network_calls": 0},
        "provider_calls": 0,
        "api_calls": 0,
        "network_calls": 0,
        "rows_sha256": rows_sha256,
        "runner_source_sha256": _sha256_file(Path(__file__)),
    }
    if scan_for_secrets(summary):
        raise ValueError("answer baseline summary failed secret scan")
    return summary


def _canonical_rows(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _validate_output(output: Path, candidate: CandidateSnapshot) -> None:
    if output.is_symlink():
        raise ValueError("answer baseline output root must not be a symlink")
    resolved = output.resolve(strict=False)
    if resolved == candidate.root or candidate.root in resolved.parents:
        raise ValueError("answer baseline output must not overlap candidate root")
    frozen = tuple((ROOT / "data" / "vnext" / name).resolve() for name in ("core", "pilot"))
    if any(resolved == item or item in resolved.parents for item in frozen):
        raise ValueError("answer baseline output must not overlap frozen release")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("answer baseline output root must be empty for no-replace publication")


def _validate_staged(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise ValueError("staged answer baseline artifact bytes changed")


def publish_answer_baseline(
    candidate: CandidateSnapshot,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_root: str | Path,
    *,
    before_publish: Callable[[], None] | None = None,
) -> Path:
    output = Path(output_root)
    _validate_output(output, candidate)
    candidate.assert_unchanged()
    if summary.get("candidate_artifact_hashes") != candidate.artifact_hashes:
        raise ValueError("summary candidate artifact hashes do not match candidate")
    rows_bytes = _canonical_rows(rows)
    bound_summary = dict(summary)
    bound_summary["rows_sha256"] = _sha256(rows_bytes)
    summary_bytes = _canonical_bytes(bound_summary)
    index = {
        "schema_version": "memupdatebench.main-track.qwen35-answer-baseline.artifact-index.v1",
        "release_id": RELEASE_ID,
        "scope": bound_summary.get("scope"),
        "evidence_class": "answer_layer_reference_state",
        "artifacts": {
            "rows.jsonl": {"sha256": _sha256(rows_bytes), "bytes": len(rows_bytes), "record_count": len(rows)},
            "summary.json": {"sha256": _sha256(summary_bytes), "bytes": len(summary_bytes), "record_count": 1},
        },
        "candidate_artifact_hashes": dict(candidate.artifact_hashes),
        "audit_attestation_sha256": bound_summary.get("audit_attestation_sha256"),
        "audit_review_status": bound_summary.get("audit_review_status"),
        "model_binding": bound_summary.get("model_binding"),
        "runtime_binding": bound_summary.get("runtime_binding"),
        "provider_calls": 0,
        "api_calls": 0,
        "network_calls": 0,
        "runner_source_sha256": _sha256_file(Path(__file__)),
    }
    index_bytes = _canonical_bytes(index)
    destinations = {output / "rows.jsonl": rows_bytes, output / "summary.json": summary_bytes, output / "artifact_index.json": index_bytes}

    def guard() -> None:
        if before_publish is not None:
            before_publish()
        candidate.assert_unchanged()

    validators = {path: (lambda staged, expected=payload: _validate_staged(staged, expected)) for path, payload in destinations.items()}
    publish_files_atomically(
        destinations,
        overwrite=False,
        source_paths=tuple(candidate.root / name for name in POST_CORE_ARTIFACT_NAMES),
        validators=validators,
        pre_publish=guard,
    )
    candidate.assert_unchanged()
    return output


def run(
    candidate_root: str | Path,
    output_root: str | Path,
    *,
    scope: str = TEST_SCOPE,
    model_snapshot: str | Path | None = None,
    model_factory: Callable[[], Any] | None = None,
    audit_attestation_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = load_candidate(candidate_root)
    audit_attestation = validate_audit_attestation(
        AUDIT_ATTESTATION_DEFAULT if audit_attestation_path is None else audit_attestation_path,
        candidate,
    )
    selected = select_tasks(candidate, scope)
    if model_factory is None:
        if model_snapshot is None:
            raise ValueError("model_snapshot is required for production execution")
        from scripts.vnext_run_letta_qwen_prompted_answer import QwenSession
        model_factory = lambda: QwenSession(Path(model_snapshot))
    model = model_factory()
    load = getattr(model, "load", None)
    close = getattr(model, "close", None)
    if callable(load):
        load()
    rows: list[dict[str, Any]] = []
    try:
        for task in selected:
            record = execute_reference_task(task, model, run_id=f"main-track-answer-{scope}-{task.task_id}")
            rows.append(build_row(task, record, model=model, candidate=candidate, audit_attestation=audit_attestation, model_binding={"snapshot": None if model_snapshot is None else str(Path(model_snapshot).resolve())}))
    finally:
        if callable(close):
            close()
    summary = build_summary(rows, scope=scope, candidate=candidate, audit_attestation=audit_attestation, model_binding={"snapshot": None if model_snapshot is None else str(Path(model_snapshot).resolve())})
    publish_answer_baseline(candidate, rows, summary, output_root)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed-reference main-track Qwen prompted-answer baseline")
    parser.add_argument("--candidate-root", type=Path, default=ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", choices=(CANARY_SCOPE, TEST_SCOPE), default=TEST_SCOPE)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--audit-attestation", type=Path, default=AUDIT_ATTESTATION_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.candidate_root, args.output_root, scope=args.scope, model_snapshot=args.model_snapshot, audit_attestation_path=args.audit_attestation)
    except Exception as exc:
        print(f"answer baseline failed: {type(exc).__name__}: {re.sub(r'[^a-zA-Z0-9_. -]', '', str(exc))}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
