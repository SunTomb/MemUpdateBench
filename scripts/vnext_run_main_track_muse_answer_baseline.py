from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, Operation, Split
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, TaskRunRecordV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, QueryGoldEvidenceV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.generation.post_core_artifacts import POST_CORE_ARTIFACT_NAMES, validate_post_core_artifact_tree
from mub.vnext.io import read_models, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_tasks_v3
from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from scripts.vnext_run_main_track_answer_baseline import (
    AuditAttestation,
    CandidateSnapshot,
    _answer_f1,
    _artifact_hashes,
    _canonical_bytes,
    _normal_form,
    _runtime_task,
    _sha256,
    _sha256_file,
    score_prediction,
    validate_audit_attestation,
)

RELEASE_ID = "memupdatebench.main-track.muse-glimmer-gguf-answer-baseline.v1"
SCHEMA_VERSION = "memupdatebench.main-track.muse-glimmer-gguf-answer-baseline.v1"
ROW_SCHEMA_VERSION = "memupdatebench.main-track.muse-glimmer-gguf-answer-baseline.row.v1"
INDEX_SCHEMA_VERSION = "memupdatebench.main-track.muse-glimmer-gguf-answer-baseline.artifact-index.v1"
CANARY_SCOPE = "canary32"
TEST_SCOPE = "test720"
EXPECTED_TEST_COUNT = 720
CANARY_COUNT = 32
RETRIEVAL_K = 16
MUSE_MODEL_ID = "meta-models/Muse-Glimmer-30B-GGUF"
MUSE_MODEL_REVISION = "70bf1b61ac09f91b24d39038091b41c582bc5d7a"
MUSE_MODEL_FILE = "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf"
MUSE_MODEL_TREE_SHA256 = "55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f"
MUSE_LLAMA_CPP_COMMIT = "c1d0e7a004015f23bc0233470b747b596f29b264"
MUSE_LLAMA_BINARY_SHA256 = "feab512b206b6a5e1714f8099dfaca62ad62a92c5353f2fd522a1890f6a3f3d2"
MUSE_RUNTIME_RECEIPT_SHA256 = "3bae8741362aafe9d3ff11e2535c898c55ce1bcadf5112efde49de470e81acfe"
# Stable aliases make the binding easy to inspect without importing Qwen defaults.
MODEL_ID = MUSE_MODEL_ID
MODEL_REVISION = MUSE_MODEL_REVISION
MODEL_TREE_SHA256 = MUSE_MODEL_TREE_SHA256
RUNTIME_RECEIPT_SHA256 = MUSE_RUNTIME_RECEIPT_SHA256
MAX_TOKENS = 2048
AUDIT_ATTESTATION_DEFAULT = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


def canonical_bytes(value: Any) -> bytes:
    return _canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_loopback_url(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("server URL must be a nonblank string")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Muse server URL must be an uncredentialed HTTP loopback URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Muse server URL must target loopback")
    try:
        if parsed.port is None:
            raise ValueError("Muse server URL must include a port")
    except ValueError as exc:
        raise ValueError("Muse server URL has an invalid port") from exc
    return value.rstrip("/")


def muse_model_binding(*, server_url: str | None = None, model_file: str = MUSE_MODEL_FILE) -> dict[str, Any]:
    return {
        "model_id": MUSE_MODEL_ID,
        "revision": MUSE_MODEL_REVISION,
        "model_file": model_file,
        "tree_sha256": MUSE_MODEL_TREE_SHA256,
        "runtime_receipt_sha256": MUSE_RUNTIME_RECEIPT_SHA256,
        "llama_cpp_commit": MUSE_LLAMA_CPP_COMMIT,
        "llama_binary_sha256": MUSE_LLAMA_BINARY_SHA256,
        "speculative_decoding": False,
        "reasoning_storage": "sha256_only",
        "server_url": None if server_url is None else validate_loopback_url(server_url),
        "endpoint": "/v1/chat/completions",
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "old_32_token_smoke_status": "BLOCKED",
        "truncation_tests": [128, 256, 512, 1024, 1536],
        "qualification_note": "2048 is the smallest tested budget that removed all observed stratified-canary truncation; remaining abstention errors are model behavior",
    }


class MuseGlimmerAnswerModel:
    """Uncredentialed loopback llama.cpp OpenAI-compatible answer adapter."""

    def __init__(self, server_url: str, *, model_name: str = MUSE_MODEL_FILE, timeout: float = 120.0) -> None:
        self.server_url = validate_loopback_url(server_url)
        self.model_name = model_name
        self.timeout = timeout
        self.last_answer_metadata: dict[str, Any] = {}

    def answer(self, request) -> AnswerPredictionV3:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.rendered_prompt}],
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        body = canonical_bytes(payload)
        http_request = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                raw_response = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Muse loopback request failed: {type(exc).__name__}") from exc
        latency_ms = (time.monotonic() - started) * 1000
        try:
            response = json.loads(raw_response)
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
            if type(content) is not str:
                raise ValueError("Muse response message.content must be a string")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Muse response did not contain a valid final message.content") from exc
        reasoning = message.get("reasoning_content")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        safe_usage = {key: int(value) for key, value in usage.items() if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(value) is int and value >= 0}
        self.last_answer_metadata = {
            "content_sha256": sha256_bytes(content.encode("utf-8")),
            "reasoning_sha256": sha256_bytes(reasoning.encode("utf-8")) if isinstance(reasoning, str) else None,
            "generated_tokens": safe_usage.get("completion_tokens"),
            "prompt_tokens": safe_usage.get("prompt_tokens"),
            "total_tokens": safe_usage.get("total_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "latency_ms": latency_ms,
        }
        prediction = parse_answer_prediction_v3(
            query_id=request.query.query_id,
            answer_schema=request.query.answer_schema,
            raw_output=content,
        )
        return prediction.model_copy(update={"latency_ms": latency_ms, "usage": safe_usage})

    def close(self) -> None:
        return None


MuseSession = MuseGlimmerAnswerModel
MuseGlimmerSession = MuseGlimmerAnswerModel


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


def _bucket(task: MemUpdateTaskV3) -> str | None:
    if task.task_family == "interleaved_multi_slot_update":
        return "B"
    if task.task_family == "noop_write_discipline":
        return "D"
    if task.task_family == "entity_attribute_grounding":
        return "C_abstained" if (task.gold_evidence[0].disposition or AnswerDisposition.ANSWERED) is AnswerDisposition.ABSTAINED else "C_answered"
    return None


def _stratified_bucket(tasks: Sequence[MemUpdateTaskV3], key: str, count: int) -> list[MemUpdateTaskV3]:
    by_language: dict[str, list[MemUpdateTaskV3]] = {language: [] for language in ("en", "es", "ja")}
    for task in sorted(tasks, key=lambda item: item.task_id):
        language = task.metadata.extra.get("language")
        if language in by_language:
            by_language[language].append(task)
    if sum(len(items) for items in by_language.values()) < count:
        raise ValueError(f"insufficient tasks for canary stratum {key}")
    selected: list[MemUpdateTaskV3] = []
    languages = ("en", "es", "ja")
    cursor = 0
    while len(selected) < count:
        language = languages[cursor % len(languages)]
        if by_language[language]:
            selected.append(by_language[language].pop(0))
        cursor += 1
        if cursor > count * 10:
            raise ValueError(f"cannot balance canary stratum {key}")
    return selected


def select_tasks(tasks: Sequence[MemUpdateTaskV3] | CandidateSnapshot, scope: str) -> tuple[MemUpdateTaskV3, ...]:
    if scope not in {CANARY_SCOPE, TEST_SCOPE}:
        raise ValueError(f"unknown scope: {scope}")
    source = tasks.test_tasks if isinstance(tasks, CandidateSnapshot) else tuple(task for task in tasks if task.metadata.split is Split.TEST)
    if len(source) != EXPECTED_TEST_COUNT:
        raise ValueError(f"scope selection requires exactly {EXPECTED_TEST_COUNT} test tasks")
    if scope == TEST_SCOPE:
        return source
    groups: dict[str, list[MemUpdateTaskV3]] = {key: [] for key in ("B", "C_answered", "C_abstained", "D")}
    for task in source:
        key = _bucket(task)
        if key is not None:
            groups[key].append(task)
    selected = [item for key in groups for item in _stratified_bucket(groups[key], key, 8)]
    if len(selected) != CANARY_COUNT or len({task.task_id for task in selected}) != CANARY_COUNT:
        raise ValueError("Muse canary selection cardinality mismatch")
    return tuple(sorted(selected, key=lambda task: task.task_id))


def execute_reference_task(task: MemUpdateTaskV3, prompted_answer_model, *, run_id: str = "main-track-muse-answer-baseline") -> TaskRunRecordV3:
    runtime_task = _runtime_task(task)
    records = execute_tasks_v3(
        (runtime_task,),
        lambda item: ReferenceAdapterV3(item, retrieval_policy="normal_topk"),
        RuntimeConfigV3(run_id=run_id, retrieval_policy="normal_topk", answer_mode="slot_prompt", retrieval_k=RETRIEVAL_K, capture_snapshots=False),
        prompted_answer_model=prompted_answer_model,
    )
    return records[0]


def _metadata(model: Any) -> tuple[dict[str, Any], dict[str, str]]:
    raw = getattr(model, "last_answer_metadata", {})
    if not isinstance(raw, dict):
        raw = {}
    hashes = {key: value for key, value in raw.items() if key.endswith("_sha256") and type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)}
    safe = dict(hashes)
    for key in ("generated_tokens", "prompt_tokens", "total_tokens", "latency_ms"):
        value = raw.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            safe[key] = value
    if type(raw.get("finish_reason")) is str:
        safe["finish_reason"] = raw["finish_reason"]
    return safe, hashes


def build_row(task: MemUpdateTaskV3, record: TaskRunRecordV3, *, model: Any, candidate: CandidateSnapshot, model_binding: dict[str, Any] | None = None, runtime_binding: dict[str, Any] | None = None, audit_attestation: AuditAttestation | None = None) -> dict[str, Any]:
    trace = record.retrieval_traces[0] if record.retrieval_traces else None
    prediction = record.answer_predictions[0] if record.answer_predictions else None
    gold = task.gold_evidence[0]
    metadata, metadata_hashes = _metadata(model)
    row: dict[str, Any] = {
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
        "muse_metadata": metadata,
        "muse_metadata_hashes": metadata_hashes,
        "task_sha256": sha256_model(task),
        "candidate_artifact_hashes": dict(candidate.artifact_hashes),
        "model_binding": model_binding or muse_model_binding(),
        "runtime_binding": runtime_binding or {"adapter_id": "reference", "answer_mode": "slot_prompt", "retrieval_policy": "normal_topk", "retrieval_k": RETRIEVAL_K, "executor": "execute_tasks_v3"},
        "evidence_class": "answer_layer_reference_state",
        "audit_attestation_sha256": None if audit_attestation is None else audit_attestation.sha256,
        "audit_review_status": None if audit_attestation is None else audit_attestation.review_status,
        "error_sha256": None,
    }
    if prediction is not None:
        row.update(score_prediction(task.queries[0], prediction, gold))
    elif record.exceptions:
        row["error_sha256"] = _sha256(_canonical_bytes(record.exceptions))
    if scan_for_secrets(row):
        raise ValueError("Muse answer baseline row failed secret scan")
    return row


def _axis_counts(rows: Sequence[dict[str, Any]], axis: str) -> dict[str, int]:
    return dict(sorted(Counter(row[axis] for row in rows).items()))


def build_summary(rows: Sequence[dict[str, Any]], *, scope: str, candidate: CandidateSnapshot, rows_sha256: str | None = None, model_binding: dict[str, Any] | None = None, audit_attestation: AuditAttestation | None = None) -> dict[str, Any]:
    expected = CANARY_COUNT if scope == CANARY_SCOPE else EXPECTED_TEST_COUNT
    if scope not in {CANARY_SCOPE, TEST_SCOPE} or not rows or len(rows) > expected:
        raise ValueError("summary rows do not match scope")
    answers = [row for row in rows if row.get("answer_outcome") is not None]
    outcomes = {name: sum(row.get("answer_outcome") == name for row in answers) for name in ("CORRECT", "WRONG", "FORMAT_INVALID", "UNAVAILABLE", "CORRECT_ABSTENTION", "WRONG_ABSTENTION")}
    denominators = {"attempted": len(rows), "evaluable": len(answers), "answerable": sum(row.get("expected_disposition") == "answered" for row in answers), "abstention": sum(row.get("expected_disposition") == "abstained" for row in answers)}
    metrics = {name: (sum(float(row[name]) for row in answers if row.get(name) is not None) / len([row for row in answers if row.get(name) is not None]) if any(row.get(name) is not None for row in answers) else None) for name in ("exact_match", "normalized_match", "typed_match", "typed_exact_match", "answer_f1")}
    binding = model_binding or muse_model_binding()
    summary = {
        "schema_version": SCHEMA_VERSION, "release_id": RELEASE_ID, "scope": scope, "evidence_class": "answer_layer_reference_state",
        "claim_boundary": "fixed ReferenceAdapterV3 state/retrieval plus loopback Muse prompted answering only; no external-system or extraction evidence",
        "rows": len(rows), "attempted_denominator": denominators["attempted"], "evaluable_denominator": denominators["evaluable"], "answerable_denominator": denominators["answerable"], "abstention_denominator": denominators["abstention"],
        "metrics": metrics, "exact_match": metrics["exact_match"], "normalized_match": metrics["normalized_match"], "typed_answer_match": metrics["typed_match"], "answer_em": metrics["exact_match"], "answer_normalized_em": metrics["normalized_match"], "answer_typed_em": metrics["typed_match"], "answer_f1": metrics["answer_f1"], "answer_outcome_counts": outcomes,
        "by_family": _axis_counts(rows, "family"), "by_domain": _axis_counts(rows, "domain"), "by_language": _axis_counts(rows, "language"), "family_counts": _axis_counts(rows, "family"), "domain_counts": _axis_counts(rows, "domain"), "attribute_counts": _axis_counts(rows, "attribute"), "language_counts": _axis_counts(rows, "language"),
        "candidate_artifact_hashes": dict(candidate.artifact_hashes), "audit_attestation_sha256": None if audit_attestation is None else audit_attestation.sha256, "audit_review_status": None if audit_attestation is None else audit_attestation.review_status,
        "model_binding": binding, "runtime_binding": {"adapter_id": "reference", "answer_mode": "slot_prompt", "retrieval_policy": "normal_topk", "retrieval_k": RETRIEVAL_K, "provider_calls": 0, "api_calls": len(rows), "network_calls": len(rows)}, "provider_calls": 0, "api_calls": len(rows), "network_calls": len(rows), "rows_sha256": rows_sha256, "runner_source_sha256": _sha256_file(Path(__file__)),
        "qualification": {"old_32_token_smoke_status": "BLOCKED", "max_tokens": MAX_TOKENS, "max_tokens_reason": "128/256/512/1024/1536 truncation tests; 2048 removed all observed stratified-canary truncation", "truncation_tests": [128, 256, 512, 1024, 1536], "speculative_decoding": False, "reasoning_storage": "sha256_only"},
        "old_32_token_smoke_status": "BLOCKED", "max_tokens": MAX_TOKENS, "max_tokens_reason": "128/256/512/1024/1536 truncation tests; 2048 removed all observed stratified-canary truncation", "speculative_decoding": False, "reasoning_storage": "sha256_only",
    }
    if scan_for_secrets(summary):
        raise ValueError("Muse answer baseline summary failed secret scan")
    return summary


def _canonical_rows(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _validate_output(output: Path, candidate: CandidateSnapshot) -> None:
    if output.is_symlink():
        raise ValueError("Muse answer baseline output root must not be a symlink")
    resolved = output.resolve(strict=False)
    if resolved == candidate.root or candidate.root in resolved.parents:
        raise ValueError("Muse answer baseline output must not overlap candidate root")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Muse answer baseline output root must be empty for no-replace publication")


def publish_muse_answer_baseline(candidate: CandidateSnapshot, rows: Sequence[dict[str, Any]], summary: dict[str, Any], output_root: str | Path, *, before_publish: Callable[[], None] | None = None) -> Path:
    output = Path(output_root)
    _validate_output(output, candidate)
    candidate.assert_unchanged()
    if summary.get("candidate_artifact_hashes") != candidate.artifact_hashes:
        raise ValueError("summary candidate artifact hashes do not match candidate")
    rows_bytes = _canonical_rows(rows)
    bound_summary = dict(summary)
    bound_summary["rows_sha256"] = _sha256(rows_bytes)
    summary_bytes = _canonical_bytes(bound_summary)
    index = {"schema_version": INDEX_SCHEMA_VERSION, "release_id": RELEASE_ID, "scope": bound_summary.get("scope"), "evidence_class": "answer_layer_reference_state", "artifacts": {"rows.jsonl": {"sha256": _sha256(rows_bytes), "bytes": len(rows_bytes), "record_count": len(rows)}, "summary.json": {"sha256": _sha256(summary_bytes), "bytes": len(summary_bytes), "record_count": 1}}, "candidate_artifact_hashes": dict(candidate.artifact_hashes), "audit_attestation_sha256": bound_summary.get("audit_attestation_sha256"), "audit_review_status": bound_summary.get("audit_review_status"), "model_binding": bound_summary.get("model_binding"), "runtime_binding": bound_summary.get("runtime_binding"), "qualification": bound_summary.get("qualification"), "provider_calls": 0, "api_calls": len(rows), "network_calls": len(rows), "runner_source_sha256": _sha256_file(Path(__file__))}
    index_bytes = _canonical_bytes(index)
    destinations = {output / "rows.jsonl": rows_bytes, output / "summary.json": summary_bytes, output / "artifact_index.json": index_bytes}
    def guard() -> None:
        if before_publish is not None:
            before_publish()
        candidate.assert_unchanged()
    validators = {path: (lambda staged, expected=payload: (_ for _ in ()).throw(ValueError("staged Muse artifact bytes changed")) if staged.read_bytes() != expected else None) for path, payload in destinations.items()}
    publish_files_atomically(destinations, overwrite=False, source_paths=tuple(candidate.root / name for name in POST_CORE_ARTIFACT_NAMES), validators=validators, pre_publish=guard)
    candidate.assert_unchanged()
    return output


def run(candidate_root: str | Path, output_root: str | Path, *, scope: str = TEST_SCOPE, server_url: str | None = None, model_factory: Callable[[], Any] | None = None, audit_attestation_path: str | Path | None = None) -> dict[str, Any]:
    candidate = load_candidate(candidate_root)
    attestation = validate_audit_attestation(AUDIT_ATTESTATION_DEFAULT if audit_attestation_path is None else audit_attestation_path, candidate)
    selected = select_tasks(candidate, scope)
    if model_factory is None:
        if server_url is None:
            raise ValueError("server_url is required for production execution")
        model_factory = lambda: MuseGlimmerAnswerModel(server_url)
    model = model_factory()
    if callable(getattr(model, "load", None)):
        model.load()
    rows = []
    try:
        for task in selected:
            record = execute_reference_task(task, model, run_id=f"main-track-muse-answer-{scope}-{task.task_id}")
            rows.append(build_row(task, record, model=model, candidate=candidate, audit_attestation=attestation, model_binding=muse_model_binding(server_url=server_url)))
    finally:
        if callable(getattr(model, "close", None)):
            model.close()
    summary = build_summary(rows, scope=scope, candidate=candidate, audit_attestation=attestation, model_binding=muse_model_binding(server_url=server_url))
    publish_muse_answer_baseline(candidate, rows, summary, output_root)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed-reference Muse Glimmer GGUF prompted-answer baseline")
    parser.add_argument("--candidate-root", type=Path, default=ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", choices=(CANARY_SCOPE, TEST_SCOPE), default=TEST_SCOPE)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--audit-attestation", type=Path, default=AUDIT_ATTESTATION_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args.candidate_root, args.output_root, scope=args.scope, server_url=args.server_url, audit_attestation_path=args.audit_attestation)
    except Exception as exc:
        print(f"Muse answer baseline failed: {type(exc).__name__}: {re.sub(r'[^a-zA-Z0-9_. -]', '', str(exc))}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
