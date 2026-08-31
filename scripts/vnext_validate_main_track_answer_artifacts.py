from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import AnswerDisposition
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.generation.post_core_artifacts import POST_CORE_ARTIFACT_NAMES
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io import sha256_model
from scripts.vnext_run_main_track_answer_baseline import (
    AuditAttestation,
    CandidateSnapshot,
    EXPECTED_TEST_COUNT,
    MODEL_ID as QWEN_MODEL_ID,
    MODEL_REVISION as QWEN_MODEL_REVISION,
    MODEL_RUNTIME_RECEIPT_SHA256 as QWEN_RUNTIME_RECEIPT_SHA256,
    MODEL_TREE_SHA256 as QWEN_MODEL_TREE_SHA256,
    _answer_f1,
    _artifact_hashes,
    _normal_form,
    load_candidate,
    score_prediction,
    validate_audit_attestation,
)
from scripts.vnext_run_main_track_muse_answer_baseline import (
    MAX_TOKENS as MUSE_MAX_TOKENS,
    MUSE_LLAMA_CPP_COMMIT,
    MUSE_LLAMA_BINARY_SHA256,
    MUSE_MODEL_FILE,
    MUSE_MODEL_ID,
    MUSE_MODEL_REVISION,
    MUSE_MODEL_TREE_SHA256,
    validate_loopback_url,
)


TEST_SCOPE = "test720"
EXPECTED_TASK_COUNT = EXPECTED_TEST_COUNT
EVIDENCE_CLASS = "answer_layer_artifact_audit"
CLAIM_BOUNDARY = "Artifact integrity and answer-score recomputation only; no model or external-system significance claim."
AUDIT_SCHEMA_VERSION = "memupdatebench.main-track.answer-artifact-audit.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_OUTCOMES = (
    "CORRECT",
    "WRONG",
    "FORMAT_INVALID",
    "UNAVAILABLE",
    "CORRECT_ABSTENTION",
    "WRONG_ABSTENTION",
)
_DISPOSITIONS = frozenset({"answered", "abstained", "unavailable"})
_REQUIRED_ARTIFACTS = ("rows.jsonl", "summary.json", "artifact_index.json")
_REQUIRED_ROW_FIELDS = (
    "task_id",
    "core_id",
    "semantic_core_id",
    "family",
    "domain",
    "attribute",
    "language",
    "split",
    "status",
    "completion_status",
    "expected_disposition",
    "gold_answer",
    "answer_disposition",
    "answer_format_valid",
    "parsed_answer",
    "answer_outcome",
    "exact_match",
    "normalized_match",
    "typed_match",
    "typed_exact_match",
    "answer_f1",
    "task_sha256",
    "candidate_artifact_hashes",
    "audit_attestation_sha256",
    "model_binding",
)


class ValidationError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, label: str = "file") -> str:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a regular file")
    return sha256_bytes(path.read_bytes())


def _strict_load_json(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValidationError(f"{label} contains non-finite JSON numbers")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid JSON") from exc
    return value


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes() if not path.is_symlink() and path.is_file() else None
    if raw is None:
        raise ValidationError(f"{label} must be a regular file")
    value = _strict_load_json(raw, label)
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must contain an object")
    if canonical_bytes(value) != raw:
        raise ValidationError(f"{label} must use canonical JSON")
    return value, raw


def _contains_raw_field(value: Any, path: str = "root") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{path} has a non-string field name"
            lowered = key.casefold()
            is_hash = lowered.endswith("_sha256")
            if not is_hash and (
                lowered
                in {
                    "prompt",
                    "raw_prompt",
                    "rendered_prompt",
                    "rendered_chat_prompt",
                    "output",
                    "raw_output",
                    "generated_text",
                    "reasoning",
                    "reasoning_content",
                    "raw_reasoning",
                }
                or lowered.startswith("raw_prompt_")
                or lowered.startswith("raw_output_")
                or lowered.startswith("raw_reasoning_")
            ):
                return f"{path}.{key} is a raw prompt/output/reasoning field"
            found = _contains_raw_field(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _contains_raw_field(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _read_rows(path: Path) -> tuple[tuple[dict[str, Any], ...], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("rows.jsonl must be a regular file")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValidationError(f"rows.jsonl line {number} must use LF JSONL")
        payload = line[:-1]
        if not payload:
            raise ValidationError(f"rows.jsonl line {number} is blank")
        value = _strict_load_json(payload, f"rows.jsonl line {number}")
        if not isinstance(value, dict):
            raise ValidationError(f"rows.jsonl line {number} must contain an object")
        if canonical_bytes(value) != payload:
            raise ValidationError(f"rows.jsonl line {number} must use canonical JSON")
        found = _contains_raw_field(value, f"rows.jsonl line {number}")
        if found:
            raise ValidationError(found)
        rows.append(value)
    if len(rows) != EXPECTED_TEST_COUNT:
        raise ValidationError(f"rows.jsonl must contain exactly {EXPECTED_TEST_COUNT} rows")
    return tuple(rows), raw


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_hash_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"{label} must be a non-empty hash map")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _HEX64.fullmatch(item) is None:
            raise ValidationError(f"{label} contains an invalid SHA-256")
        result[key] = item
    return result


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _axis_counts(rows: Sequence[Mapping[str, Any]], axis: str) -> dict[str, int]:
    return dict(sorted(Counter(row[axis] for row in rows).items()))


def _summary_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("answer_outcome") is not None]
    metrics = {
        name: _mean(evaluable, name)
        for name in ("exact_match", "normalized_match", "typed_match", "typed_exact_match", "answer_f1")
    }
    outcomes = {name: sum(row.get("answer_outcome") == name for row in evaluable) for name in _OUTCOMES}
    denominators = {
        "attempted_denominator": len(rows),
        "evaluable_denominator": len(evaluable),
        "answerable_denominator": sum(row.get("expected_disposition") == "answered" for row in evaluable),
        "abstention_denominator": sum(row.get("expected_disposition") == "abstained" for row in evaluable),
    }
    aggregates: dict[str, Any] = {
        **denominators,
        "rows": len(rows),
        "metrics": metrics,
        "exact_match": metrics["exact_match"],
        "normalized_match": metrics["normalized_match"],
        "typed_answer_match": metrics["typed_match"],
        "answer_em": metrics["exact_match"],
        "answer_normalized_em": metrics["normalized_match"],
        "answer_typed_em": metrics["typed_match"],
        "answer_f1": metrics["answer_f1"],
        "answer_outcome_counts": outcomes,
    }
    for axis in ("family", "domain", "language"):
        aggregates[f"by_{axis}"] = _axis_counts(rows, axis)
    aggregates["family_counts"] = _axis_counts(rows, "family")
    aggregates["domain_counts"] = _axis_counts(rows, "domain")
    aggregates["attribute_counts"] = _axis_counts(rows, "attribute")
    aggregates["language_counts"] = _axis_counts(rows, "language")
    return aggregates


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None:
            if actual != expected:
                raise ValidationError(f"{label} does not match recomputed value")
        elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValidationError(f"{label} does not match recomputed value")
    elif actual != expected:
        raise ValidationError(f"{label} does not match recomputed value")


def _validate_model_binding(
    binding: Any,
    model_kind: str,
    *,
    runtime_receipt: Path | None,
) -> dict[str, Any]:
    if not isinstance(binding, dict) or not binding:
        raise ValidationError("model_binding must be a non-empty object")
    if model_kind == "qwen":
        if binding.get("model_id") != QWEN_MODEL_ID:
            raise ValidationError("Qwen model_id is not exact")
        if binding.get("revision") != QWEN_MODEL_REVISION:
            raise ValidationError("Qwen revision is not exact")
        if binding.get("tree_sha256") != QWEN_MODEL_TREE_SHA256:
            raise ValidationError("Qwen tree_sha256 is not exact")
        _require_sha(binding.get("tree_sha256"), "Qwen tree_sha256")
        receipt_sha = _require_sha(binding.get("runtime_receipt_sha256"), "Qwen runtime_receipt_sha256")
        if receipt_sha != QWEN_RUNTIME_RECEIPT_SHA256:
            raise ValidationError("Qwen runtime_receipt_sha256 is not exact")
    elif model_kind == "muse":
        required = {
            "model_id": MUSE_MODEL_ID,
            "revision": MUSE_MODEL_REVISION,
            "model_file": MUSE_MODEL_FILE,
            "tree_sha256": MUSE_MODEL_TREE_SHA256,
            "llama_cpp_commit": MUSE_LLAMA_CPP_COMMIT,
            "llama_binary_sha256": MUSE_LLAMA_BINARY_SHA256,
            "max_tokens": MUSE_MAX_TOKENS,
            "reasoning_mode": "off",
            "reasoning_storage": "sha256_only",
            "speculative_decoding": False,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "stream": False,
            "endpoint": "/v1/chat/completions",
        }
        for field, expected in required.items():
            if binding.get(field) != expected:
                raise ValidationError(f"Muse {field} is not exact")
        _require_sha(binding.get("tree_sha256"), "Muse tree_sha256")
        _require_sha(binding.get("llama_binary_sha256"), "Muse llama_binary_sha256")
        receipt_sha = _require_sha(binding.get("runtime_receipt_sha256"), "Muse runtime_receipt_sha256")
        server_url = binding.get("server_url")
        if not isinstance(server_url, str):
            raise ValidationError("Muse server_url must be a loopback endpoint")
        try:
            validate_loopback_url(server_url)
        except ValueError as exc:
            raise ValidationError("Muse server_url must be a loopback endpoint") from exc
        model_file = binding.get("model_file")
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValidationError("Muse model_file is required")
    else:
        raise ValidationError("model_kind must be qwen or muse")

    if runtime_receipt is not None:
        observed = sha256_file(runtime_receipt, "runtime receipt")
        if observed != receipt_sha:
            raise ValidationError("runtime receipt SHA-256 does not match model binding")
    return dict(binding)


def _expected_row_metadata(task: Any) -> dict[str, Any]:
    gold = task.gold_evidence[0]
    return {
        "task_id": task.task_id,
        "core_id": task.metadata.split_key.semantic_core_id,
        "semantic_core_id": task.metadata.split_key.semantic_core_id,
        "family": task.task_family,
        "domain": task.metadata.extra.get("domain"),
        "attribute": task.metadata.extra.get("attribute"),
        "language": task.metadata.extra.get("language"),
        "split": task.metadata.split.value,
        "expected_disposition": (gold.disposition or AnswerDisposition.ANSWERED).value,
        "gold_answer": gold.answer,
        "task_sha256": sha256_model(task),
    }


def _prediction_from_row(row: Mapping[str, Any], query_id: str) -> AnswerPredictionV3:
    disposition = row.get("answer_disposition")
    if disposition not in _DISPOSITIONS:
        raise ValidationError("row answer_disposition is invalid")
    format_valid = row.get("answer_format_valid")
    if type(format_valid) is not bool:
        raise ValidationError("row answer_format_valid must be boolean")
    return AnswerPredictionV3(
        query_id=query_id,
        raw_output="",
        disposition=AnswerDisposition(disposition),
        parsed_answer=row.get("parsed_answer"),
        format_valid=format_valid,
        error_flags=tuple(item for item in row.get("answer_error_flags", ()) if isinstance(item, str)),
    )


def _validate_row(
    row: Mapping[str, Any],
    number: int,
    task: Any,
    *,
    candidate_hashes: dict[str, str],
    audit_sha: str,
    model_binding: dict[str, Any],
) -> None:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValidationError(f"row {number} is missing fields: {', '.join(missing)}")
    if row.get("status") != "PASS" or row.get("completion_status") != "completed":
        raise ValidationError(f"row {number} status/completion_status is not PASS/completed")
    expected_metadata = _expected_row_metadata(task)
    for field, expected in expected_metadata.items():
        if row.get(field) != expected:
            raise ValidationError(f"row {number} {field} does not match candidate task")
    for field in ("task_id", "core_id", "semantic_core_id", "family", "domain", "attribute", "language", "split"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValidationError(f"row {number} {field} must be a nonblank string")
    if row.get("expected_disposition") not in {"answered", "abstained"}:
        raise ValidationError(f"row {number} expected_disposition is invalid")
    if row.get("answer_outcome") not in _OUTCOMES:
        raise ValidationError(f"row {number} answer_outcome is invalid")
    for field in ("exact_match", "normalized_match", "typed_match", "typed_exact_match"):
        if type(row.get(field)) is not bool:
            raise ValidationError(f"row {number} {field} must be boolean")
    f1 = row.get("answer_f1")
    if type(f1) not in (int, float) or isinstance(f1, bool) or not math.isfinite(float(f1)) or not 0 <= float(f1) <= 1:
        raise ValidationError(f"row {number} answer_f1 must be a finite number in [0, 1]")
    if row.get("candidate_artifact_hashes") != candidate_hashes:
        raise ValidationError(f"row {number} candidate_artifact_hashes do not match candidate")
    if row.get("audit_attestation_sha256") != audit_sha:
        raise ValidationError(f"row {number} audit_attestation_sha256 does not match attestation")
    if row.get("audit_review_status") != "PASS":
        raise ValidationError(f"row {number} audit_review_status must be PASS")
    if row.get("model_binding") != model_binding:
        raise ValidationError(f"row {number} model_binding does not match summary")
    prediction = _prediction_from_row(row, task.queries[0].query_id)
    expected_score = score_prediction(task.queries[0], prediction, task.gold_evidence[0])
    for field, expected in expected_score.items():
        _assert_equal(row.get(field), expected, f"row {number} {field}")


def _validate_artifact_metadata(index: Mapping[str, Any], filename: str, raw: bytes, record_count: int) -> None:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"rows.jsonl", "summary.json"}:
        raise ValidationError("artifact_index.json artifacts must bind rows.jsonl and summary.json exactly")
    metadata = artifacts.get(filename)
    if not isinstance(metadata, dict):
        raise ValidationError(f"artifact_index.json is missing {filename} metadata")
    if metadata.get("sha256") != sha256_bytes(raw):
        raise ValidationError(f"artifact_index.json {filename} hash does not match bytes")
    if metadata.get("bytes") != len(raw) or metadata.get("record_count") != record_count:
        raise ValidationError(f"artifact_index.json {filename} size/count does not match bytes")


def _load_result_artifacts(result_root: Path) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    if result_root.is_symlink() or not result_root.is_dir():
        raise ValidationError("result root must be a regular directory")
    for name in _REQUIRED_ARTIFACTS:
        path = result_root / name
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"{name} must be a regular file")
    rows, rows_bytes = _read_rows(result_root / "rows.jsonl")
    summary, summary_bytes = _read_canonical_json(result_root / "summary.json", "summary.json")
    index, index_bytes = _read_canonical_json(result_root / "artifact_index.json", "artifact_index.json")
    for label, value in (("summary.json", summary), ("artifact_index.json", index)):
        found = _contains_raw_field(value, label)
        if found:
            raise ValidationError(found)
    return rows, summary, index, rows_bytes, summary_bytes, index_bytes


def validate_result_root(
    result_root: str | Path,
    candidate_root: str | Path,
    audit_attestation: str | Path,
    *,
    model_kind: str,
    runtime_receipt: str | Path | None = None,
) -> dict[str, Any]:
    result = Path(result_root).resolve(strict=True)
    candidate = load_candidate(candidate_root)
    attestation = validate_audit_attestation(audit_attestation, candidate)
    receipt = None if runtime_receipt is None else Path(runtime_receipt).resolve(strict=True)
    rows, summary, index, rows_bytes, summary_bytes, index_bytes = _load_result_artifacts(result)

    if summary.get("scope") != TEST_SCOPE or index.get("scope") != TEST_SCOPE:
        raise ValidationError("result scope must be test720")
    if summary.get("evidence_class") != "answer_layer_reference_state" or index.get("evidence_class") != "answer_layer_reference_state":
        raise ValidationError("result evidence_class must be answer_layer_reference_state")
    if summary.get("rows") != EXPECTED_TEST_COUNT:
        raise ValidationError(f"summary rows must be exactly {EXPECTED_TEST_COUNT}")
    if summary.get("release_id") != index.get("release_id"):
        raise ValidationError("summary and artifact index release_id differ")
    _validate_artifact_metadata(index, "rows.jsonl", rows_bytes, EXPECTED_TEST_COUNT)
    _validate_artifact_metadata(index, "summary.json", summary_bytes, 1)
    if summary.get("rows_sha256") != sha256_bytes(rows_bytes):
        raise ValidationError("summary rows_sha256 does not match rows.jsonl")

    candidate_hashes = _require_hash_map(summary.get("candidate_artifact_hashes"), "summary candidate_artifact_hashes")
    if candidate_hashes != candidate.artifact_hashes:
        raise ValidationError("result candidate_artifact_hashes do not match candidate")
    if index.get("candidate_artifact_hashes") != candidate_hashes:
        raise ValidationError("artifact index candidate_artifact_hashes do not match summary")
    audit_sha = _require_sha(summary.get("audit_attestation_sha256"), "summary audit_attestation_sha256")
    if audit_sha != attestation.sha256:
        raise ValidationError("result audit_attestation_sha256 does not match supplied attestation")
    if index.get("audit_attestation_sha256") != audit_sha or summary.get("audit_review_status") != "PASS" or index.get("audit_review_status") != "PASS":
        raise ValidationError("result audit attestation binding is not PASS")

    model_binding = _validate_model_binding(summary.get("model_binding"), model_kind, runtime_receipt=receipt)
    if index.get("model_binding") != model_binding:
        raise ValidationError("artifact index model_binding does not match summary")

    for number, row in enumerate(rows, start=1):
        _validate_row(
            row,
            number,
            candidate.test_tasks[number - 1],
            candidate_hashes=candidate_hashes,
            audit_sha=audit_sha,
            model_binding=model_binding,
        )
    if len({row["task_id"] for row in rows}) != EXPECTED_TEST_COUNT:
        raise ValidationError("rows task IDs must be unique")

    aggregates = _summary_aggregates(rows)
    for field, expected in aggregates.items():
        if field not in summary:
            raise ValidationError(f"summary is missing aggregate field {field}")
        _assert_equal(summary[field], expected, f"summary {field}")

    checks = {
        "canonical_artifacts": 3,
        "row_count": len(rows),
        "ordered_task_match_count": len(rows),
        "metadata_match_count": len(rows),
        "task_hash_match_count": len(rows),
        "score_recomputed_count": len(rows),
        "raw_field_scan_count": len(rows) + 2,
        "summary_aggregate_field_count": len(aggregates),
        "artifact_index_field_count": 2,
        "candidate_artifact_count": len(candidate_hashes),
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "model_kind": model_kind,
        "scope": TEST_SCOPE,
        "result_root": str(result),
        "candidate_root": str(candidate.root),
        "audit_attestation": {
            "path": str(attestation.path),
            "sha256": attestation.sha256,
            "review_status": attestation.review_status,
            "benchmark_release_eligible": True,
            "completed_packet_sha256": attestation.completed_packet_sha256,
            "source_packet_sha256": attestation.source_packet_sha256,
            "selection_artifact_sha256": attestation.selection_artifact_sha256,
            "binding": "PASS",
        },
        "bindings": {
            "result_artifact_hashes": {
                "rows.jsonl": sha256_bytes(rows_bytes),
                "summary.json": sha256_bytes(summary_bytes),
                "artifact_index.json": sha256_bytes(index_bytes),
            },
            "candidate_artifact_hashes": candidate_hashes,
            "audit_attestation_sha256": audit_sha,
            "model_binding": model_binding,
            "runtime_receipt": None if receipt is None else {"path": str(receipt), "sha256": sha256_file(receipt, "runtime receipt")},
        },
        "checks": checks,
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "errors": [],
    }


def _validate_output_root(output_root: Path, result_root: Path, candidate_root: Path) -> None:
    if output_root.is_symlink():
        raise ValidationError("audit output root must not be a symlink")
    resolved = output_root.resolve(strict=False)
    for source in (result_root, candidate_root):
        if resolved == source or source in resolved.parents:
            raise ValidationError("audit output root must be separate from input roots")
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("audit output root must be empty for no-replace publication")


def _fail_report(
    result_root: str | Path,
    candidate_root: str | Path,
    audit_attestation: str | Path,
    model_kind: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "FAIL",
        "model_kind": model_kind,
        "scope": TEST_SCOPE,
        "result_root": str(Path(result_root).resolve(strict=False)),
        "candidate_root": str(Path(candidate_root).resolve(strict=False)),
        "audit_attestation": {"path": str(Path(audit_attestation).resolve(strict=False)), "sha256": None, "review_status": None, "benchmark_release_eligible": None, "completed_packet_sha256": None, "source_packet_sha256": None, "selection_artifact_sha256": None, "binding": "FAIL"},
        "bindings": {"result_artifact_hashes": None, "candidate_artifact_hashes": None, "audit_attestation_sha256": None, "model_binding": None, "runtime_receipt": None},
        "checks": {"canonical_artifacts": 0, "row_count": 0, "ordered_task_match_count": 0, "metadata_match_count": 0, "task_hash_match_count": 0, "score_recomputed_count": 0, "summary_aggregate_field_count": 0, "artifact_index_field_count": 0, "candidate_artifact_count": 0},
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "errors": [{"type": type(exc).__name__, "message": str(exc)}],
    }


def publish_answer_artifact_audit(
    result_root: str | Path,
    candidate_root: str | Path,
    audit_attestation: str | Path,
    *,
    model_kind: str,
    output_root: str | Path,
    runtime_receipt: str | Path | None = None,
) -> Path:
    result = Path(result_root).resolve(strict=False)
    candidate = Path(candidate_root).resolve(strict=False)
    output = Path(output_root)
    _validate_output_root(output, result, candidate)
    try:
        report = validate_result_root(
            result,
            candidate,
            audit_attestation,
            model_kind=model_kind,
            runtime_receipt=runtime_receipt,
        )
    except Exception as exc:
        report = _fail_report(result, candidate, audit_attestation, model_kind, exc)
    raw = canonical_bytes(report)
    destination = output / "answer_artifact_audit.json"
    def _validate_staged(path: Path) -> None:
        if path.read_bytes() != raw:
            raise ValidationError("staged audit bytes changed")

    publish_files_atomically(
        {destination: raw},
        overwrite=False,
        source_paths=tuple(candidate / name for name in POST_CORE_ARTIFACT_NAMES),
        validators={destination: _validate_staged},
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and publish a strict main-track answer artifact audit")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-attestation", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("qwen", "muse"), required=True)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = publish_answer_artifact_audit(
            args.result_root,
            args.candidate_root,
            args.audit_attestation,
            model_kind=args.model_kind,
            runtime_receipt=args.runtime_receipt,
            output_root=args.output_root,
        )
        report = json.loads((output / "answer_artifact_audit.json").read_bytes())
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
