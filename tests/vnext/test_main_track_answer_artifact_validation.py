from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.vnext_validate_main_track_answer_artifacts import (
    EXPECTED_TASK_COUNT,
    EVIDENCE_CLASS,
    CLAIM_BOUNDARY,
    canonical_bytes,
    publish_answer_artifact_audit,
    validate_result_root,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "vnext" / "main_track_qwen35_answer_test720_tang1_ee793b1_v1"
CANDIDATE_ROOT = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
AUDIT_ATTESTATION = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


def _copy_result_root(tmp_path: Path) -> Path:
    root = tmp_path / "result"
    root.mkdir()
    for name in ("rows.jsonl", "summary.json", "artifact_index.json"):
        (root / name).write_bytes((RESULT_ROOT / name).read_bytes())
    return root


def _rewrite_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def test_current_qwen_root_passes_and_publishes_no_replace_audit(tmp_path: Path) -> None:
    report = validate_result_root(
        RESULT_ROOT,
        CANDIDATE_ROOT,
        AUDIT_ATTESTATION,
        model_kind="qwen",
    )

    assert report["status"] == "PASS"
    assert report["scope"] == "test720"
    assert report["checks"]["row_count"] == EXPECTED_TASK_COUNT
    assert report["evidence_class"] == EVIDENCE_CLASS
    assert report["claim_boundary"] == CLAIM_BOUNDARY

    output = tmp_path / "audit"
    published = publish_answer_artifact_audit(
        RESULT_ROOT,
        CANDIDATE_ROOT,
        AUDIT_ATTESTATION,
        model_kind="qwen",
        output_root=output,
    )
    assert published == output
    assert {path.name for path in output.iterdir()} == {"answer_artifact_audit.json"}
    saved = json.loads((output / "answer_artifact_audit.json").read_bytes())
    assert saved["status"] == "PASS"
    assert saved["evidence_class"] == EVIDENCE_CLASS
    with pytest.raises(FileExistsError):
        publish_answer_artifact_audit(
            RESULT_ROOT,
            CANDIDATE_ROOT,
            AUDIT_ATTESTATION,
            model_kind="qwen",
            output_root=output,
        )


def test_row_score_mutation_is_rejected_even_when_artifact_hashes_are_rebound(tmp_path: Path) -> None:
    root = _copy_result_root(tmp_path)
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_bytes().splitlines()]
    rows[0]["parsed_answer"] = "tampered-answer"
    rows_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    (root / "rows.jsonl").write_bytes(rows_bytes)

    summary = json.loads((root / "summary.json").read_bytes())
    index = json.loads((root / "artifact_index.json").read_bytes())
    index["artifacts"]["rows.jsonl"] = {
        "sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "bytes": len(rows_bytes),
        "record_count": EXPECTED_TASK_COUNT,
    }
    summary["rows_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
    summary_bytes = canonical_bytes(summary)
    (root / "summary.json").write_bytes(summary_bytes)
    index["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
        "record_count": 1,
    }
    (root / "artifact_index.json").write_bytes(canonical_bytes(index))

    with pytest.raises(ValueError, match="score|match"):
        validate_result_root(root, CANDIDATE_ROOT, AUDIT_ATTESTATION, model_kind="qwen")


def test_summary_aggregate_mutation_is_rejected(tmp_path: Path) -> None:
    root = _copy_result_root(tmp_path)
    summary = json.loads((root / "summary.json").read_bytes())
    summary["answer_em"] = 0.0
    summary_bytes = canonical_bytes(summary)
    (root / "summary.json").write_bytes(summary_bytes)
    index = json.loads((root / "artifact_index.json").read_bytes())
    index["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
        "record_count": 1,
    }
    (root / "artifact_index.json").write_bytes(canonical_bytes(index))

    with pytest.raises(ValueError, match="summary"):
        validate_result_root(root, CANDIDATE_ROOT, AUDIT_ATTESTATION, model_kind="qwen")


def test_raw_prompt_output_or_reasoning_field_is_rejected(tmp_path: Path) -> None:
    root = _copy_result_root(tmp_path)
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_bytes().splitlines()]
    rows[0]["raw_output"] = "private completion"
    (root / "rows.jsonl").write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))

    with pytest.raises(ValueError, match="raw prompt/output/reasoning"):
        validate_result_root(root, CANDIDATE_ROOT, AUDIT_ATTESTATION, model_kind="qwen")


def test_model_kind_requires_exact_model_identity(tmp_path: Path) -> None:
    root = _copy_result_root(tmp_path)
    summary = json.loads((root / "summary.json").read_bytes())
    summary["model_binding"] = copy.deepcopy(summary["model_binding"])
    summary["model_binding"]["model_id"] = "Qwen/Qwen2.5-7B-Instruct"
    summary_bytes = canonical_bytes(summary)
    (root / "summary.json").write_bytes(summary_bytes)
    index = json.loads((root / "artifact_index.json").read_bytes())
    index["model_binding"] = copy.deepcopy(summary["model_binding"])
    index["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
        "record_count": 1,
    }
    (root / "artifact_index.json").write_bytes(canonical_bytes(index))
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_bytes().splitlines()]
    for row in rows:
        row["model_binding"] = copy.deepcopy(summary["model_binding"])
    rows_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    (root / "rows.jsonl").write_bytes(rows_bytes)
    summary["rows_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
    summary_bytes = canonical_bytes(summary)
    (root / "summary.json").write_bytes(summary_bytes)
    index["artifacts"]["rows.jsonl"] = {
        "sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "bytes": len(rows_bytes),
        "record_count": EXPECTED_TASK_COUNT,
    }
    index["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
        "record_count": 1,
    }
    (root / "artifact_index.json").write_bytes(canonical_bytes(index))

    with pytest.raises(ValueError, match="Qwen|model"):
        validate_result_root(root, CANDIDATE_ROOT, AUDIT_ATTESTATION, model_kind="qwen")


def test_muse_binding_is_rejected_without_full_identity_and_runtime_controls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Muse|model"):
        validate_result_root(RESULT_ROOT, CANDIDATE_ROOT, AUDIT_ATTESTATION, model_kind="muse")
