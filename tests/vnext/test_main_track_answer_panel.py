from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_main_track_answer_panel import (
    EXPECTED_TASK_COUNT,
    build_panel_summary,
    load_result_root,
    publish_panel,
)


ROOT = Path(__file__).resolve().parents[2]
QWEN_ROOT = ROOT / "results" / "vnext" / "main_track_qwen35_answer_test720_tang1_ee793b1_v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_model_fixture(tmp_path: Path, *, raw_leak: bool = False, binding_mismatch: bool = False) -> Path:
    source_rows = [json.loads(line) for line in (QWEN_ROOT / "rows.jsonl").read_bytes().splitlines()]
    rows = copy.deepcopy(source_rows)
    binding = copy.deepcopy(rows[0]["model_binding"])
    binding["model_id"] = "synthetic/answer-panel-model"
    for index, row in enumerate(rows):
        row["model_binding"] = copy.deepcopy(binding)
        if index == 0:
            continue
        if index == 1:
            row.update(
                {
                    "parsed_answer": "synthetic-wrong-answer",
                    "answer_outcome": "WRONG",
                    "exact_match": False,
                    "normalized_match": False,
                    "typed_match": False,
                    "typed_exact_match": False,
                    "answer_f1": 0.0,
                }
            )
        elif index == 2:
            row.update(
                {
                    "answer_disposition": "abstained",
                    "answer_format_valid": True,
                    "parsed_answer": None,
                    "answer_outcome": "WRONG_ABSTENTION",
                    "exact_match": False,
                    "normalized_match": False,
                    "typed_match": False,
                    "typed_exact_match": False,
                    "answer_f1": 0.0,
                }
            )
        elif index == 3:
            row.update(
                {
                    "answer_format_valid": False,
                    "answer_outcome": "FORMAT_INVALID",
                    "exact_match": False,
                    "normalized_match": False,
                    "typed_match": False,
                    "typed_exact_match": False,
                    "answer_f1": 0.0,
                }
            )
        elif index == 4:
            row.update(
                {
                    "answer_disposition": None,
                    "answer_format_valid": None,
                    "parsed_answer": None,
                    "answer_outcome": "UNAVAILABLE",
                    "exact_match": False,
                    "normalized_match": False,
                    "typed_match": False,
                    "typed_exact_match": False,
                    "answer_f1": 0.0,
                }
            )
        if raw_leak and index == 5:
            row["raw_output"] = "secret prompt completion"

    output = tmp_path / "model-b"
    output.mkdir()
    rows_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    summary = json.loads((QWEN_ROOT / "summary.json").read_bytes())
    summary["model_binding"] = copy.deepcopy(binding)
    summary["rows_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
    summary_bytes = _canonical(summary)
    index = json.loads((QWEN_ROOT / "artifact_index.json").read_bytes())
    index["model_binding"] = copy.deepcopy(binding)
    index["artifacts"]["rows.jsonl"] = {
        "sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "bytes": len(rows_bytes),
        "record_count": len(rows),
    }
    index["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
        "record_count": 1,
    }
    if binding_mismatch:
        mismatched_hashes = dict(index["candidate_artifact_hashes"])
        mismatched_hashes["tasks.jsonl"] = "0" * 64
        for row in rows:
            row["candidate_artifact_hashes"] = dict(mismatched_hashes)
        for payload in (summary, index):
            payload["candidate_artifact_hashes"] = dict(mismatched_hashes)
        rows_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
        summary["rows_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
        summary_bytes = _canonical(summary)
        index["artifacts"]["rows.jsonl"] = {
            "sha256": hashlib.sha256(rows_bytes).hexdigest(),
            "bytes": len(rows_bytes),
            "record_count": len(rows),
        }
        index["artifacts"]["summary.json"] = {
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "bytes": len(summary_bytes),
            "record_count": 1,
        }
    (output / "rows.jsonl").write_bytes(rows_bytes)
    (output / "summary.json").write_bytes(summary_bytes)
    (output / "artifact_index.json").write_bytes(_canonical(index))
    return output


def test_panel_summary_pairs_all_720_tasks_and_reports_answer_layers(tmp_path: Path) -> None:
    model_b = _write_model_fixture(tmp_path)
    model_a = load_result_root(QWEN_ROOT)
    model_b_snapshot = load_result_root(model_b)

    summary = build_panel_summary(model_a, model_b_snapshot)

    assert summary["paired_task_count"] == EXPECTED_TASK_COUNT == 720
    assert set(summary["models"]) == {"model_a", "model_b"}
    assert summary["models"]["model_a"]["model_binding"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert summary["models"]["model_b"]["model_binding"]["model_id"] == "synthetic/answer-panel-model"
    assert summary["paired_comparison"]["agreement"] == 716
    assert summary["paired_comparison"]["disagreement"] == 4
    assert summary["disagreement_categories"] == {
        "answer": 1,
        "abstention": 1,
        "format_unavailable": 2,
    }
    assert summary["models"]["model_a"]["per_family_em"]
    assert summary["models"]["model_a"]["per_language_em"]
    assert summary["models"]["model_a"]["per_domain_em"]
    assert summary["evidence_class"] == "answer_layer_panel_comparison"
    assert "external-manager" in summary["claim_boundary"]
    assert "statistical significance" in summary["claim_boundary"]


def test_panel_rejects_candidate_binding_mismatch(tmp_path: Path) -> None:
    model_b = _write_model_fixture(tmp_path, binding_mismatch=True)
    with pytest.raises(ValueError, match="candidate_artifact_hashes"):
        build_panel_summary(load_result_root(QWEN_ROOT), load_result_root(model_b))


def test_panel_rejects_raw_prompt_output_or_reasoning_fields(tmp_path: Path) -> None:
    model_b = _write_model_fixture(tmp_path, raw_leak=True)
    with pytest.raises(ValueError, match="raw prompt/output/reasoning"):
        load_result_root(model_b)


def test_panel_publication_is_exactly_two_artifacts_and_no_replace(tmp_path: Path) -> None:
    model_b = _write_model_fixture(tmp_path)
    output = tmp_path / "panel"
    publish_panel(QWEN_ROOT, model_b, output)
    assert {path.name for path in output.iterdir()} == {"panel_summary.json", "panel_index.json"}
    with pytest.raises(FileExistsError):
        publish_panel(QWEN_ROOT, model_b, output)
