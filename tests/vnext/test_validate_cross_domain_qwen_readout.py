from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import vnext_validate_cross_domain_qwen_readout as validator


FIXTURES = Path(__file__).parent / "fixtures" / "cross_domain_qwen_blocked"


def test_validate_all_blocked_qwen_fixtures():
    reports = [validator.validate_root(FIXTURES / name) for name in ("v2", "v3", "v4", "tang1gpu1d")]
    assert [report["status"] for report in reports] == ["BLOCKED"] * 4
    assert [report["model_loads_completed"] for report in reports] == [0, 0, 0, 1]
    assert [report["generation_attempts"] for report in reports] == [0, 0, 0, 1]
    assert all(report["answer_metrics"] is None for report in reports)
    assert all(report["cleanup_ok"] for report in reports)


def test_validate_tang1_fixture_preserves_four_row_technical_boundary():
    report = validator.validate_root(FIXTURES / "tang1gpu1d")
    assert report["request_statuses"] == [
        "TECHNICAL_FAILURE", "NOT_RUN", "NOT_RUN", "NOT_RUN"
    ]
    assert report["generation_intents"] == 1
    assert report["generation_callback_entries"] == 1
    assert report["generation_attempts"] == 1
    assert report["answer_metrics"] is None
    assert report["redaction_ok"] is True


def test_validate_rejects_index_artifact_tamper(tmp_path):
    root = tmp_path / "fixture"
    import shutil
    shutil.copytree(FIXTURES / "v2", root)
    raw = (root / "summary.json").read_bytes()
    (root / "summary.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="artifact hash|artifact bytes"):
        validator.validate_root(root)


def test_publish_report_is_no_replace_and_reopenable(tmp_path):
    report = validator.validate_root(FIXTURES / "tang1gpu1d")
    output = tmp_path / "report"
    index = validator.publish_report(report, output)
    assert index["status"] == "BLOCKED"
    payload = json.loads((output / "validation.json").read_bytes())
    assert payload == report
    with pytest.raises(FileExistsError):
        validator.publish_report(report, output)
