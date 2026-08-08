from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest

import mub.vnext.audit.core_stage as core_stage


ROOT = Path(__file__).resolve().parents[2]


def test_core_audit_clis_run_from_project_root_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for script in (
        "vnext_prepare_core_audit.py",
        "vnext_gate_core_audit.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
    prepare_help = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vnext_prepare_core_audit.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    assert "--candidate-dir" in prepare_help
    gate_help = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vnext_gate_core_audit.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    assert "--allow-bounded" not in gate_help


def test_missing_required_decisions_fail_but_optional_adjudications_may_be_absent(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        core_stage._read_review_rows(missing, required=True)
    assert core_stage._read_review_rows(missing, required=False) == ()


def test_review_jsonl_loader_rejects_noncanonical_and_duplicate_key_rows(
    tmp_path,
) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_bytes(b'{ "audit_id":"audit" }\n')
    with pytest.raises(ValueError, match="canonical"):
        core_stage._read_review_rows(path)
    path.write_bytes(b'{"audit_id":"audit","audit_id":"audit"}\n')
    with pytest.raises(ValueError, match="canonical"):
        core_stage._read_review_rows(path)
