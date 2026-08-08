from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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
    assert "--candidate-dir" in subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vnext_prepare_core_audit.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
