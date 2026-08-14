from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_answer_harness_cli_exposes_only_local_model_inputs() -> None:
    script = ROOT / "scripts" / "vnext_run_core_answer_harness.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--slot" in completed.stdout
    assert "--model-snapshot" in completed.stdout
    assert "--dependency-path" in completed.stdout
    assert "--api" not in completed.stdout.lower()
    assert "--token" not in completed.stdout.lower()
    assert "--base-url" not in completed.stdout.lower()
