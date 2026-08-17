from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "vnext_prepare_core_task12_run.py"


def test_task12_run_bundle_cli_has_closed_preparation_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout
    assert "--plan" in completed.stdout
    assert "--core-root" in completed.stdout
    assert "--evidence-root" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--cell-id" in completed.stdout
    assert "--answer-model-slot" in completed.stdout
    assert "--output-leaf" in completed.stdout
    assert "--runtime-code" not in completed.stdout
    assert "--execute" not in completed.stdout
    assert "--token" not in completed.stdout.lower()
    assert "--provider" not in completed.stdout.lower()
    assert "--device" not in completed.stdout.lower()
