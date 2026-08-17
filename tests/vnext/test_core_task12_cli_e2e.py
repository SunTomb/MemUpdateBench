from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "vnext_run_core_task12.py"


def test_task12_single_run_cli_exposes_only_authenticated_bundle_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bundle-root" in completed.stdout
    assert "--core-root" in completed.stdout
    assert "--evidence-root" in completed.stdout
    assert "--model-snapshot" in completed.stdout
    assert "--execute" in completed.stdout
    assert "--fake-offline-answer" not in completed.stdout
    assert "--tree-manifest-sha256" not in completed.stdout
    assert "--task-manifest" not in completed.stdout
    assert "--run-config" not in completed.stdout
    assert "--output-root" not in completed.stdout
    assert "--provider" not in completed.stdout.lower()
    assert "--token" not in completed.stdout.lower()


def test_task12_single_run_cli_requires_explicit_execute_before_file_loading(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan",
            str(missing / "plan.json"),
            "--preparation-manifest",
            str(missing / "manifest.json"),
            "--bundle-root",
            str(missing / "bundle"),
            "--core-root",
            str(missing / "core"),
            "--evidence-root",
            str(missing / "evidence"),
            "--model-snapshot",
            str(missing / "snapshot"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "refusing to execute without --execute" in completed.stderr
    assert "FileNotFoundError" not in completed.stderr
