from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_task12_dry_run_cli_exposes_only_read_only_admission_inputs() -> None:
    script = ROOT / "scripts" / "vnext_prepare_core_task12.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout
    assert "--core-root" in completed.stdout
    assert "--evidence-root" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--execute" not in completed.stdout
    assert "--resume" not in completed.stdout
    assert "--token" not in completed.stdout.lower()
    assert "--device" not in completed.stdout.lower()


def test_task12_dry_run_cli_rejects_execute_flag(tmp_path) -> None:
    script = ROOT / "scripts" / "vnext_prepare_core_task12.py"
    before = tuple(tmp_path.iterdir())

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest", str(tmp_path / "manifest.json"),
            "--core-root", str(tmp_path / "core"),
            "--evidence-root", str(tmp_path / "evidence"),
            "--output-dir", str(tmp_path),
            "--execute",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments: --execute" in completed.stderr
    assert tuple(tmp_path.iterdir()) == before


def test_task12_dry_run_cli_imports_no_provider_or_runtime_engine() -> None:
    script = ROOT / "scripts" / "vnext_prepare_core_task12.py"
    code = (
        "import importlib.util, json, sys; "
        f"spec = importlib.util.spec_from_file_location('task12_cli', r'{script}'); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "print(json.dumps(sorted(sys.modules)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for module_name in (
        "mem0",
        "torch",
        "transformers",
        "mub.vnext.external.providers.mem0_adapter",
        "mub.vnext.runtime.engine_v3",
    ):
        assert module_name not in completed.stdout
