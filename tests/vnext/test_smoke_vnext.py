from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_vnext_contract_smoke_is_registered_and_self_contained() -> None:
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "smoke_test.py")],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "[PASS] vNext contracts, replay, serialization, and capability gating"
        in completed.stdout
    )
