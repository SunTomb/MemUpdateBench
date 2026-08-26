from __future__ import annotations

import json
from pathlib import Path

from scripts.vnext_preflight_langmem import run_preflight


def test_langmem_preflight_uses_isolated_python_and_writes_no_private_paths(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    isolated_python = (
        project_root / ".langmem-0.0.30-venv" / "Scripts" / "python.exe"
    )
    if not isolated_python.is_file():
        return

    payload = run_preflight(
        python_executable=isolated_python,
        run_prefix="langmem-real-preflight-test",
        timeout_seconds=30.0,
    )

    assert payload["passed"] is True
    assert payload["outcome"] == "pass"
    assert payload["identity"]["package_version"] == "0.0.30"
    assert payload["lifecycle"]["passed"] is True
    assert payload["namespace_reset_probe"]["passed"] is True
    payload_text = json.dumps(payload, sort_keys=True)
    assert project_root.as_posix() not in payload_text
    assert str(project_root).replace("\\", "\\\\") not in payload_text


def test_langmem_preflight_writer_accepts_canonical_dict_payload(
    tmp_path: Path,
) -> None:
    from scripts.vnext_preflight_langmem import _write_evidence

    output = tmp_path / "preflight.json"
    _write_evidence(output, {"passed": True, "outcome": "pass"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "outcome": "pass",
        "passed": True,
    }
