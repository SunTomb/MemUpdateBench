from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.vnext_package_cross_domain_qdrant_evidence import package_evidence
from scripts.vnext_validate_cross_domain_qdrant_state import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
COLLECTED = ROOT / "results/vnext/cross_domain_state_collected_20260930_v13"
BEA = ROOT / "data/vnext/family_h_cross_domain_bea_gdp/v1"
NOAA = ROOT / "data/vnext/family_h_cross_domain_noaa/v1"
EXPECTED_AUDIT = "5188cbbb7fbb3aacbf3a25994a75273060c9a2ab2857c180f3eacf7be3326eca"
EXPECTED_INDEX = "2b2b9dbfd7703f0a91cc66a9267344837f4a892bfac89e5e2db25a00b836a60a"


def test_validates_v13_collected_evidence_without_promoting_accuracy():
    report = validate_evidence(COLLECTED, BEA, NOAA)

    assert report["status"] == "VALID"
    assert report["collection_audit_sha256"] == EXPECTED_AUDIT
    assert report["artifact_index_sha256"] == EXPECTED_INDEX
    assert report["cluster_unit"] == "source_trajectory"
    assert report["cluster_n"] == 2
    assert report["metrics"] == {
        "state_step_matches": [20, 20],
        "final_state_matches": [4, 4],
        "retrieval_matches": [4, 4],
        "stale_same_slot": [0, 0],
    }
    assert report["scientific_release_allowed"] is False
    assert report["benchmark_accuracy_claimed"] is False
    assert report["answer_metrics"] is None


def test_package_is_no_replace_and_contains_only_bound_artifacts(tmp_path: Path):
    output = tmp_path / "package"
    result = package_evidence(COLLECTED, BEA, NOAA, output)

    assert result["status"] == "PUBLISHED"
    assert (output / "package_manifest.json").is_file()
    manifest = json.loads((output / "package_manifest.json").read_bytes())
    assert manifest["scientific_release_allowed"] is False
    assert manifest["benchmark_accuracy_claimed"] is False
    assert manifest["artifact_index_sha256"] == EXPECTED_INDEX
    assert not list(output.rglob("*.xlsx"))
    assert not list(output.rglob("*.pdf"))
    with pytest.raises(FileExistsError):
        package_evidence(COLLECTED, BEA, NOAA, output)


def test_package_rejects_nonapproved_validation(monkeypatch, tmp_path: Path):
    import scripts.vnext_package_cross_domain_qdrant_evidence as package_module
    monkeypatch.setattr(package_module, "validate_evidence", lambda *args: {"status": "WARNING"})
    with pytest.raises(ValueError, match="approved"):
        package_module.package_evidence(COLLECTED, BEA, NOAA, tmp_path / "package")


def test_package_public_allowlist_excludes_runtime_and_launcher_details(tmp_path: Path):
    output = tmp_path / "package"
    package_evidence(COLLECTED, BEA, NOAA, output)
    names = {path.name for path in output.iterdir()}
    assert "runtime.json" not in names
    assert "http_accounting.json" not in names
    assert "launch.json" not in names
    assert "supervisor.exitcode" not in names


def test_validator_cli_bootstraps_without_pythonpath():
    import os
    import subprocess
    import sys
    script = ROOT / "scripts/vnext_validate_cross_domain_qdrant_state.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(script), "--help"], env=env,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0


def test_package_rejects_reparse_parent(tmp_path: Path):
    import os
    parent = tmp_path / "parent"
    parent.mkdir()
    target = tmp_path / "real"
    target.mkdir()
    link = parent / "redirect"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink|reparse"):
        package_evidence(COLLECTED, BEA, NOAA, link / "package")


def test_package_rejects_input_mutation_between_validation_and_publish(tmp_path: Path, monkeypatch):
    import shutil
    import scripts.vnext_package_cross_domain_qdrant_evidence as package_module
    copied = tmp_path / "collected"
    shutil.copytree(COLLECTED, copied)
    original = package_module.validate_evidence
    def validate_then_mutate(*args):
        result = original(*args)
        path = copied / "results/input_binding.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(package_module, "validate_evidence", validate_then_mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        package_module.package_evidence(copied, BEA, NOAA, tmp_path / "package")


def test_validator_rejects_tampered_rows(tmp_path: Path):
    copied = tmp_path / "collected"
    copied.mkdir()
    for path in COLLECTED.iterdir():
        if path.is_dir():
            destination = copied / path.name
            destination.mkdir()
            for child in path.iterdir():
                destination.joinpath(child.name).write_bytes(child.read_bytes())
        else:
            copied.joinpath(path.name).write_bytes(path.read_bytes())
    rows = copied / "results/rows.jsonl"
    rows.write_bytes(rows.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact|hash|canonical"):
        validate_evidence(copied, BEA, NOAA)
