from pathlib import Path
import json
import pytest

from scripts.vnext_validate_main_track_audit_completion import validate_completed_audit

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
COMPLETED = PROJECT_ROOT / "results" / "vnext" / "main_track_v1_audit_completed_person1_person2"
CANDIDATE = WORKTREE_ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
SELECTION = WORKTREE_ROOT / "results" / "vnext" / "main_track_v1_audit_fix_selection" / "selection.json"
SOURCE = WORKTREE_ROOT / "results" / "vnext" / "main_track_v1_audit_fix_packet_v1"


def test_completed_audit_binds_current_candidate_and_passes() -> None:
    value = validate_completed_audit(COMPLETED, CANDIDATE, SELECTION, SOURCE)
    assert value["review_status"] == "PASS"
    assert value["benchmark_release_eligible"] is True
    assert value["decision_counts"] == {"pass": 240, "needs_revision": 0, "block": 0}
    assert value["unresolved_count"] == 0


def test_completed_audit_rejects_tampered_review(tmp_path: Path) -> None:
    for path in COMPLETED.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest = json.loads((tmp_path / "audit_manifest.json").read_bytes())
    manifest["review_status"] = "FAIL"
    (tmp_path / "audit_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_completed_audit(tmp_path, CANDIDATE, SELECTION, SOURCE)
