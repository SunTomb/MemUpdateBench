from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.generation.post_core_audit import (
    build_main_track_audit_packet,
    publish_main_track_audit_packet,
)


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data" / "vnext" / "main_track_v1_independence_v1"
SELECTION = ROOT / "results" / "vnext" / "main_track_v1_independence_audit_selection" / "selection.json"


def test_materialize_packet_contains_selected_rows_and_audit_blanks() -> None:
    packet = build_main_track_audit_packet(CANDIDATE, SELECTION)
    rows = [json.loads(line) for line in packet.packet_bytes.splitlines()]
    selection = json.loads(SELECTION.read_bytes())

    assert len(rows) == 240
    assert [row["task_id"] for row in rows] == selection["selected_task_ids"]
    assert packet.manifest["selection_artifact_hash"] == hashlib.sha256(SELECTION.read_bytes()).hexdigest()
    assert packet.manifest["candidate_artifact_hashes"] == selection["candidate_artifact_hashes"]
    assert packet.manifest["review_status"] == "NOT_STARTED"
    assert packet.manifest["policy_version"] == "post-core-data-audit-v1"
    assert packet.manifest["row_count"] == 240
    assert packet.manifest["packet_row_hash"] == hashlib.sha256(packet.packet_bytes).hexdigest()

    required = {
        "task_id", "core_id", "family", "domain", "attribute", "difficulty", "split",
        "surface_identity", "events", "actions", "target_objects", "queries",
        "gold_evidence", "version_history", "audit_decision", "issue_category",
        "reviewer_id", "review_note", "resolved_status",
    }
    assert required <= set(rows[0])
    assert all(row[name] == "" for name in ("audit_decision", "issue_category", "reviewer_id", "review_note", "resolved_status") for row in rows)
    assert all("raw_text" in event and "normalized_text" in event for row in rows for event in row["events"])
    assert all("role" in event and "metadata" in event for row in rows for event in row["events"])
    assert all("operation" in action and "target_object_keys" in action and "value" in action and "expected_effect" in action for row in rows for action in row["actions"])
    assert all("answer" in evidence and "supporting_event_ids" in evidence and "supporting_object_keys" in evidence for row in rows for evidence in row["gold_evidence"])


def test_publish_is_exact_three_file_no_replace_and_binds_hashes(tmp_path: Path) -> None:
    packet = build_main_track_audit_packet(CANDIDATE, SELECTION)
    output = tmp_path / "packet"

    published = publish_main_track_audit_packet(packet, output)

    assert tuple(path.name for path in published) == (
        "audit_packet.jsonl", "audit_manifest.json", "review_instructions.md"
    )
    assert {path.name for path in output.iterdir()} == {
        "audit_packet.jsonl", "audit_manifest.json", "review_instructions.md"
    }
    manifest = json.loads((output / "audit_manifest.json").read_bytes())
    assert manifest == packet.manifest
    assert hashlib.sha256((output / "audit_packet.jsonl").read_bytes()).hexdigest() == manifest["packet_row_hash"]
    with pytest.raises(FileExistsError):
        publish_main_track_audit_packet(packet, output)


def test_publish_rejects_candidate_tamper_before_install(tmp_path: Path) -> None:
    packet = build_main_track_audit_packet(CANDIDATE, SELECTION)
    output = tmp_path / "packet"
    task_path = CANDIDATE / "tasks.jsonl"
    original = task_path.read_bytes()

    def tamper() -> None:
        task_path.write_bytes(original + b"\n")

    try:
        with pytest.raises(ValueError, match="candidate"):
            publish_main_track_audit_packet(packet, output, before_publish=tamper)
        assert not output.exists() or not any(output.iterdir())
    finally:
        task_path.write_bytes(original)
