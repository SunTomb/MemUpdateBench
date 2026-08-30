from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.generation.post_core_artifacts import validate_post_core_artifact_tree


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_completed_audit(completed_root: Path | str, candidate_root: Path | str, selection_path: Path | str, source_packet_root: Path | str) -> dict:
    completed = Path(completed_root).resolve(strict=True)
    candidate = Path(candidate_root).resolve(strict=True)
    selection = Path(selection_path).resolve(strict=True)
    source_packet = Path(source_packet_root).resolve(strict=True)
    manifest_raw = (completed / "audit_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    receipt_raw = (completed / "review_receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    packet_raw = (completed / "audit_packet.jsonl").read_bytes()
    rows = [json.loads(line) for line in packet_raw.splitlines()]
    review_metadata_canonical = canonical(manifest) == manifest_raw and canonical(receipt) == receipt_raw
    if len(rows) != 240:
        raise ValueError("completed audit must contain exactly 240 rows")
    if manifest.get("review_status") != "PASS" or receipt.get("review_status") != "PASS":
        raise ValueError("completed audit status is not PASS")
    if manifest.get("benchmark_release_eligible") is not True or receipt.get("benchmark_release_eligible") is not True:
        raise ValueError("completed audit is not benchmark eligible")
    decisions = {row.get("audit_decision") for row in rows}
    if decisions != {"pass"}:
        raise ValueError("completed audit contains non-pass decisions")
    if any(row.get("issue_category") != "none" or not row.get("reviewer_id") or row.get("resolved_status") != "resolved" for row in rows):
        raise ValueError("completed audit rows have incomplete pass fields")
    packet_hash = hashlib.sha256(packet_raw).hexdigest()
    if packet_hash != manifest.get("completed_packet_sha256") or packet_hash != receipt.get("completed_packet_sha256"):
        raise ValueError("completed packet hash mismatch")
    source_hash = sha(source_packet / "audit_packet.jsonl")
    if source_hash != manifest.get("source_packet_sha256") or source_hash != receipt.get("source_packet_sha256"):
        raise ValueError("source packet hash mismatch")
    selection_hash = sha(selection)
    if selection_hash != manifest.get("selection_artifact_sha256") or selection_hash != receipt.get("selection_artifact_sha256"):
        raise ValueError("selection hash mismatch")
    candidate_report = validate_post_core_artifact_tree(candidate)
    expected_candidate = manifest.get("candidate_artifact_hashes")
    observed_candidate = {path.name: sha(path) for path in candidate.iterdir()}
    if observed_candidate != expected_candidate or receipt.get("candidate_artifact_hashes") != expected_candidate:
        raise ValueError("candidate artifact hashes do not match completed audit")
    return {
        "schema_version": "memupdatebench.main-track.audit-completion-attestation.v1",
        "release_id": "main_track_v1",
        "review_status": "PASS",
        "benchmark_release_eligible": True,
        "completed_audit_root": str(completed),
        "candidate_root": str(candidate),
        "selection_path": str(selection),
        "source_packet_root": str(source_packet),
        "completed_packet_sha256": packet_hash,
        "source_packet_sha256": source_hash,
        "selection_artifact_sha256": selection_hash,
        "candidate_artifact_hashes": expected_candidate,
        "row_count": 240,
        "decision_counts": {"pass": 240, "needs_revision": 0, "block": 0},
        "unresolved_count": 0,
        "automatic_candidate_validation": candidate_report,
        "reviewer_ids": manifest.get("reviewer_ids", receipt.get("reviewer_ids", [])),
        "review_policy_version": manifest.get("review_policy_version", receipt.get("review_policy_version")),
        "evidence_class": "human_audit_completion_attestation",
        "review_metadata_canonical": review_metadata_canonical,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a completed main-track human audit and publish an attestation.")
    parser.add_argument("--completed-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate_completed_audit(args.completed_root, args.candidate_root, args.selection, args.source_packet_root)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)
    with args.output.open("xb") as handle:
        handle.write(raw); handle.flush(); import os; os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
