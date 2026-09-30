"""Pure validation for the collected cross-domain Qdrant state/retrieval root."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mub.vnext.validation.replay_v3 import replay_task_v3
from scripts.vnext_promote_family_h_cross_domain_bea import read_release as read_bea_release
from scripts.vnext_promote_family_h_cross_domain_noaa import read_release as read_noaa_release
from scripts.vnext_run_cross_domain_state import (
    BEA_INDEX_SHA256,
    NOAA_INDEX_SHA256,
    _normalized_entries,
    _stable_row_signature,
    canonical_json_bytes,
    load_public_cells,
)

EXPECTED_ARTIFACT_INDEX_SHA256 = "2b2b9dbfd7703f0a91cc66a9267344837f4a892bfac89e5e2db25a00b836a60a"
EXPECTED_COLLECTION_AUDIT_SHA256 = "5188cbbb7fbb3aacbf3a25994a75273060c9a2ab2857c180f3eacf7be3326eca"
EXPECTED_SOURCE_MANIFEST_SHA256 = "145a19b059aa3373de39e09a2581865e9910816739c530da09c20e3deb9a8c01"
BEA_ROOT_PIN = BEA_INDEX_SHA256
NOAA_ROOT_PIN = NOAA_INDEX_SHA256
RESULT_FILES = (
    "summary.json",
    "rows.jsonl",
    "runtime.json",
    "http_accounting.json",
    "input_binding.json",
    "artifact_index.json",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} must be canonical JSON")
    return value, raw


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or canonical_json_bytes(item) != line + b"\n":
            raise ValueError(f"{label} contains non-canonical JSONL")
        rows.append(item)
    return rows, raw


def _validate_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = Path(root).resolve(strict=True)
    results = root / "results"
    index, index_raw = _read_json(results / "artifact_index.json", "artifact index")
    if _sha(index_raw) != EXPECTED_ARTIFACT_INDEX_SHA256:
        raise ValueError("artifact index hash mismatch")
    if index.get("schema") != "memupdatebench.cross-domain.qdrant-state-index.v1":
        raise ValueError("artifact index schema mismatch")
    if index.get("status") != "COMPLETE" or index.get("scientific_release_allowed") is not False:
        raise ValueError("artifact index boundary mismatch")
    payload: dict[str, bytes] = {}
    expected_paths = {row["path"] for row in index.get("artifacts", [])}
    if expected_paths != set(RESULT_FILES) - {"artifact_index.json"}:
        raise ValueError("artifact index membership mismatch")
    for row in index["artifacts"]:
        path = results / row["path"]
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or _sha(raw) != row["sha256"]:
            raise ValueError(f"artifact hash mismatch: {row['path']}")
        payload[row["path"]] = raw
    payload["artifact_index.json"] = index_raw
    return index, payload


def validate_evidence(collected_root: Path | str, bea_root: Path | str,
                      noaa_root: Path | str) -> dict[str, Any]:
    root = Path(collected_root).resolve(strict=True)
    index, payload = _validate_artifacts(root)
    summary, _ = _read_json(root / "results/summary.json", "summary")
    audit_raw = (root / "collection_audit.json").read_bytes()
    audit = json.loads(audit_raw)
    if _sha(audit_raw) != EXPECTED_COLLECTION_AUDIT_SHA256:
        raise ValueError("collection audit hash mismatch")
    if audit.get("status") != "COMPLETE":
        raise ValueError("collection audit is not complete")
    if summary.get("status") != "COMPLETE" or summary.get("error") is not None:
        raise ValueError("summary is not complete")
    if summary.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest binding mismatch")
    if summary.get("release_bindings") != {"bea": BEA_ROOT_PIN, "noaa": NOAA_ROOT_PIN}:
        raise ValueError("release binding mismatch")
    if summary.get("scientific_release_allowed") is not False or summary.get("benchmark_accuracy_claimed") is not False:
        raise ValueError("scientific boundary mismatch")
    if summary.get("answer_metrics") is not None or summary.get("generations") != 0 or summary.get("model_loads") != 0:
        raise ValueError("answer/model boundary mismatch")
    if not all(summary.get("cleanup", {}).get(key) is True for key in ("processes_stopped", "directory_removed", "port_closed")):
        raise ValueError("cleanup boundary mismatch")

    releases = {
        "bea": (Path(bea_root).resolve(strict=True), BEA_ROOT_PIN),
        "noaa": (Path(noaa_root).resolve(strict=True), NOAA_ROOT_PIN),
    }
    cells = {cell["release_name"]: cell for cell in load_public_cells(releases)}
    rows, _ = _read_jsonl(root / "results/rows.jsonl", "rows")
    if len(rows) != 4:
        raise ValueError("row count mismatch")
    seen = set()
    state_matches = final_matches = retrieval_matches = source_links = stale = 0
    per_source: dict[str, dict[str, int]] = {}
    signatures: dict[str, list[bytes]] = {}
    for row in rows:
        name = row.get("release_name")
        repetition = row.get("repetition_index")
        key = (name, repetition)
        if key in seen or name not in cells or repetition not in (0, 1):
            raise ValueError("row identity mismatch")
        seen.add(key)
        cell = cells[name]
        if row.get("task_id") != cell["task_id"] or row.get("task_sha256") != cell["task_sha256"]:
            raise ValueError("row task binding mismatch")
        if row.get("status") != "COMPLETE" or len(row.get("steps", [])) != len(cell["events"]):
            raise ValueError("row completion mismatch")
        if not all(row.get("cleanup", {}).get(key) is True for key in ("reset_empty", "client_closed", "collection_deleted")):
            raise ValueError("row cleanup mismatch")
        source_state = per_source.setdefault(name, {"state": 0, "final": 0, "retrieval": 0, "links": 0, "stale": 0})
        for step, event in zip(row["steps"], cell["events"], strict=True):
            if step["event_id"] != event["event_id"] or step["operation"] != event["operation"]:
                raise ValueError("step event binding mismatch")
            entries = _normalized_entries(step["entries"])
            if _sha(canonical_json_bytes(entries)) != step["entries_sha256"]:
                raise ValueError("state entry hash mismatch")
            source_state["state"] += int(step["state_match"] is True)
            source_state["links"] += int(step["current_source_event_link"] is True)
            source_state["stale"] += int(step["stale_same_slot_count"])
        last = cell["events"][-1]
        source_state["final"] += int(row.get("final_state_match") is True)
        retrieval = row.get("retrieval")
        if not isinstance(retrieval, dict):
            raise ValueError("retrieval evidence missing")
        retrieval_entries = _normalized_entries(retrieval["entries"])
        if _sha(canonical_json_bytes(retrieval_entries)) != retrieval["entries_sha256"]:
            raise ValueError("retrieval entry hash mismatch")
        source_state["retrieval"] += int(row.get("retrieval_typed_object_match") is True)
        state_matches += source_state["state"] - source_state["state"] + sum(step["state_match"] is True for step in row["steps"])
        source_links += sum(step["current_source_event_link"] is True for step in row["steps"])
        stale += sum(step["stale_same_slot_count"] for step in row["steps"])
        final_matches += int(row.get("final_state_match") is True)
        retrieval_matches += int(row.get("retrieval_typed_object_match") is True)
        signatures.setdefault(name, []).append(canonical_json_bytes(_stable_row_signature(row)))
    if seen != {(name, repetition) for name in cells for repetition in (0, 1)}:
        raise ValueError("row coverage mismatch")
    repetitions_equal = all(values[0] == values[1] for values in signatures.values())
    expected = {"bea": 3, "noaa": 7}
    if state_matches != 20 or source_links != 20 or stale != 0 or final_matches != 4 or retrieval_matches != 4 or not repetitions_equal:
        raise ValueError("v13 metric mismatch")
    return {
        "status": "VALID",
        "schema": "memupdatebench.cross-domain.qdrant-state-evidence.v1",
        "artifact_index_sha256": EXPECTED_ARTIFACT_INDEX_SHA256,
        "collection_audit_sha256": EXPECTED_COLLECTION_AUDIT_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "release_bindings": {"bea": BEA_ROOT_PIN, "noaa": NOAA_ROOT_PIN},
        "cluster_unit": "source_trajectory",
        "cluster_n": 2,
        "metrics": {
            "state_step_matches": [20, 20],
            "final_state_matches": [4, 4],
            "retrieval_matches": [4, 4],
            "stale_same_slot": [0, 0],
        },
        "base_events": 10,
        "executed_mutations": 20,
        "retrieval_requests": 4,
        "repetitions_equal": True,
        "scientific_release_allowed": False,
        "benchmark_accuracy_claimed": False,
        "answer_metrics": None,
        "per_source": per_source,
    }
