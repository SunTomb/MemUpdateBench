from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


RELEASE_ID = "main_track_v1"
SCHEMA_VERSION = "memupdatebench.main-track.audit-selection.v1"
FAMILIES = (
    "interleaved_multi_slot_update",
    "entity_attribute_grounding",
    "noop_write_discipline",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(root: Path) -> list[dict[str, Any]]:
    raw = (root / "tasks.jsonl").read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("candidate tasks are not canonical JSONL")
    return [json.loads(line) for line in raw.splitlines()]


def _core_key(row: dict[str, Any]) -> str:
    return row["metadata"]["split_key"]["semantic_core_id"]


def _axis(row: dict[str, Any], name: str) -> Any:
    event = row["events"][0].get("metadata", {})
    if name == "active_object_count":
        return len(row.get("target_objects", ()))
    if name == "noop_density":
        noop_count = sum(
            action.get("operation") == "NOOP"
            for action in row.get("actions", ())
        )
        return round(noop_count / 4, 2)
    if name in event:
        return event[name]
    return row["metadata"]["extra"].get(name)


def _select_family(rows: list[dict[str, Any]], family: str) -> list[str]:
    family_rows = [row for row in rows if row["task_family"] == family]
    by_core: dict[str, dict[str, Any]] = {}
    for row in family_rows:
        by_core.setdefault(_core_key(row), row)
    candidates = sorted(by_core.values(), key=lambda row: _core_key(row).encode())
    selected: list[dict[str, Any]] = []
    requirements: dict[str, set[Any]] = {}
    if family == FAMILIES[0]:
        requirements = {"active_object_count": {2, 4, 8, 12}, "interleaving_pattern": {"round_robin", "burst", "adversarial_adjacent"}}
    elif family == FAMILIES[1]:
        requirements = {"entity_condition": {"distinct", "alias", "same_name", "namespace_collision"}, "attribute_condition": {"exact", "paraphrase", "near_name"}}
    else:
        requirements = {"trap_type": {"transient", "hypothetical", "negated", "uncertain", "semantic_near_miss", "duplicate_current", "unsupported_inference"}, "noop_density": {0.25, 0.5, 0.75}}
    covered = {name: set() for name in requirements}
    remaining = list(candidates)
    while remaining and (len(selected) < 20 and any(covered[name] != needed for name, needed in requirements.items())):
        best = max(remaining, key=lambda row: (sum(_axis(row, name) not in covered[name] for name in requirements), _core_key(row)))
        remaining.remove(best)
        selected.append(best)
        for name in requirements:
            covered[name].add(_axis(best, name))
    if any(covered[name] != needed for name, needed in requirements.items()):
        raise ValueError(f"audit selection cannot cover {family} axes")
    selected.extend(remaining[: 20 - len(selected)])
    if len(selected) != 20:
        raise ValueError(f"audit selection requires 20 cores for {family}")
    return [_core_key(row) for row in selected]


def select_main_track_audit(candidate_root: Path | str) -> dict[str, Any]:
    root = Path(candidate_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("candidate root must be a real directory")
    required = {"generation_config.json", "catalog_manifest.json", "semantic_cores.jsonl", "tasks.jsonl", "split_balance.json", "task_manifest.json", "validation_report.json", "release_index.json"}
    if {path.name for path in root.iterdir()} != required:
        raise ValueError("candidate artifact set mismatch")
    rows = _load(root)
    release = json.loads((root / "release_index.json").read_bytes())
    release_refs = {item["path"]: item["sha256"] for item in release.get("artifacts", [])}
    observed_refs = {path.name: _sha(path) for path in root.iterdir() if path.name != "release_index.json"}
    if release.get("release_id") != RELEASE_ID or release_refs != observed_refs:
        raise ValueError("candidate artifact hashes do not match release index")
    if len(rows) != 3600 or len({_core_key(row) for row in rows}) != 900:
        raise ValueError("candidate cardinality mismatch")
    selected_cores = [core for family in FAMILIES for core in _select_family(rows, family)]
    by_core = {_core_key(row): row for row in rows}
    selected_rows = [row for row in rows if _core_key(row) in set(selected_cores)]
    selected_rows.sort(key=lambda row: (selected_cores.index(_core_key(row)), row["metadata"]["extra"]["surface_variant"]))
    if len(selected_rows) != 240:
        raise ValueError("audit task cardinality mismatch")
    core_rows = [by_core[core] for core in selected_cores]
    family_counts = dict(sorted(Counter(row["task_family"] for row in core_rows).items()))
    domain_counts = dict(sorted(Counter(row["metadata"]["extra"]["domain"] for row in core_rows).items()))
    attribute_counts = dict(sorted(Counter(row["metadata"]["extra"]["attribute"] for row in core_rows).items()))
    split_counts = dict(sorted(Counter(row["metadata"]["split"] for row in selected_rows).items()))
    surface_counts = dict(sorted(Counter(row["metadata"]["extra"]["surface_key"] for row in selected_rows).items()))
    coverage = {
        "family_b": {"active_object_counts": sorted({_axis(row, "active_object_count") for row in core_rows if row["task_family"] == FAMILIES[0]}), "interleaving_patterns": sorted({_axis(row, "interleaving_pattern") for row in core_rows if row["task_family"] == FAMILIES[0]})},
        "family_c": {"entity_conditions": sorted({_axis(row, "entity_condition") for row in core_rows if row["task_family"] == FAMILIES[1]}), "attribute_conditions": sorted({_axis(row, "attribute_condition") for row in core_rows if row["task_family"] == FAMILIES[1]})},
        "family_d": {"trap_types": sorted({_axis(row, "trap_type") for row in core_rows if row["task_family"] == FAMILIES[2]}), "noop_densities": sorted({_axis(row, "noop_density") for row in core_rows if row["task_family"] == FAMILIES[2]})},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "candidate_artifact_hashes": {path.name: _sha(path) for path in sorted(root.iterdir())},
        "selection_algorithm": "post-core-main-track-stratified-greedy-v1",
        "selected_semantic_core_count": len(selected_cores),
        "selected_task_count": len(selected_rows),
        "selected_semantic_core_ids": selected_cores,
        "selected_task_ids": [row["task_id"] for row in selected_rows],
        "selected_task_hashes": {row["task_id"]: hashlib.sha256(_canonical(row)).hexdigest() for row in selected_rows},
        "family_counts": family_counts,
        "domain_counts": domain_counts,
        "attribute_counts": attribute_counts,
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in core_rows).items())),
        "split_counts": split_counts,
        "surface_counts": surface_counts,
        "coverage": coverage,
        "review_status": "NOT_STARTED",
        "review_policy_version": "post-core-data-audit-v1",
    }


__all__ = ["select_main_track_audit"]
