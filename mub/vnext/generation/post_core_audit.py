from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.generation.post_core_artifacts import POST_CORE_ARTIFACT_NAMES, validate_post_core_artifact_tree
from mub.vnext.io import sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.jsonl import read_models


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


AUDIT_PACKET_SCHEMA_VERSION = "memupdatebench.main-track.audit-packet.v1"
AUDIT_POLICY_VERSION = "post-core-data-audit-v1"
AUDIT_PACKET_ARTIFACT_NAMES = (
    "audit_packet.jsonl",
    "audit_manifest.json",
    "review_instructions.md",
)
_DEFAULT_SELECTION_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "vnext"
    / "main_track_v1_independence_audit_selection"
    / "selection.json"
)


@dataclass(frozen=True, slots=True)
class MainTrackAuditPacket:
    candidate_root: Path
    selection_path: Path
    selection: dict[str, Any]
    candidate_artifact_hashes: dict[str, str]
    selection_artifact_hash: str
    rows: tuple[dict[str, Any], ...]
    packet_bytes: bytes
    manifest: dict[str, Any]
    manifest_bytes: bytes
    review_instructions: str
    instructions_bytes: bytes

    @property
    def packet_row_hash(self) -> str:
        return hashlib.sha256(self.packet_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class PublishedMainTrackAuditPacket:
    output_dir: Path
    artifact_paths: tuple[Path, ...]
    artifact_hashes: dict[str, str]

    def __iter__(self):
        return iter(self.artifact_paths)


_REVIEW_INSTRUCTIONS = """# Main-track v1 human audit

This packet prepares the selected synthetic main-track candidate for human review.

Review policy: post-core-data-audit-v1
Initial review status: NOT_STARTED

For every row, inspect the visible events, event roles and metadata, declared actions,
target objects, query selector, gold evidence, and version history. Record exactly one
value in `audit_decision`: `pass`, `needs_revision`, or `block`. Use `issue_category`
when a decision is not `pass`, add the reviewer identity and a concise review note,
and set `resolved_status` to `unresolved` or `resolved` as appropriate. Leave the
five audit fields empty until a human reviewer makes a decision.

Do not edit candidate task content in this packet. Do not infer model behavior, runtime
behavior, or benchmark approval from these records. This packet contains synthetic,
redistributable task data only and contains no model outputs. Completion of this packet
is not human-audit approval; a separate authenticated review gate is required.
"""

_DECISION_VOCABULARY = {
    "audit_decision": {"empty": "", "allowed": ["pass", "needs_revision", "block"]},
    "issue_category": {
        "empty": "",
        "allowed": [
            "none",
            "event_text",
            "event_role",
            "action",
            "target_object",
            "query",
            "gold_evidence",
            "version_history",
            "surface",
            "other",
        ],
    },
    "reviewer_id": {"empty": "", "allowed": "nonblank human reviewer identifier"},
    "review_note": {"empty": "", "allowed": "free-form human note"},
    "resolved_status": {"empty": "", "allowed": ["unresolved", "resolved"]},
}


def _regular_file(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must contain a JSON object")
    if _canonical_json(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value, raw


def _task_surface_identity(task: MemUpdateTaskV3) -> dict[str, Any]:
    extra = task.metadata.extra
    return {
        "surface_id": extra.get("surface_id"),
        "surface_key": extra.get("surface_key"),
        "surface_variant": extra.get("surface_variant"),
        "locale": extra.get("locale"),
        "language": extra.get("language"),
    }


def _task_audit_row(task: MemUpdateTaskV3) -> dict[str, Any]:
    if task.source.source_type.value != "synthetic":
        raise ValueError("main-track audit packet only accepts synthetic source tasks")
    if task.source.license_or_privacy != "synthetic_redistributable":
        raise ValueError("main-track audit packet requires redistributable synthetic tasks")
    extra = task.metadata.extra
    return {
        "task_id": task.task_id,
        "core_id": task.metadata.split_key.semantic_core_id,
        "family": task.task_family,
        "domain": extra.get("domain"),
        "attribute": extra.get("attribute"),
        "difficulty": task.difficulty.value,
        "split": task.metadata.split.value,
        "surface_identity": _task_surface_identity(task),
        "schema_version": task.schema_version,
        "source": task.source.model_dump(mode="json"),
        "metadata": task.metadata.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in task.events],
        "actions": [
            {
                "action_id": action.action_id,
                "event_id": action.event_id,
                "operation": action.operation.value,
                "scope": None if action.scope is None else action.scope.value,
                "target_object_keys": [
                    key.model_dump(mode="json") for key in action.target_object_keys
                ],
                "value": action.model_dump(mode="json")["value"],
                "effect": action.model_dump(mode="json")["expected_effect"],
                "expected_effect": action.model_dump(mode="json")["expected_effect"],
                "effective_at": action.effective_at,
            }
            for action in task.actions
        ],
        "target_objects": [key.model_dump(mode="json") for key in task.target_objects],
        "queries": [query.model_dump(mode="json") for query in task.queries],
        "gold_evidence": [
            evidence.model_dump(mode="json") for evidence in task.gold_evidence
        ],
        "version_history": [
            ledger.model_dump(mode="json") for ledger in task.version_history
        ],
        "audit_decision": "",
        "issue_category": "",
        "reviewer_id": "",
        "review_note": "",
        "resolved_status": "",
    }


def _candidate_hashes(root: Path) -> dict[str, str]:
    return {name: _sha(root / name) for name in POST_CORE_ARTIFACT_NAMES}


def _assert_sources_unchanged(packet: MainTrackAuditPacket) -> None:
    current = _candidate_hashes(packet.candidate_root)
    if current != packet.candidate_artifact_hashes:
        raise ValueError("candidate artifact hashes changed during audit packet publication")
    selection_hash = _sha(packet.selection_path)
    if selection_hash != packet.selection_artifact_hash:
        raise ValueError("selection artifact hash changed during audit packet publication")


def _validate_selection(
    candidate_root: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], bytes, str]:
    selection, raw = _read_canonical_json(selection_path, "selection artifact")
    expected = select_main_track_audit(candidate_root)
    if selection != expected:
        raise ValueError("selection artifact disagrees with the authenticated candidate")
    if selection.get("review_status") != "NOT_STARTED":
        raise ValueError("selection review_status must be NOT_STARTED")
    if selection.get("review_policy_version") != AUDIT_POLICY_VERSION:
        raise ValueError("selection policy version is not canonical")
    return selection, raw, hashlib.sha256(raw).hexdigest()


def _assert_output_safe(output_dir: Path, packet: MainTrackAuditPacket) -> None:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if output_dir.is_symlink():
        raise ValueError("audit packet output root must not be a symlink")
    resolved = output_dir.resolve(strict=False)
    if resolved == packet.candidate_root or packet.candidate_root in resolved.parents:
        raise ValueError("audit packet output must not overlap candidate root")
    frozen_roots = tuple(
        (Path(__file__).resolve().parents[3] / "data" / "vnext" / name).resolve()
        for name in ("core", "pilot")
    )
    if any(resolved == root or root in resolved.parents for root in frozen_roots):
        raise ValueError("audit packet output must not overlap frozen Core or Pilot roots")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("audit packet output root must be empty for no-replace publication")


def build_main_track_audit_packet(
    candidate_root: Path | str,
    selection_path: Path | str | None = None,
) -> MainTrackAuditPacket:
    if selection_path is None:
        selection_path = _DEFAULT_SELECTION_PATH
    if not isinstance(candidate_root, (Path, str)):
        raise TypeError("candidate_root must be a Path or string")
    candidate_input = Path(candidate_root)
    if candidate_input.is_symlink():
        raise ValueError("candidate root must not be a symlink")
    candidate = candidate_input.resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("candidate root must be a directory")
    validation_report = validate_post_core_artifact_tree(candidate)
    if validation_report.get("review_status") != "NOT_STARTED":
        raise ValueError("candidate review_status must be NOT_STARTED")
    selection_input = Path(selection_path)
    selection, selection_raw, selection_hash = _validate_selection(candidate, selection_input)
    candidate_hashes = _candidate_hashes(candidate)
    if selection.get("candidate_artifact_hashes") != candidate_hashes:
        raise ValueError("selection artifact candidate hashes do not match candidate")
    tasks = tuple(read_models(candidate / "tasks.jsonl", MemUpdateTaskV3, id_field="task_id"))
    task_by_id = {task.task_id: task for task in tasks}
    selected_ids = selection.get("selected_task_ids")
    selected_hashes = selection.get("selected_task_hashes")
    if type(selected_ids) is not list or len(selected_ids) != 240 or len(set(selected_ids)) != 240:
        raise ValueError("selection must contain exactly 240 unique task IDs")
    if type(selected_hashes) is not dict or set(selected_hashes) != set(selected_ids):
        raise ValueError("selection task hash map does not match selected task IDs")
    missing = [task_id for task_id in selected_ids if task_id not in task_by_id]
    if missing:
        raise ValueError("selection references unknown candidate task")
    rows: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = task_by_id[task_id]
        if sha256_model(task) != selected_hashes[task_id]:
            raise ValueError("selection task hash does not match candidate task")
        rows.append(_task_audit_row(task))
    if tuple(row["task_id"] for row in rows) != tuple(selected_ids):
        raise AssertionError("audit packet row order is not selection order")
    packet_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    instructions_bytes = _REVIEW_INSTRUCTIONS.encode("utf-8")
    manifest = {
        "schema_version": AUDIT_PACKET_SCHEMA_VERSION,
        "release_id": selection["release_id"],
        "evidence_class": "human_audit_preparation",
        "policy_version": AUDIT_POLICY_VERSION,
        "review_status": "NOT_STARTED",
        "candidate_artifact_hashes": candidate_hashes,
        "selection_artifact_hash": selection_hash,
        "selection_artifact_path": str(selection_input.resolve(strict=True)),
        "packet_row_hash": hashlib.sha256(packet_bytes).hexdigest(),
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "row_count": len(rows),
        "counts": {
            "selected_semantic_cores": selection["selected_semantic_core_count"],
            "selected_tasks": selection["selected_task_count"],
            "families": selection["family_counts"],
            "domains": selection["domain_counts"],
            "attributes": selection["attribute_counts"],
            "difficulty": selection["difficulty_counts"],
            "splits": selection["split_counts"],
            "surfaces": selection["surface_counts"],
        },
        "coverage_axes": selection["coverage"],
        "selected_semantic_core_ids": selection["selected_semantic_core_ids"],
        "selected_task_ids": selected_ids,
        "decision_vocabulary": _DECISION_VOCABULARY,
        "empty_review_fields": [
            "audit_decision",
            "issue_category",
            "reviewer_id",
            "review_note",
            "resolved_status",
        ],
        "artifact_hashes": {
            "audit_packet.jsonl": hashlib.sha256(packet_bytes).hexdigest(),
            "audit_manifest.json": None,
            "review_instructions.md": hashlib.sha256(instructions_bytes).hexdigest(),
        },
    }
    manifest_bytes = _canonical_json(manifest)
    packet_payload = (rows, manifest, packet_bytes, manifest_bytes, instructions_bytes)
    findings = scan_for_secrets({"rows": rows, "manifest": manifest})
    if findings:
        raise ValueError(f"audit packet failed secret scan: {len(findings)} finding(s)")
    findings = scan_for_secrets(_REVIEW_INSTRUCTIONS)
    if findings:
        raise ValueError(f"review instructions failed secret scan: {len(findings)} finding(s)")
    return MainTrackAuditPacket(
        candidate_root=candidate,
        selection_path=selection_input.resolve(strict=True),
        selection=selection,
        candidate_artifact_hashes=candidate_hashes,
        selection_artifact_hash=selection_hash,
        rows=tuple(packet_payload[0]),
        packet_bytes=packet_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        review_instructions=_REVIEW_INSTRUCTIONS,
        instructions_bytes=instructions_bytes,
    )


def _validate_stage_bytes(staged: Path, expected: bytes, label: str) -> None:
    try:
        actual = staged.read_bytes()
    except OSError as exc:
        raise ValueError(f"staged {label} could not be read") from exc
    if actual != expected:
        raise ValueError(f"staged {label} bytes changed")
    if label.endswith("jsonl"):
        try:
            rows = [json.loads(line) for line in actual.splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("staged audit packet is not canonical JSONL") from exc
        if any(scan_for_secrets(row) for row in rows):
            raise ValueError("staged audit packet failed secret scan")
    elif label.endswith("json"):
        parsed = json.loads(actual)
        if _canonical_json(parsed) != actual or scan_for_secrets(parsed):
            raise ValueError("staged audit manifest is not canonical or failed secret scan")
    else:
        try:
            text = actual.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("staged review instructions are not UTF-8") from exc
        if scan_for_secrets(text):
            raise ValueError("staged review instructions failed secret scan")


def publish_main_track_audit_packet(
    packet: MainTrackAuditPacket,
    output_dir: Path | str,
    *,
    before_publish: Callable[[], None] | None = None,
) -> PublishedMainTrackAuditPacket:
    if type(packet) is not MainTrackAuditPacket:
        raise TypeError("packet must be a MainTrackAuditPacket")
    if not isinstance(output_dir, (Path, str)):
        raise TypeError("output_dir must be a Path or string")
    output = Path(output_dir)
    _assert_output_safe(output, packet)
    _assert_sources_unchanged(packet)
    destinations = tuple(output / name for name in AUDIT_PACKET_ARTIFACT_NAMES)
    payloads = {
        destinations[0]: packet.packet_bytes,
        destinations[1]: packet.manifest_bytes,
        destinations[2]: packet.instructions_bytes,
    }

    def guard() -> None:
        if before_publish is not None:
            before_publish()
        _assert_sources_unchanged(packet)

    validators = {
        destinations[0]: lambda staged: _validate_stage_bytes(
            staged, packet.packet_bytes, "audit_packet.jsonl"
        ),
        destinations[1]: lambda staged: _validate_stage_bytes(
            staged, packet.manifest_bytes, "audit_manifest.json"
        ),
        destinations[2]: lambda staged: _validate_stage_bytes(
            staged, packet.instructions_bytes, "review_instructions.md"
        ),
    }
    publish_files_atomically(
        payloads,
        overwrite=False,
        source_paths=tuple(packet.candidate_root / name for name in POST_CORE_ARTIFACT_NAMES)
        + (packet.selection_path,),
        validators=validators,
        pre_publish=guard,
    )
    if {path.name for path in output.iterdir()} != set(AUDIT_PACKET_ARTIFACT_NAMES):
        raise RuntimeError("published audit packet root does not contain exactly three artifacts")
    for destination, expected in payloads.items():
        if destination.read_bytes() != expected:
            raise RuntimeError("published audit packet bytes differ from canonical payload")
    return PublishedMainTrackAuditPacket(
        output_dir=output,
        artifact_paths=destinations,
        artifact_hashes={
            name: hashlib.sha256(payloads[path]).hexdigest()
            for path, name in zip(destinations, AUDIT_PACKET_ARTIFACT_NAMES, strict=True)
        },
    )


__all__ = [
    "AUDIT_PACKET_ARTIFACT_NAMES",
    "AUDIT_PACKET_SCHEMA_VERSION",
    "AUDIT_POLICY_VERSION",
    "MainTrackAuditPacket",
    "PublishedMainTrackAuditPacket",
    "build_main_track_audit_packet",
    "publish_main_track_audit_packet",
    "select_main_track_audit",
]
