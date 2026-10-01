"""Promote the policy-verified BEA Family H candidate as a bounded task release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.validation.replay_v3 import replay_task_v3

CANDIDATE_INDEX_SHA256 = "2a12b35dd2972f98d6b90f8cb9bfea46a60924e910ac94cab37c79756a6f86cd"
CAPTURE_MANIFEST_SHA256 = "9395b50167b4b47f64e19b9ebcbbdafe6c3ba70e1084ebb9a273bc3ba179eb1b"
CAPTURE_INDEX_SHA256 = "17aa9bd3a576effa7b30b79f87e6090e07edf58839594518f808049ac11e6313"
POLICY_SHA256 = "e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e"
DECISIONS_SHA256 = "43c8e8ad78aa427ff16b479b5eb63e9c79e2fa27bc7843d5f7eae260dce28b94"

CANDIDATE_NAMES = frozenset({
    "SOURCE_ATTRIBUTION.txt", "manifest.json", "normalized_records.jsonl", "semantic_cores.jsonl", "tasks.jsonl",
    "audit_manifest.json", "decisions_template.json", "reference_sanity.json", "index.json",
})
REVIEW_NAMES = frozenset({"decisions.json", "review_completion_receipt.json", "index.json"})
RELEASE_NAMES = frozenset({
    "SOURCE_ATTRIBUTION.txt", "manifest.json", "normalized_records.jsonl", "semantic_cores.jsonl", "tasks.jsonl",
    "audit_manifest.json", "decisions_template.json", "reference_sanity.json", "candidate_index.json",
    "decisions.json", "review_completion_receipt.json", "human_review_index.json",
    "release_manifest.json", "release_index.json",
})
RELEASE_SCHEMA = "memupdatebench.family-h.cross-domain-bea.release-manifest.v1"
RELEASE_INDEX_SCHEMA = "memupdatebench.family-h.cross-domain-bea.release-index.v1"
SOURCE_BOUNDARY = (
    "BEA material is treated as public-domain under the verified BEA linking policy, with source attribution appreciated; "
    "third-party copyrighted material remains subject to its own terms, and derived records are not an official BEA product."
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return _sha(_canonical(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_no_reparse_components(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not contain symlink components")
        if current.parent == current:
            return
        current = current.parent


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    _assert_no_reparse_components(path, label)
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    return path


def _root(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    _assert_no_reparse_components(path, label)
    metadata = path.lstat()
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory")
    return path


def _json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _read_exact(root: Path, names: frozenset[str], label: str) -> dict[str, bytes]:
    root = _root(root, label)
    entries = {path.name for path in root.iterdir()}
    _require(entries == names, f"{label} artifact membership mismatch")
    result: dict[str, bytes] = {}
    for name in sorted(names):
        result[name] = _regular_file(root / name, f"{label}/{name}").read_bytes()
    return result


def _validate_index(index_raw: bytes, expected_sha: str, *, schema: str, status: str, names: frozenset[str], label: str) -> dict:
    _require(_sha(index_raw) == expected_sha, f"{label} hash mismatch")
    index = _json(index_raw, label)
    _require(index.get("schema") == schema and index.get("status") == status,
             f"{label} schema/status mismatch")
    _require(index.get("scientific_release_allowed") is False, f"{label} scientific boundary mismatch")
    artifacts = index.get("artifacts")
    _require(isinstance(artifacts, list) and {row.get("path") for row in artifacts} == names - {"index.json"},
             f"{label} artifact membership mismatch")
    return index


def _read_review(root: Path) -> dict[str, bytes]:
    root = _root(root, "review")
    entries = {path.name for path in root.iterdir()}
    if entries == {"decisions.json"}:
        return {"decisions.json": _regular_file(root / "decisions.json", "review/decisions.json").read_bytes()}
    _require(entries == REVIEW_NAMES, "review artifact membership mismatch")
    return {name: _regular_file(root / name, f"review/{name}").read_bytes() for name in sorted(REVIEW_NAMES)}


def _complete_review(review: dict[str, bytes], candidate_sha: str) -> dict[str, bytes]:
    if set(review) == {"decisions.json"}:
        decisions = _json(review["decisions.json"], "review decisions")
        receipt = {"candidate_index_sha256": candidate_sha, "decisions_sha256": _sha(review["decisions.json"]),
                   "decision_count": 5, "counts": {"RELEASE_READY": 5}, "status": "HUMAN_REVIEW_COMPLETE",
                   "formal_task_release": False, "scientific_release_allowed": False, "reviewer": decisions.get("reviewer")}
        receipt_raw = _canonical(receipt)
        index = {"schema": "family-h-cross-domain-bea-human-review-index.v1", "status": "HUMAN_REVIEW_COMPLETE",
                 "formal_task_release": False, "scientific_release_allowed": False,
                 "artifacts": [{"path": "decisions.json", "bytes": len(review["decisions.json"]), "sha256": _sha(review["decisions.json"])},
                               {"path": "review_completion_receipt.json", "bytes": len(receipt_raw), "sha256": _sha(receipt_raw)}]}
        return {**review, "review_completion_receipt.json": receipt_raw, "index.json": _canonical(index)}
    return review


def _validate_candidate(candidate: dict[str, bytes]) -> tuple[dict, list[MemUpdateTaskV3], dict]:
    index = _validate_index(candidate["index.json"], CANDIDATE_INDEX_SHA256,
                            schema="family-h-cross-domain-bea-candidate-v1-index-v1",
                            status="CANDIDATE_PENDING_HUMAN_REVIEW", names=CANDIDATE_NAMES, label="candidate index")
    for row in index["artifacts"]:
        raw = candidate[row["path"]]
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw), f"candidate artifact hash mismatch: {row['path']}")
    manifest = _json(candidate["manifest.json"], "candidate manifest")
    _require(manifest.get("status") == "CANDIDATE_PENDING_HUMAN_REVIEW" and manifest.get("formal_task_release") is False
             and manifest.get("scientific_release_allowed") is False and manifest.get("source_audit_status") == "NOT_STARTED"
             and manifest.get("generated_surface_audit_status") == "NOT_STARTED", "candidate historical boundary mismatch")
    _require(manifest.get("capture_manifest_sha256") == CAPTURE_MANIFEST_SHA256 and manifest.get("capture_index_sha256") == CAPTURE_INDEX_SHA256
             and manifest.get("source_kind") == "sequential_release_vintage" and manifest.get("revision_only") is True
             and manifest.get("independent_samples") is False, "candidate capture lineage mismatch")
    actual_counts = (manifest.get("semantic_core_count"), manifest.get("task_count"), manifest.get("snapshot_count"),
                     manifest.get("event_count"), manifest.get("update_events"), manifest.get("equal_value_update_events"))
    _require(actual_counts == (1, 1, 3, 3, 2, 1), "candidate counts mismatch")
    tasks = [MemUpdateTaskV3.model_validate(_json(line, "candidate task")) for line in candidate["tasks.jsonl"].splitlines() if line.strip()]
    _require(len(tasks) == 1 and replay_task_v3(tasks[0]).valid, "candidate replay invalid")
    task = tasks[0]
    _require(len(task.events) == 3 and len(task.actions) == 3 and len(task.queries) == 1, "candidate task cardinality mismatch")
    _require(task.actions[0].operation.value == "ADD" and all(action.operation.value == "UPDATE" for action in task.actions[1:])
             and sum(task.actions[i].value == task.actions[i - 1].value for i in range(1, 3)) == 1, "candidate action sequence mismatch")
    _require([action.value for action in task.actions] == [2.3, 2.3, 2.4]
             and task.queries[0].answer_schema.value == "number", "candidate numeric value/schema mismatch")
    _require(task.metadata.extra.get("scientific_release_allowed") is False and task.metadata.extra.get("formal_task_release") is False
             and task.source.provenance.get("source_audit_status") == "NOT_STARTED"
             and task.source.provenance.get("generated_surface_audit_status") == "NOT_STARTED", "candidate task boundary mismatch")
    audit = _json(candidate["audit_manifest.json"], "candidate audit manifest")
    _require(audit.get("status") == "NOT_STARTED" and audit.get("item_count") == 5 and len(audit.get("items", [])) == 5, "candidate audit boundary mismatch")
    _require(all(_digest(item["material"]) == item["binding_sha256"] for item in audit["items"]), "candidate audit binding mismatch")
    kinds = {kind: [item for item in audit["items"] if item.get("kind") == kind] for kind in ("source_admission", "source_snapshot", "generated_surface")}
    _require([len(kinds[kind]) for kind in kinds] == [1, 3, 1], "candidate audit kind counts mismatch")
    surface = kinds["generated_surface"][0]["material"]
    _require(surface.get("task_id") == task.task_id and surface.get("canonical_task_sha256") == _digest(task.model_dump(mode="json"))
             and surface.get("semantic_task_sha256") == semantic_task_hash_v3(task) and surface.get("event_count") == 3, "candidate surface binding mismatch")
    core_rows = [_json(line, "semantic core") for line in candidate["semantic_cores.jsonl"].splitlines() if line.strip()]
    _require(len(core_rows) == 1 and core_rows[0].get("core_id") == task.metadata.split_key.semantic_core_id, "candidate semantic core binding mismatch")
    return manifest, tasks, audit


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _policy_file(policy_root: Path) -> Path:
    path = Path(policy_root)
    return path if path.name == "linking.html" else path / "linking.html"


def _validate_capture(capture_root: Path, candidate: dict[str, bytes], policy_root: Path) -> dict:
    root = _root(capture_root, "capture")
    manifest_path = _regular_file(root / "capture_manifest.json", "capture manifest")
    index_path = _regular_file(root / "capture_index.json", "capture index")
    manifest_raw, index_raw = manifest_path.read_bytes(), index_path.read_bytes()
    _require(_sha(manifest_raw) == CAPTURE_MANIFEST_SHA256, "capture manifest hash mismatch")
    _require(_sha(index_raw) == CAPTURE_INDEX_SHA256, "capture index hash mismatch")
    index = _json(index_raw, "capture index")
    _require(index.get("schema") == "memupdatebench.family-h.bea-source-capture-index.v1"
             and index.get("status") == "CAPTURED_PENDING_HUMAN_REVIEW"
             and index.get("scientific_release_allowed") is False, "capture index boundary mismatch")
    paths = {row["path"] for row in index.get("artifacts", [])} | {"capture_manifest.json", "capture_index.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == paths, "capture artifact membership mismatch")
    for row in index["artifacts"]:
        raw = _regular_file(root / row["path"], f"capture/{row['path']}").read_bytes()
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw), f"capture artifact hash mismatch: {row['path']}")
    manifest = _json(manifest_raw, "capture manifest")
    _require(manifest.get("source_group_id") == "family-h-cross-domain-bea-gdp-2024q4"
             and manifest.get("source_kind") == "sequential_release_vintage"
             and manifest.get("source_type") == "other" and manifest.get("domain") == "macroeconomics"
             and manifest.get("revision_only") is True and manifest.get("independent_samples") is False
             and manifest.get("policy_status") == "POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY"
             and manifest.get("policy", {}).get("sha256") == POLICY_SHA256
             and manifest.get("policy", {}).get("url") == "https://www.bea.gov/about/policies-and-information/linking"
             and manifest.get("formal_task_release") is False and manifest.get("scientific_release_allowed") is False,
             "capture metadata mismatch")
    rows = manifest.get("records")
    _require(isinstance(rows, list) and len(rows) == 3, "capture record count mismatch")
    normalized = [_json(line, "candidate normalized record") for line in candidate["normalized_records.jsonl"].splitlines() if line.strip()]
    _require([row.get("normalized") for row in rows] == normalized, "candidate/capture normalized records mismatch")
    policy_file = _regular_file(_policy_file(policy_root), "BEA policy evidence")
    policy_raw = policy_file.read_bytes()
    _require(_sha(policy_raw) == POLICY_SHA256, "BEA policy evidence hash mismatch")
    policy_text = policy_raw.decode("utf-8")
    _require("in the public domain" in policy_text and "may be used or reproduced without specific permission" in policy_text
             and "must not contain information that suggests such an endorsement" in policy_text
             and "cannot authorize the use of copyrighted materials" in policy_text
             and "Source: U.S. Bureau of Economic Analysis" in policy_text, "BEA policy claims mismatch")
    return manifest


def _validate_review(review: dict[str, bytes], audit: dict, decisions_path: Path | None) -> tuple[dict, dict]:
    index = _validate_index(review["index.json"], _sha(review["index.json"]),
                            schema="family-h-cross-domain-bea-human-review-index.v1",
                            status="HUMAN_REVIEW_COMPLETE", names=REVIEW_NAMES, label="review index")
    _require(index.get("formal_task_release") is False, "review formal release boundary mismatch")
    for row in index["artifacts"]:
        raw = review[row["path"]]
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw), f"review artifact hash mismatch: {row['path']}")
    submitted = _regular_file(decisions_path, "submitted decisions").read_bytes() if decisions_path is not None else review["decisions.json"]
    _require(_sha(submitted) == DECISIONS_SHA256 and submitted == review["decisions.json"], "submitted review decisions binding mismatch")
    decisions = _json(review["decisions.json"], "review decisions")
    _require(decisions.get("status") == "COMPLETED" and decisions.get("candidate_index_sha256") == CANDIDATE_INDEX_SHA256, "review decision status/binding mismatch")
    expected = {item["audit_id"]: item["binding_sha256"] for item in audit["items"]}
    rows = decisions.get("decisions")
    _require(isinstance(rows, list) and len(rows) == 5 and {row.get("audit_id") for row in rows} == set(expected), "review decision cardinality mismatch")
    _require(all(row.get("binding_sha256") == expected[row["audit_id"]] and row.get("decision") == "RELEASE_READY"
                 and isinstance(row.get("rationale"), str) and row["rationale"].strip() for row in rows), "review decisions are not all release-ready")
    receipt = _json(review["review_completion_receipt.json"], "review completion receipt")
    _require(receipt.get("candidate_index_sha256") == CANDIDATE_INDEX_SHA256
             and receipt.get("decisions_sha256") == DECISIONS_SHA256
             and receipt.get("decision_count") == 5
             and receipt.get("counts") == {"RELEASE_READY": 5}
             and receipt.get("status") == "HUMAN_REVIEW_COMPLETE"
             and receipt.get("formal_task_release") is False
             and receipt.get("scientific_release_allowed") is False
             and receipt.get("reviewer") == decisions.get("reviewer"), "review receipt binding mismatch")
    return decisions, receipt


def _build_payload(candidate: dict[str, bytes], review: dict[str, bytes], manifest: dict,
                   decisions: dict, receipt: dict, audit: dict, capture: dict) -> dict[str, bytes]:
    tasks = [MemUpdateTaskV3.model_validate(_json(line, "candidate task")) for line in candidate["tasks.jsonl"].splitlines() if line.strip()]
    task = tasks[0]
    input_sha256 = {"candidate_index": CANDIDATE_INDEX_SHA256, "review_index": _sha(review["index.json"]),
                    "decisions": DECISIONS_SHA256, "review_completion_receipt": _sha(review["review_completion_receipt.json"]),
                    "capture_manifest": CAPTURE_MANIFEST_SHA256, "capture_index": CAPTURE_INDEX_SHA256,
                    "policy_evidence": POLICY_SHA256}
    policy = manifest.get("policy_evidence", {})
    release_manifest = {
        "schema": RELEASE_SCHEMA, "status": "FINAL_APPROVED", "formal_task_release": True,
        "scientific_release_allowed": False, "benchmark_accuracy_claimed": False, "model_backend_status": "NOT_RUN",
        "runtime_state_metrics": None, "runtime_retrieval_metrics": None, "answer_metrics": None,
        "source_count": 1, "source_count_semantics": "one source trajectory, not statistically independent samples",
        "source_domain": "macroeconomics", "source_type": "other", "source_kind": "sequential_release_vintage",
        "revision_only": True, "independent_samples": False,
        "counts": {"records": 3, "cores": 1, "tasks": 1, "events": 3, "decisions": 5},
        "update_counts": {"ADD": 1, "UPDATE": 2, "equal_value_updates": 1}, "answer_schema": "number",
        "input_sha256": input_sha256, "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "capture_index_sha256": CAPTURE_INDEX_SHA256, "capture_status": capture["status"],
        "policy_status": "VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY",
        "policy_evidence": {"source_sha256": POLICY_SHA256, "source_url": policy.get("source_url", "https://www.bea.gov/about/policies-and-information/linking"),
                            "exact_boundary": SOURCE_BOUNDARY, "official_citation_requirement": "appreciated, not stated as mandatory",
                            "raw_source_redistribution": False, "derivation_notice": "Derived records are not an official BEA product"},
        "historical_candidate_status": manifest["status"], "historical_review_status": receipt["status"],
        "reviewer": decisions["reviewer"], "reviewer_identity": "self-reported; not cryptographically authenticated",
        "source_boundary": SOURCE_BOUNDARY, "raw_capture_policy": "Raw BEA XLSX/PDF/source capture is retained only as hash-bound input metadata; it is not copied into this public release.",
        "claim_boundary": "Bounded policy-verified BEA task release only; no benchmark accuracy, model, backend, or statistically independent-source claim.",
        "task_id": task.task_id, "semantic_task_sha256": semantic_task_hash_v3(task), "reviewed_audit_item_count": len(audit["items"]),
    }
    archive = {**{name: candidate[name] for name in CANDIDATE_NAMES if name != "index.json"},
               "candidate_index.json": candidate["index.json"], "decisions.json": review["decisions.json"],
               "review_completion_receipt.json": review["review_completion_receipt.json"], "human_review_index.json": review["index.json"],
               "release_manifest.json": _canonical(release_manifest)}
    release_index = {"schema": RELEASE_INDEX_SCHEMA, "status": "FINAL_APPROVED", "formal_task_release": True,
                     "scientific_release_allowed": False, "input_sha256": input_sha256,
                     "artifacts": [{"path": name, "bytes": len(raw), "sha256": _sha(raw)} for name, raw in sorted(archive.items())]}
    archive["release_index.json"] = _canonical(release_index)
    return archive


def _validate_release_payload(payload: dict[str, bytes], expected_index_sha256: str) -> tuple[dict, list[MemUpdateTaskV3], dict]:
    _require(set(payload) == RELEASE_NAMES, "release artifact membership mismatch")
    _require(_sha(payload["release_index.json"]) == expected_index_sha256, "release index hash mismatch")
    index = _json(payload["release_index.json"], "release index")
    _require(index["schema"] == RELEASE_INDEX_SCHEMA and index["status"] == "FINAL_APPROVED"
             and index["formal_task_release"] is True and index["scientific_release_allowed"] is False,
             "release index boundary mismatch")
    for row in index["artifacts"]:
        _require(row["path"] in payload and row["bytes"] == len(payload[row["path"]])
                 and row["sha256"] == _sha(payload[row["path"]]),
                 f"release artifact hash mismatch: {row.get('path')}")
    _require({row["path"] for row in index["artifacts"]} == RELEASE_NAMES - {"release_index.json"},
             "release index member mismatch")
    manifest = _json(payload["release_manifest.json"], "release manifest")
    _require(manifest["schema"] == RELEASE_SCHEMA and manifest["status"] == "FINAL_APPROVED"
             and manifest["formal_task_release"] is True and manifest["scientific_release_allowed"] is False
             and manifest["benchmark_accuracy_claimed"] is False and manifest["model_backend_status"] == "NOT_RUN"
             and all(manifest[key] is None for key in ("runtime_state_metrics", "runtime_retrieval_metrics", "answer_metrics"))
             and manifest["policy_status"] == "VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY"
             and manifest["source_kind"] == "sequential_release_vintage" and manifest["revision_only"] is True
             and manifest["independent_samples"] is False
             and manifest["counts"] == {"records": 3, "cores": 1, "tasks": 1, "events": 3, "decisions": 5}
             and manifest["update_counts"] == {"ADD": 1, "UPDATE": 2, "equal_value_updates": 1}
             and manifest["answer_schema"] == "number"
             and manifest["policy_evidence"]["source_sha256"] == POLICY_SHA256
             and manifest["policy_evidence"]["source_url"] == "https://www.bea.gov/about/policies-and-information/linking"
             and manifest["policy_evidence"]["exact_boundary"] == SOURCE_BOUNDARY
             and manifest["source_boundary"] == SOURCE_BOUNDARY,
             "release manifest scientific boundary mismatch")
    candidate = {name: payload[name] for name in CANDIDATE_NAMES if name != "index.json"}
    candidate["index.json"] = payload["candidate_index.json"]
    review = {"decisions.json": payload["decisions.json"],
              "review_completion_receipt.json": payload["review_completion_receipt.json"],
              "index.json": payload["human_review_index.json"]}
    candidate_manifest, tasks, audit = _validate_candidate(candidate)
    decisions, receipt = _validate_review(review, audit, None)
    input_sha256 = manifest.get("input_sha256")
    _require(isinstance(input_sha256, dict)
             and input_sha256.get("decisions") == _sha(review["decisions.json"])
             and input_sha256.get("review_completion_receipt") == _sha(review["review_completion_receipt.json"])
             and input_sha256.get("capture_manifest") == CAPTURE_MANIFEST_SHA256
             and input_sha256.get("capture_index") == CAPTURE_INDEX_SHA256,
             "release input hash binding mismatch")
    _require(manifest["historical_candidate_status"] == candidate_manifest["status"]
             and manifest["historical_review_status"] == receipt["status"]
             and manifest["reviewer"] == decisions["reviewer"]
             and input_sha256["candidate_index"] == CANDIDATE_INDEX_SHA256
             and input_sha256["review_index"] == _sha(review["index.json"])
             and input_sha256.get("policy_evidence") == POLICY_SHA256,
             "release lineage mismatch")
    return manifest, tasks, index


def _guard_output(output_root: Path, inputs: tuple[Path, ...]) -> None:
    output_root = output_root.absolute()
    _assert_no_reparse_components(output_root, "output")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {output_root}")
    parts = output_root.parts
    frozen_sequences = (
        ("data", "vnext", "core"), ("data", "vnext", "pilot"),
        ("data", "vnext", "family_h_independent", "v1_published"),
        ("data", "vnext", "family_h_cross_domain_noaa", "v1"),
    )
    _require(not any(any(parts[i:i + len(sequence)] == sequence for i in range(len(parts) - len(sequence) + 1)) for sequence in frozen_sequences), "output overlaps frozen release root")
    _require("MemUpdateBench_releases" not in parts, "output overlaps frozen release library")
    _require(not any(any((parent / name).is_file() for name in ("index.json", "release_index.json", "artifact_index.json")) for parent in output_root.parents), "output is nested under an indexed artifact root")
    for source in inputs:
        source = source.absolute()
        _require(source not in output_root.parents and output_root not in source.parents, "output overlaps input")


def read_release(root: Path | str, expected_index_sha256: str) -> dict[str, Any]:
    """Read and validate a published release without writing any files."""
    release_root = _root(Path(root), "release")
    payload = _read_exact(release_root, RELEASE_NAMES, "release")
    manifest, tasks, index = _validate_release_payload(payload, expected_index_sha256)
    return {"manifest": manifest, "tasks": tasks, "index": index,
            "root": release_root.resolve(), "index_sha256": expected_index_sha256}


def publish_release(candidate_root: Path | str, review_root: Path | str, decisions_path: Path | str,
                    capture_root: Path | str, policy_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    """Publish a no-replace release from exact candidate, review, capture, and policy bytes."""
    candidate_root, review_root, capture_root, policy_root = map(Path, (candidate_root, review_root, capture_root, policy_root))
    decisions_path, output_root = Path(decisions_path), Path(output_root)
    _guard_output(output_root, (candidate_root, review_root, capture_root, policy_root, decisions_path))
    candidate = _read_exact(candidate_root, CANDIDATE_NAMES, "candidate")
    review = _read_review(review_root)
    _regular_file(decisions_path, "submitted decisions")
    capture = _validate_capture(capture_root, candidate, policy_root)
    manifest, tasks, audit = _validate_candidate(candidate)
    review = _complete_review(review, CANDIDATE_INDEX_SHA256)
    decisions, receipt = _validate_review(review, audit, decisions_path)
    payload = _build_payload(candidate, review, manifest, decisions, receipt, audit, capture)
    release_index_sha = _sha(payload["release_index.json"])
    _validate_release_payload(payload, release_index_sha)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    source_paths = tuple(candidate_root / name for name in CANDIDATE_NAMES)
    source_paths += tuple(review_root / name for name in REVIEW_NAMES)
    source_paths += (decisions_path, capture_root / "capture_manifest.json", capture_root / "capture_index.json", _policy_file(policy_root))
    candidate_snapshot, review_snapshot = dict(candidate), dict(review)
    decisions_snapshot = decisions_path.read_bytes()
    capture_snapshot = _snapshot_tree(capture_root)
    policy_snapshot = _policy_file(policy_root).read_bytes()

    def pre_publish() -> None:
        _require(_read_exact(candidate_root, CANDIDATE_NAMES, "candidate") == candidate_snapshot,
                 "candidate changed before publication")
        _require(_complete_review(_read_review(review_root), CANDIDATE_INDEX_SHA256) == review_snapshot,
                 "review changed before publication")
        _require(decisions_path.read_bytes() == decisions_snapshot,
                 "submitted decisions changed before publication")
        _require(_snapshot_tree(capture_root) == capture_snapshot
                 and _policy_file(policy_root).read_bytes() == policy_snapshot,
                 "capture or policy evidence changed before publication")

    publish_files_atomically(
        {output_root / name: raw for name, raw in payload.items()},
        overwrite=False, source_paths=source_paths, pre_publish=pre_publish,
    )
    return read_release(output_root, release_index_sha)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_release(args.candidate_root, args.review_root, args.decisions,
                                 args.capture_root, args.policy_root, args.output_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({**result, "root": str(result["root"]),
                      "tasks": [task.model_dump(mode="json") for task in result["tasks"]]},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
