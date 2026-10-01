"""Promote the reviewed NOAA Family H candidate as a bounded task release."""
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

CANDIDATE_INDEX_SHA256 = "17d7d7785dbd7b6bd8afa705c3e4ecea6931cec8d056ec04c04967c925e727b1"
CAPTURE_MANIFEST_SHA256 = "773c28cdeae721fce84b66ba7bc26a31390cbbfb1020c93bd05614befa2d0aee"
CAPTURE_INDEX_SHA256 = "d37a2a7a5ad5c0842fd1dae9846b3e2bb09264d2b238b31506b75766ff0be832"
REVIEW_INDEX_SHA256 = "77167dae09c8ef688fe9986ad05ee5d2d4edafc8c1a80b500ba441b453cefee1"
DECISIONS_SHA256 = "a0075eaeb2d81de1d3b039b525cfdd31243eb93f92999ab9ee13004f6f02d247"

CANDIDATE_NAMES = frozenset({
    "manifest.json", "normalized_records.jsonl", "semantic_cores.jsonl", "tasks.jsonl",
    "audit_manifest.json", "decisions_template.json", "reference_sanity.json", "index.json",
})
REVIEW_NAMES = frozenset({"decisions.json", "review_completion_receipt.json", "index.json"})
RELEASE_NAMES = frozenset({
    "manifest.json", "normalized_records.jsonl", "semantic_cores.jsonl", "tasks.jsonl",
    "audit_manifest.json", "decisions_template.json", "reference_sanity.json", "candidate_index.json",
    "decisions.json", "review_completion_receipt.json", "human_review_index.json",
    "release_manifest.json", "release_index.json",
})
RELEASE_SCHEMA = "memupdatebench.family-h.cross-domain-noaa.release-manifest.v1"
RELEASE_INDEX_SCHEMA = "memupdatebench.family-h.cross-domain-noaa.release-index.v1"
SOURCE_BOUNDARY = (
    "NWS public-domain material is used under the NWS policy; third-party material may have "
    "separate terms, and this release does not imply NWS or National Hurricane Center endorsement."
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


def _validate_candidate(candidate: dict[str, bytes]) -> tuple[dict, list[MemUpdateTaskV3], dict]:
    index = _validate_index(candidate["index.json"], CANDIDATE_INDEX_SHA256,
                            schema="family-h-cross-domain-noaa-candidate-v2-index-v3",
                            status="CANDIDATE_PENDING_HUMAN_REVIEW", names=CANDIDATE_NAMES, label="candidate index")
    for row in index["artifacts"]:
        raw = candidate[row["path"]]
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw),
                 f"candidate artifact hash mismatch: {row['path']}")
    manifest = _json(candidate["manifest.json"], "candidate manifest")
    _require(manifest["status"] == "CANDIDATE_PENDING_HUMAN_REVIEW"
             and manifest["formal_task_release"] is False
             and manifest["scientific_release_allowed"] is False
             and manifest["source_audit_status"] == "NOT_STARTED"
             and manifest["generated_surface_audit_status"] == "NOT_STARTED",
             "candidate historical boundary mismatch")
    _require(manifest["capture_manifest_sha256"] == CAPTURE_MANIFEST_SHA256
             and manifest["capture_index_sha256"] == CAPTURE_INDEX_SHA256
             and manifest["source_count_semantics"] == "one upstream storm trajectory, not statistically independent samples",
             "candidate capture lineage mismatch")
    _require((manifest["source_count_semantics"], manifest["source_domain"], manifest["source_group_count"],
              manifest["semantic_core_count"], manifest["task_count"], manifest["snapshot_count"], manifest["event_count"],
              manifest["update_events"], manifest["value_changing_updates"], manifest["equal_value_update_events"]) ==
             ("one upstream storm trajectory, not statistically independent samples", "weather", 1, 1, 1, 7, 7, 6, 5, 1),
             "candidate counts mismatch")
    tasks = [MemUpdateTaskV3.model_validate(_json(line, "candidate task"))
             for line in candidate["tasks.jsonl"].splitlines()]
    _require(len(tasks) == 1 and replay_task_v3(tasks[0]).valid, "candidate replay invalid")
    task = tasks[0]
    _require(len(task.events) == 7 and len(task.actions) == 7 and len(task.queries) == 1,
             "candidate task cardinality mismatch")
    _require(task.actions[0].operation.value == "ADD"
             and all(action.operation.value == "UPDATE" for action in task.actions[1:])
             and sum(task.actions[i].value == task.actions[i - 1].value for i in range(1, 7)) == 1,
             "candidate action sequence mismatch")
    _require(task.metadata.extra["scientific_release_allowed"] is False
             and task.source.provenance["source_audit_status"] == "NOT_STARTED"
             and task.source.provenance["generated_surface_audit_status"] == "NOT_STARTED",
             "candidate task boundary mismatch")
    audit = _json(candidate["audit_manifest.json"], "candidate audit manifest")
    _require(audit["status"] == "NOT_STARTED" and audit["item_count"] == 9
             and len(audit["items"]) == 9, "candidate audit boundary mismatch")
    _require(all(_digest(item["material"]) == item["binding_sha256"] for item in audit["items"]),
             "candidate audit binding mismatch")
    kinds = {kind: [item for item in audit["items"] if item["kind"] == kind]
             for kind in ("source_admission", "source_snapshot", "generated_surface")}
    _require([len(kinds[kind]) for kind in kinds] == [1, 7, 1], "candidate audit kind counts mismatch")
    surface = kinds["generated_surface"][0]["material"]
    _require(surface["task_id"] == task.task_id
             and surface["canonical_task_sha256"] == _digest(task.model_dump(mode="json"))
             and surface["semantic_task_sha256"] == semantic_task_hash_v3(task)
             and surface["event_count"] == 7, "candidate surface binding mismatch")
    core_rows = [ _json(line, "semantic core") for line in candidate["semantic_cores.jsonl"].splitlines() ]
    _require(len(core_rows) == 1 and core_rows[0]["core_id"] == task.metadata.split_key.semantic_core_id,
             "candidate semantic core binding mismatch")
    return manifest, tasks, audit


def _validate_capture(capture_root: Path, candidate: dict[str, bytes]) -> dict:
    root = _root(capture_root, "capture")
    manifest_path = _regular_file(root / "capture_manifest.json", "capture manifest")
    index_path = _regular_file(root / "capture_index.json", "capture index")
    manifest_raw, index_raw = manifest_path.read_bytes(), index_path.read_bytes()
    _require(_sha(manifest_raw) == CAPTURE_MANIFEST_SHA256, "capture manifest hash mismatch")
    _require(_sha(index_raw) == CAPTURE_INDEX_SHA256, "capture index hash mismatch")
    index = _json(index_raw, "capture index")
    _require(index["schema"] == "memupdatebench.family-h.cross-domain-source-capture-index.v2"
             and index["status"] == "CAPTURED_PENDING_HUMAN_REVIEW"
             and index["scientific_release_allowed"] is False, "capture index boundary mismatch")
    paths = {row["path"] for row in index["artifacts"]} | {"capture_index.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == paths, "capture artifact membership mismatch")
    for row in index["artifacts"]:
        raw = _regular_file(root / row["path"], f"capture/{row['path']}").read_bytes()
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw),
                 f"capture artifact hash mismatch: {row['path']}")
    manifest = _json(manifest_raw, "capture manifest")
    _require(manifest["source_domain"] == "weather"
             and manifest["source_type"] == "public_advisory"
             and manifest["source_kind"] == "sequential_public_advisory"
             and manifest["advisory_count"] == 7
             and manifest["formal_task_release"] is False
             and manifest["scientific_release_allowed"] is False,
             "capture metadata mismatch")
    _require((root / "normalized_records.jsonl").read_bytes() == candidate["normalized_records.jsonl"],
             "candidate/capture normalized records mismatch")
    return manifest


def _validate_review(review: dict[str, bytes], audit: dict, decisions_path: Path | None) -> tuple[dict, dict]:
    index = _validate_index(review["index.json"], REVIEW_INDEX_SHA256,
                            schema="memupdatebench.family-h.cross-domain-noaa-human-review-index.v1",
                            status="HUMAN_REVIEW_COMPLETE", names=REVIEW_NAMES, label="review index")
    _require(index["formal_task_release"] is False, "review formal release boundary mismatch")
    for row in index["artifacts"]:
        raw = review[row["path"]]
        _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw),
                 f"review artifact hash mismatch: {row['path']}")
    if decisions_path is None:
        _require(_sha(review["decisions.json"]) == DECISIONS_SHA256,
                 "archived review decisions hash mismatch")
    else:
        submitted = _regular_file(decisions_path, "submitted decisions").read_bytes()
        _require(_sha(submitted) == DECISIONS_SHA256 and submitted == review["decisions.json"],
                 "submitted review decisions binding mismatch")
    decisions = _json(review["decisions.json"], "review decisions")
    _require(decisions["status"] == "COMPLETED" and decisions["candidate_index_sha256"] == CANDIDATE_INDEX_SHA256,
             "review decision status/binding mismatch")
    expected = {item["audit_id"]: item["binding_sha256"] for item in audit["items"]}
    rows = decisions.get("decisions")
    _require(isinstance(rows, list) and len(rows) == 9 and {row["audit_id"] for row in rows} == set(expected),
             "review decision cardinality mismatch")
    _require(all(row["binding_sha256"] == expected[row["audit_id"]]
                 and row["decision"] == "RELEASE_READY"
                 and isinstance(row.get("rationale"), str) and row["rationale"].strip() for row in rows),
             "review decisions are not all release-ready")
    receipt = _json(review["review_completion_receipt.json"], "review completion receipt")
    expected_receipt = {
        "candidate_index_sha256": CANDIDATE_INDEX_SHA256,
        "capture_index_sha256": CAPTURE_INDEX_SHA256,
        "source_capture_sha256": CAPTURE_MANIFEST_SHA256,
        "decisions_sha256": DECISIONS_SHA256,
        "decision_count": 9,
        "counts": {"RELEASE_READY": 9},
        "status": "HUMAN_REVIEW_COMPLETE",
        "formal_task_release": False,
        "scientific_release_allowed": False,
        "source_semantics": "one sequential weather trajectory, not statistically independent samples",
    }
    for key, value in expected_receipt.items():
        _require(receipt.get(key) == value, f"review receipt {key} mismatch")
    _require(receipt["audit_manifest_sha256"] == _sha(_canonical(audit)),
             "review receipt audit binding malformed")
    _require(receipt["reviewer"] == decisions["reviewer"] and receipt["snapshot_ready"] == 7
             and receipt["source_admission_ready"] == 1 and receipt["surface_ready"] == 1,
             "review receipt coverage mismatch")
    return decisions, receipt


def _build_payload(candidate: dict[str, bytes], review: dict[str, bytes], manifest: dict,
                   decisions: dict, receipt: dict, audit: dict, capture: dict) -> dict[str, bytes]:
    tasks = [MemUpdateTaskV3.model_validate(_json(line, "candidate task")) for line in candidate["tasks.jsonl"].splitlines()]
    task = tasks[0]
    input_sha256 = {
        "candidate_index": CANDIDATE_INDEX_SHA256,
        "review_index": REVIEW_INDEX_SHA256,
        "decisions": DECISIONS_SHA256,
        "review_completion_receipt": _sha(review["review_completion_receipt.json"]),
        "capture_manifest": CAPTURE_MANIFEST_SHA256,
        "capture_index": CAPTURE_INDEX_SHA256,
    }
    release_manifest = {
        "schema": RELEASE_SCHEMA,
        "status": "FINAL_APPROVED",
        "formal_task_release": True,
        "scientific_release_allowed": False,
        "benchmark_accuracy_claimed": False,
        "model_backend_status": "NOT_RUN",
        "runtime_state_metrics": None,
        "runtime_retrieval_metrics": None,
        "answer_metrics": None,
        "source_count": 1,
        "source_count_semantics": "one upstream storm trajectory, not statistically independent samples",
        "source_domain": "weather",
        "source_type": "public_advisory",
        "source_kind": "sequential_public_advisory",
        "revision_only": False,
        "counts": {"records": 7, "cores": 1, "tasks": 1, "events": 7, "decisions": 9},
        "update_counts": {"ADD": 1, "UPDATE": 6, "equal_value_updates": 1},
        "input_sha256": input_sha256,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "capture_index_sha256": CAPTURE_INDEX_SHA256,
        "capture_status": capture["status"],
        "historical_candidate_status": manifest["status"],
        "historical_review_status": receipt["status"],
        "reviewer": decisions["reviewer"],
        "reviewer_identity": "self-reported; not cryptographically authenticated",
        "source_boundary": SOURCE_BOUNDARY,
        "raw_capture_policy": "Raw NOAA/NHC capture is retained as hash-bound input metadata only; it is not copied into this public release.",
        "claim_boundary": "Bounded cross-domain NOAA task release only; no benchmark accuracy, model, backend, or statistically independent-source claim.",
        "task_id": task.task_id,
        "semantic_task_sha256": semantic_task_hash_v3(task),
        "reviewed_audit_item_count": len(audit["items"]),
    }
    archive = {
        **{name: candidate[name] for name in CANDIDATE_NAMES if name != "index.json"},
        "candidate_index.json": candidate["index.json"],
        "decisions.json": review["decisions.json"],
        "review_completion_receipt.json": review["review_completion_receipt.json"],
        "human_review_index.json": review["index.json"],
        "release_manifest.json": _canonical(release_manifest),
    }
    release_index = {
        "schema": RELEASE_INDEX_SCHEMA,
        "status": "FINAL_APPROVED",
        "formal_task_release": True,
        "scientific_release_allowed": False,
        "input_sha256": input_sha256,
        "artifacts": [{"path": name, "bytes": len(raw), "sha256": _sha(raw)}
                      for name, raw in sorted(archive.items())],
    }
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
             and manifest["benchmark_accuracy_claimed"] is False
             and all(manifest[key] is None for key in ("runtime_state_metrics", "runtime_retrieval_metrics", "answer_metrics")),
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
             and input_sha256["review_index"] == REVIEW_INDEX_SHA256,
             "release lineage mismatch")
    return manifest, tasks, index


def read_release(root: Path | str, expected_index_sha256: str) -> dict[str, Any]:
    """Read and validate a published release without writing any files."""
    release_root = _root(Path(root), "release")
    payload = _read_exact(release_root, RELEASE_NAMES, "release")
    manifest, tasks, index = _validate_release_payload(payload, expected_index_sha256)
    return {"manifest": manifest, "tasks": tasks, "index": index,
            "root": release_root.resolve(), "index_sha256": expected_index_sha256}


def publish_release(candidate_root: Path | str, review_root: Path | str, decisions_path: Path | str,
                    capture_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    """Publish a no-replace release from the exact reviewed candidate bytes."""
    candidate_root, review_root = Path(candidate_root), Path(review_root)
    decisions_path, capture_root, output_root = Path(decisions_path), Path(capture_root), Path(output_root)
    candidate = _read_exact(candidate_root, CANDIDATE_NAMES, "candidate")
    review = _read_exact(review_root, REVIEW_NAMES, "review")
    _regular_file(decisions_path, "submitted decisions")
    capture = _validate_capture(capture_root, candidate)
    manifest, tasks, audit = _validate_candidate(candidate)
    decisions, receipt = _validate_review(review, audit, decisions_path)
    payload = _build_payload(candidate, review, manifest, decisions, receipt, audit, capture)
    release_index_sha = _sha(payload["release_index.json"])
    _validate_release_payload(payload, release_index_sha)
    _assert_no_reparse_components(output_root, "output")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    source_paths = tuple(candidate_root / name for name in CANDIDATE_NAMES)
    source_paths += tuple(review_root / name for name in REVIEW_NAMES)
    source_paths += (decisions_path, capture_root / "capture_manifest.json", capture_root / "capture_index.json")
    candidate_snapshot, review_snapshot = dict(candidate), dict(review)
    decisions_snapshot = decisions_path.read_bytes()
    capture_manifest_snapshot = (capture_root / "capture_manifest.json").read_bytes()
    capture_index_snapshot = (capture_root / "capture_index.json").read_bytes()

    def pre_publish() -> None:
        _require(_read_exact(candidate_root, CANDIDATE_NAMES, "candidate") == candidate_snapshot,
                 "candidate changed before publication")
        _require(_read_exact(review_root, REVIEW_NAMES, "review") == review_snapshot,
                 "review changed before publication")
        _require(decisions_path.read_bytes() == decisions_snapshot,
                 "submitted decisions changed before publication")
        _require((capture_root / "capture_manifest.json").read_bytes() == capture_manifest_snapshot
                 and (capture_root / "capture_index.json").read_bytes() == capture_index_snapshot,
                 "capture changed before publication")

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
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_release(args.candidate_root, args.review_root, args.decisions,
                                 args.capture_root, args.output_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({**result, "root": str(result["root"]),
                      "tasks": [task.model_dump(mode="json") for task in result["tasks"]]},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
