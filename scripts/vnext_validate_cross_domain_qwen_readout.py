"""Validate redacted, no-replace cross-domain Qwen readout artifacts offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_REQUESTS = ("bea-repeat-0", "bea-repeat-1", "noaa-repeat-0", "noaa-repeat-1")
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
PREPARATION_SHA = "1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687"
PROMPT_MARKER = b"Use only the retrieved memory entries"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _artifact_members(root: Path, index: dict[str, Any]) -> dict[str, bytes]:
    if index.get("schema") != "memupdatebench.cross-domain.qwen-readout-index.v1":
        raise ValueError("unexpected artifact index schema")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact index has no artifacts")
    declared: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("artifact entry schema mismatch")
        name = item["path"]
        path = PurePosixPath(name) if isinstance(name, str) else None
        if path is None or path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError("artifact path is unsafe")
        if name in declared:
            raise ValueError("duplicate artifact path")
        declared[name] = item
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "index.json"}
    if actual != set(declared):
        raise ValueError("artifact membership mismatch")
    payloads: dict[str, bytes] = {}
    for name, item in declared.items():
        raw = (root / name).read_bytes()
        if item["bytes"] != len(raw) or item["sha256"] != sha256_bytes(raw):
            raise ValueError(f"artifact hash or bytes mismatch: {name}")
        payloads[name] = raw
    return payloads


def _redacted(payloads: dict[str, bytes]) -> bool:
    return all(b'"raw_output"' not in raw and PROMPT_MARKER not in raw for raw in payloads.values())


def _validate_rows(rows: list[dict[str, Any]], *, expected_generation_attempts: int) -> tuple[list[str], int, int, int]:
    if len(rows) != 4:
        raise ValueError("worker rows must contain four requests")
    if [row.get("request_id") for row in rows] != list(EXPECTED_REQUESTS):
        raise ValueError("worker request order mismatch")
    statuses = [row.get("status") for row in rows]
    intents = sum(row.get("attempt_intent", 0) for row in rows)
    callbacks = sum(row.get("callback_entries", 0) for row in rows)
    attempts = sum(row.get("attempts", 0) for row in rows)
    if any(type(row.get("attempt_intent")) is not int or row["attempt_intent"] < 0 for row in rows):
        raise ValueError("invalid generation intent accounting")
    if any(type(row.get("callback_entries")) is not int or row["callback_entries"] < 0 for row in rows):
        raise ValueError("invalid generation callback accounting")
    if any(type(row.get("attempts")) is not int or row["attempts"] < 0 for row in rows):
        raise ValueError("invalid generation attempt accounting")
    if attempts != expected_generation_attempts:
        raise ValueError("row/backend generation accounting mismatch")
    for row in rows:
        if row["status"] in {"TECHNICAL_FAILURE", "NOT_RUN"}:
            if row.get("prediction") is not None or row.get("generated_tokens") is not None:
                raise ValueError("technical/not-run row contains answer output")
    return statuses, intents, callbacks, attempts


def validate_root(root: Path) -> dict[str, Any]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("readout root must be a regular directory")
    index = _read_json(root / "index.json")
    payloads = _artifact_members(root, index)
    summary = _read_json(root / "summary.json")
    profile = _read_json(root / "authorization_profile.json")
    if index.get("status") != "BLOCKED" or index.get("scientific_release_allowed") is not False:
        raise ValueError("readout must remain blocked and non-scientific")
    if summary.get("status") != "BLOCKED" or summary.get("answer_metrics") is not None:
        raise ValueError("blocked readout answer metric policy violated")
    if summary.get("benchmark_accuracy_claimed") is not False or summary.get("prospective_requests") != 4:
        raise ValueError("readout claim boundary violated")
    if summary.get("profile_sha256") != sha256_bytes((root / "authorization_profile.json").read_bytes()):
        raise ValueError("profile binding mismatch")
    if profile.get("model_id") != MODEL_ID or profile.get("revision") != MODEL_REVISION or profile.get("model_tree_sha256") != MODEL_TREE:
        raise ValueError("model identity mismatch")
    if profile.get("preparation_index_sha256") != PREPARATION_SHA or profile.get("maximum_load_attempts") != 1 or profile.get("maximum_generations") != 4 or profile.get("retries") != 0:
        raise ValueError("authorization limits mismatch")
    if not all(value is True for value in summary.get("cleanup", {}).values()):
        raise ValueError("cleanup policy is incomplete")
    accounting = summary.get("execution_accounting")
    if not isinstance(accounting, dict) or any(accounting.get(key) != 0 for key in ("paid_provider_calls", "qdrant_calls", "retries")):
        raise ValueError("execution accounting policy violated")
    expected_generation_attempts = summary.get("generation_attempts")
    expected_loads = summary.get("model_loads_completed")
    if type(expected_generation_attempts) is not int or type(expected_loads) is not int or expected_generation_attempts < 0 or expected_loads < 0:
        raise ValueError("invalid summary counts")
    redaction_ok = _redacted(payloads)
    if not redaction_ok:
        raise ValueError("raw prompt/output marker found")
    statuses: list[str] = []
    intents = callbacks = attempts = 0
    if "worker_result.json" in payloads:
        worker = json.loads(payloads["worker_result.json"])
        statuses, intents, callbacks, attempts = _validate_rows(worker.get("rows", []), expected_generation_attempts=expected_generation_attempts)
        if worker.get("status") != "BLOCKED" or worker.get("postvalidation") is not True:
            raise ValueError("worker result boundary mismatch")
    elif expected_generation_attempts != 0 or expected_loads != 0:
        raise ValueError("nonzero counts require worker result")
    if summary.get("generation_intents", intents) != intents or summary.get("generation_callback_entries", callbacks) != callbacks:
        raise ValueError("summary intent/callback accounting mismatch")
    if "qualification.json" in payloads:
        qualification = json.loads(payloads["qualification.json"])
        if qualification.get("profile_sha256") != sha256_bytes((root / "authorization_profile.json").read_bytes()):
            raise ValueError("qualification binding mismatch")
    return {
        "schema": "memupdatebench.cross-domain.qwen-readout-validation.v1",
        "root_name": root.name,
        "status": summary["status"],
        "scientific_release_allowed": False,
        "answer_metrics": None,
        "model_loads_completed": expected_loads,
        "generation_intents": intents,
        "generation_callback_entries": callbacks,
        "generation_attempts": attempts,
        "request_statuses": statuses,
        "cleanup_ok": True,
        "redaction_ok": redaction_ok,
        "artifact_count": len(payloads),
        "result_index_sha256": sha256_bytes((root / "index.json").read_bytes()),
        "profile_sha256": summary["profile_sha256"],
        "preparation_index_sha256": profile["preparation_index_sha256"],
        "claim_boundary": "offline artifact validation only; no model, GPU, provider, Qdrant or accuracy claim",
    }


def publish_report(report: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    payload = canonical_bytes(report)
    (output_root / "validation.json").write_bytes(payload)
    index = {
        "schema": "memupdatebench.cross-domain.qwen-readout-validation-index.v1",
        "status": report["status"],
        "scientific_release_allowed": False,
        "artifacts": [{"path": "validation.json", "bytes": len(payload), "sha256": sha256_bytes(payload)}],
    }
    (output_root / "index.json").write_bytes(canonical_bytes(index))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    reports = [validate_root(root) for root in args.roots]
    if args.output_root:
        result = publish_report({"schema": "memupdatebench.cross-domain.qwen-readout-validation-bundle.v1", "status": "BLOCKED", "scientific_release_allowed": False, "reports": reports}, args.output_root)
    else:
        result = reports if len(reports) != 1 else reports[0]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
