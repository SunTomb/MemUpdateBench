"""Attest an existing native multi-object canary without executing a provider."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_PREPARATION_SHA = "cb50151dceba0256f1962f00e51bfbd86c0f8383fa895d434e0f73f56ba578cd"
EXPECTED_RUNTIME = {
    "node": "Tang3",
    "qdrant_version": "1.19.0",
    "runtime_revision": "74f3e85b9473c62560006c043e13737ce6b48412",
    "qdrant_binary_sha256": "abfe97e1d0225111dec2f048790428f151846c8a049eefc85328b6c9eccaf419",
    "qdrant_env_sha256": "c27be94ba7ed50e4d23d78a5e4a350202f20ac74e2a40bf40f77d6069d3696fc",
}
EXPECTED_WITNESSES = {"query_points": 8, "scroll": 154, "retrieve": 74, "upsert": 54, "delete": 60,
                     "collection_exists": 8, "create_collection": 8, "get_collection": 8}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be an object")
    return value


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _artifact(root: Path, name: str) -> bytes:
    path = root / name
    _require(path.is_file() and not path.is_symlink(), f"missing artifact {name}")
    return path.read_bytes()


def _validate_evidence_index(evidence_root: Path) -> tuple[dict[str, Any], bytes]:
    raw = _artifact(evidence_root, "index.json")
    index = json.loads(raw)
    _require(index.get("schema") == "memupdatebench.native-qdrant-canary-evidence-index.v1", "evidence index schema mismatch")
    _require(index.get("status") == "OBSERVATIONS_COMPLETE", "evidence index status mismatch")
    declared = index.get("artifacts")
    _require(isinstance(declared, list), "evidence index artifacts missing")
    expected_names = {"summary.json", "observations.json", "audit.json", "profile.json", "preparation_index.json"}
    _require({item.get("path") for item in declared} == expected_names, "evidence artifact membership mismatch")
    for item in declared:
        name = item.get("path")
        _require(isinstance(name, str) and Path(name).name == name and ".." not in Path(name).parts, "unsafe evidence artifact path")
        raw_item = _artifact(evidence_root, name)
        _require(item.get("bytes") == len(raw_item) and item.get("sha256") == sha(raw_item), f"evidence artifact hash mismatch: {name}")
    return index, raw


def attest(evidence_root: Path, preparation_root: Path, preparation_index_sha256: str) -> dict[str, Any]:
    evidence_root = Path(evidence_root)
    preparation_root = Path(preparation_root)
    _require(sha(_artifact(preparation_root, "index.json")) == preparation_index_sha256, "preparation index binding mismatch")
    index, index_raw = _validate_evidence_index(evidence_root)
    _require(index.get("scientific_release_allowed") is False, "scientific release flag overclaimed")
    summary = read_json(evidence_root / "summary.json")
    observations = read_json(evidence_root / "observations.json")
    audit = read_json(evidence_root / "audit.json")
    profile = read_json(evidence_root / "profile.json")
    _require(summary.get("status") == "OBSERVATIONS_COMPLETE", "summary status mismatch")
    _require(summary.get("native_admission") == "NOT_GRANTED", "native admission overclaimed")
    _require(summary.get("answer_metrics") is None, "answer metric overclaimed")
    _require(audit.get("status") == "AUDITED" and audit.get("native_admission") == "NOT_GRANTED", "audit boundary mismatch")
    _require(all(profile.get(k) == v for k, v in EXPECTED_RUNTIME.items()), "runtime binding mismatch")
    _require(profile.get("preparation_index_sha256") == preparation_index_sha256, "profile preparation binding mismatch")
    _require(profile.get("execution_authorized") is True, "execution authorization missing")
    _require(profile.get("model_loads") == profile.get("generations") == profile.get("provider_calls") == profile.get("gpu_calls") == 0, "execution boundary mismatch")
    counts = observations.get("counts")
    _require(counts == {"mutation_attempts": 50, "mutation_completions": 50, "planned_mutation_operations": 50,
                        "reset_attempts": 48, "retrieval_attempts": 8, "retrieval_completions": 8, "setup_seed_attempts": 16}, "canary counts mismatch")
    cases = observations.get("cases")
    _require(isinstance(cases, list) and len(cases) == 8 and all(row.get("status") == "OBSERVED" for row in cases), "case observations incomplete")
    _require(all(row.get("cleanup") == {"client_closed": True, "collection_deleted": True} for row in cases), "case cleanup mismatch")
    _require(all(summary.get("cleanup", {}).get(key) is True for key in ("owned_server_stopped", "owned_worker_stopped", "temporary_removed")), "owned cleanup mismatch")
    witness_counts: dict[str, int] = {}
    for row in cases:
        for witness in row.get("observation_witnesses", ()):
            method = witness.get("method")
            witness_counts[method] = witness_counts.get(method, 0) + 1
    _require(witness_counts == EXPECTED_WITNESSES, "provider witness counts mismatch")
    _require(summary.get("counts") == counts, "summary counts mismatch")
    _require(summary.get("execution") == {"model_loads": 0, "generations": 0, "gpu_calls": 0, "paid_provider_calls": 0}, "summary execution mismatch")
    return {
        "schema": "memupdatebench.native-qdrant-canary-attestation.v1",
        "status": "BOUNDED_CANARY_ATTESTED",
        "scientific_release_allowed": False,
        "native_admission": "NOT_GRANTED",
        "independent_signer": False,
        "evidence_class": "direct_qdrant_synthetic_api_canary",
        "claim_boundary": "direct Qdrant CRUD/hash-vector retrieval only; no semantic answer, model, broad native-manager or statistical claim",
        "input_bindings": {"preparation_index_sha256": preparation_index_sha256,
                           "evidence_index_sha256": sha(index_raw),
                           "profile_sha256": sha(_artifact(evidence_root, "profile.json"))},
        "runtime": EXPECTED_RUNTIME,
        "counts": {"cases": 8, "observed_cases": 8, **counts},
        "provider_witnesses": witness_counts,
        "cleanup": {"all_case_clients_closed": True, "all_case_collections_deleted": True,
                     "owned_server_stopped": summary.get("cleanup", {}).get("owned_server_stopped"),
                     "owned_worker_stopped": summary.get("cleanup", {}).get("owned_worker_stopped"),
                     "temporary_removed": summary.get("cleanup", {}).get("temporary_removed")},
        "execution": summary["execution"],
        "producer_audit": audit,
    }


def publish(report: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    payload = canonical(report)
    (output_root / "attestation.json").write_bytes(payload)
    index = {"schema": "memupdatebench.native-qdrant-canary-attestation-index.v1",
             "status": report["status"], "scientific_release_allowed": False,
             "native_admission": "NOT_GRANTED",
             "artifacts": [{"path": "attestation.json", "bytes": len(payload), "sha256": sha(payload)}]}
    (output_root / "index.json").write_bytes(canonical(index))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--preparation-index-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = attest(args.evidence_root, args.preparation_root, args.preparation_index_sha256)
    print(json.dumps(publish(report, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
