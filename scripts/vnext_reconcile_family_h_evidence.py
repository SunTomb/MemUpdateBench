"""Reconcile existing Family-H evidence without executing any runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/vnext/family_h_independent/v1_published/release_manifest.json"
SOURCE_INDEX = ROOT / "data/vnext/family_h_independent/v1_published/release_index.json"
STATE = ROOT / "results/vnext/family_h_state_completion_receipt_20260925_v2.json"
CORRECTION = ROOT / "results/vnext/family_h_state_receipt_correction_20260927_v1.json"
ANSWER = ROOT / "results/vnext/family_h_answer_results_receipt_20260928_v6.json"
DEFAULT_OUTPUT = ROOT / "results/vnext/family_h_evidence_reconciliation_20261010_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconcile_values(source, state, correction, answer, *, source_manifest_sha256, state_receipt_sha256, answer_receipt_sha256):
    require(source.get("status") == "FINAL_APPROVED", "source release status")
    require(source.get("scientific_release_allowed") is False, "source scientific flag")
    counts = source.get("counts", {})
    require({key: counts.get(key) for key in ("semantic_cores", "tasks", "events", "sources", "records", "decisions")} ==
            {"semantic_cores": 3, "tasks": 3, "events": 27, "sources": 3, "records": 27, "decisions": 33}, "source counts")
    require(source.get("model_backend_status") == "NOT_RUN", "source model status")

    require(state.get("status") == "COMPLETE", "state status")
    require(state.get("scientific_release_allowed") is False, "state scientific flag")
    require(state.get("state_step_matches") == state.get("events") == 27, "state step metrics")
    require(state.get("current_source_event_links") == 27, "state source links")
    require(state.get("final_state_matches") == 3 and state.get("retrieval_typed_object_matches") == 3, "state final metrics")
    require(state.get("stale_same_slot_count_total") == 0, "state stale metric")
    require(state.get("model_loads") == state.get("generations") == state.get("provider_model_calls") == 0, "state execution counts")
    require(all(state.get("cleanup", {}).get(key) is True for key in ("client_closed", "collection_gone", "reset_empty")), "state cleanup")
    require(state.get("release_index_sha256") == source_manifest_sha256, "state release index binding")

    require(correction.get("status") == "DISCREPANCY_RECORDED_WITHOUT_REWRITING_PRIOR_RECEIPT", "correction status")
    require(correction.get("state_retrieval_metrics_remain_valid") is True, "correction metric status")
    require(correction.get("scientific_release_allowed") is False, "correction scientific flag")
    discrepancies = [item.get("classification") for item in correction.get("discrepancies", [])]
    require(set(discrepancies) == {"STALE_COPIED_PROVENANCE_HASH", "UNRESOLVED_ARCHIVE_IDENTITY_MISMATCH"}, "correction discrepancies")

    require(answer.get("status") == "FAMILY_H_INDEPENDENT_QWEN_RETRIEVED_ANSWERS_COMPLETE", "answer status")
    require(answer.get("scientific_release_allowed") is False, "answer scientific flag")
    require(answer.get("tasks") == 3 and answer.get("model_loads") == 1 and answer.get("generations") == 3, "answer counts")
    require(answer.get("format_valid_count") == answer.get("typed_object_matches") == 3, "answer metrics")
    require(answer.get("typed_object_accuracy") == 1.0 and answer.get("retries") == 0, "answer accuracy fields")
    require(answer.get("provider_model_calls") == 0 and answer.get("raw_generation_text_saved") is False, "answer execution fields")
    require(answer.get("release_index_sha256") == source_manifest_sha256, "answer release index binding")

    return {
        "schema": "memupdatebench.family-h.evidence-reconciliation.v1",
        "status": "DIAGNOSTIC_ONLY",
        "scientific_release_allowed": False,
        "claim_boundary": "source release, external-manager state/retrieval, and inherited retrieved-context answer evidence; no new execution or scientific promotion",
        "evidence_classes": {
            "source": "audited_task_release",
            "state_retrieval": "external_manager_state_retrieval",
            "answer": "retrieved_context_qwen_replay",
            "runtime": "inherited_receipt_not_requalified",
        },
        "source": {"release_index_sha256": source_manifest_sha256, "release_manifest_sha256": None, "semantic_cores": 3, "tasks": 3, "events": 27, "sources": 3, "decisions": 33},
        "state_retrieval": {"status": "COMPLETE", "state_step_matches": 27, "source_event_links": 27, "final_state_matches": 3, "retrieval_matches": 3, "stale_same_slot": 0, "new_model_loads": 0, "new_generations": 0, "cleanup": state["cleanup"]},
        "answer": {"status": "INHERITED_COMPLETE", "tasks": 3, "model_loads": 1, "generations": 3, "format_valid": 3, "typed_object_matches": 3, "typed_object_accuracy": 1.0, "raw_generation_text_saved": False},
        "provenance": {"status": "BLOCKED", "unresolved_discrepancies": sorted(discrepancies), "correction_receipt_sha256": sha256_bytes(canonical(correction))},
        "input_receipts": {"source_manifest_sha256": source_manifest_sha256, "state_receipt_sha256": state_receipt_sha256, "answer_receipt_sha256": answer_receipt_sha256},
        "execution": {"new_model_loads": 0, "new_generations": 0, "provider_calls": 0, "qdrant_calls": 0},
    }


def publish(reconciliation, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    payload = canonical(reconciliation)
    (output_root / "reconciliation.json").write_bytes(payload)
    index = {"schema": "memupdatebench.family-h.evidence-reconciliation-index.v1", "status": reconciliation["status"], "scientific_release_allowed": False, "artifacts": [{"path": "reconciliation.json", "bytes": len(payload), "sha256": sha256_bytes(payload)}]}
    (output_root / "index.json").write_bytes(canonical(index))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--source-index", type=Path, default=SOURCE_INDEX)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--correction", type=Path, default=CORRECTION)
    parser.add_argument("--answer", type=Path, default=ANSWER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    values = reconcile_values(read_json(args.source), read_json(args.state), read_json(args.correction), read_json(args.answer), source_manifest_sha256=sha256_file(args.source_index), state_receipt_sha256=sha256_file(args.state), answer_receipt_sha256=sha256_file(args.answer))
    values["source"]["release_manifest_sha256"] = sha256_file(args.source)
    index = publish(values, args.output_root)
    print(json.dumps({"status": values["status"], "scientific_release_allowed": False, "index_sha256": sha256_bytes(canonical(index)), "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
