from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import vnext_reconcile_family_h_evidence as reconcile


def source():
    return {
        "status": "FINAL_APPROVED",
        "scientific_release_allowed": False,
        "counts": {"semantic_cores": 3, "tasks": 3, "events": 27, "sources": 3, "records": 27, "decisions": 33},
        "model_backend_status": "NOT_RUN",
    }

def state():
    return {
        "status": "COMPLETE", "scientific_release_allowed": False,
        "state_step_matches": 27, "events": 27, "current_source_event_links": 27,
        "final_state_matches": 3, "retrieval_typed_object_matches": 3,
        "stale_same_slot_count_total": 0, "model_loads": 0, "generations": 0,
        "provider_model_calls": 0, "cleanup": {"client_closed": True, "collection_gone": True, "reset_empty": True},
        "release_index_sha256": "a" * 64,
    }

def correction():
    return {
        "status": "DISCREPANCY_RECORDED_WITHOUT_REWRITING_PRIOR_RECEIPT",
        "state_retrieval_metrics_remain_valid": True,
        "scientific_release_allowed": False,
        "discrepancies": [{"classification": "STALE_COPIED_PROVENANCE_HASH"}, {"classification": "UNRESOLVED_ARCHIVE_IDENTITY_MISMATCH"}],
    }

def answer():
    return {
        "status": "FAMILY_H_INDEPENDENT_QWEN_RETRIEVED_ANSWERS_COMPLETE",
        "scientific_release_allowed": False, "tasks": 3, "model_loads": 1, "generations": 3,
        "format_valid_count": 3, "typed_object_matches": 3, "typed_object_accuracy": 1.0,
        "retries": 0, "provider_model_calls": 0, "raw_generation_text_saved": False,
        "release_index_sha256": "a" * 64, "state_index_sha256": "b" * 64,
    }

def test_reconcile_marks_existing_provenance_discrepancy_diagnostic_only():
    result = reconcile.reconcile_values(source(), state(), correction(), answer(),
                                         source_manifest_sha256="a" * 64, state_receipt_sha256="b" * 64, answer_receipt_sha256="c" * 64)
    assert result["status"] == "DIAGNOSTIC_ONLY"
    assert result["scientific_release_allowed"] is False
    assert result["evidence_classes"] == {
        "source": "audited_task_release",
        "state_retrieval": "external_manager_state_retrieval",
        "answer": "retrieved_context_qwen_replay",
        "runtime": "inherited_receipt_not_requalified",
    }
    assert result["provenance"]["unresolved_discrepancies"] == [
        "STALE_COPIED_PROVENANCE_HASH", "UNRESOLVED_ARCHIVE_IDENTITY_MISMATCH"
    ]

def test_reconcile_rejects_answer_release_binding_drift():
    bad = answer(); bad["release_index_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="release index binding"):
        reconcile.reconcile_values(source(), state(), correction(), bad,
                                   source_manifest_sha256="a" * 64, state_receipt_sha256="b" * 64, answer_receipt_sha256="c" * 64)

def test_publish_is_no_replace_and_reopenable(tmp_path):
    values = reconcile.reconcile_values(source(), state(), correction(), answer(),
                                        source_manifest_sha256="a" * 64, state_receipt_sha256="b" * 64, answer_receipt_sha256="c" * 64)
    first = reconcile.publish(values, tmp_path / "package")
    assert first["status"] == "DIAGNOSTIC_ONLY"
    with pytest.raises(FileExistsError):
        reconcile.publish(values, tmp_path / "package")
    index = json.loads((tmp_path / "package" / "index.json").read_text())
    assert index["status"] == "DIAGNOSTIC_ONLY"
    assert index["artifacts"][0]["path"] == "reconciliation.json"
