from copy import deepcopy

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.v3.task import MemUpdateTaskV3


H = "a" * 64


def task_payload():
    key = {"object_type": "slot", "namespace": "n", "entity": "e", "attribute": "a", "subkey": None}
    return {
        "task_id": "t",
        "task_family": "f",
        "difficulty": "easy",
        "source": {
            "source_id": "s", "source_type": "synthetic", "source_uri": None,
            "license_or_privacy": "synthetic", "raw_hash": H, "normalized_hash": H,
            "normalization_version": "n1", "provenance": {"object_type": "source-a"},
            "generator": {"generator_name": "g", "seed": 1, "config_sha256": H, "code_revision": "r", "compiler_version": "3"},
        },
        "events": [
            {"event_id": f"e{i}", "sequence_index": i, "raw_text": "x", "normalized_text": "x", "role": "neutral"}
            for i in range(3)
        ],
        "target_objects": [key],
        "queries": [{"query_id": "q", "query_type": "current", "text": "?", "selector": {"kind": "current"}, "target_object_keys": [key], "answer_schema": "string", "evaluation_mode": "state_direct"}],
        "version_history": [{"object_key": key, "entries": [
            {"version_index": 0, "status": "present", "value": "v0", "valid_from_event_id": "e0", "valid_until_event_id": "e1", "source_event_ids": ["e0"]},
            {"version_index": 1, "status": "present", "value": "v1", "valid_from_event_id": "e1", "valid_until_event_id": "e2", "source_event_ids": ["e1"]},
            {"version_index": 2, "status": "present", "value": "v2", "valid_from_event_id": "e2", "source_event_ids": ["e2"]},
        ]}],
        "gold_evidence": [{"query_id": "q", "answer": "v2", "supporting_object_keys": [key], "supporting_event_ids": ["e2"], "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e2"]}], "final_derivation_step_id": "read"}],
        "metadata": {"split": "test", "split_key": {"semantic_core_id": "c", "source_group_id": "s", "trajectory_id": "t", "split_policy_version": "3"}, "profile_name": "easy", "generation_config_hash": H, "compiler_version": "3"},
    }


def test_task_rejects_duplicate_empty_inverted_and_partial_ledgers() -> None:
    base = task_payload()
    duplicate = deepcopy(base)
    duplicate["version_history"].append(deepcopy(duplicate["version_history"][0]))
    with pytest.raises(ValidationError, match="duplicate canonical"):
        MemUpdateTaskV3.model_validate(duplicate)

    empty = deepcopy(base)
    empty["version_history"][0]["entries"] = []
    with pytest.raises(ValidationError, match="at least 1|nonempty"):
        MemUpdateTaskV3.model_validate(empty)

    inverted = deepcopy(base)
    inverted["version_history"][0]["entries"][0]["valid_from_event_id"] = "e2"
    with pytest.raises(ValidationError, match="ordered"):
        MemUpdateTaskV3.model_validate(inverted)

    partial = deepcopy(base)
    partial["version_history"][0]["entries"][0]["valid_until_event_id"] = None
    with pytest.raises(ValidationError, match="partial"):
        MemUpdateTaskV3.model_validate(partial)


def test_unqueried_declared_ledger_must_also_be_nonempty() -> None:
    payload = task_payload()
    second = {"object_type": "slot", "namespace": "n", "entity": "other", "attribute": "a", "subkey": None}
    payload["target_objects"].append(second)
    payload["version_history"].append({"object_key": second, "entries": []})
    with pytest.raises(ValidationError, match="at least 1|nonempty"):
        MemUpdateTaskV3.model_validate(payload)


def test_task_binds_answer_types_and_evidence_to_selected_version() -> None:
    wrong_type = task_payload()
    wrong_type["queries"][0]["answer_schema"] = "number"
    with pytest.raises(ValidationError, match="answer_schema"):
        MemUpdateTaskV3.model_validate(wrong_type)

    wrong_support = task_payload()
    wrong_support["gold_evidence"][0]["supporting_event_ids"] = ["e1"]
    wrong_support["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = ["e1"]
    with pytest.raises(ValidationError, match="selector-selected"):
        MemUpdateTaskV3.model_validate(wrong_support)


def test_g_minimum_hops_is_enforced() -> None:
    payload = task_payload()
    payload["queries"][0].update({"query_type": "update_sensitive_multi_hop", "synthesis": {"kind": "update_sensitive_multi_hop", "minimum_hops": 3}})
    payload["gold_evidence"][0]["derivation_steps"] = [
        {"step_id": "read", "operation": "read", "supporting_event_ids": ["e2"]},
        {"step_id": "answer", "operation": "answer", "input_step_ids": ["read"]},
    ]
    payload["gold_evidence"][0]["final_derivation_step_id"] = "answer"
    with pytest.raises(ValidationError, match="minimum_hops"):
        MemUpdateTaskV3.model_validate(payload)


def test_task_and_gold_json_reject_nonfinite_values_before_hashing() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        task_json = task_payload()
        task_json["source"]["provenance"] = {"nested": [bad]}
        with pytest.raises(ValidationError):
            MemUpdateTaskV3.model_validate(task_json)
        gold_json = task_payload()
        gold_json["queries"][0]["answer_schema"] = "object"
        gold_json["gold_evidence"][0]["answer"] = {"nested": bad}
        with pytest.raises(ValidationError):
            MemUpdateTaskV3.model_validate(gold_json)


def test_application_object_type_values_remain_semantically_significant() -> None:
    left_data = task_payload()
    left_data["queries"][0]["answer_schema"] = "object"
    left_data["gold_evidence"][0]["answer"] = {"object_type": "answer-a"}
    right_data = deepcopy(left_data)
    right_data["gold_evidence"][0]["answer"] = {"object_type": "answer-b"}
    assert MemUpdateTaskV3.model_validate(left_data).semantic_hash != MemUpdateTaskV3.model_validate(right_data).semantic_hash

    provenance_changed = deepcopy(left_data)
    provenance_changed["source"]["provenance"]["object_type"] = "source-b"
    assert MemUpdateTaskV3.model_validate(left_data).semantic_hash != MemUpdateTaskV3.model_validate(provenance_changed).semantic_hash
