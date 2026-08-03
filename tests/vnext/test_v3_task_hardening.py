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


def test_missing_event_anchor_is_a_stable_validation_error() -> None:
    payload = task_payload()
    payload["queries"][0].update({
        "query_type": "point_in_time",
        "selector": {"kind": "event_anchor", "event_id": "missing-event"},
    })
    with pytest.raises(ValidationError, match="unknown event anchor|missing event anchor"):
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


def test_semantic_hash_ignores_surface_text_and_consistent_local_id_renaming() -> None:
    base = task_payload()
    surface = deepcopy(base)
    surface["events"][0]["raw_text"] = "paraphrased raw surface"
    surface["events"][0]["normalized_text"] = "paraphrased normalized surface"
    surface["queries"][0]["text"] = "different question wording"
    surface["source"]["source_id"] = "surface-source"
    surface["source"]["normalized_hash"] = "b" * 64
    surface["source"]["provenance"] = {"revision": "different", "seed": 999}
    surface["source"]["generator"]["seed"] = 999
    surface["source"]["generator"]["code_revision"] = "different-revision"
    surface["source"]["generator"]["compiler_version"] = "different-compiler"
    surface["events"][0]["metadata"] = {"surface_variant": "different"}
    assert MemUpdateTaskV3.model_validate(base).semantic_hash == MemUpdateTaskV3.model_validate(surface).semantic_hash

    anchored = deepcopy(base)
    anchored["events"][0]["source_anchor"] = {"document_id": "doc", "section_id": "s1", "object_type": "semantic-anchor"}
    reanchored = deepcopy(anchored)
    reanchored["events"][0]["source_anchor"]["section_id"] = "s2"
    assert MemUpdateTaskV3.model_validate(anchored).semantic_hash != MemUpdateTaskV3.model_validate(reanchored).semantic_hash

    renamed = deepcopy(base)
    renamed["task_id"] = "renamed-task"
    event_map = {"e0": "x0", "e1": "x1", "e2": "x2"}
    for event in renamed["events"]:
        event["event_id"] = event_map[event["event_id"]]
    for entry in renamed["version_history"][0]["entries"]:
        for field in ("valid_from_event_id", "valid_until_event_id"):
            if entry.get(field) is not None:
                entry[field] = event_map[entry[field]]
        entry["source_event_ids"] = [event_map[item] for item in entry["source_event_ids"]]
    renamed["queries"][0]["query_id"] = "renamed-query"
    renamed["gold_evidence"][0]["query_id"] = "renamed-query"
    renamed["gold_evidence"][0]["supporting_event_ids"] = [event_map[item] for item in renamed["gold_evidence"][0]["supporting_event_ids"]]
    step = renamed["gold_evidence"][0]["derivation_steps"][0]
    step["step_id"] = "renamed-step"
    step["supporting_event_ids"] = [event_map[item] for item in step["supporting_event_ids"]]
    renamed["gold_evidence"][0]["final_derivation_step_id"] = "renamed-step"
    assert MemUpdateTaskV3.model_validate(base).semantic_hash == MemUpdateTaskV3.model_validate(renamed).semantic_hash


def test_derivation_hash_uses_ordered_operands_but_not_topological_list_order_or_ids() -> None:
    base = task_payload()
    base["gold_evidence"][0]["supporting_event_ids"] = ["e1", "e2"]
    base["gold_evidence"][0]["derivation_steps"] = [
        {"step_id": "left", "operation": "read", "supporting_event_ids": ["e2"]},
        {"step_id": "right", "operation": "read", "supporting_event_ids": ["e1"]},
        {"step_id": "root", "operation": "subtract", "input_step_ids": ["left", "right"]},
    ]
    base["gold_evidence"][0]["final_derivation_step_id"] = "root"

    topological_reorder = deepcopy(base)
    topological_reorder["gold_evidence"][0]["derivation_steps"][:2] = reversed(
        topological_reorder["gold_evidence"][0]["derivation_steps"][:2]
    )
    assert MemUpdateTaskV3.model_validate(base).semantic_hash == MemUpdateTaskV3.model_validate(topological_reorder).semantic_hash

    renamed = deepcopy(base)
    for step, new_id in zip(renamed["gold_evidence"][0]["derivation_steps"], ("x", "y", "z")):
        step["step_id"] = new_id
    renamed["gold_evidence"][0]["derivation_steps"][2]["input_step_ids"] = ["x", "y"]
    renamed["gold_evidence"][0]["final_derivation_step_id"] = "z"
    assert MemUpdateTaskV3.model_validate(base).semantic_hash == MemUpdateTaskV3.model_validate(renamed).semantic_hash

    swapped = deepcopy(base)
    swapped["gold_evidence"][0]["derivation_steps"][2]["input_step_ids"] = ["right", "left"]
    assert MemUpdateTaskV3.model_validate(base).semantic_hash != MemUpdateTaskV3.model_validate(swapped).semantic_hash


def test_derivation_hash_preserves_shared_dag_identity_without_tree_expansion() -> None:
    shared = task_payload()
    shared["gold_evidence"][0]["derivation_steps"] = [
        {"step_id": "r", "operation": "read", "supporting_event_ids": ["e2"]},
        {"step_id": "a", "operation": "left", "input_step_ids": ["r"]},
        {"step_id": "b", "operation": "right", "input_step_ids": ["r"]},
        {"step_id": "f", "operation": "combine", "input_step_ids": ["a", "b"]},
    ]
    shared["gold_evidence"][0]["final_derivation_step_id"] = "f"

    duplicated = deepcopy(shared)
    duplicated["gold_evidence"][0]["derivation_steps"] = [
        {"step_id": "r1", "operation": "read", "supporting_event_ids": ["e2"]},
        {"step_id": "r2", "operation": "read", "supporting_event_ids": ["e2"]},
        {"step_id": "a", "operation": "left", "input_step_ids": ["r1"]},
        {"step_id": "b", "operation": "right", "input_step_ids": ["r2"]},
        {"step_id": "f", "operation": "combine", "input_step_ids": ["a", "b"]},
    ]
    assert MemUpdateTaskV3.model_validate(shared).semantic_hash != MemUpdateTaskV3.model_validate(duplicated).semantic_hash

    moderate = task_payload()
    steps = [
        {"step_id": "n0", "operation": "seed0", "supporting_event_ids": ["e2"]},
        {"step_id": "n1", "operation": "seed1", "supporting_event_ids": ["e2"]},
    ]
    for index in range(2, 28):
        steps.append({"step_id": f"n{index}", "operation": "merge", "input_step_ids": [f"n{index - 1}", f"n{index - 2}"]})
    moderate["gold_evidence"][0]["derivation_steps"] = steps
    moderate["gold_evidence"][0]["final_derivation_step_id"] = "n27"
    assert len(MemUpdateTaskV3.model_validate(moderate).semantic_hash) == 64


def test_duplicate_gold_evidence_rows_are_rejected_before_overwrite_or_hashing() -> None:
    payload = task_payload()
    contradictory = deepcopy(payload["gold_evidence"][0])
    contradictory["answer"] = "contradictory"
    contradictory["supporting_event_ids"] = ["unknown-event"]
    contradictory["derivation_steps"][0]["supporting_event_ids"] = ["unknown-event"]
    payload["gold_evidence"].insert(0, contradictory)
    with pytest.raises(ValidationError, match="duplicate.*query evidence|duplicate.*evidence"):
        MemUpdateTaskV3.model_validate(payload)


def test_duplicate_semantic_queries_are_rejected_even_with_different_evidence() -> None:
    payload = task_payload()
    duplicate_query = deepcopy(payload["queries"][0])
    duplicate_query["query_id"] = "q2"
    duplicate_evidence = deepcopy(payload["gold_evidence"][0])
    duplicate_evidence["query_id"] = "q2"
    duplicate_evidence["answer"] = "different-but-valid"
    payload["queries"].append(duplicate_query)
    payload["gold_evidence"].append(duplicate_evidence)
    with pytest.raises(ValidationError, match="duplicate semantic query"):
        MemUpdateTaskV3.model_validate(payload)


def test_semantic_hash_canonicalizes_query_and_support_sets_but_preserves_ledger_order() -> None:
    base = task_payload()
    second_query = deepcopy(base["queries"][0])
    second_query.update({"query_id": "q2", "query_type": "previous", "selector": {"kind": "previous"}})
    second_evidence = deepcopy(base["gold_evidence"][0])
    second_evidence.update({"query_id": "q2", "answer": "v1", "supporting_event_ids": ["e1", "e2"]})
    second_evidence["derivation_steps"][0]["operation"] = "collect"
    second_evidence["derivation_steps"][0]["supporting_event_ids"] = ["e1", "e2"]
    base["queries"].append(second_query)
    base["gold_evidence"].append(second_evidence)
    base["gold_evidence"][0]["supporting_event_ids"] = ["e2", "e1"]
    base["gold_evidence"][0]["derivation_steps"][0]["operation"] = "collect"
    base["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = ["e2", "e1"]

    reordered = deepcopy(base)
    reordered["queries"].reverse()
    reordered["gold_evidence"].reverse()
    reordered["gold_evidence"][1]["supporting_event_ids"].reverse()
    reordered["gold_evidence"][1]["derivation_steps"][0]["supporting_event_ids"].reverse()
    assert MemUpdateTaskV3.model_validate(base).semantic_hash == MemUpdateTaskV3.model_validate(reordered).semantic_hash

    reassociated = deepcopy(base)
    reassociated["gold_evidence"][0]["query_id"], reassociated["gold_evidence"][1]["query_id"] = (
        reassociated["gold_evidence"][1]["query_id"], reassociated["gold_evidence"][0]["query_id"],
    )
    assert MemUpdateTaskV3.model_validate(base).semantic_hash != MemUpdateTaskV3.model_validate(reassociated).semantic_hash

    duplicated = deepcopy(base)
    duplicated["gold_evidence"][0]["supporting_event_ids"] = ["e2", "e2"]
    with pytest.raises(ValidationError, match="unique"):
        MemUpdateTaskV3.model_validate(duplicated)

    semantic_order = deepcopy(base)
    semantic_order["version_history"][0]["entries"][0]["value"], semantic_order["version_history"][0]["entries"][1]["value"] = (
        semantic_order["version_history"][0]["entries"][1]["value"], semantic_order["version_history"][0]["entries"][0]["value"],
    )
    assert MemUpdateTaskV3.model_validate(base).semantic_hash != MemUpdateTaskV3.model_validate(semantic_order).semantic_hash


def test_application_object_type_values_remain_semantically_significant() -> None:
    left_data = task_payload()
    left_data["queries"][0]["answer_schema"] = "object"
    left_data["gold_evidence"][0]["answer"] = {"object_type": "answer-a"}
    right_data = deepcopy(left_data)
    right_data["gold_evidence"][0]["answer"] = {"object_type": "answer-b"}
    assert MemUpdateTaskV3.model_validate(left_data).semantic_hash != MemUpdateTaskV3.model_validate(right_data).semantic_hash

    ledger_changed = deepcopy(left_data)
    ledger_changed["version_history"][0]["entries"][-1]["value"] = {"object_type": "ledger-value"}
    assert MemUpdateTaskV3.model_validate(left_data).semantic_hash != MemUpdateTaskV3.model_validate(ledger_changed).semantic_hash
