from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, computed_field
from pydantic_core import PydanticSerializationError

from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerSchema,
    Difficulty,
    EvaluationMode,
    Operation,
    QueryType,
    Split,
)
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io import (
    canonical_json_bytes,
    read_models,
    semantic_task_hash,
    sha256_model,
    write_models,
)


class SerializationFixture(BaseModel):
    identifier: str
    payload: dict[str, Any]
    nullable: str | None = None

    @computed_field
    @property
    def convenience(self) -> str:
        return "not-serialized-canonically"


class LooseIdRecord(BaseModel):
    record_id: Any = None
    payload: str = "fixture"


class LoosePayloadRecord(BaseModel):
    record_id: Any = None
    payload: Any = None


class MissingIdRecord(BaseModel):
    payload: str = "fixture"


def _validated_task(payload: dict[str, Any]) -> MemUpdateTask:
    return MemUpdateTask.model_validate(payload)


def _unresolved_task(make_task, status="ambiguous") -> MemUpdateTask:
    payload = make_task().model_dump(mode="json")
    second_key = {
        **payload["target_objects"][0],
        "object_type": "profile",
        "entity": "colleague:alex",
    }
    payload["target_objects"].append(second_key)
    linked_ids = {
        "unique": ["candidate_friend"],
        "ambiguous": ["candidate_friend", "candidate_colleague"],
        "no_match": [],
    }[status]
    payload["queries"][0].update(
        query_type="unresolved_reference",
        target_object_keys=[],
        reference_candidates=[
            {
                "candidate_id": "candidate_friend",
                "object_key": payload["target_objects"][0],
                "evidence": "Friend-qualified mention in prose.",
                "source_anchors": [
                    {
                        "document_id": "query_0",
                        "section_id": "surface",
                        "start_char": 0,
                        "end_char": 4,
                    }
                ],
            },
            {
                "candidate_id": "candidate_colleague",
                "object_key": second_key,
                "evidence": "Colleague-qualified mention in prose.",
                "source_anchors": [],
            },
        ],
        surface_references=[
            {
                "reference_id": "reference_alex",
                "surface_text": "Alex",
                "normalized_text": "alex",
                "condition_kind": "same_surface_name",
                "evidence_kind": "query_span",
                "candidate_ids": linked_ids,
            }
        ],
    )
    payload["gold"]["gold_answers"] = {}
    payload["gold"]["acceptable_answers"] = {}
    if status == "unique":
        canonical = {
            "disposition": "answered",
            "resolution_status": "unique",
            "selected_candidate_ids": ["candidate_friend"],
            "abstention_reason": None,
            "value": "Qingdao",
        }
    else:
        canonical = {
            "disposition": "abstained",
            "resolution_status": status,
            "selected_candidate_ids": [],
            "abstention_reason": "not uniquely resolvable",
            "value": None,
        }
    payload["gold"]["canonical_answers"] = {"query_0": canonical}
    return _validated_task(payload)


def _task_with_anchors(make_task) -> MemUpdateTask:
    payload = make_task().model_dump(mode="json")
    payload["events"][0]["source_anchor"] = {
        "document_id": "normalized-doc-1",
        "paragraph": 2,
        "span": [0, 34],
    }
    payload["events"][1]["source_anchor"] = {
        "document_id": "normalized-doc-1",
        "paragraph": 3,
        "span": [35, 79],
    }
    payload["gold"]["actions"][1]["expected_effect"] = {"replaces": "Dalian"}
    return _validated_task(payload)


def _rename_surface_ids_and_text(task: MemUpdateTask) -> MemUpdateTask:
    payload = task.model_dump(mode="json")
    payload["task_id"] = "paraphrase-task-另一个"
    payload["source"]["source_id"] = "surface-source-id-2"
    payload["source"]["source_uri"] = "memory://surface-source-id-2"
    payload["source"]["raw_hash"] = "9" * 64
    payload["source"]["license_or_privacy"] = "different-release-label"
    payload["source"]["provenance"] = {"surface_collection": "other"}
    payload["source"]["generator"] = {
        "generator_name": "other-generator",
        "seed": 999,
        "config_sha256": "7" * 64,
        "code_revision": "other-revision",
        "compiler_version": "99.0.0",
    }

    event_ids = {"event_0": "utterance-a", "event_1": "utterance-b"}
    action_ids = {"action_0": "write-a", "action_1": "write-b"}
    for index, event in enumerate(payload["events"]):
        old_event_id = event["event_id"]
        event["event_id"] = event_ids[old_event_id]
        event["raw_text"] = ["Alex used to be in Dalian.", "Alex now resides in Qingdao."][index]
        event["normalized_text"] = ["Dalian was the old city.", "Qingdao is current."][index]
        event["speaker"] = "paraphraser"
        event["gold_action_ids"] = [action_ids[action_id] for action_id in event["gold_action_ids"]]
        event["metadata"] = {"surface_template": f"variant-{index}"}

    for action in payload["gold"]["actions"]:
        action["action_id"] = action_ids[action["action_id"]]
        action["event_id"] = event_ids[action["event_id"]]
    payload["gold"]["action_sequence"] = [
        action_ids[action_id] for action_id in payload["gold"]["action_sequence"]
    ]
    payload["gold"]["gold_source_event_ids"] = [
        event_ids[event_id] for event_id in payload["gold"]["gold_source_event_ids"]
    ]

    query = payload["queries"][0]
    query["query_id"] = "question-surface-id"
    query["text"] = "What is Alex's current city?"
    query["metadata"] = {"surface_template": "question-variant"}
    payload["gold"]["gold_answers"] = {
        "question-surface-id": payload["gold"]["gold_answers"]["query_0"]
    }
    payload["gold"]["acceptable_answers"] = {
        "question-surface-id": payload["gold"]["acceptable_answers"]["query_0"]
    }

    payload["metadata"]["split"] = Split.DEV.value
    payload["metadata"]["split_key"] = {
        "semantic_core_id": "must-not-be-read",
        "source_group_id": "other-split-group",
        "trajectory_id": "other-trajectory-id",
        "paraphrase_group_id": "other-paraphrase-id",
        "source_document_id": "other-source-document-label",
        "version_group_id": "other-version-group",
        "split_exception_id": None,
        "split_policy_version": "99.0.0",
    }
    payload["metadata"]["resolved_profile"] = {"compiler_note": "surface-only"}
    payload["metadata"]["generation_config_hash"] = "8" * 64
    payload["metadata"]["compiler_version"] = "99.0.0"
    payload["metadata"]["tags"] = ["paraphrase", "surface"]
    payload["metadata"]["extra"] = {"generation_attempt": 17}
    return _validated_task(payload)


def _change_every_object_type(task: MemUpdateTask) -> MemUpdateTask:
    payload = task.model_dump(mode="json")
    object_lists = [
        payload["target_objects"],
        payload["gold"]["expected_present_objects"],
        payload["gold"]["expected_absent_objects"],
    ]
    object_lists.extend(action["target_object_keys"] for action in payload["gold"]["actions"])
    object_lists.extend(query["target_object_keys"] for query in payload["queries"])
    for keys in object_lists:
        for key in keys:
            key["object_type"] = "metadata-only-type"
    return _validated_task(payload)


def _task_with_secondary_target(make_task) -> MemUpdateTask:
    payload = _task_with_anchors(make_task).model_dump(mode="json")
    payload["target_objects"].append(
        {
            "object_type": "slot",
            "namespace": "default",
            "entity": "friend:blair",
            "attribute": "location",
            "subkey": None,
        }
    )
    return _validated_task(payload)


def _swap_event_order(payload: dict[str, Any]) -> None:
    payload["events"].reverse()
    for sequence_index, event in enumerate(payload["events"]):
        event["sequence_index"] = sequence_index


def _swap_action_ownership(payload: dict[str, Any]) -> None:
    payload["events"][0]["gold_action_ids"] = ["action_1"]
    payload["events"][1]["gold_action_ids"] = ["action_0"]
    payload["gold"]["actions"][0]["event_id"] = "event_1"
    payload["gold"]["actions"][1]["event_id"] = "event_0"


def _task_with_second_query(make_task, *, equivalent: bool = False) -> MemUpdateTask:
    payload = _task_with_anchors(make_task).model_dump(mode="json")
    second_query = deepcopy(payload["queries"][0])
    second_query["query_id"] = "query_1"
    second_query["text"] = "Give the same answer using different surface wording."
    if equivalent:
        second_gold_answer = payload["gold"]["gold_answers"]["query_0"]
        second_acceptable = payload["gold"]["acceptable_answers"]["query_0"]
    else:
        second_query["query_type"] = QueryType.HISTORICAL_STATE.value
        second_gold_answer = "Dalian"
        second_acceptable = ["Dalian"]
    payload["queries"].append(second_query)
    payload["gold"]["gold_answers"]["query_1"] = second_gold_answer
    payload["gold"]["acceptable_answers"]["query_1"] = second_acceptable
    return _validated_task(payload)


def _task_with_shared_action_owner(make_task) -> MemUpdateTask:
    payload = _task_with_anchors(make_task).model_dump(mode="json")
    payload["events"][0]["gold_action_ids"] = ["action_0", "action_1"]
    payload["events"][1]["gold_action_ids"] = []
    payload["gold"]["actions"][1]["event_id"] = "event_0"
    return _validated_task(payload)


def test_canonical_json_uses_sorted_compact_utf8_and_explicit_nulls() -> None:
    first = SerializationFixture(
        identifier="记录-一",
        payload={"z": 1, "a": "青岛"},
        nullable=None,
    )
    second = SerializationFixture(
        identifier="记录-一",
        payload={"a": "青岛", "z": 1},
        nullable=None,
    )

    expected = '{"identifier":"记录-一","nullable":null,"payload":{"a":"青岛","z":1}}'.encode(
        "utf-8"
    )
    assert canonical_json_bytes(first) == expected
    assert canonical_json_bytes(second) == expected
    assert not expected.endswith(b"\n")
    assert b"convenience" not in expected
    assert sha256_model(first) == sha256_model(second)
    assert sha256_model(first) == hashlib.sha256(canonical_json_bytes(first)).hexdigest()
    assert len(sha256_model(first)) == 64
    assert sha256_model(first) == sha256_model(first).lower()


@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-inf", "negative-inf"],
)
def test_canonical_json_and_sha256_reject_nested_nonfinite_values(nonfinite) -> None:
    model = LoosePayloadRecord(
        record_id="nested-nonfinite",
        payload={"outer": [{"inner": nonfinite}]},
    )

    with pytest.raises(ValueError, match=r"non-finite|Out of range"):
        canonical_json_bytes(model)
    with pytest.raises(ValueError, match=r"non-finite|Out of range"):
        sha256_model(model)


def test_write_models_does_not_write_row_with_nested_nonfinite_value(tmp_path: Path) -> None:
    valid = LoosePayloadRecord(record_id="valid", payload={"city": "青岛"})
    invalid = LoosePayloadRecord(
        record_id="invalid",
        payload={"nested": [1, {"value": float("nan")}]},
    )
    path = tmp_path / "nested-nonfinite-write.jsonl"

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(path))}.*row 2.*serialization.*non-finite",
    ) as exc_info:
        write_models(path, [valid, invalid], id_field="record_id")

    assert "$.payload.nested[1].value" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert path.read_bytes() == canonical_json_bytes(valid) + b"\n"


def test_write_models_wraps_unsupported_object_serialization_with_row_context(
    tmp_path: Path,
) -> None:
    valid = LoosePayloadRecord(record_id="valid", payload={"city": "青岛"})
    invalid = LoosePayloadRecord(
        record_id="invalid-object",
        payload={"nested": [object()]},
    )
    path = tmp_path / "unsupported-object.jsonl"

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(path))}.*row 2.*serialization",
    ) as exc_info:
        write_models(path, [valid, invalid], id_field="record_id")

    assert isinstance(exc_info.value.__cause__, PydanticSerializationError)
    assert path.read_bytes() == canonical_json_bytes(valid) + b"\n"


@pytest.mark.parametrize(
    "nonfinite_key",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-inf", "negative-inf"],
)
def test_canonical_hash_and_write_reject_nonfinite_mapping_keys(
    tmp_path: Path,
    nonfinite_key: float,
) -> None:
    model = LoosePayloadRecord(
        record_id="nonfinite-key",
        payload={"nested": {nonfinite_key: "bad"}},
    )
    path = tmp_path / "nonfinite-key.jsonl"

    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes(model)
    with pytest.raises(ValueError, match="non-finite"):
        sha256_model(model)
    with pytest.raises(ValueError, match="non-finite"):
        write_models(path, [model], id_field="record_id")
    assert path.read_bytes() == b""


@pytest.mark.parametrize(
    ("decimal_text", "placement"),
    [
        ("NaN", "value"),
        ("Infinity", "value"),
        ("-Infinity", "value"),
        ("NaN", "key"),
        ("Infinity", "key"),
        ("-Infinity", "key"),
    ],
)
def test_canonical_and_hash_reject_nonfinite_decimal_values_and_keys(
    decimal_text: str,
    placement: str,
) -> None:
    decimal = Decimal(decimal_text)
    nested = {"decimal": decimal} if placement == "value" else {decimal: "bad"}
    model = LoosePayloadRecord(record_id="decimal", payload={"nested": nested})

    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes(model)
    with pytest.raises(ValueError, match="non-finite"):
        sha256_model(model)


def test_canonical_json_preserves_finite_decimal_pydantic_output() -> None:
    model = LoosePayloadRecord(
        record_id="finite-decimal",
        payload={"value": Decimal("1.25"), "mapping": {Decimal("2.5"): "finite"}},
    )

    decoded = json.loads(canonical_json_bytes(model))

    assert decoded["payload"] == {"mapping": {"2.5": "finite"}, "value": "1.25"}
    assert sha256_model(model) == hashlib.sha256(canonical_json_bytes(model)).hexdigest()


def test_literal_nonfinite_strings_remain_valid_and_distinct() -> None:
    models = [
        LoosePayloadRecord(
            record_id=f"literal-{text}",
            payload={"nested": {"value": text, text: "literal-key"}},
        )
        for text in ["NaN", "Infinity", "-Infinity"]
    ]

    assert len({canonical_json_bytes(model) for model in models}) == 3
    assert len({sha256_model(model) for model in models}) == 3


@pytest.mark.parametrize(
    ("decimal_text", "placement"),
    [
        ("NaN", "value"),
        ("Infinity", "value"),
        ("-Infinity", "value"),
        ("NaN", "key"),
        ("Infinity", "key"),
        ("-Infinity", "key"),
    ],
)
def test_semantic_hash_rejects_nonfinite_decimal_source_anchor_content(
    make_task,
    decimal_text: str,
    placement: str,
) -> None:
    decimal_task = _task_with_anchors(make_task)
    literal_task = _task_with_anchors(make_task)
    if placement == "value":
        decimal_content = {"nested": [Decimal(decimal_text)]}
        literal_content = {"nested": [decimal_text]}
    else:
        decimal_content = {"nested": {Decimal(decimal_text): "anchor"}}
        literal_content = {"nested": {decimal_text: "anchor"}}
    decimal_task.events[0].source_anchor["decimal_content"] = decimal_content
    literal_task.events[0].source_anchor["decimal_content"] = literal_content

    literal_hash = semantic_task_hash(literal_task)

    assert len(literal_hash) == 64
    with pytest.raises(ValueError, match="non-finite"):
        semantic_task_hash(decimal_task)


@pytest.mark.parametrize("container_type", [set, frozenset], ids=["set", "frozenset"])
def test_canonical_hash_and_write_reject_nested_unordered_containers(
    tmp_path: Path,
    container_type,
) -> None:
    model = LoosePayloadRecord(
        record_id="unordered",
        payload={"nested": [container_type({"alpha", "beta"})]},
    )
    path = tmp_path / "unordered.jsonl"

    with pytest.raises(ValueError, match=r"unordered.*\$\.payload\.nested\[0\]"):
        canonical_json_bytes(model)
    with pytest.raises(ValueError, match="unordered"):
        sha256_model(model)
    with pytest.raises(ValueError, match=r"row 1.*serialization.*unordered"):
        write_models(path, [model], id_field="record_id")
    assert path.read_bytes() == b""


@pytest.mark.parametrize("container_type", [set, frozenset], ids=["set", "frozenset"])
def test_unordered_container_rejection_precedes_nonfinite_element_validation(
    container_type,
) -> None:
    model = LoosePayloadRecord(
        record_id="unordered-nonfinite",
        payload={"nested": container_type({Decimal("NaN")})},
    )

    with pytest.raises(ValueError, match="unordered") as exc_info:
        canonical_json_bytes(model)

    assert "non-finite" not in str(exc_info.value)


def test_semantic_hash_rejects_unordered_source_anchor_content(make_task) -> None:
    task = _task_with_anchors(make_task)
    task.events[0].source_anchor["unordered"] = {"alpha", "beta"}

    with pytest.raises(ValueError, match=r"unordered.*source_anchor"):
        semantic_task_hash(task)


def test_unordered_inputs_fail_consistently_across_python_hash_seeds() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = """
from typing import Any
from pydantic import BaseModel
from mub.vnext.io import canonical_json_bytes
class Record(BaseModel):
    payload: Any
try:
    canonical_json_bytes(Record(payload={"unordered": {"alpha", "beta", "gamma"}}))
except ValueError as exc:
    print(f"ERROR:{exc}")
else:
    print("SERIALIZED")
"""
    outputs = []
    for seed in ["1", "2", "123"]:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(project_root)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1
    assert outputs[0].startswith("ERROR:canonical JSON cannot contain unordered")


def test_canonical_json_serializes_task5_immutable_values_without_warnings(
    make_score_record,
    make_run_manifest,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score_bytes = canonical_json_bytes(make_score_record())
        manifest_bytes = canonical_json_bytes(make_run_manifest())

    assert json.loads(score_bytes)["answer_scores"]["exact_match"] == 1.0
    assert json.loads(manifest_bytes)["seed_information"] == {"seed": 0}
    assert caught == []


def test_semantic_hash_ignores_surface_ids_wording_and_generation_metadata(make_task) -> None:
    task = _task_with_anchors(make_task)
    paraphrase = _rename_surface_ids_and_text(task)

    assert semantic_task_hash(task) == semantic_task_hash(paraphrase)


def test_unresolved_semantic_hash_ignores_surface_wording_ids_prose_and_object_type(
    make_task,
) -> None:
    task = _unresolved_task(make_task)
    payload = task.model_dump(mode="json")
    query = payload["queries"][0]
    query["query_id"] = "renamed_query"
    query["text"] = "Which differently worded Alex reference?"
    query["metadata"] = {"surface_template": "other"}
    candidate_ids = ["renamed_friend", "renamed_colleague"]
    for candidate, candidate_id in zip(query["reference_candidates"], candidate_ids):
        candidate["candidate_id"] = candidate_id
        candidate["evidence"] = "Entirely different explanatory prose."
        candidate["object_key"]["object_type"] = "metadata_only_type"
    reference = query["surface_references"][0]
    reference.update(
        reference_id="renamed_reference",
        surface_text="that person",
        normalized_text="that person",
        candidate_ids=candidate_ids,
    )
    payload["gold"]["canonical_answers"] = {
        "renamed_query": {
            **payload["gold"]["canonical_answers"]["query_0"],
            "abstention_reason": "Different prose reason.",
        }
    }

    assert semantic_task_hash(task) == semantic_task_hash(_validated_task(payload))


def test_unresolved_semantic_hash_preserves_candidate_and_linkage_order(make_task) -> None:
    task = _unresolved_task(make_task)
    candidate_order = task.model_dump(mode="json")
    candidate_order["queries"][0]["reference_candidates"].reverse()
    linkage_order = task.model_dump(mode="json")
    linkage_order["queries"][0]["surface_references"][0]["candidate_ids"].reverse()

    assert semantic_task_hash(task) != semantic_task_hash(_validated_task(candidate_order))
    assert semantic_task_hash(task) != semantic_task_hash(_validated_task(linkage_order))


def test_unresolved_semantic_hash_rejects_dangling_surface_candidate_link(make_task) -> None:
    task = _unresolved_task(make_task)
    valid_hash = semantic_task_hash(task)
    task.queries[0].surface_references[0].candidate_ids.append("dangling_surface")

    assert len(valid_hash) == 64
    with pytest.raises(
        ValueError,
        match="surface reference links unknown candidate ID 'dangling_surface'",
    ):
        semantic_task_hash(task)


def test_unresolved_semantic_hash_rejects_dangling_selected_candidate_link(make_task) -> None:
    task = _unresolved_task(make_task, "unique")
    valid_hash = semantic_task_hash(task)
    task.gold.canonical_answers["query_0"].selected_candidate_ids.append(
        "dangling_selected"
    )

    assert len(valid_hash) == 64
    with pytest.raises(
        ValueError,
        match="canonical answer selects unknown candidate ID 'dangling_selected'",
    ):
        semantic_task_hash(task)


def test_unresolved_semantic_hash_includes_resolution_graph_and_canonical_outcome(
    make_task,
) -> None:
    ambiguous = _unresolved_task(make_task, "ambiguous")
    no_match = _unresolved_task(make_task, "no_match")
    unique = _unresolved_task(make_task, "unique")
    changed_identity = ambiguous.model_dump(mode="json")
    candidate_key = changed_identity["queries"][0]["reference_candidates"][1][
        "object_key"
    ]
    candidate_key["attribute"] = "timezone"
    changed_identity["target_objects"][1]["attribute"] = "timezone"
    changed_evidence = ambiguous.model_dump(mode="json")
    changed_evidence["queries"][0]["surface_references"][0][
        "condition_kind"
    ] = "namespace_collision"
    changed_value = unique.model_dump(mode="json")
    changed_value["gold"]["canonical_answers"]["query_0"]["value"] = "Weihai"

    hashes = {
        semantic_task_hash(ambiguous),
        semantic_task_hash(no_match),
        semantic_task_hash(unique),
        semantic_task_hash(_validated_task(changed_identity)),
        semantic_task_hash(_validated_task(changed_evidence)),
        semantic_task_hash(_validated_task(changed_value)),
    }
    assert len(hashes) == 6


def test_semantic_hash_ignores_difficulty_label_only_change(make_task) -> None:
    task = _task_with_anchors(make_task)
    payload = task.model_dump(mode="json")
    payload["difficulty"] = Difficulty.HARD.value
    relabeled = _validated_task(payload)

    assert relabeled.difficulty is Difficulty.HARD
    assert semantic_task_hash(task) == semantic_task_hash(relabeled)


def test_semantic_hash_uses_object_identity_but_not_object_type_metadata(make_task) -> None:
    task = _task_with_anchors(make_task)
    metadata_variant = _change_every_object_type(task)
    target_variant_payload = task.model_dump(mode="json")
    for key in target_variant_payload["target_objects"]:
        key["entity"] = "friend:blair"
    for action in target_variant_payload["gold"]["actions"]:
        for key in action["target_object_keys"]:
            key["entity"] = "friend:blair"
    for query in target_variant_payload["queries"]:
        for key in query["target_object_keys"]:
            key["entity"] = "friend:blair"
    for key in target_variant_payload["gold"]["expected_present_objects"]:
        key["entity"] = "friend:blair"
    old_state_id = task.target_objects[0].canonical_id
    new_state_id = "default|friend:blair|location|"
    target_variant_payload["gold"]["final_state"] = {
        new_state_id: target_variant_payload["gold"]["final_state"][old_state_id]
    }
    target_variant_payload["gold"]["version_history"] = {
        new_state_id: target_variant_payload["gold"]["version_history"][old_state_id]
    }
    target_variant = _validated_task(target_variant_payload)

    assert semantic_task_hash(task) == semantic_task_hash(metadata_variant)
    assert semantic_task_hash(task) != semantic_task_hash(target_variant)


def test_semantic_hash_ignores_query_list_order_but_keeps_answers_coupled(make_task) -> None:
    task = _task_with_second_query(make_task)
    payload = task.model_dump(mode="json")
    payload["queries"].reverse()
    reordered = _validated_task(payload)

    assert semantic_task_hash(task) == semantic_task_hash(reordered)


def test_semantic_hash_ignores_owned_action_id_order(make_task) -> None:
    task = _task_with_shared_action_owner(make_task)
    payload = task.model_dump(mode="json")
    payload["events"][0]["gold_action_ids"].reverse()
    reordered = _validated_task(payload)

    assert semantic_task_hash(task) == semantic_task_hash(reordered)


def test_semantic_hash_preserves_duplicate_equivalent_query_records(make_task) -> None:
    one_query = _task_with_anchors(make_task)
    two_equivalent_queries = _task_with_second_query(make_task, equivalent=True)

    assert semantic_task_hash(one_query) != semantic_task_hash(two_equivalent_queries)


@pytest.mark.parametrize(
    ("category", "mutate"),
    [
        ("task-family", lambda payload: payload.update(task_family="multi_object_interleaving")),
        ("event-role", lambda payload: payload["events"][0].update(role="neutral")),
        ("event-order", _swap_event_order),
        ("action-ownership", _swap_action_ownership),
        (
            "action-order",
            lambda payload: payload["gold"].update(
                action_sequence=list(reversed(payload["gold"]["action_sequence"]))
            ),
        ),
        (
            "action-operation",
            lambda payload: payload["gold"]["actions"][1].update(
                operation=Operation.ADD.value
            ),
        ),
        (
            "action-scope",
            lambda payload: payload["gold"]["actions"][1].update(
                scope=ActionScope.ENTITY.value
            ),
        ),
        ("action-value", lambda payload: payload["gold"]["actions"][1].update(value="Weihai")),
        (
            "action-effective-at",
            lambda payload: payload["gold"]["actions"][1].update(
                effective_at="2026-07-20T12:00:00Z"
            ),
        ),
        (
            "action-expected-effect",
            lambda payload: payload["gold"]["actions"][1].update(
                expected_effect={"replaces": "A different prior value"}
            ),
        ),
        (
            "query-type",
            lambda payload: payload["queries"][0].update(
                query_type=QueryType.HISTORICAL_STATE.value
            ),
        ),
        (
            "query-answer-schema",
            lambda payload: payload["queries"][0].update(
                answer_schema=AnswerSchema.OBJECT.value
            ),
        ),
        (
            "query-evaluation-mode",
            lambda payload: payload["queries"][0].update(
                evaluation_mode=EvaluationMode.STATE_DIRECT.value
            ),
        ),
        (
            "gold-final-state",
            lambda payload: payload["gold"]["final_state"].update(
                {next(iter(payload["gold"]["final_state"])): "Weihai"}
            ),
        ),
        (
            "gold-version-history",
            lambda payload: payload["gold"]["version_history"].update(
                {
                    next(iter(payload["gold"]["version_history"])): [
                        "Dalian",
                        "Qingdao",
                        "Weihai",
                    ]
                }
            ),
        ),
        (
            "gold-answer",
            lambda payload: payload["gold"]["gold_answers"].update(
                {"query_0": "Weihai"}
            ),
        ),
        (
            "acceptable-answers",
            lambda payload: payload["gold"]["acceptable_answers"].update(
                {"query_0": ["Qingdao", "Tsingtao"]}
            ),
        ),
        (
            "gold-source-support",
            lambda payload: payload["gold"].update(
                gold_source_event_ids=["event_0", "event_1"]
            ),
        ),
        ("normalized-source", lambda payload: payload["source"].update(normalized_hash="7" * 64)),
        (
            "source-anchor",
            lambda payload: payload["events"][0]["source_anchor"].update(paragraph=99),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_semantic_hash_changes_for_semantic_mutations(make_task, category, mutate) -> None:
    task = _task_with_anchors(make_task)
    payload = deepcopy(task.model_dump(mode="json"))
    mutate(payload)
    variant = _validated_task(payload)

    assert variant != task, category
    assert semantic_task_hash(task) != semantic_task_hash(variant), category


def test_semantic_hash_changes_for_action_target_identity(make_task) -> None:
    task = _task_with_secondary_target(make_task)
    payload = task.model_dump(mode="json")
    payload["gold"]["actions"][1]["target_object_keys"] = [payload["target_objects"][1]]

    assert semantic_task_hash(task) != semantic_task_hash(_validated_task(payload))


def test_semantic_hash_changes_for_query_target_identity(make_task) -> None:
    task = _task_with_secondary_target(make_task)
    payload = task.model_dump(mode="json")
    payload["queries"][0]["target_object_keys"] = [payload["target_objects"][1]]

    assert semantic_task_hash(task) != semantic_task_hash(_validated_task(payload))


@pytest.mark.parametrize(
    "gold_field",
    ["expected_present_objects", "expected_absent_objects"],
)
def test_semantic_hash_changes_for_expected_object_sets(make_task, gold_field) -> None:
    task = _task_with_secondary_target(make_task)
    payload = task.model_dump(mode="json")
    payload["gold"][gold_field].append(payload["target_objects"][1])

    assert semantic_task_hash(task) != semantic_task_hash(_validated_task(payload))


def test_write_models_emits_binary_lf_rows_and_consumes_generator_once(
    tmp_path: Path,
    make_task,
) -> None:
    first = make_task()
    second_payload = first.model_dump(mode="json")
    second_payload["task_id"] = "task_二"
    second_payload["events"][1]["raw_text"] = "朋友搬到了青岛。"
    second = _validated_task(second_payload)
    yielded: list[str] = []

    def records():
        for model in (first, second):
            yielded.append(model.task_id)
            yield model

    output = tmp_path / "tasks.jsonl"
    write_models(output, records(), id_field="task_id")

    assert yielded == [first.task_id, second.task_id]
    assert output.read_bytes() == (
        canonical_json_bytes(first) + b"\n" + canonical_json_bytes(second) + b"\n"
    )
    assert "青岛".encode("utf-8") in output.read_bytes()
    assert b"\r\n" not in output.read_bytes()


def test_write_models_reports_late_duplicate_and_preserves_flushed_prefix(
    tmp_path: Path,
    make_task,
) -> None:
    first = make_task()
    second_payload = first.model_dump(mode="json")
    second_payload["task_id"] = "task_unique_2"
    second = _validated_task(second_payload)
    output = tmp_path / "duplicate-write.jsonl"

    with pytest.raises(ValueError, match=r"row 3.*duplicate.*row 1"):
        write_models(output, (model for model in (first, second, first)), id_field="task_id")

    assert output.read_bytes() == canonical_json_bytes(first) + b"\n" + canonical_json_bytes(second) + b"\n"


def test_write_models_rejects_missing_id_field_with_row_context(tmp_path: Path) -> None:
    output = tmp_path / "missing-write-id.jsonl"

    with pytest.raises(ValueError, match=r"row 1.*missing ID field.*record_id"):
        write_models(output, [MissingIdRecord()], id_field="record_id")


@pytest.mark.parametrize(
    "bad_id",
    ["", "   ", None, ["not", "scalar"], {"not": "scalar"}, float("nan"), float("inf"), float("-inf")],
    ids=["empty", "blank", "none", "list", "mapping", "nan", "positive-inf", "negative-inf"],
)
def test_write_models_rejects_invalid_ids_with_row_context(tmp_path: Path, bad_id) -> None:
    output = tmp_path / "bad-write-id.jsonl"

    with pytest.raises(ValueError, match=r"row 1.*record_id"):
        write_models(output, [LooseIdRecord(record_id=bad_id)], id_field="record_id")


@pytest.mark.parametrize(
    "valid_id",
    ["record-1", 7, 1.5, True, False],
    ids=["string", "integer", "finite-float", "true", "false"],
)
def test_write_models_accepts_nonblank_scalar_ids(tmp_path: Path, valid_id) -> None:
    output = tmp_path / "valid-write-id.jsonl"
    record = LooseIdRecord(record_id=valid_id)

    write_models(output, [record], id_field="record_id")

    assert output.read_bytes() == canonical_json_bytes(record) + b"\n"


def test_write_models_duplicate_identity_is_type_sensitive(tmp_path: Path) -> None:
    output = tmp_path / "typed-write-ids.jsonl"
    records = [
        LooseIdRecord(record_id=True),
        LooseIdRecord(record_id=1),
        LooseIdRecord(record_id=1.0),
        LooseIdRecord(record_id="1"),
    ]

    write_models(output, records, id_field="record_id")

    assert output.read_bytes().count(b"\n") == 4


def test_read_models_rejects_missing_id_field_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "missing-read-id.jsonl"
    path.write_text('{"payload":"fixture"}\n', encoding="utf-8", newline="")

    with pytest.raises(
        ValueError,
        match=r"missing-read-id\.jsonl.*line 1.*missing ID field.*record_id",
    ):
        list(read_models(path, MissingIdRecord, id_field="record_id"))


@pytest.mark.parametrize(
    "bad_id",
    ["", "   ", None, ["not", "scalar"], {"not": "scalar"}, float("nan"), float("inf"), float("-inf")],
    ids=["empty", "blank", "none", "list", "mapping", "nan", "positive-inf", "negative-inf"],
)
def test_read_models_rejects_invalid_ids_with_line_context(tmp_path: Path, bad_id) -> None:
    path = tmp_path / "bad-read-id.jsonl"
    path.write_text(
        json.dumps({"record_id": bad_id, "payload": "fixture"}) + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(
        ValueError,
        match=r"bad-read-id\.jsonl.*line 1.*(?:record_id|non-finite)",
    ):
        list(read_models(path, LooseIdRecord, id_field="record_id"))


@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity"],
    ids=["nan", "positive-inf", "negative-inf"],
)
def test_read_models_rejects_nested_nonfinite_json_with_line_context(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "nested-nonfinite-read.jsonl"
    path.write_bytes(
        (
            '{"record_id":"bad","payload":{"outer":[{"inner":'
            + constant
            + "}]}}\n"
        ).encode("utf-8")
    )

    with pytest.raises(
        ValueError,
        match=r"nested-nonfinite-read\.jsonl.*line 1.*non-finite",
    ):
        list(read_models(path, LoosePayloadRecord, id_field="record_id"))


@pytest.mark.parametrize(
    "overflow_number",
    ["1e9999", "-1e9999"],
    ids=["positive", "negative"],
)
def test_read_models_rejects_nested_exponent_overflow_with_line_context(
    tmp_path: Path,
    overflow_number: str,
) -> None:
    path = tmp_path / "exponent-overflow.jsonl"
    path.write_bytes(
        (
            '{"record_id":"overflow","payload":{"nested":['
            + overflow_number
            + "]}}\n"
        ).encode("utf-8")
    )

    with pytest.raises(
        ValueError,
        match=r"exponent-overflow\.jsonl.*line 1.*non-finite",
    ):
        list(read_models(path, LoosePayloadRecord, id_field="record_id"))


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_read_models_rejects_non_utf8_json_encodings_with_line_context(
    tmp_path: Path,
    encoding: str,
) -> None:
    path = tmp_path / f"encoded-{encoding}.jsonl"
    text = '{"record_id":"encoded","payload":"fixture"}'
    path.write_bytes(text.encode(encoding))

    with pytest.raises(
        ValueError,
        match=rf"encoded-{encoding}\.jsonl.*line 1.*UTF-8",
    ):
        list(read_models(path, LoosePayloadRecord, id_field="record_id"))


def test_read_models_rejects_invalid_utf8_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.jsonl"
    path.write_bytes(b'{"record_id":"bad","payload":"\xff"}\n')

    with pytest.raises(
        ValueError,
        match=r"invalid-utf8\.jsonl.*line 1.*UTF-8",
    ):
        list(read_models(path, LoosePayloadRecord, id_field="record_id"))


@pytest.mark.parametrize(
    "valid_id",
    ["record-1", 7, 1.5, True, False],
    ids=["string", "integer", "finite-float", "true", "false"],
)
def test_read_models_accepts_nonblank_scalar_ids(tmp_path: Path, valid_id) -> None:
    path = tmp_path / "valid-read-id.jsonl"
    path.write_text(
        json.dumps({"record_id": valid_id, "payload": "fixture"}) + "\n",
        encoding="utf-8",
        newline="",
    )

    [record] = list(read_models(path, LooseIdRecord, id_field="record_id"))

    assert record.record_id == valid_id
    assert type(record.record_id) is type(valid_id)


def test_read_models_duplicate_identity_is_type_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "typed-read-ids.jsonl"
    values = [True, 1, 1.0, "1"]
    rows = [json.dumps({"record_id": value, "payload": "fixture"}) for value in values]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")

    records = list(read_models(path, LooseIdRecord, id_field="record_id"))

    assert [record.record_id for record in records] == values
    assert [type(record.record_id) for record in records] == [bool, int, float, str]


def test_read_models_is_lazy_and_preserves_unicode(tmp_path: Path, make_task) -> None:
    missing = tmp_path / "not-created.jsonl"
    lazy_records = read_models(missing, MemUpdateTask, id_field="task_id")
    assert iter(lazy_records) is lazy_records
    with pytest.raises(FileNotFoundError):
        next(lazy_records)

    task_payload = make_task().model_dump(mode="json")
    task_payload["task_id"] = "任务-青岛"
    task = _validated_task(task_payload)
    path = tmp_path / "unicode.jsonl"
    path.write_bytes(canonical_json_bytes(task) + b"\n")

    assert list(read_models(path, MemUpdateTask, id_field="task_id")) == [task]


def test_read_models_rejects_duplicate_ids_with_both_line_numbers(
    tmp_path: Path,
    make_task,
) -> None:
    task = make_task()
    path = tmp_path / "duplicate-read.jsonl"
    path.write_bytes(canonical_json_bytes(task) + b"\n" + canonical_json_bytes(task) + b"\n")

    with pytest.raises(ValueError, match=r"duplicate-read\.jsonl.*line 2.*duplicate.*line 1"):
        list(read_models(path, MemUpdateTask, id_field="task_id"))


def test_read_models_rejects_blank_and_malformed_rows_with_line_context(
    tmp_path: Path,
    make_task,
) -> None:
    task = make_task()
    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_bytes(canonical_json_bytes(task) + b"\n\n")
    with pytest.raises(ValueError, match=r"blank\.jsonl.*line 2.*blank"):
        list(read_models(blank_path, MemUpdateTask, id_field="task_id"))

    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_bytes(canonical_json_bytes(task) + b"\n{" + b"\n")
    with pytest.raises(ValueError, match=r"malformed\.jsonl.*line 2.*JSON"):
        list(read_models(malformed_path, MemUpdateTask, id_field="task_id"))


def test_read_models_wraps_validation_errors_with_line_context(
    tmp_path: Path,
    make_task,
) -> None:
    payload = make_task().model_dump(mode="json")
    del payload["gold"]
    path = tmp_path / "invalid-model.jsonl"
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")

    with pytest.raises(ValueError, match=r"invalid-model\.jsonl.*line 1.*validation"):
        list(read_models(path, MemUpdateTask, id_field="task_id"))
