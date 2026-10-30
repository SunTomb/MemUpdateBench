from __future__ import annotations

import hashlib
import json

import pytest

from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3
from scripts.vnext_prepare_historical_mechanism import (
    CONFIG_PATH,
    build_cases,
    publish_packet,
    read_packet,
)


def test_builds_exactly_three_cases_with_distinct_roles():
    cases = build_cases(code_revision="fixture-revision")
    assert [case.role for case in cases] == [
        "target_only_control", "cross_entity_same_relation", "historical_serving"
    ]
    assert all(case.model_calls == 0 for case in cases)
    assert all(not replay_task_v3(case.task).issues for case in cases)
    assert all(case.task.source.generator.code_revision == "fixture-revision" for case in cases)


def test_cases_replay_and_queries_resolve_independently():
    cases = build_cases(code_revision="fixture-revision")
    historical = cases[2]
    assert len(historical.task.queries) == 2
    assert {query.query_type.value for query in historical.task.queries} == {"current", "point_in_time"}
    replay = replay_task_v3(historical.task)
    for query, gold in zip(historical.task.queries, historical.task.gold_evidence):
        resolved = resolve_query_v3(query, replay, historical.task.events)
        assert not resolved.issues
        assert resolved.answer == gold.answer


def test_public_event_and_query_payloads_are_gold_free():
    cases = build_cases(code_revision="fixture-revision")
    for case in cases:
        for event in case.events:
            payload = json.dumps(event.public_payload, sort_keys=True).lower()
            assert "gold" not in payload
            assert "selector" not in payload
            assert event.public_payload == visible_event_input(
                event.source_event, runtime_namespace=case.runtime_namespace
            ).model_dump(mode="json")
        for query in case.queries:
            payload = json.dumps(query.public_payload, sort_keys=True).lower()
            assert "gold" not in payload
            assert "selector" not in payload
            assert query.public_payload == visible_query_input(
                query.source_query, k=8, runtime_namespace=case.runtime_namespace
            ).model_dump(mode="json")


def test_packet_rejects_rebinding_or_replacement(tmp_path):
    receipt = publish_packet(tmp_path / "packet", code_revision="fixture-revision")
    packet = read_packet(tmp_path / "packet", receipt["index_sha256"])
    assert packet["status"] == "PREPARED_NOT_EXECUTED"
    assert packet["scientific_release_allowed"] is False
    assert packet["case_count"] == 3
    assert packet["model_calls"] == 0
    with pytest.raises(FileExistsError):
        publish_packet(tmp_path / "packet", code_revision="fixture-revision")


def test_packet_does_not_include_gold_actions_in_public_episode_payload():
    case = build_cases(code_revision="fixture-revision")[1]
    assert all("gold" not in json.dumps(event.public_payload).lower() for event in case.events)
    assert all("selector" not in json.dumps(event.public_payload).lower() for event in case.events)


def test_packet_is_hash_indexed_and_no_execution(tmp_path):
    output = tmp_path / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    index = json.loads((output / "index.json").read_bytes())
    assert receipt["index_sha256"]
    assert set(index["artifacts"]) == {"config.json", "cases.jsonl", "manifest.json"}
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["source_hash_policy"] == "python_lf_normalized"
    assert manifest["source_hashes"]["scripts/vnext_prepare_historical_mechanism.py"]
    assert manifest["execution_counters"] == {
        "model_calls": 0,
        "provider_calls": 0,
        "database_calls": 0,
        "execution_attempts": 0,
    }
    assert manifest["case_count"] == 3
    assert manifest["task_hashes"]


def test_packet_rejects_protected_or_unrelated_result_roots():
    with pytest.raises(ValueError, match="protected"):
        publish_packet(CONFIG_PATH.parents[2] / "data/vnext/core/v3/forbidden", code_revision="fixture-revision")
    with pytest.raises(ValueError, match="protected"):
        publish_packet(CONFIG_PATH.parents[2] / "results/vnext/pilot/forbidden", code_revision="fixture-revision")
    with pytest.raises(ValueError, match="unrelated"):
        publish_packet(CONFIG_PATH.parents[2] / "results/unrelated/forbidden", code_revision="fixture-revision")


def test_cases_jsonl_records_preparation_status_per_case(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    records = [json.loads(line) for line in (output / "cases.jsonl").read_bytes().splitlines()]
    assert [record["status"] for record in records] == [
        "PREPARED_NOT_EXECUTED",
        "PREPARED_NOT_EXECUTED",
        "PREPARED_NOT_EXECUTED",
    ]


def test_packet_rejects_tampered_top_level_execution_counter(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["model_calls"] = 1
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    manifest_raw = manifest_path.read_bytes()
    index["artifacts"]["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "bytes": len(manifest_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    with pytest.raises(ValueError, match="execution counters"):
        read_packet(output, index_sha256)


def _rewrite_manifest_and_index(output, mutate):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    manifest_raw = manifest_path.read_bytes()
    index["artifacts"]["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "bytes": len(manifest_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    return hashlib.sha256(index_raw).hexdigest()


def test_packet_rejects_tampered_manifest_task_hashes(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output,
        lambda manifest: manifest["task_hashes"].update({"target_only_control": "0" * 64}),
    )
    with pytest.raises(ValueError, match="manifest"):
        read_packet(output, index_sha256)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_count", 4),
        ("roles", ["historical_serving", "cross_entity_same_relation", "target_only_control"]),
    ],
)
def test_packet_rejects_tampered_manifest_case_shape(tmp_path, field, value):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output, lambda manifest: manifest.update({field: value})
    )
    with pytest.raises(ValueError, match="manifest"):
        read_packet(output, index_sha256)


def test_packet_rejects_tampered_manifest_semantic_hashes(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output,
        lambda manifest: manifest["semantic_hashes"].update({"historical_serving": "f" * 64}),
    )
    with pytest.raises(ValueError, match="manifest"):
        read_packet(output, index_sha256)
