from __future__ import annotations

import hashlib
import json

import pytest

from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3
from scripts import vnext_prepare_historical_mechanism as prep
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


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_cases_replay_and_queries_resolve_independently(case_index):
    case = build_cases(code_revision="fixture-revision")[case_index]
    if case.role == "historical_serving":
        assert len(case.task.queries) == 2
        assert {query.query_type.value for query in case.task.queries} == {"current", "point_in_time"}
    else:
        assert len(case.task.queries) == 1
    replay = replay_task_v3(case.task)
    for query, gold in zip(case.task.queries, case.task.gold_evidence):
        resolved = resolve_query_v3(query, replay, case.task.events)
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


def test_packet_rejects_tampered_config_after_index_rehash(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    config_path = output / "config.json"
    config = json.loads(config_path.read_bytes())
    config["retrieval_k"] = 9
    config_path.write_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    config_raw = config_path.read_bytes()
    index["artifacts"]["config.json"] = {
        "sha256": hashlib.sha256(config_raw).hexdigest(),
        "bytes": len(config_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    with pytest.raises(ValueError, match="config"):
        read_packet(output, hashlib.sha256(index_raw).hexdigest())


def test_build_cases_loads_one_validated_config_snapshot(monkeypatch):
    calls = []
    original = prep._load_config

    def counted_load_config(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(prep, "_load_config", counted_load_config)
    build_cases(code_revision="fixture-revision")
    assert len(calls) == 1


def test_failed_new_publication_cleans_empty_dir_for_retry(monkeypatch, tmp_path):
    output = tmp_path / "packet"
    original_publish = prep.publish_files_atomically

    def fail_publication(*args, **kwargs):
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(prep, "publish_files_atomically", fail_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        publish_packet(output, code_revision="fixture-revision")
    assert not output.exists()

    monkeypatch.setattr(prep, "publish_files_atomically", original_publish)
    receipt = publish_packet(output, code_revision="fixture-revision")
    assert read_packet(output, receipt["index_sha256"])["status"] == "PREPARED_NOT_EXECUTED"


@pytest.mark.parametrize(
    "path",
    [
        "data/vnext/staging/../core/forbidden",
        "results/vnext/staging/../pilot/forbidden",
    ],
)
def test_packet_rejects_traversal_into_protected_roots(path):
    with pytest.raises(ValueError, match="protected"):
        publish_packet(CONFIG_PATH.parents[2] / path, code_revision="fixture-revision")


def test_failed_competing_directory_is_not_deleted(monkeypatch, tmp_path):
    output = tmp_path / "packet"
    real_mkdir = prep.Path.mkdir

    def competing_mkdir(path, *args, **kwargs):
        if path == output:
            real_mkdir(path, parents=True, exist_ok=True)
            raise FileExistsError(output)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(prep.Path, "mkdir", competing_mkdir)
    with pytest.raises(FileExistsError):
        publish_packet(output, code_revision="fixture-revision")
    assert output.is_dir()


def test_namespace_template_from_config_controls_owned_runtime_namespaces(monkeypatch, tmp_path):
    config = json.loads(CONFIG_PATH.read_bytes())
    config["namespace_template"] = "owned-mechanism-{role}"
    config_path = tmp_path / "config.json"
    config_path.write_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    monkeypatch.setattr(prep, "CONFIG_PATH", config_path)
    cases = build_cases(code_revision="fixture-revision")
    assert [case.runtime_namespace for case in cases] == [
        "owned-mechanism-target_only_control",
        "owned-mechanism-cross_entity_same_relation",
        "owned-mechanism-historical_serving",
    ]


def _rewrite_cases_and_index(output, mutate):
    cases_path = output / "cases.jsonl"
    records = [json.loads(line) for line in cases_path.read_bytes().splitlines()]
    mutate(records)
    cases_path.write_bytes(
        b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
            for record in records
        )
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    cases_raw = cases_path.read_bytes()
    index["artifacts"]["cases.jsonl"] = {
        "sha256": hashlib.sha256(cases_raw).hexdigest(),
        "bytes": len(cases_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    return hashlib.sha256(index_raw).hexdigest()


@pytest.mark.parametrize("field", ["runtime_namespace", "source_code_revision", "namespace_template"])
def test_packet_rejects_tampered_case_binding_fields(tmp_path, field):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_cases_and_index(
        output,
        lambda records: records[0].update({field: "tampered-value"}),
    )
    with pytest.raises(ValueError, match="binding"):
        read_packet(output, index_sha256)


def test_config_snapshot_retains_exact_zero_execution_counters():
    snapshot = prep._load_config()
    assert snapshot.execution_counters == {
        "model_calls": 0,
        "provider_calls": 0,
        "database_calls": 0,
        "execution_attempts": 0,
    }


def test_nonzero_config_execution_counters_are_rejected(monkeypatch, tmp_path):
    config = json.loads(CONFIG_PATH.read_bytes())
    config["execution_counters"]["model_calls"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    monkeypatch.setattr(prep, "CONFIG_PATH", config_path)
    with pytest.raises(ValueError, match="execution counters"):
        build_cases(code_revision="fixture-revision")


def test_packet_rejects_tampered_config_execution_counters(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output,
        lambda manifest: manifest["execution_counters"].update({"provider_calls": 1}),
    )
    with pytest.raises(ValueError, match="execution counters"):
        read_packet(output, index_sha256)


def test_packet_rejects_tampered_packet_config_execution_counters(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    config_path = output / "config.json"
    config = json.loads(config_path.read_bytes())
    config["execution_counters"]["provider_calls"] = 1
    config_path.write_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    config_raw = config_path.read_bytes()
    index["artifacts"]["config.json"] = {
        "sha256": hashlib.sha256(config_raw).hexdigest(),
        "bytes": len(config_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    with pytest.raises(ValueError, match="execution counters"):
        read_packet(output, hashlib.sha256(index_raw).hexdigest())


@pytest.mark.parametrize("artifact", ["index", "manifest", "config"])
def test_packet_rejects_wrong_artifact_schema_after_rehash(tmp_path, artifact):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    if artifact == "index":
        index_path = output / "index.json"
        index = json.loads(index_path.read_bytes())
        index["schema"] = "tampered.index.schema"
        index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        index_path.write_bytes(index_raw)
    elif artifact == "manifest":
        index_sha256 = _rewrite_manifest_and_index(
            output, lambda manifest: manifest.update({"schema": "tampered.manifest.schema"})
        )
        with pytest.raises(ValueError, match="schema"):
            read_packet(output, index_sha256)
        return
    else:
        config_path = output / "config.json"
        config = json.loads(config_path.read_bytes())
        config["schema"] = "tampered.config.schema"
        config_path.write_bytes(
            json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        )
        index_path = output / "index.json"
        index = json.loads(index_path.read_bytes())
        config_raw = config_path.read_bytes()
        index["artifacts"]["config.json"] = {
            "sha256": hashlib.sha256(config_raw).hexdigest(),
            "bytes": len(config_raw),
        }
        index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        index_path.write_bytes(index_raw)
    with pytest.raises(ValueError, match="schema"):
        read_packet(output, hashlib.sha256(index_raw).hexdigest())





def _rewrite_index_and_return_hash(output, mutate):
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    mutate(index)
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    return hashlib.sha256(index_raw).hexdigest()


@pytest.mark.parametrize(
    ("artifact", "operation"),
    [
        ("config", "extra"),
        ("config", "missing"),
        ("index", "extra"),
        ("index", "missing"),
        ("manifest", "extra"),
        ("manifest", "missing"),
        ("cases", "extra"),
        ("cases", "missing"),
    ],
)
def test_packet_rejects_extra_or_missing_declared_keys(tmp_path, artifact, operation):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    if artifact == "config":
        path = output / "config.json"
        payload = json.loads(path.read_bytes())
        if operation == "extra":
            payload["unexpected"] = True
        else:
            payload.pop("retrieval_k")
        path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        index_sha256 = _rewrite_index_and_return_hash(
            output,
            lambda index: index["artifacts"]["config.json"].update({
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": len(path.read_bytes()),
            }),
        )
    elif artifact == "index":
        index_sha256 = _rewrite_index_and_return_hash(
            output,
            lambda index: index.update(
                {"unexpected": True} if operation == "extra" else {"schema": None}
            )
            if operation == "extra"
            else index.pop("schema"),
        )
    elif artifact == "manifest":
        index_sha256 = _rewrite_manifest_and_index(
            output,
            lambda manifest: manifest.update({"unexpected": True})
            if operation == "extra"
            else manifest.pop("retrieval_k"),
        )
    else:
        index_sha256 = _rewrite_cases_and_index(
            output,
            lambda records: records[0].update({"unexpected": True})
            if operation == "extra"
            else records[0].pop("status"),
        )
    with pytest.raises(ValueError, match="keys"):
        read_packet(output, index_sha256)


def test_config_rejects_boolean_execution_counter(tmp_path, monkeypatch):
    config = json.loads(CONFIG_PATH.read_bytes())
    config["execution_counters"]["model_calls"] = False
    config_path = tmp_path / "config.json"
    config_path.write_bytes(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    monkeypatch.setattr(prep, "CONFIG_PATH", config_path)
    with pytest.raises(ValueError, match="execution counters"):
        build_cases(code_revision="fixture-revision")


def test_manifest_rejects_boolean_execution_counter_after_rehash(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output,
        lambda manifest: manifest.update({"model_calls": False}),
    )
    with pytest.raises(ValueError, match="execution counters"):
        read_packet(output, index_sha256)


def test_case_record_rejects_boolean_execution_counter_after_rehash(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_cases_and_index(
        output,
        lambda records: records[0].update({"model_calls": False}),
    )
    with pytest.raises(ValueError, match="execution counters"):
        read_packet(output, index_sha256)


@pytest.mark.parametrize("policy", ["source_hash_policy", "public_payload_policy"])
def test_packet_rejects_tampered_policy_metadata(tmp_path, policy):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    index_sha256 = _rewrite_manifest_and_index(
        output,
        lambda manifest: manifest.update({policy: "tampered"}),
    )
    with pytest.raises(ValueError, match="policy"):
        read_packet(output, index_sha256)


def test_packet_rejects_tampered_protocol_metadata(tmp_path):
    output = tmp_path / "packet"
    publish_packet(output, code_revision="fixture-revision")
    config_path = output / "config.json"
    config = json.loads(config_path.read_bytes())
    config["study_id"] = "tampered"
    config_path.write_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    index_path = output / "index.json"
    index = json.loads(index_path.read_bytes())
    config_raw = config_path.read_bytes()
    index["artifacts"]["config.json"] = {
        "sha256": hashlib.sha256(config_raw).hexdigest(),
        "bytes": len(config_raw),
    }
    index_raw = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    index_path.write_bytes(index_raw)
    with pytest.raises(ValueError, match="config"):
        read_packet(output, hashlib.sha256(index_raw).hexdigest())


@pytest.mark.parametrize(
    "field", ["study_id", "event_pattern", "cross_entity_pattern", "historical_query_pattern"]
)
def test_current_config_rejects_changed_protocol_metadata(monkeypatch, tmp_path, field):
    config = json.loads(CONFIG_PATH.read_bytes())
    config[field] = "tampered"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(prep, "CONFIG_PATH", path)
    with pytest.raises(ValueError, match="protocol metadata"):
        build_cases(code_revision="fixture-revision")


def test_read_packet_rejects_root_replacement_after_open(monkeypatch, tmp_path):
    output = tmp_path / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    original = prep._assert_packet_ownership
    state = {"called": False}

    def replace_root(ownership):
        if not state["called"]:
            state["called"] = True
            replacement = tmp_path / "replacement"
            output.rename(replacement)
            output.mkdir()
        return original(ownership)

    monkeypatch.setattr(prep, "_assert_packet_ownership", replace_root)
    with pytest.raises(ValueError, match="ownership"):
        read_packet(output, receipt["index_sha256"])


def test_read_packet_rejects_parent_replacement(monkeypatch, tmp_path):
    parent = tmp_path / "owned-parent"
    parent.mkdir()
    output = parent / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    original = prep._assert_packet_ownership
    replaced = False

    def replace_parent(ownership):
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(tmp_path / "old-parent")
            parent.mkdir()
        return original(ownership)

    monkeypatch.setattr(prep, "_assert_packet_ownership", replace_parent)
    with pytest.raises(ValueError, match="ownership"):
        read_packet(output, receipt["index_sha256"])


def test_read_packet_rejects_index_replacement_between_reads(monkeypatch, tmp_path):
    output = tmp_path / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    original = prep._assert_packet_ownership
    calls = {"count": 0}

    def replace_index(ownership):
        calls["count"] += 1
        if calls["count"] == 2:
            ownership.index.rename(tmp_path / "index-backup.json")
        return original(ownership)

    monkeypatch.setattr(prep, "_assert_packet_ownership", replace_index)
    with pytest.raises(ValueError, match="ownership"):
        read_packet(output, receipt["index_sha256"])


@pytest.mark.parametrize("name", ["index.json", "manifest.json"])
def test_packet_rejects_in_place_change_after_verified_read(monkeypatch, tmp_path, name):
    output = tmp_path / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    original = prep._read_packet_file
    changed = False

    def rewrite_after_read(ownership, filename):
        nonlocal changed
        raw = original(ownership, filename)
        if filename == name and not changed:
            changed = True
            path = output / name
            path.write_bytes(path.read_bytes() + b" ")
        return raw

    monkeypatch.setattr(prep, "_read_packet_file", rewrite_after_read)
    with pytest.raises(ValueError, match="ownership"):
        read_packet(output, receipt["index_sha256"])


def test_read_packet_rejects_artifact_replacement_between_reads(monkeypatch, tmp_path):
    output = tmp_path / "packet"
    receipt = publish_packet(output, code_revision="fixture-revision")
    original = prep._assert_packet_ownership
    state = {"replaced": False}

    def replace_artifact(ownership):
        if "config.json" in ownership.artifacts and not state["replaced"]:
            state["replaced"] = True
            ownership.artifacts["config.json"].rename(tmp_path / "config-backup.json")
        return original(ownership)

    monkeypatch.setattr(prep, "_assert_packet_ownership", replace_artifact)
    with pytest.raises(ValueError, match="ownership"):
        read_packet(output, receipt["index_sha256"])
