from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import scripts.vnext_run_cross_domain_state as runner

import pytest

from scripts.vnext_run_cross_domain_state import (
    BEA_INDEX_SHA256,
    NOAA_INDEX_SHA256,
    load_public_cells,
    run_cross_domain_cells,
)


ROOT = Path(__file__).resolve().parents[2]
NOAA_ROOT = ROOT / "data" / "vnext" / "family_h_cross_domain_noaa" / "v1"
BEA_ROOT = ROOT / "data" / "vnext" / "family_h_cross_domain_bea_gdp" / "v1"


class _FakeManager:
    def __init__(self):
        self.entries = {}
        self.closed = False
        self.deleted = False
        self.provider_witnesses = ({"method": "fake", "call_index": 0},)

    def ingest(self, event, *, operation, object_key, value):
        key = tuple(object_key[field] for field in ("namespace", "entity", "attribute", "subkey"))
        if operation == "noop":
            return {"effective_operation": "noop", "affected_entry_ids": []}
        self.entries[key] = {
            "entry_id": f"fake:{key}",
            "object_key": dict(object_key),
            "value": value,
            "content": f"{key}={value!r}",
            "source_event_ids": [event.event_id],
            "score": 1.0,
            "rank": 1,
            "version_metadata": {"version_index": event.sequence_index},
        }
        return {"effective_operation": operation, "affected_entry_ids": [self.entries[key]["entry_id"]]}

    def export_entries(self):
        return list(self.entries.values())

    def retrieve(self, query):
        assert set(vars(query)) == {"query_id", "text"}
        return {"entries": self.export_entries(), "context_order": "fake_provider_order"}

    def reset(self, _task):
        self.entries.clear()

    def close(self):
        self.closed = True


def _cells():
    return load_public_cells(
        {
            "noaa": (NOAA_ROOT, NOAA_INDEX_SHA256),
            "bea": (BEA_ROOT, BEA_INDEX_SHA256),
        }
    )


def test_public_projection_binds_release_events_without_gold_fields():
    cells = _cells()

    assert [cell["release_name"] for cell in cells] == ["bea", "noaa"]
    assert [len(cell["events"]) for cell in cells] == [3, 7]
    assert cells[0]["events"][-1]["value"] == 2.4
    assert cells[1]["events"][-1]["value"] == 75
    assert all("actions" not in cell and "gold_evidence" not in cell for cell in cells)
    assert all(set(cell["query"]) == {"query_id", "text"} for cell in cells)


def test_cross_domain_replay_uses_only_public_cell_and_cleans_up():
    cells = _cells()
    managers = []

    def factory(cell, repetition_index):
        assert set(cell) == {
            "release_name", "release_index_sha256", "task_id", "task_sha256", "events", "query"
        }
        manager = _FakeManager()
        managers.append((cell["release_name"], repetition_index, manager))
        return manager, lambda: setattr(manager, "deleted", True)

    result = run_cross_domain_cells(cells, manager_factory=factory, repetitions=2)

    assert result["status"] == "COMPLETE"
    assert result["scientific_evidence"] is False
    assert result["model_loads"] is None
    assert result["provider_model_calls"] is None
    assert result["generations"] is None
    assert result["expected_operations"] == 20
    assert result["observed_operation_counts"]["ingest_completed"] == 20
    assert result["cells"] == 2
    assert result["events"] == 10
    assert result["state_step_matches"] == 20
    assert result["final_state_matches"] == 4
    assert result["retrieval_typed_object_matches"] == 4
    assert result["repetitions_equal"] is True
    assert len(managers) == 4
    assert all(manager.closed and manager.deleted and manager.entries == {} for _, _, manager in managers)


def test_wrong_release_hash_is_rejected():
    with pytest.raises(ValueError, match="release index hash mismatch"):
        load_public_cells({"bea": (BEA_ROOT, "0" * 64)})


def _payload(domain="bea"):
    root = BEA_ROOT if domain == "bea" else NOAA_ROOT
    task = json.loads((root / "tasks.jsonl").read_text(encoding="utf-8"))
    return json.loads(task["events"][0]["raw_text"])


@pytest.mark.parametrize("change", [
    {"gold_value": 2.3}, {"other": "hidden target"}, {"action": "DELETE"},
    {"action": "NOOP"}, {"value": True}, {"value": "2.3"},
    {"value": float("inf")}, {"value": float("nan")},
    {"estimate_stage": "unknown"}, {"release_date": "2025-02-27"},
])
def test_public_parser_rejects_noncontract_fields_operations_and_values(change):
    payload = _payload()
    payload.update(change)
    with pytest.raises(ValueError):
        runner._parse_public_event(SimpleNamespace(raw_text=json.dumps(payload)))


@pytest.mark.parametrize("domain", ["bea", "noaa"])
def test_public_parser_rejects_duplicate_keys_at_any_depth(domain):
    raw = json.dumps(_payload(domain))
    for corrupted in (raw.replace('"value":', '"value": 99, "value":'),
                      raw.replace('"namespace":', '"namespace": "bad", "namespace":')):
        with pytest.raises(ValueError, match="duplicate"):
            runner._parse_public_event(SimpleNamespace(raw_text=corrupted))


def test_noaa_descriptor_rejects_malformed_hash_or_heading():
    for change in ({"normalized_record_sha256": "z" * 64}, {"source_heading_anchor": "other"}):
        payload = _payload("noaa")
        payload.update(change)
        with pytest.raises(ValueError):
            runner._parse_public_event(SimpleNamespace(raw_text=json.dumps(payload)))


def test_release_hash_is_frozen_before_reader_is_called(monkeypatch):
    called = []
    monkeypatch.setitem(runner._RELEASE_READERS, "bea", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="release index hash mismatch"):
        load_public_cells({"bea": (BEA_ROOT, "f" * 64)})
    assert called == []


def test_public_projection_precedes_replay_and_rejects_gold_source_drift(monkeypatch):
    original = runner.read_bea_release(BEA_ROOT, BEA_INDEX_SHA256)
    task = original["tasks"][0]
    payload = json.loads(task.events[-1].raw_text)
    payload["value"] = 9.9
    altered = task.model_copy(update={"events": (*task.events[:-1], task.events[-1].model_copy(
        update={"raw_text": json.dumps(payload)}))})
    monkeypatch.setitem(runner._RELEASE_READERS, "bea", lambda *args: {**original, "tasks": [altered]})
    with pytest.raises(ValueError, match="public.*(replay|gold|action)"):
        load_public_cells({"bea": (BEA_ROOT, BEA_INDEX_SHA256)})


@pytest.mark.parametrize("case", ["empty", "duplicate", "repetitions", "event_hidden", "query_hidden", "bad_query", "nonfinite", "key_drift"])
def test_invalid_input_is_rejected_before_manager_creation(case):
    cells = deepcopy(_cells())
    repetitions = 2
    if case == "empty":
        cells = []
    elif case == "duplicate":
        cells = [cells[0], cells[0]]
    elif case == "repetitions":
        repetitions = 1
    elif case == "event_hidden":
        cells[0]["events"][0]["gold"] = 2.3
    elif case == "query_hidden":
        cells[0]["query"]["target_object_keys"] = []
    elif case == "bad_query":
        cells[0]["query"]["text"] = cells[1]["query"]["text"]
    elif case == "nonfinite":
        cells[0]["events"][0]["value"] = float("inf")
    else:
        cells[0]["events"][0]["object_key"]["entity"] = "other"
    called = []
    with pytest.raises(ValueError):
        run_cross_domain_cells(cells, manager_factory=lambda *args: called.append(args), repetitions=repetitions)
    assert called == []


def test_normalized_entries_retained_and_same_value_update_relinks_source():
    result = run_cross_domain_cells(_cells(), manager_factory=lambda *args: (_FakeManager(), lambda: None))
    bea = result["rows"][0]
    assert bea["steps"][0]["entries"][0]["value"] == bea["steps"][1]["entries"][0]["value"]
    assert bea["steps"][1]["operation"] == "update"
    assert bea["steps"][1]["entries"][0]["source_event_ids"] == [_cells()[0]["events"][1]["event_id"]]
    entry = bea["retrieval"]["entries"][0]
    assert entry["score"] == 1.0 and entry["rank"] == 1
    assert entry["version_metadata"] == {"version_index": 2}


@pytest.mark.parametrize("drift", ["score", "rank", "content", "source", "version", "order_metadata"])
def test_full_semantic_repeat_comparison_detects_drift_even_when_correct(drift):
    class Drifting(_FakeManager):
        def retrieve(self, query):
            trace = deepcopy(super().retrieve(query))
            entry = trace["entries"][0]
            if drift == "source":
                entry["source_event_ids"].append("extra-public-event")
            elif drift == "version":
                entry["version_metadata"]["extra"] = "drift"
            elif drift == "order_metadata":
                trace["version_metadata"] = {"provider_scores_preserved": False}
            else:
                entry[drift] = {"score": 0.2, "rank": 2, "content": "different"}[drift]
            return trace
    result = run_cross_domain_cells(_cells()[:1], manager_factory=lambda cell, rep: (
        Drifting() if rep else _FakeManager(), lambda: None))
    assert all(row["retrieval_typed_object_match"] for row in result["rows"])
    assert result["repetitions_equal"] is False
    assert result["status"] == "BLOCKED"
    assert result["state_step_matches"] is None


@pytest.mark.parametrize("failure", ["ingest", "export", "retrieve", "reset", "close", "delete", "nonfinite_trace"])
def test_partial_technical_failure_stops_new_managers_and_nulls_quality(failure):
    cleanup = []
    class Broken(_FakeManager):
        def ingest(self, event, **kwargs):
            result = super().ingest(event, **kwargs)
            if failure == "ingest" and event.sequence_index == 1:
                raise RuntimeError("mutation reply lost")
            return result
        def export_entries(self):
            if failure == "export" and self.entries:
                raise RuntimeError("scroll failed")
            return super().export_entries()
        def retrieve(self, query):
            if failure == "retrieve":
                raise RuntimeError("query failed")
            trace = deepcopy(super().retrieve(query))
            if failure == "nonfinite_trace":
                trace["entries"][0]["score"] = float("nan")
            return trace
        def reset(self, task):
            cleanup.append("reset")
            super().reset(task)
            if failure == "reset":
                raise RuntimeError("reset reply lost")
        def close(self):
            cleanup.append("close")
            if failure == "close":
                raise RuntimeError("close failed")
            super().close()
    calls = []
    def factory(cell, rep):
        calls.append((cell["release_name"], rep))
        def delete():
            cleanup.append("delete")
            if failure == "delete":
                raise RuntimeError("delete failed")
        return Broken(), delete
    result = run_cross_domain_cells(_cells(), manager_factory=factory)
    assert len(calls) == 1
    assert cleanup == ["reset", "close", "delete"]
    assert len(result["rows"]) == 4 and all(row["status"] == "BLOCKED" for row in result["rows"])
    assert result["repetitions_equal"] in (False, None)
    for metric in ("state_step_matches", "final_state_matches", "retrieval_typed_object_matches"):
        assert result[metric] is None
    assert result["observed_operation_counts"]["ingest_attempted"] > 0


def test_mutation_observer_receives_detached_normalized_step():
    seen = []
    def observer(row):
        seen.append(deepcopy(row))
        row["step"]["entries"].clear()
    result = run_cross_domain_cells(_cells(), manager_factory=lambda *args: (_FakeManager(), lambda: None),
                                    mutation_observer=observer)
    assert len(seen) == 20
    assert seen[0]["release_name"] == "bea" and seen[0]["repetition_index"] == 0
    assert result["rows"][0]["steps"][0]["entries"]


class _ProtocolClient:
    """Small in-memory direct-Qdrant protocol with shared collection ownership."""
    def __init__(self, server):
        self.server = server
        self.closed = False

    def collection_exists(self, *, collection_name):
        if self.server.get("auth_failure"):
            raise PermissionError("redacted")
        return self.server.get("foreign", False) or collection_name in self.server["collections"]

    def create_collection(self, *, collection_name, vectors_config):
        self.server["creates"].append(collection_name)
        assert collection_name not in self.server["collections"]
        self.server["collections"][collection_name] = {}
        if self.server.get("ambiguous_create"):
            raise TimeoutError("reply lost after create")
        return self.server.get("create_result", True)

    def get_collection(self, *, collection_name):
        return {"config": {"params": {"vectors": {"size": 1 if self.server.get("bad_schema") else 384,
                                                  "distance": "Cosine"}}}}

    def delete_collection(self, *, collection_name):
        self.server["deletes"].append(collection_name)
        self.server["collections"].pop(collection_name, None)
        return True

    def upsert(self, *, collection_name, points, wait):
        for point in points:
            self.server["collections"][collection_name][point["id"]] = point
        return {"operation_id": "upsert"}

    def retrieve(self, *, collection_name, ids, **kwargs):
        return [self.server["collections"][collection_name][key] for key in ids
                if key in self.server["collections"][collection_name]]

    def scroll(self, *, collection_name, **kwargs):
        return list(self.server["collections"][collection_name].values()), None

    def query_points(self, *, collection_name, **kwargs):
        return {"points": [{**point, "score": 0.75} for point in self.server["collections"][collection_name].values()]}

    def delete(self, *, collection_name, **kwargs):
        self.server["collections"][collection_name].clear()
        return {"operation_id": "delete"}

    def close(self):
        self.closed = True


def _server(**kwargs):
    return {"collections": {}, "creates": [], "deletes": [], **kwargs}


def _factory(tmp_path, client_factory=None, **kwargs):
    return runner.build_qdrant_manager_factory(
        endpoint="http://127.0.0.1:16333", qdrant_base_path=tmp_path / "qdrant",
        source_revision="a" * 40, runtime_revision="b" * 40,
        source_hash="c" * 64, runtime_hash="d" * 64, client_factory=client_factory, **kwargs)


def _clients_factory(server, *, fail_second=False):
    clients = []
    def make():
        if fail_second and clients:
            raise PermissionError("second-client auth creation failed")
        client = _ProtocolClient(server)
        clients.append(client)
        return client
    return clients, make


def test_factory_never_deletes_foreign_collection(tmp_path):
    server = _server(foreign=True)
    clients, make = _clients_factory(server)
    with pytest.raises(ValueError, match="already exists"):
        _factory(tmp_path, make)(_cells()[0], 0)
    assert server["deletes"] == [] and server["creates"] == []
    assert len(clients) == 2 and all(client.closed for client in clients)


@pytest.mark.parametrize("failure", ["second_client", "auth", "bad_schema"])
def test_factory_closes_all_created_clients_on_auth_or_initialization_failure(tmp_path, failure):
    server = _server(auth_failure=failure == "auth", bad_schema=failure == "bad_schema")
    clients, make = _clients_factory(server, fail_second=failure == "second_client")
    with pytest.raises(Exception):
        _factory(tmp_path, make)(_cells()[0], 0)
    assert all(client.closed for client in clients)
    if failure == "bad_schema":
        assert len(server["creates"]) == 1 and server["deletes"] == server["creates"]
    else:
        assert server["creates"] == [] and server["deletes"] == []


@pytest.mark.parametrize("mode", ["timeout", "false_reply"])
def test_factory_ambiguous_create_never_claims_deletion_authority(tmp_path, mode):
    server = _server(ambiguous_create=mode == "timeout", create_result=mode != "false_reply")
    clients, make = _clients_factory(server)
    with pytest.raises(Exception) as exc:
        _factory(tmp_path, make)(_cells()[0], 0)
    assert server["deletes"] == [] and all(client.closed for client in clients)
    assert exc.value.cleanup_diagnostics["collection_claim_status"] == "AMBIGUOUS"


def test_owned_factory_uses_explicit_create_and_can_complete_real_adapter_protocol(tmp_path):
    server = _server()
    clients, make = _clients_factory(server)
    result = run_cross_domain_cells(_cells(), manager_factory=_factory(tmp_path, make, run_scope="unit-run"))
    assert result["status"] == "COMPLETE"
    assert result["state_step_matches"] == 20
    assert len(server["creates"]) == 4 and server["deletes"] == server["creates"]
    assert server["collections"] == {} and len(clients) == 8 and all(client.closed for client in clients)
    assert result["rows"][0]["steps"][0]["entries"][0]["entry_id"] != result["rows"][1]["steps"][0]["entries"][0]["entry_id"]
    assert result["repetitions_equal"] is True


def test_production_clients_use_pinned_version_env_auth_and_per_client_transport(tmp_path, monkeypatch):
    server = _server()
    clients, kwargs_seen, transports = [], [], []
    class Transport:
        def close(self):
            pass
    def transport_factory():
        transport = Transport()
        transports.append(transport)
        return transport
    def qdrant_client(**kwargs):
        kwargs_seen.append(kwargs)
        client = _ProtocolClient(server)
        clients.append(client)
        return client
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=qdrant_client, models=SimpleNamespace()))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.19.0")
    monkeypatch.setenv("UNIT_QDRANT_KEY", "test-only-secret")
    factory = _factory(tmp_path, api_key_env="UNIT_QDRANT_KEY", transport_factory=transport_factory)
    manager, delete = factory(_cells()[0], 0)
    manager.reset(None)
    manager.close()
    delete()
    assert len(kwargs_seen) == len(transports) == 2
    for index, kwargs in enumerate(kwargs_seen):
        assert kwargs["timeout"] == 10 and kwargs["trust_env"] is False
        assert kwargs["url"] == "http://127.0.0.1:16333" and kwargs["api_key"] == "test-only-secret"
        assert kwargs["transport"] is transports[index]
    assert all(client.closed for client in clients)


def test_production_version_drift_is_rejected_before_client_creation(tmp_path, monkeypatch):
    called = []
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=lambda **kwargs: called.append(kwargs), models=SimpleNamespace()))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.18.0")
    with pytest.raises(ValueError, match="1.19.0"):
        _factory(tmp_path)
    assert called == []


def test_noaa_valid_but_unpinned_record_hash_is_rejected():
    payload = _payload("noaa")
    payload["normalized_record_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="record hash"):
        runner._parse_public_event(SimpleNamespace(raw_text=json.dumps(payload)))


@pytest.mark.parametrize("field", ["entries_sha256", "entry_id", "affected_entry_ids"])
def test_metadata_field_names_are_not_mistaken_for_run_specific_ids_or_hashes(field):
    class Drift(_FakeManager):
        def __init__(self, rep):
            super().__init__()
            self.rep = rep
        def retrieve(self, query):
            trace = deepcopy(super().retrieve(query))
            value = f"metadata-{self.rep}"
            trace["entries"][0]["version_metadata"][field] = [value] if field == "affected_entry_ids" else value
            return trace
    result = run_cross_domain_cells(_cells()[:1], manager_factory=lambda cell, rep: (Drift(rep), lambda: None))
    assert result["repetitions_equal"] is False


def test_nested_nonfinite_metadata_blocks_run():
    class Invalid(_FakeManager):
        def retrieve(self, query):
            trace = deepcopy(super().retrieve(query))
            trace["version_metadata"] = {"nested": [float("inf")]}
            return trace
    result = run_cross_domain_cells(_cells(), manager_factory=lambda *args: (Invalid(), lambda: None))
    assert result["status"] == "BLOCKED" and result["repetitions_equal"] is None
    assert result["state_step_matches"] is None
    assert result["observed_operation_counts"]["manager_created"] == 1


def test_observer_failure_is_fail_stop_and_cleanup_still_runs():
    managers = []
    def factory(*args):
        manager = _FakeManager()
        managers.append(manager)
        return manager, lambda: setattr(manager, "deleted", True)
    def observer(row):
        raise OSError("persistence unavailable")
    result = run_cross_domain_cells(_cells(), manager_factory=factory, mutation_observer=observer)
    assert result["status"] == "BLOCKED" and len(managers) == 1
    assert managers[0].closed and managers[0].deleted and managers[0].entries == {}


def test_retrieval_order_drift_is_not_hidden_by_equal_correctness():
    class Reordered(_FakeManager):
        def __init__(self, reverse):
            super().__init__()
            self.reverse = reverse
        def retrieve(self, query):
            trace = deepcopy(super().retrieve(query))
            extra = deepcopy(trace["entries"][0])
            extra.update(entry_id="other-entry", value=99, rank=2)
            trace["entries"].append(extra)
            if self.reverse:
                trace["entries"].reverse()
            return trace
    result = run_cross_domain_cells(_cells()[:1], manager_factory=lambda cell, rep: (Reordered(bool(rep)), lambda: None))
    assert all(row["final_state_match"] and row["retrieval_typed_object_match"] for row in result["rows"])
    assert result["repetitions_equal"] is False


def test_missing_same_value_update_link_is_a_state_error_not_silently_noop():
    class NoRelink(_FakeManager):
        def ingest(self, event, **kwargs):
            if self.entries and list(self.entries.values())[0]["value"] == kwargs["value"]:
                return {"effective_operation": "noop", "affected_entry_ids": []}
            return super().ingest(event, **kwargs)
    result = run_cross_domain_cells(_cells(), manager_factory=lambda *args: (NoRelink(), lambda: None))
    assert result["status"] == "COMPLETE"
    assert result["state_step_matches"] == 16
    assert result["rows"][0]["steps"][1]["operation"] == "update"
    assert result["rows"][0]["steps"][1]["current_source_event_link"] is False


def test_second_production_client_auth_failure_closes_first_client_and_second_transport(tmp_path, monkeypatch):
    server = _server()
    clients, transports = [], []
    class Transport:
        closed = False
        def close(self):
            self.closed = True
    def transport_factory():
        item = Transport()
        transports.append(item)
        return item
    def qdrant_client(**kwargs):
        if clients:
            raise PermissionError("second constructor failed")
        client = _ProtocolClient(server)
        clients.append(client)
        return client
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=qdrant_client, models=SimpleNamespace()))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.19.0")
    monkeypatch.setenv("UNIT_QDRANT_KEY", "test-only-secret")
    factory = _factory(tmp_path, api_key_env="UNIT_QDRANT_KEY", transport_factory=transport_factory)
    with pytest.raises(PermissionError):
        factory(_cells()[0], 0)
    assert clients[0].closed and transports[1].closed
    assert server["creates"] == server["deletes"] == []


def test_malformed_public_projection_is_rejected_before_replay(monkeypatch):
    original = runner.read_bea_release(BEA_ROOT, BEA_INDEX_SHA256)
    task = original["tasks"][0]
    event = task.events[0].model_copy(update={"raw_text": '{"hidden":"value"}'})
    task = task.model_copy(update={"events": (event, *task.events[1:])})
    calls = []
    monkeypatch.setitem(runner._RELEASE_READERS, "bea", lambda *args: {**original, "tasks": [task]})
    monkeypatch.setattr(runner, "replay_task_v3", lambda task: calls.append(task))
    with pytest.raises(ValueError, match="descriptor fields"):
        load_public_cells({"bea": (BEA_ROOT, BEA_INDEX_SHA256)})
    assert calls == []


@pytest.mark.parametrize("value", [{1: "coerced-key"}, {"nested": {False: "coerced-key"}}, {"nested": (1, 2)}])
def test_metadata_must_be_strict_json_without_key_or_container_coercion(value):
    class Invalid(_FakeManager):
        def retrieve(self, query):
            trace = deepcopy(super().retrieve(query))
            trace["version_metadata"] = value
            return trace
    result = run_cross_domain_cells(_cells(), manager_factory=lambda *args: (Invalid(), lambda: None))
    assert result["status"] == "BLOCKED" and result["state_step_matches"] is None
    assert result["observed_operation_counts"]["manager_created"] == 1


def test_delete_failure_preserves_owned_diagnostic_and_closes_control(tmp_path):
    server = _server(bad_schema=True)
    clients, make = _clients_factory(server)
    original_make = make
    def make_broken_control():
        client = original_make()
        if len(clients) == 2:
            def fail_delete(**kwargs):
                raise OSError("deletion failed")
            client.delete_collection = fail_delete
        return client
    with pytest.raises(Exception) as exc:
        _factory(tmp_path, make_broken_control)(_cells()[0], 0)
    assert all(client.closed for client in clients)
    assert exc.value.cleanup_diagnostics["collection_claim_status"] == "OWNED"
    assert exc.value.cleanup_diagnostics["delete_error_type"] == "OSError"
    assert server["collections"]


def test_factory_cleanup_callback_closes_control_even_when_delete_raises(tmp_path):
    server = _server()
    clients, make = _clients_factory(server)
    manager, delete = _factory(tmp_path, make)(_cells()[0], 0)
    manager.reset(None)
    manager.close()
    def fail_delete(**kwargs):
        raise OSError("deletion failed")
    clients[1].delete_collection = fail_delete
    with pytest.raises(OSError):
        delete()
    assert all(client.closed for client in clients)


def test_production_constructor_failure_survives_python310_without_add_note(tmp_path, monkeypatch):
    class Python310Error(PermissionError):
        add_note = None
    class BrokenTransport:
        def close(self):
            raise OSError('close failed')
    def qdrant_client(**kwargs):
        raise Python310Error('construction failed')
    monkeypatch.setitem(sys.modules, 'qdrant_client', SimpleNamespace(QdrantClient=qdrant_client, models=SimpleNamespace()))
    monkeypatch.setattr('importlib.metadata.version', lambda name: '1.19.0')
    monkeypatch.setenv('UNIT_QDRANT_KEY', 'test-only-secret')
    factory = _factory(tmp_path, api_key_env='UNIT_QDRANT_KEY', transport_factory=BrokenTransport)
    with pytest.raises(Python310Error):
        factory(_cells()[0], 0)
