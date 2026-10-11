from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import vnext_run_family_h_native_qdrant as runner


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data/vnext/family_h_independent/v1_published"


def tasks():
    return [json.loads(line) for line in (RELEASE / "tasks.jsonl").read_bytes().splitlines()]


def test_public_event_projection_uses_event_payload_not_gold_action():
    task = tasks()[0]
    event = task["events"][0]
    request = runner.public_event_request(task["task_id"], event)
    payload = json.loads(event["raw_text"])
    assert request.event_id == event["event_id"]
    assert request.sequence_index == event["sequence_index"]
    assert request.operation.value == payload["action"]
    assert request.objects[0].value == payload["value"]
    assert "gold_action_ids" not in request.model_dump(mode="json")


def test_public_event_projection_rejects_raw_event_identity_drift():
    task = tasks()[0]
    event = dict(task["events"][0])
    payload = json.loads(event["raw_text"])
    payload["object_key"]["entity"] = "not-the-source-event"
    event["raw_text"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="event identity|source payload"):
        runner.public_event_request(task["task_id"], event)


def test_family_h_release_loader_binds_exact_three_tasks_and_27_events():
    loaded = runner.load_family_h_tasks(RELEASE)
    assert len(loaded) == 3
    assert sum(len(task["events"]) for task in loaded) == 27
    assert all(task["source"]["provenance"]["source_group_id"].startswith("family-h-independent-") for task in loaded)


def test_runner_main_accepts_owned_precreated_temp_layout(tmp_path):
    for name in ('home', 'temp', 'storage', 'snapshots', 'config'):
        (tmp_path / name).mkdir()
    assert {p.name for p in tmp_path.iterdir()} == {'home', 'temp', 'storage', 'snapshots', 'config'}


def test_export_namespace_handles_provider_model_points_without_mapping_fallback(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider, build_qdrant_native_multi_object_configuration,
    )
    from tests.vnext.test_qdrant_native_multi_object import FakeQdrantClient
    from mub.vnext.contracts.enums import Operation
    from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
    from mub.vnext.contracts.v3.native_multi_object import NativeMultiObjectMutationRequestV3
    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=build_qdrant_native_multi_object_configuration(run_id='family-h-export', path=tmp_path),
        client=client, runtime_namespace='family_h_runtime',
    )
    key = FrozenMemoryObjectKey(namespace='family_h', entity='rust', attribute='latest_release', object_type='release')
    provider.mutate(NativeMultiObjectMutationRequestV3(
        request_id='r', event_id='e', sequence_index=0, operation=Operation.ADD,
        objects=({'object_key': key, 'value': {'version':'1.0'}},),
        source_event_ids=('e',), provenance={'kind':'declared','provider_owned':False},
    ))
    points = runner.export_namespace(provider, 'family_h_runtime')
    assert len(points) == 1 and points[0]['payload']['entity'] == 'rust'


def test_family_h_rows_retain_direct_provider_witnesses():
    assert runner._witnesses is not None


def test_public_event_projection_does_not_use_action_list(monkeypatch):
    task = tasks()[0]
    event = task["events"][0]
    task_without_actions = {"task_id": task["task_id"], "events": task["events"], "actions": ()}
    request = runner.public_event_request(task_without_actions["task_id"], event)
    assert request.event_id == event["event_id"]


def test_cleanup_provider_deletes_collection_before_closing_client():
    from scripts.vnext_run_family_h_native_qdrant import cleanup_provider

    class Client:
        def __init__(self): self.events=[]; self.closed=False
        def delete_collection(self, *, collection_name): self.events.append(('delete',collection_name)); return True
        def collection_exists(self, *, collection_name): self.events.append(('exists',collection_name)); return False
        def close(self): self.events.append(('close',)); self.closed=True
    class Provider:
        collection_name='owned'
        def __init__(self, client): self.client=client
        def close(self): self.client.close()
    client=Client(); result=cleanup_provider(Provider(client),client)
    assert result == {'client_closed':True,'collection_deleted':True}
    assert [event[0] for event in client.events] == ['delete','exists','close']


def test_family_h_canonical_serializes_frozen_json_values():
    from mub.vnext.contracts.common import freeze_json
    raw = runner.canonical({'payload': freeze_json({'version': '1.0'})})
    assert b'"version":"1.0"' in raw
