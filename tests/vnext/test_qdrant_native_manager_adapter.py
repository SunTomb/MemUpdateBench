from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.v3.native_multi_object import NativeResetResultV3


@dataclass
class Point:
    id: str
    payload: dict
    score: float | None = None


class FakeQdrantClient:
    def __init__(self) -> None:
        self.points: dict[str, Point] = {}
        self.calls: list[tuple[str, dict]] = []
        self.created = False
        self.closed = False

    def collection_exists(self, *, collection_name):
        self.calls.append(("collection_exists", {"collection_name": collection_name}))
        return self.created

    def create_collection(self, *, collection_name, vectors_config):
        self.calls.append(("create_collection", {"collection_name": collection_name}))
        self.created = True

    def get_collection(self, *, collection_name):
        self.calls.append(("get_collection", {"collection_name": collection_name}))
        return {"config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}}

    def upsert(self, *, collection_name, points, wait=True):
        self.calls.append(("upsert", {"collection_name": collection_name, "wait": wait}))
        for point in points:
            self.points[str(point["id"])] = Point(str(point["id"]), dict(point["payload"]))
        return {"operation_id": "not-public"}

    def retrieve(self, *, collection_name, ids, with_payload=True, with_vectors=False):
        self.calls.append(("retrieve", {"collection_name": collection_name, "ids": list(ids)}))
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def delete(self, *, collection_name, points_selector, wait=True):
        self.calls.append(("delete", {"collection_name": collection_name, "wait": wait}))
        doomed = []
        for point_id, point in self.points.items():
            if _matches(point.payload, points_selector):
                doomed.append(point_id)
        for point_id in doomed:
            del self.points[point_id]
        return {"operation_id": "not-public"}

    def query_points(self, *, collection_name, query, limit, query_filter=None, with_payload=True, with_vectors=False):
        self.calls.append(("query_points", {"collection_name": collection_name, "limit": limit}))
        points = [point for point in self.points.values() if _matches(point.payload, query_filter)]
        return SimpleNamespace(
            points=[Point(point.id, dict(point.payload), 1.0 - index * 0.1) for index, point in enumerate(points[:limit])]
        )

    def scroll(self, *, collection_name, scroll_filter=None, limit, with_payload=True, with_vectors=False):
        self.calls.append(("scroll", {"collection_name": collection_name, "limit": limit}))
        return [point for point in self.points.values() if _matches(point.payload, scroll_filter)][:limit], None

    def close(self):
        self.closed = True
class PaginatedFakeQdrantClient(FakeQdrantClient):
    def __init__(self, cursor_mode="valid") -> None:
        super().__init__()
        self.cursor_mode = cursor_mode
        self.scroll_offsets = []

    def scroll(self, *, collection_name, scroll_filter=None, limit, with_payload=True, with_vectors=False, offset=None):
        self.calls.append(("scroll", {"collection_name": collection_name, "limit": limit, "offset": offset}))
        points = [point for point in self.points.values() if _matches(point.payload, scroll_filter)]
        if offset is None:
            page = points[:1]
            if self.cursor_mode == "malformed":
                return page, object()
            return page, "repeat" if self.cursor_mode == "repeated" else "next"
        if offset == "next":
            return points[1:], None
        if offset == "repeat":
            return points[1:], "repeat"
        raise AssertionError(f"unexpected scroll offset: {offset!r}")


def _matches(payload: dict, selector) -> bool:
    if selector is None:
        return True
    if isinstance(selector, dict) and "filter" in selector:
        return _matches(payload, selector["filter"])
    if isinstance(selector, dict) and "must" in selector:
        return all(_matches_condition(payload, condition) for condition in selector["must"])
    if hasattr(selector, "filter"):
        return _matches(payload, selector.filter)
    if hasattr(selector, "must"):
        return all(_matches_condition(payload, condition) for condition in selector.must)
    return True


def _matches_condition(payload: dict, condition) -> bool:
    if isinstance(condition, dict):
        key = condition.get("key")
        match = condition.get("match", {})
        expected = match.value if hasattr(match, "value") else match.get("value")
    else:
        key = condition.key
        match = condition.match
        expected = match.value
    return payload.get(key) == expected


def _config(tmp_path: Path):
    from mub.vnext.external.providers.qdrant_native_adapter import (
        build_qdrant_native_manager_configuration,
    )

    return build_qdrant_native_manager_configuration(
        run_id="manager-run",
        path=tmp_path / "stores",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash="c" * 64,
        runtime_hash="d" * 64,
    )


def _key():
    from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey

    return FrozenMemoryObjectKey(
        object_type="profile", namespace="default", entity="alice", attribute="city", subkey=None
    )


def test_manager_configuration_binds_source_runtime_and_provider_config_hashes(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import (
        QdrantNativeManagerConfigurationV1,
    )

    config = _config(tmp_path)
    assert isinstance(config, QdrantNativeManagerConfigurationV1)
    assert config.provider_configuration.run_id == "manager-run"
    assert config.source_revision == "a" * 40
    assert config.runtime_revision == "b" * 40
    assert len(config.configuration_hash) == 64


def test_manager_uses_only_provider_state_and_keeps_answer_target_out_of_retrieve(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import QdrantNativeExternalManagerV1
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = FakeQdrantClient()
    manager = QdrantNativeExternalManagerV1(
        configuration=_config(tmp_path),
        provider_factory=lambda *, configuration, runtime_namespace: QdrantNativeMultiObjectProvider(
            configuration=configuration, client=client, runtime_namespace=runtime_namespace
        ),
    )
    key = _key()
    event = SimpleNamespace(event_id="event-1", sequence_index=0)
    manager.reset(SimpleNamespace(task_id="task-1"))
    result = manager.ingest(event, operation="add", value="Paris", object_key=key)
    assert result == {"effective_operation": "add", "affected_entry_ids": [manager.point_id(key)]}
    entries = manager.export_entries()
    assert entries[0]["value"] == "Paris"

    class Query:
        query_id = "query-1"
        text = "visible city"

        @property
        def target_object_keys(self):
            raise AssertionError("answer target must not be read by native retrieval")

    retrieved = manager.retrieve(Query())
    assert retrieved["entries"][0]["value"] == "Paris"
    assert retrieved["entries"][0]["entry_id"] == manager.point_id(key)
    assert retrieved["context_order"] == "provider_order"
    assert not manager.identity["uses_local_authoritative_store"]
    assert not manager.identity["uses_local_reranking"]
    assert not manager.identity["uses_local_filtering"]
    assert not manager.identity["hidden_target_injection"]
    assert manager.production_bound is False
    assert manager.provider_witness_hash
    assert "provider_operation_id" not in str(manager.identity)
    manager.close()
    assert client.closed is True


def test_manager_dispatch_exposes_only_provider_capabilities_and_witness_hashes(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import QdrantNativeExternalManagerV1
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = FakeQdrantClient()
    manager = QdrantNativeExternalManagerV1(
        configuration=_config(tmp_path),
        provider_factory=lambda *, configuration, runtime_namespace: QdrantNativeMultiObjectProvider(
            configuration=configuration, client=client, runtime_namespace=runtime_namespace
        ),
    )
    capabilities = manager.capabilities()
    assert capabilities.direct_provider_crud is True
    assert capabilities.provider_owned_collection is True
    assert capabilities.supports_native_answer is False
    assert capabilities.uses_local_authoritative_store is False
    assert capabilities.uses_local_reranking is False
    assert capabilities.uses_local_deduplication is False
    assert capabilities.uses_local_filtering is False
    assert capabilities.hidden_target_injection is False
    manager.close()


def test_manager_factory_reports_blocked_when_qdrant_dependency_is_unavailable(monkeypatch, tmp_path):
    from mub.vnext.external.providers import qdrant_native_multi_object as provider_module
    from mub.vnext.external.providers.qdrant_native_adapter import (
        QdrantNativeExternalManagerV1,
    )

    monkeypatch.setattr(provider_module, "qdrant_client_availability", lambda: provider_module.QdrantAvailabilityV1(status="UNAVAILABLE", blocker="qdrant_client_missing"))
    with pytest.raises(provider_module.QdrantNativeMultiObjectUnavailable, match="qdrant_client_missing"):
        QdrantNativeExternalManagerV1(configuration=_config(tmp_path))


def test_dispatch_factory_binds_run_and_does_not_read_task_targets_or_gold(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import (
        build_qdrant_native_manager_factory,
    )
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = FakeQdrantClient()
    factory = build_qdrant_native_manager_factory(
        base_path=tmp_path / "stores",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash="c" * 64,
        runtime_hash="d" * 64,
        provider_factory=lambda *, configuration, runtime_namespace: QdrantNativeMultiObjectProvider(
            configuration=configuration,
            client=client,
            runtime_namespace=runtime_namespace,
        ),
    )

    class Task:
        task_id = "visible-task"

        def __getattribute__(self, name):
            if name in {"target_objects", "target_object_keys", "gold_evidence"}:
                raise AssertionError(f"factory read forbidden task field: {name}")
            return object.__getattribute__(self, name)

    manager = factory(Task(), {"manager_id": "untrusted-cell"})
    assert manager.configuration.run_id == "main-track-visible-task"
    assert manager.identity["configuration_hash"]
    manager.close()


def test_manager_reset_rejects_unsuccessful_typed_provider_result(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import QdrantNativeExternalManagerV1
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectError,
        QdrantNativeMultiObjectProvider,
    )

    class FalseResetProvider(QdrantNativeMultiObjectProvider):
        def reset(self, namespace):
            return NativeResetResultV3(
                namespace=namespace,
                success=False,
                error="namespace_not_empty",
                provenance={"kind": "provider_native", "provider_owned": True},
            )

    manager = QdrantNativeExternalManagerV1(
        configuration=_config(tmp_path),
        provider_factory=lambda *, configuration, runtime_namespace: FalseResetProvider(
            configuration=configuration,
            client=FakeQdrantClient(),
            runtime_namespace=runtime_namespace,
        ),
    )
    with pytest.raises(QdrantNativeMultiObjectError, match="namespace_not_empty"):
        manager.reset(SimpleNamespace(task_id="task-1"))
    manager.close()


def test_manager_export_entries_follows_scroll_continuations_in_provider_order(tmp_path):
    from mub.vnext.external.providers.qdrant_native_adapter import QdrantNativeExternalManagerV1
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = PaginatedFakeQdrantClient()
    manager = QdrantNativeExternalManagerV1(
        configuration=_config(tmp_path),
        provider_factory=lambda *, configuration, runtime_namespace: QdrantNativeMultiObjectProvider(
            configuration=configuration, client=client, runtime_namespace=runtime_namespace
        ),
    )
    manager.ingest(SimpleNamespace(event_id="event-1", sequence_index=0), operation="add", value="Paris", object_key=_key())
    manager.ingest(
        SimpleNamespace(event_id="event-2", sequence_index=1),
        operation="add",
        value="France",
        object_key={"object_type": "profile", "namespace": "default", "entity": "alice", "attribute": "country", "subkey": None},
    )

    entries = manager.export_entries()
    assert [entry["value"] for entry in entries] == ["Paris", "France"]
    assert [entry["rank"] for entry in entries] == [1, 2]
    assert [call[1]["offset"] for call in client.calls if call[0] == "scroll"] == [None, "next"]
    manager.close()


@pytest.mark.parametrize("cursor_mode", ["malformed", "repeated"])
def test_manager_export_entries_fails_closed_on_invalid_scroll_continuation(tmp_path, cursor_mode):
    from mub.vnext.external.providers.qdrant_native_adapter import QdrantNativeExternalManagerV1
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectError,
        QdrantNativeMultiObjectProvider,
    )

    client = PaginatedFakeQdrantClient(cursor_mode=cursor_mode)
    manager = QdrantNativeExternalManagerV1(
        configuration=_config(tmp_path),
        provider_factory=lambda *, configuration, runtime_namespace: QdrantNativeMultiObjectProvider(
            configuration=configuration, client=client, runtime_namespace=runtime_namespace
        ),
    )
    manager.ingest(SimpleNamespace(event_id="event-1", sequence_index=0), operation="add", value="Paris", object_key=_key())
    with pytest.raises(QdrantNativeMultiObjectError, match="scroll continuation"):
        manager.export_entries()
    manager.close()
