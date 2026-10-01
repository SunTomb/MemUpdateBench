from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.native_multi_object import (
    NativeCapabilityDeclarationV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationRequestV3,
    NativeObjectMutationV3,
    NativeRetrievalRequestV3,
)


ROOT = Path(__file__).resolve().parents[2]


def key(namespace: str = "run", entity: str = "alice", attribute: str = "city", subkey: str | None = None):
    return FrozenMemoryObjectKey(
        object_type="profile",
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=subkey,
    )


def provenance():
    return {"kind": NativeDataProvenanceV3.PROVIDER_NATIVE, "provider_owned": True}


def request(operation: Operation, objects: tuple[NativeObjectMutationV3, ...], event_id: str = "event-1"):
    return NativeMultiObjectMutationRequestV3(
        request_id=f"request-{event_id}",
        event_id=event_id,
        sequence_index=1,
        operation=operation,
        objects=objects,
        provenance=provenance(),
    )


@dataclass
class Point:
    id: str
    score: float | None = None
    payload: dict | None = None


class FakeQdrantClient:
    def __init__(self):
        self.points: dict[str, Point] = {}
        self.created = []
        self.deleted_filters = []
        self.queries = []
        self.retrieves = []
        self.upserts = []
        self.deleted_collections = []
        self.events = []
        self.fail_delete_collection = None
        self.fail_close = None
        self.closed = False
        self.collection_schema = {
            "config": {
                "params": {
                    "vectors": {"size": 384, "distance": "Cosine"},
                },
            },
        }

    def collection_exists(self, collection_name):
        self.events.append("collection_exists")
        return bool(self.created)

    def create_collection(self, *, collection_name, vectors_config):
        self.events.append("create_collection")
        self.created.append((collection_name, vectors_config))

    def get_collection(self, *, collection_name):
        self.events.append("get_collection")
        return self.collection_schema

    def delete_collection(self, *, collection_name):
        self.events.append("delete_collection")
        if self.fail_delete_collection is not None:
            raise self.fail_delete_collection
        self.deleted_collections.append(collection_name)

    def upsert(self, *, collection_name, points, wait=True):
        self.events.append("upsert")
        self.upserts.append((collection_name, points, wait))
        for raw in points:
            if isinstance(raw, dict):
                point_id, payload = raw["id"], raw["payload"]
            else:
                point_id, payload = raw.id, raw.payload
            self.points[str(point_id)] = Point(str(point_id), payload=payload)
        return {"operation_id": f"upsert-{len(self.upserts)}"}

    def retrieve(self, *, collection_name, ids, with_payload=True, with_vectors=False):
        self.retrieves.append((collection_name, tuple(ids), with_payload, with_vectors))
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def delete(self, *, collection_name, points_selector, wait=True):
        self.deleted_filters.append(points_selector)
        doomed = []
        for point_id, point in self.points.items():
            if _matches(point.payload or {}, points_selector):
                doomed.append(point_id)
        for point_id in doomed:
            del self.points[point_id]
        return {"operation_id": f"delete-{len(self.deleted_filters)}"}

    def query_points(self, *, collection_name, query, limit, query_filter=None, with_payload=True, with_vectors=False):
        self.queries.append((collection_name, query, limit, query_filter, with_payload, with_vectors))
        points = list(self.points.values())[:limit]
        return type("QueryResult", (), {"points": [Point(point.id, score=1.0 - i * 0.1, payload=point.payload) for i, point in enumerate(points)]})()

    def scroll(self, *, collection_name, scroll_filter=None, limit, with_payload=True, with_vectors=False):
        points = [point for point in self.points.values() if _matches(point.payload or {}, scroll_filter)]
        return points[:limit], None

    def close(self):
        self.events.append("close")
        if self.fail_close is not None:
            raise self.fail_close
        self.closed = True


def _matches(payload, selector):
    if selector is None:
        return True
    if isinstance(selector, dict) and "points" in selector:
        return str(payload.get("object_id")) in {str(value) for value in selector["points"]}
    if isinstance(selector, dict) and "must" in selector:
        return all(_matches_condition(payload, item) for item in selector["must"])
    return True


def _matches_condition(payload, condition):
    key_name = condition.get("key")
    match = condition.get("match", {})
    if "value" in match:
        return payload.get(key_name) == match["value"]
    return True


def _configuration(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        build_qdrant_native_multi_object_configuration,
    )

    return build_qdrant_native_multi_object_configuration(
        run_id="unit-run",
        path=tmp_path / "qdrant-unit-run",
    )


def test_configuration_derives_distinct_run_isolated_storage_paths(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        build_qdrant_native_multi_object_configuration,
    )

    first = build_qdrant_native_multi_object_configuration(
        run_id="run-a", path=tmp_path / "qdrant"
    )
    second = build_qdrant_native_multi_object_configuration(
        run_id="run-b", path=tmp_path / "qdrant"
    )
    assert first.storage_path != second.storage_path
    assert first.storage_path.startswith(str(tmp_path / "qdrant"))
    assert second.storage_path.startswith(str(tmp_path / "qdrant"))


def test_retrieval_combines_runtime_namespace_with_supplied_provider_filter(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path),
        client=FakeQdrantClient(),
        runtime_namespace="runtime-a",
    )
    provider._models = object()
    retrieval = NativeRetrievalRequestV3(
        query_id="q-filter",
        query_text="visible",
        k=1,
        namespace="runtime-a",
        filters={"must": [{"key": "attribute", "match": {"value": "city"}}]},
    )
    provider_filter = provider._provider_filter(retrieval)
    assert provider_filter["must"][0] == {
        "key": "runtime_namespace",
        "match": {"value": "runtime-a"},
    }
    assert provider_filter["must"][1:] == [
        {"key": "attribute", "match": {"value": "city"}}
    ]


def test_model_backed_delete_uses_filter_selector_wrapper(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    class Filter:
        def __init__(self, *, must):
            self.must = must

    class FilterSelector:
        def __init__(self, *, filter):
            self.filter = filter

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )
    provider._models = type("Models", (), {"Filter": Filter, "FilterSelector": FilterSelector})
    provider.delete_scoped("runtime-a", "namespace")
    assert isinstance(client.deleted_filters[-1], FilterSelector)
    assert isinstance(client.deleted_filters[-1].filter, Filter)


def test_qdrant_client_import_is_optional_and_missing_dependency_is_typed(tmp_path, monkeypatch):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectUnavailable,
    )

    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(QdrantNativeMultiObjectUnavailable) as exc_info:
        QdrantNativeMultiObjectProvider(configuration=_configuration(tmp_path))
    assert exc_info.value.status == "UNAVAILABLE"
    assert exc_info.value.blocker == "qdrant_client_missing"


def test_native_provider_executes_crud_with_provider_owned_payload_and_uuid5_ids(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path),
        client=client,
        runtime_namespace="runtime-a",
    )
    first = key(namespace="run", entity="alice", attribute="city")
    added = provider.mutate(
        request(Operation.ADD, (NativeObjectMutationV3(object_key=first, value="Paris"),))
    )
    assert added.outcomes[0].status is ExecutionStatusV3.EXECUTED
    assert added.outcomes[0].provider_entry_id
    assert len(added.outcomes[0].provider_entry_id) == 36
    assert added.outcomes[0].provider_entry_id == provider.point_id("runtime-a", first)
    payload = next(iter(client.points.values())).payload
    assert payload == {
        "runtime_namespace": "runtime-a",
        "object_id": first.canonical_id,
        "namespace": "run",
        "entity": "alice",
        "attribute": "city",
        "subkey": None,
        "object_type": "profile",
        "value": "Paris",
        "content": '"Paris"',
        "source_event_ids": ["event-1"],
        "sequence_index": 1,
        "version": 1,
        "version_metadata": {
            "point_id_derivation": "uuid5-adapter-v1",
            "version_index": 1,
        },
    }

    duplicate = provider.mutate(
        request(Operation.ADD, (NativeObjectMutationV3(object_key=first, value="Lyon"),), event_id="event-2")
    )
    assert duplicate.outcomes[0].status is ExecutionStatusV3.NO_EFFECT
    assert len(client.upserts) == 1

    updated = provider.mutate(
        request(Operation.UPDATE, (NativeObjectMutationV3(object_key=first, value="Lyon"),), event_id="event-3")
    )
    assert updated.outcomes[0].status is ExecutionStatusV3.EXECUTED
    assert updated.outcomes[0].provider_entry_id == added.outcomes[0].provider_entry_id
    assert next(iter(client.points.values())).payload["value"] == "Lyon"

    missing = key(namespace="run", entity="alice", attribute="country")
    no_update = provider.mutate(
        request(Operation.UPDATE, (NativeObjectMutationV3(object_key=missing, value="France"),), event_id="event-4")
    )
    assert no_update.outcomes[0].status is ExecutionStatusV3.NO_EFFECT

    noop = provider.mutate(request(Operation.NOOP, (), event_id="event-5"))
    assert noop.outcomes == ()
    assert noop.atomicity.value == "not_applicable"


def test_retrieval_preserves_provider_order_score_id_payload_and_rejects_mismatch(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectError,
    )

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )
    first = key(namespace="run", entity="alice", attribute="city")
    second = key(namespace="run", entity="alice", attribute="country")
    provider.mutate(request(Operation.ADD, (
        NativeObjectMutationV3(object_key=first, value="Paris"),
        NativeObjectMutationV3(object_key=second, value="France"),
    )))
    trace = provider.retrieve(NativeRetrievalRequestV3(query_id="q-1", query_text="visible", k=2, namespace="runtime-a"))
    assert tuple(entry.provider_entry_id for entry in trace.entries) == tuple(client.points)
    assert tuple(entry.order_index for entry in trace.entries) == (0, 1)
    assert tuple(entry.rank for entry in trace.entries) == (1, 2)
    assert tuple(entry.score for entry in trace.entries) == (1.0, 0.9)
    assert trace.entries[0].provider_payload["value"] == "Paris"
    assert trace.provenance.provider_owned is True
    assert provider.observation_witnesses
    assert any(item.method == "collection_exists" for item in provider.observation_witnesses)
    assert all(len(item.raw_request_hash) == 64 and len(item.raw_response_hash) == 64 for item in provider.observation_witnesses)

    bad_id = next(iter(client.points))
    client.points[bad_id].payload["object_id"] = "run|wrong|attribute|"
    with pytest.raises(QdrantNativeMultiObjectError, match="payload"):
        provider.retrieve(NativeRetrievalRequestV3(query_id="q-2", query_text="visible", k=2, namespace="runtime-a"))

    client.points[bad_id].payload["object_id"] = first.canonical_id
    client.points[bad_id].payload["version_metadata"]["point_id_derivation"] = "not-adapter-derived"
    with pytest.raises(QdrantNativeMultiObjectError, match="version metadata"):
        provider.retrieve(NativeRetrievalRequestV3(query_id="q-3", query_text="visible", k=2, namespace="runtime-a"))


def test_scoped_delete_and_reset_use_provider_filters_and_close_client(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )
    first = key(namespace="run", entity="alice", attribute="city")
    second = key(namespace="run", entity="alice", attribute="country")
    provider.mutate(request(Operation.ADD, (
        NativeObjectMutationV3(object_key=first, value="Paris"),
        NativeObjectMutationV3(object_key=second, value="France"),
    )))
    provider.delete_scoped("runtime-a", "attribute", first)
    assert len(client.points) == 1
    assert client.deleted_filters[-1]["must"]

    provider.mutate(request(Operation.ADD, (NativeObjectMutationV3(object_key=first, value="Paris"),), event_id="event-2"))
    reset = provider.reset("runtime-a")
    assert reset.success is True
    assert client.points == {}
    assert provider.capabilities() == NativeCapabilityDeclarationV3(
        direct_provider_crud=True,
        provider_owned_collection=True,
        supports_multi_object_mutation=True,
        supports_multi_object_retrieval=True,
        supports_add=True,
        supports_update=True,
        supports_noop=True,
        supports_delete=True,
        supports_atomic_mutation=False,
        supports_isolated_reset=True,
        exports_provider_entry_ids=True,
        exports_provider_payloads=True,
        exports_provider_order=True,
        exports_provider_scores=True,
        exports_source_event_linkage=True,
        visible_only=True,
    )
    provider.close()
    provider.close()
    assert client.closed is True


def test_runtime_verification_requires_observed_provider_calls(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path),
        client=FakeQdrantClient(),
        runtime_namespace="runtime-a",
    )
    verification = provider.verify_api_runtime("a" * 64)
    assert verification.status == "BLOCKED"
    assert verification.direct_observed is False
    assert verification.blocker == "qdrant_api_runtime_not_observed"


def test_runtime_verification_ignores_failed_provider_calls(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectError,
        QdrantNativeMultiObjectProvider,
    )

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )

    def fail_query(**kwargs):
        raise RuntimeError("query unavailable")

    client.query_points = fail_query
    with pytest.raises(QdrantNativeMultiObjectError):
        provider.retrieve(
            NativeRetrievalRequestV3(
                query_id="q-failed",
                query_text="visible",
                k=1,
                namespace="runtime-a",
            )
        )
    verification = provider.verify_api_runtime("a" * 64)
    assert verification.blocker == "qdrant_api_runtime_not_observed"
    assert "query_points" in verification.missing_methods


def test_constructor_rejects_existing_collection_with_incompatible_schema(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectUnavailable,
    )

    client = FakeQdrantClient()
    client.created = True
    client.collection_schema = {
        "config": {"params": {"vectors": {"size": 383, "distance": "Cosine"}}}
    }
    with pytest.raises(QdrantNativeMultiObjectUnavailable, match="schema") as exc_info:
        QdrantNativeMultiObjectProvider(
            configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
        )
    assert exc_info.value.blocker == "qdrant_collection_schema_mismatch"
    assert client.upserts == []


def test_constructor_closes_client_but_preserves_existing_collection_on_schema_failure(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectUnavailable,
    )

    config = _configuration(tmp_path)
    client = FakeQdrantClient()
    client.created = True
    client.collection_schema = {
        "config": {"params": {"vectors": {"size": 383, "distance": "Cosine"}}}
    }
    with pytest.raises(QdrantNativeMultiObjectUnavailable) as exc_info:
        QdrantNativeMultiObjectProvider(
            configuration=config, client=client, runtime_namespace="runtime-a"
        )
    assert exc_info.value.blocker == "qdrant_collection_schema_mismatch"
    assert client.deleted_collections == []
    assert client.closed is True
    assert client.events[-1] == "close"


def test_constructor_deletes_new_collection_before_closing_on_schema_failure(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectUnavailable,
    )

    config = _configuration(tmp_path)
    client = FakeQdrantClient()
    client.collection_schema = {
        "config": {"params": {"vectors": {"size": 383, "distance": "Cosine"}}}
    }
    with pytest.raises(QdrantNativeMultiObjectUnavailable) as exc_info:
        QdrantNativeMultiObjectProvider(
            configuration=config, client=client, runtime_namespace="runtime-a"
        )
    assert exc_info.value.blocker == "qdrant_collection_schema_mismatch"
    assert client.deleted_collections == [config.collection_name]
    assert client.closed is True
    assert client.events[-2:] == ["delete_collection", "close"]


def test_constructor_cleanup_errors_do_not_mask_primary_setup_error(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
        QdrantNativeMultiObjectUnavailable,
    )

    client = FakeQdrantClient()
    client.collection_schema = {
        "config": {"params": {"vectors": {"size": 383, "distance": "Cosine"}}}
    }
    client.fail_delete_collection = RuntimeError("delete cleanup exploded")
    client.fail_close = RuntimeError("close cleanup exploded")
    with pytest.raises(QdrantNativeMultiObjectUnavailable) as exc_info:
        QdrantNativeMultiObjectProvider(
            configuration=_configuration(tmp_path),
            client=client,
            runtime_namespace="runtime-a",
        )
    assert exc_info.value.blocker == "qdrant_collection_schema_mismatch"
    notes = "\n".join(getattr(exc_info.value, "__notes__", ()))
    assert "delete_collection" in notes
    assert "close" in notes
    assert client.events[-2:] == ["delete_collection", "close"]


def test_constructor_cleanup_records_fallback_when_primary_has_no_add_note(tmp_path, monkeypatch):
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    class NoNotesError(RuntimeError):
        add_note = None

    def fail_setup(self):
        raise NoNotesError("primary setup exploded")

    monkeypatch.setattr(QdrantNativeMultiObjectProvider, "_ensure_collection", fail_setup)
    client = FakeQdrantClient()
    client.fail_close = RuntimeError("close cleanup exploded")
    with pytest.raises(NoNotesError) as exc_info:
        QdrantNativeMultiObjectProvider(
            configuration=_configuration(tmp_path),
            client=client,
            runtime_namespace="runtime-a",
        )
    assert exc_info.value.cleanup_details == (
        "Qdrant provider constructor cleanup close failed: RuntimeError: close cleanup exploded",
    )
    assert client.events == ["close"]


def test_qdrant_client_construction_failure_preserves_typed_unavailable_error(tmp_path, monkeypatch):
    import mub.vnext.external.providers.qdrant_native_multi_object as module

    class FailingModule:
        models = None

        class QdrantClient:
            def __new__(cls, **kwargs):
                raise RuntimeError("client construction exploded")

    monkeypatch.setattr(module, "_load_qdrant_client_module", lambda: FailingModule)
    with pytest.raises(module.QdrantNativeMultiObjectUnavailable) as exc_info:
        module.QdrantNativeMultiObjectProvider(configuration=_configuration(tmp_path))
    assert exc_info.value.blocker == "qdrant_client_runtime_unavailable"


def test_constructor_admits_existing_collection_with_compatible_schema(tmp_path):
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider

    client = FakeQdrantClient()
    client.created = True
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )
    assert provider.capabilities().direct_provider_crud is True


    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path),
        client=FakeQdrantClient(),
        runtime_namespace="runtime-a",
    )
    provider._client.get_collection = None
    assert provider._verify_collection_schema() is False


@pytest.mark.parametrize(
    "schema",
    (
        {},
        {"config": {}},
        {"config": {"params": {}}},
        {"config": {"params": {"vectors": {}}}},
        {"config": {"params": {"vectors": {"size": 383, "distance": "Cosine"}}}},
        {"config": {"params": {"vectors": {"size": 384, "distance": "Euclid"}}}},
    ),
)
def test_collection_schema_verification_fails_closed_for_missing_or_mismatched_vector_metadata(
    tmp_path, schema
):
    from mub.vnext.external.providers.qdrant_native_multi_object import (
        QdrantNativeMultiObjectProvider,
    )

    client = FakeQdrantClient()
    provider = QdrantNativeMultiObjectProvider(
        configuration=_configuration(tmp_path), client=client, runtime_namespace="runtime-a"
    )
    client.collection_schema = schema
    assert provider._verify_collection_schema() is False


def test_qdrant_client_availability_converts_unexpected_import_exception_to_typed_unavailable(
    monkeypatch,
):
    import mub.vnext.external.providers.qdrant_native_multi_object as module

    def fail_import(name):
        raise RuntimeError("broken qdrant import")

    monkeypatch.setattr(module.importlib, "import_module", fail_import)
    result = module.qdrant_client_availability()
    assert result.status == "UNAVAILABLE"
    assert result.blocker == "qdrant_client_import_failed"


    from scripts.vnext_qualify_qdrant_native_multi_object import (
        QdrantQualificationStatusV1,
        qualify_qdrant_native_multi_object,
    )

    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    report = qualify_qdrant_native_multi_object(
        run_id="unit-run",
        path="C:/tmp/qdrant-unit-run",
        direct_observed=False,
        evidence_anchor=None,
    )
    assert report.status is QdrantQualificationStatusV1.BLOCKED
    assert report.blocker == "qdrant_client_missing"
    assert report.scientific_evidence is False




def test_qualification_rejects_caller_supplied_direct_observation(monkeypatch):
    from scripts.vnext_qualify_qdrant_native_multi_object import qualify_qdrant_native_multi_object

    class FakeModule:
        __version__ = "1.19.0"

    monkeypatch.setitem(sys.modules, "qdrant_client", FakeModule())
    report = qualify_qdrant_native_multi_object(
        run_id="unit-run",
        path="C:/tmp/qdrant-unit-run",
        direct_observed=True,
        evidence_anchor="a" * 64,
    )
    assert report.status.value == "BLOCKED"
    assert report.direct_observed is False
    assert report.blocker == "api_runtime_not_observed"
    assert report.evidence_anchor is None


def test_qualification_never_returns_ready_without_direct_evidence_or_anchor(monkeypatch):
    from scripts.vnext_qualify_qdrant_native_multi_object import qualify_qdrant_native_multi_object

    class FakeModule:
        pass

    monkeypatch.setitem(sys.modules, "qdrant_client", FakeModule())
    report = qualify_qdrant_native_multi_object(
        run_id="unit-run",
        path="C:/tmp/qdrant-unit-run",
        direct_observed=False,
        evidence_anchor="",
    )
    assert report.status.value == "BLOCKED"
    assert report.blocker == "api_runtime_not_observed"
    assert report.evidence_anchor is None
