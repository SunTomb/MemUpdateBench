from __future__ import annotations

from mub.vnext.external.bridge import WorkerOperation, WorkerRequestV1
from mub.vnext.external.providers.langmem import (
    build_langmem_adapter_configuration,
)
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1
from mub.vnext.external.workers.langmem_worker import (
    LangMemProfileBackendV1,
    LangMemWorkerServiceV1,
)


class DirectStoreFake:
    def __init__(self) -> None:
        self.values: dict[tuple[tuple[str, ...], str], dict] = {}
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    def get(self, namespace: tuple[str, ...], key: str):
        self.calls.append(("get", namespace, key))
        value = self.values.get((namespace, key))
        return None if value is None else _Item(key, value)

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None:
        self.calls.append(("put", namespace, key))
        self.values[(namespace, key)] = value

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        self.calls.append(("delete", namespace, key))
        self.values.pop((namespace, key), None)

    def search(self, namespace: tuple[str, ...], *, limit: int):
        self.calls.append(("search", namespace, ""))
        return tuple(
            _Item(key, value)
            for (row_namespace, key), value in self.values.items()
            if row_namespace == namespace
        )[:limit]


class _Item:
    def __init__(self, key: str, value: dict) -> None:
        self.key = key
        self.value = value
        self.created_at = None
        self.updated_at = None
        self.score = None


def _event(
    event_id: str,
    sequence_index: int,
    raw_text: str,
) -> ProviderEventInputV1:
    return ProviderEventInputV1(
        event_id=event_id,
        sequence_index=sequence_index,
        logical_time="2026-08-26T00:00:00Z",
        raw_text=raw_text,
        runtime_namespace="langmem-profile-trial",
    )


def _service_and_store():
    configuration = build_langmem_adapter_configuration(
        run_id="langmem-worker-test",
    )
    store = DirectStoreFake()
    backend = LangMemProfileBackendV1(
        store=store,
        configuration=configuration,
    )
    return LangMemWorkerServiceV1(backend), store


def _ingest(service: LangMemWorkerServiceV1, event: ProviderEventInputV1):
    return service.handle(
        WorkerRequestV1(
            request_id=f"request-{event.event_id}",
            operation=WorkerOperation.INGEST_EVENT,
            payload={"event": event.model_dump(mode="json")},
        )
    )


def test_profile_backend_uses_native_store_create_update_delete_with_stable_id() -> None:
    service, store = _service_and_store()
    add = _ingest(
        service,
        _event(
            "event-add",
            0,
            'Add default|alice|city| with value "Paris".',
        ),
    )
    update = _ingest(
        service,
        _event(
            "event-update",
            1,
            'Update default|alice|city| with value "Lyon".',
        ),
    )
    delete = _ingest(
        service,
        _event(
            "event-delete",
            2,
            "Delete default|alice|city| [scope=object; "
            "enumerated_targets=default|alice|city|; "
            "event_logical_time=2026-08-26T00:00:00Z; "
            "effective_at=2026-08-26T00:00:00Z].",
        ),
    )

    assert add.status.value == "ok"
    assert update.status.value == "ok"
    assert delete.status.value == "ok"
    assert add.payload["entry_id"] == update.payload["entry_id"]
    assert add.payload["effective_operation"] == "add"
    assert update.payload["effective_operation"] == "update"
    assert delete.payload["effective_operation"] == "delete"
    assert [name for name, _, _ in store.calls] == [
        "get",
        "search",
        "put",
        "get",
        "put",
        "get",
        "delete",
    ]


def test_profile_backend_noop_does_not_touch_native_store() -> None:
    service, store = _service_and_store()

    response = _ingest(
        service,
        _event("event-noop", 0, "No memory object changes."),
    )

    assert response.status.value == "ok"
    assert response.payload["effective_operation"] == "noop"
    assert store.calls == []


def test_profile_backend_search_uses_native_store_and_deterministic_ranking() -> None:
    service, store = _service_and_store()
    _ingest(
        service,
        _event(
            "event-add",
            0,
            'Add default|alice|city| with value "Paris".',
        ),
    )
    query = ProviderQueryInputV1(
        query_id="query-1",
        query_text="Where does Alice live?",
        k=3,
        runtime_namespace="langmem-profile-trial",
    )

    response = service.handle(
        WorkerRequestV1(
            request_id="request-search",
            operation=WorkerOperation.RETRIEVE,
            payload={"query": query.model_dump(mode="json")},
        )
    )

    assert response.status.value == "ok"
    assert response.payload["entries"][0]["value"] == "Paris"
    assert store.calls[-1][0] == "search"
