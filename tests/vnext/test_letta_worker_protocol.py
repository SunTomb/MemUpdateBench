from __future__ import annotations

from mub.vnext.external.bridge import WorkerOperation, WorkerRequestV1
from mub.vnext.external.providers.letta import build_letta_adapter_configuration
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1
from mub.vnext.external.workers.letta_worker import (
    LettaBlockProfileBackendV1,
    LettaWorkerServiceV1,
)


class DirectBlockClientFake:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], dict] = {}
        self.calls: list[tuple[str, str, str]] = []

    def get_block(self, namespace: str, block_id: str):
        self.calls.append(("get", namespace, block_id))
        return self.values.get((namespace, block_id))

    def create_block(self, namespace: str, block_id: str, value: dict) -> None:
        self.calls.append(("create", namespace, block_id))
        self.values[(namespace, block_id)] = value

    def update_block(self, namespace: str, block_id: str, value: dict) -> None:
        self.calls.append(("update", namespace, block_id))
        self.values[(namespace, block_id)] = value

    def delete_block(self, namespace: str, block_id: str) -> None:
        self.calls.append(("delete", namespace, block_id))
        self.values.pop((namespace, block_id), None)

    def search_blocks(self, namespace: str):
        self.calls.append(("search", namespace, ""))
        return tuple(
            (block_id, value)
            for (row_namespace, block_id), value in self.values.items()
            if row_namespace == namespace
        )


def _event(event_id: str, index: int, text: str, namespace: str = "letta-profile-trial") -> ProviderEventInputV1:
    return ProviderEventInputV1(
        event_id=event_id,
        sequence_index=index,
        logical_time="2026-08-27T00:00:00Z",
        raw_text=text,
        runtime_namespace=namespace,
    )


def _service_and_client():
    configuration = build_letta_adapter_configuration(run_id="letta-worker-test")
    client = DirectBlockClientFake()
    return LettaWorkerServiceV1(
        LettaBlockProfileBackendV1(client=client, configuration=configuration)
    ), client


def _ingest(service: LettaWorkerServiceV1, event: ProviderEventInputV1):
    return service.handle(
        WorkerRequestV1(
            request_id=f"request-{event.event_id}",
            operation=WorkerOperation.INGEST_EVENT,
            payload={"event": event.model_dump(mode="json")},
        )
    )


def test_block_profile_uses_direct_create_update_delete_with_stable_id() -> None:
    service, client = _service_and_client()
    add = _ingest(service, _event("add", 0, 'Add default|alice|city| with value "Paris".'))
    update = _ingest(service, _event("update", 1, 'Update default|alice|city| with value "Lyon".'))
    delete = _ingest(
        service,
        _event(
            "delete", 2,
            "Delete default|alice|city| [scope=object; "
            "enumerated_targets=default|alice|city|; "
            "event_logical_time=2026-08-27T00:00:00Z; "
            "effective_at=2026-08-27T00:00:00Z].",
        ),
    )

    assert add.status.value == update.status.value == delete.status.value == "ok"
    assert add.payload["entry_id"] == update.payload["entry_id"] == delete.payload["entry_id"]
    assert [call[0] for call in client.calls] == ["get", "create", "get", "update", "get", "delete"]


def test_block_profile_noop_touches_no_client_and_namespaces_are_isolated() -> None:
    service, client = _service_and_client()

    noop = _ingest(service, _event("noop", 0, "No memory object changes."))
    _ingest(service, _event("a", 1, 'Add default|alice|city| with value "Paris".', "namespace-a"))
    query = ProviderQueryInputV1(
        query_id="query-b", query_text="Alice city", k=1, runtime_namespace="namespace-b"
    )
    result = service.handle(
        WorkerRequestV1(
            request_id="query-b", operation=WorkerOperation.RETRIEVE,
            payload={"query": query.model_dump(mode="json")},
        )
    )

    assert noop.payload["effective_operation"] == "noop"
    assert result.payload["entries"] == ()
    assert client.calls[-1] == ("search", "memupdatebench/namespace-b", "")
