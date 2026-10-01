from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.native_multi_object import (
    NativeAtomicityV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationRequestV3,
    NativeMultiObjectMutationResultV3,
    NativeObjectMutationV3,
    NativeObjectOutcomeV3,
    NativeResetResultV3,
    NativeRetrievalEntryV3,
    NativeRetrievalRequestV3,
    NativeRetrievalTraceV3,
)
from mub.vnext.external.native_multi_object import (
    NativeMultiObjectAdapter,
    NativeMultiObjectAdapterError,
)


def key(namespace: str, entity: str, attribute: str, subkey: str | None = None):
    return FrozenMemoryObjectKey(
        object_type="slot", namespace=namespace, entity=entity,
        attribute=attribute, subkey=subkey,
    )


def native_provenance():
    return {"kind": NativeDataProvenanceV3.PROVIDER_NATIVE, "provider_owned": True}


def request(operation: Operation, event_id: str, objects: tuple[NativeObjectMutationV3, ...]):
    return NativeMultiObjectMutationRequestV3(
        request_id=f"r-{event_id}", event_id=event_id, sequence_index=int(event_id[-1]),
        operation=operation, objects=objects, provenance=native_provenance(),
    )


@dataclass
class FakeNativeProvider:
    states: dict[str, dict[str, tuple[object, str]]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    closed: bool = False

    def mutate(self, mutation: NativeMultiObjectMutationRequestV3) -> NativeMultiObjectMutationResultV3:
        self.calls.append(f"mutate:{mutation.event_id}")
        namespace = "shared"
        state = self.states.setdefault(namespace, {})
        outcomes = []
        for item in mutation.objects:
            object_id = item.object_key.canonical_id
            if mutation.operation is Operation.ADD:
                if object_id in state:
                    outcomes.append(NativeObjectOutcomeV3(
                        event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                        status=ExecutionStatusV3.NO_EFFECT, source_event_ids=(mutation.event_id,),
                        write_count=0, reason="already_present", provenance=native_provenance(),
                    ))
                else:
                    provider_id = f"p-{len(state)}"
                    state[object_id] = (item.value, provider_id)
                    outcomes.append(NativeObjectOutcomeV3(
                        event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                        status=ExecutionStatusV3.EXECUTED, provider_entry_id=provider_id,
                        source_event_ids=(mutation.event_id,), write_count=1, provenance=native_provenance(),
                    ))
            elif mutation.operation is Operation.UPDATE:
                if object_id not in state:
                    outcomes.append(NativeObjectOutcomeV3(
                        event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                        status=ExecutionStatusV3.NO_EFFECT, source_event_ids=(mutation.event_id,),
                        write_count=0, reason="absent_update", provenance=native_provenance(),
                    ))
                else:
                    _, provider_id = state[object_id]
                    state[object_id] = (item.value, provider_id)
                    outcomes.append(NativeObjectOutcomeV3(
                        event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                        status=ExecutionStatusV3.EXECUTED, provider_entry_id=provider_id,
                        source_event_ids=(mutation.event_id,), write_count=1, provenance=native_provenance(),
                    ))
            elif mutation.operation is Operation.DELETE:
                existing = state.pop(object_id, None)
                outcomes.append(NativeObjectOutcomeV3(
                    event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                    status=ExecutionStatusV3.EXECUTED if existing else ExecutionStatusV3.NO_EFFECT,
                    provider_entry_id=None if existing is None else existing[1],
                    source_event_ids=(mutation.event_id,), write_count=1 if existing else 0,
                    reason=None if existing else "absent_delete", provenance=native_provenance(),
                ))
            else:
                outcomes.append(NativeObjectOutcomeV3(
                    event_id=mutation.event_id, object_key=item.object_key, operation=mutation.operation,
                    status=ExecutionStatusV3.EXECUTED, source_event_ids=(mutation.event_id,),
                    write_count=0, provenance=native_provenance(),
                ))
        statuses = {item.status for item in outcomes}
        atomicity = NativeAtomicityV3.PARTIAL if len(statuses) > 1 else NativeAtomicityV3.ATOMIC
        return NativeMultiObjectMutationResultV3(
            request_id=mutation.request_id, event_id=mutation.event_id, operation=mutation.operation,
            outcomes=tuple(outcomes), atomicity=atomicity, provenance=native_provenance(),
        )

    def retrieve(self, query: NativeRetrievalRequestV3) -> NativeRetrievalTraceV3:
        self.calls.append(f"retrieve:{query.query_id}")
        entries = []
        for index, (object_id, (value, provider_id)) in enumerate(self.states.get("shared", {}).items()):
            object_key = next(key for key in ALL_KEYS if key.canonical_id == object_id)
            entries.append(NativeRetrievalEntryV3(
                provider_entry_id=provider_id, object_key=object_key,
                provider_payload={"value": value}, rank=index + 1, order_index=index,
                score=float(1.0 / (index + 1)), source_event_ids=("event-1",),
                provenance=native_provenance(),
            ))
        return NativeRetrievalTraceV3(query=query, entries=tuple(entries[:query.k]), provenance=native_provenance())

    def reset(self, namespace: str) -> NativeResetResultV3:
        self.calls.append(f"reset:{namespace}")
        self.states.pop("shared", None)
        return NativeResetResultV3(
            namespace=namespace,
            success=True,
            provenance=native_provenance(),
        )

    def close(self) -> None:
        self.closed = True


FIRST = key("ns-a", "entity", "attribute", "one")
SECOND = key("ns-b", "entity", "attribute", "two")
ALL_KEYS = (FIRST, SECOND)




def test_adapter_preserves_omitted_delete_value_through_revalidation():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    result = adapter.mutate(request(Operation.DELETE, "event-1", (
        NativeObjectMutationV3(object_key=FIRST),
    )))
    assert result.outcomes[0].status is ExecutionStatusV3.NO_EFFECT
    assert provider.calls == ["mutate:event-1"]


def test_adapter_forwards_ordered_mutations_and_native_retrieval_without_local_store():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    result = adapter.mutate(request(Operation.ADD, "event-1", (
        NativeObjectMutationV3(object_key=FIRST, value="v1"),
        NativeObjectMutationV3(object_key=SECOND, value="v2"),
    )))
    assert tuple(item.object_key for item in result.outcomes) == ALL_KEYS
    trace = adapter.retrieve(NativeRetrievalRequestV3(query_id="q", query_text="visible", k=2, namespace="run"))
    assert tuple(entry.provider_entry_id for entry in trace.entries) == ("p-0", "p-1")
    assert tuple(entry.order_index for entry in trace.entries) == (0, 1)
    assert "_state" not in adapter.__dict__
    assert provider.calls == ["mutate:event-1", "retrieve:q"]


def test_adapter_rejects_mutation_outside_declared_target_scope():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=(FIRST,))
    with pytest.raises(NativeMultiObjectAdapterError, match="outside adapter target scope"):
        adapter.mutate(request(Operation.ADD, "event-2", (
            NativeObjectMutationV3(object_key=SECOND, value="v2"),
        )))
    assert provider.calls == []


def test_adapter_preserves_provider_order_and_rejects_hidden_target_or_local_reranking():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    with pytest.raises(NativeMultiObjectAdapterError, match="forbidden key"):
        adapter.retrieve({"query_id": "q", "query_text": "visible", "k": 1, "namespace": "run", "target_object_keys": (FIRST,)})

    class RerankingProvider(FakeNativeProvider):
        native_boundary = {"local_reranking": True}

    with pytest.raises(NativeMultiObjectAdapterError, match="local reranking"):
        NativeMultiObjectAdapter(provider=RerankingProvider(), target_objects=ALL_KEYS)


def test_adapter_reset_isolation_and_deterministic_repeats():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    add = request(Operation.ADD, "event-1", (NativeObjectMutationV3(object_key=FIRST, value="v1"),))
    adapter.mutate(add)
    first = adapter.retrieve(NativeRetrievalRequestV3(query_id="q", query_text="visible", k=2, namespace="run"))
    adapter.reset("run")
    empty = adapter.retrieve(NativeRetrievalRequestV3(query_id="q2", query_text="visible", k=2, namespace="run"))
    assert len(first.entries) == 1
    assert empty.entries == ()

    adapter.mutate(add)
    second = adapter.retrieve(NativeRetrievalRequestV3(query_id="q", query_text="visible", k=2, namespace="run"))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_adapter_rejects_provider_result_with_unbound_object_key_or_wrong_request():
    class BadProvider(FakeNativeProvider):
        def mutate(self, mutation):
            result = super().mutate(mutation)
            return result.model_copy(update={"request_id": "wrong"})

    adapter = NativeMultiObjectAdapter(provider=BadProvider(), target_objects=ALL_KEYS)
    with pytest.raises(NativeMultiObjectAdapterError, match="request ID"):
        adapter.mutate(request(Operation.NOOP, "event-1", ()))


def test_close_is_forwarded_and_calls_after_close_are_rejected():
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    adapter.close()
    assert provider.closed is True
    with pytest.raises(NativeMultiObjectAdapterError, match="closed"):
        adapter.retrieve(NativeRetrievalRequestV3(query_id="q", query_text="visible", k=1, namespace="run"))


def test_reset_provider_none_raises_adapter_error():
    class NoneResetProvider(FakeNativeProvider):
        def reset(self, namespace: str) -> None:
            return None

    adapter = NativeMultiObjectAdapter(provider=NoneResetProvider(), target_objects=ALL_KEYS)
    with pytest.raises(NativeMultiObjectAdapterError, match="returned None"):
        adapter.reset("run")


@pytest.mark.parametrize("forbidden_key", [
    "target_object_keys", "target_objects", "target_ids", "gold_answer", "gold_value", "answer",
    "expected_answer", "expected_value", "selector", "hidden_selector", "hidden_target",
    "raw_prompt", "raw_output", "reasoning",
])
def test_retrieve_rejects_forbidden_keys_recursively(forbidden_key: str):
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)

    with pytest.raises(NativeMultiObjectAdapterError, match="forbidden"):
        adapter.retrieve({
            "query_id": "q", "query_text": "visible", "k": 1, "namespace": "run",
            "options": {forbidden_key: "value"},
        })

    with pytest.raises(NativeMultiObjectAdapterError, match="forbidden"):
        adapter.retrieve({
            "query_id": "q", "query_text": "visible", "k": 1, "namespace": "run",
            "filters": {"nested": {forbidden_key: "value"}},
        })


@pytest.mark.parametrize("forbidden_value", [
    "http://localhost:8000", "http://127.0.0.1:6333", "file:///etc/passwd",
    "s3://bucket/key", "http://admin:pass@example.com", "user_token=secret_123", "Bearer abcd12345",
    "https://public.example/a?api-key=secret_123", "https://public.example/a?bearer=secret_123",
])
def test_retrieve_recursively_rejects_forbidden_urls_and_credentials(forbidden_value: str):
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)

    with pytest.raises(NativeMultiObjectAdapterError, match="forbidden"):
        adapter.retrieve({
            "query_id": "q", "query_text": "visible", "k": 1, "namespace": "run",
            "options": {"endpoint": forbidden_value},
        })


@pytest.mark.parametrize("allowed_value", [
    "https://example.com/search?q=query",
    "http://public.org/dataset/info",
    "What is the capital according to https://wikipedia.org/wiki/France?",
    "ordinary text query without any URL",
])
def test_retrieve_allows_ordinary_public_urls_and_text(allowed_value: str):
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)

    trace = adapter.retrieve({
        "query_id": "q-public",
        "query_text": allowed_value,
        "k": 1,
        "namespace": "run",
    })
    assert trace.query.query_text == allowed_value


@pytest.mark.parametrize("allowed_query", [
    "What is the capital of France?",
    "How does HTTP/1.1 work?",
    "Tell me about https protocol",
    "entity: user, attribute: location",
])
def test_retrieve_allows_ordinary_text_and_public_queries(allowed_query: str):
    provider = FakeNativeProvider()
    adapter = NativeMultiObjectAdapter(provider=provider, target_objects=ALL_KEYS)
    trace = adapter.retrieve({
        "query_id": "q", "query_text": allowed_query, "k": 1, "namespace": "run",
    })
    assert trace.query.query_text == allowed_query


def test_retrieve_recursively_rejects_forbidden_metadata_in_provider_payload():
    class ForbiddenPayloadProvider(FakeNativeProvider):
        def retrieve(self, query: NativeRetrievalRequestV3) -> NativeRetrievalTraceV3:
            trace = super().retrieve(query)
            bad_entry = NativeRetrievalEntryV3(
                provider_entry_id="p-bad", object_key=FIRST,
                provider_payload={"nested": {"gold_evidence": "leaked"}}, rank=1, order_index=0,
                score=1.0, source_event_ids=("event-1",), provenance=native_provenance(),
            )
            return NativeRetrievalTraceV3(query=query, entries=(bad_entry,), provenance=native_provenance())

    adapter = NativeMultiObjectAdapter(provider=ForbiddenPayloadProvider(), target_objects=ALL_KEYS)
    with pytest.raises(NativeMultiObjectAdapterError, match="forbidden"):
        adapter.retrieve(NativeRetrievalRequestV3(query_id="q", query_text="visible", k=1, namespace="run"))
