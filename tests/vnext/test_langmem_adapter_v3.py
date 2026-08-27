from __future__ import annotations

from mub.vnext.contracts.enums import AnswerSchema, EvaluationMode, EventRole
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryEventV3, MemoryQueryV3
from mub.vnext.external.bridge import WorkerRequestV1
from mub.vnext.external.providers.langmem import build_langmem_adapter_configuration
from mub.vnext.external.workers.langmem_worker import (
    LangMemProfileBackendV1,
    LangMemWorkerServiceV1,
)

from tests.vnext.test_langmem_worker_protocol import DirectStoreFake


class InProcessBridge:
    def __init__(self, service: LangMemWorkerServiceV1) -> None:
        self.service = service
        self.requests: list[WorkerRequestV1] = []
        self.closed = False

    def request(self, request: WorkerRequestV1):
        self.requests.append(request)
        return self.service.handle(request)

    def close(self) -> None:
        self.closed = True


def _key() -> FrozenMemoryObjectKey:
    return FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
        subkey=None,
    )


def _event(event_id: str, index: int, verb: str, value: str) -> MemoryEventV3:
    text = f'{verb} default|alice|city| with value "{value}".'
    return MemoryEventV3(
        event_id=event_id,
        sequence_index=index,
        timestamp="2026-08-26T00:00:00Z",
        raw_text=text,
        normalized_text=text,
        speaker="user",
        role=EventRole.LATEST_GOLD,
        gold_action_ids=("never-visible-gold",),
        metadata={"hidden": "never-visible"},
    )


def _query() -> MemoryQueryV3:
    return MemoryQueryV3(
        query_id="query-1",
        query_type=QueryTypeV3.CURRENT,
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.STATE_DIRECT,
    )


def _adapter():
    from mub.vnext.external.providers.langmem_adapter import LangMemExternalAdapterV3

    configuration = build_langmem_adapter_configuration(run_id="langmem-adapter-test")
    bridge = InProcessBridge(
        LangMemWorkerServiceV1(
            LangMemProfileBackendV1(
                store=DirectStoreFake(), configuration=configuration
            )
        )
    )
    return (
        LangMemExternalAdapterV3(
            bridge=bridge,
            configuration=configuration,
            target_objects=(_key(),),
        ),
        bridge,
    )


def test_langmem_adapter_reports_profile_lifecycle_and_value_exports() -> None:
    adapter, bridge = _adapter()
    adapter.reset(ResetRequestV3(namespace="langmem-adapter-namespace"))
    add = adapter.ingest_event(_event("event-add", 0, "Add", "Paris"))
    update = adapter.ingest_event(_event("event-update", 1, "Update", "Lyon"))

    assert add.execution_status.value == "executed"
    assert update.execution_status.value == "executed"
    assert add.affected_entry_ids == update.affected_entry_ids
    assert adapter.capabilities().supports_add is True
    assert adapter.capabilities().supports_update is True
    assert adapter.capabilities().supports_noop is True
    assert adapter.capabilities().supports_delete is True
    assert adapter.capabilities().exports_values is True
    assert adapter.capabilities().supports_multi_object_query is False
    entry = adapter.export_entries().entries[0]
    assert entry.value_candidate == "Lyon"
    assert entry.object_key_candidate == _key()
    assert adapter.answer(_query(), "slot_direct").prediction.parsed_answer == "Lyon"
    request_payloads = str([request.payload for request in bridge.requests])
    assert "never-visible-gold" not in request_payloads
    assert "never-visible" not in request_payloads


def test_langmem_adapter_reset_probe_uses_fresh_isolated_namespaces() -> None:
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    adapter, _ = _adapter()
    probe = run_namespace_reset_probe(
        adapter,
        candidate_id="langmem_0_0_30_profile",
        run_prefix="langmem-reset",
    )

    assert probe.passed is True
    assert len(probe.trials) == 20


def test_langmem_provider_facade_exposes_optional_sdk_free_adapter() -> None:
    from mub.vnext.external.providers import LangMemExternalAdapterV3

    assert LangMemExternalAdapterV3.__name__ == "LangMemExternalAdapterV3"
