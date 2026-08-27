from __future__ import annotations

from mub.vnext.contracts.enums import AnswerSchema, EvaluationMode, EventRole
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryEventV3, MemoryQueryV3
from mub.vnext.external.bridge import WorkerRequestV1
from mub.vnext.external.providers.letta import build_letta_adapter_configuration
from mub.vnext.external.workers.letta_worker import LettaBlockProfileBackendV1, LettaWorkerServiceV1
from tests.vnext.test_letta_worker_protocol import DirectBlockClientFake


class InProcessBridge:
    def __init__(self, service: LettaWorkerServiceV1) -> None:
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
        object_type="profile", namespace="default", entity="alice", attribute="city", subkey=None
    )


def _event(event_id: str, index: int, text: str) -> MemoryEventV3:
    return MemoryEventV3(
        event_id=event_id, sequence_index=index, timestamp="2026-08-27T00:00:00Z",
        raw_text=text, normalized_text=text, speaker="user", role=EventRole.LATEST_GOLD,
        gold_action_ids=("hidden-action",), metadata={"secret": "hidden"},
    )


def _query() -> MemoryQueryV3:
    return MemoryQueryV3(
        query_id="letta-query", query_type=QueryTypeV3.CURRENT, text="Where does Alice live?",
        selector=CurrentSelector(), target_object_keys=(_key(),), answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.STATE_DIRECT,
    )


def _adapter():
    from mub.vnext.external.providers.letta_adapter import LettaExternalAdapterV3

    configuration = build_letta_adapter_configuration(run_id="letta-adapter-test")
    bridge = InProcessBridge(LettaWorkerServiceV1(
        LettaBlockProfileBackendV1(client=DirectBlockClientFake(), configuration=configuration)
    ))
    return LettaExternalAdapterV3(bridge=bridge, configuration=configuration, target_objects=(_key(),)), bridge


def test_letta_adapter_reports_truthful_direct_block_profile_capabilities() -> None:
    adapter, bridge = _adapter()
    adapter.reset(ResetRequestV3(namespace="letta-adapter-namespace"))
    add = adapter.ingest_event(_event("add", 0, 'Add default|alice|city| with value "Paris".'))
    update = adapter.ingest_event(_event("update", 1, 'Update default|alice|city| with value "Lyon".'))
    noop = adapter.ingest_event(_event("noop", 2, "No memory object changes."))
    entries = adapter.export_entries()
    retrieval = adapter.retrieve(RetrievalRequestV3(query=_query(), k=1))

    assert add.execution_status.value == update.execution_status.value == noop.execution_status.value == "executed"
    assert entries.entries[0].value_candidate == "Lyon"
    assert retrieval.trace.retrieved_entries[0].entry_id == entries.entries[0].entry_id
    assert adapter.capabilities().supports_add is True
    assert adapter.capabilities().supports_update is True
    assert adapter.capabilities().supports_noop is True
    assert adapter.capabilities().supports_delete is True
    assert adapter.capabilities().exports_entries is True
    assert adapter.capabilities().exports_values is True
    assert adapter.capabilities().supports_native_answer is False
    assert adapter.capabilities().supports_multi_object_query is False
    assert adapter.answer(_query(), "slot_prompt").prediction.error_flags == ("answer_mode_not_supported",)
    request_text = str([request.payload for request in bridge.requests])
    assert "hidden-action" not in request_text
    assert "hidden" not in request_text


def test_letta_adapter_runs_twenty_isolated_reset_trials() -> None:
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    adapter, _ = _adapter()
    report = run_namespace_reset_probe(
        adapter, candidate_id="letta_0_16_8_block_profile", run_prefix="letta-reset"
    )

    assert report.passed is True
    assert len(report.trials) == 20


def test_letta_provider_facade_exposes_sdk_free_adapter() -> None:
    from mub.vnext.external.providers import LettaExternalAdapterV3

    assert LettaExternalAdapterV3.__name__ == "LettaExternalAdapterV3"
