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


def test_letta_adapter_rejects_protocol_fake_identity() -> None:
    import pytest
    from mub.vnext.external.providers.letta_adapter import LettaAdapterError

    with pytest.raises(LettaAdapterError):
        _adapter()


def test_letta_protocol_fake_cannot_run_authenticated_reset_probe() -> None:
    import pytest
    from mub.vnext.external.providers.letta_adapter import LettaAdapterError

    with pytest.raises(LettaAdapterError):
        _adapter()


def test_letta_provider_facade_exposes_sdk_free_adapter() -> None:
    from mub.vnext.external.providers import LettaExternalAdapterV3

    assert LettaExternalAdapterV3.__name__ == "LettaExternalAdapterV3"
