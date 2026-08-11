from __future__ import annotations

from pathlib import Path
import os
import sys

import pytest
import textwrap

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.enums import (
    AnswerSchema,
    EvaluationMode,
    EventRole,
)
from mub.vnext.contracts.v3.adapter import (
    ResetRequestV3,
    RetrievalRequestV3,
)
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    MemoryEventV3,
    MemoryQueryV3,
)
from mub.vnext.external.bridge import (
    WorkerOperation,
    WorkerRequestV1,
    WorkerResponseStatus,
    WorkerResponseV1,
)
from mub.vnext.external.providers.mem0_protocol import (
    Mem0WorkerEntryV1,
    Mem0WorkerHealthV1,
    Mem0WorkerIngestResultV1,
    Mem0WorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import (
    ProviderEventInputV1,
    ProviderQueryInputV1,
)
from mub.vnext.external.workers.mem0_worker import Mem0WorkerServiceV1


MODEL_PROVENANCE_SHA256 = (
    "8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e"
)


def _key() -> FrozenMemoryObjectKey:
    return FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
        subkey=None,
    )


def _configuration():
    from mub.vnext.external.providers.mem0 import (
        build_mem0_adapter_configuration,
    )

    return build_mem0_adapter_configuration(
        run_id="task10-mem0-adapter",
        model_provenance_ref=ArtifactRef(
            path="model_provenance.json",
            sha256=MODEL_PROVENANCE_SHA256,
            media_type="application/json",
            record_count=1,
        ),
    )


class AdapterFakeBackend:
    def __init__(self, configuration) -> None:
        self.configuration = configuration
        self.entries: dict[str, list[Mem0WorkerEntryV1]] = {}
        self.events: list[ProviderEventInputV1] = []
        self.queries: list[ProviderQueryInputV1] = []
        self.closed = False

    def health(self):
        from mub.vnext.external.providers.mem0 import (
            compute_mem0_configuration_hash,
        )

        return Mem0WorkerHealthV1(
            package_name="mem0ai",
            package_version="2.0.17",
            collection_name=self.configuration.collection_name,
            configuration_hash=compute_mem0_configuration_hash(
                self.configuration
            ),
        )

    def reset_namespace(self, namespace: str) -> None:
        self.entries[namespace] = []

    def ingest_event(self, event, *, infer: bool):
        self.events.append(event)
        entry = Mem0WorkerEntryV1(
            entry_id=f"entry-{len(self.events):03d}",
            content=event.raw_text,
            created_at=event.logical_time,
            updated_at=event.logical_time,
            source_event_ids=(event.event_id,),
            native_metadata={"mub_sequence_index": event.sequence_index},
        )
        self.entries.setdefault(event.runtime_namespace, []).append(entry)
        return Mem0WorkerIngestResultV1(
            event_id=event.event_id,
            effective_operation="add",
            affected_entry_ids=(entry.entry_id,),
        )

    def retrieve(self, query):
        self.queries.append(query)
        entries = tuple(
            self.entries.get(query.runtime_namespace, ())[-query.k :]
        )
        return Mem0WorkerRetrievalResultV1(
            query_id=query.query_id,
            entries=entries,
            scores=tuple(0.9 - index * 0.1 for index in range(len(entries))),
        )

    def export_entries(self, namespace: str):
        return tuple(self.entries.get(namespace, ()))

    def close(self) -> None:
        self.closed = True


class InProcessBridge:
    def __init__(self, service: Mem0WorkerServiceV1) -> None:
        self.service = service
        self.requests: list[WorkerRequestV1] = []
        self.closed = False

    def request(self, request: WorkerRequestV1):
        self.requests.append(request)
        return self.service.handle(request)

    def close(self) -> None:
        self.closed = True


def _adapter():
    from mub.vnext.external.providers.mem0_adapter import Mem0ExternalAdapterV3

    configuration = _configuration()
    backend = AdapterFakeBackend(configuration)
    bridge = InProcessBridge(Mem0WorkerServiceV1(backend))
    adapter = Mem0ExternalAdapterV3(
        bridge=bridge,
        configuration=configuration,
        target_objects=(_key(),),
    )
    return adapter, bridge, backend


def test_mem0_adapter_authenticates_worker_health() -> None:
    from mub.vnext.external.providers.mem0_adapter import (
        Mem0AdapterError,
        Mem0ExternalAdapterV3,
    )

    configuration = _configuration()
    backend = AdapterFakeBackend(configuration)
    backend.health = lambda: Mem0WorkerHealthV1(
        package_name="mem0ai",
        package_version="2.0.17",
        collection_name="mub_mem0_wrong",
        configuration_hash="1" * 64,
    )
    bridge = InProcessBridge(Mem0WorkerServiceV1(backend))

    with pytest.raises(Mem0AdapterError, match="health"):
        Mem0ExternalAdapterV3(
            bridge=bridge,
            configuration=configuration,
            target_objects=(_key(),),
        )


def test_mem0_adapter_rejects_mismatched_worker_response_id() -> None:
    from mub.vnext.external.providers.mem0_adapter import (
        Mem0AdapterError,
        Mem0ExternalAdapterV3,
    )

    configuration = _configuration()
    backend = AdapterFakeBackend(configuration)

    class WrongIdBridge(InProcessBridge):
        def request(self, request: WorkerRequestV1):
            response = super().request(request)
            return response.validated_replace(request_id="wrong-request-id")

    with pytest.raises(Mem0AdapterError, match="request ID"):
        Mem0ExternalAdapterV3(
            bridge=WrongIdBridge(Mem0WorkerServiceV1(backend)),
            configuration=configuration,
            target_objects=(_key(),),
        )


def test_mem0_adapter_does_not_echo_untrusted_worker_error_codes() -> None:
    from mub.vnext.external.providers.mem0_adapter import (
        Mem0AdapterError,
        Mem0ExternalAdapterV3,
    )

    configuration = _configuration()
    secret = "client_secret=hunter2"

    class FailingBridge:
        def __init__(self):
            self.closed = False

        def request(self, request: WorkerRequestV1):
            return WorkerResponseV1(
                request_id=request.request_id,
                status=WorkerResponseStatus.ERROR,
                error_code=secret,
            )

        def close(self):
            self.closed = True

    bridge = FailingBridge()
    with pytest.raises(Mem0AdapterError, match="worker request failed") as exc_info:
        Mem0ExternalAdapterV3(
            bridge=bridge,
            configuration=configuration,
            target_objects=(_key(),),
        )
    assert secret not in str(exc_info.value)
    assert "hunter2" not in str(exc_info.value)
    assert bridge.closed is True


def _event(event_id: str = "event-1", value: str = "Paris") -> MemoryEventV3:
    text = f'Add default|alice|city| with value "{value}".'
    return MemoryEventV3(
        event_id=event_id,
        sequence_index=0,
        timestamp="2026-01-01T00:00:00Z",
        raw_text=text,
        normalized_text=text,
        speaker="user",
        gold_action_ids=("gold-action-secret",),
        role=EventRole.LATEST_GOLD,
        source_anchor={"gold_evidence": "must-not-cross"},
        metadata={"expected_effect": "must-not-cross"},
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


def test_mem0_provider_facade_exports_host_adapter_without_optional_sdk() -> None:
    from mub.vnext.external.providers import Mem0ExternalAdapterV3

    assert Mem0ExternalAdapterV3.__name__ == "Mem0ExternalAdapterV3"


def test_mem0_adapter_reports_conservative_level2_capabilities() -> None:
    adapter, _, _ = _adapter()
    info = adapter.adapter_info()
    capabilities = adapter.capabilities()

    assert info.adapter_id == "mem0_oss"
    assert info.system_name == "mem0_oss"
    assert info.system_version == "2.0.17"
    assert info.sdk_version == "2.0.17"
    assert info.extractor_id == "mub_visible_slot_entry_extractor"
    assert info.extractor_version == "mub-visible-slot-entry-extractor-v1"
    assert capabilities.supports_isolated_reset is True
    assert capabilities.supports_event_ingest is True
    assert capabilities.supports_add is True
    assert capabilities.supports_update is False
    assert capabilities.supports_delete is False
    assert capabilities.exports_entries is True
    assert capabilities.exports_retrieval_ids is True
    assert capabilities.exports_retrieval_scores is True
    assert capabilities.exports_action_trace is True
    assert capabilities.requires_evaluation_extractor is True
    assert capabilities.exports_object_keys is False
    assert capabilities.exports_values is False
    assert capabilities.presentation_level(True) == 3
    assert capabilities.presentation_level(False) == 2


def test_mem0_adapter_sends_only_visible_event_and_normalizes_state() -> None:
    adapter, bridge, backend = _adapter()
    namespace = "task10-run-task-1"
    reset = adapter.reset(ResetRequestV3(namespace=namespace))
    assert reset.success is True

    result = adapter.ingest_event(_event())
    assert result.execution_status is ExecutionStatusV3.EXECUTED
    assert result.requested_action.operation.value == "ADD"
    assert result.effective_action.operation.value == "ADD"
    assert result.affected_entry_ids == ("entry-001",)

    assert len(backend.events) == 1
    visible = backend.events[0]
    assert visible.raw_text == _event().raw_text
    ingest_requests = tuple(
        request
        for request in bridge.requests
        if request.operation is WorkerOperation.INGEST_EVENT
    )
    assert len(ingest_requests) == 1
    payload_text = str(ingest_requests[0].payload)
    assert "gold-action-secret" not in payload_text
    assert "gold_evidence" not in payload_text
    assert "expected_effect" not in payload_text

    exported = adapter.export_entries().entries
    assert len(exported) == 1
    assert exported[0].object_key_candidate == _key()
    assert exported[0].value_candidate == "Paris"
    assert exported[0].source_event_ids == ("event-1",)
    assert exported[0].raw_metadata["field_provenance"] == (
        "evaluation_extractor"
    )


def test_mem0_direct_namespace_reset_clears_source_linkage() -> None:
    adapter, _, backend = _adapter()
    namespace = "task10-direct-reset"
    adapter.reset(ResetRequestV3(namespace=namespace))
    adapter.ingest_event(_event())
    adapter.reset_namespace(namespace)
    backend.entries[namespace] = [
        Mem0WorkerEntryV1(
            entry_id="post-reset-entry",
            content='Add default|alice|city| with value "Paris".',
            source_event_ids=("event-1",),
        )
    ]
    assert adapter.export_entries().entries[0].source_event_ids == ()


def test_mem0_adapter_whitelists_visible_source_event_ids() -> None:
    adapter, _, backend = _adapter()
    namespace = "task10-public-source-ids"
    adapter.reset(ResetRequestV3(namespace=namespace))
    backend.entries[namespace] = [
        Mem0WorkerEntryV1(
            entry_id="source-entry",
            content='Add default|alice|city| with value "Paris".',
            source_event_ids=("gold-action-secret",),
        )
    ]
    assert adapter.export_entries().entries[0].source_event_ids == ()

    adapter.ingest_event(_event())
    exported = adapter.export_entries().entries
    assert exported[-1].source_event_ids == ("event-1",)


def test_mem0_adapter_whitelists_public_native_metadata() -> None:
    adapter, _, backend = _adapter()
    namespace = "task10-public-metadata"
    adapter.reset(ResetRequestV3(namespace=namespace))
    backend.entries[namespace] = [
        Mem0WorkerEntryV1(
            entry_id="metadata-entry",
            content='Add default|alice|city| with value "Paris".',
            native_metadata={
                "mub_sequence_index": 0,
                "client_secret": "hunter2",
            },
        )
    ]

    metadata = adapter.export_entries().entries[0].raw_metadata
    assert metadata == {
        "mub_sequence_index": 0,
        "field_provenance": "evaluation_extractor",
    }
    assert "hunter2" not in str(metadata)


def test_mem0_adapter_fails_closed_on_multi_entry_side_effects() -> None:
    from mub.vnext.external.providers.mem0_adapter import Mem0ExternalAdapterV3

    configuration = _configuration()

    class MultiEntryBackend(AdapterFakeBackend):
        def ingest_event(self, event, *, infer: bool):
            primary = super().ingest_event(event, infer=infer)
            extra = Mem0WorkerEntryV1(
                entry_id="extra-entry",
                content="provider-generated extra memory",
                source_event_ids=(event.event_id,),
            )
            self.entries[event.runtime_namespace].append(extra)
            return Mem0WorkerIngestResultV1(
                event_id=event.event_id,
                effective_operation="add",
                affected_entry_ids=(*primary.affected_entry_ids, extra.entry_id),
            )

    backend = MultiEntryBackend(configuration)
    adapter = Mem0ExternalAdapterV3(
        bridge=InProcessBridge(Mem0WorkerServiceV1(backend)),
        configuration=configuration,
        target_objects=(_key(),),
    )
    adapter.reset(ResetRequestV3(namespace="task10-multi-entry"))

    result = adapter.ingest_event(_event())
    assert result.execution_status is ExecutionStatusV3.FAILED
    assert result.affected_entry_ids == ()
    assert result.error["code"] == "entry_extraction_failed"


def test_mem0_entry_extractor_supports_escaped_canonical_identity_parts() -> None:
    from mub.vnext.external.providers.mem0_adapter import Mem0ExternalAdapterV3

    configuration = _configuration()
    backend = AdapterFakeBackend(configuration)
    bridge = InProcessBridge(Mem0WorkerServiceV1(backend))
    key = FrozenMemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice smith|west",
        attribute="home city",
        subkey=None,
    )
    adapter = Mem0ExternalAdapterV3(
        bridge=bridge,
        configuration=configuration,
        target_objects=(key,),
    )
    namespace = "task10-run-complex-key"
    adapter.reset(ResetRequestV3(namespace=namespace))
    backend.entries[namespace] = [
        Mem0WorkerEntryV1(
            entry_id="complex-entry",
            content=f'Add {key.canonical_id} with value "Paris".',
        )
    ]

    exported = adapter.export_entries().entries
    assert exported[0].object_key_candidate == key
    assert exported[0].value_candidate == "Paris"


def test_mem0_adapter_preserves_native_retrieval_order_scores_and_ids() -> None:
    adapter, _, backend = _adapter()
    namespace = "task10-run-task-1"
    adapter.reset(ResetRequestV3(namespace=namespace))
    adapter.ingest_event(_event("event-1", "Paris"))
    second = _event("event-2", "Lyon").validated_replace(sequence_index=1)
    adapter.ingest_event(second)

    result = adapter.retrieve(RetrievalRequestV3(query=_query(), k=2))
    assert result.trace.query_id == "query-1"
    assert tuple(entry.entry_id for entry in result.trace.retrieved_entries) == (
        "entry-001",
        "entry-002",
    )
    assert result.trace.scores == (0.9, 0.8)
    assert result.trace.ranks == (1, 2)
    assert result.trace.retrieval_policy == "normal_topk"
    assert result.trace.context_order == "native"
    assert backend.queries[0].query_text == _query().text


def test_mem0_adapter_slot_direct_answer_uses_only_normalized_entries() -> None:
    adapter, _, _ = _adapter()
    adapter.reset(ResetRequestV3(namespace="task10-run-task-1"))
    adapter.ingest_event(_event())

    answer = adapter.answer(_query(), "slot_direct")
    assert answer.prediction.format_valid is True
    assert answer.prediction.parsed_answer == "Paris"
    assert answer.prediction.cited_entry_ids == ("entry-001",)
    assert answer.prediction.cited_object_keys == (_key(),)

    unavailable = adapter.answer(_query(), "slot_prompt")
    assert unavailable.prediction.disposition.value == "unavailable"
    assert unavailable.prediction.parsed_answer is None


def test_mem0_adapter_slot_direct_uses_observable_sequence_not_export_order() -> None:
    adapter, _, backend = _adapter()
    namespace = "task10-run-task-1"
    adapter.reset(ResetRequestV3(namespace=namespace))
    adapter.ingest_event(_event("event-1", "Paris"))
    adapter.ingest_event(
        _event("event-2", "Lyon").validated_replace(sequence_index=1)
    )
    backend.entries[namespace].reverse()

    answer = adapter.answer(_query(), "slot_direct")
    assert answer.prediction.parsed_answer == "Lyon"
    assert answer.prediction.cited_entry_ids == ("entry-002",)


def test_mem0_adapter_reset_probe_passes_all_trials() -> None:
    from mub.vnext.external.probe_v3 import run_namespace_reset_probe

    adapter, _, _ = _adapter()
    probe = run_namespace_reset_probe(
        adapter,
        candidate_id="mem0_oss",
        run_prefix="task10-preflight",
    )
    assert probe.passed is True
    assert len(probe.trials) == 20


def test_mem0_host_round_trips_through_jsonl_subprocess_bridge(
    tmp_path: Path,
) -> None:
    from mub.vnext.external.bridge import JsonlSubprocessBridge
    from mub.vnext.external.providers.mem0 import compute_mem0_configuration_hash
    from mub.vnext.external.providers.mem0_adapter import Mem0ExternalAdapterV3
    from mub.vnext.external.security import build_worker_environment

    configuration = _configuration()
    worker = tmp_path / "fake_mem0_worker.py"
    worker.write_text(
        textwrap.dedent(
            f'''
            import sys

            from mub.vnext.external.providers.mem0_protocol import (
                Mem0WorkerEntryV1,
                Mem0WorkerHealthV1,
                Mem0WorkerIngestResultV1,
                Mem0WorkerRetrievalResultV1,
            )
            from mub.vnext.external.workers.mem0_worker import (
                Mem0WorkerServiceV1,
                serve_mem0_worker_jsonl,
            )

            class Backend:
                def __init__(self):
                    self.entries = {{}}

                def health(self):
                    return Mem0WorkerHealthV1(
                        package_name="mem0ai",
                        package_version="2.0.17",
                        collection_name={configuration.collection_name!r},
                        configuration_hash={compute_mem0_configuration_hash(configuration)!r},
                    )

                def reset_namespace(self, namespace):
                    self.entries[namespace] = []

                def ingest_event(self, event, *, infer):
                    entry = Mem0WorkerEntryV1(
                        entry_id="subprocess-entry",
                        content=event.raw_text,
                        source_event_ids=(event.event_id,),
                        native_metadata={{
                            "mub_sequence_index": event.sequence_index,
                        }},
                    )
                    self.entries.setdefault(event.runtime_namespace, []).append(entry)
                    return Mem0WorkerIngestResultV1(
                        event_id=event.event_id,
                        effective_operation="add",
                        affected_entry_ids=(entry.entry_id,),
                    )

                def retrieve(self, query):
                    entries = tuple(self.entries.get(query.runtime_namespace, ()))
                    return Mem0WorkerRetrievalResultV1(
                        query_id=query.query_id,
                        entries=entries[:query.k],
                        scores=tuple(1.0 for _ in entries[:query.k]),
                    )

                def export_entries(self, namespace):
                    return tuple(self.entries.get(namespace, ()))

                def close(self):
                    pass

            serve_mem0_worker_jsonl(
                Mem0WorkerServiceV1(Backend()),
                input_stream=sys.stdin.buffer,
                output_stream=sys.stdout.buffer,
            )
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[2]
    source_environment = dict(os.environ)
    source_environment["PYTHONPATH"] = os.pathsep.join(
        (str(project_root), *(path for path in sys.path if path))
    )
    source_environment["PYTHONIOENCODING"] = "utf-8"
    allowed_names = tuple(
        name
        for name in (
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "PYTHONPATH",
            "PYTHONIOENCODING",
        )
        if name in source_environment
    )
    environment = build_worker_environment(
        source_environment,
        allowed_names=allowed_names,
    )
    bridge = JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=environment,
        timeout_seconds=5.0,
    )
    adapter = Mem0ExternalAdapterV3(
        bridge=bridge,
        configuration=configuration,
        target_objects=(_key(),),
    )
    adapter.reset(ResetRequestV3(namespace="task10-subprocess"))
    adapter.ingest_event(_event())
    assert adapter.export_entries().entries[0].value_candidate == "Paris"
    adapter.close()


def test_mem0_adapter_close_is_idempotent() -> None:
    adapter, bridge, backend = _adapter()
    adapter.close()
    adapter.close()
    assert bridge.closed is True
    assert backend.closed is True
    assert tuple(
        request.operation
        for request in bridge.requests
        if request.operation is WorkerOperation.CLOSE
    ) == (WorkerOperation.CLOSE,)
