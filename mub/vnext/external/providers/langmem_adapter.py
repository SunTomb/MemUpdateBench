from __future__ import annotations

import json
from typing import Protocol

from mub.vnext.contracts.common import thaw_json
from mub.vnext.contracts.enums import ActionScope, AnswerDisposition, Operation
from mub.vnext.contracts.v3.adapter import (
    AdapterActionPayloadV3,
    AdapterActionResultV3,
    AdapterAnswerResultV3,
    AdapterCapabilitiesV3,
    AdapterInfoV3,
    ExportEntriesResultV3,
    ExportStateResultV3,
    ResetRequestV3,
    ResetResultV3,
    RetrievalRequestV3,
    RetrievalResultV3,
    VersionHistoryExportRequestV3,
    VersionHistoryExportResultV3,
)
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, object_identity
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.runtime import (
    AnswerPredictionV3,
    MemoryEntryRecordV3,
    RetrievalTraceV3,
)
from mub.vnext.contracts.v3.task import MemoryEventV3, MemoryQueryV3
from mub.vnext.external.bridge import (
    WorkerOperation,
    WorkerRequestV1,
    WorkerResponseStatus,
    WorkerResponseV1,
)
from mub.vnext.external.providers.langmem import (
    LANGMEM_PACKAGE_VERSION,
    LangMemAdapterConfigurationV1,
    compute_langmem_configuration_hash,
)
from mub.vnext.external.providers.langmem_protocol import (
    LangMemWorkerCloseResultV1,
    LangMemWorkerEntryListV1,
    LangMemWorkerEntryV1,
    LangMemWorkerHealthV1,
    LangMemWorkerMutationResultV1,
    LangMemWorkerResetResultV1,
    LangMemWorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.runtime.support_v3 import (
    VisibleActionParseError,
    parse_visible_action_v3,
)

LANGMEM_ADAPTER_VERSION = "memupdatebench-langmem-adapter-v1"
_WORKER_ERROR_CODES = frozenset(
    {"invalid_request_payload", "not_supported", "worker_backend_error", "worker_closed"}
)


class LangMemAdapterError(RuntimeError):
    pass


class WorkerBridgeV1(Protocol):
    def request(self, request: WorkerRequestV1) -> WorkerResponseV1: ...

    def close(self) -> None: ...


class LangMemExternalAdapterV3:
    def __init__(
        self,
        *,
        bridge: WorkerBridgeV1,
        configuration: LangMemAdapterConfigurationV1,
        target_objects: tuple[FrozenMemoryObjectKey, ...],
    ) -> None:
        if type(configuration) is not LangMemAdapterConfigurationV1:
            raise ValueError("LangMem adapter requires exact public configuration")
        if type(target_objects) is not tuple or any(
            type(key) is not FrozenMemoryObjectKey for key in target_objects
        ):
            raise ValueError("LangMem adapter target objects require exact tuple")
        if len(target_objects) != 1:
            raise ValueError("LangMem profile mode requires exactly one target object")
        self._bridge = bridge
        self._configuration = configuration
        self._configuration_hash = compute_langmem_configuration_hash(configuration)
        self._target_objects = target_objects
        self._target_by_id = {key.canonical_id: key for key in target_objects}
        self._namespace: str | None = None
        self._ingested_event_ids: set[str] = set()
        self._request_index = 0
        self._closed = False
        try:
            self._authenticate_worker()
        except Exception:
            self._closed = True
            self._bridge.close()
            raise

    def _request_id(self, operation: WorkerOperation) -> str:
        value = f"langmem-{operation.value}-{self._request_index:08d}"
        self._request_index += 1
        return value

    def _request(
        self, operation: WorkerOperation, payload: dict | None = None
    ) -> WorkerResponseV1:
        if self._closed:
            raise LangMemAdapterError("LangMem adapter is closed")
        request = WorkerRequestV1(
            request_id=self._request_id(operation),
            operation=operation,
            payload={} if payload is None else payload,
        )
        response = self._bridge.request(request)
        if type(response) is not WorkerResponseV1:
            raise LangMemAdapterError("LangMem worker response type is invalid")
        if response.request_id != request.request_id:
            raise LangMemAdapterError("LangMem worker response request ID is invalid")
        if response.status is WorkerResponseStatus.ERROR:
            error_code = (
                response.error_code
                if response.error_code in _WORKER_ERROR_CODES
                else "untrusted_worker_error"
            )
            raise LangMemAdapterError(
                f"LangMem worker request failed ({error_code})"
            )
        return response

    def _response_model(self, response: WorkerResponseV1, model_type):
        try:
            raw = json.dumps(
                thaw_json(response.payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            return model_type.model_validate_json(raw, strict=True)
        except Exception:
            raise LangMemAdapterError("LangMem worker response payload is invalid") from None

    def _authenticate_worker(self) -> None:
        health = self._response_model(
            self._request(WorkerOperation.HEALTH), LangMemWorkerHealthV1
        )
        if (
            health.package_name,
            health.package_version,
            health.source_commit,
            health.license_id,
            health.configuration_hash,
        ) != (
            "langmem",
            LANGMEM_PACKAGE_VERSION,
            "29cbe41e58528f92e9efa773c12e15c47be3808c",
            "MIT",
            self._configuration_hash,
        ):
            raise LangMemAdapterError("LangMem worker health identity is invalid")

    def adapter_info(self) -> AdapterInfoV3:
        return AdapterInfoV3(
            adapter_id="langmem_0_0_30_profile",
            adapter_version=LANGMEM_ADAPTER_VERSION,
            system_name="langmem_0_0_30_profile",
            system_version=LANGMEM_PACKAGE_VERSION,
            sdk_version=LANGMEM_PACKAGE_VERSION,
            configuration_hash=self._configuration_hash,
            extractor_id="langmem_native_profile_value_export",
            extractor_version="langmem-native-profile-value-export-v1",
        )

    def capabilities(self) -> AdapterCapabilitiesV3:
        return AdapterCapabilitiesV3(
            supports_isolated_reset=True,
            supports_event_ingest=True,
            supports_add=True,
            supports_update=True,
            supports_noop=True,
            supports_delete=True,
            supports_ttl=False,
            supports_native_answer=False,
            exports_entries=True,
            exports_raw_state=False,
            exports_source_event_ids=True,
            exports_timestamps_or_order=True,
            exports_object_keys=True,
            exports_values=True,
            exports_retrieval_ids=True,
            exports_retrieval_scores=True,
            exports_action_trace=True,
            reports_latency=False,
            reports_token_usage=False,
            reports_cost=False,
            requires_evaluation_extractor=False,
            extractor_version="langmem-native-profile-value-export-v1",
            supports_scoped_delete=False,
            supports_historical_query=False,
            exports_version_history=False,
            supports_multi_object_query=False,
            exports_evidence_linkage=True,
        )

    def _reset_namespace(self, namespace: str) -> None:
        result = self._response_model(
            self._request(WorkerOperation.RESET, {"runtime_namespace": namespace}),
            LangMemWorkerResetResultV1,
        )
        if result.namespace != namespace or result.success is not True:
            raise LangMemAdapterError("LangMem reset response is inconsistent")

    def reset(self, request: ResetRequestV3) -> ResetResultV3:
        if type(request) is not ResetRequestV3:
            raise ValueError("LangMem reset requires exact ResetRequestV3")
        self._reset_namespace(request.namespace)
        self._namespace = request.namespace
        self._ingested_event_ids.clear()
        return ResetResultV3(success=True, namespace=request.namespace)

    def reset_namespace(self, namespace: str) -> None:
        self._reset_namespace(namespace)
        if namespace == self._namespace:
            self._ingested_event_ids.clear()

    def write_sentinel(
        self, namespace: str, sentinel_id: str, sentinel_text: str
    ) -> None:
        event = {
            "event_id": sentinel_id,
            "sequence_index": 0,
            "logical_time": None,
            "raw_text": (
                f"Add default|mubreset|sentinel| with value "
                f"{json.dumps(sentinel_text, ensure_ascii=False)}."
            ),
            "runtime_namespace": namespace,
        }
        result = self._response_model(
            self._request(WorkerOperation.INGEST_EVENT, {"event": event}),
            LangMemWorkerMutationResultV1,
        )
        if result.event_id != sentinel_id or result.effective_operation != "add":
            raise LangMemAdapterError("LangMem sentinel ingest is inconsistent")

    def sentinel_visible(self, namespace: str, sentinel_text: str) -> bool:
        return any(
            entry.value == sentinel_text
            for entry in self._export_worker_entries(namespace)
        )

    def _export_worker_entries(
        self, namespace: str
    ) -> tuple[LangMemWorkerEntryV1, ...]:
        result = self._response_model(
            self._request(
                WorkerOperation.EXPORT_ENTRIES, {"runtime_namespace": namespace}
            ),
            LangMemWorkerEntryListV1,
        )
        return result.entries

    def _record(self, entry: LangMemWorkerEntryV1) -> MemoryEntryRecordV3:
        key = self._target_by_id.get(entry.canonical_object_id)
        if key is None:
            raise LangMemAdapterError("LangMem exported an undeclared object")
        return MemoryEntryRecordV3(
            entry_id=entry.entry_id,
            content=entry.content,
            object_key_candidate=key,
            value_candidate=entry.value,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            source_event_ids=tuple(
                event_id
                for event_id in entry.source_event_ids
                if event_id in self._ingested_event_ids
            ),
            raw_metadata={
                "mub_sequence_index": entry.sequence_index,
                "field_provenance": "native_value_export",
            },
        )

    def _requested_action(self, event: MemoryEventV3) -> AdapterActionPayloadV3:
        try:
            observed = parse_visible_action_v3(event, self._target_objects)
        except VisibleActionParseError:
            text = event.normalized_text.strip()
            prefix = "Delete "
            if not text.startswith(prefix) or not text.endswith("."):
                raise
            target_text, marker, metadata_text = text[len(prefix) : -1].partition(" [")
            if not marker or not metadata_text.endswith("]"):
                raise LangMemAdapterError("LangMem visible DELETE is malformed")
            metadata = {}
            for field in metadata_text[:-1].split(";"):
                name, separator, value = field.strip().partition("=")
                if not separator or not name or not value or name in metadata:
                    raise LangMemAdapterError("LangMem visible DELETE is malformed")
                metadata[name] = value
            target = self._target_by_id.get(target_text)
            if (
                target is None
                or metadata.get("scope") != "object"
                or metadata.get("enumerated_targets") != target_text
                or metadata.get("event_logical_time") != event.timestamp
                or not metadata.get("effective_at")
            ):
                raise LangMemAdapterError("LangMem visible DELETE is inconsistent")
            return AdapterActionPayloadV3(
                operation=Operation.DELETE,
                scope=ActionScope.OBJECT,
                target_object_keys=(target,),
            )
        return AdapterActionPayloadV3(
            operation=observed.operation,
            scope=observed.scope,
            target_object_keys=observed.target_object_keys,
            value=observed.value,
        )

    def ingest_event(self, event: MemoryEventV3) -> AdapterActionResultV3:
        if self._namespace is None:
            raise LangMemAdapterError("LangMem adapter namespace is not initialized")
        requested = self._requested_action(event)
        visible = visible_event_input(event, runtime_namespace=self._namespace)
        result = self._response_model(
            self._request(
                WorkerOperation.INGEST_EVENT,
                {"event": visible.model_dump(mode="json")},
            ),
            LangMemWorkerMutationResultV1,
        )
        if result.event_id != event.event_id:
            raise LangMemAdapterError("LangMem ingest event ID is inconsistent")
        self._ingested_event_ids.add(event.event_id)
        if result.effective_operation == "noop":
            status = (
                ExecutionStatusV3.EXECUTED
                if requested.operation is Operation.NOOP
                else ExecutionStatusV3.NO_EFFECT
            )
            return AdapterActionResultV3(
                event_id=event.event_id,
                requested_action=requested,
                effective_action=AdapterActionPayloadV3(operation=Operation.NOOP),
                execution_status=status,
                reason=None if status is ExecutionStatusV3.EXECUTED else "provider_no_effect",
            )
        effective = Operation(result.effective_operation.upper())
        if effective is not requested.operation or result.entry_id is None:
            raise LangMemAdapterError("LangMem effective lifecycle is inconsistent")
        return AdapterActionResultV3(
            event_id=event.event_id,
            requested_action=requested,
            effective_action=requested,
            execution_status=ExecutionStatusV3.EXECUTED,
            affected_entry_ids=(result.entry_id,),
            raw_result={"field_provenance": "native_value_export"},
        )

    def export_entries(self) -> ExportEntriesResultV3:
        if self._namespace is None:
            raise LangMemAdapterError("LangMem adapter namespace is not initialized")
        return ExportEntriesResultV3(
            entries=tuple(
                self._record(entry)
                for entry in self._export_worker_entries(self._namespace)
            )
        )

    def export_raw_state(self) -> ExportStateResultV3:
        raise NotImplementedError("LangMem raw state export is not supported")

    def export_version_history(
        self, request: VersionHistoryExportRequestV3
    ) -> VersionHistoryExportResultV3:
        raise NotImplementedError("LangMem version history export is not supported")

    def retrieve(self, request: RetrievalRequestV3) -> RetrievalResultV3:
        if self._namespace is None:
            raise LangMemAdapterError("LangMem adapter namespace is not initialized")
        if type(request) is not RetrievalRequestV3:
            raise ValueError("LangMem retrieval requires exact RetrievalRequestV3")
        visible = visible_query_input(
            request.query, k=request.k, runtime_namespace=self._namespace
        )
        result = self._response_model(
            self._request(
                WorkerOperation.RETRIEVE, {"query": visible.model_dump(mode="json")}
            ),
            LangMemWorkerRetrievalResultV1,
        )
        if result.query_id != request.query.query_id:
            raise LangMemAdapterError("LangMem retrieval query ID is inconsistent")
        entries = tuple(self._record(entry) for entry in result.entries)
        return RetrievalResultV3(
            request=request,
            trace=RetrievalTraceV3(
                query_id=request.query.query_id,
                retrieved_entries=entries,
                scores=result.scores,
                ranks=tuple(range(1, len(entries) + 1)),
                retrieval_policy="native_inmemory_then_deterministic_local",
                context_order="native",
                version_metadata={"score_provenance": "deterministic_local"},
            ),
        )

    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3:
        if mode != "slot_direct":
            return AdapterAnswerResultV3(
                prediction=AnswerPredictionV3(
                    query_id=query.query_id,
                    raw_output="",
                    disposition=AnswerDisposition.UNAVAILABLE,
                    parsed_answer=None,
                    format_valid=False,
                    error_flags=("answer_mode_not_supported",),
                )
            )
        if len(query.target_object_keys) != 1:
            return self._unavailable_answer(query, "profile_single_record_only")
        target_identity = object_identity(query.target_object_keys[0])
        entries = tuple(
            entry
            for entry in self.export_entries().entries
            if entry.object_key_candidate is not None
            and object_identity(entry.object_key_candidate) == target_identity
            and entry.value_candidate is not None
        )
        if len(entries) != 1:
            return self._unavailable_answer(query, "normalized_state_unavailable")
        entry = entries[0]
        return AdapterAnswerResultV3(
            prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output=json.dumps(
                    entry.value_candidate,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                disposition=AnswerDisposition.ANSWERED,
                parsed_answer=entry.value_candidate,
                cited_event_ids=entry.source_event_ids,
                cited_entry_ids=(entry.entry_id,),
                cited_object_keys=(entry.object_key_candidate,),
                format_valid=True,
            )
        )

    @staticmethod
    def _unavailable_answer(
        query: MemoryQueryV3, error_flag: str
    ) -> AdapterAnswerResultV3:
        return AdapterAnswerResultV3(
            prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output="",
                disposition=AnswerDisposition.UNAVAILABLE,
                parsed_answer=None,
                format_valid=False,
                error_flags=(error_flag,),
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        error: Exception | None = None
        try:
            result = self._response_model(
                self._request(WorkerOperation.CLOSE), LangMemWorkerCloseResultV1
            )
            if result.closed is not True:
                raise LangMemAdapterError("LangMem worker close response is invalid")
        except Exception as exc:
            error = exc
        finally:
            self._closed = True
            self._bridge.close()
        if error is not None:
            raise error


__all__ = ["LANGMEM_ADAPTER_VERSION", "LangMemAdapterError", "LangMemExternalAdapterV3"]
