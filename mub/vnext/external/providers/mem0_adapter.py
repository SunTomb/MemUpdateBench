from __future__ import annotations

import json
from typing import Protocol

from mub.vnext.contracts.common import thaw_json
from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerDisposition,
    Operation,
)
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
from mub.vnext.contracts.v3.common import (
    FrozenMemoryObjectKey,
    object_identity,
)
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
from mub.vnext.external.providers.mem0 import (
    MEM0_PACKAGE_VERSION,
    Mem0AdapterConfigurationV1,
    compute_mem0_configuration_hash,
)
from mub.vnext.external.providers.mem0_protocol import (
    Mem0WorkerCloseResultV1,
    Mem0WorkerEntryListV1,
    Mem0WorkerEntryV1,
    Mem0WorkerHealthV1,
    Mem0WorkerIngestResultV1,
    Mem0WorkerResetResultV1,
    Mem0WorkerRetrievalResultV1,
)
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.runtime.support_v3 import parse_visible_action_v3

MEM0_ADAPTER_VERSION = "memupdatebench-mem0-adapter-v1"
MEM0_ENTRY_EXTRACTOR_ID = "mub_visible_slot_entry_extractor"
MEM0_ENTRY_EXTRACTOR_VERSION = "mub-visible-slot-entry-extractor-v1"
_WORKER_ERROR_CODES = frozenset(
    {
        "invalid_request_payload",
        "not_supported",
        "worker_backend_error",
        "worker_closed",
    }
)


class Mem0AdapterError(RuntimeError):
    pass


class WorkerBridgeV1(Protocol):
    def request(self, request: WorkerRequestV1) -> WorkerResponseV1: ...

    def close(self) -> None: ...


class Mem0ExternalAdapterV3:
    def __init__(
        self,
        *,
        bridge: WorkerBridgeV1,
        configuration: Mem0AdapterConfigurationV1,
        target_objects: tuple[FrozenMemoryObjectKey, ...],
    ) -> None:
        if type(configuration) is not Mem0AdapterConfigurationV1:
            raise ValueError("Mem0 adapter requires exact public configuration")
        configuration_hash = compute_mem0_configuration_hash(configuration)
        if type(target_objects) is not tuple or any(
            type(key) is not FrozenMemoryObjectKey for key in target_objects
        ):
            raise ValueError("Mem0 adapter target objects require exact tuple")
        identities = tuple(object_identity(key) for key in target_objects)
        if len(identities) != len(set(identities)):
            raise ValueError("Mem0 adapter target objects must be unique")
        self._bridge = bridge
        self._configuration = configuration
        self._configuration_hash = configuration_hash
        self._target_objects = target_objects
        self._target_by_id = {
            key.canonical_id: key for key in target_objects
        }
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
        value = (
            f"mem0-{operation.value}-{self._request_index:08d}"
        )
        self._request_index += 1
        return value

    def _request(
        self,
        operation: WorkerOperation,
        payload: dict | None = None,
    ) -> WorkerResponseV1:
        if self._closed:
            raise Mem0AdapterError("Mem0 adapter is closed")
        request = WorkerRequestV1(
            request_id=self._request_id(operation),
            operation=operation,
            payload={} if payload is None else payload,
        )
        response = self._bridge.request(request)
        if type(response) is not WorkerResponseV1:
            raise Mem0AdapterError("Mem0 worker response type is invalid")
        if response.request_id != request.request_id:
            raise Mem0AdapterError("Mem0 worker response request ID is invalid")
        if response.status is WorkerResponseStatus.ERROR:
            error_code = (
                response.error_code
                if response.error_code in _WORKER_ERROR_CODES
                else "untrusted_worker_error"
            )
            raise Mem0AdapterError(
                f"Mem0 worker request failed ({error_code})"
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
            raise Mem0AdapterError(
                "Mem0 worker response payload is invalid"
            ) from None

    def _authenticate_worker(self) -> None:
        response = self._request(WorkerOperation.HEALTH)
        health = self._response_model(response, Mem0WorkerHealthV1)
        expected = (
            "mem0ai",
            MEM0_PACKAGE_VERSION,
            self._configuration.collection_name,
            self._configuration_hash,
        )
        observed = (
            health.package_name,
            health.package_version,
            health.collection_name,
            health.configuration_hash,
        )
        if observed != expected:
            raise Mem0AdapterError("Mem0 worker health identity is invalid")

    def adapter_info(self) -> AdapterInfoV3:
        return AdapterInfoV3(
            adapter_id="mem0_oss",
            adapter_version=MEM0_ADAPTER_VERSION,
            system_name="mem0_oss",
            system_version=MEM0_PACKAGE_VERSION,
            sdk_version=MEM0_PACKAGE_VERSION,
            configuration_hash=self._configuration_hash,
            extractor_id=MEM0_ENTRY_EXTRACTOR_ID,
            extractor_version=MEM0_ENTRY_EXTRACTOR_VERSION,
        )

    def capabilities(self) -> AdapterCapabilitiesV3:
        return AdapterCapabilitiesV3(
            supports_isolated_reset=True,
            supports_event_ingest=True,
            supports_add=True,
            supports_update=False,
            supports_noop=False,
            supports_delete=False,
            supports_ttl=False,
            supports_native_answer=False,
            exports_entries=True,
            exports_raw_state=False,
            exports_source_event_ids=True,
            exports_timestamps_or_order=True,
            exports_object_keys=False,
            exports_values=False,
            exports_retrieval_ids=True,
            exports_retrieval_scores=True,
            exports_action_trace=True,
            reports_latency=False,
            reports_token_usage=False,
            reports_cost=False,
            requires_evaluation_extractor=True,
            extractor_version=MEM0_ENTRY_EXTRACTOR_VERSION,
            supports_scoped_delete=False,
            supports_historical_query=False,
            exports_version_history=False,
            supports_multi_object_query=False,
            exports_evidence_linkage=False,
        )

    def _reset_namespace(self, namespace: str) -> None:
        response = self._request(
            WorkerOperation.RESET,
            {"runtime_namespace": namespace},
        )
        result = self._response_model(response, Mem0WorkerResetResultV1)
        if not result.success or result.namespace != namespace:
            raise Mem0AdapterError("Mem0 reset response is inconsistent")

    def reset(self, request: ResetRequestV3) -> ResetResultV3:
        if type(request) is not ResetRequestV3:
            raise ValueError("Mem0 reset requires exact ResetRequestV3")
        self._reset_namespace(request.namespace)
        self._namespace = request.namespace
        self._ingested_event_ids.clear()
        return ResetResultV3(success=True, namespace=request.namespace)

    def reset_namespace(self, namespace: str) -> None:
        self._reset_namespace(namespace)
        if namespace == self._namespace:
            self._ingested_event_ids.clear()

    def write_sentinel(
        self,
        namespace: str,
        sentinel_id: str,
        sentinel_text: str,
    ) -> None:
        event = {
            "event_id": sentinel_id,
            "sequence_index": 0,
            "logical_time": None,
            "raw_text": sentinel_text,
            "runtime_namespace": namespace,
        }
        response = self._request(
            WorkerOperation.INGEST_EVENT,
            {"event": event, "probe_mode": "sentinel"},
        )
        result = self._response_model(response, Mem0WorkerIngestResultV1)
        if result.event_id != sentinel_id:
            raise Mem0AdapterError("Mem0 sentinel ingest is inconsistent")

    def sentinel_visible(self, namespace: str, sentinel_text: str) -> bool:
        entries = self._export_worker_entries(namespace)
        return any(entry.content == sentinel_text for entry in entries)

    def ingest_event(self, event: MemoryEventV3) -> AdapterActionResultV3:
        if self._namespace is None:
            raise Mem0AdapterError("Mem0 adapter namespace is not initialized")
        observed = parse_visible_action_v3(event, self._target_objects)
        requested = AdapterActionPayloadV3(
            operation=observed.operation,
            scope=observed.scope,
            target_object_keys=observed.target_object_keys,
            value=observed.value,
        )
        visible = visible_event_input(
            event,
            runtime_namespace=self._namespace,
        )
        response = self._request(
            WorkerOperation.INGEST_EVENT,
            {
                "event": visible.model_dump(mode="json"),
                "probe_mode": "normal",
            },
        )
        result = self._response_model(response, Mem0WorkerIngestResultV1)
        if result.event_id != event.event_id:
            raise Mem0AdapterError("Mem0 ingest event ID is inconsistent")
        self._ingested_event_ids.add(event.event_id)
        if result.effective_operation == "noop":
            effective = AdapterActionPayloadV3(operation=Operation.NOOP)
            if observed.operation is Operation.NOOP:
                return AdapterActionResultV3(
                    event_id=event.event_id,
                    requested_action=requested,
                    effective_action=effective,
                    execution_status=ExecutionStatusV3.EXECUTED,
                )
            return AdapterActionResultV3(
                event_id=event.event_id,
                requested_action=requested,
                effective_action=effective,
                execution_status=ExecutionStatusV3.NO_EFFECT,
                reason="provider_no_effect",
            )

        affected = self._affected_records(result.affected_entry_ids)
        parsed = tuple(
            (record.object_key_candidate, record.value_candidate)
            for record in affected
            if record.object_key_candidate is not None
            and record.value_candidate is not None
        )
        if len(affected) != 1 or len(parsed) != 1:
            return AdapterActionResultV3(
                event_id=event.event_id,
                requested_action=requested,
                execution_status=ExecutionStatusV3.FAILED,
                error={"code": "entry_extraction_failed"},
            )
        key, value = parsed[0]
        return AdapterActionResultV3(
            event_id=event.event_id,
            requested_action=requested,
            effective_action=AdapterActionPayloadV3(
                operation=Operation.ADD,
                scope=ActionScope.OBJECT,
                target_object_keys=(key,),
                value=value,
            ),
            execution_status=ExecutionStatusV3.EXECUTED,
            affected_entry_ids=result.affected_entry_ids,
            raw_result={"field_provenance": "evaluation_extractor"},
        )

    def _export_worker_entries(
        self,
        namespace: str,
    ) -> tuple[Mem0WorkerEntryV1, ...]:
        response = self._request(
            WorkerOperation.EXPORT_ENTRIES,
            {"runtime_namespace": namespace},
        )
        result = self._response_model(response, Mem0WorkerEntryListV1)
        return result.entries

    def _record(self, entry: Mem0WorkerEntryV1) -> MemoryEntryRecordV3:
        key, value = self._extract_object_value(entry.content)
        native_metadata = thaw_json(entry.native_metadata)
        metadata: dict[str, object] = {}
        sequence_index = native_metadata.get("mub_sequence_index")
        if type(sequence_index) is int and sequence_index >= 0:
            metadata["mub_sequence_index"] = sequence_index
        logical_time = native_metadata.get("mub_logical_time")
        if type(logical_time) is str and logical_time.strip():
            metadata["mub_logical_time"] = logical_time
        metadata["field_provenance"] = "evaluation_extractor"
        return MemoryEntryRecordV3(
            entry_id=entry.entry_id,
            content=entry.content,
            object_key_candidate=key,
            value_candidate=value,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            source_event_ids=tuple(
                event_id
                for event_id in entry.source_event_ids
                if event_id in self._ingested_event_ids
            ),
            raw_metadata=metadata,
        )

    def _extract_object_value(
        self,
        content: str,
    ) -> tuple[FrozenMemoryObjectKey | None, object | None]:
        normalized = content.strip()
        key: FrozenMemoryObjectKey | None = None
        serialized_value: str | None = None
        for canonical_id, candidate in self._target_by_id.items():
            for verb in ("Add", "Update"):
                prefix = f"{verb} {canonical_id} with value "
                if normalized.startswith(prefix) and normalized.endswith("."):
                    key = candidate
                    serialized_value = normalized[len(prefix) : -1]
                    break
            if key is not None:
                break
        if key is None or serialized_value is None:
            return None, None
        try:
            value = json.loads(serialized_value)
        except (TypeError, ValueError):
            return None, None
        if value is None:
            return None, None
        return key, value

    def _affected_records(
        self,
        affected_entry_ids: tuple[str, ...],
    ) -> tuple[MemoryEntryRecordV3, ...]:
        if self._namespace is None:
            raise Mem0AdapterError("Mem0 adapter namespace is not initialized")
        by_id = {
            entry.entry_id: self._record(entry)
            for entry in self._export_worker_entries(self._namespace)
        }
        try:
            return tuple(by_id[entry_id] for entry_id in affected_entry_ids)
        except KeyError:
            raise Mem0AdapterError(
                "Mem0 affected entry ID is not exportable"
            ) from None

    def export_entries(self) -> ExportEntriesResultV3:
        if self._namespace is None:
            raise Mem0AdapterError("Mem0 adapter namespace is not initialized")
        entries = tuple(
            self._record(entry)
            for entry in self._export_worker_entries(self._namespace)
        )
        return ExportEntriesResultV3(entries=entries)

    def export_raw_state(self) -> ExportStateResultV3:
        raise NotImplementedError("Mem0 raw state export is not supported")

    def export_version_history(
        self,
        request: VersionHistoryExportRequestV3,
    ) -> VersionHistoryExportResultV3:
        raise NotImplementedError("Mem0 version history export is not supported")

    def retrieve(self, request: RetrievalRequestV3) -> RetrievalResultV3:
        if self._namespace is None:
            raise Mem0AdapterError("Mem0 adapter namespace is not initialized")
        if type(request) is not RetrievalRequestV3:
            raise ValueError("Mem0 retrieval requires exact RetrievalRequestV3")
        visible = visible_query_input(
            request.query,
            k=request.k,
            runtime_namespace=self._namespace,
        )
        response = self._request(
            WorkerOperation.RETRIEVE,
            {"query": visible.model_dump(mode="json")},
        )
        result = self._response_model(
            response,
            Mem0WorkerRetrievalResultV1,
        )
        if result.query_id != request.query.query_id:
            raise Mem0AdapterError("Mem0 retrieval query ID is inconsistent")
        entries = tuple(self._record(entry) for entry in result.entries)
        trace = RetrievalTraceV3(
            query_id=request.query.query_id,
            retrieved_entries=entries,
            scores=result.scores,
            ranks=tuple(range(1, len(entries) + 1)),
            retrieval_policy="normal_topk",
            context_order="native",
            version_metadata={"score_provenance": "native"},
        )
        return RetrievalResultV3(request=request, trace=trace)

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
        target_identities = {
            object_identity(key) for key in query.target_object_keys
        }
        candidates = tuple(
            entry
            for entry in self.export_entries().entries
            if entry.object_key_candidate is not None
            and object_identity(entry.object_key_candidate)
            in target_identities
            and entry.value_candidate is not None
        )
        if len(target_identities) != 1 or not candidates:
            return AdapterAnswerResultV3(
                prediction=AnswerPredictionV3(
                    query_id=query.query_id,
                    raw_output="",
                    disposition=AnswerDisposition.UNAVAILABLE,
                    parsed_answer=None,
                    format_valid=False,
                    error_flags=("normalized_state_unavailable",),
                )
            )
        sequence_values = tuple(
            entry.raw_metadata.get("mub_sequence_index")
            for entry in candidates
        )
        if all(type(value) is int for value in sequence_values):
            selected = max(
                zip(sequence_values, range(len(candidates)), candidates),
                key=lambda item: (item[0], item[1]),
            )[2]
        else:
            timestamp_values = tuple(
                entry.updated_at or entry.created_at for entry in candidates
            )
            if all(value is not None for value in timestamp_values):
                selected = max(
                    zip(timestamp_values, range(len(candidates)), candidates),
                    key=lambda item: (item[0], item[1]),
                )[2]
            else:
                selected = candidates[-1]
        key = selected.object_key_candidate
        return AdapterAnswerResultV3(
            prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output=json.dumps(
                    selected.value_candidate,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                disposition=AnswerDisposition.ANSWERED,
                parsed_answer=selected.value_candidate,
                cited_event_ids=selected.source_event_ids,
                cited_entry_ids=(selected.entry_id,),
                cited_object_keys=(key,),
                format_valid=True,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        error: Exception | None = None
        try:
            response = self._request(WorkerOperation.CLOSE)
            result = self._response_model(response, Mem0WorkerCloseResultV1)
            if result.closed is not True:
                raise Mem0AdapterError("Mem0 worker close response is invalid")
        except Exception as exc:
            error = exc
        finally:
            self._closed = True
            self._bridge.close()
        if error is not None:
            raise error


__all__ = [
    "MEM0_ADAPTER_VERSION",
    "MEM0_ENTRY_EXTRACTOR_ID",
    "MEM0_ENTRY_EXTRACTOR_VERSION",
    "Mem0AdapterError",
    "Mem0ExternalAdapterV3",
]
