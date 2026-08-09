from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from mub.vnext.contracts.common import thaw_json
from mub.vnext.contracts.enums import AnswerDisposition, ActionScope, Operation
from mub.vnext.contracts.v3.adapter import (
    AdapterActionPayloadV3,
    AdapterActionResultV3,
    AdapterAnswerResultV3,
    AdapterCapabilitiesV3,
    AdapterInfoV3,
    ExportEntriesResultV3,
    ExportStateResultV3,
    ExportedEventAnchorV3,
    ExportedVersionRecordV3,
    ObjectVersionHistoryV3,
    ResetRequestV3,
    ResetResultV3,
    RetrievalRequestV3,
    RetrievalResultV3,
    VersionHistoryExportRequestV3,
    VersionHistoryExportResultV3,
)
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus
from mub.vnext.contracts.v3.runtime import (
    AnswerPredictionV3,
    MemoryEntryRecordV3,
    RetrievalTraceV3,
)
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, MemoryEventV3, MemoryQueryV3
from mub.vnext.runtime.support_v3 import ObservedActionV3, parse_visible_action_v3
from mub.vnext.validation.replay_v3 import (
    ReplayLedgerV3,
    ReplayResultV3,
    ReplayVersionV3,
    resolve_query_v3,
)


def _config_hash(adapter_id: str, config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"adapter_id": adapter_id, "config": config},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    value = thaw_json(value)
    return value if isinstance(value, str) else json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _entry_content(key, value, *, kind: str = "value") -> str:
    if kind == "delete_instruction":
        return f"DELETE {key.canonical_id}"
    return f"{key.canonical_id} = {_json_text(value)}"


class CoreBuiltinAdapterV3:
    adapter_id = "core_builtin"
    adapter_version = "1.0.0"
    system_name = "mub_vnext_core_builtin"
    system_version = "1.0.0"
    capabilities_config: dict[str, bool] = {}
    append_only_observation = False

    def __init__(self, task: MemUpdateTaskV3, *, retrieval_policy: str = "normal_topk") -> None:
        if not isinstance(task, MemUpdateTaskV3):
            raise TypeError("task must be a MemUpdateTaskV3")
        if retrieval_policy not in {"normal_topk", "latest_per_object"}:
            raise ValueError(f"unknown retrieval policy: {retrieval_policy}")
        self.retrieval_policy = retrieval_policy
        self._target_objects = tuple(task.target_objects)
        self._events = tuple(task.events)
        self._queries = tuple(task.queries)
        self._events_by_id = {event.event_id: event for event in self._events}
        self._entries: list[MemoryEntryRecordV3] = []
        self._state: dict[str, MemoryEntryRecordV3] = {}
        self._versions: dict[str, list[ReplayVersionV3]] = {
            key.canonical_id: [] for key in self._target_objects
        }
        self._pending_ttl: list[tuple[str, Any]] = []
        self._action_trace: list[dict[str, Any]] = []
        self._namespace: str | None = None
        self._closed = False
        self._ready = True
        self._startup_error: dict[str, Any] | None = None

    def _info_config(self) -> dict[str, Any]:
        return {"retrieval_policy": self.retrieval_policy}

    def adapter_info(self) -> AdapterInfoV3:
        return AdapterInfoV3(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            system_name=self.system_name,
            system_version=self.system_version,
            configuration_hash=_config_hash(self.adapter_id, self._info_config()),
        )

    def capabilities(self) -> AdapterCapabilitiesV3:
        return AdapterCapabilitiesV3.model_validate(self.capabilities_config)

    def reset(self, request: ResetRequestV3) -> ResetResultV3:
        if not isinstance(request, ResetRequestV3):
            request = ResetRequestV3.model_validate(request)
        self._namespace = request.namespace
        self._entries.clear()
        self._state.clear()
        self._versions = {key.canonical_id: [] for key in self._target_objects}
        self._pending_ttl.clear()
        self._action_trace.clear()
        self._closed = False
        if not self._ready:
            return ResetResultV3(
                success=False,
                namespace=request.namespace,
                error=self._startup_error or {"code": "not_supported"},
            )
        return ResetResultV3(success=True, namespace=request.namespace)

    def _observed_action(self, event: MemoryEventV3):
        if self._namespace is None:
            raise RuntimeError("reset required")
        trusted = self._events_by_id.get(event.event_id)
        if trusted is None or event != trusted:
            raise ValueError("event is not bound to the adapter task")
        return parse_visible_action_v3(event, self._target_objects)

    def _append_version(self, action, key, status, value, logical_time, boundary_event):
        versions = self._versions[key.canonical_id]
        if versions and boundary_event is not None:
            versions[-1] = versions[-1].model_copy(
                update={"valid_until_event_id": boundary_event}
            )
        version = ReplayVersionV3(
            object_key=key,
            version_index=len(versions),
            status=status,
            value=value,
            source_action_id=action.action_id,
            source_event_ids=(action.event_id,),
            logical_time=logical_time,
            valid_from_event_id=boundary_event,
        )
        versions.append(version)
        return version

    def _write_entry(
        self,
        event,
        action,
        key,
        version=None,
        *,
        append: bool,
        kind: str = "value",
    ):
        old = self._state.get(key.canonical_id)
        entry = MemoryEntryRecordV3(
            entry_id=(
                f"{self.adapter_id}:{self._namespace}:{event.sequence_index}:{len(self._entries)}"
                if append
                else f"{self.adapter_id}:{self._namespace}:{key.canonical_id}"
            ),
            content=_entry_content(key, action.value, kind=kind),
            object_key_candidate=key,
            value_candidate=action.value,
            created_at=(event.timestamp if append or old is None else old.created_at),
            updated_at=event.timestamp,
            source_event_ids=(
                (event.event_id,)
                if append or old is None
                else (*old.source_event_ids, event.event_id)
            ),
            version_index=None if version is None else version.version_index,
            raw_metadata={
                "sequence_index": event.sequence_index,
                "operation": action.operation.value,
                "entry_kind": kind,
                "effective_at": action.effective_at,
            },
        )
        if append or old is None:
            self._entries.append(entry)
        else:
            self._entries[self._entries.index(old)] = entry
        if kind == "value":
            self._state[key.canonical_id] = entry
        return entry

    def _expire_due(self, logical_time: str | None) -> None:
        if logical_time is None:
            return
        due = [item for item in self._pending_ttl if item[0] <= logical_time]
        for effective_at, action in sorted(due, key=lambda item: item[0]):
            for key in action.target_object_keys:
                if not self.append_only_observation:
                    old = self._state.pop(key.canonical_id, None)
                    if old is not None and old in self._entries:
                        self._entries.remove(old)
                self._append_version(
                    action,
                    key,
                    LedgerEntryStatus.TOMBSTONE,
                    None,
                    effective_at,
                    None,
                )
            self._pending_ttl.remove((effective_at, action))

    def _apply_exact(self, event, action) -> tuple[str, ...]:
        self._expire_due(event.timestamp)
        affected: list[str] = []
        if action.operation is Operation.NOOP:
            return ()
        if action.operation is Operation.DELETE and action.scope is ActionScope.TTL:
            self._pending_ttl.append((action.effective_at, action))
            return tuple(
                self._state[key.canonical_id].entry_id
                for key in action.target_object_keys
                if key.canonical_id in self._state
            )
        for key in action.target_object_keys:
            if action.operation in {Operation.ADD, Operation.UPDATE}:
                version = self._append_version(
                    action,
                    key,
                    LedgerEntryStatus.PRESENT,
                    action.value,
                    action.effective_at or event.timestamp,
                    event.event_id,
                )
                affected.append(
                    self._write_entry(event, action, key, version, append=False).entry_id
                )
            elif action.operation is Operation.DELETE:
                old = self._state.pop(key.canonical_id, None)
                if old is not None:
                    affected.append(old.entry_id)
                    if old in self._entries:
                        self._entries.remove(old)
                self._append_version(
                    action,
                    key,
                    LedgerEntryStatus.TOMBSTONE,
                    None,
                    action.effective_at or event.timestamp,
                    event.event_id,
                )
        return tuple(affected)

    def _apply_append(self, event, action) -> tuple[str, ...]:
        self._expire_due(event.timestamp)
        if action.operation is Operation.NOOP:
            return ()
        if action.operation is Operation.DELETE:
            instruction_ids = tuple(
                self._write_entry(
                    event,
                    action,
                    key,
                    append=True,
                    kind="delete_instruction",
                ).entry_id
                for key in action.target_object_keys
            )
            if action.scope is ActionScope.TTL:
                self._pending_ttl.append((action.effective_at, action))
            else:
                for key in action.target_object_keys:
                    self._append_version(
                        action,
                        key,
                        LedgerEntryStatus.TOMBSTONE,
                        None,
                        action.effective_at or event.timestamp,
                        event.event_id,
                    )
            return instruction_ids
        affected: list[str] = []
        for key in action.target_object_keys:
            version = self._append_version(
                action,
                key,
                LedgerEntryStatus.PRESENT,
                action.value,
                action.effective_at or event.timestamp,
                event.event_id,
            )
            affected.append(
                self._write_entry(event, action, key, version, append=True).entry_id
            )
        return tuple(affected)

    def ingest_event(self, event: MemoryEventV3) -> AdapterActionResultV3:
        if self._closed:
            raise RuntimeError("adapter closed")
        action = self._observed_action(event)
        internal_entry_ids = (
            self._apply_append(event, action)
            if self.append_only_observation
            else self._apply_exact(event, action)
        )
        requested = AdapterActionPayloadV3(
            operation=action.operation,
            scope=action.scope,
            target_object_keys=action.target_object_keys,
            value=action.value,
        )
        effective = requested
        execution_status = ExecutionStatusV3.EXECUTED
        reason = None
        affected_entry_ids = internal_entry_ids
        if self.append_only_observation and action.operation is Operation.DELETE:
            effective = AdapterActionPayloadV3(operation=Operation.NOOP)
            execution_status = ExecutionStatusV3.NO_EFFECT
            reason = "append_only_no_physical_delete"
            affected_entry_ids = ()
        self._action_trace.append({
            "event_id": event.event_id,
            "requested_operation": action.operation.value,
            "effective_operation": effective.operation.value,
            "execution_status": execution_status.value,
            "target_object_keys": [key.canonical_id for key in action.target_object_keys],
            "affected_entry_ids": list(affected_entry_ids),
            "instruction_entry_ids": (
                list(internal_entry_ids)
                if self.append_only_observation and action.operation is Operation.DELETE
                else []
            ),
        })
        return AdapterActionResultV3(
            event_id=event.event_id,
            requested_action=requested,
            effective_action=effective,
            execution_status=execution_status,
            reason=reason,
            affected_entry_ids=affected_entry_ids,
            raw_result={
                "surface_text": event.raw_text,
                "parsed_action_id": action.action_id,
            },
        )

    def export_entries(self) -> ExportEntriesResultV3:
        return ExportEntriesResultV3(entries=tuple(self._entries))

    def export_raw_state(self) -> ExportStateResultV3:
        return ExportStateResultV3(raw_state={
            "namespace": self._namespace,
            "entries": [entry.model_dump(mode="json") for entry in self._entries],
            "state_by_object": {
                object_id: thaw_json(entry.value_candidate)
                for object_id, entry in self._state.items()
            },
            "action_trace": list(self._action_trace),
        })

    def _history_result(
        self,
        object_ids: set[str] | None = None,
    ) -> VersionHistoryExportResultV3:
        event_positions = {event.event_id: event.sequence_index for event in self._events}
        event_times = {event.event_id: event.timestamp for event in self._events}
        histories = []
        for key in self._target_objects:
            if object_ids is not None and key.canonical_id not in object_ids:
                continue
            versions = self._versions[key.canonical_id]
            if not versions:
                continue
            logical_only = any(version.valid_from_event_id is None for version in versions)
            exported = []
            for version in versions:
                def anchor(event_id, *, logical_time=None):
                    if event_id is None:
                        return None
                    return ExportedEventAnchorV3(
                        event_id=event_id,
                        sequence_index=event_positions[event_id],
                        logical_time=(
                            event_times[event_id]
                            if logical_time is None
                            else logical_time
                        ),
                    )

                source_anchors = tuple(
                    anchor(
                        event_id,
                        logical_time=(version.logical_time if logical_only else None),
                    )
                    for event_id in version.source_event_ids
                )
                exported.append(ExportedVersionRecordV3(
                    version_index=version.version_index,
                    status=version.status,
                    value=version.value,
                    valid_from=(
                        None if logical_only else anchor(version.valid_from_event_id)
                    ),
                    valid_until=(
                        None if logical_only else anchor(version.valid_until_event_id)
                    ),
                    logical_time=version.logical_time,
                    source_anchors=source_anchors,
                ))
            histories.append(
                ObjectVersionHistoryV3(object_key=key, versions=tuple(exported))
            )
        return VersionHistoryExportResultV3(histories=tuple(histories))

    def export_version_history(
        self,
        request: VersionHistoryExportRequestV3,
    ) -> VersionHistoryExportResultV3:
        if not self.capabilities().exports_version_history:
            return VersionHistoryExportResultV3()
        object_ids = {key.canonical_id for key in request.object_keys} or None
        return self._history_result(object_ids)

    def retrieve(self, request: RetrievalRequestV3) -> RetrievalResultV3:
        query = request.query
        targets = {key.canonical_id for key in query.target_object_keys}
        candidates = [
            entry
            for entry in reversed(self._entries)
            if entry.object_key_candidate is not None
            and entry.object_key_candidate.canonical_id in targets
        ]
        if self.retrieval_policy == "latest_per_object":
            deduped = []
            seen = set()
            for entry in candidates:
                object_id = entry.object_key_candidate.canonical_id
                if object_id not in seen:
                    deduped.append(entry)
                    seen.add(object_id)
            candidates = deduped
        entries = tuple(candidates[: request.k])
        return RetrievalResultV3(
            request=request,
            trace=RetrievalTraceV3(
                query_id=query.query_id,
                retrieved_entries=entries,
                scores=tuple(float(len(entries) - index) for index in range(len(entries))),
                ranks=tuple(range(1, len(entries) + 1)),
                retrieval_policy=self.retrieval_policy,
                context_order="reverse_adapter_order",
            ),
        )

    def _observed_replay(self, query: MemoryQueryV3) -> ReplayResultV3:
        horizon = max(
            (
                value
                for value in (
                    *(event.timestamp for event in self._events),
                    getattr(query.selector, "logical_time", None),
                )
                if value is not None
            ),
            default=None,
        )
        self._expire_due(horizon)
        ledgers = tuple(
            ReplayLedgerV3(
                object_key=key,
                versions=tuple(self._versions[key.canonical_id]),
            )
            for key in self._target_objects
        )
        current = {}
        for key in self._target_objects:
            versions = self._versions[key.canonical_id]
            if versions and versions[-1].status is LedgerEntryStatus.PRESENT:
                current[key.canonical_id] = versions[-1]
        return ReplayResultV3(
            current_state=current,
            ledgers=ledgers,
            expected_present=tuple(
                key for key in self._target_objects if key.canonical_id in current
            ),
            expected_absent=tuple(
                key for key in self._target_objects if key.canonical_id not in current
            ),
            protected_collateral=(),
            mutation_count=sum(len(versions) for versions in self._versions.values()),
        )

    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3:
        if mode not in {"slot_direct", "slot_prompt", "native_answer"}:
            return AdapterAnswerResultV3(prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output="",
                disposition=AnswerDisposition.UNAVAILABLE,
                format_valid=False,
                error_flags=("answer_mode_not_supported",),
            ))
        resolution = resolve_query_v3(query, self._observed_replay(query), self._events)
        if resolution.issues:
            return AdapterAnswerResultV3(prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output="",
                disposition=AnswerDisposition.UNAVAILABLE,
                format_valid=False,
                error_flags=tuple(issue.code for issue in resolution.issues),
            ))
        disposition = resolution.disposition or AnswerDisposition.ANSWERED
        answer = resolution.answer if disposition is AnswerDisposition.ANSWERED else None
        return AdapterAnswerResultV3(prediction=AnswerPredictionV3(
            query_id=query.query_id,
            raw_output=(
                "ABSTAIN"
                if disposition is AnswerDisposition.ABSTAINED
                else _json_text(answer)
            ),
            disposition=disposition,
            parsed_answer=answer,
            format_valid=True,
        ))

    def close(self) -> None:
        self._closed = True


class ReferenceAdapterV3(CoreBuiltinAdapterV3):
    adapter_id = "reference"
    system_name = "oracle_smoke_only"
    capabilities_config = {
        "supports_isolated_reset": True,
        "supports_event_ingest": True,
        "supports_add": True,
        "supports_update": True,
        "supports_noop": True,
        "supports_delete": True,
        "supports_ttl": True,
        "supports_native_answer": True,
        "supports_scoped_delete": True,
        "supports_historical_query": True,
        "exports_version_history": True,
        "supports_multi_object_query": True,
        "exports_evidence_linkage": True,
        "exports_entries": True,
        "exports_raw_state": True,
        "exports_source_event_ids": True,
        "exports_timestamps_or_order": True,
        "exports_object_keys": True,
        "exports_values": True,
        "exports_retrieval_ids": True,
        "exports_retrieval_scores": True,
        "exports_action_trace": True,
    }

    def __init__(self, task: MemUpdateTaskV3, *, retrieval_policy: str = "normal_topk") -> None:
        self._reference_task = MemUpdateTaskV3.model_validate(
            task.model_dump(mode="python")
        )
        self._reference_actions = {
            action.action_id: action for action in self._reference_task.actions
        }
        super().__init__(task, retrieval_policy=retrieval_policy)

    def _observed_action(self, event: MemoryEventV3):
        if self._namespace is None:
            raise RuntimeError("reset required")
        trusted = self._events_by_id.get(event.event_id)
        if trusted is None or event != trusted:
            raise ValueError("event is not bound to the adapter task")
        actions = tuple(
            self._reference_actions[action_id]
            for action_id in event.gold_action_ids
        )
        if len(actions) != 1:
            raise ValueError("Reference requires exactly one gold action per event")
        return actions[0]

    def answer(self, query: MemoryQueryV3, mode: str) -> AdapterAnswerResultV3:
        evidence = next(
            item
            for item in self._reference_task.gold_evidence
            if item.query_id == query.query_id
        )
        disposition = evidence.disposition or AnswerDisposition.ANSWERED
        answer = evidence.answer if disposition is AnswerDisposition.ANSWERED else None
        return AdapterAnswerResultV3(prediction=AnswerPredictionV3(
            query_id=query.query_id,
            raw_output=(
                "ABSTAIN"
                if disposition is AnswerDisposition.ABSTAINED
                else _json_text(answer)
            ),
            disposition=disposition,
            parsed_answer=answer,
            cited_event_ids=evidence.supporting_event_ids,
            cited_object_keys=evidence.supporting_object_keys,
            cited_derivation_step_ids=tuple(
                step.step_id for step in evidence.derivation_steps
            ),
            format_valid=True,
        ))


class RawAppendAdapterV3(CoreBuiltinAdapterV3):
    adapter_id = "raw_add"
    system_name = "raw_append"
    append_only_observation = True
    capabilities_config = {
        **ReferenceAdapterV3.capabilities_config,
        "supports_delete": False,
        "supports_ttl": False,
        "supports_scoped_delete": False,
        "exports_evidence_linkage": False,
    }


class ExactCrudAdapterV3(CoreBuiltinAdapterV3):
    adapter_id = "exact_crud"
    system_name = "exact_object_crud"
    capabilities_config = {
        **ReferenceAdapterV3.capabilities_config,
        "supports_historical_query": False,
        "exports_version_history": False,
        "exports_evidence_linkage": False,
    }


class HeuristicCrudAdapterV3(ExactCrudAdapterV3):
    adapter_id = "heuristic_crud"

    def __init__(
        self,
        task: MemUpdateTaskV3,
        *,
        encoder: Any | None = None,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision: str = "unverified",
        backend: str = "sentence_transformers",
        retrieval_policy: str = "normal_topk",
    ) -> None:
        self.encoder = encoder
        self.encoder_model = encoder_model
        self.encoder_revision = encoder_revision
        self.backend = backend
        super().__init__(task, retrieval_policy=retrieval_policy)
        self._ready = self._verify_encoder()
        if not self._ready:
            self._startup_error = {
                "code": "not_supported",
                "reason": "verified_minilm_required",
            }

    def _verify_encoder(self) -> bool:
        if (
            self.encoder is None
            or self.encoder_model != "sentence-transformers/all-MiniLM-L6-v2"
            or self.backend != "sentence_transformers"
            or not isinstance(self.encoder_revision, str)
            or not self.encoder_revision.strip()
            or self.encoder_revision == "unverified"
        ):
            return False
        try:
            values = (
                self.encoder.encode(
                    ["memupdatebench capability probe"],
                    normalize_embeddings=True,
                )
                if hasattr(self.encoder, "encode")
                else self.encoder(
                    ["memupdatebench capability probe"],
                    normalize_embeddings=True,
                )
            )
            row = [float(value) for value in values[0]]
            return bool(row) and all(math.isfinite(value) for value in row) and any(row)
        except Exception:
            return False

    def _info_config(self) -> dict[str, Any]:
        return {
            **super()._info_config(),
            "encoder_model": self.encoder_model,
            "encoder_revision": self.encoder_revision,
            "backend": self.backend,
            "verified": self._ready,
        }


__all__ = [
    "CoreBuiltinAdapterV3",
    "ExactCrudAdapterV3",
    "HeuristicCrudAdapterV3",
    "RawAppendAdapterV3",
    "ReferenceAdapterV3",
]
