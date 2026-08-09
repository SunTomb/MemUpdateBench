from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.v3.adapter import (
    AdapterActionResultV3,
    AdapterCapabilitiesV3,
    MemoryAdapterV3,
    ResetRequestV3,
    RetrievalRequestV3,
)
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.runtime import (
    MemorySnapshotV3,
    ParserExtractorProvenanceV3,
    TaskRunRecordV3,
)
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.runtime.engine import isolated_namespace
from mub.vnext.runtime.support_v3 import resolve_task_support_v3


@dataclass(frozen=True)
class RuntimeConfigV3:
    run_id: str
    retrieval_policy: str = "normal_topk"
    answer_mode: str = "slot_direct"
    retrieval_k: int = 10
    reset_config: dict[str, JsonValue] = field(default_factory=dict)
    capture_snapshots: bool = True
    action_parser_version: str = "core-visible-action-parser-v1"
    answer_parser_version: str = "core-typed-answer-parser-v1"
    memory_entry_extractor_version: str = "core-entry-extractor-v1"
    object_value_extractor_config_hash: str = "0" * 64
    redaction_policy_version: str = "none-v1"

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("run_id must be a nonblank string")
        if type(self.retrieval_policy) is not str or not self.retrieval_policy.strip():
            raise ValueError("retrieval_policy must be a nonblank string")
        if self.answer_mode not in {"slot_direct", "slot_prompt", "native_answer"}:
            raise ValueError(f"unknown answer mode: {self.answer_mode}")
        if type(self.retrieval_k) is not int or isinstance(self.retrieval_k, bool) or self.retrieval_k <= 0:
            raise ValueError("retrieval_k must be an exact positive integer")


def _capabilities(adapter: MemoryAdapterV3) -> AdapterCapabilitiesV3:
    value = adapter.capabilities()
    return value if isinstance(value, AdapterCapabilitiesV3) else AdapterCapabilitiesV3.model_validate(value)


def _provenance(config: RuntimeConfigV3) -> ParserExtractorProvenanceV3:
    return ParserExtractorProvenanceV3(
        action_parser_version=config.action_parser_version,
        answer_parser_version=config.answer_parser_version,
        memory_entry_extractor_version=config.memory_entry_extractor_version,
        object_value_extractor_config_hash=config.object_value_extractor_config_hash,
        redaction_policy_version=config.redaction_policy_version,
    )


def _snapshot(
    adapter: MemoryAdapterV3,
    event_id: str | None,
    *,
    include_raw_state: bool,
) -> MemorySnapshotV3:
    entries = adapter.export_entries().entries
    raw_state = (
        adapter.export_raw_state().model_dump(mode="json")["raw_state"]
        if include_raw_state
        else None
    )
    state_by_object = {}
    if isinstance(raw_state, dict) and isinstance(raw_state.get("state_by_object"), dict):
        state_by_object = raw_state["state_by_object"]
    payload = {
        "after_event_id": event_id,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "state_by_object": state_by_object,
        "store_size": len(entries),
        "raw_adapter_state": raw_state,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return MemorySnapshotV3(
        after_event_id=event_id,
        entries=entries,
        state_by_object=state_by_object,
        store_size=len(entries),
        raw_adapter_state=raw_state,
        snapshot_hash=digest,
    )


def _terminal_record(
    task: MemUpdateTaskV3,
    adapter_id: str,
    config: RuntimeConfigV3,
    *,
    system_events,
    status: CompletionStatus,
    parsed_actions=(),
    snapshots=(),
    retrievals=(),
    answers=(),
    exceptions=(),
) -> TaskRunRecordV3:
    return TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id=adapter_id,
        run_id=config.run_id,
        parsed_actions=tuple(parsed_actions),
        memory_snapshots=tuple(snapshots),
        retrieval_traces=tuple(retrievals),
        answer_predictions=tuple(answers),
        system_events=tuple(system_events),
        parser_extractor_provenance=_provenance(config),
        exceptions=tuple(exceptions),
        completion_status=status,
    )


def _exception(phase: str, exc: Exception, **extra: Any) -> dict[str, Any]:
    return {
        "phase": phase,
        **extra,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def execute_task_v3(
    task: MemUpdateTaskV3,
    adapter: MemoryAdapterV3,
    run_config: RuntimeConfigV3,
) -> TaskRunRecordV3:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("task must be a MemUpdateTaskV3")
    if not isinstance(run_config, RuntimeConfigV3):
        raise TypeError("run_config must be a RuntimeConfigV3")

    adapter_id = getattr(adapter, "adapter_id", "unknown_adapter")
    system_events = [{
        "event": "runtime_identity",
        "task_id": task.task_id,
        "task_semantic_hash": task.semantic_hash,
        "run_id": run_config.run_id,
        "retrieval_policy": run_config.retrieval_policy,
        "answer_mode": run_config.answer_mode,
    }]
    parsed_actions = []
    snapshots = []
    retrievals = []
    answers = []
    exceptions: list[dict[str, Any]] = []
    status = CompletionStatus.COMPLETED

    try:
        info = adapter.adapter_info()
        adapter_id = info.adapter_id
        capabilities = _capabilities(adapter)
        declared_retrieval_policy = getattr(adapter, "retrieval_policy", None)
        if declared_retrieval_policy is not None and (
            type(declared_retrieval_policy) is not str
            or not declared_retrieval_policy.strip()
        ):
            raise ValueError(
                "adapter retrieval_policy must be a nonblank string when declared"
            )
        retrieval_policy_matched = (
            declared_retrieval_policy is None
            or declared_retrieval_policy == run_config.retrieval_policy
        )
        append_only = bool(getattr(adapter, "append_only_observation", False))
        support = resolve_task_support_v3(
            task,
            capabilities,
            allow_append_only_observation=append_only,
            answer_mode=run_config.answer_mode,
        )
        runtime_support = dict(support.runtime_support)
        runtime_support["retrieval_policy_match"] = retrieval_policy_matched
        missing_capabilities = list(support.missing_capabilities)
        if not retrieval_policy_matched:
            missing_capabilities.append("retrieval_policy_mismatch")
        terminal_supported = (
            support.terminal_supported and retrieval_policy_matched
        )
        system_events.append({
            "event": "retrieval_policy",
            "requested": run_config.retrieval_policy,
            "adapter_declared": declared_retrieval_policy,
            "effective": declared_retrieval_policy,
            "matched": (
                retrieval_policy_matched
                if declared_retrieval_policy is not None
                else None
            ),
        })
        system_events.append({
            "event": "task_support",
            "terminal_supported": terminal_supported,
            "runtime_support": runtime_support,
            "operation_support": dict(support.operation_support),
            "query_support": dict(support.query_support),
            "metric_support": dict(support.metric_support),
            "missing_capabilities": missing_capabilities,
            "append_only_observation": append_only,
        })
        if not run_config.capture_snapshots:
            snapshot_mode = "disabled"
            snapshot_reason = "capture_snapshots_false"
        elif capabilities.exports_entries and capabilities.exports_raw_state:
            snapshot_mode = "full"
            snapshot_reason = None
        elif capabilities.exports_entries:
            snapshot_mode = "entries_only"
            snapshot_reason = "raw_state_export_unavailable"
        else:
            snapshot_mode = "skipped"
            snapshot_reason = "entries_export_required_for_store_size"
        if not terminal_supported:
            status = CompletionStatus.NOT_SUPPORTED
        else:
            system_events.append({
                "event": "snapshot_capture",
                "requested": run_config.capture_snapshots,
                "mode": snapshot_mode,
                "reason": snapshot_reason,
                "exports_entries": capabilities.exports_entries,
                "exports_raw_state": capabilities.exports_raw_state,
            })
            try:
                reset = adapter.reset(ResetRequestV3(
                    namespace=isolated_namespace(run_config.run_id, task.task_id),
                    config=run_config.reset_config,
                ))
            except Exception as exc:
                status = CompletionStatus.FAILED
                exceptions.append(_exception("reset", exc))
            else:
                if not reset.success:
                    not_supported = isinstance(reset.error, Mapping) and reset.error.get("code") == "not_supported"
                    status = CompletionStatus.NOT_SUPPORTED if not_supported else CompletionStatus.FAILED
                    exceptions.append({"phase": "reset", "error": reset.error})

            if status is CompletionStatus.COMPLETED:
                for event in task.events:
                    try:
                        result = adapter.ingest_event(event)
                        if not isinstance(result, AdapterActionResultV3):
                            result = AdapterActionResultV3.model_validate(result)
                        raw_result = result.raw_result
                        action_id = (
                            raw_result.get("parsed_action_id")
                            if isinstance(raw_result, Mapping)
                            else None
                        ) or f"observed_action:{event.event_id}"
                        parsed_actions.append(result.to_parsed_manager_action(
                            action_id=action_id,
                            raw_output=event.raw_text,
                            format_valid=result.execution_status not in {
                                ExecutionStatusV3.REJECTED,
                                ExecutionStatusV3.FAILED,
                            },
                            fallback_used=False,
                        ))
                        if snapshot_mode in {"full", "entries_only"}:
                            snapshots.append(_snapshot(
                                adapter,
                                event.event_id,
                                include_raw_state=(snapshot_mode == "full"),
                            ))
                    except Exception as exc:
                        status = CompletionStatus.PARTIAL if parsed_actions or snapshots else CompletionStatus.FAILED
                        exceptions.append(_exception("ingest_event", exc, event_id=event.event_id))
                        break
                    if result.execution_status in {
                        ExecutionStatusV3.REJECTED,
                        ExecutionStatusV3.FAILED,
                        ExecutionStatusV3.NOT_SUPPORTED,
                    }:
                        status = (
                            CompletionStatus.NOT_SUPPORTED
                            if result.execution_status is ExecutionStatusV3.NOT_SUPPORTED
                            else CompletionStatus.PARTIAL
                        )
                        exceptions.append({
                            "phase": "ingest_event",
                            "event_id": event.event_id,
                            "execution_status": result.execution_status.value,
                            "reason": result.reason,
                            "error": result.error,
                        })
                        break

            if status is CompletionStatus.COMPLETED:
                for query in task.queries:
                    try:
                        retrieval = adapter.retrieve(RetrievalRequestV3(
                            query=query,
                            k=run_config.retrieval_k,
                        ))
                        observed_policy = retrieval.trace.retrieval_policy
                        if declared_retrieval_policy is None:
                            if (
                                type(observed_policy) is not str
                                or not observed_policy.strip()
                            ):
                                policy_reason = "retrieval_policy_unbound"
                            elif observed_policy != run_config.retrieval_policy:
                                policy_reason = "retrieval_policy_mismatch"
                            else:
                                policy_reason = None
                            system_events.append({
                                "event": "retrieval_policy_observed",
                                "query_id": query.query_id,
                                "requested": run_config.retrieval_policy,
                                "effective": observed_policy,
                                "matched": policy_reason is None,
                                "reason": policy_reason,
                            })
                            if policy_reason is not None:
                                status = CompletionStatus.NOT_SUPPORTED
                                break
                        elif (
                            observed_policy is not None
                            and observed_policy != run_config.retrieval_policy
                        ):
                            status = CompletionStatus.NOT_SUPPORTED
                            system_events.append({
                                "event": "retrieval_policy_observed",
                                "query_id": query.query_id,
                                "requested": run_config.retrieval_policy,
                                "effective": observed_policy,
                                "matched": False,
                                "reason": "retrieval_policy_mismatch",
                            })
                            break
                        retrievals.append(retrieval.trace)
                        answer = adapter.answer(query, run_config.answer_mode)
                        answers.append(answer.prediction)
                    except Exception as exc:
                        status = CompletionStatus.PARTIAL
                        exceptions.append(_exception("answer", exc, query_id=query.query_id))
                        break
                    if answer.prediction.disposition.value == "unavailable":
                        status = CompletionStatus.PARTIAL
                        exceptions.append({
                            "phase": "answer",
                            "query_id": query.query_id,
                            "error_flags": list(answer.prediction.error_flags),
                        })
                        break
    except Exception as exc:
        status = CompletionStatus.PARTIAL if parsed_actions or snapshots or retrievals or answers else CompletionStatus.FAILED
        exceptions.append(_exception("adapter_setup", exc))
    finally:
        try:
            adapter.close()
        except Exception as exc:
            if status is CompletionStatus.COMPLETED:
                status = CompletionStatus.PARTIAL
            exceptions.append(_exception("close", exc))

    return _terminal_record(
        task,
        adapter_id,
        run_config,
        system_events=system_events,
        status=status,
        parsed_actions=parsed_actions,
        snapshots=snapshots,
        retrievals=retrievals,
        answers=answers,
        exceptions=exceptions,
    )


def execute_tasks_v3(
    tasks,
    adapter_factory,
    run_config: RuntimeConfigV3,
) -> tuple[TaskRunRecordV3, ...]:
    task_list = tuple(tasks)
    records = []
    for task in task_list:
        try:
            adapter = adapter_factory(task)
        except Exception as exc:
            records.append(_terminal_record(
                task,
                getattr(adapter_factory, "adapter_id", "adapter_factory"),
                run_config,
                system_events=({
                    "event": "runtime_identity",
                    "task_id": task.task_id,
                    "task_semantic_hash": task.semantic_hash,
                    "run_id": run_config.run_id,
                },),
                status=CompletionStatus.FAILED,
                exceptions=(_exception("adapter_factory", exc),),
            ))
            continue
        try:
            records.append(execute_task_v3(task, adapter, run_config))
        except Exception as exc:
            try:
                adapter.close()
            except Exception as close_exc:
                close_error = _exception("close", close_exc)
            else:
                close_error = None
            failures = [_exception("execute_task", exc)]
            if close_error is not None:
                failures.append(close_error)
            records.append(_terminal_record(
                task,
                getattr(adapter, "adapter_id", "unknown_adapter"),
                run_config,
                system_events=({
                    "event": "runtime_identity",
                    "task_id": task.task_id,
                    "task_semantic_hash": task.semantic_hash,
                    "run_id": run_config.run_id,
                },),
                status=CompletionStatus.FAILED,
                exceptions=failures,
            ))

    record_ids = [record.task_id for record in records]
    task_ids = [task.task_id for task in task_list]
    if len(records) != len(task_list) or record_ids != task_ids:
        raise RuntimeError("runtime violated exactly-one ordered terminal-row invariant")
    return tuple(records)


__all__ = ["RuntimeConfigV3", "execute_task_v3", "execute_tasks_v3"]
