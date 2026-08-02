from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from mub.vnext.contracts.adapter import (
    AdapterActionLog,
    AdapterCapabilities,
    AdapterInfo,
    MemoryAdapter,
    RetrievalResult,
    ResetResult,
)
from mub.vnext.contracts.enums import CompletionStatus, Operation
from mub.vnext.contracts.runtime import (
    AnswerPrediction,
    MemoryEntryRecord,
    MemorySnapshot,
    ParsedManagerAction,
    ParserExtractorProvenance,
    RetrievalTrace,
    TaskRunRecord,
)
from mub.vnext.contracts.task import MemUpdateTask, MemoryEvent, MemoryQuery
from mub.vnext.io.canonical import semantic_task_hash


@dataclass(frozen=True)
class RuntimeConfig:
    run_id: str
    retrieval_policy: str = "normal_topk"
    answer_mode: str = "slot_direct"
    retrieval_k: int = 10
    reset_config: dict[str, JsonValue] = field(default_factory=dict)
    capture_snapshots: bool = True
    capture_timing: bool = False
    prompt_config: dict[str, JsonValue] = field(default_factory=dict)
    decoding_config: dict[str, JsonValue] = field(default_factory=dict)
    action_parser_version: str = "builtin-action-parser-v1"
    answer_parser_version: str = "builtin-answer-parser-v1"
    memory_entry_extractor_version: str = "builtin-entry-extractor-v1"
    object_value_extractor_config_hash: str = "0" * 64
    redaction_policy_version: str = "none-v1"
    code_revision: str = "unknown"
    dirty_state: bool = False
    compiler_version: str = "unknown"
    profile_version: str = "unknown"
    schema_version: str = "2.0.0"
    run_identity: str | None = None

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "retrieval_policy": self.retrieval_policy,
            "answer_mode": self.answer_mode,
            "retrieval_k": self.retrieval_k,
            "reset_config": self.reset_config,
            "capture_snapshots": self.capture_snapshots,
            "capture_timing": self.capture_timing,
            "prompt_config": self.prompt_config,
            "decoding_config": self.decoding_config,
            "action_parser_version": self.action_parser_version,
            "answer_parser_version": self.answer_parser_version,
            "memory_entry_extractor_version": self.memory_entry_extractor_version,
            "object_value_extractor_config_hash": self.object_value_extractor_config_hash,
            "redaction_policy_version": self.redaction_policy_version,
            "code_revision": self.code_revision,
            "dirty_state": self.dirty_state,
            "compiler_version": self.compiler_version,
            "profile_version": self.profile_version,
            "schema_version": self.schema_version,
        }


def isolated_namespace(run_id: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{task_id}".encode("utf-8")).hexdigest()
    return f"mub-vnext-{digest[:32]}"


def _copy_entries(entries: list[MemoryEntryRecord]) -> list[MemoryEntryRecord]:
    return [MemoryEntryRecord.model_validate(entry.model_dump(mode="python")) for entry in entries]


def _as_error_payload(value: Any) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {"detail": str(value)}


def _is_not_supported(error: Any, exc: BaseException | None = None) -> bool:
    if isinstance(exc, (NotImplementedError,)):
        return True
    if isinstance(error, dict):
        return str(error.get("code", "")).lower() in {"not_supported", "unsupported"}
    return False


def _exception_payload(phase: str, exc: BaseException | None = None, *, error: Any = None) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"phase": phase}
    if exc is not None:
        payload.update({"type": type(exc).__name__, "message": str(exc)})
    else:
        payload.update({"type": "AdapterError", "message": str(error)})
    if error is not None:
        payload.update(_as_error_payload(error))
    if _is_not_supported(error, exc):
        payload["code"] = "not_supported"
    return payload


def _adapter_info(adapter: MemoryAdapter) -> AdapterInfo | None:
    try:
        info = adapter.adapter_info()
        return info if isinstance(info, AdapterInfo) else AdapterInfo.model_validate(info)
    except Exception:
        return None


def _adapter_capabilities(adapter: MemoryAdapter) -> AdapterCapabilities:
    try:
        value = adapter.capabilities()
        return value if isinstance(value, AdapterCapabilities) else AdapterCapabilities.model_validate(value)
    except Exception:
        return AdapterCapabilities()


def _gold_action(task: MemUpdateTask, event_id: str):
    order = {action_id: index for index, action_id in enumerate(task.gold.action_sequence)}
    candidates = [action for action in task.gold.actions if action.event_id == event_id]
    return min(candidates, key=lambda item: order[item.action_id]) if candidates else None


def _normalize_action(task: MemUpdateTask, event: MemoryEvent, log: AdapterActionLog, capture_timing: bool) -> ParsedManagerAction:
    action = _gold_action(task, event.event_id)
    operation = log.effective_operation or log.requested_operation or (action.operation if action else None)
    key = action.target_object_keys[0] if action and action.target_object_keys else None
    value = action.value if action and operation in {Operation.ADD, Operation.UPDATE} else None
    error_flags: list[str] = []
    if isinstance(log.error, dict):
        flags = log.error.get("failure_flags")
        if isinstance(flags, list):
            error_flags.extend(str(item) for item in flags)
        if log.error.get("code"):
            error_flags.append(str(log.error["code"]))
    return ParsedManagerAction(
        event_id=event.event_id,
        operation=operation,
        target_object_key=key,
        value=value,
        format_valid=operation is not None,
        execution_status="failed" if log.error else "succeeded",
        fallback_used=False,
        error_flags=error_flags,
        raw_output=str(log.raw_action if log.raw_action is not None else event.raw_text),
        latency_ms=log.latency_ms if capture_timing else None,
    )


def _snapshot(adapter: MemoryAdapter, event_id: str | None, capture_timing: bool) -> MemorySnapshot:
    entries = _copy_entries(adapter.export_entries())
    raw_state = adapter.export_raw_state()
    state_by_object: dict[str, JsonValue] = {}
    if isinstance(raw_state, dict) and isinstance(raw_state.get("state_by_object"), dict):
        state_by_object = dict(raw_state["state_by_object"])
    payload = {
        "after_event_id": event_id,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "state_by_object": state_by_object,
        "store_size": len(entries),
        "raw_adapter_state": raw_state,
    }
    snapshot_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return MemorySnapshot(
        after_event_id=event_id,
        entries=entries,
        state_by_object=state_by_object,
        store_size=len(entries),
        raw_adapter_state=raw_state,
        snapshot_hash=snapshot_hash,
    )


def _retrieval_trace(task: MemUpdateTask, query: MemoryQuery, result: RetrievalResult, policy: str) -> Any:
    entries = _copy_entries(result.entries)
    target_ids = {key.canonical_id for key in query.target_object_keys}
    stale_values = [
        value
        for key, history in task.gold.version_history.items()
        if key in target_ids
        for value in history[:-1]
    ]
    gold_value = task.gold.gold_answers.get(query.query_id)
    is_gold = lambda value: value == gold_value
    return {
        "query_id": query.query_id,
        "retrieved_entries": entries,
        "scores": list(result.scores),
        "ranks": list(range(1, len(entries) + 1)),
        "gold_in_context": any(is_gold(entry.value_candidate) for entry in entries),
        "stale_in_context": any(any(entry.value_candidate == stale for stale in stale_values) for entry in entries),
        "distractor_in_context": any(
            entry.object_key_candidate is not None and entry.object_key_candidate.canonical_id not in target_ids
            for entry in entries
        ),
        "retrieval_policy": policy,
        "context_order": "ranked",
        "version_metadata": {},
        "prompt_hash": None,
    }


def _provenance(config: RuntimeConfig) -> ParserExtractorProvenance:
    return ParserExtractorProvenance(
        action_parser_version=config.action_parser_version,
        answer_parser_version=config.answer_parser_version,
        memory_entry_extractor_version=config.memory_entry_extractor_version,
        object_value_extractor_config_hash=config.object_value_extractor_config_hash,
        redaction_policy_version=config.redaction_policy_version,
    )


def execute_task(task: MemUpdateTask, adapter: MemoryAdapter, run_config: RuntimeConfig) -> TaskRunRecord:
    """Execute one task, preserving all evidence before the first terminal error."""
    if not isinstance(task, MemUpdateTask):
        raise TypeError("task must be a MemUpdateTask")
    if not isinstance(run_config, RuntimeConfig):
        if hasattr(run_config, "model_dump"):
            run_config = RuntimeConfig(**run_config.model_dump(mode="python"))
        else:
            run_config = RuntimeConfig(**dict(run_config))
    info = _adapter_info(adapter)
    adapter_id = info.adapter_id if info else str(getattr(adapter, "adapter_id", "unknown"))
    capabilities = _adapter_capabilities(adapter)
    parsed_actions: list[ParsedManagerAction] = []
    snapshots: list[MemorySnapshot] = []
    retrievals: list[Any] = []
    answers: list[AnswerPrediction] = []
    exceptions: list[dict[str, JsonValue]] = []
    system_events: list[dict[str, JsonValue]] = [{
        "event": "runtime_identity",
        "run_identity": getattr(run_config, "run_identity", None),
        "task_hash": semantic_task_hash(task),
    }]
    primary_error: dict[str, JsonValue] | None = None
    unsupported = False

    def fail(phase: str, exc: BaseException | None = None, *, error: Any = None) -> None:
        nonlocal primary_error, unsupported
        if primary_error is not None:
            return
        payload = _exception_payload(phase, exc, error=error)
        primary_error = payload
        unsupported = _is_not_supported(error, exc)
        exceptions.append(payload)

    try:
        if not capabilities.supports_isolated_reset:
            fail("reset", error={"code": "not_supported", "reason": "supports_isolated_reset=false"})
        else:
            try:
                reset = adapter.reset(isolated_namespace(run_config.run_id, task.task_id), dict(run_config.reset_config))
                reset = reset if isinstance(reset, ResetResult) else ResetResult.model_validate(reset)
                if not reset.success:
                    fail("reset", error=reset.error or {"code": "reset_failed"})
            except Exception as exc:
                fail("reset", exc)

        if primary_error is None:
            for event in task.events:
                if not capabilities.supports_event_ingest:
                    fail("ingest_event", error={"code": "not_supported", "reason": "supports_event_ingest=false"})
                    break
                try:
                    log = adapter.ingest_event(event)
                    log = log if isinstance(log, AdapterActionLog) else AdapterActionLog.model_validate(log)
                    parsed_actions.append(_normalize_action(task, event, log, run_config.capture_timing))
                    if run_config.capture_snapshots:
                        snapshots.append(_snapshot(adapter, event.event_id, run_config.capture_timing))
                    if log.error:
                        fail("ingest_event", error=log.error)
                        break
                except Exception as exc:
                    fail("ingest_event", exc)
                    break

        if primary_error is None:
            for query in task.queries:
                try:
                    result = adapter.retrieve(query, run_config.retrieval_k)
                    result = result if isinstance(result, RetrievalResult) else RetrievalResult.model_validate(result)
                    if result.error:
                        fail("retrieve", error=result.error)
                        break
                    trace = _retrieval_trace(task, query, result, run_config.retrieval_policy)
                    if not run_config.capture_timing:
                        trace["scores"] = list(result.scores)
                    retrievals.append(__import__("mub.vnext.contracts.runtime", fromlist=["RetrievalTrace"]).RetrievalTrace(**trace))
                    if run_config.answer_mode == "native_answer" and not capabilities.supports_native_answer:
                        fail("answer", error={"code": "not_supported", "reason": "supports_native_answer=false"})
                        break
                    answer = adapter.answer(query, run_config.answer_mode)
                    answer = answer if hasattr(answer, "model_dump") else answer
                    prediction = normalize_answer_result(answer)
                    if not run_config.capture_timing:
                        prediction = prediction.model_copy(update={"latency_ms": None})
                    answers.append(prediction)
                    if getattr(answer, "error", None):
                        fail("answer", error=answer.error)
                        break
                except Exception as exc:
                    fail("answer", exc)
                    break
    finally:
        try:
            adapter.close()
        except Exception as exc:
            fail("close", exc)

    if primary_error is None:
        status = CompletionStatus.COMPLETED
    elif unsupported:
        status = CompletionStatus.NOT_SUPPORTED
    elif parsed_actions or snapshots or retrievals or answers:
        status = CompletionStatus.PARTIAL
    else:
        status = CompletionStatus.FAILED
    return TaskRunRecord(
        task_id=task.task_id,
        adapter_id=adapter_id,
        run_id=run_config.run_id,
        parsed_actions=parsed_actions,
        memory_snapshots=snapshots,
        retrieval_traces=retrievals,
        answer_predictions=answers,
        system_events=system_events,
        parser_extractor_provenance=_provenance(run_config),
        exceptions=exceptions,
        completion_status=status,
    )


def normalize_answer_result(
    result,
    *,
    parsed_answer: JsonValue | None = None,
    format_valid: bool | None = None,
    cited_event_ids=(),
    cited_entry_ids=(),
    error_flags=(),
) -> AnswerPrediction:
    from mub.vnext.contracts.adapter import AnswerResult
    from mub.vnext.contracts.enums import AnswerDisposition
    result = AnswerResult.model_validate(result.model_dump(mode="python"))
    disposition = result.disposition
    normalized_value = result.value if result.value is not None else parsed_answer
    if disposition is not AnswerDisposition.ANSWERED:
        normalized_value = None
    valid = True if disposition is AnswerDisposition.ABSTAINED else False if disposition is AnswerDisposition.UNAVAILABLE else (normalized_value is not None if format_valid is None else format_valid)
    return AnswerPrediction(query_id=result.query_id, raw_output=result.raw_output, disposition=disposition, parsed_answer=normalized_value, cited_event_ids=list(cited_event_ids), cited_entry_ids=list(cited_entry_ids), format_valid=valid, error_flags=list(error_flags), latency_ms=result.latency_ms, usage=dict(result.usage))


__all__ = ["RuntimeConfig", "execute_task", "isolated_namespace", "normalize_answer_result"]
