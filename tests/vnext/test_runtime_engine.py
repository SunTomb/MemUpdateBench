from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mub.vnext.contracts import (
    AdapterActionLog,
    AdapterCapabilities,
    AdapterInfo,
    AnswerResult,
    MemoryEntryRecord,
    RetrievalResult,
    ResetResult,
)
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus
from mub.vnext.contracts.runtime import TaskRunRecord
from mub.vnext.runtime.engine import RuntimeConfig, execute_task
from tests.vnext.factories import build_task


@dataclass
class FakeAdapter:
    task: Any
    reset_error: Exception | None = None
    event_error_at: int | None = None
    answer_error: Exception | None = None
    close_error: Exception | None = None
    unsupported: bool = False
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entries: list[MemoryEntryRecord] = []
        self.closed = False
        self.namespace = None

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="fake",
            adapter_version="1.0.0",
            system_name="fake-system",
            system_version="1.0.0",
            configuration_hash="a" * 64,
        )

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_isolated_reset=not self.unsupported,
            supports_event_ingest=not self.unsupported,
            supports_add=True,
            supports_update=True,
            supports_noop=True,
            exports_entries=True,
            exports_raw_state=True,
            exports_object_keys=True,
            exports_values=True,
            exports_retrieval_ids=True,
            exports_retrieval_scores=True,
            supports_native_answer=True,
        )

    def reset(self, namespace: str, config: dict) -> ResetResult:
        self.calls.append(("reset", namespace))
        self.namespace = namespace
        if self.reset_error:
            raise self.reset_error
        if self.unsupported:
            return ResetResult(success=False, namespace=namespace, error={"code": "not_supported"})
        return ResetResult(success=True, namespace=namespace)

    def ingest_event(self, event):
        self.calls.append(("event", event.event_id))
        if self.event_error_at == event.sequence_index:
            raise RuntimeError("event boom")
        action = next(action for action in self.task.gold.actions if action.event_id == event.event_id)
        if action.value is not None:
            key = action.target_object_keys[0]
            entry = MemoryEntryRecord(
                entry_id=f"entry-{event.sequence_index}",
                content=f"{key.canonical_id}={action.value}",
                object_key_candidate=key,
                value_candidate=action.value,
                source_event_ids=[event.event_id],
                version_index=event.sequence_index,
                raw_metadata={"sequence_index": event.sequence_index},
            )
            self.entries = [old for old in self.entries if old.object_key_candidate != key] + [entry]
        return AdapterActionLog(
            event_id=event.event_id,
            requested_operation=action.operation,
            effective_operation=action.operation,
            affected_entry_ids=[entry.entry_id for entry in self.entries],
            raw_action=event.raw_text,
        )

    def export_entries(self) -> list[MemoryEntryRecord]:
        return list(self.entries)

    def export_raw_state(self) -> object:
        return {"state_by_object": {entry.object_key_candidate.canonical_id: entry.value_candidate for entry in self.entries}}

    def retrieve(self, query, k: int) -> RetrievalResult:
        self.calls.append(("retrieve", query.query_id))
        return RetrievalResult(query_id=query.query_id, entries=list(self.entries)[:k], scores=[1.0] * min(k, len(self.entries)))

    def answer(self, query, mode: str) -> AnswerResult:
        self.calls.append(("answer", query.query_id))
        if self.answer_error:
            raise self.answer_error
        value = self.task.gold.gold_answers[query.query_id]
        return AnswerResult(query_id=query.query_id, raw_output=str(value), value=value)

    def close(self) -> None:
        self.calls.append(("close", None))
        self.closed = True
        if self.close_error:
            raise self.close_error


def config(**overrides: Any) -> RuntimeConfig:
    data = {"run_id": "run-test", "retrieval_k": 3, "capture_snapshots": True}
    data.update(overrides)
    return RuntimeConfig(**data)


def test_execute_task_success_captures_actions_snapshots_retrieval_and_answer() -> None:
    task = build_task()
    adapter = FakeAdapter(task)

    row = execute_task(task, adapter, config())

    assert isinstance(row, TaskRunRecord)
    assert row.completion_status is CompletionStatus.COMPLETED
    assert len(row.parsed_actions) == len(task.events)
    assert len(row.memory_snapshots) == len(task.events)
    assert [trace.query_id for trace in row.retrieval_traces] == ["query_0"]
    assert row.answer_predictions[0].parsed_answer == "Qingdao"
    assert adapter.closed
    assert adapter.calls[0][0] == "reset"
    assert adapter.calls[-1][0] == "close"
    assert adapter.calls[0][1] != "run-test"


@pytest.mark.parametrize(
    "kwargs,phase",
    [
        ({"reset_error": RuntimeError("reset boom")}, "reset"),
        ({"event_error_at": 1}, "ingest_event"),
        ({"answer_error": RuntimeError("answer boom")}, "answer"),
        ({"close_error": RuntimeError("close boom")}, "close"),
    ],
)
def test_execute_task_always_returns_typed_error_row(kwargs: dict[str, Any], phase: str) -> None:
    task = build_task()
    adapter = FakeAdapter(task, **kwargs)

    row = execute_task(task, adapter, config())

    assert isinstance(row, TaskRunRecord)
    assert row.exceptions
    assert row.exceptions[-1]["phase"] == phase
    assert row.exceptions[-1]["type"] == "RuntimeError"
    assert adapter.closed
    if phase == "reset":
        assert row.parsed_actions == []
    if phase == "ingest_event":
        assert len(row.parsed_actions) == 1
        assert len(row.memory_snapshots) == 1
    if phase == "answer":
        assert row.retrieval_traces
    if phase == "close":
        assert row.completion_status is not CompletionStatus.COMPLETED


def test_execute_task_close_error_does_not_replace_primary_error() -> None:
    task = build_task()
    row = execute_task(task, FakeAdapter(task, event_error_at=1, close_error=RuntimeError("close boom")), config())

    assert row.exceptions[0]["phase"] == "ingest_event"
    assert [error["phase"] for error in row.exceptions] == ["ingest_event"]


def test_execute_task_classifies_unsupported_reset_without_throwing() -> None:
    task = build_task()
    row = execute_task(task, FakeAdapter(task, unsupported=True), config())

    assert row.completion_status is CompletionStatus.NOT_SUPPORTED
    assert row.exceptions[0]["code"] == "not_supported"
    assert row.parsed_actions == []
