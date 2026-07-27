from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.common import MetricFieldSupport
from mub.vnext.contracts.enums import (
    AnswerDisposition,
    CompletionStatus,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
    SupportReason,
)
from mub.vnext.contracts.manifest import (
    ANSWER_NORMALIZATION_PROFILE,
    ScorerConfig,
)
from mub.vnext.contracts.runtime import AnswerPrediction, RetrievalTrace, TaskRunRecord
from mub.vnext.contracts.score import SCORE_LAYER_TYPES, ScoreRecord
from mub.vnext.contracts.task import GoldAction, MemUpdateTask, MemoryQuery
from mub.vnext.scoring.failures import canonicalize_failure_flags, primary_failure
from mub.vnext.scoring.registry import (
    EXTRACTOR_LINKAGE_METRIC_PATHS,
    METRIC_REGISTRY,
    RUNTIME_NULL_POLICY,
    metric_applies_to_family,
    missing_capabilities,
)
from mub.vnext.version import RUNTIME_RECORD_VERSION, SCHEMA_VERSION, SCORER_VERSION

_NULL_POLICY = "exclude_from_aggregation"
_OBSERVABILITY_LAYERS = frozenset({"protocol_scores", "audit_scores"})
_TARGET_DEPENDENT_ACTION_PATHS = frozenset(
    {
        "action_scores.full_action_exact_match",
        "action_scores.object_key_accuracy",
        "action_scores.entity_accuracy",
        "action_scores.attribute_accuracy",
        "action_scores.wrong_object_write_rate",
    }
)


_CORRECTNESS_ONE_PATHS = frozenset(
    {
        "protocol_scores.action_parse_valid",
        "protocol_scores.answer_parse_valid",
        "protocol_scores.execution_success_rate",
        "action_scores.operation_accuracy",
        "action_scores.full_action_exact_match",
        "action_scores.object_key_accuracy",
        "action_scores.entity_accuracy",
        "action_scores.attribute_accuracy",
        "action_scores.value_accuracy",
        "state_scores.final_state_accuracy",
        "state_scores.state_precision",
        "state_scores.state_recall",
        "state_scores.state_f1",
        "state_scores.state_resolve_rate",
        "state_scores.expected_absence_accuracy",
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
        "answer_scores.exact_match",
        "answer_scores.normalized_match",
        "answer_scores.reference_resolution_accuracy",
        "answer_scores.token_f1",
        "answer_scores.structured_field_accuracy",
        "answer_scores.answer_state_consistency",
    }
)
_CORRECTNESS_ZERO_PATHS = frozenset(
    {
        "protocol_scores.unsupported_operation_rate",
        "protocol_scores.fallback_rate",
        "action_scores.false_write_rate",
        "action_scores.missed_write_rate",
        "action_scores.wrong_object_write_rate",
        "state_scores.collateral_corruption_rate",
        "system_scores.error_rate",
    }
)


class _StoreAnalysisUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class _StoreAnalysis:
    snapshot: Any
    obsolete: int
    stale_conflicting: int | None
    duplicate_current: int | None
    value_detail: str | None


def _revalidate(model: BaseModel, model_type):
    if not isinstance(model, model_type):
        raise TypeError(f"expected {model_type.__name__}")
    return model_type.model_validate(model.model_dump(mode="python", warnings=False))


def _support(reason: SupportReason, detail: str, null_policy: str = _NULL_POLICY):
    return MetricFieldSupport(reason=reason, null_policy=null_policy, detail=detail)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _same_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _normalize_answer(value: Any, profile: str) -> Any:
    if profile != ANSWER_NORMALIZATION_PROFILE:
        raise ValueError(f"unsupported answer normalization profile: {profile}")
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, list):
        return ("list", tuple(_normalize_answer(item, profile) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_normalize_answer(item, profile) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (key, _normalize_answer(item, profile))
                for key, item in sorted(value.items())
            ),
        )
    return value


def _accepted_values(task: MemUpdateTask, query_id: str) -> tuple[Any, ...]:
    gold = task.gold.gold_answers[query_id]
    acceptable = task.gold.acceptable_answers[query_id]
    if isinstance(acceptable, list):
        values = (gold, *acceptable)
    else:
        values = (gold, acceptable)
    unique: list[Any] = []
    for value in values:
        if not any(_same_value(value, present) for present in unique):
            unique.append(value)
    return tuple(unique)


def _current_queries(task: MemUpdateTask) -> tuple[MemoryQuery, ...]:
    return tuple(query for query in task.queries if query.query_type is QueryType.CURRENT_STATE)


def _reference_queries(task: MemUpdateTask) -> tuple[MemoryQuery, ...]:
    return tuple(
        query
        for query in task.queries
        if query.query_type is QueryType.UNRESOLVED_REFERENCE
    )


def _actions_by_event(run: TaskRunRecord):
    grouped: dict[str, list[Any]] = {}
    for action in run.parsed_actions:
        grouped.setdefault(action.event_id, []).append(action)
    return grouped


def _traces_by_query(run: TaskRunRecord) -> dict[str, RetrievalTrace]:
    return {trace.query_id: trace for trace in run.retrieval_traces}


def _answers_by_query(run: TaskRunRecord) -> dict[str, AnswerPrediction]:
    return {answer.query_id: answer for answer in run.answer_predictions}


def _final_snapshot(run: TaskRunRecord):
    return run.memory_snapshots[-1] if run.memory_snapshots else None


def _select_final_snapshot(task: MemUpdateTask, run: TaskRunRecord):
    snapshot = _final_snapshot(run)
    if snapshot is None:
        return None, "final memory snapshot is absent"
    if (
        run.completion_status is CompletionStatus.COMPLETED
        and snapshot.after_event_id is not None
        and task.events
        and snapshot.after_event_id != task.events[-1].event_id
    ):
        return None, "completed run snapshot does not reach the task's final event"
    return snapshot, None


def _action_pairs(task: MemUpdateTask, run: TaskRunRecord):
    predicted = _actions_by_event(run)
    offsets: dict[str, int] = {}
    action_by_id = {action.action_id: action for action in task.gold.actions}
    pairs = []
    for action_id in task.gold.action_sequence:
        gold = action_by_id[action_id]
        offset = offsets.get(gold.event_id, 0)
        occurrences = predicted.get(gold.event_id, [])
        pairs.append((gold, occurrences[offset] if offset < len(occurrences) else None))
        offsets[gold.event_id] = offset + 1
    return pairs


def _target_matches(gold: GoldAction, predicted) -> bool:
    if gold.operation is Operation.NOOP:
        return predicted.target_object_key is None
    if predicted.target_object_key is None or len(gold.target_object_keys) != 1:
        return False
    return predicted.target_object_key.canonical_id == gold.target_object_keys[0].canonical_id


def _current_entry_match(task: MemUpdateTask, query: MemoryQuery, entry) -> bool:
    if entry.object_key_candidate is None:
        return False
    target_ids = {key.canonical_id for key in query.target_object_keys}
    if entry.object_key_candidate.canonical_id not in target_ids:
        return False
    expected = task.gold.gold_answers[query.query_id]
    return _same_value(entry.value_candidate, expected)


def _stale_values_for_query(task: MemUpdateTask, query: MemoryQuery) -> tuple[Any, ...]:
    stale: list[Any] = []
    for key in query.target_object_keys:
        history = task.gold.version_history.get(key.canonical_id, [])
        current = task.gold.final_state.get(key.canonical_id)
        for value in history:
            if not _same_value(value, current):
                stale.append(value)
    return tuple(stale)


def _trace_has_linkage(trace: RetrievalTrace) -> bool:
    return all(
        entry.object_key_candidate is not None and entry.value_candidate is not None
        for entry in trace.retrieved_entries
    )


def _trace_has_value_linkage(trace: RetrievalTrace) -> bool:
    return all(entry.value_candidate is not None for entry in trace.retrieved_entries)


def _trace_has_current(task: MemUpdateTask, query: MemoryQuery, trace: RetrievalTrace) -> bool | None:
    if _trace_has_linkage(trace):
        return any(_current_entry_match(task, query, entry) for entry in trace.retrieved_entries)
    return trace.gold_in_context


def _trace_stale_count(
    task: MemUpdateTask,
    query: MemoryQuery,
    trace: RetrievalTrace,
    stale_values: tuple[Any, ...] | None = None,
) -> int:
    stale_values = (
        stale_values
        if stale_values is not None
        else _stale_values_for_query(task, query)
    )
    target_ids = {key.canonical_id for key in query.target_object_keys}
    return sum(
        1
        for entry in trace.retrieved_entries
        if entry.object_key_candidate is not None
        and entry.object_key_candidate.canonical_id in target_ids
        and any(_same_value(entry.value_candidate, value) for value in stale_values)
    )


def _trace_has_stale(
    task: MemUpdateTask,
    query: MemoryQuery,
    trace: RetrievalTrace,
    stale_count: int | None = None,
) -> bool | None:
    if _trace_has_linkage(trace):
        return (
            stale_count
            if stale_count is not None
            else _trace_stale_count(task, query, trace)
        ) > 0
    return trace.stale_in_context


def _runtime_failed(run: TaskRunRecord) -> bool:
    return bool(run.exceptions) or run.completion_status in {
        CompletionStatus.FAILED,
        CompletionStatus.PARTIAL,
    }


def _validate_extractor_capability_coherence(capabilities: AdapterCapabilities) -> None:
    if capabilities.requires_evaluation_extractor and (
        not isinstance(capabilities.extractor_version, str)
        or not capabilities.extractor_version.strip()
        or capabilities.extractor_version != capabilities.extractor_version.strip()
    ):
        raise ValueError(
            "requires_evaluation_extractor=True requires a canonical nonblank extractor_version"
        )


def _extractor_provenance_issue(
    path: str,
    run: TaskRunRecord,
    capabilities: AdapterCapabilities,
) -> str | None:
    if (
        path not in EXTRACTOR_LINKAGE_METRIC_PATHS
        or not capabilities.requires_evaluation_extractor
    ):
        return None
    runtime_version = run.parser_extractor_provenance.memory_entry_extractor_version
    if (
        not runtime_version.strip()
        or runtime_version != capabilities.extractor_version
    ):
        return (
            "extractor provenance mismatch: "
            f"adapter={capabilities.extractor_version!r}, runtime={runtime_version!r}"
        )
    return None


def _metric_observable(
    path: str,
    run: TaskRunRecord,
    capabilities: AdapterCapabilities,
) -> bool:
    definition = METRIC_REGISTRY[path]
    return (
        not missing_capabilities(definition, capabilities)
        and _extractor_provenance_issue(path, run, capabilities) is None
    )


def _validate_unique_nonblank(values: list[str], label: str) -> None:
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"{label} must contain nonblank exact strings")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} are not allowed")


def _validate_entry_records(entries, event_ids: set[str], label: str) -> None:
    _validate_unique_nonblank([entry.entry_id for entry in entries], f"{label} entry IDs")
    for entry in entries:
        _validate_unique_nonblank(entry.source_event_ids, f"source_event_ids for {entry.entry_id}")
        unknown = set(entry.source_event_ids) - event_ids
        if unknown:
            raise ValueError(f"unknown source_event_ids for {entry.entry_id}: {sorted(unknown)}")


def _is_finite_runtime_number(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is int:
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False
    return False


def _validate_system_events(run: TaskRunRecord) -> None:
    nonnegative_numeric = {
        "ingest_latency_ms",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "api_cost",
    }
    count_fields = {"token_usage", "input_tokens", "output_tokens"}
    for index, event in enumerate(run.system_events):
        for key, value in event.items():
            if key in nonnegative_numeric:
                if not _is_finite_runtime_number(value) or value < 0:
                    raise ValueError(
                        f"system_events[{index}].{key} must be a finite nonnegative number"
                    )
            elif key == "error_rate":
                if not _is_finite_runtime_number(value) or not 0 <= value <= 1:
                    raise ValueError(
                        f"system_events[{index}].error_rate must be a finite rate"
                    )
            elif key in count_fields:
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"system_events[{index}].{key} must be a nonnegative integer"
                    )


def _validate_runtime_integrity(task: MemUpdateTask, run: TaskRunRecord) -> None:
    event_ids = {event.event_id for event in task.events}
    query_ids = {query.query_id for query in task.queries}
    gold_by_event: dict[str, int] = {}
    for action in task.gold.actions:
        gold_by_event[action.event_id] = gold_by_event.get(action.event_id, 0) + 1
    predicted_by_event: dict[str, int] = {}
    for action in run.parsed_actions:
        if action.event_id not in event_ids:
            raise ValueError(f"parsed action references unknown event_id {action.event_id!r}")
        predicted_by_event[action.event_id] = predicted_by_event.get(action.event_id, 0) + 1
    surplus = {
        event_id: count - gold_by_event.get(event_id, 0)
        for event_id, count in predicted_by_event.items()
        if count > gold_by_event.get(event_id, 0)
    }
    if surplus:
        raise ValueError(f"surplus parsed action occurrences: {surplus}")

    trace_ids = [trace.query_id for trace in run.retrieval_traces]
    answer_ids = [answer.query_id for answer in run.answer_predictions]
    _validate_unique_nonblank(trace_ids, "retrieval query IDs")
    _validate_unique_nonblank(answer_ids, "answer query IDs")
    unknown_traces = set(trace_ids) - query_ids
    unknown_answers = set(answer_ids) - query_ids
    if unknown_traces:
        raise ValueError(f"retrieval traces reference unknown query IDs: {sorted(unknown_traces)}")
    if unknown_answers:
        raise ValueError(f"answer predictions reference unknown query IDs: {sorted(unknown_answers)}")

    event_order = {event.event_id: event.sequence_index for event in task.events}
    linked_snapshot_indices: list[int] = []
    if len(run.memory_snapshots) > 1 and any(
        snapshot.after_event_id is None for snapshot in run.memory_snapshots
    ):
        raise ValueError("multiple snapshots cannot contain ambiguous after_event_id=None")
    for index, snapshot in enumerate(run.memory_snapshots):
        if snapshot.after_event_id is not None:
            if snapshot.after_event_id not in event_order:
                raise ValueError(
                    f"snapshot references unknown after_event_id {snapshot.after_event_id!r}"
                )
            linked_snapshot_indices.append(event_order[snapshot.after_event_id])
        _validate_entry_records(snapshot.entries, event_ids, f"snapshot[{index}]")
    if linked_snapshot_indices and (
        len(linked_snapshot_indices) != len(set(linked_snapshot_indices))
        or linked_snapshot_indices != sorted(linked_snapshot_indices)
    ):
        raise ValueError("snapshot after_event_id values must be unique and monotonic")

    trace_entries: dict[str, set[str]] = {}
    for index, trace in enumerate(run.retrieval_traces):
        _validate_entry_records(trace.retrieved_entries, event_ids, f"retrieval_trace[{index}]")
        trace_entries[trace.query_id] = {entry.entry_id for entry in trace.retrieved_entries}
    for answer in run.answer_predictions:
        _validate_unique_nonblank(
            answer.cited_event_ids, f"cited_event_ids for {answer.query_id}"
        )
        _validate_unique_nonblank(
            answer.cited_entry_ids, f"cited_entry_ids for {answer.query_id}"
        )
        unknown_events = set(answer.cited_event_ids) - event_ids
        if unknown_events:
            raise ValueError(
                f"answer {answer.query_id} cites unknown event IDs: {sorted(unknown_events)}"
            )
        unknown_entries = set(answer.cited_entry_ids) - trace_entries.get(answer.query_id, set())
        if unknown_entries:
            raise ValueError(
                f"answer {answer.query_id} cites unknown or cross-query entry IDs: "
                f"{sorted(unknown_entries)}"
            )
    _validate_system_events(run)


@dataclass
class _EvidenceContext:
    task: MemUpdateTask
    run: TaskRunRecord
    capabilities: AdapterCapabilities
    config: ScorerConfig
    _accepted: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    _stale: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    _answer_matches_cache: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    _current_cache: dict[str, bool | None] = field(default_factory=dict)
    _stale_exposure_cache: dict[str, bool | None] = field(default_factory=dict)
    _stale_count_cache: dict[str, int] = field(default_factory=dict)
    _distractor_cache: dict[str, bool] = field(default_factory=dict)

    @cached_property
    def current_queries(self):
        return _current_queries(self.task)

    @cached_property
    def reference_queries(self):
        return _reference_queries(self.task)

    @cached_property
    def action_pairs(self):
        return _action_pairs(self.task, self.run)

    @cached_property
    def traces_by_query(self):
        return _traces_by_query(self.run)

    @cached_property
    def answers_by_query(self):
        return _answers_by_query(self.run)

    @cached_property
    def final_snapshot_result(self):
        return _select_final_snapshot(self.task, self.run)

    @cached_property
    def final_snapshot(self):
        return self.final_snapshot_result[0]

    @property
    def final_snapshot_detail(self):
        return self.final_snapshot_result[1]

    @cached_property
    def state_summary(self):
        if self.final_snapshot is None:
            return None
        return _state_summary(self.task, self.run, self.final_snapshot)

    @cached_property
    def store_analysis(self):
        if self.final_snapshot is None:
            return None, self.final_snapshot_detail
        try:
            return _store_counts(
                self.task,
                self.run,
                self.capabilities,
                self.final_snapshot,
            ), None
        except _StoreAnalysisUnavailable as exc:
            return None, str(exc)

    @cached_property
    def query_evidence(self):
        return tuple(
            (
                query,
                self.traces_by_query.get(query.query_id),
                self.answers_by_query.get(query.query_id),
            )
            for query in self.current_queries
        )

    def accepted_values(self, query: MemoryQuery) -> tuple[Any, ...]:
        if query.query_id not in self._accepted:
            self._accepted[query.query_id] = _accepted_values(self.task, query.query_id)
        return self._accepted[query.query_id]

    def stale_values(self, query: MemoryQuery) -> tuple[Any, ...]:
        if query.query_id not in self._stale:
            self._stale[query.query_id] = _stale_values_for_query(self.task, query)
        return self._stale[query.query_id]

    def answer_matches(self, query: MemoryQuery, prediction: AnswerPrediction):
        if query.query_id not in self._answer_matches_cache:
            self._answer_matches_cache[query.query_id] = _answer_matches(
                self.task,
                query,
                prediction,
                self.config.answer_normalization_profile,
                self.accepted_values(query),
            )
        return self._answer_matches_cache[query.query_id]

    def current_present(self, query: MemoryQuery, trace: RetrievalTrace):
        if query.query_id not in self._current_cache:
            self._current_cache[query.query_id] = _trace_has_current(
                self.task, query, trace
            )
        return self._current_cache[query.query_id]

    def stale_count(self, query: MemoryQuery, trace: RetrievalTrace) -> int:
        if query.query_id not in self._stale_count_cache:
            self._stale_count_cache[query.query_id] = _trace_stale_count(
                self.task, query, trace, self.stale_values(query)
            )
        return self._stale_count_cache[query.query_id]

    def stale_present(self, query: MemoryQuery, trace: RetrievalTrace):
        if query.query_id not in self._stale_exposure_cache:
            self._stale_exposure_cache[query.query_id] = _trace_has_stale(
                self.task,
                query,
                trace,
                self.stale_count(query, trace),
            )
        return self._stale_exposure_cache[query.query_id]

    def distractor_copied(
        self,
        query: MemoryQuery,
        trace: RetrievalTrace | None,
        prediction: AnswerPrediction,
    ) -> bool:
        if query.query_id not in self._distractor_cache:
            self._distractor_cache[query.query_id] = _is_distractor_copy(
                self.task,
                query,
                trace,
                prediction.parsed_answer,
                self.accepted_values(query),
                self.stale_values(query),
            )
        return self._distractor_cache[query.query_id]


def _query_evidence(task: MemUpdateTask, run: TaskRunRecord):
    queries = _current_queries(task)
    traces = _traces_by_query(run)
    answers = _answers_by_query(run)
    return [
        (query, traces.get(query.query_id), answers.get(query.query_id))
        for query in queries
    ]


def _compute_protocol(path: str, task: MemUpdateTask, run: TaskRunRecord):
    leaf = path.split(".", 1)[1]
    if leaf == "answer_parse_valid":
        if task.queries and not run.answer_predictions:
            return None, "answer prediction artifact is absent"
        if not task.queries:
            return None, "task has no answer queries"
        return (
            len(run.answer_predictions) == len(task.queries)
            and all(answer.format_valid for answer in run.answer_predictions)
        ), None
    if task.events and not run.parsed_actions:
        return None, "parsed action artifact is absent"
    if not task.events:
        return None, "task has no events"
    expected_count = len(task.gold.actions)
    if leaf == "action_parse_valid":
        return (
            len(run.parsed_actions) == expected_count
            and all(action.format_valid for action in run.parsed_actions)
        ), None
    if leaf == "execution_success_rate":
        successes = sum(
            action.execution_status.casefold() in {"success", "succeeded", "completed"}
            for action in run.parsed_actions
        )
        return float(successes / expected_count), None
    if leaf == "unsupported_operation_rate":
        unsupported = sum(
            "unsupported" in action.execution_status.casefold()
            or "unsupported_action" in action.error_flags
            for action in run.parsed_actions
        )
        return float(unsupported / expected_count), None
    if leaf == "fallback_rate":
        return float(sum(action.fallback_used for action in run.parsed_actions) / expected_count), None
    raise KeyError(path)


def _compute_action(
    path: str,
    task: MemUpdateTask,
    run: TaskRunRecord,
    context: _EvidenceContext | None = None,
):
    if task.gold.actions and not run.parsed_actions:
        return None, "parsed action trace is absent"
    pairs = context.action_pairs if context is not None else _action_pairs(task, run)
    if not pairs:
        return None, "task has no gold actions"
    leaf = path.split(".", 1)[1]
    target_dependent = {
        "full_action_exact_match",
        "object_key_accuracy",
        "entity_accuracy",
        "attribute_accuracy",
        "wrong_object_write_rate",
    }
    if leaf in target_dependent and any(
        len(gold.target_object_keys) > 1 for gold, _ in pairs
    ):
        return None, "single-target runtime action cannot represent multi-target gold action"
    values: list[bool] = []
    for gold, predicted in pairs:
        operation_match = predicted is not None and predicted.operation is gold.operation
        target_match = predicted is not None and _target_matches(gold, predicted)
        value_match = predicted is not None and _same_value(predicted.value, gold.value)
        if leaf == "operation_accuracy":
            values.append(operation_match)
        elif leaf == "full_action_exact_match":
            values.append(operation_match and target_match and value_match)
        elif leaf == "object_key_accuracy":
            values.append(target_match)
        elif leaf == "entity_accuracy":
            values.append(
                gold.operation is Operation.NOOP
                or (
                    predicted is not None
                    and predicted.target_object_key is not None
                    and len(gold.target_object_keys) == 1
                    and predicted.target_object_key.entity == gold.target_object_keys[0].entity
                )
            )
        elif leaf == "attribute_accuracy":
            values.append(
                gold.operation is Operation.NOOP
                or (
                    predicted is not None
                    and predicted.target_object_key is not None
                    and len(gold.target_object_keys) == 1
                    and predicted.target_object_key.attribute
                    == gold.target_object_keys[0].attribute
                )
            )
        elif leaf == "value_accuracy":
            values.append(value_match)
        elif leaf == "false_write_rate":
            values.append(
                gold.operation is Operation.NOOP
                and predicted is not None
                and predicted.operation in {Operation.ADD, Operation.UPDATE, Operation.DELETE}
            )
        elif leaf == "missed_write_rate":
            values.append(
                gold.operation in {Operation.ADD, Operation.UPDATE, Operation.DELETE}
                and (predicted is None or predicted.operation is Operation.NOOP)
            )
        elif leaf == "wrong_object_write_rate":
            values.append(
                gold.operation is not Operation.NOOP
                and predicted is not None
                and predicted.operation is not Operation.NOOP
                and not target_match
            )
        else:
            raise KeyError(path)
    return _mean([float(value) for value in values]), None


def _typed_mapping_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        _same_value(left[key], right[key]) for key in left
    )


def _state_summary(task: MemUpdateTask, run: TaskRunRecord, snapshot=None):
    snapshot = snapshot if snapshot is not None else _final_snapshot(run)
    if snapshot is None:
        return None
    predicted = dict(snapshot.state_by_object)
    gold = dict(task.gold.final_state)
    correct_keys = {
        key for key in predicted.keys() & gold.keys() if _same_value(predicted[key], gold[key])
    }
    precision = len(correct_keys) / len(predicted) if predicted else float(not gold)
    recall = len(correct_keys) / len(gold) if gold else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return predicted, gold, correct_keys, precision, recall, f1


def _compute_state(
    path: str,
    task: MemUpdateTask,
    run: TaskRunRecord,
    context: _EvidenceContext | None = None,
):
    summary = context.state_summary if context is not None else _state_summary(task, run)
    if summary is None:
        detail = context.final_snapshot_detail if context is not None else None
        return None, detail or "final memory snapshot is absent"
    predicted, gold, correct_keys, precision, recall, f1 = summary
    leaf = path.split(".", 1)[1]
    if leaf == "final_state_accuracy":
        return float(_typed_mapping_equal(predicted, gold)), None
    if leaf == "state_precision":
        return float(precision), None
    if leaf == "state_recall":
        return float(recall), None
    if leaf == "state_f1":
        return float(f1), None
    if leaf == "state_resolve_rate":
        expected_ids = {key.canonical_id for key in task.gold.expected_present_objects}
        if not expected_ids:
            return None, "task has no expected-present objects"
        return float(len(correct_keys & expected_ids) / len(expected_ids)), None
    if leaf == "collateral_corruption_rate":
        collateral = set(predicted) - set(gold)
        return float(len(collateral) / max(len(predicted), 1)), None
    if leaf == "expected_absence_accuracy":
        expected_ids = {key.canonical_id for key in task.gold.expected_absent_objects}
        if not expected_ids:
            return 1.0, None
        absent = sum(key not in predicted for key in expected_ids)
        return float(absent / len(expected_ids)), None
    raise KeyError(path)


def _parse_entry_timestamp(value: Any) -> datetime:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise _StoreAnalysisUnavailable(
            "declared entry timestamp artifact is not canonical"
        )
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None:
            raise ValueError("timestamp is not timezone-aware")
        return parsed.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _StoreAnalysisUnavailable(
            "declared entry timestamp artifact is not a timezone-aware instant"
        ) from exc


def _declared_entry_orders(
    entries: list[Any],
    event_order: Mapping[str, int],
    capabilities: AdapterCapabilities,
) -> list[int | datetime]:
    if capabilities.exports_timestamps_or_order:
        if all(entry.version_index is not None for entry in entries):
            return [entry.version_index for entry in entries]
        timestamps = [entry.updated_at or entry.created_at for entry in entries]
        if all(timestamp is not None for timestamp in timestamps):
            try:
                return [_parse_entry_timestamp(timestamp) for timestamp in timestamps]
            except _StoreAnalysisUnavailable:
                pass
    if capabilities.exports_source_event_ids:
        source_orders = [
            max(
                (
                    event_order[event_id]
                    for event_id in entry.source_event_ids
                    if event_id in event_order
                ),
                default=None,
            )
            for entry in entries
        ]
        if all(order is not None for order in source_orders):
            return source_orders
    raise _StoreAnalysisUnavailable(
        "declared entry order/linkage artifact is absent"
    )


def _store_counts(
    task: MemUpdateTask,
    run: TaskRunRecord,
    capabilities: AdapterCapabilities,
    snapshot=None,
):
    snapshot = snapshot if snapshot is not None else _final_snapshot(run)
    if snapshot is None:
        return None
    if any(entry.object_key_candidate is None for entry in snapshot.entries):
        raise _StoreAnalysisUnavailable(
            "per-entry object-key linkage artifact is absent"
        )
    event_order = {event.event_id: event.sequence_index for event in task.events}
    entries_by_key: dict[str, list[Any]] = {}
    for entry in snapshot.entries:
        key = entry.object_key_candidate
        assert key is not None
        if key.canonical_id in task.gold.final_state:
            entries_by_key.setdefault(key.canonical_id, []).append(entry)

    target_entries = tuple(
        entry for entries in entries_by_key.values() for entry in entries
    )
    values_available = all(
        entry.value_candidate is not None for entry in target_entries
    )
    value_detail = (
        None
        if values_available
        else "per-entry target value linkage artifact is absent"
    )
    obsolete = 0
    stale_conflicting = 0 if values_available else None
    duplicates = 0 if values_available else None
    for key_id, entries in entries_by_key.items():
        declared_orders = _declared_entry_orders(
            entries,
            event_order,
            capabilities,
        )
        latest_order = max(declared_orders)
        latest_entries = [
            entry
            for entry, order in zip(entries, declared_orders)
            if order == latest_order
        ]

        current = task.gold.final_state[key_id]
        if values_available and latest_entries and any(
            not _same_value(entry.value_candidate, latest_entries[0].value_candidate)
            for entry in latest_entries[1:]
        ):
            raise _StoreAnalysisUnavailable(
                "conflicting values share the maximal version/order"
            )
        latest_ids = {id(entry) for entry in latest_entries}
        for entry in entries:
            if id(entry) in latest_ids:
                continue
            obsolete += 1
            if values_available and not _same_value(entry.value_candidate, current):
                assert stale_conflicting is not None
                stale_conflicting += 1
        if values_available:
            current_latest = sum(
                _same_value(entry.value_candidate, current) for entry in latest_entries
            )
            assert duplicates is not None
            duplicates += max(current_latest - 1, 0)
    return _StoreAnalysis(
        snapshot=snapshot,
        obsolete=obsolete,
        stale_conflicting=stale_conflicting,
        duplicate_current=duplicates,
        value_detail=value_detail,
    )


def _compute_store(
    path: str,
    task: MemUpdateTask,
    run: TaskRunRecord,
    context: _EvidenceContext | None = None,
):
    leaf = path.split(".", 1)[1]
    if leaf == "write_amplification":
        if not run.parsed_actions:
            return None, "parsed action trace is absent"
        predicted_writes = sum(
            action.operation in {Operation.ADD, Operation.UPDATE, Operation.DELETE}
            for action in run.parsed_actions
        )
        gold_writes = sum(action.operation is not Operation.NOOP for action in task.gold.actions)
        return float(predicted_writes / max(gold_writes, 1)), None
    snapshot = context.final_snapshot if context is not None else _final_snapshot(run)
    if snapshot is None:
        detail = context.final_snapshot_detail if context is not None else None
        return None, detail or "final memory snapshot is absent"
    if (
        leaf in {"final_memory_size", "compaction_ratio"}
        and snapshot.store_size != len(snapshot.entries)
    ):
        return None, "final snapshot store_size does not match exported entry count"
    if leaf == "final_memory_size":
        return snapshot.store_size, None
    if leaf == "compaction_ratio":
        gold_writes = sum(action.operation is not Operation.NOOP for action in task.gold.actions)
        return float(snapshot.store_size / max(gold_writes, 1)), None
    if context is not None:
        counts, unavailable_detail = context.store_analysis
        if unavailable_detail is not None:
            return None, unavailable_detail
    else:
        return None, "adapter capability context is absent for store version analysis"
    if counts is None:
        return None, "final memory snapshot is absent"
    if leaf == "obsolete_version_count":
        return counts.obsolete, None
    if leaf == "stale_conflicting_value_count":
        if counts.stale_conflicting is None:
            return None, counts.value_detail
        return counts.stale_conflicting, None
    if leaf == "duplicate_current_count":
        if counts.duplicate_current is None:
            return None, counts.value_detail
        return counts.duplicate_current, None
    raise KeyError(path)


def _compute_retrieval(
    path: str,
    task: MemUpdateTask,
    run: TaskRunRecord,
    context: _EvidenceContext | None = None,
):
    evidence = context.query_evidence if context is not None else _query_evidence(task, run)
    if not evidence:
        return None, "task has no current-state query"
    if any(trace is None for _, trace, _ in evidence):
        return None, "retrieval trace is absent for a current-state query"
    leaf = path.split(".", 1)[1]
    values: list[float] = []
    for query, trace, _ in evidence:
        assert trace is not None
        if leaf == "current_recall_at_k":
            current_present = (
                context.current_present(query, trace)
                if context is not None
                else _trace_has_current(task, query, trace)
            )
            if current_present is None:
                return None, "retrieval object/value linkage artifact is absent"
            values.append(float(current_present))
        elif leaf == "current_mrr":
            if trace.retrieved_entries and not _trace_has_linkage(trace):
                return None, "retrieval object/value linkage artifact is absent"
            if trace.retrieved_entries and not trace.ranks:
                return None, "retrieval rank artifact is absent"
            matching_ranks = [
                rank
                for entry, rank in zip(trace.retrieved_entries, trace.ranks)
                if _current_entry_match(task, query, entry)
            ]
            values.append(0.0 if not matching_ranks else 1.0 / min(matching_ranks))
        elif leaf == "stale_exposure_rate":
            stale_present = (
                context.stale_present(query, trace)
                if context is not None
                else _trace_has_stale(task, query, trace)
            )
            if stale_present is None:
                return None, "retrieval object/value linkage artifact is absent"
            values.append(float(stale_present))
        elif leaf == "stale_count_in_context":
            if trace.retrieved_entries and not _trace_has_linkage(trace):
                return None, "retrieval object/value linkage artifact is absent"
            stale_count = (
                context.stale_count(query, trace)
                if context is not None
                else _trace_stale_count(task, query, trace)
            )
            values.append(float(stale_count))
        elif leaf == "distractor_exposure_rate":
            if trace.distractor_in_context is None:
                return None, "distractor annotation is absent from retrieval trace"
            values.append(float(trace.distractor_in_context))
        else:
            raise KeyError(path)
    if leaf == "stale_count_in_context":
        return int(sum(values)), None
    return _mean(values), None


def _answer_matches(
    task: MemUpdateTask,
    query: MemoryQuery,
    prediction: AnswerPrediction,
    profile: str,
    accepted_values: tuple[Any, ...] | None = None,
):
    parsed = prediction.parsed_answer
    exact = _same_value(parsed, task.gold.gold_answers[query.query_id])
    normalized = _normalize_answer(parsed, profile)
    accepted_values = (
        accepted_values
        if accepted_values is not None
        else _accepted_values(task, query.query_id)
    )
    normalized_match = any(
        _same_value(normalized, _normalize_answer(value, profile))
        for value in accepted_values
    )
    return exact, normalized_match


def _token_f1(predicted: Any, gold: Any, profile: str) -> float:
    if not isinstance(predicted, str) or not isinstance(gold, str):
        return float(_same_value(predicted, gold))
    predicted_tokens = str(_normalize_answer(predicted, profile)).split()
    gold_tokens = str(_normalize_answer(gold, profile)).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _structured_field_accuracy(predicted: Any, gold: Any) -> float:
    if isinstance(predicted, Mapping) and isinstance(gold, Mapping):
        if not gold:
            return float(not predicted)
        correct = sum(
            key in predicted and _same_value(predicted[key], value)
            for key, value in gold.items()
        )
        return float(correct / len(gold))
    if isinstance(predicted, (list, tuple)) and isinstance(gold, (list, tuple)):
        denominator = max(len(predicted), len(gold), 1)
        correct = sum(
            _same_value(predicted[index], gold[index])
            for index in range(min(len(predicted), len(gold)))
        )
        return float(correct / denominator)
    return float(_same_value(predicted, gold))


def _is_distractor_copy(
    task: MemUpdateTask,
    query: MemoryQuery,
    trace,
    parsed: Any,
    accepted: tuple[Any, ...] | None = None,
    stale: tuple[Any, ...] | None = None,
) -> bool:
    if trace is None or not trace.distractor_in_context or not _trace_has_value_linkage(trace):
        return False
    accepted = accepted if accepted is not None else _accepted_values(task, query.query_id)
    stale = stale if stale is not None else _stale_values_for_query(task, query)
    if any(_same_value(parsed, value) for value in accepted):
        return False
    if any(_same_value(parsed, value) for value in stale):
        return False
    return any(
        _same_value(parsed, entry.value_candidate)
        for entry in trace.retrieved_entries
    )


def _compute_reference_resolution(
    task: MemUpdateTask,
    run: TaskRunRecord,
    context: _EvidenceContext | None,
):
    queries = context.reference_queries if context is not None else _reference_queries(task)
    answers = context.answers_by_query if context is not None else _answers_by_query(run)
    evidence = tuple((query, answers.get(query.query_id)) for query in queries)
    if not evidence:
        return None, "task has no unresolved-reference query"
    if any(prediction is None for _, prediction in evidence):
        return None, "answer prediction is absent for an unresolved-reference query"
    if any(
        prediction.disposition is AnswerDisposition.UNAVAILABLE
        for _, prediction in evidence
        if prediction is not None
    ):
        return None, "reference-resolution outcome is explicitly unavailable"

    values: list[float] = []
    for query, prediction in evidence:
        assert prediction is not None
        canonical = task.gold.canonical_answers[query.query_id]
        if canonical.resolution_status in {
            ReferenceResolutionStatus.AMBIGUOUS,
            ReferenceResolutionStatus.NO_MATCH,
        }:
            values.append(
                float(
                    canonical.disposition is AnswerDisposition.ABSTAINED
                    and prediction.disposition is AnswerDisposition.ABSTAINED
                )
            )
            continue
        if prediction.disposition is AnswerDisposition.ANSWERED:
            if prediction.parsed_answer is None:
                return None, "parsed unique-reference answer artifact is absent"
            values.append(
                float(
                    canonical.disposition is AnswerDisposition.ANSWERED
                    and _same_value(prediction.parsed_answer, canonical.value)
                )
            )
        else:
            values.append(0.0)
    return _mean(values), None


def _compute_answer(
    path: str,
    task: MemUpdateTask,
    run: TaskRunRecord,
    config: ScorerConfig,
    context: _EvidenceContext | None = None,
):
    leaf = path.split(".", 1)[1]
    if leaf == "reference_resolution_accuracy":
        return _compute_reference_resolution(task, run, context)
    evidence = context.query_evidence if context is not None else _query_evidence(task, run)
    if not evidence:
        return None, "task has no current-state query"
    if any(answer is None for _, _, answer in evidence):
        return None, "answer prediction is absent for a current-state query"
    if any(answer.parsed_answer is None for _, _, answer in evidence if answer is not None):
        return None, "parsed answer artifact is absent for a current-state query"
    values: list[float] = []
    snapshot = context.final_snapshot if context is not None else _final_snapshot(run)
    for query, trace, prediction in evidence:
        assert prediction is not None
        if leaf in {"exact_match", "normalized_match", "gold_retrieved_wrong_answer"}:
            exact, normalized_match = (
                context.answer_matches(query, prediction)
                if context is not None
                else _answer_matches(
                    task, query, prediction, config.answer_normalization_profile
                )
            )
        if leaf == "exact_match":
            values.append(float(exact))
        elif leaf == "normalized_match":
            values.append(float(normalized_match))
        elif leaf == "token_f1":
            values.append(
                _token_f1(
                    prediction.parsed_answer,
                    task.gold.gold_answers[query.query_id],
                    config.answer_normalization_profile,
                )
            )
        elif leaf == "structured_field_accuracy":
            values.append(
                _structured_field_accuracy(
                    prediction.parsed_answer,
                    task.gold.gold_answers[query.query_id],
                )
            )
        elif leaf == "stale_copied":
            stale_values = (
                context.stale_values(query)
                if context is not None
                else _stale_values_for_query(task, query)
            )
            values.append(
                float(
                    any(
                        _same_value(prediction.parsed_answer, value)
                        for value in stale_values
                    )
                )
            )
        elif leaf == "distractor_copied":
            if trace is None or trace.distractor_in_context is None:
                return None, "retrieval distractor annotation artifact is absent"
            if not trace.distractor_in_context:
                values.append(0.0)
                continue
            if not trace.retrieved_entries:
                return None, "retrieved distractor context artifact is absent"
            if not _trace_has_value_linkage(trace):
                return None, "retrieved value linkage artifact is absent"
            distractor_copied = (
                context.distractor_copied(query, trace, prediction)
                if context is not None
                else _is_distractor_copy(
                    task, query, trace, prediction.parsed_answer
                )
            )
            values.append(float(distractor_copied))
        elif leaf == "gold_retrieved_wrong_answer":
            if trace is None:
                return None, "retrieval trace is absent for answer diagnostic"
            gold_retrieved = (
                context.current_present(query, trace)
                if context is not None
                else _trace_has_current(task, query, trace)
            )
            if gold_retrieved is None:
                return None, "retrieval object/value linkage artifact is absent"
            values.append(float(gold_retrieved and not normalized_match))
        elif leaf == "answer_state_consistency":
            if snapshot is None:
                detail = context.final_snapshot_detail if context is not None else None
                return None, detail or "final state artifact is absent for answer consistency"
            if not query.target_object_keys:
                return None, "query target artifact is absent for answer consistency"
            state_values = [
                snapshot.state_by_object.get(key.canonical_id)
                for key in query.target_object_keys
                if key.canonical_id in snapshot.state_by_object
            ]
            if not state_values:
                values.append(0.0)
            else:
                values.append(
                    float(
                        any(
                            _same_value(prediction.parsed_answer, state_value)
                            for state_value in state_values
                        )
                    )
                )
        else:
            raise KeyError(path)
    return _mean(values), None


def _numeric_system_event_values(run: TaskRunRecord, key: str) -> list[float]:
    values: list[float] = []
    for event in run.system_events:
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def _compute_system(path: str, run: TaskRunRecord):
    leaf = path.split(".", 1)[1]
    if leaf == "ingest_latency_ms":
        values = [action.latency_ms for action in run.parsed_actions if action.latency_ms is not None]
        return (_mean(values), None) if values else (None, "ingest latency artifact is absent")
    if leaf == "retrieval_latency_ms":
        values = _numeric_system_event_values(run, "retrieval_latency_ms")
        return (_mean(values), None) if values else (None, "retrieval latency artifact is absent")
    if leaf == "answer_latency_ms":
        values = [answer.latency_ms for answer in run.answer_predictions if answer.latency_ms is not None]
        return (_mean(values), None) if values else (None, "answer latency artifact is absent")
    if leaf == "token_usage":
        if not run.answer_predictions or not any(answer.usage for answer in run.answer_predictions):
            return None, "token usage artifact is absent"
        return sum(sum(answer.usage.values()) for answer in run.answer_predictions), None
    if leaf == "api_cost":
        values = _numeric_system_event_values(run, "api_cost")
        return (float(sum(values)), None) if values else (None, "API cost artifact is absent")
    if leaf == "error_rate":
        return float(_runtime_failed(run)), None
    raise KeyError(path)


def _compute_audit(
    path: str,
    run: TaskRunRecord,
    capabilities: AdapterCapabilities,
    context: _EvidenceContext | None = None,
):
    leaf = path.split(".", 1)[1]
    if leaf == "action_trace_available":
        return bool(capabilities.exports_action_trace and run.parsed_actions), None
    if leaf == "state_export_available":
        snapshot_available = (
            context.final_snapshot is not None
            if context is not None
            else bool(run.memory_snapshots)
        )
        return bool(
            (capabilities.exports_entries or capabilities.exports_raw_state)
            and snapshot_available
        ), None
    if leaf == "retrieval_trace_available":
        return bool(capabilities.exports_retrieval_ids and run.retrieval_traces), None
    if leaf == "source_provenance_coverage":
        snapshot = context.final_snapshot if context is not None else _final_snapshot(run)
        if snapshot is None:
            detail = context.final_snapshot_detail if context is not None else None
            return None, detail or "memory-entry provenance artifact is absent"
        if not snapshot.entries:
            return None, "memory-entry provenance artifact is absent"
        covered = sum(bool(entry.source_event_ids) for entry in snapshot.entries)
        return float(covered / len(snapshot.entries)), None
    if leaf == "manifest_completeness":
        provenance = run.parser_extractor_provenance
        fields = provenance.model_dump(mode="python")
        required = [
            "action_parser_version",
            "answer_parser_version",
            "memory_entry_extractor_version",
            "redaction_policy_version",
        ]
        complete = sum(bool(fields[name]) for name in required)
        return float(complete / len(required)), None
    raise KeyError(path)


def _compute_metric(path: str, context: _EvidenceContext):
    task = context.task
    run = context.run
    capabilities = context.capabilities
    config = context.config
    layer = path.split(".", 1)[0]
    if layer == "protocol_scores":
        return _compute_protocol(path, task, run)
    if layer == "action_scores":
        return _compute_action(path, task, run, context)
    if layer == "state_scores":
        return _compute_state(path, task, run, context)
    if layer == "store_scores":
        return _compute_store(path, task, run, context)
    if layer == "retrieval_scores":
        return _compute_retrieval(path, task, run, context)
    if layer == "answer_scores":
        return _compute_answer(path, task, run, config, context)
    if layer == "system_scores":
        return _compute_system(path, run)
    if layer == "audit_scores":
        return _compute_audit(path, run, capabilities, context)
    raise KeyError(path)


def _derive_failure_flags(context: _EvidenceContext):
    task = context.task
    run = context.run
    capabilities = context.capabilities
    flags: set[str] = set()
    if _runtime_failed(run):
        flags.add("system_exception")
    pairs = context.action_pairs
    for gold, predicted in pairs:
        if predicted is None:
            if gold.operation is not Operation.NOOP:
                flags.add("missed_update")
            continue
        if not predicted.format_valid:
            flags.add("invalid_action_format")
        if "unsupported" in predicted.execution_status.casefold() or "unsupported_action" in predicted.error_flags:
            flags.add("unsupported_action")
        if predicted.operation is not gold.operation:
            flags.add("wrong_operation")
        if gold.operation is Operation.NOOP:
            if predicted.operation in {Operation.ADD, Operation.UPDATE, Operation.DELETE}:
                flags.add("false_write")
            continue
        if predicted.operation is Operation.NOOP:
            flags.add("missed_update")
        if predicted.target_object_key is not None and len(gold.target_object_keys) == 1:
            target = gold.target_object_keys[0]
            if predicted.target_object_key.entity != target.entity:
                flags.add("wrong_entity")
            if predicted.target_object_key.attribute != target.attribute:
                flags.add("wrong_attribute")
        elif predicted.target_object_key is None:
            flags.add("missed_update")
        if not _same_value(predicted.value, gold.value):
            flags.add("wrong_value")
    if (
        _metric_observable("state_scores.final_state_accuracy", run, capabilities)
        and context.final_snapshot is not None
    ):
        summary = context.state_summary
        if summary is not None:
            predicted_state, gold_state, _, _, _, _ = summary
            if any(
                key not in predicted_state
                or not _same_value(predicted_state[key], expected)
                for key, expected in gold_state.items()
            ):
                flags.add("current_state_missing")
            if set(predicted_state) - set(gold_state):
                flags.add("collateral_corruption")
            absent_ids = {key.canonical_id for key in task.gold.expected_absent_objects}
            if any(key in predicted_state for key in absent_ids):
                flags.add("deletion_failure")
    if (
        _metric_observable("store_scores.obsolete_version_count", run, capabilities)
        and context.final_snapshot is not None
    ):
        counts, unavailable_detail = context.store_analysis
        if unavailable_detail is None and counts is not None and counts.obsolete > 0:
            flags.add("stale_retained")
    traces = context.traces_by_query
    answers = context.answers_by_query
    for query in context.current_queries:
        trace = traces.get(query.query_id)
        prediction = answers.get(query.query_id)
        gold_retrieved = None
        if trace is not None:
            if _metric_observable(
                "retrieval_scores.current_recall_at_k", run, capabilities
            ):
                gold_retrieved = context.current_present(query, trace)
                if gold_retrieved is False:
                    flags.add("current_not_retrieved")
            if _metric_observable(
                "retrieval_scores.stale_exposure_rate", run, capabilities
            ) and context.stale_present(query, trace) is True:
                flags.add("stale_retrieved")
            if capabilities.exports_retrieval_ids and trace.distractor_in_context:
                flags.add("distractor_retrieved")
        if prediction is not None:
            _, normalized_match = context.answer_matches(query, prediction)
            if any(
                _same_value(prediction.parsed_answer, value)
                for value in context.stale_values(query)
            ):
                flags.add("stale_copied")
            if _metric_observable(
                "answer_scores.distractor_copied", run, capabilities
            ) and context.distractor_copied(query, trace, prediction):
                flags.add("distractor_copied")
            if gold_retrieved and not normalized_match:
                flags.add("gold_retrieved_wrong_answer")
            if not prediction.format_valid and normalized_match:
                flags.add("answer_format_only")
    for query in context.reference_queries:
        prediction = answers.get(query.query_id)
        if prediction is None or prediction.disposition is AnswerDisposition.UNAVAILABLE:
            continue
        canonical = task.gold.canonical_answers[query.query_id]
        if (
            canonical.disposition is AnswerDisposition.ABSTAINED
            and prediction.disposition is AnswerDisposition.ANSWERED
        ):
            flags.add("wrong_reference_guess")
        elif (
            canonical.disposition is AnswerDisposition.ANSWERED
            and prediction.disposition is AnswerDisposition.ABSTAINED
        ):
            flags.add("unjustified_abstention")
    return canonicalize_failure_flags(flags)


def _metric_applies_to_task(
    path: str,
    task: MemUpdateTask,
    context: _EvidenceContext | None = None,
) -> bool:
    definition = METRIC_REGISTRY[path]
    if not metric_applies_to_family(definition, task.task_family):
        return False
    layer = path.split(".", 1)[0]
    if layer == "action_scores" and not task.gold.actions:
        return False
    if (
        layer == "protocol_scores"
        and path != "protocol_scores.answer_parse_valid"
        and not task.gold.actions
    ):
        return False
    if path in _TARGET_DEPENDENT_ACTION_PATHS and any(
        len(action.target_object_keys) > 1 for action in task.gold.actions
    ):
        return False
    if path == "answer_scores.reference_resolution_accuracy":
        queries = context.reference_queries if context is not None else _reference_queries(task)
        return bool(queries)
    if layer in {"retrieval_scores", "answer_scores"}:
        queries = context.current_queries if context is not None else _current_queries(task)
        return bool(queries)
    return True


def _validate_legacy_json_value(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain nonfinite legacy values")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_legacy_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} legacy object keys must be exact strings")
            _validate_legacy_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported legacy value type {type(value).__name__}")


def _legacy_metrics(run: TaskRunRecord, namespace: str | None) -> dict[str, Any]:
    if namespace is None:
        return {}
    prefix = f"{namespace}."
    preserved: dict[str, Any] = {}
    for event in run.system_events:
        payload = event.get("legacy_metrics")
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if isinstance(key, str) and key.startswith(prefix):
                _validate_legacy_json_value(value, f"legacy_metrics.{key}")
                preserved[key] = value
    return dict(sorted(preserved.items()))


def _derive_primary_label(
    task: MemUpdateTask,
    run: TaskRunRecord,
    flags: tuple[str, ...],
    layer_values: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, MetricFieldSupport],
    final_snapshot_available: bool,
) -> str | None:
    if run.completion_status is CompletionStatus.NOT_SUPPORTED:
        return None
    if (
        run.completion_status is CompletionStatus.COMPLETED
        and (task.gold.final_state or task.gold.expected_present_objects or task.gold.expected_absent_objects)
        and not final_snapshot_available
    ):
        return None
    if flags:
        return primary_failure(flags)
    if run.completion_status is not CompletionStatus.COMPLETED:
        return None

    decisive_paths: list[str] = []
    if task.gold.actions:
        if any(len(action.target_object_keys) > 1 for action in task.gold.actions):
            decisive_paths.extend(
                ["action_scores.operation_accuracy", "action_scores.value_accuracy"]
            )
        else:
            decisive_paths.append("action_scores.full_action_exact_match")
    if any(query.query_type is QueryType.CURRENT_STATE for query in task.queries):
        decisive_paths.append("answer_scores.normalized_match")
    if any(query.query_type is QueryType.UNRESOLVED_REFERENCE for query in task.queries):
        decisive_paths.append("answer_scores.reference_resolution_accuracy")
    if (
        task.gold.final_state
        or task.gold.expected_present_objects
        or task.gold.expected_absent_objects
    ):
        decisive_paths.append("state_scores.final_state_accuracy")
    for path in decisive_paths:
        layer, leaf = path.split(".", 1)
        if layer_values[layer][leaf] != 1.0:
            return None

    observed = False
    for path in _CORRECTNESS_ONE_PATHS | _CORRECTNESS_ZERO_PATHS:
        layer, leaf = path.split(".", 1)
        value = layer_values[layer][leaf]
        if value is None:
            field_support = support.get(path)
            if field_support is not None and field_support.null_policy != "not_requested":
                if field_support.reason in {
                    SupportReason.RUNTIME_FAILED,
                    SupportReason.MISSING_ARTIFACT,
                }:
                    return None
            continue
        observed = True
        expected = 1.0 if path in _CORRECTNESS_ONE_PATHS else 0.0
        if value != expected:
            return None
    return "correct" if observed else None


def score_task(
    task: MemUpdateTask,
    run: TaskRunRecord,
    capabilities: AdapterCapabilities,
    config: ScorerConfig,
) -> ScoreRecord:
    task = _revalidate(task, MemUpdateTask)
    run = _revalidate(run, TaskRunRecord)
    capabilities = _revalidate(capabilities, AdapterCapabilities)
    config = _revalidate(config, ScorerConfig)
    _validate_extractor_capability_coherence(capabilities)
    if task.schema_version != SCHEMA_VERSION:
        raise ValueError(f"task schema_version must equal {SCHEMA_VERSION}")
    if run.schema_version != SCHEMA_VERSION:
        raise ValueError(f"run schema_version must equal {SCHEMA_VERSION}")
    if run.runtime_record_version != RUNTIME_RECORD_VERSION:
        raise ValueError(
            f"runtime_record_version must equal {RUNTIME_RECORD_VERSION}"
        )
    if task.task_id != run.task_id:
        raise ValueError("task_id mismatch between task and run")
    _validate_runtime_integrity(task, run)
    context = _EvidenceContext(task, run, capabilities, config)

    explicitly_requested = bool(config.requested_metric_fields)
    resolved_requested = (
        tuple(config.requested_metric_fields)
        if explicitly_requested
        else tuple(METRIC_REGISTRY)
    )
    requested = set(resolved_requested)
    if config.strict_capability_check:
        incompatible: dict[str, tuple[str, ...]] = {}
        provenance_issues: dict[str, str] = {}
        for path in resolved_requested:
            definition = METRIC_REGISTRY[path]
            if not _metric_applies_to_task(path, task, context):
                continue
            missing = missing_capabilities(definition, capabilities)
            if missing:
                incompatible[path] = missing
                continue
            issue = _extractor_provenance_issue(path, run, capabilities)
            if issue is not None:
                provenance_issues[path] = issue
        if incompatible:
            detail = "; ".join(
                f"{path}: {', '.join(names)}" for path, names in sorted(incompatible.items())
            )
            raise ValueError(f"requested metrics require unsupported adapter capabilities: {detail}")
        if provenance_issues:
            detail = "; ".join(
                f"{path}: {issue}" for path, issue in sorted(provenance_issues.items())
            )
            raise ValueError(f"extractor provenance mismatch for requested metrics: {detail}")

    layer_values = {
        layer: {field: None for field in model.model_fields}
        for layer, model in SCORE_LAYER_TYPES.items()
    }
    support: dict[str, MetricFieldSupport] = {}
    for path, definition in METRIC_REGISTRY.items():
        layer, leaf = path.split(".", 1)
        if not _metric_applies_to_task(path, task, context):
            support[path] = _support(
                SupportReason.NOT_APPLICABLE,
                f"Metric does not apply to task family/query composition {task.task_family}.",
            )
            continue
        observability_override = layer in _OBSERVABILITY_LAYERS
        if path not in requested and not observability_override:
            support[path] = _support(
                SupportReason.NOT_APPLICABLE,
                "Metric was not selected by requested_metric_fields.",
                null_policy="not_requested",
            )
            continue
        missing = missing_capabilities(definition, capabilities)
        if missing:
            support[path] = _support(
                SupportReason.NOT_SUPPORTED,
                f"Adapter lacks required capabilities: {', '.join(missing)}.",
            )
            continue
        provenance_issue = _extractor_provenance_issue(path, run, capabilities)
        if provenance_issue is not None:
            support[path] = _support(
                SupportReason.MISSING_ARTIFACT,
                provenance_issue,
            )
            continue
        if (
            _runtime_failed(run)
            and definition.runtime_failure_policy == RUNTIME_NULL_POLICY
        ):
            support[path] = _support(
                SupportReason.RUNTIME_FAILED,
                f"Task completion status is {run.completion_status.value}.",
            )
            continue
        if (
            run.completion_status is CompletionStatus.NOT_SUPPORTED
            and layer not in _OBSERVABILITY_LAYERS | {"system_scores"}
        ):
            support[path] = _support(
                SupportReason.NOT_SUPPORTED,
                "Task completion status is not_supported.",
            )
            continue
        value, missing_detail = _compute_metric(path, context)
        if value is None:
            support[path] = _support(
                SupportReason.MISSING_ARTIFACT,
                missing_detail or "Required normalized artifact is absent.",
            )
            continue
        layer_values[layer][leaf] = value

    flags = _derive_failure_flags(context)
    return ScoreRecord(
        scorer_version=SCORER_VERSION,
        task_id=task.task_id,
        run_id=run.run_id,
        adapter_id=run.adapter_id,
        task_family=task.task_family,
        difficulty=task.difficulty,
        completion_status=run.completion_status,
        supported_metric_fields=support,
        **layer_values,
        failure_flags=flags,
        primary_failure=_derive_primary_label(
            task,
            run,
            flags,
            layer_values,
            support,
            context.final_snapshot is not None,
        ),
        legacy_metrics=_legacy_metrics(run, config.legacy_compatibility_mode),
    )


__all__ = ["score_task"]
