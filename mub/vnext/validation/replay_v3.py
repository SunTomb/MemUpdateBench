from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, freeze_mapping
from mub.vnext.contracts.enums import AnswerSchema, Operation
from mub.vnext.contracts.v3.common import FrozenJsonValue, MemoryObjectKeyV3, object_identity, typed_json_equal
from mub.vnext.contracts.v3.enums import LedgerEntryStatus, QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    EventAnchorSelector,
    ExactVersionSelector,
    LogicalTimeAnchorSelector,
    MemUpdateTaskV3,
    MemoryQueryV3,
    MultiObjectCurrentSelector,
    OrderedHistorySelector,
    PreviousSelector,
    QueryGoldEvidenceV3,
    TransitionSelector,
    _derivation_step_reads_support,
    _resolve_derivation_read_versions,
    _validate_consistency_read_eligibility,
    _validate_multi_hop_read_eligibility,
)
from mub.vnext.validation.issues import ValidationIssue


def _identity(key) -> tuple[str, str, str, str | None]:
    return object_identity(key)


def _canonical_id(key) -> str:
    return key.canonical_id


class ReplayVersionV3(ImmutableContractModel):
    object_key: MemoryObjectKeyV3
    version_index: int = Field(strict=True, ge=0)
    status: LedgerEntryStatus
    value: FrozenJsonValue | None = None
    source_action_id: str
    source_event_ids: tuple[str, ...]
    logical_time: str | None = None
    valid_from_event_id: str | None = None
    valid_until_event_id: str | None = None


class ReplayLedgerV3(ImmutableContractModel):
    object_key: MemoryObjectKeyV3
    versions: tuple[ReplayVersionV3, ...]


class ReplayResultV3(ImmutableContractModel):
    current_state: Mapping[str, ReplayVersionV3]
    ledgers: tuple[ReplayLedgerV3, ...]
    expected_present: tuple[MemoryObjectKeyV3, ...]
    expected_absent: tuple[MemoryObjectKeyV3, ...]
    protected_collateral: tuple[MemoryObjectKeyV3, ...]
    horizon_logical_time: str | None = None
    mutation_count: int = Field(strict=True, ge=0)
    issues: tuple[ValidationIssue, ...] = ()

    @field_validator("current_state")
    @classmethod
    def _freeze_state(cls, value):
        return freeze_mapping(value)

    @property
    def valid(self) -> bool:
        return not self.issues

    def active_versions(self, ledger: ReplayLedgerV3) -> tuple[ReplayVersionV3, ...]:
        if self.horizon_logical_time is None:
            return ledger.versions
        return tuple(
            version
            for version in ledger.versions
            if version.logical_time is None or version.logical_time <= self.horizon_logical_time
        )

    @property
    def obsolete_present_values(self) -> tuple[Any, ...]:
        values = []
        for ledger in self.ledgers:
            for version in self.active_versions(ledger)[:-1]:
                if version.status == LedgerEntryStatus.PRESENT:
                    values.append(version.value)
        return tuple(values)

    @property
    def ledger_by_identity(self) -> Mapping[tuple[str, str, str, str | None], ReplayLedgerV3]:
        return FrozenDict((_identity(ledger.object_key), ledger) for ledger in self.ledgers)


class QueryResolutionV3(ImmutableContractModel):
    query_id: str
    answer: FrozenJsonValue | None = None
    selected_versions: tuple[ReplayVersionV3, ...] = ()
    selected_event_ids: tuple[str, ...] = ()
    selected_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues


class EvidenceEvaluationV3(ImmutableContractModel):
    query_id: str
    answer: FrozenJsonValue | None = None
    stale_alternative_answer: FrozenJsonValue | None = None
    required_object_ids: tuple[str, ...] = ()
    required_event_ids: tuple[str, ...] = ()
    required_step_ids: tuple[str, ...] = ()
    stale_required_object_ids: tuple[str, ...] = ()
    stale_required_event_ids: tuple[str, ...] = ()
    stale_required_step_ids: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues


def _issue(code: str, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path, severity="error")


def _same(left: Any, right: Any) -> bool:
    return typed_json_equal(left, right)


def _action_time(action, event_index: int) -> str | None:
    return action.effective_at


def _build_replay(task: MemUpdateTaskV3) -> ReplayResultV3:
    events = {event.event_id: event for event in task.events}
    histories: dict[tuple[str, str, str, str | None], list[ReplayVersionV3]] = {_identity(key): [] for key in task.target_objects}
    keys = {_identity(key): key for key in task.target_objects}
    state: dict[tuple[str, str, str, str | None], ReplayVersionV3] = {}
    mutation_count = 0
    pending_ttl: list[tuple[str, int, Any]] = []
    horizon_candidates = [event.timestamp for event in task.events if event.timestamp is not None]
    horizon_candidates.extend(
        action.effective_at
        for action in task.actions
        if action.effective_at is not None and not (action.operation == Operation.DELETE and action.scope is not None and action.scope.value == "ttl")
    )
    horizon_candidates.extend(
        query.selector.logical_time
        for query in task.queries
        if isinstance(query.selector, LogicalTimeAnchorSelector)
    )
    horizon = max(horizon_candidates) if horizon_candidates else None

    def append_version(action, key, status, value, logical_time, boundary_event):
        nonlocal mutation_count
        identity = _identity(key)
        versions = histories[identity]
        if versions:
            previous = versions[-1]
            versions[-1] = previous.model_copy(update={"valid_until_event_id": boundary_event})
        version = ReplayVersionV3(
            object_key=keys[identity], version_index=len(versions), status=status, value=value,
            source_action_id=action.action_id, source_event_ids=(action.event_id,),
            logical_time=logical_time, valid_from_event_id=boundary_event,
        )
        versions.append(version)
        if status == LedgerEntryStatus.PRESENT:
            state[identity] = version
        else:
            state.pop(identity, None)
        mutation_count += 1

    def expire_due(now: str):
        due = sorted((item for item in pending_ttl if item[0] <= now), key=lambda item: (item[0], item[1]))
        for logical_time, _, action in due:
            identity = _identity(action.target_object_keys[0])
            if identity in state:
                append_version(action, action.target_object_keys[0], LedgerEntryStatus.TOMBSTONE, None, logical_time, None)
            pending_ttl.remove((logical_time, _, action))

    for action_index, action in enumerate(task.actions):
        event = events[action.event_id]
        now = _action_time(action, event.sequence_index)
        ttl_schedule = action.operation == Operation.DELETE and action.scope is not None and action.scope.value == "ttl"
        expiration_clock = event.timestamp if event.timestamp is not None else (None if ttl_schedule else now)
        if expiration_clock is not None:
            expire_due(expiration_clock)
        targets = tuple(action.target_object_keys)
        if action.operation == Operation.NOOP:
            continue
        if action.operation == Operation.DELETE and action.scope is not None and action.scope.value == "ttl":
            if now is None:
                identity = _identity(action.target_object_keys[0])
                if identity in state:
                    append_version(action, action.target_object_keys[0], LedgerEntryStatus.TOMBSTONE, None, None, action.event_id)
            else:
                pending_ttl.append((now, action_index, action))
            continue
        for key in targets:
            identity = _identity(key)
            if action.operation == Operation.ADD:
                if identity in state:
                    raise ValueError(f"ADD requires absent object {key.canonical_id}")
                append_version(action, key, LedgerEntryStatus.PRESENT, action.value, now, action.event_id)
            elif action.operation == Operation.UPDATE:
                if identity not in state:
                    raise ValueError(f"UPDATE requires present object {key.canonical_id}")
                append_version(action, key, LedgerEntryStatus.PRESENT, action.value, now, action.event_id)
            elif action.operation == Operation.DELETE:
                if identity not in state:
                    raise ValueError(f"DELETE requires present object {key.canonical_id}")
                append_version(action, key, LedgerEntryStatus.TOMBSTONE, None, now, action.event_id)
    for logical_time, _, action in sorted(pending_ttl, key=lambda item: (item[0], item[1])):
        identity = _identity(action.target_object_keys[0])
        if identity in state:
            append_version(action, action.target_object_keys[0], LedgerEntryStatus.TOMBSTONE, None, logical_time, None)

    ledgers = tuple(ReplayLedgerV3(object_key=key, versions=tuple(histories[_identity(key)])) for key in task.target_objects)
    state = {}
    for ledger in ledgers:
        eligible = (
            ledger.versions
            if horizon is None
            else tuple(version for version in ledger.versions if version.logical_time is None or version.logical_time <= horizon)
        )
        if eligible:
            active = max(eligible, key=lambda version: version.version_index)
            if active.status == LedgerEntryStatus.PRESENT:
                state[_identity(ledger.object_key)] = active
    current = {_canonical_id(keys[identity]): version for identity, version in state.items()}
    declared_present = tuple(key for key in task.target_objects if key.canonical_id in current)
    declared_absent = tuple(key for key in task.target_objects if key.canonical_id not in current)
    # Objects not targeted by any DELETE are protected collateral for lifecycle scoring.
    deleted = {_identity(key) for action in task.actions if action.operation == Operation.DELETE for key in action.target_object_keys}
    protected = tuple(key for key in task.target_objects if _identity(key) not in deleted)
    return ReplayResultV3(current_state=current, ledgers=ledgers, expected_present=declared_present, expected_absent=declared_absent, protected_collateral=protected, horizon_logical_time=horizon, mutation_count=mutation_count)


def _declared_matches(task: MemUpdateTaskV3, replay: ReplayResultV3) -> bool:
    declared = {_identity(ledger.object_key): ledger for ledger in task.version_history}
    for replay_ledger in replay.ledgers:
        gold = declared.get(_identity(replay_ledger.object_key))
        if gold is None or len(gold.entries) != len(replay_ledger.versions):
            return False
        for expected, observed in zip(gold.entries, replay_ledger.versions):
            if (expected.version_index != observed.version_index or expected.status != observed.status or
                not _same(expected.value, observed.value) or expected.valid_from_event_id != observed.valid_from_event_id or
                expected.valid_until_event_id != observed.valid_until_event_id or expected.logical_time != observed.logical_time or
                tuple(expected.source_event_ids) != observed.source_event_ids):
                return False
    return True


def replay_task_v3(task: MemUpdateTaskV3) -> ReplayResultV3:
    task = MemUpdateTaskV3.model_validate(task.model_dump(mode="python"))
    try:
        replay = _build_replay(task)
    except Exception as exc:
        return ReplayResultV3(current_state={}, ledgers=(), expected_present=(), expected_absent=(), protected_collateral=(), mutation_count=0, issues=(_issue("gold_replay_error", str(exc), "actions"),))
    if not _declared_matches(task, replay):
        return ReplayResultV3(current_state={}, ledgers=(), expected_present=(), expected_absent=(), protected_collateral=(), mutation_count=0, issues=(_issue("replay_version_history_mismatch", "replayed lifecycle ledger does not equal declared version_history", "version_history"),))
    evidence = {item.query_id: item for item in task.gold_evidence}
    synthesis_kinds = {QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP, QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY}
    for query in task.queries:
        if query.query_type in synthesis_kinds:
            continue
        resolved = resolve_query_v3(query, replay, task.events)
        if resolved.issues or not _same(resolved.answer, evidence[query.query_id].answer):
            return ReplayResultV3(current_state={}, ledgers=(), expected_present=(), expected_absent=(), protected_collateral=(), mutation_count=0, issues=(_issue("query_gold_answer_mismatch", "typed selector answer does not equal declared gold evidence answer", f"gold_evidence.{query.query_id}.answer"),))
    return replay


def replay_actions_v3(task: MemUpdateTaskV3) -> ReplayResultV3:
    return replay_task_v3(task)


def _shape(values: list[Any], keys, schema) -> Any:
    if len(values) == 1:
        value = values[0]
        if schema == AnswerSchema.LIST and isinstance(value, (list, tuple)):
            return value
        if schema == AnswerSchema.OBJECT and isinstance(value, Mapping):
            return value
        if schema not in {AnswerSchema.LIST, AnswerSchema.OBJECT}:
            return value
    if schema == AnswerSchema.LIST:
        return values
    if schema == AnswerSchema.OBJECT:
        return {key.canonical_id: value for key, value in zip(keys, values)}
    raise ValueError("multiple selected objects require list/object answer schema")


def _selected_for(ledger: ReplayLedgerV3, selector, event_positions, event_times, horizon) -> tuple[ReplayVersionV3, ...]:
    versions = ledger.versions
    active_versions = versions if horizon is None else tuple(version for version in versions if version.logical_time is None or version.logical_time <= horizon)
    if isinstance(selector, (CurrentSelector, MultiObjectCurrentSelector)):
        return active_versions[-1:]
    if isinstance(selector, PreviousSelector):
        return active_versions[-2:-1]
    if isinstance(selector, ExactVersionSelector):
        return versions[selector.version_index:selector.version_index + 1]
    if isinstance(selector, TransitionSelector):
        return (versions[selector.from_version_index], versions[selector.to_version_index]) if selector.to_version_index < len(versions) else ()
    if isinstance(selector, OrderedHistorySelector):
        start = 0 if selector.start_version_index is None else selector.start_version_index
        end = len(active_versions) - 1 if selector.end_version_index is None else selector.end_version_index
        return active_versions[start:end + 1]
    if isinstance(selector, EventAnchorSelector):
        anchor = event_positions.get(selector.event_id)
        if anchor is None:
            return ()
        matched = [
            version
            for version in versions
            if version.valid_from_event_id is not None
            and event_positions[version.valid_from_event_id] <= anchor
            and (
                version.valid_until_event_id is None
                or anchor < event_positions[version.valid_until_event_id]
            )
        ]
        event_time = event_times.get(selector.event_id)
        scheduled = [
            version
            for version in versions
            if version.valid_from_event_id is None
            and version.logical_time is not None
            and event_time is not None
            and version.logical_time <= event_time
            and all(event_positions[event_id] <= anchor for event_id in version.source_event_ids)
        ]
        eligible = (*matched, *scheduled)
        return () if not eligible else (max(eligible, key=lambda version: version.version_index),)
    if isinstance(selector, LogicalTimeAnchorSelector):
        eligible = [version for version in versions if version.logical_time is not None and version.logical_time <= selector.logical_time]
        if not eligible:
            return ()
        latest = max(version.logical_time for version in eligible)
        return tuple(version for version in eligible if version.logical_time == latest)
    raise TypeError("unknown typed selector")


def resolve_query_v3(query: MemoryQueryV3, replay: ReplayResultV3, events=()) -> QueryResolutionV3:
    if replay.issues:
        return QueryResolutionV3(query_id=query.query_id, issues=replay.issues)
    ledgers = replay.ledger_by_identity
    event_positions = {event.event_id: event.sequence_index for event in events}
    event_times = {event.event_id: event.timestamp for event in events if event.timestamp is not None}
    if not event_positions:
        for ledger in replay.ledgers:
            for version in ledger.versions:
                for event_id in (version.valid_from_event_id, version.valid_until_event_id, *version.source_event_ids):
                    if event_id is not None and event_id not in event_positions:
                        event_positions[event_id] = len(event_positions)
    selected_by_object = []
    for key in query.target_object_keys:
        ledger = ledgers.get(_identity(key))
        if ledger is None:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_missing_object", "selector target has no replay ledger", f"queries.{query.query_id}.target_object_keys"),))
        selected = _selected_for(ledger, query.selector, event_positions, event_times, replay.horizon_logical_time)
        if not selected:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_missing_version", "typed selector did not resolve a version", f"queries.{query.query_id}.selector"),))
        if isinstance(query.selector, (EventAnchorSelector, LogicalTimeAnchorSelector)) and len(selected) != 1:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_ambiguous_version", "typed selector resolved more than one version", f"queries.{query.query_id}.selector"),))
        selected_by_object.append((key, selected))
    try:
        if isinstance(query.selector, TransitionSelector):
            answers = [{"from": items[0].value, "to": items[1].value} for _, items in selected_by_object]
        elif isinstance(query.selector, OrderedHistorySelector):
            answers = [[item.value for item in items] for _, items in selected_by_object]
        else:
            answers = [items[-1].value for _, items in selected_by_object]
        keys = [key for key, _ in selected_by_object]
        if isinstance(query.selector, (TransitionSelector, OrderedHistorySelector)):
            answer = answers[0] if len(answers) == 1 else _shape(answers, keys, query.answer_schema)
        else:
            answer = _shape(answers, keys, query.answer_schema)
    except Exception as exc:
        return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_answer_shape_error", str(exc), f"queries.{query.query_id}.answer_schema"),))
    versions = tuple(item for _, selected in selected_by_object for item in selected)
    event_ids = tuple(dict.fromkeys(event_id for item in versions for event_id in item.source_event_ids))
    return QueryResolutionV3(query_id=query.query_id, answer=answer, selected_versions=versions, selected_event_ids=event_ids, selected_object_keys=tuple(key for key, _ in selected_by_object))


def _read_step_value(step, replay: ReplayResultV3):
    versions = _resolve_derivation_read_versions(
        step,
        replay.ledger_by_identity,
        lambda ledger: ledger.versions,
    )
    values = [version.value for version in versions]
    return values[0] if len(values) == 1 else values


def evaluate_evidence_v3(
    evidence: QueryGoldEvidenceV3,
    replay: ReplayResultV3,
    stale_alternative: QueryGoldEvidenceV3 | None = None,
    query: MemoryQueryV3 | None = None,
    events=(),
) -> EvidenceEvaluationV3:
    if replay.issues:
        return EvidenceEvaluationV3(query_id=evidence.query_id, issues=replay.issues)

    def evaluate(item):
        values = {}
        for step in item.derivation_steps:
            operands = [values[parent] for parent in step.input_step_ids]
            operation = step.operation
            if _derivation_step_reads_support(step):
                value = _read_step_value(step, replay)
            elif operation in {"identity", "answer", "left", "right"}:
                if len(operands) != 1:
                    raise ValueError(f"{operation} requires one operand")
                value = operands[0]
            elif operation in {"seed0", "seed1"}:
                value = operands[0] if len(operands) == 1 else _read_step_value(step, replay)
            elif operation in {"list", "ordered_history", "collect", "combine", "merge"}:
                value = operands if operands else _read_step_value(step, replay)
            elif operation in {"object", "multi_object", "consistency"}:
                value = operands if operands else _read_step_value(step, replay)
            elif operation == "add":
                value = sum(operands)
            elif operation == "subtract":
                if len(operands) != 2:
                    raise ValueError("subtract requires two ordered operands")
                value = operands[0] - operands[1]
            elif operation == "multiply":
                value = operands[0] * operands[1]
            elif operation == "equals":
                if len(operands) < 2:
                    raise ValueError("equals requires at least two operands")
                value = all(_same(operands[0], operand) for operand in operands[1:])
            elif operation == "all":
                value = all(operands)
            elif operation == "any":
                value = any(operands)
            elif operation == "count":
                value = len(operands[0]) if len(operands) == 1 and isinstance(operands[0], (list, tuple, Mapping)) else len(operands)
            else:
                raise ValueError(f"unsupported_derivation_operation:{operation}")
            values[step.step_id] = value
        return values[item.final_derivation_step_id]

    try:
        if query is not None and query.query_type in {
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        }:
            event_positions = {event.event_id: event.sequence_index for event in events}
            event_times = {event.event_id: event.timestamp for event in events if event.timestamp is not None}
            targets = {_identity(key) for key in query.target_object_keys}
            selected_by_target = {
                identity: _selected_for(
                    replay.ledger_by_identity[identity],
                    query.selector,
                    event_positions,
                    event_times,
                    replay.horizon_logical_time,
                )
                for identity in targets
            }
            if query.query_type == QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY:
                _validate_consistency_read_eligibility(
                    evidence,
                    targets,
                    replay.ledger_by_identity,
                    selected_by_target,
                    version_rows=lambda ledger: ledger.versions,
                )
                if stale_alternative is not None:
                    _validate_consistency_read_eligibility(
                        stale_alternative,
                        targets,
                        replay.ledger_by_identity,
                        require_exact_event_coverage=True,
                        version_rows=lambda ledger: ledger.versions,
                    )
            else:
                _validate_multi_hop_read_eligibility(
                    evidence,
                    targets,
                    replay.ledger_by_identity,
                    selected_by_target,
                    version_rows=lambda ledger: ledger.versions,
                )
                if stale_alternative is not None:
                    _validate_multi_hop_read_eligibility(
                        stale_alternative,
                        targets,
                        replay.ledger_by_identity,
                        selected_by_target,
                        stale_alternative=True,
                        version_rows=lambda ledger: ledger.versions,
                    )
        answer = evaluate(evidence)
        if not _same(answer, evidence.answer):
            raise ValueError("derivation_answer_mismatch")
        stale_answer = None if stale_alternative is None else evaluate(stale_alternative)
        if stale_alternative is not None and not _same(stale_answer, stale_alternative.answer):
            raise ValueError("stale_derivation_answer_mismatch")
    except Exception as exc:
        code = "unsupported_derivation_operation" if str(exc).startswith("unsupported_derivation_operation") else "evidence_replay_error"
        return EvidenceEvaluationV3(query_id=evidence.query_id, issues=(_issue(code, str(exc), f"gold_evidence.{evidence.query_id}"),))
    return EvidenceEvaluationV3(
        query_id=evidence.query_id, answer=answer, stale_alternative_answer=stale_answer,
        required_object_ids=tuple(key.canonical_id for key in evidence.supporting_object_keys),
        required_event_ids=tuple(evidence.supporting_event_ids),
        required_step_ids=tuple(step.step_id for step in evidence.derivation_steps),
        stale_required_object_ids=() if stale_alternative is None else tuple(key.canonical_id for key in stale_alternative.supporting_object_keys),
        stale_required_event_ids=() if stale_alternative is None else tuple(stale_alternative.supporting_event_ids),
        stale_required_step_ids=() if stale_alternative is None else tuple(step.step_id for step in stale_alternative.derivation_steps),
    )


__all__ = ["EvidenceEvaluationV3", "QueryResolutionV3", "ReplayLedgerV3", "ReplayResultV3", "ReplayVersionV3", "evaluate_evidence_v3", "replay_actions_v3", "replay_task_v3", "resolve_query_v3"]
