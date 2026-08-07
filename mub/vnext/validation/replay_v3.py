from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel, freeze_mapping
from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema, Operation, ReferenceResolutionStatus
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
    ReferenceResolutionSelector,
    TransitionSelector,
    _derivation_consumed_input_indices,
    resolve_selector_version_indices_v3,
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
    disposition: AnswerDisposition | None = None
    resolution_status: ReferenceResolutionStatus | None = None
    selected_candidate_ids: tuple[str, ...] = ()
    selected_versions: tuple[ReplayVersionV3, ...] = ()
    selected_event_ids: tuple[str, ...] = ()
    selected_object_keys: tuple[MemoryObjectKeyV3, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @model_validator(mode="after")
    def _typed_reference_resolution(self) -> Self:
        typed = self.disposition is not None or self.resolution_status is not None
        if not typed:
            if self.selected_candidate_ids:
                raise ValueError("ordinary query resolution cannot select reference candidates")
            return self
        if self.disposition is None or self.resolution_status is None:
            raise ValueError("typed reference resolution requires disposition and resolution_status")
        if self.disposition == AnswerDisposition.ANSWERED:
            if self.resolution_status != ReferenceResolutionStatus.UNIQUE:
                raise ValueError("ANSWERED resolution must be UNIQUE")
            if len(self.selected_candidate_ids) != 1 or self.answer is None:
                raise ValueError("ANSWERED resolution requires one candidate and a non-null answer")
            if len(self.selected_versions) != 1 or len(self.selected_object_keys) != 1:
                raise ValueError("ANSWERED resolution requires one selected replay version and object key")
            if not self.selected_event_ids:
                raise ValueError("ANSWERED resolution requires selected source events")
        elif self.disposition == AnswerDisposition.ABSTAINED:
            if self.resolution_status not in {
                ReferenceResolutionStatus.AMBIGUOUS,
                ReferenceResolutionStatus.NO_MATCH,
            }:
                raise ValueError("ABSTAINED resolution requires AMBIGUOUS or NO_MATCH")
            if (
                self.selected_candidate_ids
                or self.answer is not None
                or self.selected_versions
                or self.selected_event_ids
                or self.selected_object_keys
            ):
                raise ValueError("ABSTAINED resolution cannot carry selected candidates, versions, events, objects, or answer")
        else:
            raise ValueError("UNAVAILABLE is not a resolved gold-query disposition")
        return self

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
        gold = evidence[query.query_id]
        mismatch = resolved.issues or not _same(resolved.answer, gold.answer)
        if query.query_type == QueryTypeV3.UNRESOLVED_REFERENCE:
            mismatch = mismatch or (
                resolved.disposition != gold.disposition
                or resolved.resolution_status != gold.resolution_status
                or resolved.selected_candidate_ids != gold.selected_candidate_ids
            )
        if mismatch:
            return ReplayResultV3(current_state={}, ledgers=(), expected_present=(), expected_absent=(), protected_collateral=(), mutation_count=0, issues=(_issue("query_gold_answer_mismatch", "typed selector resolution does not equal declared gold evidence", f"gold_evidence.{query.query_id}"),))
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
    indices = resolve_selector_version_indices_v3(
        selector,
        ledger.versions,
        event_positions,
        event_times,
        horizon,
    )
    by_index = {version.version_index: version for version in ledger.versions}
    return tuple(by_index[index] for index in indices)


def _resolve_reference_query(
    query: MemoryQueryV3,
    replay: ReplayResultV3,
) -> QueryResolutionV3:
    selector = query.selector
    if not isinstance(selector, ReferenceResolutionSelector):
        return QueryResolutionV3(
            query_id=query.query_id,
            issues=(
                _issue(
                    "unsupported_reference_selector",
                    "unresolved-reference query requires a reference-resolution selector",
                    f"queries.{query.query_id}.selector",
                ),
            ),
        )
    ledgers = replay.ledger_by_identity
    current_by_candidate = {}
    for candidate in selector.reference_candidates:
        ledger = ledgers.get(_identity(candidate.object_key))
        if ledger is None:
            return QueryResolutionV3(
                query_id=query.query_id,
                issues=(
                    _issue(
                        "reference_candidate_missing_object",
                        "reference candidate has no replay ledger",
                        f"queries.{query.query_id}.selector.reference_candidates",
                    ),
                ),
            )
        active = replay.active_versions(ledger)
        if not active or active[-1].status != LedgerEntryStatus.PRESENT:
            return QueryResolutionV3(
                query_id=query.query_id,
                issues=(
                    _issue(
                        "reference_candidate_missing_current_version",
                        "reference candidate has no horizon-active current value",
                        f"queries.{query.query_id}.selector.reference_candidates",
                    ),
                ),
            )
        current_by_candidate[candidate.candidate_id] = (candidate.object_key, active[-1])

    linked_candidate_ids = tuple(
        dict.fromkeys(
            candidate_id
            for reference in selector.surface_references
            for candidate_id in reference.candidate_ids
        )
    )
    if any(candidate_id not in current_by_candidate for candidate_id in linked_candidate_ids):
        return QueryResolutionV3(
            query_id=query.query_id,
            issues=(
                _issue(
                    "reference_graph_unknown_candidate",
                    "surface reference links an undeclared candidate",
                    f"queries.{query.query_id}.selector.surface_references",
                ),
            ),
        )
    if not linked_candidate_ids:
        return QueryResolutionV3(
            query_id=query.query_id,
            disposition=AnswerDisposition.ABSTAINED,
            resolution_status=ReferenceResolutionStatus.NO_MATCH,
        )
    if len(linked_candidate_ids) > 1:
        return QueryResolutionV3(
            query_id=query.query_id,
            disposition=AnswerDisposition.ABSTAINED,
            resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        )

    candidate_id = linked_candidate_ids[0]
    key, version = current_by_candidate[candidate_id]
    return QueryResolutionV3(
        query_id=query.query_id,
        answer=version.value,
        disposition=AnswerDisposition.ANSWERED,
        resolution_status=ReferenceResolutionStatus.UNIQUE,
        selected_candidate_ids=(candidate_id,),
        selected_versions=(version,),
        selected_event_ids=version.source_event_ids,
        selected_object_keys=(key,),
    )


def resolve_query_v3(query: MemoryQueryV3, replay: ReplayResultV3, events=()) -> QueryResolutionV3:
    if replay.issues:
        return QueryResolutionV3(query_id=query.query_id, issues=replay.issues)
    if query.query_type == QueryTypeV3.UNRESOLVED_REFERENCE:
        return _resolve_reference_query(query, replay)
    if isinstance(query.selector, EventAnchorSelector) and not events:
        return QueryResolutionV3(
            query_id=query.query_id,
            issues=(
                _issue(
                    "selector_missing_event_order",
                    "event-anchor resolution requires ordered task events",
                    f"queries.{query.query_id}.selector",
                ),
            ),
        )
    ledgers = replay.ledger_by_identity
    event_positions = {event.event_id: event.sequence_index for event in events}
    event_times = {event.event_id: event.timestamp for event in events if event.timestamp is not None}
    selected_by_object = []
    ordered_g_selector = (
        query.query_type
        in {
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        }
        and isinstance(query.selector, MultiObjectCurrentSelector)
    )
    selection_keys = (
        query.selector.object_keys
        if ordered_g_selector
        else query.target_object_keys
    )
    for key in selection_keys:
        ledger = ledgers.get(_identity(key))
        if ledger is None:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_missing_object", "selector target has no replay ledger", f"queries.{query.query_id}.target_object_keys"),))
        selected = _selected_for(ledger, query.selector, event_positions, event_times, replay.horizon_logical_time)
        if not selected:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_missing_version", "typed selector did not resolve a version", f"queries.{query.query_id}.selector"),))
        if isinstance(query.selector, (EventAnchorSelector, LogicalTimeAnchorSelector)) and len(selected) != 1:
            return QueryResolutionV3(query_id=query.query_id, issues=(_issue("selector_ambiguous_version", "typed selector resolved more than one version", f"queries.{query.query_id}.selector"),))
        selected_by_object.append((key, selected))
    versions = tuple(item for _, selected in selected_by_object for item in selected)
    event_ids = tuple(dict.fromkeys(event_id for item in versions for event_id in item.source_event_ids))
    selected_objects = tuple(key for key, _ in selected_by_object)
    try:
        if (
            query.query_type == QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
            and ordered_g_selector
        ):
            if query.answer_schema is not AnswerSchema.NUMBER:
                return QueryResolutionV3(
                    query_id=query.query_id,
                    selected_versions=versions,
                    selected_event_ids=event_ids,
                    selected_object_keys=selected_objects,
                    issues=(
                        _issue(
                            "unsupported_g_answer_schema",
                            "update-sensitive multi-hop resolution requires number answer schema",
                            f"queries.{query.query_id}.answer_schema",
                        ),
                    ),
                )
            selected_values = [items[-1].value for _, items in selected_by_object]
            answer = selected_values[0]
            for value in selected_values[1:]:
                answer -= value
        elif (
            query.query_type == QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY
            and ordered_g_selector
        ):
            selected_values = [items[-1].value for _, items in selected_by_object]
            if query.answer_schema is AnswerSchema.BOOLEAN:
                answer = all(
                    _same(value, selected_values[0])
                    for value in selected_values[1:]
                )
            elif query.answer_schema is AnswerSchema.NUMBER:
                answer = sum(selected_values)
            else:
                return QueryResolutionV3(
                    query_id=query.query_id,
                    selected_versions=versions,
                    selected_event_ids=event_ids,
                    selected_object_keys=selected_objects,
                    issues=(
                        _issue(
                            "unsupported_g_answer_schema",
                            "multi-object consistency resolution requires boolean or number answer schema",
                            f"queries.{query.query_id}.answer_schema",
                        ),
                    ),
                )
        else:
            if isinstance(query.selector, TransitionSelector):
                answers = [
                    {"from": items[0].value, "to": items[1].value}
                    for _, items in selected_by_object
                ]
            elif isinstance(query.selector, OrderedHistorySelector):
                answers = [
                    [item.value for item in items]
                    for _, items in selected_by_object
                ]
            else:
                answers = [items[-1].value for _, items in selected_by_object]
            keys = [key for key, _ in selected_by_object]
            if isinstance(query.selector, (TransitionSelector, OrderedHistorySelector)):
                answer = answers[0] if len(answers) == 1 else _shape(answers, keys, query.answer_schema)
            else:
                answer = _shape(answers, keys, query.answer_schema)
    except Exception as exc:
        return QueryResolutionV3(
            query_id=query.query_id,
            selected_versions=versions,
            selected_event_ids=event_ids,
            selected_object_keys=selected_objects,
            issues=(_issue("selector_answer_shape_error", str(exc), f"queries.{query.query_id}.answer_schema"),),
        )
    return QueryResolutionV3(
        query_id=query.query_id,
        answer=answer,
        selected_versions=versions,
        selected_event_ids=event_ids,
        selected_object_keys=selected_objects,
    )


def _read_step_value(step, replay: ReplayResultV3):
    versions = _resolve_derivation_read_versions(
        step,
        replay.ledger_by_identity,
        replay.active_versions,
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
            all_operands = [values[parent] for parent in step.input_step_ids]
            consumed = _derivation_consumed_input_indices(step)
            indices = range(len(all_operands)) if consumed is None else consumed
            operands = [all_operands[index] for index in indices]
            operation = step.operation
            if _derivation_step_reads_support(step):
                value = _read_step_value(step, replay)
            elif operation == "abstain":
                value = None
            elif operation in {"identity", "answer", "left", "right"}:
                value = operands[0]
            elif operation in {"seed0", "seed1"}:
                value = operands[0]
            elif operation == "transition":
                value = {"from": operands[0], "to": operands[1]}
            elif operation in {"list", "ordered_history", "collect", "combine", "merge"}:
                value = operands if operands else _read_step_value(step, replay)
            elif operation in {"object", "multi_object", "consistency"}:
                value = operands if operands else _read_step_value(step, replay)
            elif operation == "add":
                value = sum(operands)
            elif operation == "subtract":
                value = operands[0] - operands[1]
            elif operation == "multiply":
                value = operands[0] * operands[1]
            elif operation == "equals":
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
                primary_version_set = _validate_consistency_read_eligibility(
                    evidence,
                    targets,
                    replay.ledger_by_identity,
                    selected_by_target,
                    require_exact_event_coverage=True,
                    version_rows=replay.active_versions,
                )
                if stale_alternative is not None:
                    stale_version_set = _validate_consistency_read_eligibility(
                        stale_alternative,
                        targets,
                        replay.ledger_by_identity,
                        require_exact_event_coverage=True,
                        version_rows=replay.active_versions,
                    )
                    if stale_version_set == primary_version_set:
                        raise ValueError("stale alternative resolves the same ledger versions; version set must differ")
            else:
                _validate_multi_hop_read_eligibility(
                    evidence,
                    targets,
                    replay.ledger_by_identity,
                    selected_by_target,
                    version_rows=replay.active_versions,
                )
                if stale_alternative is not None:
                    _validate_multi_hop_read_eligibility(
                        stale_alternative,
                        targets,
                        replay.ledger_by_identity,
                        selected_by_target,
                        stale_alternative=True,
                        version_rows=replay.active_versions,
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
