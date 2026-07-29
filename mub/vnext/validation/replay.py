from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from pydantic import field_validator

from mub.vnext.contracts.common import (
    FrozenJsonObject,
    ImmutableContractModel,
    StrictNonnegativeInt,
    freeze_json,
    thaw_json,
)
from mub.vnext.contracts.enums import EventRole, Operation, QueryType
from mub.vnext.contracts.task import GoldAction, MemUpdateTask
from mub.vnext.validation.issues import ValidationIssue, ValidationReport, build_report
from mub.vnext.validation.task import acceptable_candidates


class ReplayResult(ImmutableContractModel):
    final_state: FrozenJsonObject
    version_history: FrozenJsonObject
    mutation_count: StrictNonnegativeInt

    @field_validator("final_state", "version_history")
    @classmethod
    def _freeze_maps(cls, value):
        return freeze_json(value)


def _plain_copy(value: Any) -> Any:
    return deepcopy(thaw_json(value))


def _canonical_id(key: Any) -> str:
    try:
        object_type = key.object_type
        namespace = key.namespace
        entity = key.entity
        attribute = key.attribute
        subkey = key.subkey
        parts = (object_type, namespace, entity, attribute)
        if not all(isinstance(part, str) and bool(part.strip()) for part in parts):
            raise ValueError("identity parts must be nonblank strings")
        if subkey is not None and not isinstance(subkey, str):
            raise ValueError("subkey must be a string or null")
        namespace, entity, attribute = (part.strip() for part in (namespace, entity, attribute))
        normalized_subkey = subkey.strip() if isinstance(subkey, str) else ""
        escape = lambda part: part.replace("%", "%25").replace("|", "%7C")
        return "|".join((escape(namespace), escape(entity), escape(attribute), escape(normalized_subkey)))
    except (AttributeError, TypeError) as exc:
        raise ValueError("malformed target object identity") from exc


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _targets(action: Any) -> list[Any]:
    value = getattr(action, "target_object_keys", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _action_context(index: int, action: Any) -> str:
    action_id = getattr(action, "action_id", None)
    return f"action[{index}] {action_id if isinstance(action_id, str) and action_id else '<missing-id>'}"


def _validate_replay_action(index: int, action: Any) -> tuple[str, list[str], Any]:
    context = _action_context(index, action)
    operation = _enum_value(getattr(action, "operation", None))
    targets = _targets(action)
    value = getattr(action, "value", None)
    if operation == Operation.NOOP.value:
        if targets or value is not None:
            raise ValueError(f"{context}: NOOP requires no targets and a null value")
        return operation, [], value
    if operation in (Operation.ADD.value, Operation.UPDATE.value):
        if not targets or value is None:
            raise ValueError(f"{context}: {operation} requires targets and a non-null value")
    elif operation == Operation.DELETE.value:
        if not targets or value is not None:
            raise ValueError(f"{context}: DELETE requires targets and a null value")
    else:
        raise ValueError(f"{context}: unsupported operation {operation!r}")

    canonical_targets: list[str] = []
    for target in targets:
        try:
            canonical_id = _canonical_id(target)
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        if canonical_id in canonical_targets:
            raise ValueError(f"{context}: duplicate target {canonical_id}")
        canonical_targets.append(canonical_id)
    return operation, canonical_targets, value


class _ReplayError(ValueError):
    def __init__(self, action_index: int, message: str):
        super().__init__(message)
        self.action_index = action_index


def _replay_records(records: Iterable[tuple[int, GoldAction]]) -> ReplayResult:
    state: dict[str, Any] = {}
    history: dict[str, list[Any]] = {}
    mutation_count = 0
    for index, action in records:
        try:
            operation, targets, value = _validate_replay_action(index, action)
        except ValueError as exc:
            raise _ReplayError(index, str(exc)) from exc
        context = _action_context(index, action)
        if operation == Operation.NOOP.value:
            continue
        if operation == Operation.ADD.value:
            present = [target for target in targets if target in state]
            if present:
                raise _ReplayError(index, f"{context}: ADD requires absent target {present[0]}")
        elif operation in (Operation.UPDATE.value, Operation.DELETE.value):
            absent = [target for target in targets if target not in state]
            if absent:
                raise _ReplayError(index, f"{context}: {operation} requires present target {absent[0]}")

        if operation in (Operation.ADD.value, Operation.UPDATE.value):
            for target in targets:
                state[target] = _plain_copy(value)
                history.setdefault(target, []).append(_plain_copy(value))
                mutation_count += 1
        else:
            for target in targets:
                del state[target]
                mutation_count += 1
    return ReplayResult(final_state=_plain_copy(state), version_history=_plain_copy(history), mutation_count=mutation_count)


def replay_actions(actions: Iterable[GoldAction]) -> ReplayResult:
    return _replay_records(enumerate(actions))


def _issue(issues: list[ValidationIssue], code: str, message: str, path: str) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path, severity="error"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _resolved_shape(values: list[Any], target_ids: list[str], schema: Any) -> tuple[bool, Any]:
    schema_value = _enum_value(schema)
    if schema_value == "list":
        return True, values
    if schema_value == "object":
        return True, {target_id: value for target_id, value in zip(target_ids, values)}
    if len(values) == 1:
        return True, values[0]
    return False, None


def _resolve_query(
    query: Any,
    replay: ReplayResult,
    canonical_target_ids: list[str] | None = None,
) -> tuple[bool, Any, str]:
    """Resolve canonical query values; transitions are {"from": old, "to": new}."""
    query_type = _enum_value(getattr(query, "query_type", None))
    if canonical_target_ids is None:
        targets = _targets(query)
        if not targets:
            return False, None, "missing query targets"
        target_ids = []
        seen_target_ids: set[str] = set()
        for target_index, target in enumerate(targets):
            try:
                target_id = _canonical_id(target)
            except ValueError as exc:
                return False, None, f"malformed target at index {target_index}: {exc}"
            if target_id in seen_target_ids:
                return False, None, f"duplicate target at index {target_index}"
            seen_target_ids.add(target_id)
            target_ids.append(target_id)
    else:
        target_ids = list(canonical_target_ids)
        if not target_ids:
            return False, None, "missing query targets"
    metadata = _mapping(getattr(query, "metadata", None))
    schema = getattr(query, "answer_schema", None)
    if query_type == QueryType.CURRENT_STATE.value:
        if not target_ids or any(target not in replay.final_state for target in target_ids):
            return False, None, "current target is absent"
        values = [replay.final_state[target] for target in target_ids]
        if len(values) == 1:
            return True, values[0], ""
        ok, value = _resolved_shape(values, target_ids, schema)
        return ok, value, "current_state requires one target or a structured schema"
    if query_type == QueryType.HISTORICAL_STATE.value:
        index = metadata.get("version_index")
        if len(target_ids) != 1 or not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return False, None, "historical_state requires one target and strict version_index"
        versions = replay.version_history.get(target_ids[0], [])
        if index >= len(versions):
            return False, None, "historical version is unavailable"
        return True, versions[index], ""
    if query_type == QueryType.DELETION_COMPLIANCE.value:
        if not target_ids:
            return False, None, "deletion_compliance requires targets"
        absent = [target not in replay.final_state for target in target_ids]
        schema_value = _enum_value(schema)
        if schema_value == "boolean":
            return True, all(absent), ""
        if schema_value == "list":
            return True, absent, ""
        if schema_value == "object":
            return True, {target: value for target, value in zip(target_ids, absent)}, ""
        if schema_value == "string":
            return True, "absent" if all(absent) else "mixed", ""
        if schema_value == "number":
            return True, sum(absent), ""
        return False, None, "unsupported deletion answer schema"
    if query_type == QueryType.MULTI_OBJECT.value:
        if not target_ids or any(target not in replay.final_state for target in target_ids):
            return False, None, "multi_object target is absent"
        ok, value = _resolved_shape([replay.final_state[target] for target in target_ids], target_ids, schema)
        return ok, value, "multi_object requires list/object schema for multiple targets"
    if query_type == QueryType.TRANSITION.value:
        start = metadata.get("from_version_index")
        end = metadata.get("to_version_index")
        explicit_index = metadata.get("version_index")
        if start is None and end is None and isinstance(explicit_index, int) and not isinstance(explicit_index, bool) and explicit_index > 0:
            start, end = explicit_index - 1, explicit_index
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < 0:
            return False, None, "transition requires strict from_version_index and to_version_index"
        transitions = []
        for target in target_ids:
            versions = replay.version_history.get(target, [])
            if start >= len(versions) or end >= len(versions):
                return False, None, "transition version is unavailable"
            transitions.append({"from": versions[start], "to": versions[end]})
        if len(transitions) == 1:
            return True, transitions[0], ""
        ok, value = _resolved_shape(transitions, target_ids, schema)
        return ok, value, "transition with multiple targets requires list/object schema"
    return False, None, f"unsupported query type {query_type!r}"


def _resolution_issue_path(query_index: int, reason: str) -> str:
    match = re.search(r"(?:malformed|duplicate) target at index (\d+)", reason)
    if match:
        return f"queries[{query_index}].target_object_keys[{match.group(1)}]"
    if "target" in reason:
        return f"queries[{query_index}].target_object_keys"
    return f"queries[{query_index}]"


def _semantic_map(issues: list[ValidationIssue], value: Any, path: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _issue(issues, code, f"{path} must be a map", path)
        return {}
    return value


def _accepted_text_forms(value: Any) -> set[str]:
    forms: set[str] = set()
    if isinstance(value, str):
        if value:
            forms.add(value)
    elif isinstance(value, (bool, int, float)):
        forms.add(_canonical_json(value))
    elif isinstance(value, Mapping):
        forms.add(_canonical_json(value))
        for item in value.values():
            forms.update(_accepted_text_forms(item))
    elif isinstance(value, (list, tuple)):
        forms.add(_canonical_json(value))
        for item in value:
            forms.update(_accepted_text_forms(item))
    return forms


def _text_contains_form(text: str, form: str) -> bool:
    if form and all(ord(char) < 128 for char in form) and any(char.isalnum() for char in form):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
    return form.casefold() in text.casefold()


def _numeric_like_spans(text: str) -> list[str]:
    spans: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        allowed = char.isascii() and (char.isalnum() or char in "_.,+-")
        if not allowed:
            index += 1
            continue
        end = index + 1
        while end < len(text):
            next_char = text[end]
            if not (next_char.isascii() and (next_char.isalnum() or next_char in "_.,+-")):
                break
            end += 1
        span = text[index:end]
        if any(char.isdigit() for char in span):
            if span.endswith(".") and not span.endswith(".."):
                span = span[:-1]
            if span.endswith(",") and not span.endswith(",,"):
                span = span[:-1]
            spans.append(span)
        index = end
    return spans


def _text_contains_value(text: str, value: Any) -> bool:
    if isinstance(value, bool):
        return _text_contains_form(text, "true" if value else "false")
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            accepted = Decimal(str(value))
        except InvalidOperation:
            return False
        spans = _numeric_like_spans(text)
        grammar = re.compile(r"[+-]?(?:(?:\d+)|(?:\d{1,3}(?:,\d{3})+))(?:\.\d+)?(?:[eE][+-]?\d+)?")
        for token in spans:
            if grammar.fullmatch(token) is None:
                continue
            try:
                if Decimal(token.replace(",", "")) == accepted:
                    return True
            except InvalidOperation:
                continue
        return False
    if isinstance(value, str):
        return _text_contains_form(text, value)
    if isinstance(value, Mapping):
        return _text_contains_form(text, _canonical_json(value)) or any(_text_contains_value(text, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return _text_contains_form(text, _canonical_json(value)) or any(_text_contains_value(text, item) for item in value)
    return False


def _canonical_json(value: Any) -> str:
    plain = thaw_json(value)
    return json.dumps(
        plain,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _same_value(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return type(left) is type(right) and left == right


def _ordered_task_action_records(task: MemUpdateTask) -> list[tuple[int, Any]]:
    gold = getattr(task, "gold", None)
    actions = _list(getattr(gold, "actions", None))
    by_id: dict[str, tuple[int, Any]] = {}
    for index, action in enumerate(actions):
        action_id = getattr(action, "action_id", None)
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"gold.actions[{index}] has a missing action_id")
        if action_id in by_id:
            raise ValueError(f"gold.actions contains duplicate action_id {action_id}")
        by_id[action_id] = (index, action)
    sequence = _list(getattr(gold, "action_sequence", None))
    ordered: list[tuple[int, Any]] = []
    seen: set[str] = set()
    for index, action_id in enumerate(sequence):
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"gold.action_sequence[{index}] has a missing action_id")
        if action_id in seen:
            raise ValueError(f"gold.action_sequence[{index}] duplicates {action_id}")
        if action_id not in by_id:
            raise ValueError(f"gold.action_sequence[{index}] references unknown action {action_id}")
        seen.add(action_id)
        ordered.append(by_id[action_id])
    omitted = [action_id for action_id in by_id if action_id not in seen]
    if omitted:
        raise ValueError(f"gold.action_sequence omits action {omitted[0]}")
    return ordered


def _ordered_task_actions(task: MemUpdateTask) -> list[Any]:
    return [action for _, action in _ordered_task_action_records(task)]


def _validate_gold_replay(task: MemUpdateTask) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        records = _ordered_task_action_records(task)
        result = _replay_records(records)
    except _ReplayError as exc:
        _issue(issues, "gold_replay_error", f"gold replay failed: {exc}", f"gold.actions[{exc.action_index}]")
        return build_report(issues)
    except Exception as exc:
        _issue(issues, "gold_replay_error", f"gold replay failed: {exc}", "gold.action_sequence")
        return build_report(issues)

    gold = getattr(task, "gold", None)
    expected_final = _semantic_map(issues, getattr(gold, "final_state", None), "gold.final_state", "malformed_final_state")
    expected_history = _semantic_map(issues, getattr(gold, "version_history", None), "gold.version_history", "malformed_version_history")
    if not _same_value(result.final_state, expected_final):
        _issue(issues, "replay_final_state_mismatch", "replayed final state does not equal gold.final_state", "gold.final_state")
    if not _same_value(result.version_history, expected_history):
        _issue(issues, "replay_version_history_mismatch", "replayed version history does not equal gold.version_history", "gold.version_history")

    for index, key in enumerate(_list(getattr(gold, "expected_present_objects", None))):
        try:
            canonical_id = _canonical_id(key)
        except ValueError as exc:
            _issue(issues, "malformed_expected_present_target", str(exc), f"gold.expected_present_objects[{index}]")
            continue
        if canonical_id not in result.final_state:
            _issue(issues, "expected_present_replay_missing", f"expected-present object is absent after replay: {canonical_id}", f"gold.expected_present_objects[{index}]")
    for index, key in enumerate(_list(getattr(gold, "expected_absent_objects", None))):
        try:
            canonical_id = _canonical_id(key)
        except ValueError as exc:
            _issue(issues, "malformed_expected_absent_target", str(exc), f"gold.expected_absent_objects[{index}]")
            continue
        if canonical_id in result.final_state:
            _issue(issues, "expected_absent_replay_present", f"expected-absent object remains after replay: {canonical_id}", f"gold.expected_absent_objects[{index}]")

    gold_answers = _semantic_map(issues, getattr(gold, "gold_answers", None), "gold.gold_answers", "malformed_gold_answers")
    _semantic_map(issues, getattr(gold, "acceptable_answers", None), "gold.acceptable_answers", "malformed_acceptable_answers")
    for query_index, query in enumerate(_list(getattr(task, "queries", None))):
        query_id = getattr(query, "query_id", None)
        query_type = _enum_value(getattr(query, "query_type", None))
        if query_type == QueryType.UNRESOLVED_REFERENCE.value:
            continue
        resolved, value, reason = _resolve_query(query, result)
        if not resolved:
            if query_type == QueryType.HISTORICAL_STATE.value:
                code = "historical_version_out_of_range" if "unavailable" in reason else "invalid_historical_version_index"
            elif query_type == QueryType.CURRENT_STATE.value:
                code = "current_query_target_absent"
            else:
                code = "unresolved_query_semantics"
            _issue(issues, code, f"query {query_id} cannot be resolved: {reason}", _resolution_issue_path(query_index, reason))
            continue
        if query_id not in gold_answers or not _same_value(gold_answers[query_id], value):
            if query_type == QueryType.HISTORICAL_STATE.value:
                code = "historical_gold_answer_mismatch"
            elif query_type == QueryType.CURRENT_STATE.value:
                code = "current_gold_answer_mismatch"
            else:
                code = "query_gold_answer_mismatch"
            _issue(issues, code, f"canonical gold answer does not equal resolved query value for {query_id}", f"gold.gold_answers.{query_id}")
    return build_report(issues)


def validate_gold_replay(task: MemUpdateTask) -> ValidationReport:
    try:
        return _validate_gold_replay(task)
    except Exception as exc:
        issues: list[ValidationIssue] = []
        _issue(
            issues,
            "malformed_gold_replay_structure",
            f"could not inspect malformed task structure: {type(exc).__name__}: {exc}",
            "task",
        )
        return build_report(issues)


_DISTRACTOR_ROLES = {
    EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value,
    EventRole.SAME_NAME_OTHER_ENTITY.value,
    EventRole.NOOP_NEAR_MISS.value,
    EventRole.NEUTRAL.value,
}


def _event_order(task: MemUpdateTask) -> list[tuple[int, Any]]:
    indexed = list(enumerate(_list(getattr(task, "events", None))))
    return sorted(
        indexed,
        key=lambda item: (
            getattr(item[1], "sequence_index", math.inf)
            if isinstance(getattr(item[1], "sequence_index", None), int)
            and not isinstance(getattr(item[1], "sequence_index", None), bool)
            else math.inf,
            item[0],
        ),
    )


def _superseded_non_target_answer_overlap(
    event_actions: list[Any],
    accepted: Any,
    terminal_future: Mapping[Any, Mapping[str, Any]],
    query_target_ids: set[str],
    terminal_absence: Any,
) -> bool:
    matched = False
    for action in event_actions:
        if _enum_value(getattr(action, "operation", None)) not in (
            Operation.ADD.value,
            Operation.UPDATE.value,
        ) or not _same_value(getattr(action, "value", None), accepted):
            continue
        action_id = getattr(action, "action_id", None)
        target_ids: list[str] = []
        for target in _targets(action):
            try:
                target_ids.append(_canonical_id(target))
            except ValueError:
                return False
        if not target_ids or any(target_id in query_target_ids for target_id in target_ids):
            return False
        future = terminal_future.get(action_id, {})
        if any(
            future.get(target_id) is None
            or future.get(target_id) is terminal_absence
            or _same_value(future.get(target_id), accepted)
            for target_id in target_ids
        ):
            return False
        matched = True
    return matched


def _validate_distractors(
    task: MemUpdateTask,
    *,
    allow_superseded_non_target_answer_overlap: bool = False,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    gold = getattr(task, "gold", None)
    _semantic_map(issues, getattr(gold, "final_state", None), "gold.final_state", "malformed_final_state")
    _semantic_map(issues, getattr(gold, "version_history", None), "gold.version_history", "malformed_version_history")
    queries = _list(getattr(task, "queries", None))
    gold_answers = _semantic_map(issues, getattr(gold, "gold_answers", None), "gold.gold_answers", "malformed_gold_answers")
    acceptable_map = _semantic_map(issues, getattr(gold, "acceptable_answers", None), "gold.acceptable_answers", "malformed_acceptable_answers")
    query_ids = {getattr(query, "query_id", None) for query in queries}

    for query_id in sorted(set(gold_answers) - query_ids, key=str):
        _issue(issues, "unknown_gold_answer_query", f"gold answer references unknown query {query_id}", f"gold.gold_answers.{query_id}")
    for query_id in sorted(set(acceptable_map) - query_ids, key=str):
        _issue(issues, "unknown_acceptable_answer_query", f"acceptable answer references unknown query {query_id}", f"gold.acceptable_answers.{query_id}")

    query_support: dict[Any, list[Any]] = {}
    for query_index, query in enumerate(queries):
        query_id = getattr(query, "query_id", None)
        if query_id not in gold_answers:
            _issue(issues, "missing_query_gold_answer", f"query {query_id} has no canonical gold answer", f"gold.gold_answers.{query_id}")
        if query_id not in acceptable_map:
            _issue(issues, "missing_query_acceptable_answers", f"query {query_id} has no acceptable answer support", f"gold.acceptable_answers.{query_id}")
            query_support[query_id] = []
            continue
        candidates = acceptable_candidates(
            acceptable_map[query_id],
            getattr(query, "answer_schema", None),
            gold_answers.get(query_id),
        )
        query_support[query_id] = candidates
        if query_id in gold_answers and not any(_same_value(gold_answers[query_id], candidate) for candidate in candidates):
            _issue(issues, "canonical_answer_not_acceptable", "canonical gold answer is absent from acceptable support", f"gold.acceptable_answers.{query_id}")

    try:
        records = _ordered_task_action_records(task)
        ordered_actions = [action for _, action in records]
    except Exception as exc:
        _issue(issues, "distractor_replay_error", f"could not order actions: {exc}", "gold.action_sequence")
        return build_report(issues)

    ordered_by_id = {getattr(action, "action_id", None): action for action in ordered_actions}
    terminal_absence = object()
    terminal_future: dict[Any, dict[str, Any]] = {}
    future_effect: dict[str, Any] = {}
    for _, action in reversed(records):
        action_id = getattr(action, "action_id", None)
        target_ids: list[str] = []
        for target in _targets(action):
            try:
                target_ids.append(_canonical_id(target))
            except ValueError:
                continue
        terminal_future[action_id] = {target_id: future_effect.get(target_id) for target_id in target_ids}
        operation = _enum_value(getattr(action, "operation", None))
        for target_id in target_ids:
            if target_id not in future_effect:
                future_effect[target_id] = terminal_absence if operation == Operation.DELETE.value else getattr(action, "value", None)

    for event_index, event in _event_order(task):
        role = _enum_value(getattr(event, "role", None))
        referenced_ids = _list(getattr(event, "gold_action_ids", None))
        event_actions = [
            ordered_by_id[action_id]
            for action_id in referenced_ids
            if action_id in ordered_by_id
        ]
        write_actions = [
            action
            for action in event_actions
            if _enum_value(getattr(action, "operation", None))
            in (Operation.ADD.value, Operation.UPDATE.value)
        ]
        if role == EventRole.STALE_SAME_SLOT.value and not write_actions:
            _issue(issues, "stale_role_without_write", "stale_same_slot event must contain an ADD/UPDATE action", f"events[{event_index}].gold_action_ids")
        if role == EventRole.DUPLICATE_CURRENT.value:
            if not write_actions:
                _issue(issues, "duplicate_current_role_without_write", "duplicate_current event must contain ADD/UPDATE actions", f"events[{event_index}].gold_action_ids")
            for action in event_actions:
                if _enum_value(getattr(action, "operation", None)) not in (Operation.ADD.value, Operation.UPDATE.value) or not _targets(action):
                    _issue(issues, "invalid_duplicate_current_action", "every duplicate_current action must be an ADD/UPDATE with targets", f"events[{event_index}].gold_action_ids")
        if role == EventRole.STALE_SAME_SLOT.value:
            for action in event_actions:
                if _enum_value(getattr(action, "operation", None)) not in (Operation.ADD.value, Operation.UPDATE.value):
                    _issue(issues, "invalid_stale_action", "stale_same_slot event must carry ADD/UPDATE actions", f"events[{event_index}]")
                    continue
                for target_index, target in enumerate(_targets(action)):
                    try:
                        canonical_id = _canonical_id(target)
                    except ValueError:
                        continue
                    terminal_effect = terminal_future.get(getattr(action, "action_id", None), {}).get(canonical_id)
                    path = f"events[{event_index}].gold_action_ids"
                    if terminal_effect is None:
                        _issue(issues, "stale_not_superseded", f"stale target is not later superseded: {canonical_id}", path)
                    elif terminal_effect is not terminal_absence and _same_value(getattr(action, "value", None), terminal_effect):
                        _issue(issues, "stale_value_not_obsolete", f"stale value equals the later/current value for {canonical_id}", path)

    event_by_id = {
        getattr(event, "event_id", None): (event_index, event)
        for event_index, event in enumerate(_list(getattr(task, "events", None)))
    }
    state: dict[str, Any] = {}
    for action_storage_index, action in records:
        event_index, event = event_by_id.get(getattr(action, "event_id", None), (-1, None))
        role = _enum_value(getattr(event, "role", None))
        operation = _enum_value(getattr(action, "operation", None))
        value = getattr(action, "value", None)
        for target_index, target in enumerate(_targets(action)):
            try:
                canonical_id = _canonical_id(target)
            except ValueError as exc:
                _issue(issues, "malformed_distractor_target", str(exc), f"gold.actions[{action_storage_index}].target_object_keys[{target_index}]")
                continue
            if role == EventRole.DUPLICATE_CURRENT.value and operation in (Operation.ADD.value, Operation.UPDATE.value):
                if canonical_id not in state or not _same_value(state[canonical_id], value):
                    _issue(issues, "duplicate_current_value_mismatch", f"duplicate-current value does not equal immediately preceding current value for {canonical_id}", f"events[{event_index}].gold_action_ids")
            if role in _DISTRACTOR_ROLES and operation in (Operation.ADD.value, Operation.UPDATE.value):
                for query_index, query in enumerate(queries):
                    query_id = getattr(query, "query_id", None)
                    query_targets: set[str] = set()
                    for query_target_index, key in enumerate(_targets(query)):
                        try:
                            query_targets.add(_canonical_id(key))
                        except ValueError as exc:
                            _issue(issues, "malformed_distractor_target", str(exc), f"queries[{query_index}].target_object_keys[{query_target_index}]")
                    if canonical_id in query_targets and any(_same_value(value, accepted) for accepted in query_support.get(query_id, [])):
                        _issue(issues, "distractor_establishes_accepted_answer", f"distractor action establishes an accepted answer for query {query_id}", f"events[{event_index}].gold_action_ids")
                        break
            if operation in (Operation.ADD.value, Operation.UPDATE.value):
                state[canonical_id] = _plain_copy(value)
            elif operation == Operation.DELETE.value:
                state.pop(canonical_id, None)

    query_target_ids: set[str] = set()
    for query in queries:
        for target in _targets(query):
            try:
                query_target_ids.add(_canonical_id(target))
            except ValueError:
                continue

    for event_index, event in _event_order(task):
        role = _enum_value(getattr(event, "role", None))
        if role in _DISTRACTOR_ROLES and _mapping(getattr(event, "metadata", None)).get("allow_accepted_answer_ambiguity") is not True:
            referenced_ids = _list(getattr(event, "gold_action_ids", None))
            event_actions = [
                ordered_by_id[action_id]
                for action_id in referenced_ids
                if action_id in ordered_by_id
            ]
            accepted_candidates = [candidate for candidates in query_support.values() for candidate in candidates]
            for field_name in ("raw_text", "normalized_text"):
                text = getattr(event, field_name, None)
                if not isinstance(text, str):
                    continue
                for accepted in accepted_candidates:
                    if not _text_contains_value(text, accepted):
                        continue
                    if (
                        allow_superseded_non_target_answer_overlap
                        and role == EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
                        and _superseded_non_target_answer_overlap(
                            event_actions,
                            accepted,
                            terminal_future,
                            query_target_ids,
                            terminal_absence,
                        )
                    ):
                        continue
                    _issue(issues, "distractor_text_contains_accepted_answer", f"distractor text contains accepted string answer {accepted!r}", f"events[{event_index}].{field_name}")
                    break

    try:
        replay = _replay_records(records)
    except Exception as exc:
        _issue(issues, "distractor_replay_error", f"could not replay actions for query support: {exc}", "gold.action_sequence")
        return build_report(issues)

    for query_index, query in enumerate(queries):
        query_id = getattr(query, "query_id", None)
        if _enum_value(getattr(query, "query_type", None)) == QueryType.CURRENT_STATE.value:
            supported_target_ids: list[str] = []
            resolved_target_ids: list[str] = []
            for target_index, target in enumerate(_targets(query)):
                try:
                    target_id = _canonical_id(target)
                except ValueError as exc:
                    _issue(
                        issues,
                        "malformed_distractor_target",
                        str(exc),
                        f"queries[{query_index}].target_object_keys[{target_index}]",
                    )
                    continue
                if target_id in resolved_target_ids:
                    _issue(
                        issues,
                        "duplicate_query_target",
                        f"duplicate query target {target_id}",
                        f"queries[{query_index}].target_object_keys[{target_index}]",
                    )
                else:
                    resolved_target_ids.append(target_id)
                if target_id in replay.final_state and any(
                    _same_value(replay.final_state[target_id], accepted)
                    for accepted in query_support.get(query_id, [])
                ) and target_id not in supported_target_ids:
                    supported_target_ids.append(target_id)
            target_count = len(resolved_target_ids)
            structured_aggregation = target_count > 1 and _enum_value(getattr(query, "answer_schema", None)) in ("list", "object")
            if not supported_target_ids and not structured_aggregation:
                _issue(issues, "current_answer_not_supported", f"no queried current value supports an accepted answer for {query_id}", f"queries[{query_index}].target_object_keys")
            elif len(supported_target_ids) > 1:
                _issue(issues, "ambiguous_current_answer_support", f"multiple queried current values independently support accepted answers for {query_id}", f"queries[{query_index}].target_object_keys")
        canonical_target_ids = resolved_target_ids if _enum_value(getattr(query, "query_type", None)) == QueryType.CURRENT_STATE.value else None
        resolved, value, reason = _resolve_query(query, replay, canonical_target_ids)
        if not resolved:
            _issue(issues, "unresolved_query_semantics", f"query {query_id} cannot be resolved: {reason}", _resolution_issue_path(query_index, reason))
            continue
        if query_id not in gold_answers or not _same_value(value, gold_answers[query_id]):
            _issue(issues, "query_gold_answer_mismatch", f"resolved query value does not equal canonical gold answer for {query_id}", f"gold.gold_answers.{query_id}")
            continue
        if not any(_same_value(value, accepted) for accepted in query_support.get(query_id, [])):
            _issue(issues, "query_answer_not_supported", f"resolved query value is absent from acceptable support for {query_id}", f"gold.acceptable_answers.{query_id}")

    return build_report(issues)


def validate_distractors(
    task: MemUpdateTask,
    *,
    allow_superseded_non_target_answer_overlap: bool = False,
) -> ValidationReport:
    try:
        return _validate_distractors(
            task,
            allow_superseded_non_target_answer_overlap=(
                allow_superseded_non_target_answer_overlap
            ),
        )
    except Exception as exc:
        issues: list[ValidationIssue] = []
        _issue(
            issues,
            "malformed_distractor_structure",
            f"could not inspect malformed task structure: {type(exc).__name__}: {exc}",
            "task",
        )
        return build_report(issues)


__all__ = [
    "ReplayResult",
    "replay_actions",
    "validate_distractors",
    "validate_gold_replay",
]
