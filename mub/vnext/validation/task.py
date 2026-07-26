from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerSchema,
    Difficulty,
    EvaluationMode,
    EventRole,
    Operation,
    QueryType,
    SourceType,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.validation.issues import ValidationIssue, ValidationReport, build_report
from mub.vnext.version import SCHEMA_VERSION


_FAMILY_PROFILE_KEYS = {
    TaskFamily.REPEATED_SAME_SLOT.value: ("update_depth",),
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: (
        "update_depth",
        "active_object_count",
        "cross_slot_interleaving",
    ),
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: (
        "entity_ambiguity",
        "attribute_ambiguity",
    ),
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: ("noop_density", "write_trap_type"),
    TaskFamily.DELETION_FORGETTING.value: ("deletion_scope", "relearning_condition"),
    TaskFamily.CURRENT_HISTORICAL_QUERY.value: (
        "query_type",
        "requested_version_distance",
    ),
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value: (
        "reasoning_depth",
        "active_object_count",
    ),
    TaskFamily.REALISTIC_SOURCE_UPDATE.value: ("source_type", "provenance_class"),
}


def _issue(issues: list[ValidationIssue], code: str, message: str, path: str) -> None:
    issues.append(ValidationIssue(code=code, message=message, path=path, severity="error"))


def _list_field(
    issues: list[ValidationIssue], value: Any, path: str, code: str
) -> list[Any]:
    if not isinstance(value, list):
        _issue(issues, code, f"{path} must be a list", path)
        return []
    return value


def _map_field(
    issues: list[ValidationIssue], value: Any, path: str, code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _issue(issues, code, f"{path} must be a map", path)
        return {}
    return value


def _items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_id(key: Any) -> str | None:
    try:
        object_type = key.object_type
        namespace = key.namespace
        entity = key.entity
        attribute = key.attribute
        subkey = key.subkey
        if not all(isinstance(part, str) and bool(part.strip()) for part in (object_type, namespace, entity, attribute)):
            return None
        if subkey is not None and not isinstance(subkey, str):
            return None
        namespace, entity, attribute = (part.strip() for part in (namespace, entity, attribute))
        normalized_subkey = subkey.strip() if isinstance(subkey, str) else ""
        escape = lambda part: part.replace("%", "%25").replace("|", "%7C")
        return "|".join((escape(namespace), escape(entity), escape(attribute), escape(normalized_subkey)))
    except (AttributeError, TypeError):
        return None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _validate_ids(
    issues: list[ValidationIssue],
    records: list[Any],
    attribute: str,
    path_prefix: str,
    blank_code: str,
    duplicate_code: str,
) -> tuple[list[Any], dict[str, Any]]:
    values: list[Any] = []
    first_by_id: dict[str, Any] = {}
    for index, record in enumerate(records):
        value = getattr(record, attribute, None)
        values.append(value)
        path = f"{path_prefix}[{index}].{attribute}"
        if not _nonblank(value):
            _issue(issues, blank_code, f"{attribute} must be nonblank", path)
        elif value in first_by_id:
            _issue(issues, duplicate_code, f"duplicate {attribute}: {value}", path)
        else:
            first_by_id[value] = record
    return values, first_by_id


def _answer_matches_schema(value: Any, schema: Any) -> bool:
    schema = _enum_value(schema)
    if schema == AnswerSchema.STRING.value:
        return isinstance(value, str)
    if schema == AnswerSchema.NUMBER.value:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    if schema == AnswerSchema.BOOLEAN.value:
        return isinstance(value, bool)
    if schema == AnswerSchema.LIST.value:
        return isinstance(value, (list, tuple))
    if schema == AnswerSchema.OBJECT.value:
        return isinstance(value, Mapping)
    return False


def acceptable_candidates(value: Any, schema: Any, gold_answer: Any = None) -> list[Any]:
    schema = _enum_value(schema)
    if schema == AnswerSchema.LIST.value:
        if value == gold_answer:
            return [value]
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(candidate, (list, tuple)) for candidate in value
        ):
            return list(value)
        return [value]
    if schema == AnswerSchema.OBJECT.value:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def validate_task(task: MemUpdateTask) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        if getattr(task, "schema_version", None) != SCHEMA_VERSION:
            _issue(
                issues,
                "unsupported_schema_version",
                f"schema_version must equal {SCHEMA_VERSION}",
                "schema_version",
            )

        if not _nonblank(getattr(task, "task_id", None)):
            _issue(issues, "blank_task_id", "task_id must be nonblank", "task_id")

        task_family = getattr(task, "task_family", None)
        if not _nonblank(task_family):
            _issue(issues, "invalid_task_family", "task_family must be a nonblank string", "task_family")
        if not isinstance(getattr(task, "difficulty", None), Difficulty):
            _issue(issues, "invalid_difficulty", "difficulty must be a Difficulty value", "difficulty")

        source = getattr(task, "source", None)
        source_id = getattr(source, "source_id", None)
        if not _nonblank(source_id):
            _issue(issues, "blank_source_id", "source_id must be nonblank", "source.source_id")

        if not isinstance(getattr(source, "source_type", None), SourceType):
            _issue(issues, "invalid_source_type", "source_type must be a SourceType value", "source.source_type")

        metadata = getattr(task, "metadata", None)
        if not isinstance(getattr(metadata, "split", None), Split):
            _issue(issues, "invalid_split", "metadata split must be a Split value", "metadata.split")
        difficulty = getattr(task, "difficulty", None)
        profile_name = getattr(metadata, "profile_name", None)
        if isinstance(difficulty, Difficulty) and _enum_value(profile_name) != difficulty.value:
            _issue(
                issues,
                "profile_difficulty_mismatch",
                "metadata.profile_name must equal task difficulty",
                "metadata.profile_name",
            )

        events = _list_field(issues, getattr(task, "events", None), "events", "malformed_events")
        event_ids, event_by_id = _validate_ids(
            issues, events, "event_id", "events", "blank_event_id", "duplicate_event_id"
        )
        indices = [getattr(event, "sequence_index", None) for event in events]
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices) or indices != list(range(len(events))):
            _issue(
                issues,
                "invalid_event_sequence",
                "event sequence indices must be ordered and contiguous",
                "events",
            )

        gold = getattr(task, "gold", None)
        actions = _list_field(issues, getattr(gold, "actions", None), "gold.actions", "malformed_gold_actions")
        action_ids, action_by_id = _validate_ids(
            issues,
            actions,
            "action_id",
            "gold.actions",
            "blank_action_id",
            "duplicate_action_id",
        )
        queries = _list_field(issues, getattr(task, "queries", None), "queries", "malformed_queries")
        query_ids, query_by_id = _validate_ids(
            issues, queries, "query_id", "queries", "blank_query_id", "duplicate_query_id"
        )

        target_objects = _list_field(issues, getattr(task, "target_objects", None), "target_objects", "malformed_target_objects")
        declared_ids: set[str] = set()
        for index, key in enumerate(target_objects):
            canonical_id = _canonical_id(key)
            if canonical_id is None:
                _issue(issues, "malformed_target_object", "target object identity is malformed", f"target_objects[{index}]")
            elif canonical_id in declared_ids:
                _issue(issues, "duplicate_target_object", f"duplicate target object {canonical_id}", f"target_objects[{index}]")
            else:
                declared_ids.add(canonical_id)

        for index, action in enumerate(actions):
            event_id = getattr(action, "event_id", None)
            if not _nonblank(event_id):
                _issue(issues, "blank_action_event_id", "action event_id must be nonblank", f"gold.actions[{index}].event_id")
            elif event_id not in event_by_id:
                _issue(issues, "missing_action_event", f"action references missing event {event_id}", f"gold.actions[{index}].event_id")
            else:
                owner_ids = _list_field(
                    issues,
                    getattr(event_by_id[event_id], "gold_action_ids", None),
                    f"events[{event_ids.index(event_id)}].gold_action_ids",
                    "malformed_event_action_ids",
                )
                if getattr(action, "action_id", None) not in owner_ids:
                    _issue(issues, "action_event_ownership_mismatch", "action is not listed by its owning event", f"gold.actions[{index}].event_id")

            raw_targets = getattr(action, "target_object_keys", None)
            if not isinstance(raw_targets, list):
                _issue(issues, "malformed_action_targets", "action targets must be a list", f"gold.actions[{index}].target_object_keys")
            targets = raw_targets if isinstance(raw_targets, list) else []
            value = getattr(action, "value", None)
            operation = _enum_value(getattr(action, "operation", None))
            scope = _enum_value(getattr(action, "scope", None))
            if operation not in {item.value for item in Operation}:
                _issue(issues, "invalid_action_operation", "action operation is not a recognized enum value", f"gold.actions[{index}].operation")
            if scope not in {item.value for item in ActionScope}:
                _issue(issues, "invalid_action_scope", "action scope is not a recognized enum value", f"gold.actions[{index}].scope")
            if operation == Operation.NOOP.value:
                if targets or value is not None:
                    _issue(issues, "invalid_noop_shape", "NOOP must have no targets and a null value", f"gold.actions[{index}]")
            elif operation in (Operation.ADD.value, Operation.UPDATE.value):
                if not targets or value is None:
                    _issue(issues, "invalid_write_shape", "ADD/UPDATE require targets and a non-null value", f"gold.actions[{index}]")
            elif operation == Operation.DELETE.value:
                if not targets or value is not None:
                    _issue(issues, "invalid_delete_shape", "DELETE requires targets and a null value", f"gold.actions[{index}]")
            else:
                _issue(issues, "invalid_operation", "action operation is unsupported", f"gold.actions[{index}].operation")

            seen_targets: set[str] = set()
            for target_index, key in enumerate(targets):
                canonical_id = _canonical_id(key)
                path = f"gold.actions[{index}].target_object_keys[{target_index}]"
                if canonical_id is None:
                    _issue(issues, "malformed_action_target", "action target identity is malformed", path)
                else:
                    if canonical_id in seen_targets:
                        _issue(issues, "duplicate_action_target", f"duplicate action target {canonical_id}", path)
                    seen_targets.add(canonical_id)
                    if declared_ids is not None and canonical_id not in declared_ids:
                        _issue(issues, "undeclared_action_target", f"action target {canonical_id} is not declared", path)

        for event_index, event in enumerate(events):
            owner_id = getattr(event, "event_id", None)
            role = _enum_value(getattr(event, "role", None))
            if role not in {item.value for item in EventRole}:
                _issue(issues, "invalid_event_role", "event role is not a recognized enum value", f"events[{event_index}].role")
            raw_event_action_ids = getattr(event, "gold_action_ids", None)
            event_action_ids = _list_field(
                issues,
                raw_event_action_ids,
                f"events[{event_index}].gold_action_ids",
                "malformed_event_action_ids",
            )
            seen_event_actions: set[str] = set()
            for ref_index, action_id in enumerate(event_action_ids):
                path = f"events[{event_index}].gold_action_ids[{ref_index}]"
                if not _nonblank(action_id):
                    _issue(issues, "blank_event_action_id", "event action reference must be nonblank", path)
                    continue
                if action_id in seen_event_actions:
                    _issue(issues, "duplicate_event_action_id", f"duplicate event action reference {action_id}", path)
                elif action_id not in action_by_id:
                    _issue(issues, "missing_event_action", f"event references missing action {action_id}", path)
                elif getattr(action_by_id[action_id], "event_id", None) != owner_id:
                    _issue(issues, "action_event_ownership_mismatch", "event references an action owned by another event", path)
                seen_event_actions.add(action_id)

            anchor = _map_field(
                issues,
                getattr(event, "source_anchor", None),
                f"events[{event_index}].source_anchor",
                "malformed_source_anchor",
            )
            if "source_id" in anchor:
                anchor_source_id = anchor.get("source_id")
                if not _nonblank(anchor_source_id) or anchor_source_id != source_id:
                    _issue(issues, "source_anchor_source_mismatch", "source anchor source_id must be nonblank and equal task source_id", f"events[{event_index}].source_anchor.source_id")
            anchor_event_refs: list[tuple[str, Any]] = []
            if "event_id" in anchor:
                anchor_event_refs.append(("event_id", anchor["event_id"]))
            for field in ("event_ids", "source_event_ids"):
                if field in anchor:
                    refs = _list_field(
                        issues,
                        anchor.get(field),
                        f"events[{event_index}].source_anchor.{field}",
                        "malformed_source_anchor_event_ids",
                    )
                    for ref_index, ref in enumerate(refs):
                        anchor_event_refs.append((f"{field}[{ref_index}]", ref))
            for suffix, ref in anchor_event_refs:
                if not _nonblank(ref) or ref not in event_by_id:
                    _issue(issues, "source_anchor_missing_event", f"source anchor references missing event {ref}", f"events[{event_index}].source_anchor.{suffix}")

        sequence = _list_field(
            issues,
            getattr(gold, "action_sequence", None),
            "gold.action_sequence",
            "malformed_action_sequence",
        )
        seen_sequence: set[str] = set()
        for index, action_id in enumerate(sequence):
            path = f"gold.action_sequence[{index}]"
            if not _nonblank(action_id):
                _issue(issues, "blank_action_sequence_id", "action_sequence ID must be nonblank", path)
                continue
            if action_id in seen_sequence:
                _issue(issues, "duplicate_action_sequence_id", f"duplicate action_sequence ID {action_id}", path)
            elif action_id not in action_by_id:
                _issue(issues, "unknown_action_sequence_id", f"unknown action_sequence ID {action_id}", path)
            seen_sequence.add(action_id)
        sequence_ids = {action_id for action_id in sequence if _nonblank(action_id)}
        for index, action_id in enumerate(action_ids):
            if _nonblank(action_id) and action_id not in sequence_ids:
                _issue(issues, "missing_action_sequence_id", f"action_sequence omits {action_id}", f"gold.actions[{index}].action_id")

        gold_source_event_ids = _list_field(
            issues,
            getattr(gold, "gold_source_event_ids", None),
            "gold.gold_source_event_ids",
            "malformed_gold_source_event_ids",
        )
        seen_gold_source_events: set[str] = set()
        for index, event_id in enumerate(gold_source_event_ids):
            path = f"gold.gold_source_event_ids[{index}]"
            if not _nonblank(event_id):
                _issue(issues, "blank_gold_source_event_id", "gold source event ID must be nonblank", path)
                continue
            if event_id in seen_gold_source_events:
                _issue(issues, "duplicate_gold_source_event_id", f"duplicate gold source event ID {event_id}", path)
            elif event_id not in event_by_id:
                _issue(issues, "missing_gold_source_event", f"gold source event {event_id} does not exist", path)
            seen_gold_source_events.add(event_id)

        profile = _map_field(
            issues,
            getattr(metadata, "resolved_profile", None),
            "metadata.resolved_profile",
            "malformed_resolved_profile",
        )
        for key in _FAMILY_PROFILE_KEYS.get(_enum_value(getattr(task, "task_family", None)), ()):
            if key not in profile:
                _issue(issues, "missing_family_profile_key", f"required family profile key is missing: {key}", f"metadata.resolved_profile.{key}")

        expected_present = _list_field(issues, getattr(gold, "expected_present_objects", None), "gold.expected_present_objects", "malformed_expected_present_objects")
        expected_absent = _list_field(issues, getattr(gold, "expected_absent_objects", None), "gold.expected_absent_objects", "malformed_expected_absent_objects")
        present_ids = _validate_declared_keys(issues, expected_present, declared_ids, "gold.expected_present_objects", "undeclared_expected_present_target")
        absent_ids = _validate_declared_keys(issues, expected_absent, None, "gold.expected_absent_objects", "undeclared_expected_absent_target")
        for canonical_id in sorted(present_ids & absent_ids):
            _issue(issues, "expected_presence_overlap", f"object is both expected present and absent: {canonical_id}", "gold.expected_present_objects")

        final_state = _map_field(issues, getattr(gold, "final_state", None), "gold.final_state", "malformed_final_state")
        history = _map_field(issues, getattr(gold, "version_history", None), "gold.version_history", "malformed_version_history")
        for key in sorted(final_state, key=str):
            if key not in declared_ids:
                _issue(issues, "undeclared_final_state_key", f"final_state key is not a declared target: {key}", f"gold.final_state.{key}")
        for key in sorted(history, key=str):
            if key not in declared_ids:
                _issue(issues, "undeclared_version_history_key", f"version_history key is not a declared target: {key}", f"gold.version_history.{key}")
            if not isinstance(history[key], list):
                _issue(issues, "malformed_version_history_values", "version_history values must be lists", f"gold.version_history.{key}")
        for canonical_id in sorted(absent_ids):
            if canonical_id in final_state:
                _issue(issues, "expected_absent_in_final_state", f"expected-absent object appears in final_state: {canonical_id}", f"gold.final_state.{canonical_id}")

        gold_answers = _map_field(issues, getattr(gold, "gold_answers", None), "gold.gold_answers", "malformed_gold_answers")
        acceptable_answers = _map_field(issues, getattr(gold, "acceptable_answers", None), "gold.acceptable_answers", "malformed_acceptable_answers")
        query_id_set = {query_id for query_id in query_ids if _nonblank(query_id)}
        for query_id in sorted(set(gold_answers) - query_id_set, key=str):
            _issue(issues, "unknown_gold_answer_query", f"gold answer references unknown query {query_id}", f"gold.gold_answers.{query_id}")
        for query_id in sorted(set(acceptable_answers) - query_id_set, key=str):
            _issue(issues, "unknown_acceptable_answer_query", f"acceptable answer references unknown query {query_id}", f"gold.acceptable_answers.{query_id}")

        for query_index, query in enumerate(queries):
            query_id = getattr(query, "query_id", None)
            query_type = _enum_value(getattr(query, "query_type", None))
            answer_schema = _enum_value(getattr(query, "answer_schema", None))
            evaluation_mode = _enum_value(getattr(query, "evaluation_mode", None))
            if query_type not in {item.value for item in QueryType}:
                _issue(issues, "invalid_query_type", "query_type is not a recognized enum value", f"queries[{query_index}].query_type")
            if answer_schema not in {item.value for item in AnswerSchema}:
                _issue(issues, "invalid_answer_schema", "answer_schema is not a recognized enum value", f"queries[{query_index}].answer_schema")
            if evaluation_mode not in {item.value for item in EvaluationMode}:
                _issue(issues, "invalid_evaluation_mode", "evaluation_mode is not a recognized enum value", f"queries[{query_index}].evaluation_mode")
            raw_targets = getattr(query, "target_object_keys", None)
            query_metadata = _map_field(
                issues,
                getattr(query, "metadata", None),
                f"queries[{query_index}].metadata",
                "malformed_query_metadata",
            )
            if not isinstance(raw_targets, list):
                _issue(issues, "malformed_query_targets", "query targets must be a list", f"queries[{query_index}].target_object_keys")
            targets = raw_targets if isinstance(raw_targets, list) else []
            if query_type in {
                QueryType.CURRENT_STATE.value,
                QueryType.HISTORICAL_STATE.value,
                QueryType.TRANSITION.value,
                QueryType.MULTI_OBJECT.value,
                QueryType.DELETION_COMPLIANCE.value,
            } and not targets:
                _issue(issues, "missing_query_target", f"{query_type} query requires target(s)", f"queries[{query_index}].target_object_keys")
                if query_type == QueryType.CURRENT_STATE.value:
                    _issue(issues, "missing_current_query_target", "current-state query requires a target", f"queries[{query_index}].target_object_keys")
            seen_query_targets: set[str] = set()
            if query_type == QueryType.HISTORICAL_STATE.value:
                if not targets:
                    _issue(issues, "missing_historical_query_target", "historical-state query requires a target", f"queries[{query_index}].target_object_keys")
                version_index = query_metadata.get("version_index")
                if not isinstance(version_index, int) or isinstance(version_index, bool) or version_index < 0:
                    _issue(issues, "invalid_historical_version_index", "historical version_index must be a strict nonnegative integer", f"queries[{query_index}].metadata.version_index")

            for target_index, key in enumerate(targets):
                canonical_id = _canonical_id(key)
                path = f"queries[{query_index}].target_object_keys[{target_index}]"
                if canonical_id is None:
                    _issue(issues, "malformed_query_target", "query target identity is malformed", path)
                else:
                    if canonical_id in seen_query_targets:
                        _issue(issues, "duplicate_query_target", f"duplicate query target {canonical_id}", path)
                    seen_query_targets.add(canonical_id)
                    allowed_ids = declared_ids | absent_ids if query_type == QueryType.DELETION_COMPLIANCE.value else declared_ids
                    if canonical_id not in allowed_ids:
                        code = "undeclared_deletion_query_target" if query_type == QueryType.DELETION_COMPLIANCE.value else "undeclared_query_target"
                        _issue(issues, code, f"query target {canonical_id} is neither declared nor expected absent", path)

            if not _nonblank(query_id):
                continue
            if query_id not in gold_answers:
                _issue(issues, "missing_gold_answer", f"gold answer is missing for query {query_id}", f"gold.gold_answers.{query_id}")
            elif not _answer_matches_schema(gold_answers[query_id], getattr(query, "answer_schema", None)):
                _issue(issues, "invalid_gold_answer_schema", f"gold answer does not match query schema for {query_id}", f"gold.gold_answers.{query_id}")
            if query_id not in acceptable_answers:
                _issue(issues, "missing_acceptable_answer", f"acceptable answer is missing for query {query_id}", f"gold.acceptable_answers.{query_id}")
            else:
                candidates = acceptable_candidates(acceptable_answers[query_id], getattr(query, "answer_schema", None), gold_answers.get(query_id))
                if not candidates:
                    _issue(issues, "invalid_acceptable_answer_schema", f"acceptable answer support is empty for {query_id}", f"gold.acceptable_answers.{query_id}")
                for candidate_index, candidate in enumerate(candidates):
                    if not _answer_matches_schema(candidate, getattr(query, "answer_schema", None)):
                        _issue(issues, "invalid_acceptable_answer_schema", f"acceptable answer does not match query schema for {query_id}", f"gold.acceptable_answers.{query_id}[{candidate_index}]")
    except Exception as exc:  # final safety net for unexpected validator defects
        _issue(issues, "internal_validation_error", f"unexpected validator failure: {type(exc).__name__}: {exc}", "task")
    return build_report(issues)


def _validate_declared_keys(
    issues: list[ValidationIssue],
    keys: list[Any],
    declared_ids: set[str] | None,
    path_prefix: str,
    undeclared_code: str,
) -> set[str]:
    resolved: set[str] = set()
    for index, key in enumerate(keys):
        canonical_id = _canonical_id(key)
        path = f"{path_prefix}[{index}]"
        if canonical_id is None:
            _issue(issues, "malformed_expected_target", "expected target identity is malformed", path)
        else:
            resolved.add(canonical_id)
            if declared_ids is not None and canonical_id not in declared_ids:
                _issue(issues, undeclared_code, f"expected target {canonical_id} is not declared", path)
    return resolved


__all__ = ["acceptable_candidates", "validate_task"]
