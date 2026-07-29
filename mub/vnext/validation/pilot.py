from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mub.vnext.contracts import EventRole, Operation, TaskFamily
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.validation.issues import ValidationIssue, ValidationReport, build_report
from mub.vnext.validation.replay import validate_gold_replay
from mub.vnext.validation.task import validate_task


_MAX_ISSUES = 128
_NOOP_LIFECYCLES = frozenset({"trap_noop", "independent_noop"})
_CORRECTION_TRAPS = frozenset(
    {"other_entity_correction", "other_attribute_correction"}
)
_NOOP_TRAPS = frozenset({"semantic_near_miss", "duplicate_current"})
_TRAPS = _NOOP_TRAPS | _CORRECTION_TRAPS
_CANONICAL_NOOPS_BY_DIFFICULTY = {"easy": 3, "medium": 6, "hard": 9}
_CANONICAL_DENSITY_BY_DIFFICULTY = {"easy": 0.25, "medium": 0.50, "hard": 0.75}
_NOOP_STATEMENT_MARKERS = (
    "does not assert a current-state change",
    "remains exactly",
    "repeats the exact current target value",
    "does not direct any memory change",
)
_DUPLICATE_CURRENT_RAW_SUFFIXES = (
    "No memory change is required.",
    "Keep memory unchanged.",
    "Do not write anything to memory.",
)


def _issue(code: str, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path, severity="error")


def _items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _identity(key: Any) -> tuple[str, str, str, str | None] | None:
    try:
        namespace = key.namespace
        entity = key.entity
        attribute = key.attribute
        subkey = key.subkey
    except (AttributeError, TypeError):
        return None
    if not all(
        isinstance(part, str) and bool(part.strip())
        for part in (namespace, entity, attribute)
    ):
        return None
    if subkey is not None and not isinstance(subkey, str):
        return None
    return (
        namespace.strip(),
        entity.strip(),
        attribute.strip(),
        subkey.strip() if isinstance(subkey, str) else None,
    )


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _statement_requires_noop(value: Any) -> bool:
    return isinstance(value, str) and any(
        marker in value.casefold() for marker in _NOOP_STATEMENT_MARKERS
    )


def _identity_id(identity: tuple[str, str, str, str | None]) -> str:
    return "|".join(
        part.replace("%", "%25").replace("|", "%7C")
        for part in (*identity[:3], identity[3] or "")
    )


def _duplicate_observation_is_canonical(
    statement: Any,
    raw_text: Any,
    target: tuple[str, str, str, str | None] | None,
    value: Any,
) -> bool:
    if not isinstance(statement, str) or not isinstance(raw_text, str):
        return False
    if target is None or value is None:
        return False
    canonical_statement = (
        f"{target[1]}.{target[2]} remains exactly {value}; "
        "this repeats the exact current target value."
    )
    return statement == canonical_statement and raw_text in {
        f"{canonical_statement} {suffix}"
        for suffix in _DUPLICATE_CURRENT_RAW_SUFFIXES
    }


def _bounded_report(issues: Sequence[ValidationIssue]) -> ValidationReport:
    unique = {
        (issue.code, issue.path, issue.message, issue.severity): issue for issue in issues
    }
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) > _MAX_ISSUES:
        omitted = len(ordered) - (_MAX_ISSUES - 1)
        ordered = ordered[: _MAX_ISSUES - 1]
        ordered.append(
            _issue(
                "family_d_issue_limit_reached",
                f"validation report omitted {omitted} additional deterministic issues",
                "task",
            )
        )
        ordered.sort(key=lambda item: (item.code, item.path, item.message, item.severity))
    return build_report(ordered)


def _event_records(task: MemUpdateTask) -> tuple[list[dict[str, Any]], list[Any]]:
    actions = _items(getattr(getattr(task, "gold", None), "actions", None))
    action_by_id: dict[str, Any] = {}
    for action in actions:
        action_id = getattr(action, "action_id", None)
        if isinstance(action_id, str) and action_id not in action_by_id:
            action_by_id[action_id] = action

    records: list[dict[str, Any]] = []
    for index, event in enumerate(_items(getattr(task, "events", None))):
        metadata = _mapping(getattr(event, "metadata", None))
        action_ids = _items(getattr(event, "gold_action_ids", None))
        records.append(
            {
                "index": index,
                "event": event,
                "metadata": metadata,
                "lifecycle": metadata.get("lifecycle"),
                "trap_type": metadata.get("trap_type"),
                "actions": [
                    action_by_id[action_id]
                    for action_id in action_ids
                    if action_id in action_by_id
                ],
            }
        )
    return records, actions


def _action_targets(action: Any) -> list[tuple[str, str, str, str | None]]:
    return [
        identity
        for key in _items(getattr(action, "target_object_keys", None))
        if (identity := _identity(key)) is not None
    ]


def _family_d_issues(task: MemUpdateTask) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    records, actions = _event_records(task)
    gold = getattr(task, "gold", None)
    metadata = getattr(task, "metadata", None)
    extra = _mapping(getattr(metadata, "extra", None))
    stratification = _mapping(extra.get("stratification"))
    profile = _mapping(getattr(metadata, "resolved_profile", None))

    difficulty = _enum_value(getattr(task, "difficulty", None))
    expected_noop_count = _CANONICAL_NOOPS_BY_DIFFICULTY.get(difficulty)
    if len(records) != 12:
        issues.append(
            _issue(
                "family_d_canonical_event_count_mismatch",
                "Family D tasks require exactly 12 events",
                "events",
            )
        )
    actual_noop_count = sum(
        len(record["actions"]) == 1
        and _enum_value(getattr(record["actions"][0], "operation", None))
        == Operation.NOOP.value
        for record in records
    )
    if expected_noop_count is not None and actual_noop_count != expected_noop_count:
        issues.append(
            _issue(
                "family_d_canonical_noop_count_mismatch",
                f"Family D {difficulty} tasks require exactly {expected_noop_count} NOOP actions",
                "events",
            )
        )
    expected_density = _CANONICAL_DENSITY_BY_DIFFICULTY.get(difficulty)
    density_values = (
        stratification.get("configured_noop_density"),
        stratification.get("observed_noop_density"),
        profile.get("noop_density"),
    )
    if expected_density is not None and any(
        type(value) is not float or value != expected_density
        for value in density_values
    ):
        issues.append(
            _issue(
                "family_d_canonical_noop_density_mismatch",
                f"Family D {difficulty} density metadata must equal {expected_density} exactly",
                "metadata.extra.stratification.configured_noop_density",
            )
        )

    declared_trap = stratification.get("trap_type")
    profile_trap = profile.get("write_trap_type")
    trap_position = stratification.get("trap_position")
    position_valid = (
        type(trap_position) is int and 0 <= trap_position < len(records)
    )
    matching_traps = [
        record for record in records if record["trap_type"] == declared_trap
    ]
    trap_binding_valid = (
        declared_trap in _TRAPS
        and profile_trap == declared_trap
        and position_valid
        and len(matching_traps) == 1
        and matching_traps[0]["index"] == trap_position
    )
    if not trap_binding_valid:
        issues.append(
            _issue(
                "family_d_trap_metadata_mismatch",
                "Family D requires one event-level trap bound to declared type and position",
                "metadata.extra.stratification.trap_position",
            )
        )
    required_noop_indices = (
        {trap_position}
        if declared_trap in _NOOP_TRAPS and position_valid
        else set()
    )

    queries = _items(getattr(task, "queries", None))
    query_targets = (
        _items(getattr(queries[0], "target_object_keys", None)) if queries else []
    )
    target_identities = [
        identity
        for key in query_targets
        if (identity := _identity(key)) is not None
    ]
    target_identity = target_identities[0] if len(target_identities) == 1 else None
    if target_identity is None:
        issues.append(
            _issue(
                "family_d_target_isolation_corruption",
                "Family D requires exactly one well-formed query target",
                "queries[0].target_object_keys",
            )
        )

    for record in records:
        lifecycle = record["lifecycle"]
        event_actions = record["actions"]
        if len(event_actions) != 1:
            issues.append(
                _issue(
                    "family_d_write_discipline_mismatch",
                    "each Family D event must own exactly one gold action",
                    f"events[{record['index']}].gold_action_ids",
                )
            )
        event_role = _enum_value(getattr(record["event"], "role", None))
        statement_noop = _statement_requires_noop(
            record["metadata"].get("surface_statement")
        )
        trap_noop = record["trap_type"] in _NOOP_TRAPS
        role_noop = event_role == EventRole.NOOP_NEAR_MISS.value
        designated_noop = record["index"] in required_noop_indices
        expected_noop = (
            lifecycle in _NOOP_LIFECYCLES
            or trap_noop
            or role_noop
            or statement_noop
            or designated_noop
        )
        if (trap_noop or role_noop or designated_noop) and not (
            trap_noop
            and role_noop
            and lifecycle == "trap_noop"
            and statement_noop
        ):
            issues.append(
                _issue(
                    "family_d_noop_semantics_mismatch",
                    "NOOP trap type, role, lifecycle, and no-change statement disagree",
                    f"events[{record['index']}]",
                )
            )
        for action in event_actions:
            operation = _enum_value(getattr(action, "operation", None))
            targets = _action_targets(action)
            value = getattr(action, "value", None)
            expected_effect = _mapping(getattr(action, "expected_effect", None))
            if expected_noop and (
                operation != Operation.NOOP.value
                or targets
                or value is not None
                or bool(expected_effect)
            ):
                issues.append(
                    _issue(
                        "family_d_noop_state_mutation",
                        "a Family D semantic NOOP carries or claims a state mutation",
                        f"events[{record['index']}].gold_action_ids",
                    )
                )
            if operation == Operation.NOOP.value and not expected_noop:
                issues.append(
                    _issue(
                        "family_d_write_discipline_mismatch",
                        "a Family D write lifecycle was encoded as NOOP",
                        f"events[{record['index']}].gold_action_ids",
                    )
                )
            if (
                target_identity is not None
                and target_identity in targets
                and lifecycle != "target_current"
                and operation != Operation.NOOP.value
            ):
                issues.append(
                    _issue(
                        "family_d_target_isolation_corruption",
                        "only the target_current lifecycle may mutate the Family D query target",
                        f"events[{record['index']}].gold_action_ids",
                    )
                )

    target_records = [
        record for record in records if record["lifecycle"] == "target_current"
    ]
    target_current_value = None
    target_record_valid = False
    if len(target_records) == 1 and len(target_records[0]["actions"]) == 1:
        target_action = target_records[0]["actions"][0]
        target_current_value = getattr(target_action, "value", None)
        target_record_valid = (
            _enum_value(getattr(target_action, "operation", None)) == Operation.ADD.value
            and target_identity is not None
            and _action_targets(target_action) == [target_identity]
            and target_current_value is not None
        )
    if not target_record_valid:
        path = (
            f"events[{target_records[0]['index']}].gold_action_ids"
            if target_records
            else "events"
        )
        issues.append(
            _issue(
                "family_d_target_isolation_corruption",
                "Family D requires one ADD target_current action on only the query target",
                path,
            )
        )

    if target_current_value is not None:
        for record in records:
            for action in record["actions"]:
                operation = _enum_value(getattr(action, "operation", None))
                targets = _action_targets(action)
                if (
                    operation in {Operation.ADD.value, Operation.UPDATE.value}
                    and targets
                    and target_identity not in targets
                    and _same_value(getattr(action, "value", None), target_current_value)
                ):
                    issues.append(
                        _issue(
                            "family_d_distractor_target_value_collision",
                            "a non-target write equals the current target answer",
                            f"events[{record['index']}].gold_action_ids",
                        )
                    )
        canonical_target_id = (
            _identity_id(target_identity) if target_identity is not None else None
        )
        final_state = _mapping(getattr(gold, "final_state", None))
        history = _mapping(getattr(gold, "version_history", None))
        for object_id, value in final_state.items():
            if object_id != canonical_target_id and _same_value(
                value, target_current_value
            ):
                issues.append(
                    _issue(
                        "family_d_distractor_target_value_collision",
                        "gold final state gives a non-target object the target answer",
                        f"gold.final_state.{object_id}",
                    )
                )
        for object_id, values in history.items():
            if object_id == canonical_target_id:
                continue
            for value_index, value in enumerate(_items(values)):
                if _same_value(value, target_current_value):
                    issues.append(
                        _issue(
                            "family_d_distractor_target_value_collision",
                            "gold history gives a non-target object the target answer",
                            f"gold.version_history.{object_id}[{value_index}]",
                        )
                    )

    duplicate_records = [
        record for record in records if record["trap_type"] == "duplicate_current"
    ]
    duplicate_expected = 1 if declared_trap == "duplicate_current" else 0
    duplicate_count = stratification.get("duplicate_current_count")
    if (
        type(duplicate_count) is not int
        or duplicate_count != len(duplicate_records)
        or duplicate_count != duplicate_expected
    ):
        issues.append(
            _issue(
                "family_d_duplicate_current_count_mismatch",
                "duplicate_current_count must equal the observed and trap-implied count",
                "metadata.extra.stratification.duplicate_current_count",
            )
        )

    duplicate_condition = profile.get("duplicate_current_condition")
    duplicate_metadata_valid = (
        type(duplicate_condition) is bool
        and duplicate_condition is bool(duplicate_expected)
    )
    for record in duplicate_records:
        event = record["event"]
        event_actions = record["actions"]
        duplicate_metadata_valid = duplicate_metadata_valid and (
            record["lifecycle"] == "trap_noop"
            and record["metadata"].get("allow_accepted_answer_ambiguity") is True
            and _enum_value(getattr(event, "role", None))
            == EventRole.NOOP_NEAR_MISS.value
            and len(event_actions) == 1
            and _enum_value(getattr(event_actions[0], "operation", None))
            == Operation.NOOP.value
        )
        if not _duplicate_observation_is_canonical(
            record["metadata"].get("surface_statement"),
            getattr(event, "raw_text", None),
            target_identity,
            target_current_value,
        ):
            issues.append(
                _issue(
                    "family_d_duplicate_current_visibility_mismatch",
                    "duplicate-current observation must name the target and current value",
                    f"events[{record['index']}].raw_text",
                )
            )
    ambiguity_records = [
        record
        for record in records
        if record["metadata"].get("allow_accepted_answer_ambiguity") is True
    ]
    if len(ambiguity_records) != duplicate_expected:
        duplicate_metadata_valid = False
    if declared_trap == "duplicate_current" and len(duplicate_records) != 1:
        duplicate_metadata_valid = False
    if not duplicate_metadata_valid:
        issues.append(
            _issue(
                "family_d_duplicate_current_metadata_mismatch",
                "duplicate-current trap metadata, role, or NOOP encoding is inconsistent",
                "events",
            )
        )

    correction_records = [
        record for record in records if record["lifecycle"] == "correction_after"
    ]
    setup_records = [
        record for record in records if record["lifecycle"] == "correction_before"
    ]
    if declared_trap in _CORRECTION_TRAPS:
        lifecycle_valid = len(correction_records) == 1 and len(setup_records) == 1
        correction_action = None
        setup_action = None
        if lifecycle_valid:
            correction_record = correction_records[0]
            setup_record = setup_records[0]
            lifecycle_valid = (
                correction_record["trap_type"] == declared_trap
                and correction_record["index"] > setup_record["index"]
                and len(correction_record["actions"]) == 1
                and len(setup_record["actions"]) == 1
            )
            if lifecycle_valid:
                correction_action = correction_record["actions"][0]
                setup_action = setup_record["actions"][0]
                correction_targets = _action_targets(correction_action)
                setup_targets = _action_targets(setup_action)
                lifecycle_valid = (
                    _enum_value(getattr(correction_action, "operation", None))
                    == Operation.UPDATE.value
                    and _enum_value(getattr(setup_action, "operation", None))
                    == Operation.ADD.value
                    and len(correction_targets) == 1
                    and correction_targets == setup_targets
                    and not _same_value(
                        getattr(correction_action, "value", None),
                        getattr(setup_action, "value", None),
                    )
                )
        if not lifecycle_valid:
            issues.append(
                _issue(
                    "family_d_correction_lifecycle_corruption",
                    "correction traps require an ordered ADD-before/UPDATE-after lifecycle on one object",
                    "events",
                )
            )
        if correction_action is not None:
            correction_targets = _action_targets(correction_action)
            correction_identity = (
                correction_targets[0] if len(correction_targets) == 1 else None
            )
            isolation_valid = (
                correction_identity is not None
                and target_identity is not None
                and correction_identity != target_identity
                and correction_identity[0] == target_identity[0]
                and correction_identity[3] == target_identity[3]
            )
            if isolation_valid and declared_trap == "other_entity_correction":
                isolation_valid = (
                    correction_identity[1] != target_identity[1]
                    and correction_identity[2] == target_identity[2]
                )
            elif isolation_valid and declared_trap == "other_attribute_correction":
                isolation_valid = (
                    correction_identity[1] == target_identity[1]
                    and correction_identity[2] != target_identity[2]
                )
            if not isolation_valid:
                issues.append(
                    _issue(
                        "family_d_target_isolation_corruption",
                        "correction trap identity does not preserve the required non-target relation",
                        f"events[{correction_records[0]['index']}].gold_action_ids",
                    )
                )
    elif correction_records or setup_records:
        issues.append(
            _issue(
                "family_d_correction_lifecycle_corruption",
                "non-correction traps cannot carry correction lifecycle events",
                "events",
            )
        )

    semantic_actions = [
        record["actions"][0] for record in records if len(record["actions"]) == 1
    ]
    action_noops = sum(
        _enum_value(getattr(action, "operation", None)) == Operation.NOOP.value
        for action in semantic_actions
    )
    observed_density = (
        action_noops / len(semantic_actions) if semantic_actions else None
    )
    configured_density = stratification.get("configured_noop_density")
    expected_signature = ",".join(
        str(_enum_value(getattr(action, "operation", None)))
        for action in semantic_actions
    )
    counter_checks = (
        stratification.get("num_events") == len(records),
        stratification.get("noop_count") == action_noops,
        stratification.get("true_write_count")
        == len(semantic_actions) - action_noops,
        stratification.get("operation_signature") == expected_signature,
        profile.get("context_length") == len(records),
        type(observed_density) is float
        and type(configured_density) is float
        and observed_density == configured_density,
        type(observed_density) is float
        and type(stratification.get("observed_noop_density")) is float
        and observed_density == stratification["observed_noop_density"],
        type(observed_density) is float
        and type(profile.get("noop_density")) is float
        and observed_density == profile["noop_density"],
    )
    if not all(counter_checks):
        issues.append(
            _issue(
                "family_d_write_discipline_mismatch",
                "Family D action counts, signature, density, or context metadata do not match events",
                "metadata.extra.stratification",
            )
        )

    target_history = []
    canonical_target_id = None
    history = _mapping(getattr(gold, "version_history", None))
    final_state = _mapping(getattr(gold, "final_state", None))
    if target_identity is not None:
        canonical_target_id = _identity_id(target_identity)
        target_history = _items(history.get(canonical_target_id))
    target_writes = sum(
        target_identity is not None
        and target_identity in _action_targets(action)
        and _enum_value(getattr(action, "operation", None))
        in {Operation.ADD.value, Operation.UPDATE.value}
        for action in actions
    )
    target_gold_matches = (
        canonical_target_id is not None
        and target_current_value is not None
        and len(target_history) == 1
        and _same_value(target_history[0], target_current_value)
        and canonical_target_id in final_state
        and _same_value(final_state[canonical_target_id], target_current_value)
    )
    if target_identity is not None and (
        len(target_history) != target_writes or not target_gold_matches
    ):
        issues.append(
            _issue(
                "family_d_noop_state_mutation",
                "gold target state/history claims a change not justified by the sole target write",
                "gold.version_history",
            )
        )

    return issues


def _validate_family_d_task(task: Any) -> ValidationReport:
    if type(task) is not MemUpdateTask:
        return _bounded_report(
            [
                _issue(
                    "family_d_invalid_task_type",
                    "Family D validation requires an exact MemUpdateTask instance",
                    "task",
                )
            ]
        )
    task_family = getattr(task, "task_family", None)
    if not isinstance(task_family, str) or not task_family.strip():
        return _bounded_report(
            [
                _issue(
                    "family_d_malformed_task",
                    "MemUpdateTask.task_family must be a nonblank string",
                    "task_family",
                )
            ]
        )
    if task_family != TaskFamily.NOOP_WRITE_DISCIPLINE.value:
        return _bounded_report(
            [
                _issue(
                    "family_d_inapplicable_task_family",
                    "Family D validation is inapplicable to this task family",
                    "task_family",
                )
            ]
        )

    issues: list[ValidationIssue] = []
    for validator in (validate_task, validate_gold_replay):
        try:
            issues.extend(validator(task).issues)
        except Exception as exc:
            issues.append(
                _issue(
                    "family_d_malformed_task",
                    f"existing validator failed safely: {type(exc).__name__}: {exc}",
                    "task",
                )
            )
    try:
        issues.extend(_family_d_issues(task))
    except Exception as exc:
        issues.append(
            _issue(
                "family_d_malformed_task",
                f"could not inspect malformed Family D semantics: {type(exc).__name__}: {exc}",
                "task",
            )
        )
    return _bounded_report(issues)


def validate_family_d_task(task: Any) -> ValidationReport:
    """Validate one Family D task without mutation, external I/O, or exception leaks."""
    try:
        return _validate_family_d_task(task)
    except Exception as exc:
        return _bounded_report(
            [
                _issue(
                    "family_d_malformed_task",
                    f"could not inspect malformed Family D task: {type(exc).__name__}",
                    "task",
                )
            ]
        )


__all__ = ["validate_family_d_task"]
