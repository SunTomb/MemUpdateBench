from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from string import Template
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue, ValidationError

from mub.vnext.contracts import EventRole, Operation, QueryType, TaskFamily
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.generation.catalogs import SAME_NAME_ENTITIES, SURFACE_TEMPLATE_SETS
from mub.vnext.generation.family_d import (
    family_d_duplicate_current_statement,
    family_d_independent_noop_statement,
    family_d_semantic_near_miss_statement,
)
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
)
from mub.vnext.validation.replay import validate_distractors, validate_gold_replay
from mub.vnext.validation.task import acceptable_candidates, validate_task


_MAX_ISSUES = 128
_NOOP_LIFECYCLES = frozenset({"trap_noop", "independent_noop"})
_CORRECTION_TRAPS = frozenset(
    {"other_entity_correction", "other_attribute_correction"}
)
_NOOP_TRAPS = frozenset({"semantic_near_miss", "duplicate_current"})
_TRAPS = _NOOP_TRAPS | _CORRECTION_TRAPS
_CANONICAL_NOOPS_BY_DIFFICULTY = {"easy": 3, "medium": 6, "hard": 9}
_CANONICAL_DENSITY_BY_DIFFICULTY = {"easy": 0.25, "medium": 0.50, "hard": 0.75}
_NOOP_TEMPLATES = tuple(template_set[4] for template_set in SURFACE_TEMPLATE_SETS)
_MAX_NESTED_ITEMS = 64
_MAX_NESTED_DEPTH = 16
_MAX_SCHEMA_NODES = 4096
_PILOT_RELEASE_ID = "vnext-pilot-2026-07"
_FAMILY_A_COUNTS = {
    "easy": (0, 0, 0),
    "medium": (2, 1, 2),
    "hard": (4, 2, 4),
}
_FAMILY_A_AMBIGUITY = {
    "easy": ("none", "none"),
    "medium": ("moderate", "moderate"),
    "hard": ("high", "high"),
}
_FAMILY_A_VERSION_METADATA = {
    "easy": "latest_outdated",
    "medium": "event_index",
    "hard": "none",
}
_FAMILY_A_BASE_PROFILE = {
    "easy": ("chronological", 0.0, "synthetic_direct"),
    "medium": ("retrieval_score", 0.5, "mixed_template"),
    "hard": ("reverse_chronological", 0.75, "semi_natural"),
}
_FAMILY_A_DEPTH_BUCKETS = {1: "1", 4: "4-7", 16: "16+"}
_FAMILY_A_DEPTHS = frozenset(_FAMILY_A_DEPTH_BUCKETS)


def _issue(code: str, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path, severity="error")


def _items(value: Any) -> list[Any]:
    if type(value) is list:
        return list(value)
    if type(value) is tuple:
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError("value must be an exact dict")
    return value


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
    return _canonical_json_value(left) == _canonical_json_value(right)


def _canonical_json_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strict_int_equal(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _identity_id(identity: tuple[str, str, str, str | None]) -> str:
    return "|".join(
        part.replace("%", "%25").replace("|", "%7C")
        for part in (*identity[:3], identity[3] or "")
    )


def _raw_text_matches_noop_template(
    statement: Any,
    raw_text: Any,
    surface_variant: Any,
) -> bool:
    if not isinstance(statement, str) or not isinstance(raw_text, str):
        return False
    if type(surface_variant) is not int or not 0 <= surface_variant < len(
        _NOOP_TEMPLATES
    ):
        return False
    expected = Template(_NOOP_TEMPLATES[surface_variant]).substitute(
        statement=statement
    )
    return raw_text == expected


def _bounded_report(
    issues: list[ValidationIssue] | tuple[ValidationIssue, ...],
    *,
    family: str = "d",
) -> ValidationReport:
    unique = {
        (issue.code, issue.path, issue.message, issue.severity): issue for issue in issues
    }
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) > _MAX_ISSUES:
        omitted = len(ordered) - (_MAX_ISSUES - 1)
        ordered = ordered[: _MAX_ISSUES - 1]
        ordered.append(
            _issue(
                f"family_{family}_issue_limit_reached",
                f"validation report omitted {omitted} additional deterministic issues",
                "task",
            )
        )
        ordered.sort(key=lambda item: (item.code, item.path, item.message, item.severity))
    return build_report(ordered)


def _schema_issue(
    family: str,
    suffix: str,
    message: str,
    path: str,
) -> ValidationIssue:
    return _issue(
        f"family_{family}_{suffix}",
        message.replace("{family}", f"Family {family.upper()}"),
        path,
    )


def _inspect_json_value(
    issues: list[ValidationIssue],
    value: Any,
    path: str,
    budget: list[int],
    active: set[int],
    depth: int,
    family: str,
) -> None:
    if depth > _MAX_NESTED_DEPTH or budget[0] <= 0:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                "nested JSON structures exceed the {family} inspection limit",
                path,
            )
        )
        return
    budget[0] -= 1
    if type(value) is float:
        if not math.isfinite(value):
            issues.append(
                _schema_issue(
                    family,
                    "non_finite_json_number",
                    "nested JSON numbers must be finite",
                    path,
                )
            )
        return
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is dict:
        _inspect_json_mapping(
            issues,
            value,
            path,
            budget,
            active,
            depth,
            family,
        )
        return
    if type(value) is list:
        identity = id(value)
        if identity in active:
            issues.append(
                _schema_issue(
                    family,
                    "cyclic_json",
                    "nested JSON collections cannot contain cycles",
                    path,
                )
            )
            return
        if len(value) > _MAX_NESTED_ITEMS:
            issues.append(
                _schema_issue(
                    family,
                    "input_size_limit",
                    "nested list exceeds the {family} inspection limit",
                    path,
                )
            )
            return
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _inspect_json_value(
                    issues,
                    item,
                    f"{path}[{index}]",
                    budget,
                    active,
                    depth + 1,
                    family,
                )
                if budget[0] <= 0 and index + 1 < len(value):
                    issues.append(
                        _schema_issue(
                            family,
                            "input_size_limit",
                            "nested JSON structures exceed the {family} inspection budget",
                            path,
                        )
                    )
                    return
        finally:
            active.remove(identity)
        return
    issues.append(
        _schema_issue(
            family,
            "malformed_json",
            "nested JSON values must use exact built-in JSON types",
            path,
        )
    )


def _inspect_json_mapping(
    issues: list[ValidationIssue],
    value: Any,
    path: str,
    budget: list[int],
    active: set[int],
    depth: int,
    family: str,
) -> None:
    if depth > _MAX_NESTED_DEPTH:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                "nested JSON structures exceed the {family} depth limit",
                path,
            )
        )
        return
    if type(value) is not dict:
        issues.append(
            _schema_issue(
                family,
                "malformed_mapping",
                f"{path} must be an exact dict",
                path,
            )
        )
        return
    identity = id(value)
    if identity in active:
        issues.append(
            _schema_issue(
                family,
                "cyclic_json",
                "nested JSON collections cannot contain cycles",
                path,
            )
        )
        return
    if len(value) > _MAX_NESTED_ITEMS:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                f"{path} exceeds the {{family}} inspection limit",
                path,
            )
        )
        return
    active.add(identity)
    try:
        for index, (key, item) in enumerate(value.items()):
            if type(key) is not str:
                issues.append(
                    _schema_issue(
                        family,
                        "malformed_mapping",
                        f"{path} keys must be exact strings",
                        f"{path}[{index}]",
                    )
                )
                continue
            _inspect_json_value(
                issues,
                item,
                f"{path}.{key}",
                budget,
                active,
                depth + 1,
                family,
            )
            if budget[0] <= 0 and index + 1 < len(value):
                issues.append(
                    _schema_issue(
                        family,
                        "input_size_limit",
                        "nested JSON structures exceed the {family} inspection budget",
                        path,
                    )
                )
                return
    finally:
        active.remove(identity)


def _unwrap_annotation(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _is_json_annotation(annotation: Any) -> bool:
    return annotation is JsonValue or (
        type(annotation).__name__ == "TypeAliasType"
        and getattr(annotation, "__name__", None) == "JsonValue"
    )


def _annotation_outer_matches(value: Any, annotation: Any) -> bool:
    annotation = _unwrap_annotation(annotation)
    if annotation is Any or _is_json_annotation(annotation):
        return type(value) in {dict, list, str, bool, int, float, type(None)}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, UnionType}:
        return any(_annotation_outer_matches(value, item) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if origin is list:
        return type(value) is list
    if origin is dict or origin is Mapping:
        return type(value) is dict
    if origin is tuple:
        return type(value) is tuple
    if annotation is type(None):
        return value is None
    if isinstance(annotation, type):
        return type(value) is annotation
    return False


def _inspect_schema_value(
    issues: list[ValidationIssue],
    value: Any,
    annotation: Any,
    path: str,
    budget: list[int],
    active: set[int],
    depth: int,
    family: str,
) -> None:
    if depth > _MAX_NESTED_DEPTH or budget[0] <= 0:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                "contract graph exceeds the {family} inspection limit",
                path,
            )
        )
        return
    budget[0] -= 1
    annotation = _unwrap_annotation(annotation)
    if annotation is Any or _is_json_annotation(annotation):
        _inspect_json_value(
            issues,
            value,
            path,
            budget,
            active,
            depth,
            family,
        )
        return
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, UnionType}:
        matching = [item for item in args if _annotation_outer_matches(value, item)]
        if not matching:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    "field value does not match any declared union member",
                    path,
                )
            )
            return
        _inspect_schema_value(
            issues,
            value,
            matching[0],
            path,
            budget,
            active,
            depth,
            family,
        )
        return
    if origin is Literal:
        if not _annotation_outer_matches(value, annotation):
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    "field value does not match its declared literal",
                    path,
                )
            )
        return
    if origin is list:
        if type(value) is not list:
            issues.append(
                _schema_issue(
                    family,
                    "malformed_collection",
                    "field must be an exact list",
                    path,
                )
            )
            return
        identity = id(value)
        if identity in active:
            issues.append(
                _schema_issue(
                    family,
                    "cyclic_json",
                    "contract collections cannot contain cycles",
                    path,
                )
            )
            return
        if len(value) > _MAX_NESTED_ITEMS:
            issues.append(
                _schema_issue(
                    family,
                    "input_size_limit",
                    "field list exceeds the {family} inspection limit",
                    path,
                )
            )
            return
        item_annotation = args[0] if args else Any
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _inspect_schema_value(
                    issues,
                    item,
                    item_annotation,
                    f"{path}[{index}]",
                    budget,
                    active,
                    depth + 1,
                    family,
                )
        finally:
            active.remove(identity)
        return
    if origin is dict or origin is Mapping:
        if type(value) is not dict:
            issues.append(
                _schema_issue(
                    family,
                    "malformed_mapping",
                    "field must be an exact dict",
                    path,
                )
            )
            return
        identity = id(value)
        if identity in active:
            issues.append(
                _schema_issue(
                    family,
                    "cyclic_json",
                    "contract mappings cannot contain cycles",
                    path,
                )
            )
            return
        if len(value) > _MAX_NESTED_ITEMS:
            issues.append(
                _schema_issue(
                    family,
                    "input_size_limit",
                    "field mapping exceeds the {family} inspection limit",
                    path,
                )
            )
            return
        key_annotation, value_annotation = args if len(args) == 2 else (Any, Any)
        active.add(identity)
        try:
            for index, (key, item) in enumerate(value.items()):
                key_path = f"{path}[{index}]"
                _inspect_schema_value(
                    issues,
                    key,
                    key_annotation,
                    key_path,
                    budget,
                    active,
                    depth + 1,
                    family,
                )
                item_path = f"{path}.{key}" if type(key) is str else key_path
                _inspect_schema_value(
                    issues,
                    item,
                    value_annotation,
                    item_path,
                    budget,
                    active,
                    depth + 1,
                    family,
                )
        finally:
            active.remove(identity)
        return
    if origin is tuple:
        if type(value) is not tuple:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    "field must be an exact tuple",
                    path,
                )
            )
            return
        if len(value) > _MAX_NESTED_ITEMS:
            issues.append(
                _schema_issue(
                    family,
                    "input_size_limit",
                    "field tuple exceeds the {family} inspection limit",
                    path,
                )
            )
            return
        variadic = len(args) == 2 and args[1] is Ellipsis
        if not variadic and args and len(value) != len(args):
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    "field tuple length does not match its declaration",
                    path,
                )
            )
            return
        for index, item in enumerate(value):
            item_annotation = args[0] if variadic else args[index]
            _inspect_schema_value(
                issues,
                item,
                item_annotation,
                f"{path}[{index}]",
                budget,
                active,
                depth + 1,
                family,
            )
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if type(value) is not annotation:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    f"field must be an exact {annotation.__name__}",
                    path,
                )
            )
            return
        _inspect_contract_model(
            issues,
            value,
            annotation,
            path,
            budget,
            active,
            depth,
            family,
        )
        return
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not annotation:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_enum_type",
                    f"field must be an exact {annotation.__name__}",
                    path,
                )
            )
        return
    if annotation is float:
        if type(value) is not float:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    "field must be an exact float",
                    path,
                )
            )
        elif not math.isfinite(value):
            issues.append(
                _schema_issue(
                    family,
                    "non_finite_json_number",
                    "float fields must be finite",
                    path,
                )
            )
        return
    if annotation in {str, bool, int, type(None)}:
        if type(value) is not annotation:
            issues.append(
                _schema_issue(
                    family,
                    "invalid_field_type",
                    f"field must be an exact {annotation.__name__}",
                    path,
                )
            )
        return
    issues.append(
        _schema_issue(
            family,
            "invalid_field_type",
            "field uses an unsupported or malformed runtime annotation",
            path,
        )
    )


def _inspect_contract_model(
    issues: list[ValidationIssue],
    model: BaseModel,
    expected_type: type[BaseModel],
    path: str,
    budget: list[int],
    active: set[int],
    depth: int,
    family: str,
) -> None:
    if depth > _MAX_NESTED_DEPTH or budget[0] <= 0:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                "contract graph exceeds the {family} inspection limit",
                path,
            )
        )
        return
    if type(model) is not expected_type:
        issues.append(
            _schema_issue(
                family,
                "invalid_field_type",
                f"record must be an exact {expected_type.__name__}",
                path,
            )
        )
        return
    identity = id(model)
    if identity in active:
        issues.append(
            _schema_issue(
                family,
                "cyclic_json",
                "contract records cannot contain cycles",
                path,
            )
        )
        return
    raw = object.__getattribute__(model, "__dict__")
    if type(raw) is not dict:
        issues.append(
            _schema_issue(
                family,
                "malformed_record",
                "contract record storage must be an exact dict",
                path,
            )
        )
        return
    fields = expected_type.model_fields
    if len(raw) > len(fields) + _MAX_NESTED_ITEMS:
        issues.append(
            _schema_issue(
                family,
                "input_size_limit",
                "contract record storage exceeds the {family} inspection limit",
                path,
            )
        )
        return
    missing = [name for name in fields if name not in raw]
    extra = [key for key in raw if type(key) is not str or key not in fields]
    for name in missing:
        issues.append(
            _schema_issue(
                family,
                "malformed_record",
                f"contract record is missing declared field {name}",
                f"{path}.{name}",
            )
        )
    for index, key in enumerate(extra):
        extra_path = f"{path}.{key}" if type(key) is str else f"{path}[{index}]"
        issues.append(
            _schema_issue(
                family,
                "malformed_record",
                "contract record contains undeclared internal field",
                extra_path,
            )
        )
    try:
        pydantic_extra = object.__getattribute__(model, "__pydantic_extra__")
    except AttributeError:
        pydantic_extra = None
    if pydantic_extra is not None and (
        type(pydantic_extra) is not dict or bool(pydantic_extra)
    ):
        issues.append(
            _schema_issue(
                family,
                "malformed_record",
                "contract record contains undeclared Pydantic extras",
                path,
            )
        )
    active.add(identity)
    try:
        for name, field in fields.items():
            if name not in raw:
                continue
            _inspect_schema_value(
                issues,
                raw[name],
                field.annotation,
                f"{path}.{name}",
                budget,
                active,
                depth + 1,
                family,
            )
    finally:
        active.remove(identity)


def _schema_preflight_issues(
    task: MemUpdateTask,
    *,
    family: str = "d",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _inspect_contract_model(
        issues,
        task,
        MemUpdateTask,
        "task",
        [_MAX_SCHEMA_NODES],
        set(),
        0,
        family,
    )
    return issues


def _constraint_error_path(location: tuple[Any, ...]) -> str:
    path = "task"
    for part in location:
        if type(part) is int:
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _contract_constraint_issues(
    task: MemUpdateTask,
    *,
    family: str,
) -> list[ValidationIssue]:
    try:
        payload = task.model_dump(
            mode="python",
            round_trip=True,
            warnings="none",
        )
        MemUpdateTask.model_validate(payload)
    except ValidationError as exc:
        issues = []
        for error in exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )[:_MAX_ISSUES]:
            location = error.get("loc", ())
            error_type = error.get("type", "contract_error")
            issues.append(
                _issue(
                    f"family_{family}_contract_constraint_violation",
                    f"full contract revalidation failed ({error_type})",
                    _constraint_error_path(tuple(location)),
                )
            )
        return issues
    except Exception:
        return [
            _issue(
                f"family_{family}_contract_constraint_violation",
                "full contract revalidation rejected safe contract data",
                "task",
            )
        ]
    return []


def _preflight_issues(task: MemUpdateTask) -> list[ValidationIssue]:
    issues = _schema_preflight_issues(task)
    if issues:
        return issues

    bounded_collections = (
        ("events", task.events, 12),
        ("queries", task.queries, 8),
        ("gold.actions", task.gold.actions, 12),
        ("gold.action_sequence", task.gold.action_sequence, 12),
        ("gold.gold_source_event_ids", task.gold.gold_source_event_ids, 12),
    )
    for path, values, limit in bounded_collections:
        if len(values) <= limit:
            continue
        issues.append(
            _issue(
                "family_d_input_size_limit",
                f"{path} exceeds the Family D inspection limit",
                path,
            )
        )
        if path == "events":
            issues.append(
                _issue(
                    "family_d_canonical_event_count_mismatch",
                    "Family D tasks require exactly 12 events",
                    "events",
                )
            )

    for index, event in enumerate(task.events):
        if len(event.gold_action_ids) > 1:
            issues.append(
                _issue(
                    "family_d_input_size_limit",
                    "event gold_action_ids exceeds the Family D cardinality limit",
                    f"events[{index}].gold_action_ids",
                )
            )
    for index, action in enumerate(task.gold.actions):
        if len(action.target_object_keys) > 1:
            issues.append(
                _issue(
                    "family_d_input_size_limit",
                    "action target_object_keys exceeds the Family D cardinality limit",
                    f"gold.actions[{index}].target_object_keys",
                )
            )
    for index, query in enumerate(task.queries):
        for field_name in (
            "target_object_keys",
            "reference_candidates",
            "surface_references",
        ):
            values = object.__getattribute__(query, field_name)
            if len(values) > 1:
                issues.append(
                    _issue(
                        "family_d_input_size_limit",
                        f"query {field_name} exceeds the Family D cardinality limit",
                        f"queries[{index}].{field_name}",
                    )
                )
    return issues


def _family_a_preflight_issues(
    task: MemUpdateTask,
    *,
    schema_checked: bool = False,
) -> list[ValidationIssue]:
    if not schema_checked:
        schema_issues = _schema_preflight_issues(task, family="a")
        if schema_issues:
            return schema_issues

    issues: list[ValidationIssue] = []
    bounded_collections = (
        ("events", task.events, 32),
        ("queries", task.queries, 1),
        ("gold.actions", task.gold.actions, 32),
        ("gold.action_sequence", task.gold.action_sequence, 32),
        ("gold.gold_source_event_ids", task.gold.gold_source_event_ids, 1),
    )
    for path, values, limit in bounded_collections:
        if len(values) > limit:
            issues.append(
                _issue(
                    "family_a_input_size_limit",
                    f"{path} exceeds the Family A inspection limit",
                    path,
                )
            )
    for index, event in enumerate(task.events):
        if len(event.gold_action_ids) > 1:
            issues.append(
                _issue(
                    "family_a_input_size_limit",
                    "event gold_action_ids exceeds the Family A cardinality limit",
                    f"events[{index}].gold_action_ids",
                )
            )
    for index, action in enumerate(task.gold.actions):
        if len(action.target_object_keys) > 1:
            issues.append(
                _issue(
                    "family_a_input_size_limit",
                    "action target_object_keys exceeds the Family A cardinality limit",
                    f"gold.actions[{index}].target_object_keys",
                )
            )
    for index, query in enumerate(task.queries):
        for field_name in (
            "target_object_keys",
            "reference_candidates",
            "surface_references",
        ):
            values = object.__getattribute__(query, field_name)
            if len(values) > 1:
                issues.append(
                    _issue(
                        "family_a_input_size_limit",
                        f"query {field_name} exceeds the Family A cardinality limit",
                        f"queries[{index}].{field_name}",
                    )
                )
    return issues


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


def _current_answer_issues(
    task: MemUpdateTask,
    current_value: Any,
    *,
    family: str,
) -> list[ValidationIssue]:
    queries = _items(task.queries)
    query = queries[0] if len(queries) == 1 else None
    query_id = getattr(query, "query_id", None)
    gold_answers = _mapping(task.gold.gold_answers)
    acceptable_answers = _mapping(task.gold.acceptable_answers)
    canonical_answers = _mapping(task.gold.canonical_answers)
    candidates: list[Any] = []
    if type(query_id) is str and query_id in acceptable_answers:
        candidates = acceptable_candidates(
            acceptable_answers[query_id],
            getattr(query, "answer_schema", None),
            gold_answers.get(query_id),
        )
    valid = (
        query is not None
        and _enum_value(getattr(query, "query_type", None))
        == QueryType.CURRENT_STATE.value
        and type(query_id) is str
        and set(gold_answers) == {query_id}
        and set(acceptable_answers) == {query_id}
        and not canonical_answers
        and current_value is not None
        and _same_value(gold_answers.get(query_id), current_value)
        and len(candidates) == 1
        and _same_value(candidates[0], current_value)
    )
    if valid:
        return []
    return [
        _issue(
            f"family_{family}_multiple_current_answers",
            f"Family {family.upper()} answer structures must admit one current answer",
            "gold.acceptable_answers",
        )
    ]


def _same_name_other_entity(
    target: tuple[str, str, str, str | None],
    candidate: tuple[str, str, str, str | None],
) -> bool:
    return (
        candidate != target
        and candidate[1] != target[1]
        and candidate[2] == target[2]
        and candidate[3] == target[3]
        and any(
            target[1] in group and candidate[1] in group
            for group in SAME_NAME_ENTITIES
        )
    )


def _family_a_issues(task: MemUpdateTask) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    records, actions = _event_records(task)
    gold = task.gold
    extra = _mapping(task.metadata.extra)
    stratification = _mapping(extra.get("stratification"))
    profile = _mapping(task.metadata.resolved_profile)
    difficulty = _enum_value(task.difficulty)
    expected_counts = _FAMILY_A_COUNTS.get(difficulty)
    release_id = _mapping(task.source.provenance).get("release_id")
    if type(release_id) is not str or str.__eq__(
        release_id,
        _PILOT_RELEASE_ID,
    ) is not True:
        issues.append(
            _issue(
                "family_a_release_provenance_mismatch",
                "Family A tasks require canonical Pilot release provenance",
                "source.provenance.release_id",
            )
        )

    queries = _items(task.queries)
    query = queries[0] if len(queries) == 1 else None
    query_targets = _items(getattr(query, "target_object_keys", None))
    target_identities = [
        identity
        for key in query_targets
        if (identity := _identity(key)) is not None
    ]
    target_identity = target_identities[0] if len(target_identities) == 1 else None
    query_id = getattr(query, "query_id", None)
    if (
        query is None
        or target_identity is None
        or _enum_value(getattr(query, "query_type", None))
        != QueryType.CURRENT_STATE.value
        or _items(getattr(query, "reference_candidates", None))
        or _items(getattr(query, "surface_references", None))
    ):
        issues.append(
            _issue(
                "family_a_query_semantics_mismatch",
                "Family A requires one current-state query on one exact four-part identity",
                "queries",
            )
        )

    depth = profile.get("update_depth")
    depth_valid = type(depth) is int and depth in _FAMILY_A_DEPTHS
    if not depth_valid:
        issues.append(
            _issue(
                "family_a_update_depth_mismatch",
                "Family A update_depth must be the exact integer 1, 4, or 16",
                "metadata.resolved_profile.update_depth",
            )
        )
        depth = 0

    same_name_count, other_attribute_count, noop_count = (
        expected_counts if expected_counts is not None else (0, 0, 0)
    )
    expected_event_count = (
        depth + 1 + same_name_count + other_attribute_count + noop_count
    )
    if expected_counts is None or len(records) != expected_event_count:
        issues.append(
            _issue(
                "family_a_event_count_mismatch",
                "Family A event count must equal its target chain and difficulty-derived distractors/NOOPs",
                "events",
            )
        )

    action_ids_in_event_order: list[Any] = []
    for record in records:
        event_action_ids = _items(record["event"].gold_action_ids)
        action_ids_in_event_order.extend(event_action_ids)
        if len(event_action_ids) != 1 or len(record["actions"]) != 1:
            issues.append(
                _issue(
                    "family_a_event_action_binding_mismatch",
                    "each Family A event must bind exactly one gold action",
                    f"events[{record['index']}].gold_action_ids",
                )
            )
    if action_ids_in_event_order != _items(gold.action_sequence):
        issues.append(
            _issue(
                "family_a_event_action_order_mismatch",
                "Family A action_sequence must follow canonical event order",
                "gold.action_sequence",
            )
        )

    duplicate_current = any(
        _enum_value(record["event"].role) == EventRole.DUPLICATE_CURRENT.value
        or record["trap_type"] == "duplicate_current"
        or record["metadata"].get("lifecycle") == "duplicate_current"
        or record["metadata"].get("allow_accepted_answer_ambiguity") is True
        for record in records
    )
    if duplicate_current:
        issues.append(
            _issue(
                "family_a_duplicate_current_forbidden",
                "Family A cannot contain duplicate_current observations or ambiguity metadata",
                "events",
            )
        )

    target_records = records[: depth + 1] if depth_valid else []
    target_values: list[Any] = []
    target_chain_valid = (
        depth_valid
        and target_identity is not None
        and len(target_records) == depth + 1
    )
    for position, record in enumerate(target_records):
        action = record["actions"][0] if len(record["actions"]) == 1 else None
        operation = _enum_value(getattr(action, "operation", None))
        targets = _action_targets(action) if action is not None else []
        value = getattr(action, "value", None)
        target_values.append(value)
        is_final = position == depth
        expected_operation = Operation.ADD.value if position == 0 else Operation.UPDATE.value
        expected_role = (
            EventRole.LATEST_GOLD.value
            if is_final
            else EventRole.STALE_SAME_SLOT.value
        )
        expected_version = "latest" if is_final else "stale"
        metadata = record["metadata"]
        if (
            operation != expected_operation
            or targets != ([target_identity] if target_identity is not None else [])
            or value is None
            or _enum_value(record["event"].role) != expected_role
            or not _strict_int_equal(metadata.get("version_index"), position)
            or metadata.get("version_metadata") != expected_version
        ):
            target_chain_valid = False
    if not target_chain_valid:
        issues.append(
            _issue(
                "family_a_target_chain_corruption",
                "Family A target chain must be one ADD followed by exactly update_depth UPDATEs on one identity with canonical roles",
                "events",
            )
        )

    final_value = target_values[-1] if target_values else None
    stale_values = target_values[:-1]
    canonical_stale_values = [
        _canonical_json_value(value) for value in stale_values
    ]
    canonical_final_value = (
        _canonical_json_value(final_value) if final_value is not None else None
    )
    stale_equals_final = (
        canonical_final_value is not None
        and canonical_final_value in canonical_stale_values
    )
    duplicate_stale = len(set(canonical_stale_values)) != len(
        canonical_stale_values
    )
    if stale_equals_final:
        issues.append(
            _issue(
                "family_a_stale_value_equals_current_gold",
                "every stale same-slot value must be unequal to the current gold",
                "events",
            )
        )
    if duplicate_stale:
        issues.append(
            _issue(
                "family_a_duplicate_stale_value",
                "every stale same-slot value must be semantically distinct",
                "events",
            )
        )

    same_name_start = depth + 1
    other_attribute_start = same_name_start + same_name_count
    noop_start = other_attribute_start + other_attribute_count
    distractor_records = records[same_name_start:noop_start]
    distractor_values: list[Any] = []
    distractor_identities: list[tuple[str, str, str, str | None]] = []
    distractor_valid = len(distractor_records) == same_name_count + other_attribute_count
    for offset, record in enumerate(distractor_records):
        action = record["actions"][0] if len(record["actions"]) == 1 else None
        targets = _action_targets(action) if action is not None else []
        identity = targets[0] if len(targets) == 1 else None
        value = getattr(action, "value", None)
        same_name_slot = offset < same_name_count
        expected_role = (
            EventRole.SAME_NAME_OTHER_ENTITY.value
            if same_name_slot
            else EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
        )
        geometry_valid = False
        if target_identity is not None and identity is not None:
            if same_name_slot:
                geometry_valid = _same_name_other_entity(target_identity, identity)
            else:
                geometry_valid = (
                    identity != target_identity
                    and identity[0] == target_identity[0]
                    and identity[1] == target_identity[1]
                    and identity[2] != target_identity[2]
                    and identity[3] == target_identity[3]
                )
        if (
            action is None
            or _enum_value(getattr(action, "operation", None)) != Operation.ADD.value
            or _enum_value(record["event"].role) != expected_role
            or not geometry_valid
            or value is None
        ):
            distractor_valid = False
        if identity is not None:
            distractor_identities.append(identity)
        distractor_values.append(value)
        if final_value is not None and _same_value(value, final_value):
            issues.append(
                _issue(
                    "family_a_distractor_current_gold_collision",
                    "a Family A distractor independently establishes the current target gold",
                    f"events[{record['index']}].gold_action_ids",
                )
            )
    if (
        not distractor_valid
        or len(set(distractor_identities)) != len(distractor_identities)
        or (target_identity is not None and target_identity in distractor_identities)
    ):
        issues.append(
            _issue(
                "family_a_distractor_geometry_corruption",
                "Family A distractors must be distinct ADD-only objects with their declared role geometry",
                "events",
            )
        )

    surface_variant = extra.get("surface_variant")
    noop_records = records[noop_start:expected_event_count]
    noop_valid = len(noop_records) == noop_count
    for offset, record in enumerate(noop_records):
        action = record["actions"][0] if len(record["actions"]) == 1 else None
        statement = (
            f"Near miss {offset + 1}: the record mentions "
            f"{target_identity[1]} without changing memory."
            if target_identity is not None
            else None
        )
        if (
            action is None
            or _enum_value(getattr(action, "operation", None)) != Operation.NOOP.value
            or _action_targets(action)
            or getattr(action, "value", None) is not None
            or bool(_mapping(getattr(action, "expected_effect", None)))
            or _enum_value(record["event"].role) != EventRole.NOOP_NEAR_MISS.value
            or record["metadata"].get("surface_statement") != statement
            or not _strict_int_equal(record["metadata"].get("near_miss_index"), offset)
            or not _raw_text_matches_noop_template(
                statement,
                record["event"].raw_text,
                surface_variant,
            )
            or "lifecycle" in record["metadata"]
            or "trap_type" in record["metadata"]
        ):
            noop_valid = False
    if not noop_valid:
        issues.append(
            _issue(
                "family_a_noop_semantics_mismatch",
                "Family A near misses are canonical visible NOOPs and cannot be rewritten as writes",
                "events",
            )
        )

    expected_identities = (
        ([target_identity] if target_identity is not None else [])
        + distractor_identities
    )
    declared_identities = [
        identity
        for key in _items(task.target_objects)
        if (identity := _identity(key)) is not None
    ]
    expected_present = [
        identity
        for key in _items(gold.expected_present_objects)
        if (identity := _identity(key)) is not None
    ]
    if (
        len(declared_identities) != len(expected_identities)
        or set(declared_identities) != set(expected_identities)
        or len(expected_present) != len(expected_identities)
        or set(expected_present) != set(expected_identities)
        or bool(_items(gold.expected_absent_objects))
    ):
        issues.append(
            _issue(
                "family_a_target_identity_mismatch",
                "declared and expected-present objects must equal the canonical active identities",
                "target_objects",
            )
        )

    expected_final_state: dict[str, Any] = {}
    expected_history: dict[str, list[Any]] = {}
    if target_identity is not None and target_values:
        target_id = _identity_id(target_identity)
        expected_final_state[target_id] = final_value
        expected_history[target_id] = target_values
    for identity, value in zip(distractor_identities, distractor_values):
        object_id = _identity_id(identity)
        expected_final_state[object_id] = value
        expected_history[object_id] = [value]
    final_state = _mapping(gold.final_state)
    history = _mapping(gold.version_history)
    state_history_valid = _same_value(final_state, expected_final_state) and _same_value(
        history, expected_history
    )
    target_id = _identity_id(target_identity) if target_identity is not None else None
    target_history = _items(history.get(target_id)) if target_id is not None else []
    if final_value is not None and any(
        _same_value(value, final_value) for value in target_history[:-1]
    ):
        issues.append(
            _issue(
                "family_a_stale_value_equals_current_gold",
                "stale target history cannot retain stale labeling while equaling current gold",
                "gold.version_history",
            )
        )
    for object_id, value in final_state.items():
        if object_id != target_id and final_value is not None and _same_value(value, final_value):
            issues.append(
                _issue(
                    "family_a_distractor_current_gold_collision",
                    "non-target final state cannot equal the current target gold",
                    f"gold.final_state.{object_id}",
                )
            )
    for object_id, values in history.items():
        if object_id == target_id:
            continue
        for value_index, value in enumerate(_items(values)):
            if final_value is not None and _same_value(value, final_value):
                issues.append(
                    _issue(
                        "family_a_distractor_current_gold_collision",
                        "non-target history cannot equal the current target gold",
                        f"gold.version_history.{object_id}[{value_index}]",
                    )
                )
    if not state_history_valid:
        issues.append(
            _issue(
                "family_a_gold_state_history_mismatch",
                "Family A final state and version history must exactly match canonical writes",
                "gold",
            )
        )

    issues.extend(_current_answer_issues(task, final_value, family="a"))
    expected_gold_source = (
        [target_records[-1]["event"].event_id] if target_records else []
    )
    if _items(gold.gold_source_event_ids) != expected_gold_source:
        issues.append(
            _issue(
                "family_a_query_semantics_mismatch",
                "gold_source_event_ids must identify only the final target update",
                "gold.gold_source_event_ids",
            )
        )

    actual_target_updates = sum(
        _enum_value(getattr(action, "operation", None)) == Operation.UPDATE.value
        and target_identity is not None
        and _action_targets(action) == [target_identity]
        for action in actions
    )
    actual_noops = sum(
        _enum_value(getattr(action, "operation", None)) == Operation.NOOP.value
        for action in actions
    )
    actual_stale = max(0, len(target_values) - 1)
    active_count = len(set(expected_identities))
    density = actual_noops / len(records) if records else None
    entity_ambiguity, attribute_ambiguity = _FAMILY_A_AMBIGUITY.get(
        difficulty, (None, None)
    )
    expected_version_metadata = _FAMILY_A_VERSION_METADATA.get(difficulty)
    expected_context_order, expected_interleaving, expected_naturalness = (
        _FAMILY_A_BASE_PROFILE.get(difficulty, (None, None, None))
    )
    core_index = extra.get("core_index")
    expected_difficulty_allocation = 42 if difficulty == "easy" else 39
    expected_cell_allocation = 14 if difficulty == "easy" else 13
    integer_values = (
        stratification.get("num_events"),
        stratification.get("num_target_updates"),
        stratification.get("same_name_distractor_count"),
        stratification.get("same_entity_other_attribute_count"),
        stratification.get("noop_count"),
        stratification.get("stale_same_slot_count"),
        stratification.get("stale_count"),
        stratification.get("axis_product_index"),
        stratification.get("axis_product_size"),
        stratification.get("depth_allocation_count"),
        stratification.get("difficulty_allocation_count"),
        stratification.get("depth_difficulty_cell_count"),
        profile.get("context_length"),
        profile.get("active_object_count"),
        profile.get("stale_count"),
        extra.get("core_index"),
    )
    counters_valid = all(type(value) is int for value in integer_values)
    counters_valid = counters_valid and all(
        (
            _strict_int_equal(stratification.get("num_events"), len(records)),
            _strict_int_equal(stratification.get("num_target_updates"), actual_target_updates),
            _strict_int_equal(stratification.get("same_name_distractor_count"), len(records[same_name_start:other_attribute_start])),
            _strict_int_equal(stratification.get("same_entity_other_attribute_count"), len(records[other_attribute_start:noop_start])),
            _strict_int_equal(stratification.get("noop_count"), actual_noops),
            _strict_int_equal(stratification.get("stale_same_slot_count"), actual_stale),
            _strict_int_equal(stratification.get("stale_count"), actual_stale),
            _strict_int_equal(profile.get("context_length"), len(records)),
            _strict_int_equal(profile.get("active_object_count"), active_count),
            _strict_int_equal(profile.get("stale_count"), actual_stale),
            type(density) is float
            and type(profile.get("noop_density")) is float
            and profile.get("noop_density") == density,
            type(core_index) is int
            and 0 <= core_index < 120
            and _strict_int_equal(stratification.get("axis_product_index"), core_index),
            _strict_int_equal(stratification.get("axis_product_size"), 384),
            _strict_int_equal(stratification.get("depth_allocation_count"), 40),
            _strict_int_equal(stratification.get("difficulty_allocation_count"), expected_difficulty_allocation),
            _strict_int_equal(stratification.get("depth_difficulty_cell_count"), expected_cell_allocation),
        )
    )
    profile_valid = (
        profile.get("difficulty") == difficulty
        and profile.get("profile_name") == difficulty
        and profile.get("task_family") == TaskFamily.REPEATED_SAME_SLOT.value
        and profile.get("update_depth_bucket") == _FAMILY_A_DEPTH_BUCKETS.get(depth)
        and profile.get("entity_ambiguity") == entity_ambiguity
        and profile.get("attribute_ambiguity") == attribute_ambiguity
        and profile.get("query_type") == QueryType.CURRENT_STATE.value
        and profile.get("version_metadata") == expected_version_metadata
        and profile.get("context_order") == expected_context_order
        and type(profile.get("cross_slot_interleaving")) is float
        and profile.get("cross_slot_interleaving") == expected_interleaving
        and profile.get("source_naturalness") == expected_naturalness
        and type(surface_variant) is int
        and 0 <= surface_variant < len(SURFACE_TEMPLATE_SETS)
        and extra.get("surface_template")
        == SURFACE_TEMPLATE_SETS[surface_variant][0]
    )
    if not counters_valid or not profile_valid:
        issues.append(
            _issue(
                "family_a_counter_profile_mismatch",
                "Family A counters, allocation fields, density, ambiguity, and profile metadata must match observed canonical semantics",
                "metadata",
            )
        )

    return issues


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

    semantic_statement = None
    duplicate_statement = None
    independent_count = 0
    independent_statements: set[str] = set()
    if target_identity is not None:
        semantic_statement = family_d_semantic_near_miss_statement(
            target_identity[1],
            target_identity[2],
        )
        if isinstance(target_current_value, str):
            duplicate_statement = family_d_duplicate_current_statement(
                target_identity[1],
                target_identity[2],
                target_current_value,
            )
        independent_count = max(
            0,
            (expected_noop_count or 0) - (1 if declared_trap in _NOOP_TRAPS else 0),
        )
        independent_statements = {
            family_d_independent_noop_statement(target_identity[1], note_number)
            for note_number in range(1, independent_count + 1)
        }
    canonical_noop_statements = independent_statements | {
        statement
        for statement in (semantic_statement, duplicate_statement)
        if statement is not None
    }
    surface_variant = extra.get("surface_variant")
    independent_noop_records = [
        record for record in records if record["lifecycle"] == "independent_noop"
    ]
    observed_independent_statements = [
        record["metadata"].get("surface_statement")
        for record in independent_noop_records
    ]
    if (
        len(independent_noop_records) != independent_count
        or len(set(observed_independent_statements))
        != len(observed_independent_statements)
        or set(observed_independent_statements) != independent_statements
    ):
        issues.append(
            _issue(
                "family_d_noop_visibility_mismatch",
                "independent NOOP observations must bind one-to-one to canonical note numbers",
                "events",
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
        statement = record["metadata"].get("surface_statement")
        statement_noop = statement in canonical_noop_statements
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
        expected_trap_statement = None
        trap_kind = record["trap_type"]
        if designated_noop:
            trap_kind = declared_trap
        if trap_kind == "semantic_near_miss":
            expected_trap_statement = semantic_statement
        elif trap_kind == "duplicate_current":
            expected_trap_statement = duplicate_statement
        if (trap_noop or role_noop or designated_noop) and not (
            trap_noop
            and role_noop
            and lifecycle == "trap_noop"
            and statement == expected_trap_statement
        ):
            issues.append(
                _issue(
                    "family_d_noop_semantics_mismatch",
                    "NOOP trap type, role, lifecycle, and canonical statement disagree",
                    f"events[{record['index']}]",
                )
            )
        canonical_statement = expected_trap_statement
        if canonical_statement is None and lifecycle == "independent_noop":
            canonical_statement = statement if statement in independent_statements else None
        has_noop_action = any(
            _enum_value(getattr(action, "operation", None)) == Operation.NOOP.value
            for action in event_actions
        )
        if (expected_noop or has_noop_action) and (
            canonical_statement is None
            or statement != canonical_statement
            or not _raw_text_matches_noop_template(
                canonical_statement,
                getattr(record["event"], "raw_text", None),
                surface_variant,
            )
        ):
            issues.append(
                _issue(
                    "family_d_noop_visibility_mismatch",
                    "NOOP statement and visible text must match canonical Family D rendering",
                    f"events[{record['index']}].raw_text",
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
        if (
            record["metadata"].get("surface_statement") != duplicate_statement
            or not _raw_text_matches_noop_template(
                duplicate_statement,
                getattr(event, "raw_text", None),
                surface_variant,
            )
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
    target_update_count = sum(
        _enum_value(getattr(action, "operation", None)) == Operation.UPDATE.value
        and target_identity is not None
        and target_identity in _action_targets(action)
        for action in semantic_actions
    )
    integer_metadata = (
        stratification.get("num_events"),
        stratification.get("noop_count"),
        stratification.get("true_write_count"),
        stratification.get("num_target_updates"),
        stratification.get("duplicate_current_count"),
        stratification.get("trap_position"),
        profile.get("context_length"),
    )
    if any(type(value) is not int for value in integer_metadata):
        issues.append(
            _issue(
                "family_d_integer_metadata_type_mismatch",
                "Family D counters, trap position, and context length must be exact integers",
                "metadata.extra.stratification",
            )
        )
    counter_checks = (
        _strict_int_equal(stratification.get("num_events"), len(records)),
        _strict_int_equal(stratification.get("noop_count"), action_noops),
        _strict_int_equal(
            stratification.get("true_write_count"),
            len(semantic_actions) - action_noops,
        ),
        _strict_int_equal(
            stratification.get("num_target_updates"),
            target_update_count,
        ),
        _strict_int_equal(
            stratification.get("duplicate_current_count"),
            len(duplicate_records),
        ),
        type(stratification.get("trap_position")) is int,
        stratification.get("operation_signature") == expected_signature,
        _strict_int_equal(profile.get("context_length"), len(records)),
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

    issues.extend(
        _current_answer_issues(task, target_current_value, family="d")
    )
    return issues


def _generic_validation_issues(
    task: MemUpdateTask,
    *,
    family: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for validator in (validate_task, validate_gold_replay, validate_distractors):
        try:
            issues.extend(validator(task).issues)
        except Exception:
            issues.append(
                _issue(
                    f"family_{family}_malformed_task",
                    "existing validator rejected malformed task structure",
                    "task",
                )
            )
    return issues


def _validate_family_a_task(task: Any) -> ValidationReport:
    if type(task) is not MemUpdateTask:
        return _bounded_report(
            [
                _issue(
                    "family_a_invalid_task_type",
                    "Family A validation requires an exact MemUpdateTask instance",
                    "task",
                )
            ],
            family="a",
        )
    try:
        raw = object.__getattribute__(task, "__dict__")
    except (AttributeError, TypeError):
        raw = None
    task_family = raw.get("task_family") if type(raw) is dict else None
    identifies_family_a = isinstance(task_family, str) and str.__eq__(
        task_family,
        TaskFamily.REPEATED_SAME_SLOT.value,
    ) is True
    if not identifies_family_a and (
        type(task_family) is not str or not str.strip(task_family)
    ):
        return _bounded_report(
            [
                _issue(
                    "family_a_malformed_task",
                    "MemUpdateTask.task_family must be a nonblank exact string",
                    "task_family",
                )
            ],
            family="a",
        )
    if not identifies_family_a:
        return _bounded_report(
            [
                _issue(
                    "family_a_inapplicable_task_family",
                    "Family A validation is inapplicable to this task family",
                    "task_family",
                )
            ],
            family="a",
        )

    schema_issues = _schema_preflight_issues(task, family="a")
    if schema_issues:
        return _bounded_report(schema_issues, family="a")

    issues = _contract_constraint_issues(task, family="a")
    issues.extend(_generic_validation_issues(task, family="a"))
    issues.extend(
        _family_a_preflight_issues(task, schema_checked=True)
    )
    try:
        issues.extend(_family_a_issues(task))
    except Exception:
        issues.append(
            _issue(
                "family_a_malformed_task",
                "Family A semantic inspection rejected malformed task structure",
                "task",
            )
        )
    return _bounded_report(issues, family="a")


def validate_family_a_task(task: Any) -> ValidationReport:
    """Validate one Family A task without mutation, external I/O, or exception leaks."""
    try:
        return _validate_family_a_task(task)
    except Exception:
        return _bounded_report(
            [
                _issue(
                    "family_a_malformed_task",
                    "Family A validation rejected malformed task structure",
                    "task",
                )
            ],
            family="a",
        )


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
    try:
        raw = object.__getattribute__(task, "__dict__")
    except (AttributeError, TypeError):
        raw = None
    task_family = raw.get("task_family") if type(raw) is dict else None
    identifies_family_d = isinstance(task_family, str) and str.__eq__(
        task_family,
        TaskFamily.NOOP_WRITE_DISCIPLINE.value,
    ) is True
    if not identifies_family_d and (
        type(task_family) is not str or not str.strip(task_family)
    ):
        return _bounded_report(
            [
                _issue(
                    "family_d_malformed_task",
                    "MemUpdateTask.task_family must be a nonblank exact string",
                    "task_family",
                )
            ]
        )
    if not identifies_family_d:
        return _bounded_report(
            [
                _issue(
                    "family_d_inapplicable_task_family",
                    "Family D validation is inapplicable to this task family",
                    "task_family",
                )
            ]
        )

    preflight_issues = _preflight_issues(task)
    if preflight_issues:
        return _bounded_report(preflight_issues)

    issues = _contract_constraint_issues(task, family="d")
    issues.extend(_generic_validation_issues(task, family="d"))
    try:
        issues.extend(_family_d_issues(task))
    except Exception:
        issues.append(
            _issue(
                "family_d_malformed_task",
                "Family D semantic inspection rejected malformed task structure",
                "task",
            )
        )
    return _bounded_report(issues)


def validate_family_d_task(task: Any) -> ValidationReport:
    """Validate one Family D task without mutation, external I/O, or exception leaks."""
    try:
        return _validate_family_d_task(task)
    except Exception:
        return _bounded_report(
            [
                _issue(
                    "family_d_malformed_task",
                    "Family D validation rejected malformed task structure",
                    "task",
                )
            ]
        )


__all__ = ["validate_family_a_task", "validate_family_d_task"]
