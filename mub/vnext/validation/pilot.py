from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from string import Template
from types import SimpleNamespace, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue, ValidationError

from mub.vnext.contracts import (
    AnswerDisposition,
    EventRole,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
    TaskFamily,
)
from mub.vnext.contracts.task import CanonicalAnswer, MemUpdateTask
import mub.vnext.generation.family_c as family_c_generation
import mub.vnext.generation.render as render_generation
from mub.vnext.generation.catalogs import (
    ALIAS_MAPPINGS,
    CANONICAL_ATTRIBUTES,
    NAMESPACES,
    REFERENCE_QUERY_TEMPLATE_SETS,
    RELATION_QUALIFIED_ENTITIES,
    SAME_NAME_ENTITIES,
    SURFACE_TEMPLATE_SETS,
)
from mub.vnext.generation.core import CoreEvent
from mub.vnext.generation.family_b_schedule import (
    INTERLEAVING_PATTERNS,
    canonical_cross_slot_update_count,
    canonical_interleaving_schedule,
)
from mub.vnext.generation.family_d import (
    family_d_duplicate_current_statement,
    family_d_independent_noop_statement,
    family_d_semantic_near_miss_statement,
)
from mub.vnext.generation.identity import (
    action_id as canonical_action_id,
    event_id as canonical_event_id,
    query_id as canonical_query_id,
    source_id as canonical_source_id,
    task_id as canonical_task_id,
)
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
    merge_reports,
)
from mub.vnext.validation.replay import (
    replay_actions,
    validate_distractors,
    validate_gold_replay,
)
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
_FAMILY_B_ACTIVE_COUNTS = {"easy": 2, "medium": 4, "hard": 8}
_FAMILY_B_DENSITIES = {"easy": 0.0, "medium": 0.25, "hard": 0.5}


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


def _raw_task_family(task: Any) -> Any:
    if type(task) is not MemUpdateTask:
        return None
    try:
        raw = object.__getattribute__(task, "__dict__")
    except (AttributeError, TypeError):
        return None
    return raw.get("task_family") if type(raw) is dict else None


def _task_identifies_family(task: Any, expected_family: str) -> bool:
    task_family = _raw_task_family(task)
    return isinstance(task_family, str) and str.__eq__(
        task_family,
        expected_family,
    ) is True


def _strict_family_precheck(
    task: Any,
    *,
    family: str,
    expected_family: str,
) -> ValidationReport | None:
    label = family.upper()
    if type(task) is not MemUpdateTask:
        return _bounded_report(
            [
                _issue(
                    f"family_{family}_invalid_task_type",
                    f"Family {label} validation requires an exact MemUpdateTask instance",
                    "task",
                )
            ],
            family=family,
        )
    task_family = _raw_task_family(task)
    identifies_family = _task_identifies_family(task, expected_family)
    if not identifies_family and (
        type(task_family) is not str or not str.strip(task_family)
    ):
        return _bounded_report(
            [
                _issue(
                    f"family_{family}_malformed_task",
                    "MemUpdateTask.task_family must be a nonblank exact string",
                    "task_family",
                )
            ],
            family=family,
        )
    if not identifies_family:
        return _bounded_report(
            [
                _issue(
                    f"family_{family}_inapplicable_task_family",
                    f"Family {label} validation is inapplicable to this task family",
                    "task_family",
                )
            ],
            family=family,
        )
    return None


def _run_strict_family_validator(
    task: Any,
    validator: Any,
    *,
    family: str,
) -> ValidationReport:
    try:
        return validator(task)
    except Exception:
        return _bounded_report(
            [
                _issue(
                    f"family_{family}_malformed_task",
                    f"Family {family.upper()} validation rejected malformed task structure",
                    "task",
                )
            ],
            family=family,
        )


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


def _family_b_preflight_issues(
    task: MemUpdateTask,
    *,
    schema_checked: bool = False,
) -> list[ValidationIssue]:
    if not schema_checked:
        schema_issues = _schema_preflight_issues(task, family="b")
        if schema_issues:
            return schema_issues

    issues: list[ValidationIssue] = []
    bounded_collections = (
        ("events", task.events, 64),
        ("queries", task.queries, 1),
        ("target_objects", task.target_objects, 8),
        ("gold.actions", task.gold.actions, 64),
        ("gold.action_sequence", task.gold.action_sequence, 64),
        ("gold.gold_source_event_ids", task.gold.gold_source_event_ids, 1),
        ("gold.expected_present_objects", task.gold.expected_present_objects, 8),
    )
    for path, values, limit in bounded_collections:
        if len(values) > limit:
            issues.append(
                _issue(
                    "family_b_input_size_limit",
                    f"{path} exceeds the Family B inspection limit",
                    path,
                )
            )
    for index, event in enumerate(task.events):
        if len(event.gold_action_ids) > 1:
            issues.append(
                _issue(
                    "family_b_input_size_limit",
                    "event gold_action_ids exceeds the Family B cardinality limit",
                    f"events[{index}].gold_action_ids",
                )
            )
    for index, action in enumerate(task.gold.actions):
        if len(action.target_object_keys) > 1:
            issues.append(
                _issue(
                    "family_b_input_size_limit",
                    "action target_object_keys exceeds the Family B cardinality limit",
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
                        "family_b_input_size_limit",
                        f"query {field_name} exceeds the Family B cardinality limit",
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


def _family_b_issues(task: MemUpdateTask) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    records, _ = _event_records(task)
    gold = task.gold
    extra = _mapping(task.metadata.extra)
    raw_stratification = extra.get("stratification")
    stratification = raw_stratification if type(raw_stratification) is dict else {}
    profile = _mapping(task.metadata.resolved_profile)
    difficulty = _enum_value(task.difficulty)
    expected_active_count = _FAMILY_B_ACTIVE_COUNTS.get(difficulty)
    expected_density = _FAMILY_B_DENSITIES.get(difficulty)

    queries = _items(task.queries)
    query = queries[0] if len(queries) == 1 else None
    query_targets = _items(getattr(query, "target_object_keys", None))
    target_identities = [
        identity
        for key in query_targets
        if (identity := _identity(key)) is not None
    ]
    target_identity = target_identities[0] if len(target_identities) == 1 else None
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
                "family_b_query_semantics_mismatch",
                "Family B requires one current-state query on one exact four-part identity",
                "queries",
            )
        )

    depth = profile.get("update_depth")
    depth_valid = type(depth) is int and depth in _FAMILY_A_DEPTHS
    if not depth_valid:
        issues.append(
            _issue(
                "family_b_update_depth_mismatch",
                "Family B update_depth must be the exact integer 1, 4, or 16",
                "metadata.resolved_profile.update_depth",
            )
        )
        depth = 0

    action_ids_in_event_order: list[Any] = []
    has_noop = False
    slot_records: dict[int, list[dict[str, Any]]] = {}
    all_event_actions_valid = True
    for record in records:
        event = record["event"]
        event_action_ids = _items(event.gold_action_ids)
        action_ids_in_event_order.extend(event_action_ids)
        if len(event_action_ids) != 1 or len(record["actions"]) != 1:
            all_event_actions_valid = False
            continue
        action = record["actions"][0]
        operation = _enum_value(getattr(action, "operation", None))
        if operation == Operation.NOOP.value or _enum_value(event.role) == EventRole.NOOP_NEAR_MISS.value:
            has_noop = True
        metadata = record["metadata"]
        slot_index = metadata.get("slot_index")
        if type(slot_index) is int and 0 <= slot_index <= 7:
            slot_records.setdefault(slot_index, []).append(record)
        else:
            all_event_actions_valid = False
        if (
            operation not in {Operation.ADD.value, Operation.UPDATE.value}
            or len(_action_targets(action)) != 1
            or getattr(action, "value", None) is None
        ):
            all_event_actions_valid = False
    if not all_event_actions_valid:
        issues.append(
            _issue(
                "family_b_event_action_binding_mismatch",
                "each Family B event must bind one single-target ADD or UPDATE action",
                "events",
            )
        )
    if action_ids_in_event_order != _items(gold.action_sequence):
        issues.append(
            _issue(
                "family_b_event_action_order_mismatch",
                "Family B action_sequence must follow emitted event order",
                "gold.action_sequence",
            )
        )
    if has_noop:
        issues.append(
            _issue(
                "family_b_noop_forbidden",
                "Family B cannot contain NOOP events or actions",
                "events",
            )
        )

    slot_identities: dict[int, tuple[str, str, str, str | None]] = {}
    slot_values: dict[int, list[Any]] = {}
    target_chain_valid = depth_valid and target_identity is not None
    non_target_valid = expected_active_count is not None
    expected_slots = set(range(expected_active_count or 0))
    if set(slot_records) != expected_slots:
        target_chain_valid = False
        non_target_valid = False

    for slot_index in sorted(slot_records):
        trajectory = slot_records[slot_index]
        versions = [record["metadata"].get("version_index") for record in trajectory]
        if versions != list(range(len(trajectory))):
            if slot_index == 0:
                target_chain_valid = False
            else:
                non_target_valid = False
        identities: list[tuple[str, str, str, str | None]] = []
        values: list[Any] = []
        for version_index, record in enumerate(trajectory):
            action = record["actions"][0] if len(record["actions"]) == 1 else None
            targets = _action_targets(action) if action is not None else []
            identity = targets[0] if len(targets) == 1 else None
            if identity is not None:
                identities.append(identity)
            value = getattr(action, "value", None)
            values.append(value)
            operation = _enum_value(getattr(action, "operation", None))
            expected_operation = Operation.ADD.value if version_index == 0 else Operation.UPDATE.value
            is_latest = version_index == len(trajectory) - 1
            expected_version_metadata = "latest" if is_latest else "stale"
            metadata = record["metadata"]
            if slot_index == 0:
                expected_role = EventRole.LATEST_GOLD.value if is_latest else EventRole.STALE_SAME_SLOT.value
                record_valid = (
                    operation == expected_operation
                    and identity == target_identity
                    and value is not None
                    and not bool(_mapping(getattr(action, "expected_effect", None)))
                    and _enum_value(record["event"].role) == expected_role
                    and metadata.get("target_relation") == "target"
                    and metadata.get("version_metadata") == expected_version_metadata
                    and "distractor_kind" not in metadata
                )
                target_chain_valid = target_chain_valid and record_valid
            else:
                expected_kind = "active_non_target" if version_index == 0 else "cross_slot"
                record_valid = (
                    operation == expected_operation
                    and value is not None
                    and not bool(_mapping(getattr(action, "expected_effect", None)))
                    and _enum_value(record["event"].role)
                    == EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
                    and metadata.get("target_relation") == "same_entity_other_attribute"
                    and metadata.get("distractor_kind") == expected_kind
                    and metadata.get("version_metadata") == expected_version_metadata
                )
                non_target_valid = non_target_valid and record_valid
        if identities:
            slot_identities[slot_index] = identities[0]
            if any(identity != identities[0] for identity in identities):
                if slot_index == 0:
                    target_chain_valid = False
                else:
                    non_target_valid = False
        else:
            if slot_index == 0:
                target_chain_valid = False
            else:
                non_target_valid = False
        slot_values[slot_index] = values

    target_values = slot_values.get(0, [])
    if len(target_values) != depth + 1:
        target_chain_valid = False
    if not target_chain_valid:
        issues.append(
            _issue(
                "family_b_target_chain_corruption",
                "Family B target trajectory must be one ADD followed by exactly update_depth ordered UPDATEs on slot zero",
                "events",
            )
        )
    final_target_value = target_values[-1] if target_values else None
    if final_target_value is None or any(
        _same_value(value, final_target_value) for value in target_values[:-1]
    ):
        issues.append(
            _issue(
                "family_b_target_value_corruption",
                "Family B pre-final target values must be stale and unequal to the final target value",
                "events",
            )
        )

    active_identities = [
        slot_identities[index]
        for index in sorted(slot_identities)
        if index in slot_identities
    ]
    geometry_valid = (
        expected_active_count is not None
        and len(active_identities) == expected_active_count
        and len(set(active_identities)) == expected_active_count
        and slot_identities.get(0) == target_identity
    )
    if geometry_valid and target_identity is not None:
        for slot_index in range(1, expected_active_count):
            identity = slot_identities.get(slot_index)
            geometry_valid = geometry_valid and (
                identity is not None
                and identity[0] == target_identity[0]
                and identity[1] == target_identity[1]
                and identity[2] != target_identity[2]
                and identity[2] in CANONICAL_ATTRIBUTES
                and identity[3] is None
            )
        geometry_valid = geometry_valid and (
            target_identity[2] in CANONICAL_ATTRIBUTES and target_identity[3] is None
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
        not geometry_valid
        or len(declared_identities) != (expected_active_count or 0)
        or len(set(declared_identities)) != len(declared_identities)
        or set(declared_identities) != set(active_identities)
        or len(expected_present) != (expected_active_count or 0)
        or len(set(expected_present)) != len(expected_present)
        or set(expected_present) != set(active_identities)
        or bool(_items(gold.expected_absent_objects))
    ):
        issues.append(
            _issue(
                "family_b_active_identity_corruption",
                "Family B active objects must be distinct reviewed same-entity attributes with contiguous slot identities",
                "target_objects",
            )
        )
    if not non_target_valid:
        issues.append(
            _issue(
                "family_b_non_target_corruption",
                "each Family B non-target slot must preserve its own canonical ADD/UPDATE trajectory and metadata",
                "events",
            )
        )

    pattern = stratification.get("interleaving_pattern")
    actual_pattern = [
        (record["metadata"].get("slot_index"), record["metadata"].get("version_index"))
        for record in records
    ]
    trajectory_lengths = tuple(
        len(slot_records.get(slot_index, []))
        for slot_index in range(expected_active_count or 0)
    )
    try:
        expected_pattern = canonical_interleaving_schedule(
            trajectory_lengths,
            pattern,
        )
    except (TypeError, ValueError):
        expected_pattern = ()
    if tuple(actual_pattern) != expected_pattern:
        issues.append(
            _issue(
                "family_b_interleaving_pattern_mismatch",
                "Family B emitted order must exactly implement its declared interleaving pattern",
                "events",
            )
        )

    ordered_actions = [
        record["actions"][0]
        for record in records
        if len(record["actions"]) == 1
    ]
    try:
        replay_payload = replay_actions(ordered_actions).model_dump(mode="json")
        computed_state = replay_payload["final_state"]
        computed_history = replay_payload["version_history"]
    except Exception:
        computed_state = {}
        computed_history = {}
    final_state = _mapping(gold.final_state)
    version_history = _mapping(gold.version_history)
    if not _same_value(final_state, computed_state) or not _same_value(
        version_history, computed_history
    ):
        issues.append(
            _issue(
                "family_b_gold_state_history_mismatch",
                "Family B final state and version history must exactly equal emitted writes",
                "gold",
            )
        )
    for slot_index in range(1, expected_active_count or 0):
        identity = slot_identities.get(slot_index)
        values = slot_values.get(slot_index, [])
        if identity is None or not values:
            continue
        object_id = _identity_id(identity)
        if (
            object_id not in final_state
            or not _same_value(final_state.get(object_id), values[-1])
            or not _same_value(version_history.get(object_id), values)
        ):
            issues.append(
                _issue(
                    "family_b_non_target_state_history_corruption",
                    "a Family B non-target final state or history does not match its own trajectory",
                    f"gold.final_state.{object_id}",
                )
            )

    final_values = [values[-1] for values in slot_values.values() if values]
    canonical_final_values = [_canonical_json_value(value) for value in final_values]
    if len(canonical_final_values) != (expected_active_count or 0) or len(
        set(canonical_final_values)
    ) != len(canonical_final_values):
        issues.append(
            _issue(
                "family_b_final_value_collision",
                "Family B active objects must retain distinct final values",
                "gold.final_state",
            )
        )
    if final_target_value is not None:
        for slot_index in range(1, expected_active_count or 0):
            values = slot_values.get(slot_index, [])
            if values and _same_value(values[-1], final_target_value):
                issues.append(
                    _issue(
                        "family_b_non_target_current_gold_collision",
                        "a Family B non-target current value cannot equal the target gold",
                        f"events.slot[{slot_index}]",
                    )
                )

    issues.extend(_current_answer_issues(task, final_target_value, family="b"))
    target_id = _identity_id(target_identity) if target_identity is not None else None
    if (
        target_id is None
        or target_id not in final_state
        or not _same_value(final_state.get(target_id), final_target_value)
        or not _same_value(version_history.get(target_id), target_values)
    ):
        issues.append(
            _issue(
                "family_b_target_value_corruption",
                "Family B target answer, state, and history must equal the final target trajectory",
                "gold",
            )
        )
    target_records = slot_records.get(0, [])
    expected_source = [target_records[-1]["event"].event_id] if target_records else []
    if _items(gold.gold_source_event_ids) != expected_source:
        issues.append(
            _issue(
                "family_b_query_semantics_mismatch",
                "gold_source_event_ids must identify the final target update",
                "gold.gold_source_event_ids",
            )
        )

    actual_cross_slot_updates = sum(
        max(0, len(slot_records.get(slot_index, [])) - 1)
        for slot_index in range(1, expected_active_count or 0)
    )
    base_event_count = (expected_active_count or 0) + depth
    expected_cross_slot_updates = (
        canonical_cross_slot_update_count(base_event_count, expected_density)
        if expected_density is not None and base_event_count > 0
        else -1
    )
    expected_event_count = base_event_count + expected_cross_slot_updates
    realized_density = (
        expected_cross_slot_updates / base_event_count if base_event_count else None
    )
    core_index = extra.get("core_index")
    pattern_group_index = stratification.get("pattern_group_index")
    axis_product_index = stratification.get("axis_product_index")
    axis_product_size = stratification.get("axis_product_size")
    allocation_count = stratification.get("allocation_cell_count")
    allocation_ideal = stratification.get("allocation_cell_ideal")
    allocation_deviation = stratification.get("allocation_cell_deviation")
    difficulty_count = stratification.get("difficulty_allocation_count")
    difficulty_ideal = stratification.get("difficulty_allocation_ideal")
    difficulty_deviation = stratification.get("difficulty_allocation_deviation")
    integer_values = (
        profile.get("active_object_count"),
        profile.get("context_length"),
        profile.get("stale_count"),
        stratification.get("num_events"),
        stratification.get("num_target_updates"),
        stratification.get("active_object_count"),
        stratification.get("cross_slot_distractor_count"),
        stratification.get("base_event_count"),
        stratification.get("update_depth"),
        stratification.get("stale_count"),
        stratification.get("noop_count"),
        stratification.get("axis_product_index"),
        stratification.get("axis_product_size"),
        stratification.get("pattern_group_index"),
        stratification.get("allocation_cell_count"),
        stratification.get("difficulty_allocation_count"),
        core_index,
    )
    counters_valid = all(type(value) is int for value in integer_values)
    counters_valid = counters_valid and all(
        (
            _strict_int_equal(profile.get("active_object_count"), expected_active_count or -1),
            _strict_int_equal(profile.get("context_length"), len(records)),
            _strict_int_equal(profile.get("stale_count"), depth),
            _strict_int_equal(stratification.get("num_events"), len(records)),
            _strict_int_equal(stratification.get("num_target_updates"), depth),
            _strict_int_equal(stratification.get("active_object_count"), expected_active_count or -1),
            _strict_int_equal(stratification.get("cross_slot_distractor_count"), actual_cross_slot_updates),
            actual_cross_slot_updates == expected_cross_slot_updates,
            _strict_int_equal(stratification.get("base_event_count"), base_event_count),
            _strict_int_equal(stratification.get("update_depth"), depth),
            _strict_int_equal(stratification.get("stale_count"), depth),
            _strict_int_equal(stratification.get("noop_count"), 0),
            len(records) == expected_event_count,
            type(stratification.get("cross_slot_distractor_density")) is float
            and stratification.get("cross_slot_distractor_density") == expected_density,
            type(stratification.get("realized_cross_slot_distractor_density")) is float
            and stratification.get("realized_cross_slot_distractor_density") == realized_density,
            type(core_index) is int and core_index >= 0,
            type(pattern_group_index) is int and pattern_group_index >= 0,
            type(axis_product_index) is int
            and type(axis_product_size) is int
            and 0 <= axis_product_index < axis_product_size,
            type(allocation_count) is int and allocation_count > 0,
            type(allocation_ideal) is float
            and math.isfinite(allocation_ideal)
            and allocation_ideal > 0.0,
            type(allocation_deviation) is float
            and allocation_deviation == allocation_count - allocation_ideal,
            type(difficulty_count) is int and difficulty_count > 0,
            type(difficulty_ideal) is float
            and math.isfinite(difficulty_ideal)
            and difficulty_ideal > 0.0,
            type(difficulty_deviation) is float
            and difficulty_deviation == difficulty_count - difficulty_ideal,
            "num_updates" not in stratification,
            "num_updates" not in profile,
        )
    )
    entity_ambiguity, attribute_ambiguity = _FAMILY_A_AMBIGUITY.get(
        difficulty, (None, None)
    )
    _, _, expected_naturalness = _FAMILY_A_BASE_PROFILE.get(
        difficulty, (None, None, None)
    )
    surface_variant = extra.get("surface_variant")
    profile_valid = (
        raw_stratification is stratification
        and profile.get("difficulty") == difficulty
        and profile.get("profile_name") == difficulty
        and profile.get("task_family") == TaskFamily.INTERLEAVED_MULTI_SLOT.value
        and profile.get("update_depth_bucket") == _FAMILY_A_DEPTH_BUCKETS.get(depth)
        and profile.get("entity_ambiguity") == entity_ambiguity
        and profile.get("attribute_ambiguity") == attribute_ambiguity
        and profile.get("query_type") == QueryType.CURRENT_STATE.value
        and profile.get("version_metadata") == "event_index"
        and profile.get("context_order") == "chronological"
        and type(profile.get("noop_density")) is float
        and profile.get("noop_density") == 0.0
        and type(profile.get("cross_slot_interleaving")) is float
        and profile.get("cross_slot_interleaving") == expected_density
        and profile.get("source_naturalness") == expected_naturalness
        and pattern in INTERLEAVING_PATTERNS
        and profile.get("interleaving_pattern", pattern) == pattern
        and type(surface_variant) is int
        and 0 <= surface_variant < len(SURFACE_TEMPLATE_SETS)
        and extra.get("surface_template") == SURFACE_TEMPLATE_SETS[surface_variant][0]
    )
    if not counters_valid or not profile_valid:
        issues.append(
            _issue(
                "family_b_counter_profile_mismatch",
                "Family B counts, densities, allocation fields, context, and profile metadata must match observed semantics",
                "metadata",
            )
        )
    return issues


def _family_c_surface_integrity_issues(
    task: MemUpdateTask,
    records: list[dict[str, Any]],
    query: Any,
    candidates: list[Any],
    references: list[Any],
    extra: Mapping[str, Any],
) -> list[ValidationIssue]:
    surface_variant = extra.get("surface_variant")
    if type(surface_variant) is not int or not 0 <= surface_variant < len(
        REFERENCE_QUERY_TEMPLATE_SETS
    ):
        return [
            _issue(
                "family_c_surface_integrity_mismatch",
                "Family C visible surfaces require a canonical surface variant",
                "metadata.extra.surface_variant",
            )
        ]

    event_template_name, add_template, *_ = SURFACE_TEMPLATE_SETS[surface_variant]
    reference_template_name = REFERENCE_QUERY_TEMPLATE_SETS[surface_variant][0]
    renderer_admin = {
        "surface_template": event_template_name,
        "surface_variant": surface_variant,
    }
    valid = event_template_name == reference_template_name
    semantic_core_id = extra.get("semantic_core_id")
    core_index = extra.get("core_index")
    valid = valid and type(semantic_core_id) is str and bool(semantic_core_id.strip())
    valid = valid and type(core_index) is int
    if type(semantic_core_id) is str:
        expected_task_id = canonical_task_id(semantic_core_id, surface_variant)
        valid = valid and task.task_id == expected_task_id
        valid = valid and task.metadata.split_key.semantic_core_id == semantic_core_id
    else:
        expected_task_id = None

    core_events: list[CoreEvent] = []
    expected_raw_events: list[dict[str, str]] = []
    for index, record in enumerate(records):
        event = record["event"]
        action = record["actions"][0] if len(record["actions"]) == 1 else None
        metadata = dict(record["metadata"])
        renderer = metadata.pop(render_generation._RENDERER_METADATA_KEY, None)
        valid = valid and renderer == renderer_admin
        valid = valid and metadata == {"candidate_index": index}
        valid = valid and _mapping(getattr(event, "source_anchor", None)) == {
            "event_index": index
        }
        if action is None or expected_task_id is None:
            valid = False
            continue
        if (
            type(action.operation) is not Operation
            or action.operation is not Operation.ADD
        ):
            valid = False
            continue
        expected_event_id = canonical_event_id(expected_task_id, index)
        expected_action_id = canonical_action_id(expected_task_id, index, 0)
        valid = valid and event.event_id == expected_event_id
        valid = valid and action.action_id == expected_action_id
        valid = valid and action.event_id == expected_event_id
        core_event = CoreEvent(
            operation=action.operation,
            object_keys=list(action.target_object_keys),
            value=action.value,
            role=event.role,
            metadata=metadata,
        )
        core_events.append(core_event)
        expected_raw_text = render_generation._render_event_text(
            core_event,
            {Operation.ADD: add_template},
        )
        expected_normalized_text = render_generation._normalized_event_text(core_event)
        expected_speaker = render_generation._SPEAKERS[surface_variant]
        valid = valid and event.raw_text == expected_raw_text
        valid = valid and event.normalized_text == expected_normalized_text
        valid = valid and event.speaker == expected_speaker
        expected_raw_events.append(
            {"raw_text": expected_raw_text, "speaker": expected_speaker}
        )

    if query is None or expected_task_id is None:
        valid = False
        expected_query_text = None
    else:
        valid = valid and query.query_id == canonical_query_id(expected_task_id, 0)
        valid = valid and _mapping(query.metadata) == {
            render_generation._RENDERER_METADATA_KEY: renderer_admin
        }
        surface_core = SimpleNamespace(
            events=core_events,
            reference_candidates=candidates,
            surface_references=references,
        )
        expected_query_text = render_generation._render_unresolved_query_text(
            surface_core,
            surface_variant,
        )
        valid = valid and query.text == expected_query_text

    provenance = _mapping(task.source.provenance)
    valid = valid and extra.get("surface_template") == event_template_name
    valid = valid and provenance.get("surface_variant") == surface_variant
    valid = valid and provenance.get("surface_template") == event_template_name
    valid = valid and provenance.get("semantic_core_id") == semantic_core_id
    valid = valid and task.source.normalization_version == (
        render_generation._NORMALIZATION_VERSION
    )
    if type(semantic_core_id) is str and type(core_index) is int:
        expected_source_id = canonical_source_id(
            "vnext_pilot",
            core_index,
            {
                "semantic_core_id": semantic_core_id,
                "surface_variant": surface_variant,
            },
        )
        valid = valid and task.source.source_id == expected_source_id
        valid = valid and task.source.source_uri == f"memory://{expected_source_id}"

    if len(core_events) == len(records) and expected_query_text is not None:
        surface_core = SimpleNamespace(events=core_events)
        expected_normalized_hash = render_generation._payload_sha256(
            render_generation._normalized_source_semantic_projection(surface_core)
        )
        expected_raw_hash = render_generation._payload_sha256(
            {
                "events": expected_raw_events,
                "query_text": expected_query_text,
            }
        )
        valid = valid and task.source.normalized_hash == expected_normalized_hash
        valid = valid and task.source.raw_hash == expected_raw_hash
    else:
        valid = False

    for reference in references:
        surface_text = getattr(reference, "surface_text", None)
        valid = valid and type(surface_text) is str
        valid = valid and getattr(reference, "normalized_text", None) == (
            surface_text.casefold() if type(surface_text) is str else None
        )
    for candidate in candidates:
        valid = valid and not _items(getattr(candidate, "source_anchors", None))

    if valid:
        return []
    return [
        _issue(
            "family_c_surface_integrity_mismatch",
            "Family C query, event, reference, candidate, source, and renderer surfaces must match the canonical selected template",
            "task",
        )
    ]


def _family_c_issues(task: MemUpdateTask) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    records, actions = _event_records(task)
    gold = task.gold
    profile = _mapping(task.metadata.resolved_profile)
    extra = _mapping(task.metadata.extra)
    raw_stratification = extra.get("stratification")
    stratification = raw_stratification if type(raw_stratification) is dict else {}
    difficulty = _enum_value(task.difficulty)

    queries = _items(task.queries)
    query = queries[0] if len(queries) == 1 else None
    query_type = _enum_value(getattr(query, "query_type", None))
    candidates = _items(getattr(query, "reference_candidates", None))
    references = _items(getattr(query, "surface_references", None))
    reference = references[0] if len(references) == 1 else None
    issues.extend(
        _family_c_surface_integrity_issues(
            task,
            records,
            query,
            candidates,
            references,
            extra,
        )
    )
    if (
        query is None
        or query_type != QueryType.UNRESOLVED_REFERENCE.value
        or len(candidates) != 2
        or len(references) != 1
    ):
        issues.append(
            _issue(
                "family_c_query_semantics_mismatch",
                "Family C requires one unresolved-reference query with two candidates and one surface reference",
                "queries",
            )
        )

    candidate_ids = [getattr(candidate, "candidate_id", None) for candidate in candidates]
    candidate_identities = [
        _identity(getattr(candidate, "object_key", None)) for candidate in candidates
    ]
    valid_identities = [identity for identity in candidate_identities if identity is not None]
    candidate_geometry_valid = (
        len(candidates) == 2
        and all(type(candidate_id) is str and bool(candidate_id.strip()) for candidate_id in candidate_ids)
        and len(set(candidate_ids)) == 2
        and len(valid_identities) == 2
        and len(set(valid_identities)) == 2
    )

    entity_condition = stratification.get("entity_condition")
    attribute_condition = stratification.get("attribute_condition")
    if entity_condition not in family_c_generation._ENTITY_CONDITIONS:
        candidate_geometry_valid = False
    if attribute_condition not in family_c_generation._ATTRIBUTE_CONDITIONS:
        candidate_geometry_valid = False

    surface_text = getattr(reference, "surface_text", None)
    normalized_text = getattr(reference, "normalized_text", None)
    surface_parts = surface_text.split(".", 1) if type(surface_text) is str else []
    entity_surface = surface_parts[0] if len(surface_parts) == 2 else None
    attribute_surface = surface_parts[1] if len(surface_parts) == 2 else None
    first_identity = candidate_identities[0] if len(candidate_identities) > 0 else None
    second_identity = candidate_identities[1] if len(candidate_identities) > 1 else None

    reviewed_geometry_valid = (
        candidate_geometry_valid
        and first_identity is not None
        and second_identity is not None
        and first_identity[0] in NAMESPACES
        and second_identity[0] in NAMESPACES
        and first_identity[1] in RELATION_QUALIFIED_ENTITIES
        and second_identity[1] in RELATION_QUALIFIED_ENTITIES
        and first_identity[2] in CANONICAL_ATTRIBUTES
        and first_identity[2] == second_identity[2]
        and first_identity[3] is None
        and second_identity[3] is None
    )
    expected_entity_mapping = None
    expected_namespace_evidence = None
    expected_entity_evidence_kind = None
    if reviewed_geometry_valid:
        if entity_condition == "distinct":
            reviewed_geometry_valid = (
                first_identity[0] == second_identity[0]
                and first_identity[1] != second_identity[1]
                and first_identity[1].rsplit("_", 1)[-1]
                != second_identity[1].rsplit("_", 1)[-1]
                and entity_surface == f"{first_identity[0]}:{first_identity[1]}"
            )
            expected_entity_mapping = f"exact_entity_v1:{first_identity[1]}"
            expected_namespace_evidence = f"qualified:{first_identity[0]}"
            expected_entity_evidence_kind = "exact_qualified_entity"
        elif entity_condition == "same_name":
            bare_name = first_identity[1].rsplit("_", 1)[-1]
            reviewed_geometry_valid = (
                first_identity[0] == second_identity[0]
                and first_identity[1] != second_identity[1]
                and entity_surface == bare_name
                and any(
                    first_identity[1] in group and second_identity[1] in group
                    for group in SAME_NAME_ENTITIES
                )
            )
            expected_entity_mapping = f"same_name_group_v1:{bare_name}"
            expected_namespace_evidence = (
                f"unqualified_with_shared_namespace:{first_identity[0]}"
            )
            expected_entity_evidence_kind = "unqualified_same_name"
        elif entity_condition == "alias":
            reviewed_geometry_valid = (
                first_identity[0] == second_identity[0]
                and first_identity[1] != second_identity[1]
                and first_identity[1].rsplit("_", 1)[-1]
                != second_identity[1].rsplit("_", 1)[-1]
                and (entity_surface, first_identity[1]) in ALIAS_MAPPINGS
            )
            expected_entity_mapping = (
                f"reviewed_alias_v1:{entity_surface}->{first_identity[1]}"
            )
            expected_namespace_evidence = (
                f"unqualified_with_shared_namespace:{first_identity[0]}"
            )
            expected_entity_evidence_kind = "reviewed_alias_map"
        elif entity_condition == "namespace_collision":
            reviewed_geometry_valid = (
                first_identity[0] != second_identity[0]
                and first_identity[1] == second_identity[1]
                and entity_surface == first_identity[1]
            )
            expected_entity_mapping = (
                f"namespace_collision_v1:{first_identity[1]}:"
                f"{first_identity[0]}|{second_identity[0]}"
            )
            expected_namespace_evidence = (
                f"unqualified:{first_identity[1]}@"
                f"{first_identity[0]}|{second_identity[0]}"
            )
            expected_entity_evidence_kind = "unqualified_namespace"

    if not reviewed_geometry_valid:
        issues.append(
            _issue(
                "family_c_candidate_geometry_mismatch",
                "Family C candidates must preserve the reviewed two-object entity, namespace, attribute, and subkey geometry",
                "queries[0].reference_candidates",
            )
        )

    expected_attribute_mapping = None
    expected_near_name_evidence = None
    expected_attribute_evidence_kind = None
    attribute_mapping_valid = first_identity is not None
    canonical_attribute = first_identity[2] if first_identity is not None else None
    if attribute_mapping_valid and attribute_condition == "exact":
        attribute_mapping_valid = attribute_surface == canonical_attribute
        expected_attribute_mapping = f"exact_attribute_v1:{canonical_attribute}"
        expected_near_name_evidence = (
            f"reviewed_match:{canonical_attribute}->{canonical_attribute}"
        )
        expected_attribute_evidence_kind = "exact_attribute"
    elif attribute_mapping_valid and attribute_condition == "paraphrase":
        attribute_mapping_valid = (
            attribute_surface,
            canonical_attribute,
        ) in family_c_generation._ATTRIBUTE_PARAPHRASE_MAPPINGS
        expected_attribute_mapping = (
            f"reviewed_attribute_paraphrase_v1:"
            f"{attribute_surface}->{canonical_attribute}"
        )
        expected_near_name_evidence = (
            f"reviewed_match:{attribute_surface}->{canonical_attribute}"
        )
        expected_attribute_evidence_kind = "reviewed_attribute_paraphrase"
    elif attribute_mapping_valid and attribute_condition == "near_name":
        attribute_mapping_valid = (
            attribute_surface,
            canonical_attribute,
        ) in family_c_generation._ATTRIBUTE_NEAR_NAMES
        expected_attribute_mapping = (
            f"near_name_nonmatch_v1:{attribute_surface}!{canonical_attribute}"
        )
        expected_near_name_evidence = (
            f"noncanonical_attribute:{attribute_surface}!={canonical_attribute}"
        )
        expected_attribute_evidence_kind = "near_name_nonmatch"
    else:
        attribute_mapping_valid = False

    expected_condition_kind = (
        "attribute_paraphrase"
        if entity_condition == "distinct" and attribute_condition == "paraphrase"
        else entity_condition
    )
    expected_evidence_kind = (
        f"{expected_entity_evidence_kind}+{expected_attribute_evidence_kind}"
        if expected_entity_evidence_kind is not None
        and expected_attribute_evidence_kind is not None
        else None
    )
    candidate_evidence_valid = len(candidates) == 2
    for index, candidate in enumerate(candidates):
        identity = candidate_identities[index]
        if identity is None:
            candidate_evidence_valid = False
            continue
        expected_evidence = (
            f"event_candidate={index}; namespace={identity[0]}; "
            f"entity={identity[1]}; attribute={identity[2]}"
        )
        candidate_evidence_valid = candidate_evidence_valid and (
            getattr(candidate, "evidence", None) == expected_evidence
            and not _items(getattr(candidate, "source_anchors", None))
        )
    reviewed_mapping_valid = (
        reviewed_geometry_valid
        and attribute_mapping_valid
        and stratification.get("entity_mapping_id") == expected_entity_mapping
        and stratification.get("attribute_mapping_id") == expected_attribute_mapping
        and stratification.get("namespace_evidence") == expected_namespace_evidence
        and stratification.get("near_name_evidence") == expected_near_name_evidence
        and reference is not None
        and surface_text == normalized_text
        and getattr(reference, "condition_kind", None) == expected_condition_kind
        and getattr(reference, "evidence_kind", None) == expected_evidence_kind
        and candidate_evidence_valid
    )
    if not reviewed_mapping_valid:
        issues.append(
            _issue(
                "family_c_reviewed_mapping_mismatch",
                "Family C surface, alias, attribute, namespace, and nonmatch evidence must match the reviewed catalogs",
                "metadata.extra.stratification",
            )
        )

    event_count_valid = len(records) == 2 and len(actions) == 2
    if not event_count_valid:
        issues.append(
            _issue(
                "family_c_event_count_mismatch",
                "Family C requires exactly two candidate-establishing events and actions",
                "events",
            )
        )

    action_ids_in_event_order: list[Any] = []
    event_values: list[Any] = []
    linkage_valid = event_count_valid
    write_semantics_valid = event_count_valid
    for index, record in enumerate(records):
        event = record["event"]
        event_action_ids = _items(getattr(event, "gold_action_ids", None))
        action_ids_in_event_order.extend(event_action_ids)
        action = record["actions"][0] if len(record["actions"]) == 1 else None
        identity = candidate_identities[index] if index < len(candidate_identities) else None
        expected_candidate_id = candidate_ids[index] if index < len(candidate_ids) else None
        if action is None:
            linkage_valid = False
            write_semantics_valid = False
            event_values.append(None)
            continue
        action_targets = _action_targets(action)
        value = getattr(action, "value", None)
        event_values.append(value)
        renderer = record["metadata"].get("__surface_renderer__")
        linkage_valid = linkage_valid and (
            len(event_action_ids) == 1
            and getattr(action, "action_id", None) == event_action_ids[0]
            and getattr(action, "event_id", None) == getattr(event, "event_id", None)
            and action_targets == ([identity] if identity is not None else [])
            and _strict_int_equal(record["metadata"].get("candidate_index"), index)
            and _mapping(getattr(event, "source_anchor", None)) == {"event_index": index}
            and _strict_int_equal(getattr(event, "sequence_index", None), index)
            and type(expected_candidate_id) is str
            and type(renderer) is dict
        )
        write_semantics_valid = write_semantics_valid and (
            _enum_value(getattr(action, "operation", None)) == Operation.ADD.value
            and _enum_value(getattr(action, "scope", None)) == "attribute"
            and len(action_targets) == 1
            and value is not None
            and not bool(_mapping(getattr(action, "expected_effect", None)))
            and _enum_value(getattr(event, "role", None)) == EventRole.LATEST_GOLD.value
        )
    if action_ids_in_event_order != _items(gold.action_sequence):
        linkage_valid = False
    if not write_semantics_valid:
        issues.append(
            _issue(
                "family_c_write_semantics_mismatch",
                "Family C candidate events must be ADD-only latest-gold writes with no NOOPs or other operations",
                "gold.actions",
            )
        )

    declared_identities = [
        _identity(key) for key in _items(task.target_objects)
    ]
    query_target_identities = [
        _identity(key)
        for key in _items(getattr(query, "target_object_keys", None))
    ]
    expected_present_identities = [
        _identity(key) for key in _items(gold.expected_present_objects)
    ]
    expected_final_state: dict[str, Any] = {}
    expected_history: dict[str, list[Any]] = {}
    if len(candidate_identities) == 2 and len(event_values) == 2:
        for identity, value in zip(candidate_identities, event_values):
            if identity is not None and value is not None:
                object_id = _identity_id(identity)
                expected_final_state[object_id] = value
                expected_history[object_id] = [value]
    linkage_valid = linkage_valid and (
        declared_identities == candidate_identities
        and query_target_identities == candidate_identities
        and expected_present_identities == candidate_identities
        and not _items(gold.expected_absent_objects)
        and _same_value(_mapping(gold.final_state), expected_final_state)
        and _same_value(_mapping(gold.version_history), expected_history)
        and _items(gold.gold_source_event_ids)
        == [getattr(record["event"], "event_id", None) for record in records]
    )
    if not linkage_valid:
        issues.append(
            _issue(
                "family_c_linkage_mismatch",
                "Family C candidate, event, action, target, replay state, history, and source links must agree exactly",
                "gold",
            )
        )

    expected_status = None
    expected_disposition = None
    if (
        entity_condition in family_c_generation._ENTITY_CONDITIONS
        and attribute_condition in family_c_generation._ATTRIBUTE_CONDITIONS
    ):
        expected_status, expected_disposition = family_c_generation._resolution(
            entity_condition,
            attribute_condition,
        )
    query_id = getattr(query, "query_id", None)
    canonical_answers = _mapping(gold.canonical_answers)
    canonical = canonical_answers.get(query_id) if type(query_id) is str else None
    answer_support_valid = (
        type(query_id) is str
        and len(queries) == 1
        and set(canonical_answers) == {query_id}
        and not _mapping(gold.gold_answers)
        and not _mapping(gold.acceptable_answers)
    )
    if not answer_support_valid:
        issues.append(
            _issue(
                "family_c_answer_support_mismatch",
                "Family C requires exactly one canonical answer for its sole unresolved query and no ordinary answer maps",
                "gold.canonical_answers",
            )
        )

    expected_linked_ids: list[Any] = []
    expected_selected_ids: list[Any] = []
    expected_value = None
    expected_reason = None
    if expected_status is ReferenceResolutionStatus.UNIQUE:
        expected_linked_ids = candidate_ids[:1]
        expected_selected_ids = candidate_ids[:1]
        expected_value = event_values[0] if event_values else None
    elif expected_status is ReferenceResolutionStatus.AMBIGUOUS:
        expected_linked_ids = list(candidate_ids)
        expected_reason = "reference matches multiple exact memory objects"
    elif expected_status is ReferenceResolutionStatus.NO_MATCH:
        expected_reason = "reference attribute has no reviewed canonical match"

    resolution_valid = (
        canonical is not None
        and type(canonical) is CanonicalAnswer
        and _enum_value(getattr(canonical, "resolution_status", None))
        == _enum_value(expected_status)
        and _enum_value(getattr(canonical, "disposition", None))
        == _enum_value(expected_disposition)
        and _items(getattr(reference, "candidate_ids", None)) == expected_linked_ids
        and _items(getattr(canonical, "selected_candidate_ids", None))
        == expected_selected_ids
        and _same_value(getattr(canonical, "value", None), expected_value)
        and getattr(canonical, "abstention_reason", None) == expected_reason
    )
    if not resolution_valid:
        issues.append(
            _issue(
                "family_c_resolution_truth_table_mismatch",
                "Family C resolution status, typed disposition, reference links, selected candidates, abstention, and value must follow the condition truth table",
                "gold.canonical_answers",
            )
        )

    expected_difficulty = None
    if (
        entity_condition in family_c_generation._ENTITY_CONDITIONS
        and attribute_condition in family_c_generation._ATTRIBUTE_CONDITIONS
    ):
        expected_difficulty = family_c_generation._difficulty(
            entity_condition,
            attribute_condition,
        ).value
    expected_entity_ambiguity = family_c_generation._ENTITY_AMBIGUITY.get(
        entity_condition
    )
    expected_attribute_ambiguity = family_c_generation._ATTRIBUTE_AMBIGUITY.get(
        attribute_condition
    )
    required_profile = {
        "update_depth": 1,
        "update_depth_bucket": "1",
        "active_object_count": 2,
        "entity_ambiguity": expected_entity_ambiguity,
        "attribute_ambiguity": expected_attribute_ambiguity,
        "noop_density": 0.0,
        "cross_slot_interleaving": 0.0,
        "stale_count": 0,
        "context_length": 2,
        "context_order": "chronological",
        "version_metadata": "none",
        "query_type": QueryType.UNRESOLVED_REFERENCE.value,
        "source_naturalness": "mixed_template",
        "alias_namespace_condition": entity_condition,
        "difficulty": expected_difficulty,
        "profile_name": expected_difficulty,
        "task_family": TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value,
    }
    required_stratification = {
        "entity_condition": entity_condition,
        "attribute_condition": attribute_condition,
        "resolution_status": _enum_value(expected_status),
        "answer_disposition": _enum_value(expected_disposition),
        "candidate_count": 2,
        "entity_mapping_id": expected_entity_mapping,
        "attribute_mapping_id": expected_attribute_mapping,
        "namespace_evidence": expected_namespace_evidence,
        "near_name_evidence": expected_near_name_evidence,
        "difficulty": expected_difficulty,
        "num_events": 2,
        "num_target_updates": 0,
        "noop_count": 0,
    }
    profile_valid = difficulty == expected_difficulty
    profile_valid = profile_valid and all(
        key in profile and _same_value(profile[key], value)
        for key, value in required_profile.items()
    )
    profile_valid = profile_valid and all(
        key in stratification and _same_value(stratification[key], value)
        for key, value in required_stratification.items()
    )
    surface_variant = extra.get("surface_variant")
    expected_surface_template = (
        REFERENCE_QUERY_TEMPLATE_SETS[surface_variant][0]
        if type(surface_variant) is int
        and 0 <= surface_variant < len(REFERENCE_QUERY_TEMPLATE_SETS)
        else None
    )
    profile_valid = profile_valid and (
        expected_surface_template is not None
        and extra.get("surface_template") == expected_surface_template
    )
    for record in records:
        renderer = record["metadata"].get("__surface_renderer__")
        profile_valid = profile_valid and type(renderer) is dict and renderer == {
            "surface_template": expected_surface_template,
            "surface_variant": surface_variant,
        }
    if query is not None:
        query_renderer = _mapping(getattr(query, "metadata", None)).get(
            "__surface_renderer__"
        )
        profile_valid = profile_valid and type(query_renderer) is dict and query_renderer == {
            "surface_template": expected_surface_template,
            "surface_variant": surface_variant,
        }
    if not profile_valid:
        issues.append(
            _issue(
                "family_c_counter_profile_mismatch",
                "Family C task-local profile, difficulty, condition, count, surface, mapping, evidence, status, and disposition fields must match observed semantics",
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
    allow_superseded_non_target_answer_overlap: bool = False,
    allow_noop_answer_observation_overlap: bool = False,
    include_distractors: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    validators = [validate_task, validate_gold_replay]
    if include_distractors:
        validators.append(
            lambda candidate: validate_distractors(
                candidate,
                allow_superseded_non_target_answer_overlap=(
                    allow_superseded_non_target_answer_overlap
                ),
                allow_noop_answer_observation_overlap=(
                    allow_noop_answer_observation_overlap
                ),
            )
        )
    for validator in validators:
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
    precheck = _strict_family_precheck(
        task,
        family="a",
        expected_family=TaskFamily.REPEATED_SAME_SLOT.value,
    )
    if precheck is not None:
        return precheck

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
    return _run_strict_family_validator(task, _validate_family_a_task, family="a")


def _validate_family_b_task(task: Any) -> ValidationReport:
    precheck = _strict_family_precheck(
        task,
        family="b",
        expected_family=TaskFamily.INTERLEAVED_MULTI_SLOT.value,
    )
    if precheck is not None:
        return precheck

    schema_issues = _schema_preflight_issues(task, family="b")
    if schema_issues:
        return _bounded_report(schema_issues, family="b")

    issues = _contract_constraint_issues(task, family="b")
    issues.extend(
        _generic_validation_issues(
            task,
            family="b",
            allow_superseded_non_target_answer_overlap=True,
        )
    )
    issues.extend(_family_b_preflight_issues(task, schema_checked=True))
    try:
        issues.extend(_family_b_issues(task))
    except Exception:
        issues.append(
            _issue(
                "family_b_malformed_task",
                "Family B semantic inspection rejected malformed task structure",
                "task",
            )
        )
    return _bounded_report(issues, family="b")


def validate_family_b_task(task: Any) -> ValidationReport:
    """Validate one Family B task without mutation, external I/O, or exception leaks."""
    return _run_strict_family_validator(task, _validate_family_b_task, family="b")


def _validate_family_c_task(task: Any) -> ValidationReport:
    precheck = _strict_family_precheck(
        task,
        family="c",
        expected_family=TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value,
    )
    if precheck is not None:
        return precheck

    schema_issues = _schema_preflight_issues(task, family="c")
    if schema_issues:
        return _bounded_report(schema_issues, family="c")

    issues = _contract_constraint_issues(task, family="c")
    issues.extend(
        _generic_validation_issues(
            task,
            family="c",
            include_distractors=False,
        )
    )
    try:
        issues.extend(_family_c_issues(task))
    except Exception:
        issues.append(
            _issue(
                "family_c_malformed_task",
                "Family C semantic inspection rejected malformed task structure",
                "task",
            )
        )
    return _bounded_report(issues, family="c")


def validate_family_c_task(task: Any) -> ValidationReport:
    """Validate one Family C task without mutation, external I/O, or exception leaks."""
    return _run_strict_family_validator(task, _validate_family_c_task, family="c")


def _validate_family_d_task(task: Any) -> ValidationReport:
    precheck = _strict_family_precheck(
        task,
        family="d",
        expected_family=TaskFamily.NOOP_WRITE_DISCIPLINE.value,
    )
    if precheck is not None:
        return precheck

    preflight_issues = _preflight_issues(task)
    if preflight_issues:
        return _bounded_report(preflight_issues)

    issues = _contract_constraint_issues(task, family="d")
    issues.extend(
        _generic_validation_issues(
            task,
            family="d",
            allow_noop_answer_observation_overlap=True,
        )
    )
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
    return _run_strict_family_validator(task, _validate_family_d_task, family="d")


def validate_pilot_task(task: Any) -> ValidationReport:
    """Validate with explicit strict Pilot family semantics when available."""
    if _task_identifies_family(task, TaskFamily.REPEATED_SAME_SLOT.value):
        return validate_family_a_task(task)
    if _task_identifies_family(task, TaskFamily.INTERLEAVED_MULTI_SLOT.value):
        return validate_family_b_task(task)
    if _task_identifies_family(task, TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value):
        return validate_family_c_task(task)
    if _task_identifies_family(task, TaskFamily.NOOP_WRITE_DISCIPLINE.value):
        return validate_family_d_task(task)
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(task),
    )


__all__ = [
    "validate_family_a_task",
    "validate_family_b_task",
    "validate_family_c_task",
    "validate_family_d_task",
    "validate_pilot_task",
]
