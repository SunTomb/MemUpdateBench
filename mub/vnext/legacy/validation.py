"""Internal legacy validation boundary.

These Python-private helpers are not a security sandbox. Compiler callers must own
strict source construction, and artifact callers receive compatibility validation
only after manifest authentication binds each task ID to its canonical hash. Public
validation APIs remain strict and never produce a compatibility-waived claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mub.vnext.contracts.enums import EventRole, Operation
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.io.canonical import sha256_model
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
    merge_reports,
)
from mub.vnext.validation.replay import (
    _text_contains_value,
    _validate_distractors_with_accepted_overlap,
    validate_gold_replay,
)
from mub.vnext.validation.task import validate_task


_LEGACY_AMBIGUITY_RULE = "non_target_accepted_answer_text_overlap_v1"
_LEGACY_AMBIGUITY_METADATA_KEYS = frozenset(
    {
        "allow_accepted_answer_ambiguity",
        "compatibility_rule",
        "legacy_event",
        "legacy_role",
    }
)


@dataclass(frozen=True, slots=True)
class _AuthenticatedLegacyValidationContext:
    manifest_sha256: str
    task_hashes: tuple[tuple[str, str], ...]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    return []


def _targets(action: Any) -> list[Any]:
    return _list(getattr(action, "target_object_keys", None))


def _legacy_audited_answer_overlap(
    task: MemUpdateTask,
    event: Any,
    event_actions: list[Any],
    accepted: Any,
) -> bool:
    if _enum_value(getattr(event, "role", None)) != EventRole.NEUTRAL.value:
        return False
    metadata = _mapping(getattr(event, "metadata", None))
    if not set(metadata).issubset(_LEGACY_AMBIGUITY_METADATA_KEYS):
        return False
    if (
        metadata.get("allow_accepted_answer_ambiguity") is not True
        or metadata.get("compatibility_rule") != _LEGACY_AMBIGUITY_RULE
        or metadata.get("legacy_role") != EventRole.NEUTRAL.value
    ):
        return False
    legacy_event = metadata.get("legacy_event")
    if legacy_event is not None:
        if not isinstance(legacy_event, Mapping) or not set(legacy_event).issubset(
            {"annotation", "condition"}
        ):
            return False
        if any(
            value is not None and type(value) not in {str, bool, int, float}
            for value in legacy_event.values()
        ):
            return False
    task_metadata = getattr(task, "metadata", None)
    extra = _mapping(getattr(task_metadata, "extra", None))
    if _list(extra.get("compatibility_policies")) != [_LEGACY_AMBIGUITY_RULE]:
        return False
    if len(event_actions) != 1:
        return False
    action = event_actions[0]
    referenced_ids = _list(getattr(event, "gold_action_ids", None))
    expected_effect = _mapping(getattr(action, "expected_effect", None))
    if (
        referenced_ids != [getattr(action, "action_id", None)]
        or getattr(action, "event_id", None) != getattr(event, "event_id", None)
        or _enum_value(getattr(action, "operation", None)) != Operation.NOOP.value
        or _enum_value(getattr(action, "scope", None)) != "object"
        or _targets(action)
        or getattr(action, "value", None) is not None
        or getattr(action, "effective_at", None) is not None
        or dict(expected_effect) != {"operation": Operation.NOOP.value}
    ):
        return False
    gold = getattr(task, "gold", None)
    if getattr(event, "event_id", None) in _list(
        getattr(gold, "gold_source_event_ids", None)
    ):
        return False
    raw_text = getattr(event, "raw_text", None)
    normalized_text = getattr(event, "normalized_text", None)
    return (
        isinstance(raw_text, str)
        and normalized_text == raw_text.strip()
        and _text_contains_value(raw_text, accepted)
        and _text_contains_value(normalized_text, accepted)
    )


def _legacy_task_semantics(task: MemUpdateTask) -> ValidationReport:
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        _validate_distractors_with_accepted_overlap(
            task,
            _legacy_audited_answer_overlap,
        ),
    )


def _validate_compiler_constructed_legacy_task_semantics(
    task: MemUpdateTask,
) -> ValidationReport:
    """Validate only a task just built from strict legacy compiler inputs.

    This internal helper is not a security sandbox. Its caller must own the trusted
    construction boundary; arbitrary tasks must use the public strict validators.
    """
    return _legacy_task_semantics(task)


def _validate_authenticated_legacy_task_semantics(
    task: MemUpdateTask,
    context: _AuthenticatedLegacyValidationContext,
) -> ValidationReport:
    """Validate a task only when bound to an authenticated manifest context."""
    if type(context) is not _AuthenticatedLegacyValidationContext:
        return build_report(
            [
                ValidationIssue(
                    code="legacy_validation_context_required",
                    message="authenticated legacy validation context is required",
                    path="task",
                    severity="error",
                )
            ]
        )
    expected_hash = dict(context.task_hashes).get(getattr(task, "task_id", None))
    if expected_hash is None or expected_hash != sha256_model(task):
        return build_report(
            [
                ValidationIssue(
                    code="legacy_validation_context_task_mismatch",
                    message="task is not bound to the authenticated legacy context",
                    path="task",
                    severity="error",
                )
            ]
        )
    return _legacy_task_semantics(task)
