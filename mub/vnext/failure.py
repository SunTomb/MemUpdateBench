from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Literal

from mub.vnext.version import PRIMARY_FAILURE_PRECEDENCE_VERSION


class FailureFlag(str, Enum):
    INVALID_ACTION_FORMAT = "invalid_action_format"
    UNSUPPORTED_ACTION = "unsupported_action"
    WRONG_OPERATION = "wrong_operation"
    WRONG_ENTITY = "wrong_entity"
    WRONG_ATTRIBUTE = "wrong_attribute"
    WRONG_VALUE = "wrong_value"
    FALSE_WRITE = "false_write"
    MISSED_UPDATE = "missed_update"
    COLLATERAL_CORRUPTION = "collateral_corruption"
    DELETION_FAILURE = "deletion_failure"
    CURRENT_STATE_MISSING = "current_state_missing"
    STALE_RETAINED = "stale_retained"
    CURRENT_NOT_RETRIEVED = "current_not_retrieved"
    STALE_RETRIEVED = "stale_retrieved"
    STALE_COPIED = "stale_copied"
    DISTRACTOR_RETRIEVED = "distractor_retrieved"
    DISTRACTOR_COPIED = "distractor_copied"
    GOLD_RETRIEVED_WRONG_ANSWER = "gold_retrieved_wrong_answer"
    ANSWER_FORMAT_ONLY = "answer_format_only"
    SYSTEM_EXCEPTION = "system_exception"

    def __str__(self) -> str:
        return self.value


FAILURE_FLAGS = tuple(flag.value for flag in FailureFlag)
PRIMARY_FAILURE_PRECEDENCE = (
    ("system_exception", "invalid_action_format", "unsupported_action"),
    (
        "wrong_operation",
        "wrong_entity",
        "wrong_attribute",
        "wrong_value",
        "false_write",
        "missed_update",
    ),
    (
        "collateral_corruption",
        "deletion_failure",
        "current_state_missing",
        "stale_retained",
    ),
    ("current_not_retrieved", "stale_retrieved", "distractor_retrieved"),
    ("stale_copied", "distractor_copied", "gold_retrieved_wrong_answer"),
    ("answer_format_only",),
)
_FAILURE_FLAG_SET = frozenset(FAILURE_FLAGS)
PrimaryFailure = FailureFlag | Literal["correct"]


def _consume_exact_failure_strings(flags: Iterable[str]) -> tuple[str, ...]:
    if flags is None or type(flags) in {str, bytes} or isinstance(flags, Mapping):
        raise TypeError("failure flags must be a non-string iterable")
    try:
        iterator = iter(flags)
    except Exception as exc:
        raise TypeError("failed to obtain failure flag iterator") from exc
    supplied: list[str] = []
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise TypeError("failed to consume failure flag iterator") from exc
        if type(item) is not str:
            raise TypeError("failure flags must be exact built-in strings")
        supplied.append(item)
    return tuple(supplied)


def canonicalize_failure_flags(flags: Iterable[str]) -> tuple[str, ...]:
    supplied = _consume_exact_failure_strings(flags)
    unknown = frozenset(supplied) - _FAILURE_FLAG_SET
    if unknown:
        raise ValueError(f"unknown failure flags: {sorted(unknown)}")
    present = frozenset(supplied)
    return tuple(flag for flag in FAILURE_FLAGS if flag in present)


def primary_failure(flags: Iterable[str]) -> str:
    supplied = frozenset(canonicalize_failure_flags(flags))
    for layer in PRIMARY_FAILURE_PRECEDENCE:
        for flag in layer:
            if flag in supplied:
                return flag
    return "correct"


__all__ = [
    "FAILURE_FLAGS",
    "PRIMARY_FAILURE_PRECEDENCE",
    "PRIMARY_FAILURE_PRECEDENCE_VERSION",
    "FailureFlag",
    "PrimaryFailure",
    "canonicalize_failure_flags",
    "primary_failure",
]
