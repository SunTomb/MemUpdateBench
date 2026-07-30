from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict, field_validator, model_validator

from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.contracts.common import ImmutableContractModel


Verdict = Literal["pass", "block", "needs_revision"]
_MAX_AUDIT_RECORDS = 4096
_MAX_CONDITIONS = 128


class _StrictFrozenAuditModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


class _FrozenList(list[str]):
    def _frozen(self, *args, **kwargs):
        raise TypeError("audit conditions are frozen")

    __delitem__ = __setitem__ = __iadd__ = __imul__ = _frozen
    append = clear = extend = insert = pop = remove = reverse = sort = _frozen


class AuditSelection(_StrictFrozenAuditModel):
    audit_id: str
    task_id: str
    family: TaskFamily
    difficulty: Difficulty
    split: Split
    covered_conditions: list[str]
    selection_reason: str

    @field_validator("audit_id", "task_id", "selection_reason")
    @classmethod
    def _require_nonblank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("covered_conditions", mode="before")
    @classmethod
    def _require_exact_condition_list(cls, value: Any) -> list[str]:
        if type(value) is not list:
            raise ValueError("covered_conditions must be an exact list")
        if len(value) > _MAX_CONDITIONS:
            raise ValueError("covered_conditions exceeds the bounded audit limit")
        return value

    @field_validator("covered_conditions")
    @classmethod
    def _normalize_and_freeze_conditions(cls, value: list[str]) -> _FrozenList:
        conditions: list[str] = []
        seen: set[str] = set()
        for condition in value:
            condition = condition.strip()
            if not condition:
                raise ValueError("covered_conditions must contain nonblank strings")
            if condition in seen:
                raise ValueError("covered_conditions must be unique")
            seen.add(condition)
            conditions.append(condition)
        return _FrozenList(conditions)


class AuditDecision(_StrictFrozenAuditModel):
    audit_id: str
    reviewer: str
    verdict: Verdict
    answer_unique: bool
    actions_correct: bool
    roles_correct: bool
    surface_natural: bool
    notes: str

    @field_validator("audit_id", "reviewer")
    @classmethod
    def _require_nonblank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @property
    def all_checks_pass(self) -> bool:
        return all(
            (
                self.answer_unique,
                self.actions_correct,
                self.roles_correct,
                self.surface_natural,
            )
        )


class AuditDecisionTemplate(_StrictFrozenAuditModel):
    """A blank placeholder that cannot be mistaken for a human decision."""

    audit_id: str
    reviewer: str | None = None
    verdict: Verdict | None = None
    answer_unique: bool | None = None
    actions_correct: bool | None = None
    roles_correct: bool | None = None
    surface_natural: bool | None = None
    notes: str | None = None

    @field_validator("audit_id")
    @classmethod
    def _require_nonblank_audit_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("audit_id must not be blank")
        return value

    @model_validator(mode="after")
    def _remain_blank(self):
        optional_fields = (
            self.reviewer,
            self.verdict,
            self.answer_unique,
            self.actions_correct,
            self.roles_correct,
            self.surface_natural,
            self.notes,
        )
        if any(value is not None for value in optional_fields):
            raise ValueError("audit decision templates must remain blank")
        return self

    @property
    def release_ready(self) -> bool:
        return False


class AuditGateReport(_StrictFrozenAuditModel):
    selected_audit_ids: tuple[str, ...]
    malformed_selection_ids: tuple[str, ...] = ()
    duplicate_selection_ids: tuple[str, ...] = ()
    missing_audit_ids: tuple[str, ...] = ()
    duplicate_audit_ids: tuple[str, ...] = ()
    foreign_audit_ids: tuple[str, ...] = ()
    malformed_decision_ids: tuple[str, ...] = ()
    non_pass_audit_ids: tuple[str, ...] = ()
    failed_check_audit_ids: tuple[str, ...] = ()
    release_ready: bool

    @field_validator(
        "selected_audit_ids",
        "malformed_selection_ids",
        "duplicate_selection_ids",
        "missing_audit_ids",
        "duplicate_audit_ids",
        "foreign_audit_ids",
        "malformed_decision_ids",
        "non_pass_audit_ids",
        "failed_check_audit_ids",
        mode="before",
    )
    @classmethod
    def _validate_id_sequences(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("audit ID report fields must be lists or tuples")
        if len(value) > _MAX_AUDIT_RECORDS:
            raise ValueError("audit ID report field exceeds the bounded audit limit")
        result = []
        for item in value:
            if type(item) is not str or not item.strip():
                raise ValueError("audit ID report fields require nonblank strings")
            result.append(item)
        return tuple(result)

    @model_validator(mode="after")
    def _validate_release_ready_claim(self):
        issue_fields = (
            self.malformed_selection_ids,
            self.duplicate_selection_ids,
            self.missing_audit_ids,
            self.duplicate_audit_ids,
            self.foreign_audit_ids,
            self.malformed_decision_ids,
            self.non_pass_audit_ids,
            self.failed_check_audit_ids,
        )
        if self.release_ready and (
            not self.selected_audit_ids or any(issue_fields)
        ):
            raise ValueError("release_ready conflicts with reported audit issues")
        return self

    @property
    def duplicate_decision_ids(self) -> tuple[str, ...]:
        return self.duplicate_audit_ids

    @property
    def foreign_decision_ids(self) -> tuple[str, ...]:
        return self.foreign_audit_ids


def audit_decision_template(audit_id: str) -> AuditDecisionTemplate:
    return AuditDecisionTemplate(audit_id=audit_id)


def _snapshot_input(value: Any, *, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be an exact list or tuple")
    if len(value) > _MAX_AUDIT_RECORDS:
        raise ValueError(f"{name} exceeds the bounded audit limit")
    return tuple(value)


def _raw_fields(value: Any) -> dict[str, Any] | None:
    if type(value) is dict:
        return value
    if type(value) not in (AuditSelection, AuditDecision, AuditDecisionTemplate):
        return None
    try:
        raw = object.__getattribute__(value, "__dict__")
    except Exception:
        return None
    return raw if type(raw) is dict else None


def _record_audit_id(value: Any, index: int) -> str:
    raw = _raw_fields(value)
    audit_id = raw.get("audit_id") if raw is not None else None
    if type(audit_id) is str and audit_id.strip():
        return audit_id
    return f"<index:{index}>"


def _snapshot_record(value: Any, model_type: type):
    if type(value) is not model_type:
        return None
    raw = _raw_fields(value)
    if raw is None or set(raw) != set(model_type.model_fields):
        return None
    candidate = dict(raw)
    if model_type is AuditSelection and type(candidate.get("covered_conditions")) is _FrozenList:
        candidate["covered_conditions"] = list(candidate["covered_conditions"])
    try:
        return model_type.model_validate(candidate)
    except Exception:
        return None


def _sort_report_ids(values: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(values, key=lambda audit_id: (audit_id.startswith("<index:"), audit_id))
    )


def evaluate_audit_gate(
    selections: Sequence[Any], decisions: Sequence[Any]
) -> AuditGateReport:
    """Evaluate frozen, bounded snapshots of selections and human decisions."""
    selected_snapshot = _snapshot_input(selections, name="selections")
    decision_snapshot = _snapshot_input(decisions, name="decisions")

    valid_selections: list[AuditSelection] = []
    malformed_selection_ids: set[str] = set()
    for index, item in enumerate(selected_snapshot):
        snapshot = _snapshot_record(item, AuditSelection)
        if snapshot is not None:
            valid_selections.append(snapshot)
        else:
            malformed_selection_ids.add(_record_audit_id(item, index))

    selection_counts = Counter(item.audit_id for item in valid_selections)
    selected_ids = tuple(sorted(selection_counts))
    selected_set = set(selected_ids)
    duplicate_selection_ids = tuple(
        sorted(audit_id for audit_id, count in selection_counts.items() if count > 1)
    )

    valid_decisions: list[AuditDecision] = []
    malformed_decision_ids: set[str] = set()
    observed_decision_ids: list[str] = []
    for index, item in enumerate(decision_snapshot):
        audit_id = _record_audit_id(item, index)
        if not audit_id.startswith("<index:"):
            observed_decision_ids.append(audit_id)
        snapshot = _snapshot_record(item, AuditDecision)
        if snapshot is not None:
            valid_decisions.append(snapshot)
        else:
            malformed_decision_ids.add(audit_id)

    decision_counts = Counter(observed_decision_ids)
    valid_decision_ids = {item.audit_id for item in valid_decisions}
    duplicate_ids = tuple(
        sorted(audit_id for audit_id, count in decision_counts.items() if count > 1)
    )
    foreign_ids = tuple(
        sorted({audit_id for audit_id in observed_decision_ids if audit_id not in selected_set})
    )
    missing_ids = tuple(sorted(selected_set - valid_decision_ids))
    non_pass_ids = tuple(
        sorted(
            {
                item.audit_id
                for item in valid_decisions
                if item.audit_id in selected_set and item.verdict != "pass"
            }
        )
    )
    failed_check_ids = tuple(
        sorted(
            {
                item.audit_id
                for item in valid_decisions
                if item.audit_id in selected_set and not item.all_checks_pass
            }
        )
    )
    release_ready = bool(
        selected_ids
        and not malformed_selection_ids
        and not duplicate_selection_ids
        and not missing_ids
        and not duplicate_ids
        and not foreign_ids
        and not malformed_decision_ids
        and not non_pass_ids
        and not failed_check_ids
        and len(valid_decisions) == len(selected_ids)
    )
    return AuditGateReport(
        selected_audit_ids=selected_ids,
        malformed_selection_ids=_sort_report_ids(malformed_selection_ids),
        duplicate_selection_ids=duplicate_selection_ids,
        missing_audit_ids=missing_ids,
        duplicate_audit_ids=duplicate_ids,
        foreign_audit_ids=foreign_ids,
        malformed_decision_ids=_sort_report_ids(malformed_decision_ids),
        non_pass_audit_ids=non_pass_ids,
        failed_check_audit_ids=failed_check_ids,
        release_ready=release_ready,
    )


validate_audit_decisions = evaluate_audit_gate
validate_audit_gate = evaluate_audit_gate


__all__ = [
    "AuditDecision",
    "AuditDecisionTemplate",
    "AuditGateReport",
    "AuditSelection",
    "Verdict",
    "audit_decision_template",
    "evaluate_audit_gate",
    "validate_audit_decisions",
    "validate_audit_gate",
]
