from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, computed_field, field_validator, model_validator

from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.contracts.common import ImmutableContractModel


Verdict = Literal["pass", "block", "needs_revision"]
_MAX_AUDIT_RECORDS = 4096
_MAX_CONDITIONS = 128


class _StrictFrozenAuditModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


class AuditSelection(_StrictFrozenAuditModel):
    """A frozen selection; conditions are stored as a sorted tuple."""

    audit_id: str
    task_id: str
    family: TaskFamily
    difficulty: Difficulty
    split: Split
    covered_conditions: tuple[str, ...]
    selection_reason: str

    @model_validator(mode="before")
    @classmethod
    def _accept_canonical_enum_strings(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        candidate = dict(value)
        enum_fields = {
            "family": TaskFamily,
            "difficulty": Difficulty,
            "split": Split,
        }
        for field_name, enum_type in enum_fields.items():
            field_value = candidate.get(field_name)
            if type(field_value) is str:
                try:
                    candidate[field_name] = enum_type(field_value)
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} must be an exact {enum_type.__name__} value"
                    ) from exc
            elif isinstance(field_value, str) and type(field_value) is not enum_type:
                raise ValueError(f"{field_name} must be an exact built-in string value")
        return candidate

    @field_validator("audit_id", "task_id", "selection_reason")
    @classmethod
    def _require_nonblank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("covered_conditions", mode="before")
    @classmethod
    def _require_exact_condition_sequence(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("covered_conditions must be an exact list or tuple")
        if len(value) > _MAX_CONDITIONS:
            raise ValueError("covered_conditions exceeds the bounded audit limit")
        if any(type(condition) is not str for condition in value):
            raise ValueError("covered_conditions must contain exact strings")
        return tuple(value)

    @field_validator("covered_conditions")
    @classmethod
    def _normalize_and_freeze_conditions(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
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
        return tuple(sorted(conditions))


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
    decision_evidence: tuple[AuditDecision, ...] = ()
    malformed_selection_ids: tuple[str, ...] = ()
    duplicate_selection_ids: tuple[str, ...] = ()
    missing_audit_ids: tuple[str, ...] = ()
    duplicate_audit_ids: tuple[str, ...] = ()
    foreign_audit_ids: tuple[str, ...] = ()
    malformed_decision_ids: tuple[str, ...] = ()
    non_pass_audit_ids: tuple[str, ...] = ()
    failed_check_audit_ids: tuple[str, ...] = ()

    @field_validator("decision_evidence", mode="before")
    @classmethod
    def _validate_decision_evidence(cls, value: Any) -> tuple[AuditDecision, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("decision_evidence must be a list or tuple")
        if len(value) > _MAX_AUDIT_RECORDS:
            raise ValueError("decision_evidence exceeds the bounded audit limit")
        decisions: list[AuditDecision] = []
        for item in value:
            if type(item) is AuditDecision:
                try:
                    raw = object.__getattribute__(item, "__dict__")
                except Exception as exc:
                    raise ValueError(
                        "decision_evidence requires intact AuditDecision records"
                    ) from exc
                if type(raw) is not dict or set(raw) != set(AuditDecision.model_fields):
                    raise ValueError(
                        "decision_evidence requires intact AuditDecision records"
                    )
                payload = dict(raw)
            elif type(item) is dict:
                payload = item
            else:
                raise ValueError(
                    "decision_evidence requires AuditDecision records or exact dictionaries"
                )
            decisions.append(AuditDecision.model_validate(payload))
        return tuple(sorted(decisions, key=_decision_sort_key))

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

    @computed_field(return_type=tuple[str, ...])
    @property
    def passed_audit_ids(self) -> tuple[str, ...]:
        selected = set(self.selected_audit_ids)
        evidence_counts = Counter(
            decision.audit_id for decision in self.decision_evidence
        )
        return tuple(
            sorted(
                decision.audit_id
                for decision in self.decision_evidence
                if (
                    decision.audit_id in selected
                    and evidence_counts[decision.audit_id] == 1
                    and decision.verdict == "pass"
                    and decision.all_checks_pass
                )
            )
        )

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
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
        evidence_ids = tuple(
            sorted(decision.audit_id for decision in self.decision_evidence)
        )
        return bool(
            self.selected_audit_ids
            and evidence_ids == self.selected_audit_ids
            and self.passed_audit_ids == self.selected_audit_ids
            and not any(issue_fields)
        )

    @model_validator(mode="after")
    def _validate_evidence(self):
        if len(set(self.selected_audit_ids)) != len(self.selected_audit_ids):
            raise ValueError("selected_audit_ids must be unique")
        if self.selected_audit_ids != tuple(sorted(self.selected_audit_ids)):
            raise ValueError("selected_audit_ids must be sorted")
        if any(type(decision) is not AuditDecision for decision in self.decision_evidence):
            raise ValueError("decision_evidence requires exact AuditDecision records")
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


@dataclass(frozen=True)
class _AuditIdObservation:
    index: int
    audit_id: str | None


def _observe_audit_id(value: Any, index: int) -> _AuditIdObservation:
    raw = _raw_fields(value)
    audit_id = raw.get("audit_id") if raw is not None else None
    if type(audit_id) is str and audit_id.strip():
        return _AuditIdObservation(index=index, audit_id=audit_id)
    return _AuditIdObservation(index=index, audit_id=None)


def _decision_sort_key(decision: AuditDecision) -> tuple[Any, ...]:
    return (
        decision.audit_id,
        decision.reviewer,
        decision.verdict,
        decision.answer_unique,
        decision.actions_correct,
        decision.roles_correct,
        decision.surface_natural,
        decision.notes,
    )


def _snapshot_record(value: Any, model_type: type):
    if type(value) is not model_type:
        return None
    raw = _raw_fields(value)
    if raw is None or set(raw) != set(model_type.model_fields):
        return None
    try:
        return model_type.model_validate(dict(raw))
    except Exception:
        return None


def _format_malformed_report_ids(
    audit_ids: set[str], positions: set[int]
) -> tuple[str, ...]:
    return tuple(sorted(audit_ids)) + tuple(
        f"<index:{index}>" for index in sorted(positions)
    )


def evaluate_audit_gate(
    selections: Sequence[Any], decisions: Sequence[Any]
) -> AuditGateReport:
    """Evaluate frozen, bounded snapshots of selections and human decisions."""
    selected_snapshot = _snapshot_input(selections, name="selections")
    decision_snapshot = _snapshot_input(decisions, name="decisions")

    valid_selections: list[AuditSelection] = []
    malformed_selection_ids: set[str] = set()
    malformed_selection_positions: set[int] = set()
    for index, item in enumerate(selected_snapshot):
        observation = _observe_audit_id(item, index)
        snapshot = _snapshot_record(item, AuditSelection)
        if snapshot is not None:
            valid_selections.append(snapshot)
        elif observation.audit_id is None:
            malformed_selection_positions.add(index)
        else:
            malformed_selection_ids.add(observation.audit_id)

    selection_counts = Counter(item.audit_id for item in valid_selections)
    selected_ids = tuple(sorted(selection_counts))
    selected_set = set(selected_ids)
    duplicate_selection_ids = tuple(
        sorted(audit_id for audit_id, count in selection_counts.items() if count > 1)
    )

    valid_decisions: list[AuditDecision] = []
    malformed_decision_ids: set[str] = set()
    malformed_decision_positions: set[int] = set()
    observed_decision_ids: list[str] = []
    for index, item in enumerate(decision_snapshot):
        observation = _observe_audit_id(item, index)
        if observation.audit_id is not None:
            observed_decision_ids.append(observation.audit_id)
        snapshot = _snapshot_record(item, AuditDecision)
        if snapshot is not None:
            valid_decisions.append(snapshot)
        elif observation.audit_id is None:
            malformed_decision_positions.add(index)
        else:
            malformed_decision_ids.add(observation.audit_id)

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
    return AuditGateReport(
        selected_audit_ids=selected_ids,
        decision_evidence=tuple(sorted(valid_decisions, key=_decision_sort_key)),
        malformed_selection_ids=_format_malformed_report_ids(
            malformed_selection_ids, malformed_selection_positions
        ),
        duplicate_selection_ids=duplicate_selection_ids,
        missing_audit_ids=missing_ids,
        duplicate_audit_ids=duplicate_ids,
        foreign_audit_ids=foreign_ids,
        malformed_decision_ids=_format_malformed_report_ids(
            malformed_decision_ids, malformed_decision_positions
        ),
        non_pass_audit_ids=non_pass_ids,
        failed_check_audit_ids=failed_check_ids,
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
