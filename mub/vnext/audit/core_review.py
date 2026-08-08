"""Strict-v3 Core human review, adjudication, and release gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict, computed_field, field_validator, model_validator

from mub.vnext.audit.core import (
    CORE_AUDIT_FAMILIES,
    CORE_AUDIT_SCHEMA_VERSION,
    CoreAuditSelection,
    CoreAuditSelectionPackage,
    core_audit_selection_hash,
)
from mub.vnext.contracts import TaskFamily
from mub.vnext.contracts.common import ImmutableContractModel


ReviewOutcome = Literal["pass", "fail", "not_applicable"]
ReviewVerdict = Literal["pass", "block", "needs_revision"]
ReviewerRole = Literal["primary", "secondary", "adjudicator"]
_CHECK_FIELDS = (
    "answer_unique",
    "gold_actions_scope_correct",
    "event_roles_correct",
    "selector_history_evidence_correct",
    "four_surface_semantic_equivalence",
    "surface_natural",
)
_DUAL_REVIEW_FAMILIES = frozenset(
    {
        TaskFamily.DELETION_FORGETTING,
        TaskFamily.CURRENT_HISTORICAL_QUERY,
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
    }
)
_SELECTOR_REVIEW_FAMILIES = frozenset(
    {
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
        TaskFamily.DELETION_FORGETTING,
        TaskFamily.CURRENT_HISTORICAL_QUERY,
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
    }
)
_MAX_REVIEW_RECORDS = 1024


class _StrictFrozenCoreReviewModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


def _hash_value(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be lowercase sha256")
    return value


def applicable_core_audit_checks(family: TaskFamily) -> frozenset[str]:
    """Return the exact reviewed applicability policy for one Core family."""
    if type(family) is not TaskFamily or family not in CORE_AUDIT_FAMILIES:
        raise ValueError("family must be one of Core A-G")
    applicable = set(_CHECK_FIELDS)
    if family not in _SELECTOR_REVIEW_FAMILIES:
        applicable.remove("selector_history_evidence_correct")
    return frozenset(applicable)


class CoreAuditChecks(_StrictFrozenCoreReviewModel):
    answer_unique: ReviewOutcome
    gold_actions_scope_correct: ReviewOutcome
    event_roles_correct: ReviewOutcome
    selector_history_evidence_correct: ReviewOutcome
    four_surface_semantic_equivalence: ReviewOutcome
    surface_natural: ReviewOutcome

    @property
    def all_applicable_pass(self) -> bool:
        return all(value != "fail" for value in self.values())

    def values(self) -> tuple[ReviewOutcome, ...]:
        return tuple(getattr(self, field) for field in _CHECK_FIELDS)


class CoreAuditDecision(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    audit_id: str
    task_id: str
    task_hash: str
    source_task_manifest_hash: str
    selection_hash: str
    reviewer_id: str
    reviewer_role: ReviewerRole
    review_record_id: str
    independent_review_attestation: bool
    verdict: ReviewVerdict
    checks: CoreAuditChecks
    task_specific_observation: str
    notes: str

    @field_validator(
        "audit_id",
        "task_id",
        "reviewer_id",
        "review_record_id",
        "task_specific_observation",
    )
    @classmethod
    def _nonblank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("task_hash", "source_task_manifest_hash", "selection_hash")
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @model_validator(mode="after")
    def _attested(self) -> CoreAuditDecision:
        if self.independent_review_attestation is not True:
            raise ValueError("human review requires an explicit independence attestation")
        return self

    @property
    def terminal_pass(self) -> bool:
        return self.verdict == "pass" and self.checks.all_applicable_pass


class CoreAuditDecisionTemplate(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    audit_id: str
    task_id: str
    task_hash: str
    source_task_manifest_hash: str
    selection_hash: str
    reviewer_role: ReviewerRole
    reviewer_id: None = None
    review_record_id: None = None
    independent_review_attestation: None = None
    verdict: None = None
    checks: None = None
    task_specific_observation: None = None
    notes: None = None

    @field_validator("task_hash", "source_task_manifest_hash", "selection_hash")
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
        return False


class CoreAuditGateReport(_StrictFrozenCoreReviewModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    source_task_manifest_hash: str
    selection_hash: str
    decision_evidence: tuple[CoreAuditDecision, ...] = ()
    adjudication_evidence: tuple[CoreAuditDecision, ...] = ()
    missing_review_roles: tuple[str, ...] = ()
    duplicate_review_roles: tuple[str, ...] = ()
    unknown_audit_ids: tuple[str, ...] = ()
    binding_mismatch_audit_ids: tuple[str, ...] = ()
    malformed_review_records: tuple[str, ...] = ()
    duplicate_review_record_ids: tuple[str, ...] = ()
    copied_observation_audit_ids: tuple[str, ...] = ()
    non_independent_audit_ids: tuple[str, ...] = ()
    invalid_applicability_audit_ids: tuple[str, ...] = ()
    required_adjudication_ids: tuple[str, ...] = ()
    unresolved_adjudication_ids: tuple[str, ...] = ()
    adjudicated_audit_ids: tuple[str, ...] = ()
    non_pass_terminal_audit_ids: tuple[str, ...] = ()
    terminal_pass_audit_ids: tuple[str, ...] = ()
    agreement_item_count: int = 0
    raw_agreement: float | None = None
    cohens_kappa: float | None = None
    issues: tuple[str, ...] = ()

    @field_validator("source_task_manifest_hash", "selection_hash")
    @classmethod
    def _hash(cls, value: str, info) -> str:
        return _hash_value(value, info.field_name)

    @field_validator("decision_evidence", "adjudication_evidence", mode="before")
    @classmethod
    def _evidence(cls, value: Any) -> tuple[CoreAuditDecision, ...]:
        if type(value) not in (list, tuple) or len(value) > _MAX_REVIEW_RECORDS:
            raise ValueError("review evidence must be a bounded list or tuple")
        result = []
        for item in value:
            if type(item) is not CoreAuditDecision:
                raise ValueError("gate evidence requires exact CoreAuditDecision records")
            raw = object.__getattribute__(item, "__dict__")
            if type(raw) is not dict or set(raw) != set(CoreAuditDecision.model_fields):
                raise ValueError("gate evidence requires intact CoreAuditDecision records")
            result.append(CoreAuditDecision.model_validate(dict(raw)))
        return tuple(sorted(result, key=_decision_key))

    @field_validator(
        "missing_review_roles",
        "duplicate_review_roles",
        "unknown_audit_ids",
        "binding_mismatch_audit_ids",
        "malformed_review_records",
        "duplicate_review_record_ids",
        "copied_observation_audit_ids",
        "non_independent_audit_ids",
        "invalid_applicability_audit_ids",
        "required_adjudication_ids",
        "unresolved_adjudication_ids",
        "adjudicated_audit_ids",
        "non_pass_terminal_audit_ids",
        "terminal_pass_audit_ids",
        "issues",
        mode="before",
    )
    @classmethod
    def _sorted_text(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("gate issue fields must be lists or tuples")
        result = tuple(sorted(value))
        if any(type(item) is not str or not item.strip() for item in result):
            raise ValueError("gate issue fields require nonblank strings")
        if len(result) != len(set(result)):
            raise ValueError("gate issue fields must be unique")
        return result

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
        try:
            raw = object.__getattribute__(self, "__dict__")
            snapshot = type(self).model_validate(dict(raw))
        except Exception:
            return False
        return bool(
            len(snapshot.terminal_pass_audit_ids) == 224
            and not snapshot.issues
            and not snapshot.unresolved_adjudication_ids
            and not snapshot.non_pass_terminal_audit_ids
        )


def _decision_key(item: CoreAuditDecision) -> tuple[str, str, str, str]:
    return (
        item.audit_id,
        item.reviewer_role,
        item.reviewer_id,
        item.review_record_id,
    )


def _expected_roles(selection: CoreAuditSelection) -> tuple[str, ...]:
    if selection.family in _DUAL_REVIEW_FAMILIES:
        return ("primary", "secondary")
    return ("primary",)


def core_audit_decision_templates(
    package: CoreAuditSelectionPackage,
) -> tuple[CoreAuditDecisionTemplate, ...]:
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    if core_audit_selection_hash(package) != package.selection_hash:
        raise ValueError("selection package hash mismatch")
    templates = []
    for selected in package.selections:
        for role in _expected_roles(selected):
            templates.append(
                CoreAuditDecisionTemplate(
                    audit_id=selected.audit_id,
                    task_id=selected.task_id,
                    task_hash=selected.task_hash,
                    source_task_manifest_hash=package.source_task_manifest_hash,
                    selection_hash=package.selection_hash,
                    reviewer_role=role,
                )
            )
    return tuple(sorted(templates, key=lambda item: (item.audit_id, item.reviewer_role)))


def _snapshot_sequence(value: Sequence[Any], name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be an exact list or tuple")
    if len(value) > _MAX_REVIEW_RECORDS:
        raise ValueError(f"{name} exceeds the bounded review limit")
    return tuple(value)


def _is_bound(
    decision: CoreAuditDecision,
    selected: CoreAuditSelection,
    package: CoreAuditSelectionPackage,
) -> bool:
    return (
        decision.task_id == selected.task_id
        and decision.task_hash == selected.task_hash
        and decision.source_task_manifest_hash == package.source_task_manifest_hash
        and decision.selection_hash == package.selection_hash
    )


def _valid_applicability(
    decision: CoreAuditDecision, family: TaskFamily
) -> bool:
    applicable = applicable_core_audit_checks(family)
    return all(
        (getattr(decision.checks, field) != "not_applicable")
        if field in applicable
        else (getattr(decision.checks, field) == "not_applicable")
        for field in _CHECK_FIELDS
    )


def _human_pass(decision: CoreAuditDecision, family: TaskFamily) -> bool:
    if not _valid_applicability(decision, family):
        return False
    applicable = applicable_core_audit_checks(family)
    return decision.verdict == "pass" and all(
        getattr(decision.checks, field) == "pass" for field in applicable
    )


def _review_signature(decision: CoreAuditDecision, family: TaskFamily) -> tuple[str, ...]:
    applicable = applicable_core_audit_checks(family)
    return (
        decision.verdict,
        *(getattr(decision.checks, field) for field in _CHECK_FIELDS if field in applicable),
    )


def _agreement(
    package: CoreAuditSelectionPackage,
    by_role: dict[tuple[str, str], list[CoreAuditDecision]],
) -> tuple[int, float | None, float | None]:
    first: list[str] = []
    second: list[str] = []
    for selected in package.selections:
        if selected.family not in _DUAL_REVIEW_FAMILIES:
            continue
        primary = by_role.get((selected.audit_id, "primary"), [])
        secondary = by_role.get((selected.audit_id, "secondary"), [])
        if len(primary) != 1 or len(secondary) != 1:
            continue
        p_signature = _review_signature(primary[0], selected.family)
        s_signature = _review_signature(secondary[0], selected.family)
        first.extend(p_signature)
        second.extend(s_signature)
    count = len(first)
    if count == 0:
        return 0, None, None
    observed = sum(a == b for a, b in zip(first, second)) / count
    first_counts = Counter(first)
    second_counts = Counter(second)
    categories = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[value] / count) * (second_counts[value] / count)
        for value in categories
    )
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return count, observed, kappa


def evaluate_core_audit_gate(
    package: CoreAuditSelectionPackage,
    decisions: Sequence[Any],
    adjudications: Sequence[Any],
) -> CoreAuditGateReport:
    """Fail-closed gate over bound primary/secondary and adjudication records."""
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    if core_audit_selection_hash(package) != package.selection_hash:
        raise ValueError("selection package hash mismatch")
    decision_input = _snapshot_sequence(decisions, "decisions")
    adjudication_input = _snapshot_sequence(adjudications, "adjudications")
    selected_by_id = {item.audit_id: item for item in package.selections}
    malformed: set[str] = set()
    unknown: set[str] = set()
    binding: set[str] = set()
    invalid_applicability: set[str] = set()
    valid_decisions: list[CoreAuditDecision] = []
    valid_adjudications: list[CoreAuditDecision] = []

    def collect(
        values: tuple[Any, ...], expected_adjudication: bool
    ) -> list[CoreAuditDecision]:
        result = []
        for index, item in enumerate(values):
            if type(item) is not CoreAuditDecision:
                malformed.add(f"<index:{index}:{'adjudication' if expected_adjudication else 'decision'}>")
                continue
            try:
                raw = object.__getattribute__(item, "__dict__")
                snapshot = CoreAuditDecision.model_validate(dict(raw))
            except Exception:
                malformed.add(getattr(item, "audit_id", f"<index:{index}>") or f"<index:{index}>")
                continue
            selected = selected_by_id.get(snapshot.audit_id)
            if selected is None:
                unknown.add(snapshot.audit_id)
                continue
            if not _is_bound(snapshot, selected, package):
                binding.add(snapshot.audit_id)
                continue
            if (snapshot.reviewer_role == "adjudicator") != expected_adjudication:
                malformed.add(snapshot.audit_id)
                continue
            if not _valid_applicability(snapshot, selected.family):
                invalid_applicability.add(snapshot.audit_id)
            result.append(snapshot)
        return result

    valid_decisions = collect(decision_input, False)
    valid_adjudications = collect(adjudication_input, True)
    by_role: dict[tuple[str, str], list[CoreAuditDecision]] = defaultdict(list)
    for item in valid_decisions:
        by_role[(item.audit_id, item.reviewer_role)].append(item)
    adjudication_by_id: dict[str, list[CoreAuditDecision]] = defaultdict(list)
    for item in valid_adjudications:
        adjudication_by_id[item.audit_id].append(item)

    missing = set()
    duplicates = set()
    for selected in package.selections:
        for role in _expected_roles(selected):
            rows = by_role.get((selected.audit_id, role), [])
            label = f"{selected.audit_id}:{role}"
            if not rows:
                missing.add(label)
            elif len(rows) > 1:
                duplicates.add(label)
        unexpected_roles = {
            role
            for (audit_id, role), rows in by_role.items()
            if audit_id == selected.audit_id and role not in _expected_roles(selected) and rows
        }
        duplicates.update(f"{selected.audit_id}:{role}" for role in unexpected_roles)

    record_counts = Counter(
        item.review_record_id for item in (*valid_decisions, *valid_adjudications)
    )
    duplicate_records = {
        record_id for record_id, count in record_counts.items() if count > 1
    }
    copied_observations: set[str] = set()
    observation_groups: dict[tuple[str, str], list[CoreAuditDecision]] = defaultdict(list)
    for item in (*valid_decisions, *valid_adjudications):
        normalized = " ".join(item.task_specific_observation.casefold().split())
        observation_groups[(item.reviewer_id, normalized)].append(item)
    for rows in observation_groups.values():
        if len(rows) > 1:
            copied_observations.update(item.audit_id for item in rows)

    non_independent: set[str] = set()
    required_adjudication: set[str] = set()
    terminal_pass: set[str] = set()
    non_pass_terminal: set[str] = set()
    adjudicated: set[str] = set()
    for selected in package.selections:
        expected = _expected_roles(selected)
        rows = [
            by_role[(selected.audit_id, role)][0]
            for role in expected
            if len(by_role.get((selected.audit_id, role), [])) == 1
        ]
        if len(rows) == 2 and rows[0].reviewer_id == rows[1].reviewer_id:
            non_independent.add(selected.audit_id)
        complete = len(rows) == len(expected)
        clean_rows = complete and selected.audit_id not in invalid_applicability
        disagreement = (
            len(rows) == 2
            and _review_signature(rows[0], selected.family)
            != _review_signature(rows[1], selected.family)
        )
        any_nonpass = bool(rows) and any(
            not _human_pass(item, selected.family) for item in rows
        )
        needs_adjudication = complete and clean_rows and (disagreement or any_nonpass)
        if needs_adjudication:
            required_adjudication.add(selected.audit_id)
        adjudication_rows = adjudication_by_id.get(selected.audit_id, [])
        if needs_adjudication:
            if len(adjudication_rows) == 1:
                adjudicator = adjudication_rows[0]
                if adjudicator.reviewer_id in {item.reviewer_id for item in rows}:
                    non_independent.add(selected.audit_id)
                elif _human_pass(adjudicator, selected.family):
                    adjudicated.add(selected.audit_id)
                    terminal_pass.add(selected.audit_id)
                else:
                    non_pass_terminal.add(selected.audit_id)
            elif len(adjudication_rows) > 1:
                duplicates.add(f"{selected.audit_id}:adjudicator")
        elif adjudication_rows:
            duplicates.add(f"{selected.audit_id}:unexpected_adjudicator")
        elif complete and clean_rows and all(
            _human_pass(item, selected.family) for item in rows
        ):
            terminal_pass.add(selected.audit_id)
        elif complete:
            non_pass_terminal.add(selected.audit_id)

    unresolved = required_adjudication - adjudicated
    agreement_count, raw_agreement, kappa = _agreement(package, by_role)
    issue_sets = {
        "missing_review_roles": missing,
        "duplicate_review_roles": duplicates,
        "unknown_audit_ids": unknown,
        "binding_mismatch_audit_ids": binding,
        "malformed_review_records": malformed,
        "duplicate_review_record_ids": duplicate_records,
        "copied_observation_audit_ids": copied_observations,
        "non_independent_audit_ids": non_independent,
        "invalid_applicability_audit_ids": invalid_applicability,
        "unresolved_adjudication_ids": unresolved,
        "non_pass_terminal_audit_ids": non_pass_terminal,
    }
    issues = tuple(
        f"{name}:{value}"
        for name, values in issue_sets.items()
        for value in sorted(values)
    )
    return CoreAuditGateReport(
        source_task_manifest_hash=package.source_task_manifest_hash,
        selection_hash=package.selection_hash,
        decision_evidence=tuple(valid_decisions),
        adjudication_evidence=tuple(valid_adjudications),
        missing_review_roles=tuple(missing),
        duplicate_review_roles=tuple(duplicates),
        unknown_audit_ids=tuple(unknown),
        binding_mismatch_audit_ids=tuple(binding),
        malformed_review_records=tuple(malformed),
        duplicate_review_record_ids=tuple(duplicate_records),
        copied_observation_audit_ids=tuple(copied_observations),
        non_independent_audit_ids=tuple(non_independent),
        invalid_applicability_audit_ids=tuple(invalid_applicability),
        required_adjudication_ids=tuple(required_adjudication),
        unresolved_adjudication_ids=tuple(unresolved),
        adjudicated_audit_ids=tuple(adjudicated),
        non_pass_terminal_audit_ids=tuple(non_pass_terminal),
        terminal_pass_audit_ids=tuple(terminal_pass),
        agreement_item_count=agreement_count,
        raw_agreement=raw_agreement,
        cohens_kappa=kappa,
        issues=issues,
    )


__all__ = [
    "CoreAuditChecks",
    "CoreAuditDecision",
    "CoreAuditDecisionTemplate",
    "CoreAuditGateReport",
    "ReviewOutcome",
    "ReviewVerdict",
    "ReviewerRole",
    "applicable_core_audit_checks",
    "core_audit_decision_templates",
    "evaluate_core_audit_gate",
]
