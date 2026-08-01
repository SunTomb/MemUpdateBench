"""Authenticated, deterministic 96-task audit selection for the canonical Pilot.

The reviewed condition policy deliberately includes only generator/config knobs:
Family A uses update depth and its three difficulty-indexed distractor/NOOP
counts; Family B uses update depth, active-object count, cross-slot density, and
interleaving pattern; Family C uses entity and attribute conditions; Family D
uses configured NOOP density and trap type. Every family additionally covers
train/dev/test, every authenticated difficulty, and surface variants 0/1/2.
Derived counters, IDs, hashes, indices, allocation/cardinality fields, and free
metadata are never condition tokens.

Within each family the selector first maximizes uncovered required tokens, then
prefers a fresh semantic core, underrepresented split/surface/difficulty, and a
stable task-id tie-break. It fills to 24 with the same spread priority and uses
a duplicate core only after fresh cores are exhausted. The release is
canonically snapshotted before calling validate_pilot_release, so selected
metadata is read only from the authenticated snapshot.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import ConfigDict, computed_field, field_validator, model_validator

from mub.vnext.audit.sample import AuditSelection
from mub.vnext.contracts import (
    Difficulty,
    MemUpdateTask,
    Split,
    TaskFamily,
    TaskManifest,
)
from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.generation.identity import stable_id
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation import validate_pilot_release


SELECTION_ALGORITHM = "pilot_audit_greedy_set_cover"
SELECTION_VERSION = "1"
_MAX_RELEASE_TASKS = 1440
_MAX_ISSUES = 128
_TASKS_PER_FAMILY = 24
_TOTAL_SELECTIONS = 96

_PILOT_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)

# Reviewed generator/config knobs only. Derived counters, indices, IDs, hashes,
# mapping IDs, allocation/cardinality fields, and free-form metadata are excluded.
FAMILY_CONDITION_POLICY: Mapping[
    TaskFamily, tuple[tuple[str, str], ...]
] = MappingProxyType(
    {
        TaskFamily.REPEATED_SAME_SLOT: (
            ("profile", "update_depth"),
            ("stratification", "same_name_distractor_count"),
            ("stratification", "same_entity_other_attribute_count"),
            ("stratification", "noop_count"),
        ),
        TaskFamily.INTERLEAVED_MULTI_SLOT: (
            ("profile", "update_depth"),
            ("stratification", "active_object_count"),
            ("stratification", "cross_slot_distractor_density"),
            ("stratification", "interleaving_pattern"),
        ),
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: (
            ("stratification", "entity_condition"),
            ("stratification", "attribute_condition"),
        ),
        TaskFamily.NOOP_WRITE_DISCIPLINE: (
            ("stratification", "configured_noop_density"),
            ("stratification", "trap_type"),
        ),
    }
)


class _StrictFrozenSelectionModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


class AuditSelectionIssue(_StrictFrozenSelectionModel):
    code: str
    message: str
    path: str

    @field_validator("code", "message", "path")
    @classmethod
    def _require_nonblank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value


class AuditFamilySelectionReport(_StrictFrozenSelectionModel):
    family: TaskFamily
    required_conditions: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    uncovered_required_conditions: tuple[str, ...] = ()
    impossible_reasons: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_canonical_family_string(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        candidate = dict(value)
        family = candidate.get("family")
        if type(family) is str:
            try:
                candidate["family"] = TaskFamily(family)
            except ValueError as exc:
                raise ValueError("family must be an exact TaskFamily value") from exc
        return candidate

    @field_validator(
        "required_conditions",
        "selected_task_ids",
        "uncovered_required_conditions",
        "impossible_reasons",
        mode="before",
    )
    @classmethod
    def _validate_text_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_text_tuple(value)


class AuditSelectionResult(_StrictFrozenSelectionModel):
    selection_algorithm: str
    selection_version: str
    selections: tuple[AuditSelection, ...] = ()
    family_reports: tuple[AuditFamilySelectionReport, ...] = ()
    uncovered_required_conditions: tuple[str, ...] = ()
    impossible_reasons: tuple[str, ...] = ()
    issues: tuple[AuditSelectionIssue, ...] = ()

    @field_validator("selection_algorithm", "selection_version")
    @classmethod
    def _require_nonblank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("uncovered_required_conditions", "impossible_reasons", mode="before")
    @classmethod
    def _validate_text_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_text_tuple(value)

    @field_validator("selections", mode="before")
    @classmethod
    def _snapshot_selections(cls, value: Any) -> tuple[AuditSelection, ...]:
        return _snapshot_model_tuple(value, AuditSelection, limit=_TOTAL_SELECTIONS)

    @field_validator("family_reports", mode="before")
    @classmethod
    def _snapshot_family_reports(
        cls, value: Any
    ) -> tuple[AuditFamilySelectionReport, ...]:
        reports = _snapshot_model_tuple(
            value,
            AuditFamilySelectionReport,
            limit=len(_PILOT_FAMILIES),
        )
        return tuple(sorted(reports, key=lambda item: _PILOT_FAMILIES.index(item.family)))

    @field_validator("issues", mode="before")
    @classmethod
    def _snapshot_issues(cls, value: Any) -> tuple[AuditSelectionIssue, ...]:
        issues = _snapshot_model_tuple(value, AuditSelectionIssue, limit=_MAX_ISSUES)
        return tuple(sorted(issues, key=lambda item: (item.code, item.path, item.message)))

    def _exact_field_snapshot(self) -> dict[str, Any]:
        raw = object.__getattribute__(self, "__dict__")
        if type(raw) is not dict or set(raw) != set(type(self).model_fields):
            raise ValueError("AuditSelectionResult requires intact fields")
        return dict(raw)

    def validated_replace(self, **changes) -> AuditSelectionResult:
        data = self._exact_field_snapshot()
        data.update(changes)
        return type(self).model_validate(data)

    @computed_field(return_type=bool)
    @property
    def valid(self) -> bool:
        try:
            snapshot = type(self).model_validate(self._exact_field_snapshot())
            return _valid_selection_snapshot(snapshot)
        except Exception:
            return False


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    family: TaskFamily
    difficulty: Difficulty
    split: Split
    semantic_core_id: str
    surface_variant: int
    conditions: tuple[str, ...]


def _normalized_text_tuple(value: Any) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("report text fields must be exact lists or tuples")
    if len(value) > _MAX_RELEASE_TASKS:
        raise ValueError("report text field exceeds the bounded selection limit")
    normalized: list[str] = []
    for item in value:
        if type(item) is not str or not item.strip():
            raise ValueError("report text fields require nonblank exact strings")
        normalized.append(item.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError("report text fields must be unique")
    return tuple(sorted(normalized))


def _snapshot_model_tuple(value: Any, model_type: type, *, limit: int) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("report record fields must be exact lists or tuples")
    if len(value) > limit:
        raise ValueError("report record field exceeds its bounded limit")
    snapshots = []
    for item in value:
        if type(item) is model_type:
            raw = object.__getattribute__(item, "__dict__")
            if type(raw) is not dict or set(raw) != set(model_type.model_fields):
                raise ValueError(f"report records require intact {model_type.__name__} models")
            payload = dict(raw)
        elif type(item) is dict:
            payload = item
        else:
            raise ValueError(
                f"report records require exact {model_type.__name__} models or dictionaries"
            )
        snapshots.append(model_type.model_validate(payload))
    return tuple(snapshots)


def _valid_selection_snapshot(result: AuditSelectionResult) -> bool:
    if result.selection_algorithm != SELECTION_ALGORITHM:
        return False
    if result.selection_version != SELECTION_VERSION:
        return False
    if result.issues or result.impossible_reasons or result.uncovered_required_conditions:
        return False
    if len(result.selections) != _TOTAL_SELECTIONS:
        return False
    if len({item.task_id for item in result.selections}) != _TOTAL_SELECTIONS:
        return False
    if len({item.audit_id for item in result.selections}) != _TOTAL_SELECTIONS:
        return False
    if tuple(report.family for report in result.family_reports) != _PILOT_FAMILIES:
        return False

    by_family: dict[TaskFamily, list[AuditSelection]] = {
        family: [] for family in _PILOT_FAMILIES
    }
    for selection in result.selections:
        if selection.family not in by_family:
            return False
        if selection.selection_reason not in {"greedy_set_cover", "spread_fill"}:
            return False
        expected_id = audit_selection_id(
            task_id=selection.task_id,
            family=selection.family,
            difficulty=selection.difficulty,
            split=selection.split,
            covered_conditions=selection.covered_conditions,
            selection_reason=selection.selection_reason,
            selection_algorithm=result.selection_algorithm,
            selection_version=result.selection_version,
        )
        if selection.audit_id != expected_id:
            return False
        by_family[selection.family].append(selection)

    for report in result.family_reports:
        selections = by_family[report.family]
        if len(selections) != _TASKS_PER_FAMILY:
            return False
        if report.uncovered_required_conditions or report.impossible_reasons:
            return False
        if report.selected_task_ids != tuple(sorted(item.task_id for item in selections)):
            return False
        covered = {
            token
            for selection in selections
            for token in selection.covered_conditions
        }
        if set(report.required_conditions) != covered:
            return False
    return True


def audit_selection_id(
    *,
    task_id: str,
    family: TaskFamily,
    difficulty: Difficulty,
    split: Split,
    covered_conditions: Sequence[str],
    selection_reason: str,
    selection_algorithm: str = SELECTION_ALGORITHM,
    selection_version: str = SELECTION_VERSION,
) -> str:
    """Bind every selection-contract field into one canonical stable audit ID."""
    if type(task_id) is not str or not task_id.strip():
        raise ValueError("task_id must be a nonblank exact string")
    if type(family) is not TaskFamily:
        raise TypeError("family must be an exact TaskFamily")
    if type(difficulty) is not Difficulty:
        raise TypeError("difficulty must be an exact Difficulty")
    if type(split) is not Split:
        raise TypeError("split must be an exact Split")
    conditions = _normalized_text_tuple(covered_conditions)
    for name, value in (
        ("selection_reason", selection_reason),
        ("selection_algorithm", selection_algorithm),
        ("selection_version", selection_version),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be a nonblank exact string")
    return stable_id(
        "audit",
        {
            "task_id": task_id,
            "family": family.value,
            "difficulty": difficulty.value,
            "split": split.value,
            "covered_conditions": list(conditions),
            "selection_algorithm": selection_algorithm,
            "selection_version": selection_version,
            "selection_reason": selection_reason,
        },
    )


def _issue(code: str, message: str, path: str) -> AuditSelectionIssue:
    return AuditSelectionIssue(code=code, message=message, path=path)


def _invalid_result(issues: Sequence[AuditSelectionIssue]) -> AuditSelectionResult:
    bounded = tuple(
        sorted(
            {
                (issue.code, issue.path, issue.message): issue
                for issue in issues
            }.values(),
            key=lambda issue: (issue.code, issue.path, issue.message),
        )[:_MAX_ISSUES]
    )
    return AuditSelectionResult(
        selection_algorithm=SELECTION_ALGORITHM,
        selection_version=SELECTION_VERSION,
        issues=bounded,
    )


def _bounded_tasks(tasks: Any) -> tuple[tuple[Any, ...] | None, AuditSelectionIssue | None]:
    copied: list[Any] = []
    try:
        iterator = iter(tasks)
        for index in range(_MAX_RELEASE_TASKS + 1):
            try:
                item = next(iterator)
            except StopIteration:
                break
            if index == _MAX_RELEASE_TASKS:
                return None, _issue(
                    "audit_selection_input_size_limit",
                    "tasks iterable exceeds the 1,440-task Pilot release limit",
                    "tasks",
                )
            copied.append(item)
    except Exception:
        return None, _issue(
            "audit_selection_malformed_tasks_iterable",
            "tasks must be a finite iterable of Pilot task models",
            "tasks",
        )
    return tuple(copied), None


def _task_diagnostic_label(value: Any) -> str:
    if type(value) is MemUpdateTask:
        try:
            raw = object.__getattribute__(value, "__dict__")
        except Exception:
            raw = None
        if type(raw) is dict:
            task_id = raw.get("task_id")
            if type(task_id) is str and task_id.strip():
                return task_id.strip()
            family = raw.get("task_family")
            if type(family) is str and family.strip():
                return f"<malformed-{family.strip()}>"
        return "<malformed-task>"
    value_type = type(value)
    return f"<invalid-{value_type.__module__}.{value_type.__qualname__}>"


def _task_snapshot_sort_key(value: Any) -> tuple[int, str]:
    return (0 if type(value) is MemUpdateTask else 1, _task_diagnostic_label(value))


def _snapshot_exact_model(value: Any, model_type: type, path: str):
    if type(value) is not model_type:
        raise TypeError(f"{path} must be an exact {model_type.__name__}")
    raw = object.__getattribute__(value, "__dict__")
    if type(raw) is not dict or set(raw) != set(model_type.model_fields):
        raise ValueError(f"{path} must be an intact {model_type.__name__}")
    serialized = canonical_json_bytes(value)
    return model_type.model_validate_json(serialized)


def _snapshot_release(
    tasks: Any, manifest: Any
) -> tuple[tuple[MemUpdateTask, ...] | None, TaskManifest | None, tuple[AuditSelectionIssue, ...]]:
    copied, iterable_issue = _bounded_tasks(tasks)
    if iterable_issue is not None or copied is None:
        return None, None, (iterable_issue,) if iterable_issue is not None else ()
    try:
        manifest_snapshot = _snapshot_exact_model(manifest, TaskManifest, "manifest")
    except Exception:
        return None, None, (
            _issue(
                "audit_selection_malformed_manifest_snapshot",
                "manifest could not be canonically snapshotted as an exact TaskManifest",
                "manifest",
            ),
        )

    snapshots: list[MemUpdateTask] = []
    for task in sorted(copied, key=_task_snapshot_sort_key):
        label = _task_diagnostic_label(task)
        try:
            snapshots.append(
                _snapshot_exact_model(task, MemUpdateTask, f"tasks.{label}")
            )
        except Exception:
            return None, None, (
                _issue(
                    "audit_selection_malformed_task_snapshot",
                    "a task could not be canonically snapshotted as an exact MemUpdateTask",
                    f"tasks.{label}",
                ),
            )
    return tuple(snapshots), manifest_snapshot, ()


def _canonical_scalar(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _condition_token(label: str, value: Any) -> str:
    return f"{label}={_canonical_scalar(value)}"


def _candidate(task: MemUpdateTask) -> _Candidate:
    family = TaskFamily(task.task_family)
    difficulty = task.difficulty
    split = task.metadata.split
    extra = task.metadata.extra
    profile = task.metadata.resolved_profile
    stratification = extra["stratification"]
    surface_variant = extra["surface_variant"]
    conditions = {
        _condition_token("split", split.value),
        _condition_token("difficulty", difficulty.value),
        _condition_token("surface_variant", surface_variant),
    }
    sources = {"profile": profile, "stratification": stratification}
    for source, key in FAMILY_CONDITION_POLICY[family]:
        conditions.add(_condition_token(f"{source}.{key}", sources[source][key]))
    return _Candidate(
        task_id=task.task_id,
        family=family,
        difficulty=difficulty,
        split=split,
        semantic_core_id=task.metadata.split_key.semantic_core_id,
        surface_variant=surface_variant,
        conditions=tuple(sorted(conditions)),
    )


def _selection_sort_key(selection: AuditSelection) -> tuple[int, str]:
    return (_PILOT_FAMILIES.index(selection.family), selection.task_id)


def _exact_condition_cover(
    candidates: tuple[_Candidate, ...],
    required: set[str],
    budget: int,
) -> tuple[_Candidate, ...] | None:
    """Return a stable complete cover when the greedy pass reaches a dead end."""
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.task_id))
    condition_sets = {
        candidate.task_id: frozenset(candidate.conditions) for candidate in ordered
    }
    by_condition = {
        condition: tuple(
            candidate
            for candidate in ordered
            if condition in condition_sets[candidate.task_id]
        )
        for condition in sorted(required)
    }
    if any(not options for options in by_condition.values()):
        return None

    failed: set[tuple[frozenset[str], int]] = set()

    def search(
        remaining: frozenset[str], slots: int
    ) -> tuple[_Candidate, ...] | None:
        if not remaining:
            return ()
        if slots <= 0:
            return None
        state = (remaining, slots)
        if state in failed:
            return None
        max_gain = max(
            len(condition_sets[candidate.task_id] & remaining)
            for candidate in ordered
        )
        if max_gain == 0 or (len(remaining) + max_gain - 1) // max_gain > slots:
            failed.add(state)
            return None
        pivot = min(
            remaining,
            key=lambda condition: (len(by_condition[condition]), condition),
        )
        options = sorted(
            by_condition[pivot],
            key=lambda candidate: (
                -len(condition_sets[candidate.task_id] & remaining),
                candidate.task_id,
            ),
        )
        for candidate in options:
            candidate_conditions = condition_sets[candidate.task_id]
            tail = search(remaining - candidate_conditions, slots - 1)
            if tail is not None:
                return (candidate, *tail)
        failed.add(state)
        return None

    return search(frozenset(required), budget)


def _select_family(
    family: TaskFamily,
    candidates: tuple[_Candidate, ...],
) -> tuple[
    tuple[AuditSelection, ...],
    AuditFamilySelectionReport,
    tuple[AuditSelectionIssue, ...],
]:
    required = {
        condition for candidate in candidates for condition in candidate.conditions
    }
    uncovered = set(required)
    selected: list[tuple[_Candidate, str]] = []
    selected_ids: set[str] = set()
    selected_cores: set[str] = set()
    split_counts: Counter[Split] = Counter()
    surface_counts: Counter[int] = Counter()
    difficulty_counts: Counter[Difficulty] = Counter()
    condition_counts: Counter[str] = Counter()

    def add(candidate: _Candidate, reason: str) -> None:
        selected.append((candidate, reason))
        selected_ids.add(candidate.task_id)
        selected_cores.add(candidate.semantic_core_id)
        split_counts[candidate.split] += 1
        surface_counts[candidate.surface_variant] += 1
        difficulty_counts[candidate.difficulty] += 1
        condition_counts.update(candidate.conditions)
        uncovered.difference_update(candidate.conditions)

    while uncovered and len(selected) < _TASKS_PER_FAMILY:
        eligible = [candidate for candidate in candidates if candidate.task_id not in selected_ids]
        if not eligible:
            break
        best = min(
            eligible,
            key=lambda candidate: (
                -len(set(candidate.conditions) & uncovered),
                -(candidate.semantic_core_id not in selected_cores),
                split_counts[candidate.split],
                surface_counts[candidate.surface_variant],
                difficulty_counts[candidate.difficulty],
                candidate.task_id,
            ),
        )
        if not (set(best.conditions) & uncovered):
            break
        add(best, "greedy_set_cover")

    if uncovered:
        exact_cover = _exact_condition_cover(
            candidates,
            required,
            _TASKS_PER_FAMILY,
        )
        if exact_cover is not None:
            selected.clear()
            selected_ids.clear()
            selected_cores.clear()
            split_counts.clear()
            surface_counts.clear()
            difficulty_counts.clear()
            condition_counts.clear()
            uncovered = set(required)
            for candidate in exact_cover:
                add(candidate, "greedy_set_cover")

    while len(selected) < _TASKS_PER_FAMILY:
        eligible = [candidate for candidate in candidates if candidate.task_id not in selected_ids]
        if not eligible:
            break
        fresh_core_candidates = [
            candidate
            for candidate in eligible
            if candidate.semantic_core_id not in selected_cores
        ]
        if fresh_core_candidates:
            eligible = fresh_core_candidates
        best = min(
            eligible,
            key=lambda candidate: (
                split_counts[candidate.split],
                surface_counts[candidate.surface_variant],
                difficulty_counts[candidate.difficulty],
                sum(condition_counts[token] for token in candidate.conditions),
                candidate.task_id,
            ),
        )
        add(best, "spread_fill")

    selections = []
    for candidate, reason in selected:
        audit_id = audit_selection_id(
            task_id=candidate.task_id,
            family=candidate.family,
            difficulty=candidate.difficulty,
            split=candidate.split,
            covered_conditions=candidate.conditions,
            selection_reason=reason,
        )
        selections.append(
            AuditSelection(
                audit_id=audit_id,
                task_id=candidate.task_id,
                family=candidate.family,
                difficulty=candidate.difficulty,
                split=candidate.split,
                covered_conditions=candidate.conditions,
                selection_reason=reason,
            )
        )

    reasons: list[str] = []
    issues: list[AuditSelectionIssue] = []
    if uncovered:
        reasons.append(
            f"{family.value} cannot cover {len(uncovered)} required condition tokens within "
            f"the {_TASKS_PER_FAMILY}-task family budget"
        )
        issues.append(
            _issue(
                "audit_selection_impossible_cover",
                reasons[-1],
                f"families.{family.value}.required_conditions",
            )
        )
    if len(selections) != _TASKS_PER_FAMILY:
        reasons.append(
            f"{family.value} produced {len(selections)} selections instead of "
            f"{_TASKS_PER_FAMILY}"
        )
        issues.append(
            _issue(
                "audit_selection_family_count_mismatch",
                reasons[-1],
                f"families.{family.value}.selections",
            )
        )
    report = AuditFamilySelectionReport(
        family=family,
        required_conditions=tuple(required),
        selected_task_ids=tuple(sorted(item.task_id for item in selections)),
        uncovered_required_conditions=tuple(uncovered),
        impossible_reasons=tuple(reasons),
    )
    return tuple(sorted(selections, key=lambda item: item.task_id)), report, tuple(issues)


def select_pilot_audit_sample(tasks: Iterable[Any], manifest: Any) -> AuditSelectionResult:
    """Authenticate and select the canonical deterministic 96-task Pilot audit sample.

    Selection first greedily maximizes uncovered reviewed condition tokens, then
    prefers a new semantic core, underrepresented split/surface/difficulty, and
    finally the lexicographically smallest task ID. The fill phase preserves the
    same spread, always exhausting fresh semantic cores before any duplicate core.
    """
    snapshots, manifest_snapshot, snapshot_issues = _snapshot_release(tasks, manifest)
    if snapshot_issues or snapshots is None or manifest_snapshot is None:
        return _invalid_result(snapshot_issues)

    try:
        release_report = validate_pilot_release(snapshots, manifest_snapshot)
    except Exception:
        return _invalid_result(
            (
                _issue(
                    "audit_selection_release_validator_exception",
                    "canonical Pilot release authentication raised an exception",
                    "release",
                ),
            )
        )
    if not release_report.valid:
        return _invalid_result(
            tuple(
                _issue(issue.code, issue.message, issue.path)
                for issue in release_report.issues[:_MAX_ISSUES]
            )
        )

    try:
        candidates = tuple(
            sorted(
                (_candidate(task) for task in snapshots),
                key=lambda item: item.task_id,
            )
        )
        all_selections: list[AuditSelection] = []
        family_reports: list[AuditFamilySelectionReport] = []
        issues: list[AuditSelectionIssue] = []
        for family in _PILOT_FAMILIES:
            family_candidates = tuple(
                candidate for candidate in candidates if candidate.family is family
            )
            selections, family_report, family_issues = _select_family(
                family, family_candidates
            )
            all_selections.extend(selections)
            family_reports.append(family_report)
            issues.extend(family_issues)
    except Exception:
        return _invalid_result(
            (
                _issue(
                    "audit_selection_authenticated_projection_error",
                    "authenticated Pilot models could not be projected into audit conditions",
                    "release.tasks",
                ),
            )
        )

    uncovered = tuple(
        sorted(
            {
                token
                for report in family_reports
                for token in report.uncovered_required_conditions
            }
        )
    )
    impossible = tuple(
        sorted(
            {
                reason
                for report in family_reports
                for reason in report.impossible_reasons
            }
        )
    )
    result = AuditSelectionResult(
        selection_algorithm=SELECTION_ALGORITHM,
        selection_version=SELECTION_VERSION,
        selections=tuple(sorted(all_selections, key=_selection_sort_key)),
        family_reports=tuple(family_reports),
        uncovered_required_conditions=uncovered,
        impossible_reasons=impossible,
        issues=tuple(issues),
    )
    if result.valid:
        return result
    if result.issues:
        return result
    return result.validated_replace(
        issues=(
            _issue(
                "audit_selection_invalid_result",
                "selection output failed the exact 96-task/24-per-family validity contract",
                "selection",
            ),
        )
    )


__all__ = [
    "AuditFamilySelectionReport",
    "AuditSelectionIssue",
    "AuditSelectionResult",
    "FAMILY_CONDITION_POLICY",
    "SELECTION_ALGORITHM",
    "SELECTION_VERSION",
    "audit_selection_id",
    "select_pilot_audit_sample",
]
