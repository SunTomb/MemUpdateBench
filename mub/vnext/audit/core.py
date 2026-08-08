"""Strict-v3 authenticated Core human-audit selection contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field, field_validator, model_validator

from mub.vnext.contracts import Difficulty, Split, TaskFamily
from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.core_config import (
    CORE_DIFFICULTIES,
    CORE_FAMILY_A_CONDITIONS,
    CORE_FAMILY_A_DEPTHS,
    CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS,
    CORE_FAMILY_B_DEPTHS,
    CORE_FAMILY_B_INTERLEAVING_PATTERNS,
    CORE_FAMILY_C_ATTRIBUTE_CONDITIONS,
    CORE_FAMILY_C_ENTITY_CONDITIONS,
    CORE_FAMILY_D_DENSITIES,
    CORE_FAMILY_D_TRAPS,
    CORE_FAMILY_E_DELETION_POSITIONS,
    CORE_FAMILY_E_LIFECYCLE_CELLS,
    CORE_FAMILY_F_SELECTOR_KINDS,
    CORE_FAMILY_G_ANSWER_KINDS,
    CORE_FAMILY_G_HOP_COUNTS,
    CORE_FAMILY_G_OBJECT_COUNTS,
    CORE_FAMILY_G_SYNTHESIS_KINDS,
)
from mub.vnext.generation.core_catalogs import CORE_SURFACE_IDS
from mub.vnext.generation.identity import stable_id
from mub.vnext.io import canonical_json_bytes, sha256_model


CORE_AUDIT_SCHEMA_VERSION = "memupdatebench.core.audit.v3"
CORE_AUDIT_SELECTION_POLICY_VERSION = "core-audit-v3"
CORE_AUDIT_SELECTION_ALGORITHM = "authenticated_quota_set_cover_v1"
CORE_AUDIT_CONDITION_POLICY_VERSION = "core-audit-conditions-v1"
CORE_AUDIT_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
    TaskFamily.DELETION_FORGETTING,
    TaskFamily.CURRENT_HISTORICAL_QUERY,
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
)
CORE_AUDIT_SURFACES = tuple(CORE_SURFACE_IDS)
_CORE_SPLIT_QUOTA = MappingProxyType({Split.TRAIN: 22, Split.DEV: 3, Split.TEST: 7})
_MAX_TASKS = 12_000


class _StrictFrozenCoreAuditModel(ImmutableContractModel):
    model_config = ConfigDict(strict=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(
            value.model_dump(mode="python", exclude_computed_fields=True)
        )
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _hash_bytes(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token(name: str, value: Any) -> str:
    if isinstance(value, (TaskFamily, Difficulty, Split)):
        value = value.value
    return f"{name}={value}"


def _tokens(name: str, values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(_token(name, value) for value in values)


_FAMILY_REQUIRED = MappingProxyType(
    {
        TaskFamily.REPEATED_SAME_SLOT: (
            *_tokens("update_depth", CORE_FAMILY_A_DEPTHS),
            *_tokens("condition", CORE_FAMILY_A_CONDITIONS),
        ),
        TaskFamily.INTERLEAVED_MULTI_SLOT: (
            *_tokens("update_depth", CORE_FAMILY_B_DEPTHS),
            *_tokens("active_object_count", CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS),
            *_tokens("interleaving_pattern", CORE_FAMILY_B_INTERLEAVING_PATTERNS),
            *_tokens("cross_slot_distractor_density", (0.0, 0.25, 0.5)),
        ),
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: (
            *_tokens("entity_condition", CORE_FAMILY_C_ENTITY_CONDITIONS),
            *_tokens("attribute_condition", CORE_FAMILY_C_ATTRIBUTE_CONDITIONS),
            *_tokens("resolution_status", ("unique", "ambiguous", "no_match")),
        ),
        TaskFamily.NOOP_WRITE_DISCIPLINE: (
            *_tokens("configured_noop_density", CORE_FAMILY_D_DENSITIES),
            *_tokens("trap_type", CORE_FAMILY_D_TRAPS),
        ),
        TaskFamily.DELETION_FORGETTING: (
            *_tokens("lifecycle_cell", CORE_FAMILY_E_LIFECYCLE_CELLS),
            *_tokens("deletion_position", CORE_FAMILY_E_DELETION_POSITIONS),
        ),
        TaskFamily.CURRENT_HISTORICAL_QUERY: (
            *_tokens("selector_kind", CORE_FAMILY_F_SELECTOR_KINDS),
        ),
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS: (
            *_tokens("synthesis_kind", CORE_FAMILY_G_SYNTHESIS_KINDS),
            *_tokens("hop_count", CORE_FAMILY_G_HOP_COUNTS),
            *_tokens("object_count", CORE_FAMILY_G_OBJECT_COUNTS),
            *_tokens("answer_kind", CORE_FAMILY_G_ANSWER_KINDS),
        ),
    }
)

CORE_AUDIT_REQUIRED_CONDITIONS = MappingProxyType(
    {
        family: tuple(
            sorted(
                {
                    *_tokens("split", (Split.TRAIN, Split.DEV, Split.TEST)),
                    *_tokens("difficulty", CORE_DIFFICULTIES),
                    *_tokens("surface_id", CORE_AUDIT_SURFACES),
                    *_FAMILY_REQUIRED[family],
                }
            )
        )
        for family in CORE_AUDIT_FAMILIES
    }
)


class CoreAuditSelectorConfig(_StrictFrozenCoreAuditModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    selection_policy_version: Literal[CORE_AUDIT_SELECTION_POLICY_VERSION] = (
        CORE_AUDIT_SELECTION_POLICY_VERSION
    )
    selection_algorithm: Literal[CORE_AUDIT_SELECTION_ALGORITHM] = (
        CORE_AUDIT_SELECTION_ALGORITHM
    )
    condition_policy_version: Literal[CORE_AUDIT_CONDITION_POLICY_VERSION] = (
        CORE_AUDIT_CONDITION_POLICY_VERSION
    )
    tasks_per_family: Literal[32] = 32
    tasks_per_surface_per_family: Literal[8] = 8
    train_per_family: Literal[22] = 22
    dev_per_family: Literal[3] = 3
    test_per_family: Literal[7] = 7
    semantic_core_policy: Literal["exactly_one_selected_surface_per_semantic_core"] = (
        "exactly_one_selected_surface_per_semantic_core"
    )


CORE_AUDIT_SELECTOR_CONFIG = CoreAuditSelectorConfig()


def selector_config_hash(config: CoreAuditSelectorConfig) -> str:
    if type(config) is not CoreAuditSelectorConfig:
        raise TypeError("config must be an exact CoreAuditSelectorConfig")
    return sha256_model(config)


class CoreAuditSurfaceVariant(_StrictFrozenCoreAuditModel):
    surface_id: str
    task_id: str
    task_hash: str

    @field_validator("surface_id", "task_id")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("surface variant identifiers must not be blank")
        return value

    @field_validator("task_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("surface variant task_hash must be lowercase sha256")
        return value


class CoreAuditSelection(_StrictFrozenCoreAuditModel):
    audit_id: str
    task_id: str
    task_hash: str
    semantic_core_id: str
    family: TaskFamily
    difficulty: Difficulty
    split: Split
    surface_id: str
    surface_variants: tuple[CoreAuditSurfaceVariant, ...]
    covered_conditions: tuple[str, ...]
    selection_reason: Literal["quota_set_cover", "quota_spread_fill"]

    @model_validator(mode="before")
    @classmethod
    def _enum_strings(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        payload = dict(value)
        for name, enum_type in (
            ("family", TaskFamily),
            ("difficulty", Difficulty),
            ("split", Split),
        ):
            if type(payload.get(name)) is str:
                payload[name] = enum_type(payload[name])
        return payload

    @field_validator("audit_id", "task_id", "semantic_core_id", "surface_id")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selection identifiers must not be blank")
        return value

    @field_validator("task_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("task_hash must be lowercase sha256")
        return value

    @field_validator("surface_variants", mode="before")
    @classmethod
    def _surface_variants(cls, value: Any) -> tuple[CoreAuditSurfaceVariant, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("surface_variants must be a list or tuple")
        variants = tuple(
            item
            if type(item) is CoreAuditSurfaceVariant
            else CoreAuditSurfaceVariant.model_validate(item)
            for item in value
        )
        return tuple(sorted(variants, key=lambda item: CORE_AUDIT_SURFACES.index(item.surface_id)))

    @field_validator("covered_conditions", mode="before")
    @classmethod
    def _conditions(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("covered_conditions must be a list or tuple")
        if any(type(item) is not str or not item.strip() for item in value):
            raise ValueError("covered_conditions require nonblank strings")
        normalized = tuple(sorted(value))
        if len(normalized) != len(set(normalized)):
            raise ValueError("covered_conditions must be unique")
        return normalized

    @model_validator(mode="after")
    def _surface_matrix(self) -> CoreAuditSelection:
        if len(self.surface_variants) != 4:
            raise ValueError("selection must bind all four semantic-core surfaces")
        if tuple(item.surface_id for item in self.surface_variants) != CORE_AUDIT_SURFACES:
            raise ValueError("selection surface variants must use the canonical surface order")
        if len({item.task_id for item in self.surface_variants}) != 4:
            raise ValueError("selection surface variant task IDs must be unique")
        selected = next(
            (item for item in self.surface_variants if item.surface_id == self.surface_id),
            None,
        )
        if selected is None or (selected.task_id, selected.task_hash) != (
            self.task_id,
            self.task_hash,
        ):
            raise ValueError("selected task must match its bound surface variant")
        return self


class CoreAuditFamilySelectionReport(_StrictFrozenCoreAuditModel):
    family: TaskFamily
    required_conditions: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    uncovered_required_conditions: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _family_string(cls, value: Any) -> Any:
        if type(value) is dict and type(value.get("family")) is str:
            return {**value, "family": TaskFamily(value["family"])}
        return value

    @field_validator(
        "required_conditions", "selected_task_ids", "uncovered_required_conditions", mode="before"
    )
    @classmethod
    def _text_tuple(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("report fields must be lists or tuples")
        result = tuple(sorted(value))
        if any(type(item) is not str or not item.strip() for item in result):
            raise ValueError("report fields require nonblank strings")
        if len(result) != len(set(result)):
            raise ValueError("report fields must be unique")
        return result


class CoreAuditSelectionPackage(_StrictFrozenCoreAuditModel):
    schema_version: Literal[CORE_AUDIT_SCHEMA_VERSION] = CORE_AUDIT_SCHEMA_VERSION
    selection_policy_version: Literal[CORE_AUDIT_SELECTION_POLICY_VERSION]
    source_task_manifest_hash: str
    selector_config: CoreAuditSelectorConfig
    selector_config_hash: str
    selections: tuple[CoreAuditSelection, ...]
    family_reports: tuple[CoreAuditFamilySelectionReport, ...]
    selection_hash: str

    @field_validator("source_task_manifest_hash", "selector_config_hash", "selection_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("audit provenance hashes must be lowercase sha256")
        return value

    @field_validator("selections", mode="before")
    @classmethod
    def _selection_tuple(cls, value: Any) -> tuple[CoreAuditSelection, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("selections must be a list or tuple")
        return tuple(
            item if type(item) is CoreAuditSelection else CoreAuditSelection.model_validate(item)
            for item in value
        )

    @field_validator("family_reports", mode="before")
    @classmethod
    def _report_tuple(cls, value: Any) -> tuple[CoreAuditFamilySelectionReport, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("family_reports must be a list or tuple")
        return tuple(
            item
            if type(item) is CoreAuditFamilySelectionReport
            else CoreAuditFamilySelectionReport.model_validate(item)
            for item in value
        )

    @model_validator(mode="after")
    def _coherent(self) -> CoreAuditSelectionPackage:
        if self.selection_policy_version != self.selector_config.selection_policy_version:
            raise ValueError("selection policy version mismatch")
        if self.selector_config_hash != selector_config_hash(self.selector_config):
            raise ValueError("selector configuration hash mismatch")
        if len(self.selections) != 224:
            raise ValueError("Core audit requires exactly 224 selections")
        if len({item.task_id for item in self.selections}) != 224:
            raise ValueError("Core audit selected task IDs must be unique")
        if len({item.audit_id for item in self.selections}) != 224:
            raise ValueError("Core audit IDs must be unique")
        if len({item.semantic_core_id for item in self.selections}) != 224:
            raise ValueError("Core audit must select exactly one surface per semantic core")
        expected_reports = tuple(CORE_AUDIT_FAMILIES)
        if tuple(report.family for report in self.family_reports) != expected_reports:
            raise ValueError("family reports must cover A-G in canonical order")
        by_family: dict[TaskFamily, list[CoreAuditSelection]] = defaultdict(list)
        for item in self.selections:
            by_family[item.family].append(item)
            if item.surface_id not in CORE_AUDIT_SURFACES:
                raise ValueError("selection uses an unknown Core surface")
            if item.audit_id != _audit_id(
                item,
                source_task_manifest_hash=self.source_task_manifest_hash,
                selector_config_hash=self.selector_config_hash,
            ):
                raise ValueError("audit ID does not bind the selection provenance")
        for report in self.family_reports:
            rows = by_family[report.family]
            if len(rows) != 32:
                raise ValueError("each Core family requires 32 selections")
            if Counter(row.surface_id for row in rows) != {
                surface: 8 for surface in CORE_AUDIT_SURFACES
            }:
                raise ValueError("each Core family requires eight tasks per surface")
            if Counter(row.split for row in rows) != dict(_CORE_SPLIT_QUOTA):
                raise ValueError("each Core family requires 22/3/7 train/dev/test")
            required = CORE_AUDIT_REQUIRED_CONDITIONS[report.family]
            covered = {token for row in rows for token in row.covered_conditions}
            if report.required_conditions != required:
                raise ValueError("family report condition universe mismatch")
            if report.selected_task_ids != tuple(sorted(row.task_id for row in rows)):
                raise ValueError("family report selected task IDs mismatch")
            if report.uncovered_required_conditions != tuple(
                sorted(set(required) - covered)
            ):
                raise ValueError("family report uncovered conditions mismatch")
            if report.uncovered_required_conditions:
                raise ValueError("Core audit selection must cover all required conditions")
        if self.selection_hash != core_audit_selection_hash(self):
            raise ValueError("selection hash mismatch")
        return self

    @computed_field(return_type=bool)
    @property
    def release_ready(self) -> bool:
        return False


class _CoreCandidate:
    __slots__ = (
        "core_id",
        "family",
        "difficulty",
        "split",
        "conditions",
        "tasks_by_surface",
    )

    def __init__(
        self,
        *,
        core_id: str,
        family: TaskFamily,
        difficulty: Difficulty,
        split: Split,
        conditions: tuple[str, ...],
        tasks_by_surface: Mapping[str, MemUpdateTaskV3],
    ) -> None:
        self.core_id = core_id
        self.family = family
        self.difficulty = difficulty
        self.split = split
        self.conditions = conditions
        self.tasks_by_surface = tasks_by_surface


def _task_conditions(task: MemUpdateTaskV3) -> tuple[str, ...]:
    family = TaskFamily(task.task_family)
    stratification = task.metadata.extra["stratification"]
    conditions = {
        _token("split", task.metadata.split),
        _token("difficulty", task.difficulty),
        _token("surface_id", task.metadata.extra["surface_template"]),
    }
    keys = {
        TaskFamily.REPEATED_SAME_SLOT: ("condition",),
        TaskFamily.INTERLEAVED_MULTI_SLOT: (
            "update_depth",
            "active_object_count",
            "interleaving_pattern",
            "cross_slot_distractor_density",
        ),
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: (
            "entity_condition",
            "attribute_condition",
            "resolution_status",
        ),
        TaskFamily.NOOP_WRITE_DISCIPLINE: (
            "configured_noop_density",
            "trap_type",
        ),
        TaskFamily.DELETION_FORGETTING: (
            "lifecycle_cell",
            "deletion_position",
        ),
        TaskFamily.CURRENT_HISTORICAL_QUERY: (),
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS: (
            "synthesis_kind",
            "hop_count",
            "object_count",
            "answer_kind",
        ),
    }[family]
    for key in keys:
        if key in stratification:
            conditions.add(_token(key, stratification[key]))
    if family is TaskFamily.REPEATED_SAME_SLOT:
        conditions.add(
            _token("update_depth", task.metadata.resolved_profile["update_depth"])
        )
    if family is TaskFamily.CURRENT_HISTORICAL_QUERY:
        conditions.add(_token("selector_kind", task.queries[0].selector.kind))
    return tuple(sorted(conditions))


def _snapshot_authenticated_release(
    tasks: Iterable[Any],
    manifest: TaskManifestV3,
    source_task_manifest_hash: str,
) -> tuple[MemUpdateTaskV3, ...]:
    if type(manifest) is not TaskManifestV3:
        raise TypeError("Core audit requires an exact strict-v3 TaskManifestV3")
    if source_task_manifest_hash != sha256_model(manifest):
        raise ValueError("source task manifest hash does not match the authenticated manifest")
    copied_list = []
    iterator = iter(tasks)
    for _ in range(_MAX_TASKS + 1):
        try:
            copied_list.append(next(iterator))
        except StopIteration:
            break
    if len(copied_list) > _MAX_TASKS:
        raise ValueError("Core candidate exceeds the 12,000-task audit bound")
    copied = tuple(copied_list)
    if any(type(task) is not MemUpdateTaskV3 for task in copied):
        raise TypeError("Core audit requires exact strict-v3 MemUpdateTaskV3 records")
    ids = tuple(task.task_id for task in copied)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate task IDs must be unique")
    if set(ids) != set(manifest.task_record_hashes):
        raise ValueError("candidate task IDs must exactly match the task manifest")
    snapshots = []
    for task in copied:
        snapshot = MemUpdateTaskV3.model_validate_json(canonical_json_bytes(task))
        if sha256_model(snapshot) != manifest.task_record_hashes[snapshot.task_id]:
            raise ValueError(f"task record hash mismatch for {snapshot.task_id}")
        snapshots.append(snapshot)
    return tuple(sorted(snapshots, key=lambda task: task.task_id))


def _core_candidates(tasks: tuple[MemUpdateTaskV3, ...]) -> tuple[_CoreCandidate, ...]:
    grouped: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    for task in tasks:
        family = TaskFamily(task.task_family)
        if family not in CORE_AUDIT_FAMILIES:
            raise ValueError("Core audit candidate contains a non-Core family")
        grouped[task.metadata.split_key.semantic_core_id].append(task)
    candidates = []
    for core_id, variants in sorted(grouped.items()):
        surfaces = {task.metadata.extra["surface_template"]: task for task in variants}
        if len(variants) != 4 or set(surfaces) != set(CORE_AUDIT_SURFACES):
            raise ValueError("each semantic core must expose exactly four canonical surfaces")
        reference = variants[0]
        invariant = {
            (TaskFamily(task.task_family), task.difficulty, task.metadata.split)
            for task in variants
        }
        if len(invariant) != 1:
            raise ValueError("semantic-core variants disagree on family/difficulty/split")
        core_conditions = {
            token
            for token in _task_conditions(reference)
            if not token.startswith("surface_id=")
        }
        candidates.append(
            _CoreCandidate(
                core_id=core_id,
                family=TaskFamily(reference.task_family),
                difficulty=reference.difficulty,
                split=reference.metadata.split,
                conditions=tuple(sorted(core_conditions)),
                tasks_by_surface=surfaces,
            )
        )
    return tuple(candidates)


def _exact_quota_cover(
    candidates: Sequence[_CoreCandidate],
    required: set[str],
    quotas: Mapping[Split, int],
    target_count: int,
) -> tuple[_CoreCandidate, ...] | None:
    """Deterministic exact cover with split capacities, followed by quota-safe fill."""
    ordered = tuple(sorted(candidates, key=lambda item: item.core_id))
    condition_sets = {
        item.core_id: frozenset(item.conditions) for item in ordered
    }
    by_condition = {
        condition: tuple(
            item for item in ordered if condition in condition_sets[item.core_id]
        )
        for condition in sorted(required)
    }
    if any(not options for options in by_condition.values()):
        return None
    failed: set[tuple[frozenset[str], tuple[tuple[str, int], ...]]] = set()

    def search(
        uncovered: frozenset[str], remaining: Counter[Split]
    ) -> tuple[_CoreCandidate, ...] | None:
        if not uncovered:
            return ()
        state = (
            uncovered,
            tuple(
                (split.value, remaining[split])
                for split in sorted(quotas, key=lambda item: item.value)
            ),
        )
        if state in failed:
            return None
        pivot = min(
            uncovered,
            key=lambda token: (
                sum(
                    remaining[item.split] > 0 for item in by_condition[token]
                ),
                token,
            ),
        )
        options = sorted(
            (
                item
                for item in by_condition[pivot]
                if remaining[item.split] > 0
            ),
            key=lambda item: (
                -len(condition_sets[item.core_id] & uncovered),
                item.core_id,
            ),
        )
        for item in options:
            next_remaining = remaining.copy()
            next_remaining[item.split] -= 1
            tail = search(
                uncovered - condition_sets[item.core_id], next_remaining
            )
            if tail is not None:
                return (item, *tail)
        failed.add(state)
        return None

    cover = search(frozenset(required), Counter(quotas))
    if cover is None or len(cover) > target_count:
        return None
    used = {item.core_id for item in cover}
    remaining = Counter(quotas)
    for item in cover:
        remaining[item.split] -= 1
    filled = list(cover)
    for split in sorted(quotas, key=lambda item: item.value):
        options = [
            item
            for item in ordered
            if item.split is split and item.core_id not in used
        ]
        if len(options) < remaining[split]:
            return None
        for item in options[: remaining[split]]:
            filled.append(item)
            used.add(item.core_id)
    if len(filled) != target_count:
        return None
    return tuple(filled)


def _select_family_cores(
    family: TaskFamily, candidates: Sequence[_CoreCandidate]
) -> tuple[tuple[_CoreCandidate, str], ...]:
    required = {
        token
        for token in CORE_AUDIT_REQUIRED_CONDITIONS[family]
        if not token.startswith("surface_id=")
    }
    uncovered = set(required)
    remaining = Counter(_CORE_SPLIT_QUOTA)
    selected: list[tuple[_CoreCandidate, str]] = []
    used: set[str] = set()
    condition_counts: Counter[str] = Counter()
    while len(selected) < 32:
        eligible = [
            item
            for item in candidates
            if item.core_id not in used and remaining[item.split] > 0
        ]
        if not eligible:
            raise ValueError(f"family {family.value} cannot satisfy the split/core quota")
        def scarcity(item: _CoreCandidate) -> int:
            covered = set(item.conditions) & uncovered
            if not covered:
                return len(eligible) + 1
            return min(
                sum(
                    candidate.split in remaining
                    and remaining[candidate.split] > 0
                    and token in candidate.conditions
                    for candidate in eligible
                )
                for token in covered
            )

        best = min(
            eligible,
            key=lambda item: (
                scarcity(item),
                -len(set(item.conditions) & uncovered),
                sum(condition_counts[token] for token in item.conditions),
                item.core_id,
            ),
        )
        reason = "quota_set_cover" if set(best.conditions) & uncovered else "quota_spread_fill"
        selected.append((best, reason))
        used.add(best.core_id)
        remaining[best.split] -= 1
        uncovered.difference_update(best.conditions)
        condition_counts.update(best.conditions)
    if any(remaining.values()):
        raise ValueError(f"family {family.value} did not satisfy exact split quotas")
    if uncovered:
        exact = _exact_quota_cover(
            candidates, required, _CORE_SPLIT_QUOTA, 32
        )
        if exact is None:
            availability = {
                token: sum(token in item.conditions for item in candidates)
                for token in sorted(uncovered)
            }
            raise ValueError(
                f"family {family.value} has an unsatisfiable 32-item 22/3/7 "
                f"condition cover; uncovered availability={availability}"
            )
        selected = []
        exact_uncovered = set(required)
        for item in exact:
            contributes = bool(set(item.conditions) & exact_uncovered)
            selected.append(
                (
                    item,
                    "quota_set_cover" if contributes else "quota_spread_fill",
                )
            )
            exact_uncovered.difference_update(item.conditions)
    return tuple(selected)


def _audit_id(
    selection: CoreAuditSelection,
    *,
    source_task_manifest_hash: str,
    selector_config_hash: str,
) -> str:
    payload = selection.model_dump(mode="json", exclude={"audit_id"})
    return stable_id(
        "core_audit",
        {
            "schema_version": CORE_AUDIT_SCHEMA_VERSION,
            "selection_policy_version": CORE_AUDIT_SELECTION_POLICY_VERSION,
            "source_task_manifest_hash": source_task_manifest_hash,
            "selector_config_hash": selector_config_hash,
            "selection": payload,
        },
    )


def core_audit_selection_hash(package: CoreAuditSelectionPackage | Mapping[str, Any]) -> str:
    if isinstance(package, CoreAuditSelectionPackage):
        payload = package.model_dump(
            mode="json",
            exclude={"selection_hash", "release_ready"},
            exclude_computed_fields=True,
        )
    elif isinstance(package, Mapping):
        payload = dict(package)
        payload.pop("selection_hash", None)
        payload.pop("release_ready", None)
    else:
        raise TypeError("selection package must be a CoreAuditSelectionPackage or mapping")
    return _hash_bytes(payload)


def core_audit_review_context_hash(package: CoreAuditSelectionPackage) -> str:
    """Bind the ordered four-surface task IDs and authenticated record hashes."""
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    return _hash_bytes(
        {
            "schema_version": CORE_AUDIT_SCHEMA_VERSION,
            "source_task_manifest_hash": package.source_task_manifest_hash,
            "selection_hash": package.selection_hash,
            "surface_variants": [
                {
                    "audit_id": selection.audit_id,
                    "semantic_core_id": selection.semantic_core_id,
                    "variants": [variant.model_dump(mode="json") for variant in selection.surface_variants],
                }
                for selection in package.selections
            ],
        }
    )


def select_core_audit_sample(
    tasks: Iterable[Any],
    manifest: TaskManifestV3,
    *,
    source_task_manifest_hash: str,
    selector_config: CoreAuditSelectorConfig = CORE_AUDIT_SELECTOR_CONFIG,
) -> CoreAuditSelectionPackage:
    """Select the deterministic 224-task strict-v3 Core human-audit sample.

    The policy chooses 32 different semantic cores per family, with exact 22/3/7
    split quotas, then binds one surface to each selected core in an exact 8/8/8/8
    allocation. Duplicate task or semantic-core selections are never permitted.
    """
    if type(selector_config) is not CoreAuditSelectorConfig:
        raise TypeError("selector_config must be an exact CoreAuditSelectorConfig")
    snapshots = _snapshot_authenticated_release(
        tasks, manifest, source_task_manifest_hash
    )
    candidates = _core_candidates(snapshots)
    config_hash = selector_config_hash(selector_config)
    selections: list[CoreAuditSelection] = []
    reports: list[CoreAuditFamilySelectionReport] = []
    for family in CORE_AUDIT_FAMILIES:
        family_candidates = tuple(item for item in candidates if item.family is family)
        chosen = _select_family_cores(family, family_candidates)
        family_rows = []
        for index, (candidate, reason) in enumerate(
            sorted(chosen, key=lambda pair: pair[0].core_id)
        ):
            surface = CORE_AUDIT_SURFACES[index % len(CORE_AUDIT_SURFACES)]
            task = candidate.tasks_by_surface[surface]
            row = CoreAuditSelection(
                audit_id="pending",
                task_id=task.task_id,
                task_hash=manifest.task_record_hashes[task.task_id],
                semantic_core_id=candidate.core_id,
                family=family,
                difficulty=task.difficulty,
                split=task.metadata.split,
                surface_id=surface,
                surface_variants=tuple(
                    CoreAuditSurfaceVariant(
                        surface_id=surface_id,
                        task_id=candidate.tasks_by_surface[surface_id].task_id,
                        task_hash=manifest.task_record_hashes[
                            candidate.tasks_by_surface[surface_id].task_id
                        ],
                    )
                    for surface_id in CORE_AUDIT_SURFACES
                ),
                covered_conditions=_task_conditions(task),
                selection_reason=reason,
            )
            row = row.model_copy(
                update={
                    "audit_id": _audit_id(
                        row,
                        source_task_manifest_hash=source_task_manifest_hash,
                        selector_config_hash=config_hash,
                    )
                }
            )
            family_rows.append(row)
        selections.extend(family_rows)
        covered = {
            condition for row in family_rows for condition in row.covered_conditions
        }
        required = CORE_AUDIT_REQUIRED_CONDITIONS[family]
        reports.append(
            CoreAuditFamilySelectionReport(
                family=family,
                required_conditions=required,
                selected_task_ids=tuple(row.task_id for row in family_rows),
                uncovered_required_conditions=tuple(set(required) - covered),
            )
        )
    payload = {
        "schema_version": CORE_AUDIT_SCHEMA_VERSION,
        "selection_policy_version": CORE_AUDIT_SELECTION_POLICY_VERSION,
        "source_task_manifest_hash": source_task_manifest_hash,
        "selector_config": selector_config,
        "selector_config_hash": config_hash,
        "selections": tuple(selections),
        "family_reports": tuple(reports),
    }
    return CoreAuditSelectionPackage(
        **payload,
        selection_hash=core_audit_selection_hash(payload),
    )


__all__ = [
    "CORE_AUDIT_CONDITION_POLICY_VERSION",
    "CORE_AUDIT_FAMILIES",
    "CORE_AUDIT_REQUIRED_CONDITIONS",
    "CORE_AUDIT_SCHEMA_VERSION",
    "CORE_AUDIT_SELECTION_ALGORITHM",
    "CORE_AUDIT_SELECTION_POLICY_VERSION",
    "CORE_AUDIT_SELECTOR_CONFIG",
    "CORE_AUDIT_SURFACES",
    "CoreAuditFamilySelectionReport",
    "CoreAuditSelection",
    "CoreAuditSelectionPackage",
    "CoreAuditSelectorConfig",
    "CoreAuditSurfaceVariant",
    "core_audit_review_context_hash",
    "core_audit_selection_hash",
    "select_core_audit_sample",
    "selector_config_hash",
]
