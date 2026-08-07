from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts import Split, TaskFamily
from mub.vnext.contracts.common import (
    FrozenJsonObject,
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    freeze_json,
    freeze_mapping,
)
from mub.vnext.generation.core_build import CompiledCoreSnapshot
from mub.vnext.io.canonical import _canonical_payload_bytes

HashString = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
_POLICY = "core-hard-v1"
_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
    TaskFamily.DELETION_FORGETTING,
    TaskFamily.CURRENT_HISTORICAL_QUERY,
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
)
_CONDITION_KEYS = {
    TaskFamily.REPEATED_SAME_SLOT: "condition",
    TaskFamily.INTERLEAVED_MULTI_SLOT: "interleaving_pattern",
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: "resolution_status",
    TaskFamily.NOOP_WRITE_DISCIPLINE: "trap_type",
    TaskFamily.DELETION_FORGETTING: "lifecycle_cell",
    TaskFamily.CURRENT_HISTORICAL_QUERY: "query_type",
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS: "synthesis_kind",
}


class CoreHardSuiteManifest(ImmutableContractModel):
    selection_policy_version: Literal["core-hard-v1"] = _POLICY
    source_task_manifest_hash: HashString
    semantic_core_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    per_family_core_count: int
    family_task_counts: FrozenNonnegativeIntMap
    condition_coverage: FrozenJsonObject
    suite_hash: HashString

    @field_validator("family_task_counts")
    @classmethod
    def _freeze_counts(cls, value):
        return freeze_mapping(value)

    @field_validator("condition_coverage")
    @classmethod
    def _freeze_coverage(cls, value):
        return freeze_json(value)

    @model_validator(mode="after")
    def _canonical(self):
        if self.per_family_core_count <= 0:
            raise ValueError("per_family_core_count must be positive")
        if self.semantic_core_ids != tuple(sorted(self.semantic_core_ids)):
            raise ValueError("semantic_core_ids must be sorted")
        if self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError("task_ids must be sorted")
        if len(self.semantic_core_ids) != len(set(self.semantic_core_ids)):
            raise ValueError("semantic_core_ids must be unique")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique")
        if self.suite_hash != core_hard_suite_hash(self):
            raise ValueError("suite_hash is not canonical")
        return self


def _hash_payload(manifest) -> dict:
    payload = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)
    payload.pop("suite_hash", None)
    return payload


def core_hard_suite_hash(manifest: CoreHardSuiteManifest | dict) -> str:
    return hashlib.sha256(_canonical_payload_bytes(_hash_payload(manifest))).hexdigest()


def _ranking(core_id: str, condition: str) -> str:
    return hashlib.sha256(f"{_POLICY}|{condition}|{core_id}".encode("utf-8")).hexdigest()


def build_core_hard_suite(
    snapshot: CompiledCoreSnapshot,
    *,
    source_task_manifest_hash: str,
    per_family: int = 20,
) -> CoreHardSuiteManifest:
    if type(snapshot) is not CompiledCoreSnapshot:
        raise TypeError("snapshot must be a CompiledCoreSnapshot")
    if type(per_family) is not int or per_family <= 0:
        raise ValueError("per_family must be a positive integer")
    assignment_by_core = {
        assignment.semantic_core_id: assignment
        for assignment in snapshot.assignments
        if assignment.split is Split.TEST
    }
    tasks_by_core = defaultdict(list)
    for task in snapshot.tasks:
        core_id = task.metadata.split_key.semantic_core_id
        if core_id in assignment_by_core:
            tasks_by_core[core_id].append(task)

    selected_ids: set[str] = set()
    coverage: dict[str, dict[str, list[str]]] = {}
    family_counts = Counter()
    for family in _FAMILIES:
        candidates = []
        condition_key = _CONDITION_KEYS[family]
        for core_id, assignment in assignment_by_core.items():
            if assignment.task_family is not family:
                continue
            representative = min(
                tasks_by_core[core_id],
                key=lambda task: task.metadata.extra["surface_variant"],
            )
            if family is TaskFamily.CURRENT_HISTORICAL_QUERY:
                condition = str(representative.queries[0].selector.kind)
            else:
                condition = str(
                    representative.metadata.extra["stratification"].get(condition_key)
                )
            candidates.append((_ranking(core_id, condition), core_id, condition))
        if len(candidates) < per_family:
            raise ValueError(f"Family {family.value} has too few eligible test cores")
        by_condition = defaultdict(list)
        for candidate in candidates:
            by_condition[candidate[2]].append(candidate)
        chosen = []
        for condition in sorted(by_condition)[:per_family]:
            chosen.append(min(by_condition[condition]))
        chosen_ids = {item[1] for item in chosen}
        remainder = sorted(
            (item for item in candidates if item[1] not in chosen_ids),
            key=lambda item: (item[0], item[1]),
        )
        chosen.extend(remainder[: per_family - len(chosen)])
        family_core_ids = sorted(item[1] for item in chosen)
        selected_ids.update(family_core_ids)
        family_counts[family.value] = len(family_core_ids) * 4
        coverage[family.value] = {
            condition: sorted(item[1] for item in chosen if item[2] == condition)
            for condition in sorted({item[2] for item in chosen})
        }

    task_ids = sorted(
        task.task_id
        for task in snapshot.tasks
        if task.metadata.split_key.semantic_core_id in selected_ids
    )
    expected_tasks = len(selected_ids) * 4
    if len(task_ids) != expected_tasks:
        raise ValueError("hard suite must reference exactly four existing tasks per core")
    payload = {
        "selection_policy_version": _POLICY,
        "source_task_manifest_hash": source_task_manifest_hash,
        "semantic_core_ids": sorted(selected_ids),
        "task_ids": task_ids,
        "per_family_core_count": per_family,
        "family_task_counts": {
            family.value: family_counts[family.value] for family in _FAMILIES
        },
        "condition_coverage": coverage,
    }
    return CoreHardSuiteManifest(**payload, suite_hash=core_hard_suite_hash(payload))


__all__ = [
    "CoreHardSuiteManifest",
    "build_core_hard_suite",
    "core_hard_suite_hash",
]
