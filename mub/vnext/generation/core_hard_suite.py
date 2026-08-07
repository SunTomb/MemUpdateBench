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
_REQUIRED_CONDITIONS = {
    TaskFamily.REPEATED_SAME_SLOT: (
        "condition=stale_burden",
        "condition=duplicate_current",
        "condition=other_attribute_distractor",
        "condition=same_name_other_entity_distractor",
        "update_depth=32",
    ),
    TaskFamily.INTERLEAVED_MULTI_SLOT: (
        "interleaving_pattern=round_robin",
        "interleaving_pattern=burst",
        "interleaving_pattern=adversarial_adjacent",
        "active_object_count=12",
        "update_depth=16",
    ),
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: (
        "entity_condition=distinct",
        "entity_condition=same_name",
        "entity_condition=alias",
        "entity_condition=namespace_collision",
        "attribute_condition=exact",
        "attribute_condition=paraphrase",
        "attribute_condition=near_name",
        "resolution_status=unique",
        "resolution_status=ambiguous",
        "resolution_status=no_match",
    ),
    TaskFamily.NOOP_WRITE_DISCIPLINE: tuple(
        f"trap_type={trap}"
        for trap in (
            "transient",
            "hypothetical",
            "negated",
            "uncertain",
            "semantic_near_miss",
            "duplicate_current",
            "unsupported_inference",
        )
    ) + ("configured_noop_density=0.75",),
    TaskFamily.DELETION_FORGETTING: tuple(
        f"lifecycle_cell={cell}"
        for cell in (
            "explicit_object_or_attribute_deletion",
            "entity_wide_deletion",
            "namespace_privacy_wipe",
            "correction_versus_deletion_hard_negative",
            "logical_ttl_expiry",
            "post_delete_similar_retrieval",
            "delete_then_relearn",
            "scoped_delete_protected_collateral",
        )
    ) + tuple(
        f"deletion_position={position}"
        for position in ("early", "middle", "final", "not_applicable")
    ),
    TaskFamily.CURRENT_HISTORICAL_QUERY: tuple(
        f"selector_kind={selector}"
        for selector in (
            "current",
            "previous",
            "exact_version",
            "event_anchor",
            "logical_time_anchor",
            "transition",
            "ordered_history",
        )
    ),
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS: (
        "synthesis_kind=update_sensitive_multi_hop",
        "synthesis_kind=multi_object_current_consistency",
        "hop_count=4",
        "object_count=8",
        "answer_kind=boolean_consistency",
        "answer_kind=exact_inconsistent_object",
    ),
}
_DIFFICULTY_RANK = {"hard": 0, "challenge": 0, "medium": 1, "easy": 2}



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


def _condition_labels(task, family: TaskFamily) -> frozenset[str]:
    strata = task.metadata.extra["stratification"]
    labels = {f"difficulty={task.difficulty.value}"}
    for required in _REQUIRED_CONDITIONS[family]:
        axis, expected = required.split("=", 1)
        if axis == "selector_kind":
            observed = task.queries[0].selector.kind
        elif axis == "update_depth" and family is TaskFamily.REPEATED_SAME_SLOT:
            observed = task.metadata.resolved_profile.get("update_depth")
        else:
            observed = strata.get(axis)
        if str(observed) == expected:
            labels.add(required)
    return frozenset(labels)


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
        for core_id, assignment in assignment_by_core.items():
            if assignment.task_family is not family:
                continue
            representative = min(
                tasks_by_core[core_id],
                key=lambda task: task.metadata.extra["surface_variant"],
            )
            labels = _condition_labels(representative, family)
            difficulty_rank = _DIFFICULTY_RANK[representative.difficulty.value]
            candidates.append(
                (
                    difficulty_rank,
                    _ranking(core_id, "hard-first-fixed-strata"),
                    core_id,
                    labels,
                )
            )
        if len(candidates) < per_family:
            raise ValueError(f"Family {family.value} has too few eligible test cores")
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = []
        chosen_ids = set()
        required_conditions = (
            _REQUIRED_CONDITIONS[family] if per_family == 20 else ()
        )
        for condition in required_conditions:
            match = next(
                (
                    item
                    for item in candidates
                    if item[2] not in chosen_ids and condition in item[3]
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"Family {family.value} cannot cover required hard condition {condition}"
                )
            chosen.append(match)
            chosen_ids.add(match[2])
        chosen.extend(
            item
            for item in candidates
            if item[2] not in chosen_ids
        )
        chosen = chosen[:per_family]
        family_core_ids = sorted(item[2] for item in chosen)
        selected_ids.update(family_core_ids)
        family_counts[family.value] = len(family_core_ids) * 4
        selected_labels = sorted(set().union(*(item[3] for item in chosen)))
        coverage[family.value] = {
            condition: sorted(item[2] for item in chosen if condition in item[3])
            for condition in selected_labels
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
