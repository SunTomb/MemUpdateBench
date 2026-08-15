from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from mub.vnext.contracts import MemoryObjectKey, MemUpdateTask, Split, TaskFamily
from mub.vnext.generation.config import MechanismCondition, PilotConfig
from mub.vnext.io import semantic_task_hash, sha256_model
from mub.vnext.mechanisms.context import (
    APPROVED_CONTEXT_CONDITIONS,
    RenderedContext,
    entries_from_task,
    render_context,
)

APPROVED_CONDITIONS = APPROVED_CONTEXT_CONDITIONS
ANSWER_MODEL = "deterministic_reference_smoke"


class MechanismContextRecord(BaseModel):
    model_config = {"extra": "forbid"}

    condition_id: str
    task_id: str
    semantic_core_id: str
    stale_count: int
    seed: int
    retrieval_composition: str
    answer_model: str
    smoke_only: bool
    not_model_result: bool
    expected_comparison: str
    gold_value: Any
    rendered_context: str
    entry_ids: list[str]
    entry_order: list[int]
    labels: dict[str, str]
    context_order: str
    context_annotation: str


@dataclass(frozen=True, slots=True)
class MechanismSlice:
    records: tuple[MechanismContextRecord, ...]
    manifest: dict[str, Any]

    def __iter__(self):
        yield self.records
        yield self.manifest


def _condition_id(stale_count: int, order: str, annotation: str) -> str:
    return f"stale{stale_count}_{order}_{annotation}"


def _validate_config(config: PilotConfig) -> tuple[MechanismCondition, ...]:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if set(config.mechanism_slice.stale_counts) != {1, 16} or len(config.mechanism_slice.stale_counts) != 2:
        raise ValueError("mechanism slice stale_counts must be exactly [1, 16]")
    pairs = tuple((condition.context_order, condition.context_annotation) for condition in config.mechanism_slice.conditions)
    if set(pairs) != set(APPROVED_CONDITIONS) or len(pairs) != len(APPROVED_CONDITIONS):
        raise ValueError("mechanism slice conditions must be exactly the three approved cells")
    return tuple(MechanismCondition(context_order=order, context_annotation=annotation) for order, annotation in APPROVED_CONDITIONS)


def _task_core_id(task: MemUpdateTask) -> str:
    core_id = task.metadata.split_key.semantic_core_id
    if not isinstance(core_id, str) or not core_id.strip():
        raise ValueError(f"task {task.task_id} has no semantic-core ID")
    return core_id


def _query_target(task: MemUpdateTask) -> MemoryObjectKey:
    if len(task.queries) != 1 or len(task.queries[0].target_object_keys) != 1:
        raise ValueError(
            f"Family A mechanism task {task.task_id} must have one query target"
        )
    target = task.queries[0].target_object_keys[0]
    if target not in task.target_objects:
        raise ValueError(f"task {task.task_id} query target is not a declared object")
    return target


def _gold_value(task: MemUpdateTask) -> Any:
    target_id = _query_target(task).canonical_id
    if target_id not in task.gold.final_state:
        raise ValueError(f"task {task.task_id} lacks canonical final-state value")
    return task.gold.final_state[target_id]


def _select_tasks(tasks: tuple[MemUpdateTask, ...], config_hash: str) -> tuple[MemUpdateTask, ...]:
    by_id: dict[str, MemUpdateTask] = {}
    for task in tasks:
        if not isinstance(task, MemUpdateTask):
            raise TypeError("tasks must contain MemUpdateTask values")
        if task.task_id in by_id:
            raise ValueError(f"duplicate task ID: {task.task_id}")
        by_id[task.task_id] = task
    selected: dict[str, MemUpdateTask] = {}
    semantic_hashes: dict[str, str] = {}
    for task in sorted(by_id.values(), key=lambda item: item.task_id):
        if task.metadata.split is not Split.TEST:
            continue
        if task.task_family != TaskFamily.REPEATED_SAME_SLOT.value:
            continue
        stale = task.metadata.resolved_profile.get("stale_count")
        if type(stale) is not int or stale not in {1, 16}:
            continue
        if task.metadata.generation_config_hash != config_hash:
            raise ValueError(f"task {task.task_id} generation config hash does not match config")
        core_id = _task_core_id(task)
        task_semantic_hash = semantic_task_hash(task)
        if core_id in selected:
            if semantic_hashes[core_id] != task_semantic_hash:
                raise ValueError(f"semantic-core variants disagree for {core_id}")
            # Surface variants are paired views of one core. Keep one canonical
            # task, rather than silently multiplying the mechanism sample.
            continue
        semantic_hashes[core_id] = task_semantic_hash
        entries = entries_from_task(task)
        target = _query_target(task)
        actual_stale = sum(
            entry.object_key == target and entry.version_index < stale
            for entry in entries
        )
        if actual_stale != stale:
            raise ValueError(f"task {task.task_id} stale_count does not match canonical entries")
        selected[core_id] = task
    result = tuple(selected.values())
    if not result:
        raise ValueError("no matched Family A test semantic cores with stale_count 1 or 16")
    if {task.metadata.resolved_profile["stale_count"] for task in result} != {1, 16}:
        raise ValueError("matched mechanism slice must contain both stale_count 1 and 16")
    return tuple(sorted(result, key=lambda task: (_task_core_id(task), task.task_id)))


def _record(task: MemUpdateTask, condition: MechanismCondition, seed: int) -> MechanismContextRecord:
    stale = int(task.metadata.resolved_profile["stale_count"])
    entries = entries_from_task(task)
    rendered: RenderedContext = render_context(entries, condition.context_order, condition.context_annotation)
    # The comparison is a declared diagnostic contrast, not a result claim.
    return MechanismContextRecord(
        condition_id=_condition_id(stale, condition.context_order, condition.context_annotation),
        task_id=task.task_id,
        semantic_core_id=_task_core_id(task),
        stale_count=stale,
        seed=seed,
        retrieval_composition="identical_entry_multiset",
        answer_model=ANSWER_MODEL,
        smoke_only=True,
        not_model_result=True,
        expected_comparison="same_gold_and_entry_multiset; presentation_order_or_version_labels_only",
        gold_value=_gold_value(task),
        rendered_context=rendered.rendered_context,
        entry_ids=rendered.entry_ids,
        entry_order=rendered.entry_order,
        labels=rendered.labels,
        context_order=rendered.context_order,
        context_annotation=rendered.context_annotation,
    )


def build_mechanism_slice(tasks: Any, config: PilotConfig) -> MechanismSlice:
    """Select authenticated canonical tasks and build the approved smoke slice."""
    conditions = _validate_config(config)
    if not isinstance(tasks, (list, tuple)):
        tasks = tuple(tasks)
    selected = _select_tasks(tuple(tasks), sha256_model(config))
    records = tuple(
        record
        for task in selected
        for condition in conditions
        for record in (_record(task, condition, config.seed),)
    )
    records = tuple(sorted(records, key=lambda record: (record.stale_count, record.semantic_core_id, APPROVED_CONDITIONS.index((record.context_order, record.context_annotation)))))
    by_condition: defaultdict[str, list[MechanismContextRecord]] = defaultdict(list)
    for record in records:
        by_condition[record.condition_id].append(record)
    manifest_conditions = []
    for condition in conditions:
        for stale in (1, 16):
            condition_id = _condition_id(stale, condition.context_order, condition.context_annotation)
            rows = by_condition[condition_id]
            manifest_conditions.append({
                "condition_id": condition_id,
                "task_ids": [row.task_id for row in rows],
                "semantic_core_ids": [row.semantic_core_id for row in rows],
                "stale_count": stale,
                "n": len(rows),
                "seed": config.seed,
                "retrieval_composition": "identical_entry_multiset",
                "answer_model": ANSWER_MODEL,
                "smoke_only": True,
                "not_model_result": True,
                "expected_comparison": "same_gold_and_entry_multiset; presentation_order_or_version_labels_only",
            })
    manifest = {
        "schema_version": config.schema_version,
        "release_id": config.release_id,
        "seed": config.seed,
        "answer_model": ANSWER_MODEL,
        "smoke_only": True,
        "not_model_result": True,
        "stale_counts": [1, 16],
        "condition_ids": [item["condition_id"] for item in manifest_conditions],
        "conditions": manifest_conditions,
        "n": len(records),
        "task_ids": sorted({record.task_id for record in records}),
        "semantic_core_ids": sorted({record.semantic_core_id for record in records}),
        "task_hashes": {task.task_id: sha256_model(task) for task in selected},
    }
    return MechanismSlice(records=records, manifest=manifest)


ContextRecord = MechanismContextRecord

__all__ = [
    "ANSWER_MODEL",
    "APPROVED_CONDITIONS",
    "ContextRecord",
    "MechanismContextRecord",
    "MechanismSlice",
    "build_mechanism_slice",
]
