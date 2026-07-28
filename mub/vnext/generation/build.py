from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from mub.vnext.contracts import MemUpdateTask, Split, TaskFamily
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.generation.family_b import generate_family_b_cores
from mub.vnext.generation.family_c import generate_family_c_cores
from mub.vnext.generation.family_d import generate_family_d_cores
from mub.vnext.generation.render import render_core
from mub.vnext.generation.splits import SplitAssignmentResult, assign_splits
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation import validate_gold_replay, validate_task

_EXPECTED_CORE_COUNT = 480
_EXPECTED_TASK_COUNT = 1440
_EXPECTED_SPLIT_COUNTS = {
    Split.TRAIN: 1008,
    Split.DEV: 144,
    Split.TEST: 288,
}
_EXPECTED_FAMILY_COUNT = 360
_SPLIT_ORDER = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
_FAMILY_ORDER = {
    TaskFamily.REPEATED_SAME_SLOT.value: 0,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: 1,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: 2,
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: 3,
}


@dataclass(frozen=True, slots=True)
class CompiledPilotTasks:
    tasks: tuple[MemUpdateTask, ...]
    split_assignment: SplitAssignmentResult
    config_sha256: str
    tasks_jsonl: bytes


def _task_sort_key(task: MemUpdateTask) -> tuple[int, int, str, int]:
    return (
        _SPLIT_ORDER[task.metadata.split],
        _FAMILY_ORDER[task.task_family],
        task.metadata.split_key.semantic_core_id,
        task.metadata.extra["surface_variant"],
    )


def _canonical_jsonl(tasks: tuple[MemUpdateTask, ...]) -> bytes:
    return b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)


def _raise_invalid(task: MemUpdateTask, stage: str, report) -> None:
    detail = "; ".join(
        f"{issue.code}@{issue.path}: {issue.message}" for issue in report.issues
    )
    raise ValueError(
        f"compiled task {task.task_id!r} failed {stage} validation: {detail}"
    )


def _validate_compiled_tasks(tasks: tuple[MemUpdateTask, ...]) -> None:
    validators: tuple[tuple[str, Callable], ...] = (
        ("task", validate_task),
        ("gold replay", validate_gold_replay),
    )
    for task in tasks:
        for stage, validator in validators:
            report = validator(task)
            if not report.valid:
                _raise_invalid(task, stage, report)


def _verify_round_trip(
    tasks: tuple[MemUpdateTask, ...],
    tasks_jsonl: bytes,
) -> None:
    rows = tasks_jsonl.splitlines(keepends=True)
    if len(rows) != len(tasks):
        raise ValueError("canonical tasks JSONL row count changed during serialization")

    reparsed: list[MemUpdateTask] = []
    for row_number, (task, row) in enumerate(zip(tasks, rows, strict=True), start=1):
        if not row.endswith(b"\n") or row.endswith(b"\r\n"):
            raise ValueError(f"canonical tasks JSONL row {row_number} is not binary-LF")
        parsed = MemUpdateTask.model_validate_json(row[:-1])
        if parsed != task:
            raise ValueError(
                f"canonical tasks JSONL row {row_number} changed its task record"
            )
        if canonical_json_bytes(parsed) + b"\n" != row:
            raise ValueError(
                f"canonical tasks JSONL row {row_number} is not byte-canonical"
            )
        reparsed.append(parsed)

    if tuple(reparsed) != tasks or _canonical_jsonl(tuple(reparsed)) != tasks_jsonl:
        raise ValueError("canonical tasks JSONL failed exact record/byte round trip")


def compile_pilot_tasks(
    config: PilotConfig,
    *,
    code_revision: str,
) -> CompiledPilotTasks:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if type(code_revision) is not str:
        raise TypeError("code_revision must be a string")
    if not code_revision.strip():
        raise ValueError("code_revision must not be blank")

    context = GenerationContext(config=config, code_revision=code_revision)
    immutable_config = context.config
    cores = (
        *generate_family_a_cores(immutable_config),
        *generate_family_b_cores(immutable_config),
        *generate_family_c_cores(immutable_config),
        *generate_family_d_cores(immutable_config),
    )
    if len(cores) != _EXPECTED_CORE_COUNT:
        raise ValueError(
            "Pilot compilation requires exactly 480 semantic cores; "
            f"observed {len(cores)}"
        )

    split_assignment = assign_splits(cores, immutable_config.seed)
    split_by_core = {
        assignment.semantic_core_id: assignment.split
        for assignment in split_assignment.assignments
    }
    ordered_cores = sorted(
        cores,
        key=lambda core: (
            _SPLIT_ORDER[split_by_core[core.core_id]],
            _FAMILY_ORDER[core.task_family.value],
            core.core_id,
        ),
    )
    tasks = tuple(
        render_core(
            core,
            split=split_by_core[core.core_id],
            surface_variant=surface_variant,
            context=context,
        )
        for core in ordered_cores
        for surface_variant in range(3)
    )
    tasks = tuple(sorted(tasks, key=_task_sort_key))

    if len(tasks) != _EXPECTED_TASK_COUNT:
        raise ValueError(
            "Pilot compilation requires exactly 1440 tasks; "
            f"observed {len(tasks)}"
        )
    split_counts = Counter(task.metadata.split for task in tasks)
    if split_counts != Counter(_EXPECTED_SPLIT_COUNTS):
        raise ValueError(
            f"Pilot task split counts must be {_EXPECTED_SPLIT_COUNTS}; "
            f"observed {dict(split_counts)}"
        )
    family_counts = Counter(task.task_family for task in tasks)
    expected_family_counts = Counter(
        {family: _EXPECTED_FAMILY_COUNT for family in _FAMILY_ORDER}
    )
    if family_counts != expected_family_counts:
        raise ValueError(
            "Pilot task family counts must be 360 each; "
            f"observed {dict(family_counts)}"
        )

    _validate_compiled_tasks(tasks)
    tasks_jsonl = _canonical_jsonl(tasks)
    _verify_round_trip(tasks, tasks_jsonl)
    return CompiledPilotTasks(
        tasks=tasks,
        split_assignment=split_assignment,
        config_sha256=context.config_sha256,
        tasks_jsonl=tasks_jsonl,
    )


__all__ = ["CompiledPilotTasks", "compile_pilot_tasks"]
