from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated

from pydantic import Field, field_validator

from mub.vnext.contracts.common import (
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    freeze_mapping,
)
from mub.vnext.contracts.enums import Split, TaskFamily
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.config import SplitConfig
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.core_config import CoreConfig
from mub.vnext.generation.core_render_v3 import render_core_v3
from mub.vnext.generation.family_a import generate_core_family_a_cores
from mub.vnext.generation.family_b import generate_core_family_b_cores
from mub.vnext.generation.family_c import generate_core_family_c_cores
from mub.vnext.generation.family_d import generate_core_family_d_cores
from mub.vnext.generation.splits import (
    CoreSplitAssignment,
    _ranking_sha256,
    _resolved_strata,
)
from mub.vnext.io import semantic_task_hash_v3


_CORE_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)
_SPLIT_ORDER = (Split.TRAIN, Split.DEV, Split.TEST)
_VARIANTS_PER_CORE = 4
HashString = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class CompiledCoreSnapshot(ImmutableContractModel):
    config_sha256: HashString
    assignments: tuple[CoreSplitAssignment, ...]
    tasks: tuple[MemUpdateTaskV3, ...]
    family_core_counts: FrozenNonnegativeIntMap
    core_counts: FrozenNonnegativeIntMap
    task_counts: FrozenNonnegativeIntMap

    @field_validator("family_core_counts", "core_counts", "task_counts")
    @classmethod
    def _freeze_counts(cls, value: Mapping[str, int]):
        return freeze_mapping(value)


def _split_quotas(
    selected_count: int,
    splits: SplitConfig,
) -> tuple[int, int, int]:
    ratios = (splits.train, splits.dev, splits.test)
    quotas = []
    for split, ratio in zip(_SPLIT_ORDER, ratios, strict=True):
        quota = Decimal(selected_count) * Decimal(str(ratio))
        if quota != quota.to_integral_value():
            raise ValueError(
                f"selected Core family count {selected_count} must allocate a whole "
                f"number of cores to split {split.value}"
            )
        quotas.append(int(quota))
    if sum(quotas) != selected_count:
        raise ValueError("configured Core split quotas must preserve the selected count")
    return tuple(quotas)


def _validate_limit(config: CoreConfig, cores_per_family: int | None) -> int | None:
    if cores_per_family is None:
        return None
    if type(cores_per_family) is not int:
        raise TypeError("cores_per_family must be an exact integer or None")
    if cores_per_family <= 0:
        raise ValueError("cores_per_family must be positive")
    if any(cores_per_family > count for count in config.family_core_counts.values()):
        raise ValueError("cores_per_family cannot exceed any configured family count")
    _split_quotas(cores_per_family, config.splits)
    return cores_per_family


def _validate_core_diagnostic_schedule(
    cores: tuple[SemanticCore, ...],
    config: CoreConfig,
) -> None:
    by_family = {
        family: tuple(core for core in cores if core.task_family is family)
        for family in _CORE_FAMILIES
    }

    family_a = config.families.repeated_same_slot_update
    a_cores = by_family[TaskFamily.REPEATED_SAME_SLOT]
    a_depths = Counter(core.profile["update_depth"] for core in a_cores)
    if dict(a_depths) != {
        depth: family_a.schedule.cores_per_update_depth
        for depth in family_a.update_depths
    }:
        raise ValueError("Core Family A schedule depth marginal is invalid")
    a_cells = Counter(
        (core.profile["update_depth"], core.stratification.get("condition"))
        for core in a_cores
    )
    expected_a_cells = {
        (depth, condition): family_a.schedule.cores_per_depth_condition_cell
        for depth in family_a.update_depths
        for condition in family_a.conditions
    }
    if dict(a_cells) != expected_a_cells:
        raise ValueError("Core Family A schedule condition cells are invalid")

    family_b = config.families.interleaved_multi_slot_update
    b_cores = by_family[TaskFamily.INTERLEAVED_MULTI_SLOT]
    for core in b_cores:
        profile_depth = core.profile["update_depth"]
        strata_depth = core.stratification.get("update_depth")
        if profile_depth != strata_depth:
            raise ValueError("Core Family B schedule depth metadata is inconsistent")
    b_depths = Counter(core.stratification.get("update_depth") for core in b_cores)
    if dict(b_depths) != {
        depth: family_b.schedule.cores_per_update_depth
        for depth in family_b.update_depths
    }:
        raise ValueError("Core Family B schedule depth marginal is invalid")
    b_active = Counter(core.stratification.get("active_object_count") for core in b_cores)
    if dict(b_active) != {
        count: family_b.schedule.cores_per_active_object_count
        for count in family_b.active_object_counts
    }:
        raise ValueError("Core Family B schedule active-object marginal is invalid")
    b_patterns = Counter(core.stratification.get("interleaving_pattern") for core in b_cores)
    if dict(b_patterns) != {
        pattern: family_b.schedule.cores_per_pattern_within_active_object_count
        * len(family_b.active_object_counts)
        for pattern in family_b.interleaving_patterns
    }:
        raise ValueError("Core Family B schedule pattern marginal is invalid")
    b_cells = Counter(
        (
            core.stratification.get("active_object_count"),
            core.stratification.get("update_depth"),
            core.stratification.get("interleaving_pattern"),
        )
        for core in b_cores
    )
    expected_b_cells = {
        (active_count, depth, pattern): b_cells[(active_count, depth, pattern)]
        for active_count in family_b.active_object_counts
        for depth in family_b.update_depths
        for pattern in family_b.interleaving_patterns
    }
    for active_count in family_b.active_object_counts:
        stratum_cells = [
            b_cells[(active_count, depth, pattern)]
            for depth in family_b.update_depths
            for pattern in family_b.interleaving_patterns
        ]
        if set(stratum_cells) != {
            family_b.schedule.depth_pattern_cell_min,
            family_b.schedule.depth_pattern_cell_max,
        } or max(stratum_cells) - min(stratum_cells) > family_b.schedule.max_depth_pattern_cell_imbalance:
            raise ValueError("Core Family B schedule depth-pattern cells are invalid")
    if dict(expected_b_cells) != dict(b_cells):
        raise ValueError("Core Family B schedule contains unknown cells")

    family_c = config.families.entity_attribute_grounding
    c_cores = by_family[TaskFamily.ENTITY_ATTRIBUTE_GROUNDING]
    c_cells = Counter(
        (
            core.stratification.get("entity_condition"),
            core.stratification.get("attribute_condition"),
        )
        for core in c_cores
    )
    expected_c_cells = {
        (entity_condition, attribute_condition): family_c.schedule.cores_per_entity_attribute_cell
        for entity_condition in family_c.entity_conditions
        for attribute_condition in family_c.attribute_conditions
    }
    if dict(c_cells) != expected_c_cells:
        raise ValueError("Core Family C schedule condition cells are invalid")
    c_outcomes = Counter(core.stratification.get("resolution_status") for core in c_cores)
    if dict(c_outcomes) != {
        status: family_c.schedule.cores_per_resolution_outcome
        for status in ("unique", "ambiguous", "no_match")
    }:
        raise ValueError("Core Family C schedule resolution outcomes are invalid")

    family_d = config.families.noop_write_discipline
    d_cores = by_family[TaskFamily.NOOP_WRITE_DISCIPLINE]
    d_cells = Counter(
        (
            core.stratification.get("trap_type"),
            core.stratification.get("configured_noop_density"),
        )
        for core in d_cores
    )
    expected_d_cells = {
        (trap, density): family_d.schedule.cores_per_trap_density_cell
        for trap in family_d.trap_types
        for density in family_d.noop_densities
    }
    if dict(d_cells) != expected_d_cells:
        raise ValueError("Core Family D schedule trap-density cells are invalid")


def _generated_cores(config: CoreConfig) -> tuple[SemanticCore, ...]:
    families = (
        generate_core_family_a_cores(config),
        generate_core_family_b_cores(config),
        generate_core_family_c_cores(config),
        generate_core_family_d_cores(config),
    )
    cores = tuple(core for family in families for core in family)
    expected = config.family_core_counts
    observed = Counter(core.task_family.value for core in cores)
    if observed != Counter(expected):
        raise ValueError(
            f"Core generators returned unexpected family counts: {dict(observed)}"
        )
    core_ids = [core.core_id for core in cores]
    if len(core_ids) != len(set(core_ids)):
        raise ValueError("Core generators returned duplicate semantic core IDs")
    _validate_core_diagnostic_schedule(cores, config)
    return cores


def _select_and_assign(
    cores: tuple[SemanticCore, ...],
    *,
    seed: int,
    splits: SplitConfig,
    cores_per_family: int | None,
) -> tuple[CoreSplitAssignment, ...]:
    assignments = []
    for family in _CORE_FAMILIES:
        ranked = []
        for core in cores:
            if core.task_family is not family:
                continue
            strata = _resolved_strata(core)
            ranking = _ranking_sha256(core, strata, seed)
            ranked.append((ranking, core.core_id, core, strata))
        ranked.sort(key=lambda item: (item[0], item[1]))
        selected = ranked[:cores_per_family] if cores_per_family is not None else ranked
        quotas = _split_quotas(len(selected), splits)
        offset = 0
        for split, quota in zip(_SPLIT_ORDER, quotas, strict=True):
            for ranking, _, core, strata in selected[offset : offset + quota]:
                assignments.append(
                    CoreSplitAssignment(
                        semantic_core_id=core.core_id,
                        task_family=core.task_family,
                        difficulty=core.difficulty,
                        strata=strata,
                        split=split,
                        ranking_sha256=ranking,
                    )
                )
            offset += quota
    return tuple(assignments)


def _validate_snapshot(snapshot: CompiledCoreSnapshot, config: CoreConfig) -> None:
    assignment_by_core = {
        assignment.semantic_core_id: assignment for assignment in snapshot.assignments
    }
    if len(assignment_by_core) != len(snapshot.assignments):
        raise ValueError("Core snapshot contains duplicate split assignments")
    tasks_by_core: dict[str, list[MemUpdateTaskV3]] = defaultdict(list)
    for task in snapshot.tasks:
        core_id = task.metadata.split_key.semantic_core_id
        if core_id not in assignment_by_core:
            raise ValueError("Core snapshot task has no split assignment")
        if task.metadata.split is not assignment_by_core[core_id].split:
            raise ValueError("Core snapshot surface is not co-located with its core")
        tasks_by_core[core_id].append(task)
    if set(tasks_by_core) != set(assignment_by_core):
        raise ValueError("Core snapshot assignments and rendered cores differ")

    all_task_ids = []
    all_raw_hashes = []
    for core_id, tasks in tasks_by_core.items():
        if len(tasks) != _VARIANTS_PER_CORE:
            raise ValueError(f"Core {core_id} does not have exactly four surfaces")
        variants = {task.metadata.extra["surface_variant"] for task in tasks}
        if variants != set(range(_VARIANTS_PER_CORE)):
            raise ValueError(f"Core {core_id} has invalid surface variants")
        task_ids = {task.task_id for task in tasks}
        raw_hashes = {task.source.raw_hash for task in tasks}
        semantic_hashes = {semantic_task_hash_v3(task) for task in tasks}
        if len(task_ids) != 4 or len(raw_hashes) != 4 or len(semantic_hashes) != 1:
            raise ValueError(f"Core {core_id} surface identity validation failed")
        all_task_ids.extend(task_ids)
        all_raw_hashes.extend(raw_hashes)
    if len(all_task_ids) != len(set(all_task_ids)):
        raise ValueError("Core snapshot task IDs must be globally unique")
    if len(all_raw_hashes) != len(set(all_raw_hashes)):
        raise ValueError("Core snapshot raw source hashes must be globally unique")

    split_key_fields = (
        "semantic_core_id",
        "source_group_id",
        "trajectory_id",
        "paraphrase_group_id",
        "version_group_id",
    )
    for field in split_key_fields:
        values = {
            split: {
                getattr(task.metadata.split_key, field)
                for task in snapshot.tasks
                if task.metadata.split is split
            }
            for split in _SPLIT_ORDER
        }
        for index, left in enumerate(_SPLIT_ORDER):
            for right in _SPLIT_ORDER[index + 1 :]:
                if not values[left].isdisjoint(values[right]):
                    raise ValueError(f"Core snapshot has cross-split {field} overlap")
    hashes = {
        split: {
            semantic_task_hash_v3(task)
            for task in snapshot.tasks
            if task.metadata.split is split
        }
        for split in _SPLIT_ORDER
    }
    for index, left in enumerate(_SPLIT_ORDER):
        for right in _SPLIT_ORDER[index + 1 :]:
            if not hashes[left].isdisjoint(hashes[right]):
                raise ValueError("Core snapshot has cross-split semantic hash overlap")

    observed_core_counts = Counter(assignment.split.value for assignment in snapshot.assignments)
    observed_task_counts = Counter(task.metadata.split.value for task in snapshot.tasks)
    if dict(snapshot.core_counts) != {
        split.value: observed_core_counts[split.value] for split in _SPLIT_ORDER
    }:
        raise ValueError("Core snapshot core counts are inconsistent")
    if dict(snapshot.task_counts) != {
        split.value: observed_task_counts[split.value] for split in _SPLIT_ORDER
    }:
        raise ValueError("Core snapshot task counts are inconsistent")
    if len(snapshot.tasks) != len(snapshot.assignments) * _VARIANTS_PER_CORE:
        raise ValueError("Core snapshot total task count is inconsistent")
    observed_family_split_counts = Counter(
        (assignment.task_family.value, assignment.split.value)
        for assignment in snapshot.assignments
    )
    for family in _CORE_FAMILIES:
        family_name = family.value
        selected_count = sum(
            observed_family_split_counts[(family_name, split.value)]
            for split in _SPLIT_ORDER
        )
        if selected_count == config.family_core_counts[family_name]:
            family_config = getattr(config.families, family_name)
            expected = family_config.schedule.split_core_counts.model_dump()
        else:
            expected = dict(
                zip(
                    (split.value for split in _SPLIT_ORDER),
                    _split_quotas(selected_count, config.splits),
                    strict=True,
                )
            )
        observed = {
            split.value: observed_family_split_counts[(family_name, split.value)]
            for split in _SPLIT_ORDER
        }
        if observed != expected:
            raise ValueError("Core snapshot per-family split schedule is invalid")


def compile_core_snapshot(
    config: CoreConfig,
    *,
    cores_per_family: int | None = None,
    code_revision: str = "in-memory-core-snapshot-v1",
) -> CompiledCoreSnapshot:
    """Compile a deterministic, immutable Core A-D snapshot without publication."""
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    if type(code_revision) is not str:
        raise TypeError("code_revision must be a string")
    if not code_revision.strip():
        raise ValueError("code_revision must not be blank")
    limit = _validate_limit(config, cores_per_family)
    context = GenerationContext(
        config=config,
        code_revision=code_revision,
        generator_name="memupdatebench_vnext_core",
    )
    cores = _generated_cores(config)
    assignments = _select_and_assign(
        cores,
        seed=config.seed,
        splits=config.splits,
        cores_per_family=limit,
    )
    core_by_id = {core.core_id: core for core in cores}
    tasks = tuple(
        render_core_v3(
            core_by_id[assignment.semantic_core_id],
            split=assignment.split,
            surface_variant=surface_variant,
            context=context,
        )
        for assignment in assignments
        for surface_variant in range(_VARIANTS_PER_CORE)
    )
    family_counts = Counter(assignment.task_family.value for assignment in assignments)
    core_counts = Counter(assignment.split.value for assignment in assignments)
    task_counts = Counter(task.metadata.split.value for task in tasks)
    snapshot = CompiledCoreSnapshot(
        config_sha256=context.config_sha256,
        assignments=assignments,
        tasks=tasks,
        family_core_counts={family.value: family_counts[family.value] for family in _CORE_FAMILIES},
        core_counts={split.value: core_counts[split.value] for split in _SPLIT_ORDER},
        task_counts={split.value: task_counts[split.value] for split in _SPLIT_ORDER},
    )
    _validate_snapshot(snapshot, config)
    return snapshot


__all__ = ["CompiledCoreSnapshot", "compile_core_snapshot"]
