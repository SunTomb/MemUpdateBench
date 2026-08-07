from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
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
from mub.vnext.generation.family_e import (
    generate_core_family_e_cores,
    validate_family_e_core,
)
from mub.vnext.generation.family_f import (
    generate_core_family_f_cores,
    validate_family_f_full_core,
)
from mub.vnext.generation.family_g import (
    generate_core_family_g_cores,
    validate_family_g_full_core,
)
from mub.vnext.generation.render import _resolve_core_profile
from mub.vnext.generation.splits import (
    CoreSplitAssignment,
    _ranking_sha256,
    _resolved_strata,
)
from mub.vnext.io import semantic_task_hash_v3, sha256_model


_CORE_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
    TaskFamily.DELETION_FORGETTING,
    TaskFamily.CURRENT_HISTORICAL_QUERY,
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS,
)
_SPLIT_ORDER = (Split.TRAIN, Split.DEV, Split.TEST)
_VARIANTS_PER_CORE = 4
HashString = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


@dataclass(frozen=True)
class _DiagnosticRecord:
    task_family: TaskFamily
    profile: Mapping[str, object]
    stratification: Mapping[str, object]


class CompiledCoreSnapshot(ImmutableContractModel):
    config_sha256: HashString
    assignments: tuple[CoreSplitAssignment, ...]
    semantic_cores: tuple[SemanticCore, ...]
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
    cores: tuple[SemanticCore | _DiagnosticRecord, ...],
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
    b_patterns_within_active = Counter(
        (
            core.stratification.get("active_object_count"),
            core.stratification.get("interleaving_pattern"),
        )
        for core in b_cores
    )
    expected_b_patterns_within_active = {
        (active_count, pattern): (
            family_b.schedule.cores_per_pattern_within_active_object_count
        )
        for active_count in family_b.active_object_counts
        for pattern in family_b.interleaving_patterns
    }
    if dict(b_patterns_within_active) != expected_b_patterns_within_active:
        raise ValueError(
            "Core Family B schedule pattern-within-active-object quota is invalid"
        )
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

    family_e = config.families.deletion_forgetting
    e_cores = by_family[TaskFamily.DELETION_FORGETTING]
    for core in e_cores:
        validate_family_e_core(core)
    e_cells = Counter(core.stratification.get("lifecycle_cell") for core in e_cores)
    if dict(e_cells) != {
        cell: family_e.schedule.cores_per_lifecycle_cell
        for cell in family_e.lifecycle_cells
    }:
        raise ValueError("Core Family E schedule lifecycle cells are invalid")
    e_difficulties = Counter(core.difficulty.value for core in e_cores)
    if dict(e_difficulties) != {
        difficulty: family_e.schedule.cores_per_difficulty
        for difficulty in family_e.difficulties
    }:
        raise ValueError("Core Family E schedule difficulty marginal is invalid")
    e_positions = Counter(
        core.stratification.get("deletion_position") for core in e_cores
    )
    expected_e_positions = {
        position: family_e.schedule.cores_per_deletion_position
        for position in family_e.deletion_positions
    }
    expected_e_positions["not_applicable"] = (
        family_e.schedule.non_deletion_hard_negative_count
    )
    if dict(e_positions) != expected_e_positions:
        raise ValueError("Core Family E schedule deletion-position marginal is invalid")

    family_f = config.families.current_historical_query
    f_cores = by_family[TaskFamily.CURRENT_HISTORICAL_QUERY]
    for core in f_cores:
        validate_family_f_full_core(core, config)
    f_trajectories = Counter(core.trajectory_id for core in f_cores)
    if len(f_trajectories) != family_f.schedule.trajectory_count or set(
        f_trajectories.values()
    ) != {family_f.schedule.selectors_per_trajectory}:
        raise ValueError("Core Family F trajectory schedule is invalid")

    family_g = config.families.long_horizon_memory_synthesis
    g_cores = by_family[TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS]
    for core in g_cores:
        validate_family_g_full_core(core)
    g_kinds = Counter(core.stratification.get("synthesis_kind") for core in g_cores)
    if dict(g_kinds) != {
        "update_sensitive_multi_hop": family_g.schedule.update_sensitive_core_count,
        "multi_object_current_consistency": family_g.schedule.consistency_core_count,
    }:
        raise ValueError("Core Family G synthesis-kind schedule is invalid")
    g_hops = Counter(
        core.stratification.get("hop_count")
        for core in g_cores
        if core.stratification.get("synthesis_kind") == "update_sensitive_multi_hop"
    )
    if dict(g_hops) != {
        hop: family_g.schedule.cores_per_hop_count for hop in family_g.hop_counts
    }:
        raise ValueError("Core Family G hop-count schedule is invalid")
    g_consistency = Counter(
        (
            core.stratification.get("object_count"),
            core.stratification.get("answer_kind"),
        )
        for core in g_cores
        if core.stratification.get("synthesis_kind")
        == "multi_object_current_consistency"
    )
    if dict(g_consistency) != {
        (count, answer_kind): family_g.schedule.cores_per_object_answer_kind_cell
        for count in family_g.consistency_object_counts
        for answer_kind in family_g.consistency_answer_kinds
    }:
        raise ValueError("Core Family G consistency schedule is invalid")
    fingerprints = {
        core.stratification.get("evidence_fingerprint") for core in g_cores
    }
    evidence_groups = {
        core.stratification.get("evidence_group_id") for core in g_cores
    }
    if len(fingerprints) != len(g_cores) or len(evidence_groups) != len(g_cores):
        raise ValueError("Core Family G evidence identities must be unique")


def _generated_cores(config: CoreConfig) -> tuple[SemanticCore, ...]:
    families = (
        generate_core_family_a_cores(config),
        generate_core_family_b_cores(config),
        generate_core_family_c_cores(config),
        generate_core_family_d_cores(config),
        generate_core_family_e_cores(config, profile="full"),
        generate_core_family_f_cores(config, profile="full"),
        generate_core_family_g_cores(config, profile="full"),
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
        if cores_per_family is not None:
            grouped = defaultdict(list)
            for item in ranked:
                grouped[item[2].trajectory_id].append(item)
            if cores_per_family <= len(grouped):
                selected = []
                selected_trajectories = set()
                for item in ranked:
                    trajectory_id = item[2].trajectory_id
                    if trajectory_id in selected_trajectories:
                        continue
                    selected.append(item)
                    selected_trajectories.add(trajectory_id)
                    if len(selected) == cores_per_family:
                        break
                quotas = _split_quotas(len(selected), splits)
                offset = 0
                for split, quota in zip(_SPLIT_ORDER, quotas, strict=True):
                    for ranking, _, core, strata in selected[
                        offset : offset + quota
                    ]:
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
                continue

            group_sizes = {len(group) for group in grouped.values()}
            if len(group_sizes) != 1:
                raise ValueError(
                    "partial Core selection requires uniform complete trajectory groups"
                )
            group_size = next(iter(group_sizes))
            quotas = _split_quotas(cores_per_family, splits)
            if any(quota % group_size for quota in quotas):
                raise ValueError(
                    "partial Core split quota cannot preserve complete trajectory groups"
                )
            ranked_groups = sorted(
                grouped.values(),
                key=lambda group: (
                    min(item[0] for item in group),
                    group[0][2].trajectory_id,
                ),
            )
            required_groups = cores_per_family // group_size
            if len(ranked_groups) < required_groups:
                raise ValueError(
                    "partial Core selection cannot satisfy complete trajectory quota"
                )
            group_offset = 0
            for split, quota in zip(_SPLIT_ORDER, quotas, strict=True):
                split_groups = quota // group_size
                for group in ranked_groups[
                    group_offset : group_offset + split_groups
                ]:
                    for ranking, _, core, strata in group:
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
                group_offset += split_groups
            continue

        grouped = defaultdict(list)
        for item in ranked:
            grouped[item[2].trajectory_id].append(item)
        ranked_groups = sorted(
            grouped.values(),
            key=lambda group: (min(item[0] for item in group), group[0][2].trajectory_id),
        )
        quotas = _split_quotas(len(ranked), splits)
        group_offset = 0
        for split, quota in zip(_SPLIT_ORDER, quotas, strict=True):
            allocated = 0
            while allocated < quota and group_offset < len(ranked_groups):
                group = ranked_groups[group_offset]
                if allocated + len(group) > quota:
                    raise ValueError(
                        "Core split quota cannot be filled by complete trajectory groups"
                    )
                for ranking, _, core, strata in group:
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
                allocated += len(group)
                group_offset += 1
            if allocated != quota:
                raise ValueError("Core split quota was not filled exactly")
    return tuple(assignments)


def _validate_snapshot(
    snapshot: CompiledCoreSnapshot,
    config: CoreConfig,
    expected_cores: tuple[SemanticCore, ...] | None = None,
) -> None:
    canonical_config_hash = sha256_model(config)
    if snapshot.config_sha256 != canonical_config_hash or any(
        task.metadata.generation_config_hash != canonical_config_hash
        for task in snapshot.tasks
    ):
        raise ValueError("Core snapshot does not match the canonical config hash")
    if any(
        task.source.generator.config_sha256 != canonical_config_hash
        for task in snapshot.tasks
    ):
        raise ValueError(
            "Core snapshot source generator config hash is not canonical"
        )

    assignment_by_core = {
        assignment.semantic_core_id: assignment for assignment in snapshot.assignments
    }
    if len(assignment_by_core) != len(snapshot.assignments):
        raise ValueError("Core snapshot contains duplicate split assignments")
    if not snapshot.assignments:
        raise ValueError("Core snapshot requires a positive cores_per_family quota")
    semantic_core_ids = tuple(
        core.core_id for core in snapshot.semantic_cores
    )
    assignment_core_ids = tuple(
        assignment.semantic_core_id for assignment in snapshot.assignments
    )
    if (
        len(semantic_core_ids) != len(set(semantic_core_ids))
        or semantic_core_ids != assignment_core_ids
    ):
        raise ValueError(
            "Core snapshot semantic cores must be unique and follow assignments exactly"
        )
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
        "source_document_id",
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
    normalized_hashes = {
        split: {
            task.source.normalized_hash
            for task in snapshot.tasks
            if task.metadata.split is split
        }
        for split in _SPLIT_ORDER
    }
    evidence_fingerprints = {
        split: {
            task.metadata.extra["stratification"].get("evidence_fingerprint")
            for task in snapshot.tasks
            if task.metadata.split is split
            and task.task_family == TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value
        }
        for split in _SPLIT_ORDER
    }
    for values, label in (
        (normalized_hashes, "normalized source hash"),
        (evidence_fingerprints, "Family G evidence fingerprint"),
    ):
        for index, left in enumerate(_SPLIT_ORDER):
            for right in _SPLIT_ORDER[index + 1 :]:
                if not values[left].isdisjoint(values[right]):
                    raise ValueError(f"Core snapshot has cross-split {label} overlap")

    observed_family_core_counts = Counter(
        assignment.task_family.value for assignment in snapshot.assignments
    )
    expected_family_core_counts = {
        family.value: observed_family_core_counts[family.value]
        for family in _CORE_FAMILIES
    }
    if dict(snapshot.family_core_counts) != expected_family_core_counts:
        raise ValueError("Core snapshot family counts are inconsistent")

    canonical_cores = expected_cores if expected_cores is not None else _generated_cores(config)
    _validate_core_diagnostic_schedule(canonical_cores, config)
    if expected_family_core_counts == config.family_core_counts:
        selection_limit = None
    else:
        partial_family_quotas = set(expected_family_core_counts.values())
        if len(partial_family_quotas) != 1:
            raise ValueError(
                "Partial Core snapshot families must share one common "
                "cores_per_family quota"
            )
        selection_limit = next(iter(partial_family_quotas))
        if selection_limit <= 0:
            raise ValueError("Core snapshot requires a positive cores_per_family quota")
    canonical_assignments = _select_and_assign(
        canonical_cores,
        seed=config.seed,
        splits=config.splits,
        cores_per_family=selection_limit,
    )
    canonical_assignment_by_core = {
        assignment.semantic_core_id: assignment
        for assignment in canonical_assignments
    }
    if assignment_by_core != canonical_assignment_by_core:
        raise ValueError(
            "Core snapshot selected semantic core IDs or canonical Core split "
            "assignments are invalid"
        )

    diagnostic_records = {}
    for core_id, tasks in tasks_by_core.items():
        diagnostics = []
        for task in tasks:
            stratification = task.metadata.extra.get("stratification")
            profile = task.metadata.resolved_profile
            if not isinstance(stratification, Mapping) or not isinstance(profile, Mapping):
                raise ValueError("Core snapshot diagnostic metadata is invalid")
            diagnostics.append((profile, stratification))
        if any(diagnostic != diagnostics[0] for diagnostic in diagnostics[1:]):
            raise ValueError(
                f"Core {core_id} has inconsistent diagnostic metadata across surfaces"
            )
        representative = next(
            task
            for task in tasks
            if task.metadata.extra.get("surface_variant") == 0
        )
        diagnostic_records[core_id] = _DiagnosticRecord(
            task_family=assignment_by_core[core_id].task_family,
            profile=representative.metadata.resolved_profile,
            stratification=representative.metadata.extra["stratification"],
        )

    canonical_by_id = {core.core_id: core for core in canonical_cores}
    if len(canonical_by_id) != len(canonical_cores):
        raise ValueError("Expected Core diagnostics contain duplicate semantic core IDs")
    if any(
        canonical_by_id.get(core.core_id) != core
        for core in snapshot.semantic_cores
    ):
        raise ValueError(
            "Core snapshot semantic core payload is not canonical"
        )
    for core_id, record in diagnostic_records.items():
        expected = canonical_by_id.get(core_id)
        if expected is None:
            raise ValueError("Core snapshot contains an unknown semantic core ID")
        family_label = chr(ord("A") + _CORE_FAMILIES.index(expected.task_family))
        expected_profile = dict(
            _resolve_core_profile(expected, expected.query_type)
        )
        if (
            expected.task_family is TaskFamily.DELETION_FORGETTING
            and expected.stratification.get("lifecycle_cell")
            != "correction_versus_deletion_hard_negative"
        ):
            expected_profile["query_type"] = "deletion_compliance"
        profile_matches = dict(record.profile) == expected_profile
        if record.task_family is not expected.task_family or not profile_matches:
            raise ValueError(
                "Core snapshot canonical resolved profile does not match "
                f"Core Family {family_label} schedule"
            )
        if dict(record.stratification) != dict(expected.stratification):
            raise ValueError(
                "Core snapshot diagnostic metadata does not match "
                f"Core Family {family_label} schedule"
            )

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
        semantic_cores=tuple(
            core_by_id[assignment.semantic_core_id] for assignment in assignments
        ),
        tasks=tasks,
        family_core_counts={family.value: family_counts[family.value] for family in _CORE_FAMILIES},
        core_counts={split.value: core_counts[split.value] for split in _SPLIT_ORDER},
        task_counts={split.value: task_counts[split.value] for split in _SPLIT_ORDER},
    )
    _validate_snapshot(snapshot, config, cores)
    return snapshot


__all__ = ["CompiledCoreSnapshot", "compile_core_snapshot"]
