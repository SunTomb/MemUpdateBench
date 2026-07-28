from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from pydantic import Field, JsonValue, field_validator

from mub.vnext.contracts.common import (
    FrozenJsonObject,
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    StrictNonnegativeInt,
    StrictNumericScore,
    freeze_json,
    freeze_mapping,
)
from mub.vnext.contracts.enums import Difficulty, Split, TaskFamily
from mub.vnext.generation.core import SemanticCore
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.profiles import build_generic_profile
from mub.vnext.validation.split import FAMILY_STRATIFICATION_AXES


_CORES_PER_FAMILY = 120
_VARIANTS_PER_CORE = 3
_SPLIT_QUOTAS = {
    Split.TRAIN: 84,
    Split.DEV: 12,
    Split.TEST: 24,
}
_SPLIT_ORDER = (Split.TRAIN, Split.DEV, Split.TEST)
_PILOT_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)
_FAMILY_INDEX = {family: index for index, family in enumerate(_PILOT_FAMILIES)}
_DIFFICULTY_INDEX = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 2,
}
_UPDATE_DEPTH_BUCKETS = frozenset({"1", "2-3", "4-7", "8-15", "16+"})
StrictNonnegativeFloat = Annotated[
    float,
    Field(ge=0.0, strict=True, allow_inf_nan=False),
]


class CoreSplitAssignment(ImmutableContractModel):
    semantic_core_id: str = Field(min_length=1, strict=True)
    task_family: TaskFamily
    difficulty: Difficulty
    strata: FrozenJsonObject
    split: Split
    ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)

    @field_validator("semantic_core_id")
    @classmethod
    def _reject_blank_core_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("semantic_core_id must not be blank")
        return value

    @field_validator("strata")
    @classmethod
    def _freeze_strata(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)


class SplitBalanceCell(ImmutableContractModel):
    task_family: TaskFamily
    difficulty: Difficulty
    strata: FrozenJsonObject
    split: Split
    expected: StrictNonnegativeFloat
    observed: StrictNonnegativeInt
    deviation: StrictNumericScore
    total: StrictNonnegativeInt

    @field_validator("strata")
    @classmethod
    def _freeze_strata(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)


class SplitBalanceReport(ImmutableContractModel):
    seed: StrictNonnegativeInt
    core_counts: FrozenNonnegativeIntMap
    projected_task_counts: FrozenNonnegativeIntMap
    cells: tuple[SplitBalanceCell, ...]

    @field_validator("core_counts", "projected_task_counts")
    @classmethod
    def _freeze_counts(cls, value: Mapping[str, int]):
        return freeze_mapping(value)


class SplitAssignmentResult(ImmutableContractModel):
    assignments: tuple[CoreSplitAssignment, ...]
    split_balance: SplitBalanceReport


class _RankingMaterial(ImmutableContractModel):
    seed: StrictNonnegativeInt
    task_family: TaskFamily
    difficulty: Difficulty
    strata: FrozenJsonObject
    semantic_core_id: str

    @field_validator("strata")
    @classmethod
    def _freeze_strata(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)


class _StratumMaterial(ImmutableContractModel):
    task_family: TaskFamily
    difficulty: Difficulty
    strata: FrozenJsonObject

    @field_validator("strata")
    @classmethod
    def _freeze_strata(cls, value: Mapping[str, JsonValue]):
        return freeze_json(value)


def _derive_update_depth_bucket(value: Any) -> str | None:
    if type(value) is not int or value <= 0:
        return None
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    if value <= 15:
        return "8-15"
    return "16+"


def _axis_value(core: SemanticCore, axis: str) -> str | int | float | bool:
    if axis == "update_depth_bucket":
        raw_bucket = core.profile.get(axis)
        derived_bucket = _derive_update_depth_bucket(core.profile.get("update_depth"))
        if raw_bucket is not None:
            if type(raw_bucket) is not str or raw_bucket not in _UPDATE_DEPTH_BUCKETS:
                raise ValueError(
                    f"core {core.core_id} has invalid update_depth_bucket"
                )
            if derived_bucket is not None and raw_bucket != derived_bucket:
                raise ValueError(
                    f"core {core.core_id} has inconsistent update_depth_bucket"
                )
            return raw_bucket
        if derived_bucket is None:
            raise ValueError(
                f"core {core.core_id} cannot resolve update_depth_bucket"
            )
        return derived_bucket

    if axis in core.profile:
        value = core.profile[axis]
    else:
        resolved_defaults = build_generic_profile(
            core.difficulty,
            core.task_family.value,
        ).parameters
        if axis not in resolved_defaults:
            raise ValueError(
                f"core {core.core_id} is missing authoritative stratum axis {axis}"
            )
        value = resolved_defaults[axis]
    if type(value) not in {str, int, float, bool}:
        raise ValueError(
            f"core {core.core_id} has non-scalar stratum axis {axis}"
        )
    if type(value) is str and not value.strip():
        raise ValueError(f"core {core.core_id} has blank stratum axis {axis}")
    return value


def _resolved_strata(core: SemanticCore) -> dict[str, str | int | float | bool]:
    axes = FAMILY_STRATIFICATION_AXES[core.task_family.value]
    return {axis: _axis_value(core, axis) for axis in axes}


def _ranking_sha256_from_material(
    *,
    seed: int,
    task_family: TaskFamily,
    difficulty: Difficulty,
    strata: Mapping[str, str | int | float | bool],
    semantic_core_id: str,
) -> str:
    material = _RankingMaterial(
        seed=seed,
        task_family=task_family,
        difficulty=difficulty,
        strata=dict(strata),
        semantic_core_id=semantic_core_id,
    )
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _ranking_sha256(
    core: SemanticCore,
    strata: Mapping[str, str | int | float | bool],
    seed: int,
) -> str:
    return _ranking_sha256_from_material(
        seed=seed,
        task_family=core.task_family,
        difficulty=core.difficulty,
        strata=strata,
        semantic_core_id=core.core_id,
    )


def _stratum_sort_key(
    family: TaskFamily,
    difficulty: Difficulty,
    strata: Mapping[str, str | int | float | bool],
) -> bytes:
    return canonical_json_bytes(
        _StratumMaterial(
            task_family=family,
            difficulty=difficulty,
            strata=dict(strata),
        )
    )


def _allocation_cost(total: int, allocation: tuple[int, int, int]) -> int:
    return sum(
        abs(observed * _CORES_PER_FAMILY - total * _SPLIT_QUOTAS[split])
        for split, observed in zip(_SPLIT_ORDER, allocation, strict=True)
    )


def _optimal_allocations(cell_sizes: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    target_train = _SPLIT_QUOTAS[Split.TRAIN]
    target_dev = _SPLIT_QUOTAS[Split.DEV]
    target_test = _SPLIT_QUOTAS[Split.TEST]
    processed = 0
    states: dict[
        tuple[int, int],
        tuple[int, tuple[tuple[int, int, int], ...]],
    ] = {(0, 0): (0, ())}

    for total in cell_sizes:
        processed += total
        remaining = _CORES_PER_FAMILY - processed
        next_states: dict[
            tuple[int, int],
            tuple[int, tuple[tuple[int, int, int], ...]],
        ] = {}
        for (used_train, used_dev), (cost, allocations) in states.items():
            used_test = processed - total - used_train - used_dev
            for train_count in range(total + 1):
                for dev_count in range(total - train_count + 1):
                    test_count = total - train_count - dev_count
                    new_train = used_train + train_count
                    new_dev = used_dev + dev_count
                    new_test = used_test + test_count
                    if not (
                        new_train <= target_train <= new_train + remaining
                        and new_dev <= target_dev <= new_dev + remaining
                        and new_test <= target_test <= new_test + remaining
                    ):
                        continue
                    allocation = (train_count, dev_count, test_count)
                    candidate = (
                        cost + _allocation_cost(total, allocation),
                        (*allocations, allocation),
                    )
                    key = (new_train, new_dev)
                    incumbent = next_states.get(key)
                    if incumbent is None or candidate < incumbent:
                        next_states[key] = candidate
        states = next_states

    final = states.get((target_train, target_dev))
    if final is None:
        raise ValueError("unable to satisfy exact Pilot split quotas")
    return final[1]


def _validate_inputs(cores: Iterable[SemanticCore], seed: int) -> tuple[SemanticCore, ...]:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    try:
        records = tuple(cores)
    except TypeError as exc:
        raise TypeError("cores must be an iterable of SemanticCore records") from exc
    if any(not isinstance(core, SemanticCore) for core in records):
        raise TypeError("cores must contain only SemanticCore records")

    core_ids = [core.core_id for core in records]
    duplicate_ids = sorted(
        core_id for core_id, count in Counter(core_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate semantic_core_id records are not allowed: "
            + ", ".join(duplicate_ids)
        )

    unsupported = sorted(
        {
            core.task_family.value
            for core in records
            if core.task_family not in _PILOT_FAMILIES
        }
    )
    counts = Counter(core.task_family for core in records)
    if unsupported or any(counts[family] != _CORES_PER_FAMILY for family in _PILOT_FAMILIES):
        observed = ", ".join(
            f"{family.value}={counts[family]}" for family in _PILOT_FAMILIES
        )
        if unsupported:
            observed += f"; unsupported={unsupported}"
        raise ValueError(
            "assign_splits requires exactly 120 unique cores per Pilot family; "
            f"observed {observed}"
        )
    if any(core.difficulty not in _DIFFICULTY_INDEX for core in records):
        raise ValueError("Pilot split cores must use easy, medium, or hard difficulty")
    return records


def _validate_split_assignment_result(result: SplitAssignmentResult) -> None:
    if not isinstance(result, SplitAssignmentResult):
        raise TypeError("result must be a SplitAssignmentResult")
    assignments = result.assignments
    seed = result.split_balance.seed
    issues: list[str] = []

    core_ids = [assignment.semantic_core_id for assignment in assignments]
    if len(assignments) != len(_PILOT_FAMILIES) * _CORES_PER_FAMILY:
        issues.append(f"assignment_count={len(assignments)}")
    if len(set(core_ids)) != len(core_ids):
        issues.append("duplicate_semantic_core_id")

    expected_assignment_order = tuple(
        sorted(
            assignments,
            key=lambda assignment: (
                _FAMILY_INDEX.get(assignment.task_family, len(_FAMILY_INDEX)),
                _DIFFICULTY_INDEX.get(
                    assignment.difficulty, len(_DIFFICULTY_INDEX)
                ),
                _stratum_sort_key(
                    assignment.task_family,
                    assignment.difficulty,
                    assignment.strata,
                ),
                assignment.semantic_core_id,
            ),
        )
    )
    if assignments != expected_assignment_order:
        issues.append("noncanonical_assignment_order")

    family_counts = Counter(assignment.task_family for assignment in assignments)
    split_counts = Counter(assignment.split for assignment in assignments)
    for family in _PILOT_FAMILIES:
        if family_counts[family] != _CORES_PER_FAMILY:
            issues.append(f"family_count:{family.value}={family_counts[family]}")
        per_family = Counter(
            assignment.split
            for assignment in assignments
            if assignment.task_family is family
        )
        for split in _SPLIT_ORDER:
            if per_family[split] != _SPLIT_QUOTAS[split]:
                issues.append(
                    f"family_split_count:{family.value}:{split.value}="
                    f"{per_family[split]}"
                )
    for split in _SPLIT_ORDER:
        expected = _SPLIT_QUOTAS[split] * len(_PILOT_FAMILIES)
        if split_counts[split] != expected:
            issues.append(f"global_split_count:{split.value}={split_counts[split]}")

    expected_cells: list[SplitBalanceCell] = []
    for family in _PILOT_FAMILIES:
        grouped: dict[bytes, list[CoreSplitAssignment]] = defaultdict(list)
        for assignment in assignments:
            if assignment.task_family is not family:
                continue
            expected_axes = FAMILY_STRATIFICATION_AXES[family.value]
            if tuple(assignment.strata) != expected_axes:
                issues.append(
                    f"strata_axes:{assignment.semantic_core_id}="
                    f"{tuple(assignment.strata)!r}"
                )
            expected_ranking = _ranking_sha256_from_material(
                seed=seed,
                task_family=assignment.task_family,
                difficulty=assignment.difficulty,
                strata=assignment.strata,
                semantic_core_id=assignment.semantic_core_id,
            )
            if assignment.ranking_sha256 != expected_ranking:
                issues.append(f"ranking_sha256:{assignment.semantic_core_id}")
            grouped[
                _stratum_sort_key(
                    assignment.task_family,
                    assignment.difficulty,
                    assignment.strata,
                )
            ].append(assignment)

        ordered_keys = sorted(grouped)
        allocations = _optimal_allocations(
            tuple(len(grouped[key]) for key in ordered_keys)
        )
        for key, allocation in zip(ordered_keys, allocations, strict=True):
            cell = grouped[key]
            ranked = sorted(
                cell,
                key=lambda assignment: (
                    assignment.ranking_sha256,
                    assignment.semantic_core_id,
                ),
            )
            offset = 0
            exemplar = cell[0]
            for split, observed in zip(_SPLIT_ORDER, allocation, strict=True):
                selected = ranked[offset : offset + observed]
                offset += observed
                if any(assignment.split is not split for assignment in selected):
                    issues.append(
                        f"noncontiguous_rank_segment:{family.value}:"
                        f"{exemplar.difficulty.value}:{split.value}:"
                        f"{canonical_json_bytes(_StratumMaterial(task_family=family, difficulty=exemplar.difficulty, strata=exemplar.strata)).hex()}"
                    )
                expected = len(ranked) * _SPLIT_QUOTAS[split] / _CORES_PER_FAMILY
                expected_cells.append(
                    SplitBalanceCell(
                        task_family=family,
                        difficulty=exemplar.difficulty,
                        strata=dict(exemplar.strata),
                        split=split,
                        expected=float(expected),
                        observed=observed,
                        deviation=float(observed - expected),
                        total=len(ranked),
                    )
                )

    expected_core_counts = {
        split.value: split_counts[split] for split in _SPLIT_ORDER
    }
    expected_projected_counts = {
        split.value: split_counts[split] * _VARIANTS_PER_CORE
        for split in _SPLIT_ORDER
    }
    if dict(result.split_balance.core_counts) != expected_core_counts:
        issues.append("split_balance_core_counts")
    if dict(result.split_balance.projected_task_counts) != expected_projected_counts:
        issues.append("split_balance_projected_task_counts")
    if result.split_balance.cells != tuple(expected_cells):
        issues.append("split_balance_cells")
    if issues:
        ordered = list(dict.fromkeys(issues))
        raise ValueError("inconsistent split assignment: " + "; ".join(ordered[:64]))


def assign_splits(
    cores: Iterable[SemanticCore],
    seed: int,
) -> SplitAssignmentResult:
    """Assign all Pilot semantic cores to exact, deterministic, group-safe splits."""
    records = _validate_inputs(cores, seed)
    assignments: list[CoreSplitAssignment] = []
    balance_cells: list[SplitBalanceCell] = []

    for family in _PILOT_FAMILIES:
        grouped: dict[
            bytes,
            list[tuple[SemanticCore, dict[str, str | int | float | bool], str]],
        ] = defaultdict(list)
        group_metadata: dict[
            bytes,
            tuple[Difficulty, dict[str, str | int | float | bool]],
        ] = {}
        for core in records:
            if core.task_family is not family:
                continue
            strata = _resolved_strata(core)
            ranking_sha256 = _ranking_sha256(core, strata, seed)
            key = _stratum_sort_key(family, core.difficulty, strata)
            grouped[key].append((core, strata, ranking_sha256))
            group_metadata.setdefault(key, (core.difficulty, strata))

        ordered_cells = []
        for key in sorted(grouped):
            difficulty, strata = group_metadata[key]
            ordered_cells.append(
                (
                    (difficulty, tuple(strata.items())),
                    grouped[key],
                )
            )
        allocations = _optimal_allocations(
            tuple(len(cell_records) for _, cell_records in ordered_cells)
        )

        for ((difficulty, strata_items), cell_records), allocation in zip(
            ordered_cells,
            allocations,
            strict=True,
        ):
            strata = dict(strata_items)
            ranked = sorted(cell_records, key=lambda item: (item[2], item[0].core_id))
            offset = 0
            for split, observed in zip(_SPLIT_ORDER, allocation, strict=True):
                selected = ranked[offset : offset + observed]
                offset += observed
                assignments.extend(
                    CoreSplitAssignment(
                        semantic_core_id=core.core_id,
                        task_family=family,
                        difficulty=difficulty,
                        strata=dict(core_strata),
                        split=split,
                        ranking_sha256=ranking_sha256,
                    )
                    for core, core_strata, ranking_sha256 in selected
                )
                expected = len(ranked) * _SPLIT_QUOTAS[split] / _CORES_PER_FAMILY
                balance_cells.append(
                    SplitBalanceCell(
                        task_family=family,
                        difficulty=difficulty,
                        strata=dict(strata),
                        split=split,
                        expected=float(expected),
                        observed=observed,
                        deviation=float(observed - expected),
                        total=len(ranked),
                    )
                )

    assignments.sort(
        key=lambda assignment: (
            _FAMILY_INDEX[assignment.task_family],
            _DIFFICULTY_INDEX[assignment.difficulty],
            canonical_json_bytes(
                _StratumMaterial(
                    task_family=assignment.task_family,
                    difficulty=assignment.difficulty,
                    strata=assignment.strata,
                )
            ),
            assignment.semantic_core_id,
        )
    )
    counts = Counter(assignment.split for assignment in assignments)
    core_counts = {split.value: counts[split] for split in _SPLIT_ORDER}
    projected_task_counts = {
        split.value: counts[split] * _VARIANTS_PER_CORE for split in _SPLIT_ORDER
    }
    report = SplitBalanceReport(
        seed=seed,
        core_counts=core_counts,
        projected_task_counts=projected_task_counts,
        cells=tuple(balance_cells),
    )
    result = SplitAssignmentResult(
        assignments=tuple(assignments),
        split_balance=report,
    )
    _validate_split_assignment_result(result)
    return result


__all__ = [
    "CoreSplitAssignment",
    "SplitAssignmentResult",
    "SplitBalanceCell",
    "SplitBalanceReport",
    "assign_splits",
]
