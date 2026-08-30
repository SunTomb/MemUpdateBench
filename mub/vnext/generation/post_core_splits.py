from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Annotated, Any

from pydantic import Field, PlainSerializer, field_validator

from mub.vnext.contracts.common import (
    FrozenNonnegativeIntMap,
    ImmutableContractModel,
    StrictNonnegativeFloat,
    StrictNonnegativeInt,
    StrictNumericScore,
    freeze_mapping,
    thaw_json,
)
from mub.vnext.contracts.enums import Difficulty, Split
from mub.vnext.generation.post_core_config import PostCoreDataConfig
from mub.vnext.generation.post_core_families import PostCoreSemanticCore
from mub.vnext.io import canonical_json_bytes


_SPLIT_ORDER = (Split.TRAIN, Split.DEV, Split.TEST)
_DIFFICULTY_ORDER = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)

_FrozenNestedCounts = Annotated[
    Mapping[str, Mapping[str, StrictNonnegativeInt]],
    PlainSerializer(
        thaw_json,
        return_type=dict[str, dict[str, StrictNonnegativeInt]],
        when_used="always",
    ),
]
_FrozenSplitMap = Annotated[
    Mapping[str, Split],
    PlainSerializer(thaw_json, return_type=dict[str, Split], when_used="always"),
]


class _RankingMaterial(ImmutableContractModel):
    seed: StrictNonnegativeInt
    expansion_id: str = Field(min_length=1, strict=True)


class PostCoreSplitAssignment(ImmutableContractModel):
    expansion_id: str = Field(min_length=1, strict=True)
    family_id: str = Field(min_length=1, strict=True)
    difficulty: Difficulty
    split: Split
    ranking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)

    @field_validator("expansion_id", "family_id")
    @classmethod
    def _reject_blank_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("split identifiers must not be blank")
        return value


class PostCoreSplitBalanceCell(ImmutableContractModel):
    family_id: str = Field(min_length=1, strict=True)
    difficulty: Difficulty
    split: Split
    expected: StrictNonnegativeFloat
    observed: StrictNonnegativeInt
    deviation: StrictNumericScore
    total: StrictNonnegativeInt


class PostCoreSplitBalanceReport(ImmutableContractModel):
    seed: StrictNonnegativeInt
    core_counts: FrozenNonnegativeIntMap
    family_counts: _FrozenNestedCounts
    difficulty_counts: _FrozenNestedCounts
    cells: tuple[PostCoreSplitBalanceCell, ...]

    @property
    def allocation_key(self) -> int:
        return self.seed

    @field_validator("core_counts")
    @classmethod
    def _freeze_core_counts(cls, value: Mapping[str, int]):
        return freeze_mapping(value)

    @field_validator("family_counts", "difficulty_counts")
    @classmethod
    def _freeze_nested_counts(cls, value: Mapping[str, Mapping[str, int]]):
        return freeze_mapping(
            {
                key: freeze_mapping(row)
                for key, row in value.items()
            }
        )


class PostCoreSplitAssignmentResult(ImmutableContractModel):
    assignments: tuple[PostCoreSplitAssignment, ...]
    split_by_expansion_id: _FrozenSplitMap
    split_balance: PostCoreSplitBalanceReport

    @field_validator("split_by_expansion_id")
    @classmethod
    def _freeze_split_map(cls, value: Mapping[str, Split]):
        return freeze_mapping(value)


PostCoreSplitResult = PostCoreSplitAssignmentResult


def _validate_allocation_key(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("allocation_key must be an exact integer")
    if value < 0:
        raise ValueError("allocation_key must be nonnegative")
    return value


def _validate_inputs(
    config: PostCoreDataConfig,
    cores: Iterable[PostCoreSemanticCore],
) -> tuple[PostCoreSemanticCore, ...]:
    if not isinstance(config, PostCoreDataConfig):
        raise TypeError("config must be a PostCoreDataConfig")
    try:
        records = tuple(cores)
    except TypeError as exc:
        raise TypeError(
            "cores must be an iterable of PostCoreSemanticCore records"
        ) from exc
    if any(not isinstance(core, PostCoreSemanticCore) for core in records):
        raise TypeError("cores must contain only PostCoreSemanticCore records")

    expansion_ids = []
    for core in records:
        if type(core.expansion_id) is not str or not core.expansion_id.strip():
            raise ValueError("expansion_id must be a nonblank string")
        expansion_ids.append(core.expansion_id)
    if len(expansion_ids) != len(set(expansion_ids)):
        raise ValueError("duplicate expansion_id records are not allowed")
    if len(records) != config.total_semantic_cores:
        raise ValueError(
            "assign_post_core_splits requires exactly "
            f"{config.total_semantic_cores} cores; observed {len(records)}"
        )

    family_ids = set(config.family_ids)
    invalid_families: list[str] = []
    for core in records:
        if type(core.family_id) is not str:
            raise TypeError("family_id must be an exact string")
        if core.family_id not in family_ids:
            invalid_families.append(core.family_id)
    if invalid_families:
        raise ValueError(
            "cores contain unsupported post-core family IDs: "
            + ", ".join(sorted(set(invalid_families)))
        )

    family_counts = Counter(core.family_id for core in records)
    expected_family_counts = config.family_core_counts
    if any(
        family_counts[family_id] != expected_family_counts[family_id]
        for family_id in config.family_ids
    ):
        observed = ", ".join(
            f"{family_id}={family_counts[family_id]}"
            for family_id in config.family_ids
        )
        raise ValueError(
            "post-core split input must preserve configured family counts; "
            f"observed {observed}"
        )

    expected_difficulties = set(_DIFFICULTY_ORDER)
    for core in records:
        if type(core.difficulty) is not Difficulty:
            raise TypeError("difficulty must be a Difficulty")
        if core.difficulty not in expected_difficulties:
            raise ValueError(
                "post-core split cores must use easy, medium, or hard difficulty"
            )
    for family_id in config.family_ids:
        family = getattr(config.families, family_id)
        observed = Counter(
            core.difficulty.value
            for core in records
            if core.family_id == family_id
        )
        if any(
            observed[difficulty.value] != family.difficulty_quotas.as_dict[
                difficulty.value
            ]
            for difficulty in _DIFFICULTY_ORDER
        ):
            raise ValueError(
                "post-core split input must preserve configured difficulty quotas "
                f"for family {family_id}"
            )
    return records


def _ranking_sha256(expansion_id: str, seed: int) -> str:
    material = _RankingMaterial(
        seed=seed,
        expansion_id=expansion_id,
    )
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _ratio_allocation(
    total: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    exact = tuple(Decimal(total) * Decimal(str(ratio)) for ratio in ratios)
    allocation = [int(value) for value in exact]
    remainder = total - sum(allocation)
    fractional_order = sorted(
        range(len(exact)),
        key=lambda index: (-(exact[index] - allocation[index]), index),
    )
    for index in fractional_order[:remainder]:
        allocation[index] += 1
    return tuple(allocation)  # type: ignore[return-value]


def assign_post_core_splits(
    config: PostCoreDataConfig,
    cores: Iterable[PostCoreSemanticCore],
    *,
    allocation_key: int | None = None,
) -> PostCoreSplitAssignmentResult:
    """Assign post-Core expansion cores to deterministic group-first splits.

    Each family/difficulty cell is ranked by a hash of its stable expansion ID
    and the configured seed (or an explicit allocation key), then assigned in
    train/dev/test quota order.  The returned split map and balance records are
    immutable and ready for a later task renderer.
    """
    records = _validate_inputs(config, cores)
    key = config.seed if allocation_key is None else _validate_allocation_key(allocation_key)
    ratios = (config.splits.train, config.splits.dev, config.splits.test)

    assignments: list[PostCoreSplitAssignment] = []
    family_counts: dict[str, dict[str, int]] = {
        family_id: {split.value: 0 for split in _SPLIT_ORDER}
        for family_id in config.family_ids
    }
    difficulty_counts: dict[str, dict[str, int]] = {
        difficulty.value: {split.value: 0 for split in _SPLIT_ORDER}
        for difficulty in _DIFFICULTY_ORDER
    }
    cells: list[PostCoreSplitBalanceCell] = []

    for family_id in config.family_ids:
        for difficulty in _DIFFICULTY_ORDER:
            cell_records = [
                core
                for core in records
                if core.family_id == family_id and core.difficulty is difficulty
            ]
            ranked = sorted(
                cell_records,
                key=lambda core: (
                    _ranking_sha256(core.expansion_id, key),
                    core.expansion_id,
                ),
            )
            allocation = _ratio_allocation(len(ranked), ratios)
            offset = 0
            for split, observed in zip(_SPLIT_ORDER, allocation, strict=True):
                selected = ranked[offset : offset + observed]
                offset += observed
                for core in selected:
                    ranking_sha256 = _ranking_sha256(core.expansion_id, key)
                    assignments.append(
                        PostCoreSplitAssignment(
                            expansion_id=core.expansion_id,
                            family_id=family_id,
                            difficulty=difficulty,
                            split=split,
                            ranking_sha256=ranking_sha256,
                        )
                    )
                family_counts[family_id][split.value] += observed
                difficulty_counts[difficulty.value][split.value] += observed
                expected = len(ranked) * ratios[_SPLIT_ORDER.index(split)]
                cells.append(
                    PostCoreSplitBalanceCell(
                        family_id=family_id,
                        difficulty=difficulty,
                        split=split,
                        expected=float(expected),
                        observed=observed,
                        deviation=float(observed - expected),
                        total=len(ranked),
                    )
                )

    family_order = {family_id: index for index, family_id in enumerate(config.family_ids)}
    difficulty_order = {
        difficulty: index for index, difficulty in enumerate(_DIFFICULTY_ORDER)
    }
    split_order = {split: index for index, split in enumerate(_SPLIT_ORDER)}
    assignments.sort(
        key=lambda assignment: (
            family_order[assignment.family_id],
            difficulty_order[assignment.difficulty],
            split_order[assignment.split],
            assignment.ranking_sha256,
            assignment.expansion_id,
        )
    )
    core_counts = {
        split.value: sum(
            assignment.split is split for assignment in assignments
        )
        for split in _SPLIT_ORDER
    }
    report = PostCoreSplitBalanceReport(
        seed=key,
        core_counts=core_counts,
        family_counts=family_counts,
        difficulty_counts=difficulty_counts,
        cells=tuple(cells),
    )
    return PostCoreSplitAssignmentResult(
        assignments=tuple(assignments),
        split_by_expansion_id={
            assignment.expansion_id: assignment.split for assignment in assignments
        },
        split_balance=report,
    )


__all__ = [
    "PostCoreSplitAssignment",
    "PostCoreSplitAssignmentResult",
    "PostCoreSplitBalanceCell",
    "PostCoreSplitBalanceReport",
    "PostCoreSplitResult",
    "assign_post_core_splits",
]
