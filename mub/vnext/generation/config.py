from __future__ import annotations

import ntpath
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import ContractModel, StrictBool, StrictNonnegativeInt
from mub.vnext.contracts.enums import Difficulty
from mub.vnext.version import PROFILE_VERSION, SCHEMA_VERSION

StrictPositiveInt = Annotated[int, Field(gt=0, strict=True)]
StrictProbability = Annotated[
    float,
    Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
]
StrictSplitRatio = Annotated[
    float,
    Field(gt=0.0, lt=1.0, strict=True, allow_inf_nan=False),
]
NonEmptyPositiveInts = Annotated[list[StrictPositiveInt], Field(min_length=1)]
NonEmptyProbabilities = Annotated[list[StrictProbability], Field(min_length=1)]
NonEmptyDifficulties = Annotated[list[Difficulty], Field(min_length=1)]

InterleavingPattern = Literal["round_robin", "burst", "adversarial_adjacent"]
EntityCondition = Literal["distinct", "same_name", "alias", "namespace_collision"]
AttributeCondition = Literal["exact", "paraphrase", "near_name"]
TrapType = Literal[
    "semantic_near_miss",
    "duplicate_current",
    "other_entity_correction",
    "other_attribute_correction",
]
ContextOrder = Literal["chronological", "reverse_chronological"]
ContextAnnotation = Literal["none", "latest_outdated_label"]

AxisValue = TypeVar("AxisValue")
PILOT_DIFFICULTIES = frozenset(
    {Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD}
)


def _require_unique(values: list[AxisValue], field_name: str) -> list[AxisValue]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
    return values


def _lexical_path_identity(value: str) -> str:
    return ntpath.normcase(ntpath.normpath(value))


class DifficultyNonnegativeCounts(ContractModel):
    easy: StrictNonnegativeInt
    medium: StrictNonnegativeInt
    hard: StrictNonnegativeInt


class DifficultyPositiveCounts(ContractModel):
    easy: StrictPositiveInt
    medium: StrictPositiveInt
    hard: StrictPositiveInt


class DifficultyDensities(ContractModel):
    easy: StrictProbability
    medium: StrictProbability
    hard: StrictProbability


class SplitConfig(ContractModel):
    train: StrictSplitRatio
    dev: StrictSplitRatio
    test: StrictSplitRatio


class _PilotFamilyConfig(ContractModel):
    enabled: StrictBool
    difficulties: NonEmptyDifficulties

    @field_validator("enabled")
    @classmethod
    def _require_enabled(cls, value: bool) -> bool:
        if not value:
            raise ValueError("all four Pilot families must be enabled")
        return value

    @field_validator("difficulties")
    @classmethod
    def _validate_difficulties(
        cls,
        values: list[Difficulty],
    ) -> list[Difficulty]:
        _require_unique(values, "difficulties")
        if set(values) != PILOT_DIFFICULTIES:
            raise ValueError("Pilot difficulties must include easy, medium, and hard")
        return values


class RepeatedSameSlotUpdateConfig(_PilotFamilyConfig):
    update_depths: NonEmptyPositiveInts
    same_name_distractors: DifficultyNonnegativeCounts
    same_entity_other_attribute: DifficultyNonnegativeCounts
    noop_near_miss: DifficultyNonnegativeCounts

    @field_validator("update_depths")
    @classmethod
    def _validate_update_depths(cls, values: list[int]) -> list[int]:
        return _require_unique(values, "update_depths")


class InterleavedMultiSlotUpdateConfig(_PilotFamilyConfig):
    update_depths: NonEmptyPositiveInts
    active_object_counts: DifficultyPositiveCounts
    interleaving_patterns: Annotated[list[InterleavingPattern], Field(min_length=1)]
    cross_slot_distractor_density: DifficultyDensities

    @field_validator("update_depths", "interleaving_patterns")
    @classmethod
    def _validate_unique_axes(cls, values: list[AxisValue], info) -> list[AxisValue]:
        return _require_unique(values, info.field_name)


class EntityAttributeGroundingConfig(_PilotFamilyConfig):
    entity_conditions: Annotated[list[EntityCondition], Field(min_length=1)]
    attribute_conditions: Annotated[list[AttributeCondition], Field(min_length=1)]

    @field_validator("entity_conditions", "attribute_conditions")
    @classmethod
    def _validate_unique_axes(cls, values: list[str], info) -> list[str]:
        return _require_unique(values, info.field_name)


class NoopWriteDisciplineConfig(_PilotFamilyConfig):
    noop_densities: NonEmptyProbabilities
    trap_types: Annotated[list[TrapType], Field(min_length=1)]

    @field_validator("noop_densities", "trap_types")
    @classmethod
    def _validate_unique_axes(cls, values: list[AxisValue], info) -> list[AxisValue]:
        return _require_unique(values, info.field_name)


class PilotFamiliesConfig(ContractModel):
    repeated_same_slot_update: RepeatedSameSlotUpdateConfig
    interleaved_multi_slot_update: InterleavedMultiSlotUpdateConfig
    entity_attribute_grounding: EntityAttributeGroundingConfig
    noop_write_discipline: NoopWriteDisciplineConfig


class MechanismCondition(ContractModel):
    context_order: ContextOrder
    context_annotation: ContextAnnotation

    @model_validator(mode="after")
    def _validate_supported_pair(self) -> MechanismCondition:
        pair = (self.context_order, self.context_annotation)
        supported = {
            ("chronological", "none"),
            ("reverse_chronological", "none"),
            ("reverse_chronological", "latest_outdated_label"),
        }
        if pair not in supported:
            raise ValueError("unsupported Pilot mechanism condition")
        return self


class MechanismSliceConfig(ContractModel):
    stale_counts: NonEmptyPositiveInts
    conditions: Annotated[list[MechanismCondition], Field(min_length=1)]

    @field_validator("stale_counts")
    @classmethod
    def _validate_stale_counts(cls, values: list[int]) -> list[int]:
        return _require_unique(values, "stale_counts")

    @field_validator("conditions")
    @classmethod
    def _validate_conditions(
        cls,
        values: list[MechanismCondition],
    ) -> list[MechanismCondition]:
        pairs = [
            (condition.context_order, condition.context_annotation)
            for condition in values
        ]
        _require_unique(pairs, "conditions")
        return values


class OutputConfig(ContractModel):
    staging_dir: Annotated[str, Field(min_length=1, strict=True)]
    release_dir: Annotated[str, Field(min_length=1, strict=True)]

    @field_validator("staging_dir", "release_dir")
    @classmethod
    def _reject_blank_paths(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output paths must not be blank")
        return value

    @model_validator(mode="after")
    def _require_distinct_paths(self) -> OutputConfig:
        if _lexical_path_identity(self.staging_dir) == _lexical_path_identity(
            self.release_dir
        ):
            raise ValueError("staging_dir and release_dir must be distinct")
        return self


class PilotConfig(ContractModel):
    schema_version: Annotated[str, Field(min_length=1, strict=True)]
    profile_version: Annotated[str, Field(min_length=1, strict=True)]
    release_id: Annotated[str, Field(min_length=1, strict=True)]
    seed: StrictPositiveInt
    surface_variants_per_core: StrictPositiveInt
    cores_per_family: StrictPositiveInt
    splits: SplitConfig
    families: PilotFamiliesConfig
    mechanism_slice: MechanismSliceConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_fixed_contract(self) -> PilotConfig:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must match the canonical version {SCHEMA_VERSION}"
            )
        if self.profile_version != PROFILE_VERSION:
            raise ValueError(
                f"profile_version must match the canonical version {PROFILE_VERSION}"
            )
        if not self.release_id.strip():
            raise ValueError("release_id must not be blank")

        ratios = {
            "train": self.splits.train,
            "dev": self.splits.dev,
            "test": self.splits.test,
        }
        if sum(Decimal(str(ratio)) for ratio in ratios.values()) != Decimal("1"):
            raise ValueError("split ratios must sum to 1")
        for name, ratio in ratios.items():
            core_count = Decimal(self.cores_per_family) * Decimal(str(ratio))
            if core_count != core_count.to_integral_value():
                raise ValueError(
                    f"split {name} must allocate a whole number of cores per family"
                )
        return self

    @property
    def total_semantic_cores(self) -> int:
        return self.cores_per_family * len(type(self.families).model_fields)

    @property
    def total_tasks(self) -> int:
        return self.total_semantic_cores * self.surface_variants_per_core

    @property
    def expected_split_tasks(self) -> dict[str, int]:
        family_count = len(type(self.families).model_fields)
        return {
            name: int(
                Decimal(self.cores_per_family)
                * Decimal(str(ratio))
                * family_count
                * self.surface_variants_per_core
            )
            for name, ratio in (
                ("train", self.splits.train),
                ("dev", self.splits.dev),
                ("test", self.splits.test),
            )
        }


def load_pilot_config(path: Path) -> PilotConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Pilot config YAML root must be a mapping")
    return PilotConfig.model_validate(payload)


__all__ = [
    "EntityAttributeGroundingConfig",
    "InterleavedMultiSlotUpdateConfig",
    "MechanismCondition",
    "MechanismSliceConfig",
    "NoopWriteDisciplineConfig",
    "OutputConfig",
    "PilotConfig",
    "RepeatedSameSlotUpdateConfig",
    "load_pilot_config",
]
