from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from mub.vnext.contracts.common import ContractModel, StrictBool, StrictNonnegativeInt
from mub.vnext.generation.config import (
    EntityAttributeGroundingConfig,
    InterleavingPattern,
    MechanismSliceConfig,
    NoopWriteDisciplineConfig,
    NonEmptyPositiveInts,
    OutputConfig,
    SplitConfig,
    StrictPositiveInt,
)
from mub.vnext.generation.core_catalogs import (
    CORE_SURFACE_CATALOG_VERSION,
    CORE_SURFACE_IDS,
)
from mub.vnext.version import PROFILE_VERSION, SCHEMA_VERSION


CoreSurfaceId = Literal[
    "explicit_canonical",
    "concise_natural",
    "short_dialogue_lifecycle_intent",
    "controlled_adversarial_paraphrase",
]
CoreFamilyACondition = Literal[
    "stale_burden",
    "duplicate_current",
    "other_attribute_distractor",
    "same_name_other_entity_distractor",
]
CORE_FAMILY_A_DEPTHS = (1, 2, 4, 8, 16, 32)
CORE_FAMILY_A_CONDITIONS = (
    "stale_burden",
    "duplicate_current",
    "other_attribute_distractor",
    "same_name_other_entity_distractor",
)
CORE_FAMILY_B_DEPTHS = (1, 4, 16)
CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS = (2, 4, 8, 12)
CORE_FAMILY_B_INTERLEAVING_PATTERNS = (
    "round_robin",
    "burst",
    "adversarial_adjacent",
)
CORE_FAMILY_C_ENTITY_CONDITIONS = (
    "distinct",
    "same_name",
    "alias",
    "namespace_collision",
)
CORE_FAMILY_C_ATTRIBUTE_CONDITIONS = ("exact", "paraphrase", "near_name")
CORE_FAMILY_D_DENSITIES = (0.25, 0.50, 0.75)

CoreFamilyDTrap = Literal[
    "transient",
    "hypothetical",
    "negated",
    "uncertain",
    "semantic_near_miss",
    "duplicate_current",
    "unsupported_inference",
]
CORE_FAMILY_D_TRAPS = (
    "transient",
    "hypothetical",
    "negated",
    "uncertain",
    "semantic_near_miss",
    "duplicate_current",
    "unsupported_inference",
)

class CoreSplitCoreCounts(ContractModel):
    train: StrictPositiveInt
    dev: StrictPositiveInt
    test: StrictPositiveInt


class CoreFamilyASchedule(ContractModel):
    cores_per_update_depth: StrictPositiveInt
    cores_per_depth_condition_cell: StrictPositiveInt
    split_core_counts: CoreSplitCoreCounts


class CoreFamilyBSchedule(ContractModel):
    cores_per_active_object_count: StrictPositiveInt
    cores_per_update_depth: StrictPositiveInt
    cores_per_pattern_within_active_object_count: StrictPositiveInt
    depth_pattern_cell_min: StrictPositiveInt
    depth_pattern_cell_max: StrictPositiveInt
    max_depth_pattern_cell_imbalance: StrictNonnegativeInt
    split_core_counts: CoreSplitCoreCounts


class CoreFamilyCSchedule(ContractModel):
    cores_per_entity_attribute_cell: StrictPositiveInt
    cores_per_resolution_outcome: StrictPositiveInt
    split_core_counts: CoreSplitCoreCounts


class CoreFamilyDSchedule(ContractModel):
    cores_per_trap_density_cell: StrictPositiveInt
    split_core_counts: CoreSplitCoreCounts


CORE_FAMILY_COUNTS = MappingProxyType(
    {
        "repeated_same_slot_update": 480,
        "interleaved_multi_slot_update": 480,
        "entity_attribute_grounding": 420,
        "noop_write_discipline": 420,
    }
)


class CoreSurfaceDeclaration(ContractModel):
    surface_id: CoreSurfaceId


class CoreRepeatedSameSlotUpdateConfig(ContractModel):
    enabled: StrictBool
    semantic_core_count: StrictPositiveInt
    update_depths: NonEmptyPositiveInts
    conditions: Annotated[list[CoreFamilyACondition], Field(min_length=1)]
    schedule: CoreFamilyASchedule


class CoreInterleavedMultiSlotUpdateConfig(ContractModel):
    enabled: StrictBool
    semantic_core_count: StrictPositiveInt
    update_depths: NonEmptyPositiveInts
    active_object_counts: NonEmptyPositiveInts
    interleaving_patterns: Annotated[list[InterleavingPattern], Field(min_length=1)]
    schedule: CoreFamilyBSchedule


class CoreEntityAttributeGroundingConfig(EntityAttributeGroundingConfig):
    semantic_core_count: StrictPositiveInt
    schedule: CoreFamilyCSchedule


class CoreNoopWriteDisciplineConfig(NoopWriteDisciplineConfig):
    semantic_core_count: StrictPositiveInt
    trap_types: Annotated[list[CoreFamilyDTrap], Field(min_length=1)]
    schedule: CoreFamilyDSchedule



class CoreFamiliesConfig(ContractModel):
    repeated_same_slot_update: CoreRepeatedSameSlotUpdateConfig
    interleaved_multi_slot_update: CoreInterleavedMultiSlotUpdateConfig
    entity_attribute_grounding: CoreEntityAttributeGroundingConfig
    noop_write_discipline: CoreNoopWriteDisciplineConfig


class CoreConfig(ContractModel):
    schema_version: Annotated[str, Field(min_length=1, strict=True)]
    profile_version: Annotated[str, Field(min_length=1, strict=True)]
    release_id: Annotated[str, Field(min_length=1, strict=True)]
    seed: StrictPositiveInt
    surface_catalog_version: Literal["vnext-core-surfaces-v1"]
    surfaces: Annotated[list[CoreSurfaceDeclaration], Field(min_length=1)]
    split_strategy: Literal["group_first"]
    splits: SplitConfig
    families: CoreFamiliesConfig
    mechanism_slice: MechanismSliceConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_core_contract(self) -> CoreConfig:
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
        if self.surface_catalog_version != CORE_SURFACE_CATALOG_VERSION:
            raise ValueError("unsupported Core surface catalog version")
        if self.surface_ids != CORE_SURFACE_IDS:
            raise ValueError(
                "Core surfaces must declare the four canonical surface IDs in order"
            )
        if self.family_core_counts != dict(CORE_FAMILY_COUNTS):
            raise ValueError(
                "Core family counts must be A=480, B=480, C=420, and D=420"
            )
        family_a = self.families.repeated_same_slot_update
        if not family_a.enabled:
            raise ValueError("Core Family A must be enabled")
        if tuple(family_a.update_depths) != CORE_FAMILY_A_DEPTHS:
            raise ValueError(
                "Core Family A update_depths must be [1, 2, 4, 8, 16, 32]"
            )
        if tuple(family_a.conditions) != CORE_FAMILY_A_CONDITIONS:
            raise ValueError("Core Family A conditions must match the approved order")
        if family_a.schedule.model_dump(mode="python") != {
            "cores_per_update_depth": 80,
            "cores_per_depth_condition_cell": 20,
            "split_core_counts": {"train": 336, "dev": 48, "test": 96},
        }:
            raise ValueError("Core Family A approved schedule is invalid")
        family_b = self.families.interleaved_multi_slot_update
        if not family_b.enabled:
            raise ValueError("Core Family B must be enabled")
        if tuple(family_b.update_depths) != CORE_FAMILY_B_DEPTHS:
            raise ValueError("Core Family B update_depths must be [1, 4, 16]")
        if (
            tuple(family_b.active_object_counts)
            != CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS
        ):
            raise ValueError(
                "Core Family B active_object_counts must be [2, 4, 8, 12]"
            )
        if (
            tuple(family_b.interleaving_patterns)
            != CORE_FAMILY_B_INTERLEAVING_PATTERNS
        ):
            raise ValueError(
                "Core Family B interleaving_patterns must match the approved order"
            )
        if family_b.schedule.model_dump(mode="python") != {
            "cores_per_active_object_count": 120,
            "cores_per_update_depth": 160,
            "cores_per_pattern_within_active_object_count": 40,
            "depth_pattern_cell_min": 13,
            "depth_pattern_cell_max": 14,
            "max_depth_pattern_cell_imbalance": 1,
            "split_core_counts": {"train": 336, "dev": 48, "test": 96},
        }:
            raise ValueError("Core Family B approved schedule is invalid")
        family_c = self.families.entity_attribute_grounding
        if set(family_c.entity_conditions) != set(CORE_FAMILY_C_ENTITY_CONDITIONS):
            raise ValueError("Core Family C entity_conditions universe is invalid")
        if set(family_c.attribute_conditions) != set(CORE_FAMILY_C_ATTRIBUTE_CONDITIONS):
            raise ValueError("Core Family C attribute_conditions universe is invalid")
        if family_c.schedule.model_dump(mode="python") != {
            "cores_per_entity_attribute_cell": 35,
            "cores_per_resolution_outcome": 140,
            "split_core_counts": {"train": 294, "dev": 42, "test": 84},
        }:
            raise ValueError("Core Family C approved schedule is invalid")
        family_d = self.families.noop_write_discipline
        if tuple(family_d.noop_densities) != CORE_FAMILY_D_DENSITIES:
            raise ValueError(
                "Core Family D noop_densities must be exactly 0.25, 0.50, and 0.75"
            )
        if family_d.schedule.model_dump(mode="python") != {
            "cores_per_trap_density_cell": 20,
            "split_core_counts": {"train": 294, "dev": 42, "test": 84},
        }:
            raise ValueError("Core Family D approved schedule is invalid")
        if tuple(family_d.trap_types) != CORE_FAMILY_D_TRAPS:
            raise ValueError(
                "Core Family D trap_types must match the seven approved traps in order"
            )

        ratios = {
            "train": self.splits.train,
            "dev": self.splits.dev,
            "test": self.splits.test,
        }
        if ratios != {"train": 0.70, "dev": 0.10, "test": 0.20}:
            raise ValueError("Core split ratios must be exactly 70/10/20")
        if sum(Decimal(str(ratio)) for ratio in ratios.values()) != Decimal("1"):
            raise ValueError("split ratios must sum to 1")
        for family_name, family_count in self.family_core_counts.items():
            for split_name, ratio in ratios.items():
                core_count = Decimal(family_count) * Decimal(str(ratio))
                if core_count != core_count.to_integral_value():
                    raise ValueError(
                        f"split {split_name} must allocate a whole number of "
                        f"cores for Core family {family_name}"
                    )
        return self

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(surface.surface_id for surface in self.surfaces)

    @property
    def family_core_counts(self) -> dict[str, int]:
        return {
            family_name: getattr(self.families, family_name).semantic_core_count
            for family_name in type(self.families).model_fields
        }

    @property
    def total_semantic_cores(self) -> int:
        return sum(self.family_core_counts.values())

    @property
    def total_tasks(self) -> int:
        return self.total_semantic_cores * len(self.surfaces)

    @property
    def expected_split_cores(self) -> dict[str, int]:
        return {
            split_name: sum(
                int(Decimal(count) * Decimal(str(ratio)))
                for count in self.family_core_counts.values()
            )
            for split_name, ratio in (
                ("train", self.splits.train),
                ("dev", self.splits.dev),
                ("test", self.splits.test),
            )
        }

    @property
    def expected_split_tasks(self) -> dict[str, int]:
        return {
            split_name: core_count * len(self.surfaces)
            for split_name, core_count in self.expected_split_cores.items()
        }


def load_core_config(path: Path) -> CoreConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Core config YAML root must be a mapping")
    return CoreConfig.model_validate(payload)


__all__ = [
    "CORE_FAMILY_A_CONDITIONS",
    "CORE_FAMILY_A_DEPTHS",
    "CORE_FAMILY_B_ACTIVE_OBJECT_COUNTS",
    "CORE_FAMILY_B_DEPTHS",
    "CORE_FAMILY_B_INTERLEAVING_PATTERNS",
    "CORE_FAMILY_C_ATTRIBUTE_CONDITIONS",
    "CORE_FAMILY_C_ENTITY_CONDITIONS",
    "CORE_FAMILY_D_DENSITIES",
    "CORE_FAMILY_D_TRAPS",
    "CORE_FAMILY_COUNTS",
    "CoreConfig",
    "CoreFamiliesConfig",
    "CoreFamilyASchedule",
    "CoreFamilyBSchedule",
    "CoreFamilyCSchedule",
    "CoreFamilyDSchedule",
    "CoreSplitCoreCounts",
    "CoreFamilyACondition",
    "CoreFamilyDTrap",
    "CoreSurfaceDeclaration",
    "CoreSurfaceId",
    "load_core_config",
]
