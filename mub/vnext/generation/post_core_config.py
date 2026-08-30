from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.post_core_data import (
    POST_CORE_DATA_SURFACE_CATALOG_VERSION,
    PostCoreAttribute,
    PostCoreDomain,
    PostCoreFamiliesConfig,
    PostCoreSplitConfig,
    PostCoreSurfaceDeclaration,
    StrictPositiveInt,
)
from mub.vnext.generation.post_core_catalogs import (
    POST_CORE_ATTRIBUTE_IDS,
    POST_CORE_DOMAIN_IDS,
    POST_CORE_FAMILY_DOMAIN_MATRIX,
    POST_CORE_FAMILY_IDS,
    POST_CORE_SURFACE_KEYS,
)
from mub.vnext.version import COMPILER_VERSION, PROFILE_VERSION, SCHEMA_VERSION


POST_CORE_DATA_SCHEMA_VERSION = SCHEMA_VERSION
POST_CORE_DATA_PROFILE_VERSION = PROFILE_VERSION
POST_CORE_DATA_COMPILER_VERSION = COMPILER_VERSION
POST_CORE_DATA_RELEASE_ID = "main_track_v1"
POST_CORE_DATA_SEED = 20260829
POST_CORE_DATA_CORES_PER_FAMILY = 300
POST_CORE_DATA_DIFFICULTY_QUOTAS = {"easy": 150, "medium": 90, "hard": 60}


class PostCoreDataConfig(ImmutableContractModel):
    schema_version: Literal[SCHEMA_VERSION]
    profile_version: Literal[PROFILE_VERSION]
    compiler_version: Literal[COMPILER_VERSION]
    release_id: Annotated[str, Field(min_length=1, strict=True)]
    seed: StrictPositiveInt
    surface_catalog_version: Literal[POST_CORE_DATA_SURFACE_CATALOG_VERSION]
    surfaces: tuple[PostCoreSurfaceDeclaration, ...] = Field(min_length=1)
    domains: tuple[PostCoreDomain, ...] = Field(min_length=1)
    attributes: tuple[PostCoreAttribute, ...] = Field(min_length=1)
    split_strategy: Literal["group_first"]
    splits: PostCoreSplitConfig
    families: PostCoreFamiliesConfig

    @model_validator(mode="after")
    def _validate_fixed_contract(self) -> PostCoreDataConfig:
        if self.release_id != POST_CORE_DATA_RELEASE_ID:
            raise ValueError(f"release_id must equal {POST_CORE_DATA_RELEASE_ID}")
        if self.seed != POST_CORE_DATA_SEED:
            raise ValueError(f"seed must equal {POST_CORE_DATA_SEED}")
        if self.surface_catalog_version != POST_CORE_DATA_SURFACE_CATALOG_VERSION:
            raise ValueError("unsupported post-core surface catalog version")
        if self.surface_keys != POST_CORE_SURFACE_KEYS:
            raise ValueError("post-core surfaces must match the canonical locale/variant order")
        if self.domains != POST_CORE_DOMAIN_IDS:
            raise ValueError("post-core domains must match the reviewed twelve-domain catalog")
        if self.attributes != POST_CORE_ATTRIBUTE_IDS:
            raise ValueError("post-core attributes must match the reviewed twelve-attribute catalog")
        if self.family_ids != POST_CORE_FAMILY_IDS:
            raise ValueError("post-core families must be exactly B, C, and D in order")
        if self.family_core_counts != {
            family: POST_CORE_DATA_CORES_PER_FAMILY for family in POST_CORE_FAMILY_IDS
        }:
            raise ValueError("every post-core family must contain exactly 300 semantic cores")
        expected_quotas = POST_CORE_DATA_DIFFICULTY_QUOTAS
        for family_id in self.family_ids:
            quotas = getattr(self.families, family_id).difficulty_quotas.as_dict
            if quotas != expected_quotas:
                raise ValueError("post-core difficulty quotas must be 150/90/60")
            if tuple(getattr(self.families, family_id).domains) != POST_CORE_FAMILY_DOMAIN_MATRIX[family_id]:
                raise ValueError(f"post-core domain matrix is invalid for {family_id}")
        return self

    @property
    def surface_keys(self) -> tuple[str, ...]:
        return tuple(surface.surface_key for surface in self.surfaces)

    @property
    def domain_ids(self) -> tuple[str, ...]:
        return tuple(self.domains)

    @property
    def attribute_ids(self) -> tuple[str, ...]:
        return tuple(self.attributes)

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(type(self.families).model_fields)

    @property
    def family_core_counts(self) -> dict[str, int]:
        return {
            family_id: getattr(self.families, family_id).semantic_core_count
            for family_id in self.family_ids
        }

    @property
    def difficulty_quotas(self) -> dict[str, int]:
        return self.families.interleaved_multi_slot_update.difficulty_quotas.as_dict

    @property
    def family_domain_matrix(self) -> dict[str, tuple[str, ...]]:
        return {
            family_id: tuple(getattr(self.families, family_id).domains)
            for family_id in self.family_ids
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
            split_name: int(
                Decimal(self.total_semantic_cores) * Decimal(str(ratio))
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


def load_post_core_data_config(path: Path) -> PostCoreDataConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("post-core data config YAML root must be a mapping")
    return PostCoreDataConfig.model_validate(payload)


__all__ = [
    "POST_CORE_DATA_COMPILER_VERSION",
    "POST_CORE_DATA_CORES_PER_FAMILY",
    "POST_CORE_DATA_DIFFICULTY_QUOTAS",
    "POST_CORE_DATA_PROFILE_VERSION",
    "POST_CORE_DATA_RELEASE_ID",
    "POST_CORE_DATA_SCHEMA_VERSION",
    "POST_CORE_DATA_SEED",
    "PostCoreDataConfig",
    "load_post_core_data_config",
]
