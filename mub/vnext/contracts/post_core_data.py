from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel, StrictBool, StrictNonnegativeInt


StrictPositiveInt = Annotated[int, Field(gt=0, strict=True)]
StrictSplitRatio = Annotated[
    float,
    Field(gt=0.0, lt=1.0, strict=True, allow_inf_nan=False),
]
StrictProbability = Annotated[
    float,
    Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False),
]


POST_CORE_DATA_SURFACE_CATALOG_VERSION = "vnext-post-core-data-surfaces-v1"

PostCoreLocale = Literal["en-US", "es-ES", "ja-JP"]
PostCoreSurfaceId = Literal["explicit_canonical", "concise_natural"]
PostCoreDomain = Literal[
    "personal",
    "work",
    "community",
    "services",
    "education",
    "travel",
    "household",
    "software",
    "finance",
    "health",
    "media",
    "civic",
]
PostCoreAttribute = Literal[
    "location",
    "company",
    "preference",
    "language",
    "timezone",
    "hobby",
    "instrument",
    "project",
    "role",
    "status",
    "priority",
    "contact_method",
]
PostCoreFamilyId = Literal[
    "interleaved_multi_slot_update",
    "entity_attribute_grounding",
    "noop_write_discipline",
]
InterleavingPattern = Literal["round_robin", "burst", "adversarial_adjacent"]
EntityCondition = Literal["distinct", "alias", "same_name", "namespace_collision"]
AttributeCondition = Literal["exact", "paraphrase", "near_name"]
TrapType = Literal[
    "transient",
    "hypothetical",
    "negated",
    "uncertain",
    "semantic_near_miss",
    "duplicate_current",
    "unsupported_inference",
]


class PostCoreSurfaceDeclaration(ImmutableContractModel):
    locale: PostCoreLocale
    surface_id: PostCoreSurfaceId

    @property
    def surface_key(self) -> str:
        return f"{self.locale}/{self.surface_id}"


class PostCoreDifficultyQuotas(ImmutableContractModel):
    easy: StrictNonnegativeInt
    medium: StrictNonnegativeInt
    hard: StrictNonnegativeInt

    @model_validator(mode="after")
    def _validate_fixed_quotas(self) -> PostCoreDifficultyQuotas:
        if self.as_dict != {"easy": 150, "medium": 90, "hard": 60}:
            raise ValueError("difficulty quotas must be exactly 150/90/60")
        return self

    @property
    def as_dict(self) -> dict[str, int]:
        return {"easy": self.easy, "medium": self.medium, "hard": self.hard}


class PostCoreFamilyBaseConfig(ImmutableContractModel):
    semantic_core_count: Literal[300]
    domains: tuple[PostCoreDomain, ...] = Field(min_length=1)
    difficulty_quotas: PostCoreDifficultyQuotas

    @field_validator("domains")
    @classmethod
    def _validate_domains(
        cls, values: tuple[PostCoreDomain, ...]
    ) -> tuple[PostCoreDomain, ...]:
        if len(values) != len(set(values)):
            raise ValueError("family domains must be unique")
        if len(values) != 4:
            raise ValueError("each family must declare exactly four domains")
        return values


class PostCoreFamilyBConfig(PostCoreFamilyBaseConfig):
    active_object_counts: tuple[StrictPositiveInt, ...] = Field(min_length=1)
    interleaving_patterns: tuple[InterleavingPattern, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_axes(self) -> PostCoreFamilyBConfig:
        if self.active_object_counts != (2, 4, 8, 12):
            raise ValueError("Family B active_object_counts must be 2/4/8/12")
        if self.interleaving_patterns != (
            "round_robin",
            "burst",
            "adversarial_adjacent",
        ):
            raise ValueError("Family B interleaving_patterns are not canonical")
        return self


class PostCoreFamilyCConfig(PostCoreFamilyBaseConfig):
    attribute_conditions: tuple[AttributeCondition, ...] = Field(min_length=1)
    entity_conditions: tuple[EntityCondition, ...] = Field(min_length=1)
    typed_abstain: StrictBool

    @model_validator(mode="after")
    def _validate_axes(self) -> PostCoreFamilyCConfig:
        if self.attribute_conditions != ("exact", "paraphrase", "near_name"):
            raise ValueError("Family C attribute_conditions are not canonical")
        if self.entity_conditions != (
            "distinct",
            "alias",
            "same_name",
            "namespace_collision",
        ):
            raise ValueError("Family C entity_conditions are not canonical")
        if not self.typed_abstain:
            raise ValueError("Family C must enable typed abstention")
        return self


class PostCoreFamilyDConfig(PostCoreFamilyBaseConfig):
    noop_densities: tuple[StrictProbability, ...] = Field(min_length=1)
    trap_types: tuple[TrapType, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_axes(self) -> PostCoreFamilyDConfig:
        if self.noop_densities != (0.25, 0.50, 0.75):
            raise ValueError("Family D noop_densities must be 25/50/75 percent")
        if self.trap_types != (
            "transient",
            "hypothetical",
            "negated",
            "uncertain",
            "semantic_near_miss",
            "duplicate_current",
            "unsupported_inference",
        ):
            raise ValueError("Family D trap_types are not canonical")
        return self


class PostCoreFamiliesConfig(ImmutableContractModel):
    interleaved_multi_slot_update: PostCoreFamilyBConfig
    entity_attribute_grounding: PostCoreFamilyCConfig
    noop_write_discipline: PostCoreFamilyDConfig


class PostCoreSplitConfig(ImmutableContractModel):
    train: StrictSplitRatio
    dev: StrictSplitRatio
    test: StrictSplitRatio

    @model_validator(mode="after")
    def _validate_ratios(self) -> PostCoreSplitConfig:
        if (self.train, self.dev, self.test) != (0.70, 0.10, 0.20):
            raise ValueError("post-core split ratios must be exactly 70/10/20")
        return self


__all__ = [
    "AttributeCondition",
    "EntityCondition",
    "InterleavingPattern",
    "PostCoreAttribute",
    "POST_CORE_DATA_SURFACE_CATALOG_VERSION",
    "PostCoreDifficultyQuotas",
    "PostCoreDomain",
    "PostCoreFamiliesConfig",
    "PostCoreFamilyBConfig",
    "PostCoreFamilyCConfig",
    "PostCoreFamilyDConfig",
    "PostCoreFamilyId",
    "PostCoreLocale",
    "PostCoreSplitConfig",
    "PostCoreSurfaceDeclaration",
    "PostCoreSurfaceId",
    "TrapType",
]
