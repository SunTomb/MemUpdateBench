from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping

from mub.vnext.contracts.enums import Difficulty, TaskFamily
from mub.vnext.generation.identity import stable_id
from mub.vnext.generation.post_core_config import PostCoreDataConfig


_DIFFICULTY_ORDER = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
_FAMILY_B = "interleaved_multi_slot_update"
_FAMILY_C = "entity_attribute_grounding"
_FAMILY_D = "noop_write_discipline"


@dataclass(frozen=True, slots=True)
class PostCoreSemanticCore:
    """Metadata-only semantic core for the first post-Core data slice.

    The object deliberately carries no rendered surface or v3 task payload.  Its
    stable ``expansion_id`` binds the configured family axes and semantic
    metadata so a later renderer can consume these cores without regenerating
    the expansion.
    """

    expansion_id: str
    family_id: str
    difficulty: Difficulty
    core_index: int
    domain: str
    attribute: str
    family_axes: Mapping[str, object]
    metadata: Mapping[str, object]
    profile: Mapping[str, object]
    stratification: Mapping[str, object]

    @property
    def core_id(self) -> str:
        """Expose the expansion identity under the existing core vocabulary."""
        return self.expansion_id

    @property
    def family(self) -> str:
        return self.family_id

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily(self.family_id)

    @property
    def axes(self) -> Mapping[str, object]:
        return self.family_axes


def _mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def _difficulty_schedule(config: PostCoreDataConfig, family_id: str) -> tuple[Difficulty, ...]:
    family = getattr(config.families, family_id)
    quotas = family.difficulty_quotas.as_dict
    schedule = tuple(
        difficulty
        for difficulty in _DIFFICULTY_ORDER
        for _ in range(quotas[difficulty.value])
    )
    if len(schedule) != family.semantic_core_count:
        raise ValueError(f"difficulty quotas do not fill {family_id} semantic cores")
    return schedule


def _family_axis_product(config: PostCoreDataConfig, family_id: str) -> tuple[Mapping[str, object], ...]:
    family = getattr(config.families, family_id)
    if family_id == _FAMILY_B:
        return tuple(
            {"active_object_count": active_count, "interleaving_pattern": pattern}
            for active_count, pattern in product(
                family.active_object_counts,
                family.interleaving_patterns,
            )
        )
    if family_id == _FAMILY_C:
        return tuple(
            {
                "entity_condition": entity_condition,
                "attribute_condition": attribute_condition,
                "typed_abstain": family.typed_abstain,
            }
            for entity_condition, attribute_condition in product(
                family.entity_conditions,
                family.attribute_conditions,
            )
        )
    if family_id == _FAMILY_D:
        return tuple(
            {"noop_density": density, "trap_type": trap_type}
            for density, trap_type in product(
                family.noop_densities,
                family.trap_types,
            )
        )
    raise ValueError(f"unsupported post-core family: {family_id}")


def _build_family_cores(
    config: PostCoreDataConfig,
    family_id: str,
) -> list[PostCoreSemanticCore]:
    if not isinstance(config, PostCoreDataConfig):
        raise TypeError("config must be a PostCoreDataConfig")
    if family_id not in config.family_ids:
        raise ValueError(f"unsupported post-core family: {family_id}")

    family = getattr(config.families, family_id)
    difficulties = _difficulty_schedule(config, family_id)
    axis_product = _family_axis_product(config, family_id)
    domains = tuple(family.domains)
    attributes = tuple(config.attributes)
    cores: list[PostCoreSemanticCore] = []

    for core_index in range(family.semantic_core_count):
        domain = domains[core_index % len(domains)]
        attribute = attributes[(core_index // len(domains)) % len(attributes)]
        axes = dict(axis_product[core_index % len(axis_product)])
        difficulty = difficulties[core_index]
        identity_payload = {
            "release_id": config.release_id,
            "seed": config.seed,
            "family_id": family_id,
            "core_index": core_index,
            "difficulty": difficulty.value,
            "domain": domain,
            "attribute": attribute,
            "family_axes": axes,
        }
        expansion_id = stable_id("expansion", identity_payload)
        metadata = {
            "family_id": family_id,
            "domain": domain,
            "attribute": attribute,
            "difficulty": difficulty.value,
            **axes,
        }
        profile = {
            "family_id": family_id,
            "domain": domain,
            "attribute": attribute,
            "difficulty": difficulty.value,
            "family_axis_index": core_index % len(axis_product),
            "family_axis_size": len(axis_product),
            **axes,
        }
        stratification = {
            "family_id": family_id,
            "domain": domain,
            "attribute": attribute,
            "difficulty": difficulty.value,
            "family_axis_index": core_index % len(axis_product),
            "family_axis_size": len(axis_product),
            **axes,
        }
        cores.append(
            PostCoreSemanticCore(
                expansion_id=expansion_id,
                family_id=family_id,
                difficulty=difficulty,
                core_index=core_index,
                domain=domain,
                attribute=attribute,
                family_axes=_mapping(axes),
                metadata=_mapping(metadata),
                profile=_mapping(profile),
                stratification=_mapping(stratification),
            )
        )

    identifiers = [core.expansion_id for core in cores]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"post-core {family_id} expansion IDs are not unique")
    return cores


def generate_post_core_family_b_cores(
    config: PostCoreDataConfig,
) -> list[PostCoreSemanticCore]:
    return _build_family_cores(config, _FAMILY_B)


def generate_post_core_family_c_cores(
    config: PostCoreDataConfig,
) -> list[PostCoreSemanticCore]:
    return _build_family_cores(config, _FAMILY_C)


def generate_post_core_family_d_cores(
    config: PostCoreDataConfig,
) -> list[PostCoreSemanticCore]:
    return _build_family_cores(config, _FAMILY_D)


def generate_post_core_family_cores(
    config: PostCoreDataConfig,
    family_id: str,
) -> list[PostCoreSemanticCore]:
    return _build_family_cores(config, family_id)


def generate_post_core_families(
    config: PostCoreDataConfig,
) -> dict[str, list[PostCoreSemanticCore]]:
    if not isinstance(config, PostCoreDataConfig):
        raise TypeError("config must be a PostCoreDataConfig")
    return {
        family_id: _build_family_cores(config, family_id)
        for family_id in config.family_ids
    }


def generate_post_core_cores(
    config: PostCoreDataConfig,
) -> list[PostCoreSemanticCore]:
    return [
        core
        for family_id in config.family_ids
        for core in _build_family_cores(config, family_id)
    ]


# Short aliases mirror the existing Family A-D generator naming while keeping
# the post-Core namespace explicit for callers that need it.
generate_post_core_family_b = generate_post_core_family_b_cores
generate_post_core_family_c = generate_post_core_family_c_cores
generate_post_core_family_d = generate_post_core_family_d_cores


__all__ = [
    "PostCoreSemanticCore",
    "generate_post_core_cores",
    "generate_post_core_families",
    "generate_post_core_family_b",
    "generate_post_core_family_b_cores",
    "generate_post_core_family_c",
    "generate_post_core_family_c_cores",
    "generate_post_core_family_d",
    "generate_post_core_family_d_cores",
    "generate_post_core_family_cores",
]
