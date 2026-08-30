from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.post_core_data import PostCoreDifficultyQuotas
from mub.vnext.generation.post_core_config import (
    POST_CORE_DATA_COMPILER_VERSION,
    POST_CORE_DATA_PROFILE_VERSION,
    POST_CORE_DATA_SCHEMA_VERSION,
    PostCoreDataConfig,
    load_post_core_data_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "post_core_data.yaml"


def _set_path(payload: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_post_core_data_config_binds_release_surfaces_domains_families_and_quotas() -> None:
    config = load_post_core_data_config(CONFIG_PATH)

    assert config.schema_version == POST_CORE_DATA_SCHEMA_VERSION
    assert config.profile_version == POST_CORE_DATA_PROFILE_VERSION
    assert config.compiler_version == POST_CORE_DATA_COMPILER_VERSION
    assert config.release_id == "main_track_v1"
    assert config.seed == 20260829
    assert config.surface_keys == (
        "en-US/explicit_canonical",
        "en-US/concise_natural",
        "es-ES/concise_natural",
        "ja-JP/concise_natural",
    )
    assert config.domain_ids == (
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
    )
    assert config.attribute_ids == (
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
    )
    assert config.family_ids == (
        "interleaved_multi_slot_update",
        "entity_attribute_grounding",
        "noop_write_discipline",
    )
    assert config.family_core_counts == {family: 300 for family in config.family_ids}
    assert config.total_semantic_cores == 900
    assert config.total_tasks == 3600
    assert config.difficulty_quotas == {"easy": 150, "medium": 90, "hard": 60}
    assert config.splits.model_dump() == {"train": 0.70, "dev": 0.10, "test": 0.20}
    assert config.expected_split_cores == {"train": 630, "dev": 90, "test": 180}
    assert config.expected_split_tasks == {"train": 2520, "dev": 360, "test": 720}
    assert config.families.entity_attribute_grounding.typed_abstain is True
    assert config.families.interleaved_multi_slot_update.active_object_counts == (2, 4, 8, 12)
    assert config.families.interleaved_multi_slot_update.interleaving_patterns == (
        "round_robin", "burst", "adversarial_adjacent"
    )
    assert config.families.entity_attribute_grounding.attribute_conditions == (
        "exact", "paraphrase", "near_name"
    )
    assert config.families.entity_attribute_grounding.entity_conditions == (
        "distinct", "alias", "same_name", "namespace_collision"
    )
    assert config.families.noop_write_discipline.trap_types == (
        "transient", "hypothetical", "negated", "uncertain",
        "semantic_near_miss", "duplicate_current", "unsupported_inference"
    )
    assert config.families.noop_write_discipline.noop_densities == (0.25, 0.50, 0.75)


def test_post_core_data_catalog_exposes_the_proposed_family_domain_matrix() -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    assert config.family_domain_matrix == {
        "interleaved_multi_slot_update": ("work", "education", "software", "services"),
        "entity_attribute_grounding": ("personal", "community", "media", "civic"),
        "noop_write_discipline": ("finance", "health", "travel", "household"),
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unknown",), 1),
        (("surfaces", 0, "unknown"), "x"),
        (("families", "interleaved_multi_slot_update", "unknown"), 1),
        (("families", "unknown_family"), {}),
        (("splits", "holdout"), 0.1),
    ],
)
def test_post_core_data_config_rejects_unknown_keys(
    path: tuple[str | int, ...], value: object
) -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    payload = config.model_dump(mode="json")
    _set_path(payload, path, value)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PostCoreDataConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_core_count", 299),
        ("semantic_core_count", 301),
        ("difficulty_quotas", {"easy": 149, "medium": 90, "hard": 60}),
        ("difficulty_quotas", {"easy": 150, "medium": 91, "hard": 60}),
    ],
)
def test_post_core_data_config_rejects_invalid_family_quotas(field: str, value: object) -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    payload = config.model_dump(mode="json")
    if field == "semantic_core_count":
        payload["families"]["interleaved_multi_slot_update"][field] = value
    else:
        payload["families"]["interleaved_multi_slot_update"][field] = value
    with pytest.raises(ValidationError, match="300|difficulty quotas"):
        PostCoreDataConfig.model_validate(payload)


def test_post_core_data_config_rejects_wrong_surface_locale_or_split() -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    payload = config.model_dump(mode="json")
    payload["surfaces"][2]["locale"] = "fr-FR"
    with pytest.raises(ValidationError, match="unsupported|literal"):
        PostCoreDataConfig.model_validate(payload)

    payload = config.model_dump(mode="json")
    payload["splits"]["train"] = 0.71
    with pytest.raises(ValidationError, match="70/10/20"):
        PostCoreDataConfig.model_validate(payload)



def test_post_core_data_contracts_are_frozen_and_quotas_are_exact() -> None:
    quotas = PostCoreDifficultyQuotas(easy=150, medium=90, hard=60)
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        quotas.easy = 149  # type: ignore[misc]
    with pytest.raises(ValidationError, match="150/90/60"):
        PostCoreDifficultyQuotas(easy=149, medium=91, hard=60)


def test_post_core_data_config_yaml_root_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML root must be a mapping"):
        load_post_core_data_config(path)
