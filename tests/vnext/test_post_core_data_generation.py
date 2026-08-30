from __future__ import annotations

from collections import Counter
from pathlib import Path

from mub.vnext.contracts import Difficulty
from mub.vnext.generation.post_core_config import load_post_core_data_config
from mub.vnext.generation.post_core_families import (
    generate_post_core_families,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "post_core_data.yaml"


def test_post_core_generation_has_exact_family_and_difficulty_counts() -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    generated = generate_post_core_families(config)

    assert tuple(generated) == config.family_ids
    assert {
        family_id: len(cores) for family_id, cores in generated.items()
    } == {family_id: 300 for family_id in config.family_ids}

    for family_id, cores in generated.items():
        assert Counter(core.family_id for core in cores) == {family_id: 300}
        assert Counter(core.difficulty for core in cores) == {
            Difficulty.EASY: 150,
            Difficulty.MEDIUM: 90,
            Difficulty.HARD: 60,
        }

    all_ids = [core.expansion_id for cores in generated.values() for core in cores]
    assert len(all_ids) == 900
    assert len(set(all_ids)) == 900
    assert all(identifier.startswith("expansion_") for identifier in all_ids)


def test_post_core_generation_is_deterministic_and_keeps_domain_attribute_axes() -> None:
    config = load_post_core_data_config(CONFIG_PATH)
    first = generate_post_core_families(config)
    second = generate_post_core_families(config)

    assert first == second
    for family_id, cores in first.items():
        family = getattr(config.families, family_id)
        assert {core.domain for core in cores} == set(family.domains)
        assert {core.attribute for core in cores} == set(config.attributes)
        assert [core.core_index for core in cores] == list(range(300))
        assert all(core.metadata["family_id"] == family_id for core in cores)
        assert all(core.metadata["domain"] == core.domain for core in cores)
        assert all(core.metadata["attribute"] == core.attribute for core in cores)

    family_b = first["interleaved_multi_slot_update"]
    assert {
        core.family_axes["active_object_count"] for core in family_b
    } == {2, 4, 8, 12}
    assert {
        core.family_axes["interleaving_pattern"] for core in family_b
    } == {"round_robin", "burst", "adversarial_adjacent"}

    family_c = first["entity_attribute_grounding"]
    assert {core.family_axes["entity_condition"] for core in family_c} == {
        "distinct",
        "alias",
        "same_name",
        "namespace_collision",
    }
    assert {core.family_axes["attribute_condition"] for core in family_c} == {
        "exact",
        "paraphrase",
        "near_name",
    }

    family_d = first["noop_write_discipline"]
    assert {core.family_axes["noop_density"] for core in family_d} == {
        0.25,
        0.50,
        0.75,
    }
    assert {core.family_axes["trap_type"] for core in family_d} == set(
        config.families.noop_write_discipline.trap_types
    )
