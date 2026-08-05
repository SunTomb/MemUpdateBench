from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mub.vnext.contracts.enums import Split
from mub.vnext.generation import (
    GenerationContext,
    SURFACE_TEMPLATE_SETS,
    generate_family_a_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"
PILOT_CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"


def _core_api():
    from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
    from mub.vnext.generation.core_config import CoreConfig, load_core_config
    from mub.vnext.generation.render import render_core_with_catalog

    return CoreConfig, load_core_config, CORE_SURFACE_CATALOG_V1, render_core_with_catalog


def _set_path(
    payload: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_core_config_loads_exact_ad_counts_four_surfaces_and_group_first_split() -> None:
    _, load_core_config, surface_catalog, _ = _core_api()

    config = load_core_config(CORE_CONFIG_PATH)

    assert config.family_core_counts == {
        "repeated_same_slot_update": 480,
        "interleaved_multi_slot_update": 480,
        "entity_attribute_grounding": 420,
        "noop_write_discipline": 420,
    }
    assert config.surface_ids == (
        "explicit_canonical",
        "concise_natural",
        "short_dialogue_lifecycle_intent",
        "controlled_adversarial_paraphrase",
    )
    assert config.surface_ids == surface_catalog.surface_ids
    assert config.split_strategy == "group_first"
    assert config.total_semantic_cores == 1800
    assert config.total_tasks == 7200
    assert config.expected_split_cores == {"train": 1260, "dev": 180, "test": 360}
    assert config.expected_split_tasks == {"train": 5040, "dev": 720, "test": 1440}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unreviewed_axis",), 1),
        (("surfaces", 0, "unreviewed_axis"), "value"),
        (("families", "repeated_same_slot_update", "unreviewed_axis"), 1),
        (("splits", "holdout"), 0.1),
        (("output", "unreviewed_axis"), "value"),
    ],
)
def test_core_config_rejects_unknown_keys_at_each_new_contract_level(
    path: tuple[str | int, ...],
    value: object,
) -> None:
    CoreConfig, load_core_config, _, _ = _core_api()
    payload = load_core_config(CORE_CONFIG_PATH).model_dump(mode="json")
    _set_path(payload, path, value)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("family", "count"),
    [
        ("repeated_same_slot_update", 479),
        ("interleaved_multi_slot_update", 481),
        ("entity_attribute_grounding", 419),
        ("noop_write_discipline", 421),
    ],
)
def test_core_config_rejects_noncanonical_ad_counts(family: str, count: int) -> None:
    CoreConfig, load_core_config, _, _ = _core_api()
    payload = load_core_config(CORE_CONFIG_PATH).model_dump(mode="json")
    payload["families"][family]["semantic_core_count"] = count

    with pytest.raises(ValidationError, match="Core family counts"):
        CoreConfig.model_validate(payload)


def test_core_config_authenticates_approved_family_schedules() -> None:
    _, load_core_config, _, _ = _core_api()

    config = load_core_config(CORE_CONFIG_PATH)

    assert config.families.repeated_same_slot_update.schedule.model_dump() == {
        "cores_per_update_depth": 80,
        "cores_per_depth_condition_cell": 20,
        "split_core_counts": {"train": 336, "dev": 48, "test": 96},
    }
    assert config.families.interleaved_multi_slot_update.schedule.model_dump() == {
        "cores_per_active_object_count": 120,
        "cores_per_update_depth": 160,
        "cores_per_pattern_within_active_object_count": 40,
        "depth_pattern_cell_min": 13,
        "depth_pattern_cell_max": 14,
        "max_depth_pattern_cell_imbalance": 1,
        "split_core_counts": {"train": 336, "dev": 48, "test": 96},
    }
    assert config.families.entity_attribute_grounding.schedule.model_dump() == {
        "cores_per_entity_attribute_cell": 35,
        "cores_per_resolution_outcome": 140,
        "split_core_counts": {"train": 294, "dev": 42, "test": 84},
    }
    assert config.families.noop_write_discipline.schedule.model_dump() == {
        "cores_per_trap_density_cell": 20,
        "split_core_counts": {"train": 294, "dev": 42, "test": 84},
    }


def test_core_config_rejects_missing_schedule_metadata() -> None:
    CoreConfig, _, _, _ = _core_api()
    with CORE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["families"]["repeated_same_slot_update"].pop("schedule", None)

    with pytest.raises(ValidationError, match="Field required"):
        CoreConfig.model_validate(payload)


def test_core_config_rejects_invalid_schedule_metadata() -> None:
    CoreConfig, _, _, _ = _core_api()
    with CORE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["families"]["repeated_same_slot_update"]["schedule"] = {
        "cores_per_update_depth": 79,
        "cores_per_depth_condition_cell": 20,
        "split_core_counts": {"train": 336, "dev": 48, "test": 96},
    }

    with pytest.raises(ValidationError, match="approved schedule"):
        CoreConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("entity_conditions", ["distinct", "same_name", "alias"]),
        ("attribute_conditions", ["exact", "paraphrase"]),
    ],
)
def test_core_config_rejects_incomplete_family_c_axis_universe(
    axis: str,
    values: list[str],
) -> None:
    CoreConfig, load_core_config, _, _ = _core_api()
    payload = load_core_config(CORE_CONFIG_PATH).model_dump(mode="json")
    payload["families"]["entity_attribute_grounding"][axis] = values

    with pytest.raises(ValidationError, match=f"Core Family C {axis}"):
        CoreConfig.model_validate(payload)


def test_core_config_rejects_wrong_family_d_density_bands() -> None:
    CoreConfig, load_core_config, _, _ = _core_api()
    payload = load_core_config(CORE_CONFIG_PATH).model_dump(mode="json")
    payload["families"]["noop_write_discipline"]["noop_densities"] = [0.25, 0.5]

    with pytest.raises(ValidationError, match="Core Family D noop_densities"):
        CoreConfig.model_validate(payload)


def test_core_surface_catalog_is_versioned_immutable_and_operation_complete() -> None:
    _, _, surface_catalog, _ = _core_api()

    assert surface_catalog.catalog_version == "vnext-core-surfaces-v1"
    assert isinstance(surface_catalog.template_sets, tuple)
    assert isinstance(surface_catalog.reference_query_template_sets, tuple)
    assert isinstance(surface_catalog.speakers, tuple)
    assert len(surface_catalog.template_sets) == 4
    assert len(surface_catalog.reference_query_template_sets) == 4
    assert len(surface_catalog.speakers) == 4
    assert all(len(template_set) == 10 for template_set in surface_catalog.template_sets)
    assert all(len(template_set) == 4 for template_set in surface_catalog.reference_query_template_sets)
    assert all(template_set[0] == surface_id for template_set, surface_id in zip(
        surface_catalog.template_sets, surface_catalog.surface_ids, strict=True
    ))
    assert all(template_set[0] == surface_id for template_set, surface_id in zip(
        surface_catalog.reference_query_template_sets,
        surface_catalog.surface_ids,
        strict=True,
    ))

    with pytest.raises(TypeError):
        surface_catalog.template_sets[0] = surface_catalog.template_sets[1]  # type: ignore[index]


def test_core_catalog_renders_four_semantically_shared_distinct_outputs() -> None:
    _, _, surface_catalog, render_core_with_catalog = _core_api()
    pilot_config = load_pilot_config(PILOT_CONFIG_PATH)
    context = GenerationContext(
        config=pilot_config,
        code_revision="task-634-render-test",
    )
    core = generate_family_a_cores(pilot_config)[0]

    tasks = tuple(
        render_core_with_catalog(
            core,
            split=Split.TRAIN,
            surface_variant=surface_variant,
            context=context,
            surface_catalog=surface_catalog,
        )
        for surface_variant in range(4)
    )

    assert len({task.task_id for task in tasks}) == 4
    assert len({task.source.raw_hash for task in tasks}) == 4
    assert len({semantic_task_hash(task) for task in tasks}) == 1
    assert all(
        tuple(
            (key.namespace, key.entity, key.attribute, key.subkey)
            for key in task.target_objects
        )
        == tuple(
            (key.namespace, key.entity, key.attribute, key.subkey)
            for key in tasks[0].target_objects
        )
        for task in tasks
    )


def test_pilot_config_catalog_and_default_renderer_remain_exactly_compatible() -> None:
    _, _, _, render_core_with_catalog = _core_api()
    from mub.vnext.generation.render import PILOT_SURFACE_CATALOG

    pilot_config = load_pilot_config(PILOT_CONFIG_PATH)
    context = GenerationContext(
        config=pilot_config,
        code_revision="task-634-pilot-compatibility",
    )
    core = generate_family_a_cores(pilot_config)[0]
    default_task = render_core(
        core,
        split=Split.TRAIN,
        surface_variant=2,
        context=context,
    )
    parameterized_task = render_core_with_catalog(
        core,
        split=Split.TRAIN,
        surface_variant=2,
        context=context,
        surface_catalog=PILOT_SURFACE_CATALOG,
    )
    reviewed_catalog_bytes = json.dumps(
        SURFACE_TEMPLATE_SETS,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert pilot_config.surface_variants_per_core == 3
    assert pilot_config.cores_per_family == 120
    assert tuple(template[0] for template in SURFACE_TEMPLATE_SETS) == (
        "direct",
        "conversational",
        "correction",
    )
    assert hashlib.sha256(reviewed_catalog_bytes).hexdigest() == (
        "cb00084f079c368bc7a85c96a684e1bf036cf81a2a0493b65a005d959a3e6c9d"
    )
    assert canonical_json_bytes(parameterized_task) == canonical_json_bytes(default_task)
