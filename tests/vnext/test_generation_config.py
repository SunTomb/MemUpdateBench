from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.enums import Difficulty
from mub.vnext.generation.config import PilotConfig, load_pilot_config
from mub.vnext.io import canonical_json_bytes


CONFIG_PATH = Path("configs/vnext/pilot.yaml")


def fixed_payload() -> dict[str, Any]:
    return load_pilot_config(CONFIG_PATH).model_dump(mode="json")


def _set_path(
    payload: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_fixed_config_computes_release_size() -> None:
    config = load_pilot_config(CONFIG_PATH)

    assert config.total_semantic_cores == 480
    assert config.total_tasks == 1440
    assert config.expected_split_tasks == {
        "train": 1008,
        "dev": 144,
        "test": 288,
    }
    assert config.families.repeated_same_slot_update.difficulties == [
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    ]


def test_config_canonical_serialization_round_trip_is_stable() -> None:
    config = load_pilot_config(CONFIG_PATH)
    reordered_payload = dict(reversed(config.model_dump(mode="json").items()))
    reordered = PilotConfig.model_validate(reordered_payload)

    canonical = canonical_json_bytes(config)
    restored = PilotConfig.model_validate_json(canonical)

    assert canonical_json_bytes(reordered) == canonical
    assert restored == config
    assert canonical_json_bytes(restored) == canonical


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unreviewed_axis",), 7),
        (("splits", "holdout"), 0.1),
        (("families", "unreviewed_family"), {}),
        (
            ("families", "repeated_same_slot_update", "unreviewed_axis"),
            7,
        ),
        (
            (
                "families",
                "repeated_same_slot_update",
                "same_name_distractors",
                "challenge",
            ),
            8,
        ),
        (
            ("families", "interleaved_multi_slot_update", "unreviewed_axis"),
            7,
        ),
        (
            ("families", "entity_attribute_grounding", "unreviewed_axis"),
            7,
        ),
        (("families", "noop_write_discipline", "unreviewed_axis"), 7),
        (("mechanism_slice", "unreviewed_axis"), 7),
        (("mechanism_slice", "conditions", 0, "unreviewed_axis"), 7),
        (("output", "unreviewed_axis"), 7),
    ],
)
def test_unknown_keys_at_every_model_level_are_rejected(
    path: tuple[str | int, ...],
    value: object,
) -> None:
    payload = fixed_payload()
    _set_path(payload, path, value)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("seed",), 0),
        (("surface_variants_per_core",), 0),
        (("cores_per_family",), 0),
        (("families", "repeated_same_slot_update", "update_depths"), [0, 4]),
        (
            (
                "families",
                "repeated_same_slot_update",
                "same_name_distractors",
                "easy",
            ),
            -1,
        ),
        (
            (
                "families",
                "interleaved_multi_slot_update",
                "active_object_counts",
                "easy",
            ),
            0,
        ),
        (
            (
                "families",
                "interleaved_multi_slot_update",
                "cross_slot_distractor_density",
                "hard",
            ),
            1.01,
        ),
        (("families", "noop_write_discipline", "noop_densities"), [-0.01]),
        (("mechanism_slice", "stale_counts"), [0, 16]),
    ],
)
def test_numeric_values_enforce_semantic_bounds(
    path: tuple[str, ...],
    invalid_value: object,
) -> None:
    payload = fixed_payload()
    _set_path(payload, path, invalid_value)

    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("seed", True),
        ("surface_variants_per_core", 3.0),
        ("cores_per_family", "120"),
    ],
)
def test_fixed_count_numbers_reject_scalar_coercion(
    field: str,
    invalid_value: object,
) -> None:
    payload = fixed_payload()
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)


def test_split_ratios_must_sum_to_one() -> None:
    payload = fixed_payload()
    payload["splits"] = {"train": 0.60, "dev": 0.10, "test": 0.20}

    with pytest.raises(ValidationError, match="sum to 1"):
        PilotConfig.model_validate(payload)


def test_split_ratios_must_allocate_whole_cores_per_family() -> None:
    payload = fixed_payload()
    payload["splits"] = {"train": 0.71, "dev": 0.09, "test": 0.20}

    with pytest.raises(ValidationError, match="whole number of cores"):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("staging_dir", "release_dir"),
    [
        ("data/vnext/pilot", "./data/vnext/pilot"),
        (r"data\vnext\pilot", "data/vnext/pilot"),
        (r"DATA\VNEXT\PILOT", "data/vnext/pilot"),
        ("data/vnext/staging/../pilot", "data/vnext/pilot"),
    ],
)
def test_output_paths_reject_cross_platform_lexical_aliases(
    staging_dir: str,
    release_dir: str,
) -> None:
    payload = fixed_payload()
    payload["output"] = {
        "staging_dir": staging_dir,
        "release_dir": release_dir,
    }

    with pytest.raises(ValidationError, match="must be distinct"):
        PilotConfig.model_validate(payload)


def test_all_four_pilot_families_must_be_enabled() -> None:
    payload = fixed_payload()
    payload["families"]["noop_write_discipline"]["enabled"] = False

    with pytest.raises(ValidationError, match="all four Pilot families"):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("families", "repeated_same_slot_update", "update_depths"),
            [1, 1],
        ),
        (
            ("families", "entity_attribute_grounding", "entity_conditions"),
            ["distinct", "distinct"],
        ),
        (("mechanism_slice", "stale_counts"), [1, 1]),
        (
            ("mechanism_slice", "conditions"),
            [
                {"context_order": "chronological", "context_annotation": "none"},
                {"context_order": "chronological", "context_annotation": "none"},
            ],
        ),
    ],
)
def test_reviewed_axes_must_not_contain_duplicates(
    path: tuple[str | int, ...],
    value: object,
) -> None:
    payload = fixed_payload()
    _set_path(payload, path, value)

    with pytest.raises(ValidationError, match="unique"):
        PilotConfig.model_validate(payload)


def test_family_difficulty_axes_cover_exact_pilot_difficulties() -> None:
    payload = fixed_payload()
    payload["families"]["entity_attribute_grounding"]["difficulties"] = [
        "easy",
        "medium",
    ]

    with pytest.raises(ValidationError, match="easy, medium, and hard"):
        PilotConfig.model_validate(payload)


@pytest.mark.parametrize("root", ["- not\n- a\n- mapping\n", ""])
def test_loader_rejects_non_mapping_yaml_root_cleanly(
    tmp_path: Path,
    root: str,
) -> None:
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(root, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML root must be a mapping"):
        load_pilot_config(config_path)


def test_loader_rejects_unsafe_yaml_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        "!!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="constructor|tag"):
        load_pilot_config(config_path)
