from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import ActionScope, EventRole, Operation, Split
from mub.vnext.contracts.task import GoldAction
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.family_a import (
    generate_core_family_a_cores,
    generate_family_a_cores,
)
from mub.vnext.generation.family_b import (
    generate_core_family_b_cores,
    generate_family_b_cores,
)
from mub.vnext.generation.config import load_pilot_config
from mub.vnext.generation.render import render_core_with_catalog
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.validation.replay import replay_actions


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"
PILOT_CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
A_CONDITIONS = (
    "stale_burden",
    "duplicate_current",
    "other_attribute_distractor",
    "same_name_other_entity_distractor",
)
B_PATTERNS = ("round_robin", "burst", "adversarial_adjacent")


def _replay(core):
    return replay_actions(
        [
            GoldAction(
                action_id=f"a-{index}",
                event_id=f"e-{index}",
                operation=event.operation,
                scope=(
                    ActionScope.OBJECT
                    if event.operation is Operation.NOOP
                    else ActionScope.ATTRIBUTE
                ),
                target_object_keys=list(event.object_keys),
                value=event.value,
            )
            for index, event in enumerate(core.events)
        ]
    )


def _digest(cores) -> str:
    payload = [core.model_dump(mode="json") for core in cores]
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_core_config_declares_approved_family_a_and_b_axes() -> None:
    config = load_core_config(CORE_CONFIG_PATH)

    family_a = config.families.repeated_same_slot_update
    assert family_a.update_depths == [1, 2, 4, 8, 16, 32]
    assert family_a.conditions == list(A_CONDITIONS)

    family_b = config.families.interleaved_multi_slot_update
    assert family_b.update_depths == [1, 4, 16]
    assert family_b.active_object_counts == [2, 4, 8, 12]
    assert family_b.interleaving_patterns == list(B_PATTERNS)


def test_generation_context_freezes_and_hashes_core_config_without_aliasing() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    expected_bytes = canonical_json_bytes(config)
    context = GenerationContext(config=config, code_revision="task-637-core-ab")
    original_hash = context.config_sha256

    assert context.config is not config
    assert canonical_json_bytes(context.config) == expected_bytes
    assert original_hash == sha256_model(config)
    with pytest.raises(ValidationError, match="frozen"):
        context.config.seed = 1
    with pytest.raises(TypeError):
        context.config.families.repeated_same_slot_update.update_depths.append(64)

    config.seed += 1
    config.families.repeated_same_slot_update.update_depths.append(64)
    assert context.config_sha256 == original_hash
    assert context.config.families.repeated_same_slot_update.update_depths == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


def test_core_family_a_has_exact_depth_and_condition_marginals() -> None:
    cores = generate_core_family_a_cores(load_core_config(CORE_CONFIG_PATH))

    assert len(cores) == 480
    assert [core.core_index for core in cores] == list(range(480))
    assert Counter(core.profile["update_depth"] for core in cores) == {
        depth: 80 for depth in (1, 2, 4, 8, 16, 32)
    }
    cells = Counter(
        (core.profile["update_depth"], core.stratification["condition"])
        for core in cores
    )
    assert cells == {
        cell: 20 for cell in product((1, 2, 4, 8, 16, 32), A_CONDITIONS)
    }


def test_core_family_a_conditions_are_semantically_explicit_and_depth_32_replays() -> None:
    cores = generate_core_family_a_cores(load_core_config(CORE_CONFIG_PATH))

    for core in cores:
        target = core.query_targets[0]
        condition = core.stratification["condition"]
        stale = [event for event in core.events if event.role is EventRole.STALE_SAME_SLOT]
        duplicate = [event for event in core.events if event.role is EventRole.DUPLICATE_CURRENT]
        other_attribute = [
            event
            for event in core.events
            if event.role is EventRole.SAME_ENTITY_OTHER_ATTRIBUTE
        ]
        same_name = [
            event for event in core.events if event.role is EventRole.SAME_NAME_OTHER_ENTITY
        ]
        assert len(stale) == core.profile["update_depth"]
        assert len(duplicate) == (condition == "duplicate_current")
        assert len(other_attribute) == (condition == "other_attribute_distractor")
        assert len(same_name) == (condition == "same_name_other_entity_distractor")
        assert all(event.value != core.expected_answer for event in stale)
        assert all(
            event.value != core.expected_answer
            for event in (*other_attribute, *same_name)
        )
        assert _replay(core).final_state[target.canonical_id] == core.expected_answer

    depth_32 = [core for core in cores if core.profile["update_depth"] == 32]
    for core in depth_32:
        target = core.query_targets[0]
        target_values = [
            event.value
            for event in core.events
            if target in event.object_keys and event.role is not EventRole.DUPLICATE_CURRENT
        ]
        assert len(target_values) == 33
        assert len(set(target_values)) == 33


def test_core_family_b_balances_depth_pattern_cells_inside_each_object_stratum() -> None:
    cores = generate_core_family_b_cores(load_core_config(CORE_CONFIG_PATH))

    assert len(cores) == 480
    assert [core.core_index for core in cores] == list(range(480))
    assert Counter(core.profile["active_object_count"] for core in cores) == {
        count: 120 for count in (2, 4, 8, 12)
    }
    for active_count in (2, 4, 8, 12):
        stratum = [
            core for core in cores if core.profile["active_object_count"] == active_count
        ]
        cells = Counter(
            (
                core.profile["update_depth"],
                core.stratification["interleaving_pattern"],
            )
            for core in stratum
        )
        assert set(cells) == set(product((1, 4, 16), B_PATTERNS))
        assert max(cells.values()) - min(cells.values()) <= 1
        assert Counter(
            core.stratification["interleaving_pattern"] for core in stratum
        ) == {pattern: 40 for pattern in B_PATTERNS}


def test_core_family_b_count_12_uses_exact_multi_entity_replay() -> None:
    cores = generate_core_family_b_cores(load_core_config(CORE_CONFIG_PATH))

    for core in cores:
        identities = {
            key.canonical_id: key
            for event in core.events
            for key in event.object_keys
        }
        replay = _replay(core)
        assert len(identities) == core.profile["active_object_count"]
        assert set(replay.final_state) == set(identities)
        assert replay.final_state[core.query_targets[0].canonical_id] == core.expected_answer
        if core.profile["active_object_count"] == 12:
            assert len({key.entity for key in identities.values()}) > 1
            assert all(key.subkey is None for key in identities.values())


def test_representative_core_family_a_and_b_cores_render_with_core_context() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(
        config=config,
        code_revision="task-637-core-render",
        generator_name="memupdatebench_vnext_core",
    )
    family_a = generate_core_family_a_cores(config)
    family_b = generate_core_family_b_cores(config)
    representatives = [
        next(
            core
            for core in family_a
            if core.profile["update_depth"] == 32
            and core.stratification["condition"] == condition
        )
        for condition in A_CONDITIONS
    ] + [
        next(
            core
            for core in family_b
            if core.profile["active_object_count"] == 12
            and core.profile["update_depth"] == 16
            and core.stratification["interleaving_pattern"] == pattern
        )
        for pattern in B_PATTERNS
    ]

    tasks = [
        render_core_with_catalog(
            core,
            split=Split.TEST,
            surface_variant=0,
            context=context,
            surface_catalog=CORE_SURFACE_CATALOG_V1,
        )
        for core in representatives
    ]

    assert len(tasks) == 7
    assert len({task.task_id for task in tasks}) == 7


def test_core_family_a_and_b_are_deterministic() -> None:
    config = load_core_config(CORE_CONFIG_PATH)

    first_a = generate_core_family_a_cores(config)
    second_a = generate_core_family_a_cores(config)
    first_b = generate_core_family_b_cores(config)
    second_b = generate_core_family_b_cores(config)

    assert _digest(first_a) == _digest(second_a)
    assert _digest(first_b) == _digest(second_b)
    assert len({core.core_id for core in first_a}) == 480
    assert len({core.core_id for core in first_b}) == 480


def test_pilot_family_a_and_b_outputs_remain_byte_exact() -> None:
    config = load_pilot_config(PILOT_CONFIG_PATH)

    assert _digest(generate_family_a_cores(config)) == (
        "0f692b460edd16785d4f66c59d2653dff9da1f28c1b501ba998133b2ff9f1242"
    )
    assert _digest(generate_family_b_cores(config)) == (
        "ba781ec67195e1941fd4af4e67369387dce2d50d01b1abb26bb1863ef1f6cefc"
    )
