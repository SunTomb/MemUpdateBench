from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from mub.vnext.contracts import ActionScope, Difficulty, EventRole, Operation, Split
from mub.vnext.contracts.task import GoldAction
from mub.vnext.generation import GenerationContext, load_pilot_config, render_core
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.validation.replay import replay_actions, validate_gold_replay
from mub.vnext.validation.task import validate_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return generate_family_a_cores(config)


def test_family_a_has_exact_balanced_depth_and_difficulty_axes(cores):
    assert len(cores) == 120
    assert {core.profile["update_depth"] for core in cores} == {1, 4, 16}
    assert {core.difficulty for core in cores} == {
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    }
    assert [core.profile["update_depth"] for core in cores[:9]] == [1, 4, 16] * 3
    assert [core.difficulty for core in cores[:9]] == [
        Difficulty.EASY,
        Difficulty.EASY,
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.MEDIUM,
        Difficulty.MEDIUM,
        Difficulty.HARD,
        Difficulty.HARD,
        Difficulty.HARD,
    ]
    assert [core.core_index for core in cores] == list(range(120))


def test_family_a_follows_reordered_valid_config_axes(config):
    family = config.families.repeated_same_slot_update.model_copy(
        update={
            "update_depths": [16, 1, 4],
            "difficulties": [Difficulty.HARD, Difficulty.EASY, Difficulty.MEDIUM],
        }
    )
    families = config.families.model_copy(update={"repeated_same_slot_update": family})
    reordered = config.model_copy(update={"families": families})
    cores = generate_family_a_cores(reordered)

    assert [core.profile["update_depth"] for core in cores[:9]] == [16, 1, 4] * 3
    assert [core.difficulty for core in cores[:9]] == [
        Difficulty.HARD,
        Difficulty.HARD,
        Difficulty.HARD,
        Difficulty.EASY,
        Difficulty.EASY,
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.MEDIUM,
        Difficulty.MEDIUM,
    ]
    assert len(cores) == 120
    assert {core.profile["update_depth"] for core in cores} == {1, 4, 16}
    assert {core.difficulty for core in cores} == {
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    }
    actual = Counter((core.profile["update_depth"], core.difficulty) for core in cores)
    assert set(actual.values()) == {13, 14}
    for core in cores:
        assert core.stratification["depth_difficulty_cell_count"] == actual[
            (core.profile["update_depth"], core.difficulty)
        ]


def test_family_a_target_chain_has_exact_depth_and_final_answer(cores):
    for core in cores:
        target = core.query_targets[0]
        target_events = [event for event in core.events if target in event.object_keys]
        depth = core.profile["update_depth"]
        assert len(core.query_targets) == 1
        assert target_events[0].operation is Operation.ADD
        assert sum(event.operation is Operation.UPDATE for event in target_events) == depth
        assert all(event.operation is not Operation.NOOP for event in target_events)
        assert target_events[-1].role is EventRole.LATEST_GOLD
        assert all(event.role is EventRole.STALE_SAME_SLOT for event in target_events[:-1])
        assert core.expected_answer == target_events[-1].value
        assert all(event.value != core.expected_answer for event in target_events[:-1])
        assert len({event.value for event in target_events[:-1]}) == len(target_events) - 1
        assert core.profile["stale_count"] == depth
        assert core.stratification["num_target_updates"] == depth
        assert core.stratification["stale_same_slot_count"] == depth


def test_family_a_distractors_have_distinct_identity_and_preserve_target(config, cores):
    family = config.families.repeated_same_slot_update
    expected = {
        difficulty: (
            getattr(family.same_name_distractors, difficulty.value),
            getattr(family.same_entity_other_attribute, difficulty.value),
        )
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    }
    for core in cores:
        same_name_count, other_attr_count = expected[core.difficulty]
        target = core.query_targets[0]
        same_name = [
            event for event in core.events if event.role is EventRole.SAME_NAME_OTHER_ENTITY
        ]
        other_attr = [
            event
            for event in core.events
            if event.role is EventRole.SAME_ENTITY_OTHER_ATTRIBUTE
        ]
        assert len(same_name) == same_name_count
        assert len(other_attr) == other_attr_count
        all_distractor_keys = [key for event in (*same_name, *other_attr) for key in event.object_keys]
        assert target not in all_distractor_keys
        assert len(set(all_distractor_keys)) == len(all_distractor_keys)
        assert all(event.operation is Operation.ADD for event in (*same_name, *other_attr))
        assert all(event.role is not EventRole.DUPLICATE_CURRENT for event in core.events)
        assert EventRole.STALE_SAME_SLOT.value != EventRole.DUPLICATE_CURRENT.value
        replay = replay_actions(
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
        assert replay.final_state[target.canonical_id] == core.expected_answer


def test_family_a_near_misses_are_safe_noops(config, cores):
    family = config.families.repeated_same_slot_update
    expected = {
        difficulty: getattr(family.noop_near_miss, difficulty.value)
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    }
    for core in cores:
        near_misses = [
            event for event in core.events if event.role is EventRole.NOOP_NEAR_MISS
        ]
        assert len(near_misses) == expected[core.difficulty]
        for event in near_misses:
            assert event.operation is Operation.NOOP
            assert event.object_keys == ()
            assert event.value is None
            assert isinstance(event.metadata["surface_statement"], str)
            assert event.metadata["surface_statement"].strip()


def test_family_a_profile_and_stratification_are_separate_and_machine_readable(cores):
    profile_keys = {
        "update_depth",
        "stale_count",
        "active_object_count",
        "noop_density",
        "entity_ambiguity",
        "attribute_ambiguity",
        "context_length",
        "query_type",
        "version_metadata",
    }
    count_keys = {
        "num_events",
        "num_target_updates",
        "same_name_distractor_count",
        "same_entity_other_attribute_count",
        "noop_count",
        "stale_same_slot_count",
    }
    for core in cores:
        assert profile_keys <= set(core.profile)
        assert count_keys <= set(core.stratification)
        assert "num_updates" not in core.profile
        assert "num_updates" not in core.stratification
        assert core.stratification["num_events"] == len(core.events)
        assert core.stratification["noop_count"] == sum(
            event.operation is Operation.NOOP for event in core.events
        )
        assert core.profile["noop_density"] == pytest.approx(
            core.stratification["noop_count"] / len(core.events)
        )
        assert core.profile["active_object_count"] == 1 + core.stratification[
            "same_name_distractor_count"
        ] + core.stratification["same_entity_other_attribute_count"]


def test_family_a_is_deterministic_and_semantic_ids_change_with_index(config, cores):
    assert [core.model_dump(mode="json") for core in cores] == [
        core.model_dump(mode="json") for core in generate_family_a_cores(config)
    ]
    assert len({core.core_id for core in cores}) == 120
    assert len({core.trajectory_id for core in cores}) == 120
    assert all(core.core_id.startswith("core_") for core in cores)
    assert all(core.trajectory_id.startswith("trajectory_") for core in cores)
    assert cores[0].core_id != cores[1].core_id
    assert cores[0].trajectory_id != cores[1].trajectory_id


def test_family_a_is_independent_of_hash_seed_and_cwd(tmp_path):
    script = """
import json
from pathlib import Path
from mub.vnext.generation import load_pilot_config
from mub.vnext.generation.family_a import generate_family_a_cores
cores = generate_family_a_cores(load_pilot_config(Path(r'{config}')))
print(json.dumps([c.model_dump(mode='json') for c in cores], sort_keys=True, separators=(',', ':')))
""".format(config=CONFIG_PATH)
    env = {**os.environ, "PYTHONHASHSEED": "17", "PYTHONPATH": str(ROOT)}
    first = subprocess.check_output([sys.executable, "-c", script], cwd=tmp_path, env=env, text=True)
    env["PYTHONHASHSEED"] = "991"
    second = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, env=env, text=True)
    assert json.loads(first) == json.loads(second)


def test_family_a_renders_and_validates_representative_cores(config, cores):
    context = GenerationContext(config=config, code_revision="family-a-test")
    for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        core = next(item for item in cores if item.difficulty is difficulty)
        task = render_core(core, split=Split.TEST, surface_variant=0, context=context)
        assert validate_task(task).valid
        assert validate_gold_replay(task).valid
        assert task.gold.gold_answers[task.queries[0].query_id] == core.expected_answer


def test_family_a_rejects_malformed_or_unsupported_configuration(config):
    with pytest.raises(TypeError):
        generate_family_a_cores(object())
    with pytest.raises(ValueError, match="120"):
        generate_family_a_cores(config.model_copy(update={"cores_per_family": 1}))
    family = config.families.repeated_same_slot_update.model_copy(update={"update_depths": [2]})
    families = config.families.model_copy(update={"repeated_same_slot_update": family})
    unsupported = config.model_copy(update={"families": families})
    with pytest.raises(ValueError, match="update_depths"):
        generate_family_a_cores(unsupported)
    disabled = config.model_copy(deep=True)
    object.__setattr__(disabled.families.repeated_same_slot_update, "enabled", False)
    with pytest.raises(ValueError, match="enabled"):
        generate_family_a_cores(disabled)


def test_family_a_generator_is_publicly_exported():
    from mub.vnext import generation

    assert generation.generate_family_a_cores is generate_family_a_cores
