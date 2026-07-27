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
from mub.vnext.generation.family_b import generate_family_b_cores
from mub.vnext.validation.replay import replay_actions, validate_gold_replay
from mub.vnext.validation.task import validate_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
PATTERNS = ("round_robin", "burst", "adversarial_adjacent")


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return generate_family_b_cores(config)


def _target_events(core):
    target = core.query_targets[0]
    return [event for event in core.events if target in event.object_keys]


def _target_indices(core):
    target = core.query_targets[0]
    return [index for index, event in enumerate(core.events) if target in event.object_keys]


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


def _active_keys(core):
    return tuple(
        dict.fromkeys(
            key.canonical_id
            for event in core.events
            for key in event.object_keys
        )
    )


def test_family_b_has_exact_balanced_configured_axes(cores):
    assert len(cores) == 120
    assert [core.core_index for core in cores] == list(range(120))
    assert {core.profile["update_depth"] for core in cores} == {1, 4, 16}
    assert {core.difficulty for core in cores} == {
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    }
    assert {core.stratification["interleaving_pattern"] for core in cores} == set(PATTERNS)
    assert Counter(core.stratification["interleaving_pattern"] for core in cores) == {
        pattern: 40 for pattern in PATTERNS
    }
    cells = Counter(
        (
            core.profile["update_depth"],
            core.difficulty,
            core.stratification["interleaving_pattern"],
        )
        for core in cores
    )
    assert set(cells.values()) == {4, 5}
    for core in cores:
        cell = (
            core.profile["update_depth"],
            core.difficulty,
            core.stratification["interleaving_pattern"],
        )
        assert core.stratification["allocation_cell_count"] == cells[cell]


def test_family_b_matched_patterns_share_target_trajectory_but_change_indices(cores):
    for start in range(0, 120, 3):
        matched = cores[start : start + 3]
        assert {core.stratification["interleaving_pattern"] for core in matched} == set(PATTERNS)
        trajectories = [
            [
                (
                    event.operation,
                    event.value,
                    event.role,
                    dict(event.metadata),
                )
                for event in _target_events(core)
            ]
            for core in matched
        ]
        assert trajectories[0] == trajectories[1] == trajectories[2]
        assert len({tuple(_target_indices(core)) for core in matched}) == 3


def test_family_b_interleaving_pattern_semantics_are_explicit(cores):
    for start in range(0, 120, 3):
        by_pattern = {
            core.stratification["interleaving_pattern"]: core
            for core in cores[start : start + 3]
        }
        burst = by_pattern["burst"]
        assert _target_indices(burst) == list(range(burst.profile["update_depth"] + 1))
        first_indices = {}
        round_robin = by_pattern["round_robin"]
        for index, event in enumerate(round_robin.events):
            for key in event.object_keys:
                first_indices.setdefault(key.canonical_id, index)
        assert sorted(first_indices.values()) == list(range(round_robin.profile["active_object_count"]))
        assert _target_indices(round_robin)[0] == round_robin.profile["active_object_count"] - 1
        adversarial = by_pattern["adversarial_adjacent"]
        target_indices = _target_indices(adversarial)
        assert target_indices[:-1] == list(range(adversarial.profile["update_depth"]))
        assert target_indices[-1] == len(adversarial.events) - 1
        assert adversarial.events[-2].object_keys[0] != adversarial.query_targets[0]
        assert adversarial.events[-1].role is EventRole.LATEST_GOLD


def test_family_b_replay_preserves_distinct_active_objects_and_values(cores):
    for core in cores:
        target = core.query_targets[0]
        active_ids = _active_keys(core)
        assert len(core.query_targets) == 1
        assert len(active_ids) == core.profile["active_object_count"]
        assert target.canonical_id in active_ids
        replay = _replay(core)
        assert set(replay.final_state) == set(active_ids)
        assert replay.final_state[target.canonical_id] == core.expected_answer
        assert len(set(replay.final_state.values())) == len(replay.final_state)
        expected_histories = {}
        for event in core.events:
            identity = event.object_keys[0].canonical_id
            expected_histories.setdefault(identity, []).append(event.value)
        assert {
            identity: tuple(values)
            for identity, values in replay.version_history.items()
        } == {
            identity: tuple(values)
            for identity, values in expected_histories.items()
        }


def test_family_b_slot_trajectories_are_valid_and_roles_are_explicit(cores):
    for core in cores:
        target = core.query_targets[0]
        seen = set()
        latest_by_slot = {}
        for event in core.events:
            assert event.operation is not Operation.NOOP
            key = event.object_keys[0]
            identity = key.canonical_id
            if event.operation is Operation.ADD:
                assert identity not in seen
                seen.add(identity)
            else:
                assert event.operation is Operation.UPDATE
                assert identity in seen
            latest_by_slot[identity] = event.value
            if key == target:
                assert event.role in {EventRole.STALE_SAME_SLOT, EventRole.LATEST_GOLD}
            else:
                assert event.role is EventRole.SAME_ENTITY_OTHER_ATTRIBUTE
                assert event.metadata["distractor_kind"] == "cross_slot"
                assert event.metadata["target_relation"] == "same_entity_other_attribute"
        target_events = _target_events(core)
        assert target_events[0].operation is Operation.ADD
        assert sum(event.operation is Operation.UPDATE for event in target_events) == core.profile["update_depth"]
        assert all(event.role is EventRole.STALE_SAME_SLOT for event in target_events[:-1])
        assert target_events[-1].role is EventRole.LATEST_GOLD
        assert core.expected_answer == target_events[-1].value
        assert latest_by_slot[target.canonical_id] == core.expected_answer


def test_family_b_density_counts_and_stratification_are_exact(config, cores):
    family = config.families.interleaved_multi_slot_update
    required = {
        "num_events",
        "num_target_updates",
        "active_object_count",
        "cross_slot_distractor_count",
        "interleaving_pattern",
        "update_depth",
        "stale_count",
        "noop_count",
    }
    for core in cores:
        depth = core.profile["update_depth"]
        active_count = getattr(family.active_object_counts, core.difficulty.value)
        density = getattr(family.cross_slot_distractor_density, core.difficulty.value)
        base_event_count = active_count + depth
        expected_distractors = int(base_event_count * density + 0.5)
        non_target_updates = sum(
            event.operation is Operation.UPDATE
            and core.query_targets[0] not in event.object_keys
            for event in core.events
        )
        assert required <= set(core.stratification)
        assert "num_updates" not in core.profile
        assert "num_updates" not in core.stratification
        assert core.profile["active_object_count"] == active_count
        assert core.profile["cross_slot_interleaving"] == density
        assert core.stratification["base_event_count"] == base_event_count
        assert core.stratification["cross_slot_distractor_count"] == expected_distractors
        assert core.stratification["cross_slot_distractor_count"] == non_target_updates
        assert core.stratification["realized_cross_slot_distractor_density"] == pytest.approx(
            expected_distractors / base_event_count
        )
        assert core.stratification["num_events"] == len(core.events) == base_event_count + expected_distractors
        assert core.stratification["num_target_updates"] == depth
        assert core.stratification["update_depth"] == depth
        assert core.stratification["stale_count"] == depth
        assert core.stratification["noop_count"] == 0


def test_family_b_follows_reordered_valid_config_axes(config):
    family = config.families.interleaved_multi_slot_update.model_copy(
        update={
            "update_depths": [16, 1, 4],
            "difficulties": [Difficulty.HARD, Difficulty.EASY, Difficulty.MEDIUM],
            "interleaving_patterns": ["adversarial_adjacent", "burst", "round_robin"],
        }
    )
    families = config.families.model_copy(update={"interleaved_multi_slot_update": family})
    reordered = config.model_copy(update={"families": families})
    cores = generate_family_b_cores(reordered)
    assert [core.stratification["interleaving_pattern"] for core in cores[:9]] == [
        "adversarial_adjacent",
        "burst",
        "round_robin",
    ] * 3
    assert [core.profile["update_depth"] for core in cores[:9]] == [16] * 3 + [1] * 3 + [4] * 3
    assert [core.difficulty for core in cores[:9]] == [Difficulty.HARD] * 9


def test_family_b_is_deterministic_across_repeats_cwd_and_hash_seed(config, cores, tmp_path):
    expected = [core.model_dump(mode="json") for core in cores]
    assert expected == [
        core.model_dump(mode="json") for core in generate_family_b_cores(config)
    ]
    assert len({core.core_id for core in cores}) == 120
    assert len({core.trajectory_id for core in cores}) == 120
    script = """
import json
from pathlib import Path
from mub.vnext.generation import load_pilot_config
from mub.vnext.generation.family_b import generate_family_b_cores
cores = generate_family_b_cores(load_pilot_config(Path(r'{config}')))
print(json.dumps([c.model_dump(mode='json') for c in cores], sort_keys=True, separators=(',', ':')))
""".format(config=CONFIG_PATH)
    env = {**os.environ, "PYTHONHASHSEED": "17", "PYTHONPATH": str(ROOT)}
    first = subprocess.check_output([sys.executable, "-c", script], cwd=tmp_path, env=env, text=True)
    env["PYTHONHASHSEED"] = "991"
    second = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, env=env, text=True)
    assert json.loads(first) == json.loads(second) == expected


def test_family_b_renders_and_validates_representative_cores(config, cores):
    context = GenerationContext(config=config, code_revision="family-b-test")
    for pattern in PATTERNS:
        core = next(
            item
            for item in cores
            if item.difficulty is Difficulty.HARD
            and item.stratification["interleaving_pattern"] == pattern
        )
        task = render_core(core, split=Split.TEST, surface_variant=0, context=context)
        assert validate_task(task).valid
        assert validate_gold_replay(task).valid
        assert task.gold.gold_answers[task.queries[0].query_id] == core.expected_answer


def test_family_b_rejects_malformed_or_unsupported_configuration(config):
    with pytest.raises(TypeError):
        generate_family_b_cores(object())
    with pytest.raises(ValueError, match="120"):
        generate_family_b_cores(config.model_copy(update={"cores_per_family": 1}))
    family = config.families.interleaved_multi_slot_update.model_copy(update={"update_depths": [2]})
    families = config.families.model_copy(update={"interleaved_multi_slot_update": family})
    with pytest.raises(ValueError, match="update_depths"):
        generate_family_b_cores(config.model_copy(update={"families": families}))
    disabled = config.model_copy(deep=True)
    object.__setattr__(disabled.families.interleaved_multi_slot_update, "enabled", False)
    with pytest.raises(ValueError, match="enabled"):
        generate_family_b_cores(disabled)
    duplicate_patterns = config.model_copy(deep=True)
    object.__setattr__(
        duplicate_patterns.families.interleaved_multi_slot_update,
        "interleaving_patterns",
        ["round_robin", "burst", "adversarial_adjacent", "burst"],
    )
    with pytest.raises(ValueError, match="interleaving_patterns"):
        generate_family_b_cores(duplicate_patterns)
    duplicate_difficulties = config.model_copy(deep=True)
    object.__setattr__(
        duplicate_difficulties.families.interleaved_multi_slot_update,
        "difficulties",
        [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.HARD],
    )
    with pytest.raises(ValueError, match="difficulties"):
        generate_family_b_cores(duplicate_difficulties)


def test_family_b_generator_is_publicly_exported():
    from mub.vnext import generation

    assert generation.generate_family_b_cores is generate_family_b_cores
