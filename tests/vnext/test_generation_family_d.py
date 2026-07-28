from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import mub.vnext.generation.family_d as family_d_module
from mub.vnext.contracts import ActionScope, Difficulty, EventRole, Operation, Split
from mub.vnext.contracts.task import GoldAction
from mub.vnext.generation import (
    GenerationContext,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.io import semantic_task_hash
from mub.vnext.validation import validate_task_semantics
from mub.vnext.validation.replay import replay_actions, validate_gold_replay
from mub.vnext.validation.task import validate_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
DENSITY_TO_DIFFICULTY = {
    0.25: Difficulty.EASY,
    0.50: Difficulty.MEDIUM,
    0.75: Difficulty.HARD,
}
TRAPS = (
    "semantic_near_miss",
    "duplicate_current",
    "other_entity_correction",
    "other_attribute_correction",
)


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return generate_family_d_cores(config)


def _actions(events):
    return [
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
        for index, event in enumerate(events)
    ]


def _replay(events):
    return replay_actions(_actions(events))


def test_family_d_has_exact_grid_balance_counts_and_densities(config, cores):
    assert len(cores) == 120
    assert [core.core_index for core in cores] == list(range(120))
    assert Counter(
        (core.stratification["configured_noop_density"], core.stratification["trap_type"])
        for core in cores
    ) == {(density, trap): 10 for density in DENSITY_TO_DIFFICULTY for trap in TRAPS}

    for core in cores:
        density = core.stratification["configured_noop_density"]
        noop_count = int(8 * density)
        assert core.difficulty is DENSITY_TO_DIFFICULTY[density]
        assert core.profile["noop_density"] == density
        assert core.profile["write_trap_type"] == core.stratification["trap_type"]
        assert core.stratification["num_events"] == len(core.events) == 8
        assert core.stratification["noop_count"] == noop_count
        assert core.stratification["true_write_count"] == 8 - noop_count
        assert core.stratification["observed_noop_density"] == pytest.approx(density)
        assert core.stratification["num_target_updates"] == 0
        assert "num_updates" not in core.stratification


def test_family_d_noops_never_mutate_and_counts_match_replay(cores):
    for core in cores:
        replay = _replay(core.events)
        assert replay.mutation_count == core.stratification["true_write_count"]
        assert sum(len(history) for history in replay.version_history.values()) == replay.mutation_count
        assert replay.final_state[core.query_targets[0].canonical_id] == core.expected_answer

        for index, event in enumerate(core.events):
            if event.operation is not Operation.NOOP:
                continue
            before = _replay(core.events[:index]) if index else None
            after = _replay(core.events[: index + 1])
            assert after.mutation_count == (before.mutation_count if before else 0)
            assert dict(after.final_state) == (dict(before.final_state) if before else {})
            assert dict(after.version_history) == (dict(before.version_history) if before else {})


def test_family_d_target_history_grows_only_on_target_writes(cores):
    for core in cores:
        target_id = core.query_targets[0].canonical_id
        replay = _replay(core.events)
        target_writes = [
            event
            for event in core.events
            if event.operation is not Operation.NOOP
            and any(key.canonical_id == target_id for key in event.object_keys)
        ]
        assert len(replay.version_history[target_id]) == len(target_writes) == 1
        assert all(
            target_id not in {key.canonical_id for key in event.object_keys}
            for event in core.events
            if event.operation is Operation.NOOP
        )


def test_family_d_generation_does_not_mutate_config(config):
    before = config.model_dump(mode="json")
    generate_family_d_cores(config)
    assert config.model_dump(mode="json") == before


def test_family_d_traps_are_target_isolated_and_duplicate_is_visible(cores):
    for core in cores:
        target = core.query_targets[0]
        trap = core.stratification["trap_type"]
        trap_events = [event for event in core.events if event.metadata.get("trap_type") == trap]
        assert len(trap_events) == 1
        event = trap_events[0]
        if trap in {"semantic_near_miss", "duplicate_current"}:
            assert event.operation is Operation.NOOP
            assert not event.object_keys
            assert event.value is None
        ambiguity_events = [
            item
            for item in core.events
            if item.metadata.get("allow_accepted_answer_ambiguity") is True
        ]
        if trap == "duplicate_current":
            assert ambiguity_events == [event]
            assert event.role is EventRole.NOOP_NEAR_MISS
            assert event.metadata["allow_accepted_answer_ambiguity"] is True
            assert core.stratification["duplicate_current_count"] == 1
            statement = event.metadata["surface_statement"]
            assert target.entity in statement
            assert target.attribute in statement
            assert str(core.expected_answer) in statement
        else:
            assert not ambiguity_events
            assert "allow_accepted_answer_ambiguity" not in event.metadata
            assert core.stratification["duplicate_current_count"] == 0
        if trap == "other_entity_correction":
            assert event.operation is Operation.ADD
            assert event.value != core.expected_answer
            key = event.object_keys[0]
            assert key.entity != target.entity
            assert key.attribute == target.attribute
            assert key.canonical_id != target.canonical_id
        if trap == "other_attribute_correction":
            assert event.operation is Operation.ADD
            assert event.value != core.expected_answer
            key = event.object_keys[0]
            assert key.entity == target.entity
            assert key.attribute != target.attribute
            assert key.canonical_id != target.canonical_id
        assert all(
            event.value != core.expected_answer
            for event in core.events
            if event.metadata.get("event_kind") == "ordinary_write"
        )
        assert all(
            target not in event.object_keys
            for event in core.events
            if event.metadata.get("trap_type") in {
                "other_entity_correction",
                "other_attribute_correction",
            }
        )


def test_family_d_generation_is_deterministic_and_ids_are_unique(config, cores):
    regenerated = generate_family_d_cores(config)
    assert [core.model_dump(mode="json") for core in regenerated] == [
        core.model_dump(mode="json") for core in cores
    ]
    assert len({core.core_id for core in cores}) == 120
    assert len({core.trajectory_id for core in cores}) == 120


def test_family_d_core_id_payload_excludes_coordinates_and_object_type(config, monkeypatch):
    payloads = []
    original = family_d_module.core_id

    def capture(family, payload):
        payloads.append(payload)
        return original(family, payload)

    monkeypatch.setattr(family_d_module, "core_id", capture)
    generate_family_d_cores(config)
    forbidden = {"seed", "index", "core_index", "cell_index", "example_index", "axis_index", "object_type"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert len(payloads) == 120
    assert all(not (keys(payload) & forbidden) for payload in payloads)


def test_family_d_three_rendered_variants_validate_replay_and_share_hash(config, cores):
    context = GenerationContext(config=config, code_revision="family-d-test")
    for core in cores:
        tasks = [
            render_core(core, split=Split.TEST, surface_variant=variant, context=context)
            for variant in range(3)
        ]
        assert len({task.task_id for task in tasks}) == 3
        assert len({task.source.source_id for task in tasks}) == 3
        assert len({event.event_id for task in tasks for event in task.events}) == 24
        assert len({action.action_id for task in tasks for action in task.gold.actions}) == 24
        assert len({query.query_id for task in tasks for query in task.queries}) == 3
        assert len({semantic_task_hash(task) for task in tasks}) == 1
        assert all(validate_task(task).valid for task in tasks)
        assert all(validate_gold_replay(task).valid for task in tasks)
        assert all(not validate_task_semantics(task).issues for task in tasks)


def test_family_a_through_c_generation_smoke(config):
    assert len(generate_family_a_cores(config)) == 120
    assert len(generate_family_b_cores(config)) == 120
    assert len(generate_family_c_cores(config)) == 120
