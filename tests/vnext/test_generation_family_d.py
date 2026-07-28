from __future__ import annotations

from collections import Counter, defaultdict
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
from mub.vnext.generation.catalogs import CANONICAL_ATTRIBUTES
from mub.vnext.generation.core import CoreEvent
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
EVENT_COUNT = 12


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


def _trap_event(core):
    trap_type = core.stratification["trap_type"]
    events = [event for event in core.events if event.metadata.get("trap_type") == trap_type]
    assert len(events) == 1
    return events[0]


def test_family_d_has_exact_grid_balance_and_derived_12_event_counters(config, cores):
    assert len(cores) == 120
    assert [core.core_index for core in cores] == list(range(120))
    assert Counter(
        (core.stratification["configured_noop_density"], core.stratification["trap_type"])
        for core in cores
    ) == {(density, trap): 10 for density in DENSITY_TO_DIFFICULTY for trap in TRAPS}

    for core in cores:
        density = core.stratification["configured_noop_density"]
        observed_noops = sum(event.operation is Operation.NOOP for event in core.events)
        observed_writes = sum(event.operation is not Operation.NOOP for event in core.events)
        observed_target_updates = sum(
            event.operation is Operation.UPDATE
            and core.query_targets[0] in event.object_keys
            for event in core.events
        )
        observed_duplicates = sum(
            event.metadata.get("trap_type") == "duplicate_current"
            for event in core.events
        )
        assert core.difficulty is DENSITY_TO_DIFFICULTY[density]
        assert core.profile["noop_density"] == density
        assert core.profile["write_trap_type"] == core.stratification["trap_type"]
        assert core.profile["context_length"] == len(core.events) == EVENT_COUNT
        assert core.stratification["num_events"] == len(core.events)
        assert core.stratification["noop_count"] == observed_noops == int(EVENT_COUNT * density)
        assert core.stratification["true_write_count"] == observed_writes
        assert core.stratification["num_target_updates"] == observed_target_updates == 0
        assert core.stratification["duplicate_current_count"] == observed_duplicates
        assert core.stratification["observed_noop_density"] == pytest.approx(
            observed_noops / len(core.events)
        )
        assert core.stratification["observed_noop_density"] == density
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


def test_family_d_corrections_have_real_before_after_history_and_isolate_target(cores):
    correction_types = {"other_entity_correction", "other_attribute_correction"}
    for core in (item for item in cores if item.stratification["trap_type"] in correction_types):
        target = core.query_targets[0]
        trap = _trap_event(core)
        assert trap.operation is Operation.UPDATE
        assert trap.metadata["lifecycle"] == "correction_after"
        key = trap.object_keys[0]
        setup_events = [
            event
            for event in core.events
            if event.metadata.get("lifecycle") == "correction_before"
            and event.object_keys == (key,)
        ]
        assert len(setup_events) == 1
        setup = setup_events[0]
        assert setup.operation is Operation.ADD
        assert core.events.index(setup) < core.events.index(trap)
        target_setup = next(
            event for event in core.events if event.metadata.get("lifecycle") == "target_current"
        )
        assert core.events.index(target_setup) < core.events.index(trap)
        assert len({setup.value, trap.value, core.expected_answer}) == 3
        history = _replay(core.events).version_history[key.canonical_id]
        assert tuple(history) == (setup.value, trap.value)
        assert key.canonical_id != target.canonical_id
        if core.stratification["trap_type"] == "other_entity_correction":
            assert key.entity != target.entity
            assert key.attribute == target.attribute
        else:
            assert key.entity == target.entity
            assert key.attribute != target.attribute


def test_family_d_duplicate_current_is_semantic_noop_at_contract_boundary(cores):
    for core in cores:
        target = core.query_targets[0]
        trap_type = core.stratification["trap_type"]
        event = _trap_event(core)
        if trap_type in {"semantic_near_miss", "duplicate_current"}:
            assert event.operation is Operation.NOOP
            assert not event.object_keys
            assert event.value is None
        ambiguity_events = [
            item
            for item in core.events
            if item.metadata.get("allow_accepted_answer_ambiguity") is True
        ]
        if trap_type == "duplicate_current":
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


def test_family_d_filler_values_exclude_target_answer(cores):
    for core in cores:
        assert all(
            event.value != core.expected_answer
            for event in core.events
            if event.metadata.get("lifecycle") == "independent_current"
        )


def test_family_d_seeded_schedule_varies_signatures_and_trap_positions_per_cell(cores):
    signatures = defaultdict(set)
    trap_positions = defaultdict(set)
    for core in cores:
        cell = (
            core.stratification["configured_noop_density"],
            core.stratification["trap_type"],
        )
        signatures[cell].add(tuple(event.operation for event in core.events))
        trap_positions[cell].add(core.events.index(_trap_event(core)))
    assert set(signatures) == {
        (density, trap) for density in DENSITY_TO_DIFFICULTY for trap in TRAPS
    }
    assert all(len(cell_signatures) > 1 for cell_signatures in signatures.values())
    assert all(len(positions) > 1 for positions in trap_positions.values())


def test_family_d_other_attribute_corrections_are_balanced(cores):
    for density in DENSITY_TO_DIFFICULTY:
        attributes = Counter(
            _trap_event(core).object_keys[0].attribute
            for core in cores
            if core.stratification["configured_noop_density"] == density
            and core.stratification["trap_type"] == "other_attribute_correction"
        )
        assert set(attributes) == set(CANONICAL_ATTRIBUTES)
        assert max(attributes.values()) - min(attributes.values()) <= 1


def test_family_d_generation_does_not_mutate_config(config):
    before = config.model_dump(mode="json")
    generate_family_d_cores(config)
    assert config.model_dump(mode="json") == before


def test_family_d_generation_is_deterministic_and_ids_are_unique(config, cores):
    regenerated = generate_family_d_cores(config)
    assert [core.model_dump(mode="json") for core in regenerated] == [
        core.model_dump(mode="json") for core in cores
    ]
    assert len({core.core_id for core in cores}) == 120
    assert len({core.trajectory_id for core in cores}) == 120


def test_family_d_core_id_payload_excludes_admin_surface_control_and_object_type(config, monkeypatch):
    payloads = []
    original = family_d_module.core_id

    def capture(family, payload):
        payloads.append(payload)
        return original(family, payload)

    monkeypatch.setattr(family_d_module, "core_id", capture)
    generate_family_d_cores(config)
    forbidden = {
        "seed",
        "index",
        "core_index",
        "cell_index",
        "example_index",
        "axis_index",
        "object_type",
        "metadata",
        "surface_statement",
        "allow_accepted_answer_ambiguity",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert len(payloads) == 120
    assert all(not (keys(payload) & forbidden) for payload in payloads)
    assert all(
        {event["lifecycle"] for event in payload["events"]}
        for payload in payloads
    )


def test_family_d_semantic_id_ignores_surface_and_validator_control_metadata(cores):
    core = next(item for item in cores if item.stratification["trap_type"] == "duplicate_current")
    payload = core.model_dump(mode="python")
    for event in payload["events"]:
        if event["metadata"].get("trap_type") == "duplicate_current":
            event["metadata"]["surface_statement"] = "Administrative wording changed."
            event["metadata"]["allow_accepted_answer_ambiguity"] = False
            event["metadata"]["admin_coordinate"] = 999
    modified_events = tuple(CoreEvent.model_validate(event) for event in payload["events"])
    original_semantics = family_d_module._semantic_payload(
        core.events,
        core.query_targets[0],
        core.stratification["configured_noop_density"],
        core.stratification["trap_type"],
    )
    modified_semantics = family_d_module._semantic_payload(
        modified_events,
        core.query_targets[0],
        core.stratification["configured_noop_density"],
        core.stratification["trap_type"],
    )
    assert modified_semantics == original_semantics
    assert family_d_module.core_id(core.task_family.value, modified_semantics) == core.core_id


def test_family_d_trajectory_is_invariant_to_allocation_index(config, cores):
    reference = cores[0]
    target = reference.query_targets[0]
    density = reference.stratification["configured_noop_density"]
    trap_type = reference.stratification["trap_type"]
    first = family_d_module._build_core(
        config,
        core_index=0,
        example_index=4,
        target=target,
        density=density,
        trap_type=trap_type,
    )
    relocated = family_d_module._build_core(
        config,
        core_index=999,
        example_index=4,
        target=target,
        density=density,
        trap_type=trap_type,
    )
    assert first.core_id == relocated.core_id
    assert first.trajectory_id == relocated.trajectory_id
    assert first.core_index != relocated.core_index


def test_family_d_three_rendered_variants_validate_replay_and_share_hash(config, cores):
    context = GenerationContext(config=config, code_revision="family-d-test")
    for core in cores:
        tasks = [
            render_core(core, split=Split.TEST, surface_variant=variant, context=context)
            for variant in range(3)
        ]
        assert len({task.task_id for task in tasks}) == 3
        assert len({task.source.source_id for task in tasks}) == 3
        assert len({event.event_id for task in tasks for event in task.events}) == 36
        assert len({action.action_id for task in tasks for action in task.gold.actions}) == 36
        assert len({query.query_id for task in tasks for query in task.queries}) == 3
        assert len({semantic_task_hash(task) for task in tasks}) == 1
        assert all(validate_task(task).valid for task in tasks)
        assert all(validate_gold_replay(task).valid for task in tasks)
        assert all(not validate_task_semantics(task).issues for task in tasks)


def test_family_a_through_c_generation_smoke(config):
    assert len(generate_family_a_cores(config)) == 120
    assert len(generate_family_b_cores(config)) == 120
    assert len(generate_family_c_cores(config)) == 120
