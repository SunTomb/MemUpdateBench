from __future__ import annotations

from pathlib import Path

import pytest

from mub.vnext.contracts import AnswerDisposition, ReferenceResolutionStatus, Split
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, ReferenceResolutionSelector
from mub.vnext.generation.post_core_catalogs import POST_CORE_SURFACES
from mub.vnext.generation.post_core_config import load_post_core_data_config
from mub.vnext.generation.post_core_families import (
    PostCoreSemanticCore,
    generate_post_core_family_b_cores,
    generate_post_core_family_c_cores,
    generate_post_core_family_d_cores,
)
from mub.vnext.generation.post_core_render import (
    render_post_core_tasks_v3,
    render_post_core_v3,
)
from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "post_core_data.yaml"


@pytest.fixture(scope="module")
def config():
    return load_post_core_data_config(CONFIG_PATH)


def test_each_metadata_core_expands_to_four_valid_surface_tasks(config) -> None:
    for core in (
        generate_post_core_family_b_cores(config)[0],
        generate_post_core_family_c_cores(config)[0],
        generate_post_core_family_d_cores(config)[0],
    ):
        tasks = render_post_core_tasks_v3(
            core,
            config=config,
            split=Split.TRAIN,
            code_revision="post-core-render-test",
        )

        assert len(tasks) == 4
        assert all(isinstance(task, MemUpdateTaskV3) for task in tasks)
        assert {task.schema_version for task in tasks} == {"3.0.0"}
        assert len({task.task_id for task in tasks}) == 4
        assert len({task.source.source_id for task in tasks}) == 4
        assert len({task.source.raw_hash for task in tasks}) == 4
        assert len({task.source.normalized_hash for task in tasks}) == 1
        assert len({semantic_task_hash_v3(task) for task in tasks}) == 1

        for task in tasks:
            assert not replay_task_v3(task).issues
            assert task.metadata.split_key.semantic_core_id == core.expansion_id
            assert task.metadata.split_key.trajectory_id == tasks[0].metadata.split_key.trajectory_id
            assert task.metadata.split_key.source_group_id == tasks[0].metadata.split_key.source_group_id
            assert task.metadata.split_key.version_group_id == tasks[0].metadata.split_key.version_group_id
            assert task.metadata.extra == {
                **task.metadata.extra,
                "domain": core.domain,
                "attribute": core.attribute,
                "locale": task.metadata.extra["locale"],
                "language": task.metadata.extra["language"],
                "surface_id": task.metadata.extra["surface_id"],
                "semantic_core_id": core.expansion_id,
                "trajectory_id": tasks[0].metadata.split_key.trajectory_id,
                "source_group_id": tasks[0].metadata.split_key.source_group_id,
                "version_group_id": tasks[0].metadata.split_key.version_group_id,
                "translation_catalog_version": "vnext-post-core-data-surfaces-v1",
            }


def test_renderer_uses_the_configured_surface_order_and_is_deterministic(config) -> None:
    core = generate_post_core_family_b_cores(config)[17]
    first = render_post_core_tasks_v3(
        core,
        config=config,
        split=Split.DEV,
        code_revision="post-core-render-test",
    )
    second = render_post_core_tasks_v3(
        core,
        config=config,
        split=Split.DEV,
        code_revision="post-core-render-test",
    )

    assert first == second
    assert tuple(task.metadata.extra["surface_key"] for task in first) == tuple(
        surface.surface_key for surface in POST_CORE_SURFACES
    )
    assert [task.events[0].raw_text for task in first] == [
        task.events[0].raw_text for task in second
    ]
    assert len({task.events[0].raw_text for task in first}) == 4


def test_family_b_preserves_interleaved_mutation_semantics(config) -> None:
    core = next(
        item
        for item in generate_post_core_family_b_cores(config)
        if item.family_axes["active_object_count"] == 4
        and item.family_axes["interleaving_pattern"] == "adversarial_adjacent"
    )
    tasks = render_post_core_tasks_v3(
        core,
        config=config,
        split=Split.TEST,
        code_revision="post-core-render-test",
    )

    for task in tasks:
        assert task.task_family == "interleaved_multi_slot_update"
        assert len(task.target_objects) == 4
        assert any(action.operation.value == "UPDATE" for action in task.actions)
        assert task.queries[0].target_object_keys[0] == task.target_objects[0]
        assert task.gold_evidence[0].answer == task.version_history[0].entries[-1].value
        assert not replay_task_v3(task).issues


def test_family_c_exposes_typed_unique_ambiguous_and_no_match_outcomes(config) -> None:
    cores = generate_post_core_family_c_cores(config)
    selected = {
        "unique": next(
            core for core in cores if core.family_axes["entity_condition"] == "distinct"
            and core.family_axes["attribute_condition"] == "exact"
        ),
        "ambiguous": next(
            core for core in cores if core.family_axes["entity_condition"] == "same_name"
            and core.family_axes["attribute_condition"] == "exact"
        ),
        "no_match": next(
            core for core in cores if core.family_axes["attribute_condition"] == "near_name"
        ),
    }

    for expected_status, core in selected.items():
        task = render_post_core_v3(
            core,
            config=config,
            split=Split.TRAIN,
            surface_variant=0,
            code_revision="post-core-render-test",
        )
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        replay = replay_task_v3(task)
        evaluation = evaluate_evidence_v3(
            evidence,
            replay,
            evidence.stale_alternative,
            query,
            task.events,
        )

        assert isinstance(query.selector, ReferenceResolutionSelector)
        assert not replay.issues
        assert not evaluation.issues
        assert evidence.resolution_status.value == {
            "unique": "unique",
            "ambiguous": "ambiguous",
            "no_match": "no_match",
        }[expected_status]
        if expected_status == "unique":
            assert evidence.disposition is AnswerDisposition.ANSWERED
            assert evidence.resolution_status is ReferenceResolutionStatus.UNIQUE
            assert evidence.answer is not None
        else:
            assert evidence.disposition is AnswerDisposition.ABSTAINED
            assert evidence.answer is None


def test_family_d_noop_traps_have_no_mutation_and_replay_clean(config) -> None:
    core = next(
        item
        for item in generate_post_core_family_d_cores(config)
        if item.family_axes["trap_type"] == "transient"
    )
    tasks = render_post_core_tasks_v3(
        core,
        config=config,
        split=Split.TRAIN,
        code_revision="post-core-render-test",
    )

    for task in tasks:
        noop_events = [
            (event, action)
            for event, action in zip(task.events, task.actions)
            if action.operation.value == "NOOP"
        ]
        assert noop_events
        assert all(not action.target_object_keys and action.value is None for _, action in noop_events)
        assert all(event.metadata["semantic_effect"] == "noop" for event, _ in noop_events)
        replay = replay_task_v3(task)
        assert not replay.issues
        assert replay.current_state[task.target_objects[0].canonical_id].value == task.gold_evidence[0].answer
