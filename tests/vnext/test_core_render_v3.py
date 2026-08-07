from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts import AnswerDisposition, MemoryObjectKey, ReferenceResolutionStatus, Split
from mub.vnext.contracts.v3.common import object_identity
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, ReferenceResolutionSelector
from mub.vnext.generation import render_core_v3
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.family_a import generate_core_family_a_cores
from mub.vnext.generation.family_c import generate_core_family_c_cores
from mub.vnext.generation.family_e import generate_core_family_e_cores
from mub.vnext.io import semantic_task_hash_v3
from mub.vnext.validation.replay_v3 import (
    QueryResolutionV3,
    evaluate_evidence_v3,
    replay_task_v3,
    resolve_query_v3,
)


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"


def test_render_core_v3_promotes_one_a_core_across_all_four_surfaces() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="task-636-render-v3")
    core = generate_core_family_a_cores(config)[0]

    tasks = tuple(
        render_core_v3(
            core,
            split=Split.TRAIN,
            surface_variant=surface_variant,
            context=context,
        )
        for surface_variant in range(4)
    )

    assert all(isinstance(task, MemUpdateTaskV3) for task in tasks)
    assert {task.schema_version for task in tasks} == {"3.0.0"}
    assert len({task.task_id for task in tasks}) == 4
    assert len({task.source.raw_hash for task in tasks}) == 4
    assert len({semantic_task_hash_v3(task) for task in tasks}) == 1
    assert all(not replay_task_v3(task).issues for task in tasks)
    assert all(
        tuple(object_identity(key) for key in task.target_objects)
        == tuple(object_identity(key) for key in tasks[0].target_objects)
        for task in tasks
    )


def _family_c_core_with_status(config, status: ReferenceResolutionStatus):
    return next(
        core
        for core in generate_core_family_c_cores(config)
        if core.canonical_answer.resolution_status is status
    )


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        (ReferenceResolutionStatus.UNIQUE, AnswerDisposition.ANSWERED),
        (ReferenceResolutionStatus.AMBIGUOUS, AnswerDisposition.ABSTAINED),
        (ReferenceResolutionStatus.NO_MATCH, AnswerDisposition.ABSTAINED),
    ],
)
def test_render_core_v3_promotes_family_c_reference_outcomes_across_surfaces(
    status: ReferenceResolutionStatus,
    disposition: AnswerDisposition,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="task-643-family-c-v3")
    core = _family_c_core_with_status(config, status)
    tasks = tuple(
        render_core_v3(
            core,
            split=Split.TRAIN,
            surface_variant=surface_variant,
            context=context,
        )
        for surface_variant in range(4)
    )

    assert len({semantic_task_hash_v3(task) for task in tasks}) == 1
    for task in tasks:
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        replay = replay_task_v3(task)
        resolution = resolve_query_v3(query, replay, task.events)
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
        assert not resolution.issues
        assert resolution.disposition is disposition
        assert resolution.resolution_status is status
        assert resolution.selected_candidate_ids == evidence.selected_candidate_ids
        if disposition is AnswerDisposition.ANSWERED:
            assert resolution.answer == evidence.answer == core.canonical_answer.value
            assert len(resolution.selected_versions) == 1
            assert resolution.selected_versions[0].value == evidence.answer
        else:
            assert resolution.answer is None
            assert resolution.selected_versions == ()
            assert evidence.answer is None


@pytest.mark.parametrize(
    "lifecycle_cell",
    (
        "explicit_object_or_attribute_deletion",
        "correction_versus_deletion_hard_negative",
        "logical_ttl_expiry",
        "delete_then_relearn",
    ),
)
def test_render_core_v3_family_e_evidence_replays_normatively(
    lifecycle_cell: str,
) -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="task-561-family-e-evidence")
    core = next(
        core
        for core in generate_core_family_e_cores(config, profile="full")
        if core.stratification["lifecycle_cell"] == lifecycle_cell
    )
    task = render_core_v3(
        core,
        split=Split.TEST,
        surface_variant=0,
        context=context,
    )
    replay = replay_task_v3(task)
    evidence = task.gold_evidence[0]
    evaluation = evaluate_evidence_v3(
        evidence,
        replay,
        evidence.stale_alternative,
        task.queries[0],
        task.events,
    )

    assert not replay.issues
    assert not evaluation.issues
    assert evaluation.answer == evidence.answer


def test_typed_reference_resolution_rejects_partial_selection_shapes() -> None:
    with pytest.raises(ValidationError, match="selected replay version"):
        QueryResolutionV3(
            query_id="query-0",
            answer="Paris",
            disposition="answered",
            resolution_status="unique",
            selected_candidate_ids=("candidate-0",),
        )
    with pytest.raises(ValidationError, match="cannot carry selected"):
        QueryResolutionV3(
            query_id="query-0",
            disposition="abstained",
            resolution_status="ambiguous",
            selected_object_keys=(
                MemoryObjectKey(
                    object_type="slot",
                    namespace="people",
                    entity="alex",
                    attribute="city",
                ),
            ),
        )


def test_family_c_v3_replay_rejects_forged_answer_and_contract_rejects_typed_forgery() -> None:
    config = load_core_config(CORE_CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="task-643-family-c-v3")
    core = _family_c_core_with_status(config, ReferenceResolutionStatus.UNIQUE)
    task = render_core_v3(
        core,
        split=Split.TRAIN,
        surface_variant=0,
        context=context,
    )

    wrong_answer = task.model_dump(mode="python")
    wrong_answer["gold_evidence"][0]["answer"] = "forged-current-value"
    forged_task = MemUpdateTaskV3.model_validate(wrong_answer)
    replay = replay_task_v3(forged_task)
    assert [issue.code for issue in replay.issues] == ["query_gold_answer_mismatch"]

    wrong_candidate = task.model_dump(mode="python")
    wrong_candidate["gold_evidence"][0]["selected_candidate_ids"] = [
        wrong_candidate["queries"][0]["selector"]["reference_candidates"][1]["candidate_id"]
    ]
    with pytest.raises(ValidationError, match="not linked"):
        MemUpdateTaskV3.model_validate(wrong_candidate)

    wrong_disposition = task.model_dump(mode="python")
    wrong_disposition["gold_evidence"][0].update(
        {
            "answer": None,
            "disposition": "abstained",
            "resolution_status": "ambiguous",
            "selected_candidate_ids": [],
            "abstention_reason": "forged ambiguity",
        }
    )
    with pytest.raises(ValidationError, match="AMBIGUOUS.*multiple candidates"):
        MemUpdateTaskV3.model_validate(wrong_disposition)

    wrong_graph = task.model_dump(mode="python")
    wrong_graph["queries"][0]["selector"]["surface_references"][0]["candidate_ids"] = [
        "unknown-candidate"
    ]
    with pytest.raises(ValidationError, match="unknown candidates"):
        MemUpdateTaskV3.model_validate(wrong_graph)
