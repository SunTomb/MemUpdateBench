from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.statistics.contracts_v3 import Task13RunSourceV1
from mub.vnext.statistics.input_v3 import Task13AuthenticatedObservationV1
from mub.vnext.statistics.cases_v3 import (
    build_task13_cases_v1,
    project_task13_case_v1,
    select_task13_cases_for_run_v1,
    verify_task13_cases_v1,
)
from tests.vnext.task12_fixtures import ROOT
from tests.vnext.task13_input_fixtures import _compact_bundle, _prompted_row, _scores


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _run_config(run_id: str = "run-case-a"):
    return SimpleNamespace(
        run_id=run_id,
        adapter_info=SimpleNamespace(adapter_id="adapter-case"),
        action_parser_version="action-parser-v1",
        answer_parser_version="answer-parser-v1",
        memory_entry_extractor_version="entry-extractor-v1",
        object_value_extractor_config_hash=None,
        redaction_policy_version="redaction-v1",
        answer_model_slot="answer_model_a",
    )


def _scored(base: ScoreRecordV3, *, exact: float, stale: float) -> ScoreRecordV3:
    payload = base.model_dump(mode="json")
    payload["answer_scores"]["exact_match"] = exact
    payload["answer_scores"]["stale_copied"] = stale
    payload["protocol_scores"]["answer_parse_valid"] = True
    for path in (
        "answer_scores.exact_match",
        "answer_scores.stale_copied",
        "protocol_scores.answer_parse_valid",
    ):
        payload["supported_metric_fields"].pop(path)
    return ScoreRecordV3.model_validate(payload)


def _invalid_answer(row):
    payload = row.model_dump(mode="json")
    payload["answer_predictions"][0]["parsed_answer"] = None
    payload["answer_predictions"][0]["format_valid"] = False
    payload["answer_predictions"][0]["error_flags"] = ["invalid-answer"]
    return type(row).model_validate(payload)


@pytest.fixture(scope="module")
def case_fixture():
    tasks = _compact_bundle(ROOT).snapshot.tasks[:5]
    config = _run_config()
    source = Task13RunSourceV1(
        run_id=config.run_id,
        run_manifest_sha256=SHA_A,
        score_artifact_sha256=SHA_B,
    )
    base_scores = _scores(tasks, config)
    categories = (
        (1.0, 0.0, False),
        (0.0, 1.0, False),
        (0.0, 0.0, True),
        (0.0, 0.0, False),
        (1.0, 0.0, False),
    )
    observations = []
    for task, base_score, (exact, stale, invalid) in zip(tasks, base_scores, categories):
        row = _prompted_row(task, config).model_copy(
            update={
                "memory_snapshots": (
                    MemorySnapshotV3(
                        after_event_id=task.events[-1].event_id,
                        state_by_object={
                            task.target_objects[0].canonical_id: "case-final-value"
                        },
                        store_size=1,
                    ),
                )
            }
        )
        if invalid:
            row = _invalid_answer(row)
        observations.append(
            Task13AuthenticatedObservationV1(
                cell_id="cell-case-a",
                slot="answer_model_a",
                k=4,
                context_order="chronological",
                context_annotation="none",
                semantic_core_id=task.metadata.split_key.semantic_core_id,
                task=task,
                run=row,
                score=_scored(base_score, exact=exact, stale=stale),
                source=source,
            )
        )
    run = SimpleNamespace(
        source=source,
        observations=tuple(observations),
        run_configuration=config,
    )
    matrix = SimpleNamespace(
        runs=(run,),
        input_hashes={
            "core_tasks": SHA_C,
            "core_task_manifest": SHA_A,
            "task12_matrix_summary": SHA_B,
        },
    )
    full_runs = []
    for index in range(18):
        run_id = f"run-case-{index:02d}"
        cloned_source = Task13RunSourceV1(
            run_id=run_id,
            run_manifest_sha256=SHA_A,
            score_artifact_sha256=SHA_B,
        )
        cloned_observations = tuple(
            replace(
                observation,
                cell_id=f"cell-case-{index:02d}",
                run=observation.run.model_copy(update={"run_id": run_id}),
                score=observation.score.model_copy(update={"run_id": run_id}),
                source=cloned_source,
            )
            for observation in observations
        )
        full_runs.append(
            SimpleNamespace(
                source=cloned_source,
                observations=cloned_observations,
                run_configuration=_run_config(run_id),
            )
        )
    full_matrix = SimpleNamespace(
        runs=tuple(full_runs),
        input_hashes=matrix.input_hashes,
    )
    return SimpleNamespace(
        run=run,
        matrix=matrix,
        full_matrix=full_matrix,
        observations=tuple(observations),
    )


def test_case_selection_is_stratified_and_order_invariant(case_fixture):
    hashes = case_fixture.matrix.input_hashes
    forward = select_task13_cases_for_run_v1(case_fixture.run, input_hashes=hashes)
    shuffled_run = SimpleNamespace(
        **{
            **case_fixture.run.__dict__,
            "observations": tuple(reversed(case_fixture.run.observations)),
        }
    )
    reverse = select_task13_cases_for_run_v1(shuffled_run, input_hashes=hashes)

    assert tuple(case.category for case in forward) == (
        "correct",
        "stale_copied",
        "answer_parse_invalid",
        "other_wrong",
    )
    assert tuple(case.case_id for case in forward) == tuple(case.case_id for case in reverse)
    assert len({case.task_id for case in forward}) == 4
    expected_correct = min(
        (
            observation
            for observation in case_fixture.observations
            if observation.score.answer_scores.exact_match == 1
        ),
        key=lambda observation: (
            observation.semantic_core_id.encode("utf-8"),
            observation.task.task_id.encode("utf-8"),
        ),
    )
    assert forward[0].task_id == expected_correct.task.task_id


def test_case_metrics_are_copied_not_recomputed(case_fixture):
    case = project_task13_case_v1(
        case_fixture.observations[1], input_hashes=case_fixture.matrix.input_hashes
    )

    assert case.score.metric_layers["answer_scores"]["stale_copied"] == 1.0
    assert case.score.support == {
        path: support.model_dump(mode="json")
        for path, support in case_fixture.observations[1].score.supported_metric_fields.items()
    }
    assert case.run.model_dump(mode="json")["final_state"] == case_fixture.observations[
        1
    ].run.memory_snapshots[-1].model_dump(mode="json")["state_by_object"]
    assert case.task.model_dump(mode="json")["gold_actions"] == [
        action.model_dump(mode="json")
        for action in case_fixture.observations[1].task.actions
    ]


def test_case_verifier_rejects_changed_score_or_trace(case_fixture):
    result = build_task13_cases_v1(case_fixture.full_matrix)
    verify_task13_cases_v1(result.cases, case_fixture.full_matrix)

    first = result.cases[0]
    changed_layers = dict(first.score.metric_layers)
    changed_answer = dict(changed_layers["answer_scores"])
    changed_answer["exact_match"] = 0.125
    changed_layers["answer_scores"] = changed_answer
    changed_score = first.score.model_copy(update={"metric_layers": changed_layers})
    changed_case = first.model_copy(update={"score": changed_score})
    with pytest.raises(ValueError, match="does not equal authenticated source evidence"):
        verify_task13_cases_v1(
            (changed_case, *result.cases[1:]), case_fixture.full_matrix
        )

    first_run = case_fixture.full_matrix.runs[0]
    changed_row = first_run.observations[0].run.model_copy(
        update={"system_events": ({"changed": True},)}
    )
    changed_observation = replace(first_run.observations[0], run=changed_row)
    changed_run = SimpleNamespace(
        **{
            **first_run.__dict__,
            "observations": (changed_observation, *first_run.observations[1:]),
        }
    )
    changed_matrix = SimpleNamespace(
        runs=(changed_run, *case_fixture.full_matrix.runs[1:]),
        input_hashes=case_fixture.full_matrix.input_hashes,
    )
    with pytest.raises(ValueError, match="does not equal authenticated source evidence"):
        verify_task13_cases_v1(result.cases, changed_matrix)


def test_case_build_and_verify_have_no_run_count_bypass(case_fixture):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_task13_cases_v1(case_fixture.matrix, require_18_runs=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        verify_task13_cases_v1((), case_fixture.matrix, require_18_runs=False)


def test_private_source_text_is_redacted(case_fixture):
    observation = case_fixture.observations[0]
    private_source = observation.task.source.model_copy(
        update={"provenance": {"redistributable": False}}
    )
    private_task = observation.task.model_copy(update={"source": private_source})
    private_observation = replace(observation, task=private_task)

    case = project_task13_case_v1(
        private_observation, input_hashes=case_fixture.matrix.input_hashes
    )

    assert case.task.source["source_uri"] is None
    assert case.timeline.redacted is True
    assert all("raw_text" not in event for event in case.timeline.items)
    assert all("normalized_text" not in event for event in case.timeline.items)
