from __future__ import annotations

import hashlib
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.common import MetricFieldSupport
from mub.vnext.contracts.enums import CompletionStatus, Difficulty, SupportReason
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.io import sha256_model
from mub.vnext.statistics.bootstrap_v3 import build_bootstrap_indices_v1
from mub.vnext.statistics.contracts_v3 import (
    TASK13_METRIC_PATHS,
    Task13StatisticStatus,
)
from mub.vnext.statistics.statistics_v3 import (
    compute_task13_statistics_v1,
    decimal_metric_v1,
    project_metric_v1,
)


CORE_IDS = tuple(f"core-{i:02d}" for i in range(20))
TASKS = tuple(
    (core_id, f"task-{core_index:02d}-{task_index}")
    for core_index, core_id in enumerate(CORE_IDS)
    for task_index in range(4)
)
BOOTSTRAP = build_bootstrap_indices_v1(CORE_IDS)


def _score(task_id: str, run_id: str, value: float | None, support=None) -> ScoreRecordV3:
    supports = {
        path: support or MetricFieldSupport(
            reason=SupportReason.NOT_SUPPORTED,
            null_policy="emit_null",
            detail="fixture",
        )
        for path in TASK13_METRIC_PATHS
    }
    if value is None:
        from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS

        supports = {
            path: support or MetricFieldSupport(
                reason=SupportReason.NOT_SUPPORTED,
                null_policy="emit_null",
                detail="fixture",
            )
            for path in CORE_METRIC_FIELD_PATHS
        }
    else:
        from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS

        supports = {
            path: MetricFieldSupport(
                reason=SupportReason.NOT_SUPPORTED,
                null_policy="emit_null",
                detail="fixture",
            )
            for path in CORE_METRIC_FIELD_PATHS
        }
        supports.pop("answer_scores.exact_match")
    from mub.vnext.contracts.score import AnswerScores

    answer_scores = AnswerScores(exact_match=value) if value is not None else AnswerScores()
    score = ScoreRecordV3.empty(
        task_id=task_id,
        run_id=run_id,
        adapter_id="fixture",
        task_family="family",
        difficulty=Difficulty.EASY,
        completion_status=CompletionStatus.COMPLETED,
        supported_metric_fields=supports,
        answer_scores=answer_scores,
    )
    return score


def _run(values, *, run_id="run-a", order="chronological", annotation="none", k=4, slot="answer_model_a"):
    observations = []
    for core_id, task_id in values:
        task = SimpleNamespace(
            task_id=task_id,
            metadata=SimpleNamespace(split_key=SimpleNamespace(semantic_core_id=core_id)),
        )
        observations.append(
            SimpleNamespace(
                cell_id=f"cell-{order}-{annotation}-k{k}",
                slot=slot,
                k=k,
                context_order=order,
                context_annotation=annotation,
                semantic_core_id=core_id,
                task=task,
                run=SimpleNamespace(task_id=task_id, run_id=run_id),
                score=values[(core_id, task_id)] if isinstance(values, dict) else None,
                source=SimpleNamespace(
                    run_id=run_id,
                    run_manifest_sha256="a" * 64,
                    score_artifact_sha256="b" * 64,
                ),
            )
        )
    return SimpleNamespace(
        cell=SimpleNamespace(
            cell_id=f"cell-{order}-{annotation}-k{k}",
            context_intervention=SimpleNamespace(
                context_order=order,
                context_annotation=annotation,
            ),
            retrieval=SimpleNamespace(configuration=SimpleNamespace(retrieval_k=k)),
        ),
        run_configuration=SimpleNamespace(
            answer_model_slot=slot,
            run_id=run_id,
        ),
        source=SimpleNamespace(
            run_id=run_id,
            run_manifest_sha256="a" * 64,
            score_artifact_sha256="b" * 64,
        ),
        observations=tuple(observations),
    )


def _copy_namespace(value, **changes):
    return SimpleNamespace(**{**vars(value), **changes})


def _make_run(
    run_id, order="chronological", annotation="none", k=4, slot="answer_model_a", base=0
):
    scores = {}
    for core_index, core_id in enumerate(CORE_IDS):
        for task_index in range(4):
            task_id = f"task-{core_index:02d}-{task_index}"
            scores[(core_id, task_id)] = _score(task_id, run_id, float(base + task_index) / 10)
    observations = []
    for core_id, task_id in TASKS:
        task = SimpleNamespace(
            task_id=task_id,
            metadata=SimpleNamespace(split_key=SimpleNamespace(semantic_core_id=core_id)),
        )
        observations.append(
            SimpleNamespace(
                cell_id=f"raw-add-{('chronological-none' if order == 'chronological' else 'reverse-none' if annotation == 'none' else 'reverse-version-labeled')}-k{k:02d}",
                slot=slot,
                k=k,
                context_order=order,
                context_annotation=annotation,
                semantic_core_id=core_id,
                task=task,
                run=SimpleNamespace(task_id=task_id, run_id=run_id),
                score=scores[(core_id, task_id)],
                source=SimpleNamespace(
                    run_id=run_id,
                    run_manifest_sha256=hashlib.sha256((run_id + "-run").encode()).hexdigest(),
                    score_artifact_sha256=hashlib.sha256((run_id + "-score").encode()).hexdigest(),
                ),
            )
        )
    return SimpleNamespace(
        cell=SimpleNamespace(
            cell_id=observations[0].cell_id,
            context_intervention=SimpleNamespace(
                context_order=order, context_annotation=annotation
            ),
            retrieval=SimpleNamespace(configuration=SimpleNamespace(retrieval_k=k)),
        ),
        run_configuration=SimpleNamespace(answer_model_slot=slot, run_id=run_id),
        source=observations[0].source,
        observations=tuple(observations),
    )


def test_decimal_metric_reads_canonical_score_as_decimal():
    score = _score("task", "run", 0.3)
    value = decimal_metric_v1(score, "answer_scores.exact_match")
    assert value == Decimal("0.3")
    assert type(value) is Decimal


def test_projection_aggregates_four_tasks_per_core_and_is_order_invariant():
    run = _make_run("run-a")
    projection = project_metric_v1(run, "answer_scores.exact_match", BOOTSTRAP)
    shuffled = SimpleNamespace(**vars(run))
    shuffled.observations = tuple(reversed(run.observations))
    shuffled_projection = project_metric_v1(
        shuffled, "answer_scores.exact_match", BOOTSTRAP
    )
    assert projection.core_values == shuffled_projection.core_values
    assert projection.core_values["core-00"] == Decimal("0.15")
    assert projection.status is Task13StatisticStatus.NUMERIC


def test_all_unsupported_emits_typed_null():
    run = _make_run("run-null")
    observations = tuple(
        _copy_namespace(
            observation,
            score=_score(observation.task.task_id, "run-null", None),
        )
        for observation in run.observations
    )
    loaded = _copy_namespace(run, observations=observations)
    projection = project_metric_v1(
        loaded,
        "answer_scores.exact_match",
        BOOTSTRAP,
    )
    assert projection.status is Task13StatisticStatus.UNSUPPORTED
    assert projection.interval.estimate is None
    assert projection.interval.support is not None


def test_mixed_support_and_wrong_core_multiplicity_fail():
    run = _make_run("run-b")
    observations = list(run.observations)
    observations[0] = _copy_namespace(
        observations[0],
        score=_score(observations[0].task.task_id, "run-b", None),
    )
    with pytest.raises(ValueError, match="mixed|support"):
        project_metric_v1(
            _copy_namespace(run, observations=tuple(observations)),
            "answer_scores.exact_match",
            BOOTSTRAP,
        )
    observations = list(run.observations)
    observations[-1] = _copy_namespace(observations[-1], semantic_core_id="core-00")
    loaded = _copy_namespace(run, observations=tuple(observations))
    with pytest.raises(ValueError, match="core|task"):
        project_metric_v1(
            loaded,
            "answer_scores.exact_match",
            BOOTSTRAP,
        )


def test_compute_cells_and_predeclared_contrasts_have_frozen_order():
    runs = []
    for slot in ("answer_model_a", "answer_model_b"):
        for k in (4, 8, 16):
            runs.extend(
                [
                    _make_run(f"c-{slot}-{k}", "chronological", "none", k, slot, 0),
                    _make_run(f"r-{slot}-{k}", "reverse_chronological", "none", k, slot, 1),
                    _make_run(f"l-{slot}-{k}", "reverse_chronological", "latest_outdated_label", k, slot, 2),
                ]
            )
    matrix = SimpleNamespace(runs=tuple(runs), canonical_core_ids=CORE_IDS)
    result = compute_task13_statistics_v1(matrix, BOOTSTRAP)
    assert len(result.cell_statistics) == 18 * len(TASK13_METRIC_PATHS)
    assert len(result.paired_contrasts) == 12 * len(TASK13_METRIC_PATHS)
    first = result.paired_contrasts[0]
    assert first.left_cell_id.startswith("raw-add-reverse-none")
    assert first.right_cell_id.startswith("raw-add-chronological-none")
    assert first.direction == "left_minus_right"


def test_paired_contrast_rejects_shape_preserving_task_substitution():
    runs = []
    for slot in ("answer_model_a", "answer_model_b"):
        for k in (4, 8, 16):
            runs.extend(
                [
                    _make_run(f"c-{slot}-{k}", "chronological", "none", k, slot, 0),
                    _make_run(f"r-{slot}-{k}", "reverse_chronological", "none", k, slot, 1),
                    _make_run(f"l-{slot}-{k}", "reverse_chronological", "latest_outdated_label", k, slot, 2),
                ]
            )
    target = runs[1]
    observations = list(target.observations)
    original = observations[0]
    substituted_id = "task-00-substituted"
    substituted_task = _copy_namespace(original.task, task_id=substituted_id)
    observations[0] = _copy_namespace(
        original,
        task=substituted_task,
        run=SimpleNamespace(task_id=substituted_id, run_id=target.run_configuration.run_id),
        score=_score(substituted_id, target.run_configuration.run_id, 0.1),
    )
    runs[1] = _copy_namespace(target, observations=tuple(observations))
    with pytest.raises(ValueError, match="identical task IDs"):
        compute_task13_statistics_v1(
            SimpleNamespace(runs=tuple(runs), canonical_core_ids=CORE_IDS),
            BOOTSTRAP,
        )


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        decimal_metric_v1(_score("task", "run", 1.0), "answer_scores.not_a_metric")
