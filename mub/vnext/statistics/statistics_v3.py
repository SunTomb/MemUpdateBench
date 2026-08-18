from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import json
from types import MappingProxyType
from typing import Any

from mub.vnext.contracts.common import MetricFieldSupport
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.statistics.bootstrap_v3 import (
    BootstrapIndicesV1,
    DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1,
    clustered_percentile_interval_v1,
    paired_percentile_interval_v1,
)
from mub.vnext.statistics.contracts_v3 import (
    TASK13_METRIC_PATHS,
    TASK13_SEMANTIC_CORE_COUNT,
    TASK13_TASK_COUNT,
    TASK13_TASKS_PER_CORE,
    Task13CellStatisticV1,
    Task13IntervalV1,
    Task13PairedContrastV1,
    Task13RunSourceV1,
    Task13StatisticStatus,
)


@dataclass(frozen=True, slots=True)
class Task13MetricProjectionV1:
    metric_path: str
    status: Task13StatisticStatus
    core_values: Mapping[str, Decimal] | None
    interval: Task13IntervalV1
    task_ids_by_core: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class Task13StatisticsResultV1:
    cell_statistics: tuple[Task13CellStatisticV1, ...]
    paired_contrasts: tuple[Task13PairedContrastV1, ...]


def _validate_metric_path(metric_path: str) -> None:
    if metric_path not in TASK13_METRIC_PATHS:
        raise ValueError("metric_path must belong to the frozen Task 13 metric paths")
    if metric_path.count(".") != 1:
        raise ValueError("metric_path must have one layer and one field")


def decimal_metric_v1(score: Any, metric_path: str) -> Decimal | None:
    """Extract one score field using Decimal-preserving canonical JSON parsing."""

    _validate_metric_path(metric_path)
    layer, field = metric_path.split(".", 1)
    payload = json.loads(
        canonical_json_bytes(score),
        parse_float=Decimal,
        parse_int=Decimal,
    )
    value = payload[layer][field]
    if value is None:
        return None
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("Task 13 metric values must be finite Decimal values")
    return value


def _decimal_string(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("Task 13 statistics must be finite Decimals")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _observation_identity(observation: Any) -> tuple[str, str]:
    task = observation.task
    task_id = task.task_id
    core_id = observation.semantic_core_id
    if type(task_id) is not str or type(core_id) is not str:
        raise ValueError("Task 13 observations require string task and semantic-core IDs")
    metadata_core = task.metadata.split_key.semantic_core_id
    if metadata_core != core_id:
        raise ValueError("observation semantic-core ID differs from task metadata")
    if getattr(observation.score, "task_id", task_id) != task_id:
        raise ValueError("score task ID differs from task identity")
    if getattr(observation.run, "task_id", task_id) != task_id:
        raise ValueError("run task ID differs from task identity")
    return task_id, core_id


def _typed_support(observation: Any, metric_path: str) -> MetricFieldSupport:
    support = observation.score.supported_metric_fields.get(metric_path)
    if not isinstance(support, MetricFieldSupport):
        raise ValueError("null Task 13 metric must carry typed support metadata")
    return support


def _canonical_task_groups(
    observations: Sequence[Any], matrix: BootstrapIndicesV1
) -> tuple[dict[str, tuple[Any, ...]], tuple[tuple[str, tuple[str, ...]], ...]]:
    if len(observations) != TASK13_TASK_COUNT:
        raise ValueError("Task 13 cells require exactly 80 observations")
    by_task: dict[str, Any] = {}
    grouped: dict[str, list[Any]] = defaultdict(list)
    for observation in observations:
        task_id, core_id = _observation_identity(observation)
        if task_id in by_task:
            raise ValueError("Task 13 cells cannot contain duplicate task IDs")
        by_task[task_id] = observation
        grouped[core_id].append(observation)
    if set(grouped) != set(matrix.ordered_core_ids):
        raise ValueError("Task 13 observations must cover exactly the bootstrap cores")
    if any(len(items) != TASK13_TASKS_PER_CORE for items in grouped.values()):
        raise ValueError("each semantic core must contain exactly four task rows")
    ordered_groups: dict[str, tuple[Any, ...]] = {}
    identity: list[tuple[str, tuple[str, ...]]] = []
    for core_id in matrix.ordered_core_ids:
        ordered = tuple(sorted(grouped[core_id], key=lambda item: item.task.task_id.encode("utf-8")))
        ordered_groups[core_id] = ordered
        identity.append((core_id, tuple(item.task.task_id for item in ordered)))
    return ordered_groups, tuple(identity)


def _unsupported_interval(support: MetricFieldSupport) -> Task13IntervalV1:
    return Task13IntervalV1(
        status=Task13StatisticStatus.UNSUPPORTED,
        support=support,
        support_sha256=sha256_model(support),
    )


def project_metric_v1(
    run: Any,
    metric_path: str,
    matrix: BootstrapIndicesV1,
    config=None,
) -> Task13MetricProjectionV1:
    """Validate one 80-row cell and project it to 20 semantic-core means."""

    _validate_metric_path(metric_path)
    if not isinstance(matrix, BootstrapIndicesV1):
        raise TypeError("matrix must be BootstrapIndicesV1")
    resolved_config = matrix.config if config is None else config
    if resolved_config != DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1:
        raise ValueError("Task 13 statistics require the frozen bootstrap config")
    observations = tuple(run.observations)
    groups, identity = _canonical_task_groups(observations, matrix)
    values: dict[str, Decimal] = {}
    support: MetricFieldSupport | None = None
    saw_null = False
    saw_numeric = False
    for core_id in matrix.ordered_core_ids:
        task_values: list[Decimal] = []
        for observation in groups[core_id]:
            value = decimal_metric_v1(observation.score, metric_path)
            if value is None:
                saw_null = True
                candidate = _typed_support(observation, metric_path)
                if support is None:
                    support = candidate
                elif candidate != support:
                    raise ValueError("all unsupported metric rows must share one support record")
            else:
                saw_numeric = True
                task_values.append(value)
        if task_values and len(task_values) != TASK13_TASKS_PER_CORE:
            raise ValueError("metric support state is mixed within a semantic core")
        if task_values:
            values[core_id] = sum(task_values, Decimal(0)) / Decimal(TASK13_TASKS_PER_CORE)
    if saw_null and saw_numeric:
        raise ValueError("Task 13 metric support cannot be mixed across task rows")
    if saw_null:
        if support is None:
            raise ValueError("unsupported metric rows require support metadata")
        interval = _unsupported_interval(support)
        return Task13MetricProjectionV1(
            metric_path=metric_path,
            status=Task13StatisticStatus.UNSUPPORTED,
            core_values=None,
            interval=interval,
            task_ids_by_core=identity,
        )
    if len(values) != TASK13_SEMANTIC_CORE_COUNT:
        raise ValueError("supported metric must project all 20 semantic cores")
    interval = clustered_percentile_interval_v1(values, matrix, config=resolved_config)
    return Task13MetricProjectionV1(
        metric_path=metric_path,
        status=Task13StatisticStatus.NUMERIC,
        core_values=MappingProxyType(dict(values)),
        interval=interval,
        task_ids_by_core=identity,
    )


def _run_source(run: Any) -> Task13RunSourceV1:
    source = run.source
    return Task13RunSourceV1(
        run_id=source.run_id,
        run_manifest_sha256=source.run_manifest_sha256,
        score_artifact_sha256=source.score_artifact_sha256,
    )


def build_cell_statistic_v1(
    run: Any,
    metric_path: str,
    matrix: BootstrapIndicesV1,
    config=None,
) -> tuple[Task13CellStatisticV1, Task13MetricProjectionV1]:
    projection = project_metric_v1(run, metric_path, matrix, config=config)
    source = _run_source(run)
    cell = run.cell
    k = cell.retrieval.configuration.retrieval_k
    slot = run.run_configuration.answer_model_slot
    record = Task13CellStatisticV1(
        cell_id=cell.cell_id,
        answer_model_slot=slot,
        k=k,
        metric_path=metric_path,
        interval=projection.interval,
        task_count=TASK13_TASK_COUNT,
        core_count=TASK13_SEMANTIC_CORE_COUNT,
        core_ids_sha256=matrix.core_ids_sha256,
        run_id=source.run_id,
        run_manifest_sha256=source.run_manifest_sha256,
        score_artifact_sha256=source.score_artifact_sha256,
        bootstrap_config_sha256=sha256_model(matrix.config),
        bootstrap_indices_sha256=matrix.sha256,
    )
    return record, projection


def _coordinate(run: Any) -> tuple[str, int, str, str]:
    cell = run.cell
    return (
        run.run_configuration.answer_model_slot,
        cell.retrieval.configuration.retrieval_k,
        cell.context_intervention.context_order,
        cell.context_intervention.context_annotation,
    )


def task13_contrast_id_v1(
    slot: str,
    k: int,
    left_cell_id: str,
    right_cell_id: str,
    metric: str,
) -> str:
    """Return the frozen deterministic ID for one directed metric contrast."""

    metric_token = metric.replace(".", "-")
    return f"contrast-{slot}-k{k:02d}-{left_cell_id}-minus-{right_cell_id}-{metric_token}"


def compute_task13_statistics_v1(
    matrix_input: Any,
    bootstrap: BootstrapIndicesV1,
    config=None,
) -> Task13StatisticsResultV1:
    """Compute all 126 cells and 84 predeclared paired contrasts."""

    runs = tuple(matrix_input.runs)
    if len(runs) != 18:
        raise ValueError("Task 13 statistics require exactly 18 authenticated runs")
    projections: dict[tuple[int, str], Task13MetricProjectionV1] = {}
    cell_records: list[Task13CellStatisticV1] = []
    for run_index, run in enumerate(runs):
        for metric_path in TASK13_METRIC_PATHS:
            record, projection = build_cell_statistic_v1(
                run, metric_path, bootstrap, config=config
            )
            cell_records.append(record)
            projections[(run_index, metric_path)] = projection
    by_coordinate: dict[tuple[str, int, str, str], tuple[int, Any]] = {}
    for index, run in enumerate(runs):
        coordinate = _coordinate(run)
        if coordinate in by_coordinate:
            raise ValueError("Task 13 runs contain duplicate cell coordinates")
        by_coordinate[coordinate] = (index, run)
    contrasts: list[Task13PairedContrastV1] = []
    for slot in ("answer_model_a", "answer_model_b"):
        for k in (4, 8, 16):
            pairs = (
                (("reverse_chronological", "none"), ("chronological", "none")),
                (("reverse_chronological", "latest_outdated_label"), ("reverse_chronological", "none")),
            )
            for (left_condition, right_condition) in pairs:
                left_index, left_run = by_coordinate[(slot, k, *left_condition)]
                right_index, right_run = by_coordinate[(slot, k, *right_condition)]
                left_source = _run_source(left_run)
                right_source = _run_source(right_run)
                for metric_path in TASK13_METRIC_PATHS:
                    left_projection = projections[(left_index, metric_path)]
                    right_projection = projections[(right_index, metric_path)]
                    if left_projection.status != right_projection.status:
                        raise ValueError("paired contrast has mixed support states")
                    if left_projection.task_ids_by_core != right_projection.task_ids_by_core:
                        raise ValueError(
                            "paired contrast cells must share identical task IDs and core assignments"
                        )
                    if left_projection.status is Task13StatisticStatus.NUMERIC:
                        assert left_projection.core_values is not None
                        assert right_projection.core_values is not None
                        interval = paired_percentile_interval_v1(
                            left_projection.core_values,
                            right_projection.core_values,
                            bootstrap,
                            config=config,
                        )
                    else:
                        if left_projection.interval.support != right_projection.interval.support:
                            raise ValueError("unsupported paired cells have differing support records")
                        interval = left_projection.interval
                    contrasts.append(
                        Task13PairedContrastV1(
                            contrast_id=task13_contrast_id_v1(
                                slot,
                                k,
                                left_run.cell.cell_id,
                                right_run.cell.cell_id,
                                metric_path,
                            ),
                            left_cell_id=left_run.cell.cell_id,
                            right_cell_id=right_run.cell.cell_id,
                            direction="left_minus_right",
                            answer_model_slot=slot,
                            k=k,
                            metric_path=metric_path,
                            interval=interval,
                            core_count=TASK13_SEMANTIC_CORE_COUNT,
                            core_ids_sha256=bootstrap.core_ids_sha256,
                            left_source=left_source,
                            right_source=right_source,
                            bootstrap_config_sha256=sha256_model(bootstrap.config),
                            bootstrap_indices_sha256=bootstrap.sha256,
                        )
                    )
    if len(cell_records) != 126 or len(contrasts) != 84:
        raise AssertionError("Task 13 statistics cardinality is not frozen")
    return Task13StatisticsResultV1(tuple(cell_records), tuple(contrasts))


__all__ = [
    "Task13MetricProjectionV1",
    "Task13StatisticsResultV1",
    "build_cell_statistic_v1",
    "compute_task13_statistics_v1",
    "decimal_metric_v1",
    "project_metric_v1",
    "task13_contrast_id_v1",
]
