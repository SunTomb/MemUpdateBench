from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import ArtifactRef, MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS
from mub.vnext.io import sha256_model
from mub.vnext.statistics.contracts_v3 import (
    CORE_TASK13_METRIC_PATHS,
    TASK13_METRIC_PATHS,
    Task13ArtifactBindingV1,
    Task13ArtifactIndexV1,
    Task13BootstrapConfigV1,
    Task13CaseBindingV1,
    Task13CaseIndexV1,
    Task13CaseRecordV1,
    Task13CaseSelectorV1,
    Task13CellStatisticV1,
    Task13ClaimLedgerRecordV1,
    Task13DenominatorV1,
    Task13IntervalV1,
    Task13PairedContrastV1,
    Task13RunCaseCoverageV1,
    Task13RunSourceV1,
    Task13StatisticStatus,
    Task13TaskProjectionV1,
    Task13TimelineProjectionV1,
    Task13RunProjectionV1,
    Task13StatisticsReceiptV1,
    Task13ScoreProjectionV1,
    Task13RetrievalProjectionV1,
    Task13AnswerProjectionV1,
    task13_case_id_v1,
)


SHA = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


TASK13_ARTIFACT_NAMES = (
    "bootstrap_indices.bin",
    "cell_statistics.jsonl",
    "paired_contrasts.jsonl",
    "statistics_receipt.json",
    "cases.jsonl",
    "case_index.json",
    "claim_ledger.jsonl",
)


def _support() -> MetricFieldSupport:
    return MetricFieldSupport(
        reason=SupportReason.NOT_SUPPORTED,
        null_policy="all_null",
        detail="fixture",
    )


def _binding(
    identifier: str = "artifact-a",
    path: str = "cell_statistics.jsonl",
    role: str | None = None,
) -> Task13ArtifactBindingV1:
    return Task13ArtifactBindingV1(
        artifact_id=identifier,
        artifact=ArtifactRef(path=path, sha256=SHA, media_type="application/jsonl"),
        role=role,
    )


def _source(run_id: str, manifest: str = SHA, score: str = SHA_B) -> Task13RunSourceV1:
    return Task13RunSourceV1(
        run_id=run_id,
        run_manifest_sha256=manifest,
        score_artifact_sha256=score,
    )


def _cell(metric_path: str = TASK13_METRIC_PATHS[0], *, k: int = 4, task_count: int = 80) -> Task13CellStatisticV1:
    return Task13CellStatisticV1(
        cell_id="cell-a",
        answer_model_slot="qwen",
        k=k,
        context_order="chronological",
        context_annotation="none",
        metric_path=metric_path,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.4",
            upper="0.6",
        ),
        task_count=task_count,
        core_count=20,
        core_ids_sha256=SHA,
        task_identity_sha256=SHA,
        run_id="run-a",
        run_manifest_sha256=SHA,
        score_artifact_sha256=SHA_B,
        bootstrap_config_sha256=SHA,
        bootstrap_indices_sha256=SHA_B,
    )


def _contrast(metric_path: str = TASK13_METRIC_PATHS[0], *, k: int = 4) -> Task13PairedContrastV1:
    return Task13PairedContrastV1(
        contrast_id="contrast-a",
        left_cell_id="cell-left",
        right_cell_id="cell-right",
        direction="left_minus_right",
        answer_model_slot="qwen",
        k=k,
        metric_path=metric_path,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.1",
            lower="0",
            upper="0.2",
        ),
        core_count=20,
        core_ids_sha256=SHA,
        task_identity_sha256=SHA,
        left_source=_source("run-left", SHA, SHA_B),
        right_source=_source("run-right", SHA_B, SHA_C),
        bootstrap_config_sha256=SHA,
        bootstrap_indices_sha256=SHA_B,
    )


def _task_projection() -> Task13TaskProjectionV1:
    return Task13TaskProjectionV1(
        task_id="task-a",
        semantic_core_id="core-a",
        family="A",
        difficulty="hard",
        metadata={"semantic_core_id": "core-a"},
        source={
            "source_id": "source-a",
            "source_type": "synthetic",
            "source_uri": "https://example.invalid/source-a",
            "license_or_privacy": "public-test-fixture",
            "raw_hash": SHA,
            "normalized_hash": SHA_B,
            "normalization_version": "norm-v1",
            "provenance": {"redistributable": True, "source": "fixture"},
            "generator": {
                "generator_name": "fixture-generator",
                "seed": 1,
                "config_sha256": SHA,
                "code_revision": "fixture-revision",
                "compiler_version": "fixture-compiler",
            },
            "redacted": False,
        },
        target_objects=({"object_id": "object-a"},),
        queries=({"query_id": "query-a"},),
        gold_actions=({"action_id": "action-a"},),
    )


def _timeline_projection() -> Task13TimelineProjectionV1:
    return Task13TimelineProjectionV1(
        redacted=False,
        items=({"event_id": "event-a", "sequence_index": 1},),
    )


def _run_projection() -> Task13RunProjectionV1:
    return Task13RunProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        completion_status="complete",
        parsed_actions=({"action": "ADD"},),
        memory_snapshots=(
            {"after_event_id": "event-a", "state_by_object": {"object-a": "value-a"}},
        ),
        final_state={"object-a": "value-a"},
        system_events=(),
        provenance={"runtime_revision": "rev-a"},
        exceptions=(),
    )


def _score_projection() -> Task13ScoreProjectionV1:
    return Task13ScoreProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        metric_layers={"exact_match": 1.0},
        support={"supported": True},
        failure_flags=(),
        primary_failure=None,
    )


def _retrieval_projection() -> Task13RetrievalProjectionV1:
    return Task13RetrievalProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        available=True,
        items=({"rank": 1},),
    )


def _answer_projection() -> Task13AnswerProjectionV1:
    return Task13AnswerProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        available=True,
        items=({"raw": "value"},),
    )


def _case(case_id: str, category: str = "correct") -> Task13CaseRecordV1:
    return Task13CaseRecordV1(
        case_id=task13_case_id_v1("run-a", "task-a", category),
        category=category,
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        answer_model_slot="qwen",
        k=4,
        task_artifact_sha256=SHA,
        task_manifest_sha256=SHA,
        run_manifest_sha256=SHA,
        score_artifact_sha256=SHA_B,
        matrix_summary_sha256=SHA_C,
        task=_task_projection(),
        timeline=_timeline_projection(),
        run=_run_projection(),
        score=_score_projection(),
        retrieval=_retrieval_projection(),
        answer=_answer_projection(),
    )


def _case_index(case_ids: tuple[str, ...] | None = None) -> Task13CaseIndexV1:
    case_ids = case_ids or tuple(
        task13_case_id_v1(
            f"run-{index:02d}",
            f"task-{index:02d}",
            ("correct", "stale_copied", "answer_parse_invalid", "other_wrong")[index % 4],
        )
        for index in range(18)
    )
    run_ids = tuple(f"run-{index:02d}" for index in range(18))
    categories = ("correct", "stale_copied", "answer_parse_invalid", "other_wrong")
    coverage = []
    bindings = []
    for index, run_id in enumerate(run_ids):
        category = categories[index % len(categories)]
        field = f"{category}_case_id"
        row = {
            "run_id": run_id,
            "correct_case_id": None,
            "stale_copied_case_id": None,
            "answer_parse_invalid_case_id": None,
            "other_wrong_case_id": None,
        }
        row[field] = case_ids[index]
        coverage.append(row)
        bindings.append(
            Task13CaseBindingV1(
                case_id=case_ids[index],
                run_id=run_id,
                task_id=f"task-{index:02d}",
                category=category,
            )
        )
    return Task13CaseIndexV1(
        cases_artifact=ArtifactRef(path="cases.jsonl", sha256=SHA, media_type="application/jsonl"),
        record_count=len(case_ids),
        case_bindings=tuple(bindings),
        coverage=tuple(coverage),
        run_sources=tuple(_source(run_id, SHA, SHA_B) for run_id in run_ids),
        source_bindings=(_binding("run-a", "run.json"),),
    )


def _receipt() -> Task13StatisticsReceiptV1:
    return Task13StatisticsReceiptV1(
        receipt_id="receipt-a",
        task12_preparation_manifest_sha256=SHA,
        task12_plan_sha256=SHA_B,
        task12_matrix_manifest_sha256=SHA_C,
        task12_matrix_summary_sha256=SHA,
        task12_integrity_audit_sha256=SHA_B,
        statistics_config_sha256=SHA_C,
        task13_runtime_revision="rev-a",
        task13_runtime_tree_sha256=SHA,
        semantic_core_count=20,
        task_count=1440,
        run_count=18,
        cell_statistic_count=126,
        paired_contrast_count=84,
        core_ids_sha256=SHA_B,
        bootstrap_indices_sha256=SHA_C,
        cell_statistics_artifact_id="cell_statistics",
        cell_statistics_artifact=ArtifactRef(path="cell_statistics.jsonl", sha256=SHA, media_type="application/jsonl"),
        paired_contrasts_artifact_id="paired_contrasts",
        paired_contrasts_artifact=ArtifactRef(path="paired_contrasts.jsonl", sha256=SHA_B, media_type="application/jsonl"),
    )


def _artifact_index() -> Task13ArtifactIndexV1:
    return Task13ArtifactIndexV1(
        artifacts=tuple(
            _binding(name, name, name)
            for name in TASK13_ARTIFACT_NAMES
        )
    )


def test_task13_bootstrap_contract_is_exact() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs/vnext/core_task13_statistics_v1.json"
    raw = config_path.read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 1
    config = Task13BootstrapConfigV1.model_validate_json(raw)
    assert config.cluster_key == "semantic_core_id"
    assert config.expected_cluster_count == 20
    assert config.seed_hex == "9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542"
    assert config.replicates == 10_000
    assert config.draws_per_replicate == 20
    assert config.confidence_level == "0.95"
    assert config.interval_method == "clustered_percentile"
    assert config.quantile_method == "inverted_cdf"
    assert config.lower_order_statistic == 250
    assert config.upper_order_statistic == 9_750
    assert config.decimal_precision == 50
    assert config.decimal_rounding == "ROUND_HALF_EVEN"
    assert config.support_policy == "all_supported_or_all_unsupported"
    assert config.metric_paths == CORE_TASK13_METRIC_PATHS
    assert config.metric_paths == TASK13_METRIC_PATHS


def test_task13_denominator_binds_task_observations_to_balanced_core_means() -> None:
    denominator = Task13DenominatorV1(
        task_count=80,
        semantic_core_count=20,
        tasks_per_core=4,
    )
    assert denominator.task_count == 80
    assert denominator.semantic_core_count == 20
    assert denominator.tasks_per_core == 4
    with pytest.raises((ValidationError, ValueError)):
        Task13DenominatorV1(task_count=80, semantic_core_count=80, tasks_per_core=1)

@pytest.mark.parametrize(
    "metric_paths",
    [
        ("answer_scores.exact_match", "answer_scores.unknown"),
        ("answer_scores.exact_match", "answer_scores.exact_match"),
        tuple(reversed(TASK13_METRIC_PATHS)),
    ],
)
def test_task13_bootstrap_rejects_unknown_duplicate_or_reordered_metrics(metric_paths: tuple[str, ...]) -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/vnext/core_task13_statistics_v1.json").read_text()
    )
    payload["metric_paths"] = list(metric_paths)
    with pytest.raises((ValidationError, ValueError)):
        Task13BootstrapConfigV1.model_validate(payload)


def test_numeric_and_unsupported_intervals_are_mutually_exclusive() -> None:
    numeric = Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC,
        estimate="0.5",
        lower="0.4",
        upper="0.6",
    )
    assert numeric.model_dump(mode="json")["estimate"] == "0.5"

    support = _support()
    unsupported = Task13IntervalV1(
        status=Task13StatisticStatus.UNSUPPORTED,
        estimate=None,
        lower=None,
        upper=None,
        support=support,
        support_sha256=sha256_model(support),
    )
    assert unsupported.estimate is None
    assert unsupported.support is not None

    with pytest.raises((ValidationError, ValueError)):
        Task13IntervalV1(
            status=Task13StatisticStatus.UNSUPPORTED,
            estimate="0.0",
            lower=None,
            upper=None,
            support=support,
            support_sha256=sha256_model(support),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.4",
            upper="0.6",
            support=support,
            support_sha256=sha256_model(support),
        )


def test_intervals_reject_noncanonical_decimals_and_inverted_bounds() -> None:
    for value in ("01", "1.0", "0.00", "+1", "1e-2"):
        with pytest.raises((ValidationError, ValueError)):
            Task13IntervalV1(
                status=Task13StatisticStatus.NUMERIC,
                estimate=value,
                lower="0",
                upper="1",
            )
    with pytest.raises((ValidationError, ValueError)):
        Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.7",
            upper="0.6",
        )


def _claim(
    claim_id: str,
    *,
    kind: str = "direct_cell",
    direction: str = "self",
) -> Task13ClaimLedgerRecordV1:
    sources = (_source("run-a", SHA, SHA_B),)
    if kind == "paired_contrast":
        sources = (_source("run-a", SHA, SHA_B), _source("run-b", SHA_B, SHA_C))
    return Task13ClaimLedgerRecordV1(
        claim_id=claim_id,
        kind=kind,
        direction=direction,
        slot="qwen",
        cell_or_contrast="cell-a",
        metric_path=TASK13_METRIC_PATHS[0],
        slice_payload={},
        denominator=Task13DenominatorV1(
            task_count=80,
            semantic_core_count=20,
            tasks_per_core=4,
        ),
        status=Task13StatisticStatus.NUMERIC,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.4",
            upper="0.6",
        ),
        run_sources=sources,
        statistics_receipt_sha256=SHA,
        case_ids=("case-a",),
        case_index_sha256=SHA,
    )


def test_case_claim_and_artifact_ids_and_paths_are_unique() -> None:
    index = _case_index()
    assert len(index.case_bindings) == 18
    duplicate = index.model_dump(mode="python")
    duplicate["case_bindings"] = (*duplicate["case_bindings"][:-1], duplicate["case_bindings"][0])
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(duplicate)
    assert tuple(binding.artifact_id for binding in _artifact_index().artifacts) == TASK13_ARTIFACT_NAMES
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(artifacts=())
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(artifacts=tuple(_binding(name, name) for name in TASK13_ARTIFACT_NAMES[:-1]))


def test_sha256_fields_are_lowercase_exact_hashes() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactBindingV1(
            artifact_id="artifact-a",
            artifact=ArtifactRef(path="a.json", sha256="A" * 64, media_type="application/json"),
        )
    payload = _case("case-a").model_dump(mode="python")
    payload["task_artifact_sha256"] = "short"
    with pytest.raises(ValidationError):
        Task13CaseRecordV1.model_validate(payload)


def test_canonical_roundtrip_preserves_immutable_interval() -> None:
    interval = Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC,
        estimate="-0",
        lower="-0",
        upper="0",
    )
    rebuilt = Task13IntervalV1.model_validate_json(interval.model_dump_json())
    assert rebuilt == interval
    assert rebuilt.estimate == rebuilt.lower == rebuilt.upper == "0"
    with pytest.raises((TypeError, ValidationError, ValueError)):
        interval.estimate = "1"  # type: ignore[misc]


def test_case_and_claim_helpers_construct_complete_records() -> None:
    case = _case("case-a")
    claim = _claim("claim-a")
    assert case.case_id == task13_case_id_v1("run-a", "task-a", "correct")
    assert claim.claim_id == "claim-a"
    assert claim.denominator.task_count == 80
    assert claim.denominator.semantic_core_count == 20
    assert claim.denominator.tasks_per_core == 4


def test_only_frozen_task13_metrics_are_accepted_by_all_statistic_records() -> None:
    foreign_metric = next(path for path in CORE_METRIC_FIELD_PATHS if path not in TASK13_METRIC_PATHS)
    with pytest.raises((ValidationError, ValueError)):
        _cell(foreign_metric)
    with pytest.raises((ValidationError, ValueError)):
        _contrast(foreign_metric)
    claim = _claim("claim-foreign")
    payload = claim.model_dump(mode="python")
    payload["metric_path"] = foreign_metric
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)


def test_case_selector_uses_exact_per_category_coverage() -> None:
    coverage = Task13RunCaseCoverageV1(
        run_id="run-a",
        correct_case_id="case-a",
        stale_copied_case_id="case-b",
        answer_parse_invalid_case_id="case-c",
        other_wrong_case_id="case-d",
    )
    selector = Task13CaseSelectorV1(selector_id="selector-a", run_id="run-a", coverage=coverage)
    assert selector.coverage.correct_case_id == "case-a"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseSelectorV1(selector_id="selector-mismatch", run_id="run-b", coverage=coverage)
    with pytest.raises((ValidationError, ValueError)):
        Task13RunCaseCoverageV1(run_id="run-a")
    with pytest.raises((ValidationError, ValueError)):
        Task13RunCaseCoverageV1(
            run_id="run-a", correct_case_id="case-a", stale_copied_case_id="case-a"
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseSelectorV1(
            selector_id="selector-flat", run_id="run-a", selected_case_ids=("case-a",)
        )


def test_case_record_requires_complete_typed_projections() -> None:
    case = _case("case-a")
    assert isinstance(case.task, Task13TaskProjectionV1)
    assert isinstance(case.timeline, Task13TimelineProjectionV1)
    assert isinstance(case.run, Task13RunProjectionV1)
    assert isinstance(case.score, Task13ScoreProjectionV1)
    assert isinstance(case.retrieval, Task13RetrievalProjectionV1)
    assert isinstance(case.answer, Task13AnswerProjectionV1)
    payload = case.model_dump(mode="python")
    payload["task"]["bogus"] = True
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)
    for field in ("task", "timeline", "run", "score", "retrieval", "answer"):
        payload = case.model_dump(mode="python")
        payload[field] = {}
        with pytest.raises((ValidationError, ValueError)):
            Task13CaseRecordV1.model_validate(payload)
    payload = case.model_dump(mode="python")
    payload.pop("task")
    payload["task_payload"] = {}
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)
    payload = case.model_dump(mode="python")
    payload["answer"] = Task13AnswerProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        available=False,
        items=(),
    )
    assert Task13CaseRecordV1.model_validate(payload).answer.available is False


def test_case_index_requires_ordered_18_run_coverage_and_aligned_sources() -> None:
    index = _case_index()
    assert len(index.run_sources) == 18
    assert tuple(row.run_id for row in index.coverage) == tuple(
        source.run_id for source in index.run_sources
    )
    payload = index.model_dump(mode="python")
    payload["coverage"] = payload["coverage"][:-1]
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["run_sources"] = payload["run_sources"][:-1]
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["coverage"][0]["correct_case_id"] = "foreign-case"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["coverage"][1]["correct_case_id"] = payload["coverage"][0]["correct_case_id"]
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["case_bindings"] = (*payload["case_bindings"], Task13CaseBindingV1(
        case_id=task13_case_id_v1("run-00", "task-extra", "correct"), run_id="run-00", task_id="task-extra", category="correct"
    ))
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["run_sources"] = ("run-a",) * 18
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)


def test_claim_ledger_requires_canonical_direction_and_typed_source_cardinality() -> None:
    assert _claim("claim-a").direction == "self"
    with pytest.raises((ValidationError, ValueError)):
        _claim("claim-hyphen", kind="paired_contrast", direction="left-minus-right")
    with pytest.raises((ValidationError, ValueError)):
        _claim("claim-bad-direction", direction="left_minus_right")
    payload = _claim("claim-direct-extra").model_dump(mode="python")
    payload["run_sources"] = (*payload["run_sources"], _source("run-b"))
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)
    paired = _claim("claim-paired", kind="paired_contrast", direction="left_minus_right")
    paired_payload = paired.model_dump(mode="python")
    paired_payload["run_sources"] = paired_payload["run_sources"][:1]
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(paired_payload)
    paired_payload["run_sources"] = ("run-a", "run-b")
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(paired_payload)
    payload = _claim("claim-bad-denominator").model_dump(mode="python")
    payload["denominator"]["task_count"] = 79
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)


def test_exact_k_and_count_literals_reject_out_of_contract_values() -> None:
    cell = _cell()
    contrast = _contrast()
    assert cell.task_count == 80
    assert cell.core_count == 20
    assert contrast.core_count == 20
    for bad_k in (1, 3, 5, 12):
        with pytest.raises((ValidationError, ValueError)):
            _cell(k=bad_k)
        with pytest.raises((ValidationError, ValueError)):
            _contrast(k=bad_k)
    with pytest.raises((ValidationError, ValueError)):
        _cell(task_count=79)
    payload = _cell().model_dump(mode="python")
    payload["core_count"] = 19
    with pytest.raises((ValidationError, ValueError)):
        Task13CellStatisticV1.model_validate(payload)


def test_receipt_count_literals_and_artifact_id_path_collisions_are_rejected() -> None:
    receipt = _receipt()
    assert receipt.semantic_core_count == 20
    assert receipt.task_count == 1440
    for field, bad in (
        ("semantic_core_count", 19),
        ("task_count", 1439),
        ("run_count", 17),
        ("cell_statistic_count", 125),
        ("paired_contrast_count", 83),
    ):
        payload = receipt.model_dump(mode="python")
        payload[field] = bad
        with pytest.raises((ValidationError, ValueError)):
            Task13StatisticsReceiptV1.model_validate(payload)
    payload = receipt.model_dump(mode="python")
    payload["paired_contrasts_artifact_id"] = payload["cell_statistics_artifact_id"]
    with pytest.raises((ValidationError, ValueError)):
        Task13StatisticsReceiptV1.model_validate(payload)
    payload = receipt.model_dump(mode="python")
    payload["paired_contrasts_artifact"]["path"] = payload["cell_statistics_artifact"]["path"]
    with pytest.raises((ValidationError, ValueError)):
        Task13StatisticsReceiptV1.model_validate(payload)


def test_paired_sources_reject_same_run_and_duplicate_hash_pair() -> None:
    payload = _contrast().model_dump(mode="python")
    payload["right_source"]["run_id"] = payload["left_source"]["run_id"]
    with pytest.raises((ValidationError, ValueError)):
        Task13PairedContrastV1.model_validate(payload)
    payload = _contrast().model_dump(mode="python")
    payload["right_source"]["run_manifest_sha256"] = payload["left_source"]["run_manifest_sha256"]
    payload["right_source"]["score_artifact_sha256"] = payload["left_source"]["score_artifact_sha256"]
    with pytest.raises((ValidationError, ValueError)):
        Task13PairedContrastV1.model_validate(payload)


def test_artifact_index_rejects_partial_foreign_and_duplicate_paths() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(artifacts=tuple(_binding(name, name) for name in TASK13_ARTIFACT_NAMES[:-1]))
    foreign = list(_artifact_index().artifacts)
    foreign[0] = _binding("foreign.json", "foreign.json")
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(artifacts=tuple(foreign))
    collision = list(_artifact_index().artifacts)
    collision[1] = _binding(TASK13_ARTIFACT_NAMES[1], TASK13_ARTIFACT_NAMES[0])
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(artifacts=tuple(collision))


def test_projection_envelopes_reject_bogus_keys_but_allow_empty_unavailable_items() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["retrieval"] = {"available": False, "items": (), "bogus": True}
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)
    payload = _case("case-a").model_dump(mode="python")
    payload["retrieval"] = Task13RetrievalProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        available=False,
        items=(),
    )
    payload["answer"] = Task13AnswerProjectionV1(
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        category="correct",
        available=False,
        items=(),
    )
    assert Task13CaseRecordV1.model_validate(payload).retrieval.items == ()


def test_numeric_interval_requires_estimate_inside_endpoints() -> None:
    for estimate, lower, upper in (("0.9", "0.1", "0.8"), ("0.1", "0.2", "0.8")):
        with pytest.raises((ValidationError, ValueError)):
            Task13IntervalV1(status=Task13StatisticStatus.NUMERIC, estimate=estimate, lower=lower, upper=upper)


def test_case_record_rejects_projection_identity_splices_and_case_index_alternate_path() -> None:
    case = _case("case-a")
    payload = case.model_dump(mode="python")
    payload["task"]["task_id"] = "task-b"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["run"]["run_id"] = "run-b"
    payload["run"]["task_id"] = "task-a"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["score"]["run_id"] = "run-b"
    payload["score"]["task_id"] = "task-a"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["task"]["semantic_core_id"] = "core-b"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["run"]["semantic_core_id"] = "core-b"
    payload["run"]["category"] = "stale_copied"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    payload = case.model_dump(mode="python")
    payload["score"]["semantic_core_id"] = "core-b"
    payload["score"]["category"] = "stale_copied"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)

    index_payload = _case_index().model_dump(mode="python")
    index_payload["cases_artifact"]["path"] = "alternate_cases.jsonl"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(index_payload)


def test_claim_ledger_paired_sources_require_distinct_complete_records() -> None:
    claim = _claim("claim-paired", kind="paired_contrast", direction="left_minus_right")
    payload = claim.model_dump(mode="python")
    payload["run_sources"] = (payload["run_sources"][0], payload["run_sources"][0])
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)
    payload = claim.model_dump(mode="python")
    payload["run_sources"][1]["run_id"] = payload["run_sources"][0]["run_id"]
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)


def test_case_projection_identity_fields_are_required() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13TaskProjectionV1(
            task_id="task-a",
            family="A",
            difficulty="hard",
            metadata={},
            source={},
            target_objects=(),
            queries=(),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13RunProjectionV1(
            completion_status="complete",
            parsed_actions=(),
            memory_snapshots=(),
            provenance={},
            exceptions=(),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13ScoreProjectionV1(
            metric_layers={},
            support={},
            failure_flags=(),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13RetrievalProjectionV1(available=False, items=())
    with pytest.raises((ValidationError, ValueError)):
        Task13AnswerProjectionV1(available=False, items=())
    valid = _case("case-a").model_dump(mode="python")
    projection_fields = {
        "task": (Task13TaskProjectionV1, ("semantic_core_id",)),
        "run": (Task13RunProjectionV1, ("run_id", "task_id", "semantic_core_id", "category")),
        "score": (Task13ScoreProjectionV1, ("run_id", "task_id", "semantic_core_id", "category")),
        "retrieval": (Task13RetrievalProjectionV1, ("run_id", "task_id", "semantic_core_id", "category")),
        "answer": (Task13AnswerProjectionV1, ("run_id", "task_id", "semantic_core_id", "category")),
    }
    for field_name, (projection_type, fields) in projection_fields.items():
        for identity_field in fields:
            projection_payload = dict(valid[field_name])
            projection_payload.pop(identity_field)
            with pytest.raises((ValidationError, ValueError)):
                projection_type.model_validate(projection_payload)


def test_case_record_rejects_retrieval_and_answer_identity_splices() -> None:
    case = _case("case-a")
    payload = case.model_dump(mode="python")
    payload["retrieval"].update(
        run_id="run-b", task_id="task-a", semantic_core_id="core-a", category="correct"
    )
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)
    payload = case.model_dump(mode="python")
    payload["answer"].update(
        run_id="run-a", task_id="task-b", semantic_core_id="core-a", category="correct"
    )
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)


def test_case_binding_requires_task_id_and_same_run_task_ids_are_unique() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseBindingV1(case_id="case-a", run_id="run-a", category="correct")
    payload = _case_index().model_dump(mode="python")
    extra = dict(payload["case_bindings"][0])
    extra["case_id"] = "case-extra"
    extra["category"] = "stale_copied"
    payload["case_bindings"] = (*payload["case_bindings"], extra)
    payload["record_count"] += 1
    payload["coverage"][0]["stale_copied_case_id"] = "case-extra"
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)





def test_redacted_timeline_rejects_private_event_fields_recursively() -> None:
    with pytest.raises((ValidationError, ValueError), match="redacted|forbidden"):
        Task13TimelineProjectionV1(
            redacted=True,
            items=({"event_id": "event-a", "metadata": {"secret": "value"}},),
        )
    with pytest.raises((ValidationError, ValueError), match="redacted|forbidden"):
        Task13TimelineProjectionV1(
            redacted=True,
            items=({"event_id": "event-a", "nested": {"raw_text": "secret"}},),
        )
    with pytest.raises((ValidationError, ValueError), match="redacted|allowlisted|nested"):
        Task13TimelineProjectionV1(
            redacted=True,
            items=({"gold_action_ids": [[{"secret": "value"}]]},),
        )


def test_run_projection_final_state_must_match_a_snapshot() -> None:
    with pytest.raises((ValidationError, ValueError), match="final_state|snapshot"):
        Task13RunProjectionV1(
            run_id="run-a",
            task_id="task-a",
            semantic_core_id="core-a",
            category="correct",
            completion_status="complete",
            parsed_actions=(),
            memory_snapshots=({"state_by_object": {"object-a": "actual"}},),
            final_state={"object-a": "forged"},
            system_events=(),
            provenance={},
            exceptions=(),
        )
    with pytest.raises((ValidationError, ValueError), match="final_state|snapshot"):
        Task13RunProjectionV1(
            run_id="run-a",
            task_id="task-a",
            semantic_core_id="core-a",
            category="correct",
            completion_status="complete",
            parsed_actions=(),
            memory_snapshots=(),
            final_state={},
            system_events=(),
            provenance={},
            exceptions=(),
        )
    payload = _artifact_index().model_dump(mode="python")
    assert all(binding["role"] == binding["artifact_id"] for binding in payload["artifacts"])
    payload["artifacts"][0]["role"] = None
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1.model_validate(payload)
    payload = _artifact_index().model_dump(mode="python")
    payload["artifacts"][0]["role"] = "wrong-role"
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1.model_validate(payload)


def test_run_projection_rejects_bool_int_final_state_alias() -> None:
    with pytest.raises((ValidationError, ValueError), match="final_state|snapshot"):
        Task13RunProjectionV1(
            run_id="run-a",
            task_id="task-a",
            semantic_core_id="core-a",
            category="correct",
            completion_status="complete",
            parsed_actions=(),
            memory_snapshots=({"state_by_object": {"object-a": 1}},),
            final_state={"object-a": True},
            system_events=(),
            provenance={},
            exceptions=(),
        )


def test_case_record_rejects_duplicate_event_id_when_later_item_lacks_sequence() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["timeline"] = {
        "redacted": False,
        "items": (
            {"event_id": "event-a", "sequence_index": 1},
            {"event_id": "event-a"},
        ),
    }
    payload["run"]["memory_snapshots"] = (
        {"after_event_id": "event-a", "state_by_object": {"object-a": "value-a"}},
    )
    payload["run"]["final_state"] = {"object-a": "value-a"}
    with pytest.raises((ValidationError, ValueError), match="timeline|event|sequence"):
        Task13CaseRecordV1.model_validate(payload)


@pytest.mark.parametrize(
    "snapshots",
    (
        (
            {"after_event_id": "event-a", "state_by_object": {"object-a": "same"}},
            {"after_event_id": "event-b", "state_by_object": {"object-a": "same"}},
        ),
        (
            {"after_event_id": "event-b", "state_by_object": {"object-a": "same"}},
            {"after_event_id": "event-a", "state_by_object": {"object-a": "same"}},
        ),
    ),
    ids=("original-snapshot-order", "reversed-snapshot-order"),
)
def test_case_record_rejects_duplicate_sequence_indices_independent_of_snapshot_order(snapshots) -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["timeline"] = {
        "redacted": False,
        "items": (
            {"event_id": "event-a", "sequence_index": 1},
            {"event_id": "event-b", "sequence_index": 1},
        ),
    }
    payload["run"]["memory_snapshots"] = snapshots
    payload["run"]["final_state"] = snapshots[-1]["state_by_object"]
    with pytest.raises((ValidationError, ValueError), match="timeline|sequence|chronology"):
        Task13CaseRecordV1.model_validate(payload)


def test_case_record_accepts_no_snapshots_with_null_final_state() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["run"]["memory_snapshots"] = ()
    payload["run"]["final_state"] = None
    case = Task13CaseRecordV1.model_validate(payload)
    assert case.run.memory_snapshots == ()
    assert case.run.final_state is None


def test_case_record_accepts_single_unanchored_typed_equal_snapshot() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["run"]["memory_snapshots"] = (
        {"state_by_object": {"object-a": "value-a"}},
    )
    payload["run"]["final_state"] = {"object-a": "value-a"}
    case = Task13CaseRecordV1.model_validate(payload)
    assert case.run.memory_snapshots[0].get("after_event_id") is None
    assert case.run.final_state == {"object-a": "value-a"}


def test_case_record_accepts_normal_anchored_unique_chronology() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["timeline"] = {
        "redacted": False,
        "items": (
            {"event_id": "event-early", "sequence_index": 1},
            {"event_id": "event-late", "sequence_index": 2},
        ),
    }
    payload["run"]["memory_snapshots"] = (
        {
            "after_event_id": "event-early",
            "state_by_object": {"object-a": "early"},
        },
        {
            "after_event_id": "event-late",
            "state_by_object": {"object-a": "late"},
        },
    )
    payload["run"]["final_state"] = {"object-a": "late"}
    case = Task13CaseRecordV1.model_validate(payload)
    assert case.run.final_state == {"object-a": "late"}


def test_case_record_rejects_stale_early_final_state() -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["timeline"] = {
        "redacted": False,
        "items": (
            {"event_id": "event-early", "sequence_index": 1},
            {"event_id": "event-late", "sequence_index": 2},
        ),
    }
    payload["run"]["memory_snapshots"] = (
        {
            "after_event_id": "event-early",
            "state_by_object": {"object-a": "early"},
        },
        {
            "after_event_id": "event-late",
            "state_by_object": {"object-a": "late"},
        },
    )
    payload["run"]["final_state"] = {"object-a": "early"}
    with pytest.raises((ValidationError, ValueError), match="chronology|final_state|snapshot"):
        Task13CaseRecordV1.model_validate(payload)


@pytest.mark.parametrize(
    "snapshots",
    (
        (
            {"state_by_object": {"object-a": "unanchored"}},
            {
                "after_event_id": "event-late",
                "state_by_object": {"object-a": "late"},
            },
        ),
        (
            {
                "after_event_id": "event-late",
                "state_by_object": {"object-a": "first"},
            },
            {
                "after_event_id": "event-late",
                "state_by_object": {"object-a": "second"},
            },
        ),
        (
            {
                "after_event_id": "event-unknown",
                "state_by_object": {"object-a": "unknown"},
            },
        ),
    ),
    ids=("mixed-unanchored", "duplicate-anchor", "unknown-anchor"),
)
def test_case_record_rejects_invalid_snapshot_chronology(snapshots) -> None:
    payload = _case("case-a").model_dump(mode="python")
    payload["timeline"] = {
        "redacted": False,
        "items": (
            {"event_id": "event-early", "sequence_index": 1},
            {"event_id": "event-late", "sequence_index": 2},
        ),
    }
    payload["run"]["memory_snapshots"] = snapshots
    payload["run"]["final_state"] = snapshots[-1]["state_by_object"]
    with pytest.raises((ValidationError, ValueError), match="chronology|anchor|snapshot"):
        Task13CaseRecordV1.model_validate(payload)


def test_statistics_receipt_requires_role_specific_artifact_ids_and_paths() -> None:
    receipt = _receipt()
    assert receipt.cell_statistics_artifact_id == "cell_statistics"
    assert receipt.cell_statistics_artifact.path == "cell_statistics.jsonl"
    assert receipt.paired_contrasts_artifact_id == "paired_contrasts"
    assert receipt.paired_contrasts_artifact.path == "paired_contrasts.jsonl"
    for field in ("cell_statistics_artifact_id", "paired_contrasts_artifact_id"):
        payload = receipt.model_dump(mode="python")
        payload[field] = "foo"
        with pytest.raises((ValidationError, ValueError)):
            Task13StatisticsReceiptV1.model_validate(payload)
    for field in ("cell_statistics_artifact", "paired_contrasts_artifact"):
        payload = receipt.model_dump(mode="python")
        payload[field]["path"] = "bar.jsonl"
        with pytest.raises((ValidationError, ValueError)):
            Task13StatisticsReceiptV1.model_validate(payload)