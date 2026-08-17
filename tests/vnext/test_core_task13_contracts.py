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
    Task13CaseIndexV1,
    Task13CaseRecordV1,
    Task13CaseSelectorV1,
    Task13CellStatisticV1,
    Task13ClaimLedgerRecordV1,
    Task13IntervalV1,
    Task13PairedContrastV1,
    Task13StatisticStatus,
)


SHA = "a" * 64


def _support() -> MetricFieldSupport:
    return MetricFieldSupport(
        reason=SupportReason.NOT_SUPPORTED,
        null_policy="all_null",
        detail="fixture",
    )


def _binding(identifier: str = "artifact-a", path: str = "cell_statistics.jsonl") -> Task13ArtifactBindingV1:
    return Task13ArtifactBindingV1(
        artifact_id=identifier,
        artifact=ArtifactRef(path=path, sha256=SHA, media_type="application/jsonl"),
    )


def _cell(metric_path: str = TASK13_METRIC_PATHS[0]) -> Task13CellStatisticV1:
    return Task13CellStatisticV1(
        cell_id="cell-a",
        answer_model_slot="qwen",
        k=4,
        metric_path=metric_path,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.4",
            upper="0.6",
        ),
        task_count=80,
        core_count=80,
        core_ids_sha256=SHA,
        run_id="run-a",
        run_manifest_sha256=SHA,
        score_artifact_sha256=SHA,
        bootstrap_config_sha256=SHA,
        bootstrap_indices_sha256=SHA,
    )


def _contrast(metric_path: str = TASK13_METRIC_PATHS[0]) -> Task13PairedContrastV1:
    return Task13PairedContrastV1(
        contrast_id="contrast-a",
        left_cell_id="cell-left",
        right_cell_id="cell-right",
        direction="left_minus_right",
        answer_model_slot="qwen",
        k=4,
        metric_path=metric_path,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.1",
            lower="0",
            upper="0.2",
        ),
        core_count=80,
        core_ids_sha256=SHA,
        left_run_id="run-left",
        left_run_manifest_sha256=SHA,
        left_score_artifact_sha256=SHA,
        right_run_id="run-right",
        right_run_manifest_sha256=SHA,
        right_score_artifact_sha256=SHA,
        bootstrap_config_sha256=SHA,
        bootstrap_indices_sha256=SHA,
    )


def _case(case_id: str, category: str = "correct") -> Task13CaseRecordV1:
    return Task13CaseRecordV1(
        case_id=case_id,
        category=category,
        run_id="run-a",
        task_id="task-a",
        semantic_core_id="core-a",
        answer_model_slot="qwen",
        k=4,
        task_artifact_sha256=SHA,
        task_manifest_sha256=SHA,
        run_manifest_sha256=SHA,
        score_artifact_sha256=SHA,
        matrix_summary_sha256=SHA,
        task_payload={"task": "payload"},
        timeline_payload={"timeline": "payload"},
        run_payload={"run": "payload"},
        score_payload={"score": "payload"},
        retrieval_payload={"retrieval": "payload"},
        answer_payload={"answer": "payload"},
    )


def _case_index(case_ids: tuple[str, ...] = ("case-a", "case-b", "case-c", "case-d")) -> Task13CaseIndexV1:
    run_ids = tuple(f"run-{index:02d}" for index in range(18))
    return Task13CaseIndexV1(
        case_ids=case_ids,
        cases_artifact=ArtifactRef(path="cases.jsonl", sha256=SHA, media_type="application/jsonl"),
        record_count=len(case_ids),
        run_ids=run_ids,
        run_manifest_hashes=(SHA,) * 18,
        score_artifact_hashes=(SHA,) * 18,
        category_coverage={
            "correct": ["case-a"],
            "stale_copied": ["case-b"],
            "answer_parse_invalid": ["case-c"],
            "other_wrong": ["case-d"],
        },
        source_bindings=(_binding("run-a", "run.json"),),
    )


def _claim(claim_id: str, *, kind: str = "direct_cell", direction: str = "self") -> Task13ClaimLedgerRecordV1:
    return Task13ClaimLedgerRecordV1(
        claim_id=claim_id,
        kind=kind,
        direction=direction,
        slot="qwen",
        cell_or_contrast="cell-a",
        metric_path=TASK13_METRIC_PATHS[0],
        slice_payload={},
        denominator=80,
        status=Task13StatisticStatus.NUMERIC,
        interval=Task13IntervalV1(
            status=Task13StatisticStatus.NUMERIC,
            estimate="0.5",
            lower="0.4",
            upper="0.6",
        ),
        run_ids=("run-a",),
        run_manifest_sha256s=(SHA,),
        score_artifact_sha256s=(SHA,),
        statistics_receipt_sha256=SHA,
        case_ids=("case-a",),
        case_index_sha256=SHA,
    )


def test_task13_bootstrap_contract_is_exact() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs/vnext/core_task13_statistics_v1.json"
    raw = config_path.read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 1
    config = Task13BootstrapConfigV1.model_validate_json(raw)
    assert config.cluster_key == "semantic_core_id"
    assert config.expected_cluster_count == 80
    assert config.seed_hex == "9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542"
    assert config.replicates == 10_000
    assert config.draws_per_replicate == 80
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


def test_case_claim_and_artifact_ids_and_paths_are_unique() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1(
            case_ids=("case-a", "case-a"),
            cases_artifact=ArtifactRef(path="cases.jsonl", sha256=SHA, media_type="application/jsonl"),
            record_count=2,
            run_ids=tuple(f"run-{index:02d}" for index in range(18)),
            run_manifest_hashes=(SHA,) * 18,
            score_artifact_hashes=(SHA,) * 18,
            category_coverage={
                "correct": ["case-a"],
                "stale_copied": [],
                "answer_parse_invalid": [],
                "other_wrong": [],
            },
            source_bindings=(_binding("run-a", "run.json"),),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(
            artifacts=(_binding("artifact-a", "a.json"), _binding("artifact-b", "a.json")),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(
            artifacts=(_binding("artifact-a", "a.json"), _binding("artifact-a", "b.json")),
        )

    first = _claim("claim-a")
    second = _claim("claim-a")
    assert first == second
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactIndexV1(
            artifacts=(_binding("artifact-a", "a.json"), _binding("artifact-a", "b.json")),
        )


def test_sha256_fields_are_lowercase_exact_hashes() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13ArtifactBindingV1(
            artifact_id="artifact-a",
            artifact=ArtifactRef(path="a.json", sha256="A" * 64, media_type="application/json"),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1(
            case_id="case-a",
            category="correct",
            run_id="run-a",
            task_id="task-a",
            semantic_core_id="core-a",
            answer_model_slot="qwen",
            k=4,
            task_artifact_sha256="short",
            task_manifest_sha256=SHA,
            run_manifest_sha256=SHA,
            score_artifact_sha256=SHA,
            matrix_summary_sha256=SHA,
            task_payload={},
            timeline_payload={},
            run_payload={},
            score_payload={},
            retrieval_payload={},
            answer_payload={},
        )


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
    assert case.case_id == "case-a"
    assert claim.claim_id == "claim-a"


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


def test_case_selector_requires_exact_category_universe_and_order() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseSelectorV1(
            selector_id="selector-a",
            run_id="run-a",
            categories=("correct",),
        )
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseSelectorV1(
            selector_id="selector-a",
            run_id="run-a",
            categories=("stale_copied", "correct", "answer_parse_invalid", "other_wrong"),
        )


def test_case_record_requires_complete_projection_and_source_hashes() -> None:
    case = _case("case-a")
    assert all(
        getattr(case, field) is not None
        for field in (
            "task_payload",
            "timeline_payload",
            "run_payload",
            "score_payload",
            "retrieval_payload",
            "answer_payload",
            "task_manifest_sha256",
            "matrix_summary_sha256",
        )
    )
    payload = case.model_dump(mode="python")
    payload["answer"] = None
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)
    payload = case.model_dump(mode="python")
    payload.pop("task_manifest_sha256")
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseRecordV1.model_validate(payload)


def test_case_index_requires_all_18_runs_and_aligned_hashes_and_categories() -> None:
    index = _case_index()
    assert len(index.run_ids) == 18
    assert set(index.category_coverage) == {
        "correct",
        "stale_copied",
        "answer_parse_invalid",
        "other_wrong",
    }
    payload = index.model_dump(mode="python")
    payload["run_ids"] = payload["run_ids"][:-1]
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["run_manifest_hashes"] = payload["run_manifest_hashes"][:-1]
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)
    payload = index.model_dump(mode="python")
    payload["category_coverage"] = {"correct": ["case-a"]}
    with pytest.raises((ValidationError, ValueError)):
        Task13CaseIndexV1.model_validate(payload)


def test_claim_ledger_requires_direction_nonempty_sources_and_case_index() -> None:
    assert _claim("claim-a").direction == "self"
    assert _claim("claim-contrast", kind="paired_contrast", direction="left-minus-right").direction == "left-minus-right"
    with pytest.raises((ValidationError, ValueError)):
        _claim("claim-bad-direction", direction="left_minus_right")
    payload = _claim("claim-empty").model_dump(mode="python")
    payload["run_ids"] = ()
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)
    payload = _claim("claim-no-case-index").model_dump(mode="python")
    payload["case_index_sha256"] = None
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)
    payload = _claim("claim-no-cases").model_dump(mode="python")
    payload["case_ids"] = ()
    with pytest.raises((ValidationError, ValueError)):
        Task13ClaimLedgerRecordV1.model_validate(payload)
