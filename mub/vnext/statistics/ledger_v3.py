from __future__ import annotations

"""Pure receipt, case-index, and claim-ledger builders for Core Task 13.

This module deliberately accepts already authenticated contracts and artifact
references.  It does not open files, recalculate intervals, or publish output.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from pydantic import BaseModel

from mub.vnext.contracts.common import ArtifactRef, ImmutableContractModel
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.statistics.statistics_v3 import task13_contrast_id_v1
from mub.vnext.statistics.contracts_v3 import (
    TASK13_CELL_STATISTICS_ARTIFACT_ID,
    TASK13_CELL_STATISTICS_ARTIFACT_PATH,
    TASK13_CONTEXT_CONDITIONS,
    TASK13_CONTRAST_PAIRS,
    TASK13_METRIC_PATHS,
    TASK13_PAIRED_CONTRASTS_ARTIFACT_ID,
    TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH,
    TASK13_SEMANTIC_CORE_COUNT,
    Task13ArtifactBindingV1,
    Task13CaseBindingV1,
    Task13CaseIndexV1,
    Task13CaseRecordV1,
    Task13ClaimLedgerRecordV1,
    Task13DenominatorV1,
    Task13PairedContrastV1,
    Task13RunCaseCoverageV1,
    Task13RunSourceV1,
    Task13CellStatisticV1,
    Task13StatisticsReceiptV1,
)


_EXPECTED_CELL_COUNT = 126
_EXPECTED_CONTRAST_COUNT = 84
_EXPECTED_RUN_COUNT = 18
_EXPECTED_METRIC_COUNT = len(TASK13_METRIC_PATHS)
_EXPECTED_SLOTS = ("answer_model_a", "answer_model_b")
_EXPECTED_K = (4, 8, 16)
_CASE_CATEGORY_ORDER = {
    "correct": 0,
    "stale_copied": 1,
    "answer_parse_invalid": 2,
    "other_wrong": 3,
}


@dataclass(frozen=True, slots=True)
class Task13LedgerResultV1:
    """The in-memory Task 13 receipt, case index, and generated claims."""

    receipt: Task13StatisticsReceiptV1
    case_index: Task13CaseIndexV1
    claims: tuple[Task13ClaimLedgerRecordV1, ...]

    @property
    def statistics_receipt(self) -> Task13StatisticsReceiptV1:
        return self.receipt

    @property
    def claim_ledger(self) -> tuple[Task13ClaimLedgerRecordV1, ...]:
        return self.claims


class _ClaimIdentityPayloadV1(ImmutableContractModel):
    """Canonical JSON payload used for stable claim IDs."""

    kind: str
    slot: str
    cell_or_contrast: str
    metric_path: str
    slice: dict[str, Any]


# A BaseModel assertion here keeps accidental changes to the ID payload from
# becoming an untyped dictionary hash.
assert issubclass(_ClaimIdentityPayloadV1, BaseModel)


def _coerce_statistics(
    cell_statistics: Sequence[Task13CellStatisticV1] | Any,
    paired_contrasts: Sequence[Task13PairedContrastV1] | None,
) -> tuple[tuple[Task13CellStatisticV1, ...], tuple[Task13PairedContrastV1, ...]]:
    if paired_contrasts is None and hasattr(cell_statistics, "cell_statistics"):
        result = cell_statistics
        paired_contrasts = result.paired_contrasts
        cell_statistics = result.cell_statistics
    cells = tuple(cell_statistics)
    contrasts = tuple(paired_contrasts or ())
    return cells, contrasts


def _metric_index(metric_path: str) -> int:
    try:
        return TASK13_METRIC_PATHS.index(metric_path)
    except ValueError as exc:
        raise ValueError(f"foreign Task 13 metric path: {metric_path!r}") from exc


def canonical_jsonl_bytes_v1(rows: Sequence[BaseModel]) -> bytes:
    """Serialize ordered records as the authenticated canonical JSONL bytes."""

    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _canonical_jsonl_sha256_v1(rows: Sequence[BaseModel]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes_v1(rows)).hexdigest()


def _validate_artifact(artifact: ArtifactRef, *, path: str, label: str) -> ArtifactRef:
    if not isinstance(artifact, ArtifactRef):
        raise TypeError(f"{label} must be an ArtifactRef")
    if artifact.path != path:
        raise ValueError(f"{label}.path must be {path!r}")
    return artifact


def _validate_cells(
    cell_statistics: Sequence[Task13CellStatisticV1],
) -> tuple[Task13CellStatisticV1, ...]:
    cells = tuple(cell_statistics)
    if len(cells) != _EXPECTED_CELL_COUNT:
        raise ValueError("Task 13 requires exactly 126 cell statistics")
    if any(not isinstance(record, Task13CellStatisticV1) for record in cells):
        raise TypeError("cell statistics must be Task13CellStatisticV1 records")

    coordinates = [(record.answer_model_slot, record.k, record.cell_id) for record in cells]
    if len(set(coordinates)) != 18:
        raise ValueError("cell statistics must have globally unique cell coordinates")
    condition_keys = [
        (record.answer_model_slot, record.k, record.context_order, record.context_annotation)
        for record in cells
    ]
    if len(set(condition_keys)) != 18:
        raise ValueError("cell statistics contain duplicate typed intervention coordinates")
    for slot in _EXPECTED_SLOTS:
        for k in _EXPECTED_K:
            observed = {
                (record.context_order, record.context_annotation)
                for record in cells
                if record.answer_model_slot == slot and record.k == k
            }
            if observed != set(TASK13_CONTEXT_CONDITIONS):
                raise ValueError(
                    "cell statistics must contain the frozen typed intervention conditions"
                )
    metric_keys = [(coordinate, record.metric_path) for coordinate, record in zip(coordinates, cells)]
    if len(set(metric_keys)) != len(metric_keys):
        raise ValueError("cell statistics contain duplicate statistic rows")
    coordinate_counts = Counter(coordinates)
    if len(coordinate_counts) != 18 or set(coordinate_counts.values()) != {_EXPECTED_METRIC_COUNT}:
        raise ValueError("cell statistics must contain 18 complete cells with seven metrics")
    if set(record.answer_model_slot for record in cells) != set(_EXPECTED_SLOTS):
        raise ValueError("cell statistics contain a foreign answer-model slot")
    if set(record.k for record in cells) != set(_EXPECTED_K):
        raise ValueError("cell statistics contain a foreign retrieval k")
    slot_k_counts = Counter((coordinate[0], coordinate[1]) for coordinate in coordinate_counts)
    if len(slot_k_counts) != 6 or set(slot_k_counts.values()) != {3}:
        raise ValueError("cell statistics must contain exactly three cells per slot and k")
    for coordinate, count in coordinate_counts.items():
        metrics = {
            record.metric_path
            for record in cells
            if (record.answer_model_slot, record.k, record.cell_id) == coordinate
        }
        if count != _EXPECTED_METRIC_COUNT or metrics != set(TASK13_METRIC_PATHS):
            raise ValueError("cell statistics have missing or foreign metric rows")

    # All metrics from one cell must refer to the same authenticated run and
    # source hashes.  This also makes source binding deterministic.
    for coordinate in coordinate_counts:
        source_rows = [
            record
            for record in cells
            if (record.answer_model_slot, record.k, record.cell_id) == coordinate
        ]
        source_tuples = {
            (record.run_id, record.run_manifest_sha256, record.score_artifact_sha256)
            for record in source_rows
        }
        if len(source_tuples) != 1:
            raise ValueError("cell metrics do not share one exact run source")
    if len({record.task_identity_sha256 for record in cells}) != 1:
        raise ValueError("cell statistics do not share one task identity digest")

    return tuple(
        sorted(
            cells,
            key=lambda record: (
                record.answer_model_slot.encode("utf-8"),
                record.k,
                record.cell_id.encode("utf-8"),
                _metric_index(record.metric_path),
            ),
        )
    )


def _cell_source(record: Task13CellStatisticV1) -> Task13RunSourceV1:
    return Task13RunSourceV1(
        run_id=record.run_id,
        run_manifest_sha256=record.run_manifest_sha256,
        score_artifact_sha256=record.score_artifact_sha256,
    )


def _validate_contrasts(
    paired_contrasts: Sequence[Task13PairedContrastV1],
    cells: Sequence[Task13CellStatisticV1],
) -> tuple[Task13PairedContrastV1, ...]:
    contrasts = tuple(paired_contrasts)
    if len(contrasts) != _EXPECTED_CONTRAST_COUNT:
        raise ValueError("Task 13 requires exactly 84 paired contrasts")
    if any(not isinstance(record, Task13PairedContrastV1) for record in contrasts):
        raise TypeError("paired contrasts must be Task13PairedContrastV1 records")

    cell_by_key = {
        (record.answer_model_slot, record.k, record.cell_id, record.metric_path): record
        for record in cells
    }
    keys = [
        (
            record.answer_model_slot,
            record.k,
            record.left_cell_id,
            record.right_cell_id,
            record.metric_path,
        )
        for record in contrasts
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("paired contrasts contain duplicate statistic rows")
    coordinate_counts = Counter(
        (
            record.answer_model_slot,
            record.k,
            record.left_cell_id,
            record.right_cell_id,
        )
        for record in contrasts
    )
    if len(coordinate_counts) != 12 or set(coordinate_counts.values()) != {_EXPECTED_METRIC_COUNT}:
        raise ValueError("paired contrasts must contain 12 complete pairs with seven metrics")
    if set(record.answer_model_slot for record in contrasts) != set(_EXPECTED_SLOTS):
        raise ValueError("paired contrasts contain a foreign answer-model slot")
    if set(record.k for record in contrasts) != set(_EXPECTED_K):
        raise ValueError("paired contrasts contain a foreign retrieval k")
    slot_k_contrast_counts = Counter(
        (coordinate[0], coordinate[1]) for coordinate in coordinate_counts
    )
    if len(slot_k_contrast_counts) != 6 or set(slot_k_contrast_counts.values()) != {2}:
        raise ValueError("paired contrasts must contain exactly two contrasts per slot and k")
    for slot in _EXPECTED_SLOTS:
        for k in _EXPECTED_K:
            expected_pairs = set()
            for left_condition, right_condition in TASK13_CONTRAST_PAIRS:
                left = next(
                    record for record in cells
                    if record.answer_model_slot == slot
                    and record.k == k
                    and (record.context_order, record.context_annotation) == left_condition
                )
                right = next(
                    record for record in cells
                    if record.answer_model_slot == slot
                    and record.k == k
                    and (record.context_order, record.context_annotation) == right_condition
                )
                expected_pairs.add((left.cell_id, right.cell_id))
            observed_pairs = {
                (record.left_cell_id, record.right_cell_id)
                for record in contrasts
                if record.answer_model_slot == slot and record.k == k
            }
            if observed_pairs != expected_pairs:
                raise ValueError("paired contrasts do not use the frozen typed intervention pairs")
    for coordinate, count in coordinate_counts.items():
        metric_paths = {
            record.metric_path
            for record in contrasts
            if (
                record.answer_model_slot,
                record.k,
                record.left_cell_id,
                record.right_cell_id,
            ) == coordinate
        }
        if count != _EXPECTED_METRIC_COUNT or metric_paths != set(TASK13_METRIC_PATHS):
            raise ValueError("paired contrasts have missing or foreign metric rows")

    if len({record.contrast_id for record in contrasts}) != len(contrasts):
        raise ValueError("paired contrast IDs must be globally unique")

    for record in contrasts:
        expected_contrast_id = task13_contrast_id_v1(
            record.answer_model_slot,
            record.k,
            record.left_cell_id,
            record.right_cell_id,
            record.metric_path,
        )
        if record.contrast_id != expected_contrast_id:
            raise ValueError("paired contrast ID is not deterministic for its coordinates")
        left_key = (record.answer_model_slot, record.k, record.left_cell_id, record.metric_path)
        right_key = (record.answer_model_slot, record.k, record.right_cell_id, record.metric_path)
        left_cell = cell_by_key.get(left_key)
        right_cell = cell_by_key.get(right_key)
        if left_cell is None or right_cell is None:
            raise ValueError("paired contrast references a foreign cell statistic")
        if record.left_source != _cell_source(left_cell):
            raise ValueError("paired contrast left source does not match its cell statistic")
        if record.right_source != _cell_source(right_cell):
            raise ValueError("paired contrast right source does not match its cell statistic")
        if record.task_identity_sha256 != left_cell.task_identity_sha256:
            raise ValueError("paired contrast task identity does not match its left cell")
        if left_cell.task_identity_sha256 != right_cell.task_identity_sha256:
            raise ValueError("paired contrast cells have different task identities")
        if record.core_ids_sha256 != left_cell.core_ids_sha256 or record.core_ids_sha256 != right_cell.core_ids_sha256:
            raise ValueError("paired contrast core-ID hash does not match its cells")
        if record.bootstrap_indices_sha256 != left_cell.bootstrap_indices_sha256 or record.bootstrap_indices_sha256 != right_cell.bootstrap_indices_sha256:
            raise ValueError("paired contrast bootstrap hash does not match its cells")
        left_interval = left_cell.interval
        right_interval = right_cell.interval
        if left_interval.status != right_interval.status or record.interval.status != left_interval.status:
            raise ValueError("paired contrast has mixed support states")
        if left_interval.status.value == "unsupported":
            if (
                left_interval.support != right_interval.support
                or left_interval.support_sha256 != right_interval.support_sha256
                or record.interval.support != left_interval.support
                or record.interval.support_sha256 != left_interval.support_sha256
            ):
                raise ValueError("unsupported paired contrast support metadata must match both cells")

    return tuple(
        sorted(
            contrasts,
            key=lambda record: (
                record.answer_model_slot.encode("utf-8"),
                record.k,
                record.left_cell_id.encode("utf-8"),
                record.right_cell_id.encode("utf-8"),
                record.contrast_id.encode("utf-8"),
                _metric_index(record.metric_path),
            ),
        )
    )


def _resolve_hash(
    explicit: str | None,
    hashes: Mapping[str, str] | None,
    *keys: str,
) -> str:
    """Resolve a hash while rejecting every recognized disagreement."""

    candidates: list[tuple[str, str]] = []
    if explicit is not None:
        candidates.append(("explicit", explicit))
    if hashes is not None:
        for key in keys:
            if key in hashes:
                candidates.append((key, hashes[key]))
    if not candidates:
        raise TypeError(f"missing required hash; expected one of {keys!r}")
    values = {value for _, value in candidates}
    if len(values) != 1:
        labels = ", ".join(label for label, _ in candidates)
        raise ValueError(f"recognized hash aliases disagree: {labels}")
    return candidates[0][1]


def build_task13_statistics_receipt_v1(
    cell_statistics: Sequence[Task13CellStatisticV1] | Any,
    paired_contrasts: Sequence[Task13PairedContrastV1] | None = None,
    *,
    task12_preparation_manifest_sha256: str | None = None,
    task12_plan_sha256: str | None = None,
    task12_matrix_manifest_sha256: str | None = None,
    task12_matrix_summary_sha256: str | None = None,
    task12_integrity_audit_sha256: str | None = None,
    task12_hashes: Mapping[str, str] | None = None,
    statistics_config_sha256: str | None = None,
    task13_runtime_revision: str,
    task13_runtime_tree_sha256: str,
    core_ids_sha256: str,
    bootstrap_indices_sha256: str,
    cell_statistics_artifact: ArtifactRef,
    paired_contrasts_artifact: ArtifactRef,
    receipt_id: str = "task13-statistics-receipt-v1",
) -> Task13StatisticsReceiptV1:
    """Build the authenticated statistics receipt without touching the filesystem."""

    cells, contrasts = _coerce_statistics(cell_statistics, paired_contrasts)
    ordered_cells = _validate_cells(cells)
    ordered_contrasts = _validate_contrasts(contrasts, ordered_cells)
    if statistics_config_sha256 is None:
        raise TypeError("statistics_config_sha256 is required")
    cell_artifact = _validate_artifact(
        cell_statistics_artifact,
        path=TASK13_CELL_STATISTICS_ARTIFACT_PATH,
        label="cell_statistics_artifact",
    )
    contrast_artifact = _validate_artifact(
        paired_contrasts_artifact,
        path=TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH,
        label="paired_contrasts_artifact",
    )
    expected_cell_sha256 = _canonical_jsonl_sha256_v1(ordered_cells)
    expected_contrast_sha256 = _canonical_jsonl_sha256_v1(ordered_contrasts)
    if cell_artifact.sha256 != expected_cell_sha256:
        raise ValueError("cell_statistics_artifact.sha256 does not match canonical cell JSONL")
    if contrast_artifact.sha256 != expected_contrast_sha256:
        raise ValueError("paired_contrasts_artifact.sha256 does not match canonical contrast JSONL")
    run_ids = {record.run_id for record in ordered_cells}
    run_source_tuples = {
        (record.run_id, record.run_manifest_sha256, record.score_artifact_sha256)
        for record in ordered_cells
    }
    if len(run_source_tuples) != _EXPECTED_RUN_COUNT or len(run_ids) != _EXPECTED_RUN_COUNT:
        raise ValueError("cell statistics must cover exactly 18 distinct run sources")
    all_statistics = (*ordered_cells, *ordered_contrasts)
    core_hashes = {record.core_ids_sha256 for record in all_statistics}
    bootstrap_hashes = {record.bootstrap_indices_sha256 for record in all_statistics}
    if len(core_hashes) != 1 or next(iter(core_hashes)) != core_ids_sha256:
        raise ValueError("core_ids_sha256 must equal the unique source row hash")
    if len(bootstrap_hashes) != 1 or next(iter(bootstrap_hashes)) != bootstrap_indices_sha256:
        raise ValueError("bootstrap_indices_sha256 must equal the unique source row hash")
    for record in all_statistics:
        if record.core_ids_sha256 != core_ids_sha256:
            raise ValueError("statistics use inconsistent core-ID hashes")
        if record.bootstrap_indices_sha256 != bootstrap_indices_sha256:
            raise ValueError("statistics use inconsistent bootstrap-index hashes")
    bootstrap_config_hashes = {
        record.bootstrap_config_sha256 for record in all_statistics
    }
    if len(bootstrap_config_hashes) != 1:
        raise ValueError("statistics use inconsistent bootstrap-config hashes")
    bootstrap_config_sha256 = next(iter(bootstrap_config_hashes))
    if statistics_config_sha256 != bootstrap_config_sha256:
        raise ValueError(
            "statistics_config_sha256 must equal the sole bootstrap-config hash"
        )

    return Task13StatisticsReceiptV1(
        receipt_id=receipt_id,
        task12_preparation_manifest_sha256=_resolve_hash(
            task12_preparation_manifest_sha256,
            task12_hashes,
            "task12_preparation_manifest_sha256",
            "preparation_manifest",
        ),
        task12_plan_sha256=_resolve_hash(
            task12_plan_sha256,
            task12_hashes,
            "task12_plan_sha256",
            "plan",
        ),
        task12_matrix_manifest_sha256=_resolve_hash(
            task12_matrix_manifest_sha256,
            task12_hashes,
            "task12_matrix_manifest_sha256",
            "matrix_manifest",
        ),
        task12_matrix_summary_sha256=_resolve_hash(
            task12_matrix_summary_sha256,
            task12_hashes,
            "task12_matrix_summary_sha256",
            "matrix_summary",
        ),
        task12_integrity_audit_sha256=_resolve_hash(
            task12_integrity_audit_sha256,
            task12_hashes,
            "task12_integrity_audit_sha256",
            "integrity_audit",
        ),
        statistics_config_sha256=statistics_config_sha256,
        task13_runtime_revision=task13_runtime_revision,
        task13_runtime_tree_sha256=task13_runtime_tree_sha256,
        semantic_core_count=TASK13_SEMANTIC_CORE_COUNT,
        task_count=1440,
        run_count=_EXPECTED_RUN_COUNT,
        cell_statistic_count=len(ordered_cells),
        paired_contrast_count=len(ordered_contrasts),
        core_ids_sha256=core_ids_sha256,
        bootstrap_indices_sha256=bootstrap_indices_sha256,
        cell_statistics_artifact_id=TASK13_CELL_STATISTICS_ARTIFACT_ID,
        cell_statistics_artifact=cell_artifact,
        paired_contrasts_artifact_id=TASK13_PAIRED_CONTRASTS_ARTIFACT_ID,
        paired_contrasts_artifact=contrast_artifact,
    )


def _binding_from_case(case: Task13CaseRecordV1) -> Task13CaseBindingV1:
    if not isinstance(case, Task13CaseRecordV1):
        raise TypeError("cases must be full Task13CaseRecordV1 records")
    return Task13CaseBindingV1(
        case_id=case.case_id,
        run_id=case.run_id,
        task_id=case.task_id,
        category=case.category,
    )


def _canonical_case_bindings(
    bindings: Sequence[Task13CaseBindingV1],
    sources: Sequence[Task13RunSourceV1],
) -> tuple[Task13CaseBindingV1, ...]:
    source_order = {source.run_id: index for index, source in enumerate(sources)}
    try:
        return tuple(
            sorted(
                bindings,
                key=lambda binding: (
                    source_order[binding.run_id],
                    _CASE_CATEGORY_ORDER[binding.category],
                    binding.case_id.encode("utf-8"),
                ),
            )
        )
    except KeyError as exc:
        raise ValueError("case binding uses an unknown category or run source") from exc


def _validate_canonical_case_binding_order(
    bindings: Sequence[Task13CaseBindingV1],
    sources: Sequence[Task13RunSourceV1],
) -> None:
    expected = _canonical_case_bindings(bindings, sources)
    if tuple(bindings) != expected:
        raise ValueError(
            "case bindings are not in canonical order for the supplied run sources"
        )


def _revalidate_case_index(index: Task13CaseIndexV1) -> Task13CaseIndexV1:
    if not isinstance(index, Task13CaseIndexV1):
        raise TypeError("case_index must be Task13CaseIndexV1")
    validated = Task13CaseIndexV1.model_validate(index.model_dump(mode="python"))
    _validate_canonical_case_binding_order(
        validated.case_bindings,
        validated.run_sources,
    )
    return validated


def build_task13_case_index_v1(
    cases: Sequence[Task13CaseRecordV1],
    coverage: Sequence[Task13RunCaseCoverageV1],
    run_sources: Sequence[Task13RunSourceV1],
    cases_artifact: ArtifactRef,
    *,
    source_bindings: Sequence[Task13ArtifactBindingV1] = (),
) -> Task13CaseIndexV1:
    """Build a case index only from the complete authenticated case records."""

    supplied_cases = tuple(cases)
    if not supplied_cases:
        raise ValueError("case index requires non-empty full case records")
    if any(not isinstance(case, Task13CaseRecordV1) for case in supplied_cases):
        raise TypeError("cases must contain only Task13CaseRecordV1 records")
    supplied_coverage = tuple(coverage)
    supplied_sources = tuple(run_sources)
    if len(supplied_sources) != _EXPECTED_RUN_COUNT:
        raise ValueError("case index requires exactly 18 ordered run sources")
    if any(not isinstance(source, Task13RunSourceV1) for source in supplied_sources):
        raise TypeError("run_sources must be Task13RunSourceV1 records")
    if len({source.run_id for source in supplied_sources}) != _EXPECTED_RUN_COUNT:
        raise ValueError("case index run sources must have unique run IDs")
    if tuple(row.run_id for row in supplied_coverage) != tuple(source.run_id for source in supplied_sources):
        raise ValueError("coverage must use the exact ordered run-source IDs")
    if any(not isinstance(row, Task13RunCaseCoverageV1) for row in supplied_coverage):
        raise TypeError("coverage must be Task13RunCaseCoverageV1 records")
    source_by_id = {source.run_id: source for source in supplied_sources}
    bindings = tuple(_binding_from_case(case) for case in supplied_cases)
    for case in supplied_cases:
        source = source_by_id.get(case.run_id)
        if source is None:
            raise ValueError("case record references a foreign run source")
        if (
            case.run_manifest_sha256 != source.run_manifest_sha256
            or case.score_artifact_sha256 != source.score_artifact_sha256
        ):
            raise ValueError("case record source hashes do not match the run source")
    if len({binding.case_id for binding in bindings}) != len(bindings):
        raise ValueError("case index case IDs must be unique")
    _validate_canonical_case_binding_order(bindings, supplied_sources)
    expected_cases_sha256 = _canonical_jsonl_sha256_v1(supplied_cases)
    artifact = _validate_artifact(cases_artifact, path="cases.jsonl", label="cases_artifact")
    if artifact.sha256 != expected_cases_sha256:
        raise ValueError("cases_artifact.sha256 does not match canonical supplied cases")
    index = Task13CaseIndexV1(
        cases_artifact=artifact,
        record_count=len(bindings),
        case_bindings=bindings,
        coverage=supplied_coverage,
        run_sources=supplied_sources,
        source_bindings=tuple(source_bindings),
    )
    return _revalidate_case_index(index)


def _case_ids_for_sources(
    case_index: Task13CaseIndexV1,
    sources: Sequence[Task13RunSourceV1],
    *,
    source_by_id: Mapping[str, Task13RunSourceV1] | None = None,
    case_ids_by_run: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    source_map = source_by_id or {source.run_id: source for source in case_index.run_sources}
    for source in sources:
        if source_map.get(source.run_id) != source:
            raise ValueError("claim source does not exactly match the case-index run source")
    if case_ids_by_run is None:
        grouped: dict[str, list[str]] = {}
        for binding in case_index.case_bindings:
            grouped.setdefault(binding.run_id, []).append(binding.case_id)
        case_ids_by_run = {run_id: tuple(ids) for run_id, ids in grouped.items()}
    relevant = tuple(
        case_id
        for source in sources
        for case_id in case_ids_by_run.get(source.run_id, ())
    )
    if not relevant or any(source.run_id not in case_ids_by_run for source in sources):
        raise ValueError("claim source has no case-index coverage")
    return relevant


def _revalidate_receipt(receipt: Task13StatisticsReceiptV1) -> Task13StatisticsReceiptV1:
    if not isinstance(receipt, Task13StatisticsReceiptV1):
        raise TypeError("receipt must be Task13StatisticsReceiptV1")
    return Task13StatisticsReceiptV1.model_validate(receipt.model_dump(mode="python"))


def _claim_identity(
    *,
    kind: str,
    slot: str,
    cell_or_contrast: str,
    metric_path: str,
    slice_payload: dict[str, Any],
) -> str:
    payload = _ClaimIdentityPayloadV1(
        kind=kind,
        slot=slot,
        cell_or_contrast=cell_or_contrast,
        metric_path=metric_path,
        slice=slice_payload,
    )
    return f"claim-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _claim_from_cell(
    record: Task13CellStatisticV1,
    *,
    receipt_sha256: str,
    case_index: Task13CaseIndexV1,
    case_index_sha256: str,
    source_by_id: Mapping[str, Task13RunSourceV1],
    case_ids_by_run: Mapping[str, tuple[str, ...]],
    denominator: Task13DenominatorV1,
) -> Task13ClaimLedgerRecordV1:
    source = _cell_source(record)
    slice_payload = {"cell_id": record.cell_id, "k": record.k}
    return Task13ClaimLedgerRecordV1(
        claim_id=_claim_identity(
            kind="direct_cell",
            slot=record.answer_model_slot,
            cell_or_contrast=record.cell_id,
            metric_path=record.metric_path,
            slice_payload=slice_payload,
        ),
        kind="direct_cell",
        direction="self",
        slot=record.answer_model_slot,
        cell_or_contrast=record.cell_id,
        metric_path=record.metric_path,
        slice_payload=slice_payload,
        denominator=denominator,
        status=record.interval.status,
        interval=record.interval,
        run_sources=(source,),
        statistics_receipt_sha256=receipt_sha256,
        case_ids=_case_ids_for_sources(
            case_index,
            (source,),
            source_by_id=source_by_id,
            case_ids_by_run=case_ids_by_run,
        ),
        case_index_sha256=case_index_sha256,
    )


def _claim_from_contrast(
    record: Task13PairedContrastV1,
    *,
    receipt_sha256: str,
    case_index: Task13CaseIndexV1,
    case_index_sha256: str,
    source_by_id: Mapping[str, Task13RunSourceV1],
    case_ids_by_run: Mapping[str, tuple[str, ...]],
    denominator: Task13DenominatorV1,
) -> Task13ClaimLedgerRecordV1:
    sources = (record.left_source, record.right_source)
    slice_payload = {
        "k": record.k,
        "left_cell_id": record.left_cell_id,
        "right_cell_id": record.right_cell_id,
    }
    return Task13ClaimLedgerRecordV1(
        claim_id=_claim_identity(
            kind="paired_contrast",
            slot=record.answer_model_slot,
            cell_or_contrast=record.contrast_id,
            metric_path=record.metric_path,
            slice_payload=slice_payload,
        ),
        kind="paired_contrast",
        direction="left_minus_right",
        slot=record.answer_model_slot,
        cell_or_contrast=record.contrast_id,
        metric_path=record.metric_path,
        slice_payload=slice_payload,
        denominator=denominator,
        status=record.interval.status,
        interval=record.interval,
        run_sources=sources,
        statistics_receipt_sha256=receipt_sha256,
        case_ids=_case_ids_for_sources(
            case_index,
            sources,
            source_by_id=source_by_id,
            case_ids_by_run=case_ids_by_run,
        ),
        case_index_sha256=case_index_sha256,
    )


def build_task13_claim_ledger_v1(
    cell_statistics: Sequence[Task13CellStatisticV1] | Any,
    paired_contrasts: Sequence[Task13PairedContrastV1] | None = None,
    *,
    receipt: Task13StatisticsReceiptV1,
    case_index: Task13CaseIndexV1,
    denominator: Task13DenominatorV1 | None = None,
    expected_statistics_receipt_sha256: str | None = None,
    expected_case_index_sha256: str | None = None,
) -> Task13LedgerResultV1:
    """Generate exactly one canonical claim row per statistic and contrast."""

    cells, contrasts = _coerce_statistics(cell_statistics, paired_contrasts)
    ordered_cells = _validate_cells(cells)
    ordered_contrasts = _validate_contrasts(contrasts, ordered_cells)
    validated_receipt = _revalidate_receipt(receipt)
    if validated_receipt.cell_statistic_count != len(ordered_cells) or validated_receipt.paired_contrast_count != len(ordered_contrasts):
        raise ValueError("statistics receipt counts do not match the supplied statistics")
    if validated_receipt.cell_statistic_count != _EXPECTED_CELL_COUNT or validated_receipt.paired_contrast_count != _EXPECTED_CONTRAST_COUNT:
        raise ValueError("statistics receipt does not contain the frozen Task 13 cardinalities")
    all_statistics = (*ordered_cells, *ordered_contrasts)
    if len({record.core_ids_sha256 for record in all_statistics}) != 1 or validated_receipt.core_ids_sha256 != ordered_cells[0].core_ids_sha256:
        raise ValueError("statistics receipt core-ID hash does not match supplied rows")
    if len({record.bootstrap_indices_sha256 for record in all_statistics}) != 1 or validated_receipt.bootstrap_indices_sha256 != ordered_cells[0].bootstrap_indices_sha256:
        raise ValueError("statistics receipt bootstrap hash does not match supplied rows")
    config_hashes = {record.bootstrap_config_sha256 for record in all_statistics}
    if len(config_hashes) != 1 or validated_receipt.statistics_config_sha256 != next(iter(config_hashes)):
        raise ValueError("statistics receipt config hash does not match supplied rows")
    cell_sha256 = _canonical_jsonl_sha256_v1(ordered_cells)
    contrast_sha256 = _canonical_jsonl_sha256_v1(ordered_contrasts)
    if validated_receipt.cell_statistics_artifact.sha256 != cell_sha256:
        raise ValueError("statistics receipt cell artifact is stale for supplied rows")
    if validated_receipt.paired_contrasts_artifact.sha256 != contrast_sha256:
        raise ValueError("statistics receipt contrast artifact is stale for supplied rows")
    source_ids = {
        (record.run_id, record.run_manifest_sha256, record.score_artifact_sha256)
        for record in ordered_cells
    }
    if len(source_ids) != _EXPECTED_RUN_COUNT or len({record.run_id for record in ordered_cells}) != _EXPECTED_RUN_COUNT:
        raise ValueError("supplied statistics must contain exactly 18 distinct run sources")
    canonical_case_index = _revalidate_case_index(case_index)
    receipt_sha256 = sha256_model(validated_receipt)
    case_index_sha256 = sha256_model(canonical_case_index)
    if expected_statistics_receipt_sha256 is not None and receipt_sha256 != expected_statistics_receipt_sha256:
        raise ValueError("statistics receipt hash does not match the expected binding")
    if expected_case_index_sha256 is not None and case_index_sha256 != expected_case_index_sha256:
        raise ValueError("case index hash does not match the expected binding")

    denominator = denominator or Task13DenominatorV1(
        task_count=80,
        semantic_core_count=20,
        tasks_per_core=4,
    )
    if denominator != Task13DenominatorV1(task_count=80, semantic_core_count=20, tasks_per_core=4):
        raise ValueError("Task 13 claims require the frozen denominator")
    source_by_id = {source.run_id: source for source in canonical_case_index.run_sources}
    case_ids_by_run_mutable: dict[str, list[str]] = {}
    for binding in canonical_case_index.case_bindings:
        case_ids_by_run_mutable.setdefault(binding.run_id, []).append(binding.case_id)
    case_ids_by_run = {run_id: tuple(ids) for run_id, ids in case_ids_by_run_mutable.items()}
    claims = tuple(
        _claim_from_cell(
            record,
            receipt_sha256=receipt_sha256,
            case_index=canonical_case_index,
            case_index_sha256=case_index_sha256,
            source_by_id=source_by_id,
            case_ids_by_run=case_ids_by_run,
            denominator=denominator,
        )
        for record in ordered_cells
    ) + tuple(
        _claim_from_contrast(
            record,
            receipt_sha256=receipt_sha256,
            case_index=canonical_case_index,
            case_index_sha256=case_index_sha256,
            source_by_id=source_by_id,
            case_ids_by_run=case_ids_by_run,
            denominator=denominator,
        )
        for record in ordered_contrasts
    )
    if len(claims) != _EXPECTED_CELL_COUNT + _EXPECTED_CONTRAST_COUNT:
        raise AssertionError("Task 13 claim ledger cardinality is not frozen")
    if len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("Task 13 claim IDs are not unique")
    if any(claim.case_index_sha256 != case_index_sha256 for claim in claims):
        raise AssertionError("Task 13 claims do not share one case-index binding")
    return Task13LedgerResultV1(
        receipt=validated_receipt,
        case_index=canonical_case_index,
        claims=claims,
    )


def verify_task13_claim_ledger_v1(
    claims: Sequence[Task13ClaimLedgerRecordV1],
    cell_statistics: Sequence[Task13CellStatisticV1] | Any,
    paired_contrasts: Sequence[Task13PairedContrastV1] | None = None,
    *,
    receipt: Task13StatisticsReceiptV1,
    case_index: Task13CaseIndexV1,
) -> None:
    """Rebuild and byte-compare claims to detect receipt/source tampering."""

    expected = build_task13_claim_ledger_v1(
        cell_statistics,
        paired_contrasts,
        receipt=receipt,
        case_index=case_index,
    ).claims
    supplied = tuple(claims)
    if len(supplied) != len(expected):
        raise ValueError("claim ledger cardinality differs from authenticated statistics")
    if any(canonical_json_bytes(actual) != canonical_json_bytes(wanted) for actual, wanted in zip(supplied, expected)):
        raise ValueError("claim ledger does not equal the authenticated statistics binding")


# Short aliases make the focused API convenient while keeping the v1 names
# explicit for callers that persist contracts.
build_task13_receipt_v1 = build_task13_statistics_receipt_v1
build_task13_ledger_v1 = build_task13_claim_ledger_v1


__all__ = [
    "Task13LedgerResultV1",
    "build_task13_case_index_v1",
    "build_task13_claim_ledger_v1",
    "build_task13_ledger_v1",
    "build_task13_receipt_v1",
    "build_task13_statistics_receipt_v1",
    "canonical_jsonl_bytes_v1",
    "verify_task13_claim_ledger_v1",
]
