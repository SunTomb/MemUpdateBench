from __future__ import annotations

from enum import Enum
import re
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BeforeValidator, Field, field_validator, model_validator

from mub.vnext.contracts.common import (
    ArtifactRef,
    ImmutableContractModel,
    MetricFieldSupport,
    SHA256_PATTERN,
    StrictNonnegativeInt,
)
from mub.vnext.contracts.v3.common import FrozenJsonObjectV3, StrictIdentifier, StrictPositiveInt
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS
from mub.vnext.io import sha256_model


TASK13_CONFIG_SCHEMA_VERSION = "memupdatebench.core-task13-statistics-config.v1"
TASK13_CONTRACT_SCHEMA_VERSION = "memupdatebench.core-task13-statistics.v1"
TASK13_SEED_HEX = "9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542"

# The order here is part of the tracked configuration contract.  It is not
# derived from set or mapping iteration order.
TASK13_METRIC_PATHS = (
    "answer_scores.exact_match",
    "answer_scores.gold_retrieved_wrong_answer",
    "answer_scores.stale_copied",
    "answer_scores.token_f1",
    "protocol_scores.answer_parse_valid",
    "retrieval_scores.stale_count_in_context",
    "retrieval_scores.stale_exposure_rate",
)
# Backwards-compatible name retained for the first contract commit.  Both
# names reference the same immutable tuple, so no parallel metric dictionary
# can drift from the frozen Task 13 set.
CORE_TASK13_METRIC_PATHS = TASK13_METRIC_PATHS
if len(TASK13_METRIC_PATHS) != 7 or not set(TASK13_METRIC_PATHS) <= CORE_METRIC_FIELD_PATHS:
    raise RuntimeError("Task 13 metric paths must be a unique subset of the v3 core registry")

SHA256 = Annotated[str, Field(strict=True, pattern=SHA256_PATTERN)]

_CANONICAL_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?")


def canonical_decimal_string(value: Any) -> str:
    """Validate a finite, non-exponent, minimally formatted decimal string."""

    if type(value) is not str:
        raise ValueError("decimal statistics must be strings")
    if value == "-0":
        return "0"
    if not _CANONICAL_DECIMAL.fullmatch(value) and not (
        value.startswith("-") and _CANONICAL_DECIMAL.fullmatch(value[1:])
    ):
        raise ValueError("decimal string is not canonical")
    # The grammar excludes exponents and non-finite values; this conversion is
    # retained as an explicit finite check against future grammar changes.
    from decimal import Decimal, InvalidOperation

    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("decimal string is invalid") from exc
    if not parsed.is_finite():
        raise ValueError("decimal string must be finite")
    return value


CanonicalDecimal = Annotated[str, BeforeValidator(canonical_decimal_string)]


class Task13StatisticStatus(str, Enum):
    NUMERIC = "numeric"
    UNSUPPORTED = "unsupported"


def _require_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique identifiers")
    return values


def _require_nonblank_path(value: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise ValueError("artifact paths must be nonblank strings")
    return value


class Task13BootstrapConfigV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONFIG_SCHEMA_VERSION] = TASK13_CONFIG_SCHEMA_VERSION
    cluster_key: Literal["semantic_core_id"]
    expected_cluster_count: Literal[80]
    seed_hex: Literal[TASK13_SEED_HEX]
    replicates: Literal[10_000]
    draws_per_replicate: Literal[80]
    confidence_level: Literal["0.95"]
    interval_method: Literal["clustered_percentile"]
    quantile_method: Literal["inverted_cdf"]
    lower_order_statistic: Literal[250]
    upper_order_statistic: Literal[9_750]
    decimal_precision: Literal[50]
    decimal_rounding: Literal["ROUND_HALF_EVEN"]
    support_policy: Literal["all_supported_or_all_unsupported"]
    metric_paths: tuple[str, ...]

    @field_validator("metric_paths", mode="before")
    @classmethod
    def _exact_metric_paths(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in {list, tuple}:
            raise ValueError("metric_paths must be an ordered list or tuple")
        if any(type(path) is not str for path in value):
            raise ValueError("metric paths must be exact built-in strings")
        paths = tuple(value)
        if paths != TASK13_METRIC_PATHS:
            raise ValueError("metric_paths must equal the frozen Task 13 metric order")
        if len(paths) != len(set(paths)) or not set(paths) <= CORE_METRIC_FIELD_PATHS:
            raise ValueError("metric_paths contain unknown or duplicate core metrics")
        return paths


class Task13ArtifactBindingV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    artifact_id: StrictIdentifier
    artifact: ArtifactRef
    role: StrictIdentifier | None = None

    @field_validator("artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        _require_nonblank_path(value.path)
        return value


class Task13IntervalV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    status: Task13StatisticStatus
    estimate: CanonicalDecimal | None = None
    lower: CanonicalDecimal | None = None
    upper: CanonicalDecimal | None = None
    support: MetricFieldSupport | None = None
    support_sha256: SHA256 | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> Task13IntervalV1:
        from decimal import Decimal

        values = (self.estimate, self.lower, self.upper)
        if self.status == Task13StatisticStatus.NUMERIC:
            if any(value is None for value in values):
                raise ValueError("numeric intervals require estimate, lower, and upper")
            if self.support is not None or self.support_sha256 is not None:
                raise ValueError("numeric intervals cannot carry support metadata")
            if Decimal(self.lower) > Decimal(self.upper):
                raise ValueError("interval lower endpoint must not exceed upper endpoint")
            return self

        if any(value is not None for value in values):
            raise ValueError("unsupported intervals require null estimate and endpoints")
        if self.support is None or self.support_sha256 is None:
            raise ValueError("unsupported intervals require typed support metadata and hash")
        expected_hash = sha256_model(self.support)
        if self.support_sha256 != expected_hash:
            raise ValueError("support_sha256 does not match the typed support payload")
        return self


class Task13CellStatisticV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    cell_id: StrictIdentifier
    answer_model_slot: StrictIdentifier
    k: StrictPositiveInt
    metric_path: StrictIdentifier
    interval: Task13IntervalV1
    task_count: StrictPositiveInt
    core_count: Literal[80]
    core_ids_sha256: SHA256
    run_id: StrictIdentifier
    run_manifest_sha256: SHA256
    score_artifact_sha256: SHA256
    bootstrap_config_sha256: SHA256
    bootstrap_indices_sha256: SHA256

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value


class Task13PairedContrastV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    contrast_id: StrictIdentifier
    left_cell_id: StrictIdentifier
    right_cell_id: StrictIdentifier
    direction: Literal["left_minus_right"]
    answer_model_slot: StrictIdentifier
    k: StrictPositiveInt
    metric_path: StrictIdentifier
    interval: Task13IntervalV1
    core_count: Literal[80]
    core_ids_sha256: SHA256
    left_run_id: StrictIdentifier
    left_run_manifest_sha256: SHA256
    left_score_artifact_sha256: SHA256
    right_run_id: StrictIdentifier
    right_run_manifest_sha256: SHA256
    right_score_artifact_sha256: SHA256
    bootstrap_config_sha256: SHA256
    bootstrap_indices_sha256: SHA256

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value

    @model_validator(mode="after")
    def _distinct_cells(self) -> Task13PairedContrastV1:
        if self.left_cell_id == self.right_cell_id:
            raise ValueError("paired contrast left and right cells must differ")
        return self


class Task13StatisticsReceiptV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    receipt_id: StrictIdentifier
    task12_preparation_manifest_sha256: SHA256
    task12_plan_sha256: SHA256
    task12_matrix_manifest_sha256: SHA256
    task12_matrix_summary_sha256: SHA256
    task12_integrity_audit_sha256: SHA256
    statistics_config_sha256: SHA256
    task13_runtime_revision: StrictIdentifier
    task13_runtime_tree_sha256: SHA256
    core_count: Literal[80]
    task_count: StrictPositiveInt
    core_ids_sha256: SHA256
    bootstrap_indices_sha256: SHA256
    cell_statistic_count: StrictNonnegativeInt
    paired_contrast_count: StrictNonnegativeInt
    cell_statistics_artifact: ArtifactRef
    paired_contrasts_artifact: ArtifactRef

    @field_validator("cell_statistics_artifact", "paired_contrasts_artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        _require_nonblank_path(value.path)
        return value


_CASE_CATEGORIES = ("correct", "stale_copied", "answer_parse_invalid", "other_wrong")
CaseCategory = Literal["correct", "stale_copied", "answer_parse_invalid", "other_wrong"]


class Task13CaseSelectorV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    selector_id: StrictIdentifier
    run_id: StrictIdentifier
    categories: tuple[CaseCategory, ...] = _CASE_CATEGORIES
    max_cases_per_category: Literal[1] = 1
    selected_case_ids: tuple[StrictIdentifier, ...] = ()

    @field_validator("categories")
    @classmethod
    def _exact_categories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != _CASE_CATEGORIES:
            raise ValueError("categories must equal the frozen Task 13 category order")
        return value

    @field_validator("selected_case_ids")
    @classmethod
    def _unique_selected_case_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_unique(value, "selected_case_ids")


class Task13CaseRecordV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    case_id: StrictIdentifier
    category: CaseCategory
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    answer_model_slot: StrictIdentifier
    k: StrictPositiveInt
    task_artifact_sha256: SHA256
    task_manifest_sha256: SHA256
    run_manifest_sha256: SHA256
    score_artifact_sha256: SHA256
    matrix_summary_sha256: SHA256
    task: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("task", "task_payload"))
    timeline: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("timeline", "timeline_payload"))
    run: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("run", "run_payload"))
    score: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("score", "score_payload"))
    retrieval: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("retrieval", "retrieval_payload"))
    answer: FrozenJsonObjectV3 = Field(validation_alias=AliasChoices("answer", "answer_payload"))

    @property
    def task_payload(self) -> FrozenJsonObjectV3:
        return self.task

    @property
    def timeline_payload(self) -> FrozenJsonObjectV3:
        return self.timeline

    @property
    def run_payload(self) -> FrozenJsonObjectV3:
        return self.run

    @property
    def score_payload(self) -> FrozenJsonObjectV3:
        return self.score

    @property
    def retrieval_payload(self) -> FrozenJsonObjectV3:
        return self.retrieval

    @property
    def answer_payload(self) -> FrozenJsonObjectV3:
        return self.answer


class Task13CaseIndexV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    case_ids: tuple[StrictIdentifier, ...]
    cases_artifact: ArtifactRef
    record_count: StrictPositiveInt
    run_ids: tuple[StrictIdentifier, ...]
    run_manifest_hashes: tuple[SHA256, ...]
    score_artifact_hashes: tuple[SHA256, ...]
    category_coverage: FrozenJsonObjectV3
    source_bindings: tuple[Task13ArtifactBindingV1, ...] = ()

    @field_validator("case_ids")
    @classmethod
    def _unique_case_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_unique(value, "case_ids")

    @field_validator("run_ids")
    @classmethod
    def _exact_run_coverage(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 18:
            raise ValueError("case index must cover exactly 18 Task 12 runs")
        return _require_unique(value, "run_ids")

    @field_validator("cases_artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        _require_nonblank_path(value.path)
        return value

    @model_validator(mode="after")
    def _count_sources_and_categories(self) -> Task13CaseIndexV1:
        if self.record_count != len(self.case_ids):
            raise ValueError("record_count must equal the number of case_ids")
        if not self.run_ids or not self.run_manifest_hashes or not self.score_artifact_hashes:
            raise ValueError("case index run IDs and source hashes must be non-empty")
        if not (
            len(self.run_ids)
            == len(self.run_manifest_hashes)
            == len(self.score_artifact_hashes)
        ):
            raise ValueError("run IDs and source hashes must have equal lengths")
        if set(self.category_coverage) != set(_CASE_CATEGORIES):
            raise ValueError("category_coverage must contain exactly all four case categories")
        case_id_set = set(self.case_ids)
        for category in _CASE_CATEGORIES:
            selected = self.category_coverage[category]
            if not isinstance(selected, (list, tuple)):
                raise ValueError("category_coverage values must be case-ID lists")
            if any(type(case_id) is not str or case_id not in case_id_set for case_id in selected):
                raise ValueError("category_coverage must reference known case IDs")
            if len(selected) != len(set(selected)):
                raise ValueError("category_coverage case IDs must be unique per category")
        _validate_unique_bindings(self.source_bindings)
        return self


class Task13ClaimLedgerRecordV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    claim_id: StrictIdentifier
    kind: Literal["direct_cell", "paired_contrast"]
    direction: Literal["self", "left_minus_right"]
    slot: StrictIdentifier
    cell_or_contrast: StrictIdentifier
    metric_path: StrictIdentifier
    slice_payload: FrozenJsonObjectV3
    denominator: StrictPositiveInt
    status: Task13StatisticStatus
    interval: Task13IntervalV1
    run_ids: tuple[StrictIdentifier, ...]
    run_manifest_sha256s: tuple[SHA256, ...]
    score_artifact_sha256s: tuple[SHA256, ...]
    statistics_receipt_sha256: SHA256
    case_ids: tuple[StrictIdentifier, ...]
    case_index_sha256: SHA256

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value

    @field_validator("run_ids", "run_manifest_sha256s", "score_artifact_sha256s", "case_ids")
    @classmethod
    def _unique_sequences(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return _require_unique(value, info.field_name)

    @model_validator(mode="after")
    def _ledger_consistency(self) -> Task13ClaimLedgerRecordV1:
        if self.status != self.interval.status:
            raise ValueError("claim status must equal the bound interval status")
        if self.kind == "direct_cell" and self.direction != "self":
            raise ValueError("direct_cell claims require direction='self'")
        if self.kind == "paired_contrast" and self.direction != "left_minus_right":
            raise ValueError("paired_contrast claims require direction='left_minus_right'")
        if len(self.run_ids) != len(self.run_manifest_sha256s) or len(self.run_ids) != len(self.score_artifact_sha256s):
            raise ValueError("claim run IDs and source hashes must have equal lengths")
        expected_sources = 1 if self.kind == "direct_cell" else 2
        if len(self.run_ids) != expected_sources:
            raise ValueError(
                f"{self.kind} claims must bind exactly {expected_sources} run/manifest/score sources"
            )
        return self


class Task13ArtifactIndexV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    artifacts: tuple[Task13ArtifactBindingV1, ...]

    @model_validator(mode="after")
    def _unique_artifacts(self) -> Task13ArtifactIndexV1:
        _validate_unique_bindings(self.artifacts)
        return self


def _validate_unique_bindings(bindings: tuple[Task13ArtifactBindingV1, ...]) -> None:
    ids = [binding.artifact_id for binding in bindings]
    paths = [binding.artifact.path for binding in bindings]
    if len(ids) != len(set(ids)):
        raise ValueError("artifact IDs must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("artifact paths must be unique")


__all__ = [
    "CORE_TASK13_METRIC_PATHS",
    "TASK13_METRIC_PATHS",
    "CanonicalDecimal",
    "Task13ArtifactBindingV1",
    "Task13ArtifactIndexV1",
    "Task13BootstrapConfigV1",
    "Task13CaseIndexV1",
    "Task13CaseRecordV1",
    "Task13CaseSelectorV1",
    "Task13CellStatisticV1",
    "Task13ClaimLedgerRecordV1",
    "Task13IntervalV1",
    "Task13PairedContrastV1",
    "Task13StatisticStatus",
    "Task13StatisticsReceiptV1",
    "canonical_decimal_string",
]
