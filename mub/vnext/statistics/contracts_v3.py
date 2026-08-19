from __future__ import annotations

from enum import Enum
import hashlib
import json
from collections.abc import Mapping
import re
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator

from mub.vnext.contracts.common import (
    ArtifactRef,
    ImmutableContractModel,
    MetricFieldSupport,
    SHA256_PATTERN,
    StrictNonnegativeInt,
    thaw_json,
)
from mub.vnext.contracts.v3.common import (
    FrozenJsonObjectV3,
    StrictIdentifier,
    StrictPositiveInt,
    typed_json_equal,
)
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS
from mub.vnext.contracts.v3.task import SourceRecordV3
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
TASK13_CONTEXT_CONDITIONS = (
    ("chronological", "none"),
    ("reverse_chronological", "none"),
    ("reverse_chronological", "latest_outdated_label"),
)
TASK13_CONTRAST_PAIRS = (
    (("reverse_chronological", "none"), ("chronological", "none")),
    (("reverse_chronological", "latest_outdated_label"), ("reverse_chronological", "none")),
)
TASK13_K_VALUES = (4, 8, 16)
TASK13_SLOTS = ("answer_model_a", "answer_model_b")
Task13AnswerModelSlot = Literal["answer_model_a", "answer_model_b"]
Task13RetrievalK = Literal[4, 8, 16]
TASK13_K = Task13RetrievalK
TASK13_CONTEXT_ORDER = Literal["chronological", "reverse_chronological"]
TASK13_CONTEXT_ANNOTATION = Literal["none", "latest_outdated_label"]
TASK13_TASK_COUNT = 80
TASK13_SEMANTIC_CORE_COUNT = 20
TASK13_TASKS_PER_CORE = 4
TASK13_CELL_STATISTICS_ARTIFACT_ID = "cell_statistics"
TASK13_CELL_STATISTICS_ARTIFACT_PATH = "cell_statistics.jsonl"
TASK13_PAIRED_CONTRASTS_ARTIFACT_ID = "paired_contrasts"
TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH = "paired_contrasts.jsonl"

TASK13_ARTIFACT_PATHS = (
    "bootstrap_indices.bin",
    TASK13_CELL_STATISTICS_ARTIFACT_PATH,
    TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH,
    "statistics_receipt.json",
    "cases.jsonl",
    "case_index.json",
    "claim_ledger.jsonl",
)
Task13ArtifactRole = Literal[
    "bootstrap_indices.bin",
    "cell_statistics.jsonl",
    "paired_contrasts.jsonl",
    "statistics_receipt.json",
    "cases.jsonl",
    "case_index.json",
    "claim_ledger.jsonl",
]
if len(TASK13_ARTIFACT_PATHS) != 7 or len(set(TASK13_ARTIFACT_PATHS)) != 7:
    raise RuntimeError("Task 13 artifact paths must be a unique seven-artifact publication set")

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


_TASK13_SAFE_SOURCE_FIELDS = frozenset({
    "source_id",
    "source_type",
    "source_uri",
    "license_or_privacy",
    "raw_hash",
    "normalized_hash",
    "normalization_version",
    "redacted",
})
_TASK13_PUBLIC_SOURCE_FIELDS = frozenset(SourceRecordV3.model_fields) | {"redacted"}
_TASK13_REDACTED_TIMELINE_FIELDS = frozenset({
    "event_id",
    "sequence_index",
    "timestamp",
    "speaker",
    "gold_action_ids",
    "role",
})
_TASK13_REDACTED_FORBIDDEN_FIELDS = frozenset({
    "raw_text",
    "normalized_text",
    "source_anchor",
    "metadata",
    "provenance",
    "generator",
})


def _contains_forbidden_field(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _TASK13_REDACTED_FORBIDDEN_FIELDS:
                return key
            found = _contains_forbidden_field(item)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _contains_forbidden_field(item)
            if found is not None:
                return found
    return None


def _is_json_scalar(value: Any) -> bool:
    return value is None or type(value) in {bool, int, float, str}


def _redacted_timeline_value_violation(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return "nested mapping"
    if isinstance(value, (tuple, list)):
        if any(not _is_json_scalar(item) for item in value):
            return "nested sequence"
        return None
    if not _is_json_scalar(value):
        return "non-scalar value"
    return None


def _redacted_timeline_violation(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key not in _TASK13_REDACTED_TIMELINE_FIELDS:
                return str(key)
            violation = _redacted_timeline_value_violation(item)
            if violation is not None:
                return str(key)
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _redacted_timeline_violation(item)
            if found is not None:
                return found
    return None


def _source_nested_container(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key, item in value.items():
        if isinstance(item, (Mapping, tuple, list)):
            return str(key)
    return None


def _validate_public_source_projection(source: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in source.items() if key != "redacted"}
    try:
        source_record = SourceRecordV3.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("public task source must match SourceRecordV3") from exc
    if not typed_json_equal(source_record.model_dump(mode="json"), thaw_json(payload)):
        raise ValueError("public task source must use canonical SourceRecordV3 content")


class Task13TaskIdentityPayloadV1(ImmutableContractModel):
    """Typed immutable payload for the canonical task-to-core assignment."""

    task_ids_by_core: tuple[tuple[StrictIdentifier, tuple[StrictIdentifier, ...]], ...]


class Task13ContrastIdentityPayloadV1(ImmutableContractModel):
    """Typed immutable payload for a directed Task 13 contrast ID."""

    slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    left_cell_id: StrictIdentifier
    right_cell_id: StrictIdentifier
    metric_path: StrictIdentifier

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value


def task13_task_identity_sha256_v1(
    task_ids_by_core: tuple[tuple[str, tuple[str, ...]], ...],
) -> str:
    payload = Task13TaskIdentityPayloadV1(task_ids_by_core=task_ids_by_core)
    return sha256_model(payload)


def task13_contrast_id_v1(
    slot: Task13AnswerModelSlot,
    k: Task13RetrievalK,
    left_cell_id: str,
    right_cell_id: str,
    metric_path: str,
) -> str:
    payload = Task13ContrastIdentityPayloadV1(
        slot=slot,
        k=k,
        left_cell_id=left_cell_id,
        right_cell_id=right_cell_id,
        metric_path=metric_path,
    )
    return f"contrast-{sha256_model(payload)}"


class Task13ClaimIdentityPayloadV1(ImmutableContractModel):
    """Typed immutable payload used for stable Task 13 claim IDs."""

    kind: Literal["direct_cell", "paired_contrast"]
    slot: Task13AnswerModelSlot
    cell_or_contrast: StrictIdentifier
    metric_path: StrictIdentifier
    slice: FrozenJsonObjectV3


def task13_claim_id_v1(
    kind: Literal["direct_cell", "paired_contrast"],
    slot: Task13AnswerModelSlot,
    cell_or_contrast: str,
    metric_path: str,
    slice_payload: Mapping[str, Any],
) -> str:
    payload = Task13ClaimIdentityPayloadV1(
        kind=kind,
        slot=slot,
        cell_or_contrast=cell_or_contrast,
        metric_path=metric_path,
        slice=slice_payload,
    )
    return f"claim-{sha256_model(payload)}"


def task13_case_id_v1(run_id: str, task_id: str, category: str) -> str:
    raw = json.dumps(
        {"category": category, "run_id": run_id, "task_id": task_id},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"case-{hashlib.sha256(raw).hexdigest()}"


class Task13BootstrapConfigV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONFIG_SCHEMA_VERSION] = TASK13_CONFIG_SCHEMA_VERSION
    cluster_key: Literal["semantic_core_id"]
    expected_cluster_count: Literal[20]
    seed_hex: Literal[TASK13_SEED_HEX]
    replicates: Literal[10_000]
    draws_per_replicate: Literal[20]
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
    role: Task13ArtifactRole | None = None

    @field_validator("artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        _require_nonblank_path(value.path)
        return value


class Task13RunSourceV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    answer_model_slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    run_manifest_sha256: SHA256
    score_artifact_sha256: SHA256


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
            if not (Decimal(self.lower) <= Decimal(self.estimate) <= Decimal(self.upper)):
                raise ValueError("interval estimate must lie between lower and upper endpoints")
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
    answer_model_slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    context_order: TASK13_CONTEXT_ORDER
    context_annotation: TASK13_CONTEXT_ANNOTATION
    metric_path: StrictIdentifier
    interval: Task13IntervalV1
    task_count: Literal[80]
    core_count: Literal[20]
    core_ids_sha256: SHA256
    task_identity_sha256: SHA256
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
    answer_model_slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    metric_path: StrictIdentifier
    interval: Task13IntervalV1
    core_count: Literal[20]
    core_ids_sha256: SHA256
    task_identity_sha256: SHA256
    left_source: Task13RunSourceV1
    right_source: Task13RunSourceV1
    bootstrap_config_sha256: SHA256
    bootstrap_indices_sha256: SHA256

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value

    @model_validator(mode="after")
    def _distinct_cells_and_sources(self) -> Task13PairedContrastV1:
        if self.left_cell_id == self.right_cell_id:
            raise ValueError("paired contrast left and right cells must differ")
        if self.left_source.run_id == self.right_source.run_id:
            raise ValueError("paired contrast sources must use different run IDs")
        if (
            self.left_source.run_manifest_sha256 == self.right_source.run_manifest_sha256
            and self.left_source.score_artifact_sha256 == self.right_source.score_artifact_sha256
        ):
            raise ValueError("paired contrast sources must not duplicate both source hashes")
        for label, source in (("left", self.left_source), ("right", self.right_source)):
            if (
                source.answer_model_slot != self.answer_model_slot
                or source.k != self.k
            ):
                raise ValueError(
                    f"paired contrast {label} source coordinates must match the contrast slot/k"
                )
        return self


class Task13StatisticsReceiptV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    receipt_id: StrictIdentifier
    task12_preparation_manifest_sha256: SHA256
    task12_plan_sha256: SHA256
    task12_matrix_manifest_sha256: SHA256
    task12_matrix_summary_sha256: SHA256
    task12_integrity_audit_sha256: SHA256
    core_tasks_sha256: SHA256
    core_task_manifest_sha256: SHA256
    statistics_config_sha256: SHA256
    task13_runtime_revision: StrictIdentifier
    task13_runtime_tree_sha256: SHA256
    semantic_core_count: Literal[20]
    task_count: Literal[1440]
    run_count: Literal[18]
    cell_statistic_count: Literal[126]
    paired_contrast_count: Literal[84]
    core_ids_sha256: SHA256
    bootstrap_indices_sha256: SHA256
    cell_statistics_artifact_id: StrictIdentifier
    cell_statistics_artifact: ArtifactRef
    paired_contrasts_artifact_id: StrictIdentifier
    paired_contrasts_artifact: ArtifactRef

    @field_validator("cell_statistics_artifact", "paired_contrasts_artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        _require_nonblank_path(value.path)
        return value

    @model_validator(mode="after")
    def _distinct_artifacts(self) -> Task13StatisticsReceiptV1:
        if self.cell_statistics_artifact_id != TASK13_CELL_STATISTICS_ARTIFACT_ID:
            raise ValueError("cell_statistics_artifact_id must be 'cell_statistics'")
        if self.cell_statistics_artifact.path != TASK13_CELL_STATISTICS_ARTIFACT_PATH:
            raise ValueError("cell_statistics_artifact.path must be 'cell_statistics.jsonl'")
        if self.paired_contrasts_artifact_id != TASK13_PAIRED_CONTRASTS_ARTIFACT_ID:
            raise ValueError("paired_contrasts_artifact_id must be 'paired_contrasts'")
        if self.paired_contrasts_artifact.path != TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH:
            raise ValueError("paired_contrasts_artifact.path must be 'paired_contrasts.jsonl'")
        if self.cell_statistics_artifact_id == self.paired_contrasts_artifact_id:
            raise ValueError("receipt artifact IDs must be distinct")
        if self.cell_statistics_artifact.path == self.paired_contrasts_artifact.path:
            raise ValueError("receipt artifact paths must be distinct")
        return self


class Task13DenominatorV1(ImmutableContractModel):
    """Frozen binding between task observations and independent core means."""

    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    task_count: Literal[80]
    semantic_core_count: Literal[20]
    tasks_per_core: Literal[4]

    @model_validator(mode="after")
    def _balanced_partition(self) -> Task13DenominatorV1:
        if self.task_count != self.semantic_core_count * self.tasks_per_core:
            raise ValueError("task_count must equal semantic_core_count * tasks_per_core")
        return self


TASK13_CASE_CATEGORIES = (
    "correct",
    "stale_copied",
    "answer_parse_invalid",
    "other_wrong",
)
CaseCategory = Literal["correct", "stale_copied", "answer_parse_invalid", "other_wrong"]


def task13_case_category_order_v1(category: str) -> int:
    try:
        return TASK13_CASE_CATEGORIES.index(category)
    except ValueError as exc:
        raise ValueError(f"unknown Task 13 case category: {category!r}") from exc


class Task13RunCaseCoverageV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    correct_case_id: StrictIdentifier | None = None
    stale_copied_case_id: StrictIdentifier | None = None
    answer_parse_invalid_case_id: StrictIdentifier | None = None
    other_wrong_case_id: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _at_least_one_unique_case(self) -> Task13RunCaseCoverageV1:
        case_ids = (
            self.correct_case_id,
            self.stale_copied_case_id,
            self.answer_parse_invalid_case_id,
            self.other_wrong_case_id,
        )
        selected = tuple(case_id for case_id in case_ids if case_id is not None)
        if not selected:
            raise ValueError("each run must have at least one selected case")
        if len(selected) != len(set(selected)):
            raise ValueError("non-null case IDs must be unique within each run")
        return self


class Task13CaseSelectorV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    selector_id: StrictIdentifier
    run_id: StrictIdentifier
    coverage: Task13RunCaseCoverageV1

    @model_validator(mode="after")
    def _coverage_run_matches(self) -> Task13CaseSelectorV1:
        if self.coverage.run_id != self.run_id:
            raise ValueError("selector coverage run_id must match selector run_id")
        return self


class Task13CaseBindingV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    case_id: StrictIdentifier
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    category: CaseCategory
    answer_model_slot: Task13AnswerModelSlot
    k: Task13RetrievalK

    @model_validator(mode="after")
    def _derived_case_id(self) -> Task13CaseBindingV1:
        if self.case_id != task13_case_id_v1(self.run_id, self.task_id, self.category):
            raise ValueError("case_id must equal the derived Task 13 case ID")
        return self


class Task13TaskProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    family: StrictIdentifier
    difficulty: StrictIdentifier
    metadata: FrozenJsonObjectV3
    source: FrozenJsonObjectV3
    target_objects: tuple[FrozenJsonObjectV3, ...]
    queries: tuple[FrozenJsonObjectV3, ...]
    gold_actions: tuple[FrozenJsonObjectV3, ...]


class Task13TimelineProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    redacted: bool = Field(strict=True)
    items: tuple[FrozenJsonObjectV3, ...]

    @model_validator(mode="after")
    def _redaction_policy(self) -> Task13TimelineProjectionV1:
        if self.redacted:
            forbidden = _contains_forbidden_field(self.items)
            if forbidden is not None:
                raise ValueError(
                    f"redacted timeline contains forbidden field: {forbidden}"
                )
            violation = _redacted_timeline_violation(self.items)
            if violation is not None:
                raise ValueError(
                    f"redacted timeline contains non-allowlisted field: {violation}"
                )
        return self


class Task13RunProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    category: CaseCategory
    completion_status: StrictIdentifier
    parsed_actions: tuple[FrozenJsonObjectV3, ...]
    memory_snapshots: tuple[FrozenJsonObjectV3, ...]
    final_state: FrozenJsonObjectV3 | None = None
    system_events: tuple[FrozenJsonObjectV3, ...]
    provenance: FrozenJsonObjectV3
    exceptions: tuple[FrozenJsonObjectV3, ...]

    @model_validator(mode="after")
    def _final_state_consistency(self) -> Task13RunProjectionV1:
        if not self.memory_snapshots:
            if self.final_state is not None:
                raise ValueError("final_state requires at least one memory snapshot")
            return self
        if self.final_state is None:
            raise ValueError("memory snapshots require a final_state projection")
        snapshot_states = tuple(
            snapshot.get("state_by_object")
            for snapshot in self.memory_snapshots
            if "state_by_object" in snapshot
        )
        if len(snapshot_states) != len(self.memory_snapshots):
            raise ValueError("memory snapshots must expose state_by_object")
        if not any(typed_json_equal(self.final_state, state) for state in snapshot_states):
            raise ValueError("final_state must equal one authenticated snapshot state")
        return self


def _validate_case_snapshot_chronology(
    timeline: Task13TimelineProjectionV1,
    run: Task13RunProjectionV1,
) -> None:
    if not run.memory_snapshots:
        return

    anchors = tuple(snapshot.get("after_event_id") for snapshot in run.memory_snapshots)
    if len(anchors) == 1 and anchors[0] is None:
        selected_snapshot = run.memory_snapshots[0]
        if not typed_json_equal(run.final_state, selected_snapshot["state_by_object"]):
            raise ValueError("case final_state does not match chronology-resolved snapshot")
        return

    if any(anchor is None for anchor in anchors):
        raise ValueError("case snapshots must not mix anchored and unanchored snapshots")
    if any(type(anchor) is not str for anchor in anchors):
        raise ValueError("case snapshots contain unknown anchors")
    if len(anchors) != len(set(anchors)):
        raise ValueError("case snapshots contain duplicate anchors")

    event_positions: dict[str, int] = {}
    sequence_indices: set[int] = set()
    for item in timeline.items:
        event_id = item.get("event_id")
        sequence_index = item.get("sequence_index")
        if type(event_id) is not str or not event_id:
            raise ValueError("anchored case timeline event_id must be a non-empty string")
        if type(sequence_index) is not int:
            raise ValueError("anchored case timeline sequence_index must be an integer")
        if event_id in event_positions:
            raise ValueError("case timeline contains duplicate event IDs")
        if sequence_index in sequence_indices:
            raise ValueError("case timeline contains duplicate sequence indices")
        event_positions[event_id] = sequence_index
        sequence_indices.add(sequence_index)

    unknown = tuple(anchor for anchor in anchors if anchor not in event_positions)
    if unknown:
        raise ValueError("case snapshots contain unknown anchors")
    selected_snapshot = max(
        run.memory_snapshots,
        key=lambda snapshot: event_positions[snapshot["after_event_id"]],
    )
    if not typed_json_equal(run.final_state, selected_snapshot["state_by_object"]):
        raise ValueError("case final_state does not match chronology-resolved snapshot")


class Task13ScoreProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    category: CaseCategory
    metric_layers: FrozenJsonObjectV3
    support: FrozenJsonObjectV3
    failure_flags: tuple[StrictIdentifier, ...]
    primary_failure: StrictIdentifier | None = None


class Task13RetrievalProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    category: CaseCategory
    available: bool = Field(strict=True)
    items: tuple[FrozenJsonObjectV3, ...]


class Task13AnswerProjectionV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    category: CaseCategory
    available: bool = Field(strict=True)
    items: tuple[FrozenJsonObjectV3, ...]


class Task13CaseRecordV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    case_id: StrictIdentifier
    category: CaseCategory
    run_id: StrictIdentifier
    task_id: StrictIdentifier
    semantic_core_id: StrictIdentifier
    answer_model_slot: Task13AnswerModelSlot
    k: Task13RetrievalK
    task_artifact_sha256: SHA256
    task_manifest_sha256: SHA256
    run_manifest_sha256: SHA256
    score_artifact_sha256: SHA256
    matrix_summary_sha256: SHA256
    task: Task13TaskProjectionV1
    timeline: Task13TimelineProjectionV1
    run: Task13RunProjectionV1
    score: Task13ScoreProjectionV1
    retrieval: Task13RetrievalProjectionV1
    answer: Task13AnswerProjectionV1

    @model_validator(mode="after")
    def _projection_identity_matches(self) -> Task13CaseRecordV1:
        if self.case_id != task13_case_id_v1(self.run_id, self.task_id, self.category):
            raise ValueError("case_id does not match its run, task, and category")
        if self.task.task_id != self.task_id:
            raise ValueError("task projection task_id must match the case task_id")
        if self.task.semantic_core_id != self.semantic_core_id:
            raise ValueError("task projection semantic_core_id must match the case semantic_core_id")
        source = self.task.source
        expected_source_fields = (
            _TASK13_SAFE_SOURCE_FIELDS
            if self.timeline.redacted
            else _TASK13_PUBLIC_SOURCE_FIELDS
        )
        if set(source) != expected_source_fields:
            raise ValueError("task source projection must use its exact redaction field set")
        if source.get("redacted") is not self.timeline.redacted:
            raise ValueError("task source and timeline redaction flags must agree")
        if self.timeline.redacted:
            forbidden = _contains_forbidden_field(source)
            if forbidden is not None:
                raise ValueError(f"task source contains forbidden field: {forbidden}")
            nested = _source_nested_container(source)
            if nested is not None:
                raise ValueError(f"task source safe fields must be scalar: {nested}")
            if source.get("source_uri") is not None:
                raise ValueError("redacted task source must not expose source_uri")
        else:
            _validate_public_source_projection(source)
        _validate_case_snapshot_chronology(self.timeline, self.run)
        for projection in (self.run, self.score, self.retrieval, self.answer):
            if projection.run_id != self.run_id:
                raise ValueError("projection run_id must match the case run_id")
            if projection.task_id != self.task_id:
                raise ValueError("projection task_id must match the case task_id")
            if projection.semantic_core_id != self.semantic_core_id:
                raise ValueError("projection semantic_core_id must match the case semantic_core_id")
            if projection.category != self.category:
                raise ValueError("projection category must match the case category")
        return self


class Task13CaseIndexV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    cases_artifact: ArtifactRef
    record_count: StrictPositiveInt
    task_artifact_sha256: SHA256
    task_manifest_sha256: SHA256
    matrix_summary_sha256: SHA256
    case_bindings: tuple[Task13CaseBindingV1, ...]
    coverage: tuple[Task13RunCaseCoverageV1, ...]
    run_sources: tuple[Task13RunSourceV1, ...]
    source_bindings: tuple[Task13ArtifactBindingV1, ...] = ()

    @field_validator("case_bindings")
    @classmethod
    def _unique_case_bindings(
        cls, value: tuple[Task13CaseBindingV1, ...]
    ) -> tuple[Task13CaseBindingV1, ...]:
        case_ids = tuple(binding.case_id for binding in value)
        if not case_ids:
            raise ValueError("case_bindings must be non-empty")
        _require_unique(case_ids, "case_bindings.case_id")
        return value

    @field_validator("coverage", "run_sources")
    @classmethod
    def _exact_18_rows(cls, value: tuple[Any, ...], info) -> tuple[Any, ...]:
        if len(value) != 18:
            raise ValueError(f"{info.field_name} must contain exactly 18 rows")
        return value

    @field_validator("cases_artifact")
    @classmethod
    def _artifact_path(cls, value: ArtifactRef) -> ArtifactRef:
        if value.path != "cases.jsonl":
            raise ValueError("cases_artifact.path must be 'cases.jsonl'")
        return value

    @model_validator(mode="after")
    def _validate_case_coverage(self) -> Task13CaseIndexV1:
        if self.record_count != len(self.case_bindings):
            raise ValueError("record_count must equal the number of case_bindings")
        if any(binding.answer_model_slot not in TASK13_SLOTS for binding in self.case_bindings):
            raise ValueError("case bindings must use a frozen Task 13 answer-model slot")
        if any(binding.k not in TASK13_K_VALUES for binding in self.case_bindings):
            raise ValueError("case bindings must use a frozen Task 13 retrieval k")
        run_task_keys = tuple((binding.run_id, binding.task_id) for binding in self.case_bindings)
        if len(run_task_keys) != len(set(run_task_keys)):
            raise ValueError("task_id must be unique within each run")
        run_ids = tuple(source.run_id for source in self.run_sources)
        coverage_run_ids = tuple(row.run_id for row in self.coverage)
        if len(set(run_ids)) != 18:
            raise ValueError("run_sources must contain unique run IDs")
        if run_ids != coverage_run_ids:
            raise ValueError("coverage and run_sources must use identical ordered run IDs")
        if run_ids != tuple(sorted(run_ids, key=lambda run_id: run_id.encode("utf-8"))):
            raise ValueError("run_sources must use canonical UTF-8 run-ID order")

        source_by_id = {source.run_id: source for source in self.run_sources}
        for binding in self.case_bindings:
            source = source_by_id.get(binding.run_id)
            if source is None:
                raise ValueError("case_bindings must reference a known run source")
            if (
                binding.answer_model_slot != source.answer_model_slot
                or binding.k != source.k
            ):
                raise ValueError(
                    "case binding slot/k must match its run-source coordinates"
                )

        source_order = {run_id: index for index, run_id in enumerate(run_ids)}
        try:
            expected_binding_order = tuple(
                sorted(
                    self.case_bindings,
                    key=lambda binding: (
                        source_order[binding.run_id],
                        task13_case_category_order_v1(binding.category),
                        binding.case_id.encode("utf-8"),
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError("case_bindings must reference a known run source") from exc
        if self.case_bindings != expected_binding_order:
            raise ValueError(
                "case_bindings must use canonical run-source, category, and UTF-8 case-ID order"
            )

        bindings = {binding.case_id: binding for binding in self.case_bindings}
        covered_case_ids: list[str] = []
        category_fields = (
            ("correct_case_id", "correct"),
            ("stale_copied_case_id", "stale_copied"),
            ("answer_parse_invalid_case_id", "answer_parse_invalid"),
            ("other_wrong_case_id", "other_wrong"),
        )
        for row in self.coverage:
            for field_name, category in category_fields:
                case_id = getattr(row, field_name)
                if case_id is None:
                    continue
                binding = bindings.get(case_id)
                if binding is None:
                    raise ValueError("coverage must reference known case bindings")
                if binding.run_id != row.run_id or binding.category != category:
                    raise ValueError("coverage must match each case binding run and category")
                covered_case_ids.append(case_id)
        if len(covered_case_ids) != len(set(covered_case_ids)):
            raise ValueError("each case ID must occur exactly once in coverage")
        if set(covered_case_ids) != set(bindings):
            raise ValueError("coverage union must equal the case binding IDs")
        _validate_unique_bindings(self.source_bindings)
        return self


class Task13ClaimLedgerRecordV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    claim_id: StrictIdentifier
    kind: Literal["direct_cell", "paired_contrast"]
    direction: Literal["self", "left_minus_right"]
    slot: Task13AnswerModelSlot
    cell_or_contrast: StrictIdentifier
    metric_path: StrictIdentifier
    slice_payload: FrozenJsonObjectV3
    denominator: Task13DenominatorV1
    status: Task13StatisticStatus
    interval: Task13IntervalV1
    run_sources: tuple[Task13RunSourceV1, ...]
    statistics_receipt_sha256: SHA256
    case_ids: tuple[StrictIdentifier, ...]
    case_index_sha256: SHA256

    @field_validator("metric_path")
    @classmethod
    def _core_metric(cls, value: str) -> str:
        if value not in TASK13_METRIC_PATHS:
            raise ValueError("metric_path must belong to the frozen TASK13_METRIC_PATHS")
        return value

    @field_validator("run_sources", "case_ids")
    @classmethod
    def _nonempty_unique_sequences(cls, value: tuple[Any, ...], info) -> tuple[Any, ...]:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        if info.field_name == "case_ids":
            return _require_unique(value, info.field_name)
        return value

    @model_validator(mode="after")
    def _ledger_consistency(self) -> Task13ClaimLedgerRecordV1:
        if self.status != self.interval.status:
            raise ValueError("claim status must equal the bound interval status")
        if self.kind == "direct_cell" and self.direction != "self":
            raise ValueError("direct_cell claims require direction='self'")
        if self.kind == "paired_contrast" and self.direction != "left_minus_right":
            raise ValueError("paired_contrast claims require direction='left_minus_right'")
        expected_sources = 1 if self.kind == "direct_cell" else 2
        if len(self.run_sources) != expected_sources:
            raise ValueError(
                f"{self.kind} claims must bind exactly {expected_sources} typed run sources"
            )

        slice_payload = dict(self.slice_payload)
        if self.kind == "direct_cell":
            if set(slice_payload) != {"cell_id", "k"}:
                raise ValueError("direct_cell claim slice must contain exactly cell_id and k")
            cell_id = slice_payload.get("cell_id")
            slice_k = slice_payload.get("k")
            if type(cell_id) is not str or not cell_id.strip():
                raise ValueError("direct_cell claim slice cell_id must be a nonblank string")
            if cell_id != self.cell_or_contrast:
                raise ValueError("direct_cell claim slice cell_id must match cell_or_contrast")
        else:
            if set(slice_payload) != {"k", "left_cell_id", "right_cell_id"}:
                raise ValueError(
                    "paired_contrast claim slice must contain exactly k, left_cell_id, and right_cell_id"
                )
            left_cell_id = slice_payload.get("left_cell_id")
            right_cell_id = slice_payload.get("right_cell_id")
            slice_k = slice_payload.get("k")
            if (
                type(left_cell_id) is not str
                or not left_cell_id.strip()
                or type(right_cell_id) is not str
                or not right_cell_id.strip()
            ):
                raise ValueError("paired_contrast claim slice cell IDs must be nonblank strings")
            if left_cell_id == right_cell_id:
                raise ValueError("paired_contrast claim slice cell IDs must differ")
            expected_contrast_id = task13_contrast_id_v1(
                self.slot,
                slice_k,
                left_cell_id,
                right_cell_id,
                self.metric_path,
            ) if type(slice_k) is int and slice_k in TASK13_K_VALUES else None
            if self.cell_or_contrast != expected_contrast_id:
                raise ValueError(
                    "paired_contrast claim slice cells must reconstruct cell_or_contrast"
                )
        if type(slice_k) is not int or slice_k not in TASK13_K_VALUES:
            raise ValueError("claim slice k must be a strict frozen Task 13 retrieval k")

        for source in self.run_sources:
            if source.answer_model_slot != self.slot or source.k != slice_k:
                raise ValueError("claim run-source coordinates must match claim slot and slice k")
        if self.kind == "paired_contrast":
            left, right = self.run_sources
            if left.run_id == right.run_id:
                raise ValueError("paired_contrast sources must use different run IDs")
            if left == right:
                raise ValueError("paired_contrast sources must be different complete records")
        expected_claim_id = task13_claim_id_v1(
            self.kind,
            self.slot,
            self.cell_or_contrast,
            self.metric_path,
            self.slice_payload,
        )
        if self.claim_id != expected_claim_id:
            raise ValueError("claim_id must equal the derived Task 13 claim identity")
        return self


class Task13ArtifactIndexV1(ImmutableContractModel):
    schema_version: Literal[TASK13_CONTRACT_SCHEMA_VERSION] = TASK13_CONTRACT_SCHEMA_VERSION
    artifacts: tuple[Task13ArtifactBindingV1, ...]

    @model_validator(mode="after")
    def _exact_public_artifacts(self) -> Task13ArtifactIndexV1:
        _validate_unique_bindings(self.artifacts)
        if len(self.artifacts) != len(TASK13_ARTIFACT_PATHS):
            raise ValueError("Task 13 final artifact index must contain exactly seven artifacts")
        paths = tuple(binding.artifact.path for binding in self.artifacts)
        ids = tuple(binding.artifact_id for binding in self.artifacts)
        roles = tuple(binding.role for binding in self.artifacts)
        if paths != TASK13_ARTIFACT_PATHS or ids != TASK13_ARTIFACT_PATHS:
            raise ValueError("Task 13 artifact index must contain the ordered non-self publication set")
        if roles != TASK13_ARTIFACT_PATHS:
            raise ValueError("Task 13 artifact roles must be present and match artifact IDs")
        if any(binding.role != binding.artifact_id for binding in self.artifacts):
            raise ValueError("Task 13 artifact role must match its artifact ID")
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
    "TASK13_ARTIFACT_PATHS",
    "TASK13_CONTEXT_ANNOTATION",
    "TASK13_CONTEXT_CONDITIONS",
    "TASK13_CONTEXT_ORDER",
    "TASK13_CONTRAST_PAIRS",
    "TASK13_CASE_CATEGORIES",
    "TASK13_K_VALUES",
    "TASK13_SLOTS",
    "TASK13_CELL_STATISTICS_ARTIFACT_ID",
    "TASK13_CELL_STATISTICS_ARTIFACT_PATH",
    "TASK13_K",
    "TASK13_TASK_COUNT",
    "TASK13_SEMANTIC_CORE_COUNT",
    "TASK13_TASKS_PER_CORE",
    "TASK13_METRIC_PATHS",
    "TASK13_PAIRED_CONTRASTS_ARTIFACT_ID",
    "TASK13_PAIRED_CONTRASTS_ARTIFACT_PATH",
    "Task13AnswerModelSlot",
    "Task13RetrievalK",
    "Task13AnswerProjectionV1",
    "Task13ArtifactRole",
    "Task13ContrastIdentityPayloadV1",
    "Task13TaskIdentityPayloadV1",
    "CanonicalDecimal",
    "CaseCategory",
    "Task13ArtifactBindingV1",
    "Task13ArtifactIndexV1",
    "Task13BootstrapConfigV1",
    "Task13CaseBindingV1",
    "Task13CaseIndexV1",
    "Task13CaseRecordV1",
    "Task13CaseSelectorV1",
    "Task13CellStatisticV1",
    "Task13ClaimIdentityPayloadV1",
    "Task13ClaimLedgerRecordV1",
    "Task13DenominatorV1",
    "Task13IntervalV1",
    "Task13PairedContrastV1",
    "Task13RunCaseCoverageV1",
    "Task13RunProjectionV1",
    "Task13RunSourceV1",
    "Task13ScoreProjectionV1",
    "Task13StatisticStatus",
    "Task13StatisticsReceiptV1",
    "Task13TaskProjectionV1",
    "Task13TimelineProjectionV1",
    "Task13RetrievalProjectionV1",
    "canonical_decimal_string",
    "task13_case_category_order_v1",
    "task13_case_id_v1",
    "task13_claim_id_v1",
    "task13_contrast_id_v1",
    "task13_task_identity_sha256_v1",
]
