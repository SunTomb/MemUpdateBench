from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import StrictStr, field_validator, model_validator

from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel
from mub.vnext.contracts.enums import TaskFamily
from mub.vnext.contracts.score import METRIC_FIELD_PATHS, SCORE_LAYER_TYPES
from mub.vnext.version import METRIC_REGISTRY_VERSION, SCORER_VERSION

ALL_TASK_FAMILIES = "*"
RUNTIME_NULL_POLICY = "null_runtime_failed_exclude_from_aggregation"
RUNTIME_SCORE_AVAILABLE_POLICY = "score_available_normalized_artifacts"
CANONICAL_METRIC_PATHS = (
    "action_scores.attribute_accuracy",
    "action_scores.entity_accuracy",
    "action_scores.false_write_rate",
    "action_scores.full_action_exact_match",
    "action_scores.missed_write_rate",
    "action_scores.object_key_accuracy",
    "action_scores.operation_accuracy",
    "action_scores.value_accuracy",
    "action_scores.wrong_object_write_rate",
    "answer_scores.answer_state_consistency",
    "answer_scores.distractor_copied",
    "answer_scores.exact_match",
    "answer_scores.gold_retrieved_wrong_answer",
    "answer_scores.normalized_match",
    "answer_scores.stale_copied",
    "answer_scores.structured_field_accuracy",
    "answer_scores.token_f1",
    "audit_scores.action_trace_available",
    "audit_scores.manifest_completeness",
    "audit_scores.retrieval_trace_available",
    "audit_scores.source_provenance_coverage",
    "audit_scores.state_export_available",
    "protocol_scores.action_parse_valid",
    "protocol_scores.answer_parse_valid",
    "protocol_scores.execution_success_rate",
    "protocol_scores.fallback_rate",
    "protocol_scores.unsupported_operation_rate",
    "retrieval_scores.current_mrr",
    "retrieval_scores.current_recall_at_k",
    "retrieval_scores.distractor_exposure_rate",
    "retrieval_scores.stale_count_in_context",
    "retrieval_scores.stale_exposure_rate",
    "state_scores.collateral_corruption_rate",
    "state_scores.expected_absence_accuracy",
    "state_scores.final_state_accuracy",
    "state_scores.state_f1",
    "state_scores.state_precision",
    "state_scores.state_recall",
    "state_scores.state_resolve_rate",
    "store_scores.compaction_ratio",
    "store_scores.duplicate_current_count",
    "store_scores.final_memory_size",
    "store_scores.obsolete_version_count",
    "store_scores.stale_conflicting_value_count",
    "store_scores.write_amplification",
    "system_scores.answer_latency_ms",
    "system_scores.api_cost",
    "system_scores.error_rate",
    "system_scores.ingest_latency_ms",
    "system_scores.retrieval_latency_ms",
    "system_scores.token_usage",
)
_FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_scores\.[a-z][a-z0-9_]*$")
_ALIAS_PATTERN = re.compile(r"^legacy_[a-z0-9_]+(?:\.[a-z][a-z0-9_]*)+$")
_FAMILY_NAME_TO_VALUE = {family.name: family.value for family in TaskFamily}
_CANONICAL_FAMILIES = frozenset(family.value for family in TaskFamily)
_CAPABILITY_FIELDS = frozenset(AdapterCapabilities.model_fields) - {"extractor_version"}

_STATE_CONTENT_PATHS = (
    ("exports_object_keys", "exports_values"),
    ("requires_evaluation_extractor",),
)
_OBSOLETE_ENTRY_PATHS = (
    ("exports_timestamps_or_order", "exports_object_keys"),
    ("exports_source_event_ids", "exports_object_keys"),
    ("exports_timestamps_or_order", "requires_evaluation_extractor"),
    ("exports_source_event_ids", "requires_evaluation_extractor"),
)
_VALUE_AND_ORDER_ENTRY_PATHS = (
    ("exports_timestamps_or_order", "exports_object_keys", "exports_values"),
    ("exports_source_event_ids", "exports_object_keys", "exports_values"),
    ("exports_timestamps_or_order", "requires_evaluation_extractor"),
    ("exports_source_event_ids", "requires_evaluation_extractor"),
)
ALTERNATIVE_CAPABILITY_PATHS = FrozenDict(
    {
        **{
            f"state_scores.{field}": _STATE_CONTENT_PATHS
            for field in SCORE_LAYER_TYPES["state_scores"].model_fields
        },
        "store_scores.obsolete_version_count": _OBSOLETE_ENTRY_PATHS,
        "store_scores.stale_conflicting_value_count": _VALUE_AND_ORDER_ENTRY_PATHS,
        "store_scores.duplicate_current_count": _VALUE_AND_ORDER_ENTRY_PATHS,
        "answer_scores.answer_state_consistency": _STATE_CONTENT_PATHS,
    }
)
EXTRACTOR_LINKAGE_METRIC_PATHS = frozenset(
    {
        *(f"state_scores.{field}" for field in SCORE_LAYER_TYPES["state_scores"].model_fields),
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
        "retrieval_scores.stale_exposure_rate",
        "retrieval_scores.stale_count_in_context",
        "answer_scores.distractor_copied",
        "answer_scores.gold_retrieved_wrong_answer",
        "answer_scores.answer_state_consistency",
    }
)

for _metric_path, _alternatives in ALTERNATIVE_CAPABILITY_PATHS.items():
    if _metric_path not in CANONICAL_METRIC_PATHS:
        raise RuntimeError(f"unknown alternative-capability metric path: {_metric_path}")
    if not _alternatives or any(not alternative for alternative in _alternatives):
        raise RuntimeError(f"empty alternative capability path for {_metric_path}")
    unknown = {
        capability
        for alternative in _alternatives
        for capability in alternative
        if capability not in _CAPABILITY_FIELDS
    }
    if unknown:
        raise RuntimeError(f"unknown alternative capabilities: {sorted(unknown)}")
if not EXTRACTOR_LINKAGE_METRIC_PATHS <= set(CANONICAL_METRIC_PATHS):
    raise RuntimeError("extractor-linkage paths must be canonical metrics")


class MetricDefinition(ImmutableContractModel):
    field_name: StrictStr
    layer: StrictStr
    value_type: Literal["bool", "rate", "count", "nonnegative_float"]
    numerator_definition: StrictStr
    denominator_definition: StrictStr
    aggregation_rule: StrictStr
    applicable_task_families: tuple[StrictStr, ...]
    required_adapter_capabilities: tuple[StrictStr, ...] = ()
    unsupported_value_policy: StrictStr
    runtime_failure_policy: StrictStr
    legacy_aliases: tuple[StrictStr, ...] = ()
    introduced_in_scorer_version: StrictStr

    @field_validator(
        "applicable_task_families",
        "required_adapter_capabilities",
        "legacy_aliases",
        mode="before",
    )
    @classmethod
    def _canonicalize_semantic_tuple(cls, value, info):
        if type(value) not in {tuple, list, set, frozenset}:
            raise ValueError(f"{info.field_name} must be a tuple-like semantic set")
        supplied = tuple(value)
        if any(type(item) is not str for item in supplied):
            raise ValueError(f"{info.field_name} values must be exact built-in strings")
        if info.field_name == "applicable_task_families":
            normalized = tuple(
                _FAMILY_NAME_TO_VALUE.get(item, item) for item in supplied
            )
        else:
            normalized = supplied
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} values must be unique")
        return tuple(sorted(normalized))

    @field_validator(
        "field_name",
        "layer",
        "numerator_definition",
        "denominator_definition",
        "aggregation_rule",
        "unsupported_value_policy",
        "runtime_failure_policy",
        "introduced_in_scorer_version",
    )
    @classmethod
    def _reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metric definition strings must not be blank")
        return value

    @field_validator(
        "applicable_task_families",
        "required_adapter_capabilities",
        "legacy_aliases",
    )
    @classmethod
    def _reject_duplicate_or_blank_tuple_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("metric definition tuple values must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("metric definition tuple values must be unique")
        return values

    @model_validator(mode="after")
    def _validate_canonical_references(self):
        if not _FIELD_PATTERN.fullmatch(self.field_name):
            raise ValueError("field_name must be a canonical fully-qualified metric path")
        if self.layer not in SCORE_LAYER_TYPES:
            raise ValueError(f"unknown score layer: {self.layer}")
        if not self.field_name.startswith(f"{self.layer}."):
            raise ValueError("field_name layer prefix must match layer")
        leaf = self.field_name.split(".", 1)[1]
        if leaf not in SCORE_LAYER_TYPES[self.layer].model_fields:
            raise ValueError(f"unknown score field: {self.field_name}")
        if not self.applicable_task_families:
            raise ValueError("applicable_task_families must not be empty")
        families = set(self.applicable_task_families)
        if ALL_TASK_FAMILIES in families and families != {ALL_TASK_FAMILIES}:
            raise ValueError("all-family marker cannot be combined with named families")
        unknown_families = families - _CANONICAL_FAMILIES - {ALL_TASK_FAMILIES}
        if unknown_families:
            raise ValueError(f"unknown task families: {sorted(unknown_families)}")
        unknown_capabilities = set(self.required_adapter_capabilities) - _CAPABILITY_FIELDS
        if unknown_capabilities:
            raise ValueError(f"unknown adapter capabilities: {sorted(unknown_capabilities)}")
        malformed_aliases = [
            alias for alias in self.legacy_aliases if not _ALIAS_PATTERN.fullmatch(alias)
        ]
        if malformed_aliases:
            raise ValueError(f"malformed legacy aliases: {sorted(malformed_aliases)}")
        if self.introduced_in_scorer_version != SCORER_VERSION:
            raise ValueError("current canonical metrics must use the current scorer version")
        return self


_ALL = (ALL_TASK_FAMILIES,)
_STALE_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT.value,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value,
    TaskFamily.CURRENT_HISTORICAL_QUERY.value,
    TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS.value,
    TaskFamily.REALISTIC_SOURCE_UPDATE.value,
)
_DELETE_FAMILY = (TaskFamily.DELETION_FORGETTING.value,)

_ACTION_CAPS = ("exports_action_trace",)
_LEVEL_TWO_ENTRY_CAPS = ("exports_entries", "supports_isolated_reset")
_RETRIEVAL_CAPS = ("exports_retrieval_ids",)
_RANK_CAPS = ("exports_retrieval_ids", "exports_retrieval_scores")
_ANSWER_CAPS: tuple[str, ...] = ()

_BOOL_FIELDS = {
    "protocol_scores.action_parse_valid",
    "protocol_scores.answer_parse_valid",
    "audit_scores.action_trace_available",
    "audit_scores.state_export_available",
    "audit_scores.retrieval_trace_available",
}
_COUNT_FIELDS = {
    "store_scores.obsolete_version_count",
    "store_scores.stale_conflicting_value_count",
    "store_scores.duplicate_current_count",
    "store_scores.final_memory_size",
    "retrieval_scores.stale_count_in_context",
    "system_scores.token_usage",
}
_NONNEGATIVE_FLOAT_FIELDS = {
    "store_scores.compaction_ratio",
    "store_scores.write_amplification",
    "system_scores.ingest_latency_ms",
    "system_scores.retrieval_latency_ms",
    "system_scores.answer_latency_ms",
    "system_scores.api_cost",
}

_CAPABILITIES_BY_PATH = {
    **{
        f"action_scores.{field}": _ACTION_CAPS
        for field in SCORE_LAYER_TYPES["action_scores"].model_fields
    },
    **{
        f"state_scores.{field}": _LEVEL_TWO_ENTRY_CAPS
        for field in SCORE_LAYER_TYPES["state_scores"].model_fields
    },
    "store_scores.obsolete_version_count": _LEVEL_TWO_ENTRY_CAPS,
    "store_scores.stale_conflicting_value_count": _LEVEL_TWO_ENTRY_CAPS,
    "store_scores.duplicate_current_count": _LEVEL_TWO_ENTRY_CAPS,
    "store_scores.final_memory_size": ("exports_entries",),
    "store_scores.compaction_ratio": ("exports_entries",),
    "store_scores.write_amplification": _ACTION_CAPS,
    "retrieval_scores.current_recall_at_k": _RETRIEVAL_CAPS,
    "retrieval_scores.current_mrr": _RANK_CAPS,
    "retrieval_scores.stale_exposure_rate": _RETRIEVAL_CAPS,
    "retrieval_scores.stale_count_in_context": _RETRIEVAL_CAPS,
    "retrieval_scores.distractor_exposure_rate": _RETRIEVAL_CAPS,
    **{
        f"answer_scores.{field}": _ANSWER_CAPS
        for field in SCORE_LAYER_TYPES["answer_scores"].model_fields
    },
    "answer_scores.distractor_copied": _RETRIEVAL_CAPS,
    "answer_scores.gold_retrieved_wrong_answer": _RETRIEVAL_CAPS,
    "answer_scores.answer_state_consistency": _LEVEL_TWO_ENTRY_CAPS,
    "system_scores.ingest_latency_ms": ("reports_latency",),
    "system_scores.retrieval_latency_ms": ("reports_latency",),
    "system_scores.answer_latency_ms": ("reports_latency",),
    "system_scores.token_usage": ("reports_token_usage",),
    "system_scores.api_cost": ("reports_cost",),
    "audit_scores.source_provenance_coverage": ("exports_source_event_ids",),
}

_FAMILIES_BY_PATH = {
    "state_scores.expected_absence_accuracy": _DELETE_FAMILY,
    "store_scores.obsolete_version_count": _STALE_FAMILIES,
    "store_scores.stale_conflicting_value_count": _STALE_FAMILIES,
    "retrieval_scores.stale_exposure_rate": _STALE_FAMILIES,
    "retrieval_scores.stale_count_in_context": _STALE_FAMILIES,
    "answer_scores.stale_copied": _STALE_FAMILIES,
}

_ALIASES_BY_PATH: dict[str, tuple[str, ...]] = {}


_SEMANTICS_BY_PATH = {
    "protocol_scores.action_parse_valid": (
        "One when every expected manager action has a valid normalized parse; zero otherwise.",
        "One task with at least one expected manager action.",
    ),
    "protocol_scores.answer_parse_valid": (
        "One when every expected answer has a valid normalized parse; zero otherwise.",
        "One task with at least one answer query.",
    ),
    "protocol_scores.execution_success_rate": (
        "Count of normalized manager actions whose execution status is successful.",
        "Count of expected manager actions with execution records.",
    ),
    "protocol_scores.unsupported_operation_rate": (
        "Count of normalized manager actions marked as unsupported operations.",
        "Count of expected manager actions with execution records.",
    ),
    "protocol_scores.fallback_rate": (
        "Count of normalized manager actions that used fallback execution.",
        "Count of expected manager actions with execution records.",
    ),
    "action_scores.operation_accuracy": (
        "Count of predicted operations exactly equal to their gold operations.",
        "Count of canonical gold actions.",
    ),
    "action_scores.full_action_exact_match": (
        "Count of predictions matching gold operation, object key, and typed value.",
        "Count of canonical gold actions.",
    ),
    "action_scores.object_key_accuracy": (
        "Count of predicted canonical object keys exactly equal to gold targets.",
        "Count of canonical gold actions.",
    ),
    "action_scores.entity_accuracy": (
        "Count of predicted target entities exactly equal to gold entities.",
        "Count of canonical gold actions.",
    ),
    "action_scores.attribute_accuracy": (
        "Count of predicted target attributes exactly equal to gold attributes.",
        "Count of canonical gold actions.",
    ),
    "action_scores.value_accuracy": (
        "Count of predicted typed values exactly equal to gold action values.",
        "Count of canonical gold actions.",
    ),
    "action_scores.false_write_rate": (
        "Count of gold NOOP actions for which a write or delete was predicted.",
        "Count of canonical gold actions under the task-level rate convention.",
    ),
    "action_scores.missed_write_rate": (
        "Count of gold write or delete actions with no corresponding predicted mutation.",
        "Count of canonical gold actions under the task-level rate convention.",
    ),
    "action_scores.wrong_object_write_rate": (
        "Count of predicted mutations targeting a non-gold canonical object key.",
        "Count of canonical gold actions under the task-level rate convention.",
    ),
    "state_scores.final_state_accuracy": (
        "One when predicted and gold final-state mappings match in keys and typed values.",
        "One task with an observable final-state artifact.",
    ),
    "state_scores.state_precision": (
        "Count of predicted object/value pairs exactly present in the gold final state.",
        "Count of predicted final-state object/value pairs.",
    ),
    "state_scores.state_recall": (
        "Count of gold object/value pairs exactly recovered in predicted final state.",
        "Count of gold final-state object/value pairs.",
    ),
    "state_scores.state_f1": (
        "Harmonic mean numerator derived from canonical state precision and recall.",
        "Canonical precision-plus-recall normalization for one task.",
    ),
    "state_scores.state_resolve_rate": (
        "Count of expected-present objects resolved to their exact current typed values.",
        "Count of expected-present canonical objects.",
    ),
    "state_scores.collateral_corruption_rate": (
        "Count of predicted state objects outside the canonical gold final state.",
        "Count of predicted final-state objects, with empty-state convention defined by scorer.",
    ),
    "state_scores.expected_absence_accuracy": (
        "Count of expected-absent canonical objects absent from predicted final state.",
        "Count of expected-absent canonical objects.",
    ),
    "store_scores.obsolete_version_count": (
        "Count of retained target-object entries obsolete by normalized version or event order.",
        "One task with order-resolved exported target-object entries.",
    ),
    "store_scores.stale_conflicting_value_count": (
        "Count of obsolete target-object entries whose typed values conflict with current gold.",
        "One task with order-resolved exported target-object entries.",
    ),
    "store_scores.duplicate_current_count": (
        "Count of additional latest entries duplicating the current typed value beyond one.",
        "One task with order-resolved exported target-object entries.",
    ),
    "store_scores.final_memory_size": (
        "Number of entries declared in the final normalized memory snapshot.",
        "One task with an exported final memory snapshot.",
    ),
    "store_scores.compaction_ratio": (
        "Final normalized memory size.",
        "Count of canonical non-NOOP gold mutations, lower-bounded at one.",
    ),
    "store_scores.write_amplification": (
        "Count of predicted ADD, UPDATE, or DELETE mutations.",
        "Count of canonical non-NOOP gold mutations, lower-bounded at one.",
    ),
    "retrieval_scores.current_recall_at_k": (
        "Count of current-state queries whose current gold evidence appears in retrieved context.",
        "Count of current-state queries with observable retrieval linkage.",
    ),
    "retrieval_scores.current_mrr": (
        "Sum of reciprocal ranks of the first retrieved current-gold entry per query.",
        "Count of current-state queries with observable ranks and retrieval linkage.",
    ),
    "retrieval_scores.stale_exposure_rate": (
        "Count of current-state query contexts containing at least one obsolete same-object entry.",
        "Count of current-state queries with observable retrieval linkage or stale annotation.",
    ),
    "retrieval_scores.stale_count_in_context": (
        "Number of obsolete same-object entries exposed across current-state query contexts.",
        "One task with observable retrieval linkage for all current-state queries.",
    ),
    "retrieval_scores.distractor_exposure_rate": (
        "Count of current-state query contexts annotated as containing distractor evidence.",
        "Count of current-state queries with distractor annotations.",
    ),
    "answer_scores.exact_match": (
        "Count of parsed answers with exact typed equality to canonical gold answers.",
        "Count of current-state queries with parsed answer artifacts.",
    ),
    "answer_scores.normalized_match": (
        "Count of parsed answers matching a gold or accepted answer after versioned normalization.",
        "Count of current-state queries with parsed answer artifacts.",
    ),
    "answer_scores.token_f1": (
        "Sum of duplicate-aware token overlap precision/recall harmonic means.",
        "Count of current-state queries with parsed answer artifacts.",
    ),
    "answer_scores.structured_field_accuracy": (
        "Count-equivalent share of gold structured fields or positions with exact typed matches.",
        "Count of gold structured fields or maximum sequence positions per current-state answer.",
    ),
    "answer_scores.stale_copied": (
        "Count of parsed answers exactly copying a non-current value from canonical version history.",
        "Count of current-state queries with parsed answer artifacts.",
    ),
    "answer_scores.distractor_copied": (
        "Count of parsed answers copying annotated retrieved distractor values.",
        "Count of current-state queries with parsed answers and distractor retrieval evidence.",
    ),
    "answer_scores.gold_retrieved_wrong_answer": (
        "Count of queries with current gold retrieved but normalized answer incorrect.",
        "Count of current-state queries with answer and current-retrieval evidence.",
    ),
    "answer_scores.answer_state_consistency": (
        "Count of parsed answers exactly equal to the normalized final-state target value.",
        "Count of current-state queries with answer and final-state artifacts.",
    ),
    "system_scores.ingest_latency_ms": (
        "Sum of reported normalized manager-action latencies in milliseconds.",
        "Count of manager actions with reported ingest latency.",
    ),
    "system_scores.retrieval_latency_ms": (
        "Sum of reported retrieval latencies in milliseconds.",
        "Count of normalized retrieval latency observations.",
    ),
    "system_scores.answer_latency_ms": (
        "Sum of reported answer latencies in milliseconds.",
        "Count of answer predictions with reported latency.",
    ),
    "system_scores.token_usage": (
        "Sum of all normalized token-usage counters across answer predictions.",
        "One task with at least one token-usage artifact.",
    ),
    "system_scores.api_cost": (
        "Sum of normalized API cost observations for the task.",
        "One task with at least one reported API cost artifact.",
    ),
    "system_scores.error_rate": (
        "One when the task has a runtime exception or failed/partial completion; zero otherwise.",
        "One available task run row.",
    ),
    "audit_scores.action_trace_available": (
        "One when declared action-trace capability and normalized action records are both present.",
        "One available task run row.",
    ),
    "audit_scores.state_export_available": (
        "One when declared state-export capability and a normalized memory snapshot are present.",
        "One available task run row.",
    ),
    "audit_scores.retrieval_trace_available": (
        "One when declared retrieval-trace capability and normalized retrieval records are present.",
        "One available task run row.",
    ),
    "audit_scores.source_provenance_coverage": (
        "Count of final-snapshot entries carrying at least one normalized source event ID.",
        "Count of entries in the final normalized memory snapshot.",
    ),
    "audit_scores.manifest_completeness": (
        "Count of required parser and extractor provenance identity fields present on the run row.",
        "Count of required parser and extractor provenance identity fields.",
    ),
}

if set(_SEMANTICS_BY_PATH) != set(CANONICAL_METRIC_PATHS):
    raise RuntimeError("metric semantics must explicitly cover every canonical metric path")

_RUNTIME_NULL_PATHS = frozenset(
    {
        *(f"action_scores.{field}" for field in SCORE_LAYER_TYPES["action_scores"].model_fields),
        *(f"state_scores.{field}" for field in SCORE_LAYER_TYPES["state_scores"].model_fields),
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
        "answer_scores.exact_match",
        "answer_scores.normalized_match",
        "answer_scores.token_f1",
        "answer_scores.structured_field_accuracy",
        "answer_scores.answer_state_consistency",
    }
)


def _value_type(path: str) -> str:
    if path in _BOOL_FIELDS:
        return "bool"
    if path in _COUNT_FIELDS:
        return "count"
    if path in _NONNEGATIVE_FLOAT_FIELDS:
        return "nonnegative_float"
    return "rate"


def _definition_payload(path: str) -> dict[str, object]:
    value_type = _value_type(path)
    numerator_definition, denominator_definition = _SEMANTICS_BY_PATH[path]
    aggregation = (
        "Sum and arithmetic mean over non-null task values."
        if value_type == "count"
        else "Arithmetic mean over non-null task values."
    )
    return {
        "field_name": path,
        "layer": path.split(".", 1)[0],
        "value_type": value_type,
        "numerator_definition": numerator_definition,
        "denominator_definition": denominator_definition,
        "aggregation_rule": aggregation,
        "applicable_task_families": _FAMILIES_BY_PATH.get(path, _ALL),
        "required_adapter_capabilities": _CAPABILITIES_BY_PATH.get(path, ()),
        "unsupported_value_policy": "Serialize null with support reason and exclude from aggregation.",
        "runtime_failure_policy": (
            RUNTIME_NULL_POLICY
            if path in _RUNTIME_NULL_PATHS
            else RUNTIME_SCORE_AVAILABLE_POLICY
        ),
        "legacy_aliases": _ALIASES_BY_PATH.get(path, ()),
        "introduced_in_scorer_version": SCORER_VERSION,
    }


def validate_metric_registry(
    registry: Mapping[str, object],
    *,
    require_complete: bool = True,
) -> FrozenDict:
    validated: dict[str, MetricDefinition] = {}
    aliases: dict[str, str] = {}
    for path, raw_definition in registry.items():
        if not isinstance(path, str):
            raise TypeError("metric registry keys must be strings")
        definition = MetricDefinition.model_validate(
            raw_definition.model_dump(mode="python")
            if isinstance(raw_definition, MetricDefinition)
            else raw_definition
        )
        if path != definition.field_name:
            raise ValueError("metric registry key must equal definition.field_name")
        validated[path] = definition
        for alias in definition.legacy_aliases:
            previous = aliases.get(alias)
            if previous is not None and previous != path:
                raise ValueError(f"ambiguous legacy alias {alias}: {previous}, {path}")
            aliases[alias] = path
    if require_complete:
        missing = METRIC_FIELD_PATHS - set(validated)
        extra = set(validated) - METRIC_FIELD_PATHS
        if missing or extra:
            raise ValueError(
                f"metric registry completeness failure: missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return FrozenDict((path, validated[path]) for path in sorted(validated))


METRIC_REGISTRY = validate_metric_registry(
    {path: _definition_payload(path) for path in CANONICAL_METRIC_PATHS}
)
LEGACY_ALIAS_TO_FIELD = FrozenDict(
    sorted(
        (alias, path)
        for path, definition in METRIC_REGISTRY.items()
        for alias in definition.legacy_aliases
    )
)


def metric_applies_to_family(definition: MetricDefinition, task_family: str) -> bool:
    families = definition.applicable_task_families
    return ALL_TASK_FAMILIES in families or task_family in families


def missing_capabilities(
    definition: MetricDefinition,
    capabilities: AdapterCapabilities,
) -> tuple[str, ...]:
    missing = [
        name
        for name in definition.required_adapter_capabilities
        if getattr(capabilities, name) is not True
    ]
    alternatives = ALTERNATIVE_CAPABILITY_PATHS.get(definition.field_name)
    if alternatives and not any(
        all(getattr(capabilities, name) is True for name in alternative)
        for alternative in alternatives
    ):
        rendered = "|".join("+".join(alternative) for alternative in alternatives)
        missing.append(f"one_of({rendered})")
    return tuple(missing)


__all__ = [
    "ALL_TASK_FAMILIES",
    "ALTERNATIVE_CAPABILITY_PATHS",
    "CANONICAL_METRIC_PATHS",
    "EXTRACTOR_LINKAGE_METRIC_PATHS",
    "LEGACY_ALIAS_TO_FIELD",
    "METRIC_REGISTRY",
    "METRIC_REGISTRY_VERSION",
    "MetricDefinition",
    "RUNTIME_NULL_POLICY",
    "RUNTIME_SCORE_AVAILABLE_POLICY",
    "metric_applies_to_family",
    "missing_capabilities",
    "validate_metric_registry",
]
