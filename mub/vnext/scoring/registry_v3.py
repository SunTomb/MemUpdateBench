from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, StrictBool, StrictStr, field_validator, model_validator

from mub.vnext.contracts.common import FrozenDict, ImmutableContractModel
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS
from mub.vnext.contracts.v3.version import SCORER_VERSION_V3
from mub.vnext.scoring.registry import METRIC_REGISTRY as METRIC_REGISTRY_V2

ALL_FAMILIES = "*"
ALL_QUERY_KINDS = "*"


class MetricDescriptorV3(ImmutableContractModel):
    field_path: StrictStr
    layer: StrictStr
    direction: Literal["higher", "lower", "neutral"]
    principal: StrictBool
    applicable_task_families: tuple[StrictStr, ...]
    applicable_query_kinds: tuple[StrictStr, ...]
    required_capabilities: tuple[StrictStr, ...] = ()
    aggregation_rule: StrictStr
    numerator_definition: StrictStr
    denominator_definition: StrictStr
    unsupported_value_policy: Literal["null_with_typed_support_reason"] = "null_with_typed_support_reason"
    introduced_in_scorer_version: Literal[SCORER_VERSION_V3] = SCORER_VERSION_V3

    @field_validator("applicable_task_families", "applicable_query_kinds", "required_capabilities", mode="before")
    @classmethod
    def _canonical_tuple(cls, value):
        if type(value) not in {tuple, list, set, frozenset}:
            raise ValueError("descriptor semantic sets must be tuple-like")
        if any(type(item) is not str or not item.strip() for item in value):
            raise ValueError("descriptor semantic sets require nonblank built-in strings")
        if len(value) != len(set(value)):
            raise ValueError("descriptor semantic sets must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _coherent(self):
        if self.field_path not in CORE_METRIC_FIELD_PATHS:
            raise ValueError("unknown v3 metric path")
        if self.layer != self.field_path.split(".", 1)[0]:
            raise ValueError("metric layer does not match path")
        for value in (self.aggregation_rule, self.numerator_definition, self.denominator_definition):
            if not value.strip():
                raise ValueError("metric semantics cannot be blank")
        if self.principal and (not self.applicable_task_families or not self.applicable_query_kinds):
            raise ValueError("principal metrics require explicit applicability")
        return self


_LOWER_LEAVES = {
    "unsupported_operation_rate", "fallback_rate", "false_write_rate", "missed_write_rate",
    "wrong_object_write_rate", "collateral_corruption_rate", "obsolete_version_count",
    "stale_conflicting_value_count", "duplicate_current_count", "final_memory_size",
    "compaction_ratio", "write_amplification", "stale_exposure_rate", "stale_count_in_context",
    "distractor_exposure_rate", "stale_copied", "distractor_copied", "gold_retrieved_wrong_answer",
    "ingest_latency_ms", "retrieval_latency_ms", "answer_latency_ms", "token_usage", "api_cost",
    "error_rate", "collateral_damage_rate", "forgotten_exposure_rate", "forgotten_value_leakage_rate",
    "version_confusion_rate", "stale_propagation_rate",
}
_COUNT_LEAVES = {
    "obsolete_version_count", "stale_conflicting_value_count", "duplicate_current_count",
    "final_memory_size", "stale_count_in_context", "token_usage",
}
_PRINCIPAL = {
    "action_scores.full_action_exact_match", "state_scores.final_state_accuracy",
    "state_scores.expected_absence_accuracy", "answer_scores.exact_match",
    "answer_scores.reference_resolution_accuracy", "deletion_scores.deletion_accuracy",
    "deletion_scores.delete_scope_accuracy", "deletion_scores.collateral_damage_rate",
    "deletion_scores.ttl_compliance_rate", "deletion_scores.relearn_accuracy",
    "deletion_scores.forgotten_exposure_rate", "deletion_scores.forgotten_value_leakage_rate",
    "historical_scores.current_state_accuracy", "historical_scores.previous_state_accuracy",
    "historical_scores.point_in_time_accuracy", "historical_scores.transition_accuracy",
    "historical_scores.ordered_history_accuracy", "historical_scores.historical_support_recall",
    "historical_scores.historical_distance_accuracy", "synthesis_scores.multi_hop_accuracy",
    "synthesis_scores.multi_object_accuracy", "synthesis_scores.evidence_f1",
    "synthesis_scores.reasoning_support_accuracy", "synthesis_scores.stale_propagation_rate",
}
_FAMILY_BY_LAYER = {
    "deletion_scores": ("E", "deletion_forgetting"),
    "historical_scores": ("F", "current_historical_query"),
    "synthesis_scores": ("G", "long_horizon_memory_synthesis"),
}
_QUERY_BY_LEAF = {
    "answer_state_consistency": ("current", "multi_object_current"),
    "current_recall_at_k": ("current", "multi_object_current"),
    "current_mrr": ("current", "multi_object_current"),
    "stale_exposure_rate": ("current", "multi_object_current"),
    "stale_count_in_context": ("current", "multi_object_current"),
    "distractor_exposure_rate": ("current", "multi_object_current"),
    "current_state_accuracy": ("current",),
    "previous_state_accuracy": ("previous",),
    "point_in_time_accuracy": ("point_in_time",),
    "transition_accuracy": ("transition",),
    "ordered_history_accuracy": ("ordered_history",),
    "multi_hop_accuracy": ("update_sensitive_multi_hop",),
    "multi_object_accuracy": ("multi_object_current", "multi_object_current_consistency"),
}
_CAPS_BY_LAYER = {
    "historical_scores": ("supports_historical_query",),
    "synthesis_scores": ("supports_multi_object_query",),
}
_CAPS_BY_LEAF = {
    "deletion_accuracy": ("supports_delete", "exports_action_trace"),
    "delete_scope_accuracy": ("supports_delete", "supports_scoped_delete", "exports_action_trace"),
    "collateral_damage_rate": ("supports_delete", "supports_isolated_reset", "exports_entries", "exports_object_keys", "exports_values"),
    "ttl_compliance_rate": ("supports_ttl", "exports_action_trace", "exports_entries", "exports_object_keys", "exports_values"),
    "relearn_accuracy": ("supports_delete", "exports_entries", "exports_object_keys", "exports_values"),
    "forgotten_exposure_rate": ("supports_delete", "exports_retrieval_ids", "exports_object_keys", "exports_values"),
    "forgotten_value_leakage_rate": ("supports_delete", "supports_native_answer"),
    "historical_support_recall": ("exports_version_history", "exports_retrieval_ids"),
    "reasoning_support_accuracy": ("exports_evidence_linkage", "exports_retrieval_ids"),
    "evidence_precision": ("exports_evidence_linkage",),
    "evidence_recall": ("exports_evidence_linkage",),
    "evidence_f1": ("exports_evidence_linkage",),
}


def _descriptor(path: str) -> MetricDescriptorV3:
    layer, leaf = path.split(".", 1)
    legacy = METRIC_REGISTRY_V2.get(path)
    families = tuple(legacy.applicable_task_families) if legacy is not None else _FAMILY_BY_LAYER.get(layer, (ALL_FAMILIES,))
    if layer == "retrieval_scores":
        families = (ALL_FAMILIES,)
    query_kinds = _QUERY_BY_LEAF.get(leaf, (ALL_QUERY_KINDS,))
    caps = tuple(legacy.required_adapter_capabilities) if legacy is not None else _CAPS_BY_LEAF.get(leaf, _CAPS_BY_LAYER.get(layer, ()))
    count = leaf in _COUNT_LEAVES
    return MetricDescriptorV3(
        field_path=path,
        layer=layer,
        direction="lower" if leaf in _LOWER_LEAVES else "higher",
        principal=path in _PRINCIPAL,
        applicable_task_families=families,
        applicable_query_kinds=query_kinds,
        required_capabilities=caps,
        aggregation_rule="sum_and_non_null_mean" if count else "non_null_arithmetic_mean",
        numerator_definition=f"Canonical task-level numerator for {path}.",
        denominator_definition=f"Applicable supported observations for {path}; null rows are excluded.",
    )


def validate_metric_registry_v3(registry: Mapping[str, object]) -> FrozenDict:
    validated: dict[str, MetricDescriptorV3] = {}
    for path, raw in registry.items():
        if type(path) is not str or path in validated:
            raise ValueError("v3 registry paths must be unique built-in strings")
        descriptor = raw if isinstance(raw, MetricDescriptorV3) else MetricDescriptorV3.model_validate(raw)
        if descriptor.field_path != path:
            raise ValueError("registry key must equal descriptor field_path")
        validated[path] = descriptor
    if set(validated) != CORE_METRIC_FIELD_PATHS:
        raise ValueError("v3 metric registry must exactly cover CORE_METRIC_FIELD_PATHS")
    for descriptor in validated.values():
        if descriptor.principal and (not descriptor.applicable_task_families or not descriptor.denominator_definition.strip()):
            raise ValueError("principal metric lacks applicability or denominator policy")
    return FrozenDict((path, validated[path]) for path in sorted(validated))


CORE_METRIC_REGISTRY_V3 = validate_metric_registry_v3({path: _descriptor(path) for path in sorted(CORE_METRIC_FIELD_PATHS)})


_FAMILY_ALIASES = {
    "E": frozenset({"E", "deletion_forgetting"}),
    "deletion_forgetting": frozenset({"E", "deletion_forgetting"}),
    "F": frozenset({"F", "current_historical_query"}),
    "current_historical_query": frozenset({"F", "current_historical_query"}),
    "G": frozenset({"G", "long_horizon_memory_synthesis"}),
    "long_horizon_memory_synthesis": frozenset({"G", "long_horizon_memory_synthesis"}),
}


def metric_applies_v3(descriptor: MetricDescriptorV3, family: str, query_kinds: set[str]) -> bool:
    family_names = _FAMILY_ALIASES.get(family, frozenset({family}))
    family_ok = ALL_FAMILIES in descriptor.applicable_task_families or bool(
        family_names & set(descriptor.applicable_task_families)
    )
    query_ok = ALL_QUERY_KINDS in descriptor.applicable_query_kinds or bool(query_kinds & set(descriptor.applicable_query_kinds))
    return family_ok and query_ok


def missing_capabilities_v3(descriptor: MetricDescriptorV3, capabilities) -> tuple[str, ...]:
    return tuple(name for name in descriptor.required_capabilities if getattr(capabilities, name, False) is not True)


__all__ = ["ALL_FAMILIES", "ALL_QUERY_KINDS", "CORE_METRIC_REGISTRY_V3", "MetricDescriptorV3", "metric_applies_v3", "missing_capabilities_v3", "validate_metric_registry_v3"]
