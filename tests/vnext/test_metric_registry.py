from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from mub.utils import compute_exact_match, compute_f1
from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.enums import TaskFamily
from mub.vnext.contracts.manifest import ScorerConfig
from mub.vnext.contracts.score import METRIC_FIELD_PATHS, SCORE_LAYER_TYPES
from mub.vnext.scoring.registry import (
    ALL_TASK_FAMILIES,
    ALTERNATIVE_CAPABILITY_PATHS,
    CANONICAL_METRIC_PATHS,
    EXTRACTOR_LINKAGE_METRIC_PATHS,
    LEGACY_ALIAS_TO_FIELD,
    METRIC_REGISTRY,
    MetricDefinition,
    metric_applies_to_family,
    validate_metric_registry,
)
from mub.vnext.scoring.scorer import _same_value, _token_f1, _typed_mapping_equal
from mub.vnext.version import METRIC_REGISTRY_VERSION, SCORER_VERSION


def test_registry_exactly_covers_all_score_layer_fields_once() -> None:
    flattened = [
        f"{layer}.{field}"
        for layer, model in SCORE_LAYER_TYPES.items()
        for field in model.model_fields
    ]
    assert len(flattened) == 51
    assert set(flattened) == METRIC_FIELD_PATHS == set(METRIC_REGISTRY)
    assert len(METRIC_REGISTRY) == len(set(METRIC_REGISTRY))
    assert tuple(METRIC_REGISTRY) == tuple(sorted(METRIC_REGISTRY))
    assert tuple(METRIC_REGISTRY) == CANONICAL_METRIC_PATHS


def test_every_metric_definition_is_complete_canonical_and_resolvable() -> None:
    capability_fields = set(AdapterCapabilities.model_fields)
    canonical_families = {
        "repeated_same_slot_update",
        "interleaved_multi_slot_update",
        "entity_attribute_grounding",
        "noop_write_discipline",
        "deletion_forgetting",
        "current_historical_query",
        "long_horizon_memory_synthesis",
        "realistic_source_update",
    }
    for path, definition in METRIC_REGISTRY.items():
        layer, leaf = path.split(".")
        assert definition.field_name == path
        assert definition.layer == layer
        assert leaf in SCORE_LAYER_TYPES[layer].model_fields
        assert definition.value_type in {"bool", "rate", "count", "nonnegative_float"}
        assert definition.numerator_definition.strip()
        assert definition.denominator_definition.strip()
        assert definition.aggregation_rule.strip()
        assert definition.unsupported_value_policy.strip()
        assert definition.runtime_failure_policy.strip()
        assert definition.introduced_in_scorer_version == SCORER_VERSION
        assert definition.applicable_task_families
        assert set(definition.applicable_task_families) == {ALL_TASK_FAMILIES} or set(
            definition.applicable_task_families
        ) <= canonical_families
        assert set(definition.required_adapter_capabilities) <= capability_fields


def test_registry_definitions_are_metric_specific_not_circular_placeholders() -> None:
    for definition in METRIC_REGISTRY.values():
        assert "per-task observed" not in definition.numerator_definition.casefold()
    obsolete = METRIC_REGISTRY["store_scores.obsolete_version_count"]
    conflicting = METRIC_REGISTRY["store_scores.stale_conflicting_value_count"]
    assert "order" in obsolete.numerator_definition.casefold()
    assert "conflict" in conflicting.numerator_definition.casefold()
    assert obsolete.numerator_definition != conflicting.numerator_definition
    assert (
        METRIC_REGISTRY["action_scores.operation_accuracy"].runtime_failure_policy
        != METRIC_REGISTRY["store_scores.final_memory_size"].runtime_failure_policy
    )


def test_capability_paths_encode_level_two_state_and_metric_specific_store_gates() -> None:
    state_content_paths = (
        ("exports_object_keys", "exports_values"),
        ("requires_evaluation_extractor",),
    )
    for field in SCORE_LAYER_TYPES["state_scores"].model_fields:
        path = f"state_scores.{field}"
        assert METRIC_REGISTRY[path].required_adapter_capabilities == (
            "exports_entries",
            "supports_isolated_reset",
        )
        assert ALTERNATIVE_CAPABILITY_PATHS[path] == state_content_paths
    assert METRIC_REGISTRY[
        "answer_scores.answer_state_consistency"
    ].required_adapter_capabilities == (
        "exports_entries",
        "supports_isolated_reset",
    )
    assert ALTERNATIVE_CAPABILITY_PATHS[
        "answer_scores.answer_state_consistency"
    ] == state_content_paths

    assert "audit_scores.state_export_available" not in ALTERNATIVE_CAPABILITY_PATHS
    assert METRIC_REGISTRY[
        "store_scores.final_memory_size"
    ].required_adapter_capabilities == ("exports_entries",)
    assert "store_scores.final_memory_size" not in ALTERNATIVE_CAPABILITY_PATHS

    obsolete_paths = ALTERNATIVE_CAPABILITY_PATHS[
        "store_scores.obsolete_version_count"
    ]
    value_paths = ALTERNATIVE_CAPABILITY_PATHS[
        "store_scores.stale_conflicting_value_count"
    ]
    assert obsolete_paths == (
        ("exports_timestamps_or_order", "exports_object_keys"),
        ("exports_source_event_ids", "exports_object_keys"),
        ("exports_timestamps_or_order", "requires_evaluation_extractor"),
        ("exports_source_event_ids", "requires_evaluation_extractor"),
    )
    assert value_paths == (
        ("exports_timestamps_or_order", "exports_object_keys", "exports_values"),
        ("exports_source_event_ids", "exports_object_keys", "exports_values"),
        ("exports_timestamps_or_order", "requires_evaluation_extractor"),
        ("exports_source_event_ids", "requires_evaluation_extractor"),
    )
    assert ALTERNATIVE_CAPABILITY_PATHS[
        "store_scores.duplicate_current_count"
    ] == value_paths
    for path in (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    ):
        assert METRIC_REGISTRY[path].required_adapter_capabilities == (
            "exports_entries",
            "supports_isolated_reset",
        )
        assert path in EXTRACTOR_LINKAGE_METRIC_PATHS

    assert "state_scores.final_state_accuracy" in EXTRACTOR_LINKAGE_METRIC_PATHS
    assert "store_scores.final_memory_size" not in EXTRACTOR_LINKAGE_METRIC_PATHS
    with pytest.raises(TypeError):
        ALTERNATIVE_CAPABILITY_PATHS["state_scores.final_state_accuracy"] = ()  # type: ignore[index]


def test_registry_definitions_and_containers_are_immutable() -> None:
    assert isinstance(METRIC_REGISTRY, Mapping)
    with pytest.raises(TypeError):
        METRIC_REGISTRY["answer_scores.exact_match"] = METRIC_REGISTRY[  # type: ignore[index]
            "answer_scores.exact_match"
        ]
    with pytest.raises(ValidationError):
        METRIC_REGISTRY["answer_scores.exact_match"].layer = "state_scores"
    with pytest.raises(TypeError):
        METRIC_REGISTRY["answer_scores.exact_match"].legacy_aliases[0] = "changed"  # type: ignore[index]


def test_registry_validation_rejects_model_construct_bypass_and_bad_capability() -> None:
    invalid = MetricDefinition.model_construct(
        field_name="answer_scores.exact_match",
        layer="missing_layer",
        value_type="rate",
        numerator_definition="x",
        denominator_definition="y",
        aggregation_rule="mean",
        applicable_task_families=(ALL_TASK_FAMILIES,),
        required_adapter_capabilities=("not_a_capability",),
        unsupported_value_policy="null",
        runtime_failure_policy="null",
        legacy_aliases=(),
        introduced_in_scorer_version=SCORER_VERSION,
    )
    with pytest.raises(ValueError):
        validate_metric_registry({"answer_scores.exact_match": invalid})


def test_metric_definition_rejects_blank_malformed_and_mutable_shape_inputs() -> None:
    base = METRIC_REGISTRY["answer_scores.exact_match"].model_dump(mode="python")
    for field in (
        "field_name",
        "layer",
        "value_type",
        "numerator_definition",
        "denominator_definition",
        "aggregation_rule",
        "unsupported_value_policy",
        "runtime_failure_policy",
        "introduced_in_scorer_version",
    ):
        payload = dict(base)
        payload[field] = " "
        with pytest.raises(ValidationError):
            MetricDefinition.model_validate(payload)


def test_legacy_aliases_are_exact_global_unambiguous_and_inventory_safe() -> None:
    aliases = [alias for definition in METRIC_REGISTRY.values() for alias in definition.legacy_aliases]
    assert len(METRIC_REGISTRY) == 51
    assert len(aliases) == len(set(aliases)) == len(LEGACY_ALIAS_TO_FIELD)
    assert dict(LEGACY_ALIAS_TO_FIELD) == {}
    for nonidentical_alias in (
        "legacy_p63.final_state_accuracy",
        "legacy_p63.stale_same_slot_count",
        "legacy_p63.answer_exact_match",
        "legacy_p63.answer_token_f1",
        "legacy_p84.answer_rerun_exact_match",
        "legacy_p83.stale_removal_trace.original_em_avg",
    ):
        assert nonidentical_alias not in LEGACY_ALIAS_TO_FIELD

    collision_alias = "legacy_test.same_metric"
    exact = METRIC_REGISTRY["answer_scores.exact_match"].validated_replace(
        legacy_aliases=(collision_alias,)
    )
    state = METRIC_REGISTRY["state_scores.final_state_accuracy"].validated_replace(
        legacy_aliases=(collision_alias,)
    )
    with pytest.raises(ValueError, match="ambiguous legacy alias"):
        validate_metric_registry(
            {exact.field_name: exact, state.field_name: state},
            require_complete=False,
        )


def test_removed_p63_aliases_have_demonstrably_different_semantics() -> None:
    assert compute_exact_match("  QingDao ", "qingdao") == 1.0
    assert _same_value("  QingDao ", "qingdao") is False

    assert compute_f1("value value", "value value") == 0.5
    assert _token_f1("value value", "value value", "normalized_exact_v1") == 1.0

    gold = {"default::slot::friend:alex::location": "Qingdao"}
    collateral = {
        **gold,
        "default::slot::friend:bob::location": "Shanghai",
    }
    assert collateral[next(iter(gold))] == gold[next(iter(gold))]
    assert _typed_mapping_equal(collateral, gold) is False


def test_scorer_config_rejects_unknown_duplicate_malformed_alias_and_version_mismatch() -> None:
    base = {
        "value_normalization_profile": "typed_exact_v1",
        "answer_normalization_profile": "normalized_exact_v1",
    }
    for fields in (
        ("answer_scores.unknown",),
        ("answer_scores.exact_match", "answer_scores.exact_match"),
        ("answer_scores",),
        ("answer_scores.ExactMatch",),
        ("legacy_p84.answer_rerun_exact_match",),
    ):
        with pytest.raises(ValidationError):
            ScorerConfig(**base, requested_metric_fields=fields)
    with pytest.raises(ValidationError):
        ScorerConfig(**base, scorer_version="9.0.0")
    with pytest.raises(ValidationError):
        ScorerConfig(**base, metric_registry_version="9.0.0")
    with pytest.raises(ValidationError):
        ScorerConfig(**base, primary_failure_precedence_version="9.0.0")
    with pytest.raises(ValidationError):
        ScorerConfig(**base, legacy_compatibility_mode="p63.slot_prompt")
    assert ScorerConfig(**base, legacy_compatibility_mode="legacy_p84").legacy_compatibility_mode == "legacy_p84"


def test_semantic_tuple_inputs_are_exact_and_deterministically_canonicalized() -> None:
    base = METRIC_REGISTRY["answer_scores.exact_match"].model_dump(mode="python")
    paths = ["state_scores.final_state_accuracy", "answer_scores.exact_match"]
    config = ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=set(paths),
    )
    assert config.requested_metric_fields == tuple(sorted(paths))
    for invalid in ("answer_scores.exact_match", {"x": 1}, b"bytes", (item for item in paths)):
        with pytest.raises(ValidationError):
            ScorerConfig(
                value_normalization_profile="typed_exact_v1",
                answer_normalization_profile="normalized_exact_v1",
                requested_metric_fields=invalid,
            )

    reversed_payload = dict(base)
    reversed_payload["required_adapter_capabilities"] = list(
        reversed(("exports_retrieval_ids", "exports_retrieval_scores"))
    )
    definition = MetricDefinition.model_validate(reversed_payload)
    assert definition.required_adapter_capabilities == (
        "exports_retrieval_ids",
        "exports_retrieval_scores",
    )
    for field in (
        "applicable_task_families",
        "required_adapter_capabilities",
        "legacy_aliases",
    ):
        payload = dict(base)
        payload[field] = "not-a-tuple-like-set"
        with pytest.raises(ValidationError):
            MetricDefinition.model_validate(payload)


@pytest.mark.parametrize("family", list(TaskFamily))
def test_task_family_names_and_values_normalize_to_exact_values(family: TaskFamily) -> None:
    base = METRIC_REGISTRY["answer_scores.exact_match"].model_dump(mode="python")
    for supplied in (family.name, family.value):
        definition = MetricDefinition.model_validate(
            {**base, "applicable_task_families": [supplied]}
        )
        assert definition.applicable_task_families == (family.value,)
        assert metric_applies_to_family(definition, family.value)
        assert not metric_applies_to_family(definition, family.name)


def test_legacy_alias_regex_rejects_empty_repeated_and_trailing_components() -> None:
    base = METRIC_REGISTRY["answer_scores.exact_match"].model_dump(mode="python")
    valid = MetricDefinition.model_validate(
        {**base, "legacy_aliases": ["legacy_p63.answer.metrics.exact_match"]}
    )
    assert valid.legacy_aliases == ("legacy_p63.answer.metrics.exact_match",)
    for alias in (
        "legacy_p63.",
        "legacy_p63..exact_match",
        "legacy_p63.answer.",
        "legacy_p63.answer..exact_match",
    ):
        with pytest.raises(ValidationError):
            MetricDefinition.model_validate({**base, "legacy_aliases": [alias]})


def test_scorer_config_is_reexported_from_scoring_package() -> None:
    from mub.vnext.scoring import ScorerConfig as ExportedScorerConfig

    assert ExportedScorerConfig is ScorerConfig


def test_scorer_config_empty_request_means_all_and_versions_are_current() -> None:
    config = ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
    )
    assert config.requested_metric_fields == ()
    assert config.scorer_version == SCORER_VERSION
    assert config.metric_registry_version == METRIC_REGISTRY_VERSION
