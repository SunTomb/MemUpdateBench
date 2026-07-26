from __future__ import annotations

import json
import warnings

import pytest
from pydantic import ValidationError

import mub.vnext.contracts as contracts
from mub.vnext.contracts.common import ArtifactRef, FrozenDict, MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.manifest import RunManifest, ScorerConfig, TaskManifest
from mub.vnext.contracts.score import (
    ActionScores,
    AnswerScores,
    AuditScores,
    ProtocolScores,
    RetrievalScores,
    ScoreRecord,
    StateScores,
    StoreScores,
    SystemScores,
)
from mub.vnext.version import (
    METRIC_REGISTRY_VERSION,
    PRIMARY_FAILURE_PRECEDENCE_VERSION,
    PROFILE_VERSION,
    RUN_MANIFEST_VERSION,
    RUNTIME_RECORD_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TASK_MANIFEST_VERSION,
)

SCORE_RECORD_FIELDS = [
    "schema_version", "scorer_version", "task_id", "run_id", "adapter_id",
    "task_family", "difficulty", "completion_status", "supported_metric_fields",
    "protocol_scores", "action_scores", "state_scores", "store_scores",
    "retrieval_scores", "answer_scores", "system_scores", "audit_scores",
    "failure_flags", "primary_failure", "legacy_metrics",
]
TASK_MANIFEST_FIELDS = [
    "schema_version", "task_manifest_version", "data_release_id",
    "split_policy_version", "task_schema_version", "compiler_versions",
    "source_manifest_paths_and_hashes", "generation_configs_and_hashes",
    "split_counts", "family_difficulty_counts", "semantic_core_counts",
    "task_file_paths_and_hashes", "leakage_check_summary",
    "human_audit_artifacts", "created_at", "code_revision",
]
RUN_MANIFEST_FIELDS = [
    "schema_version", "run_manifest_version", "run_id", "timestamp",
    "code_revision", "dirty_state", "task_manifest", "task_schema_version",
    "runtime_record_version", "scorer_version", "metric_registry_version",
    "profile_version", "adapter_info", "adapter_capabilities",
    "capability_verification_artifact", "model_name", "provider",
    "model_revision", "prompt_config", "decoding_config", "seed_information",
    "action_parser_version", "answer_parser_version",
    "memory_entry_extractor_version", "object_value_extractor_config_hash",
    "redaction_policy_version", "environment_summary", "package_summary",
    "expected_task_count", "completed_task_count", "failed_task_count",
    "not_supported_task_count", "raw_provider_response_artifacts",
    "raw_adapter_state_artifacts", "normalized_runtime_artifacts",
    "score_artifacts", "native_vs_extracted_field_summary",
]
SCORER_CONFIG_FIELDS = [
    "scorer_version", "metric_registry_version", "value_normalization_profile",
    "answer_normalization_profile", "primary_failure_precedence_version",
    "requested_metric_fields", "legacy_compatibility_mode",
    "strict_capability_check",
]
SCORE_LAYER_FIELDS = {
    ProtocolScores: [
        "action_parse_valid", "answer_parse_valid", "execution_success_rate",
        "unsupported_operation_rate", "fallback_rate",
    ],
    ActionScores: [
        "operation_accuracy", "full_action_exact_match", "object_key_accuracy",
        "entity_accuracy", "attribute_accuracy", "value_accuracy",
        "false_write_rate", "missed_write_rate", "wrong_object_write_rate",
    ],
    StateScores: [
        "final_state_accuracy", "state_precision", "state_recall", "state_f1",
        "state_resolve_rate", "collateral_corruption_rate",
        "expected_absence_accuracy",
    ],
    StoreScores: [
        "obsolete_version_count", "stale_conflicting_value_count",
        "duplicate_current_count", "final_memory_size", "compaction_ratio",
        "write_amplification",
    ],
    RetrievalScores: [
        "current_recall_at_k", "current_mrr", "stale_exposure_rate",
        "stale_count_in_context", "distractor_exposure_rate",
    ],
    AnswerScores: [
        "exact_match", "normalized_match", "token_f1",
        "structured_field_accuracy", "stale_copied", "distractor_copied",
        "gold_retrieved_wrong_answer", "answer_state_consistency",
    ],
    SystemScores: [
        "ingest_latency_ms", "retrieval_latency_ms", "answer_latency_ms",
        "token_usage", "api_cost", "error_rate",
    ],
    AuditScores: [
        "action_trace_available", "state_export_available",
        "retrieval_trace_available", "source_provenance_coverage",
        "manifest_completeness",
    ],
}


def artifact(path: str, char: str = "a", record_count: object = 1) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=char * 64,
        media_type="application/json",
        record_count=record_count,
    )


def make_task_manifest(**overrides) -> TaskManifest:
    data = {
        "data_release_id": "vnext-phase0-fixture",
        "split_policy_version": "split-policy-v1",
        "compiler_versions": {"task_compiler": "1.0.0"},
        "source_manifest_paths_and_hashes": [artifact("sources/manifest.json")],
        "generation_configs_and_hashes": [artifact("configs/generation.json", "b")],
        "split_counts": {"test": 1},
        "family_difficulty_counts": {"repeated_same_slot_update.easy": 1},
        "semantic_core_counts": {"test": 1},
        "task_file_paths_and_hashes": [artifact("tasks/test.jsonl", "c")],
        "leakage_check_summary": {"passed": True, "overlap_count": 0},
        "human_audit_artifacts": [artifact("audits/sample.json", "d")],
        "created_at": "2026-07-20T00:00:00Z",
        "code_revision": "fixed-test-revision",
    }
    data.update(overrides)
    return TaskManifest(**data)


def make_scorer_config(**overrides) -> ScorerConfig:
    data = {
        "value_normalization_profile": "typed_exact_v1",
        "answer_normalization_profile": "normalized_exact_v1",
    }
    data.update(overrides)
    return ScorerConfig(**data)


def test_scorer_config_accepts_only_exact_current_normalization_profiles() -> None:
    valid = make_scorer_config()
    assert valid.value_normalization_profile == "typed_exact_v1"
    assert valid.answer_normalization_profile == "normalized_exact_v1"

    cases = (
        ("value_normalization_profile", "unknown_value_profile"),
        ("value_normalization_profile", " typed_exact_v1"),
        ("value_normalization_profile", b"typed_exact_v1"),
        ("value_normalization_profile", 1),
        ("answer_normalization_profile", "unknown_answer_profile"),
        ("answer_normalization_profile", "normalized_exact_v1 "),
        ("answer_normalization_profile", b"normalized_exact_v1"),
        ("answer_normalization_profile", 1),
    )
    for field, invalid in cases:
        with pytest.raises(ValidationError):
            make_scorer_config(**{field: invalid})


def test_score_record_and_layers_have_exact_field_sets(make_score_record) -> None:
    score = make_score_record()
    assert list(ScoreRecord.model_fields) == SCORE_RECORD_FIELDS
    assert list(score.model_dump(mode="json")) == SCORE_RECORD_FIELDS
    for model_type, fields in SCORE_LAYER_FIELDS.items():
        assert list(model_type.model_fields) == fields


def test_every_score_metric_is_nullable() -> None:
    for model_type, fields in SCORE_LAYER_FIELDS.items():
        assert model_type().model_dump() == {field: None for field in fields}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_task5_float_metrics_reject_non_finite_values(make_score_record, invalid) -> None:
    for layer, field in (
        ("answer_scores", "exact_match"),
        ("store_scores", "compaction_ratio"),
        ("system_scores", "ingest_latency_ms"),
    ):
        with pytest.raises(ValidationError):
            make_score_record(**{layer: {field: invalid}})


def test_score_record_is_deeply_immutable_and_replaces_atomically(make_score_record) -> None:
    score = make_score_record()
    support = MetricFieldSupport(
        reason=SupportReason.NOT_SUPPORTED,
        null_policy="exclude_from_mean",
    )

    with pytest.raises(ValidationError):
        score.retrieval_scores.current_mrr = None
    with pytest.raises(ValidationError):
        score.retrieval_scores = RetrievalScores(current_mrr=None)
    with pytest.raises(TypeError):
        score.supported_metric_fields["retrieval_scores.current_mrr"] = support
    with pytest.raises(ValidationError):
        support.detail = "changed"

    with pytest.raises(ValidationError):
        score.validated_replace(retrieval_scores={"current_mrr": None})
    with pytest.raises(ValidationError):
        score.model_copy(update={"retrieval_scores": {"current_mrr": None}})
    assert score.retrieval_scores.current_mrr == 1.0
    assert score.supported_metric_fields == {}

    retrieval = score.retrieval_scores.model_dump(mode="python")
    retrieval["current_mrr"] = None
    replacement = score.validated_replace(
        retrieval_scores=retrieval,
        supported_metric_fields={"retrieval_scores.current_mrr": support},
    )
    assert replacement.retrieval_scores.current_mrr is None
    assert set(replacement.supported_metric_fields) == {"retrieval_scores.current_mrr"}
    assert score.retrieval_scores.current_mrr == 1.0


def test_immutable_task5_collections_preserve_json_shapes_and_round_trip(
    make_score_record, make_run_manifest
) -> None:
    task_manifest = make_task_manifest()
    run_manifest = make_run_manifest()
    config = make_scorer_config(requested_metric_fields=["answer_scores.exact_match"])

    with pytest.raises(TypeError):
        task_manifest.split_counts["dev"] = 1
    with pytest.raises(AttributeError):
        task_manifest.task_file_paths_and_hashes.append(artifact("tasks/dev.jsonl"))
    with pytest.raises(AttributeError):
        config.requested_metric_fields.append("state_scores.state_f1")
    with pytest.raises(ValidationError):
        run_manifest.expected_task_count = 2
    with pytest.raises(ValidationError):
        run_manifest.adapter_info.system_version = "changed"

    for model in (make_score_record(), task_manifest, run_manifest, config):
        dumped = model.model_dump(mode="json")
        assert type(model).model_validate(dumped) == model
        assert model.model_copy(deep=True) == model
    score_dump = make_score_record().model_dump(mode="json")
    assert isinstance(score_dump["supported_metric_fields"], dict)
    assert isinstance(task_manifest.model_dump(mode="json")["split_counts"], dict)
    assert isinstance(task_manifest.model_dump(mode="json")["task_file_paths_and_hashes"], list)
    assert isinstance(config.model_dump(mode="json")["requested_metric_fields"], list)


def test_run_manifest_count_updates_are_atomic(make_run_manifest) -> None:
    manifest = make_run_manifest(expected=2, completed=1, failed=1, not_supported=0)
    with pytest.raises(ValidationError):
        manifest.validated_replace(completed_task_count=2)
    with pytest.raises(ValidationError):
        manifest.model_copy(update={"completed_task_count": 2})
    assert manifest.completed_task_count == 1
    assert manifest.failed_task_count == 1

    replacement = manifest.validated_replace(
        completed_task_count=2,
        failed_task_count=0,
    )
    assert replacement.completed_task_count == 2
    assert replacement.failed_task_count == 0
    assert manifest.completed_task_count == 1


def test_frozen_mapping_has_no_dict_base_mutation_bypass(make_score_record) -> None:
    source = {"a": 1}
    frozen = FrozenDict(source)
    source["a"] = 9
    assert frozen["a"] == 1
    assert list(frozen) == ["a"]
    assert len(frozen) == 1
    assert frozen == {"a": 1}
    assert repr(frozen) == "{'a': 1}"

    for mutation in (
        lambda: frozen.__init__({"replaced": 2}),
        lambda: dict.__setitem__(frozen, "b", 2),
        lambda: dict.update(frozen, {"b": 2}),
        lambda: frozen.__setitem__("b", 2),
        lambda: frozen.update({"b": 2}),
        lambda: frozen.pop("a"),
    ):
        with pytest.raises((AttributeError, TypeError)):
            mutation()
    assert frozen == {"a": 1}

    storage = frozen._FrozenDict__data
    with pytest.raises(TypeError):
        storage["b"] = 2
    with pytest.raises(AttributeError):
        frozen._FrozenDict__data = {"b": 2}

    score = make_score_record(legacy_metrics={"nested": [{"x": [1, 2]}]})
    support_map = score.supported_metric_fields
    nested_map = score.legacy_metrics["nested"][0]
    for mapping in (support_map, nested_map):
        with pytest.raises(TypeError):
            mapping.__init__({"bypass": True})
        with pytest.raises(TypeError):
            dict.__setitem__(mapping, "bypass", True)
        with pytest.raises(TypeError):
            dict.update(mapping, {"bypass": True})
    assert score.supported_metric_fields == {}
    assert "bypass" not in nested_map
    assert ScoreRecord.model_validate(score.model_dump(mode="python")) == score


def test_frozen_nested_json_serializes_without_warnings_and_stays_immutable(
    make_score_record, make_run_manifest
) -> None:
    nested = {"nested": [{"x": [1, 2]}]}
    score = make_score_record(legacy_metrics=nested)
    task_manifest = make_task_manifest(leakage_check_summary=nested)
    run_manifest = make_run_manifest(
        prompt_config=nested,
        decoding_config=nested,
        seed_information=nested,
        environment_summary=nested,
        package_summary=nested,
        native_vs_extracted_field_summary=nested,
    )

    with pytest.raises(AttributeError):
        score.legacy_metrics["nested"].append({"x": [3]})
    with pytest.raises(TypeError):
        score.legacy_metrics["nested"][0]["x"][0] = 3

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for model, field_name in (
            (score, "legacy_metrics"),
            (task_manifest, "leakage_check_summary"),
            (run_manifest, "prompt_config"),
        ):
            python_dump = model.model_dump(mode="python")
            json_dump = model.model_dump(mode="json")
            json_text_dump = json.loads(model.model_dump_json())
            assert type(model).model_validate(python_dump) == model
            assert isinstance(python_dump[field_name]["nested"], list)
            assert isinstance(python_dump[field_name]["nested"][0]["x"], list)
            assert json_dump[field_name] == python_dump[field_name]
            assert json_text_dump[field_name] == python_dump[field_name]

        replacement = score.validated_replace(legacy_metrics=nested)
        assert replacement.model_dump(mode="python")["legacy_metrics"] == nested

        run_dump = run_manifest.model_dump(mode="python")
        for field_name in (
            "prompt_config",
            "decoding_config",
            "seed_information",
            "environment_summary",
            "package_summary",
            "native_vs_extracted_field_summary",
        ):
            assert isinstance(run_dump[field_name]["nested"], list)
            assert isinstance(run_dump[field_name]["nested"][0]["x"], list)


def test_manifest_and_scorer_config_have_exact_field_sets(make_run_manifest) -> None:
    task_manifest = make_task_manifest()
    run_manifest = make_run_manifest()
    scorer_config = make_scorer_config()
    assert list(TaskManifest.model_fields) == TASK_MANIFEST_FIELDS
    assert list(task_manifest.model_dump(mode="json")) == TASK_MANIFEST_FIELDS
    assert list(RunManifest.model_fields) == RUN_MANIFEST_FIELDS
    assert list(run_manifest.model_dump(mode="json")) == RUN_MANIFEST_FIELDS
    assert list(ScorerConfig.model_fields) == SCORER_CONFIG_FIELDS
    assert list(scorer_config.model_dump(mode="json")) == SCORER_CONFIG_FIELDS


def test_task5_models_canonical_json_revalidate(make_score_record, make_run_manifest) -> None:
    models = [
        make_score_record(), make_task_manifest(), make_run_manifest(),
        make_scorer_config(requested_metric_fields=["answer_scores.exact_match"]),
    ]
    for model in models:
        assert type(model).model_validate_json(model.model_dump_json()) == model


def test_unsupported_metric_is_null_with_exact_reason(make_score_record) -> None:
    support = MetricFieldSupport(
        reason=SupportReason.NOT_SUPPORTED,
        null_policy="exclude_from_mean",
    )
    score = make_score_record(
        retrieval_scores={"current_mrr": None},
        supported_metric_fields={"retrieval_scores.current_mrr": support},
    )
    assert score.retrieval_scores.current_mrr is None
    assert score.supported_metric_fields["retrieval_scores.current_mrr"].reason is SupportReason.NOT_SUPPORTED


@pytest.mark.parametrize("reason", list(SupportReason))
def test_exact_support_reasons_are_accepted(make_score_record, reason) -> None:
    score = make_score_record(
        retrieval_scores={"current_mrr": None},
        supported_metric_fields={
            "retrieval_scores.current_mrr": MetricFieldSupport(
                reason=reason, null_policy="exclude_from_mean"
            )
        },
    )
    assert score.supported_metric_fields["retrieval_scores.current_mrr"].reason is reason


def test_every_null_metric_requires_support_entry(make_score_record) -> None:
    data = make_score_record().model_dump(mode="json")
    data["retrieval_scores"]["current_mrr"] = None
    with pytest.raises(ValidationError, match="null metric fields missing support entries"):
        ScoreRecord(**data)


def test_support_map_rejects_unknown_and_nonnull_paths(make_score_record) -> None:
    base = make_score_record().model_dump(mode="json")
    support = {
        "reason": SupportReason.NOT_SUPPORTED.value,
        "null_policy": "exclude_from_mean",
        "detail": None,
    }
    with pytest.raises(ValidationError, match="unknown metric support paths"):
        ScoreRecord(**{**base, "supported_metric_fields": {"retrieval_scores.not_a_metric": support}})
    with pytest.raises(ValidationError, match="support entries for non-null metrics"):
        ScoreRecord(**{**base, "supported_metric_fields": {"retrieval_scores.current_mrr": support}})
    with pytest.raises(ValidationError, match="unknown metric support paths"):
        ScoreRecord(**{**base, "supported_metric_fields": {"current_mrr": support}})


def test_support_reason_taxonomy_is_not_open_ended(make_score_record) -> None:
    data = make_score_record().model_dump(mode="json")
    data["retrieval_scores"]["current_mrr"] = None
    data["supported_metric_fields"] = {
        "retrieval_scores.current_mrr": {
            "reason": "unscorable_for_new_reason",
            "null_policy": "exclude_from_mean",
        }
    }
    with pytest.raises(ValidationError):
        ScoreRecord(**data)


def test_requested_metric_fields_validate_paths_and_empty_means_all() -> None:
    config = make_scorer_config(requested_metric_fields=[])
    selected = make_scorer_config(
        requested_metric_fields=[
            "protocol_scores.action_parse_valid", "answer_scores.exact_match"
        ]
    )
    assert config.requested_metric_fields == ()
    assert "all registered metrics" in ScorerConfig.model_fields["requested_metric_fields"].description.lower()
    assert selected.requested_metric_fields == (
        "answer_scores.exact_match", "protocol_scores.action_parse_valid"
    )
    for requested in (
        ["answer_scores.exact_match", "answer_scores.exact_match"],
        ["answer_scores.unknown_metric"],
        ["exact_match"],
    ):
        with pytest.raises(ValidationError):
            make_scorer_config(requested_metric_fields=requested)


@pytest.mark.parametrize("invalid", [True, "1", "0.5"])
def test_score_float_metrics_reject_bool_and_numeric_strings(make_score_record, invalid) -> None:
    with pytest.raises(ValidationError):
        make_score_record(answer_scores={"exact_match": invalid})
    with pytest.raises(ValidationError):
        make_score_record(system_scores={"ingest_latency_ms": invalid})


@pytest.mark.parametrize("valid", [0, 0.0, 0.5, 1, 1.0])
def test_score_float_metrics_accept_json_numbers(make_score_record, valid) -> None:
    assert make_score_record(answer_scores={"exact_match": valid}).answer_scores.exact_match == valid


@pytest.mark.parametrize("invalid", [True, "2", 2.0])
def test_count_metrics_reject_non_strict_integers(make_score_record, invalid) -> None:
    with pytest.raises(ValidationError):
        make_score_record(store_scores={"obsolete_version_count": invalid})
    with pytest.raises(ValidationError):
        make_score_record(system_scores={"token_usage": invalid})


@pytest.mark.parametrize("invalid", [1, 0, "true", "false"])
def test_availability_fields_reject_non_strict_booleans(make_score_record, invalid) -> None:
    with pytest.raises(ValidationError):
        make_score_record(protocol_scores={"action_parse_valid": invalid})
    with pytest.raises(ValidationError):
        make_score_record(audit_scores={"state_export_available": invalid})


def test_score_metric_bounds_are_enforced(make_score_record) -> None:
    for invalid_rate in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            make_score_record(action_scores={"operation_accuracy": invalid_rate})
        with pytest.raises(ValidationError):
            make_score_record(system_scores={"error_rate": invalid_rate})
    for layer, field, invalid in (
        ("store_scores", "obsolete_version_count", -1),
        ("retrieval_scores", "stale_count_in_context", -1),
        ("store_scores", "compaction_ratio", -0.01),
        ("system_scores", "api_cost", -0.01),
    ):
        with pytest.raises(ValidationError):
            make_score_record(**{layer: {field: invalid}})


def test_manifest_counts_are_strict_nonnegative_integers(make_run_manifest) -> None:
    for invalid in (True, "1", 1.0, -1):
        with pytest.raises(ValidationError):
            make_task_manifest(split_counts={"test": invalid})
        with pytest.raises(ValidationError):
            make_task_manifest(family_difficulty_counts={"family.easy": invalid})
        with pytest.raises(ValidationError):
            make_task_manifest(semantic_core_counts={"test": invalid})
        with pytest.raises(ValidationError):
            make_run_manifest(expected=invalid, completed=0, failed=0, not_supported=0)


def test_run_manifest_strict_bool_hash_and_count_reconciliation(make_run_manifest) -> None:
    for invalid in (1, 0, "true"):
        with pytest.raises(ValidationError):
            make_run_manifest(dirty_state=invalid)
    for invalid_hash in ("A" * 64, "short"):
        with pytest.raises(ValidationError):
            make_run_manifest(object_value_extractor_config_hash=invalid_hash)
    manifest = make_run_manifest(expected=10, completed=7, failed=2, not_supported=1)
    assert manifest.completed_task_count + manifest.failed_task_count + manifest.not_supported_task_count == 10
    with pytest.raises(ValidationError, match="must equal expected_task_count"):
        make_run_manifest(expected=10, completed=7, failed=1, not_supported=1)


def test_artifact_references_reject_invalid_hashes(make_run_manifest) -> None:
    with pytest.raises(ValidationError):
        make_task_manifest(task_file_paths_and_hashes=[{
            "path": "tasks.jsonl", "sha256": "A" * 64,
            "media_type": "application/jsonl",
        }])
    with pytest.raises(ValidationError):
        make_run_manifest(task_manifest={
            "path": "manifest.json", "sha256": "short",
            "media_type": "application/json",
        })


def test_run_manifest_optional_model_fields_accept_none(make_run_manifest) -> None:
    manifest = make_run_manifest(
        capability_verification_artifact=None,
        model_name=None,
        provider=None,
        model_revision=None,
    )
    assert manifest.capability_verification_artifact is None
    assert manifest.model_name is None
    assert manifest.provider is None
    assert manifest.model_revision is None


def test_manifest_timestamps_have_no_dynamic_defaults(make_run_manifest) -> None:
    task_data = make_task_manifest().model_dump(mode="json")
    task_data.pop("created_at")
    with pytest.raises(ValidationError):
        TaskManifest(**task_data)

    run_data = make_run_manifest().model_dump(mode="json")
    run_data.pop("timestamp")
    with pytest.raises(ValidationError):
        RunManifest(**run_data)


def test_artifact_record_count_is_a_strict_nonnegative_integer(make_run_manifest) -> None:
    for invalid in (True, "1", 1.0, -1):
        with pytest.raises(ValidationError):
            artifact("artifact.json", record_count=invalid)
    assert artifact("artifact.json", record_count=0).record_count == 0
    valid_artifact = artifact("artifact.json", record_count=2)
    assert valid_artifact.record_count == 2
    with pytest.raises(ValidationError):
        valid_artifact.record_count = 3


def test_adapter_capability_bits_are_strict_and_immutable(make_run_manifest) -> None:
    for invalid in (1, 0, "true", "false", "yes"):
        with pytest.raises(ValidationError):
            contracts.AdapterCapabilities(supports_add=invalid)
        with pytest.raises(ValidationError):
            make_run_manifest(adapter_capabilities={"supports_add": invalid})

    capabilities = contracts.AdapterCapabilities(supports_add=True)
    assert capabilities.supports_add is True
    with pytest.raises(ValidationError):
        capabilities.supports_add = False


def test_builders_merge_nested_and_count_overrides(make_score_record, make_run_manifest) -> None:
    score = make_score_record(
        answer_scores={"exact_match": 0.25},
        retrieval_scores={"current_mrr": 0.5},
    )
    manifest = make_run_manifest(
        expected=2, completed=1, failed=0, not_supported=1,
        adapter_info={"system_version": "2.0.0"},
        prompt_config={"template": "alternate"},
    )
    assert score.answer_scores.exact_match == 0.25
    assert score.answer_scores.normalized_match == 1.0
    assert score.retrieval_scores.current_mrr == 0.5
    assert manifest.adapter_info.adapter_id == "adapter_fixture"
    assert manifest.adapter_info.system_version == "2.0.0"
    assert manifest.prompt_config == {"template": "alternate", "version": "prompt-v1"}


def test_version_defaults_are_canonical(make_score_record, make_run_manifest) -> None:
    score = make_score_record()
    task_manifest = make_task_manifest()
    run_manifest = make_run_manifest()
    config = make_scorer_config()
    assert (score.schema_version, score.scorer_version) == (SCHEMA_VERSION, SCORER_VERSION)
    assert (task_manifest.schema_version, task_manifest.task_manifest_version) == (SCHEMA_VERSION, TASK_MANIFEST_VERSION)
    assert task_manifest.task_schema_version == SCHEMA_VERSION
    assert run_manifest.schema_version == SCHEMA_VERSION
    assert run_manifest.run_manifest_version == RUN_MANIFEST_VERSION
    assert run_manifest.task_schema_version == SCHEMA_VERSION
    assert run_manifest.runtime_record_version == RUNTIME_RECORD_VERSION
    assert run_manifest.scorer_version == SCORER_VERSION
    assert run_manifest.metric_registry_version == METRIC_REGISTRY_VERSION
    assert run_manifest.profile_version == PROFILE_VERSION
    assert config.scorer_version == SCORER_VERSION
    assert config.metric_registry_version == METRIC_REGISTRY_VERSION
    assert config.primary_failure_precedence_version == PRIMARY_FAILURE_PRECEDENCE_VERSION


def test_contracts_reject_extra_fields(make_score_record, make_run_manifest) -> None:
    records = [
        (ScoreRecord, make_score_record().model_dump(mode="json")),
        (TaskManifest, make_task_manifest().model_dump(mode="json")),
        (RunManifest, make_run_manifest().model_dump(mode="json")),
        (ScorerConfig, make_scorer_config().model_dump(mode="json")),
    ]
    for model_type, data in records:
        with pytest.raises(ValidationError):
            model_type(**{**data, "unexpected": True})


def test_legacy_mode_is_exact_namespace_or_none_and_strict_bool_is_enforced() -> None:
    assert make_scorer_config(legacy_compatibility_mode=None).legacy_compatibility_mode is None
    assert make_scorer_config(legacy_compatibility_mode="legacy_p63").legacy_compatibility_mode == "legacy_p63"
    for invalid in ("p63.slot_prompt", "legacy_P63", "legacy_p63.slot_prompt", True, False, 1):
        with pytest.raises(ValidationError):
            make_scorer_config(legacy_compatibility_mode=invalid)
    for invalid in (1, 0, "true"):
        with pytest.raises(ValidationError):
            make_scorer_config(strict_capability_check=invalid)


def test_task5_public_models_are_exported() -> None:
    expected = {
        "ActionScores", "AnswerScores", "AuditScores", "ProtocolScores",
        "RetrievalScores", "RunManifest", "ScoreRecord", "ScorerConfig",
        "StateScores", "StoreScores", "SystemScores", "TaskManifest",
    }
    for name in expected:
        assert getattr(contracts, name).__name__ == name
        assert name in contracts.__all__
