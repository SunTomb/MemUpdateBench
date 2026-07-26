from __future__ import annotations

from copy import deepcopy
import warnings

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.enums import CompletionStatus, SupportReason
from mub.vnext.contracts.manifest import ScorerConfig
from mub.vnext.contracts.runtime import AnswerPrediction, ParsedManagerAction, TaskRunRecord
from mub.vnext.contracts.score import SCORE_LAYER_TYPES
from mub.vnext.io.canonical import canonical_json_bytes
import mub.vnext.scoring.scorer as scorer_module
from mub.vnext.scoring.scorer import score_task


def _all_capabilities(**overrides) -> AdapterCapabilities:
    data = {
        name: True
        for name in AdapterCapabilities.model_fields
        if name != "extractor_version"
    }
    data["requires_evaluation_extractor"] = False
    data["extractor_version"] = None
    data.update(overrides)
    return AdapterCapabilities(**data)


def _config(*requested: str, strict: bool = False, legacy: str | None = None) -> ScorerConfig:
    return ScorerConfig(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        requested_metric_fields=requested,
        legacy_compatibility_mode=legacy,
        strict_capability_check=strict,
    )


def _replace_run(run: TaskRunRecord, **changes) -> TaskRunRecord:
    payload = run.model_dump(mode="python")
    payload.update(changes)
    return TaskRunRecord.model_validate(payload)


def _replace_answer(run: TaskRunRecord, **changes) -> TaskRunRecord:
    payload = run.model_dump(mode="python")
    payload["answer_predictions"][0].update(changes)
    return TaskRunRecord.model_validate(payload)


def _complete_action_trace(run: TaskRunRecord) -> TaskRunRecord:
    payload = run.model_dump(mode="python")
    key = payload["parsed_actions"][0]["target_object_key"]
    payload["parsed_actions"].insert(
        0,
        {
            "event_id": "event_0",
            "operation": "ADD",
            "target_object_key": key,
            "value": "Dalian",
            "format_valid": True,
            "execution_status": "succeeded",
            "fallback_used": False,
            "error_flags": [],
            "raw_output": "ADD friend:alex.location = Dalian",
            "latency_ms": 1.0,
        },
    )
    return TaskRunRecord.model_validate(payload)


def _null_paths(score) -> set[str]:
    return {
        f"{layer}.{field}"
        for layer in SCORE_LAYER_TYPES
        for field, value in getattr(score, layer)
        if value is None
    }


def test_store_counts_distinguish_obsolete_same_value_from_conflicting_stale(
    make_task, make_task_run
) -> None:
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    key_id = next(iter(task_payload["gold"]["final_state"]))
    task_payload["gold"]["version_history"][key_id] = ["Qingdao", "Dalian", "Qingdao"]
    task = type(base_task).model_validate(task_payload)

    run_payload = make_task_run().model_dump(mode="python")
    current = run_payload["memory_snapshots"][0]["entries"][0]
    old_same = {**current, "entry_id": "entry_old_same", "version_index": 0}
    stale = {
        **current,
        "entry_id": "entry_stale",
        "value_candidate": "Dalian",
        "content": "friend:alex.location = Dalian",
        "version_index": 1,
    }
    latest = {**current, "entry_id": "entry_latest", "version_index": 2}
    run_payload["memory_snapshots"][0]["entries"] = [old_same, stale, latest]
    run_payload["memory_snapshots"][0]["store_size"] = 3
    run = TaskRunRecord.model_validate(run_payload)

    score = score_task(
        task,
        run,
        _all_capabilities(),
        _config(
            "store_scores.obsolete_version_count",
            "store_scores.stale_conflicting_value_count",
            "store_scores.duplicate_current_count",
        ),
    )
    assert score.store_scores.obsolete_version_count == 2
    assert score.store_scores.stale_conflicting_value_count == 1
    assert score.store_scores.duplicate_current_count == 0
    assert "stale_retained" in score.failure_flags


def test_missing_retrieval_linkage_is_missing_artifact_not_negative_evidence(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    trace = payload["retrieval_traces"][0]
    trace["gold_in_context"] = None
    trace["stale_in_context"] = None
    trace["retrieved_entries"][0]["object_key_candidate"] = None
    trace["retrieved_entries"][0]["value_candidate"] = None
    run = TaskRunRecord.model_validate(payload)
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("retrieval_scores.current_recall_at_k"),
    )
    assert score.retrieval_scores.current_recall_at_k is None
    assert score.supported_metric_fields[
        "retrieval_scores.current_recall_at_k"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert "current_not_retrieved" not in score.failure_flags


def test_benchmark_answer_artifact_does_not_require_native_adapter_answer_capability(
    make_task, make_task_run
) -> None:
    score = score_task(
        make_task(),
        make_task_run(),
        _all_capabilities(supports_native_answer=False),
        _config("answer_scores.exact_match", strict=True),
    )
    assert score.answer_scores.exact_match == 1.0
    assert "answer_scores.exact_match" not in score.supported_metric_fields


def test_cross_artifact_answer_diagnostics_are_null_when_retrieval_is_absent(
    make_task, make_task_run
) -> None:
    run = _replace_answer(
        _replace_run(make_task_run(), retrieval_traces=[]),
        cited_entry_ids=[],
    )
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config(
            "answer_scores.distractor_copied",
            "answer_scores.gold_retrieved_wrong_answer",
        ),
    )
    for path in (
        "answer_scores.distractor_copied",
        "answer_scores.gold_retrieved_wrong_answer",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_final_state_accuracy_and_failure_flags_use_typed_value_equality(
    make_task, make_task_run
) -> None:
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    key_id = next(iter(task_payload["gold"]["final_state"]))
    task_payload["gold"]["final_state"][key_id] = True
    task = type(base_task).model_validate(task_payload)
    run_payload = make_task_run().model_dump(mode="python")
    run_payload["memory_snapshots"][0]["state_by_object"][key_id] = 1
    run = TaskRunRecord.model_validate(run_payload)
    score = score_task(
        task,
        run,
        _all_capabilities(),
        _config("state_scores.final_state_accuracy"),
    )
    assert score.state_scores.final_state_accuracy == 0.0
    assert "current_state_missing" in score.failure_flags
    assert score.primary_failure == "missed_update"


def test_structured_field_accuracy_scores_fields_not_whole_object_exact_match(
    make_task, make_task_run
) -> None:
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    task_payload["queries"][0]["answer_schema"] = "object"
    task_payload["gold"]["gold_answers"]["query_0"] = {"city": "Qingdao", "country": "CN"}
    task_payload["gold"]["acceptable_answers"]["query_0"] = [
        {"city": "Qingdao", "country": "CN"}
    ]
    task = type(base_task).model_validate(task_payload)
    run = _replace_answer(
        make_task_run(),
        parsed_answer={"city": "Qingdao", "country": "wrong"},
    )
    score = score_task(
        task,
        run,
        _all_capabilities(),
        _config("answer_scores.structured_field_accuracy"),
    )
    assert score.answer_scores.structured_field_accuracy == 0.5


def test_partial_row_preserves_observable_nonaccuracy_store_diagnostics(
    make_task, make_task_run
) -> None:
    run = make_task_run(status=CompletionStatus.PARTIAL, exception_type="late_failure")
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config(
            "action_scores.operation_accuracy",
            "store_scores.final_memory_size",
        ),
    )
    assert score.action_scores.operation_accuracy is None
    assert score.supported_metric_fields[
        "action_scores.operation_accuracy"
    ].reason is SupportReason.RUNTIME_FAILED
    assert score.store_scores.final_memory_size == 1


def test_protocol_rates_include_missing_expected_outputs_in_denominators(
    make_task, make_task_run
) -> None:
    score = score_task(
        make_task(),
        make_task_run(),
        _all_capabilities(),
        _config("protocol_scores.action_parse_valid"),
    )
    assert score.protocol_scores.action_parse_valid is False
    assert score.protocol_scores.execution_success_rate == 0.5


def test_completed_full_capability_row_is_valid_deterministic_and_correct(
    make_task, make_task_run
) -> None:
    task = make_task()
    run = _complete_action_trace(make_task_run())
    task_before = deepcopy(task.model_dump(mode="python"))
    run_before = deepcopy(run.model_dump(mode="python"))

    first = score_task(task, run, _all_capabilities(), _config())
    second = score_task(task, run, _all_capabilities(), _config())

    assert first.task_id == task.task_id
    assert first.run_id == run.run_id
    assert first.adapter_id == run.adapter_id
    assert first.task_family == task.task_family
    assert first.difficulty == task.difficulty
    assert first.completion_status is CompletionStatus.COMPLETED
    assert first.protocol_scores.action_parse_valid is True
    assert first.audit_scores.action_trace_available is True
    assert first.state_scores.final_state_accuracy == 1.0
    assert first.answer_scores.exact_match == 1.0
    assert first.answer_scores.normalized_match == 1.0
    assert first.failure_flags == ()
    assert first.primary_failure == "correct"
    assert set(first.supported_metric_fields) == _null_paths(first)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert task.model_dump(mode="python") == task_before
    assert run.model_dump(mode="python") == run_before


def test_exact_support_map_completeness_has_one_entry_per_null_and_none_per_value(
    make_task, make_task_run
) -> None:
    score = score_task(
        make_task(),
        _complete_action_trace(make_task_run()),
        _all_capabilities(),
        _config("action_scores.operation_accuracy"),
    )
    null_paths = _null_paths(score)
    nonnull_paths = {
        f"{layer}.{field}"
        for layer in SCORE_LAYER_TYPES
        for field, value in getattr(score, layer)
        if value is not None
    }
    assert set(score.supported_metric_fields) == null_paths
    assert not (set(score.supported_metric_fields) & nonnull_paths)
    assert score.action_scores.operation_accuracy == 1.0
    assert score.protocol_scores.execution_success_rate == 1.0
    assert score.audit_scores.retrieval_trace_available is True
    assert score.state_scores.final_state_accuracy is None
    support = score.supported_metric_fields["state_scores.final_state_accuracy"]
    assert support.reason is SupportReason.NOT_APPLICABLE
    assert support.null_policy == "not_requested"


def test_empty_request_resolves_all_for_strict_and_nonstrict_capability_checks(
    make_task, make_task_run
) -> None:
    with pytest.raises(ValueError, match="requested metrics require unsupported"):
        score_task(
            make_task(),
            make_task_run(),
            AdapterCapabilities(),
            _config(strict=True),
        )

    nonstrict = score_task(
        make_task(),
        make_task_run(),
        AdapterCapabilities(),
        _config(strict=False),
    )
    assert nonstrict.supported_metric_fields[
        "action_scores.operation_accuracy"
    ].reason is SupportReason.NOT_SUPPORTED
    assert nonstrict.protocol_scores.action_parse_valid is False

    explicit_protocol = score_task(
        make_task(),
        make_task_run(),
        AdapterCapabilities(),
        _config("protocol_scores.action_parse_valid", strict=True),
    )
    assert explicit_protocol.protocol_scores.action_parse_valid is False


def test_structured_retrieval_linkage_overrides_contradictory_annotations(
    make_task, make_task_run
) -> None:
    current_payload = make_task_run().model_dump(mode="python")
    current_trace = current_payload["retrieval_traces"][0]
    current_trace["gold_in_context"] = False
    current_trace["stale_in_context"] = True
    current_run = TaskRunRecord.model_validate(current_payload)
    current_score = score_task(
        make_task(),
        current_run,
        _all_capabilities(),
        _config(
            "retrieval_scores.current_recall_at_k",
            "retrieval_scores.stale_exposure_rate",
        ),
    )
    assert current_score.retrieval_scores.current_recall_at_k == 1.0
    assert current_score.retrieval_scores.stale_exposure_rate == 0.0
    assert "current_not_retrieved" not in current_score.failure_flags
    assert "stale_retrieved" not in current_score.failure_flags

    stale_payload = make_task_run().model_dump(mode="python")
    stale_trace = stale_payload["retrieval_traces"][0]
    stale_trace["retrieved_entries"][0]["value_candidate"] = "Dalian"
    stale_trace["retrieved_entries"][0]["content"] = "friend:alex.location = Dalian"
    stale_trace["gold_in_context"] = True
    stale_trace["stale_in_context"] = False
    stale_run = TaskRunRecord.model_validate(stale_payload)
    stale_score = score_task(
        make_task(),
        stale_run,
        _all_capabilities(),
        _config(
            "retrieval_scores.current_recall_at_k",
            "retrieval_scores.stale_exposure_rate",
        ),
    )
    assert stale_score.retrieval_scores.current_recall_at_k == 0.0
    assert stale_score.retrieval_scores.stale_exposure_rate == 1.0
    assert "current_not_retrieved" in stale_score.failure_flags
    assert "stale_retrieved" in stale_score.failure_flags


def test_retrieval_annotations_are_fallback_when_structured_linkage_is_unavailable(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    trace = payload["retrieval_traces"][0]
    trace["retrieved_entries"][0]["object_key_candidate"] = None
    trace["retrieved_entries"][0]["value_candidate"] = None
    trace["gold_in_context"] = True
    trace["stale_in_context"] = True
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        _all_capabilities(),
        _config(
            "retrieval_scores.current_recall_at_k",
            "retrieval_scores.stale_exposure_rate",
        ),
    )
    assert score.retrieval_scores.current_recall_at_k == 1.0
    assert score.retrieval_scores.stale_exposure_rate == 1.0
    assert "current_not_retrieved" not in score.failure_flags
    assert "stale_retrieved" in score.failure_flags


def test_extractor_provenance_match_mismatch_and_unaffected_metrics(
    make_task, make_task_run
) -> None:
    matching = _all_capabilities(
        requires_evaluation_extractor=True,
        extractor_version="entry-extractor-v1",
    )
    matched = score_task(
        make_task(),
        make_task_run(),
        matching,
        _config("state_scores.final_state_accuracy", strict=True),
    )
    assert matched.state_scores.final_state_accuracy == 1.0

    mismatched = matching.validated_replace(extractor_version="other-extractor-v1")
    with pytest.raises(ValueError, match="extractor provenance mismatch"):
        score_task(
            make_task(),
            make_task_run(),
            mismatched,
            _config("state_scores.final_state_accuracy", strict=True),
        )
    with pytest.raises(ValueError, match="extractor provenance mismatch"):
        score_task(
            make_task(),
            make_task_run(),
            mismatched,
            _config(strict=True),
        )
    nonstrict = score_task(
        make_task(),
        make_task_run(),
        mismatched,
        _config(
            "protocol_scores.action_parse_valid",
            "state_scores.final_state_accuracy",
            strict=False,
        ),
    )
    assert nonstrict.state_scores.final_state_accuracy is None
    assert nonstrict.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert nonstrict.protocol_scores.action_parse_valid is False

    all_nonstrict = score_task(
        make_task(),
        make_task_run(),
        mismatched,
        _config(strict=False),
    )
    assert all_nonstrict.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert all_nonstrict.answer_scores.exact_match == 1.0

    answer_only = score_task(
        make_task(),
        make_task_run(),
        mismatched,
        _config("answer_scores.exact_match", strict=True),
    )
    assert answer_only.answer_scores.exact_match == 1.0


def test_extractor_version_coherence_and_model_construct_bypass_are_rejected(
    make_task, make_task_run
) -> None:
    for invalid_version in (None, " ", " entry-extractor-v1"):
        incoherent = AdapterCapabilities(
            exports_entries=True,
            requires_evaluation_extractor=True,
            extractor_version=invalid_version,
        )
        with pytest.raises(ValueError, match="extractor_version"):
            score_task(make_task(), make_task_run(), incoherent, _config(strict=False))

    payload = _all_capabilities().model_dump(mode="python")
    payload["requires_evaluation_extractor"] = True
    payload["extractor_version"] = None
    bypassed = AdapterCapabilities.model_construct(**payload)
    with pytest.raises(ValueError, match="extractor_version"):
        score_task(make_task(), make_task_run(), bypassed, _config(strict=False))


def test_missing_run_extractor_version_is_strict_error_or_nonstrict_missing_artifact(
    make_task, make_task_run
) -> None:
    run_payload = make_task_run().model_dump(mode="python")
    run_payload["parser_extractor_provenance"]["memory_entry_extractor_version"] = ""
    run = TaskRunRecord.model_validate(run_payload)
    capabilities = _all_capabilities(
        requires_evaluation_extractor=True,
        extractor_version="entry-extractor-v1",
    )
    with pytest.raises(ValueError, match="extractor provenance mismatch"):
        score_task(
            make_task(),
            run,
            capabilities,
            _config("state_scores.final_state_accuracy", strict=True),
        )
    nonstrict = score_task(
        make_task(),
        run,
        capabilities,
        _config("state_scores.final_state_accuracy", strict=False),
    )
    assert nonstrict.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_raw_state_only_cannot_score_normalized_state_metrics(
    make_task, make_task_run
) -> None:
    state_paths = tuple(
        f"state_scores.{field}"
        for field in SCORE_LAYER_TYPES["state_scores"].model_fields
    )
    raw_only = AdapterCapabilities(exports_raw_state=True)
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    task_payload["task_family"] = "deletion_forgetting"
    task = type(base_task).model_validate(task_payload)
    score = score_task(
        task,
        make_task_run(),
        raw_only,
        _config(*state_paths, strict=False),
    )
    for path in state_paths:
        layer, leaf = path.split(".")
        assert getattr(getattr(score, layer), leaf) is None
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_SUPPORTED
    assert "current_state_missing" not in score.failure_flags
    assert score.audit_scores.state_export_available is True

    no_export = score_task(
        make_task(),
        make_task_run(),
        AdapterCapabilities(),
        _config("audit_scores.state_export_available", strict=False),
    )
    assert no_export.audit_scores.state_export_available is False
    assert "audit_scores.state_export_available" not in no_export.supported_metric_fields

    with pytest.raises(ValueError, match="requested metrics require unsupported"):
        score_task(
            make_task(),
            make_task_run(),
            raw_only,
            _config("state_scores.final_state_accuracy", strict=True),
        )


@pytest.mark.parametrize(
    "capabilities",
    (
        AdapterCapabilities(exports_entries=True),
        AdapterCapabilities(
            supports_isolated_reset=True,
            exports_entries=True,
            exports_object_keys=True,
        ),
        AdapterCapabilities(
            supports_isolated_reset=True,
            exports_entries=True,
            exports_values=True,
        ),
        AdapterCapabilities(
            exports_entries=True,
            exports_object_keys=True,
            exports_values=True,
        ),
    ),
)
def test_state_metrics_require_reset_entries_and_structured_content_or_extractor(
    make_task, make_task_run, capabilities
) -> None:
    score = score_task(
        make_task(),
        make_task_run(),
        capabilities,
        _config("state_scores.final_state_accuracy", strict=False),
    )
    assert score.state_scores.final_state_accuracy is None
    assert score.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.NOT_SUPPORTED


def test_structured_level_two_entries_can_score_state_metrics(
    make_task, make_task_run
) -> None:
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        make_task_run(),
        capabilities,
        _config("state_scores.final_state_accuracy", strict=True),
    )
    assert score.state_scores.final_state_accuracy == 1.0


def test_versioned_extractor_level_two_entries_can_score_state_metrics(
    make_task, make_task_run
) -> None:
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        requires_evaluation_extractor=True,
        extractor_version="entry-extractor-v1",
    )
    score = score_task(
        make_task(),
        make_task_run(),
        capabilities,
        _config("state_scores.final_state_accuracy", strict=True),
    )
    assert score.state_scores.final_state_accuracy == 1.0


def test_state_capability_with_missing_snapshot_is_missing_artifact(
    make_task, make_task_run
) -> None:
    run = _replace_run(make_task_run(), memory_snapshots=[])
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        run,
        capabilities,
        _config("state_scores.final_state_accuracy", strict=False),
    )
    assert score.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_answer_state_consistency_uses_normalized_level_two_authority(
    make_task, make_task_run
) -> None:
    raw_only = AdapterCapabilities(exports_raw_state=True)
    unsupported = score_task(
        make_task(),
        make_task_run(),
        raw_only,
        _config("answer_scores.answer_state_consistency", strict=False),
    )
    assert unsupported.answer_scores.answer_state_consistency is None
    assert unsupported.supported_metric_fields[
        "answer_scores.answer_state_consistency"
    ].reason is SupportReason.NOT_SUPPORTED

    structured = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
    )
    consistent = score_task(
        make_task(),
        make_task_run(),
        structured,
        _config("answer_scores.answer_state_consistency", strict=True),
    )
    assert consistent.answer_scores.answer_state_consistency == 1.0

    run_payload = make_task_run().model_dump(mode="python")
    run_payload["memory_snapshots"][0]["state_by_object"] = {}
    missing_state = score_task(
        make_task(),
        TaskRunRecord.model_validate(run_payload),
        structured,
        _config("state_scores.final_state_accuracy", strict=True),
    )
    assert missing_state.state_scores.final_state_accuracy == 0.0
    assert "current_state_missing" in missing_state.failure_flags


def test_store_metric_capabilities_are_authoritative_and_metric_specific(
    make_task, make_task_run
) -> None:
    paths = (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    )
    entry_only = AdapterCapabilities(exports_entries=True)
    entry_score = score_task(
        make_task(),
        make_task_run(),
        entry_only,
        _config("store_scores.final_memory_size", *paths, strict=False),
    )
    assert entry_score.store_scores.final_memory_size == 1
    for path in paths:
        assert entry_score.supported_metric_fields[path].reason is SupportReason.NOT_SUPPORTED

    key_and_order = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
    )
    key_score = score_task(
        make_task(),
        make_task_run(),
        key_and_order,
        _config(*paths, strict=False),
    )
    assert key_score.store_scores.obsolete_version_count == 0
    for path in paths[1:]:
        assert key_score.supported_metric_fields[path].reason is SupportReason.NOT_SUPPORTED

    structured_linkage = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_source_event_ids=True,
        exports_object_keys=True,
        exports_values=True,
    )
    structured_score = score_task(
        make_task(),
        make_task_run(),
        structured_linkage,
        _config(*paths, strict=True),
    )
    assert structured_score.store_scores.obsolete_version_count == 0
    assert structured_score.store_scores.stale_conflicting_value_count == 0
    assert structured_score.store_scores.duplicate_current_count == 0

    extractor_linkage = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        requires_evaluation_extractor=True,
        extractor_version="entry-extractor-v1",
    )
    extractor_score = score_task(
        make_task(),
        make_task_run(),
        extractor_linkage,
        _config(*paths, strict=True),
    )
    assert extractor_score.store_scores.obsolete_version_count == 0
    assert extractor_score.store_scores.stale_conflicting_value_count == 0
    assert extractor_score.store_scores.duplicate_current_count == 0


def test_store_order_scoring_does_not_consume_undeclared_source_linkage(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=["event_0"],
        version_index=None,
    )
    current["source_event_ids"] = ["event_1"]
    current["version_index"] = None
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config(
            "store_scores.obsolete_version_count",
            "store_scores.stale_conflicting_value_count",
            "store_scores.duplicate_current_count",
            strict=True,
        ),
    )
    for path in (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_store_single_entry_requires_declared_order_artifact(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    entry = payload["memory_snapshots"][0]["entries"][0]
    entry["version_index"] = None
    entry["created_at"] = None
    entry["updated_at"] = None
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.obsolete_version_count", strict=True),
    )
    assert score.supported_metric_fields[
        "store_scores.obsolete_version_count"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_store_timestamp_order_path_works_without_source_linkage(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=[],
        version_index=0,
    )
    current["source_event_ids"] = []
    current["version_index"] = 1
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config(
            "store_scores.obsolete_version_count",
            "store_scores.stale_conflicting_value_count",
            "store_scores.duplicate_current_count",
            strict=True,
        ),
    )
    assert score.store_scores.obsolete_version_count == 1
    assert score.store_scores.stale_conflicting_value_count == 1
    assert score.store_scores.duplicate_current_count == 0


def test_store_timestamp_order_normalizes_timezone_offsets(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=[],
        version_index=None,
        created_at="2026-01-01T01:00:00+02:00",
        updated_at="2026-01-01T01:00:00+02:00",
    )
    current.update(
        source_event_ids=[],
        version_index=None,
        created_at="2025-12-31T23:30:00Z",
        updated_at="2025-12-31T23:30:00Z",
    )
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.stale_conflicting_value_count", strict=True),
    )
    assert score.store_scores.stale_conflicting_value_count == 1


def test_invalid_declared_timestamp_falls_back_to_declared_source_linkage(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=["event_0"],
        version_index=None,
        created_at="event_9",
        updated_at="event_9",
    )
    current.update(
        source_event_ids=["event_1"],
        version_index=None,
        created_at="event_10",
        updated_at="event_10",
    )
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_source_event_ids=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.stale_conflicting_value_count", strict=True),
    )
    assert score.store_scores.stale_conflicting_value_count == 1


def test_valid_declared_timestamp_precedes_declared_source_linkage(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=["event_1"],
        version_index=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    current.update(
        source_event_ids=["event_0"],
        version_index=None,
        created_at="2026-01-02T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_source_event_ids=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.stale_conflicting_value_count", strict=True),
    )
    assert score.store_scores.stale_conflicting_value_count == 1


@pytest.mark.parametrize("strict", (False, True))
def test_store_timestamp_order_rejects_opaque_or_naive_values(
    make_task, make_task_run, strict
) -> None:
    for stale_timestamp, current_timestamp in (
        ("event_9", "event_10"),
        ("2026-01-01T00:00:00", "2026-01-02T00:00:00"),
    ):
        payload = make_task_run().model_dump(mode="python")
        current = payload["memory_snapshots"][0]["entries"][0]
        stale = deepcopy(current)
        stale.update(
            entry_id="entry_stale",
            value_candidate="Dalian",
            content="friend:alex.location = Dalian",
            source_event_ids=[],
            version_index=None,
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        current.update(
            source_event_ids=[],
            version_index=None,
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )
        payload["memory_snapshots"][0]["entries"] = [stale, current]
        payload["memory_snapshots"][0]["store_size"] = 2
        capabilities = AdapterCapabilities(
            supports_isolated_reset=True,
            exports_entries=True,
            exports_timestamps_or_order=True,
            exports_object_keys=True,
            exports_values=True,
        )
        score = score_task(
            make_task(),
            TaskRunRecord.model_validate(payload),
            capabilities,
            _config("store_scores.obsolete_version_count", strict=strict),
        )
        assert score.supported_metric_fields[
            "store_scores.obsolete_version_count"
        ].reason is SupportReason.MISSING_ARTIFACT


@pytest.mark.parametrize("capability_path", ("native", "extractor"))
@pytest.mark.parametrize("strict", (False, True))
def test_store_object_metrics_require_complete_per_entry_object_linkage(
    make_task, make_task_run, capability_path, strict
) -> None:
    payload = make_task_run().model_dump(mode="python")
    unlinked = deepcopy(payload["memory_snapshots"][0]["entries"][0])
    unlinked.update(
        entry_id="entry_unlinked",
        object_key_candidate=None,
        value_candidate="Dalian",
        version_index=0,
        source_event_ids=["event_0"],
    )
    payload["memory_snapshots"][0]["entries"].insert(0, unlinked)
    payload["memory_snapshots"][0]["store_size"] = 2
    common = {
        "supports_isolated_reset": True,
        "exports_entries": True,
        "exports_timestamps_or_order": True,
    }
    capabilities = (
        AdapterCapabilities(
            **common,
            exports_object_keys=True,
            exports_values=True,
        )
        if capability_path == "native"
        else AdapterCapabilities(
            **common,
            requires_evaluation_extractor=True,
            extractor_version="entry-extractor-v1",
        )
    )
    paths = (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.final_memory_size", *paths, strict=strict),
    )
    assert score.store_scores.final_memory_size == 2
    for path in paths:
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


@pytest.mark.parametrize("capability_path", ("native", "extractor"))
def test_unrelated_missing_value_does_not_null_target_store_metrics(
    make_task, make_task_run, capability_path
) -> None:
    payload = make_task_run().model_dump(mode="python")
    unrelated = deepcopy(payload["memory_snapshots"][0]["entries"][0])
    unrelated["entry_id"] = "entry_unrelated"
    unrelated["object_key_candidate"]["entity"] = "friend:bob"
    unrelated["value_candidate"] = None
    unrelated["version_index"] = 0
    unrelated["source_event_ids"] = ["event_0"]
    payload["memory_snapshots"][0]["entries"].insert(0, unrelated)
    payload["memory_snapshots"][0]["store_size"] = 2
    common = {
        "supports_isolated_reset": True,
        "exports_entries": True,
        "exports_timestamps_or_order": True,
    }
    capabilities = (
        AdapterCapabilities(
            **common,
            exports_object_keys=True,
            exports_values=True,
        )
        if capability_path == "native"
        else AdapterCapabilities(
            **common,
            requires_evaluation_extractor=True,
            extractor_version="entry-extractor-v1",
        )
    )
    paths = (
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config(*paths, strict=True),
    )
    assert score.store_scores.stale_conflicting_value_count == 0
    assert score.store_scores.duplicate_current_count == 0
    for path in paths:
        assert path not in score.supported_metric_fields


@pytest.mark.parametrize("capability_path", ("native", "extractor"))
@pytest.mark.parametrize("strict", (False, True))
def test_store_value_metrics_require_complete_values_but_obsolete_remains_observable(
    make_task, make_task_run, capability_path, strict
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    missing_value = deepcopy(current)
    missing_value.update(
        entry_id="entry_missing_value",
        value_candidate=None,
        version_index=0,
        source_event_ids=["event_0"],
    )
    current["version_index"] = 1
    current["source_event_ids"] = ["event_1"]
    payload["memory_snapshots"][0]["entries"] = [missing_value, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    common = {
        "supports_isolated_reset": True,
        "exports_entries": True,
        "exports_timestamps_or_order": True,
    }
    capabilities = (
        AdapterCapabilities(
            **common,
            exports_object_keys=True,
            exports_values=True,
        )
        if capability_path == "native"
        else AdapterCapabilities(
            **common,
            requires_evaluation_extractor=True,
            extractor_version="entry-extractor-v1",
        )
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config(
            "store_scores.obsolete_version_count",
            "store_scores.stale_conflicting_value_count",
            "store_scores.duplicate_current_count",
            strict=strict,
        ),
    )
    assert score.store_scores.obsolete_version_count == 1
    for path in (
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_store_source_linkage_ignores_undeclared_version_order(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    current = payload["memory_snapshots"][0]["entries"][0]
    stale = deepcopy(current)
    stale.update(
        entry_id="entry_stale",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=["event_0"],
        version_index=2,
    )
    current["source_event_ids"] = ["event_1"]
    current["version_index"] = 0
    payload["memory_snapshots"][0]["entries"] = [stale, current]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_source_event_ids=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.stale_conflicting_value_count", strict=True),
    )
    assert score.store_scores.stale_conflicting_value_count == 1


def test_wrong_latest_legacy_stale_count_differs_from_canonical_obsolete_conflict(
    make_task, make_task_run
) -> None:
    payload = make_task_run().model_dump(mode="python")
    old_gold = payload["memory_snapshots"][0]["entries"][0]
    old_gold["entry_id"] = "entry_old_gold"
    old_gold["source_event_ids"] = ["event_0"]
    old_gold["version_index"] = 0
    wrong_latest = deepcopy(old_gold)
    wrong_latest.update(
        entry_id="entry_wrong_latest",
        value_candidate="Dalian",
        content="friend:alex.location = Dalian",
        source_event_ids=["event_1"],
        version_index=1,
    )
    payload["memory_snapshots"][0]["entries"] = [old_gold, wrong_latest]
    payload["memory_snapshots"][0]["store_size"] = 2
    capabilities = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
    )
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(payload),
        capabilities,
        _config("store_scores.stale_conflicting_value_count", strict=True),
    )
    legacy_wrong_value_count = sum(
        entry["value_candidate"] != "Qingdao"
        for entry in payload["memory_snapshots"][0]["entries"]
    )
    assert legacy_wrong_value_count == 1
    assert score.store_scores.stale_conflicting_value_count == 0


def test_strict_explicit_capability_request_fails_early_and_non_strict_is_null(
    make_task, make_task_run
) -> None:
    capabilities = _all_capabilities(exports_retrieval_scores=False)
    with pytest.raises(ValueError, match="exports_retrieval_scores"):
        score_task(
            make_task(),
            make_task_run(),
            capabilities,
            _config("retrieval_scores.current_mrr", strict=True),
        )
    score = score_task(
        make_task(),
        make_task_run(),
        capabilities,
        _config("retrieval_scores.current_mrr", strict=False),
    )
    support = score.supported_metric_fields["retrieval_scores.current_mrr"]
    assert score.retrieval_scores.current_mrr is None
    assert support.reason is SupportReason.NOT_SUPPORTED
    assert support.null_policy == "exclude_from_aggregation"


def test_capability_true_but_artifact_missing_is_missing_artifact(
    make_task, make_task_run
) -> None:
    run = _replace_answer(
        _replace_run(make_task_run(), retrieval_traces=[]),
        cited_entry_ids=[],
    )
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("retrieval_scores.current_mrr"),
    )
    support = score.supported_metric_fields["retrieval_scores.current_mrr"]
    assert support.reason is SupportReason.MISSING_ARTIFACT
    assert "retrieval" in (support.detail or "").lower()


def test_rank_metric_requires_normalized_rank_artifact(make_task, make_task_run) -> None:
    payload = make_task_run().model_dump(mode="python")
    payload["retrieval_traces"][0]["ranks"] = []
    run = TaskRunRecord.model_validate(payload)
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("retrieval_scores.current_mrr"),
    )
    assert score.retrieval_scores.current_mrr is None
    assert score.supported_metric_fields[
        "retrieval_scores.current_mrr"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_audit_provenance_coverage_honors_capability_registry(make_task, make_task_run) -> None:
    score = score_task(
        make_task(),
        make_task_run(),
        _all_capabilities(exports_source_event_ids=False),
        _config("audit_scores.source_provenance_coverage"),
    )
    assert score.audit_scores.source_provenance_coverage is None
    assert score.supported_metric_fields[
        "audit_scores.source_provenance_coverage"
    ].reason is SupportReason.NOT_SUPPORTED


def test_query_level_nonapplicability_precedes_capability_and_missing_artifact(
    make_task, make_task_run
) -> None:
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    task_payload["queries"][0]["query_type"] = "historical_state"
    task = type(base_task).model_validate(task_payload)
    score = score_task(
        task,
        make_task_run(),
        AdapterCapabilities(),
        _config("retrieval_scores.current_mrr", strict=True),
    )
    assert score.retrieval_scores.current_mrr is None
    assert score.supported_metric_fields[
        "retrieval_scores.current_mrr"
    ].reason is SupportReason.NOT_APPLICABLE


def test_family_nonapplicability_precedes_capability_and_artifact_checks(
    make_task, make_task_run
) -> None:
    base_task = make_task()
    task_payload = base_task.model_dump(mode="python")
    task_payload["task_family"] = "noop_write_discipline"
    task = type(base_task).model_validate(task_payload)
    score = score_task(
        task,
        make_task_run(),
        AdapterCapabilities(),
        _config("state_scores.expected_absence_accuracy"),
    )
    support = score.supported_metric_fields["state_scores.expected_absence_accuracy"]
    assert support.reason is SupportReason.NOT_APPLICABLE
    assert support.null_policy == "exclude_from_aggregation"


@pytest.mark.parametrize("status", [CompletionStatus.FAILED, CompletionStatus.PARTIAL])
def test_completion_failure_keeps_row_and_marks_accuracy_runtime_failed_not_zero(
    make_task, make_task_run, status
) -> None:
    run = make_task_run(status=status, exception_type="timeout")
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("action_scores.operation_accuracy", "system_scores.error_rate"),
    )
    assert score.completion_status is status
    assert score.action_scores.operation_accuracy is None
    support = score.supported_metric_fields["action_scores.operation_accuracy"]
    assert support.reason is SupportReason.RUNTIME_FAILED
    assert "system_exception" in score.failure_flags
    assert score.primary_failure == "system_exception"
    assert score.system_scores.error_rate == 1.0


def test_not_supported_completion_keeps_row_with_not_supported_metrics(
    make_task, make_task_run
) -> None:
    run = make_task_run(status=CompletionStatus.NOT_SUPPORTED)
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("action_scores.operation_accuracy"),
    )
    assert score.completion_status is CompletionStatus.NOT_SUPPORTED
    assert score.supported_metric_fields[
        "action_scores.operation_accuracy"
    ].reason is SupportReason.NOT_SUPPORTED


def test_no_answer_prediction_is_missing_artifact_not_invented_zero(
    make_task, make_task_run
) -> None:
    run = _replace_run(make_task_run(), answer_predictions=[])
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("answer_scores.exact_match"),
    )
    assert score.answer_scores.exact_match is None
    assert score.supported_metric_fields[
        "answer_scores.exact_match"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_answer_prediction_without_parsed_artifact_is_missing_not_wrong(
    make_task, make_task_run
) -> None:
    run = _replace_answer(make_task_run(), parsed_answer=None, format_valid=False)
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("answer_scores.exact_match"),
    )
    assert score.answer_scores.exact_match is None
    assert score.supported_metric_fields[
        "answer_scores.exact_match"
    ].reason is SupportReason.MISSING_ARTIFACT


def test_current_answer_wrong_stale_copy_uses_parsed_normalized_artifact_only(
    make_task, make_task_run
) -> None:
    run = _replace_answer(
        _complete_action_trace(make_task_run()),
        raw_output="raw text that must not be reparsed as Qingdao",
        parsed_answer="Dalian",
        format_valid=True,
    )
    score = score_task(make_task(), run, _all_capabilities(), _config())
    assert score.answer_scores.exact_match == 0.0
    assert score.answer_scores.normalized_match == 0.0
    assert score.answer_scores.stale_copied == 1.0
    assert "stale_copied" in score.failure_flags
    assert score.primary_failure == "stale_copied"


def test_semantically_correct_invalid_answer_is_format_only_without_correct_flag(
    make_task, make_task_run
) -> None:
    run = _replace_answer(
        _complete_action_trace(make_task_run()),
        raw_output="noncanonical wrapper",
        parsed_answer="Qingdao",
        format_valid=False,
    )
    score = score_task(make_task(), run, _all_capabilities(), _config())
    assert score.answer_scores.exact_match == 1.0
    assert score.protocol_scores.answer_parse_valid is False
    assert score.failure_flags == ("answer_format_only",)
    assert score.primary_failure == "answer_format_only"
    assert "correct" not in score.failure_flags


def test_native_mode_drops_unknown_legacy_values_and_explicit_namespace_preserves_only_its_values(
    make_task, make_task_run
) -> None:
    run = _replace_run(
        make_task_run(),
        system_events=[
            {
                "legacy_metrics": {
                    "legacy_p63.unknown_old_metric": 0.25,
                    "legacy_p84.other_metric": 0.75,
                    "unnamespaced": 1.0,
                }
            }
        ],
    )
    native = score_task(make_task(), run, _all_capabilities(), _config())
    compatible = score_task(
        make_task(), run, _all_capabilities(), _config(legacy="legacy_p63")
    )
    assert dict(native.legacy_metrics) == {}
    assert dict(compatible.legacy_metrics) == {"legacy_p63.unknown_old_metric": 0.25}


def test_score_task_defensively_revalidates_constructed_normalization_profiles(
    make_task, make_task_run
) -> None:
    config = ScorerConfig.model_construct(
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="unknown_answer_profile",
        requested_metric_fields=("action_scores.operation_accuracy",),
        strict_capability_check=False,
    )
    with pytest.raises(ValidationError, match="answer_normalization_profile"):
        score_task(make_task(), make_task_run(), _all_capabilities(), config)


def test_completed_status_with_runtime_exception_does_not_enter_accuracy_denominator(
    make_task, make_task_run
) -> None:
    run = make_task_run(status=CompletionStatus.COMPLETED, exception_type="late_failure")
    score = score_task(
        make_task(),
        run,
        _all_capabilities(),
        _config("action_scores.operation_accuracy"),
    )
    assert score.action_scores.operation_accuracy is None
    assert score.supported_metric_fields[
        "action_scores.operation_accuracy"
    ].reason is SupportReason.RUNTIME_FAILED


def test_score_task_rejects_mismatched_schema_and_runtime_versions(make_task, make_task_run) -> None:
    task = make_task()
    task_payload = task.model_dump(mode="python")
    task_payload["schema_version"] = "9.0.0"
    with pytest.raises(ValueError, match="schema_version"):
        score_task(
            type(task).model_validate(task_payload),
            make_task_run(),
            _all_capabilities(),
            _config(),
        )

    run_payload = make_task_run().model_dump(mode="python")
    run_payload["runtime_record_version"] = "9.0.0"
    with pytest.raises(ValueError, match="runtime_record_version"):
        score_task(
            make_task(),
            TaskRunRecord.model_validate(run_payload),
            _all_capabilities(),
            _config(),
        )


def test_defensive_validation_rejects_model_constructed_invalid_config_and_runtime(
    make_task, make_task_run
) -> None:
    task = make_task()
    run = make_task_run()
    invalid_config = ScorerConfig.model_construct(
        scorer_version="wrong",
        metric_registry_version="wrong",
        value_normalization_profile="typed_exact_v1",
        answer_normalization_profile="normalized_exact_v1",
        primary_failure_precedence_version="wrong",
        requested_metric_fields=("answer_scores.exact_match",),
        legacy_compatibility_mode=None,
        strict_capability_check=False,
    )
    with pytest.raises(ValidationError):
        score_task(task, run, _all_capabilities(), invalid_config)

    action_data = run.parsed_actions[0].model_dump(mode="python")
    action_data["format_valid"] = "true"
    action_data["latency_ms"] = float("nan")
    invalid_action = ParsedManagerAction.model_construct(**action_data)
    invalid_run = TaskRunRecord.model_construct(
        **{
            **run.model_dump(mode="python"),
            "parsed_actions": [invalid_action],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError):
            score_task(task, invalid_run, _all_capabilities(), _config())


def test_primary_failure_requires_completed_successful_correctness_evidence(
    make_task, make_task_run
) -> None:
    complete = _complete_action_trace(make_task_run())
    missing_retrieval = _replace_answer(
        _replace_run(complete, retrieval_traces=[]),
        cited_entry_ids=[],
    )
    missing_score = score_task(
        make_task(), missing_retrieval, _all_capabilities(), _config()
    )
    assert missing_score.failure_flags == ()
    assert missing_score.primary_failure is None

    wrong = _replace_answer(
        _replace_run(complete, retrieval_traces=[]),
        parsed_answer="unattributed wrong answer",
        cited_entry_ids=[],
    )
    wrong_score = score_task(
        make_task(),
        wrong,
        _all_capabilities(),
        _config("answer_scores.exact_match"),
    )
    assert wrong_score.failure_flags == ()
    assert wrong_score.answer_scores.exact_match == 0.0
    assert wrong_score.primary_failure is None

    unrelated_subset = score_task(
        make_task(),
        wrong,
        _all_capabilities(),
        _config("store_scores.final_memory_size"),
    )
    assert unrelated_subset.failure_flags == ()
    assert unrelated_subset.primary_failure is None

    correct_full = score_task(
        make_task(), complete, _all_capabilities(), _config()
    )
    correct_subset = score_task(
        make_task(),
        complete,
        _all_capabilities(),
        _config("store_scores.final_memory_size"),
    )
    assert correct_full.primary_failure == "correct"
    assert correct_subset.primary_failure is None

    no_state = score_task(
        make_task(),
        complete,
        _all_capabilities(exports_entries=False, exports_raw_state=False),
        _config(
            "action_scores.full_action_exact_match",
            "answer_scores.normalized_match",
        ),
    )
    assert no_state.primary_failure is None

    task_payload = make_task().model_dump(mode="python")
    for event in task_payload["events"]:
        event["gold_action_ids"] = []
    task_payload["gold"]["actions"] = []
    task_payload["gold"]["action_sequence"] = []
    zero_action_task = type(make_task()).model_validate(task_payload)
    unsupported = _replace_run(
        make_task_run(status=CompletionStatus.NOT_SUPPORTED),
        parsed_actions=[],
        retrieval_traces=[],
        answer_predictions=[],
    )
    unsupported_score = score_task(
        zero_action_task,
        unsupported,
        AdapterCapabilities(),
        _config(strict=False),
    )
    assert unsupported_score.failure_flags == ()
    assert unsupported_score.primary_failure is None


def test_runtime_integrity_rejects_duplicate_unknown_and_cross_linked_artifacts(
    make_task, make_task_run
) -> None:
    task = make_task()
    base = make_task_run().model_dump(mode="python")
    cases = []
    duplicate_trace = deepcopy(base)
    duplicate_trace["retrieval_traces"].append(deepcopy(duplicate_trace["retrieval_traces"][0]))
    cases.append(duplicate_trace)
    unknown_answer = deepcopy(base)
    unknown_answer["answer_predictions"][0]["query_id"] = "unknown_query"
    cases.append(unknown_answer)
    duplicate_entry = deepcopy(base)
    duplicate_entry["memory_snapshots"][0]["entries"].append(
        deepcopy(duplicate_entry["memory_snapshots"][0]["entries"][0])
    )
    duplicate_entry["memory_snapshots"][0]["store_size"] = 2
    cases.append(duplicate_entry)
    unknown_source = deepcopy(base)
    unknown_source["retrieval_traces"][0]["retrieved_entries"][0]["source_event_ids"] = [
        "unknown_event"
    ]
    cases.append(unknown_source)
    bad_citation = deepcopy(base)
    bad_citation["answer_predictions"][0]["cited_entry_ids"] = ["other_entry"]
    cases.append(bad_citation)
    for payload in cases:
        with pytest.raises(ValueError):
            score_task(
                task,
                TaskRunRecord.model_validate(payload),
                _all_capabilities(),
                _config(),
            )


def test_two_actions_for_one_event_pair_by_occurrence_order(make_task, make_task_run) -> None:
    task_payload = make_task().model_dump(mode="python")
    second = deepcopy(task_payload["gold"]["actions"][0])
    second["action_id"] = "action_0_second"
    task_payload["gold"]["actions"].insert(1, second)
    task_payload["gold"]["action_sequence"] = [
        "action_0",
        "action_0_second",
        "action_1",
    ]
    task_payload["events"][0]["gold_action_ids"] = ["action_0", "action_0_second"]
    task = type(make_task()).model_validate(task_payload)
    run = _complete_action_trace(make_task_run())
    run_payload = run.model_dump(mode="python")
    run_payload["parsed_actions"].insert(1, deepcopy(run_payload["parsed_actions"][0]))
    score = score_task(
        task,
        TaskRunRecord.model_validate(run_payload),
        _all_capabilities(),
        _config(
            "action_scores.operation_accuracy",
            "action_scores.full_action_exact_match",
        ),
    )
    assert score.action_scores.operation_accuracy == 1.0
    assert score.action_scores.full_action_exact_match == 1.0


def test_multi_target_action_nulls_only_target_dependent_metrics(make_task, make_task_run) -> None:
    task_payload = make_task().model_dump(mode="python")
    second_key = deepcopy(task_payload["target_objects"][0])
    second_key["entity"] = "friend:bob"
    task_payload["target_objects"].append(second_key)
    task_payload["events"][0]["gold_action_ids"] = []
    task_payload["events"][1]["gold_action_ids"] = ["action_1"]
    action = task_payload["gold"]["actions"][1]
    action["operation"] = "DELETE"
    action["target_object_keys"] = [task_payload["target_objects"][0], second_key]
    action["value"] = None
    task_payload["gold"]["actions"] = [action]
    task_payload["gold"]["action_sequence"] = ["action_1"]
    task = type(make_task()).model_validate(task_payload)
    run_payload = make_task_run().model_dump(mode="python")
    predicted = run_payload["parsed_actions"][0]
    predicted["operation"] = "DELETE"
    predicted["value"] = None
    score = score_task(
        task,
        TaskRunRecord.model_validate(run_payload),
        _all_capabilities(),
        _config(
            "action_scores.operation_accuracy",
            "action_scores.value_accuracy",
            "action_scores.object_key_accuracy",
            "action_scores.full_action_exact_match",
        ),
    )
    assert score.action_scores.operation_accuracy == 1.0
    assert score.action_scores.value_accuracy == 1.0
    for path in (
        "action_scores.object_key_accuracy",
        "action_scores.full_action_exact_match",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE

    target_only = score_task(
        task,
        TaskRunRecord.model_validate(run_payload),
        AdapterCapabilities(),
        _config("action_scores.object_key_accuracy", strict=True),
    )
    assert target_only.supported_metric_fields[
        "action_scores.object_key_accuracy"
    ].reason is SupportReason.NOT_APPLICABLE
    with pytest.raises(ValueError, match="exports_action_trace"):
        score_task(
            task,
            TaskRunRecord.model_validate(run_payload),
            AdapterCapabilities(),
            _config("action_scores.operation_accuracy", strict=True),
        )


def test_zero_gold_actions_are_not_applicable_and_extraneous_actions_rejected(
    make_task, make_task_run
) -> None:
    task_payload = make_task().model_dump(mode="python")
    for event in task_payload["events"]:
        event["gold_action_ids"] = []
    task_payload["gold"]["actions"] = []
    task_payload["gold"]["action_sequence"] = []
    task = type(make_task()).model_validate(task_payload)
    empty_run = _replace_run(make_task_run(), parsed_actions=[])
    score = score_task(task, empty_run, _all_capabilities(), _config(strict=False))
    for path in (
        "protocol_scores.action_parse_valid",
        "protocol_scores.execution_success_rate",
        "protocol_scores.unsupported_operation_rate",
        "protocol_scores.fallback_rate",
        "action_scores.operation_accuracy",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE
    assert score.protocol_scores.answer_parse_valid is True

    invalid_answer = _replace_answer(empty_run, format_valid=False)
    invalid_score = score_task(
        task, invalid_answer, _all_capabilities(), _config(strict=False)
    )
    assert invalid_score.protocol_scores.answer_parse_valid is False

    no_answer = _replace_run(empty_run, answer_predictions=[])
    missing_score = score_task(
        task, no_answer, _all_capabilities(), _config(strict=False)
    )
    assert missing_score.protocol_scores.answer_parse_valid is None
    assert missing_score.supported_metric_fields[
        "protocol_scores.answer_parse_valid"
    ].reason is SupportReason.MISSING_ARTIFACT
    with pytest.raises(ValueError, match="surplus parsed action"):
        score_task(task, make_task_run(), _all_capabilities(), _config(strict=False))


def test_snapshot_integrity_and_size_mismatch_semantics(make_task, make_task_run) -> None:
    task = make_task()
    base = make_task_run().model_dump(mode="python")
    for after_ids in (
        ["event_1", "event_0"],
        ["event_0", "event_0"],
        ["unknown_event"],
        [None, "event_1"],
    ):
        payload = deepcopy(base)
        snapshot = payload["memory_snapshots"][0]
        payload["memory_snapshots"] = []
        for after_id in after_ids:
            item = deepcopy(snapshot)
            item["after_event_id"] = after_id
            payload["memory_snapshots"].append(item)
        with pytest.raises(ValueError):
            score_task(
                task,
                TaskRunRecord.model_validate(payload),
                _all_capabilities(),
                _config(),
            )

    valid = deepcopy(base)
    latest = deepcopy(valid["memory_snapshots"][0])
    earlier = deepcopy(latest)
    earlier["after_event_id"] = "event_0"
    earlier["state_by_object"][next(iter(earlier["state_by_object"]))] = "Dalian"
    earlier["entries"][0]["value_candidate"] = "Dalian"
    earlier["entries"][0]["source_event_ids"] = ["event_0"]
    earlier["entries"][0]["version_index"] = 0
    valid["memory_snapshots"] = [earlier, latest]
    selected = score_task(
        task,
        TaskRunRecord.model_validate(valid),
        _all_capabilities(),
        _config("state_scores.final_state_accuracy"),
    )
    assert selected.state_scores.final_state_accuracy == 1.0

    prefinal = deepcopy(base)
    prefinal["memory_snapshots"][0] = earlier
    completed_prefinal = score_task(
        task,
        TaskRunRecord.model_validate(prefinal),
        _all_capabilities(),
        _config(
            "state_scores.final_state_accuracy",
            "store_scores.final_memory_size",
        ),
    )
    for path in (
        "state_scores.final_state_accuracy",
        "store_scores.final_memory_size",
    ):
        assert completed_prefinal.supported_metric_fields[
            path
        ].reason is SupportReason.MISSING_ARTIFACT
    assert "current_state_missing" not in completed_prefinal.failure_flags
    assert "stale_retained" not in completed_prefinal.failure_flags
    assert completed_prefinal.primary_failure is None

    partial_payload = deepcopy(prefinal)
    partial_payload["completion_status"] = "partial"
    partial = score_task(
        task,
        TaskRunRecord.model_validate(partial_payload),
        _all_capabilities(),
        _config(
            "state_scores.final_state_accuracy",
            "store_scores.final_memory_size",
        ),
    )
    assert partial.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.RUNTIME_FAILED
    assert partial.store_scores.final_memory_size == 1
    assert partial.primary_failure != "correct"

    unlinked = deepcopy(base)
    unlinked["memory_snapshots"][0]["after_event_id"] = None
    assert score_task(
        task,
        TaskRunRecord.model_validate(unlinked),
        _all_capabilities(),
        _config("state_scores.final_state_accuracy"),
    ).state_scores.final_state_accuracy == 1.0

    mismatch = deepcopy(base)
    mismatch["memory_snapshots"][0]["store_size"] = 2
    score = score_task(
        task,
        TaskRunRecord.model_validate(mismatch),
        _all_capabilities(),
        _config(
            "store_scores.final_memory_size",
            "state_scores.final_state_accuracy",
            "store_scores.duplicate_current_count",
        ),
    )
    assert score.supported_metric_fields[
        "store_scores.final_memory_size"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert score.state_scores.final_state_accuracy == 1.0
    assert score.store_scores.duplicate_current_count == 0


def test_max_version_ties_distinguish_ambiguity_from_duplicate_current(
    make_task, make_task_run
) -> None:
    base = make_task_run().model_dump(mode="python")
    current = base["memory_snapshots"][0]["entries"][0]
    conflict = deepcopy(current)
    conflict["entry_id"] = "entry_conflict"
    conflict["value_candidate"] = "Dalian"
    conflict["version_index"] = current["version_index"]
    base["memory_snapshots"][0]["entries"].append(conflict)
    base["memory_snapshots"][0]["store_size"] = 2
    ambiguous = score_task(
        make_task(),
        TaskRunRecord.model_validate(base),
        _all_capabilities(),
        _config("store_scores.obsolete_version_count", "store_scores.duplicate_current_count"),
    )
    assert ambiguous.store_scores.obsolete_version_count is None
    assert ambiguous.store_scores.duplicate_current_count is None
    assert "stale_retained" not in ambiguous.failure_flags

    same = make_task_run().model_dump(mode="python")
    duplicate = deepcopy(same["memory_snapshots"][0]["entries"][0])
    duplicate["entry_id"] = "entry_duplicate"
    same["memory_snapshots"][0]["entries"].append(duplicate)
    same["memory_snapshots"][0]["store_size"] = 2
    duplicate_score = score_task(
        make_task(),
        TaskRunRecord.model_validate(same),
        _all_capabilities(),
        _config("store_scores.duplicate_current_count"),
    )
    assert duplicate_score.store_scores.duplicate_current_count == 1


def test_distractor_copied_requires_retrieved_value_linkage(make_task, make_task_run) -> None:
    missing = make_task_run().model_dump(mode="python")
    missing["retrieval_traces"][0]["distractor_in_context"] = True
    missing["retrieval_traces"][0]["retrieved_entries"][0]["value_candidate"] = None
    score = score_task(
        make_task(),
        TaskRunRecord.model_validate(missing),
        _all_capabilities(),
        _config("answer_scores.distractor_copied"),
    )
    assert score.supported_metric_fields[
        "answer_scores.distractor_copied"
    ].reason is SupportReason.MISSING_ARTIFACT

    empty_payload = make_task_run().model_dump(mode="python")
    empty_trace = empty_payload["retrieval_traces"][0]
    empty_trace["retrieved_entries"] = []
    empty_trace["scores"] = []
    empty_trace["ranks"] = []
    empty_trace["distractor_in_context"] = True
    empty_payload["answer_predictions"][0]["cited_entry_ids"] = []
    empty_true = score_task(
        make_task(),
        TaskRunRecord.model_validate(empty_payload),
        _all_capabilities(),
        _config("answer_scores.distractor_copied"),
    )
    assert empty_true.supported_metric_fields[
        "answer_scores.distractor_copied"
    ].reason is SupportReason.MISSING_ARTIFACT

    empty_trace["distractor_in_context"] = False
    empty_false = score_task(
        make_task(),
        TaskRunRecord.model_validate(empty_payload),
        _all_capabilities(),
        _config("answer_scores.distractor_copied"),
    )
    assert empty_false.answer_scores.distractor_copied == 0.0

    linked = make_task_run().model_dump(mode="python")
    trace = linked["retrieval_traces"][0]
    distractor = deepcopy(trace["retrieved_entries"][0])
    distractor["entry_id"] = "entry_distractor"
    distractor["value_candidate"] = "WrongTown"
    distractor["content"] = "friend:alex.location = WrongTown"
    trace["retrieved_entries"].append(distractor)
    trace["scores"].append(0.8)
    trace["ranks"].append(2)
    trace["distractor_in_context"] = True
    positive = _replace_answer(
        TaskRunRecord.model_validate(linked), parsed_answer="WrongTown"
    )
    positive_score = score_task(
        make_task(),
        positive,
        _all_capabilities(),
        _config("answer_scores.distractor_copied"),
    )
    assert positive_score.answer_scores.distractor_copied == 1.0
    assert "distractor_copied" in positive_score.failure_flags
    negative_score = score_task(
        make_task(),
        TaskRunRecord.model_validate(linked),
        _all_capabilities(),
        _config("answer_scores.distractor_copied"),
    )
    assert negative_score.answer_scores.distractor_copied == 0.0


def test_system_event_and_selected_legacy_values_are_validated_at_boundary(
    make_task, make_task_run
) -> None:
    for key, value in (
        ("retrieval_latency_ms", -1.0),
        ("api_cost", float("nan")),
        ("error_rate", float("inf")),
        ("token_usage", True),
        ("input_tokens", "3"),
    ):
        run = _replace_run(make_task_run(), system_events=[{key: value}])
        with pytest.raises(ValueError, match="system_events"):
            score_task(make_task(), run, _all_capabilities(), _config())
    for key in (
        "ingest_latency_ms",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "api_cost",
        "error_rate",
    ):
        huge_run = _replace_run(make_task_run(), system_events=[{key: 10**10000}])
        with pytest.raises(ValueError, match=key):
            score_task(make_task(), huge_run, _all_capabilities(), _config())
        bypassed = TaskRunRecord.model_construct(
            **{
                **make_task_run().model_dump(mode="python"),
                "system_events": [{key: 10**10000}],
            }
        )
        with pytest.raises(ValueError, match=key):
            score_task(make_task(), bypassed, _all_capabilities(), _config())

    legacy_run = _replace_run(
        make_task_run(),
        system_events=[{"legacy_metrics": {"legacy_p63.bad": [float("nan")]}}],
    )
    with pytest.raises(ValueError, match="nonfinite legacy"):
        score_task(
            make_task(),
            legacy_run,
            _all_capabilities(),
            _config(legacy="legacy_p63"),
        )


def test_recursive_typed_equality_rejects_nested_numeric_type_aliases(
    make_task, make_task_run
) -> None:
    for gold, predicted in (
        ({"nested": [True]}, {"nested": [1]}),
        ({"nested": [1]}, {"nested": [1.0]}),
        ([{"value": True}], [{"value": 1}]),
    ):
        task_payload = make_task().model_dump(mode="python")
        task_payload["queries"][0]["answer_schema"] = "object" if isinstance(gold, dict) else "list"
        task_payload["gold"]["gold_answers"]["query_0"] = gold
        task_payload["gold"]["acceptable_answers"]["query_0"] = [gold]
        task = type(make_task()).model_validate(task_payload)
        run = _replace_answer(make_task_run(), parsed_answer=predicted)
        score = score_task(
            task,
            run,
            _all_capabilities(),
            _config("answer_scores.exact_match", "answer_scores.normalized_match"),
        )
        assert score.answer_scores.exact_match == 0.0
        assert score.answer_scores.normalized_match == 0.0


def test_full_score_builds_each_task_sized_evidence_component_at_most_once(
    make_task, make_task_run, monkeypatch
) -> None:
    names = (
        "_current_queries",
        "_action_pairs",
        "_traces_by_query",
        "_answers_by_query",
        "_final_snapshot",
        "_state_summary",
        "_store_counts",
        "_accepted_values",
        "_stale_values_for_query",
        "_is_distractor_copy",
    )
    counts = {name: 0 for name in names}
    for name in names:
        original = getattr(scorer_module, name)

        def wrapper(*args, __name=name, __original=original, **kwargs):
            counts[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(scorer_module, name, wrapper)

    score_task(
        make_task(),
        _complete_action_trace(make_task_run()),
        _all_capabilities(),
        _config(),
    )
    assert all(count <= 1 for count in counts.values()), counts
    assert counts["_current_queries"] == 1
    assert counts["_action_pairs"] == 1
    assert counts["_final_snapshot"] == 1


def test_recursive_typed_equality_operation_growth_is_linear(monkeypatch) -> None:
    original = scorer_module._same_value

    def measured(depth: int) -> int:
        calls = 0

        def wrapper(left, right):
            nonlocal calls
            calls += 1
            return original(left, right)

        monkeypatch.setattr(scorer_module, "_same_value", wrapper)
        left = value = []
        right = other = []
        for _ in range(depth):
            next_left: list = []
            next_right: list = []
            value.append(next_left)
            other.append(next_right)
            value = next_left
            other = next_right
        assert wrapper(left, right)
        return calls

    depth16 = measured(16)
    depth32 = measured(32)
    assert depth32 <= 2 * depth16 + 2


def test_identity_mismatch_is_rejected_before_scoring(make_task, make_task_run) -> None:
    run = _replace_run(make_task_run(), task_id="different_task")
    with pytest.raises(ValueError, match="task_id"):
        score_task(make_task(), run, _all_capabilities(), _config())
