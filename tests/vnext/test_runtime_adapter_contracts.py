from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

import mub.vnext.contracts as contracts
import mub.vnext.contracts.common as common_contracts
from mub.vnext.contracts.adapter import (
    AdapterActionLog,
    AdapterCapabilities,
    AdapterInfo,
    AnswerResult,
    MemoryAdapter,
    ResetResult,
    RetrievalResult,
)
from mub.vnext.contracts.enums import CompletionStatus, Operation
from mub.vnext.contracts.runtime import (
    AnswerPrediction,
    MemoryEntryRecord,
    MemorySnapshot,
    ParsedManagerAction,
    ParserExtractorProvenance,
    RetrievalTrace,
    TaskRunRecord,
)
from mub.vnext.contracts.task import MemoryEvent, MemoryQuery
from mub.vnext.version import RUNTIME_RECORD_VERSION, SCHEMA_VERSION


TASK_RUN_FIELDS = [
    "schema_version",
    "runtime_record_version",
    "task_id",
    "adapter_id",
    "run_id",
    "parsed_actions",
    "memory_snapshots",
    "retrieval_traces",
    "answer_predictions",
    "system_events",
    "parser_extractor_provenance",
    "exceptions",
    "completion_status",
]

PARSED_ACTION_FIELDS = [
    "event_id",
    "operation",
    "target_object_key",
    "value",
    "format_valid",
    "execution_status",
    "fallback_used",
    "error_flags",
    "raw_output",
    "latency_ms",
]

MEMORY_ENTRY_FIELDS = [
    "entry_id",
    "content",
    "object_key_candidate",
    "value_candidate",
    "created_at",
    "updated_at",
    "source_event_ids",
    "version_index",
    "raw_metadata",
]

MEMORY_SNAPSHOT_FIELDS = [
    "after_event_id",
    "entries",
    "state_by_object",
    "store_size",
    "raw_adapter_state",
    "snapshot_hash",
]

RETRIEVAL_TRACE_FIELDS = [
    "query_id",
    "retrieved_entries",
    "scores",
    "ranks",
    "gold_in_context",
    "stale_in_context",
    "distractor_in_context",
    "retrieval_policy",
    "context_order",
    "version_metadata",
    "prompt_hash",
]

ANSWER_PREDICTION_FIELDS = [
    "query_id",
    "raw_output",
    "disposition",
    "parsed_answer",
    "cited_event_ids",
    "cited_entry_ids",
    "format_valid",
    "error_flags",
    "latency_ms",
    "usage",
]

PROVENANCE_FIELDS = [
    "action_parser_version",
    "answer_parser_version",
    "memory_entry_extractor_version",
    "object_value_extractor_config_hash",
    "redaction_policy_version",
    "raw_provider_artifact_path",
    "raw_provider_artifact_hash",
    "raw_adapter_state_path",
    "raw_adapter_state_hash",
]

ADAPTER_INFO_FIELDS = [
    "adapter_id",
    "adapter_version",
    "system_name",
    "system_version",
    "sdk_version",
    "configuration_hash",
    "extractor_id",
    "extractor_version",
]

CAPABILITY_FIELDS = [
    "supports_isolated_reset",
    "supports_event_ingest",
    "supports_add",
    "supports_update",
    "supports_noop",
    "supports_delete",
    "supports_ttl",
    "supports_native_answer",
    "exports_entries",
    "exports_raw_state",
    "exports_source_event_ids",
    "exports_timestamps_or_order",
    "exports_object_keys",
    "exports_values",
    "exports_retrieval_ids",
    "exports_retrieval_scores",
    "exports_action_trace",
    "reports_latency",
    "reports_token_usage",
    "reports_cost",
    "requires_evaluation_extractor",
    "extractor_version",
]


def test_runtime_record_top_level_fields_match_design(make_task_run) -> None:
    run = make_task_run(status=CompletionStatus.FAILED, exception_type="timeout")

    assert list(TaskRunRecord.model_fields) == TASK_RUN_FIELDS
    assert list(run.model_dump(mode="json")) == TASK_RUN_FIELDS
    assert run.schema_version == SCHEMA_VERSION
    assert run.runtime_record_version == RUNTIME_RECORD_VERSION
    assert run.exceptions[0]["type"] == "timeout"


def test_task_run_record_rejects_v1_top_level_record(make_task_run) -> None:
    data = make_task_run().model_dump(mode="json")

    for field_name in ("schema_version", "runtime_record_version"):
        incompatible = {**data, field_name: "1.0.0"}
        with pytest.raises(ValidationError, match=field_name):
            TaskRunRecord.model_validate(incompatible)


def test_parsed_manager_action_uses_exact_design_fields_and_target_key(make_object_key) -> None:
    action = ParsedManagerAction(
        event_id="event_0",
        operation=Operation.UPDATE,
        target_object_key=make_object_key(),
        value="Qingdao",
        format_valid=True,
        execution_status="succeeded",
        fallback_used=False,
        error_flags=[],
        raw_output="UPDATE friend:alex.location = Qingdao",
        latency_ms=1.5,
    )

    assert list(ParsedManagerAction.model_fields) == PARSED_ACTION_FIELDS
    assert action.target_object_key.attribute == "location"
    assert action.model_dump(mode="json") == {
        "event_id": "event_0",
        "operation": "UPDATE",
        "target_object_key": make_object_key().model_dump(mode="json"),
        "value": "Qingdao",
        "format_valid": True,
        "execution_status": "succeeded",
        "fallback_used": False,
        "error_flags": [],
        "raw_output": "UPDATE friend:alex.location = Qingdao",
        "latency_ms": 1.5,
    }


def test_runtime_subrecords_use_exact_design_fields() -> None:
    assert list(MemoryEntryRecord.model_fields) == MEMORY_ENTRY_FIELDS
    assert list(MemorySnapshot.model_fields) == MEMORY_SNAPSHOT_FIELDS
    assert list(RetrievalTrace.model_fields) == RETRIEVAL_TRACE_FIELDS
    assert list(AnswerPrediction.model_fields) == ANSWER_PREDICTION_FIELDS
    assert list(ParserExtractorProvenance.model_fields) == PROVENANCE_FIELDS


def test_capability_levels_are_derived_shortcuts() -> None:
    l0 = AdapterCapabilities(supports_native_answer=True)
    l1 = l0.model_copy(update={"exports_retrieval_ids": True})
    l2 = l1.model_copy(
        update={
            "supports_isolated_reset": True,
            "exports_entries": True,
            "requires_evaluation_extractor": True,
        }
    )
    l3 = l2.model_copy(update={"exports_action_trace": True})

    assert [
        cap.presentation_level(state_transition_linkage_available=(cap is l3))
        for cap in [l0, l1, l2, l3]
    ] == [0, 1, 2, 3]
    assert l3.presentation_level(state_transition_linkage_available=False) == 2


def test_adapter_info_and_capabilities_use_exact_design_fields_and_safe_defaults() -> None:
    assert list(AdapterInfo.model_fields) == ADAPTER_INFO_FIELDS
    assert list(AdapterCapabilities.model_fields) == CAPABILITY_FIELDS

    capabilities = AdapterCapabilities()
    dumped = capabilities.model_dump(mode="json")
    assert all(dumped[field] is False for field in CAPABILITY_FIELDS[:-1])
    assert dumped["extractor_version"] is None
    assert capabilities.presentation_level() is None

    native_structured_l2 = AdapterCapabilities(
        supports_isolated_reset=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
    )
    exported_entries_l1 = AdapterCapabilities(exports_entries=True)
    assert native_structured_l2.presentation_level() == 2
    assert exported_entries_l1.presentation_level() == 1


def test_adapter_result_records_are_direct_design_records() -> None:
    reset = ResetResult(success=True, namespace="fixture", error=None)
    action_log = AdapterActionLog(
        event_id="event_0",
        requested_operation=Operation.UPDATE,
        effective_operation=Operation.UPDATE,
        affected_entry_ids=["entry_0"],
        raw_action={"operation": "UPDATE"},
        latency_ms=1.0,
        error=None,
    )
    retrieval = RetrievalResult(
        query_id="query_0",
        entries=[],
        scores=[],
        raw_result={"matches": []},
        latency_ms=2.0,
        error=None,
    )
    answer = AnswerResult(
        query_id="query_0",
        raw_output="Qingdao",
        usage={"output_tokens": 3},
        cost=0.01,
        latency_ms=4.0,
        error=None,
    )

    assert reset.model_dump(mode="json") == {"success": True, "namespace": "fixture", "error": None}
    assert list(AdapterActionLog.model_fields) == [
        "event_id",
        "requested_operation",
        "effective_operation",
        "affected_entry_ids",
        "raw_action",
        "latency_ms",
        "error",
    ]
    assert action_log.affected_entry_ids == ["entry_0"]
    assert retrieval.model_dump(mode="json") == {
        "query_id": "query_0",
        "entries": [],
        "scores": [],
        "raw_result": {"matches": []},
        "latency_ms": 2.0,
        "error": None,
    }
    assert answer.model_dump(mode="json") == {
        "query_id": "query_0",
        "raw_output": "Qingdao",
        "disposition": "answered",
        "value": None,
        "usage": {"output_tokens": 3},
        "cost": 0.01,
        "latency_ms": 4.0,
        "error": None,
    }


def test_public_exports_include_task4_models_without_alternate_wrappers() -> None:
    expected_exports = {
        "AdapterActionLog",
        "AdapterCapabilities",
        "AdapterInfo",
        "AnswerPrediction",
        "AnswerResult",
        "MemoryAdapter",
        "MemoryEntryRecord",
        "MemorySnapshot",
        "ParsedManagerAction",
        "ParserExtractorProvenance",
        "ResetResult",
        "RetrievalResult",
        "RetrievalTrace",
        "TaskRunRecord",
    }
    for name in expected_exports:
        assert getattr(contracts, name).__name__ == name
        assert name in contracts.__all__

    for forbidden_name in ("ManagerAction", "ActionResult", "RetrievalRecord", "AnswerRecord"):
        assert forbidden_name not in contracts.__all__
        assert not hasattr(contracts, forbidden_name)


def test_task_run_safe_defaults_and_json_round_trip(make_task_run) -> None:
    first = TaskRunRecord(
        task_id="task_a",
        adapter_id="adapter_a",
        run_id="run_a",
        parser_extractor_provenance=make_task_run().parser_extractor_provenance,
        completion_status=CompletionStatus.COMPLETED,
    )
    second = TaskRunRecord(
        task_id="task_b",
        adapter_id="adapter_b",
        run_id="run_b",
        parser_extractor_provenance=make_task_run().parser_extractor_provenance,
        completion_status=CompletionStatus.COMPLETED,
    )
    first.parsed_actions.append(make_task_run().parsed_actions[0])

    assert second.parsed_actions == []
    assert second.memory_snapshots == []
    assert second.retrieval_traces == []
    assert second.answer_predictions == []
    assert second.system_events == []
    assert second.exceptions == []

    full = make_task_run()
    assert TaskRunRecord.model_validate_json(full.model_dump_json()) == full


def test_hash_fields_reject_non_lowercase_sha256_values(make_task_run) -> None:
    assert common_contracts.SHA256_PATTERN == r"^[0-9a-f]{64}$"

    provenance_data = make_task_run().parser_extractor_provenance.model_dump(mode="json")
    for field_name in (
        "object_value_extractor_config_hash",
        "raw_provider_artifact_hash",
        "raw_adapter_state_hash",
    ):
        invalid = dict(provenance_data)
        invalid[field_name] = "A" * 64
        with pytest.raises(ValidationError):
            ParserExtractorProvenance(**invalid)

    snapshot_data = make_task_run().memory_snapshots[0].model_dump(mode="json")
    snapshot_data["snapshot_hash"] = "short"
    with pytest.raises(ValidationError):
        MemorySnapshot(**snapshot_data)

    with pytest.raises(ValidationError):
        RetrievalTrace(**{**make_task_run().retrieval_traces[0].model_dump(mode="json"), "prompt_hash": "g" * 64})
    with pytest.raises(ValidationError):
        AdapterInfo(
            adapter_id="adapter",
            adapter_version="1.0.0",
            system_name="system",
            system_version="1.0.0",
            sdk_version=None,
            configuration_hash="A" * 64,
            extractor_id=None,
            extractor_version=None,
        )


def test_nonnegative_runtime_and_adapter_numeric_fields_are_enforced(make_task_run) -> None:
    entry_data = make_task_run().memory_snapshots[0].entries[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        MemoryEntryRecord(**{**entry_data, "version_index": -1})

    snapshot_data = make_task_run().memory_snapshots[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        MemorySnapshot(**{**snapshot_data, "store_size": -1})

    action_data = make_task_run().parsed_actions[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        ParsedManagerAction(**{**action_data, "latency_ms": -0.1})

    answer_data = make_task_run().answer_predictions[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        AnswerPrediction(**{**answer_data, "usage": {"output_tokens": -1}})
    with pytest.raises(ValidationError):
        AnswerResult(query_id="query_0", raw_output="", value="answer", usage={"output_tokens": -1}, cost=0.0, latency_ms=0.0, error=None)
    with pytest.raises(ValidationError):
        AnswerResult(query_id="query_0", raw_output="", value="answer", usage={}, cost=-0.1, latency_ms=0.0, error=None)


def test_strict_integer_fields_reject_bool_and_string_values(make_task_run) -> None:
    entry_data = make_task_run().memory_snapshots[0].entries[0].model_dump(mode="json")
    snapshot_data = make_task_run().memory_snapshots[0].model_dump(mode="json")
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")
    answer_data = make_task_run().answer_predictions[0].model_dump(mode="json")

    assert MemoryEntryRecord(**{**entry_data, "version_index": 2}).version_index == 2
    assert MemorySnapshot(**{**snapshot_data, "store_size": 2}).store_size == 2
    assert RetrievalTrace(**{**trace_data, "ranks": [2]}).ranks == [2]
    assert AnswerPrediction(**{**answer_data, "usage": {"output_tokens": 2}}).usage == {"output_tokens": 2}
    assert AnswerResult(query_id="query_0", raw_output="", value="answer", usage={"output_tokens": 2}, cost=0.0, latency_ms=0.0, error=None).usage == {"output_tokens": 2}

    for invalid_value in (True, "2"):
        with pytest.raises(ValidationError):
            MemoryEntryRecord(**{**entry_data, "version_index": invalid_value})
        with pytest.raises(ValidationError):
            MemorySnapshot(**{**snapshot_data, "store_size": invalid_value})
        with pytest.raises(ValidationError):
            RetrievalTrace(**{**trace_data, "ranks": [invalid_value]})
        with pytest.raises(ValidationError):
            AnswerPrediction(**{**answer_data, "usage": {"output_tokens": invalid_value}})
        with pytest.raises(ValidationError):
            AnswerResult(query_id="query_0", raw_output="", value="answer", usage={"output_tokens": invalid_value}, cost=0.0, latency_ms=0.0, error=None)


@pytest.mark.parametrize("valid_value", [0, 0.0, 1, 1.5])
def test_task4_nonnegative_float_fields_accept_json_numeric_values(make_task_run, valid_value) -> None:
    action_data = make_task_run().parsed_actions[0].model_dump(mode="json")
    answer_data = make_task_run().answer_predictions[0].model_dump(mode="json")
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")

    assert ParsedManagerAction(**{**action_data, "latency_ms": valid_value}).latency_ms == valid_value
    assert AnswerPrediction(**{**answer_data, "latency_ms": valid_value}).latency_ms == valid_value
    assert AdapterActionLog(
        event_id="event_0",
        requested_operation=Operation.UPDATE,
        effective_operation=Operation.UPDATE,
        affected_entry_ids=[],
        raw_action={},
        latency_ms=valid_value,
        error=None,
    ).latency_ms == valid_value
    assert RetrievalResult(query_id="query_0", entries=[], scores=[], raw_result={}, latency_ms=valid_value, error=None).latency_ms == valid_value
    assert AnswerResult(query_id="query_0", raw_output="", value="answer", usage={}, cost=valid_value, latency_ms=valid_value, error=None).cost == valid_value
    assert AnswerResult(query_id="query_0", raw_output="", value="answer", usage={}, cost=0.0, latency_ms=valid_value, error=None).latency_ms == valid_value
    assert RetrievalTrace(**{**trace_data, "scores": [valid_value]}).scores == [valid_value]
    assert RetrievalResult(query_id="query_0", entries=trace_data["retrieved_entries"], scores=[valid_value], raw_result={}, latency_ms=0.0, error=None).scores == [valid_value]


@pytest.mark.parametrize("invalid_value", [True, "1", "1.5"])
def test_task4_float_fields_reject_bool_and_numeric_strings(make_task_run, invalid_value) -> None:
    action_data = make_task_run().parsed_actions[0].model_dump(mode="json")
    answer_data = make_task_run().answer_predictions[0].model_dump(mode="json")
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")

    with pytest.raises(ValidationError):
        ParsedManagerAction(**{**action_data, "latency_ms": invalid_value})
    with pytest.raises(ValidationError):
        AnswerPrediction(**{**answer_data, "latency_ms": invalid_value})
    with pytest.raises(ValidationError):
        AdapterActionLog(
            event_id="event_0",
            requested_operation=Operation.UPDATE,
            effective_operation=Operation.UPDATE,
            affected_entry_ids=[],
            raw_action={},
            latency_ms=invalid_value,
            error=None,
        )
    with pytest.raises(ValidationError):
        RetrievalResult(query_id="query_0", entries=[], scores=[], raw_result={}, latency_ms=invalid_value, error=None)
    with pytest.raises(ValidationError):
        AnswerResult(query_id="query_0", raw_output="", value="answer", usage={}, cost=invalid_value, latency_ms=0.0, error=None)
    with pytest.raises(ValidationError):
        AnswerResult(query_id="query_0", raw_output="", value="answer", usage={}, cost=0.0, latency_ms=invalid_value, error=None)
    with pytest.raises(ValidationError):
        RetrievalTrace(**{**trace_data, "scores": [invalid_value]})
    with pytest.raises(ValidationError):
        RetrievalResult(query_id="query_0", entries=trace_data["retrieved_entries"], scores=[invalid_value], raw_result={}, latency_ms=0.0, error=None)


def test_retrieval_scores_accept_negative_numeric_scores(make_task_run) -> None:
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")

    assert RetrievalTrace(**{**trace_data, "scores": [-1.5]}).scores == [-1.5]
    assert RetrievalResult(query_id="query_0", entries=trace_data["retrieved_entries"], scores=[-1.5], raw_result={}, latency_ms=0.0, error=None).scores == [-1.5]


def test_retrieval_score_and_rank_lengths_allow_unavailable_lists(make_task_run) -> None:
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")
    entries = trace_data["retrieved_entries"]

    no_score_trace = RetrievalTrace(**{**trace_data, "scores": [], "ranks": [1]})
    no_rank_trace = RetrievalTrace(**{**trace_data, "scores": [0.9], "ranks": []})
    no_score_result = RetrievalResult(query_id="query_0", entries=entries, scores=[], raw_result={}, latency_ms=0.0, error=None)

    assert no_score_trace.retrieved_entries
    assert no_score_trace.scores == []
    assert no_rank_trace.ranks == []
    assert no_score_result.entries
    assert no_score_result.scores == []

    result = RetrievalResult(query_id="query_0", entries=entries, scores=[0.9], raw_result={}, latency_ms=0.0, error=None)
    assert len(result.entries) == len(result.scores)

    with pytest.raises(ValidationError, match="scores length must match retrieved_entries length"):
        RetrievalTrace(**{**trace_data, "scores": [0.9, 0.8], "ranks": [1]})
    with pytest.raises(ValidationError, match="ranks length must match retrieved_entries length"):
        RetrievalTrace(**{**trace_data, "scores": [0.9], "ranks": [1, 2]})
    with pytest.raises(ValidationError, match="scores length must match entries length"):
        RetrievalResult(query_id="query_0", entries=entries, scores=[0.9, 0.8], raw_result={}, latency_ms=0.0, error=None)


def test_retrieval_records_support_adapter_without_retrieval_scores(make_task_run) -> None:
    trace_data = make_task_run().retrieval_traces[0].model_dump(mode="json")
    capabilities = AdapterCapabilities(exports_retrieval_ids=True, exports_retrieval_scores=False)
    trace = RetrievalTrace(**{**trace_data, "scores": [], "ranks": [1]})
    result = RetrievalResult(query_id="query_0", entries=trace_data["retrieved_entries"], scores=[], raw_result={}, latency_ms=0.0, error=None)

    assert capabilities.presentation_level() == 1
    assert trace.scores == []
    assert result.scores == []


def test_memory_adapter_protocol_uses_corrected_plan_signatures() -> None:
    expected = {
        "adapter_info": ([], AdapterInfo),
        "capabilities": ([], AdapterCapabilities),
        "reset": ([('namespace', str), ('config', dict)], ResetResult),
        "ingest_event": ([('event', MemoryEvent)], AdapterActionLog),
        "export_entries": ([], list[MemoryEntryRecord]),
        "export_raw_state": ([], object),
        "retrieve": ([('query', MemoryQuery), ('k', int)], RetrievalResult),
        "answer": ([('query', MemoryQuery), ('mode', str)], AnswerResult),
        "close": ([], None),
    }
    for method_name, (parameters, return_annotation) in expected.items():
        signature = inspect.signature(getattr(MemoryAdapter, method_name))
        assert list(signature.parameters) == ["self", *[name for name, _ in parameters]]
        for name, annotation in parameters:
            assert signature.parameters[name].annotation == annotation
        assert signature.return_annotation == return_annotation


def test_runtime_scoring_booleans_are_strict(make_task_run) -> None:
    run = make_task_run().model_dump(mode="python")
    for section, field in (
        ("parsed_actions", "format_valid"),
        ("parsed_actions", "fallback_used"),
        ("retrieval_traces", "gold_in_context"),
        ("retrieval_traces", "stale_in_context"),
        ("retrieval_traces", "distractor_in_context"),
        ("answer_predictions", "format_valid"),
    ):
        payload = make_task_run().model_dump(mode="python")
        payload[section][0][field] = "true"
        with pytest.raises(ValidationError):
            TaskRunRecord.model_validate(payload)
        payload[section][0][field] = 1
        with pytest.raises(ValidationError):
            TaskRunRecord.model_validate(payload)
