from __future__ import annotations

from mub.vnext.contracts import (
    ActionScope,
    AnswerSchema,
    ArtifactRef,
    CompletionStatus,
    Difficulty,
    EvaluationMode,
    EventRole,
    MemoryObjectKey,
    Operation,
    QueryType,
    RunManifest,
    ScoreRecord,
    SourceRecord,
    SourceType,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.runtime import (
    AnswerPrediction,
    MemoryEntryRecord,
    MemorySnapshot,
    ParsedManagerAction,
    ParserExtractorProvenance,
    RetrievalTrace,
    TaskRunRecord,
)
from mub.vnext.contracts.task import (
    GoldAction,
    GoldRecord,
    MemUpdateTask,
    MemoryEvent,
    MemoryQuery,
    SplitKey,
    TaskMetadata,
)

RAW_HASH = "a" * 64
NORMALIZED_HASH = "b" * 64
GENERATION_HASH = "c" * 64
PROVIDER_HASH = "d" * 64
ADAPTER_STATE_HASH = "e" * 64
PROMPT_HASH = "f" * 64
CONFIG_HASH = "1" * 64
SNAPSHOT_HASH = "2" * 64


def _artifact_ref(path: str, char: str, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=char * 64,
        media_type=media_type,
        record_count=1,
    )


def make_score_record(**overrides) -> ScoreRecord:
    layer_defaults = {
        "protocol_scores": {
            "action_parse_valid": True,
            "answer_parse_valid": True,
            "execution_success_rate": 1.0,
            "unsupported_operation_rate": 0.0,
            "fallback_rate": 0.0,
        },
        "action_scores": {
            "operation_accuracy": 1.0,
            "full_action_exact_match": 1.0,
            "object_key_accuracy": 1.0,
            "entity_accuracy": 1.0,
            "attribute_accuracy": 1.0,
            "value_accuracy": 1.0,
            "false_write_rate": 0.0,
            "missed_write_rate": 0.0,
            "wrong_object_write_rate": 0.0,
        },
        "state_scores": {
            "final_state_accuracy": 1.0,
            "state_precision": 1.0,
            "state_recall": 1.0,
            "state_f1": 1.0,
            "state_resolve_rate": 1.0,
            "collateral_corruption_rate": 0.0,
            "expected_absence_accuracy": 1.0,
        },
        "store_scores": {
            "obsolete_version_count": 0,
            "stale_conflicting_value_count": 0,
            "duplicate_current_count": 0,
            "final_memory_size": 1,
            "compaction_ratio": 1.0,
            "write_amplification": 1.0,
        },
        "retrieval_scores": {
            "current_recall_at_k": 1.0,
            "current_mrr": 1.0,
            "stale_exposure_rate": 0.0,
            "stale_count_in_context": 0,
            "distractor_exposure_rate": 0.0,
        },
        "answer_scores": {
            "exact_match": 1.0,
            "normalized_match": 1.0,
            "token_f1": 1.0,
            "structured_field_accuracy": 1.0,
            "reference_resolution_accuracy": 1.0,
            "stale_copied": 0.0,
            "distractor_copied": 0.0,
            "gold_retrieved_wrong_answer": 0.0,
            "answer_state_consistency": 1.0,
        },
        "system_scores": {
            "ingest_latency_ms": 1.0,
            "retrieval_latency_ms": 1.0,
            "answer_latency_ms": 1.0,
            "token_usage": 13,
            "api_cost": 0.0,
            "error_rate": 0.0,
        },
        "audit_scores": {
            "action_trace_available": True,
            "state_export_available": True,
            "retrieval_trace_available": True,
            "source_provenance_coverage": 1.0,
            "manifest_completeness": 1.0,
        },
    }
    data = {
        "task_id": "task_repeated_same_slot_easy_0001",
        "run_id": "run_fixture_0001",
        "adapter_id": "adapter_fixture",
        "task_family": TaskFamily.REPEATED_SAME_SLOT.value,
        "difficulty": Difficulty.EASY,
        "completion_status": CompletionStatus.COMPLETED,
        "supported_metric_fields": {},
        "failure_flags": [],
        "primary_failure": None,
        "legacy_metrics": {},
    }
    for layer_name, defaults in layer_defaults.items():
        layer_override = overrides.pop(layer_name, {})
        data[layer_name] = (
            {**defaults, **layer_override}
            if isinstance(layer_override, dict)
            else layer_override
        )
    data.update(overrides)
    return ScoreRecord(**data)


def make_run_manifest(**overrides) -> RunManifest:
    expected = overrides.pop("expected", 1)
    completed = overrides.pop("completed", 1)
    failed = overrides.pop("failed", 0)
    not_supported = overrides.pop("not_supported", 0)
    nested_defaults = {
        "adapter_info": {
            "adapter_id": "adapter_fixture",
            "adapter_version": "1.0.0",
            "system_name": "fixture_memory_system",
            "system_version": "1.0.0",
            "sdk_version": None,
            "configuration_hash": "3" * 64,
            "extractor_id": "fixture_extractor",
            "extractor_version": "1.0.0",
        },
        "adapter_capabilities": {
            "supports_isolated_reset": True,
            "supports_event_ingest": True,
            "supports_add": True,
            "supports_update": True,
            "supports_noop": True,
            "exports_entries": True,
            "exports_raw_state": True,
            "exports_source_event_ids": True,
            "exports_timestamps_or_order": True,
            "exports_object_keys": True,
            "exports_values": True,
            "exports_retrieval_ids": True,
            "exports_retrieval_scores": True,
            "exports_action_trace": True,
            "reports_latency": True,
            "reports_token_usage": True,
        },
        "prompt_config": {"template": "fixture", "version": "prompt-v1"},
        "decoding_config": {"temperature": 0.0},
        "seed_information": {"seed": 0},
        "environment_summary": {"python": "fixture"},
        "package_summary": {"mub": "fixture"},
        "native_vs_extracted_field_summary": {"mode": "native"},
    }
    data = {
        "run_id": "run_fixture_0001",
        "timestamp": "2026-07-20T00:00:00Z",
        "code_revision": "fixed-test-revision",
        "dirty_state": False,
        "task_manifest": _artifact_ref("manifests/task_manifest.json", "4"),
        "capability_verification_artifact": _artifact_ref(
            "artifacts/capability_verification.json", "5"
        ),
        "model_name": "fixture-model",
        "provider": "fixture-provider",
        "model_revision": "fixture-revision",
        "action_parser_version": "action-parser-v1",
        "answer_parser_version": "answer-parser-v1",
        "memory_entry_extractor_version": "entry-extractor-v1",
        "object_value_extractor_config_hash": CONFIG_HASH,
        "redaction_policy_version": "redaction-v1",
        "expected_task_count": expected,
        "completed_task_count": completed,
        "failed_task_count": failed,
        "not_supported_task_count": not_supported,
        "raw_provider_response_artifacts": [
            _artifact_ref("artifacts/provider/responses.jsonl", "6", "application/jsonl")
        ],
        "raw_adapter_state_artifacts": [
            _artifact_ref("artifacts/adapter/state.json", "7")
        ],
        "normalized_runtime_artifacts": [
            _artifact_ref("artifacts/runtime/records.jsonl", "8", "application/jsonl")
        ],
        "score_artifacts": [
            _artifact_ref("artifacts/scores/records.jsonl", "9", "application/jsonl")
        ],
    }
    for field_name, defaults in nested_defaults.items():
        nested_override = overrides.pop(field_name, {})
        data[field_name] = (
            {**defaults, **nested_override}
            if isinstance(nested_override, dict)
            else nested_override
        )
    data.update(overrides)
    return RunManifest(**data)


def make_object_key() -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="friend:alex",
        attribute="location",
    )


def build_task_run(
    status: CompletionStatus = CompletionStatus.COMPLETED,
    exception_type: str | None = None,
) -> TaskRunRecord:
    key = make_object_key()
    entry = MemoryEntryRecord(
        entry_id="entry_0",
        content="friend:alex.location = Qingdao",
        object_key_candidate=key,
        value_candidate="Qingdao",
        created_at="event_1",
        updated_at="event_1",
        source_event_ids=["event_1"],
        version_index=1,
        raw_metadata={"source": "factory"},
    )
    provenance = ParserExtractorProvenance(
        action_parser_version="action-parser-v1",
        answer_parser_version="answer-parser-v1",
        memory_entry_extractor_version="entry-extractor-v1",
        object_value_extractor_config_hash=CONFIG_HASH,
        redaction_policy_version="redaction-v1",
        raw_provider_artifact_path="artifacts/provider/task_run.jsonl",
        raw_provider_artifact_hash=PROVIDER_HASH,
        raw_adapter_state_path="artifacts/adapter/task_run.json",
        raw_adapter_state_hash=ADAPTER_STATE_HASH,
    )
    exceptions = []
    if exception_type is not None:
        exceptions.append({"type": exception_type, "message": "fixture failure"})

    return TaskRunRecord(
        task_id="task_repeated_same_slot_easy_0001",
        adapter_id="adapter_fixture",
        run_id="run_fixture_0001",
        parsed_actions=[
            ParsedManagerAction(
                event_id="event_1",
                operation=Operation.UPDATE,
                target_object_key=key,
                value="Qingdao",
                format_valid=True,
                execution_status="succeeded",
                fallback_used=False,
                error_flags=[],
                raw_output="UPDATE friend:alex.location = Qingdao",
                latency_ms=1.5,
            )
        ],
        memory_snapshots=[
            MemorySnapshot(
                after_event_id="event_1",
                entries=[entry],
                state_by_object={key.canonical_id: "Qingdao"},
                store_size=1,
                raw_adapter_state={"entries": 1},
                snapshot_hash=SNAPSHOT_HASH,
            )
        ],
        retrieval_traces=[
            RetrievalTrace(
                query_id="query_0",
                retrieved_entries=[entry],
                scores=[0.9],
                ranks=[1],
                gold_in_context=True,
                stale_in_context=False,
                distractor_in_context=False,
                retrieval_policy="top_k",
                context_order="ranked",
                version_metadata={"latest_first": True},
                prompt_hash=PROMPT_HASH,
            )
        ],
        answer_predictions=[
            AnswerPrediction(
                query_id="query_0",
                raw_output="Qingdao",
                parsed_answer="Qingdao",
                cited_event_ids=["event_1"],
                cited_entry_ids=["entry_0"],
                format_valid=True,
                error_flags=[],
                latency_ms=2.0,
                usage={"input_tokens": 10, "output_tokens": 3},
            )
        ],
        system_events=[{"event": "completed"}],
        parser_extractor_provenance=provenance,
        exceptions=exceptions,
        completion_status=status,
    )


def build_task() -> MemUpdateTask:
    key = MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="friend:alex",
        attribute="location",
    )

    events = [
        MemoryEvent(
            event_id="event_0",
            sequence_index=0,
            timestamp=None,
            raw_text="My friend Alex lives in Dalian.",
            normalized_text="My friend Alex lives in Dalian.",
            speaker=None,
            gold_action_ids=["action_0"],
            role=EventRole.STALE_SAME_SLOT,
        ),
        MemoryEvent(
            event_id="event_1",
            sequence_index=1,
            timestamp=None,
            raw_text="My friend Alex relocated to Qingdao.",
            normalized_text="My friend Alex relocated to Qingdao.",
            speaker=None,
            gold_action_ids=["action_1"],
            role=EventRole.LATEST_GOLD,
        ),
    ]

    actions = [
        GoldAction(
            action_id="action_0",
            event_id="event_0",
            operation=Operation.ADD,
            scope=ActionScope.ATTRIBUTE,
            target_object_keys=[key],
            value="Dalian",
            effective_at=None,
        ),
        GoldAction(
            action_id="action_1",
            event_id="event_1",
            operation=Operation.UPDATE,
            scope=ActionScope.ATTRIBUTE,
            target_object_keys=[key],
            value="Qingdao",
            effective_at=None,
        ),
    ]

    return MemUpdateTask(
        task_id="task_repeated_same_slot_easy_0001",
        task_family=TaskFamily.REPEATED_SAME_SLOT.value,
        difficulty=Difficulty.EASY,
        source=SourceRecord(
            source_id="source_synthetic_0001",
            source_type=SourceType.SYNTHETIC,
            source_uri="memory://source_synthetic_0001",
            license_or_privacy="synthetic_redistributable",
            raw_hash=RAW_HASH,
            normalized_hash=NORMALIZED_HASH,
            normalization_version="1.0.0",
            provenance={"source_group_id": "source_group_0001", "redistributable": True},
            generator={
                "generator_name": "vnext_phase0_factory",
                "seed": 0,
                "config_sha256": GENERATION_HASH,
                "code_revision": "fixed-test-revision",
                "compiler_version": "1.0.0",
            },
        ),
        events=events,
        target_objects=[key],
        queries=[
            MemoryQuery(
                query_id="query_0",
                query_type=QueryType.CURRENT_STATE,
                text="Where does my friend Alex live now?",
                target_object_keys=[key],
                answer_schema=AnswerSchema.STRING,
                evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
            )
        ],
        gold=GoldRecord(
            actions=actions,
            action_sequence=["action_0", "action_1"],
            final_state={key.canonical_id: "Qingdao"},
            version_history={key.canonical_id: ["Dalian", "Qingdao"]},
            expected_present_objects=[key],
            expected_absent_objects=[],
            gold_source_event_ids=["event_1"],
            gold_answers={"query_0": "Qingdao"},
            acceptable_answers={"query_0": ["Qingdao"]},
        ),
        metadata=TaskMetadata(
            split=Split.TEST,
            split_key=SplitKey(
                semantic_core_id="semantic_core_0001",
                source_group_id="source_group_0001",
                trajectory_id="trajectory_0001",
                split_policy_version="1.0.0",
            ),
            profile_name=Difficulty.EASY,
            resolved_profile={"update_depth": 1},
            generation_config_hash=GENERATION_HASH,
            compiler_version="1.0.0",
        ),
    )
