import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.contracts.enums import ActionScope, CompletionStatus, Operation
from mub.vnext.contracts.v3.enums import ExecutionStatusV3, LedgerEntryStatus
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.core_config import load_core_config


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"


@pytest.fixture(scope="module")
def all_core_tasks():
    from mub.vnext.generation.family_e import compile_family_e_micro_pilot
    from mub.vnext.generation.family_f import compile_family_f_micro_pilot
    from mub.vnext.generation.family_g import compile_family_g_micro_pilot

    config = load_core_config(CORE_CONFIG)
    return (
        *compile_family_e_micro_pilot(config, code_revision="core-runtime-red").tasks,
        *compile_family_f_micro_pilot(config, code_revision="core-runtime-red").tasks,
        *compile_family_g_micro_pilot(config, code_revision="core-runtime-red").tasks,
    )


@pytest.fixture(scope="module")
def core_tasks(all_core_tasks):
    return tuple(
        task
        for task in all_core_tasks
        if task.metadata.extra["surface_variant"] == 0
    )


def _task(core_tasks, family, *, condition=None):
    return next(
        task
        for task in core_tasks
        if task.task_family == family
        and (
            condition is None
            or condition(task)
        )
    )


def _run_config(run_id="core-runtime-test"):
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3

    return RuntimeConfigV3(run_id=run_id)


def test_support_resolver_distinguishes_task_and_metric_support(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3, ReferenceAdapterV3
    from mub.vnext.runtime.support_v3 import resolve_task_support_v3

    family_e = _task(
        core_tasks,
        "deletion_forgetting",
        condition=lambda task: task.metadata.extra["stratification"]["lifecycle_cell"] == "entity_wide_deletion",
    )
    family_f = _task(
        core_tasks,
        "current_historical_query",
        condition=lambda task: task.queries[0].selector.kind == "previous",
    )

    reference = resolve_task_support_v3(
        family_f,
        ReferenceAdapterV3(family_f).capabilities(),
    )
    assert reference.terminal_supported is True
    assert all(reference.operation_support.values())
    assert all(reference.query_support.values())
    assert reference.metric_support["exports_version_history"] is True
    assert reference.metric_support["exports_evidence_linkage"] is True

    exact_e = resolve_task_support_v3(
        family_e,
        ExactCrudAdapterV3(family_e).capabilities(),
    )
    assert exact_e.terminal_supported is True
    assert exact_e.operation_support[Operation.DELETE.value] is True
    assert exact_e.metric_support["exports_version_history"] is False

    exact_f = resolve_task_support_v3(
        family_f,
        ExactCrudAdapterV3(family_f).capabilities(),
    )
    assert exact_f.terminal_supported is False
    assert exact_f.query_support[family_f.queries[0].query_id] is False
    assert "supports_historical_query" in exact_f.missing_capabilities


def test_unsupported_core_task_still_returns_one_terminal_row(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(
        core_tasks,
        "current_historical_query",
        condition=lambda task: task.queries[0].selector.kind == "previous",
    )
    run = execute_task_v3(task, ExactCrudAdapterV3(task), _run_config())

    assert run.task_id == task.task_id
    assert run.completion_status is CompletionStatus.NOT_SUPPORTED
    assert run.parsed_actions == ()
    assert run.answer_predictions == ()
    assert len(run.system_events) == 3
    assert run.system_events[-1]["event"] == "task_support"
    assert run.system_events[-1]["terminal_supported"] is False


def test_prompted_model_route_uses_retrieval_trace_without_native_answer(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    class TrackingPromptedModel:
        def __init__(self) -> None:
            self.requests = []

        def answer(self, request):
            self.requests.append(request)
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output='{"disposition":"answered","answer":"prompted"}',
                parsed_answer="prompted",
                format_valid=True,
                usage={"completion_tokens": 1},
            )

        def close(self) -> None:
            pass

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    adapter = ReferenceAdapterV3(task)
    adapter.answer = lambda *_args, **_kwargs: pytest.fail(
        "prompted model route must not call MemoryAdapterV3.answer"
    )
    prompted_model = TrackingPromptedModel()

    run = execute_task_v3(
        task,
        adapter,
        RuntimeConfigV3(run_id="prompted-route", answer_mode="slot_prompt"),
        prompted_answer_model=prompted_model,
    )

    assert run.completion_status is CompletionStatus.COMPLETED
    assert len(prompted_model.requests) == len(task.queries)
    assert len(run.retrieval_traces) == len(task.queries)
    assert all(trace.prompt_hash is not None for trace in run.retrieval_traces)
    assert [prediction.parsed_answer for prediction in run.answer_predictions] == [
        "prompted"
    ] * len(task.queries)


def test_slot_prompt_requires_an_explicit_prompted_model(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    run = execute_task_v3(
        task,
        ReferenceAdapterV3(task),
        RuntimeConfigV3(run_id="prompted-required", answer_mode="slot_prompt"),
    )

    assert run.completion_status is CompletionStatus.NOT_SUPPORTED
    assert run.answer_predictions == ()
    support = next(event for event in run.system_events if event["event"] == "task_support")
    assert "prompted_answer_model" in support["missing_capabilities"]


def test_prompted_unavailable_prediction_preserves_error_flags(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.contracts.enums import AnswerDisposition
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    class UnavailablePromptedModel:
        def answer(self, request):
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output="",
                disposition=AnswerDisposition.UNAVAILABLE,
                format_valid=False,
                error_flags=("offline_model_unavailable",),
            )

        def close(self) -> None:
            pass

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    run = execute_task_v3(
        task,
        ReferenceAdapterV3(task),
        RuntimeConfigV3(run_id="prompted-unavailable", answer_mode="slot_prompt"),
        prompted_answer_model=UnavailablePromptedModel(),
    )

    assert run.completion_status is CompletionStatus.PARTIAL
    assert run.retrieval_traces
    assert run.answer_predictions[0].error_flags == ("offline_model_unavailable",)
    answer_exception = next(
        exception for exception in run.exceptions if exception["phase"] == "answer"
    )
    assert answer_exception["error_flags"] == ("offline_model_unavailable",)


@pytest.mark.parametrize(
    "family",
    (
        "deletion_forgetting",
        "current_historical_query",
        "long_horizon_memory_synthesis",
    ),
)
def test_reference_v3_is_complete_and_exact_for_core_families(core_tasks, family) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(core_tasks, family)
    run = execute_task_v3(task, ReferenceAdapterV3(task), _run_config(f"reference-{family}"))

    assert run.completion_status is CompletionStatus.COMPLETED
    assert len(run.parsed_actions) == len(task.actions)
    assert len(run.answer_predictions) == len(task.queries)
    for prediction, evidence in zip(run.answer_predictions, task.gold_evidence):
        assert prediction.parsed_answer == evidence.answer
        assert prediction.cited_event_ids == evidence.supporting_event_ids
        assert prediction.cited_object_keys == evidence.supporting_object_keys
        assert evidence.final_derivation_step_id in prediction.cited_derivation_step_ids


def test_raw_append_records_delete_without_claiming_physical_deletion(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(
        core_tasks,
        "deletion_forgetting",
        condition=lambda task: task.metadata.extra["stratification"]["lifecycle_cell"] == "explicit_object_or_attribute_deletion",
    )
    adapter = RawAppendAdapterV3(task)
    assert adapter.capabilities().supports_delete is False

    run = execute_task_v3(task, adapter, _run_config("raw-delete"))

    assert run.completion_status is CompletionStatus.COMPLETED
    delete_rows = [
        action for action in run.parsed_actions if action.operation is Operation.DELETE
    ]
    assert delete_rows
    assert all("append_only_no_physical_delete" in row.error_flags for row in delete_rows)
    final_snapshot = run.memory_snapshots[-1]
    assert any(
        entry.raw_metadata.get("entry_kind") == "delete_instruction"
        for entry in final_snapshot.entries
    )
    forgotten = task.metadata.extra["stratification"].get("forgotten_value")
    if forgotten is not None:
        assert any(entry.value_candidate == forgotten for entry in final_snapshot.entries)


def test_exact_crud_applies_scoped_delete_and_preserves_collateral(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(
        core_tasks,
        "deletion_forgetting",
        condition=lambda task: task.metadata.extra["stratification"]["lifecycle_cell"] == "scoped_delete_protected_collateral",
    )
    delete = next(action for action in task.actions if action.operation is Operation.DELETE)
    assert delete.scope is ActionScope.ATTRIBUTE

    run = execute_task_v3(task, ExactCrudAdapterV3(task), _run_config("exact-delete"))

    assert run.completion_status is CompletionStatus.COMPLETED
    final_state = run.memory_snapshots[-1].state_by_object
    assert all(key.canonical_id not in final_state for key in delete.target_object_keys)
    protected = task.metadata.extra["family_e"]["protected_collateral_ids"]
    assert all(object_id in final_state for object_id in protected)
    assert run.answer_predictions[0].parsed_answer == task.gold_evidence[0].answer


def test_exact_crud_computes_g_answer_without_fabricating_evidence(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    adapter = ExactCrudAdapterV3(task)
    assert adapter.capabilities().supports_multi_object_query is True
    assert adapter.capabilities().exports_evidence_linkage is False

    run = execute_task_v3(task, adapter, _run_config("exact-g"))

    assert run.completion_status is CompletionStatus.COMPLETED
    assert run.answer_predictions[0].parsed_answer == task.gold_evidence[0].answer
    assert run.answer_predictions[0].cited_event_ids == ()
    assert run.answer_predictions[0].cited_object_keys == ()
    assert run.answer_predictions[0].cited_derivation_step_ids == ()


def _corrupted_non_runtime_gold(task: MemUpdateTaskV3) -> MemUpdateTaskV3:
    def corrupt(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1000
        if isinstance(value, float):
            return value + 1000.0
        if isinstance(value, str):
            return f"corrupted::{value}"
        if isinstance(value, list):
            return [corrupt(item) if item is not None else None for item in value]
        if isinstance(value, dict):
            return {key: corrupt(item) for key, item in value.items()}
        return value

    payload = task.model_dump(mode="python")
    payload["actions"] = []
    for event in payload["events"]:
        event["gold_action_ids"] = []
    for ledger in payload["version_history"]:
        for entry in ledger["entries"]:
            if entry["status"] is LedgerEntryStatus.PRESENT:
                entry["value"] = corrupt(entry["value"])
    for evidence in payload["gold_evidence"]:
        if evidence["answer"] is not None:
            evidence["answer"] = corrupt(evidence["answer"])
        alternative = evidence.get("stale_alternative")
        if alternative is not None:
            alternative["answer"] = corrupt(alternative["answer"])
    return MemUpdateTaskV3.model_validate(payload)


def _runtime_projection(run):
    payload = run.model_dump(mode="json")
    payload.pop("task_id")
    payload.pop("run_id")
    payload["system_events"] = [
        {
            key: value
            for key, value in event.items()
            if key not in {"task_id", "task_semantic_hash", "run_id"}
        }
        for event in payload["system_events"]
    ]
    return payload


def test_non_reference_runtime_decisions_ignore_contract_valid_gold_corruption(
    all_core_tasks,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3, RawAppendAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    selected = tuple(
        task
        for task in all_core_tasks
        if task.task_family in {
            "deletion_forgetting",
            "long_horizon_memory_synthesis",
        }
        or task.queries[0].selector.kind == "current"
    )
    assert {task.metadata.extra["surface_variant"] for task in selected} == {0, 1, 2, 3}

    for adapter_type in (RawAppendAdapterV3, ExactCrudAdapterV3):
        adapter_tasks = all_core_tasks if adapter_type is RawAppendAdapterV3 else selected
        for task in adapter_tasks:
            corrupted = _corrupted_non_runtime_gold(task)
            baseline = execute_task_v3(
                task,
                adapter_type(task),
                _run_config("leakage-check"),
            )
            observed = execute_task_v3(
                corrupted,
                adapter_type(corrupted),
                _run_config("leakage-check"),
            )
            assert _runtime_projection(observed) == _runtime_projection(baseline), (
                adapter_type.__name__,
                task.task_id,
            )


def test_reference_is_perfect_for_every_e_f_g_micro_pilot_task(all_core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_tasks_v3

    records = execute_tasks_v3(
        all_core_tasks,
        ReferenceAdapterV3,
        _run_config("reference-entire-micro-pilot"),
    )

    assert len(records) == len(all_core_tasks)
    assert [record.task_id for record in records] == [task.task_id for task in all_core_tasks]
    for task, record in zip(all_core_tasks, records):
        assert record.completion_status is CompletionStatus.COMPLETED, task.task_id
        assert len(record.parsed_actions) == len(task.events)
        assert len(record.answer_predictions) == len(task.queries)
        for prediction, evidence in zip(record.answer_predictions, task.gold_evidence):
            assert prediction.parsed_answer == evidence.answer
            assert prediction.cited_event_ids == evidence.supporting_event_ids
            assert prediction.cited_object_keys == evidence.supporting_object_keys


@pytest.mark.parametrize("adapter_name", ("raw", "exact", "heuristic"))
def test_builtin_batch_has_one_ordered_terminal_row_for_all_micro_pilot_tasks(
    all_core_tasks,
    adapter_name,
) -> None:
    from mub.vnext.adapters.core_v3 import (
        ExactCrudAdapterV3,
        HeuristicCrudAdapterV3,
        RawAppendAdapterV3,
    )
    from mub.vnext.runtime.engine_v3 import execute_tasks_v3

    class FiniteEncoder:
        def encode(self, texts, *, normalize_embeddings):
            assert normalize_embeddings is True
            return [[0.25, 0.75] for _ in texts]

    if adapter_name == "raw":
        factory = RawAppendAdapterV3
    elif adapter_name == "exact":
        factory = ExactCrudAdapterV3
    else:
        factory = lambda task: HeuristicCrudAdapterV3(
            task,
            encoder=FiniteEncoder(),
            encoder_revision="minilm-test-revision",
        )

    records = execute_tasks_v3(
        iter(all_core_tasks),
        factory,
        _run_config(f"{adapter_name}-entire-micro-pilot"),
    )

    assert len(records) == len(all_core_tasks)
    assert [record.task_id for record in records] == [task.task_id for task in all_core_tasks]
    for task, record in zip(all_core_tasks, records):
        historical = task.queries[0].selector.kind in {
            "previous",
            "exact_version",
            "event_anchor",
            "logical_time_anchor",
            "transition",
            "ordered_history",
        }
        expected = (
            CompletionStatus.NOT_SUPPORTED
            if adapter_name in {"exact", "heuristic"} and historical
            else CompletionStatus.COMPLETED
        )
        assert record.completion_status is expected, (
            adapter_name,
            task.task_id,
            record.exceptions,
        )
        if expected is CompletionStatus.COMPLETED:
            assert record.answer_predictions[0].parsed_answer == task.gold_evidence[0].answer
            assert record.answer_predictions[0].cited_event_ids == ()
            assert record.answer_predictions[0].cited_object_keys == ()
            assert record.answer_predictions[0].cited_derivation_step_ids == ()


def test_execute_tasks_preserves_batch_completeness_after_factory_failure(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_tasks_v3

    tasks = core_tasks[:3]

    def factory(task):
        if task is tasks[1]:
            raise RuntimeError("factory exploded")
        return ExactCrudAdapterV3(task)

    records = execute_tasks_v3(tasks, factory, _run_config("factory-failure"))

    assert [record.task_id for record in records] == [task.task_id for task in tasks]
    assert len(records) == len(tasks)
    assert records[1].completion_status is CompletionStatus.FAILED
    assert records[1].exceptions == (
        {
            "phase": "adapter_factory",
            "type": "RuntimeError",
            "message": "factory exploded",
        },
    )


def test_raw_delete_reports_requested_delete_and_effective_noop(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3
    from mub.vnext.contracts.v3.adapter import ResetRequestV3

    task = _task(
        core_tasks,
        "deletion_forgetting",
        condition=lambda task: task.metadata.extra["stratification"]["lifecycle_cell"]
        == "explicit_object_or_attribute_deletion",
    )
    adapter = RawAppendAdapterV3(task)
    adapter.reset(ResetRequestV3(namespace="raw-delete-action-contract"))
    results = [adapter.ingest_event(event) for event in task.events]
    delete = next(
        result
        for result in results
        if result.requested_action.operation is Operation.DELETE
    )

    assert delete.requested_action.operation is Operation.DELETE
    assert delete.effective_action.operation is Operation.NOOP
    assert delete.effective_action.target_object_keys == ()
    assert delete.execution_status is ExecutionStatusV3.NO_EFFECT
    assert delete.reason == "append_only_no_physical_delete"
    assert delete.affected_entry_ids == ()
    assert any(
        entry.raw_metadata.get("entry_kind") == "delete_instruction"
        for entry in adapter.export_entries().entries
    )


def test_raw_ttl_retains_physical_value_and_exports_logical_tombstone(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3
    from mub.vnext.contracts.v3.adapter import (
        ResetRequestV3,
        VersionHistoryExportRequestV3,
    )

    task = _task(
        core_tasks,
        "deletion_forgetting",
        condition=lambda task: task.metadata.extra["stratification"]["lifecycle_cell"]
        == "logical_ttl_expiry",
    )
    adapter = RawAppendAdapterV3(task)
    namespace = "raw-ttl-history"
    adapter.reset(ResetRequestV3(namespace=namespace))
    for event in task.events:
        adapter.ingest_event(event)

    state = adapter.export_raw_state().model_dump(mode="json")["raw_state"]
    target_id = task.target_objects[0].canonical_id
    assert state["state_by_object"][target_id] is not None
    history = adapter.export_version_history(
        VersionHistoryExportRequestV3(namespace=namespace)
    )
    target_history = next(
        item for item in history.histories if item.object_key == task.target_objects[0]
    )
    assert target_history.versions[-1].status is LedgerEntryStatus.TOMBSTONE
    assert target_history.versions[-1].logical_time == task.actions[1].effective_at


def test_malformed_visible_surface_fails_closed(core_tasks) -> None:


    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    payload = task.model_dump(mode="python")
    payload["events"][0]["raw_text"] = "This text contains no memory instruction."
    payload["events"][0]["normalized_text"] = "Not an operation."
    malformed = MemUpdateTaskV3.model_validate(payload)

    record = execute_task_v3(
        malformed,
        ExactCrudAdapterV3(malformed),
        _run_config("malformed-surface"),
    )

    assert record.completion_status is CompletionStatus.NOT_SUPPORTED
    assert record.parsed_actions == ()
    assert "visible_action_parse" in record.system_events[-1]["missing_capabilities"]


def test_verified_minilm_is_required_for_heuristic_runtime(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import HeuristicCrudAdapterV3
    from mub.vnext.contracts.v3.adapter import ResetRequestV3

    class FiniteEncoder:
        def encode(self, texts, *, normalize_embeddings):
            return [[1.0, 0.0] for _ in texts]

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    unpinned = HeuristicCrudAdapterV3(task, encoder=FiniteEncoder())
    verified = HeuristicCrudAdapterV3(
        task,
        encoder=FiniteEncoder(),
        encoder_revision="minilm-test-revision",
    )

    assert unpinned.reset(ResetRequestV3(namespace="unpinned")).success is False
    assert verified.reset(ResetRequestV3(namespace="verified")).success is True


def test_reset_and_close_failures_are_retained_in_terminal_record(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    class BrokenLifecycleAdapter(ExactCrudAdapterV3):
        def reset(self, request):
            raise RuntimeError("reset exploded")

        def close(self):
            raise RuntimeError("close exploded")

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        BrokenLifecycleAdapter(task),
        _run_config("broken-lifecycle"),
    )

    assert record.completion_status is CompletionStatus.FAILED
    assert [item["phase"] for item in record.exceptions] == ["reset", "close"]
    assert [item["message"] for item in record.exceptions] == [
        "reset exploded",
        "close exploded",
    ]


def test_snapshot_hash_uses_canonical_json(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        ExactCrudAdapterV3(task),
        _run_config("snapshot-hash"),
    )
    snapshot = record.memory_snapshots[0]
    dumped = snapshot.model_dump(mode="json")
    payload = {
        "after_event_id": dumped["after_event_id"],
        "entries": dumped["entries"],
        "state_by_object": dumped["state_by_object"],
        "store_size": dumped["store_size"],
        "raw_adapter_state": dumped["raw_adapter_state"],
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert snapshot.snapshot_hash == expected
    assert all(
        "slot" not in object_id
        for object_id in snapshot.state_by_object
    )


def _historical_selector_g_task(core_tasks):
    task = next(
        task
        for task in core_tasks
        if task.task_family == "long_horizon_memory_synthesis"
        and task.queries[0].query_type.value == "update_sensitive_multi_hop"
    )
    payload = task.model_dump(mode="python")
    query = payload["queries"][0]
    query["selector"] = {"kind": "exact_version", "version_index": 0}
    history = {
        ledger.object_key.canonical_id: ledger.entries[0]
        for ledger in task.version_history
    }
    values = [history[key.canonical_id].value for key in task.queries[0].target_object_keys]
    steps = []
    supporting_events = []
    for index, key in enumerate(task.queries[0].target_object_keys):
        entry = history[key.canonical_id]
        event_ids = list(entry.source_event_ids)
        supporting_events.extend(event_ids)
        steps.append({
            "step_id": f"historical_read_{index}",
            "operation": "read_version",
            "input_step_ids": [],
            "supporting_object_keys": [key.model_dump(mode="python")],
            "supporting_event_ids": event_ids,
        })
    answer = values[0]
    left_step = steps[0]["step_id"]
    for index, value in enumerate(values[1:], start=1):
        answer -= value
        step_id = f"historical_subtract_{index}"
        steps.append({
            "step_id": step_id,
            "operation": "subtract",
            "input_step_ids": [left_step, f"historical_read_{index}"],
            "supporting_object_keys": [],
            "supporting_event_ids": [],
        })
        left_step = step_id
    payload["gold_evidence"] = [{
        "query_id": query["query_id"],
        "answer": answer,
        "supporting_object_keys": query["target_object_keys"],
        "supporting_event_ids": list(dict.fromkeys(supporting_events)),
        "derivation_steps": steps,
        "final_derivation_step_id": left_step,
        "stale_alternative": None,
    }]
    return MemUpdateTaskV3.model_validate(payload)


def test_historical_selector_nested_in_g_requires_historical_capability(
    core_tasks,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3, HeuristicCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3
    from mub.vnext.runtime.support_v3 import resolve_task_support_v3

    class FiniteEncoder:
        def encode(self, texts, *, normalize_embeddings):
            return [[0.25, 0.75] for _ in texts]

    task = _historical_selector_g_task(core_tasks)
    exact = ExactCrudAdapterV3(task)
    support = resolve_task_support_v3(task, exact.capabilities())

    assert support.terminal_supported is False
    assert support.query_support[task.queries[0].query_id] is False
    assert "supports_historical_query" in support.missing_capabilities

    adapters = (
        exact,
        HeuristicCrudAdapterV3(
            task,
            encoder=FiniteEncoder(),
            encoder_revision="minilm-test-revision",
        ),
    )
    for adapter in adapters:
        record = execute_task_v3(
            task,
            adapter,
            _run_config(f"historical-g-{adapter.adapter_id}"),
        )
        assert record.completion_status is CompletionStatus.NOT_SUPPORTED
        assert record.parsed_actions == ()
        assert record.answer_predictions == ()


@pytest.mark.parametrize(
    ("missing_capability", "answer_mode"),
    (
        ("supports_isolated_reset", "slot_direct"),
        ("supports_event_ingest", "slot_direct"),
        ("supports_native_answer", "native_answer"),
    ),
)
def test_foundational_capability_gaps_are_terminally_not_supported(
    core_tasks,
    missing_capability,
    answer_mode,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import (
        RuntimeConfigV3,
        execute_tasks_v3,
    )

    class GuardedAdapter(ExactCrudAdapterV3):
        def capabilities(self):
            return super().capabilities().model_copy(
                update={missing_capability: False}
            )

        def reset(self, request):
            raise AssertionError("unsupported adapter must not be reset")

        def ingest_event(self, event):
            raise AssertionError("unsupported adapter must not ingest")

        def answer(self, query, mode):
            raise AssertionError("unsupported adapter must not answer")

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    records = execute_tasks_v3(
        (task,),
        GuardedAdapter,
        RuntimeConfigV3(
            run_id=f"missing-{missing_capability}",
            answer_mode=answer_mode,
        ),
    )

    assert len(records) == 1
    record = records[0]
    assert record.task_id == task.task_id
    assert record.completion_status is CompletionStatus.NOT_SUPPORTED
    assert record.exceptions == ()
    assert record.parsed_actions == ()
    assert record.answer_predictions == ()
    support_event = next(
        event for event in record.system_events if event["event"] == "task_support"
    )
    assert support_event["terminal_supported"] is False
    assert support_event["runtime_support"][missing_capability] is False
    assert missing_capability in support_event["missing_capabilities"]


def test_slot_direct_does_not_require_native_answer_capability(core_tasks) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    class DirectOnlyAdapter(ExactCrudAdapterV3):
        def capabilities(self):
            return super().capabilities().model_copy(
                update={"supports_native_answer": False}
            )

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        DirectOnlyAdapter(task),
        _run_config("slot-direct-without-native"),
    )

    assert record.completion_status is CompletionStatus.COMPLETED
    assert record.answer_predictions


def test_runtime_config_rejects_unknown_answer_mode() -> None:
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3

    with pytest.raises(ValueError, match="unknown answer mode"):
        RuntimeConfigV3(run_id="bad-answer-mode", answer_mode="mystery")


@pytest.mark.parametrize(
    ("exports_entries", "exports_raw_state", "expected_mode"),
    (
        (True, False, "entries_only"),
        (False, True, "skipped"),
        (False, False, "skipped"),
    ),
)
def test_snapshot_capture_respects_export_capabilities(
    core_tasks,
    exports_entries,
    exports_raw_state,
    expected_mode,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import execute_task_v3

    class CapabilityLimitedExportAdapter(ExactCrudAdapterV3):
        def capabilities(self):
            return super().capabilities().model_copy(update={
                "exports_entries": exports_entries,
                "exports_raw_state": exports_raw_state,
            })

        def export_entries(self):
            if not exports_entries:
                raise AssertionError("entries export must not be called")
            return super().export_entries()

        def export_raw_state(self):
            if not exports_raw_state:
                raise AssertionError("raw state export must not be called")
            return super().export_raw_state()

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        CapabilityLimitedExportAdapter(task),
        _run_config(
            f"snapshot-capabilities-{exports_entries}-{exports_raw_state}"
        ),
    )

    assert record.completion_status is CompletionStatus.COMPLETED
    policy = next(
        event
        for event in record.system_events
        if event["event"] == "snapshot_capture"
    )
    assert policy["mode"] == expected_mode
    if expected_mode == "entries_only":
        assert record.memory_snapshots
        assert all(snapshot.entries for snapshot in record.memory_snapshots)
        assert all(
            snapshot.raw_adapter_state is None
            for snapshot in record.memory_snapshots
        )
        assert all(
            snapshot.state_by_object == {}
            for snapshot in record.memory_snapshots
        )
        assert all(
            snapshot.store_size == len(snapshot.entries)
            for snapshot in record.memory_snapshots
        )
    else:
        assert record.memory_snapshots == ()


def test_retrieval_policy_mismatch_fails_closed_and_matching_policy_is_bound(
    core_tasks,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    mismatched = execute_task_v3(
        task,
        ExactCrudAdapterV3(task, retrieval_policy="latest_per_object"),
        RuntimeConfigV3(
            run_id="retrieval-policy-mismatch",
            retrieval_policy="normal_topk",
        ),
    )

    assert mismatched.completion_status is CompletionStatus.NOT_SUPPORTED
    assert mismatched.exceptions == ()
    assert mismatched.parsed_actions == ()
    assert mismatched.retrieval_traces == ()
    mismatch_binding = next(
        event
        for event in mismatched.system_events
        if event["event"] == "retrieval_policy"
    )
    assert mismatch_binding["requested"] == "normal_topk"
    assert mismatch_binding["adapter_declared"] == "latest_per_object"
    assert mismatch_binding["matched"] is False
    support = next(
        event
        for event in mismatched.system_events
        if event["event"] == "task_support"
    )
    assert support["terminal_supported"] is False
    assert "retrieval_policy_mismatch" in support["missing_capabilities"]

    matched = execute_task_v3(
        task,
        ExactCrudAdapterV3(task, retrieval_policy="latest_per_object"),
        RuntimeConfigV3(
            run_id="retrieval-policy-match",
            retrieval_policy="latest_per_object",
        ),
    )
    assert matched.completion_status is CompletionStatus.COMPLETED
    assert matched.retrieval_traces
    assert all(
        trace.retrieval_policy == "latest_per_object"
        for trace in matched.retrieval_traces
    )
    identity = matched.system_events[0]
    assert identity["retrieval_policy"] == "latest_per_object"
    binding = next(
        event
        for event in matched.system_events
        if event["event"] == "retrieval_policy"
    )
    assert binding["effective"] == "latest_per_object"
    assert binding["matched"] is True


@pytest.mark.parametrize(
    ("reported_policy", "expected_reason"),
    (
        (None, "retrieval_policy_unbound"),
        ("", "retrieval_policy_unbound"),
        ("latest_per_object", "retrieval_policy_mismatch"),
    ),
)
def test_undeclared_adapter_must_report_matching_effective_retrieval_policy(
    core_tasks,
    reported_policy,
    expected_reason,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.contracts.v3.adapter import RetrievalResultV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    class UndeclaredPolicyAdapter:
        def __init__(self, task):
            self._inner = ExactCrudAdapterV3(task)

        def __getattr__(self, name):
            if name == "retrieval_policy":
                raise AttributeError(name)
            return getattr(self._inner, name)

        def retrieve(self, request):
            result = self._inner.retrieve(request)
            return RetrievalResultV3.model_validate({
                **result.model_dump(mode="python"),
                "trace": {
                    **result.trace.model_dump(mode="python"),
                    "retrieval_policy": reported_policy,
                },
            })

        def answer(self, query, mode):
            raise AssertionError("unbound policy must stop before answer")

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        UndeclaredPolicyAdapter(task),
        RuntimeConfigV3(
            run_id=f"undeclared-unbound-{reported_policy!r}",
            retrieval_policy="normal_topk",
        ),
    )

    assert record.completion_status is CompletionStatus.NOT_SUPPORTED
    assert record.exceptions == ()
    assert record.answer_predictions == ()
    assert record.retrieval_traces == ()
    observed = next(
        event
        for event in record.system_events
        if event["event"] == "retrieval_policy_observed"
    )
    assert observed["effective"] == reported_policy
    assert observed["matched"] is False
    assert observed["reason"] == expected_reason


def test_undeclared_adapter_with_truthful_observed_policy_remains_supported(
    core_tasks,
) -> None:
    from mub.vnext.adapters.core_v3 import ExactCrudAdapterV3
    from mub.vnext.contracts.v3.adapter import RetrievalResultV3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    class TruthfullyObservedPolicyAdapter:
        def __init__(self, task):
            self._inner = ExactCrudAdapterV3(task)

        def __getattr__(self, name):
            if name == "retrieval_policy":
                raise AttributeError(name)
            return getattr(self._inner, name)

        def retrieve(self, request):
            result = self._inner.retrieve(request)
            return RetrievalResultV3.model_validate({
                **result.model_dump(mode="python"),
                "trace": {
                    **result.trace.model_dump(mode="python"),
                    "retrieval_policy": "normal_topk",
                },
            })

    task = _task(core_tasks, "long_horizon_memory_synthesis")
    record = execute_task_v3(
        task,
        TruthfullyObservedPolicyAdapter(task),
        RuntimeConfigV3(
            run_id="undeclared-truthful-policy",
            retrieval_policy="normal_topk",
        ),
    )

    assert record.completion_status is CompletionStatus.COMPLETED
    assert record.answer_predictions
    assert record.retrieval_traces[0].retrieval_policy == "normal_topk"
    observed = next(
        event
        for event in record.system_events
        if event["event"] == "retrieval_policy_observed"
    )
    assert observed["effective"] == "normal_topk"
    assert observed["matched"] is True
    assert observed["reason"] is None
