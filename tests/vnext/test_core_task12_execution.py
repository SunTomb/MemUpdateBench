from __future__ import annotations

import hashlib
import json

import pytest

from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.preparation.task12 import (
    RawAppendTrajectoryV1,
    Task12AdmittedAnswerRunV1,
    Task12AdmittedCellV1,
)
from mub.vnext.runtime.task12_execution_v3 import (
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
    persist_task12_rows_v3,
    persist_task12_scores_v3,
    select_admitted_answer_run_v3,
    load_finalized_task12_run_v3,
    transform_retrieval_trace_v3,
    task12_runtime_configuration_sha256_v3,
    validate_task12_runtime_configuration_v3,
    verify_task12_score_artifact_v3,
    validate_task12_runtime_code_binding_v3,
)


def _entry(entry_id: str, event_index: int, version_index: int, value: str) -> MemoryEntryRecordV3:
    return MemoryEntryRecordV3(
        entry_id=entry_id,
        content=f"entity.attribute = {value}",
        object_key_candidate=FrozenMemoryObjectKey(
            object_type="fact",
            namespace="default",
            entity="entity",
            attribute="attribute",
        ),
        value_candidate=value,
        source_event_ids=(f"event-{event_index}",),
        version_index=version_index,
        raw_metadata={"event_index": event_index},
    )


def _trajectory_receipt(
    task_id: str,
    entries: tuple[MemoryEntryRecordV3, ...],
) -> RawAppendTrajectoryV1:
    rows = [
        {
            "entry_id": (
                f"{entry.source_event_ids[0]}:"
                f"{entry.object_key_candidate.canonical_id}:"
                f"{entry.version_index}"
            ),
            "event_index": entry.raw_metadata["event_index"],
            "version_index": entry.version_index,
            "object_key": entry.object_key_candidate.canonical_id,
            "value": entry.value_candidate,
        }
        for entry in entries
    ]
    latest_id = rows[-1]["entry_id"]
    return RawAppendTrajectoryV1(
        task_id=task_id,
        entry_ids=tuple(row["entry_id"] for row in rows),
        object_ids=tuple(row["object_key"] for row in rows),
        event_indices=tuple(row["event_index"] for row in rows),
        version_indices=tuple(row["version_index"] for row in rows),
        latest_entry_ids=(latest_id,),
        trajectory_sha256=hashlib.sha256(
            json.dumps(
                rows,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    )


def test_task12_runtime_configuration_rejects_k_or_context_drift() -> None:
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3

    runtime = RuntimeConfigV3(
        run_id="task12-run",
        retrieval_policy="normal_topk",
        answer_mode="slot_prompt",
        retrieval_k=16,
        capture_snapshots=False,
    )
    expected_hash = task12_runtime_configuration_sha256_v3(
        runtime,
        context_order="reverse_chronological",
        context_annotation="latest_outdated_label",
    )
    validate_task12_runtime_configuration_v3(
        expected_hash,
        expected_run_id="task12-run",
        runtime_config=runtime,
        context_order="reverse_chronological",
        context_annotation="latest_outdated_label",
    )
    with pytest.raises(ValueError, match="runtime configuration"):
        validate_task12_runtime_configuration_v3(
            expected_hash,
            expected_run_id="task12-run",
            runtime_config=RuntimeConfigV3(
                run_id="task12-run",
                retrieval_policy="normal_topk",
                answer_mode="slot_prompt",
                retrieval_k=4,
                capture_snapshots=False,
            ),
            context_order="reverse_chronological",
            context_annotation="latest_outdated_label",
        )


def test_task12_runtime_code_binding_rejects_revision_or_tree_drift() -> None:
    expected = Task12RuntimeCodeBindingV1(
        code_revision="a" * 40,
        code_tree_sha256="b" * 64,
    )
    assert validate_task12_runtime_code_binding_v3(expected, expected) == expected
    with pytest.raises(ValueError, match="runtime code binding"):
        validate_task12_runtime_code_binding_v3(
            expected,
            expected.model_copy(update={"code_tree_sha256": "c" * 64}),
        )


def test_task12_transform_reverse_orders_by_full_event_version_metadata() -> None:
    old = _entry("old", 1, 0, "old-value")
    latest = _entry("latest", 2, 1, "latest-value")
    trace = RetrievalTraceV3(
        query_id="q1",
        retrieved_entries=(old, latest),
        scores=(2.0, 1.0),
        ranks=(1, 2),
        retrieval_policy="normal_topk",
    )

    transformed = transform_retrieval_trace_v3(
        trace,
        context_order="reverse_chronological",
        context_annotation="none",
        full_trajectory=(old, latest),
        frozen_trajectory=_trajectory_receipt("task-1", (old, latest)),
    )

    assert [entry.entry_id for entry in transformed.retrieved_entries] == ["latest", "old"]
    assert transformed.ranks == (1, 2)
    assert transformed.context_order == "reverse_chronological"


def test_task12_labels_use_full_trajectory_not_retrieved_subset() -> None:
    old = _entry("old", 1, 0, "old-value")
    latest = _entry("latest", 2, 1, "latest-value")
    trace = RetrievalTraceV3(
        query_id="q1",
        retrieved_entries=(old,),
        scores=(1.0,),
        ranks=(1,),
        retrieval_policy="normal_topk",
    )

    transformed = transform_retrieval_trace_v3(
        trace,
        context_order="reverse_chronological",
        context_annotation="latest_outdated_label",
        full_trajectory=(old, latest),
        frozen_trajectory=_trajectory_receipt("task-1", (old, latest)),
    )

    assert transformed.version_metadata["old"] == "outdated"
    assert transformed.retrieved_entries[0].raw_metadata["version_label"] == "outdated"


def test_task12_transform_preserves_retrieved_entry_multiset() -> None:
    old = _entry("old", 1, 0, "old-value")
    latest = _entry("latest", 2, 1, "latest-value")
    trace = RetrievalTraceV3(
        query_id="q1",
        retrieved_entries=(old, latest),
        scores=(2.0, 1.0),
        ranks=(1, 2),
        retrieval_policy="normal_topk",
    )

    transformed = transform_retrieval_trace_v3(
        trace,
        context_order="chronological",
        context_annotation="none",
        full_trajectory=(old, latest),
        frozen_trajectory=_trajectory_receipt("task-1", (old, latest)),
    )

    assert {entry.entry_id for entry in transformed.retrieved_entries} == {"old", "latest"}
    assert transformed.scores == (2.0, 1.0)








def test_task12_action_id_rebinding_accepts_only_current_observed_event_id() -> None:
    from types import SimpleNamespace

    from mub.vnext.contracts.enums import Operation
    from mub.vnext.contracts.v3.adapter import (
        AdapterActionPayloadV3,
        AdapterActionResultV3,
    )
    from mub.vnext.contracts.v3.enums import ExecutionStatusV3
    from mub.vnext.runtime.task12_execution_v3 import (
        canonical_task12_action_id_adapter_factory_v3,
    )

    event = SimpleNamespace(event_id="event-0")
    task = SimpleNamespace(
        events=(
            SimpleNamespace(
                event_id=event.event_id,
                gold_action_ids=("action-0",),
            ),
        )
    )

    class Adapter:
        adapter_id = "raw_add"
        append_only_observation = True
        retrieval_policy = "normal_topk"

        def __init__(self, parsed_action_id: str) -> None:
            self.parsed_action_id = parsed_action_id

        def ingest_event(self, observed_event):
            action = AdapterActionPayloadV3(operation=Operation.NOOP)
            return AdapterActionResultV3(
                event_id=observed_event.event_id,
                requested_action=action,
                effective_action=action,
                execution_status=ExecutionStatusV3.EXECUTED,
                raw_result={"parsed_action_id": self.parsed_action_id},
            )

    factory = canonical_task12_action_id_adapter_factory_v3(
        lambda _task: Adapter("observed_action:event-0")
    )
    result = factory(task).ingest_event(event)
    assert result.raw_result["parsed_action_id"] == "action-0"

    bad_factory = canonical_task12_action_id_adapter_factory_v3(
        lambda _task: Adapter("observed_action:other-event")
    )
    with pytest.raises(ValueError, match="observed action ID"):
        bad_factory(task).ingest_event(event)


def test_persist_task12_rows_supports_create_and_resume(tmp_path) -> None:
    from tests.vnext.test_run_core_v3_persistence import _prompted_config, _prompted_row

    config = _prompted_config()
    rows = (
        _prompted_row("task-1", action_id="action-1", query_id="query-1"),
        _prompted_row("task-2", action_id="action-2", query_id="query-2"),
    )
    manifest = persist_task12_rows_v3(tmp_path / "run", config, rows)
    assert manifest.run_id == config.run_id
    assert (tmp_path / "run" / "run_manifest.json").is_file()
    loaded_manifest, loaded_rows = load_finalized_task12_run_v3(
        tmp_path / "run",
        config,
    )
    assert loaded_manifest == manifest
    assert loaded_rows == rows
    cell = Task12AdmittedCellV1(
        cell_id="raw-add-chronological-none-k04",
        scope_id="core-hard-v1-family-a",
        canonical_binding_sha256="c" * 64,
    )
    run = Task12AdmittedAnswerRunV1(
        cell_id=cell.cell_id,
        answer_model_slot="answer_model_a",
        cell_binding_sha256=cell.canonical_binding_sha256,
        answer_model_binding_sha256="d" * 64,
        canonical_run_binding_sha256="e" * 64,
    )
    authorization = Task12ExecutionAuthorizationV1(
        preparation_manifest_sha256="a" * 64,
        plan_fingerprint_sha256="b" * 64,
        runtime_code_binding=Task12RuntimeCodeBindingV1(
            code_revision="a" * 40,
            code_tree_sha256="b" * 64,
        ),
        cell_id=cell.cell_id,
        answer_model_slot="answer_model_a",
        cell_binding_sha256=cell.canonical_binding_sha256,
        answer_model_binding_sha256=run.answer_model_binding_sha256,
        canonical_run_binding_sha256=run.canonical_run_binding_sha256,
        task_manifest_sha256="1" * 64,
        task_view_sha256="2" * 64,
        run_config_sha256="3" * 64,
        expected_task_count=80,
        output_leaf="task12-execution",
    )

    assert select_admitted_answer_run_v3(
        authorization,
        preparation_manifest_sha256="a" * 64,
        plan_fingerprint_sha256="b" * 64,
        admitted_cells=(cell,),
        admitted_answer_runs=(run,),
    ) == run

    with pytest.raises(ValueError, match="plan fingerprint"):
        select_admitted_answer_run_v3(
            authorization,
            preparation_manifest_sha256="a" * 64,
            plan_fingerprint_sha256="f" * 64,
            admitted_cells=(cell,),
            admitted_answer_runs=(run,),
        )


def test_persist_task12_scores_binds_finalized_run_and_task_manifest(tmp_path) -> None:
    from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS, MetricFieldSupport, ScoreRecordV3
    from mub.vnext.contracts.enums import SupportReason

    support = {
        path: MetricFieldSupport(reason=SupportReason.NOT_SUPPORTED, null_policy="emit_null")
        for path in CORE_METRIC_FIELD_PATHS
    }
    score = ScoreRecordV3.empty(
        task_id="task-1",
        run_id="run-1",
        adapter_id="raw_add",
        task_family="repeated_same_slot_update",
        difficulty="medium",
        completion_status="completed",
        supported_metric_fields=support,
    )
    receipt = persist_task12_scores_v3(
        tmp_path,
        (score,),
        run_manifest_sha256="a" * 64,
        task_manifest_sha256="b" * 64,
    )
    assert receipt["score_count"] == 1
    assert (tmp_path / "scores.jsonl").is_file()
    assert (tmp_path / "score_receipt.json").is_file()
    verified_scores, verified_receipt = verify_task12_score_artifact_v3(
        tmp_path,
        expected_task_ids=("task-1",),
        run_manifest_sha256="a" * 64,
        task_manifest_sha256="b" * 64,
    )
    assert verified_scores == (score,)
    assert verified_receipt == receipt
    with pytest.raises(FileExistsError):
        persist_task12_scores_v3(
            tmp_path,
            (score,),
            run_manifest_sha256="a" * 64,
            task_manifest_sha256="b" * 64,
        )

    score_path = tmp_path / "scores.jsonl"
    score_path.write_bytes(
        score_path.read_bytes().replace(b'"run_id":"run-1"', b'"run_id":"run-2"')
    )
    with pytest.raises(ValueError, match="score artifact hash"):
        verify_task12_score_artifact_v3(
            tmp_path,
            expected_task_ids=("task-1",),
            run_manifest_sha256="a" * 64,
            task_manifest_sha256="b" * 64,
        )
