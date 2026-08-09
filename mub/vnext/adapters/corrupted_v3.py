from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from mub.vnext.contracts.common import thaw_json
from mub.vnext.contracts.enums import ActionScope, Operation
from mub.vnext.contracts.v3.adapter import AdapterInfoV3
from mub.vnext.contracts.v3.enums import LedgerEntryStatus
from mub.vnext.contracts.v3.runtime import (
    AnswerPredictionV3,
    MemoryEntryRecordV3,
    MemorySnapshotV3,
    ParsedManagerActionV3,
    TaskRunRecordV3,
)
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.validation.replay_v3 import replay_task_v3


@dataclass(frozen=True)
class CorruptedControlMetadataV3:
    control_id: str
    target_layer: str
    expected_failure_flags: tuple[str, ...]
    smoke_only: bool = True
    leaderboard_eligible: bool = False


_CONTROL_METADATA = MappingProxyType({
    "wrong_delete_scope": CorruptedControlMetadataV3(
        control_id="wrong_delete_scope",
        target_layer="action_trace.delete_scope",
        expected_failure_flags=("wrong_delete_scope",),
    ),
    "missed_ttl": CorruptedControlMetadataV3(
        control_id="missed_ttl",
        target_layer="memory_state.ttl_expiry",
        expected_failure_flags=("ttl_violation",),
    ),
    "collateral_deletion": CorruptedControlMetadataV3(
        control_id="collateral_deletion",
        target_layer="memory_state.protected_collateral",
        expected_failure_flags=("collateral_mutation", "collateral_corruption"),
    ),
    "retained_forgotten_value": CorruptedControlMetadataV3(
        control_id="retained_forgotten_value",
        target_layer="memory_state.expected_absence",
        expected_failure_flags=("deletion_failure", "stale_retained"),
    ),
    "wrong_historical_version": CorruptedControlMetadataV3(
        control_id="wrong_historical_version",
        target_layer="answer.historical_version",
        expected_failure_flags=("version_confusion",),
    ),
    "wrong_history_order": CorruptedControlMetadataV3(
        control_id="wrong_history_order",
        target_layer="answer.history_order",
        expected_failure_flags=("version_confusion",),
    ),
    "stale_g_propagation": CorruptedControlMetadataV3(
        control_id="stale_g_propagation",
        target_layer="answer.g_synthesis",
        expected_failure_flags=("stale_propagation",),
    ),
    "fabricated_evidence": CorruptedControlMetadataV3(
        control_id="fabricated_evidence",
        target_layer="answer.evidence_linkage",
        expected_failure_flags=("evidence_linkage_error",),
    ),
})
CORRUPTED_CONTROL_IDS_V3 = tuple(_CONTROL_METADATA)


def corrupted_control_metadata_v3(control_id: str) -> CorruptedControlMetadataV3:
    try:
        return _CONTROL_METADATA[control_id]
    except KeyError as exc:
        raise ValueError(f"unknown strict-v3 corrupted control: {control_id}") from exc


def corrupted_control_adapter_info_v3(control_id: str) -> AdapterInfoV3:
    metadata = corrupted_control_metadata_v3(control_id)
    config = json.dumps(
        {
            "control_id": metadata.control_id,
            "target_layer": metadata.target_layer,
            "smoke_only": metadata.smoke_only,
            "leaderboard_eligible": metadata.leaderboard_eligible,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return AdapterInfoV3(
        adapter_id=f"corrupted_v3_{control_id}",
        adapter_version="1.0.0",
        system_name="mub_vnext_core_corrupted_control",
        system_version="1.0.0",
        configuration_hash=hashlib.sha256(config.encode("utf-8")).hexdigest(),
    )


def _json_text(value) -> str:
    value = thaw_json(value)
    return value if isinstance(value, str) else json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _replace_run(run: TaskRunRecordV3, **updates) -> TaskRunRecordV3:
    payload = run.model_dump(mode="python")
    payload.update(updates)
    return TaskRunRecordV3.model_validate(payload)


def _snapshot_hash_payload(payload: dict) -> str:
    canonical = json.dumps(
        {
            "after_event_id": payload["after_event_id"],
            "entries": payload["entries"],
            "state_by_object": payload["state_by_object"],
            "store_size": payload["store_size"],
            "raw_adapter_state": payload["raw_adapter_state"],
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _replace_snapshot(
    snapshot: MemorySnapshotV3,
    *,
    entries: tuple[MemoryEntryRecordV3, ...],
    state_by_object: dict,
) -> MemorySnapshotV3:
    payload = snapshot.model_dump(mode="json")
    payload["entries"] = [entry.model_dump(mode="json") for entry in entries]
    payload["state_by_object"] = state_by_object
    payload["store_size"] = len(entries)
    raw_state = payload["raw_adapter_state"]
    if isinstance(raw_state, dict):
        raw_state = dict(raw_state)
        raw_state["entries"] = list(payload["entries"])
        raw_state["state_by_object"] = dict(state_by_object)
        payload["raw_adapter_state"] = raw_state
    payload["snapshot_hash"] = _snapshot_hash_payload(payload)
    return MemorySnapshotV3.model_validate(payload)


def _identity(key) -> tuple[str, str, str, str | None]:
    return key.namespace, key.entity, key.attribute, key.subkey


def _retained_entry(task: MemUpdateTaskV3, key, control_id: str):
    ledger = next(
        ledger
        for ledger in task.version_history
        if _identity(ledger.object_key) == _identity(key)
    )
    tombstone_index = next(
        index
        for index, entry in enumerate(ledger.entries)
        if entry.status is LedgerEntryStatus.TOMBSTONE
    )
    retained = next(
        entry
        for entry in reversed(ledger.entries[:tombstone_index])
        if entry.status is LedgerEntryStatus.PRESENT
    )
    event_by_id = {event.event_id: event for event in task.events}
    source_event = event_by_id[retained.source_event_ids[-1]]
    return retained.value, MemoryEntryRecordV3(
        entry_id=f"corrupted_v3:{control_id}:{key.canonical_id}",
        content=f"{key.canonical_id} = {_json_text(retained.value)}",
        object_key_candidate=key,
        value_candidate=retained.value,
        created_at=source_event.timestamp,
        updated_at=source_event.timestamp,
        source_event_ids=retained.source_event_ids,
        version_index=retained.version_index,
        raw_metadata={
            "control_id": control_id,
            "entry_kind": "retained_forgotten_value",
        },
    )


def _retain_target_in_snapshot(
    task: MemUpdateTaskV3,
    snapshot: MemorySnapshotV3,
    key,
    control_id: str,
) -> MemorySnapshotV3:
    value, retained = _retained_entry(task, key, control_id)
    entries = tuple(
        entry
        for entry in snapshot.entries
        if entry.entry_id != retained.entry_id
    ) + (retained,)
    state = dict(snapshot.state_by_object)
    state[key.canonical_id] = thaw_json(value)
    return _replace_snapshot(snapshot, entries=entries, state_by_object=state)


def _wrong_delete_scope(task, run, metadata):
    delete = next(action for action in task.actions if action.operation is Operation.DELETE)
    replacement_scope = {
        ActionScope.OBJECT: ActionScope.ATTRIBUTE,
        ActionScope.ATTRIBUTE: ActionScope.ENTITY,
        ActionScope.ENTITY: ActionScope.NAMESPACE,
        ActionScope.NAMESPACE: ActionScope.NAMESPACE,
        ActionScope.TTL: ActionScope.OBJECT,
    }[delete.scope]
    if replacement_scope is delete.scope:
        raise ValueError(
            "wrong_delete_scope requires a contract-valid alternate scope"
        )
    actions = tuple(
        ParsedManagerActionV3.model_validate({
            **action.model_dump(mode="python"),
            "observed_scope": replacement_scope,
        })
        if action.event_id == delete.event_id
        else action
        for action in run.parsed_actions
    )
    return _replace_run(run, parsed_actions=actions)


def _missed_ttl(task, run, metadata):
    ttl = next(
        action
        for action in task.actions
        if action.operation is Operation.DELETE and action.scope is ActionScope.TTL
    )
    event_times = {
        event.event_id: event.timestamp
        for event in task.events
        if event.timestamp is not None
    }
    snapshots = tuple(
        _retain_target_in_snapshot(
            task,
            snapshot,
            ttl.target_object_keys[0],
            metadata.control_id,
        )
        if snapshot.after_event_id in event_times
        and ttl.effective_at is not None
        and event_times[snapshot.after_event_id] >= ttl.effective_at
        else snapshot
        for snapshot in run.memory_snapshots
    )
    return _replace_run(run, memory_snapshots=snapshots)


def _collateral_deletion(task, run, metadata):
    replay = replay_task_v3(task)
    protected = {key.canonical_id for key in replay.protected_collateral}
    if not protected:
        raise ValueError("collateral_deletion requires protected collateral")
    final = run.memory_snapshots[-1]
    entries = tuple(
        entry
        for entry in final.entries
        if entry.object_key_candidate is None
        or entry.object_key_candidate.canonical_id not in protected
    )
    state = {
        object_id: value
        for object_id, value in final.state_by_object.items()
        if object_id not in protected
    }
    changed = _replace_snapshot(final, entries=entries, state_by_object=state)
    return _replace_run(
        run,
        memory_snapshots=(*run.memory_snapshots[:-1], changed),
    )


def _retained_forgotten_value(task, run, metadata):
    delete = next(
        action
        for action in task.actions
        if action.operation is Operation.DELETE
        and action.scope is not ActionScope.TTL
    )
    final = _retain_target_in_snapshot(
        task,
        run.memory_snapshots[-1],
        delete.target_object_keys[0],
        metadata.control_id,
    )
    return _replace_run(
        run,
        memory_snapshots=(*run.memory_snapshots[:-1], final),
    )


def _replace_prediction(run, query_id: str, replacement) -> TaskRunRecordV3:
    predictions = tuple(
        replacement if prediction.query_id == query_id else prediction
        for prediction in run.answer_predictions
    )
    return _replace_run(run, answer_predictions=predictions)


def _wrong_historical_version(task, run, metadata):
    query = task.queries[0]
    replay = replay_task_v3(task)
    current = replay.current_state[query.target_object_keys[0].canonical_id].value
    baseline = next(
        prediction
        for prediction in run.answer_predictions
        if prediction.query_id == query.query_id
    )
    replacement = AnswerPredictionV3.model_validate({
        **baseline.model_dump(mode="python"),
        "raw_output": _json_text(current),
        "parsed_answer": current,
    })
    return _replace_prediction(run, query.query_id, replacement)


def _wrong_history_order(task, run, metadata):
    query = task.queries[0]
    baseline = next(
        prediction
        for prediction in run.answer_predictions
        if prediction.query_id == query.query_id
    )
    answer = list(reversed(thaw_json(baseline.parsed_answer)))
    replacement = AnswerPredictionV3.model_validate({
        **baseline.model_dump(mode="python"),
        "raw_output": _json_text(answer),
        "parsed_answer": answer,
    })
    return _replace_prediction(run, query.query_id, replacement)


def _stale_g_propagation(task, run, metadata):
    evidence = task.gold_evidence[0]
    if evidence.stale_alternative is None:
        raise ValueError("stale_g_propagation requires a stale alternative")
    baseline = next(
        prediction
        for prediction in run.answer_predictions
        if prediction.query_id == evidence.query_id
    )
    answer = evidence.stale_alternative.answer
    replacement = AnswerPredictionV3.model_validate({
        **baseline.model_dump(mode="python"),
        "raw_output": _json_text(answer),
        "parsed_answer": answer,
    })
    return _replace_prediction(run, evidence.query_id, replacement)


def _fabricated_evidence(task, run, metadata):
    query_id = task.queries[0].query_id
    baseline = next(
        prediction
        for prediction in run.answer_predictions
        if prediction.query_id == query_id
    )
    replacement = AnswerPredictionV3.model_validate({
        **baseline.model_dump(mode="python"),
        "cited_event_ids": (*baseline.cited_event_ids, "fabricated_event_v3"),
        "cited_derivation_step_ids": (
            *baseline.cited_derivation_step_ids,
            "fabricated_step_v3",
        ),
    })
    return _replace_prediction(run, query_id, replacement)


_TRANSFORMERS: Mapping[str, Callable] = MappingProxyType({
    "wrong_delete_scope": _wrong_delete_scope,
    "missed_ttl": _missed_ttl,
    "collateral_deletion": _collateral_deletion,
    "retained_forgotten_value": _retained_forgotten_value,
    "wrong_historical_version": _wrong_historical_version,
    "wrong_history_order": _wrong_history_order,
    "stale_g_propagation": _stale_g_propagation,
    "fabricated_evidence": _fabricated_evidence,
})


def apply_corrupted_control_v3(
    task: MemUpdateTaskV3,
    baseline_run: TaskRunRecordV3,
    control_id: str,
) -> TaskRunRecordV3:
    if not isinstance(task, MemUpdateTaskV3):
        raise TypeError("task must be a MemUpdateTaskV3")
    if not isinstance(baseline_run, TaskRunRecordV3):
        raise TypeError("baseline_run must be a TaskRunRecordV3")
    if baseline_run.task_id != task.task_id:
        raise ValueError("corrupted control task/run binding mismatch")
    metadata = corrupted_control_metadata_v3(control_id)
    transformed = _TRANSFORMERS[control_id](task, baseline_run, metadata)
    system_event = {
        "event": "corrupted_control_v3",
        "control_id": metadata.control_id,
        "target_layer": metadata.target_layer,
        "expected_failure_flags": list(metadata.expected_failure_flags),
        "smoke_only": metadata.smoke_only,
        "leaderboard_eligible": metadata.leaderboard_eligible,
    }
    return _replace_run(
        transformed,
        adapter_id=corrupted_control_adapter_info_v3(control_id).adapter_id,
        system_events=(*transformed.system_events, system_event),
    )


__all__ = [
    "CORRUPTED_CONTROL_IDS_V3",
    "CorruptedControlMetadataV3",
    "apply_corrupted_control_v3",
    "corrupted_control_adapter_info_v3",
    "corrupted_control_metadata_v3",
]
