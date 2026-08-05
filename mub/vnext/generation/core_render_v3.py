from __future__ import annotations

from typing import Any

from mub.vnext.contracts.enums import ActionScope, AnswerSchema, Operation, QueryType, Split, TaskFamily
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
from mub.vnext.generation.render import render_core_with_catalog
from mub.vnext.validation.replay_v3 import replay_task_v3


def _key_payload(key: Any) -> dict[str, Any]:
    return key.model_dump(mode="python")


def _identity(value: Any) -> tuple[str, str, str, str | None]:
    if isinstance(value, dict):
        return (
            value["namespace"],
            value["entity"],
            value["attribute"],
            value.get("subkey"),
        )
    return value.namespace, value.entity, value.attribute, value.subkey


def _promote_actions(task, core: SemanticCore) -> list[dict[str, Any]]:
    promoted = []
    family_e = core.task_family is TaskFamily.DELETION_FORGETTING
    for action in task.gold.actions:
        if action.operation is Operation.NOOP:
            scope = None
            targets = []
        elif family_e:
            scope = action.scope.value
            targets = [_key_payload(key) for key in action.target_object_keys]
        else:
            if len(action.target_object_keys) != 1:
                raise ValueError("Core v3 promotion requires one target per mutation")
            scope = ActionScope.OBJECT.value
            targets = [_key_payload(action.target_object_keys[0])]
        promoted.append(
            {
                "action_id": action.action_id,
                "event_id": action.event_id,
                "operation": action.operation.value,
                "scope": scope,
                "target_object_keys": targets,
                "value": action.value,
                "effective_at": action.effective_at,
                "expected_effect": action.expected_effect,
            }
        )
    return promoted


def _build_version_history(task, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_by_identity = {_identity(key): key for key in task.target_objects}
    entries_by_identity: dict[
        tuple[str, str, str, str | None], list[dict[str, Any]]
    ] = {identity: [] for identity in target_by_identity}
    pending_ttl: list[dict[str, Any]] = []

    def append_entry(action: dict[str, Any], key: dict[str, Any], *, ttl: bool = False) -> None:
        operation = Operation(action["operation"])
        status = "tombstone" if operation is Operation.DELETE else "present"
        identity = _identity(key)
        if identity not in entries_by_identity:
            raise ValueError("Core v3 action targets an undeclared object")
        entries = entries_by_identity[identity]
        boundary_event = None if ttl else action["event_id"]
        if entries and not ttl:
            entries[-1]["valid_until_event_id"] = boundary_event
        entries.append(
            {
                "version_index": len(entries),
                "status": status,
                "value": None if status == "tombstone" else action["value"],
                "valid_from_event_id": boundary_event,
                "valid_until_event_id": None,
                "logical_time": action["effective_at"],
                "source_event_ids": [action["event_id"]],
            }
        )

    for action in actions:
        operation = Operation(action["operation"])
        if operation is Operation.NOOP:
            continue
        if operation is Operation.DELETE and action["scope"] == ActionScope.TTL.value:
            pending_ttl.append(action)
            continue
        for key in action["target_object_keys"]:
            append_entry(action, key)
    for action in sorted(pending_ttl, key=lambda item: item["effective_at"]):
        append_entry(action, action["target_object_keys"][0], ttl=True)

    missing = [identity for identity, entries in entries_by_identity.items() if not entries]
    if missing:
        raise ValueError(f"Core v3 targets require mutation history: {missing}")
    return [
        {
            "object_key": _key_payload(key),
            "entries": entries_by_identity[identity],
        }
        for identity, key in target_by_identity.items()
    ]


def _promote_reference_query_and_evidence(task, query, version_history):
    targets = [_key_payload(key) for key in query.target_object_keys]
    selector = {
        "kind": "reference_resolution",
        "reference_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "object_key": _key_payload(candidate.object_key),
                "evidence": candidate.evidence,
                "source_anchors": [
                    anchor.model_dump(mode="python")
                    for anchor in candidate.source_anchors
                ],
            }
            for candidate in query.reference_candidates
        ],
        "surface_references": [
            reference.model_dump(mode="python")
            for reference in query.surface_references
        ],
    }
    promoted_query = {
        "query_id": query.query_id,
        "query_type": QueryTypeV3.UNRESOLVED_REFERENCE.value,
        "text": query.text,
        "selector": selector,
        "target_object_keys": targets,
        "answer_schema": query.answer_schema.value,
        "evaluation_mode": query.evaluation_mode.value,
        "synthesis": None,
    }

    history_by_identity = {
        _identity(ledger["object_key"]): ledger for ledger in version_history
    }
    supporting_events = []
    steps = []
    for index, key in enumerate(query.target_object_keys):
        ledger = history_by_identity[_identity(key)]
        event_ids = list(ledger["entries"][-1]["source_event_ids"])
        supporting_events.extend(event_ids)
        steps.append(
            {
                "step_id": f"derive_{query.query_id}_{index}",
                "operation": "read_current",
                "input_step_ids": [],
                "supporting_object_keys": [_key_payload(key)],
                "supporting_event_ids": event_ids,
            }
        )
    final_step_id = steps[0]["step_id"]
    if len(steps) > 1:
        final_step_id = f"derive_{query.query_id}_reference"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "collect",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": targets,
                "supporting_event_ids": list(dict.fromkeys(supporting_events)),
            }
        )
    canonical = task.gold.canonical_answers.get(query.query_id)
    if canonical is None:
        raise ValueError("Core v3 reference-resolution query requires a canonical answer")
    evidence = {
        "query_id": query.query_id,
        "answer": canonical.value,
        "disposition": canonical.disposition.value,
        "resolution_status": canonical.resolution_status.value,
        "selected_candidate_ids": list(canonical.selected_candidate_ids),
        "abstention_reason": canonical.abstention_reason,
        "supporting_object_keys": targets,
        "supporting_event_ids": list(dict.fromkeys(supporting_events)),
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
        "stale_alternative": None,
    }
    return promoted_query, evidence


def _promote_family_e_query_and_evidence(task, core: SemanticCore, version_history):
    if len(task.queries) != 1:
        raise ValueError("Core Family E v3 promotion requires exactly one query")
    source_query = task.queries[0]
    targets = [_key_payload(key) for key in core.query_targets]
    history_by_identity = {
        _identity(ledger["object_key"]): ledger for ledger in version_history
    }
    query_time = core.stratification.get("query_logical_time")
    selected_entries = []
    for key in core.query_targets:
        entries = history_by_identity[_identity(key)]["entries"]
        if query_time is not None:
            eligible = [
                entry
                for entry in entries
                if entry["logical_time"] is None or entry["logical_time"] <= query_time
            ]
            if not eligible:
                raise ValueError("Core Family E logical query precedes target history")
            selected_entries.append(eligible[-1])
        else:
            selected_entries.append(entries[-1])

    values = [
        None if entry["status"] == "tombstone" else entry["value"]
        for entry in selected_entries
    ]
    multi_object = len(targets) > 1
    if multi_object:
        query_type = QueryTypeV3.MULTI_OBJECT_CURRENT.value
        selector = {"kind": "multi_object_current", "object_keys": targets}
    elif query_time is not None:
        query_type = QueryTypeV3.POINT_IN_TIME.value
        selector = {"kind": "logical_time_anchor", "logical_time": query_time}
    else:
        query_type = QueryTypeV3.CURRENT.value
        selector = {"kind": "current"}
    answer = values
    answer_schema = AnswerSchema.LIST.value

    promoted_query = {
        "query_id": source_query.query_id,
        "query_type": query_type,
        "text": source_query.text,
        "selector": selector,
        "target_object_keys": targets,
        "answer_schema": answer_schema,
        "evaluation_mode": source_query.evaluation_mode.value,
        "synthesis": None,
    }
    supporting_events: list[str] = []
    steps = []
    for index, (key, entry) in enumerate(zip(core.query_targets, selected_entries)):
        event_ids = list(entry["source_event_ids"])
        supporting_events.extend(event_ids)
        steps.append(
            {
                "step_id": f"derive_{source_query.query_id}_{index}",
                "operation": "read_current" if query_time is None else "read_version",
                "input_step_ids": [],
                "supporting_object_keys": [_key_payload(key)],
                "supporting_event_ids": event_ids,
            }
        )
    final_step_id = steps[0]["step_id"]
    if multi_object:
        final_step_id = f"derive_{source_query.query_id}_final"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "multi_object",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": targets,
                "supporting_event_ids": list(dict.fromkeys(supporting_events)),
            }
        )
    evidence = {
        "query_id": source_query.query_id,
        "answer": answer,
        "supporting_object_keys": targets,
        "supporting_event_ids": list(dict.fromkeys(supporting_events)),
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
        "stale_alternative": None,
    }
    return promoted_query, evidence


def _promote_query_and_evidence(task, version_history, core: SemanticCore):
    if core.task_family is TaskFamily.DELETION_FORGETTING:
        return _promote_family_e_query_and_evidence(task, core, version_history)
    if len(task.queries) != 1:
        raise ValueError("Core v3 promotion requires exactly one query")
    query = task.queries[0]
    if query.query_type is QueryType.UNRESOLVED_REFERENCE:
        return _promote_reference_query_and_evidence(task, query, version_history)
    if query.query_type is not QueryType.CURRENT_STATE:
        raise ValueError("Core v3 promotion supports current-state and unresolved-reference queries only")

    targets = [_key_payload(key) for key in query.target_object_keys]
    multi_object = len(targets) > 1
    query_type = (
        QueryTypeV3.MULTI_OBJECT_CURRENT.value
        if multi_object
        else QueryTypeV3.CURRENT.value
    )
    selector = (
        {"kind": "multi_object_current", "object_keys": targets}
        if multi_object
        else {"kind": "current"}
    )
    promoted_query = {
        "query_id": query.query_id,
        "query_type": query_type,
        "text": query.text,
        "selector": selector,
        "target_object_keys": targets,
        "answer_schema": query.answer_schema.value,
        "evaluation_mode": query.evaluation_mode.value,
        "synthesis": None,
    }

    history_by_identity = {
        _identity(ledger["object_key"]): ledger for ledger in version_history
    }
    supporting_events = []
    steps = []
    for index, key in enumerate(query.target_object_keys):
        ledger = history_by_identity[_identity(key)]
        event_ids = list(ledger["entries"][-1]["source_event_ids"])
        supporting_events.extend(event_ids)
        steps.append(
            {
                "step_id": f"derive_{query.query_id}_{index}",
                "operation": "read_current",
                "input_step_ids": [],
                "supporting_object_keys": [_key_payload(key)],
                "supporting_event_ids": event_ids,
            }
        )
    final_step_id = steps[0]["step_id"]
    if multi_object:
        final_step_id = f"derive_{query.query_id}_final"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "multi_object",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": targets,
                "supporting_event_ids": list(dict.fromkeys(supporting_events)),
            }
        )
    answer = task.gold.gold_answers.get(query.query_id)
    if query.query_id not in task.gold.gold_answers:
        raise ValueError("Core v3 current-state query requires a direct gold answer")
    evidence = {
        "query_id": query.query_id,
        "answer": answer,
        "supporting_object_keys": targets,
        "supporting_event_ids": list(dict.fromkeys(supporting_events)),
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
        "stale_alternative": None,
    }
    return promoted_query, evidence


def render_core_v3(
    core: SemanticCore,
    *,
    split: Split,
    surface_variant: int,
    context: GenerationContext,
) -> MemUpdateTaskV3:
    """Render one Core A-D current-state semantic core as a strict v3 task."""
    task_v2 = render_core_with_catalog(
        core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        surface_catalog=CORE_SURFACE_CATALOG_V1,
    )
    actions = _promote_actions(task_v2, core)
    version_history = _build_version_history(task_v2, actions)
    query, evidence = _promote_query_and_evidence(task_v2, version_history, core)

    source = task_v2.source.model_dump(mode="python")
    source["provenance"] = dict(source["provenance"])
    source["provenance"]["schema_version"] = "3.0.0"
    metadata = task_v2.metadata.model_dump(mode="python")
    if core.task_family is TaskFamily.DELETION_FORGETTING:
        protected_ids = [
            key.canonical_id
            for event in core.events
            if event.metadata.get("protected_collateral") is True
            for key in event.object_keys
        ]
        metadata["extra"] = dict(metadata["extra"])
        metadata["extra"]["family_e"] = {
            "protected_collateral_ids": protected_ids,
        }
    payload = {
        "task_id": task_v2.task_id,
        "schema_version": "3.0.0",
        "task_family": task_v2.task_family,
        "difficulty": task_v2.difficulty,
        "source": source,
        "events": [event.model_dump(mode="python") for event in task_v2.events],
        "target_objects": [_key_payload(key) for key in task_v2.target_objects],
        "actions": actions,
        "queries": [query],
        "version_history": version_history,
        "gold_evidence": [evidence],
        "metadata": metadata,
    }
    promoted = MemUpdateTaskV3.model_validate(payload)
    replay = replay_task_v3(promoted)
    if replay.issues:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in replay.issues)
        raise ValueError(f"Core v3 replay validation failed: {details}")
    return promoted


__all__ = ["render_core_v3"]
