from __future__ import annotations

from typing import Any

from mub.vnext.contracts.enums import (
    ActionScope,
    AnswerDisposition,
    AnswerSchema,
    Operation,
    QueryType,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3, VersionHistoryLedger
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
from mub.vnext.generation.render import (
    _payload_sha256,
    _render_query_text,
    render_core_with_catalog,
)
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3


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


def _inject_family_e_surface_cues(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    query: dict[str, Any],
) -> None:
    deletes = [action for action in actions if action["operation"] == Operation.DELETE.value]
    if not deletes:
        query["text"] += " UPDATE_NOT_DELETE: apply the correction as an update."
        return
    event_by_id = {event["event_id"]: event for event in events}
    for action in deletes:
        event = event_by_id[action["event_id"]]
        target_ids = ",".join(
            "|".join(
                (
                    key["namespace"],
                    key["entity"],
                    key["attribute"],
                    key.get("subkey") or "",
                )
            )
            for key in action["target_object_keys"]
        )
        scope = action["scope"]
        logical_time = event["timestamp"]
        effective_at = action["effective_at"]
        common = (
            f"scope={scope}; enumerated_targets={target_ids}; "
            f"event_logical_time={logical_time}; effective_at={effective_at}"
        )
        if scope == ActionScope.TTL.value:
            event["raw_text"] += (
                f" [{common}; scheduled_at={logical_time}]"
            )
            boundary = query["selector"]["logical_time"]
            query["text"] += (
                f" [{common}; scheduled_at={logical_time}; "
                f"boundary_anchor={boundary}]"
            )
        else:
            event["raw_text"] += f" [{common}]"
            query["text"] += f" [{common}]"


def _promote_actions(task, core: SemanticCore) -> list[dict[str, Any]]:
    if len(task.gold.actions) != len(core.events):
        raise ValueError("Core v3 promotion requires one action per semantic event")
    promoted = []
    family_e = core.task_family is TaskFamily.DELETION_FORGETTING
    family_f = core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY
    family_g = core.task_family is TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS
    for action, core_event in zip(task.gold.actions, core.events):
        if action.operation is Operation.NOOP:
            scope = None
            targets = []
        elif family_e:
            scope = core_event.metadata["action_scope"]
            targets = [_key_payload(key) for key in core_event.object_keys]
        else:
            if len(action.target_object_keys) != 1:
                raise ValueError("Core v3 promotion requires one target per mutation")
            scope = ActionScope.OBJECT.value
            targets = [_key_payload(action.target_object_keys[0])]
        effective_at = (
            core_event.metadata.get("effective_at")
            if family_e
            else core_event.metadata.get("logical_time")
            if family_f or family_g
            else action.effective_at
        )
        promoted.append(
            {
                "action_id": action.action_id,
                "event_id": action.event_id,
                "operation": action.operation.value,
                "scope": scope,
                "target_object_keys": targets,
                "value": action.value,
                "effective_at": effective_at,
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

    canonical = task.gold.canonical_answers.get(query.query_id)
    if canonical is None:
        raise ValueError("Core v3 reference-resolution query requires a canonical answer")
    candidate_keys = {
        candidate.candidate_id: candidate.object_key
        for candidate in query.reference_candidates
    }
    if canonical.disposition is AnswerDisposition.ANSWERED:
        evidence_keys = [candidate_keys[canonical.selected_candidate_ids[0]]]
    else:
        evidence_keys = list(query.target_object_keys)

    history_by_identity = {
        _identity(ledger["object_key"]): ledger for ledger in version_history
    }
    supporting_events = []
    steps = []
    for index, key in enumerate(evidence_keys):
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
    if canonical.disposition is AnswerDisposition.ABSTAINED:
        final_step_id = f"derive_{query.query_id}_abstain"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "abstain",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": [
                    _key_payload(key) for key in evidence_keys
                ],
                "supporting_event_ids": list(dict.fromkeys(supporting_events)),
            }
        )
    evidence_targets = targets
    evidence = {
        "query_id": query.query_id,
        "answer": canonical.value,
        "disposition": canonical.disposition.value,
        "resolution_status": canonical.resolution_status.value,
        "selected_candidate_ids": list(canonical.selected_candidate_ids),
        "abstention_reason": canonical.abstention_reason,
        "supporting_object_keys": evidence_targets,
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

    surface_variant = task.metadata.extra["surface_variant"]
    current_query_template = CORE_SURFACE_CATALOG_V1.template_sets[surface_variant][5]
    query_text = _render_query_text(core, current_query_template) + (
        " Return a list aligned to the target order, using null for a "
        "missing current value."
    )
    promoted_query = {
        "query_id": source_query.query_id,
        "query_type": query_type,
        "text": query_text,
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
    final_step_id = f"derive_{source_query.query_id}_final"
    steps.append(
        {
            "step_id": final_step_id,
            "operation": "list",
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


def _family_f_query_material(task, core: SemanticCore, version_history):
    from mub.vnext.generation.family_f import bind_family_f_core_selector

    selector, resolution, selected, entries = bind_family_f_core_selector(
        core,
        tuple(event.event_id for event in task.events),
    )
    declared = VersionHistoryLedger.model_validate(version_history[0])
    if declared.entries != entries:
        raise ValueError("Core Family F promoted ledger differs from selector source")
    return selector, resolution, selected


def _promote_family_f_query_and_evidence(task, core: SemanticCore, version_history):
    if len(task.queries) != 1 or len(version_history) != 1:
        raise ValueError("Core Family F v3 promotion requires one query and one ledger")
    source_query = task.queries[0]
    selector, resolution, selected = _family_f_query_material(
        task, core, version_history
    )
    kind = selector.kind
    target = _key_payload(core.query_targets[0])
    promoted_query = {
        "query_id": source_query.query_id,
        "query_type": resolution.task_query_type.value,
        "text": source_query.text,
        "selector": selector.model_dump(mode="python"),
        "target_object_keys": [target],
        "answer_schema": resolution.answer_schema.value,
        "evaluation_mode": source_query.evaluation_mode.value,
        "synthesis": None,
    }
    read_steps = [
        {
            "step_id": f"derive_{source_query.query_id}_{index}",
            "operation": "read_current" if kind == "current" else "read_version",
            "input_step_ids": [],
            "supporting_object_keys": [target],
            "supporting_event_ids": list(entry.source_event_ids),
        }
        for index, entry in enumerate(selected)
    ]
    final_step_id = read_steps[0]["step_id"]
    steps = list(read_steps)
    if len(read_steps) > 1:
        final_step_id = f"derive_{source_query.query_id}_final"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "transition" if kind == "transition" else "ordered_history",
                "input_step_ids": [step["step_id"] for step in read_steps],
                "supporting_object_keys": [target],
                "supporting_event_ids": list(
                    dict.fromkeys(
                        event_id
                        for entry in selected
                        for event_id in entry.source_event_ids
                    )
                ),
            }
        )
    supporting_events = list(
        dict.fromkeys(
            event_id
            for entry in selected
            for event_id in entry.source_event_ids
        )
    )
    evidence = {
        "query_id": source_query.query_id,
        "answer": resolution.answer,
        "supporting_object_keys": [target],
        "supporting_event_ids": supporting_events,
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
        "stale_alternative": None,
    }
    return promoted_query, evidence


_FAMILY_G_QUERY_TEMPLATES = (
    "Using the listed objects in order, $instruction",
    "For the ordered object sequence, $instruction",
    "Follow the typed derivation over the objects in their displayed order: $instruction",
    "To resolve the ordered current-state derivation, $instruction",
)


def _family_g_ordered_keys(core: SemanticCore):
    ordered = []
    seen = set()
    for event in core.events:
        for key in event.object_keys:
            identity = _identity(key)
            if identity not in seen:
                ordered.append(key)
                seen.add(identity)
    return tuple(ordered)


def _family_g_event_staging_core(core: SemanticCore) -> SemanticCore:
    from mub.vnext.generation.family_g import validate_family_g_core

    validate_family_g_core(core)
    current_by_identity = {}
    for event in core.events:
        for key in event.object_keys:
            if event.operation in {Operation.ADD, Operation.UPDATE}:
                current_by_identity[_identity(key)] = event.value
    try:
        values = [
            current_by_identity[_identity(key)] for key in core.query_targets
        ]
    except KeyError as exc:
        raise ValueError(
            "Core Family G staging requires one current value per operand"
        ) from exc
    return core.model_copy(update={"expected_answer": values})


def _family_g_query_text(task, core: SemanticCore, keys) -> str:
    kind = core.stratification["synthesis_kind"]
    object_order = ", ".join(key.canonical_id for key in keys)
    if kind == "update_sensitive_multi_hop":
        instruction = (
            "read each object's current numeric operand, start with the first operand, "
            "then subtract each later operand from the running result; return only the final number."
        )
    elif core.stratification["answer_kind"] == "boolean_consistency":
        instruction = (
            "read every current consistency code and return true exactly when all codes are equal, "
            "otherwise false."
        )
    else:
        instruction = (
            "read every current discrepancy code and add them; code 0 means consistent and a positive "
            "code is the exact 1-based position of the inconsistent object; return only that position."
        )
    template = _FAMILY_G_QUERY_TEMPLATES[task.metadata.extra["surface_variant"]]
    return template.replace("$instruction", instruction) + f" [object_order={object_order}]"


def _family_g_derivation(query_id, keys, selected_entries, *, kind: str, prefix: str):
    key_payloads = [_key_payload(key) for key in keys]
    steps = []
    supporting_events = []
    values = []
    for index, (key, entry, stale) in enumerate(selected_entries):
        event_ids = list(entry["source_event_ids"])
        supporting_events.extend(event_ids)
        values.append(entry["value"])
        steps.append(
            {
                "step_id": f"{prefix}_{query_id}_read_{index}",
                "operation": "read_version" if stale else "read_current",
                "input_step_ids": [],
                "supporting_object_keys": [_key_payload(key)],
                "supporting_event_ids": event_ids,
            }
        )
    if kind == "update_sensitive_multi_hop":
        answer = values[0]
        left_step = steps[0]["step_id"]
        for index, value in enumerate(values[1:], start=1):
            answer -= value
            final_step = f"{prefix}_{query_id}_subtract_{index}"
            steps.append(
                {
                    "step_id": final_step,
                    "operation": "subtract",
                    "input_step_ids": [left_step, steps[index]["step_id"]],
                    "supporting_object_keys": [],
                    "supporting_event_ids": [],
                }
            )
            left_step = final_step
        final_step_id = left_step
    elif kind == "boolean_consistency":
        answer = all(value == values[0] for value in values[1:])
        final_step_id = f"{prefix}_{query_id}_equals"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "equals",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": [],
                "supporting_event_ids": [],
            }
        )
    else:
        answer = sum(values)
        final_step_id = f"{prefix}_{query_id}_add"
        steps.append(
            {
                "step_id": final_step_id,
                "operation": "add",
                "input_step_ids": [step["step_id"] for step in steps],
                "supporting_object_keys": [],
                "supporting_event_ids": [],
            }
        )
    return {
        "answer": answer,
        "supporting_object_keys": key_payloads,
        "supporting_event_ids": list(dict.fromkeys(supporting_events)),
        "derivation_steps": steps,
        "final_derivation_step_id": final_step_id,
    }


def _promote_family_g_query_and_evidence(task, core: SemanticCore, version_history):
    if len(task.queries) != 1:
        raise ValueError("Core Family G v3 promotion requires exactly one query")
    source_query = task.queries[0]
    keys = _family_g_ordered_keys(core)
    history_by_identity = {
        _identity(ledger["object_key"]): ledger["entries"]
        for ledger in version_history
    }
    if set(history_by_identity) != {_identity(key) for key in keys}:
        raise ValueError("Core Family G promoted ledgers differ from semantic operands")
    current_entries = [
        (key, history_by_identity[_identity(key)][-1], False)
        for key in keys
    ]
    kind = core.stratification["synthesis_kind"]
    query_targets = [_key_payload(key) for key in keys]
    selector = {"kind": "multi_object_current", "object_keys": query_targets}
    if kind == "update_sensitive_multi_hop":
        query_type = QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP.value
        synthesis = {"kind": kind, "minimum_hops": core.stratification["hop_count"]}
        derivation_kind = kind
        answer_schema = AnswerSchema.NUMBER.value
        stale_indices = {core.stratification["stale_operand_index"]}
    else:
        query_type = QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY.value
        synthesis = {"kind": kind, "minimum_objects": core.stratification["object_count"]}
        derivation_kind = core.stratification["answer_kind"]
        answer_schema = (
            AnswerSchema.BOOLEAN.value
            if derivation_kind == "boolean_consistency"
            else AnswerSchema.NUMBER.value
        )
        stale_indices = {
            int(value)
            for value in core.stratification["stale_indices"].split(",")
        }
    stale_entries = [
        (
            key,
            history_by_identity[_identity(key)][0 if index in stale_indices else -1],
            index in stale_indices,
        )
        for index, key in enumerate(keys)
    ]
    primary = _family_g_derivation(
        source_query.query_id,
        keys,
        current_entries,
        kind=derivation_kind,
        prefix="gold",
    )
    stale = _family_g_derivation(
        source_query.query_id,
        keys,
        stale_entries,
        kind=derivation_kind,
        prefix="stale",
    )
    promoted_query = {
        "query_id": source_query.query_id,
        "query_type": query_type,
        "text": _family_g_query_text(task, core, keys),
        "selector": selector,
        "target_object_keys": query_targets,
        "answer_schema": answer_schema,
        "evaluation_mode": source_query.evaluation_mode.value,
        "synthesis": synthesis,
    }
    evidence = {
        "query_id": source_query.query_id,
        **primary,
        "stale_alternative": stale,
    }
    return promoted_query, evidence


def _inject_family_f_event_cues(events, version_history) -> None:
    event_by_id = {event["event_id"]: event for event in events}
    for entry in version_history[0]["entries"]:
        event_id = entry["source_event_ids"][0]
        event_by_id[event_id]["raw_text"] += (
            " ["
            f"version_index={entry['version_index']}; "
            f"event_id={event_id}; "
            f"logical_time={entry['logical_time']}"
            "]"
        )


def _promote_query_and_evidence(task, version_history, core: SemanticCore):
    if core.task_family is TaskFamily.DELETION_FORGETTING:
        return _promote_family_e_query_and_evidence(task, core, version_history)
    if core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY:
        return _promote_family_f_query_and_evidence(task, core, version_history)
    if core.task_family is TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS:
        return _promote_family_g_query_and_evidence(task, core, version_history)
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
    """Render one Core semantic core as a strict v3 task."""
    family_e = core.task_family is TaskFamily.DELETION_FORGETTING
    family_f = core.task_family is TaskFamily.CURRENT_HISTORICAL_QUERY
    family_g = core.task_family is TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS
    if family_e:
        from mub.vnext.generation.family_e import validate_family_e_core

        validate_family_e_core(core)
    if family_f:
        from mub.vnext.generation.family_f import validate_family_f_core

        validate_family_f_core(core)
    render_source_core = core
    if family_g:
        render_source_core = _family_g_event_staging_core(core)
    task_v2 = render_core_with_catalog(
        render_source_core,
        split=split,
        surface_variant=surface_variant,
        context=context,
        surface_catalog=CORE_SURFACE_CATALOG_V1,
    )
    actions = _promote_actions(task_v2, core)
    version_history = _build_version_history(task_v2, actions)
    query, evidence = _promote_query_and_evidence(task_v2, version_history, core)
    events = [event.model_dump(mode="python") for event in task_v2.events]
    if family_e or family_f or family_g:
        for event, core_event in zip(events, core.events):
            event["timestamp"] = core_event.metadata["logical_time"]
    if family_e:
        _inject_family_e_surface_cues(actions, events, query)
    if family_f:
        _inject_family_f_event_cues(events, version_history)

    source = task_v2.source.model_dump(mode="python")
    source["provenance"] = dict(source["provenance"])
    source["provenance"]["schema_version"] = "3.0.0"
    if family_e or family_f or family_g:
        source["raw_hash"] = _payload_sha256(
            {
                "events": [
                    {"raw_text": event["raw_text"], "speaker": event["speaker"]}
                    for event in events
                ],
                "query_text": query["text"],
            }
        )
    metadata = task_v2.metadata.model_dump(mode="python")
    if family_e:
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
        "events": events,
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
    query_by_id = {query.query_id: query for query in promoted.queries}
    for gold_evidence in promoted.gold_evidence:
        evaluation = evaluate_evidence_v3(
            gold_evidence,
            replay,
            gold_evidence.stale_alternative,
            query_by_id[gold_evidence.query_id],
            promoted.events,
        )
        if evaluation.issues:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in evaluation.issues
            )
            raise ValueError(f"Core v3 evidence validation failed: {details}")
    if family_f:
        from mub.vnext.generation.family_f import validate_family_f_task

        validate_family_f_task(promoted)
    if family_g:
        from mub.vnext.generation.family_g import validate_family_g_task

        validate_family_g_task(promoted)
    return promoted


__all__ = ["render_core_v3"]
