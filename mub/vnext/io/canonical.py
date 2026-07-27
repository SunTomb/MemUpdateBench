from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from mub.vnext.contracts.task import MemUpdateTask


def canonical_json_bytes(model: BaseModel) -> bytes:
    """Serialize a Pydantic model as canonical compact UTF-8 JSON without a newline."""
    dump_options = {
        "exclude_none": False,
        "exclude_computed_fields": True,
    }
    _reject_nonfinite_values(model.model_dump(mode="python", **dump_options))
    payload = model.model_dump(mode="json", **dump_options)
    return _canonical_payload_bytes(payload)


def sha256_model(model: BaseModel) -> str:
    """Return the lowercase SHA-256 hex digest of ``canonical_json_bytes(model)``."""
    return hashlib.sha256(canonical_json_bytes(model)).hexdigest()


def semantic_task_hash(task: MemUpdateTask) -> str:
    """Hash the task's ID-independent semantic-core projection.

    The projection includes family semantics, normalized source identity, event
    order/roles/anchors/action ownership, exact object identities, ordered gold
    actions, query semantics, and gold state/history/answers/source support. It
    intentionally excludes difficulty and other artifact, split, compiler, legacy,
    tag, surface-text, and surface-ID fields. ``object_type`` is also excluded
    because vNext object reference identity is exactly
    ``(namespace, entity, attribute, subkey)``.
    """
    dump_options = {
        "exclude_none": False,
        "exclude_computed_fields": True,
    }
    _reject_nonfinite_values(task.model_dump(mode="python", **dump_options))
    payload = task.model_dump(mode="json", **dump_options)
    projection = _semantic_task_projection(payload)
    return hashlib.sha256(_canonical_payload_bytes(projection)).hexdigest()


def _reject_nonfinite_values(value: Any, path: str = "$") -> None:
    if isinstance(value, (set, frozenset)):
        raise ValueError(
            f"canonical JSON cannot contain unordered container "
            f"{type(value).__name__} at {path}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"canonical JSON cannot contain non-finite float at {path}")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError(f"canonical JSON cannot contain non-finite Decimal at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_values(key, f"{path}.<key>")
            _reject_nonfinite_values(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_values(item, f"{path}[{index}]")


def _canonical_payload_bytes(payload: Any) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("canonical JSON cannot contain non-finite float values") from exc
    return serialized.encode("utf-8")


def _object_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": record["namespace"],
        "entity": record["entity"],
        "attribute": record["attribute"],
        "subkey": record["subkey"],
    }


def _sorted_object_identities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = [_object_identity(record) for record in records]
    return sorted(projected, key=_canonical_payload_bytes)


def _unresolved_reference_projection(
    query: dict[str, Any], canonical: dict[str, Any]
) -> dict[str, Any]:
    candidates = query.get("reference_candidates", [])
    candidate_indices = {
        candidate["candidate_id"]: index
        for index, candidate in enumerate(candidates)
    }
    projected_references = []
    for reference in query.get("surface_references", []):
        projected_references.append(
            {
                "condition_kind": reference["condition_kind"],
                "evidence_kind": reference["evidence_kind"],
                "candidate_indices": [
                    candidate_indices[candidate_id]
                    for candidate_id in reference["candidate_ids"]
                ],
            }
        )
    return {
        "reference_candidates": [
            {"identity": _object_identity(candidate["object_key"])}
            for candidate in candidates
        ],
        "surface_references": projected_references,
        "canonical_answer": {
            "disposition": canonical["disposition"],
            "resolution_status": canonical["resolution_status"],
            "selected_candidate_indices": [
                candidate_indices[candidate_id]
                for candidate_id in canonical["selected_candidate_ids"]
            ],
            "value": canonical["value"],
        },
    }


def _semantic_task_projection(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload["events"]
    gold = payload["gold"]
    actions_by_id = {action["action_id"]: action for action in gold["actions"]}
    action_order = {action_id: index for index, action_id in enumerate(gold["action_sequence"])}
    event_order = {event["event_id"]: index for index, event in enumerate(events)}

    ordered_actions = []
    for action_id in gold["action_sequence"]:
        action = actions_by_id[action_id]
        ordered_actions.append(
            {
                "event_index": event_order[action["event_id"]],
                "operation": action["operation"],
                "scope": action["scope"],
                "target_object_keys": _sorted_object_identities(
                    action["target_object_keys"]
                ),
                "value": action["value"],
                "effective_at": action["effective_at"],
                "expected_effect": action["expected_effect"],
            }
        )

    projected_events = []
    for event in events:
        projected_events.append(
            {
                "sequence_index": event["sequence_index"],
                "timestamp": event["timestamp"],
                "role": event["role"],
                "source_anchor": event["source_anchor"],
                "owned_action_indices": sorted(
                    action_order[action_id] for action_id in event["gold_action_ids"]
                ),
            }
        )

    projected_queries = []
    for query in payload["queries"]:
        query_id = query["query_id"]
        projected_query = {
            "query_type": query["query_type"],
            "target_object_keys": _sorted_object_identities(
                query["target_object_keys"]
            ),
            "answer_schema": query["answer_schema"],
            "evaluation_mode": query["evaluation_mode"],
        }
        if query["query_type"] == "unresolved_reference":
            projected_query["reference_resolution"] = _unresolved_reference_projection(
                query, gold["canonical_answers"][query_id]
            )
        else:
            projected_query.update(
                {
                    "gold_answer": gold["gold_answers"][query_id],
                    "acceptable_answers": gold["acceptable_answers"][query_id],
                }
            )
        projected_queries.append(projected_query)
    projected_queries.sort(key=_canonical_payload_bytes)

    source = payload["source"]
    return {
        "task_family": payload["task_family"],
        "source": {
            "source_type": source["source_type"],
            "normalized_hash": source["normalized_hash"],
            "normalization_version": source["normalization_version"],
        },
        "target_objects": _sorted_object_identities(payload["target_objects"]),
        "events": projected_events,
        "gold_actions": ordered_actions,
        "queries": projected_queries,
        "gold": {
            "final_state": gold["final_state"],
            "version_history": gold["version_history"],
            "expected_present_objects": _sorted_object_identities(
                gold["expected_present_objects"]
            ),
            "expected_absent_objects": _sorted_object_identities(
                gold["expected_absent_objects"]
            ),
            "source_event_indices": sorted(
                event_order[event_id] for event_id in gold["gold_source_event_ids"]
            ),
        },
    }


__all__ = ["canonical_json_bytes", "semantic_task_hash", "sha256_model"]
