"""Execute and audit the bounded native multi-object API canary.

This module deliberately keeps transport construction outside the runner.  The
factory is the only injection point for a provider and its cleanup operation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import math

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, object_identity, typed_json_equal
from mub.vnext.contracts.v3.native_multi_object import (
    NativeMultiObjectMutationRequestV3,
    NativeRetrievalRequestV3,
)


_CASE_COUNT = 8
_MAX_SCROLL_PAGES = 64
_MAX_EXPORTED_POINTS = 256
_GUARD_KEY = FrozenMemoryObjectKey(
    object_type="profile", namespace="canary_control", entity="sentinel", attribute="guard"
)
_SUBJECT_DIRTY_KEY = FrozenMemoryObjectKey(
    object_type="profile", namespace="canary_control", entity="sentinel", attribute="dirty"
)


def _dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_unset=True)
    if isinstance(value, Mapping):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


def _request(value):
    return NativeMultiObjectMutationRequestV3.model_validate(value)


def _retrieval(value):
    return NativeRetrievalRequestV3.model_validate(value)


def _error(exc: BaseException, stage: str) -> dict[str, str]:
    return {"type": type(exc).__name__, "stage": stage}


def _point_field(point, name, default=None):
    if isinstance(point, Mapping):
        return point.get(name, default)
    return getattr(point, name, default)


def _scroll_points(provider, response):
    parser = getattr(provider, "_scroll_points", None)
    if callable(parser):
        return parser(response)
    if isinstance(response, tuple) and len(response) == 2:
        return list(response[0]), response[1]
    if isinstance(response, Mapping):
        return list(response.get("points", ())), response.get("next_page_offset")
    points = getattr(response, "points", response)
    return list(points), getattr(response, "next_page_offset", None)


def _export(provider, namespace: str) -> list[dict]:
    """Read the provider collection directly; this is never a local state export."""
    rows: list[dict] = []
    cursor = None
    seen_cursors = set()
    for _ in range(_MAX_SCROLL_PAGES):
        kwargs = {
            "collection_name": provider.collection_name,
            "scroll_filter": provider._filter_for_scope(namespace, "namespace"),
            "limit": 64,
            "with_payload": True,
            "with_vectors": False,
        }
        if cursor is not None:
            kwargs["offset"] = cursor
        response = provider._provider_call("scroll", **kwargs)
        points, next_cursor = _scroll_points(provider, response)
        for point in points:
            rows.append({"id": str(_point_field(point, "id")), "payload": _dump(_point_field(point, "payload", {}))})
            if len(rows) > _MAX_EXPORTED_POINTS:
                raise ValueError("export bound exceeded")
        if next_cursor is None:
            return rows
        marker = repr(next_cursor)
        if marker in seen_cursors:
            raise ValueError("export pagination cursor repeated")
        seen_cursors.add(marker)
        cursor = next_cursor
    raise ValueError("export pagination bound exceeded")


def _seed_request(case_id: str, namespace: str, key: FrozenMemoryObjectKey, value: str):
    return NativeMultiObjectMutationRequestV3(
        request_id=f"{case_id}-setup-{key.attribute}",
        event_id=f"{case_id}-setup-{key.attribute}",
        sequence_index=0,
        operation=Operation.ADD,
        objects=( {"object_key": key, "value": value}, ),
        source_event_ids=(f"{case_id}-setup-{key.attribute}",),
        provenance={"kind": "declared", "provider_owned": False},
    )


def _witnesses(provider):
    value = getattr(provider, "observation_witnesses", ())
    if callable(value):
        value = value()
    return [_dump(item) for item in value]


def _identity(provider):
    config = getattr(provider, "configuration", None)
    config = _dump(config) if config is not None else {}
    return {
        "provider_type": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "provider_version": str(getattr(provider, "PROVIDER_VERSION", getattr(provider, "provider_version", "unknown"))),
        "surface_version": str(getattr(provider, "SURFACE_VERSION", "native_multi_object_v3")),
        "collection_name": str(getattr(provider, "collection_name", config.get("collection_name", "unknown"))),
        "configuration": {k: v for k, v in config.items() if k not in {"storage_path", "path", "qdrant_path"}},
    }


def _raw_query_points(provider, trace):
    observed = getattr(provider, "last_query_points_observation", None)
    if observed is None:
        raise ValueError("provider did not expose independent raw query response")
    points = [_dump(point) for point in observed]
    if len(points) != len(trace.entries):
        raise ValueError("raw query response count disagrees with normalized trace")
    return points


def _case_groups(requests):
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, bytearray)):
        raise ValueError("requests must be an ordered sequence")
    groups = []
    for raw in requests:
        if not isinstance(raw, Mapping):
            raise ValueError("request case must be a mapping")
        if not groups or (groups[-1]["case_id"], groups[-1]["repetition"]) != (raw.get("case_id"), raw.get("repetition")):
            groups.append({"case_id": raw.get("case_id"), "repetition": raw.get("repetition"), "rows": []})
        groups[-1]["rows"].append(raw)
    if any(len(group["rows"]) != 1 for group in groups):
        raise ValueError("each ordered case must have exactly one public row")
    return [group["rows"][0] for group in groups]


def run_cases(requests, factory: Callable):
    """Run the eight declared cases without reading any expectation or gold data."""
    cases = _case_groups(requests)
    if len(cases) != _CASE_COUNT:
        raise ValueError("native canary requires exactly eight ordered cases")
    parsed_cases = []
    planned = 0
    for case in cases:
        if type(case.get("runtime_namespace")) is not str or not case["runtime_namespace"].strip():
            raise ValueError("case runtime namespace is invalid")
        mutations = tuple(_request(item) for item in case.get("mutations", ()))
        retrieval = _retrieval(case["retrieval"])
        if retrieval.namespace != case["runtime_namespace"]:
            raise ValueError("retrieval namespace disagrees with case namespace")
        parsed_cases.append((case, mutations, retrieval))
        planned += len(mutations)
    rows = []
    blocked = False
    counts = {
        "planned_mutation_operations": planned,
        "mutation_attempts": 0,
        "mutation_completions": 0,
        "retrieval_attempts": 0,
        "retrieval_completions": 0,
        "setup_seed_attempts": 0,
        "reset_attempts": 0,
    }
    for index, (case, mutations, retrieval) in enumerate(parsed_cases):
        if blocked:
            rows.append({"case_id": case["case_id"], "repetition": case["repetition"], "status": "NOT_RUN"})
            continue
        provider = cleanup = None
        row = {
            "case_id": case["case_id"], "repetition": case["repetition"],
            "runtime_namespace": case["runtime_namespace"], "status": "BLOCKED",
            "mutation_events": [], "provider_query_points": [],
        }
        try:
            provider, cleanup = factory({k: copy.deepcopy(v) for k, v in case.items()})
            row["identity"] = _identity(provider)
            row["collection_name"] = str(getattr(provider, "collection_name", "unknown"))
            guard_namespace = case["runtime_namespace"] + "__guard"
            row["isolated_reset_observations"] = {}
            counts["reset_attempts"] += 1
            provider.reset(guard_namespace)
            counts["setup_seed_attempts"] += 1
            provider.mutate(_seed_request(case["case_id"], guard_namespace, _GUARD_KEY, "guard-fixed"))
            row["isolated_reset_observations"]["guard_export"] = _export(provider, guard_namespace)
            counts["reset_attempts"] += 1
            provider.reset(case["runtime_namespace"])
            counts["setup_seed_attempts"] += 1
            provider.mutate(_seed_request(case["case_id"], case["runtime_namespace"], _SUBJECT_DIRTY_KEY, "dirty-before-reset"))
            counts["reset_attempts"] += 1
            provider.reset(case["runtime_namespace"])
            row["isolated_reset_observations"]["subject_empty_after_reset"] = _export(provider, case["runtime_namespace"])
            row["isolated_reset_observations"]["guard_survived_subject_reset"] = _export(provider, guard_namespace)
            counts["reset_attempts"] += 1
            provider.reset(case["runtime_namespace"])
            for mutation in mutations:
                counts["mutation_attempts"] += 1
                result = provider.mutate(mutation)
                counts["mutation_completions"] += 1
                row["mutation_events"].append({
                    "request": _dump(mutation), "result": _dump(result),
                    "after_snapshot": _export(provider, case["runtime_namespace"]),
                })
            counts["retrieval_attempts"] += 1
            trace = provider.retrieve(retrieval)
            counts["retrieval_completions"] += 1
            row["retrieval_trace"] = _dump(trace)
            row["provider_query_points"] = _raw_query_points(provider, trace)
            row["final_export"] = _export(provider, case["runtime_namespace"])
            row["final_guard_export"] = _export(provider, guard_namespace)
            counts["reset_attempts"] += 1
            provider.reset(case["runtime_namespace"])
            row["isolated_reset_observations"]["final_subject_empty"] = _export(provider, case["runtime_namespace"])
            counts["reset_attempts"] += 1
            provider.reset(guard_namespace)
            row["isolated_reset_observations"]["final_guard_empty"] = _export(provider, guard_namespace)
            row["status"] = "OBSERVED"
        except BaseException as exc:
            row["error"] = _error(exc, "execution")
            blocked = True
        finally:
            if cleanup is not None:
                try:
                    row["cleanup"] = _dump(cleanup())
                except BaseException as exc:
                    row["cleanup"] = {"client_closed": False, "collection_deleted": False}
                    row["cleanup_error"] = _error(exc, "cleanup")
                    blocked = True
            if provider is not None:
                row["observation_witnesses"] = _witnesses(provider)
            rows.append(row)
            if row["status"] != "OBSERVED":
                blocked = True
    return {
        "schema": "memupdatebench.native-api-canary-observations.v1",
        "status": "BLOCKED" if blocked else "OBSERVATIONS_COMPLETE",
        "scope": "synthetic_api_canary_only",
        "counts": counts,
        "cases": rows,
        "native_admission": "NOT_GRANTED",
        "answer_metrics": None,
        "scientific_release_allowed": False,
    }


def _semantic_key(raw_key):
    if not isinstance(raw_key, Mapping):
        raise ValueError("snapshot object key is malformed")
    fields = ("namespace", "entity", "attribute", "subkey")
    if any(type(raw_key.get(field)) is not str and not (field == "subkey" and raw_key.get(field) is None) for field in fields):
        raise ValueError("snapshot canonical identity is malformed")
    return tuple(raw_key[field] for field in fields)


def _payload_projection(point):
    payload = point.get("payload") if isinstance(point, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot payload is malformed")
    key = _semantic_key(payload)
    if "value" not in payload or "source_event_ids" not in payload or "sequence_index" not in payload:
        raise ValueError("snapshot payload linkage is incomplete")
    return {
        "identity": key,
        "value": payload["value"],
        "source_event_ids": list(payload["source_event_ids"]),
        "sequence_index": payload["sequence_index"],
    }


def _snapshot_projection(snapshot):
    if not isinstance(snapshot, Sequence) or isinstance(snapshot, (str, bytes, bytearray)):
        raise ValueError("snapshot is malformed")
    values = [_payload_projection(point) for point in snapshot]
    identities = [item["identity"] for item in values]
    if len(identities) != len(set(identities)):
        raise ValueError("snapshot contains duplicate canonical identity")
    return sorted(values, key=lambda item: repr(item["identity"]))


def _typed_equal(left, right):
    return typed_json_equal(left, right)


def _expected_projection(state):
    result = []
    for item in state:
        result.append({
            "identity": _semantic_key(item["object_key"]),
            "value": item["value"],
            "source_event_ids": list(item["source_event_ids"]),
            "sequence_index": item["sequence_index"],
        })
    return sorted(result, key=lambda item: repr(item["identity"]))


def _same_projection(actual, expected):
    if len(actual) != len(expected):
        return False
    return all(
        left["identity"] == right["identity"]
        and _typed_equal(left["value"], right["value"])
        and left["source_event_ids"] == right["source_event_ids"]
        and type(left["sequence_index"]) is type(right["sequence_index"])
        and left["sequence_index"] == right["sequence_index"]
        for left, right in zip(actual, expected)
    )


def _result_projection(result):
    if not isinstance(result, Mapping):
        raise ValueError("mutation result is malformed")
    outcomes = []
    for outcome in result.get("outcomes", []):
        key = outcome.get("object_key", outcome.get("key"))
        outcomes.append({
            "event_id": outcome.get("event_id"), "object_id": _semantic_key(key),
            "operation": outcome.get("operation"), "status": outcome.get("status"),
            "reason": outcome.get("reason"),
        })
    return {
        "request_id": result.get("request_id"), "event_id": result.get("event_id"),
        "operation": result.get("operation"), "atomicity": result.get("atomicity"),
        "outcomes": outcomes,
    }


def _expected_result(request, step):
    outcomes = []
    request_objects = request.get("objects", [])
    if len(request_objects) != len(step["outcomes"]):
        raise ValueError("expectation outcome count disagrees with request")
    for object_item, item in zip(request_objects, step["outcomes"]):
        identity = _semantic_key(object_item["object_key"])
        expected_object_id = item.get("object_id")
        if expected_object_id != "|".join(part or "" for part in identity):
            raise ValueError("expectation object identity mismatch")
        outcomes.append({
            "event_id": request["event_id"], "object_id": identity,
            "operation": request["operation"], "status": item["status"], "reason": item.get("reason"),
        })
    return {
        "request_id": request["request_id"], "event_id": request["event_id"],
        "operation": request["operation"], "atomicity": step["atomicity"], "outcomes": outcomes,
    }


def _finite_scores(points):
    for point in points:
        score = point.get("score")
        if type(score) not in (int, float) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("retrieval score is not finite")


def _retrieval_projection(trace, raw_points, final_export):
    if not isinstance(trace, Mapping) or not isinstance(trace.get("entries"), list):
        raise ValueError("retrieval trace is malformed")
    if not isinstance(raw_points, list):
        raise ValueError("provider query points are required")
    if len(trace["entries"]) != len(raw_points):
        raise ValueError("retrieval trace does not match raw query response")
    _finite_scores(raw_points)
    export_by_id = {str(point["id"]): point for point in final_export}
    projected = []
    for index, (entry, point) in enumerate(zip(trace["entries"], raw_points)):
        if str(entry.get("provider_entry_id")) != str(point.get("id")):
            raise ValueError("retrieval provider ID mismatch")
        if not _typed_equal(entry.get("score"), point.get("score")):
            raise ValueError("retrieval score mismatch")
        if entry.get("order_index") != index or entry.get("rank") != index + 1:
            raise ValueError("retrieval order or rank mismatch")
        if str(point.get("id")) not in export_by_id:
            raise ValueError("retrieval entry is absent from final export")
        if not _typed_equal(entry.get("provider_payload"), point.get("payload")):
            raise ValueError("retrieval payload mismatch")
        projected.append({
            "identity": _semantic_key(entry["object_key"]), "value": entry["provider_payload"].get("value"),
            "source_event_ids": list(entry["source_event_ids"]), "rank": entry["rank"],
            "order_index": entry["order_index"], "score": entry["score"],
        })
    return projected


def audit_cases(requests, expectations, observations):
    """Validate a complete observation package against offline preparation expectations."""
    cases = _case_groups(requests)
    if len(cases) != _CASE_COUNT or not isinstance(expectations, Sequence) or len(expectations) != _CASE_COUNT:
        raise ValueError("audit requires exactly eight ordered cases and expectations")
    if not isinstance(observations, Mapping) or observations.get("status") != "OBSERVATIONS_COMPLETE":
        raise ValueError("audit requires complete observations")
    observed_rows = observations.get("cases")
    if not isinstance(observed_rows, list) or len(observed_rows) != _CASE_COUNT:
        raise ValueError("audit observation case count mismatch")
    semantic_rows = []
    for case, expectation, row in zip(cases, expectations, observed_rows):
        if row.get("status") != "OBSERVED":
            raise ValueError("blocked or incomplete observation row")
        cleanup = row.get("cleanup")
        if not isinstance(cleanup, Mapping) or cleanup.get("client_closed") is not True or cleanup.get("collection_deleted") is not True:
            raise ValueError("cleanup did not prove client close and collection deletion")
        witness_methods = {item.get("method") for item in row.get("observation_witnesses", ())}
        if "query_points" not in witness_methods or "scroll" not in witness_methods:
            raise ValueError("provider trace is missing direct retrieval/export witness")
        if (case.get("case_id"), case.get("repetition")) != (expectation.get("case_id"), expectation.get("repetition")):
            raise ValueError("case ordering mismatch")
        mutations = [_request(item) for item in case.get("mutations", [])]
        if len(row.get("mutation_events", [])) != len(mutations):
            raise ValueError("mutation event count mismatch")
        expected_steps = expectation.get("steps")
        if not isinstance(expected_steps, list) or len(expected_steps) != len(mutations):
            raise ValueError("expectation step count mismatch")
        previous_ids = {}
        for mutation, expected_step, event in zip(mutations, expected_steps, row["mutation_events"]):
            actual_result = _result_projection(event.get("result"))
            expected_result = _expected_result(_dump(mutation), expected_step)
            if actual_result != expected_result:
                raise ValueError("mutation result event/id/operation/outcome mismatch")
            actual_snapshot = _snapshot_projection(event.get("after_snapshot"))
            expected_snapshot = _expected_projection(expected_step.get("after_state", []))
            if not _same_projection(actual_snapshot, expected_snapshot):
                raise ValueError("snapshot mismatch")
            for point in event["after_snapshot"]:
                identity = _semantic_key(point["payload"])
                point_id = str(point["id"])
                if identity in previous_ids and previous_ids[identity] != point_id:
                    raise ValueError("provider ID changed within repetition")
                previous_ids[identity] = point_id
        final_projection = _snapshot_projection(row.get("final_export"))
        if not _same_projection(final_projection, _expected_projection(expectation.get("final_state", []))):
            raise ValueError("final snapshot mismatch")
        guard_projection = _snapshot_projection(row.get("final_guard_export"))
        if not guard_projection:
            raise ValueError("guard was not preserved in isolated export")
        if any(item["identity"] in {(_GUARD_KEY.namespace, _GUARD_KEY.entity, _GUARD_KEY.attribute, _GUARD_KEY.subkey)} for item in final_projection):
            raise ValueError("guard leaked into subject final export")
        reset = row.get("isolated_reset_observations", {})
        if reset.get("subject_empty_after_reset") != [] or reset.get("final_subject_empty") != [] or reset.get("final_guard_empty") != []:
            raise ValueError("isolated reset did not produce empty export")
        if not reset.get("guard_survived_subject_reset"):
            raise ValueError("guard did not survive subject reset")
        projected_retrieval = _retrieval_projection(row.get("retrieval_trace"), row.get("provider_query_points"), row["final_export"])
        ids = {str(point["id"]): _semantic_key(point["payload"]) for point in row["final_export"]}
        for point in row["provider_query_points"]:
            if str(point["id"]) not in ids:
                raise ValueError("query response leaked an unknown point")
        semantic_rows.append({"snapshots": [_snapshot_projection(e["after_snapshot"]) for e in row["mutation_events"]], "retrieval": projected_retrieval})
    for first, second in zip(semantic_rows[::2], semantic_rows[1::2]):
        if first != second:
            raise ValueError("repeated semantic projection mismatch")
    return {
        "scope": "synthetic_api_canary_only",
        "native_admission": "NOT_GRANTED",
        "answer_metrics": None,
        "scientific_release_allowed": False,
        "status": "AUDITED",
        "case_count": _CASE_COUNT,
    }


__all__ = ["run_cases", "audit_cases"]
