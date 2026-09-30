"""Source-bound state/retrieval runner for formal cross-domain releases.

This module projects only the public event and query surfaces from the immutable
NOAA/BEA task releases. It never passes task actions, target selectors, version
history, or gold evidence to a manager. The manager factory is injected so the
projection and scoring contract can be tested without starting Qdrant.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, typed_json_equal
from mub.vnext.validation.replay_v3 import replay_task_v3
from scripts.vnext_promote_family_h_cross_domain_bea import read_release as read_bea_release
from scripts.vnext_promote_family_h_cross_domain_noaa import read_release as read_noaa_release

BEA_INDEX_SHA256 = "5525357436510b0c4e89efa53cdb6caf551c59234171c71bb97563e4c641790a"
NOAA_INDEX_SHA256 = "738f46932d8e2e8c26da0a2820dc67c7b0bad9aa43659f4be266f22f0fc8ee3f"
_RELEASE_READERS = {"bea": read_bea_release, "noaa": read_noaa_release}
_KEY_FIELDS = ("namespace", "entity", "attribute", "subkey")
_RELEASE_HASHES = {"bea": BEA_INDEX_SHA256, "noaa": NOAA_INDEX_SHA256}
_PUBLIC_KEYS = {
    "bea": {"namespace": "family_h", "entity": "us-gdp-2024q4",
            "attribute": "real_gdp_growth_annual_rate", "subkey": None,
            "object_type": "macroeconomic_estimate"},
    "noaa": {"namespace": "family_h", "entity": "hurricane-beryl-2024",
             "attribute": "max_sustained_wind_mph", "subkey": None,
             "object_type": "weather_observation"},
}
_QUERY_PREFIXES = {
    "bea": "Within the supplied BEA release vintages for 2024-Q4, return the current value for object ",
    "noaa": "Within the supplied public advisories, return the current wind value for object ",
}
_CELL_FIELDS = {"release_name", "release_index_sha256", "task_id", "task_sha256", "events", "query"}
_EVENT_FIELDS = {"event_id", "sequence_index", "operation", "object_key", "value"}
_ENTRY_FIELDS = {"entry_id", "object_key", "value", "content", "source_event_ids", "score", "rank", "version_metadata"}
_NOAA_RECORD_HASHES = {
    "beryl-public-001": "98f66a70da5cf3b511a5c5a8ed1369afb631d395d1af1609fe985e896d4f9afa",
    "beryl-public-002": "0e53333535c6910114c7b4ed55183d1041334ee687725d86f1e0d02eddd17cbf",
    "beryl-public-003": "048621bb7af2a71181e79d5c1b472385902ab5cb2fda132831e70d4ec2634623",
    "beryl-public-003a": "17249c248b51c464b41174938a886f0aa41d5813db927298c567f1d6fdf8c12b",
    "beryl-public-004": "36110b72a6816de065946cdf9949b8326e3490262e88d29f59464ac2baaa32b9",
    "beryl-public-004a": "fb15df12408ac000f72566eddb5a03ccde740b316bf314671ac32438fa56f9fe",
    "beryl-public-005": "670ed30d0ab5bbe98efa01ecc2b4804976501cf44b3a40cdba817f6dd008b59a",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _nonblank(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _exact_key(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {*_KEY_FIELDS, "object_type"},
             "public object key fields mismatch")
    return FrozenMemoryObjectKey.model_validate(dict(value), strict=True).model_dump(mode="json")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate public JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite public JSON constant: {value}")


def _parse_public_event(event: Any) -> dict[str, Any]:
    raw_text = getattr(event, "raw_text", None)
    _require(_nonblank(raw_text), "public event raw_text is required")
    try:
        payload = json.loads(raw_text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cross-domain events must use structured public JSON") from exc
    _require(type(payload) is dict, "public event must be a JSON object")
    common = {"action", "object_key", "value"}
    if set(payload) == common | {"estimate_stage", "release_date"}:
        release_name = "bea"
        dates = {"advance": "2025-01-30", "second": "2025-02-27", "third": "2025-03-27"}
        stage = payload["estimate_stage"]
        _require(type(stage) is str and stage in dates and payload["release_date"] == dates[stage],
                 "public BEA descriptor mismatch")
        _require(payload["action"] == ("ADD" if stage == "advance" else "UPDATE"),
                 "public BEA operation mismatch")
    elif set(payload) == common | {"normalized_record_sha256", "source_heading_anchor"}:
        release_name = "noaa"
        anchor = payload["source_heading_anchor"]
        _require(type(anchor) is str and anchor in _NOAA_RECORD_HASHES, "public NOAA heading mismatch")
        digest = payload["normalized_record_sha256"]
        _require(type(digest) is str and digest == _NOAA_RECORD_HASHES[anchor],
                 "public NOAA record hash mismatch")
        _require(payload["action"] == ("ADD" if anchor == "beryl-public-001" else "UPDATE"),
                 "public NOAA operation mismatch")
    else:
        raise ValueError("public event descriptor fields mismatch")
    key = _exact_key(payload["object_key"])
    _require(key == _PUBLIC_KEYS[release_name], "public event source object key mismatch")
    _require(_finite_number(payload["value"]), "public event requires finite numeric value")
    return {"operation": payload["action"].lower(), "object_key": key, "value": payload["value"]}


def _task_public_cell(release_name: str, release_index_sha256: str, task: Any) -> dict[str, Any]:
    _require(len(task.events) > 0 and len(task.queries) == 1, "cross-domain task cardinality mismatch")
    events: list[dict[str, Any]] = []
    for event in task.events:
        parsed = _parse_public_event(event)
        events.append({
            "event_id": event.event_id,
            "sequence_index": event.sequence_index,
            **parsed,
        })
    _require([event["sequence_index"] for event in events] == list(range(len(events))),
             "cross-domain event sequence is not contiguous")
    mutation_keys = [event["object_key"] for event in events if event["object_key"] is not None]
    _require(mutation_keys and len({_key_tuple(key) for key in mutation_keys}) == 1,
             "cross-domain task must use one canonical object slot")
    query = task.queries[0]
    query_public = {"query_id": query.query_id, "text": query.text}
    _require(all(type(value) is str and value.strip() for value in query_public.values()),
             "public query must contain query_id and text only")
    task_sha256 = sha256_bytes(canonical_json_bytes(task.model_dump(mode="json")))
    return {
        "release_name": release_name,
        "release_index_sha256": release_index_sha256,
        "task_id": task.task_id,
        "task_sha256": task_sha256,
        "events": events,
        "query": query_public,
    }


def _key_tuple(object_key: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(object_key[field] for field in _KEY_FIELDS)


def _validate_public_cell(cell: Mapping[str, Any]) -> None:
    _require(isinstance(cell, Mapping) and set(cell) == _CELL_FIELDS,
             "cross-domain cell contains non-public fields")
    name = cell["release_name"]
    _require(type(name) is str and name in _RELEASE_HASHES, "unsupported cross-domain release")
    _require(cell["release_index_sha256"] == _RELEASE_HASHES[name], "release index hash mismatch")
    _require(_nonblank(cell["task_id"]) and type(cell["task_sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", cell["task_sha256"]) is not None,
             "public task identity mismatch")
    events = cell["events"]
    _require(type(events) is list and len(events) == {"bea": 3, "noaa": 7}[name],
             "cross-domain public event cardinality mismatch")
    ids: set[str] = set()
    for index, event in enumerate(events):
        _require(isinstance(event, Mapping) and set(event) == _EVENT_FIELDS,
                 "cross-domain event contains non-public fields")
        _require(_nonblank(event["event_id"]) and event["event_id"] not in ids,
                 "public event identity is invalid or duplicated")
        ids.add(event["event_id"])
        _require(type(event["sequence_index"]) is int and event["sequence_index"] == index,
                 "cross-domain event sequence is not contiguous")
        _require(event["operation"] == ("add" if index == 0 else "update"),
                 "public event operation mismatch")
        _require(_exact_key(event["object_key"]) == _PUBLIC_KEYS[name], "public event source key mismatch")
        _require(_finite_number(event["value"]), "public event requires finite numeric value")
    query = cell["query"]
    _require(isinstance(query, Mapping) and set(query) == {"query_id", "text"}
             and _nonblank(query["query_id"]), "public query fields mismatch")
    key = FrozenMemoryObjectKey.model_validate(_PUBLIC_KEYS[name], strict=True)
    _require(query["text"] == _QUERY_PREFIXES[name] + key.canonical_id + ".",
             "public query key/text mismatch")


def _admit_public_projection(task: Any, cell: Mapping[str, Any]) -> None:
    # Hidden oracle data is consulted only after the visible projection is fixed.
    replay = replay_task_v3(task)
    _require(replay.valid, "cross-domain release replay failed")
    events = cell["events"]
    _require(len(task.actions) == len(events), "public action cardinality mismatch")
    for event, action in zip(events, task.actions, strict=True):
        _require(action.event_id == event["event_id"]
                 and action.operation.value.lower() == event["operation"]
                 and len(action.target_object_keys) == 1
                 and action.target_object_keys[0].model_dump(mode="json") == event["object_key"]
                 and typed_json_equal(action.value, event["value"]), "public action/replay mismatch")
    last = events[-1]
    key = FrozenMemoryObjectKey.model_validate(last["object_key"], strict=True)
    current = replay.current_state.get(key.canonical_id)
    _require(current is not None and typed_json_equal(current.value, last["value"])
             and list(current.source_event_ids) == [last["event_id"]], "public replay final state mismatch")
    _require(len(task.gold_evidence) == 1, "public gold cardinality mismatch")
    gold = task.gold_evidence[0]
    query = task.queries[0]
    _require(gold.query_id == cell["query"]["query_id"]
             and typed_json_equal(gold.answer, last["value"])
             and list(gold.supporting_event_ids) == [last["event_id"]]
             and len(gold.supporting_object_keys) == 1
             and gold.supporting_object_keys[0].model_dump(mode="json") == last["object_key"]
             and len(query.target_object_keys) == 1
             and query.target_object_keys[0].model_dump(mode="json") == last["object_key"],
             "public gold/query mismatch")


def load_public_cells(releases: Mapping[str, tuple[Path | str, str]]) -> list[dict[str, Any]]:
    """Authenticate the pinned formal releases, project, then check oracle admission."""
    _require(isinstance(releases, Mapping) and 1 <= len(releases) <= 2,
             "cross-domain releases must contain one or two cells")
    cells: list[dict[str, Any]] = []
    for release_name in sorted(releases):
        _require(release_name in _RELEASE_READERS, f"unsupported cross-domain release: {release_name}")
        root, expected_index_sha256 = releases[release_name]
        _require(expected_index_sha256 == _RELEASE_HASHES[release_name], "release index hash mismatch")
        result = _RELEASE_READERS[release_name](Path(root).absolute(), expected_index_sha256)
        manifest = result["manifest"]
        _require(manifest.get("status") == "FINAL_APPROVED"
                 and manifest.get("formal_task_release") is True
                 and manifest.get("scientific_release_allowed") is False
                 and manifest.get("benchmark_accuracy_claimed") is False,
                 "cross-domain release boundary mismatch")
        _require(len(result["tasks"]) == 1, "cross-domain release must contain one task")
        task = result["tasks"][0]
        cell = _task_public_cell(release_name, expected_index_sha256, task)
        _validate_public_cell(cell)
        _admit_public_projection(task, cell)
        cells.append(cell)
    return cells


def _observe_state(entries: Sequence[Mapping[str, Any]], expected_key: Mapping[str, Any],
                   expected_value: Any, event_id: str) -> dict[str, Any]:
    expected = _key_tuple(expected_key)
    same_slot = [entry for entry in entries if _key_tuple(entry["object_key"]) == expected]
    current = [entry for entry in same_slot if event_id in entry.get("source_event_ids", [])]
    state_match = (len(entries) == 1 and len(same_slot) == 1 and len(current) == 1
                   and typed_json_equal(current[0]["value"], expected_value))
    return {
        "state_match": state_match,
        "current_source_event_link": len(current) == 1,
        "stale_same_slot_count": sum(event_id not in entry.get("source_event_ids", []) for entry in same_slot),
        "memory_size": len(entries),
        "entries": list(entries),
        "entries_sha256": sha256_bytes(canonical_json_bytes(list(entries))),
    }


def _retrieval_match(trace: Mapping[str, Any], expected_key: Mapping[str, Any], expected_value: Any,
                     expected_event_id: str) -> bool:
    entries = trace.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ValueError("manager retrieval trace entries are malformed")
    expected = _key_tuple(expected_key)
    return any(
        _key_tuple(entry["object_key"]) == expected
        and typed_json_equal(entry["value"], expected_value)
        and expected_event_id in entry.get("source_event_ids", [])
        for entry in entries
    )


def _json_snapshot(value: Any) -> Any:
    def validate(item: Any) -> None:
        if item is None or type(item) in (str, bool) or _finite_number(item):
            return
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "normalized JSON keys must be strings")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        else:
            raise ValueError("normalized traces must contain finite strict JSON")
    validate(value)
    return json.loads(canonical_json_bytes(value))


def _normalized_entries(entries: Any) -> list[dict[str, Any]]:
    _require(isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)),
             "manager normalized entries are malformed")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, Mapping) and set(entry) == _ENTRY_FIELDS,
                 "manager normalized entry fields mismatch")
        item = dict(entry)
        _require(_nonblank(item["entry_id"]) and item["entry_id"] not in ids,
                 "manager entry ID is invalid or duplicated")
        ids.add(item["entry_id"])
        item["object_key"] = _exact_key(item["object_key"])
        _require(_finite_number(item["value"]) and _finite_number(item["score"]),
                 "manager normalized entry value/score must be finite numeric")
        _require(type(item["content"]) is str and type(item["rank"]) is int and item["rank"] >= 1,
                 "manager normalized entry content/rank is malformed")
        sources = item["source_event_ids"]
        _require(isinstance(sources, (list, tuple)) and bool(sources)
                 and all(_nonblank(source) for source in sources) and len(set(sources)) == len(sources),
                 "manager normalized source linkage is malformed")
        _require(isinstance(item["version_metadata"], Mapping), "manager version metadata is malformed")
        item["source_event_ids"] = list(sources)
        item["version_metadata"] = dict(item["version_metadata"])
        result.append(_json_snapshot(item))
    return result


def _normalized_retrieval(trace: Any) -> dict[str, Any]:
    _require(isinstance(trace, Mapping) and {"entries", "context_order"} <= set(trace)
             and set(trace) <= {"entries", "context_order", "version_metadata"},
             "manager normalized retrieval fields mismatch")
    _require(_nonblank(trace["context_order"]), "manager retrieval context order is malformed")
    metadata = trace.get("version_metadata", {})
    _require(isinstance(metadata, Mapping), "manager retrieval metadata is malformed")
    return {"entries": _normalized_entries(trace["entries"]), "context_order": trace["context_order"],
            "version_metadata": _json_snapshot(dict(metadata))}


def _stable_row_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compare all normalized semantics, preserving ordering and within-run ID linkage.

    Only opaque entry IDs are alpha-renamed in first-seen order: Qdrant UUID5 IDs
    include the run namespace. Canonical object namespaces and source event IDs
    are semantic and are never removed. Raw-ID hashes are excluded accordingly.
    """
    ids: dict[str, str] = {}

    def stable_id(value: str) -> str:
        return ids.setdefault(value, f"entry-{len(ids)}")

    def entries(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{**entry, "entry_id": stable_id(entry["entry_id"])} for entry in values]

    steps = []
    for step in row["steps"]:
        normalized = {key: value for key, value in step.items() if key != "entries_sha256"}
        normalized["entries"] = entries(step["entries"])
        normalized["affected_entry_ids"] = [stable_id(item) for item in step["affected_entry_ids"]]
        steps.append(normalized)
    retrieval = {key: value for key, value in row["retrieval"].items() if key != "entries_sha256"}
    retrieval["entries"] = entries(row["retrieval"]["entries"])
    return {"status": row["status"], "steps": steps, "retrieval": retrieval,
            "final_state_match": row.get("final_state_match"),
            "retrieval_typed_object_match": row.get("retrieval_typed_object_match"),
            "cleanup": row["cleanup"]}


def run_cross_domain_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    manager_factory: Callable[[Mapping[str, Any], int], tuple[Any, Callable[[], None]]],
    repetitions: int = 2,
    mutation_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Bounded visible-input replay with fail-stop lifecycles and normalized evidence.

    Callback resources are unknown; only the production supervisor may establish
    actual model/resource totals. Observer failures are technical failures too.
    """
    _require(type(repetitions) is int and repetitions == 2, "repetitions must equal two")
    _require(isinstance(cells, Sequence) and not isinstance(cells, (str, bytes))
             and 1 <= len(cells) <= 2, "cross-domain cells must contain one or two cells")
    for cell in cells:
        _validate_public_cell(cell)
    _require(len({cell["release_name"] for cell in cells}) == len(cells)
             and len({cell["task_id"] for cell in cells}) == len(cells), "cross-domain cells must be unique")
    _require(callable(manager_factory) and (mutation_observer is None or callable(mutation_observer)),
             "manager factory/observer must be callable")
    cells = deepcopy(list(cells))
    rows: list[dict[str, Any]] = []
    stopped = False
    counts = {"manager_attempted": 0, "manager_created": 0, "ingest_attempted": 0,
              "ingest_completed": 0, "state_exports_completed": 0, "retrieve_attempted": 0,
              "retrieve_completed": 0}
    for cell in cells:
        for repetition_index in range(repetitions):
            manager = None
            delete_owned_collection = None
            row: dict[str, Any] = {
                "release_name": cell["release_name"], "release_index_sha256": cell["release_index_sha256"],
                "task_id": cell["task_id"], "task_sha256": cell["task_sha256"],
                "repetition_index": repetition_index, "status": "BLOCKED", "steps": [],
                "cleanup": {}, "retrieval": None, "provider_witness_count": None,
                "final_state_match": None, "retrieval_typed_object_match": None,
            }
            if stopped:
                row["blocker"] = "previous_technical_failure"
                rows.append(row)
                continue
            try:
                counts["manager_attempted"] += 1
                manager, delete_owned_collection = manager_factory(deepcopy(cell), repetition_index)
                _require(manager is not None and callable(delete_owned_collection),
                         "manager factory returned invalid handle")
                counts["manager_created"] += 1
                for event in cell["events"]:
                    event_object = SimpleNamespace(event_id=event["event_id"], sequence_index=event["sequence_index"])
                    counts["ingest_attempted"] += 1
                    result = manager.ingest(event_object, operation=event["operation"],
                                            object_key=deepcopy(event["object_key"]), value=event["value"])
                    counts["ingest_completed"] += 1
                    _require(isinstance(result, Mapping) and set(result) == {"effective_operation", "affected_entry_ids"}
                             and result["effective_operation"] in {"add", "update", "noop"},
                             "manager normalized mutation result is malformed")
                    affected = result["affected_entry_ids"]
                    _require(isinstance(affected, (list, tuple)) and all(_nonblank(item) for item in affected),
                             "manager affected entry IDs are malformed")
                    entries = _normalized_entries(list(manager.export_entries()))
                    counts["state_exports_completed"] += 1
                    observation = _observe_state(entries, event["object_key"], event["value"], event["event_id"])
                    step = {"event_id": event["event_id"], "sequence_index": event["sequence_index"],
                            "operation": event["operation"], **observation,
                            "effective_operation": result["effective_operation"], "affected_entry_ids": list(affected)}
                    row["steps"].append(step)
                    if mutation_observer is not None:
                        mutation_observer(deepcopy({"release_name": cell["release_name"], "task_id": cell["task_id"],
                                                   "repetition_index": repetition_index, "step": step}))
                last = cell["events"][-1]
                row["final_state_match"] = row["steps"][-1]["state_match"]
                counts["retrieve_attempted"] += 1
                raw_trace = manager.retrieve(SimpleNamespace(**cell["query"]))
                counts["retrieve_completed"] += 1
                trace = _normalized_retrieval(raw_trace)
                row["retrieval_typed_object_match"] = _retrieval_match(
                    trace, last["object_key"], last["value"], last["event_id"])
                row["retrieval"] = {**trace, "typed_object_match": row["retrieval_typed_object_match"],
                                    "entries_sha256": sha256_bytes(canonical_json_bytes(trace["entries"]))}
                row["status"] = "COMPLETE"
            except BaseException as exc:
                row["status"] = "BLOCKED"
                row["error_type"] = type(exc).__name__
                if getattr(exc, "cleanup_diagnostics", None):
                    row["factory_cleanup_diagnostics"] = deepcopy(exc.cleanup_diagnostics)
            finally:
                if manager is not None:
                    try:
                        manager.reset(None)
                        row["cleanup"]["reset_empty"] = list(manager.export_entries()) == []
                    except BaseException as exc:
                        row["cleanup"]["reset_error_type"] = type(exc).__name__
                    try:
                        row["provider_witness_count"] = len(getattr(manager, "provider_witnesses", ()))
                    except BaseException as exc:
                        row["cleanup"]["witness_error_type"] = type(exc).__name__
                        row["status"] = "BLOCKED"
                    try:
                        manager.close()
                        row["cleanup"]["client_closed"] = True
                    except BaseException as exc:
                        row["cleanup"]["close_error_type"] = type(exc).__name__
                if callable(delete_owned_collection):
                    try:
                        delete_owned_collection()
                        row["cleanup"]["collection_deleted"] = True
                    except BaseException as exc:
                        row["cleanup"]["delete_error_type"] = type(exc).__name__
                if not all(row["cleanup"].get(key) is True for key in ("reset_empty", "client_closed", "collection_deleted")):
                    row["status"] = "BLOCKED"
            stopped = row["status"] != "COMPLETE"
            rows.append(row)
    all_complete = all(row["status"] == "COMPLETE" for row in rows)
    equal = None
    if all_complete:
        equal = all(typed_json_equal(_stable_row_signature(rows[index]), _stable_row_signature(rows[index + 1]))
                    for index in range(0, len(rows), 2))
    complete = all_complete and equal is True
    steps = [step for row in rows for step in row["steps"]]
    observed_matches = {"state_step_matches": sum(step["state_match"] is True for step in steps),
                        "final_state_matches": sum(row["final_state_match"] is True for row in rows),
                        "retrieval_typed_object_matches": sum(row["retrieval_typed_object_match"] is True for row in rows)}
    return {
        "status": "COMPLETE" if complete else "BLOCKED",
        "evidence_class": "cross_domain_manager_state_retrieval", "scientific_evidence": False,
        "benchmark_accuracy_claimed": False, "source_formal_task_release": True,
        "cells": len(cells), "events": sum(len(cell["events"]) for cell in cells),
        "repetitions": repetitions, "expected_operations": sum(len(cell["events"]) for cell in cells) * repetitions,
        **{metric: value if complete else None for metric, value in observed_matches.items()},
        "observed_matches": observed_matches, "observed_operation_counts": counts,
        "repetitions_equal": equal, "repeatability_id_policy": "entry_ids_alpha_renamed_first_seen; canonical_namespaces_and_source_ids_preserved",
        "model_loads": None, "generations": None, "provider_model_calls": None, "rows": rows,
    }


def build_qdrant_manager_factory(
    *,
    endpoint: str,
    qdrant_base_path: str | Path,
    source_revision: str,
    runtime_revision: str,
    source_hash: str,
    runtime_hash: str,
    api_key_env: str = "MUB_DIAGNOSTIC_API_KEY",
    client_factory: Callable[..., Any] | None = None,
    transport_factory: Callable[[], Any] | None = None,
    run_scope: str | None = None,
) -> Callable[[Mapping[str, Any], int], tuple[Any, Callable[[], None]]]:
    """Build a direct-Qdrant manager factory for an already qualified server.

    The returned factory creates one collection per cell/repetition and a
    separate control client for deletion after the manager closes. The API key
    is read only from the named process environment variable.
    """
    _require(endpoint == "http://127.0.0.1:16333", "cross-domain Qdrant endpoint must be loopback")
    base = Path(qdrant_base_path)
    _require(base.is_absolute(), "cross-domain Qdrant base path must be absolute")
    _require(type(source_revision) is str and len(source_revision) == 40 and source_revision != "0" * 40,
             "source revision must be a nonzero Git SHA")
    _require(type(runtime_revision) is str and len(runtime_revision) == 40 and runtime_revision != "0" * 40,
             "runtime revision must be a nonzero Git SHA")
    _require(type(source_hash) is str and len(source_hash) == 64 and source_hash != "0" * 64,
             "source hash must be a nonzero SHA-256")
    _require(type(runtime_hash) is str and len(runtime_hash) == 64 and runtime_hash != "0" * 64,
             "runtime hash must be a nonzero SHA-256")
    _require(type(api_key_env) is str and api_key_env and "=" not in api_key_env,
             "API-key environment variable name is invalid")
    scope = uuid.uuid4().hex if run_scope is None else run_scope
    _require(type(scope) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scope) is not None,
             "cross-domain run scope is invalid")
    _require(transport_factory is None or callable(transport_factory), "transport factory must be callable")
    from mub.vnext.external.providers.qdrant_native_multi_object import QdrantNativeMultiObjectProvider
    from mub.vnext.external.providers.qdrant_native_adapter import (
        QdrantNativeExternalManagerV1,
        build_qdrant_native_manager_configuration,
    )
    if client_factory is None:
        _require(importlib.metadata.version("qdrant-client") == "1.19.0",
                 "cross-domain Qdrant client must equal 1.19.0")
        from qdrant_client import QdrantClient, models

        def make_client() -> Any:
            api_key = os.environ.get(api_key_env)
            _require(type(api_key) is str and api_key, "Qdrant API-key environment variable is unset")
            transport = transport_factory() if transport_factory is not None else None
            kwargs = {} if transport is None else {"transport": transport}
            client = None
            try:
                client = QdrantClient(url=endpoint, api_key=api_key, prefer_grpc=False,
                                      timeout=10, trust_env=False, check_compatibility=False, **kwargs)
                client.models = models
                return client
            except BaseException as exc:
                resource = client if client is not None else transport
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as cleanup_exc:
                        exc.client_cleanup_error_type = type(cleanup_exc).__name__
                raise
    else:
        _require(callable(client_factory), "client factory must be callable")
        _require(transport_factory is None, "transport factory is for production clients only")
        make_client = client_factory

    def factory(cell: Mapping[str, Any], repetition_index: int) -> tuple[Any, Callable[[], None]]:
        _validate_public_cell(cell)
        _require(type(repetition_index) is int and repetition_index in {0, 1}, "invalid repetition index")
        release_name = cell["release_name"]
        task_id = cell["task_id"]
        run_id = f"cross-domain-{scope}-{release_name}-{task_id}-{repetition_index}"
        storage_path = base / run_id
        manager_client = None
        control_client = None
        manager = None
        collection_name = None
        owned = False
        diagnostics: dict[str, Any] = {"collection_claim_status": "NOT_ATTEMPTED"}
        try:
            manager_client = make_client()
            control_client = make_client()
            configuration = build_qdrant_native_manager_configuration(
                run_id=run_id, path=storage_path, source_revision=source_revision,
                runtime_revision=runtime_revision, source_hash=source_hash, runtime_hash=runtime_hash,
            )
            collection_name = configuration.provider_configuration.collection_name
            diagnostics["collection_name"] = collection_name
            if control_client.collection_exists(collection_name=collection_name):
                diagnostics["collection_claim_status"] = "FOREIGN_REFUSED"
                raise ValueError("cross-domain Qdrant collection already exists")
            client_models = getattr(control_client, "models", None)
            vector_params = getattr(client_models, "VectorParams", None)
            if callable(vector_params):
                distance = getattr(getattr(client_models, "Distance", None), "COSINE", "Cosine")
                vectors_config = vector_params(size=384, distance=distance)
            else:
                vectors_config = {"size": 384, "distance": "Cosine"}
            # Only a successful explicit create confers deletion authority. A
            # lost reply is ambiguous and must never be resolved by deleting.
            diagnostics["collection_claim_status"] = "AMBIGUOUS"
            created = control_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
            _require(created is True, "cross-domain collection claim did not return true")
            owned = True
            diagnostics["collection_claim_status"] = "OWNED"
            manager = QdrantNativeExternalManagerV1(
                configuration=configuration,
                provider_factory=lambda **kwargs: QdrantNativeMultiObjectProvider(client=manager_client, **kwargs),
            )
        except BaseException as exc:
            if manager is not None:
                try:
                    manager.reset(None)
                    diagnostics["reset_empty"] = list(manager.export_entries()) == []
                except BaseException as cleanup_exc:
                    diagnostics["reset_error_type"] = type(cleanup_exc).__name__
                try:
                    manager.close()
                    diagnostics["manager_client_closed"] = True
                except BaseException as cleanup_exc:
                    diagnostics["manager_close_error_type"] = type(cleanup_exc).__name__
            if manager_client is not None and diagnostics.get("manager_client_closed") is not True:
                try:
                    manager_client.close()
                    diagnostics["manager_client_closed"] = True
                except BaseException as cleanup_exc:
                    diagnostics["manager_close_error_type"] = type(cleanup_exc).__name__
            if control_client is not None:
                if owned:
                    try:
                        _require(control_client.delete_collection(collection_name=collection_name) is True,
                                 "owned collection deletion failed")
                        _require(not control_client.collection_exists(collection_name=collection_name),
                                 "owned collection remains")
                        diagnostics["collection_deleted"] = True
                    except BaseException as cleanup_exc:
                        diagnostics["delete_error_type"] = type(cleanup_exc).__name__
                try:
                    control_client.close()
                    diagnostics["control_client_closed"] = True
                except BaseException as cleanup_exc:
                    diagnostics["control_close_error_type"] = type(cleanup_exc).__name__
            exc.cleanup_diagnostics = diagnostics
            raise

        def delete_owned_collection() -> None:
            _require(owned, "collection deletion is not authorized")
            primary_error = None
            try:
                _require(control_client.delete_collection(collection_name=collection_name) is True,
                         "cross-domain Qdrant collection deletion failed")
                _require(not control_client.collection_exists(collection_name=collection_name),
                         "cross-domain Qdrant collection remains after deletion")
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    control_client.close()
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    primary_error.control_cleanup_error_type = type(exc).__name__

        return manager, delete_owned_collection

    return factory


__all__ = [
    "BEA_INDEX_SHA256",
    "NOAA_INDEX_SHA256",
    "load_public_cells",
    "run_cross_domain_cells",
    "build_qdrant_manager_factory",
]
