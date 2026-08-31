from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, object_identity, typed_json_equal
from mub.vnext.contracts.v3.adapter import AdapterActionResultV3
from mub.vnext.external.canaries_v3 import _rename_no_replace
from mub.vnext.io import sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.external.security import scan_for_secrets
from scripts import vnext_plan_main_track_factorial as planner

SCHEMA_VERSION = "memupdatebench.main-track.factorial.manager-fixture.v1"
ROW_SCHEMA_VERSION = f"{SCHEMA_VERSION}.row"
INDEX_SCHEMA_VERSION = f"{SCHEMA_VERSION}.artifact-index"
EVIDENCE_CLASS = "manager_state_retrieval_fixture"
TEST_ONLY_EVIDENCE_CLASS = "manager_state_retrieval_fixture_test_only"
EXTERNAL_BOUNDARY = (
    "External-manager state and retrieval fixture only; no prompted-answer replay, "
    "provider/model/network/database/GPU execution, or broad scientific claim."
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RAW_FIELD_NAMES = {
    "prompt", "raw_prompt", "rendered_prompt", "rendered_chat_prompt",
    "output", "raw_output", "generated_text", "reasoning", "reasoning_content",
    "raw_reasoning",
}
_MANAGER_IDS = {
    "letta": "letta_0_16_8_block_profile",
    "langgraph": "langmem_0_0_30_profile",
}
_CELL_IDS = {
    "letta": "letta_profile__qwen35_answer",
    "langgraph": "langgraph_store_custom_adapter__qwen35_answer",
}
_EXECUTION_BOUNDARY = {
    "provider_calls": 0,
    "model_loads": 0,
    "database_accesses": 0,
    "network_calls": 0,
    "gpu_calls": 0,
    "executable_calls": 0,
    "remote_operations": 0,
}
_ALLOWED_REASON_KINDS = {"manager_contract", "manager_capability", "query_contract"}
_ALLOWED_REASON_CODES = {
    "single_query_required", "historical_query_not_supported", "multi_object_query_not_supported",
    "single_target_object_required", "current_single_object_retrieved_prompt_required",
    "manager_capability_not_supported", "ttl_delete_not_supported", "scoped_delete_not_supported",
}


class RuntimeIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunnerArgs:
    manifest: Path
    candidate_root: Path
    audit_attestation: Path
    manager_kind: str
    cell_id: str
    output_root: Path
    execution_mode: str = "production"


class VisibleEventExtractor(Protocol):
    """Production seam for an offline Qwen visible-event CRUD extractor.

    The factory must return an already configured implementation. ``extract`` returns
    only normalized metadata and hashes; raw prompts, generations, and reasoning must
    never be returned to or serialized by this runner.
    """

    identity: Mapping[str, Any]

    def extract(self, event: Any, object_key: Any) -> Mapping[str, Any]: ...


class ExternalManager(Protocol):
    """Production seam for a Letta or LangGraph manager adapter."""

    identity: Mapping[str, Any]

    def reset(self, task: Any) -> None: ...

    def ingest(self, event: Any, *, operation: str, value: Any, object_key: Any) -> Mapping[str, Any]: ...

    def export_entries(self) -> Sequence[Mapping[str, Any]]: ...

    def retrieve(self, query: Any) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def canonical_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _raw_field_location(value: Any, location: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{location} has a non-string field name"
            lowered = key.casefold()
            if lowered in _RAW_FIELD_NAMES or lowered.startswith(("raw_prompt_", "raw_output_", "raw_reasoning_")):
                return f"{location}.{key} is a raw prompt/output/reasoning field"
            found = _raw_field_location(child, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _raw_field_location(child, f"{location}[{index}]")
            if found:
                return found
    return None


def validate_public_payload(value: Any) -> Any:
    """Reject raw model material and secrets before public artifact publication."""
    found = _raw_field_location(value)
    if found:
        raise ValueError(found)
    if scan_for_secrets(value):
        raise ValueError("public artifact failed security scan")
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("manifest must be an absolute regular file")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if type(manifest) is not dict or canonical_json_bytes(manifest) != raw:
        raise ValueError("manifest must be canonical JSON")
    if manifest.get("schema_version") != "memupdatebench.main-track.factorial-plan.v1":
        raise ValueError("factorial manifest schema mismatch")
    planner._validate_manifest_shape(manifest)
    return manifest, raw


def _validate_inputs(args: RunnerArgs, manifest: dict[str, Any], manifest_raw: bytes) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if args.manager_kind not in _MANAGER_IDS:
        raise ValueError("manager_kind must be letta or langgraph")
    if args.cell_id != _CELL_IDS[args.manager_kind]:
        raise ValueError("cell_id must be the Qwen extraction cell for manager_kind")
    cell = next((item for item in manifest["cells"] if item.get("cell_id") == args.cell_id), None)
    if cell is None:
        raise ValueError("requested manifest cell is absent")
    if manifest.get("execution_boundary") != _EXECUTION_BOUNDARY:
        raise ValueError("manifest execution boundary is invalid")
    supplied_candidate = Path(args.candidate_root).resolve(strict=True)
    declared_candidate = Path(manifest["candidate"].get("root", ""))
    try:
        declared_candidate = declared_candidate.resolve(strict=True)
    except FileNotFoundError:
        declared_candidate = declared_candidate.resolve(strict=False)
    if declared_candidate != supplied_candidate:
        raise ValueError("manifest candidate root binding does not match supplied path")
    supplied_audit = Path(args.audit_attestation).resolve(strict=True)
    declared_audit = Path(manifest["audit_attestation"].get("path", ""))
    try:
        declared_audit = declared_audit.resolve(strict=True)
    except FileNotFoundError:
        declared_audit = declared_audit.resolve(strict=False)
    if declared_audit != supplied_audit:
        raise ValueError("manifest audit attestation path binding does not match supplied path")
    expected_manager = {
        key: value for key, value in planner._MANAGER_SPECS[args.manager_kind].items()
        if key != "capabilities"
    }
    if cell.get("manager") != expected_manager or cell.get("manager_id") != expected_manager["manager_id"]:
        raise ValueError("manifest manager specification does not match manager_kind")
    if cell.get("extractor") != planner._extractor_spec(args.manager_kind):
        raise ValueError("manifest extractor specification does not match manager_kind")
    if cell.get("extractor", {}).get("role") != "visible_event_crud_extraction":
        raise ValueError("manifest extractor role is invalid")

    provenance = planner.validate_candidate(args.candidate_root, args.audit_attestation)
    if manifest["candidate_artifact_hashes"] != provenance["artifact_hashes"]:
        raise ValueError("manifest candidate hashes do not match candidate")
    if manifest["candidate"]["release_index_sha256"] != provenance["release_index_sha256"]:
        raise ValueError("manifest candidate release index hash does not match candidate")
    if manifest["audit_attestation_sha256"] != provenance["audit_attestation_sha256"]:
        raise ValueError("manifest audit attestation hash does not match attestation")

    tasks = planner.select_test_tasks(args.candidate_root)
    task_ids = [task.task_id for task in tasks]
    if task_ids != manifest["task_view"]["task_ids"]:
        raise ValueError("manifest task order does not match candidate test order")
    expected_supported: list[str] = []
    expected_unsupported: list[dict[str, Any]] = []
    for task in tasks:
        reason = planner.task_support_reason(task, args.manager_kind)
        if reason is None:
            expected_supported.append(task.task_id)
        else:
            expected_unsupported.append({"task_id": task.task_id, **reason})
    if cell.get("supported_task_ids") != expected_supported or cell.get("unsupported_tasks") != expected_unsupported:
        raise ValueError("manifest support reasons, partition, or order do not match planner")
    try:
        post_parse_provenance = planner.validate_candidate(args.candidate_root, args.audit_attestation)
    except Exception as exc:
        raise ValueError("candidate changed after parsing") from exc
    if (
        post_parse_provenance["release_index_sha256"] != provenance["release_index_sha256"]
        or post_parse_provenance["artifact_hashes"] != provenance["artifact_hashes"]
    ):
        raise ValueError("candidate changed after parsing")
    supported_ids = set(expected_supported)
    tasks_by_id = {task.task_id: task for task in tasks}
    if any(tasks_by_id[task_id].task_family != "noop_write_discipline" for task_id in expected_supported):
        raise ValueError("manager fixture supported tasks must be Family D")
    hashes = {task_id: sha256_model(tasks_by_id[task_id]) for task_id in task_ids}
    return tasks, cell, provenance, {"manifest_sha256": _sha256_bytes(manifest_raw), "task_hashes": hashes}


def _identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} identity must be a non-empty object")
    result = dict(value)
    validate_public_payload(result)
    return result


def _validate_runtime_identity(
    actual: Any, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    observed = _identity(actual, label)
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            raise RuntimeIdentityError(f"{label} identity mismatch for {key}")
    return observed


def _factory_is_production_bound(factory: Any) -> bool:
    return bool(
        getattr(factory, "production_bound", False)
        or getattr(factory, "execution_mode", None) == "production"
    )


def _extraction(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_payload(value)
    required = {"operation", "value", "output_sha256", "generated_tokens", "latency_ms"}
    if set(value) != required:
        raise ValueError("extractor result must contain normalized fields only")
    operation = value["operation"]
    if not isinstance(operation, str) or operation.lower() not in {"add", "update", "noop", "delete"}:
        raise ValueError("extractor operation is invalid")
    tokens = value["generated_tokens"]
    latency = value["latency_ms"]
    if type(tokens) is not int or tokens < 0:
        raise ValueError("generated_tokens must be a nonnegative integer")
    if type(latency) not in (int, float) or isinstance(latency, bool) or not math.isfinite(float(latency)) or latency < 0:
        raise ValueError("latency_ms must be a finite nonnegative number")
    return {
        "operation": operation.lower(),
        "value": value["value"],
        "output_sha256": _require_sha(value["output_sha256"], "output_sha256"),
        "generated_tokens": tokens,
        "latency_ms": float(latency),
    }


def _manager_result(
    value: Mapping[str, Any] | AdapterActionResultV3,
    *,
    requested_operation: str | None = None,
) -> tuple[str, list[str]]:
    declared_requested: str | None = None
    if isinstance(value, AdapterActionResultV3):
        requested_value = value.requested_action.operation
        declared_requested = requested_value.value.lower() if requested_value is not None else None
        effective_value = value.effective_action.operation
        effective = effective_value.value.lower() if effective_value is not None else "noop"
        affected = list(value.affected_entry_ids)
    else:
        validate_public_payload(value)
        if set(value) - {"effective_operation", "affected_entry_ids"} or "effective_operation" not in value:
            raise ValueError("manager result contains unsupported fields")
        if "affected_entry_ids" not in value:
            raise ValueError("manager result requires affected_entry_ids")
        effective = value["effective_operation"]
        if not isinstance(effective, str) or effective.lower() not in {"add", "update", "noop", "delete"}:
            raise ValueError("manager effective operation is invalid")
        affected = value["affected_entry_ids"]
        if type(affected) is not list or any(type(item) is not str for item in affected):
            raise ValueError("manager affected_entry_ids must be a list of strings")
        effective = effective.lower()
        affected = list(affected)
    if requested_operation is not None:
        requested = requested_operation.lower()
        if requested not in {"add", "update", "noop", "delete"}:
            raise ValueError("requested operation is invalid")
        if declared_requested is not None and declared_requested != requested:
            raise ValueError("manager result does not match requested operation")
        if effective != requested and not (effective == "noop" and requested in {"add", "update", "delete"}):
            raise ValueError("manager result does not match requested operation")
    if effective in {"add", "update", "delete"} and not affected:
        raise ValueError("manager mutations require affected_entry_ids")
    return effective, affected


def _entry(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_payload(value)
    required = {"entry_id", "object_key", "value", "content", "source_event_ids", "score", "rank", "version_metadata"}
    if set(value) != required:
        raise ValueError("manager entry shape is invalid")
    if type(value["entry_id"]) is not str or not value["entry_id"]:
        raise ValueError("manager entry_id is invalid")
    if not isinstance(value["object_key"], Mapping):
        raise ValueError("manager entry object key is invalid")
    try:
        key = FrozenMemoryObjectKey.model_validate(value["object_key"], strict=True)
        object_identity(key)
    except (TypeError, ValueError) as exc:
        raise ValueError("manager entry object key is invalid") from exc
    if not isinstance(value["source_event_ids"], (list, tuple)) or any(type(item) is not str for item in value["source_event_ids"]):
        raise ValueError("manager entry source_event_ids are invalid")
    if type(value["rank"]) is not int or value["rank"] < 1:
        raise ValueError("manager entry rank is invalid")
    if type(value["score"]) not in (int, float) or isinstance(value["score"], bool) or not math.isfinite(float(value["score"])):
        raise ValueError("manager entry score is invalid")
    if not isinstance(value["version_metadata"], Mapping):
        raise ValueError("manager entry version_metadata is invalid")
    result = dict(value)
    result["object_key"] = key.model_dump(mode="json")
    return result


def _retrieval(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_payload(value)
    if set(value) != {"entries", "context_order", "version_metadata"}:
        raise ValueError("retrieval result shape is invalid")
    entries = value["entries"]
    if not isinstance(entries, (list, tuple)):
        raise ValueError("retrieval entries must be ordered")
    if type(value["context_order"]) is not str or not value["context_order"]:
        raise ValueError("retrieval context_order is invalid")
    if not isinstance(value["version_metadata"], Mapping):
        raise ValueError("retrieval version_metadata is invalid")
    normalized = {
        "entries": [_entry(item) for item in entries],
        "context_order": value["context_order"],
        "version_metadata": dict(value["version_metadata"]),
    }
    return normalized


def _bind_retrieval_trace(trace: dict[str, Any], query: Any, target_identity: str) -> dict[str, Any]:
    query_target_identities = [object_identity(key) for key in query.target_object_keys]
    if query_target_identities != [target_identity]:
        raise ValueError("retrieval query binding does not match target object")
    bound = dict(trace)
    bound["query_id"] = query.query_id
    bound["query_target_object_identities"] = query_target_identities
    return bound


def _validate_row_consistency(row: Mapping[str, Any]) -> None:
    state = row.get("state")
    state_pairs = {
        "state_accuracy": "state_accuracy",
        "parsed_final_value": "final_value",
        "final_memory_size": "final_memory_size",
        "stable_entry_id": "stable_entry_id",
    }
    if state is None:
        if any(row.get(flat) is not None for flat in state_pairs):
            raise ValueError("flat state fields drift from nested state")
    else:
        if not isinstance(state, Mapping):
            raise ValueError("state must be an object")
        for flat, nested in state_pairs.items():
            if not typed_json_equal(row.get(flat), state.get(nested)):
                raise ValueError(f"flat state field drifts from nested state: {flat}")
    retrieval = row.get("retrieval")
    retrieval_pairs = {
        "retrieval_trace": "trace",
        "retrieval_trace_sha256": "trace_sha256",
        "gold_retrieved": "gold_retrieved",
    }
    if retrieval is None:
        if any(row.get(flat) is not None for flat in retrieval_pairs):
            raise ValueError("flat retrieval fields drift from nested retrieval")
    else:
        if not isinstance(retrieval, Mapping):
            raise ValueError("retrieval must be an object")
        for flat, nested in retrieval_pairs.items():
            if not typed_json_equal(row.get(flat), retrieval.get(nested)):
                raise ValueError(f"flat retrieval field drifts from nested retrieval: {flat}")


def _row_base(task: Any, cell: Mapping[str, Any], manager_kind: str, task_sha: str) -> dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "task_id": task.task_id,
        "task_sha256": task_sha,
        "cell_id": cell["cell_id"],
        "manager": {
            "kind": manager_kind,
            "manager_id": cell["manager_id"],
            "system_name": cell["manager"].get("system_name"),
            "system_version": cell["manager"].get("system_version"),
        },
        "status": None,
        "execution_status": None,
        "reason_code": None,
        "reason_kind": None,
        "extractor": None,
        "manager_runtime_identity": None,
        "event_records": None,
        "extractions": None,
        "affected_entry_ids": None,
        "reconciliation_count": None,
        "state": None,
        "state_accuracy": None,
        "parsed_final_value": None,
        "final_memory_size": None,
        "stable_entry_id": None,
        "retrieval": None,
        "retrieval_trace": None,
        "retrieval_trace_sha256": None,
        "gold_retrieved": None,
    }


def _unsupported_row(task: Any, cell: Mapping[str, Any], manager_kind: str, task_sha: str, reason: Mapping[str, Any]) -> dict[str, Any]:
    row = _row_base(task, cell, manager_kind, task_sha)
    row.update({
        "status": "UNSUPPORTED",
        "execution_status": "NOT_RUN",
        "reason_code": reason.get("reason_code"),
        "reason_kind": reason.get("reason_kind"),
    })
    if not row["reason_code"] or not row["reason_kind"]:
        raise ValueError("unsupported manifest rows require typed reasons")
    _validate_row_consistency(row)
    validate_public_payload(row)
    return row


def _supported_row(task: Any, cell: Mapping[str, Any], manager_kind: str, task_sha: str, extractor: VisibleEventExtractor, manager: ExternalManager) -> dict[str, Any]:
    row = _row_base(task, cell, manager_kind, task_sha)
    row["status"] = "SUPPORTED"
    row["execution_status"] = "PASS"
    row["extractor"] = _identity(extractor.identity, "extractor")
    row["manager_runtime_identity"] = _identity(manager.identity, "manager")
    manager.reset(task)
    records = []
    affected_entry_ids: list[str] = []
    reconciliation_count = 0
    initialized = False
    for event in task.events:
        parsed = _extraction(extractor.extract(event, task.target_objects[0]))
        requested = parsed["operation"]
        operation = requested
        if requested == "update" and not initialized:
            operation = "add"
            reconciliation_count += 1
        result = manager.ingest(event, operation=operation, value=parsed["value"], object_key=task.target_objects[0])
        effective, affected = _manager_result(result, requested_operation=operation)
        affected_entry_ids.extend(affected)
        if effective in {"add", "update"}:
            initialized = True
        elif effective == "delete":
            initialized = False
        records.append({
            "event_id": event.event_id,
            "operation": requested,
            "effective_operation": effective,
            "affected_entry_ids": list(affected),
            "output_sha256": parsed["output_sha256"],
            "generated_tokens": parsed["generated_tokens"],
            "latency_ms": parsed["latency_ms"],
        })
    exported = [_entry(item) for item in manager.export_entries()]
    target_identity = object_identity(task.target_objects[0])
    target_entries = [
        item
        for item in exported
        if object_identity(FrozenMemoryObjectKey.model_validate(item["object_key"], strict=True)) == target_identity
    ]
    final_value = target_entries[0]["value"] if len(target_entries) == 1 else None
    gold = task.gold_evidence[0].answer
    row["event_records"] = records
    row["extractions"] = records
    row["affected_entry_ids"] = list(dict.fromkeys(affected_entry_ids))
    row["reconciliation_count"] = reconciliation_count
    row["state"] = {
        "state_accuracy": len(target_entries) == 1 and typed_json_equal(final_value, gold),
        "final_value": final_value,
        "final_memory_size": len(exported),
        "stable_entry_id": len(target_entries) == 1,
        "gold_sha256": _sha256_bytes(canonical_json_bytes(gold)),
    }
    trace = _bind_retrieval_trace(
        _retrieval(manager.retrieve(task.queries[0])),
        task.queries[0],
        target_identity,
    )
    trace_hash = _sha256_bytes(canonical_json_bytes(trace))
    row["retrieval"] = {
        "trace": trace,
        "trace_sha256": trace_hash,
        "gold_retrieved": any(
            object_identity(FrozenMemoryObjectKey.model_validate(item["object_key"], strict=True)) == target_identity
            and typed_json_equal(item["value"], gold)
            for item in trace["entries"]
        ),
        "retrieved_count": len(trace["entries"]),
    }
    row["state_accuracy"] = row["state"]["state_accuracy"]
    row["parsed_final_value"] = row["state"]["final_value"]
    row["final_memory_size"] = row["state"]["final_memory_size"]
    row["stable_entry_id"] = row["state"]["stable_entry_id"]
    row["retrieval_trace"] = trace
    row["retrieval_trace_sha256"] = trace_hash
    row["gold_retrieved"] = row["retrieval"]["gold_retrieved"]
    _validate_row_consistency(row)
    validate_public_payload(row)
    return row


def _validate_output_root(output: Path, inputs: Sequence[Path]) -> Path:
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve(strict=False)
    for source in inputs:
        source_resolved = source.resolve(strict=True)
        if resolved == source_resolved or source_resolved in resolved.parents or resolved in source_resolved.parents:
            raise ValueError("output_root must be separate from inputs")
    return output


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _publish_output(
    output: Path,
    artifacts: Mapping[str, bytes],
    source_paths: Sequence[Path],
) -> Path:
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("output_root parent directory does not exist")
    if not artifacts:
        raise ValueError("output publication requires at least one artifact")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        destinations: dict[Path, bytes] = {}
        for name, raw in artifacts.items():
            relative = Path(name)
            if relative.is_absolute() or relative.name != name:
                raise ValueError("output artifact names must be single path components")
            if not isinstance(raw, bytes):
                raise TypeError("output artifact payloads must be bytes")
            destinations[stage / name] = raw
        publish_files_atomically(
            destinations,
            overwrite=False,
            source_paths=source_paths,
        )
        _rename_no_replace(stage, output)
        return output
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def run(
    args: RunnerArgs,
    *,
    extractor_factory: Callable[[], VisibleEventExtractor] | None = None,
    manager_factory: Callable[[Any, Mapping[str, Any]], ExternalManager] | None = None,
    manifest: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest or args.manifest)
    output = _validate_output_root(Path(args.output_root), (manifest_path, Path(args.candidate_root), Path(args.audit_attestation)))
    if args.execution_mode not in {"production", "injected_test_only"}:
        raise ValueError("execution_mode must be production or injected_test_only")
    if extractor_factory is None or manager_factory is None:
        raise RuntimeError("production adapters are not configured; inject extractor_factory and manager_factory")
    if args.execution_mode == "production":
        if not _factory_is_production_bound(extractor_factory) or not _factory_is_production_bound(manager_factory):
            raise RuntimeError("production mode requires production-bound factories")
    loaded_manifest, manifest_raw = _load_manifest(manifest_path)
    tasks, cell, provenance, binding = _validate_inputs(args, loaded_manifest, manifest_raw)
    reason_by_id = {item["task_id"]: item for item in cell["unsupported_tasks"]}
    supported_ids = set(cell["supported_task_ids"])
    extractor = extractor_factory()
    if not hasattr(extractor, "extract"):
        raise RuntimeError("injected extractor does not implement extract")
    expected_extractor_identity = cell["extractor"].get("identity")
    if args.execution_mode == "production":
        if expected_extractor_identity is None:
            raise RuntimeError("production manifest extractor identity is missing")
        _validate_runtime_identity(extractor.identity, expected_extractor_identity, "extractor")
    rows: list[dict[str, Any]] = []
    cleanup_errors: list[BaseException] = []
    try:
        for task in tasks:
            task_sha = binding["task_hashes"][task.task_id]
            if task.task_id not in supported_ids:
                rows.append(_unsupported_row(task, cell, args.manager_kind, task_sha, reason_by_id[task.task_id]))
                continue
            manager = None
            try:
                manager = manager_factory(task, cell)
                if args.execution_mode == "production":
                    _validate_runtime_identity(manager.identity, cell["manager"], "manager")
                rows.append(_supported_row(task, cell, args.manager_kind, task_sha, extractor, manager))
            except RuntimeIdentityError:
                raise
            except Exception:
                row = _row_base(task, cell, args.manager_kind, task_sha)
                row.update({
                    "status": "SUPPORTED",
                    "execution_status": "FAIL",
                    "reason_code": "execution_failed",
                    "reason_kind": "runtime",
                    "extractor": _identity(extractor.identity, "extractor"),
                    "event_records": [],
                    "extractions": [],
                    "affected_entry_ids": [],
                    "reconciliation_count": 0,
                })
                _validate_row_consistency(row)
                validate_public_payload(row)
                rows.append(row)
            finally:
                close = getattr(manager, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:
                cleanup_errors.append(exc)
    if cleanup_errors:
        raise RuntimeError(
            "resource cleanup failed"
        ) from cleanup_errors[0]
    expected_ids = [task.task_id for task in tasks]
    if [row["task_id"] for row in rows] != expected_ids or len(rows) != 720:
        raise ValueError("terminal rows must preserve exact manifest task order and cardinality")
    supported = [row for row in rows if row["status"] == "SUPPORTED"]
    unsupported = [row for row in rows if row["status"] == "UNSUPPORTED"]
    state_rows = [row for row in supported if row["state"] is not None]
    retrieval_rows = [row for row in supported if row["retrieval"] is not None]
    failed = sum(row["execution_status"] == "FAIL" for row in supported)
    all_supported_terminal = (
        len(supported) == len(cell["supported_task_ids"])
        and all(row["execution_status"] in {"PASS", "FAIL"} for row in supported)
    )
    status = "PASS" if len(rows) == 720 and all_supported_terminal and failed == 0 else "FAIL"
    evidence_class = (
        EVIDENCE_CLASS
        if args.execution_mode == "production"
        else TEST_ONLY_EVIDENCE_CLASS
    )
    boundary_observed = args.execution_mode == "production"
    reason_counts = dict(sorted(Counter(row["reason_code"] for row in unsupported).items()))
    rows_raw = _canonical_jsonl(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "cell_id": cell["cell_id"],
        "manager_kind": args.manager_kind,
        "manager_id": cell["manager_id"],
        "evidence_class": evidence_class,
        "scientific_evidence": args.execution_mode == "production",
        "execution_boundary_observed": boundary_observed,
        "external_manager_boundary": EXTERNAL_BOUNDARY,
        "requested": len(tasks),
        "terminal_rows": len(rows),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "failed": failed,
        "state_accuracy": sum(bool(row["state"]["state_accuracy"]) for row in state_rows) / len(state_rows) if state_rows else None,
        "state_accuracy_denominator": len(state_rows),
        "gold_retrieval_rate": sum(bool(row["retrieval"]["gold_retrieved"]) for row in retrieval_rows) / len(retrieval_rows) if retrieval_rows else None,
        "retrieval_denominator": len(retrieval_rows),
        "unsupported_reason_counts": reason_counts,
        "execution_boundary": dict(_EXECUTION_BOUNDARY),
        "provider_calls": 0,
        "retries": 0,
        "execution_mode": args.execution_mode,
        "manifest_sha256": binding["manifest_sha256"],
        "candidate_artifact_hashes": provenance["artifact_hashes"],
        "audit_attestation_sha256": provenance["audit_attestation_sha256"],
        "runner_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "rows_sha256": _sha256_bytes(rows_raw),
    }
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": status,
        "cell_id": cell["cell_id"],
        "manager_kind": args.manager_kind,
        "evidence_class": evidence_class,
        "scientific_evidence": args.execution_mode == "production",
        "execution_boundary_observed": boundary_observed,
        "external_manager_boundary": EXTERNAL_BOUNDARY,
        "manifest_sha256": binding["manifest_sha256"],
        "candidate_artifact_hashes": provenance["artifact_hashes"],
        "audit_attestation_sha256": provenance["audit_attestation_sha256"],
        "runner_source_sha256": summary["runner_source_sha256"],
        "artifacts": {
            "manager_rows.jsonl": {"sha256": _sha256_bytes(rows_raw), "bytes": len(rows_raw), "record_count": len(rows)},
            "manager_summary.json": {"sha256": _sha256_bytes(canonical_json_bytes(summary)), "bytes": len(canonical_json_bytes(summary)), "record_count": 1},
        },
        "execution_boundary": dict(_EXECUTION_BOUNDARY),
        "provider_calls": 0,
        "retries": 0,
        "execution_mode": args.execution_mode,
    }
    validate_public_payload(summary)
    validate_public_payload(index)
    payloads = {
        "manager_rows.jsonl": rows_raw,
        "manager_summary.json": canonical_json_bytes(summary),
        "artifact_index.json": canonical_json_bytes(index),
    }
    _publish_output(
        output,
        payloads,
        (
            manifest_path,
            Path(args.candidate_root) / "tasks.jsonl",
            Path(args.candidate_root) / "release_index.json",
            Path(args.audit_attestation),
        ),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one audited main-track external-manager state/retrieval fixture cell.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-attestation", type=Path, required=True)
    parser.add_argument("--manager-kind", choices=tuple(_MANAGER_IDS), required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("production", "injected_test_only"), default="production")
    args = parser.parse_args(argv)
    try:
        summary = run(RunnerArgs(**vars(args)))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__}, sort_keys=True))
        return 1
    status = summary.get("status")
    print(json.dumps({"status": status, "evidence_class": summary.get("evidence_class")}, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
