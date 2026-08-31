from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.io.atomic import publish_files_atomically

CANDIDATE_ROOT = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
AUDIT_ATTESTATION = (
    ROOT
    / "results"
    / "vnext"
    / "main_track_v1_audit_completion_attestation_v1"
    / "review_attestation.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "vnext"
    / "main_track_v1_factorial_plan_v1"
    / "factorial_manifest.json"
)

SCHEMA_VERSION = "memupdatebench.main-track.factorial-plan.v1"
QWEN_MODEL = {
    "model_id": "Qwen/Qwen3.5-9B",
    "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "tree_sha256": "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db",
}
MUSE_MODEL = {
    "model_id": "meta-models/Muse-Glimmer-30B-GGUF",
    "revision": "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
    "tree_sha256": "55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f",
}
CELL_IDS = (
    "reference__qwen35_answer",
    "reference__muse_answer",
    "letta_profile__qwen35_answer",
    "letta_profile__muse_answer",
    "langgraph_store_custom_adapter__qwen35_answer",
    "langgraph_store_custom_adapter__muse_answer",
)

_REFERENCE_CAPABILITIES = dict(ReferenceAdapterV3.capabilities_config)
_PROFILE_CAPABILITIES = {
    "supports_isolated_reset": True,
    "supports_event_ingest": True,
    "supports_add": True,
    "supports_update": True,
    "supports_noop": True,
    "supports_delete": True,
    "supports_ttl": False,
    "supports_native_answer": False,
    "exports_entries": True,
    "exports_raw_state": False,
    "exports_source_event_ids": True,
    "exports_timestamps_or_order": True,
    "exports_object_keys": True,
    "exports_values": True,
    "exports_retrieval_ids": True,
    "exports_retrieval_scores": True,
    "exports_action_trace": True,
    "supports_scoped_delete": False,
    "supports_historical_query": False,
    "exports_version_history": False,
    "supports_multi_object_query": False,
    "exports_evidence_linkage": True,
}

_MANAGER_SPECS = {
    "reference": {
        "manager_id": "reference",
        "adapter_id": "reference",
        "adapter_version": "1.0.0",
        "system_name": "oracle_smoke_only",
        "system_version": "1.0.0",
        "backend": "in_process_reference_replay",
        "capabilities": _REFERENCE_CAPABILITIES,
    },
    "letta": {
        "manager_id": "letta_0_16_8_block_profile",
        "adapter_id": "letta_0_16_8_block_profile",
        "adapter_version": "memupdatebench-letta-adapter-v1",
        "system_name": "letta_0_16_8_block_profile",
        "system_version": "0.16.8",
        "backend": "letta_block_profile",
        "capabilities": _PROFILE_CAPABILITIES,
    },
    "langgraph": {
        "manager_id": "langmem_0_0_30_profile",
        "adapter_id": "langmem_0_0_30_profile",
        "adapter_version": "memupdatebench-langmem-adapter-v1",
        "system_name": "langmem_0_0_30_profile",
        "system_version": "0.0.30",
        "backend": "langgraph_in_memory_store",
        "capabilities": _PROFILE_CAPABILITIES,
    },
}


_EXTRACTOR_SPECS = {
    "reference": {"role": "none", "identity": None},
    "letta": {
        "role": "visible_event_crud_extraction",
        "identity": {
            **QWEN_MODEL,
            "extractor_id": "qwen35_visible_event_crud_extractor",
            "extractor_version": "qwen35-visible-event-crud-extraction-v1",
        },
    },
    "langgraph": {
        "role": "visible_event_crud_extraction",
        "identity": {
            **QWEN_MODEL,
            "extractor_id": "qwen35_visible_event_crud_extractor",
            "extractor_version": "qwen35-visible-event-crud-extraction-v1",
        },
    },
}
_ANSWER_SPECS = {
    "qwen": {
        "role": "retrieved_context_prompted_answer",
        "identity": {
            **QWEN_MODEL,
            "answer_id": "qwen35_retrieved_context_prompted_answer",
            "answer_mode": "retrieved_prompt",
        },
    },
    "muse": {
        "role": "retrieved_context_prompted_answer",
        "identity": {
            **MUSE_MODEL,
            "answer_id": "muse_glimmer_retrieved_context_prompted_answer",
            "answer_mode": "retrieved_prompt",
        },
    },
}
_CELL_SPECS = {
    "reference__qwen35_answer": ("reference", "qwen"),
    "reference__muse_answer": ("reference", "muse"),
    "letta_profile__qwen35_answer": ("letta", "qwen"),
    "letta_profile__muse_answer": ("letta", "muse"),
    "langgraph_store_custom_adapter__qwen35_answer": ("langgraph", "qwen"),
    "langgraph_store_custom_adapter__muse_answer": ("langgraph", "muse"),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _real_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _real_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return path.resolve(strict=True)


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = _real_file(path, label)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} must be canonical JSON")
    return value, raw


def _candidate_hashes(candidate: Path, release_index: dict[str, Any]) -> dict[str, str]:
    artifacts = release_index.get("artifacts")
    if type(artifacts) is not list:
        raise ValueError("candidate release index artifacts are invalid")
    expected_from_index: dict[str, str] = {}
    for item in artifacts:
        if type(item) is not dict or type(item.get("path")) is not str or type(item.get("sha256")) is not str:
            raise ValueError("candidate release index artifact row is invalid")
        name = item["path"]
        digest = item["sha256"]
        if name in expected_from_index:
            raise ValueError(f"candidate release index contains duplicate artifact path: {name}")
        expected_from_index[name] = digest
    release_index_name = "release_index.json"
    expected_names = set(expected_from_index) | {release_index_name}
    entries = tuple(candidate.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("candidate artifacts must be regular files")
    observed_names = {item.name for item in entries}
    if observed_names != expected_names:
        raise ValueError("candidate artifact file set mismatch")
    observed: dict[str, str] = {}
    for name in sorted(expected_names):
        path = candidate / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate artifacts must be regular files")
        observed[name] = sha256_bytes(path.read_bytes())
    for name, expected in expected_from_index.items():
        if observed[name] != expected:
            raise ValueError(f"candidate artifact hash mismatch: {name}")
    return observed


def validate_candidate(candidate_root: Path | str, attestation_path: Path | str) -> dict[str, Any]:
    candidate = _real_directory(Path(candidate_root), "candidate root")
    index, index_raw = _load_canonical_json(candidate / "release_index.json", "candidate release index")
    if index.get("release_id") != "main_track_v1":
        raise ValueError("candidate release ID mismatch")
    if index.get("review_status") != "NOT_STARTED":
        raise ValueError("candidate review status must remain NOT_STARTED")
    hashes = _candidate_hashes(candidate, index)
    attestation, attestation_raw = _load_canonical_json(Path(attestation_path), "audit attestation")
    if attestation.get("release_id") != "main_track_v1":
        raise ValueError("audit attestation release ID mismatch")
    if attestation.get("review_status") != "PASS":
        raise ValueError("audit attestation review status is not PASS")
    if attestation.get("benchmark_release_eligible") is not True:
        raise ValueError("audit attestation is not benchmark eligible")
    if attestation.get("evidence_class") != "human_audit_completion_attestation":
        raise ValueError("audit attestation evidence class mismatch")
    if attestation.get("row_count") != 240 or attestation.get("unresolved_count") != 0:
        raise ValueError("audit attestation completion counts are invalid")
    if attestation.get("decision_counts") != {"block": 0, "needs_revision": 0, "pass": 240}:
        raise ValueError("audit attestation decision counts are invalid")
    for field in ("completed_packet_sha256", "source_packet_sha256", "selection_artifact_sha256"):
        if type(attestation.get(field)) is not str or re.fullmatch(r"[0-9a-f]{64}", attestation[field]) is None:
            raise ValueError(f"audit attestation {field} is invalid")
    if attestation.get("candidate_artifact_hashes") != hashes:
        raise ValueError("audit attestation candidate hashes mismatch")
    return {
        "candidate_root": str(candidate),
        "release_index_sha256": sha256_bytes(index_raw),
        "artifact_hashes": hashes,
        "release_index": index,
        "audit_attestation_path": str(Path(attestation_path).resolve(strict=True)),
        "audit_attestation_sha256": sha256_bytes(attestation_raw),
        "audit_attestation": attestation,
    }


def select_test_tasks(candidate_root: Path | str) -> list[MemUpdateTaskV3]:
    candidate = _real_directory(Path(candidate_root), "candidate root")
    tasks_path = candidate / "tasks.jsonl"
    if tasks_path.is_symlink() or not tasks_path.is_file():
        raise ValueError("candidate tasks must be a regular file")
    tasks = []
    for line in tasks_path.read_bytes().splitlines():
        if line.strip():
            tasks.append(MemUpdateTaskV3.model_validate(json.loads(line)))
    selected = [task for task in tasks if task.metadata.split.value == "test"]
    selected.sort(key=lambda task: task.task_id.encode("utf-8"))
    if len(selected) != 720:
        raise ValueError("candidate test task count is not exactly 720")
    ids = [task.task_id for task in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate test task IDs are not unique")
    return selected


def _capability_reason(task: MemUpdateTaskV3, capabilities: dict[str, bool]) -> dict[str, Any] | None:
    if len(task.queries) != 1:
        return {
            "reason_code": "single_query_required",
            "reason_kind": "manager_contract",
            "detail": "profile manager admits exactly one query",
            "observed_query_count": len(task.queries),
        }
    query = task.queries[0]
    query_type = query.query_type.value
    historical_query_types = {"previous", "point_in_time", "transition", "ordered_history"}
    historical_selector_kinds = {
        "previous",
        "exact_version",
        "event_anchor",
        "logical_time_anchor",
        "transition",
        "ordered_history",
    }
    requires_historical = query_type in historical_query_types or query.selector.kind in historical_selector_kinds
    requires_multi_object = query_type in {
        "multi_object_current",
        "update_sensitive_multi_hop",
        "multi_object_current_consistency",
    }
    if requires_historical and not capabilities.get("supports_historical_query", False):
        return {
            "reason_code": "historical_query_not_supported",
            "reason_kind": "manager_capability",
            "detail": "manager does not declare supports_historical_query",
        }
    if requires_multi_object and not capabilities.get("supports_multi_object_query", False):
        return {
            "reason_code": "multi_object_query_not_supported",
            "reason_kind": "manager_capability",
            "detail": "manager does not declare supports_multi_object_query",
        }
    if len(task.target_objects) != 1 and not capabilities.get("supports_multi_object_query", False):
        return {
            "reason_code": "single_target_object_required",
            "reason_kind": "manager_contract",
            "detail": "profile manager admits exactly one target object",
            "observed_target_object_count": len(task.target_objects),
        }
    if not requires_historical and not requires_multi_object:
        if query_type != "current" or query.selector.kind != "current" or len(query.target_object_keys) != 1 or query.evaluation_mode.value != "retrieved_prompt":
            return {
                "reason_code": "current_single_object_retrieved_prompt_required",
                "reason_kind": "query_contract",
                "detail": "profile manager supports only one current single-object retrieved-prompt query",
                "observed_query_type": query_type,
                "observed_selector_kind": query.selector.kind,
                "observed_query_target_count": len(query.target_object_keys),
                "observed_evaluation_mode": query.evaluation_mode.value,
            }
    for action in task.actions:
        operation = action.operation.value.lower()
        if not capabilities.get(f"supports_{operation}", False):
            return {
                "reason_code": "manager_capability_not_supported",
                "reason_kind": "manager_capability",
                "detail": f"manager does not declare supports_{operation}",
                "operation": operation,
            }
        if action.scope is not None and action.scope.value == "ttl" and not capabilities.get("supports_ttl", False):
            return {
                "reason_code": "ttl_delete_not_supported",
                "reason_kind": "manager_capability",
                "detail": "manager does not declare supports_ttl",
            }
        if action.scope is not None and action.scope.value == "object" and operation == "delete" and not capabilities.get("supports_scoped_delete", False):
            return {
                "reason_code": "scoped_delete_not_supported",
                "reason_kind": "manager_capability",
                "detail": "manager does not declare supports_scoped_delete",
            }
    return None


def task_support_reason(task: MemUpdateTaskV3, manager_kind: str) -> dict[str, Any] | None:
    try:
        spec = _MANAGER_SPECS[manager_kind]
    except KeyError as exc:
        raise ValueError(f"unknown manager kind: {manager_kind}") from exc
    return _capability_reason(task, spec["capabilities"])


def _extractor_spec(manager_kind: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_EXTRACTOR_SPECS[manager_kind])
    except KeyError as exc:
        raise ValueError(f"unknown manager kind: {manager_kind}") from exc


def _answer_spec(model_kind: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_ANSWER_SPECS[model_kind])
    except KeyError as exc:
        raise ValueError(f"unknown answer model kind: {model_kind}") from exc


def _cell_spec(cell_id: str) -> tuple[str, str]:
    try:
        return _CELL_SPECS[cell_id]
    except KeyError as exc:
        raise ValueError(f"unknown factorial cell ID: {cell_id}") from exc


def _task_view(tasks: list[MemUpdateTaskV3]) -> dict[str, Any]:
    ids = [task.task_id for task in tasks]
    return {
        "split": "test",
        "count": len(ids),
        "task_ids": ids,
        "task_ids_sha256": sha256_bytes(canonical_json_bytes(ids)),
        "source": "tasks.jsonl",
    }


def build_factorial_manifest(candidate_root: Path | str = CANDIDATE_ROOT, attestation_path: Path | str = AUDIT_ATTESTATION) -> dict[str, Any]:
    candidate = Path(candidate_root)
    provenance = validate_candidate(candidate, Path(attestation_path))
    tasks = select_test_tasks(candidate)
    post_parse_provenance = validate_candidate(candidate, Path(attestation_path))
    if (
        post_parse_provenance["release_index_sha256"] != provenance["release_index_sha256"]
        or post_parse_provenance["artifact_hashes"] != provenance["artifact_hashes"]
    ):
        raise ValueError("candidate changed while building factorial manifest")

    support_cache: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    cells = []
    for cell_id in CELL_IDS:
        manager_kind, answer_kind = _cell_spec(cell_id)
        manager = _MANAGER_SPECS[manager_kind]
        if manager_kind not in support_cache:
            supported: list[str] = []
            unsupported: list[dict[str, Any]] = []
            for task in tasks:
                reason = None if manager_kind == "reference" else task_support_reason(task, manager_kind)
                if reason is None:
                    supported.append(task.task_id)
                else:
                    unsupported.append({"task_id": task.task_id, **reason})
            support_cache[manager_kind] = (supported, unsupported)
        supported, unsupported = support_cache[manager_kind]
        cells.append({
            "cell_id": cell_id,
            "manager": {
                key: value for key, value in manager.items() if key != "capabilities"
            },
            "manager_id": manager["manager_id"],
            "extractor": _extractor_spec(manager_kind),
            "answer": _answer_spec(answer_kind),
            "capabilities_declared": manager["capabilities"],
            "supported_count": len(supported),
            "supported_task_ids": list(supported),
            "unsupported_count": len(unsupported),
            "unsupported_tasks": [dict(item) for item in unsupported],
            "evidence_boundary": "planning_only_support_matrix_no_runtime_accuracy_or_external_system_evidence",
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "main_track_v1",
        "planning_status": "PLANNING_ONLY_NO_EXECUTION",
        "execution_boundary": {
            "provider_calls": 0,
            "model_loads": 0,
            "database_accesses": 0,
            "network_calls": 0,
            "gpu_calls": 0,
            "executable_calls": 0,
            "remote_operations": 0,
        },
        "evidence_boundary": "factorial support planning only; no benchmark generation, state accuracy, retrieval accuracy, prompted-answer accuracy, or external-system evidence",
        "candidate_artifact_hashes": provenance["artifact_hashes"],
        "audit_attestation_sha256": provenance["audit_attestation_sha256"],
        "audit_review_status": provenance["audit_attestation"]["review_status"],
        "audit_evidence_class": provenance["audit_attestation"]["evidence_class"],
        "candidate": {
            "root": provenance["candidate_root"],
            "release_index_sha256": provenance["release_index_sha256"],
            "artifact_hashes": provenance["artifact_hashes"],
        },
        "audit_attestation": {
            "path": provenance["audit_attestation_path"],
            "sha256": provenance["audit_attestation_sha256"],
            "review_status": provenance["audit_attestation"]["review_status"],
            "benchmark_release_eligible": provenance["audit_attestation"]["benchmark_release_eligible"],
            "evidence_class": provenance["audit_attestation"]["evidence_class"],
            "completed_packet_sha256": provenance["audit_attestation"]["completed_packet_sha256"],
        },
        "task_view": _task_view(tasks),
        "cells": cells,
    }
    manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("release_id") != "main_track_v1" or manifest.get("planning_status") != "PLANNING_ONLY_NO_EXECUTION":
        raise ValueError("factorial manifest release or planning status is invalid")
    if manifest.get("candidate_artifact_hashes") != manifest.get("candidate", {}).get("artifact_hashes"):
        raise ValueError("factorial manifest candidate hashes are inconsistent")
    if manifest.get("audit_attestation_sha256") != manifest.get("audit_attestation", {}).get("sha256"):
        raise ValueError("factorial manifest audit hash is inconsistent")
    if manifest.get("audit_review_status") != "PASS" or manifest.get("audit_evidence_class") != "human_audit_completion_attestation":
        raise ValueError("factorial manifest audit binding is invalid")
    task_view = manifest.get("task_view")
    if type(task_view) is not dict or task_view.get("split") != "test" or task_view.get("count") != 720:
        raise ValueError("factorial manifest task view is invalid")
    task_ids = task_view.get("task_ids")
    if type(task_ids) is not list or len(task_ids) != 720 or len(set(task_ids)) != 720:
        raise ValueError("factorial manifest task IDs are invalid")
    if task_view.get("task_ids_sha256") != sha256_bytes(canonical_json_bytes(task_ids)):
        raise ValueError("factorial manifest task ID hash is invalid")
    cells = manifest.get("cells")
    if type(cells) is not list or tuple(cell.get("cell_id") for cell in cells) != CELL_IDS:
        raise ValueError("factorial manifest cell IDs are invalid")
    expected_ids = set(task_ids)
    for cell in cells:
        manager = cell.get("manager")
        if type(manager) is not dict or manager.get("manager_id") != cell.get("manager_id"):
            raise ValueError("factorial manifest manager_id binding is invalid")
        supported = cell.get("supported_task_ids")
        unsupported = cell.get("unsupported_tasks")
        if type(supported) is not list or type(unsupported) is not list:
            raise ValueError("factorial manifest support rows are invalid")
        unsupported_ids = [item.get("task_id") for item in unsupported if type(item) is dict]
        if len(unsupported_ids) != len(unsupported) or any(not item.get("reason_code") for item in unsupported):
            raise ValueError("factorial manifest unsupported rows require typed reasons")
        if set(supported) & set(unsupported_ids) or set(supported) | set(unsupported_ids) != expected_ids:
            raise ValueError("factorial manifest support partition is invalid")
        if cell.get("supported_count") != len(supported) or cell.get("unsupported_count") != len(unsupported):
            raise ValueError("factorial manifest support counts are invalid")
        if cell.get("manager_id") != "reference" and cell.get("unsupported_count") == 0:
            raise ValueError("profile cell cannot have unsupported count zero")
    payload = dict(manifest)
    declared = payload.pop("payload_sha256", None)
    if type(declared) is not str or declared != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("factorial manifest payload hash is invalid")


def _ensure_output_absent(output_path: Path | str) -> Path:
    output = Path(output_path)
    if not output.is_absolute():
        raise ValueError("output path must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    return output


def publish_factorial_manifest(manifest: dict[str, Any], output_path: Path | str) -> str:
    if type(manifest) is not dict or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("factorial manifest schema mismatch")
    _validate_manifest_shape(manifest)
    output = _ensure_output_absent(output_path)
    raw = canonical_json_bytes(manifest)
    publish_files_atomically({output: raw}, overwrite=False)
    return sha256_bytes(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan the audited main-track fixed-extractor/manager factorial.")
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--audit-attestation", type=Path, default=AUDIT_ATTESTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = _ensure_output_absent(args.output)
    manifest = build_factorial_manifest(args.candidate_root, args.audit_attestation)
    digest = publish_factorial_manifest(manifest, output)
    print(json.dumps({"status": "PASS", "output": str(output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
