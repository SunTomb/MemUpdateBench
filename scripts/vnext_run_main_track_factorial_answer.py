from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus
from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3
from mub.vnext.contracts.v3.common import object_identity, typed_json_equal
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.score import ScorerConfigV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.contracts.v3.version import (
    METRIC_REGISTRY_VERSION_V3,
    PRIMARY_FAILURE_PRECEDENCE_VERSION_V3,
    SCORER_VERSION_V3,
)
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.io import sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.runtime.answer_model_v3 import (
    ANSWER_MODEL_PARSER_VERSION_V3,
    parse_answer_prediction_v3,
    render_visible_prompt_v3,
)
from mub.vnext.scoring.scorer_v3 import _normalized as _canonical_normalized
from mub.vnext.scoring.scorer_v3 import _token_f1 as _canonical_token_f1
from scripts import vnext_plan_main_track_factorial as planner
from scripts.vnext_run_letta_qwen_extraction_canary import verify_model_provenance

SCHEMA_VERSION = "memupdatebench.main-track.factorial.answer-replay.v1"
ROW_SCHEMA_VERSION = SCHEMA_VERSION + ".row"
INDEX_SCHEMA_VERSION = SCHEMA_VERSION + ".artifact-index"
EVIDENCE_CLASS = "manager_fixture_answer_replay"
CANARY_EVIDENCE_CLASS = EVIDENCE_CLASS + "_canary"
BLOCKED_EVIDENCE_CLASS = EVIDENCE_CLASS + "_blocked"
TEST_ONLY_EVIDENCE_CLASS = EVIDENCE_CLASS + "_test_only"
RETRIEVAL_K = 16
EXECUTION_MODES = ("production", "injected_test_only")
MANAGER_FIXTURE_RECEIPT_SCHEMA = "memupdatebench.main-track.manager-fixture-execution-receipt.v1"
MANAGER_FIXTURE_SCHEMA = "memupdatebench.main-track.factorial.manager-fixture.v1"
MANAGER_FIXTURE_INDEX_SCHEMA = MANAGER_FIXTURE_SCHEMA + ".artifact-index"
PUBLIC_FORBIDDEN_FIELDS = frozenset(
    {
        "prompt",
        "raw_prompt",
        "rendered_prompt",
        "rendered_chat_prompt",
        "output",
        "raw_output",
        "generated_text",
        "reasoning",
        "reasoning_content",
        "raw_reasoning",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACTS = ("manager_rows.jsonl", "manager_summary.json", "artifact_index.json")
_OUTCOMES = (
    "CORRECT",
    "WRONG",
    "FORMAT_INVALID",
    "UNAVAILABLE",
    "CORRECT_ABSTENTION",
    "WRONG_ABSTENTION",
)
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FROZEN_IMMUTABLE_ROOTS = (
    ROOT / "data" / "vnext" / "core" / "v3",
    ROOT / "data" / "vnext" / "pilot",
)


SCORING_BINDING = {
    "parser_version": ANSWER_MODEL_PARSER_VERSION_V3,
    "scorer_version": SCORER_VERSION_V3,
    "metric_registry_version": METRIC_REGISTRY_VERSION_V3,
    "primary_failure_precedence_version": PRIMARY_FAILURE_PRECEDENCE_VERSION_V3,
    "value_normalization_profile": "scorer_v3_canonical",
    "null_policy": "serialize_null_exclude_from_aggregation",
    "scorer_configuration_hash": ScorerConfigV3().configuration_hash,
}


class AnswerReplayModel(Protocol):
    identity: Mapping[str, Any]
    last_answer_metadata: Mapping[str, Any]

    def answer(self, request: PromptedAnswerRequestV3) -> AnswerPredictionV3 | str: ...

    def close(self) -> None: ...


class AnswerExecutionError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _load_json(path: Path, label: str) -> tuple[Any, bytes]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if canonical_bytes(value) != raw:
        raise ValueError(f"{label} must use canonical JSON")
    return value, raw


def _forbidden_field(value: Any, location: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{location} has a non-string field name"
            lowered = key.casefold()
            if lowered.endswith("_sha256"):
                if child is None:
                    continue
                if key != lowered or type(child) is not str or _HEX64.fullmatch(child) is None:
                    return f"{location}.{key} is a raw prompt/output/reasoning or invalid hash field"
                continue
            if lowered in PUBLIC_FORBIDDEN_FIELDS or lowered.startswith(
                ("raw_prompt_", "raw_output_", "raw_reasoning_")
            ):
                return f"{location}.{key} is a raw prompt/output/reasoning field"
            found = _forbidden_field(child, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _forbidden_field(child, f"{location}[{index}]")
            if found:
                return found
    return None


def _absolute_path_location(value: Any, location: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            found = _absolute_path_location(child, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _absolute_path_location(child, f"{location}[{index}]")
            if found:
                return found
    elif type(value) is str and (
        _WINDOWS_PATH.match(value)
        or value.startswith("/")
        or value.startswith(("~/", "\\\\"))
    ):
        return f"{location} contains an absolute path"
    return None


def _validate_private_payload(value: Any) -> Any:
    found = _forbidden_field(value)
    if found:
        raise ValueError(found)
    if scan_for_secrets(value):
        raise ValueError("private fixture failed security scan")
    return value


def validate_public_payload(value: Any) -> Any:
    found = _forbidden_field(value)
    if found:
        raise ValueError(found)
    found = _absolute_path_location(value)
    if found:
        raise ValueError(found)
    if scan_for_secrets(value):
        raise ValueError("public artifact failed security scan")
    return value


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("manager_rows.jsonl must be a regular file")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n") or line[:-1] == b"":
            raise ValueError(f"manager_rows.jsonl line {number} must be canonical LF JSON")
        try:
            value = json.loads(line[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"manager_rows.jsonl line {number} is invalid JSON") from exc
        if type(value) is not dict or canonical_bytes(value) != line[:-1]:
            raise ValueError(f"manager_rows.jsonl line {number} must use canonical JSON")
        _validate_private_payload(value)
        rows.append(value)
    return rows, raw


def _artifact_metadata(index: Mapping[str, Any], name: str, raw: bytes, count: int) -> None:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "manager_rows.jsonl",
        "manager_summary.json",
    }:
        raise ValueError("manager fixture artifact_index artifacts are invalid")
    item = artifacts.get(name)
    if not isinstance(item, Mapping):
        raise ValueError(f"artifact_index is missing {name}")
    if (
        item.get("sha256") != sha256_bytes(raw)
        or item.get("bytes") != len(raw)
        or item.get("record_count") != count
    ):
        raise ValueError(f"artifact_index {name} hash/size/count mismatch")


def _fixture_scope(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    supported_count: int | None = None,
    execution_mode: str | None = None,
) -> tuple[str, bool, int]:
    scope = summary.get("scope")
    full_match = re.fullmatch(r"full([1-9][0-9]*)", scope or "")
    canary_match = re.fullmatch(r"canary([1-9][0-9]*)", scope or "")
    if full_match:
        selected = int(full_match.group(1))
        if supported_count is not None and selected != supported_count:
            raise ValueError("manager fixture full scope does not match manifest supported count")
    elif canary_match:
        selected = int(canary_match.group(1))
        if supported_count is not None and selected > supported_count:
            raise ValueError("manager fixture canary scope exceeds manifest supported count")
    else:
        raise ValueError("manager fixture scope is invalid")
    scientific = summary.get("scientific_evidence")
    if type(scientific) is not bool:
        raise ValueError("manager fixture scientific_evidence must be boolean")
    if summary.get("status") != "PASS":
        raise ValueError("manager fixture status must be PASS")
    if summary.get("terminal_rows") != len(rows) or summary.get("executed_supported_count") != selected:
        raise ValueError("manager fixture summary cardinality is inconsistent")
    if execution_mode is not None:
        if execution_mode not in EXECUTION_MODES:
            raise ValueError("execution_mode must be production or injected_test_only")
        observed_mode = summary.get("execution_mode")
        if observed_mode != execution_mode:
            raise ValueError("manager fixture execution mode binding mismatch")
        evidence_class = summary.get("evidence_class")
        if execution_mode == "production":
            expected_class = (
                "manager_state_retrieval_fixture"
                if full_match
                else "manager_state_retrieval_fixture_canary"
            )
            expected_scientific = bool(full_match)
            if scientific is not expected_scientific or evidence_class != expected_class:
                raise ValueError("production manager fixture evidence/scientific boundary is invalid")
        else:
            expected_class = (
                "manager_state_retrieval_fixture_test_only"
                if full_match
                else "manager_state_retrieval_fixture_test_only_canary"
            )
            if scientific is not False or evidence_class != expected_class:
                raise ValueError("injected_test_only fixture must be explicitly non-scientific test-only evidence")
    return scope, scientific, selected


def build_relocation_metadata(
    *,
    manifest_sha256: str,
    candidate_release_index_sha256: str,
    audit_attestation_sha256: str,
    allow_relocated_authenticated_inputs: bool,
    manager_fixture_root_digest: str | None = None,
) -> dict[str, Any]:
    result = {
        "enabled": bool(allow_relocated_authenticated_inputs),
        "candidate_release_id": "main_track_v1",
        "candidate_release_index_sha256": _require_sha(
            candidate_release_index_sha256, "candidate_release_index_sha256"
        ),
        "manifest_sha256": _require_sha(manifest_sha256, "manifest_sha256"),
        "audit_attestation_sha256": _require_sha(
            audit_attestation_sha256, "audit_attestation_sha256"
        ),
        "authenticated_equivalence": True,
    }
    if manager_fixture_root_digest is not None:
        result["manager_fixture_root_digest"] = _require_sha(
            manager_fixture_root_digest, "manager_fixture_root_digest"
        )
    validate_public_payload(result)
    return result


def _validate_manifest_and_candidate(
    manifest_path: Path,
    candidate_root: Path,
    audit_path: Path,
    *,
    allow_relocated_authenticated_inputs: bool,
) -> tuple[dict[str, Any], bytes, dict[str, Any], list[MemUpdateTaskV3], dict[str, Any]]:
    for path in (manifest_path, candidate_root, audit_path):
        assert_no_reparse_components(path)
    manifest, manifest_raw = _load_json(manifest_path, "factorial manifest")
    if type(manifest) is not dict or manifest.get("schema_version") != planner.SCHEMA_VERSION:
        raise ValueError("factorial manifest schema mismatch")
    planner._validate_manifest_shape(manifest)
    candidate_root = candidate_root.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    declared_candidate = Path(manifest["candidate"]["root"]).resolve(strict=False)
    declared_audit = Path(manifest["audit_attestation"]["path"]).resolve(strict=False)
    if not allow_relocated_authenticated_inputs and declared_candidate != candidate_root:
        raise ValueError("manifest candidate root binding does not match supplied path")
    if not allow_relocated_authenticated_inputs and declared_audit != audit_path:
        raise ValueError("manifest audit attestation path binding does not match supplied path")
    provenance = planner.validate_candidate(candidate_root, audit_path)
    manifest_sha = sha256_bytes(manifest_raw)
    if (
        manifest["candidate_artifact_hashes"] != provenance["artifact_hashes"]
        or manifest["candidate"]["release_index_sha256"] != provenance["release_index_sha256"]
    ):
        raise ValueError("manifest candidate hashes do not match candidate")
    if manifest["audit_attestation_sha256"] != provenance["audit_attestation_sha256"]:
        raise ValueError("manifest audit attestation hash does not match attestation")
    tasks = planner.select_test_tasks(candidate_root)
    if [task.task_id for task in tasks] != manifest["task_view"]["task_ids"]:
        raise ValueError("manifest task order does not match candidate test order")
    if sha256_bytes(canonical_bytes([task.task_id for task in tasks])) != manifest["task_view"]["task_ids_sha256"]:
        raise ValueError("manifest task IDs hash does not match candidate")
    relocation = build_relocation_metadata(
        manifest_sha256=manifest_sha,
        candidate_release_index_sha256=provenance["release_index_sha256"],
        audit_attestation_sha256=provenance["audit_attestation_sha256"],
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    return manifest, manifest_raw, provenance, tasks, relocation


def _load_fixture(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    root = Path(root)
    assert_no_reparse_components(root)
    if root.is_symlink():
        raise ValueError("manager fixture root must be a regular directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("manager fixture root must be a regular directory")
    for name in _ARTIFACTS:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"manager fixture {name} must be a regular file")
    rows, rows_raw = _read_rows(root / "manager_rows.jsonl")
    summary, summary_raw = _load_json(root / "manager_summary.json", "manager_summary.json")
    index, index_raw = _load_json(root / "artifact_index.json", "artifact_index.json")
    if type(summary) is not dict or type(index) is not dict:
        raise ValueError("manager fixture summary/index must be objects")
    _validate_private_payload(summary)
    _validate_private_payload(index)
    _artifact_metadata(index, "manager_rows.jsonl", rows_raw, len(rows))
    _artifact_metadata(index, "manager_summary.json", summary_raw, 1)
    if summary.get("schema_version") != MANAGER_FIXTURE_SCHEMA:
        raise ValueError("manager fixture summary schema mismatch")
    if index.get("schema_version") != MANAGER_FIXTURE_INDEX_SCHEMA:
        raise ValueError("manager fixture index schema mismatch")
    for key in ("cell_id", "manager_kind", "manifest_sha256", "evidence_class", "scientific_evidence"):
        if index.get(key) != summary.get(key):
            label = "manifest binding" if key == "manifest_sha256" else f"{key} binding"
            raise ValueError(f"manager fixture summary/index {label} mismatch")
    if summary.get("rows_sha256") != sha256_bytes(rows_raw):
        raise ValueError("manager fixture summary rows hash mismatch")
    if index.get("status") != summary.get("status") or index.get("scope") != summary.get("scope"):
        raise ValueError("manager fixture summary/index status or scope mismatch")
    return rows, summary, index, {
        "manager_rows.jsonl": sha256_bytes(rows_raw),
        "manager_summary.json": sha256_bytes(summary_raw),
        "artifact_index.json": sha256_bytes(index_raw),
    }


def _fixture_root_digest(fixture_hashes: Mapping[str, str]) -> str:
    return sha256_bytes(
        canonical_bytes({name: _require_sha(fixture_hashes.get(name), name) for name in _ARTIFACTS})
    )


def _validate_manager_fixture_attestation(
    path: str | Path | None,
    *,
    manager_fixture_attestation_sha256: str | None,
    fixture_root: Path,
    fixture_hashes: Mapping[str, str],
    summary: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest_sha256: str,
    candidate_hashes: Mapping[str, str],
    audit_sha: str,
) -> dict[str, Any]:
    if path is None:
        raise ValueError("production replay requires an authenticated manager fixture receipt")
    expected_receipt_sha = _require_sha(
        manager_fixture_attestation_sha256,
        "manager_fixture_attestation_sha256",
    )
    receipt_path = Path(path)
    assert_no_reparse_components(receipt_path)
    receipt_path = receipt_path.resolve(strict=True)
    if fixture_root in receipt_path.parents or receipt_path == fixture_root:
        raise ValueError("manager fixture receipt must be independently supplied")
    receipt, raw = _load_json(receipt_path, "manager fixture receipt")
    if sha256_bytes(raw) != expected_receipt_sha:
        raise ValueError("manager fixture receipt SHA-256 does not match caller-supplied digest")
    if type(receipt) is not dict:
        raise ValueError("manager fixture receipt must be an object")
    validate_public_payload(receipt)
    if receipt.get("schema_version") != MANAGER_FIXTURE_RECEIPT_SCHEMA:
        raise ValueError("manager fixture receipt schema mismatch")
    if receipt.get("status") != "PASS" or receipt.get("execution_mode") != "production":
        raise ValueError("manager fixture receipt must attest a production PASS")
    if receipt.get("evidence_class") != summary.get("evidence_class"):
        raise ValueError("manager fixture receipt evidence class mismatch")
    if receipt.get("attestation_status") != "PASS" or receipt.get("authentication_method") != "hash_bound_release_attestation":
        raise ValueError("manager fixture receipt is not hash-bound")
    if receipt.get("manifest_sha256") != manifest_sha256 or receipt.get("audit_attestation_sha256") != audit_sha:
        raise ValueError("manager fixture receipt manifest/audit binding mismatch")
    if receipt.get("candidate_artifact_hashes") != dict(candidate_hashes):
        raise ValueError("manager fixture receipt candidate binding mismatch")
    observed_artifacts = receipt.get("manager_fixture_artifact_hashes")
    if observed_artifacts != dict(fixture_hashes):
        raise ValueError("manager fixture receipt artifact binding mismatch")
    if receipt.get("manager_fixture_root_digest") != _fixture_root_digest(fixture_hashes):
        raise ValueError("manager fixture receipt root digest mismatch")
    if receipt.get("cell_id") != summary.get("cell_id") or receipt.get("scope") != summary.get("scope"):
        raise ValueError("manager fixture receipt scope binding mismatch")
    producer = receipt.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError("manager fixture receipt producer/runtime identity is incomplete")
    if any(
        type(producer.get(key)) is not str or not producer.get(key).strip()
        for key in ("producer_id", "producer_revision", "manager_id", "cell_id")
    ):
        raise ValueError("manager fixture receipt producer identity is incomplete")
    runtime_identity = producer.get("runtime_identity")
    if not (
        (type(runtime_identity) is str and runtime_identity.strip())
        or (isinstance(runtime_identity, Mapping) and runtime_identity)
    ):
        raise ValueError("manager fixture receipt runtime identity is incomplete")
    runtime_identity_value = (
        runtime_identity
        if type(runtime_identity) is str
        else sha256_bytes(canonical_bytes(runtime_identity))
    )
    for field, summary_field in (("manager_id", "manager_id"), ("cell_id", "cell_id")):
        if field in producer and producer[field] != summary.get(summary_field):
            raise ValueError(f"manager fixture receipt producer {field} binding mismatch")
        if field in receipt and receipt[field] != summary.get(summary_field):
            raise ValueError(f"manager fixture receipt {field} binding mismatch")
    for field, summary_field in (
        ("producer_manager_id", "manager_id"),
        ("producer_cell_id", "cell_id"),
    ):
        if field in receipt and receipt[field] != summary.get(summary_field):
            raise ValueError(f"manager fixture receipt {field} binding mismatch")
    return {
        "receipt_sha256": sha256_bytes(raw),
        "producer_id": producer["producer_id"],
        "producer_revision": producer["producer_revision"],
        "runtime_identity": runtime_identity_value,
    }


def _expected_cell(
    manifest: Mapping[str, Any],
    cell_id: str,
    *,
    tasks: Sequence[MemUpdateTaskV3] | None = None,
) -> Mapping[str, Any]:
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise ValueError("factorial manifest cells are invalid")
    cell = next(
        (item for item in cells if isinstance(item, Mapping) and item.get("cell_id") == cell_id),
        None,
    )
    if cell is None:
        raise ValueError("requested factorial cell is absent")
    _, model_kind = planner._cell_spec(cell_id)
    if cell.get("answer", {}).get("identity") != planner._answer_spec(model_kind)["identity"]:
        raise ValueError("manifest answer identity is invalid")
    if tasks is not None:
        planned_cell = _planned_cell(cell_id, tasks)
        if canonical_bytes(cell) != canonical_bytes(planned_cell):
            raise ValueError("requested factorial cell does not match deterministic planner")
    return cell


def _planned_cell(cell_id: str, tasks: Sequence[MemUpdateTaskV3]) -> dict[str, Any]:
    manager_kind, answer_kind = planner._cell_spec(cell_id)
    manager = planner._MANAGER_SPECS[manager_kind]
    supported: list[str] = []
    unsupported: list[dict[str, Any]] = []
    for task in tasks:
        reason = None if manager_kind == "reference" else planner.task_support_reason(task, manager_kind)
        if reason is None:
            supported.append(task.task_id)
        else:
            unsupported.append({"task_id": task.task_id, **reason})
    return {
        "cell_id": cell_id,
        "manager": {key: value for key, value in manager.items() if key != "capabilities"},
        "manager_id": manager["manager_id"],
        "extractor": planner._extractor_spec(manager_kind),
        "answer": planner._answer_spec(answer_kind),
        "capabilities_declared": manager["capabilities"],
        "supported_count": len(supported),
        "supported_task_ids": supported,
        "unsupported_count": len(unsupported),
        "unsupported_tasks": unsupported,
        "evidence_boundary": "planning_only_support_matrix_no_runtime_accuracy_or_external_system_evidence",
    }


def _validate_fixture_rows(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    tasks: Sequence[MemUpdateTaskV3],
    provenance: Mapping[str, Any],
    audit_sha: str,
    *,
    manifest_sha256: str | None = None,
    execution_mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    supported_ids = list(cell["supported_task_ids"])
    unsupported_items = list(cell["unsupported_tasks"])
    unsupported_ids = [item["task_id"] for item in unsupported_items]
    if len(set(supported_ids)) != len(supported_ids) or len(set(unsupported_ids)) != len(unsupported_ids):
        raise ValueError("manager fixture manifest membership contains duplicate task IDs")
    if set(supported_ids) & set(unsupported_ids) or set(supported_ids) | set(unsupported_ids) != {task.task_id for task in tasks}:
        raise ValueError("manager fixture manifest membership partition is invalid")
    scope, scientific, selected_count = _fixture_scope(
        summary, rows, len(supported_ids), execution_mode
    )
    expected_selected = supported_ids if selected_count == len(supported_ids) else supported_ids[:selected_count]
    expected_ids = [
        task.task_id
        for task in tasks
        if task.task_id in set(unsupported_ids) or task.task_id in set(expected_selected)
    ]
    if [row.get("task_id") for row in rows] != expected_ids:
        raise ValueError("manager fixture rows do not preserve exact selected supported/unsupported membership")
    expected_scope = f"full{len(supported_ids)}" if selected_count == len(supported_ids) else f"canary{selected_count}"
    for source, label in ((summary, "summary"), (index, "index")):
        if source.get("scope") != expected_scope:
            raise ValueError(f"manager fixture {label} scope does not match manifest membership")
        if source.get("selected_supported_task_ids") != expected_selected:
            raise ValueError(f"manager fixture {label} selected supported membership mismatch")
        if source.get("selected_supported_task_ids_sha256") != sha256_bytes(canonical_bytes(expected_selected)):
            raise ValueError(f"manager fixture {label} selected supported membership hash mismatch")
        if source.get("eligible_supported_count") != len(supported_ids):
            raise ValueError(f"manager fixture {label} eligible supported count mismatch")
        if source.get("not_requested_supported_count") != len(supported_ids) - selected_count:
            raise ValueError(f"manager fixture {label} not-requested supported count mismatch")
    if execution_mode is not None and index.get("execution_mode") != execution_mode:
        raise ValueError("manager fixture index execution mode binding mismatch")
    if index.get("scientific_evidence") != scientific:
        raise ValueError("manager fixture summary/index scientific boundary mismatch")
    if index.get("evidence_class") != summary.get("evidence_class"):
        raise ValueError("manager fixture summary/index evidence class mismatch")
    for key in ("cell_id", "manager_kind", "manifest_sha256", "scientific_evidence"):
        if index.get(key) != summary.get(key):
            raise ValueError(f"manager fixture summary/index {key} binding mismatch")
    if manifest_sha256 is not None:
        if summary.get("manifest_sha256") != manifest_sha256 or index.get("manifest_sha256") != manifest_sha256:
            raise ValueError("manager fixture manifest SHA binding mismatch")
    if summary.get("candidate_artifact_hashes") != provenance["artifact_hashes"]:
        raise ValueError("manager fixture summary candidate binding mismatch")
    if summary.get("audit_attestation_sha256") != audit_sha:
        raise ValueError("manager fixture summary audit binding mismatch")
    if index.get("candidate_artifact_hashes") != provenance["artifact_hashes"]:
        raise ValueError("manager fixture index candidate binding mismatch")
    if index.get("audit_attestation_sha256") != audit_sha:
        raise ValueError("manager fixture index audit binding mismatch")
    if summary.get("manager_kind") != planner._cell_spec(cell["cell_id"])[0] or summary.get("cell_id") != cell["cell_id"]:
        raise ValueError("manager fixture cell binding does not match requested cell")
    if len(rows) != len(expected_ids) or summary.get("requested_task_count") != len(expected_ids):
        raise ValueError("manager fixture row cardinality is invalid")
    tasks_by_id = {task.task_id: task for task in tasks}
    supported_rows = []
    for number, row in enumerate(rows, 1):
        task = tasks_by_id[row["task_id"]]
        if row.get("task_sha256") != sha256_model(task):
            raise ValueError(f"manager fixture row {number} task hash mismatch")
        if row.get("audit_attestation_sha256") not in (None, audit_sha):
            raise ValueError(f"manager fixture row {number} audit binding mismatch")
        if row.get("schema_version") != MANAGER_FIXTURE_SCHEMA + ".row":
            raise ValueError(f"manager fixture row {number} schema mismatch")
        expected_status = "UNSUPPORTED" if row["task_id"] in set(unsupported_ids) else "SUPPORTED"
        if row.get("status") != expected_status:
            raise ValueError(f"manager fixture row {number} status does not match manifest membership")
        if row.get("status") == "UNSUPPORTED":
            if row.get("execution_status") not in {"NOT_RUN", None}:
                raise ValueError(f"manager fixture row {number} unsupported execution status is invalid")
            unsupported = next(item for item in unsupported_items if item["task_id"] == row["task_id"])
            for key in ("reason_code", "reason_kind"):
                if type(row.get(key)) is not str or not row[key].strip():
                    raise ValueError(f"manager fixture row {number} unsupported {key} must be a nonblank string")
                if row[key] != unsupported[key]:
                    raise ValueError(f"manager fixture row {number} unsupported {key} does not match manifest")
            if "detail" in row:
                if type(row["detail"]) is not str or row["detail"] != unsupported.get("detail"):
                    raise ValueError(f"manager fixture row {number} unsupported detail does not match manifest")
            unsupported_null_fields = (
                "state_accuracy", "parsed_final_value", "final_memory_size", "stable_entry_id",
                "gold_retrieved", "retrieved_count", "retrieval_trace", "retrieval_trace_sha256",
                "answer_disposition", "answer_format_valid", "parsed_answer", "answer_outcome",
                "exact_match", "normalized_match", "typed_match", "typed_exact_match", "answer_f1",
                "visible_prompt_sha256", "answer_output_sha256",
            )
            if any(row.get(key) is not None for key in unsupported_null_fields):
                raise ValueError(f"manager fixture row {number} unsupported state/retrieval/answer metrics must be null")
            continue
        if row.get("status") != "SUPPORTED" or row.get("execution_status") != "PASS":
            raise ValueError(f"manager fixture row {number} is not a completed PASS row")
        if row.get("state") is None or row.get("retrieval") is None:
            raise ValueError(f"manager fixture row {number} lacks state/retrieval fixture")
        state = row["state"]
        state_pairs = {
            "state_accuracy": "state_accuracy",
            "parsed_final_value": "final_value",
            "final_memory_size": "final_memory_size",
            "stable_entry_id": "stable_entry_id",
        }
        if not isinstance(state, Mapping) or any(
            row.get(key) != state.get(nested) for key, nested in state_pairs.items()
        ):
            raise ValueError(f"manager fixture row {number} state fields drift")
        if type(state.get("state_accuracy")) is not bool:
            raise ValueError(f"manager fixture row {number} state_accuracy must be boolean")
        if type(state.get("stable_entry_id")) is not bool:
            raise ValueError(f"manager fixture row {number} stable_entry_id must be boolean")
        final_memory_size = state.get("final_memory_size")
        if type(final_memory_size) is not int or final_memory_size < 0:
            raise ValueError(f"manager fixture row {number} final_memory_size must be a nonnegative integer")
        if type(row.get("state_accuracy")) is not bool or type(row.get("stable_entry_id")) is not bool:
            raise ValueError(f"manager fixture row {number} state accuracy/entry flags must be boolean")
        retrieval = row["retrieval"]
        retrieval_pairs = {
            "retrieval_trace": "trace",
            "retrieval_trace_sha256": "trace_sha256",
            "gold_retrieved": "gold_retrieved",
        }
        if not isinstance(retrieval, Mapping) or any(
            row.get(key) != retrieval.get(nested) for key, nested in retrieval_pairs.items()
        ):
            raise ValueError(f"manager fixture row {number} retrieval fields drift")
        if type(retrieval.get("gold_retrieved")) is not bool or type(row.get("gold_retrieved")) is not bool:
            raise ValueError(f"manager fixture row {number} gold_retrieved must be boolean")
        trace = retrieval.get("trace")
        if not isinstance(trace, Mapping) or retrieval.get("trace_sha256") != sha256_bytes(canonical_bytes(trace)):
            raise ValueError(f"manager fixture row {number} retrieval trace hash mismatch")
        if retrieval.get("retrieved_count") != trace.get("retrieved_count"):
            raise ValueError(f"manager fixture row {number} retrieved_count binding mismatch")
        reconstructed = reconstruct_retrieval_trace(row, task.queries[0])
        _validate_retrieval_source_event_bindings(reconstructed, task.events)
        gold_evidence = task.gold_evidence[0]
        expected_state_accuracy = (
            state.get("stable_entry_id") is True
            and typed_json_equal(state.get("final_value"), gold_evidence.answer)
        )
        if state.get("state_accuracy") is not expected_state_accuracy:
            raise ValueError(f"manager fixture row {number} state accuracy does not match final value and gold")
        target_identities = {object_identity(key) for key in task.queries[0].target_object_keys}
        expected_gold_retrieved = any(
            object_identity(entry.object_key_candidate) in target_identities
            and typed_json_equal(entry.value_candidate, gold_evidence.answer)
            for entry in reconstructed.retrieved_entries
        )
        if retrieval.get("gold_retrieved") is not expected_gold_retrieved:
            raise ValueError(f"manager fixture row {number} gold retrieved flag does not match trace and gold")
        supported_rows.append(row)
    if summary.get("supported") is not None and summary.get("supported") != len(supported_rows):
        raise ValueError("manager fixture summary supported count mismatch")
    if summary.get("unsupported") is not None and summary.get("unsupported") != len(rows) - len(supported_rows):
        raise ValueError("manager fixture summary unsupported count mismatch")
    if summary.get("failed") is not None and summary.get("failed") != 0:
        raise ValueError("manager fixture summary reports failed rows")
    if summary.get("state_accuracy_denominator") is not None:
        state_values = [row["state"]["state_accuracy"] for row in supported_rows if row["state"].get("state_accuracy") is not None]
        if summary["state_accuracy_denominator"] != len(state_values) or (
            state_values and not math.isclose(float(summary.get("state_accuracy")), sum(bool(value) for value in state_values) / len(state_values), abs_tol=1e-12)
        ):
            raise ValueError("manager fixture state aggregate mismatch")
    if summary.get("retrieval_denominator") is not None:
        retrieval_values = [row["retrieval"]["gold_retrieved"] for row in supported_rows if row["retrieval"].get("gold_retrieved") is not None]
        if summary["retrieval_denominator"] != len(retrieval_values) or (
            retrieval_values and not math.isclose(float(summary.get("gold_retrieval_rate")), sum(bool(value) for value in retrieval_values) / len(retrieval_values), abs_tol=1e-12)
        ):
            raise ValueError("manager fixture retrieval aggregate mismatch")
    return [dict(row) for row in rows], {
        "scope": scope,
        "scientific_evidence": scientific,
        "selected_supported_task_ids": expected_selected,
    }


def _task_event_ids(task_events: Sequence[Any]) -> set[str]:
    event_ids: set[str] = set()
    for event in task_events:
        event_id = getattr(event, "event_id", None)
        if type(event_id) is not str or not event_id:
            raise ValueError("current task event IDs are invalid")
        event_ids.add(event_id)
    return event_ids


def _entry_from_fixture(
    value: Mapping[str, Any], *, task_event_ids: set[str] | None = None
) -> MemoryEntryRecordV3:
    required = {
        "entry_id",
        "object_key",
        "value",
        "content",
        "source_event_ids",
        "score",
        "rank",
        "version_metadata",
    }
    if set(value) != required:
        raise ValueError("manager fixture retrieval entry shape is invalid")
    if type(value["score"]) not in (int, float) or isinstance(value["score"], bool) or not math.isfinite(float(value["score"])):
        raise ValueError("manager fixture retrieval score is invalid")
    if type(value["rank"]) is not int or value["rank"] < 1:
        raise ValueError("manager fixture retrieval rank is invalid")
    from mub.vnext.contracts.v3.common import MemoryObjectKeyV3

    key = MemoryObjectKeyV3.model_validate(value["object_key"], strict=True)
    if not isinstance(value["source_event_ids"], (list, tuple)) or any(type(item) is not str for item in value["source_event_ids"]):
        raise ValueError("manager fixture retrieval source event IDs are invalid")
    source_event_ids = tuple(value["source_event_ids"])
    if task_event_ids is not None and any(item not in task_event_ids for item in source_event_ids):
        raise ValueError("manager fixture retrieval source event IDs are not bound to current task events")
    if not isinstance(value["version_metadata"], Mapping):
        raise ValueError("manager fixture retrieval version metadata is invalid")
    return MemoryEntryRecordV3(
        entry_id=value["entry_id"],
        content=value["content"],
        object_key_candidate=key,
        value_candidate=value["value"],
        source_event_ids=source_event_ids,
        version_index=(
            value["version_metadata"].get("version_index")
            if type(value["version_metadata"].get("version_index")) is int
            else None
        ),
        raw_metadata=dict(value["version_metadata"]),
    )


def _validate_retrieval_source_event_bindings(
    trace: RetrievalTraceV3, task_events: Sequence[Any]
) -> None:
    event_ids = _task_event_ids(task_events)
    for entry in trace.retrieved_entries:
        if any(event_id not in event_ids for event_id in entry.source_event_ids):
            raise ValueError(
                "manager fixture retrieval source event IDs are not bound to current task events"
            )


def reconstruct_retrieval_trace(
    row: Mapping[str, Any],
    query: Any,
    *,
    task_events: Sequence[Any] | None = None,
) -> RetrievalTraceV3:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, Mapping):
        retrieval = {
            "trace": row.get("retrieval_trace"),
            "trace_sha256": row.get("retrieval_trace_sha256"),
        }
    trace = retrieval.get("trace")
    if not isinstance(trace, Mapping) or retrieval.get("trace_sha256") != sha256_bytes(canonical_bytes(trace)):
        raise ValueError("manager fixture retrieval trace is not hash-bound")
    if trace.get("query_id") != query.query_id:
        raise ValueError("manager fixture retrieval query binding mismatch")
    expected_targets = [list(object_identity(key)) for key in query.target_object_keys]
    if trace.get("query_target_object_identities") != expected_targets:
        raise ValueError("manager fixture retrieval target object binding mismatch")
    if trace.get("retrieval_policy") != "normal_topk" or trace.get("retrieval_k") != RETRIEVAL_K:
        raise ValueError("manager fixture retrieval policy or retrieval_k mismatch")
    entries = trace.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manager fixture retrieval entries must be ordered")
    for location, value in (
        ("trace.retrieved_count", trace.get("retrieved_count")),
        ("retrieval.retrieved_count", retrieval.get("retrieved_count")),
    ):
        if type(value) is not int or value < 0 or value != len(entries):
            raise ValueError(f"manager fixture retrieval {location} mismatch")
    if len(entries) > RETRIEVAL_K:
        raise ValueError("manager fixture retrieval exceeds k=16")
    task_event_ids = None if task_events is None else _task_event_ids(task_events)
    ordered_entries = tuple(
        _entry_from_fixture(item, task_event_ids=task_event_ids) for item in entries
    )
    entry_ids = [entry.entry_id for entry in ordered_entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("manager fixture retrieval entries contain duplicate entry_id")
    scores = tuple(float(item["score"]) for item in entries)
    ranks = tuple(item["rank"] for item in entries)
    if len(scores) != len(ordered_entries) or len(ranks) != len(ordered_entries):
        raise ValueError("manager fixture retrieval scores/ranks are not aligned with ordered entries")
    if ranks != tuple(range(1, len(entries) + 1)):
        raise ValueError("manager fixture retrieval ranks do not preserve order")
    if type(trace.get("context_order")) is not str or not trace["context_order"]:
        raise ValueError("manager fixture retrieval context order is missing")
    if not isinstance(trace.get("version_metadata"), Mapping):
        raise ValueError("manager fixture retrieval version metadata is missing")
    return RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=ordered_entries,
        scores=scores,
        ranks=ranks,
        retrieval_policy="normal_topk",
        context_order=trace["context_order"],
        version_metadata=dict(trace["version_metadata"]),
    )


def _identity(
    model: Any,
    expected: Mapping[str, Any],
    *,
    execution_mode: str = "production",
    model_kind: str | None = None,
    required_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = getattr(model, "identity", None)
    if not isinstance(observed, Mapping) or not observed:
        raise ValueError("answer model identity is missing")
    result = dict(observed)
    validate_public_payload(result)
    if execution_mode in EXECUTION_MODES:
        for key, expected_value in expected.items():
            if result.get(key) != expected_value:
                raise ValueError(f"answer model identity mismatch for {key}")
        if required_binding is not None:
            for key, expected_value in required_binding.items():
                if result.get(key) != expected_value:
                    raise ValueError(f"{model_kind or 'answer model'} identity mismatch for {key}")
        if execution_mode == "production" and model_kind == "muse":
            for key in ("endpoint", "response_model", "runtime_identity", "runtime_receipt_sha256"):
                if key not in result or result[key] in (None, ""):
                    raise ValueError(f"Muse answer model identity missing {key}")
    elif execution_mode != "injected_test_only":
        raise ValueError("execution_mode must be production or injected_test_only")
    return result


def _factory_is_production_bound(factory: Any) -> bool:
    return bool(
        getattr(factory, "production_bound", False)
        or getattr(factory, "execution_mode", None) == "production"
    )


class _ProvenanceBoundModel:
    def __init__(self, model: Any, identity: Mapping[str, Any]):
        self._model = model
        self._identity = dict(identity)
        self.last_answer_metadata = getattr(model, "last_answer_metadata", {})

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def load(self) -> None:
        self._model.load()

    def answer(self, request: PromptedAnswerRequestV3) -> AnswerPredictionV3 | str:
        value = self._model.answer(request)
        self.last_answer_metadata = getattr(self._model, "last_answer_metadata", {})
        return value

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()


def _qwen_public_identity(
    expected: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    adapter_source_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        **dict(expected),
        "snapshot_release": f"{provenance['repo']}@{provenance['revision']}",
        "runtime_receipt_sha256": provenance["runtime_receipt_sha256"],
        "snapshot_binding_receipt_sha256": provenance["snapshot_binding_receipt_sha256"],
        "snapshot_binding_payload_sha256": provenance["snapshot_binding_payload_sha256"],
    }
    if adapter_source_sha256 is not None:
        identity["qwen_adapter_source_sha256"] = _require_sha(
            adapter_source_sha256,
            "qwen_adapter_source_sha256",
        )
    return identity


def _build_qwen_production_factory(
    model_snapshot: Path,
    provenance: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    adapter_source_sha256: str | None = None,
) -> Callable[[], AnswerReplayModel]:
    from scripts.vnext_run_letta_qwen_prompted_answer import QwenSession

    native_identity = _qwen_public_identity(expected_identity, provenance)
    output_identity = _qwen_public_identity(
        expected_identity,
        provenance,
        adapter_source_sha256=adapter_source_sha256,
    )

    def factory() -> AnswerReplayModel:
        model = QwenSession(model_snapshot)
        observed = getattr(model, "identity", None)
        if observed is not None:
            if not isinstance(observed, Mapping):
                raise ValueError("Qwen answer model identity is invalid")
            for key, expected_value in native_identity.items():
                if observed.get(key) != expected_value:
                    raise ValueError(f"Qwen answer model identity mismatch for {key}")
        return _ProvenanceBoundModel(model, output_identity)

    factory.production_bound = True  # type: ignore[attr-defined]
    return factory


def _safe_model_metadata(model: Any, raw_output: str) -> dict[str, Any]:
    metadata = getattr(model, "last_answer_metadata", {})
    safe: dict[str, Any] = {"answer_output_sha256": sha256_bytes(raw_output.encode("utf-8"))}
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            key = str(key)
            if key.casefold() in PUBLIC_FORBIDDEN_FIELDS or key.casefold().startswith(("raw_", "rendered_")):
                continue
            if key.endswith("_sha256") and type(value) is str and _HEX64.fullmatch(value):
                safe[key] = value
            elif key in {"generated_tokens", "prompt_tokens", "total_tokens", "latency_ms"} and type(value) in (int, float) and math.isfinite(float(value)) and float(value) >= 0:
                safe[key] = value
            elif key == "finish_reason" and value in _FINISH_REASONS:
                safe[key] = value
    validate_public_payload(safe)
    return safe


def _validate_muse_response_binding(model: Any, model_identity: Mapping[str, Any]) -> None:
    if "response_model" not in model_identity:
        return
    metadata = getattr(model, "last_answer_metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Muse answer response identity is missing")
    response_model = metadata.get("response_model", metadata.get("served_model"))
    if response_model != model_identity["response_model"]:
        raise ValueError("Muse answer response model identity mismatch")
    endpoint = metadata.get("endpoint", metadata.get("server_url"))
    expected_endpoint = model_identity.get("endpoint")
    expected_server = model_identity.get("server_url")
    if endpoint not in {expected_endpoint, expected_server, f"{expected_server}{expected_endpoint}"}:
        raise ValueError("Muse answer endpoint identity mismatch")


def _answer_prediction(model: Any, request: PromptedAnswerRequestV3) -> tuple[AnswerPredictionV3, str]:
    try:
        value = model.answer(request)
    except Exception as exc:
        raise AnswerExecutionError("answer model execution failed") from exc
    if isinstance(value, AnswerPredictionV3):
        if value.query_id != request.query.query_id:
            raise ValueError("answer prediction query_id does not match request")
        raw = value.raw_output
    elif isinstance(value, str):
        raw = value
    else:
        raise AnswerExecutionError("answer model must return AnswerPredictionV3 or JSON text")
    if type(raw) is not str:
        raise AnswerExecutionError("answer model raw output must be text")
    if scan_for_secrets(raw):
        raise ValueError("answer model output failed security scan")
    try:
        parsed_prediction = parse_answer_prediction_v3(
            query_id=request.query.query_id,
            answer_schema=request.query.answer_schema,
            raw_output=raw,
        )
    except Exception as exc:
        raise AnswerExecutionError("answer output parsing failed") from exc
    if isinstance(value, AnswerPredictionV3):
        if (
            value.disposition is not parsed_prediction.disposition
            or value.format_valid is not parsed_prediction.format_valid
            or value.error_flags != parsed_prediction.error_flags
            or not typed_json_equal(value.parsed_answer, parsed_prediction.parsed_answer)
        ):
            raise ValueError("AnswerPredictionV3 raw output is inconsistent with prediction fields")
        return value, raw
    return parsed_prediction, raw


def score_prediction(query: Any, prediction: AnswerPredictionV3, gold: Any) -> dict[str, Any]:
    expected = gold.disposition or AnswerDisposition.ANSWERED
    if expected is AnswerDisposition.ABSTAINED:
        correct = prediction.disposition is AnswerDisposition.ABSTAINED and prediction.format_valid
        outcome = "CORRECT_ABSTENTION" if correct else "WRONG_ABSTENTION"
        result = {
            "expected_disposition": expected.value,
            "answer_outcome": outcome,
            "exact_match": correct,
            "normalized_match": correct,
            "typed_match": correct,
            "typed_exact_match": correct,
            "answer_f1": 1.0 if correct else 0.0,
        }
    elif prediction.disposition is not AnswerDisposition.ANSWERED:
        result = {
            "expected_disposition": expected.value,
            "answer_outcome": "UNAVAILABLE",
            "exact_match": False,
            "normalized_match": False,
            "typed_match": False,
            "typed_exact_match": False,
            "answer_f1": 0.0,
        }
    elif not prediction.format_valid:
        result = {
            "expected_disposition": expected.value,
            "answer_outcome": "FORMAT_INVALID",
            "exact_match": False,
            "normalized_match": False,
            "typed_match": False,
            "typed_exact_match": False,
            "answer_f1": 0.0,
        }
    else:
        typed = typed_json_equal(prediction.parsed_answer, gold.answer)
        normalized = typed_json_equal(
            _canonical_normalized(prediction.parsed_answer),
            _canonical_normalized(gold.answer),
        )
        result = {
            "expected_disposition": expected.value,
            "answer_outcome": "CORRECT" if typed else "WRONG",
            "exact_match": typed,
            "normalized_match": normalized,
            "typed_match": typed,
            "typed_exact_match": typed,
            "answer_f1": _canonical_token_f1(prediction.parsed_answer, gold.answer),
        }
    result.update(SCORING_BINDING)
    return result


def _task_fields(task: MemUpdateTaskV3) -> dict[str, Any]:
    return {
        "core_id": task.metadata.split_key.semantic_core_id,
        "semantic_core_id": task.metadata.split_key.semantic_core_id,
        "family": task.task_family,
        "domain": task.metadata.extra.get("domain"),
        "attribute": task.metadata.extra.get("attribute"),
        "language": task.metadata.extra.get("language"),
        "split": task.metadata.split.value,
    }


def _fixture_public_binding(fixture_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "root_digest": _fixture_root_digest(fixture_hashes),
        "artifact_hashes": dict(fixture_hashes),
    }


def _build_row(
    task: MemUpdateTaskV3,
    fixture: Mapping[str, Any],
    model: Any,
    model_identity: Mapping[str, Any],
    *,
    candidate_hashes: Mapping[str, str],
    audit_sha: str,
    fixture_hashes: Mapping[str, str],
    cell_id: str,
    relocation: Mapping[str, Any],
    execution_mode: str,
    fixture_authentication: Mapping[str, Any] | None,
) -> dict[str, Any]:
    task_fields = _task_fields(task)
    common = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "task_id": task.task_id,
        **task_fields,
        "task_sha256": sha256_model(task),
        "cell_id": cell_id,
        "candidate_artifact_hashes": dict(candidate_hashes),
        "audit_attestation_sha256": audit_sha,
        "model_binding": dict(model_identity),
        "manager_fixture_binding": _fixture_public_binding(fixture_hashes),
        "input_relocation": dict(relocation),
        "scoring_binding": dict(SCORING_BINDING),
        "execution_mode": execution_mode,
        "manager_fixture_authentication": None if fixture_authentication is None else dict(fixture_authentication),
    }
    if fixture.get("status") == "UNSUPPORTED":
        row = {
            **common,
            "status": "UNSUPPORTED",
            "completion_status": "not_run",
            "execution_status": "NOT_RUN",
            "reason_code": fixture.get("reason_code"),
            "reason_kind": fixture.get("reason_kind"),
            "detail": fixture.get("detail"),
            "state_accuracy": None,
            "gold_retrieved": None,
            "answer_disposition": None,
            "answer_format_valid": None,
            "parsed_answer": None,
            "answer_outcome": None,
            "exact_match": None,
            "normalized_match": None,
            "typed_match": None,
            "typed_exact_match": None,
            "answer_f1": None,
            "retrieval_trace_sha256": None,
            "retrieved_count": None,
            "visible_prompt_sha256": None,
            "answer_output_sha256": None,
            "answer_model_metadata": {},
        }
        validate_public_payload(row)
        return row
    query = task.queries[0]
    trace = reconstruct_retrieval_trace(fixture, query)
    visible = render_visible_prompt_v3(query=query, retrieval_trace=trace)
    prompt_hash = sha256_bytes(visible.encode("utf-8"))
    bound_trace = trace.model_copy(update={"prompt_hash": prompt_hash})
    request = PromptedAnswerRequestV3(
        query=query,
        retrieval_trace=bound_trace,
        rendered_prompt=visible,
        prompt_hash=prompt_hash,
    )
    prediction, raw = _answer_prediction(model, request)
    _validate_muse_response_binding(model, model_identity)
    scored = score_prediction(query, prediction, task.gold_evidence[0])
    state = fixture["state"]
    retrieval = fixture["retrieval"]
    row = {
        **common,
        "status": "PASS",
        "completion_status": CompletionStatus.COMPLETED.value,
        "execution_status": "PASS",
        "state_accuracy": state.get("state_accuracy"),
        "final_memory_size": state.get("final_memory_size"),
        "gold_retrieved": retrieval.get("gold_retrieved"),
        "retrieved_count": retrieval.get("retrieved_count"),
        "retrieval_trace_sha256": sha256_bytes(canonical_bytes(fixture["retrieval"]["trace"])),
        "visible_prompt_sha256": prompt_hash,
        "answer_disposition": prediction.disposition.value,
        "answer_format_valid": prediction.format_valid,
        "parsed_answer": prediction.parsed_answer,
        "answer_error_flags": list(prediction.error_flags),
        "answer_output_sha256": sha256_bytes(raw.encode("utf-8")),
        "answer_model_metadata": _safe_model_metadata(model, raw),
        **scored,
    }
    validate_public_payload(row)
    return row


def _failed_row(
    task: MemUpdateTaskV3,
    fixture: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    *,
    candidate_hashes: Mapping[str, str],
    audit_sha: str,
    fixture_hashes: Mapping[str, str],
    cell_id: str,
    relocation: Mapping[str, Any],
    execution_mode: str,
    fixture_authentication: Mapping[str, Any] | None,
    error: BaseException,
) -> dict[str, Any]:
    row = {
        **_task_fields(task),
        "row_schema_version": ROW_SCHEMA_VERSION,
        "task_id": task.task_id,
        "task_sha256": sha256_model(task),
        "cell_id": cell_id,
        "status": "FAIL",
        "completion_status": "failed",
        "execution_status": "FAIL",
        "state_accuracy": fixture.get("state", {}).get("state_accuracy"),
        "gold_retrieved": fixture.get("retrieval", {}).get("gold_retrieved"),
        "retrieved_count": fixture.get("retrieval", {}).get("retrieved_count"),
        "answer_disposition": None,
        "answer_format_valid": None,
        "parsed_answer": None,
        "answer_error_flags": [],
        "answer_outcome": None,
        "exact_match": None,
        "normalized_match": None,
        "typed_match": None,
        "typed_exact_match": None,
        "answer_f1": None,
        "retrieval_trace_sha256": None,
        "visible_prompt_sha256": None,
        "answer_output_sha256": None,
        "answer_model_metadata": {},
        "error_type": type(error).__name__,
        "error_sha256": sha256_bytes(type(error).__name__.encode("utf-8")),
        "candidate_artifact_hashes": dict(candidate_hashes),
        "audit_attestation_sha256": audit_sha,
        "model_binding": dict(model_identity),
        "manager_fixture_binding": _fixture_public_binding(fixture_hashes),
        "input_relocation": dict(relocation),
        "scoring_binding": dict(SCORING_BINDING),
        "execution_mode": execution_mode,
        "manager_fixture_authentication": None if fixture_authentication is None else dict(fixture_authentication),
    }
    validate_public_payload(row)
    return row


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    fixture_summary: Mapping[str, Any],
    *,
    scope: str,
    scientific_evidence: bool,
    model_identity: Mapping[str, Any],
    candidate_hashes: Mapping[str, str],
    audit_sha: str,
    fixture_hashes: Mapping[str, str],
    relocation: Mapping[str, Any] | None = None,
    execution_mode: str = "injected_test_only",
    fixture_authentication: Mapping[str, Any] | None = None,
    fixture_root: Path | None = None,
    manifest_sha256: str | None = None,
    eligible_supported_count: int | None = None,
    selected_supported_task_ids: Sequence[str] | None = None,
    unsupported_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    relocation = {} if relocation is None else dict(relocation)
    supported = [row for row in rows if row.get("status") in {"PASS", "FAIL"}]
    failed = [row for row in supported if row.get("status") == "FAIL"]
    answers = [row for row in supported if row.get("answer_outcome") is not None]
    eligible_count = (
        eligible_supported_count
        if eligible_supported_count is not None
        else fixture_summary.get("eligible_supported_count")
    )
    selected_ids = list(
        selected_supported_task_ids
        if selected_supported_task_ids is not None
        else fixture_summary.get("selected_supported_task_ids", ())
    )
    unsupported_ids = list(
        unsupported_task_ids
        if unsupported_task_ids is not None
        else [row["task_id"] for row in rows if row.get("status") == "UNSUPPORTED"]
    )
    selected_ids_sha = sha256_bytes(canonical_bytes(selected_ids))
    unsupported_ids_sha = sha256_bytes(canonical_bytes(unsupported_ids))
    metrics = {name: _mean(answers, name) for name in ("exact_match", "normalized_match", "typed_match", "typed_exact_match", "answer_f1")}
    outcomes = {name: sum(row.get("answer_outcome") == name for row in answers) for name in _OUTCOMES}
    status = "PASS" if not failed and len(supported) == fixture_summary.get("executed_supported_count") else "FAIL"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scope": scope,
        "evidence_class": (
            BLOCKED_EVIDENCE_CLASS
            if status != "PASS"
            else (
                CANARY_EVIDENCE_CLASS
                if scope.startswith("canary")
                else (EVIDENCE_CLASS if execution_mode == "production" else TEST_ONLY_EVIDENCE_CLASS)
            )
        ),
        "scientific_evidence": bool(scientific_evidence and execution_mode == "production" and status == "PASS"),
        "execution_mode": execution_mode,
        "manifest_sha256": manifest_sha256,
        "eligible_supported_count": eligible_count,
        "executed_supported_count": len(supported),
        "not_requested_supported_count": (
            eligible_count - len(supported)
            if type(eligible_count) is int
            else None
        ),
        "selected_supported_task_ids": selected_ids,
        "selected_supported_task_ids_sha256": selected_ids_sha,
        "unsupported_count": len(rows) - len(supported),
        "unsupported_task_ids_sha256": unsupported_ids_sha,
        "rows": len(rows),
        "supported": len(supported),
        "failed": len(failed),
        "unsupported": len(rows) - len(supported),
        "attempted_answer_denominator": len(supported),
        "evaluable_answer_denominator": len(answers),
        "answer_metrics": metrics,
        "answer_outcome_counts": outcomes,
        "answer_em": metrics["exact_match"],
        "answer_f1": metrics["answer_f1"],
        "inherited_state_accuracy": fixture_summary.get("state_accuracy"),
        "inherited_state_accuracy_denominator": fixture_summary.get("state_accuracy_denominator"),
        "inherited_gold_retrieval_rate": fixture_summary.get("gold_retrieval_rate"),
        "inherited_retrieval_denominator": fixture_summary.get("retrieval_denominator"),
        "candidate_artifact_hashes": dict(candidate_hashes),
        "audit_attestation_sha256": audit_sha,
        "model_binding": dict(model_identity),
        "manager_fixture_binding": _fixture_public_binding(fixture_hashes),
        "manager_fixture_authentication": None if fixture_authentication is None else dict(fixture_authentication),
        "input_relocation": dict(relocation),
        "scoring_binding": dict(SCORING_BINDING),
        "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "rows_sha256": None,
    }
    validate_public_payload(summary)
    return summary


def _validate_staged_public_artifact(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise ValueError("staged answer artifact bytes changed")
    if path.name.startswith("answer_rows.jsonl"):
        _read_rows(path)
    else:
        value, _ = _load_json(path, path.name)
        validate_public_payload(value)


def _publish(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    source_paths: Sequence[Path],
    pre_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    output_preexisted = output.exists() or output.is_symlink()
    if output_preexisted:
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("output_root parent directory does not exist")
    for row in rows:
        validate_public_payload(row)
    validate_public_payload(summary)
    rows_raw = _canonical_jsonl(rows)
    bound_summary = dict(summary)
    bound_summary["rows_sha256"] = sha256_bytes(rows_raw)
    validate_public_payload(bound_summary)
    summary_raw = canonical_bytes(bound_summary)
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": bound_summary["status"],
        "scope": bound_summary["scope"],
        "evidence_class": bound_summary["evidence_class"],
        "scientific_evidence": bound_summary["scientific_evidence"],
        "execution_mode": bound_summary["execution_mode"],
        "manifest_sha256": bound_summary["manifest_sha256"],
        "eligible_supported_count": bound_summary["eligible_supported_count"],
        "executed_supported_count": bound_summary["executed_supported_count"],
        "not_requested_supported_count": bound_summary["not_requested_supported_count"],
        "selected_supported_task_ids": bound_summary["selected_supported_task_ids"],
        "selected_supported_task_ids_sha256": bound_summary["selected_supported_task_ids_sha256"],
        "unsupported_count": bound_summary["unsupported_count"],
        "unsupported_task_ids_sha256": bound_summary["unsupported_task_ids_sha256"],
        "rows_sha256": bound_summary["rows_sha256"],
        "manager_fixture_binding": bound_summary["manager_fixture_binding"],
        "manager_fixture_authentication": bound_summary["manager_fixture_authentication"],
        "candidate_artifact_hashes": bound_summary["candidate_artifact_hashes"],
        "audit_attestation_sha256": bound_summary["audit_attestation_sha256"],
        "input_relocation": bound_summary["input_relocation"],
        "model_binding": bound_summary["model_binding"],
        "scoring_binding": bound_summary["scoring_binding"],
        "runner_source_sha256": bound_summary["runner_source_sha256"],
        "artifacts": {
            "answer_rows.jsonl": {
                "sha256": sha256_bytes(rows_raw),
                "bytes": len(rows_raw),
                "record_count": len(rows),
            },
            "answer_summary.json": {
                "sha256": sha256_bytes(summary_raw),
                "bytes": len(summary_raw),
                "record_count": 1,
            },
        },
    }
    validate_public_payload(index)
    index_raw = canonical_bytes(index)
    destinations = {
        output / "answer_rows.jsonl": rows_raw,
        output / "answer_summary.json": summary_raw,
        output / "artifact_index.json": index_raw,
    }
    validators = {
        path: (lambda staged, expected=payload: _validate_staged_public_artifact(staged, expected))
        for path, payload in destinations.items()
    }

    def guard() -> None:
        if pre_publish is not None:
            pre_publish()

    try:
        publish_files_atomically(
            destinations,
            overwrite=False,
            source_paths=tuple(source_paths),
            validators=validators,
            pre_publish=guard,
        )
    except BaseException:
        if (
            not output_preexisted
            and output.is_dir()
            and not output.is_symlink()
            and not any(output.iterdir())
        ):
            output.rmdir()
        raise
    return bound_summary


def _validate_output_root(output: Path, source_paths: Sequence[Path]) -> Path:
    for path in (output, *source_paths, *_FROZEN_IMMUTABLE_ROOTS):
        assert_no_reparse_components(path)
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve(strict=False)
    for source in source_paths:
        source_resolved = source.resolve(strict=False)
        if resolved == source_resolved or source_resolved in resolved.parents or resolved in source_resolved.parents:
            raise ValueError("output_root must be separate from inputs")
    for frozen_root in _FROZEN_IMMUTABLE_ROOTS:
        frozen_resolved = frozen_root.resolve(strict=False)
        if resolved == frozen_resolved or frozen_resolved in resolved.parents:
            raise ValueError("output_root must be outside frozen immutable roots")
    if not output.parent.is_dir():
        raise ValueError("output_root parent directory does not exist")
    return output


def _validate_qwen_production_inputs(
    model_snapshot: str | Path | None,
    model_runtime_receipt: str | Path | None,
    model_snapshot_binding: str | Path | None,
) -> dict[str, Any]:
    if model_snapshot is None or model_runtime_receipt is None or model_snapshot_binding is None:
        raise ValueError("Qwen production replay requires model snapshot, runtime receipt, and snapshot binding")
    snapshot = Path(model_snapshot)
    runtime = Path(model_runtime_receipt)
    binding_path = Path(model_snapshot_binding)
    for path, label in ((snapshot, "model snapshot"), (runtime, "model runtime receipt"), (binding_path, "model snapshot binding")):
        assert_no_reparse_components(path)
        if not path.is_absolute():
            raise ValueError(f"{label} must be absolute")
    _, binding_raw = _load_json(binding_path.resolve(strict=True), "model snapshot binding")
    binding = json.loads(binding_raw.decode("utf-8"))
    provenance = verify_model_provenance(
        snapshot.resolve(strict=True),
        runtime.resolve(strict=True),
        binding,
        binding_raw=binding_raw,
        binding_path=binding_path.resolve(strict=True),
    )
    expected = {
        "repo": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "tree_sha256": "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db",
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise ValueError(f"Qwen production provenance mismatch for {key}")
    return provenance


def _qwen_adapter_source_path() -> Path:
    path = Path(__file__).with_name("vnext_run_letta_qwen_prompted_answer.py")
    assert_no_reparse_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Qwen adapter source must be a regular file")
    return path.resolve(strict=True)


def _source_sha256(path: Path, label: str) -> str:
    assert_no_reparse_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return sha256_bytes(path.read_bytes())
def _revalidate_fixture_and_inputs(
    *,
    fixture_root: Path,
    manifest_path: Path,
    candidate: Path,
    audit: Path,
    cell: Mapping[str, Any],
    initial_manifest_sha: str,
    initial_candidate_hashes: Mapping[str, str],
    initial_audit_sha: str,
    initial_fixture_hashes: Mapping[str, str],
    initial_rows: Sequence[Mapping[str, Any]],
    allow_relocated_authenticated_inputs: bool,
    execution_mode: str,
    manager_fixture_attestation: str | Path | None,
    manager_fixture_attestation_sha256: str | None,
    initial_fixture_authentication: Mapping[str, Any] | None,
    model_kind: str,
    model_snapshot: str | Path | None,
    model_runtime_receipt: str | Path | None,
    model_snapshot_binding: str | Path | None,
    initial_qwen_provenance: Mapping[str, Any] | None,
    qwen_adapter_source: Path | None,
    initial_qwen_adapter_source_sha256: str | None,
) -> None:
    _, manifest_raw, provenance, tasks, _ = _validate_manifest_and_candidate(
        manifest_path,
        candidate,
        audit,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    if sha256_bytes(manifest_raw) != initial_manifest_sha or provenance["artifact_hashes"] != dict(initial_candidate_hashes) or provenance["audit_attestation_sha256"] != initial_audit_sha:
        raise ValueError("candidate or manifest changed before publication")
    rows, summary, index, fixture_hashes = _load_fixture(fixture_root)
    if fixture_hashes != dict(initial_fixture_hashes) or rows != list(initial_rows):
        raise ValueError("manager fixture changed before publication")
    _validate_fixture_rows(
        rows,
        summary,
        index,
        json.loads(manifest_raw.decode("utf-8")),
        cell,
        tasks,
        provenance,
        initial_audit_sha,
        manifest_sha256=initial_manifest_sha,
        execution_mode=execution_mode,
    )
    if execution_mode == "production":
        observed_authentication = _validate_manager_fixture_attestation(
            manager_fixture_attestation,
            manager_fixture_attestation_sha256=manager_fixture_attestation_sha256,
            fixture_root=fixture_root,
            fixture_hashes=fixture_hashes,
            summary=summary,
            index=index,
            manifest_sha256=initial_manifest_sha,
            candidate_hashes=provenance["artifact_hashes"],
            audit_sha=initial_audit_sha,
        )
        if observed_authentication != dict(initial_fixture_authentication or {}):
            raise ValueError("manager fixture receipt changed before publication")
    if execution_mode == "production" and model_kind == "qwen":
        observed_qwen_provenance = _validate_qwen_production_inputs(
            model_snapshot,
            model_runtime_receipt,
            model_snapshot_binding,
        )
        if observed_qwen_provenance != dict(initial_qwen_provenance or {}):
            raise ValueError("Qwen production provenance changed before publication")
        if qwen_adapter_source is None or initial_qwen_adapter_source_sha256 is None:
            raise ValueError("Qwen adapter source provenance is missing")
        observed_qwen_adapter_source_sha256 = _source_sha256(
            qwen_adapter_source,
            "Qwen adapter source",
        )
        if observed_qwen_adapter_source_sha256 != initial_qwen_adapter_source_sha256:
            raise ValueError("Qwen adapter source changed before publication")


def run(
    *,
    manager_fixture_root: str | Path,
    manifest: str | Path,
    candidate_root: str | Path,
    audit_attestation: str | Path,
    output_root: str | Path,
    cell_id: str,
    execution_mode: str | None = None,
    answer_model_factory: Callable[[], AnswerReplayModel] | None = None,
    model_kind: str | None = None,
    model_snapshot: str | Path | None = None,
    model_runtime_receipt: str | Path | None = None,
    model_snapshot_binding: str | Path | None = None,
    muse_server_url: str | None = None,
    manager_fixture_attestation: str | Path | None = None,
    manager_fixture_attestation_sha256: str | None = None,
    allow_relocated_authenticated_inputs: bool = False,
) -> dict[str, Any]:
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("execution_mode must be explicitly production or injected_test_only")
    fixture_root = Path(manager_fixture_root)
    assert_no_reparse_components(fixture_root)
    if fixture_root.is_symlink():
        raise ValueError("manager fixture root must be a regular directory")
    fixture_root = fixture_root.resolve(strict=True)
    manifest_path = Path(manifest).resolve(strict=True)
    candidate = Path(candidate_root).resolve(strict=True)
    audit = Path(audit_attestation).resolve(strict=True)
    _, expected_model_kind = planner._cell_spec(cell_id)
    qwen_adapter_source = None
    initial_qwen_adapter_source_sha256 = None
    output_sources = [
        manifest_path,
        candidate,
        audit,
        fixture_root,
        fixture_root / "manager_rows.jsonl",
        fixture_root / "manager_summary.json",
        fixture_root / "artifact_index.json",
    ]
    if manager_fixture_attestation is not None:
        output_sources.append(Path(manager_fixture_attestation))
    if execution_mode == "production" and expected_model_kind == "qwen":
        qwen_adapter_source = _qwen_adapter_source_path()
        initial_qwen_adapter_source_sha256 = _source_sha256(
            qwen_adapter_source,
            "Qwen adapter source",
        )
        output_sources.extend(
            Path(path)
            for path in (model_snapshot, model_runtime_receipt, model_snapshot_binding)
            if path is not None
        )
        output_sources.append(qwen_adapter_source)
    output = _validate_output_root(Path(output_root), tuple(output_sources))
    rows, fixture_summary, fixture_index, fixture_hashes = _load_fixture(fixture_root)
    loaded_manifest, manifest_raw, provenance, tasks, relocation = _validate_manifest_and_candidate(
        manifest_path,
        candidate,
        audit,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
    )
    manifest_sha = sha256_bytes(manifest_raw)
    cell = _expected_cell(loaded_manifest, cell_id, tasks=tasks)
    expected_manager_kind, expected_model_kind = planner._cell_spec(cell_id)
    if fixture_summary.get("manager_kind") != expected_manager_kind:
        raise ValueError("manager fixture manager kind does not match factorial cell")
    if model_kind is not None and model_kind != expected_model_kind:
        raise ValueError("model_kind does not match factorial cell")
    validated_rows, scope_info = _validate_fixture_rows(
        rows,
        fixture_summary,
        fixture_index,
        loaded_manifest,
        cell,
        tasks,
        provenance,
        provenance["audit_attestation_sha256"],
        manifest_sha256=manifest_sha,
        execution_mode=execution_mode,
    )
    fixture_root_digest = _fixture_root_digest(fixture_hashes)
    relocation = build_relocation_metadata(
        manifest_sha256=manifest_sha,
        candidate_release_index_sha256=provenance["release_index_sha256"],
        audit_attestation_sha256=provenance["audit_attestation_sha256"],
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
        manager_fixture_root_digest=fixture_root_digest,
    )
    fixture_authentication = None
    if execution_mode == "production":
        fixture_authentication = _validate_manager_fixture_attestation(
            manager_fixture_attestation,
            manager_fixture_attestation_sha256=manager_fixture_attestation_sha256,
            fixture_root=fixture_root,
            fixture_hashes=fixture_hashes,
            summary=fixture_summary,
            index=fixture_index,
            manifest_sha256=manifest_sha,
            candidate_hashes=provenance["artifact_hashes"],
            audit_sha=provenance["audit_attestation_sha256"],
        )
    expected_identity = cell["answer"]["identity"]
    qwen_provenance = None
    muse_required_binding = None
    if execution_mode == "production" and expected_model_kind == "qwen":
        qwen_provenance = _validate_qwen_production_inputs(
            model_snapshot,
            model_runtime_receipt,
            model_snapshot_binding,
        )
    if execution_mode == "production" and expected_model_kind == "muse":
        raise ValueError("Muse production replay is blocked pending authenticated Muse runtime attestation")
    if answer_model_factory is None:
        if execution_mode != "production":
            raise ValueError("injected_test_only mode requires an explicit answer_model_factory")
        if expected_model_kind == "qwen":
            assert qwen_provenance is not None
            answer_model_factory = _build_qwen_production_factory(
                Path(model_snapshot).resolve(strict=True),
                qwen_provenance,
                expected_identity,
                adapter_source_sha256=initial_qwen_adapter_source_sha256,
            )
        else:
            if muse_server_url is None:
                raise ValueError("muse_server_url is required for Muse production replay")
            from scripts.vnext_run_main_track_muse_answer_baseline import MuseGlimmerAnswerModel

            def muse_factory() -> AnswerReplayModel:
                return MuseGlimmerAnswerModel(muse_server_url)

            muse_factory.production_bound = True  # type: ignore[attr-defined]
            answer_model_factory = muse_factory
    elif execution_mode == "production":
        raise ValueError("injected answer factory cannot be used on a production publication path")
    if execution_mode == "injected_test_only" and fixture_summary.get("scientific_evidence") is not False:
        raise ValueError("injected_test_only fixture must remain non-scientific")
    model: Any | None = None
    model_identity: dict[str, Any] | None = None
    output_rows: list[dict[str, Any]] = []
    task_by_id = {task.task_id: task for task in tasks}
    load = None
    close = None
    try:
        model = answer_model_factory()
        if model is None:
            raise ValueError("answer model factory returned no model")
        close = getattr(model, "close", None)
        model_identity = _identity(
            model,
            expected_identity,
            execution_mode=execution_mode,
            model_kind=expected_model_kind,
            required_binding=(
                _qwen_public_identity(
                    expected_identity,
                    qwen_provenance,
                    adapter_source_sha256=initial_qwen_adapter_source_sha256,
                )
                if qwen_provenance is not None
                else muse_required_binding
            ),
        )
        load = getattr(model, "load", None)
        close = getattr(model, "close", None)
        if callable(load):
            load()
        for fixture in validated_rows:
            task = task_by_id[fixture["task_id"]]
            if fixture.get("status") == "UNSUPPORTED":
                unsupported_spec = next(
                    item for item in cell["unsupported_tasks"] if item["task_id"] == fixture["task_id"]
                )
                fixture_for_output = dict(fixture)
                fixture_for_output.setdefault("detail", unsupported_spec.get("detail"))
                output_rows.append(
                    _build_row(
                        task,
                        fixture_for_output,
                        model,
                        model_identity,
                        candidate_hashes=provenance["artifact_hashes"],
                        audit_sha=provenance["audit_attestation_sha256"],
                        fixture_hashes=fixture_hashes,
                        cell_id=cell_id,
                        relocation=relocation,
                        execution_mode=execution_mode,
                        fixture_authentication=fixture_authentication,
                    )
                )
                continue
            try:
                output_rows.append(
                    _build_row(
                        task,
                        fixture,
                        model,
                        model_identity,
                        candidate_hashes=provenance["artifact_hashes"],
                        audit_sha=provenance["audit_attestation_sha256"],
                        fixture_hashes=fixture_hashes,
                        cell_id=cell_id,
                        relocation=relocation,
                        execution_mode=execution_mode,
                        fixture_authentication=fixture_authentication,
                    )
                )
            except AnswerExecutionError as exc:
                output_rows.append(
                    _failed_row(
                        task,
                        fixture,
                        model_identity,
                        candidate_hashes=provenance["artifact_hashes"],
                        audit_sha=provenance["audit_attestation_sha256"],
                        fixture_hashes=fixture_hashes,
                        cell_id=cell_id,
                        relocation=relocation,
                        execution_mode=execution_mode,
                        fixture_authentication=fixture_authentication,
                        error=exc,
                    )
                )
    finally:
        if callable(close):
            close()
    if model_identity is None:
        raise ValueError("answer model identity was not established")
    _revalidate_fixture_and_inputs(
        fixture_root=fixture_root,
        manifest_path=manifest_path,
        candidate=candidate,
        audit=audit,
        cell=cell,
        initial_manifest_sha=manifest_sha,
        initial_candidate_hashes=provenance["artifact_hashes"],
        initial_audit_sha=provenance["audit_attestation_sha256"],
        initial_fixture_hashes=fixture_hashes,
        initial_rows=validated_rows,
        allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
        execution_mode=execution_mode,
        manager_fixture_attestation=manager_fixture_attestation,
        manager_fixture_attestation_sha256=manager_fixture_attestation_sha256,
        initial_fixture_authentication=fixture_authentication,
        model_kind=expected_model_kind,
        model_snapshot=model_snapshot,
        model_runtime_receipt=model_runtime_receipt,
        model_snapshot_binding=model_snapshot_binding,
        initial_qwen_provenance=qwen_provenance,
        qwen_adapter_source=qwen_adapter_source,
        initial_qwen_adapter_source_sha256=initial_qwen_adapter_source_sha256,
    )
    summary = build_summary(
        output_rows,
        fixture_summary,
        scope=scope_info["scope"],
        scientific_evidence=scope_info["scientific_evidence"],
        model_identity=model_identity,
        candidate_hashes=provenance["artifact_hashes"],
        audit_sha=provenance["audit_attestation_sha256"],
        fixture_hashes=fixture_hashes,
        relocation=relocation,
        execution_mode=execution_mode,
        fixture_authentication=fixture_authentication,
        manifest_sha256=manifest_sha,
        eligible_supported_count=len(cell["supported_task_ids"]),
        selected_supported_task_ids=scope_info["selected_supported_task_ids"],
        unsupported_task_ids=[
            row["task_id"] for row in validated_rows if row.get("status") == "UNSUPPORTED"
        ],
    )
    publication_sources = [
        manifest_path,
        candidate / "tasks.jsonl",
        candidate / "release_index.json",
        audit,
        fixture_root / "manager_rows.jsonl",
        fixture_root / "manager_summary.json",
        fixture_root / "artifact_index.json",
    ]
    if manager_fixture_attestation is not None:
        publication_sources.append(Path(manager_fixture_attestation))
    if execution_mode == "production" and expected_model_kind == "qwen":
        publication_sources.extend(
            Path(path)
            for path in (model_snapshot, model_runtime_receipt, model_snapshot_binding)
            if path is not None
        )
        publication_sources.append(qwen_adapter_source)
    summary = _publish(
        output,
        output_rows,
        summary,
        source_paths=tuple(publication_sources),
        pre_publish=lambda: _revalidate_fixture_and_inputs(
            fixture_root=fixture_root,
            manifest_path=manifest_path,
            candidate=candidate,
            audit=audit,
            cell=cell,
            initial_manifest_sha=manifest_sha,
            initial_candidate_hashes=provenance["artifact_hashes"],
            initial_audit_sha=provenance["audit_attestation_sha256"],
            initial_fixture_hashes=fixture_hashes,
            initial_rows=validated_rows,
            allow_relocated_authenticated_inputs=allow_relocated_authenticated_inputs,
            execution_mode=execution_mode,
            manager_fixture_attestation=manager_fixture_attestation,
            manager_fixture_attestation_sha256=manager_fixture_attestation_sha256,
            initial_fixture_authentication=fixture_authentication,
            model_kind=expected_model_kind,
            model_snapshot=model_snapshot,
            model_runtime_receipt=model_runtime_receipt,
            model_snapshot_binding=model_snapshot_binding,
            initial_qwen_provenance=qwen_provenance,
        qwen_adapter_source=qwen_adapter_source,
        initial_qwen_adapter_source_sha256=initial_qwen_adapter_source_sha256,
        ),
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay prompted answers from an immutable main-track manager fixture")
    parser.add_argument("--manager-fixture-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "vnext" / "main_track_v1_factorial_plan_v2" / "factorial_manifest.json")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-attestation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--model-kind", choices=("qwen", "muse"))
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--model-runtime-receipt", type=Path)
    parser.add_argument("--model-snapshot-binding", type=Path)
    parser.add_argument("--muse-server-url")
    parser.add_argument("--manager-fixture-attestation", type=Path)
    parser.add_argument("--manager-fixture-attestation-sha256")
    parser.add_argument("--allow-relocated-authenticated-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run(**vars(args))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": summary["status"],
                "evidence_class": summary["evidence_class"],
                "scope": summary["scope"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
