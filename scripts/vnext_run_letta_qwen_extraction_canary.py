from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Protocol
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.bridge import JsonlSubprocessBridge
from mub.vnext.external.providers.letta import (
    LettaAdapterConfigurationV1,
    build_letta_adapter_configuration,
    compute_letta_configuration_hash,
)
from mub.vnext.external.providers.letta_adapter import LettaExternalAdapterV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.runtime.answer_model_v3 import snapshot_tree_sha256_v3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3

TASK_SHA256 = "ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE_SHA256 = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
MODEL_RUNTIME_RECEIPT_SHA256 = "5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f"
MODEL_SNAPSHOT_PATH = "/NAS/HuggingFaceModels/Qwen3.5-9B"
# Short aliases retained for provenance-oriented callers.
TASK_SHA = TASK_SHA256
MODEL = MODEL_ID
REV = MODEL_REVISION
SCHEMA_VERSION = "memupdatebench.external.letta-qwen-extraction.canary.v2"
FULL_SCHEMA_VERSION = "memupdatebench.external.letta-qwen-extraction.full-family-a.v1"
CANARY_SCOPE = "canary32"
FULL_SCOPE = "full-family-a80"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def task_core(task: MemUpdateTaskV3) -> str:
    return task.metadata.extra["semantic_core_id"]


def _parse_authenticated_tasks(raw: bytes) -> list[MemUpdateTaskV3]:
    return [
        MemUpdateTaskV3.model_validate(json.loads(line))
        for line in raw.splitlines()
        if line.strip()
    ]


def select_tasks(raw: bytes, *, scope: str = CANARY_SCOPE) -> list[MemUpdateTaskV3]:
    if scope not in {CANARY_SCOPE, FULL_SCOPE}:
        raise ValueError(f"unsupported task scope: {scope}")
    if sha256_bytes(raw) != TASK_SHA256:
        raise ValueError("task view SHA-256 mismatch")
    tasks = _parse_authenticated_tasks(raw)
    tasks.sort(key=lambda task: (task_core(task).encode("utf-8"), task.task_id.encode("utf-8")))
    if scope == FULL_SCOPE:
        if len(tasks) != 80:
            raise ValueError("authenticated Family-A full scope is not exactly 80 tasks")
        return tasks
    cores = sorted({task_core(task) for task in tasks}, key=lambda x: x.encode("utf-8"))[:8]
    selected = [task for task in tasks if task_core(task) in cores]
    if len(selected) != 32 or len(cores) != 8:
        raise ValueError("canonical Family-A selection is not 32 tasks from 8 cores")
    return selected


def build_extraction_prompt(raw_text: str, attribute: str) -> str:
    if type(raw_text) is not str or type(attribute) is not str or not raw_text.strip() or not attribute.strip():
        raise ValueError("visible extraction inputs must be nonblank strings")
    return (
        "Extract the visible memory event into exactly one JSON object with exactly the keys "
        "operation and value. operation must be one of add, update, noop, delete. "
        "The admitted attribute is " + json.dumps(attribute, ensure_ascii=False) + ". "
        "For add or update, value must be the scalar value of that attribute, not an object or profile. "
        "For noop or delete, value must be null. Do not infer or mention any other object.\n"
        "Visible event:\n" + raw_text
    )


def validate_loopback_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Letta endpoint must be an uncredentialed HTTP loopback URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        raise ValueError("Letta endpoint must be an explicit loopback URL")
    return url.rstrip("/")


def validate_loopback_binding(url: str, closure: dict) -> str:
    url = validate_loopback_url(url)
    measured = closure.get("runtime", {}).get("measured", {})
    try:
        expected_port = int(measured["server_port"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("qualification closure lacks measured server port") from None
    if urlsplit(url).port != expected_port:
        raise ValueError("canary endpoint does not match qualified server port")
    return url


def current_retrieval_metric(entries) -> None:
    return None

    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def _is_reparse(path: Path) -> bool:
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def _within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_root(output: Path, *, frozen_roots: tuple[Path, ...] = ()) -> Path:
    if not output.is_absolute():
        raise ValueError("output root must be absolute")
    assert_no_reparse_components(output.parent)
    if output.exists() and _is_reparse(output):
        raise ValueError("output root must not be a symlink or reparse point")
    resolved = output.resolve(strict=False)
    for frozen in frozen_roots:
        frozen_resolved = frozen.resolve(strict=False)
        if _within(frozen_resolved, resolved) or _within(resolved, frozen_resolved):
            raise ValueError("output root overlaps frozen release root")
    return resolved


def stable_tree_sha256(root: Path) -> str:
    root = root.resolve(strict=True)
    assert_no_reparse_components(root)
    if not root.is_dir() or _is_reparse(root):
        raise ValueError("model snapshot must be a real directory")
    for item in root.rglob("*"):
        if _is_reparse(item):
            raise ValueError("model snapshot contains symlink or reparse point")
    return snapshot_tree_sha256_v3(snapshot_path=root, model_id=MODEL_ID, revision=MODEL_REVISION)


_BINDING_FIELDS = frozenset({
    "schema_version",
    "repo",
    "revision",
    "shared_snapshot_path",
    "tree_sha256",
    "file_count",
    "total_bytes",
    "entries",
    "receipt_payload_sha256",
})
_BINDING_AUDIT_FIELDS = frozenset({
    "available_bytes_after",
    "available_bytes_before",
    "model_loads",
    "operation_id",
    "provider_calls",
    "remaining_stage_roots",
    "removed_allocated_bytes",
    "removed_duplicate_path",
    "removed_logical_bytes",
})
_BINDING_ENTRY_FIELDS = frozenset({"path", "sha256", "bytes"})
_BINDING_ENTRY_AUDIT_FIELDS = frozenset({"source_digest", "source_digest_kind"})
_SOURCE_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _binding_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"snapshot binding {label} must be lowercase SHA-256")
    return value


def _binding_nonnegative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"snapshot binding {label} must be a nonnegative int")


def _binding_audit_path(value: object, label: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if type(value) is not str or not value or any(ord(char) == 0 for char in value):
        raise ValueError(f"snapshot binding {label} path is invalid")
    normalized = value.replace("\\", "/")
    if any(part in {".", ".."} for part in normalized.split("/")):
        raise ValueError(f"snapshot binding {label} path traversal is invalid")


def _validate_binding_audit_fields(binding: dict) -> None:
    for field in (
        "available_bytes_after",
        "available_bytes_before",
        "model_loads",
        "provider_calls",
        "removed_allocated_bytes",
        "removed_logical_bytes",
    ):
        if field in binding:
            _binding_nonnegative_int(binding[field], field)
    if "operation_id" in binding and (type(binding["operation_id"]) is not str or not binding["operation_id"]):
        raise ValueError("snapshot binding operation_id must be a nonempty string")
    if "remaining_stage_roots" in binding:
        roots = binding["remaining_stage_roots"]
        if type(roots) is not list:
            raise ValueError("snapshot binding remaining_stage_roots must be a list")
        for root in roots:
            _binding_audit_path(root, "remaining_stage_roots")
    if "removed_duplicate_path" in binding:
        _binding_audit_path(binding["removed_duplicate_path"], "removed_duplicate_path", allow_none=True)


def _validate_binding_entry_fields(entry: dict) -> None:
    entry_fields = set(entry)
    allowed_fields = _BINDING_ENTRY_FIELDS | _BINDING_ENTRY_AUDIT_FIELDS
    if not _BINDING_ENTRY_FIELDS <= entry_fields or entry_fields - allowed_fields:
        raise ValueError("snapshot binding entry fields mismatch")
    has_source_digest = "source_digest" in entry
    has_source_kind = "source_digest_kind" in entry
    if has_source_digest != has_source_kind:
        raise ValueError("snapshot binding source digest fields must be provided together")
    if has_source_digest:
        source_digest = entry["source_digest"]
        if type(source_digest) is not str or _SOURCE_DIGEST_PATTERN.fullmatch(source_digest) is None:
            raise ValueError("snapshot binding source_digest must be lowercase 40-64 hex")
        source_digest_kind = entry["source_digest_kind"]
        if type(source_digest_kind) is not str or source_digest_kind not in {"git", "sha256"}:
            raise ValueError("snapshot binding source_digest_kind is invalid")


def _binding_snapshot_root(snapshot: Path) -> Path:
    selected = Path(snapshot)
    assert_no_reparse_components(selected)
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    if not selected.is_dir() or _is_reparse(selected):
        raise ValueError("snapshot binding path is unsafe")
    resolved = selected.resolve(strict=True)
    assert_no_reparse_components(resolved)
    for item in resolved.rglob("*"):
        if _is_reparse(item):
            raise ValueError("snapshot binding contains symlink or reparse point")
        if not item.is_dir() and not item.is_file():
            raise ValueError("snapshot binding contains a non-file entry")
    return resolved


def _binding_entry_path(snapshot: Path, raw_path: object) -> Path:
    if type(raw_path) is not str or not raw_path or "\\" in raw_path or any(ord(char) == 0 for char in raw_path):
        raise ValueError("snapshot binding entry path is invalid")
    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or raw_path.startswith("/"):
        raise ValueError("snapshot binding entry path is not a normalized relative path")
    candidate = snapshot.joinpath(*parts)
    try:
        candidate.relative_to(snapshot)
    except ValueError as exc:
        raise ValueError("snapshot binding entry path escapes snapshot") from exc
    return candidate


def _binding_receipt_bytes(
    binding: dict,
    *,
    binding_raw: bytes | None,
    binding_path: Path | None,
) -> tuple[bytes, Path | None]:
    if binding_raw is not None and type(binding_raw) is not bytes:
        raise TypeError("snapshot binding receipt bytes must be bytes")
    receipt_path = None
    if binding_path is not None:
        receipt_path = Path(binding_path)
        assert_no_reparse_components(receipt_path)
        if not receipt_path.is_absolute():
            receipt_path = Path.cwd() / receipt_path
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("snapshot binding receipt must be a real regular file")
        receipt_path = receipt_path.resolve(strict=True)
        file_raw = receipt_path.read_bytes()
        if binding_raw is None:
            binding_raw = file_raw
        elif file_raw != binding_raw:
            raise ValueError("snapshot binding receipt bytes changed while being read")
    if binding_raw is None:
        binding_raw = canonical_json_bytes(binding)
    if canonical_json_bytes(binding) != binding_raw:
        raise ValueError("snapshot binding receipt is not canonical")
    return binding_raw, receipt_path


def validate_snapshot_binding(
    snapshot: Path,
    binding: dict,
    *,
    binding_raw: bytes | None = None,
    binding_path: Path | None = None,
) -> dict:
    if type(binding) is not dict or binding.get("schema_version") != "memupdatebench.post-core.shared-snapshot-binding.v1":
        raise ValueError("snapshot binding schema mismatch")
    if not _BINDING_FIELDS <= set(binding):
        raise ValueError("snapshot binding fields mismatch")
    if scan_for_secrets(binding):
        raise ValueError("snapshot binding failed secret scan")
    _validate_binding_audit_fields(binding)
    if binding["repo"] != MODEL_ID or binding["revision"] != MODEL_REVISION or binding["shared_snapshot_path"] != MODEL_SNAPSHOT_PATH:
        raise ValueError("snapshot binding identity mismatch")
    if type(binding["shared_snapshot_path"]) is not str or not binding["shared_snapshot_path"]:
        raise ValueError("snapshot binding shared path is invalid")
    tree_hash = _binding_sha256(binding["tree_sha256"], "tree_sha256")
    if tree_hash != MODEL_TREE_SHA256:
        raise ValueError("snapshot binding tree identity mismatch")
    if type(binding["file_count"]) is not int or binding["file_count"] < 0 or type(binding["total_bytes"]) is not int or binding["total_bytes"] < 0:
        raise ValueError("snapshot binding aggregate metadata is invalid")
    receipt_payload_sha256 = _binding_sha256(binding["receipt_payload_sha256"], "receipt_payload_sha256")
    payload = dict(binding)
    payload.pop("receipt_payload_sha256")
    if receipt_payload_sha256 != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("snapshot binding receipt payload hash mismatch")
    receipt_raw, receipt_path = _binding_receipt_bytes(binding, binding_raw=binding_raw, binding_path=binding_path)

    entries = binding["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("snapshot binding entries must be a nonempty list")
    snapshot_root = _binding_snapshot_root(snapshot)
    declared: dict[str, dict[str, object]] = {}
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("snapshot binding entry fields mismatch")
        _validate_binding_entry_fields(entry)
        entry_path = entry["path"]
        candidate = _binding_entry_path(snapshot_root, entry_path)
        digest = _binding_sha256(entry["sha256"], "entry sha256")
        size_bytes = entry["bytes"]
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("snapshot binding entry bytes must be a nonnegative int")
        if entry_path in declared:
            raise ValueError("snapshot binding entries contain duplicate paths")
        if _is_reparse(candidate) or not candidate.is_file():
            raise ValueError("snapshot binding entry path is not a regular file")
        declared[entry_path] = {"sha256": digest, "bytes": size_bytes}

    observed: dict[str, dict[str, object]] = {}
    for item in snapshot_root.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(snapshot_root).as_posix()
        observed[rel] = {"sha256": sha256_bytes(item.read_bytes()), "bytes": item.stat().st_size}
    if observed != declared:
        raise ValueError("snapshot binding entries do not exactly match snapshot files")
    if len(entries) != binding["file_count"] or len(observed) != binding["file_count"]:
        raise ValueError("snapshot binding file count mismatch")
    if sum(item["bytes"] for item in observed.values()) != binding["total_bytes"]:
        raise ValueError("snapshot binding total bytes mismatch")
    return {
        "path": str(receipt_path) if receipt_path is not None else None,
        "receipt_path": str(receipt_path) if receipt_path is not None else None,
        "receipt_file_sha256": sha256_bytes(receipt_raw),
        "receipt_payload_sha256": receipt_payload_sha256,
        "snapshot_binding_receipt_sha256": sha256_bytes(receipt_raw),
        "snapshot_binding_payload_sha256": receipt_payload_sha256,
        "repo": binding["repo"],
        "revision": binding["revision"],
        "shared_snapshot_path": binding["shared_snapshot_path"],
        "tree_sha256": tree_hash,
        "file_count": binding["file_count"],
        "total_bytes": binding["total_bytes"],
    }


def verify_model_provenance(
    snapshot: Path,
    runtime_receipt: Path,
    binding: dict | None = None,
    *,
    binding_raw: bytes | None = None,
    binding_path: Path | None = None,
) -> dict:
    if binding is None:
        raise ValueError("authoritative snapshot binding is required")
    binding_provenance = validate_snapshot_binding(
        snapshot,
        binding,
        binding_raw=binding_raw,
        binding_path=binding_path,
    )
    snapshot_root = _binding_snapshot_root(snapshot)
    tree_hash = binding_provenance["tree_sha256"]
    runtime_receipt = runtime_receipt.resolve(strict=True)
    if _is_reparse(runtime_receipt) or not runtime_receipt.is_file():
        raise ValueError("runtime receipt must be a regular non-symlink file")
    raw = runtime_receipt.read_bytes()
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw or scan_for_secrets(value):
        raise ValueError("runtime receipt is not canonical or is sensitive")
    if sha256_bytes(raw) != MODEL_RUNTIME_RECEIPT_SHA256:
        raise ValueError("runtime receipt hash mismatch")
    if value.get("schema_version") != "memupdatebench.post-core.qwen-runtime-source-receipt.v1":
        raise ValueError("runtime receipt schema mismatch")
    required = ("load_status", "generation_status", "determinism_status", "unload_status", "provider_calls", "network_calls", "benchmark_generations", "gpu_index", "node")
    if any(field not in value for field in required) or any(value[field] != "PASS" for field in ("load_status", "generation_status", "determinism_status", "unload_status")) or any(value[field] != 0 for field in ("provider_calls", "network_calls", "benchmark_generations")):
        raise ValueError("runtime receipt affirmative fields mismatch")
    if type(value["gpu_index"]) is not int or value["gpu_index"] < 0 or type(value["node"]) is not str or not value["node"]:
        raise ValueError("runtime receipt execution identity mismatch")
    return {
        "snapshot": str(snapshot_root),
        "tree_sha256": tree_hash,
        "snapshot_binding": binding_provenance,
        "snapshot_binding_receipt_sha256": binding_provenance["snapshot_binding_receipt_sha256"],
        "snapshot_binding_payload_sha256": binding_provenance["snapshot_binding_payload_sha256"],
        "repo": binding_provenance["repo"],
        "revision": binding_provenance["revision"],
        "shared_snapshot_path": binding_provenance["shared_snapshot_path"],
        "file_count": binding_provenance["file_count"],
        "total_bytes": binding_provenance["total_bytes"],
        "runtime_receipt_sha256": sha256_bytes(raw),
        "runtime_identity": value["schema_version"],
        "runtime_receipt": value,
    }


def validate_extraction(parsed: object) -> dict:
    if type(parsed) is not dict or set(parsed) != {"operation", "value"}:
        raise ValueError("invalid extraction JSON")
    operation, value = parsed["operation"], parsed["value"]
    if type(operation) is not str or operation not in {"add", "update", "noop", "delete"}:
        raise ValueError("invalid extraction operation")
    if operation in {"noop", "delete"}:
        if value is not None:
            raise ValueError("noop/delete extraction value must be null")
    elif value is None or type(value) not in {str, int, float, bool} or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("add/update extraction value must be a finite scalar")
    return parsed


def validate_qualification_artifacts(root: Path) -> dict:
    if not root.is_absolute():
        raise ValueError("qualification root must be absolute")
    assert_no_reparse_components(root)
    root = root.resolve(strict=True)
    names = ("letta_runtime_qualification.json", "letta_runtime_preflight.json", "letta_runtime_admission.json")
    values = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing qualification artifact: {name}")
        raw = path.read_bytes()
        value = json.loads(raw)
        if canonical_json_bytes(value) != raw or scan_for_secrets(value):
            raise ValueError(f"qualification artifact is not canonical or is sensitive: {name}")
        values[name] = value
    closure, preflight, admission = (values[name] for name in names)
    if closure.get("schema_version") != "memupdatebench.external.letta.runtime_qualification.v1" or preflight.get("schema_version") != "memupdatebench.external.letta.preflight.v2" or admission.get("schema_version") != "memupdatebench.external.letta.admission.v2":
        raise ValueError("qualification artifact schemas are not recognized")
    if closure.get("outcome") != "PASS" or preflight.get("outcome") not in {"pass", "PASS"} or preflight.get("passed") is not True or admission.get("outcome") not in {"pass", "PASS"} or admission.get("admitted") is not True:
        raise ValueError("Letta qualification closure, preflight, and admission must all PASS")
    if closure.get("candidate_id") != "letta_0_16_8_song1_local_linux" or preflight.get("candidate_id") != "letta_0_16_8_profile" or preflight.get("mode") != "profile_single_record_runtime" or admission.get("candidate_id") != "letta_0_16_8_profile" or admission.get("admission_scope") != "profile_single_record_runtime":
        raise ValueError("qualification candidate or mode mismatch")
    required_closure = ("candidate_id", "identity", "source", "project_source", "runner_source_sha256", "worker_source_sha256", "runtime", "boundary", "cleanup", "preflight", "admission")
    if any(field not in closure for field in required_closure):
        raise ValueError("qualification closure is incomplete")
    for field in ("runner_source_sha256", "worker_source_sha256"):
        if type(closure[field]) is not str or not re.fullmatch(r"[0-9a-f]{64}", closure[field]):
            raise ValueError(f"qualification closure has invalid {field}")
    artifact_hashes = {name: sha256_bytes((root / name).read_bytes()) for name in names}
    for field, name in (("preflight", "letta_runtime_preflight.json"), ("admission", "letta_runtime_admission.json")):
        declared = closure.get(field, {}).get("sha256") if isinstance(closure.get(field), dict) else None
        if type(declared) is not str or declared != artifact_hashes[name]:
            raise ValueError(f"qualification {field} hash mismatch")
    if closure["boundary"].get("llm_used") is not False or closure["boundary"].get("api_used") is not False or closure["boundary"].get("gpu_used") is not False:
        raise ValueError("qualification execution boundary is not affirmative")
    if closure["cleanup"].get("status") != "PASS" or closure["runtime"].get("loopback_only") is not True or closure["runtime"].get("measured", {}).get("database") is None or closure["runtime"].get("measured", {}).get("server_port") is None:
        raise ValueError("qualification runtime or cleanup is not affirmative")
    if any(field not in preflight for field in ("candidate_id", "mode", "identity", "official_health", "runtime", "namespace_reset_probe", "lifecycle", "clean_close", "security", "boundary", "unsupported")):
        raise ValueError("preflight artifact is incomplete")
    if any(field not in admission for field in ("candidate_id", "admission_scope", "gates", "reasons")):
        raise ValueError("admission artifact is incomplete")
    identity = closure.get("identity", {})
    if not isinstance(identity, dict) or identity.get("package_name") != "letta" or identity.get("package_version") != "0.16.8" or identity.get("source_commit") != "1131535716e8a31c9a437f8695e25ac98f203a24":
        raise ValueError("qualification closure package/source identity mismatch")
    runtime = closure.get("runtime", {})
    if runtime and runtime.get("loopback_only") is False:
        raise ValueError("qualification runtime is not loopback-only")
    artifact_hashes = {name: sha256_bytes((root / name).read_bytes()) for name in names}
    return {"closure": closure, "preflight": preflight, "admission": admission, "hashes": artifact_hashes}


def _require_real_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    assert_no_reparse_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real regular file")
    return path.resolve(strict=True)


def _require_real_project_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("Letta project root must be absolute")
    assert_no_reparse_components(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Letta project root must be a real directory")
    resolved = path.resolve(strict=True)
    for item in resolved.rglob("*"):
        assert_no_reparse_components(item)
        if item.is_symlink():
            raise ValueError("Letta project tree must not contain symlinks")
    return resolved


def _git_value(project: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(project), *args), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError("Letta project git identity unavailable")
    return result.stdout.strip()


def _tracked_tree_identity(project: Path) -> tuple[str, int]:
    raw = subprocess.run(("git", "-C", str(project), "ls-files", "-z"), check=False, capture_output=True).stdout
    paths = [project / item.decode("utf-8") for item in raw.split(b"\0") if item]
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(project).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(paths)


def validate_worker_runtime_binding(letta_python_executable: Path | str, letta_project_root: Path | str, closure: dict, *, expected_revision: str | None = None) -> dict:
    executable = _require_real_absolute_file(Path(letta_python_executable), "Letta Python executable")
    project = _require_real_project_root(Path(letta_project_root))
    if not isinstance(closure, dict):
        raise ValueError("qualification closure is required")
    project_source = closure.get("project_source")
    expected_commit = expected_revision or (project_source or {}).get("commit")
    expected_tree_hash = (project_source or {}).get("tree_sha256")
    expected_file_count = (project_source or {}).get("file_count")
    expected_runner_hash = closure.get("runner_source_sha256")
    expected_worker_hash = closure.get("worker_source_sha256")
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("qualification closure lacks expected project revision")
    if not isinstance(expected_tree_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_tree_hash) or type(expected_file_count) is not int:
        raise ValueError("qualification closure lacks expected project tree identity")
    if not isinstance(expected_runner_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_runner_hash):
        raise ValueError("qualification closure lacks expected runner source hash")
    if not isinstance(expected_worker_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_worker_hash):
        raise ValueError("qualification closure lacks expected worker source hash")
    observed_commit = _git_value(project, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise ValueError("Letta project revision mismatch")
    if _git_value(project, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Letta project git worktree must be clean")
    observed_tree_hash, observed_file_count = _tracked_tree_identity(project)
    if observed_tree_hash != expected_tree_hash or observed_file_count != expected_file_count:
        raise ValueError("Letta project tree identity mismatch")
    worker = project / "mub" / "vnext" / "external" / "workers" / "letta_worker.py"
    worker = _require_real_absolute_file(worker, "Letta worker source")
    observed_worker_hash = sha256_bytes(worker.read_bytes())
    if observed_worker_hash != expected_worker_hash:
        raise ValueError("Letta worker source hash mismatch")
    return {"python_executable": str(executable), "project_root": str(project), "project_revision": observed_commit, "worker_source": str(worker), "worker_source_sha256": observed_worker_hash, "runner_source_sha256": expected_runner_hash, "qualification_runner_source_sha256": expected_runner_hash}


def build_worker_command(letta_python_executable: Path | str, letta_project_root: Path | str, configuration_json: str) -> tuple[str, ...]:
    executable = _require_real_absolute_file(Path(letta_python_executable), "Letta Python executable")
    project = _require_real_project_root(Path(letta_project_root))
    worker = _require_real_absolute_file(project / "mub" / "vnext" / "external" / "workers" / "letta_worker.py", "Letta worker source")
    if type(configuration_json) is not str or not configuration_json:
        raise ValueError("Letta worker configuration must be canonical JSON")
    command = (str(executable), str(worker), "--configuration-json", configuration_json)
    if scan_for_secrets(command):
        raise ValueError("Letta worker command security scan failed")
    return command


def safe_worker_environment(base_url: str, project_root: Path) -> dict[str, str]:
    base_url = validate_loopback_url(base_url)
    project_root = _require_real_project_root(project_root)
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"}
    env = {name: value for name, value in os.environ.items() if name in allowed and type(value) is str}
    env.update({"LETTA_NATIVE_API_BASE_URL": base_url, "PYTHONPATH": str(project_root), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
    return env


def _write_new(path: Path, raw: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_bytes(raw)


def _append_row(path: Path, row: dict) -> None:
    raw = canonical_json_bytes(row) + b"\n"
    if scan_for_secrets(row):
        raise ValueError("canary row failed secret scan")
    with path.open("ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


class ModelExtractor(Protocol):
    def load(self) -> None: ...
    def extract(self, raw_text: str, attribute: str) -> tuple[dict, str, int, float]: ...
    def close(self) -> None: ...


class QwenExtractor:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot.resolve(strict=True)

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.snapshot), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()

    def extract(self, raw_text: str, attribute: str) -> tuple[dict, str, int, float]:
        prompt = build_extraction_prompt(raw_text, attribute)
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(rendered, return_tensors="pt").to("cuda:0")
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=96, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(output[0, encoded.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        parsed = json.loads(raw)
        if type(parsed) is not dict or set(parsed) != {"operation", "value"} or parsed["operation"] not in {"add", "update", "noop", "delete"}:
            raise ValueError("invalid extraction JSON")
        return parsed, raw, int(output.shape[-1] - encoded.input_ids.shape[-1]), (time.monotonic() - started) * 1000

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize()


def _adapter_from_bridge(bridge, task: MemUpdateTaskV3):
    key = FrozenMemoryObjectKey.model_validate(task.target_objects[0].model_dump(mode="python"), strict=True)
    config = build_letta_adapter_configuration(run_id="letta-qwen-" + task.task_id)
    return LettaExternalAdapterV3(bridge=bridge, configuration=config, target_objects=(key,)), config


def _row(task, status: str, *, error_class=None, state_accuracy=None, memory_size=None, stable_id=None, gold_retrieved=None, stale_retrieved=None, extra=None):
    row = {"task_id": task.task_id, "semantic_core_id": task_core(task), "status": status, "error_class": error_class, "state_accuracy": state_accuracy, "final_memory_size": memory_size, "stable_entry_id": stable_id, "gold_retrieved_k16": gold_retrieved, "stale_retrieved_k16": stale_retrieved}
    if extra:
        row.update(extra)
    return row


def _safe_error_detail(exc: Exception) -> str:
    detail = re.sub(r"[^a-zA-Z0-9_. -]", "", str(exc))[:240]
    return detail or type(exc).__name__


def cleanup_adapter(adapter, namespace: str) -> None:
    adapter.reset(ResetRequestV3(namespace=namespace))
    adapter.close()


def build_summary(
    rows: list[dict], *, scope: str, requested: int, rows_sha256: str,
    qualification_hashes: dict, qualification_identity: dict, letta_binding: dict,
    endpoint: str, model_provenance: dict, execution_mode: str = "production",
) -> dict:
    supported = [row for row in rows if row["status"] != "NOT_SUPPORTED"]
    passed = [row for row in supported if row["status"] == "PASS"]
    state_rows = [row for row in passed if row.get("state_accuracy") is not None]
    retrieval_rows = [row for row in passed if row.get("gold_retrieved_k16") is not None]
    memory_rows = [row for row in passed if row.get("final_memory_size") is not None]
    stable_rows = [row for row in passed if row.get("stable_entry_id") is not None]
    operations = {"add": 0, "update": 0, "noop": 0, "delete": 0}
    effective_operations = dict(operations)
    total_reconciliation = 0
    for row in rows:
        total_reconciliation += int(row.get("reconciliation_count") or 0)
        for extraction in row.get("extractions", []):
            operation = extraction.get("operation")
            effective = extraction.get("effective_operation")
            if isinstance(operation, str):
                operation = operation.lower()
            if isinstance(effective, str):
                effective = effective.lower()
            if operation in operations:
                operations[operation] += 1
            if effective in effective_operations:
                effective_operations[effective] += 1
    summary = {
        "schema_version": FULL_SCHEMA_VERSION if scope == FULL_SCOPE else SCHEMA_VERSION,
        "outcome": "PASS" if not [row for row in supported if row["status"] == "FAIL"] else "FAIL",
        "requested": requested,
        "terminal_rows": len(rows),
        "supported": len(supported),
        "unsupported": len(rows) - len(supported),
        "pass": len(passed),
        "fail": len(supported) - len(passed),
        "state_accuracy": sum(bool(row["state_accuracy"]) for row in state_rows) / len(state_rows) if state_rows else None,
        "state_accuracy_denominator": len(state_rows),
        "gold_retrieval_rate": sum(bool(row["gold_retrieved_k16"]) for row in passed) / len(passed) if passed else None,
        "avg_memory_size": sum(row["final_memory_size"] for row in passed) / len(passed) if passed else None,
        "rows_sha256": rows_sha256,
        "task_view_sha256": TASK_SHA256,
        "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "qualification_hashes": qualification_hashes,
        "qualification_identity": qualification_identity,
        "letta_worker_runtime": {**letta_binding, "configuration_hash": compute_letta_configuration_hash(build_letta_adapter_configuration(run_id="letta-qwen-manifest")), "cwd": letta_binding.get("project_root"), "environment": {"PYTHONPATH": letta_binding.get("project_root"), "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}},
        "letta_base_url": endpoint,
        "letta_configuration_hash": compute_letta_configuration_hash(build_letta_adapter_configuration(run_id="letta-qwen-manifest")),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, **model_provenance},
        "provider_calls": 0,
        "api_calls": 0,
        "answer_model_metrics": None,
        "execution_mode": execution_mode,
    }
    if scope == FULL_SCOPE:
        summary.update({
            "scope": FULL_SCOPE,
            "stable_entry_id_rate": sum(bool(row["stable_entry_id"]) for row in stable_rows) / len(stable_rows) if stable_rows else None,
            "stable_entry_id_denominator": len(stable_rows),
            "gold_retrieval_denominator": len(retrieval_rows),
            "memory_size_denominator": len(memory_rows),
            "total_reconciliation_count": total_reconciliation,
            "operation_counts": {"requested": operations, "effective": effective_operations},
            "prompted_exact_match": None,
            "prompted_answer_em": None,
            "prompted_answer_f1": None,
            "prompted_metrics": None,
            "prompted_metrics_denominator": 0,
        })
    return summary


def run(args, *, extractor_factory: Callable[[], ModelExtractor] | None = None, adapter_factory=None) -> dict:
    tasks_path = Path(args.tasks)
    output = validate_output_root(Path(args.output_root), frozen_roots=(ROOT / "data" / "vnext" / "core", ROOT / "data" / "vnext" / "phase0", ROOT / "configs" / "vnext" / "post_core"))
    if output.exists():
        raise FileExistsError(output)
    binding_path = _require_real_absolute_file(Path(args.model_snapshot_binding), "model snapshot binding receipt")
    binding_raw = binding_path.read_bytes()
    model_binding = json.loads(binding_raw)
    if canonical_json_bytes(model_binding) != binding_raw or scan_for_secrets(model_binding):
        raise ValueError("model snapshot binding must be canonical and secret-free")
    model_provenance = verify_model_provenance(
        Path(args.model_snapshot),
        Path(args.model_runtime_receipt),
        model_binding,
        binding_raw=binding_raw,
        binding_path=binding_path,
    )
    qualification = validate_qualification_artifacts(Path(args.qualification_root))
    letta_binding = validate_worker_runtime_binding(
        args.letta_python_executable,
        args.letta_project_root,
        qualification["closure"],
        expected_revision=getattr(args, "expected_letta_project_revision", None),
    )
    endpoint = validate_loopback_binding(args.letta_base_url, qualification["closure"])
    scope = getattr(args, "scope", CANARY_SCOPE)
    selected = select_tasks(tasks_path.read_bytes()) if scope == CANARY_SCOPE else select_tasks(tasks_path.read_bytes(), scope=scope)
    requested = 32 if scope == CANARY_SCOPE else 80
    rows_path = output / "rows.jsonl"
    extractor = QwenExtractor(Path(args.model_snapshot)) if extractor_factory is None else extractor_factory()
    extractor.load()
    output.mkdir(parents=True)
    rows = []
    try:
        for task in selected:
            if len(task.target_objects) != 1:
                row = _row(task, "NOT_SUPPORTED", error_class="multi_object_task", extra={"reason": "single_object_only"})
                _append_row(rows_path, row); rows.append(row); continue
            started = time.monotonic()
            adapter = None
            close_error = None
            cleanup_detail = None
            stage = "adapter_initialization"
            affected_entry_ids = []
            reconciliation_count = 0
            extractions = []
            namespace = "letta_qwen_" + task.task_id
            try:
                if adapter_factory is None:
                    config = build_letta_adapter_configuration(run_id="letta-qwen-" + task.task_id)
                    config_json = canonical_json_bytes(config.model_dump(mode="json")).decode("utf-8")
                    command = build_worker_command(args.letta_python_executable, args.letta_project_root, config_json)
                    bridge = JsonlSubprocessBridge(command=command, cwd=letta_binding["project_root"], environment=safe_worker_environment(endpoint, Path(letta_binding["project_root"])), timeout_seconds=60.0)
                    adapter = LettaExternalAdapterV3(bridge=bridge, configuration=config, target_objects=(FrozenMemoryObjectKey.model_validate(task.target_objects[0].model_dump(mode="python"), strict=True),))
                else:
                    adapter = adapter_factory(task)
                namespace = "letta_qwen_" + task.task_id
                stage = "namespace_reset"
                adapter.reset(ResetRequestV3(namespace=namespace))
                for event in task.events:
                    stage = "model_extraction"
                    parsed, raw_output, tokens, latency = extractor.extract(event.raw_text, task.target_objects[0].attribute)
                    parsed = validate_extraction(parsed)
                    operation = parsed["operation"]
                    if operation == "update" and not adapter.export_entries().entries:
                        operation = "add"
                        reconciliation_count += 1
                    value = parsed["value"]
                    if operation in {"add", "update"} and value is None:
                        raise ValueError("mutation value cannot be null")
                    if operation == "noop":
                        visible = "No memory object changes."
                    elif operation == "delete":
                        visible = f"Delete {task.target_objects[0].canonical_id} [scope=object; enumerated_targets={task.target_objects[0].canonical_id}; event_logical_time={event.timestamp or 'none'}; effective_at={event.timestamp or 'now'}]."
                    else:
                        visible = f"{operation.title()} {task.target_objects[0].canonical_id} with value {json.dumps(value, ensure_ascii=False)}."
                    result = adapter.ingest_event(event.model_copy(update={"raw_text": visible, "normalized_text": visible}))
                    stage = "state_export"
                    effective = result.effective_action.operation.value
                    affected_entry_ids.extend(result.affected_entry_ids)
                    extractions.append({"event_id": event.event_id, "operation": parsed["operation"], "effective_operation": effective, "output_sha256": sha256_bytes(raw_output.encode()), "generated_tokens": tokens, "latency_ms": latency})
                entries = adapter.export_entries().entries
                stage = "retrieval"
                gold = task.gold_evidence[0].answer
                answer = entries[0].value_candidate if len(entries) == 1 else None
                retrieval = adapter.retrieve(RetrievalRequestV3(query=task.queries[0], k=16)).trace.retrieved_entries
                row = _row(task, "PASS", state_accuracy=answer == gold, memory_size=len(entries), stable_id=bool(affected_entry_ids) and len(set(affected_entry_ids)) == 1, gold_retrieved=any(entry.value_candidate == gold for entry in retrieval), stale_retrieved=sum(entry.value_candidate != gold for entry in retrieval), extra={"gold_sha256": sha256_bytes(canonical_json_bytes(gold)), "extractions": extractions, "reconciliation_count": reconciliation_count, "affected_entry_ids": tuple(dict.fromkeys(affected_entry_ids)), "latency_ms": (time.monotonic() - started) * 1000})
            except Exception as exc:
                row = _row(task, "FAIL", error_class=type(exc).__name__, extra={"stage": stage, "error_detail": _safe_error_detail(exc), "extractions": extractions, "latency_ms": (time.monotonic() - started) * 1000})
            finally:
                if adapter is not None:
                    try:
                        stage = "namespace_cleanup"
                        adapter.reset(ResetRequestV3(namespace=namespace))
                    except Exception as exc:
                        close_error = "namespace_cleanup_failed"
                        cleanup_detail = _safe_error_detail(exc)
                    try:
                        stage = "adapter_close"
                        adapter.close()
                    except Exception as exc:
                        close_error = "adapter_close_failed"
                        cleanup_detail = _safe_error_detail(exc)
                if close_error is not None:
                    row = _row(task, "FAIL", error_class=close_error, extra={"stage": stage, "error_detail": cleanup_detail, "extractions": extractions, "reconciliation_count": reconciliation_count, "affected_entry_ids": tuple(dict.fromkeys(affected_entry_ids)), "latency_ms": (time.monotonic() - started) * 1000})
            _append_row(rows_path, row); rows.append(row)
    finally:
        extractor.close()
    if len(rows) != requested:
        raise RuntimeError(f"{scope} terminal row count is incomplete")
    model_summary = {"snapshot": model_provenance["snapshot"], "tree_sha256": model_provenance["tree_sha256"], "snapshot_binding": model_provenance["snapshot_binding"], "runtime_receipt_sha256": model_provenance["runtime_receipt_sha256"], "runtime_receipt_schema": model_provenance["runtime_identity"], "dtype": "bf16", "decoding": "greedy", "attn_implementation": "eager", "trust_remote_code": False, "thinking_enabled": False, "timeout_seconds": 60.0, "seed": None, "device": "cuda:0", "runtime_executable": str(Path(sys.executable).resolve())}
    summary = build_summary(rows, scope=scope, requested=requested, rows_sha256=sha256_bytes(rows_path.read_bytes()), qualification_hashes=qualification["hashes"], qualification_identity={"package": qualification["closure"].get("identity"), "source": qualification["closure"].get("source"), "project_source": qualification["closure"].get("project_source"), "runtime": qualification["closure"].get("runtime")}, letta_binding=letta_binding, endpoint=endpoint, model_provenance=model_summary, execution_mode="injected_test_only" if adapter_factory is not None else "production")
    if scan_for_secrets(summary):
        raise ValueError("canary receipt failed secret scan")
    summary["payload_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    receipt_name = "canary_receipt.json" if scope == CANARY_SCOPE else "full_family_a_receipt.json"
    _write_new(output / receipt_name, canonical_json_bytes(summary))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded authenticated Letta plus Qwen extraction canary.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qualification-root", required=True)
    parser.add_argument("--letta-base-url", required=True)
    parser.add_argument("--letta-python-executable", required=True)
    parser.add_argument("--letta-project-root", required=True)
    parser.add_argument("--expected-letta-project-revision")
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--model-runtime-receipt", required=True)
    parser.add_argument("--model-snapshot-binding", required=True)
    parser.add_argument("--scope", choices=(CANARY_SCOPE, FULL_SCOPE), default=CANARY_SCOPE)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["outcome"] == "PASS" else 1
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {re.sub(r'[^a-zA-Z0-9_. -]', '', str(exc))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
