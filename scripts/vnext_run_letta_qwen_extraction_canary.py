from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3

TASK_SHA256 = "ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE_SHA256 = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
MODEL_RUNTIME_RECEIPT_SHA256 = "5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f"
MODEL_RUNTIME_IDENTITY = "transformers-qwen35-9b-bf16-eager"
# Short aliases retained for provenance-oriented callers.
TASK_SHA = TASK_SHA256
MODEL = MODEL_ID
REV = MODEL_REVISION
SCHEMA_VERSION = "memupdatebench.external.letta-qwen-extraction.canary.v2"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def task_core(task: MemUpdateTaskV3) -> str:
    return task.metadata.extra["semantic_core_id"]


def select_tasks(raw: bytes) -> list[MemUpdateTaskV3]:
    if sha256_bytes(raw) != TASK_SHA256:
        raise ValueError("task view SHA-256 mismatch")
    tasks = [MemUpdateTaskV3.model_validate(line, strict=True) for line in (json.loads(x) for x in raw.splitlines() if x.strip())]
    tasks.sort(key=lambda task: (task_core(task).encode("utf-8"), task.task_id.encode("utf-8")))
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
    if not root.is_dir() or _is_reparse(root):
        raise ValueError("model snapshot must be a real directory")
    files = []
    for path in root.rglob("*"):
        if _is_reparse(path):
            raise ValueError("model snapshot contains symlink or reparse point")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError("model snapshot contains non-regular entry")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big")); digest.update(rel)
        digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest()


def verify_model_provenance(snapshot: Path, runtime_receipt: Path) -> dict:
    snapshot = snapshot.resolve(strict=True)
    if _is_reparse(snapshot):
        raise ValueError("model snapshot must not be symlink or reparse")
    tree_hash = stable_tree_sha256(snapshot)
    if tree_hash != MODEL_TREE_SHA256:
        raise ValueError("model snapshot tree hash mismatch")
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
    return {"snapshot": str(snapshot), "tree_sha256": tree_hash, "runtime_receipt_sha256": sha256_bytes(raw), "runtime_identity": value["schema_version"], "runtime_receipt": value}


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
    required_closure = ("candidate_id", "identity", "source", "project_source", "runner_source_sha256", "runtime", "boundary", "cleanup", "preflight", "admission")
    if any(field not in closure for field in required_closure):
        raise ValueError("qualification closure is incomplete")
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


def safe_worker_environment(base_url: str, project_root: Path) -> dict[str, str]:
    base_url = validate_loopback_url(base_url)
    if not project_root.is_absolute() or not project_root.is_dir() or project_root.is_symlink():
        raise ValueError("worker project root must be a real absolute directory")
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"}
    env = {name: value for name, value in os.environ.items() if name in allowed and type(value) is str}
    env.update({"LETTA_NATIVE_API_BASE_URL": base_url, "PYTHONPATH": str(project_root.resolve()), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
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


def run(args, *, extractor_factory: Callable[[], ModelExtractor] | None = None, adapter_factory=None) -> dict:
    tasks_path = Path(args.tasks)
    output = validate_output_root(Path(args.output_root), frozen_roots=(ROOT / "data" / "vnext" / "core", ROOT / "data" / "vnext" / "phase0", ROOT / "configs" / "vnext" / "post_core"))
    if output.exists():
        raise FileExistsError(output)
    model_provenance = verify_model_provenance(Path(args.model_snapshot), Path(args.model_runtime_receipt))
    qualification = validate_qualification_artifacts(Path(args.qualification_root))
    endpoint = validate_loopback_binding(args.letta_base_url, qualification["closure"])
    selected = select_tasks(tasks_path.read_bytes())
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
            affected_entry_ids = []
            reconciliation_count = 0
            namespace = "letta_qwen_" + task.task_id
            try:
                if adapter_factory is None:
                    config = build_letta_adapter_configuration(run_id="letta-qwen-" + task.task_id)
                    config_json = canonical_json_bytes(config).decode("utf-8")
                    command = (sys.executable, str(ROOT / "mub" / "vnext" / "external" / "workers" / "letta_worker.py"), "--configuration-json", config_json)
                    bridge = JsonlSubprocessBridge(command=command, cwd=ROOT, environment=safe_worker_environment(endpoint, ROOT), timeout_seconds=60.0)
                    adapter = LettaExternalAdapterV3(bridge=bridge, configuration=config, target_objects=(FrozenMemoryObjectKey.model_validate(task.target_objects[0].model_dump(mode="python"), strict=True),))
                else:
                    adapter = adapter_factory(task)
                namespace = "letta_qwen_" + task.task_id
                adapter.reset(ResetRequestV3(namespace=namespace))
                for event in task.events:
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
                    effective = result.effective_action.operation.value
                    affected_entry_ids.extend(result.affected_entry_ids)
                    extractions.append({"event_id": event.event_id, "operation": parsed["operation"], "effective_operation": effective, "output_sha256": sha256_bytes(raw_output.encode()), "generated_tokens": tokens, "latency_ms": latency})
                entries = adapter.export_entries().entries
                gold = task.gold_evidence[0].answer
                answer = entries[0].value_candidate if len(entries) == 1 else None
                retrieval = adapter.retrieve(RetrievalRequestV3(query=task.queries[0], k=16)).trace.retrieved_entries
                row = _row(task, "PASS", state_accuracy=answer == gold, memory_size=len(entries), stable_id=bool(affected_entry_ids) and len(set(affected_entry_ids)) == 1, gold_retrieved=any(entry.value_candidate == gold for entry in retrieval), stale_retrieved=sum(entry.value_candidate != gold for entry in retrieval), extra={"gold_sha256": sha256_bytes(canonical_json_bytes(gold)), "extractions": extractions, "reconciliation_count": reconciliation_count, "affected_entry_ids": tuple(dict.fromkeys(affected_entry_ids)), "latency_ms": (time.monotonic() - started) * 1000})
            except Exception as exc:
                row = _row(task, "FAIL", error_class=type(exc).__name__, extra={"extractions": extractions, "latency_ms": (time.monotonic() - started) * 1000})
            finally:
                if adapter is not None:
                    try:
                        adapter.reset_namespace(namespace)
                    except Exception:
                        close_error = "namespace_cleanup_failed"
                    try:
                        adapter.close()
                    except Exception:
                        close_error = "adapter_close_failed"
                if close_error is not None:
                    row = _row(task, "FAIL", error_class=close_error, extra={"extractions": extractions, "reconciliation_count": reconciliation_count, "affected_entry_ids": tuple(dict.fromkeys(affected_entry_ids)), "latency_ms": (time.monotonic() - started) * 1000})
            _append_row(rows_path, row); rows.append(row)
    finally:
        extractor.close()
    if len(rows) != 32:
        raise RuntimeError("canary terminal row count is incomplete")
    supported = [row for row in rows if row["status"] != "NOT_SUPPORTED"]
    passed = [row for row in supported if row["status"] == "PASS"]
    state_rows = [row for row in passed if row["state_accuracy"] is not None]
    summary = {"schema_version": SCHEMA_VERSION, "outcome": "PASS" if not [row for row in supported if row["status"] == "FAIL"] else "FAIL", "requested": 32, "terminal_rows": len(rows), "supported": len(supported), "unsupported": len(rows) - len(supported), "pass": len(passed), "fail": len(supported) - len(passed), "state_accuracy": sum(bool(row["state_accuracy"]) for row in state_rows) / len(state_rows) if state_rows else None, "state_accuracy_denominator": len(state_rows), "gold_retrieval_rate": sum(bool(row["gold_retrieved_k16"]) for row in passed) / len(passed) if passed else None, "avg_memory_size": sum(row["final_memory_size"] for row in passed) / len(passed) if passed else None, "rows_sha256": sha256_bytes(rows_path.read_bytes()), "task_view_sha256": TASK_SHA256, "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()), "qualification_hashes": qualification["hashes"], "qualification_identity": {"package": qualification["closure"].get("identity"), "source": qualification["closure"].get("source"), "project_source": qualification["closure"].get("project_source"), "runtime": qualification["closure"].get("runtime")}, "letta_base_url": endpoint, "letta_configuration_hash": compute_letta_configuration_hash(build_letta_adapter_configuration(run_id="letta-qwen-manifest")), "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": model_provenance["snapshot"], "tree_sha256": model_provenance["tree_sha256"], "runtime_receipt_sha256": model_provenance["runtime_receipt_sha256"], "runtime_receipt_schema": model_provenance["runtime_identity"], "dtype": "bf16", "decoding": "greedy", "attn_implementation": "eager", "trust_remote_code": False, "thinking_enabled": False, "timeout_seconds": 60.0, "seed": None, "device": "cuda:0"}, "provider_calls": 0, "api_calls": 0, "answer_model_metrics": None, "execution_mode": "injected_test_only" if adapter_factory is not None else "production"}
    if scan_for_secrets(summary):
        raise ValueError("canary receipt failed secret scan")
    summary["payload_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    _write_new(output / "canary_receipt.json", canonical_json_bytes(summary))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded authenticated Letta plus Qwen extraction canary.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qualification-root", required=True)
    parser.add_argument("--letta-base-url", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--model-runtime-receipt", required=True)
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
