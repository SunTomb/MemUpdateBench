from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
import sys
import time
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.enums import AnswerDisposition, EvaluationMode
from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3, ResetRequestV3, RetrievalRequestV3
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey, typed_json_equal
from mub.vnext.external.bridge import JsonlSubprocessBridge
from mub.vnext.external.providers.letta import (
    build_letta_adapter_configuration,
    compute_letta_configuration_hash,
)
from mub.vnext.external.providers.letta_adapter import LettaExternalAdapterV3
from mub.vnext.external.security import scan_for_secrets
from mub.vnext.runtime.answer_model_v3 import (
    parse_answer_prediction_v3,
    render_visible_prompt_v3,
)

from scripts import vnext_run_letta_qwen_extraction_canary as extraction

TASK_SHA256 = extraction.TASK_SHA256
MODEL_ID = extraction.MODEL_ID
MODEL_REVISION = extraction.MODEL_REVISION
MODEL_TREE_SHA256 = extraction.MODEL_TREE_SHA256
MODEL_RUNTIME_RECEIPT_SHA256 = extraction.MODEL_RUNTIME_RECEIPT_SHA256
MODEL_SNAPSHOT_PATH = extraction.MODEL_SNAPSHOT_PATH
SCHEMA_VERSION = "memupdatebench.external.letta-qwen-prompted-answer.canary.v1"
FULL_SCHEMA_VERSION = "memupdatebench.external.letta-qwen-prompted-answer.full-family-a.v1"
CANARY_SCOPE = extraction.CANARY_SCOPE
FULL_SCOPE = extraction.FULL_SCOPE
ANSWER_OUTCOMES = ("CORRECT", "WRONG", "FORMAT_INVALID", "UNAVAILABLE")
ROW_SCHEMA_VERSION = "memupdatebench.external.letta-qwen-prompted-answer.row.v1"
ROW_NULLABLE_FIELDS = (
    "parsed_final_value", "stable_entry_id", "stale_retrieved_k16", "retrieval_trace_sha256",
    "visible_prompt_sha256", "prompted_answer", "prompted_exact_match", "answer_outcome", "answer_f1",
    "answer_format_valid", "answer_disposition", "answer_error_flags", "answer_output_sha256",
    "answer_metadata", "letta_configuration_hash", "final_memory_size", "state_accuracy",
    "gold_retrieved_k16",
    "gold_sha256", "extractions", "reconciliation_count", "affected_entry_ids", "latency_ms",
)
PASS_REQUIRED_FIELDS = (
    "state_accuracy", "final_memory_size", "stable_entry_id", "gold_retrieved_k16",
    "stale_retrieved_k16", "retrieval_trace_sha256", "visible_prompt_sha256",
    "prompted_exact_match", "answer_outcome", "answer_f1", "answer_format_valid",
    "answer_disposition", "answer_output_sha256", "answer_metadata", "gold_sha256",
    "extractions", "reconciliation_count", "affected_entry_ids", "latency_ms",
)
ROW_HASH_FIELDS = (
    "retrieval_trace_sha256", "visible_prompt_sha256", "answer_output_sha256",
    "gold_sha256", "letta_configuration_hash",
)

LLM_ROLES = ["visible_event_crud_extraction", "retrieved_context_prompted_answer"]

canonical_json_bytes = extraction.canonical_json_bytes
sha256_bytes = extraction.sha256_bytes
select_tasks = extraction.select_tasks
validate_output_root = extraction.validate_output_root
validate_qualification_artifacts = extraction.validate_qualification_artifacts
validate_worker_runtime_binding = extraction.validate_worker_runtime_binding
validate_loopback_binding = extraction.validate_loopback_binding
verify_model_provenance = extraction.verify_model_provenance
build_worker_command = extraction.build_worker_command
safe_worker_environment = extraction.safe_worker_environment
validate_extraction = extraction.validate_extraction


class PromptedAnswerModel(Protocol):
    def load(self) -> None: ...

    def answer(self, request: PromptedAnswerRequestV3): ...

    def close(self) -> None: ...


def build_prompted_answer_request(query, retrieval_trace) -> PromptedAnswerRequestV3:
    """Bind the visible prompt and its hash to the actual Letta retrieval trace."""
    rendered_prompt = render_visible_prompt_v3(
        query=query,
        retrieval_trace=retrieval_trace,
    )
    prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    prompted_trace = retrieval_trace.model_copy(update={"prompt_hash": prompt_hash})
    return PromptedAnswerRequestV3(
        query=query,
        retrieval_trace=prompted_trace,
        rendered_prompt=rendered_prompt,
        prompt_hash=prompt_hash,
    )


def answer_from_retrieval(model: PromptedAnswerModel, query, retrieval_trace):
    request = build_prompted_answer_request(query, retrieval_trace)
    prediction = model.answer(request)
    if prediction.query_id != query.query_id:
        raise ValueError("prompted answer prediction query_id must match query")
    return prediction


def classify_answer_prediction(prediction, gold) -> str:
    if prediction.disposition is not AnswerDisposition.ANSWERED:
        return "UNAVAILABLE"
    if not prediction.format_valid:
        return "FORMAT_INVALID"
    return "CORRECT" if typed_json_equal(prediction.parsed_answer, gold) else "WRONG"


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _answer_f1(prediction, gold) -> float:
    if prediction.disposition is not AnswerDisposition.ANSWERED or not prediction.format_valid:
        return 0.0
    if typed_json_equal(prediction.parsed_answer, gold):
        return 1.0
    predicted_tokens = str(_plain(prediction.parsed_answer)).casefold().split()
    gold_tokens = str(_plain(gold)).casefold().split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def task_support_reason(task) -> str | None:
    if len(task.target_objects) != 1:
        return "single_target_object_required"
    if len(task.queries) != 1:
        return "single_query_required"
    query = task.queries[0]
    mode = getattr(query.evaluation_mode, "value", query.evaluation_mode)
    query_type = getattr(query.query_type, "value", query.query_type)
    selector_kind = getattr(query.selector, "kind", None)
    if (
        mode != EvaluationMode.RETRIEVED_PROMPT.value
        or query_type != "current"
        or selector_kind != "current"
        or len(query.target_object_keys) != 1
    ):
        return "current_single_object_retrieved_prompt_required"
    return None


def safe_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _load_model_pair(model, extractor) -> None:
    try:
        model.load()
        if extractor is not model:
            extractor.load()
    except Exception:
        closed_ids = set()
        for resource in (extractor, model):
            if resource is None or id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        raise


def _close_resources(model, extractor) -> None:
    first_error = None
    if extractor is not None and extractor is not model:
        close = getattr(extractor, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                first_error = exc
    if model is not None:
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                first_error = first_error or exc
    if first_error is not None:
        raise first_error


def _canonical_model_sha256(value) -> str:
    return sha256_bytes(canonical_json_bytes(value.model_dump(mode="json")))


def _answer_metadata_or_empty(model) -> dict:
    metadata = getattr(model, "last_answer_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def validate_terminal_rows(rows: list[dict], *, scope: str, execution_mode: str = "injected_test_only") -> None:
    requested = 80 if scope == FULL_SCOPE else 32
    if len(rows) != requested:
        raise ValueError(f"{scope} must have exactly {requested} terminal rows")
    task_ids = [row.get("task_id") for row in rows]
    if any(type(task_id) is not str or not task_id for task_id in task_ids):
        raise ValueError("terminal rows require nonblank task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("terminal rows must contain exactly one row per task")
    if any(row.get("status") not in {"PASS", "FAIL", "NOT_SUPPORTED"} for row in rows):
        raise ValueError("terminal rows contain an invalid status")
    if any("row_schema_version" in row for row in rows):
        for row in rows:
            validate_row_shape(row, execution_mode=execution_mode)
    if scope == FULL_SCOPE:
        supported = sum(row.get("status") != "NOT_SUPPORTED" for row in rows)
        unsupported = sum(row.get("status") == "NOT_SUPPORTED" for row in rows)
        if supported != 52 or unsupported != 28:
            raise ValueError("full-family-a80 requires 52 supported and 28 NOT_SUPPORTED rows")
    else:
        supported = sum(row.get("status") != "NOT_SUPPORTED" for row in rows)
        unsupported = sum(row.get("status") == "NOT_SUPPORTED" for row in rows)
        if supported != 24 or unsupported != 8:
            raise ValueError("canary32 requires 24 supported and 8 NOT_SUPPORTED rows")


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _validate_nested_hash_fields(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_label = f"{label}.{key}"
            if str(key).endswith("_sha256") and item is not None:
                _require_digest(item, child_label)
            _validate_nested_hash_fields(item, child_label)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_nested_hash_fields(item, f"{label}[{index}]")


def _validate_pass_answer_coherence(row: dict) -> None:
    outcome = row["answer_outcome"]
    disposition = row["answer_disposition"]
    if outcome not in ANSWER_OUTCOMES:
        raise ValueError("answer outcome is invalid")
    valid_dispositions = {item.value for item in AnswerDisposition}
    if disposition not in valid_dispositions:
        raise ValueError("answer outcome has invalid disposition")
    if outcome == "CORRECT":
        coherent = (
            disposition == AnswerDisposition.ANSWERED.value
            and row["answer_format_valid"] is True
            and row["prompted_exact_match"] is True
        )
    elif outcome == "WRONG":
        coherent = (
            disposition == AnswerDisposition.ANSWERED.value
            and row["answer_format_valid"] is True
            and row["prompted_exact_match"] is False
        )
    elif outcome == "FORMAT_INVALID":
        coherent = (
            disposition == AnswerDisposition.ANSWERED.value
            and row["answer_format_valid"] is False
            and row["prompted_exact_match"] is False
        )
    else:
        coherent = (
            disposition in {AnswerDisposition.ABSTAINED.value, AnswerDisposition.UNAVAILABLE.value}
            and row["prompted_exact_match"] is False
        )
    if not coherent:
        raise ValueError("answer outcome is incoherent with answer evidence")


def _execution_mode(*, extractor_factory, adapter_factory, answer_model_factory) -> str:
    return (
        "production"
        if extractor_factory is None and adapter_factory is None and answer_model_factory is None
        else "injected_test_only"
    )


def write_artifact_no_replace(path: Path, value: object) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if scan_for_secrets(value):
        raise ValueError("artifact failed secret scan")
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_bytes(raw)


def claim_rows_file(path: Path) -> None:
    """Claim the rows artifact before any append, without replacing a race winner."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _append_row(path: Path, row: dict) -> None:
    if scan_for_secrets(row):
        raise ValueError("row failed secret scan")
    raw = canonical_json_bytes(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


class QwenSession:
    """One deterministic offline Qwen session with extraction and answer roles."""

    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot).resolve(strict=True)
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.last_answer_metadata = {}

    def load(self) -> None:
        safe_offline_environment()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.snapshot), revision=MODEL_REVISION, local_files_only=True,
                trust_remote_code=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.snapshot), revision=MODEL_REVISION, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation="eager",
            )
            self.model = self.model.eval()
        except Exception:
            self.close()
            raise

    def _generate(self, rendered: str, max_new_tokens: int) -> tuple[str, int, float]:
        encoded = self.tokenizer(rendered, return_tensors="pt").to("cuda:0")
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            generated[0, encoded.input_ids.shape[-1]:], skip_special_tokens=True,
        ).strip()
        return raw, int(generated.shape[-1] - encoded.input_ids.shape[-1]), (time.monotonic() - started) * 1000

    def extract(self, raw_text: str, attribute: str):
        prompt = extraction.build_extraction_prompt(raw_text, attribute)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        raw, tokens, latency = self._generate(rendered, 96)
        parsed = validate_extraction(json.loads(raw))
        return parsed, raw, tokens, latency

    def answer(self, request: PromptedAnswerRequestV3):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": request.rendered_prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        raw, tokens, latency = self._generate(rendered, 64)
        self.last_answer_metadata = {
            "rendered_chat_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
            "rendered_prompt_sha256": sha256_bytes(request.rendered_prompt.encode("utf-8")),
            "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
            "generated_tokens": tokens,
            "latency_ms": latency,
            "chat_template_sha256": sha256_bytes((self.tokenizer.chat_template or "").encode("utf-8")),
        }
        prediction = parse_answer_prediction_v3(
            query_id=request.query.query_id,
            answer_schema=request.query.answer_schema,
            raw_output=raw,
        )
        return prediction.model_copy(update={"latency_ms": latency, "usage": {"generated_tokens": tokens}})

    def close(self) -> None:
        model, tokenizer, torch = self.model, self.tokenizer, self.torch
        self.model = None
        self.tokenizer = None
        self.torch = None
        closed_ids = set()
        for resource in (model, tokenizer):
            if resource is None or id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if torch is not None:
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass


def validate_row_shape(row: dict, *, execution_mode: str = "production") -> None:
    if type(row) is not dict:
        raise ValueError("row must be an object")
    required = {"row_schema_version", "task_id", "semantic_core_id", "status", *ROW_NULLABLE_FIELDS}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"row is missing required fields: {', '.join(missing)}")
    if row["row_schema_version"] != ROW_SCHEMA_VERSION:
        raise ValueError("row schema version mismatch")
    if execution_mode not in {"production", "injected_test_only"}:
        raise ValueError("execution mode is invalid")
    status = row["status"]
    if status == "NOT_SUPPORTED":
        if any(row[field] is not None for field in ROW_NULLABLE_FIELDS):
            raise ValueError("NOT_SUPPORTED rows require null state, retrieval, and answer fields")
    elif status in {"PASS", "FAIL"}:
        for field in ROW_HASH_FIELDS:
            if row[field] is not None:
                _require_digest(row[field], field)
        _validate_nested_hash_fields(row["answer_metadata"], "answer_metadata")
        _validate_nested_hash_fields(row["extractions"], "extractions")
        if execution_mode == "production" and row["letta_configuration_hash"] is None:
            raise ValueError("production supported rows require Letta configuration hash")
        if execution_mode == "injected_test_only" and row["letta_configuration_hash"] is not None:
            raise ValueError("injected supported rows must not carry Letta configuration hash")
        if status == "PASS":
            missing_evidence = [field for field in PASS_REQUIRED_FIELDS if row[field] is None]
            if missing_evidence:
                raise ValueError(
                    "PASS rows require non-null completed evidence: " + ", ".join(missing_evidence)
                )
            _validate_pass_answer_coherence(row)
    else:
        raise ValueError("row status is invalid")


def _row(task, status: str, *, execution_mode: str = "injected_test_only", **fields) -> dict:
    row = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "task_id": task.task_id,
        "semantic_core_id": extraction.task_core(task),
        "status": status,
        "reason": None,
        "support_status": None,
        "error_class": None,
        "stage": None,
        "error_detail": None,
        **{field: None for field in ROW_NULLABLE_FIELDS},
    }
    row.update(fields)
    validate_row_shape(row, execution_mode=execution_mode)
    if scan_for_secrets(row):
        raise ValueError("row failed secret scan")
    return row


def _visible_action(operation: str, value, target_id: str, event) -> str:
    if operation == "noop":
        return "No memory object changes."
    if operation == "delete":
        logical = event.timestamp or "none"
        effective = event.timestamp or "now"
        return f"Delete {target_id} [scope=object; enumerated_targets={target_id}; event_logical_time={logical}; effective_at={effective}]."
    return f"{operation.title()} {target_id} with value {json.dumps(value, ensure_ascii=False)}."


def _task_letta_configuration_hash(task_id: str) -> str:
    return compute_letta_configuration_hash(
        build_letta_adapter_configuration(run_id="letta-qwen-prompted-" + task_id)
    )


def _adapter_for_task(args, task, binding: dict, endpoint: str):
    config = build_letta_adapter_configuration(run_id="letta-qwen-prompted-" + task.task_id)
    config_json = canonical_json_bytes(config.model_dump(mode="json")).decode("utf-8")
    command = build_worker_command(args.letta_python_executable, args.letta_project_root, config_json)
    environment = safe_worker_environment(endpoint, Path(binding["project_root"]))
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    bridge = JsonlSubprocessBridge(
        command=command,
        cwd=binding["project_root"],
        environment=environment,
        timeout_seconds=60.0,
    )
    key = FrozenMemoryObjectKey.model_validate(
        task.target_objects[0].model_dump(mode="python"), strict=True,
    )
    return LettaExternalAdapterV3(
        bridge=bridge, configuration=config,
        target_objects=(key,),
    )


def build_summary(
    rows: list[dict], *, scope: str, requested: int, rows_sha256: str,
    qualification_hashes: dict, qualification_identity: dict, letta_binding: dict,
    endpoint: str, model_provenance: dict, execution_mode: str = "production",
) -> dict:
    supported = [row for row in rows if row["status"] != "NOT_SUPPORTED"]
    for row in supported:
        answer_outcome = row.get("answer_outcome")
        if answer_outcome is not None and answer_outcome not in ANSWER_OUTCOMES:
            raise ValueError("answer outcome is invalid")
    state_rows = [row for row in supported if row.get("state_accuracy") is not None]
    retrieval_rows = [row for row in supported if row.get("gold_retrieved_k16") is not None]
    answer_rows = [row for row in supported if row.get("answer_outcome") in ANSWER_OUTCOMES]
    answer_evaluable_rows = [row for row in answer_rows if row.get("answer_outcome") in {"CORRECT", "WRONG"}]
    memory_rows = [row for row in supported if row.get("final_memory_size") is not None]
    outcome_counts = {outcome: sum(row.get("answer_outcome") == outcome for row in answer_rows) for outcome in ANSWER_OUTCOMES}
    state_correct = sum(bool(row["state_accuracy"]) for row in state_rows)
    retrieval_correct = sum(bool(row["gold_retrieved_k16"]) for row in retrieval_rows)
    exact = outcome_counts["CORRECT"]
    f1_values = [
        float(row.get("answer_f1", 1.0 if row.get("answer_outcome") == "CORRECT" else 0.0))
        for row in answer_rows
    ]
    operations = {name: 0 for name in ("add", "update", "noop", "delete")}
    effective_operations = dict(operations)
    reconciliation_count = 0
    configuration_hashes = {}
    typed_rows = any("row_schema_version" in row for row in rows)
    for row in rows:
        if typed_rows:
            validate_row_shape(row, execution_mode=execution_mode)
        if row.get("status") != "NOT_SUPPORTED" and execution_mode == "production" and row.get("letta_configuration_hash") is not None:
            configuration_hashes[row["task_id"]] = row["letta_configuration_hash"]
        reconciliation_count += int(row.get("reconciliation_count") or 0)
        for item in row.get("extractions") or ():
            operation = str(item.get("operation", "")).lower()
            effective = str(item.get("effective_operation", "")).lower()
            if operation in operations:
                operations[operation] += 1
            if effective in effective_operations:
                effective_operations[effective] += 1
    summary = {
        "schema_version": FULL_SCHEMA_VERSION if scope == FULL_SCOPE else SCHEMA_VERSION,
        "scope": scope,
        "evidence_class": "joint_pipeline",
        "outcome": "PASS" if not [row for row in supported if row["status"] == "FAIL"] else "FAIL",
        "requested": requested,
        "terminal_rows": len(rows),
        "supported": len(supported),
        "unsupported": len(rows) - len(supported),
        "pass": sum(row["status"] == "PASS" for row in supported),
        "fail": sum(row["status"] == "FAIL" for row in supported),
        "state_accuracy": state_correct / len(state_rows) if state_rows else None,
        "state_accuracy_denominator": len(state_rows),
        "gold_retrieval_rate": retrieval_correct / len(retrieval_rows) if retrieval_rows else None,
        "gold_retrieval_denominator": len(retrieval_rows),
        "prompted_exact_match": exact / len(answer_rows) if answer_rows else None,
        "prompted_answer_em": exact / len(answer_rows) if answer_rows else None,
        "prompted_answer_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "answer_attempted_denominator": len(answer_rows),
        "answer_evaluable_denominator": len(answer_evaluable_rows),
        "answer_metrics_denominator": len(answer_rows),
        "prompted_metrics_denominator": len(answer_rows),
        "answer_outcome_counts": outcome_counts,
        "avg_memory_size": sum(row["final_memory_size"] for row in memory_rows) / len(memory_rows) if memory_rows else None,
        "memory_size_denominator": len(memory_rows),
        "total_reconciliation_count": reconciliation_count,
        "operation_counts": {"requested": operations, "effective": effective_operations},
        "rows_sha256": _require_digest(rows_sha256, "rows_sha256"),
        "task_view_sha256": TASK_SHA256,
        "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "llm_roles": list(LLM_ROLES),
        "qualification_hashes": qualification_hashes,
        "qualification_identity": qualification_identity,
        "letta_configuration_hashes_by_task": configuration_hashes,
        "letta_configuration_hashes_sha256": sha256_bytes(canonical_json_bytes(configuration_hashes)),
        "letta_worker_runtime": {
            **letta_binding,
            "cwd": letta_binding.get("project_root"),
            "environment": {"PYTHONPATH": letta_binding.get("project_root"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        },
        "letta_base_url": endpoint,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, **model_provenance},
        "provider_calls": 0,
        "api_calls": 0,
        "retries": 0,
        "answer_model_metrics": {
            "outcome_counts": outcome_counts,
            "attempted_denominator": len(answer_rows),
            "evaluable_denominator": len(answer_evaluable_rows),
            "f1_includes_invalid_and_unavailable_as_zero": True,
        },
        "execution_mode": execution_mode,
    }
    if scan_for_secrets(summary):
        raise ValueError("summary failed secret scan")
    return summary


def finalize_rows(tasks, rows: list[dict]) -> list[dict]:
    expected = [task.task_id for task in tasks]
    observed = [row.get("task_id") for row in rows]
    if len(observed) != len(expected) or len(observed) != len(set(observed)):
        raise ValueError("terminal rows must contain exactly one row per task")
    if observed != expected:
        raise ValueError("terminal rows must preserve task order")
    return rows


def run(args, *, extractor_factory: Callable[[], object] | None = None, adapter_factory=None, answer_model_factory: Callable[[], PromptedAnswerModel] | None = None) -> dict:
    output = validate_output_root(
        Path(args.output_root),
        frozen_roots=(ROOT / "data" / "vnext" / "core", ROOT / "data" / "vnext" / "phase0", ROOT / "configs" / "vnext" / "post_core"),
    )
    if output.exists():
        raise FileExistsError(output)
    binding_path = Path(args.model_snapshot_binding)
    if not binding_path.is_absolute() or binding_path.is_symlink() or not binding_path.is_file():
        raise ValueError("model snapshot binding receipt must be an absolute regular file")
    binding_path = binding_path.resolve(strict=True)
    binding_raw = binding_path.read_bytes()
    model_binding = json.loads(binding_raw)
    if canonical_json_bytes(model_binding) != binding_raw or scan_for_secrets(model_binding):
        raise ValueError("model snapshot binding must be canonical and secret-free")
    model_provenance = verify_model_provenance(
        Path(args.model_snapshot), Path(args.model_runtime_receipt), model_binding,
        binding_raw=binding_raw, binding_path=binding_path,
    )
    qualification = validate_qualification_artifacts(Path(args.qualification_root))
    letta_binding = validate_worker_runtime_binding(
        args.letta_python_executable, args.letta_project_root, qualification["closure"],
        expected_revision=getattr(args, "expected_letta_project_revision", None),
    )
    endpoint = validate_loopback_binding(args.letta_base_url, qualification["closure"])
    scope = getattr(args, "scope", FULL_SCOPE)
    task_bytes = Path(args.tasks).read_bytes()
    selected = select_tasks(task_bytes) if scope == CANARY_SCOPE else select_tasks(task_bytes, scope=scope)
    requested = len(selected)
    rows_path = output / "rows.jsonl"
    model = QwenSession(Path(args.model_snapshot)) if answer_model_factory is None else answer_model_factory()
    extractor = model if extractor_factory is None else extractor_factory()
    rows: list[dict] = []
    resources_loaded = False
    execution_mode = _execution_mode(
        extractor_factory=extractor_factory,
        adapter_factory=adapter_factory,
        answer_model_factory=answer_model_factory,
    )
    production = execution_mode == "production"
    try:
        _load_model_pair(model, extractor)
        resources_loaded = True
        output.mkdir(parents=True)
        claim_rows_file(rows_path)
        for task in selected:
            support_reason = task_support_reason(task)
            if support_reason is not None:
                row = _row(
                    task,
                    "NOT_SUPPORTED",
                    execution_mode=execution_mode,
                    reason=support_reason,
                    support_status="NOT_SUPPORTED",
                )
                _append_row(rows_path, row)
                rows.append(row)
                continue
            configuration_hash = _task_letta_configuration_hash(task.task_id) if production else None
            started = time.monotonic()
            adapter = None
            extraction_rows = []
            affected_entry_ids = []
            reconciliation_count = 0
            namespace = "letta_qwen_prompted_" + task.task_id
            try:
                adapter = _adapter_for_task(args, task, letta_binding, endpoint) if adapter_factory is None else adapter_factory(task)
                adapter.reset(ResetRequestV3(namespace=namespace))
                store_initialized = bool(adapter.export_entries().entries)
                for event in task.events:
                    parsed, raw_output, tokens, latency = extractor.extract(event.raw_text, task.target_objects[0].attribute)
                    parsed = validate_extraction(parsed)
                    requested_operation = parsed["operation"]
                    operation = requested_operation
                    if operation == "update" and not store_initialized:
                        operation = "add"
                        reconciliation_count += 1
                    visible = _visible_action(operation, parsed["value"], task.target_objects[0].canonical_id, event)
                    result = adapter.ingest_event(event.model_copy(update={"raw_text": visible, "normalized_text": visible}))
                    effective = result.effective_action.operation.value.lower() if result.effective_action.operation is not None else "noop"
                    if effective in {"add", "update"}:
                        store_initialized = True
                    elif effective == "delete":
                        store_initialized = False
                    affected_entry_ids.extend(result.affected_entry_ids)
                    extraction_rows.append({"event_id": event.event_id, "operation": requested_operation, "effective_operation": effective, "output_sha256": sha256_bytes(raw_output.encode()), "generated_tokens": tokens, "latency_ms": latency})
                entries = adapter.export_entries().entries
                gold = task.gold_evidence[0].answer
                state_value = entries[0].value_candidate if len(entries) == 1 else None
                retrieval = adapter.retrieve(RetrievalRequestV3(query=task.queries[0], k=16))
                trace = retrieval.trace
                gold_retrieved = any(typed_json_equal(entry.value_candidate, gold) for entry in trace.retrieved_entries)
                request = build_prompted_answer_request(task.queries[0], trace)
                prediction = model.answer(request)
                if prediction.query_id != task.queries[0].query_id:
                    raise ValueError("prompted answer prediction query_id must match query")
                answer_outcome = classify_answer_prediction(prediction, gold)
                answer_metadata = _answer_metadata_or_empty(model)
                row = _row(
                    task, "PASS", execution_mode=execution_mode,
                    state_accuracy=typed_json_equal(state_value, gold),
                    letta_configuration_hash=configuration_hash,
                    parsed_final_value=state_value,
                    final_memory_size=len(entries),
                    stable_entry_id=bool(affected_entry_ids) and len(set(affected_entry_ids)) == 1,
                    gold_retrieved_k16=gold_retrieved,
                    stale_retrieved_k16=sum(not typed_json_equal(entry.value_candidate, gold) for entry in trace.retrieved_entries),
                    retrieval_trace_sha256=_canonical_model_sha256(request.retrieval_trace),
                    visible_prompt_sha256=request.prompt_hash,
                    prompted_answer=prediction.parsed_answer,
                    prompted_exact_match=answer_outcome == "CORRECT",
                    answer_outcome=answer_outcome,
                    answer_format_valid=prediction.format_valid,
                    answer_disposition=prediction.disposition.value,
                    answer_error_flags=prediction.error_flags,
                    answer_output_sha256=sha256_bytes(prediction.raw_output.encode()),
                    answer_metadata=answer_metadata,
                    answer_f1=_answer_f1(prediction, gold),
                    gold_sha256=sha256_bytes(canonical_json_bytes(gold)),
                    extractions=extraction_rows,
                    reconciliation_count=reconciliation_count,
                    affected_entry_ids=tuple(dict.fromkeys(affected_entry_ids)),
                    latency_ms=(time.monotonic() - started) * 1000,
                )
            except Exception as exc:
                row = _row(task, "FAIL", execution_mode=execution_mode, letta_configuration_hash=configuration_hash, error_class=type(exc).__name__, stage="joint_pipeline", error_detail=re.sub(r"[^a-zA-Z0-9_. -]", "", str(exc))[:240], extractions=extraction_rows, reconciliation_count=reconciliation_count, latency_ms=(time.monotonic() - started) * 1000)
            finally:
                if adapter is not None:
                    cleanup_error = None
                    try:
                        adapter.reset(ResetRequestV3(namespace=namespace))
                    except Exception as exc:
                        cleanup_error = exc
                    try:
                        adapter.close()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
                    if cleanup_error is not None:
                        row = _row(task, "FAIL", execution_mode=execution_mode, letta_configuration_hash=configuration_hash, error_class="cleanup_failed", stage="cleanup", error_detail=type(cleanup_error).__name__, extractions=extraction_rows, reconciliation_count=reconciliation_count, latency_ms=(time.monotonic() - started) * 1000)
            _append_row(rows_path, row)
            rows.append(row)
    finally:
        if resources_loaded:
            _close_resources(model, extractor)
    rows = finalize_rows(selected, rows)
    validate_terminal_rows(rows, scope=scope, execution_mode=execution_mode)
    summary = build_summary(
        rows, scope=scope, requested=requested,
        rows_sha256=sha256_bytes(rows_path.read_bytes()),
        qualification_hashes=qualification["hashes"],
        qualification_identity={"package": qualification["closure"].get("identity"), "source": qualification["closure"].get("source"), "project_source": qualification["closure"].get("project_source"), "runtime": qualification["closure"].get("runtime")},
        letta_binding=letta_binding, endpoint=endpoint,
        model_provenance={"snapshot": model_provenance["snapshot"], "tree_sha256": model_provenance["tree_sha256"], "snapshot_binding": model_provenance["snapshot_binding"], "runtime_receipt_sha256": model_provenance["runtime_receipt_sha256"], "runtime_receipt_schema": model_provenance["runtime_identity"], "dtype": "bf16", "decoding": "greedy", "attn_implementation": "eager", "trust_remote_code": False, "thinking_enabled": False, "device": "cuda:0", "runtime_executable": str(Path(sys.executable).resolve())},
        execution_mode=execution_mode,
    )
    receipt_name = "full_family_a_receipt.json" if scope == FULL_SCOPE else "canary_receipt.json"
    receipt_hash = write_artifact_no_replace(output / receipt_name, summary)
    index = {
        "schema_version": "memupdatebench.external.letta-qwen-prompted-answer.artifact-index.v1",
        "scope": scope,
        "evidence_class": "joint_pipeline",
        "artifacts": {"rows.jsonl": {"sha256": sha256_bytes(rows_path.read_bytes()), "bytes": rows_path.stat().st_size}, receipt_name: {"sha256": receipt_hash, "bytes": (output / receipt_name).stat().st_size}},
        "task_view_sha256": TASK_SHA256,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tree_sha256": model_provenance["tree_sha256"],
            "snapshot_binding_receipt_sha256": model_provenance.get("snapshot_binding", {}).get("receipt_file_sha256"),
            "snapshot_binding_payload_sha256": model_provenance.get("snapshot_binding", {}).get("receipt_payload_sha256"),
            "runtime_receipt_sha256": model_provenance["runtime_receipt_sha256"],
        },
        "qualification_hashes": qualification["hashes"],
        "qualification_identity": {"package": qualification["closure"].get("identity"), "source": qualification["closure"].get("source"), "project_source": qualification["closure"].get("project_source"), "runtime": qualification["closure"].get("runtime")},
        "letta_configuration_hashes_by_task": summary["letta_configuration_hashes_by_task"],
        "letta_configuration_hashes_sha256": summary["letta_configuration_hashes_sha256"],
        "worker_binding": letta_binding,
        "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "llm_roles": list(LLM_ROLES),
        "provider_calls": 0,
        "api_calls": 0,
        "retries": 0,
    }
    write_artifact_no_replace(output / "artifact_index.json", index)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the authenticated Letta plus Qwen prompted-answer Family-A evaluation.")
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
    parser.add_argument("--scope", choices=(CANARY_SCOPE, FULL_SCOPE), default=FULL_SCOPE)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {re.sub(r'[^a-zA-Z0-9_. -]', '', str(exc))}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
