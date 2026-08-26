from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mub.vnext.adapters.core_v3 import RawAppendAdapterV3
from mub.vnext.contracts.v3.adapter import ResetRequestV3, RetrievalRequestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.preparation.task12 import RawAppendTrajectoryV1
from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3, render_visible_prompt_v3
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3
from mub.vnext.runtime.task12_execution_v3 import execute_task12_task_v3, transform_retrieval_trace_v3

RELEASE_ID = "memupdatebench.post-core.qwen35-canary.v1"
TASK_VIEW_SHA256 = "ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6"
TRAJECTORIES_SHA256 = "c615ee14b556faab566dd9b902c56b5b3cf793f0e4c0426ef3ddd94398245d0a"
PREPARATION_SHA256 = "7ab4af67e3cf84e2fcba9baa9b7ea6ee9a768cf4c3defcdc36dea78c0278e542"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE_SHA256 = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
RUNTIME_RECEIPT_SHA256 = "5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f"
CONDITIONS = (
    ("reverse-none", "reverse_chronological", "none"),
    ("reverse-labeled", "reverse_chronological", "latest_outdated_label"),
)
SEEDS = (0, 1)
RETRIEVAL_K = 16
MAX_NEW_TOKENS = 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl(path: Path, model=None):
    rows = []
    for raw in path.read_bytes().splitlines():
        if not raw:
            raise ValueError(f"empty JSONL row: {path}")
        payload = json.loads(raw)
        rows.append(model.model_validate(payload) if model else payload)
    return tuple(rows)


def task_core_id(task: MemUpdateTaskV3) -> str:
    value = task.metadata.extra.get("semantic_core_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task missing semantic_core_id")
    return value


def task_sha(task: MemUpdateTaskV3) -> str:
    return sha256_bytes(canonical_bytes(task.model_dump(mode="json")))


def build_presented_trace(task: MemUpdateTaskV3, trajectory: RawAppendTrajectoryV1, condition):
    adapter = RawAppendAdapterV3(task, retrieval_policy="normal_topk")
    reset = adapter.reset(ResetRequestV3(namespace=f"canary_{task.task_id}"))
    if not reset.success:
        raise RuntimeError("raw append adapter reset failed")
    for event in task.events:
        adapter.ingest_event(event)
    query = task.queries[0]
    result = adapter.retrieve(RetrievalRequestV3(query=query, k=RETRIEVAL_K))
    full = adapter.export_entries().entries
    return transform_retrieval_trace_v3(
        result.trace,
        context_order=condition[1],
        context_annotation=condition[2],
        full_trajectory=full,
        frozen_trajectory=trajectory,
    )


def semantic_entry_key(entry) -> tuple[Any, ...]:
    return (
        entry.object_key_candidate.canonical_id if entry.object_key_candidate is not None else None,
        tuple(entry.source_event_ids),
        entry.version_index,
        json.dumps(entry.value_candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        entry.raw_metadata.get("event_index"),
    )


def prepare(args) -> int:
    tasks_path = Path(args.tasks)
    trajectories_path = Path(args.trajectories)
    prep_path = Path(args.preparation_manifest)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("canary output root already exists")
    if sha256_file(tasks_path) != TASK_VIEW_SHA256:
        raise ValueError("task view hash mismatch")
    if sha256_file(trajectories_path) != TRAJECTORIES_SHA256:
        raise ValueError("trajectory hash mismatch")
    if sha256_file(prep_path) != PREPARATION_SHA256:
        raise ValueError("preparation manifest hash mismatch")
    tasks = load_jsonl(tasks_path, MemUpdateTaskV3)
    trajectories = load_jsonl(trajectories_path, RawAppendTrajectoryV1)
    by_trajectory = {row.task_id: row for row in trajectories}
    if len(tasks) != 80 or len(by_trajectory) != 80 or set(by_trajectory) != {task.task_id for task in tasks}:
        raise ValueError("canary requires exact 80-task trajectory coverage")
    groups: dict[str, list[MemUpdateTaskV3]] = {}
    for task in tasks:
        if len(task.queries) != 1:
            raise ValueError("canary tasks require exactly one query")
        groups.setdefault(task_core_id(task), []).append(task)
    if len(groups) != 20 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("canary requires 20 semantic cores x 4 tasks")
    ordered = tuple(sorted(tasks, key=lambda task: (task_core_id(task).encode(), task.task_id.encode())))
    for task in ordered:
        traces = [build_presented_trace(task, by_trajectory[task.task_id], condition) for condition in CONDITIONS]
        if sorted(semantic_entry_key(entry) for entry in traces[0].retrieved_entries) != sorted(semantic_entry_key(entry) for entry in traces[1].retrieved_entries):
            raise ValueError("paired conditions changed retrieved semantic multiset")
        if any("version_label" in entry.raw_metadata for entry in traces[0].retrieved_entries):
            raise ValueError("no-label condition leaked version labels")
        if not all("version_label" in entry.raw_metadata for entry in traces[1].retrieved_entries):
            raise ValueError("labeled condition omitted version labels")
        render_visible_prompt_v3(query=task.queries[0], retrieval_trace=traces[0])
        render_visible_prompt_v3(query=task.queries[0], retrieval_trace=traces[1])
    script_hash = sha256_file(Path(__file__))
    plan = []
    ordinal = 0
    for seed in SEEDS:
        for condition in CONDITIONS:
            for task in ordered:
                ordinal += 1
                identity = {
                    "release_id": RELEASE_ID,
                    "task_view_sha256": TASK_VIEW_SHA256,
                    "trajectory_sha256": TRAJECTORIES_SHA256,
                    "semantic_core_id": task_core_id(task),
                    "task_id": task.task_id,
                    "task_sha256": task_sha(task),
                    "condition": condition[0],
                    "context_order": condition[1],
                    "context_annotation": condition[2],
                    "retrieval_policy": "normal_topk",
                    "retrieval_k": RETRIEVAL_K,
                    "seed": seed,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "model_tree_sha256": MODEL_TREE_SHA256,
                    "runtime_receipt_sha256": RUNTIME_RECEIPT_SHA256,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "enable_thinking": False,
                    "runner_sha256": script_hash,
                    "ordinal": ordinal,
                }
                plan.append({**identity, "call_id": sha256_bytes(canonical_bytes(identity))})
    if len(plan) != 320 or len({row["call_id"] for row in plan}) != 320:
        raise ValueError("canary plan cardinality mismatch")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": "memupdatebench.post-core.qwen35-canary-manifest.v1",
        "release_id": RELEASE_ID,
        "source_bindings": {
            "tasks_sha256": TASK_VIEW_SHA256,
            "trajectories_sha256": TRAJECTORIES_SHA256,
            "preparation_manifest_sha256": PREPARATION_SHA256,
            "runtime_receipt_sha256": RUNTIME_RECEIPT_SHA256,
            "runner_sha256": script_hash,
        },
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "tree_sha256": MODEL_TREE_SHA256},
        "conditions": [row[0] for row in CONDITIONS],
        "condition_assumption": "reverse/no-label versus reverse/latest-outdated-label at k=16; seed axis is deterministic repetition",
        "seeds": list(SEEDS),
        "retrieval_policy": "normal_topk",
        "retrieval_k": RETRIEVAL_K,
        "decode": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "enable_thinking": False, "dtype": "bf16", "attention": "eager"},
        "task_count": 80,
        "semantic_core_count": 20,
        "call_count": 320,
        "max_retries": 0,
        "scientific_status": "CANARY_ONLY",
    }
    write_exclusive(output / "canary_manifest.json", canonical_bytes(manifest))
    write_exclusive(output / "call_plan.jsonl", b"".join(canonical_bytes(row) + b"\n" for row in plan))
    write_exclusive(output / "progress.json", canonical_bytes({"completed": 0, "call_count": 320, "status": "PREPARED"}))
    print(json.dumps({"status": "PREPARED", "tasks": 80, "cores": 20, "calls": 320, "output_root": str(output)}, sort_keys=True))
    return 0


class QwenCanaryModel:
    def __init__(self, snapshot: str, seed: int, device: str):
        self.snapshot = snapshot
        self.seed = seed
        self.device = device
        self.last_meta: dict[str, Any] = {}

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)
        self.chat_template_sha256 = sha256_bytes((self.tokenizer.chat_template or "").encode("utf-8"))
        self.model = AutoModelForCausalLM.from_pretrained(self.snapshot, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()

    def answer(self, request):
        torch = self.torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": request.rendered_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        begin = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        new_tokens = generated[0][encoded.input_ids.shape[-1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        prediction = parse_answer_prediction_v3(query_id=request.query.query_id, answer_schema=request.query.answer_schema, raw_output=raw_output)
        prediction = prediction.model_copy(update={"latency_ms": (time.monotonic() - begin) * 1000, "usage": {"generated_tokens": int(new_tokens.numel())}})
        self.last_meta = {"visible_prompt_sha256": sha256_bytes(request.rendered_prompt.encode("utf-8")), "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")), "raw_output_sha256": sha256_bytes(raw_output.encode("utf-8")), "generated_tokens": int(new_tokens.numel()), "chat_template_sha256": self.chat_template_sha256}
        return prediction

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()


def token_f1(pred: Any, gold: Any) -> float | None:
    if not isinstance(pred, str) or not isinstance(gold, str):
        return None
    a, b = pred.lower().split(), gold.lower().split()
    if not a and not b:
        return 1.0
    common = 0
    counts = {}
    for token in a:
        counts[token] = counts.get(token, 0) + 1
    for token in b:
        if counts.get(token, 0):
            common += 1
            counts[token] -= 1
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def run_seed(args) -> int:
    output = Path(args.output_root)
    manifest = json.loads((output / "canary_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_bindings"]["runner_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("runner source changed after prepare")
    plan = load_jsonl(output / "call_plan.jsonl")
    selected = tuple(row for row in plan if row["seed"] == args.seed)
    if len(selected) != 160:
        raise ValueError("seed plan must contain 160 calls")
    attempts_path = output / f"attempts_seed_{args.seed}.jsonl"
    existing = load_jsonl(attempts_path) if attempts_path.exists() else ()
    if tuple(row["call_id"] for row in existing) != tuple(row["call_id"] for row in selected[: len(existing)]):
        raise ValueError("persisted attempts are not an exact plan prefix")
    tasks = load_jsonl(Path(args.tasks), MemUpdateTaskV3)
    trajectories = load_jsonl(Path(args.trajectories), RawAppendTrajectoryV1)
    task_map = {task.task_id: task for task in tasks}
    trajectory_map = {row.task_id: row for row in trajectories}
    model = QwenCanaryModel(args.model_snapshot, args.seed, args.device)
    started = time.monotonic()
    model.load()
    load_seconds = time.monotonic() - started
    try:
        for coordinate in selected[len(existing):]:
            task = task_map[coordinate["task_id"]]
            condition = next(row for row in CONDITIONS if row[0] == coordinate["condition"])
            runtime = RuntimeConfigV3(run_id=f"qwen35-canary-s{args.seed}-{condition[0]}-{task.task_id}", retrieval_policy="normal_topk", answer_mode="slot_prompt", retrieval_k=RETRIEVAL_K, capture_snapshots=False)
            begin = time.monotonic()
            try:
                run = execute_task12_task_v3(task, RawAppendAdapterV3(task, retrieval_policy="normal_topk"), runtime, prompted_answer_model=model, context_order=condition[1], context_annotation=condition[2], frozen_trajectory=trajectory_map[task.task_id])
                prediction = run.answer_predictions[0]
                gold = task.gold_evidence[0].answer
                values = [entry.value_candidate for entry in run.retrieval_traces[0].retrieved_entries]
                row = {**coordinate, "status": "PASS", "completion_status": run.completion_status.value, "format_valid": prediction.format_valid, "disposition": prediction.disposition.value, "parsed_answer": prediction.parsed_answer, "gold_answer": gold, "exact_match": bool(prediction.format_valid and prediction.parsed_answer == gold), "token_f1": token_f1(prediction.parsed_answer, gold), "stale_copied": bool(prediction.format_valid and prediction.parsed_answer != gold and prediction.parsed_answer in values), "error_flags": list(prediction.error_flags), "latency_ms": (time.monotonic() - begin) * 1000, "model_meta": model.last_meta, "task_run": run.model_dump(mode="json")}
            except Exception as exc:
                row = {**coordinate, "status": "FAIL", "completion_status": "failed", "error_class": type(exc).__name__, "error_message_sha256": sha256_bytes(str(exc).encode("utf-8")), "latency_ms": (time.monotonic() - begin) * 1000}
            append_row(attempts_path, row)
            completed = len(existing) + 1
            existing = existing + (row,)
            tmp = output / "progress.tmp"
            tmp.write_bytes(canonical_bytes({"completed": sum(len(load_jsonl(output / f"attempts_seed_{seed}.jsonl")) if (output / f"attempts_seed_{seed}.jsonl").exists() else 0 for seed in SEEDS), "call_count": 320, "last_call_id": row["call_id"], "status": "RUNNING"}))
            os.replace(tmp, output / "progress.json")
    finally:
        model.close()
    receipt = {"schema_version": "memupdatebench.post-core.qwen35-canary-seed-receipt.v1", "seed": args.seed, "attempt_count": len(load_jsonl(attempts_path)), "load_seconds": load_seconds, "runner_sha256": sha256_file(Path(__file__)), "attempts_sha256": sha256_file(attempts_path), "model_id": MODEL_ID, "revision": MODEL_REVISION, "tree_sha256": MODEL_TREE_SHA256, "status": "COMPLETE"}
    receipt_path = output / f"seed_{args.seed}_runtime_receipt.json"
    if receipt_path.exists():
        receipt_path.write_bytes(canonical_bytes(receipt))
    else:
        write_exclusive(receipt_path, canonical_bytes(receipt))
    print(json.dumps({"status": "COMPLETE", "seed": args.seed, "attempts": 160, "output": str(attempts_path)}, sort_keys=True))
    return 0


def finalize(args) -> int:
    output = Path(args.output_root)
    plan = load_jsonl(output / "call_plan.jsonl")
    rows = tuple(row for seed in SEEDS for row in load_jsonl(output / f"attempts_seed_{seed}.jsonl"))
    if len(rows) != 320 or tuple(row["call_id"] for row in rows) != tuple(row["call_id"] for row in plan):
        raise ValueError("canary terminal rows do not match complete call plan")
    summary = {
        "schema_version": "memupdatebench.post-core.qwen35-canary-receipt.v1",
        "release_id": RELEASE_ID,
        "attempts": 320,
        "passes": sum(row["status"] == "PASS" for row in rows),
        "failures": sum(row["status"] != "PASS" for row in rows),
        "format_valid": sum(bool(row.get("format_valid")) for row in rows),
        "exact_match_mean": sum(bool(row.get("exact_match")) for row in rows) / 320,
        "stale_copied_mean": sum(bool(row.get("stale_copied")) for row in rows) / 320,
        "by_condition": {},
        "by_seed": {},
        "scientific_status": "CANARY_ONLY",
        "confirmatory_status": "NOT_RUN",
        "benchmark_status": "NOT_RUN",
        "max_retries": 0,
    }
    for condition in (row[0] for row in CONDITIONS):
        selected = [row for row in rows if row["condition"] == condition]
        summary["by_condition"][condition] = {"n": len(selected), "pass": sum(row["status"] == "PASS" for row in selected), "format_valid": sum(bool(row.get("format_valid")) for row in selected), "exact_match_mean": sum(bool(row.get("exact_match")) for row in selected) / len(selected), "stale_copied_mean": sum(bool(row.get("stale_copied")) for row in selected) / len(selected)}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        summary["by_seed"][str(seed)] = {"n": len(selected), "pass": sum(row["status"] == "PASS" for row in selected), "exact_match_mean": sum(bool(row.get("exact_match")) for row in selected) / len(selected)}
    summary["attempts_sha256"] = {str(seed): sha256_file(output / f"attempts_seed_{seed}.jsonl") for seed in SEEDS}
    summary["payload_sha256"] = sha256_bytes(canonical_bytes(summary))
    write_exclusive(output / "canary_receipt.json", canonical_bytes(summary))
    artifacts = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_index.json":
            artifacts[path.name] = sha256_file(path)
    index = {"schema_version": "memupdatebench.post-core.qwen35-canary-index.v1", "release_id": RELEASE_ID, "artifacts": artifacts}
    write_exclusive(output / "artifact_index.json", canonical_bytes(index))
    (output / "progress.json").write_bytes(canonical_bytes({"completed": 320, "call_count": 320, "status": "FINALIZED", "receipt_sha256": sha256_file(output / "canary_receipt.json")}))
    print(json.dumps(summary, sort_keys=True))
    return 0


def parser():
    value = argparse.ArgumentParser(allow_abbrev=False)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", allow_abbrev=False)
    for item in ("tasks", "trajectories", "preparation-manifest", "output-root"):
        prepare_parser.add_argument(f"--{item}", required=True)
    run_parser = sub.add_parser("run-seed", allow_abbrev=False)
    for item in ("tasks", "trajectories", "model-snapshot", "device", "output-root"):
        run_parser.add_argument(f"--{item}", required=True)
    run_parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    final_parser = sub.add_parser("finalize", allow_abbrev=False)
    final_parser.add_argument("--output-root", required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "run-seed":
        return run_seed(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
