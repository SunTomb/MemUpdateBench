from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def new_exec_stats() -> dict:
    return {
        "requested": {},
        "effective": {},
        "success": 0,
        "failed": 0,
        "fallback_target": 0,
    }


def add_counts(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def load_shard(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("**/evomemory_results.json")))
        else:
            files.append(path)
    unique = []
    seen = set()
    for file in files:
        resolved = file.resolve()
        if resolved not in seen:
            unique.append(file)
            seen.add(resolved)
    return unique


def mean(results: list[dict], key: str) -> float:
    values = [float(result.get(key, 0.0)) for result in results]
    return float(np.mean(values)) if values else 0.0


def build_summary(shards: list[dict], results: list[dict]) -> dict:
    base = dict(shards[0].get("summary", {})) if shards else {}
    total_exec_stats = new_exec_stats()
    for result in results:
        stats = result.get("exec_stats", {})
        add_counts(total_exec_stats["requested"], stats.get("requested", {}))
        add_counts(total_exec_stats["effective"], stats.get("effective", {}))
        total_exec_stats["success"] += stats.get("success", 0)
        total_exec_stats["failed"] += stats.get("failed", 0)
        total_exec_stats["fallback_target"] += stats.get("fallback_target", 0)

    elapsed = sum(float(shard.get("summary", {}).get("elapsed_seconds", 0.0)) for shard in shards)
    summary = {
        **base,
        "num_examples": len(results),
        "start_idx": min((int(result.get("example_id", 0)) for result in results), default=0),
        "end_idx": max((int(result.get("example_id", -1)) for result in results), default=-1) + 1,
        "avg_f1": mean(results, "f1"),
        "avg_em": mean(results, "em"),
        "value_em": mean(results, "value_em"),
        "answer_value_present_rate": mean(results, "answer_value_present"),
        "op_counts": total_exec_stats["requested"],
        "exec_stats": total_exec_stats,
        "final_memory_size_avg": mean(results, "memory_size"),
        "same_slot_entry_count_avg": mean(results, "same_slot_entry_count"),
        "stale_same_slot_entry_count_avg": mean(results, "stale_same_slot_entry_count"),
        "gold_same_slot_entry_count_avg": mean(results, "gold_same_slot_entry_count"),
        "write_amplification_avg": mean(results, "write_amplification"),
        "target_slot_write_count_avg": mean(results, "target_slot_write_count"),
        "state_resolve_rate": mean(results, "state_resolved"),
        "state_accuracy": mean(results, "state_value_em"),
        "gold_present_rate": mean(results, "gold_value_present_anywhere"),
        "stale_present_rate": mean(results, "stale_value_present_same_slot"),
        "elapsed_seconds": elapsed,
        "merged_shards": len(shards),
    }
    return summary


def write_csv(summary: dict, output_dir: Path) -> None:
    fields = [
        "num_examples",
        "avg_em",
        "avg_f1",
        "value_em",
        "state_accuracy",
        "stale_same_slot_entry_count_avg",
        "same_slot_entry_count_avg",
        "final_memory_size_avg",
        "elapsed_seconds",
        "merged_shards",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded EvoMemory evaluation outputs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard result files or directories")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("No evomemory_results.json shard files found")

    shards = [load_shard(file) for file in files]
    results = []
    seen_ids = set()
    for shard in shards:
        for result in shard.get("results", []):
            example_id = result.get("example_id")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id in shards: {example_id}")
            seen_ids.add(example_id)
            results.append(result)
    results.sort(key=lambda result: result.get("example_id", -1))

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary = build_summary(shards, results)
    with (output_dir / "evomemory_results.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    write_csv(summary, output_dir)
    print(
        f"Merged {len(files)} shards / {len(results)} examples -> "
        f"EM={summary['avg_em']:.4f} F1={summary['avg_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
