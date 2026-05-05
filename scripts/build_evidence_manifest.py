from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "result_path",
    "result_group",
    "run_name",
    "benchmark",
    "mode",
    "answer_mode",
    "retrieval_policy",
    "data_file",
    "num_examples",
    "k_updates",
    "avg_em",
    "avg_f1",
    "value_em",
    "answer_value_present_rate",
    "state_accuracy",
    "state_resolve_rate",
    "same_slot_entry_count_avg",
    "stale_same_slot_entry_count_avg",
    "gold_same_slot_entry_count_avg",
    "final_memory_size_avg",
    "write_amplification_avg",
    "target_slot_write_count_avg",
    "slot_prompt_variant",
    "answer_topk",
    "lora_checkpoint",
    "decision_temperature",
    "status",
    "reason",
]


CORE_GROUPS = [
    "update_frequency_p63",
    "p68_answer_layer_diagnostics",
    "p68_stale_intervention",
    "p68_heuristic_threshold_sweep",
    "p69_expanded_slot_prompt",
    "p69_expanded_slot_prompt_allk",
    "p69_k32_slot_prompt",
    "p70_stale_filter_intervention",
    "p69_external_baselines",
    "p70_external_baselines",
]


def infer_group(path: Path, result_root: Path) -> str:
    try:
        rel = path.relative_to(result_root)
    except ValueError:
        rel = path
    return rel.parts[0] if rel.parts else "unknown"


def first_k(results: list[dict[str, Any]]) -> Any:
    for row in results:
        if "k_updates" in row:
            return row.get("k_updates")
    return ""


def normalize_row(path: Path, result_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {}) or {}
    results = payload.get("results", []) or []
    group = infer_group(path, result_root)
    row = {key: "" for key in FIELDNAMES}
    row.update({
        "result_path": str(path),
        "result_group": group,
        "run_name": path.parent.name,
        "benchmark": summary.get("benchmark", ""),
        "mode": summary.get("mode", summary.get("method", "")),
        "answer_mode": summary.get("answer_mode", ""),
        "retrieval_policy": summary.get("retrieval_policy", "normal"),
        "data_file": summary.get("data_file", ""),
        "num_examples": summary.get("num_examples", len(results) if results else ""),
        "k_updates": summary.get("k_updates", first_k(results)),
        "avg_em": summary.get("avg_em", ""),
        "avg_f1": summary.get("avg_f1", ""),
        "value_em": summary.get("value_em", ""),
        "answer_value_present_rate": summary.get("answer_value_present_rate", summary.get("answer_value_present", "")),
        "state_accuracy": summary.get("state_accuracy", ""),
        "state_resolve_rate": summary.get("state_resolve_rate", ""),
        "same_slot_entry_count_avg": summary.get("same_slot_entry_count_avg", ""),
        "stale_same_slot_entry_count_avg": summary.get("stale_same_slot_entry_count_avg", ""),
        "gold_same_slot_entry_count_avg": summary.get("gold_same_slot_entry_count_avg", ""),
        "final_memory_size_avg": summary.get("final_memory_size_avg", ""),
        "write_amplification_avg": summary.get("write_amplification_avg", ""),
        "target_slot_write_count_avg": summary.get("target_slot_write_count_avg", ""),
        "slot_prompt_variant": summary.get("slot_prompt_variant", ""),
        "answer_topk": summary.get("answer_topk", ""),
        "lora_checkpoint": summary.get("lora_checkpoint", ""),
        "decision_temperature": summary.get("decision_temperature", ""),
        "status": summary.get("status", "completed"),
        "reason": summary.get("reason", ""),
    })
    return row


def sort_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    try:
        k = int(row.get("k_updates") or -1)
    except (TypeError, ValueError):
        k = -1
    return (str(row["result_group"]), str(row["mode"]), str(row["answer_mode"]), k, str(row["run_name"]))


def fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    core_rows = [row for row in rows if row["result_group"] in CORE_GROUPS]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in core_rows:
        grouped.setdefault(str(row["result_group"]), []).append(row)

    lines = [
        "# MemUpdateBench Evidence Manifest",
        "",
        "This file indexes existing result artifacts for the v3 benchmark + analysis plan.",
        "It is a provenance aid, not a paper table.",
        "",
        "## Group counts",
        "",
        "| group | runs |",
        "| --- | ---: |",
    ]
    for group in sorted(grouped):
        lines.append(f"| {group} | {len(grouped[group])} |")

    lines.extend([
        "",
        "## Core completed rows",
        "",
        "| group | run | mode | answer | retrieval | k | n | EM | F1 | state | stale | memory |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in core_rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["result_group"]),
                str(row["run_name"]),
                str(row["mode"]),
                str(row["answer_mode"]),
                str(row["retrieval_policy"]),
                fmt(row["k_updates"], 0),
                fmt(row["num_examples"], 0),
                fmt(row["avg_em"]),
                fmt(row["avg_f1"]),
                fmt(row["state_accuracy"]),
                fmt(row["stale_same_slot_entry_count_avg"]),
                fmt(row["final_memory_size_avg"]),
            ])
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.glob("**/evomemory_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append({**{key: "" for key in FIELDNAMES}, "result_path": str(path), "status": "invalid_json", "reason": str(exc)})
            continue
        rows.append(normalize_row(path, result_root, payload))
    return sorted(rows, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a provenance manifest for MemUpdateBench result artifacts")
    parser.add_argument("--result_root", default="results")
    parser.add_argument("--output_dir", default="results/p80_evidence_manifest")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(result_root)
    (output_dir / "evidence_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "evidence_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_dir / "evidence_manifest.md")
    print(json.dumps({"num_rows": len(rows), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
