from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "run_name",
    "result_path",
    "update_policy",
    "store_policy",
    "answer_mode",
    "num_examples",
    "avg_em",
    "avg_f1",
    "state_accuracy",
    "stale_same_slot_entry_count_avg",
    "final_memory_size_avg",
    "answer_value_present_rate",
]


def load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return {
        "run_name": path.parent.name,
        "result_path": str(path),
        "update_policy": summary.get("update_policy", ""),
        "store_policy": summary.get("store_policy", ""),
        "answer_mode": summary.get("answer_mode", ""),
        "num_examples": summary.get("num_examples", ""),
        "avg_em": summary.get("avg_em", ""),
        "avg_f1": summary.get("avg_f1", ""),
        "state_accuracy": summary.get("state_accuracy", ""),
        "stale_same_slot_entry_count_avg": summary.get("stale_same_slot_entry_count_avg", ""),
        "final_memory_size_avg": summary.get("final_memory_size_avg", ""),
        "answer_value_present_rate": summary.get("answer_value_present_rate", ""),
    }


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# P8.0 Simple External Pipeline Summary",
        "",
        "| run | update | store | answer | n | EM | F1 | state | stale same-slot | memory size | answer value present |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['update_policy']} | {row['store_policy']} | {row['answer_mode']} | "
            f"{row['num_examples']} | {fmt(row['avg_em'])} | {fmt(row['avg_f1'])} | {fmt(row['state_accuracy'])} | "
            f"{fmt(row['stale_same_slot_entry_count_avg'])} | {fmt(row['final_memory_size_avg'])} | "
            f"{fmt(row['answer_value_present_rate'])} |"
        )
    lines.extend([
        "",
        "This is a deliberately simple extract-then-store diagnostic baseline. It uses the project parser to extract `(entity, attribute, value)` records, stores inspectable memory entries, and reports the same state/stale/compactness/answer metrics as the main benchmark. It is not a Mem0 replacement or external SDK leaderboard row.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simple external pipeline runs")
    parser.add_argument("--result_root", default="results/p80_simple_external_pipeline")
    parser.add_argument("--output_dir", default="results/p80_simple_external_pipeline_summary")
    args = parser.parse_args()

    rows = [load_row(path) for path in sorted(Path(args.result_root).glob("*/evomemory_results.json"))]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "simple_external_pipeline_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "simple_external_pipeline_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, out / "simple_external_pipeline_summary.md")
    print(json.dumps({"num_rows": len(rows), "output_dir": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
