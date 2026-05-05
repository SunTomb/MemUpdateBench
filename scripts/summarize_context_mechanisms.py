from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


FIELDNAMES = [
    "run_name",
    "result_path",
    "model_name",
    "context_order",
    "context_annotation",
    "retrieval_policy",
    "num_examples",
    "avg_em",
    "avg_f1",
    "answer_value_present_rate",
    "state_accuracy",
    "gold_retrieved_rate",
    "stale_retrieved_rate",
    "stale_retrieved_avg",
    "same_name_distractor_rate",
]


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) or {}
    results = payload.get("results", []) or []

    gold_retrieved = []
    stale_retrieved = []
    stale_counts = []
    same_name = []
    for row in results:
        trace = row.get("answer_trace") or {}
        gold_retrieved.append(float(bool(trace.get("gold_value_in_retrieved", False))))
        stale_retrieved.append(float(bool(trace.get("stale_same_slot_in_retrieved", False))))
        stale_counts.append(float(len(trace.get("stale_same_slot_values", []) or [])))
        same_name.append(float(bool(trace.get("same_name_distractor_in_retrieved", False))))

    return {
        "run_name": path.parent.name,
        "result_path": str(path),
        "model_name": summary.get("model_name", ""),
        "context_order": summary.get("context_order", "normal"),
        "context_annotation": summary.get("context_annotation", "none"),
        "retrieval_policy": summary.get("retrieval_policy", "normal"),
        "num_examples": summary.get("num_examples", len(results)),
        "avg_em": summary.get("avg_em", ""),
        "avg_f1": summary.get("avg_f1", ""),
        "answer_value_present_rate": summary.get("answer_value_present_rate", ""),
        "state_accuracy": summary.get("state_accuracy", ""),
        "gold_retrieved_rate": mean(gold_retrieved) if gold_retrieved else "",
        "stale_retrieved_rate": mean(stale_retrieved) if stale_retrieved else "",
        "stale_retrieved_avg": mean(stale_counts) if stale_counts else "",
        "same_name_distractor_rate": mean(same_name) if same_name else "",
    }


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "## Results",
        "",
        "| run | model | order | annotation | retrieval | EM | F1 | answer value present | gold retrieved | stale retrieved | stale retrieved avg |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['model_name']} | {row['context_order']} | {row['context_annotation']} | "
            f"{row['retrieval_policy']} | {fmt(row['avg_em'])} | {fmt(row['avg_f1'])} | {fmt(row['answer_value_present_rate'])} | "
            f"{fmt(row['gold_retrieved_rate'])} | {fmt(row['stale_retrieved_rate'])} | {fmt(row['stale_retrieved_avg'])} |"
        )
    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- If current_first improves over current_last, position bias is implicated.",
        "- If timestamp or latest_outdated_label improves over none, semantic/version disambiguation is implicated.",
        "- If gold/stale retrieved rates are unchanged but EM moves, the effect is answer-layer behavior rather than retrieval composition.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize context order/annotation mechanism probes")
    parser.add_argument("--result_root", default="results/p80_mechanism_probes")
    parser.add_argument("--output_dir", default="results/p80_mechanism_probe_summary")
    parser.add_argument("--prefix", default="context_mechanism_summary")
    parser.add_argument("--title", default="P8.0 Context Mechanism Probe Summary")
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = [load_result(path) for path in sorted(root.glob("*/evomemory_results.json"))]
    rows.sort(key=lambda row: (str(row["context_annotation"]), str(row["context_order"]), str(row["run_name"])))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / f"{args.prefix}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, out / f"{args.prefix}.md", args.title)
    print(json.dumps({"num_rows": len(rows), "output_dir": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
