from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


NAME_RE = re.compile(r"^(?P<method>.+)_(?P<variant>v\d+_[a-z_]+)_k(?P<k>\d+)$")


def load_rows(result_root: Path) -> list[dict]:
    rows = []
    for path in sorted(result_root.glob("*/evomemory_results.json")):
        match = NAME_RE.match(path.parent.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        summary = payload.get("summary", {})
        results = payload.get("results", [])
        rows.append({
            "name": path.parent.name,
            "method": match.group("method"),
            "variant": match.group("variant"),
            "k": int(match.group("k")),
            "num_examples": summary.get("num_examples", len(results)),
            "avg_em": summary.get("avg_em", 0.0),
            "avg_f1": summary.get("avg_f1", 0.0),
            "state_accuracy": summary.get("state_accuracy", 0.0),
            "stale_same_slot_entry_count_avg": summary.get("stale_same_slot_entry_count_avg", 0.0),
            "final_memory_size_avg": summary.get("final_memory_size_avg", 0.0),
            "gold_retrieved_rate": mean([
                float((result.get("answer_trace") or {}).get("gold_value_in_retrieved", False))
                for result in results
            ]) if results else 0.0,
            "stale_retrieved_rate": mean([
                float((result.get("answer_trace") or {}).get("stale_same_slot_in_retrieved", False))
                for result in results
            ]) if results else 0.0,
            "distractor_retrieved_rate": mean([
                float((result.get("answer_trace") or {}).get("same_name_distractor_in_retrieved", False))
                for result in results
            ]) if results else 0.0,
            "path": str(path),
        })
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "variant", "k", "num_examples", "avg_em", "avg_f1",
        "state_accuracy", "stale_same_slot_entry_count_avg", "final_memory_size_avg",
        "gold_retrieved_rate", "stale_retrieved_rate", "distractor_retrieved_rate", "path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prompt Robustness Summary",
        "",
        "| Method | Variant | k | EM | F1 | State acc. | Stale | Mem. size | Gold retrieved | Stale retrieved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['k']} | "
            f"{row['avg_em']:.2f} | {row['avg_f1']:.2f} | {row['state_accuracy']:.2f} | "
            f"{row['stale_same_slot_entry_count_avg']:.2f} | {row['final_memory_size_avg']:.2f} | "
            f"{row['gold_retrieved_rate']:.2f} | {row['stale_retrieved_rate']:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prompt robustness sweeps")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.result_root))
    rows.sort(key=lambda row: (row["method"], row["variant"], row["k"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_robustness_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(rows, output_dir / "prompt_robustness_summary.csv")
    write_md(rows, output_dir / "prompt_robustness_summary.md")
    print(f"Loaded {len(rows)} prompt robustness rows")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
