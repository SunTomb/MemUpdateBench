from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NAME_RE = re.compile(r"(?P<answer_mode>direct|prompt)_t(?P<threshold>\d+\.\d+)_k(?P<k>\d+)$")


def load_rows(result_root: Path) -> list[dict]:
    rows = []
    for path in sorted(result_root.glob("*/evomemory_results.json")):
        match = NAME_RE.match(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        rows.append({
            "name": path.parent.name,
            "answer_mode": match.group("answer_mode"),
            "threshold": float(match.group("threshold")),
            "k": int(match.group("k")),
            "num_examples": summary.get("num_examples", 0),
            "avg_em": summary.get("avg_em", 0.0),
            "avg_f1": summary.get("avg_f1", 0.0),
            "state_accuracy": summary.get("state_accuracy", 0.0),
            "stale_same_slot_entry_count_avg": summary.get("stale_same_slot_entry_count_avg", 0.0),
            "same_slot_entry_count_avg": summary.get("same_slot_entry_count_avg", 0.0),
            "final_memory_size_avg": summary.get("final_memory_size_avg", 0.0),
            "path": str(path),
        })
    rows.sort(key=lambda row: (row["answer_mode"], row["k"], row["threshold"]))
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fields = [
        "answer_mode", "threshold", "k", "num_examples", "avg_em", "avg_f1", "state_accuracy",
        "stale_same_slot_entry_count_avg", "same_slot_entry_count_avg", "final_memory_size_avg", "path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Heuristic Threshold Sweep Summary",
        "",
        "| Answer | k | Threshold | EM | F1 | State acc. | Stale | Same-slot | Mem. size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['answer_mode']} | {row['k']} | {row['threshold']:.2f} | "
            f"{row['avg_em']:.3f} | {row['avg_f1']:.3f} | {row['state_accuracy']:.3f} | "
            f"{row['stale_same_slot_entry_count_avg']:.2f} | {row['same_slot_entry_count_avg']:.2f} | "
            f"{row['final_memory_size_avg']:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize heuristic CRUD threshold sweeps")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.result_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heuristic_threshold_summary.json").write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, output_dir / "heuristic_threshold_summary.csv")
    write_md(rows, output_dir / "heuristic_threshold_summary.md")
    print(f"Loaded {len(rows)} rows")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
