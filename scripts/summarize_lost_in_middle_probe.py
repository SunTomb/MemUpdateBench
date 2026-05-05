from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = ["run_name", "model_name", "data_file", "num_distractors", "gold_position", "num_examples", "avg_em", "avg_f1", "answer_value_present_rate"]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    rows = []
    for condition in summary.get("conditions", []):
        rows.append({
            "run_name": path.parent.name,
            "model_name": summary.get("model_name", ""),
            "data_file": summary.get("data_file", ""),
            "num_distractors": summary.get("num_distractors", ""),
            "gold_position": condition.get("gold_position", ""),
            "num_examples": condition.get("num_examples", ""),
            "avg_em": condition.get("avg_em", ""),
            "avg_f1": condition.get("avg_f1", ""),
            "answer_value_present_rate": condition.get("answer_value_present_rate", ""),
        })
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# P8.0 Lost-in-the-Middle Gold-Position Probe Summary",
        "",
        "| run | model | distractors | gold position | n | EM | F1 | answer value present |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    order = {"beginning": 0, "middle": 1, "end": 2}
    rows = sorted(rows, key=lambda row: (row["run_name"], order.get(str(row["gold_position"]), 99)))
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['model_name']} | {row['num_distractors']} | {row['gold_position']} | "
            f"{row['num_examples']} | {fmt(row['avg_em'])} | {fmt(row['avg_f1'])} | {fmt(row['answer_value_present_rate'])} |"
        )
    lines.extend([
        "",
        "This probe differs from chronological/reverse-chronological context probes: it holds a synthetic context set fixed and moves the gold entry to the beginning, middle, or end of the answer context. It is a direct Lost-in-the-Middle-style diagnostic for slot-answering.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Lost-in-the-Middle probe outputs")
    parser.add_argument("--result_root", default="results/p80_lost_in_middle_probe")
    parser.add_argument("--output_dir", default="results/p80_lost_in_middle_probe_summary")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.result_root).glob("*/lost_in_middle_results.json")):
        rows.extend(load_rows(path))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lost_in_middle_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "lost_in_middle_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, out / "lost_in_middle_summary.md")
    print(json.dumps({"num_rows": len(rows), "output_dir": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
