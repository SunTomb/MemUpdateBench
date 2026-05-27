from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if isinstance(value, bool):
        return float(value)
    if str(value).lower() in {"true", "false"}:
        return 1.0 if str(value).lower() == "true" else 0.0
    return float(value or 0.0)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    final_only_em = 0.0
    if "final_only" in grouped:
        final_only_em = mean(as_float(row, "em") for row in grouped["final_only"])
    output = []
    for condition, items in sorted(grouped.items()):
        em = mean(as_float(row, "em") for row in items)
        output.append({
            "condition": condition,
            "n": len(items),
            "distractor_count": items[0].get("distractor_count", ""),
            "em": em,
            "f1": mean(as_float(row, "f1") for row in items),
            "value_em": mean(as_float(row, "value_em") for row in items),
            "answer_value_present": mean(as_float(row, "answer_value_present") for row in items),
            "stale_value_copied": mean(as_float(row, "stale_value_copied") for row in items),
            "em_drop_from_final_only": final_only_em - em,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Conflict-Type Probe Summary",
        "",
        "| condition | n | distractors | EM | F1 | value EM | answer value present | stale copied | EM drop vs final-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['distractor_count']} | "
            f"{row['em']:.3f} | {row['f1']:.3f} | {row['value_em']:.3f} | "
            f"{row['answer_value_present']:.3f} | {row['stale_value_copied']:.3f} | "
            f"{row['em_drop_from_final_only']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize conflict-type probe examples")
    parser.add_argument("--input_csv", default="results/p83_conflict_type_probe/conflict_type_examples.csv")
    parser.add_argument("--output_dir", default="results/p83_conflict_type_probe_summary")
    args = parser.parse_args()

    rows = read_rows(Path(args.input_csv))
    summary = summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "conflict_type_summary.csv", summary)
    (output_dir / "conflict_type_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "conflict_type_summary.md", summary)
    print(json.dumps({"num_rows": len(rows), "num_conditions": len(summary), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
