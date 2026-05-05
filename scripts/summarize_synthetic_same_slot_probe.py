from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


GROUP_FIELDS = ["value_policy", "context_order", "context_annotation", "stale_count"]
SUMMARY_FIELDS = GROUP_FIELDS + ["n", "em", "f1", "answer_value_present"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in {"True", "true"}:
        return 1.0
    if value in {"False", "false"}:
        return 0.0
    return float(value)


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["value_policy"],
            row["context_order"],
            row["context_annotation"],
            int(row["stale_count"]),
        )
        groups[key].append(row)
    out = []
    for (value_policy, order, annotation, stale_count), items in sorted(groups.items()):
        out.append({
            "value_policy": value_policy,
            "context_order": order,
            "context_annotation": annotation,
            "stale_count": stale_count,
            "n": len(items),
            "em": mean(as_float(item, "em") for item in items),
            "f1": mean(as_float(item, "f1") for item in items),
            "answer_value_present": mean(as_float(item, "answer_value_present") for item in items),
        })
    return out


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P8.0 Synthetic Same-Slot Probe Analysis",
        "",
        "## Grouped results",
        "",
        "| value policy | order | annotation | stale count | n | EM | F1 | answer value present |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['value_policy']} | {row['context_order']} | {row['context_annotation']} | {row['stale_count']} | "
            f"{row['n']} | {fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['answer_value_present'])} |"
        )

    lines.extend([
        "",
        "## Interpretation checks",
        "",
        "- Conflict vs same_as_current at the same stale count tests value conflict versus attention dilution.",
        "- Chronological vs reverse_chronological tests order sensitivity when gold is always present.",
        "- latest/outdated labels test whether explicit version semantics mitigates conflict.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic same-slot probe outputs")
    parser.add_argument("--input_csv", default="results/p80_synthetic_same_slot_probe/synthetic_same_slot_examples.csv")
    parser.add_argument("--output_dir", default="results/p80_synthetic_same_slot_probe_analysis")
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    summary = grouped_summary(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "synthetic_same_slot_grouped_summary.csv", summary)
    (out / "synthetic_same_slot_grouped_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out / "synthetic_same_slot_grouped_summary.md", summary)
    print(json.dumps({"num_examples": len(rows), "num_groups": len(summary), "output_dir": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
