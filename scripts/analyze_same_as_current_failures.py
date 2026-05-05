from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_evomemory import answer_contains_value, slot_value_match


CASE_FIELDS = [
    "source_path",
    "condition",
    "example_id",
    "attribute",
    "stale_count",
    "context_order",
    "context_annotation",
    "gold_answer",
    "predicted",
    "normalized_exact_match",
    "failure_type",
    "diagnosis",
    "answer_value_present",
    "prompt",
]

SUMMARY_FIELDS = ["condition", "attribute", "stale_count", "context_order", "context_annotation", "failure_type", "n"]


DEFAULT_INPUT = "results/p80_synthetic_same_slot_probe/synthetic_same_slot_examples.csv"
DEFAULT_OUTPUT_DIR = "results/p81_same_as_current_failure_analysis"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cleaned_prediction(text: str) -> str:
    return str(text).strip().split("\n", 1)[0].strip()


def classify_failure(row: dict[str, Any]) -> tuple[str, str]:
    predicted = cleaned_prediction(row.get("predicted", ""))
    gold = str(row.get("gold_answer", "")).strip()
    if slot_value_match(predicted, gold):
        return "exact_after_normalization", "Prediction differs from exact string but matches after benchmark normalization."

    normalized_gold = " ".join(gold.strip().lower().split())
    normalized_pred = " ".join(predicted.strip().lower().split())
    if normalized_gold and normalized_gold in normalized_pred:
        suffix = normalized_pred.replace(normalized_gold, "", 1).strip()
        if any(token in suffix for token in ["latest", "outdated", "event idx", "question", "final value only"]):
            return "gold_plus_annotation_or_prompt_spillover", "Prediction contains the gold value plus label or prompt spillover text."
        return "gold_plus_extraneous_text", "Prediction contains the gold value plus extra text beyond the required exact answer."

    values = ["python", "java", "go", "rust", "kotlin", "scala", "typescript", "c++", "swift", "ruby", "tencent", "alibaba", "baidu", "bytedance", "meituan", "jd", "huawei", "netease", "google", "microsoft", "dalian", "qingdao", "wuhan", "ningbo", "suzhou", "xiamen", "harbin", "chengdu", "jinan", "hefei", "espresso", "oolong tea", "matcha", "latte", "mocha", "cold brew", "green tea", "black coffee", "cocoa", "jasmine tea"]
    mentioned = [value for value in values if value in normalized_pred]
    unique_mentioned = sorted(set(mentioned))
    if len(unique_mentioned) >= 2:
        return "multiple_candidate_values", "Prediction mentions multiple plausible values, suggesting diluted selection rather than a clean exact answer."

    if not answer_contains_value(predicted, gold):
        return "behavioral_non_exact_other", "Prediction does not preserve the gold value string; likely a genuine answer-generation change under repeated same-value context."

    return "unexpected_answer_value_present_false_positive", "answer_value_present stayed positive, but the output does not match the expected formatting buckets."


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("value_policy") != "same_as_current":
            continue
        if float(row.get("em", 0.0)) >= 1.0:
            continue
        if str(row.get("answer_value_present", "")).lower() not in {"1", "1.0", "true"}:
            continue
        failure_type, diagnosis = classify_failure(row)
        selected.append({
            "source_path": row.get("source_path", DEFAULT_INPUT),
            "condition": row.get("condition", ""),
            "example_id": row.get("example_id", ""),
            "attribute": row.get("attribute", ""),
            "stale_count": int(row.get("stale_count", 0)),
            "context_order": row.get("context_order", ""),
            "context_annotation": row.get("context_annotation", ""),
            "gold_answer": row.get("gold_answer", ""),
            "predicted": row.get("predicted", ""),
            "normalized_exact_match": slot_value_match(row.get("predicted", ""), row.get("gold_answer", "")),
            "failure_type": failure_type,
            "diagnosis": diagnosis,
            "answer_value_present": row.get("answer_value_present", ""),
            "prompt": row.get("prompt", ""),
        })
    return selected


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (
            row["condition"],
            row["attribute"],
            int(row["stale_count"]),
            row["context_order"],
            row["context_annotation"],
            row["failure_type"],
        )
        grouped[key] += 1
    out = []
    for (condition, attribute, stale_count, context_order, context_annotation, failure_type), n in sorted(grouped.items()):
        out.append({
            "condition": condition,
            "attribute": attribute,
            "stale_count": stale_count,
            "context_order": context_order,
            "context_annotation": context_annotation,
            "failure_type": failure_type,
            "n": n,
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P8.1 Same-as-Current Failure Analysis",
        "",
        "## Scope",
        "",
        "This analysis targets `same_as_current` rows where exact match fails but `answer_value_present=1`, to separate formatting spillover from genuine behavior change.",
        "",
        "## Failure-type summary",
        "",
        "| condition | attribute | stale count | order | annotation | failure type | n |",
        "| --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['attribute']} | {row['stale_count']} | {row['context_order']} | {row['context_annotation']} | {row['failure_type']} | {row['n']} |"
        )
    lines.extend([
        "",
        "## Representative cases",
        "",
    ])
    for row in case_rows[:12]:
        lines.extend([
            f"### {row['condition']} example {row['example_id']}",
            "",
            f"- Failure type: {row['failure_type']}",
            f"- Diagnosis: {row['diagnosis']}",
            f"- Gold: {row['gold_answer']}",
            f"- Predicted: {row['predicted']}",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze same_as_current EM failures with answer_value_present=1")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    rows = load_rows(input_path)
    selected = select_rows(rows)
    for row in selected:
        row["source_path"] = str(input_path)
    summary_rows = summarize(selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "same_as_current_failure_cases.csv", selected, CASE_FIELDS)
    write_csv(output_dir / "same_as_current_failure_type_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_markdown(output_dir / "same_as_current_failure_analysis.md", summary_rows, selected)
    (output_dir / "same_as_current_failure_analysis.json").write_text(
        json.dumps({"cases": selected, "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"input_rows": len(rows), "selected_cases": len(selected), "summary_rows": len(summary_rows), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
