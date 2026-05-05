from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_FIELDS = [
    "source_path",
    "mode",
    "answer_mode",
    "retrieval_policy",
    "k_updates",
    "attribute",
    "n",
    "em",
    "f1",
    "answer_value_present",
    "state_accuracy",
    "gold_retrieved_rate",
    "stale_retrieved_avg",
    "stale_same_slot_avg",
]

CASE_FIELDS = [
    "source_path",
    "mode",
    "answer_mode",
    "retrieval_policy",
    "k_updates",
    "attribute",
    "example_id",
    "question",
    "gold_answer",
    "predicted",
    "em",
    "f1",
    "answer_value_present",
    "state_value_em",
    "gold_retrieved_count",
    "stale_retrieved_count",
    "same_slot_entry_count",
    "stale_same_slot_entry_count",
    "error_type",
    "retrieved_memory_entries",
]

GOLD_RETRIEVED_FIELDS = CASE_FIELDS + ["diagnosis"]


DEFAULT_INPUTS = [
    "results/p69_expanded_slot_prompt_allk/constrained_slot_crud_prompt_k16_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k16_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/constrained_slot_crud_prompt_k8_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k8_dev_merged/evomemory_results.json",
]


def slot_value_match(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def retrieved_counts_and_text(row: dict[str, Any]) -> tuple[int, int, str]:
    trace = row.get("answer_trace") or {}
    entries = trace.get("retrieved_entries") or []
    gold_state = row.get("gold_state") or {}
    gold_answer = row.get("gold_answer", "")
    gold_count = 0
    stale_count = 0
    texts = []
    for entry in entries:
        content = str(entry.get("content", ""))
        texts.append(content)
        slot = entry.get("slot") or {}
        if slot.get("entity") != gold_state.get("entity") or slot.get("attribute") != gold_state.get("attribute"):
            continue
        if slot_value_match(slot.get("value", ""), gold_answer):
            gold_count += 1
        else:
            stale_count += 1
    return gold_count, stale_count, " || ".join(texts[:8])


def classify_error(row: dict[str, Any]) -> str:
    if float(row.get("em", 0.0)) >= 1.0:
        return "correct"
    if int(row.get("state_value_em", 0)) <= 0:
        return "state_wrong"
    if int(row.get("gold_retrieved_count", 0)) <= 0:
        return "gold_not_retrieved"
    if int(row.get("answer_value_present", 0)) > 0:
        return "gold_present_but_non_exact"
    if int(row.get("stale_retrieved_count", 0)) > 0:
        return "gold_retrieved_stale_competition"
    return "gold_retrieved_answer_layer_failure"


def diagnose_gold_retrieved_case(row: dict[str, Any]) -> str:
    predicted = str(row.get("predicted", "")).strip()
    gold = str(row.get("gold_answer", "")).strip()
    retrieved = str(row.get("retrieved_memory_entries", ""))
    if int(row.get("answer_value_present", 0)) > 0:
        return "gold value appears in prediction but exact match failed, likely formatting or extra-token issue"
    if predicted and predicted.lower() in retrieved.lower() and predicted.lower() != gold.lower():
        return "prediction copies a retrieved non-gold value or distractor"
    if int(row.get("stale_retrieved_count", 0)) > 0:
        return "gold is retrieved but stale same-slot alternatives are also retrieved"
    return "gold is retrieved without stale same-slot alternatives; likely answer-layer selection or instruction-following failure"


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) or {}
    rows = []
    for row in payload.get("results", []) or []:
        gold_state = row.get("gold_state") or {}
        gold_retrieved, stale_retrieved, retrieved_text = retrieved_counts_and_text(row)
        loaded = {
            "source_path": str(path),
            "mode": summary.get("mode", ""),
            "answer_mode": summary.get("answer_mode", ""),
            "retrieval_policy": summary.get("retrieval_policy", "normal"),
            "k_updates": row.get("k_updates", ""),
            "attribute": gold_state.get("attribute", ""),
            "example_id": row.get("example_id", ""),
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "predicted": row.get("predicted", ""),
            "em": float(row.get("em", 0.0)),
            "f1": float(row.get("f1", 0.0)),
            "answer_value_present": int(bool(row.get("answer_value_present", False))),
            "state_value_em": int(bool(row.get("state_value_em", False))),
            "gold_retrieved_count": gold_retrieved,
            "stale_retrieved_count": stale_retrieved,
            "same_slot_entry_count": row.get("same_slot_entry_count", 0),
            "stale_same_slot_entry_count": row.get("stale_same_slot_entry_count", 0),
            "retrieved_memory_entries": retrieved_text,
        }
        loaded["error_type"] = classify_error(loaded)
        rows.append(loaded)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source_path"],
            row["mode"],
            row["answer_mode"],
            row["retrieval_policy"],
            int(row["k_updates"]),
            row["attribute"],
        )
        grouped[key].append(row)
    out = []
    for (source, mode, answer_mode, retrieval_policy, k_updates, attribute), items in sorted(grouped.items()):
        out.append({
            "source_path": source,
            "mode": mode,
            "answer_mode": answer_mode,
            "retrieval_policy": retrieval_policy,
            "k_updates": k_updates,
            "attribute": attribute,
            "n": len(items),
            "em": mean(float(item["em"]) for item in items),
            "f1": mean(float(item["f1"]) for item in items),
            "answer_value_present": mean(float(item["answer_value_present"]) for item in items),
            "state_accuracy": mean(float(item["state_value_em"]) for item in items),
            "gold_retrieved_rate": mean(1.0 if int(item["gold_retrieved_count"]) > 0 else 0.0 for item in items),
            "stale_retrieved_avg": mean(float(item["stale_retrieved_count"]) for item in items),
            "stale_same_slot_avg": mean(float(item["stale_same_slot_entry_count"]) for item in items),
        })
    return out


def select_cases(rows: list[dict[str, Any]], max_cases_per_attr: int) -> list[dict[str, Any]]:
    targets = {"company", "language", "project", "hobby", "timezone"}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["attribute"] not in targets or row["em"] >= 1.0:
            continue
        key = (row["mode"], int(row["k_updates"]), row["attribute"])
        grouped[key].append(row)
    cases = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda row: (
                -int(row["gold_retrieved_count"]),
                int(row["answer_value_present"]),
                -int(row["stale_retrieved_count"]),
                int(row["example_id"]),
            ),
        )
        cases.extend(items[:max_cases_per_attr])
    return cases


def select_gold_retrieved_wrong_cases(rows: list[dict[str, Any]], max_cases_per_attr: int) -> list[dict[str, Any]]:
    targets = {"company", "language"}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["attribute"] not in targets:
            continue
        if row["em"] >= 1.0 or int(row["gold_retrieved_count"]) <= 0:
            continue
        key = (row["mode"], int(row["k_updates"]), row["attribute"])
        enriched = dict(row)
        enriched["diagnosis"] = diagnose_gold_retrieved_case(row)
        grouped[key].append(enriched)
    cases = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda row: (
                row["error_type"],
                int(row["answer_value_present"]),
                -int(row["stale_retrieved_count"]),
                int(row["example_id"]),
            ),
        )
        cases.extend(items[:max_cases_per_attr])
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def error_type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["attribute"] not in {"company", "language"}:
            continue
        key = (row["mode"], int(row["k_updates"]), row["attribute"], row["error_type"])
        grouped[key].append(row)
    out = []
    for (mode, k_updates, attribute, error_type), items in sorted(grouped.items()):
        out.append({
            "mode": mode,
            "k_updates": k_updates,
            "attribute": attribute,
            "error_type": error_type,
            "n": len(items),
        })
    return out


def write_gold_retrieved_markdown(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# P8.0 Gold-Retrieved Attribute Error Cases",
        "",
        "## Scope",
        "",
        "This note focuses on `company` and `language` failures where the gold value is present in retrieved context but the answer is still wrong. These are the cases closest to the advisor's question about residual attribute sensitivity under gold-retrieved conditions.",
        "",
    ]
    for row in cases:
        lines.extend([
            f"### {row['mode']} k={row['k_updates']} {row['attribute']} example {row['example_id']}",
            "",
            f"- Error type: {row['error_type']}",
            f"- Diagnosis: {row['diagnosis']}",
            f"- Question: {row['question']}",
            f"- Gold: {row['gold_answer']}",
            f"- Predicted: {row['predicted']}",
            f"- Gold retrieved count: {row['gold_retrieved_count']}",
            f"- Stale retrieved count: {row['stale_retrieved_count']}",
            f"- Retrieved entries: {row['retrieved_memory_entries']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P8.0 Attribute Failure Analysis",
        "",
        "## Scope",
        "",
        "This note summarizes per-attribute answer failures from existing expanded split slot-prompt runs.",
        "It is intended to explain residual answer-layer failures after state and retrieval are correct.",
        "",
        "## Per-attribute summary",
        "",
        "| mode | k | attribute | n | EM | F1 | gold retrieved | state acc. | stale retrieved avg |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if int(row["k_updates"]) != 16:
            continue
        lines.append(
            f"| {row['mode']} | {row['k_updates']} | {row['attribute']} | {row['n']} | "
            f"{fmt(row['em'])} | {fmt(row['f1'])} | {fmt(row['gold_retrieved_rate'])} | "
            f"{fmt(row['state_accuracy'])} | {fmt(row['stale_retrieved_avg'])} |"
        )
    lines.extend([
        "",
        "## Company/language error-type summary",
        "",
        "| mode | k | attribute | error type | n |",
        "| --- | ---: | --- | --- | ---: |",
    ])
    for row in error_rows:
        if int(row["k_updates"]) != 16:
            continue
        lines.append(
            f"| {row['mode']} | {row['k_updates']} | {row['attribute']} | {row['error_type']} | {row['n']} |"
        )
    lines.extend([
        "",
        "## Representative failure cases",
        "",
    ])
    for row in case_rows[:40]:
        lines.extend([
            f"### {row['mode']} k={row['k_updates']} {row['attribute']} example {row['example_id']}",
            "",
            f"- Question: {row['question']}",
            f"- Gold: {row['gold_answer']}",
            f"- Predicted: {row['predicted']}",
            f"- Gold retrieved count: {row['gold_retrieved_count']}",
            f"- Stale retrieved count: {row['stale_retrieved_count']}",
            f"- Error type: {row['error_type']}",
            f"- Retrieved entries: {row['retrieved_memory_entries']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze expanded split failures by attribute")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", default="results/p80_attribute_error_analysis")
    parser.add_argument("--max_cases_per_attr", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"missing input: {path}")
            continue
        rows.extend(load_rows(path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    case_rows = select_cases(rows, args.max_cases_per_attr)
    gold_retrieved_cases = select_gold_retrieved_wrong_cases(rows, args.max_cases_per_attr)
    error_rows = error_type_summary(rows)
    write_csv(output_dir / "attribute_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "attribute_error_cases.csv", case_rows, CASE_FIELDS)
    write_csv(output_dir / "gold_retrieved_wrong_cases.csv", gold_retrieved_cases, GOLD_RETRIEVED_FIELDS)
    write_csv(output_dir / "company_language_error_type_summary.csv", error_rows, ["mode", "k_updates", "attribute", "error_type", "n"])
    write_markdown(output_dir / "attribute_failure_analysis.md", summary_rows, case_rows, error_rows)
    write_gold_retrieved_markdown(output_dir / "gold_retrieved_wrong_cases.md", gold_retrieved_cases)
    print(json.dumps({"num_examples": len(rows), "summary_rows": len(summary_rows), "case_rows": len(case_rows), "gold_retrieved_wrong_cases": len(gold_retrieved_cases), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
