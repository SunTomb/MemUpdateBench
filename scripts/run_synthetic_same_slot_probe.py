from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


SELECTED_V4_CELLS = {
    ("conflict", "reverse_chronological", "none", 1),
    ("conflict", "reverse_chronological", "none", 2),
    ("conflict", "reverse_chronological", "none", 4),
    ("conflict", "reverse_chronological", "none", 8),
    ("conflict", "reverse_chronological", "none", 16),
    ("conflict", "reverse_chronological", "latest_outdated_label", 1),
    ("conflict", "reverse_chronological", "latest_outdated_label", 2),
    ("conflict", "reverse_chronological", "latest_outdated_label", 4),
    ("conflict", "reverse_chronological", "latest_outdated_label", 8),
    ("conflict", "reverse_chronological", "latest_outdated_label", 16),
    ("conflict", "chronological", "none", 1),
    ("conflict", "chronological", "none", 2),
    ("conflict", "chronological", "none", 4),
    ("conflict", "chronological", "none", 8),
    ("conflict", "chronological", "none", 16),
    ("same_as_current", "chronological", "none", 4),
    ("same_as_current", "chronological", "none", 8),
    ("same_as_current", "chronological", "none", 16),
    ("same_as_current", "reverse_chronological", "none", 4),
    ("same_as_current", "reverse_chronological", "none", 8),
    ("same_as_current", "reverse_chronological", "none", 16),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.utils import compute_exact_match, compute_f1, generate_text, load_model_and_tokenizer, set_seed
from scripts.eval_evomemory import build_slot_answer_prompt


VALUES = {
    "location": ["Dalian", "Qingdao", "Wuhan", "Ningbo", "Suzhou", "Xiamen", "Harbin", "Chengdu", "Jinan", "Hefei"],
    "company": ["Tencent", "Alibaba", "Baidu", "ByteDance", "Meituan", "JD", "Huawei", "NetEase", "Google", "Microsoft"],
    "language": ["Python", "Java", "Go", "Rust", "Kotlin", "Scala", "TypeScript", "C++", "Swift", "Ruby"],
    "preference": ["espresso", "oolong tea", "matcha", "latte", "mocha", "cold brew", "green tea", "black coffee", "cocoa", "jasmine tea"],
}

RELATIONS = ["friend", "coworker", "sister", "brother", "manager", "neighbor"]
NAMES = ["Alice", "Bob", "Lily", "Wang", "Ivy", "Hank", "Nora", "Leo"]


FIELDNAMES = [
    "condition",
    "example_id",
    "attribute",
    "stale_count",
    "value_policy",
    "context_order",
    "context_annotation",
    "gold_answer",
    "predicted",
    "em",
    "f1",
    "answer_value_present",
    "prompt",
]


def event_text(relation: str, name: str, attribute: str, value: str, latest: bool) -> str:
    if attribute == "location":
        verb = "relocated to" if latest else "lives in"
        return f"User says: my {relation} {name} {verb} {value}."
    if attribute == "company":
        verb = "switched to" if latest else "works at"
        return f"User says: my {relation} {name} {verb} {value}."
    if attribute == "language":
        verb = "now codes in" if latest else "programming language is"
        return f"User says: my {relation} {name} {verb} {value}."
    if attribute == "preference":
        verb = "started preferring" if latest else "prefers"
        return f"User says: my {relation} {name} {verb} {value}."
    return f"User says: my {relation} {name} {attribute} is {value}."


def build_context(
    relation: str,
    name: str,
    attribute: str,
    gold: str,
    stale_values: list[str],
    context_order: str,
    context_annotation: str,
) -> str:
    entries = []
    for idx, value in enumerate(stale_values):
        entries.append({"idx": idx, "latest": False, "text": event_text(relation, name, attribute, value, latest=False)})
    entries.append({"idx": len(stale_values), "latest": True, "text": event_text(relation, name, attribute, gold, latest=True)})

    if context_order == "current_first":
        entries = sorted(entries, key=lambda item: (not item["latest"], item["idx"]))
    elif context_order == "current_last":
        entries = sorted(entries, key=lambda item: (item["latest"], item["idx"]))
    elif context_order == "reverse_chronological":
        entries = sorted(entries, key=lambda item: item["idx"], reverse=True)
    elif context_order == "chronological":
        entries = sorted(entries, key=lambda item: item["idx"])
    elif context_order != "normal":
        raise ValueError(f"unknown context_order: {context_order}")

    lines = []
    for entry in entries:
        prefix = ""
        if context_annotation == "timestamp":
            prefix = f"[event_idx={entry['idx']}] "
        elif context_annotation == "latest_outdated_label":
            prefix = "[latest] " if entry["latest"] else "[outdated] "
        elif context_annotation != "none":
            raise ValueError(f"unknown context_annotation: {context_annotation}")
        lines.append(f"- {prefix}{entry['text']}")
    return "\n".join(lines)


def make_example(example_id: int, attribute: str, stale_count: int, value_policy: str, context_order: str, context_annotation: str) -> dict[str, Any]:
    relation = RELATIONS[example_id % len(RELATIONS)]
    name = NAMES[(example_id // len(RELATIONS)) % len(NAMES)]
    values = VALUES[attribute]
    gold = values[(example_id + stale_count + 3) % len(values)]
    if value_policy == "conflict":
        stale_values = [value for value in values if value != gold]
        while len(stale_values) < stale_count:
            stale_values.extend([value for value in values if value != gold])
        stale_values = stale_values[:stale_count]
    elif value_policy == "same_as_current":
        stale_values = [gold] * stale_count
    elif value_policy == "mixed":
        stale_values = []
        for i in range(stale_count):
            stale_values.append(gold if i % 2 == 0 else values[(example_id + i) % len(values)])
    else:
        raise ValueError(f"unknown value_policy: {value_policy}")
    entity = f"{relation}_{name.lower()}"
    question = {
        "location": f"Where does my {relation} {name} currently live?",
        "company": f"Which company does my {relation} {name} currently work for?",
        "language": f"What programming language does my {relation} {name} currently prefer?",
        "preference": f"What drink does my {relation} {name} currently prefer?",
    }[attribute]
    context = build_context(relation, name, attribute, gold, stale_values, context_order, context_annotation)
    prompt = build_slot_answer_prompt(question, context, {"entity": entity, "attribute": attribute, "answer": gold}, "v0_current")
    return {
        "condition": f"{value_policy}_stale{stale_count}_{context_order}_{context_annotation}",
        "example_id": example_id,
        "attribute": attribute,
        "stale_count": stale_count,
        "value_policy": value_policy,
        "context_order": context_order,
        "context_annotation": context_annotation,
        "gold_answer": gold,
        "prompt": prompt,
    }


def answer_contains_value(predicted: str, gold: str) -> bool:
    return str(gold).strip().lower() in str(predicted).strip().lower()


def run_condition(model, tokenizer, args: argparse.Namespace, stale_count: int, value_policy: str, context_order: str, context_annotation: str) -> list[dict[str, Any]]:
    rows = []
    attributes = args.attributes.split(",")
    for i in range(args.examples_per_condition):
        attribute = attributes[i % len(attributes)]
        example = make_example(i, attribute, stale_count, value_policy, context_order, context_annotation)
        predicted = generate_text(model, tokenizer, example["prompt"], max_new_tokens=32, temperature=0.1, do_sample=False)
        predicted = predicted.strip().split("\n")[0].strip()
        gold = example["gold_answer"]
        rows.append({
            **example,
            "predicted": predicted,
            "em": compute_exact_match(predicted, gold),
            "f1": compute_f1(predicted, gold),
            "answer_value_present": answer_contains_value(predicted, gold),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    out = []
    for condition, items in sorted(groups.items()):
        first = items[0]
        out.append({
            "condition": condition,
            "n": len(items),
            "stale_count": first["stale_count"],
            "value_policy": first["value_policy"],
            "context_order": first["context_order"],
            "context_annotation": first["context_annotation"],
            "em": mean(float(item["em"]) for item in items),
            "f1": mean(float(item["f1"]) for item in items),
            "answer_value_present": mean(float(item["answer_value_present"]) for item in items),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P8.0 Synthetic Same-Slot Probe Summary",
        "",
        "| condition | n | stale | value policy | order | annotation | EM | F1 | answer value present |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['stale_count']} | {row['value_policy']} | "
            f"{row['context_order']} | {row['context_annotation']} | {row['em']:.3f} | {row['f1']:.3f} | {row['answer_value_present']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled synthetic same-slot context probes")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="results/p80_synthetic_same_slot_probe")
    parser.add_argument("--examples_per_condition", type=int, default=32)
    parser.add_argument("--attributes", default="location,company,language,preference")
    parser.add_argument("--stale_counts", default="0,1,2,4,8,16")
    parser.add_argument("--value_policies", default="conflict,same_as_current")
    parser.add_argument("--context_orders", default="chronological,reverse_chronological")
    parser.add_argument("--context_annotations", default="none,latest_outdated_label")
    parser.add_argument("--condition_set", default="full", choices=["full", "selected_v4"], help="Predefined condition subset to run")
    parser.add_argument("--no_qlora", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_qlora=not args.no_qlora)
    rows = []
    for stale_count in [int(item) for item in args.stale_counts.split(",") if item]:
        for value_policy in [item for item in args.value_policies.split(",") if item]:
            for context_order in [item for item in args.context_orders.split(",") if item]:
                for context_annotation in [item for item in args.context_annotations.split(",") if item]:
                    if args.condition_set == "selected_v4" and (
                        value_policy,
                        context_order,
                        context_annotation,
                        stale_count,
                    ) not in SELECTED_V4_CELLS:
                        continue
                    rows.extend(run_condition(model, tokenizer, args, stale_count, value_policy, context_order, context_annotation))
                    print(f"finished stale={stale_count} policy={value_policy} order={context_order} annotation={context_annotation}", flush=True)

    if args.condition_set == "selected_v4" and not rows:
        raise ValueError("selected_v4 condition set produced no rows; check filter arguments")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    write_csv(output_dir / "synthetic_same_slot_examples.csv", rows, FIELDNAMES)
    write_csv(output_dir / "synthetic_same_slot_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    (output_dir / "synthetic_same_slot_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "synthetic_same_slot_summary.md", summary_rows)
    print(json.dumps({"num_examples": len(rows), "num_conditions": len(summary_rows), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
