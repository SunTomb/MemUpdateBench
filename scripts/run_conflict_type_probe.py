from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.utils import compute_exact_match, compute_f1, generate_text, load_model_and_tokenizer, set_seed
from scripts.eval_evomemory import answer_contains_value, build_slot_answer_prompt, slot_value_match

ATTR_VALUES = {
    "location": ["Dalian", "Qingdao", "Wuhan", "Ningbo", "Suzhou", "Xiamen", "Harbin", "Chengdu"],
    "company": ["Tencent", "Alibaba", "Baidu", "ByteDance", "Meituan", "JD", "Huawei", "NetEase"],
    "language": ["Python", "Java", "Go", "Rust", "Kotlin", "Scala", "TypeScript", "C++"],
    "preference": ["espresso", "oolong tea", "matcha", "latte", "mocha", "cold brew", "green tea", "black coffee"],
}

RELATIONS = ["friend", "coworker", "sister", "brother", "manager", "neighbor", "advisor", "teammate"]
NAMES = ["Alice", "Bob", "Lily", "Wang", "Ivy", "Hank", "Nora", "Leo"]
CONDITIONS = [
    "final_only",
    "unrelated_distractors",
    "same_entity_different_attribute",
    "different_entity_same_attribute",
    "stale_same_slot",
]
FIELDNAMES = [
    "condition",
    "example_id",
    "attribute",
    "entity",
    "gold_answer",
    "distractor_count",
    "predicted",
    "em",
    "f1",
    "value_em",
    "answer_value_present",
    "stale_value_copied",
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


def question_for(relation: str, name: str, attribute: str) -> str:
    if attribute == "location":
        return f"Where does my {relation} {name} currently live?"
    if attribute == "company":
        return f"Which company does my {relation} {name} currently work for?"
    if attribute == "language":
        return f"What programming language does my {relation} {name} currently prefer?"
    if attribute == "preference":
        return f"What drink does my {relation} {name} currently prefer?"
    return f"What is my {relation} {name}'s current {attribute}?"


def choose_other_attribute(attribute: str, offset: int) -> str:
    attrs = list(ATTR_VALUES)
    candidates = [item for item in attrs if item != attribute]
    return candidates[offset % len(candidates)]


def choose_other_relation(relation: str, offset: int) -> str:
    candidates = [item for item in RELATIONS if item != relation]
    return candidates[offset % len(candidates)]


def build_distractors(
    condition: str,
    relation: str,
    name: str,
    attribute: str,
    gold: str,
    count: int,
    example_id: int,
) -> tuple[list[str], list[str]]:
    distractors = []
    stale_values = []
    if condition == "final_only":
        return distractors, stale_values
    for idx in range(count):
        if condition == "unrelated_distractors":
            other_attr = choose_other_attribute(attribute, idx)
            other_relation = choose_other_relation(relation, idx)
            other_name = NAMES[(example_id + idx + 1) % len(NAMES)]
            value = ATTR_VALUES[other_attr][(example_id + idx) % len(ATTR_VALUES[other_attr])]
            distractors.append(event_text(other_relation, other_name, other_attr, value, latest=False))
        elif condition == "same_entity_different_attribute":
            other_attr = choose_other_attribute(attribute, idx)
            value = ATTR_VALUES[other_attr][(example_id + idx + 2) % len(ATTR_VALUES[other_attr])]
            distractors.append(event_text(relation, name, other_attr, value, latest=False))
        elif condition == "different_entity_same_attribute":
            other_relation = choose_other_relation(relation, idx)
            other_name = NAMES[(example_id + idx + 3) % len(NAMES)]
            value = ATTR_VALUES[attribute][(example_id + idx + 1) % len(ATTR_VALUES[attribute])]
            if slot_value_match(value, gold):
                value = ATTR_VALUES[attribute][(example_id + idx + 2) % len(ATTR_VALUES[attribute])]
            distractors.append(event_text(other_relation, other_name, attribute, value, latest=False))
        elif condition == "stale_same_slot":
            values = [value for value in ATTR_VALUES[attribute] if not slot_value_match(value, gold)]
            value = values[(example_id + idx) % len(values)]
            stale_values.append(value)
            distractors.append(event_text(relation, name, attribute, value, latest=False))
        else:
            raise ValueError(f"unknown condition: {condition}")
    return distractors, stale_values


def make_example(example_id: int, attribute: str, condition: str, distractor_count: int) -> dict[str, Any]:
    relation = RELATIONS[example_id % len(RELATIONS)]
    name = NAMES[(example_id // len(RELATIONS)) % len(NAMES)]
    entity = f"{relation}_{name.lower()}"
    values = ATTR_VALUES[attribute]
    gold = values[(example_id + distractor_count + 3) % len(values)]
    final_entry = event_text(relation, name, attribute, gold, latest=True)
    distractors, stale_values = build_distractors(condition, relation, name, attribute, gold, distractor_count, example_id)
    context_entries = distractors + [final_entry]
    context = "\n".join(f"- {entry}" for entry in context_entries)
    question = question_for(relation, name, attribute)
    prompt = build_slot_answer_prompt(question, context, {"entity": entity, "attribute": attribute, "answer": gold}, "v0_current")
    return {
        "condition": condition,
        "example_id": example_id,
        "attribute": attribute,
        "entity": entity,
        "gold_answer": gold,
        "distractor_count": distractor_count,
        "stale_values": stale_values,
        "prompt": prompt,
    }


def stale_value_copied(predicted: str, stale_values: list[str]) -> bool:
    return any(answer_contains_value(predicted, value) for value in stale_values)


def run_condition(model, tokenizer, args: argparse.Namespace, condition: str) -> list[dict[str, Any]]:
    rows = []
    attributes = [item for item in args.attributes.split(",") if item]
    for example_id in range(args.examples_per_condition):
        attribute = attributes[example_id % len(attributes)]
        example = make_example(example_id, attribute, condition, args.distractor_count)
        predicted = generate_text(model, tokenizer, example["prompt"], max_new_tokens=32, temperature=0.1, do_sample=False)
        predicted = predicted.strip().split("\n")[0].strip()
        gold = example["gold_answer"]
        rows.append({
            "condition": condition,
            "example_id": example_id,
            "attribute": attribute,
            "entity": example["entity"],
            "gold_answer": gold,
            "distractor_count": args.distractor_count,
            "predicted": predicted,
            "em": compute_exact_match(predicted, gold),
            "f1": compute_f1(predicted, gold),
            "value_em": slot_value_match(predicted, gold),
            "answer_value_present": answer_contains_value(predicted, gold),
            "stale_value_copied": stale_value_copied(predicted, example["stale_values"]),
            "prompt": example["prompt"],
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    output = []
    final_only_em = None
    if "final_only" in grouped:
        final_only_em = mean(float(item["em"]) for item in grouped["final_only"])
    for condition, items in sorted(grouped.items()):
        em = mean(float(item["em"]) for item in items)
        output.append({
            "condition": condition,
            "n": len(items),
            "distractor_count": items[0]["distractor_count"],
            "em": em,
            "f1": mean(float(item["f1"]) for item in items),
            "value_em": mean(float(item["value_em"]) for item in items),
            "answer_value_present": mean(float(item["answer_value_present"]) for item in items),
            "stale_value_copied": mean(float(item["stale_value_copied"]) for item in items),
            "em_drop_from_final_only": 0.0 if final_only_em is None else final_only_em - em,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Conflict-Type Probe Summary",
        "",
        "| condition | n | distractors | EM | F1 | value EM | answer value present | stale copied | EM drop vs final-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['distractor_count']} | "
            f"{row['em']:.3f} | {row['f1']:.3f} | {row['value_em']:.3f} | "
            f"{row['answer_value_present']:.3f} | {row['stale_value_copied']:.3f} | "
            f"{row['em_drop_from_final_only']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled conflict-type context probes")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="results/p83_conflict_type_probe")
    parser.add_argument("--examples_per_condition", type=int, default=64)
    parser.add_argument("--attributes", default="location,company,language,preference")
    parser.add_argument("--distractor_count", type=int, default=4)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--no_qlora", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_qlora=not args.no_qlora)
    rows = []
    for condition in [item for item in args.conditions.split(",") if item]:
        rows.extend(run_condition(model, tokenizer, args, condition))
        print(f"finished condition={condition}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    write_csv(output_dir / "conflict_type_examples.csv", rows, FIELDNAMES)
    write_csv(output_dir / "conflict_type_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    (output_dir / "conflict_type_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "conflict_type_summary.md", summary_rows)
    print(json.dumps({"num_examples": len(rows), "num_conditions": len(summary_rows), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
