"""Analyze OOD EvoMemory learned constrained state errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def norm(value: str) -> str:
    import re
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def classify_error(result: dict) -> str:
    gold = result.get("gold_state", {})
    pred = result.get("predicted_state") or {}
    if not pred:
        return "missing_state"
    if pred.get("entity") != gold.get("entity"):
        return "wrong_entity"
    if pred.get("attribute") != gold.get("attribute"):
        return "wrong_attribute"
    if norm(pred.get("value", "")) != norm(gold.get("value", "")):
        return "wrong_value"
    return "correct"


def summarize(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]

    by_error = Counter()
    by_entity = defaultdict(Counter)
    by_attribute = defaultdict(Counter)
    by_pair = defaultdict(Counter)
    by_k_updates = defaultdict(Counter)
    by_stress_type = defaultdict(Counter)
    by_distractor_level = defaultdict(Counter)
    by_noop_level = defaultdict(Counter)
    action_ops = Counter()
    raw_outputs = Counter()
    same_slot_entry_counts = []
    stale_same_slot_entry_counts = []
    gold_same_slot_entry_counts = []
    write_amplifications = []
    target_slot_write_counts = []
    memory_sizes = []
    examples = []

    for result in results:
        error = classify_error(result)
        gold = result.get("gold_state", {})
        pred = result.get("predicted_state") or {}
        entity = gold.get("entity", "unknown")
        attribute = gold.get("attribute", "unknown")

        by_error[error] += 1
        by_entity[entity][error] += 1
        by_attribute[attribute][error] += 1
        by_pair[f"{entity}.{attribute}"][error] += 1
        by_k_updates[str(result.get("k_updates", "unknown"))][error] += 1
        by_stress_type[result.get("stress_type", "unknown")][error] += 1
        by_distractor_level[result.get("distractor_level", "unknown")][error] += 1
        by_noop_level[result.get("noop_level", "unknown")][error] += 1

        same_slot_entry_counts.append(result.get("same_slot_entry_count", 0))
        stale_same_slot_entry_counts.append(result.get("stale_same_slot_entry_count", 0))
        gold_same_slot_entry_counts.append(result.get("gold_same_slot_entry_count", 0))
        write_amplifications.append(result.get("write_amplification", 0.0))
        target_slot_write_counts.append(result.get("target_slot_write_count", 0))
        memory_sizes.append(result.get("memory_size", 0))

        for action in result.get("slot_actions", []):
            action_ops[action.get("operation", "UNKNOWN")] += 1
            raw = action.get("raw_output")
            if raw:
                raw_outputs[raw] += 1

        if error != "correct":
            examples.append({
                "example_id": result.get("example_id"),
                "question": result.get("question"),
                "gold_answer": result.get("gold_answer"),
                "predicted_answer": result.get("predicted"),
                "error": error,
                "stress_type": result.get("stress_type"),
                "k_updates": result.get("k_updates"),
                "distractor_level": result.get("distractor_level"),
                "noop_level": result.get("noop_level"),
                "gold_state": gold,
                "predicted_state": pred,
                "slot_actions_tail": result.get("slot_actions", [])[-6:],
            })

    total = len(results)
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "path": str(path),
        "total": total,
        "correct": by_error["correct"],
        "state_accuracy": by_error["correct"] / total if total else 0.0,
        "same_slot_entry_count_avg": avg(same_slot_entry_counts),
        "stale_same_slot_entry_count_avg": avg(stale_same_slot_entry_counts),
        "gold_same_slot_entry_count_avg": avg(gold_same_slot_entry_counts),
        "write_amplification_avg": avg(write_amplifications),
        "target_slot_write_count_avg": avg(target_slot_write_counts),
        "final_memory_size_avg": avg(memory_sizes),
        "by_error": dict(by_error),
        "by_entity": {k: dict(v) for k, v in sorted(by_entity.items())},
        "by_attribute": {k: dict(v) for k, v in sorted(by_attribute.items())},
        "by_pair": {k: dict(v) for k, v in sorted(by_pair.items())},
        "by_k_updates": {k: dict(v) for k, v in sorted(by_k_updates.items())},
        "by_stress_type": {k: dict(v) for k, v in sorted(by_stress_type.items())},
        "by_distractor_level": {k: dict(v) for k, v in sorted(by_distractor_level.items())},
        "by_noop_level": {k: dict(v) for k, v in sorted(by_noop_level.items())},
        "action_ops": dict(action_ops),
        "top_raw_outputs": raw_outputs.most_common(20),
        "error_examples": examples[:20],
    }


def main(args):
    summaries = [summarize(Path(path)) for path in args.inputs]
    output = {"summaries": summaries}
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OOD EvoMemory state errors")
    parser.add_argument("--inputs", nargs="+", required=True, help="evomemory_results.json files")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()
    main(args)
