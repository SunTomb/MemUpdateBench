from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mub.utils import compute_exact_match, compute_f1, generate_text, load_model_and_tokenizer, set_seed
from scripts.eval_evomemory import answer_contains_value, build_slot_answer_prompt, parse_event_slot, slot_value_match


def target_event_indices(example: dict[str, Any]) -> list[int]:
    entity = example.get("entity", "user")
    attribute = example.get("attribute", "unknown")
    indices = []
    for idx, event in enumerate(example.get("events", [])):
        slot = parse_event_slot(event, idx)
        if slot and slot.get("entity") == entity and slot.get("attribute") == attribute:
            indices.append(idx)
    return indices


def make_entry(content: str, slot: dict[str, Any], tag: str) -> dict[str, Any]:
    return {"content": content, "slot": slot, "tag": tag}


def build_context_entries(example: dict[str, Any], num_distractors: int) -> list[dict[str, Any]]:
    entity = example.get("entity", "user")
    attribute = example.get("attribute", "unknown")
    gold = str(example.get("answer", ""))
    target_indices = target_event_indices(example)
    if not target_indices:
        return []
    latest_idx = max(target_indices)
    gold_event = example["events"][latest_idx]
    gold_slot = parse_event_slot(gold_event, latest_idx) or {
        "entity": entity,
        "attribute": attribute,
        "value": gold,
        "event_idx": latest_idx,
    }
    gold_entry = make_entry(gold_event, gold_slot, "gold")

    distractors = []
    for idx, event in enumerate(example.get("events", [])):
        if idx == latest_idx:
            continue
        slot = parse_event_slot(event, idx)
        if not slot:
            continue
        tag = "stale_same_slot" if slot.get("entity") == entity and slot.get("attribute") == attribute else "near_miss"
        distractors.append(make_entry(event, slot, tag))
    distractors.sort(key=lambda row: (row["tag"] != "stale_same_slot", -int(row["slot"].get("event_idx", -1))))
    return [gold_entry] + distractors[:num_distractors]


def order_entries(entries: list[dict[str, Any]], gold_position: str) -> list[dict[str, Any]]:
    gold = [entry for entry in entries if entry["tag"] == "gold"]
    others = [entry for entry in entries if entry["tag"] != "gold"]
    if not gold:
        return entries
    gold_entry = gold[0]
    if gold_position == "beginning":
        return [gold_entry] + others
    if gold_position == "end":
        return others + [gold_entry]
    if gold_position == "middle":
        mid = len(others) // 2
        return others[:mid] + [gold_entry] + others[mid:]
    raise ValueError(f"Unknown gold_position: {gold_position}")


def format_context(entries: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {entry['content']}" for entry in entries)


def run_condition(model: Any, tokenizer: Any, examples: list[dict[str, Any]], args: argparse.Namespace, gold_position: str) -> list[dict[str, Any]]:
    rows = []
    for idx, example in enumerate(examples):
        entries = build_context_entries(example, args.num_distractors)
        ordered = order_entries(entries, gold_position)
        if not ordered:
            continue
        memory_context = format_context(ordered)
        slot_prompt = {
            "entity": example.get("entity", "user"),
            "attribute": example.get("attribute", "unknown"),
            "answer": example.get("answer", ""),
        }
        prompt = build_slot_answer_prompt(example["question"], memory_context, slot_prompt, args.slot_prompt_variant)
        prediction = generate_text(model, tokenizer, prompt, max_new_tokens=64, temperature=0.1).strip().split("\n")[0].strip()
        gold = example["answer"]
        tag_sequence = [entry["tag"] for entry in ordered]
        rows.append({
            "example_id": idx,
            "question": example["question"],
            "gold_answer": gold,
            "predicted": prediction,
            "em": compute_exact_match(prediction, gold),
            "f1": compute_f1(prediction, gold),
            "value_em": slot_value_match(prediction, gold),
            "answer_value_present": answer_contains_value(prediction, gold),
            "gold_position": gold_position,
            "context_size": len(ordered),
            "gold_index": tag_sequence.index("gold"),
            "tag_sequence": tag_sequence,
            "memory_context": memory_context,
        })
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    by_position: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_position.setdefault(row["gold_position"], []).append(row)
    conditions = []
    for position, items in sorted(by_position.items()):
        conditions.append({
            "gold_position": position,
            "num_examples": len(items),
            "avg_em": sum(float(item["em"]) for item in items) / len(items),
            "avg_f1": sum(float(item["f1"]) for item in items) / len(items),
            "answer_value_present_rate": sum(float(item["answer_value_present"]) for item in items) / len(items),
        })
    return {
        "benchmark": "lost_in_middle_gold_position_probe",
        "model_name": args.model_name,
        "data_file": args.data_file,
        "num_examples_per_position": len(next(iter(by_position.values()))) if by_position else 0,
        "num_distractors": args.num_distractors,
        "slot_prompt_variant": args.slot_prompt_variant,
        "conditions": conditions,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled gold-position Lost-in-the-Middle probe")
    parser.add_argument("--data_file", default="data/evomemory_update_frequency_hard_k16_p63_dev.json")
    parser.add_argument("--output_dir", default="results/p80_lost_in_middle_probe/qwen_k16_dev")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument("--num_distractors", type=int, default=8)
    parser.add_argument("--slot_prompt_variant", default="v0_current")
    args = parser.parse_args()

    set_seed(42)
    examples = json.loads(Path(args.data_file).read_text(encoding="utf-8"))[: args.end_idx]
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_qlora=False, load_in_4bit=False)
    start = time.time()
    rows = []
    for position in ["beginning", "middle", "end"]:
        rows.extend(run_condition(model, tokenizer, examples, args, position))
    elapsed = time.time() - start
    summary = summarize(rows, args, elapsed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lost_in_middle_results.json").write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
