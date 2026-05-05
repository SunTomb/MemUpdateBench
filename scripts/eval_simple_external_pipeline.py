from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from mub.config import MUBConfig
from mub.memory.store import MemoryStore
from mub.utils import compute_exact_match, compute_f1, set_seed
from scripts.eval_evomemory import (
    EpisodeEntityResolver,
    answer_contains_value,
    answer_from_slot,
    answer_question,
    compute_state_metrics,
    filter_latest_per_slot,
    new_exec_stats,
    parse_event_slot,
    slot_value_match,
)


def should_store(slot_meta: dict[str, Any] | None, policy: str) -> bool:
    if policy == "all_events":
        return True
    if policy == "parsed_only":
        return slot_meta is not None
    raise ValueError(f"Unknown store_policy: {policy}")


def run_extract_then_store(
    store: MemoryStore,
    events: list[str],
    update_policy: str,
    store_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats = new_exec_stats()
    actions = []
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        action = {
            "event_idx": idx,
            "operation": "NOOP",
            "entity": None,
            "attribute": None,
            "value": None,
            "content": event,
        }
        if slot_meta:
            action.update({
                "entity": slot_meta.get("entity"),
                "attribute": slot_meta.get("attribute"),
                "value": slot_meta.get("value"),
            })
        if not should_store(slot_meta, store_policy):
            stats["requested"]["NOOP"] += 1
            stats["effective"]["NOOP"] += 1
            stats["success"] += 1
            actions.append(action)
            continue
        if update_policy == "append":
            entry = store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
            action["operation"] = "ADD"
            action["entry_id"] = entry.id
            stats["requested"]["ADD"] += 1
            stats["effective"]["ADD"] += 1
            stats["success"] += 1
        elif update_policy == "slot_update" and slot_meta:
            target = store.get_latest_by_slot(slot_meta.get("entity", ""), slot_meta.get("attribute", ""))
            if target:
                result = store.update(target.id, event, env_reward=0.5, slot_meta=slot_meta)
                action["operation"] = "UPDATE"
                action["target_id"] = target.id
                stats["requested"]["UPDATE"] += 1
                stats["effective"]["UPDATE"] += 1
                stats["success"] += int(result is not None)
                stats["failed"] += int(result is None)
            else:
                entry = store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
                action["operation"] = "ADD"
                action["entry_id"] = entry.id
                stats["requested"]["ADD"] += 1
                stats["effective"]["ADD"] += 1
                stats["success"] += 1
        elif update_policy == "slot_update":
            entry = store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
            action["operation"] = "ADD"
            action["entry_id"] = entry.id
            stats["requested"]["ADD"] += 1
            stats["effective"]["ADD"] += 1
            stats["success"] += 1
        else:
            raise ValueError(f"Unknown update_policy: {update_policy}")
        actions.append(action)
    return stats, actions


def memory_records(store: MemoryStore) -> list[dict[str, Any]]:
    records = []
    for entry in store.entries.values():
        records.append({
            "id": entry.id,
            "content": entry.content,
            "slot": entry.slot,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        })
    return records


def summarize(results: list[dict[str, Any]], elapsed: float, args: argparse.Namespace, data_path: str) -> dict[str, Any]:
    def avg(key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in results]
        return float(mean(values)) if values else 0.0

    total_exec_stats = new_exec_stats()
    for row in results:
        stats = row["exec_stats"]
        for bucket in ("requested", "effective"):
            for op, count in stats[bucket].items():
                total_exec_stats[bucket][op] = total_exec_stats[bucket].get(op, 0) + count
        total_exec_stats["success"] += stats["success"]
        total_exec_stats["failed"] += stats["failed"]
        total_exec_stats["fallback_target"] += stats["fallback_target"]
    return {
        "benchmark": "simple_external_pipeline",
        "mode": "simple_extract_then_store",
        "model_name": args.model_name,
        "data_file": data_path,
        "num_examples": len(results),
        "total_examples": args.total_examples,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "update_policy": args.update_policy,
        "store_policy": args.store_policy,
        "answer_mode": args.answer_mode,
        "answer_topk": args.answer_topk,
        "retrieval_policy": args.retrieval_policy,
        "avg_em": avg("em"),
        "avg_f1": avg("f1"),
        "value_em": avg("value_em"),
        "answer_value_present_rate": avg("answer_value_present"),
        "state_accuracy": avg("state_value_em"),
        "state_resolve_rate": avg("state_resolved"),
        "gold_present_rate": avg("gold_value_present_anywhere"),
        "stale_present_rate": avg("stale_value_present_same_slot"),
        "same_slot_entry_count_avg": avg("same_slot_entry_count"),
        "stale_same_slot_entry_count_avg": avg("stale_same_slot_entry_count"),
        "gold_same_slot_entry_count_avg": avg("gold_same_slot_entry_count"),
        "final_memory_size_avg": avg("memory_size"),
        "exec_stats": total_exec_stats,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple extract-then-store external memory pipeline")
    parser.add_argument("--data_file", default="data/evomemory_update_frequency_hard_k16_p63_dev.json")
    parser.add_argument("--output_dir", default="results/p80_simple_external_pipeline/simple_slot_update_k16_dev")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--update_policy", default="slot_update", choices=["append", "slot_update"])
    parser.add_argument("--store_policy", default="parsed_only", choices=["all_events", "parsed_only"])
    parser.add_argument("--answer_mode", default="slot_direct", choices=["slot_direct", "slot_prompt"])
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--answer_topk", type=int, default=5)
    parser.add_argument("--retrieval_policy", default="normal", choices=["normal", "latest_per_slot"])
    parser.add_argument("--save_memory_records", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    config = MUBConfig()
    data_path = Path(args.data_file)
    examples = json.loads(data_path.read_text(encoding="utf-8"))
    args.total_examples = len(examples)
    end_idx = args.end_idx if args.end_idx is not None else len(examples)
    end_idx = min(end_idx, len(examples))
    args.end_idx = end_idx
    examples = examples[args.start_idx:end_idx]

    model = None
    tokenizer = None
    if args.answer_mode == "slot_prompt":
        from mub.utils import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_name, use_qlora=False, load_in_4bit=False)

    results = []
    start = time.time()
    for local_id, example in enumerate(examples):
        original_id = args.start_idx + local_id
        store = MemoryStore(config.memory)
        exec_stats, actions = run_extract_then_store(store, example["events"], args.update_policy, args.store_policy)
        gold_answer = example["answer"]
        if args.answer_mode == "slot_direct":
            predicted = answer_from_slot(example, store)
            answer_trace = None
        else:
            answer_output = answer_question(
                model,
                tokenizer,
                example["question"],
                store,
                slot_prompt={
                    "entity": example.get("entity", "user"),
                    "attribute": example.get("attribute", "unknown"),
                    "answer": gold_answer,
                },
                answer_topk=args.answer_topk,
                retrieval_policy=args.retrieval_policy,
                return_trace=True,
            )
            predicted, answer_trace = answer_output
        f1 = compute_f1(predicted, gold_answer)
        em = compute_exact_match(predicted, gold_answer)
        state_metrics = compute_state_metrics(example, store)
        row = {
            "example_id": original_id,
            "question": example["question"],
            "gold_answer": gold_answer,
            "predicted": predicted,
            "f1": f1,
            "em": em,
            "value_em": slot_value_match(predicted, gold_answer),
            "answer_value_present": answer_contains_value(predicted, gold_answer),
            "num_events": len(example["events"]),
            "k_updates": example.get("k_updates"),
            "memory_size": store.size(),
            "exec_stats": exec_stats,
            "slot_actions": actions,
            **state_metrics,
        }
        if answer_trace is not None:
            row["answer_trace"] = answer_trace
        if args.save_memory_records:
            row["memory_records"] = memory_records(store)
        results.append(row)

    elapsed = time.time() - start
    summary = summarize(results, elapsed, args, str(data_path))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evomemory_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
