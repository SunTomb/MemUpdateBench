from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.config import MUBConfig
from mub.memory.store import MemoryStore
from mub.utils import compute_exact_match, compute_f1, generate_text, load_model_and_tokenizer, set_seed
from scripts.eval_evomemory import (
    answer_contains_value,
    answer_question,
    build_slot_answer_prompt,
    compute_state_metrics,
    retrieved_trace,
    run_constrained_slot_crud,
    run_raw_add,
    slot_value_match,
)


def run_memory_mode(mode: str, store: MemoryStore, events: list[str]) -> tuple[dict, list[dict]]:
    if mode == "constrained_slot_crud":
        return run_constrained_slot_crud(store, events)
    if mode == "raw_add":
        return run_raw_add(store, events), []
    raise ValueError(f"Unsupported mode for answer-layer diagnostics: {mode}")


def gold_context(example: dict) -> str:
    events = example.get("events", [])
    latest_idx = int(example.get("latest_event_idx", len(events) - 1))
    if 0 <= latest_idx < len(events):
        return f"- {events[latest_idx]}"
    return f"- {example.get('entity', 'user')}.{example.get('attribute', 'unknown')} = {example.get('answer', '')}"


def answer_with_gold_context(model, tokenizer, example: dict, variant: str) -> tuple[str, dict]:
    question = example.get("question", "")
    context = gold_context(example)
    prompt = build_slot_answer_prompt(
        question,
        context,
        {
            "entity": example.get("entity", "user"),
            "attribute": example.get("attribute", "unknown"),
            "answer": example.get("answer", ""),
        },
        variant,
    )
    answer = generate_text(model, tokenizer, prompt, max_new_tokens=64, temperature=0.1, do_sample=False)
    answer = answer.strip().split("\n")[0].strip()
    trace = {
        "prompt": prompt,
        "slot_prompt_variant": variant,
        "context_mode": "gold_context",
        "gold_context": context,
        "gold_value_in_retrieved": True,
        "stale_same_slot_in_retrieved": False,
        "same_name_distractor_in_retrieved": False,
        "stale_same_slot_values": [],
        "retrieved_entries": [],
    }
    return answer, trace


def evaluate(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    config = MUBConfig()
    model, tokenizer = load_model_and_tokenizer(config.model.model_name, use_qlora=not args.no_qlora)
    if args.lora_checkpoint and os.path.exists(os.path.join(args.lora_checkpoint, "adapter_config.json")):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_checkpoint)

    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    rows = []
    for i, example in enumerate(data):
        store = MemoryStore(config.memory)
        exec_stats, slot_actions = run_memory_mode(args.mode, store, example.get("events", []))
        state_metrics = compute_state_metrics(example, store)
        question = example.get("question", "")
        gold = str(example.get("answer", ""))

        for topk in args.answer_topks:
            predicted, trace = answer_question(
                model,
                tokenizer,
                question,
                store,
                slot_prompt={
                    "entity": example.get("entity", "user"),
                    "attribute": example.get("attribute", "unknown"),
                    "answer": gold,
                },
                slot_prompt_variant=args.slot_prompt_variant,
                answer_topk=topk,
                return_trace=True,
            )
            rows.append(make_row(i, example, args.mode, f"retrieval_topk_{topk}", predicted, gold, trace, state_metrics, store.size()))

        if args.include_gold_context:
            predicted, trace = answer_with_gold_context(model, tokenizer, example, args.slot_prompt_variant)
            rows.append(make_row(i, example, args.mode, "gold_context", predicted, gold, trace, state_metrics, store.size()))

        if (i + 1) % max(args.log_every, 1) == 0 or i + 1 == len(data):
            print(f"Progress: {i + 1}/{len(data)}")

    summary_rows = []
    for context_mode in sorted({row["context_mode"] for row in rows}):
        subset = [row for row in rows if row["context_mode"] == context_mode]
        summary_rows.append({
            "mode": args.mode,
            "context_mode": context_mode,
            "num_examples": len(subset),
            "avg_em": avg(subset, "em"),
            "avg_f1": avg(subset, "f1"),
            "value_em": avg(subset, "value_em"),
            "answer_value_present": avg(subset, "answer_value_present"),
            "state_accuracy": avg(subset, "state_value_em"),
            "gold_retrieved_rate": avg(subset, "gold_value_in_retrieved"),
            "stale_retrieved_rate": avg(subset, "stale_same_slot_in_retrieved"),
            "distractor_retrieved_rate": avg(subset, "same_name_distractor_in_retrieved"),
        })

    return {
        "summary": {
            "data_file": args.data_file,
            "mode": args.mode,
            "slot_prompt_variant": args.slot_prompt_variant,
            "num_examples": len(data),
            "answer_topks": args.answer_topks,
            "include_gold_context": args.include_gold_context,
            "rows": summary_rows,
        },
        "results": rows,
    }


def make_row(example_id: int, example: dict, mode: str, context_mode: str, predicted: str, gold: str, trace: dict, state_metrics: dict, memory_size: int) -> dict:
    em = compute_exact_match(predicted, gold)
    f1 = compute_f1(predicted, gold)
    return {
        "example_id": example_id,
        "mode": mode,
        "context_mode": context_mode,
        "question": example.get("question", ""),
        "gold_answer": gold,
        "predicted": predicted,
        "em": em,
        "f1": f1,
        "value_em": slot_value_match(predicted, gold),
        "answer_value_present": answer_contains_value(predicted, gold),
        "k_updates": example.get("k_updates"),
        "memory_size": memory_size,
        **state_metrics,
        "gold_value_in_retrieved": bool(trace.get("gold_value_in_retrieved", False)),
        "stale_same_slot_in_retrieved": bool(trace.get("stale_same_slot_in_retrieved", False)),
        "same_name_distractor_in_retrieved": bool(trace.get("same_name_distractor_in_retrieved", False)),
        "stale_same_slot_values": trace.get("stale_same_slot_values", []),
        "answer_trace": trace,
    }


def avg(rows: list[dict], key: str) -> float:
    return mean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


def write_markdown(payload: dict, output_path: Path) -> None:
    lines = [
        "# Answer-Layer Mechanism Summary",
        "",
        "| Mode | Context | N | EM | F1 | Value EM | Answer value present | State acc. | Gold retrieved | Stale retrieved | Distractor retrieved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["rows"]:
        lines.append(
            f"| {row['mode']} | {row['context_mode']} | {row['num_examples']} | "
            f"{row['avg_em']:.3f} | {row['avg_f1']:.3f} | {row['value_em']:.3f} | "
            f"{row['answer_value_present']:.3f} | {row['state_accuracy']:.3f} | "
            f"{row['gold_retrieved_rate']:.3f} | {row['stale_retrieved_rate']:.3f} | "
            f"{row['distractor_retrieved_rate']:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle/gold-context answer-layer diagnostics")
    parser.add_argument("--mode", choices=["constrained_slot_crud", "raw_add"], required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--answer_topks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--slot_prompt_variant", default="v0_current", choices=["v0_current", "v1_value_only", "v2_ignore_distractors"])
    parser.add_argument("--include_gold_context", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--lora_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    payload = evaluate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answer_layer_mechanism.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, output_dir / "answer_layer_mechanism.md")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
