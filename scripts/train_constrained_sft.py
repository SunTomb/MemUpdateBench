"""
Train and evaluate a constrained slot-action Memory Manager with SFT.

Expected action format:
    ADD friend_alex.location = Shanghai
    UPDATE friend_alex.location = Xiamen
    NOOP
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.config import MUBConfig
from mub.manager.memory_manager import MemoryManager
from mub.utils import load_model_and_tokenizer, set_seed


def load_jsonl(path: str, limit: int = 0) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def build_batch(example: dict, tokenizer, max_length: int, device) -> dict:
    prompt = example["prompt"]
    completion = " " + example["completion"]
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def normalize_value(value: str) -> str:
    import re
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def score_actions(prediction: str, gold: str) -> dict:
    pred = MemoryManager.parse_constrained_slot_operation(prediction)
    target = MemoryManager.parse_constrained_slot_operation(gold)
    invalid = pred["operation"] == "INVALID"
    return {
        "invalid": float(invalid),
        "operation": float(pred["operation"] == target["operation"]),
        "entity": float(pred["entity"] == target["entity"]),
        "attribute": float(pred["attribute"] == target["attribute"]),
        "value": float(normalize_value(pred["value"]) == normalize_value(target["value"])),
        "full": float(pred == target),
    }


@torch.no_grad()
def evaluate_actions(model, tokenizer, examples: list[dict], max_new_tokens: int = 32) -> dict:
    from mub.utils import generate_text

    totals = {
        "invalid": 0.0,
        "operation": 0.0,
        "entity": 0.0,
        "attribute": 0.0,
        "value": 0.0,
        "full": 0.0,
    }
    predictions = []
    for example in examples:
        pred = generate_text(
            model,
            tokenizer,
            example["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        ).strip().split("\n", 1)[0].strip()
        metrics = score_actions(pred, example["completion"])
        for key, value in metrics.items():
            totals[key] += value
        predictions.append({
            "gold": example["completion"],
            "predicted": pred,
            "metrics": metrics,
        })

    n = max(len(examples), 1)
    return {
        "num_examples": len(examples),
        "invalid_rate": totals["invalid"] / n,
        "operation_accuracy": totals["operation"] / n,
        "entity_accuracy": totals["entity"] / n,
        "attribute_accuracy": totals["attribute"] / n,
        "value_em": totals["value"] / n,
        "full_action_accuracy": totals["full"] / n,
        "predictions": predictions,
    }


def train(args):
    set_seed(args.seed)
    config = MUBConfig()
    train_data = load_jsonl(args.train_file, args.train_limit)
    dev_data = load_jsonl(args.dev_file, args.eval_limit)
    logger.info(f"Loaded train={len(train_data)} dev={len(dev_data)}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name or config.model.model_name,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
    )

    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    best_full = -1.0

    for epoch in range(args.num_epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        for idx, example in enumerate(train_data):
            batch = build_batch(example, tokenizer, args.max_length, model.device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if (idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.grad_accum_steps

            if args.max_steps and idx + 1 >= args.max_steps:
                break

        avg_loss = total_loss / max(min(len(train_data), args.max_steps or len(train_data)), 1)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} | loss={avg_loss:.4f}")

        eval_summary = evaluate_actions(model, tokenizer, dev_data[:args.eval_limit])
        logger.info(json.dumps({k: v for k, v in eval_summary.items() if k != "predictions"}, ensure_ascii=False))

        if eval_summary["full_action_accuracy"] > best_full:
            best_full = eval_summary["full_action_accuracy"]
            output_dir = Path(args.output_dir) / "best"
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
                json.dump(eval_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved best checkpoint to {output_dir}")

    logger.info(f"Training complete | best_full_action_accuracy={best_full:.4f}")


def evaluate_only(args):
    config = MUBConfig()
    examples = load_jsonl(args.dev_file, args.eval_limit)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name or config.model.model_name,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
    )
    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    summary = evaluate_actions(model, tokenizer, examples)
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train constrained slot-action SFT manager")
    parser.add_argument("--train_file", default="data/constrained_sft_hard_train.jsonl")
    parser.add_argument("--dev_file", default="data/constrained_sft_hard_dev.jsonl")
    parser.add_argument("--output_dir", default="outputs/constrained_sft")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--eval_limit", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_only:
        evaluate_only(args)
    else:
        train(args)
