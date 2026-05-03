from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.manager.memory_manager import MemoryManager
from scripts.eval_evomemory import EpisodeEntityResolver, parse_event_slot

NAME_RE = re.compile(r"_(?P<variant>v\d+_[a-z_]+)_k(?P<k>\d+)$|_k(?P<plain_k>\d+)$")


def norm(value: str) -> str:
    import re

    normalized = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_name(path: Path) -> tuple[str, int]:
    name = path.parent.name
    match = NAME_RE.search(name)
    if not match:
        raise ValueError(f"Cannot infer k from result directory: {name}")
    variant = match.group("variant") or ""
    k = int(match.group("k") or match.group("plain_k"))
    return variant, k


def gold_actions(events: list[str]) -> list[dict]:
    actions = []
    slot_state: dict[tuple[str, str], str] = {}
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot = parse_event_slot(event, idx, resolver=resolver)
        if not slot:
            action = {"event_idx": idx, "operation": "NOOP", "entity": "", "attribute": "", "value": ""}
        else:
            key = (slot["entity"], slot["attribute"])
            operation = "UPDATE" if key in slot_state else "ADD"
            action = {
                "event_idx": idx,
                "operation": operation,
                "entity": slot["entity"],
                "attribute": slot["attribute"],
                "value": slot["value"],
            }
            slot_state[key] = slot["value"]
        actions.append(action)
    return actions


def equivalent(pred: dict, gold: dict) -> bool:
    if pred.get("operation") != gold.get("operation"):
        return False
    if gold.get("operation") == "NOOP":
        return True
    return (
        pred.get("entity") == gold.get("entity")
        and pred.get("attribute") == gold.get("attribute")
        and norm(pred.get("value", "")) == norm(gold.get("value", ""))
    )


def score_pair(pred: dict, gold: dict) -> tuple[Counter, dict | None]:
    counts = Counter(total_actions=1)
    op = pred.get("operation", "INVALID")
    gold_op = gold.get("operation", "INVALID")
    if op == "INVALID":
        counts["invalid_action"] += 1
    if op == gold_op:
        counts["operation_correct"] += 1
    else:
        counts["wrong_operation"] += 1
    if op == "NOOP" and gold_op != "NOOP":
        counts["unnecessary_noop"] += 1
    if op != "NOOP" and gold_op == "NOOP":
        counts["missed_noop"] += 1
    if equivalent(pred, gold):
        counts["full_action_correct"] += 1
        return counts, None
    if gold_op in {"ADD", "UPDATE"}:
        if pred.get("entity") == gold.get("entity"):
            counts["entity_correct"] += 1
        else:
            counts["wrong_entity"] += 1
        if pred.get("attribute") == gold.get("attribute"):
            counts["attribute_correct"] += 1
        else:
            counts["wrong_attribute"] += 1
        if norm(pred.get("value", "")) == norm(gold.get("value", "")):
            counts["value_correct"] += 1
        else:
            counts["wrong_value"] += 1
    example = {
        "event_idx": gold.get("event_idx"),
        "gold": gold,
        "pred": pred,
        "raw_output": pred.get("raw_output", ""),
    }
    return counts, example


def data_path_for(result_path: Path, data_root: Path, split: str) -> Path:
    _, k = parse_name(result_path)
    return data_root / f"evomemory_update_frequency_hard_k{k}_p63_{split}.json"


def summarize_result(result_path: Path, data_root: Path, split: str) -> dict:
    variant, k = parse_name(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_rows = payload.get("results", [])
    data_path = data_path_for(result_path, data_root, split)
    data = json.loads(data_path.read_text(encoding="utf-8"))

    counts = Counter()
    examples = []
    for fallback_idx, result in enumerate(result_rows):
        example_id = result.get("example_id", fallback_idx)
        episode = data[int(example_id)]
        gold_by_idx = {action["event_idx"]: action for action in gold_actions(episode.get("events", []))}
        pred_by_idx = {action.get("event_idx"): action for action in result.get("slot_actions", [])}
        for event_idx, gold in gold_by_idx.items():
            pred = pred_by_idx.get(event_idx, {"event_idx": event_idx, "operation": "MISSING", "entity": "", "attribute": "", "value": "", "raw_output": ""})
            pair_counts, example = score_pair(pred, gold)
            counts.update(pair_counts)
            if example and len(examples) < 20:
                example["example_id"] = example_id
                example["question"] = episode.get("question", "")
                examples.append(example)

    total = counts["total_actions"]

    def rate(key: str) -> float:
        return counts[key] / total if total else 0.0

    return {
        "name": result_path.parent.name,
        "path": str(result_path),
        "data_path": str(data_path),
        "variant": variant,
        "k": k,
        "total_actions": total,
        "full_action_accuracy": rate("full_action_correct"),
        "operation_accuracy": rate("operation_correct"),
        "invalid_action_rate": rate("invalid_action"),
        "wrong_operation_rate": rate("wrong_operation"),
        "wrong_entity_rate": rate("wrong_entity"),
        "wrong_attribute_rate": rate("wrong_attribute"),
        "wrong_value_rate": rate("wrong_value"),
        "unnecessary_noop_rate": rate("unnecessary_noop"),
        "missed_noop_rate": rate("missed_noop"),
        "counts": dict(counts),
        "examples": examples,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "variant", "k", "total_actions", "full_action_accuracy", "operation_accuracy",
        "invalid_action_rate", "wrong_operation_rate", "wrong_entity_rate", "wrong_attribute_rate",
        "wrong_value_rate", "unnecessary_noop_rate", "missed_noop_rate", "path", "data_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Long25 Action Pathology",
        "",
        "| Name | k | Variant | Full action acc. | Op acc. | Invalid | Wrong op | Wrong entity | Wrong attr. | Wrong value | Unnecessary NOOP | Missed NOOP |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['k']} | {row['variant']} | "
            f"{row['full_action_accuracy']:.3f} | {row['operation_accuracy']:.3f} | "
            f"{row['invalid_action_rate']:.3f} | {row['wrong_operation_rate']:.3f} | "
            f"{row['wrong_entity_rate']:.3f} | {row['wrong_attribute_rate']:.3f} | "
            f"{row['wrong_value_rate']:.3f} | {row['unnecessary_noop_rate']:.3f} | "
            f"{row['missed_noop_rate']:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze learned constrained action pathology")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = [summarize_result(Path(path), Path(args.data_root), args.split) for path in args.inputs]
    rows.sort(key=lambda row: (row["variant"], row["k"], row["name"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "long25_action_pathology_by_k.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(rows, output_dir / "long25_action_pathology_by_k.csv")
    write_md(rows, output_dir / "long25_action_pathology_by_k.md")
    print(f"Wrote {len(rows)} action-pathology rows to {output_dir}")


if __name__ == "__main__":
    main()
