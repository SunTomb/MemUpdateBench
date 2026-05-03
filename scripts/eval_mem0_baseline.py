from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.utils import compute_exact_match, compute_f1
from scripts.eval_evomemory import answer_contains_value, normalize_value, parse_event_slot, slot_value_match


def import_mem0():
    try:
        module = importlib.import_module("mem0")
    except ImportError:
        try:
            module = importlib.import_module("mem0ai")
        except ImportError as exc:
            raise RuntimeError(
                "Mem0 is not installed in this environment. Install it in an isolated "
                "environment before running a real external-baseline evaluation."
            ) from exc
    memory_cls = getattr(module, "Memory", None)
    if memory_cls is None:
        try:
            memory_cls = importlib.import_module("mem0.memory.main").Memory
        except Exception as exc:
            raise RuntimeError("Mem0 package was found, but Memory class could not be located.") from exc
    return memory_cls


def make_mem0(memory_cls, config_path: str = "", run_id: str = "", example_idx: int = 0):
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        vector_config = config.get("vector_store", {}).get("config", {})
        if run_id and "path" in vector_config:
            vector_config["path"] = str(Path(vector_config["path"]) / f"{run_id}_{example_idx}")
        if hasattr(memory_cls, "from_config"):
            return memory_cls.from_config(config)
        return memory_cls(config=config)
    if hasattr(memory_cls, "from_config"):
        try:
            return memory_cls.from_config({})
        except Exception:
            pass
    return memory_cls()


def call_mem0_add(memory: Any, text: str, user_id: str) -> None:
    try:
        memory.add(text, user_id=user_id)
        return
    except TypeError:
        pass
    try:
        memory.add([{"role": "user", "content": text}], user_id=user_id)
        return
    except TypeError:
        pass
    memory.add(text)


def call_mem0_search(memory: Any, query: str, user_id: str) -> Any:
    for kwargs in (
        {"user_id": user_id},
        {"filters": {"user_id": user_id}},
        {"filters": {"AND": [{"user_id": user_id}]}},
        {},
    ):
        try:
            return memory.search(query, **kwargs)
        except (TypeError, ValueError):
            continue
    return []


def call_mem0_get_all(memory: Any, user_id: str) -> Any:
    if not hasattr(memory, "get_all"):
        return []
    for kwargs in (
        {"user_id": user_id},
        {"filters": {"user_id": user_id}},
        {"filters": {"AND": [{"user_id": user_id}]}},
        {},
    ):
        try:
            return memory.get_all(**kwargs)
        except (TypeError, ValueError):
            continue
    return []


def extract_texts(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        if "results" in payload:
            return extract_texts(payload["results"])
        for key in ("memory", "text", "content", "value"):
            if key in payload and isinstance(payload[key], str):
                return [payload[key]]
        texts = []
        for value in payload.values():
            texts.extend(extract_texts(value))
        return texts
    if isinstance(payload, list):
        texts = []
        for item in payload:
            texts.extend(extract_texts(item))
        return texts
    return []


def choose_prediction(search_payload: Any, example: dict) -> str:
    texts = [text.strip() for text in extract_texts(search_payload) if str(text).strip()]
    if not texts:
        return ""

    attribute = str(example.get("attribute", "")).lower()
    entity_tail = str(example.get("entity", "")).split("_")[-1].lower()
    entity_relation = "_".join(str(example.get("entity", "")).split("_")[:-1]).lower()

    def score(text: str) -> tuple[int, int, int, int]:
        lower = text.lower()
        slot = parse_event_slot(text, 0)
        same_entity = int(entity_tail in lower and (not entity_relation or entity_relation in lower))
        same_attribute = int(bool(slot and slot.get("attribute") == attribute))
        same_slot = int(bool(slot and slot.get("attribute") == attribute and str(slot.get("entity", "")).endswith(entity_tail)))
        freshness = 0
        if attribute == "company":
            freshness = int("works at" in lower) * 3 + int("joined" in lower) * 2 + int("switched to" in lower)
        elif attribute == "location":
            freshness = int("lives in" in lower) * 3 + int("relocated to" in lower) * 2 + int("moved to" in lower)
        elif attribute == "preference":
            freshness = int("prefers" in lower) * 3 + int("started preferring" in lower) * 2
        elif attribute == "language":
            freshness = int("programming language is" in lower) * 3 + int("now codes in" in lower) * 2 + int("switched to" in lower)
        return (same_slot, same_attribute, same_entity, freshness)

    best = max(texts, key=score)
    slot = parse_event_slot(best, 0)
    if slot and slot.get("attribute") == attribute:
        return str(slot.get("value", "")).strip()
    if attribute == "company":
        if "works at" in best:
            return best.split("works at", 1)[1].strip().rstrip(".")
        if "joined" in best:
            return best.split("joined", 1)[1].strip().rstrip(".")
    if attribute == "location":
        if "lives in" in best:
            return best.split("lives in", 1)[1].strip().rstrip(".")
        if "relocated to" in best:
            return best.split("relocated to", 1)[1].strip().rstrip(".")
        if "moved to" in best:
            return best.split("moved to", 1)[1].strip().rstrip(".")
    return normalize_value(best)


def inspect_memories(memory: Any, user_id: str) -> tuple[list[str], bool]:
    payload = call_mem0_get_all(memory, user_id)
    texts = extract_texts(payload)
    return texts, bool(texts)


def text_has_same_slot_value(text: str, example: dict, value: str) -> bool:
    lower = text.lower()
    entity_tail = str(example.get("entity", "")).split("_")[-1].lower()
    attribute = str(example.get("attribute", "")).lower()
    return entity_tail in lower and attribute in lower and answer_contains_value(text, value)


def memory_metrics(example: dict, memory_texts: list[str]) -> dict:
    gold = str(example.get("answer", example.get("value", "")))
    same_slot_entry_count = 0
    stale_same_slot_entry_count = 0
    gold_same_slot_entry_count = 0
    stale_values = []
    for text in memory_texts:
        slot = parse_event_slot(text, 0)
        if slot and slot.get("entity") == example.get("entity") and slot.get("attribute") == example.get("attribute"):
            same_slot_entry_count += 1
            value = str(slot.get("value", ""))
            if slot_value_match(value, gold):
                gold_same_slot_entry_count += 1
            else:
                stale_same_slot_entry_count += 1
                stale_values.append(value)
        elif text_has_same_slot_value(text, example, gold):
            same_slot_entry_count += 1
            gold_same_slot_entry_count += 1
    return {
        "memory_size": len(memory_texts),
        "same_slot_entry_count": same_slot_entry_count,
        "stale_same_slot_entry_count": stale_same_slot_entry_count,
        "gold_same_slot_entry_count": gold_same_slot_entry_count,
        "stale_same_slot_values": stale_values,
    }


def evaluate(args: argparse.Namespace) -> dict:
    memory_cls = import_mem0()
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))[: args.limit or None]
    results = []
    inspectable_count = 0
    shared_memory = make_mem0(memory_cls, args.config, args.run_id, 0) if args.reuse_memory_instance else None

    for idx, example in enumerate(data):
        memory = shared_memory or make_mem0(memory_cls, args.config, args.run_id, idx)
        user_id = f"mub_{args.run_id}_{idx}"
        for event in example.get("events", []):
            call_mem0_add(memory, event, user_id)
        search_payload = call_mem0_search(memory, example.get("question", ""), user_id)
        predicted = choose_prediction(search_payload, example)
        memory_texts, inspectable = inspect_memories(memory, user_id)
        inspectable_count += int(inspectable)
        mem_metrics = memory_metrics(example, memory_texts) if inspectable else {
            "memory_size": 0,
            "same_slot_entry_count": 0,
            "stale_same_slot_entry_count": 0,
            "gold_same_slot_entry_count": 0,
            "stale_same_slot_values": [],
        }
        gold_answer = str(example.get("answer", ""))
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)
        result = {
            "example_id": idx,
            "question": example.get("question", ""),
            "gold_answer": gold_answer,
            "predicted": predicted,
            "em": em,
            "f1": f1,
            "value_em": slot_value_match(predicted, gold_answer),
            "answer_value_present": answer_contains_value(predicted, gold_answer),
            "k_updates": example.get("k_updates"),
            "stress_type": example.get("stress_type"),
            "memory_inspectable": inspectable,
            "search_payload": search_payload,
            "memory_texts": memory_texts,
            **mem_metrics,
        }
        results.append(result)
        if (idx + 1) % max(args.log_every, 1) == 0:
            print(f"Progress: {idx + 1}/{len(data)} EM={mean([r['em'] for r in results]):.3f}")

    def avg(key: str) -> float:
        return mean([float(row.get(key, 0.0)) for row in results]) if results else 0.0

    return {
        "summary": {
            "method": "mem0_external",
            "data_file": args.data_file,
            "num_examples": len(results),
            "avg_em": avg("em"),
            "avg_f1": avg("f1"),
            "value_em": avg("value_em"),
            "answer_value_present": avg("answer_value_present"),
            "memory_inspectable_rate": inspectable_count / len(results) if results else 0.0,
            "stale_same_slot_entry_count_avg": avg("stale_same_slot_entry_count"),
            "same_slot_entry_count_avg": avg("same_slot_entry_count"),
            "gold_same_slot_entry_count_avg": avg("gold_same_slot_entry_count"),
            "final_memory_size_avg": avg("memory_size"),
        },
        "results": results,
    }


def write_unavailable_report(args: argparse.Namespace, error: Exception) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "method": "mem0_external",
            "status": "unavailable",
            "reason": str(error),
            "data_file": args.data_file,
            "num_examples": 0,
        },
        "results": [],
    }
    (output_dir / "evomemory_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "feasibility_report.md").write_text(
        "# Mem0 Feasibility Report\n\n"
        "Status: unavailable in the current environment.\n\n"
        f"Reason: {error}\n\n"
        "Install Mem0 in an isolated environment before interpreting this as an external baseline result.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real Mem0 as an external MemUpdateBench baseline")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--config", default="", help="Optional JSON config passed to Mem0")
    parser.add_argument("--run_id", default="feasibility")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--write_unavailable_report", action="store_true")
    parser.add_argument("--reuse_memory_instance", action="store_true",
                        help="Reuse one Mem0 instance across examples, isolating examples by user_id")
    args = parser.parse_args()

    try:
        payload = evaluate(args)
    except Exception as exc:
        if args.write_unavailable_report:
            write_unavailable_report(args, exc)
            print(f"Mem0 unavailable: {exc}")
            return
        raise

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evomemory_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
