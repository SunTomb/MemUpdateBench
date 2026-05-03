"""
EvoMemory Evaluation: Knowledge Evolution Tracking.

Tests the agent's ability to track evolving knowledge through CRUD operations.
Each example contains a sequence of events with contradictory updates
(e.g., "I live in Beijing" → "I'm moving to Chengdu" → "I'm moving to Shanghai"),
and the question tests whether the agent returns the LATEST state.

This is the primary evaluation for MemUpdateBench v2, where CRUD operations
(especially UPDATE) provide a clear advantage over raw ADD.

Evaluation modes:
  1. raw_add: All events → store.add() (no management)
  2. heuristic_crud: Rule-based UPDATE (cosine > threshold)
  3. rl_crud: RL-trained memory manager decide() + execute()
  4. rl_crud_compact: RL-trained + LLM compaction

Usage:
    python scripts/eval_evomemory.py \\
        --mode rl_crud \\
        --lora_checkpoint outputs/phase1/best \\
        --no_qlora
"""

import argparse
import copy
import json
import os
import re
import time

import numpy as np
from loguru import logger

from mub.config import MUBConfig
from mub.memory.store import MemoryStore
from mub.utils import (
    compute_exact_match,
    compute_f1,
    load_model_and_tokenizer,
    set_seed,
)


def run_raw_add(store: MemoryStore, events: list[str]) -> dict:
    """Baseline: ADD all events without any management."""
    stats = new_exec_stats()
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
        stats["requested"]["ADD"] += 1
        stats["effective"]["ADD"] += 1
        stats["success"] += 1
    return stats


def run_heuristic_crud(
    store: MemoryStore,
    events: list[str],
    update_threshold: float = 0.85,
) -> dict:
    """Heuristic baseline: UPDATE if cosine > threshold, else ADD."""
    stats = new_exec_stats()
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        # Check if any existing memory is very similar
        results = store.retrieve(event, topk=1)
        if results:
            entry, score = results[0]
            if score >= update_threshold:
                # UPDATE: replace the old entry
                result = store.update(entry.id, event, env_reward=0.5, slot_meta=slot_meta)
                stats["requested"]["UPDATE"] += 1
                stats["effective"]["UPDATE"] += 1
                stats["success"] += int(result is not None)
                stats["failed"] += int(result is None)
                continue
        # ADD new entry
        store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
        stats["requested"]["ADD"] += 1
        stats["effective"]["ADD"] += 1
        stats["success"] += 1
    return stats


def run_heuristic_slot_crud(store: MemoryStore, events: list[str]) -> dict:
    """Slot-aware heuristic: UPDATE same entity/attribute, else ADD."""
    stats, _ = run_slot_crud(store, events, record_actions=False)
    return stats


def run_constrained_slot_crud(store: MemoryStore, events: list[str]) -> tuple[dict, list[dict]]:
    """Constrained slot baseline with explicit structured action traces."""
    return run_slot_crud(store, events, record_actions=True)


def run_slot_crud(
    store: MemoryStore,
    events: list[str],
    record_actions: bool = False,
) -> tuple[dict, list[dict]]:
    stats = new_exec_stats()
    slot_actions = []
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        action = {
            "event_idx": idx,
            "operation": "NOOP",
            "entity": None,
            "attribute": None,
            "value": None,
        }
        if slot_meta:
            action.update({
                "entity": slot_meta.get("entity"),
                "attribute": slot_meta.get("attribute"),
                "value": slot_meta.get("value"),
            })
            target = store.get_latest_by_slot(
                slot_meta.get("entity", ""), slot_meta.get("attribute", "")
            )
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
        else:
            store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
            action["operation"] = "ADD"
            stats["requested"]["ADD"] += 1
            stats["effective"]["ADD"] += 1
            stats["success"] += 1
        if record_actions:
            slot_actions.append(action)
    return stats, slot_actions


def parse_event_slot(
    event: str,
    event_idx: int,
    resolver: "EpisodeEntityResolver | None" = None,
) -> dict | None:
    text = str(event).strip()
    lower = text.lower()
    resolver = resolver or EpisodeEntityResolver()
    entity = resolver.resolve(text)

    location_patterns = [
        r"(?:live|lives) in ([^.]+)",
        r"(?:is based|relocated|moved|settled) in ([^.]+)",
        r"(?:moving|moved|relocated|settled) to (?!project\b)([^.]+)",
    ]
    value = _first_pattern_value(text, location_patterns)
    if value:
        return _slot(entity, "location", value, event_idx)

    company_patterns = [
        r"(?:work|works) at ([^.]+)",
        r"is employed at ([^.]+)",
        r"(?:switching|switched) to ([^.]+)",
        r"joined ([^.]+)",
    ]
    value = _first_pattern_value(text, company_patterns)
    if value and _looks_like_company_context(lower, value):
        return _slot(entity, "company", value, event_idx)

    language_patterns = [
        r"programming language is ([^.]+)",
        r"(?:prefer|prefers) ([^.]+)",
        r"(?:switched to|now codes in) ([^.]+)",
    ]
    value = _first_pattern_value(text, language_patterns)
    if value and _looks_like_language_context(lower, value):
        return _slot(entity, "language", value, event_idx)

    preference_patterns = [
        r"(?:prefer|prefers) ([^.]+)",
        r"started preferring ([^.]+)",
        r"now prefers ([^.]+)",
    ]
    value = _first_pattern_value(text, preference_patterns)
    if value:
        return _slot(entity, "preference", value, event_idx)

    timezone_patterns = [
        r"timezone is ([^.]+)",
        r"uses ([^.]+) time",
        r"switched timezone to ([^.]+)",
    ]
    value = _first_pattern_value(text, timezone_patterns)
    if value:
        return _slot(entity, "timezone", value, event_idx)

    hobby_patterns = [
        r"hobby is ([^.]+)",
        r"took up ([^.]+)",
        r"now practices ([^.]+)",
    ]
    value = _first_pattern_value(text, hobby_patterns)
    if value:
        return _slot(entity, "hobby", value, event_idx)

    instrument_patterns = [
        r"instrument is ([^.]+)",
        r"plays ([^.]+)",
        r"switched instrument to ([^.]+)",
    ]
    value = _first_pattern_value(text, instrument_patterns)
    if value:
        return _slot(entity, "instrument", value, event_idx)

    project_patterns = [
        r"project is ([^.]+)",
        r"works on project ([^.]+)",
        r"moved to project ([^.]+)",
        r"now works on project ([^.]+)",
    ]
    value = _first_pattern_value(text, project_patterns)
    if value:
        return _slot(entity, "project", value, event_idx)

    relation_patterns = [
        r"(?:is|became) my ([^.]+)",
    ]
    value = _first_pattern_value(text, relation_patterns)
    if value and entity != "user":
        return _slot(entity, "relation", value, event_idx)

    return None


def _slot(entity: str, attribute: str, value: str, event_idx: int) -> dict:
    return {
        "entity": entity,
        "attribute": attribute,
        "value": value.strip().rstrip("."),
        "event_idx": event_idx,
    }


ENTITY_NAME_MAP = {
    "alex": "friend_alex",
    "alice": "coworker_alice",
    "bob": "friend_bob",
    "lily": "sister_lily",
    "chen": "mentor_chen",
}


class EpisodeEntityResolver:
    def __init__(self):
        self.alias_to_entity = dict(ENTITY_NAME_MAP)

    def resolve(self, text: str) -> str:
        lower = text.lower()
        dynamic = re.search(
            r"\bmy (friend|sister|brother|mentor|coworker|manager|advisor|teammate|neighbor|cousin) ([a-z]+)\b",
            lower,
        )
        if dynamic:
            relation, name = dynamic.groups()
            entity = f"{relation}_{name}"
            self.alias_to_entity[name] = entity
            return entity
        if "my mother" in lower or "user's mother" in lower:
            return "mother"
        for alias, entity in self.alias_to_entity.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                return entity
        return "user"


def _parse_event_entity(lower: str) -> str:
    return EpisodeEntityResolver().resolve(lower)


def _first_pattern_value(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _looks_like_company_context(lower: str, value: str) -> bool:
    companies = {
        "google", "microsoft", "alibaba", "bytedance", "huawei",
        "tencent", "baidu", "jd", "netease", "meituan", "pinduoduo",
    }
    return any(marker in lower for marker in ["work", "works", "employed", "switching", "switched", "joined"]) and value.lower() in companies


def _looks_like_language_context(lower: str, value: str) -> bool:
    languages = {"python", "java", "rust", "go", "typescript", "kotlin", "scala"}
    return any(marker in lower for marker in ["code", "codes", "language", "switched", "prefer"]) and value.lower() in languages


def normalize_value(text: str) -> str:
    import re
    normalized = str(text).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def slot_value_match(pred: str, gold: str) -> bool:
    return normalize_value(pred) == normalize_value(gold)


def answer_contains_value(answer: str, value: str) -> bool:
    gold = normalize_value(value)
    pred = normalize_value(answer)
    return bool(gold) and gold in pred


def memory_contains_value(entry, value: str) -> bool:
    slot_value = ""
    if entry.slot:
        slot_value = str(entry.slot.get("value", ""))
    return answer_contains_value(entry.content, value) or slot_value_match(slot_value, value)


def retrieved_trace(example: dict, relevant: list[tuple], answer_topk: int) -> dict:
    gold_value = str(example.get("answer", ""))
    gold_entity = example.get("entity", "user")
    gold_attribute = example.get("attribute", "unknown")
    stale_same_slot_values = []
    trace_entries = []
    gold_in_retrieved = False
    stale_in_retrieved = False
    same_name_distractor_in_retrieved = False
    for rank, (entry, score) in enumerate(relevant, start=1):
        slot = entry.slot or {}
        has_gold = memory_contains_value(entry, gold_value)
        same_slot = slot.get("entity") == gold_entity and slot.get("attribute") == gold_attribute
        slot_value = str(slot.get("value", ""))
        stale_same_slot = bool(same_slot and slot_value and not slot_value_match(slot_value, gold_value))
        if has_gold:
            gold_in_retrieved = True
        if stale_same_slot:
            stale_in_retrieved = True
            stale_same_slot_values.append(slot_value)
        if slot and slot.get("entity") != gold_entity and gold_entity.split("_")[-1] in slot.get("entity", ""):
            same_name_distractor_in_retrieved = True
        trace_entries.append({
            "rank": rank,
            "id": entry.id,
            "score": float(score),
            "content": entry.content,
            "slot": slot,
            "has_gold_value": has_gold,
            "stale_same_slot": stale_same_slot,
        })
    return {
        "answer_topk": answer_topk,
        "retrieved_entries": trace_entries,
        "gold_value_in_retrieved": gold_in_retrieved,
        "stale_same_slot_in_retrieved": stale_in_retrieved,
        "same_name_distractor_in_retrieved": same_name_distractor_in_retrieved,
        "stale_same_slot_values": stale_same_slot_values,
    }


def compute_state_metrics(example: dict, store: MemoryStore) -> dict:
    gold_state = {
        "entity": example.get("entity", "user"),
        "attribute": example.get("attribute", "unknown"),
        "value": example.get("value", example.get("answer", "")),
        "latest_event_idx": example.get("latest_event_idx", -1),
    }
    entry = store.get_latest_by_slot(gold_state["entity"], gold_state["attribute"])
    predicted_state = entry.slot if entry and entry.slot else None
    state_resolved = predicted_state is not None
    state_value_em = bool(
        state_resolved and slot_value_match(predicted_state.get("value", ""), gold_state["value"])
    )
    gold_value = str(gold_state["value"]).strip().lower()
    gold_present = any(gold_value and gold_value in entry.content.lower() for entry in store.entries.values())
    same_slot_entry_count = 0
    stale_same_slot_entry_count = 0
    gold_same_slot_entry_count = 0
    for entry in store.entries.values():
        slot = entry.slot or {}
        if slot.get("entity") == gold_state["entity"] and slot.get("attribute") == gold_state["attribute"]:
            same_slot_entry_count += 1
            value = str(slot.get("value", "")).strip().lower()
            if value and value != gold_value:
                stale_same_slot_entry_count += 1
            elif value and value == gold_value:
                gold_same_slot_entry_count += 1
    stale_present = stale_same_slot_entry_count > 0
    return {
        "gold_state": gold_state,
        "predicted_state": predicted_state,
        "state_resolved": state_resolved,
        "state_value_em": state_value_em,
        "gold_value_present_anywhere": gold_present,
        "stale_value_present_same_slot": stale_present,
        "same_slot_entry_count": same_slot_entry_count,
        "stale_same_slot_entry_count": stale_same_slot_entry_count,
        "gold_same_slot_entry_count": gold_same_slot_entry_count,
    }


def get_op_type(operation: str) -> str:
    op_type = operation.strip().split(maxsplit=1)[0].upper().rstrip(":")
    return op_type if op_type in {"ADD", "UPDATE", "DELETE", "NOOP"} else "NOOP"


def new_exec_stats() -> dict:
    return {
        "requested": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0},
        "effective": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0},
        "success": 0,
        "failed": 0,
        "fallback_target": 0,
    }


def record_execution(stats: dict, requested_op: str, result: dict):
    stats["requested"][requested_op] += 1
    effective_op = result.get("op", requested_op)
    if effective_op not in stats["effective"]:
        effective_op = requested_op
    stats["effective"][effective_op] += 1
    if result.get("success", False):
        stats["success"] += 1
    else:
        stats["failed"] += 1
    if result.get("requested_id") and result.get("entry_id") != result.get("requested_id"):
        stats["fallback_target"] += 1


def run_rl_crud(
    store: MemoryStore,
    events: list[str],
    memory_manager,
    temperature: float = 0.1,
) -> dict:
    """RL-trained CRUD: use memory manager's decide() + execute()."""
    stats = new_exec_stats()
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        op_str, _ = memory_manager.decide(event, temperature=temperature)
        requested_op = get_op_type(op_str)
        result = memory_manager.execute_operation(
            op_str, event, env_reward=0.5, slot_meta=slot_meta
        )
        record_execution(stats, requested_op, result)
    return stats


def build_constrained_prompt(slot_state: dict[tuple[str, str], str], event: str) -> str:
    if slot_state:
        state_text = "\n".join(
            f"- {entity}.{attribute} = {value}"
            for (entity, attribute), value in sorted(slot_state.items())
        )
    else:
        state_text = "(empty slot state)"
    return (
        "You are a constrained Memory Manager. Given the current slot state and a new event, "
        "output exactly one structured memory action.\n\n"
        "Allowed output formats:\n"
        "- ADD <entity>.<attribute> = <value>\n"
        "- UPDATE <entity>.<attribute> = <value>\n"
        "- NOOP\n\n"
        "Entity key normalization examples:\n"
        "- my friend Bob -> friend_bob\n"
        "- my sister Lily -> sister_lily\n"
        "- my mentor Chen -> mentor_chen\n"
        "- my coworker Alice -> coworker_alice\n"
        "- my manager Tom -> manager_tom\n\n"
        "Current slot state:\n"
        f"{state_text}\n\n"
        "New event:\n"
        f"{event}\n\n"
        "Action:"
    )


def run_learned_constrained_slot(store: MemoryStore, events: list[str], model, tokenizer) -> tuple[dict, list[dict]]:
    from mub.manager.memory_manager import MemoryManager
    from mub.utils import generate_text

    stats = new_exec_stats()
    slot_actions = []
    slot_state: dict[tuple[str, str], str] = {}
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        parse_event_slot(event, idx, resolver=resolver)
        prompt = build_constrained_prompt(slot_state, event)
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=32, temperature=0.1, do_sample=False,
        ).strip().split("\n", 1)[0].strip()
        parsed = MemoryManager.parse_constrained_slot_operation(generated)
        op = parsed.get("operation", "INVALID")
        action = {
            "event_idx": idx,
            "operation": op,
            "entity": parsed.get("entity", ""),
            "attribute": parsed.get("attribute", ""),
            "value": parsed.get("value", ""),
            "raw_output": generated,
        }

        if op in {"ADD", "UPDATE"}:
            slot_meta = {
                "entity": parsed["entity"],
                "attribute": parsed["attribute"],
                "value": parsed["value"],
                "event_idx": idx,
            }
            target = store.get_latest_by_slot(parsed["entity"], parsed["attribute"])
            if op == "UPDATE" and target:
                result = store.update(target.id, event, env_reward=0.5, slot_meta=slot_meta)
                action["target_id"] = target.id
                stats["requested"]["UPDATE"] += 1
                stats["effective"]["UPDATE"] += 1
                stats["success"] += int(result is not None)
                stats["failed"] += int(result is None)
            else:
                entry = store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
                action["entry_id"] = entry.id
                stats["requested"]["ADD"] += 1
                stats["effective"]["ADD"] += 1
                stats["success"] += 1
            slot_state[(parsed["entity"], parsed["attribute"])] = parsed["value"]
        elif op == "NOOP":
            stats["requested"]["NOOP"] += 1
            stats["effective"]["NOOP"] += 1
            stats["success"] += 1
        else:
            stats["failed"] += 1
        slot_actions.append(action)
    return stats, slot_actions


def run_rl_crud_seed(
    store: MemoryStore,
    events: list[str],
    memory_manager,
    temperature: float = 0.1,
) -> dict:
    """RL CRUD with forced ADD on empty store (Scheme A).

    Fixes train/eval mismatch: during training, epsilon-greedy exploration
    would ADD initial entries. During eval, we seed the store with the first
    event, then let the RL model decide for subsequent events.
    """
    stats = new_exec_stats()
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        if store.size() == 0:
            # Force ADD when store is empty (replaces missing exploration)
            store.add(content=event, env_reward=0.5, slot_meta=slot_meta)
            stats["requested"]["ADD"] += 1
            stats["effective"]["ADD"] += 1
            stats["success"] += 1
        else:
            op_str, _ = memory_manager.decide(event, temperature=temperature)
            requested_op = get_op_type(op_str)
            result = memory_manager.execute_operation(
                op_str, event, env_reward=0.5, slot_meta=slot_meta
            )
            record_execution(stats, requested_op, result)
    return stats


def run_rl_crud_explore(
    store: MemoryStore,
    events: list[str],
    memory_manager,
    epsilon: float = 0.10,
) -> dict:
    """RL CRUD with light exploration during eval (Scheme B).

    Uses the same epsilon-greedy exploration as training (at low rate)
    to ensure the store gets initial ADD entries, then RL decides.
    """
    stats = new_exec_stats()
    resolver = EpisodeEntityResolver()
    for idx, event in enumerate(events):
        slot_meta = parse_event_slot(event, idx, resolver=resolver)
        op_str, _, _ = memory_manager.decide_with_exploration(
            event, epsilon=epsilon
        )
        requested_op = get_op_type(op_str)
        result = memory_manager.execute_operation(
            op_str, event, env_reward=0.5, slot_meta=slot_meta
        )
        record_execution(stats, requested_op, result)
    return stats


def answer_from_slot(example: dict, store: MemoryStore) -> str:
    entry = store.get_latest_by_slot(
        example.get("entity", "user"), example.get("attribute", "unknown")
    )
    if entry and entry.slot:
        return str(entry.slot.get("value", ""))
    return ""


def build_slot_answer_prompt(
    question: str,
    memory_context: str,
    slot_prompt: dict,
    variant: str,
) -> str:
    entity = slot_prompt.get("entity", "user")
    attribute = slot_prompt.get("attribute", "unknown")
    if variant == "v1_value_only":
        instruction = (
            "Return only the final value for the target slot. "
            "Do not include a sentence, label, punctuation, or explanation."
        )
    elif variant == "v2_ignore_distractors":
        instruction = (
            "Return only the final value for the target slot. "
            "Ignore all memories about other entities, other attributes, older values, and distractors. "
            "Do not include a sentence, label, punctuation, or explanation."
        )
    else:
        instruction = (
            "Answer using only the final value for the specified target slot. "
            "Do not return a sentence or explanation. Ignore facts about other entities or attributes."
        )
    return (
        f"{instruction}\n\n"
        f"Target entity: {entity}\n"
        f"Target attribute: {attribute}\n\n"
        f"Memory entries:\n{memory_context}\n\n"
        f"Question: {question}\n\n"
        "Final value only:"
    )


def filter_latest_per_slot(
    relevant: list,
    store: MemoryStore,
    topk: int,
) -> list:
    latest: dict[tuple[str, str], tuple] = {}
    unscoped = []
    for entry, score in relevant:
        slot = entry.slot or {}
        key = (slot.get("entity"), slot.get("attribute"))
        if not key[0] or not key[1]:
            unscoped.append((entry, score))
            continue
        current = latest.get(key)
        if current is None:
            latest[key] = (entry, score)
            continue
        current_entry = current[0]
        if (
            slot.get("event_idx", -1),
            entry.updated_at,
            entry.created_at,
        ) > (
            (current_entry.slot or {}).get("event_idx", -1),
            current_entry.updated_at,
            current_entry.created_at,
        ):
            latest[key] = (entry, score)
    filtered = list(latest.values()) + unscoped
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[:topk]


def answer_question(
    model,
    tokenizer,
    question: str,
    store: MemoryStore,
    short_answer: bool = False,
    slot_prompt: dict | None = None,
    slot_prompt_variant: str = "v0_current",
    answer_topk: int = 5,
    retrieval_policy: str = "normal",
    return_trace: bool = False,
) -> str | tuple[str, dict]:
    """Answer a question using retrieved memories."""
    from mub.utils import generate_text

    retrieval_topk = max(answer_topk, store.size()) if retrieval_policy == "latest_per_slot" else answer_topk
    relevant = store.retrieve(question, topk=retrieval_topk)
    if retrieval_policy == "latest_per_slot":
        relevant = filter_latest_per_slot(relevant, store, answer_topk)
    memory_context = "\n".join([
        f"- {entry.content}" for entry, score in relevant
    ]) if relevant else "(no relevant memories)"

    if slot_prompt:
        prompt = build_slot_answer_prompt(question, memory_context, slot_prompt, slot_prompt_variant)
    elif short_answer:
        prompt = (
            "Answer using only the final value, with no sentence or explanation. "
            "If the information has been updated multiple times, use the LATEST version.\n\n"
            f"Memory entries:\n{memory_context}\n\n"
            f"Question: {question}\n\n"
            "Final value only:"
        )
    else:
        prompt = (
            "Based on the following memory entries, answer the question concisely. "
            "If the information has been updated multiple times, use the LATEST version.\n\n"
            f"Memory entries:\n{memory_context}\n\n"
            f"Question: {question}\n\n"
            "Answer (short, factual):"
        )

    answer = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=64, temperature=0.1,
    )

    # Post-process: extract first line, strip
    answer = answer.strip().split("\n")[0].strip()
    if return_trace:
        return answer, {
            "prompt": prompt,
            "slot_prompt_variant": slot_prompt_variant if slot_prompt else "",
            "retrieval_policy": retrieval_policy,
            **retrieved_trace(slot_prompt or {}, relevant, answer_topk),
        }
    return answer


def main(args):
    set_seed(42)
    logger.info(f"EvoMemory Evaluation | mode={args.mode}")

    # ---- Load model ----
    config = MUBConfig()
    use_qlora = not args.no_qlora
    model = None
    tokenizer = None
    needs_model = args.mode in {"learned_constrained_slot"}
    needs_model = needs_model or args.answer_mode in {"short_prompt", "slot_prompt", "rag"}
    if needs_model:
        model, tokenizer = load_model_and_tokenizer(
            config.model.model_name,
            use_qlora=use_qlora,
            load_in_4bit=args.load_in_4bit,
        )

        lora_path = args.lora_checkpoint
        if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info(f"Loaded LoRA adapter from: {lora_path}")

    memory_manager = None

    # ---- Load evaluation data ----
    if args.data_file:
        data_path = args.data_file
    else:
        data_filename = f"evomemory_{args.split}.json"
        data_path = os.path.join(args.data_dir, data_filename)
        if not os.path.exists(data_path):
            legacy_path = os.path.join(args.data_dir, "evomemory_test.json")
            if args.split == "test" and os.path.exists(legacy_path):
                logger.warning(
                    "EvoMemory test split not found; using legacy evomemory_test.json."
                )
                data_path = legacy_path
            else:
                raise FileNotFoundError(f"EvoMemory split not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    total_examples = len(eval_data)
    start_idx = max(args.start_idx, 0)
    end_idx = args.end_idx if args.end_idx is not None else total_examples
    end_idx = min(end_idx, total_examples)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid shard range: start_idx={start_idx}, end_idx={end_idx}, total={total_examples}")
    eval_data = eval_data[start_idx:end_idx]
    logger.info(
        f"Loaded {len(eval_data)} EvoMemory examples "
        f"from range [{start_idx}, {end_idx}) of {total_examples}"
    )

    # ---- Evaluation loop ----
    results = []
    f1_scores = []
    em_scores = []
    start_time = time.time()

    for i, example in enumerate(eval_data):
        original_example_id = start_idx + i
        events = example["events"]
        question = example["question"]
        gold_answer = example["answer"]
        num_updates = int(example.get("num_updates", len(events)))

        # Fresh memory store for each example
        store = MemoryStore(config.memory)

        slot_actions = []

        # Process events according to mode
        if args.mode == "raw_add":
            exec_stats = run_raw_add(store, events)
        elif args.mode == "heuristic_crud":
            exec_stats = run_heuristic_crud(store, events, args.update_threshold)
        elif args.mode == "heuristic_slot_crud":
            exec_stats = run_heuristic_slot_crud(store, events)
        elif args.mode == "constrained_slot_crud":
            exec_stats, slot_actions = run_constrained_slot_crud(store, events)
        elif args.mode == "learned_constrained_slot":
            exec_stats, slot_actions = run_learned_constrained_slot(store, events, model, tokenizer)
        elif args.mode in ("rl_crud", "rl_crud_compact"):
            memory_manager.store = store
            exec_stats = run_rl_crud(store, events, memory_manager, temperature=args.decision_temperature)

            # Optional: run compaction after processing
            if args.mode == "rl_crud_compact":
                from mub.consolidation.compaction import MemoryCompactor
                compactor = MemoryCompactor(config.compaction)
                compactor.config.trigger_memory_threshold = 2  # Low threshold for small examples
                compactor.run(store, model, tokenizer)
        elif args.mode == "rl_crud_seed":
            memory_manager.store = store
            exec_stats = run_rl_crud_seed(
                store, events, memory_manager, temperature=args.decision_temperature
            )
        elif args.mode == "rl_crud_explore":
            memory_manager.store = store
            exec_stats = run_rl_crud_explore(store, events, memory_manager,
                                             epsilon=args.eval_epsilon)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Answer the question
        answer_trace = None
        if args.answer_mode == "slot_direct":
            predicted = answer_from_slot(example, store)
        else:
            answer_output = answer_question(
                model, tokenizer, question, store,
                short_answer=args.answer_mode == "short_prompt",
                slot_prompt={
                    "entity": example.get("entity", "user"),
                    "attribute": example.get("attribute", "unknown"),
                    "answer": gold_answer,
                } if args.answer_mode == "slot_prompt" else None,
                slot_prompt_variant=args.slot_prompt_variant,
                answer_topk=args.answer_topk,
                retrieval_policy=args.retrieval_policy,
                return_trace=args.save_answer_traces,
            )
            if args.save_answer_traces:
                predicted, answer_trace = answer_output
            else:
                predicted = answer_output

        # Compute metrics
        f1 = compute_f1(predicted, gold_answer)
        em = compute_exact_match(predicted, gold_answer)
        answer_value_present = answer_contains_value(predicted, gold_answer)
        state_metrics = compute_state_metrics(example, store)
        f1_scores.append(f1)
        em_scores.append(em)

        target_slot_write_count = sum(
            1 for action in slot_actions
            if action.get("entity") == example.get("entity", "user")
            and action.get("attribute") == example.get("attribute", "unknown")
            and action.get("operation") in {"ADD", "UPDATE"}
        )
        write_amplification = store.size() / max(len(events), 1)
        result = {
            "example_id": original_example_id,
            "shard_local_example_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted,
            "f1": f1,
            "em": em,
            "value_em": slot_value_match(predicted, gold_answer),
            "answer_value_present": answer_value_present,
            "stress_type": example.get("stress_type", ""),
            "k_updates": example.get("k_updates"),
            "distractor_level": example.get("distractor_level", ""),
            "noop_level": example.get("noop_level", ""),
            "num_events": len(events),
            "num_target_updates": example.get("num_target_updates", example.get("k_updates")),
            "num_updates": num_updates,
            "memory_size": store.size(),
            "write_amplification": write_amplification,
            "target_slot_write_count": target_slot_write_count,
            "op_counts": exec_stats["requested"],
            "exec_stats": exec_stats,
            "slot_actions": slot_actions,
            **state_metrics,
        }
        if answer_trace is not None:
            answer_trace.update({
                "predicted_answer": predicted,
                "gold_answer": gold_answer,
                "em": em,
                "f1": f1,
                "value_em": slot_value_match(predicted, gold_answer),
                "answer_value_present": answer_value_present,
                "state_value_em": state_metrics.get("state_value_em", False),
            })
            result["answer_trace"] = answer_trace
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(eval_data):
            avg_f1 = np.mean(f1_scores)
            avg_em = np.mean(em_scores)
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {i+1}/{len(eval_data)} | "
                f"F1={avg_f1:.4f} | EM={avg_em:.4f} | "
                f"mem={store.size()} | "
                f"Time: {elapsed:.1f}s"
            )

    # ---- Summary ----
    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)
    elapsed = time.time() - start_time

    total_exec_stats = new_exec_stats()
    final_memory_sizes = []
    same_slot_entry_counts = []
    stale_same_slot_entry_counts = []
    gold_same_slot_entry_counts = []
    write_amplifications = []
    target_slot_write_counts = []
    state_resolved = []
    state_value_em = []
    gold_present = []
    stale_present = []
    answer_value_present = []
    value_em = []
    for result in results:
        stats = result["exec_stats"]
        final_memory_sizes.append(result["memory_size"])
        same_slot_entry_counts.append(result.get("same_slot_entry_count", 0))
        stale_same_slot_entry_counts.append(result.get("stale_same_slot_entry_count", 0))
        gold_same_slot_entry_counts.append(result.get("gold_same_slot_entry_count", 0))
        write_amplifications.append(result.get("write_amplification", 0.0))
        target_slot_write_counts.append(result.get("target_slot_write_count", 0))
        for bucket in ("requested", "effective"):
            for op, count in stats[bucket].items():
                total_exec_stats[bucket][op] = total_exec_stats[bucket].get(op, 0) + count
        total_exec_stats["success"] += stats["success"]
        total_exec_stats["failed"] += stats["failed"]
        total_exec_stats["fallback_target"] += stats["fallback_target"]
        state_resolved.append(float(result["state_resolved"]))
        state_value_em.append(float(result["state_value_em"]))
        gold_present.append(float(result["gold_value_present_anywhere"]))
        stale_present.append(float(result["stale_value_present_same_slot"]))
        answer_value_present.append(float(result["answer_value_present"]))
        value_em.append(float(result["value_em"]))
    total_op_counts = total_exec_stats["requested"]

    summary = {
        "benchmark": "evomemory",
        "mode": args.mode,
        "data_file": data_path,
        "num_examples": len(eval_data),
        "total_examples": total_examples,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "value_em": float(np.mean(value_em)) if value_em else 0.0,
        "answer_value_present_rate": float(np.mean(answer_value_present)) if answer_value_present else 0.0,
        "op_counts": total_op_counts,
        "exec_stats": total_exec_stats,
        "final_memory_size_avg": float(np.mean(final_memory_sizes)) if final_memory_sizes else 0.0,
        "same_slot_entry_count_avg": float(np.mean(same_slot_entry_counts)) if same_slot_entry_counts else 0.0,
        "stale_same_slot_entry_count_avg": float(np.mean(stale_same_slot_entry_counts)) if stale_same_slot_entry_counts else 0.0,
        "gold_same_slot_entry_count_avg": float(np.mean(gold_same_slot_entry_counts)) if gold_same_slot_entry_counts else 0.0,
        "write_amplification_avg": float(np.mean(write_amplifications)) if write_amplifications else 0.0,
        "target_slot_write_count_avg": float(np.mean(target_slot_write_counts)) if target_slot_write_counts else 0.0,
        "state_resolve_rate": float(np.mean(state_resolved)) if state_resolved else 0.0,
        "state_accuracy": float(np.mean(state_value_em)) if state_value_em else 0.0,
        "gold_present_rate": float(np.mean(gold_present)) if gold_present else 0.0,
        "stale_present_rate": float(np.mean(stale_present)) if stale_present else 0.0,
        "decision_temperature": args.decision_temperature,
        "answer_mode": args.answer_mode,
        "slot_prompt_variant": args.slot_prompt_variant,
        "answer_topk": args.answer_topk,
        "retrieval_policy": args.retrieval_policy,
        "save_answer_traces": args.save_answer_traces,
        "elapsed_seconds": elapsed,
        "lora_checkpoint": args.lora_checkpoint,
    }

    logger.info("=" * 60)
    logger.info(f"EvoMemory Results ({args.mode})")
    logger.info(f"  F1:  {avg_f1:.4f}")
    logger.info(f"  EM:  {avg_em:.4f}")
    logger.info(f"  Value EM: {summary['value_em']:.4f}")
    logger.info(f"  Answer value present: {summary['answer_value_present_rate']:.4f}")
    logger.info(f"  Operations: {total_op_counts}")
    logger.info(f"  Execution stats: {total_exec_stats}")
    logger.info(
        f"  State: resolve={summary['state_resolve_rate']:.4f} "
        f"acc={summary['state_accuracy']:.4f} "
        f"gold_present={summary['gold_present_rate']:.4f} "
        f"stale_present={summary['stale_present_rate']:.4f}"
    )
    logger.info(
        f"  Slot burden: same_slot={summary['same_slot_entry_count_avg']:.2f} "
        f"stale_same_slot={summary['stale_same_slot_entry_count_avg']:.2f} "
        f"gold_same_slot={summary['gold_same_slot_entry_count_avg']:.2f} "
        f"write_amp={summary['write_amplification_avg']:.3f} "
        f"target_writes={summary['target_slot_write_count_avg']:.2f}"
    )
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evomemory_results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_dir}/evomemory_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MemUpdateBench v2: EvoMemory Knowledge Evolution Evaluation",
    )
    parser.add_argument("--mode", default="raw_add",
                        choices=["raw_add", "heuristic_crud", "heuristic_slot_crud",
                                 "constrained_slot_crud", "learned_constrained_slot"],
                        help="Evaluation mode")
    parser.add_argument("--lora_checkpoint", default="checkpoints/long25/best",
                        help="LoRA adapter directory")
    parser.add_argument("--output_dir", default="results/evomemory")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--data_file", default="",
                        help="Explicit EvoMemory JSON file; overrides --data_dir/--split")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Inclusive example start index for sharded evaluation")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Exclusive example end index for sharded evaluation")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="EvoMemory split to evaluate")
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--update_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for heuristic UPDATE")
    parser.add_argument("--eval_epsilon", type=float, default=0.10,
                        help="Exploration rate for rl_crud_explore mode")
    parser.add_argument("--decision_temperature", type=float, default=0.1,
                        help="Temperature for learned CRUD decisions during eval")
    parser.add_argument("--answer_mode", default="rag", choices=["rag", "short_prompt", "slot_prompt", "slot_direct"],
                        help="Answering mode: normal RAG, short prompt, slot-conditioned prompt, or direct slot value")
    parser.add_argument("--slot_prompt_variant", default="v0_current",
                        choices=["v0_current", "v1_value_only", "v2_ignore_distractors"],
                        help="Prompt template variant for slot_prompt answer mode")
    parser.add_argument("--answer_topk", type=int, default=5,
                        help="Number of retrieved memories used for prompted answering")
    parser.add_argument("--retrieval_policy", default="normal", choices=["normal", "latest_per_slot"],
                        help="Answer-time retrieval policy for prompted answering")
    parser.add_argument("--save_answer_traces", action="store_true",
                        help="Save retrieved-memory and answer-layer traces per example")
    args = parser.parse_args()
    main(args)
