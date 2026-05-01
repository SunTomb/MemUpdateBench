"""
Generate constrained slot-action SFT data from EvoMemory examples.

The output teaches a manager to emit structured actions:
    ADD friend_alex.location = Shanghai
    UPDATE friend_alex.location = Xiamen
    NOOP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.manager.memory_manager import MemoryManager
from scripts.eval_evomemory import EpisodeEntityResolver, parse_event_slot


def build_prompt(slot_state: dict[tuple[str, str], str], event: str) -> str:
    if slot_state:
        state_lines = [
            f"- {entity}.{attribute} = {value}"
            for (entity, attribute), value in sorted(slot_state.items())
        ]
        state_text = "\n".join(state_lines)
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


def action_from_slot(slot_state: dict[tuple[str, str], str], slot: dict | None) -> str:
    if not slot:
        return "NOOP"
    key = (slot["entity"], slot["attribute"])
    op = "UPDATE" if key in slot_state else "ADD"
    return f"{op} {slot['entity']}.{slot['attribute']} = {slot['value']}"


def generate_examples(data: list[dict]) -> list[dict]:
    examples = []
    for episode_idx, episode in enumerate(data):
        slot_state: dict[tuple[str, str], str] = {}
        resolver = EpisodeEntityResolver()
        for event_idx, event in enumerate(episode.get("events", [])):
            slot = parse_event_slot(event, event_idx, resolver=resolver)
            prompt = build_prompt(slot_state, event)
            completion = action_from_slot(slot_state, slot)
            parsed = MemoryManager.parse_constrained_slot_operation(completion)
            if parsed["operation"] == "INVALID":
                raise ValueError(f"Invalid generated action: {completion}")

            examples.append({
                "text": prompt + " " + completion,
                "prompt": prompt,
                "completion": completion,
                "episode_id": episode_idx,
                "event_idx": event_idx,
                "source_question": episode.get("question", ""),
                "source_answer": episode.get("answer", ""),
                "slot": slot,
            })

            if slot:
                slot_state[(slot["entity"], slot["attribute"])] = slot["value"]
    return examples


def write_jsonl(examples: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples = generate_examples(data)
    write_jsonl(examples, output_path)

    counts = {"ADD": 0, "UPDATE": 0, "NOOP": 0, "INVALID": 0}
    for example in examples:
        parsed = MemoryManager.parse_constrained_slot_operation(example["completion"])
        counts[parsed["operation"]] = counts.get(parsed["operation"], 0) + 1

    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "episodes": len(data),
        "examples": len(examples),
        "counts": counts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate constrained slot-action SFT data")
    parser.add_argument("--input", required=True, help="Input EvoMemory JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    main(args)
