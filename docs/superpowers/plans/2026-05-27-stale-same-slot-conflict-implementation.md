# Stale Same-Slot Conflict Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal novelty package from `docs/superpowers/specs/2026-05-27-stale-same-slot-conflict-design.md`: conflict-type decomposition, version-conflict dose-response refinement, and stale-specific causal removal, with summaries ready for paper integration.

**Architecture:** Add small probe and analysis scripts that reuse existing answer prompting and metric utilities instead of changing the core memory store. Keep context construction deterministic and inspectable, write raw CSV/JSON plus Markdown summaries, then integrate the resulting artifacts into the existing P8 evidence and paper-note workflow.

**Tech Stack:** Python 3, existing `mub.utils` metrics/model helpers, existing `scripts.eval_evomemory` prompt helpers, CSV/JSON/Markdown artifacts, cluster shell runners for A40/A100 execution.

---

## File structure

- Create `scripts/run_conflict_type_probe.py`: constructs matched-length controlled contexts for final-only, unrelated distractors, same-entity/different-attribute, different-entity/same-attribute, and stale same-slot conditions; runs an answer model; writes examples and summary artifacts.
- Create `scripts/summarize_conflict_type_probe.py`: reads conflict probe examples and writes aggregate tables including degradation from `final_only` and stale-specific copied-value rates.
- Modify `scripts/run_synthetic_same_slot_probe.py`: add `middle` and `random` context orders so the existing synthetic probe can support the refined version-conflict dose-response matrix.
- Modify `scripts/smoke_test.py`: add deterministic smoke checks for conflict context construction and the new synthetic context orders.
- Create `scripts/analyze_stale_specific_removal.py`: reads existing `evomemory_results.json` files with answer traces, simulates removal interventions over retrieved entries, and summarizes trace-level recoverability proxies before full model reruns.
- Create `scripts/run_p83_conflict_type_probe_sui3.sh`: cluster runner for the full Qwen conflict-type probe.
- Create `scripts/run_p83_stale_conflict_dose_sui3.sh`: cluster runner for the refined dose-response conditions.
- Create `paper/p83_stale_same_slot_conflict_plan_note.md`: paper-facing note explaining the planned new P8.3 mechanism package and expected claim changes.
- Modify `WORKFLOW.md`: after actual experiment execution, append commands, metrics, conclusions, and result paths.

## Task 1: Add conflict-type context construction and runner

**Files:**
- Create: `scripts/run_conflict_type_probe.py`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Create `scripts/run_conflict_type_probe.py` with deterministic context builders**

Write this file:

```python
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
```

- [ ] **Step 2: Add smoke coverage for the conflict builder**

Modify `scripts/smoke_test.py` inside `test_constrained_slots` after the existing `policy_trace` assertions and before `results.ok(...)`:

```python
        from scripts.run_conflict_type_probe import make_example as make_conflict_example
        conflict_example = make_conflict_example(
            example_id=0,
            attribute="location",
            condition="stale_same_slot",
            distractor_count=2,
        )
        assert "Target entity: friend_alice" in conflict_example["prompt"]
        assert "Target attribute: location" in conflict_example["prompt"]
        assert conflict_example["prompt"].count("User says:") == 3
```

- [ ] **Step 3: Run targeted smoke test**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected output includes:

```text
SMOKE TEST: 26/26 passed
```

The pass count may increase if the worker also adds a separate smoke test case; failure is acceptable only before Step 4 if it points to the new conflict builder import or assertion.

- [ ] **Step 4: Fix only import/assertion issues if the smoke test fails**

If the failure is an import path issue, ensure `scripts/run_conflict_type_probe.py` contains:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

If the prompt count assertion fails because the prompt text changes, inspect the constructed prompt and update the assertion to check the same invariant:

```python
assert "friend_alice" in conflict_example["prompt"]
assert "location" in conflict_example["prompt"]
assert "Final value only:" in conflict_example["prompt"]
```

- [ ] **Step 5: Run a tiny live probe only if a local model path is configured**

Run this only on a machine where `Qwen/Qwen2.5-7B-Instruct` is available:

```bash
PYTHONPATH=. python scripts/run_conflict_type_probe.py --examples_per_condition 2 --conditions final_only,stale_same_slot --distractor_count 2 --output_dir results/p83_conflict_type_probe_smoke --no_qlora
```

Expected output:

```json
{
  "num_examples": 4,
  "num_conditions": 2,
  "output_dir": "results/p83_conflict_type_probe_smoke"
}
```

If the model is not available locally, skip this live probe and record that the first live run should be on the cluster.

## Task 2: Add standalone conflict-type summarizer

**Files:**
- Create: `scripts/summarize_conflict_type_probe.py`

- [ ] **Step 1: Create the summarizer**

Write this file:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if isinstance(value, bool):
        return float(value)
    if str(value).lower() in {"true", "false"}:
        return 1.0 if str(value).lower() == "true" else 0.0
    return float(value or 0.0)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    final_only_em = 0.0
    if "final_only" in grouped:
        final_only_em = mean(as_float(row, "em") for row in grouped["final_only"])
    output = []
    for condition, items in sorted(grouped.items()):
        em = mean(as_float(row, "em") for row in items)
        output.append({
            "condition": condition,
            "n": len(items),
            "distractor_count": items[0].get("distractor_count", ""),
            "em": em,
            "f1": mean(as_float(row, "f1") for row in items),
            "value_em": mean(as_float(row, "value_em") for row in items),
            "answer_value_present": mean(as_float(row, "answer_value_present") for row in items),
            "stale_value_copied": mean(as_float(row, "stale_value_copied") for row in items),
            "em_drop_from_final_only": final_only_em - em,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Conflict-Type Probe Summary",
        "",
        "| condition | n | distractors | EM | F1 | value EM | answer value present | stale copied | EM drop vs final-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['distractor_count']} | "
            f"{row['em']:.3f} | {row['f1']:.3f} | {row['value_em']:.3f} | "
            f"{row['answer_value_present']:.3f} | {row['stale_value_copied']:.3f} | "
            f"{row['em_drop_from_final_only']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize conflict-type probe examples")
    parser.add_argument("--input_csv", default="results/p83_conflict_type_probe/conflict_type_examples.csv")
    parser.add_argument("--output_dir", default="results/p83_conflict_type_probe_summary")
    args = parser.parse_args()

    rows = read_rows(Path(args.input_csv))
    summary = summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "conflict_type_summary.csv", summary)
    (output_dir / "conflict_type_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "conflict_type_summary.md", summary)
    print(json.dumps({"num_rows": len(rows), "num_conditions": len(summary), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run py_compile for the two conflict scripts**

Run:

```bash
PYTHONPATH=. python -m py_compile scripts/run_conflict_type_probe.py scripts/summarize_conflict_type_probe.py
```

Expected: command exits with status 0 and prints no traceback.

- [ ] **Step 3: Run summarizer on a tiny fixture**

Create a temporary fixture with Python and run the summarizer:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
p = Path('results/tmp_conflict_summary_fixture')
p.mkdir(parents=True, exist_ok=True)
(p / 'conflict_type_examples.csv').write_text('condition,example_id,attribute,entity,gold_answer,distractor_count,predicted,em,f1,value_em,answer_value_present,stale_value_copied,prompt\nfinal_only,0,location,friend_alice,Dalian,2,Dalian,1.0,1.0,1,1,0,x\nstale_same_slot,0,location,friend_alice,Dalian,2,Qingdao,0.0,0.0,0,0,1,x\n', encoding='utf-8')
PY
PYTHONPATH=. python scripts/summarize_conflict_type_probe.py --input_csv results/tmp_conflict_summary_fixture/conflict_type_examples.csv --output_dir results/tmp_conflict_summary_fixture/summary
```

Expected output:

```json
{
  "num_rows": 2,
  "num_conditions": 2,
  "output_dir": "results/tmp_conflict_summary_fixture/summary"
}
```

Do not commit the temporary `results/tmp_conflict_summary_fixture` directory.

## Task 3: Extend synthetic same-slot probe for refined dose-response

**Files:**
- Modify: `scripts/run_synthetic_same_slot_probe.py:88-123`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Add `middle` and `random` order support**

In `scripts/run_synthetic_same_slot_probe.py`, replace the context ordering block inside `build_context` with:

```python
    if context_order == "current_first":
        entries = sorted(entries, key=lambda item: (not item["latest"], item["idx"]))
    elif context_order == "current_last":
        entries = sorted(entries, key=lambda item: (item["latest"], item["idx"]))
    elif context_order == "reverse_chronological":
        entries = sorted(entries, key=lambda item: item["idx"], reverse=True)
    elif context_order == "chronological":
        entries = sorted(entries, key=lambda item: item["idx"])
    elif context_order == "middle":
        stale_entries = [entry for entry in entries if not entry["latest"]]
        latest_entries = [entry for entry in entries if entry["latest"]]
        midpoint = len(stale_entries) // 2
        entries = stale_entries[:midpoint] + latest_entries + stale_entries[midpoint:]
    elif context_order == "random":
        rng = random.Random(len(stale_values) * 1009 + len(gold) * 17 + len(attribute))
        rng.shuffle(entries)
    elif context_order != "normal":
        raise ValueError(f"unknown context_order: {context_order}")
```

The file already imports `random`, so no new import is needed.

- [ ] **Step 2: Add smoke assertions for new orders**

Modify `scripts/smoke_test.py` inside `test_constrained_slots`, after the conflict builder assertions from Task 1:

```python
        from scripts.run_synthetic_same_slot_probe import build_context as build_synthetic_context
        middle_context = build_synthetic_context(
            "friend",
            "Alex",
            "location",
            "Chengdu",
            ["Shanghai", "Beijing", "Wuhan", "Nanjing"],
            "middle",
            "latest_outdated_label",
        )
        assert middle_context.count("User says:") == 5
        assert "[latest]" in middle_context
        random_context = build_synthetic_context(
            "friend",
            "Alex",
            "location",
            "Chengdu",
            ["Shanghai", "Beijing"],
            "random",
            "none",
        )
        assert random_context.count("User says:") == 3
```

- [ ] **Step 3: Run smoke test**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected: all smoke tests pass.

- [ ] **Step 4: Run py_compile on affected scripts**

Run:

```bash
PYTHONPATH=. python -m py_compile scripts/run_synthetic_same_slot_probe.py scripts/smoke_test.py
```

Expected: command exits with status 0.

## Task 4: Add stale-specific removal analysis over existing traces

**Files:**
- Create: `scripts/analyze_stale_specific_removal.py`

- [ ] **Step 1: Create trace-level removal analyzer**

Write this file:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

INTERVENTIONS = [
    "normal",
    "remove_random_non_gold",
    "remove_unrelated",
    "remove_near_slot",
    "remove_stale_same_slot",
    "latest_per_slot",
]


def slot_value_match(a: str, b: str) -> bool:
    return str(a).strip().lower().rstrip(".") == str(b).strip().lower().rstrip(".")


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def classify_entry(entry: dict[str, Any], result: dict[str, Any]) -> str:
    slot = entry.get("slot") or {}
    entity = result.get("gold_state", {}).get("entity", result.get("answer_trace", {}).get("entity", ""))
    attribute = result.get("gold_state", {}).get("attribute", result.get("answer_trace", {}).get("attribute", ""))
    gold = str(result.get("gold_answer", ""))
    slot_entity = slot.get("entity")
    slot_attribute = slot.get("attribute")
    slot_value = str(slot.get("value", ""))
    if slot_entity == entity and slot_attribute == attribute and slot_value_match(slot_value, gold):
        return "gold_same_slot"
    if slot_entity == entity and slot_attribute == attribute:
        return "stale_same_slot"
    if slot_entity == entity:
        return "same_entity_different_attribute"
    if slot_attribute == attribute:
        return "different_entity_same_attribute"
    return "unrelated"


def has_gold(entries: list[dict[str, Any]], result: dict[str, Any]) -> bool:
    return any(classify_entry(entry, result) == "gold_same_slot" for entry in entries)


def stale_count(entries: list[dict[str, Any]], result: dict[str, Any]) -> int:
    return sum(1 for entry in entries if classify_entry(entry, result) == "stale_same_slot")


def latest_per_slot(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    unscoped = []
    for entry in entries:
        slot = entry.get("slot") or {}
        key = (slot.get("entity"), slot.get("attribute"))
        if not key[0] or not key[1]:
            unscoped.append(entry)
            continue
        current = latest.get(key)
        if current is None:
            latest[key] = entry
            continue
        event_idx = int(slot.get("event_idx", -1))
        current_idx = int((current.get("slot") or {}).get("event_idx", -1))
        if event_idx > current_idx:
            latest[key] = entry
    output = list(latest.values()) + unscoped
    return sorted(output, key=lambda item: int(item.get("rank", 999)))


def apply_intervention(entries: list[dict[str, Any]], result: dict[str, Any], intervention: str) -> list[dict[str, Any]]:
    if intervention == "normal":
        return entries
    if intervention == "latest_per_slot":
        return latest_per_slot(entries)
    classes = [classify_entry(entry, result) for entry in entries]
    if intervention == "remove_stale_same_slot":
        return [entry for entry, cls in zip(entries, classes) if cls != "stale_same_slot"]
    if intervention == "remove_unrelated":
        return [entry for entry, cls in zip(entries, classes) if cls != "unrelated"]
    if intervention == "remove_near_slot":
        return [
            entry for entry, cls in zip(entries, classes)
            if cls not in {"same_entity_different_attribute", "different_entity_same_attribute"}
        ]
    if intervention == "remove_random_non_gold":
        removable_indices = [idx for idx, cls in enumerate(classes) if cls != "gold_same_slot"]
        stale_to_remove = stale_count(entries, result)
        remove_set = set(removable_indices[:stale_to_remove])
        return [entry for idx, entry in enumerate(entries) if idx not in remove_set]
    raise ValueError(f"unknown intervention: {intervention}")


def analyze_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = result.get("answer_trace") or {}
    entries = list(trace.get("retrieved_entries") or [])
    if not entries:
        return []
    rows = []
    normal_gold = has_gold(entries, result)
    normal_stale = stale_count(entries, result)
    for intervention in INTERVENTIONS:
        kept = apply_intervention(entries, result, intervention)
        rows.append({
            "example_id": result.get("example_id"),
            "intervention": intervention,
            "gold_in_context": has_gold(kept, result),
            "stale_count": stale_count(kept, result),
            "entry_count": len(kept),
            "removed_count": len(entries) - len(kept),
            "normal_gold_in_context": normal_gold,
            "normal_stale_count": normal_stale,
            "original_em": result.get("em", 0.0),
            "original_f1": result.get("f1", 0.0),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["intervention"]), []).append(row)
    output = []
    for intervention, items in sorted(grouped.items()):
        output.append({
            "intervention": intervention,
            "n": len(items),
            "gold_in_context_rate": mean(float(item["gold_in_context"]) for item in items),
            "stale_count_avg": mean(float(item["stale_count"]) for item in items),
            "entry_count_avg": mean(float(item["entry_count"]) for item in items),
            "removed_count_avg": mean(float(item["removed_count"]) for item in items),
            "original_em_avg": mean(float(item["original_em"]) for item in items),
            "original_f1_avg": mean(float(item["original_f1"]) for item in items),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stale-Specific Removal Trace Analysis",
        "",
        "This trace-level analysis estimates which interventions remove stale same-slot exposure before full answer-model reruns.",
        "",
        "| intervention | n | gold in context | stale count | entries | removed | original EM | original F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['intervention']} | {row['n']} | {row['gold_in_context_rate']:.3f} | "
            f"{row['stale_count_avg']:.3f} | {row['entry_count_avg']:.3f} | {row['removed_count_avg']:.3f} | "
            f"{row['original_em_avg']:.3f} | {row['original_f1_avg']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stale-specific removal interventions over saved answer traces")
    parser.add_argument("--input_json", default="results/p68_answer_layer_diagnostics/raw_add_k16_dev/evomemory_results.json")
    parser.add_argument("--output_dir", default="results/p83_stale_specific_removal_trace")
    args = parser.parse_args()

    results = load_results(Path(args.input_json))
    rows = []
    for result in results:
        rows.extend(analyze_result(result))
    summary = summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stale_specific_removal_examples.csv", rows)
    write_csv(output_dir / "stale_specific_removal_summary.csv", summary)
    (output_dir / "stale_specific_removal_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "stale_specific_removal_summary.md", summary)
    print(json.dumps({"num_rows": len(rows), "num_interventions": len(summary), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run py_compile**

Run:

```bash
PYTHONPATH=. python -m py_compile scripts/analyze_stale_specific_removal.py
```

Expected: command exits with status 0.

- [ ] **Step 3: Run on an existing traced result file if present**

Run:

```bash
PYTHONPATH=. python scripts/analyze_stale_specific_removal.py --input_json results/p68_answer_layer_diagnostics/raw_add_k16_dev/evomemory_results.json --output_dir results/p83_stale_specific_removal_trace
```

Expected output resembles:

```json
{
  "num_rows": 600,
  "num_interventions": 6,
  "output_dir": "results/p83_stale_specific_removal_trace"
}
```

If the input file does not include `answer_trace.retrieved_entries`, rerun the appropriate raw_add evaluation with `--save_answer_traces` before using this analyzer.

## Task 5: Add cluster runner scripts

**Files:**
- Create: `scripts/run_p83_conflict_type_probe_sui3.sh`
- Create: `scripts/run_p83_stale_conflict_dose_sui3.sh`

- [ ] **Step 1: Create conflict-type runner**

Write `scripts/run_p83_conflict_type_probe_sui3.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh

PYTHONPATH=. python scripts/run_conflict_type_probe.py \
  --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct \
  --examples_per_condition 128 \
  --distractor_count 4 \
  --conditions final_only,unrelated_distractors,same_entity_different_attribute,different_entity_same_attribute,stale_same_slot \
  --output_dir results/p83_conflict_type_probe/qwen25_7b_d4 \
  --no_qlora

PYTHONPATH=. python scripts/summarize_conflict_type_probe.py \
  --input_csv results/p83_conflict_type_probe/qwen25_7b_d4/conflict_type_examples.csv \
  --output_dir results/p83_conflict_type_probe_summary/qwen25_7b_d4
```

- [ ] **Step 2: Create stale-conflict dose runner**

Write `scripts/run_p83_stale_conflict_dose_sui3.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh

PYTHONPATH=. python scripts/run_synthetic_same_slot_probe.py \
  --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct \
  --examples_per_condition 64 \
  --stale_counts 0,1,2,4,8,16 \
  --value_policies conflict \
  --context_orders chronological,reverse_chronological,middle,random \
  --context_annotations none,latest_outdated_label \
  --output_dir results/p83_stale_conflict_dose/qwen25_7b \
  --no_qlora

PYTHONPATH=. python scripts/summarize_synthetic_same_slot_probe.py \
  --input_csv results/p83_stale_conflict_dose/qwen25_7b/synthetic_same_slot_examples.csv \
  --output_dir results/p83_stale_conflict_dose_summary/qwen25_7b
```

- [ ] **Step 3: Run shell syntax checks locally**

Run:

```bash
bash -n scripts/run_p83_conflict_type_probe_sui3.sh
bash -n scripts/run_p83_stale_conflict_dose_sui3.sh
```

Expected: both commands exit with status 0.

Do not launch the cluster jobs until after all local compile/smoke checks pass.

## Task 6: Add paper-facing plan note

**Files:**
- Create: `paper/p83_stale_same_slot_conflict_plan_note.md`

- [ ] **Step 1: Write the plan note**

Write this file:

```markdown
# P8.3 Stale Same-Slot Conflict Mechanism Plan

## Motivation

The next novelty-focused step is to sharpen MemUpdateBench from a repeated-update diagnostic into a mechanism analysis of stale same-slot context contamination. The target claim is that stale same-slot entries are not generic retrieval noise: they are high-similarity version conflicts that remain valid for the same entity and attribute while competing with the current value.

## Minimal experiment package

1. **Conflict-type decomposition** compares matched contexts with final-only evidence, unrelated distractors, same-entity/different-attribute distractors, different-entity/same-attribute distractors, and stale same-slot distractors.
2. **Version-conflict dose-response** extends the synthetic same-slot probe with middle/random placement and selected stale-count/order/annotation conditions.
3. **Stale-specific removal analysis** estimates whether removing stale same-slot entries is more targeted than removing unrelated or near-slot entries.

## Main expected paper claim

If the planned results match the hypothesis, the paper should claim that stale same-slot conflict is a distinct answer-layer failure mode in memory-augmented LLMs. Generic distractors may degrade performance, but obsolete same-slot values should cause disproportionate collapse because the answer model must arbitrate between historical versions of the same slot.

## Integration points

- Use `results/p83_conflict_type_probe_summary/` for the main novelty table.
- Use `results/p83_stale_conflict_dose_summary/` for the dose-response and order/annotation appendix table.
- Use `results/p83_stale_specific_removal_trace/` as a cheap trace-level guide before running full answer-model removal interventions.
- Keep existing Qwen/Llama/Mistral ceiling-recovery results as supporting evidence, not the main novelty.

## Guardrails

- Do not reframe MemUpdateBench as a broad memory benchmark.
- Do not present latest-per-slot as a deployed method.
- Do not claim external SDK failure without fair adapter runs.
- Treat synthetic probes as controlled mechanism evidence, not ecological realism.
```

- [ ] **Step 2: Check note renders as plain Markdown**

Run:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
text = Path('paper/p83_stale_same_slot_conflict_plan_note.md').read_text(encoding='utf-8')
assert '## Motivation' in text
assert '## Guardrails' in text
print('paper note ok')
PY
```

Expected:

```text
paper note ok
```

## Task 7: Run full local validation

**Files:**
- Validate all new/modified scripts.

- [ ] **Step 1: Compile new and modified scripts**

Run:

```bash
PYTHONPATH=. python -m py_compile \
  scripts/run_conflict_type_probe.py \
  scripts/summarize_conflict_type_probe.py \
  scripts/run_synthetic_same_slot_probe.py \
  scripts/analyze_stale_specific_removal.py \
  scripts/smoke_test.py
```

Expected: command exits with status 0.

- [ ] **Step 2: Run smoke tests**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected:

```text
SMOKE TEST: 26/26 passed
```

If the pass count changes because new named tests are added, all tests must still pass.

- [ ] **Step 3: Run deterministic summary fixture**

Run the fixture command from Task 2 Step 3.

Expected: `num_rows` is 2 and `num_conditions` is 2.

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
bash -n scripts/run_p83_conflict_type_probe_sui3.sh
bash -n scripts/run_p83_stale_conflict_dose_sui3.sh
```

Expected: both commands exit with status 0.

## Task 8: Update workflow after first real run

**Files:**
- Modify: `WORKFLOW.md`

- [ ] **Step 1: Append a P8.3 workflow section after obtaining real results**

After the first successful cluster run, append this section template to `WORKFLOW.md` and replace the metric values with actual outputs from the generated Markdown summaries:

```markdown
## P8.3 stale same-slot conflict mechanism package

### Motivation

P8.3 sharpens the paper's novelty by testing whether stale same-slot entries are a distinct high-similarity version-conflict mechanism rather than generic retrieval noise.

### Files changed/generated

```text
scripts/run_conflict_type_probe.py
scripts/summarize_conflict_type_probe.py
scripts/analyze_stale_specific_removal.py
scripts/run_p83_conflict_type_probe_sui3.sh
scripts/run_p83_stale_conflict_dose_sui3.sh
results/p83_conflict_type_probe/
results/p83_conflict_type_probe_summary/
results/p83_stale_conflict_dose/
results/p83_stale_conflict_dose_summary/
results/p83_stale_specific_removal_trace/
paper/p83_stale_same_slot_conflict_plan_note.md
```

### Commands run

```bash
PYTHONPATH=. python -m py_compile scripts/run_conflict_type_probe.py scripts/summarize_conflict_type_probe.py scripts/run_synthetic_same_slot_probe.py scripts/analyze_stale_specific_removal.py scripts/smoke_test.py
PYTHONPATH=. python scripts/smoke_test.py
bash scripts/run_p83_conflict_type_probe_sui3.sh
bash scripts/run_p83_stale_conflict_dose_sui3.sh
PYTHONPATH=. python scripts/analyze_stale_specific_removal.py --input_json <trace-result-json> --output_dir results/p83_stale_specific_removal_trace
```

### Results

- Conflict-type decomposition: `<fill from results/p83_conflict_type_probe_summary/.../conflict_type_summary.md>`.
- Version-conflict dose-response: `<fill from results/p83_stale_conflict_dose_summary/...>`.
- Stale-specific removal trace analysis: `<fill from results/p83_stale_specific_removal_trace/stale_specific_removal_summary.md>`.

### Conclusion

`<state whether stale same-slot distractors caused disproportionate degradation relative to generic and near-slot distractors, and whether this supports the revised mechanism-first paper framing.>`
```

Do not append this section before actual results exist.

- [ ] **Step 2: Validate Markdown after editing `WORKFLOW.md`**

Run:

```bash
PYTHONPATH=. python - <<'PY'
from pathlib import Path
text = Path('WORKFLOW.md').read_text(encoding='utf-8')
assert 'P8.3 stale same-slot conflict mechanism package' in text
assert '<fill' not in text
print('workflow p83 section ok')
PY
```

Expected:

```text
workflow p83 section ok
```

## Task 9: Commit completed planning and implementation changes

**Files:**
- Include only intentional spec/plan/script/note/workflow changes.

- [ ] **Step 1: Inspect git status**

Run:

```bash
git status --short
```

Expected: shows only intentional files. Do not stage generated temporary fixture directories under `results/tmp_conflict_summary_fixture`.

- [ ] **Step 2: Inspect diff**

Run:

```bash
git diff -- docs/superpowers/specs/2026-05-27-stale-same-slot-conflict-design.md docs/superpowers/plans/2026-05-27-stale-same-slot-conflict-implementation.md scripts/run_conflict_type_probe.py scripts/summarize_conflict_type_probe.py scripts/run_synthetic_same_slot_probe.py scripts/analyze_stale_specific_removal.py scripts/smoke_test.py scripts/run_p83_conflict_type_probe_sui3.sh scripts/run_p83_stale_conflict_dose_sui3.sh paper/p83_stale_same_slot_conflict_plan_note.md WORKFLOW.md
```

Expected: diff matches completed tasks; no unrelated manuscript or binary files are included.

- [ ] **Step 3: Stage intentional files**

For the planning-only commit requested before implementation, stage only:

```bash
git add docs/superpowers/specs/2026-05-27-stale-same-slot-conflict-design.md docs/superpowers/plans/2026-05-27-stale-same-slot-conflict-implementation.md
```

For a later implementation commit after tasks are executed, stage the relevant scripts, paper note, and workflow updates explicitly.

- [ ] **Step 4: Commit planning docs**

Run:

```bash
git commit -m "$(cat <<'EOF'
Plan stale same-slot conflict analysis

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. If a hook fails, fix the underlying issue and create a new commit; do not bypass hooks.
