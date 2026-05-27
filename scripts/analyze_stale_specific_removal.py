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
