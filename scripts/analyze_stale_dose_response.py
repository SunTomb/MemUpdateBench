from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FIELDNAMES = [
    "source_path",
    "result_group",
    "run_name",
    "mode",
    "answer_mode",
    "retrieval_policy",
    "data_file",
    "example_id",
    "k_updates",
    "attribute",
    "em",
    "f1",
    "value_em",
    "answer_value_present",
    "same_slot_entry_count",
    "stale_same_slot_entry_count",
    "gold_same_slot_entry_count",
    "memory_size",
    "gold_retrieved_count",
    "stale_retrieved_count",
]


DEFAULT_INPUTS = [
    "results/update_frequency_p63/raw_add_slot_prompt_k1/evomemory_results.json",
    "results/update_frequency_p63/raw_add_slot_prompt_k2/evomemory_results.json",
    "results/update_frequency_p63/raw_add_slot_prompt_k4/evomemory_results.json",
    "results/update_frequency_p63/raw_add_slot_prompt_k8/evomemory_results.json",
    "results/update_frequency_p63/raw_add_slot_prompt_k16/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k1_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k2_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k4_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k8_dev_merged/evomemory_results.json",
    "results/p69_expanded_slot_prompt_allk/raw_add_prompt_k16_dev_merged/evomemory_results.json",
]


def result_group(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def retrieved_counts(row: dict[str, Any]) -> tuple[int | str, int | str]:
    trace = row.get("answer_trace") or {}
    entries = trace.get("retrieved_entries") or []
    gold = row.get("gold_answer", "")
    gold_state = row.get("gold_state") or {}
    if not entries:
        return "", ""
    gold_count = 0
    stale_count = 0
    for entry in entries:
        slot = entry.get("slot") or {}
        if slot.get("entity") != gold_state.get("entity") or slot.get("attribute") != gold_state.get("attribute"):
            continue
        value = str(slot.get("value", ""))
        if value.strip().lower() == str(gold).strip().lower():
            gold_count += 1
        else:
            stale_count += 1
    return gold_count, stale_count


def load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) or {}
    rows = []
    for row in payload.get("results", []) or []:
        gold_state = row.get("gold_state") or {}
        gold_ret, stale_ret = retrieved_counts(row)
        rows.append({
            "source_path": str(path),
            "result_group": result_group(path),
            "run_name": path.parent.name,
            "mode": summary.get("mode", ""),
            "answer_mode": summary.get("answer_mode", ""),
            "retrieval_policy": summary.get("retrieval_policy", "normal"),
            "data_file": summary.get("data_file", ""),
            "example_id": row.get("example_id", ""),
            "k_updates": row.get("k_updates", ""),
            "attribute": gold_state.get("attribute", ""),
            "em": float(row.get("em", 0.0)),
            "f1": float(row.get("f1", 0.0)),
            "value_em": int(bool(row.get("value_em", False))),
            "answer_value_present": int(bool(row.get("answer_value_present", False))),
            "same_slot_entry_count": row.get("same_slot_entry_count", ""),
            "stale_same_slot_entry_count": row.get("stale_same_slot_entry_count", ""),
            "gold_same_slot_entry_count": row.get("gold_same_slot_entry_count", ""),
            "memory_size": row.get("memory_size", ""),
            "gold_retrieved_count": gold_ret,
            "stale_retrieved_count": stale_ret,
        })
    return rows


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bin_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    bins: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = safe_float(row.get(key))
        if value is None:
            continue
        bins[value].append(row)
    out = []
    for value in sorted(bins):
        items = bins[value]
        out.append({
            key: value,
            "n": len(items),
            "em": mean(float(item["em"]) for item in items),
            "f1": mean(float(item["f1"]) for item in items),
            "answer_value_present": mean(float(item["answer_value_present"]) for item in items),
        })
    return out


def fit_logistic(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    xs = []
    ys = []
    for row in rows:
        x = safe_float(row.get(key))
        if x is None:
            continue
        xs.append(x)
        ys.append(float(row.get("em", 0.0)))
    if len(set(xs)) < 2 or not ys:
        return {"status": "insufficient_variation"}

    intercept = 0.0
    slope = 0.0
    lr = 0.01
    n = len(xs)
    for _ in range(8000):
        grad_i = 0.0
        grad_s = 0.0
        for x, y in zip(xs, ys):
            z = max(min(intercept + slope * x, 30.0), -30.0)
            p = 1.0 / (1.0 + math.exp(-z))
            grad_i += p - y
            grad_s += (p - y) * x
        intercept -= lr * grad_i / n
        slope -= lr * grad_s / n
    ed50 = -intercept / slope if slope else None
    return {
        "status": "ok",
        "n": n,
        "intercept": intercept,
        "slope": slope,
        "ed50": ed50,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], stale_bins: list[dict[str, Any]], retrieved_bins: list[dict[str, Any]], models: dict[str, Any]) -> None:
    lines = [
        "# P8.0 Stale Dose-Response Analysis",
        "",
        "## Scope",
        "",
        "This analysis uses existing raw_add slot-prompt result files and extracts per-example stale burden versus answer correctness.",
        "It is an analysis artifact for deciding whether stale contamination is gradual or threshold-like.",
        "",
        "## Inputs",
        "",
    ]
    for source in sorted({row["source_path"] for row in rows}):
        lines.append(f"- `{source}`")
    lines.extend([
        "",
        "## Binned by stored stale same-slot count",
        "",
        "| stale same-slot count | n | EM | F1 | answer value present |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in stale_bins:
        lines.append(f"| {row['stale_same_slot_entry_count']:.2f} | {row['n']} | {row['em']:.3f} | {row['f1']:.3f} | {row['answer_value_present']:.3f} |")
    lines.extend([
        "",
        "## Binned by retrieved stale same-slot count",
        "",
        "| retrieved stale count | n | EM | F1 | answer value present |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in retrieved_bins:
        lines.append(f"| {row['stale_retrieved_count']:.2f} | {row['n']} | {row['em']:.3f} | {row['f1']:.3f} | {row['answer_value_present']:.3f} |")
    lines.extend([
        "",
        "## Logistic fits",
        "",
        "These are lightweight first-pass fits, not final statistical claims.",
        "",
        "| predictor | status | n | slope | ED50 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for name, model in models.items():
        lines.append(f"| {name} | {model.get('status')} | {model.get('n', '—')} | {model.get('slope', 0.0):.4f} | {model.get('ed50', '—')} |")
    lines.extend([
        "",
        "## Initial interpretation checklist",
        "",
        "- If EM drops sharply from stale=0 to stale=1, the paper can claim even one stale same-slot entry is harmful.",
        "- If retrieved-stale count predicts EM better than stored-stale count, the mechanism is answer-context exposure rather than mere memory-store pollution.",
        "- If latest_per_slot rows are included later, they should be analyzed separately because the intervention changes retrieval scope.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stale burden dose-response from existing EvoMemory results")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", default="results/p80_stale_dose_response")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"missing input: {path}")
            continue
        rows.extend(load_examples(path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "stale_dose_examples.csv", rows, FIELDNAMES)
    stale_bins = bin_rows(rows, "stale_same_slot_entry_count")
    retrieved_bins = bin_rows(rows, "stale_retrieved_count")
    write_csv(output_dir / "stale_dose_bins.csv", stale_bins, ["stale_same_slot_entry_count", "n", "em", "f1", "answer_value_present"])
    write_csv(output_dir / "retrieved_stale_dose_bins.csv", retrieved_bins, ["stale_retrieved_count", "n", "em", "f1", "answer_value_present"])
    models = {
        "stored_stale_same_slot_count": fit_logistic(rows, "stale_same_slot_entry_count"),
        "retrieved_stale_same_slot_count": fit_logistic(rows, "stale_retrieved_count"),
    }
    (output_dir / "stale_dose_logistic.json").write_text(json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "stale_dose_response.md", rows, stale_bins, retrieved_bins, models)
    print(json.dumps({"num_examples": len(rows), "output_dir": str(output_dir), "models": models}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
