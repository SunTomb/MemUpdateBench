from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def table(lines: list[str], header: list[str], rows: list[list[Any]], numeric_from: int = 1) -> None:
    lines.append("| " + " | ".join(header) + " |")
    aligns = ["---" if idx < numeric_from else "---:" for idx in range(len(header))]
    lines.append("| " + " | ".join(aligns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")


def simple_external_table(lines: list[str]) -> None:
    path = ROOT / "results/p80_simple_external_pipeline_summary/simple_external_pipeline_summary.csv"
    if not path.exists():
        return
    rows = load_csv("results/p80_simple_external_pipeline_summary/simple_external_pipeline_summary.csv")
    rows = [row for row in rows if not row["run_name"].startswith("smoke_")]
    rows.sort(key=lambda row: row["run_name"])
    lines.extend(["## Simple extract-then-store external pipeline", ""])
    table(lines, ["run", "update", "answer", "EM", "F1", "state", "stale same-slot", "memory size"], [
        [row["run_name"], row["update_policy"], row["answer_mode"], fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["state_accuracy"]), fmt(row["stale_same_slot_entry_count_avg"]), fmt(row["final_memory_size_avg"])]
        for row in rows
    ], numeric_from=3)
    lines.append("")


def lost_in_middle_table(lines: list[str]) -> None:
    path = ROOT / "results/p80_lost_in_middle_probe_summary/lost_in_middle_summary.csv"
    if not path.exists():
        return
    rows = load_csv("results/p80_lost_in_middle_probe_summary/lost_in_middle_summary.csv")
    order = {"beginning": 0, "middle": 1, "end": 2}
    rows.sort(key=lambda row: (row["run_name"], order.get(row["gold_position"], 99)))
    lines.extend(["## Lost-in-the-Middle gold-position probe", ""])
    table(lines, ["run", "gold position", "distractors", "EM", "F1", "answer value present"], [
        [row["run_name"], row["gold_position"], row["num_distractors"], fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["answer_value_present_rate"])]
        for row in rows
    ], numeric_from=2)
    lines.append("")


def expanded_latest_table(lines: list[str]) -> None:
    rows = load_csv("results/p80_expanded_latest_per_slot_summary/expanded_latest_per_slot_summary.csv")
    parsed = []
    for row in rows:
        run = row["run_name"]
        k = run.split("_k", 1)[1].split("_", 1)[0]
        parsed.append((int(k), row))
    parsed.sort()
    lines.extend(["## Expanded latest-per-slot all-k", ""])
    table(lines, ["k", "EM", "F1", "answer value present", "gold retrieved", "stale retrieved"], [
        [k, fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["answer_value_present_rate"]), fmt(row["gold_retrieved_rate"]), fmt(row["stale_retrieved_rate"])]
        for k, row in parsed
    ])
    lines.append("")


MODEL_RUN_ORDER = {
    "raw_add_normal_topk5": 0,
    "raw_add_latest_per_slot_topk5": 1,
    "raw_add_latest_label_topk5": 2,
    "raw_add_chronological_topk5": 3,
    "raw_add_reverse_chronological_topk5": 4,
}

MODEL_RUN_LABELS = {
    "raw_add_normal_topk5": "normal top-k5",
    "raw_add_latest_per_slot_topk5": "latest_per_slot",
    "raw_add_latest_label_topk5": "latest/outdated labels",
    "raw_add_chronological_topk5": "chronological",
    "raw_add_reverse_chronological_topk5": "reverse chronological",
}


def model_stale_table(lines: list[str], title: str, csv_path: str) -> None:
    path = ROOT / csv_path
    if not path.exists():
        return
    rows = load_csv(csv_path)
    rows.sort(key=lambda row: MODEL_RUN_ORDER.get(row["run_name"], 99))
    lines.extend([title, ""])
    table(lines, ["condition", "EM", "F1", "answer value present", "gold retrieved", "stale retrieved"], [
        [MODEL_RUN_LABELS.get(row["run_name"], row["run_name"]), fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["answer_value_present_rate"]), fmt(row["gold_retrieved_rate"]), fmt(row["stale_retrieved_rate"])]
        for row in rows
    ])
    lines.append("")


def llama_table(lines: list[str]) -> None:
    model_stale_table(lines, "## Llama3.1-8B stale susceptibility", "results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.csv")


def mistral_table(lines: list[str]) -> None:
    model_stale_table(lines, "## Mistral-7B third-model stale susceptibility", "results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.csv")


def zero_stale_table(lines: list[str], title: str, prompt_path: str, direct_path: str) -> None:
    prompt = ROOT / prompt_path
    direct = ROOT / direct_path
    if not prompt.exists() or not direct.exists():
        return
    rows = []
    for label, path in [("slot_prompt", prompt), ("slot_direct", direct)]:
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.append([
            label,
            fmt(summary["avg_em"]),
            fmt(summary["avg_f1"]),
            fmt(summary.get("value_em")),
            fmt(summary.get("answer_value_present_rate")),
            fmt(summary["state_accuracy"]),
            fmt(summary["stale_same_slot_entry_count_avg"]),
        ])
    lines.extend([title, ""])
    table(lines, ["answer", "EM", "F1", "value EM", "answer value present", "state", "stale same-slot"], rows)
    lines.append("")


def llama_zero_stale_table(lines: list[str]) -> None:
    zero_stale_table(
        lines,
        "## Llama3.1-8B constrained zero-stale control",
        "results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json",
        "results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json",
    )


def mistral_zero_stale_table(lines: list[str]) -> None:
    zero_stale_table(
        lines,
        "## Mistral-7B constrained zero-stale control",
        "results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json",
        "results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json",
    )


def ceiling_recovery_table(lines: list[str]) -> None:
    llama_path = ROOT / "results/p81_llama_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json"
    mistral_path = ROOT / "results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json"
    if not llama_path.exists() or not mistral_path.exists():
        return
    lines.extend(["## Ceiling-recovery comparison", ""])
    table(lines, ["model", "normal raw_add EM", "latest_per_slot EM", "zero-stale EM", "latest - ceiling"], [
        ["Qwen2.5-7B", "0.110", "0.690", "0.700", "-0.010"],
        ["Llama3.1-8B", "0.060", "0.290", fmt(json.loads(llama_path.read_text(encoding="utf-8"))["summary"]["avg_em"]), "+0.020"],
        ["Mistral-7B", "0.080", "0.720", fmt(json.loads(mistral_path.read_text(encoding="utf-8"))["summary"]["avg_em"]), "0.000"],
    ])
    lines.append("")


def p81_synthetic_table(lines: list[str]) -> None:
    path = ROOT / "results/p81_synthetic_same_slot_probe_expanded_analysis/synthetic_same_slot_grouped_summary.csv"
    if not path.exists():
        return
    rows = load_csv("results/p81_synthetic_same_slot_probe_expanded_analysis/synthetic_same_slot_grouped_summary.csv")
    keep = []
    for row in rows:
        condition = f"{row['value_policy']} + {row['context_order']} + {row['context_annotation']}"
        keep.append([condition, row["stale_count"], row["n"], fmt(row["em"]), fmt(row["f1"]), fmt(row["answer_value_present"])])
    lines.extend(["## P8.1 expanded synthetic same-slot diagnostic subset", ""])
    table(lines, ["condition", "stale", "n", "EM", "F1", "answer value present"], keep, numeric_from=1)
    lines.append("")


def p81_heuristic_table(lines: list[str]) -> None:
    path = ROOT / "results/p81_heuristic_threshold_k16_rigor_summary/heuristic_threshold_k16_summary.csv"
    if not path.exists():
        return
    rows = load_csv("results/p81_heuristic_threshold_k16_rigor_summary/heuristic_threshold_k16_summary.csv")
    lines.extend(["## P8.1 fixed-k heuristic threshold sweep", ""])
    table(lines, ["answer", "threshold", "EM", "F1", "state", "stale", "same-slot", "memory size"], [
        [row["answer_mode"], row["threshold"], fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["state_accuracy"]), fmt(row["stale_same_slot_entry_count_avg"]), fmt(row["same_slot_entry_count_avg"]), fmt(row["final_memory_size_avg"])]
        for row in rows
    ], numeric_from=1)
    lines.append("")


def real_context_table(lines: list[str]) -> None:
    rows = load_csv("results/p80_mechanism_probe_summary/context_mechanism_summary.csv")
    rows = [row for row in rows if not row["run_name"].startswith("smoke_")]
    order = {
        "normal_order_none": 0,
        "chronological_none": 1,
        "reverse_chronological_none": 2,
        "normal_timestamp": 3,
        "normal_latest_outdated_label": 4,
        "current_first_none": 5,
        "current_last_none": 6,
    }
    rows.sort(key=lambda row: order.get(row["run_name"], 99))
    lines.extend(["## Real-context mechanism probe", ""])
    table(lines, ["condition", "EM", "F1", "answer value present", "gold retrieved", "stale retrieved avg"], [
        [row["run_name"], fmt(row["avg_em"]), fmt(row["avg_f1"]), fmt(row["answer_value_present_rate"]), fmt(row["gold_retrieved_rate"]), fmt(row["stale_retrieved_avg"])]
        for row in rows
    ])
    lines.append("")


def dose_tables(lines: list[str]) -> None:
    bins = load_csv("results/p80_stale_dose_response/stale_dose_bins.csv")
    retrieved = load_csv("results/p80_stale_dose_response/retrieved_stale_dose_bins.csv")
    fits = load_json("results/p80_stale_dose_response/stale_dose_logistic.json")
    lines.extend(["## Stored stale dose-response", ""])
    table(lines, ["stored stale count", "n", "EM", "F1", "answer value present"], [
        [fmt(row["stale_same_slot_entry_count"]), row["n"], fmt(row["em"]), fmt(row["f1"]), fmt(row["answer_value_present"])]
        for row in bins
    ])
    lines.append("")
    lines.extend(["## Retrieved stale dose-response", ""])
    table(lines, ["retrieved stale count", "n", "EM", "F1", "answer value present"], [
        [fmt(row["stale_retrieved_count"]), row["n"], fmt(row["em"]), fmt(row["f1"]), fmt(row["answer_value_present"])]
        for row in retrieved
    ])
    lines.append("")
    lines.extend(["## Logistic dose-response fit", ""])
    table(lines, ["predictor", "n", "slope", "ED50"], [
        [predictor, fit["n"], fmt(fit["slope"]), fmt(fit["ed50"])]
        for predictor, fit in fits.items()
    ])
    lines.append("")


def main() -> None:
    lines = ["# P8.0 Paper Table Pack", ""]
    lines.append("These tables are generated from result artifacts and are intended as manuscript/appendix source material, not final LaTeX.")
    lines.append("")
    dose_tables(lines)
    real_context_table(lines)
    llama_table(lines)
    llama_zero_stale_table(lines)
    mistral_table(lines)
    mistral_zero_stale_table(lines)
    ceiling_recovery_table(lines)
    p81_synthetic_table(lines)
    p81_heuristic_table(lines)
    simple_external_table(lines)
    lost_in_middle_table(lines)
    expanded_latest_table(lines)
    out = ROOT / "paper" / "p80_paper_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
