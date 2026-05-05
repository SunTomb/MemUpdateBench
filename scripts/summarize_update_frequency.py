from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "result_path",
    "method",
    "answer_mode",
    "k_updates",
    "mode",
    "data_file",
    "num_examples",
    "state_accuracy",
    "state_resolve_rate",
    "avg_em",
    "avg_f1",
    "answer_value_present_rate",
    "same_slot_entry_count_avg",
    "stale_same_slot_entry_count_avg",
    "gold_same_slot_entry_count_avg",
    "final_memory_size_avg",
    "write_amplification_avg",
    "target_slot_write_count_avg",
    "retrieval_policy",
    "lora_checkpoint",
]

METHOD_ORDER = ["oracle", "constrained_slot_crud", "raw_add", "heuristic_crud", "long25"]
ANSWER_MODE_ORDER = ["slot_direct", "slot_prompt", "rag", "short_prompt", "unknown"]
K_ORDER = [1, 2, 4, 8, 16, 32]


def parse_result_name(name: str) -> tuple[str, str, int | None]:
    match = re.match(r"^(?P<prefix>.+)_k(?P<k>\d+)(?:_.+)?$", name)
    if not match:
        return name, "unknown", None

    prefix = match.group("prefix")
    k_updates = int(match.group("k"))
    mode_aliases = {
        "direct": "slot_direct",
        "prompt": "slot_prompt",
    }
    for suffix, normalized in mode_aliases.items():
        marker = f"_{suffix}"
        if prefix.endswith(marker):
            method = prefix[: -len(marker)]
            return method, normalized, k_updates
    for answer_mode in ("slot_direct", "slot_prompt", "short_prompt", "rag"):
        suffix = f"_{answer_mode}"
        if prefix.endswith(suffix):
            method = prefix[: -len(suffix)]
            return method, answer_mode, k_updates
    return prefix, "unknown", k_updates


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    method = row["method"]
    answer_mode = row["answer_mode"]
    k_updates = row["k_updates"]
    method_idx = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    mode_idx = ANSWER_MODE_ORDER.index(answer_mode) if answer_mode in ANSWER_MODE_ORDER else len(ANSWER_MODE_ORDER)
    k_idx = K_ORDER.index(k_updates) if k_updates in K_ORDER else len(K_ORDER)
    return method_idx, mode_idx, k_idx, str(k_updates)


def load_rows(result_roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in result_roots:
        for path in sorted(root.glob("*/evomemory_results.json")):
            method, parsed_answer_mode, k_updates = parse_result_name(path.parent.name)
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            summary = payload.get("summary", {})
            row = {
                "result_path": str(path),
                "method": method,
                "answer_mode": summary.get("answer_mode", parsed_answer_mode),
                "k_updates": k_updates,
                "mode": summary.get("mode", ""),
                "data_file": summary.get("data_file", ""),
                "num_examples": summary.get("num_examples", ""),
                "state_accuracy": summary.get("state_accuracy", ""),
                "state_resolve_rate": summary.get("state_resolve_rate", ""),
                "avg_em": summary.get("avg_em", ""),
                "avg_f1": summary.get("avg_f1", ""),
                "answer_value_present_rate": summary.get("answer_value_present_rate", ""),
                "same_slot_entry_count_avg": summary.get("same_slot_entry_count_avg", ""),
                "stale_same_slot_entry_count_avg": summary.get("stale_same_slot_entry_count_avg", ""),
                "gold_same_slot_entry_count_avg": summary.get("gold_same_slot_entry_count_avg", ""),
                "final_memory_size_avg": summary.get("final_memory_size_avg", ""),
                "write_amplification_avg": summary.get("write_amplification_avg", ""),
                "target_slot_write_count_avg": summary.get("target_slot_write_count_avg", ""),
                "retrieval_policy": summary.get("retrieval_policy", "normal"),
                "lora_checkpoint": summary.get("lora_checkpoint", ""),
            }
            rows.append(row)
    return sorted(rows, key=sort_key)


def fmt(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    mapped = {}
    for row in rows:
        if row["k_updates"] is None:
            continue
        key = (row["method"], row["answer_mode"], row["k_updates"])
        if key in mapped:
            previous = mapped[key]
            if previous.get("retrieval_policy") != row.get("retrieval_policy"):
                raise ValueError(
                    "Duplicate summary cell with different retrieval policies: "
                    f"{key} -> {previous.get('result_path')} vs {row.get('result_path')}"
                )
        mapped[key] = row
    return mapped


def make_metric_table(rows: list[dict[str, Any]], answer_mode: str, metric: str, methods: list[str], digits: int = 2) -> str:
    by_key = row_map(rows)
    headers = ["method"] + [f"k={k}" for k in K_ORDER if any((m, answer_mode, k) in by_key for m in methods)]
    ks = [int(header.split("=")[1]) for header in headers[1:]]
    body = []
    for method in methods:
        if not any((method, answer_mode, k) in by_key for k in ks):
            continue
        body.append([method] + [fmt(by_key.get((method, answer_mode, k), {}).get(metric, ""), digits) for k in ks])
    return table(headers, body)


def make_slot_prompt_em_f1_table(rows: list[dict[str, Any]], methods: list[str]) -> str:
    by_key = row_map(rows)
    ks = [k for k in K_ORDER if any((m, "slot_prompt", k) in by_key for m in methods)]
    body = []
    for method in methods:
        if not any((method, "slot_prompt", k) in by_key for k in ks):
            continue
        row = [method]
        for k in ks:
            data = by_key.get((method, "slot_prompt", k), {})
            if not data:
                row.append("—")
            else:
                row.append(f"{fmt(data.get('avg_em'), 2)} / {fmt(data.get('avg_f1'), 2)}")
        body.append(row)
    return table(["method"] + [f"k={k} EM/F1" for k in ks], body)


def make_k16_thesis_table(rows: list[dict[str, Any]], methods: list[str]) -> str:
    by_key = row_map(rows)
    body = []
    for method in methods:
        direct = by_key.get((method, "slot_direct", 16), {})
        prompt = by_key.get((method, "slot_prompt", 16), {})
        if not direct and not prompt:
            continue
        burden_source = prompt or direct
        body.append([
            method,
            fmt(direct.get("state_accuracy", ""), 2),
            f"{fmt(prompt.get('avg_em', ''), 2)} / {fmt(prompt.get('avg_f1', ''), 2)}",
            fmt(burden_source.get("stale_same_slot_entry_count_avg", ""), 2),
            fmt(burden_source.get("final_memory_size_avg", ""), 2),
        ])
    return table(["method", "k=16 slot_direct state_acc", "k=16 slot_prompt EM/F1", "stale same-slot", "final memory size"], body)


def expected_missing(rows: list[dict[str, Any]]) -> list[str]:
    present = {(row["method"], row["answer_mode"], row["k_updates"]) for row in rows}
    expected = []
    for method in ["constrained_slot_crud", "raw_add", "heuristic_crud", "long25"]:
        for answer_mode in ["slot_direct", "slot_prompt"]:
            for k in [1, 2, 4, 8, 16]:
                if (method, answer_mode, k) not in present:
                    expected.append(f"{method}_{answer_mode}_k{k}")
    return expected


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    methods = ["constrained_slot_crud", "raw_add", "heuristic_crud", "long25"]
    missing = expected_missing(rows)
    sections = [
        "# Update-frequency P6 summary",
        "",
        "## Missing expected P6.3 cells",
        "",
        "None" if not missing else "\n".join(f"- {item}" for item in missing),
        "",
        "## Slot-direct state accuracy",
        "",
        make_metric_table(rows, "slot_direct", "state_accuracy", methods, digits=2),
        "",
        "## Slot-prompt EM/F1",
        "",
        make_slot_prompt_em_f1_table(rows, methods),
        "",
        "## Stale same-slot entries",
        "",
        make_metric_table(rows, "slot_prompt", "stale_same_slot_entry_count_avg", methods, digits=2),
        "",
        "## Final memory size",
        "",
        make_metric_table(rows, "slot_prompt", "final_memory_size_avg", methods, digits=2),
        "",
        "## k=16 thesis table",
        "",
        make_k16_thesis_table(rows, methods),
        "",
    ]
    output_path.write_text("\n".join(sections), encoding="utf-8")


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PNG plots because matplotlib is unavailable: {exc}")
        return []

    methods = ["constrained_slot_crud", "raw_add", "heuristic_crud", "long25"]
    by_key = row_map(rows)
    specs = [
        ("slot_direct", "state_accuracy", "Slot-direct state accuracy", "slot_direct_state_accuracy.png"),
        ("slot_prompt", "avg_em", "Slot-prompt EM", "slot_prompt_em.png"),
        ("slot_prompt", "stale_same_slot_entry_count_avg", "Stale same-slot entries", "stale_same_slot.png"),
        ("slot_prompt", "final_memory_size_avg", "Final memory size", "final_memory_size.png"),
    ]
    written = []
    for answer_mode, metric, title, filename in specs:
        plt.figure(figsize=(7, 4.5))
        has_data = False
        for method in methods:
            points = [(k, by_key[(method, answer_mode, k)][metric]) for k in K_ORDER if (method, answer_mode, k) in by_key]
            if not points:
                continue
            has_data = True
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", label=method)
        if not has_data:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("k updates")
        plt.ylabel(metric)
        plt.xticks([1, 2, 4, 8, 16])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize update-frequency EvoMemory results")
    parser.add_argument("--result_root", nargs="+", default=["results/update_frequency_p63"], help="Result roots containing */evomemory_results.json")
    parser.add_argument("--output_dir", default="results/update_frequency_p63_summary")
    args = parser.parse_args()

    roots = [Path(path) for path in args.result_root]
    rows = load_rows(roots)
    if not rows:
        raise SystemExit(f"No evomemory_results.json files found under: {', '.join(map(str, roots))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "update_frequency_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "update_frequency_summary.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output_dir / "update_frequency_tables.md"
    write_markdown(rows, md_path)

    plot_paths = write_plots(rows, output_dir)

    print(f"Loaded {len(rows)} result files")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
