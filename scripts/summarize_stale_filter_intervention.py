from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary", {})


def fmt(value, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize retrieval-time stale-filter intervention results")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--filtered", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_note", default="")
    args = parser.parse_args()

    normal = load_summary(Path(args.normal))
    filtered = load_summary(Path(args.filtered))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_matches = ["mode", "answer_mode", "num_examples", "answer_topk", "slot_prompt_variant"]
    mismatches = {
        key: [normal.get(key), filtered.get(key)]
        for key in required_matches
        if normal.get(key) != filtered.get(key)
    }
    if mismatches and not args.baseline_note:
        raise ValueError(f"Summary provenance mismatch without baseline note: {mismatches}")

    rows = [
        {"condition": "normal", **normal},
        {"condition": "latest_per_slot", **filtered},
    ]
    effect = {
        "em_delta": filtered.get("avg_em", 0.0) - normal.get("avg_em", 0.0),
        "f1_delta": filtered.get("avg_f1", 0.0) - normal.get("avg_f1", 0.0),
        "answer_value_present_delta": filtered.get("answer_value_present_rate", 0.0) - normal.get("answer_value_present_rate", 0.0),
        "memory_size_delta": filtered.get("final_memory_size_avg", 0.0) - normal.get("final_memory_size_avg", 0.0),
        "stale_same_slot_delta": filtered.get("stale_same_slot_entry_count_avg", 0.0) - normal.get("stale_same_slot_entry_count_avg", 0.0),
    }
    summary = {"rows": rows, "effect": effect, "baseline_note": args.baseline_note}
    (output_dir / "stale_filter_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_rows = []
    for row in rows:
        md_rows.append(
            "| {condition} | {n} | {em} | {f1} | {present} | {state} | {stale} | {same} | {gold} | {mem} | {policy} |".format(
                condition=row["condition"],
                n=row.get("num_examples", ""),
                em=fmt(row.get("avg_em")),
                f1=fmt(row.get("avg_f1")),
                present=fmt(row.get("answer_value_present_rate")),
                state=fmt(row.get("state_accuracy")),
                stale=fmt(row.get("stale_same_slot_entry_count_avg")),
                same=fmt(row.get("same_slot_entry_count_avg")),
                gold=fmt(row.get("gold_same_slot_entry_count_avg")),
                mem=fmt(row.get("final_memory_size_avg")),
                policy=row.get("retrieval_policy", ""),
            )
        )
    md = "\n".join([
        "# Retrieval-Time Stale Filter Summary",
        "",
        "| Condition | N | EM | F1 | Answer value present | State acc. | Stale same-slot | Same-slot | Gold same-slot | Memory size | Retrieval policy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *md_rows,
        "",
        "## Effect",
        "",
        f"- EM delta: {fmt(effect['em_delta'])}",
        f"- F1 delta: {fmt(effect['f1_delta'])}",
        f"- Answer-value-present delta: {fmt(effect['answer_value_present_delta'])}",
        f"- Memory-size delta: {fmt(effect['memory_size_delta'])}",
        f"- Stale-burden delta: {fmt(effect['stale_same_slot_delta'])}",
        "",
        args.baseline_note,
        "",
    ])
    (output_dir / "stale_filter_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
