from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def by_example(rows: list[dict]) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["example_id"])][row["context_mode"]] = row
    return grouped


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for context in sorted({row["context_mode"] for row in rows}):
        subset = [row for row in rows if row["context_mode"] == context]
        output.append({
            "context_mode": context,
            "num_examples": len(subset),
            "avg_em": avg(subset, "em"),
            "avg_f1": avg(subset, "f1"),
            "value_em": avg(subset, "value_em"),
            "answer_value_present": avg(subset, "answer_value_present"),
            "gold_retrieved_rate": avg(subset, "gold_value_in_retrieved"),
            "stale_retrieved_rate": avg(subset, "stale_same_slot_in_retrieved"),
            "distractor_retrieved_rate": avg(subset, "same_name_distractor_in_retrieved"),
        })
    return output


def intervention_effects(rows: list[dict], baseline_context: str, target_contexts: list[str]) -> list[dict]:
    grouped = by_example(rows)
    effects = []
    for target in target_contexts:
        pairs = []
        for example_id, contexts in grouped.items():
            if baseline_context in contexts and target in contexts:
                base = contexts[baseline_context]
                other = contexts[target]
                pairs.append((base, other))
        effects.append({
            "baseline_context": baseline_context,
            "target_context": target,
            "num_pairs": len(pairs),
            "em_delta": mean([other.get("em", 0.0) - base.get("em", 0.0) for base, other in pairs]) if pairs else 0.0,
            "f1_delta": mean([other.get("f1", 0.0) - base.get("f1", 0.0) for base, other in pairs]) if pairs else 0.0,
            "value_em_delta": mean([float(other.get("value_em", 0.0)) - float(base.get("value_em", 0.0)) for base, other in pairs]) if pairs else 0.0,
            "gold_retrieved_delta": mean([float(other.get("gold_value_in_retrieved", 0.0)) - float(base.get("gold_value_in_retrieved", 0.0)) for base, other in pairs]) if pairs else 0.0,
            "stale_retrieved_delta": mean([float(other.get("stale_same_slot_in_retrieved", 0.0)) - float(base.get("stale_same_slot_in_retrieved", 0.0)) for base, other in pairs]) if pairs else 0.0,
            "fixed_examples": [
                base.get("example_id") for base, other in pairs
                if not base.get("value_em", False) and other.get("value_em", False)
            ][:20],
            "regressed_examples": [
                base.get("example_id") for base, other in pairs
                if base.get("value_em", False) and not other.get("value_em", False)
            ][:20],
        })
    return effects


def avg(rows: list[dict], key: str) -> float:
    return mean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


def write_md(payload: dict, output_path: Path) -> None:
    lines = [
        "# Stale Intervention Summary",
        "",
        "## Context summaries",
        "",
        "| Context | N | EM | F1 | Value EM | Answer value present | Gold retrieved | Stale retrieved | Distractor retrieved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["context_summaries"]:
        lines.append(
            f"| {row['context_mode']} | {row['num_examples']} | {row['avg_em']:.3f} | {row['avg_f1']:.3f} | "
            f"{row['value_em']:.3f} | {row['answer_value_present']:.3f} | {row['gold_retrieved_rate']:.3f} | "
            f"{row['stale_retrieved_rate']:.3f} | {row['distractor_retrieved_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired intervention effects",
        "",
        "| Baseline | Target | Pairs | ΔEM | ΔF1 | ΔValue EM | ΔGold retrieved | ΔStale retrieved | Fixed | Regressed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["intervention_effects"]:
        lines.append(
            f"| {row['baseline_context']} | {row['target_context']} | {row['num_pairs']} | "
            f"{row['em_delta']:.3f} | {row['f1_delta']:.3f} | {row['value_em_delta']:.3f} | "
            f"{row['gold_retrieved_delta']:.3f} | {row['stale_retrieved_delta']:.3f} | "
            f"{len(row['fixed_examples'])} | {len(row['regressed_examples'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stale-burden intervention effects from answer-layer diagnostics")
    parser.add_argument("--input", required=True, help="answer_layer_mechanism.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_context", default="retrieval_topk_5")
    parser.add_argument("--target_contexts", nargs="+", default=["gold_context", "retrieval_topk_10", "retrieval_topk_1"])
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    payload = {
        "input": args.input,
        "baseline_context": args.baseline_context,
        "target_contexts": args.target_contexts,
        "context_summaries": summarize(rows),
        "intervention_effects": intervention_effects(rows, args.baseline_context, args.target_contexts),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stale_intervention_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload, output_dir / "stale_intervention_summary.md")
    print(json.dumps(payload["intervention_effects"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
