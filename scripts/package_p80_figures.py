from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


PAPER_DIR = ROOT / "paper"


def import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping P8 figures because matplotlib is unavailable: {exc}")
        return None
    return plt


def load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return [str(png_path.relative_to(ROOT)), str(pdf_path.relative_to(ROOT))]


def plot_dose_response(output_dir: Path) -> list[str]:
    plt = import_matplotlib()
    if plt is None:
        return []
    stored = load_csv("results/p80_stale_dose_response/stale_dose_bins.csv")
    retrieved = load_csv("results/p80_stale_dose_response/retrieved_stale_dose_bins.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([as_float(r, "stale_same_slot_entry_count") for r in stored], [as_float(r, "em") for r in stored], marker="o", linewidth=2, label="Stored stale count")
    ax.plot([as_float(r, "stale_retrieved_count") for r in retrieved], [as_float(r, "em") for r in retrieved], marker="s", linewidth=2, label="Retrieved stale count")
    ax.set_title("Stale same-slot dose response", loc="left", fontsize=11)
    ax.set_xlabel("Stale same-slot count")
    ax.set_ylabel("Slot-prompt EM")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "p80_stale_dose_response")
    plt.close(fig)
    return paths


def plot_synthetic_matrix(output_dir: Path) -> list[str]:
    plt = import_matplotlib()
    if plt is None:
        return []
    rows = load_csv("results/p80_synthetic_same_slot_probe_analysis/synthetic_same_slot_grouped_summary.csv")
    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["value_policy"], row["context_order"], row["context_annotation"])
        by_key.setdefault(key, []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex=True, sharey=True)
    value_policies = [("conflict", "Conflict stale values"), ("same_as_current", "Same-value repetition")]
    orders = [("chronological", "Chronological"), ("reverse_chronological", "Reverse chronological")]
    styles = {
        "none": {"marker": "o", "label": "No labels"},
        "latest_outdated_label": {"marker": "s", "label": "Latest/outdated labels"},
    }
    for i, (policy, policy_label) in enumerate(value_policies):
        for j, (order, order_label) in enumerate(orders):
            ax = axes[i][j]
            for annotation in ["none", "latest_outdated_label"]:
                items = sorted(by_key.get((policy, order, annotation), []), key=lambda row: int(row["stale_count"]))
                if not items:
                    continue
                ax.plot([int(row["stale_count"]) for row in items], [float(row["em"]) for row in items], linewidth=2, **styles[annotation])
            ax.set_title(f"{policy_label} / {order_label}", loc="left", fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks([0, 1, 2, 4, 8, 16])
            ax.grid(True, alpha=0.25)
            if i == 1:
                ax.set_xlabel("Stale entries")
            if j == 0:
                ax.set_ylabel("Exact match")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths = save_figure(fig, output_dir, "p80_synthetic_same_slot_matrix")
    plt.close(fig)
    return paths


def plot_expanded_latest(output_dir: Path) -> list[str]:
    plt = import_matplotlib()
    if plt is None:
        return []
    rows = load_csv("results/p80_expanded_latest_per_slot_summary/expanded_latest_per_slot_summary.csv")
    points = []
    for row in rows:
        k = int(row["run_name"].split("_k", 1)[1].split("_", 1)[0])
        points.append((k, float(row["avg_em"]), float(row["gold_retrieved_rate"]), float(row["answer_value_present_rate"])))
    points.sort()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label="EM")
    ax.plot([p[0] for p in points], [p[2] for p in points], marker="s", linewidth=2, label="Gold retrieved")
    ax.plot([p[0] for p in points], [p[3] for p in points], marker="^", linewidth=2, label="Answer value present")
    ax.set_title("Expanded latest-per-slot sweep", loc="left", fontsize=11)
    ax.set_xlabel("Repeated updates per slot (k)")
    ax.set_ylabel("Rate")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "p80_expanded_latest_per_slot")
    plt.close(fig)
    return paths


def plot_llama_comparison(output_dir: Path) -> list[str]:
    plt = import_matplotlib()
    if plt is None:
        return []
    rows = load_csv("results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.csv")
    order = [
        ("raw_add_normal_topk5", "Normal"),
        ("raw_add_latest_per_slot_topk5", "Latest\nper-slot"),
        ("raw_add_latest_label_topk5", "Labels"),
        ("raw_add_chronological_topk5", "Chron."),
        ("raw_add_reverse_chronological_topk5", "Rev.\nchron."),
    ]
    by_name = {row["run_name"]: row for row in rows}
    labels = [label for name, label in order if name in by_name]
    ems = [float(by_name[name]["avg_em"]) for name, _ in order if name in by_name]
    avps = [float(by_name[name]["answer_value_present_rate"]) for name, _ in order if name in by_name]
    xs = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar([x - width / 2 for x in xs], ems, width=width, label="EM")
    ax.bar([x + width / 2 for x in xs], avps, width=width, label="Answer value present")
    ax.set_title("Llama3.1-8B stale susceptibility", loc="left", fontsize=11)
    ax.set_ylabel("Rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "p80_llama_stale_susceptibility")
    plt.close(fig)
    return paths


def main() -> None:
    output_dir = PAPER_DIR / "figures"
    generated = []
    generated.extend(plot_dose_response(output_dir))
    generated.extend(plot_synthetic_matrix(output_dir))
    generated.extend(plot_expanded_latest(output_dir))
    generated.extend(plot_llama_comparison(output_dir))
    manifest = {
        "generated": generated,
        "count": len(generated),
    }
    out = output_dir / "p80_figure_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"count": len(generated), "manifest": str(out.relative_to(ROOT))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
