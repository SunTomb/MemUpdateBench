from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_update_frequency import (
    make_k16_thesis_table,
    make_metric_table,
    make_slot_prompt_em_f1_table,
    row_map,
)

METHODS = ["constrained_slot_crud", "raw_add", "heuristic_crud", "long25"]
PAPER_KS = [1, 2, 4, 8, 16]
PAPER_LABELS = {
    "constrained_slot_crud": "Constrained CRUD",
    "raw_add": "Raw append",
    "heuristic_crud": "Heuristic CRUD",
    "long25": "Long25",
}
LINE_STYLES = {
    "constrained_slot_crud": {"marker": "o", "linestyle": "-"},
    "raw_add": {"marker": "s", "linestyle": "--"},
    "heuristic_crud": {"marker": "^", "linestyle": "-."},
    "long25": {"marker": "D", "linestyle": ":"},
}


def load_summary_rows(summary_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a 'rows' list in {summary_json}")
    return rows


def fmt(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def paper_label(method: str) -> str:
    return PAPER_LABELS.get(method, method)


def metric_points(rows: list[dict[str, Any]], answer_mode: str, metric: str, method: str) -> tuple[list[int], list[float]]:
    by_key = row_map(rows)
    xs = []
    ys = []
    for k in PAPER_KS:
        data = by_key.get((method, answer_mode, k))
        if not data:
            continue
        value = data.get(metric)
        if value == "" or value is None:
            continue
        xs.append(k)
        ys.append(float(value))
    return xs, ys


def import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping paper figures because matplotlib is unavailable: {exc}")
        return None
    return plt


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


def write_tradeoff_figure(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = import_matplotlib()
    if plt is None:
        return []

    specs = [
        ("slot_direct", "state_accuracy", "A. Final-state reliability", "State accuracy"),
        ("slot_prompt", "avg_em", "B. Slot-prompt answering", "Exact match"),
        ("slot_prompt", "stale_same_slot_entry_count_avg", "C. Stale same-slot burden", "Stale entries"),
        ("slot_prompt", "final_memory_size_avg", "D. Memory compactness", "Final memory size"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    handles = []
    labels = []
    for ax, (answer_mode, metric, title, ylabel) in zip(axes.ravel(), specs):
        for method in METHODS:
            xs, ys = metric_points(rows, answer_mode, metric, method)
            if not xs:
                continue
            line, = ax.plot(xs, ys, label=paper_label(method), linewidth=2, markersize=5, **LINE_STYLES[method])
            if paper_label(method) not in labels:
                handles.append(line)
                labels.append(paper_label(method))
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(PAPER_KS)
        ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("Number of repeated updates per slot (k)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = save_figure(fig, output_dir, "p63_update_frequency_tradeoff")
    plt.close(fig)
    return paths


def write_gap_figure(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = import_matplotlib()
    if plt is None:
        return []

    by_key = row_map(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        xs = []
        ys = []
        for k in PAPER_KS:
            direct = by_key.get((method, "slot_direct", k), {}).get("state_accuracy")
            prompt = by_key.get((method, "slot_prompt", k), {}).get("avg_em")
            if direct == "" or direct is None or prompt == "" or prompt is None:
                continue
            xs.append(k)
            ys.append(float(direct) - float(prompt))
        ax.plot(xs, ys, label=paper_label(method), linewidth=2, markersize=5, **LINE_STYLES[method])
    ax.set_title("Oracle-state vs slot-prompt gap", loc="left", fontsize=11)
    ax.set_xlabel("Number of repeated updates per slot (k)")
    ax.set_ylabel("State accuracy - slot-prompt EM")
    ax.set_xticks(PAPER_KS)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "p63_gap_slot_direct_vs_prompt")
    plt.close(fig)
    return paths


def write_k16_scatter(rows: list[dict[str, Any]], output_dir: Path, x_metric: str, xlabel: str, stem: str, title: str) -> list[Path]:
    plt = import_matplotlib()
    if plt is None:
        return []

    by_key = row_map(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in METHODS:
        data = by_key.get((method, "slot_prompt", 16), {})
        x_value = data.get(x_metric)
        y_value = data.get("avg_em")
        if x_value == "" or x_value is None or y_value == "" or y_value is None:
            continue
        ax.scatter(float(x_value), float(y_value), s=70, label=paper_label(method))
        ax.annotate(paper_label(method), (float(x_value), float(y_value)), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("k=16 slot-prompt EM")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, stem)
    plt.close(fig)
    return paths


def write_derived_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths = []
    paths.extend(write_gap_figure(rows, output_dir))
    paths.extend(write_k16_scatter(rows, output_dir, "stale_same_slot_entry_count_avg", "k=16 stale same-slot entries", "p63_stale_vs_prompt_em_k16", "Stale burden vs answer robustness"))
    paths.extend(write_k16_scatter(rows, output_dir, "final_memory_size_avg", "k=16 final memory size", "p63_memory_size_vs_prompt_em_k16", "Memory size vs answer robustness"))
    return paths


def k16_value(rows: list[dict[str, Any]], method: str, answer_mode: str, metric: str) -> Any:
    return row_map(rows).get((method, answer_mode, 16), {}).get(metric, "")


def k16_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    body = []
    for method in METHODS:
        direct = k16_value(rows, method, "slot_direct", "state_accuracy")
        prompt_em = k16_value(rows, method, "slot_prompt", "avg_em")
        prompt_f1 = k16_value(rows, method, "slot_prompt", "avg_f1")
        stale = k16_value(rows, method, "slot_prompt", "stale_same_slot_entry_count_avg")
        size = k16_value(rows, method, "slot_prompt", "final_memory_size_avg")
        body.append([paper_label(method), fmt(direct), f"{fmt(prompt_em)} / {fmt(prompt_f1)}", fmt(stale), fmt(size)])
    return body


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def latex_metric(value: Any) -> str:
    return latex_escape(fmt(value))


def latex_sweep_rows(rows: list[dict[str, Any]], answer_mode: str, metric: str) -> list[str]:
    by_key = row_map(rows)
    table_rows = []
    for method in METHODS:
        values = [latex_metric(by_key.get((method, answer_mode, k), {}).get(metric, "")) for k in PAPER_KS]
        table_rows.append(" & ".join([paper_label(method), *values]) + r" \\")
    return table_rows


def latex_prompt_em_f1_rows(rows: list[dict[str, Any]]) -> list[str]:
    by_key = row_map(rows)
    table_rows = []
    for method in METHODS:
        values = []
        for k in PAPER_KS:
            data = by_key.get((method, "slot_prompt", k), {})
            values.append(f"{fmt(data.get('avg_em', ''))}/{fmt(data.get('avg_f1', ''))}" if data else "—")
        table_rows.append(" & ".join([paper_label(method), *values]) + r" \\")
    return table_rows


def latex_figure_snippet(filename: str, caption: str, label: str) -> list[str]:
    return [
        r"\begin{figure}[t]",
        r"  \centering",
        f"  \\includegraphics[width=0.78\\linewidth]{{paper/{filename}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]


def latex_sweep_table(title: str, label: str, rows: list[str]) -> list[str]:
    return [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lccccc}",
        r"    \hline",
        r"    Method & k=1 & k=2 & k=4 & k=8 & k=16 \\ ",
        r"    \hline",
        *[f"    {row}" for row in rows],
        r"    \hline",
        r"  \end{tabular}",
        f"  \\caption{{{title}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
        "",
    ]


def write_latex_snippets(rows: list[dict[str, Any]], output_path: Path) -> None:
    table_rows = [" & ".join(latex_escape(item) for item in row) + r" \\" for row in k16_table_rows(rows)]
    lines = [
        r"% Auto-generated by scripts/package_update_frequency_paper_assets.py",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=0.95\linewidth]{paper/p63_update_frequency_tradeoff.pdf}",
        r"  \caption{Update-frequency stress test on P6.3 hard splits. Oracle-like slot-direct state evaluation hides stale-memory burden for append-only and heuristic methods, while slot-prompt answering reveals answer collapse as repeated same-slot updates accumulate stale entries. Long25 is substantially more compact but sacrifices final-state reliability under hard distractors.}",
        r"  \label{fig:p63-update-frequency-tradeoff}",
        r"\end{figure}",
        "",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lcccc}",
        r"    \hline",
        r"    Method & State acc. & EM/F1 & Stale & Mem. size \\ ",
        r"    \hline",
        *[f"    {row}" for row in table_rows],
        r"    \hline",
        r"  \end{tabular}",
        r"  \caption{k=16 tradeoff among final-state reliability, slot-prompt answer quality, stale same-slot burden, and memory compactness. Append-only and heuristic managers preserve slot-direct recoverability but collapse under slot-prompt answering; Long25 is compact but not fully reliable.}",
        r"  \label{tab:p63-k16-tradeoff}",
        r"\end{table}",
        "",
        r"% Derived diagnostic figures",
        *latex_figure_snippet(
            "p63_gap_slot_direct_vs_prompt.pdf",
            "Gap between oracle-like slot-state accuracy and slot-conditioned exact match across update frequencies. Larger gaps indicate cases where final-state recoverability is preserved while realistic answering degrades.",
            "fig:p63-direct-prompt-gap",
        ),
        *latex_figure_snippet(
            "p63_stale_vs_prompt_em_k16.pdf",
            "At k=16, stale same-slot burden is strongly associated with lower slot-conditioned exact match. This view highlights stale evidence as the main failure pressure for append-only and partial-compaction methods.",
            "fig:p63-stale-vs-em",
        ),
        *latex_figure_snippet(
            "p63_memory_size_vs_prompt_em_k16.pdf",
            "At k=16, final memory size and slot-conditioned exact match expose the compactness--robustness frontier. Long25 is compact but imperfectly reliable, while append-only memory is large and brittle under prompting.",
            "fig:p63-size-vs-em",
        ),
        r"% Appendix-ready k-sweep tables",
        *latex_sweep_table(
            "Slot-direct state accuracy across update frequencies.",
            "tab:p63-slot-direct-sweep",
            latex_sweep_rows(rows, "slot_direct", "state_accuracy"),
        ),
        *latex_sweep_table(
            "Slot-prompt EM/F1 across update frequencies.",
            "tab:p63-slot-prompt-sweep",
            latex_prompt_em_f1_rows(rows),
        ),
        *latex_sweep_table(
            "Stale same-slot entries across update frequencies.",
            "tab:p63-stale-sweep",
            latex_sweep_rows(rows, "slot_prompt", "stale_same_slot_entry_count_avg"),
        ),
        *latex_sweep_table(
            "Final memory size across update frequencies.",
            "tab:p63-memory-size-sweep",
            latex_sweep_rows(rows, "slot_prompt", "final_memory_size_avg"),
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_asset_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    raw_em = k16_value(rows, "raw_add", "slot_prompt", "avg_em")
    raw_f1 = k16_value(rows, "raw_add", "slot_prompt", "avg_f1")
    raw_stale = k16_value(rows, "raw_add", "slot_prompt", "stale_same_slot_entry_count_avg")
    raw_size = k16_value(rows, "raw_add", "slot_prompt", "final_memory_size_avg")
    heuristic_em = k16_value(rows, "heuristic_crud", "slot_prompt", "avg_em")
    heuristic_f1 = k16_value(rows, "heuristic_crud", "slot_prompt", "avg_f1")
    heuristic_stale = k16_value(rows, "heuristic_crud", "slot_prompt", "stale_same_slot_entry_count_avg")
    long_direct = k16_value(rows, "long25", "slot_direct", "state_accuracy")
    long_em = k16_value(rows, "long25", "slot_prompt", "avg_em")
    long_f1 = k16_value(rows, "long25", "slot_prompt", "avg_f1")
    long_stale = k16_value(rows, "long25", "slot_prompt", "stale_same_slot_entry_count_avg")
    long_size = k16_value(rows, "long25", "slot_prompt", "final_memory_size_avg")
    constrained_direct = k16_value(rows, "constrained_slot_crud", "slot_direct", "state_accuracy")
    constrained_em = k16_value(rows, "constrained_slot_crud", "slot_prompt", "avg_em")
    constrained_f1 = k16_value(rows, "constrained_slot_crud", "slot_prompt", "avg_f1")
    constrained_stale = k16_value(rows, "constrained_slot_crud", "slot_prompt", "stale_same_slot_entry_count_avg")

    sections = [
        "# P6.3 Update-Frequency Paper Assets",
        "",
        "## Main figure",
        "",
        "- `p63_update_frequency_tradeoff.png`",
        "- `p63_update_frequency_tradeoff.pdf`",
        "- `p63_update_frequency_latex_snippets.tex`",
        "- `p63_gap_slot_direct_vs_prompt.png` / `.pdf`",
        "- `p63_stale_vs_prompt_em_k16.png` / `.pdf`",
        "- `p63_memory_size_vs_prompt_em_k16.png` / `.pdf`",
        "",
        "Suggested caption: Update-frequency stress test on P6.3 hard splits. Oracle-like slot-direct state evaluation hides stale-memory burden for append-only and heuristic methods, while slot-prompt answering reveals answer collapse as repeated same-slot updates accumulate stale entries. Long25 is substantially more compact but sacrifices final-state reliability under hard distractors.",
        "",
        "## Main body table: k=16 tradeoff",
        "",
        make_k16_thesis_table(rows, METHODS),
        "",
        "Suggested caption: At k=16, append-only and heuristic managers preserve final-state recoverability under slot-direct evaluation but collapse under slot-prompt answering because stale same-slot entries remain in memory. Long25 reduces stale burden and memory size, but its final-state accuracy is below the constrained CRUD upper bound.",
        "",
        "## Experimental narrative bullets",
        "",
        f"- The constrained slot CRUD upper-bound reaches k=16 slot-direct state accuracy {fmt(constrained_direct)} with no stale same-slot burden ({fmt(constrained_stale)}) and slot-prompt EM/F1 {fmt(constrained_em)} / {fmt(constrained_f1)}.",
        f"- Raw append preserves k=16 slot-direct recoverability but retains {fmt(raw_stale)} stale same-slot entries, grows to final memory size {fmt(raw_size)}, and drops to slot-prompt EM/F1 {fmt(raw_em)} / {fmt(raw_f1)}.",
        f"- Heuristic CRUD partially limits memory growth but still leaves {fmt(heuristic_stale)} stale same-slot entries at k=16 and reaches only slot-prompt EM/F1 {fmt(heuristic_em)} / {fmt(heuristic_f1)}.",
        f"- Long25 is much more compact at k=16, with stale same-slot {fmt(long_stale)} and final memory size {fmt(long_size)}, but its slot-direct state accuracy is {fmt(long_direct)} rather than 1.00.",
        "- The central result is therefore a tradeoff curve: recoverability, stale burden, compactness, and answer robustness move differently as repeated updates increase.",
        "- These results support the P6.7 manuscript package; external baseline feasibility should be considered only if the integrated draft still needs ecosystem grounding.",
        "",
        "## Appendix table: slot-direct state accuracy",
        "",
        make_metric_table(rows, "slot_direct", "state_accuracy", METHODS, digits=2),
        "",
        "## Appendix table: slot-prompt EM/F1",
        "",
        make_slot_prompt_em_f1_table(rows, METHODS),
        "",
        "## Appendix table: stale same-slot entries",
        "",
        make_metric_table(rows, "slot_prompt", "stale_same_slot_entry_count_avg", METHODS, digits=2),
        "",
        "## Appendix table: final memory size",
        "",
        make_metric_table(rows, "slot_prompt", "final_memory_size_avg", METHODS, digits=2),
        "",
    ]
    output_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package update-frequency summary results into paper-ready assets")
    parser.add_argument("--summary_json", default="results/update_frequency_p63_summary/update_frequency_summary.json")
    parser.add_argument("--paper_dir", default="paper")
    args = parser.parse_args()

    summary_json = Path(args.summary_json)
    paper_dir = Path(args.paper_dir)
    rows = load_summary_rows(summary_json)
    paper_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = write_tradeoff_figure(rows, paper_dir)
    figure_paths.extend(write_derived_figures(rows, paper_dir))
    asset_path = paper_dir / "p63_update_frequency_assets.md"
    latex_path = paper_dir / "p63_update_frequency_latex_snippets.tex"
    write_asset_markdown(rows, asset_path)
    write_latex_snippets(rows, latex_path)

    print(f"Loaded {len(rows)} summary rows from {summary_json}")
    for path in figure_paths:
        print(f"Wrote {path}")
    print(f"Wrote {asset_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
