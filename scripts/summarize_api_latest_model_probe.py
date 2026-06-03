from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.glob("*/api_synthetic_dose_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary["model"]
        for condition, by_stale in summary["by_condition_and_stale_count"].items():
            for stale_count, stats in by_stale.items():
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "stale_count": int(stale_count),
                        "n": int(stats["n"]),
                        "em": float(stats["em"]),
                        "stale_copied": float(stats["stale_copied"]),
                    }
                )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "api_latest_model_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "api_latest_model_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "stale_count", "n", "em", "stale_copied"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# P8.4 API latest-model summary",
        "",
        "| Model | Condition | Stale count | n | EM | Stale copied |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['stale_count']} | {row['n']} | "
            f"{row['em']:.3f} | {row['stale_copied']:.3f} |"
        )
    (output_dir / "api_latest_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize P8.4 API latest-model stale-conflict probes")
    parser.add_argument("--result-root", default="results/p84_api_latest_model_probe")
    parser.add_argument("--output-dir", default="results/p84_api_latest_model_probe_summary")
    args = parser.parse_args()
    rows = load_rows(Path(args.result_root))
    write_outputs(rows, Path(args.output_dir))
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
