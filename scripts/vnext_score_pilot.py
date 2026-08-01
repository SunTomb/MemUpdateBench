from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mub.vnext.scoring.aggregate import aggregate_scores
from mub.vnext.scoring.pilot import authenticate_pilot_files, publish_scores, score_pilot_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score normalized MemUpdateBench vNext Pilot records.")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--run-records", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = authenticate_pilot_files(
            args.tasks,
            args.task_manifest,
            args.run_records,
            args.run_manifest,
        )
        scores = score_pilot_records(
            bundle.tasks,
            bundle.runs,
            bundle.task_manifest,
            bundle.run_manifest,
        )
        summary = aggregate_scores(scores, bundle.tasks, bundle.run_manifest)
        publish_scores(args.output_dir, scores, summary, bundle.run_manifest)
    except (OSError, TypeError, ValueError) as exc:
        print(f"vNext Pilot scoring failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
