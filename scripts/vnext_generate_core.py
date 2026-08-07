from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from mub.vnext.generation.core_orchestrate import stage_core_candidate


def _revision() -> str:
    for command in (("git", "diff", "--quiet"), ("git", "diff", "--cached", "--quiet")):
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError("Core candidate generation requires a clean tracked revision")
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a MemUpdateBench vNext Core candidate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cores-per-family", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = stage_core_candidate(
        config_path=args.config,
        output_dir=args.output_dir,
        code_revision=_revision(),
        cores_per_family=args.cores_per_family,
    )
    print(json.dumps({
        "status": "VALID_STAGED_CANDIDATE",
        "release_dir": str(result.release_dir),
        "semantic_core_count": result.semantic_core_count,
        "task_count": result.task_count,
        "split_core_counts": result.split_core_counts,
        "split_task_counts": result.split_task_counts,
        "hard_suite_core_count": result.hard_suite_core_count,
        "hard_suite_task_count": result.hard_suite_task_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
