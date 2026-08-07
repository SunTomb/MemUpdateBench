from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.generation.core_orchestrate import stage_core_candidate


def _revision(expected: str | None = None) -> str:
    for command in (("git", "diff", "--quiet"), ("git", "diff", "--cached", "--quiet")):
        result = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError("Core candidate generation requires a clean tracked revision")
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True, cwd=PROJECT_ROOT
    ).strip()
    if expected is not None and revision != expected:
        raise RuntimeError("Core candidate generation revision changed during staging")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a MemUpdateBench vNext Core candidate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cores-per-family", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    revision = _revision()
    result = None
    try:
        result = stage_core_candidate(
            config_path=args.config,
            output_dir=args.output_dir,
            code_revision=revision,
            cores_per_family=args.cores_per_family,
        )
        _revision(revision)
    except Exception:
        if result is not None:
            result.remove_if_unchanged()
        raise
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
