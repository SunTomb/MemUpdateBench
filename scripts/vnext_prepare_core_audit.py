from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.audit.core_stage import stage_core_audit_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage the strict-v3 224-task Core human-audit package"
    )
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-bounded", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    staged = stage_core_audit_package(
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        expected_full=not args.allow_bounded,
    )
    print(
        json.dumps(
            {
                "status": "DONE_FOR_HUMAN_INPUT",
                "output_dir": str(staged.output_dir),
                "selection_hash": staged.package.selection_hash,
                "source_task_manifest_hash": staged.package.source_task_manifest_hash,
                "selected_task_count": staged.selected_task_count,
                "review_surface_task_count": staged.review_surface_task_count,
                "decision_template_count": staged.decision_template_count,
                "adjudication_template_count": staged.adjudication_template_count,
                "release_ready": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
