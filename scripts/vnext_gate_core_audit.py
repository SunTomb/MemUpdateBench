from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.audit.core_stage import gate_core_audit_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate strict-v3 Core human review and adjudication evidence"
    )
    parser.add_argument("--selection-package", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--selected-tasks", required=True, type=Path)
    parser.add_argument("--surface-context", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = gate_core_audit_files(
        selection_package_path=args.selection_package,
        candidate_dir=args.candidate_dir,
        selected_tasks_path=args.selected_tasks,
        surface_context_path=args.surface_context,
        decisions_path=args.decisions,
        adjudications_path=args.adjudications,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    release_ready = report.release_ready
    status = (
        "TEST_ONLY_HUMAN_INPUT_REQUIRED"
        if not report.full_candidate
        else "RELEASE_READY"
        if release_ready
        else "REMEDIATION_REQUIRED"
        if report.remediations
        else "HUMAN_INPUT_REQUIRED"
    )
    print(
        json.dumps(
            {
                "status": status,
                "release_ready": release_ready,
                "terminal_pass_count": len(report.terminal_pass_audit_ids),
                "required_adjudication_count": len(report.required_adjudication_ids),
                "unresolved_adjudication_count": len(report.unresolved_adjudication_ids),
                "remediation_count": len(report.remediations),
                "raw_agreement": report.raw_agreement,
                "cohens_kappa": report.cohens_kappa,
                "issue_count": len(report.issues),
            },
            sort_keys=True,
        )
    )
    return 0 if release_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
