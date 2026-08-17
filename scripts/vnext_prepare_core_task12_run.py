from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_text = str(_PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _project_root_text]
sys.path.insert(0, _project_root_text)

from mub.vnext.io import canonical_json_bytes
from mub.vnext.preparation.task12 import Task12DryRunPlanV1, Task12PreparationManifestV1
from mub.vnext.runtime.task12_bundle_v3 import build_task12_run_bundle_v3
from mub.vnext.runtime.task12_execution_v3 import task12_runtime_code_binding_v3


def _load_canonical(path: Path, model_type):
    raw = path.read_bytes()
    model = model_type.model_validate_json(raw)
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"noncanonical artifact: {path}")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one authorized Task 12 cell/slot execution bundle without executing it."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument(
        "--answer-model-slot",
        required=True,
        choices=("answer_model_a", "answer_model_b"),
    )
    parser.add_argument("--output-leaf", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _load_canonical(args.manifest, Task12PreparationManifestV1)
    plan = _load_canonical(args.plan, Task12DryRunPlanV1)
    bundle = build_task12_run_bundle_v3(
        manifest=manifest,
        plan=plan,
        core_root=args.core_root,
        evidence_root=args.evidence_root,
        repository_root=_PROJECT_ROOT,
        runtime_code_binding=task12_runtime_code_binding_v3(_PROJECT_ROOT),
        output_root=args.output_root,
        cell_id=args.cell_id,
        answer_model_slot=args.answer_model_slot,
        output_leaf=args.output_leaf,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "cell_id": bundle.cell_id,
                "answer_model_slot": bundle.answer_model_slot,
                "task_count": len(bundle.tasks),
                "bundle_root": str(bundle.bundle_root),
                "execution_output_root": str(bundle.execution_output_root),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
