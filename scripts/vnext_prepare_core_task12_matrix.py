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
from mub.vnext.runtime.task12_execution_v3 import task12_runtime_code_binding_v3
from mub.vnext.runtime.task12_matrix_v3 import build_task12_matrix_bundles_v3


def _load_canonical(path: Path, model_type):
    raw = path.read_bytes()
    model = model_type.model_validate_json(raw)
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"noncanonical artifact: {path}")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare all 18 authorized Task 12 cell/slot execution bundles without executing them."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _load_canonical(args.manifest, Task12PreparationManifestV1)
    plan = _load_canonical(args.plan, Task12DryRunPlanV1)
    matrix = build_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        core_root=args.core_root,
        evidence_root=args.evidence_root,
        repository_root=_PROJECT_ROOT,
        runtime_code_binding=task12_runtime_code_binding_v3(_PROJECT_ROOT),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "bundle_count": len(matrix.bundles),
                "matrix_root": str(matrix.matrix_root),
                "matrix_manifest_path": str(matrix.matrix_manifest_path),
                "run_bundles": [
                    {
                        "cell_id": bundle.cell_id,
                        "answer_model_slot": bundle.answer_model_slot,
                        "bundle_root": str(bundle.bundle_root),
                        "execution_output_root": str(bundle.execution_output_root),
                    }
                    for bundle in matrix.bundles
                ],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
