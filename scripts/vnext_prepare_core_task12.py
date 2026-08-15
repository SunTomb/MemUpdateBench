from __future__ import annotations

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_text = str(_PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _project_root_text]
sys.path.insert(0, _project_root_text)

from mub.vnext.io import canonical_json_bytes
from mub.vnext.preparation.task12 import (
    Task12PreparationManifestV1,
    admit_task12_dry_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 12 pre-run admission dry-run only; execution is unavailable."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--core-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_manifest = Path(args.manifest).read_bytes()
    manifest = Task12PreparationManifestV1.model_validate_json(raw_manifest)
    if canonical_json_bytes(manifest) != raw_manifest:
        raise ValueError("Task 12 preparation manifest must be canonical JSON")
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=args.core_root,
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
    )
    sys.stdout.buffer.write(canonical_json_bytes(plan))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
