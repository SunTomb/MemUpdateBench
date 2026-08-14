from __future__ import annotations

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_text = str(_PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _project_root_text]
sys.path.insert(0, _project_root_text)

from mub.vnext.runtime.answer_model_v3 import (
    AnswerModelSlotV3,
    OfflinePromptedAnswerModelV3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight one frozen offline Core answer-model slot."
    )
    parser.add_argument("--slot", required=True, choices=("answer_model_a", "answer_model_b"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license-id", required=True, choices=("apache-2.0",))
    parser.add_argument("--tree-manifest-sha256", required=True)
    parser.add_argument("--dependency-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only:
        raise ValueError("Task 11 runner only supports --preflight-only")
    if args.dependency_path is not None:
        dependency_path = Path(args.dependency_path).resolve(strict=True)
        if not dependency_path.is_dir() or dependency_path.is_symlink():
            raise ValueError("dependency path must be a real directory")
        sys.path.insert(0, str(dependency_path))
    model = OfflinePromptedAnswerModelV3(
        slot=AnswerModelSlotV3(
            slot_id=args.slot,
            model_id=args.model_id,
            snapshot_path=args.model_snapshot,
            revision=args.revision,
            license_id=args.license_id,
            tree_manifest_sha256=args.tree_manifest_sha256,
        ),
        device=args.device,
    )
    try:
        model.load()
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
