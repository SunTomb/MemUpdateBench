from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.generation.post_core_artifacts import (
    build_post_core_artifact_bundle,
    publish_post_core_artifact_bundle,
    validate_post_core_artifact_tree,
)
from mub.vnext.generation.post_core_config import load_post_core_data_config


_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "vnext" / "post_core_data.yaml"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "vnext" / "main_track_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and publish the post-Core main-track v1 artifact set.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--validate", action="store_true", help="Validate an existing artifact root without building.")
    args = parser.parse_args()

    if args.validate:
        report = validate_post_core_artifact_tree(args.output_dir)
        print(f"validated {args.output_dir}: {report['semantic_core_count']} cores, {report['task_count']} tasks, review_status={report['review_status']}")
        return 0

    if type(args.code_revision) is not str or not args.code_revision.strip():
        parser.error("--code-revision is required when building")
    config = load_post_core_data_config(args.config)
    bundle = build_post_core_artifact_bundle(config, code_revision=args.code_revision)
    published = publish_post_core_artifact_bundle(bundle, args.output_dir)
    print(f"published {published.output_dir}: {bundle.semantic_core_count} cores, {bundle.task_count} tasks, review_status={bundle.validation_report['review_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
