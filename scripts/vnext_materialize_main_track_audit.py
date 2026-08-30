from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.generation.post_core_audit import (
    build_main_track_audit_packet,
    publish_main_track_audit_packet,
)


_DEFAULT_CANDIDATE = _PROJECT_ROOT / "data" / "vnext" / "main_track_v1_independence_v1"
_DEFAULT_SELECTION = (
    _PROJECT_ROOT
    / "results"
    / "vnext"
    / "main_track_v1_independence_audit_selection"
    / "selection.json"
)
_DEFAULT_OUTPUT = _PROJECT_ROOT / "results" / "vnext" / "main_track_v1_independence_audit_packet_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a no-replace main-track human-audit packet"
    )
    parser.add_argument("--candidate-root", type=Path, default=_DEFAULT_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=_DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        packet = build_main_track_audit_packet(args.candidate_root, args.selection)
        published = publish_main_track_audit_packet(packet, args.output_root)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_root": str(published.output_dir),
                "artifact_paths": [str(path) for path in published.artifact_paths],
                "artifact_hashes": published.artifact_hashes,
                "candidate_artifact_hashes": packet.candidate_artifact_hashes,
                "selection_artifact_hash": packet.selection_artifact_hash,
                "packet_row_hash": packet.packet_row_hash,
                "row_count": len(packet.rows),
                "review_status": packet.manifest["review_status"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
