from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.schema_export import export_schemas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic vNext JSON Schemas.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exported = export_schemas(args.output_dir)
    print(f"Exported {len(exported)} vNext schemas to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
