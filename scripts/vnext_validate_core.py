from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mub.vnext.validation.core_release import validate_core_release


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a staged MemUpdateBench vNext Core candidate")
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--allow-bounded", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = validate_core_release(args.release_dir, expected_full=not args.allow_bounded)
    print(json.dumps({
        "status": "VALID",
        **report.model_dump(mode="json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
