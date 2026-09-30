from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.validation.cross_domain_qdrant_state import validate_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--bea-root", type=Path, required=True)
    parser.add_argument("--noaa-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_evidence(args.collected_root, args.bea_root, args.noaa_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
