from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mub.vnext.post_core.contracts_v1 import canonical_bytes
from mub.vnext.post_core.identity_v1 import (
    build_identity_evidence_receipt_v1,
    load_identity_evidence_v1,
)


EXIT_SUCCESS = 0
EXIT_USAGE = 11
EXIT_STALE_SOURCE = 12
EXIT_UNTRUSTED_RUNTIME = 14


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate official-document model identity evidence against the frozen "
            "post-Core Phase 0 index without network, provider, or model execution."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--evidence", required=True, help="canonical official identity evidence JSON")
    parser.add_argument("--phase0-index", required=True, help="authoritative post-Core Phase 0 artifact index")
    parser.add_argument("--execute", action="store_true", help="validate metadata and print a zero-call receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if not args.execute:
        print("identity validation requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        evidence_path = Path(args.evidence)
        bundle = load_identity_evidence_v1(evidence_path, Path(args.phase0_index))
        receipt = build_identity_evidence_receipt_v1(bundle, evidence_path)
    except ValueError:
        print("identity validation rejected untrusted or stale evidence", file=sys.stderr)
        return EXIT_STALE_SOURCE
    except OSError:
        print("identity validation could not read an authenticated source", file=sys.stderr)
        return EXIT_STALE_SOURCE
    except Exception as exc:
        print(f"identity validation rejected runtime: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNTRUSTED_RUNTIME
    sys.stdout.buffer.write(canonical_bytes(receipt) + b"\n")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
