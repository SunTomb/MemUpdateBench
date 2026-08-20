from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mub.vnext.post_core.model_registry_v1 import build_initial_model_registry_v1
from mub.vnext.post_core.qualification_v1 import qualify_registry_offline_v1
from mub.vnext.post_core.release_v1 import (
    EXIT_PUBLICATION,
    EXIT_SUCCESS_WITH_PENDING,
    EXIT_UNTRUSTED_RUNTIME,
    EXIT_USAGE,
    PostCoreReleaseError,
    UnsafePathError,
    load_post_core_config_v1,
    load_post_core_registry_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the no-network MemUpdateBench post-Core Phase 0 qualification metadata check.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="frozen Phase 0 release config JSON")
    parser.add_argument("--registry", help="optional canonical model registry JSON")
    parser.add_argument("--core-manifest", help="optional Core source manifest for metadata context")
    parser.add_argument("--task14-index", help="optional Task 14 root index for metadata context")
    parser.add_argument("--provenance", help="optional canonical provenance JSONL metadata input")
    parser.add_argument("--output-root", help="reserved output-root context; qualification does not publish")
    parser.add_argument("--execute", action="store_true", help="validate/print metadata only; never execute calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if not args.execute:
        print("post-Core qualification requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        config = load_post_core_config_v1(Path(args.config))
        registry = load_post_core_registry_v1(Path(args.registry), config) if args.registry else build_initial_model_registry_v1()
        report, probes = qualify_registry_offline_v1(registry)
        pending = sum(row.status == "PENDING" for row in report.gates)
        blocked = sum(row.status == "BLOCKED" for row in report.gates)
        if blocked:
            status = "BLOCKED"
            code = 10
        else:
            status = "SUCCESS_WITH_PENDING" if pending else "SUCCESS"
            code = EXIT_SUCCESS_WITH_PENDING
        summary = {
            "status": status,
            "release_id": config.release_id,
            "pending_count": pending,
            "blocked_count": blocked,
            "network_allowed": probes.network_allowed,
            "provider_calls": probes.provider_calls,
            "model_loads": probes.model_loads,
            "network_calls": 0,
            "executable_call_count": 0,
        }
    except (ValueError, UnsafePathError) as exc:
        print(f"post-Core qualification contract/usage rejected: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except PostCoreReleaseError as exc:
        print(f"post-Core qualification failed: {exc}", file=sys.stderr)
        return EXIT_PUBLICATION
    except OSError as exc:
        print(f"post-Core qualification failed: {exc}", file=sys.stderr)
        return EXIT_PUBLICATION
    except Exception as exc:
        print(f"post-Core qualification rejected untrusted runtime: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNTRUSTED_RUNTIME
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
