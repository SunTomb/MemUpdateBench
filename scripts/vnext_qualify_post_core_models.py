from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from mub.vnext.post_core.qualification_v1 import CapabilityProbeReportV1, QualificationReportV1
from mub.vnext.post_core.release_v1 import (
    EXIT_PUBLICATION,
    EXIT_SUCCESS_WITH_PENDING,
    EXIT_UNTRUSTED_RUNTIME,
    EXIT_USAGE,
    PostCoreReleaseError,
    UnsafePathError,
    build_post_core_release_v1,
    load_post_core_config_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the no-network MemUpdateBench post-Core Phase 0 qualification metadata check.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="frozen Phase 0 release config JSON")
    parser.add_argument("--registry", help="optional canonical model registry JSON")
    parser.add_argument("--core-manifest", required=True, help="authenticated immutable Core source manifest")
    parser.add_argument("--task14-index", required=True, help="authenticated immutable Task 14 root index")
    parser.add_argument("--provenance", help="optional canonical provenance JSONL metadata input")
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
        publication = build_post_core_release_v1(
            config,
            Path(args.core_manifest),
            Path(args.task14_index),
            registry=Path(args.registry) if args.registry else None,
            provenance_path=Path(args.provenance) if args.provenance else None,
        )
        report = QualificationReportV1.model_validate_json(publication.artifact_bytes["qualification_report.json"])
        probes = CapabilityProbeReportV1.model_validate_json(publication.artifact_bytes["capability_probe_report.json"])
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
            "network_calls": probes.network_calls,
            "executable_call_count": publication.executable_call_count,
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
