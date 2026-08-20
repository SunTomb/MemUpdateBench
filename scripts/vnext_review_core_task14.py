from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mub.vnext.release.task14_publish import publish_task14_review_v1
from mub.vnext.release.task14_sources import (
    TASK14_EXPECTED_TASK13_INDEX_SHA256,
    Task14SourcePathsV1,
    load_task14_sources_v1,
)
from mub.vnext.statistics.task13_v3 import current_clean_task13_runtime_v3


EXIT_APPROVED = 0
EXIT_NOT_APPROVED = 10
EXIT_USAGE = 11
EXIT_STALE_SOURCE = 12
EXIT_PUBLICATION = 13
EXIT_UNTRUSTED_RUNTIME = 14


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only MemUpdateBench Core Task 14 final review.",
        allow_abbrev=False,
    )
    parser.add_argument("--core-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--task13-root", required=True)
    parser.add_argument("--task13-audit", required=True)
    parser.add_argument("--remote-task13-staging", required=True)
    parser.add_argument("--expected-task13-index-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    repository_root: Path | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        raise
    if not args.execute:
        print("Task 14 final review requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    if args.expected_task13_index_sha256 != TASK14_EXPECTED_TASK13_INDEX_SHA256:
        print("Task 14 rejected: expected Task 13 index hash is not frozen", file=sys.stderr)
        return EXIT_USAGE
    repository = repository_root or Path(__file__).resolve().parents[1]
    try:
        runtime = current_clean_task13_runtime_v3(repository)
    except Exception as exc:
        print(f"Task 14 rejected untrusted runtime: {exc}", file=sys.stderr)
        return EXIT_UNTRUSTED_RUNTIME
    if runtime.runtime_revision != args.source_revision:
        print("Task 14 rejected untrusted runtime: source revision mismatch", file=sys.stderr)
        return EXIT_UNTRUSTED_RUNTIME
    try:
        loaded = load_task14_sources_v1(
            Task14SourcePathsV1(
                core_root=Path(args.core_root),
                evidence_root=Path(args.evidence_root),
                task13_root=Path(args.task13_root),
                task13_audit_path=Path(args.task13_audit),
                repository_root=Path(repository),
                remote_task13_staging_path=args.remote_task13_staging,
            )
        )
        result = publish_task14_review_v1(
            loaded,
            review_id=args.review_id,
            trusted_source_revision=runtime.runtime_revision,
            output_root=Path(args.output_root),
        )
    except FileExistsError as exc:
        print(f"Task 14 publication rejected: {exc}", file=sys.stderr)
        return EXIT_PUBLICATION
    except RuntimeError as exc:
        if "source" in str(exc).lower() or "snapshot" in str(exc).lower():
            print(f"Task 14 stale source rejected: {exc}", file=sys.stderr)
            return EXIT_STALE_SOURCE
        print(f"Task 14 review rejected: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"Task 14 publication failed: {exc}", file=sys.stderr)
        return EXIT_PUBLICATION
    except (TypeError, ValueError) as exc:
        print(f"Task 14 review rejected: {exc}", file=sys.stderr)
        return EXIT_USAGE

    decision = "FINAL_APPROVED" if result.final_approved else "NOT_APPROVED"
    print(f"decision={decision}")
    print(f"core_final_root_index_sha256={result.index_sha256}")
    print(f"core_final_attestation_sha256={result.attestation_sha256}")
    print(f"output_root={result.output_root}")
    return EXIT_APPROVED if result.final_approved else EXIT_NOT_APPROVED


if __name__ == "__main__":
    raise SystemExit(main())
