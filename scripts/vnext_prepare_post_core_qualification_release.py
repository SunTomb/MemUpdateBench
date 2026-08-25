from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mub.vnext.post_core.qualification_receipts_v1 import QualificationValidationReceiptV1
from mub.vnext.post_core.qualification_release_v1 import (
    CommittedQualificationReleaseError,
    NoReplacePrimitiveUnavailableError,
    QualificationReleaseError,
    UnsafeQualificationPathError,
    publish_qualification_release_v1,
)


EXIT_SUCCESS = 0
EXIT_BLOCKED = 10
EXIT_USAGE = 11
EXIT_STALE_SOURCE = 12
EXIT_PUBLICATION = 13
EXIT_UNTRUSTED_RUNTIME = 14


class _ArgumentUsageError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentUsageError


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Validate and publish offline MemUpdateBench post-Core qualification metadata.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-manifest", required=True)
    parser.add_argument("--task14-index", required=True)
    parser.add_argument("--phase0-index", required=True)
    parser.add_argument("--identity-evidence", required=True)
    parser.add_argument("--workflow-source", required=True)
    parser.add_argument("--handoff-source", required=True)
    parser.add_argument("--open-snapshot-closure-receipt", required=True)
    parser.add_argument("--open-snapshot-audit-receipt", required=True)
    parser.add_argument("--qwen-load-receipt", required=True)
    parser.add_argument("--provider-attestations", required=True)
    parser.add_argument("--runtime-receipts", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def _summary(result: object) -> dict[str, object]:
    receipt = QualificationValidationReceiptV1.model_validate_json(
        result.artifact_bytes["qualification_validation_receipt.json"]
    )
    return {
        "status": receipt.status,
        "release_id": receipt.release_id,
        "output_root": str(result.output_root),
        "index_sha256": result.index_sha256,
        "decision_counts": dict(receipt.decision_counts),
        "provider_calls_during_publication": receipt.provider_calls_during_publication,
        "model_loads_during_publication": receipt.model_loads_during_publication,
        "network_calls_during_publication": receipt.network_calls_during_publication,
        "credential_reads_during_publication": receipt.credential_reads_during_publication,
        "benchmark_generations": receipt.benchmark_generations,
    }


def _is_unsafe_path_rejection(exception: ValueError) -> bool:
    message = str(exception).lower()
    return "link" in message or "reparse" in message


def _is_stale_source_or_config(exception: ValueError) -> bool:
    message = str(exception).lower()
    return any(token in message for token in ("source", "snapshot", "hash", "config"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _ArgumentUsageError:
        print("qualification contract/usage rejected: invalid arguments", file=sys.stderr)
        return EXIT_USAGE
    except SystemExit as exception:
        return EXIT_SUCCESS if exception.code == 0 else EXIT_USAGE
    if not args.execute:
        print("qualification release requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = publish_qualification_release_v1(
            Path(args.output_root),
            Path(args.config),
            source_paths={
                "core_manifest": Path(args.core_manifest),
                "handoff_source": Path(args.handoff_source),
                "identity_evidence": Path(args.identity_evidence),
                "open_snapshot_audit_receipt": Path(args.open_snapshot_audit_receipt),
                "open_snapshot_closure_receipt": Path(args.open_snapshot_closure_receipt),
                "phase0_index": Path(args.phase0_index),
                "qwen_load_receipt": Path(args.qwen_load_receipt),
                "task14_index": Path(args.task14_index),
                "workflow_source": Path(args.workflow_source),
            },
            provider_attestations_path=Path(args.provider_attestations),
            runtime_receipts_path=Path(args.runtime_receipts),
        )
    except FileExistsError:
        print("qualification publication rejected: output root is unavailable", file=sys.stderr)
        return EXIT_PUBLICATION
    except (UnsafeQualificationPathError, NoReplacePrimitiveUnavailableError, CommittedQualificationReleaseError):
        print("qualification publication rejected: unsafe publication path", file=sys.stderr)
        return EXIT_PUBLICATION
    except QualificationReleaseError:
        print("qualification publication rejected: publication invariant", file=sys.stderr)
        return EXIT_PUBLICATION
    except ValueError as exception:
        if _is_unsafe_path_rejection(exception):
            print("qualification publication rejected: unsafe publication path", file=sys.stderr)
            return EXIT_PUBLICATION
        if _is_stale_source_or_config(exception):
            print("qualification stale source rejected: source/config mismatch", file=sys.stderr)
            return EXIT_STALE_SOURCE
        print("qualification contract/usage rejected: invalid source input", file=sys.stderr)
        return EXIT_USAGE
    except TypeError:
        print("qualification contract/usage rejected: invalid source input", file=sys.stderr)
        return EXIT_USAGE
    except OSError:
        print("qualification publication rejected: filesystem operation", file=sys.stderr)
        return EXIT_PUBLICATION
    except Exception as exception:
        print(
            f"qualification rejected untrusted runtime: {type(exception).__name__}",
            file=sys.stderr,
        )
        return EXIT_UNTRUSTED_RUNTIME
    print(json.dumps(_summary(result), sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
