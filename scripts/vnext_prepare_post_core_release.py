from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mub.vnext.post_core.release_v1 import (
    EXIT_BLOCKED,
    EXIT_PUBLICATION,
    EXIT_STALE_SOURCE,
    EXIT_SUCCESS_WITH_PENDING,
    EXIT_UNTRUSTED_RUNTIME,
    EXIT_USAGE,
    PostCoreReleaseError,
    StaleSourceError,
    UnsafePathError,
    load_post_core_config_v1,
    load_post_core_registry_v1,
    publish_post_core_release_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and atomically publish the offline MemUpdateBench post-Core Phase 0 release.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="frozen Phase 0 release config JSON")
    parser.add_argument("--registry", help="optional canonical model registry JSON")
    parser.add_argument("--core-manifest", required=True, help="authenticated immutable Core source manifest")
    parser.add_argument("--task14-index", required=True, help="external Core Task 14 root index")
    parser.add_argument("--provenance", help="optional canonical provenance JSONL input")
    parser.add_argument("--output-root", required=True, help="new absent output directory")
    parser.add_argument("--execute", action="store_true", help="validate/build/publish metadata only")
    return parser


def _summary(result) -> dict[str, object]:
    return {
        "status": "SUCCESS_WITH_PENDING" if result.pending_count else "SUCCESS",
        "release_id": result.release_id,
        "output_root": str(result.output_root) if result.output_root is not None else None,
        "index_sha256": result.index_sha256,
        "pending_count": result.pending_count,
        "provider_calls": result.provider_calls,
        "model_loads": result.model_loads,
        "network_calls": result.network_calls,
        "executable_call_count": result.executable_call_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if not args.execute:
        print("post-Core preparation requires explicit --execute", file=sys.stderr)
        return EXIT_USAGE
    try:
        config = load_post_core_config_v1(Path(args.config))
        registry = load_post_core_registry_v1(Path(args.registry), config) if args.registry else None
        result = publish_post_core_release_v1(
            config,
            Path(args.core_manifest),
            Path(args.task14_index),
            Path(args.output_root),
            registry=registry,
            provenance_path=Path(args.provenance) if args.provenance else None,
        )
    except StaleSourceError:
        print("post-Core stale source rejected: authenticated source mismatch", file=sys.stderr)
        return EXIT_STALE_SOURCE
    except FileExistsError:
        print("post-Core publication rejected: output root is unavailable", file=sys.stderr)
        return EXIT_PUBLICATION
    except UnsafePathError:
        print("post-Core publication rejected: unsafe path", file=sys.stderr)
        return EXIT_PUBLICATION
    except ValueError:
        print("post-Core contract/usage rejected: invalid untrusted contract input", file=sys.stderr)
        return EXIT_USAGE
    except PostCoreReleaseError:
        print("post-Core publication failed: publication invariant rejected", file=sys.stderr)
        return EXIT_PUBLICATION
    except OSError:
        print("post-Core publication failed: filesystem operation rejected", file=sys.stderr)
        return EXIT_PUBLICATION
    except Exception as exc:
        # Do not expose arbitrary runtime details as a success or as a secret-bearing summary.
        print(f"post-Core rejected untrusted runtime: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNTRUSTED_RUNTIME
    print(json.dumps(_summary(result), sort_keys=True, separators=(",", ":")))
    return EXIT_SUCCESS_WITH_PENDING if result.pending_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
