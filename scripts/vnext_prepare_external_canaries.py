from __future__ import annotations

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mub.vnext.external.canaries_v3 import (
    authenticate_core_release,
    build_canary_set,
    publish_canary_set,
)


def _project_root() -> Path:
    return _PROJECT_ROOT


def _report_output(path: Path) -> None:
    payload = f"{path}\n".encode("utf-8")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(payload.decode("utf-8"))
        sys.stdout.flush()
        return
    stream.write(payload)
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive authenticated dev-only external-system canaries from Core v3."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=_project_root() / "data" / "vnext" / "core" / "v3",
        help="immutable authenticated Core task-release root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new caller-owned destination outside the Core release",
    )
    args = parser.parse_args(argv)
    try:
        release = authenticate_core_release(args.release_root)
        bundle = build_canary_set(release)
        output = publish_canary_set(bundle, args.output_root, release=release)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        _report_output(output)
    except (OSError, UnicodeError):
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
