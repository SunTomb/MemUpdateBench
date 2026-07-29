from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.generation import build_pilot, load_pilot_config


class _ArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError from None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _SafeArgumentParser(description="Generate the MemUpdateBench vNext Pilot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_code_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise RuntimeError("revision unavailable")
    revision = completed.stdout.strip()
    if not revision:
        raise RuntimeError("revision unavailable")
    return revision


def _success_payload(config, published, code_revision: str) -> dict[str, object]:
    return {
        "release_id": config.release_id,
        "code_revision": code_revision,
        "task_count": config.total_tasks,
        "split_counts": dict(config.expected_split_tasks),
        "output_dir": str(published.output_dir),
        "artifact_refs": [
            {
                "path": ref.path,
                "sha256": ref.sha256,
                "media_type": ref.media_type,
                "record_count": ref.record_count,
            }
            for ref in published.artifact_refs
        ],
    }


def _canonical_json_line(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _write_error(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except _ArgumentError:
        _write_error("invalid command-line arguments")
        return 2

    try:
        code_revision = _resolve_code_revision()
    except Exception:
        _write_error("could not resolve code revision")
        return 2

    try:
        config = load_pilot_config(args.config)
        published = build_pilot(
            config,
            args.output_dir,
            code_revision=code_revision,
            overwrite=args.overwrite,
        )
        line = _canonical_json_line(
            _success_payload(config, published, code_revision)
        )
    except Exception:
        _write_error("pilot generation failed")
        return 2

    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
