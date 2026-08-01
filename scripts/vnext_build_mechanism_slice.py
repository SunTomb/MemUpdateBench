from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.generation.config import load_pilot_config
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.io.jsonl import read_models
from mub.vnext.mechanisms.matrix import build_mechanism_slice


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def build_outputs(tasks_path: Path, config_path: Path, output_dir: Path) -> tuple[Path, Path]:
    config = load_pilot_config(config_path)
    tasks = tuple(read_models(tasks_path, MemUpdateTask, id_field="task_id"))
    result = build_mechanism_slice(tasks, config)
    context_bytes = b"".join(canonical_json_bytes(record) + b"\n" for record in result.records)
    manifest_bytes = _json_bytes(result.manifest)
    context_path = output_dir / "contexts.jsonl"
    manifest_path = output_dir / "condition_manifest.json"
    publish_files_atomically(
        {context_path: context_bytes, manifest_path: manifest_bytes},
        overwrite=True,
        source_paths=(tasks_path, config_path),
    )
    return context_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the vNext paired mechanism smoke slice")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    context_path, manifest_path = build_outputs(args.tasks, args.config, args.output_dir)
    print(json.dumps({"contexts": str(context_path), "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
