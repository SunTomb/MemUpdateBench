from __future__ import annotations

import argparse
from pathlib import Path
import json
import os

from mub.vnext.generation.post_core_audit import select_main_track_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a stratified audit sample for main_track_v1.")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = select_main_track_audit(args.candidate_root)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(selection, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with args.output.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"output": str(args.output), "cores": selection["selected_semantic_core_count"], "tasks": selection["selected_task_count"], "review_status": selection["review_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
