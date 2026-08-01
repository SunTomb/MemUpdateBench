from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.adapters import build_adapter
from mub.vnext.adapters.corrupted import build_corrupted_adapter
from mub.vnext.contracts import MemUpdateTask
from mub.vnext.io.jsonl import read_models
from mub.vnext.runtime import RuntimeConfig, run_tasks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MemUpdateBench vNext Pilot runtime without API calls")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--adapter", required=True, help="reference, raw_add, heuristic_crud, exact_crud, or control/*")
    parser.add_argument("--retrieval-policy", choices=("normal_topk", "latest_per_object"), default="normal_topk")
    parser.add_argument("--answer-mode", choices=("slot_direct", "slot_prompt", "native_answer"), default="slot_direct")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _manifest_hash(path: Path | None, tasks: list[MemUpdateTask]) -> str:
    if path is not None:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256("".join(task.task_id for task in tasks).encode("utf-8")).hexdigest()


def _make_factory(adapter_id: str, retrieval_policy: str):
    def factory(task: MemUpdateTask) -> Any:
        if adapter_id.startswith("control/"):
            return build_corrupted_adapter(adapter_id, task=task, retrieval_policy=retrieval_policy)
        if adapter_id == "reference":
            from mub.vnext.adapters.reference import ReferenceAdapter
            return ReferenceAdapter(task, retrieval_policy=retrieval_policy)
        if adapter_id == "heuristic_crud":
            # No network/model loading is permitted; the adapter records this as unsupported.
            return build_adapter(adapter_id, retrieval_policy=retrieval_policy)
        return build_adapter(adapter_id, retrieval_policy=retrieval_policy)
    return factory


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tasks = list(read_models(args.tasks, MemUpdateTask, id_field="task_id"))
    config = RuntimeConfig(
        run_id="vnext-pilot-run",
        retrieval_policy=args.retrieval_policy,
        answer_mode=args.answer_mode,
    )
    run_tasks(
        tasks,
        adapter_factory=_make_factory(args.adapter, args.retrieval_policy),
        run_config=config,
        output_dir=args.output_dir,
        task_manifest_hash=_manifest_hash(args.task_manifest, tasks),
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
