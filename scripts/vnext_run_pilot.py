from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mub.vnext.adapters import build_adapter
from mub.vnext.adapters.corrupted import build_corrupted_adapter
from mub.vnext.contracts import MemUpdateTask, TaskManifest
from mub.vnext.io.jsonl import read_models
from mub.vnext.runtime import RuntimeConfig, run_tasks
from mub.vnext.version import COMPILER_VERSION, PROFILE_VERSION, SCHEMA_VERSION


_DEFAULT_ENCODER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_GIT_CONTEXT_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_QUARANTINE_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
        "GIT_REPLACE_REF_BASE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_PREFIX",
        "GIT_INTERNAL_SUPER_PREFIX",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MemUpdateBench vNext Pilot runtime without API calls")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True, help="reference, raw_add, heuristic_crud, exact_crud, or control/*")
    parser.add_argument("--retrieval-policy", choices=("normal_topk", "latest_per_object"), default="normal_topk")
    parser.add_argument("--answer-mode", choices=("slot_direct", "slot_prompt", "native_answer"), default="slot_direct")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument(
        "--encoder-model-id",
        default=_DEFAULT_ENCODER_MODEL_ID,
    )
    parser.add_argument("--encoder-revision")
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _sanitized_git_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in _GIT_CONTEXT_ENV
    }


def _git(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
        env=_sanitized_git_environment(),
    )


def _resolve_code_revision() -> str:
    completed = _git(["git", "rev-parse", "HEAD"])
    revision = completed.stdout.strip() if isinstance(completed.stdout, str) else ""
    if completed.returncode != 0 or not revision:
        raise RuntimeError("code revision unavailable")
    return revision


def _tracked_tree_is_dirty() -> bool:
    completed = _git(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    )
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise RuntimeError("tracked state unavailable")
    return bool(completed.stdout.strip())


def _validate_encoder_arguments(args: argparse.Namespace) -> None:
    if args.adapter != "heuristic_crud":
        if (
            args.encoder_checkpoint is not None
            or args.encoder_revision is not None
            or args.encoder_model_id != _DEFAULT_ENCODER_MODEL_ID
            or args.encoder_device != "cpu"
        ):
            raise ValueError("encoder options are only valid with heuristic_crud")
        return
    if args.encoder_checkpoint is None:
        raise ValueError("heuristic_crud requires --encoder-checkpoint")
    if not args.encoder_checkpoint.is_dir():
        raise ValueError("encoder-checkpoint must be an existing directory")
    if not isinstance(args.encoder_revision, str) or not args.encoder_revision.strip():
        raise ValueError("heuristic_crud requires --encoder-revision")
    if not isinstance(args.encoder_model_id, str) or not args.encoder_model_id.strip():
        raise ValueError("encoder-model-id must be nonblank")
    if args.encoder_device != "cpu" and not (
        args.encoder_device == "cuda"
        or args.encoder_device.startswith("cuda:")
        and args.encoder_device[5:].isdigit()
    ):
        raise ValueError("encoder-device must be cpu, cuda, or cuda:N")


def _load_offline_encoder(checkpoint: Path, *, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(checkpoint.resolve()),
        device=device,
        local_files_only=True,
    )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_factory(
    adapter_id: str,
    retrieval_policy: str,
    *,
    encoder=None,
    encoder_model: str = _DEFAULT_ENCODER_MODEL_ID,
    encoder_revision: str = "unverified",
):
    def factory(task: MemUpdateTask) -> Any:
        if adapter_id.startswith("control/"):
            return build_corrupted_adapter(adapter_id, task=task, retrieval_policy=retrieval_policy)
        if adapter_id == "reference":
            from mub.vnext.adapters.reference import ReferenceAdapter
            return ReferenceAdapter(task, retrieval_policy=retrieval_policy)
        if adapter_id == "heuristic_crud":
            from mub.vnext.adapters.heuristic_crud import HeuristicCrudAdapter

            return HeuristicCrudAdapter(
                encoder=encoder,
                encoder_model=encoder_model,
                encoder_revision=encoder_revision,
                retrieval_policy=retrieval_policy,
            )
        return build_adapter(adapter_id, retrieval_policy=retrieval_policy)
    return factory


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_encoder_arguments(args)
        code_revision = _resolve_code_revision()
        if _tracked_tree_is_dirty():
            raise RuntimeError("tracked source tree is dirty")

        tasks = list(read_models(args.tasks, MemUpdateTask, id_field="task_id"))
        if not tasks:
            raise ValueError("task file must contain at least one task")
        task_manifest = TaskManifest.model_validate_json(
            args.task_manifest.read_bytes()
        )
        task_digest = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
        if not any(
            ref.sha256 == task_digest
            and ref.record_count == len(tasks)
            for ref in task_manifest.task_file_paths_and_hashes
        ):
            raise ValueError("task file is not authenticated by task manifest")

        encoder = None
        if args.adapter == "heuristic_crud":
            encoder = _load_offline_encoder(
                args.encoder_checkpoint,
                device=args.encoder_device,
            )
        factory = _make_factory(
            args.adapter,
            args.retrieval_policy,
            encoder=encoder,
            encoder_model=args.encoder_model_id,
            encoder_revision=args.encoder_revision or "unverified",
        )
        if args.adapter == "heuristic_crud":
            probe = factory(tasks[0])
            reset = probe.reset("heuristic-capability-probe", {})
            probe.close()
            if not reset.success:
                raise RuntimeError("heuristic encoder verification failed")

        config = RuntimeConfig(
            run_id="vnext-pilot-run",
            retrieval_policy=args.retrieval_policy,
            answer_mode=args.answer_mode,
            code_revision=code_revision,
            dirty_state=False,
            compiler_version=COMPILER_VERSION,
            profile_version=PROFILE_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        run_tasks(
            tasks,
            adapter_factory=factory,
            run_config=config,
            output_dir=args.output_dir,
            task_manifest_hash=_manifest_hash(args.task_manifest),
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"vNext Pilot runtime failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
