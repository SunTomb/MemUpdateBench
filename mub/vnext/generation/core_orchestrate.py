from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.core_release import validate_core_release


@dataclass(frozen=True, slots=True)
class StagedCoreCandidate:
    release_dir: Path
    semantic_core_count: int
    task_count: int
    split_core_counts: dict[str, int]
    split_task_counts: dict[str, int]
    hard_suite_core_count: int
    hard_suite_task_count: int


def stage_core_candidate(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    code_revision: str,
    cores_per_family: int | None = None,
) -> StagedCoreCandidate:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_core_config(config_path)
    immutable = (config_path.resolve().parents[2] / config.output.release_dir).resolve()
    if output_dir.resolve() == immutable:
        raise ValueError("Core candidates must be staged outside the immutable release root")
    if output_dir.exists():
        raise FileExistsError(f"candidate output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot = compile_core_snapshot(
        config,
        cores_per_family=cores_per_family,
        code_revision=code_revision,
    )
    bundle = build_core_artifact_bundle(snapshot, config)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for artifact in bundle.artifacts:
            (temporary / artifact.path).write_bytes(artifact.content)
        validate_core_release(temporary, expected_full=cores_per_family is None)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return StagedCoreCandidate(
        release_dir=output_dir,
        semantic_core_count=len(snapshot.semantic_cores),
        task_count=len(snapshot.tasks),
        split_core_counts=dict(snapshot.core_counts),
        split_task_counts=dict(snapshot.task_counts),
        hard_suite_core_count=len(bundle.hard_suite.semantic_core_ids),
        hard_suite_task_count=len(bundle.hard_suite.task_ids),
    )


__all__ = ["StagedCoreCandidate", "stage_core_candidate"]
