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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMMUTABLE_RELEASE_ROOTS = (
    (_PROJECT_ROOT / "data" / "vnext" / "core").resolve(),
    (_PROJECT_ROOT / "data" / "vnext" / "pilot").resolve(),
)


@dataclass(frozen=True, slots=True)
class StagedCoreCandidate:
    release_dir: Path
    semantic_core_count: int
    task_count: int
    split_core_counts: dict[str, int]
    split_task_counts: dict[str, int]
    hard_suite_core_count: int
    hard_suite_task_count: int


@dataclass(frozen=True, slots=True)
class _StagingPathBinding:
    requested_output: Path
    parent_path: Path
    resolved_parent: Path
    resolved_output: Path
    parent_identity: tuple[int, int]


def _resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _path_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    if path.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
        raise ValueError("Core candidate staging parent cannot be a symlink or junction")
    return stat.st_dev, stat.st_ino


def _assert_outside_immutable(resolved_output: Path) -> None:
    if any(
        resolved_output == immutable or immutable in resolved_output.parents
        for immutable in _IMMUTABLE_RELEASE_ROOTS
    ):
        raise ValueError("Core candidates must be staged outside immutable release roots")


def _bind_staging_path(output_dir: Path) -> _StagingPathBinding:
    requested = Path(os.path.abspath(output_dir))
    _assert_outside_immutable(_resolve_path(requested))
    if requested.exists():
        raise FileExistsError(f"candidate output already exists: {output_dir}")
    _assert_outside_immutable(_resolve_path(requested.parent) / requested.name)
    requested.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = _resolve_path(requested.parent)
    resolved_output = resolved_parent / requested.name
    _assert_outside_immutable(resolved_output)
    return _StagingPathBinding(
        requested_output=requested,
        parent_path=requested.parent,
        resolved_parent=resolved_parent,
        resolved_output=resolved_output,
        parent_identity=_path_identity(resolved_parent),
    )


def _recheck_staging_path(binding: _StagingPathBinding) -> Path:
    current_parent = _resolve_path(binding.parent_path)
    current_output = _resolve_path(binding.requested_output)
    if (
        current_parent != binding.resolved_parent
        or current_output != binding.resolved_output
        or _path_identity(current_parent) != binding.parent_identity
    ):
        raise ValueError("Core candidate staging parent changed after path binding")
    _assert_outside_immutable(current_output)
    if binding.requested_output.exists():
        raise FileExistsError(
            f"candidate output appeared during staging: {binding.requested_output}"
        )
    return binding.resolved_output


def _verify_staged_bundle(temporary: Path, bundle) -> None:
    if {path.name for path in temporary.iterdir() if path.is_file()} != {
        artifact.path for artifact in bundle.artifacts
    }:
        raise ValueError("staged Core artifact set is incomplete")
    for artifact in bundle.artifacts:
        if (temporary / artifact.path).read_bytes() != artifact.content:
            raise ValueError(f"staged Core artifact bytes differ: {artifact.path}")


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
    binding = _bind_staging_path(output_dir)
    snapshot = compile_core_snapshot(
        config,
        cores_per_family=cores_per_family,
        code_revision=code_revision,
    )
    _recheck_staging_path(binding)
    bundle = build_core_artifact_bundle(snapshot, config)
    _recheck_staging_path(binding)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{binding.resolved_output.name}.staging-",
            dir=binding.resolved_parent,
        )
    )
    try:
        _recheck_staging_path(binding)
        if temporary.parent != binding.resolved_parent:
            raise ValueError("Core candidate temporary directory escaped bound parent")
        for artifact in bundle.artifacts:
            _recheck_staging_path(binding)
            (temporary / artifact.path).write_bytes(artifact.content)
        _recheck_staging_path(binding)
        _verify_staged_bundle(temporary, bundle)
        destination = _recheck_staging_path(binding)
        os.replace(temporary, destination)
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
