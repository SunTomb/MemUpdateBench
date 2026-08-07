from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
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
    _cleanup_path: Path | None = field(default=None, repr=False)
    _destination_identity: tuple[int, int] | None = field(default=None, repr=False)
    _artifact_hashes: tuple[tuple[str, str], ...] | None = field(
        default=None,
        repr=False,
    )

    def remove_if_unchanged(self) -> bool:
        if (
            self._cleanup_path is None
            or self._destination_identity is None
            or self._artifact_hashes is None
        ):
            return False
        return _remove_tree_if_verified(
            self._cleanup_path,
            self._destination_identity,
            self._artifact_hashes,
        )


@dataclass(frozen=True, slots=True)
class _StagingPathBinding:
    requested_output: Path
    parent_path: Path
    resolved_parent: Path
    resolved_output: Path
    parent_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _VerifiedTemporaryTree:
    identity: tuple[int, int]
    artifact_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _QuarantinedTree:
    original_path: Path
    holder_path: Path
    captured_path: Path


def _resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _assert_no_lexical_reparse_points(path: Path) -> None:
    candidates = tuple(reversed(path.parents)) + (path,)
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        stat = candidate.stat(follow_symlinks=False)
        if candidate.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
            raise ValueError(
                "Core candidate staging path contains a symlink or junction"
            )


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
    _assert_no_lexical_reparse_points(requested.parent)
    _assert_outside_immutable(_resolve_path(requested))
    if requested.exists():
        raise FileExistsError(f"candidate output already exists: {output_dir}")
    _assert_outside_immutable(_resolve_path(requested.parent) / requested.name)
    requested.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_lexical_reparse_points(requested.parent)
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


def _recheck_staging_path(
    binding: _StagingPathBinding,
    *,
    expect_output: bool = False,
) -> Path:
    _assert_no_lexical_reparse_points(binding.parent_path)
    current_parent = _resolve_path(binding.parent_path)
    current_output = _resolve_path(binding.requested_output)
    if (
        current_parent != binding.resolved_parent
        or current_output != binding.resolved_output
        or _path_identity(current_parent) != binding.parent_identity
    ):
        raise ValueError("Core candidate staging parent changed after path binding")
    _assert_outside_immutable(current_output)
    output_exists = binding.requested_output.exists()
    if expect_output and not output_exists:
        raise ValueError("Core candidate output disappeared after final transfer")
    if not expect_output and output_exists:
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


def _artifact_hashes(bundle) -> tuple[tuple[str, str], ...]:
    return tuple(
        (artifact.path, artifact.ref.sha256)
        for artifact in bundle.artifacts
    )


def _tree_matches_hashes(
    root: Path,
    artifact_hashes: tuple[tuple[str, str], ...],
) -> bool:
    expected_names = {name for name, _ in artifact_hashes}
    actual_entries = tuple(root.iterdir())
    actual_names = {path.name for path in actual_entries}
    if actual_names != expected_names or any(not path.is_file() for path in actual_entries):
        return False
    return all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash
        for name, expected_hash in artifact_hashes
    )


def _bind_verified_temporary(temporary: Path, bundle) -> _VerifiedTemporaryTree:
    _verify_staged_bundle(temporary, bundle)
    hashes = _artifact_hashes(bundle)
    if not _tree_matches_hashes(temporary, hashes):
        raise ValueError("Core candidate verified temporary content changed")
    return _VerifiedTemporaryTree(
        identity=_path_identity(temporary),
        artifact_hashes=hashes,
    )


def _recheck_verified_tree(
    root: Path,
    verified: _VerifiedTemporaryTree,
) -> None:
    try:
        identity_matches = _path_identity(root) == verified.identity
        content_matches = _tree_matches_hashes(root, verified.artifact_hashes)
    except (FileNotFoundError, OSError, ValueError):
        identity_matches = False
        content_matches = False
    if not identity_matches or not content_matches:
        raise ValueError("installed object is not the verified temporary tree")


def _quarantine_tree(path: Path) -> _QuarantinedTree | None:
    try:
        holder = Path(
            tempfile.mkdtemp(
                prefix=f".{path.name}.cleanup-",
                dir=path.parent,
            )
        )
    except OSError:
        return None
    captured = holder / "tree"
    try:
        os.replace(path, captured)
    except OSError:
        try:
            holder.rmdir()
        except OSError:
            pass
        return None
    return _QuarantinedTree(
        original_path=path,
        holder_path=holder,
        captured_path=captured,
    )


def _restore_quarantined_tree(quarantined: _QuarantinedTree) -> None:
    if os.path.lexists(quarantined.original_path):
        return
    try:
        os.rename(
            quarantined.captured_path,
            quarantined.original_path,
        )
    except OSError:
        return
    try:
        quarantined.holder_path.rmdir()
    except OSError:
        pass


def _delete_quarantined_tree(quarantined: _QuarantinedTree) -> bool:
    try:
        shutil.rmtree(quarantined.captured_path)
    except OSError:
        return False
    try:
        quarantined.holder_path.rmdir()
    except OSError:
        pass
    return True


def _remove_tree_if_verified(
    path: Path,
    expected_identity: tuple[int, int],
    artifact_hashes: tuple[tuple[str, str], ...],
) -> bool:
    quarantined = _quarantine_tree(path)
    if quarantined is None:
        return False
    try:
        identity_matches = (
            _path_identity(quarantined.captured_path)
            == expected_identity
        )
        content_matches = _tree_matches_hashes(
            quarantined.captured_path,
            artifact_hashes,
        )
    except (FileNotFoundError, OSError, ValueError):
        identity_matches = False
        content_matches = False
    if not identity_matches or not content_matches:
        _restore_quarantined_tree(quarantined)
        return False
    return _delete_quarantined_tree(quarantined)


def _remove_tree_if_identity_matches(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    quarantined = _quarantine_tree(path)
    if quarantined is None:
        return
    try:
        identity_matches = (
            _path_identity(quarantined.captured_path)
            == expected_identity
        )
    except (FileNotFoundError, OSError, ValueError):
        identity_matches = False
    if not identity_matches:
        _restore_quarantined_tree(quarantined)
        return
    _delete_quarantined_tree(quarantined)


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
    temporary_identity = _path_identity(temporary)
    verified = None
    destination = binding.resolved_output
    try:
        _recheck_staging_path(binding)
        if temporary.parent != binding.resolved_parent:
            raise ValueError("Core candidate temporary directory escaped bound parent")
        for artifact in bundle.artifacts:
            _recheck_staging_path(binding)
            (temporary / artifact.path).write_bytes(artifact.content)
        _recheck_staging_path(binding)
        verified = _bind_verified_temporary(temporary, bundle)
        destination = _recheck_staging_path(binding)
        _recheck_verified_tree(temporary, verified)
        os.replace(temporary, destination)
        _recheck_staging_path(binding, expect_output=True)
        _recheck_verified_tree(destination, verified)
    except Exception:
        if verified is not None:
            _remove_tree_if_verified(
                destination,
                verified.identity,
                verified.artifact_hashes,
            )
        _remove_tree_if_identity_matches(temporary, temporary_identity)
        raise
    assert verified is not None
    return StagedCoreCandidate(
        release_dir=output_dir,
        semantic_core_count=len(snapshot.semantic_cores),
        task_count=len(snapshot.tasks),
        split_core_counts=dict(snapshot.core_counts),
        split_task_counts=dict(snapshot.task_counts),
        hard_suite_core_count=len(bundle.hard_suite.semantic_core_ids),
        hard_suite_task_count=len(bundle.hard_suite.task_ids),
        _cleanup_path=destination,
        _destination_identity=verified.identity,
        _artifact_hashes=verified.artifact_hashes,
    )


__all__ = ["StagedCoreCandidate", "stage_core_candidate"]
