from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from mub.vnext.contracts import ArtifactRef, MemUpdateTask, TaskManifest
from mub.vnext.generation.artifacts import (
    InMemoryPilotArtifact,
    PilotArtifactBundle,
)
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.splits import SplitBalanceReport
from mub.vnext.io import canonical_json_bytes, read_models
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation import ValidationReport, validate_splits


_ARTIFACT_NAMES: Final = (
    "tasks.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "validation_report.json",
)
_ARTIFACT_MODELS: Final = (
    None,
    PilotConfig,
    SplitBalanceReport,
    TaskManifest,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class PublishedPilotBundle:
    """Immutable evidence of a successful five-artifact publication."""

    output_dir: Path
    artifact_paths: tuple[Path, ...]
    artifact_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path")
        if type(self.artifact_paths) is not tuple:
            raise TypeError("artifact_paths must be an immutable tuple")
        if type(self.artifact_refs) is not tuple:
            raise TypeError("artifact_refs must be an immutable tuple")
        if len(self.artifact_paths) != len(_ARTIFACT_NAMES):
            raise ValueError("publication result must contain exactly five paths")
        if len(self.artifact_refs) != len(_ARTIFACT_NAMES):
            raise ValueError("publication result must contain exactly five refs")
        expected_paths = tuple(self.output_dir / name for name in _ARTIFACT_NAMES)
        if self.artifact_paths != expected_paths:
            raise ValueError("publication result paths are not canonical")
        if tuple(ref.path for ref in self.artifact_refs) != _ARTIFACT_NAMES:
            raise ValueError("publication result refs are not canonical")


def publish_pilot_artifact_bundle(
    bundle: PilotArtifactBundle,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> PublishedPilotBundle:
    """Publish a validated in-memory pilot bundle as one atomic artifact set."""
    if type(bundle) is not PilotArtifactBundle:
        raise TypeError("bundle must be a real PilotArtifactBundle")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")

    _validate_bundle_contract(bundle)
    artifacts = bundle.artifacts
    destinations = tuple(output_dir / artifact.path for artifact in artifacts)
    payloads = {
        destination: artifact.content
        for artifact, destination in zip(artifacts, destinations, strict=True)
    }
    validators = {
        destination: _stage_validator(bundle, artifact)
        for destination, artifact in zip(destinations, artifacts, strict=True)
    }
    publish_files_atomically(
        payloads,
        overwrite=overwrite,
        source_paths=(),
        validators=validators,
    )
    return PublishedPilotBundle(
        output_dir=output_dir,
        artifact_paths=destinations,
        artifact_refs=tuple(artifact.ref for artifact in artifacts),
    )


def _validate_bundle_contract(bundle: PilotArtifactBundle) -> None:
    artifacts = bundle.artifacts
    if type(artifacts) is not tuple or len(artifacts) != len(_ARTIFACT_NAMES):
        raise ValueError("bundle must contain exactly five canonical artifacts")
    expected_records = (None, 1, 1, 1, 1)
    for index, (artifact, name, record_count) in enumerate(
        zip(artifacts, _ARTIFACT_NAMES, expected_records, strict=True)
    ):
        if type(artifact) is not InMemoryPilotArtifact:
            raise TypeError(f"artifact {name} must be an InMemoryPilotArtifact")
        expected_media_type = "application/x-ndjson" if index == 0 else "application/json"
        if artifact.path != name or artifact.media_type != expected_media_type:
            raise ValueError(f"artifact {name} has non-canonical metadata")
        if record_count is not None and artifact.record_count != record_count:
            raise ValueError(f"artifact {name} has non-canonical record count")
        if hashlib.sha256(artifact.content).hexdigest() != artifact.ref.sha256:
            raise ValueError(f"artifact {name} hash does not match its content")

    typed_payloads = (
        canonical_json_bytes(bundle.resolved_config),
        canonical_json_bytes(bundle.split_balance_report),
        canonical_json_bytes(bundle.task_manifest),
        canonical_json_bytes(bundle.validation_report),
    )
    if tuple(artifact.content for artifact in artifacts[1:]) != typed_payloads:
        raise ValueError("typed bundle records disagree with canonical artifact bytes")
    if not bundle.validation_report.valid or bundle.validation_report.issues:
        raise ValueError("validation report must be valid with zero issues")
    if bundle.task_manifest.task_file_paths_and_hashes != (artifacts[0].ref,):
        raise ValueError("task manifest does not bind the exact task artifact")
    if bundle.task_manifest.generation_configs_and_hashes != (artifacts[1].ref,):
        raise ValueError("task manifest does not bind the exact config artifact")


def _stage_validator(
    bundle: PilotArtifactBundle,
    artifact: InMemoryPilotArtifact,
) -> Callable[[Path], None]:
    if artifact.path == "tasks.jsonl":
        return lambda staged: _validate_tasks_stage(bundle, artifact, staged)
    model_type = _ARTIFACT_MODELS[_ARTIFACT_NAMES.index(artifact.path)]
    assert model_type is not None
    return lambda staged: _validate_json_stage(
        staged, artifact, model_type, getattr(bundle, _typed_field(artifact.path))
    )


def _typed_field(path: str) -> str:
    return {
        "generation_config.json": "resolved_config",
        "split_balance.json": "split_balance_report",
        "task_manifest.json": "task_manifest",
        "validation_report.json": "validation_report",
    }[path]


def _validate_stage_bytes(staged: Path, artifact: InMemoryPilotArtifact) -> bytes:
    try:
        content = staged.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read staged artifact {artifact.path}") from exc
    if content != artifact.content:
        raise ValueError(f"staged artifact bytes disagree with bundle: {artifact.path}")
    digest = hashlib.sha256(content).hexdigest()
    if digest != artifact.ref.sha256:
        raise ValueError(f"staged artifact hash disagrees with bundle: {artifact.path}")
    return content


def _validate_json_stage(
    staged: Path,
    artifact: InMemoryPilotArtifact,
    model_type: type,
    expected_model,
) -> None:
    content = _validate_stage_bytes(staged, artifact)
    try:
        parsed = model_type.model_validate_json(content)
    except Exception as exc:
        raise ValueError(f"staged artifact is not valid canonical JSON: {artifact.path}") from exc
    try:
        canonical = canonical_json_bytes(parsed)
    except Exception as exc:
        raise ValueError(f"staged artifact cannot be canonically serialized: {artifact.path}") from exc
    if canonical != content:
        raise ValueError(f"staged artifact is not canonical JSON: {artifact.path}")
    if canonical_json_bytes(expected_model) != content:
        raise ValueError(f"staged artifact record disagrees with bundle: {artifact.path}")


def _validate_tasks_stage(
    bundle: PilotArtifactBundle,
    artifact: InMemoryPilotArtifact,
    staged: Path,
) -> None:
    content = _validate_stage_bytes(staged, artifact)
    try:
        tasks = tuple(read_models(staged, MemUpdateTask, id_field="task_id"))
    except Exception as exc:
        raise ValueError("staged task JSONL is not valid canonical task data") from exc
    if len(tasks) != artifact.record_count:
        raise ValueError("staged task JSONL record count disagrees with bundle")
    canonical = b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)
    if canonical != content:
        raise ValueError("staged task JSONL is not canonical")
    manifest = bundle.task_manifest
    if manifest.task_file_paths_and_hashes != (artifact.ref,):
        raise ValueError("staged task JSONL is not manifest-backed")
    report = validate_splits(tasks, task_manifest=manifest)
    if report != bundle.validation_report:
        raise ValueError("staged task JSONL disagrees with the manifest validation report")


__all__ = ["PublishedPilotBundle", "publish_pilot_artifact_bundle"]
