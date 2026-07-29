from __future__ import annotations

from pathlib import Path

from mub.vnext.generation.artifacts import build_pilot_artifact_bundle
from mub.vnext.generation.build import compile_pilot_tasks
from mub.vnext.generation.config import PilotConfig
from mub.vnext.generation.publish import (
    PublishedPilotBundle,
    publish_pilot_artifact_bundle,
)


def build_pilot(
    config: PilotConfig,
    output_dir: Path,
    *,
    code_revision: str,
    overwrite: bool = False,
) -> PublishedPilotBundle:
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if type(code_revision) is not str:
        raise TypeError("code_revision must be a string")
    if not code_revision.strip():
        raise ValueError("code_revision must not be blank")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")

    compiled = compile_pilot_tasks(config, code_revision=code_revision)
    bundle = build_pilot_artifact_bundle(compiled, config)
    return publish_pilot_artifact_bundle(
        bundle,
        output_dir,
        overwrite=overwrite,
    )


__all__ = ["build_pilot"]
