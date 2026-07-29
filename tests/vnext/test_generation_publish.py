from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path

import pytest

import mub.vnext.io.atomic as atomic
from mub.vnext.generation import (
    PilotArtifactBundle,
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
    publish_pilot_artifact_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
ARTIFACT_NAMES = (
    "tasks.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "validation_report.json",
)


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def bundle(config) -> PilotArtifactBundle:
    compiled = compile_pilot_tasks(config, code_revision="publish-test-revision-a")
    return build_pilot_artifact_bundle(compiled, config)


@pytest.fixture(scope="module")
def replacement_bundle(config) -> PilotArtifactBundle:
    compiled = compile_pilot_tasks(config, code_revision="publish-test-revision-b")
    return build_pilot_artifact_bundle(compiled, config)


def _snapshot(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in ARTIFACT_NAMES}


def test_publish_writes_exact_bundle_and_returns_immutable_ordered_result(
    tmp_path, bundle
) -> None:
    output_dir = tmp_path / "pilot"

    published = publish_pilot_artifact_bundle(bundle, output_dir)

    assert published.output_dir == output_dir
    assert published.artifact_paths == tuple(output_dir / name for name in ARTIFACT_NAMES)
    assert published.artifact_refs == tuple(artifact.ref for artifact in bundle.artifacts)
    assert tuple(path.name for path in published.artifact_paths) == ARTIFACT_NAMES
    assert tuple(ref.sha256 for ref in published.artifact_refs) == tuple(
        hashlib.sha256(artifact.content).hexdigest() for artifact in bundle.artifacts
    )
    assert _snapshot(output_dir) == {
        artifact.path: artifact.content for artifact in bundle.artifacts
    }
    with pytest.raises((FrozenInstanceError, AttributeError)):
        published.output_dir = tmp_path / "other"


def test_publish_requires_exact_public_input_types(tmp_path, bundle) -> None:
    with pytest.raises(TypeError, match="PilotArtifactBundle"):
        publish_pilot_artifact_bundle(object(), tmp_path / "pilot")
    with pytest.raises(TypeError, match="Path"):
        publish_pilot_artifact_bundle(bundle, str(tmp_path / "pilot"))
    with pytest.raises(TypeError, match="bool"):
        publish_pilot_artifact_bundle(bundle, tmp_path / "pilot", overwrite=1)


def test_publish_rejects_output_file_as_directory(tmp_path, bundle) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_bytes(b"existing")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        publish_pilot_artifact_bundle(bundle, output_path)

    assert output_path.read_bytes() == b"existing"


    output_dir = tmp_path / "pilot"
    publish_pilot_artifact_bundle(bundle, output_dir)
    before = _snapshot(output_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        publish_pilot_artifact_bundle(bundle, output_dir)

    assert _snapshot(output_dir) == before


def test_overwrite_replaces_all_five_artifacts_coherently(
    tmp_path, bundle, replacement_bundle
) -> None:
    output_dir = tmp_path / "pilot"
    publish_pilot_artifact_bundle(bundle, output_dir)

    published = publish_pilot_artifact_bundle(
        replacement_bundle, output_dir, overwrite=True
    )

    assert _snapshot(output_dir) == {
        artifact.path: artifact.content for artifact in replacement_bundle.artifacts
    }
    assert published.artifact_refs == tuple(
        artifact.ref for artifact in replacement_bundle.artifacts
    )
    assert any(
        old.content != new.content
        for old, new in zip(bundle.artifacts, replacement_bundle.artifacts, strict=True)
    )


def test_staged_tamper_is_rejected_before_any_final_publication(
    tmp_path, monkeypatch, bundle
) -> None:
    publish_module = import_module("mub.vnext.generation.publish")
    output_dir = tmp_path / "pilot"

    def tampering_helper(
        payloads, *, overwrite, source_paths=(), validators=None, pre_publish=None
    ):
        del overwrite, source_paths, pre_publish
        output_dir.mkdir()
        staged = {}
        for destination, content in payloads.items():
            stage = output_dir / f"{destination.name}.tmp.test"
            stage.write_bytes(content)
            staged[destination] = stage
        tasks_destination = output_dir / "tasks.jsonl"
        tasks_stage = staged[tasks_destination]
        tasks_stage.write_bytes(b" " + tasks_stage.read_bytes())
        validators[tasks_destination](tasks_stage)

    monkeypatch.setattr(
        publish_module, "publish_files_atomically", tampering_helper
    )

    with pytest.raises(ValueError, match="staged artifact bytes"):
        publish_pilot_artifact_bundle(bundle, output_dir)

    assert not any((output_dir / name).exists() for name in ARTIFACT_NAMES)


def test_atomic_fault_rolls_back_without_a_mixed_final_set(
    tmp_path, monkeypatch, bundle, replacement_bundle
) -> None:
    output_dir = tmp_path / "pilot"
    publish_pilot_artifact_bundle(bundle, output_dir)
    before = _snapshot(output_dir)

    def fail_mid_publish(stage: str) -> None:
        if stage == "publish:1":
            raise RuntimeError("injected publication fault")

    monkeypatch.setattr(atomic, "_transaction_fault_point", fail_mid_publish)

    with pytest.raises(RuntimeError, match="injected publication fault"):
        publish_pilot_artifact_bundle(
            replacement_bundle, output_dir, overwrite=True
        )

    assert _snapshot(output_dir) == before
    monkeypatch.setattr(atomic, "_transaction_fault_point", lambda stage: None)

    publish_pilot_artifact_bundle(replacement_bundle, output_dir, overwrite=True)
    assert _snapshot(output_dir) == {
        artifact.path: artifact.content for artifact in replacement_bundle.artifacts
    }


def test_reparse_output_directory_is_rejected_where_supported(
    tmp_path, bundle
) -> None:
    real_output = tmp_path / "real"
    real_output.mkdir()
    alias_output = tmp_path / "alias"
    try:
        alias_output.symlink_to(real_output, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available on this platform")

    with pytest.raises(ValueError, match="reparse-point"):
        publish_pilot_artifact_bundle(bundle, alias_output)

    assert not any((real_output / name).exists() for name in ARTIFACT_NAMES)


def test_adapter_performs_no_disk_writes_before_atomic_helper(
    tmp_path, monkeypatch, bundle
) -> None:
    publish_module = import_module("mub.vnext.generation.publish")
    output_dir = tmp_path / "pilot"
    observed = {}

    def inspect_helper(
        payloads, *, overwrite, source_paths=(), validators=None, pre_publish=None
    ):
        observed["payloads"] = payloads
        observed["overwrite"] = overwrite
        observed["source_paths"] = source_paths
        observed["validators"] = validators
        observed["pre_publish"] = pre_publish
        assert not output_dir.exists()

    monkeypatch.setattr(publish_module, "publish_files_atomically", inspect_helper)

    published = publish_pilot_artifact_bundle(bundle, output_dir)

    assert tuple(observed["payloads"]) == published.artifact_paths
    assert tuple(observed["payloads"].values()) == tuple(
        artifact.content for artifact in bundle.artifacts
    )
    assert observed["overwrite"] is False
    assert observed["source_paths"] == ()
    assert tuple(observed["validators"]) == published.artifact_paths
    assert observed["pre_publish"] is None
    assert not output_dir.exists()
