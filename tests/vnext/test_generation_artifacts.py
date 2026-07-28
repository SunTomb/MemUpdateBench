from __future__ import annotations

import builtins
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from mub.vnext.contracts import MemUpdateTask, Split
from mub.vnext.generation import (
    CompiledPilotTasks,
    PilotArtifactBundle,
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes
from mub.vnext.validation import validate_splits


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "generation-artifacts-test-revision"
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
def compiled(config) -> CompiledPilotTasks:
    return compile_pilot_tasks(config, code_revision=REVISION)


@pytest.fixture(scope="module")
def bundle(compiled, config) -> PilotArtifactBundle:
    return build_pilot_artifact_bundle(compiled, config)


def _clone_compiled(compiled: CompiledPilotTasks) -> CompiledPilotTasks:
    return replace(compiled)


def test_bundle_contains_exact_canonical_counts_hashes_and_artifacts(
    bundle, compiled, config
) -> None:
    tasks = compiled.tasks
    manifest = bundle.task_manifest

    assert bundle.tasks_jsonl is compiled.tasks_jsonl
    assert len(tasks) == 1440
    assert dict(manifest.split_counts) == {
        "train": 1008,
        "dev": 144,
        "test": 288,
        "evaluation_only": 0,
    }
    assert dict(manifest.semantic_core_counts) == {
        "train": 336,
        "dev": 48,
        "test": 96,
        "evaluation_only": 0,
    }
    expected_family_difficulty = Counter(
        f"{task.task_family}|{task.difficulty.value}" for task in tasks
    )
    assert len(manifest.family_difficulty_counts) == 12
    assert dict(manifest.family_difficulty_counts) == {
        key: expected_family_difficulty[key]
        for key in sorted(expected_family_difficulty)
    }
    assert manifest.data_release_id == config.release_id
    assert manifest.code_revision == REVISION
    assert manifest.task_schema_version == config.schema_version
    assert manifest.compiler_versions == {
        compiled.generator_name: compiled.compiler_version
    }
    assert manifest.source_manifest_paths_and_hashes == ()
    assert manifest.human_audit_artifacts == ()

    task_ref = manifest.task_file_paths_and_hashes[0]
    assert task_ref.path == "tasks.jsonl"
    assert task_ref.media_type == "application/x-ndjson"
    assert task_ref.record_count == 1440
    assert task_ref.sha256 == hashlib.sha256(compiled.tasks_jsonl).hexdigest()

    config_ref = manifest.generation_configs_and_hashes[0]
    assert config_ref.path == "generation_config.json"
    assert config_ref.media_type == "application/json"
    assert config_ref.record_count == 1
    assert config_ref.sha256 == compiled.config_sha256
    assert bundle.config_sha256 == compiled.config_sha256
    assert hashlib.sha256(bundle.resolved_config_bytes).hexdigest() == compiled.config_sha256
    assert canonical_json_bytes(bundle.resolved_config) == bundle.resolved_config_bytes

    assert canonical_json_bytes(manifest) == bundle.task_manifest_bytes
    assert canonical_json_bytes(bundle.split_balance_report) == bundle.split_balance_bytes
    assert canonical_json_bytes(bundle.validation_report) == bundle.validation_report_bytes
    assert bundle.split_balance_report == compiled.split_assignment.split_balance

    assert tuple(artifact.ref.path for artifact in bundle.artifacts) == ARTIFACT_NAMES
    for artifact in bundle.artifacts:
        assert artifact.ref.sha256 == hashlib.sha256(artifact.content).hexdigest()
        assert artifact.ref.media_type in {
            "application/json",
            "application/x-ndjson",
        }
    assert bundle.artifacts[0].content is compiled.tasks_jsonl

    declared_hashes = manifest.leakage_check_summary["task_hashes"]
    assert len(declared_hashes) == 1440
    assert declared_hashes == {
        task.task_id: hashlib.sha256(row).hexdigest()
        for task, row in zip(tasks, compiled.tasks_jsonl.splitlines(), strict=True)
    }


def test_repeat_construction_is_byte_identical_and_fresh(compiled, config, bundle) -> None:
    repeated = build_pilot_artifact_bundle(compiled, config)

    assert repeated is not bundle
    assert repeated.resolved_config is not bundle.resolved_config
    assert repeated.task_manifest is not bundle.task_manifest
    assert repeated.split_balance_report is not bundle.split_balance_report
    assert repeated.validation_report is not bundle.validation_report
    assert tuple(artifact.content for artifact in repeated.artifacts) == tuple(
        artifact.content for artifact in bundle.artifacts
    )
    assert repeated.task_manifest_bytes == bundle.task_manifest_bytes
    assert repeated.validation_report_bytes == bundle.validation_report_bytes


def test_bundle_manifest_passes_split_validation_without_issues(bundle, compiled) -> None:
    direct_report = validate_splits(compiled.tasks, task_manifest=bundle.task_manifest)

    assert bundle.validation_report == direct_report
    assert direct_report.valid is True
    assert direct_report.issues == ()


def test_required_strata_are_complete_and_deviations_are_honest(bundle, compiled) -> None:
    required = bundle.task_manifest.leakage_check_summary[
        "required_minimum_strata"
    ]
    deviations = bundle.task_manifest.leakage_check_summary[
        "small_cell_deviations"
    ]
    required_keys = {
        (
            record["task_family"],
            record["difficulty"],
            record["update_depth_bucket"],
        )
        for record in required
    }
    observed = Counter(
        (
            task.task_family,
            task.difficulty.value,
            task.metadata.resolved_profile["update_depth_bucket"],
            task.metadata.split.value,
        )
        for task in compiled.tasks
    )

    assert tuple(
        (
            record["task_family"],
            record["difficulty"],
            record["update_depth_bucket"],
        )
        for record in required
    ) == tuple(sorted(required_keys))
    assert required_keys
    assert all(
        observed[(*key, split.value)] > 0
        for key in required_keys
        for split in (Split.TRAIN, Split.DEV, Split.TEST)
    )
    assert deviations == ()


def test_manifest_maps_have_deterministic_key_order(bundle) -> None:
    manifest = bundle.task_manifest

    assert tuple(manifest.split_counts) == (
        "train",
        "dev",
        "test",
        "evaluation_only",
    )
    assert tuple(manifest.semantic_core_counts) == (
        "train",
        "dev",
        "test",
        "evaluation_only",
    )
    assert tuple(manifest.family_difficulty_counts) == tuple(
        sorted(manifest.family_difficulty_counts)
    )
    task_hashes = manifest.leakage_check_summary["task_hashes"]
    assert tuple(task_hashes) == tuple(sorted(task_hashes))


def test_bundle_is_deeply_immutable_and_detached_from_input(compiled, config) -> None:
    mutable_config = type(config).model_validate(config.model_dump(mode="python"))
    snapshot = build_pilot_artifact_bundle(compiled, mutable_config)
    original_bytes = snapshot.resolved_config_bytes

    mutable_config.families.repeated_same_slot_update.update_depths.append(999)

    assert snapshot.resolved_config_bytes == original_bytes
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.resolved_config.release_id = "changed"
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.resolved_config.families.repeated_same_slot_update.update_depths.append(
            999
        )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.task_manifest.split_counts["train"] = 0
    with pytest.raises((AttributeError, TypeError, ValueError)):
        snapshot.validation_report.issues += ()


def test_config_mismatch_is_rejected(compiled, config) -> None:
    mismatched = config.model_copy(update={"release_id": "different-release"})

    with pytest.raises(ValueError, match="config.*compiled snapshot"):
        build_pilot_artifact_bundle(compiled, mismatched)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_sha256", "0" * 64, "compiled snapshot"),
        ("code_revision", "wrong-revision", "compiled snapshot"),
        ("compiler_version", "0.0.0", "compiler version"),
        ("generator_name", "wrong-generator", "generator"),
    ],
)
def test_modified_compiled_metadata_is_rejected(
    compiled, config, field, value, message
) -> None:
    tampered = _clone_compiled(compiled)
    object.__setattr__(tampered, field, value)

    with pytest.raises(ValueError, match=message):
        build_pilot_artifact_bundle(tampered, config)


def test_tampered_compiled_bytes_are_rejected(compiled, config) -> None:
    tampered = _clone_compiled(compiled)
    lines = compiled.tasks_jsonl.splitlines()
    payload = MemUpdateTask.model_validate_json(lines[0]).model_dump(mode="python")
    payload["metadata"]["extra"]["surface_variant"] = 9
    lines[0] = canonical_json_bytes(MemUpdateTask.model_validate(payload))
    object.__setattr__(tampered, "tasks_jsonl", b"\n".join(lines) + b"\n")

    with pytest.raises(ValueError, match="compiled Pilot snapshot failed"):
        build_pilot_artifact_bundle(tampered, config)


def test_noncanonical_compiled_snapshot_is_rejected(compiled, config) -> None:
    tampered = _clone_compiled(compiled)
    first_line, rest = compiled.tasks_jsonl.split(b"\n", 1)
    object.__setattr__(tampered, "tasks_jsonl", b" " + first_line + b"\n" + rest)

    with pytest.raises(ValueError, match="canonical"):
        build_pilot_artifact_bundle(tampered, config)


def test_invalid_split_validation_is_rejected(monkeypatch, compiled, config) -> None:
    import mub.vnext.generation.artifacts as artifacts_module
    from mub.vnext.validation import ValidationIssue, build_report

    invalid = build_report(
        (
            ValidationIssue(
                code="injected_split_failure",
                message="injected failure",
                path="tasks",
                severity="error",
            ),
        )
    )
    monkeypatch.setattr(artifacts_module, "validate_splits", lambda *args, **kwargs: invalid)

    with pytest.raises(ValueError, match="split validation"):
        build_pilot_artifact_bundle(compiled, config)


def test_bundle_construction_performs_no_disk_io(monkeypatch, compiled, config) -> None:
    def reject_io(*args, **kwargs):
        raise AssertionError("disk I/O is forbidden")

    monkeypatch.setattr(builtins, "open", reject_io)
    monkeypatch.setattr(Path, "open", reject_io)
    monkeypatch.setattr(Path, "write_bytes", reject_io)
    monkeypatch.setattr(Path, "write_text", reject_io)

    result = build_pilot_artifact_bundle(compiled, config)

    assert result.tasks_jsonl == compiled.tasks_jsonl
