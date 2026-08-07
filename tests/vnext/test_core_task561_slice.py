from pathlib import Path
import os
import subprocess
import sys

import pytest

import mub.vnext.generation.core_orchestrate as core_orchestrate
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.core_hard_suite import build_core_hard_suite
from mub.vnext.generation.core_orchestrate import stage_core_candidate
from mub.vnext.validation.core_release import validate_core_release


ROOT = Path(__file__).resolve().parents[2]


def test_core_clis_run_from_project_root_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for script in ("vnext_generate_core.py", "vnext_validate_core.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr


def test_bounded_core_bundle_is_canonical_and_manifest_bound():
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=10, code_revision="test-revision")
    bundle = build_core_artifact_bundle(snapshot, config)

    assert [artifact.path for artifact in bundle.artifacts] == [
        "tasks.jsonl",
        "semantic_cores.jsonl",
        "generation_config.json",
        "split_balance.json",
        "task_manifest.json",
        "core-hard-v1.json",
        "validation_report.json",
    ]
    assert bundle.task_manifest.task_file_paths_and_hashes[0].sha256 == bundle.artifacts[0].ref.sha256
    assert bundle.task_manifest.generation_configs_and_hashes[0].sha256 == bundle.artifacts[2].ref.sha256


def test_core_bundle_rejects_semantic_core_payload_drift():
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=10, code_revision="test-revision")
    victim = snapshot.semantic_cores[0]
    changed_profile = dict(victim.profile)
    changed_profile["context_length"] += 1
    corrupted = snapshot.validated_replace(
        semantic_cores=(
            victim.model_copy(update={"profile": changed_profile}),
            *snapshot.semantic_cores[1:],
        )
    )
    with pytest.raises(ValueError, match="semantic core"):
        build_core_artifact_bundle(corrupted, config)


def test_hard_suite_is_manifest_only_and_authenticated():
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=20, code_revision="test-revision")
    task_manifest_hash = "a" * 64
    suite = build_core_hard_suite(snapshot, source_task_manifest_hash=task_manifest_hash, per_family=2)

    assert suite.selection_policy_version == "core-hard-v1"
    assert suite.source_task_manifest_hash == task_manifest_hash
    assert len(suite.semantic_core_ids) == 14
    assert len(suite.task_ids) == 56
    assert tuple(sorted(suite.task_ids)) == suite.task_ids
    assert not hasattr(suite, "tasks")
    selected_tasks = {
        task.metadata.split_key.semantic_core_id: task
        for task in snapshot.tasks
        if task.metadata.extra["surface_variant"] == 0
        and task.metadata.split_key.semantic_core_id in suite.semantic_core_ids
    }
    for selector_kind, core_ids in suite.condition_coverage[
        "current_historical_query"
    ].items():
        assert {
            selected_tasks[core_id].queries[0].selector.kind
            for core_id in core_ids
        } == {selector_kind}


def test_candidate_staging_is_transactional_and_standalone_validated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    config_path = ROOT / "configs" / "vnext" / "core.yaml"
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        core_orchestrate,
        "validate_core_release",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("staging repeated standalone validation")
        ),
    )
    result = stage_core_candidate(
        config_path=config_path,
        output_dir=output,
        code_revision="test-revision",
        cores_per_family=10,
    )

    assert result.release_dir == output
    assert not any(path.name.startswith("IMMUTABLE") for path in output.iterdir())
    report = validate_core_release(output, expected_full=False)
    assert report.valid
    assert report.semantic_core_count == 70
    assert report.task_count == 280
    (output / "split_balance.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Balance|split_balance"):
        validate_core_release(output, expected_full=False)


def test_candidate_staging_rejects_immutable_release_descendants(
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = ROOT / "configs" / "vnext" / "core.yaml"
    monkeypatch.setattr(
        core_orchestrate,
        "compile_core_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compiled")),
    )
    with pytest.raises(ValueError, match="outside the immutable release root"):
        stage_core_candidate(
            config_path=config_path,
            output_dir=ROOT / "data" / "vnext" / "core" / "v3",
            code_revision="test-revision",
        )

