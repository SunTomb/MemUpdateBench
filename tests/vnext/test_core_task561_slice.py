from pathlib import Path
import inspect
import os
import subprocess
import sys

import pytest
import yaml

import mub.vnext.generation.core_orchestrate as core_orchestrate
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.core_hard_suite import build_core_hard_suite
from mub.vnext.generation.core_orchestrate import stage_core_candidate
from mub.vnext.validation.core_release import (
    _trusted_code_revision,
    validate_core_release,
)


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG_PATH = ROOT / "configs" / "vnext" / "core.yaml"
TEST_REVISION = subprocess.check_output(
    ("git", "rev-parse", "HEAD"),
    cwd=ROOT,
    text=True,
).strip()


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
        if script == "vnext_validate_core.py":
            assert "--trusted-config" not in result.stdout
            assert "--expected-code-revision" not in result.stdout
    assert "trusted_config_path" not in inspect.signature(
        validate_core_release
    ).parameters
    assert "expected_code_revision" not in inspect.signature(
        validate_core_release
    ).parameters


def test_trusted_revision_ignores_git_environment_redirects(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    attacker_git_dir = tmp_path / "attacker.git"
    attacker_git_dir.mkdir()
    (attacker_git_dir / "HEAD").write_text("b" * 40 + "\n", encoding="ascii")
    monkeypatch.setenv("GIT_DIR", str(attacker_git_dir))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))

    assert _trusted_code_revision() == TEST_REVISION


def test_bounded_core_bundle_is_canonical_and_manifest_bound():
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(config, cores_per_family=10, code_revision=TEST_REVISION)
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
    snapshot = compile_core_snapshot(config, cores_per_family=10, code_revision=TEST_REVISION)
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
    snapshot = compile_core_snapshot(config, cores_per_family=20, code_revision=TEST_REVISION)
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
    family_f_coverage = suite.condition_coverage["current_historical_query"]
    assert any(condition.startswith("selector_kind=") for condition in family_f_coverage)
    for condition, core_ids in family_f_coverage.items():
        if not condition.startswith("selector_kind="):
            continue
        selector_kind = condition.split("=", 1)[1]
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
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )

    assert result.release_dir == output
    assert not any(path.name.startswith("IMMUTABLE") for path in output.iterdir())
    report = validate_core_release(
        output,
        expected_full=False,
    )
    assert report.valid
    assert report.semantic_core_count == 70
    assert report.task_count == 280
    (output / "split_balance.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Balance|split_balance"):
        validate_core_release(
            output,
            expected_full=False,
        )


def test_candidate_staging_rejects_immutable_release_descendants(
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = ROOT / "configs" / "vnext" / "core.yaml"
    monkeypatch.setattr(
        core_orchestrate,
        "compile_core_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compiled")),
    )
    with pytest.raises(ValueError, match="outside immutable release roots"):
        stage_core_candidate(
            config_path=config_path,
            output_dir=ROOT / "data" / "vnext" / "core" / "v3",
            code_revision=TEST_REVISION,
        )


def test_standalone_validation_rejects_tampered_seed_and_release_id(tmp_path):
    payload = yaml.safe_load(CORE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["seed"] = 123456789
    payload["release_id"] = "attacker-relabelled-core"
    attacker_config = tmp_path / "attacker-core.yaml"
    attacker_config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "attacker-candidate"
    stage_core_candidate(
        config_path=attacker_config,
        output_dir=output,
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )

    with pytest.raises(ValueError, match="trusted approved config"):
        validate_core_release(
            output,
            expected_full=False,
        )


def test_standalone_validation_rejects_self_asserted_invalid_revision(tmp_path):
    output = tmp_path / "invalid-revision-candidate"
    stage_core_candidate(
        config_path=CORE_CONFIG_PATH,
        output_dir=output,
        code_revision="not-a-git-commit",
        cores_per_family=10,
    )

    with pytest.raises(ValueError, match="trusted source revision"):
        validate_core_release(
            output,
            expected_full=False,
        )


def test_candidate_staging_rechecks_parent_after_compile(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = tmp_path / "safe-parent"
    output = parent / "candidate"
    original_resolve = Path.resolve
    state = {"swapped": False}

    def redirected_resolve(path, *args, **kwargs):
        if state["swapped"] and path == parent:
            return tmp_path / "redirected-parent"
        return original_resolve(path, *args, **kwargs)

    def compile_then_swap(*args, **kwargs):
        state["swapped"] = True
        return None

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    monkeypatch.setattr(core_orchestrate, "compile_core_snapshot", compile_then_swap)

    with pytest.raises(ValueError, match="staging parent changed"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )

