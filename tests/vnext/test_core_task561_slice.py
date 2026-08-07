from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

import mub.vnext.generation.core_orchestrate as core_orchestrate
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.generation.core_hard_suite import (
    CoreHardSuiteManifest,
    build_core_hard_suite,
    core_hard_suite_hash,
)
from mub.vnext.io import canonical_json_bytes
from mub.vnext.generation.core_orchestrate import stage_core_candidate
from mub.vnext.validation.core_release import (
    _assert_tracked_core_sources_clean,
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
DIRTY_GENERATOR_PATH = ROOT / "mub" / "vnext" / "generation" / "family_g.py"


@pytest.fixture(scope="module")
def bounded_candidate_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("core-candidate-template")
    config = load_core_config(CORE_CONFIG_PATH)
    snapshot = compile_core_snapshot(
        config,
        cores_per_family=10,
        code_revision=TEST_REVISION,
    )
    bundle = build_core_artifact_bundle(snapshot, config)
    for artifact in bundle.artifacts:
        (root / artifact.path).write_bytes(artifact.content)
    return root


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
    _assert_tracked_core_sources_clean()


def _write_dirty_core_config() -> bytes:
    original = CORE_CONFIG_PATH.read_bytes()
    payload = yaml.safe_load(original.decode("utf-8"))
    payload["seed"] += 1
    CORE_CONFIG_PATH.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return original


def _write_dirty_generator_source() -> bytes:
    original = DIRTY_GENERATOR_PATH.read_bytes()
    DIRTY_GENERATOR_PATH.write_bytes(
        original + b"\n# deterministic dirty-source probe\n"
    )
    return original


def test_staging_rejects_valid_dirty_tracked_core_config(tmp_path):
    output = tmp_path / "dirty-config-candidate"
    original = _write_dirty_core_config()
    try:
        with pytest.raises(ValueError, match="tracked Core source"):
            stage_core_candidate(
                config_path=CORE_CONFIG_PATH,
                output_dir=output,
                code_revision=TEST_REVISION,
                cores_per_family=10,
            )
    finally:
        CORE_CONFIG_PATH.write_bytes(original)
        shutil.rmtree(output, ignore_errors=True)


def test_validation_rejects_self_consistent_dirty_tracked_core_config(
    tmp_path,
):
    original = _write_dirty_core_config()
    candidate = tmp_path / "dirty-config-candidate"
    candidate.mkdir()
    try:
        config = load_core_config(CORE_CONFIG_PATH)
        snapshot = compile_core_snapshot(
            config,
            cores_per_family=10,
            code_revision=TEST_REVISION,
        )
        bundle = build_core_artifact_bundle(snapshot, config)
        for artifact in bundle.artifacts:
            (candidate / artifact.path).write_bytes(artifact.content)
        with pytest.raises(ValueError, match="tracked Core source"):
            validate_core_release(candidate, expected_full=False)
    finally:
        CORE_CONFIG_PATH.write_bytes(original)


def test_staging_rejects_dirty_tracked_generator_source(tmp_path):
    output = tmp_path / "dirty-generator-candidate"
    original = _write_dirty_generator_source()
    try:
        with pytest.raises(ValueError, match="tracked Core source"):
            stage_core_candidate(
                config_path=CORE_CONFIG_PATH,
                output_dir=output,
                code_revision=TEST_REVISION,
                cores_per_family=10,
            )
    finally:
        DIRTY_GENERATOR_PATH.write_bytes(original)
        shutil.rmtree(output, ignore_errors=True)


def test_validation_rejects_dirty_tracked_generator_source(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    original = _write_dirty_generator_source()
    try:
        with pytest.raises(ValueError, match="tracked Core source"):
            validate_core_release(candidate, expected_full=False)
    finally:
        DIRTY_GENERATOR_PATH.write_bytes(original)


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
    generator_versions = {
        task.source.generator.generator_name:
        task.source.generator.compiler_version
        for task in snapshot.tasks
    }
    assert dict(bundle.task_manifest.compiler_versions) == generator_versions
    task_ref = bundle.task_manifest.task_file_paths_and_hashes[0]
    core_ref = bundle.task_manifest.source_manifest_paths_and_hashes[0]
    config_ref = bundle.task_manifest.generation_configs_and_hashes[0]
    assert (task_ref.media_type, task_ref.record_count) == (
        "application/x-ndjson",
        len(snapshot.tasks),
    )
    assert (core_ref.media_type, core_ref.record_count) == (
        "application/x-ndjson",
        len(snapshot.semantic_cores),
    )
    assert (config_ref.media_type, config_ref.record_count) == (
        "application/json",
        1,
    )


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


def test_core_bundle_rejects_duplicate_semantic_core_rows():
    config = load_core_config(CORE_CONFIG_PATH)
    snapshot = compile_core_snapshot(
        config,
        cores_per_family=10,
        code_revision=TEST_REVISION,
    )
    corrupted = snapshot.validated_replace(
        semantic_cores=(
            *snapshot.semantic_cores,
            snapshot.semantic_cores[0],
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
    validation_calls = []
    real_validate = core_orchestrate.validate_core_release

    def record_validation(release_dir, *, expected_full=True):
        validation_calls.append((Path(release_dir), expected_full))
        return real_validate(release_dir, expected_full=expected_full)

    monkeypatch.setattr(
        core_orchestrate,
        "validate_core_release",
        record_validation,
    )
    result = stage_core_candidate(
        config_path=config_path,
        output_dir=output,
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )

    assert len(validation_calls) == 1
    assert validation_calls[0][1] is False
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


def test_candidate_staging_fails_closed_when_standalone_validator_raises(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        core_orchestrate,
        "validate_core_release",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("standalone validator failed")
        ),
    )

    with pytest.raises(RuntimeError, match="standalone validator failed"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )
    assert not output.exists()


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
    with pytest.raises(ValueError, match="trusted approved config"):
        stage_core_candidate(
            config_path=attacker_config,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )
    assert not output.exists()


def test_standalone_validation_rejects_self_asserted_invalid_revision(tmp_path):
    output = tmp_path / "invalid-revision-candidate"
    with pytest.raises(ValueError, match="trusted source revision"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision="not-a-git-commit",
            cores_per_family=10,
        )
    assert not output.exists()


def _copy_candidate_template(template: Path, tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    shutil.copytree(template, candidate)
    return candidate


def test_standalone_validation_rejects_extra_directory_entry(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    (candidate / "extra-directory").mkdir()

    with pytest.raises(ValueError, match="exactly seven|regular files"):
        validate_core_release(candidate, expected_full=False)


def test_standalone_validation_rejects_dangling_symlink_entry(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    try:
        (candidate / "dangling-link").symlink_to(candidate / "missing-target")
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="exactly seven|regular files"):
        validate_core_release(candidate, expected_full=False)


def test_standalone_validation_rejects_directory_link_entry(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    target = tmp_path / "directory-target"
    target.mkdir()
    alias = candidate / "directory-link"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"directory junction unavailable: {result.stderr}")
    else:
        alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="exactly seven|regular files"):
        validate_core_release(candidate, expected_full=False)


def test_standalone_validation_rejects_required_file_symlink(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    required = candidate / "tasks.jsonl"
    target = tmp_path / "tasks-target.jsonl"
    required.replace(target)
    try:
        required.symlink_to(target)
    except OSError as exc:
        target.replace(required)
        pytest.skip(f"file symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="regular files"):
        validate_core_release(candidate, expected_full=False)


def test_standalone_validation_rejects_coordinated_artifact_ref_rebinding(
    bounded_candidate_template,
    tmp_path,
):
    candidate = _copy_candidate_template(bounded_candidate_template, tmp_path)
    manifest_path = candidate / "task_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["task_file_paths_and_hashes"][0]["media_type"] = (
        "application/json"
    )
    manifest_payload["generation_configs_and_hashes"][0]["record_count"] = 999
    manifest = TaskManifestV3.model_validate(manifest_payload)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)

    hard_path = candidate / "core-hard-v1.json"
    hard_payload = json.loads(hard_path.read_text(encoding="utf-8"))
    hard_payload["source_task_manifest_hash"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    hard_payload["suite_hash"] = core_hard_suite_hash(hard_payload)
    hard = CoreHardSuiteManifest.model_validate(hard_payload)
    hard_path.write_bytes(canonical_json_bytes(hard))

    with pytest.raises(ValueError, match="task manifest"):
        validate_core_release(candidate, expected_full=False)


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


def test_staging_rejects_dangling_output_leaf_before_compile(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "dangling-output"
    try:
        output.symlink_to(tmp_path / "missing-target")
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")
    monkeypatch.setattr(
        core_orchestrate,
        "compile_core_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compiled")
        ),
    )

    with pytest.raises(ValueError, match="symlink or junction"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )


def test_staging_rejects_verified_temporary_tree_substitution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "candidate"
    real_replace = os.replace

    def substitute_then_replace(source, destination):
        source = Path(source)
        verified_aside = source.with_name(source.name + ".verified-aside")
        real_replace(source, verified_aside)
        source.mkdir()
        (source / "attacker.txt").write_text("substituted", encoding="utf-8")
        real_replace(source, destination)

    monkeypatch.setattr(core_orchestrate.os, "replace", substitute_then_replace)
    with pytest.raises(ValueError, match="verified temporary"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )
    assert (output / "attacker.txt").read_text(encoding="utf-8") == "substituted"


def test_staging_rechecks_parent_after_final_transfer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = tmp_path / "parent"
    output = parent / "candidate"
    real_resolve = core_orchestrate._resolve_path
    real_replace = os.replace
    state = {"changed": False}

    def changed_parent(path):
        if state["changed"] and path == parent:
            return tmp_path / "redirected-parent"
        return real_resolve(path)

    def replace_then_change(source, destination):
        real_replace(source, destination)
        state["changed"] = True

    monkeypatch.setattr(core_orchestrate, "_resolve_path", changed_parent)
    monkeypatch.setattr(core_orchestrate.os, "replace", replace_then_change)
    with pytest.raises(ValueError, match="staging parent changed"):
        stage_core_candidate(
            config_path=CORE_CONFIG_PATH,
            output_dir=output,
            code_revision=TEST_REVISION,
            cores_per_family=10,
        )
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_staging_rejects_lexical_junction_parent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr}")
    monkeypatch.setattr(
        core_orchestrate,
        "compile_core_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compiled")),
    )
    try:
        with pytest.raises(ValueError, match="symlink or junction"):
            stage_core_candidate(
                config_path=CORE_CONFIG_PATH,
                output_dir=junction / "candidate",
                code_revision=TEST_REVISION,
                cores_per_family=10,
            )
    finally:
        if junction.exists():
            junction.rmdir()


def test_staged_candidate_cleanup_preserves_substituted_output(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "candidate"
    result = stage_core_candidate(
        config_path=CORE_CONFIG_PATH,
        output_dir=output,
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )
    verified_aside = tmp_path / "verified-aside"
    os.replace(output, verified_aside)
    output.mkdir()
    marker = output / "unrelated.txt"
    marker.write_text("do not delete", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "task561_generate_cli",
        ROOT / "scripts" / "vnext_generate_core.py",
    )
    assert spec is not None and spec.loader is not None
    generate_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_cli)

    def revision_check(expected=None):
        if expected is None:
            return TEST_REVISION
        raise RuntimeError("revision changed")

    monkeypatch.setattr(generate_cli, "_revision", revision_check)
    monkeypatch.setattr(generate_cli, "stage_core_candidate", lambda **kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vnext_generate_core.py",
            "--config",
            str(CORE_CONFIG_PATH),
            "--output-dir",
            str(output),
        ],
    )

    try:
        with pytest.raises(RuntimeError, match="revision changed"):
            generate_cli.main()
        assert marker.read_text(encoding="utf-8") == "do not delete"
    finally:
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(verified_aside, ignore_errors=True)


def test_verified_cleanup_cannot_delete_replacement_in_check_delete_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "candidate"
    result = stage_core_candidate(
        config_path=CORE_CONFIG_PATH,
        output_dir=output,
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )
    verified_aside = tmp_path / "verified-aside"
    marker = output / "unrelated.txt"
    real_rmtree = shutil.rmtree
    injected = False

    def substitute_before_delete(path, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            if output.exists():
                os.replace(output, verified_aside)
            output.mkdir()
            marker.write_text("do not delete", encoding="utf-8")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(core_orchestrate.shutil, "rmtree", substitute_before_delete)
    try:
        assert result.remove_if_unchanged()
        assert injected
        assert marker.read_text(encoding="utf-8") == "do not delete"
    finally:
        real_rmtree(output, ignore_errors=True)
        real_rmtree(verified_aside, ignore_errors=True)


def test_temporary_cleanup_cannot_delete_replacement_in_check_delete_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    (temporary / "staged.txt").write_text("staged", encoding="utf-8")
    temporary_identity = core_orchestrate._path_identity(temporary)
    verified_aside = tmp_path / "verified-aside"
    marker = temporary / "unrelated.txt"
    real_rmtree = shutil.rmtree
    injected = False

    def substitute_before_delete(path, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            if temporary.exists():
                os.replace(temporary, verified_aside)
            temporary.mkdir()
            marker.write_text("do not delete", encoding="utf-8")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(core_orchestrate.shutil, "rmtree", substitute_before_delete)
    try:
        core_orchestrate._remove_tree_if_identity_matches(
            temporary,
            temporary_identity,
        )
        assert injected
        assert marker.read_text(encoding="utf-8") == "do not delete"
    finally:
        real_rmtree(temporary, ignore_errors=True)
        real_rmtree(verified_aside, ignore_errors=True)


def test_staged_candidate_constructor_remains_backward_compatible():
    candidate = core_orchestrate.StagedCoreCandidate(
        release_dir=Path("relative-candidate"),
        semantic_core_count=1,
        task_count=4,
        split_core_counts={"test": 1},
        split_task_counts={"test": 4},
        hard_suite_core_count=1,
        hard_suite_task_count=4,
    )

    assert candidate.release_dir == Path("relative-candidate")
    assert not candidate.remove_if_unchanged()


def test_staging_preserves_relative_release_dir_representation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    output = Path("relative-candidate")
    result = stage_core_candidate(
        config_path=CORE_CONFIG_PATH,
        output_dir=output,
        code_revision=TEST_REVISION,
        cores_per_family=10,
    )

    try:
        assert result.release_dir == output
    finally:
        assert result.remove_if_unchanged()

