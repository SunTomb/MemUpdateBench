from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import mub.vnext.post_core.release_v1 as release_v1
from mub.vnext.post_core.contracts_v1 import POST_CORE_ARTIFACT_ORDER
from mub.vnext.post_core.qualification_v1 import CapabilityProbeReportV1
from mub.vnext.post_core.release_v1 import (
    EXIT_BLOCKED,
    EXIT_PUBLICATION,
    EXIT_STALE_SOURCE,
    CommittedPostCoreReleaseError,
    PostCoreReleaseError,
    build_post_core_release_v1,
    load_post_core_config_v1,
    load_post_core_registry_v1,
    publish_post_core_release_v1,
    verify_post_core_release_v1,
)
from mub.vnext.post_core.model_registry_v1 import build_initial_model_registry_v1


CONFIG = Path(__file__).parents[2] / "configs" / "vnext" / "post_core" / "release_v1.json"
ROOT = Path(__file__).parents[2]
CORE_SOURCE_ROOT_CANDIDATES = (
    ROOT / "data" / "vnext" / "core" / "v3",
    ROOT.parent / "vnext-phase0" / "data" / "vnext" / "core" / "v3",
)
TASK14_SOURCE_ROOT_CANDIDATES = (
    Path("D:/USTC/2026Winter/MemUpdateBench_releases/core_task14_84beabb_v1"),
)
TASK14_SHA = "2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035"
EXPECTED_KEYS = (
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5",
)


def _source_root(candidates: tuple[Path, ...], marker: str) -> Path:
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    pytest.skip(f"authenticated post-Core source is unavailable: {marker}")


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    core_source = _source_root(
        CORE_SOURCE_ROOT_CANDIDATES, "task_release_manifest.json"
    )
    task14_source = _source_root(
        TASK14_SOURCE_ROOT_CANDIDATES, "core_final_root_index.json"
    )
    core = tmp_path / "core_manifest.json"
    core.write_bytes((core_source / "task_release_manifest.json").read_bytes())
    task14_root = tmp_path / "task14"
    task14_root.mkdir()
    for source in task14_source.iterdir():
        if source.is_file():
            (task14_root / source.name).write_bytes(source.read_bytes())
    return core, task14_root / "core_final_root_index.json"


def _config(tmp_path: Path, task14: Path) -> Path:
    path = tmp_path / "release.json"
    path.write_bytes(CONFIG.read_bytes())
    return path


def test_release_has_exact_seven_artifacts_and_zero_calls(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    assert config.release_id == "memupdatebench.post-core.phase0.v1"
    assert config.registry_keys == EXPECTED_KEYS

    # The source bytes are not interpreted as executable input in Phase 0.
    publication = build_post_core_release_v1(config, core, task14)
    assert tuple(publication.artifact_bytes) == (*POST_CORE_ARTIFACT_ORDER, "post_core_artifact_index.json")
    index = json.loads(publication.artifact_bytes["post_core_artifact_index.json"])
    assert tuple(row["path"] for row in index["artifacts"]) == POST_CORE_ARTIFACT_ORDER
    assert "post_core_artifact_index.json" not in tuple(row["path"] for row in index["artifacts"])
    assert publication.pending_count == 8
    assert publication.provider_calls == 0
    assert publication.model_loads == 0
    assert publication.network_calls == 0
    assert publication.executable_call_count == 0


def test_release_build_is_deterministic_and_qwen_plan_remains_non_executable(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    first = build_post_core_release_v1(config, core, task14)
    second = build_post_core_release_v1(config, core, task14)
    assert first.artifact_bytes == second.artifact_bytes
    plan = json.loads(first.artifact_bytes["execution_plan.json"])
    qwen = next(row for row in plan["scopes"] if row["registry_key"] == "qwen35_9b_bf16")
    assert qwen["requested_calls"] == 320
    assert qwen["executable_calls"] == 0
    assert plan["executable_call_count"] == 0


def test_publish_reopen_and_no_clobber(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    output = tmp_path / "published"
    result = publish_post_core_release_v1(config, core, task14, output)
    assert result.output_root == output.resolve()
    reopened = verify_post_core_release_v1(output, config, core, task14)
    assert reopened.index_sha256 == result.index_sha256
    with pytest.raises(FileExistsError):
        publish_post_core_release_v1(config, core, task14, output)


def test_source_mutation_is_rejected_before_commit(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))

    def mutate() -> None:
        core.write_bytes(core.read_bytes() + b"x")

    with pytest.raises(PostCoreReleaseError, match="source"):
        publish_post_core_release_v1(config, core, task14, tmp_path / "output", before_commit=mutate)


def test_task14_sibling_mutation_is_rejected_before_commit(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    sibling = task14.parent / "core_final_review_report.json"

    def mutate() -> None:
        sibling.write_bytes(sibling.read_bytes() + b"x")

    with pytest.raises(PostCoreReleaseError, match="Task 14|sibling|source"):
        publish_post_core_release_v1(
            config,
            core,
            task14,
            tmp_path / "output",
            before_commit=mutate,
        )
    assert not (tmp_path / "output").exists()
    assert not tuple(tmp_path.glob(".mub-post-core-stage-*"))


def test_registry_source_mutation_is_rejected_and_staging_is_cleaned(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    registry_path = tmp_path / "model_registry.json"
    baseline = build_post_core_release_v1(config, core, task14)
    registry_path.write_bytes(baseline.artifact_bytes["model_registry.json"])
    registry = load_post_core_registry_v1(registry_path, config)

    def mutate() -> None:
        registry_path.write_bytes(registry_path.read_bytes() + b"x")

    with pytest.raises(PostCoreReleaseError, match="model registry|bytes changed"):
        publish_post_core_release_v1(
            config,
            core,
            task14,
            tmp_path / "output",
            registry=registry,
            before_commit=mutate,
        )
    assert not (tmp_path / "output").exists()
    assert not tuple(tmp_path.glob(".mub-post-core-stage-*"))


def test_provenance_source_mutation_is_rejected_and_staging_is_cleaned(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    provenance_path = tmp_path / "provenance.jsonl"
    baseline = build_post_core_release_v1(config, core, task14)
    provenance_path.write_bytes(baseline.artifact_bytes["provenance.jsonl"])

    def mutate() -> None:
        provenance_path.write_bytes(provenance_path.read_bytes() + b"x")

    with pytest.raises(PostCoreReleaseError, match="provenance|bytes changed"):
        publish_post_core_release_v1(
            config,
            core,
            task14,
            tmp_path / "output",
            provenance_path=provenance_path,
            before_commit=mutate,
        )
    assert not (tmp_path / "output").exists()
    assert not tuple(tmp_path.glob(".mub-post-core-stage-*"))


@pytest.mark.parametrize(
    "field",
    (
        "identity_status",
        "artifact_sha256",
        "byte_count",
        "evidence_type",
        "source_location",
        "credential_env_var",
        "runtime",
    ),
)
def test_provided_provenance_must_equal_pending_intent_derivation(
    tmp_path: Path, field: str
) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    baseline = build_post_core_release_v1(config, core, task14)
    rows = [
        json.loads(line)
        for line in baseline.artifact_bytes["provenance.jsonl"].decode().splitlines()
    ]
    row = rows[0]
    if field == "identity_status":
        row[field] = "QUALIFIED"
    elif field == "artifact_sha256":
        row[field] = "0" * 64
    elif field == "byte_count":
        row[field] += 1
    elif field == "evidence_type":
        row[field] = "fabricated_evidence"
    elif field == "source_location":
        row[field] = "fabricated://source"
    elif field == "credential_env_var":
        row[field] = (
            "ANTHROPIC_API_KEY"
            if row[field] == "OPENAI_API_KEY"
            else "OPENAI_API_KEY"
        )
    else:
        row[field] = {"fabricated": "value"}
    provenance_path = tmp_path / "provenance.jsonl"
    provenance_path.write_bytes(
        b"".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            for item in rows
        )
    )
    with pytest.raises(ValueError, match="provenance|pending|deterministic|expected"):
        build_post_core_release_v1(
            config, core, task14, provenance_path=provenance_path
        )


def test_config_source_mutation_is_rejected_and_staging_is_cleaned(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config_path = _config(tmp_path, task14)
    config = load_post_core_config_v1(config_path)

    def mutate() -> None:
        config_path.write_bytes(config_path.read_bytes() + b"x")

    with pytest.raises(PostCoreReleaseError, match="post-Core config|bytes changed"):
        publish_post_core_release_v1(
            config,
            core,
            task14,
            tmp_path / "output",
            before_commit=mutate,
        )
    assert not (tmp_path / "output").exists()
    assert not tuple(tmp_path.glob(".mub-post-core-stage-*"))


def _replace_with_same_bytes(path: Path) -> None:
    replacement = path.with_name(f".{path.name}.replacement")
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)


def test_config_original_snapshot_rejects_identity_replacement(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config_path = _config(tmp_path, task14)
    config = load_post_core_config_v1(config_path)
    _replace_with_same_bytes(config_path)

    with pytest.raises(PostCoreReleaseError, match="config|identity"):
        publish_post_core_release_v1(config, core, task14, tmp_path / "output")


def test_config_mutation_after_rename_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, task14 = _sources(tmp_path)
    config_path = _config(tmp_path, task14)
    config = load_post_core_config_v1(config_path)
    original_commit = release_v1._directory_commit_noreplace

    def commit_then_mutate(staging: Path, output: Path) -> None:
        original_commit(staging, output)
        _replace_with_same_bytes(config_path)

    monkeypatch.setattr(release_v1, "_directory_commit_noreplace", commit_then_mutate)
    with pytest.raises(CommittedPostCoreReleaseError, match="config|identity|committed root"):
        publish_post_core_release_v1(config, core, task14, tmp_path / "output")


def test_independent_verify_rebuilds_against_external_registry_source(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    baseline = build_post_core_release_v1(config, core, task14)
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_bytes(baseline.artifact_bytes["model_registry.json"])
    registry = load_post_core_registry_v1(registry_path, config)
    output = tmp_path / "output"
    publication = publish_post_core_release_v1(
        config, core, task14, output, registry=registry
    )
    reopened = verify_post_core_release_v1(
        output, config, core, task14, registry=registry
    )
    assert reopened.index_sha256 == publication.index_sha256


def test_registry_mutation_after_rename_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    baseline = build_post_core_release_v1(config, core, task14)
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_bytes(baseline.artifact_bytes["model_registry.json"])
    registry = load_post_core_registry_v1(registry_path, config)
    original_commit = release_v1._directory_commit_noreplace

    def commit_then_mutate(staging: Path, output: Path) -> None:
        original_commit(staging, output)
        _replace_with_same_bytes(registry_path)

    monkeypatch.setattr(release_v1, "_directory_commit_noreplace", commit_then_mutate)
    with pytest.raises(CommittedPostCoreReleaseError, match="registry|identity|committed root"):
        publish_post_core_release_v1(
            config, core, task14, tmp_path / "output", registry=registry
        )


def test_provenance_mutation_after_rename_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    baseline = build_post_core_release_v1(config, core, task14)
    provenance_path = tmp_path / "provenance.jsonl"
    provenance_path.write_bytes(baseline.artifact_bytes["provenance.jsonl"])
    original_commit = release_v1._directory_commit_noreplace

    def commit_then_mutate(staging: Path, output: Path) -> None:
        original_commit(staging, output)
        _replace_with_same_bytes(provenance_path)

    monkeypatch.setattr(release_v1, "_directory_commit_noreplace", commit_then_mutate)
    with pytest.raises(CommittedPostCoreReleaseError, match="provenance|identity|committed root"):
        publish_post_core_release_v1(
            config,
            core,
            task14,
            tmp_path / "output",
            provenance_path=provenance_path,
        )


def test_post_rename_verification_failure_preserves_committed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    output = tmp_path / "output"
    original_verify = release_v1.verify_post_core_release_v1
    calls = 0

    def fail_after_commit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PostCoreReleaseError("forced post-rename verification failure")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(release_v1, "verify_post_core_release_v1", fail_after_commit)
    with pytest.raises(CommittedPostCoreReleaseError, match="committed root") as failure:
        publish_post_core_release_v1(config, core, task14, output)
    assert calls == 2
    assert failure.value.committed_root == output.resolve()
    assert output.is_dir()
    assert not tuple(tmp_path.glob(".mub-post-core-stage-*"))


def test_tampered_artifact_fails_exact_reopen_verification(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    output = tmp_path / "published"
    publish_post_core_release_v1(config, core, task14, output)
    artifact = output / "model_registry.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(PostCoreReleaseError, match="canonical|artifact|index|binding"):
        verify_post_core_release_v1(output, config, core, task14)


def test_source_output_overlap_and_reparse_are_rejected(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    with pytest.raises((ValueError, PostCoreReleaseError, FileExistsError), match="overlap|source|exists"):
        publish_post_core_release_v1(config, core, task14, core)
    with pytest.raises((ValueError, PostCoreReleaseError, FileExistsError), match="overlap|source|exists"):
        publish_post_core_release_v1(config, core, task14, task14.parent)


def test_cli_help_and_unsafe_flags() -> None:
    for script in ("scripts/vnext_prepare_post_core_release.py", "scripts/vnext_qualify_post_core_models.py"):
        help_run = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert help_run.returncode == 0
        assert "--allow-network" not in help_run.stdout
        unsafe = subprocess.run([sys.executable, script, "--allow-network"], capture_output=True, text=True)
        assert unsafe.returncode != 0


def test_prepare_and_qualification_clis_emit_secret_free_json(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = _config(tmp_path, task14)
    output = tmp_path / "published"
    command = [
        sys.executable,
        "scripts/vnext_prepare_post_core_release.py",
        "--config", str(config),
        "--core-manifest", str(core),
        "--task14-index", str(task14),
        "--output-root", str(output),
        "--execute",
    ]
    prepared = subprocess.run(command, capture_output=True, text=True)
    assert prepared.returncode == 0
    summary = json.loads(prepared.stdout.strip().splitlines()[-1])
    assert summary["status"] == "SUCCESS_WITH_PENDING"
    assert summary["provider_calls"] == 0
    assert "secret" not in prepared.stdout.lower()

    qualified = subprocess.run(
        [
            sys.executable,
            "scripts/vnext_qualify_post_core_models.py",
            "--config", str(config),
            "--core-manifest", str(core),
            "--task14-index", str(task14),
            "--execute",
        ],
        capture_output=True,
        text=True,
    )
    assert qualified.returncode == 0
    assert json.loads(qualified.stdout.strip().splitlines()[-1])["status"] == "SUCCESS_WITH_PENDING"


def test_phase0_paths_contain_no_forbidden_runtime_imports_or_calls() -> None:
    root = Path(__file__).parents[2]
    paths = [
        root / "mub" / "vnext" / "post_core" / "release_v1.py",
        root / "scripts" / "vnext_prepare_post_core_release.py",
        root / "scripts" / "vnext_qualify_post_core_models.py",
    ]
    forbidden = ("import socket", "import requests", "import httpx", "import subprocess", "boto3", "openai", "anthropic", "google.generativeai", "transformers")
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(token in source for token in forbidden), path


def test_cli_stale_source_exit_code(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = _config(tmp_path, task14)
    task14.write_bytes(task14.read_bytes() + b"x")
    run = subprocess.run(
        [
            sys.executable,
            "scripts/vnext_prepare_post_core_release.py",
            "--config", str(config),
            "--core-manifest", str(core),
            "--task14-index", str(task14),
            "--output-root", str(tmp_path / "out"),
            "--execute",
        ], capture_output=True, text=True,
    )
    assert run.returncode == EXIT_STALE_SOURCE
    assert "secret" not in (run.stdout + run.stderr).lower()


def test_qualification_cli_stale_source_exit_code(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = _config(tmp_path, task14)
    task14.write_bytes(task14.read_bytes() + b"x")
    run = subprocess.run(
        [
            sys.executable,
            "scripts/vnext_qualify_post_core_models.py",
            "--config", str(config),
            "--core-manifest", str(core),
            "--task14-index", str(task14),
            "--execute",
        ], capture_output=True, text=True,
    )
    assert run.returncode == EXIT_STALE_SOURCE
    assert "secret" not in (run.stdout + run.stderr).lower()


def test_tracked_config_is_canonical_and_pins_both_immutable_sources() -> None:
    raw = CONFIG.read_bytes()
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode()
    config = load_post_core_config_v1(CONFIG)
    assert config.core_manifest_sha256 == "dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b"
    assert config.core_task14_index_sha256 == TASK14_SHA


def test_config_rejects_caller_selected_immutable_source_hash(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["core_manifest_sha256"] = "0" * 64
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="frozen|hash"):
        load_post_core_config_v1(alternate)


def test_capability_probe_contract_records_zero_network_calls() -> None:
    probe = CapabilityProbeReportV1(rows=())
    assert probe.network_calls == 0


def test_registry_validation_requires_frozen_candidate_semantics(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    registry = dict(build_initial_model_registry_v1())
    candidate = registry["qwen35_9b_bf16"]
    registry["qwen35_9b_bf16"] = candidate.model_copy(update={"scopes": ("none",)})
    with pytest.raises(ValueError, match="frozen|initial|semantics"):
        build_post_core_release_v1(
            load_post_core_config_v1(CONFIG),
            core,
            task14,
            registry=registry,
        )
