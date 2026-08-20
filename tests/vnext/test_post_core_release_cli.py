from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mub.vnext.post_core.contracts_v1 import POST_CORE_ARTIFACT_ORDER
from mub.vnext.post_core.release_v1 import (
    EXIT_BLOCKED,
    EXIT_PUBLICATION,
    EXIT_STALE_SOURCE,
    PostCoreReleaseError,
    build_post_core_release_v1,
    load_post_core_config_v1,
    publish_post_core_release_v1,
    verify_post_core_release_v1,
)
from mub.vnext.post_core.model_registry_v1 import build_initial_model_registry_v1


CONFIG = Path(__file__).parents[2] / "configs" / "vnext" / "post_core" / "release_v1.json"
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


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    core = tmp_path / "core_manifest.json"
    task14 = tmp_path / "core_final_root_index.json"
    core.write_bytes(b'{"schema_version":"memupdatebench.core.manifest.v3","release":"frozen"}')
    task14.write_bytes(b'{"schema_version":"memupdatebench.core-task14-index.v1","artifacts":[]}')
    return core, task14


def _config(tmp_path: Path, task14: Path) -> Path:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data["core_task14_index_sha256"] = hashlib.sha256(task14.read_bytes()).hexdigest()
    path = tmp_path / "release.json"
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
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


def test_tampered_artifact_fails_exact_reopen_verification(tmp_path: Path) -> None:
    core, task14 = _sources(tmp_path)
    config = load_post_core_config_v1(_config(tmp_path, task14))
    output = tmp_path / "published"
    publish_post_core_release_v1(config, core, task14, output)
    artifact = output / "model_registry.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(PostCoreReleaseError, match="canonical|artifact|index"):
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
