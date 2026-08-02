from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.vnext_run_pilot as cli
from mub.vnext.adapters.heuristic_crud import HeuristicCrudAdapter


def _completed(*, returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_runtime_revision_and_clean_state_are_resolved_from_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        if command == ["git", "rev-parse", "HEAD"]:
            return _completed(stdout="formal-run-revision\n")
        if command == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return _completed(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._resolve_code_revision() == "formal-run-revision"
    assert cli._tracked_tree_is_dirty() is False
    assert calls == [
        (["git", "rev-parse", "HEAD"], cli.PROJECT_ROOT),
        (["git", "status", "--porcelain", "--untracked-files=no"], cli.PROJECT_ROOT),
    ]


def test_runtime_dirty_state_check_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _completed(stdout=" M mub/vnext/runtime/run.py\n"),
    )

    assert cli._tracked_tree_is_dirty() is True


def test_formal_main_rejects_tracked_dirty_tree_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "_resolve_code_revision", lambda: "formal-revision")
    monkeypatch.setattr(cli, "_tracked_tree_is_dirty", lambda: True)
    monkeypatch.setattr(
        cli,
        "read_models",
        lambda *args, **kwargs: pytest.fail("dirty source must not read tasks"),
    )

    output_dir = tmp_path / "run"
    status = cli.main(
        [
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--task-manifest",
            str(tmp_path / "task_manifest.json"),
            "--adapter",
            "reference",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    assert not output_dir.exists()


def test_formal_main_passes_authenticated_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_task,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "task_manifest.json"
    tasks_path.write_text("placeholder\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli, "_resolve_code_revision", lambda: "formal-revision")
    monkeypatch.setattr(cli, "_tracked_tree_is_dirty", lambda: False)
    monkeypatch.setattr(cli, "read_models", lambda *args, **kwargs: [make_task()])
    monkeypatch.setattr(
        cli,
        "TaskManifest",
        SimpleNamespace(
            model_validate_json=lambda payload: SimpleNamespace(
                task_file_paths_and_hashes=(
                    SimpleNamespace(
                        sha256=cli.hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
                        record_count=1,
                    ),
                )
            )
        ),
    )

    def fake_run_tasks(tasks, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    status = cli.main(
        [
            "--tasks",
            str(tasks_path),
            "--task-manifest",
            str(manifest_path),
            "--adapter",
            "reference",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert status == 0
    runtime_config = captured["run_config"]
    assert runtime_config.code_revision == "formal-revision"
    assert runtime_config.dirty_state is False
    assert runtime_config.compiler_version == cli.COMPILER_VERSION
    assert runtime_config.profile_version == cli.PROFILE_VERSION
    assert runtime_config.schema_version == cli.SCHEMA_VERSION


def test_formal_main_rejects_unauthenticated_task_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_task,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "task_manifest.json"
    tasks_path.write_text("tampered\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_resolve_code_revision", lambda: "formal-revision")
    monkeypatch.setattr(cli, "_tracked_tree_is_dirty", lambda: False)
    monkeypatch.setattr(cli, "read_models", lambda *args, **kwargs: [make_task()])
    monkeypatch.setattr(
        cli,
        "TaskManifest",
        SimpleNamespace(
            model_validate_json=lambda payload: SimpleNamespace(
                task_file_paths_and_hashes=(
                    SimpleNamespace(sha256="0" * 64, record_count=1),
                )
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_tasks",
        lambda *args, **kwargs: pytest.fail("unauthenticated tasks must not run"),
    )

    status = cli.main(
        [
            "--tasks",
            str(tasks_path),
            "--task-manifest",
            str(manifest_path),
            "--adapter",
            "reference",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert status == 2
    assert not (tmp_path / "run").exists()


def test_heuristic_encoder_is_local_only_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_task,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    encoder = cli._load_offline_encoder(checkpoint, device="cuda:2")
    factory = cli._make_factory(
        "heuristic_crud",
        "latest_per_object",
        encoder=encoder,
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision="snapshot-revision",
    )
    left = factory(make_task())
    right = factory(make_task())

    assert isinstance(left, HeuristicCrudAdapter)
    assert left.encoder is encoder
    assert right.encoder is encoder
    assert left.retrieval_policy == "latest_per_object"
    assert left.adapter_info().system_version == "sentence-transformers/all-MiniLM-L6-v2"
    assert left.adapter_info().sdk_version == "snapshot-revision"
    assert calls == [
        (
            str(checkpoint.resolve()),
            {"device": "cuda:2", "local_files_only": True},
        )
    ]


def test_heuristic_requires_complete_local_encoder_arguments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="encoder-checkpoint"):
        cli._validate_encoder_arguments(
            SimpleNamespace(
                adapter="heuristic_crud",
                encoder_checkpoint=None,
                encoder_revision="snapshot-revision",
                encoder_model_id="sentence-transformers/all-MiniLM-L6-v2",
                encoder_device="cpu",
            )
        )

    with pytest.raises(ValueError, match="encoder-revision"):
        cli._validate_encoder_arguments(
            SimpleNamespace(
                adapter="heuristic_crud",
                encoder_checkpoint=checkpoint,
                encoder_revision=None,
                encoder_model_id="sentence-transformers/all-MiniLM-L6-v2",
                encoder_device="cpu",
            )
        )


def test_nonheuristic_adapter_rejects_encoder_options(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    invalid_options = (
        {"encoder_checkpoint": checkpoint},
        {"encoder_revision": "snapshot-revision"},
        {"encoder_model_id": "custom/encoder"},
        {"encoder_device": "cuda:0"},
    )
    defaults = {
        "adapter": "exact_crud",
        "encoder_checkpoint": None,
        "encoder_revision": None,
        "encoder_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "encoder_device": "cpu",
    }
    for options in invalid_options:
        with pytest.raises(ValueError, match="only valid with heuristic_crud"):
            cli._validate_encoder_arguments(
                SimpleNamespace(**(defaults | options))
            )
