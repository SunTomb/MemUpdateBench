from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from mub.vnext.contracts import ArtifactRef
from mub.vnext.generation import PublishedPilotBundle
from scripts import vnext_generate_pilot as cli


_ARTIFACT_NAMES = (
    "tasks.jsonl",
    "generation_config.json",
    "split_balance.json",
    "task_manifest.json",
    "validation_report.json",
)
_GIT_SELECTION_ENV = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
}


def _published(output_dir: Path) -> PublishedPilotBundle:
    refs = tuple(
        ArtifactRef(
            path=name,
            sha256=f"{index:x}" * 64,
            media_type=(
                "application/x-ndjson" if index == 0 else "application/json"
            ),
            record_count=3 if index == 0 else 1,
        )
        for index, name in enumerate(_ARTIFACT_NAMES)
    )
    return PublishedPilotBundle(
        output_dir=output_dir,
        artifact_paths=tuple(output_dir / name for name in _ARTIFACT_NAMES),
        artifact_refs=refs,
    )


def _completed(returncode: int = 0, stdout: str = "abc123\n"):
    return subprocess.CompletedProcess(
        ["git", "rev-parse", "HEAD"], returncode, stdout, ""
    )


def test_main_passes_exact_inputs_and_prints_canonical_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pilot.yaml"
    output_dir = tmp_path / "release"
    config = SimpleNamespace(
        release_id="pilot-é",
        total_tasks=3,
        expected_split_tasks={"train": 1, "dev": 1, "test": 1},
        raw_payload="TASK CONTENT SECRET",
    )
    calls: dict[str, object] = {}
    for name in _GIT_SELECTION_ENV:
        monkeypatch.setenv(name, "FOREIGN REPOSITORY SECRET")
    monkeypatch.setenv("MUB_ENV_PRESERVED", "present")

    def fake_run(*args, **kwargs):
        run_env = kwargs["env"]
        assert run_env is not os.environ
        assert run_env["MUB_ENV_PRESERVED"] == "present"
        run_keys = {name.upper() for name in run_env}
        assert not (_GIT_SELECTION_ENV & run_keys)
        safe_kwargs = {**kwargs, "env": "<sanitized>"}
        calls["run"] = (args, safe_kwargs)
        return _completed()

    def fake_load(path: Path):
        calls["load"] = path
        return config

    def fake_build(*args, **kwargs):
        calls["build"] = (args, kwargs)
        return _published(output_dir)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "load_pilot_config", fake_load)
    monkeypatch.setattr(cli, "build_pilot", fake_build)

    status = cli.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    expected = {
        "artifact_refs": [ref.model_dump(mode="json") for ref in _published(output_dir).artifact_refs],
        "code_revision": "abc123",
        "output_dir": str(output_dir),
        "release_id": "pilot-é",
        "split_counts": {"train": 1, "dev": 1, "test": 1},
        "task_count": 3,
    }
    assert status == 0
    assert captured.out == json.dumps(
        expected,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    assert captured.err == ""
    assert "TASK CONTENT SECRET" not in captured.out
    assert calls == {
        "run": (
            (["git", "rev-parse", "HEAD"],),
            {
                "capture_output": True,
                "text": True,
                "check": False,
                "cwd": cli.PROJECT_ROOT,
                "env": "<sanitized>",
            },
        ),
        "load": config_path,
        "build": (
            (config, output_dir),
            {"code_revision": "abc123", "overwrite": True},
        ),
    }
    assert [item["path"] for item in json.loads(captured.out)["artifact_refs"]] == list(
        _ARTIFACT_NAMES
    )


def test_revision_resolution_is_anchored_outside_project_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    caller_dir = tmp_path / "caller-repository"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)

    def fake_run(*args, **kwargs):
        assert Path.cwd() == caller_dir
        assert kwargs["cwd"] == cli.PROJECT_ROOT
        return _completed(stdout="project-revision\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._resolve_code_revision() == "project-revision"


def test_revision_resolution_ignores_foreign_git_dir_with_real_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    foreign = tmp_path / "foreign-repository"
    subprocess.run(
        ["git", "init", "-q", str(foreign)],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(foreign),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "foreign",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    project_revision = cli._resolve_code_revision()
    foreign_revision = subprocess.run(
        ["git", "-C", str(foreign), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert foreign_revision != project_revision

    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.chdir(foreign)

    assert cli._resolve_code_revision() == project_revision


@pytest.mark.parametrize("failure", ["nonzero", "missing", "blank"])
def test_revision_failure_is_safe_and_skips_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    failure: str,
) -> None:
    def fake_run(*args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("SECRET executable detail")
        if failure == "blank":
            return _completed(stdout=" \n")
        return _completed(returncode=1, stdout="SECRET revision payload")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli, "load_pilot_config", lambda path: pytest.fail("config must not load")
    )
    monkeypatch.setattr(cli, "build_pilot", lambda *args, **kwargs: pytest.fail("no build"))

    status = cli.main(
        ["--config", str(tmp_path / "pilot.yaml"), "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err == "error: could not resolve code revision\n"
    assert len(captured.err) <= 128
    assert "SECRET" not in captured.err


def test_build_failure_is_safe_and_has_no_success_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(release_id="release-secret")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _completed())
    monkeypatch.setattr(cli, "load_pilot_config", lambda path: config)

    def fail_build(*args, **kwargs):
        raise ValueError("TASK CONTENT SECRET")

    monkeypatch.setattr(cli, "build_pilot", fail_build)

    status = cli.main(
        ["--config", str(tmp_path / "pilot.yaml"), "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err == "error: pilot generation failed\n"
    assert "SECRET" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [[], ["--config", "pilot.yaml"], ["--output-dir", "release"]],
)
def test_required_arguments_return_nonzero_without_starting_stages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("no subprocess")
    )

    status = cli.main(argv)

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err == "error: invalid command-line arguments\n"


def test_import_has_no_generation_or_subprocess_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mub.vnext.generation as generation

    def fail(*args, **kwargs):
        pytest.fail("import must not execute CLI stages")

    monkeypatch.setattr(generation, "load_pilot_config", fail)
    monkeypatch.setattr(generation, "build_pilot", fail)
    monkeypatch.setattr(subprocess, "run", fail)
    script = Path(cli.__file__)
    spec = importlib.util.spec_from_file_location("pilot_cli_import_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
