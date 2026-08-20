from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mub.vnext.release.task14_sources import TASK14_EXPECTED_TASK13_INDEX_SHA256
from mub.vnext.statistics.task13_v3 import Task13RuntimeBindingV1
from scripts import vnext_review_core_task14 as command


def args(tmp_path: Path, *, execute: bool = True) -> list[str]:
    values = [
        "--core-root", str(tmp_path / "core"),
        "--evidence-root", str(tmp_path / "evidence"),
        "--task13-root", str(tmp_path / "task13"),
        "--task13-audit", str(tmp_path / "task13-audit.json"),
        "--remote-task13-staging", "/NAS/project/.mub-task13-stage-fixture",
        "--expected-task13-index-sha256", TASK14_EXPECTED_TASK13_INDEX_SHA256,
        "--source-revision", "a" * 40,
        "--review-id", "core-task14-test",
        "--output-root", str(tmp_path / "output"),
    ]
    if execute:
        values.append("--execute")
    return values


def install_success(monkeypatch, tmp_path: Path, *, approved: bool = True) -> None:
    monkeypatch.setattr(
        command,
        "current_clean_task13_runtime_v3",
        lambda root: Task13RuntimeBindingV1("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(command, "load_task14_sources_v1", lambda paths: object())
    monkeypatch.setattr(
        command,
        "publish_task14_review_v1",
        lambda *args, **kwargs: SimpleNamespace(
            final_approved=approved,
            index_sha256="c" * 64,
            attestation_sha256="d" * 64,
            output_root=(tmp_path / "output").resolve(),
        ),
    )


def test_parser_has_exact_safe_surface() -> None:
    parser = command.build_parser()
    options = {
        value
        for action in parser._actions
        for value in action.option_strings
        if value.startswith("--") and value != "--help"
    }
    assert options == {
        "--core-root", "--evidence-root", "--task13-root", "--task13-audit",
        "--remote-task13-staging", "--expected-task13-index-sha256",
        "--source-revision", "--review-id", "--output-root", "--execute",
    }
    forbidden = ("model", "provider", "token", "api", "fake", "offline", "slot-direct", "override", "force")
    assert not any(term in option for term in forbidden for option in options)
    with pytest.raises(SystemExit):
        parser.parse_args(["--exec"])


def test_help_returns_zero(capsys) -> None:
    assert command.main(["--help"]) == command.EXIT_APPROVED
    assert "Core Task 14" in capsys.readouterr().out


def test_execute_gate_returns_usage(tmp_path: Path, capsys) -> None:
    assert command.main(args(tmp_path, execute=False), repository_root=tmp_path) == command.EXIT_USAGE
    assert "--execute" in capsys.readouterr().err


def test_approved_and_not_approved_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    install_success(monkeypatch, tmp_path, approved=True)
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_APPROVED
    assert "decision=FINAL_APPROVED" in capsys.readouterr().out
    install_success(monkeypatch, tmp_path, approved=False)
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_NOT_APPROVED
    assert "decision=NOT_APPROVED" in capsys.readouterr().out


def test_wrong_task13_hash_is_usage_error(tmp_path: Path) -> None:
    values = args(tmp_path)
    index = values.index("--expected-task13-index-sha256") + 1
    values[index] = "f" * 64
    assert command.main(values, repository_root=tmp_path) == command.EXIT_USAGE


def test_dirty_runtime_is_exit_14(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        command,
        "current_clean_task13_runtime_v3",
        lambda root: (_ for _ in ()).throw(RuntimeError("dirty")),
    )
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_UNTRUSTED_RUNTIME


def test_stale_source_and_existing_output_have_distinct_codes(
    tmp_path: Path, monkeypatch
) -> None:
    install_success(monkeypatch, tmp_path)
    monkeypatch.setattr(
        command,
        "publish_task14_review_v1",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("source snapshot changed")),
    )
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_STALE_SOURCE
    monkeypatch.setattr(
        command,
        "publish_task14_review_v1",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileExistsError("exists")),
    )
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_PUBLICATION
    monkeypatch.setattr(
        command,
        "publish_task14_review_v1",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("reopened final root differs")),
    )
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_PUBLICATION
    monkeypatch.setattr(
        command,
        "publish_task14_review_v1",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("runtime changed during publication")),
    )
    assert command.main(args(tmp_path), repository_root=tmp_path) == command.EXIT_UNTRUSTED_RUNTIME
