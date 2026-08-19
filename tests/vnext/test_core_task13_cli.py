from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from mub.vnext.io import canonical_json_bytes
from mub.vnext.runtime.task12_execution_v3 import Task12RuntimeCodeBindingV1
from mub.vnext.statistics import DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1
from mub.vnext.statistics.task13_v3 import Task13RuntimeBindingV1
from tests.vnext.task13_input_fixtures import build_compact_authenticated_task13_fixture


RUNTIME = Task12RuntimeCodeBindingV1(
    code_revision="8" * 40,
    code_tree_sha256="9" * 64,
)


@pytest.fixture(scope="module")
def authenticated_fixture(tmp_path_factory):
    repository_root = Path(__file__).resolve().parents[2]
    return build_compact_authenticated_task13_fixture(
        tmp_path_factory.mktemp("task13-cli"),
        repository_root,
        RUNTIME,
    )


@pytest.fixture
def task13_arguments(tmp_path, authenticated_fixture):
    config_path = tmp_path / "statistics-config.json"
    config_path.write_bytes(canonical_json_bytes(DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1))
    inputs = authenticated_fixture["inputs"]
    return {
        "manifest": authenticated_fixture["preparation_manifest_path"],
        "plan": authenticated_fixture["plan_path"],
        "core_root": inputs["core_root"],
        "evidence_root": inputs["evidence_root"],
        "matrix_root": authenticated_fixture["matrix"].matrix_root,
        "matrix_bundle_manifest": authenticated_fixture["matrix_manifest_path"],
        "matrix_summary": authenticated_fixture["summary_path"],
        "matrix_integrity_audit": authenticated_fixture["audit_path"],
        "statistics_config": config_path,
        "output_root": tmp_path / "published-task13",
    }


@pytest.fixture
def fixed_runtime_binding(monkeypatch):
    import scripts.vnext_run_core_task13 as command

    monkeypatch.setattr(
        command.task13_publication,
        "current_clean_task13_runtime_v3",
        lambda repository_root: Task13RuntimeBindingV1("a" * 40, "b" * 64),
    )


def _cli_args(arguments: dict[str, Path], *, execute: bool = True) -> list[str]:
    args = [
        "--manifest", str(arguments["manifest"]),
        "--plan", str(arguments["plan"]),
        "--core-root", str(arguments["core_root"]),
        "--evidence-root", str(arguments["evidence_root"]),
        "--matrix-root", str(arguments["matrix_root"]),
        "--matrix-bundle-manifest", str(arguments["matrix_bundle_manifest"]),
        "--matrix-summary", str(arguments["matrix_summary"]),
        "--matrix-integrity-audit", str(arguments["matrix_integrity_audit"]),
        "--statistics-config", str(arguments["statistics_config"]),
        "--output-root", str(arguments["output_root"]),
    ]
    if execute:
        args.append("--execute")
    return args


def test_task13_cli_parser_is_execute_gated_and_has_exact_safe_flags():
    from scripts.vnext_run_core_task13 import build_parser

    parser = build_parser()
    option_names = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert option_names == {
        "--manifest", "--plan", "--core-root", "--evidence-root",
        "--matrix-root", "--matrix-bundle-manifest", "--matrix-summary",
        "--matrix-integrity-audit", "--statistics-config", "--output-root", "--execute",
    }
    forbidden = ("model", "provider", "token", "api", "fake", "metric", "override")
    assert not any(part in option.lower() for option in option_names for part in forbidden)
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_task13_cli_rejects_missing_execute(capsys):
    from scripts.vnext_run_core_task13 import main

    arguments = {
        "manifest": Path("manifest"), "plan": Path("plan"), "core_root": Path("core"),
        "evidence_root": Path("evidence"), "matrix_root": Path("matrix"),
        "matrix_bundle_manifest": Path("matrix-manifest"), "matrix_summary": Path("summary"),
        "matrix_integrity_audit": Path("audit"), "statistics_config": Path("config"),
        "output_root": Path("output"),
    }
    assert main(_cli_args(arguments, execute=False)) == 2
    assert "--execute" in capsys.readouterr().err


def test_task13_cli_publishes_exactly_eight_closed_artifacts(
    task13_arguments, fixed_runtime_binding, capsys
):
    from scripts.vnext_run_core_task13 import main
    from mub.vnext.statistics.task13_v3 import verify_task13_artifact_root_v3

    assert main(_cli_args(task13_arguments)) == 0
    output_root = task13_arguments["output_root"]
    assert sorted(path.name for path in output_root.iterdir()) == [
        "bootstrap_indices.bin", "case_index.json", "cases.jsonl", "cell_statistics.jsonl",
        "claim_ledger.jsonl", "paired_contrasts.jsonl", "statistics_receipt.json",
        "task13_artifact_index.json",
    ]
    result = verify_task13_artifact_root_v3(output_root)
    assert result.artifact_index.artifacts[-1].artifact.path == "claim_ledger.jsonl"
    stdout = capsys.readouterr().out
    assert hashlib.sha256((output_root / "task13_artifact_index.json").read_bytes()).hexdigest() in stdout
    assert str(output_root) in stdout


def test_task13_output_overlap_and_existing_root_are_rejected(task13_arguments):
    from scripts.vnext_run_core_task13 import main

    task13_arguments["output_root"].mkdir()
    assert main(_cli_args(task13_arguments)) == 2
    task13_arguments["output_root"].rmdir()
    task13_arguments["output_root"] = task13_arguments["core_root"] / "illegal-task13-output"
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()


def test_task13_compute_and_publish_failure_leave_no_final_and_only_clean_owned_staging(
    task13_arguments, fixed_runtime_binding, monkeypatch, tmp_path
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    foreign = tmp_path / ".mub-task13-stage-foreign"
    foreign.mkdir()
    monkeypatch.setattr(
        publication,
        "build_task13_publication_v3",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected compute failure")),
    )
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()
    assert foreign.exists()

    monkeypatch.undo()
    monkeypatch.setattr(
        publication,
        "_commit_staged_task13_root_v3",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected publish failure")),
    )
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()
    assert foreign.exists()
    assert tuple(tmp_path.glob(".mub-task13-stage-*")) == (foreign,)


def test_task13_cross_artifact_closure_tamper_is_rejected(
    task13_arguments, fixed_runtime_binding
):
    from scripts.vnext_run_core_task13 import main
    from mub.vnext.statistics.task13_v3 import verify_task13_artifact_root_v3

    assert main(_cli_args(task13_arguments)) == 0
    artifact = task13_arguments["output_root"] / "cell_statistics.jsonl"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical|hash|binding|closure"):
        verify_task13_artifact_root_v3(task13_arguments["output_root"])


def test_task13_final_directory_no_replace_failure_never_clobbers(
    task13_arguments, fixed_runtime_binding, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    monkeypatch.setattr(
        publication,
        "_directory_commit_noreplace_v3",
        lambda staging, final_root: (_ for _ in ()).throw(FileExistsError("injected race")),
    )
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()


def test_task13_refuses_when_no_safe_posix_noreplace_primitive_exists(monkeypatch, tmp_path):
    import mub.vnext.statistics.task13_v3 as publication

    staging = tmp_path / "staging"
    final_root = tmp_path / "final"
    staging.mkdir()

    class NoRenameAt2:
        pass

    monkeypatch.setattr(publication.os, "name", "posix")
    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *args, **kwargs: NoRenameAt2())
    with pytest.raises(RuntimeError, match="no safe no-replace"):
        publication._directory_commit_noreplace_v3(staging, final_root)
    assert staging.exists()
    assert not final_root.exists()


# Task 7 remediation contracts: publication safety boundaries.
def test_task13_success_does_not_call_postrename_verifier(
    task13_arguments, fixed_runtime_binding, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    original = publication.verify_task13_artifact_root_v3
    final_root = task13_arguments["output_root"]

    def reject_only_final(root):
        if Path(root) == final_root:
            raise AssertionError("post-rename verifier was called")
        return original(root)

    monkeypatch.setattr(publication, "verify_task13_artifact_root_v3", reject_only_final)
    assert main(_cli_args(task13_arguments)) == 0
    assert final_root.is_dir()


def test_task13_snapshot_rejects_recursive_membership_changes(tmp_path):
    from mub.vnext.statistics.task13_v3 import (
        _revalidate_source_snapshot,
        capture_task13_source_snapshot_v3,
    )

    root = tmp_path / "root"
    root.mkdir()
    source = root / "input.json"
    source.write_bytes(b"{}")
    snapshot = capture_task13_source_snapshot_v3((source,), (root,))
    (root / "new-file").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="membership"):
        _revalidate_source_snapshot(snapshot)


@pytest.mark.parametrize("mutation", ("delete", "rename", "directory"))
def test_task13_snapshot_rejects_all_recursive_membership_changes(tmp_path, mutation):
    from mub.vnext.statistics.task13_v3 import (
        _revalidate_source_snapshot,
        capture_task13_source_snapshot_v3,
    )

    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "input.json"
    source.write_bytes(b"{}")
    snapshot = capture_task13_source_snapshot_v3((source,), (root,))
    if mutation == "delete":
        source.unlink()
    elif mutation == "rename":
        source.rename(nested / "renamed.json")
    else:
        (root / "added-directory").mkdir()
    with pytest.raises(RuntimeError, match="membership"):
        _revalidate_source_snapshot(snapshot)


def test_task13_snapshot_rejects_explicit_input_replacement(tmp_path):
    from mub.vnext.statistics.task13_v3 import (
        _revalidate_source_snapshot,
        capture_task13_source_snapshot_v3,
    )

    root = tmp_path / "root"
    root.mkdir()
    source = root / "input.json"
    source.write_bytes(b'{"v":1}')
    snapshot = capture_task13_source_snapshot_v3((source,), (root,))
    source.write_bytes(b'{"v":2}')
    with pytest.raises(RuntimeError, match="membership|source changed"):
        _revalidate_source_snapshot(snapshot)


def test_task13_cli_revalidates_snapshot_after_loader_before_compute(
    monkeypatch, tmp_path, fixed_runtime_binding
):
    import scripts.vnext_run_core_task13 as command

    roots = {name: tmp_path / name for name in ("core", "evidence", "matrix")}
    for root in roots.values():
        root.mkdir()
        (root / "member").write_bytes(b"member")
    files = {name: tmp_path / f"{name}.json" for name in (
        "manifest", "plan", "matrix_manifest", "matrix_summary", "integrity_audit", "statistics_config"
    )}
    for path in files.values():
        path.write_bytes(b"{}")
    output = tmp_path / "output"

    def mutate_loader(**kwargs):
        files["manifest"].write_bytes(b'{"changed":true}')
        return object()

    called = False

    def must_not_compute(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("compute reached after source mutation")

    monkeypatch.setattr(command, "load_task13_authenticated_matrix_v1", mutate_loader)
    monkeypatch.setattr(command.task13_publication, "build_task13_publication_v3", must_not_compute)
    arguments = {
        "manifest": files["manifest"], "plan": files["plan"],
        "core_root": roots["core"], "evidence_root": roots["evidence"],
        "matrix_root": roots["matrix"], "matrix_bundle_manifest": files["matrix_manifest"],
        "matrix_summary": files["matrix_summary"], "matrix_integrity_audit": files["integrity_audit"],
        "statistics_config": files["statistics_config"], "output_root": output,
    }
    assert command.main(_cli_args(arguments)) == 2
    assert not called
    assert not output.exists()


def test_task13_commit_race_and_exdev_preserve_contract(tmp_path, monkeypatch):
    import mub.vnext.statistics.task13_v3 as publication

    root = tmp_path / "sources"
    root.mkdir()
    source = root / "source"
    source.write_bytes(b"source")
    snapshot = publication.capture_task13_source_snapshot_v3((source,), (root,))
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))
    staging = parent / "stage"
    staging.mkdir()
    final_root = parent / "final"
    final_root.mkdir()

    with pytest.raises(FileExistsError):
        publication._commit_staged_task13_root_v3(
            staging=staging, final_root=final_root,
            parent_identity=parent_identity, source_snapshot=snapshot,
        )
    assert final_root.is_dir()
    assert staging.is_dir()

    final_root.rmdir()
    monkeypatch.setattr(
        publication,
        "_directory_commit_noreplace_v3",
        lambda staging, final_root: (_ for _ in ()).throw(OSError(errno.EXDEV, "injected EXDEV")),
    )
    with pytest.raises(OSError, match="EXDEV"):
        publication._commit_staged_task13_root_v3(
            staging=staging, final_root=final_root,
            parent_identity=parent_identity, source_snapshot=snapshot,
        )
    assert staging.is_dir()
    assert not final_root.exists()


def test_task13_precommit_validator_failure_leaves_no_final_root(
    task13_arguments, fixed_runtime_binding, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    monkeypatch.setattr(
        publication,
        "validate_task13_staging_root_v3",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("injected precommit validation failure")),
    )
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()


def test_task13_publish_exdev_retains_owned_stage(
    task13_arguments, fixed_runtime_binding, monkeypatch, tmp_path
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    monkeypatch.setattr(
        publication,
        "_directory_commit_noreplace_v3",
        lambda staging, final_root: (_ for _ in ()).throw(OSError(errno.EXDEV, "injected EXDEV")),
    )
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()
    owned = tuple(path for path in tmp_path.glob(".mub-task13-stage-*") if path.is_dir())
    assert len(owned) == 1
    assert sorted(path.name for path in owned[0].iterdir()) == [
        "bootstrap_indices.bin", "case_index.json", "cases.jsonl", "cell_statistics.jsonl",
        "claim_ledger.jsonl", "paired_contrasts.jsonl", "statistics_receipt.json",
        "task13_artifact_index.json",
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows MoveFileExW no-replace behavior")
def test_task13_windows_noreplace_collision_preserves_existing_final(tmp_path):
    from mub.vnext.statistics.task13_v3 import _directory_commit_noreplace_v3

    staging = tmp_path / "stage"
    final_root = tmp_path / "final"
    staging.mkdir()
    final_root.mkdir()
    with pytest.raises(FileExistsError):
        _directory_commit_noreplace_v3(staging, final_root)
    assert staging.is_dir()
    assert final_root.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX renameat2 no-replace behavior")
def test_task13_posix_noreplace_collision_preserves_existing_final(tmp_path):
    from mub.vnext.statistics.task13_v3 import _directory_commit_noreplace_v3

    staging = tmp_path / "stage"
    final_root = tmp_path / "final"
    staging.mkdir()
    final_root.mkdir()
    with pytest.raises(FileExistsError):
        _directory_commit_noreplace_v3(staging, final_root)
    assert staging.is_dir()
    assert final_root.is_dir()
