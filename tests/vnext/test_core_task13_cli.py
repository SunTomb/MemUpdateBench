from __future__ import annotations

import errno
import hashlib
from dataclasses import replace
import os
import subprocess
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


def test_task13_runtime_binding_hashes_raw_nul_tree_and_rejects_untracked(tmp_path):
    from mub.vnext.statistics.task13_v3 import current_clean_task13_runtime_v3

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.email", "task13@example.test"), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.name", "Task 13"), check=True)
    (repository / "tracked.txt").write_bytes(b"tracked\n")
    subprocess.run(("git", "-C", str(repository), "add", "tracked.txt"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "initial"), check=True)

    binding = current_clean_task13_runtime_v3(repository)
    expected_revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    raw_tree = subprocess.run(
        ("git", "-C", str(repository), "ls-tree", "-r", "-z", "HEAD"),
        check=True, capture_output=True,
    ).stdout
    assert binding.runtime_revision == expected_revision
    assert binding.runtime_tree_sha256 == hashlib.sha256(raw_tree).hexdigest()

    (repository / "untracked.py").write_bytes(b"print('untracked')\n")
    with pytest.raises(RuntimeError, match="clean repository"):
        current_clean_task13_runtime_v3(repository)


def test_task13_parser_rejects_execute_abbreviation_without_output(tmp_path):
    from scripts.vnext_run_core_task13 import build_parser

    parser = build_parser()
    required = [
        "--manifest", "m", "--plan", "p", "--core-root", "c", "--evidence-root", "e",
        "--matrix-root", "r", "--matrix-bundle-manifest", "bm", "--matrix-summary", "s",
        "--matrix-integrity-audit", "a", "--statistics-config", "sc", "--output-root", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--exec"])
    assert not (tmp_path / "out").exists()


def test_task13_cleanup_preserves_substituted_expected_stage_member(tmp_path):
    import mub.vnext.statistics.task13_v3 as publication

    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / "staging"
    staging.mkdir()
    expected = ("one", "two")
    for name in expected:
        (staging / name).write_bytes(name.encode("utf-8"))
    staging_identity = publication._DirectoryIdentity(staging, publication._identity(staging))
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))
    ownership = publication._capture_staging_ownership_v3(staging, expected)

    replaced = staging / "one"
    replaced.unlink()
    replaced.write_bytes(b"foreign replacement")
    with pytest.raises(RuntimeError, match="owned staging|preserving"):
        publication._safe_cleanup_staging_v3(
            staging, staging_identity, parent_identity, ownership
        )
    assert staging.is_dir()


def test_task13_bootstrap_verifier_rejects_forged_200000_byte_payload():
    import mub.vnext.statistics.task13_v3 as publication

    with pytest.raises(ValueError, match="frozen bootstrap"):
        publication._validate_frozen_bootstrap_bytes_v3(b"x" * 200_000)


def test_task13_commit_rejects_postrename_path_substitution(tmp_path, monkeypatch):
    import mub.vnext.statistics.task13_v3 as publication

    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "source"
    source.write_bytes(b"source")
    snapshot = publication.capture_task13_source_snapshot_v3((source,), (sources,))
    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / "staging"
    staging.mkdir()
    final_root = parent / "final"
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))

    def install_then_substitute(stage, final):
        stage.rename(final)
        final.rmdir()
        final.mkdir()

    monkeypatch.setattr(publication, "_directory_commit_noreplace_v3", install_then_substitute)
    with pytest.raises(RuntimeError, match="committed-path-substitution"):
        publication._commit_staged_task13_root_v3(
            staging=staging,
            final_root=final_root,
            parent_identity=parent_identity,
            source_snapshot=snapshot,
        )
    assert final_root.is_dir()


def test_task13_cli_rejects_postcommit_path_substitution_without_deleting_final(
    task13_arguments, fixed_runtime_binding, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    def install_then_substitute(stage, final):
        stage.rename(final)
        for member in final.iterdir():
            member.unlink()
        final.rmdir()
        final.mkdir()
        (final / "foreign").write_bytes(b"foreign")

    monkeypatch.setattr(publication, "_directory_commit_noreplace_v3", install_then_substitute)
    assert main(_cli_args(task13_arguments)) == 2
    final_root = task13_arguments["output_root"]
    assert final_root.is_dir()
    assert (final_root / "foreign").read_bytes() == b"foreign"


def test_task13_direct_publish_rejects_unsealed_forged_publication(
    task13_arguments, fixed_runtime_binding, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from scripts.vnext_run_core_task13 import main

    forged = object.__new__(publication.Task13PublicationV1)
    monkeypatch.setattr(publication, "build_task13_publication_v3", lambda **kwargs: forged)
    assert main(_cli_args(task13_arguments)) == 2
    assert not task13_arguments["output_root"].exists()


def test_task13_runtime_tree_is_bound_to_captured_revision_during_head_race(tmp_path, monkeypatch):
    import mub.vnext.statistics.task13_v3 as publication

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.email", "task13@example.test"), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.name", "Task 13"), check=True)
    tracked = repository / "tracked.txt"
    tracked.write_bytes(b"first\n")
    subprocess.run(("git", "-C", str(repository), "add", "tracked.txt"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "first"), check=True)
    first_revision = subprocess.run(("git", "-C", str(repository), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
    first_tree = subprocess.run(("git", "-C", str(repository), "ls-tree", "-r", "-z", first_revision), check=True, capture_output=True).stdout

    original_run = publication.subprocess.run
    moved = False

    def moving_run(args, *args_tail, **kwargs):
        nonlocal moved
        result = original_run(args, *args_tail, **kwargs)
        if tuple(args[-2:]) == ("rev-parse", "HEAD") and not moved:
            moved = True
            tracked.write_bytes(b"second\n")
            original_run(("git", "-C", str(repository), "add", "tracked.txt"), check=True)
            original_run(("git", "-C", str(repository), "commit", "-qm", "second"), check=True)
        return result

    monkeypatch.setattr(publication.subprocess, "run", moving_run)
    binding = publication.current_clean_task13_runtime_v3(repository)
    assert binding.runtime_revision == first_revision
    assert binding.runtime_tree_sha256 == hashlib.sha256(first_tree).hexdigest()


def test_task13_precommit_ownership_recheck_rejects_changed_same_name(tmp_path):
    import mub.vnext.statistics.task13_v3 as publication

    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / "staging"
    staging.mkdir()
    (staging / "one").write_bytes(b"one")
    ownership = publication._capture_staging_ownership_v3(staging, ("one",))
    (staging / "one").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ownership|hash"):
        publication._validate_staging_ownership_v3(staging, ownership)
    assert (staging / "one").read_bytes() == b"changed"


def test_task13_directory_fsync_failure_preserves_committed_final(tmp_path, monkeypatch):
    import mub.vnext.statistics.task13_v3 as publication

    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "source"
    source.write_bytes(b"source")
    snapshot = publication.capture_task13_source_snapshot_v3((source,), (sources,))
    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / "staging"
    staging.mkdir()
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))
    monkeypatch.setattr(publication, "_fsync_parent_directory_v3", lambda parent: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(RuntimeError, match="committed|durability"):
        publication._commit_staged_task13_root_v3(
            staging=staging, final_root=parent / "final",
            parent_identity=parent_identity, source_snapshot=snapshot,
        )
    assert (parent / "final").is_dir()


def test_task13_renameat2_uses_syscall_fallback_when_symbol_missing(monkeypatch, tmp_path):
    import mub.vnext.statistics.task13_v3 as publication

    calls = []

    class Syscall:
        restype = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    class Libc:
        syscall = Syscall()

    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *args, **kwargs: Libc())
    monkeypatch.setattr(publication.os, "name", "posix")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    publication._renameat2_noreplace_v3(tmp_path / "stage", tmp_path / "final")


@pytest.mark.skipif(os.name != "nt", reason="Windows 8.3 alias protection")
def test_task13_short_path_alias_cannot_bypass_output_overlap(tmp_path):
    import ctypes
    import mub.vnext.statistics.task13_v3 as publication

    protected = tmp_path / "protected root with long name"
    protected.mkdir()
    size = ctypes.windll.kernel32.GetShortPathNameW(str(protected), None, 0)
    if not size:
        pytest.skip("8.3 short paths unavailable on this volume")
    buffer = ctypes.create_unicode_buffer(size)
    ctypes.windll.kernel32.GetShortPathNameW(str(protected), buffer, size)
    short_root = Path(buffer.value)
    with pytest.raises(ValueError, match="overlaps"):
        publication._assert_nonoverlap(short_root / "output", (protected,))



def test_loader_registry_rejects_same_object_content_mutation(authenticated_fixture):
    from mub.vnext.statistics.input_v3 import (
        load_task13_authenticated_matrix_v1,
        require_loader_registered_task13_matrix_v1,
    )

    inputs = authenticated_fixture["inputs"]
    paths = {
        "manifest": authenticated_fixture["preparation_manifest_path"],
        "plan": authenticated_fixture["plan_path"],
        "matrix_manifest": authenticated_fixture["matrix_manifest_path"],
        "matrix_summary": authenticated_fixture["summary_path"],
        "integrity_audit": authenticated_fixture["audit_path"],
    }
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    matrix = load_task13_authenticated_matrix_v1(
        preparation_manifest_path=paths["manifest"], plan_path=paths["plan"],
        core_root=inputs["core_root"], evidence_root=inputs["evidence_root"],
        matrix_root=authenticated_fixture["matrix"].matrix_root,
        matrix_manifest_path=paths["matrix_manifest"], matrix_summary_path=paths["matrix_summary"],
        integrity_audit_path=paths["integrity_audit"], repository_root=Path(__file__).resolve().parents[2],
        expected_preparation_manifest_sha256=hashes["manifest"], expected_plan_sha256=hashes["plan"],
        expected_matrix_manifest_sha256=hashes["matrix_manifest"], expected_matrix_summary_sha256=hashes["matrix_summary"],
        expected_integrity_audit_sha256=hashes["integrity_audit"],
    )
    capability = require_loader_registered_task13_matrix_v1(matrix)
    assert capability.matrix_digest == capability.digest
    assert set(capability.roots) == {"repository", "core", "evidence", "matrix"}
    assert set(capability.controls) == {
        "preparation_manifest", "plan", "matrix_manifest", "matrix_summary", "integrity_audit",
    }
    assert capability.repository_root == Path(__file__).resolve().parents[2]
    assert capability.core_root == inputs["core_root"].resolve()
    assert capability.matrix_root == authenticated_fixture["matrix"].matrix_root.resolve()
    assert all(len(entry.sha256) == 64 for entry in capability.controls.values())
    object.__setattr__(matrix, "canonical_core_ids", tuple(reversed(matrix.canonical_core_ids)))
    with pytest.raises(ValueError, match="content changed"):
        require_loader_registered_task13_matrix_v1(matrix)


def test_task13_builder_rejects_unregistered_clone_before_compute(
    authenticated_fixture, monkeypatch
):
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1
    import mub.vnext.statistics.task13_v3 as publication

    inputs = authenticated_fixture["inputs"]
    paths = {
        "manifest": authenticated_fixture["preparation_manifest_path"],
        "plan": authenticated_fixture["plan_path"],
        "matrix_manifest": authenticated_fixture["matrix_manifest_path"],
        "matrix_summary": authenticated_fixture["summary_path"],
        "integrity_audit": authenticated_fixture["audit_path"],
    }
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    genuine = load_task13_authenticated_matrix_v1(
        preparation_manifest_path=paths["manifest"], plan_path=paths["plan"],
        core_root=inputs["core_root"], evidence_root=inputs["evidence_root"],
        matrix_root=authenticated_fixture["matrix"].matrix_root,
        matrix_manifest_path=paths["matrix_manifest"], matrix_summary_path=paths["matrix_summary"],
        integrity_audit_path=paths["integrity_audit"], repository_root=Path(__file__).resolve().parents[2],
        expected_preparation_manifest_sha256=hashes["manifest"], expected_plan_sha256=hashes["plan"],
        expected_matrix_manifest_sha256=hashes["matrix_manifest"], expected_matrix_summary_sha256=hashes["matrix_summary"],
        expected_integrity_audit_sha256=hashes["integrity_audit"],
    )
    clone = replace(genuine, _loader_token=object())
    monkeypatch.setattr(publication, "compute_task13_statistics_v1", lambda *args: (_ for _ in ()).throw(AssertionError("compute called")))
    with pytest.raises(ValueError, match="loader-registered"):
        publication.build_task13_publication_v3(
            matrix=clone, bootstrap_config=DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1,
            statistics_config_sha256=hashlib.sha256(canonical_json_bytes(DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1)).hexdigest(),
            runtime=Task13RuntimeBindingV1("a" * 40, "b" * 64),
            source_hashes={
                "preparation_manifest": genuine.input_hashes["task12_preparation_manifest"],
                "plan": genuine.input_hashes["task12_plan"],
                "matrix_manifest": genuine.input_hashes["task12_matrix_manifest"],
                "matrix_summary": genuine.input_hashes["task12_matrix_summary"],
                "integrity_audit": genuine.input_hashes["task12_integrity_audit"],
                "core_tasks": genuine.input_hashes["core_tasks"],
                "core_task_manifest": genuine.input_hashes["core_task_manifest"],
            },
        )


def test_task13_direct_builder_rejects_valid_but_forged_source_hash_mapping(
    authenticated_fixture
):
    from mub.vnext.statistics.task13_v3 import (
        Task13RuntimeBindingV1,
        build_task13_publication_v3,
    )
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1

    inputs = authenticated_fixture["inputs"]
    paths = {
        "manifest": authenticated_fixture["preparation_manifest_path"],
        "plan": authenticated_fixture["plan_path"],
        "matrix_manifest": authenticated_fixture["matrix_manifest_path"],
        "matrix_summary": authenticated_fixture["summary_path"],
        "integrity_audit": authenticated_fixture["audit_path"],
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    matrix = load_task13_authenticated_matrix_v1(
        preparation_manifest_path=paths["manifest"],
        plan_path=paths["plan"],
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        matrix_root=authenticated_fixture["matrix"].matrix_root,
        matrix_manifest_path=paths["matrix_manifest"],
        matrix_summary_path=paths["matrix_summary"],
        integrity_audit_path=paths["integrity_audit"],
        repository_root=Path(__file__).resolve().parents[2],
        expected_preparation_manifest_sha256=hashes["manifest"],
        expected_plan_sha256=hashes["plan"],
        expected_matrix_manifest_sha256=hashes["matrix_manifest"],
        expected_matrix_summary_sha256=hashes["matrix_summary"],
        expected_integrity_audit_sha256=hashes["integrity_audit"],
    )
    forged = {
        "preparation_manifest": "f" * 64,
        "plan": "f" * 64,
        "matrix_manifest": "f" * 64,
        "matrix_summary": "f" * 64,
        "integrity_audit": "f" * 64,
        "core_tasks": "f" * 64,
        "core_task_manifest": "f" * 64,
    }
    with pytest.raises(ValueError, match="source hashes"):
        build_task13_publication_v3(
            matrix=matrix,
            bootstrap_config=DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1,
            statistics_config_sha256=hashlib.sha256(
                canonical_json_bytes(DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1)
            ).hexdigest(),
            runtime=Task13RuntimeBindingV1("a" * 40, "b" * 64),
            source_hashes=forged,
        )


def test_task13_main_uses_real_runtime_in_clean_repository(tmp_path, monkeypatch):
    import scripts.vnext_run_core_task13 as command
    import mub.vnext.statistics.task13_v3 as publication

    repository = tmp_path / "clean-repository"
    repository.mkdir()
    for args in (("init", "-q", str(repository)), ("-C", str(repository), "config", "user.email", "task13@example.test"), ("-C", str(repository), "config", "user.name", "Task 13")):
        subprocess.run(("git", *args), check=True)
    (repository / "tracked").write_bytes(b"tracked")
    subprocess.run(("git", "-C", str(repository), "add", "tracked"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-qm", "initial"), check=True)
    expected = publication.current_clean_task13_runtime_v3(repository)
    roots = [tmp_path / name for name in ("core", "evidence", "matrix")]
    for root in roots:
        root.mkdir(); (root / "member").write_bytes(b"member")
    config = tmp_path / "config.json"
    config.write_bytes(canonical_json_bytes(DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1))
    files = [tmp_path / name for name in ("manifest", "plan", "matrix-manifest", "summary", "audit")]
    for path in files: path.write_bytes(b"{}")
    capture: dict[str, object] = {}
    monkeypatch.setattr(command.task13_publication, "capture_task13_source_snapshot_v3", lambda *args, **kwargs: object())
    monkeypatch.setattr(command.task13_publication, "_revalidate_source_snapshot", lambda snapshot: None)
    monkeypatch.setattr(command.task13_publication, "source_snapshot_sha256_v3", lambda snapshot, path: "a" * 64)
    matrix = type("Matrix", (), {"input_hashes": {"task12_preparation_manifest": "a" * 64, "task12_plan": "a" * 64, "task12_matrix_manifest": "a" * 64, "task12_matrix_summary": "a" * 64, "task12_integrity_audit": "a" * 64, "core_tasks": "a" * 64, "core_task_manifest": "a" * 64}})()
    monkeypatch.setattr(command, "load_task13_authenticated_matrix_v1", lambda **kwargs: matrix)
    monkeypatch.setattr(command.task13_publication, "build_task13_publication_v3", lambda **kwargs: capture.update(kwargs) or object())
    result = type("Result", (), {"output_root": tmp_path / "output", "artifact_index_sha256": "b" * 64})()
    monkeypatch.setattr(command.task13_publication, "publish_task13_artifacts_v3", lambda *args, **kwargs: result)
    arguments = {"manifest": files[0], "plan": files[1], "core_root": roots[0], "evidence_root": roots[1], "matrix_root": roots[2], "matrix_bundle_manifest": files[2], "matrix_summary": files[3], "matrix_integrity_audit": files[4], "statistics_config": config, "output_root": tmp_path / "output"}
    assert command.main(_cli_args(arguments), repository_root=repository) == 0
    assert capture["runtime"] == expected


def test_task13_commit_rechecks_ownership_after_source_revalidation(tmp_path, monkeypatch):
    import mub.vnext.statistics.task13_v3 as publication

    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "source"
    source.write_bytes(b"source")
    snapshot = publication.capture_task13_source_snapshot_v3((source,), (sources,))
    parent = tmp_path / "parent"
    parent.mkdir()
    staging = parent / "staging"
    staging.mkdir()
    (staging / "one").write_bytes(b"one")
    ownership = publication._capture_staging_ownership_v3(staging, ("one",))
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))

    original = publication._revalidate_source_snapshot

    def mutate_stage(selected):
        original(selected)
        (staging / "one").write_bytes(b"rewritten")
        (staging / "foreign").write_bytes(b"foreign")

    monkeypatch.setattr(publication, "_revalidate_source_snapshot", mutate_stage)
    with pytest.raises(RuntimeError, match="ownership|staging"):
        publication._commit_staged_task13_root_v3(
            staging=staging,
            final_root=parent / "final",
            parent_identity=parent_identity,
            source_snapshot=snapshot,
            ownership=ownership,
        )
    assert staging.is_dir()
    assert (staging / "foreign").read_bytes() == b"foreign"
    assert not (parent / "final").exists()


def test_task13_absolute_path_rejects_existing_lexical_symlink(tmp_path):
    import mub.vnext.statistics.task13_v3 as publication

    target = tmp_path / "safe-target"
    target.mkdir()
    link = tmp_path / "lexical-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="reparse-point"):
        publication._absolute_no_reparse(link / "missing", require_exists=False)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync semantics")
def test_task13_parent_fsync_unsupported_is_safe_but_other_errors_report_commit(
    tmp_path, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication

    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "source"
    source.write_bytes(b"source")
    snapshot = publication.capture_task13_source_snapshot_v3((source,), (sources,))
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_identity = publication._DirectoryIdentity(parent, publication._identity(parent))

    def unsupported(_descriptor):
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(publication.os, "fsync", unsupported)
    staging = parent / "stage-unsupported"
    staging.mkdir()
    publication._commit_staged_task13_root_v3(
        staging=staging,
        final_root=parent / "final-unsupported",
        parent_identity=parent_identity,
        source_snapshot=snapshot,
    )
    assert (parent / "final-unsupported").is_dir()

    def io_error(_descriptor):
        raise OSError(errno.EIO, "directory fsync failed")

    monkeypatch.setattr(publication.os, "fsync", io_error)
    staging = parent / "stage-error"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="committed|durability"):
        publication._commit_staged_task13_root_v3(
            staging=staging,
            final_root=parent / "final-error",
            parent_identity=parent_identity,
            source_snapshot=snapshot,
        )
    assert (parent / "final-error").is_dir()


def test_task13_direct_publish_rejects_unrelated_or_omitted_registered_root_before_staging(
    tmp_path, monkeypatch
):
    import mub.vnext.statistics.task13_v3 as publication
    from mub.vnext.statistics.input_v3 import (
        Task13LoaderCapabilityV1,
        Task13LoaderFileCapabilityV1,
        Task13LoaderRootCapabilityV1,
    )

    repository = tmp_path / "repository"
    core = tmp_path / "core"
    evidence = tmp_path / "evidence"
    matrix_root = tmp_path / "matrix"
    other = tmp_path / "other"
    for root in (repository, core, evidence, matrix_root, other):
        root.mkdir()
    controls = {}
    for name in (
        "preparation_manifest", "plan", "matrix_manifest", "matrix_summary", "integrity_audit"
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(name.encode("utf-8"))
        result = path.stat()
        controls[name] = Task13LoaderFileCapabilityV1(
            name, path.resolve(), (result.st_dev, result.st_ino), hashlib.sha256(path.read_bytes()).hexdigest()
        )
    roots = {
        name: Task13LoaderRootCapabilityV1(name, path.resolve(), publication._identity(path))
        for name, path in {
            "repository": repository, "core": core, "evidence": evidence, "matrix": matrix_root,
        }.items()
    }
    capability = Task13LoaderCapabilityV1("a" * 64, roots, controls)
    monkeypatch.setattr(publication, "require_loader_registered_task13_matrix_v1", lambda matrix: capability)
    monkeypatch.setattr(publication, "validate_task13_authenticated_matrix_v1", lambda matrix: matrix)
    monkeypatch.setattr(publication, "require_builder_registered_task13_publication_v1", lambda *args: None)
    source_paths = tuple(entry.path for entry in controls.values())
    for label, source_roots in (
        ("unrelated", (other, core, evidence, matrix_root)),
        ("omitted", (repository, core, evidence)),
    ):
        snapshot = publication.capture_task13_source_snapshot_v3(source_paths, source_roots)
        output = tmp_path / f"output-{label}"
        with pytest.raises(ValueError, match="exact loader source roots"):
            publication.publish_task13_artifacts_v3(
                object(), matrix=object(), output_root=output,
                source_snapshot=snapshot, repository_root=repository,
            )
        assert not output.exists()
        assert not tuple(tmp_path.glob(".mub-task13-stage-*"))
