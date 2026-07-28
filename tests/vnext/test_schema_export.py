from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mub.vnext.schema_export as schema_export
from mub.vnext.schema_export import TOP_LEVEL_SCHEMA_MODELS, export_schemas


EXPECTED_SCHEMA_FILES = {
    "mem_update_task.schema.json": "MemUpdateTask",
    "task_run_record.schema.json": "TaskRunRecord",
    "score_record.schema.json": "ScoreRecord",
    "task_manifest.schema.json": "TaskManifest",
    "run_manifest.schema.json": "RunManifest",
}
EXPECTED_VERSION_FIELDS = {
    "mem_update_task.schema.json": ("schema_version",),
    "task_run_record.schema.json": ("schema_version", "runtime_record_version"),
    "score_record.schema.json": ("schema_version", "scorer_version"),
    "task_manifest.schema.json": (
        "schema_version",
        "task_manifest_version",
        "task_schema_version",
    ),
    "run_manifest.schema.json": (
        "schema_version",
        "run_manifest_version",
        "task_schema_version",
        "runtime_record_version",
        "scorer_version",
        "metric_registry_version",
        "profile_version",
    ),
}


def test_registry_contains_exactly_five_top_level_artifacts_and_is_immutable() -> None:
    assert tuple(
        (filename, model.__name__) for filename, model in TOP_LEVEL_SCHEMA_MODELS
    ) == tuple(EXPECTED_SCHEMA_FILES.items())

    with pytest.raises(TypeError):
        TOP_LEVEL_SCHEMA_MODELS[0] = TOP_LEVEL_SCHEMA_MODELS[0]
    assert not hasattr(TOP_LEVEL_SCHEMA_MODELS, "append")


def test_export_schemas_is_deterministic_and_uses_serialization_schema(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = export_schemas(first_dir)
    second_paths = export_schemas(second_dir)

    assert {path.name for path in first_paths} == set(EXPECTED_SCHEMA_FILES)
    assert {path.name for path in second_paths} == set(EXPECTED_SCHEMA_FILES)
    for filename, title in EXPECTED_SCHEMA_FILES.items():
        first_bytes = (first_dir / filename).read_bytes()
        second_bytes = (second_dir / filename).read_bytes()
        assert first_bytes == second_bytes
        assert first_bytes.endswith(b"\n")
        assert not first_bytes.endswith(b"\n\n")
        schema = json.loads(first_bytes)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == title
        for field_name in EXPECTED_VERSION_FIELDS[filename]:
            version_schema = schema["properties"][field_name]
            assert version_schema["default"] == "2.0.0"
            assert version_schema["const"] == "2.0.0"


def test_reexport_replaces_only_generated_schema_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "schemas"
    export_schemas(output_dir)
    unrelated = output_dir / "keep-me.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    generated = output_dir / "mem_update_task.schema.json"
    generated.write_text("stale", encoding="utf-8")

    export_schemas(output_dir)

    assert unrelated.read_text(encoding="utf-8") == "unrelated"
    assert json.loads(generated.read_bytes())["title"] == "MemUpdateTask"


@pytest.mark.parametrize("existing", [False, True])
def test_schema_export_second_publish_failure_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    output_dir = tmp_path / "atomic"
    output_dir.mkdir()
    old = {
        filename: f"old-{index}".encode()
        for index, filename in enumerate(EXPECTED_SCHEMA_FILES)
    }
    if existing:
        for filename, content in old.items():
            (output_dir / filename).write_bytes(content)
    real_replace = atomic_module.os.replace
    publications = 0

    def fail_second_publish(source: Path, destination: Path) -> None:
        nonlocal publications
        if ".tmp." in Path(source).name:
            publications += 1
            if publications == 2:
                raise OSError("injected second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected second publish failure"):
        export_schemas(output_dir)

    if existing:
        assert {
            filename: (output_dir / filename).read_bytes()
            for filename in EXPECTED_SCHEMA_FILES
        } == old
    else:
        assert not any((output_dir / filename).exists() for filename in EXPECTED_SCHEMA_FILES)
    assert not list(output_dir.glob("*.tmp*"))
    assert not list(output_dir.glob("*.bak*"))


def test_registry_rejects_casefold_filename_collisions_before_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_type = TOP_LEVEL_SCHEMA_MODELS[0][1]
    output_dir = tmp_path / "casefold-collision"
    monkeypatch.setattr(
        schema_export,
        "_TOP_LEVEL_SCHEMA_MODELS",
        (
            ("Case.schema.json", model_type),
            ("case.schema.json", model_type),
        ),
    )

    with pytest.raises(ValueError, match=r"schema filename.*case-insensitive|collides"):
        export_schemas(output_dir)

    assert not output_dir.exists()


def test_registry_rejects_unsafe_paths_before_creating_output(tmp_path: Path, monkeypatch) -> None:
    if hasattr(TOP_LEVEL_SCHEMA_MODELS, "values"):
        model_type = next(iter(TOP_LEVEL_SCHEMA_MODELS.values()))
    else:
        model_type = TOP_LEVEL_SCHEMA_MODELS[0][1]
    bad_names = [
        "",
        "../escape.schema.json",
        "nested/file.schema.json",
        r"nested\file.schema.json",
        str((tmp_path / "absolute.schema.json").resolve()),
        "wrong-extension.json",
    ]

    for index, bad_name in enumerate(bad_names):
        output_dir = tmp_path / f"unsafe-{index}"
        monkeypatch.setattr(
            schema_export,
            "_TOP_LEVEL_SCHEMA_MODELS",
            ((bad_name, model_type),),
            raising=False,
        )

        with pytest.raises(ValueError, match="schema filename"):
            export_schemas(output_dir)

        assert not output_dir.exists()


def test_committed_schemas_match_fresh_export_exactly(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    committed_dir = project_root / "schemas" / "vnext"
    fresh_dir = tmp_path / "fresh"

    export_schemas(fresh_dir)

    expected_names = set(EXPECTED_SCHEMA_FILES)
    assert {path.name for path in fresh_dir.iterdir()} == expected_names
    assert {path.name for path in committed_dir.glob("*.schema.json")} == expected_names
    for filename in expected_names:
        assert (fresh_dir / filename).read_bytes() == (committed_dir / filename).read_bytes()


def test_cli_exports_to_requested_directory(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "cli-output"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/vnext_export_schemas.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout == f"Exported 5 vNext schemas to {output_dir}\n"
    assert {path.name for path in output_dir.iterdir()} == set(EXPECTED_SCHEMA_FILES)


def test_schema_export_recovers_full_set_after_hard_crash(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "crash-recovery"
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
from mub.vnext.schema_export import export_schemas
atomic._transaction_fault_point = lambda point: os._exit(93) if point == 'publish:1' else None
export_schemas(Path(sys.argv[1]))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output_dir)],
        cwd=project_root,
        env=env,
        check=False,
    )
    assert crashed.returncode == 93

    exported = export_schemas(output_dir)
    assert {path.name for path in exported} == set(EXPECTED_SCHEMA_FILES)
    assert {path.name for path in output_dir.iterdir()} == set(EXPECTED_SCHEMA_FILES)
    assert not list(output_dir.glob("*.tmp.*"))
    assert not list(output_dir.glob("*.bak.*"))
    assert not list(output_dir.glob(".mub-vnext-transaction*"))




@pytest.mark.parametrize("position", (0, 4))
def test_schema_stage_mutation_before_publish_preserves_old_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: int
) -> None:
    import mub.vnext.io.atomic as atomic

    output_dir = tmp_path / f"schema-stage-mutation-{position}"
    export_schemas(output_dir)
    old = {
        name: (output_dir / name).read_bytes() for name in EXPECTED_SCHEMA_FILES
    }
    original = atomic.publish_files_atomically

    def wrapped(payloads, **kwargs):
        def mutate() -> None:
            sorted(output_dir.glob("*.tmp.*"))[position].write_bytes(b"EVIL")
        kwargs["pre_publish"] = mutate
        return original(payloads, **kwargs)

    monkeypatch.setattr(schema_export, "publish_files_atomically", wrapped)
    with pytest.raises(RuntimeError, match="stage|content|integrity"):
        export_schemas(output_dir)
    assert {
        name: (output_dir / name).read_bytes() for name in EXPECTED_SCHEMA_FILES
    } == old


@pytest.mark.parametrize("position", (0, 4))
def test_schema_publish_time_stage_mutation_rolls_back_old_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: int
) -> None:
    import mub.vnext.io.atomic as atomic

    output_dir = tmp_path / f"schema-publish-race-{position}"
    export_schemas(output_dir)
    old = {
        name: (output_dir / name).read_bytes() for name in EXPECTED_SCHEMA_FILES
    }
    original = atomic.os.replace
    seen = 0
    changed = False

    def replace(source, destination):
        nonlocal seen, changed
        if ".tmp." in Path(source).name:
            if seen == position and not changed:
                Path(source).write_bytes(b"EVIL")
                changed = True
            seen += 1
        return original(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace)
    with pytest.raises(RuntimeError, match="stage|content|integrity"):
        export_schemas(output_dir)
    assert changed
    assert {
        name: (output_dir / name).read_bytes() for name in EXPECTED_SCHEMA_FILES
    } == old




def test_schema_export_recovers_committed_retirement_crash(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "schema-retirement-crash"
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
from mub.vnext.schema_export import export_schemas
atomic._transaction_fault_point=lambda point: os._exit(99) if point == 'retirement_unlinked' else None
export_schemas(Path(sys.argv[1]))
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(project_root)
    crashed = subprocess.run([sys.executable, "-c", script, str(output_dir)], cwd=project_root, env=env, check=False)
    assert crashed.returncode == 99
    assert {path.name for path in output_dir.glob("*.schema.json")} == set(EXPECTED_SCHEMA_FILES)
    exported = export_schemas(output_dir)
    assert {path.name for path in exported} == set(EXPECTED_SCHEMA_FILES)
    assert not list(output_dir.glob(".mub-vnext-transaction*"))


def test_schema_reconstructed_witness_directory_fsync_failure_is_precise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import mub.vnext.io.atomic as atomic

    output_dir = tmp_path / "schema-witness-fsync"
    real_sync = atomic._fsync_directory
    failures = 0

    def failing(directory: Path) -> None:
        nonlocal failures
        active = directory / ".mub-vnext-transaction.json"
        evidence = list(directory.glob(".mub-vnext-transaction.committed*"))
        committed = all(
            (directory / filename).exists() for filename in EXPECTED_SCHEMA_FILES
        )
        if failures == 0 and committed and not active.exists() and not evidence:
            failures = 1
            raise OSError(errno.EIO, "final schema retirement barrier")
        if failures == 1 and evidence:
            failures = 2
            raise OSError(errno.EIO, "schema witness barrier")
        real_sync(directory)

    monkeypatch.setattr(atomic, "_fsync_directory", failing)
    with pytest.raises(RuntimeError, match="durability unconfirmed"):
        export_schemas(output_dir)
    assert failures == 2
    assert {
        path.name for path in output_dir.glob("*.schema.json")
    } == set(EXPECTED_SCHEMA_FILES)
    assert list(output_dir.glob(".mub-vnext-transaction.committed*"))
    export_schemas(output_dir)
    assert not list(output_dir.glob(".mub-vnext-transaction*"))



def test_schema_recovery_rejects_committed_content_tampering(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "schema-tamper"
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
from mub.vnext.schema_export import export_schemas
atomic._transaction_fault_point=lambda point: os._exit(96) if point == 'commit_marked' else None
export_schemas(Path(sys.argv[1]))
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(project_root)
    crashed = subprocess.run([sys.executable, "-c", script, str(output_dir)], cwd=project_root, env=env, check=False)
    assert crashed.returncode == 96
    journal = json.loads((output_dir / ".mub-vnext-transaction.json").read_text(encoding="utf-8"))
    destination = output_dir / journal["entries"][0]["destination"]
    with destination.open("r+b") as handle:
        handle.seek(0); handle.write(b"XX"); handle.flush(); os.fsync(handle.fileno())
    with pytest.raises(RuntimeError, match="content|mismatch"):
        export_schemas(output_dir)
    assert (output_dir / ".mub-vnext-transaction.json").exists()
