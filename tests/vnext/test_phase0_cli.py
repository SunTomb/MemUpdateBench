from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mub.vnext.contracts import (
    MemUpdateTask,
    RunManifest,
    ScoreRecord,
    TaskManifest,
    TaskRunRecord,
)
from mub.vnext.validation import (
    merge_reports,
    validate_distractors,
    validate_family_a_task,
    validate_gold_replay,
    validate_pilot_task,
    validate_task,
    validate_task_semantics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILE_CLI = PROJECT_ROOT / "scripts" / "vnext_compile_legacy.py"
VALIDATE_CLI = PROJECT_ROOT / "scripts" / "vnext_validate_artifacts.py"
LEGACY_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, str(script), *(str(arg) for arg in args)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_canonical_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
        newline="",
    )


def _write_canonical_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="",
    )


def test_dataset_cli_writes_canonical_tasks_and_exact_manifest(tmp_path: Path) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    before = _sha256(source)
    output_dir = tmp_path / "compiled"

    result = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )

    assert result.returncode == 0, result.stderr
    tasks_path = output_dir / "tasks.jsonl"
    manifest_path = output_dir / "task_manifest.json"
    rows = _load_jsonl(tasks_path)
    tasks = [MemUpdateTask.model_validate(row) for row in rows]
    manifest_payload = _load_json(manifest_path)
    manifest = TaskManifest.model_validate(manifest_payload)
    assert len(tasks) == 2
    for task in tasks:
        expected = merge_reports(
            validate_task(task),
            validate_gold_replay(task),
            validate_distractors(task),
        )
        aggregate = validate_task_semantics(task)
        direct = validate_family_a_task(task)
        explicit = validate_pilot_task(task)
        assert expected.valid
        assert aggregate == expected
        assert explicit == direct
        assert not direct.valid
    assert set(manifest_payload) == set(TaskManifest.model_fields)
    assert manifest.leakage_check_summary["compatibility_only"] is True
    assert manifest.split_counts == {
        "train": 0,
        "dev": 0,
        "test": 2,
        "evaluation_only": 0,
    }
    assert manifest.task_file_paths_and_hashes[0].sha256 == _sha256(tasks_path)
    assert manifest.source_manifest_paths_and_hashes[0].sha256 == before
    assert [task.metadata.extra["legacy_example_index"] for task in tasks] == [0, 1]
    assert _sha256(source) == before
    validated = _run(
        VALIDATE_CLI,
        "--kind",
        "task-manifest",
        "--input",
        manifest_path,
        "--manifest",
        manifest_path,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr
    assert json.loads(validated.stdout)["valid"] is True


def test_compile_rejects_missing_input_without_partial_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "compiled"

    result = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        tmp_path / "missing.json",
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )

    assert result.returncode != 0
    assert not (output_dir / "tasks.jsonl").exists()
    assert not (output_dir / "task_manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp*"))


def test_compile_requires_overwrite_and_rejects_source_destination_alias(
    tmp_path: Path,
) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    output_dir = tmp_path / "compiled"
    args = (
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    first = _run(COMPILE_CLI, *args)
    assert first.returncode == 0, first.stderr
    tasks_hash = _sha256(output_dir / "tasks.jsonl")
    manifest_hash = _sha256(output_dir / "task_manifest.json")

    refused = _run(COMPILE_CLI, *args)
    assert refused.returncode != 0
    assert _sha256(output_dir / "tasks.jsonl") == tasks_hash
    assert _sha256(output_dir / "task_manifest.json") == manifest_hash
    assert not list(output_dir.glob("*.tmp*"))

    overwritten = _run(COMPILE_CLI, *args, "--overwrite")
    assert overwritten.returncode == 0, overwritten.stderr
    assert _sha256(output_dir / "tasks.jsonl") == tasks_hash

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias_source = alias_dir / "tasks.jsonl"
    alias_source.write_bytes(source.read_bytes())
    alias = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        alias_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        alias_dir,
        "--overwrite",
    )
    assert alias.returncode != 0
    assert alias_source.read_bytes() == source.read_bytes()
    assert not (alias_dir / "task_manifest.json").exists()
    assert not list(alias_dir.glob("*.tmp*"))


def test_results_cli_writes_canonical_records_and_authenticated_manifest(
    tmp_path: Path,
) -> None:
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    result_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    source_hashes = {
        dataset_source: _sha256(dataset_source),
        result_source: _sha256(result_source),
    }
    task_dir = tmp_path / "tasks"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    output_dir = tmp_path / "results"

    result = _run(
        COMPILE_CLI,
        "results",
        "--input",
        result_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        output_dir,
    )

    assert result.returncode == 0, result.stderr
    run_rows = _load_jsonl(output_dir / "task_runs.jsonl")
    score_rows = _load_jsonl(output_dir / "scores.jsonl")
    runs = [TaskRunRecord.model_validate(row) for row in run_rows]
    scores = [ScoreRecord.model_validate(row) for row in score_rows]
    manifest_payload = _load_json(output_dir / "run_manifest.json")
    manifest = RunManifest.model_validate(manifest_payload)
    from mub.vnext.legacy.results import (
        LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH,
        is_legacy_evomemory_adapter_identity,
    )

    assert len(runs) == len(scores) == 2
    assert is_legacy_evomemory_adapter_identity(manifest)
    assert (
        manifest.object_value_extractor_config_hash
        == LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH
    )
    assert all(
        row.parser_extractor_provenance.object_value_extractor_config_hash is None
        for row in runs
    )
    assert all(set(row) == set(TaskRunRecord.model_fields) for row in run_rows)
    assert all(set(row) == set(ScoreRecord.model_fields) for row in score_rows)
    assert set(manifest_payload) == set(RunManifest.model_fields)
    assert {row.run_id for row in runs + scores} == {manifest.run_id}
    assert [row.task_id for row in runs] == [row.task_id for row in scores]
    assert manifest.task_manifest.sha256 == _sha256(task_dir / "task_manifest.json")
    assert manifest.normalized_runtime_artifacts[0].sha256 == _sha256(
        output_dir / "task_runs.jsonl"
    )
    assert manifest.score_artifacts[0].sha256 == _sha256(output_dir / "scores.jsonl")
    assert manifest.native_vs_extracted_field_summary["compatibility_only"] is True
    assert (
        manifest.prompt_config["legacy_result_import"]["canonical_evaluation_mode"]
        is None
    )
    assert all(_sha256(path) == digest for path, digest in source_hashes.items())
    validations = {
        "task-runs": output_dir / "task_runs.jsonl",
        "scores": output_dir / "scores.jsonl",
        "run-manifest": output_dir / "run_manifest.json",
    }
    for kind, artifact in validations.items():
        validated = _run(
            VALIDATE_CLI,
            "--kind",
            kind,
            "--input",
            artifact,
            "--manifest",
            output_dir / "run_manifest.json",
        )
        assert validated.returncode == 0, validated.stdout + validated.stderr
        assert json.loads(validated.stdout)["valid"] is True


def test_atomic_legacy_validation_authenticates_snapshot_before_waiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.legacy.artifacts as artifact_module
    import mub.vnext.legacy.validation as legacy_validation_module

    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    output_dir = tmp_path / "authenticated-context"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    tasks_path = output_dir / "tasks.jsonl"
    manifest_path = output_dir / "task_manifest.json"
    tasks = [MemUpdateTask.model_validate(row) for row in _load_jsonl(tasks_path)]
    manifest = TaskManifest.model_validate(_load_json(manifest_path))

    assert "tasks_bytes" not in inspect.signature(
        artifact_module.authenticate_legacy_task_manifest
    ).parameters
    assert "tasks_bytes" not in inspect.signature(
        artifact_module.build_expected_legacy_task_manifest
    ).parameters
    assert "_build_expected_legacy_task_manifest_snapshot" not in artifact_module.__all__
    assert not hasattr(legacy_validation_module, "_AuthenticatedLegacyValidationContext")
    assert not hasattr(
        legacy_validation_module,
        "_validate_authenticated_legacy_task_semantics",
    )
    authenticated, reports = artifact_module._authenticate_and_validate_legacy_tasks(
        manifest,
        tasks,
        tasks_path=tasks_path,
    )
    assert authenticated == manifest
    assert all(report.valid for report in reports)

    forged_payload = tasks[0].model_dump(mode="json")
    forged_payload["source"]["generator"]["generator_name"] = "attacker_generator"
    forged = MemUpdateTask.model_validate(forged_payload)
    with pytest.raises(ValueError, match="snapshot|canonical task bytes"):
        artifact_module._authenticate_and_validate_legacy_tasks(
            manifest,
            [forged, *tasks[1:]],
            tasks_path=tasks_path,
        )

    original_bytes = tasks_path.read_bytes()
    tasks_path.write_bytes(original_bytes.replace(b"Suzhou", b"Xuzhou", 1))
    with pytest.raises(ValueError, match="snapshot|canonical task bytes|manifest"):
        artifact_module._authenticate_and_validate_legacy_tasks(
            manifest,
            tasks,
            tasks_path=tasks_path,
        )
    tasks_path.write_bytes(original_bytes)

    forged_manifest_payload = manifest.model_dump(mode="json")
    forged_manifest_payload["code_revision"] = "attacker-controlled"
    forged_manifest = TaskManifest.model_validate(forged_manifest_payload)
    privileged_called = False

    def unexpected_privileged_validation(*args, **kwargs):
        nonlocal privileged_called
        privileged_called = True
        raise AssertionError("privileged validation ran before manifest authentication")

    monkeypatch.setattr(
        artifact_module,
        "_validate_trusted_legacy_task_semantics",
        unexpected_privileged_validation,
    )
    with pytest.raises(ValueError, match="authenticated deterministic compilation"):
        artifact_module._authenticate_and_validate_legacy_tasks(
            forged_manifest,
            tasks,
            tasks_path=tasks_path,
        )
    assert privileged_called is False


@pytest.mark.parametrize("race_stage", ["snapshot", "parse", "recompile"])
def test_atomic_legacy_validation_rejects_source_races_before_waiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_stage: str,
) -> None:
    import mub.vnext.legacy.artifacts as artifact_module

    source = tmp_path / "source.json"
    source.write_bytes((LEGACY_FIXTURES / "p63_dataset_minimal.json").read_bytes())
    output_dir = tmp_path / "compiled"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    tasks_path = output_dir / "tasks.jsonl"
    tasks = [MemUpdateTask.model_validate(row) for row in _load_jsonl(tasks_path)]
    manifest = TaskManifest.model_validate(_load_json(output_dir / "task_manifest.json"))
    original_source = source.read_bytes()
    changed_source = original_source.replace(b"Suzhou", b"Xuzhou", 1)
    assert changed_source != original_source

    if race_stage == "snapshot":
        original_read = artifact_module._read_legacy_source_snapshot

        def racing_read(source_path: Path, declared_hash: str):
            source.write_bytes(changed_source)
            return original_read(source_path, declared_hash)

        monkeypatch.setattr(
            artifact_module,
            "_read_legacy_source_snapshot",
            racing_read,
        )
    elif race_stage == "parse":
        original_parse = artifact_module._parse_dataset

        def racing_parse(raw: bytes, source_path: Path):
            rows = original_parse(raw, source_path)
            source.write_bytes(changed_source)
            return rows

        monkeypatch.setattr(artifact_module, "_parse_dataset", racing_parse)
    else:
        original_compile = artifact_module.compile_legacy_episode
        mutation_pending = True

        def racing_compile(*args, **kwargs):
            nonlocal mutation_pending
            task = original_compile(*args, **kwargs)
            if mutation_pending:
                mutation_pending = False
                source.write_bytes(changed_source)
            return task

        monkeypatch.setattr(artifact_module, "compile_legacy_episode", racing_compile)

    privileged_called = False

    def unexpected_privileged_validation(*args, **kwargs):
        nonlocal privileged_called
        privileged_called = True
        raise AssertionError("privileged validation ran after a source race")

    monkeypatch.setattr(
        artifact_module,
        "_validate_trusted_legacy_task_semantics",
        unexpected_privileged_validation,
    )
    with pytest.raises(RuntimeError, match="source changed during authentication"):
        artifact_module._authenticate_and_validate_legacy_tasks(
            manifest,
            tasks,
            tasks_path=tasks_path,
        )
    assert privileged_called is False


def test_atomic_legacy_validation_rechecks_source_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.legacy.artifacts as artifact_module

    source = tmp_path / "source.json"
    source.write_bytes((LEGACY_FIXTURES / "p63_dataset_minimal.json").read_bytes())
    output_dir = tmp_path / "compiled"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    tasks_path = output_dir / "tasks.jsonl"
    tasks = [MemUpdateTask.model_validate(row) for row in _load_jsonl(tasks_path)]
    manifest = TaskManifest.model_validate(_load_json(output_dir / "task_manifest.json"))
    original_semantics = artifact_module._validate_trusted_legacy_task_semantics
    original_source = source.read_bytes()
    changed_source = original_source.replace(b"Suzhou", b"Xuzhou", 1)
    mutation_pending = True

    def racing_semantics(task: MemUpdateTask):
        nonlocal mutation_pending
        report = original_semantics(task)
        if mutation_pending:
            mutation_pending = False
            source.write_bytes(changed_source)
        return report

    monkeypatch.setattr(
        artifact_module,
        "_validate_trusted_legacy_task_semantics",
        racing_semantics,
    )
    with pytest.raises(RuntimeError, match="source changed during authentication"):
        artifact_module._authenticate_and_validate_legacy_tasks(
            manifest,
            tasks,
            tasks_path=tasks_path,
        )


def test_public_legacy_manifest_apis_require_exact_canonical_task_file(
    tmp_path: Path,
) -> None:
    import mub.vnext.legacy.artifacts as artifact_module

    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    output_dir = tmp_path / "compiled"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    tasks_path = output_dir / "tasks.jsonl"
    original_snapshot = tasks_path.read_bytes()
    tasks = [MemUpdateTask.model_validate(row) for row in _load_jsonl(tasks_path)]
    manifest = TaskManifest.model_validate(_load_json(output_dir / "task_manifest.json"))

    assert artifact_module.build_expected_legacy_task_manifest(
        tasks,
        tasks_path=tasks_path,
    ) == manifest
    assert artifact_module.authenticate_legacy_task_manifest(
        manifest,
        tasks,
        tasks_path=tasks_path,
    ) == manifest

    rows = original_snapshot.splitlines(keepends=True)
    invalid_snapshots = {
        "arbitrary": b"not a task artifact",
        "reordered_file": b"".join(reversed(rows)),
        "noncanonical_whitespace": b" " + original_snapshot,
        "extra_newline": original_snapshot + b"\n",
        "encoding_bom": b"\xef\xbb\xbf" + original_snapshot,
        "malformed_row": b'{"task_id":\n',
    }
    for label, invalid_snapshot in invalid_snapshots.items():
        crafted_manifest = artifact_module._build_expected_legacy_task_manifest_snapshot(
            tasks,
            tasks_path=tasks_path,
            tasks_bytes=invalid_snapshot,
        )
        tasks_path.write_bytes(invalid_snapshot)
        with pytest.raises(ValueError, match="canonical task bytes"):
            artifact_module.build_expected_legacy_task_manifest(
                tasks,
                tasks_path=tasks_path,
            )
        with pytest.raises(ValueError, match="canonical task bytes"):
            artifact_module.authenticate_legacy_task_manifest(
                crafted_manifest,
                tasks,
                tasks_path=tasks_path,
            )
        tasks_path.write_bytes(original_snapshot)

    changed_payload = tasks[0].model_dump(mode="json")
    changed_payload["source"]["generator"]["generator_name"] = "changed-model"
    changed_model = MemUpdateTask.model_validate(changed_payload)
    mismatched_task_lists = {
        "reordered_models": list(reversed(tasks)),
        "missing_model": tasks[:-1],
        "changed_model": [changed_model, *tasks[1:]],
    }
    for label, mismatched_tasks in mismatched_task_lists.items():
        with pytest.raises(ValueError, match="canonical task bytes"):
            artifact_module.build_expected_legacy_task_manifest(
                mismatched_tasks,
                tasks_path=tasks_path,
            )
        with pytest.raises(ValueError, match="canonical task bytes"):
            artifact_module.authenticate_legacy_task_manifest(
                manifest,
                mismatched_tasks,
                tasks_path=tasks_path,
            )


def test_validator_accepts_intact_tasks_and_rejects_tampered_hash(
    tmp_path: Path,
) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    output_dir = tmp_path / "compiled"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    tasks = output_dir / "tasks.jsonl"
    manifest = output_dir / "task_manifest.json"

    valid = _run(
        VALIDATE_CLI,
        "--kind",
        "tasks",
        "--input",
        tasks,
        "--manifest",
        manifest,
    )
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout) == {
        "errors": [],
        "kind": "tasks",
        "valid": True,
        "warnings": [],
    }

    original = tasks.read_bytes()
    tasks.write_bytes(original.replace(b"Suzhou", b"Xuzhou", 1))
    invalid = _run(
        VALIDATE_CLI,
        "--kind",
        "tasks",
        "--input",
        tasks,
        "--manifest",
        manifest,
    )
    assert invalid.returncode != 0
    report = json.loads(invalid.stdout)
    assert report["valid"] is False
    assert report["kind"] == "tasks"
    assert report["errors"]
    assert tasks.read_bytes() != original


def test_legacy_analysis_manifest_is_strict_immutable_and_compatibility_only() -> None:
    from pydantic import ValidationError

    from mub.vnext.contracts.common import ArtifactRef
    from mub.vnext.legacy.artifacts import LegacyAnalysisManifest

    manifest = LegacyAnalysisManifest(
        analysis_kind="conflict",
        compiler_version="vnext-phase0-cli-1.0.0",
        compatibility_only=True,
        source_artifacts=(
            ArtifactRef(
                path="source.csv",
                sha256="a" * 64,
                media_type="text/csv",
                record_count=1,
            ),
        ),
        output_artifacts=(
            ArtifactRef(
                path="legacy_analysis.jsonl",
                sha256="b" * 64,
                media_type="application/x-ndjson",
                record_count=1,
            ),
        ),
        row_counts={"legacy_analysis.jsonl": 1},
        warnings=(),
        caveats=("compatibility-only evidence",),
        code_revision="legacy-compatibility-import",
    )
    assert manifest.compatibility_only is True
    with pytest.raises(ValidationError):
        LegacyAnalysisManifest.model_validate(
            {**manifest.model_dump(mode="python"), "compatibility_only": False}
        )
    with pytest.raises(ValidationError):
        LegacyAnalysisManifest.model_validate(
            {**manifest.model_dump(mode="python"), "extra": "forbidden"}
        )
    with pytest.raises(Exception):
        manifest.analysis_kind = "dose"


def test_mechanism_and_ledger_subcommands_emit_typed_compatibility_manifests(
    tmp_path: Path,
) -> None:
    from mub.vnext.legacy.artifacts import LegacyAnalysisManifest
    from mub.vnext.legacy.ledger import LedgerReferenceAudit

    fixtures = {
        "conflict": "p83_conflict_rows.csv",
        "dose": "p83_synthetic_dose_rows.csv",
        "stale-removal": "p83_stale_removal_rows.csv",
        "api": "p84_api_rows.csv",
    }
    for kind, fixture_name in fixtures.items():
        source = LEGACY_FIXTURES / fixture_name
        source_hash = _sha256(source)
        output_dir = tmp_path / kind
        command = _run(
            COMPILE_CLI,
            "mechanism",
            "--kind",
            kind,
            "--input",
            source,
            "--output-dir",
            output_dir,
        )
        assert command.returncode == 0, command.stderr
        rows = _load_jsonl(output_dir / "legacy_analysis.jsonl")
        manifest = LegacyAnalysisManifest.model_validate(
            _load_json(output_dir / "legacy_analysis_manifest.json")
        )
        assert rows
        assert manifest.analysis_kind == kind
        assert manifest.compatibility_only is True
        assert manifest.source_artifacts[0].sha256 == source_hash
        first_bytes = {
            path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
        }
        repeated = _run(
            COMPILE_CLI,
            "mechanism",
            "--kind",
            kind,
            "--input",
            source,
            "--output-dir",
            output_dir,
            "--overwrite",
        )
        assert repeated.returncode == 0, repeated.stderr
        assert first_bytes == {
            path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
        }
        assert _sha256(source) == source_hash
        assert not list(output_dir.glob("*.tmp*"))

    ledger_source = LEGACY_FIXTURES / "ledger_references.md"
    ledger_hash = _sha256(ledger_source)
    ledger_dir = tmp_path / "ledger"
    ledger = _run(
        COMPILE_CLI,
        "ledger",
        "--input",
        ledger_source,
        "--project-root",
        PROJECT_ROOT,
        "--output-dir",
        ledger_dir,
    )
    assert ledger.returncode == 0, ledger.stderr
    audit = LedgerReferenceAudit.model_validate(_load_json(ledger_dir / "ledger_audit.json"))
    ledger_manifest = LegacyAnalysisManifest.model_validate(
        _load_json(ledger_dir / "legacy_analysis_manifest.json")
    )
    assert audit.unresolved
    assert ledger_manifest.analysis_kind == "ledger"
    assert ledger_manifest.row_counts == {"ledger_audit.json": 1}
    assert _sha256(ledger_source) == ledger_hash


def test_mechanism_duplicate_rows_fail_without_partial_artifacts(tmp_path: Path) -> None:
    source_fixture = LEGACY_FIXTURES / "p83_conflict_rows.csv"
    lines = source_fixture.read_text(encoding="utf-8").splitlines()
    duplicate_source = tmp_path / "duplicate-conflict.csv"
    duplicate_source.write_text(
        "\n".join([*lines, lines[1]]) + "\n",
        encoding="utf-8",
        newline="",
    )
    output_dir = tmp_path / "duplicate-output"

    result = _run(
        COMPILE_CLI,
        "mechanism",
        "--kind",
        "conflict",
        "--input",
        duplicate_source,
        "--output-dir",
        output_dir,
    )

    assert result.returncode != 0
    assert not (output_dir / "legacy_analysis.jsonl").exists()
    assert not (output_dir / "legacy_analysis_manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp*"))


def test_results_failure_preserves_existing_set_and_leaves_no_temps(
    tmp_path: Path,
) -> None:
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    task_dir = tmp_path / "tasks"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    invalid_source = tmp_path / "invalid-results.json"
    invalid_source.write_text('{"summary":{},"results":[', encoding="utf-8")
    output_dir = tmp_path / "existing-results"
    output_dir.mkdir()
    originals = {
        "task_runs.jsonl": b"old runs\n",
        "scores.jsonl": b"old scores\n",
        "run_manifest.json": b"old manifest\n",
    }
    for name, content in originals.items():
        (output_dir / name).write_bytes(content)

    result = _run(
        COMPILE_CLI,
        "results",
        "--input",
        invalid_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        output_dir,
        "--overwrite",
    )

    assert result.returncode != 0
    assert {name: (output_dir / name).read_bytes() for name in originals} == originals
    assert not list(output_dir.glob("*.tmp*"))
    assert not list(output_dir.glob("*.bak*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point coverage")
def test_atomic_and_cli_reject_windows_directory_junction(tmp_path: Path) -> None:
    from mub.vnext.io.atomic import publish_files_atomically

    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("junction creation unavailable")
    with pytest.raises(ValueError, match="reparse"):
        publish_files_atomically(
            {junction / "out.json": b"unsafe"}, overwrite=False
        )
    cli = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        LEGACY_FIXTURES / "p63_dataset_minimal.json",
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        junction,
    )
    assert cli.returncode != 0
    assert not (target / "tasks.jsonl").exists()


def test_compile_rejects_hardlink_source_destination_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes((LEGACY_FIXTURES / "p63_dataset_minimal.json").read_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    destination = output_dir / "tasks.jsonl"
    os.link(source, destination)
    before = source.read_bytes()

    result = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
        "--overwrite",
    )

    assert result.returncode != 0
    assert source.read_bytes() == destination.read_bytes() == before
    assert not (output_dir / "task_manifest.json").exists()
    assert not list(output_dir.glob("*.tmp*"))


def test_validator_rejects_count_schema_canonical_and_identity_tampering(
    tmp_path: Path,
) -> None:
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    result_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    task_dir = tmp_path / "tasks"
    result_dir = tmp_path / "results"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    )
    assert compiled.returncode == 0, compiled.stderr
    imported = _run(
        COMPILE_CLI,
        "results",
        "--input",
        result_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        result_dir,
    )
    assert imported.returncode == 0, imported.stderr

    count_dir = tmp_path / "tamper-count"
    shutil.copytree(task_dir, count_dir)
    count_manifest = _load_json(count_dir / "task_manifest.json")
    count_manifest["split_counts"]["test"] = 3
    _write_canonical_json(count_dir / "task_manifest.json", count_manifest)
    count_report = _run(
        VALIDATE_CLI,
        "--kind",
        "task-manifest",
        "--input",
        count_dir / "task_manifest.json",
        "--manifest",
        count_dir / "task_manifest.json",
    )
    assert count_report.returncode != 0

    schema_dir = tmp_path / "tamper-schema"
    shutil.copytree(task_dir, schema_dir)
    schema_rows = _load_jsonl(schema_dir / "tasks.jsonl")
    schema_rows[0]["schema_version"] = "9.9.9"
    _write_canonical_jsonl(schema_dir / "tasks.jsonl", schema_rows)
    schema_manifest = _load_json(schema_dir / "task_manifest.json")
    schema_manifest["task_file_paths_and_hashes"][0]["path"] = str(
        (schema_dir / "tasks.jsonl").resolve()
    )
    schema_manifest["task_file_paths_and_hashes"][0]["sha256"] = _sha256(
        schema_dir / "tasks.jsonl"
    )
    _write_canonical_json(schema_dir / "task_manifest.json", schema_manifest)
    schema_report = _run(
        VALIDATE_CLI,
        "--kind",
        "tasks",
        "--input",
        schema_dir / "tasks.jsonl",
        "--manifest",
        schema_dir / "task_manifest.json",
    )
    assert schema_report.returncode != 0

    canonical_dir = tmp_path / "tamper-canonical"
    shutil.copytree(task_dir, canonical_dir)
    canonical_tasks = canonical_dir / "tasks.jsonl"
    canonical_tasks.write_bytes(b" " + canonical_tasks.read_bytes())
    canonical_manifest = _load_json(canonical_dir / "task_manifest.json")
    canonical_manifest["task_file_paths_and_hashes"][0]["path"] = str(
        canonical_tasks.resolve()
    )
    canonical_manifest["task_file_paths_and_hashes"][0]["sha256"] = _sha256(
        canonical_tasks
    )
    _write_canonical_json(canonical_dir / "task_manifest.json", canonical_manifest)
    canonical_report = _run(
        VALIDATE_CLI,
        "--kind",
        "tasks",
        "--input",
        canonical_tasks,
        "--manifest",
        canonical_dir / "task_manifest.json",
    )
    assert canonical_report.returncode != 0

    identity_dir = tmp_path / "tamper-identity"
    shutil.copytree(result_dir, identity_dir)
    run_rows = _load_jsonl(identity_dir / "task_runs.jsonl")
    run_rows[0]["task_id"] = "task_forged_identity"
    _write_canonical_jsonl(identity_dir / "task_runs.jsonl", run_rows)
    run_manifest = _load_json(identity_dir / "run_manifest.json")
    run_manifest["normalized_runtime_artifacts"][0]["path"] = str(
        (identity_dir / "task_runs.jsonl").resolve()
    )
    run_manifest["normalized_runtime_artifacts"][0]["sha256"] = _sha256(
        identity_dir / "task_runs.jsonl"
    )
    run_manifest["score_artifacts"][0]["path"] = str(
        (identity_dir / "scores.jsonl").resolve()
    )
    _write_canonical_json(identity_dir / "run_manifest.json", run_manifest)
    identity_report = _run(
        VALIDATE_CLI,
        "--kind",
        "task-runs",
        "--input",
        identity_dir / "task_runs.jsonl",
        "--manifest",
        identity_dir / "run_manifest.json",
    )
    assert identity_report.returncode != 0


def test_validator_rejects_legacy_answer_mode_as_canonical_identity(
    tmp_path: Path,
) -> None:
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    result_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    task_dir = tmp_path / "tasks"
    result_dir = tmp_path / "results"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    assert _run(
        COMPILE_CLI,
        "results",
        "--input",
        result_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        result_dir,
    ).returncode == 0
    manifest_path = result_dir / "run_manifest.json"
    manifest = _load_json(manifest_path)
    assert manifest["prompt_config"]["legacy_result_import"][
        "canonical_evaluation_mode"
    ] is None
    manifest["prompt_config"]["legacy_result_import"][
        "canonical_evaluation_mode"
    ] = "slot_prompt"
    _write_canonical_json(manifest_path, manifest)

    result = _run(
        VALIDATE_CLI,
        "--kind",
        "run-manifest",
        "--input",
        manifest_path,
        "--manifest",
        manifest_path,
    )
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["valid"] is False
    assert "slot_prompt" not in result.stdout


def test_atomic_publication_rolls_back_staged_set_on_prepublish_failure(
    tmp_path: Path,
) -> None:
    from mub.vnext.io.atomic import publish_files_atomically

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")

    def fail_after_staging() -> None:
        raise RuntimeError("injected pre-publication failure")

    with pytest.raises(RuntimeError, match="injected"):
        publish_files_atomically(
            {first: b"new-first", second: b"new-second"},
            overwrite=True,
            pre_publish=fail_after_staging,
        )

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))


def test_atomic_lock_is_stable_across_process_temp_roots(tmp_path: Path) -> None:
    output_dir = tmp_path / "shared-temp-roots"
    output_dir.mkdir()
    marker = tmp_path / "a-first.ready"
    temp_a = tmp_path / "temp-a"
    temp_b = tmp_path / "temp-b"
    temp_a.mkdir()
    temp_b.mkdir()
    script = r'''
import os, sys, time
from pathlib import Path
import mub.vnext.io.atomic as atomic
out = Path(sys.argv[1]); label = sys.argv[2]; marker = Path(sys.argv[3])
real_replace = atomic.os.replace
published = 0
def delayed_replace(source, destination):
    global published
    real_replace(source, destination)
    if '.tmp.' in Path(source).name:
        published += 1
        if label == 'A' and published == 1:
            marker.write_text('ready', encoding='utf-8')
            time.sleep(2.5)
atomic.os.replace = delayed_replace
atomic.publish_files_atomically(
    {out / 'first.json': (label + '1').encode(), out / 'second.json': (label + '2').encode()},
    overwrite=True,
)
'''
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(PROJECT_ROOT)

    def environment(root: Path) -> dict[str, str]:
        env = dict(base_env)
        env.update({"TEMP": str(root), "TMP": str(root), "TMPDIR": str(root)})
        return env

    first = subprocess.Popen(
        [sys.executable, "-c", script, str(output_dir), "A", str(marker)],
        cwd=PROJECT_ROOT,
        env=environment(temp_a),
    )
    for _ in range(500):
        if marker.exists():
            break
        if first.poll() is not None:
            break
        import time

        time.sleep(0.01)
    assert marker.exists()
    second = subprocess.Popen(
        [sys.executable, "-c", script, str(output_dir), "B", str(marker)],
        cwd=PROJECT_ROOT,
        env=environment(temp_b),
    )
    assert first.wait(timeout=30) == 0
    assert second.wait(timeout=30) == 0
    generation = (
        (output_dir / "first.json").read_bytes(),
        (output_dir / "second.json").read_bytes(),
    )
    assert generation in {(b"A1", b"A2"), (b"B1", b"B2")}


@pytest.mark.parametrize(
    "stage",
    (
        "journal_prepared",
        "backup:0",
        "backup:1",
        "publish:0",
        "publish:1",
        "commit_marked",
        "cleanup:0",
    ),
)
def test_atomic_hard_crash_is_recovered_before_next_publish(
    tmp_path: Path,
    stage: str,
) -> None:
    output_dir = tmp_path / stage.replace(":", "-")
    output_dir.mkdir()
    first = output_dir / "first.json"
    second = output_dir / "second.json"
    first.write_bytes(b"O1")
    second.write_bytes(b"O2")
    crash_script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
stage = sys.argv[2]
def crash(point):
    if point == stage:
        os._exit(91)
atomic._transaction_fault_point = crash
out = Path(sys.argv[1])
atomic.publish_files_atomically(
    {out / 'first.json': b'N1', out / 'second.json': b'N2'}, overwrite=True
)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(output_dir), stage],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert crashed.returncode == 91

    observed = output_dir / "observed.txt"
    recover_script = r'''
import sys
from pathlib import Path
from mub.vnext.io.atomic import publish_files_atomically
out = Path(sys.argv[1]); observed = Path(sys.argv[2])
def inspect():
    pair = ((out / 'first.json').read_bytes(), (out / 'second.json').read_bytes())
    if pair not in {(b'O1', b'O2'), (b'N1', b'N2')}:
        raise RuntimeError(f'incoherent recovered generation: {pair!r}')
    observed.write_text(repr(pair), encoding='utf-8')
publish_files_atomically(
    {out / 'first.json': b'R1', out / 'second.json': b'R2'},
    overwrite=True,
    pre_publish=inspect,
)
'''
    recovered = subprocess.run(
        [sys.executable, "-c", recover_script, str(output_dir), str(observed)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert observed.exists()
    assert (first.read_bytes(), second.read_bytes()) == (b"R1", b"R2")
    assert not list(output_dir.glob("*.tmp.*"))
    assert not list(output_dir.glob("*.bak.*"))
    assert not list(output_dir.glob(".mub-vnext-transaction*"))


def test_atomic_no_clobber_hard_crash_recovers_before_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "no-clobber-crash"
    output_dir.mkdir()
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
atomic._transaction_fault_point = lambda point: os._exit(92) if point == 'publish:0' else None
out = Path(sys.argv[1])
atomic.publish_files_atomically(
    {out / 'first.json': b'N1', out / 'second.json': b'N2'}, overwrite=False
)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output_dir)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert crashed.returncode == 92
    recovery = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from mub.vnext.io.atomic import publish_files_atomically; p=Path(r'%s'); publish_files_atomically({p/'first.json':b'R1',p/'second.json':b'R2'}, overwrite=True)"
            % str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    assert (
        (output_dir / "first.json").read_bytes(),
        (output_dir / "second.json").read_bytes(),
    ) == (b"R1", b"R2")
    assert not list(output_dir.glob("*.tmp.*"))
    assert not list(output_dir.glob("*.bak.*"))
    assert not list(output_dir.glob(".mub-vnext-transaction*"))




def test_atomic_rejects_hardlinked_lock_without_mutating_source(tmp_path: Path) -> None:
    import mub.vnext.io.atomic as atomic

    output_dir = tmp_path / "hardlinked-lock-output"
    output_dir.mkdir()
    protected = tmp_path / "secret-source.bin"
    protected.write_bytes(b"")
    canonical = os.path.normcase(str(output_dir.resolve(strict=True)))
    lock_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lock_path = output_dir.parent / f".mub-vnext-publish-{lock_hash}.lock"
    os.link(protected, lock_path)

    with pytest.raises(ValueError, match="lock") as exc_info:
        atomic.publish_files_atomically(
            {output_dir / "out.json": b"new"},
            overwrite=False,
            source_paths=(protected,),
        )
    assert protected.read_bytes() == b""
    assert "secret-source" not in str(exc_info.value)


def test_atomic_recovers_prejournal_new_file_after_hard_crash(tmp_path: Path) -> None:
    output_dir = tmp_path / "prejournal-crash"
    output_dir.mkdir()
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
atomic._transaction_fault_point = lambda point: os._exit(94) if point == 'journal_new_fsynced' else None
out = Path(sys.argv[1])
atomic.publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'}, overwrite=True)
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT)
    crashed = subprocess.run([sys.executable, "-c", script, str(output_dir)], cwd=PROJECT_ROOT, env=env, check=False)
    assert crashed.returncode == 94
    recovered = subprocess.run([sys.executable, "-c", "from pathlib import Path; from mub.vnext.io.atomic import publish_files_atomically; p=Path(r'%s'); publish_files_atomically({p/'first.json':b'R1',p/'second.json':b'R2'},overwrite=True)" % str(output_dir)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert ((output_dir / "first.json").read_bytes(), (output_dir / "second.json").read_bytes()) == (b"R1", b"R2")
    assert not (output_dir / ".mub-vnext-transaction.json.new").exists()
    assert not list(output_dir.glob("*.tmp.*"))




@pytest.mark.parametrize("overwrite", (False, True))
@pytest.mark.parametrize("position", (0, 1))
def test_prejournal_recovery_rejects_stage_hardlinked_to_destination(
    tmp_path: Path,
    overwrite: bool,
    position: int,
) -> None:
    output_dir = tmp_path / f"prejournal-effect-{overwrite}-{position}"
    output_dir.mkdir()
    if overwrite:
        (output_dir / "first.json").write_bytes(b"O1")
        (output_dir / "second.json").write_bytes(b"O2")
    crash_script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
atomic._transaction_fault_point=lambda point: os._exit(97) if point == 'journal_new_fsynced' else None
out=Path(sys.argv[1]); overwrite=sys.argv[2]=='true'
atomic.publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'},overwrite=overwrite)
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT)
    crashed = subprocess.run([sys.executable, "-c", crash_script, str(output_dir), str(overwrite).lower()], cwd=PROJECT_ROOT, env=env, check=False)
    assert crashed.returncode == 97
    intent_path = output_dir / ".mub-vnext-transaction.json.new"
    intent = _load_json(intent_path)
    entry = intent["entries"][position]
    stage = output_dir / entry["temporary"]
    destination = output_dir / entry["destination"]
    if destination.exists():
        destination.unlink()
    os.link(stage, destination)
    marker = tmp_path / f"prejournal-callback-{overwrite}-{position}"
    recovery_script = "from pathlib import Path; from mub.vnext.io.atomic import publish_files_atomically; p=Path(r'%s'); m=Path(r'%s'); publish_files_atomically({p/'first.json':b'R1',p/'second.json':b'R2'},overwrite=True,pre_publish=lambda:m.write_text('partial'))" % (str(output_dir), str(marker))
    recovered = subprocess.run([sys.executable, "-c", recovery_script], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    assert recovered.returncode != 0
    assert not marker.exists()
    assert intent_path.exists()
    assert stage.exists() and destination.exists()
    assert os.path.samefile(stage, destination)


@pytest.mark.parametrize("position", (0, 4))
def test_schema_prejournal_recovery_rejects_stage_destination_hardlink(
    tmp_path: Path,
    position: int,
) -> None:
    project_root = PROJECT_ROOT
    output_dir = tmp_path / f"schema-prejournal-{position}"
    crash_script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
from mub.vnext.schema_export import export_schemas
atomic._transaction_fault_point=lambda point: os._exit(98) if point == 'journal_new_fsynced' else None
export_schemas(Path(sys.argv[1]))
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(project_root)
    crashed = subprocess.run([sys.executable, "-c", crash_script, str(output_dir)], cwd=project_root, env=env, check=False)
    assert crashed.returncode == 98
    intent_path = output_dir / ".mub-vnext-transaction.json.new"
    intent = _load_json(intent_path)
    entry = intent["entries"][position]
    stage = output_dir / entry["temporary"]
    destination = output_dir / entry["destination"]
    os.link(stage, destination)
    marker = tmp_path / f"schema-prejournal-callback-{position}"
    recovery_script = r'''
import sys
from pathlib import Path
import mub.vnext.schema_export as schema
import mub.vnext.io.atomic as atomic
out=Path(sys.argv[1]); marker=Path(sys.argv[2]); original=atomic.publish_files_atomically
def wrapped(payloads, **kwargs):
    kwargs['pre_publish']=lambda: marker.write_text('partial', encoding='utf-8')
    return original(payloads, **kwargs)
schema.publish_files_atomically=wrapped
schema.export_schemas(out)
'''
    recovered = subprocess.run([sys.executable, "-c", recovery_script, str(output_dir), str(marker)], cwd=project_root, env=env, capture_output=True, text=True, check=False)
    assert recovered.returncode != 0
    assert not marker.exists()
    assert intent_path.exists()
    assert os.path.samefile(stage, destination)


@pytest.mark.parametrize("stage,target_kind", (("journal_prepared", "temporary"), ("backup:0", "backup"), ("commit_marked", "destination")))
def test_atomic_recovery_rejects_in_place_content_tampering(tmp_path: Path, stage: str, target_kind: str) -> None:
    output_dir = tmp_path / f"tamper-{target_kind}"
    output_dir.mkdir()
    for name, data in (("first.json", b"O1"), ("second.json", b"O2")):
        (output_dir / name).write_bytes(data)
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
stage=sys.argv[2]
atomic._transaction_fault_point=lambda point: os._exit(95) if point==stage else None
out=Path(sys.argv[1]); atomic.publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'},overwrite=True)
'''
    env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT)
    crashed = subprocess.run([sys.executable, "-c", script, str(output_dir), stage], cwd=PROJECT_ROOT, env=env, check=False)
    assert crashed.returncode == 95
    journal = _load_json(output_dir / ".mub-vnext-transaction.json")
    entry = journal["entries"][0]
    target = output_dir / (entry[target_kind] if target_kind != "destination" else entry["destination"])
    with target.open("r+b") as handle:
        handle.seek(0); handle.write(b"XX"); handle.flush(); os.fsync(handle.fileno())
    marker = tmp_path / f"callback-{target_kind}"
    recovery_script = "from pathlib import Path; from mub.vnext.io.atomic import publish_files_atomically; p=Path(r'%s'); m=Path(r'%s'); publish_files_atomically({p/'first.json':b'R1',p/'second.json':b'R2'},overwrite=True,pre_publish=lambda:m.write_text('called'))" % (str(output_dir), str(marker))
    recovered = subprocess.run([sys.executable, "-c", recovery_script], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    assert recovered.returncode != 0
    assert not marker.exists()
    assert (output_dir / ".mub-vnext-transaction.json").exists()
    if target_kind in {"backup", "destination"}:
        assert any(output_dir.glob("*.bak.*"))


def test_validator_rejects_runtime_provenance_and_raw_artifact_mismatches(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "tasks"
    run_dir = tmp_path / "runs"
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    assert _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        run_dir,
    ).returncode == 0

    def copied(name: str) -> Path:
        destination = tmp_path / name
        shutil.copytree(run_dir, destination)
        manifest = _load_json(destination / "run_manifest.json")
        manifest["normalized_runtime_artifacts"][0]["path"] = str(
            (destination / "task_runs.jsonl").resolve()
        )
        manifest["score_artifacts"][0]["path"] = str(
            (destination / "scores.jsonl").resolve()
        )
        manifest["native_vs_extracted_field_summary"]["output_hashes"] = {
            str((destination / "task_runs.jsonl").resolve()): _sha256(
                destination / "task_runs.jsonl"
            ),
            str((destination / "scores.jsonl").resolve()): _sha256(
                destination / "scores.jsonl"
            ),
        }
        _write_canonical_json(destination / "run_manifest.json", manifest)
        return destination

    unrelated_hash = _sha256(dataset_source)
    run_mutations = {
        "action_parser": lambda provenance: provenance.__setitem__("action_parser_version", "forged"),
        "answer_parser": lambda provenance: provenance.__setitem__("answer_parser_version", "forged"),
        "entry_extractor": lambda provenance: provenance.__setitem__("memory_entry_extractor_version", "forged"),
        "redaction": lambda provenance: provenance.__setitem__("redaction_policy_version", "forged"),
        "object_hash": lambda provenance: provenance.__setitem__("object_value_extractor_config_hash", "a" * 64),
        "provider_path": lambda provenance: provenance.__setitem__("raw_provider_artifact_path", str(dataset_source.resolve())),
        "provider_hash": lambda provenance: provenance.__setitem__("raw_provider_artifact_hash", unrelated_hash),
        "provider_path_only": lambda provenance: provenance.__setitem__("raw_provider_artifact_hash", None),
        "provider_hash_only": lambda provenance: provenance.__setitem__("raw_provider_artifact_path", None),
        "adapter_pair": lambda provenance: provenance.update({"raw_adapter_state_path": str(dataset_source.resolve()), "raw_adapter_state_hash": unrelated_hash}),
        "adapter_path_only": lambda provenance: provenance.__setitem__("raw_adapter_state_path", str(dataset_source.resolve())),
        "adapter_hash_only": lambda provenance: provenance.__setitem__("raw_adapter_state_hash", unrelated_hash),
    }
    for name, mutate in run_mutations.items():
        directory = copied(f"run-{name}")
        rows = _load_jsonl(directory / "task_runs.jsonl")
        mutate(rows[0]["parser_extractor_provenance"])
        _write_canonical_jsonl(directory / "task_runs.jsonl", rows)
        manifest = _load_json(directory / "run_manifest.json")
        digest = _sha256(directory / "task_runs.jsonl")
        manifest["normalized_runtime_artifacts"][0]["sha256"] = digest
        manifest["native_vs_extracted_field_summary"]["output_hashes"][
            str((directory / "task_runs.jsonl").resolve())
        ] = digest
        _write_canonical_json(directory / "run_manifest.json", manifest)
        for kind, artifact in (
            ("task-runs", directory / "task_runs.jsonl"),
            ("run-manifest", directory / "run_manifest.json"),
        ):
            result = _run(
                VALIDATE_CLI,
                "--kind",
                kind,
                "--input",
                artifact,
                "--manifest",
                directory / "run_manifest.json",
            )
            assert result.returncode != 0, (name, kind)

    manifest_mutations = {
        "duplicate_provider": lambda manifest: manifest["raw_provider_response_artifacts"].append(dict(manifest["raw_provider_response_artifacts"][0])),
        "missing_provider": lambda manifest: manifest.__setitem__("raw_provider_response_artifacts", []),
        "manifest_action_parser": lambda manifest: manifest.__setitem__("action_parser_version", "forged"),
        "manifest_answer_parser": lambda manifest: manifest.__setitem__("answer_parser_version", "forged"),
        "manifest_entry_extractor": lambda manifest: manifest.__setitem__("memory_entry_extractor_version", "forged"),
        "manifest_redaction": lambda manifest: manifest.__setitem__("redaction_policy_version", "forged"),
        "manifest_object_hash": lambda manifest: manifest.__setitem__("object_value_extractor_config_hash", "a" * 64),
    }
    for name, mutate in manifest_mutations.items():
        directory = copied(f"manifest-provenance-{name}")
        manifest = _load_json(directory / "run_manifest.json")
        mutate(manifest)
        _write_canonical_json(directory / "run_manifest.json", manifest)
        for kind, artifact in (
            ("task-runs", directory / "task_runs.jsonl"),
            ("run-manifest", directory / "run_manifest.json"),
        ):
            result = _run(
                VALIDATE_CLI,
                "--kind",
                kind,
                "--input",
                artifact,
                "--manifest",
                directory / "run_manifest.json",
            )
            assert result.returncode != 0, (name, kind)


def test_atomic_no_clobber_rollback_failure_preserves_staged_recovery_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    real_link = atomic_module.os.link
    real_unlink = Path.unlink
    links = 0

    def fail_second_link(source: Path, destination: Path, *args, **kwargs):
        nonlocal links
        links += 1
        if links == 2:
            raise FileExistsError("injected second link failure")
        return real_link(source, destination, *args, **kwargs)

    def deny_first_rollback(path: Path, *args, **kwargs):
        if path == first:
            raise PermissionError("injected rollback delete failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(atomic_module.os, "link", fail_second_link)
    monkeypatch.setattr(Path, "unlink", deny_first_rollback)
    with pytest.raises(RuntimeError, match="rollback.*recovery") as exc_info:
        atomic_module.publish_files_atomically(
            {first: b"A1", second: b"A2"}, overwrite=False
        )
    assert isinstance(exc_info.value.__cause__, FileExistsError)
    recovery = list(tmp_path.glob("first.json.tmp.*"))
    assert len(recovery) == 1
    assert first.read_bytes() == recovery[0].read_bytes() == b"A1"
    assert os.path.samefile(first, recovery[0])
    assert not second.exists()

    monkeypatch.undo()
    first.unlink()
    atomic_module.publish_files_atomically(
        {first: b"A1", second: b"A2"}, overwrite=False
    )
    assert (first.read_bytes(), second.read_bytes()) == (b"A1", b"A2")
    assert recovery[0].read_bytes() == b"A1"
    recovery[0].unlink()


def test_validator_rejects_coordinated_legacy_extractor_and_identity_tampering(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "tasks"
    run_dir = tmp_path / "runs"
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    assert _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        run_dir,
    ).returncode == 0

    def tampered(name: str, mode: str) -> Path:
        directory = tmp_path / name
        shutil.copytree(run_dir, directory)
        run_path = directory / "task_runs.jsonl"
        score_path = directory / "scores.jsonl"
        rows = _load_jsonl(run_path)
        scores = _load_jsonl(score_path)
        manifest = _load_json(directory / "run_manifest.json")
        arbitrary = "a" * 64
        if mode == "coordinated":
            for row in rows:
                row["parser_extractor_provenance"][
                    "object_value_extractor_config_hash"
                ] = arbitrary
            manifest["object_value_extractor_config_hash"] = arbitrary
        elif mode == "mixed":
            rows[-1]["parser_extractor_provenance"][
                "object_value_extractor_config_hash"
            ] = arbitrary
        elif mode == "identity":
            manifest["adapter_info"].update(
                {
                    "adapter_id": "adapter_forged",
                    "adapter_version": "1.0.0",
                    "system_name": "fixture_memory_system",
                    "system_version": "1.0.0",
                }
            )
            for row in rows:
                row["adapter_id"] = "adapter_forged"
            for score in scores:
                score["adapter_id"] = "adapter_forged"
        else:
            raise AssertionError(mode)
        _write_canonical_jsonl(run_path, rows)
        _write_canonical_jsonl(score_path, scores)
        manifest["normalized_runtime_artifacts"][0].update(
            {"path": str(run_path.resolve()), "sha256": _sha256(run_path)}
        )
        manifest["score_artifacts"][0].update(
            {"path": str(score_path.resolve()), "sha256": _sha256(score_path)}
        )
        manifest["native_vs_extracted_field_summary"]["output_hashes"] = {
            str(run_path.resolve()): _sha256(run_path),
            str(score_path.resolve()): _sha256(score_path),
        }
        _write_canonical_json(directory / "run_manifest.json", manifest)
        return directory

    for mode in ("coordinated", "mixed", "identity"):
        directory = tampered(f"tampered-{mode}", mode)
        for kind, artifact in (
            ("task-runs", directory / "task_runs.jsonl"),
            ("run-manifest", directory / "run_manifest.json"),
        ):
            result = _run(
                VALIDATE_CLI,
                "--kind",
                kind,
                "--input",
                artifact,
                "--manifest",
                directory / "run_manifest.json",
            )
            assert result.returncode != 0, (mode, kind, result.stdout)


def test_validator_rejects_legacy_raw_artifact_manifest_tampering(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "tasks"
    run_dir = tmp_path / "runs"
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    assert _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        run_dir,
    ).returncode == 0

    extra_ref = {
        "path": str(dataset_source.resolve()),
        "sha256": _sha256(dataset_source),
        "media_type": "application/json",
        "record_count": 2,
    }

    def tampered(name: str, mode: str) -> Path:
        directory = tmp_path / name
        shutil.copytree(run_dir, directory)
        run_path = directory / "task_runs.jsonl"
        score_path = directory / "scores.jsonl"
        rows = _load_jsonl(run_path)
        manifest = _load_json(directory / "run_manifest.json")
        provider = manifest["raw_provider_response_artifacts"][0]
        if mode == "provider_media":
            provider["media_type"] = "application/octet-stream"
        elif mode == "provider_null_count":
            provider["record_count"] = None
        elif mode == "provider_wrong_count":
            provider["record_count"] = 1
        elif mode == "provider_extra":
            manifest["raw_provider_response_artifacts"].append(dict(extra_ref))
        elif mode == "adapter_extra":
            manifest["raw_adapter_state_artifacts"].append(dict(extra_ref))
        elif mode == "adapter_pair":
            manifest["raw_adapter_state_artifacts"].append(dict(extra_ref))
            for row in rows:
                row["parser_extractor_provenance"].update(
                    {
                        "raw_adapter_state_path": extra_ref["path"],
                        "raw_adapter_state_hash": extra_ref["sha256"],
                    }
                )
        else:
            raise AssertionError(mode)
        _write_canonical_jsonl(run_path, rows)
        manifest["normalized_runtime_artifacts"][0].update(
            {"path": str(run_path.resolve()), "sha256": _sha256(run_path)}
        )
        manifest["score_artifacts"][0]["path"] = str(score_path.resolve())
        manifest["native_vs_extracted_field_summary"]["output_hashes"] = {
            str(run_path.resolve()): _sha256(run_path),
            str(score_path.resolve()): _sha256(score_path),
        }
        manifest["native_vs_extracted_field_summary"]["input_hashes"][
            extra_ref["path"]
        ] = extra_ref["sha256"]
        _write_canonical_json(directory / "run_manifest.json", manifest)
        return directory

    modes = (
        "provider_media",
        "provider_null_count",
        "provider_wrong_count",
        "provider_extra",
        "adapter_extra",
        "adapter_pair",
    )
    for mode in modes:
        directory = tampered(f"raw-{mode}", mode)
        for kind, artifact in (
            ("task-runs", directory / "task_runs.jsonl"),
            ("run-manifest", directory / "run_manifest.json"),
        ):
            result = _run(
                VALIDATE_CLI,
                "--kind",
                kind,
                "--input",
                artifact,
                "--manifest",
                directory / "run_manifest.json",
            )
            assert result.returncode != 0, (mode, kind, result.stdout)




def test_validator_reconstructs_legacy_results_from_authenticated_source(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "reconstruct-tasks"
    run_dir = tmp_path / "reconstruct-runs"
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    assert _run(COMPILE_CLI, "dataset", "--input", dataset_source, "--split", "test", "--legacy-phase", "P6.3", "--output-dir", task_dir).returncode == 0
    assert _run(COMPILE_CLI, "results", "--input", results_source, "--tasks", task_dir / "tasks.jsonl", "--output-dir", run_dir).returncode == 0

    def forged(mode: str) -> Path:
        directory = tmp_path / f"reconstruct-{mode}"
        shutil.copytree(run_dir, directory)
        run_path, score_path = directory / "task_runs.jsonl", directory / "scores.jsonl"
        runs, scores = _load_jsonl(run_path), _load_jsonl(score_path)
        manifest = _load_json(directory / "run_manifest.json")
        if mode == "runtime":
            runs[0]["system_events"].append({"forged": True})
        elif mode == "score":
            scores[0]["answer_scores"]["exact_match"] = 0.0
        elif mode == "legacy_metrics":
            scores[0]["legacy_metrics"]["legacy_p63"]["em"] = 0.5
        elif mode == "coordinated":
            run_id = "legacy_run_" + "a" * 64
            adapter_id = "legacy_evomemory_" + "a" * 16
            manifest["run_id"] = run_id
            manifest["adapter_info"]["adapter_id"] = adapter_id
            manifest["code_revision"] = "forged-revision"
            manifest["environment_summary"] = {"legacy_import": True, "forged": True}
            manifest["completed_task_count"] = 1
            manifest["failed_task_count"] = 1
            runs[0]["completion_status"] = "failed"
            scores[0]["completion_status"] = "failed"
            for row in runs + scores:
                row["run_id"] = run_id
                row["adapter_id"] = adapter_id
        else:
            raise AssertionError(mode)
        _write_canonical_jsonl(run_path, runs)
        _write_canonical_jsonl(score_path, scores)
        manifest["normalized_runtime_artifacts"][0].update({"path": str(run_path.resolve()), "sha256": _sha256(run_path)})
        manifest["score_artifacts"][0].update({"path": str(score_path.resolve()), "sha256": _sha256(score_path)})
        manifest["native_vs_extracted_field_summary"]["output_hashes"] = {str(run_path.resolve()): _sha256(run_path), str(score_path.resolve()): _sha256(score_path)}
        _write_canonical_json(directory / "run_manifest.json", manifest)
        return directory

    for mode in ("runtime", "score", "legacy_metrics", "coordinated"):
        directory = forged(mode)
        for kind, artifact in (("task-runs", directory / "task_runs.jsonl"), ("run-manifest", directory / "run_manifest.json")):
            result = _run(VALIDATE_CLI, "--kind", kind, "--input", artifact, "--manifest", directory / "run_manifest.json")
            assert result.returncode != 0, (mode, kind, result.stdout)

    relocated = tmp_path / "relocated-authenticated"
    shutil.copytree(run_dir, relocated)
    manifest = _load_json(relocated / "run_manifest.json")
    manifest["normalized_runtime_artifacts"][0]["path"] = str((relocated / "task_runs.jsonl").resolve())
    manifest["score_artifacts"][0]["path"] = str((relocated / "scores.jsonl").resolve())
    manifest["native_vs_extracted_field_summary"]["output_hashes"] = {str((relocated / "task_runs.jsonl").resolve()): _sha256(relocated / "task_runs.jsonl"), str((relocated / "scores.jsonl").resolve()): _sha256(relocated / "scores.jsonl")}
    _write_canonical_json(relocated / "run_manifest.json", manifest)
    accepted = _run(VALIDATE_CLI, "--kind", "run-manifest", "--input", relocated / "run_manifest.json", "--manifest", relocated / "run_manifest.json")
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr




def test_results_require_explicit_declaration_for_partial_task_coverage(tmp_path: Path) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    task_dir = tmp_path / "selection-tasks"
    assert _run(COMPILE_CLI, "dataset", "--input", source, "--split", "test", "--legacy-phase", "P6.3", "--output-dir", task_dir).returncode == 0
    payload = _load_json(LEGACY_FIXTURES / "evomemory_results_old.json")
    payload["results"] = [payload["results"][1]]
    payload["results"][0]["shard_local_example_id"] = 0
    payload["summary"]["num_examples"] = 1
    payload["summary"].pop("start_idx", None)
    payload["summary"].pop("end_idx", None)
    undeclared = tmp_path / "undeclared-partial.json"
    undeclared.write_text(json.dumps(payload), encoding="utf-8")
    rejected = _run(COMPILE_CLI, "results", "--input", undeclared, "--tasks", task_dir / "tasks.jsonl", "--output-dir", tmp_path / "rejected")
    assert rejected.returncode != 0
    assert not (tmp_path / "rejected" / "run_manifest.json").exists()

    payload["summary"].update({"start_idx": 1, "end_idx": 2})
    declared = tmp_path / "declared-partial.json"
    declared.write_text(json.dumps(payload), encoding="utf-8")
    accepted_dir = tmp_path / "accepted"
    accepted = _run(COMPILE_CLI, "results", "--input", declared, "--tasks", task_dir / "tasks.jsonl", "--output-dir", accepted_dir)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    validated = _run(VALIDATE_CLI, "--kind", "run-manifest", "--input", accepted_dir / "run_manifest.json", "--manifest", accepted_dir / "run_manifest.json")
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_runtime_provenance_preserves_nonlegacy_exact_hash_semantics(
    tmp_path: Path,
) -> None:
    from mub.vnext.contracts import ArtifactRef
    from scripts.vnext_validate_artifacts import _validate_runtime_provenance
    from tests.vnext.factories import build_task_run, make_run_manifest

    run = build_task_run()
    run_payload = run.model_dump(mode="python")
    run_payload["parser_extractor_provenance"].update(
        {
            "raw_provider_artifact_path": None,
            "raw_provider_artifact_hash": None,
            "raw_adapter_state_path": None,
            "raw_adapter_state_hash": None,
        }
    )
    run = TaskRunRecord.model_validate(run_payload)
    manifest = make_run_manifest(
        raw_provider_response_artifacts=[],
        raw_adapter_state_artifacts=[],
    )

    _validate_runtime_provenance(manifest, [run], tmp_path / "run_manifest.json")

    mismatched_payload = run.model_dump(mode="python")
    mismatched_payload["parser_extractor_provenance"][
        "object_value_extractor_config_hash"
    ] = "2" * 64
    mismatched = TaskRunRecord.model_validate(mismatched_payload)
    with pytest.raises(ValueError, match="object extractor hash"):
        _validate_runtime_provenance(
            manifest, [mismatched], tmp_path / "run_manifest.json"
        )

    provider_path = tmp_path / "provider.bin"
    adapter_path = tmp_path / "adapter.bin"
    provider_path.write_bytes(b"provider")
    adapter_path.write_bytes(b"adapter")
    provider_hash = _sha256(provider_path)
    adapter_hash = _sha256(adapter_path)
    materialized_payload = run.model_dump(mode="python")
    materialized_payload["parser_extractor_provenance"].update(
        {
            "raw_provider_artifact_path": str(provider_path),
            "raw_provider_artifact_hash": provider_hash,
            "raw_adapter_state_path": str(adapter_path),
            "raw_adapter_state_hash": adapter_hash,
        }
    )
    materialized = TaskRunRecord.model_validate(materialized_payload)
    materialized_manifest = make_run_manifest(
        raw_provider_response_artifacts=[
            ArtifactRef(
                path=str(provider_path),
                sha256=provider_hash,
                media_type="application/octet-stream",
                record_count=None,
            )
        ],
        raw_adapter_state_artifacts=[
            ArtifactRef(
                path=str(adapter_path),
                sha256=adapter_hash,
                media_type="application/octet-stream",
                record_count=None,
            )
        ],
    )
    _validate_runtime_provenance(
        materialized_manifest,
        [materialized],
        tmp_path / "run_manifest.json",
    )


def test_atomic_cross_process_lock_prevents_mixed_generations(tmp_path: Path) -> None:
    output_dir = tmp_path / "shared"
    output_dir.mkdir()
    script = r'''
import sys, time
from pathlib import Path
from mub.vnext.io.atomic import publish_files_atomically
out = Path(sys.argv[1]); label = sys.argv[2]; mine = Path(sys.argv[3]); other = Path(sys.argv[4])
mine.write_text(label, encoding="utf-8")
while not other.exists():
    time.sleep(0.01)
publish_files_atomically(
    {out / "first.json": (label + "1").encode(), out / "second.json": (label + "2").encode()},
    overwrite=True,
    pre_publish=lambda: time.sleep(0.2),
)
'''
    marker_a = tmp_path / "a.ready"
    marker_b = tmp_path / "b.ready"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(output_dir), "A", str(marker_a), str(marker_b)],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
        subprocess.Popen(
            [sys.executable, "-c", script, str(output_dir), "B", str(marker_b), str(marker_a)],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
    generation = (
        (output_dir / "first.json").read_bytes(),
        (output_dir / "second.json").read_bytes(),
    )
    assert generation in {(b"A1", b"A2"), (b"B1", b"B2")}
    assert not list(output_dir.glob("*.tmp*"))
    assert not list(output_dir.glob("*.bak*"))


def test_cli_rejects_argument_abbreviations_and_undocumented_run_name(
    tmp_path: Path,
) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    output_dir = tmp_path / "abbreviated"
    abbreviated = _run(
        COMPILE_CLI,
        "dataset",
        "--inp",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        output_dir,
    )
    assert abbreviated.returncode != 0
    assert not output_dir.exists()

    undocumented = _run(
        COMPILE_CLI,
        "results",
        "--input",
        LEGACY_FIXTURES / "evomemory_results_old.json",
        "--tasks",
        tmp_path / "tasks.jsonl",
        "--output-dir",
        tmp_path / "runs",
        "--run-name",
        "raw_add_slot_prompt_k16",
    )
    assert undocumented.returncode != 0


def test_validator_argument_errors_are_one_canonical_nonsecret_json_report() -> None:
    expected = {
        "errors": [{"code": "invalid_arguments", "type": "ArgumentError"}],
        "kind": None,
        "valid": False,
        "warnings": [],
    }
    expected_text = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"
    cases = [
        ("--ki", "tasks", "--input", "x", "--manifest", "y"),
        ("--kind", "tasks", "--secret-token", "SENSITIVE"),
        ("--kind", "unknown", "--input", "x", "--manifest", "y"),
    ]
    for args in cases:
        result = _run(VALIDATE_CLI, *args)
        assert result.returncode != 0
        assert result.stdout == expected_text
        assert result.stderr == ""
        assert "SENSITIVE" not in result.stdout


def test_results_rejects_tampered_task_manifest_and_profile_linkage(
    tmp_path: Path,
) -> None:
    source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    original = tmp_path / "original"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        original,
    ).returncode == 0

    count_dir = tmp_path / "count"
    shutil.copytree(original, count_dir)
    count_manifest = _load_json(count_dir / "task_manifest.json")
    count_manifest["task_file_paths_and_hashes"][0]["path"] = str(
        (count_dir / "tasks.jsonl").resolve()
    )
    count_manifest["split_counts"]["test"] = 999
    _write_canonical_json(count_dir / "task_manifest.json", count_manifest)
    count_result = _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        count_dir / "tasks.jsonl",
        "--output-dir",
        tmp_path / "count-runs",
    )
    assert count_result.returncode != 0
    assert not (tmp_path / "count-runs" / "run_manifest.json").exists()

    profile_dir = tmp_path / "profile"
    shutil.copytree(original, profile_dir)
    rows = _load_jsonl(profile_dir / "tasks.jsonl")
    rows[0]["metadata"]["resolved_profile"].pop("update_depth")
    _write_canonical_jsonl(profile_dir / "tasks.jsonl", rows)
    profile_hash = _sha256(profile_dir / "tasks.jsonl")
    profile_manifest = _load_json(profile_dir / "task_manifest.json")
    profile_manifest["task_file_paths_and_hashes"][0].update(
        {"path": str((profile_dir / "tasks.jsonl").resolve()), "sha256": profile_hash}
    )
    profile_manifest["leakage_check_summary"]["output_hashes"] = {
        str((profile_dir / "tasks.jsonl").resolve()): profile_hash
    }
    _write_canonical_json(profile_dir / "task_manifest.json", profile_manifest)
    profile_result = _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        profile_dir / "tasks.jsonl",
        "--output-dir",
        tmp_path / "profile-runs",
    )
    assert profile_result.returncode != 0
    assert not (tmp_path / "profile-runs" / "run_manifest.json").exists()


def test_results_supports_authenticated_nonzero_shard_of_full_task_artifact(
    tmp_path: Path,
) -> None:
    names = [
        "Alice", "Bob", "Carol", "David", "Erin", "Frank",
        "Grace", "Henry", "Irene", "Jack", "Karen", "Liam",
    ]
    template = _load_json(LEGACY_FIXTURES / "p63_dataset_minimal.json")[0]
    episodes = []
    for index, name in enumerate(names):
        episode = json.loads(json.dumps(template))
        episode["episode_id"] = f"shard-task-{index}"
        episode["events"] = [text.replace("Alex", name) for text in episode["events"]]
        episode["question"] = episode["question"].replace("Alex", name)
        episode["entity"] = f"friend_{name.lower()}"
        episode["same_name_distractor"]["entity"] = f"manager_{name.lower()}"
        episode["same_name_distractor"]["surface_name"] = name
        episode["semantic_near_miss"]["entity"] = f"friend_{name.lower()}"
        episodes.append(episode)
    dataset_source = tmp_path / "twelve-tasks.json"
    dataset_source.write_text(
        json.dumps(episodes, ensure_ascii=False), encoding="utf-8", newline=""
    )
    task_dir = tmp_path / "tasks"
    compiled = _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    )
    assert compiled.returncode == 0, compiled.stderr

    payload = {
        "summary": {
            "benchmark": "evomemory",
            "mode": "raw_add",
            "answer_mode": "slot_prompt",
            "num_examples": 2,
            "start_idx": 10,
            "end_idx": 12,
            "legacy_analysis_metadata": {"legacy_phase": "P6.3"},
        },
        "results": [
            {
                "example_id": index,
                "shard_local_example_id": index - 10,
                "question": episodes[index]["question"],
                "gold_answer": "Suzhou",
                "predicted": "Suzhou",
                "em": 1.0,
                "f1": 1.0,
                "state_value_em": True,
            }
            for index in (10, 11)
        ],
    }
    result_source = tmp_path / "shard-results.json"
    result_source.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline=""
    )
    output_dir = tmp_path / "runs"
    imported = _run(
        COMPILE_CLI,
        "results",
        "--input",
        result_source,
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        output_dir,
    )
    assert imported.returncode == 0, imported.stderr
    runs = [TaskRunRecord.model_validate(row) for row in _load_jsonl(output_dir / "task_runs.jsonl")]
    all_tasks = [MemUpdateTask.model_validate(row) for row in _load_jsonl(task_dir / "tasks.jsonl")]
    assert [run.task_id for run in runs] == [all_tasks[10].task_id, all_tasks[11].task_id]
    manifest = RunManifest.model_validate(_load_json(output_dir / "run_manifest.json"))
    assert manifest.task_manifest.record_count == 12
    assert manifest.prompt_config["legacy_result_import"]["compiled_task_selection"][
        "legacy_indices"
    ] == (10, 11)
    validated = _run(
        VALIDATE_CLI,
        "--kind",
        "run-manifest",
        "--input",
        output_dir / "run_manifest.json",
        "--manifest",
        output_dir / "run_manifest.json",
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_atomic_rejects_existing_destination_aliases(tmp_path: Path) -> None:
    from mub.vnext.io.atomic import publish_files_atomically

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"original")
    os.link(first, second)
    with pytest.raises(ValueError, match="alias|same file"):
        publish_files_atomically(
            {first: b"new-first", second: b"new-second"}, overwrite=True
        )
    assert first.read_bytes() == second.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))


def test_atomic_no_overwrite_rechecks_after_prepublish(tmp_path: Path) -> None:
    from mub.vnext.io.atomic import publish_files_atomically

    destination = tmp_path / "created-concurrently.json"

    def create_destination() -> None:
        destination.write_bytes(b"concurrent")

    with pytest.raises(FileExistsError):
        publish_files_atomically(
            {destination: b"new"},
            overwrite=False,
            pre_publish=create_destination,
        )
    assert destination.read_bytes() == b"concurrent"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))


def test_atomic_postcommit_backup_cleanup_failure_never_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    real_unlink = Path.unlink
    failed_once = False

    def transient_backup_unlink(path: Path, *args, **kwargs):
        nonlocal failed_once
        if ".bak." in path.name and not failed_once:
            failed_once = True
            raise OSError("injected backup cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transient_backup_unlink)
    atomic_module.publish_files_atomically(
        {first: b"new-first", second: b"new-second"}, overwrite=True
    )
    assert first.read_bytes() == b"new-first"
    assert second.read_bytes() == b"new-second"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))


def test_atomic_restore_failure_preserves_recoverable_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    real_replace = atomic_module.os.replace

    def injected_replace(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == second and ".tmp." in source_path.name:
            raise OSError("injected publish failure")
        if destination_path == first and ".bak." in source_path.name:
            raise OSError("injected restore failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(atomic_module.os, "replace", injected_replace)
    with pytest.raises(Exception, match="restore|rollback|publish"):
        atomic_module.publish_files_atomically(
            {first: b"new-first", second: b"new-second"}, overwrite=True
        )
    backups = list(tmp_path.glob("first.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert not list(tmp_path.glob("*.tmp*"))


def test_mechanism_rejects_conflicting_duplicate_logical_identity(
    tmp_path: Path,
) -> None:
    fixture = LEGACY_FIXTURES / "p83_conflict_rows.csv"
    lines = fixture.read_text(encoding="utf-8").splitlines()
    headers = lines[0].split(",")
    values = lines[1].split(",")
    em_index = headers.index("em")
    values[em_index] = "0.123"
    source = tmp_path / "conflicting.csv"
    source.write_text(
        "\n".join([*lines, ",".join(values)]) + "\n",
        encoding="utf-8",
        newline="",
    )
    output_dir = tmp_path / "out"
    result = _run(
        COMPILE_CLI,
        "mechanism",
        "--kind",
        "conflict",
        "--input",
        source,
        "--output-dir",
        output_dir,
    )
    assert result.returncode != 0
    assert not output_dir.exists()


def test_legacy_analysis_source_media_types_are_exact(tmp_path: Path) -> None:
    from mub.vnext.legacy.artifacts import LegacyAnalysisManifest

    mechanism_dir = tmp_path / "mechanism"
    assert _run(
        COMPILE_CLI,
        "mechanism",
        "--kind",
        "conflict",
        "--input",
        LEGACY_FIXTURES / "p83_conflict_rows.csv",
        "--output-dir",
        mechanism_dir,
    ).returncode == 0
    mechanism_manifest = LegacyAnalysisManifest.model_validate(
        _load_json(mechanism_dir / "legacy_analysis_manifest.json")
    )
    assert mechanism_manifest.source_artifacts[0].media_type == "text/csv"

    json_source = tmp_path / "conflict.json"
    with (LEGACY_FIXTURES / "p83_conflict_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        json_source.write_text(
            json.dumps(list(csv.DictReader(handle)), ensure_ascii=False),
            encoding="utf-8",
            newline="",
        )
    json_dir = tmp_path / "json-mechanism"
    assert _run(
        COMPILE_CLI,
        "mechanism",
        "--kind",
        "conflict",
        "--input",
        json_source,
        "--output-dir",
        json_dir,
    ).returncode == 0
    json_manifest = LegacyAnalysisManifest.model_validate(
        _load_json(json_dir / "legacy_analysis_manifest.json")
    )
    assert json_manifest.source_artifacts[0].media_type == "application/json"

    ledger_dir = tmp_path / "ledger"
    assert _run(
        COMPILE_CLI,
        "ledger",
        "--input",
        LEGACY_FIXTURES / "ledger_references.md",
        "--project-root",
        PROJECT_ROOT,
        "--output-dir",
        ledger_dir,
    ).returncode == 0
    ledger_manifest = LegacyAnalysisManifest.model_validate(
        _load_json(ledger_dir / "legacy_analysis_manifest.json")
    )
    assert ledger_manifest.source_artifacts[0].media_type == "text/markdown"

    wrong_extension = tmp_path / "ledger.txt"
    wrong_extension.write_bytes(
        (LEGACY_FIXTURES / "ledger_references.md").read_bytes()
    )
    refused = _run(
        COMPILE_CLI,
        "ledger",
        "--input",
        wrong_extension,
        "--project-root",
        PROJECT_ROOT,
        "--output-dir",
        tmp_path / "wrong-ledger",
    )
    assert refused.returncode != 0
    assert not (tmp_path / "wrong-ledger").exists()


def test_results_and_validator_authenticate_entire_relocated_task_manifest(
    tmp_path: Path,
) -> None:
    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    results_source = LEGACY_FIXTURES / "evomemory_results_old.json"
    original = tmp_path / "original"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        original,
    ).returncode == 0

    def relocate(destination: Path) -> dict[str, object]:
        shutil.copytree(original, destination)
        manifest = _load_json(destination / "task_manifest.json")
        tasks_path = (destination / "tasks.jsonl").resolve()
        manifest["task_file_paths_and_hashes"][0]["path"] = str(tasks_path)
        manifest["leakage_check_summary"]["output_hashes"] = {
            str(tasks_path): _sha256(tasks_path)
        }
        _write_canonical_json(destination / "task_manifest.json", manifest)
        return manifest

    valid_dir = tmp_path / "relocated-valid"
    relocate(valid_dir)
    valid_result = _run(
        COMPILE_CLI,
        "results",
        "--input",
        results_source,
        "--tasks",
        valid_dir / "tasks.jsonl",
        "--output-dir",
        tmp_path / "valid-runs",
    )
    assert valid_result.returncode == 0, valid_result.stderr

    def source_ref(manifest: dict[str, object]) -> dict[str, object]:
        return json.loads(json.dumps(manifest["source_manifest_paths_and_hashes"][0]))

    mutations = {
        "data_release": lambda manifest: manifest.__setitem__("data_release_id", "forged"),
        "split_policy": lambda manifest: manifest.__setitem__("split_policy_version", "forged"),
        "compiler_map": lambda manifest: manifest.__setitem__("compiler_versions", {"forged": "1.0.0"}),
        "source_record_count": lambda manifest: manifest["source_manifest_paths_and_hashes"][0].__setitem__("record_count", 999),
        "source_media_type": lambda manifest: manifest["source_manifest_paths_and_hashes"][0].__setitem__("media_type", "text/plain"),
        "generation_configs": lambda manifest: manifest.__setitem__("generation_configs_and_hashes", [source_ref(manifest)]),
        "input_hashes": lambda manifest: manifest["leakage_check_summary"].__setitem__("input_hashes", {str(dataset_source.resolve()): "f" * 64}),
        "warnings": lambda manifest: manifest["leakage_check_summary"].__setitem__("warnings", ["forged"]),
        "task_hashes": lambda manifest: manifest["leakage_check_summary"].__setitem__("task_hashes", {}),
        "strata": lambda manifest: manifest["leakage_check_summary"].__setitem__("required_minimum_strata", []),
        "deviations": lambda manifest: manifest["leakage_check_summary"].__setitem__("small_cell_deviations", [{"forged": True}]),
        "audit_artifacts": lambda manifest: manifest.__setitem__("human_audit_artifacts", [source_ref(manifest)]),
        "created_at": lambda manifest: manifest.__setitem__("created_at", "forged"),
        "code_revision": lambda manifest: manifest.__setitem__("code_revision", "forged"),
    }
    for name, mutate in mutations.items():
        forged_dir = tmp_path / f"forged-{name}"
        manifest = relocate(forged_dir)
        mutate(manifest)
        _write_canonical_json(forged_dir / "task_manifest.json", manifest)

        compiled = _run(
            COMPILE_CLI,
            "results",
            "--input",
            results_source,
            "--tasks",
            forged_dir / "tasks.jsonl",
            "--output-dir",
            tmp_path / f"runs-{name}",
        )
        assert compiled.returncode != 0, name
        assert not (tmp_path / f"runs-{name}" / "run_manifest.json").exists()

        validated = _run(
            VALIDATE_CLI,
            "--kind",
            "tasks",
            "--input",
            forged_dir / "tasks.jsonl",
            "--manifest",
            forged_dir / "task_manifest.json",
        )
        assert validated.returncode != 0, name
        assert json.loads(validated.stdout)["valid"] is False


def test_source_recompilation_rejects_forged_canonical_task_and_rebuilt_manifest(
    tmp_path: Path,
) -> None:
    from mub.vnext.io.canonical import sha256_model
    from mub.vnext.legacy.artifacts import build_expected_legacy_task_manifest

    dataset_source = LEGACY_FIXTURES / "p63_dataset_minimal.json"
    task_dir = tmp_path / "tasks"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        dataset_source,
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    rows = _load_jsonl(task_dir / "tasks.jsonl")
    rows[0]["queries"][0]["text"] += " forged"
    _write_canonical_jsonl(task_dir / "tasks.jsonl", rows)
    tasks = [MemUpdateTask.model_validate(row) for row in rows]
    with pytest.raises(ValueError, match="source compilation"):
        build_expected_legacy_task_manifest(
            tasks, tasks_path=task_dir / "tasks.jsonl"
        )
    forged_manifest = _load_json(task_dir / "task_manifest.json")
    tasks_hash = _sha256(task_dir / "tasks.jsonl")
    forged_manifest["task_file_paths_and_hashes"][0]["sha256"] = tasks_hash
    forged_manifest["leakage_check_summary"]["output_hashes"] = {
        str((task_dir / "tasks.jsonl").resolve()): tasks_hash
    }
    forged_manifest["leakage_check_summary"]["task_hashes"] = {
        task.task_id: sha256_model(task)
        for task in sorted(tasks, key=lambda item: item.task_id)
    }
    _write_canonical_json(task_dir / "task_manifest.json", forged_manifest)

    compiled = _run(
        COMPILE_CLI,
        "results",
        "--input",
        LEGACY_FIXTURES / "evomemory_results_old.json",
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        tmp_path / "runs",
    )
    assert compiled.returncode != 0
    assert not (tmp_path / "runs").exists()
    validated = _run(
        VALIDATE_CLI,
        "--kind",
        "tasks",
        "--input",
        task_dir / "tasks.jsonl",
        "--manifest",
        task_dir / "task_manifest.json",
    )
    assert validated.returncode != 0


def test_compile_argument_errors_are_canonical_and_secret_free() -> None:
    expected = (
        '{"errors":[{"code":"invalid_arguments","type":"ArgumentError"}],'
        '"kind":null,"valid":false,"warnings":[]}\n'
    )
    for args in (
        ("dataset", "--inp", "SECRET"),
        ("--secret-token", "SENSITIVE"),
        ("results", "--run-name", "SECRET"),
    ):
        result = _run(COMPILE_CLI, *args)
        assert result.returncode != 0
        assert result.stdout == expected
        assert result.stderr == ""
        assert "SECRET" not in result.stdout
        assert "SENSITIVE" not in result.stdout


def test_validator_rejects_score_linkage_media_counts_and_actual_row_summary(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "tasks"
    run_dir = tmp_path / "runs"
    assert _run(
        COMPILE_CLI,
        "dataset",
        "--input",
        LEGACY_FIXTURES / "p63_dataset_minimal.json",
        "--split",
        "test",
        "--legacy-phase",
        "P6.3",
        "--output-dir",
        task_dir,
    ).returncode == 0
    assert _run(
        COMPILE_CLI,
        "results",
        "--input",
        LEGACY_FIXTURES / "evomemory_results_old.json",
        "--tasks",
        task_dir / "tasks.jsonl",
        "--output-dir",
        run_dir,
    ).returncode == 0

    def copied(name: str) -> Path:
        destination = tmp_path / name
        shutil.copytree(run_dir, destination)
        manifest = _load_json(destination / "run_manifest.json")
        manifest["normalized_runtime_artifacts"][0]["path"] = str(
            (destination / "task_runs.jsonl").resolve()
        )
        manifest["score_artifacts"][0]["path"] = str(
            (destination / "scores.jsonl").resolve()
        )
        manifest["native_vs_extracted_field_summary"]["output_hashes"] = {
            str((destination / "task_runs.jsonl").resolve()): _sha256(
                destination / "task_runs.jsonl"
            ),
            str((destination / "scores.jsonl").resolve()): _sha256(
                destination / "scores.jsonl"
            ),
        }
        _write_canonical_json(destination / "run_manifest.json", manifest)
        return destination

    baseline = copied("baseline")
    baseline_validation = _run(
        VALIDATE_CLI,
        "--kind",
        "run-manifest",
        "--input",
        baseline / "run_manifest.json",
        "--manifest",
        baseline / "run_manifest.json",
    )
    assert baseline_validation.returncode == 0, baseline_validation.stdout

    score_mutations = {
        "family": lambda row: row.__setitem__("task_family", "forged_family"),
        "difficulty": lambda row: row.__setitem__("difficulty", "easy"),
        "status": lambda row: row.__setitem__("completion_status", "failed"),
    }
    for name, mutate in score_mutations.items():
        directory = copied(f"score-{name}")
        rows = _load_jsonl(directory / "scores.jsonl")
        mutate(rows[0])
        _write_canonical_jsonl(directory / "scores.jsonl", rows)
        manifest = _load_json(directory / "run_manifest.json")
        digest = _sha256(directory / "scores.jsonl")
        manifest["score_artifacts"][0]["sha256"] = digest
        manifest["native_vs_extracted_field_summary"]["output_hashes"][
            str((directory / "scores.jsonl").resolve())
        ] = digest
        _write_canonical_json(directory / "run_manifest.json", manifest)
        for kind, artifact in (
            ("scores", directory / "scores.jsonl"),
            ("task-runs", directory / "task_runs.jsonl"),
            ("run-manifest", directory / "run_manifest.json"),
        ):
            result = _run(
                VALIDATE_CLI,
                "--kind",
                kind,
                "--input",
                artifact,
                "--manifest",
                directory / "run_manifest.json",
            )
            assert result.returncode != 0, (name, kind)

    manifest_mutations = {
        "runtime_media": lambda manifest: manifest["normalized_runtime_artifacts"][0].__setitem__("media_type", "application/json"),
        "runtime_null_count": lambda manifest: manifest["normalized_runtime_artifacts"][0].__setitem__("record_count", None),
        "score_media": lambda manifest: manifest["score_artifacts"][0].__setitem__("media_type", "application/json"),
        "score_null_count": lambda manifest: manifest["score_artifacts"][0].__setitem__("record_count", None),
        "task_manifest_media": lambda manifest: manifest["task_manifest"].__setitem__("media_type", "text/plain"),
        "task_manifest_null_count": lambda manifest: manifest["task_manifest"].__setitem__("record_count", None),
        "empty_row_counts": lambda manifest: manifest["native_vs_extracted_field_summary"].__setitem__("row_counts", {}),
    }
    for name, mutate in manifest_mutations.items():
        directory = copied(f"manifest-{name}")
        manifest = _load_json(directory / "run_manifest.json")
        mutate(manifest)
        _write_canonical_json(directory / "run_manifest.json", manifest)
        result = _run(
            VALIDATE_CLI,
            "--kind",
            "run-manifest",
            "--input",
            directory / "run_manifest.json",
            "--manifest",
            directory / "run_manifest.json",
        )
        assert result.returncode != 0, name


def test_legacy_analysis_manifest_rejects_versions_empty_artifacts_and_bad_linkage() -> None:
    from pydantic import ValidationError
    from mub.vnext.legacy.artifacts import LegacyAnalysisManifest

    base = {
        "analysis_kind": "conflict",
        "compiler_version": "vnext-phase0-cli-1.0.0",
        "compatibility_only": True,
        "source_artifacts": [{"path": "source.csv", "sha256": "a" * 64, "media_type": "text/csv", "record_count": 1}],
        "output_artifacts": [{"path": "legacy_analysis.jsonl", "sha256": "b" * 64, "media_type": "application/x-ndjson", "record_count": 1}],
        "row_counts": {"legacy_analysis.jsonl": 1},
        "warnings": [],
        "caveats": [],
        "code_revision": "legacy-compatibility-import",
    }
    LegacyAnalysisManifest.model_validate(base)
    mutations = (
        {"schema_version": "9.9.9"},
        {"legacy_analysis_manifest_version": "9.9.9"},
        {"compiler_version": "forged"},
        {"compatibility_only": False},
        {"source_artifacts": []},
        {"output_artifacts": []},
        {"row_counts": {}},
        {"row_counts": {"legacy_analysis.jsonl": 2}},
        {"output_artifacts": [{"path": "legacy_analysis.jsonl", "sha256": "b" * 64, "media_type": "application/x-ndjson", "record_count": None}]},
    )
    for change in mutations:
        with pytest.raises(ValidationError):
            LegacyAnalysisManifest.model_validate({**base, **change})


def test_atomic_no_clobber_survives_destination_created_inside_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    destination = tmp_path / "out.json"
    real_link = os.link

    def racing_link(source: Path, target: Path, *args, **kwargs):
        Path(target).write_bytes(b"concurrent")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(atomic_module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        atomic_module.publish_files_atomically(
            {destination: b"new"}, overwrite=False
        )
    assert destination.read_bytes() == b"concurrent"
    assert not list(tmp_path.glob("*.tmp*"))


def test_atomic_reconciles_post_effect_backup_and_publish_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    destination = tmp_path / "out.json"
    destination.write_bytes(b"old")
    real_replace = atomic_module.os.replace
    raised = False

    def backup_then_interrupt(source: Path, target: Path):
        nonlocal raised
        real_replace(source, target)
        if ".bak." in Path(target).name and not raised:
            raised = True
            raise KeyboardInterrupt("after backup effect")

    monkeypatch.setattr(atomic_module.os, "replace", backup_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        atomic_module.publish_files_atomically(
            {destination: b"new"}, overwrite=True
        )
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))

    destination.unlink()
    raised = False

    def publish_then_interrupt(source: Path, target: Path):
        nonlocal raised
        real_replace(source, target)
        if ".tmp." in Path(source).name and not raised:
            raised = True
            raise KeyboardInterrupt("after publish effect")

    monkeypatch.setattr(atomic_module.os, "replace", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        atomic_module.publish_files_atomically(
            {destination: b"new"}, overwrite=True
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))


def test_legacy_analysis_manifest_schema_is_current_and_deterministic() -> None:
    from mub.vnext.legacy.artifacts import LegacyAnalysisManifest
    from mub.vnext.schema_export import DRAFT_2020_12_URI

    schema_path = PROJECT_ROOT / "schemas" / "legacy" / "legacy_analysis_manifest.schema.json"
    expected = LegacyAnalysisManifest.model_json_schema(mode="serialization")
    expected["$schema"] = DRAFT_2020_12_URI
    expected["title"] = "LegacyAnalysisManifest"
    expected_bytes = (
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert schema_path.read_bytes() == expected_bytes


def test_atomic_publication_rolls_back_after_partial_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mub.vnext.io.atomic as atomic_module

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    real_replace = atomic_module.os.replace

    def fail_second_publish(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == second and ".tmp." in source_path.name:
            raise OSError("injected second publication failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(atomic_module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected"):
        atomic_module.publish_files_atomically(
            {first: b"new-first", second: b"new-second"},
            overwrite=True,
        )

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert not list(tmp_path.glob("*.tmp*"))
    assert not list(tmp_path.glob("*.bak*"))
