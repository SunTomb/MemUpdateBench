from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mub.vnext.audit import (
    AuditDecision,
    AuditDecisionTemplate,
    AuditGateReport,
    AuditSelection,
    select_pilot_audit_sample,
)
from mub.vnext.generation import (
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes
from mub.vnext.io.jsonl import read_models
from mub.vnext.validation import ValidationReport, validate_pilot_release
from scripts import vnext_validate_pilot as cli


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "vnext_validate_pilot.py"
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "pilot-validation-cli-test-revision"
OUTPUT_NAMES = (
    "automated_validation_report.json",
    "audit_selections.jsonl",
    "audit_decision_template.jsonl",
    "audit_gate_report.json",
)


@pytest.fixture(scope="session")
def canonical_release():
    config = load_pilot_config(CONFIG_PATH)
    compiled = compile_pilot_tasks(config, code_revision=REVISION)
    bundle = build_pilot_artifact_bundle(compiled, config)
    report = validate_pilot_release(compiled.tasks, bundle.task_manifest)
    selection = select_pilot_audit_sample(compiled.tasks, bundle.task_manifest)
    assert report.valid
    assert selection.valid
    return bundle, report, selection


def _write_release(tmp_path: Path, canonical_release) -> tuple[Path, Path]:
    bundle, _, _ = canonical_release
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "task_manifest.json"
    tasks_path.write_bytes(bundle.tasks_jsonl)
    manifest_path.write_bytes(bundle.task_manifest_bytes)
    return tasks_path, manifest_path


def _args(
    tasks_path: Path,
    manifest_path: Path,
    output_dir: Path,
    decisions_path: Path | None = None,
) -> list[str]:
    args = [
        "--tasks",
        str(tasks_path),
        "--task-manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    if decisions_path is not None:
        args.extend(("--audit-decisions", str(decisions_path)))
    return args


def _decision(selection: AuditSelection, **updates) -> AuditDecision:
    payload = {
        "audit_id": selection.audit_id,
        "reviewer": "human-reviewer",
        "verdict": "pass",
        "answer_unique": True,
        "actions_correct": True,
        "roles_correct": True,
        "surface_natural": True,
        "notes": "human-reviewed test evidence",
    }
    payload.update(updates)
    return AuditDecision.model_validate(payload)


def _write_decisions(path: Path, decisions: list[AuditDecision]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(decision) + b"\n" for decision in decisions)
    )


def _load_outputs(output_dir: Path):
    report = ValidationReport.model_validate_json(
        (output_dir / OUTPUT_NAMES[0]).read_bytes()
    )
    selections = list(
        read_models(
            output_dir / OUTPUT_NAMES[1], AuditSelection, id_field="audit_id"
        )
    )
    templates = list(
        read_models(
            output_dir / OUTPUT_NAMES[2], AuditDecisionTemplate, id_field="audit_id"
        )
    )
    gate = AuditGateReport.model_validate_json(
        (output_dir / OUTPUT_NAMES[3]).read_bytes()
    )
    return report, selections, templates, gate


def test_cli_help_and_import() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    for option in (
        "--tasks",
        "--task-manifest",
        "--audit-decisions",
        "--output-dir",
    ):
        assert option in completed.stdout

    spec = importlib.util.spec_from_file_location("vnext_validate_pilot", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert module.EXIT_RELEASE_READY == 0
    assert module.EXIT_AUTOMATED_INVALID == 1
    assert module.EXIT_USAGE_OR_IO_ERROR == 2
    assert module.EXIT_AUDIT_NOT_READY == 3


def test_canonical_release_without_decisions_is_valid_non_ready_and_stable(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    output_dir = tmp_path / "validation"

    first_status = cli.main(_args(tasks_path, manifest_path, output_dir))
    first_capture = capsys.readouterr()
    first_bytes = {
        name: (output_dir / name).read_bytes() for name in OUTPUT_NAMES
    }
    report, selections, templates, gate = _load_outputs(output_dir)

    assert first_status == cli.EXIT_AUDIT_NOT_READY
    assert first_capture.err == ""
    assert json.loads(first_capture.out) == {
        "automated_valid": True,
        "decision_status": "not_supplied",
        "release_ready": False,
        "selection_count": 96,
    }
    assert report.valid
    assert len(selections) == 96
    assert len(templates) == 96
    assert [item.audit_id for item in templates] == [
        item.audit_id for item in selections
    ]
    assert all(
        item.reviewer is None
        and item.verdict is None
        and item.answer_unique is None
        and item.actions_correct is None
        and item.roles_correct is None
        and item.surface_natural is None
        and item.notes is None
        for item in templates
    )
    assert gate.release_ready is False
    assert gate.missing_audit_ids == tuple(
        sorted(item.audit_id for item in selections)
    )

    second_status = cli.main(_args(tasks_path, manifest_path, output_dir))
    second_capture = capsys.readouterr()
    second_bytes = {
        name: (output_dir / name).read_bytes() for name in OUTPUT_NAMES
    }

    assert second_status == first_status
    assert second_capture == first_capture
    assert second_bytes == first_bytes
    assert set(path.name for path in output_dir.iterdir()) == set(OUTPUT_NAMES)


def test_all_pass_human_decisions_make_release_ready(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    _, release_report, selection = canonical_release
    decisions_path = tmp_path / "audit_decisions.jsonl"
    _write_decisions(
        decisions_path,
        [_decision(item) for item in selection.selections],
    )
    monkeypatch.setattr(cli, "validate_pilot_release", lambda tasks, manifest: release_report)
    monkeypatch.setattr(cli, "select_pilot_audit_sample", lambda tasks, manifest: selection)

    status = cli.main(
        _args(tasks_path, manifest_path, tmp_path / "validation", decisions_path)
    )
    captured = capsys.readouterr()
    report, selections, templates, gate = _load_outputs(tmp_path / "validation")

    assert status == cli.EXIT_RELEASE_READY
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "automated_valid": True,
        "decision_status": "evaluated",
        "release_ready": True,
        "selection_count": 96,
    }
    assert report.valid
    assert len(selections) == len(templates) == 96
    assert gate.release_ready is True
    assert gate.passed_audit_ids == gate.selected_audit_ids
    assert len(gate.decision_evidence) == 96


@pytest.mark.parametrize(
    ("case", "expected_field"),
    (
        ("block", "non_pass_audit_ids"),
        ("needs_revision", "non_pass_audit_ids"),
        ("failed_check", "failed_check_audit_ids"),
        ("missing", "missing_audit_ids"),
        ("duplicate", "duplicate_audit_ids"),
        ("foreign", "foreign_audit_ids"),
    ),
)
def test_non_pass_missing_duplicate_and_foreign_decisions_remain_non_ready(
    case: str,
    expected_field: str,
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    _, release_report, selection = canonical_release
    decisions = [_decision(item) for item in selection.selections]
    if case == "block":
        decisions[0] = _decision(selection.selections[0], verdict="block")
    elif case == "needs_revision":
        decisions[0] = _decision(selection.selections[0], verdict="needs_revision")
    elif case == "failed_check":
        decisions[0] = _decision(selection.selections[0], roles_correct=False)
    elif case == "missing":
        decisions.pop()
    elif case == "duplicate":
        decisions.append(decisions[0])
    elif case == "foreign":
        payload = decisions[0].model_dump(mode="python")
        payload["audit_id"] = "audit-foreign"
        decisions.append(AuditDecision.model_validate(payload))
    decisions_path = tmp_path / "audit_decisions.jsonl"
    _write_decisions(decisions_path, decisions)
    monkeypatch.setattr(cli, "validate_pilot_release", lambda tasks, manifest: release_report)
    monkeypatch.setattr(cli, "select_pilot_audit_sample", lambda tasks, manifest: selection)

    status = cli.main(
        _args(tasks_path, manifest_path, tmp_path / "validation", decisions_path)
    )
    captured = capsys.readouterr()
    report, selections, templates, gate = _load_outputs(tmp_path / "validation")

    assert status == cli.EXIT_AUDIT_NOT_READY
    assert captured.err == ""
    assert json.loads(captured.out)["automated_valid"] is True
    assert json.loads(captured.out)["release_ready"] is False
    assert report.valid
    assert len(selections) == len(templates) == 96
    assert gate.release_ready is False
    assert getattr(gate, expected_field)


def test_malformed_decision_file_is_rejected_without_false_readiness(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    _, release_report, selection = canonical_release
    decisions_path = tmp_path / "audit_decisions.jsonl"
    decisions_path.write_bytes(
        b'{"audit_id":"audit-malformed","reviewer":"PRIVATE-REVIEWER"}\n'
    )
    monkeypatch.setattr(cli, "validate_pilot_release", lambda tasks, manifest: release_report)
    monkeypatch.setattr(cli, "select_pilot_audit_sample", lambda tasks, manifest: selection)

    status = cli.main(
        _args(tasks_path, manifest_path, tmp_path / "validation", decisions_path)
    )
    captured = capsys.readouterr()
    report, selections, templates, gate = _load_outputs(tmp_path / "validation")

    assert status == cli.EXIT_AUDIT_NOT_READY
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "automated_valid": True,
        "decision_status": "malformed",
        "release_ready": False,
        "selection_count": 96,
    }
    assert report.valid
    assert len(selections) == len(templates) == 96
    assert gate.release_ready is False
    assert gate.malformed_decision_ids == ("<artifact>",)
    assert len(gate.missing_audit_ids) == 96
    assert b"PRIVATE-REVIEWER" not in b"".join(
        (tmp_path / "validation" / name).read_bytes() for name in OUTPUT_NAMES
    )
    assert "PRIVATE-REVIEWER" not in captured.out + captured.err


def test_invalid_manifest_writes_atomic_non_ready_output_set(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    bundle, _, _ = canonical_release
    invalid_manifest = bundle.task_manifest.validated_replace(
        code_revision="different-revision"
    )
    manifest_path.write_bytes(canonical_json_bytes(invalid_manifest))
    output_dir = tmp_path / "validation"

    status = cli.main(_args(tasks_path, manifest_path, output_dir))
    captured = capsys.readouterr()
    report, selections, templates, gate = _load_outputs(output_dir)

    assert status == cli.EXIT_AUTOMATED_INVALID
    assert captured.err == ""
    assert json.loads(captured.out)["automated_valid"] is False
    assert json.loads(captured.out)["release_ready"] is False
    assert not report.valid
    assert selections == []
    assert templates == []
    assert gate.release_ready is False
    assert set(path.name for path in output_dir.iterdir()) == set(OUTPUT_NAMES)


def test_malformed_private_task_payload_is_not_accepted_or_leaked(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path, manifest_path = _write_release(tmp_path, canonical_release)
    secret = "PRIVATE-TASK-PAYLOAD"
    tasks_path.write_text(secret + "\n", encoding="utf-8")
    output_dir = tmp_path / "validation"

    status = cli.main(_args(tasks_path, manifest_path, output_dir))
    captured = capsys.readouterr()
    report, selections, templates, gate = _load_outputs(output_dir)

    assert status == cli.EXIT_AUTOMATED_INVALID
    assert not report.valid
    assert selections == []
    assert templates == []
    assert gate.release_ready is False
    assert secret not in captured.out + captured.err
    assert secret.encode() not in b"".join(
        (output_dir / name).read_bytes() for name in OUTPUT_NAMES
    )


def test_output_paths_cannot_alias_inputs_or_each_other(
    canonical_release,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, _, _ = canonical_release
    output_dir = tmp_path / "validation"
    output_dir.mkdir()
    tasks_path = output_dir / OUTPUT_NAMES[0]
    manifest_path = tmp_path / "task_manifest.json"
    tasks_path.write_bytes(bundle.tasks_jsonl)
    manifest_path.write_bytes(bundle.task_manifest_bytes)
    original_tasks = tasks_path.read_bytes()

    status = cli.main(_args(tasks_path, manifest_path, output_dir))
    captured = capsys.readouterr()

    assert status == cli.EXIT_USAGE_OR_IO_ERROR
    assert tasks_path.read_bytes() == original_tasks
    assert "Traceback" not in captured.out + captured.err
    assert not any((output_dir / name).exists() for name in OUTPUT_NAMES[1:])

    alias_dir = tmp_path / "alias-validation"
    alias_dir.mkdir()
    first = alias_dir / OUTPUT_NAMES[0]
    second = alias_dir / OUTPUT_NAMES[1]
    first.write_bytes(b"existing-output")
    os.link(first, second)
    status = cli.main(_args(tasks_path, manifest_path, alias_dir))
    capsys.readouterr()
    assert status == cli.EXIT_USAGE_OR_IO_ERROR
    assert first.read_bytes() == second.read_bytes() == b"existing-output"
