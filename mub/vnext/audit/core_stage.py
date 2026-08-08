"""Transactional staging and loading for the strict-v3 Core audit package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mub.vnext.audit.core import CoreAuditSelectionPackage, select_core_audit_sample
from mub.vnext.audit.core_review import (
    CoreAuditDecision,
    CoreAuditGateReport,
    core_audit_adjudication_templates,
    core_audit_decision_templates,
    evaluate_core_audit_gate,
)
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.io import canonical_json_bytes, read_models
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.validation.core_release import validate_core_release


SELECTION_PACKAGE_NAME = "audit_selection.json"
SELECTED_TASKS_NAME = "selected_tasks.jsonl"
DECISION_TEMPLATE_NAME = "decisions.template.jsonl"
ADJUDICATION_TEMPLATE_NAME = "adjudications.template.jsonl"
GATE_REPORT_NAME = "gate_report.json"
REQUIRED_ADJUDICATION_TEMPLATE_NAME = "required_adjudications.template.jsonl"


@dataclass(frozen=True)
class StagedCoreAuditPackage:
    output_dir: Path
    package: CoreAuditSelectionPackage
    selected_task_count: int
    decision_template_count: int
    adjudication_template_count: int


def _jsonl_bytes(models) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def load_core_audit_selection_package(path: Path) -> CoreAuditSelectionPackage:
    return CoreAuditSelectionPackage.model_validate_json(path.read_bytes())


def _load_candidate(candidate_dir: Path, *, expected_full: bool):
    report = validate_core_release(candidate_dir, expected_full=expected_full)
    if not report.valid:
        raise ValueError("Core candidate validation did not return a valid report")
    manifest_path = candidate_dir / "task_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = TaskManifestV3.model_validate_json(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    tasks = tuple(
        read_models(candidate_dir / "tasks.jsonl", MemUpdateTaskV3, id_field="task_id")
    )
    return tasks, manifest, manifest_hash


def stage_core_audit_package(
    *,
    candidate_dir: Path,
    output_dir: Path,
    expected_full: bool = True,
) -> StagedCoreAuditPackage:
    """Validate a candidate and atomically stage blank, non-release audit files."""
    candidate_dir = Path(candidate_dir)
    output_dir = Path(output_dir)
    tasks, manifest, manifest_hash = _load_candidate(
        candidate_dir, expected_full=expected_full
    )
    package = select_core_audit_sample(
        tasks,
        manifest,
        source_task_manifest_hash=manifest_hash,
    )
    by_id = {task.task_id: task for task in tasks}
    selected_tasks = tuple(by_id[item.task_id] for item in package.selections)
    decision_templates = core_audit_decision_templates(package)
    adjudication_templates = core_audit_adjudication_templates(
        package, (item.audit_id for item in package.selections)
    )
    payloads = {
        output_dir / SELECTION_PACKAGE_NAME: canonical_json_bytes(package),
        output_dir / SELECTED_TASKS_NAME: _jsonl_bytes(selected_tasks),
        output_dir / DECISION_TEMPLATE_NAME: _jsonl_bytes(decision_templates),
        output_dir / ADJUDICATION_TEMPLATE_NAME: _jsonl_bytes(
            adjudication_templates
        ),
    }
    publish_files_atomically(
        payloads,
        overwrite=False,
        source_paths=(candidate_dir / "task_manifest.json", candidate_dir / "tasks.jsonl"),
    )
    return StagedCoreAuditPackage(
        output_dir=output_dir,
        package=package,
        selected_task_count=len(selected_tasks),
        decision_template_count=len(decision_templates),
        adjudication_template_count=len(adjudication_templates),
    )


def _read_review_rows(path: Path) -> tuple[Any, ...]:
    if not path.exists():
        return ()
    rows = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}: line {line_number}: blank row")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path}: line {line_number}: malformed strict JSON"
                ) from exc
            try:
                rows.append(CoreAuditDecision.model_validate(payload))
            except Exception:
                rows.append(payload)
    return tuple(rows)


def gate_core_audit_files(
    *,
    selection_package_path: Path,
    decisions_path: Path,
    adjudications_path: Path | None,
    output_dir: Path,
    overwrite: bool = False,
) -> CoreAuditGateReport:
    """Evaluate human files and atomically write a gate report plus required blanks."""
    package = load_core_audit_selection_package(Path(selection_package_path))
    decisions = _read_review_rows(Path(decisions_path))
    adjudications = (
        ()
        if adjudications_path is None
        else _read_review_rows(Path(adjudications_path))
    )
    report = evaluate_core_audit_gate(package, decisions, adjudications)
    required_templates = core_audit_adjudication_templates(
        package, report.required_adjudication_ids
    )
    output_dir = Path(output_dir)
    sources = [Path(selection_package_path), Path(decisions_path)]
    if adjudications_path is not None and Path(adjudications_path).exists():
        sources.append(Path(adjudications_path))
    publish_files_atomically(
        {
            output_dir / GATE_REPORT_NAME: canonical_json_bytes(report),
            output_dir
            / REQUIRED_ADJUDICATION_TEMPLATE_NAME: _jsonl_bytes(required_templates),
        },
        overwrite=overwrite,
        source_paths=tuple(sources),
    )
    return report


__all__ = [
    "ADJUDICATION_TEMPLATE_NAME",
    "DECISION_TEMPLATE_NAME",
    "GATE_REPORT_NAME",
    "REQUIRED_ADJUDICATION_TEMPLATE_NAME",
    "SELECTED_TASKS_NAME",
    "SELECTION_PACKAGE_NAME",
    "StagedCoreAuditPackage",
    "gate_core_audit_files",
    "load_core_audit_selection_package",
    "stage_core_audit_package",
]
