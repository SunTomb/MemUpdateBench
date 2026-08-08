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
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.canonical import _canonical_payload_bytes
from mub.vnext.validation.core_release import validate_core_release


SELECTION_PACKAGE_NAME = "audit_selection.json"
SELECTED_TASKS_NAME = "selected_tasks.jsonl"
SELECTED_CORE_SURFACES_NAME = "selected_core_surfaces.jsonl"
DECISION_TEMPLATE_NAME = "decisions.template.jsonl"
ADJUDICATION_TEMPLATE_NAME = "adjudications.template.jsonl"
GATE_REPORT_NAME = "gate_report.json"
REQUIRED_ADJUDICATION_TEMPLATE_NAME = "required_adjudications.template.jsonl"


@dataclass(frozen=True)
class StagedCoreAuditPackage:
    output_dir: Path
    package: CoreAuditSelectionPackage
    selected_task_count: int
    review_surface_task_count: int
    decision_template_count: int
    adjudication_template_count: int


def _jsonl_bytes(models) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def core_audit_review_task_ids(
    package: CoreAuditSelectionPackage,
) -> tuple[str, ...]:
    """Return all four authenticated surface variants for every selected core."""
    if type(package) is not CoreAuditSelectionPackage:
        raise TypeError("package must be an exact CoreAuditSelectionPackage")
    identifiers = tuple(
        variant.task_id
        for selection in package.selections
        for variant in selection.surface_variants
    )
    if len(identifiers) != 896 or len(set(identifiers)) != 896:
        raise ValueError("Core audit review matrix must contain 896 unique surface tasks")
    return identifiers


def load_core_audit_selection_package(path: Path) -> CoreAuditSelectionPackage:
    raw = path.read_bytes()
    package = CoreAuditSelectionPackage.model_validate_json(raw)
    if canonical_json_bytes(package) != raw:
        raise ValueError("audit selection package must use canonical JSON bytes")
    return package


def _canonical_task_rows(raw: bytes, label: str) -> tuple[MemUpdateTaskV3, ...]:
    rows = []
    seen = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label}: line {line_number}: blank row")
        task = MemUpdateTaskV3.model_validate_json(line)
        if canonical_json_bytes(task) != line:
            raise ValueError(f"{label}: line {line_number}: noncanonical task row")
        if task.task_id in seen:
            raise ValueError(f"{label}: duplicate task ID {task.task_id}")
        seen.add(task.task_id)
        rows.append(task)
    return tuple(rows)


def _load_candidate(candidate_dir: Path, *, expected_full: bool):
    manifest_path = candidate_dir / "task_manifest.json"
    tasks_path = candidate_dir / "tasks.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    tasks_bytes = tasks_path.read_bytes()
    report = validate_core_release(candidate_dir, expected_full=expected_full)
    if not report.valid:
        raise ValueError("Core candidate validation did not return a valid report")
    if (
        manifest_path.read_bytes() != manifest_bytes
        or tasks_path.read_bytes() != tasks_bytes
    ):
        raise ValueError("Core candidate manifest/tasks changed during validation")
    manifest = TaskManifestV3.model_validate_json(manifest_bytes)
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("task manifest changed or is not canonical after validation")
    task_ref = next(
        (
            ref
            for ref in manifest.task_file_paths_and_hashes
            if ref.path == "tasks.jsonl"
        ),
        None,
    )
    if task_ref is None or hashlib.sha256(tasks_bytes).hexdigest() != task_ref.sha256:
        raise ValueError("tasks bytes changed or do not match the authenticated manifest")
    tasks = _canonical_task_rows(tasks_bytes, "tasks.jsonl")
    if len(tasks) != task_ref.record_count:
        raise ValueError("tasks record count does not match the authenticated manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
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
    review_surface_tasks = tuple(
        by_id[task_id] for task_id in core_audit_review_task_ids(package)
    )
    decision_templates = core_audit_decision_templates(package)
    adjudication_templates = core_audit_adjudication_templates(
        package, (item.audit_id for item in package.selections)
    )
    payloads = {
        output_dir / SELECTION_PACKAGE_NAME: canonical_json_bytes(package),
        output_dir / SELECTED_TASKS_NAME: _jsonl_bytes(selected_tasks),
        output_dir / SELECTED_CORE_SURFACES_NAME: _jsonl_bytes(
            review_surface_tasks
        ),
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
        review_surface_task_count=len(review_surface_tasks),
        decision_template_count=len(decision_templates),
        adjudication_template_count=len(adjudication_templates),
    )


def _read_review_rows(
    path: Path, *, required: bool = True
) -> tuple[Any, ...]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required human review file does not exist: {path}")
        return ()
    rows = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}: line {line_number}: blank row")
            try:
                text = raw.decode("utf-8", errors="strict")
                payload = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path}: line {line_number}: malformed strict JSON"
                ) from exc
            if _canonical_payload_bytes(payload) != raw.rstrip(b"\n"):
                raise ValueError(
                    f"{path}: line {line_number}: review row must use canonical JSON bytes without duplicate keys"
                )
            try:
                rows.append(CoreAuditDecision.model_validate(payload))
            except Exception:
                rows.append(payload)
    return tuple(rows)


def _adjudication_templates_for_report(
    package: CoreAuditSelectionPackage,
    report: CoreAuditGateReport,
):
    """Return blanks only for adjudications that remain unresolved."""
    return core_audit_adjudication_templates(
        package, report.unresolved_adjudication_ids
    )


def gate_core_audit_files(
    *,
    selection_package_path: Path,
    candidate_dir: Path,
    selected_tasks_path: Path,
    surface_context_path: Path,
    decisions_path: Path,
    adjudications_path: Path | None,
    output_dir: Path,
    overwrite: bool = False,
    expected_full: bool = True,
) -> CoreAuditGateReport:
    """Evaluate human files and atomically write a gate report plus required blanks."""
    selection_package_path = Path(selection_package_path)
    candidate_dir = Path(candidate_dir)
    selected_tasks_path = Path(selected_tasks_path)
    surface_context_path = Path(surface_context_path)
    decisions_path = Path(decisions_path)
    adjudications_path = (
        None if adjudications_path is None else Path(adjudications_path)
    )
    selection_bytes = selection_package_path.read_bytes()
    package = load_core_audit_selection_package(selection_package_path)
    if selection_package_path.read_bytes() != selection_bytes:
        raise ValueError("selection package changed while it was being loaded")
    candidate_tasks, manifest, manifest_hash = _load_candidate(
        candidate_dir, expected_full=expected_full
    )
    if manifest_hash != package.source_task_manifest_hash:
        raise ValueError("trusted candidate task manifest hash does not match the selection")
    recomputed = select_core_audit_sample(
        candidate_tasks,
        manifest,
        source_task_manifest_hash=manifest_hash,
    )
    if recomputed != package:
        raise ValueError(
            "staged selection does not equal deterministic selection from the trusted candidate"
        )
    selected_bytes = selected_tasks_path.read_bytes()
    context_bytes = surface_context_path.read_bytes()
    selected_tasks = _canonical_task_rows(selected_bytes, "selected_tasks.jsonl")
    surface_tasks = _canonical_task_rows(
        context_bytes, "selected_core_surfaces.jsonl"
    )
    for task in (*selected_tasks, *surface_tasks):
        declared = manifest.task_record_hashes.get(task.task_id)
        if declared is None or sha256_model(task) != declared:
            raise ValueError(
                f"reviewed task {task.task_id} is not authenticated by the source manifest"
            )
    decision_bytes = decisions_path.read_bytes()
    decisions = _read_review_rows(decisions_path)
    if decisions_path.read_bytes() != decision_bytes:
        raise ValueError("decisions file changed while it was being loaded")
    if adjudications_path is None:
        adjudication_bytes = None
        adjudications = ()
    else:
        adjudication_bytes = adjudications_path.read_bytes()
        adjudications = _read_review_rows(adjudications_path)
        if adjudications_path.read_bytes() != adjudication_bytes:
            raise ValueError("adjudications file changed while it was being loaded")
    report = evaluate_core_audit_gate(
        package,
        decisions,
        adjudications,
        source_task_manifest=manifest,
        selected_tasks=selected_tasks,
        surface_context_tasks=surface_tasks,
    )
    required_templates = _adjudication_templates_for_report(package, report)
    output_dir = Path(output_dir)
    sources = [
        selection_package_path,
        candidate_dir / "task_manifest.json",
        candidate_dir / "tasks.jsonl",
        selected_tasks_path,
        surface_context_path,
        decisions_path,
    ]
    expected_source_bytes = {
        selection_package_path: selection_bytes,
        candidate_dir / "task_manifest.json": canonical_json_bytes(manifest),
        candidate_dir / "tasks.jsonl": _jsonl_bytes(candidate_tasks),
        selected_tasks_path: selected_bytes,
        surface_context_path: context_bytes,
        decisions_path: decision_bytes,
    }
    if adjudications_path is not None:
        sources.append(adjudications_path)
        assert adjudication_bytes is not None
        expected_source_bytes[adjudications_path] = adjudication_bytes

    def recheck_sources() -> None:
        if any(
            path.read_bytes() != expected
            for path, expected in expected_source_bytes.items()
        ):
            raise ValueError("audit source bytes changed before report publication")

    publish_files_atomically(
        {
            output_dir / GATE_REPORT_NAME: canonical_json_bytes(report),
            output_dir
            / REQUIRED_ADJUDICATION_TEMPLATE_NAME: _jsonl_bytes(required_templates),
        },
        overwrite=overwrite,
        source_paths=tuple(sources),
        pre_publish=recheck_sources,
    )
    return report


__all__ = [
    "ADJUDICATION_TEMPLATE_NAME",
    "DECISION_TEMPLATE_NAME",
    "GATE_REPORT_NAME",
    "REQUIRED_ADJUDICATION_TEMPLATE_NAME",
    "SELECTED_CORE_SURFACES_NAME",
    "SELECTED_TASKS_NAME",
    "SELECTION_PACKAGE_NAME",
    "StagedCoreAuditPackage",
    "core_audit_review_task_ids",
    "gate_core_audit_files",
    "load_core_audit_selection_package",
    "stage_core_audit_package",
]
