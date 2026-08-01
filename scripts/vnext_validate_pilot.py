from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel

from mub.vnext.audit import (
    AuditDecision,
    AuditGateReport,
    AuditSelection,
    audit_decision_template,
    evaluate_audit_gate,
    select_pilot_audit_sample,
)
from mub.vnext.contracts import MemUpdateTask, TaskManifest
from mub.vnext.io import canonical_json_bytes
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.jsonl import read_models
from mub.vnext.validation import (
    ValidationIssue,
    ValidationReport,
    build_report,
    validate_pilot_release,
)


EXIT_RELEASE_READY = 0
EXIT_AUTOMATED_INVALID = 1
EXIT_USAGE_OR_IO_ERROR = 2
EXIT_AUDIT_NOT_READY = 3

AUTOMATED_REPORT_NAME = "automated_validation_report.json"
AUDIT_SELECTIONS_NAME = "audit_selections.jsonl"
AUDIT_TEMPLATE_NAME = "audit_decision_template.jsonl"
AUDIT_GATE_REPORT_NAME = "audit_gate_report.json"
_OUTPUT_NAMES = (
    AUTOMATED_REPORT_NAME,
    AUDIT_SELECTIONS_NAME,
    AUDIT_TEMPLATE_NAME,
    AUDIT_GATE_REPORT_NAME,
)


class _ArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError from None


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Validate and audit a canonical MemUpdateBench vNext Pilot release.",
        epilog=(
            "Exit codes: 0 release ready; 1 automated validation invalid; "
            "2 invalid arguments/input-output failure; 3 automated validation "
            "passed but human audit is not release ready."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--audit-decisions", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _canonical_json_line(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _write_error(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")


def _require_regular_file(path: Path) -> Path:
    result = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("input must be a regular file")
    return path.resolve(strict=True)


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        if left.exists() and right.exists():
            return os.path.samefile(left, right)
    except OSError:
        pass
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except (OSError, RuntimeError):
        return os.path.abspath(left) == os.path.abspath(right)


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, tuple[Path, ...], tuple[Path, ...]]:
    tasks_path = _require_regular_file(args.tasks)
    manifest_path = _require_regular_file(args.task_manifest)
    decisions_path = (
        _require_regular_file(args.audit_decisions)
        if args.audit_decisions is not None
        else None
    )
    source_paths = tuple(
        path
        for path in (tasks_path, manifest_path, decisions_path)
        if path is not None
    )
    for index, left in enumerate(source_paths):
        for right in source_paths[index + 1 :]:
            if _paths_alias(left, right):
                raise ValueError("input artifacts must identify distinct files")

    output_dir = Path(os.path.abspath(args.output_dir))
    destinations = tuple(output_dir / name for name in _OUTPUT_NAMES)
    for source in source_paths:
        if any(_paths_alias(source, destination) for destination in destinations):
            raise ValueError("outputs may not alias inputs")
    for index, left in enumerate(destinations):
        for right in destinations[index + 1 :]:
            if _paths_alias(left, right):
                raise ValueError("outputs may not alias each other")
    return tasks_path, manifest_path, decisions_path, destinations, source_paths


def _jsonl_bytes(models: Iterable[BaseModel]) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def _load_canonical_tasks(path: Path) -> tuple[MemUpdateTask, ...]:
    raw = path.read_bytes()
    tasks = tuple(read_models(path, MemUpdateTask, id_field="task_id"))
    if raw != _jsonl_bytes(tasks):
        raise ValueError("tasks artifact is not canonical JSONL")
    return tasks


def _load_canonical_manifest(path: Path) -> TaskManifest:
    raw = path.read_bytes()
    manifest = TaskManifest.model_validate_json(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("task manifest is not canonical JSON")
    return manifest


def _load_canonical_decisions(path: Path) -> tuple[AuditDecision, ...]:
    raw = path.read_bytes()
    if not raw:
        return ()
    lines = raw.splitlines(keepends=True)
    decisions: list[AuditDecision] = []
    for line in lines:
        if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise ValueError("audit decisions must be canonical JSONL")
        record = line[:-1]
        decision = AuditDecision.model_validate_json(record)
        if record != canonical_json_bytes(decision):
            raise ValueError("audit decisions artifact is not canonical JSONL")
        if len(decisions) == 4096:
            raise ValueError("audit decisions exceed the bounded audit limit")
        decisions.append(decision)
    return tuple(decisions)


def _input_error_report() -> ValidationReport:
    return build_report(
        (
            ValidationIssue(
                code="pilot_cli_input_invalid",
                message="canonical Pilot input could not be loaded",
                path="inputs",
                severity="error",
            ),
        )
    )


def _output_payloads(
    destinations: tuple[Path, ...],
    report: ValidationReport,
    selections: Sequence[AuditSelection],
    gate: AuditGateReport,
) -> dict[Path, bytes]:
    templates = tuple(
        audit_decision_template(selection.audit_id) for selection in selections
    )
    return {
        destinations[0]: canonical_json_bytes(report),
        destinations[1]: _jsonl_bytes(selections),
        destinations[2]: _jsonl_bytes(templates),
        destinations[3]: canonical_json_bytes(gate),
    }


def _publish_outputs(
    *,
    destinations: tuple[Path, ...],
    report: ValidationReport,
    selections: Sequence[AuditSelection],
    gate: AuditGateReport,
    source_paths: tuple[Path, ...],
) -> None:
    publish_files_atomically(
        _output_payloads(destinations, report, selections, gate),
        overwrite=True,
        source_paths=source_paths,
    )


def _print_summary(
    *,
    automated_valid: bool,
    decision_status: str,
    release_ready: bool,
    selection_count: int,
) -> None:
    sys.stdout.write(
        _canonical_json_line(
            {
                "automated_valid": automated_valid,
                "decision_status": decision_status,
                "release_ready": release_ready,
                "selection_count": selection_count,
            }
        )
    )


def _finish_automated_invalid(
    *,
    destinations: tuple[Path, ...],
    report: ValidationReport,
    source_paths: tuple[Path, ...],
) -> int:
    try:
        _publish_outputs(
            destinations=destinations,
            report=report,
            selections=(),
            gate=evaluate_audit_gate([], []),
            source_paths=source_paths,
        )
    except Exception:
        _write_error("could not publish validation outputs")
        return EXIT_USAGE_OR_IO_ERROR
    _print_summary(
        automated_valid=False,
        decision_status="not_evaluated",
        release_ready=False,
        selection_count=0,
    )
    return EXIT_AUTOMATED_INVALID


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except _ArgumentError:
        _write_error("invalid command-line arguments")
        return EXIT_USAGE_OR_IO_ERROR

    try:
        (
            tasks_path,
            manifest_path,
            decisions_path,
            destinations,
            source_paths,
        ) = _resolve_paths(args)
    except Exception:
        _write_error("invalid input or output paths")
        return EXIT_USAGE_OR_IO_ERROR

    try:
        tasks = _load_canonical_tasks(tasks_path)
        manifest = _load_canonical_manifest(manifest_path)
    except ValueError:
        return _finish_automated_invalid(
            destinations=destinations,
            report=_input_error_report(),
            source_paths=source_paths,
        )
    except Exception:
        _write_error("could not read canonical Pilot inputs")
        return EXIT_USAGE_OR_IO_ERROR

    try:
        report = validate_pilot_release(tasks, manifest)
    except Exception:
        _write_error("automated validation failed")
        return EXIT_USAGE_OR_IO_ERROR
    if not report.valid:
        return _finish_automated_invalid(
            destinations=destinations,
            report=report,
            source_paths=source_paths,
        )

    try:
        selection_result = select_pilot_audit_sample(tasks, manifest)
        if not selection_result.valid or len(selection_result.selections) != 96:
            raise ValueError("audit selection did not produce the exact Pilot sample")
        selections = selection_result.selections
    except Exception:
        _write_error("audit selection failed")
        return EXIT_USAGE_OR_IO_ERROR

    decision_status = "not_supplied"
    decisions: tuple[AuditDecision, ...] = ()
    if decisions_path is not None:
        try:
            decisions = _load_canonical_decisions(decisions_path)
            decision_status = "evaluated"
        except ValueError:
            decision_status = "malformed"
            decisions = ()
        except Exception:
            _write_error("could not read canonical audit decisions")
            return EXIT_USAGE_OR_IO_ERROR
    gate = evaluate_audit_gate(selections, decisions)
    if decision_status == "malformed":
        gate = gate.validated_replace(malformed_decision_ids=("<artifact>",))

    try:
        _publish_outputs(
            destinations=destinations,
            report=report,
            selections=selections,
            gate=gate,
            source_paths=source_paths,
        )
    except Exception:
        _write_error("could not publish validation outputs")
        return EXIT_USAGE_OR_IO_ERROR

    _print_summary(
        automated_valid=True,
        decision_status=decision_status,
        release_ready=gate.release_ready,
        selection_count=len(selections),
    )
    return EXIT_RELEASE_READY if gate.release_ready else EXIT_AUDIT_NOT_READY


if __name__ == "__main__":
    raise SystemExit(main())
