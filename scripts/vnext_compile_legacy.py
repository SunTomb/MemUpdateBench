from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Iterable

from pydantic import BaseModel

from mub.vnext.contracts import MemUpdateTask, RunManifest, ScoreRecord, TaskManifest, TaskRunRecord
from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.enums import Split
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.io.jsonl import read_models
from mub.vnext.legacy.artifacts import (
    LEGACY_CLI_CODE_REVISION,
    LEGACY_CLI_COMPILER_VERSION,
    LegacyAnalysisManifest,
    authenticate_legacy_task_manifest,
    build_expected_legacy_task_manifest,
    with_legacy_index_profile,
)
from mub.vnext.legacy.dataset import compile_legacy_episode
from mub.vnext.legacy.ledger import audit_ledger_references
from mub.vnext.legacy.loaders import load_evomemory_dataset, load_evomemory_results
from mub.vnext.legacy.mechanisms import (
    import_api_probe,
    import_conflict_probe,
    import_stale_removal_trace,
    import_synthetic_dose,
)
from mub.vnext.legacy.results import (
    authenticate_legacy_result_selection,
    import_evomemory_results,
)
from mub.vnext.validation import validate_splits, validate_task_semantics
from mub.vnext.version import (
    SCHEMA_VERSION,
    TASK_MANIFEST_VERSION,
)


CLI_COMPILER_VERSION = LEGACY_CLI_COMPILER_VERSION
_CODE_REVISION = LEGACY_CLI_CODE_REVISION
_JSON_MEDIA_TYPE = "application/json"
_JSONL_MEDIA_TYPE = "application/x-ndjson"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_unchanged(expected: dict[Path, str], label: str) -> None:
    observed = {path: _sha256_file(path) for path in expected}
    if observed != expected:
        raise RuntimeError(f"an input artifact changed during {label}")


def _require_regular_file(path: Path, label: str) -> Path:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _canonical_jsonl(models: Iterable[BaseModel]) -> bytes:
    payload = bytearray()
    for model in models:
        payload.extend(canonical_json_bytes(model))
        payload.extend(b"\n")
    return bytes(payload)


def _strict_json_load(path: Path) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value!r}")
        return parsed

    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON artifact {path}: {exc}") from exc


def _load_canonical_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    payload = _strict_json_load(path)
    model = model_type.model_validate(payload)
    if path.read_bytes() != canonical_json_bytes(model):
        raise ValueError(f"artifact is not canonical JSON: {path}")
    return model


def _load_canonical_jsonl(
    path: Path, model_type: type[BaseModel], id_field: str
) -> list[BaseModel]:
    models = list(read_models(path, model_type, id_field=id_field))
    if path.read_bytes() != _canonical_jsonl(models):
        raise ValueError(f"artifact is not canonical JSONL: {path}")
    return models


def _load_unkeyed_canonical_jsonl(
    path: Path, model_type: type[BaseModel]
) -> list[BaseModel]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"artifact is not canonical JSONL: {path}")
    models: list[BaseModel] = []
    seen: set[bytes] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL row at line {line_number}")
        try:
            payload = json.loads(
                line.decode("utf-8", errors="strict"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
            model = model_type.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"invalid JSONL row at line {line_number}") from exc
        canonical = canonical_json_bytes(model)
        if line != canonical:
            raise ValueError(f"noncanonical JSONL row at line {line_number}")
        if canonical in seen:
            raise ValueError(f"duplicate JSONL record at line {line_number}")
        seen.add(canonical)
        models.append(model)
    return models


def _artifact(path: Path, payload: bytes, media_type: str, count: int | None) -> ArtifactRef:
    return ArtifactRef(
        path=str(path.resolve(strict=False)),
        sha256=_sha256_bytes(payload),
        media_type=media_type,
        record_count=count,
    )


def _with_legacy_index(task: MemUpdateTask, index: int) -> MemUpdateTask:
    return with_legacy_index_profile(task, index)


def _task_manifest(
    tasks: list[MemUpdateTask],
    *,
    source_path: Path,
    source_hash: str,
    split: Split,
    phase: str,
    tasks_path: Path,
    tasks_bytes: bytes,
) -> TaskManifest:
    manifest = build_expected_legacy_task_manifest(
        tasks,
        tasks_path=tasks_path,
        tasks_bytes=tasks_bytes,
    )
    source_ref = manifest.source_manifest_paths_and_hashes[0]
    if (
        Path(source_ref.path) != source_path
        or source_ref.sha256 != source_hash
        or set(manifest.split_counts) != {item.value for item in Split}
        or any(task.metadata.split != split for task in tasks)
        or any(
            task.metadata.legacy_provenance is None
            or task.metadata.legacy_provenance.legacy_phase != phase
            for task in tasks
        )
    ):
        raise ValueError("dataset compiler inputs disagree with authenticated task identity")
    return manifest


def _validate_task_stage(path: Path) -> None:
    tasks = _load_canonical_jsonl(path, MemUpdateTask, "task_id")
    for task in tasks:
        report = validate_task_semantics(task)
        if not report.valid:
            raise ValueError(f"staged task semantic validation failed: {report.issues}")


def _compile_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source = _require_regular_file(Path(args.input), "dataset input")
    source_hash = _sha256_file(source)
    split = Split(args.split)
    rows = load_evomemory_dataset(source)
    tasks = [
        _with_legacy_index(
            compile_legacy_episode(
                row,
                source_path=source,
                source_sha256=source_hash,
                split=split,
                example_index=index,
                legacy_phase=args.legacy_phase,
            ),
            index,
        )
        for index, row in enumerate(rows)
    ]
    if not tasks:
        raise ValueError("dataset input contains no episodes")
    output_dir = Path(args.output_dir)
    tasks_path = output_dir / "tasks.jsonl"
    manifest_path = output_dir / "task_manifest.json"
    tasks_bytes = _canonical_jsonl(tasks)
    manifest = _task_manifest(
        tasks,
        source_path=source,
        source_hash=source_hash,
        split=split,
        phase=args.legacy_phase,
        tasks_path=tasks_path,
        tasks_bytes=tasks_bytes,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    publish_files_atomically(
        {tasks_path: tasks_bytes, manifest_path: manifest_bytes},
        overwrite=args.overwrite,
        source_paths=(source,),
        validators={
            tasks_path: _validate_task_stage,
            manifest_path: lambda path: _load_canonical_model(path, TaskManifest),
        },
        pre_publish=lambda: _verify_unchanged(
            {source: source_hash}, "dataset compilation"
        ),
    )
    return {
        "artifacts": {
            "task_manifest.json": _sha256_bytes(manifest_bytes),
            "tasks.jsonl": _sha256_bytes(tasks_bytes),
        },
        "command": "dataset",
        "row_counts": {"tasks.jsonl": len(tasks)},
        "success": True,
    }


def _task_manifest_for_tasks(
    tasks_path: Path, tasks: list[MemUpdateTask]
) -> tuple[TaskManifest, Path]:
    manifest_path = tasks_path.with_name("task_manifest.json")
    _require_regular_file(manifest_path, "task manifest")
    manifest = _load_canonical_model(manifest_path, TaskManifest)
    assert isinstance(manifest, TaskManifest)
    manifest = authenticate_legacy_task_manifest(
        manifest,
        tasks,
        tasks_path=tasks_path,
    )
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("task manifest schema_version is not current")
    if manifest.task_manifest_version != TASK_MANIFEST_VERSION:
        raise ValueError("task manifest version is not current")
    if manifest.task_schema_version != SCHEMA_VERSION:
        raise ValueError("task schema version is not current")
    if manifest.compiler_versions.get("vnext_compile_legacy") != CLI_COMPILER_VERSION:
        raise ValueError("task manifest compiler version is not current")
    if len(manifest.task_file_paths_and_hashes) != 1:
        raise ValueError("task manifest must reference exactly one task artifact")
    reference = manifest.task_file_paths_and_hashes[0]
    if Path(reference.path).resolve(strict=False) != tasks_path.resolve(strict=True):
        raise ValueError("task manifest path does not match supplied tasks artifact")
    tasks_hash = _sha256_file(tasks_path)
    if reference.sha256 != tasks_hash:
        raise ValueError("task manifest tasks hash does not match supplied tasks")
    if reference.record_count != len(tasks):
        raise ValueError("task manifest task count does not match supplied tasks")
    actual_splits = {
        item.value: sum(task.metadata.split == item for task in tasks)
        for item in Split
    }
    if dict(manifest.split_counts) != actual_splits:
        raise ValueError("task manifest split counts do not match supplied tasks")
    actual_family = dict(
        sorted(
            Counter(
                f"{task.task_family}|{task.difficulty.value}" for task in tasks
            ).items()
        )
    )
    if dict(manifest.family_difficulty_counts) != actual_family:
        raise ValueError("task manifest family counts do not match supplied tasks")
    semantic_counts = {
        item.value: len(
            {
                task.metadata.split_key.semantic_core_id
                for task in tasks
                if task.metadata.split == item
            }
        )
        for item in Split
    }
    if dict(manifest.semantic_core_counts) != semantic_counts:
        raise ValueError("task manifest semantic counts do not match supplied tasks")
    summary = manifest.leakage_check_summary
    if summary.get("compatibility_only") is not True:
        raise ValueError("task manifest is not compatibility-only")
    expected_output_hashes = {str(tasks_path.resolve(strict=True)): tasks_hash}
    if dict(summary.get("output_hashes", {})) != expected_output_hashes:
        raise ValueError("task manifest output hashes do not match supplied tasks")
    if dict(summary.get("row_counts", {})) != {"tasks.jsonl": len(tasks)}:
        raise ValueError("task manifest row counts do not match supplied tasks")
    for task in tasks:
        report = validate_task_semantics(task)
        if not report.valid:
            raise ValueError(f"compiled task semantic validation failed: {report.issues}")
        update_depth = task.metadata.resolved_profile.get("update_depth")
        expected_depth = task.metadata.extra.get("num_target_updates")
        if type(update_depth) is not int or update_depth <= 0:
            raise ValueError("compiled task resolved_profile.update_depth is missing or invalid")
        if type(expected_depth) is not int or update_depth != expected_depth:
            raise ValueError("compiled task update-depth linkage is inconsistent")
    split_report = validate_splits(tasks, task_manifest=manifest)
    if not split_report.valid:
        raise ValueError(f"compiled task split validation failed: {split_report.issues}")
    return manifest, manifest_path.resolve(strict=True)


def _task_index_map(
    tasks: list[MemUpdateTask], task_manifest: TaskManifest
) -> tuple[dict[int, MemUpdateTask], dict[Path, str]]:
    indexed: dict[int, MemUpdateTask] = {}
    authenticated_sources: dict[Path, str] = {}
    declared_sources = {
        (Path(ref.path).resolve(strict=False), ref.sha256)
        for ref in task_manifest.source_manifest_paths_and_hashes
    }
    for task in tasks:
        provenance = task.metadata.legacy_provenance
        if provenance is None:
            raise ValueError(f"task {task.task_id} lacks truthful LegacyProvenance")
        raw_index = task.metadata.extra.get("legacy_example_index")
        if type(raw_index) is not int or raw_index < 0:
            raise ValueError(f"task {task.task_id} has missing or invalid legacy_example_index")
        if raw_index in indexed:
            raise ValueError(f"duplicate legacy_example_index {raw_index}")
        source_path = _require_regular_file(
            Path(provenance.source_artifact_path), "task legacy source"
        )
        actual_hash = authenticated_sources.get(source_path)
        if actual_hash is None:
            actual_hash = _sha256_file(source_path)
        if actual_hash != provenance.source_artifact_hash:
            raise ValueError("task LegacyProvenance source hash is not authenticated")
        if (source_path, actual_hash) not in declared_sources:
            raise ValueError("task LegacyProvenance source is absent from TaskManifest")
        authenticated_sources[source_path] = actual_hash
        indexed[raw_index] = task
    expected = set(range(len(tasks)))
    if set(indexed) != expected:
        raise ValueError(
            f"legacy indices must be contiguous 0..{len(tasks) - 1}; got {sorted(indexed)}"
        )
    return indexed, dict(
        sorted(authenticated_sources.items(), key=lambda item: str(item[0]))
    )


def _validate_result_records(
    task_runs: list[TaskRunRecord],
    scores: list[ScoreRecord],
    manifest: RunManifest,
) -> None:
    if len(task_runs) != manifest.completed_task_count or len(scores) != len(task_runs):
        raise ValueError("result artifact counts do not match RunManifest")
    if {row.task_id for row in task_runs} != {row.task_id for row in scores}:
        raise ValueError("task-run and score task identities differ")
    if {row.run_id for row in [*task_runs, *scores]} != {manifest.run_id}:
        raise ValueError("result run identities differ from RunManifest")
    if {row.adapter_id for row in [*task_runs, *scores]} != {
        manifest.adapter_info.adapter_id
    }:
        raise ValueError("result adapter identities differ from RunManifest")


def _authenticated_result_indices(payload: dict[str, Any], source: Path) -> tuple[int, ...]:
    rows = payload.get("results")
    if type(rows) is not list or not rows:
        raise ValueError(f"{source} field=results: must be a nonempty exact list")
    indices: list[int] = []
    for row_number, row in enumerate(rows):
        if type(row) is not dict:
            raise ValueError(f"{source} field=results[{row_number}]: must be an exact object")
        index = row.get("example_id")
        if type(index) is not int or index < 0:
            raise ValueError(
                f"{source} field=results[{row_number}].example_id: "
                "must be an exact nonnegative integer"
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{source} field=results.example_id: duplicate global indices")
    return tuple(indices)


def _compile_results(args: argparse.Namespace) -> dict[str, Any]:
    source = _require_regular_file(Path(args.input), "results input")
    tasks_path = _require_regular_file(Path(args.tasks), "canonical tasks")
    source_hash = _sha256_file(source)
    tasks_hash = _sha256_file(tasks_path)
    task_models = _load_canonical_jsonl(tasks_path, MemUpdateTask, "task_id")
    tasks = [task for task in task_models if isinstance(task, MemUpdateTask)]
    task_manifest, task_manifest_path = _task_manifest_for_tasks(tasks_path, tasks)
    task_manifest_hash = _sha256_file(task_manifest_path)
    task_by_index, legacy_hashes = _task_index_map(tasks, task_manifest)
    legacy_sources = tuple(legacy_hashes)
    payload = load_evomemory_results(source)
    result_indices = authenticate_legacy_result_selection(
        payload,
        full_task_count=len(tasks),
        source_path=source,
    )
    missing_indices = [index for index in result_indices if index not in task_by_index]
    if missing_indices:
        raise ValueError(
            f"results reference missing compiled task indices: {missing_indices}"
        )
    selected_tasks = {index: task_by_index[index] for index in result_indices}
    run_name = source.parent.name or None
    manifest, task_runs, scores, warnings = import_evomemory_results(
        payload,
        source_path=source,
        source_sha256=source_hash,
        run_name=run_name,
        task_by_legacy_index=selected_tasks,
    )
    prompt_config = manifest.model_dump(mode="json")["prompt_config"]
    legacy_import = prompt_config["legacy_result_import"]
    legacy_import["compiled_task_selection"] = {
        "legacy_indices": list(result_indices),
        "task_ids": [selected_tasks[index].task_id for index in result_indices],
        "full_task_count": len(tasks),
        "task_file_sha256": tasks_hash,
    }
    prompt_config["legacy_result_import"] = legacy_import
    manifest = manifest.validated_replace(prompt_config=prompt_config)
    canonical_mode = manifest.prompt_config["legacy_result_import"][
        "canonical_evaluation_mode"
    ]
    if canonical_mode in {"slot_direct", "slot_prompt"}:
        raise ValueError("legacy answer mode cannot be a canonical evaluation_mode")
    output_dir = Path(args.output_dir)
    run_path = output_dir / "task_runs.jsonl"
    score_path = output_dir / "scores.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    run_bytes = _canonical_jsonl(task_runs)
    score_bytes = _canonical_jsonl(scores)
    input_hashes = {
        str(source): source_hash,
        str(tasks_path): tasks_hash,
        str(task_manifest_path): task_manifest_hash,
        **{str(path): digest for path, digest in legacy_hashes.items()},
    }
    output_hashes = {
        str(run_path.resolve(strict=False)): _sha256_bytes(run_bytes),
        str(score_path.resolve(strict=False)): _sha256_bytes(score_bytes),
    }
    native_summary = manifest.model_dump(mode="json")[
        "native_vs_extracted_field_summary"
    ]
    updated_manifest = manifest.validated_replace(
        task_manifest=ArtifactRef(
            path=str(task_manifest_path),
            sha256=task_manifest_hash,
            media_type=_JSON_MEDIA_TYPE,
            record_count=len(tasks),
        ),
        normalized_runtime_artifacts=(
            _artifact(run_path, run_bytes, _JSONL_MEDIA_TYPE, len(task_runs)),
        ),
        score_artifacts=(
            _artifact(score_path, score_bytes, _JSONL_MEDIA_TYPE, len(scores)),
        ),
        package_summary={
            **dict(manifest.package_summary),
            "compiler_version": CLI_COMPILER_VERSION,
        },
        native_vs_extracted_field_summary={
            **dict(native_summary),
            "compatibility_only": True,
            "warnings": warnings,
            "row_counts": {
                "task_runs.jsonl": len(task_runs),
                "scores.jsonl": len(scores),
            },
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
        },
    )
    manifest_bytes = canonical_json_bytes(updated_manifest)
    _validate_result_records(task_runs, scores, updated_manifest)
    publish_files_atomically(
        {
            run_path: run_bytes,
            score_path: score_bytes,
            manifest_path: manifest_bytes,
        },
        overwrite=args.overwrite,
        source_paths=(source, tasks_path, task_manifest_path, *legacy_sources),
        validators={
            run_path: lambda path: _load_canonical_jsonl(path, TaskRunRecord, "task_id"),
            score_path: lambda path: _load_canonical_jsonl(path, ScoreRecord, "task_id"),
            manifest_path: lambda path: _load_canonical_model(path, RunManifest),
        },
        pre_publish=lambda: _verify_unchanged(
            {
                source: source_hash,
                tasks_path: tasks_hash,
                task_manifest_path: task_manifest_hash,
                **legacy_hashes,
            },
            "results compilation",
        ),
    )
    return {
        "artifacts": {
            "run_manifest.json": _sha256_bytes(manifest_bytes),
            "scores.jsonl": _sha256_bytes(score_bytes),
            "task_runs.jsonl": _sha256_bytes(run_bytes),
        },
        "command": "results",
        "row_counts": {
            "scores.jsonl": len(scores),
            "task_runs.jsonl": len(task_runs),
        },
        "success": True,
        "warnings": warnings,
    }


def _legacy_manifest(
    *,
    kind: str,
    source: Path,
    source_hash: str,
    output_path: Path,
    output_bytes: bytes,
    row_count: int,
    warnings: tuple[str, ...],
    caveats: tuple[str, ...],
    media_type: str,
    source_media_type: str,
    row_counts: dict[str, int] | None = None,
) -> LegacyAnalysisManifest:
    return LegacyAnalysisManifest(
        analysis_kind=kind,
        compiler_version=CLI_COMPILER_VERSION,
        compatibility_only=True,
        source_artifacts=(
            ArtifactRef(
                path=str(source),
                sha256=source_hash,
                media_type=source_media_type,
                record_count=row_count,
            ),
        ),
        output_artifacts=(
            _artifact(output_path, output_bytes, media_type, row_count),
        ),
        row_counts=(
            row_counts if row_counts is not None else {output_path.name: row_count}
        ),
        warnings=warnings,
        caveats=caveats,
        code_revision=_CODE_REVISION,
    )


def _compile_mechanism(args: argparse.Namespace) -> dict[str, Any]:
    source = _require_regular_file(Path(args.input), "mechanism input")
    source_hash = _sha256_file(source)
    importer = {
        "conflict": import_conflict_probe,
        "dose": import_synthetic_dose,
        "stale-removal": import_stale_removal_trace,
        "api": import_api_probe,
    }[args.kind]
    cells = importer(source)
    logical_identities = [
        (
            type(cell).__name__,
            cell.legacy_namespace,
            cell.config_sha256,
            getattr(cell, "status", None),
        )
        for cell in cells
    ]
    if len(logical_identities) != len(set(logical_identities)):
        raise ValueError("mechanism input contains duplicate logical cell identity")
    serialized = sorted(
        ((canonical_json_bytes(cell), cell) for cell in cells),
        key=lambda item: item[0],
    )
    identities = [row for row, _ in serialized]
    if len(identities) != len(set(identities)):
        raise ValueError("mechanism input contains duplicate canonical records")
    ordered = [cell for _, cell in serialized]
    row_bytes = b"".join(row + b"\n" for row in identities)
    caveats = tuple(
        dict.fromkeys(caveat for cell in ordered for caveat in cell.caveats)
    )
    output_dir = Path(args.output_dir)
    output_path = output_dir / "legacy_analysis.jsonl"
    manifest_path = output_dir / "legacy_analysis_manifest.json"
    manifest = _legacy_manifest(
        kind=args.kind,
        source=source,
        source_hash=source_hash,
        output_path=output_path,
        output_bytes=row_bytes,
        row_count=len(ordered),
        warnings=(),
        caveats=caveats,
        media_type=_JSONL_MEDIA_TYPE,
        source_media_type={
            ".csv": "text/csv",
            ".json": _JSON_MEDIA_TYPE,
        }[source.suffix.lower()],
    )
    manifest_bytes = canonical_json_bytes(manifest)
    publish_files_atomically(
        {output_path: row_bytes, manifest_path: manifest_bytes},
        overwrite=args.overwrite,
        source_paths=(source,),
        validators={
            output_path: lambda path: _load_unkeyed_canonical_jsonl(
                path, type(ordered[0])
            ),
            manifest_path: lambda path: _load_canonical_model(
                path, LegacyAnalysisManifest
            ),
        },
        pre_publish=lambda: _verify_unchanged(
            {source: source_hash}, "mechanism compilation"
        ),
    )
    return {
        "artifacts": {
            output_path.name: _sha256_bytes(row_bytes),
            manifest_path.name: _sha256_bytes(manifest_bytes),
        },
        "command": "mechanism",
        "kind": args.kind,
        "row_counts": {output_path.name: len(ordered)},
        "success": True,
    }


def _compile_ledger(args: argparse.Namespace) -> dict[str, Any]:
    source = _require_regular_file(Path(args.input), "ledger input")
    if source.suffix.lower() != ".md":
        raise ValueError("ledger input must have an exact .md extension")
    source_hash = _sha256_file(source)
    root = Path(args.project_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"project root is not a directory: {root}")
    audit = audit_ledger_references(source, root)
    audit_bytes = canonical_json_bytes(audit)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "ledger_audit.json"
    manifest_path = output_dir / "legacy_analysis_manifest.json"
    reasons = tuple(dict.fromkeys(item.reason for item in audit.unresolved))
    warnings = ("unresolved_ledger_references",) if audit.unresolved else ()
    caveats = tuple(f"unresolved_reason:{reason}" for reason in reasons)
    manifest = _legacy_manifest(
        kind="ledger",
        source=source,
        source_hash=source_hash,
        output_path=output_path,
        output_bytes=audit_bytes,
        row_count=1,
        warnings=warnings,
        caveats=caveats,
        media_type=_JSON_MEDIA_TYPE,
        source_media_type="text/markdown",
    )
    manifest_bytes = canonical_json_bytes(manifest)
    publish_files_atomically(
        {output_path: audit_bytes, manifest_path: manifest_bytes},
        overwrite=args.overwrite,
        source_paths=(source,),
        validators={
            output_path: lambda path: _load_canonical_model(path, type(audit)),
            manifest_path: lambda path: _load_canonical_model(
                path, LegacyAnalysisManifest
            ),
        },
        pre_publish=lambda: _verify_unchanged(
            {source: source_hash}, "ledger compilation"
        ),
    )
    return {
        "artifacts": {
            output_path.name: _sha256_bytes(audit_bytes),
            manifest_path.name: _sha256_bytes(manifest_bytes),
        },
        "command": "ledger",
        "row_counts": dict(manifest.row_counts),
        "success": True,
        "warnings": list(warnings),
    }


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true")


class _ArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError("invalid command line")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Compile authenticated legacy artifacts into vNext compatibility outputs.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", allow_abbrev=False)
    dataset.add_argument("--input", required=True)
    dataset.add_argument("--split", required=True, choices=[item.value for item in Split])
    dataset.add_argument("--legacy-phase", required=True)
    dataset.add_argument("--output-dir", required=True)
    _add_overwrite(dataset)
    dataset.set_defaults(handler=_compile_dataset)

    results = subparsers.add_parser("results", allow_abbrev=False)
    results.add_argument("--input", required=True)
    results.add_argument("--tasks", required=True)
    results.add_argument("--output-dir", required=True)
    _add_overwrite(results)
    results.set_defaults(handler=_compile_results)

    mechanism = subparsers.add_parser("mechanism", allow_abbrev=False)
    mechanism.add_argument(
        "--kind", required=True, choices=("conflict", "dose", "stale-removal", "api")
    )
    mechanism.add_argument("--input", required=True)
    mechanism.add_argument("--output-dir", required=True)
    _add_overwrite(mechanism)
    mechanism.set_defaults(handler=_compile_mechanism)

    ledger = subparsers.add_parser("ledger", allow_abbrev=False)
    ledger.add_argument("--input", required=True)
    ledger.add_argument("--project-root", required=True)
    ledger.add_argument("--output-dir", required=True)
    _add_overwrite(ledger)
    ledger.set_defaults(handler=_compile_ledger)
    return parser


def _print_json(payload: dict[str, Any], *, stream) -> None:
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        file=stream,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except _ArgumentError:
        _print_json(
            {
                "errors": [{"code": "invalid_arguments", "type": "ArgumentError"}],
                "kind": None,
                "valid": False,
                "warnings": [],
            },
            stream=sys.stdout,
        )
        return 2
    try:
        report = args.handler(args)
    except Exception as exc:
        message = " ".join(str(exc).splitlines())
        _print_json(
            {
                "error": type(exc).__name__,
                "message": message[:2000],
                "success": False,
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(report, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
