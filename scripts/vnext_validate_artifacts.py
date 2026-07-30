from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any

from pydantic import BaseModel

from mub.vnext.contracts import MemUpdateTask, RunManifest, ScoreRecord, TaskManifest, TaskRunRecord
from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.enums import CompletionStatus, Split, SupportReason
from mub.vnext.io.canonical import canonical_json_bytes, sha256_model
from mub.vnext.io.jsonl import read_models
from mub.vnext.legacy.artifacts import (
    LEGACY_CLI_COMPILER_VERSION,
    _authenticate_and_validate_legacy_tasks,
)
from mub.vnext.legacy.loaders import load_evomemory_results
from mub.vnext.legacy.results import (
    LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH,
    authenticate_legacy_result_selection,
    import_evomemory_results,
    is_legacy_evomemory_adapter_identity,
)
from mub.vnext.validation import validate_splits
from mub.vnext.version import (
    METRIC_REGISTRY_VERSION,
    PROFILE_VERSION,
    RUN_MANIFEST_VERSION,
    RUNTIME_RECORD_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TASK_MANIFEST_VERSION,
)


KINDS = ("tasks", "task-runs", "scores", "task-manifest", "run-manifest")
CLI_COMPILER_VERSION = LEGACY_CLI_COMPILER_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist") from exc
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return path.resolve(strict=True)


def _strict_json_payload(raw: bytes) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant")

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
        parse_float=parse_float,
    )


def _load_canonical_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    raw = path.read_bytes()
    model = model_type.model_validate(_strict_json_payload(raw))
    if raw != canonical_json_bytes(model):
        raise ValueError("artifact is not canonical JSON")
    return model


def _canonical_jsonl(models: list[BaseModel]) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def _load_canonical_jsonl(
    path: Path,
    model_type: type[BaseModel],
    id_field: str,
) -> list[BaseModel]:
    raw = path.read_bytes()
    models = list(read_models(path, model_type, id_field=id_field))
    if raw != _canonical_jsonl(models):
        raise ValueError("artifact is not canonical JSONL")
    return models


def _resolve_ref(ref: ArtifactRef, manifest_path: Path) -> Path:
    candidate = Path(ref.path)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return _require_regular_file(candidate, "referenced artifact")


def _verify_ref(ref: ArtifactRef, manifest_path: Path) -> Path:
    path = _resolve_ref(ref, manifest_path)
    if _sha256_file(path) != ref.sha256:
        raise ValueError("referenced artifact hash mismatch")
    return path


def _verify_current_task_manifest(manifest: TaskManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("task manifest schema_version is not current")
    if manifest.task_manifest_version != TASK_MANIFEST_VERSION:
        raise ValueError("task_manifest_version is not current")
    if manifest.task_schema_version != SCHEMA_VERSION:
        raise ValueError("task_schema_version is not current")
    if manifest.compiler_versions.get("vnext_compile_legacy") != CLI_COMPILER_VERSION:
        raise ValueError("task manifest compiler version is not current")
    if manifest.leakage_check_summary.get("compatibility_only") is not True:
        raise ValueError("legacy task manifest must declare compatibility_only=true")


def _validate_declared_summary(
    summary: Any,
    *,
    input_refs: tuple[ArtifactRef, ...],
    output_refs: tuple[ArtifactRef, ...],
    expected_rows: dict[str, int],
    manifest_path: Path,
) -> None:
    input_hashes = summary.get("input_hashes")
    output_hashes = summary.get("output_hashes")
    row_counts = summary.get("row_counts")
    warnings = summary.get("warnings")
    if not hasattr(input_hashes, "items") or not hasattr(output_hashes, "items"):
        raise ValueError("manifest summary must declare input/output hashes")
    if not hasattr(row_counts, "items") or dict(row_counts) != expected_rows:
        raise ValueError("manifest summary row counts do not match artifacts")
    if type(warnings) not in {list, tuple} or any(
        type(item) is not str or not item.strip() for item in warnings
    ):
        raise ValueError("manifest warnings must be explicit strings")
    expected_input = {
        str(_resolve_ref(ref, manifest_path)): ref.sha256 for ref in input_refs
    }
    declared_input = {
        str(Path(path).resolve(strict=False)): digest
        for path, digest in input_hashes.items()
    }
    if declared_input != expected_input:
        raise ValueError("manifest summary input hashes do not match artifact refs")
    expected_output = {
        str(_resolve_ref(ref, manifest_path)): ref.sha256 for ref in output_refs
    }
    declared_output = {
        str(Path(path).resolve(strict=False)): digest
        for path, digest in output_hashes.items()
    }
    if declared_output != expected_output:
        raise ValueError("manifest summary output hashes do not match artifact refs")


def _load_tasks_from_manifest(
    manifest: TaskManifest, manifest_path: Path
) -> list[MemUpdateTask]:
    """Authenticate the legacy compiler graph before legacy semantic validation."""
    all_tasks: list[MemUpdateTask] = []
    task_paths: list[Path] = []
    for ref in manifest.task_file_paths_and_hashes:
        path = _verify_ref(ref, manifest_path)
        task_paths.append(path)
        rows = _load_canonical_jsonl(path, MemUpdateTask, "task_id")
        if ref.record_count is not None and ref.record_count != len(rows):
            raise ValueError("task artifact count mismatch")
        all_tasks.extend(row for row in rows if isinstance(row, MemUpdateTask))
    if len(task_paths) != 1:
        raise ValueError("TaskManifest must reference exactly one task artifact")
    manifest, semantic_reports = _authenticate_and_validate_legacy_tasks(
        manifest,
        all_tasks,
        tasks_path=task_paths[0],
    )
    ids = [task.task_id for task in all_tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate tasks across declared task artifacts")
    if sum(manifest.split_counts.values()) != len(all_tasks):
        raise ValueError("TaskManifest split count mismatch")
    actual_splits = {
        item.value: sum(task.metadata.split == item for task in all_tasks)
        for item in Split
    }
    if dict(manifest.split_counts) != dict(sorted(actual_splits.items())):
        raise ValueError("TaskManifest split counts do not match task rows")
    actual_family = Counter(
        f"{task.task_family}|{task.difficulty.value}" for task in all_tasks
    )
    if dict(manifest.family_difficulty_counts) != dict(sorted(actual_family.items())):
        raise ValueError("TaskManifest family/difficulty counts do not match task rows")
    summary = dict(manifest.semantic_core_counts)
    expected_semantic = {
        item.value: len(
            {
                task.metadata.split_key.semantic_core_id
                for task in all_tasks
                if task.metadata.split == item
            }
        )
        for item in Split
    }
    if summary != expected_semantic:
        raise ValueError("TaskManifest semantic-core counts do not match task rows")
    for task, report in zip(all_tasks, semantic_reports, strict=True):
        if task.schema_version != SCHEMA_VERSION:
            raise ValueError("task schema_version is not current")
        if not report.valid:
            raise ValueError("canonical task semantic validation failed")
        if any(
            query.evaluation_mode.value in {"slot_direct", "slot_prompt"}
            for query in task.queries
        ):
            raise ValueError("legacy answer mode used as canonical evaluation_mode")
    for ref in manifest.source_manifest_paths_and_hashes:
        source = _verify_ref(ref, manifest_path)
        if ref.record_count is not None and source.suffix.lower() == ".json":
            payload = _strict_json_payload(source.read_bytes())
            if type(payload) is list and len(payload) != ref.record_count:
                raise ValueError("source artifact count mismatch")
    for ref in manifest.generation_configs_and_hashes:
        _verify_ref(ref, manifest_path)
    for ref in manifest.human_audit_artifacts:
        _verify_ref(ref, manifest_path)
    declared_task_hashes = manifest.leakage_check_summary.get("task_hashes")
    expected_task_hashes = {
        task.task_id: sha256_model(task)
        for task in sorted(all_tasks, key=lambda item: item.task_id)
    }
    if not hasattr(declared_task_hashes, "items") or dict(
        declared_task_hashes
    ) != expected_task_hashes:
        raise ValueError("TaskManifest task hashes do not match task rows")
    split_report = validate_splits(all_tasks, task_manifest=manifest)
    if not split_report.valid:
        raise ValueError("TaskManifest split validation failed")
    _validate_declared_summary(
        manifest.leakage_check_summary,
        input_refs=manifest.source_manifest_paths_and_hashes,
        output_refs=manifest.task_file_paths_and_hashes,
        expected_rows={
            Path(ref.path).name: ref.record_count
            for ref in manifest.task_file_paths_and_hashes
            if ref.record_count is not None
        },
        manifest_path=manifest_path,
    )
    declared_sources = {
        (_resolve_ref(ref, manifest_path), ref.sha256)
        for ref in manifest.source_manifest_paths_and_hashes
    }
    legacy_indices: list[int] = []
    for task in all_tasks:
        provenance = task.metadata.legacy_provenance
        if provenance is None:
            raise ValueError("compiled legacy task lacks LegacyProvenance")
        source = Path(provenance.source_artifact_path).resolve(strict=False)
        if (source, provenance.source_artifact_hash) not in declared_sources:
            raise ValueError("task LegacyProvenance is not authenticated by TaskManifest")
        if task.source.raw_hash != provenance.source_artifact_hash:
            raise ValueError("task source and LegacyProvenance hashes differ")
        index = task.metadata.extra.get("legacy_example_index")
        if type(index) is not int or index < 0:
            raise ValueError("task lacks a valid legacy_example_index")
        legacy_indices.append(index)
    if sorted(legacy_indices) != list(range(len(all_tasks))):
        raise ValueError("legacy_example_index values are duplicate or missing")
    return all_tasks


def _verify_current_run_manifest(manifest: RunManifest) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "task_schema_version": SCHEMA_VERSION,
        "runtime_record_version": RUNTIME_RECORD_VERSION,
        "scorer_version": SCORER_VERSION,
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "profile_version": PROFILE_VERSION,
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise ValueError(f"RunManifest {field} is not current")
    if manifest.native_vs_extracted_field_summary.get("compatibility_only") is not True:
        raise ValueError("legacy RunManifest must declare compatibility_only=true")
    if manifest.package_summary.get("compiler_version") != CLI_COMPILER_VERSION:
        raise ValueError("RunManifest compiler version is not current")
    if not is_legacy_evomemory_adapter_identity(manifest):
        raise ValueError("legacy RunManifest adapter identity is not importer-authentic")
    metadata = manifest.prompt_config.get("legacy_result_import")
    if type(metadata) is not dict and not hasattr(metadata, "get"):
        raise ValueError("legacy RunManifest lacks import metadata")
    canonical_mode = metadata.get("canonical_evaluation_mode")
    if canonical_mode in {"slot_direct", "slot_prompt"}:
        raise ValueError("legacy answer mode used as canonical evaluation_mode")
    if canonical_mode not in {None, "state_direct", "retrieved_prompt", "native_system"}:
        raise ValueError("unknown canonical evaluation_mode evidence")


def _load_task_context(manifest: RunManifest, manifest_path: Path) -> tuple[TaskManifest, Path, list[MemUpdateTask]]:
    if manifest.task_manifest.media_type != "application/json":
        raise ValueError("RunManifest task_manifest media_type is invalid")
    if manifest.task_manifest.record_count is None:
        raise ValueError("RunManifest task_manifest record_count is required")
    task_manifest_path = _verify_ref(manifest.task_manifest, manifest_path)
    task_manifest = _load_canonical_model(task_manifest_path, TaskManifest)
    assert isinstance(task_manifest, TaskManifest)
    _verify_current_task_manifest(task_manifest)
    tasks = _load_tasks_from_manifest(task_manifest, task_manifest_path)
    if manifest.task_manifest.record_count is not None and manifest.task_manifest.record_count != len(tasks):
        raise ValueError("RunManifest task manifest count mismatch")
    return task_manifest, task_manifest_path, tasks


def _load_run_rows(manifest: RunManifest, manifest_path: Path) -> list[TaskRunRecord]:
    if not manifest.normalized_runtime_artifacts:
        raise ValueError("RunManifest requires runtime artifacts")
    rows: list[TaskRunRecord] = []
    for ref in manifest.normalized_runtime_artifacts:
        if ref.media_type != "application/x-ndjson" or ref.record_count is None:
            raise ValueError("runtime artifacts require exact media_type and record_count")
        path = _verify_ref(ref, manifest_path)
        loaded = _load_canonical_jsonl(path, TaskRunRecord, "task_id")
        if ref.record_count != len(loaded):
            raise ValueError("task-run artifact count mismatch")
        rows.extend(row for row in loaded if isinstance(row, TaskRunRecord))
    if len({row.task_id for row in rows}) != len(rows):
        raise ValueError("duplicate task-run records across artifacts")
    return rows


def _load_score_rows(manifest: RunManifest, manifest_path: Path) -> list[ScoreRecord]:
    if not manifest.score_artifacts:
        raise ValueError("RunManifest requires score artifacts")
    rows: list[ScoreRecord] = []
    for ref in manifest.score_artifacts:
        if ref.media_type != "application/x-ndjson" or ref.record_count is None:
            raise ValueError("score artifacts require exact media_type and record_count")
        path = _verify_ref(ref, manifest_path)
        loaded = _load_canonical_jsonl(path, ScoreRecord, "task_id")
        if ref.record_count != len(loaded):
            raise ValueError("score artifact count mismatch")
        rows.extend(row for row in loaded if isinstance(row, ScoreRecord))
    if len({row.task_id for row in rows}) != len(rows):
        raise ValueError("duplicate score records across artifacts")
    return rows


def _validate_score_support(score: ScoreRecord) -> None:
    for support in score.supported_metric_fields.values():
        if not isinstance(support.reason, SupportReason):
            raise ValueError("ScoreRecord support reason is not canonical")


def _selected_tasks_for_run(
    manifest: RunManifest,
    task_manifest: TaskManifest,
    tasks: list[MemUpdateTask],
) -> list[MemUpdateTask]:
    metadata = manifest.prompt_config["legacy_result_import"]
    selection = metadata.get("compiled_task_selection")
    if selection is None:
        return tasks
    if not hasattr(selection, "get"):
        raise ValueError("compiled task selection must be an object")
    indices = selection.get("legacy_indices")
    task_ids = selection.get("task_ids")
    full_count = selection.get("full_task_count")
    task_hash = selection.get("task_file_sha256")
    if type(indices) not in {list, tuple} or not indices:
        raise ValueError("compiled task selection indices are missing")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("compiled task selection indices are invalid")
    if len(indices) != len(set(indices)):
        raise ValueError("compiled task selection indices are duplicate")
    if type(task_ids) not in {list, tuple} or any(
        type(task_id) is not str or not task_id for task_id in task_ids
    ):
        raise ValueError("compiled task selection task IDs are invalid")
    if full_count != len(tasks):
        raise ValueError("compiled task selection full count is invalid")
    if len(task_manifest.task_file_paths_and_hashes) != 1:
        raise ValueError("compiled task selection requires one task artifact")
    if task_hash != task_manifest.task_file_paths_and_hashes[0].sha256:
        raise ValueError("compiled task selection task hash is invalid")
    by_index: dict[int, MemUpdateTask] = {}
    for task in tasks:
        index = task.metadata.extra.get("legacy_example_index")
        if type(index) is not int or index in by_index:
            raise ValueError("compiled task legacy indices are invalid")
        by_index[index] = task
    try:
        selected = [by_index[index] for index in indices]
    except KeyError as exc:
        raise ValueError("compiled task selection references missing index") from exc
    if list(task_ids) != [task.task_id for task in selected]:
        raise ValueError("compiled task selection IDs do not match indices")
    return selected


def _validate_runtime_provenance(
    manifest: RunManifest,
    runs: list[TaskRunRecord],
    manifest_path: Path,
) -> None:
    legacy_evomemory = is_legacy_evomemory_adapter_identity(manifest)
    if legacy_evomemory:
        if len(manifest.raw_provider_response_artifacts) != 1:
            raise ValueError(
                "legacy RunManifest requires exactly one raw provider artifact"
            )
        provider_ref = manifest.raw_provider_response_artifacts[0]
        if (
            provider_ref.media_type != "application/json"
            or provider_ref.record_count != manifest.expected_task_count
        ):
            raise ValueError(
                "legacy raw provider artifact media type or count is invalid"
            )
        if manifest.raw_adapter_state_artifacts:
            raise ValueError("legacy RunManifest may not declare raw adapter state")
        if len(runs) != manifest.expected_task_count:
            raise ValueError("legacy runtime row count does not match RunManifest")
    raw_groups = {
        "raw provider": manifest.raw_provider_response_artifacts,
        "raw adapter state": manifest.raw_adapter_state_artifacts,
    }
    authenticated: dict[str, list[tuple[Path, str]]] = {}
    for label, refs in raw_groups.items():
        identities = [(_verify_ref(ref, manifest_path), ref.sha256) for ref in refs]
        if len(identities) != len(set(identities)):
            raise ValueError(f"RunManifest contains duplicate {label} artifact references")
        authenticated[label] = identities

    expected_versions = {
        "action_parser_version": manifest.action_parser_version,
        "answer_parser_version": manifest.answer_parser_version,
        "memory_entry_extractor_version": manifest.memory_entry_extractor_version,
        "redaction_policy_version": manifest.redaction_policy_version,
    }
    if (
        legacy_evomemory
        and manifest.object_value_extractor_config_hash
        != LEGACY_OBJECT_EXTRACTOR_UNAVAILABLE_HASH
    ):
        raise ValueError(
            "legacy unavailable object extractor does not match RunManifest"
        )
    for run in runs:
        provenance = run.parser_extractor_provenance
        for field, expected in expected_versions.items():
            if getattr(provenance, field) != expected:
                raise ValueError(
                    f"TaskRunRecord {field} does not match RunManifest"
                )
        object_hash = provenance.object_value_extractor_config_hash
        if legacy_evomemory:
            if object_hash is not None:
                raise ValueError(
                    "legacy TaskRunRecord must declare unavailable object extractor"
                )
        elif object_hash != manifest.object_value_extractor_config_hash:
            raise ValueError(
                "TaskRunRecord object extractor hash does not match RunManifest"
            )

        if legacy_evomemory:
            if (
                provenance.raw_provider_artifact_path is None
                or provenance.raw_provider_artifact_hash is None
            ):
                raise ValueError(
                    "legacy TaskRunRecord requires raw provider provenance"
                )
            if (
                provenance.raw_adapter_state_path is not None
                or provenance.raw_adapter_state_hash is not None
            ):
                raise ValueError(
                    "legacy TaskRunRecord may not declare raw adapter state"
                )
        pairs = (
            (
                "raw provider",
                provenance.raw_provider_artifact_path,
                provenance.raw_provider_artifact_hash,
            ),
            (
                "raw adapter state",
                provenance.raw_adapter_state_path,
                provenance.raw_adapter_state_hash,
            ),
        )
        for label, raw_path, raw_hash in pairs:
            if (raw_path is None) != (raw_hash is None):
                raise ValueError(f"TaskRunRecord {label} path/hash pair is incomplete")
            if raw_path is None:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            candidate = _require_regular_file(candidate, f"TaskRunRecord {label} artifact")
            matches = [
                identity
                for identity in authenticated[label]
                if identity == (candidate, raw_hash)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"TaskRunRecord {label} artifact is not authenticated exactly once"
                )


def _authenticate_legacy_result_derivation(
    manifest: RunManifest,
    manifest_path: Path,
    task_manifest: TaskManifest,
    task_manifest_path: Path,
    all_tasks: list[MemUpdateTask],
    runs: list[TaskRunRecord],
    scores: list[ScoreRecord],
) -> list[MemUpdateTask]:
    if (
        len(manifest.raw_provider_response_artifacts) != 1
        or len(manifest.normalized_runtime_artifacts) != 1
        or len(manifest.score_artifacts) != 1
        or len(task_manifest.task_file_paths_and_hashes) != 1
    ):
        raise ValueError("legacy result reconstruction requires singular artifacts")
    raw_ref = manifest.raw_provider_response_artifacts[0]
    raw_path = _verify_ref(raw_ref, manifest_path)
    payload = load_evomemory_results(raw_path)
    indices = list(
        authenticate_legacy_result_selection(
            payload,
            full_task_count=len(all_tasks),
            source_path=raw_path,
        )
    )
    tasks_by_index: dict[int, MemUpdateTask] = {}
    for task in all_tasks:
        index = task.metadata.extra.get("legacy_example_index")
        if type(index) is not int or index in tasks_by_index:
            raise ValueError("legacy task indices are not reconstructable")
        tasks_by_index[index] = task
    try:
        selected_map = {index: tasks_by_index[index] for index in indices}
    except KeyError as exc:
        raise ValueError("legacy result source references missing task") from exc
    expected_manifest, expected_runs, expected_scores, warnings = import_evomemory_results(
        payload,
        source_path=raw_path,
        source_sha256=raw_ref.sha256,
        run_name=raw_path.parent.name or None,
        task_by_legacy_index=selected_map,
    )
    if [canonical_json_bytes(row) for row in runs] != [
        canonical_json_bytes(row) for row in expected_runs
    ]:
        raise ValueError("runtime rows do not match authenticated legacy derivation")
    if [canonical_json_bytes(row) for row in scores] != [
        canonical_json_bytes(row) for row in expected_scores
    ]:
        raise ValueError("score rows do not match authenticated legacy derivation")

    task_ref = task_manifest.task_file_paths_and_hashes[0]
    task_path = _verify_ref(task_ref, task_manifest_path)
    prompt_config = expected_manifest.model_dump(mode="json")["prompt_config"]
    prompt_config["legacy_result_import"]["compiled_task_selection"] = {
        "legacy_indices": indices,
        "task_ids": [selected_map[index].task_id for index in indices],
        "full_task_count": len(all_tasks),
        "task_file_sha256": task_ref.sha256,
    }
    run_ref = manifest.normalized_runtime_artifacts[0]
    score_ref = manifest.score_artifacts[0]
    expected_run_bytes = _canonical_jsonl(expected_runs)
    expected_score_bytes = _canonical_jsonl(expected_scores)
    expected_input_hashes = {
        str(raw_path): raw_ref.sha256,
        str(task_path): task_ref.sha256,
        str(task_manifest_path): manifest.task_manifest.sha256,
        **{
            str(_resolve_ref(ref, task_manifest_path)): ref.sha256
            for ref in task_manifest.source_manifest_paths_and_hashes
        },
    }
    run_path = _resolve_ref(run_ref, manifest_path)
    score_path = _resolve_ref(score_ref, manifest_path)
    native_summary = expected_manifest.model_dump(mode="json")[
        "native_vs_extracted_field_summary"
    ]
    reconstructed = expected_manifest.validated_replace(
        task_manifest=ArtifactRef(
            path=str(task_manifest_path),
            sha256=manifest.task_manifest.sha256,
            media_type="application/json",
            record_count=len(all_tasks),
        ),
        prompt_config=prompt_config,
        normalized_runtime_artifacts=(
            ArtifactRef(
                path=str(run_path),
                sha256=hashlib.sha256(expected_run_bytes).hexdigest(),
                media_type="application/x-ndjson",
                record_count=len(expected_runs),
            ),
        ),
        score_artifacts=(
            ArtifactRef(
                path=str(score_path),
                sha256=hashlib.sha256(expected_score_bytes).hexdigest(),
                media_type="application/x-ndjson",
                record_count=len(expected_scores),
            ),
        ),
        package_summary={
            **dict(expected_manifest.package_summary),
            "compiler_version": CLI_COMPILER_VERSION,
        },
        native_vs_extracted_field_summary={
            **dict(native_summary),
            "compatibility_only": True,
            "warnings": warnings,
            "row_counts": {
                "task_runs.jsonl": len(expected_runs),
                "scores.jsonl": len(expected_scores),
            },
            "input_hashes": expected_input_hashes,
            "output_hashes": {
                str(run_path): hashlib.sha256(expected_run_bytes).hexdigest(),
                str(score_path): hashlib.sha256(expected_score_bytes).hexdigest(),
            },
        },
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(reconstructed):
        raise ValueError("RunManifest does not match authenticated legacy derivation")
    return [selected_map[index] for index in indices]


def _validate_run_graph(manifest: RunManifest, manifest_path: Path) -> tuple[list[TaskRunRecord], list[ScoreRecord]]:
    task_manifest, task_manifest_path, tasks = _load_task_context(manifest, manifest_path)
    selected_tasks = _selected_tasks_for_run(manifest, task_manifest, tasks)
    task_ids = {task.task_id for task in selected_tasks}
    runs = _load_run_rows(manifest, manifest_path)
    scores = _load_score_rows(manifest, manifest_path)
    reconstructed_tasks = _authenticate_legacy_result_derivation(
        manifest,
        manifest_path,
        task_manifest,
        task_manifest_path,
        tasks,
        runs,
        scores,
    )
    if [task.task_id for task in selected_tasks] != [
        task.task_id for task in reconstructed_tasks
    ]:
        raise ValueError("manifest task selection differs from source reconstruction")
    _validate_runtime_provenance(manifest, runs, manifest_path)
    run_ids = {row.run_id for row in runs}
    score_run_ids = {row.run_id for row in scores}
    if run_ids - {manifest.run_id} or score_run_ids - {manifest.run_id}:
        raise ValueError("cross-record run_id mismatch")
    adapter_ids = {row.adapter_id for row in runs + scores}
    if adapter_ids - {manifest.adapter_info.adapter_id}:
        raise ValueError("cross-record adapter_id mismatch")
    if {row.task_id for row in runs} != task_ids:
        raise ValueError("task-run task IDs do not match TaskManifest")
    if {row.task_id for row in scores} != task_ids:
        raise ValueError("score task IDs do not match TaskManifest")
    canonical_mode = manifest.prompt_config["legacy_result_import"].get(
        "canonical_evaluation_mode"
    )
    if canonical_mode is not None:
        if any(
            query.evaluation_mode.value != canonical_mode
            for task in selected_tasks
            for query in task.queries
        ):
            raise ValueError("canonical evaluation_mode lacks linked task semantics")
        if canonical_mode == "state_direct" and any(
            not row.memory_snapshots for row in runs
        ):
            raise ValueError("state_direct mapping lacks materialized state evidence")
        if canonical_mode == "retrieved_prompt" and any(
            not row.retrieval_traces for row in runs
        ):
            raise ValueError("retrieved_prompt mapping lacks materialized retrieval evidence")
    task_by_id = {task.task_id: task for task in selected_tasks}
    run_by_id = {row.task_id: row for row in runs}
    for score in scores:
        task = task_by_id[score.task_id]
        run = run_by_id[score.task_id]
        if score.task_family != task.task_family:
            raise ValueError("ScoreRecord task_family does not match linked task")
        if score.difficulty != task.difficulty:
            raise ValueError("ScoreRecord difficulty does not match linked task")
        if score.completion_status != run.completion_status:
            raise ValueError("score and task-run completion_status differ")
    if len(runs) != manifest.expected_task_count or len(scores) != manifest.expected_task_count:
        raise ValueError("RunManifest expected task count mismatch")
    status_counts = Counter(row.completion_status for row in runs)
    if status_counts[CompletionStatus.COMPLETED] != manifest.completed_task_count:
        raise ValueError("RunManifest completed task count mismatch")
    if status_counts[CompletionStatus.FAILED] != manifest.failed_task_count:
        raise ValueError("RunManifest failed task count mismatch")
    if status_counts[CompletionStatus.NOT_SUPPORTED] != manifest.not_supported_task_count:
        raise ValueError("RunManifest not-supported task count mismatch")
    for score in scores:
        if score.schema_version != SCHEMA_VERSION or score.scorer_version != SCORER_VERSION:
            raise ValueError("ScoreRecord version is not current")
        _validate_score_support(score)
    for run in runs:
        if run.schema_version != SCHEMA_VERSION or run.runtime_record_version != RUNTIME_RECORD_VERSION:
            raise ValueError("TaskRunRecord version is not current")
    expected_rows: dict[str, int] = {}
    for ref in (*manifest.normalized_runtime_artifacts, *manifest.score_artifacts):
        assert ref.record_count is not None
        name = Path(ref.path).name
        if name in expected_rows:
            raise ValueError("runtime and score artifact basenames must be unique")
        expected_rows[name] = ref.record_count
    _validate_declared_summary(
        manifest.native_vs_extracted_field_summary,
        input_refs=(
            *manifest.raw_provider_response_artifacts,
            manifest.task_manifest,
            *task_manifest.task_file_paths_and_hashes,
            *task_manifest.source_manifest_paths_and_hashes,
            *manifest.raw_adapter_state_artifacts,
        ),
        output_refs=(
            *manifest.normalized_runtime_artifacts,
            *manifest.score_artifacts,
        ),
        expected_rows=expected_rows,
        manifest_path=manifest_path,
    )
    if manifest.capability_verification_artifact is not None:
        _verify_ref(manifest.capability_verification_artifact, manifest_path)
    return runs, scores


def _referenced_input(kind: str, input_path: Path, manifest: BaseModel, manifest_path: Path) -> None:
    if kind == "tasks":
        refs = manifest.task_file_paths_and_hashes
    elif kind == "task-runs":
        refs = manifest.normalized_runtime_artifacts
    elif kind == "scores":
        refs = manifest.score_artifacts
    else:
        return
    matches = [ref for ref in refs if _resolve_ref(ref, manifest_path) == input_path]
    if len(matches) != 1:
        raise ValueError("input artifact is not referenced exactly once by manifest")
    _verify_ref(matches[0], manifest_path)


def validate(kind: str, input_path: Path, manifest_path: Path) -> None:
    input_path = _require_regular_file(input_path, "input artifact")
    manifest_path = _require_regular_file(manifest_path, "manifest artifact")
    if kind in {"tasks", "task-manifest"}:
        task_manifest = _load_canonical_model(manifest_path, TaskManifest)
        assert isinstance(task_manifest, TaskManifest)
        _verify_current_task_manifest(task_manifest)
        tasks = _load_tasks_from_manifest(task_manifest, manifest_path)
        if kind == "tasks":
            _referenced_input(kind, input_path, task_manifest, manifest_path)
            direct = _load_canonical_jsonl(input_path, MemUpdateTask, "task_id")
            declared_ids = {task.task_id for task in tasks}
            if {task.task_id for task in direct} - declared_ids:
                raise ValueError("input tasks are absent from TaskManifest")
        elif input_path != manifest_path:
            raise ValueError("task-manifest input must be the supplied manifest")
        return

    run_manifest = _load_canonical_model(manifest_path, RunManifest)
    assert isinstance(run_manifest, RunManifest)
    _verify_current_run_manifest(run_manifest)
    runs, scores = _validate_run_graph(run_manifest, manifest_path)
    if kind == "task-runs":
        _referenced_input(kind, input_path, run_manifest, manifest_path)
        direct = _load_canonical_jsonl(input_path, TaskRunRecord, "task_id")
        if {row.task_id for row in direct} - {row.task_id for row in runs}:
            raise ValueError("input task-runs are absent from RunManifest")
    elif kind == "scores":
        _referenced_input(kind, input_path, run_manifest, manifest_path)
        direct = _load_canonical_jsonl(input_path, ScoreRecord, "task_id")
        if {row.task_id for row in direct} - {row.task_id for row in scores}:
            raise ValueError("input scores are absent from RunManifest")
    elif kind == "run-manifest" and input_path != manifest_path:
        raise ValueError("run-manifest input must be the supplied manifest")


def _report(kind: str | None, *, valid: bool, errors: list[Any]) -> dict[str, Any]:
    return {"errors": errors, "kind": kind, "valid": valid, "warnings": []}


def _print_report(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


class _ArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError("invalid command line")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Validate canonical vNext artifacts without mutation.",
        allow_abbrev=False,
    )
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except _ArgumentError:
        _print_report(
            _report(
                None,
                valid=False,
                errors=[{"code": "invalid_arguments", "type": "ArgumentError"}],
            )
        )
        return 2
    try:
        validate(args.kind, Path(args.input), Path(args.manifest))
    except Exception as exc:
        _print_report(
            _report(
                args.kind,
                valid=False,
                errors=[
                    {
                        "code": "validation_failed",
                        "type": type(exc).__name__,
                    }
                ],
            )
        )
        return 1
    _print_report(_report(args.kind, valid=True, errors=[]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
