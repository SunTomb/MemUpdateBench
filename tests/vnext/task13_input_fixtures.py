from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.enums import CompletionStatus, SupportReason
from mub.vnext.contracts.v3.adapter import AdapterActionResultV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.runtime import (
    AnswerPredictionV3,
    ParserExtractorProvenanceV3,
    RetrievalTraceV3,
    TaskRunRecordV3,
)
from mub.vnext.contracts.v3.score import (
    CORE_METRIC_FIELD_PATHS,
    MetricFieldSupport,
    ScoreRecordV3,
)
from mub.vnext.generation.artifacts import InMemoryPilotArtifact
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.preparation.task12 import (
    Task12AdmittedAnswerRunV1,
    Task12AdmittedCellV1,
    Task12DryRunPlanV1,
    _canonical_answer_model_binding_hash,
    _canonical_answer_run_binding_hash,
    _canonical_cell_binding_hash,
)
from mub.vnext.runtime.task12_execution_v3 import (
    Task12RuntimeCodeBindingV1,
    persist_task12_rows_v3,
    persist_task12_scores_v3,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    Task12MatrixRunRecordV1,
    Task12MatrixRunSummaryV1,
    build_task12_matrix_bundles_v3,
)
from tests.vnext.task12_fixtures import build_task12_inputs, build_task12_manifest


_FAMILIES = {
    "repeated_same_slot_update",
    "current_historical_query",
    "long_horizon_memory_synthesis",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: str, raw: bytes, media_type: str, record_count: int) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        media_type=media_type,
        record_count=record_count,
    )


def _compact_bundle(root: Path):
    config = load_core_config(root / "configs" / "vnext" / "core.yaml")
    snapshot = compile_core_snapshot(
        config,
        cores_per_family=20,
        code_revision="8" * 40,
    )
    tasks = tuple(
        sorted(
            (task for task in snapshot.tasks if task.task_family in _FAMILIES),
            key=lambda task: task.task_id.encode("utf-8"),
        )
    )
    if len(tasks) != 240:
        raise AssertionError("compact Task 13 fixture must contain 80 A/F/G tasks each")
    tasks_raw = b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)
    task_ref = _artifact(
        "tasks.jsonl",
        tasks_raw,
        "application/x-ndjson",
        len(tasks),
    )
    split_counts = Counter(task.metadata.split.value for task in tasks)
    family_counts = Counter(
        f"{task.task_family}/{task.difficulty.value}" for task in tasks
    )
    core_counts = Counter(task.metadata.split_key.semantic_core_id for task in tasks)
    task_manifest = TaskManifestV3(
        data_release_id="core-task13-compact-fixture",
        split_policy_version="core-split-v1",
        compiler_versions={"fixture": "v1"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts=dict(split_counts),
        family_difficulty_counts=dict(family_counts),
        semantic_core_counts=dict(core_counts),
        task_file_paths_and_hashes=(task_ref,),
        leakage_check_summary={"status": "fixture"},
        human_audit_artifacts=(),
        created_at="2026-08-17T00:00:00Z",
        code_revision="8" * 40,
        task_record_hashes={task.task_id: sha256_model(task) for task in tasks},
    )
    task_manifest_raw = canonical_json_bytes(task_manifest)
    hard_raw = json.dumps(
        {"task_ids": [task.task_id for task in tasks]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifacts = (
        InMemoryPilotArtifact(
            "tasks.jsonl",
            tasks_raw,
            "application/x-ndjson",
            len(tasks),
        ),
        InMemoryPilotArtifact(
            "task_manifest.json",
            task_manifest_raw,
            "application/json",
            1,
        ),
        InMemoryPilotArtifact(
            "core-hard-v1.json",
            hard_raw,
            "application/json",
            1,
        ),
    )
    return SimpleNamespace(
        artifacts=artifacts,
        hard_suite=SimpleNamespace(task_ids=tuple(task.task_id for task in tasks)),
        snapshot=SimpleNamespace(tasks=tasks),
    )


def _compact_inputs(root: Path, repository_root: Path) -> dict[str, object]:
    import tests.vnext.task12_fixtures as fixture_module

    bundle = _compact_bundle(repository_root)
    original_compile = fixture_module.compile_core_snapshot
    original_bundle = fixture_module.build_core_artifact_bundle
    fixture_module.compile_core_snapshot = lambda *args, **kwargs: object()
    fixture_module.build_core_artifact_bundle = lambda *args, **kwargs: bundle
    try:
        return build_task12_inputs(root)
    finally:
        fixture_module.compile_core_snapshot = original_compile
        fixture_module.build_core_artifact_bundle = original_bundle


def _plan_for_manifest(manifest) -> Task12DryRunPlanV1:
    cells = tuple(
        Task12AdmittedCellV1(
            cell_id=cell.cell_id,
            scope_id=cell.scope_id,
            canonical_binding_sha256=_canonical_cell_binding_hash(
                cell=cell,
                scope=manifest.semantic_matrix.task_scope,
                raw_append_intervention=manifest.semantic_matrix.raw_append_intervention,
            ),
        )
        for cell in manifest.semantic_matrix.intervention_cells
    )
    answer_hashes = tuple(
        _canonical_answer_model_binding_hash(binding)
        for binding in manifest.answer_models
    )
    cell_hashes = {cell.cell_id: cell.canonical_binding_sha256 for cell in cells}
    slot_hashes = dict(zip(manifest.scientific_design.answer_model_slots, answer_hashes))
    runs = tuple(
        Task12AdmittedAnswerRunV1(
            cell_id=cell.cell_id,
            answer_model_slot=slot,
            cell_binding_sha256=cell_hashes[cell.cell_id],
            answer_model_binding_sha256=slot_hashes[slot],
            canonical_run_binding_sha256=_canonical_answer_run_binding_hash(
                cell_binding_sha256=cell_hashes[cell.cell_id],
                answer_model_binding_sha256=slot_hashes[slot],
            ),
        )
        for cell in cells
        for slot in manifest.scientific_design.answer_model_slots
    )
    return Task12DryRunPlanV1(
        run_id=manifest.run_id,
        plan_fingerprint_sha256="1" * 64,
        core_task_manifest_sha256=manifest.task_manifest.artifact.sha256,
        core_hard_suite_sha256=manifest.core_hard_suite.artifact.sha256,
        core_tasks_sha256=manifest.tasks.artifact.sha256,
        scientific_design_sha256=sha256_model(manifest.scientific_design),
        semantic_matrix_sha256=sha256_model(manifest.semantic_matrix),
        main_manager_policy_sha256=sha256_model(manifest.main_manager_policy),
        answer_model_slots=manifest.scientific_design.answer_model_slots,
        answer_model_binding_sha256=answer_hashes,
        admitted_cells=cells,
        admitted_answer_runs=runs,
        hard_source_task_count=240,
        hard_source_task_selection_sha256="2" * 64,
        matrix_task_count=80,
        matrix_task_selection_sha256="3" * 64,
        main_test_task_count=2400,
        main_test_task_selection_sha256="4" * 64,
        output_leaf="task12-dry-run",
        code_revision="c" * 40,
        code_tree_sha256="d" * 64,
    )


def _prompted_row(task, run_config) -> TaskRunRecordV3:
    actions = tuple(
        AdapterActionResultV3(
            event_id=action.event_id,
            requested_action={"operation": "NOOP"},
            effective_action={"operation": "NOOP"},
            execution_status="executed",
        ).to_parsed_manager_action(
            action_id=action.action_id,
            raw_output="NOOP",
            format_valid=True,
            fallback_used=False,
        )
        for action in task.actions
    )
    traces = tuple(
        RetrievalTraceV3(query_id=query.query_id, prompt_hash="a" * 64)
        for query in task.queries
    )
    predictions = tuple(
        AnswerPredictionV3(
            query_id=query.query_id,
            raw_output='{"disposition":"answered","answer":"fixture"}',
            parsed_answer="fixture",
            format_valid=True,
        )
        for query in task.queries
    )
    return TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id=run_config.adapter_info.adapter_id,
        run_id=run_config.run_id,
        parsed_actions=actions,
        retrieval_traces=traces,
        answer_predictions=predictions,
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version=run_config.action_parser_version,
            answer_parser_version=run_config.answer_parser_version,
            memory_entry_extractor_version=run_config.memory_entry_extractor_version,
            object_value_extractor_config_hash=(
                run_config.object_value_extractor_config_hash
            ),
            redaction_policy_version=run_config.redaction_policy_version,
        ),
        completion_status=CompletionStatus.COMPLETED,
    )


def _scores(tasks, run_config) -> tuple[ScoreRecordV3, ...]:
    support = {
        path: MetricFieldSupport(
            reason=SupportReason.NOT_SUPPORTED,
            null_policy="emit_null",
        )
        for path in CORE_METRIC_FIELD_PATHS
    }
    return tuple(
        ScoreRecordV3.empty(
            task_id=task.task_id,
            run_id=run_config.run_id,
            adapter_id=run_config.adapter_info.adapter_id,
            task_family=task.task_family,
            difficulty=task.difficulty,
            completion_status=CompletionStatus.COMPLETED,
            supported_metric_fields=support,
        )
        for task in tasks
    )


def build_compact_authenticated_task13_fixture(
    root: Path,
    repository_root: Path,
    runtime: Task12RuntimeCodeBindingV1,
) -> dict[str, object]:
    inputs = _compact_inputs(root, repository_root)
    manifest = build_task12_manifest(inputs)
    plan = _plan_for_manifest(manifest)
    preparation_root = root / "preparation"
    preparation_root.mkdir()
    preparation_manifest_path = preparation_root / "task12_preparation_manifest.json"
    preparation_manifest_path.write_bytes(canonical_json_bytes(manifest))
    plan_path = preparation_root / "dry_run_plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")
    matrix = build_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=repository_root,
        runtime_code_binding=runtime,
        output_root=root / "matrix",
    )
    completed = []
    bundle_by_pair = {
        (bundle.cell_id, bundle.answer_model_slot): bundle
        for bundle in matrix.bundles
    }
    for ref in matrix.manifest.run_bundles:
        bundle = bundle_by_pair[(ref.cell_id, ref.answer_model_slot)]
        from mub.vnext.runtime.run_v3 import ExternalRunConfigV1

        configuration = ExternalRunConfigV1.model_validate_json(
            bundle.run_config_path.read_bytes()
        )
        rows = tuple(_prompted_row(task, configuration) for task in bundle.tasks)
        run_manifest = persist_task12_rows_v3(
            bundle.execution_output_root,
            configuration,
            rows,
        )
        run_manifest_sha256 = hashlib.sha256(
            canonical_json_bytes(run_manifest)
        ).hexdigest()
        receipt = persist_task12_scores_v3(
            bundle.execution_output_root / "scores",
            _scores(bundle.tasks, configuration),
            run_manifest_sha256=run_manifest_sha256,
            task_manifest_sha256=ref.task_manifest_sha256,
        )
        completed.append(
            Task12MatrixRunRecordV1(
                cell_id=ref.cell_id,
                answer_model_slot=ref.answer_model_slot,
                bundle_leaf=ref.bundle_leaf,
                output_leaf=ref.output_leaf,
                task_count=80,
                score_count=80,
                run_manifest_sha256=run_manifest_sha256,
                score_artifact_sha256=receipt["score_artifact_sha256"],
            )
        )
    summary = Task12MatrixRunSummaryV1(
        preparation_manifest_sha256=sha256_model(manifest),
        plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
        matrix_bundle_manifest_sha256=_sha256(matrix.matrix_manifest_path),
        total_task_rows=1440,
        total_score_rows=1440,
        completed_runs=tuple(completed),
    )
    summary_path = matrix.matrix_root / "matrix_run_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))
    audit = {
        "status": "verified",
        "runtime_code_binding": runtime.model_dump(mode="json"),
        "matrix_bundle_manifest_sha256": _sha256(matrix.matrix_manifest_path),
        "matrix_summary_sha256": _sha256(summary_path),
        "counts": {
            "run_count": 18,
            "total_task_rows": 1440,
            "total_score_rows": 1440,
            "failed": 0,
            "partial": 0,
            "semantic_multiset_mismatches": 0,
        },
    }
    audit_path = root / "logs" / "matrix_integrity_audit.json"
    audit_path.parent.mkdir()
    audit_path.write_bytes(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    expected_hashes = {
        "preparation_manifest": _sha256(preparation_manifest_path),
        "plan": _sha256(plan_path),
        "matrix_manifest": _sha256(matrix.matrix_manifest_path),
        "matrix_summary": _sha256(summary_path),
        "integrity_audit": _sha256(audit_path),
    }
    return {
        "root": root,
        "inputs": inputs,
        "manifest": manifest,
        "plan": plan,
        "preparation_manifest_path": preparation_manifest_path,
        "plan_path": plan_path,
        "matrix": matrix,
        "matrix_manifest_path": matrix.matrix_manifest_path,
        "summary_path": summary_path,
        "audit_path": audit_path,
        "expected_hashes": expected_hashes,
    }
