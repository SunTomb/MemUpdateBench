from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Literal
import uuid

from pydantic import Field, field_validator, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.contracts.v3.adapter import AdapterInfoV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.external.artifacts import RawPayloadLicenseStatus
from mub.vnext.external.registry import _validate_portable_path
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.preparation.task12 import Task12DryRunPlanV1, Task12PreparationManifestV1
from mub.vnext.runtime.answer_model_v3 import OfflinePromptedAnswerModelV3
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3
from mub.vnext.runtime.task12_bundle_v3 import (
    Task12RunBundleV1,
    _artifact,
    _binding_for_slot,
    _ensure_output_root,
    _json_sha256,
    _load_capabilities,
    _prompt_expectations,
    _prompt_template_hash,
    _publish_bundle_artifacts,
    _qualified_decode_config,
    _read_artifact,
    _read_canonical_json,
    _read_core_tasks,
    _selected_answer_run,
    _task_manifest_for_view,
    _validate_existing_bundle_root,
    _validate_task12_run_bundle_v3,
    _write_new,
    load_task12_frozen_trajectories_v3,
    validate_task12_manifest_plan_v3,
)
from mub.vnext.runtime.task12_execution_v3 import (
    ContextAnnotation,
    ContextOrder,
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
    _TASK12_TEST_MODEL_TOKEN,
    read_task12_regular_file_v3,
    run_task12_cell_v3,
    task12_runtime_configuration_sha256_v3,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _single_component(value: str) -> str:
    validated = _validate_portable_path(value)
    if "/" in validated:
        raise ValueError("matrix bundle path components must be single path leaves")
    return validated


class Task12MatrixRunBundleRefV1(ImmutableContractModel):
    cell_id: str = Field(strict=True, min_length=1)
    answer_model_slot: Literal["answer_model_a", "answer_model_b"]
    bundle_leaf: str = Field(strict=True, min_length=1)
    output_leaf: str = Field(strict=True, min_length=1)
    task_manifest_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    task_view_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    run_config_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    authorization_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)

    @field_validator("bundle_leaf", "output_leaf")
    @classmethod
    def _validate_leaf(cls, value: str) -> str:
        return _single_component(value)


class Task12MatrixBundleManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task12-matrix-bundle.v1"] = (
        "memupdatebench.core-task12-matrix-bundle.v1"
    )
    preparation_manifest_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    plan_fingerprint_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    bundle_count: Literal[18] = 18
    run_bundles: tuple[Task12MatrixRunBundleRefV1, ...]

    @model_validator(mode="after")
    def _validate_complete_matrix(self):
        pairs = tuple(
            (bundle.cell_id, bundle.answer_model_slot)
            for bundle in self.run_bundles
        )
        if len(pairs) != 18 or len(set(pairs)) != 18:
            raise ValueError("Task 12 matrix bundle manifest must bind 18 unique runs")
        if len({bundle.bundle_leaf for bundle in self.run_bundles}) != 18:
            raise ValueError("Task 12 matrix bundle leaves must be unique")
        return self


class Task12MatrixRunRecordV1(ImmutableContractModel):
    cell_id: str = Field(strict=True, min_length=1)
    answer_model_slot: Literal["answer_model_a", "answer_model_b"]
    bundle_leaf: str = Field(strict=True, min_length=1)
    output_leaf: str = Field(strict=True, min_length=1)
    task_count: int = Field(strict=True, gt=0)
    score_count: int = Field(strict=True, gt=0)
    run_manifest_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    score_artifact_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)

    @field_validator("bundle_leaf", "output_leaf")
    @classmethod
    def _validate_leaf(cls, value: str) -> str:
        return _single_component(value)


class Task12MatrixRunSummaryV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.core-task12-matrix-run-summary.v1"] = (
        "memupdatebench.core-task12-matrix-run-summary.v1"
    )
    preparation_manifest_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    plan_fingerprint_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    matrix_bundle_manifest_sha256: str = Field(strict=True, pattern=_SHA256_PATTERN)
    run_count: Literal[18] = 18
    total_task_rows: int = Field(strict=True, gt=0)
    total_score_rows: int = Field(strict=True, gt=0)
    completed_runs: tuple[Task12MatrixRunRecordV1, ...]

    @model_validator(mode="after")
    def _validate_complete_summary(self):
        pairs = tuple(
            (run.cell_id, run.answer_model_slot)
            for run in self.completed_runs
        )
        if len(pairs) != 18 or len(set(pairs)) != 18:
            raise ValueError("Task 12 matrix run summary must bind 18 unique runs")
        if self.total_task_rows != sum(run.task_count for run in self.completed_runs):
            raise ValueError("Task 12 matrix task-row total must match completed runs")
        if self.total_score_rows != sum(run.score_count for run in self.completed_runs):
            raise ValueError("Task 12 matrix score-row total must match completed runs")
        return self


@dataclass(frozen=True)
class Task12MatrixBundleV1:
    matrix_root: Path
    matrix_manifest_path: Path
    bundles: tuple[Task12RunBundleV1, ...]
    manifest: Task12MatrixBundleManifestV1


@dataclass(frozen=True)
class Task12MatrixRunResultV1:
    matrix_root: Path
    summary_path: Path
    summary: Task12MatrixRunSummaryV1


def _bundle_leaf(cell_id: str, answer_model_slot: str) -> str:
    return f"{cell_id}__{answer_model_slot}"


def _build_loaded_bundle(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core: Path,
    evidence: Path,
    repository: Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    source_manifest: TaskManifestV3,
    tasks_by_id: dict[str, object],
    output_root: Path,
    cell_id: str,
    answer_model_slot: Literal["answer_model_a", "answer_model_b"],
    output_leaf: str,
) -> Task12RunBundleV1:
    selected_run = _selected_answer_run(
        manifest=manifest,
        plan=plan,
        cell_id=cell_id,
        answer_model_slot=answer_model_slot,
    )
    selected_cell = next(
        cell
        for cell in manifest.semantic_matrix.intervention_cells
        if cell.cell_id == selected_run.cell_id
    )
    binding = _binding_for_slot(manifest, answer_model_slot)
    missing = tuple(
        task_id for task_id in selected_cell.task_ids if task_id not in tasks_by_id
    )
    if missing:
        raise ValueError("Core task artifact is missing selected Task 12 IDs")
    tasks = tuple(tasks_by_id[task_id] for task_id in selected_cell.task_ids)
    task_view_raw = b"".join(canonical_json_bytes(task) + b"\n" for task in tasks)
    task_view_ref = _artifact(
        "tasks.jsonl",
        task_view_raw,
        media_type="application/x-ndjson",
        record_count=len(tasks),
    )
    task_manifest = _task_manifest_for_view(
        source_manifest=source_manifest,
        task_view_ref=task_view_ref,
        selected_tasks=tasks,
    )
    task_manifest_raw = canonical_json_bytes(task_manifest)
    task_manifest_ref = _artifact(
        "task_manifest.json",
        task_manifest_raw,
        media_type="application/json",
        record_count=1,
    )
    adapter_info_raw = _read_artifact(
        root=evidence,
        location=selected_cell.adapter_info,
    )
    adapter_info = AdapterInfoV3.model_validate_json(adapter_info_raw)
    if canonical_json_bytes(adapter_info) != adapter_info_raw:
        raise ValueError("Task 12 adapter info is not canonical")
    _read_artifact(root=evidence, location=selected_cell.adapter_configuration)
    _read_artifact(root=evidence, location=selected_cell.retrieval.artifact)
    adapter_capabilities = _load_capabilities(
        evidence,
        selected_cell.capability_verification,
    )
    if answer_model_slot == "answer_model_b":
        _read_artifact(
            root=evidence,
            location=manifest.task11_mistral_provenance,
        )
    decoding_config = _qualified_decode_config(evidence_root=evidence, binding=binding)
    runtime_hash = task12_runtime_configuration_sha256_v3(
        RuntimeConfigV3(
            run_id=f"task12-{cell_id}-{answer_model_slot}",
            retrieval_policy=selected_cell.retrieval.configuration.retrieval_policy,
            answer_mode="slot_prompt",
            retrieval_k=selected_cell.retrieval.configuration.retrieval_k,
            capture_snapshots=False,
        ),
        context_order=selected_cell.context_intervention.context_order,
        context_annotation=selected_cell.context_intervention.context_annotation,
    )
    evaluation_hash = _json_sha256({
        "cell_id": cell_id,
        "answer_model_slot": answer_model_slot,
        "canonical_run_binding_sha256": selected_run.canonical_run_binding_sha256,
        "task_manifest_sha256": task_manifest_ref.sha256,
    })
    run_config = ExternalRunConfigV1(
        run_id=f"task12-{cell_id}-{answer_model_slot}",
        code_revision=runtime_code_binding.code_revision,
        dirty_state=False,
        source_task_manifest_ref=task_manifest_ref,
        task_view_ref=task_view_ref,
        adapter_configuration_ref=selected_cell.adapter_configuration.artifact,
        capability_verification_ref=selected_cell.capability_verification.artifact,
        model_provenance_ref=(
            manifest.task11_mistral_provenance.artifact
            if answer_model_slot == "answer_model_b"
            else binding.qualification_report.artifact
        ),
        package_provenance_ref=binding.qualification_report.artifact,
        environment_lock_ref=binding.qualification_report.artifact,
        adapter_info=adapter_info,
        adapter_capabilities=adapter_capabilities,
        retrieval_policy=manifest.scientific_design.retrieval_policy,
        answer_mode="slot_prompt",
        runtime_configuration_hash=runtime_hash,
        evaluation_configuration_hash=evaluation_hash,
        model_name=binding.model_id,
        provider="offline_hf",
        model_revision=binding.revision,
        answer_model_slot=answer_model_slot,
        prompt_config={
            "prompt_protocol_version": "prompted-answer-v1",
            "renderer_version": "visible-context-v1",
            "template_hash": _prompt_template_hash(),
            "typed_output_format": "answer-envelope-v1",
        },
        decoding_config=decoding_config,
        seed_information={"seed": decoding_config["seed"]},
        environment_summary={
            "hf_hub_offline": True,
            "transformers_offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "runtime_code_tree_sha256": runtime_code_binding.code_tree_sha256,
        },
        package_summary={},
        action_parser_version="core-visible-action-parser-v1",
        answer_parser_version="core-typed-answer-parser-v1",
        memory_entry_extractor_version="core-entry-extractor-v1",
        object_value_extractor_config_hash="0" * 64,
        redaction_policy_version="none-v1",
        normalized_license_status=RawPayloadLicenseStatus.REDISTRIBUTABLE,
        repetition_index=0,
        repetition_count=1,
        expected_task_ids=tuple(task.task_id for task in tasks),
        task_record_hashes={task.task_id: sha256_model(task) for task in tasks},
        prompted_task_expectations=_prompt_expectations(tasks),
    )
    run_config_raw = canonical_json_bytes(run_config)
    authorization = Task12ExecutionAuthorizationV1(
        preparation_manifest_sha256=sha256_model(manifest),
        plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
        runtime_code_binding=runtime_code_binding,
        cell_id=cell_id,
        answer_model_slot=answer_model_slot,
        cell_binding_sha256=selected_run.cell_binding_sha256,
        answer_model_binding_sha256=selected_run.answer_model_binding_sha256,
        canonical_run_binding_sha256=selected_run.canonical_run_binding_sha256,
        task_manifest_sha256=hashlib.sha256(task_manifest_raw).hexdigest(),
        task_view_sha256=hashlib.sha256(task_view_raw).hexdigest(),
        run_config_sha256=hashlib.sha256(run_config_raw).hexdigest(),
        output_leaf=output_leaf,
    )
    root = _ensure_output_root(
        output_root=output_root,
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
    )
    task_view_path = root / "tasks.jsonl"
    task_manifest_path = root / "task_manifest.json"
    run_config_path = root / "run_config.json"
    authorization_path = root / "authorization.json"
    _publish_bundle_artifacts(
        root,
        {
            task_view_path: task_view_raw,
            task_manifest_path: task_manifest_raw,
            run_config_path: run_config_raw,
            authorization_path: canonical_json_bytes(authorization),
        },
    )
    return Task12RunBundleV1(
        cell_id=cell_id,
        answer_model_slot=answer_model_slot,
        bundle_root=root,
        task_manifest_path=task_manifest_path,
        task_view_path=task_view_path,
        run_config_path=run_config_path,
        authorization_path=authorization_path,
        execution_output_root=root / output_leaf,
        tasks=tasks,
    )


def task12_cell_runtime_v3(
    manifest: Task12PreparationManifestV1,
    cell_id: str,
) -> tuple[ContextOrder, ContextAnnotation, int]:
    matches = tuple(
        cell
        for cell in manifest.semantic_matrix.intervention_cells
        if cell.cell_id == cell_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown Task 12 cell ID: {cell_id}")
    cell = matches[0]
    return (
        cell.context_intervention.context_order,
        cell.context_intervention.context_annotation,
        cell.retrieval.configuration.retrieval_k,
    )


def _execute_task12_matrix_bundles_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    matrix_bundle_manifest: Task12MatrixBundleManifestV1,
    matrix_root: str | Path,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    adapter_factory,
    prompted_answer_models: Mapping[Literal["answer_model_a", "answer_model_b"], Any],
    resume: bool = False,
    _test_model_token: object | None = None,
) -> Task12MatrixRunResultV1:
    core = Path(core_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    validate_task12_manifest_plan_v3(manifest, plan)
    root = _validate_existing_bundle_root(
        bundle_root=matrix_root,
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
    )
    matrix_manifest_raw = read_task12_regular_file_v3(
        root / "matrix_bundle_manifest.json"
    )
    if hashlib.sha256(matrix_manifest_raw).hexdigest() != sha256_model(matrix_bundle_manifest):
        raise ValueError("Task 12 matrix bundle manifest file differs from supplied model")
    if matrix_bundle_manifest.preparation_manifest_sha256 != sha256_model(manifest):
        raise ValueError("Task 12 matrix bundle manifest is not bound to preparation manifest")
    if matrix_bundle_manifest.plan_fingerprint_sha256 != plan.plan_fingerprint_sha256:
        raise ValueError("Task 12 matrix bundle manifest is not bound to dry-run plan")
    observed_pairs = tuple(
        (ref.cell_id, ref.answer_model_slot)
        for ref in matrix_bundle_manifest.run_bundles
    )
    admitted_pairs = tuple(
        (run.cell_id, run.answer_model_slot)
        for run in plan.admitted_answer_runs
    )
    if observed_pairs != admitted_pairs:
        raise ValueError("Task 12 matrix bundles differ from admitted answer runs")
    expected_slots = {"answer_model_a", "answer_model_b"}
    if set(prompted_answer_models) != expected_slots:
        raise ValueError("Task 12 matrix execution requires both answer-model slots")
    for slot_id, model in prompted_answer_models.items():
        binding = _binding_for_slot(manifest, slot_id)
        if _test_model_token is _TASK12_TEST_MODEL_TOKEN:
            if getattr(model, "slot_id", None) != slot_id:
                raise ValueError("Task 12 test model object is bound to the wrong slot")
            continue
        if type(model) is not OfflinePromptedAnswerModelV3:
            raise TypeError(
                "Task 12 production matrix requires OfflinePromptedAnswerModelV3"
            )
        if (
            model.slot.slot_id != binding.slot_id
            or model.slot.model_id != binding.model_id
            or model.slot.revision != binding.revision
            or model.slot.license_id != binding.license_id
            or model.slot.tree_manifest_sha256 != binding.tree_manifest_sha256
            or sha256_model(model.decoding) != binding.decoding_config_sha256
        ):
            raise ValueError("Task 12 answer model differs from frozen slot binding")
    source_manifest_raw = _read_artifact(
        root=core,
        location=manifest.task_manifest,
    )
    source_manifest = TaskManifestV3.model_validate_json(source_manifest_raw)
    if (
        canonical_json_bytes(source_manifest) != source_manifest_raw
        or sha256_model(source_manifest) != plan.core_task_manifest_sha256
    ):
        raise ValueError("source task manifest differs from admitted plan")
    source_tasks_raw = _read_artifact(root=core, location=manifest.tasks)
    if hashlib.sha256(source_tasks_raw).hexdigest() != plan.core_tasks_sha256:
        raise ValueError("Core task artifact differs from admitted plan")
    source_tasks_by_id = _read_core_tasks(source_tasks_raw)
    matrix_tasks = tuple(
        source_tasks_by_id[task_id]
        for task_id in manifest.semantic_matrix.task_scope.task_ids
    )
    frozen_trajectories = load_task12_frozen_trajectories_v3(
        manifest=manifest,
        evidence_root=evidence,
        tasks=matrix_tasks,
    )

    loaded_slots: list[Any] = []
    completed: list[Task12MatrixRunRecordV1] = []
    models_loaded = False
    try:
        for bundle_ref in matrix_bundle_manifest.run_bundles:
            bundle_root = root / bundle_ref.bundle_leaf
            if not bundle_root.is_dir() or bundle_root.is_symlink():
                raise ValueError("Task 12 matrix bundle root must be an existing real directory")
            authorization_sha256 = hashlib.sha256(
                read_task12_regular_file_v3(
                    bundle_root / "authorization.json"
                )
            ).hexdigest()
            if authorization_sha256 != bundle_ref.authorization_sha256:
                raise ValueError("Task 12 matrix authorization hash mismatch")

            bundle = _validate_task12_run_bundle_v3(
                manifest=manifest,
                plan=plan,
                core_root=core,
                evidence_root=evidence,
                repository_root=repository,
                runtime_code_binding=runtime_code_binding,
                bundle_root=bundle_root,
                source_manifest=source_manifest,
                source_tasks_by_id=source_tasks_by_id,
                frozen_trajectories=frozen_trajectories,
            )
            if (
                bundle.authorization.cell_id != bundle_ref.cell_id
                or bundle.authorization.answer_model_slot
                != bundle_ref.answer_model_slot
                or bundle.authorization.output_leaf != bundle_ref.output_leaf
                or bundle.authorization.task_manifest_sha256
                != bundle_ref.task_manifest_sha256
                or bundle.authorization.task_view_sha256
                != bundle_ref.task_view_sha256
                or bundle.authorization.run_config_sha256
                != bundle_ref.run_config_sha256
            ):
                raise ValueError(
                    "Task 12 matrix authorization does not match bundle reference"
                )
            task_manifest = bundle.task_manifest
            run_config = bundle.run_configuration
            tasks = bundle.tasks
            context_order, context_annotation, retrieval_k = task12_cell_runtime_v3(
                manifest,
                bundle_ref.cell_id,
            )
            runtime_config = RuntimeConfigV3(
                run_id=run_config.run_id,
                retrieval_policy=run_config.retrieval_policy,
                answer_mode=run_config.answer_mode,
                retrieval_k=retrieval_k,
                capture_snapshots=False,
            )
            if (
                not models_loaded
                and not (
                    resume
                    and (bundle.execution_output_root / "run_manifest.json").is_file()
                )
            ):
                for slot in ("answer_model_a", "answer_model_b"):
                    model = prompted_answer_models[slot]
                    loaded_slots.append(model)
                    model.load()
                models_loaded = True
            run_manifest, rows, scores, receipt = run_task12_cell_v3(
                tasks,
                adapter_factory=adapter_factory,
                run_configuration=run_config,
                runtime_config=runtime_config,
                prompted_answer_model=prompted_answer_models[bundle_ref.answer_model_slot],
                context_order=context_order,
                context_annotation=context_annotation,
                frozen_trajectories=bundle.frozen_trajectories,
                output_root=bundle.execution_output_root,
                task_manifest=task_manifest,
                run_manifest_artifact=None,
                task_artifact=run_config.task_view_ref,
                authenticated_task_manifest_sha256=(
                    bundle.authorization.task_manifest_sha256
                ),
                resume=resume,
                _test_model_token=_test_model_token,
            )
            if len(rows) != len(tasks) or len(scores) != len(tasks):
                raise ValueError("Task 12 matrix run did not complete all task and score rows")
            run_manifest_sha256 = hashlib.sha256(canonical_json_bytes(run_manifest)).hexdigest()
            completed.append(
                Task12MatrixRunRecordV1(
                    cell_id=bundle_ref.cell_id,
                    answer_model_slot=bundle_ref.answer_model_slot,
                    bundle_leaf=bundle_ref.bundle_leaf,
                    output_leaf=bundle_ref.output_leaf,
                    task_count=len(rows),
                    score_count=len(scores),
                    run_manifest_sha256=run_manifest_sha256,
                    score_artifact_sha256=receipt[
                        "score_artifact_sha256"
                    ],
                )
            )
    finally:
        for model in reversed(loaded_slots):
            model.close()

    summary = Task12MatrixRunSummaryV1(
        preparation_manifest_sha256=sha256_model(manifest),
        plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
        matrix_bundle_manifest_sha256=sha256_model(matrix_bundle_manifest),
        total_task_rows=sum(run.task_count for run in completed),
        total_score_rows=sum(run.score_count for run in completed),
        completed_runs=tuple(completed),
    )
    summary_path = root / "matrix_run_summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        if not resume:
            raise FileExistsError("Task 12 matrix run summary already exists")
        summary_raw = read_task12_regular_file_v3(summary_path)
        existing_summary = Task12MatrixRunSummaryV1.model_validate_json(summary_raw)
        if (
            canonical_json_bytes(existing_summary) != summary_raw
            or existing_summary != summary
        ):
            raise ValueError("existing Task 12 matrix summary differs from resumed runs")
        summary = existing_summary
    else:
        publish_files_atomically(
            {summary_path: canonical_json_bytes(summary)},
            overwrite=False,
        )
    return Task12MatrixRunResultV1(
        matrix_root=root,
        summary_path=summary_path,
        summary=summary,
    )

def execute_task12_matrix_bundles_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    matrix_bundle_manifest: Task12MatrixBundleManifestV1,
    matrix_root: str | Path,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    adapter_factory,
    prompted_answer_models: Mapping[
        Literal["answer_model_a", "answer_model_b"],
        OfflinePromptedAnswerModelV3,
    ],
    resume: bool = False,
) -> Task12MatrixRunResultV1:
    return _execute_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        matrix_bundle_manifest=matrix_bundle_manifest,
        matrix_root=matrix_root,
        core_root=core_root,
        evidence_root=evidence_root,
        repository_root=repository_root,
        runtime_code_binding=runtime_code_binding,
        adapter_factory=adapter_factory,
        prompted_answer_models=prompted_answer_models,
        resume=resume,
    )


def _execute_task12_matrix_bundles_for_test_v3(
    **kwargs,
) -> Task12MatrixRunResultV1:
    return _execute_task12_matrix_bundles_v3(
        **kwargs,
        _test_model_token=_TASK12_TEST_MODEL_TOKEN,
    )


def build_task12_matrix_bundles_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    output_root: str | Path,
) -> Task12MatrixBundleV1:
    core = Path(core_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    validate_task12_manifest_plan_v3(manifest, plan)
    if plan.core_task_manifest_sha256 != manifest.task_manifest.artifact.sha256:
        raise ValueError("plan task-manifest hash does not match preparation manifest")
    if plan.core_hard_suite_sha256 != manifest.core_hard_suite.artifact.sha256:
        raise ValueError("plan hard-suite hash does not match preparation manifest")
    if plan.core_tasks_sha256 != manifest.tasks.artifact.sha256:
        raise ValueError("plan task artifact hash does not match preparation manifest")
    if len(plan.admitted_answer_runs) != 18:
        raise ValueError("Task 12 matrix requires exactly 18 admitted answer runs")
    source_manifest_raw = _read_artifact(
        root=core,
        location=manifest.task_manifest,
    )
    source_manifest = TaskManifestV3.model_validate_json(source_manifest_raw)
    if canonical_json_bytes(source_manifest) != source_manifest_raw:
        raise ValueError("source task manifest is not canonical")
    if sha256_model(source_manifest) != plan.core_task_manifest_sha256:
        raise ValueError("source task manifest does not match the admitted plan")
    source_tasks_raw = _read_artifact(root=core, location=manifest.tasks)
    tasks_by_id = _read_core_tasks(source_tasks_raw)
    final_root = _ensure_output_root(
        output_root=Path(output_root),
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
        create=False,
    )
    staging_root = _ensure_output_root(
        output_root=final_root.with_name(
            f".{final_root.name}.task12-stage-{uuid.uuid4().hex}"
        ),
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
    )
    try:
        staged_bundles = []
        refs = []
        for run in plan.admitted_answer_runs:
            leaf = _bundle_leaf(run.cell_id, run.answer_model_slot)
            bundle = _build_loaded_bundle(
                manifest=manifest,
                plan=plan,
                core=core,
                evidence=evidence,
                repository=repository,
                runtime_code_binding=runtime_code_binding,
                source_manifest=source_manifest,
                tasks_by_id=tasks_by_id,
                output_root=staging_root / leaf,
                cell_id=run.cell_id,
                answer_model_slot=run.answer_model_slot,
                output_leaf="run",
            )
            staged_bundles.append(bundle)
            refs.append(
                Task12MatrixRunBundleRefV1(
                    cell_id=bundle.cell_id,
                    answer_model_slot=bundle.answer_model_slot,
                    bundle_leaf=leaf,
                    output_leaf="run",
                    task_manifest_sha256=hashlib.sha256(
                        read_task12_regular_file_v3(bundle.task_manifest_path)
                    ).hexdigest(),
                    task_view_sha256=hashlib.sha256(
                        read_task12_regular_file_v3(bundle.task_view_path)
                    ).hexdigest(),
                    run_config_sha256=hashlib.sha256(
                        read_task12_regular_file_v3(bundle.run_config_path)
                    ).hexdigest(),
                    authorization_sha256=hashlib.sha256(
                        read_task12_regular_file_v3(bundle.authorization_path)
                    ).hexdigest(),
                )
            )
        matrix_manifest = Task12MatrixBundleManifestV1(
            preparation_manifest_sha256=sha256_model(manifest),
            plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
            run_bundles=tuple(refs),
        )
        _write_new(
            staging_root / "matrix_bundle_manifest.json",
            canonical_json_bytes(matrix_manifest),
        )
        if os.path.lexists(final_root):
            raise FileExistsError("Task 12 matrix output appeared during preparation")
        os.rename(staging_root, final_root)
    except BaseException:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        raise
    bundles = tuple(
        Task12RunBundleV1(
            cell_id=bundle.cell_id,
            answer_model_slot=bundle.answer_model_slot,
            bundle_root=final_root / bundle.bundle_root.name,
            task_manifest_path=(
                final_root / bundle.bundle_root.name / "task_manifest.json"
            ),
            task_view_path=final_root / bundle.bundle_root.name / "tasks.jsonl",
            run_config_path=final_root / bundle.bundle_root.name / "run_config.json",
            authorization_path=(
                final_root / bundle.bundle_root.name / "authorization.json"
            ),
            execution_output_root=(
                final_root / bundle.bundle_root.name / bundle.execution_output_root.name
            ),
            tasks=bundle.tasks,
        )
        for bundle in staged_bundles
    )
    return Task12MatrixBundleV1(
        matrix_root=final_root,
        matrix_manifest_path=final_root / "matrix_bundle_manifest.json",
        bundles=bundles,
        manifest=matrix_manifest,
    )


__all__ = [
    "Task12MatrixBundleManifestV1",
    "Task12MatrixBundleV1",
    "Task12MatrixRunBundleRefV1",
    "Task12MatrixRunRecordV1",
    "Task12MatrixRunResultV1",
    "Task12MatrixRunSummaryV1",
    "build_task12_matrix_bundles_v3",
    "execute_task12_matrix_bundles_v3",
    "task12_cell_runtime_v3",
]
