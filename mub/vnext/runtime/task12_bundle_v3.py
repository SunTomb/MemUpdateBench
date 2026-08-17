from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.artifacts import RawPayloadLicenseStatus, assert_no_reparse_components
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.io.atomic import publish_files_atomically
from mub.vnext.preparation.task12 import (
    RawAppendTrajectoryV1,
    Task11AnswerModelBindingV1,
    Task11QualificationReportV1,
    Task12AdmittedAnswerRunV1,
    Task12AdmittedCellV1,
    Task12CapabilityVerificationV1,
    Task12DryRunPlanV1,
    Task12PreparationManifestV1,
    _canonical_answer_model_binding_hash,
    _canonical_answer_run_binding_hash,
    _canonical_cell_binding_hash,
    _read_artifact,
    _read_raw_append_trajectories,
)
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1, PromptedTaskExpectationV1
from mub.vnext.runtime.task12_execution_v3 import (
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
    find_admitted_answer_run_v3,
    read_task12_regular_file_v3,
    select_admitted_answer_run_v3,
    task12_runtime_configuration_sha256_v3,
    validate_task12_runtime_code_binding_v3,
)


@dataclass(frozen=True)
class Task12RunBundleV1:
    cell_id: str
    answer_model_slot: Literal["answer_model_a", "answer_model_b"]
    bundle_root: Path
    task_manifest_path: Path
    task_view_path: Path
    run_config_path: Path
    authorization_path: Path
    execution_output_root: Path
    tasks: tuple[MemUpdateTaskV3, ...]


@dataclass(frozen=True)
class ValidatedTask12RunBundleV1:
    bundle_root: Path
    task_manifest: TaskManifestV3
    tasks: tuple[MemUpdateTaskV3, ...]
    run_configuration: ExternalRunConfigV1
    authorization: Task12ExecutionAuthorizationV1
    frozen_trajectories: dict[str, RawAppendTrajectoryV1]
    execution_output_root: Path


def _read_canonical_json(path: Path, model_type):
    raw = read_task12_regular_file_v3(path)
    model = model_type.model_validate_json(raw)
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"noncanonical artifact: {path}")
    return model


def _read_core_tasks(raw_artifact: bytes) -> dict[str, MemUpdateTaskV3]:
    if not raw_artifact.endswith(b"\n"):
        raise ValueError("core task artifact must end with a newline")
    tasks: dict[str, MemUpdateTaskV3] = {}
    for line_number, raw in enumerate(raw_artifact.splitlines(), start=1):
        if not raw:
            raise ValueError(f"core task line {line_number} is not canonical")
        task = MemUpdateTaskV3.model_validate_json(raw)
        if canonical_json_bytes(task) != raw:
            raise ValueError("core task artifact contains a noncanonical row")
        if task.task_id in tasks:
            raise ValueError(
                f"duplicate task ID in core task artifact: {task.task_id}"
            )
        tasks[task.task_id] = task
    return tasks


def load_task12_frozen_trajectories_v3(
    *,
    manifest: Task12PreparationManifestV1,
    evidence_root: str | Path,
    tasks: tuple[MemUpdateTaskV3, ...],
) -> dict[str, RawAppendTrajectoryV1]:
    evidence = Path(evidence_root).resolve(strict=True)
    location = manifest.semantic_matrix.raw_append_intervention.trajectory_artifact
    raw = _read_artifact(root=evidence, location=location)
    expected_ids = tuple(task.task_id for task in tasks)
    if expected_ids != manifest.semantic_matrix.task_scope.task_ids:
        raise ValueError("Task 12 trajectory tasks must equal the frozen matrix scope")
    _read_raw_append_trajectories(
        raw,
        expected_task_ids=expected_ids,
        expected_tasks=tasks,
    )
    records = tuple(
        RawAppendTrajectoryV1.model_validate_json(line)
        for line in raw.splitlines()
    )
    return {record.task_id: record for record in records}


def _artifact(
    path: str,
    raw: bytes,
    *,
    media_type: str,
    record_count: int,
) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        media_type=media_type,
        record_count=record_count,
    )


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _publish_bundle_artifacts(root: Path, payloads: Mapping[Path, bytes]) -> None:
    try:
        publish_files_atomically(payloads, overwrite=False)
    except BaseException:
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
        raise


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_output_root(
    *,
    output_root: Path,
    core_root: Path,
    evidence_root: Path,
    repository_root: Path,
    create: bool = True,
) -> Path:
    root = output_root.absolute()
    core = core_root.resolve(strict=True)
    evidence = evidence_root.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    assert_no_reparse_components(root)
    if root.exists() or root.is_symlink():
        raise ValueError("Task 12 run-bundle output root must not already exist")
    parent = root.parent
    if not parent.is_dir():
        raise ValueError("Task 12 run-bundle output parent must exist")
    parent_resolved = parent.resolve(strict=True)
    if _contains(core, root) or _contains(root, core) or _contains(core, parent_resolved):
        raise ValueError(
            "Task 12 run-bundle output must be outside the immutable Core root"
        )
    if (
        _contains(evidence, root)
        or _contains(root, evidence)
        or _contains(evidence, parent_resolved)
    ):
        raise ValueError("Task 12 run-bundle output must be outside the evidence root")
    if (
        _contains(repository, root)
        or _contains(root, repository)
        or _contains(repository, parent_resolved)
    ):
        raise ValueError("Task 12 run-bundle output must be outside the repository root")
    if create:
        root.mkdir()
    return root


def _binding_for_slot(
    manifest: Task12PreparationManifestV1,
    answer_model_slot: Literal["answer_model_a", "answer_model_b"],
) -> Task11AnswerModelBindingV1:
    matches = tuple(
        binding for binding in manifest.answer_models if binding.slot_id == answer_model_slot
    )
    if len(matches) != 1:
        raise ValueError("Task 12 manifest must bind exactly one requested answer-model slot")
    return matches[0]


def validate_task12_manifest_plan_v3(
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
) -> None:
    manifest = Task12PreparationManifestV1.model_validate(
        manifest.model_dump(mode="python")
    )
    plan = Task12DryRunPlanV1.model_validate(plan.model_dump(mode="python"))
    checks = {
        "run ID": plan.run_id == manifest.run_id,
        "task manifest": plan.core_task_manifest_sha256
        == manifest.task_manifest.artifact.sha256,
        "hard suite": plan.core_hard_suite_sha256
        == manifest.core_hard_suite.artifact.sha256,
        "task artifact": plan.core_tasks_sha256 == manifest.tasks.artifact.sha256,
        "scientific design": plan.scientific_design_sha256
        == sha256_model(manifest.scientific_design),
        "semantic matrix": plan.semantic_matrix_sha256
        == sha256_model(manifest.semantic_matrix),
        "main manager policy": plan.main_manager_policy_sha256
        == sha256_model(manifest.main_manager_policy),
        "answer slots": plan.answer_model_slots
        == manifest.scientific_design.answer_model_slots,
        "answer bindings": plan.answer_model_binding_sha256
        == tuple(
            _canonical_answer_model_binding_hash(binding)
            for binding in manifest.answer_models
        ),
    }
    failed = tuple(label for label, valid in checks.items() if not valid)
    if failed:
        raise ValueError(f"Task 12 manifest/plan binding mismatch: {failed[0]}")
    expected_cells = tuple(
        Task12AdmittedCellV1(
            cell_id=cell.cell_id,
            scope_id=cell.scope_id,
            canonical_binding_sha256=_canonical_cell_binding_hash(
                cell=cell,
                scope=manifest.semantic_matrix.task_scope,
                raw_append_intervention=(
                    manifest.semantic_matrix.raw_append_intervention
                ),
            ),
        )
        for cell in manifest.semantic_matrix.intervention_cells
    )
    if plan.admitted_cells != expected_cells:
        raise ValueError("Task 12 admitted cells differ from semantic matrix")
    cell_hashes = {
        cell.cell_id: cell.canonical_binding_sha256
        for cell in expected_cells
    }
    answer_hashes = dict(
        zip(plan.answer_model_slots, plan.answer_model_binding_sha256)
    )
    expected_runs = tuple(
        Task12AdmittedAnswerRunV1(
            cell_id=cell.cell_id,
            answer_model_slot=slot,
            cell_binding_sha256=cell_hashes[cell.cell_id],
            answer_model_binding_sha256=answer_hashes[slot],
            canonical_run_binding_sha256=_canonical_answer_run_binding_hash(
                cell_binding_sha256=cell_hashes[cell.cell_id],
                answer_model_binding_sha256=answer_hashes[slot],
            ),
        )
        for cell in expected_cells
        for slot in plan.answer_model_slots
    )
    if plan.admitted_answer_runs != expected_runs:
        raise ValueError("Task 12 admitted runs differ from manifest bindings")


def _selected_answer_run(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    cell_id: str,
    answer_model_slot: Literal["answer_model_a", "answer_model_b"],
) -> Task12AdmittedAnswerRunV1:
    validate_task12_manifest_plan_v3(manifest, plan)
    return find_admitted_answer_run_v3(
        cell_id=cell_id,
        answer_model_slot=answer_model_slot,
        admitted_cells=plan.admitted_cells,
        admitted_answer_runs=plan.admitted_answer_runs,
    )


def _load_capabilities(
    root: Path,
    location,
) -> AdapterCapabilitiesV3:
    raw = _read_artifact(root=root, location=location)
    record = Task12CapabilityVerificationV1.model_validate_json(raw)
    if canonical_json_bytes(record) != raw:
        raise ValueError("noncanonical Task 12 capability verification")
    return record.capabilities


def _task_manifest_for_view(
    *,
    source_manifest: TaskManifestV3,
    task_view_ref: ArtifactRef,
    selected_tasks: tuple[MemUpdateTaskV3, ...],
) -> TaskManifestV3:
    family_counts: dict[str, int] = {}
    core_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for task in selected_tasks:
        split = task.metadata.split.value
        split_counts[split] = split_counts.get(split, 0) + 1
        family_key = f"{task.task_family}/{task.difficulty.value}"
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        core_id = task.metadata.split_key.semantic_core_id
        core_counts[core_id] = core_counts.get(core_id, 0) + 1
    return TaskManifestV3(
        data_release_id=f"{source_manifest.data_release_id}-task12-view",
        split_policy_version=source_manifest.split_policy_version,
        compiler_versions=source_manifest.compiler_versions,
        source_manifest_paths_and_hashes=source_manifest.source_manifest_paths_and_hashes,
        generation_configs_and_hashes=source_manifest.generation_configs_and_hashes,
        split_counts=split_counts,
        family_difficulty_counts=family_counts,
        semantic_core_counts=core_counts,
        task_file_paths_and_hashes=(task_view_ref,),
        leakage_check_summary=source_manifest.leakage_check_summary,
        human_audit_artifacts=source_manifest.human_audit_artifacts,
        created_at=source_manifest.created_at,
        code_revision=source_manifest.code_revision,
        task_record_hashes={task.task_id: sha256_model(task) for task in selected_tasks},
    )


def _prompt_expectations(
    tasks: tuple[MemUpdateTaskV3, ...],
) -> tuple[PromptedTaskExpectationV1, ...]:
    return tuple(
        PromptedTaskExpectationV1(
            task_id=task.task_id,
            action_ids=tuple(action.action_id for action in task.actions),
            query_ids=tuple(query.query_id for query in task.queries),
        )
        for task in tasks
    )


def _qualified_decode_config(
    *,
    evidence_root: Path,
    binding: Task11AnswerModelBindingV1,
) -> dict[str, object]:
    qualification_raw = _read_artifact(
        root=evidence_root,
        location=binding.qualification_report,
    )
    qualification = Task11QualificationReportV1.model_validate_json(
        qualification_raw
    )
    if canonical_json_bytes(qualification) != qualification_raw:
        raise ValueError("noncanonical Task 11 qualification report")
    matches = tuple(slot for slot in qualification.slots if slot.slot_id == binding.slot_id)
    if len(matches) != 1:
        raise ValueError("Task 11 qualification does not bind the requested slot")
    slot = matches[0]
    if (
        slot.model_id != binding.model_id
        or slot.revision != binding.revision
        or slot.tree_manifest_sha256 != binding.tree_manifest_sha256
        or slot.license_id != binding.license_id
    ):
        raise ValueError("Task 11 qualification slot differs from preparation binding")
    decoding = qualification.offline_contract.decoding
    if sha256_model(decoding) != binding.decoding_config_sha256:
        raise ValueError("Task 11 decoding config hash differs from preparation binding")
    return {
        **decoding.model_dump(mode="json"),
        "decoding_config_sha256": binding.decoding_config_sha256,
    }


def _prompt_template_hash() -> str:
    return hashlib.sha256(
        b"memupdatebench:prompted-answer-v1:visible-context-v1:answer-envelope-v1"
    ).hexdigest()


def _json_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_existing_bundle_root(
    *,
    bundle_root: str | Path,
    core_root: Path,
    evidence_root: Path,
    repository_root: Path,
) -> Path:
    root = Path(bundle_root).absolute()
    assert_no_reparse_components(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Task 12 run bundle must be an existing real directory")
    resolved = root.resolve(strict=True)
    for restricted, label in (
        (core_root.resolve(strict=True), "immutable Core"),
        (evidence_root.resolve(strict=True), "evidence"),
        (repository_root.resolve(strict=True), "repository"),
    ):
        if _contains(restricted, resolved) or _contains(resolved, restricted):
            raise ValueError(f"Task 12 run bundle must be outside the {label} root")
    return resolved


def _read_exact_bundle_tasks(
    raw: bytes,
    expected_ids: tuple[str, ...],
) -> tuple[MemUpdateTaskV3, ...]:
    if not raw.endswith(b"\n"):
        raise ValueError("Task 12 task view must end with a newline")
    tasks = tuple(
        MemUpdateTaskV3.model_validate_json(line)
        for line in raw.splitlines()
    )
    if any(canonical_json_bytes(task) != line for task, line in zip(tasks, raw.splitlines())):
        raise ValueError("Task 12 task view contains a noncanonical row")
    observed_ids = tuple(task.task_id for task in tasks)
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Task 12 task view must equal the frozen 80-task scope")
    return tasks


def _validate_task12_run_bundle_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    bundle_root: str | Path,
    source_manifest: TaskManifestV3 | None = None,
    source_tasks_by_id: Mapping[str, MemUpdateTaskV3] | None = None,
    frozen_trajectories: Mapping[str, RawAppendTrajectoryV1] | None = None,
) -> ValidatedTask12RunBundleV1:
    core = Path(core_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    validate_task12_manifest_plan_v3(manifest, plan)
    root = _validate_existing_bundle_root(
        bundle_root=bundle_root,
        core_root=core,
        evidence_root=evidence,
        repository_root=repository,
    )
    task_manifest_path = root / "task_manifest.json"
    task_view_path = root / "tasks.jsonl"
    run_config_path = root / "run_config.json"
    authorization_path = root / "authorization.json"
    authorization = _read_canonical_json(
        authorization_path,
        Task12ExecutionAuthorizationV1,
    )
    validate_task12_runtime_code_binding_v3(
        authorization.runtime_code_binding,
        runtime_code_binding,
    )
    task_manifest_raw = read_task12_regular_file_v3(task_manifest_path)
    task_view_raw = read_task12_regular_file_v3(task_view_path)
    run_config_raw = read_task12_regular_file_v3(run_config_path)
    selected_run = select_admitted_answer_run_v3(
        authorization,
        preparation_manifest_sha256=sha256_model(manifest),
        plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
        admitted_cells=plan.admitted_cells,
        admitted_answer_runs=plan.admitted_answer_runs,
    )
    artifact_hashes = {
        "task-manifest": hashlib.sha256(task_manifest_raw).hexdigest(),
        "task-view": hashlib.sha256(task_view_raw).hexdigest(),
        "run-config": hashlib.sha256(run_config_raw).hexdigest(),
    }
    expected_hashes = {
        "task-manifest": authorization.task_manifest_sha256,
        "task-view": authorization.task_view_sha256,
        "run-config": authorization.run_config_sha256,
    }
    if artifact_hashes != expected_hashes:
        mismatch = next(
            label
            for label in artifact_hashes
            if artifact_hashes[label] != expected_hashes[label]
        )
        raise ValueError(f"Task 12 {mismatch} hash differs from authorization")

    selected_cell = next(
        cell
        for cell in manifest.semantic_matrix.intervention_cells
        if cell.cell_id == selected_run.cell_id
    )
    if authorization.expected_task_count != len(selected_cell.task_ids):
        raise ValueError("Task 12 authorization task count differs from cell scope")
    tasks = _read_exact_bundle_tasks(task_view_raw, selected_cell.task_ids)
    if source_tasks_by_id is None:
        source_tasks_raw = _read_artifact(root=core, location=manifest.tasks)
        if hashlib.sha256(source_tasks_raw).hexdigest() != plan.core_tasks_sha256:
            raise ValueError("Core task artifact differs from admitted plan")
        loaded_source_tasks = _read_core_tasks(source_tasks_raw)
    else:
        loaded_source_tasks = dict(source_tasks_by_id)
    if tasks != tuple(
        loaded_source_tasks[task_id]
        for task_id in selected_cell.task_ids
    ):
        raise ValueError("Task 12 task view differs from immutable Core tasks")

    if source_manifest is None:
        source_manifest_raw = _read_artifact(
            root=core,
            location=manifest.task_manifest,
        )
        loaded_source_manifest = TaskManifestV3.model_validate_json(
            source_manifest_raw
        )
        if canonical_json_bytes(loaded_source_manifest) != source_manifest_raw:
            raise ValueError("source task manifest is not canonical")
    else:
        loaded_source_manifest = source_manifest
    task_view_ref = _artifact(
        "tasks.jsonl",
        task_view_raw,
        media_type="application/x-ndjson",
        record_count=len(tasks),
    )
    expected_task_manifest = _task_manifest_for_view(
        source_manifest=loaded_source_manifest,
        task_view_ref=task_view_ref,
        selected_tasks=tasks,
    )
    task_manifest = TaskManifestV3.model_validate_json(task_manifest_raw)
    if canonical_json_bytes(task_manifest) != task_manifest_raw:
        raise ValueError("Task 12 task manifest is not canonical")
    if task_manifest != expected_task_manifest:
        raise ValueError("Task 12 task manifest differs from the authenticated view")
    task_manifest_ref = _artifact(
        "task_manifest.json",
        task_manifest_raw,
        media_type="application/json",
        record_count=1,
    )

    run_configuration = ExternalRunConfigV1.model_validate_json(run_config_raw)
    if canonical_json_bytes(run_configuration) != run_config_raw:
        raise ValueError("Task 12 run configuration is not canonical")
    binding = _binding_for_slot(manifest, authorization.answer_model_slot)
    adapter_info_raw = _read_artifact(
        root=evidence,
        location=selected_cell.adapter_info,
    )
    adapter_info = AdapterInfoV3.model_validate_json(adapter_info_raw)
    adapter_capabilities = _load_capabilities(
        evidence,
        selected_cell.capability_verification,
    )
    _read_artifact(root=evidence, location=selected_cell.adapter_configuration)
    _read_artifact(root=evidence, location=selected_cell.retrieval.artifact)
    decoding_config = _qualified_decode_config(
        evidence_root=evidence,
        binding=binding,
    )
    if authorization.answer_model_slot == "answer_model_b":
        _read_artifact(
            root=evidence,
            location=manifest.task11_mistral_provenance,
        )
    expected_runtime_hash = task12_runtime_configuration_sha256_v3(
        RuntimeConfigV3(
            run_id=(
                f"task12-{selected_cell.cell_id}-"
                f"{authorization.answer_model_slot}"
            ),
            retrieval_policy=selected_cell.retrieval.configuration.retrieval_policy,
            answer_mode="slot_prompt",
            retrieval_k=selected_cell.retrieval.configuration.retrieval_k,
            capture_snapshots=False,
        ),
        context_order=selected_cell.context_intervention.context_order,
        context_annotation=selected_cell.context_intervention.context_annotation,
    )
    expected_evaluation_hash = _json_sha256(
        {
            "cell_id": selected_cell.cell_id,
            "answer_model_slot": authorization.answer_model_slot,
            "canonical_run_binding_sha256": selected_run.canonical_run_binding_sha256,
            "task_manifest_sha256": authorization.task_manifest_sha256,
        }
    )
    expected_model_provenance = (
        manifest.task11_mistral_provenance.artifact
        if authorization.answer_model_slot == "answer_model_b"
        else binding.qualification_report.artifact
    )
    checks = {
        "run ID": run_configuration.run_id
        == f"task12-{selected_cell.cell_id}-{authorization.answer_model_slot}",
        "code revision": run_configuration.code_revision
        == runtime_code_binding.code_revision,
        "clean state": run_configuration.dirty_state is False,
        "task manifest ref": run_configuration.source_task_manifest_ref
        == task_manifest_ref,
        "task view ref": run_configuration.task_view_ref == task_view_ref,
        "adapter configuration": run_configuration.adapter_configuration_ref
        == selected_cell.adapter_configuration.artifact,
        "capability verification": run_configuration.capability_verification_ref
        == selected_cell.capability_verification.artifact,
        "model provenance": run_configuration.model_provenance_ref
        == expected_model_provenance,
        "package provenance": run_configuration.package_provenance_ref
        == binding.qualification_report.artifact,
        "environment lock": run_configuration.environment_lock_ref
        == binding.qualification_report.artifact,
        "adapter info": run_configuration.adapter_info == adapter_info,
        "adapter capabilities": run_configuration.adapter_capabilities
        == adapter_capabilities,
        "retrieval policy": run_configuration.retrieval_policy == "normal_topk",
        "answer mode": run_configuration.answer_mode == "slot_prompt",
        "runtime hash": run_configuration.runtime_configuration_hash
        == expected_runtime_hash,
        "evaluation hash": run_configuration.evaluation_configuration_hash
        == expected_evaluation_hash,
        "provider": run_configuration.provider == "offline_hf",
        "model ID": run_configuration.model_name == binding.model_id,
        "model revision": run_configuration.model_revision == binding.revision,
        "answer slot": run_configuration.answer_model_slot == binding.slot_id,
        "prompt config": run_configuration.prompt_config
        == {
            "prompt_protocol_version": "prompted-answer-v1",
            "renderer_version": "visible-context-v1",
            "template_hash": _prompt_template_hash(),
            "typed_output_format": "answer-envelope-v1",
        },
        "decode config": run_configuration.decoding_config == decoding_config,
        "seed": run_configuration.seed_information
        == {"seed": decoding_config["seed"]},
        "offline environment": run_configuration.environment_summary
        == {
            "hf_hub_offline": True,
            "transformers_offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "runtime_code_tree_sha256": runtime_code_binding.code_tree_sha256,
        },
        "task order": run_configuration.expected_task_ids
        == tuple(task.task_id for task in tasks),
        "task hashes": dict(run_configuration.task_record_hashes)
        == {task.task_id: sha256_model(task) for task in tasks},
        "prompt expectations": run_configuration.prompted_task_expectations
        == _prompt_expectations(tasks),
    }
    failed = tuple(label for label, valid in checks.items() if not valid)
    if failed:
        raise ValueError(f"Task 12 run config binding mismatch: {failed[0]}")
    if frozen_trajectories is None:
        loaded_frozen_trajectories = load_task12_frozen_trajectories_v3(
            manifest=manifest,
            evidence_root=evidence,
            tasks=tasks,
        )
    else:
        loaded_frozen_trajectories = dict(frozen_trajectories)
        if set(loaded_frozen_trajectories) != set(selected_cell.task_ids):
            raise ValueError("preloaded Task 12 trajectories differ from cell scope")
    return ValidatedTask12RunBundleV1(
        bundle_root=root,
        task_manifest=task_manifest,
        tasks=tasks,
        run_configuration=run_configuration,
        authorization=authorization,
        frozen_trajectories=loaded_frozen_trajectories,
        execution_output_root=root / authorization.output_leaf,
    )


def validate_task12_run_bundle_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    bundle_root: str | Path,
) -> ValidatedTask12RunBundleV1:
    return _validate_task12_run_bundle_v3(
        manifest=manifest,
        plan=plan,
        core_root=core_root,
        evidence_root=evidence_root,
        repository_root=repository_root,
        runtime_code_binding=runtime_code_binding,
        bundle_root=bundle_root,
    )


def build_task12_run_bundle_v3(
    *,
    manifest: Task12PreparationManifestV1,
    plan: Task12DryRunPlanV1,
    core_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    runtime_code_binding: Task12RuntimeCodeBindingV1,
    output_root: str | Path,
    cell_id: str,
    answer_model_slot: Literal["answer_model_a", "answer_model_b"],
    output_leaf: str,
) -> Task12RunBundleV1:
    core = Path(core_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    validate_task12_manifest_plan_v3(manifest, plan)
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
        output_root=Path(output_root),
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


__all__ = [
    "Task12RunBundleV1",
    "ValidatedTask12RunBundleV1",
    "build_task12_run_bundle_v3",
    "load_task12_frozen_trajectories_v3",
    "validate_task12_manifest_plan_v3",
    "validate_task12_run_bundle_v3",
]
