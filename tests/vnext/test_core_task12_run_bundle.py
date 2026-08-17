from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.preparation.task12 import admit_task12_dry_run
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.task12_bundle_v3 import (
    _ensure_output_root,
    build_task12_run_bundle_v3,
    validate_task12_manifest_plan_v3,
    validate_task12_run_bundle_v3,
)
from mub.vnext.runtime.task12_execution_v3 import (
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
)
from tests.vnext.task12_fixtures import build_task12_inputs, build_task12_manifest
from tests.vnext.test_core_task12_preparation import _authorize_fixture_release


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CODE_BINDING = Task12RuntimeCodeBindingV1(
    code_revision="8" * 40,
    code_tree_sha256="9" * 64,
)


def test_task12_run_bundle_derives_single_authorized_cell_slot_config(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    )
    bundle_root = tmp_path / "bundle"
    cell_id = "raw-add-reverse-version-labeled-k04"
    assert RUNTIME_CODE_BINDING.code_revision != plan.code_revision
    assert RUNTIME_CODE_BINDING.code_tree_sha256 != plan.code_tree_sha256
    mutated_answer_model = manifest.answer_models[0].model_copy(
        update={"tree_manifest_sha256": "0" * 64}
    )
    mutated_manifest = manifest.model_copy(
        update={
            "answer_models": (
                mutated_answer_model,
                manifest.answer_models[1],
            )
        }
    )
    with pytest.raises(ValueError, match="manifest/plan binding mismatch"):
        validate_task12_manifest_plan_v3(mutated_manifest, plan)
    core_tasks_path = inputs["core_root"] / manifest.tasks.relative_path
    core_tasks_raw = core_tasks_path.read_bytes()
    core_tasks_path.write_bytes(core_tasks_raw + b"\n")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        build_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            output_root=bundle_root,
            cell_id=cell_id,
            answer_model_slot="answer_model_b",
            output_leaf="cell-b-slot-b",
        )
    core_tasks_path.write_bytes(core_tasks_raw)

    bundle = build_task12_run_bundle_v3(
        manifest=manifest,
        plan=plan,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        output_root=bundle_root,
        cell_id=cell_id,
        answer_model_slot="answer_model_b",
        output_leaf="cell-b-slot-b",
    )

    assert bundle.cell_id == cell_id
    assert bundle.answer_model_slot == "answer_model_b"
    assert tuple(task.task_id for task in bundle.tasks) == manifest.semantic_matrix.task_scope.task_ids
    assert bundle.task_manifest_path == bundle_root / "task_manifest.json"
    assert bundle.task_view_path == bundle_root / "tasks.jsonl"
    assert bundle.run_config_path == bundle_root / "run_config.json"
    assert bundle.authorization_path == bundle_root / "authorization.json"
    assert bundle.execution_output_root == bundle_root / "cell-b-slot-b"
    assert not bundle.execution_output_root.exists()

    authorization = Task12ExecutionAuthorizationV1.model_validate_json(
        bundle.authorization_path.read_bytes()
    )
    run_config = ExternalRunConfigV1.model_validate_json(bundle.run_config_path.read_bytes())
    selected_cell = next(cell for cell in manifest.semantic_matrix.intervention_cells if cell.cell_id == cell_id)
    answer_binding = next(binding for binding in manifest.answer_models if binding.slot_id == "answer_model_b")

    assert authorization.preparation_manifest_sha256 == sha256_model(manifest)
    assert authorization.plan_fingerprint_sha256 == plan.plan_fingerprint_sha256
    assert authorization.runtime_code_binding == RUNTIME_CODE_BINDING
    selected_run = next(
        run
        for run in plan.admitted_answer_runs
        if run.cell_id == cell_id and run.answer_model_slot == "answer_model_b"
    )
    assert authorization.cell_binding_sha256 == selected_run.cell_binding_sha256
    assert (
        authorization.answer_model_binding_sha256
        == selected_run.answer_model_binding_sha256
    )
    assert (
        authorization.canonical_run_binding_sha256
        == selected_run.canonical_run_binding_sha256
    )
    assert authorization.task_manifest_sha256 == hashlib.sha256(
        bundle.task_manifest_path.read_bytes()
    ).hexdigest()
    assert authorization.task_view_sha256 == hashlib.sha256(
        bundle.task_view_path.read_bytes()
    ).hexdigest()
    assert authorization.run_config_sha256 == hashlib.sha256(
        bundle.run_config_path.read_bytes()
    ).hexdigest()
    assert authorization.expected_task_count == 80
    assert authorization.execution_authorized is True
    assert run_config.source_task_manifest_ref.sha256 == hashlib.sha256(bundle.task_manifest_path.read_bytes()).hexdigest()
    assert run_config.task_view_ref.sha256 == hashlib.sha256(bundle.task_view_path.read_bytes()).hexdigest()
    assert run_config.adapter_configuration_ref == selected_cell.adapter_configuration.artifact
    assert run_config.capability_verification_ref == selected_cell.capability_verification.artifact
    assert run_config.model_name == answer_binding.model_id
    assert run_config.model_revision == answer_binding.revision
    assert run_config.code_revision == RUNTIME_CODE_BINDING.code_revision
    assert (
        run_config.environment_summary["runtime_code_tree_sha256"]
        == RUNTIME_CODE_BINDING.code_tree_sha256
    )
    assert run_config.answer_model_slot == "answer_model_b"
    assert run_config.expected_task_ids == tuple(task.task_id for task in bundle.tasks)
    assert dict(run_config.task_record_hashes) == {
        task.task_id: sha256_model(task) for task in bundle.tasks
    }
    assert tuple(expectation.task_id for expectation in run_config.prompted_task_expectations) == run_config.expected_task_ids
    assert tuple(
        expectation.action_ids for expectation in run_config.prompted_task_expectations
    ) == tuple(tuple(action.action_id for action in task.actions) for task in bundle.tasks)
    assert tuple(
        expectation.query_ids for expectation in run_config.prompted_task_expectations
    ) == tuple(tuple(query.query_id for query in task.queries) for task in bundle.tasks)
    assert canonical_json_bytes(authorization) == bundle.authorization_path.read_bytes()
    assert canonical_json_bytes(run_config) == bundle.run_config_path.read_bytes()

    validated = validate_task12_run_bundle_v3(
        manifest=manifest,
        plan=plan,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        bundle_root=bundle_root,
    )
    assert validated.tasks == bundle.tasks
    assert validated.execution_output_root == bundle.execution_output_root
    assert set(validated.frozen_trajectories) == set(run_config.expected_task_ids)
    with pytest.raises(ValueError, match="runtime code binding"):
        validate_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING.model_copy(
                update={"code_tree_sha256": "0" * 64}
            ),
            bundle_root=bundle_root,
        )

    adapter_info_path = (
        inputs["evidence_root"] / selected_cell.adapter_info.relative_path
    )
    adapter_info_raw = adapter_info_path.read_bytes()
    adapter_info_path.write_bytes(adapter_info_raw + b" ")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        validate_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            bundle_root=bundle_root,
        )
    adapter_info_path.write_bytes(adapter_info_raw)

    task_view_raw = bundle.task_view_path.read_bytes()
    bundle.task_view_path.write_bytes(b"\n".join(task_view_raw.splitlines()[:-1]) + b"\n")
    with pytest.raises(ValueError, match="task-view hash"):
        validate_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            bundle_root=bundle_root,
        )


def test_task12_run_bundle_rejects_mismatched_plan_hash(tmp_path, monkeypatch) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    ).model_copy(update={"core_task_manifest_sha256": "0" * 64})

    with pytest.raises(ValueError, match="manifest/plan binding mismatch: task manifest"):
        build_task12_run_bundle_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            output_root=tmp_path / "bundle",
            cell_id="raw-add-chronological-none-k04",
            answer_model_slot="answer_model_a",
            output_leaf="cell-a-slot-a",
        )


def test_task12_run_bundle_rejects_output_inside_core(tmp_path) -> None:
    core_root = tmp_path / "core"
    evidence_root = tmp_path / "evidence"
    repository_root = tmp_path / "repository"
    core_root.mkdir()
    evidence_root.mkdir()
    repository_root.mkdir()

    with pytest.raises(ValueError, match="outside the immutable Core root"):
        _ensure_output_root(
            output_root=core_root / "bundle",
            core_root=core_root,
            evidence_root=evidence_root,
            repository_root=repository_root,
        )


def test_task12_run_bundle_rejects_output_inside_repository(tmp_path) -> None:
    repository_root = tmp_path / "repository"
    core_root = tmp_path / "core"
    evidence_root = tmp_path / "evidence"
    repository_root.mkdir()
    (repository_root / "results").mkdir()
    core_root.mkdir()
    evidence_root.mkdir()

    with pytest.raises(ValueError, match="outside the repository root"):
        _ensure_output_root(
            output_root=repository_root / "results" / "task12",
            core_root=core_root,
            evidence_root=evidence_root,
            repository_root=repository_root,
        )
