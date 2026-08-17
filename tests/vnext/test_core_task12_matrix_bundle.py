from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.preparation.task12 import admit_task12_dry_run
from mub.vnext.runtime.run_v3 import ExternalRunConfigV1
from mub.vnext.runtime.task12_execution_v3 import (
    Task12ExecutionAuthorizationV1,
    Task12RuntimeCodeBindingV1,
)
from mub.vnext.runtime.task12_matrix_v3 import (
    Task12MatrixBundleManifestV1,
    Task12MatrixRunSummaryV1,
    build_task12_matrix_bundles_v3,
    execute_task12_matrix_bundles_v3,
)
from tests.vnext.task12_fixtures import build_task12_inputs, build_task12_manifest
from tests.vnext.test_core_task12_preparation import _authorize_fixture_release


class _FakePromptedAnswerModel:
    def __init__(self, answer: str, slot_id: str) -> None:
        self.answer_text = answer
        self.slot_id = slot_id
        self.load_count = 0
        self.close_count = 0
        self.answer_count = 0
        self.loaded = False

    def load(self) -> None:
        self.load_count += 1
        self.loaded = True

    def answer(self, request: PromptedAnswerRequestV3) -> AnswerPredictionV3:
        if not self.loaded:
            raise RuntimeError("fake matrix model is not loaded")
        self.answer_count += 1
        return AnswerPredictionV3(
            query_id=request.query.query_id,
            raw_output=json.dumps(
                {"disposition": "answered", "answer": self.answer_text},
                separators=(",", ":"),
                sort_keys=True,
            ),
            parsed_answer=self.answer_text,
            format_valid=True,
        )

    def close(self) -> None:
        self.close_count += 1
        self.loaded = False


class _FailingLoadModel(_FakePromptedAnswerModel):
    def load(self) -> None:
        super().load()
        raise RuntimeError("intentional load failure")


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CODE_BINDING = Task12RuntimeCodeBindingV1(
    code_revision="8" * 40,
    code_tree_sha256="9" * 64,
)


def _admitted_pairs(plan):
    return tuple(
        (run.cell_id, run.answer_model_slot)
        for run in plan.admitted_answer_runs
    )


def test_task12_matrix_bundle_prepares_all_18_cell_slot_bundles(tmp_path, monkeypatch) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    )

    matrix = build_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        output_root=tmp_path / "matrix",
    )

    assert len(matrix.bundles) == 18
    assert matrix.matrix_manifest_path == tmp_path / "matrix" / "matrix_bundle_manifest.json"
    assert tuple((bundle.cell_id, bundle.answer_model_slot) for bundle in matrix.bundles) == _admitted_pairs(plan)
    assert len({bundle.bundle_root.name for bundle in matrix.bundles}) == 18
    assert all(bundle.execution_output_root == bundle.bundle_root / "run" for bundle in matrix.bundles)
    assert all(not bundle.execution_output_root.exists() for bundle in matrix.bundles)
    assert all(len(bundle.tasks) == 80 for bundle in matrix.bundles)
    assert all((bundle.bundle_root / "tasks.jsonl").is_file() for bundle in matrix.bundles)
    assert all((bundle.bundle_root / "task_manifest.json").is_file() for bundle in matrix.bundles)
    assert all((bundle.bundle_root / "run_config.json").is_file() for bundle in matrix.bundles)
    assert all((bundle.bundle_root / "authorization.json").is_file() for bundle in matrix.bundles)

    matrix_manifest = Task12MatrixBundleManifestV1.model_validate_json(
        matrix.matrix_manifest_path.read_bytes()
    )
    assert canonical_json_bytes(matrix_manifest) == matrix.matrix_manifest_path.read_bytes()
    assert matrix_manifest.preparation_manifest_sha256 == sha256_model(manifest)
    assert matrix_manifest.plan_fingerprint_sha256 == plan.plan_fingerprint_sha256
    assert matrix_manifest.bundle_count == 18
    assert tuple((ref.cell_id, ref.answer_model_slot) for ref in matrix_manifest.run_bundles) == _admitted_pairs(plan)
    for ref in matrix_manifest.run_bundles:
        bundle_root = matrix.matrix_root / ref.bundle_leaf
        assert ref.task_manifest_sha256 == hashlib.sha256((bundle_root / "task_manifest.json").read_bytes()).hexdigest()
        assert ref.task_view_sha256 == hashlib.sha256((bundle_root / "tasks.jsonl").read_bytes()).hexdigest()
        assert ref.run_config_sha256 == hashlib.sha256((bundle_root / "run_config.json").read_bytes()).hexdigest()
        assert ref.authorization_sha256 == hashlib.sha256((bundle_root / "authorization.json").read_bytes()).hexdigest()

    first = matrix.bundles[0]
    authorization = Task12ExecutionAuthorizationV1.model_validate_json(
        first.authorization_path.read_bytes()
    )
    run_config = ExternalRunConfigV1.model_validate_json(first.run_config_path.read_bytes())
    assert authorization.preparation_manifest_sha256 == sha256_model(manifest)
    assert authorization.cell_id == first.cell_id
    assert authorization.answer_model_slot == first.answer_model_slot
    assert authorization.runtime_code_binding == RUNTIME_CODE_BINDING
    assert run_config.source_task_manifest_ref.sha256 == hashlib.sha256(first.task_manifest_path.read_bytes()).hexdigest()
    assert run_config.task_view_ref.sha256 == hashlib.sha256(first.task_view_path.read_bytes()).hexdigest()
    assert run_config.expected_task_ids == tuple(task.task_id for task in first.tasks)

    tampered_ref = matrix.manifest.run_bundles[0].model_copy(
        update={"cell_id": "raw-add-chronological-none-k32"}
    )
    tampered_manifest = matrix.manifest.model_copy(
        update={
            "run_bundles": (
                tampered_ref,
                *matrix.manifest.run_bundles[1:],
            )
        }
    )
    matrix.matrix_manifest_path.write_bytes(canonical_json_bytes(tampered_manifest))
    unloaded_models = {
        "answer_model_a": _FakePromptedAnswerModel("city-80", "answer_model_a"),
        "answer_model_b": _FakePromptedAnswerModel("city-80", "answer_model_b"),
    }
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3

    with pytest.raises(ValueError, match="admitted answer runs"):
        execute_task12_matrix_bundles_v3(
            manifest=manifest,
            plan=plan,
            matrix_bundle_manifest=tampered_manifest,
            matrix_root=matrix.matrix_root,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            adapter_factory=lambda task: RawAppendAdapterV3(
                task,
                retrieval_policy="normal_topk",
            ),
            prompted_answer_models=unloaded_models,
        )
    assert unloaded_models["answer_model_a"].load_count == 0
    assert unloaded_models["answer_model_b"].load_count == 0

    matrix.matrix_manifest_path.write_bytes(canonical_json_bytes(matrix.manifest))
    models = {
        "answer_model_a": _FakePromptedAnswerModel("city-80", "answer_model_a"),
        "answer_model_b": _FailingLoadModel("city-80", "answer_model_b"),
    }
    with pytest.raises(RuntimeError, match="intentional load failure"):
        execute_task12_matrix_bundles_v3(
            manifest=manifest,
            plan=plan,
            matrix_bundle_manifest=matrix.manifest,
            matrix_root=matrix.matrix_root,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            adapter_factory=lambda task: RawAppendAdapterV3(
                task,
                retrieval_policy="normal_topk",
            ),
            prompted_answer_models=models,
        )
    assert models["answer_model_a"].close_count == 1
    assert models["answer_model_b"].close_count == 1
    assert not (matrix.matrix_root / "matrix_run_summary.json").exists()


def test_task12_matrix_runner_executes_all_18_fake_offline_runs(tmp_path, monkeypatch) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    )
    matrix = build_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        output_root=tmp_path / "matrix",
    )
    models = {
        "answer_model_a": _FakePromptedAnswerModel("city-80", "answer_model_a"),
        "answer_model_b": _FakePromptedAnswerModel("city-80", "answer_model_b"),
    }
    from mub.vnext.adapters.core_v3 import RawAppendAdapterV3

    result = execute_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        matrix_bundle_manifest=matrix.manifest,
        matrix_root=matrix.matrix_root,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        adapter_factory=lambda task: RawAppendAdapterV3(task, retrieval_policy="normal_topk"),
        prompted_answer_models=models,
    )

    assert result.summary_path == matrix.matrix_root / "matrix_run_summary.json"
    loaded_summary = Task12MatrixRunSummaryV1.model_validate_json(
        result.summary_path.read_bytes()
    )
    assert canonical_json_bytes(loaded_summary) == result.summary_path.read_bytes()
    assert loaded_summary == result.summary
    assert loaded_summary.run_count == 18
    assert len(loaded_summary.completed_runs) == 18
    assert loaded_summary.total_task_rows == 18 * 80
    assert loaded_summary.total_score_rows == 18 * 80
    assert tuple((run.cell_id, run.answer_model_slot) for run in loaded_summary.completed_runs) == _admitted_pairs(plan)
    assert all((matrix.matrix_root / run.bundle_leaf / run.output_leaf / "run_manifest.json").is_file() for run in loaded_summary.completed_runs)
    assert all((matrix.matrix_root / run.bundle_leaf / run.output_leaf / "scores" / "scores.jsonl").is_file() for run in loaded_summary.completed_runs)
    assert models["answer_model_a"].load_count == 1
    assert models["answer_model_b"].load_count == 1
    assert models["answer_model_a"].close_count == 1
    assert models["answer_model_b"].close_count == 1
    assert models["answer_model_a"].answer_count > 0
    assert models["answer_model_b"].answer_count > 0

    resumed_models = {
        "answer_model_a": _FakePromptedAnswerModel("city-80", "answer_model_a"),
        "answer_model_b": _FakePromptedAnswerModel("city-80", "answer_model_b"),
    }
    resumed = execute_task12_matrix_bundles_v3(
        manifest=manifest,
        plan=plan,
        matrix_bundle_manifest=matrix.manifest,
        matrix_root=matrix.matrix_root,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        repository_root=ROOT,
        runtime_code_binding=RUNTIME_CODE_BINDING,
        adapter_factory=lambda task: RawAppendAdapterV3(
            task,
            retrieval_policy="normal_topk",
        ),
        prompted_answer_models=resumed_models,
        resume=True,
    )
    assert resumed.summary == result.summary
    assert resumed_models["answer_model_a"].load_count == 0
    assert resumed_models["answer_model_b"].load_count == 0
    assert resumed_models["answer_model_a"].answer_count == 0
    assert resumed_models["answer_model_b"].answer_count == 0

def test_task12_matrix_bundle_rejects_existing_output_root(tmp_path, monkeypatch) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    plan = admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    )
    output_root = tmp_path / "matrix"
    output_root.mkdir()

    with pytest.raises(ValueError, match="output root must not already exist"):
        build_task12_matrix_bundles_v3(
            manifest=manifest,
            plan=plan,
            core_root=inputs["core_root"],
            evidence_root=inputs["evidence_root"],
            repository_root=ROOT,
            runtime_code_binding=RUNTIME_CODE_BINDING,
            output_root=output_root,
        )


def test_task12_matrix_runner_cli_requires_execute_flag() -> None:
    script = ROOT / "scripts" / "vnext_run_core_task12_matrix.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--matrix-root" in completed.stdout
    assert "--matrix-bundle-manifest" in completed.stdout
    assert "--execute" in completed.stdout
    assert "--token" not in completed.stdout.lower()
    assert "--provider" not in completed.stdout.lower()


def test_task12_matrix_bundle_cli_has_no_execution_or_provider_flags() -> None:
    script = ROOT / "scripts" / "vnext_prepare_core_task12_matrix.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout
    assert "--plan" in completed.stdout
    assert "--execute" not in completed.stdout
    assert "--token" not in completed.stdout.lower()
    assert "--provider" not in completed.stdout.lower()
    assert "--device" not in completed.stdout.lower()
