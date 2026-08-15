from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

import mub.vnext.preparation.task12 as task12


_SHA256 = "a" * 64
_CELL_IDS = (
    "raw-add-chronological-none-k04",
    "raw-add-chronological-none-k08",
    "raw-add-chronological-none-k16",
    "raw-add-reverse-none-k04",
    "raw-add-reverse-none-k08",
    "raw-add-reverse-none-k16",
    "raw-add-reverse-version-labeled-k04",
    "raw-add-reverse-version-labeled-k08",
    "raw-add-reverse-version-labeled-k16",
)
_SLOTS = ("answer_model_a", "answer_model_b")


def _answer_binding(slot_id: str) -> task12.Task11AnswerModelBindingV1:
    from mub.vnext.contracts.common import ArtifactRef

    path = "task11/qualification_summary.json"
    location = task12.Task12ArtifactLocationV1(
        root="evidence",
        artifact=ArtifactRef(
            path=path,
            sha256=_SHA256,
            media_type="application/json",
            record_count=1,
        ),
        relative_path=path,
    )
    if slot_id == "answer_model_a":
        return task12.Task11AnswerModelBindingV1(
            slot_id=slot_id,
            qualification_report=location,
            qualification_report_sha256=_SHA256,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="b" * 40,
            license_id="apache-2.0",
            tree_manifest_sha256="c" * 64,
            decoding_config_sha256="d" * 64,
        )
    return task12.Task11AnswerModelBindingV1(
        slot_id=slot_id,
        qualification_report=location,
        qualification_report_sha256=_SHA256,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="e" * 40,
        license_id="apache-2.0",
        tree_manifest_sha256="f" * 64,
        decoding_config_sha256="d" * 64,
    )


def _main_policy() -> task12.Task12MainManagerPolicyV1:
    return task12.Task12MainManagerPolicyV1(
        manager_ids=(
            "reference",
            "raw_add",
            "exact_crud",
            "heuristic_crud",
            "mem0_oss",
        ),
        task_split="test",
        task_count=2400,
        task_selection_sha256="1" * 64,
        one_terminal_row_per_requested_task=True,
        unsupported_policy="explicit_terminal_row_with_reason",
        reference_sanity_required=True,
        excluded_from_intervention_matrix=True,
    )


def _admitted_cells() -> tuple[task12.Task12AdmittedCellV1, ...]:
    return tuple(
        task12.Task12AdmittedCellV1(
            cell_id=cell_id,
            scope_id="core-hard-v1-family-a",
            canonical_binding_sha256=f"{index + 1:064x}",
        )
        for index, cell_id in enumerate(_CELL_IDS)
    )


def _answer_runs(
    cells: tuple[task12.Task12AdmittedCellV1, ...],
) -> tuple[task12.Task12AdmittedAnswerRunV1, ...]:
    return tuple(
        task12.Task12AdmittedAnswerRunV1(
            cell_id=cell.cell_id,
            answer_model_slot=slot,
            cell_binding_sha256=cell.canonical_binding_sha256,
            answer_model_binding_sha256=("a" if slot.endswith("a") else "b")
            * 64,
            canonical_run_binding_sha256=(
                f"{2 * index + slot_index + 100:064x}"
            ),
        )
        for index, cell in enumerate(cells)
        for slot_index, slot in enumerate(_SLOTS)
    )


def test_task12_main_manager_policy_is_separate_and_exact() -> None:
    policy = _main_policy()

    assert policy.manager_ids == (
        "reference",
        "raw_add",
        "exact_crud",
        "heuristic_crud",
        "mem0_oss",
    )
    assert policy.task_count == 2400
    assert policy.excluded_from_intervention_matrix is True

    with pytest.raises(ValidationError):
        task12.Task12MainManagerPolicyV1(
            **policy.model_dump(mode="python")
            | {"manager_ids": policy.manager_ids[:-1]}
        )


def test_task12_dry_plan_requires_three_distinct_scope_receipts_and_18_runs() -> None:
    cells = _admitted_cells()
    runs = _answer_runs(cells)
    plan = task12.Task12DryRunPlanV1(
        run_id="task12-dry-run",
        plan_fingerprint_sha256=_SHA256,
        core_task_manifest_sha256=_SHA256,
        core_hard_suite_sha256=_SHA256,
        core_tasks_sha256=_SHA256,
        scientific_design_sha256=_SHA256,
        semantic_matrix_sha256=_SHA256,
        main_manager_policy_sha256=_SHA256,
        answer_model_slots=_SLOTS,
        answer_model_binding_sha256=("a" * 64, "b" * 64),
        admitted_cells=cells,
        admitted_answer_runs=runs,
        hard_source_task_count=240,
        hard_source_task_selection_sha256="2" * 64,
        matrix_task_count=80,
        matrix_task_selection_sha256="3" * 64,
        main_test_task_count=2400,
        main_test_task_selection_sha256="1" * 64,
        output_leaf="task12-dry-run",
        code_revision="c" * 40,
        code_tree_sha256="d" * 64,
    )

    assert len(plan.admitted_cells) == 9
    assert len(plan.admitted_answer_runs) == 18
    assert tuple(
        (run.cell_id, run.answer_model_slot)
        for run in plan.admitted_answer_runs
    ) == tuple(product(_CELL_IDS, _SLOTS))
    assert (
        plan.hard_source_task_count,
        plan.matrix_task_count,
        plan.main_test_task_count,
    ) == (240, 80, 2400)
    assert plan.execution_authorized is False


def test_task12_dry_plan_rejects_missing_or_duplicate_answer_run() -> None:
    cells = _admitted_cells()
    runs = _answer_runs(cells)
    base = {
        "run_id": "task12-dry-run",
        "plan_fingerprint_sha256": _SHA256,
        "core_task_manifest_sha256": _SHA256,
        "core_hard_suite_sha256": _SHA256,
        "core_tasks_sha256": _SHA256,
        "scientific_design_sha256": _SHA256,
        "semantic_matrix_sha256": _SHA256,
        "main_manager_policy_sha256": _SHA256,
        "answer_model_slots": _SLOTS,
        "answer_model_binding_sha256": ("a" * 64, "b" * 64),
        "admitted_cells": cells,
        "hard_source_task_count": 240,
        "hard_source_task_selection_sha256": "2" * 64,
        "matrix_task_count": 80,
        "matrix_task_selection_sha256": "3" * 64,
        "main_test_task_count": 2400,
        "main_test_task_selection_sha256": "1" * 64,
        "output_leaf": "task12-dry-run",
        "code_revision": "c" * 40,
        "code_tree_sha256": "d" * 64,
    }

    for invalid_runs in (runs[:-1], (*runs[:-1], runs[0])):
        with pytest.raises(ValidationError, match="18"):
            task12.Task12DryRunPlanV1(
                **base,
                admitted_answer_runs=invalid_runs,
            )


def test_task12_manifest_has_one_matrix_authority_and_two_answer_bindings() -> None:
    fields = task12.Task12PreparationManifestV1.model_fields

    assert "scientific_design" in fields
    assert "semantic_matrix" in fields
    assert "answer_models" in fields
    assert "main_manager_policy" in fields
    assert "task10_mem0_admission" in fields
    assert "answer_model" not in fields
    assert "adapter_cells" not in fields
    assert "raw_append_intervention" not in fields
