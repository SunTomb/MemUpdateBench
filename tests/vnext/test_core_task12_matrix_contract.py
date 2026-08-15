from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import ArtifactRef
import mub.vnext.preparation.task12 as task12


_A_IDS = tuple(f"a-{index:03d}" for index in range(80))
_SHA256 = "a" * 64
_CONDITIONS = (
    ("chronological", "none"),
    ("reverse_chronological", "none"),
    ("reverse_chronological", "latest_outdated_label"),
)


def _location(path: str) -> task12.Task12ArtifactLocationV1:
    return task12.Task12ArtifactLocationV1(
        root="evidence",
        artifact=ArtifactRef(
            path=path,
            sha256=_SHA256,
            media_type="application/json",
            record_count=1,
        ),
        relative_path=path,
    )


def _trajectory() -> task12.Task12ArtifactLocationV1:
    path = "raw_add/trajectories.jsonl"
    return task12.Task12ArtifactLocationV1(
        root="evidence",
        artifact=ArtifactRef(
            path=path,
            sha256=_SHA256,
            media_type="application/x-ndjson",
            record_count=80,
        ),
        relative_path=path,
    )


def _design() -> task12.Task12ScientificDesignV1:
    return task12.Task12ScientificDesignV1(
        matrix_scope="core-hard-v1-family-a",
        matrix_adapter_id="raw_add",
        context_conditions=tuple(
            task12.Task12ContextInterventionV1(
                context_order=order,
                context_annotation=annotation,
            )
            for order, annotation in _CONDITIONS
        ),
        retrieval_policy="normal_topk",
        retrieval_k_values=(4, 8, 16),
        answer_model_slots=("answer_model_a", "answer_model_b"),
        label_reference_scope="full_raw_trajectory",
        transformation_order=(
            "frozen_raw_trajectory",
            "normal_topk",
            "presentation_order",
            "full_trajectory_version_labels",
        ),
        same_k_retrieved_entry_multiset=True,
        main_manager_ids=(
            "reference",
            "raw_add",
            "exact_crud",
            "heuristic_crud",
            "mem0_oss",
        ),
        main_task_split="test",
        main_test_task_count=2400,
        one_terminal_row_per_requested_task=True,
        unsupported_policy="explicit_terminal_row_with_reason",
        reference_sanity_required=True,
    )


def _cell(
    order: str,
    annotation: str,
    retrieval_k: int,
) -> task12.Task12InterventionCellV1:
    condition_slug = (
        "chronological-none"
        if (order, annotation) == ("chronological", "none")
        else "reverse-none"
        if annotation == "none"
        else "reverse-version-labeled"
    )
    return task12.Task12InterventionCellV1(
        cell_id=f"raw-add-{condition_slug}-k{retrieval_k:02d}",
        scope_id="core-hard-v1-family-a",
        task_ids=_A_IDS,
        context_intervention=task12.Task12ContextInterventionV1(
            context_order=order,
            context_annotation=annotation,
        ),
        adapter_configuration=_location("adapters/raw_add/config.json"),
        adapter_info=_location("adapters/raw_add/info.json"),
        capability_verification=_location("adapters/raw_add/capability.json"),
        retrieval=task12.Task12RetrievalBindingV1(
            configuration=task12.Task12RetrievalConfigurationV1(
                retrieval_policy="normal_topk",
                retrieval_k=retrieval_k,
            ),
            artifact=_location(
                f"adapters/raw_add/retrieval-k{retrieval_k}.json"
            ),
        ),
    )


def _matrix() -> task12.Task12SemanticMatrixV1:
    cells = tuple(
        _cell(order, annotation, retrieval_k)
        for (order, annotation), retrieval_k in product(
            _CONDITIONS,
            (4, 8, 16),
        )
    )
    return task12.Task12SemanticMatrixV1(
        scientific_design=_design(),
        task_scope=task12.Task12CoreTaskScopeV1(
            scope_id="core-hard-v1-family-a",
            family_ids=("repeated_same_slot_update",),
            task_ids=_A_IDS,
        ),
        intervention_cells=cells,
        raw_append_intervention=task12.RawAppendInterventionV1(
            trajectory_artifact=_trajectory(),
            task_ids=_A_IDS,
        ),
    )


def test_task12_matrix_is_exact_row_major_raw_family_a_cartesian_product() -> None:
    matrix = _matrix()

    assert tuple(
        (
            cell.context_intervention.context_order,
            cell.context_intervention.context_annotation,
            cell.retrieval.configuration.retrieval_k,
        )
        for cell in matrix.intervention_cells
    ) == tuple(
        (order, annotation, retrieval_k)
        for (order, annotation), retrieval_k in product(
            _CONDITIONS,
            (4, 8, 16),
        )
    )
    assert {cell.adapter_id for cell in matrix.intervention_cells} == {
        "raw_add"
    }
    assert {cell.scope_id for cell in matrix.intervention_cells} == {
        "core-hard-v1-family-a"
    }


@pytest.mark.parametrize("retrieval_k", (1, 7, 32))
def test_task12_retrieval_contract_rejects_nonapproved_k(
    retrieval_k: int,
) -> None:
    with pytest.raises(ValidationError):
        task12.Task12RetrievalConfigurationV1(
            retrieval_policy="normal_topk",
            retrieval_k=retrieval_k,
        )


def _unsafe_replace(cell, **changes):
    return type(cell).model_construct(**{**cell.__dict__, **changes})


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda cells: cells[:-1],
            "exact row-major",
        ),
        (
            lambda cells: (cells[1], cells[0], *cells[2:]),
            "exact row-major",
        ),
        (
            lambda cells: (
                cells[0],
                _unsafe_replace(cells[0], cell_id=cells[1].cell_id),
                *cells[2:],
            ),
            "coordinate",
        ),
        (
            lambda cells: (
                _unsafe_replace(cells[0], adapter_id="mem0_oss"),
                *cells[1:],
            ),
            "raw_add",
        ),
        (
            lambda cells: (
                _unsafe_replace(cells[0], scope_id="family-f"),
                *cells[1:],
            ),
            "Family A",
        ),
    ),
)
def test_task12_matrix_rejects_missing_duplicate_reordered_or_nonraw_cells(
    mutate,
    match: str,
) -> None:
    matrix = _matrix()

    with pytest.raises(ValidationError, match=match):
        task12.Task12SemanticMatrixV1(
            scientific_design=matrix.scientific_design,
            task_scope=matrix.task_scope,
            intervention_cells=mutate(matrix.intervention_cells),
            raw_append_intervention=matrix.raw_append_intervention,
        )


def test_task12_matrix_requires_same_retrieval_binding_for_each_k() -> None:
    matrix = _matrix()
    cells = list(matrix.intervention_cells)
    cells[3] = cells[3].model_copy(
        update={
            "retrieval": cells[3].retrieval.model_copy(
                update={
                    "artifact": _location(
                        "adapters/raw_add/alternate-retrieval-k4.json"
                    )
                }
            )
        }
    )

    with pytest.raises(ValidationError, match="same retrieval binding"):
        task12.Task12SemanticMatrixV1(
            scientific_design=matrix.scientific_design,
            task_scope=matrix.task_scope,
            intervention_cells=tuple(cells),
            raw_append_intervention=matrix.raw_append_intervention,
        )
