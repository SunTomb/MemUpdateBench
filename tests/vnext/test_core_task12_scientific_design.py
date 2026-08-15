from __future__ import annotations

from pathlib import Path

from mub.vnext.io import canonical_json_bytes
import mub.vnext.preparation.task12 as task12


ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = ROOT / "configs" / "vnext" / "core_task12_scientific_design.json"


def test_task12_scientific_design_authority_is_exact_and_canonical() -> None:
    design_type = getattr(task12, "Task12ScientificDesignV1", None)
    assert design_type is not None
    assert DESIGN_PATH.is_file()

    raw = DESIGN_PATH.read_bytes()
    design = design_type.model_validate_json(raw)

    assert canonical_json_bytes(design) == raw
    assert design.context_conditions == (
        task12.Task12ContextInterventionV1(
            context_order="chronological",
            context_annotation="none",
        ),
        task12.Task12ContextInterventionV1(
            context_order="reverse_chronological",
            context_annotation="none",
        ),
        task12.Task12ContextInterventionV1(
            context_order="reverse_chronological",
            context_annotation="latest_outdated_label",
        ),
    )
    assert design.retrieval_k_values == (4, 8, 16)
    assert design.answer_model_slots == ("answer_model_a", "answer_model_b")
    assert design.main_manager_ids == (
        "reference",
        "raw_add",
        "exact_crud",
        "heuristic_crud",
        "mem0_oss",
    )
    assert design.execution_authorized is False
