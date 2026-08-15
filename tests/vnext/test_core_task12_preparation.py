from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import ArtifactRef
import mub.vnext.preparation.task12 as task12
from tests.vnext.task12_fixtures import (
    build_task12_inputs,
    build_task12_manifest,
)


_SHA256 = "a" * 64
_A_IDS = tuple(f"a-{index:03d}" for index in range(80))
_F_IDS = tuple(f"f-{index:03d}" for index in range(80))
_G_IDS = tuple(f"g-{index:03d}" for index in range(80))
_AFG_FAMILIES = (
    "repeated_same_slot_update",
    "current_historical_query",
    "long_horizon_memory_synthesis",
)


def _artifact(path: str) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=_SHA256,
        media_type="application/json",
        record_count=1,
    )


def _raw_trajectory() -> task12.Task12ArtifactLocationV1:
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


def _hard_subset() -> dict:
    return {
        "selection_policy_version": "core-hard-v1",
        "family_ids": _AFG_FAMILIES,
        "task_ids": tuple(sorted(_A_IDS + _F_IDS + _G_IDS)),
    }


def _authorize_fixture_release(monkeypatch, inputs, manifest) -> None:
    monkeypatch.setattr(
        task12,
        "_repository_identity",
        lambda: ("c" * 40, "d" * 64),
    )
    monkeypatch.setattr(
        task12,
        "_APPROVED_CORE_RELEASE_MANIFEST_HASH",
        inputs["release_manifest_hash"],
    )
    monkeypatch.setattr(
        task12,
        "_APPROVED_CORE_RELEASE_ROOT_DIGEST",
        "f" * 64,
    )
    monkeypatch.setattr(
        task12,
        "_APPROVED_CORE_TASK_MANIFEST_SHA256",
        manifest.task_manifest.artifact.sha256,
    )
    monkeypatch.setattr(
        task12,
        "_APPROVED_CORE_HARD_SUITE_SHA256",
        manifest.core_hard_suite.artifact.sha256,
    )
    monkeypatch.setattr(
        task12,
        "_APPROVED_CORE_TASKS_SHA256",
        manifest.tasks.artifact.sha256,
    )


def test_task12_scope_contract_rejects_unsupported_partial_family_scope() -> None:
    with pytest.raises(ValidationError):
        task12.Task12CoreTaskScopeV1(
            scope_id="partial-af",
            family_ids=(
                "repeated_same_slot_update",
                "current_historical_query",
            ),
            task_ids=tuple(sorted(_A_IDS + _F_IDS)),
        )


@pytest.mark.parametrize(
    "path",
    (
        "../task_manifest.json",
        "nested/../task_manifest.json",
        "nested\\file.json",
        "/absolute.json",
        "CON",
        "C:artifact.json",
        "artifact.json ",
        "artifact.json.",
        "​hidden.json",
    ),
)
def test_task12_artifact_locator_rejects_unsafe_relative_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        task12.Task12ArtifactLocationV1(
            root="core",
            artifact=_artifact(path),
            relative_path=path,
        )


def test_task12_hard_subset_requires_canonical_afg_240_ordered_ids() -> None:
    subset = task12.Task12HardSubsetV1(**_hard_subset())

    assert len(subset.task_ids) == 240
    assert subset.task_ids == tuple(sorted(subset.task_ids))


@pytest.mark.parametrize(
    "change",
    (
        lambda value: value.update({"family_ids": value["family_ids"][::-1]}),
        lambda value: value.update({"task_ids": value["task_ids"][:-1]}),
        lambda value: value.update(
            {"task_ids": value["task_ids"][:-1] + (value["task_ids"][-2],)}
        ),
        lambda value: value.update(
            {"task_ids": tuple(reversed(value["task_ids"]))}
        ),
    ),
)
def test_task12_hard_subset_rejects_noncanonical_composition(change) -> None:
    payload = deepcopy(_hard_subset())
    change(payload)

    with pytest.raises(ValidationError):
        task12.Task12HardSubsetV1(**payload)


def test_raw_append_intervention_freezes_behavior_but_not_global_k() -> None:
    intervention = task12.RawAppendInterventionV1(
        trajectory_artifact=_raw_trajectory(),
        task_ids=_A_IDS,
    )

    assert intervention.on_add == "append"
    assert intervention.on_update == "append"
    assert intervention.on_noop == "no_write"
    assert "retrieval_k" not in type(intervention).model_fields
    assert "retrieval_policy" not in type(intervention).model_fields


@pytest.mark.parametrize(
    "change",
    (
        {"task_ids": _A_IDS[:-1]},
        {"on_update": "replace"},
        {"on_noop": "append"},
        {"append_only_observation": False},
    ),
)
def test_raw_append_intervention_rejects_semantic_drift(change) -> None:
    payload = {
        "trajectory_artifact": _raw_trajectory(),
        "task_ids": _A_IDS,
    }
    payload.update(change)

    with pytest.raises(ValidationError):
        task12.RawAppendInterventionV1(**payload)


def test_task12_admission_rejects_missing_external_roots(tmp_path) -> None:
    with pytest.raises(ValueError, match="Core root"):
        task12.admit_task12_dry_run(
            manifest={},
            core_root=tmp_path / "missing-core",
            evidence_root=tmp_path / "missing-evidence",
            output_dir=tmp_path,
        )


def test_task12_admission_authenticates_three_scopes_and_writes_nothing(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    _authorize_fixture_release(monkeypatch, inputs, manifest)
    before = tuple(
        sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    )

    plan = task12.admit_task12_dry_run(
        manifest=manifest,
        core_root=inputs["core_root"],
        evidence_root=inputs["evidence_root"],
        output_dir=tmp_path,
    )

    after = tuple(
        sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    )
    assert after == before
    assert (
        plan.hard_source_task_count,
        plan.matrix_task_count,
        plan.main_test_task_count,
    ) == (240, 80, 2400)
    assert len(plan.admitted_cells) == 9
    assert len(plan.admitted_answer_runs) == 18
    assert plan.answer_model_slots == ("answer_model_a", "answer_model_b")
    assert len(set(plan.answer_model_binding_sha256)) == 2
    assert len(
        {cell.canonical_binding_sha256 for cell in plan.admitted_cells}
    ) == 9
    assert len(
        {run.canonical_run_binding_sha256 for run in plan.admitted_answer_runs}
    ) == 18
    assert plan.main_test_task_selection_sha256 == (
        manifest.main_manager_policy.task_selection_sha256
    )


def test_task12_rejects_noncanonical_task11_qualification_artifact(
    tmp_path,
) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    binding = manifest.answer_models[0]
    qualification_path = inputs["evidence_root"] / binding.qualification_report.relative_path
    raw = qualification_path.read_bytes()
    pretty = json.dumps(json.loads(raw), indent=2).encode("utf-8")
    qualification_path.write_bytes(pretty)
    artifact = binding.qualification_report.artifact.model_copy(
        update={"sha256": hashlib.sha256(pretty).hexdigest()}
    )
    updated_binding = binding.model_copy(
        update={
            "qualification_report": binding.qualification_report.model_copy(
                update={"artifact": artifact}
            ),
            "qualification_report_sha256": artifact.sha256,
        }
    )

    with pytest.raises(ValueError, match="canonical"):
        task12._validate_answer_model_evidence(
            binding=updated_binding,
            evidence_root=inputs["evidence_root"],
        )


@pytest.mark.parametrize("mutation", ("status", "report_hash", "evaluation_hash"))
def test_task12_rejects_non_admitted_or_mismatched_mem0_decision(
    tmp_path,
    mutation,
) -> None:
    inputs = build_task12_inputs(tmp_path)
    manifest = build_task12_manifest(inputs)
    binding = manifest.task10_mem0_admission
    decision_path = inputs["evidence_root"] / binding.decision.relative_path
    payload = json.loads(decision_path.read_bytes())
    if mutation == "status":
        payload["status"] = "release_stopped"
        payload["admitted_report"] = None
        payload["reasons"] = ["no_candidate_admitted"]
    elif mutation == "evaluation_hash":
        payload["evaluation_configuration_hash"] = "1" * 64
    else:
        payload["admitted_report"]["report_hash"] = "0" * 64
    updated_raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    decision_path.write_bytes(updated_raw)
    updated_binding = binding.model_copy(
        update={
            "decision": binding.decision.model_copy(
                update={
                    "artifact": binding.decision.artifact.model_copy(
                        update={
                            "sha256": hashlib.sha256(updated_raw).hexdigest()
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="Task 10"):
        task12._validate_task10_mem0_admission(
            binding=updated_binding,
            evidence_root=inputs["evidence_root"],
            source_task_manifest_sha256=manifest.task_manifest.artifact.sha256,
        )