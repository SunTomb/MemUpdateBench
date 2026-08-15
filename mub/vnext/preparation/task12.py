from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from mub.vnext.contracts.common import ArtifactRef, ImmutableContractModel
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.common import StrictIdentifier
from mub.vnext.contracts.v3.manifest import TaskManifestV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.contracts import (
    AdmissionDecisionStatus,
    AdmissionDecisionV1,
    ExternalAdmissionReportV1,
)
from mub.vnext.external.registry import (
    _validate_portable_path,
    validate_artifact_provenance,
)
from mub.vnext.generation.core_hard_suite import CoreHardSuiteManifest
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.mechanisms.context import (
    APPROVED_CONTEXT_CONDITIONS,
    ContextAnnotation,
    ContextOrder,
)
from mub.vnext.runtime.answer_model_v3 import DeterministicDecodeConfigV3
from mub.vnext.runtime.support_v3 import resolve_task_support_v3


_TASK12_PREPARATION_SCHEMA_VERSION = "memupdatebench.core-task12-preparation.v1"
_TASK12_SCIENTIFIC_DESIGN_SCHEMA_VERSION = (
    "memupdatebench.core-task12-scientific-design.v1"
)
_APPROVED_CORE_RELEASE_MANIFEST_HASH = "f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d"
_APPROVED_CORE_RELEASE_ROOT_DIGEST = "458d169a4732139f45361d90ea528f5ed0133f126a32bc5a16de23da6f8a2aba"
_APPROVED_CORE_TASK_MANIFEST_SHA256 = "38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3"
_APPROVED_CORE_HARD_SUITE_SHA256 = "ae4ff033857c7145115612a214ecbbbfd91c4ff37f60cf68400208dd4191044c"
_APPROVED_CORE_TASKS_SHA256 = "5c4fd518542b0665d7313d68f1a339de38502c376aa93fbda228196587cdd2c6"
_AFG_FAMILIES = (
    "repeated_same_slot_update",
    "current_historical_query",
    "long_horizon_memory_synthesis",
)
_EXTERNAL_ADAPTER_IDS = frozenset({
    "mem0_oss",
    "langgraph_store_extract_then_store",
})


def _canonical_relative_path(value: str) -> str:
    return _validate_portable_path(value)


class Task12ArtifactLocationV1(ImmutableContractModel):
    root: Literal["core", "evidence"]
    artifact: ArtifactRef
    relative_path: str

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return _canonical_relative_path(value)

    @model_validator(mode="after")
    def _validate_reference_path(self):
        validate_artifact_provenance(self.artifact)
        if self.artifact.path != self.relative_path:
            raise ValueError("artifact reference path must equal relative_path")
        return self


class Task12HardSubsetV1(ImmutableContractModel):
    selection_policy_version: Literal["core-hard-v1"] = "core-hard-v1"
    family_ids: tuple[str, str, str]
    task_ids: tuple[StrictIdentifier, ...]

    @model_validator(mode="after")
    def _validate_afg_hard_subset(self):
        if self.family_ids != _AFG_FAMILIES:
            raise ValueError("hard subset families must be canonical A/F/G order")
        if len(self.task_ids) != 240:
            raise ValueError("hard subset must contain exactly 240 task IDs")
        if self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError("hard subset task IDs must be sorted")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("hard subset task IDs must be unique")
        return self


class RawAppendInterventionV1(ImmutableContractModel):
    adapter_id: Literal["raw_add"] = "raw_add"
    trajectory_artifact: Task12ArtifactLocationV1
    task_ids: tuple[StrictIdentifier, ...]
    append_only_observation: Literal[True] = True
    on_add: Literal["append"] = "append"
    on_update: Literal["append"] = "append"
    on_noop: Literal["no_write"] = "no_write"

    @model_validator(mode="after")
    def _validate_complete_canonical_scope(self):
        if (
            self.trajectory_artifact.root != "evidence"
            or self.trajectory_artifact.artifact.media_type != "application/x-ndjson"
            or self.trajectory_artifact.artifact.record_count != 80
        ):
            raise ValueError("raw append trajectories must be an 80-record evidence artifact")
        if len(self.task_ids) != 80:
            raise ValueError("raw append intervention must declare 80 Family A task IDs")
        if self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError("raw append task IDs must be sorted")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("raw append task IDs must be unique")
        return self


class RawAppendTrajectoryV1(ImmutableContractModel):
    task_id: StrictIdentifier
    entry_ids: tuple[StrictIdentifier, ...]
    object_ids: tuple[StrictIdentifier, ...]
    event_indices: tuple[int, ...]
    version_indices: tuple[int, ...]
    latest_entry_ids: tuple[StrictIdentifier, ...]
    trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    on_add: Literal["append"] = "append"
    on_update: Literal["append"] = "append"
    on_noop: Literal["no_write"] = "no_write"

    @model_validator(mode="after")
    def _validate_receipt_shape(self):
        if not self.entry_ids:
            raise ValueError("raw trajectory receipts require entries")
        if len(self.entry_ids) != len(set(self.entry_ids)):
            raise ValueError("raw trajectory entry IDs must be unique")
        if len(self.event_indices) != len(self.entry_ids):
            raise ValueError("raw trajectory event indices must cover every entry")
        if len(self.object_ids) != len(self.entry_ids):
            raise ValueError("raw trajectory object IDs must cover every entry")
        if len(self.version_indices) != len(self.entry_ids):
            raise ValueError("raw trajectory version indices must cover every entry")
        if any(index < 0 for index in (*self.event_indices, *self.version_indices)):
            raise ValueError("raw trajectory indices must be nonnegative")
        if any(
            left > right
            for left, right in zip(self.event_indices, self.event_indices[1:])
        ):
            raise ValueError("raw trajectory event order must be chronological")
        if not self.latest_entry_ids:
            raise ValueError("raw trajectory latest IDs must cover every object")
        if not set(self.latest_entry_ids) <= set(self.entry_ids):
            raise ValueError("raw trajectory latest IDs must be full-trajectory entries")
        if len(self.latest_entry_ids) != len(set(self.latest_entry_ids)):
            raise ValueError("raw trajectory latest IDs must be unique")
        latest_objects = {
            self.object_ids[self.entry_ids.index(entry_id)]
            for entry_id in self.latest_entry_ids
        }
        if (
            latest_objects != set(self.object_ids)
            or len(self.latest_entry_ids) != len(set(self.object_ids))
        ):
            raise ValueError("raw trajectory latest IDs must cover each object exactly once")
        latest_by_object: dict[str, str] = {}
        for object_id in set(self.object_ids):
            candidate_indices = [
                index
                for index, observed_object_id in enumerate(self.object_ids)
                if observed_object_id == object_id
            ]
            latest_index = max(
                candidate_indices,
                key=lambda index: (
                    self.event_indices[index],
                    self.version_indices[index],
                    self.entry_ids[index],
                ),
            )
            latest_by_object[object_id] = self.entry_ids[latest_index]
        if set(self.latest_entry_ids) != set(latest_by_object.values()):
            raise ValueError("raw trajectory latest IDs must identify the final object versions")
        return self


class Task11QualificationSlotV1(ImmutableContractModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    slot_id: Literal["answer_model_a", "answer_model_b"]
    model_id: str = Field(min_length=1, strict=True)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", strict=True)
    license_id: Literal["apache-2.0"]
    tree_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)


class Task11OfflineContractV1(ImmutableContractModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    hf_hub_offline: Literal[True]
    transformers_offline: Literal[True]
    local_files_only: Literal[True]
    trust_remote_code: Literal[False]
    decoding: DeterministicDecodeConfigV3


class Task11QualificationReportV1(ImmutableContractModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    schema_version: Literal["memupdatebench.core-task11-answer-harness.v1"]
    status: Literal["qualified"]
    offline_contract: Task11OfflineContractV1
    slots: tuple[Task11QualificationSlotV1, ...]

    @model_validator(mode="after")
    def _unique_slots(self):
        slot_ids = tuple(slot.slot_id for slot in self.slots)
        if slot_ids != ("answer_model_a", "answer_model_b"):
            raise ValueError("Task 11 qualification must bind both frozen answer-model slots")
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("Task 11 qualification slots must be unique")
        return self


class Task11AnswerModelBindingV1(ImmutableContractModel):
    slot_id: Literal["answer_model_a", "answer_model_b"]
    qualification_report: Task12ArtifactLocationV1
    qualification_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    model_id: str = Field(min_length=1, strict=True)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", strict=True)
    license_id: Literal["apache-2.0"]
    tree_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    decoding_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)

    @model_validator(mode="after")
    def _validate_qualification_location(self):
        if self.qualification_report.root != "evidence":
            raise ValueError("Task 11 qualification report must be evidence-rooted")
        if self.qualification_report.artifact.sha256 != self.qualification_report_sha256:
            raise ValueError("Task 11 qualification report hash must bind its artifact")
        return self


class Task12CapabilityVerificationV1(ImmutableContractModel):
    adapter_id: StrictIdentifier
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    source_task_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    capabilities: AdapterCapabilitiesV3


class Task12CoreTaskScopeV1(ImmutableContractModel):
    scope_id: StrictIdentifier
    family_ids: tuple[str, ...]
    task_ids: tuple[StrictIdentifier, ...]

    @model_validator(mode="after")
    def _validate_scope_shape(self):
        if self.family_ids not in {
            (_AFG_FAMILIES[0],),
            (_AFG_FAMILIES[1],),
            (_AFG_FAMILIES[2],),
            _AFG_FAMILIES,
        }:
            raise ValueError("Task 12 scopes must be A, F, G, or complete A/F/G")
        if not self.task_ids or self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError("Task 12 scope task IDs must be nonempty and sorted")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("Task 12 scope task IDs must be unique")
        return self


class Task12ContextInterventionV1(ImmutableContractModel):
    context_order: ContextOrder
    context_annotation: ContextAnnotation

    @model_validator(mode="after")
    def _validate_approved_condition(self):
        if (
            self.context_order,
            self.context_annotation,
        ) not in APPROVED_CONTEXT_CONDITIONS:
            raise ValueError("Task 12 context intervention is not approved")
        return self


class Task12ScientificDesignV1(ImmutableContractModel):
    schema_version: Literal[_TASK12_SCIENTIFIC_DESIGN_SCHEMA_VERSION] = (
        _TASK12_SCIENTIFIC_DESIGN_SCHEMA_VERSION
    )
    matrix_scope: Literal["core-hard-v1-family-a"]
    matrix_adapter_id: Literal["raw_add"]
    context_conditions: tuple[
        Task12ContextInterventionV1,
        Task12ContextInterventionV1,
        Task12ContextInterventionV1,
    ]
    retrieval_policy: Literal["normal_topk"]
    retrieval_k_values: tuple[Literal[4], Literal[8], Literal[16]]
    answer_model_slots: tuple[
        Literal["answer_model_a"],
        Literal["answer_model_b"],
    ]
    label_reference_scope: Literal["full_raw_trajectory"]
    transformation_order: tuple[
        Literal["frozen_raw_trajectory"],
        Literal["normal_topk"],
        Literal["presentation_order"],
        Literal["full_trajectory_version_labels"],
    ]
    same_k_retrieved_entry_multiset: Literal[True]
    main_manager_ids: tuple[
        Literal["reference"],
        Literal["raw_add"],
        Literal["exact_crud"],
        Literal["heuristic_crud"],
        Literal["mem0_oss"],
    ]
    main_task_split: Literal["test"]
    main_test_task_count: Literal[2400]
    one_terminal_row_per_requested_task: Literal[True]
    unsupported_policy: Literal["explicit_terminal_row_with_reason"]
    reference_sanity_required: Literal[True]
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _validate_exact_design(self):
        observed = tuple(
            (condition.context_order, condition.context_annotation)
            for condition in self.context_conditions
        )
        if observed != APPROVED_CONTEXT_CONDITIONS:
            raise ValueError("Task 12 context conditions must use approved row order")
        return self


class Task12RetrievalConfigurationV1(ImmutableContractModel):
    retrieval_policy: Literal["normal_topk"]
    retrieval_k: Literal[4, 8, 16]


class Task12RetrievalBindingV1(ImmutableContractModel):
    configuration: Task12RetrievalConfigurationV1
    artifact: Task12ArtifactLocationV1

    @model_validator(mode="after")
    def _validate_artifact(self):
        if (
            self.artifact.root != "evidence"
            or self.artifact.artifact.media_type != "application/json"
            or self.artifact.artifact.record_count != 1
        ):
            raise ValueError("retrieval configuration must be one evidence JSON artifact")
        return self


def _task12_cell_id(
    context: Task12ContextInterventionV1,
    retrieval_k: int,
) -> str:
    condition = (
        "chronological-none"
        if (
            context.context_order,
            context.context_annotation,
        ) == ("chronological", "none")
        else "reverse-none"
        if context.context_annotation == "none"
        else "reverse-version-labeled"
    )
    return f"raw-add-{condition}-k{retrieval_k:02d}"


class Task12InterventionCellV1(ImmutableContractModel):
    cell_id: StrictIdentifier
    adapter_id: Literal["raw_add"] = "raw_add"
    adapter_kind: Literal["built_in"] = "built_in"
    scope_id: Literal["core-hard-v1-family-a"]
    task_ids: tuple[StrictIdentifier, ...]
    context_intervention: Task12ContextInterventionV1
    adapter_configuration: Task12ArtifactLocationV1
    adapter_info: Task12ArtifactLocationV1
    capability_verification: Task12ArtifactLocationV1
    retrieval: Task12RetrievalBindingV1

    @model_validator(mode="after")
    def _validate_cell_bindings(self):
        if len(self.task_ids) != 80:
            raise ValueError("Task 12 intervention cells require 80 Family A tasks")
        if self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError("Task 12 intervention task IDs must be sorted")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("Task 12 intervention task IDs must be unique")
        if self.cell_id != _task12_cell_id(
            self.context_intervention,
            self.retrieval.configuration.retrieval_k,
        ):
            raise ValueError("Task 12 cell ID must be derived from its exact coordinate")
        locations = (
            self.adapter_configuration,
            self.adapter_info,
            self.capability_verification,
        )
        if any(location.root != "evidence" for location in locations):
            raise ValueError("intervention cell evidence must be evidence-rooted")
        return self


def _canonical_cell_binding_hash(
    *,
    cell: Task12InterventionCellV1,
    scope: Task12CoreTaskScopeV1,
    raw_append_intervention: RawAppendInterventionV1,
) -> str:
    payload = {
        "scope": {"family_ids": scope.family_ids, "task_ids": scope.task_ids},
        "adapter_id": cell.adapter_id,
        "adapter_kind": cell.adapter_kind,
        "adapter_configuration_sha256": cell.adapter_configuration.artifact.sha256,
        "adapter_info_sha256": cell.adapter_info.artifact.sha256,
        "capability_verification_sha256": cell.capability_verification.artifact.sha256,
        "retrieval_configuration": cell.retrieval.configuration.model_dump(mode="json"),
        "retrieval_configuration_sha256": cell.retrieval.artifact.artifact.sha256,
        "context_intervention": cell.context_intervention.model_dump(mode="json"),
        "label_reference_scope": "full_raw_trajectory",
        "raw_append_behavior": {
            "on_add": raw_append_intervention.on_add,
            "on_update": raw_append_intervention.on_update,
            "on_noop": raw_append_intervention.on_noop,
        },
        "raw_append_trajectory_sha256": (
            raw_append_intervention.trajectory_artifact.artifact.sha256
        ),
    }
    return _canonical_json_sha256(payload)


class Task12SemanticMatrixV1(ImmutableContractModel):
    scientific_design: Task12ScientificDesignV1
    task_scope: Task12CoreTaskScopeV1
    intervention_cells: tuple[Task12InterventionCellV1, ...]
    raw_append_intervention: RawAppendInterventionV1

    @model_validator(mode="after")
    def _validate_matrix_structure(self):
        if (
            self.task_scope.scope_id != self.scientific_design.matrix_scope
            or self.task_scope.family_ids != (_AFG_FAMILIES[0],)
            or len(self.task_scope.task_ids) != 80
        ):
            raise ValueError("Task 12 matrix requires the exact Family A scope")
        if self.raw_append_intervention.task_ids != self.task_scope.task_ids:
            raise ValueError("raw trajectory and matrix must bind the same Family A tasks")
        if any(cell.adapter_id != "raw_add" for cell in self.intervention_cells):
            raise ValueError("Task 12 intervention cells must all use raw_add")
        if any(
            cell.scope_id != self.task_scope.scope_id
            or cell.task_ids != self.task_scope.task_ids
            for cell in self.intervention_cells
        ):
            raise ValueError("Task 12 cells must use the exact Family A scope")
        expected_coordinates = tuple(
            (order, annotation, retrieval_k)
            for order, annotation in APPROVED_CONTEXT_CONDITIONS
            for retrieval_k in self.scientific_design.retrieval_k_values
        )
        observed_coordinates = tuple(
            (
                cell.context_intervention.context_order,
                cell.context_intervention.context_annotation,
                cell.retrieval.configuration.retrieval_k,
            )
            for cell in self.intervention_cells
        )
        expected_ids = tuple(
            _task12_cell_id(
                Task12ContextInterventionV1(
                    context_order=order,
                    context_annotation=annotation,
                ),
                retrieval_k,
            )
            for order, annotation, retrieval_k in expected_coordinates
        )
        observed_ids = tuple(cell.cell_id for cell in self.intervention_cells)
        if (
            observed_coordinates != expected_coordinates
            or observed_ids != expected_ids
        ):
            raise ValueError("Task 12 matrix must use the exact row-major 3x3 coordinates")
        static_bindings = {
            (
                cell.adapter_configuration,
                cell.adapter_info,
                cell.capability_verification,
            )
            for cell in self.intervention_cells
        }
        if len(static_bindings) != 1:
            raise ValueError("all Task 12 cells must share one frozen raw adapter binding")
        by_k: dict[int, set[Task12RetrievalBindingV1]] = {}
        for cell in self.intervention_cells:
            by_k.setdefault(
                cell.retrieval.configuration.retrieval_k,
                set(),
            ).add(cell.retrieval)
        if any(len(bindings) != 1 for bindings in by_k.values()):
            raise ValueError("each k must use the same retrieval binding across rows")
        return self

    @property
    def task_scopes(self) -> tuple[Task12CoreTaskScopeV1, ...]:
        return (self.task_scope,)

    @property
    def adapter_cells(self) -> tuple[Task12InterventionCellV1, ...]:
        return self.intervention_cells


Task12AdapterCellV1 = Task12InterventionCellV1


class Task12MainManagerPolicyV1(ImmutableContractModel):
    manager_ids: tuple[
        Literal["reference"],
        Literal["raw_add"],
        Literal["exact_crud"],
        Literal["heuristic_crud"],
        Literal["mem0_oss"],
    ]
    task_split: Literal["test"]
    task_count: Literal[2400]
    task_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    one_terminal_row_per_requested_task: Literal[True]
    unsupported_policy: Literal["explicit_terminal_row_with_reason"]
    reference_sanity_required: Literal[True]
    excluded_from_intervention_matrix: Literal[True]

    @model_validator(mode="after")
    def _validate_exact_policy(self):
        if self.manager_ids != (
            "reference",
            "raw_add",
            "exact_crud",
            "heuristic_crud",
            "mem0_oss",
        ):
            raise ValueError("main manager policy must use the exact manager order")
        return self


class Task12AdmittedCellV1(ImmutableContractModel):
    cell_id: StrictIdentifier
    scope_id: Literal["core-hard-v1-family-a"]
    canonical_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)


class Task12AdmittedAnswerRunV1(ImmutableContractModel):
    cell_id: StrictIdentifier
    answer_model_slot: Literal["answer_model_a", "answer_model_b"]
    cell_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    answer_model_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    canonical_run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)


class Task12ExternalAdmissionBindingV1(ImmutableContractModel):
    decision: Task12ArtifactLocationV1
    report: Task12ArtifactLocationV1

    @model_validator(mode="after")
    def _validate_evidence_roots(self):
        if self.decision.root != "evidence" or self.report.root != "evidence":
            raise ValueError("Task 10 admission evidence must be evidence-rooted")
        return self


class Task12PreparationManifestV1(ImmutableContractModel):
    schema_version: Literal[_TASK12_PREPARATION_SCHEMA_VERSION] = _TASK12_PREPARATION_SCHEMA_VERSION
    run_id: StrictIdentifier
    release_manifest: Task12ArtifactLocationV1
    task_manifest: Task12ArtifactLocationV1
    core_hard_suite: Task12ArtifactLocationV1
    tasks: Task12ArtifactLocationV1
    hard_subset: Task12HardSubsetV1
    scientific_design: Task12ScientificDesignV1
    answer_models: tuple[
        Task11AnswerModelBindingV1,
        Task11AnswerModelBindingV1,
    ]
    semantic_matrix: Task12SemanticMatrixV1
    main_manager_policy: Task12MainManagerPolicyV1
    task10_mem0_admission: Task12ExternalAdmissionBindingV1
    output_leaf: str

    @field_validator("output_leaf")
    @classmethod
    def _validate_output_leaf(cls, value: str) -> str:
        validated = _canonical_relative_path(value)
        if len(PurePosixPath(validated).parts) != 1:
            raise ValueError("output leaf must be a single path component")
        return validated

    @model_validator(mode="after")
    def _validate_frozen_design(self):
        if any(
            location.root != "core"
            for location in (
                self.release_manifest,
                self.task_manifest,
                self.core_hard_suite,
                self.tasks,
            )
        ):
            raise ValueError("Core task inputs must be core-rooted")
        if self.semantic_matrix.scientific_design != self.scientific_design:
            raise ValueError("semantic matrix must bind the approved scientific design")
        slots = tuple(binding.slot_id for binding in self.answer_models)
        if slots != self.scientific_design.answer_model_slots:
            raise ValueError("Task 12 manifest must bind both answer-model slots in order")
        first, second = self.answer_models
        if (
            first.qualification_report != second.qualification_report
            or first.qualification_report_sha256
            != second.qualification_report_sha256
            or first.decoding_config_sha256 != second.decoding_config_sha256
        ):
            raise ValueError("both answer slots must share one qualification and decode contract")
        design = self.scientific_design
        policy = self.main_manager_policy
        if (
            policy.manager_ids != design.main_manager_ids
            or policy.task_split != design.main_task_split
            or policy.task_count != design.main_test_task_count
            or policy.one_terminal_row_per_requested_task
            != design.one_terminal_row_per_requested_task
            or policy.unsupported_policy != design.unsupported_policy
            or policy.reference_sanity_required != design.reference_sanity_required
        ):
            raise ValueError("main manager policy must match the approved scientific design")
        return self


class Task12DryRunPlanV1(ImmutableContractModel):
    schema_version: Literal[_TASK12_PREPARATION_SCHEMA_VERSION] = _TASK12_PREPARATION_SCHEMA_VERSION
    run_id: StrictIdentifier
    plan_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    core_task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    core_hard_suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    core_tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    scientific_design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    semantic_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    main_manager_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    answer_model_slots: tuple[
        Literal["answer_model_a"],
        Literal["answer_model_b"],
    ]
    answer_model_binding_sha256: tuple[str, str]
    admitted_cells: tuple[Task12AdmittedCellV1, ...]
    admitted_answer_runs: tuple[Task12AdmittedAnswerRunV1, ...]
    hard_source_task_count: Literal[240]
    hard_source_task_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    matrix_task_count: Literal[80]
    matrix_task_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    main_test_task_count: Literal[2400]
    main_test_task_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    output_leaf: str
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$", strict=True)
    code_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    execution_authorized: Literal[False] = False

    @field_validator("answer_model_binding_sha256")
    @classmethod
    def _validate_answer_binding_hashes(cls, value: tuple[str, str]):
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in value
        ):
            raise ValueError("answer-model binding hashes must be lowercase SHA-256")
        if len(set(value)) != 2:
            raise ValueError("answer-model binding hashes must be distinct")
        return value

    @model_validator(mode="after")
    def _validate_frozen_receipts(self):
        expected_cell_ids = tuple(
            _task12_cell_id(
                Task12ContextInterventionV1(
                    context_order=order,
                    context_annotation=annotation,
                ),
                retrieval_k,
            )
            for order, annotation in APPROVED_CONTEXT_CONDITIONS
            for retrieval_k in (4, 8, 16)
        )
        cell_ids = tuple(cell.cell_id for cell in self.admitted_cells)
        cell_hashes = tuple(
            cell.canonical_binding_sha256 for cell in self.admitted_cells
        )
        if (
            cell_ids != expected_cell_ids
            or len(set(cell_hashes)) != 9
        ):
            raise ValueError("dry-run plan must bind nine exact unique semantic cells")
        expected_runs = tuple(
            (cell_id, slot)
            for cell_id in expected_cell_ids
            for slot in self.answer_model_slots
        )
        observed_runs = tuple(
            (run.cell_id, run.answer_model_slot)
            for run in self.admitted_answer_runs
        )
        if len(self.admitted_answer_runs) != 18 or observed_runs != expected_runs:
            raise ValueError("dry-run plan must bind exactly 18 ordered answer runs")
        cell_hash_by_id = {
            cell.cell_id: cell.canonical_binding_sha256
            for cell in self.admitted_cells
        }
        answer_hash_by_slot = dict(
            zip(self.answer_model_slots, self.answer_model_binding_sha256)
        )
        if any(
            run.cell_binding_sha256 != cell_hash_by_id[run.cell_id]
            or run.answer_model_binding_sha256
            != answer_hash_by_slot[run.answer_model_slot]
            for run in self.admitted_answer_runs
        ):
            raise ValueError("answer-run receipts must bind their cell and answer slot")
        run_hashes = tuple(
            run.canonical_run_binding_sha256
            for run in self.admitted_answer_runs
        )
        if len(set(run_hashes)) != 18:
            raise ValueError("all 18 answer-run binding hashes must be unique")
        selection_hashes = {
            self.hard_source_task_selection_sha256,
            self.matrix_task_selection_sha256,
            self.main_test_task_selection_sha256,
        }
        if len(selection_hashes) != 3:
            raise ValueError("240/80/2400 scope receipts must remain distinct")
        return self


def _require_real_directory(root: str | Path, *, label: str) -> Path:
    path = Path(root).absolute()
    if not path.is_dir():
        raise ValueError(f"{label} root must be an existing real directory")
    current = Path(path.anchor)
    for part in path.relative_to(current).parts:
        current = current / part
        metadata = current.lstat()
        if current.is_symlink() or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"{label} root cannot traverse a symlink or reparse point")
    return path


def _validate_distinct_roots(core_root: Path, evidence_root: Path) -> None:
    if core_root == evidence_root or core_root in evidence_root.parents or evidence_root in core_root.parents:
        raise ValueError("Core and evidence roots must not overlap")


def _validate_core_artifact_locations(manifest: Task12PreparationManifestV1) -> None:
    if (
        manifest.task_manifest.artifact.sha256
        != _APPROVED_CORE_TASK_MANIFEST_SHA256
        or manifest.core_hard_suite.artifact.sha256
        != _APPROVED_CORE_HARD_SUITE_SHA256
        or manifest.tasks.artifact.sha256 != _APPROVED_CORE_TASKS_SHA256
    ):
        raise ValueError("task manifest does not match the approved immutable Core release")
    expected = (
        ("release manifest", manifest.release_manifest, "task_release_manifest.json", "application/json", 1),
        ("task manifest", manifest.task_manifest, "candidate/task_manifest.json", "application/json", 1),
        ("core hard suite", manifest.core_hard_suite, "candidate/core-hard-v1.json", "application/json", 1),
        ("tasks", manifest.tasks, "candidate/tasks.jsonl", "application/x-ndjson", 12000),
    )
    for label, location, path, media_type, record_count in expected:
        if (
            location.relative_path != path
            or location.artifact.path != path
            or location.artifact.media_type != media_type
            or location.artifact.record_count != record_count
        ):
            raise ValueError(f"{label} location must bind {path}")


def _read_artifact(
    *,
    root: Path,
    location: Task12ArtifactLocationV1,
) -> bytes:
    candidate = root.joinpath(*PurePosixPath(location.relative_path).parts)
    current = root
    for part in PurePosixPath(location.relative_path).parts:
        current = current / part
        metadata = current.lstat()
        if current.is_symlink() or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError("Task 12 artifact path cannot traverse a symlink or reparse point")
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError("Task 12 artifacts must be single-link regular files")
    payload = candidate.read_bytes()
    if hashlib.sha256(payload).hexdigest() != location.artifact.sha256:
        raise ValueError(f"Task 12 artifact digest mismatch: {location.relative_path}")
    return payload


def _canonical_json_model(raw: bytes, model_type, *, label: str):
    try:
        model = model_type.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if canonical_json_bytes(model) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return model


def _read_canonical_tasks(raw: bytes) -> tuple[MemUpdateTaskV3, ...]:
    if not raw.endswith(b"\n"):
        raise ValueError("tasks artifact must end with a newline")
    tasks: list[MemUpdateTaskV3] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"tasks artifact contains blank row {line_number}")
        try:
            task = MemUpdateTaskV3.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"tasks artifact contains invalid row {line_number}") from exc
        if canonical_json_bytes(task) != line:
            raise ValueError(f"tasks artifact row {line_number} is not canonical")
        if task.task_id in seen:
            raise ValueError("tasks artifact contains duplicate task IDs")
        seen.add(task.task_id)
        tasks.append(task)
    if tuple(task.task_id for task in tasks) != tuple(sorted(task.task_id for task in tasks)):
        raise ValueError("tasks artifact task IDs must be sorted")
    return tuple(tasks)


def _trajectory_receipt_for_task(task: MemUpdateTaskV3) -> RawAppendTrajectoryV1:
    events = {event.event_id: event for event in task.events}
    actions = {action.action_id: action for action in task.actions}
    versions: dict[str, int] = {}
    entries: list[dict[str, object]] = []
    for event in sorted(task.events, key=lambda item: item.sequence_index):
        for action_id in event.gold_action_ids:
            action = actions.get(action_id)
            if action is None:
                raise ValueError(f"event {event.event_id} references missing action {action_id}")
            operation = getattr(action.operation, "value", action.operation)
            if operation not in {"ADD", "UPDATE", "add", "update"}:
                continue
            for key in action.target_object_keys:
                key_id = key.canonical_id
                version_index = versions.get(key_id, 0)
                versions[key_id] = version_index + 1
                entries.append({
                    "entry_id": f"{event.event_id}:{key_id}:{version_index}",
                    "event_index": event.sequence_index,
                    "version_index": version_index,
                    "object_key": key_id,
                    "value": action.value,
                })
    if not entries:
        raise ValueError(f"task {task.task_id} has no raw trajectory entries")
    latest_by_key: dict[str, dict[str, object]] = {}
    for entry in entries:
        previous = latest_by_key.get(entry["object_key"])
        if previous is None or (
            entry["event_index"], entry["version_index"]
        ) > (previous["event_index"], previous["version_index"]):
            latest_by_key[entry["object_key"]] = entry
    latest_ids = {entry["entry_id"] for entry in latest_by_key.values()}
    trajectory_sha256 = hashlib.sha256(
        json.dumps(
            entries,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return RawAppendTrajectoryV1(
        task_id=task.task_id,
        entry_ids=tuple(entry["entry_id"] for entry in entries),
        object_ids=tuple(entry["object_key"] for entry in entries),
        event_indices=tuple(entry["event_index"] for entry in entries),
        version_indices=tuple(entry["version_index"] for entry in entries),
        latest_entry_ids=tuple(
            entry["entry_id"] for entry in entries if entry["entry_id"] in latest_ids
        ),
        trajectory_sha256=trajectory_sha256,
    )


def _read_raw_append_trajectories(
    raw: bytes,
    *,
    expected_task_ids: tuple[str, ...],
    expected_tasks: tuple[MemUpdateTaskV3, ...],
) -> None:
    if not raw.endswith(b"\n"):
        raise ValueError("raw append trajectory artifact must end with a newline")
    records: list[RawAppendTrajectoryV1] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"raw append trajectory artifact contains blank row {line_number}")
        try:
            record = RawAppendTrajectoryV1.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"raw append trajectory row {line_number} is invalid") from exc
        if canonical_json_bytes(record) != line:
            raise ValueError(f"raw append trajectory row {line_number} is not canonical")
        records.append(record)
    observed_ids = tuple(record.task_id for record in records)
    if observed_ids != expected_task_ids or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("raw append trajectories must cover ordered authenticated Family A task IDs exactly")
    expected_by_id = {
        task.task_id: _trajectory_receipt_for_task(task)
        for task in expected_tasks
    }
    if tuple(expected_by_id) != expected_task_ids:
        raise ValueError("raw trajectory validation tasks must match ordered Family A IDs")
    if any(
        record != expected_by_id[record.task_id]
        for record in records
    ):
        raise ValueError(
            "raw append trajectory receipt does not bind event order, versions, or full latest truth"
        )


def _selected_hard_tasks(
    *,
    task_manifest: TaskManifestV3,
    hard_suite: CoreHardSuiteManifest,
    tasks: tuple[MemUpdateTaskV3, ...],
) -> tuple[MemUpdateTaskV3, ...]:
    observed_hashes = {task.task_id: sha256_model(task) for task in tasks}
    if dict(task_manifest.task_record_hashes) != observed_hashes:
        raise ValueError("task manifest record hashes do not authenticate tasks")
    task_by_id = {task.task_id: task for task in tasks}
    if any(task_id not in task_by_id for task_id in hard_suite.task_ids):
        raise ValueError("hard suite references an unknown task")
    families = set(_AFG_FAMILIES)
    selected = tuple(
        task_by_id[task_id]
        for task_id in hard_suite.task_ids
        if task_by_id[task_id].task_family in families
    )
    if any(task.metadata.split.value != "test" for task in selected):
        raise ValueError("Task 12 hard subset must contain test tasks only")
    family_counts = {
        family: sum(task.task_family == family for task in selected)
        for family in _AFG_FAMILIES
    }
    if family_counts != {family: 80 for family in _AFG_FAMILIES}:
        raise ValueError("Task 12 hard subset must contain 80 tasks per A/F/G family")
    return selected


def _validate_semantic_matrix_scope(
    *,
    matrix: Task12SemanticMatrixV1,
    selected_tasks: tuple[MemUpdateTaskV3, ...],
) -> tuple[MemUpdateTaskV3, ...]:
    family_a_tasks = tuple(
        task for task in selected_tasks if task.task_family == _AFG_FAMILIES[0]
    )
    family_a_ids = tuple(task.task_id for task in family_a_tasks)
    if matrix.task_scope.task_ids != family_a_ids:
        raise ValueError(
            "Task 12 matrix scope does not equal the authenticated Family A cohort"
        )
    if matrix.raw_append_intervention.task_ids != family_a_ids:
        raise ValueError(
            "raw append intervention must bind the authenticated Family A cohort"
        )
    if any(cell.task_ids != family_a_ids for cell in matrix.intervention_cells):
        raise ValueError("Task 12 cells must bind the authenticated Family A cohort")
    return family_a_tasks


def _task_selection_digest(tasks: tuple[MemUpdateTaskV3, ...]) -> str:
    payload = [
        {"task_id": task.task_id, "task_sha256": sha256_model(task)}
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_authenticated_core(
    *,
    manifest: Task12PreparationManifestV1,
    core_root: Path,
) -> tuple[
    tuple[MemUpdateTaskV3, ...],
    tuple[MemUpdateTaskV3, ...],
    str,
    str,
    str,
]:
    release_raw = _read_artifact(root=core_root, location=manifest.release_manifest)
    try:
        release = json.loads(release_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Core release manifest is not JSON") from exc
    if (
        type(release) is not dict
        or release.get("schema_version") != "memupdatebench.core.task_release.v1"
        or release.get("release_status") != "FINAL_APPROVED"
        or release.get("release_stage") != "task_release"
        or release.get("release_manifest_hash")
        != _APPROVED_CORE_RELEASE_MANIFEST_HASH
        or release.get("release_root_digest")
        != _APPROVED_CORE_RELEASE_ROOT_DIGEST
        or release.get("source_task_manifest_hash")
        != manifest.task_manifest.artifact.sha256
        or release.get("task_count") != 12000
        or release.get("hard_suite_task_count") != 560
    ):
        raise ValueError("Core release manifest does not match the approved task release")
    release_hash_payload = dict(release)
    release_hash_payload.pop("release_manifest_hash", None)
    if _canonical_json_sha256(release_hash_payload) != _APPROVED_CORE_RELEASE_MANIFEST_HASH:
        raise ValueError("Core release manifest self-hash is invalid")
    artifact_hashes = {
        item.get("path"): item.get("sha256")
        for item in release.get("artifact_refs", ())
        if type(item) is dict
    }
    if artifact_hashes != {**artifact_hashes, **{
        "candidate/task_manifest.json": _APPROVED_CORE_TASK_MANIFEST_SHA256,
        "candidate/core-hard-v1.json": _APPROVED_CORE_HARD_SUITE_SHA256,
        "candidate/tasks.jsonl": _APPROVED_CORE_TASKS_SHA256,
    }}:
        raise ValueError("Core release artifact references do not bind approved candidate artifacts")
    manifest_raw = _read_artifact(root=core_root, location=manifest.task_manifest)
    hard_raw = _read_artifact(root=core_root, location=manifest.core_hard_suite)
    tasks_raw = _read_artifact(root=core_root, location=manifest.tasks)
    task_manifest = _canonical_json_model(manifest_raw, TaskManifestV3, label="task manifest")
    hard_suite = _canonical_json_model(hard_raw, CoreHardSuiteManifest, label="core hard suite")
    tasks = _read_canonical_tasks(tasks_raw)
    expected_task_ref = ArtifactRef(
        path="tasks.jsonl",
        sha256=hashlib.sha256(tasks_raw).hexdigest(),
        media_type="application/x-ndjson",
        record_count=len(tasks),
    )
    if tuple(task_manifest.task_file_paths_and_hashes) != (expected_task_ref,):
        raise ValueError("task manifest does not authenticate candidate/tasks.jsonl")
    if hard_suite.source_task_manifest_hash != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError("core hard suite does not bind the task manifest")
    selected = _selected_hard_tasks(
        task_manifest=task_manifest,
        hard_suite=hard_suite,
        tasks=tasks,
    )
    selected_ids = tuple(task.task_id for task in selected)
    if manifest.hard_subset.task_ids != selected_ids:
        raise ValueError("declared hard subset does not match authenticated Core A/F/G selection")
    family_a_tasks = tuple(
        task for task in selected if task.task_family == _AFG_FAMILIES[0]
    )
    family_a_ids = tuple(task.task_id for task in family_a_tasks)
    if manifest.semantic_matrix.task_scope.task_ids != family_a_ids:
        raise ValueError("Task 12 matrix must bind the authenticated Family A subset")
    test_tasks = tuple(
        task for task in tasks if task.metadata.split.value == "test"
    )
    if len(test_tasks) != 2400:
        raise ValueError("authenticated Core release must contain 2,400 test tasks")
    hard_source_digest = _task_selection_digest(selected)
    matrix_digest = _task_selection_digest(family_a_tasks)
    main_test_digest = _task_selection_digest(test_tasks)
    if manifest.main_manager_policy.task_selection_sha256 != main_test_digest:
        raise ValueError("main manager policy does not bind the authenticated Core test split")
    return (
        selected,
        test_tasks,
        hard_source_digest,
        matrix_digest,
        main_test_digest,
    )


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_answer_model_evidence(
    *,
    binding: Task11AnswerModelBindingV1,
    evidence_root: Path,
) -> None:
    raw = _read_artifact(root=evidence_root, location=binding.qualification_report)
    try:
        report = Task11QualificationReportV1.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("Task 11 qualification report is invalid") from exc
    if canonical_json_bytes(report) != raw:
        raise ValueError("Task 11 qualification report must be canonical JSON")
    matches = [slot for slot in report.slots if slot.slot_id == binding.slot_id]
    if len(matches) != 1:
        raise ValueError("Task 11 qualification report does not bind the selected slot")
    slot = matches[0]
    if (
        slot.model_id != binding.model_id
        or slot.revision != binding.revision
        or slot.license_id != binding.license_id
        or slot.tree_manifest_sha256 != binding.tree_manifest_sha256
    ):
        raise ValueError("Task 11 qualification slot does not match answer-model binding")
    if _canonical_json_sha256(
        report.offline_contract.decoding.model_dump(mode="json")
    ) != binding.decoding_config_sha256:
        raise ValueError("Task 11 decoding contract does not match answer-model binding")


def _validate_task10_mem0_admission(
    *,
    binding: Task12ExternalAdmissionBindingV1,
    evidence_root: Path,
    source_task_manifest_sha256: str,
) -> None:
    decision_raw = _read_artifact(root=evidence_root, location=binding.decision)
    report_raw = _read_artifact(root=evidence_root, location=binding.report)
    try:
        decision = _canonical_json_model(
            decision_raw, AdmissionDecisionV1, label="Task 10 decision"
        )
        report = _canonical_json_model(
            report_raw, ExternalAdmissionReportV1, label="Task 10 report"
        )
    except ValueError as exc:
        raise ValueError("Task 10 admission evidence is invalid") from exc
    admitted_ref = decision.admitted_report
    if (
        decision.status is not AdmissionDecisionStatus.ADMITTED
        or admitted_ref is None
        or admitted_ref.candidate_id.value != "mem0_oss"
        or admitted_ref.report_hash != binding.report.artifact.sha256
        or report.candidate_id.value != "mem0_oss"
        or report.adapter_info.adapter_id != "mem0_oss"
        or report.adapter_info.system_name != "mem0_oss"
        or report.outcome.value != "pass"
        or report.source_task_manifest_hash != source_task_manifest_sha256
        or decision.source_task_manifest_hash != source_task_manifest_sha256
        or decision.evaluation_configuration_hash
        != report.evaluation_configuration_hash
    ):
        raise ValueError("Task 10 Mem0 admission evidence is not an admitted matching report")


def _validate_cell_evidence(
    *,
    cells: tuple[Task12InterventionCellV1, ...],
    evidence_root: Path,
    selected_tasks: tuple[MemUpdateTaskV3, ...],
    raw_append_intervention: RawAppendInterventionV1,
    source_task_manifest_sha256: str,
) -> None:
    task_by_id = {task.task_id: task for task in selected_tasks}
    raw_append_ids = raw_append_intervention.task_ids
    for cell in cells:
        if not set(cell.task_ids) <= set(task_by_id):
            raise ValueError("intervention cell selects tasks outside authenticated Family A")
        config_raw = _read_artifact(root=evidence_root, location=cell.adapter_configuration)
        info_raw = _read_artifact(root=evidence_root, location=cell.adapter_info)
        capability_raw = _read_artifact(root=evidence_root, location=cell.capability_verification)
        retrieval_raw = _read_artifact(root=evidence_root, location=cell.retrieval.artifact)
        retrieval_config = _canonical_json_model(
            retrieval_raw,
            Task12RetrievalConfigurationV1,
            label=f"retrieval configuration for {cell.cell_id}",
        )
        if retrieval_config != cell.retrieval.configuration:
            raise ValueError("retrieval artifact does not match its typed cell binding")
        adapter_info = _canonical_json_model(
            info_raw,
            AdapterInfoV3,
            label=f"adapter info for {cell.cell_id}",
        )
        capability_verification = _canonical_json_model(
            capability_raw,
            Task12CapabilityVerificationV1,
            label=f"capability verification for {cell.cell_id}",
        )
        if (
            capability_verification.adapter_id != "raw_add"
            or capability_verification.configuration_hash
            != hashlib.sha256(config_raw).hexdigest()
            or capability_verification.source_task_manifest_hash
            != source_task_manifest_sha256
        ):
            raise ValueError("capability verification does not bind this raw cell")
        capabilities = capability_verification.capabilities
        if adapter_info.adapter_id != "raw_add":
            raise ValueError("adapter info does not bind raw_add")
        if adapter_info.configuration_hash != hashlib.sha256(config_raw).hexdigest():
            raise ValueError("adapter info does not bind the configuration artifact")
        if cell.task_ids != raw_append_ids:
            raise ValueError("raw_add cells must use exactly the Family A intervention")
        for task_id in cell.task_ids:
            support = resolve_task_support_v3(
                task_by_id[task_id],
                capabilities,
                allow_append_only_observation=True,
                answer_mode="slot_prompt",
            )
            if not support.terminal_supported:
                missing = ", ".join(support.missing_capabilities)
                raise ValueError(
                    f"intervention cell {cell.cell_id} lacks required task capabilities: {missing}"
                )


def _canonical_answer_model_binding_hash(
    binding: Task11AnswerModelBindingV1,
) -> str:
    return _canonical_json_sha256({
        "slot_id": binding.slot_id,
        "qualification_report_sha256": binding.qualification_report_sha256,
        "model_id": binding.model_id,
        "revision": binding.revision,
        "license_id": binding.license_id,
        "tree_manifest_sha256": binding.tree_manifest_sha256,
        "decoding_config_sha256": binding.decoding_config_sha256,
    })


def _canonical_answer_run_binding_hash(
    *,
    cell_binding_sha256: str,
    answer_model_binding_sha256: str,
) -> str:
    return _canonical_json_sha256({
        "cell_binding_sha256": cell_binding_sha256,
        "answer_model_binding_sha256": answer_model_binding_sha256,
    })


def _repository_identity() -> tuple[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    commands = (
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=normal"),
        ("ls-tree", "-r", "-z", "HEAD"),
    )
    outputs: list[bytes] = []
    for command in commands:
        result = subprocess.run(
            ("git", "-C", str(project_root), *command),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError("Task 12 cannot determine repository identity")
        outputs.append(result.stdout)
    revision = outputs[0].decode("ascii").strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Task 12 repository revision is not an immutable Git commit")
    if outputs[1].strip():
        raise ValueError("Task 12 requires a clean repository worktree")
    return revision, hashlib.sha256(outputs[2]).hexdigest()


def _validate_output_target(
    *,
    output_dir: str | Path,
    output_leaf: str,
    core_root: Path,
    evidence_root: Path,
) -> str:
    parent = _require_real_directory(output_dir, label="output")
    candidate = parent / output_leaf
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("Task 12 output target must not already exist")
    if core_root == candidate or core_root in candidate.parents or candidate in core_root.parents:
        raise ValueError("Task 12 output target must be outside the immutable Core root")
    if evidence_root == candidate or evidence_root in candidate.parents or candidate in evidence_root.parents:
        raise ValueError("Task 12 output target must be outside the evidence root")
    project_root = Path(__file__).resolve().parents[3]
    if project_root == candidate or project_root in candidate.parents or candidate in project_root.parents:
        raise ValueError("Task 12 output target must be outside the repository worktree")
    return str(candidate)


def _plan_fingerprint(
    *,
    manifest: Task12PreparationManifestV1,
    task_selection_sha256: str,
    code_revision: str,
    code_tree_sha256: str,
) -> str:
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "task_selection_sha256": task_selection_sha256,
        "code_revision": code_revision,
        "code_tree_sha256": code_tree_sha256,
    }
    return _canonical_json_sha256(payload)


def admit_task12_dry_run(
    *,
    manifest: Task12PreparationManifestV1,
    core_root: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
) -> Task12DryRunPlanV1:
    resolved_core = _require_real_directory(core_root, label="Core")
    resolved_evidence = _require_real_directory(evidence_root, label="evidence")
    if not isinstance(manifest, Task12PreparationManifestV1):
        manifest = Task12PreparationManifestV1.model_validate(manifest)
    _validate_distinct_roots(resolved_core, resolved_evidence)
    _validate_core_artifact_locations(manifest)
    (
        selected_tasks,
        test_tasks,
        hard_source_digest,
        matrix_digest,
        main_test_digest,
    ) = _validate_authenticated_core(
        manifest=manifest,
        core_root=resolved_core,
    )
    _validate_task10_mem0_admission(
        binding=manifest.task10_mem0_admission,
        evidence_root=resolved_evidence,
        source_task_manifest_sha256=manifest.task_manifest.artifact.sha256,
    )
    semantic_matrix = manifest.semantic_matrix
    family_a_tasks = _validate_semantic_matrix_scope(
        matrix=semantic_matrix,
        selected_tasks=selected_tasks,
    )
    for binding in manifest.answer_models:
        _validate_answer_model_evidence(
            binding=binding,
            evidence_root=resolved_evidence,
        )
    raw_intervention = semantic_matrix.raw_append_intervention
    raw_trajectory_bytes = _read_artifact(
        root=resolved_evidence,
        location=raw_intervention.trajectory_artifact,
    )
    _read_raw_append_trajectories(
        raw_trajectory_bytes,
        expected_task_ids=raw_intervention.task_ids,
        expected_tasks=family_a_tasks,
    )
    _validate_cell_evidence(
        cells=semantic_matrix.intervention_cells,
        evidence_root=resolved_evidence,
        selected_tasks=family_a_tasks,
        raw_append_intervention=raw_intervention,
        source_task_manifest_sha256=manifest.task_manifest.artifact.sha256,
    )
    _validate_output_target(
        output_dir=output_dir,
        output_leaf=manifest.output_leaf,
        core_root=resolved_core,
        evidence_root=resolved_evidence,
    )
    code_revision, code_tree_sha256 = _repository_identity()
    admitted_cells = tuple(
        Task12AdmittedCellV1(
            cell_id=cell.cell_id,
            scope_id=cell.scope_id,
            canonical_binding_sha256=_canonical_cell_binding_hash(
                cell=cell,
                scope=semantic_matrix.task_scope,
                raw_append_intervention=raw_intervention,
            ),
        )
        for cell in semantic_matrix.intervention_cells
    )
    answer_binding_hashes = tuple(
        _canonical_answer_model_binding_hash(binding)
        for binding in manifest.answer_models
    )
    answer_hash_by_slot = dict(
        zip(
            manifest.scientific_design.answer_model_slots,
            answer_binding_hashes,
        )
    )
    admitted_answer_runs = tuple(
        Task12AdmittedAnswerRunV1(
            cell_id=cell.cell_id,
            answer_model_slot=slot,
            cell_binding_sha256=cell.canonical_binding_sha256,
            answer_model_binding_sha256=answer_hash_by_slot[slot],
            canonical_run_binding_sha256=(
                _canonical_answer_run_binding_hash(
                    cell_binding_sha256=cell.canonical_binding_sha256,
                    answer_model_binding_sha256=answer_hash_by_slot[slot],
                )
            ),
        )
        for cell in admitted_cells
        for slot in manifest.scientific_design.answer_model_slots
    )
    return Task12DryRunPlanV1(
        run_id=manifest.run_id,
        plan_fingerprint_sha256=_plan_fingerprint(
            manifest=manifest,
            task_selection_sha256=hard_source_digest,
            code_revision=code_revision,
            code_tree_sha256=code_tree_sha256,
        ),
        core_task_manifest_sha256=manifest.task_manifest.artifact.sha256,
        core_hard_suite_sha256=manifest.core_hard_suite.artifact.sha256,
        core_tasks_sha256=manifest.tasks.artifact.sha256,
        scientific_design_sha256=sha256_model(manifest.scientific_design),
        semantic_matrix_sha256=sha256_model(semantic_matrix),
        main_manager_policy_sha256=sha256_model(manifest.main_manager_policy),
        answer_model_slots=manifest.scientific_design.answer_model_slots,
        answer_model_binding_sha256=answer_binding_hashes,
        admitted_cells=admitted_cells,
        admitted_answer_runs=admitted_answer_runs,
        hard_source_task_count=len(selected_tasks),
        hard_source_task_selection_sha256=hard_source_digest,
        matrix_task_count=len(family_a_tasks),
        matrix_task_selection_sha256=matrix_digest,
        main_test_task_count=len(test_tasks),
        main_test_task_selection_sha256=main_test_digest,
        output_leaf=manifest.output_leaf,
        code_revision=code_revision,
        code_tree_sha256=code_tree_sha256,
    )


__all__ = [
    "admit_task12_dry_run",
    "RawAppendInterventionV1",
    "Task11AnswerModelBindingV1",
    "Task12CapabilityVerificationV1",
    "Task12ContextInterventionV1",
    "Task12CoreTaskScopeV1",
    "Task12InterventionCellV1",
    "Task12MainManagerPolicyV1",
    "Task12RetrievalBindingV1",
    "Task12RetrievalConfigurationV1",
    "Task12ScientificDesignV1",
    "Task12SemanticMatrixV1",
    "Task12AdmittedCellV1",
    "Task12AdmittedAnswerRunV1",
    "Task12AdapterCellV1",
    "Task12ArtifactLocationV1",
    "Task12DryRunPlanV1",
    "Task12HardSubsetV1",
    "Task12PreparationManifestV1",
]
