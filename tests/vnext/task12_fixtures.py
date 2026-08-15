from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from mub.vnext.contracts.common import ArtifactRef
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.external.contracts import (
    ADMISSION_GATE_NAMES,
    AdmissionDecisionStatus,
    AdmissionDecisionV1,
    CandidateReportRefV1,
    ExternalAdmissionReportV1,
    ExternalCandidateId,
    GateResultV1,
    GateStatus,
)
from mub.vnext.generation.core_artifacts import build_core_artifact_bundle
from mub.vnext.generation.core_build import compile_core_snapshot
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.preparation.task12 import (
    RawAppendInterventionV1,
    RawAppendTrajectoryV1,
    Task11AnswerModelBindingV1,
    Task12ArtifactLocationV1,
    Task12CapabilityVerificationV1,
    Task12ExternalAdmissionBindingV1,
    Task12ContextInterventionV1,
    Task12CoreTaskScopeV1,
    Task12HardSubsetV1,
    Task12InterventionCellV1,
    Task12MainManagerPolicyV1,
    Task12PreparationManifestV1,
    Task12RetrievalBindingV1,
    Task12RetrievalConfigurationV1,
    Task12ScientificDesignV1,
    Task12SemanticMatrixV1,
)


ROOT = Path(__file__).resolve().parents[2]
_A_FAMILY = "repeated_same_slot_update"
_AFG_FAMILIES = (
    _A_FAMILY,
    "current_historical_query",
    "long_horizon_memory_synthesis",
)


def _write(
    path: Path,
    payload: bytes,
    *,
    media_type: str = "application/json",
    record_count: int = 1,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": media_type,
        "record_count": record_count,
    }


def _location(
    reference: dict[str, object],
    *,
    root: str,
    root_path: Path,
) -> Task12ArtifactLocationV1:
    path = Path(reference["path"])
    relative_path = (
        path.relative_to(root_path).as_posix()
        if path.is_absolute()
        else path.as_posix()
    )
    return Task12ArtifactLocationV1(
        root=root,
        artifact=ArtifactRef(
            path=relative_path,
            sha256=reference["sha256"],
            media_type=reference["media_type"],
            record_count=reference["record_count"],
        ),
        relative_path=relative_path,
    )


def _task_selection_digest(tasks) -> str:
    payload = [
        {"task_id": task.task_id, "task_sha256": sha256_model(task)}
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cell_id(order: str, annotation: str, retrieval_k: int) -> str:
    condition = (
        "chronological-none"
        if (order, annotation) == ("chronological", "none")
        else "reverse-none"
        if annotation == "none"
        else "reverse-version-labeled"
    )
    return f"raw-add-{condition}-k{retrieval_k:02d}"


def _trajectory_receipt_for_task(task):
    actions = {action.action_id: action for action in task.actions}
    versions: dict[str, int] = {}
    entries = []
    for event in sorted(task.events, key=lambda item: item.sequence_index):
        for action_id in event.gold_action_ids:
            action = actions[action_id]
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
    latest_by_key: dict[str, dict[str, object]] = {}
    for entry in entries:
        previous = latest_by_key.get(entry["object_key"])
        if previous is None or (
            entry["event_index"], entry["version_index"]
        ) > (previous["event_index"], previous["version_index"]):
            latest_by_key[entry["object_key"]] = entry
    latest_ids = {entry["entry_id"] for entry in latest_by_key.values()}
    trajectory_hash = hashlib.sha256(
        json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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
        trajectory_sha256=trajectory_hash,
    )


def build_task12_inputs(tmp_path: Path) -> dict[str, object]:
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    config = load_core_config(ROOT / "configs" / "vnext" / "core.yaml")
    bundle = build_core_artifact_bundle(
        compile_core_snapshot(config, code_revision=revision), config
    )
    core_root = tmp_path / "core"
    evidence_root = tmp_path / "evidence"
    candidate = core_root / "candidate"
    refs: dict[str, dict[str, object]] = {}
    for artifact in bundle.artifacts:
        path = candidate / artifact.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.content)
        refs[artifact.path] = {
            "path": f"candidate/{artifact.path}",
            "sha256": hashlib.sha256(artifact.content).hexdigest(),
            "media_type": artifact.media_type,
            "record_count": artifact.record_count,
        }

    release = {
        "schema_version": "memupdatebench.core.task_release.v1",
        "release_status": "FINAL_APPROVED",
        "release_stage": "task_release",
        "release_root_digest": "f" * 64,
        "source_task_manifest_hash": refs["task_manifest.json"]["sha256"],
        "task_count": 12000,
        "hard_suite_task_count": 560,
        "artifact_refs": [
            {"path": reference["path"], "sha256": reference["sha256"]}
            for reference in refs.values()
        ],
    }
    release["release_manifest_hash"] = hashlib.sha256(
        json.dumps(release, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    release_ref = _write(
        core_root / "task_release_manifest.json",
        json.dumps(release, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    decoding = {
        "do_sample": False,
        "max_new_tokens": 32,
        "num_beams": 1,
        "seed": 0,
    }
    qualification = {
        "schema_version": "memupdatebench.core-task11-answer-harness.v1",
        "status": "qualified",
        "offline_contract": {
            "hf_hub_offline": True,
            "transformers_offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "decoding": decoding,
        },
        "slots": [
            {
                "slot_id": "answer_model_a",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "b" * 40,
                "license_id": "apache-2.0",
                "tree_manifest_sha256": "a" * 64,
            },
            {
                "slot_id": "answer_model_b",
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "revision": "c" * 40,
                "license_id": "apache-2.0",
                "tree_manifest_sha256": "d" * 64,
            },
        ],
    }
    qualification_ref = _write(
        evidence_root / "task11" / "qualification_summary.json",
        json.dumps(qualification, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    source_manifest_hash = refs["task_manifest.json"]["sha256"]
    evaluation_hash = "e" * 64
    mem0_config_hash = "9" * 64

    def task10_ref(path: str, sha256: str) -> ArtifactRef:
        return ArtifactRef(
            path=f"task10/{path}.json",
            sha256=sha256,
            media_type="application/json",
            record_count=1,
        )

    gate_evidence = task10_ref("gate-evidence", "7" * 64)
    task10_report = ExternalAdmissionReportV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        source_task_manifest_hash=source_manifest_hash,
        source_task_manifest_ref=task10_ref("source-task-manifest", source_manifest_hash),
        evaluation_configuration_hash=evaluation_hash,
        evaluation_configuration_ref=task10_ref("evaluation-configuration", evaluation_hash),
        adapter_configuration_ref=task10_ref("adapter-configuration", mem0_config_hash),
        probe_ref=task10_ref("probe", "1" * 64),
        canary_ref=task10_ref("canary", "2" * 64),
        package_provenance_ref=task10_ref("package-provenance", "3" * 64),
        model_provenance_ref=task10_ref("model-provenance", "4" * 64),
        adapter_info=AdapterInfoV3(
            adapter_id="mem0_oss",
            adapter_version="2.0.17",
            system_name="mem0_oss",
            system_version="2.0.17",
            configuration_hash=mem0_config_hash,
        ),
        adapter_capabilities=AdapterCapabilitiesV3(
            supports_isolated_reset=True,
            supports_event_ingest=True,
            supports_add=True,
            exports_entries=True,
        ),
        state_transition_linkage_available=False,
        gates=tuple(
            GateResultV1(
                name=name,
                status=GateStatus.PASS,
                evidence_artifacts=(gate_evidence,),
            )
            for name in ADMISSION_GATE_NAMES
        ),
        outcome=GateStatus.PASS,
    )
    task10_report_ref = _write(
        evidence_root / "task10" / "mem0_report.json",
        canonical_json_bytes(task10_report),
    )
    report_ref = CandidateReportRefV1(
        candidate_id=ExternalCandidateId.MEM0_OSS,
        report_hash=task10_report_ref["sha256"],
    )
    task10_decision = AdmissionDecisionV1(
        status=AdmissionDecisionStatus.ADMITTED,
        source_task_manifest_hash=source_manifest_hash,
        evaluation_configuration_hash=evaluation_hash,
        reports=(report_ref,),
        admitted_report=report_ref,
        reasons=("admitted_mem0_primary",),
    )
    task10_decision_ref = _write(
        evidence_root / "task10" / "admission_decision.json",
        canonical_json_bytes(task10_decision),
    )

    raw_config_ref = _write(
        evidence_root / "adapters" / "raw_add" / "config.json",
        json.dumps(
            {"adapter": "raw_add"}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )
    raw_info = AdapterInfoV3(
        adapter_id="raw_add",
        adapter_version="v1",
        system_name="raw_add",
        system_version="v1",
        configuration_hash=raw_config_ref["sha256"],
    )
    capabilities = AdapterCapabilitiesV3(
        supports_isolated_reset=True,
        supports_event_ingest=True,
        supports_add=True,
        supports_update=True,
        supports_noop=True,
        supports_delete=True,
        supports_ttl=True,
        supports_historical_query=True,
        supports_multi_object_query=True,
        exports_entries=True,
    )
    raw_evidence_refs = {
        "config": raw_config_ref,
        "info": _write(
            evidence_root / "adapters" / "raw_add" / "info.json",
            canonical_json_bytes(raw_info),
        ),
        "capability": _write(
            evidence_root / "adapters" / "raw_add" / "capability.json",
            canonical_json_bytes(
                Task12CapabilityVerificationV1(
                    adapter_id="raw_add",
                    configuration_hash=raw_config_ref["sha256"],
                    source_task_manifest_hash=refs["task_manifest.json"]["sha256"],
                    capabilities=capabilities,
                )
            ),
        ),
    }
    retrieval_refs = {
        retrieval_k: _write(
            evidence_root
            / "adapters"
            / "raw_add"
            / f"retrieval-k{retrieval_k}.json",
            canonical_json_bytes(
                Task12RetrievalConfigurationV1(
                    retrieval_policy="normal_topk",
                    retrieval_k=retrieval_k,
                )
            ),
        )
        for retrieval_k in (4, 8, 16)
    }

    hard_ids = set(bundle.hard_suite.task_ids)
    selected_tasks = tuple(sorted(
        (
            task
            for task in bundle.snapshot.tasks
            if task.task_id in hard_ids and task.task_family in set(_AFG_FAMILIES)
        ),
        key=lambda task: task.task_id,
    ))
    a_tasks = tuple(
        task for task in selected_tasks if task.task_family == _A_FAMILY
    )
    test_tasks = tuple(sorted(
        (
            task
            for task in bundle.snapshot.tasks
            if task.metadata.split.value == "test"
        ),
        key=lambda task: task.task_id,
    ))
    a_ids = tuple(task.task_id for task in a_tasks)
    trajectory_payload = b"".join(
        canonical_json_bytes(_trajectory_receipt_for_task(task)) + b"\n"
        for task in a_tasks
    )
    trajectory_ref = _write(
        evidence_root / "raw_add" / "trajectories.jsonl",
        trajectory_payload,
        media_type="application/x-ndjson",
        record_count=80,
    )
    return {
        "core_root": core_root,
        "evidence_root": evidence_root,
        "refs": refs,
        "release_ref": release_ref,
        "release_manifest_hash": release["release_manifest_hash"],
        "qualification_ref": qualification_ref,
        "task10_report_ref": task10_report_ref,
        "task10_decision_ref": task10_decision_ref,
        "raw_evidence_refs": raw_evidence_refs,
        "retrieval_refs": retrieval_refs,
        "trajectory_ref": trajectory_ref,
        "decoding_sha256": hashlib.sha256(
            json.dumps(decoding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "selected_tasks": selected_tasks,
        "a_tasks": a_tasks,
        "test_tasks": test_tasks,
        "bundle": bundle,
    }


def build_task12_manifest(inputs: dict[str, object]) -> Task12PreparationManifestV1:
    core_root = inputs["core_root"]
    evidence_root = inputs["evidence_root"]
    refs = inputs["refs"]
    selected_tasks = inputs["selected_tasks"]
    a_tasks = inputs["a_tasks"]
    test_tasks = inputs["test_tasks"]
    a_ids = tuple(task.task_id for task in a_tasks)
    selected_ids = tuple(task.task_id for task in selected_tasks)

    design = Task12ScientificDesignV1.model_validate_json(
        (ROOT / "configs" / "vnext" / "core_task12_scientific_design.json").read_bytes()
    )
    qualification = _location(
        inputs["qualification_ref"], root="evidence", root_path=evidence_root
    )
    answer_models = (
        Task11AnswerModelBindingV1(
            slot_id="answer_model_a",
            qualification_report=qualification,
            qualification_report_sha256=qualification.artifact.sha256,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="b" * 40,
            license_id="apache-2.0",
            tree_manifest_sha256="a" * 64,
            decoding_config_sha256=inputs["decoding_sha256"],
        ),
        Task11AnswerModelBindingV1(
            slot_id="answer_model_b",
            qualification_report=qualification,
            qualification_report_sha256=qualification.artifact.sha256,
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            revision="c" * 40,
            license_id="apache-2.0",
            tree_manifest_sha256="d" * 64,
            decoding_config_sha256=inputs["decoding_sha256"],
        ),
    )
    raw_refs = inputs["raw_evidence_refs"]
    retrieval_refs = inputs["retrieval_refs"]
    raw_locations = {
        name: _location(reference, root="evidence", root_path=evidence_root)
        for name, reference in raw_refs.items()
    }
    retrieval_locations = {
        retrieval_k: _location(
            reference, root="evidence", root_path=evidence_root
        )
        for retrieval_k, reference in retrieval_refs.items()
    }
    cells = tuple(
        Task12InterventionCellV1(
            cell_id=_cell_id(
                condition.context_order,
                condition.context_annotation,
                retrieval_k,
            ),
            scope_id="core-hard-v1-family-a",
            task_ids=a_ids,
            context_intervention=Task12ContextInterventionV1(
                context_order=condition.context_order,
                context_annotation=condition.context_annotation,
            ),
            adapter_configuration=raw_locations["config"],
            adapter_info=raw_locations["info"],
            capability_verification=raw_locations["capability"],
            retrieval=Task12RetrievalBindingV1(
                configuration=Task12RetrievalConfigurationV1(
                    retrieval_policy="normal_topk",
                    retrieval_k=retrieval_k,
                ),
                artifact=retrieval_locations[retrieval_k],
            ),
        )
        for condition in design.context_conditions
        for retrieval_k in design.retrieval_k_values
    )
    trajectory = _location(
        inputs["trajectory_ref"], root="evidence", root_path=evidence_root
    )
    matrix = Task12SemanticMatrixV1(
        scientific_design=design,
        task_scope=Task12CoreTaskScopeV1(
            scope_id="core-hard-v1-family-a",
            family_ids=(_A_FAMILY,),
            task_ids=a_ids,
        ),
        intervention_cells=cells,
        raw_append_intervention=RawAppendInterventionV1(
            trajectory_artifact=trajectory,
            task_ids=a_ids,
        ),
    )
    main_test_digest = _task_selection_digest(test_tasks)
    return Task12PreparationManifestV1(
        run_id="task12-dry-run",
        release_manifest=_location(
            inputs["release_ref"], root="core", root_path=core_root
        ),
        task_manifest=_location(
            refs["task_manifest.json"], root="core", root_path=core_root
        ),
        core_hard_suite=_location(
            refs["core-hard-v1.json"], root="core", root_path=core_root
        ),
        tasks=_location(refs["tasks.jsonl"], root="core", root_path=core_root),
        hard_subset=Task12HardSubsetV1(
            task_ids=selected_ids,
            family_ids=_AFG_FAMILIES,
        ),
        scientific_design=design,
        answer_models=answer_models,
        semantic_matrix=matrix,
        main_manager_policy=Task12MainManagerPolicyV1(
            manager_ids=design.main_manager_ids,
            task_split="test",
            task_count=2400,
            task_selection_sha256=main_test_digest,
            one_terminal_row_per_requested_task=True,
            unsupported_policy="explicit_terminal_row_with_reason",
            reference_sanity_required=True,
            excluded_from_intervention_matrix=True,
        ),
        task10_mem0_admission=Task12ExternalAdmissionBindingV1(
            decision=_location(
                inputs["task10_decision_ref"], root="evidence", root_path=evidence_root
            ),
            report=_location(
                inputs["task10_report_ref"], root="evidence", root_path=evidence_root
            ),
        ),
        output_leaf="task12-dry-run",
    )
