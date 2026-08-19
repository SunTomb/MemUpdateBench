from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.scoring.scorer_v3 import resolve_final_snapshot_v3
from mub.vnext.statistics.contracts_v3 import (
    CaseCategory,
    TASK13_CASE_CATEGORIES,
    Task13AnswerProjectionV1,
    Task13CaseRecordV1,
    Task13RunCaseCoverageV1,
    Task13RunProjectionV1,
    Task13ScoreProjectionV1,
    Task13TaskProjectionV1,
    Task13TimelineProjectionV1,
    Task13RetrievalProjectionV1,
    task13_case_id_v1,
)
from mub.vnext.statistics.input_v3 import (
    Task13AuthenticatedMatrixV1,
    Task13AuthenticatedObservationV1,
    Task13AuthenticatedRunV1,
    _task13_observation_evidence_sha256,
    _task13_observation_membership_root_sha256,
)


_EXPECTED_RUN_COUNT = 18
_EXPECTED_TASK_COUNT = 80
_EXPECTED_CORE_COUNT = 20
_EXPECTED_TASKS_PER_CORE = 4
_PRIVATE_TIMELINE_FIELDS = (
    "event_id",
    "sequence_index",
    "timestamp",
    "speaker",
    "gold_action_ids",
    "role",
)


@dataclass(frozen=True, slots=True)
class Task13CasesResultV1:
    cases: tuple[Task13CaseRecordV1, ...]
    coverage: tuple[Task13RunCaseCoverageV1, ...]


def _json(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _derived_input_hashes(matrix: Task13AuthenticatedMatrixV1) -> tuple[str, str, str]:
    expected = (
        matrix.plan.core_tasks_sha256,
        matrix.plan.core_task_manifest_sha256,
        sha256_model(matrix.summary),
    )
    supplied = matrix.input_hashes
    labels = ("core_tasks", "core_task_manifest", "task12_matrix_summary")
    for label, expected_hash in zip(labels, expected):
        if supplied.get(label) != expected_hash:
            raise ValueError(f"authenticated input hash mismatch for {label}")
    return expected


def _validate_authenticated_matrix(matrix: Any) -> Task13AuthenticatedMatrixV1:
    if not isinstance(matrix, Task13AuthenticatedMatrixV1):
        raise TypeError("case export requires a Task13AuthenticatedMatrixV1")
    runs = tuple(matrix.runs)
    if len(runs) != _EXPECTED_RUN_COUNT:
        raise ValueError("case export requires exactly 18 authenticated runs")
    expected_pairs = tuple(
        (item.cell_id, item.answer_model_slot)
        for item in matrix.plan.admitted_answer_runs
    )
    manifest_pairs = tuple(
        (item.cell_id, item.answer_model_slot)
        for item in matrix.matrix_manifest.run_bundles
    )
    summary_pairs = tuple(
        (item.cell_id, item.answer_model_slot)
        for item in matrix.summary.completed_runs
    )
    if (
        len(expected_pairs) != _EXPECTED_RUN_COUNT
        or len(set(expected_pairs)) != _EXPECTED_RUN_COUNT
        or manifest_pairs != expected_pairs
        or summary_pairs != expected_pairs
    ):
        raise ValueError("authenticated matrix run coordinates are not canonical")
    observed_pairs = tuple(
        (run.cell.cell_id, run.run_configuration.answer_model_slot) for run in runs
    )
    if observed_pairs != expected_pairs:
        raise ValueError("authenticated matrix run order is not canonical")
    run_ids = tuple(run.source.run_id for run in runs)
    config_ids = tuple(run.run_configuration.run_id for run in runs)
    if len(set(run_ids)) != _EXPECTED_RUN_COUNT or len(set(config_ids)) != _EXPECTED_RUN_COUNT:
        raise ValueError("authenticated matrix run IDs must be unique")
    if run_ids != config_ids:
        raise ValueError("authenticated matrix run source IDs differ from configurations")

    cells = {cell.cell_id: cell for cell in matrix.manifest.semantic_matrix.intervention_cells}
    refs = {
        (item.cell_id, item.answer_model_slot): item
        for item in matrix.matrix_manifest.run_bundles
    }
    summary_by_pair = {
        (item.cell_id, item.answer_model_slot): item
        for item in matrix.summary.completed_runs
    }
    admitted_by_pair = {
        (item.cell_id, item.answer_model_slot): item
        for item in matrix.plan.admitted_answer_runs
    }
    canonical_identity: tuple[tuple[str, str], ...] | None = None
    canonical_core_ids: tuple[str, ...] | None = None
    _derived_input_hashes(matrix)
    for run in runs:
        pair = (run.cell.cell_id, run.run_configuration.answer_model_slot)
        cell = cells.get(run.cell.cell_id)
        if cell is None:
            raise ValueError("authenticated run references an unknown cell")
        config = run.run_configuration
        authorization = run.authorization
        source = run.source
        ref = refs[pair]
        summary = summary_by_pair[pair]
        admitted = admitted_by_pair[pair]
        if run.cell != cell:
            raise ValueError("authenticated run cell differs from preparation matrix")
        if config.answer_model_slot != pair[1] or config.run_id != source.run_id:
            raise ValueError("authenticated run configuration coordinate mismatch")
        run_config_digest = sha256_model(config)
        if (
            run_config_digest != authorization.run_config_sha256
            or run_config_digest != ref.run_config_sha256
        ):
            raise ValueError("authenticated run configuration digest mismatch")
        authorization_digest = sha256_model(authorization)
        if authorization_digest != ref.authorization_sha256:
            raise ValueError("authenticated authorization digest mismatch")
        if config.expected_task_ids != cell.task_ids:
            raise ValueError("authenticated run task scope differs from cell")
        if authorization.cell_id != pair[0] or authorization.answer_model_slot != pair[1]:
            raise ValueError("authenticated authorization coordinate mismatch")
        if authorization.runtime_code_binding != matrix.runtime:
            raise ValueError("authenticated run runtime binding mismatch")
        if authorization.canonical_run_binding_sha256 != admitted.canonical_run_binding_sha256:
            raise ValueError("authenticated run binding differs from admitted plan")
        if authorization.cell_binding_sha256 != admitted.cell_binding_sha256:
            raise ValueError("authenticated cell binding differs from admitted plan")
        if authorization.answer_model_binding_sha256 != admitted.answer_model_binding_sha256:
            raise ValueError("authenticated answer-model binding differs from admitted plan")
        if (
            authorization.task_manifest_sha256 != ref.task_manifest_sha256
            or authorization.task_view_sha256 != ref.task_view_sha256
            or authorization.run_config_sha256 != ref.run_config_sha256
            or authorization.output_leaf != ref.output_leaf
        ):
            raise ValueError("authenticated run artifact linkage differs from matrix manifest")
        if (
            source.run_manifest_sha256 != summary.run_manifest_sha256
            or source.score_artifact_sha256 != summary.score_artifact_sha256
            or summary.bundle_leaf != ref.bundle_leaf
            or summary.output_leaf != ref.output_leaf
            or summary.task_count != _EXPECTED_TASK_COUNT
            or summary.score_count != _EXPECTED_TASK_COUNT
        ):
            raise ValueError("authenticated run source linkage differs from matrix summary")
        observations = tuple(run.observations)
        if len(observations) != _EXPECTED_TASK_COUNT:
            raise ValueError("each authenticated run requires exactly 80 observations")
        identities = tuple(
            (observation.task.task_id, observation.semantic_core_id)
            for observation in observations
        )
        expected_identity = tuple(
            sorted(
                identities,
                key=lambda pair: (pair[1].encode("utf-8"), pair[0].encode("utf-8")),
            )
        )
        if identities != expected_identity:
            raise ValueError("authenticated task identity sequence is not canonical")
        if len({task_id for task_id, _ in identities}) != _EXPECTED_TASK_COUNT:
            raise ValueError("authenticated observations require unique task IDs")
        core_counts = Counter(core_id for _, core_id in identities)
        if len(core_counts) != _EXPECTED_CORE_COUNT or any(
            count != _EXPECTED_TASKS_PER_CORE for count in core_counts.values()
        ):
            raise ValueError("authenticated observations require exactly 20 cores x 4 tasks")
        core_ids = tuple(sorted(core_counts, key=lambda value: value.encode("utf-8")))
        if canonical_identity is None:
            canonical_identity = identities
            canonical_core_ids = core_ids
        elif identities != canonical_identity:
            raise ValueError("authenticated runs do not share one canonical task identity sequence")
        for observation in observations:
            _validate_observation_linkage(
                observation,
                run=run,
                cell=cell,
                config=config,
            )
        if (
            run.observation_membership_root_sha256
            != matrix.observation_membership_roots.get(run.source.run_id)
            or run.observation_membership_root_sha256
            != _task13_observation_membership_root_sha256(observations)
        ):
            raise ValueError("authenticated observation membership root mismatch")
    if not matrix._loader_seal_valid:
        raise ValueError("authenticated loader seal is invalid")
    if canonical_core_ids != matrix.canonical_core_ids:
        raise ValueError("authenticated canonical core IDs differ from run observations")
    return matrix


def _validate_observation_linkage(
    observation: Task13AuthenticatedObservationV1,
    *,
    run: Task13AuthenticatedRunV1,
    cell: Any,
    config: Any,
) -> None:
    task = observation.task
    runtime_row = observation.run
    score = observation.score
    source = observation.source
    if observation.source != run.source:
        raise ValueError("case observation source differs from authenticated run")
    if (
        observation.cell_id != cell.cell_id
        or observation.slot != config.answer_model_slot
        or observation.k != cell.retrieval.configuration.retrieval_k
        or observation.context_order != cell.context_intervention.context_order
        or observation.context_annotation != cell.context_intervention.context_annotation
    ):
        raise ValueError("case observation coordinate differs from authenticated cell")
    if observation.semantic_core_id != task.metadata.split_key.semantic_core_id:
        raise ValueError("case observation semantic-core ID differs from task metadata")
    if task.task_id not in cell.task_ids or task.task_id not in config.expected_task_ids:
        raise ValueError("case observation task is outside its authenticated cell")
    if config.task_record_hashes.get(task.task_id) != sha256_model(task):
        raise ValueError("case observation task hash differs from authenticated run configuration")
    if runtime_row.task_id != task.task_id or score.task_id != task.task_id:
        raise ValueError("case task ID differs across authenticated evidence")
    if runtime_row.run_id != source.run_id or score.run_id != source.run_id:
        raise ValueError("case run ID differs across authenticated evidence")
    if runtime_row.adapter_id != config.adapter_info.adapter_id or score.adapter_id != config.adapter_info.adapter_id:
        raise ValueError("case adapter ID differs from authenticated run configuration")
    if score.task_family != task.task_family or score.difficulty != task.difficulty:
        raise ValueError("case task metadata differs from authenticated score")
    if observation.evidence_sha256 != _task13_observation_evidence_sha256(observation):
        raise ValueError("case observation evidence digest mismatch")


def classify_task13_case_v1(observation: Any) -> CaseCategory:
    score = observation.score
    if score.answer_scores.exact_match == 1:
        return "correct"
    if score.answer_scores.stale_copied == 1:
        return "stale_copied"
    if any(not prediction.format_valid for prediction in observation.run.answer_predictions):
        return "answer_parse_invalid"
    return "other_wrong"


_PRIVATE_METADATA_SPLIT_KEY_FIELDS = (
    "semantic_core_id",
    "source_group_id",
    "trajectory_id",
    "paraphrase_group_id",
    "source_document_id",
    "version_group_id",
    "split_exception_id",
    "split_policy_version",
)


def _task_metadata_projection(task: Any, private: bool) -> dict[str, Any]:
    payload = _json(task.metadata)
    if not private:
        return payload
    split_key = payload["split_key"]
    return {
        "split": payload["split"],
        "split_key": {
            field: split_key[field] for field in _PRIVATE_METADATA_SPLIT_KEY_FIELDS
        },
        "profile_name": payload["profile_name"],
        "generation_config_hash": payload["generation_config_hash"],
        "compiler_version": payload["compiler_version"],
    }


def _source_projection(task: Any, private: bool) -> dict[str, Any]:
    source = task.source
    if not private:
        payload = _json(source)
        payload["redacted"] = False
        return payload
    return {
        "source_id": source.source_id,
        "source_type": source.source_type.value if hasattr(source.source_type, "value") else source.source_type,
        "source_uri": None,
        "license_or_privacy": source.license_or_privacy,
        "raw_hash": source.raw_hash,
        "normalized_hash": source.normalized_hash,
        "normalization_version": source.normalization_version,
        "redacted": True,
    }


def _private_timeline_item(event: Any) -> dict[str, Any]:
    payload = _json(event)
    return {field: payload[field] for field in _PRIVATE_TIMELINE_FIELDS if field in payload}


def _validate_observation(observation: Any) -> None:
    task = observation.task
    run = observation.run
    score = observation.score
    source = observation.source
    core_id = task.metadata.split_key.semantic_core_id
    if observation.semantic_core_id != core_id:
        raise ValueError("case observation semantic-core ID differs from task metadata")
    if run.task_id != task.task_id or score.task_id != task.task_id:
        raise ValueError("case task ID differs across authenticated evidence")
    if run.run_id != score.run_id or source.run_id != run.run_id:
        raise ValueError("case run ID differs across authenticated evidence")
    if run.adapter_id != score.adapter_id:
        raise ValueError("case adapter ID differs across authenticated evidence")
    if score.task_family != task.task_family or score.difficulty != task.difficulty:
        raise ValueError("case task metadata differs from authenticated score")


def _project_task13_case_v1(
    observation: Task13AuthenticatedObservationV1,
    matrix: Task13AuthenticatedMatrixV1,
) -> Task13CaseRecordV1:
    if not isinstance(matrix, Task13AuthenticatedMatrixV1):
        raise TypeError("case projection requires a Task13AuthenticatedMatrixV1")
    _validate_observation(observation)
    run = next((item for item in matrix.runs if item.source.run_id == observation.source.run_id), None)
    if run is None or observation not in run.observations:
        raise ValueError("case observation is not a member of the authenticated matrix")
    cell = next(item for item in matrix.manifest.semantic_matrix.intervention_cells if item.cell_id == run.cell.cell_id)
    _validate_observation_linkage(observation, run=run, cell=cell, config=run.run_configuration)
    task_hash, task_manifest_hash, summary_hash = _derived_input_hashes(matrix)
    task = observation.task
    runtime_row = observation.run
    score = observation.score
    category = classify_task13_case_v1(observation)
    private = task.source.provenance.get("redistributable") is not True
    source_payload = _source_projection(task, private)
    timeline = tuple(
        _private_timeline_item(event) if private else _json(event)
        for event in task.events
    )
    metric_layers = {
        field_name: _json(getattr(score, field_name))
        for field_name in type(score).model_fields
        if field_name.endswith("_scores")
    }
    support = {
        path: _json(record)
        for path, record in score.supported_metric_fields.items()
    }
    flags = tuple(
        flag.value if hasattr(flag, "value") else flag for flag in score.failure_flags
    )
    final_snapshot = resolve_final_snapshot_v3(runtime_row, task)
    return Task13CaseRecordV1(
        case_id=task13_case_id_v1(runtime_row.run_id, task.task_id, category),
        category=category,
        run_id=runtime_row.run_id,
        task_id=task.task_id,
        semantic_core_id=observation.semantic_core_id,
        answer_model_slot=observation.slot,
        k=observation.k,
        task_artifact_sha256=task_hash,
        task_manifest_sha256=task_manifest_hash,
        run_manifest_sha256=observation.source.run_manifest_sha256,
        score_artifact_sha256=observation.source.score_artifact_sha256,
        matrix_summary_sha256=summary_hash,
        task=Task13TaskProjectionV1(
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            family=task.task_family,
            difficulty=task.difficulty.value,
            metadata=_task_metadata_projection(task, private),
            source=source_payload,
            target_objects=tuple(_json(item) for item in task.target_objects),
            queries=tuple(_json(item) for item in task.queries),
            gold_actions=tuple(_json(item) for item in task.actions),
        ),
        timeline=Task13TimelineProjectionV1(
            redacted=private,
            items=timeline,
        ),
        run=Task13RunProjectionV1(
            run_id=runtime_row.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            completion_status=runtime_row.completion_status.value,
            parsed_actions=tuple(_json(item) for item in runtime_row.parsed_actions),
            memory_snapshots=tuple(_json(item) for item in runtime_row.memory_snapshots),
            final_state=(
                _json(final_snapshot)["state_by_object"]
                if final_snapshot is not None
                else None
            ),
            system_events=tuple(dict(item) for item in runtime_row.system_events),
            provenance=_json(runtime_row.parser_extractor_provenance),
            exceptions=tuple(dict(item) for item in runtime_row.exceptions),
        ),
        score=Task13ScoreProjectionV1(
            run_id=runtime_row.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            metric_layers=metric_layers,
            support=support,
            failure_flags=flags,
            primary_failure=score.primary_failure,
        ),
        retrieval=Task13RetrievalProjectionV1(
            run_id=runtime_row.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            available=bool(runtime_row.retrieval_traces),
            items=tuple(_json(item) for item in runtime_row.retrieval_traces),
        ),
        answer=Task13AnswerProjectionV1(
            run_id=runtime_row.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            available=bool(runtime_row.answer_predictions),
            items=tuple(_json(item) for item in runtime_row.answer_predictions),
        ),
    )


def _select_task13_cases_for_run_v1(
    run: Task13AuthenticatedRunV1,
    matrix: Task13AuthenticatedMatrixV1,
) -> tuple[Task13CaseRecordV1, ...]:
    if not isinstance(matrix, Task13AuthenticatedMatrixV1):
        raise TypeError("case selection requires a Task13AuthenticatedMatrixV1")
    observations = tuple(run.observations)
    if not observations:
        raise ValueError("each Task 13 run must contain authenticated observations")
    if any(observation.source != run.source for observation in observations):
        raise ValueError("case observations must share the authenticated run source")
    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.semantic_core_id.encode("utf-8"),
            observation.task.task_id.encode("utf-8"),
        ),
    )
    selected: list[Task13CaseRecordV1] = []
    selected_tasks: set[str] = set()
    for category in TASK13_CASE_CATEGORIES:
        candidate = next(
            (
                observation
                for observation in ordered
                if observation.task.task_id not in selected_tasks
                and classify_task13_case_v1(observation) == category
            ),
            None,
        )
        if candidate is None:
            continue
        case = _project_task13_case_v1(candidate, matrix)
        selected.append(case)
        selected_tasks.add(case.task_id)
    if not selected:
        raise ValueError("each Task 13 run must select at least one case")
    return tuple(selected)


def _coverage(run_id: str, cases: Sequence[Task13CaseRecordV1]) -> Task13RunCaseCoverageV1:
    by_category = {case.category: case.case_id for case in cases}
    return Task13RunCaseCoverageV1(
        run_id=run_id,
        correct_case_id=by_category.get("correct"),
        stale_copied_case_id=by_category.get("stale_copied"),
        answer_parse_invalid_case_id=by_category.get("answer_parse_invalid"),
        other_wrong_case_id=by_category.get("other_wrong"),
    )


def build_task13_cases_v1(matrix_input: Task13AuthenticatedMatrixV1) -> Task13CasesResultV1:
    matrix = _validate_authenticated_matrix(matrix_input)
    cases: list[Task13CaseRecordV1] = []
    coverage: list[Task13RunCaseCoverageV1] = []
    for run in matrix.runs:
        run_cases = _select_task13_cases_for_run_v1(run, matrix)
        cases.extend(run_cases)
        coverage.append(_coverage(run.source.run_id, run_cases))
    if not 18 <= len(cases) <= 72:
        raise AssertionError("Task 13 case export must contain between 18 and 72 cases")
    return Task13CasesResultV1(tuple(cases), tuple(coverage))


def verify_task13_cases_v1(
    cases: Sequence[Task13CaseRecordV1],
    matrix_input: Task13AuthenticatedMatrixV1,
) -> None:
    matrix = _validate_authenticated_matrix(matrix_input)
    supplied = tuple(cases)
    expected = build_task13_cases_v1(matrix).cases
    if len(supplied) != len(expected):
        raise ValueError("case export does not equal authenticated source evidence")
    if tuple(case.case_id for case in supplied) != tuple(case.case_id for case in expected):
        raise ValueError("case export does not equal authenticated source evidence")
    for candidate, source in zip(supplied, expected):
        if canonical_json_bytes(candidate) != canonical_json_bytes(source):
            raise ValueError("case export does not equal authenticated source evidence")


__all__ = [
    "Task13CasesResultV1",
    "build_task13_cases_v1",
    "classify_task13_case_v1",
    "verify_task13_cases_v1",
]
