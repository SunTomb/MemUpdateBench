from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.statistics.contracts_v3 import Task13RunSourceV1
from mub.vnext.statistics.input_v3 import (
    Task13AuthenticatedObservationV1,
    _task13_observation_evidence_sha256,
    _task13_observation_membership_root_sha256,
)
from mub.vnext.statistics.cases_v3 import (
    _project_task13_case_v1,
    _select_task13_cases_for_run_v1,
    build_task13_cases_v1,
    verify_task13_cases_v1,
)
from tests.vnext.task12_fixtures import ROOT
from tests.vnext.task13_input_fixtures import _compact_bundle, _prompted_row, _scores


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _run_config(run_id: str = "run-case-a"):
    return SimpleNamespace(
        run_id=run_id,
        adapter_info=SimpleNamespace(adapter_id="adapter-case"),
        action_parser_version="action-parser-v1",
        answer_parser_version="answer-parser-v1",
        memory_entry_extractor_version="entry-extractor-v1",
        object_value_extractor_config_hash=None,
        redaction_policy_version="redaction-v1",
        answer_model_slot="answer_model_a",
    )


def _scored(base: ScoreRecordV3, *, exact: float, stale: float) -> ScoreRecordV3:
    payload = base.model_dump(mode="json")
    payload["answer_scores"]["exact_match"] = exact
    payload["answer_scores"]["stale_copied"] = stale
    payload["protocol_scores"]["answer_parse_valid"] = True
    for path in (
        "answer_scores.exact_match",
        "answer_scores.stale_copied",
        "protocol_scores.answer_parse_valid",
    ):
        payload["supported_metric_fields"].pop(path)
    return ScoreRecordV3.model_validate(payload)


def _invalid_answer(row):
    payload = row.model_dump(mode="json")
    payload["answer_predictions"][0]["parsed_answer"] = None
    payload["answer_predictions"][0]["format_valid"] = False
    payload["answer_predictions"][0]["error_flags"] = ["invalid-answer"]
    return type(row).model_validate(payload)


@pytest.fixture(scope="module")
def case_fixture():
    tasks = _compact_bundle(ROOT).snapshot.tasks[:5]
    config = _run_config()
    source = Task13RunSourceV1(
        run_id=config.run_id,
        run_manifest_sha256=SHA_A,
        score_artifact_sha256=SHA_B,
    )
    base_scores = _scores(tasks, config)
    categories = (
        (1.0, 0.0, False),
        (0.0, 1.0, False),
        (0.0, 0.0, True),
        (0.0, 0.0, False),
        (1.0, 0.0, False),
    )
    observations = []
    for task, base_score, (exact, stale, invalid) in zip(tasks, base_scores, categories):
        row = _prompted_row(task, config).model_copy(
            update={
                "memory_snapshots": (
                    MemorySnapshotV3(
                        after_event_id=task.events[-1].event_id,
                        state_by_object={
                            task.target_objects[0].canonical_id: "case-final-value"
                        },
                        store_size=1,
                    ),
                )
            }
        )
        if invalid:
            row = _invalid_answer(row)
        observations.append(
            Task13AuthenticatedObservationV1(
                cell_id="cell-case-a",
                slot="answer_model_a",
                k=4,
                context_order="chronological",
                context_annotation="none",
                semantic_core_id=task.metadata.split_key.semantic_core_id,
                task=task,
                run=row,
                score=_scored(base_score, exact=exact, stale=stale),
                source=source,
            )
        )
    run = SimpleNamespace(
        source=source,
        observations=tuple(observations),
        run_configuration=config,
    )
    matrix = SimpleNamespace(
        runs=(run,),
        input_hashes={
            "core_tasks": SHA_C,
            "core_task_manifest": SHA_A,
            "task12_matrix_summary": SHA_B,
        },
    )
    full_runs = []
    for index in range(18):
        run_id = f"run-case-{index:02d}"
        cloned_source = Task13RunSourceV1(
            run_id=run_id,
            run_manifest_sha256=SHA_A,
            score_artifact_sha256=SHA_B,
        )
        cloned_observations = tuple(
            replace(
                observation,
                cell_id=f"cell-case-{index:02d}",
                run=observation.run.model_copy(update={"run_id": run_id}),
                score=observation.score.model_copy(update={"run_id": run_id}),
                source=cloned_source,
            )
            for observation in observations
        )
        full_runs.append(
            SimpleNamespace(
                source=cloned_source,
                observations=cloned_observations,
                run_configuration=_run_config(run_id),
            )
        )
    full_matrix = SimpleNamespace(
        runs=tuple(full_runs),
        input_hashes=matrix.input_hashes,
    )
    return SimpleNamespace(
        run=run,
        matrix=matrix,
        full_matrix=full_matrix,
        observations=tuple(observations),
    )


def test_case_selection_is_stratified_and_order_invariant(authenticated_case_matrix):
    matrix = authenticated_case_matrix
    run = matrix.runs[0]
    forward = _select_task13_cases_for_run_v1(run, matrix)
    shuffled_run = replace(run, observations=tuple(reversed(run.observations)))
    reverse = _select_task13_cases_for_run_v1(shuffled_run, matrix)

    assert tuple(case.case_id for case in forward) == tuple(case.case_id for case in reverse)
    assert len({case.task_id for case in forward}) == len(forward)
    assert all(case.run_id == run.source.run_id for case in forward)


def test_case_metrics_are_copied_not_recomputed(authenticated_case_matrix):
    observation = authenticated_case_matrix.runs[0].observations[0]
    case = _project_task13_case_v1(observation, authenticated_case_matrix)

    assert case.score.metric_layers["answer_scores"] == observation.score.answer_scores.model_dump(mode="json")
    assert case.score.support == {
        path: support.model_dump(mode="json")
        for path, support in observation.score.supported_metric_fields.items()
    }
    assert case.run.model_dump(mode="json")["final_state"] is None
    assert case.task.model_dump(mode="json")["gold_actions"] == [
        action.model_dump(mode="json") for action in observation.task.actions
    ]


def test_public_case_source_preserves_authenticated_source_fields(authenticated_case_matrix):
    observation = authenticated_case_matrix.runs[0].observations[0]
    case = _project_task13_case_v1(observation, authenticated_case_matrix)

    expected_source = observation.task.source.model_dump(mode="json")
    expected_source["redacted"] = False
    assert case.task.source == expected_source
    assert case.task.source["provenance"] == observation.task.source.provenance
    assert case.task.source["generator"] == (
        observation.task.source.generator.model_dump(mode="json")
        if observation.task.source.generator is not None
        else None
    )
    assert case.timeline.redacted is False


@pytest.mark.parametrize(
    "provenance",
    ({}, {"redistributable": "false"}),
    ids=("missing-redistributable", "string-false-redistributable"),
)
def test_unknown_redistributability_defaults_to_private(authenticated_case_matrix, provenance):
    matrix = authenticated_case_matrix
    run = matrix.runs[0]
    observation = run.observations[0]
    private_source = observation.task.source.model_copy(update={"provenance": provenance})
    private_task = observation.task.model_copy(update={"source": private_source})
    private_observation = replace(
        observation,
        task=private_task,
        evidence_sha256="",
    )
    private_observation = replace(
        private_observation,
        evidence_sha256=_task13_observation_evidence_sha256(private_observation),
    )
    task_hashes = dict(run.run_configuration.task_record_hashes)
    task_hashes[private_task.task_id] = sha256_model(private_task)
    private_config = run.run_configuration.model_copy(update={"task_record_hashes": task_hashes})
    private_run = replace(
        run,
        observations=(private_observation, *run.observations[1:]),
        run_configuration=private_config,
    )
    private_matrix = replace(matrix, runs=(private_run, *matrix.runs[1:]))

    case = _project_task13_case_v1(private_observation, private_matrix)

    assert case.timeline.redacted is True
    assert case.task.source["source_uri"] is None
    assert "provenance" not in case.task.source
    assert "generator" not in case.task.source
    assert all("raw_text" not in item for item in case.timeline.items)
    assert all("normalized_text" not in item for item in case.timeline.items)


def test_case_verifier_rejects_changed_score_or_trace(authenticated_case_matrix):
    result = build_task13_cases_v1(authenticated_case_matrix)
    verify_task13_cases_v1(result.cases, authenticated_case_matrix)

    first = result.cases[0]
    changed_layers = dict(first.score.metric_layers)
    changed_answer = dict(changed_layers["answer_scores"])
    changed_answer["exact_match"] = 0.125
    changed_layers["answer_scores"] = changed_answer
    changed_score = first.score.model_copy(update={"metric_layers": changed_layers})
    changed_case = first.model_copy(update={"score": changed_score})
    with pytest.raises(ValueError, match="does not equal authenticated source evidence"):
        verify_task13_cases_v1((changed_case, *result.cases[1:]), authenticated_case_matrix)

    first_run = authenticated_case_matrix.runs[0]
    changed_row = first_run.observations[0].run.model_copy(update={"system_events": ({"changed": True},)})
    changed_observation = replace(first_run.observations[0], run=changed_row)
    changed_run = replace(first_run, observations=(changed_observation, *first_run.observations[1:]))
    changed_matrix = replace(authenticated_case_matrix, runs=(changed_run, *authenticated_case_matrix.runs[1:]))
    with pytest.raises(ValueError, match="authenticated|source|linkage|digest"):
        verify_task13_cases_v1(result.cases, changed_matrix)


def test_case_build_and_verify_have_no_run_count_bypass(case_fixture):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_task13_cases_v1(case_fixture.matrix, require_18_runs=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        verify_task13_cases_v1((), case_fixture.matrix, require_18_runs=False)


def test_case_build_rejects_rebound_run_configuration_digest(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    forged_config = run.run_configuration.model_copy(
        update={"answer_parser_version": "forged-answer-parser-v999"}
    )
    forged_run = replace(run, run_configuration=forged_config)
    forged_matrix = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    with pytest.raises(ValueError, match="run configuration digest"):
        build_task13_cases_v1(forged_matrix)


def test_case_build_rejects_rebound_authorization_digest(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    forged_authorization = run.authorization.model_copy(
        update={"preparation_manifest_sha256": "0" * 64}
    )
    forged_run = replace(run, authorization=forged_authorization)
    forged_matrix = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    with pytest.raises(ValueError, match="authorization digest"):
        build_task13_cases_v1(forged_matrix)


def test_case_build_rejects_rebound_observation_evidence(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    observation = run.observations[0]
    support_path, support = next(iter(observation.score.supported_metric_fields.items()))
    forged_support = dict(observation.score.supported_metric_fields)
    forged_support[support_path] = support.model_copy(update={"detail": "forged-evidence"})
    forged_score = observation.score.model_copy(
        update={"supported_metric_fields": forged_support}
    )
    forged_observation = replace(observation, score=forged_score)
    forged_run = replace(run, observations=(forged_observation, *run.observations[1:]))
    forged_matrix = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    with pytest.raises(ValueError, match="observation evidence digest"):
        build_task13_cases_v1(forged_matrix)


def test_case_build_rejects_rehashed_rebound_observation_membership(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    observation = run.observations[0]
    support_path, support = next(iter(observation.score.supported_metric_fields.items()))
    forged_support = dict(observation.score.supported_metric_fields)
    forged_support[support_path] = support.model_copy(update={"detail": "forged-membership"})
    forged_score = observation.score.model_copy(
        update={"supported_metric_fields": forged_support}
    )
    forged_observation = replace(observation, score=forged_score, evidence_sha256="")
    forged_observation = replace(
        forged_observation,
        evidence_sha256=_task13_observation_evidence_sha256(forged_observation),
    )
    forged_observations = (forged_observation, *run.observations[1:])
    forged_run = replace(
        run,
        observations=forged_observations,
        observation_membership_root_sha256=_task13_observation_membership_root_sha256(
            forged_observations
        ),
    )
    forged_matrix = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    forged_root = _task13_observation_membership_root_sha256(forged_observations)
    forged_roots = dict(authenticated_case_matrix.observation_membership_roots)
    forged_roots[run.source.run_id] = forged_root
    try:
        rebound_matrix = replace(
            forged_matrix,
            observation_membership_roots=forged_roots,
        )
    except ValueError as exc:
        assert "init=False" in str(exc)
        rebound_matrix = forged_matrix
    with pytest.raises(ValueError, match="membership root|loader seal"):
        build_task13_cases_v1(rebound_matrix)


def test_private_task_metadata_projection_excludes_free_text(authenticated_case_matrix):
    matrix = authenticated_case_matrix
    run = matrix.runs[0]
    observation = run.observations[0]
    private_source = observation.task.source.model_copy(
        update={"provenance": {"redistributable": False}}
    )
    private_metadata = observation.task.metadata.model_copy(
        update={
            "resolved_profile": {"sentinel": "DO_NOT_EXPORT_PROFILE"},
            "extra": {"sentinel": "DO_NOT_EXPORT_EXTRA"},
        }
    )
    private_task = observation.task.model_copy(
        update={"source": private_source, "metadata": private_metadata}
    )
    private_observation = replace(
        observation,
        task=private_task,
        evidence_sha256="",
    )
    private_observation = replace(
        private_observation,
        evidence_sha256=_task13_observation_evidence_sha256(private_observation),
    )
    task_hashes = dict(run.run_configuration.task_record_hashes)
    task_hashes[private_task.task_id] = sha256_model(private_task)
    private_config = run.run_configuration.model_copy(update={"task_record_hashes": task_hashes})
    private_run = replace(
        run,
        observations=(private_observation, *run.observations[1:]),
        run_configuration=private_config,
    )
    private_matrix = replace(matrix, runs=(private_run, *matrix.runs[1:]))

    case = _project_task13_case_v1(private_observation, private_matrix)
    serialized = canonical_json_bytes(case)
    assert b"DO_NOT_EXPORT_PROFILE" not in serialized
    assert b"DO_NOT_EXPORT_EXTRA" not in serialized
    assert "resolved_profile" not in case.task.metadata
    assert "legacy_provenance" not in case.task.metadata
    assert "extra" not in case.task.metadata
    assert case.task.source["source_uri"] is None
    assert case.timeline.redacted is True


@pytest.fixture(scope="module")
def authenticated_case_matrix(tmp_path_factory):
    from mub.vnext.runtime.task12_execution_v3 import Task12RuntimeCodeBindingV1
    from mub.vnext.statistics.input_v3 import load_task13_authenticated_matrix_v1
    from tests.vnext.task13_input_fixtures import build_compact_authenticated_task13_fixture

    fixture = build_compact_authenticated_task13_fixture(
        tmp_path_factory.mktemp("task13-cases-authenticated"),
        ROOT,
        Task12RuntimeCodeBindingV1(code_revision="8" * 40, code_tree_sha256="9" * 64),
    )
    expected = fixture["expected_hashes"]
    return load_task13_authenticated_matrix_v1(
        preparation_manifest_path=fixture["preparation_manifest_path"],
        plan_path=fixture["plan_path"],
        core_root=fixture["inputs"]["core_root"],
        evidence_root=fixture["inputs"]["evidence_root"],
        matrix_root=fixture["matrix"].matrix_root,
        matrix_manifest_path=fixture["matrix_manifest_path"],
        matrix_summary_path=fixture["summary_path"],
        integrity_audit_path=fixture["audit_path"],
        repository_root=ROOT,
        expected_preparation_manifest_sha256=expected["preparation_manifest"],
        expected_plan_sha256=expected["plan"],
        expected_matrix_manifest_sha256=expected["matrix_manifest"],
        expected_matrix_summary_sha256=expected["matrix_summary"],
        expected_integrity_audit_sha256=expected["integrity_audit"],
    )


def test_case_build_requires_the_authenticated_matrix_type(case_fixture):
    with pytest.raises(TypeError, match="Task13AuthenticatedMatrixV1"):
        build_task13_cases_v1(case_fixture.full_matrix)


def test_case_build_revalidates_exact_production_shape(authenticated_case_matrix):
    matrix = authenticated_case_matrix
    assert len(matrix.runs) == 18
    assert all(len(run.observations) == 80 for run in matrix.runs)
    truncated = replace(
        matrix,
        runs=(
            replace(matrix.runs[0], observations=matrix.runs[0].observations[:-1]),
            *matrix.runs[1:],
        ),
    )
    with pytest.raises(ValueError, match="exactly 80 observations"):
        build_task13_cases_v1(truncated)


def test_case_build_rejects_duplicate_task_observation(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    duplicate = replace(
        run,
        observations=(run.observations[0], run.observations[0], *run.observations[2:]),
    )
    forged = replace(authenticated_case_matrix, runs=(duplicate, *authenticated_case_matrix.runs[1:]))
    with pytest.raises(ValueError, match="unique task|canonical task identity"):
        build_task13_cases_v1(forged)


def test_case_build_rejects_cross_run_task_identity_mismatch(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[1]
    observation = run.observations[0]
    forged_observation = replace(observation, semantic_core_id="foreign-core")
    forged_run = replace(run, observations=(forged_observation, *run.observations[1:]))
    forged = replace(
        authenticated_case_matrix,
        runs=(authenticated_case_matrix.runs[0], forged_run, *authenticated_case_matrix.runs[2:]),
    )
    with pytest.raises(ValueError, match="semantic-core|canonical task identity|identity sequence|membership root"):
        build_task13_cases_v1(forged)


def test_case_build_rejects_noncanonical_run_order(authenticated_case_matrix):
    with pytest.raises(ValueError, match="run order|canonical"):
        build_task13_cases_v1(
            replace(authenticated_case_matrix, runs=tuple(reversed(authenticated_case_matrix.runs)))
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_id", "forged-cell"),
        ("slot", "answer_model_b"),
        ("k", 8),
        ("context_order", "reverse_chronological"),
        ("context_annotation", "latest_outdated_label"),
    ],
)
def test_case_build_rejects_coordinate_relabeling(authenticated_case_matrix, field, value):
    run = authenticated_case_matrix.runs[0]
    observation = run.observations[0]
    forged_observation = replace(observation, **{field: value})
    forged_run = replace(run, observations=(forged_observation, *run.observations[1:]))
    forged = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    with pytest.raises(ValueError, match="coordinate|cell|slot|k|context"):
        build_task13_cases_v1(forged)


def test_case_build_rejects_spoofed_input_hashes(authenticated_case_matrix):
    spoofed = dict(authenticated_case_matrix.input_hashes)
    spoofed["core_tasks"] = "f" * 64
    with pytest.raises(ValueError, match="input hash|authenticated"):
        build_task13_cases_v1(replace(authenticated_case_matrix, input_hashes=spoofed))


def test_case_build_rejects_spoofed_source_hash(authenticated_case_matrix):
    run = authenticated_case_matrix.runs[0]
    observation = run.observations[0]
    forged_observation = replace(
        observation,
        source=Task13RunSourceV1(
            run_id=observation.source.run_id,
            run_manifest_sha256="f" * 64,
            score_artifact_sha256=observation.source.score_artifact_sha256,
        ),
    )
    forged_run = replace(run, observations=(forged_observation, *run.observations[1:]))
    forged = replace(
        authenticated_case_matrix,
        runs=(forged_run, *authenticated_case_matrix.runs[1:]),
    )
    with pytest.raises(ValueError, match="source|hash"):
        build_task13_cases_v1(forged)


def test_case_projection_redacts_nested_private_fields(authenticated_case_matrix):
    matrix = authenticated_case_matrix
    run = matrix.runs[0]
    observation = run.observations[0]
    private_source = observation.task.source.model_copy(
        update={
            "provenance": {
                "redistributable": False,
                "nested_secret": {"token": "DO_NOT_EXPORT"},
            }
        }
    )
    private_events = tuple(
        event.model_copy(
            update={
                "metadata": {"secret": "DO_NOT_EXPORT"},
                "source_anchor": {"secret": "DO_NOT_EXPORT"},
                "raw_text": "DO_NOT_EXPORT",
                "normalized_text": "DO_NOT_EXPORT",
            }
        )
        for event in observation.task.events
    )
    private_task = observation.task.model_copy(update={"source": private_source, "events": private_events})
    private_observation = replace(
        observation,
        task=private_task,
        evidence_sha256="",
    )
    private_observation = replace(
        private_observation,
        evidence_sha256=_task13_observation_evidence_sha256(private_observation),
    )
    task_hashes = dict(run.run_configuration.task_record_hashes)
    task_hashes[private_task.task_id] = sha256_model(private_task)
    private_config = run.run_configuration.model_copy(update={"task_record_hashes": task_hashes})
    private_run = replace(run, observations=(private_observation, *run.observations[1:]), run_configuration=private_config)
    private_matrix = replace(matrix, runs=(private_run, *matrix.runs[1:]))
    case = _project_task13_case_v1(private_observation, private_matrix)
    serialized = canonical_json_bytes(case)
    assert b"DO_NOT_EXPORT" not in serialized
    assert "provenance" not in case.task.source
    assert all("metadata" not in item and "source_anchor" not in item for item in case.timeline.items)


def test_final_snapshot_resolves_by_event_chronology_and_rejects_bad_anchors(case_fixture):
    from mub.vnext.scoring.scorer_v3 import resolve_final_snapshot_v3

    observation = case_fixture.observations[0]
    early = MemorySnapshotV3(
        after_event_id=observation.task.events[0].event_id,
        state_by_object={"object": "early"},
        store_size=1,
    )
    late = MemorySnapshotV3(
        after_event_id=observation.task.events[-1].event_id,
        state_by_object={"object": "late"},
        store_size=1,
    )
    reversed_run = observation.run.model_copy(update={"memory_snapshots": (late, early)})
    assert resolve_final_snapshot_v3(reversed_run, observation.task).state_by_object == {"object": "late"}

    duplicate = observation.run.model_copy(update={"memory_snapshots": (early, early)})
    with pytest.raises(ValueError, match="duplicate"):
        resolve_final_snapshot_v3(duplicate, observation.task)
    unknown = observation.run.model_copy(
        update={"memory_snapshots": (MemorySnapshotV3(after_event_id="unknown", store_size=0),)}
    )
    with pytest.raises(ValueError, match="unknown"):
        resolve_final_snapshot_v3(unknown, observation.task)
