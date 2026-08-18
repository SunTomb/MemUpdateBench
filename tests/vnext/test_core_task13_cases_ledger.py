from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.common import ArtifactRef, MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
from mub.vnext.contracts.v3.score import ScoreRecordV3
from mub.vnext.io import canonical_json_bytes, sha256_model
from mub.vnext.statistics.contracts_v3 import (
    TASK13_METRIC_PATHS,
    Task13CellStatisticV1,
    Task13CaseBindingV1,
    Task13CaseIndexV1,
    Task13DenominatorV1,
    Task13IntervalV1,
    Task13PairedContrastV1,
    Task13RunCaseCoverageV1,
    Task13RunSourceV1,
    Task13StatisticStatus,
    Task13StatisticsReceiptV1,
)
from mub.vnext.statistics.ledger_v3 import (
    Task13LedgerResultV1,
    build_task13_case_index_v1,
    build_task13_claim_ledger_v1,
    build_task13_statistics_receipt_v1,
)
from mub.vnext.statistics.input_v3 import (
    Task13AuthenticatedObservationV1,
    _task13_observation_evidence_sha256,
    _task13_observation_membership_root_sha256,
)
from mub.vnext.statistics.statistics_v3 import task13_contrast_id_v1
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
    except (ValueError, TypeError) as exc:
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


_LEDGER_SHA = tuple(f"{index:064x}" for index in range(1, 128))


def _ledger_source(index: int) -> Task13RunSourceV1:
    return Task13RunSourceV1(
        run_id=f"run-ledger-{index:02d}",
        run_manifest_sha256=_LEDGER_SHA[index],
        score_artifact_sha256=_LEDGER_SHA[index + 1],
    )


def _ledger_interval(unsupported: bool = False) -> Task13IntervalV1:
    if unsupported:
        support = MetricFieldSupport(
            reason=SupportReason.NOT_SUPPORTED,
            null_policy="emit_null",
            detail="ledger fixture",
        )
        return Task13IntervalV1(
            status=Task13StatisticStatus.UNSUPPORTED,
            support=support,
            support_sha256=sha256_model(support),
        )
    return Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC,
        estimate="0.5",
        lower="0",
        upper="1",
    )


def _ledger_statistics() -> tuple[tuple[Task13CellStatisticV1, ...], tuple[Task13PairedContrastV1, ...]]:
    cells = []
    cell_sources = {}
    cell_conditions = ("chronological-none", "reverse-none", "reverse-label")
    run_index = 0
    for slot in ("answer_model_a", "answer_model_b"):
        for k in (4, 8, 16):
            for condition in cell_conditions:
                cell_id = f"{slot}-k{k:02d}-{condition}"
                source = _ledger_source(run_index)
                run_index += 1
                cell_sources[(slot, k, cell_id)] = source
                for metric_index, metric_path in enumerate(TASK13_METRIC_PATHS):
                    cells.append(
                        Task13CellStatisticV1(
                            cell_id=cell_id,
                            answer_model_slot=slot,
                            k=k,
                            metric_path=metric_path,
                            interval=_ledger_interval(
                                unsupported=(
                                    slot == "answer_model_a"
                                    and k == 4
                                    and condition == cell_conditions[0]
                                    and metric_index == 0
                                )
                            ),
                            task_count=80,
                            core_count=20,
                            core_ids_sha256=_LEDGER_SHA[0],
                            run_id=source.run_id,
                            run_manifest_sha256=source.run_manifest_sha256,
                            score_artifact_sha256=source.score_artifact_sha256,
                            bootstrap_config_sha256=_LEDGER_SHA[2],
                            bootstrap_indices_sha256=_LEDGER_SHA[3],
                        )
                    )
    contrasts = []
    for slot in ("answer_model_a", "answer_model_b"):
        for k in (4, 8, 16):
            pairs = (
                (cell_conditions[1], cell_conditions[0]),
                (cell_conditions[2], cell_conditions[1]),
            )
            for left_condition, right_condition in pairs:
                left_id = f"{slot}-k{k:02d}-{left_condition}"
                right_id = f"{slot}-k{k:02d}-{right_condition}"
                left = cell_sources[(slot, k, left_id)]
                right = cell_sources[(slot, k, right_id)]
                for metric_path in TASK13_METRIC_PATHS:
                    contrast_id = task13_contrast_id_v1(
                        slot,
                        k,
                        left_id,
                        right_id,
                        metric_path,
                    )
                    contrasts.append(
                        Task13PairedContrastV1(
                            contrast_id=contrast_id,
                            left_cell_id=left_id,
                            right_cell_id=right_id,
                            direction="left_minus_right",
                            answer_model_slot=slot,
                            k=k,
                            metric_path=metric_path,
                            interval=_ledger_interval(),
                            core_count=20,
                            core_ids_sha256=_LEDGER_SHA[0],
                            left_source=left,
                            right_source=right,
                            bootstrap_config_sha256=_LEDGER_SHA[2],
                            bootstrap_indices_sha256=_LEDGER_SHA[3],
                        )
                    )
    assert len(cells) == 126
    assert len(contrasts) == 84
    return tuple(cells), tuple(contrasts)


def _ledger_case_index() -> Task13CaseIndexV1:
    sources = tuple(_ledger_source(index) for index in range(18))
    bindings = tuple(
        Task13CaseBindingV1(
            case_id=f"case-ledger-{index:02d}",
            run_id=source.run_id,
            task_id=f"task-ledger-{index:02d}",
            category="other_wrong",
        )
        for index, source in enumerate(sources)
    )
    coverage = tuple(
        Task13RunCaseCoverageV1(
            run_id=source.run_id,
            other_wrong_case_id=binding.case_id,
        )
        for source, binding in zip(sources, bindings)
    )
    return build_task13_case_index_v1(
        case_bindings=bindings,
        coverage=coverage,
        run_sources=sources,
        cases_artifact=ArtifactRef(
            path="cases.jsonl",
            sha256=_LEDGER_SHA[4],
            media_type="application/jsonl",
        ),
    )


def _ledger_receipt(
    cells,
    contrasts,
    *,
    statistics_config_sha256=_LEDGER_SHA[2],
) -> Task13StatisticsReceiptV1:
    return build_task13_statistics_receipt_v1(
        cells,
        contrasts,
        task12_preparation_manifest_sha256=_LEDGER_SHA[5],
        task12_plan_sha256=_LEDGER_SHA[6],
        task12_matrix_manifest_sha256=_LEDGER_SHA[7],
        task12_matrix_summary_sha256=_LEDGER_SHA[8],
        task12_integrity_audit_sha256=_LEDGER_SHA[9],
        statistics_config_sha256=statistics_config_sha256,
        task13_runtime_revision="a" * 40,
        task13_runtime_tree_sha256=_LEDGER_SHA[11],
        core_ids_sha256=_LEDGER_SHA[0],
        bootstrap_indices_sha256=_LEDGER_SHA[3],
        cell_statistics_artifact=ArtifactRef(
            path="cell_statistics.jsonl",
            sha256=_LEDGER_SHA[12],
            media_type="application/jsonl",
        ),
        paired_contrasts_artifact=ArtifactRef(
            path="paired_contrasts.jsonl",
            sha256=_LEDGER_SHA[13],
            media_type="application/jsonl",
        ),
    )


def _ledger_build():
    cells, contrasts = _ledger_statistics()
    receipt = _ledger_receipt(cells, contrasts)
    case_index = _ledger_case_index()
    result = build_task13_claim_ledger_v1(
        cells,
        contrasts,
        receipt=receipt,
        case_index=case_index,
    )
    return cells, contrasts, receipt, case_index, result


def test_ledger_has_one_row_per_statistic_and_contrast():
    cells, contrasts, receipt, case_index, result = _ledger_build()

    assert isinstance(result, Task13LedgerResultV1)
    assert len(result.claims) == 210
    assert len([claim for claim in result.claims if claim.kind == "direct_cell"]) == len(cells)
    assert len([claim for claim in result.claims if claim.kind == "paired_contrast"]) == len(contrasts)
    assert len({claim.claim_id for claim in result.claims}) == 210
    assert receipt.cell_statistic_count == 126
    assert receipt.paired_contrast_count == 84
    assert receipt.task_count == 1440
    assert receipt.semantic_core_count == 20
    assert receipt.run_count == 18
    assert receipt.cell_statistics_artifact.path == "cell_statistics.jsonl"
    assert receipt.cell_statistics_artifact.sha256 == _LEDGER_SHA[12]
    assert receipt.paired_contrasts_artifact.path == "paired_contrasts.jsonl"
    assert receipt.paired_contrasts_artifact.sha256 == _LEDGER_SHA[13]
    assert case_index.record_count == 18


def test_ledger_ids_are_stable_under_input_shuffle():
    cells, contrasts, receipt, case_index, forward = _ledger_build()
    reverse = build_task13_claim_ledger_v1(
        tuple(reversed(cells)),
        tuple(reversed(contrasts)),
        receipt=receipt,
        case_index=case_index,
    )

    assert canonical_json_bytes(forward.claims[0]) == canonical_json_bytes(reverse.claims[0])
    assert tuple(claim.claim_id for claim in forward.claims) == tuple(
        claim.claim_id for claim in reverse.claims
    )
    assert canonical_json_bytes(forward.claims[-1]) == canonical_json_bytes(reverse.claims[-1])


def test_ledger_binds_receipt_case_index_runs_and_scores():
    _, _, receipt, case_index, result = _ledger_build()
    receipt_sha256 = sha256_model(receipt)
    case_index_sha256 = sha256_model(case_index)

    for claim in result.claims:
        assert claim.statistics_receipt_sha256 == receipt_sha256
        assert claim.case_index_sha256 == case_index_sha256
        assert claim.case_ids
        source_run_ids = {source.run_id for source in claim.run_sources}
        assert all(
            case_index.case_bindings[index].run_id in source_run_ids
            for index, _ in enumerate(case_index.case_bindings)
            if case_index.case_bindings[index].case_id in claim.case_ids
        )
        assert all(source.run_id in {item.run_id for item in case_index.run_sources} for source in claim.run_sources)


def test_ledger_preserves_unsupported_as_null():
    _, _, _, _, result = _ledger_build()
    unsupported = next(
        claim for claim in result.claims if claim.kind == "direct_cell" and claim.status is Task13StatisticStatus.UNSUPPORTED
    )

    assert unsupported.interval.status is Task13StatisticStatus.UNSUPPORTED
    assert unsupported.interval.estimate is None
    assert unsupported.interval.lower is None
    assert unsupported.interval.upper is None
    assert unsupported.denominator == Task13DenominatorV1(
        task_count=80,
        semantic_core_count=20,
        tasks_per_core=4,
    )


def test_ledger_rejects_duplicate_missing_and_foreign_statistics():
    cells, contrasts = _ledger_statistics()
    receipt = _ledger_receipt(cells, contrasts)
    case_index = _ledger_case_index()

    with pytest.raises(ValueError, match="126|duplicate|complete"):
        build_task13_claim_ledger_v1(
            (*cells[:-1], cells[0]),
            contrasts,
            receipt=receipt,
            case_index=case_index,
        )
    with pytest.raises(ValueError, match="foreign|cell|source"):
        foreign = cells[0].model_copy(update={"cell_id": "foreign-cell"})
        build_task13_claim_ledger_v1(
            (foreign, *cells[1:]),
            contrasts,
            receipt=receipt,
            case_index=case_index,
        )


def test_ledger_rejects_altered_receipt_and_case_hash_bindings():
    cells, contrasts, receipt, case_index, _ = _ledger_build()
    with pytest.raises(ValueError, match="receipt hash"):
        build_task13_claim_ledger_v1(
            cells,
            contrasts,
            receipt=receipt.model_copy(update={"task13_runtime_revision": "b" * 40}),
            case_index=case_index,
            expected_statistics_receipt_sha256=sha256_model(receipt),
        )
    with pytest.raises(ValueError, match="case index hash"):
        build_task13_claim_ledger_v1(
            cells,
            contrasts,
            receipt=receipt,
            case_index=case_index,
            expected_case_index_sha256="0" * 64,
        )


def test_receipt_rejects_statistics_config_different_from_bootstrap_config():
    cells, contrasts = _ledger_statistics()

    with pytest.raises(ValueError, match="statistics.*config|bootstrap.*config"):
        _ledger_receipt(
            cells,
            contrasts,
            statistics_config_sha256=_LEDGER_SHA[10],
        )


def test_case_index_rejects_noncanonical_case_binding_order():
    index = _ledger_case_index()

    with pytest.raises(ValueError, match="canonical|ordered"):
        build_task13_case_index_v1(
            case_bindings=tuple(reversed(index.case_bindings)),
            coverage=index.coverage,
            run_sources=index.run_sources,
            cases_artifact=index.cases_artifact,
        )


def test_case_index_joint_source_and_coverage_reorder_is_a_new_supplied_authority():
    index = _ledger_case_index()
    sources = tuple(reversed(index.run_sources))
    coverage = tuple(reversed(index.coverage))
    bindings = tuple(reversed(index.case_bindings))

    rebuilt = build_task13_case_index_v1(
        case_bindings=bindings,
        coverage=coverage,
        run_sources=sources,
        cases_artifact=index.cases_artifact,
    )

    assert tuple(source.run_id for source in rebuilt.run_sources) == tuple(
        source.run_id for source in sources
    )
    assert tuple(row.run_id for row in rebuilt.coverage) == tuple(
        row.run_id for row in coverage
    )
    assert tuple(binding.case_id for binding in rebuilt.case_bindings) == tuple(
        binding.case_id for binding in bindings
    )


def test_ledger_rejects_changed_contrast_id():
    cells, contrasts = _ledger_statistics()
    target_id = contrasts[0].contrast_id
    forged_contrasts = tuple(
        record.model_copy(update={"contrast_id": "forged-contrast-id"})
        if record.contrast_id == target_id
        else record
        for record in contrasts
    )
    receipt = _ledger_receipt(cells, contrasts)
    case_index = _ledger_case_index()

    with pytest.raises(ValueError, match="contrast ID|contrast_id|deterministic"):
        build_task13_claim_ledger_v1(
            cells,
            forged_contrasts,
            receipt=receipt,
            case_index=case_index,
        )


def test_ledger_rejects_tampered_source_and_case_coverage():
    cells, contrasts = _ledger_statistics()
    receipt = _ledger_receipt(cells, contrasts)
    case_index = _ledger_case_index()
    tampered = cells[0].model_copy(update={"run_manifest_sha256": "f" * 64})
    with pytest.raises(ValueError, match="source|case index"):
        build_task13_claim_ledger_v1(
            (tampered, *cells[1:]),
            contrasts,
            receipt=receipt,
            case_index=case_index,
        )
    missing_coverage = case_index.model_copy()
    object.__setattr__(
        missing_coverage,
        "coverage",
        case_index.coverage[:-1] + (case_index.coverage[0],),
    )
    with pytest.raises(ValueError, match="coverage|run IDs|case"):
        build_task13_claim_ledger_v1(
            cells,
            contrasts,
            receipt=receipt,
            case_index=missing_coverage,
        )
