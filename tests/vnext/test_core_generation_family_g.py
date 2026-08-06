from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from mub.vnext.contracts.enums import Split, SupportReason, TaskFamily
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"
FULL_CAPABILITIES = AdapterCapabilitiesV3(
    supports_isolated_reset=True,
    supports_event_ingest=True,
    supports_add=True,
    supports_update=True,
    supports_native_answer=True,
    supports_multi_object_query=True,
    exports_version_history=True,
    exports_entries=True,
    exports_raw_state=True,
    exports_source_event_ids=True,
    exports_timestamps_or_order=True,
    exports_object_keys=True,
    exports_values=True,
    exports_retrieval_ids=True,
    exports_retrieval_scores=True,
    exports_action_trace=True,
    exports_evidence_linkage=True,
)
ORACLE_INFO = AdapterInfoV3(
    adapter_id="family-g-oracle",
    adapter_version="1",
    system_name="reference",
    system_version="1",
    configuration_hash="a" * 64,
)
SYNTHESIS_PATHS = (
    "synthesis_scores.multi_hop_accuracy",
    "synthesis_scores.multi_object_accuracy",
    "synthesis_scores.evidence_precision",
    "synthesis_scores.evidence_recall",
    "synthesis_scores.evidence_f1",
    "synthesis_scores.reasoning_support_accuracy",
    "synthesis_scores.stale_propagation_rate",
)


def _config():
    return load_core_config(CORE_CONFIG)


def _api():
    from mub.vnext.generation.family_g import (
        compile_family_g_micro_pilot,
        generate_core_family_g_cores,
        validate_family_g_core,
        validate_family_g_micro_core,
        validate_family_g_micro_task,
        validate_family_g_task,
    )

    return (
        generate_core_family_g_cores,
        validate_family_g_core,
        validate_family_g_task,
        validate_family_g_micro_core,
        validate_family_g_micro_task,
        compile_family_g_micro_pilot,
    )


def test_family_g_public_generation_exports_are_available():
    import mub.vnext.generation as generation

    assert generation.FAMILY_G_MICRO_PROFILE_ID == "family_g_selected_micro_v1"
    assert generation.FAMILY_G_SYNTHESIS_KINDS == (
        "update_sensitive_multi_hop",
        "multi_object_current_consistency",
    )
    assert generation.compile_family_g_micro_pilot is not None
    assert generation.generate_core_family_g_cores is not None
    assert generation.validate_family_g_core is not None
    assert generation.validate_family_g_micro_core is not None
    assert generation.validate_family_g_task is not None
    assert generation.validate_family_g_micro_task is not None


def _mutate_core(core, mutate):
    from mub.vnext.generation.core import SemanticCore

    payload = core.model_dump(mode="python")
    mutate(payload)
    return SemanticCore.model_validate(payload)


def test_family_g_generic_core_validation_is_reusable_but_micro_validation_is_exact():
    generate, validate_core, _, validate_micro_core, _, _ = _api()
    source = next(
        core
        for core in generate(_config())
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    )

    def make_valid_noncanonical(payload):
        payload["events"][1]["value"] += 10
        current = [event["value"] for event in payload["events"][1::2]]
        payload["expected_answer"] = current[0] - sum(current[1:])

    reusable = _mutate_core(source, make_valid_noncanonical)
    validate_core(reusable)
    with pytest.raises(ValueError, match="canonical"):
        validate_micro_core(reusable)

    wrong_answer = _mutate_core(
        source,
        lambda payload: payload.__setitem__(
            "expected_answer", payload["expected_answer"] + 1
        ),
    )
    with pytest.raises(ValueError, match="answer|derivation"):
        validate_core(wrong_answer)


def test_family_g_generic_task_validator_accepts_valid_nonmicro_projection_only():
    from mub.vnext.generation.core import GenerationContext
    from mub.vnext.generation.core_render_v3 import render_core_v3

    generate, validate_core, validate_task, _, validate_micro_task, _ = _api()
    source = next(
        core
        for core in generate(_config())
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    )

    def make_valid_noncanonical(payload):
        payload["events"][1]["value"] += 10
        current = [event["value"] for event in payload["events"][1::2]]
        payload["expected_answer"] = current[0] - sum(current[1:])

    reusable = _mutate_core(source, make_valid_noncanonical)
    validate_core(reusable)
    task = render_core_v3(
        reusable,
        split=Split.TEST,
        surface_variant=0,
        context=GenerationContext(config=_config(), code_revision="5423ef7"),
    )
    validate_task(task)
    with pytest.raises(ValueError, match="evaluation_only|canonical"):
        validate_micro_task(task)


def test_family_g_generator_has_exact_selected_micro_pilot_marginals():
    generate, validate_core, _, validate_micro_core, _, _ = _api()
    cores = generate(_config())

    assert len(cores) == 24
    assert cores == generate(_config())
    assert {core.task_family for core in cores} == {
        TaskFamily.LONG_HORIZON_MEMORY_SYNTHESIS
    }
    assert Counter(core.stratification["synthesis_kind"] for core in cores) == Counter(
        {"update_sensitive_multi_hop": 12, "multi_object_current_consistency": 12}
    )

    multi_hop = [
        core
        for core in cores
        if core.stratification["synthesis_kind"] == "update_sensitive_multi_hop"
    ]
    consistency = [
        core
        for core in cores
        if core.stratification["synthesis_kind"]
        == "multi_object_current_consistency"
    ]
    assert Counter(core.stratification["hop_count"] for core in multi_hop) == Counter(
        {2: 4, 3: 4, 4: 4}
    )
    assert Counter(
        core.stratification["stale_sensitive_position"] for core in multi_hop
    ) == Counter({"early": 4, "middle": 4, "final": 4})
    assert Counter(core.stratification["object_count"] for core in consistency) == Counter(
        {3: 4, 5: 4, 8: 4}
    )
    assert Counter(core.stratification["answer_kind"] for core in consistency) == Counter(
        {"boolean_consistency": 6, "exact_inconsistent_object": 6}
    )
    for object_count in (3, 5, 8):
        stratum = [
            core
            for core in consistency
            if core.stratification["object_count"] == object_count
        ]
        assert Counter(core.stratification["answer_kind"] for core in stratum) == Counter(
            {"boolean_consistency": 2, "exact_inconsistent_object": 2}
        )

    by_group = defaultdict(list)
    for core in cores:
        validate_core(core)
        validate_micro_core(core)
        by_group[core.stratification["synthesis_kind"]].append(core)
    assert all(len({core.trajectory_id for core in group}) == 1 for group in by_group.values())


def test_family_g_compiler_renders_exactly_four_replayable_v3_surfaces_per_core():
    _, _, validate_task, _, validate_micro_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")

    assert compiled.profile_id == "family_g_selected_micro_v1"
    assert compiled == compile_micro(_config(), code_revision="5423ef7")
    assert len(compiled.cores) == 24
    assert len(compiled.tasks) == 96
    by_core = defaultdict(list)
    for task in compiled.tasks:
        validate_task(task)
        validate_micro_task(task)
        assert task.metadata.split is Split.EVALUATION_ONLY
        assert len(task.queries) == len(task.gold_evidence) == 1
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        assert query.query_type in {
            QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
            QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
        }
        assert query.synthesis.kind == query.query_type.value
        assert evidence.stale_alternative is not None
        assert evidence.answer != evidence.stale_alternative.answer
        replay = replay_task_v3(task)
        assert replay.issues == ()
        assert resolve_query_v3(query, replay, task.events).issues == ()
        evaluated = evaluate_evidence_v3(
            evidence,
            replay,
            evidence.stale_alternative,
            query,
            task.events,
        )
        assert evaluated.issues == ()
        assert evaluated.answer == evidence.answer
        assert evaluated.stale_alternative_answer == evidence.stale_alternative.answer
        by_core[task.metadata.split_key.semantic_core_id].append(task)

    assert set(by_core) == {core.core_id for core in compiled.cores}
    by_derivation_family = defaultdict(list)
    for task in compiled.tasks:
        by_derivation_family[task.queries[0].query_type].append(task)
    assert Counter(len(group) for group in by_derivation_family.values()) == Counter({48: 2})
    assert all(
        len({task.metadata.split_key.version_group_id for task in group}) == 1
        and {task.metadata.split for task in group} == {Split.EVALUATION_ONLY}
        for group in by_derivation_family.values()
    )
    for surfaces in by_core.values():
        assert len(surfaces) == 4
        assert {task.metadata.extra["surface_variant"] for task in surfaces} == {0, 1, 2, 3}
        assert len({task.semantic_hash for task in surfaces}) == 1
        assert len({task.task_id for task in surfaces}) == 4
        assert len({task.source.raw_hash for task in surfaces}) == 4


def _derivation_depth(item):
    depths = {}
    for step in item.derivation_steps:
        depths[step.step_id] = 1 + max(
            (depths[parent] for parent in step.input_step_ids),
            default=0,
        )
    return depths[item.final_derivation_step_id]


def _assert_connected_topological_derivation(item):
    steps = {step.step_id: step for step in item.derivation_steps}
    positions = {step.step_id: index for index, step in enumerate(item.derivation_steps)}
    reached = set()

    def visit(step_id):
        if step_id in reached:
            return
        reached.add(step_id)
        for parent in steps[step_id].input_step_ids:
            assert positions[parent] < positions[step_id]
            visit(parent)

    visit(item.final_derivation_step_id)
    assert reached == set(steps)


def test_family_g_corrupted_evidence_controls_reject_removed_replaced_fabricated_and_inconsistent_derivations():
    _, _, validate_task, _, _, compile_micro = _api()
    tasks = compile_micro(_config(), code_revision="5423ef7").tasks
    multi_hop = next(
        task
        for task in tasks
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
        and task.metadata.extra["surface_variant"] == 0
    )

    removed = multi_hop.model_dump(mode="json")
    removed["gold_evidence"][0]["derivation_steps"][-1]["input_step_ids"] = removed[
        "gold_evidence"
    ][0]["derivation_steps"][-1]["input_step_ids"][:1]
    with pytest.raises(ValueError, match="operand|derivation|subtract"):
        MemUpdateTaskV3.model_validate(removed)

    replaced = multi_hop.model_dump(mode="json")
    final_inputs = replaced["gold_evidence"][0]["derivation_steps"][-1][
        "input_step_ids"
    ]
    replaced["gold_evidence"][0]["derivation_steps"][-1]["input_step_ids"] = list(
        reversed(final_inputs)
    )
    replaced_task = MemUpdateTaskV3.model_validate(replaced)
    with pytest.raises(ValueError, match="answer|evidence replay"):
        validate_task(replaced_task)

    fabricated = multi_hop.model_dump(mode="json")
    primary = fabricated["gold_evidence"][0]
    unrelated = next(
        event["event_id"]
        for event in fabricated["events"]
        if event["event_id"] not in primary["supporting_event_ids"]
    )
    primary["supporting_event_ids"].append(unrelated)
    primary["derivation_steps"][0]["supporting_event_ids"].append(unrelated)
    with pytest.raises(ValueError, match="support|event|provenance"):
        MemUpdateTaskV3.model_validate(fabricated)

    consistency = next(
        task
        for task in tasks
        if task.queries[0].query_type
        is QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY
        and task.metadata.extra["stratification"]["answer_kind"]
        == "exact_inconsistent_object"
        and task.metadata.extra["surface_variant"] == 0
    )
    inconsistent = consistency.model_dump(mode="json")
    inconsistent["gold_evidence"][0]["answer"] += 1
    inconsistent_task = MemUpdateTaskV3.model_validate(inconsistent)
    with pytest.raises(ValueError, match="answer|evidence replay"):
        validate_task(inconsistent_task)

    tunneled = multi_hop.model_dump(mode="json")
    tunneled["queries"][0]["text"] = "CURRENT_STATE"
    tunneled_task = MemUpdateTaskV3.model_validate(tunneled)
    with pytest.raises(ValueError, match="visible|tunnel|intent"):
        validate_task(tunneled_task)


def _metric_value(score, path):
    layer, leaf = path.split(".", 1)
    return getattr(getattr(score, layer), leaf)


def _authenticated_score_context(task, run, caps):
    from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
    from mub.vnext.contracts.v3.score import ScorerConfigV3
    from mub.vnext.io import sha256_model
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task_artifact = {
        "path": "family-g-tasks.jsonl",
        "sha256": "b" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    run_artifact = {
        "path": "family-g-runs.jsonl",
        "sha256": "c" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    task_manifest = TaskManifestV3(
        data_release_id="family-g-diagnostic",
        split_policy_version="diagnostic-no-release-split",
        compiler_versions={"core_family_g_micro": "3"},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={"evaluation_only": 1},
        family_difficulty_counts={f"G.{task.difficulty.value}": 1},
        semantic_core_counts={task.metadata.split_key.semantic_core_id: 1},
        task_file_paths_and_hashes=(task_artifact,),
        task_record_hashes={task.task_id: sha256_model(task)},
        leakage_check_summary={},
        human_audit_artifacts=(),
        created_at="2026-08-06T00:00:00Z",
        code_revision="5423ef7",
    )
    task_manifest_hash = sha256_model(task_manifest)
    config = ScorerConfigV3()
    run_manifest = RunManifestV3(
        run_id=run.run_id,
        timestamp="2026-08-06T00:00:00Z",
        code_revision="5423ef7",
        dirty_state=False,
        task_manifest={
            "path": "task-manifest.json",
            "sha256": task_manifest_hash,
            "media_type": "application/json",
        },
        scorer_config=config,
        adapter_info=ORACLE_INFO,
        adapter_capabilities=caps,
        capability_verification_artifact={
            "path": "caps.json",
            "sha256": "d" * 64,
            "media_type": "application/json",
        },
        model_name=None,
        provider=None,
        model_revision=None,
        prompt_config={},
        decoding_config={},
        seed_information={},
        action_parser_version="1",
        answer_parser_version="1",
        memory_entry_extractor_version="1",
        object_value_extractor_config_hash="a" * 64,
        redaction_policy_version="1",
        environment_summary={},
        package_summary={},
        expected_task_count=1,
        completed_task_count=1,
        failed_task_count=0,
        not_supported_task_count=0,
        raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(run_artifact,),
        run_record_hashes={run.task_id: sha256_model(run)},
        score_artifacts=(),
        native_vs_extracted_field_summary={},
    )
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task,
        run=run,
        task_manifest=task_manifest,
        run_manifest=run_manifest,
        task_artifact=task_artifact,
        run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_hash,
        authenticated_run_manifest_sha256=sha256_model(run_manifest),
    )


def _oracle_run(task, *, stale_prediction=False):
    from mub.vnext.contracts.v3.runtime import (
        AnswerPredictionV3,
        MemoryEntryRecordV3,
        MemorySnapshotV3,
        ParsedManagerActionV3,
        ParserExtractorProvenanceV3,
        RetrievalTraceV3,
        TaskRunRecordV3,
    )

    replay = replay_task_v3(task)
    evidence = task.gold_evidence[0]
    query = task.queries[0]
    all_entries = tuple(
        MemoryEntryRecordV3(
            entry_id=f"{ledger_index}-v-{version.version_index}",
            content=str(version.value),
            object_key_candidate=version.object_key,
            value_candidate=version.value,
            version_index=version.version_index,
            source_event_ids=version.source_event_ids,
        )
        for ledger_index, ledger in enumerate(replay.ledgers)
        for version in ledger.versions
    )
    required_events = set(evidence.supporting_event_ids)
    retrieved = tuple(
        entry
        for entry in all_entries
        if required_events & set(entry.source_event_ids)
    )
    current_entries = tuple(
        entry
        for entry in all_entries
        if any(
            entry.object_key_candidate.canonical_id == version.object_key.canonical_id
            and entry.version_index == version.version_index
            for version in replay.current_state.values()
        )
    )
    answer = (
        evidence.stale_alternative.answer
        if stale_prediction
        else evidence.answer
    )
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id,
            event_id=action.event_id,
            operation=action.operation,
            observed_scope=action.scope,
            target_object_keys=action.target_object_keys,
            value=action.value,
            format_valid=True,
            execution_status="executed",
            fallback_used=False,
            raw_output="oracle",
        )
        for action in task.actions
    )
    prediction = AnswerPredictionV3(
        query_id=query.query_id,
        raw_output="oracle",
        parsed_answer=answer,
        cited_event_ids=evidence.supporting_event_ids,
        cited_object_keys=evidence.supporting_object_keys,
        cited_derivation_step_ids=tuple(
            step.step_id for step in evidence.derivation_steps
        ),
        format_valid=True,
    )
    return TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id=ORACLE_INFO.adapter_id,
        run_id=f"run-{task.task_id}-stale-{stale_prediction}",
        parsed_actions=parsed_actions,
        memory_snapshots=(
            MemorySnapshotV3(
                after_event_id=task.events[-1].event_id,
                entries=current_entries,
                state_by_object={
                    key: version.value for key, version in replay.current_state.items()
                },
                store_size=len(current_entries),
            ),
        ),
        retrieval_traces=(
            RetrievalTraceV3(
                query_id=query.query_id,
                retrieved_entries=retrieved,
                ranks=tuple(range(1, len(retrieved) + 1)),
                gold_in_context=True,
                stale_in_context=False,
                distractor_in_context=False,
            ),
        ),
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            object_value_extractor_config_hash="a" * 64,
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )


def test_family_g_reference_oracle_has_perfect_applicable_answer_evidence_retrieval_and_stale_metrics():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    *_, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    for task in compiled.tasks:
        run = _oracle_run(task)
        score = score_task_v3(
            task,
            run,
            _authenticated_score_context(task, run, FULL_CAPABILITIES),
        )
        assert score.action_scores.full_action_exact_match == 1.0
        assert score.state_scores.final_state_accuracy == 1.0
        assert score.answer_scores.exact_match == 1.0
        expected = {
            "synthesis_scores.evidence_precision": 1.0,
            "synthesis_scores.evidence_recall": 1.0,
            "synthesis_scores.evidence_f1": 1.0,
            "synthesis_scores.reasoning_support_accuracy": 1.0,
            "synthesis_scores.stale_propagation_rate": 0.0,
        }
        if task.queries[0].query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
            expected["synthesis_scores.multi_hop_accuracy"] = 1.0
            not_applicable = "synthesis_scores.multi_object_accuracy"
        else:
            expected["synthesis_scores.multi_object_accuracy"] = 1.0
            not_applicable = "synthesis_scores.multi_hop_accuracy"
        for path, value in expected.items():
            assert _metric_value(score, path) == value, (task.task_id, path)
        assert _metric_value(score, not_applicable) is None
        assert score.supported_metric_fields[not_applicable].reason is SupportReason.NOT_APPLICABLE


def test_family_g_withheld_capabilities_are_typed_nulls_and_stale_predictions_propagate():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    *_, compile_micro = _api()
    representatives = {}
    for task in compile_micro(_config(), code_revision="5423ef7").tasks:
        representatives.setdefault(task.queries[0].query_type, task)
    assert set(representatives) == {
        QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP,
        QueryTypeV3.MULTI_OBJECT_CURRENT_CONSISTENCY,
    }

    for query_type, task in representatives.items():
        run = _oracle_run(task)
        withheld = score_task_v3(
            task,
            run,
            _authenticated_score_context(task, run, AdapterCapabilitiesV3()),
        )
        applicable_accuracy = (
            "synthesis_scores.multi_hop_accuracy"
            if query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP
            else "synthesis_scores.multi_object_accuracy"
        )
        for path in SYNTHESIS_PATHS:
            assert _metric_value(withheld, path) is None
            expected_reason = (
                SupportReason.NOT_APPLICABLE
                if path
                in {
                    "synthesis_scores.multi_hop_accuracy",
                    "synthesis_scores.multi_object_accuracy",
                }
                and path != applicable_accuracy
                else SupportReason.NOT_SUPPORTED
            )
            assert withheld.supported_metric_fields[path].reason is expected_reason

        stale_run = _oracle_run(task, stale_prediction=True)
        stale_score = score_task_v3(
            task,
            stale_run,
            _authenticated_score_context(task, stale_run, FULL_CAPABILITIES),
        )
        assert _metric_value(stale_score, applicable_accuracy) == 0.0
        assert stale_score.synthesis_scores.stale_propagation_rate == 1.0
        assert "stale_propagation" in stale_score.failure_flags


def test_family_g_derivations_are_ordered_connected_and_stale_sensitive_at_declared_hop():
    *_, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="5423ef7")
    one_surface = {
        task.metadata.split_key.semantic_core_id: task
        for task in compiled.tasks
        if task.metadata.extra["surface_variant"] == 0
    }
    core_by_id = {core.core_id: core for core in compiled.cores}
    boolean_answers = []
    exact_answers = []

    for core_id, task in one_surface.items():
        core = core_by_id[core_id]
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        stale = evidence.stale_alternative
        for item in (evidence, stale):
            _assert_connected_topological_derivation(item)
            read_events = {
                event_id
                for step in item.derivation_steps
                if step.operation in {"read", "read_current", "read_version"}
                for event_id in step.supporting_event_ids
            }
            assert read_events == set(item.supporting_event_ids)
        assert all(key.canonical_id in query.text for key in task.target_objects)
        assert "CURRENT_STATE" not in query.text

        if query.query_type is QueryTypeV3.UPDATE_SENSITIVE_MULTI_HOP:
            hop_count = core.stratification["hop_count"]
            stale_index = core.stratification["stale_operand_index"]
            assert _derivation_depth(evidence) == hop_count
            assert _derivation_depth(stale) == hop_count
            primary_reads = evidence.derivation_steps[:hop_count]
            stale_reads = stale.derivation_steps[:hop_count]
            assert [step.operation for step in primary_reads] == ["read_current"] * hop_count
            assert [step.operation for step in stale_reads].count("read_version") == 1
            assert stale_reads[stale_index].operation == "read_version"
            assert all(
                step.input_step_ids[1] == primary_reads[index].step_id
                for index, step in enumerate(
                    evidence.derivation_steps[hop_count:],
                    start=1,
                )
            )
        elif core.stratification["answer_kind"] == "boolean_consistency":
            boolean_answers.append(evidence.answer)
            assert type(evidence.answer) is bool
            assert stale.answer is (not evidence.answer)
        else:
            exact_answers.append(evidence.answer)
            assert type(evidence.answer) is int and evidence.answer > 0
            assert type(stale.answer) is int and stale.answer > 0
            assert stale.answer != evidence.answer

    assert Counter(boolean_answers) == Counter({True: 3, False: 3})
    assert len(exact_answers) == 6
