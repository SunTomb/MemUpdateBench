from collections import Counter
from pathlib import Path

import pytest

from mub.vnext.contracts.enums import ActionScope, AnswerSchema, EventRole, Operation, Split, TaskFamily
from mub.vnext.generation.core import SemanticCore
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"
LIFECYCLE_CELLS = (
    "explicit_object_or_attribute_deletion",
    "entity_wide_deletion",
    "namespace_privacy_wipe",
    "correction_versus_deletion_hard_negative",
    "logical_ttl_expiry",
    "post_delete_similar_retrieval",
    "delete_then_relearn",
    "scoped_delete_protected_collateral",
)


def _api():
    from mub.vnext.generation.family_e import (
        compile_family_e_micro_pilot,
        generate_core_family_e_cores,
        validate_family_e_core,
    )

    return generate_core_family_e_cores, validate_family_e_core, compile_family_e_micro_pilot


def _config():
    return load_core_config(CORE_CONFIG)


def _mutate_core(core: SemanticCore, mutate):
    payload = core.model_dump(mode="python")
    mutate(payload)
    return SemanticCore.model_validate(payload)


def test_family_e_generator_balances_exactly_three_cores_per_approved_cell():
    generate, validate, _ = _api()
    config = _config()

    cores = generate(config)

    assert len(cores) == 24
    assert {core.task_family for core in cores} == {TaskFamily.DELETION_FORGETTING}
    assert Counter(core.stratification["lifecycle_cell"] for core in cores) == {
        cell: 3 for cell in LIFECYCLE_CELLS
    }
    assert len({core.core_id for core in cores}) == 24
    assert cores == generate(config)
    for core in cores:
        validate(core)


def test_family_e_core_semantics_preserve_scope_ttl_relearn_and_privacy_invariants():
    generate, _, _ = _api()
    cores = generate(_config())
    by_cell = {cell: [core for core in cores if core.stratification["lifecycle_cell"] == cell] for cell in LIFECYCLE_CELLS}

    correction = by_cell["correction_versus_deletion_hard_negative"][0]
    assert any(event.operation is Operation.UPDATE for event in correction.events)
    assert not any(event.operation is Operation.DELETE for event in correction.events)

    ttl = by_cell["logical_ttl_expiry"][0]
    ttl_delete = next(event for event in ttl.events if event.operation is Operation.DELETE)
    assert ttl_delete.metadata["action_scope"] == ActionScope.TTL.value
    assert ttl_delete.metadata["effective_at"] == ttl.stratification["ttl_expiry_at"]
    assert ttl.stratification["query_logical_time"] == ttl.stratification["ttl_expiry_at"]
    assert "wall_clock" not in ttl_delete.metadata

    relearn = by_cell["delete_then_relearn"][0]
    writes = [event for event in relearn.events if event.operation in {Operation.ADD, Operation.UPDATE}]
    assert writes[0].value != writes[-1].value
    assert relearn.stratification["forgotten_value"] == writes[0].value
    assert relearn.expected_answer == writes[-1].value

    privacy = by_cell["namespace_privacy_wipe"][0]
    assert privacy.profile["source_naturalness"] == "synthetic"
    assert all(
        str(event.value).startswith("synthetic_private_")
        for event in privacy.events
        if event.value is not None
    )


def test_family_e_micro_compiler_emits_exactly_four_equivalent_surfaces_per_core():
    _, _, compile_micro = _api()
    snapshot = compile_micro(_config(), code_revision="4bbc446")

    assert snapshot.profile_id == "family_e_diagnostic_micro_v1"
    assert len(snapshot.cores) == 24
    assert len(snapshot.tasks) == 96
    assert Counter(task.metadata.split for task in snapshot.tasks) == {Split.TEST: 96}
    assert all(task.schema_version == "3.0.0" for task in snapshot.tasks)
    assert all(task.task_family == TaskFamily.DELETION_FORGETTING.value for task in snapshot.tasks)

    tasks_by_core = {}
    for task in snapshot.tasks:
        tasks_by_core.setdefault(task.metadata.split_key.semantic_core_id, []).append(task)
    assert set(tasks_by_core) == {core.core_id for core in snapshot.cores}
    for tasks in tasks_by_core.values():
        assert len(tasks) == 4
        assert len({task.task_id for task in tasks}) == 4
        assert len({task.source.raw_hash for task in tasks}) == 4
        assert len({task.semantic_hash for task in tasks}) == 1
        assert {task.metadata.extra["surface_variant"] for task in tasks} == {0, 1, 2, 3}
        for task in tasks:
            assert replay_task_v3(task).issues == ()


def test_family_e_visible_queries_match_typed_v3_current_state_answers():
    _, _, compile_micro = _api()
    snapshot = compile_micro(_config(), code_revision="4bbc446")

    for task in snapshot.tasks:
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        replay = replay_task_v3(task)
        resolution = resolve_query_v3(query, replay, task.events)
        assert query.answer_schema is AnswerSchema.LIST
        assert "Return a list aligned to the target order, using null for a missing current value." in query.text
        assert len(evidence.answer) == len(query.target_object_keys)
        assert resolution.issues == ()
        assert resolution.answer == evidence.answer


def test_family_e_profiles_report_actual_peak_active_object_counts():
    generate, _, _ = _api()
    cores = generate(_config())
    expected = {
        "explicit_object_or_attribute_deletion": {1, 2},
        "entity_wide_deletion": {2},
        "namespace_privacy_wipe": {3},
        "correction_versus_deletion_hard_negative": {1},
        "logical_ttl_expiry": {1},
        "post_delete_similar_retrieval": {2},
        "delete_then_relearn": {1},
        "scoped_delete_protected_collateral": {3},
    }
    for cell, counts in expected.items():
        assert {
            core.profile["active_object_count"]
            for core in cores
            if core.stratification["lifecycle_cell"] == cell
        } == counts


def test_family_e_scoped_gold_actions_enumerate_exact_keys_and_protect_collateral():
    _, _, compile_micro = _api()
    snapshot = compile_micro(_config(), code_revision="4bbc446")

    scoped_cells = {
        "entity_wide_deletion": ActionScope.ENTITY,
        "namespace_privacy_wipe": ActionScope.NAMESPACE,
        "scoped_delete_protected_collateral": ActionScope.ATTRIBUTE,
    }
    for cell, scope in scoped_cells.items():
        tasks = [task for task in snapshot.tasks if task.metadata.extra["stratification"]["lifecycle_cell"] == cell]
        assert len(tasks) == 12
        for task in tasks:
            delete = next(action for action in task.actions if action.operation is Operation.DELETE)
            assert delete.scope is scope
            assert len(delete.target_object_keys) >= 2
            declared = {
                key.canonical_id for key in task.target_objects
                if key.canonical_id not in {target.canonical_id for target in delete.target_object_keys}
            }
            protected = set(task.metadata.extra["family_e"]["protected_collateral_ids"])
            assert protected <= declared
            assert protected.isdisjoint(target.canonical_id for target in delete.target_object_keys)


def test_family_e_validator_fails_closed_on_required_corrupted_controls():
    generate, validate, _ = _api()
    cores = generate(_config())
    by_cell = {cell: next(core for core in cores if core.stratification["lifecycle_cell"] == cell) for cell in LIFECYCLE_CELLS}

    def omit_scope_key(payload):
        delete = next(event for event in payload["events"] if event["operation"] == "DELETE")
        delete["object_keys"] = delete["object_keys"][:-1]

    with pytest.raises(ValueError, match="enumerate exact scope targets"):
        validate(_mutate_core(by_cell["entity_wide_deletion"], omit_scope_key))

    def delete_collateral(payload):
        protected = next(
            event["object_keys"][0]
            for event in payload["events"]
            if event["metadata"].get("protected_collateral") is True
        )
        delete = next(event for event in payload["events"] if event["operation"] == "DELETE")
        delete["object_keys"].append(protected)

    with pytest.raises(ValueError, match="protected collateral"):
        validate(_mutate_core(by_cell["scoped_delete_protected_collateral"], delete_collateral))

    def shift_ttl_boundary(payload):
        delete = next(event for event in payload["events"] if event["operation"] == "DELETE")
        delete["metadata"]["effective_at"] = "00000021"

    with pytest.raises(ValueError, match="TTL boundary"):
        validate(_mutate_core(by_cell["logical_ttl_expiry"], shift_ttl_boundary))

    def shift_ttl_probe(payload):
        probe = next(event for event in payload["events"] if event["metadata"].get("lifecycle") == "ttl_boundary_probe")
        probe["metadata"]["logical_time"] = "00000019"
        probe["metadata"]["effective_at"] = "00000019"

    with pytest.raises(ValueError, match="TTL boundary"):
        validate(_mutate_core(by_cell["logical_ttl_expiry"], shift_ttl_probe))

    def retain_forgotten_value(payload):
        forgotten = payload["stratification"]["forgotten_value"]
        payload["events"].append({
            "operation": Operation.UPDATE,
            "object_keys": [payload["query_targets"][0]],
            "value": forgotten,
            "role": EventRole.LATEST_GOLD,
            "metadata": {"lifecycle": "corrupt_forgotten_retention", "action_scope": "object", "logical_time": "00000099", "effective_at": "00000099"},
        })
        payload["expected_answer"] = forgotten

    with pytest.raises(ValueError, match="forgotten value"):
        validate(_mutate_core(by_cell["delete_then_relearn"], retain_forgotten_value))

    def fail_relearn(payload):
        relearn = [event for event in payload["events"] if event["metadata"].get("lifecycle") == "relearn"][0]
        relearn["value"] = payload["stratification"]["forgotten_value"]
        payload["expected_answer"] = payload["stratification"]["forgotten_value"]

    with pytest.raises(ValueError, match="relearn"):
        validate(_mutate_core(by_cell["delete_then_relearn"], fail_relearn))


def test_family_e_micro_compiler_rejects_count_and_surface_corruption(monkeypatch):
    from mub.vnext.generation import family_e

    generate, _, compile_micro = _api()
    cores = generate(_config())
    monkeypatch.setattr(family_e, "generate_core_family_e_cores", lambda config: cores[:-1])
    with pytest.raises(ValueError, match="exactly 24"):
        compile_micro(_config(), code_revision="4bbc446")

    monkeypatch.setattr(family_e, "generate_core_family_e_cores", lambda config: cores)

    def omit_scope_and_query(payload):
        delete = next(event for event in payload["events"] if event["operation"] == "DELETE")
        delete["object_keys"] = delete["object_keys"][:-1]
        payload["query_targets"] = payload["query_targets"][:-1]

    entity_index = next(
        index
        for index, core in enumerate(cores)
        if core.stratification["lifecycle_cell"] == "entity_wide_deletion"
    )
    corrupted = list(cores)
    corrupted[entity_index] = _mutate_core(cores[entity_index], omit_scope_and_query)
    monkeypatch.setattr(family_e, "generate_core_family_e_cores", lambda config: corrupted)
    with pytest.raises(ValueError, match="enumerate exact scope targets"):
        compile_micro(_config(), code_revision="4bbc446")

    monkeypatch.setattr(family_e, "generate_core_family_e_cores", lambda config: cores)
    original = family_e.render_core_v3
    monkeypatch.setattr(
        family_e,
        "render_core_v3",
        lambda core, **kwargs: original(core, **{**kwargs, "surface_variant": 0}),
    )
    with pytest.raises(ValueError, match="four surface variants"):
        compile_micro(_config(), code_revision="4bbc446")


def _authenticated_score_context(task, run, info, caps):
    from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
    from mub.vnext.contracts.v3.score import ScorerConfigV3
    from mub.vnext.io import sha256_model
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task_artifact = {"path": "family-e-tasks.jsonl", "sha256": "b" * 64, "media_type": "application/jsonl", "record_count": 1}
    run_artifact = {"path": "family-e-runs.jsonl", "sha256": "c" * 64, "media_type": "application/jsonl", "record_count": 1}
    task_manifest = TaskManifestV3(
        data_release_id="family-e-diagnostic", split_policy_version="diagnostic-no-release-split",
        compiler_versions={"core_family_e_micro": "3"}, source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(), split_counts={"test": 1},
        family_difficulty_counts={f"E.{task.difficulty.value}": 1},
        semantic_core_counts={task.metadata.split_key.semantic_core_id: 1},
        task_file_paths_and_hashes=(task_artifact,),
        task_record_hashes={task.task_id: sha256_model(task)}, leakage_check_summary={},
        human_audit_artifacts=(), created_at="2026-08-04T00:00:00Z", code_revision="4bbc446",
    )
    task_manifest_hash = sha256_model(task_manifest)
    config = ScorerConfigV3()
    run_manifest = RunManifestV3(
        run_id=run.run_id, timestamp="2026-08-04T00:00:00Z", code_revision="4bbc446",
        dirty_state=False,
        task_manifest={"path": "task-manifest.json", "sha256": task_manifest_hash, "media_type": "application/json"},
        scorer_config=config, adapter_info=info, adapter_capabilities=caps,
        capability_verification_artifact={"path": "caps.json", "sha256": "d" * 64, "media_type": "application/json"},
        model_name=None, provider=None, model_revision=None, prompt_config={}, decoding_config={},
        seed_information={}, action_parser_version="1", answer_parser_version="1",
        memory_entry_extractor_version="1", object_value_extractor_config_hash="a" * 64,
        redaction_policy_version="1", environment_summary={}, package_summary={},
        expected_task_count=1, completed_task_count=1, failed_task_count=0,
        not_supported_task_count=0, raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(), normalized_runtime_artifacts=(run_artifact,),
        run_record_hashes={run.task_id: sha256_model(run)}, score_artifacts=(),
        native_vs_extracted_field_summary={},
    )
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task, run=run, task_manifest=task_manifest, run_manifest=run_manifest,
        task_artifact=task_artifact, run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_hash,
        authenticated_run_manifest_sha256=sha256_model(run_manifest),
    )


def _oracle_run(task):
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
    entries = tuple(
        MemoryEntryRecordV3(
            entry_id=f"current-{index}", content=str(version.value),
            object_key_candidate=version.object_key, value_candidate=version.value,
            version_index=version.version_index, source_event_ids=version.source_event_ids,
        )
        for index, version in enumerate(replay.current_state.values())
    )
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id, event_id=action.event_id,
            operation=action.operation, observed_scope=action.scope,
            target_object_keys=action.target_object_keys, value=action.value,
            format_valid=True, execution_status="executed", fallback_used=False,
            raw_output="oracle",
        )
        for action in task.actions
    )
    traces = []
    predictions = []
    for query, evidence in zip(task.queries, task.gold_evidence):
        target_ids = {key.canonical_id for key in query.target_object_keys}
        retrieved = tuple(entry for entry in entries if entry.object_key_candidate.canonical_id in target_ids)
        traces.append(RetrievalTraceV3(
            query_id=query.query_id, retrieved_entries=retrieved,
            ranks=tuple(range(1, len(retrieved) + 1)), gold_in_context=True,
            stale_in_context=False, distractor_in_context=False,
        ))
        predictions.append(AnswerPredictionV3(
            query_id=query.query_id, raw_output="oracle", parsed_answer=evidence.answer,
            cited_event_ids=evidence.supporting_event_ids,
            cited_object_keys=evidence.supporting_object_keys,
            cited_derivation_step_ids=(evidence.final_derivation_step_id,),
            format_valid=True,
        ))
    state = {key: version.value for key, version in replay.current_state.items()}
    return TaskRunRecordV3(
        task_id=task.task_id, adapter_id="family-e-oracle", run_id=f"run-{task.task_id}",
        parsed_actions=parsed_actions,
        memory_snapshots=(MemorySnapshotV3(
            after_event_id=task.events[-1].event_id, entries=entries,
            state_by_object=state, store_size=len(entries),
        ),),
        retrieval_traces=tuple(traces), answer_predictions=tuple(predictions),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", object_value_extractor_config_hash="a" * 64,
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )


def test_family_e_corrupted_oracle_detects_forgotten_value_leakage():
    from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    _, _, compile_micro = _api()
    snapshot = compile_micro(_config(), code_revision="4bbc446")
    task = next(
        task
        for task in snapshot.tasks
        if task.metadata.extra["stratification"]["lifecycle_cell"] == "delete_then_relearn"
        and task.metadata.extra["surface_variant"] == 0
    )
    forgotten = task.metadata.extra["stratification"]["forgotten_value"]
    run = _oracle_run(task)
    leaked = run.answer_predictions[0].model_copy(update={"parsed_answer": forgotten})
    corrupted = run.model_copy(
        update={"run_id": f"leak-{task.task_id}", "answer_predictions": (leaked,)}
    )
    caps = AdapterCapabilitiesV3(
        supports_native_answer=True,
        supports_delete=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
        exports_retrieval_ids=True,
        exports_action_trace=True,
    )
    info = AdapterInfoV3(
        adapter_id="family-e-oracle", adapter_version="1", system_name="reference",
        system_version="1", configuration_hash="a" * 64,
    )
    score = score_task_v3(
        task,
        corrupted,
        _authenticated_score_context(task, corrupted, info, caps),
    )
    assert score.deletion_scores.forgotten_value_leakage_rate == 1.0
    assert "forgotten_value_exposed" in score.failure_flags


def test_family_e_reference_oracle_is_perfect_on_every_applicable_principal_metric():
    from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
    from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    _, _, compile_micro = _api()
    snapshot = compile_micro(_config(), code_revision="4bbc446")
    caps = AdapterCapabilitiesV3(
        supports_isolated_reset=True, supports_event_ingest=True, supports_add=True,
        supports_update=True, supports_noop=True, supports_delete=True, supports_ttl=True,
        supports_native_answer=True, supports_scoped_delete=True,
        supports_historical_query=True, supports_multi_object_query=True,
        exports_version_history=True, exports_entries=True, exports_raw_state=True,
        exports_source_event_ids=True, exports_timestamps_or_order=True,
        exports_object_keys=True, exports_values=True, exports_retrieval_ids=True,
        exports_retrieval_scores=True, exports_action_trace=True,
        exports_evidence_linkage=True,
    )
    info = AdapterInfoV3(
        adapter_id="family-e-oracle", adapter_version="1", system_name="reference",
        system_version="1", configuration_hash="a" * 64,
    )
    observed_paths = set()
    for task in snapshot.tasks:
        run = _oracle_run(task)
        score = score_task_v3(task, run, _authenticated_score_context(task, run, info, caps))
        for path, descriptor in CORE_METRIC_REGISTRY_V3.items():
            if not descriptor.principal:
                continue
            layer, leaf = path.split(".", 1)
            value = getattr(getattr(score, layer), leaf)
            if value is None:
                continue
            observed_paths.add(path)
            expected = 0.0 if descriptor.direction == "lower" else 1.0
            assert value == expected, (task.task_id, path, value)
    assert {
        "deletion_scores.deletion_accuracy",
        "deletion_scores.delete_scope_accuracy",
        "deletion_scores.collateral_damage_rate",
        "deletion_scores.ttl_compliance_rate",
        "deletion_scores.relearn_accuracy",
        "deletion_scores.forgotten_exposure_rate",
        "deletion_scores.forgotten_value_leakage_rate",
    } <= observed_paths
