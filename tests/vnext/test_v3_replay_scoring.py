from copy import deepcopy

import hashlib
import json
import pytest

from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
from mub.vnext.contracts.v3.runtime import ParserExtractorProvenanceV3, TaskRunRecordV3
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS, ScorerConfigV3
from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3

H = "a" * 64


def payload():
    key = {"object_type": "slot", "namespace": "n", "entity": "e", "attribute": "a", "subkey": None}
    events = [
        {"event_id": f"e{i}", "sequence_index": i, "raw_text": "x", "normalized_text": "x", "role": "neutral", "gold_action_ids": [f"a{i}"]}
        for i in range(4)
    ]
    actions = [
        {"action_id": "a0", "event_id": "e0", "operation": "ADD", "scope": "object", "target_object_keys": [key], "value": "v0", "effective_at": "000"},
        {"action_id": "a1", "event_id": "e1", "operation": "UPDATE", "scope": "object", "target_object_keys": [key], "value": "v1", "effective_at": "001"},
        {"action_id": "a2", "event_id": "e2", "operation": "DELETE", "scope": "object", "target_object_keys": [key], "effective_at": "002"},
        {"action_id": "a3", "event_id": "e3", "operation": "ADD", "scope": "object", "target_object_keys": [key], "value": "v2", "effective_at": "003"},
    ]
    entries = [
        {"version_index": 0, "status": "present", "value": "v0", "valid_from_event_id": "e0", "valid_until_event_id": "e1", "logical_time": "000", "source_event_ids": ["e0"]},
        {"version_index": 1, "status": "present", "value": "v1", "valid_from_event_id": "e1", "valid_until_event_id": "e2", "logical_time": "001", "source_event_ids": ["e1"]},
        {"version_index": 2, "status": "tombstone", "valid_from_event_id": "e2", "valid_until_event_id": "e3", "logical_time": "002", "source_event_ids": ["e2"]},
        {"version_index": 3, "status": "present", "value": "v2", "valid_from_event_id": "e3", "logical_time": "003", "source_event_ids": ["e3"]},
    ]
    query = {"query_id": "q", "query_type": "ordered_history", "text": "?", "selector": {"kind": "ordered_history"}, "target_object_keys": [key], "answer_schema": "list", "evaluation_mode": "state_direct"}
    return {
        "task_id": "t", "task_family": "F", "difficulty": "easy",
        "source": {"source_id": "s", "source_type": "synthetic", "source_uri": None, "license_or_privacy": "synthetic", "raw_hash": H, "normalized_hash": H, "normalization_version": "n1", "generator": {"generator_name": "g", "seed": 1, "config_sha256": H, "code_revision": "r", "compiler_version": "3"}},
        "events": events, "target_objects": [key], "actions": actions, "queries": [query],
        "version_history": [{"object_key": key, "entries": entries}],
        "gold_evidence": [{"query_id": "q", "answer": ["v0", "v1", None, "v2"], "supporting_object_keys": [key], "supporting_event_ids": ["e0", "e1", "e2", "e3"], "derivation_steps": [{"step_id": "read", "operation": "ordered_history", "supporting_object_keys": [key], "supporting_event_ids": ["e0", "e1", "e2", "e3"]}], "final_derivation_step_id": "read"}],
        "metadata": {"split": "test", "split_key": {"semantic_core_id": "c", "source_group_id": "s", "trajectory_id": "t", "split_policy_version": "3"}, "profile_name": "easy", "generation_config_hash": H, "compiler_version": "3"},
    }


def authenticated_manifests(task, run, info, caps, config):
    task_artifact = {"path": "tasks.jsonl", "sha256": "b" * 64, "media_type": "application/jsonl", "record_count": 1}
    run_artifact = {"path": "runs.jsonl", "sha256": "c" * 64, "media_type": "application/jsonl", "record_count": 1}
    task_manifest = TaskManifestV3(
        data_release_id="release", split_policy_version="3", compiler_versions={"core": "3"},
        source_manifest_paths_and_hashes=(), generation_configs_and_hashes=(),
        split_counts={"test": 1}, family_difficulty_counts={"F.easy": 1}, semantic_core_counts={"c": 1},
        task_file_paths_and_hashes=(task_artifact,), task_record_hashes={task.task_id: hashlib.sha256(json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        leakage_check_summary={}, human_audit_artifacts=(),
        created_at="2026-08-02T00:00:00Z", code_revision="r",
    )
    task_manifest_sha256 = hashlib.sha256(json.dumps(task_manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_manifest = RunManifestV3(
        run_id=run.run_id, timestamp="2026-08-02T00:00:00Z", code_revision="r", dirty_state=False,
        task_manifest={"path": "task-manifest.json", "sha256": task_manifest_sha256, "media_type": "application/json"},
        scorer_config=config, adapter_info=info, adapter_capabilities=caps,
        capability_verification_artifact={"path": "capabilities.json", "sha256": "e" * 64, "media_type": "application/json"},
        model_name=None, provider=None, model_revision=None, prompt_config={}, decoding_config={}, seed_information={},
        action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1",
        object_value_extractor_config_hash=H, redaction_policy_version="1", environment_summary={}, package_summary={},
        expected_task_count=1, completed_task_count=int(run.completion_status.value == "completed"),
        failed_task_count=int(run.completion_status.value in {"failed", "partial"}),
        not_supported_task_count=int(run.completion_status.value == "not_supported"),
        raw_provider_response_artifacts=(), raw_adapter_state_artifacts=(), normalized_runtime_artifacts=(run_artifact,),
        run_record_hashes={run.task_id: hashlib.sha256(json.dumps(run.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        score_artifacts=(), native_vs_extracted_field_summary={},
    )
    run_manifest_sha256 = hashlib.sha256(json.dumps(run_manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return task_manifest, run_manifest, task_artifact, run_artifact, task_manifest_sha256, run_manifest_sha256


def authenticated_context(task, run, info, caps, config=None):
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3
    config = config or ScorerConfigV3()
    task_manifest, run_manifest, task_artifact, run_artifact, task_manifest_sha256, run_manifest_sha256 = authenticated_manifests(task, run, info, caps, config)
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task, run=run, task_manifest=task_manifest, run_manifest=run_manifest,
        task_artifact=task_artifact, run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_sha256,
        authenticated_run_manifest_sha256=run_manifest_sha256,
    )




def current_structured_payload(value, answer_schema):
    changed = payload()
    changed["queries"][0] = {"query_id": "q", "query_type": "current", "text": "?", "selector": {"kind": "current"}, "target_object_keys": changed["target_objects"], "answer_schema": answer_schema, "evaluation_mode": "state_direct"}
    changed["actions"][3]["value"] = value
    changed["version_history"][0]["entries"][3]["value"] = value
    changed["gold_evidence"][0] = {"query_id": "q", "answer": value, "supporting_object_keys": changed["target_objects"], "supporting_event_ids": ["e3"], "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": changed["target_objects"], "supporting_event_ids": ["e3"]}], "final_derivation_step_id": "read"}
    return changed


def test_authenticated_context_uses_canonical_unicode_model_hashes():
    from mub.vnext.io import sha256_model
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task = MemUpdateTaskV3.model_validate(payload())
    run = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id="unicode", parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="failed", exceptions=({"type": "boom"},))
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="你好", system_version="1", configuration_hash=H)
    manifests = authenticated_manifests(task, run, info, AdapterCapabilitiesV3(), ScorerConfigV3())
    task_manifest, run_manifest, task_artifact, run_artifact = manifests[:4]
    context = VerifiedScoringContextV3.from_authenticated_manifests(task=task, run=run, task_manifest=task_manifest, run_manifest=run_manifest, task_artifact=task_artifact, run_artifact=run_artifact, authenticated_task_manifest_sha256=sha256_model(task_manifest), authenticated_run_manifest_sha256=sha256_model(run_manifest))
    assert context.adapter_info.system_name == "你好"


@pytest.mark.parametrize(("value", "answer_schema"), [(["nested", 1], "list"), ({"x": [1]}, "object")])
def test_replay_accepts_already_frozen_structured_values(value, answer_schema):
    task = MemUpdateTaskV3.model_validate(current_structured_payload(value, answer_schema))
    replay = replay_task_v3(task)
    assert replay.issues == ()
    assert replay.current_state[task.target_objects[0].canonical_id].value == task.gold_evidence[0].answer


@pytest.mark.parametrize("corruption", [True, 1.0])
def test_nested_leaf_types_are_exact_in_replay_and_entry_matching(corruption):
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.scorer_v3 import _entry_matches_version

    good = current_structured_payload({"x": [1]}, "object")
    task = MemUpdateTaskV3.model_validate(good)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    entry = MemoryEntryRecordV3(entry_id="nested-corrupt", content="bad", object_key_candidate=task.target_objects[0], value_candidate={"x": [corruption]}, version_index=3, source_event_ids=("e3",))
    assert not _entry_matches_version(entry, replay.ledgers[0].versions[3])

    forged = deepcopy(good)
    forged["version_history"][0]["entries"][3]["value"] = {"x": [corruption]}
    forged_replay = replay_task_v3(MemUpdateTaskV3.model_validate(forged))
    assert {issue.code for issue in forged_replay.issues} == {"replay_version_history_mismatch"}


def test_event_anchor_does_not_look_ahead_at_equal_timestamp():
    changed = payload()
    changed["events"][0]["timestamp"] = "001"
    changed["events"][1]["timestamp"] = "001"
    changed["queries"][0] = {"query_id": "q", "query_type": "point_in_time", "text": "?", "selector": {"kind": "event_anchor", "event_id": "e0"}, "target_object_keys": changed["target_objects"], "answer_schema": "string", "evaluation_mode": "state_direct"}
    changed["gold_evidence"][0] = {"query_id": "q", "answer": "v0", "supporting_object_keys": changed["target_objects"], "supporting_event_ids": ["e0"], "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": changed["target_objects"], "supporting_event_ids": ["e0"]}], "final_derivation_step_id": "read"}
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    resolution = resolve_query_v3(task.queries[0], replay, task.events)
    assert resolution.answer == "v0"
    assert resolution.selected_event_ids == ("e0",)


def test_event_anchor_accepts_globally_known_noop_inside_version_interval():
    changed = payload()
    changed["events"].insert(1, {
        "event_id": "e-noop", "sequence_index": 1, "raw_text": "ordinary",
        "normalized_text": "ordinary", "role": "neutral", "gold_action_ids": ["a-noop"],
    })
    for index, event in enumerate(changed["events"]):
        event["sequence_index"] = index
    changed["actions"].insert(1, {
        "action_id": "a-noop", "event_id": "e-noop", "operation": "NOOP",
    })
    changed["queries"][0] = {
        "query_id": "q", "query_type": "point_in_time", "text": "?",
        "selector": {"kind": "event_anchor", "event_id": "e-noop"},
        "target_object_keys": changed["target_objects"], "answer_schema": "string",
        "evaluation_mode": "state_direct",
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": "v0", "supporting_object_keys": changed["target_objects"],
        "supporting_event_ids": ["e0"],
        "derivation_steps": [{
            "step_id": "read", "operation": "read",
            "supporting_object_keys": changed["target_objects"], "supporting_event_ids": ["e0"],
        }],
        "final_derivation_step_id": "read",
    }

    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    resolution = resolve_query_v3(task.queries[0], replay, task.events)
    assert replay.issues == ()
    assert resolution.issues == ()
    assert resolution.answer == "v0"
    assert tuple(version.version_index for version in resolution.selected_versions) == (0,)

    unknown = deepcopy(changed)
    unknown["queries"][0]["selector"]["event_id"] = "not-an-event"
    with pytest.raises(ValueError, match="unknown event anchor"):
        MemUpdateTaskV3.model_validate(unknown)


def test_metric_family_aliases_include_canonical_letters_registry_wide():
    from mub.vnext.scoring.registry_v3 import metric_applies_v3

    aliases = {"deletion_forgetting": "E", "current_historical_query": "F", "long_horizon_memory_synthesis": "G"}
    for descriptor in CORE_METRIC_REGISTRY_V3.values():
        for alias, canonical in aliases.items():
            if alias in descriptor.applicable_task_families:
                query_kinds = {
                    "current"
                    if descriptor.applicable_query_kinds == ("*",)
                    else descriptor.applicable_query_kinds[0]
                }
                assert metric_applies_v3(descriptor, canonical, query_kinds), descriptor.field_path


def test_replay_preserves_tombstone_history_and_relearn_current_state():
    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    assert replay.issues == ()
    assert replay.current_state[task.target_objects[0].canonical_id].value == "v2"
    assert [version.status.value for version in replay.ledgers[0].versions] == ["present", "present", "tombstone", "present"]
    resolution = resolve_query_v3(task.queries[0], replay)
    assert resolution.answer == ("v0", "v1", None, "v2")
    assert resolution.selected_event_ids == ("e0", "e1", "e2", "e3")


def test_replay_fails_closed_on_declared_ledger_disagreement():
    changed = payload()
    changed["version_history"][0]["entries"][1]["value"] = "forged"
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    assert replay.current_state == {}
    assert {issue.code for issue in replay.issues} == {"replay_version_history_mismatch"}


def test_v3_registry_exactly_covers_all_score_fields_and_principal_policies():
    assert tuple(CORE_METRIC_REGISTRY_V3) == tuple(sorted(CORE_METRIC_FIELD_PATHS))
    assert len(CORE_METRIC_REGISTRY_V3) == 74
    for descriptor in CORE_METRIC_REGISTRY_V3.values():
        if descriptor.principal:
            assert descriptor.applicable_task_families
            assert descriptor.denominator_definition.strip()


def test_e_metric_registry_has_exact_leaf_capabilities_and_support_precedence():
    from mub.vnext.scoring.registry_v3 import missing_capabilities_v3
    from mub.vnext.scoring.scorer_v3 import score_task_v3
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

    expected = {
        "deletion_scores.deletion_accuracy": {"supports_delete", "exports_action_trace"},
        "deletion_scores.delete_scope_accuracy": {"supports_delete", "supports_scoped_delete", "exports_action_trace"},
        "deletion_scores.collateral_damage_rate": {"supports_delete", "supports_isolated_reset", "exports_entries", "exports_object_keys", "exports_values"},
        "deletion_scores.ttl_compliance_rate": {"supports_ttl", "exports_action_trace", "exports_entries", "exports_object_keys", "exports_values"},
        "deletion_scores.relearn_accuracy": {"supports_delete", "exports_entries", "exports_object_keys", "exports_values"},
        "deletion_scores.forgotten_exposure_rate": {"supports_delete", "exports_retrieval_ids", "exports_object_keys", "exports_values"},
        "deletion_scores.forgotten_value_leakage_rate": {"supports_delete", "supports_native_answer"},
    }
    for path, required in expected.items():
        assert set(CORE_METRIC_REGISTRY_V3[path].required_capabilities) == required
        assert set(missing_capabilities_v3(CORE_METRIC_REGISTRY_V3[path], AdapterCapabilitiesV3())) == required

    changed = payload()
    changed["task_family"] = "E"
    task = MemUpdateTaskV3.model_validate(changed)
    prediction = AnswerPredictionV3(query_id="q", raw_output="ok", parsed_answer=["v0", "v1", None, "v2"], format_valid=True)
    run = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id="e-caps", answer_predictions=(prediction,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed")
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    config = ScorerConfigV3(requested_metric_fields=("deletion_scores.deletion_accuracy",))
    score = score_task_v3(task, run, authenticated_context(task, run, info, AdapterCapabilitiesV3(), config))
    assert score.supported_metric_fields["deletion_scores.deletion_accuracy"].reason.value == "not_supported"
    declared = AdapterCapabilitiesV3(supports_delete=True, exports_action_trace=True)
    score = score_task_v3(task, run, authenticated_context(task, run, info, declared, config))
    assert score.supported_metric_fields["deletion_scores.deletion_accuracy"].reason.value == "missing_artifact"

    family_f = MemUpdateTaskV3.model_validate(payload())
    family_f_run = run.model_copy(update={"task_id": family_f.task_id})
    score = score_task_v3(family_f, family_f_run, authenticated_context(family_f, family_f_run, info, AdapterCapabilitiesV3(), config))
    assert score.supported_metric_fields["deletion_scores.deletion_accuracy"].reason.value == "not_applicable"

    failed = run.model_copy(update={"completion_status": "failed", "answer_predictions": (), "exceptions": ({"type": "boom"},)})
    score = score_task_v3(task, failed, authenticated_context(task, failed, info, AdapterCapabilitiesV3(), config))
    assert score.supported_metric_fields["deletion_scores.deletion_accuracy"].reason.value == "runtime_failed"


def test_authenticated_scoring_context_rejects_manifest_capability_config_and_task_substitution():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    run = TaskRunRecordV3(
        task_id="t", adapter_id="adapter", run_id="run",
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="failed", exceptions=({"type": "boom"},),
    )
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    caps = AdapterCapabilitiesV3()
    context = authenticated_context(task, run, info, caps)
    assert context.task_id == task.task_id
    assert context.run_id == run.run_id
    assert context.capabilities_sha256 == context.capability_hash(caps)
    assert context.adapter_configuration_hash == info.configuration_hash
    assert context.scorer_configuration_hash == ScorerConfigV3().configuration_hash

    forged_task = task.model_copy(update={"source": task.source.model_copy(update={"source_id": "substituted"})})
    with pytest.raises(ValueError, match="authenticated task substitution"):
        score_task_v3(forged_task, run, context)

    forged_caps = context.run_manifest.model_copy(update={"adapter_capabilities": AdapterCapabilitiesV3(supports_historical_query=True)})
    with pytest.raises(ValueError, match="manifest|capabilit"):
        score_task_v3(task, run, context.model_copy(update={"run_manifest": forged_caps}))

    forged_config = context.run_manifest.model_copy(update={"scorer_config": ScorerConfigV3(requested_metric_fields=("answer_scores.exact_match",))})
    with pytest.raises(ValueError, match="manifest|config"):
        score_task_v3(task, run, context.model_copy(update={"run_manifest": forged_config}))

    bundle = authenticated_manifests(task, run, info, caps, ScorerConfigV3())
    task_manifest, run_manifest, task_artifact, run_artifact, task_manifest_sha256, run_manifest_sha256 = bundle
    wrong_binding = run_manifest.model_copy(update={"task_manifest": run_manifest.task_manifest.model_copy(update={"sha256": "0" * 64})})
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3
    with pytest.raises(ValueError, match="manifest"):
        VerifiedScoringContextV3.from_authenticated_manifests(task=task, run=run, task_manifest=task_manifest, run_manifest=wrong_binding, task_artifact=task_artifact, run_artifact=run_artifact, authenticated_task_manifest_sha256=task_manifest_sha256, authenticated_run_manifest_sha256=run_manifest_sha256)
    with pytest.raises(ValueError, match="unauthenticated"):
        VerifiedScoringContextV3.create_verified(adapter_info=info, capabilities=caps, scorer_config=ScorerConfigV3(), run_id="run")
    unavailable = run_manifest.model_copy(update={"capability_verification_artifact": None})
    with pytest.raises(ValueError, match="manifest|capability verification"):
        VerifiedScoringContextV3.from_authenticated_manifests(task=task, run=run, task_manifest=task_manifest, run_manifest=unavailable, task_artifact=task_artifact, run_artifact=run_artifact, authenticated_task_manifest_sha256=task_manifest_sha256, authenticated_run_manifest_sha256=run_manifest_sha256)
    with pytest.raises(ValueError, match="authenticated run manifest hash"):
        VerifiedScoringContextV3.from_authenticated_manifests(task=task, run=run, task_manifest=task_manifest, run_manifest=run_manifest, task_artifact=task_artifact, run_artifact=run_artifact, authenticated_task_manifest_sha256=task_manifest_sha256, authenticated_run_manifest_sha256="0" * 64)


def test_manifest_record_hashes_require_exact_multi_record_coverage_and_membership():
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task1 = MemUpdateTaskV3.model_validate(payload())
    task2 = task1.model_copy(update={"task_id": "t2"})
    provenance = ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1")
    run1 = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id="run", parser_extractor_provenance=provenance, completion_status="failed", exceptions=({"type": "boom"},))
    run2 = TaskRunRecordV3(task_id="t2", adapter_id="adapter", run_id="run", parser_extractor_provenance=provenance, completion_status="failed", exceptions=({"type": "boom"},))
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    caps = AdapterCapabilitiesV3()
    base_task_manifest, base_run_manifest, task_artifact, run_artifact, _, _ = authenticated_manifests(task1, run1, info, caps, ScorerConfigV3())
    task_hashes = {item.task_id: hashlib.sha256(json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest() for item in (task1, task2)}
    run_hashes = {item.task_id: hashlib.sha256(json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest() for item in (run1, run2)}
    task_data = base_task_manifest.model_dump(mode="python")
    task_data.update(split_counts={"test": 2}, family_difficulty_counts={"F.easy": 2}, semantic_core_counts={"c": 2}, task_record_hashes=task_hashes, task_file_paths_and_hashes=({**task_artifact, "record_count": 2},))
    task_manifest = TaskManifestV3.model_validate(task_data)
    task_manifest_hash = hashlib.sha256(json.dumps(task_manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_data = base_run_manifest.model_dump(mode="python")
    run_data.update(task_manifest={"path": "task-manifest.json", "sha256": task_manifest_hash, "media_type": "application/json"}, expected_task_count=2, completed_task_count=0, failed_task_count=2, not_supported_task_count=0, run_record_hashes=run_hashes, normalized_runtime_artifacts=({**run_artifact, "record_count": 2},))
    run_manifest = RunManifestV3.model_validate(run_data)
    run_manifest_hash = hashlib.sha256(json.dumps(run_manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    context = VerifiedScoringContextV3.from_authenticated_manifests(task=task1, run=run1, task_manifest=task_manifest, run_manifest=run_manifest, task_artifact=task_manifest.task_file_paths_and_hashes[0], run_artifact=run_manifest.normalized_runtime_artifacts[0], authenticated_task_manifest_sha256=task_manifest_hash, authenticated_run_manifest_sha256=run_manifest_hash)
    assert context.task_id == "t"

    swapped_data = run_manifest.model_dump(mode="python")
    swapped_data["run_record_hashes"] = {"t": run_hashes["t2"], "t2": run_hashes["t"]}
    swapped = RunManifestV3.model_validate(swapped_data)
    swapped_hash = hashlib.sha256(json.dumps(swapped.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="run record membership"):
        VerifiedScoringContextV3.from_authenticated_manifests(task=task1, run=run1, task_manifest=task_manifest, run_manifest=swapped, task_artifact=task_manifest.task_file_paths_and_hashes[0], run_artifact=swapped.normalized_runtime_artifacts[0], authenticated_task_manifest_sha256=task_manifest_hash, authenticated_run_manifest_sha256=swapped_hash)

    for hashes in ({"t": run_hashes["t"]}, {**run_hashes, "extra": H}):
        invalid = run_manifest.model_dump(mode="python")
        invalid["run_record_hashes"] = hashes
        with pytest.raises(Exception, match="cover expected tasks exactly"):
            RunManifestV3.model_validate(invalid)


def test_scorer_module_exposes_verified_context_and_rejects_identity_mismatch():
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    caps = AdapterCapabilitiesV3(
        supports_isolated_reset=True, supports_event_ingest=True, supports_add=True,
        supports_update=True, supports_delete=True, supports_scoped_delete=True,
        supports_historical_query=True, exports_version_history=True,
        exports_entries=True, exports_object_keys=True, exports_values=True,
        exports_action_trace=True,
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    run = TaskRunRecordV3(
        task_id="other", adapter_id="adapter", run_id="run",
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="failed", exceptions=({"type": "boom"},),
    )
    context = authenticated_context(task, run.model_copy(update={"task_id": "t"}), info, caps)
    with pytest.raises(ValueError, match="task_id mismatch"):
        score_task_v3(task, run, context)


def test_runtime_failure_precedes_capability_absence_in_support_map():
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    caps = AdapterCapabilitiesV3()
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    run = TaskRunRecordV3(
        task_id="t", adapter_id="adapter", run_id="run",
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="failed", exceptions=({"type": "boom"},),
    )
    context = authenticated_context(task, run, info, caps)
    score = score_task_v3(task, run, context)
    assert score.supported_metric_fields["historical_scores.ordered_history_accuracy"].reason.value == "runtime_failed"


def test_all_single_object_f_selectors_resolve_from_typed_ledger():
    from mub.vnext.contracts.v3.task import MemoryQueryV3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    key = task.target_objects[0]
    cases = (
        ("current", "current", {"kind": "current"}, "string", "v2"),
        ("previous", "previous", {"kind": "previous"}, "string", None),
        ("exact", "point_in_time", {"kind": "exact_version", "version_index": 1}, "string", "v1"),
        ("event", "point_in_time", {"kind": "event_anchor", "event_id": "e2"}, "string", None),
        ("time", "point_in_time", {"kind": "logical_time_anchor", "logical_time": "002"}, "string", None),
        ("transition", "transition", {"kind": "transition", "from_version_index": 1, "to_version_index": 3}, "object", {"from": "v1", "to": "v2"}),
        ("history", "ordered_history", {"kind": "ordered_history"}, "list", ("v0", "v1", None, "v2")),
    )
    for query_id, query_type, selector, schema, answer in cases:
        query = MemoryQueryV3(query_id=query_id, query_type=query_type, text="?", selector=selector, target_object_keys=(key,), answer_schema=schema, evaluation_mode="state_direct")
        resolved = resolve_query_v3(query, replay, task.events)
        assert resolved.issues == ()
        assert resolved.answer == answer


def test_ttl_boundary_is_half_open_and_expiry_inclusive():
    from mub.vnext.contracts.v3.task import MemoryQueryV3

    changed = payload()
    changed["actions"][2]["scope"] = "ttl"
    changed["version_history"][0]["entries"][1]["valid_until_event_id"] = None
    changed["version_history"][0]["entries"][2]["valid_from_event_id"] = None
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    key = task.target_objects[0]
    before = MemoryQueryV3(query_id="before", query_type="point_in_time", text="?", selector={"kind": "logical_time_anchor", "logical_time": "0019"}, target_object_keys=(key,), answer_schema="string", evaluation_mode="state_direct")
    boundary = before.model_copy(update={"query_id": "boundary", "selector": {"kind": "logical_time_anchor", "logical_time": "002"}})
    assert resolve_query_v3(before, replay, task.events).answer == "v1"
    assert resolve_query_v3(boundary, replay, task.events).answer is None


def test_g_evidence_evaluator_uses_ordered_operands_and_rejects_unknown_ops():
    from mub.vnext.contracts.v3.task import QueryGoldEvidenceV3
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    key = task.target_objects[0]
    valid = QueryGoldEvidenceV3(
        query_id="g", answer="v2", supporting_object_keys=(key,), supporting_event_ids=("e3",),
        derivation_steps=(
            {"step_id": "read", "operation": "read", "supporting_object_keys": (key,), "supporting_event_ids": ("e3",)},
            {"step_id": "answer", "operation": "answer", "input_step_ids": ("read",)},
        ), final_derivation_step_id="answer",
    )
    assert evaluate_evidence_v3(valid, replay).answer == "v2"
    hostile = valid.model_copy(update={"derivation_steps": (valid.derivation_steps[0], valid.derivation_steps[1].model_copy(update={"operation": "python_eval"}))})
    result = evaluate_evidence_v3(hostile, replay)
    assert result.answer is None
    assert result.issues[0].code == "unsupported_derivation_operation"


def test_reference_shaped_completed_run_is_perfect_on_supported_principal_metrics():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, MemorySnapshotV3, ParsedManagerActionV3, RetrievalTraceV3
    from mub.vnext.scoring.registry_v3 import metric_applies_v3, missing_capabilities_v3
    from mub.vnext.scoring.scorer_v3 import _metric_applies_to_task_v3, score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    caps = AdapterCapabilitiesV3(
        supports_isolated_reset=True, supports_event_ingest=True, supports_add=True,
        supports_update=True, supports_delete=True, supports_historical_query=True,
        exports_version_history=True, exports_entries=True, exports_object_keys=True,
        exports_values=True, exports_action_trace=True, exports_retrieval_ids=True,
    )
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    parsed_actions = tuple(
        ParsedManagerActionV3(
            event_id=action.event_id, operation=action.operation, observed_scope=action.scope,
            target_object_keys=action.target_object_keys, value=action.value, format_valid=True,
            execution_status="executed", fallback_used=False, raw_output="ok",
        )
        for action in task.actions
    )
    key = task.target_objects[0]
    history_entries = tuple(
        MemoryEntryRecordV3(
            entry_id=f"version-{index}", content=str(value), object_key_candidate=key,
            value_candidate=value, version_index=index, source_event_ids=(f"e{index}",),
        )
        for index, value in enumerate(("v0", "v1", None, "v2"))
    )
    trace = RetrievalTraceV3(query_id="q", retrieved_entries=history_entries, ranks=(1, 2, 3, 4), gold_in_context=True, stale_in_context=True, distractor_in_context=False)
    run = TaskRunRecordV3(
        task_id=task.task_id, adapter_id="adapter", run_id="run", parsed_actions=parsed_actions,
        memory_snapshots=(MemorySnapshotV3(after_event_id="e3", state_by_object={key.canonical_id: "v2"}, store_size=1),),
        retrieval_traces=(trace,),
        answer_predictions=(AnswerPredictionV3(query_id="q", raw_output="ok", parsed_answer=["v0", "v1", None, "v2"], format_valid=True),),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    context = authenticated_context(task, run, info, caps)
    score = score_task_v3(task, run, context)
    observed = 0
    for path, descriptor in CORE_METRIC_REGISTRY_V3.items():
        if not descriptor.principal:
            continue
        layer, leaf = path.split(".", 1)
        value = getattr(getattr(score, layer), leaf)
        if value is None:
            assert score.supported_metric_fields[path].reason.value in {"not_applicable", "not_supported"}, path
            continue
        observed += 1
        assert value == (0.0 if descriptor.direction == "lower" else 1.0), path
    assert observed >= 5


def test_event_anchored_replay_does_not_invent_logical_time():
    changed = payload()
    for action in changed["actions"]:
        action.pop("effective_at", None)
    for entry in changed["version_history"][0]["entries"]:
        entry.pop("logical_time", None)
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    assert all(version.logical_time is None for version in replay.ledgers[0].versions)


def test_current_selector_resolves_tombstone_as_structured_absence():
    changed = payload()
    changed["events"] = changed["events"][:3]
    changed["actions"] = changed["actions"][:3]
    changed["version_history"][0]["entries"] = changed["version_history"][0]["entries"][:3]
    changed["version_history"][0]["entries"][-1].pop("valid_until_event_id", None)
    changed["queries"][0] = {
        "query_id": "q", "query_type": "current", "text": "?", "selector": {"kind": "current"},
        "target_object_keys": changed["target_objects"], "answer_schema": "list", "evaluation_mode": "state_direct",
    }
    changed["gold_evidence"][0]["answer"] = [None]
    changed["gold_evidence"][0]["supporting_event_ids"] = ["e2"]
    changed["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = ["e2"]
    task = MemUpdateTaskV3.model_validate(changed)
    resolution = resolve_query_v3(task.queries[0], replay_task_v3(task), task.events)
    assert resolution.issues == ()
    assert resolution.answer == (None,)


def test_replay_fails_closed_when_f_answer_disagrees_with_typed_selector():
    changed = payload()
    changed["gold_evidence"][0]["answer"] = ["forged"]
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    assert replay.current_state == {}
    assert replay.issues[0].code == "query_gold_answer_mismatch"


def test_replay_exposes_update_superseded_values_as_obsolete():
    replay = replay_task_v3(MemUpdateTaskV3.model_validate(payload()))
    assert replay.obsolete_present_values == ("v0", "v1")


def test_supported_stale_count_metric_executes_without_dead_branch_name_error():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    query = task.queries[0]
    trace = RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=(MemoryEntryRecordV3(entry_id="old", content="v0", object_key_candidate=task.target_objects[0], value_candidate="v0", version_index=0, source_event_ids=("e0",)),),
    )
    value, issue = _metric_value(
        "retrieval_scores.stale_count_in_context", task,
        TaskRunRecordV3(task_id="t", adapter_id="a", run_id="r", retrieval_traces=(trace,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed"),
        None, replay, {query.query_id: resolve_query_v3(query, replay, task.events)},
        {item.query_id: item for item in task.gold_evidence}, {}, {query.query_id: trace}, [],
    )
    assert issue is None
    assert value == 1


def test_same_value_different_version_is_not_current_entry_match():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.scorer_v3 import _entry_matches_version

    changed = payload()
    changed["actions"][0]["value"] = "x"
    changed["actions"][1]["value"] = "y"
    changed["actions"][3]["value"] = "x"
    changed["version_history"][0]["entries"][0]["value"] = "x"
    changed["version_history"][0]["entries"][1]["value"] = "y"
    changed["version_history"][0]["entries"][3]["value"] = "x"
    changed["gold_evidence"][0]["answer"] = ["x", "y", None, "x"]
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    stale_v0 = MemoryEntryRecordV3(entry_id="v0", content="x", object_key_candidate=task.target_objects[0], value_candidate="x", version_index=0, source_event_ids=("e0",))
    assert not _entry_matches_version(stale_v0, replay.ledgers[0].versions[3])
    assert replay.obsolete_present_values == ("x", "y")

    ambiguous = MemoryEntryRecordV3(entry_id="ambiguous", content="x", object_key_candidate=task.target_objects[0], value_candidate="x")
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
    from mub.vnext.scoring.scorer_v3 import _metric_value
    run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="ambiguous", memory_snapshots=(MemorySnapshotV3(after_event_id="e3", entries=(ambiguous,), store_size=1),), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed")
    value, detail = _metric_value("store_scores.stale_conflicting_value_count", task, run, None, replay, {}, {}, {}, {}, [])
    assert value is None
    assert "version identity" in detail

    current_payload = deepcopy(changed)
    current_payload["queries"][0] = {"query_id": "q", "query_type": "current", "text": "?", "selector": {"kind": "current"}, "target_object_keys": current_payload["target_objects"], "answer_schema": "string", "evaluation_mode": "state_direct"}
    current_payload["gold_evidence"][0] = {"query_id": "q", "answer": "x", "supporting_object_keys": current_payload["target_objects"], "supporting_event_ids": ["e3"], "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": current_payload["target_objects"], "supporting_event_ids": ["e3"]}], "final_derivation_step_id": "read"}
    current_task = MemUpdateTaskV3.model_validate(current_payload)
    current_replay = replay_task_v3(current_task)
    from mub.vnext.contracts.v3.runtime import RetrievalTraceV3
    ambiguous_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(ambiguous,))
    value, detail = _metric_value("retrieval_scores.current_recall_at_k", current_task, run.model_copy(update={"retrieval_traces": (ambiguous_trace,)}), None, current_replay, {"q": resolve_query_v3(current_task.queries[0], current_replay, current_task.events)}, {"q": current_task.gold_evidence[0]}, {}, {"q": ambiguous_trace}, [])
    assert value is None
    assert "version identity" in detail
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    correct_x = AnswerPredictionV3(query_id="q", raw_output="x", parsed_answer="x", format_valid=True)
    value, detail = _metric_value("answer_scores.stale_copied", current_task, run.model_copy(update={"answer_predictions": (correct_x,)}), None, current_replay, {"q": resolve_query_v3(current_task.queries[0], current_replay, current_task.events)}, {"q": current_task.gold_evidence[0]}, {"q": correct_x}, {}, [])
    assert detail is None
    assert value == 0.0

    wrong_value_v3 = MemoryEntryRecordV3(entry_id="wrong-v3", content="wrong", object_key_candidate=current_task.target_objects[0], value_candidate="wrong", version_index=3, source_event_ids=("e3",))
    assert not _entry_matches_version(wrong_value_v3, current_replay.ledgers[0].versions[3])
    wrong_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(wrong_value_v3,))
    value, detail = _metric_value("retrieval_scores.current_recall_at_k", current_task, run.model_copy(update={"retrieval_traces": (wrong_trace,)}), None, current_replay, {"q": resolve_query_v3(current_task.queries[0], current_replay, current_task.events)}, {"q": current_task.gold_evidence[0]}, {}, {"q": wrong_trace}, [])
    assert detail is None
    assert value == 0.0

    missing_value_v3 = MemoryEntryRecordV3(entry_id="missing-v3", content="missing", object_key_candidate=current_task.target_objects[0], version_index=3, source_event_ids=("e3",))
    missing_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(missing_value_v3,))
    value, detail = _metric_value("retrieval_scores.current_recall_at_k", current_task, run.model_copy(update={"retrieval_traces": (missing_trace,)}), None, current_replay, {"q": resolve_query_v3(current_task.queries[0], current_replay, current_task.events)}, {"q": current_task.gold_evidence[0]}, {}, {"q": missing_trace}, [])
    assert value is None
    assert "value evidence" in detail
    missing_store = run.model_copy(update={"memory_snapshots": (MemorySnapshotV3(after_event_id="e3", entries=(missing_value_v3,), store_size=1),)})
    value, detail = _metric_value("store_scores.stale_conflicting_value_count", current_task, missing_store, None, current_replay, {}, {"q": current_task.gold_evidence[0]}, {}, {}, [])
    assert value is None
    assert "value evidence" in detail

    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3
    evidence = {item.query_id: item for item in task.gold_evidence}
    layers = {"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}}
    current_v3 = MemoryEntryRecordV3(entry_id="v3", content="x", object_key_candidate=task.target_objects[0], value_candidate="x", version_index=3, source_event_ids=("e3",))
    current_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(current_v3,), stale_in_context=False)
    current_run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="current", retrieval_traces=(current_trace,), parser_extractor_provenance=run.parser_extractor_provenance, completion_status="completed")
    flags = derive_failure_flags_v3(task=task, run=current_run, replay=replay, layer_values=layers, predictions={}, traces={"q": current_trace}, evidence=evidence)
    assert "stale_retrieved" not in flags
    stale_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(stale_v0,), stale_in_context=True)
    stale_run = current_run.model_copy(update={"retrieval_traces": (stale_trace,)})
    flags = derive_failure_flags_v3(task=task, run=stale_run, replay=replay, layer_values=layers, predictions={}, traces={"q": stale_trace}, evidence=evidence)
    assert "stale_retrieved" in flags
    from mub.vnext.scoring.scorer_v3 import _action_facts
    value, detail = _metric_value("deletion_scores.forgotten_exposure_rate", task, stale_run, None, replay, {}, evidence, {}, {"q": stale_trace}, _action_facts(task, stale_run))
    assert detail is None
    assert value == 1.0
    explicit_run = run.model_copy(update={"memory_snapshots": (MemorySnapshotV3(after_event_id="e3", entries=(stale_v0,), store_size=1),)})
    value, detail = _metric_value("store_scores.stale_conflicting_value_count", task, explicit_run, None, replay, {}, evidence, {}, {}, _action_facts(task, explicit_run))
    assert detail is None
    assert value == 1


def future_ttl_payload():
    changed = payload()
    for index, event in enumerate(changed["events"]):
        event["timestamp"] = ("000", "001", "002", "015")[index]
    changed["actions"][2]["scope"] = "ttl"
    changed["actions"][2]["effective_at"] = "010"
    changed["actions"][3]["effective_at"] = "015"
    changed["events"].insert(3, {"event_id": "e-later", "sequence_index": 3, "timestamp": "012", "raw_text": "later", "normalized_text": "later", "role": "neutral", "gold_action_ids": ["a-later"]})
    changed["events"][4]["sequence_index"] = 4
    changed["actions"].insert(3, {"action_id": "a-later", "event_id": "e-later", "operation": "NOOP"})
    changed["version_history"][0]["entries"][1]["valid_until_event_id"] = None
    changed["version_history"][0]["entries"][2]["valid_from_event_id"] = None
    changed["version_history"][0]["entries"][2]["valid_until_event_id"] = "e3"
    changed["version_history"][0]["entries"][2]["logical_time"] = "010"
    changed["version_history"][0]["entries"][3]["logical_time"] = "015"
    return changed


def test_future_ttl_transition_preserves_state_until_exact_expiry():
    from mub.vnext.contracts.v3.task import MemoryQueryV3

    task = MemUpdateTaskV3.model_validate(future_ttl_payload())
    replay = replay_task_v3(task)
    assert replay.issues == ()
    key = task.target_objects[0]
    cases = (
        ({"kind": "event_anchor", "event_id": "e2"}, "v1"),
        ({"kind": "logical_time_anchor", "logical_time": "009"}, "v1"),
        ({"kind": "logical_time_anchor", "logical_time": "010"}, None),
        ({"kind": "logical_time_anchor", "logical_time": "011"}, None),
        ({"kind": "event_anchor", "event_id": "e-later"}, None),
    )
    for index, (selector, expected) in enumerate(cases):
        query = MemoryQueryV3(query_id=f"ttl-{index}", query_type="point_in_time", text="?", selector=selector, target_object_keys=(key,), answer_schema="string", evaluation_mode="state_direct")
        resolution = resolve_query_v3(query, replay, task.events)
        assert resolution.issues == ()
        assert resolution.answer == expected
    assert replay.current_state[key.canonical_id].value == "v2"


@pytest.mark.parametrize(
    ("event_id", "answer", "answer_schema", "support_event_id"),
    (("e2", "v1", "string", "e1"), ("e-later", [None], "list", "e2")),
)
def test_event_anchor_contract_uses_interval_and_logical_time_without_lookahead(
    event_id, answer, answer_schema, support_event_id,
):
    changed = future_ttl_payload()
    changed["queries"][0] = {
        "query_id": "q", "query_type": "point_in_time", "text": "?",
        "selector": {"kind": "event_anchor", "event_id": event_id},
        "target_object_keys": changed["target_objects"], "answer_schema": answer_schema,
        "evaluation_mode": "state_direct",
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": answer, "supporting_object_keys": changed["target_objects"],
        "supporting_event_ids": [support_event_id],
        "derivation_steps": [{
            "step_id": "read", "operation": "read",
            "supporting_object_keys": changed["target_objects"],
            "supporting_event_ids": [support_event_id],
        }],
        "final_derivation_step_id": "read",
    }

    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    resolution = resolve_query_v3(task.queries[0], replay, task.events)
    assert replay.issues == ()
    assert resolution.issues == ()
    assert resolution.answer == task.gold_evidence[0].answer
    assert resolution.selected_event_ids == (support_event_id,)


def ttl_horizon_payload(query_time, answer):
    changed = future_ttl_payload()
    changed["events"] = changed["events"][:3]
    changed["actions"] = changed["actions"][:3]
    changed["version_history"][0]["entries"] = changed["version_history"][0]["entries"][:3]
    changed["version_history"][0]["entries"][2]["valid_until_event_id"] = None
    changed["queries"][0] = {
        "query_id": "q", "query_type": "point_in_time", "text": "?",
        "selector": {"kind": "logical_time_anchor", "logical_time": query_time},
        "target_object_keys": changed["target_objects"], "answer_schema": "list" if answer is None else "string", "evaluation_mode": "state_direct",
    }
    support_event = "e1" if answer == "v1" else "e2"
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": [None] if answer is None else answer, "supporting_object_keys": changed["target_objects"],
        "supporting_event_ids": [support_event],
        "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": changed["target_objects"], "supporting_event_ids": [support_event]}],
        "final_derivation_step_id": "read",
    }
    return changed


def test_ttl_horizon_keeps_future_expiry_pending_until_query_reaches_boundary():
    before_task = MemUpdateTaskV3.model_validate(ttl_horizon_payload("005", "v1"))
    before = replay_task_v3(before_task)
    key = before_task.target_objects[0]
    assert before.issues == ()
    assert before.horizon_logical_time == "005"
    assert before.current_state[key.canonical_id].value == "v1"

    boundary_task = MemUpdateTaskV3.model_validate(ttl_horizon_payload("010", None))
    boundary = replay_task_v3(boundary_task)
    assert boundary.issues == ()
    assert boundary.horizon_logical_time == "010"
    assert key.canonical_id not in boundary.current_state


def test_current_query_contract_uses_pre_expiry_horizon_not_future_tombstone():
    changed = ttl_horizon_payload("005", "v1")
    changed["queries"][0]["query_type"] = "current"
    changed["queries"][0]["selector"] = {"kind": "current"}
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    assert resolve_query_v3(task.queries[0], replay, task.events).answer == "v1"


def test_ordered_history_contract_and_replay_exclude_future_ttl_versions():
    changed = ttl_horizon_payload("005", "v1")
    changed["queries"][0].update(
        query_type="ordered_history",
        selector={"kind": "ordered_history"},
        answer_schema="list",
    )
    changed["gold_evidence"][0].update(
        answer=["v0", "v1"],
        supporting_event_ids=["e0", "e1"],
        derivation_steps=[{
            "step_id": "history",
            "operation": "ordered_history",
            "supporting_object_keys": changed["target_objects"],
            "supporting_event_ids": ["e0", "e1"],
        }],
        final_derivation_step_id="history",
    )

    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    resolution = resolve_query_v3(task.queries[0], replay, task.events)
    assert replay.issues == ()
    assert resolution.issues == ()
    assert resolution.answer == ("v0", "v1")
    assert tuple(version.version_index for version in resolution.selected_versions) == (0, 1)

    out_of_horizon = deepcopy(changed)
    out_of_horizon["queries"][0]["selector"]["end_version_index"] = 2
    out_of_horizon["gold_evidence"][0]["supporting_event_ids"].append("e2")
    out_of_horizon["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"].append("e2")
    with pytest.raises(ValueError, match="unknown end version"):
        MemUpdateTaskV3.model_validate(out_of_horizon)


def test_ttl_scorer_uses_expiry_snapshot_not_scheduling_snapshot():
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3, ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    task = MemUpdateTaskV3.model_validate(future_ttl_payload())
    replay = replay_task_v3(task)
    ttl = next(action for action in task.actions if action.scope is not None and action.scope.value == "ttl")
    parsed = ParsedManagerActionV3(event_id=ttl.event_id, operation=ttl.operation, observed_scope=ttl.scope, target_object_keys=ttl.target_object_keys, format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
    key = task.target_objects[0]
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="r", parsed_actions=(parsed,),
        memory_snapshots=(
            MemorySnapshotV3(after_event_id="e3", state_by_object={key.canonical_id: "v2"}, store_size=1),
            MemorySnapshotV3(after_event_id="e2", state_by_object={key.canonical_id: "v1"}, store_size=1),
            MemorySnapshotV3(after_event_id="e-later", state_by_object={}, store_size=0),
        ),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed",
    )
    value, issue = _metric_value("deletion_scores.ttl_compliance_rate", task, run, None, replay, {}, {}, {}, {}, _action_facts(task, run))
    assert issue is None
    assert value == 1.0
    value, issue = _metric_value("state_scores.final_state_accuracy", task, run, None, replay, {}, {}, {}, {}, _action_facts(task, run))
    assert issue is None
    assert value == 1.0


def test_ttl_and_relearn_metric_applicability_is_condition_specific():
    from mub.vnext.scoring.scorer_v3 import _metric_applies_to_task_v3

    relearn_task = MemUpdateTaskV3.model_validate(payload())
    assert not _metric_applies_to_task_v3("deletion_scores.ttl_compliance_rate", relearn_task, replay_task_v3(relearn_task))
    assert _metric_applies_to_task_v3("deletion_scores.relearn_accuracy", relearn_task, replay_task_v3(relearn_task))

    ttl_task = MemUpdateTaskV3.model_validate(future_ttl_payload())
    assert _metric_applies_to_task_v3("deletion_scores.ttl_compliance_rate", ttl_task, replay_task_v3(ttl_task))
    assert _metric_applies_to_task_v3("deletion_scores.relearn_accuracy", ttl_task, replay_task_v3(ttl_task))


def test_lifecycle_support_map_uses_not_applicable_before_runtime_or_capability():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    def failed_score(task):
        run = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id=f"run-{task.task_id}", parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="failed", exceptions=({"type": "boom"},))
        info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
        return score_task_v3(task, run, authenticated_context(task, run, info, AdapterCapabilitiesV3()))

    object_payload = payload()
    object_payload["task_family"] = "E"
    object_task = MemUpdateTaskV3.model_validate(object_payload)
    object_score = failed_score(object_task)
    assert object_score.supported_metric_fields["deletion_scores.ttl_compliance_rate"].reason.value == "not_applicable"
    assert object_score.supported_metric_fields["deletion_scores.relearn_accuracy"].reason.value == "runtime_failed"

    ttl_payload = future_ttl_payload()
    ttl_payload["task_family"] = "E"
    ttl_task = MemUpdateTaskV3.model_validate(ttl_payload)
    ttl_score = failed_score(ttl_task)
    assert ttl_score.supported_metric_fields["deletion_scores.ttl_compliance_rate"].reason.value == "runtime_failed"
    assert ttl_score.supported_metric_fields["deletion_scores.relearn_accuracy"].reason.value == "runtime_failed"


def test_failure_flags_distinguish_format_only_and_authenticated_distractor_copy():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    evidence = {item.query_id: item for item in task.gold_evidence}
    empty_layers = {"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}}

    wrong_bad_format = AnswerPredictionV3(query_id="q", raw_output="bad", parsed_answer="wrong", format_valid=False)
    run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="r", answer_predictions=(wrong_bad_format,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed")
    flags = derive_failure_flags_v3(task=task, run=run, replay=replay, layer_values=empty_layers, predictions={"q": wrong_bad_format}, traces={}, evidence=evidence)
    assert "answer_format_only" not in flags

    format_only = AnswerPredictionV3(query_id="q", raw_output="bad", parsed_answer=["v0", "v1", None, "v2"], format_valid=False)
    flags = derive_failure_flags_v3(task=task, run=run.model_copy(update={"answer_predictions": (format_only,)}), replay=replay, layer_values=empty_layers, predictions={"q": format_only}, traces={}, evidence=evidence)
    assert "answer_format_only" in flags

    distractor = MemoryEntryRecordV3(entry_id="d", content="distractor", value_candidate="distractor", raw_metadata={"is_distractor": True})
    trace = RetrievalTraceV3(query_id="q", retrieved_entries=(distractor,), distractor_in_context=True)
    wrong = AnswerPredictionV3(query_id="q", raw_output="other", parsed_answer="other", format_valid=True)
    flags = derive_failure_flags_v3(task=task, run=run.model_copy(update={"answer_predictions": (wrong,), "retrieval_traces": (trace,)}), replay=replay, layer_values=empty_layers, predictions={"q": wrong}, traces={"q": trace}, evidence=evidence)
    assert "distractor_retrieved" in flags
    assert "distractor_copied" not in flags
    from mub.vnext.scoring.scorer_v3 import _metric_value
    value, detail = _metric_value("answer_scores.distractor_copied", task, run, None, replay, {}, evidence, {"q": wrong}, {"q": trace}, [])
    assert detail is None
    assert value == 0.0

    copied = AnswerPredictionV3(query_id="q", raw_output="distractor", parsed_answer="distractor", format_valid=True)
    flags = derive_failure_flags_v3(task=task, run=run.model_copy(update={"answer_predictions": (copied,), "retrieval_traces": (trace,)}), replay=replay, layer_values=empty_layers, predictions={"q": copied}, traces={"q": trace}, evidence=evidence)
    assert "distractor_copied" in flags

    gold_candidate = MemoryEntryRecordV3(entry_id="gold-overlap", content="v0", value_candidate="v0", raw_metadata={"is_distractor": True})
    overlap_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(gold_candidate,), distractor_in_context=True)
    correct = AnswerPredictionV3(query_id="q", raw_output="correct", parsed_answer=["v0", "v1", None, "v2"], format_valid=True)
    flags = derive_failure_flags_v3(task=task, run=run.model_copy(update={"answer_predictions": (correct,), "retrieval_traces": (overlap_trace,)}), replay=replay, layer_values=empty_layers, predictions={"q": correct}, traces={"q": overlap_trace}, evidence=evidence)
    assert "distractor_copied" not in flags
    value, detail = _metric_value("answer_scores.distractor_copied", task, run, None, replay, {}, evidence, {"q": correct}, {"q": overlap_trace}, [])
    assert detail is None
    assert value == 0.0
    format_only_overlap = AnswerPredictionV3(query_id="q", raw_output="bad-format", parsed_answer=["v0", "v1", None, "v2"], format_valid=False)
    flags = derive_failure_flags_v3(task=task, run=run.model_copy(update={"answer_predictions": (format_only_overlap,), "retrieval_traces": (overlap_trace,)}), replay=replay, layer_values=empty_layers, predictions={"q": format_only_overlap}, traces={"q": overlap_trace}, evidence=evidence)
    assert "answer_format_only" in flags
    assert "distractor_copied" not in flags


def g_stale_payload():
    changed = payload()
    key = changed["target_objects"][0]
    changed["queries"][0] = {
        "query_id": "q", "query_type": "update_sensitive_multi_hop", "text": "?",
        "selector": {"kind": "current"}, "target_object_keys": [key], "answer_schema": "string",
        "evaluation_mode": "state_direct", "synthesis": {"kind": "update_sensitive_multi_hop", "minimum_hops": 2},
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": "v2", "supporting_object_keys": [key], "supporting_event_ids": ["e3"],
        "derivation_steps": [
            {"step_id": "read", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e3"]},
            {"step_id": "answer", "operation": "answer", "input_step_ids": ["read"]},
        ],
        "final_derivation_step_id": "answer",
        "stale_alternative": {
            "answer": "v1", "supporting_object_keys": [key], "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "stale-read", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e1"]},
                {"step_id": "stale-answer", "operation": "answer", "input_step_ids": ["stale-read"]},
            ],
            "final_derivation_step_id": "stale-answer",
        },
    }
    return changed


def g_difference_payload():
    changed = payload()
    changed["task_family"] = "G"
    changed["events"] = changed["events"][:2]
    changed["actions"] = changed["actions"][:2]
    changed["actions"][0]["value"] = 2
    changed["actions"][1]["value"] = 3
    changed["version_history"][0]["entries"] = [
        {"version_index": 0, "status": "present", "value": 2, "valid_from_event_id": "e0", "valid_until_event_id": "e1", "logical_time": "000", "source_event_ids": ["e0"]},
        {"version_index": 1, "status": "present", "value": 3, "valid_from_event_id": "e1", "logical_time": "001", "source_event_ids": ["e1"]},
    ]
    key = changed["target_objects"][0]
    changed["queries"][0] = {
        "query_id": "q", "query_type": "update_sensitive_multi_hop", "text": "?",
        "selector": {"kind": "current"}, "target_object_keys": [key], "answer_schema": "number",
        "evaluation_mode": "state_direct", "synthesis": {"kind": "update_sensitive_multi_hop", "minimum_hops": 2},
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": 1, "supporting_object_keys": [key], "supporting_event_ids": ["e0", "e1"],
        "derivation_steps": [
            {"step_id": "previous", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e0"]},
            {"step_id": "current", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e1"]},
            {"step_id": "difference", "operation": "subtract", "input_step_ids": ["current", "previous"]},
        ],
        "final_derivation_step_id": "difference",
    }
    return changed


def test_transformed_g_answer_state_consistency_is_typed_not_applicable():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(g_difference_payload())
    prediction = AnswerPredictionV3(query_id="q", raw_output="1", parsed_answer=1, format_valid=True)
    run = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id="g-difference", answer_predictions=(prediction,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed")
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    caps = AdapterCapabilitiesV3(exports_entries=True, exports_object_keys=True, exports_values=True, supports_isolated_reset=True)
    config = ScorerConfigV3(requested_metric_fields=("answer_scores.answer_state_consistency",))
    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))
    assert score.answer_scores.answer_state_consistency is None
    assert score.supported_metric_fields["answer_scores.answer_state_consistency"].reason.value == "not_applicable"


def test_evidence_metrics_tag_identifier_domains_before_overlap():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    changed = g_stale_payload()
    changed["task_family"] = "G"
    changed["gold_evidence"][0]["derivation_steps"][0]["step_id"] = "e3"
    changed["gold_evidence"][0]["derivation_steps"][1]["input_step_ids"] = ["e3"]
    task = MemUpdateTaskV3.model_validate(changed)
    key = task.target_objects[0]
    prediction = AnswerPredictionV3(
        query_id="q", raw_output="v2", parsed_answer="v2", format_valid=True,
        cited_event_ids=("e3",), cited_object_keys=(key,), cited_derivation_step_ids=("answer",),
    )
    run = TaskRunRecordV3(task_id="t", adapter_id="adapter", run_id="g-collision", answer_predictions=(prediction,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed")
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    caps = AdapterCapabilitiesV3(exports_evidence_linkage=True)
    config = ScorerConfigV3(requested_metric_fields=("synthesis_scores.evidence_recall", "synthesis_scores.evidence_f1"))
    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))
    assert score.synthesis_scores.evidence_recall == 0.75
    assert score.synthesis_scores.evidence_f1 == pytest.approx(6 / 7)


def test_g_gold_contract_registers_strict_stale_alternative_derivation():
    task = MemUpdateTaskV3.model_validate(g_stale_payload())
    alternative = task.gold_evidence[0].stale_alternative
    assert alternative.answer == "v1"
    assert alternative.supporting_event_ids == ("e1",)
    evaluation = __import__("mub.vnext.validation.replay_v3", fromlist=["evaluate_evidence_v3"]).evaluate_evidence_v3(task.gold_evidence[0], replay_task_v3(task), alternative)
    assert evaluation.stale_alternative_answer == "v1"
    assert evaluation.stale_required_event_ids == ("e1",)
    assert evaluation.stale_required_step_ids == ("stale-read", "stale-answer")
    with pytest.raises(Exception):
        alternative.derivation_steps[0].operation = "eval"
    shallow = g_stale_payload()
    shallow_alt = shallow["gold_evidence"][0]["stale_alternative"]
    shallow_alt["derivation_steps"] = [shallow_alt["derivation_steps"][0]]
    shallow_alt["final_derivation_step_id"] = "stale-read"
    with pytest.raises(Exception, match="stale alternative.*minimum_hops"):
        MemUpdateTaskV3.model_validate(shallow)

    multi = payload()
    first = multi["target_objects"][0]
    second = {**first, "entity": "e2"}
    multi["target_objects"].append(second)
    multi["version_history"].append({"object_key": second, "entries": [{"version_index": 0, "status": "present", "value": "z", "valid_from_event_id": "e0", "logical_time": "000", "source_event_ids": ["e0"]}]})
    multi["queries"][0] = {"query_id": "q", "query_type": "multi_object_current_consistency", "text": "?", "selector": {"kind": "multi_object_current", "object_keys": [first, second]}, "target_object_keys": [first, second], "answer_schema": "boolean", "evaluation_mode": "state_direct", "synthesis": {"kind": "multi_object_current_consistency", "minimum_objects": 2}}
    multi["gold_evidence"][0] = {
        "query_id": "q", "answer": False, "supporting_object_keys": [first, second], "supporting_event_ids": ["e0", "e3"],
        "derivation_steps": [
            {"step_id": "first-current", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "second-current", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "both", "operation": "equals", "input_step_ids": ["first-current", "second-current"]},
        ], "final_derivation_step_id": "both",
        "stale_alternative": {"answer": True, "supporting_object_keys": [first, second], "supporting_event_ids": ["e1"], "derivation_steps": [{"step_id": "one", "operation": "equals", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]}], "final_derivation_step_id": "one"},
    }
    with pytest.raises(Exception, match="stale alternative.*minimum_objects"):
        MemUpdateTaskV3.model_validate(multi)


def multi_object_consistency_payload():
    changed = payload()
    first = changed["target_objects"][0]
    second = {**first, "entity": "e2"}
    changed["target_objects"].append(second)
    changed["version_history"].append({
        "object_key": second,
        "entries": [{
            "version_index": 0, "status": "present", "value": "z",
            "valid_from_event_id": "e0", "logical_time": "000", "source_event_ids": ["e0"],
        }],
    })
    changed["queries"][0] = {
        "query_id": "q", "query_type": "multi_object_current_consistency", "text": "?",
        "selector": {"kind": "multi_object_current", "object_keys": [first, second]},
        "target_object_keys": [first, second], "answer_schema": "boolean", "evaluation_mode": "state_direct",
        "synthesis": {"kind": "multi_object_current_consistency", "minimum_objects": 2},
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": False, "supporting_object_keys": [first, second],
        "supporting_event_ids": ["e0", "e3"],
        "derivation_steps": [
            {"step_id": "first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "second"]},
        ],
        "final_derivation_step_id": "equals",
    }
    return changed


@pytest.mark.parametrize("location", ["primary-no-input", "primary-one-input", "stale-no-input"])
def test_multi_object_consistency_requires_two_distinct_reachable_read_operands(location):
    changed = multi_object_consistency_payload()
    evidence = changed["gold_evidence"][0]
    if location == "primary-no-input":
        evidence["derivation_steps"] = [{
            "step_id": "equals", "operation": "equals",
            "supporting_object_keys": evidence["supporting_object_keys"],
            "supporting_event_ids": evidence["supporting_event_ids"],
        }]
    elif location == "primary-one-input":
        evidence["derivation_steps"] = [evidence["derivation_steps"][0], {
            "step_id": "equals", "operation": "equals", "input_step_ids": ["first"],
        }]
    else:
        evidence["stale_alternative"] = {
            "answer": True, "supporting_object_keys": evidence["supporting_object_keys"],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [{
                "step_id": "stale-equals", "operation": "equals",
                "supporting_object_keys": evidence["supporting_object_keys"],
                "supporting_event_ids": ["e0", "e1"],
            }],
            "final_derivation_step_id": "stale-equals",
        }
    with pytest.raises(ValueError, match="minimum_objects"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_multi_object_consistency_does_not_count_non_target_read_support(location):
    changed = multi_object_consistency_payload()
    first, second = changed["queries"][0]["target_object_keys"]
    third = {**first, "entity": "e3"}
    changed["target_objects"].append(third)
    changed["version_history"].append({
        "object_key": third,
        "entries": [{
            "version_index": 0, "status": "present", "value": "other",
            "valid_from_event_id": "e0", "logical_time": "000", "source_event_ids": ["e0"],
        }],
    })
    evidence = changed["gold_evidence"][0]
    if location == "primary":
        evidence["supporting_object_keys"].append(third)
        evidence["derivation_steps"] = [
            {"step_id": "first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "third", "operation": "read", "supporting_object_keys": [third], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "third"]},
        ]
    else:
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second, third],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "third", "operation": "read", "supporting_object_keys": [third], "supporting_event_ids": ["e0"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "third"]},
            ],
            "final_derivation_step_id": "equals",
        }
    with pytest.raises(ValueError, match="minimum_objects"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("operand_count", [0, 1])
def test_evidence_evaluator_rejects_vacuous_equals(operand_count):
    from mub.vnext.contracts.v3.task import QueryGoldEvidenceV3
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    task = MemUpdateTaskV3.model_validate(payload())
    key = task.target_objects[0]
    steps = []
    input_ids = []
    if operand_count:
        steps.append({
            "step_id": "read", "operation": "read",
            "supporting_object_keys": [key], "supporting_event_ids": ["e3"],
        })
        input_ids.append("read")
    steps.append({"step_id": "equals", "operation": "equals", "input_step_ids": input_ids})
    evidence = QueryGoldEvidenceV3(
        query_id="vacuous", answer=True, supporting_object_keys=(key,), supporting_event_ids=("e3",),
        derivation_steps=steps, final_derivation_step_id="equals",
    )
    evaluation = evaluate_evidence_v3(evidence, replay_task_v3(task))
    assert evaluation.answer is None
    assert evaluation.issues[0].code == "evidence_replay_error"
    assert "at least two operands" in evaluation.issues[0].message


def test_stale_propagation_uses_registered_alternative_not_any_obsolete_value():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    def score(payload_value, prediction_value):
        task = MemUpdateTaskV3.model_validate(payload_value)
        replay = replay_task_v3(task)
        prediction = AnswerPredictionV3(query_id="q", raw_output=prediction_value, parsed_answer=prediction_value, format_valid=True)
        value, issue = _metric_value(
            "synthesis_scores.stale_propagation_rate", task,
            TaskRunRecordV3(task_id="t", adapter_id="a", run_id="r", answer_predictions=(prediction,), parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed"),
            None, replay, {}, {item.query_id: item for item in task.gold_evidence}, {"q": prediction}, {}, [],
        )
        assert issue is None
        return value

    assert score(g_stale_payload(), "v1") == 1.0
    changed = g_stale_payload()
    alternative = changed["gold_evidence"][0]["stale_alternative"]
    alternative["answer"] = "v0"
    alternative["supporting_event_ids"] = ["e0"]
    alternative["derivation_steps"][0]["supporting_event_ids"] = ["e0"]
    assert score(changed, "v1") == 0.0


def test_every_v3_metric_path_executes_its_dispatch_without_exception():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, MemorySnapshotV3, ParsedManagerActionV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    key = task.target_objects[0]
    parsed_actions = tuple(ParsedManagerActionV3(event_id=action.event_id, operation=action.operation, observed_scope=action.scope, target_object_keys=action.target_object_keys, value=action.value, format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok") for action in task.actions)
    entry = MemoryEntryRecordV3(entry_id="current", content="v2", object_key_candidate=key, value_candidate="v2", version_index=3, source_event_ids=("e3",))
    trace = RetrievalTraceV3(query_id="q", retrieved_entries=(entry,), ranks=(1,), gold_in_context=True, stale_in_context=False, distractor_in_context=False)
    prediction = AnswerPredictionV3(query_id="q", raw_output="ok", parsed_answer=["v0", "v1", None, "v2"], format_valid=True)
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="r", parsed_actions=parsed_actions,
        memory_snapshots=(MemorySnapshotV3(after_event_id="e3", entries=(entry,), state_by_object={key.canonical_id: "v2"}, store_size=1),),
        retrieval_traces=(trace,), answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"), completion_status="completed",
    )
    resolutions = {query.query_id: resolve_query_v3(query, replay, task.events) for query in task.queries}
    evidence = {item.query_id: item for item in task.gold_evidence}
    for path in sorted(CORE_METRIC_FIELD_PATHS):
        result = _metric_value(path, task, run, None, replay, resolutions, evidence, {"q": prediction}, {"q": trace}, _action_facts(task, run))
        assert isinstance(result, tuple) and len(result) == 2, path


def test_each_new_v3_failure_flag_has_a_targeted_corrupted_run():
    from mub.vnext.contracts.v3.enums import FailureFlagV3
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, ParsedManagerActionV3, RetrievalTraceV3
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    base_task = MemUpdateTaskV3.model_validate(payload())
    base_replay = replay_task_v3(base_task)
    base_evidence = {item.query_id: item for item in base_task.gold_evidence}
    provenance = ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1")
    empty = {"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}}
    cases = []

    delete = next(action for action in base_task.actions if action.operation.value == "DELETE")
    wrong_scope_action = ParsedManagerActionV3(event_id=delete.event_id, operation="DELETE", observed_scope="namespace", target_object_keys=delete.target_object_keys, format_valid=True, execution_status="executed", fallback_used=False, raw_output="wrong-scope")
    cases.append(("wrong_delete_scope", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="scope", parsed_actions=(wrong_scope_action,), parser_extractor_provenance=provenance, completion_status="completed"), empty, {}, {}, base_evidence, base_replay))
    add = base_task.actions[0]
    wrong_key = add.target_object_keys[0].model_copy(update={"namespace": "other"})
    wrong_key_action = ParsedManagerActionV3(event_id=add.event_id, operation=add.operation, observed_scope=add.scope, target_object_keys=(wrong_key,), value=add.value, format_valid=True, execution_status="executed", fallback_used=False, raw_output="wrong-key")
    cases.append(("wrong_object_key", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="object-key", parsed_actions=(wrong_key_action,), parser_extractor_provenance=provenance, completion_status="completed"), empty, {}, {}, base_evidence, base_replay))
    cases.append(("collateral_mutation", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="collateral", parser_extractor_provenance=provenance, completion_status="completed"), {"deletion_scores": {"collateral_damage_rate": 1.0}, "historical_scores": {}, "synthesis_scores": {}}, {}, {}, base_evidence, base_replay))
    cases.append(("ttl_violation", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="ttl", parser_extractor_provenance=provenance, completion_status="completed"), {"deletion_scores": {"ttl_compliance_rate": 0.0}, "historical_scores": {}, "synthesis_scores": {}}, {}, {}, base_evidence, base_replay))
    forgotten_entry = MemoryEntryRecordV3(entry_id="forgotten", content="v1", object_key_candidate=base_task.target_objects[0], value_candidate="v1", version_index=1, source_event_ids=("e1",))
    forgotten_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(forgotten_entry,), stale_in_context=True)
    forgotten_run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="forgotten", retrieval_traces=(forgotten_trace,), parser_extractor_provenance=provenance, completion_status="completed")
    cases.append(("forgotten_value_exposed", base_task, forgotten_run, empty, {}, {"q": forgotten_trace}, base_evidence, base_replay))
    cases.append(("version_confusion", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="version", parser_extractor_provenance=provenance, completion_status="completed"), {"deletion_scores": {}, "historical_scores": {"version_confusion_rate": 1.0}, "synthesis_scores": {}}, {}, {}, base_evidence, base_replay))

    g_task = MemUpdateTaskV3.model_validate(g_stale_payload())
    g_replay = replay_task_v3(g_task)
    g_evidence = {item.query_id: item for item in g_task.gold_evidence}
    g_prediction = AnswerPredictionV3(query_id="q", raw_output="v2", parsed_answer="v2", format_valid=True)
    g_run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="evidence", answer_predictions=(g_prediction,), parser_extractor_provenance=provenance, completion_status="completed")
    cases.append(("evidence_linkage_error", g_task, g_run, empty, {"q": g_prediction}, {}, g_evidence, g_replay))
    cases.append(("stale_propagation", g_task, g_run, {"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {"stale_propagation_rate": 1.0}}, {"q": g_prediction}, {}, g_evidence, g_replay))

    activated = set()
    for expected, task, run, layers, predictions, traces, evidence, replay in cases:
        flags = derive_failure_flags_v3(task=task, run=run, replay=replay, layer_values=layers, predictions=predictions, traces=traces, evidence=evidence)
        assert expected in flags, (expected, flags)
        activated.add(expected)
    assert activated == {flag.value for flag in FailureFlagV3}


def test_failure_coverage_matrix_uses_exact_approved_denominator_and_keeps_zero_families():
    from mub.vnext.contracts.common import MetricFieldSupport
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.score import CORE_SCORE_LAYER_TYPES, ScoreRecordV3
    from mub.vnext.scoring.failures_v3 import failure_taxonomy_coverage_v3

    scored_path = "action_scores.full_action_exact_match"
    support = {
        path: MetricFieldSupport(reason=SupportReason.NOT_APPLICABLE, null_policy="test", detail="test fixture")
        for path in CORE_METRIC_FIELD_PATHS if path != scored_path
    }
    layers = {layer: {field: None for field in model.model_fields} for layer, model in CORE_SCORE_LAYER_TYPES.items()}
    layers["action_scores"]["full_action_exact_match"] = 0.0
    common = dict(task_id="coverage", run_id="run", adapter_id="adapter", task_family="F", difficulty="easy", completion_status="completed", supported_metric_fields=support, **layers)
    attributed = ScoreRecordV3.empty(**common, failure_flags=("wrong_operation",))
    system_only = ScoreRecordV3.empty(**{**common, "task_id": "system-only"}, failure_flags=("system_exception",))
    report = failure_taxonomy_coverage_v3((attributed, system_only), families=("E", "F", "G"))
    assert report.overall.numerator == 1
    assert report.overall.denominator == 1
    assert report.overall.coverage == 1.0
    assert tuple(report.by_family) == ("E", "F", "G")
    assert report.by_family["E"].denominator == 0
    assert report.by_family["G"].denominator == 0


def test_snapshot_anchors_reject_duplicates_unknowns_and_mixed_unanchored_rows():
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
    from mub.vnext.scoring.scorer_v3 import _final_snapshot

    task = MemUpdateTaskV3.model_validate(payload())
    key = task.target_objects[0]
    provenance = ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1")
    correct = MemorySnapshotV3(after_event_id="e3", state_by_object={key.canonical_id: "v2"}, store_size=1)
    contradictory = MemorySnapshotV3(after_event_id="e3", state_by_object={}, store_size=0)
    duplicate = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="duplicate", memory_snapshots=(correct, contradictory), parser_extractor_provenance=provenance, completion_status="completed")
    with pytest.raises(ValueError, match="duplicate.*snapshot anchor"):
        _final_snapshot(duplicate, task)

    unknown = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="unknown", memory_snapshots=(MemorySnapshotV3(after_event_id="not-a-task-event", state_by_object={key.canonical_id: "v2"}, store_size=1),), parser_extractor_provenance=provenance, completion_status="completed")
    with pytest.raises(ValueError, match="unknown.*snapshot anchor"):
        _final_snapshot(unknown, task)

    unanchored = MemorySnapshotV3(state_by_object={key.canonical_id: "v2"}, store_size=1)
    single = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="single", memory_snapshots=(unanchored,), parser_extractor_provenance=provenance, completion_status="completed")
    assert _final_snapshot(single, task) == unanchored
    mixed = single.model_copy(update={"memory_snapshots": (correct, unanchored)})
    with pytest.raises(ValueError, match="unanchored.*only snapshot"):
        _final_snapshot(mixed, task)


def test_final_state_corruption_has_canonical_failure_attribution():
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="state-corrupt",
        memory_snapshots=(MemorySnapshotV3(after_event_id="e3", state_by_object={"n|extra|a|": "bad"}, store_size=1),),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    flags = derive_failure_flags_v3(
        task=task, run=run, replay=replay,
        layer_values={"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}},
        predictions={}, traces={}, evidence={item.query_id: item for item in task.gold_evidence},
    )
    assert "current_state_missing" in flags
    assert "collateral_corruption" in flags


def test_namespace_or_subkey_target_mismatch_has_object_key_failure_flag():
    from mub.vnext.contracts.v3.runtime import ParsedManagerActionV3
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    task = MemUpdateTaskV3.model_validate(payload())
    gold = task.actions[0]
    wrong_key = gold.target_object_keys[0].model_copy(update={"namespace": "other", "subkey": "wrong"})
    observed = ParsedManagerActionV3(
        event_id=gold.event_id, operation=gold.operation, observed_scope=gold.scope,
        target_object_keys=(wrong_key,), value=gold.value, format_valid=True,
        execution_status="executed", fallback_used=False, raw_output="wrong-key",
    )
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="wrong-key", parsed_actions=(observed,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    flags = derive_failure_flags_v3(
        task=task, run=run, replay=replay_task_v3(task),
        layer_values={"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}},
        predictions={}, traces={}, evidence={item.query_id: item for item in task.gold_evidence},
    )
    assert "wrong_object_key" in flags


def test_action_facts_reject_runtime_actions_for_unknown_events():
    from mub.vnext.contracts.v3.runtime import ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import _action_facts

    task = MemUpdateTaskV3.model_validate(payload())
    gold = task.actions[0]
    extra = ParsedManagerActionV3(
        event_id="unknown", operation=gold.operation, observed_scope=gold.scope,
        target_object_keys=gold.target_object_keys, value=gold.value, format_valid=True,
        execution_status="executed", fallback_used=False, raw_output="extra",
    )
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="extra-action", parsed_actions=(extra,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    with pytest.raises(ValueError, match="unknown action event"):
        _action_facts(task, run)


def test_answer_state_consistency_denominator_uses_only_applicable_queries():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    changed = payload()
    key = changed["target_objects"][0]
    changed["queries"].append({"query_id": "current", "query_type": "current", "text": "current?", "selector": {"kind": "current"}, "target_object_keys": [key], "answer_schema": "string", "evaluation_mode": "state_direct"})
    changed["gold_evidence"].append({"query_id": "current", "answer": "v2", "supporting_object_keys": [key], "supporting_event_ids": ["e3"], "derivation_steps": [{"step_id": "current-read", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e3"]}], "final_derivation_step_id": "current-read"})
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    predictions = {
        "q": AnswerPredictionV3(query_id="q", raw_output="wrong", parsed_answer=["wrong"], format_valid=True),
        "current": AnswerPredictionV3(query_id="current", raw_output="v2", parsed_answer="v2", format_valid=True),
    }
    resolutions = {query.query_id: resolve_query_v3(query, replay, task.events) for query in task.queries}
    run = TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id="mixed", answer_predictions=tuple(predictions.values()),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    value, detail = _metric_value(
        "answer_scores.answer_state_consistency", task, run, None, replay, resolutions,
        {item.query_id: item for item in task.gold_evidence}, predictions, {}, [],
    )
    assert detail is None
    assert value == 1.0


def test_future_ttl_is_excluded_from_pre_horizon_stale_and_forgotten_status():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.scorer_v3 import _entry_forgotten_status, _entry_obsolete_status

    task = MemUpdateTaskV3.model_validate(ttl_horizon_payload("005", "v1"))
    replay = replay_task_v3(task)
    current = MemoryEntryRecordV3(
        entry_id="current-before-expiry", content="v1",
        object_key_candidate=task.target_objects[0], value_candidate="v1",
        version_index=1, source_event_ids=("e1",),
    )
    assert replay.obsolete_present_values == ("v0",)
    assert _entry_obsolete_status(current, replay) is False
    assert _entry_forgotten_status(current, replay) is False


def test_mixed_version_provenance_uses_unique_value_consistent_version():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.scorer_v3 import _entry_obsolete_status

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    mixed = MemoryEntryRecordV3(
        entry_id="mixed-provenance", content="v0",
        object_key_candidate=task.target_objects[0], value_candidate="v0",
        source_event_ids=("e0", "e3"),
    )
    assert _entry_obsolete_status(mixed, replay) is True
