from copy import deepcopy

import hashlib
import json
import pytest
from pydantic import ValidationError

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


def test_normalized_match_preserves_typed_bool_number_distinction():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(current_structured_payload(True, "boolean"))
    prediction = AnswerPredictionV3(
        query_id="q", raw_output="1", parsed_answer=1, format_valid=True
    )
    run = TaskRunRecordV3(
        task_id="t", adapter_id="adapter", run_id="typed-normalized-match",
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    config = ScorerConfigV3(requested_metric_fields=(
        "answer_scores.exact_match", "answer_scores.normalized_match",
    ))
    score = score_task_v3(
        task, run,
        authenticated_context(
            task, run, info,
            AdapterCapabilitiesV3(supports_native_answer=True), config,
        ),
    )

    assert score.answer_scores.exact_match == 0.0
    assert score.answer_scores.normalized_match == 0.0


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
    from mub.vnext.scoring.lifecycle_v3 import resolve_entry_version_v3

    good = current_structured_payload({"x": [1]}, "object")
    task = MemUpdateTaskV3.model_validate(good)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    entry = MemoryEntryRecordV3(entry_id="nested-corrupt", content="bad", object_key_candidate=task.target_objects[0], value_candidate={"x": [corruption]}, version_index=3, source_event_ids=("e3",))
    assert resolve_entry_version_v3(
        entry, replay.active_versions(replay.ledgers[0])
    ) is None

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


def test_v3_retrieval_registry_applies_only_to_current_query_types():
    from mub.vnext.contracts.v3.enums import QueryTypeV3

    paths = (
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
        "retrieval_scores.stale_exposure_rate",
        "retrieval_scores.stale_count_in_context",
        "retrieval_scores.distractor_exposure_rate",
    )
    expected_query_types = tuple(sorted((QueryTypeV3.CURRENT.value, QueryTypeV3.MULTI_OBJECT_CURRENT.value)))
    for path in paths:
        descriptor = CORE_METRIC_REGISTRY_V3[path]
        assert descriptor.applicable_task_families == ("*",)
        assert descriptor.applicable_query_kinds == expected_query_types


def test_v3_historical_diagnostic_registry_excludes_current_queries():
    expected_query_types = (
        "ordered_history", "point_in_time", "previous", "transition",
    )
    paths = (
        "historical_scores.version_confusion_rate",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    )

    for path in paths:
        assert CORE_METRIC_REGISTRY_V3[path].applicable_query_kinds == expected_query_types


def test_current_only_f_failed_run_marks_historical_diagnostics_not_applicable_first():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    paths = (
        "historical_scores.version_confusion_rate",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    )
    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id="current-only-failed",
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="failed",
        exceptions=({"type": "boom"},),
    )
    info = AdapterInfoV3(
        adapter_id="adapter",
        adapter_version="1",
        system_name="system",
        system_version="1",
        configuration_hash=H,
    )
    config = ScorerConfigV3(requested_metric_fields=paths)

    score = score_task_v3(
        task,
        run,
        authenticated_context(task, run, info, AdapterCapabilitiesV3(), config),
    )

    for path in paths:
        layer, leaf = path.split(".", 1)
        assert getattr(getattr(score, layer), leaf) is None
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE


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


def test_valid_ttl_delete_replays_at_its_effective_logical_time() -> None:
    changed = payload()
    changed["actions"][2]["scope"] = "ttl"
    changed["version_history"][0]["entries"][1]["valid_until_event_id"] = None
    changed["version_history"][0]["entries"][2]["valid_from_event_id"] = None

    task = MemUpdateTaskV3.model_validate(changed)
    ttl_action = task.actions[2]
    replay = replay_task_v3(task)

    assert ttl_action.effective_at == "002"
    assert replay.issues == ()
    assert replay.ledgers[0].versions[2].logical_time == ttl_action.effective_at
    assert replay.ledgers[0].versions[2].valid_from_event_id is None


@pytest.mark.parametrize("effective_at", ["", " ", "\t\n"])
def test_full_task_rejects_blank_ttl_effective_at(effective_at: str) -> None:
    changed = payload()
    changed["actions"][2]["scope"] = "ttl"
    changed["actions"][2]["effective_at"] = effective_at

    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        MemUpdateTaskV3.model_validate(changed)


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
            action_id=action.action_id,
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

    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
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


def test_mixed_historical_trace_does_not_affect_current_retrieval_metrics():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    changed = payload()
    changed["actions"][0]["value"] = "v2"
    changed["version_history"][0]["entries"][0]["value"] = "v2"
    changed["gold_evidence"][0]["answer"][0] = "v2"
    key = changed["target_objects"][0]
    changed["queries"].append({
        "query_id": "q-current", "query_type": "current", "text": "?",
        "selector": {"kind": "current"}, "target_object_keys": [key],
        "answer_schema": "string", "evaluation_mode": "state_direct",
    })
    changed["gold_evidence"].append({
        "query_id": "q-current", "answer": "v2", "supporting_object_keys": [key],
        "supporting_event_ids": ["e3"],
        "derivation_steps": [{
            "step_id": "read-current", "operation": "read",
            "supporting_object_keys": [key], "supporting_event_ids": ["e3"],
        }],
        "final_derivation_step_id": "read-current",
    })
    task = MemUpdateTaskV3.model_validate(changed)
    object_key = task.target_objects[0]
    current_trace = RetrievalTraceV3(
        query_id="q-current",
        retrieved_entries=(MemoryEntryRecordV3(
            entry_id="current", content="v2", object_key_candidate=object_key,
            value_candidate="v2", version_index=3, source_event_ids=("e3",),
        ),),
        ranks=(7,),
        stale_in_context=False,
        distractor_in_context=False,
    )
    historical_trace = RetrievalTraceV3(
        query_id="q",
        retrieved_entries=(MemoryEntryRecordV3(
            entry_id="ambiguous-history", content="v2",
            object_key_candidate=object_key, value_candidate="v2",
        ),),
        ranks=(1,),
        stale_in_context=True,
        distractor_in_context=True,
    )
    predictions = tuple(
        AnswerPredictionV3(
            query_id=evidence.query_id,
            raw_output=str(evidence.answer),
            parsed_answer=evidence.answer,
            format_valid=True,
        )
        for evidence in task.gold_evidence
    )
    run = TaskRunRecordV3(
        task_id=task.task_id, adapter_id="adapter", run_id="mixed-retrieval",
        retrieval_traces=(current_trace, historical_trace),
        answer_predictions=predictions,
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_entries=True, exports_object_keys=True, exports_values=True,
        exports_retrieval_ids=True, exports_retrieval_scores=True,
    )
    paths = tuple(
        f"retrieval_scores.{leaf}"
        for leaf in (
            "current_recall_at_k", "current_mrr", "stale_exposure_rate",
            "stale_count_in_context", "distractor_exposure_rate",
        )
    )
    config = ScorerConfigV3(requested_metric_fields=paths)

    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))

    assert score.retrieval_scores.current_recall_at_k == 1.0
    assert score.retrieval_scores.current_mrr == pytest.approx(1 / 7)
    assert score.retrieval_scores.stale_exposure_rate == 0.0
    assert score.retrieval_scores.stale_count_in_context == 0
    assert score.retrieval_scores.distractor_exposure_rate == 0.0


@pytest.mark.parametrize(
    ("case", "missing_query_id"),
    (("historical_only_trace", "q-current"), ("one_of_two_current_traces", "q-current-2")),
)
def test_current_retrieval_metrics_require_exact_trace_coverage(case, missing_query_id):
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    if case == "historical_only_trace":
        changed = payload()
        key = changed["target_objects"][0]
        changed["queries"].append({
            "query_id": missing_query_id, "query_type": "current", "text": "?",
            "selector": {"kind": "current"}, "target_object_keys": [key],
            "answer_schema": "string", "evaluation_mode": "state_direct",
        })
        changed["gold_evidence"].append({
            "query_id": missing_query_id, "answer": "v2", "supporting_object_keys": [key],
            "supporting_event_ids": ["e3"],
            "derivation_steps": [{
                "step_id": "read-current", "operation": "read",
                "supporting_object_keys": [key], "supporting_event_ids": ["e3"],
            }],
            "final_derivation_step_id": "read-current",
        })
        traced_query_id, traced_value, traced_version, traced_event = "q", "v0", 0, "e0"
    else:
        changed = replayable_multi_object_consistency_payload("e1")
        changed["task_family"] = "F"
        first, second = changed["target_objects"]
        changed["queries"] = [
            {
                "query_id": "q", "query_type": "current", "text": "?",
                "selector": {"kind": "current"}, "target_object_keys": [first],
                "answer_schema": "string", "evaluation_mode": "state_direct",
            },
            {
                "query_id": missing_query_id, "query_type": "current", "text": "?",
                "selector": {"kind": "current"}, "target_object_keys": [second],
                "answer_schema": "string", "evaluation_mode": "state_direct",
            },
        ]
        changed["gold_evidence"] = [
            {
                "query_id": "q", "answer": "v2", "supporting_object_keys": [first],
                "supporting_event_ids": ["e3"],
                "derivation_steps": [{
                    "step_id": "read-first", "operation": "read",
                    "supporting_object_keys": [first], "supporting_event_ids": ["e3"],
                }],
                "final_derivation_step_id": "read-first",
            },
            {
                "query_id": missing_query_id, "answer": "z", "supporting_object_keys": [second],
                "supporting_event_ids": ["e1"],
                "derivation_steps": [{
                    "step_id": "read-second", "operation": "read",
                    "supporting_object_keys": [second], "supporting_event_ids": ["e1"],
                }],
                "final_derivation_step_id": "read-second",
            },
        ]
        traced_query_id, traced_value, traced_version, traced_event = "q", "v2", 3, "e3"

    task = MemUpdateTaskV3.model_validate(changed)
    trace = RetrievalTraceV3(
        query_id=traced_query_id,
        retrieved_entries=(MemoryEntryRecordV3(
            entry_id="only-supplied-trace", content=traced_value,
            object_key_candidate=task.target_objects[0], value_candidate=traced_value,
            version_index=traced_version, source_event_ids=(traced_event,),
        ),),
        ranks=(1,),
    )
    predictions = tuple(
        AnswerPredictionV3(
            query_id=evidence.query_id, raw_output=str(evidence.answer),
            parsed_answer=evidence.answer, format_valid=True,
        )
        for evidence in task.gold_evidence
    )
    run = TaskRunRecordV3(
        task_id=task.task_id, adapter_id="adapter", run_id=f"coverage-{case}",
        retrieval_traces=(trace,), answer_predictions=predictions,
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_entries=True, exports_object_keys=True, exports_values=True,
        exports_retrieval_ids=True, exports_retrieval_scores=True,
    )
    paths = tuple(
        f"retrieval_scores.{leaf}"
        for leaf in (
            "current_recall_at_k", "current_mrr", "stale_exposure_rate",
            "stale_count_in_context", "distractor_exposure_rate",
        )
    )
    config = ScorerConfigV3(requested_metric_fields=paths)

    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))

    for path in paths:
        layer, leaf = path.split(".", 1)
        assert getattr(getattr(score, layer), leaf) is None
        support = score.supported_metric_fields[path]
        assert support.reason is SupportReason.MISSING_ARTIFACT
        assert missing_query_id in support.detail


def _retrieval_annotation_metric(
    leaf,
    entry_kind,
    *,
    gold_in_context=None,
    stale_in_context=None,
    ranks=(),
):
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    changed = current_structured_payload("x", "string")
    changed["actions"][0]["value"] = "x"
    changed["version_history"][0]["entries"][0]["value"] = "x"
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    key = task.target_objects[0]
    entry_specs = {
        "empty": (),
        "current": (("current", "x", 3, "e3"),),
        "stale": (("stale", "v1", 1, "e1"),),
        "two_stale": (
            ("stale-0", "v1", 1, "e1"),
            ("stale-1", "v1", 1, "e1"),
        ),
        "ambiguous": (("ambiguous", "x", None, None),),
    }
    entries = tuple(
        MemoryEntryRecordV3(
            entry_id=entry_id,
            content=value,
            object_key_candidate=key,
            value_candidate=value,
            version_index=version_index,
            source_event_ids=() if event_id is None else (event_id,),
        )
        for entry_id, value, version_index, event_id in entry_specs[entry_kind]
    )
    trace = RetrievalTraceV3(
        query_id="q",
        retrieved_entries=entries,
        ranks=ranks,
        gold_in_context=gold_in_context,
        stale_in_context=stale_in_context,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="a",
        run_id="retrieval-annotation",
        retrieval_traces=(trace,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    return _metric_value(
        f"retrieval_scores.{leaf}",
        task,
        run,
        None,
        replay,
        {"q": resolve_query_v3(task.queries[0], replay, task.events)},
        {"q": task.gold_evidence[0]},
        {},
        {"q": trace},
        [],
    )


@pytest.mark.parametrize(
    ("gold_in_context", "entry_kind", "expected"),
    [
        (True, "empty", 1.0),
        (True, "current", 1.0),
        (True, "ambiguous", 1.0),
        (False, "empty", 0.0),
        (False, "current", 0.0),
        (False, "ambiguous", 0.0),
    ],
)
def test_current_recall_annotation_overrides_detailed_entries(
    gold_in_context, entry_kind, expected,
):
    value, detail = _retrieval_annotation_metric(
        "current_recall_at_k",
        entry_kind,
        gold_in_context=gold_in_context,
    )

    assert detail is None
    assert value == expected


@pytest.mark.parametrize(
    ("entry_kind", "expected", "missing_artifact"),
    [
        ("empty", 0.0, False),
        ("current", 1.0, False),
        ("stale", 0.0, False),
        ("ambiguous", None, True),
    ],
)
def test_current_recall_null_annotation_falls_back_to_detailed_entries(
    entry_kind, expected, missing_artifact,
):
    value, detail = _retrieval_annotation_metric(
        "current_recall_at_k", entry_kind, gold_in_context=None,
    )

    assert value == expected
    assert (detail is not None) is missing_artifact


def _score_effective_current_retrieval(
    entry_kind,
    *,
    gold_in_context=None,
    stale_in_context=None,
    wrong_answer=True,
    format_valid=True,
):
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    changed = current_structured_payload("x", "string")
    changed["actions"][0]["value"] = "x"
    changed["version_history"][0]["entries"][0]["value"] = "x"
    task = MemUpdateTaskV3.model_validate(changed)
    key = task.target_objects[0]
    entry_specs = {
        "empty": (),
        "current": (("current", "x", 3, "e3"),),
        "stale": (("stale", "v1", 1, "e1"),),
        "ambiguous": (("ambiguous", "x", None, None),),
    }
    entries = tuple(
        MemoryEntryRecordV3(
            entry_id=entry_id,
            content=value,
            object_key_candidate=key,
            value_candidate=value,
            version_index=version_index,
            source_event_ids=() if event_id is None else (event_id,),
        )
        for entry_id, value, version_index, event_id in entry_specs[entry_kind]
    )
    trace = RetrievalTraceV3(
        query_id="q",
        retrieved_entries=entries,
        gold_in_context=gold_in_context,
        stale_in_context=stale_in_context,
    )
    answer = "wrong" if wrong_answer else "x"
    prediction = AnswerPredictionV3(
        query_id="q", raw_output=answer, parsed_answer=answer, format_valid=format_valid,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id=f"effective-{entry_kind}",
        retrieval_traces=(trace,),
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
        exports_retrieval_ids=True,
    )
    config = ScorerConfigV3(requested_metric_fields=(
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.stale_exposure_rate",
        "answer_scores.gold_retrieved_wrong_answer",
    ))
    return score_task_v3(
        task, run, authenticated_context(task, run, info, caps, config),
    )


@pytest.mark.parametrize(
    ("gold_in_context", "entry_kind", "expected_recall", "expected_wrong", "expected_flags", "missing"),
    [
        (False, "current", 0.0, 0.0, {"current_not_retrieved"}, False),
        (True, "empty", 1.0, 1.0, {"gold_retrieved_wrong_answer"}, False),
        (None, "current", 1.0, 1.0, {"gold_retrieved_wrong_answer"}, False),
        (None, "ambiguous", None, None, set(), True),
    ],
)
def test_effective_gold_status_keeps_recall_answer_metric_and_flags_in_parity(
    gold_in_context, entry_kind, expected_recall, expected_wrong, expected_flags, missing,
):
    from mub.vnext.contracts.enums import SupportReason

    score = _score_effective_current_retrieval(
        entry_kind, gold_in_context=gold_in_context,
    )

    assert score.retrieval_scores.current_recall_at_k == expected_recall
    assert score.answer_scores.gold_retrieved_wrong_answer == expected_wrong
    assert ({"current_not_retrieved", "gold_retrieved_wrong_answer"} & set(score.failure_flags)) == expected_flags
    for path in (
        "retrieval_scores.current_recall_at_k",
        "answer_scores.gold_retrieved_wrong_answer",
    ):
        if missing:
            assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT
        else:
            assert path not in score.supported_metric_fields


def test_effective_gold_with_format_only_error_is_not_wrong_answer_attribution():
    score = _score_effective_current_retrieval(
        "empty",
        gold_in_context=True,
        wrong_answer=False,
        format_valid=False,
    )

    assert score.answer_scores.gold_retrieved_wrong_answer == 0.0
    assert "answer_format_only" in score.failure_flags
    assert "gold_retrieved_wrong_answer" not in score.failure_flags


@pytest.mark.parametrize(
    ("stale_in_context", "entry_kind", "expected_exposure", "expected_flag", "missing"),
    [
        (False, "stale", 0.0, False, False),
        (True, "empty", 1.0, True, False),
        (None, "stale", 1.0, True, False),
        (None, "ambiguous", None, False, True),
    ],
)
def test_effective_stale_status_keeps_exposure_metric_and_flag_in_parity(
    stale_in_context, entry_kind, expected_exposure, expected_flag, missing,
):
    from mub.vnext.contracts.enums import SupportReason

    score = _score_effective_current_retrieval(
        entry_kind,
        gold_in_context=True,
        stale_in_context=stale_in_context,
        wrong_answer=False,
    )

    assert score.retrieval_scores.stale_exposure_rate == expected_exposure
    assert ("stale_retrieved" in score.failure_flags) is expected_flag
    support_path = "retrieval_scores.stale_exposure_rate"
    if missing:
        assert score.supported_metric_fields[support_path].reason is SupportReason.MISSING_ARTIFACT
    else:
        assert support_path not in score.supported_metric_fields


@pytest.mark.parametrize(
    ("stale_in_context", "entry_kind", "expected"),
    [
        (True, "current", 1.0),
        (True, "ambiguous", 1.0),
        (False, "stale", 0.0),
        (False, "ambiguous", 0.0),
    ],
)
def test_stale_exposure_annotation_overrides_detailed_entries(
    stale_in_context, entry_kind, expected,
):
    value, detail = _retrieval_annotation_metric(
        "stale_exposure_rate",
        entry_kind,
        stale_in_context=stale_in_context,
    )

    assert detail is None
    assert value == expected


@pytest.mark.parametrize(
    ("entry_kind", "expected", "missing_artifact"),
    [
        ("empty", 0.0, False),
        ("current", 0.0, False),
        ("stale", 1.0, False),
        ("ambiguous", None, True),
    ],
)
def test_stale_exposure_null_annotation_falls_back_to_detailed_entries(
    entry_kind, expected, missing_artifact,
):
    value, detail = _retrieval_annotation_metric(
        "stale_exposure_rate", entry_kind, stale_in_context=None,
    )

    assert value == expected
    assert (detail is not None) is missing_artifact


@pytest.mark.parametrize(
    ("stale_in_context", "entry_kind", "expected", "missing_artifact"),
    [
        (False, "ambiguous", 0, False),
        (True, "empty", None, True),
        (True, "current", None, True),
        (True, "ambiguous", None, True),
        (True, "stale", 1, False),
        (None, "two_stale", 2, False),
        (None, "empty", 0, False),
        (None, "ambiguous", None, True),
    ],
)
def test_stale_count_annotation_controls_exact_multiplicity_artifacts(
    stale_in_context, entry_kind, expected, missing_artifact,
):
    value, detail = _retrieval_annotation_metric(
        "stale_count_in_context",
        entry_kind,
        stale_in_context=stale_in_context,
    )

    assert value == expected
    assert (detail is not None) is missing_artifact


@pytest.mark.parametrize(
    ("entry_kind", "expected", "missing_artifact"),
    [
        ("empty", 0.0, False),
        ("current", None, True),
    ],
)
def test_current_mrr_does_not_infer_rank_from_positive_gold_annotation(
    entry_kind, expected, missing_artifact,
):
    value, detail = _retrieval_annotation_metric(
        "current_mrr",
        entry_kind,
        gold_in_context=True,
    )

    assert value == expected
    assert (detail is not None) is missing_artifact


def _current_mrr_fixture(*entries, ranks=()):
    from mub.vnext.contracts.v3.runtime import RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    replay = replay_task_v3(task)
    query = task.queries[0]
    trace = RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=entries,
        ranks=ranks,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="a",
        run_id="mrr",
        retrieval_traces=(trace,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    value, detail = _metric_value(
        "retrieval_scores.current_mrr",
        task,
        run,
        None,
        replay,
        {query.query_id: resolve_query_v3(query, replay, task.events)},
        {item.query_id: item for item in task.gold_evidence},
        {},
        {query.query_id: trace},
        [],
    )
    return task, trace, run, value, detail


def _current_mrr_entry(task, entry_id, value, version_index, event_id):
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    return MemoryEntryRecordV3(
        entry_id=entry_id,
        content=str(value),
        object_key_candidate=task.target_objects[0],
        value_candidate=value,
        version_index=version_index,
        source_event_ids=(event_id,),
    )


def test_current_mrr_uses_supplied_rank_for_first_current_entry():
    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    current = _current_mrr_entry(task, "current", "v2", 3, "e3")

    _, _, _, value, detail = _current_mrr_fixture(current, ranks=(7,))

    assert detail is None
    assert value == pytest.approx(1 / 7)


def test_current_mrr_uses_minimum_supplied_rank_across_matching_entries():
    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    current_a = _current_mrr_entry(task, "current-a", "v2", 3, "e3")
    current_b = _current_mrr_entry(task, "current-b", "v2", 3, "e3")

    _, _, _, value, detail = _current_mrr_fixture(
        current_a, current_b, ranks=(9, 3),
    )

    assert detail is None
    assert value == pytest.approx(1 / 3)


def test_current_mrr_missing_ranks_is_missing_artifact():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    current = _current_mrr_entry(task, "current", "v2", 3, "e3")
    task, trace, run, _, _ = _current_mrr_fixture(current)
    run = run.model_copy(
        update={
            "answer_predictions": (
                AnswerPredictionV3(
                    query_id="q",
                    raw_output="v2",
                    parsed_answer="v2",
                    format_valid=True,
                ),
            ),
        },
    )
    info = AdapterInfoV3(
        adapter_id="a",
        adapter_version="1",
        system_name="test",
        system_version="1",
        configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_retrieval_ids=True,
        exports_retrieval_scores=True,
    )
    config = ScorerConfigV3(
        requested_metric_fields=("retrieval_scores.current_mrr",),
    )

    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))

    assert trace.retrieved_entries
    assert trace.ranks == ()
    assert score.retrieval_scores.current_mrr is None
    assert score.supported_metric_fields[
        "retrieval_scores.current_mrr"
    ].reason is SupportReason.MISSING_ARTIFACT


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize("bad_rank", [1.0, True, "1", 0, -1])
def test_current_mrr_scorer_rejects_non_exact_positive_rank(bad_rank):
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    current = _current_mrr_entry(task, "current", "v2", 3, "e3")
    task, trace, run, _, _ = _current_mrr_fixture(current, ranks=(1,))
    run = run.model_copy(
        update={
            "answer_predictions": (
                AnswerPredictionV3(
                    query_id="q",
                    raw_output="v2",
                    parsed_answer="v2",
                    format_valid=True,
                ),
            ),
        },
    )
    info = AdapterInfoV3(
        adapter_id="a",
        adapter_version="1",
        system_name="test",
        system_version="1",
        configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_retrieval_ids=True,
        exports_retrieval_scores=True,
    )
    config = ScorerConfigV3(
        requested_metric_fields=("retrieval_scores.current_mrr",),
    )
    context = authenticated_context(task, run, info, caps, config)
    corrupted_trace = RetrievalTraceV3.model_construct(
        **{**trace.__dict__, "ranks": (bad_rank,)}
    )
    corrupted_run = TaskRunRecordV3.model_construct(
        **{**run.__dict__, "retrieval_traces": (corrupted_trace,)}
    )

    with pytest.raises(ValidationError):
        score_task_v3(task, corrupted_run, context)


def test_current_mrr_uses_rank_when_tuple_order_disagrees():
    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    stale = _current_mrr_entry(task, "stale", "v1", 1, "e1")
    current = _current_mrr_entry(task, "current", "v2", 3, "e3")

    _, _, _, value, detail = _current_mrr_fixture(stale, current, ranks=(2, 9))

    assert detail is None
    assert value == pytest.approx(1 / 9)


def test_current_mrr_fails_closed_on_value_inconsistent_explicit_evidence():
    task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    wrong_value = _current_mrr_entry(task, "wrong", "not-v2", 3, "e3")
    stale = _current_mrr_entry(task, "stale", "v1", 1, "e1")
    current = _current_mrr_entry(task, "current", "v2", 3, "e3")

    _, _, _, value, detail = _current_mrr_fixture(
        wrong_value, stale, current, ranks=(1, 2, 11),
    )

    assert value is None
    assert "version identity is ambiguous" in detail


def test_same_value_different_version_is_not_current_entry_match():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.lifecycle_v3 import resolve_entry_version_v3

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
    resolved_stale = resolve_entry_version_v3(
        stale_v0, replay.active_versions(replay.ledgers[0])
    )
    assert resolved_stale is not None
    assert resolved_stale[1].version_index == 0
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
    assert resolve_entry_version_v3(
        wrong_value_v3, current_replay.active_versions(current_replay.ledgers[0])
    ) is None
    wrong_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(wrong_value_v3,))
    value, detail = _metric_value("retrieval_scores.current_recall_at_k", current_task, run.model_copy(update={"retrieval_traces": (wrong_trace,)}), None, current_replay, {"q": resolve_query_v3(current_task.queries[0], current_replay, current_task.events)}, {"q": current_task.gold_evidence[0]}, {}, {"q": wrong_trace}, [])
    assert value is None
    assert "version identity" in detail

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
    current_evidence = {item.query_id: item for item in current_task.gold_evidence}
    layers = {"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}}
    current_v3 = MemoryEntryRecordV3(entry_id="v3", content="x", object_key_candidate=current_task.target_objects[0], value_candidate="x", version_index=3, source_event_ids=("e3",))
    current_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(current_v3,), stale_in_context=False)
    current_run = TaskRunRecordV3(task_id="t", adapter_id="a", run_id="current", retrieval_traces=(current_trace,), parser_extractor_provenance=run.parser_extractor_provenance, completion_status="completed")
    flags = derive_failure_flags_v3(task=current_task, run=current_run, replay=current_replay, layer_values=layers, predictions={}, traces={"q": current_trace}, evidence=current_evidence)
    assert "stale_retrieved" not in flags
    stale_trace = RetrievalTraceV3(query_id="q", retrieved_entries=(stale_v0,), stale_in_context=True)
    stale_run = current_run.model_copy(update={"retrieval_traces": (stale_trace,)})
    flags = derive_failure_flags_v3(task=current_task, run=stale_run, replay=current_replay, layer_values=layers, predictions={}, traces={"q": stale_trace}, evidence=current_evidence)
    assert "stale_retrieved" in flags
    from mub.vnext.scoring.scorer_v3 import _action_facts
    value, detail = _metric_value(
        "deletion_scores.forgotten_exposure_rate", task, stale_run, None,
        replay, {}, evidence, {}, {"q": stale_trace}, _action_facts(task, stale_run),
    )
    assert detail is None
    assert value == 0.0
    explicit_run = run.model_copy(update={"memory_snapshots": (MemorySnapshotV3(after_event_id="e3", entries=(stale_v0,), store_size=1),)})
    obsolete, detail = _metric_value("store_scores.obsolete_version_count", task, explicit_run, None, replay, {}, evidence, {}, {}, _action_facts(task, explicit_run))
    assert detail is None
    assert obsolete == 1
    conflict, detail = _metric_value("store_scores.stale_conflicting_value_count", task, explicit_run, None, replay, {}, evidence, {}, {}, _action_facts(task, explicit_run))
    assert detail is None
    assert conflict == 0


def _two_version_current_task():
    changed = payload()
    changed["events"] = changed["events"][:2]
    changed["actions"] = changed["actions"][:2]
    changed["actions"][0]["value"] = "old"
    changed["actions"][1]["value"] = "new"
    changed["version_history"][0]["entries"] = changed["version_history"][0]["entries"][:2]
    changed["version_history"][0]["entries"][0]["value"] = "old"
    changed["version_history"][0]["entries"][1]["value"] = "new"
    changed["version_history"][0]["entries"][1].pop("valid_until_event_id", None)
    changed["queries"][0] = {
        "query_id": "q", "query_type": "current", "text": "?",
        "selector": {"kind": "current"},
        "target_object_keys": changed["target_objects"],
        "answer_schema": "string", "evaluation_mode": "state_direct",
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": "new",
        "supporting_object_keys": changed["target_objects"],
        "supporting_event_ids": ["e1"],
        "derivation_steps": [{
            "step_id": "read", "operation": "read",
            "supporting_object_keys": changed["target_objects"],
            "supporting_event_ids": ["e1"],
        }],
        "final_derivation_step_id": "read",
    }
    return MemUpdateTaskV3.model_validate(changed)


def _score_entry_attribution(
    task, entry, paths, *, retrieve=False, extra_entries=(),
):
    from mub.vnext.contracts.v3.runtime import (
        AnswerPredictionV3, MemorySnapshotV3, RetrievalTraceV3,
    )
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    entries = (entry, *extra_entries)
    trace = RetrievalTraceV3(
        query_id="q", retrieved_entries=entries, ranks=tuple(range(1, len(entries) + 1)),
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id=f"entry-attribution-{'retrieval' if retrieve else 'store'}",
        memory_snapshots=() if retrieve else (
            MemorySnapshotV3(
                after_event_id="e1", entries=entries, store_size=len(entries),
            ),
        ),
        retrieval_traces=(trace,) if retrieve else (),
        answer_predictions=(AnswerPredictionV3(
            query_id="q", raw_output="new", parsed_answer="new",
            format_valid=True,
        ),),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        supports_isolated_reset=True,
        exports_version_history=True,
        exports_entries=True,
        exports_source_event_ids=True,
        exports_timestamps_or_order=True,
        exports_object_keys=True,
        exports_values=True,
        exports_retrieval_ids=True,
        exports_retrieval_scores=True,
    )
    config = ScorerConfigV3(requested_metric_fields=paths)
    return score_task_v3(
        task, run, authenticated_context(task, run, info, caps, config),
    )


def test_store_metrics_fail_closed_on_conflicting_version_and_source_evidence():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    task = _two_version_current_task()
    entry = MemoryEntryRecordV3(
        entry_id="conflicting-attribution", content="old",
        object_key_candidate=task.target_objects[0], value_candidate="old",
        version_index=0, source_event_ids=("e1",),
    )
    paths = (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    )

    score = _score_entry_attribution(task, entry, paths)

    for path in paths:
        assert getattr(score.store_scores, path.rsplit(".", 1)[1]) is None
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_retrieval_metrics_fail_closed_on_split_source_provenance():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    task = _two_version_current_task()
    entry = MemoryEntryRecordV3(
        entry_id="split-provenance", content="new",
        object_key_candidate=task.target_objects[0], value_candidate="new",
        source_event_ids=("e0", "e1"),
    )
    paths = (
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
    )

    score = _score_entry_attribution(task, entry, paths, retrieve=True)

    for path in paths:
        assert getattr(score.retrieval_scores, path.rsplit(".", 1)[1]) is None
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_store_metrics_fail_closed_on_incomplete_present_value_evidence():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    task = _two_version_current_task()
    entry = MemoryEntryRecordV3(
        entry_id="incomplete-present", content="unknown",
        object_key_candidate=task.target_objects[0], value_candidate=None,
    )
    paths = (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
        "store_scores.duplicate_current_count",
    )

    score = _score_entry_attribution(task, entry, paths)

    for path in paths:
        assert getattr(score.store_scores, path.rsplit(".", 1)[1]) is None
        assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def test_duplicate_current_count_fails_closed_on_conflicting_status_provenance():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    task = _two_version_current_task()
    current = MemoryEntryRecordV3(
        entry_id="current", content="new",
        object_key_candidate=task.target_objects[0], value_candidate="new",
        version_index=1, source_event_ids=("e1",),
    )
    conflicting = current.model_copy(update={
        "entry_id": "status-conflict",
        "raw_metadata": {"status": "tombstone"},
    })
    path = "store_scores.duplicate_current_count"

    score = _score_entry_attribution(
        task, current, (path,), extra_entries=(conflicting,),
    )

    assert score.store_scores.duplicate_current_count is None
    assert score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT


def _store_metric_value(path, task, replay, *entries):
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3
    from mub.vnext.scoring.scorer_v3 import _metric_value

    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="a",
        run_id="store-metric",
        memory_snapshots=(
            MemorySnapshotV3(
                after_event_id=task.events[-1].event_id,
                entries=entries,
                store_size=len(entries),
            ),
        ),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    return _metric_value(path, task, run, None, replay, {}, {}, {}, {}, [])


def test_store_stale_conflict_counts_typed_different_obsolete_value():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    stale = MemoryEntryRecordV3(
        entry_id="stale-different",
        content="v0",
        object_key_candidate=task.target_objects[0],
        value_candidate="v0",
        version_index=0,
        source_event_ids=("e0",),
    )
    obsolete, obsolete_detail = _store_metric_value(
        "store_scores.obsolete_version_count", task, replay, stale,
    )
    conflict, conflict_detail = _store_metric_value(
        "store_scores.stale_conflicting_value_count", task, replay, stale,
    )
    assert (obsolete, obsolete_detail) == (1, None)
    assert (conflict, conflict_detail) == (1, None)


def test_store_stale_conflict_counts_retained_present_value_against_current_tombstone():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    changed = payload()
    changed["events"] = changed["events"][:3]
    changed["actions"] = changed["actions"][:3]
    changed["version_history"][0]["entries"] = changed["version_history"][0]["entries"][:3]
    changed["version_history"][0]["entries"][-1].pop("valid_until_event_id", None)
    changed["gold_evidence"][0]["answer"] = ["v0", "v1", None]
    changed["gold_evidence"][0]["supporting_event_ids"] = ["e0", "e1", "e2"]
    changed["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = ["e0", "e1", "e2"]
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    stale = MemoryEntryRecordV3(
        entry_id="stale-before-tombstone",
        content="v1",
        object_key_candidate=task.target_objects[0],
        value_candidate="v1",
        version_index=1,
        source_event_ids=("e1",),
    )
    obsolete, obsolete_detail = _store_metric_value(
        "store_scores.obsolete_version_count", task, replay, stale,
    )
    conflict, conflict_detail = _store_metric_value(
        "store_scores.stale_conflicting_value_count", task, replay, stale,
    )
    assert (obsolete, obsolete_detail) == (1, None)
    assert (conflict, conflict_detail) == (1, None)


@pytest.mark.parametrize(
    ("current_value", "expected_conflict"),
    [
        ({"items": [True, {"n": 1}]}, 0),
        ({"items": [1, {"n": 1}]}, 1),
    ],
)
def test_store_stale_conflict_uses_structured_typed_equality(
    current_value, expected_conflict,
):
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    stale_value = {"items": [True, {"n": 1}]}
    changed = current_structured_payload(current_value, "object")
    changed["actions"][0]["value"] = stale_value
    changed["version_history"][0]["entries"][0]["value"] = stale_value
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    stale = MemoryEntryRecordV3(
        entry_id="stale-structured",
        content="structured",
        object_key_candidate=task.target_objects[0],
        value_candidate=stale_value,
        version_index=0,
        source_event_ids=("e0",),
    )
    obsolete, obsolete_detail = _store_metric_value(
        "store_scores.obsolete_version_count", task, replay, stale,
    )
    conflict, conflict_detail = _store_metric_value(
        "store_scores.stale_conflicting_value_count", task, replay, stale,
    )
    assert (obsolete, obsolete_detail) == (1, None)
    assert (conflict, conflict_detail) == (expected_conflict, None)


def test_store_stale_conflict_uses_horizon_active_current_version():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    changed = ttl_horizon_payload("005", "v1")
    changed["actions"][0]["value"] = "v1"
    changed["version_history"][0]["entries"][0]["value"] = "v1"
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    stale = MemoryEntryRecordV3(
        entry_id="stale-before-future-tombstone",
        content="v1",
        object_key_candidate=task.target_objects[0],
        value_candidate="v1",
        version_index=0,
        source_event_ids=("e0",),
    )
    obsolete, obsolete_detail = _store_metric_value(
        "store_scores.obsolete_version_count", task, replay, stale,
    )
    conflict, conflict_detail = _store_metric_value(
        "store_scores.stale_conflicting_value_count", task, replay, stale,
    )
    assert (obsolete, obsolete_detail) == (1, None)
    assert (conflict, conflict_detail) == (0, None)


def test_store_stale_conflict_preserves_repeated_value_ambiguity_null():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

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
    ambiguous = MemoryEntryRecordV3(
        entry_id="ambiguous-repeat",
        content="x",
        object_key_candidate=task.target_objects[0],
        value_candidate="x",
    )
    for path in (
        "store_scores.obsolete_version_count",
        "store_scores.stale_conflicting_value_count",
    ):
        value, detail = _store_metric_value(path, task, replay, ambiguous)
        assert value is None
        assert "version identity is ambiguous" in detail


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


def horizon_exact_transition_payload(selector):
    changed = ttl_horizon_payload("002", "v1")
    changed["events"][1]["timestamp"] = "002"
    changed["actions"][1]["effective_at"] = "002"
    changed["version_history"][0]["entries"][1]["logical_time"] = "002"
    is_transition = selector["kind"] == "transition"
    selected_indices = (
        (selector["from_version_index"], selector["to_version_index"])
        if is_transition
        else (selector["version_index"],)
    )
    selected_entries = [changed["version_history"][0]["entries"][index] for index in selected_indices]
    changed["queries"][0].update(
        query_type="transition" if is_transition else "point_in_time",
        selector=selector,
        answer_schema="object" if is_transition else ("list" if selected_entries[0]["status"] == "tombstone" else "string"),
    )
    if is_transition:
        answer = {"from": selected_entries[0].get("value"), "to": selected_entries[1].get("value")}
        derivation_steps = [
            {
                "step_id": f"read-{position}",
                "operation": "read_version",
                "supporting_object_keys": changed["target_objects"],
                "supporting_event_ids": entry["source_event_ids"],
            }
            for position, entry in zip(("from", "to"), selected_entries)
        ]
        derivation_steps.append({
            "step_id": "answer",
            "operation": "object",
            "input_step_ids": ["read-from", "read-to"],
        })
        final_step = "answer"
    else:
        value = selected_entries[0].get("value")
        answer = [value] if selected_entries[0]["status"] == "tombstone" else value
        derivation_steps = [{
            "step_id": "read",
            "operation": "read_version",
            "supporting_object_keys": changed["target_objects"],
            "supporting_event_ids": selected_entries[0]["source_event_ids"],
        }]
        final_step = "read"
    changed["gold_evidence"][0].update(
        answer=answer,
        supporting_event_ids=[
            event_id
            for entry in selected_entries
            for event_id in entry["source_event_ids"]
        ],
        derivation_steps=derivation_steps,
        final_derivation_step_id=final_step,
    )
    return changed


@pytest.mark.parametrize(
    "selector",
    (
        {"kind": "exact_version", "version_index": 2},
        {"kind": "transition", "from_version_index": 1, "to_version_index": 2},
    ),
)
def test_contract_rejects_exact_or_transition_endpoint_beyond_horizon(selector):
    with pytest.raises(ValueError, match="unknown version"):
        MemUpdateTaskV3.model_validate(horizon_exact_transition_payload(selector))


@pytest.mark.parametrize(
    ("selector", "expected_indices"),
    (
        ({"kind": "exact_version", "version_index": 1}, (1,)),
        ({"kind": "transition", "from_version_index": 0, "to_version_index": 1}, (0, 1)),
    ),
)
def test_horizon_boundary_exact_and_transition_versions_remain_valid(selector, expected_indices):
    task = MemUpdateTaskV3.model_validate(horizon_exact_transition_payload(selector))
    replay = replay_task_v3(task)
    resolution = resolve_query_v3(task.queries[0], replay, task.events)

    assert replay.horizon_logical_time == "002"
    assert replay.issues == ()
    assert resolution.issues == ()
    assert tuple(version.version_index for version in resolution.selected_versions) == expected_indices
    assert resolution.selected_versions[-1].logical_time == "002"


@pytest.mark.parametrize(
    "selector",
    (
        {"kind": "exact_version", "version_index": 2},
        {"kind": "transition", "from_version_index": 1, "to_version_index": 2},
    ),
)
def test_defensive_query_resolution_never_selects_future_version(selector):
    from mub.vnext.contracts.v3.task import MemoryQueryV3

    task = MemUpdateTaskV3.model_validate(
        horizon_exact_transition_payload({"kind": "exact_version", "version_index": 1})
    )
    replay = replay_task_v3(task)
    query = MemoryQueryV3(
        query_id="defensive",
        query_type="transition" if selector["kind"] == "transition" else "point_in_time",
        text="?",
        selector=selector,
        target_object_keys=task.target_objects,
        answer_schema="object" if selector["kind"] == "transition" else "list",
        evaluation_mode="state_direct",
    )

    resolution = resolve_query_v3(query, replay, task.events)

    assert resolution.selected_versions == ()
    assert tuple(issue.code for issue in resolution.issues) == ("selector_missing_version",)


def test_ttl_scorer_uses_expiry_snapshot_not_scheduling_snapshot():
    from mub.vnext.contracts.v3.runtime import MemorySnapshotV3, ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    task = MemUpdateTaskV3.model_validate(future_ttl_payload())
    replay = replay_task_v3(task)
    ttl = next(action for action in task.actions if action.scope is not None and action.scope.value == "ttl")
    parsed = ParsedManagerActionV3(action_id=ttl.action_id, event_id=ttl.event_id, operation=ttl.operation, observed_scope=ttl.scope, target_object_keys=ttl.target_object_keys, format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
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

    current_task = MemUpdateTaskV3.model_validate(current_structured_payload("v2", "string"))
    current_replay = replay_task_v3(current_task)
    current_evidence = {item.query_id: item for item in current_task.gold_evidence}
    distractor = MemoryEntryRecordV3(entry_id="d", content="distractor", value_candidate="distractor", raw_metadata={"is_distractor": True})
    trace = RetrievalTraceV3(query_id="q", retrieved_entries=(distractor,), distractor_in_context=True)
    wrong = AnswerPredictionV3(query_id="q", raw_output="other", parsed_answer="other", format_valid=True)
    flags = derive_failure_flags_v3(task=current_task, run=run.model_copy(update={"answer_predictions": (wrong,), "retrieval_traces": (trace,)}), replay=current_replay, layer_values=empty_layers, predictions={"q": wrong}, traces={"q": trace}, evidence=current_evidence)
    assert "distractor_retrieved" in flags
    assert "distractor_copied" not in flags
    from mub.vnext.scoring.scorer_v3 import _metric_value
    value, detail = _metric_value("answer_scores.distractor_copied", current_task, run, None, current_replay, {}, current_evidence, {"q": wrong}, {"q": trace}, [])
    assert detail is None
    assert value == 0.0

    copied = AnswerPredictionV3(query_id="q", raw_output="distractor", parsed_answer="distractor", format_valid=True)
    flags = derive_failure_flags_v3(task=current_task, run=run.model_copy(update={"answer_predictions": (copied,), "retrieval_traces": (trace,)}), replay=current_replay, layer_values=empty_layers, predictions={"q": copied}, traces={"q": trace}, evidence=current_evidence)
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


def test_non_g_derivation_contract_rejects_future_horizon_read():
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    control = ttl_horizon_payload("002", "v1")
    control["events"][1]["timestamp"] = "002"
    control["actions"][1]["effective_at"] = "002"
    control["version_history"][0]["entries"][1]["logical_time"] = "002"
    control["gold_evidence"][0]["derivation_steps"][0]["operation"] = "read_version"
    task = MemUpdateTaskV3.model_validate(control)
    replay = replay_task_v3(task)
    evaluation = evaluate_evidence_v3(task.gold_evidence[0], replay)
    assert replay.horizon_logical_time == "002"
    assert evaluation.issues == ()
    assert evaluation.answer == "v1"

    future = deepcopy(control)
    future["gold_evidence"][0]["supporting_event_ids"] = ["e1", "e2"]
    future["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = ["e2"]

    defensive_steps = list(task.gold_evidence[0].derivation_steps)
    defensive_steps[0] = defensive_steps[0].model_copy(update={"supporting_event_ids": ("e2",)})
    defensive_evidence = task.gold_evidence[0].model_copy(update={"derivation_steps": tuple(defensive_steps)})
    defensive_evaluation = evaluate_evidence_v3(defensive_evidence, replay)
    assert tuple(issue.code for issue in defensive_evaluation.issues) == ("evidence_replay_error",)
    assert "missing" in defensive_evaluation.issues[0].message

    with pytest.raises(ValueError, match="support|missing|future|horizon"):
        MemUpdateTaskV3.model_validate(future)


def g_horizon_payload(query_type):
    changed = ttl_horizon_payload("002", "v1")
    changed["task_family"] = "G"
    changed["events"][1]["timestamp"] = "002"
    changed["actions"][1]["effective_at"] = "002"
    changed["version_history"][0]["entries"][1]["logical_time"] = "002"
    first = changed["target_objects"][0]
    if query_type == "update_sensitive_multi_hop":
        changed["queries"][0] = {
            "query_id": "q", "query_type": query_type, "text": "?",
            "selector": {"kind": "current"}, "target_object_keys": [first],
            "answer_schema": "string", "evaluation_mode": "state_direct",
            "synthesis": {"kind": query_type, "minimum_hops": 2},
        }
        changed["gold_evidence"][0] = {
            "query_id": "q", "answer": "v1", "supporting_object_keys": [first],
            "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "current", "operation": "read_current", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "answer", "operation": "answer", "input_step_ids": ["current"]},
            ],
            "final_derivation_step_id": "answer",
            "stale_alternative": {
                "answer": "v0", "supporting_object_keys": [first],
                "supporting_event_ids": ["e0"],
                "derivation_steps": [
                    {"step_id": "historical", "operation": "read_version", "supporting_object_keys": [first], "supporting_event_ids": ["e0"]},
                    {"step_id": "stale-answer", "operation": "answer", "input_step_ids": ["historical"]},
                ],
                "final_derivation_step_id": "stale-answer",
            },
        }
        return changed

    second = {**first, "entity": "e2-control"}
    changed["target_objects"].append(second)
    changed["events"][0]["gold_action_ids"].append("a-second")
    changed["actions"].insert(1, {
        "action_id": "a-second", "event_id": "e0", "operation": "ADD", "scope": "object",
        "target_object_keys": [second], "value": "v1", "effective_at": "000",
    })
    changed["version_history"].append({
        "object_key": second,
        "entries": [{
            "version_index": 0, "status": "present", "value": "v1",
            "valid_from_event_id": "e0", "logical_time": "000", "source_event_ids": ["e0"],
        }],
    })
    changed["queries"][0] = {
        "query_id": "q", "query_type": query_type, "text": "?",
        "selector": {"kind": "multi_object_current", "object_keys": [first, second]},
        "target_object_keys": [first, second], "answer_schema": "boolean",
        "evaluation_mode": "state_direct",
        "synthesis": {"kind": query_type, "minimum_objects": 2},
    }
    changed["gold_evidence"][0] = {
        "query_id": "q", "answer": True, "supporting_object_keys": [first, second],
        "supporting_event_ids": ["e0", "e1"],
        "derivation_steps": [
            {"step_id": "first-current", "operation": "read_current", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
            {"step_id": "second-current", "operation": "read_current", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-current", "second-current"]},
        ],
        "final_derivation_step_id": "equals",
        "stale_alternative": {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e0"],
            "derivation_steps": [
                {"step_id": "first-historical", "operation": "read_version", "supporting_object_keys": [first], "supporting_event_ids": ["e0"]},
                {"step_id": "second-historical", "operation": "read_version", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "stale-equals", "operation": "equals", "input_step_ids": ["first-historical", "second-historical"]},
            ],
            "final_derivation_step_id": "stale-equals",
        },
    }
    return changed


@pytest.mark.parametrize("query_type", ["update_sensitive_multi_hop", "multi_object_current_consistency"])
def test_g_horizon_boundary_current_and_active_history_reads_pass(query_type):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    task = MemUpdateTaskV3.model_validate(g_horizon_payload(query_type))
    replay = replay_task_v3(task)
    evaluation = evaluate_evidence_v3(
        task.gold_evidence[0], replay, task.gold_evidence[0].stale_alternative,
        task.queries[0], task.events,
    )

    assert replay.issues == ()
    assert replay.horizon_logical_time == "002"
    assert tuple(version.version_index for version in replay.active_versions(replay.ledgers[0])) == (0, 1)
    assert replay.active_versions(replay.ledgers[0])[-1].logical_time == "002"
    assert evaluation.issues == ()


@pytest.mark.parametrize("query_type", ["update_sensitive_multi_hop", "multi_object_current_consistency"])
def test_g_current_read_cannot_bind_active_historical_version(query_type):
    changed = g_horizon_payload(query_type)
    alternative = changed["gold_evidence"][0]["stale_alternative"]
    alternative["derivation_steps"][0]["operation"] = "read_current"

    with pytest.raises(ValueError, match="current|support|eligible|provenance"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("query_type", ["update_sensitive_multi_hop", "multi_object_current_consistency"])
@pytest.mark.parametrize("location", ["primary", "stale"])
def test_g_contract_rejects_future_horizon_derivation_reads(query_type, location):
    changed = g_horizon_payload(query_type)
    evidence = changed["gold_evidence"][0]
    item = evidence if location == "primary" else evidence["stale_alternative"]
    item["derivation_steps"][0]["supporting_event_ids"] = ["e2"]
    item["supporting_event_ids"] = sorted({
        event_id
        for step in item["derivation_steps"]
        for event_id in step.get("supporting_event_ids", [])
    })

    with pytest.raises(ValueError, match="support|eligible|provenance|selector"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("query_type", ["update_sensitive_multi_hop", "multi_object_current_consistency"])
@pytest.mark.parametrize("location", ["primary", "stale"])
def test_defensive_g_replay_marks_future_derivation_read_unavailable(query_type, location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    task = MemUpdateTaskV3.model_validate(g_horizon_payload(query_type))
    replay = replay_task_v3(task)
    evidence = task.gold_evidence[0]
    if location == "primary":
        steps = list(evidence.derivation_steps)
        steps[0] = steps[0].model_copy(update={"supporting_event_ids": ("e2",)})
        expected = None if query_type == "update_sensitive_multi_hop" else False
        evidence = evidence.model_copy(update={"answer": expected, "derivation_steps": tuple(steps)})
        alternative = None
    else:
        alternative = evidence.stale_alternative
        steps = list(alternative.derivation_steps)
        steps[0] = steps[0].model_copy(update={"supporting_event_ids": ("e2",)})
        expected = None if query_type == "update_sensitive_multi_hop" else False
        alternative = alternative.model_copy(update={"answer": expected, "derivation_steps": tuple(steps)})

    evaluation = evaluate_evidence_v3(evidence, replay, alternative)

    assert evaluation.answer is None
    assert tuple(issue.code for issue in evaluation.issues) == ("evidence_replay_error",)
    assert "missing" in evaluation.issues[0].message


def test_defensive_g_count_does_not_include_future_horizon_row():
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    task = MemUpdateTaskV3.model_validate(g_horizon_payload("update_sensitive_multi_hop"))
    replay = replay_task_v3(task)
    evidence_type = task.gold_evidence[0].__class__
    active_evidence = evidence_type.model_validate({
        "query_id": "q", "answer": 1, "supporting_object_keys": task.target_objects,
        "supporting_event_ids": ["e1"],
        "derivation_steps": [
            {"step_id": "boundary", "operation": "read_version", "supporting_object_keys": task.target_objects, "supporting_event_ids": ["e1"]},
            {"step_id": "rows", "operation": "collect", "input_step_ids": ["boundary"]},
            {"step_id": "count", "operation": "count", "input_step_ids": ["rows"]},
        ],
        "final_derivation_step_id": "count",
    })
    active_evaluation = evaluate_evidence_v3(active_evidence, replay)
    assert active_evaluation.issues == ()
    assert active_evaluation.answer == 1

    future_evidence = evidence_type.model_validate({
        "query_id": "q", "answer": 2, "supporting_object_keys": task.target_objects,
        "supporting_event_ids": ["e1", "e2"],
        "derivation_steps": [
            {"step_id": "boundary", "operation": "read_version", "supporting_object_keys": task.target_objects, "supporting_event_ids": ["e1"]},
            {"step_id": "future", "operation": "read_version", "supporting_object_keys": task.target_objects, "supporting_event_ids": ["e2"]},
            {"step_id": "rows", "operation": "collect", "input_step_ids": ["boundary", "future"]},
            {"step_id": "count", "operation": "count", "input_step_ids": ["rows"]},
        ],
        "final_derivation_step_id": "count",
    })
    future_evaluation = evaluate_evidence_v3(future_evidence, replay)

    assert future_evaluation.answer is None
    assert tuple(issue.code for issue in future_evaluation.issues) == ("evidence_replay_error",)
    assert "missing" in future_evaluation.issues[0].message


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


@pytest.mark.parametrize("task_payload", [payload, g_difference_payload])
def test_retrieval_metrics_are_not_applicable_to_historical_or_synthesis_queries(task_payload):
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    paths = (
        "retrieval_scores.current_recall_at_k",
        "retrieval_scores.current_mrr",
        "retrieval_scores.stale_exposure_rate",
        "retrieval_scores.stale_count_in_context",
        "retrieval_scores.distractor_exposure_rate",
        "answer_scores.gold_retrieved_wrong_answer",
    )
    task = MemUpdateTaskV3.model_validate(task_payload())
    first_version = task.version_history[0].entries[0]
    trace = RetrievalTraceV3(
        query_id=task.queries[0].query_id,
        retrieved_entries=(MemoryEntryRecordV3(
            entry_id="supplied-trace",
            content=str(first_version.value),
            object_key_candidate=task.target_objects[0],
            value_candidate=first_version.value,
            version_index=first_version.version_index,
            source_event_ids=first_version.source_event_ids,
        ),),
        ranks=(1,),
        gold_in_context=True,
        stale_in_context=True,
        distractor_in_context=True,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id=f"non-applicable-{task.task_family}",
        retrieval_traces=(trace,),
        answer_predictions=(AnswerPredictionV3(
            query_id=task.queries[0].query_id,
            raw_output=str(task.gold_evidence[0].answer),
            parsed_answer=task.gold_evidence[0].answer,
            format_valid=True,
        ),),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter",
        adapter_version="1",
        system_name="system",
        system_version="1",
        configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
        exports_retrieval_ids=True,
        exports_retrieval_scores=True,
    )
    config = ScorerConfigV3(requested_metric_fields=paths)

    score = score_task_v3(task, run, authenticated_context(task, run, info, caps, config))

    for path in paths:
        layer, leaf = path.split(".", 1)
        assert getattr(getattr(score, layer), leaf) is None
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE
    assert "stale_retrieved" not in score.failure_flags
    assert "distractor_retrieved" not in score.failure_flags
    assert "distractor_copied" not in score.failure_flags


def test_failed_historical_support_recall_does_not_imply_current_not_retrieved():
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, RetrievalTraceV3
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    trace = RetrievalTraceV3(
        query_id="q",
        gold_in_context=False,
        stale_in_context=False,
        distractor_in_context=False,
    )
    prediction = AnswerPredictionV3(
        query_id="q",
        raw_output=str(task.gold_evidence[0].answer),
        parsed_answer=task.gold_evidence[0].answer,
        format_valid=True,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id="historical-linkage-failure",
        retrieval_traces=(trace,),
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    flags = derive_failure_flags_v3(
        task=task,
        run=run,
        replay=replay,
        layer_values={
            "deletion_scores": {},
            "historical_scores": {"historical_support_recall": 0.0},
            "synthesis_scores": {},
        },
        predictions={"q": prediction},
        traces={"q": trace},
        evidence={"q": task.gold_evidence[0]},
    )

    assert "evidence_linkage_error" in flags
    assert "current_not_retrieved" not in flags


@pytest.mark.parametrize("family", ["multi_hop", "consistency"])
@pytest.mark.parametrize("location", ["primary", "stale"])
@pytest.mark.parametrize("laundering", ["mixed_read", "unused_top_level"])
def test_g_evidence_rejects_unrelated_or_unused_event_support(family, location, laundering):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    if family == "multi_hop":
        valid = g_stale_payload()
    else:
        common_event = "e3" if location == "primary" else "e1"
        valid = replayable_multi_object_consistency_payload(common_event)
        evidence = valid["gold_evidence"][0]
        if location == "stale":
            first, second = evidence["supporting_object_keys"]
            evidence["stale_alternative"] = {
                "answer": False, "supporting_object_keys": [first, second],
                "supporting_event_ids": ["e1"],
                "derivation_steps": [
                    {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                    {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e1"]},
                    {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
                ],
                "final_derivation_step_id": "equals",
            }
    valid["events"].append({
        "event_id": "e-noop", "sequence_index": len(valid["events"]),
        "raw_text": "unrelated", "normalized_text": "unrelated", "role": "neutral",
        "gold_action_ids": ["a-noop"],
    })
    valid["actions"].append({"action_id": "a-noop", "event_id": "e-noop", "operation": "NOOP"})
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    corrupted = deepcopy(valid)
    evidence = corrupted["gold_evidence"][0]
    item = evidence if location == "primary" else evidence["stale_alternative"]
    item["supporting_event_ids"].append("e-noop")
    if laundering == "mixed_read":
        item["derivation_steps"][0]["supporting_event_ids"].append("e-noop")
    with pytest.raises(ValueError, match="support|event|provenance"):
        MemUpdateTaskV3.model_validate(corrupted)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_update_sensitive_multi_hop_rejects_unbound_selector_or_stale_provenance(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    valid = g_stale_payload()
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    wrong = deepcopy(valid)
    evidence = wrong["gold_evidence"][0]
    if location == "primary":
        evidence.update(
            answer="v1",
            supporting_event_ids=["e1", "e3"],
            derivation_steps=[
                {"step_id": "stale-read", "operation": "read", "supporting_object_keys": wrong["target_objects"], "supporting_event_ids": ["e1"]},
                {"step_id": "answer", "operation": "answer", "input_step_ids": ["stale-read"]},
            ],
            final_derivation_step_id="answer",
        )
    else:
        evidence["stale_alternative"].update(
            answer="v2",
            supporting_event_ids=["e3"],
            derivation_steps=[
                {"step_id": "current-read", "operation": "read", "supporting_object_keys": wrong["target_objects"], "supporting_event_ids": ["e3"]},
                {"step_id": "stale-answer", "operation": "answer", "input_step_ids": ["current-read"]},
            ],
            final_derivation_step_id="stale-answer",
        )
    with pytest.raises(ValueError, match="eligible|provenance|selector"):
        MemUpdateTaskV3.model_validate(wrong)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_multi_hop_rejects_surplus_operands_with_false_current_influence(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    valid = g_difference_payload()
    evidence = valid["gold_evidence"][0]
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": 2, "supporting_object_keys": valid["target_objects"],
            "supporting_event_ids": ["e0"],
            "derivation_steps": [
                {"step_id": "previous", "operation": "read", "supporting_object_keys": valid["target_objects"], "supporting_event_ids": ["e0"]},
                {"step_id": "stale-answer", "operation": "answer", "input_step_ids": ["previous"]},
            ],
            "final_derivation_step_id": "stale-answer",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    bypass = deepcopy(valid)
    bypass_evidence = bypass["gold_evidence"][0]
    item = bypass_evidence if location == "primary" else bypass_evidence["stale_alternative"]
    item.update(
        answer=4,
        supporting_event_ids=["e0", "e1"],
        derivation_steps=[
            {"step_id": "previous-a", "operation": "read", "supporting_object_keys": bypass["target_objects"], "supporting_event_ids": ["e0"]},
            {"step_id": "previous-b", "operation": "read", "supporting_object_keys": bypass["target_objects"], "supporting_event_ids": ["e0"]},
            {"step_id": "current", "operation": "read", "supporting_object_keys": bypass["target_objects"], "supporting_event_ids": ["e1"]},
            {"step_id": "multiply", "operation": "multiply", "input_step_ids": ["previous-a", "previous-b", "current"]},
        ],
        final_derivation_step_id="multiply",
    )
    with pytest.raises(ValueError, match="multiply|operand|arity"):
        MemUpdateTaskV3.model_validate(bypass)


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
        "stale_alternative": {
            "answer": True, "supporting_object_keys": [first, second], "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "one-a", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "one-b", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "one", "operation": "equals", "input_step_ids": ["one-a", "one-b"]},
            ],
            "final_derivation_step_id": "one",
        },
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


def test_multi_object_consistency_rejects_stale_alternative_with_same_resolved_versions():
    changed = multi_object_consistency_payload()
    evidence = changed["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    evidence["stale_alternative"] = {
        "answer": False,
        "supporting_object_keys": [first, second],
        "supporting_event_ids": ["e0", "e3"],
        "derivation_steps": [
            {"step_id": "stale-first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "stale-first-copy", "operation": "identity", "input_step_ids": ["stale-first"]},
            {"step_id": "stale-second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "stale-second-copy", "operation": "identity", "input_step_ids": ["stale-second"]},
            {"step_id": "stale-equals", "operation": "equals", "input_step_ids": ["stale-first-copy", "stale-second-copy"]},
        ],
        "final_derivation_step_id": "stale-equals",
    }

    with pytest.raises(ValueError, match="same ledger versions"):
        MemUpdateTaskV3.model_validate(changed)


def replayable_multi_object_consistency_payload(second_event_id):
    changed = multi_object_consistency_payload()
    second = changed["target_objects"][1]
    event_time = {"e0": "000", "e1": "001", "e2": "002", "e3": "003"}[second_event_id]
    second_action = {
        "action_id": "a-second", "event_id": second_event_id, "operation": "ADD", "scope": "object",
        "target_object_keys": [second], "value": "z", "effective_at": event_time,
    }
    if second_event_id == "e0":
        changed["actions"].insert(1, second_action)
    elif second_event_id == "e1":
        changed["actions"].insert(2, second_action)
    elif second_event_id == "e2":
        changed["actions"].insert(2, second_action)
    else:
        changed["actions"].append(second_action)
    event = next(event for event in changed["events"] if event["event_id"] == second_event_id)
    if second_event_id == "e2":
        event["gold_action_ids"].insert(0, "a-second")
    else:
        event["gold_action_ids"].append("a-second")
    second_entry = changed["version_history"][1]["entries"][0]
    second_entry.update(
        valid_from_event_id=second_event_id,
        logical_time=event_time,
        source_event_ids=[second_event_id],
    )
    evidence = changed["gold_evidence"][0]
    evidence["supporting_event_ids"] = sorted({"e3", second_event_id})
    evidence["derivation_steps"][1]["supporting_event_ids"] = [second_event_id]
    return changed


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_event_only_multi_hop_reads_receive_resolved_target_influence(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    changed = g_stale_payload()
    item = changed["gold_evidence"][0] if location == "primary" else changed["gold_evidence"][0]["stale_alternative"]
    item["derivation_steps"][0]["supporting_object_keys"] = []
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    evaluation = evaluate_evidence_v3(
        task.gold_evidence[0], replay, task.gold_evidence[0].stale_alternative,
        task.queries[0], task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_event_only_consistency_reads_receive_resolved_target_influence(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    changed = multi_object_consistency_payload()
    first, second = changed["target_objects"]
    changed["events"].append({
        "event_id": "e4", "sequence_index": 4, "raw_text": "second", "normalized_text": "second",
        "role": "neutral", "gold_action_ids": ["a-second"],
    })
    changed["actions"].append({
        "action_id": "a-second", "event_id": "e4", "operation": "ADD", "scope": "object",
        "target_object_keys": [second], "value": "z", "effective_at": "004",
    })
    changed["version_history"][1]["entries"][0].update(
        valid_from_event_id="e4", logical_time="004", source_event_ids=["e4"],
    )
    evidence = changed["gold_evidence"][0]
    evidence.update(
        supporting_event_ids=["e3", "e4"],
        derivation_steps=[
            {"step_id": "first", "operation": "read", "supporting_event_ids": ["e3"]},
            {"step_id": "second", "operation": "read", "supporting_event_ids": ["e4"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "second"]},
        ],
    )
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e1", "e4"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_event_ids": ["e4"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    evaluation = evaluate_evidence_v3(
        task.gold_evidence[0], replay, task.gold_evidence[0].stale_alternative,
        task.queries[0], task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()
    assert evaluation.answer is False
    if location == "stale":
        assert evaluation.stale_alternative_answer is False


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_event_only_reads_fail_closed_when_event_resolves_multiple_targets(location):
    changed = replayable_multi_object_consistency_payload("e3")
    evidence = changed["gold_evidence"][0]
    ambiguous = {
        "answer": True, "supporting_object_keys": evidence["supporting_object_keys"],
        "supporting_event_ids": ["e3"],
        "derivation_steps": [
            {"step_id": "ambiguous-a", "operation": "read", "supporting_event_ids": ["e3"]},
            {"step_id": "ambiguous-b", "operation": "read", "supporting_event_ids": ["e3"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["ambiguous-a", "ambiguous-b"]},
        ],
        "final_derivation_step_id": "equals",
    }
    if location == "primary":
        evidence.update(ambiguous)
    else:
        evidence["stale_alternative"] = ambiguous
    with pytest.raises(ValueError, match="missing or ambiguous"):
        MemUpdateTaskV3.model_validate(changed)


def replayable_multi_target_hop_payload():
    changed = replayable_multi_object_consistency_payload("e0")
    first, second = changed["target_objects"]
    changed["actions"].append({
        "action_id": "a-second-update", "event_id": "e3", "operation": "UPDATE", "scope": "object",
        "target_object_keys": [second], "value": "z2", "effective_at": "003",
    })
    next(event for event in changed["events"] if event["event_id"] == "e3")["gold_action_ids"].append("a-second-update")
    second_entry = changed["version_history"][1]["entries"][0]
    second_entry["valid_until_event_id"] = "e3"
    changed["version_history"][1]["entries"].append({
        "version_index": 1, "status": "present", "value": "z2",
        "valid_from_event_id": "e3", "logical_time": "003", "source_event_ids": ["e3"],
    })
    changed["queries"][0].update(
        query_type="update_sensitive_multi_hop",
        selector={"kind": "multi_object_current", "object_keys": [first, second]},
        synthesis={"kind": "update_sensitive_multi_hop", "minimum_hops": 2},
    )
    changed["gold_evidence"][0].update(
        answer=False,
        supporting_event_ids=["e3"],
        derivation_steps=[
            {"step_id": "first-current", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "second-current", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e3"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-current", "second-current"]},
        ],
        final_derivation_step_id="equals",
        stale_alternative={
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e1", "e3"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second-current", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e3"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second-current"]},
            ],
            "final_derivation_step_id": "equals",
        },
    )
    return changed


@pytest.mark.parametrize("missing_target", ["first", "second"])
def test_stale_multi_hop_requires_read_influence_for_every_target(missing_target):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    valid = replayable_multi_target_hop_payload()
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()
    assert evaluation.stale_alternative_answer is False

    omitted = deepcopy(valid)
    first, second = omitted["target_objects"]
    retained = second if missing_target == "first" else first
    stale_event = "e0" if missing_target == "first" else "e1"
    alternative = omitted["gold_evidence"][0]["stale_alternative"]
    alternative.update(
        answer=True,
        supporting_event_ids=[stale_event],
        derivation_steps=[
            {"step_id": "duplicate-a", "operation": "read", "supporting_object_keys": [retained], "supporting_event_ids": [stale_event]},
            {"step_id": "duplicate-b", "operation": "read", "supporting_object_keys": [retained], "supporting_event_ids": [stale_event]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["duplicate-a", "duplicate-b"]},
        ],
        final_derivation_step_id="equals",
    )
    with pytest.raises(ValueError, match="target|coverage|influence"):
        MemUpdateTaskV3.model_validate(omitted)


def set_repeated_first_values(changed, value="same"):
    first = changed["target_objects"][0]
    for action in changed["actions"]:
        if action["operation"] in {"ADD", "UPDATE"} and first in action.get("target_object_keys", []):
            action["value"] = value
    for entry in changed["version_history"][0]["entries"]:
        if entry["status"] == "present":
            entry["value"] = value


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_multi_object_consistency_rejects_duplicate_multi_object_read_paths(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    common_event = "e3" if location == "primary" else "e1"
    valid = replayable_multi_object_consistency_payload(common_event)
    evidence = valid["gold_evidence"][0]
    if location == "stale":
        first, second = evidence["supporting_object_keys"]
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e1"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    alternative = valid_task.gold_evidence[0].stale_alternative
    evaluation = evaluate_evidence_v3(valid_task.gold_evidence[0], replay, alternative)
    assert replay.issues == ()
    assert evaluation.issues == ()
    assert evaluation.answer is False
    if location == "stale":
        assert evaluation.stale_alternative_answer is False

    duplicate = deepcopy(valid)
    duplicate_evidence = duplicate["gold_evidence"][0]
    item = duplicate_evidence if location == "primary" else duplicate_evidence["stale_alternative"]
    first, second = duplicate_evidence["supporting_object_keys"]
    item.update(
        answer=True,
        derivation_steps=[
            {"step_id": "both-a", "operation": "read", "supporting_object_keys": [first, second], "supporting_event_ids": [common_event]},
            {"step_id": "both-b", "operation": "read", "supporting_object_keys": [first, second], "supporting_event_ids": [common_event]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["both-a", "both-b"]},
        ],
        final_derivation_step_id="equals",
    )
    with pytest.raises(ValueError, match="minimum_objects"):
        MemUpdateTaskV3.model_validate(duplicate)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_multi_object_consistency_rejects_same_value_wrong_version_provenance(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    second_event = "e3" if location == "primary" else "e0"
    valid = replayable_multi_object_consistency_payload(second_event)
    set_repeated_first_values(valid)
    first, second = valid["gold_evidence"][0]["supporting_object_keys"]
    evidence = valid["gold_evidence"][0]
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0],
        replay,
        valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0],
        valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    wrong = deepcopy(valid)
    wrong_evidence = wrong["gold_evidence"][0]
    item = wrong_evidence if location == "primary" else wrong_evidence["stale_alternative"]
    item["supporting_event_ids"] = ["e1", "e3"] if location == "primary" else ["e0", "e1"]
    item["derivation_steps"][0]["supporting_event_ids"] = ["e1"] if location == "primary" else ["e0"]
    with pytest.raises(ValueError, match="eligible|provenance"):
        MemUpdateTaskV3.model_validate(wrong)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_consistency_rejects_answer_bearing_multi_key_read_bypass(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    second_event = "e1" if location == "primary" else "e0"
    valid = replayable_multi_object_consistency_payload(second_event)
    set_repeated_first_values(valid)
    evidence = valid["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    bypass = deepcopy(valid)
    bypass_evidence = bypass["gold_evidence"][0]
    item = bypass_evidence if location == "primary" else bypass_evidence["stale_alternative"]
    if location == "primary":
        item.update(
            answer=True,
            supporting_event_ids=["e1", "e3"],
            derivation_steps=[
                {"step_id": "current-first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
                {"step_id": "current-second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e1"]},
                {"step_id": "current-pair", "operation": "collect", "input_step_ids": ["current-first", "current-second"]},
                {"step_id": "mixed-pair", "operation": "read", "supporting_object_keys": [first, second], "supporting_event_ids": ["e1"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["current-pair", "mixed-pair"]},
            ],
            final_derivation_step_id="equals",
        )
    else:
        item.update(
            answer=True,
            supporting_event_ids=["e0", "e1"],
            derivation_steps=[
                {"step_id": "stale-first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "current-second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "authenticated-pair", "operation": "collect", "input_step_ids": ["stale-first", "current-second"]},
                {"step_id": "mixed-pair", "operation": "read", "supporting_object_keys": [first, second], "supporting_event_ids": ["e0"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["authenticated-pair", "mixed-pair"]},
            ],
            final_derivation_step_id="equals",
        )
    with pytest.raises(ValueError, match="multi-key|eligible|provenance"):
        MemUpdateTaskV3.model_validate(bypass)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_consistency_rejects_surplus_operands_with_false_target_influence(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    common_event = "e3" if location == "primary" else "e1"
    valid = replayable_multi_object_consistency_payload(common_event)
    set_repeated_first_values(valid, 2)
    second = valid["target_objects"][1]
    for action in valid["actions"]:
        if second in action.get("target_object_keys", []):
            action["value"] = 3
    valid["version_history"][1]["entries"][0]["value"] = 3
    evidence = valid["gold_evidence"][0]
    first = evidence["supporting_object_keys"][0]
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e1"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    bypass = deepcopy(valid)
    bypass_evidence = bypass["gold_evidence"][0]
    item = bypass_evidence if location == "primary" else bypass_evidence["stale_alternative"]
    item.update(
        answer=True,
        derivation_steps=[
            {"step_id": "first-a", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
            {"step_id": "first-b", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
            {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": [common_event]},
            {"step_id": "with-surplus", "operation": "multiply", "input_step_ids": ["first-a", "first-b", "second"]},
            {"step_id": "without-surplus", "operation": "multiply", "input_step_ids": ["first-a", "first-b"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["with-surplus", "without-surplus"]},
        ],
        final_derivation_step_id="equals",
    )
    with pytest.raises(ValueError, match="multiply|operand|arity"):
        MemUpdateTaskV3.model_validate(bypass)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_consistency_requires_operation_aware_target_influence(location):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    common_event = "e3" if location == "primary" else "e1"
    valid = replayable_multi_object_consistency_payload(common_event)
    evidence = valid["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    if location == "stale":
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e1"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()
    assert evaluation.answer is False
    if location == "stale":
        assert evaluation.stale_alternative_answer is False

    bypass = deepcopy(valid)
    bypass_evidence = bypass["gold_evidence"][0]
    item = bypass_evidence if location == "primary" else bypass_evidence["stale_alternative"]
    item.update(
        answer=True,
        derivation_steps=[
            {"step_id": "first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
            {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": [common_event]},
            {"step_id": "select-first", "operation": "seed0", "input_step_ids": ["first", "second"], "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "select-first"]},
        ],
        final_derivation_step_id="equals",
    )
    with pytest.raises(ValueError, match="seed|influence|minimum_objects"):
        MemUpdateTaskV3.model_validate(bypass)


@pytest.mark.parametrize("location", ["primary", "stale"])
@pytest.mark.parametrize(
    ("operation", "binding"),
    (("seed0", "missing"), ("collect", "ambiguous"), ("consistency", "missing"), ("ordered_history", "ambiguous")),
)
def test_implicit_read_operations_require_unambiguous_binding(location, operation, binding):
    from mub.vnext.validation.replay_v3 import evaluate_evidence_v3

    common_event = "e3" if location == "primary" else "e1"
    valid = replayable_multi_object_consistency_payload(common_event)
    evidence = valid["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    implicit_graph = [
        {"step_id": "first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
        {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": [common_event]},
        {"step_id": "implicit", "operation": operation, "supporting_object_keys": [first], "supporting_event_ids": [common_event]},
        {"step_id": "left", "operation": "collect", "input_step_ids": ["first", "implicit"]},
        {"step_id": "right", "operation": "collect", "input_step_ids": ["second", "implicit"]},
        {"step_id": "equals", "operation": "equals", "input_step_ids": ["left", "right"]},
    ]
    if location == "primary":
        evidence.update(
            answer=False,
            supporting_event_ids=[common_event],
            derivation_steps=implicit_graph,
            final_derivation_step_id="equals",
        )
    else:
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": [common_event],
            "derivation_steps": implicit_graph,
            "final_derivation_step_id": "equals",
        }
    valid_task = MemUpdateTaskV3.model_validate(valid)
    replay = replay_task_v3(valid_task)
    evaluation = evaluate_evidence_v3(
        valid_task.gold_evidence[0], replay, valid_task.gold_evidence[0].stale_alternative,
        valid_task.queries[0], valid_task.events,
    )
    assert replay.issues == ()
    assert evaluation.issues == ()

    invalid = deepcopy(valid)
    invalid_evidence = invalid["gold_evidence"][0]
    item = invalid_evidence if location == "primary" else invalid_evidence["stale_alternative"]
    if binding == "missing":
        item["derivation_steps"][2]["supporting_event_ids"] = []
    else:
        item["supporting_event_ids"] = ["e0", "e1", "e3"] if location == "primary" else ["e0", "e1"]
        item["derivation_steps"][2]["supporting_event_ids"] = ["e0", "e1"]
    with pytest.raises(ValueError, match="read support is missing or ambiguous"):
        MemUpdateTaskV3.model_validate(invalid)


@pytest.mark.parametrize("location", ["primary", "stale"])
def test_task_contract_rejects_unary_equals_after_two_valid_target_reads(location):
    changed = multi_object_consistency_payload()
    evidence = changed["gold_evidence"][0]
    if location == "primary":
        evidence["derivation_steps"] = [
            *evidence["derivation_steps"][:2],
            {"step_id": "both", "operation": "collect", "input_step_ids": ["first", "second"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["both"]},
        ]
    else:
        first, second = evidence["supporting_object_keys"]
        evidence["stale_alternative"] = {
            "answer": True, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "both", "operation": "collect", "input_step_ids": ["first-stale", "second"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["both"]},
            ],
            "final_derivation_step_id": "equals",
        }
    with pytest.raises(ValueError, match="equals requires at least two operands"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("location", ["primary", "stale"])
@pytest.mark.parametrize("binding", ["missing", "ambiguous"])
def test_task_contract_requires_unambiguous_read_version_binding(location, binding):
    changed = multi_object_consistency_payload()
    evidence = changed["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    first_events = [] if binding == "missing" else ["e0", "e1"]
    if location == "primary":
        evidence["supporting_event_ids"] = ["e0", "e1", "e3"]
        evidence["derivation_steps"] = [
            {"step_id": "first", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": first_events},
            {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "second"]},
        ]
    else:
        evidence["stale_alternative"] = {
            "answer": False, "supporting_object_keys": [first, second],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-stale", "operation": "read", "supporting_object_keys": [first], "supporting_event_ids": first_events},
                {"step_id": "second", "operation": "read", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-stale", "second"]},
            ],
            "final_derivation_step_id": "equals",
        }
    with pytest.raises(ValueError, match="read support is missing or ambiguous"):
        MemUpdateTaskV3.model_validate(changed)


@pytest.mark.parametrize("location", ["primary-no-input", "primary-one-input", "stale-no-input"])
def test_multi_object_consistency_requires_two_distinct_reachable_read_operands(location):
    changed = multi_object_consistency_payload()
    evidence = changed["gold_evidence"][0]
    first, second = evidence["supporting_object_keys"]
    if location == "primary-no-input":
        evidence["derivation_steps"] = [
            {"step_id": "first-seed", "operation": "constant0", "supporting_object_keys": [first], "supporting_event_ids": ["e3"]},
            {"step_id": "second-seed", "operation": "constant1", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first-seed", "second-seed"]},
        ]
    elif location == "primary-one-input":
        evidence["derivation_steps"] = [
            evidence["derivation_steps"][0],
            {"step_id": "second-seed", "operation": "constant0", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["first", "second-seed"]},
        ]
    else:
        evidence["stale_alternative"] = {
            "answer": True, "supporting_object_keys": evidence["supporting_object_keys"],
            "supporting_event_ids": ["e0", "e1"],
            "derivation_steps": [
                {"step_id": "first-seed", "operation": "constant0", "supporting_object_keys": [first], "supporting_event_ids": ["e1"]},
                {"step_id": "second-seed", "operation": "constant1", "supporting_object_keys": [second], "supporting_event_ids": ["e0"]},
                {"step_id": "stale-equals", "operation": "equals", "input_step_ids": ["first-seed", "second-seed"]},
            ],
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
    valid = QueryGoldEvidenceV3(
        query_id="vacuous", answer=True, supporting_object_keys=(key,), supporting_event_ids=("e3",),
        derivation_steps=(
            {"step_id": "read-a", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e3"]},
            {"step_id": "read-b", "operation": "read", "supporting_object_keys": [key], "supporting_event_ids": ["e3"]},
            {"step_id": "equals", "operation": "equals", "input_step_ids": ["read-a", "read-b"]},
        ),
        final_derivation_step_id="equals",
    )
    equals = valid.derivation_steps[-1].model_copy(
        update={"input_step_ids": () if operand_count == 0 else ("read-a",)}
    )
    evidence = QueryGoldEvidenceV3.model_construct(
        query_id=valid.query_id,
        answer=valid.answer,
        supporting_object_keys=valid.supporting_object_keys,
        supporting_event_ids=valid.supporting_event_ids,
        derivation_steps=(equals,) if operand_count == 0 else (valid.derivation_steps[0], equals),
        final_derivation_step_id=valid.final_derivation_step_id,
        stale_alternative=None,
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
    parsed_actions = tuple(ParsedManagerActionV3(action_id=action.action_id, event_id=action.event_id, operation=action.operation, observed_scope=action.scope, target_object_keys=action.target_object_keys, value=action.value, format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok") for action in task.actions)
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
    wrong_scope_action = ParsedManagerActionV3(action_id=delete.action_id, event_id=delete.event_id, operation="DELETE", observed_scope="namespace", target_object_keys=delete.target_object_keys, format_valid=True, execution_status="executed", fallback_used=False, raw_output="wrong-scope")
    cases.append(("wrong_delete_scope", base_task, TaskRunRecordV3(task_id="t", adapter_id="a", run_id="scope", parsed_actions=(wrong_scope_action,), parser_extractor_provenance=provenance, completion_status="completed"), empty, {}, {}, base_evidence, base_replay))
    add = base_task.actions[0]
    wrong_key = add.target_object_keys[0].model_copy(update={"namespace": "other"})
    wrong_key_action = ParsedManagerActionV3(action_id=add.action_id, event_id=add.event_id, operation=add.operation, observed_scope=add.scope, target_object_keys=(wrong_key,), value=add.value, format_valid=True, execution_status="executed", fallback_used=False, raw_output="wrong-key")
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
        action_id=gold.action_id,
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


def _parsed_action_for(action, *, action_id=None, event_id=None):
    from mub.vnext.contracts.v3.runtime import ParsedManagerActionV3

    return ParsedManagerActionV3(
        action_id=action.action_id if action_id is None else action_id,
        event_id=action.event_id if event_id is None else event_id,
        operation=action.operation, observed_scope=action.scope,
        target_object_keys=action.target_object_keys, value=action.value, format_valid=True,
        execution_status="executed", fallback_used=False, raw_output="ok",
    )


def _run_with_parsed_actions(parsed_actions, run_id):
    return TaskRunRecordV3(
        task_id="t", adapter_id="a", run_id=run_id, parsed_actions=tuple(parsed_actions),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )


def _failure_flags_for_actions(task, parsed_actions, run_id):
    from mub.vnext.scoring.failures_v3 import derive_failure_flags_v3

    run = _run_with_parsed_actions(parsed_actions, run_id)
    replay = replay_task_v3(task)
    assert replay.issues == ()
    return derive_failure_flags_v3(
        task=task,
        run=run,
        replay=replay,
        layer_values={"deletion_scores": {}, "historical_scores": {}, "synthesis_scores": {}},
        predictions={},
        traces={},
        evidence={item.query_id: item for item in task.gold_evidence},
    )


def _multi_target_delete_task():
    changed = payload()
    first = {**changed["target_objects"][0], "subkey": "first"}
    second = {**first, "subkey": "second"}
    third = {**first, "entity": "other", "attribute": "b", "subkey": None}
    changed["target_objects"] = [first, second, third]
    changed["events"][0]["gold_action_ids"] = ["a0", "a0-second", "a0-third"]
    changed["actions"][0]["target_object_keys"] = [first]
    changed["actions"][1]["target_object_keys"] = [first]
    changed["actions"][2].update(
        {"scope": "namespace", "target_object_keys": [first, second, third]}
    )
    changed["actions"][3]["target_object_keys"] = [first]
    changed["actions"][1:1] = [
        {
            "action_id": "a0-second", "event_id": "e0", "operation": "ADD",
            "scope": "object", "target_object_keys": [second], "value": "second-v0",
            "effective_at": "000",
        },
        {
            "action_id": "a0-third", "event_id": "e0", "operation": "ADD",
            "scope": "object", "target_object_keys": [third], "value": "third-v0",
            "effective_at": "000",
        },
    ]
    changed["version_history"][0]["object_key"] = first
    for entry in changed["version_history"][0]["entries"]:
        entry["source_event_ids"] = [entry["valid_from_event_id"]]
    changed["version_history"].extend(
        {
            "object_key": key,
            "entries": [
                {
                    "version_index": 0, "status": "present", "value": value,
                    "valid_from_event_id": "e0", "valid_until_event_id": "e2",
                    "logical_time": "000", "source_event_ids": ["e0"],
                },
                {
                    "version_index": 1, "status": "tombstone",
                    "valid_from_event_id": "e2", "logical_time": "002",
                    "source_event_ids": ["e2"],
                },
            ],
        }
        for key, value in ((second, "second-v0"), (third, "third-v0"))
    )
    changed["queries"][0]["target_object_keys"] = [first]
    changed["gold_evidence"][0]["supporting_object_keys"] = [first]
    changed["gold_evidence"][0]["derivation_steps"][0]["supporting_object_keys"] = [first]
    return MemUpdateTaskV3.model_validate(changed)


def _action_score(task, run, metric):
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    replay = replay_task_v3(task)
    assert replay.issues == ()
    value, detail = _metric_value(
        f"action_scores.{metric}", task, run, None, replay, {}, {}, {}, {},
        _action_facts(task, run),
    )
    assert detail is None
    return value


def test_multi_target_delete_reordering_is_semantically_exact_without_key_flags():
    task = _multi_target_delete_task()
    delete = next(action for action in task.actions if action.operation.value == "DELETE")
    reversed_targets = tuple(reversed(delete.target_object_keys))
    parsed_actions = tuple(
        _parsed_action_for(action).model_copy(update={"target_object_keys": reversed_targets})
        if action.action_id == delete.action_id
        else _parsed_action_for(action)
        for action in task.actions
    )
    run = _run_with_parsed_actions(parsed_actions, "reversed-multi-target-delete")

    assert {
        metric: _action_score(task, run, metric)
        for metric in (
            "object_key_accuracy", "entity_accuracy", "attribute_accuracy",
            "full_action_exact_match",
        )
    } == {
        "object_key_accuracy": 1.0,
        "entity_accuracy": 1.0,
        "attribute_accuracy": 1.0,
        "full_action_exact_match": 1.0,
    }
    assert tuple(
        action.target_object_keys
        for action in run.parsed_actions
        if action.action_id == delete.action_id
    )[0] == reversed_targets
    flags = _failure_flags_for_actions(task, parsed_actions, "reversed-multi-target-delete")
    assert not {"wrong_object_key", "wrong_entity", "wrong_attribute"} & set(flags)


@pytest.mark.parametrize(
    ("component", "expected_related_flags", "expected_entity", "expected_attribute"),
    [
        ("namespace", {"wrong_object_key"}, 1.0, 1.0),
        ("subkey", {"wrong_object_key"}, 1.0, 1.0),
        ("entity", {"wrong_object_key", "wrong_entity"}, 0.0, 1.0),
        ("attribute", {"wrong_object_key", "wrong_attribute"}, 1.0, 0.0),
    ],
)
def test_multi_target_delete_identity_component_controls_still_fail(
    component, expected_related_flags, expected_entity, expected_attribute,
):
    task = _multi_target_delete_task()
    delete = next(action for action in task.actions if action.operation.value == "DELETE")
    targets = list(delete.target_object_keys)
    if component == "namespace":
        targets = [key.model_copy(update={"namespace": "other-ns"}) for key in targets]
    elif component == "subkey":
        targets[0] = targets[0].model_copy(update={"subkey": "changed"})
    elif component == "entity":
        targets[0] = targets[0].model_copy(update={"entity": targets[2].entity})
    else:
        targets[0] = targets[0].model_copy(update={"attribute": targets[2].attribute})
    observed = _parsed_action_for(delete).model_copy(update={"target_object_keys": tuple(targets)})
    parsed_actions = tuple(
        observed if action.action_id == delete.action_id else _parsed_action_for(action)
        for action in task.actions
    )
    run = _run_with_parsed_actions(parsed_actions, f"changed-{component}")

    incorrect_rate = (len(task.actions) - 1) / len(task.actions)
    assert _action_score(task, run, "object_key_accuracy") == incorrect_rate
    assert _action_score(task, run, "entity_accuracy") == (
        1.0 if expected_entity == 1.0 else incorrect_rate
    )
    assert _action_score(task, run, "attribute_accuracy") == (
        1.0 if expected_attribute == 1.0 else incorrect_rate
    )
    assert _action_score(task, run, "full_action_exact_match") == incorrect_rate
    related = {"wrong_object_key", "wrong_entity", "wrong_attribute"}
    assert set(_failure_flags_for_actions(task, parsed_actions, f"changed-{component}")) & related == expected_related_flags


@pytest.mark.parametrize("projection", ["entity", "attribute"])
def test_multi_target_projection_comparison_preserves_duplicate_multiplicity(projection):
    task = _multi_target_delete_task()
    delete = next(action for action in task.actions if action.operation.value == "DELETE")
    targets = list(delete.target_object_keys)
    targets[0] = targets[0].model_copy(
        update={projection: getattr(targets[2], projection)}
    )
    assert {
        getattr(key, projection) for key in targets
    } == {
        getattr(key, projection) for key in delete.target_object_keys
    }
    observed = _parsed_action_for(delete).model_copy(update={"target_object_keys": tuple(targets)})
    parsed_actions = tuple(
        observed if action.action_id == delete.action_id else _parsed_action_for(action)
        for action in task.actions
    )
    run = _run_with_parsed_actions(parsed_actions, f"duplicate-{projection}")

    assert _action_score(task, run, f"{projection}_accuracy") == (
        len(task.actions) - 1
    ) / len(task.actions)
    assert f"wrong_{projection}" in _failure_flags_for_actions(
        task, parsed_actions, f"duplicate-{projection}",
    )


def test_failure_flags_bind_both_correct_same_event_actions_independently():
    task = MemUpdateTaskV3.model_validate(replayable_multi_object_consistency_payload("e3"))

    flags = _failure_flags_for_actions(
        task,
        (_parsed_action_for(action) for action in task.actions),
        "failure-both-correct",
    )

    assert flags == ()


def test_failure_flags_attribute_wrong_second_same_event_action_only_to_second():
    task = MemUpdateTaskV3.model_validate(replayable_multi_object_consistency_payload("e3"))
    second = next(action for action in task.actions if action.action_id == "a-second")
    parsed_actions = [
        _parsed_action_for(action).model_copy(update={"value": "wrong"})
        if action.action_id == second.action_id
        else _parsed_action_for(action)
        for action in task.actions
    ]

    flags = _failure_flags_for_actions(task, parsed_actions, "failure-second-wrong")

    assert flags == ("wrong_value",)


def test_failure_flags_attribute_one_missing_same_event_deletion_once():
    task = MemUpdateTaskV3.model_validate(replayable_multi_object_consistency_payload("e2"))
    same_event = [action for action in task.actions if action.event_id == "e2"]
    missing = same_event[1]
    assert missing.operation.value == "DELETE"

    flags = _failure_flags_for_actions(
        task,
        (_parsed_action_for(action) for action in task.actions if action.action_id != missing.action_id),
        "failure-second-missing",
    )

    assert flags == ("missed_update", "deletion_failure")
    assert flags.count("missed_update") == 1
    assert flags.count("deletion_failure") == 1


def test_failure_flags_reject_unknown_action_id_with_scorer_message():
    from mub.vnext.scoring.scorer_v3 import _action_facts

    task = MemUpdateTaskV3.model_validate(payload())
    observed = _parsed_action_for(task.actions[0], action_id="unknown-action")
    run = _run_with_parsed_actions((observed,), "failure-unknown-id")

    with pytest.raises(ValueError) as scorer_error:
        _action_facts(task, run)
    with pytest.raises(ValueError) as failure_error:
        _failure_flags_for_actions(task, (observed,), "failure-unknown-id")

    assert str(failure_error.value) == str(scorer_error.value)
    assert str(failure_error.value) == "runtime contains unknown observed action_id values: ['unknown-action']"


def test_failure_flags_reject_known_action_id_with_wrong_event_using_scorer_message():
    from mub.vnext.scoring.scorer_v3 import _action_facts

    task = MemUpdateTaskV3.model_validate(payload())
    observed = _parsed_action_for(task.actions[0], event_id=task.actions[1].event_id)
    run = _run_with_parsed_actions((observed,), "failure-wrong-event")

    with pytest.raises(ValueError) as scorer_error:
        _action_facts(task, run)
    with pytest.raises(ValueError) as failure_error:
        _failure_flags_for_actions(task, (observed,), "failure-wrong-event")

    assert str(failure_error.value) == str(scorer_error.value)
    assert str(failure_error.value) == (
        "observed action event_id mismatch for action_id 'a0': expected 'e0', got 'e1'"
    )


def test_action_facts_bind_same_event_actions_by_action_id_with_perfect_metrics():
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    task = MemUpdateTaskV3.model_validate(replayable_multi_object_consistency_payload("e3"))
    replay = replay_task_v3(task)
    same_event = [action for action in task.actions if action.event_id == "e3"]
    assert replay.issues == ()
    assert len(same_event) == 2
    assert len({action.action_id for action in same_event}) == 2

    run = _run_with_parsed_actions(
        (_parsed_action_for(action) for action in task.actions),
        "same-event-actions",
    )
    facts = _action_facts(task, run)
    same_event_facts = [fact for fact in facts if fact[0].event_id == "e3"]
    assert len(same_event_facts) == 2
    assert [(gold.action_id, observed.action_id if observed else None) for gold, observed, *_ in facts] == [
        (action.action_id, action.action_id) for action in task.actions
    ]

    expected_metrics = {
        "operation_accuracy": 1.0,
        "entity_accuracy": 1.0,
        "attribute_accuracy": 1.0,
        "object_key_accuracy": 1.0,
        "value_accuracy": 1.0,
        "full_action_exact_match": 1.0,
        "false_write_rate": 0.0,
        "missed_write_rate": 0.0,
        "wrong_object_write_rate": 0.0,
    }
    for leaf, expected in expected_metrics.items():
        value, detail = _metric_value(
            f"action_scores.{leaf}", task, run, None, replay, {}, {}, {}, {}, facts,
        )
        assert detail is None
        assert value == expected


@pytest.mark.parametrize("event_id", ["e0", "unknown-event"])
def test_action_facts_reject_unknown_action_id_without_event_fallback(event_id):
    from mub.vnext.scoring.scorer_v3 import _action_facts

    task = MemUpdateTaskV3.model_validate(payload())
    observed = _parsed_action_for(task.actions[0], action_id="unknown-action", event_id=event_id)
    run = _run_with_parsed_actions((observed,), f"unknown-id-{event_id}")
    with pytest.raises(ValueError, match="unknown observed action_id"):
        _action_facts(task, run)


def test_action_facts_reject_known_action_id_with_wrong_event_id():
    from mub.vnext.scoring.scorer_v3 import _action_facts

    task = MemUpdateTaskV3.model_validate(payload())
    observed = _parsed_action_for(task.actions[0], event_id=task.actions[1].event_id)
    run = _run_with_parsed_actions((observed,), "wrong-event")
    with pytest.raises(ValueError, match="event_id mismatch"):
        _action_facts(task, run)


def test_action_facts_represent_one_missing_same_event_action_once():
    from mub.vnext.scoring.scorer_v3 import _action_facts, _metric_value

    task = MemUpdateTaskV3.model_validate(replayable_multi_object_consistency_payload("e3"))
    replay = replay_task_v3(task)
    missing = next(action for action in task.actions if action.action_id == "a-second")
    run = _run_with_parsed_actions(
        (_parsed_action_for(action) for action in task.actions if action.action_id != missing.action_id),
        "one-missing-action",
    )
    facts = _action_facts(task, run)
    missing_facts = [fact for fact in facts if fact[1] is None]
    assert len(facts) == len(task.actions)
    assert len(missing_facts) == 1
    assert missing_facts[0][0].action_id == missing.action_id

    execution_rate, detail = _metric_value(
        "protocol_scores.execution_success_rate", task, run, None, replay, {}, {}, {}, {}, facts,
    )
    assert detail is None
    assert execution_rate == (len(task.actions) - 1) / len(task.actions)


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
    from mub.vnext.scoring.lifecycle_v3 import TargetLifecycleClassifierV3
    from mub.vnext.scoring.scorer_v3 import _entry_obsolete_status

    task = MemUpdateTaskV3.model_validate(ttl_horizon_payload("005", "v1"))
    replay = replay_task_v3(task)
    current = MemoryEntryRecordV3(
        entry_id="current-before-expiry", content="v1",
        object_key_candidate=task.target_objects[0], value_candidate="v1",
        version_index=1, source_event_ids=("e1",),
    )
    lifecycle = TargetLifecycleClassifierV3.for_query(
        task.queries[0], replay,
    ).classify_entry(current)
    assert replay.obsolete_present_values == ("v0",)
    assert _entry_obsolete_status(current, replay) is False
    assert lifecycle.forgotten is False


def test_mixed_version_provenance_is_indeterminate():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.scorer_v3 import _entry_obsolete_status

    task = MemUpdateTaskV3.model_validate(payload())
    replay = replay_task_v3(task)
    mixed = MemoryEntryRecordV3(
        entry_id="mixed-provenance", content="v0",
        object_key_candidate=task.target_objects[0], value_candidate="v0",
        source_event_ids=("e0", "e3"),
    )
    assert _entry_obsolete_status(mixed, replay) is None


_LIFECYCLE_METRICS = (
    "retrieval_scores.stale_exposure_rate",
    "retrieval_scores.stale_count_in_context",
    "answer_scores.stale_copied",
    "deletion_scores.forgotten_exposure_rate",
    "deletion_scores.forgotten_value_leakage_rate",
)


def _two_object_lifecycle_payload(query_entity):
    changed = payload()
    key_a = deepcopy(changed["target_objects"][0])
    key_a["entity"] = "A"
    key_b = deepcopy(key_a)
    key_b["entity"] = "B"
    changed["task_family"] = "E"
    changed["events"] = [
        {"event_id": "e0", "sequence_index": 0, "raw_text": "A", "normalized_text": "A", "role": "neutral", "gold_action_ids": ["a0"]},
        {"event_id": "e1", "sequence_index": 1, "raw_text": "B", "normalized_text": "B", "role": "neutral", "gold_action_ids": ["a1"]},
        {"event_id": "e2", "sequence_index": 2, "raw_text": "forget B", "normalized_text": "forget B", "role": "neutral", "gold_action_ids": ["a2"]},
    ]
    changed["target_objects"] = [key_a, key_b]
    changed["actions"] = [
        {"action_id": "a0", "event_id": "e0", "operation": "ADD", "scope": "object", "target_object_keys": [key_a], "value": "a-current", "effective_at": "000"},
        {"action_id": "a1", "event_id": "e1", "operation": "ADD", "scope": "object", "target_object_keys": [key_b], "value": "b-deleted", "effective_at": "001"},
        {"action_id": "a2", "event_id": "e2", "operation": "DELETE", "scope": "object", "target_object_keys": [key_b], "effective_at": "002"},
    ]
    changed["version_history"] = [
        {"object_key": key_a, "entries": [
            {"version_index": 0, "status": "present", "value": "a-current", "valid_from_event_id": "e0", "logical_time": "000", "source_event_ids": ["e0"]},
        ]},
        {"object_key": key_b, "entries": [
            {"version_index": 0, "status": "present", "value": "b-deleted", "valid_from_event_id": "e1", "valid_until_event_id": "e2", "logical_time": "001", "source_event_ids": ["e1"]},
            {"version_index": 1, "status": "tombstone", "valid_from_event_id": "e2", "logical_time": "002", "source_event_ids": ["e2"]},
        ]},
    ]
    target = key_a if query_entity == "A" else key_b
    answer = "a-current" if query_entity == "A" else [None]
    support_event = "e0" if query_entity == "A" else "e2"
    changed["queries"] = [{
        "query_id": "q", "query_type": "current", "text": "?", "selector": {"kind": "current"},
        "target_object_keys": [target], "answer_schema": "string" if query_entity == "A" else "list",
        "evaluation_mode": "state_direct",
    }]
    changed["gold_evidence"] = [{
        "query_id": "q", "answer": answer, "supporting_object_keys": [target],
        "supporting_event_ids": [support_event],
        "derivation_steps": [{"step_id": "read", "operation": "read", "supporting_object_keys": [target], "supporting_event_ids": [support_event]}],
        "final_derivation_step_id": "read",
    }]
    return changed


def _two_object_shared_value_collision_payload():
    changed = _two_object_lifecycle_payload("A")
    key_a, key_b = changed["target_objects"]
    changed["actions"][0]["value"] = "shared"
    changed["actions"][1]["value"] = "shared"
    changed["version_history"][0]["entries"][0]["value"] = "shared"
    changed["version_history"][1]["entries"][0]["value"] = "shared"
    changed["queries"][0] = {
        "query_id": "q",
        "query_type": "multi_object_current",
        "text": "?",
        "selector": {
            "kind": "multi_object_current",
            "object_keys": [key_a, key_b],
        },
        "target_object_keys": [key_a, key_b],
        "answer_schema": "list",
        "evaluation_mode": "state_direct",
    }
    changed["gold_evidence"][0] = {
        "query_id": "q",
        "answer": ["shared", None],
        "supporting_object_keys": [key_a, key_b],
        "supporting_event_ids": ["e0", "e2"],
        "derivation_steps": [{
            "step_id": "read",
            "operation": "read",
            "supporting_object_keys": [key_a, key_b],
            "supporting_event_ids": ["e0", "e2"],
        }],
        "final_derivation_step_id": "read",
    }
    return changed


def _task_with_family(task, family):
    data = task.model_dump(mode="python")
    data["task_family"] = family
    return MemUpdateTaskV3.model_validate(data)


def _score_lifecycle_wiring(task, entries, prediction):
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, ParserExtractorProvenanceV3, RetrievalTraceV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    trace = RetrievalTraceV3(query_id="q", retrieved_entries=tuple(entries))
    answer = AnswerPredictionV3(
        query_id="q", raw_output=json.dumps(prediction), parsed_answer=prediction, format_valid=True,
    )
    run = TaskRunRecordV3(
        task_id=task.task_id, adapter_id="adapter", run_id=f"lifecycle-{task.task_id}",
        retrieval_traces=(trace,), answer_predictions=(answer,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1",
            memory_entry_extractor_version="1", redaction_policy_version="1",
        ),
        completion_status="completed",
    )
    info = AdapterInfoV3(
        adapter_id="adapter", adapter_version="1", system_name="system",
        system_version="1", configuration_hash=H,
    )
    caps = AdapterCapabilitiesV3(
        supports_delete=True, supports_native_answer=True, exports_entries=True,
        exports_object_keys=True, exports_values=True, exports_retrieval_ids=True,
    )
    config = ScorerConfigV3(requested_metric_fields=_LIFECYCLE_METRICS)
    return score_task_v3(task, run, authenticated_context(task, run, info, caps, config))


@pytest.mark.parametrize(
    ("family", "metric_path", "failure_flag"),
    (
        ("F", "answer_scores.stale_copied", "stale_copied"),
        (
            "E",
            "deletion_scores.forgotten_value_leakage_rate",
            "forgotten_value_exposed",
        ),
    ),
)
def test_raw_value_collision_scores_missing_artifact_without_failure_flag(
    family, metric_path, failure_flag
):
    from mub.vnext.contracts.enums import SupportReason

    task = MemUpdateTaskV3.model_validate(
        _two_object_shared_value_collision_payload()
    )
    score = _score_lifecycle_wiring(
        _task_with_family(task, family), (), "shared"
    )
    layer, leaf = metric_path.split(".", 1)

    assert getattr(getattr(score, layer), leaf) is None
    support = score.supported_metric_fields[metric_path]
    assert support.reason is SupportReason.MISSING_ARTIFACT
    assert "query targets" in support.detail
    assert failure_flag not in score.failure_flags


def test_lifecycle_metrics_and_flags_are_scoped_to_each_query_targets():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    scores = {}
    for query_entity in ("A", "B"):
        base_task = MemUpdateTaskV3.model_validate(_two_object_lifecycle_payload(query_entity))
        for family in ("E", "F"):
            task = _task_with_family(base_task, family)
            b_key = next(key for key in task.target_objects if key.entity == "B")
            entry = MemoryEntryRecordV3(
                entry_id=f"b-old-for-{query_entity}-{family}", content="b-deleted",
                object_key_candidate=b_key, value_candidate="b-deleted",
                version_index=0, source_event_ids=("e1",),
            )
            scores[query_entity, family] = _score_lifecycle_wiring(task, (entry,), "b-deleted")

    unrelated_stale = scores["A", "F"]
    assert unrelated_stale.retrieval_scores.stale_exposure_rate == 0.0
    assert unrelated_stale.retrieval_scores.stale_count_in_context == 0
    assert unrelated_stale.answer_scores.stale_copied == 0.0
    assert not {"stale_retrieved", "stale_copied", "forgotten_value_exposed"} & set(unrelated_stale.failure_flags)

    unrelated_forgotten = scores["A", "E"]
    assert unrelated_forgotten.deletion_scores.forgotten_exposure_rate is None
    assert unrelated_forgotten.deletion_scores.forgotten_value_leakage_rate is None
    assert unrelated_forgotten.supported_metric_fields["deletion_scores.forgotten_exposure_rate"].reason is SupportReason.NOT_APPLICABLE
    assert unrelated_forgotten.supported_metric_fields["deletion_scores.forgotten_value_leakage_rate"].reason is SupportReason.NOT_APPLICABLE
    assert "forgotten_value_exposed" not in unrelated_forgotten.failure_flags

    targeted_stale = scores["B", "F"]
    assert targeted_stale.retrieval_scores.stale_exposure_rate == 1.0
    assert targeted_stale.retrieval_scores.stale_count_in_context == 1
    assert targeted_stale.answer_scores.stale_copied == 1.0
    assert {"stale_retrieved", "stale_copied", "forgotten_value_exposed"} <= set(targeted_stale.failure_flags)

    targeted_forgotten = scores["B", "E"]
    assert targeted_forgotten.deletion_scores.forgotten_exposure_rate == 1.0
    assert targeted_forgotten.deletion_scores.forgotten_value_leakage_rate == 1.0
    assert "forgotten_value_exposed" in targeted_forgotten.failure_flags


def test_relearned_identical_value_is_obsolete_but_not_stale_or_forgotten():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3
    from mub.vnext.scoring.lifecycle_v3 import TargetLifecycleClassifierV3

    changed = current_structured_payload("x", "string")
    changed["task_family"] = "E"
    changed["actions"][0]["value"] = "x"
    changed["version_history"][0]["entries"][0]["value"] = "x"
    changed["events"] = [changed["events"][index] for index in (0, 2, 3)]
    for sequence_index, event in enumerate(changed["events"]):
        event["sequence_index"] = sequence_index
    changed["actions"] = [changed["actions"][index] for index in (0, 2, 3)]
    history = changed["version_history"][0]["entries"]
    changed["version_history"][0]["entries"] = [history[index] for index in (0, 2, 3)]
    changed["version_history"][0]["entries"][0]["valid_until_event_id"] = "e2"
    for version_index, version in enumerate(changed["version_history"][0]["entries"]):
        version["version_index"] = version_index
    task = MemUpdateTaskV3.model_validate(changed)
    replay = replay_task_v3(task)
    entry = MemoryEntryRecordV3(
        entry_id="old-explicit-x", content="x", object_key_candidate=task.target_objects[0],
        value_candidate="x", version_index=0, source_event_ids=("e0",),
    )
    lifecycle = TargetLifecycleClassifierV3.for_query(task.queries[0], replay).classify_entry(entry)
    assert lifecycle.obsolete is True
    assert lifecycle.stale is False
    assert lifecycle.forgotten is False

    stale_score = _score_lifecycle_wiring(_task_with_family(task, "F"), (entry,), "x")
    assert stale_score.retrieval_scores.stale_exposure_rate == 0.0
    assert stale_score.retrieval_scores.stale_count_in_context == 0
    assert stale_score.answer_scores.stale_copied == 0.0
    assert not {"stale_retrieved", "stale_copied", "forgotten_value_exposed"} & set(stale_score.failure_flags)

    forgotten_score = _score_lifecycle_wiring(_task_with_family(task, "E"), (entry,), "x")
    assert forgotten_score.deletion_scores.forgotten_exposure_rate is None
    assert forgotten_score.deletion_scores.forgotten_value_leakage_rate is None
    assert forgotten_score.supported_metric_fields["deletion_scores.forgotten_exposure_rate"].reason is SupportReason.NOT_APPLICABLE
    assert "forgotten_value_exposed" not in forgotten_score.failure_flags


def test_future_ttl_tombstone_does_not_create_forgotten_applicability_or_exposure():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    changed = ttl_horizon_payload("005", "v1")
    changed["task_family"] = "E"
    changed["queries"][0]["query_type"] = "current"
    changed["queries"][0]["selector"] = {"kind": "current"}
    task = MemUpdateTaskV3.model_validate(changed)
    entry = MemoryEntryRecordV3(
        entry_id="pre-expiry-current", content="v1", object_key_candidate=task.target_objects[0],
        value_candidate="v1", version_index=1, source_event_ids=("e1",),
    )

    score = _score_lifecycle_wiring(task, (entry,), "v1")
    for path in (
        "deletion_scores.forgotten_exposure_rate",
        "deletion_scores.forgotten_value_leakage_rate",
    ):
        assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE
    assert score.deletion_scores.forgotten_exposure_rate is None
    assert score.deletion_scores.forgotten_value_leakage_rate is None
    assert "forgotten_value_exposed" not in score.failure_flags


def test_nested_unhashable_values_and_bool_int_distinction_reach_lifecycle_scores():
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    old_value = {"nested": [1, {"flag": 1}]}
    current_value = {"nested": [True, {"flag": True}]}
    changed = current_structured_payload(current_value, "object")
    changed["task_family"] = "E"
    changed["actions"][0]["value"] = old_value
    changed["version_history"][0]["entries"][0]["value"] = old_value
    task = MemUpdateTaskV3.model_validate(changed)
    entry = MemoryEntryRecordV3(
        entry_id="nested-old", content="nested", object_key_candidate=task.target_objects[0],
        value_candidate=old_value, version_index=0, source_event_ids=("e0",),
    )

    stale_score = _score_lifecycle_wiring(_task_with_family(task, "F"), (entry,), old_value)
    assert stale_score.retrieval_scores.stale_exposure_rate == 1.0
    assert stale_score.retrieval_scores.stale_count_in_context == 1
    assert stale_score.answer_scores.stale_copied == 1.0
    assert {"stale_retrieved", "stale_copied", "forgotten_value_exposed"} <= set(stale_score.failure_flags)

    forgotten_score = _score_lifecycle_wiring(_task_with_family(task, "E"), (entry,), old_value)
    assert forgotten_score.deletion_scores.forgotten_exposure_rate == 1.0
    assert forgotten_score.deletion_scores.forgotten_value_leakage_rate == 1.0
    assert "forgotten_value_exposed" in forgotten_score.failure_flags


def test_definite_and_ambiguous_target_entries_fail_closed_without_lifecycle_flags():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    changed = current_structured_payload("x", "string")
    changed["task_family"] = "E"
    changed["actions"][0]["value"] = "x"
    changed["version_history"][0]["entries"][0]["value"] = "x"
    task = MemUpdateTaskV3.model_validate(changed)
    key = task.target_objects[0]
    entries = (
        MemoryEntryRecordV3(
            entry_id="definite-stale-forgotten", content="v1", object_key_candidate=key,
            value_candidate="v1", version_index=1, source_event_ids=("e1",),
        ),
        MemoryEntryRecordV3(
            entry_id="ambiguous-repeated-x", content="x", object_key_candidate=key,
            value_candidate="x",
        ),
    )

    stale_score = _score_lifecycle_wiring(_task_with_family(task, "F"), entries, "x")
    for path in (
        "retrieval_scores.stale_exposure_rate",
        "retrieval_scores.stale_count_in_context",
    ):
        assert getattr(getattr(stale_score, path.split(".")[0]), path.split(".")[1]) is None
        assert stale_score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT
    assert "stale_retrieved" not in stale_score.failure_flags

    forgotten_score = _score_lifecycle_wiring(_task_with_family(task, "E"), entries, "x")
    path = "deletion_scores.forgotten_exposure_rate"
    assert forgotten_score.deletion_scores.forgotten_exposure_rate is None
    assert forgotten_score.supported_metric_fields[path].reason is SupportReason.MISSING_ARTIFACT
    assert "forgotten_value_exposed" not in forgotten_score.failure_flags


def test_forgotten_entry_exposure_remains_applicable_when_relearn_masks_value_leakage():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

    changed = current_structured_payload("y", "string")
    changed["task_family"] = "E"
    changed["actions"][0]["value"] = "x"
    changed["actions"][1]["operation"] = "DELETE"
    changed["actions"][1].pop("value")
    changed["actions"][2]["operation"] = "ADD"
    changed["actions"][2]["value"] = "x"
    changed["actions"][3]["operation"] = "UPDATE"
    history = changed["version_history"][0]["entries"]
    for entry, (status, value) in zip(
        history,
        (("present", "x"), ("tombstone", None), ("present", "x"), ("present", "y")),
    ):
        entry["status"] = status
        if value is None:
            entry.pop("value", None)
        else:
            entry["value"] = value
    task = MemUpdateTaskV3.model_validate(changed)
    old_entry = MemoryEntryRecordV3(
        entry_id="old-x-before-delete", content="x",
        object_key_candidate=task.target_objects[0], value_candidate="x",
        version_index=0, source_event_ids=("e0",),
    )

    score = _score_lifecycle_wiring(task, (old_entry,), "y")
    assert score.deletion_scores.forgotten_exposure_rate == 1.0
    assert score.deletion_scores.forgotten_value_leakage_rate is None
    assert score.supported_metric_fields[
        "deletion_scores.forgotten_value_leakage_rate"
    ].reason is SupportReason.NOT_APPLICABLE
    assert "forgotten_value_exposed" in score.failure_flags


def test_clean_not_supported_preserves_requested_observability_metrics():
    from mub.vnext.contracts.enums import SupportReason
    from mub.vnext.contracts.v3.runtime import ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id,
            event_id=action.event_id,
            operation=action.operation,
            observed_scope=action.scope,
            target_object_keys=action.target_object_keys,
            value=action.value,
            format_valid=True,
            execution_status="not_supported",
            fallback_used=False,
            raw_output="unsupported",
        )
        for action in task.actions
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id="not-supported-observable",
        parsed_actions=parsed_actions,
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status="not_supported",
        exceptions=(),
    )
    info = AdapterInfoV3(
        adapter_id="adapter",
        adapter_version="1",
        system_name="system",
        system_version="1",
        configuration_hash=H,
    )
    capabilities = AdapterCapabilitiesV3(
        supports_isolated_reset=True,
        supports_event_ingest=True,
        supports_add=True,
        supports_update=True,
        supports_delete=True,
        supports_historical_query=True,
        exports_version_history=True,
        exports_entries=True,
        exports_object_keys=True,
        exports_values=True,
        exports_action_trace=True,
        exports_retrieval_ids=True,
        exports_source_event_ids=True,
    )
    requested = (
        "protocol_scores.unsupported_operation_rate",
        "system_scores.error_rate",
        "audit_scores.action_trace_available",
        "audit_scores.manifest_completeness",
        "audit_scores.source_provenance_coverage",
        "state_scores.final_state_accuracy",
    )
    context = authenticated_context(
        task,
        run,
        info,
        capabilities,
        ScorerConfigV3(requested_metric_fields=requested),
    )

    score = score_task_v3(task, run, context)

    assert score.protocol_scores.unsupported_operation_rate == 1.0
    assert score.system_scores.error_rate == 0.0
    assert score.audit_scores.action_trace_available is True
    assert score.audit_scores.manifest_completeness == 1.0
    for path in (
        "protocol_scores.unsupported_operation_rate",
        "system_scores.error_rate",
        "audit_scores.action_trace_available",
        "audit_scores.manifest_completeness",
    ):
        assert path not in score.supported_metric_fields
    assert score.audit_scores.source_provenance_coverage is None
    assert score.supported_metric_fields[
        "audit_scores.source_provenance_coverage"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert score.state_scores.final_state_accuracy is None
    assert score.supported_metric_fields[
        "state_scores.final_state_accuracy"
    ].reason is SupportReason.NOT_SUPPORTED
    assert score.audit_scores.state_export_available is None
    assert score.supported_metric_fields[
        "audit_scores.state_export_available"
    ].reason is SupportReason.NOT_APPLICABLE
    assert "unsupported_action" in score.failure_flags
    assert "system_exception" not in score.failure_flags


@pytest.mark.parametrize(
    ("completion_status", "has_exception", "expected_error_rate"),
    (
        ("not_supported", False, 0.0),
        ("not_supported", True, 1.0),
        ("failed", False, 1.0),
        ("partial", False, 1.0),
    ),
)
def test_system_error_rate_tracks_runtime_failure_lifecycle(
    completion_status, has_exception, expected_error_rate,
):
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id=f"lifecycle-{completion_status}-{has_exception}",
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status=completion_status,
        exceptions=({"type": "boom"},) if has_exception else (),
    )
    info = AdapterInfoV3(
        adapter_id="adapter",
        adapter_version="1",
        system_name="system",
        system_version="1",
        configuration_hash=H,
    )
    config = ScorerConfigV3(
        requested_metric_fields=("system_scores.error_rate",),
    )
    context = authenticated_context(task, run, info, AdapterCapabilitiesV3(), config)

    score = score_task_v3(task, run, context)

    assert score.system_scores.error_rate == expected_error_rate
    assert ("system_exception" in score.failure_flags) is (
        has_exception or completion_status in {"failed", "partial"}
    )


@pytest.mark.parametrize(
    ("completion_status", "has_exception", "expected_reason"),
    (
        ("not_supported", False, "not_supported"),
        ("not_supported", True, "runtime_failed"),
        ("failed", False, "runtime_failed"),
        ("partial", False, "runtime_failed"),
    ),
)
def test_system_performance_metrics_remain_terminally_gated(
    completion_status, has_exception, expected_reason,
):
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id,
            event_id=action.event_id,
            operation=action.operation,
            observed_scope=action.scope,
            target_object_keys=action.target_object_keys,
            value=action.value,
            format_valid=True,
            execution_status="not_supported",
            fallback_used=False,
            raw_output="unsupported",
            latency_ms=12.5,
        )
        for action in task.actions
    )
    prediction = AnswerPredictionV3(
        query_id="q",
        raw_output="answer",
        parsed_answer=["v0", "v1", None, "v2"],
        format_valid=True,
        latency_ms=8.5,
        usage={"total_tokens": 9},
    )
    run = TaskRunRecordV3(
        task_id=task.task_id,
        adapter_id="adapter",
        run_id=f"system-performance-{completion_status}-{has_exception}",
        parsed_actions=parsed_actions,
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status=completion_status,
        exceptions=({"type": "boom"},) if has_exception else (),
    )
    info = AdapterInfoV3(
        adapter_id="adapter",
        adapter_version="1",
        system_name="system",
        system_version="1",
        configuration_hash=H,
    )
    capabilities = AdapterCapabilitiesV3(
        supports_event_ingest=True,
        supports_historical_query=True,
        reports_latency=True,
        reports_token_usage=True,
        reports_cost=True,
    )
    paths = (
        "system_scores.ingest_latency_ms",
        "system_scores.answer_latency_ms",
        "system_scores.token_usage",
        "system_scores.api_cost",
    )
    context = authenticated_context(
        task,
        run,
        info,
        capabilities,
        ScorerConfigV3(requested_metric_fields=paths),
    )

    score = score_task_v3(task, run, context)

    for path in paths:
        layer, leaf = path.split(".")
        assert getattr(getattr(score, layer), leaf) is None
        assert score.supported_metric_fields[path].reason.value == expected_reason
