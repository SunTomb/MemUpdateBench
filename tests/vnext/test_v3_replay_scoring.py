from copy import deepcopy

import pytest

from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.manifest import RunManifestV3
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
    context = VerifiedScoringContextV3.create_verified(
        adapter_info=info, capabilities=caps, scorer_config=ScorerConfigV3(), run_id="run",
    )
    with pytest.raises(ValueError, match="task_id mismatch"):
        score_task_v3(task, run, context)


def test_runtime_failure_precedes_capability_absence_in_support_map():
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    caps = AdapterCapabilitiesV3()
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    context = VerifiedScoringContextV3.create_verified(adapter_info=info, capabilities=caps, scorer_config=ScorerConfigV3(), run_id="run")
    run = TaskRunRecordV3(
        task_id="t", adapter_id="adapter", run_id="run",
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="failed", exceptions=({"type": "boom"},),
    )
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
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemorySnapshotV3, ParsedManagerActionV3
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3, score_task_v3

    task = MemUpdateTaskV3.model_validate(payload())
    caps = AdapterCapabilitiesV3(
        supports_isolated_reset=True, supports_event_ingest=True, supports_add=True,
        supports_update=True, supports_delete=True, supports_historical_query=True,
        exports_version_history=True, exports_entries=True, exports_object_keys=True,
        exports_values=True, exports_action_trace=True,
    )
    info = AdapterInfoV3(adapter_id="adapter", adapter_version="1", system_name="system", system_version="1", configuration_hash=H)
    context = VerifiedScoringContextV3.create_verified(adapter_info=info, capabilities=caps, scorer_config=ScorerConfigV3(), run_id="run")
    parsed_actions = tuple(
        ParsedManagerActionV3(
            event_id=action.event_id, operation=action.operation, observed_scope=action.scope,
            target_object_keys=action.target_object_keys, value=action.value, format_valid=True,
            execution_status="executed", fallback_used=False, raw_output="ok",
        )
        for action in task.actions
    )
    key = task.target_objects[0]
    run = TaskRunRecordV3(
        task_id=task.task_id, adapter_id="adapter", run_id="run", parsed_actions=parsed_actions,
        memory_snapshots=(MemorySnapshotV3(after_event_id="e3", state_by_object={key.canonical_id: "v2"}, store_size=1),),
        answer_predictions=(AnswerPredictionV3(query_id="q", raw_output="ok", parsed_answer=["v0", "v1", None, "v2"], format_valid=True),),
        parser_extractor_provenance=ParserExtractorProvenanceV3(action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1", redaction_policy_version="1"),
        completion_status="completed",
    )
    score = score_task_v3(task, run, context)
    observed = 0
    for path, descriptor in CORE_METRIC_REGISTRY_V3.items():
        if not descriptor.principal:
            continue
        layer, leaf = path.split(".", 1)
        value = getattr(getattr(score, layer), leaf)
        if value is None:
            continue
        observed += 1
        assert value == (0.0 if descriptor.direction == "lower" else 1.0), path
    assert observed >= 4


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
