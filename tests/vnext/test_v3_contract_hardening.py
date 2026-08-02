import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.adapter import (
    AdapterActionPayloadV3,
    AdapterActionResultV3,
    AdapterAnswerResultV3,
    AdapterCapabilitiesV3,
    AdapterInfoV3,
    ExportEntriesResultV3,
    ExportStateResultV3,
    ExportedEventAnchorV3,
    ExportedVersionRecordV3,
    MemoryAdapterV3,
    ObjectVersionHistoryV3,
    ResetRequestV3,
    ResetResultV3,
    RetrievalRequestV3,
    RetrievalResultV3,
    VersionHistoryExportRequestV3,
    VersionHistoryExportResultV3,
)
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, ParsedManagerActionV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    DerivationStepV3,
    GoldActionV3,
    MemoryQueryV3,
    QueryGoldEvidenceV3,
    VersionHistoryEntry,
    VersionHistoryLedger,
)


def key(entity="e", object_type="slot"):
    return MemoryObjectKey(object_type=object_type, namespace="n", entity=entity, attribute="a")


def entry(index=0, value="x", start="e0", end=None, source=None):
    return VersionHistoryEntry(version_index=index, status="present", value=value, valid_from_event_id=start, valid_until_event_id=end, source_event_ids=(source or start,))


def test_semantic_hash_only_ignores_key_classification_object_type() -> None:
    base = VersionHistoryLedger(object_key=key(object_type="slot"), entries=(entry(value={"object_type": "application-a"}),))
    reclassified = VersionHistoryLedger(object_key=key(object_type="profile"), entries=base.entries)
    changed_value = VersionHistoryLedger(object_key=key(), entries=(entry(value={"object_type": "application-b"}),))
    assert base.semantic_hash == reclassified.semantic_hash
    assert base.semantic_hash != changed_value.semantic_hash


def test_nested_v3_state_is_deeply_immutable() -> None:
    ledger = VersionHistoryLedger(object_key=key(), entries=(entry(value={"nested": [1]}),))
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        ledger.object_key.entity = "changed"
    with pytest.raises((TypeError, AttributeError)):
        ledger.entries[0].value["nested"].append(2)

    prediction = AnswerPredictionV3(query_id="q", raw_output="1", parsed_answer=1, format_valid=True, usage={"tokens": 1})
    with pytest.raises((TypeError, AttributeError)):
        prediction.usage["tokens"] = 2


def test_gold_actions_use_immutable_targets_and_shared_scope_rules() -> None:
    action = GoldActionV3(action_id="a", event_id="e0", operation="DELETE", scope="object", target_object_keys=(key(),))
    with pytest.raises((TypeError, AttributeError)):
        action.target_object_keys.append(key("other"))
    with pytest.raises(ValidationError):
        GoldActionV3(action_id="a", event_id="e0", operation="DELETE", scope="object", target_object_keys=(key(), key("other")))
    with pytest.raises(ValidationError):
        GoldActionV3(action_id="a", event_id="e0", operation="UPDATE", scope="entity", target_object_keys=(key(),), value="x")


def test_runtime_executed_actions_require_coherent_operation_scope_and_payload() -> None:
    common = dict(event_id="e0", format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
    with pytest.raises(ValidationError):
        ParsedManagerActionV3(**common)
    with pytest.raises(ValidationError):
        ParsedManagerActionV3(**common, operation="DELETE", observed_scope="namespace", target_object_keys=(key(), key("other")), value="forbidden")


def test_adapter_action_result_is_strict_frozen_and_coherent() -> None:
    requested = AdapterActionPayloadV3(operation="UPDATE", scope="object", target_object_keys=(key(),), value="requested")
    effective = AdapterActionPayloadV3(operation="NOOP")
    result = AdapterActionResultV3(event_id="e0", requested_action=requested, effective_action=effective, execution_status="no_effect", reason="adapter_no_effect", raw_result={"nested": [1]})
    assert result.requested_action.operation.value == "UPDATE"
    assert result.effective_action.operation.value == "NOOP"
    with pytest.raises((TypeError, AttributeError)):
        result.raw_result["nested"].append(2)
    different = AdapterActionResultV3(
        event_id="e0",
        requested_action=requested,
        effective_action=AdapterActionPayloadV3(operation="ADD", scope="object", target_object_keys=(key(),), value="effective"),
        execution_status="executed",
        affected_entry_ids=("entry",),
    )
    assert different.requested_action.value != different.effective_action.value
    for changes in ({"event_id": ""}, {"affected_entry_ids": ("",)}, {"affected_entry_ids": ("id", "id")}):
        data = result.model_dump(mode="python")
        data.update(changes)
        with pytest.raises(ValidationError):
            AdapterActionResultV3.model_validate(data)
    with pytest.raises(ValidationError):
        AdapterActionPayloadV3(operation="NOOP", scope="object", target_object_keys=(key(),))
    with pytest.raises(ValidationError):
        AdapterActionPayloadV3(operation="DELETE", scope="object", target_object_keys=(key(),), value="mixed")

    parsed = result.to_parsed_manager_action(raw_output="noop", format_valid=True, fallback_used=False)
    assert parsed.execution_status.value == "no_effect" and parsed.operation.value == "UPDATE"
    assert parsed.value == "requested" and parsed.target_object_keys[0].canonical_id == key().canonical_id
    for status in ("rejected", "not_supported"):
        rejected = AdapterActionResultV3(event_id="e0", requested_action=requested, execution_status=status, reason="policy")
        assert rejected.effective_action.operation is None
        converted = rejected.to_parsed_manager_action(raw_output=status, format_valid=True, fallback_used=False)
        assert converted.execution_status.value == status
        assert converted.operation.value == "UPDATE" and converted.value == "requested"
    failed = AdapterActionResultV3(event_id="e0", requested_action=requested, execution_status="failed", error={"code": "boom"})
    assert failed.error == {"code": "boom"}
    failed_parsed = failed.to_parsed_manager_action(raw_output="failed", format_valid=True, fallback_used=False)
    assert failed_parsed.execution_status.value == "failed" and failed_parsed.operation.value == "UPDATE"
    noop = AdapterActionResultV3(event_id="e0", requested_action={"operation": "NOOP"}, effective_action={"operation": "NOOP"}, execution_status="executed")
    assert noop.execution_status.value == "executed" and noop.affected_entry_ids == ()
    with pytest.raises(ValidationError, match="affected|NOOP"):
        AdapterActionResultV3(event_id="e0", requested_action={"operation": "NOOP"}, effective_action={"operation": "NOOP"}, execution_status="executed", affected_entry_ids=("impossible",))
    with pytest.raises(ValidationError):
        AdapterActionResultV3(event_id="e0", requested_action=requested, effective_action=effective, execution_status="executed")
    with pytest.raises(ValidationError):
        AdapterActionResultV3(event_id="e0", requested_action=requested, execution_status="failed")


def test_nonfinite_values_fail_during_contract_construction() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            entry(value={"nested": [bad]})
        with pytest.raises(ValidationError):
            AnswerPredictionV3(query_id="q", raw_output="x", parsed_answer={"nested": bad}, format_valid=True)
        with pytest.raises(ValidationError):
            AdapterActionResultV3(event_id="e0", execution_status="failed", error="failure", raw_result={"nested": bad})
        with pytest.raises(ValidationError):
            RetrievalTraceV3(query_id="q", scores=(bad,))


def test_retrieval_scores_require_exact_builtin_floats() -> None:
    class HostileFloat(float):
        pass

    item = MemoryEntryRecordV3(entry_id="id", content="x")
    for bad in (1, HostileFloat(1.0)):
        with pytest.raises(ValidationError):
            RetrievalTraceV3(query_id="q", retrieved_entries=(item,), scores=(bad,))
    assert RetrievalTraceV3(query_id="q", retrieved_entries=(item,), scores=(1.0,)).scores == (1.0,)


def test_identifiers_reject_whitespace_and_string_subclasses() -> None:
    class HostileString(str):
        pass

    for bad in (" ", HostileString("id")):
        with pytest.raises(ValidationError):
            GoldActionV3(action_id=bad, event_id="e0", operation="NOOP")
        with pytest.raises(ValidationError):
            ParsedManagerActionV3(event_id=bad, format_valid=False, execution_status="failed", fallback_used=False, raw_output="")
        with pytest.raises(ValidationError):
            AdapterActionResultV3(event_id=bad, execution_status="failed", error="failure")
        with pytest.raises(ValidationError):
            AnswerPredictionV3(query_id=bad, raw_output="", disposition="unavailable", format_valid=False)


def test_adapter_requests_and_exported_history_are_strict_and_frozen() -> None:
    class HostileInt(int):
        pass

    reset = ResetRequestV3(namespace="n", config={"nested": [1]})
    with pytest.raises((TypeError, AttributeError)):
        reset.config["nested"].append(2)
    query = MemoryQueryV3(query_id="q", query_type="update_sensitive_multi_hop", text="question", selector=CurrentSelector(), target_object_keys=(key(),), answer_schema="string", evaluation_mode="state_direct", synthesis={"kind": "update_sensitive_multi_hop", "minimum_hops": 2})
    request = RetrievalRequestV3(query=query, k=1, options={"nested": [1]})
    assert request.query.selector.kind == "current"
    assert request.query.target_object_keys[0].canonical_id == key().canonical_id
    assert request.query.answer_schema.value == "string"
    assert request.query.synthesis.kind == "update_sensitive_multi_hop"
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        request.query.text = "mutated"
    with pytest.raises((TypeError, AttributeError)):
        request.options["nested"].append(2)
    with pytest.raises(ValidationError, match="match bound request"):
        RetrievalResultV3(request=request, trace=RetrievalTraceV3(query_id="other"))
    bound = RetrievalResultV3(request=request, trace=RetrievalTraceV3(query_id="q"))
    round_trip = RetrievalResultV3.model_validate(bound.model_dump(mode="python"))
    assert round_trip == bound
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        bound.request.query.text = "mutated"
    for bad in (True, 1.0, HostileInt(1)):
        with pytest.raises(ValidationError):
            RetrievalRequestV3(query=query, k=bad)
    bad_query = query.model_dump(mode="python")
    bad_query["query_id"] = " "
    for model, data in (
        (ResetRequestV3, {"namespace": " ", "config": {}}),
        (RetrievalRequestV3, {"query": bad_query, "k": 1}),
        (VersionHistoryExportRequestV3, {"namespace": " "}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(data)

    e0 = ExportedEventAnchorV3(event_id="e0", sequence_index=0)
    e1 = ExportedEventAnchorV3(event_id="e1", sequence_index=1)
    e2 = ExportedEventAnchorV3(event_id="e2", sequence_index=2)
    e3 = ExportedEventAnchorV3(event_id="e3", sequence_index=3)
    present = ExportedVersionRecordV3(version_index=0, status="present", value={"nested": [1]}, valid_from=e0, valid_until=e1, source_anchors=(e0,))
    tombstone = ExportedVersionRecordV3(version_index=1, status="tombstone", valid_from=e1, valid_until=e2, source_anchors=(e1,))
    history = ObjectVersionHistoryV3(object_key=key(), versions=(present, tombstone))
    result = VersionHistoryExportResultV3(histories=(history,))
    with pytest.raises((TypeError, AttributeError)):
        result.histories[0].versions[0].value["nested"].append(2)
    with pytest.raises(ValidationError):
        ExportedVersionRecordV3(version_index=0, status="tombstone", value="forbidden", valid_from=e0, valid_until=e1, source_anchors=(e0,))
    with pytest.raises(ValidationError):
        ExportedVersionRecordV3(version_index=0, status="present", value="x", valid_from=e2, valid_until=e1, source_anchors=(e1,))
    open_final = ExportedVersionRecordV3(version_index=0, status="present", value="current", valid_from=e2, source_anchors=(e2,))
    assert ObjectVersionHistoryV3(object_key=key(), versions=(open_final,)).versions[-1].valid_until is None
    partial_nonfinal = ExportedVersionRecordV3(version_index=0, status="present", value="x", valid_from=e0, source_anchors=(e0,))
    with pytest.raises(ValidationError, match="final"):
        ObjectVersionHistoryV3(object_key=key(), versions=(partial_nonfinal, tombstone))
    with pytest.raises(ValidationError):
        ObjectVersionHistoryV3(object_key=key(), versions=(present, tombstone.model_copy(update={"valid_from": e2, "valid_until": e3})))
    conflicting_e1 = ExportedEventAnchorV3(event_id="other-e1", sequence_index=1)
    conflicting = tombstone.model_copy(update={"valid_from": conflicting_e1})
    with pytest.raises(ValidationError, match="sequence_index|continuous"):
        ObjectVersionHistoryV3(object_key=key(), versions=(present, conflicting))
    logical_z = ExportedVersionRecordV3(version_index=0, status="present", value="x", logical_time="z", source_anchors=(e0,))
    logical_a = ExportedVersionRecordV3(version_index=1, status="present", value="y", logical_time="a", source_anchors=(e1,))
    with pytest.raises(ValidationError):
        ObjectVersionHistoryV3(object_key=key(), versions=(logical_z, logical_a))
    logical_equal_0 = logical_z.model_copy(update={"logical_time": "same"})
    logical_equal_1 = logical_a.model_copy(update={"logical_time": "same"})
    with pytest.raises(ValidationError, match="strict|logical"):
        ObjectVersionHistoryV3(object_key=key(), versions=(logical_equal_0, logical_equal_1))

    source_a5 = ExportedEventAnchorV3(event_id="sa5", sequence_index=0, logical_time="a5")
    source_b = ExportedEventAnchorV3(event_id="sb", sequence_index=1, logical_time="b")
    source_z = ExportedEventAnchorV3(event_id="sz", sequence_index=0, logical_time="z")
    logical_a0 = ExportedVersionRecordV3(version_index=0, status="present", value="x", logical_time="a", source_anchors=(source_a5,))
    logical_b0 = ExportedVersionRecordV3(version_index=1, status="present", value="y", logical_time="b", source_anchors=(source_b,))
    assert len(ObjectVersionHistoryV3(object_key=key(), versions=(logical_a0, logical_b0)).versions) == 2
    with pytest.raises(ValidationError, match="half-open|logical"):
        ObjectVersionHistoryV3(object_key=key(), versions=(logical_a0.model_copy(update={"source_anchors": (source_z,)}), logical_b0))
    with pytest.raises(ValidationError, match="half-open|logical"):
        ObjectVersionHistoryV3(object_key=key(), versions=(logical_a0.model_copy(update={"source_anchors": (source_b,)}), logical_b0))
    reversed_next = logical_b0.model_copy(update={"source_anchors": (ExportedEventAnchorV3(event_id="reversed", sequence_index=0, logical_time="b"),)})
    with pytest.raises(ValidationError, match="sequence|precede"):
        ObjectVersionHistoryV3(object_key=key(), versions=(logical_a0, reversed_next))
    with pytest.raises(ValidationError):
        ExportedVersionRecordV3(version_index=0, status="present", value="x", valid_from=e0, valid_until=e2, source_anchors=(e1, e0))
    logical_from = ExportedEventAnchorV3(event_id="l0", sequence_index=0, logical_time="b")
    logical_until = ExportedEventAnchorV3(event_id="l2", sequence_index=2, logical_time="d")
    logical_outside = ExportedEventAnchorV3(event_id="l1", sequence_index=1, logical_time="a")
    with pytest.raises(ValidationError, match="logical-time interval"):
        ExportedVersionRecordV3(version_index=0, status="present", value="x", valid_from=logical_from, valid_until=logical_until, source_anchors=(logical_outside,))
    with pytest.raises(ValidationError):
        ObjectVersionHistoryV3(object_key=key(), versions=(present, present.model_copy(update={"version_index": 2})))


def test_history_export_enforces_global_anchor_bijection_across_objects() -> None:
    shared0 = ExportedEventAnchorV3(event_id="shared", sequence_index=0)
    shared9 = ExportedEventAnchorV3(event_id="shared", sequence_index=9)
    other0 = ExportedEventAnchorV3(event_id="other", sequence_index=0)

    def history(entity, anchor):
        version = ExportedVersionRecordV3(version_index=0, status="present", value=entity, valid_from=anchor, source_anchors=(anchor,))
        return ObjectVersionHistoryV3(object_key=key(entity), versions=(version,))

    first = history("one", shared0)
    assert len(VersionHistoryExportResultV3(histories=(first, history("two", shared0))).histories) == 2
    with pytest.raises(ValidationError, match="global|consistent|event"):
        VersionHistoryExportResultV3(histories=(first, history("two", shared9)))
    with pytest.raises(ValidationError, match="sequence_index|global|event"):
        VersionHistoryExportResultV3(histories=(first, history("two", other0)))
    shared0_timed = ExportedEventAnchorV3(event_id="shared", sequence_index=0, logical_time="t0")
    with pytest.raises(ValidationError, match="logical|consistent|global"):
        VersionHistoryExportResultV3(histories=(first, history("two", shared0_timed)))


def test_memory_adapter_v3_protocol_exposes_all_typed_capability_paths() -> None:
    class Fixture:
        def adapter_info(self): return AdapterInfoV3(adapter_id="a", adapter_version="1", system_name="s", system_version="1", configuration_hash="a" * 64)
        def capabilities(self): return AdapterCapabilitiesV3(supports_isolated_reset=True, exports_entries=True, exports_raw_state=True, exports_retrieval_ids=True, exports_version_history=True)
        def reset(self, request): return ResetResultV3(success=True, namespace=request.namespace)
        def ingest_event(self, event): return AdapterActionResultV3(event_id=event.event_id, requested_action={"operation": "NOOP"}, effective_action={"operation": "NOOP"}, execution_status="executed")
        def export_entries(self): return ExportEntriesResultV3(entries=())
        def export_raw_state(self): return ExportStateResultV3(raw_state={})
        def export_version_history(self, request): return VersionHistoryExportResultV3(histories=())
        def retrieve(self, request): return RetrievalResultV3(request=request, trace=RetrievalTraceV3(query_id=request.query.query_id))
        def answer(self, query, mode): return AdapterAnswerResultV3(prediction=AnswerPredictionV3(query_id=query.query_id, raw_output="", disposition="unavailable", format_valid=False))
        def close(self): return None

    fixture = Fixture()
    assert isinstance(fixture, MemoryAdapterV3)
    reset_request = ResetRequestV3(namespace="n", config={"nested": [1]})
    assert fixture.reset(reset_request).success
    assert fixture.export_entries().entries == ()
    assert fixture.export_raw_state().raw_state == {}
    assert fixture.export_version_history(VersionHistoryExportRequestV3(namespace="n")).histories == ()
    query = MemoryQueryV3(query_id="q", query_type="current", text="?", selector=CurrentSelector(), target_object_keys=(key(),), answer_schema="string", evaluation_mode="state_direct")
    assert fixture.retrieve(RetrievalRequestV3(query=query, k=1)).trace.query_id == "q"


def test_derivation_steps_must_be_topologically_ordered() -> None:
    with pytest.raises(ValidationError, match="topological"):
        QueryGoldEvidenceV3(query_id="q", answer="x", supporting_object_keys=(key(),), supporting_event_ids=("e0",), derivation_steps=(
            DerivationStepV3(step_id="answer", operation="answer", input_step_ids=("read",)),
            DerivationStepV3(step_id="read", operation="read", supporting_event_ids=("e0",)),
        ), final_derivation_step_id="answer")
