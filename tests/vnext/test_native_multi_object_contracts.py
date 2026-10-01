from __future__ import annotations

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.enums import Operation
from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey
from mub.vnext.contracts.v3.enums import ExecutionStatusV3
from mub.vnext.contracts.v3.native_multi_object import (
    NativeAtomicityV3,
    NativeDataProvenanceV3,
    NativeMultiObjectMutationRequestV3,
    NativeMultiObjectMutationResultV3,
    NativeObjectMutationV3,
    NativeObjectOutcomeV3,
    NativeRetrievalEntryV3,
    NativeRetrievalTraceV3,
    NativeRetrievalRequestV3,
)


def key(namespace: str, entity: str, attribute: str, subkey: str | None = None, object_type: str = "slot"):
    return FrozenMemoryObjectKey(
        object_type=object_type,
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=subkey,
    )


def provenance(kind: NativeDataProvenanceV3 = NativeDataProvenanceV3.PROVIDER_NATIVE):
    return {"kind": kind, "provider_owned": kind is not NativeDataProvenanceV3.EXTRACTED}


def request(operation: Operation, objects: tuple[NativeObjectMutationV3, ...], event_id: str = "event-1"):
    return NativeMultiObjectMutationRequestV3(
        request_id=f"request-{event_id}",
        event_id=event_id,
        sequence_index=0,
        operation=operation,
        objects=objects,
        provenance=provenance(),
    )


def test_canonical_identity_uses_namespace_entity_attribute_subkey_and_ignores_object_type():
    left = key("ns-a", "same", "field", "one", object_type="profile")
    right = key("ns-a", "same", "field", "one", object_type="different-metadata")
    other_namespace = key("ns-b", "same", "field", "one")
    assert left == right
    assert left.canonical_id == right.canonical_id
    assert left != other_namespace
    assert left.canonical_id != other_namespace.canonical_id


def test_ordered_multi_object_mutation_requests_cover_add_update_delete_and_noop():
    objects = (
        NativeObjectMutationV3(object_key=key("ns-a", "e", "a", "one"), value="v1"),
        NativeObjectMutationV3(object_key=key("ns-b", "e", "a", "two"), value="v2"),
    )
    add = request(Operation.ADD, objects)
    update = request(Operation.UPDATE, objects, "event-2")
    delete = request(
        Operation.DELETE,
        tuple(NativeObjectMutationV3(object_key=item.object_key) for item in objects),
        "event-3",
    )
    noop = request(Operation.NOOP, (), "event-4")
    assert tuple(item.object_key.namespace for item in add.objects) == ("ns-a", "ns-b")
    assert update.operation is Operation.UPDATE
    assert delete.operation is Operation.DELETE
    assert noop.objects == ()


def test_mutation_values_distinguish_omission_from_explicit_json_null():
    object_key = key("ns", "e", "a")
    explicit_null = NativeObjectMutationV3(object_key=object_key, value=None)
    assert explicit_null.value is None
    assert explicit_null.value_supplied is True
    assert request(Operation.ADD, (explicit_null,)).objects[0].value_supplied is True
    assert request(Operation.UPDATE, (explicit_null,), "event-null").objects[0].value_supplied is True

    omitted = NativeObjectMutationV3(object_key=object_key)
    assert omitted.value is None
    assert omitted.value_supplied is False
    assert request(Operation.DELETE, (omitted,), "event-delete").objects[0].value_supplied is False

    with pytest.raises(ValidationError, match="requires a value"):
        request(Operation.ADD, (omitted,))
    with pytest.raises(ValidationError, match="requires a value"):
        request(Operation.UPDATE, (omitted,), "event-2")
    with pytest.raises(ValidationError, match="cannot carry values"):
        request(Operation.DELETE, (explicit_null,), "event-3")


def test_mutation_request_rejects_implicit_add_for_absent_update_and_noop_writes():
    obj = NativeObjectMutationV3(object_key=key("ns", "e", "a"), value="new")
    with pytest.raises(ValidationError, match="UPDATE"):
        request(Operation.UPDATE, (NativeObjectMutationV3(object_key=obj.object_key),))

    outcome = NativeObjectOutcomeV3(
        event_id="event-2",
        object_key=obj.object_key,
        operation=Operation.UPDATE,
        status=ExecutionStatusV3.NO_EFFECT,
        provider_entry_id=None,
        source_event_ids=("event-2",),
        write_count=0,
        reason="absent_update",
        provenance=provenance(),
    )
    result = NativeMultiObjectMutationResultV3(
        request_id="request-event-2",
        event_id="event-2",
        operation=Operation.UPDATE,
        outcomes=(outcome,),
        atomicity=NativeAtomicityV3.ATOMIC,
        provenance=provenance(),
    )
    assert result.outcomes[0].status is ExecutionStatusV3.NO_EFFECT
    assert result.outcomes[0].write_count == 0
    assert result.outcomes[0].provider_entry_id is None

    noop_outcome = NativeObjectOutcomeV3(
        event_id="event-3",
        object_key=key("ns", "e", "a"),
        operation=Operation.NOOP,
        status=ExecutionStatusV3.EXECUTED,
        source_event_ids=("event-3",),
        write_count=0,
        provenance=provenance(),
    )
    assert noop_outcome.write_count == 0


def test_partial_and_atomic_multi_object_results_validate_order_and_linkage():
    first = key("ns-a", "e", "a", "one")
    second = key("ns-b", "e", "a", "two")
    outcomes = (
        NativeObjectOutcomeV3(
            event_id="event-1", object_key=first, operation=Operation.ADD,
            status=ExecutionStatusV3.EXECUTED, provider_entry_id="p-1",
            source_event_ids=("event-1",), write_count=1, provenance=provenance(),
        ),
        NativeObjectOutcomeV3(
            event_id="event-1", object_key=second, operation=Operation.ADD,
            status=ExecutionStatusV3.FAILED, provider_entry_id=None,
            source_event_ids=("event-1",), write_count=0, error="provider_error",
            provenance=provenance(),
        ),
    )
    result = NativeMultiObjectMutationResultV3(
        request_id="request-event-1", event_id="event-1", operation=Operation.ADD,
        outcomes=outcomes, atomicity=NativeAtomicityV3.PARTIAL,
        provenance=provenance(),
    )
    assert result.atomicity is NativeAtomicityV3.PARTIAL
    with pytest.raises(ValidationError, match="atomicity"):
        NativeMultiObjectMutationResultV3(
            request_id="request-event-1", event_id="event-1", operation=Operation.ADD,
            outcomes=outcomes, atomicity=NativeAtomicityV3.ATOMIC,
            provenance=provenance(),
        )


def test_native_retrieval_requires_provider_ids_payloads_rank_order_scores_and_events():
    entries = (
        NativeRetrievalEntryV3(
            provider_entry_id="entry-2", object_key=key("ns-b", "e", "a", "two"),
            provider_payload={"value": "v2"}, rank=1, order_index=0, score=0.9,
            source_event_ids=("event-2",), provenance=provenance(),
        ),
        NativeRetrievalEntryV3(
            provider_entry_id="entry-1", object_key=key("ns-a", "e", "a", "one"),
            provider_payload={"value": "v1"}, rank=2, order_index=1, score=0.8,
            source_event_ids=("event-1",), provenance=provenance(),
        ),
    )
    trace = NativeRetrievalTraceV3(
        query=NativeRetrievalRequestV3(query_id="q-1", query_text="all", k=2, namespace="run"),
        entries=entries,
        provenance=provenance(),
    )
    assert tuple(entry.provider_entry_id for entry in trace.entries) == ("entry-2", "entry-1")
    assert tuple(entry.order_index for entry in trace.entries) == (0, 1)
    assert trace.entries[0].provenance.kind is NativeDataProvenanceV3.PROVIDER_NATIVE
    with pytest.raises(ValidationError, match="provider_entry_id"):
        NativeRetrievalEntryV3(
            object_key=key("ns", "e", "a"), provider_payload={}, rank=1,
            order_index=0, score=0.1, source_event_ids=("event",), provenance=provenance()
        )


def test_retrieval_request_rejects_hidden_target_injection_and_trace_reorders():
    with pytest.raises(ValidationError):
        NativeRetrievalRequestV3(
            query_id="q", query_text="visible", k=1, namespace="run",
            target_object_keys=(key("ns", "e", "a"),),
        )
    entry = NativeRetrievalEntryV3(
        provider_entry_id="entry", object_key=key("ns", "e", "a"),
        provider_payload={"value": "v"}, rank=2, order_index=1, score=0.1,
        source_event_ids=("event",), provenance=provenance(),
    )
    with pytest.raises(ValidationError, match="rank"):
        NativeRetrievalTraceV3(
            query=NativeRetrievalRequestV3(query_id="q", query_text="visible", k=1, namespace="run"),
            entries=(entry,), provenance=provenance(),
        )


def test_native_plan_rejects_custom_singleton_ids_and_frozen_prefixes():
    from scripts.vnext_plan_native_multi_object import plan_native_multi_object

    forbidden_ids = (
        "mem0_oss",
        "langgraph_store_custom_adapter",
        "langgraph_store_extract_then_store",
        "letta_profile",
        "letta_0_16_8_block_profile",
        "six_cell_factorial",
        "mem0_custom_v1",
        "langgraph_custom_v1",
        "letta_custom_v1",
        "reference_custom_v1",
        "six_cell_v1",
        "six-cell_v1",
        "factorial_v1",
        "frozen_v1",
        "p63_v1",
        "p83_v1",
        "p84_v1",
        "core_v1",
        "pilot_v1",
    )
    for candidate_id in forbidden_ids:
        with pytest.raises(ValueError, match="new candidate identity"):
            plan_native_multi_object(
                candidate_id=candidate_id,
                provider_name="qdrant",
                provider_version="1.0.0",
                source_revision="a" * 40,
                runtime_revision="b" * 40,
                source_hash="a" * 64,
                runtime_hash="b" * 64,
                configuration={"collection": "test"},
            )

    valid_plan = plan_native_multi_object(
        candidate_id="valid_native_candidate_v1",
        provider_name="qdrant",
        provider_version="1.0.0",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash="a" * 64,
        runtime_hash="b" * 64,
        configuration={"collection": "test"},
    )
    assert valid_plan.candidate_id == "valid_native_candidate_v1"


def test_native_plan_requires_explicit_sha256_hashes():
    from scripts.vnext_plan_native_multi_object import plan_native_multi_object

    with pytest.raises(ValueError, match="explicit actual source_hash"):
        plan_native_multi_object(
            candidate_id="valid_native_candidate_v1",
            provider_name="qdrant",
            provider_version="1.0.0",
            source_revision="a" * 40,
            runtime_revision="b" * 40,
            source_hash=None,
            runtime_hash="b" * 64,
            configuration={"collection": "test"},
        )

    with pytest.raises(ValueError, match="explicit actual runtime_hash"):
        plan_native_multi_object(
            candidate_id="valid_native_candidate_v1",
            provider_name="qdrant",
            provider_version="1.0.0",
            source_revision="a" * 40,
            runtime_revision="b" * 40,
            source_hash="a" * 64,
            runtime_hash=None,
            configuration={"collection": "test"},
        )


def test_native_plan_requires_explicit_sha256_hashes():
    from scripts.vnext_plan_native_multi_object import NativePlanV1, plan_native_multi_object

    explicit_hash = "c" * 64
    plan = plan_native_multi_object(
        candidate_id="valid_native_candidate_v2",
        provider_name="qdrant",
        provider_version="1.0.0",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash=explicit_hash,
        runtime_hash=explicit_hash,
        configuration={"collection": "test"},
    )
    assert plan.source_hash == explicit_hash
    assert plan.runtime_hash == explicit_hash

    # Check that NativePlanV1 does NOT derive source_hash from sha256(source_revision)
    raw_plan = NativePlanV1(
        candidate_id="valid_native_candidate_v3",
        provider_name="qdrant",
        provider_version="1.0.0",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash=explicit_hash,
        runtime_hash=explicit_hash,
        configuration={"collection": "test"},
        configuration_hash=plan.configuration_hash,
    )
    assert raw_plan.source_hash == explicit_hash
    assert raw_plan.runtime_hash == explicit_hash


def test_native_plan_factory_rejects_all_zero_identity_placeholders():
    from scripts.vnext_plan_native_multi_object import plan_native_multi_object

    base = {
        "candidate_id": "valid_native_candidate_v4",
        "provider_name": "qdrant",
        "provider_version": "1.0.0",
        "source_revision": "a" * 40,
        "runtime_revision": "b" * 40,
        "source_hash": "a" * 64,
        "runtime_hash": "b" * 64,
        "configuration": {"collection": "test"},
    }
    for field, placeholder in (
        ("source_revision", "0" * 40),
        ("runtime_revision", "0" * 40),
        ("source_hash", "0" * 64),
        ("runtime_hash", "0" * 64),
    ):
        arguments = {**base, field: placeholder}
        with pytest.raises(ValueError, match=field):
            plan_native_multi_object(**arguments)


def test_native_plan_direct_construction_rejects_all_zero_identity_placeholders():
    from scripts.vnext_plan_native_multi_object import NativePlanV1, plan_native_multi_object

    valid = plan_native_multi_object(
        candidate_id="valid_native_candidate_v5",
        provider_name="qdrant",
        provider_version="1.0.0",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash="c" * 64,
        runtime_hash="d" * 64,
        configuration={"collection": "test"},
    )
    base = {
        "candidate_id": "valid_native_candidate_v6",
        "provider_name": "qdrant",
        "provider_version": "1.0.0",
        "source_revision": "a" * 40,
        "runtime_revision": "b" * 40,
        "source_hash": "c" * 64,
        "runtime_hash": "d" * 64,
        "configuration": {"collection": "test"},
        "configuration_hash": valid.configuration_hash,
    }
    for field, placeholder in (
        ("source_revision", "0" * 40),
        ("runtime_revision", "0" * 40),
        ("source_hash", "0" * 64),
        ("runtime_hash", "0" * 64),
    ):
        arguments = {**base, field: placeholder}
        with pytest.raises(ValueError, match=field):
            NativePlanV1(**arguments)


def test_native_plan_factory_rejects_malformed_identity_values():
    from scripts.vnext_plan_native_multi_object import plan_native_multi_object

    base = {
        "candidate_id": "valid_native_candidate_v7",
        "provider_name": "qdrant",
        "provider_version": "1.0.0",
        "source_revision": "a" * 40,
        "runtime_revision": "b" * 40,
        "source_hash": "a" * 64,
        "runtime_hash": "b" * 64,
        "configuration": {"collection": "test"},
    }
    for field in ("source_revision", "runtime_revision", "source_hash", "runtime_hash"):
        arguments = {**base, field: "malformed"}
        with pytest.raises(ValueError, match=field):
            plan_native_multi_object(**arguments)


def test_native_plan_direct_construction_rejects_malformed_identity_values():
    from scripts.vnext_plan_native_multi_object import NativePlanV1, plan_native_multi_object

    valid = plan_native_multi_object(
        candidate_id="valid_native_candidate_v8",
        provider_name="qdrant",
        provider_version="1.0.0",
        source_revision="a" * 40,
        runtime_revision="b" * 40,
        source_hash="c" * 64,
        runtime_hash="d" * 64,
        configuration={"collection": "test"},
    )
    base = {
        "candidate_id": "valid_native_candidate_v9",
        "provider_name": "qdrant",
        "provider_version": "1.0.0",
        "source_revision": "a" * 40,
        "runtime_revision": "b" * 40,
        "source_hash": "c" * 64,
        "runtime_hash": "d" * 64,
        "configuration": {"collection": "test"},
        "configuration_hash": valid.configuration_hash,
    }
    for field in ("source_revision", "runtime_revision", "source_hash", "runtime_hash"):
        arguments = {**base, field: "malformed"}
        with pytest.raises(ValueError, match=field):
            NativePlanV1(**arguments)
