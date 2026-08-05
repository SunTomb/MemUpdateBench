import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    DerivationStepV3,
    EventAnchorSelector,
    GoldActionV3,
    LedgerEntryStatus,
    MemoryQueryV3,
    QueryGoldEvidenceV3,
    ReferenceCandidateV3,
    ReferenceResolutionSelector,
    SurfaceReferenceV3,
    VersionHistoryEntry,
    VersionHistoryLedger,
    _query_semantic_projection,
)


def key(object_type: str = "slot") -> MemoryObjectKey:
    return MemoryObjectKey(object_type=object_type, namespace="n", entity="e", attribute="a")


def test_v3_selectors_and_ledger_are_strict_and_frozen() -> None:
    selector = CurrentSelector()
    with pytest.raises(ValidationError):
        selector.kind = "previous"
    assert EventAnchorSelector(event_id="ev-1").kind == "event_anchor"
    with pytest.raises(ValidationError):
        EventAnchorSelector(event_id="")

    ledger = VersionHistoryLedger(
        object_key=key(),
        entries=(
            VersionHistoryEntry(version_index=0, status=LedgerEntryStatus.PRESENT, value="old", valid_from_event_id="ev-1", source_event_ids=("ev-1",)),
            VersionHistoryEntry(version_index=1, status=LedgerEntryStatus.PRESENT, value="new", valid_from_event_id="ev-2", source_event_ids=("ev-2",)),
        ),
    )
    assert [entry.version_index for entry in ledger.entries] == [0, 1]
    with pytest.raises(ValidationError):
        VersionHistoryLedger(object_key=key(), entries=(ledger.entries[0], ledger.entries[1].model_copy(update={"version_index": 2})))


def test_object_type_does_not_change_v3_ledger_semantic_identity() -> None:
    entry = VersionHistoryEntry(version_index=0, status=LedgerEntryStatus.PRESENT, value="x", valid_from_event_id="ev", source_event_ids=("ev",))
    left = VersionHistoryLedger(object_key=key("slot"), entries=(entry,))
    right = VersionHistoryLedger(object_key=key("profile"), entries=(entry,))
    assert left.semantic_identity == right.semantic_identity
    assert left.semantic_hash == right.semantic_hash


def test_ttl_delete_requires_effective_at_but_object_delete_does_not() -> None:
    action = {
        "action_id": "a",
        "event_id": "ev",
        "operation": "DELETE",
        "scope": "ttl",
        "target_object_keys": (key(),),
    }
    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(**action)
    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(**action, effective_at=None)

    scheduled = GoldActionV3(**action, effective_at="010")
    assert scheduled.effective_at == "010"
    assert GoldActionV3(**{**action, "scope": "object"}).effective_at is None
    assert GoldActionV3(**{**action, "scope": "object"}, effective_at="").effective_at == ""


@pytest.mark.parametrize("effective_at", ["", " ", "\t\n"])
def test_ttl_delete_rejects_blank_effective_at(effective_at: str) -> None:
    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(
            action_id="a",
            event_id="ev",
            operation="DELETE",
            scope="ttl",
            target_object_keys=(key(),),
            effective_at=effective_at,
        )


def test_ttl_delete_requires_exact_builtin_string_effective_at() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValidationError, match="TTL.*effective_at"):
        GoldActionV3(
            action_id="a",
            event_id="ev",
            operation="DELETE",
            scope="ttl",
            target_object_keys=(key(),),
            effective_at=StringSubclass("010"),
        )


@pytest.mark.parametrize("answer_schema", ["string", "number", "boolean"])
def test_direct_multi_object_current_rejects_unshapeable_answer_schema(answer_schema: str) -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    with pytest.raises(ValidationError, match="multi-object.*list/object answer schema"):
        MemoryQueryV3(
            query_id="q",
            query_type="multi_object_current",
            text="?",
            selector={"kind": "multi_object_current", "object_keys": targets},
            target_object_keys=targets,
            answer_schema=answer_schema,
            evaluation_mode="state_direct",
        )


@pytest.mark.parametrize("answer_schema", ["list", "object"])
def test_direct_multi_object_current_accepts_shapeable_answer_schema(answer_schema: str) -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    query = MemoryQueryV3(
        query_id="q",
        query_type="multi_object_current",
        text="?",
        selector={"kind": "multi_object_current", "object_keys": targets},
        target_object_keys=targets,
        answer_schema=answer_schema,
        evaluation_mode="state_direct",
    )
    assert query.answer_schema.value == answer_schema


def test_multi_object_current_consistency_preserves_boolean_answer_schema() -> None:
    other = MemoryObjectKey(object_type="slot", namespace="n", entity="other", attribute="a")
    targets = (key(), other)
    query = MemoryQueryV3(
        query_id="q",
        query_type="multi_object_current_consistency",
        text="?",
        selector={"kind": "multi_object_current", "object_keys": targets},
        target_object_keys=targets,
        answer_schema="boolean",
        evaluation_mode="state_direct",
        synthesis={"kind": "multi_object_current_consistency", "minimum_objects": 2},
    )
    assert query.answer_schema.value == "boolean"


def _reference_keys(object_type: str = "slot") -> tuple[MemoryObjectKey, MemoryObjectKey]:
    return (
        MemoryObjectKey(
            object_type=object_type,
            namespace="people",
            entity="alex_primary",
            attribute="city",
        ),
        MemoryObjectKey(
            object_type=object_type,
            namespace="people",
            entity="alex_secondary",
            attribute="city",
        ),
    )


def _reference_selector(
    candidate_ids: tuple[str, ...],
    *,
    object_type: str = "slot",
    surface_text: str = "Alex.city",
) -> ReferenceResolutionSelector:
    keys = _reference_keys(object_type)
    return ReferenceResolutionSelector(
        reference_candidates=tuple(
            ReferenceCandidateV3(candidate_id=candidate_id, object_key=key)
            for candidate_id, key in zip(("candidate-0", "candidate-1"), keys)
        ),
        surface_references=(
            SurfaceReferenceV3(
                reference_id="reference-0",
                surface_text=surface_text,
                normalized_text=surface_text.casefold(),
                condition_kind="same_name",
                evidence_kind="exact_attribute",
                candidate_ids=candidate_ids,
            ),
        ),
    )


def _reference_evidence(**changes) -> QueryGoldEvidenceV3:
    payload = {
        "query_id": "query-0",
        "answer": "Paris",
        "disposition": "answered",
        "resolution_status": "unique",
        "selected_candidate_ids": ("candidate-0",),
        "supporting_object_keys": (_reference_keys()[0],),
        "supporting_event_ids": ("event-0",),
        "derivation_steps": (
            DerivationStepV3(
                step_id="read-0",
                operation="read_current",
                supporting_object_keys=(_reference_keys()[0],),
                supporting_event_ids=("event-0",),
            ),
        ),
        "final_derivation_step_id": "read-0",
    }
    payload.update(changes)
    return QueryGoldEvidenceV3(**payload)


@pytest.mark.parametrize(
    ("candidate_ids", "disposition", "resolution_status", "answer", "selected", "reason"),
    [
        (("candidate-0",), "answered", "unique", "Paris", ("candidate-0",), None),
        (("candidate-0", "candidate-1"), "abstained", "ambiguous", None, (), "multiple exact objects"),
        ((), "abstained", "no_match", None, (), "no reviewed canonical match"),
    ],
)
def test_reference_resolution_contract_accepts_typed_outcomes(
    candidate_ids,
    disposition,
    resolution_status,
    answer,
    selected,
    reason,
) -> None:
    selector = _reference_selector(candidate_ids)
    query = MemoryQueryV3(
        query_id="query-0",
        query_type="unresolved_reference",
        text="What is Alex's city?",
        selector=selector,
        target_object_keys=_reference_keys(),
        answer_schema="string",
        evaluation_mode="state_direct",
    )
    evidence = _reference_evidence(
        answer=answer,
        disposition=disposition,
        resolution_status=resolution_status,
        selected_candidate_ids=selected,
        abstention_reason=reason,
    )

    assert query.selector.kind == "reference_resolution"
    assert evidence.disposition.value == disposition
    assert evidence.resolution_status.value == resolution_status


@pytest.mark.parametrize(
    "changes",
    [
        {"disposition": "unavailable"},
        {"resolution_status": "ambiguous"},
        {"selected_candidate_ids": ()},
        {"answer": None},
        {"abstention_reason": "not allowed"},
        {
            "disposition": "abstained",
            "resolution_status": "unique",
            "answer": None,
            "selected_candidate_ids": (),
            "abstention_reason": "wrong status",
        },
        {
            "disposition": "abstained",
            "resolution_status": "ambiguous",
            "answer": "sentinel",
            "selected_candidate_ids": (),
            "abstention_reason": "multiple",
        },
        {
            "disposition": "abstained",
            "resolution_status": "ambiguous",
            "answer": None,
            "selected_candidate_ids": ("candidate-0",),
            "abstention_reason": "multiple",
        },
        {
            "disposition": "abstained",
            "resolution_status": "no_match",
            "answer": None,
            "selected_candidate_ids": (),
            "abstention_reason": "   ",
        },
    ],
)
def test_reference_gold_evidence_rejects_forged_dispositions(changes) -> None:
    with pytest.raises(ValidationError):
        _reference_evidence(**changes)


def test_reference_selector_rejects_duplicate_or_unknown_candidate_links() -> None:
    keys = _reference_keys()
    with pytest.raises(ValidationError, match="candidate IDs.*unique"):
        ReferenceResolutionSelector(
            reference_candidates=(
                ReferenceCandidateV3(candidate_id="duplicate", object_key=keys[0]),
                ReferenceCandidateV3(candidate_id="duplicate", object_key=keys[1]),
            ),
            surface_references=(
                SurfaceReferenceV3(
                    reference_id="reference-0",
                    surface_text="Alex.city",
                    normalized_text="alex.city",
                    candidate_ids=("duplicate",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="unknown candidates"):
        ReferenceResolutionSelector(
            reference_candidates=(
                ReferenceCandidateV3(candidate_id="candidate-0", object_key=keys[0]),
            ),
            surface_references=(
                SurfaceReferenceV3(
                    reference_id="reference-0",
                    surface_text="Alex.city",
                    normalized_text="alex.city",
                    candidate_ids=("forged",),
                ),
            ),
        )


def test_reference_selector_semantics_ignore_surface_and_object_type_only() -> None:
    left = MemoryQueryV3(
        query_id="left",
        query_type="unresolved_reference",
        text="surface one",
        selector=_reference_selector(("candidate-0",), object_type="slot", surface_text="Alex.city"),
        target_object_keys=_reference_keys("slot"),
        answer_schema="string",
        evaluation_mode="state_direct",
    )
    right = MemoryQueryV3(
        query_id="right",
        query_type="unresolved_reference",
        text="surface two",
        selector=_reference_selector(("candidate-0",), object_type="profile", surface_text="Where Alex lives"),
        target_object_keys=_reference_keys("profile"),
        answer_schema="string",
        evaluation_mode="state_direct",
    )
    forged = right.model_copy(
        update={
            "selector": {
                "kind": "reference_resolution",
                "reference_candidates": (
                    {"candidate_id": "candidate-0", "object_key": _reference_keys("profile")[0]},
                    {
                        "candidate_id": "candidate-1",
                        "object_key": MemoryObjectKey(
                            object_type="profile",
                            namespace="people",
                            entity="forged",
                            attribute="city",
                        ),
                    },
                ),
                "surface_references": right.selector.surface_references,
            },
            "target_object_keys": (
                _reference_keys("profile")[0],
                MemoryObjectKey(
                    object_type="profile",
                    namespace="people",
                    entity="forged",
                    attribute="city",
                ),
            ),
        }
    )

    assert _query_semantic_projection(left, {}) == _query_semantic_projection(right, {})
    assert _query_semantic_projection(left, {}) != _query_semantic_projection(forged, {})
