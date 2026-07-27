from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

import mub.vnext as vnext
import mub.vnext.contracts as contracts
from mub.vnext.contracts import (
    AnswerDisposition,
    CanonicalAnswer,
    QueryType,
    ReferenceCandidate,
    ReferenceResolutionStatus,
    SurfaceReference,
)
from mub.vnext.contracts.adapter import AnswerResult
from mub.vnext.contracts.runtime import AnswerPrediction
from mub.vnext.legacy.artifacts import (
    LEGACY_CLI_COMPILER_VERSION,
    LEGACY_SCHEMA_VERSION,
    LegacyAnalysisManifest,
)
from mub.vnext.version import (
    COMPILER_VERSION,
    METRIC_REGISTRY_VERSION,
    PRIMARY_FAILURE_PRECEDENCE_VERSION,
    PROFILE_VERSION,
    RUN_MANIFEST_VERSION,
    RUNTIME_RECORD_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TASK_MANIFEST_VERSION,
)


def _unresolved_payload(make_task, *, status: str = "ambiguous") -> dict:
    data = make_task().model_dump(mode="json")
    second_key = {
        **data["target_objects"][0],
        "object_type": "profile",
        "entity": "colleague:alex",
    }
    data["target_objects"].append(second_key)
    data["queries"][0].update(
        {
            "query_type": "unresolved_reference",
            "reference_candidates": [
                {
                    "candidate_id": "candidate_alex_friend",
                    "object_key": data["target_objects"][0],
                    "evidence": "The query says friend Alex.",
                    "source_anchors": [
                        {
                            "document_id": "query_0",
                            "section_id": "surface",
                            "start_char": 0,
                            "end_char": 4,
                        }
                    ],
                },
                {
                    "candidate_id": "candidate_alex_colleague",
                    "object_key": second_key,
                    "evidence": None,
                    "source_anchors": [],
                },
            ],
            "surface_references": [
                {
                    "reference_id": "reference_alex",
                    "surface_text": "Alex",
                    "normalized_text": "alex",
                    "condition_kind": "same_surface_name",
                    "evidence_kind": "query_span",
                    "candidate_ids": [
                        "candidate_alex_friend",
                        "candidate_alex_colleague",
                    ],
                }
            ],
        }
    )
    data["gold"]["gold_answers"] = {}
    data["gold"]["acceptable_answers"] = {}
    if status == "unique":
        canonical = {
            "disposition": "answered",
            "resolution_status": "unique",
            "selected_candidate_ids": ["candidate_alex_friend"],
            "abstention_reason": None,
            "value": "Qingdao",
        }
    else:
        canonical = {
            "disposition": "abstained",
            "resolution_status": status,
            "selected_candidate_ids": [],
            "abstention_reason": "reference is not uniquely resolvable",
            "value": None,
        }
        if status == "no_match":
            data["queries"][0]["surface_references"][0]["candidate_ids"] = []
    if status == "unique":
        data["queries"][0]["surface_references"][0]["candidate_ids"] = [
            "candidate_alex_friend"
        ]
    data["gold"]["canonical_answers"] = {"query_0": canonical}
    return data


def test_unresolved_reference_vocabularies_are_public_and_exact() -> None:
    assert QueryType.UNRESOLVED_REFERENCE.value == "unresolved_reference"
    assert [item.value for item in AnswerDisposition] == [
        "answered",
        "abstained",
        "unavailable",
    ]
    assert [item.value for item in ReferenceResolutionStatus] == [
        "unique",
        "ambiguous",
        "no_match",
    ]
    for name in (
        "AnswerDisposition",
        "CanonicalAnswer",
        "ReferenceCandidate",
        "ReferenceResolutionStatus",
        "SurfaceReference",
    ):
        assert getattr(contracts, name).__name__ == name
        assert name in contracts.__all__


def test_reference_contracts_expose_complete_typed_shape(make_object_key) -> None:
    key = make_object_key().model_copy(
        update={"namespace": "contacts", "subkey": "primary"}
    )
    candidate = ReferenceCandidate(
        candidate_id="candidate_alex",
        object_key=key,
        evidence="Friend-qualified mention",
        source_anchors=[
            {
                "document_id": "query_0",
                "section_id": "surface",
                "start_char": 0,
                "end_char": 4,
            }
        ],
    )
    reference = SurfaceReference(
        reference_id="reference_alex",
        surface_text="Alex",
        normalized_text="alex",
        condition_kind="same_surface_name",
        evidence_kind="query_span",
        candidate_ids=[candidate.candidate_id],
    )

    round_tripped = ReferenceCandidate.model_validate_json(candidate.model_dump_json())

    assert list(ReferenceCandidate.model_fields) == [
        "candidate_id",
        "object_key",
        "evidence",
        "source_anchors",
    ]
    assert list(SurfaceReference.model_fields) == [
        "reference_id",
        "surface_text",
        "normalized_text",
        "condition_kind",
        "evidence_kind",
        "candidate_ids",
    ]
    assert round_tripped.object_key.canonical_id == (
        "contacts|friend:alex|location|primary"
    )
    assert round_tripped.source_anchors[0].document_id == "query_0"
    assert reference.candidate_ids == ["candidate_alex"]
    with pytest.raises(ValidationError):
        ReferenceCandidate(candidate_id="candidate", object_key=key, extra=True)


def test_object_type_remains_excluded_from_reference_candidate_identity(make_object_key) -> None:
    slot = ReferenceCandidate(candidate_id="slot", object_key=make_object_key())
    profile = ReferenceCandidate(
        candidate_id="profile",
        object_key=make_object_key().model_copy(update={"object_type": "profile"}),
    )

    assert slot.object_key == profile.object_key
    assert slot.object_key.canonical_id == profile.object_key.canonical_id
    assert "object_type" not in slot.object_key.canonical_id


def test_canonical_answer_encodes_resolution_and_abstention_explicitly() -> None:
    answered = CanonicalAnswer(
        disposition=AnswerDisposition.ANSWERED,
        resolution_status=ReferenceResolutionStatus.UNIQUE,
        selected_candidate_ids=["candidate_alex"],
        value="Qingdao",
    )
    abstained = CanonicalAnswer(
        disposition=AnswerDisposition.ABSTAINED,
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        abstention_reason="multiple candidates remain",
    )

    assert answered.selected_candidate_ids == ["candidate_alex"]
    assert abstained.value is None
    with pytest.raises(ValidationError, match="runtime-only"):
        CanonicalAnswer(
            disposition=AnswerDisposition.UNAVAILABLE,
            resolution_status=ReferenceResolutionStatus.NO_MATCH,
        )
    with pytest.raises(ValidationError, match="abstention_reason"):
        CanonicalAnswer(
            disposition=AnswerDisposition.ABSTAINED,
            resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        )


def test_unresolved_task_accepts_ambiguous_no_match_and_unique_contracts(make_task) -> None:
    for status in ("ambiguous", "no_match", "unique"):
        task = type(make_task()).model_validate(
            _unresolved_payload(make_task, status=status)
        )
        assert task.queries[0].query_type == QueryType.UNRESOLVED_REFERENCE
        assert task.gold.canonical_answers["query_0"].resolution_status.value == status


def test_ordinary_task_defaults_and_answer_rules_are_unchanged(make_task) -> None:
    task = make_task()

    assert task.queries[0].reference_candidates == []
    assert task.queries[0].surface_references == []
    assert task.gold.canonical_answers == {}
    assert type(task).model_validate_json(task.model_dump_json()) == task


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_canonical", "require canonical_answers"),
        ("duplicate_candidates", "duplicate reference candidate IDs"),
        ("unknown_surface_candidate", "references unknown candidates"),
        ("unknown_selected_candidate", "selects unknown candidates"),
        ("answered_ambiguous", "must have UNIQUE"),
        ("answered_without_selection", "must select one candidate"),
        ("abstained_unique", "cannot have UNIQUE"),
        ("unresolved_bare_none", "unknown or unresolved query ID"),
    ],
)
def test_unresolved_task_rejects_invalid_status_disposition_and_linkage(
    make_task, mutation: str, message: str
) -> None:
    data = _unresolved_payload(make_task)
    if mutation == "missing_canonical":
        data["gold"]["canonical_answers"] = {}
    elif mutation == "duplicate_candidates":
        data["queries"][0]["reference_candidates"][1]["candidate_id"] = (
            "candidate_alex_friend"
        )
    elif mutation == "unknown_surface_candidate":
        data["queries"][0]["surface_references"][0]["candidate_ids"] = ["missing"]
    elif mutation == "unknown_selected_candidate":
        data["queries"][0]["surface_references"][0]["candidate_ids"] = [
            "candidate_alex_friend"
        ]
        canonical = data["gold"]["canonical_answers"]["query_0"]
        canonical.update(
            disposition="answered",
            resolution_status="unique",
            selected_candidate_ids=["missing"],
            abstention_reason=None,
            value="Qingdao",
        )
    elif mutation == "answered_ambiguous":
        canonical = data["gold"]["canonical_answers"]["query_0"]
        canonical.update(
            disposition="answered",
            selected_candidate_ids=["candidate_alex_friend"],
            abstention_reason=None,
            value="Qingdao",
        )
    elif mutation == "answered_without_selection":
        data["queries"][0]["surface_references"][0]["candidate_ids"] = [
            "candidate_alex_friend"
        ]
        canonical = data["gold"]["canonical_answers"]["query_0"]
        canonical.update(
            disposition="answered",
            resolution_status="unique",
            abstention_reason=None,
            value="Qingdao",
        )
    elif mutation == "abstained_unique":
        data["queries"][0]["surface_references"][0]["candidate_ids"] = [
            "candidate_alex_friend"
        ]
        data["gold"]["canonical_answers"]["query_0"]["resolution_status"] = "unique"
    elif mutation == "unresolved_bare_none":
        data["gold"]["gold_answers"] = {"query_0": None}

    with pytest.raises(ValidationError, match=message):
        type(make_task()).model_validate(data)


def test_runtime_disposition_rejects_inconsistent_prediction_payloads() -> None:
    for disposition in (AnswerDisposition.ABSTAINED, AnswerDisposition.UNAVAILABLE):
        with pytest.raises(ValidationError, match="cannot carry parsed_answer"):
            AnswerPrediction(
                query_id="query_0",
                raw_output="Qingdao",
                disposition=disposition,
                parsed_answer="Qingdao",
                format_valid=True,
            )
    with pytest.raises(ValidationError, match="require parsed_answer"):
        AnswerPrediction(
            query_id="query_0",
            raw_output="Qingdao",
            disposition=AnswerDisposition.ANSWERED,
            parsed_answer=None,
            format_valid=True,
        )


def test_runtime_disposition_rejects_inconsistent_adapter_results() -> None:
    for disposition in (AnswerDisposition.ABSTAINED, AnswerDisposition.UNAVAILABLE):
        with pytest.raises(ValidationError, match="cannot carry value"):
            AnswerResult(
                query_id="query_0",
                raw_output="",
                disposition=disposition,
                value="Qingdao",
            )
    with pytest.raises(ValidationError, match="require raw_output or value"):
        AnswerResult(
            query_id="query_0",
            raw_output="",
            disposition=AnswerDisposition.ANSWERED,
        )

    assert AnswerResult(
        query_id="query_0",
        raw_output="",
        disposition=AnswerDisposition.ANSWERED,
        value="Qingdao",
    ).value == "Qingdao"


def test_runtime_answer_records_preserve_ordinary_answered_defaults() -> None:
    prediction = AnswerPrediction(
        query_id="query_0",
        raw_output="Qingdao",
        parsed_answer="Qingdao",
        format_valid=True,
    )
    result = AnswerResult(query_id="query_0", raw_output="Qingdao")

    assert prediction.disposition == AnswerDisposition.ANSWERED
    assert result.disposition == AnswerDisposition.ANSWERED


def test_v2_contract_version_matrix_and_v1_legacy_defaults_are_coherent(
    make_task, make_task_run
) -> None:
    versions = (
        SCHEMA_VERSION,
        SCORER_VERSION,
        METRIC_REGISTRY_VERSION,
        COMPILER_VERSION,
        PROFILE_VERSION,
        RUNTIME_RECORD_VERSION,
        TASK_MANIFEST_VERSION,
        RUN_MANIFEST_VERSION,
        PRIMARY_FAILURE_PRECEDENCE_VERSION,
    )

    assert set(versions) == {"2.0.0"}
    assert make_task().schema_version == "2.0.0"
    assert make_task_run().schema_version == "2.0.0"
    assert make_task_run().runtime_record_version == "2.0.0"
    assert vnext.SCHEMA_VERSION == "2.0.0"
    assert LEGACY_SCHEMA_VERSION == "1.0.0"
    assert LEGACY_CLI_COMPILER_VERSION == "vnext-phase0-cli-1.0.0"
    assert LegacyAnalysisManifest.model_fields["schema_version"].default == "1.0.0"
