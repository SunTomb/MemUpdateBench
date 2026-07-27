from __future__ import annotations

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


def test_reference_contracts_use_typed_four_part_object_keys(make_object_key) -> None:
    first_key = make_object_key().model_copy(
        update={"namespace": "contacts", "subkey": "primary"}
    )
    second_key = first_key.model_copy(
        update={"object_type": "profile", "entity": "friend:alexander"}
    )
    reference = SurfaceReference(
        text="Alex's location",
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        candidates=[
            ReferenceCandidate(object_key=first_key),
            ReferenceCandidate(object_key=second_key),
        ],
    )

    round_tripped = SurfaceReference.model_validate_json(reference.model_dump_json())

    assert round_tripped == reference
    assert round_tripped.candidates[0].object_key.canonical_id == (
        "contacts|friend:alex|location|primary"
    )
    assert round_tripped.candidates[1].object_key.object_type == "profile"
    assert "object_type" not in round_tripped.candidates[1].object_key.canonical_id


def test_reference_contracts_remain_strict_and_frozen_by_task_conventions(
    make_object_key,
) -> None:
    with pytest.raises(ValidationError):
        ReferenceCandidate(object_key=make_object_key(), extra="forbidden")
    with pytest.raises(ValidationError):
        SurfaceReference(
            text="Alex",
            resolution_status="unknown",
            candidates=[],
        )


def test_canonical_answer_represents_answered_and_abstained_gold_explicitly() -> None:
    answered = CanonicalAnswer(
        disposition=AnswerDisposition.ANSWERED,
        value="Qingdao",
    )
    abstained = CanonicalAnswer(disposition=AnswerDisposition.ABSTAINED)

    assert answered.model_dump(mode="json") == {
        "disposition": "answered",
        "value": "Qingdao",
    }
    assert abstained.model_dump(mode="json") == {
        "disposition": "abstained",
        "value": None,
    }
    with pytest.raises(ValidationError, match="runtime-only"):
        CanonicalAnswer(disposition=AnswerDisposition.UNAVAILABLE)


def test_task_contracts_add_typed_fields_without_changing_ordinary_defaults(
    make_task,
) -> None:
    task = make_task()

    assert task.queries[0].surface_references == []
    assert task.gold.canonical_answers == {}

    data = task.model_dump(mode="json")
    data["queries"][0]["query_type"] = "unresolved_reference"
    data["queries"][0]["surface_references"] = [
        {
            "text": "Alex's location",
            "resolution_status": "ambiguous",
            "candidates": [
                {"object_key": data["target_objects"][0]},
                {
                    "object_key": {
                        **data["target_objects"][0],
                        "entity": "colleague:alex",
                    }
                },
            ],
        }
    ]
    data["gold"]["canonical_answers"] = {
        "query_0": {"disposition": "abstained", "value": None}
    }

    unresolved = type(task).model_validate(data)

    assert unresolved.queries[0].query_type == QueryType.UNRESOLVED_REFERENCE
    assert isinstance(
        unresolved.queries[0].surface_references[0], SurfaceReference
    )
    assert unresolved.gold.canonical_answers["query_0"].disposition == (
        AnswerDisposition.ABSTAINED
    )
    assert type(task).model_validate_json(task.model_dump_json()) == task


def test_runtime_answer_records_have_explicit_dispositions_with_safe_defaults() -> None:
    ordinary_prediction = AnswerPrediction(
        query_id="query_0",
        raw_output="Qingdao",
        parsed_answer="Qingdao",
        format_valid=True,
    )
    unavailable_prediction = AnswerPrediction(
        query_id="query_0",
        raw_output="",
        disposition=AnswerDisposition.UNAVAILABLE,
        parsed_answer=None,
        format_valid=False,
    )
    ordinary_result = AnswerResult(query_id="query_0", raw_output="Qingdao")
    abstained_result = AnswerResult(
        query_id="query_0",
        raw_output="I cannot resolve that reference.",
        disposition=AnswerDisposition.ABSTAINED,
    )

    assert ordinary_prediction.disposition == AnswerDisposition.ANSWERED
    assert unavailable_prediction.disposition == AnswerDisposition.UNAVAILABLE
    assert ordinary_result.disposition == AnswerDisposition.ANSWERED
    assert abstained_result.disposition == AnswerDisposition.ABSTAINED


def test_v2_contract_version_matrix_is_pinned_consistently(make_task, make_task_run) -> None:
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
