from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from mub.vnext.contracts.score import ScoreRecord
from mub.vnext.scoring.failures import (
    FAILURE_FLAGS,
    PRIMARY_FAILURE_PRECEDENCE,
    PRIMARY_FAILURE_PRECEDENCE_VERSION,
    canonicalize_failure_flags,
    primary_failure,
)
from mub.vnext.version import PRIMARY_FAILURE_PRECEDENCE_VERSION as CONTRACT_VERSION


EXPECTED_FLAGS = (
    "invalid_action_format",
    "unsupported_action",
    "wrong_operation",
    "wrong_entity",
    "wrong_attribute",
    "wrong_value",
    "false_write",
    "missed_update",
    "collateral_corruption",
    "deletion_failure",
    "current_state_missing",
    "stale_retained",
    "current_not_retrieved",
    "stale_retrieved",
    "stale_copied",
    "distractor_retrieved",
    "distractor_copied",
    "gold_retrieved_wrong_answer",
    "wrong_reference_guess",
    "unjustified_abstention",
    "answer_format_only",
    "system_exception",
)

EXPECTED_PRECEDENCE = (
    (
        "system_exception",
        "invalid_action_format",
        "unsupported_action",
    ),
    (
        "wrong_operation",
        "wrong_entity",
        "wrong_attribute",
        "wrong_value",
        "false_write",
        "missed_update",
    ),
    (
        "collateral_corruption",
        "deletion_failure",
        "current_state_missing",
        "stale_retained",
    ),
    (
        "current_not_retrieved",
        "stale_retrieved",
        "distractor_retrieved",
    ),
    (
        "wrong_reference_guess",
        "unjustified_abstention",
    ),
    (
        "stale_copied",
        "distractor_copied",
        "gold_retrieved_wrong_answer",
    ),
    ("answer_format_only",),
)


def test_failure_flag_vocabulary_and_precedence_are_exact_immutable_and_versioned() -> None:
    assert FAILURE_FLAGS == EXPECTED_FLAGS
    assert "correct" not in FAILURE_FLAGS
    assert PRIMARY_FAILURE_PRECEDENCE == EXPECTED_PRECEDENCE
    assert PRIMARY_FAILURE_PRECEDENCE_VERSION == CONTRACT_VERSION
    assert isinstance(FAILURE_FLAGS, tuple)
    assert all(isinstance(layer, tuple) for layer in PRIMARY_FAILURE_PRECEDENCE)


@pytest.mark.parametrize("flag", EXPECTED_FLAGS)
def test_every_independent_flag_is_its_own_primary_when_alone(flag: str) -> None:
    assert primary_failure([flag]) == flag


def test_empty_flags_use_correct_only_as_derived_primary_label() -> None:
    assert primary_failure(()) == "correct"


def test_overlap_uses_explicit_layer_and_intra_layer_precedence() -> None:
    assert primary_failure(["answer_format_only", "stale_copied", "wrong_value"]) == "wrong_value"
    assert primary_failure(["unsupported_action", "invalid_action_format"]) == "invalid_action_format"
    assert primary_failure(["stale_retained", "current_state_missing"]) == "current_state_missing"
    assert primary_failure(["distractor_retrieved", "stale_retrieved"]) == "stale_retrieved"
    assert primary_failure(["gold_retrieved_wrong_answer", "distractor_copied"]) == "distractor_copied"
    assert primary_failure(["stale_copied", "unjustified_abstention"]) == "unjustified_abstention"
    assert primary_failure(["unjustified_abstention", "wrong_reference_guess"]) == "wrong_reference_guess"


def test_primary_is_permutation_and_duplicate_invariant_without_mutating_caller() -> None:
    original = ["answer_format_only", "stale_copied", "wrong_attribute"]
    duplicated = [*original, "wrong_attribute"]
    snapshot = list(duplicated)
    expected = "wrong_attribute"
    assert primary_failure(duplicated) == expected
    assert duplicated == snapshot
    for permutation in itertools.permutations(original):
        assert primary_failure(permutation) == expected


@pytest.mark.parametrize(
    "flags",
    [
        ["unknown"],
        [""],
        [" correct "],
        [1],
        "stale_copied",
        {"stale_copied": True},
        None,
    ],
)
def test_unknown_or_malformed_failure_inputs_are_rejected(flags) -> None:
    with pytest.raises((TypeError, ValueError)):
        primary_failure(flags)


class StringSubclass(str):
    pass


class ExplodingHash:
    def __hash__(self):
        raise RuntimeError("hash exploded")


class ExplodingEquality:
    def __eq__(self, other):
        raise RuntimeError("equality exploded")


class RaisingIterator:
    def __iter__(self):
        yield "stale_copied"
        raise RuntimeError("iteration exploded")


class NonIterable:
    def __iter__(self):
        raise TypeError("cannot iterate")


class ValueErrorOnIter:
    def __iter__(self):
        raise ValueError("value acquisition exploded")


class LookupErrorDuringIteration:
    def __iter__(self):
        yield "stale_copied"
        raise LookupError("lookup consumption exploded")


def test_canonicalize_failure_flags_orders_deduplicates_and_preserves_input() -> None:
    supplied = ["stale_copied", "wrong_attribute", "stale_copied"]
    before = list(supplied)
    assert canonicalize_failure_flags(supplied) == ("wrong_attribute", "stale_copied")
    assert supplied == before
    assert canonicalize_failure_flags(set(reversed(supplied))) == (
        "wrong_attribute",
        "stale_copied",
    )


@pytest.mark.parametrize(
    "flags",
    [
        [StringSubclass("stale_copied")],
        [["stale_copied"]],
        [ExplodingHash()],
        [ExplodingEquality()],
        [b"stale_copied"],
    ],
)
def test_failure_functions_reject_non_exact_strings_before_hash_or_equality(flags) -> None:
    with pytest.raises(TypeError, match="exact built-in strings"):
        canonicalize_failure_flags(flags)
    with pytest.raises(TypeError, match="exact built-in strings"):
        primary_failure(flags)


def test_failure_iterator_errors_are_translated_with_chaining() -> None:
    for flags, message, cause_type in (
        (NonIterable(), "obtain failure flag iterator", TypeError),
        (ValueErrorOnIter(), "obtain failure flag iterator", ValueError),
        (RaisingIterator(), "consume failure flag iterator", RuntimeError),
        (LookupErrorDuringIteration(), "consume failure flag iterator", LookupError),
    ):
        for function in (canonicalize_failure_flags, primary_failure):
            with pytest.raises(TypeError, match=message) as raised:
                function(flags)
            assert isinstance(raised.value.__cause__, cause_type)


def test_score_record_canonicalizes_flags_and_validates_primary(make_score_record) -> None:
    score = make_score_record(
        failure_flags={"stale_copied", "wrong_attribute"},
        primary_failure="wrong_attribute",
    )
    assert tuple(str(flag) for flag in score.failure_flags) == (
        "wrong_attribute",
        "stale_copied",
    )
    with pytest.raises(ValidationError, match="primary_failure"):
        make_score_record(
            failure_flags=["stale_copied", "wrong_attribute"],
            primary_failure="stale_copied",
        )
    with pytest.raises(ValidationError):
        make_score_record(failure_flags=["correct"], primary_failure="correct")
    assert make_score_record(failure_flags=[], primary_failure=None).primary_failure is None


def test_score_record_revalidates_model_constructed_failure_bypass(make_score_record) -> None:
    valid = make_score_record()
    bypassed = ScoreRecord.model_construct(
        **{
            **valid.model_dump(mode="python"),
            "failure_flags": ("unknown",),
            "primary_failure": "unknown",
        }
    )
    with pytest.raises(ValidationError):
        ScoreRecord.model_validate(bypassed.model_dump(mode="python", warnings=False))


def test_score_record_schema_failure_flags_excludes_correct() -> None:
    schema = ScoreRecord.model_json_schema()
    assert "invalid_action_format" in str(schema["$defs"])
    assert "correct" not in str(schema["properties"]["failure_flags"])
