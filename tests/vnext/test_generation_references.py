from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import mub.vnext.generation as generation
from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import (
    AnswerDisposition,
    Difficulty,
    EventRole,
    Operation,
    QueryType,
    ReferenceResolutionStatus,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.task import (
    CanonicalAnswer,
    ReferenceCandidate,
    SurfaceReference,
)
from mub.vnext.generation import (
    CoreEvent,
    GenerationContext,
    SemanticCore,
    load_pilot_config,
    render_core,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import validate_gold_replay, validate_task


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/vnext/pilot.yaml"
_CONTEXT = GenerationContext(
    config=load_pilot_config(_CONFIG_PATH),
    code_revision="task-230-test",
)
_EXPECTED_CONDITION_LABELS = (
    ("alias", "alias"),
    ("same_name", "same-name"),
    ("namespace_collision", "namespace collision"),
    ("attribute_paraphrase", "attribute paraphrase"),
)


def _key(
    entity: str,
    *,
    namespace: str = "personal",
    attribute: str = "city",
    object_type: str = "slot",
) -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type=object_type,
        namespace=namespace,
        entity=entity,
        attribute=attribute,
        subkey=None,
    )


def _candidate(candidate_id: str, key: MemoryObjectKey) -> ReferenceCandidate:
    return ReferenceCandidate(
        candidate_id=candidate_id,
        object_key=key,
        evidence=f"Candidate evidence for {key.entity}",
        source_anchors=[],
    )


def _reference(
    reference_id: str,
    surface_text: str,
    condition_kind: str,
    candidate_ids: list[str],
) -> SurfaceReference:
    return SurfaceReference(
        reference_id=reference_id,
        surface_text=surface_text,
        normalized_text=surface_text.casefold().replace(" ", "_"),
        condition_kind=condition_kind,
        evidence_kind="visible_surface_form",
        candidate_ids=candidate_ids,
    )


def _unresolved_core(
    *,
    status: ReferenceResolutionStatus,
    disposition: AnswerDisposition,
    candidates: list[ReferenceCandidate],
    references: list[SurfaceReference],
    selected_candidate_ids: list[str],
    value: object = None,
    abstention_reason: str | None = None,
) -> SemanticCore:
    events = [
        CoreEvent(
            operation=Operation.ADD,
            object_keys=[candidate.object_key],
            value=("Osaka" if index == 0 else "Quito"),
            role=EventRole.LATEST_GOLD,
        )
        for index, candidate in enumerate(candidates)
    ]
    return SemanticCore(
        core_id="core_2300000000000000",
        task_family=TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
        difficulty=Difficulty.HARD,
        core_index=230,
        trajectory_id="trajectory_2300000000000000",
        events=events,
        query_targets=[candidate.object_key for candidate in candidates],
        query_type=QueryType.UNRESOLVED_REFERENCE,
        reference_candidates=candidates,
        surface_references=references,
        canonical_answer=CanonicalAnswer(
            disposition=disposition,
            resolution_status=status,
            selected_candidate_ids=selected_candidate_ids,
            abstention_reason=abstention_reason,
            value=value,
        ),
        expected_answer=None,
        profile={},
        stratification={"reference_resolution": status.value},
    )


def _unique_core() -> SemanticCore:
    candidate = _candidate("candidate_alex_book_club", _key("friend_alex"))
    return _unresolved_core(
        status=ReferenceResolutionStatus.UNIQUE,
        disposition=AnswerDisposition.ANSWERED,
        candidates=[candidate],
        references=[
            _reference(
                "reference_alex_alias",
                "Alex from book club",
                "alias",
                [candidate.candidate_id],
            ),
            _reference(
                "reference_city_paraphrase",
                "where they live",
                "attribute_paraphrase",
                [candidate.candidate_id],
            ),
        ],
        selected_candidate_ids=[candidate.candidate_id],
        value="Osaka",
    )


def test_semantic_core_carries_frozen_typed_reference_semantics() -> None:
    source_candidate = _candidate(
        "candidate_alex_book_club",
        _key("friend_alex"),
    )
    source_reference = _reference(
        "reference_alex_alias",
        "Alex from book club",
        "alias",
        [source_candidate.candidate_id],
    )
    core = _unresolved_core(
        status=ReferenceResolutionStatus.UNIQUE,
        disposition=AnswerDisposition.ANSWERED,
        candidates=[source_candidate],
        references=[source_reference],
        selected_candidate_ids=[source_candidate.candidate_id],
        value="Osaka",
    )

    source_candidate.object_key.entity = "mutated"
    source_reference.surface_text = "mutated"
    source_reference.candidate_ids.append("mutated")

    assert core.query_type is QueryType.UNRESOLVED_REFERENCE
    assert core.reference_candidates[0].object_key.entity == "friend_alex"
    assert core.surface_references[0].surface_text == "Alex from book club"
    assert core.surface_references[0].candidate_ids == (
        "candidate_alex_book_club",
    )
    assert core.canonical_answer is not None
    assert core.canonical_answer.value == "Osaka"

    with pytest.raises(ValidationError, match="frozen"):
        core.reference_candidates[0].object_key.entity = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        core.surface_references[0].surface_text = "changed"
    with pytest.raises(AttributeError):
        core.reference_candidates.append(source_candidate)


def test_semantic_core_reference_model_copy_revalidates_and_deep_copies() -> None:
    core = _unique_core()

    shallow = core.model_copy()
    deep = core.model_copy(deep=True)
    tuple_updated = core.model_copy(
        update={
            "reference_candidates": core.reference_candidates,
            "surface_references": core.surface_references,
        }
    )

    assert shallow.reference_candidates is core.reference_candidates
    assert shallow.surface_references is core.surface_references
    assert deep.reference_candidates is not core.reference_candidates
    assert deep.surface_references is not core.surface_references
    assert deep.canonical_answer is not core.canonical_answer
    assert tuple_updated.reference_candidates == core.reference_candidates
    assert tuple_updated.surface_references == core.surface_references

    with pytest.raises(ValidationError, match="unknown|candidate"):
        core.model_copy(
            update={
                "surface_references": [
                    _reference("reference_bad", "unknown", "alias", ["missing"])
                ]
            }
        )


def test_semantic_core_rejects_unresolved_bare_or_guessed_answers() -> None:
    payload = _unique_core().model_dump(mode="python")

    with pytest.raises(ValidationError, match="expected_answer"):
        SemanticCore.model_validate({**payload, "expected_answer": "Osaka"})
    with pytest.raises(ValidationError, match="canonical_answer"):
        SemanticCore.model_validate({**payload, "canonical_answer": None})


def test_semantic_core_rejects_duplicate_candidate_identity_ignoring_object_type() -> None:
    slot = _candidate("candidate_slot", _key("friend_alex", object_type="slot"))
    profile = _candidate(
        "candidate_profile",
        _key("friend_alex", object_type="profile"),
    )

    payload = _unique_core().model_dump(mode="python")
    payload["reference_candidates"] = [
        slot.model_dump(mode="python"),
        profile.model_dump(mode="python"),
    ]
    payload["surface_references"] = [
        _reference(
            "reference_same_name",
            "Alex",
            "same_name",
            [slot.candidate_id, profile.candidate_id],
        ).model_dump(mode="python")
    ]
    payload["canonical_answer"] = CanonicalAnswer(
        disposition=AnswerDisposition.ABSTAINED,
        resolution_status=ReferenceResolutionStatus.AMBIGUOUS,
        selected_candidate_ids=[],
        abstention_reason="multiple exact identities remain",
        value=None,
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match="duplicate.*candidate.*identit"):
        SemanticCore.model_validate(payload)


def test_reference_catalogs_are_reviewed_immutable_and_condition_complete() -> None:
    reference_query_template_sets = generation.REFERENCE_QUERY_TEMPLATE_SETS
    reference_condition_labels = generation.REFERENCE_CONDITION_LABELS

    assert len(reference_query_template_sets) == 3
    assert all(isinstance(item, tuple) and len(item) == 4 for item in reference_query_template_sets)
    assert [item[0] for item in reference_query_template_sets] == [
        "direct",
        "conversational",
        "correction",
    ]
    for _, query_template, resolution_instruction, abstention_instruction in reference_query_template_sets:
        assert "$candidates" in query_template
        assert "$references" in query_template
        assert "$resolution_instruction" in query_template
        assert "$abstention_instruction" in query_template
        assert "value" in resolution_instruction.lower()
        assert "abstain" in abstention_instruction.lower()
        assert "guess" in abstention_instruction.lower()

    labels = dict(reference_condition_labels)
    assert labels == dict(_EXPECTED_CONDITION_LABELS)
    with pytest.raises(TypeError):
        reference_condition_labels[0] = ("changed", "changed")  # type: ignore[index]


def test_renderer_maps_unique_reference_semantics_without_parallel_gold_answers() -> None:
    core = _unique_core()
    tasks = [
        render_core(
            core,
            split=Split.TEST,
            surface_variant=variant,
            context=_CONTEXT,
        )
        for variant in range(3)
    ]

    assert len({canonical_json_bytes(task) for task in tasks}) == 3
    assert len({semantic_task_hash(task) for task in tasks}) == 1
    for task in tasks:
        query = task.queries[0]
        canonical = task.gold.canonical_answers[query.query_id]
        assert query.query_type is QueryType.UNRESOLVED_REFERENCE
        assert [candidate.candidate_id for candidate in query.reference_candidates] == [
            "candidate_alex_book_club"
        ]
        assert [reference.candidate_ids for reference in query.surface_references] == [
            ["candidate_alex_book_club"],
            ["candidate_alex_book_club"],
        ]
        assert canonical.disposition is AnswerDisposition.ANSWERED
        assert canonical.resolution_status is ReferenceResolutionStatus.UNIQUE
        assert canonical.selected_candidate_ids == ["candidate_alex_book_club"]
        assert canonical.value == "Osaka"
        assert task.gold.gold_answers == {}
        assert task.gold.acceptable_answers == {}
        assert validate_task(task).valid
        assert validate_gold_replay(task).valid


def test_renderer_makes_all_reviewed_reference_conditions_visible() -> None:
    candidate = _candidate("candidate_alex", _key("friend_alex"))
    references = [
        _reference("reference_alias", "Alex from book club", "alias", [candidate.candidate_id]),
        _reference("reference_same_name", "Alex", "same_name", [candidate.candidate_id]),
        _reference(
            "reference_namespace",
            "Alex in the personal namespace",
            "namespace_collision",
            [candidate.candidate_id],
        ),
        _reference(
            "reference_attribute",
            "where Alex lives",
            "attribute_paraphrase",
            [candidate.candidate_id],
        ),
    ]
    core = _unresolved_core(
        status=ReferenceResolutionStatus.UNIQUE,
        disposition=AnswerDisposition.ANSWERED,
        candidates=[candidate],
        references=references,
        selected_candidate_ids=[candidate.candidate_id],
        value="Osaka",
    )

    task = render_core(
        core,
        split=Split.TEST,
        surface_variant=0,
        context=_CONTEXT,
    )
    text = task.queries[0].text

    assert all(reference.surface_text in text for reference in references)
    assert all(label in text.lower() for _, label in _EXPECTED_CONDITION_LABELS)
    assert "object_type" not in text
    assert "candidate_alex" not in text


@pytest.mark.parametrize(
    ("status", "candidate_ids", "reason"),
    [
        (
            ReferenceResolutionStatus.AMBIGUOUS,
            ["candidate_friend_alex", "candidate_manager_alex"],
            "the same-name reference matches multiple candidates",
        ),
        (
            ReferenceResolutionStatus.NO_MATCH,
            [],
            "the alias matches no candidate",
        ),
    ],
)
def test_renderer_emits_typed_abstention_without_guessing_replay_values(
    status: ReferenceResolutionStatus,
    candidate_ids: list[str],
    reason: str,
) -> None:
    candidates = [
        _candidate("candidate_friend_alex", _key("friend_alex")),
        _candidate("candidate_manager_alex", _key("manager_alex", namespace="work")),
    ]
    core = _unresolved_core(
        status=status,
        disposition=AnswerDisposition.ABSTAINED,
        candidates=candidates,
        references=[
            _reference(
                "reference_alex",
                "Alex",
                "same_name" if status is ReferenceResolutionStatus.AMBIGUOUS else "alias",
                candidate_ids,
            )
        ],
        selected_candidate_ids=[],
        abstention_reason=reason,
    )

    task = render_core(
        core,
        split=Split.TEST,
        surface_variant=1,
        context=_CONTEXT,
    )
    query = task.queries[0]
    canonical = task.gold.canonical_answers[query.query_id]

    assert "abstain" in query.text.lower()
    assert "guess" in query.text.lower()
    assert "Osaka" not in query.text
    assert "Quito" not in query.text
    assert canonical.disposition is AnswerDisposition.ABSTAINED
    assert canonical.resolution_status is status
    assert canonical.value is None
    assert canonical.abstention_reason == reason
    assert task.gold.gold_answers == {}
    assert task.gold.acceptable_answers == {}
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid


def test_reference_wording_and_ids_do_not_change_semantic_hash() -> None:
    original_core = _unique_core()
    changed_payload = original_core.model_dump(mode="python")
    changed_payload["reference_candidates"][0]["candidate_id"] = "candidate_renamed"
    for reference in changed_payload["surface_references"]:
        reference["reference_id"] = f"renamed_{reference['reference_id']}"
        reference["surface_text"] = f"Paraphrased {reference['surface_text']}"
        reference["normalized_text"] = f"paraphrased_{reference['normalized_text']}"
        reference["candidate_ids"] = ["candidate_renamed"]
    changed_payload["canonical_answer"]["selected_candidate_ids"] = [
        "candidate_renamed"
    ]
    changed_core = SemanticCore.model_validate(changed_payload)

    original = render_core(
        original_core,
        split=Split.TEST,
        surface_variant=0,
        context=_CONTEXT,
    )
    changed = render_core(
        changed_core,
        split=Split.TEST,
        surface_variant=0,
        context=_CONTEXT,
    )

    assert original.queries[0].text != changed.queries[0].text
    assert original.source.raw_hash != changed.source.raw_hash
    assert semantic_task_hash(original) == semantic_task_hash(changed)


def test_ordinary_core_defaults_and_gold_behavior_remain_unchanged_at_v2() -> None:
    target = _key("friend_alex")
    core = SemanticCore(
        core_id="core_2300000000000001",
        task_family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        core_index=231,
        trajectory_id="trajectory_2300000000000001",
        events=[
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[target],
                value="Osaka",
                role=EventRole.LATEST_GOLD,
            )
        ],
        query_targets=[target],
        expected_answer="Osaka",
        profile={},
        stratification={},
    )

    assert core.query_type is QueryType.CURRENT_STATE
    assert core.reference_candidates == ()
    assert core.surface_references == ()
    assert core.canonical_answer is None

    task = render_core(
        core,
        split=Split.TEST,
        surface_variant=0,
        context=_CONTEXT,
    )
    query_id = task.queries[0].query_id

    assert task.queries[0].query_type is QueryType.CURRENT_STATE
    assert task.queries[0].reference_candidates == []
    assert task.queries[0].surface_references == []
    assert task.gold.gold_answers == {query_id: "Osaka"}
    assert task.gold.acceptable_answers == {query_id: "Osaka"}
    assert task.gold.canonical_answers == {}
    assert task.source.normalization_version == "vnext-pilot-semantic-v2"
    assert task.metadata.split_key.split_policy_version == "vnext-pilot-core-v2"
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
