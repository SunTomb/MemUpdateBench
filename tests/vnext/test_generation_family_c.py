from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import mub.vnext.generation.family_c as family_c_module
from mub.vnext.contracts import (
    AnswerDisposition,
    CompletionStatus,
    Difficulty,
    QueryType,
    ReferenceResolutionStatus,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.adapter import AdapterCapabilities
from mub.vnext.contracts.manifest import ScorerConfig
from mub.vnext.contracts.runtime import (
    AnswerPrediction,
    ParserExtractorProvenance,
    TaskRunRecord,
)
from mub.vnext.generation import (
    GenerationContext,
    generate_family_c_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.generation.family_b import generate_family_b_cores
from mub.vnext.io import semantic_task_hash, sha256_model
from mub.vnext.scoring.scorer import score_task
from mub.vnext.validation.replay import replay_actions, validate_gold_replay
from mub.vnext.validation.task import validate_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
_METRIC = "answer_scores.reference_resolution_accuracy"


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def cores(config):
    return generate_family_c_cores(config)


def _cell(core):
    return (
        core.stratification["entity_condition"],
        core.stratification["attribute_condition"],
    )


def _expected_resolution(entity_condition: str, attribute_condition: str):
    if attribute_condition == "near_name":
        return (
            ReferenceResolutionStatus.NO_MATCH,
            AnswerDisposition.ABSTAINED,
        )
    if entity_condition in {"same_name", "namespace_collision"}:
        return (
            ReferenceResolutionStatus.AMBIGUOUS,
            AnswerDisposition.ABSTAINED,
        )
    return ReferenceResolutionStatus.UNIQUE, AnswerDisposition.ANSWERED


def test_family_c_has_exact_ten_per_configured_grid_cell(config, cores):
    family = config.families.entity_attribute_grounding
    expected_cells = {
        (entity_condition, attribute_condition)
        for entity_condition in family.entity_conditions
        for attribute_condition in family.attribute_conditions
    }

    assert len(cores) == 120
    assert Counter(map(_cell, cores)) == Counter({cell: 10 for cell in expected_cells})
    assert [core.core_index for core in cores] == list(range(120))
    assert {core.task_family for core in cores} == {
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING
    }
    assert {core.difficulty for core in cores} == {
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    }


def test_family_c_follows_reordered_valid_condition_axes(config):
    family = config.families.entity_attribute_grounding.model_copy(
        update={
            "entity_conditions": [
                "namespace_collision",
                "alias",
                "same_name",
                "distinct",
            ],
            "attribute_conditions": ["near_name", "exact", "paraphrase"],
        }
    )
    families = config.families.model_copy(
        update={"entity_attribute_grounding": family}
    )
    reordered = config.model_copy(update={"families": families})
    reordered_cores = generate_family_c_cores(reordered)

    assert _cell(reordered_cores[0]) == ("namespace_collision", "near_name")
    assert Counter(map(_cell, reordered_cores)) == Counter(
        {
            (entity_condition, attribute_condition): 10
            for entity_condition in family.entity_conditions
            for attribute_condition in family.attribute_conditions
        }
    )


def test_family_c_populates_typed_reference_graph_and_dispositions(cores):
    for core in cores:
        entity_condition, attribute_condition = _cell(core)
        expected_status, expected_disposition = _expected_resolution(
            entity_condition, attribute_condition
        )
        canonical = core.canonical_answer

        assert core.query_type is QueryType.UNRESOLVED_REFERENCE
        assert core.expected_answer is None
        assert len(core.reference_candidates) == 2
        assert len(core.surface_references) == 1
        assert canonical is not None
        assert canonical.resolution_status is expected_status
        assert canonical.disposition is expected_disposition
        assert core.stratification["resolution_status"] == expected_status.value
        assert core.stratification["answer_disposition"] == expected_disposition.value
        assert core.stratification["candidate_count"] == 2

        reference = core.surface_references[0]
        if expected_status is ReferenceResolutionStatus.UNIQUE:
            assert len(reference.candidate_ids) == 1
            assert tuple(canonical.selected_candidate_ids) == tuple(
                reference.candidate_ids
            )
            assert canonical.value is not None
            assert canonical.abstention_reason is None
        else:
            assert canonical.selected_candidate_ids == ()
            assert canonical.value is None
            assert canonical.abstention_reason
            if expected_status is ReferenceResolutionStatus.NO_MATCH:
                assert reference.candidate_ids == ()
            else:
                assert len(reference.candidate_ids) == 2


def test_family_c_identity_and_reviewed_mapping_evidence_are_explicit(cores):
    for core in cores:
        entity_condition, attribute_condition = _cell(core)
        candidates = core.reference_candidates
        identities = [
            (
                candidate.object_key.namespace,
                candidate.object_key.entity,
                candidate.object_key.attribute,
                candidate.object_key.subkey,
            )
            for candidate in candidates
        ]
        assert len(set(identities)) == len(identities)
        assert {candidate.object_key.object_type for candidate in candidates} == {
            "slot"
        }
        assert core.profile["alias_namespace_condition"] == entity_condition
        assert core.stratification["entity_mapping_id"] != ""
        assert core.stratification["attribute_mapping_id"] != ""
        assert core.stratification["difficulty"] == core.difficulty.value

        first, second = candidates
        if entity_condition == "same_name":
            assert first.object_key.namespace == second.object_key.namespace
            assert first.object_key.entity != second.object_key.entity
            assert first.object_key.entity.rsplit("_", 1)[-1] == second.object_key.entity.rsplit("_", 1)[-1]
            assert core.stratification["entity_mapping_id"].startswith(
                "same_name_group_v1:"
            )
        elif entity_condition == "namespace_collision":
            assert first.object_key.namespace != second.object_key.namespace
            assert first.object_key.entity == second.object_key.entity
            assert first.object_key.attribute == second.object_key.attribute
            assert core.stratification["namespace_evidence"].startswith(
                "unqualified:"
            )
        elif entity_condition == "alias":
            alias_surface = core.surface_references[0].normalized_text.split(".", 1)[0]
            assert core.stratification["entity_mapping_id"] == (
                f"reviewed_alias_v1:{alias_surface}->{first.object_key.entity}"
            )
            assert alias_surface not in {
                candidate.object_key.entity for candidate in candidates
            }
        else:
            assert core.stratification["entity_mapping_id"].startswith(
                "exact_entity_v1:"
            )

        if attribute_condition == "paraphrase":
            attribute_surface = core.surface_references[0].normalized_text.split(
                ".", 1
            )[1]
            assert core.stratification["attribute_mapping_id"].startswith(
                "reviewed_attribute_paraphrase_v1:"
            )
            assert attribute_surface != first.object_key.attribute
            assert (
                f"{attribute_surface}->{first.object_key.attribute}"
                in core.stratification["attribute_mapping_id"]
            )
        elif attribute_condition == "near_name":
            assert core.stratification["near_name_evidence"].startswith(
                "noncanonical_attribute:"
            )
            assert core.surface_references[0].candidate_ids == ()
        else:
            assert core.stratification["attribute_mapping_id"].startswith(
                "exact_attribute_v1:"
            )


def test_family_c_surface_and_namespace_evidence_agree_for_every_entity_condition(
    cores,
):
    representatives = {
        entity_condition: next(
            core
            for core in cores
            if core.stratification["entity_condition"] == entity_condition
            and core.stratification["attribute_condition"] == "exact"
        )
        for entity_condition in (
            "distinct",
            "same_name",
            "alias",
            "namespace_collision",
        )
    }

    for entity_condition, core in representatives.items():
        entity_surface = core.surface_references[0].normalized_text.split(".", 1)[0]
        candidates = core.reference_candidates
        first = candidates[0].object_key
        evidence = core.stratification["namespace_evidence"]

        if entity_condition == "distinct":
            assert entity_surface == f"{first.namespace}:{first.entity}"
            assert evidence == f"qualified:{first.namespace}"
        elif entity_condition in {"same_name", "alias"}:
            assert ":" not in entity_surface
            assert len({candidate.object_key.namespace for candidate in candidates}) == 1
            assert evidence == (
                f"unqualified_with_shared_namespace:{first.namespace}"
            )
        else:
            second = candidates[1].object_key
            assert ":" not in entity_surface
            assert first.entity == second.entity == entity_surface
            assert first.namespace != second.namespace
            assert evidence == (
                f"unqualified:{first.entity}@{first.namespace}|{second.namespace}"
            )


def test_unique_answers_equal_selected_candidate_replay_state(cores, config):
    context = GenerationContext(config=config, code_revision="family-c-test")
    for core in cores:
        task = render_core(
            core,
            split=Split.TEST,
            surface_variant=0,
            context=context,
        )
        query_id = task.queries[0].query_id
        canonical = task.gold.canonical_answers[query_id]
        if canonical.resolution_status is not ReferenceResolutionStatus.UNIQUE:
            assert canonical.value is None
            continue
        selected = next(
            candidate
            for candidate in task.queries[0].reference_candidates
            if candidate.candidate_id == canonical.selected_candidate_ids[0]
        )
        replay = replay_actions(task.gold.actions)
        assert canonical.value == replay.final_state[selected.object_key.canonical_id]


def test_family_c_core_id_excludes_administrative_coordinates(
    config,
    monkeypatch,
):
    monkeypatch.setattr(
        family_c_module,
        "_entity_spec",
        lambda *_args: (
            "personal:friend_alex",
            (("personal", "friend_alex"), ("personal", "friend_jordan")),
            "exact_entity_v1:friend_alex",
            "qualified:personal",
            "exact_qualified_entity",
        ),
    )
    monkeypatch.setattr(
        family_c_module,
        "_attribute_spec",
        lambda *_args: (
            "city",
            "city",
            "exact_attribute_v1:city",
            "reviewed_match:city->city",
            "exact_attribute",
        ),
    )
    monkeypatch.setattr(
        family_c_module,
        "_candidate_values",
        lambda *_args: ("Berlin", "Lisbon"),
    )

    first = family_c_module._build_core(
        config,
        core_index=3,
        cell_index=0,
        example_index=3,
        entity_condition="distinct",
        attribute_condition="exact",
    )
    second = family_c_module._build_core(
        config,
        core_index=103,
        cell_index=10,
        example_index=93,
        entity_condition="distinct",
        attribute_condition="exact",
    )

    assert first.core_index != second.core_index
    assert first.stratification["cell_index"] != second.stratification["cell_index"]
    assert first.stratification["cell_example_index"] != second.stratification[
        "cell_example_index"
    ]
    assert [candidate.candidate_id for candidate in first.reference_candidates] != [
        candidate.candidate_id for candidate in second.reference_candidates
    ]
    assert first.surface_references[0].reference_id != (
        second.surface_references[0].reference_id
    )
    assert first.events == second.events
    assert [candidate.object_key for candidate in first.reference_candidates] == [
        candidate.object_key for candidate in second.reference_candidates
    ]
    assert first.canonical_answer.value == second.canonical_answer.value
    assert first.core_id == second.core_id


def test_family_c_is_deterministic_with_unique_ids_and_hashes(config, cores):
    regenerated = generate_family_c_cores(config)
    assert [core.model_dump(mode="json") for core in cores] == [
        core.model_dump(mode="json") for core in regenerated
    ]
    assert len({core.core_id for core in cores}) == 120
    assert len({core.trajectory_id for core in cores}) == 120
    assert len({sha256_model(core) for core in cores}) == 120


def test_all_family_c_surface_variants_validate_and_share_semantic_hash(config, cores):
    context = GenerationContext(config=config, code_revision="family-c-test")
    rendered_count = 0
    for core in cores:
        tasks = [
            render_core(
                core,
                split=Split.TEST,
                surface_variant=surface_variant,
                context=context,
            )
            for surface_variant in range(3)
        ]
        rendered_count += len(tasks)
        assert len({task.task_id for task in tasks}) == 3
        assert len({semantic_task_hash(task) for task in tasks}) == 1
        for task in tasks:
            assert validate_task(task).valid
            assert validate_gold_replay(task).valid
    assert rendered_count == 360


def _renamed_reference_ids(core):
    payload = core.model_dump(mode="python")
    replacements = {}
    for index, candidate in enumerate(payload["reference_candidates"]):
        old_id = candidate["candidate_id"]
        new_id = f"renamed_candidate_{index}"
        candidate["candidate_id"] = new_id
        replacements[old_id] = new_id
    for reference in payload["surface_references"]:
        reference["reference_id"] = "renamed_reference"
        reference["candidate_ids"] = [
            replacements[candidate_id]
            for candidate_id in reference["candidate_ids"]
        ]
    payload["canonical_answer"]["selected_candidate_ids"] = [
        replacements[candidate_id]
        for candidate_id in payload["canonical_answer"]["selected_candidate_ids"]
    ]
    for event in payload["events"]:
        candidate_id = event["metadata"].get("candidate_id")
        if candidate_id is not None:
            event["metadata"]["candidate_id"] = replacements[candidate_id]
    return type(core).model_validate(payload)


def _changed_object_types(core):
    payload = core.model_dump(mode="python")
    for event in payload["events"]:
        for key in event["object_keys"]:
            key["object_type"] = "classification_only"
    for key in payload["query_targets"]:
        key["object_type"] = "classification_only"
    for candidate in payload["reference_candidates"]:
        candidate["object_key"]["object_type"] = "classification_only"
    return type(core).model_validate(payload)


def test_family_c_semantic_hash_excludes_linked_ids_and_object_type(config, cores):
    context = GenerationContext(config=config, code_revision="family-c-test")
    core = next(
        item
        for item in cores
        if item.canonical_answer.resolution_status
        is ReferenceResolutionStatus.UNIQUE
    )
    variants = (core, _renamed_reference_ids(core), _changed_object_types(core))
    renamed_ids = variants[1]
    changed_types = variants[2]
    assert [candidate.candidate_id for candidate in core.reference_candidates] != [
        candidate.candidate_id for candidate in renamed_ids.reference_candidates
    ]
    assert [
        reference.reference_id for reference in core.surface_references
    ] != [
        reference.reference_id for reference in renamed_ids.surface_references
    ]
    assert {
        candidate.object_key.object_type
        for candidate in core.reference_candidates
    } != {
        candidate.object_key.object_type
        for candidate in changed_types.reference_candidates
    }
    assert {core.core_id, renamed_ids.core_id, changed_types.core_id} == {
        core.core_id
    }

    tasks = [
        render_core(
            item,
            split=Split.TEST,
            surface_variant=0,
            context=context,
        )
        for item in variants
    ]

    assert len({semantic_task_hash(task) for task in tasks}) == 1
    original_key = core.reference_candidates[0].object_key
    changed_key = variants[2].reference_candidates[0].object_key
    assert original_key == changed_key
    assert original_key.canonical_id == changed_key.canonical_id


def test_family_c_resolved_profiles_report_no_cross_slot_interleaving(config, cores):
    context = GenerationContext(config=config, code_revision="family-c-test")
    for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        core = next(item for item in cores if item.difficulty is difficulty)
        task = render_core(
            core,
            split=Split.TEST,
            surface_variant=0,
            context=context,
        )
        assert task.metadata.resolved_profile["cross_slot_interleaving"] == 0.0


def _run_for(task, disposition: AnswerDisposition, value=None) -> TaskRunRecord:
    query_id = task.queries[0].query_id
    return TaskRunRecord(
        task_id=task.task_id,
        adapter_id="family-c-test-adapter",
        run_id="family-c-test-run",
        answer_predictions=[
            AnswerPrediction(
                query_id=query_id,
                raw_output=(
                    "ABSTAIN"
                    if disposition is AnswerDisposition.ABSTAINED
                    else str(value)
                ),
                disposition=disposition,
                parsed_answer=(
                    value if disposition is AnswerDisposition.ANSWERED else None
                ),
                format_valid=True,
            )
        ],
        parser_extractor_provenance=ParserExtractorProvenance(
            action_parser_version="test",
            answer_parser_version="test",
            memory_entry_extractor_version="test",
            redaction_policy_version="test",
        ),
        completion_status=CompletionStatus.COMPLETED,
    )


def _score(task, disposition: AnswerDisposition, value=None):
    return score_task(
        task,
        _run_for(task, disposition, value),
        AdapterCapabilities(),
        ScorerConfig(
            value_normalization_profile="typed_exact_v1",
            answer_normalization_profile="normalized_exact_v1",
            requested_metric_fields=(_METRIC,),
        ),
    )


def test_generated_family_c_reference_resolution_scoring(config, cores):
    context = GenerationContext(config=config, code_revision="family-c-test")
    unique = next(
        core
        for core in cores
        if core.canonical_answer.resolution_status
        is ReferenceResolutionStatus.UNIQUE
    )
    ambiguous = next(
        core
        for core in cores
        if core.canonical_answer.resolution_status
        is ReferenceResolutionStatus.AMBIGUOUS
    )
    no_match = next(
        core
        for core in cores
        if core.canonical_answer.resolution_status
        is ReferenceResolutionStatus.NO_MATCH
    )
    unique_task, ambiguous_task, no_match_task = [
        render_core(
            core,
            split=Split.TEST,
            surface_variant=0,
            context=context,
        )
        for core in (unique, ambiguous, no_match)
    ]

    correct_unique = _score(
        unique_task,
        AnswerDisposition.ANSWERED,
        unique.canonical_answer.value,
    )
    correct_ambiguous = _score(
        ambiguous_task,
        AnswerDisposition.ABSTAINED,
    )
    correct_no_match = _score(no_match_task, AnswerDisposition.ABSTAINED)
    wrong_guess = _score(
        ambiguous_task,
        AnswerDisposition.ANSWERED,
        "guessed-value",
    )
    unjustified = _score(unique_task, AnswerDisposition.ABSTAINED)

    assert correct_unique.answer_scores.reference_resolution_accuracy == 1.0
    assert correct_ambiguous.answer_scores.reference_resolution_accuracy == 1.0
    assert correct_no_match.answer_scores.reference_resolution_accuracy == 1.0
    assert wrong_guess.answer_scores.reference_resolution_accuracy == 0.0
    assert "wrong_reference_guess" in wrong_guess.failure_flags
    assert unjustified.answer_scores.reference_resolution_accuracy == 0.0
    assert "unjustified_abstention" in unjustified.failure_flags


def test_family_a_and_b_generation_regressions(config):
    family_a = generate_family_a_cores(config)
    family_b = generate_family_b_cores(config)

    assert len(family_a) == 120
    assert len(family_b) == 120
    assert {core.task_family for core in family_a} == {
        TaskFamily.REPEATED_SAME_SLOT
    }
    assert {core.task_family for core in family_b} == {
        TaskFamily.INTERLEAVED_MULTI_SLOT
    }


def test_family_c_rejects_malformed_or_unsupported_configuration(config):
    with pytest.raises(TypeError):
        generate_family_c_cores(object())
    with pytest.raises(ValueError, match="120"):
        generate_family_c_cores(config.model_copy(update={"cores_per_family": 1}))

    family = config.families.entity_attribute_grounding.model_copy(
        update={"entity_conditions": ["alias"]}
    )
    families = config.families.model_copy(
        update={"entity_attribute_grounding": family}
    )
    unsupported = config.model_copy(update={"families": families})
    with pytest.raises(ValueError, match="entity_conditions"):
        generate_family_c_cores(unsupported)


def test_family_c_generator_is_publicly_exported():
    from mub.vnext import generation

    assert generation.generate_family_c_cores is generate_family_c_cores
