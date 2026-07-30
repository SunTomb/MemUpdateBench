from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

import mub.vnext.validation as validation_api

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import EvaluationMode, EventRole, Operation, SourceType, Split
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash
from mub.vnext.legacy import (
    LEGACY_CAVEATS,
    LEGACY_NAMESPACES,
    compile_legacy_episode,
    load_evomemory_dataset,
)
from mub.vnext.legacy import dataset as dataset_module
from mub.vnext.validation import validate_task_semantics
from mub.vnext.validation.replay import replay_actions, validate_distractors, validate_gold_replay
from mub.vnext.validation.split import validate_splits
from mub.vnext.validation.task import validate_task


FIXTURE = Path(__file__).parent / "fixtures" / "legacy" / "p63_dataset_minimal.json"


def _episodes() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _compile(episode: dict, *, split: Split = Split.TEST, index: int = 0):
    return compile_legacy_episode(
        episode,
        source_path=FIXTURE,
        source_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        split=split,
        example_index=index,
        legacy_phase="P6.3",
    )


def _generator_record() -> dict:
    return {
        "events": [
            "User says: my friend Alex visited Ningbo last year.",
            "User says: my friend Alex lives in Wuxi.",
            "User says: my manager Alex moved to Dalian.",
            "User says: This came up during a long planning chat, but no facts changed.",
            "User says: my friend Alex moved to Suzhou.",
        ],
        "question": "Where does my friend Alex currently live?",
        "answer": "Suzhou",
        "entity": "friend_alex",
        "attribute": "location",
        "value": "Suzhou",
        "latest_event_idx": 4,
        "category": "update_frequency_hard_evolution_tracking",
        "stress_type": "update_frequency_hard",
        "k_updates": 2,
        "distractor_level": "same_name_multi_entity",
        "noop_level": "semantic_near_miss",
        "num_events": 5,
        "num_target_updates": 2,
        "num_updates": 5,
    }

@pytest.mark.parametrize("fixture_index", [0, 1])
def test_fixture_compilation_is_reachable_deterministic_and_immutable(fixture_index: int) -> None:
    episode = _episodes()[fixture_index]
    before = copy.deepcopy(episode)
    first = _compile(episode, index=fixture_index)
    second = _compile(episode, index=fixture_index)

    assert first.task_id == second.task_id
    assert [event.event_id for event in first.events] == [event.event_id for event in second.events]
    assert [action.action_id for action in first.gold.actions] == [action.action_id for action in second.gold.actions]
    assert [query.query_id for query in first.queries] == [query.query_id for query in second.queries]
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.source.raw_hash == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert first.source.normalized_hash == second.source.normalized_hash
    assert first.metadata.legacy_provenance == second.metadata.legacy_provenance
    assert LEGACY_CAVEATS["p63_split_leakage"] in first.metadata.legacy_provenance.known_caveats
    assert episode == before


def test_compiler_emits_truthful_deterministic_generator_provenance() -> None:
    first = _compile(_episodes()[0])
    second = _compile(_episodes()[0])
    generator = first.source.generator

    assert generator is not None
    assert generator.model_dump() == {
        "generator_name": "legacy_p63_episode_compiler",
        "seed": 0,
        "config_sha256": dataset_module._CONFIG_HASH,
        "code_revision": "legacy-compatibility-import",
        "compiler_version": dataset_module.COMPILER_VERSION,
    }
    assert second.source.generator == generator
    assert dataset_module._CONFIG["randomness"] == "none"
    assert dataset_module._CONFIG["seed_convention"] == "zero_means_no_rng"

    legacy = first.metadata.legacy_provenance
    assert legacy.source_artifact_path == str(FIXTURE)
    assert legacy.source_artifact_hash == first.source.raw_hash
    assert first.source.provenance == {"normalization_version": "semantic-source-v1"}


def test_ids_and_semantic_core_ignore_split_index_and_surface_fields() -> None:
    episode = _episodes()[0]
    base = _compile(episode, split=Split.TEST, index=0)
    changed = copy.deepcopy(episode)
    changed["question"] = "  " + changed["question"] + "  "
    changed["category"] = "surface-only"
    changed["stress_type"] = "surface-only"
    altered = _compile(changed, split=Split.DEV, index=99)

    assert base.metadata.split_key.semantic_core_id == altered.metadata.split_key.semantic_core_id
    assert base.task_id != altered.task_id
    assert [x.event_id for x in base.events] == [x.event_id for x in altered.events]
    assert [x.action_id for x in base.gold.actions] == [x.action_id for x in altered.gold.actions]
    assert [x.query_id for x in base.queries] == [x.query_id for x in altered.queries]


def test_compiler_emits_exact_slot_identity_counts_roles_actions_and_answers() -> None:
    k1, k2 = [_compile(row, index=i) for i, row in enumerate(_episodes())]
    key = k2.target_objects[0]
    assert key.namespace == "default"
    assert key.entity == "friend_alex"
    assert key.attribute == "location"
    assert key.subkey is None
    assert key.object_type == "slot"
    assert k1.metadata.extra["legacy_num_updates"] == 3
    assert k1.metadata.extra["num_events"] == 3
    assert k1.metadata.extra["num_target_updates"] == 1
    assert [event.role for event in k1.events] == [
        EventRole.NOOP_NEAR_MISS,
        EventRole.SAME_NAME_OTHER_ENTITY,
        EventRole.LATEST_GOLD,
    ]
    assert [action.operation for action in k1.gold.actions] == [Operation.NOOP, Operation.NOOP, Operation.ADD]
    assert [action.operation for action in k2.gold.actions] == [Operation.ADD, Operation.NOOP, Operation.NOOP, Operation.UPDATE]
    assert all(len(event.gold_action_ids) == 1 for event in k2.events)
    assert k2.gold.final_state[key.canonical_id] == "Suzhou"
    assert k2.gold.version_history[key.canonical_id] == ["Wuxi", "Suzhou"]
    assert k2.gold.gold_answers[k2.queries[0].query_id] == "Suzhou"
    assert "gold_answers" not in k2.queries[0].metadata
    assert k2.queries[0].answer_schema.value == "string"


def test_compiler_replays_and_validates_canonical_task() -> None:
    task = _compile(_episodes()[1], index=1)
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
    replay = replay_actions(task.gold.actions)
    assert replay.final_state == task.gold.final_state
    assert {key: list(values) for key, values in replay.version_history.items()} == task.gold.version_history


@pytest.mark.parametrize("field", ["entity", "attribute", "latest_event_idx"])
def test_missing_semantic_fields_fail_with_source_and_example_context(field: str) -> None:
    episode = _episodes()[0]
    episode.pop(field)
    with pytest.raises(ValueError, match=field):
        _compile(episode)


def test_malformed_count_and_unsupported_phase_fail_path_rich() -> None:
    episode = _episodes()[0]
    episode["num_events"] = "3"
    with pytest.raises(ValueError, match="num_events"):
        _compile(episode)
    with pytest.raises(ValueError, match="legacy_phase"):
        compile_legacy_episode(
            _episodes()[0], source_path=FIXTURE, source_sha256="0" * 64,
            split=Split.TEST, example_index=0, legacy_phase="P6.9",
        )


def test_legacy_identity_is_only_in_legacy_provenance() -> None:
    task = _compile(_episodes()[0])
    source_provenance = task.source.provenance
    assert set(source_provenance) <= {"normalization_version"}
    legacy = task.metadata.legacy_provenance
    assert legacy.legacy_metric_namespace == "legacy_p63"
    assert legacy.legacy_phase == "P6.3"
    assert legacy.source_artifact_path == str(FIXTURE)
    assert legacy.source_artifact_hash == task.source.raw_hash


def test_generation_config_tracks_semantic_projection_domain() -> None:
    task = _compile(_episodes()[0])
    assert dataset_module._CONFIG["semantic_identity_version"] == "semantic-core-v4"
    assert task.metadata.generation_config_hash == dataset_module._CONFIG_HASH


def test_malformed_source_anchor_source_id_is_rejected_after_task_validation() -> None:
    episode = _episodes()[0]
    text = episode["events"][0]
    episode["events"][0] = {"text": text, "source_anchor": {"source_id": "wrong"}}
    with pytest.raises(ValueError, match="source_anchor"):
        _compile(episode)


def test_malformed_source_anchor_event_id_is_rejected_after_task_validation() -> None:
    episode = _episodes()[0]
    text = episode["events"][0]
    episode["events"][0] = {"text": text, "source_anchor": {"event_id": "missing"}}
    with pytest.raises(ValueError, match="source_anchor"):
        _compile(episode)


def test_valid_source_anchor_fixture_still_compiles() -> None:
    task = _compile(_episodes()[0])
    assert validate_task(task).valid


def test_legacy_namespace_registry_is_exact_and_immutable() -> None:
    assert dict(LEGACY_NAMESPACES) == {
        "p63": "legacy_p63", "p65": "legacy_p65", "p68_p70": "legacy_p68_p70",
        "p80_p82": "legacy_p80_p82", "p83": "legacy_p83", "p84": "legacy_p84",
        "p85_api_replacement": "legacy_p85_api_replacement",
    }
    with pytest.raises(TypeError):
        LEGACY_NAMESPACES["p63"] = "alias"


def test_latest_event_must_be_terminal_target_event() -> None:
    episode = _episodes()[1]
    episode["latest_event_idx"] = 0
    with pytest.raises(ValueError, match="latest_event_idx"):
        _compile(episode, index=1)


def test_train_assets_compile_but_split_validation_exposes_overlap() -> None:
    episode = _episodes()[0]
    train = _compile(episode, split=Split.TRAIN)
    test = _compile(episode, split=Split.TEST)
    assert train.metadata.split == Split.TRAIN
    assert train.metadata.legacy_provenance.legacy_split_id == "train"
    assert LEGACY_CAVEATS["p63_split_leakage"] in train.metadata.legacy_provenance.known_caveats
    assert train.metadata.split_key.semantic_core_id == test.metadata.split_key.semantic_core_id
    report = validate_splits((train, test))
    assert any(issue.code == "group_leakage_training" for issue in report.issues)


def test_overlapping_legacy_role_markers_fail_instead_of_precedence() -> None:
    episode = _episodes()[0]
    episode["same_name_distractor"]["event_idx"] = episode["semantic_near_miss"]["event_idx"]
    with pytest.raises(ValueError, match="overlap"):
        _compile(episode)


def test_private_and_raw_payload_fields_never_enter_canonical_extra() -> None:
    episode = _episodes()[0]
    episode["private_payload"] = {"api_key": "should-not-survive"}
    episode["raw_response"] = "secret raw response"
    task = _compile(episode)
    extra = task.metadata.extra
    assert "private_payload" not in extra
    assert "raw_response" not in extra
    assert "legacy_summary" not in extra
    assert extra["legacy_num_updates"] == 3
    assert extra["num_events"] == 3


def test_same_name_distractor_rejects_target_entity_other_attribute() -> None:
    episode = _episodes()[0]
    episode.pop("semantic_near_miss")
    episode["same_name_distractor"] = {
        "entity": "friend_alex",
        "surface_name": "Alex",
        "event_idx": 0,
    }
    episode["events"][0] = "User says: my friend Alex prefers tea."
    with pytest.raises(ValueError, match="same_name_distractor"):
        _compile(episode)

    episode = _episodes()[0]
    episode["semantic_near_miss"]["entity"] = "manager_alex"
    with pytest.raises(ValueError, match="semantic_near_miss"):
        _compile(episode)

    episode = _episodes()[0]
    episode["same_name_distractor"]["entity"] = "friend_alex"
    with pytest.raises(ValueError, match="same_name_distractor"):
        _compile(episode)


def test_semantic_core_projection_changes_for_semantic_dimensions_but_not_surface_fields() -> None:
    base = _compile(_episodes()[1], index=1)
    surface = _episodes()[1]
    surface["question"] += " Please answer briefly."
    surface["category"] = "surface-only"
    assert _compile(surface, index=99).metadata.split_key.semantic_core_id == base.metadata.split_key.semantic_core_id

    anchor_changed = _episodes()[1]
    anchor_changed["events"] = [
        {"text": text, "source_anchor": {"start_char": i * 10, "end_char": i * 10 + len(text)}}
        for i, text in enumerate(anchor_changed["events"])
    ]
    assert _compile(anchor_changed, index=1).metadata.split_key.semantic_core_id != base.metadata.split_key.semantic_core_id

    object_changed = _episodes()[1]
    object_changed["entity"] = "friend_bob"
    object_changed["events"] = [text.replace("Alex", "Bob") for text in object_changed["events"]]
    object_changed["answer"] = "Suzhou"
    object_changed["semantic_near_miss"]["entity"] = "friend_bob"
    object_changed["same_name_distractor"]["entity"] = "manager_bob"
    object_changed["same_name_distractor"]["surface_name"] = "Bob"
    assert _compile(object_changed, index=1).metadata.split_key.semantic_core_id != base.metadata.split_key.semantic_core_id

    value_changed = _episodes()[1]
    value_changed["events"][-1] = value_changed["events"][-1].replace("Suzhou", "Hangzhou")
    value_changed["answer"] = "Hangzhou"
    assert _compile(value_changed, index=1).metadata.split_key.semantic_core_id != base.metadata.split_key.semantic_core_id


def test_generator_shaped_record_loads_and_compiles_without_enrichment(tmp_path: Path) -> None:
    path = tmp_path / "p63_generator.json"
    path.write_text(json.dumps([_generator_record()]), encoding="utf-8")
    loaded = load_evomemory_dataset(path)
    assert "episode_id" not in loaded[0]
    assert "semantic_near_miss" not in loaded[0]
    task = compile_legacy_episode(
        loaded[0], source_path=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        split=Split.TEST, example_index=0, legacy_phase="P6.3",
    )
    assert [action.operation for action in task.gold.actions if action.operation != Operation.NOOP] == [Operation.ADD, Operation.UPDATE]
    assert task.events[0].role == EventRole.NEUTRAL
    assert task.events[2].role == EventRole.NEUTRAL
    assert validate_task(task).valid and validate_gold_replay(task).valid


def test_episode_scoped_parser_resolves_alias_followup() -> None:
    episode = {
        **_generator_record(),
        "events": [
            "User says: my manager Alex lives in Wuxi.",
            "User says: Alex moved to Suzhou.",
        ],
        "entity": "manager_alex",
        "latest_event_idx": 1,
        "num_events": 2,
        "num_target_updates": 2,
        "num_updates": 2,
        "k_updates": 2,
    }
    task = _compile(episode)
    assert [action.operation for action in task.gold.actions] == [Operation.ADD, Operation.UPDATE]
    assert task.gold.version_history[task.target_objects[0].canonical_id] == ["Wuxi", "Suzhou"]


def test_source_path_spelling_does_not_change_semantic_hashes() -> None:
    episode = _generator_record()
    digest = "a" * 64
    first = compile_legacy_episode(
        episode, source_path=Path("relative/p63.json"), source_sha256=digest,
        split=Split.TEST, example_index=7, legacy_phase="P6.3",
    )
    second = compile_legacy_episode(
        episode, source_path=Path(r"C:\\data\\p63.json"), source_sha256=digest,
        split=Split.TEST, example_index=7, legacy_phase="P6.3",
    )
    assert first.metadata.split_key.semantic_core_id == second.metadata.split_key.semantic_core_id
    assert semantic_task_hash(first) == semantic_task_hash(second)
    assert all("source" not in event.source_anchor for event in first.events)


@pytest.mark.parametrize("reserved", ["source", "source_id", "event_id", "document_id", "section_id", "paragraph"])
def test_raw_anchor_cannot_override_compiler_owned_fields(reserved: str) -> None:
    episode = _generator_record()
    episode["events"][0] = {"text": episode["events"][0], "source_anchor": {reserved: "forged"}}
    with pytest.raises(ValueError, match=r"events\[0\].source_anchor"):
        _compile(episode)


@pytest.mark.parametrize(
    "anchor",
    [
        {"start_char": 0},
        {"start_char": -1, "end_char": 2},
        {"start_char": 3, "end_char": 2},
        {"start_char": float("nan"), "end_char": 2},
        {"span": {"start": 0, "end": float("inf")}},
    ],
)
def test_malformed_or_nonfinite_anchor_is_rejected(anchor: dict) -> None:
    episode = _generator_record()
    episode["events"][0] = {"text": episode["events"][0], "source_anchor": anchor}
    with pytest.raises(ValueError, match=r"events\[0\].source_anchor"):
        _compile(episode)


@pytest.mark.parametrize("key", ["legacy_role", "private_payload", "raw_response", "secret"])
def test_event_metadata_rejects_forged_or_sensitive_keys(key: str) -> None:
    episode = _generator_record()
    episode["events"][0] = {"text": episode["events"][0], "metadata": {key: "forged"}}
    with pytest.raises(ValueError, match=r"events\[0\].metadata"):
        _compile(episode)


def test_event_metadata_rejects_nested_sensitive_payload() -> None:
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0],
        "metadata": {"annotation": {"private_payload": "secret"}},
    }
    with pytest.raises(ValueError, match=r"events\[0\].metadata.annotation"):
        _compile(episode)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_event_metadata_rejects_nonfinite_values(value: float) -> None:
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0],
        "metadata": {"annotation": value},
    }
    with pytest.raises(ValueError, match=r"events\[0\].metadata.annotation"):
        _compile(episode)


def test_event_metadata_retains_only_explicit_benign_namespace() -> None:
    episode = _generator_record()
    episode["events"][0] = {"text": episode["events"][0], "metadata": {"annotation": "benign"}}
    task = _compile(episode)
    assert task.events[0].metadata == {"legacy_role": EventRole.NEUTRAL.value, "legacy_event": {"annotation": "benign"}}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_answer_is_rejected_before_canonical_serialization(value: float) -> None:
    episode = _generator_record()
    episode["answer"] = value
    with pytest.raises(ValueError, match="answer"):
        _compile(episode)


def test_valid_compiled_task_always_has_canonical_json_bytes() -> None:
    assert canonical_json_bytes(_compile(_generator_record()))


def test_unrelated_parsed_entity_is_neutral_not_same_name() -> None:
    task = _compile(_generator_record())
    assert task.events[2].role == EventRole.NEUTRAL


def test_duplicate_current_write_keeps_mutation_and_history() -> None:
    episode = {
        **_generator_record(),
        "events": [
            "User says: my friend Alex lives in Wuxi.",
            "User says: my friend Alex moved to Wuxi.",
            "User says: my friend Alex moved to Suzhou.",
        ],
        "latest_event_idx": 2,
        "num_events": 3,
        "num_target_updates": 3,
        "num_updates": 3,
        "k_updates": 3,
    }
    task = _compile(episode)
    assert [event.role for event in task.events] == [EventRole.STALE_SAME_SLOT, EventRole.DUPLICATE_CURRENT, EventRole.LATEST_GOLD]
    assert [action.operation for action in task.gold.actions] == [Operation.ADD, Operation.UPDATE, Operation.UPDATE]
    assert task.gold.version_history[task.target_objects[0].canonical_id] == ["Wuxi", "Wuxi", "Suzhou"]


def test_semantic_object_projection_ignores_object_type_metadata() -> None:
    slot = MemoryObjectKey(object_type="slot", namespace="default", entity="friend_alex", attribute="location")
    other = MemoryObjectKey(object_type="record", namespace="default", entity="friend_alex", attribute="location")
    assert dataset_module._semantic_object_projection(slot) == dataset_module._semantic_object_projection(other)


@pytest.mark.parametrize("explicit_mode", [None, "state_direct"])
def test_task11_uses_only_justified_state_direct_mode(explicit_mode) -> None:
    episode = _generator_record()
    if explicit_mode is not None:
        episode["evaluation_mode"] = explicit_mode
    task = _compile(episode)
    assert task.queries[0].evaluation_mode == EvaluationMode.STATE_DIRECT
    assert task.metadata.legacy_provenance.answer_mode is None
    assert LEGACY_CAVEATS["state_direct_oracle"] not in task.metadata.legacy_provenance.known_caveats
    assert LEGACY_CAVEATS["retrieved_prompt_legacy"] not in task.metadata.legacy_provenance.known_caveats


@pytest.mark.parametrize("answer_mode", ["slot_direct", "slot_prompt", "unknown"])
def test_task11_rejects_run_specific_legacy_answer_mode(answer_mode: str) -> None:
    episode = _generator_record()
    episode["answer_mode"] = answer_mode
    with pytest.raises(ValueError, match="answer_mode"):
        _compile(episode)


@pytest.mark.parametrize("evaluation_mode", ["retrieved_prompt", "native_system"])
def test_task11_rejects_non_state_direct_evaluation_mode(evaluation_mode: str) -> None:
    episode = _generator_record()
    episode["evaluation_mode"] = evaluation_mode
    with pytest.raises(ValueError, match="evaluation_mode"):
        _compile(episode)


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_updates", 4), ("k_updates", 3), ("k_updates", True)],
)
def test_legacy_count_contradictions_are_rejected(field: str, value) -> None:
    episode = _generator_record()
    episode[field] = value
    with pytest.raises(ValueError, match=field):
        _compile(episode)


def test_event_mapping_requires_exact_string_keys_before_sorting() -> None:
    episode = _generator_record()
    episode["events"][0] = {"text": episode["events"][0], 1: "hostile"}
    with pytest.raises(ValueError, match=r"example_index=0.*events\[0\]"):
        _compile(episode)


@pytest.mark.parametrize(
    "parser_result",
    [
        ["not", "a", "mapping"],
        {"entity": "", "attribute": "location", "value": "X", "event_idx": 0},
        {"entity": "friend_alex", "attribute": "location", "value": "X", "event_idx": 9},
        {"entity": "friend_alex", "attribute": "location", "value": float("nan"), "event_idx": 0},
    ],
)
def test_malformed_parser_results_fail_contextually(monkeypatch, parser_result) -> None:
    import scripts.eval_evomemory as legacy_evaluator

    monkeypatch.setattr(legacy_evaluator, "parse_event_slot", lambda *args, **kwargs: parser_result)
    with pytest.raises(ValueError, match=r"example_index=0.*events\[0\]"):
        _compile(_generator_record())


def test_parser_exception_is_chained_with_event_context(monkeypatch) -> None:
    import scripts.eval_evomemory as legacy_evaluator

    def explode(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(legacy_evaluator, "parse_event_slot", explode)
    with pytest.raises(ValueError, match=r"example_index=0.*events\[0\]") as captured:
        _compile(_generator_record())
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_answer_bearing_non_target_text_uses_audited_ambiguity_exception() -> None:
    episode = _generator_record()
    episode["events"][0] = "User says: my friend Alex visited Suzhou last year."
    task = _compile(episode)
    event = task.events[0]
    assert event.role == EventRole.NEUTRAL
    assert task.gold.actions[0].operation == Operation.NOOP
    assert event.metadata["allow_accepted_answer_ambiguity"] is True
    assert event.metadata["compatibility_rule"] == "non_target_accepted_answer_text_overlap_v1"
    assert "non_target_accepted_answer_text_overlap_v1" in task.metadata.extra["compatibility_policies"]
    strict_codes = {
        issue.code for issue in validate_task_semantics(task).issues
    }
    assert "distractor_text_contains_accepted_answer" in strict_codes
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
    assert not validate_distractors(task).valid


def _audited_overlap_payload() -> dict:
    episode = _generator_record()
    episode["events"][0] = "User says: my friend Alex visited Suzhou last year."
    return _compile(episode).model_dump(mode="json")


def test_coordinated_forgery_has_no_public_legacy_waiver() -> None:
    payload = _audited_overlap_payload()
    payload["metadata"]["legacy_provenance"] = None
    payload["source"]["generator"].update(
        generator_name="attacker_generator",
        code_revision="attacker-controlled",
        config_sha256="a" * 64,
    )
    payload["events"][0]["raw_text"] = "Attacker supplied current answer Suzhou"
    payload["events"][0]["normalized_text"] = "Attacker supplied current answer Suzhou"
    forged = type(_compile(_generator_record())).model_validate(payload)

    assert "distractor_text_contains_accepted_answer" in {
        issue.code for issue in validate_task_semantics(forged).issues
    }
    assert "distractor_text_contains_accepted_answer" in {
        issue.code for issue in validate_distractors(forged).issues
    }
    assert not hasattr(validation_api, "validate_legacy_task_semantics")
    assert not hasattr(validation_api, "DistractorValidationPolicy")
    public_parameters = inspect.signature(validate_distractors).parameters
    assert "policy" not in public_parameters
    assert "accepted_overlap_policy" not in public_parameters


def test_legacy_policy_rejects_structured_auxiliary_write_leak() -> None:
    payload = _audited_overlap_payload()
    event = payload["events"][0]
    action = payload["gold"]["actions"][0]
    query_key = payload["queries"][0]["target_object_keys"][0]
    auxiliary_key = {**query_key, "attribute": "historical_visit"}
    auxiliary = MemoryObjectKey.model_validate(auxiliary_key)
    answer = payload["gold"]["gold_answers"][payload["queries"][0]["query_id"]]
    payload["target_objects"].append(auxiliary_key)
    action.update(
        operation=Operation.ADD.value,
        target_object_keys=[auxiliary_key],
        value=answer,
        expected_effect={
            "canonical_id": auxiliary.canonical_id,
            "operation": Operation.ADD.value,
            "value": answer,
        },
    )
    payload["gold"]["final_state"][auxiliary.canonical_id] = answer
    payload["gold"]["version_history"][auxiliary.canonical_id] = [answer]
    payload["gold"]["expected_present_objects"].append(auxiliary_key)
    task = type(_compile(_generator_record())).model_validate(payload)

    codes = {issue.code for issue in validate_task_semantics(task).issues}
    assert "distractor_text_contains_accepted_answer" in codes


def test_legacy_policy_rejects_unmarked_gold_bearing_noop() -> None:
    payload = _audited_overlap_payload()
    payload["events"][0]["metadata"] = {"legacy_role": EventRole.NEUTRAL.value}
    task = type(_compile(_generator_record())).model_validate(payload)

    codes = {issue.code for issue in validate_task_semantics(task).issues}
    assert "distractor_text_contains_accepted_answer" in codes


def test_legacy_policy_rejects_forged_normalized_ambiguity_text() -> None:
    payload = _audited_overlap_payload()
    payload["events"][0]["normalized_text"] = "Forged neutral mention of Suzhou"
    task = type(_compile(_generator_record())).model_validate(payload)

    codes = {issue.code for issue in validate_task_semantics(task).issues}
    assert "distractor_text_contains_accepted_answer" in codes


def test_legacy_policy_rejects_unlinked_same_name_leak() -> None:
    task = _compile(
        {
            **_generator_record(),
            "events": [
                "User says: my friend Alex visited Suzhou last year.",
                *_generator_record()["events"][1:],
            ],
        }
    )
    event = task.events[0].model_copy(
        update={
            "role": EventRole.SAME_NAME_OTHER_ENTITY,
            "metadata": {
                **dict(task.events[0].metadata),
                "legacy_role": EventRole.SAME_NAME_OTHER_ENTITY.value,
            },
            "gold_action_ids": (),
        }
    )
    malformed = task.model_copy(update={"events": (event, *task.events[1:])})

    report = validate_task_semantics(malformed)
    codes = {issue.code for issue in report.issues}
    assert not report.valid
    assert "distractor_text_contains_accepted_answer" in codes


def test_cyclic_history_uses_historical_support_not_stale() -> None:
    episode = {
        **_generator_record(),
        "events": [
            "User says: my friend Alex lives in Wuxi.",
            "User says: my friend Alex moved to Suzhou.",
            "User says: my friend Alex moved to Wuxi.",
        ],
        "answer": "Wuxi",
        "value": "Wuxi",
        "latest_event_idx": 2,
        "num_events": 3,
        "num_target_updates": 3,
        "num_updates": 3,
        "k_updates": 3,
    }
    task = _compile(episode)
    assert [event.role for event in task.events] == [
        EventRole.HISTORICAL_SUPPORT,
        EventRole.STALE_SAME_SLOT,
        EventRole.LATEST_GOLD,
    ]
    assert [action.operation for action in task.gold.actions] == [Operation.ADD, Operation.UPDATE, Operation.UPDATE]
    assert task.gold.version_history[task.target_objects[0].canonical_id] == ["Wuxi", "Suzhou", "Wuxi"]
    assert validate_task(task).valid
    assert validate_gold_replay(task).valid
    assert validate_distractors(task).valid


def test_surface_paraphrases_preserve_both_semantic_hashes() -> None:
    base_episode = _episodes()[1]
    paraphrase = copy.deepcopy(base_episode)
    paraphrase["events"] = [
        text.replace("User says:", "During the chat, user says:")
        for text in paraphrase["events"]
    ]
    paraphrase["events"][1] = "During the chat, my friend Alex read a travel guide about Ningbo."
    base = _compile(base_episode, index=1)
    changed = _compile(paraphrase, index=1)
    assert [event.role for event in base.events] == [event.role for event in changed.events]
    assert [action.operation for action in base.gold.actions] == [action.operation for action in changed.gold.actions]
    assert base.gold.final_state == changed.gold.final_state
    assert base.gold.version_history == changed.gold.version_history
    assert base.metadata.split_key.semantic_core_id == changed.metadata.split_key.semantic_core_id
    assert semantic_task_hash(base) == semantic_task_hash(changed)
    base_other_artifact = compile_legacy_episode(
        base_episode, source_path=Path("base.json"), source_sha256="a" * 64,
        split=Split.TEST, example_index=1, legacy_phase="P6.3",
    )
    changed_other_artifact = compile_legacy_episode(
        paraphrase, source_path=Path("paraphrase.json"), source_sha256="b" * 64,
        split=Split.TEST, example_index=1, legacy_phase="P6.3",
    )
    assert base_other_artifact.metadata.split_key.semantic_core_id == changed_other_artifact.metadata.split_key.semantic_core_id
    assert semantic_task_hash(base_other_artifact) == semantic_task_hash(changed_other_artifact)


def test_raw_normalized_text_hash_is_rejected_as_surface_dependent() -> None:
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0],
        "source_anchor": {"normalized_text_sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="normalized_text_sha256"):
        _compile(episode)


def test_cross_phase_compilation_changes_only_legacy_provenance_identity() -> None:
    episode = _episodes()[0]
    source_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    p63 = compile_legacy_episode(
        episode, source_path=FIXTURE, source_sha256=source_hash,
        split=Split.TEST, example_index=0, legacy_phase="P6.3",
    )
    p65 = compile_legacy_episode(
        episode, source_path=FIXTURE, source_sha256=source_hash,
        split=Split.TEST, example_index=0, legacy_phase="P6.5",
    )
    assert p63.metadata.legacy_provenance != p65.metadata.legacy_provenance
    assert p63.task_id == p65.task_id
    assert p63.source.source_id == p65.source.source_id
    assert p63.metadata.split_key.source_group_id == p65.metadata.split_key.source_group_id
    assert p63.metadata.split_key.semantic_core_id == p65.metadata.split_key.semantic_core_id
    assert p63.metadata.split_key.trajectory_id == p65.metadata.split_key.trajectory_id
    assert [event.event_id for event in p63.events] == [event.event_id for event in p65.events]
    assert [action.action_id for action in p63.gold.actions] == [action.action_id for action in p65.gold.actions]
    assert [query.query_id for query in p63.queries] == [query.query_id for query in p65.queries]
    assert semantic_task_hash(p63) == semantic_task_hash(p65)


def test_dataset_and_run_identity_fields_are_not_duplicated_in_extra() -> None:
    episode = _generator_record()
    episode.update(
        category="dataset-identity-unique",
        stress_type="run-condition-unique",
        retrieval_policy="retrieval-identity-unique",
        context_order="context-identity-unique",
        condition="condition-identity-unique",
    )
    task = _compile(episode)
    analysis = task.metadata.extra["legacy_analysis"]
    assert set(analysis) == {"k_updates", "distractor_level", "noop_level"}
    legacy = task.metadata.legacy_provenance
    assert legacy.legacy_dataset_id == "dataset-identity-unique"
    assert legacy.legacy_run_condition_id == "run-condition-unique"
    dumped = json.dumps(task.model_dump(mode="json"), sort_keys=True)
    assert "retrieval-identity-unique" not in dumped
    assert "context-identity-unique" not in dumped
    assert "condition-identity-unique" not in dumped


def test_legacy_identity_values_are_path_aware_and_provenance_only() -> None:
    task = _compile(_episodes()[0])
    payload = task.model_dump(mode="json")
    legacy = payload["metadata"].pop("legacy_provenance")

    def walk(value, path=()):
        yield path, value
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, (*path, index))

    forbidden_fields = {
        "legacy_phase", "legacy_metric_namespace", "legacy_dataset_id",
        "legacy_run_condition_id", "answer_mode", "checkpoint_family",
        "training_seed", "memory_trajectory_id",
    }
    assert not any(path and path[-1] in forbidden_fields for path, _ in walk(payload))
    for field in forbidden_fields:
        legacy_value = legacy[field]
        if legacy_value is not None:
            assert not any(value == legacy_value for _, value in walk(payload))
    assert payload["metadata"]["split"] == legacy["legacy_split_id"]
    assert payload["source"]["raw_hash"] == legacy["source_artifact_hash"]
    assert payload["metadata"]["extra"]["legacy_num_updates"] == 3



class _HostileStr(str):
    pass


class _HostileInt(int):
    pass


class _ExplosivePrivate:
    def __deepcopy__(self, memo):
        raise AssertionError("ignored private payload was scanned")


class _HostileDict(dict):
    hook_called = False

    def _explode(self):
        type(self).hook_called = True
        raise AssertionError("hostile dict hook executed")

    def items(self):
        self._explode()

    def keys(self):
        self._explode()

    def __iter__(self):
        self._explode()


class _HostileList(list):
    hook_called = False

    def __iter__(self):
        type(self).hook_called = True
        raise AssertionError("hostile list hook executed")


@pytest.mark.parametrize(
    "value",
    [None, False, 7, {"private_payload": "secret"}, ["list"], "   ", _HostileStr("hostile"), "\ud800"],
    ids=["null", "false", "number", "mapping", "list", "blank", "subclass", "surrogate"],
)
@pytest.mark.parametrize("field", ["category", "stress_type"])
def test_provenance_identity_fields_reject_non_exact_strings(field: str, value) -> None:
    episode = _generator_record()
    episode[field] = value
    with pytest.raises(ValueError, match=field):
        _compile(episode)


def test_provenance_identity_fields_accept_exact_strings_and_absence() -> None:
    episode = _generator_record()
    episode["category"] = "dataset-valid"
    episode["stress_type"] = "condition-valid"
    task = _compile(episode)
    assert task.metadata.legacy_provenance.legacy_dataset_id == "dataset-valid"
    assert task.metadata.legacy_provenance.legacy_run_condition_id == "condition-valid"
    absent = _generator_record()
    absent.pop("category")
    absent.pop("stress_type")
    task = _compile(absent)
    assert task.metadata.legacy_provenance.legacy_dataset_id == "p63"
    assert task.metadata.legacy_provenance.legacy_run_condition_id is None


@pytest.mark.parametrize(
    "value",
    [None, False, _HostileStr("state_direct")],
    ids=["null", "bool", "subclass"],
)
def test_explicit_evaluation_mode_requires_exact_builtin_state_direct(value) -> None:
    episode = _generator_record()
    episode["evaluation_mode"] = value
    with pytest.raises(ValueError, match="evaluation_mode"):
        _compile(episode)


@pytest.mark.parametrize("phase", ["P6.3", "P6.5", "p68_p70"])
def test_documented_p6_episode_phases_compile(phase: str) -> None:
    task = compile_legacy_episode(
        _episodes()[0], source_path=FIXTURE,
        source_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        split=Split.TEST, example_index=0, legacy_phase=phase,
    )
    assert task.metadata.legacy_provenance.legacy_phase == phase


@pytest.mark.parametrize("phase", ["p80_p82", "p83", "p84", "p85_api_replacement"])
def test_p8_mechanism_and_api_phases_fail_closed(phase: str) -> None:
    with pytest.raises(ValueError, match="legacy_phase"):
        compile_legacy_episode(
            _episodes()[0], source_path=FIXTURE,
            source_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            split=Split.TEST, example_index=0, legacy_phase=phase,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k_updates", {"private_payload": "x"}),
        ("k_updates", [2]),
        ("k_updates", _HostileInt(2)),
        ("distractor_level", {"private_payload": "x"}),
        ("distractor_level", ["same"]),
        ("distractor_level", False),
        ("distractor_level", " "),
        ("distractor_level", _HostileStr("same")),
        ("noop_level", {"raw_response": "x"}),
        ("noop_level", 1),
        ("noop_level", "\ud800"),
        ("explicit_zero", False),
        ("explicit_zero", 1),
        ("explicit_zero", 0.0),
        ("explicit_zero", _HostileInt(0)),
        ("explicit_false", 0),
        ("explicit_false", True),
        ("explicit_false", None),
        ("explicit_null", False),
        ("explicit_null", 0),
        ("explicit_null", ""),
        ("explicit_null", {}),
    ],
)
def test_legacy_analysis_fields_reject_wrong_schema(field: str, value) -> None:
    episode = _generator_record()
    episode[field] = value
    with pytest.raises(ValueError, match=field):
        _compile(episode)


def test_legacy_analysis_fields_preserve_exact_valid_values() -> None:
    episode = _generator_record()
    episode.update(
        distractor_level="same_name_multi_entity",
        noop_level="semantic_near_miss",
        explicit_zero=0,
        explicit_false=False,
        explicit_null=None,
    )
    analysis = _compile(episode).metadata.extra["legacy_analysis"]
    assert analysis == {
        "distractor_level": "same_name_multi_entity",
        "explicit_false": False,
        "explicit_null": None,
        "explicit_zero": 0,
        "k_updates": 2,
        "noop_level": "semantic_near_miss",
    }


def test_same_name_marker_without_surface_name_cannot_bypass_shared_evidence() -> None:
    episode = _episodes()[0]
    episode["events"][1] = "User says: my manager Bob lives in Berlin."
    episode["same_name_distractor"] = {"entity": "manager_bob", "event_idx": 1}
    with pytest.raises(ValueError, match="same_name_distractor.*surface_name"):
        _compile(episode)


def test_same_name_marker_requires_name_shared_with_target_evidence() -> None:
    episode = _episodes()[0]
    episode["events"][1] = "User says: my manager Bob lives in Berlin."
    episode["same_name_distractor"] = {
        "entity": "manager_bob", "surface_name": "Bob", "event_idx": 1,
    }
    with pytest.raises(ValueError, match="same_name_distractor"):
        _compile(episode)


def test_strict_json_rejects_active_cycles_and_excessive_depth() -> None:
    cyclic = {}
    cyclic["self"] = cyclic
    episode = _generator_record()
    episode["answer"] = cyclic
    with pytest.raises(ValueError, match="answer"):
        _compile(episode)

    metadata_cycle = {}
    metadata_cycle["self"] = metadata_cycle
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0], "metadata": {"annotation": metadata_cycle},
    }
    with pytest.raises(ValueError, match=r"events\[0\].metadata"):
        _compile(episode)

    nested = "leaf"
    for _ in range(80):
        nested = [nested]
    episode = _generator_record()
    episode["answer"] = nested
    with pytest.raises(ValueError, match="depth|answer"):
        _compile(episode)


def _shared_dag(depth: int = 20):
    node = ["leaf"]
    for _ in range(depth):
        node = [node, node]
    return node


def test_strict_json_rejects_shared_dag_and_total_node_exhaustion() -> None:
    episode = _generator_record()
    episode["answer"] = _shared_dag()
    with pytest.raises(ValueError, match="answer.*repeated|answer.*shared"):
        _compile(episode)

    episode = _generator_record()
    episode["answer"] = list(range(dataset_module._MAX_JSON_NODES + 1))
    with pytest.raises(ValueError, match="answer.*budget"):
        _compile(episode)

    assert dataset_module._strict_json_copy(
        {"a": [1, {"b": 2}]}, "probe", FIXTURE, 0
    ) == {"a": [1, {"b": 2}]}


def test_strict_json_rejects_shared_dag_from_parser_and_event_metadata(monkeypatch) -> None:
    import scripts.eval_evomemory as legacy_evaluator

    parser_dag = _shared_dag()
    monkeypatch.setattr(
        legacy_evaluator,
        "parse_event_slot",
        lambda text, event_idx, resolver=None: {
            "entity": "friend_alex",
            "attribute": "location",
            "value": parser_dag,
            "event_idx": event_idx,
        },
    )
    with pytest.raises(ValueError, match=r"events\[0\].parsed_slot.value"):
        _compile(_generator_record())

    metadata_dag = _shared_dag()
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0],
        "metadata": {"annotation": metadata_dag},
    }
    with pytest.raises(ValueError, match=r"events\[0\].metadata.annotation"):
        _compile(episode)


@pytest.mark.parametrize("field", ["episode_id", "category"])
def test_identity_strings_reject_lone_surrogates(field: str) -> None:
    episode = _generator_record()
    episode[field] = "\ud800"
    with pytest.raises(ValueError, match=field):
        _compile(episode)


def test_event_text_rejects_lone_surrogate_contextually() -> None:
    episode = _generator_record()
    episode["events"][0] = "\ud800"
    with pytest.raises(ValueError, match=r"events\[0\]"):
        _compile(episode)


def test_ignored_private_payload_is_not_deep_scanned() -> None:
    episode = _generator_record()
    episode["private_payload"] = _ExplosivePrivate()
    assert _compile(episode)


def _assert_linear_target_membership(source: str) -> None:
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "target_index_set" for target in node.targets)
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name) and value.func.id == "frozenset"
    assert len(value.args) == 1
    assert isinstance(value.args[0], ast.Name) and value.args[0].id == "target_indices"

    membership_comparators = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(operator, (ast.In, ast.NotIn)) and isinstance(comparator, ast.Name):
                membership_comparators.append(comparator.id)
    assert "target_indices" not in membership_comparators
    assert membership_comparators.count("target_index_set") == 3


def test_target_membership_uses_frozenset_in_all_full_event_passes() -> None:
    source = inspect.getsource(dataset_module.compile_legacy_episode)
    _assert_linear_target_membership(source)
    mutated = source.replace(
        "target_index_set = frozenset(target_indices)",
        "target_index_set = list(target_indices)",
    )
    assert mutated != source
    with pytest.raises(AssertionError):
        _assert_linear_target_membership(mutated)


def _reset_hostile_hooks() -> None:
    _HostileDict.hook_called = False
    _HostileList.hook_called = False


def test_top_level_episode_rejects_hostile_dict_without_hooks() -> None:
    _reset_hostile_hooks()
    episode = _HostileDict(_generator_record())
    with pytest.raises(ValueError, match=r"example_index=0.*episode"):
        _compile(episode)
    assert _HostileDict.hook_called is False


def test_events_reject_hostile_list_without_hooks() -> None:
    _reset_hostile_hooks()
    episode = _generator_record()
    episode["events"] = _HostileList(episode["events"])
    with pytest.raises(ValueError, match=r"example_index=0.*events"):
        _compile(episode)
    assert _HostileList.hook_called is False


@pytest.mark.parametrize("boundary", ["event", "source_anchor", "metadata", "marker"])
def test_json_object_boundaries_reject_hostile_dict_without_hooks(boundary: str) -> None:
    _reset_hostile_hooks()
    episode = _generator_record()
    if boundary == "event":
        episode["events"][0] = _HostileDict(text=episode["events"][0])
        match = r"events\[0\]"
    elif boundary == "source_anchor":
        episode["events"][0] = {
            "text": episode["events"][0], "source_anchor": _HostileDict(),
        }
        match = r"events\[0\].source_anchor"
    elif boundary == "metadata":
        episode["events"][0] = {
            "text": episode["events"][0], "metadata": _HostileDict(annotation="x"),
        }
        match = r"events\[0\].metadata"
    else:
        episode["semantic_near_miss"] = _HostileDict(
            entity="friend_alex", attribute="location", event_idx=0,
        )
        match = "semantic_near_miss"
    with pytest.raises(ValueError, match=match):
        _compile(episode)
    assert _HostileDict.hook_called is False


def test_nested_json_rejects_hostile_list_without_hooks() -> None:
    _reset_hostile_hooks()
    episode = _generator_record()
    episode["events"][0] = {
        "text": episode["events"][0],
        "metadata": {"annotation": _HostileList(["x"])},
    }
    with pytest.raises(ValueError, match=r"events\[0\].metadata.annotation"):
        _compile(episode)
    assert _HostileList.hook_called is False


def test_parser_result_rejects_hostile_dict_without_hooks(monkeypatch) -> None:
    import scripts.eval_evomemory as legacy_evaluator

    _reset_hostile_hooks()
    result = _HostileDict(
        entity="friend_alex", attribute="location", value="Wuxi", event_idx=0,
    )
    monkeypatch.setattr(
        legacy_evaluator, "parse_event_slot", lambda *args, **kwargs: result,
    )
    with pytest.raises(ValueError, match=r"events\[0\].parsed_slot"):
        _compile(_generator_record())
    assert _HostileDict.hook_called is False


def test_fixed_seed_real_p63_dev_test_corpus_compiles_all_records(tmp_path: Path) -> None:
    from scripts.prepare_data import prepare_update_frequency_hard_evomemory

    prepare_update_frequency_hard_evomemory(str(tmp_path), seed=67)
    totals = Counter()
    for split_name, split in (("dev", Split.DEV), ("test", Split.TEST)):
        path = tmp_path / f"evomemory_update_frequency_hard_{split_name}.json"
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        records = load_evomemory_dataset(path)
        assert len(records) == 500
        for index, episode in enumerate(records):
            task = compile_legacy_episode(
                episode,
                source_path=path,
                source_sha256=source_hash,
                split=split,
                example_index=index,
                legacy_phase="P6.3",
            )
            assert validate_task(task).valid
            assert validate_gold_replay(task).valid
            strict_report = validate_distractors(task)
            if task.metadata.extra["compatibility_policies"]:
                assert not strict_report.valid
            else:
                assert strict_report.valid
            totals[(split_name, episode["k_updates"])] += 1
    assert totals == Counter({(split_name, k): 100 for split_name in ("dev", "test") for k in (1, 2, 4, 8, 16)})


def test_task11_package_surface_exports_compiler_and_registries() -> None:
    import mub.vnext.legacy as legacy

    assert {"compile_legacy_episode", "LEGACY_CAVEATS", "LEGACY_NAMESPACES", "legacy_namespace"} <= set(legacy.__all__)
