from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytest

import mub.vnext.generation.identity as identity_module
import mub.vnext.generation.render as render_module
from mub.vnext.contracts import (
    AnswerDisposition,
    EventRole,
    MemoryObjectKey,
    Operation,
    ReferenceResolutionStatus,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.task import GoldAction, MemUpdateTask
from mub.vnext.generation import (
    GenerationContext,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.io import semantic_task_hash
from mub.vnext.validation import (
    replay_actions,
    validate_family_a_task,
    validate_family_b_task,
    validate_family_c_task,
    validate_family_d_task,
    validate_pilot_task,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
_GROUP_FIELDS = (
    "source_group_id",
    "trajectory_id",
    "paraphrase_group_id",
    "source_document_id",
    "version_group_id",
)


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def family_c_tasks(config):
    context = GenerationContext(config=config, code_revision="family-c-validation-test")
    return [
        render_core(core, split=Split.TEST, surface_variant=variant, context=context)
        for core in generate_family_c_cores(config)
        for variant in range(3)
    ]


def _codes(report):
    return {issue.code for issue in report.issues}


def _payload(task):
    return task.model_dump(mode="json")


def _task(family_c_tasks, entity_condition, attribute_condition, variant=0):
    return next(
        task
        for task in family_c_tasks
        if task.metadata.extra["stratification"]["entity_condition"]
        == entity_condition
        and task.metadata.extra["stratification"]["attribute_condition"]
        == attribute_condition
        and task.metadata.extra["surface_variant"] == variant
    )


def _cell_tasks(family_c_tasks, entity_condition, attribute_condition, variant=0):
    return [
        task
        for task in family_c_tasks
        if task.metadata.extra["stratification"]["entity_condition"]
        == entity_condition
        and task.metadata.extra["stratification"]["attribute_condition"]
        == attribute_condition
        and task.metadata.extra["surface_variant"] == variant
    ]


def _action_for_event(payload, event):
    action_id = event["gold_action_ids"][0]
    return next(
        action
        for action in payload["gold"]["actions"]
        if action["action_id"] == action_id
    )


def _rewrite_replay(payload):
    action_by_id = {
        action["action_id"]: action for action in payload["gold"]["actions"]
    }
    payload["gold"]["action_sequence"] = [
        action_id
        for event in payload["events"]
        for action_id in event["gold_action_ids"]
    ]
    replay = replay_actions(
        [
            GoldAction.model_validate(action_by_id[action_id])
            for action_id in payload["gold"]["action_sequence"]
        ]
    ).model_dump(mode="json")
    payload["gold"]["final_state"] = replay["final_state"]
    payload["gold"]["version_history"] = replay["version_history"]


def _canonical_id(key):
    return MemoryObjectKey.model_validate(key).canonical_id


def _rewrite_raw_source_hash(payload):
    payload["source"]["raw_hash"] = render_module._payload_sha256(
        {
            "events": [
                {
                    "raw_text": event["raw_text"],
                    "speaker": event["speaker"],
                }
                for event in payload["events"]
            ],
            "query_text": payload["queries"][0]["text"],
        }
    )


def _rewrite_as_core_id_impersonation(payload, donor):
    payload["task_id"] = donor["task_id"]
    for event, action, donor_event, donor_action in zip(
        payload["events"],
        payload["gold"]["actions"],
        donor["events"],
        donor["gold"]["actions"],
    ):
        event["event_id"] = donor_event["event_id"]
        event["gold_action_ids"] = list(donor_event["gold_action_ids"])
        action["action_id"] = donor_action["action_id"]
        action["event_id"] = donor_action["event_id"]
    payload["gold"]["action_sequence"] = list(
        donor["gold"]["action_sequence"]
    )
    payload["gold"]["gold_source_event_ids"] = list(
        donor["gold"]["gold_source_event_ids"]
    )

    query = payload["queries"][0]
    donor_query = donor["queries"][0]
    old_query_id = query["query_id"]
    query["query_id"] = donor_query["query_id"]
    canonical = payload["gold"]["canonical_answers"].pop(old_query_id)
    payload["gold"]["canonical_answers"] = {query["query_id"]: canonical}

    candidate_id_map = {}
    for candidate, donor_candidate in zip(
        query["reference_candidates"], donor_query["reference_candidates"]
    ):
        candidate_id_map[candidate["candidate_id"]] = donor_candidate["candidate_id"]
        candidate["candidate_id"] = donor_candidate["candidate_id"]
    reference = query["surface_references"][0]
    reference["reference_id"] = donor_query["surface_references"][0][
        "reference_id"
    ]
    reference["candidate_ids"] = [
        candidate_id_map[candidate_id] for candidate_id in reference["candidate_ids"]
    ]
    canonical["selected_candidate_ids"] = [
        candidate_id_map[candidate_id]
        for candidate_id in canonical["selected_candidate_ids"]
    ]

    payload["metadata"]["extra"]["semantic_core_id"] = donor["metadata"][
        "extra"
    ]["semantic_core_id"]
    payload["metadata"]["extra"]["core_index"] = donor["metadata"]["extra"][
        "core_index"
    ]
    payload["metadata"]["split_key"] = deepcopy(
        donor["metadata"]["split_key"]
    )
    for field in ("semantic_core_id", *_GROUP_FIELDS):
        payload["source"]["provenance"][field] = donor["source"]["provenance"][
            field
        ]
    payload["source"]["source_id"] = donor["source"]["source_id"]
    payload["source"]["source_uri"] = donor["source"]["source_uri"]


def _render_first(config, generator, label):
    context = GenerationContext(config=config, code_revision=f"family-{label}-parity")
    return render_core(
        generator(config)[0],
        split=Split.TEST,
        surface_variant=0,
        context=context,
    )


def test_validate_family_c_accepts_all_360_surfaces_and_dispatches(family_c_tasks):
    assert len(family_c_tasks) == 360
    assert {task.metadata.extra["surface_variant"] for task in family_c_tasks} == {
        0,
        1,
        2,
    }
    for task in family_c_tasks:
        direct = validate_family_c_task(task)
        explicit = validate_pilot_task(task)
        assert direct.valid, (task.task_id, direct.issues)
        assert direct.issues == ()
        assert explicit == direct


def test_family_a_b_d_dispatch_parity_is_unchanged(config):
    for generator, validator, label in (
        (generate_family_a_cores, validate_family_a_task, "a"),
        (generate_family_b_cores, validate_family_b_task, "b"),
        (generate_family_d_cores, validate_family_d_task, "d"),
    ):
        task = _render_first(config, generator, label)
        assert validate_pilot_task(task) == validator(task)


def test_validate_family_c_is_inapplicable_to_other_families(config):
    for generator, label in (
        (generate_family_a_cores, "a"),
        (generate_family_b_cores, "b"),
        (generate_family_d_cores, "d"),
    ):
        report = validate_family_c_task(_render_first(config, generator, label))
        assert _codes(report) == {"family_c_inapplicable_task_family"}


def test_family_c_object_type_is_classification_only(family_c_tasks):
    payload = _payload(_task(family_c_tasks, "alias", "paraphrase"))
    for key in payload["target_objects"]:
        key["object_type"] = "classification_only"
    for event in payload["events"]:
        for key in _action_for_event(payload, event)["target_object_keys"]:
            key["object_type"] = "event_classification"
    for key in payload["queries"][0]["target_object_keys"]:
        key["object_type"] = "query_classification"
    for candidate in payload["queries"][0]["reference_candidates"]:
        candidate["object_key"]["object_type"] = "candidate_classification"
    for key in payload["gold"]["expected_present_objects"]:
        key["object_type"] = "gold_classification"
    changed = MemUpdateTask.model_validate(payload)
    assert validate_family_c_task(changed).valid


@pytest.mark.parametrize("mutation", ("missing", "wrong"))
def test_family_c_rejects_missing_or_wrong_alias_mapping(
    family_c_tasks, mutation
):
    payload = _payload(_task(family_c_tasks, "alias", "exact"))
    stratification = payload["metadata"]["extra"]["stratification"]
    if mutation == "missing":
        stratification.pop("entity_mapping_id")
    else:
        stratification["entity_mapping_id"] = (
            "reviewed_alias_v1:alex_at_work->manager_alex"
        )
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_reviewed_mapping_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


@pytest.mark.parametrize(
    ("entity_condition", "attribute_condition", "field", "value"),
    (
        ("distinct", "paraphrase", "attribute_mapping_id", "reviewed_attribute_paraphrase_v1:home_town->employer"),
        ("distinct", "paraphrase", "near_name_evidence", "reviewed_match:wrong->city"),
        ("distinct", "near_name", "attribute_mapping_id", "exact_attribute_v1:city"),
        ("distinct", "near_name", "near_name_evidence", "reviewed_match:city->city"),
    ),
)
def test_family_c_rejects_corrupted_attribute_mapping_and_evidence(
    family_c_tasks,
    entity_condition,
    attribute_condition,
    field,
    value,
):
    payload = _payload(_task(family_c_tasks, entity_condition, attribute_condition))
    payload["metadata"]["extra"]["stratification"][field] = value
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_reviewed_mapping_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


@pytest.mark.parametrize(
    ("entity_condition", "attribute_condition"),
    (("alias", "exact"), ("distinct", "paraphrase"), ("distinct", "near_name")),
)
def test_family_c_rejects_surface_reference_evidence_rewrite(
    family_c_tasks, entity_condition, attribute_condition
):
    payload = _payload(_task(family_c_tasks, entity_condition, attribute_condition))
    payload["queries"][0]["surface_references"][0]["evidence_kind"] = "forged"
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_reviewed_mapping_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


def test_family_c_rejects_coordinated_forged_visible_surface(family_c_tasks):
    payload = _payload(_task(family_c_tasks, "alias", "paraphrase", variant=0))
    for index, event in enumerate(payload["events"]):
        event["raw_text"] = f"Forged candidate event {index}."
        event["normalized_text"] = f"Forged normalized candidate event {index}."
        event["speaker"] = "Records clerk"
        event["metadata"]["__surface_renderer__"] = {
            "surface_template": "correction",
            "surface_variant": 2,
        }
    query = payload["queries"][0]
    query["text"] = "Forged unresolved-reference prompt with no candidates."
    query["metadata"]["__surface_renderer__"] = {
        "surface_template": "correction",
        "surface_variant": 2,
    }
    payload["metadata"]["extra"]["surface_template"] = "correction"
    payload["metadata"]["extra"]["surface_variant"] = 2
    payload["source"]["provenance"]["surface_template"] = "correction"
    payload["source"]["provenance"]["surface_variant"] = 2
    _rewrite_raw_source_hash(payload)
    forged = MemUpdateTask.model_validate(payload)

    assert "family_c_surface_integrity_mismatch" in _codes(
        validate_family_c_task(forged)
    )


def test_family_c_rejects_coordinated_cross_variant_surface_substitution(
    family_c_tasks,
):
    original = _payload(_task(family_c_tasks, "same_name", "exact", variant=0))
    substitute = _payload(_task(family_c_tasks, "same_name", "exact", variant=1))
    for original_event, substitute_event in zip(
        original["events"], substitute["events"]
    ):
        original_event["raw_text"] = substitute_event["raw_text"]
        original_event["normalized_text"] = substitute_event["normalized_text"]
        original_event["speaker"] = substitute_event["speaker"]
        original_event["metadata"]["__surface_renderer__"] = deepcopy(
            substitute_event["metadata"]["__surface_renderer__"]
        )
    original["queries"][0]["text"] = substitute["queries"][0]["text"]
    original["queries"][0]["metadata"]["__surface_renderer__"] = deepcopy(
        substitute["queries"][0]["metadata"]["__surface_renderer__"]
    )
    original["metadata"]["extra"]["surface_template"] = "conversational"
    original["metadata"]["extra"]["surface_variant"] = 1
    original["source"]["provenance"]["surface_template"] = "conversational"
    original["source"]["provenance"]["surface_variant"] = 1
    original["source"]["raw_hash"] = substitute["source"]["raw_hash"]
    substituted = MemUpdateTask.model_validate(original)

    assert "family_c_surface_integrity_mismatch" in _codes(
        validate_family_c_task(substituted)
    )


@pytest.mark.parametrize("replacement", ("arbitrary", "other_core"))
def test_family_c_rejects_noncanonical_surface_reference_id(
    family_c_tasks,
    replacement,
):
    tasks = _cell_tasks(family_c_tasks, "alias", "exact", variant=0)
    payload = _payload(tasks[0])
    payload["queries"][0]["surface_references"][0]["reference_id"] = (
        "reference_forged"
        if replacement == "arbitrary"
        else tasks[1].queries[0].surface_references[0].reference_id
    )
    corrupted = MemUpdateTask.model_validate(payload)

    assert "family_c_surface_integrity_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


@pytest.mark.parametrize("field", _GROUP_FIELDS)
def test_family_c_rejects_independent_provenance_group_rewrite(
    family_c_tasks,
    field,
):
    payload = _payload(_task(family_c_tasks, "distinct", "exact", variant=0))
    payload["source"]["provenance"][field] = f"{field}_forged"
    corrupted = MemUpdateTask.model_validate(payload)

    assert "family_c_provenance_link_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


def test_family_c_rejects_coordinated_all_group_rewrite(family_c_tasks):
    payload = _payload(_task(family_c_tasks, "alias", "paraphrase", variant=1))
    for field in _GROUP_FIELDS:
        forged = f"{field}_coordinated_forgery"
        payload["source"]["provenance"][field] = forged
        payload["metadata"]["split_key"][field] = forged
    corrupted = MemUpdateTask.model_validate(payload)

    assert "family_c_provenance_link_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


def test_family_c_rejects_cross_core_group_substitution(family_c_tasks):
    tasks = _cell_tasks(family_c_tasks, "same_name", "near_name", variant=2)
    payload = _payload(tasks[0])
    other = _payload(tasks[1])
    for field in _GROUP_FIELDS:
        payload["source"]["provenance"][field] = other["source"]["provenance"][
            field
        ]
        payload["metadata"]["split_key"][field] = other["metadata"]["split_key"][
            field
        ]
    corrupted = MemUpdateTask.model_validate(payload)

    assert "family_c_provenance_link_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


def test_family_c_rejects_coordinated_cross_core_identity_impersonation(
    family_c_tasks,
):
    tasks = _cell_tasks(family_c_tasks, "distinct", "exact", variant=0)
    payload = _payload(tasks[0])
    donor = _payload(tasks[1])
    _rewrite_as_core_id_impersonation(payload, donor)
    impersonated = MemUpdateTask.model_validate(payload)

    assert "family_c_semantic_core_id_mismatch" in _codes(
        validate_family_c_task(impersonated)
    )


def test_family_c_admin_core_index_can_change_without_changing_semantic_core(
    family_c_tasks,
):
    task = _task(family_c_tasks, "alias", "paraphrase", variant=1)
    payload = _payload(task)
    semantic_core_id = payload["metadata"]["extra"]["semantic_core_id"]
    old_groups = {
        field: payload["metadata"]["split_key"][field]
        for field in (
            "source_group_id",
            "paraphrase_group_id",
            "source_document_id",
        )
    }
    core_index = 999
    surface_variant = payload["metadata"]["extra"]["surface_variant"]
    payload["metadata"]["extra"]["core_index"] = core_index
    payload["source"]["source_id"] = identity_module.source_id(
        "vnext_pilot",
        core_index,
        {
            "semantic_core_id": semantic_core_id,
            "surface_variant": surface_variant,
        },
    )
    payload["source"]["source_uri"] = (
        f"memory://{payload['source']['source_id']}"
    )
    payload["queries"][0]["surface_references"][0]["reference_id"] = (
        identity_module.stable_id(
            "reference",
            {
                "family": TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value,
                "core_index": core_index,
            },
        )
    )
    trajectory = identity_module.trajectory_id(
        semantic_core_id,
        f"family_c_{core_index:03d}",
    )
    version_group = identity_module.stable_id(
        "version_group",
        {"trajectory_id": trajectory},
    )
    for container in (
        payload["source"]["provenance"],
        payload["metadata"]["split_key"],
    ):
        container["trajectory_id"] = trajectory
        container["version_group_id"] = version_group
    rewritten = MemUpdateTask.model_validate(payload)

    report = validate_family_c_task(rewritten)
    assert report.valid, report.issues
    assert semantic_task_hash(rewritten) == semantic_task_hash(task)
    assert {
        field: payload["metadata"]["split_key"][field]
        for field in old_groups
    } == old_groups


def test_family_c_allows_evaluation_only_split_exception(family_c_tasks):
    payload = _payload(
        _task(family_c_tasks, "namespace_collision", "near_name", variant=2)
    )
    payload["metadata"]["split_key"]["split_exception_id"] = (
        "family_c_eval_robustness_exception_v1"
    )
    payload["metadata"]["split_key"]["split_policy_version"] = (
        "evaluation_only_robustness_v1"
    )
    exceptional = MemUpdateTask.model_validate(payload)

    report = validate_family_c_task(exceptional)
    assert report.valid, report.issues


def test_family_c_rejects_coordinated_candidate_geometry_collapse(family_c_tasks):
    task = _task(family_c_tasks, "distinct", "exact")
    query = task.queries[0]
    collapsed = query.reference_candidates[1].model_copy(
        update={"object_key": query.reference_candidates[0].object_key}
    )
    malformed_query = query.model_copy(
        update={"reference_candidates": [query.reference_candidates[0], collapsed]}
    )
    malformed = MemUpdateTask.model_construct(
        **{**task.__dict__, "queries": [malformed_query]}
    )
    assert "family_c_candidate_geometry_mismatch" in _codes(
        validate_family_c_task(malformed)
    )


@pytest.mark.parametrize(
    ("entity_condition", "attribute_condition", "status", "disposition", "linked"),
    (
        ("alias", "exact", "ambiguous", "abstained", "both"),
        ("same_name", "exact", "unique", "answered", "first"),
        ("distinct", "near_name", "unique", "answered", "first"),
    ),
)
def test_family_c_rejects_coordinated_truth_table_rewrites(
    family_c_tasks,
    entity_condition,
    attribute_condition,
    status,
    disposition,
    linked,
):
    payload = _payload(_task(family_c_tasks, entity_condition, attribute_condition))
    query = payload["queries"][0]
    query_id = query["query_id"]
    candidate_ids = [candidate["candidate_id"] for candidate in query["reference_candidates"]]
    reference = query["surface_references"][0]
    canonical = payload["gold"]["canonical_answers"][query_id]
    canonical["resolution_status"] = status
    canonical["disposition"] = disposition
    if disposition == "answered":
        reference["candidate_ids"] = candidate_ids[:1]
        canonical["selected_candidate_ids"] = candidate_ids[:1]
        canonical["value"] = _action_for_event(payload, payload["events"][0])["value"]
        canonical["abstention_reason"] = None
    else:
        reference["candidate_ids"] = candidate_ids if linked == "both" else []
        canonical["selected_candidate_ids"] = []
        canonical["value"] = None
        canonical["abstention_reason"] = "coordinated forged abstention"
    stratification = payload["metadata"]["extra"]["stratification"]
    stratification["resolution_status"] = status
    stratification["answer_disposition"] = disposition
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_resolution_truth_table_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


def test_family_c_rejects_no_match_bare_none_and_absent_impersonation(
    family_c_tasks,
):
    task = _task(family_c_tasks, "distinct", "near_name")
    query_id = task.queries[0].query_id
    bare_gold = type(task.gold).model_construct(
        **{
            **task.gold.__dict__,
            "canonical_answers": {query_id: None},
            "expected_present_objects": [],
            "expected_absent_objects": list(task.target_objects),
        }
    )
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "gold": bare_gold})
    report = validate_family_c_task(malformed)
    codes = _codes(report)
    assert "family_c_invalid_field_type" in codes
    assert len(report.issues) <= 128


@pytest.mark.parametrize(
    "mutation",
    ("candidate_index", "action_target", "final_state", "history", "action_order"),
)
def test_family_c_rejects_candidate_event_action_state_history_cross_link_rewrites(
    family_c_tasks, mutation
):
    payload = _payload(_task(family_c_tasks, "alias", "paraphrase"))
    if mutation == "candidate_index":
        payload["events"][0]["metadata"]["candidate_index"] = 1
    elif mutation == "action_target":
        first_action = _action_for_event(payload, payload["events"][0])
        second_action = _action_for_event(payload, payload["events"][1])
        first_action["target_object_keys"] = deepcopy(second_action["target_object_keys"])
    elif mutation == "final_state":
        object_id = _canonical_id(payload["target_objects"][0])
        payload["gold"]["final_state"][object_id] = "forged"
    elif mutation == "history":
        object_id = _canonical_id(payload["target_objects"][0])
        payload["gold"]["version_history"][object_id] = ["forged"]
    else:
        payload["gold"]["action_sequence"].reverse()
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_linkage_mismatch" in _codes(validate_family_c_task(corrupted))


@pytest.mark.parametrize("duplicate", ("candidate_id", "event_id", "identity"))
def test_family_c_rejects_duplicate_ids_and_identities(family_c_tasks, duplicate):
    task = _task(family_c_tasks, "same_name", "exact")
    if duplicate == "candidate_id":
        query = task.queries[0]
        duplicate_candidate = query.reference_candidates[1].model_copy(
            update={"candidate_id": query.reference_candidates[0].candidate_id}
        )
        malformed_query = query.model_copy(
            update={"reference_candidates": [query.reference_candidates[0], duplicate_candidate]}
        )
        malformed = MemUpdateTask.model_construct(
            **{**task.__dict__, "queries": [malformed_query]}
        )
    elif duplicate == "event_id":
        duplicate_event = task.events[1].model_copy(update={"event_id": task.events[0].event_id})
        malformed = MemUpdateTask.model_construct(
            **{**task.__dict__, "events": [task.events[0], duplicate_event]}
        )
    else:
        query = task.queries[0]
        duplicate_candidate = query.reference_candidates[1].model_copy(
            update={"object_key": query.reference_candidates[0].object_key}
        )
        malformed_query = query.model_copy(
            update={"reference_candidates": [query.reference_candidates[0], duplicate_candidate]}
        )
        malformed = MemUpdateTask.model_construct(
            **{**task.__dict__, "queries": [malformed_query]}
        )
    report = validate_family_c_task(malformed)
    assert not report.valid
    assert len(report.issues) <= 128


@pytest.mark.parametrize(
    ("location", "field", "value"),
    (
        ("profile", "update_depth", 2),
        ("profile", "active_object_count", 1),
        ("profile", "context_order", "reverse_chronological"),
        ("profile", "query_type", "current_state"),
        ("profile", "source_naturalness", "synthetic_direct"),
        ("profile", "entity_ambiguity", "high"),
        ("stratification", "candidate_count", 1),
        ("stratification", "num_events", 3),
        ("stratification", "num_target_updates", 1),
        ("stratification", "noop_count", 1),
        ("stratification", "difficulty", "hard"),
        ("stratification", "answer_disposition", "abstained"),
    ),
)
def test_family_c_rejects_profile_count_and_evidence_rewrites(
    family_c_tasks, location, field, value
):
    payload = _payload(_task(family_c_tasks, "alias", "paraphrase"))
    target = (
        payload["metadata"]["resolved_profile"]
        if location == "profile"
        else payload["metadata"]["extra"]["stratification"]
    )
    target[field] = value
    corrupted = MemUpdateTask.model_validate(payload)
    assert "family_c_counter_profile_mismatch" in _codes(
        validate_family_c_task(corrupted)
    )


@pytest.mark.parametrize("change", ("extra", "missing", "noop"))
def test_family_c_rejects_extra_missing_and_noop_events(family_c_tasks, change):
    payload = _payload(_task(family_c_tasks, "distinct", "exact"))
    if change == "extra":
        event = deepcopy(payload["events"][-1])
        action = deepcopy(_action_for_event(payload, payload["events"][-1]))
        event["event_id"] += "_extra"
        event["sequence_index"] = 2
        event["gold_action_ids"] = [action["action_id"] + "_extra"]
        action["action_id"] += "_extra"
        action["event_id"] = event["event_id"]
        action["operation"] = Operation.UPDATE.value
        payload["events"].append(event)
        payload["gold"]["actions"].append(action)
    elif change == "missing":
        event = payload["events"].pop()
        removed = event["gold_action_ids"][0]
        payload["gold"]["actions"] = [
            action
            for action in payload["gold"]["actions"]
            if action["action_id"] != removed
        ]
    else:
        event = payload["events"][1]
        action = _action_for_event(payload, event)
        event["role"] = EventRole.NOOP_NEAR_MISS.value
        action["operation"] = Operation.NOOP.value
        action["target_object_keys"] = []
        action["value"] = None
    _rewrite_replay(payload)
    payload["gold"]["gold_source_event_ids"] = [
        event["event_id"] for event in payload["events"]
    ]
    corrupted = MemUpdateTask.model_validate(payload)
    codes = _codes(validate_family_c_task(corrupted))
    assert "family_c_event_count_mismatch" in codes or "family_c_write_semantics_mismatch" in codes


def test_family_c_rejects_multiple_canonical_answer_support(family_c_tasks):
    task = _task(family_c_tasks, "distinct", "exact")
    query_id = task.queries[0].query_id
    forged_gold = type(task.gold).model_construct(
        **{
            **task.gold.__dict__,
            "canonical_answers": {
                query_id: task.gold.canonical_answers[query_id],
                "query_extra": task.gold.canonical_answers[query_id],
            },
        }
    )
    malformed = MemUpdateTask.model_construct(**{**task.__dict__, "gold": forged_gold})
    assert "family_c_answer_support_mismatch" in _codes(
        validate_family_c_task(malformed)
    )


class HostileFamilyString(str):
    __hash__ = str.__hash__

    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.override_access_count = 0
        return instance

    def _fail(self):
        self.override_access_count += 1
        raise RuntimeError("hostile family override executed")

    def __eq__(self, other):
        return self._fail()

    def __str__(self):
        return self._fail()

    def strip(self, *args, **kwargs):
        return self._fail()


class ExplosiveSequence(Sequence):
    def __init__(self):
        self.iteration_count = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError(f"explosive item {index}")

    def __iter__(self):
        self.iteration_count += 1
        raise RuntimeError("explosive iteration")


def test_family_c_malformed_hostile_cycle_and_size_are_bounded_deterministic(
    family_c_tasks,
):
    task = _task(family_c_tasks, "distinct", "exact")
    hostile = HostileFamilyString(TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value)
    hostile_task = MemUpdateTask.model_construct(
        **{**task.__dict__, "task_family": hostile}
    )
    explosive = ExplosiveSequence()
    explosive_task = MemUpdateTask.model_construct(
        **{**task.__dict__, "events": explosive}
    )
    cycle = {}
    cycle["self"] = cycle
    cycle_metadata = task.metadata.model_copy(update={"extra": cycle})
    cycle_task = MemUpdateTask.model_construct(
        **{**task.__dict__, "metadata": cycle_metadata}
    )
    oversized_metadata = task.metadata.model_copy(
        update={"extra": {"items": list(range(65))}}
    )
    oversized_task = MemUpdateTask.model_construct(
        **{**task.__dict__, "metadata": oversized_metadata}
    )

    for malformed in (hostile_task, explosive_task, cycle_task, oversized_task):
        first = validate_family_c_task(malformed)
        second = validate_family_c_task(malformed)
        explicit = validate_pilot_task(malformed)
        assert first == second == explicit
        assert not first.valid
        assert len(first.issues) <= 128
    assert hostile.override_access_count == 0
    assert explosive.iteration_count == 0


def test_validate_family_c_rejects_non_task_and_malformed_family():
    assert _codes(validate_family_c_task(object())) == {
        "family_c_invalid_task_type"
    }
    malformed = MemUpdateTask.model_construct(
        task_family=TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value
    )
    direct = validate_family_c_task(malformed)
    explicit = validate_pilot_task(malformed)
    assert direct == explicit
    assert "family_c_malformed_record" in _codes(direct)
