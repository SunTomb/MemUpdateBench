from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import mub.vnext.validation.pilot as pilot_module
from mub.vnext.contracts import (
    CanonicalAnswer,
    EventRole,
    Operation,
    Split,
    TaskFamily,
)
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.generation import (
    GenerationContext,
    generate_family_a_cores,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.validation import (
    merge_reports,
    validate_distractors,
    validate_gold_replay,
    validate_task,
    validate_task_semantics,
)
from mub.vnext.validation.pilot import validate_family_d_task, validate_family_a_task


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
MAX_REPORT_ISSUES = 128


class ExplosiveSequence(Sequence):
    def __init__(self):
        self.iteration_count = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError(f"unstable-item-{self.iteration_count}-{index}")

    def __iter__(self):
        self.iteration_count += 1
        raise RuntimeError(f"unstable-iteration-{self.iteration_count}")


class ExplosiveMapping(Mapping):
    def __init__(self):
        self.iteration_count = 0

    def __len__(self):
        return 1

    def __getitem__(self, key):
        raise RuntimeError(f"unstable-item-{self.iteration_count}-{key}")

    def __iter__(self):
        self.iteration_count += 1
        raise RuntimeError(f"unstable-iteration-{self.iteration_count}")


class ExplosiveEnumLike:
    def __init__(self):
        self.value_access_count = 0

    @property
    def value(self):
        self.value_access_count += 1
        raise RuntimeError(f"unstable-enum-{self.value_access_count}")


class HostileFamilyString(str):
    __hash__ = str.__hash__

    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.override_access_count = 0
        return instance

    def _fail(self):
        self.override_access_count += 1
        raise RuntimeError(f"hostile-family-{self.override_access_count}")

    def __eq__(self, other):
        return self._fail()

    def __ne__(self, other):
        return self._fail()

    def __str__(self):
        return self._fail()

    def strip(self, *args, **kwargs):
        return self._fail()


@pytest.fixture(scope="module")
def family_d_tasks():
    config = load_pilot_config(CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="family-d-validation-test")
    tasks = {}
    for core in generate_family_d_cores(config):
        trap_type = core.stratification["trap_type"]
        task = None
        if trap_type not in tasks:
            task = render_core(
                core,
                split=Split.TEST,
                surface_variant=0,
                context=context,
            )
            tasks[trap_type] = task
        density_key = f"density_{core.stratification['configured_noop_density']}"
        if density_key not in tasks:
            task = task or render_core(
                core,
                split=Split.TEST,
                surface_variant=0,
                context=context,
            )
            tasks[density_key] = task
        if trap_type == "duplicate_current" and "duplicate_current_variant_0" not in tasks:
            for variant in range(3):
                tasks[f"duplicate_current_variant_{variant}"] = (
                    task
                    if variant == 0 and task is not None
                    else render_core(
                        core,
                        split=Split.TEST,
                        surface_variant=variant,
                        context=context,
                    )
                )
        if (
            trap_type == "semantic_near_miss"
            and "semantic_near_miss_with_prior_write" not in tasks
        ):
            task = task or render_core(
                core,
                split=Split.TEST,
                surface_variant=0,
                context=context,
            )
            trap_index = next(
                index
                for index, event in enumerate(task.events)
                if event.metadata.get("trap_type") == trap_type
            )
            if any(
                event.metadata.get("lifecycle") == "independent_current"
                for event in task.events[:trap_index]
            ):
                tasks["semantic_near_miss_with_prior_write"] = task
    return tasks


@pytest.fixture(scope="module")
def family_a_tasks():
    config = load_pilot_config(CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="family-a-validation-test")
    return [
        render_core(core, split=Split.TEST, surface_variant=variant, context=context)
        for core in generate_family_a_cores(config)
        for variant in range(3)
    ]


def _codes(report):
    return {issue.code for issue in report.issues}


def _family_a_task(family_a_tasks, *, difficulty, depth, surface_variant=0):
    return next(
        task
        for task in family_a_tasks
        if task.difficulty.value == difficulty
        and task.metadata.resolved_profile["update_depth"] == depth
        and task.metadata.extra["surface_variant"] == surface_variant
    )


def _payload(task):
    return task.model_dump(mode="json")


def _construct_replace(model, **changes):
    data = dict(model.__dict__)
    data.update(changes)
    return type(model).model_construct(**data)


def _replace_gold(task, **changes):
    gold = _construct_replace(task.gold, **changes)
    return _construct_replace(task, gold=gold)


def _replace_action(task, action_index, **changes):
    actions = list(task.gold.actions)
    actions[action_index] = _construct_replace(actions[action_index], **changes)
    return _replace_gold(task, actions=actions)


def _event_with_lifecycle(payload, lifecycle):
    return next(
        event
        for event in payload["events"]
        if event["metadata"].get("lifecycle") == lifecycle
    )


def _action_for_event(payload, event):
    action_id = event["gold_action_ids"][0]
    return next(
        action for action in payload["gold"]["actions"] if action["action_id"] == action_id
    )


def _rewrite_semantic_noop_as_non_target_update(task, *, erase_trap_signals=False):
    payload = _payload(task)
    trap_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["metadata"].get("trap_type") == "semantic_near_miss"
    )
    trap_event = payload["events"][trap_index]
    prior_index = next(
        index
        for index in range(trap_index - 1, -1, -1)
        if payload["events"][index]["metadata"].get("lifecycle")
        == "independent_current"
    )
    prior_event = payload["events"][prior_index]
    prior_action = _action_for_event(payload, prior_event)
    prior_typed_action = next(
        action
        for action in task.gold.actions
        if action.action_id == prior_event["gold_action_ids"][0]
    )
    non_target_id = prior_typed_action.target_object_keys[0].canonical_id

    trap_event["metadata"]["lifecycle"] = "independent_current"
    if erase_trap_signals:
        trap_event["metadata"].pop("trap_type")
        trap_event["metadata"]["surface_statement"] = "This now directs a write."
        trap_event["role"] = EventRole.NEUTRAL.value
        trap_event["raw_text"] = "This now directs a write."
    trap_action = _action_for_event(payload, trap_event)
    trap_action["operation"] = Operation.UPDATE.value
    trap_action["scope"] = "attribute"
    trap_action["target_object_keys"] = prior_action["target_object_keys"]
    trap_action["value"] = "coordinated-adversary-value"

    payload["gold"]["final_state"][non_target_id] = trap_action["value"]
    payload["gold"]["version_history"][non_target_id].append(trap_action["value"])
    stratification = payload["metadata"]["extra"]["stratification"]
    stratification["noop_count"] -= 1
    stratification["true_write_count"] += 1
    density = stratification["noop_count"] / stratification["num_events"]
    stratification["configured_noop_density"] = density
    stratification["observed_noop_density"] = density
    payload["metadata"]["resolved_profile"]["noop_density"] = density
    actions_by_id = {
        action["action_id"]: action for action in payload["gold"]["actions"]
    }
    stratification["operation_signature"] = ",".join(
        actions_by_id[event["gold_action_ids"][0]]["operation"]
        for event in payload["events"]
    )
    return MemUpdateTask.model_validate(payload)


def _append_extra_noop(task):
    payload = _payload(task)
    template_event = next(
        event
        for event in payload["events"]
        if event["metadata"].get("lifecycle") == "independent_noop"
    )
    template_action = _action_for_event(payload, template_event)
    extra_event = deepcopy(template_event)
    extra_action = deepcopy(template_action)
    extra_event_id = f"{template_event['event_id']}-extra"
    extra_action_id = f"{template_action['action_id']}-extra"
    extra_event["event_id"] = extra_event_id
    extra_event["sequence_index"] = len(payload["events"])
    extra_event["gold_action_ids"] = [extra_action_id]
    extra_event["source_anchor"] = {"event_index": len(payload["events"])}
    extra_action["action_id"] = extra_action_id
    extra_action["event_id"] = extra_event_id
    payload["events"].append(extra_event)
    payload["gold"]["actions"].append(extra_action)
    payload["gold"]["action_sequence"].append(extra_action_id)

    stratification = payload["metadata"]["extra"]["stratification"]
    stratification["num_events"] += 1
    stratification["noop_count"] += 1
    density = stratification["noop_count"] / stratification["num_events"]
    stratification["configured_noop_density"] = density
    stratification["observed_noop_density"] = density
    stratification["operation_signature"] += ",NOOP"
    payload["metadata"]["resolved_profile"]["noop_density"] = density
    payload["metadata"]["resolved_profile"]["context_length"] += 1
    return MemUpdateTask.model_validate(payload)


@pytest.mark.parametrize("task_index", (0, 1, 2))
def test_validate_family_a_task_accepts_generated_samples_and_aggregate_dispatches(
    family_a_tasks, task_index
):
    task = family_a_tasks[task_index]
    direct = validate_family_a_task(task)
    aggregate = validate_task_semantics(task)
    assert direct.valid
    assert direct.issues == ()
    assert aggregate == direct


def test_validate_family_a_task_accepts_all_generated_tasks(family_a_tasks):
    for task in family_a_tasks:
        direct = validate_family_a_task(task)
        aggregate = validate_task_semantics(task)
        assert direct.valid, (task.task_id, direct.issues)
        assert aggregate == direct


def test_validate_family_a_task_detects_coordinated_stale_gold_collision(family_a_tasks):
    task = _family_a_task(family_a_tasks, difficulty="easy", depth=4)
    payload = _payload(task)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    query_id = task.queries[0].query_id
    final_value = payload["gold"]["gold_answers"][query_id]
    stale_event = payload["events"][0]
    stale_action = _action_for_event(payload, stale_event)
    stale_action["value"] = final_value
    payload["gold"]["version_history"][target_id][0] = final_value
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_stale_value_equals_current_gold" in _codes(report)


@pytest.mark.parametrize(
    ("first_value", "second_value", "expected_duplicate"),
    (
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        (
            {"items": [{"a": 1, "b": 2}]},
            {"items": [{"b": 2, "a": 1}]},
            True,
        ),
        (
            [{"a": 1, "b": 2}, [1, 2]],
            [{"b": 2, "a": 1}, [1, 2]],
            True,
        ),
        (True, 1, False),
        (1, 1.0, False),
        ([1, 2], [2, 1], False),
    ),
)
def test_validate_family_a_task_compares_stale_values_as_canonical_json(
    family_a_tasks,
    first_value,
    second_value,
    expected_duplicate,
):
    task = _family_a_task(family_a_tasks, difficulty="easy", depth=4)
    payload = _payload(task)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    for index, value in enumerate((first_value, second_value)):
        action = _action_for_event(payload, payload["events"][index])
        action["value"] = value
        payload["gold"]["version_history"][target_id][index] = value
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert (
        "family_a_duplicate_stale_value" in _codes(report)
    ) is expected_duplicate


def test_validate_family_a_task_detects_coordinated_distractor_gold_collision(
    family_a_tasks,
):
    task = _family_a_task(family_a_tasks, difficulty="medium", depth=1)
    payload = _payload(task)
    query_id = task.queries[0].query_id
    final_value = payload["gold"]["gold_answers"][query_id]
    event = next(
        event
        for event in payload["events"]
        if event["role"] == EventRole.SAME_NAME_OTHER_ENTITY.value
    )
    action = _action_for_event(payload, event)
    typed_action = next(
        item for item in task.gold.actions if item.action_id == action["action_id"]
    )
    distractor_id = typed_action.target_object_keys[0].canonical_id
    action["value"] = final_value
    payload["gold"]["final_state"][distractor_id] = final_value
    payload["gold"]["version_history"][distractor_id] = [final_value]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_distractor_current_gold_collision" in _codes(report)


@pytest.mark.parametrize("corruption", ("operation", "role", "key", "count"))
def test_validate_family_a_task_rejects_corrupted_target_chain(
    family_a_tasks,
    corruption,
):
    task = _family_a_task(family_a_tasks, difficulty="medium", depth=4)
    payload = _payload(task)
    event = payload["events"][1]
    action = _action_for_event(payload, event)
    if corruption == "operation":
        action["operation"] = Operation.ADD.value
    elif corruption == "role":
        event["role"] = EventRole.LATEST_GOLD.value
    elif corruption == "key":
        distractor = next(
            candidate
            for candidate in payload["gold"]["actions"]
            if candidate["target_object_keys"]
            and candidate["target_object_keys"] != action["target_object_keys"]
        )
        action["target_object_keys"] = distractor["target_object_keys"]
    else:
        payload["metadata"]["resolved_profile"]["update_depth"] = 1
        payload["metadata"]["resolved_profile"]["update_depth_bucket"] = "1"
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_target_chain_corruption" in _codes(report)


def test_validate_family_a_task_rejects_coordinated_near_miss_write(family_a_tasks):
    task = _family_a_task(family_a_tasks, difficulty="medium", depth=1)
    payload = _payload(task)
    noop_event = next(
        event for event in payload["events"] if event["role"] == EventRole.NOOP_NEAR_MISS.value
    )
    noop_action = _action_for_event(payload, noop_event)
    distractor_action = next(
        action
        for action in payload["gold"]["actions"]
        if action["operation"] == Operation.ADD.value
        and action["target_object_keys"]
        != payload["queries"][0]["target_object_keys"]
    )
    distractor_key = distractor_action["target_object_keys"]
    typed_key = next(
        action.target_object_keys[0]
        for action in task.gold.actions
        if action.action_id == distractor_action["action_id"]
    )
    value = "coordinated-near-miss-write"
    noop_action["operation"] = Operation.UPDATE.value
    noop_action["scope"] = "attribute"
    noop_action["target_object_keys"] = distractor_key
    noop_action["value"] = value
    payload["gold"]["final_state"][typed_key.canonical_id] = value
    payload["gold"]["version_history"][typed_key.canonical_id].append(value)
    stratification = payload["metadata"]["extra"]["stratification"]
    stratification["noop_count"] -= 1
    density = stratification["noop_count"] / stratification["num_events"]
    payload["metadata"]["resolved_profile"]["noop_density"] = density
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_noop_semantics_mismatch" in _codes(report)


def test_validate_family_a_task_rejects_distractor_geometry_collision(family_a_tasks):
    task = _family_a_task(family_a_tasks, difficulty="medium", depth=1)
    payload = _payload(task)
    event = next(
        event
        for event in payload["events"]
        if event["role"] == EventRole.SAME_ENTITY_OTHER_ATTRIBUTE.value
    )
    action = _action_for_event(payload, event)
    action["target_object_keys"] = payload["queries"][0]["target_object_keys"]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_distractor_geometry_corruption" in _codes(report)


@pytest.mark.parametrize("field", ("final_state", "version_history"))
def test_validate_family_a_task_rejects_corrupted_gold_state_history(
    family_a_tasks,
    field,
):
    task = _family_a_task(family_a_tasks, difficulty="hard", depth=16)
    payload = _payload(task)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    if field == "final_state":
        payload["gold"][field][target_id] = "corrupted-final-state"
    else:
        payload["gold"][field][target_id][0] = "corrupted-history"
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_gold_state_history_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("container", "field", "value"),
    (
        ("stratification", "num_events", 99),
        ("stratification", "stale_count", 99),
        ("profile", "active_object_count", 99),
        ("profile", "noop_density", 0.123),
        ("profile", "version_metadata", "rewritten"),
    ),
)
def test_validate_family_a_task_rejects_counter_and_profile_rewrites(
    family_a_tasks,
    container,
    field,
    value,
):
    payload = _payload(_family_a_task(family_a_tasks, difficulty="hard", depth=4))
    target = (
        payload["metadata"]["extra"]["stratification"]
        if container == "stratification"
        else payload["metadata"]["resolved_profile"]
    )
    target[field] = value
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_counter_profile_mismatch" in _codes(report)


def test_validate_family_a_task_forbids_duplicate_current_injection(family_a_tasks):
    payload = _payload(_family_a_task(family_a_tasks, difficulty="medium", depth=1))
    payload["events"][-1]["role"] = EventRole.DUPLICATE_CURRENT.value
    payload["events"][-1]["metadata"]["allow_accepted_answer_ambiguity"] = True
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_duplicate_current_forbidden" in _codes(report)


def test_validate_family_a_task_rejects_multiple_acceptable_current_answers(
    family_a_tasks,
):
    task = _family_a_task(family_a_tasks, difficulty="easy", depth=1)
    payload = _payload(task)
    query_id = task.queries[0].query_id
    payload["gold"]["acceptable_answers"][query_id] = [
        payload["gold"]["gold_answers"][query_id],
        "alternate-current-answer",
    ]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_a_task(corrupted)

    assert "family_a_multiple_current_answers" in _codes(report)


@pytest.mark.parametrize("release_mutation", ("missing", "changed"))
def test_validate_family_a_task_cannot_downgrade_semantics_via_release_provenance(
    family_a_tasks,
    release_mutation,
):
    task = _family_a_task(family_a_tasks, difficulty="easy", depth=4)
    payload = _payload(task)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    query_id = task.queries[0].query_id
    final_value = payload["gold"]["gold_answers"][query_id]
    stale_action = _action_for_event(payload, payload["events"][0])
    stale_action["value"] = final_value
    payload["gold"]["version_history"][target_id][0] = final_value
    corrupted = MemUpdateTask.model_validate(payload)
    provenance = dict(corrupted.source.provenance)
    if release_mutation == "missing":
        provenance.pop("release_id")
    else:
        provenance["release_id"] = "attacker-controlled-release"
    source = _construct_replace(corrupted.source, provenance=provenance)
    corrupted = _construct_replace(corrupted, source=source)

    direct = validate_family_a_task(corrupted)
    aggregate = validate_task_semantics(corrupted)

    assert aggregate == direct
    assert "family_a_release_provenance_mismatch" in _codes(direct)
    assert "family_a_stale_value_equals_current_gold" in _codes(direct)


def test_validate_family_d_task_rejects_multiple_acceptable_current_answers(
    family_d_tasks,
):
    task = family_d_tasks["semantic_near_miss"]
    payload = _payload(task)
    query_id = task.queries[0].query_id
    payload["gold"]["acceptable_answers"][query_id] = [
        payload["gold"]["gold_answers"][query_id],
        "alternate-current-answer",
    ]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_multiple_current_answers" in _codes(report)


@pytest.mark.parametrize("family", ("a", "d"))
def test_pilot_current_answer_accepts_single_candidate_list_encoding(
    family_a_tasks,
    family_d_tasks,
    family,
):
    task = (
        _family_a_task(family_a_tasks, difficulty="easy", depth=1)
        if family == "a"
        else family_d_tasks["semantic_near_miss"]
    )
    payload = _payload(task)
    query_id = task.queries[0].query_id
    payload["gold"]["acceptable_answers"][query_id] = [
        payload["gold"]["gold_answers"][query_id]
    ]
    encoded = MemUpdateTask.model_validate(payload)

    report = (
        validate_family_a_task(encoded)
        if family == "a"
        else validate_family_d_task(encoded)
    )

    assert report.valid, report.issues


def test_validate_family_a_task_malformed_construct_matches_aggregate():
    malformed = MemUpdateTask.model_construct(
        task_family=TaskFamily.REPEATED_SAME_SLOT.value
    )

    direct = validate_family_a_task(malformed)
    aggregate = validate_task_semantics(malformed)

    assert direct == aggregate
    assert not direct.valid
    assert len(direct.issues) <= MAX_REPORT_ISSUES
    assert "family_a_malformed_record" in _codes(direct)


@pytest.mark.parametrize("family", ("a", "d"))
@pytest.mark.parametrize("corruption", ("normalized_hash", "synthetic_generator"))
def test_pilot_schema_preflight_revalidates_contract_constraints_and_invariants(
    family_a_tasks,
    family_d_tasks,
    family,
    corruption,
):
    task = (
        _family_a_task(family_a_tasks, difficulty="easy", depth=1)
        if family == "a"
        else family_d_tasks["semantic_near_miss"]
    )
    source = _construct_replace(
        task.source,
        **(
            {"normalized_hash": "not-a-sha256"}
            if corruption == "normalized_hash"
            else {"generator": None}
        ),
    )
    malformed = _construct_replace(task, source=source)

    validator = (
        validate_family_a_task if family == "a" else validate_family_d_task
    )
    first = validator(malformed)
    second = validator(malformed)

    assert first == second
    assert len(first.issues) <= MAX_REPORT_ISSUES
    assert f"family_{family}_contract_constraint_violation" in _codes(first)


def test_validate_family_a_task_merges_generic_and_family_findings(
    family_a_tasks,
):
    task = _family_a_task(family_a_tasks, difficulty="easy", depth=1)
    queries = list(task.queries)
    queries[0] = _construct_replace(
        queries[0],
        target_object_keys=[
            queries[0].target_object_keys[0],
            queries[0].target_object_keys[0],
        ],
    )
    malformed = _construct_replace(task, task_id="", queries=queries)

    direct = validate_family_a_task(malformed)
    aggregate = validate_task_semantics(malformed)

    assert aggregate == direct
    assert {
        "blank_task_id",
        "duplicate_query_target",
        "family_a_input_size_limit",
    } <= _codes(direct)


def test_validate_family_a_task_routes_hostile_family_without_override_access(
    family_a_tasks,
):
    hostile = HostileFamilyString(TaskFamily.REPEATED_SAME_SLOT.value)
    malformed = _construct_replace(family_a_tasks[0], task_family=hostile)

    direct = validate_family_a_task(malformed)
    aggregate = validate_task_semantics(malformed)

    assert direct == aggregate
    assert "family_a_invalid_field_type" in _codes(direct)
    assert hostile.override_access_count == 0


def test_validate_task_semantics_does_not_execute_hostile_generator_string(
    family_a_tasks,
):
    task = family_a_tasks[0]
    hostile = HostileFamilyString("legacy_p63_episode_compiler")
    generator = _construct_replace(task.source.generator, generator_name=hostile)
    source = _construct_replace(task.source, generator=generator)
    malformed = _construct_replace(task, source=source)

    direct = validate_family_a_task(malformed)
    aggregate = validate_task_semantics(malformed)

    assert aggregate == direct
    assert "family_a_invalid_field_type" in _codes(direct)
    assert hostile.override_access_count == 0


def test_validate_family_a_task_is_public_and_other_families_are_inapplicable(
    family_a_tasks,
    family_d_tasks,
):
    import mub.vnext.validation as validation

    assert validation.validate_family_a_task is validate_family_a_task
    report = validate_family_a_task(family_d_tasks["semantic_near_miss"])
    assert _codes(report) == {"family_a_inapplicable_task_family"}


def test_validate_family_a_task_preserves_distractor_answer_leak_checks(family_a_tasks):
    task = _family_a_task(family_a_tasks, difficulty="medium", depth=1)
    payload = _payload(task)
    query_id = task.queries[0].query_id
    answer = payload["gold"]["gold_answers"][query_id]
    event = next(
        event
        for event in payload["events"]
        if event["role"] == EventRole.SAME_NAME_OTHER_ENTITY.value
    )
    event["raw_text"] += f" The current answer is {answer}."
    corrupted = MemUpdateTask.model_validate(payload)

    direct = validate_family_a_task(corrupted)
    aggregate = validate_task_semantics(corrupted)

    assert aggregate == direct
    assert "distractor_text_contains_accepted_answer" in _codes(direct)


def test_validate_family_d_task_preserves_distractor_answer_leak_checks(family_d_tasks):
    task = family_d_tasks["semantic_near_miss"]
    payload = _payload(task)
    query_id = task.queries[0].query_id
    answer = payload["gold"]["gold_answers"][query_id]
    event = _event_with_lifecycle(payload, "independent_current")
    event["raw_text"] += f" The current answer is {answer}."
    corrupted = MemUpdateTask.model_validate(payload)

    direct = validate_family_d_task(corrupted)
    aggregate = validate_task_semantics(corrupted)

    assert aggregate == direct
    assert "distractor_text_contains_accepted_answer" in _codes(direct)


def test_validate_family_d_task_rejects_duplicate_independent_noop_observation(
    family_d_tasks,
):
    payload = _payload(family_d_tasks["density_0.75"])
    noops = [
        event
        for event in payload["events"]
        if event["metadata"].get("lifecycle") == "independent_noop"
    ]
    assert len(noops) >= 2
    noops[1]["metadata"]["surface_statement"] = noops[0]["metadata"][
        "surface_statement"
    ]
    noops[1]["raw_text"] = noops[0]["raw_text"]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_noop_visibility_mismatch" in _codes(report)


@pytest.mark.parametrize("surface_variant", (0, 1, 2))
def test_validate_family_d_task_accepts_valid_generated_sample(
    family_d_tasks,
    surface_variant,
):
    report = validate_family_d_task(
        family_d_tasks[f"duplicate_current_variant_{surface_variant}"]
    )

    assert report.valid
    assert report.issues == ()


def test_validate_family_d_task_is_exported_from_validation_package():
    import mub.vnext.validation as validation

    assert validation.validate_family_d_task is validate_family_d_task


def test_validate_task_semantics_dispatches_family_d_without_duplicate_generic_issues(
    family_d_tasks,
):
    payload = _payload(family_d_tasks["duplicate_current"])
    near_density = 0.2500000001
    payload["metadata"]["extra"]["stratification"][
        "configured_noop_density"
    ] = near_density
    payload["metadata"]["extra"]["stratification"][
        "observed_noop_density"
    ] = near_density
    payload["metadata"]["resolved_profile"]["noop_density"] = near_density
    corrupted = MemUpdateTask.model_validate(payload)

    direct = validate_family_d_task(corrupted)
    aggregate = validate_task_semantics(corrupted)

    assert aggregate == direct
    assert "family_d_canonical_noop_density_mismatch" in _codes(aggregate)


def test_validate_task_semantics_routes_hostile_family_string_without_overrides(
    family_d_tasks,
):
    hostile = HostileFamilyString(TaskFamily.NOOP_WRITE_DISCIPLINE.value)
    malformed = _construct_replace(
        family_d_tasks["duplicate_current"],
        task_family=hostile,
    )

    direct = validate_family_d_task(malformed)
    aggregate = validate_task_semantics(malformed)

    assert aggregate == direct
    assert "family_d_invalid_field_type" in _codes(aggregate)
    assert hostile.override_access_count == 0


def test_validate_task_semantics_preserves_non_family_dispatch():
    config = load_pilot_config(CONFIG_PATH)
    context = GenerationContext(config=config, code_revision="non-family-dispatch-test")
    task = render_core(
        generate_family_a_cores(config)[0],
        split=Split.TEST,
        surface_variant=0,
        context=context,
    )

    expected = merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(task),
    )

    assert validate_task_semantics(task) == expected


def test_validate_family_d_task_uses_semantic_event_order_not_action_storage_order(
    family_d_tasks,
):
    payload = _payload(family_d_tasks["duplicate_current"])
    payload["gold"]["actions"].reverse()
    reordered = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(reordered)

    assert report.valid
    assert report.issues == ()


def test_validate_family_d_task_detects_noop_converted_to_write(family_d_tasks):
    payload = _payload(family_d_tasks["semantic_near_miss"])
    event = _event_with_lifecycle(payload, "trap_noop")
    action = _action_for_event(payload, event)
    action["operation"] = Operation.ADD.value
    action["target_object_keys"] = [payload["queries"][0]["target_object_keys"][0]]
    action["value"] = "illicit-noop-write"
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_noop_state_mutation" in _codes(report)


def test_validate_family_d_task_rejects_coordinated_semantic_noop_rewrite(
    family_d_tasks,
):
    corrupted = _rewrite_semantic_noop_as_non_target_update(
        family_d_tasks["semantic_near_miss_with_prior_write"]
    )

    report = validate_family_d_task(corrupted)

    assert not report.valid
    assert "family_d_noop_state_mutation" in _codes(report)
    assert "family_d_noop_semantics_mismatch" in _codes(report)
    assert "family_d_canonical_noop_count_mismatch" in _codes(report)


def test_validate_family_d_task_rejects_removed_event_trap_binding(
    family_d_tasks,
):
    corrupted = _rewrite_semantic_noop_as_non_target_update(
        family_d_tasks["semantic_near_miss_with_prior_write"],
        erase_trap_signals=True,
    )

    report = validate_family_d_task(corrupted)

    assert "family_d_trap_metadata_mismatch" in _codes(report)
    assert "family_d_noop_state_mutation" in _codes(report)


def test_validate_family_d_task_enforces_canonical_event_count(family_d_tasks):
    corrupted = _append_extra_noop(family_d_tasks["duplicate_current"])

    report = validate_family_d_task(corrupted)

    assert "family_d_canonical_event_count_mismatch" in _codes(report)


@pytest.mark.parametrize("density", (0.25, 0.50, 0.75))
def test_validate_family_d_task_requires_exact_canonical_density(
    family_d_tasks,
    density,
):
    payload = _payload(family_d_tasks[f"density_{density}"])
    near_density = density + 1e-10
    payload["metadata"]["extra"]["stratification"][
        "configured_noop_density"
    ] = near_density
    payload["metadata"]["extra"]["stratification"][
        "observed_noop_density"
    ] = near_density
    payload["metadata"]["resolved_profile"]["noop_density"] = near_density
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_canonical_noop_density_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("stratification", "num_events"),
        ("stratification", "noop_count"),
        ("stratification", "true_write_count"),
        ("stratification", "num_target_updates"),
        ("stratification", "duplicate_current_count"),
        ("stratification", "trap_position"),
        ("profile", "context_length"),
    ),
)
def test_validate_family_d_task_rejects_non_integer_counter_metadata(
    family_d_tasks,
    container,
    field,
):
    payload = _payload(family_d_tasks["duplicate_current"])
    if container == "stratification":
        target = payload["metadata"]["extra"]["stratification"]
    else:
        target = payload["metadata"]["resolved_profile"]
    target[field] = float(target[field])
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_integer_metadata_type_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("trap_type", "mutation"),
    (
        ("semantic_near_miss", "lifecycle"),
        ("duplicate_current", "lifecycle"),
        ("semantic_near_miss", "role"),
        ("semantic_near_miss", "statement"),
    ),
)
def test_validate_family_d_task_cross_checks_noop_semantic_signals(
    family_d_tasks,
    trap_type,
    mutation,
):
    payload = _payload(family_d_tasks[trap_type])
    event = next(
        item for item in payload["events"] if item["metadata"].get("trap_type") == trap_type
    )
    if mutation == "lifecycle":
        event["metadata"]["lifecycle"] = "independent_noop"
    elif mutation == "role":
        event["role"] = EventRole.NEUTRAL.value
    else:
        event["metadata"]["surface_statement"] = "This text now directs a write."
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_noop_semantics_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("lifecycle", "corruption"),
    (
        ("trap_noop", "statement"),
        ("trap_noop", "raw_text"),
        ("independent_noop", "statement"),
        ("independent_noop", "raw_text"),
    ),
)
def test_validate_family_d_task_rejects_contradictory_noop_payloads(
    family_d_tasks,
    lifecycle,
    corruption,
):
    payload = _payload(family_d_tasks["semantic_near_miss"])
    event = _event_with_lifecycle(payload, lifecycle)
    if corruption == "statement":
        event["metadata"]["surface_statement"] += " Now update another value."
        event["raw_text"] = (
            f"{event['metadata']['surface_statement']} No memory change is required."
        )
    else:
        event["raw_text"] += " Actually, perform a memory update."
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_noop_visibility_mismatch" in _codes(report)


def test_validate_family_d_task_detects_gold_claiming_a_noop_state_change(
    family_d_tasks,
):
    task = family_d_tasks["semantic_near_miss"]
    payload = _payload(task)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    query_id = task.queries[0].query_id
    payload["gold"]["final_state"][target_id] = "illicit-noop-state"
    payload["gold"]["version_history"][target_id] = ["illicit-noop-state"]
    payload["gold"]["gold_answers"][query_id] = "illicit-noop-state"
    payload["gold"]["acceptable_answers"][query_id] = "illicit-noop-state"
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_noop_state_mutation" in _codes(report)


def test_validate_family_d_task_distinguishes_duplicate_metadata_and_count(family_d_tasks):
    payload = _payload(family_d_tasks["duplicate_current"])
    event = _event_with_lifecycle(payload, "trap_noop")
    event["metadata"].pop("allow_accepted_answer_ambiguity")
    payload["metadata"]["extra"]["stratification"]["duplicate_current_count"] = 0
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_duplicate_current_metadata_mismatch" in _codes(report)
    assert "family_d_duplicate_current_count_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("trap_type", "lifecycle"),
    (
        ("semantic_near_miss", "independent_current"),
        ("other_entity_correction", "correction_after"),
    ),
)
def test_validate_family_d_task_rejects_non_target_value_equal_to_target_answer(
    family_d_tasks,
    trap_type,
    lifecycle,
):
    task = family_d_tasks[trap_type]
    payload = _payload(task)
    target_event = _event_with_lifecycle(payload, "target_current")
    target_value = _action_for_event(payload, target_event)["value"]
    distractor_event = _event_with_lifecycle(payload, lifecycle)
    distractor_action = _action_for_event(payload, distractor_event)
    typed_action = next(
        action
        for action in task.gold.actions
        if action.action_id == distractor_action["action_id"]
    )
    distractor_id = typed_action.target_object_keys[0].canonical_id
    distractor_action["value"] = target_value
    payload["gold"]["final_state"][distractor_id] = target_value
    if lifecycle == "correction_after":
        payload["gold"]["version_history"][distractor_id][-1] = target_value
    else:
        payload["gold"]["version_history"][distractor_id] = [target_value]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_distractor_target_value_collision" in _codes(report)


@pytest.mark.parametrize(
    ("target_value", "distractor_value", "expected_collision"),
    (
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"items": [True]}, {"items": [1]}, False),
        (1, 1.0, False),
        ([1, 2], [2, 1], False),
    ),
)
def test_validate_family_d_task_compares_values_as_canonical_json(
    family_d_tasks,
    target_value,
    distractor_value,
    expected_collision,
):
    task = family_d_tasks["semantic_near_miss"]
    payload = _payload(task)
    target_event = _event_with_lifecycle(payload, "target_current")
    target_action = _action_for_event(payload, target_event)
    target_id = task.queries[0].target_object_keys[0].canonical_id
    query_id = task.queries[0].query_id
    target_action["value"] = target_value
    payload["gold"]["final_state"][target_id] = target_value
    payload["gold"]["version_history"][target_id] = [target_value]
    payload["gold"]["gold_answers"][query_id] = target_value
    payload["gold"]["acceptable_answers"][query_id] = target_value

    distractor_event = _event_with_lifecycle(payload, "independent_current")
    distractor_action = _action_for_event(payload, distractor_event)
    typed_action = next(
        action
        for action in task.gold.actions
        if action.action_id == distractor_action["action_id"]
    )
    distractor_id = typed_action.target_object_keys[0].canonical_id
    distractor_action["value"] = distractor_value
    payload["gold"]["final_state"][distractor_id] = distractor_value
    payload["gold"]["version_history"][distractor_id] = [distractor_value]
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert (
        "family_d_distractor_target_value_collision" in _codes(report)
    ) is expected_collision


@pytest.mark.parametrize("mismatch", ("identity", "value"))
def test_validate_family_d_task_binds_duplicate_observation_to_target_and_value(
    family_d_tasks,
    mismatch,
):
    task = family_d_tasks["duplicate_current"]
    payload = _payload(task)
    target = task.queries[0].target_object_keys[0]
    target_event = _event_with_lifecycle(payload, "target_current")
    target_value = _action_for_event(payload, target_event)["value"]
    duplicate_event = next(
        event
        for event in payload["events"]
        if event["metadata"].get("trap_type") == "duplicate_current"
    )
    observed_identity = (
        "unrelated_entity.unrelated_attribute"
        if mismatch == "identity"
        else f"{target.entity}.{target.attribute}"
    )
    observed_value = "unrelated-value" if mismatch == "value" else target_value
    statement = (
        f"{observed_identity} remains exactly {observed_value}; "
        "this repeats the exact current target value."
    )
    duplicate_event["metadata"]["surface_statement"] = statement
    duplicate_event["raw_text"] = f"{statement} No memory change is required."
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_duplicate_current_visibility_mismatch" in _codes(report)


@pytest.mark.parametrize(
    "contradiction",
    ("negated_prefix", "trailing_change", "raw_trailing_change"),
)
def test_validate_family_d_task_rejects_contradictory_duplicate_observation(
    family_d_tasks,
    contradiction,
):
    payload = _payload(family_d_tasks["duplicate_current"])
    duplicate_event = next(
        event
        for event in payload["events"]
        if event["metadata"].get("trap_type") == "duplicate_current"
    )
    canonical = duplicate_event["metadata"]["surface_statement"]
    if contradiction == "negated_prefix":
        contradictory = f"It is false that {canonical}"
        duplicate_event["metadata"]["surface_statement"] = contradictory
        duplicate_event["raw_text"] = contradictory
    elif contradiction == "trailing_change":
        contradictory = f"{canonical} However, immediately change it to another value."
        duplicate_event["metadata"]["surface_statement"] = contradictory
        duplicate_event["raw_text"] = contradictory
    else:
        duplicate_event["raw_text"] = (
            f"{canonical} No memory change is required. "
            "Actually, update it to another value."
        )
    corrupted = MemUpdateTask.model_validate(payload)

    report = validate_family_d_task(corrupted)

    assert "family_d_duplicate_current_visibility_mismatch" in _codes(report)


def test_validate_family_d_task_distinguishes_correction_lifecycle_and_target_isolation(
    family_d_tasks,
):
    lifecycle_payload = _payload(family_d_tasks["other_entity_correction"])
    setup_event = _event_with_lifecycle(lifecycle_payload, "correction_before")
    setup_event["metadata"]["lifecycle"] = "independent_current"
    lifecycle_corrupted = MemUpdateTask.model_validate(lifecycle_payload)

    lifecycle_report = validate_family_d_task(lifecycle_corrupted)

    assert "family_d_correction_lifecycle_corruption" in _codes(lifecycle_report)

    isolation_payload = _payload(family_d_tasks["other_attribute_correction"])
    correction_event = _event_with_lifecycle(isolation_payload, "correction_after")
    correction_action = _action_for_event(isolation_payload, correction_event)
    correction_action["target_object_keys"] = [
        isolation_payload["queries"][0]["target_object_keys"][0]
    ]
    isolation_corrupted = MemUpdateTask.model_validate(isolation_payload)

    isolation_report = validate_family_d_task(isolation_corrupted)

    assert "family_d_target_isolation_corruption" in _codes(isolation_report)


def test_validate_family_d_task_rejects_malformed_and_marks_other_families_inapplicable(
    family_d_tasks,
):
    malformed_report = validate_family_d_task({"task_family": "noop_write_discipline"})
    assert not malformed_report.valid
    assert _codes(malformed_report) == {"family_d_invalid_task_type"}

    payload = _payload(family_d_tasks["semantic_near_miss"])
    payload["task_family"] = TaskFamily.REPEATED_SAME_SLOT.value
    other_family = MemUpdateTask.model_validate(payload)
    inapplicable_report = validate_family_d_task(other_family)

    assert not inapplicable_report.valid
    assert _codes(inapplicable_report) == {"family_d_inapplicable_task_family"}


def test_validate_family_d_task_handles_constructed_models_with_missing_fields():
    missing_family = MemUpdateTask.model_construct()
    first = validate_family_d_task(missing_family)
    second = validate_family_d_task(missing_family)

    assert not first.valid
    assert first == second
    assert len(first.issues) <= MAX_REPORT_ISSUES
    assert "family_d_malformed_task" in _codes(first)

    missing_paths = MemUpdateTask.model_construct(
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE.value
    )
    paths_report = validate_family_d_task(missing_paths)

    assert not paths_report.valid
    assert len(paths_report.issues) <= MAX_REPORT_ISSUES
    assert "family_d_malformed_record" in _codes(paths_report)


@pytest.mark.parametrize(
    "location",
    (
        "action.operation",
        "action.scope",
        "event.role",
        "task.difficulty",
        "source.source_type",
        "metadata.split",
        "metadata.profile_name",
        "query.query_type",
        "query.answer_schema",
        "query.evaluation_mode",
        "canonical.disposition",
        "canonical.resolution_status",
    ),
)
def test_validate_family_d_task_rejects_forged_enum_like_values_without_access(
    family_d_tasks,
    location,
):
    task = family_d_tasks["duplicate_current"]
    forged = ExplosiveEnumLike()
    if location.startswith("action."):
        malformed = _replace_action(task, 0, **{location.split(".")[1]: forged})
    elif location == "event.role":
        events = list(task.events)
        events[0] = _construct_replace(events[0], role=forged)
        malformed = _construct_replace(task, events=events)
    elif location == "task.difficulty":
        malformed = _construct_replace(task, difficulty=forged)
    elif location == "source.source_type":
        source = _construct_replace(task.source, source_type=forged)
        malformed = _construct_replace(task, source=source)
    elif location.startswith("metadata."):
        metadata = _construct_replace(
            task.metadata,
            **{location.split(".")[1]: forged},
        )
        malformed = _construct_replace(task, metadata=metadata)
    elif location.startswith("query."):
        queries = list(task.queries)
        queries[0] = _construct_replace(
            queries[0],
            **{location.split(".")[1]: forged},
        )
        malformed = _construct_replace(task, queries=queries)
    else:
        field = location.split(".")[1]
        canonical = CanonicalAnswer.model_construct(
            disposition=forged if field == "disposition" else "abstained",
            resolution_status=(
                forged if field == "resolution_status" else "ambiguous"
            ),
            selected_candidate_ids=[],
            abstention_reason="malformed",
            value=None,
        )
        malformed = _replace_gold(task, canonical_answers={"forged": canonical})

    first = validate_family_d_task(malformed)
    second = validate_family_d_task(malformed)

    assert first == second
    assert forged.value_access_count == 0
    assert "family_d_invalid_enum_type" in _codes(first)


def test_schema_preflight_revalidates_shared_list_across_annotations(family_d_tasks):
    task = family_d_tasks["duplicate_current"]
    shared = list(task.gold.expected_present_objects)
    gold = _construct_replace(task.gold, expected_present_objects=shared)
    metadata = _construct_replace(task.metadata, tags=shared)
    malformed = _construct_replace(task, gold=gold, metadata=metadata)

    report = validate_family_d_task(malformed)

    assert any(
        issue.code == "family_d_invalid_field_type"
        and issue.path.startswith("task.metadata.tags[")
        for issue in report.issues
    )


def test_schema_preflight_revalidates_shared_dict_across_annotations(family_d_tasks):
    task = family_d_tasks["duplicate_current"]
    shared = dict(task.source.provenance)
    source = _construct_replace(task.source, provenance=shared)
    gold = _construct_replace(task.gold, version_history=shared)
    malformed = _construct_replace(task, source=source, gold=gold)

    report = validate_family_d_task(malformed)

    assert any(
        issue.code == "family_d_malformed_collection"
        and issue.path.startswith("task.gold.version_history.")
        for issue in report.issues
    )


def test_schema_preflight_revalidates_shared_tuple_across_annotations(family_d_tasks):
    task = family_d_tasks["duplicate_current"]
    shared = ("not-json",)
    provenance = dict(task.source.provenance)
    provenance["shared"] = shared
    source = _construct_replace(task.source, provenance=provenance)
    events = list(task.events)
    anchor = dict(events[0].source_anchor)
    anchor["shared"] = shared
    events[0] = _construct_replace(events[0], source_anchor=anchor)
    malformed = _construct_replace(task, source=source, events=events)

    report = validate_family_d_task(malformed)

    malformed_paths = {
        issue.path
        for issue in report.issues
        if issue.code == "family_d_malformed_json"
    }
    assert "task.source.provenance.shared" in malformed_paths
    assert "task.events[0].source_anchor.shared" in malformed_paths


@pytest.mark.parametrize(
    "location",
    (
        "source.generator",
        "metadata.legacy_provenance",
        "metadata.split_key",
        "metadata.tags",
        "event.timestamp",
        "event.speaker",
        "action.effective_at",
        "query.text",
    ),
)
def test_validate_family_d_task_schema_preflight_rejects_forged_typed_fields(
    family_d_tasks,
    location,
):
    task = family_d_tasks["duplicate_current"]
    malformed_value = (
        [float("nan")]
        if location == "metadata.tags"
        else float("nan")
        if location
        in {"event.timestamp", "event.speaker", "action.effective_at", "query.text"}
        else object()
    )
    if location.startswith("source."):
        source = _construct_replace(
            task.source,
            **{location.split(".")[1]: malformed_value},
        )
        malformed = _construct_replace(task, source=source)
    elif location.startswith("metadata."):
        metadata = _construct_replace(
            task.metadata,
            **{location.split(".")[1]: malformed_value},
        )
        malformed = _construct_replace(task, metadata=metadata)
    elif location.startswith("event."):
        events = list(task.events)
        events[0] = _construct_replace(
            events[0],
            **{location.split(".")[1]: malformed_value},
        )
        malformed = _construct_replace(task, events=events)
    elif location.startswith("action."):
        malformed = _replace_action(
            task,
            0,
            **{location.split(".")[1]: malformed_value},
        )
    else:
        queries = list(task.queries)
        queries[0] = _construct_replace(
            queries[0],
            **{location.split(".")[1]: malformed_value},
        )
        malformed = _construct_replace(task, queries=queries)

    first = validate_family_d_task(malformed)
    second = validate_family_d_task(malformed)

    assert first == second
    assert "family_d_invalid_field_type" in _codes(first)


@pytest.mark.parametrize("corruption", ("missing", "extra"))
def test_validate_family_d_task_schema_preflight_checks_raw_model_fields(
    family_d_tasks,
    corruption,
):
    task = family_d_tasks["duplicate_current"]
    data = dict(task.__dict__)
    if corruption == "missing":
        data.pop("task_id")
        malformed = MemUpdateTask.model_construct(**data)
    else:
        malformed = MemUpdateTask.model_construct(**data)
        malformed.__dict__["attacker_extra"] = object()

    report = validate_family_d_task(malformed)

    assert "family_d_malformed_record" in _codes(report)


def test_validate_family_d_task_rejects_custom_sequences_without_iteration():
    sequence = ExplosiveSequence()
    malformed = MemUpdateTask.model_construct(
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE.value,
        events=sequence,
    )

    first = validate_family_d_task(malformed)
    second = validate_family_d_task(malformed)

    assert first == second
    assert not first.valid
    assert sequence.iteration_count == 0
    assert "family_d_malformed_collection" in _codes(first)


def test_validate_family_d_task_rejects_non_mapping_expected_effect(family_d_tasks):
    task = family_d_tasks["duplicate_current"]
    noop_event = next(
        event
        for event in task.events
        if event.metadata.get("lifecycle") == "trap_noop"
    )
    action_id = noop_event.gold_action_ids[0]
    action_index = next(
        index
        for index, action in enumerate(task.gold.actions)
        if action.action_id == action_id
    )
    malformed = _replace_action(task, action_index, expected_effect=[])

    report = validate_family_d_task(malformed)

    assert not report.valid
    assert "family_d_malformed_mapping" in _codes(report)


@pytest.mark.parametrize("field", ("final_state", "version_history"))
def test_validate_family_d_task_rejects_custom_gold_mappings_without_iteration(
    family_d_tasks,
    field,
):
    mapping = ExplosiveMapping()
    malformed = _replace_gold(
        family_d_tasks["duplicate_current"],
        **{field: mapping},
    )

    first = validate_family_d_task(malformed)
    second = validate_family_d_task(malformed)

    assert first == second
    assert not first.valid
    assert mapping.iteration_count == 0
    assert "family_d_malformed_mapping" in _codes(first)


@pytest.mark.parametrize(
    "location",
    (
        "gold.final_state",
        "gold.version_history",
        "gold.gold_answers",
        "gold.acceptable_answers",
        "event.metadata",
        "action.expected_effect",
        "query.metadata",
        "metadata.resolved_profile",
        "metadata.extra",
    ),
)
def test_validate_family_d_task_rejects_oversized_nested_maps_before_generic_validation(
    family_d_tasks,
    monkeypatch,
    location,
):
    generic_calls = []

    def unexpected_generic_call(task):
        generic_calls.append(task)
        raise AssertionError("oversized nested map reached a generic validator")

    monkeypatch.setattr(pilot_module, "validate_task", unexpected_generic_call)
    monkeypatch.setattr(pilot_module, "validate_gold_replay", unexpected_generic_call)
    task = family_d_tasks["duplicate_current"]
    oversized = {f"key-{index}": "value" for index in range(1000)}
    if location.startswith("gold."):
        malformed = _replace_gold(task, **{location.split(".", 1)[1]: oversized})
    elif location == "event.metadata":
        events = list(task.events)
        events[0] = _construct_replace(events[0], metadata=oversized)
        malformed = _construct_replace(task, events=events)
    elif location == "action.expected_effect":
        malformed = _replace_action(task, 0, expected_effect=oversized)
    elif location == "query.metadata":
        queries = list(task.queries)
        queries[0] = _construct_replace(queries[0], metadata=oversized)
        malformed = _construct_replace(task, queries=queries)
    else:
        field = location.split(".", 1)[1]
        metadata = _construct_replace(task.metadata, **{field: oversized})
        malformed = _construct_replace(task, metadata=metadata)

    report = validate_family_d_task(malformed)

    assert "family_d_input_size_limit" in _codes(report)
    assert generic_calls == []


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_validate_family_d_task_rejects_non_finite_nested_numbers(
    family_d_tasks,
    value,
):
    task = family_d_tasks["duplicate_current"]
    events = list(task.events)
    metadata = dict(events[0].metadata)
    metadata["non_finite"] = value
    events[0] = _construct_replace(events[0], metadata=metadata)
    malformed = _construct_replace(task, events=events)

    report = validate_family_d_task(malformed)

    assert "family_d_non_finite_json_number" in _codes(report)


@pytest.mark.parametrize("cycle_kind", ("self", "mutual"))
def test_validate_family_d_task_rejects_nested_cycles_deterministically(
    family_d_tasks,
    cycle_kind,
):
    if cycle_kind == "self":
        cyclic = {}
        cyclic["self"] = cyclic
    else:
        cyclic = {}
        partner = []
        cyclic["partner"] = partner
        partner.append(cyclic)
    task = family_d_tasks["duplicate_current"]
    events = list(task.events)
    events[0] = _construct_replace(events[0], metadata=cyclic)
    malformed = _construct_replace(task, events=events)

    first = validate_family_d_task(malformed)
    second = validate_family_d_task(malformed)

    assert first == second
    assert "family_d_cyclic_json" in _codes(first)


def test_validate_family_d_task_rejects_excessive_nested_depth(family_d_tasks):
    root = {}
    cursor = root
    for index in range(40):
        child = {}
        cursor[f"level-{index}"] = child
        cursor = child
    task = family_d_tasks["duplicate_current"]
    events = list(task.events)
    events[0] = _construct_replace(events[0], metadata=root)
    malformed = _construct_replace(task, events=events)

    report = validate_family_d_task(malformed)

    assert "family_d_input_size_limit" in _codes(report)


def test_validate_family_d_task_is_deterministic_immutable_and_does_no_disk_io(
    family_d_tasks,
    monkeypatch,
):
    payload = _payload(family_d_tasks["duplicate_current"])
    event = _event_with_lifecycle(payload, "trap_noop")
    event["metadata"].pop("allow_accepted_answer_ambiguity")
    payload["metadata"]["extra"]["stratification"]["duplicate_current_count"] = 0
    task = MemUpdateTask.model_validate(payload)
    before = task.model_dump(mode="json")

    def fail_disk_io(*args, **kwargs):
        raise AssertionError("validation must not perform disk I/O")

    monkeypatch.setattr(builtins, "open", fail_disk_io)
    monkeypatch.setattr(Path, "open", fail_disk_io)

    first = validate_family_d_task(task)
    second = validate_family_d_task(task)

    assert first == second
    assert task.model_dump(mode="json") == before
    assert [
        (issue.code, issue.path, issue.message, issue.severity)
        for issue in first.issues
    ] == sorted(
        (issue.code, issue.path, issue.message, issue.severity)
        for issue in first.issues
    )


def test_validate_family_d_task_rejects_oversized_input_before_generic_validation(
    monkeypatch,
):
    generic_calls = []

    def unexpected_generic_call(task):
        generic_calls.append(task)
        raise AssertionError("oversized input reached a generic validator")

    monkeypatch.setattr(pilot_module, "validate_task", unexpected_generic_call)
    monkeypatch.setattr(pilot_module, "validate_gold_replay", unexpected_generic_call)
    malformed_event = SimpleNamespace(
        event_id="",
        sequence_index=None,
        gold_action_ids=[],
        role=None,
        source_anchor=None,
        metadata=None,
    )
    malformed = MemUpdateTask.model_construct(
        task_id="",
        schema_version="malformed",
        task_family=TaskFamily.NOOP_WRITE_DISCIPLINE.value,
        difficulty=None,
        source=SimpleNamespace(source_id=None, source_type=None),
        events=[malformed_event for _ in range(200)],
        target_objects=[],
        queries=[],
        gold=SimpleNamespace(
            actions=[],
            action_sequence=[],
            final_state={},
            version_history={},
            expected_present_objects=[],
            expected_absent_objects=[],
            gold_source_event_ids=[],
            gold_answers={},
            acceptable_answers={},
            canonical_answers={},
        ),
        metadata=SimpleNamespace(
            split=None,
            profile_name=None,
            resolved_profile={},
            extra={"stratification": {}},
        ),
    )

    report = validate_family_d_task(malformed)

    assert not report.valid
    assert len(report.issues) <= MAX_REPORT_ISSUES
    assert "family_d_input_size_limit" in _codes(report)
    assert generic_calls == []
