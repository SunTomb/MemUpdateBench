from collections import Counter, defaultdict
from pathlib import Path

import pytest

from mub.vnext.contracts.common import thaw_json
from mub.vnext.contracts.enums import AnswerSchema, Operation, QueryType, Split, SupportReason, TaskFamily
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3, AdapterInfoV3
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import (
    CurrentSelector,
    EventAnchorSelector,
    ExactVersionSelector,
    LogicalTimeAnchorSelector,
    MemUpdateTaskV3,
    OrderedHistorySelector,
    PreviousSelector,
    TransitionSelector,
    VersionHistoryEntry,
)
from mub.vnext.generation.core import GenerationContext, SemanticCore
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.validation.replay_v3 import evaluate_evidence_v3, replay_task_v3, resolve_query_v3


ROOT = Path(__file__).resolve().parents[2]
CORE_CONFIG = ROOT / "configs" / "vnext" / "core.yaml"
SELECTOR_KINDS = (
    "current",
    "previous",
    "exact_version",
    "event_anchor",
    "logical_time_anchor",
    "transition",
    "ordered_history",
)
HISTORICAL_PRINCIPAL_PATHS = frozenset({
    "historical_scores.current_state_accuracy",
    "historical_scores.previous_state_accuracy",
    "historical_scores.point_in_time_accuracy",
    "historical_scores.transition_accuracy",
    "historical_scores.ordered_history_accuracy",
    "historical_scores.historical_support_recall",
    "historical_scores.historical_distance_accuracy",
})
COMMON_PRINCIPAL_PATHS = frozenset({
    "action_scores.full_action_exact_match",
    "state_scores.final_state_accuracy",
    "answer_scores.exact_match",
})
EXPECTED_PRINCIPAL_BY_QUERY = {
    QueryTypeV3.CURRENT: COMMON_PRINCIPAL_PATHS | {"historical_scores.current_state_accuracy"},
    QueryTypeV3.PREVIOUS: COMMON_PRINCIPAL_PATHS | {
        "historical_scores.previous_state_accuracy",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    },
    QueryTypeV3.POINT_IN_TIME: COMMON_PRINCIPAL_PATHS | {
        "historical_scores.point_in_time_accuracy",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    },
    QueryTypeV3.TRANSITION: COMMON_PRINCIPAL_PATHS | {
        "historical_scores.transition_accuracy",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    },
    QueryTypeV3.ORDERED_HISTORY: COMMON_PRINCIPAL_PATHS | {
        "historical_scores.ordered_history_accuracy",
        "historical_scores.historical_support_recall",
        "historical_scores.historical_distance_accuracy",
    },
}
FULL_CAPABILITIES = AdapterCapabilitiesV3(
    supports_isolated_reset=True,
    supports_event_ingest=True,
    supports_add=True,
    supports_update=True,
    supports_noop=True,
    supports_native_answer=True,
    supports_historical_query=True,
    exports_version_history=True,
    exports_entries=True,
    exports_raw_state=True,
    exports_source_event_ids=True,
    exports_timestamps_or_order=True,
    exports_object_keys=True,
    exports_values=True,
    exports_retrieval_ids=True,
    exports_retrieval_scores=True,
    exports_action_trace=True,
)
ORACLE_INFO = AdapterInfoV3(
    adapter_id="family-f-oracle",
    adapter_version="1",
    system_name="reference",
    system_version="1",
    configuration_hash="a" * 64,
)


def _api():
    from mub.vnext.generation.family_f import (
        compile_family_f_micro_pilot,
        generate_core_family_f_cores,
        validate_family_f_core,
        validate_family_f_task,
    )

    return (
        generate_core_family_f_cores,
        validate_family_f_core,
        validate_family_f_task,
        compile_family_f_micro_pilot,
    )


def _micro_validators():
    from mub.vnext.generation.family_f import (
        validate_family_f_micro_core,
        validate_family_f_micro_task,
    )

    return validate_family_f_micro_core, validate_family_f_micro_task


def _config():
    return load_core_config(CORE_CONFIG)


def _mutate_core(core: SemanticCore, mutate):
    payload = core.model_dump(mode="python")
    mutate(payload)
    return SemanticCore.model_validate(payload)


def _metric_value(score, path):
    layer, leaf = path.split(".", 1)
    return getattr(getattr(score, layer), leaf)


def test_family_f_generator_has_three_trajectories_and_exactly_seven_typed_selector_cores_each():
    generate, validate_core, _, _ = _api()
    cores = generate(_config())

    assert len(cores) == 21
    assert cores == generate(_config())
    assert {core.task_family for core in cores} == {TaskFamily.CURRENT_HISTORICAL_QUERY}
    by_trajectory = defaultdict(list)
    for core in cores:
        by_trajectory[core.trajectory_id].append(core)
        validate_core(core)
    assert len(by_trajectory) == 3
    assert all(len(group) == 7 for group in by_trajectory.values())
    assert all(
        Counter(core.query_selector.kind for core in group)
        == Counter({kind: 1 for kind in SELECTOR_KINDS})
        for group in by_trajectory.values()
    )
    for group in by_trajectory.values():
        for core in group:
            assert len(core.events) >= 4
            assert all(event.operation in {Operation.ADD, Operation.UPDATE} for event in core.events)
            assert all(event.value is not None for event in core.events)
            assert not any("selector" in key for event in core.events for key in event.metadata)


def test_family_f_full_schedule_has_sixty_seven_selector_trajectories():
    from mub.vnext.generation.core_render_v3 import render_core_v3
    from mub.vnext.generation.identity import stable_id

    generate, _, _, _ = _api()
    config = _config()
    cores = generate(config, profile="full")

    assert len(cores) == 420
    assert len({core.core_id for core in cores}) == 420
    assert Counter(core.difficulty.value for core in cores) == {
        "easy": 140,
        "medium": 140,
        "hard": 140,
    }
    by_trajectory = defaultdict(list)
    for core in cores:
        by_trajectory[core.trajectory_id].append(core)
        assert len(core.events) >= 4
        assert not any(event.operation is Operation.DELETE for event in core.events)
    assert len(by_trajectory) == 60
    assert all(len(group) == 7 for group in by_trajectory.values())
    assert all(
        Counter(core.query_selector.kind for core in group)
        == Counter({kind: 1 for kind in SELECTOR_KINDS})
        for group in by_trajectory.values()
    )

    group = next(iter(by_trajectory.values()))
    tasks = [
        render_core_v3(
            core,
            split=Split.TRAIN,
            surface_variant=0,
            context=GenerationContext(config=config, code_revision="task-561-red"),
        )
        for core in group
    ]
    expected_version_group = stable_id(
        "version_group", {"trajectory_id": group[0].trajectory_id}
    )
    assert {task.metadata.split_key.trajectory_id for task in tasks} == {
        group[0].trajectory_id
    }
    assert {task.metadata.split_key.version_group_id for task in tasks} == {
        expected_version_group
    }
    assert {task.metadata.split for task in tasks} == {Split.TRAIN}


def test_family_f_full_generator_builds_trajectory_material_once_per_trajectory(
    monkeypatch,
):
    import mub.vnext.generation.family_f as family_f

    config = _config()
    schedule = config.families.current_historical_query.schedule
    original_core_event = family_f.CoreEvent
    construction_count = 0

    def instrumented_core_event(*args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        return original_core_event(*args, **kwargs)

    monkeypatch.setattr(family_f, "CoreEvent", instrumented_core_event)
    cores = family_f.generate_core_family_f_cores(config, profile="full")

    assert len(cores) == schedule.trajectory_count * len(SELECTOR_KINDS)
    assert construction_count == (
        schedule.trajectory_count * schedule.present_versions_per_trajectory
    )


def test_family_f_full_validator_authenticates_core_and_trajectory_groups():
    from mub.vnext.generation.family_f import validate_family_f_full_core

    cores = _api()[0](_config(), profile="full")
    source = cores[0]
    other = next(core for core in cores if core.trajectory_id != source.trajectory_id)

    wrong_trajectory = source.model_copy(
        update={"trajectory_id": other.trajectory_id}
    )
    with pytest.raises(ValueError, match="trajectory|canonical"):
        validate_family_f_full_core(wrong_trajectory, _config())

    wrong_core_id = source.model_copy(update={"core_id": other.core_id})
    with pytest.raises(ValueError, match="core|canonical"):
        validate_family_f_full_core(wrong_core_id, _config())


@pytest.mark.parametrize(
    ("selector_kind", "selected_index"),
    (
        ("current", 3),
        ("previous", 2),
        ("exact_version", 1),
        ("event_anchor", 2),
        ("logical_time_anchor", 1),
    ),
)
@pytest.mark.parametrize(
    ("values", "expected_schema"),
    (
        ((10, 20, 30, 40), AnswerSchema.NUMBER),
        ((False, True, False, True), AnswerSchema.BOOLEAN),
        ((["a"], ["b"], ["c"], ["d"]), AnswerSchema.LIST),
        (({"rank": 0}, {"rank": 1}, {"rank": 2}, {"rank": 3}), AnswerSchema.OBJECT),
    ),
)
def test_family_f_generic_single_version_json_values_render_with_canonical_schema(
    values,
    expected_schema,
    selector_kind,
    selected_index,
):
    from mub.vnext.generation.core_render_v3 import render_core_v3

    generate, validate_core, validate_task, _ = _api()
    source = next(
        core
        for core in generate(_config())
        if core.query_selector.kind == selector_kind
    )
    payload = source.model_dump(mode="python")
    for event, value in zip(payload["events"], values):
        event["value"] = value
    payload["expected_answer"] = values[selected_index]
    core = SemanticCore.model_validate(payload)

    validate_core(core)
    task = render_core_v3(
        core,
        split=Split.TEST,
        surface_variant=0,
        context=GenerationContext(config=_config(), code_revision="4d3f9a6"),
    )

    assert task.queries[0].answer_schema is expected_schema
    assert thaw_json(task.gold_evidence[0].answer) == values[selected_index]
    validate_task(task)


def test_family_f_generic_event_values_may_contain_anchor_label_literals():
    from mub.vnext.generation.core_render_v3 import render_core_v3

    generate, validate_core, validate_task, _ = _api()
    source = next(
        core for core in generate(_config()) if core.query_selector.kind == "current"
    )
    values = tuple(
        f"value-{index} includes [version_index=99; "
        "event_id=inside-value; logical_time=99999999]"
        for index in range(4)
    )
    payload = source.model_dump(mode="python")
    for event, value in zip(payload["events"], values):
        event["value"] = value
    payload["expected_answer"] = values[-1]
    core = SemanticCore.model_validate(payload)

    validate_core(core)
    task = render_core_v3(
        core,
        split=Split.TEST,
        surface_variant=0,
        context=GenerationContext(config=_config(), code_revision="8d8c3ea"),
    )

    assert thaw_json(task.gold_evidence[0].answer) == values[-1]
    validate_task(task)


def test_family_f_micro_single_version_values_remain_exact_strings():
    generate, _, _, _ = _api()
    expected_trajectories = {
        ("email", "sms", "push", "voice"),
        ("planned", "active", "paused", "complete"),
        ("room_a", "room_b", "room_c", "room_d"),
    }

    cores = generate(_config())
    assert {tuple(event.value for event in core.events) for core in cores} == (
        expected_trajectories
    )
    assert {
        core.query_selector.kind: type(thaw_json(core.expected_answer))
        for core in cores[:7]
        if core.query_selector.kind not in {"transition", "ordered_history"}
    } == {
        "current": str,
        "previous": str,
        "exact_version": str,
        "event_anchor": str,
        "logical_time_anchor": str,
    }


def test_family_f_selector_expected_answer_matrix_is_independent_of_generation_and_replay():
    from mub.vnext.generation.family_f import resolve_family_f_selector

    values = ("draft", "review", "approved", "published")
    anchors = ("event-0", "event-1", "event-2", "event-3")
    logical_times = ("00000010", "00000020", "00000030", "00000040")
    entries = tuple(
        VersionHistoryEntry(
            version_index=index,
            status="present",
            value=value,
            valid_from_event_id=anchors[index],
            valid_until_event_id=(anchors[index + 1] if index < 3 else None),
            logical_time=logical_times[index],
            source_event_ids=(anchors[index],),
        )
        for index, value in enumerate(values)
    )
    event_position = {event_id: index for index, event_id in enumerate(anchors)}
    event_times = dict(zip(anchors, logical_times))
    cases = (
        (CurrentSelector(), (3,), "published"),
        (PreviousSelector(), (2,), "approved"),
        (ExactVersionSelector(version_index=1), (1,), "review"),
        (EventAnchorSelector(event_id="event-2"), (2,), "approved"),
        (LogicalTimeAnchorSelector(logical_time="00000025"), (1,), "review"),
        (
            TransitionSelector(from_version_index=1, to_version_index=3),
            (1, 3),
            {"from": "review", "to": "published"},
        ),
        (
            OrderedHistorySelector(start_version_index=0, end_version_index=3),
            (0, 1, 2, 3),
            ["draft", "review", "approved", "published"],
        ),
    )

    for selector, expected_indices, expected_answer in cases:
        resolution = resolve_family_f_selector(
            selector,
            entries,
            event_position,
            event_times,
            logical_times[-1],
        )
        assert resolution.selected_indices == expected_indices
        assert resolution.answer == expected_answer


def test_family_f_compiler_emits_84_strict_v3_tasks_grouped_by_trajectory_and_four_surfaces():
    _, _, validate_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="4d3f9a6")

    assert compiled.profile_id == "family_f_diagnostic_micro_v1"
    assert len(compiled.cores) == 21
    assert len(compiled.tasks) == 84
    assert {task.schema_version for task in compiled.tasks} == {"3.0.0"}
    assert {task.task_family for task in compiled.tasks} == {TaskFamily.CURRENT_HISTORICAL_QUERY.value}
    assert Counter(task.metadata.split for task in compiled.tasks) == {Split.EVALUATION_ONLY: 84}

    by_core = defaultdict(list)
    by_trajectory = defaultdict(list)
    for task in compiled.tasks:
        by_core[task.metadata.split_key.semantic_core_id].append(task)
        by_trajectory[task.metadata.split_key.trajectory_id].append(task)
        validate_task(task)
        assert replay_task_v3(task).issues == ()
    assert all(len(tasks) == 4 for tasks in by_core.values())
    for tasks in by_core.values():
        assert {task.metadata.extra["surface_variant"] for task in tasks} == {0, 1, 2, 3}
        assert len({task.semantic_hash for task in tasks}) == 1
    assert all(len(tasks) == 28 for tasks in by_trajectory.values())
    for tasks in by_trajectory.values():
        assert len({task.metadata.split_key.version_group_id for task in tasks}) == 1
        assert {task.metadata.split for task in tasks} == {Split.EVALUATION_ONLY}


def test_family_f_all_selector_answers_and_evidence_reproduce_exactly_from_contiguous_horizon_active_ledgers():
    _, _, _, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="4d3f9a6")

    observed = Counter()
    for task in compiled.tasks:
        query = task.queries[0]
        evidence = task.gold_evidence[0]
        replay = replay_task_v3(task)
        resolution = resolve_query_v3(query, replay, task.events)
        observed[query.selector.kind] += 1
        assert len(task.version_history) == 1
        ledger = task.version_history[0]
        assert [entry.version_index for entry in ledger.entries] == list(range(len(ledger.entries)))
        assert len(ledger.entries) >= 4
        assert all(entry.status.value == "present" for entry in ledger.entries)
        assert all(entry.source_event_ids for entry in ledger.entries)
        assert all(entry.logical_time <= replay.horizon_logical_time for entry in ledger.entries)
        assert resolution.issues == ()
        assert resolution.answer == evidence.answer
        assert evaluate_evidence_v3(
            evidence, replay, query=query, events=task.events
        ).issues == ()
        assert set(resolution.selected_event_ids) == set(evidence.supporting_event_ids)
        assert set(resolution.selected_object_keys) == set(evidence.supporting_object_keys)
    assert observed == Counter({kind: 12 for kind in SELECTOR_KINDS})


def test_family_f_visible_surfaces_expose_every_typed_anchor_and_history_order():
    _, _, validate_task, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="4d3f9a6")

    for task in compiled.tasks:
        query = task.queries[0]
        text = query.text
        selector = query.selector
        assert f"selector={selector.kind}" in text
        if selector.kind != "current":
            assert "latest value" not in text.casefold()
            assert "query current" not in text.casefold()
            assert "current value" not in text.casefold()
        for entry in task.version_history[0].entries:
            event_id = entry.source_event_ids[0]
            event = next(item for item in task.events if item.event_id == event_id)
            assert f"version_index={entry.version_index}" in event.raw_text
            assert f"event_id={event_id}" in event.raw_text
            assert f"logical_time={entry.logical_time}" in event.raw_text
        if selector.kind == "current":
            assert f"version_index={task.version_history[0].entries[-1].version_index}" in text
        elif selector.kind == "previous":
            assert f"version_index={task.version_history[0].entries[-2].version_index}" in text
        elif selector.kind == "exact_version":
            assert f"version_index={selector.version_index}" in text
        elif selector.kind == "event_anchor":
            assert f"event_id={selector.event_id}" in text
        elif selector.kind == "logical_time_anchor":
            assert f"logical_time={selector.logical_time}" in text
            assert "at_or_before" in text
        elif selector.kind == "transition":
            assert f"from_version_index={selector.from_version_index}" in text
            assert f"to_version_index={selector.to_version_index}" in text
            assert "from_event_id=" in text and "to_event_id=" in text
        else:
            assert "history_order=oldest_to_newest" in text
            assert f"start_version_index={selector.start_version_index}" in text
            assert f"end_version_index={selector.end_version_index}" in text
        validate_task(task)

    task = compiled.tasks[0]
    payload = task.model_dump(mode="python")
    payload["queries"][0]["text"] = "What is the answer?"
    hidden = MemUpdateTaskV3.model_validate(payload)
    with pytest.raises(ValueError, match="visible selector anchors"):
        validate_task(hidden)


@pytest.mark.parametrize(
    ("label", "contradictory_value"),
    (
        ("version_index", "99"),
        ("event_id", "contradictory-event"),
        ("logical_time", "99999999"),
    ),
)
def test_family_f_generic_validator_rejects_standalone_contradictory_anchor_blocks(
    label,
    contradictory_value,
):
    _, _, validate_task, compile_micro = _api()
    task = compile_micro(_config(), code_revision="4d3f9a6").tasks[0]
    payload = task.model_dump(mode="python")
    entry = payload["version_history"][0]["entries"][0]
    suffix = (
        " ["
        f"version_index={entry['version_index']}; "
        f"event_id={entry['source_event_ids'][0]}; "
        f"logical_time={entry['logical_time']}"
        "]"
    )
    raw_text = payload["events"][0]["raw_text"]
    assert raw_text.endswith(suffix)
    payload["events"][0]["raw_text"] = (
        raw_text[: -len(suffix)]
        + f" [{label}={contradictory_value}]"
        + suffix
    )
    contradictory = MemUpdateTaskV3.model_validate(payload)

    with pytest.raises(ValueError, match="visible version ledger anchors"):
        validate_task(contradictory)


@pytest.mark.parametrize(
    ("label", "contradictory_value"),
    (
        ("version_index", "99"),
        ("event_id", "contradictory-event"),
        ("logical_time", "99999999"),
    ),
)
def test_family_f_generic_validator_rejects_duplicate_contradictory_event_anchor_labels(
    label,
    contradictory_value,
):
    _, _, validate_task, compile_micro = _api()
    task = compile_micro(_config(), code_revision="4d3f9a6").tasks[0]
    payload = task.model_dump(mode="python")
    entry = payload["version_history"][0]["entries"][0]
    anchor = {
        "version_index": str(entry["version_index"]),
        "event_id": entry["source_event_ids"][0],
        "logical_time": entry["logical_time"],
    }
    anchor[label] = contradictory_value
    payload["events"][0]["raw_text"] += (
        " ["
        f"version_index={anchor['version_index']}; "
        f"event_id={anchor['event_id']}; "
        f"logical_time={anchor['logical_time']}"
        "]"
    )
    contradictory = MemUpdateTaskV3.model_validate(payload)

    with pytest.raises(ValueError, match="visible version ledger anchors"):
        validate_task(contradictory)


def test_family_f_validator_rejects_selector_version_order_and_coordinated_metadata_corruption():
    generate, _, _, _ = _api()
    validate_core, _ = _micro_validators()
    cores = generate(_config())
    by_kind = {core.query_selector.kind: core for core in cores[:7]}

    def wrong_exact_version(payload):
        payload["query_selector"]["version_index"] = 2
        payload["expected_answer"] = payload["events"][2]["value"]
        payload["stratification"]["requested_version_distance"] = 1

    with pytest.raises(ValueError, match="canonical|selector"):
        validate_core(_mutate_core(by_kind["exact_version"], wrong_exact_version))

    def reordered_history(payload):
        payload["expected_answer"] = list(reversed(payload["expected_answer"]))

    with pytest.raises(ValueError, match="answer|history"):
        validate_core(_mutate_core(by_kind["ordered_history"], reordered_history))

    def current_history_confusion(payload):
        payload["query_selector"] = {
            "kind": "ordered_history",
            "start_version_index": 0,
            "end_version_index": 3,
        }
        payload["query_type"] = QueryType.HISTORICAL_STATE
        payload["expected_answer"] = [event["value"] for event in payload["events"]]
        payload["profile"]["query_type"] = "historical_state"
        payload["stratification"]["query_type"] = "historical_state"
        payload["stratification"]["requested_version_distance"] = 3

    with pytest.raises(ValueError, match="canonical|selector"):
        validate_core(_mutate_core(by_kind["current"], current_history_confusion))


def test_family_f_task_validator_rejects_coordinated_future_timestamp_and_wrong_transition_endpoints():
    _, _, validate_task, compile_micro = _api()
    _, validate_micro_task = _micro_validators()
    tasks = compile_micro(_config(), code_revision="4d3f9a6").tasks

    logical = next(task for task in tasks if task.queries[0].selector.kind == "logical_time_anchor")
    logical_payload = logical.model_dump(mode="python")
    logical_entries = logical_payload["version_history"][0]["entries"]
    current = logical_entries[-1]
    logical_payload["queries"][0]["selector"]["logical_time"] = "99999999"
    logical_payload["gold_evidence"][0]["answer"] = current["value"]
    logical_payload["gold_evidence"][0]["supporting_event_ids"] = current["source_event_ids"]
    logical_payload["gold_evidence"][0]["derivation_steps"][0]["supporting_event_ids"] = current["source_event_ids"]
    logical_payload["metadata"]["extra"]["stratification"]["requested_version_distance"] = 0
    coordinated_future = MemUpdateTaskV3.model_validate(logical_payload)
    assert replay_task_v3(coordinated_future).issues == ()
    with pytest.raises(ValueError, match="horizon|timestamp|canonical"):
        validate_task(coordinated_future)

    transition = next(task for task in tasks if task.queries[0].selector.kind == "transition")
    transition_payload = transition.model_dump(mode="python")
    transition_entries = transition_payload["version_history"][0]["entries"]
    selected = (transition_entries[0], transition_entries[2])
    transition_payload["queries"][0]["selector"].update({
        "from_version_index": 0,
        "to_version_index": 2,
    })
    transition_payload["gold_evidence"][0]["answer"] = {
        "from": selected[0]["value"],
        "to": selected[1]["value"],
    }
    support = [entry["source_event_ids"][0] for entry in selected]
    transition_payload["gold_evidence"][0]["supporting_event_ids"] = support
    for step, entry in zip(
        transition_payload["gold_evidence"][0]["derivation_steps"][:2], selected
    ):
        step["supporting_event_ids"] = entry["source_event_ids"]
    transition_payload["gold_evidence"][0]["derivation_steps"][-1]["supporting_event_ids"] = support
    text = transition_payload["queries"][0]["text"]
    original = (transition_entries[1], transition_entries[3])
    replacements = {
        "from_version_index=1": "from_version_index=0",
        "to_version_index=3": "to_version_index=2",
        f"from_event_id={original[0]['source_event_ids'][0]}": f"from_event_id={selected[0]['source_event_ids'][0]}",
        f"to_event_id={original[1]['source_event_ids'][0]}": f"to_event_id={selected[1]['source_event_ids'][0]}",
        f"from_logical_time={original[0]['logical_time']}": f"from_logical_time={selected[0]['logical_time']}",
        f"to_logical_time={original[1]['logical_time']}": f"to_logical_time={selected[1]['logical_time']}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    transition_payload["queries"][0]["text"] = text
    coordinated_transition = MemUpdateTaskV3.model_validate(transition_payload)
    assert replay_task_v3(coordinated_transition).issues == ()
    with pytest.raises(ValueError, match="transition|canonical"):
        validate_micro_task(coordinated_transition)


def test_family_f_task_validator_rejects_coordinated_current_to_history_relabeling():
    _, _, _, compile_micro = _api()
    _, validate_task = _micro_validators()
    current = next(
        task
        for task in compile_micro(_config(), code_revision="4d3f9a6").tasks
        if task.queries[0].selector.kind == "current"
    )
    payload = current.model_dump(mode="python")
    entries = payload["version_history"][0]["entries"]
    old_entry, new_entry = entries[-1], entries[-2]
    payload["queries"][0]["selector"] = {"kind": "previous"}
    payload["queries"][0]["query_type"] = QueryTypeV3.PREVIOUS
    text = payload["queries"][0]["text"].replace("selector=current", "selector=previous")
    text = text.replace(
        f"version_index={old_entry['version_index']}",
        f"version_index={new_entry['version_index']}",
    ).replace(
        f"event_id={old_entry['source_event_ids'][0]}",
        f"event_id={new_entry['source_event_ids'][0]}",
    ).replace(
        f"logical_time={old_entry['logical_time']}",
        f"logical_time={new_entry['logical_time']}",
    )
    payload["queries"][0]["text"] = text
    evidence = payload["gold_evidence"][0]
    evidence["answer"] = new_entry["value"]
    evidence["supporting_event_ids"] = new_entry["source_event_ids"]
    evidence["derivation_steps"][0]["operation"] = "read_version"
    evidence["derivation_steps"][0]["supporting_event_ids"] = new_entry["source_event_ids"]
    payload["metadata"]["extra"]["stratification"] = {
        "query_type": QueryType.HISTORICAL_STATE.value,
        "requested_version_distance": 1,
    }
    relabeled = MemUpdateTaskV3.model_validate(payload)
    assert replay_task_v3(relabeled).issues == ()
    with pytest.raises(ValueError, match="semantic core|canonical|binding"):
        validate_task(relabeled)


def test_family_f_generic_validator_is_reusable_while_micro_gate_keeps_exact_profile():
    _, _, validate_task, compile_micro = _api()
    _, validate_micro_task = _micro_validators()
    task = compile_micro(_config(), code_revision="4d3f9a6").tasks[0]
    payload = task.model_dump(mode="python")
    payload["metadata"]["split"] = Split.TEST
    reusable = MemUpdateTaskV3.model_validate(payload)

    validate_task(reusable)
    with pytest.raises(ValueError, match="evaluation_only|micro-pilot"):
        validate_micro_task(reusable)


def test_family_f_shared_selector_resolution_cannot_drift_from_v3_replay():
    from mub.vnext.contracts.v3.task import resolve_selector_version_indices_v3

    _, _, _, compile_micro = _api()
    for task in compile_micro(_config(), code_revision="4d3f9a6").tasks:
        query = task.queries[0]
        ledger = task.version_history[0]
        event_positions = {event.event_id: event.sequence_index for event in task.events}
        event_times = {
            event.event_id: event.timestamp
            for event in task.events
            if event.timestamp is not None
        }
        indices = resolve_selector_version_indices_v3(
            query.selector,
            ledger.entries,
            event_positions,
            event_times,
            replay_task_v3(task).horizon_logical_time,
        )
        replay = replay_task_v3(task)
        resolution = resolve_query_v3(query, replay, task.events)
        assert indices == tuple(version.version_index for version in resolution.selected_versions)


def test_family_f_v2_surface_staging_preserves_declared_query_semantics():
    from mub.vnext.generation.core import GenerationContext
    from mub.vnext.generation.core_catalogs import CORE_SURFACE_CATALOG_V1
    from mub.vnext.generation.render import render_core_with_catalog

    generate, _, _, _ = _api()
    core = next(core for core in generate(_config()) if core.query_selector.kind == "previous")
    rendered = render_core_with_catalog(
        core,
        split=Split.EVALUATION_ONLY,
        surface_variant=0,
        context=GenerationContext(config=_config(), code_revision="4d3f9a6"),
        surface_catalog=CORE_SURFACE_CATALOG_V1,
    )
    assert rendered.queries[0].query_type is QueryType.HISTORICAL_STATE
    assert rendered.gold.gold_answers[rendered.queries[0].query_id] == core.expected_answer
    assert rendered.metadata.resolved_profile["query_type"] == QueryType.HISTORICAL_STATE.value


def test_family_f_micro_gate_rejects_complete_task_projection_and_contradictory_anchor_tampering():
    _, _, validate_task, compile_micro = _api()
    _, validate_micro_task = _micro_validators()
    task = compile_micro(_config(), code_revision="4d3f9a6").tasks[0]

    contradictory_payload = task.model_dump(mode="python")
    contradictory_payload["queries"][0]["text"] += (
        " [selector=previous; version_index=2]"
    )
    contradictory = MemUpdateTaskV3.model_validate(contradictory_payload)
    with pytest.raises(ValueError, match="visible selector anchors|contradictory"):
        validate_task(contradictory)

    difficulty_payload = task.model_dump(mode="python")
    difficulty_payload["difficulty"] = "hard"
    difficulty_payload["metadata"]["profile_name"] = "hard"
    difficulty = MemUpdateTaskV3.model_validate(difficulty_payload)
    with pytest.raises(ValueError, match="core projection|difficulty|canonical"):
        validate_micro_task(difficulty)

    timestamp_payload = task.model_dump(mode="python")
    timestamp_payload["events"][0]["timestamp"] = "00000011"
    timestamp = MemUpdateTaskV3.model_validate(timestamp_payload)
    with pytest.raises(ValueError, match="core projection|timestamp|canonical"):
        validate_micro_task(timestamp)

    surface_payload = task.model_dump(mode="python")
    surface_payload["metadata"]["extra"]["surface_template"] = "tampered"
    surface = MemUpdateTaskV3.model_validate(surface_payload)
    with pytest.raises(ValueError, match="surface|core projection|canonical"):
        validate_micro_task(surface)


def _authenticated_score_context(task, run, caps):
    from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
    from mub.vnext.contracts.v3.score import ScorerConfigV3
    from mub.vnext.io import sha256_model
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    task_artifact = {"path": "family-f-tasks.jsonl", "sha256": "b" * 64, "media_type": "application/jsonl", "record_count": 1}
    run_artifact = {"path": "family-f-runs.jsonl", "sha256": "c" * 64, "media_type": "application/jsonl", "record_count": 1}
    task_manifest = TaskManifestV3(
        data_release_id="family-f-diagnostic", split_policy_version="diagnostic-no-release-split",
        compiler_versions={"core_family_f_micro": "3"}, source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(), split_counts={"evaluation_only": 1},
        family_difficulty_counts={f"F.{task.difficulty.value}": 1},
        semantic_core_counts={task.metadata.split_key.semantic_core_id: 1},
        task_file_paths_and_hashes=(task_artifact,), task_record_hashes={task.task_id: sha256_model(task)},
        leakage_check_summary={}, human_audit_artifacts=(), created_at="2026-08-05T00:00:00Z",
        code_revision="4d3f9a6",
    )
    task_manifest_hash = sha256_model(task_manifest)
    config = ScorerConfigV3()
    run_manifest = RunManifestV3(
        run_id=run.run_id, timestamp="2026-08-05T00:00:00Z", code_revision="4d3f9a6",
        dirty_state=False,
        task_manifest={"path": "task-manifest.json", "sha256": task_manifest_hash, "media_type": "application/json"},
        scorer_config=config, adapter_info=ORACLE_INFO, adapter_capabilities=caps,
        capability_verification_artifact={"path": "caps.json", "sha256": "d" * 64, "media_type": "application/json"},
        model_name=None, provider=None, model_revision=None, prompt_config={}, decoding_config={},
        seed_information={}, action_parser_version="1", answer_parser_version="1",
        memory_entry_extractor_version="1", object_value_extractor_config_hash="a" * 64,
        redaction_policy_version="1", environment_summary={}, package_summary={}, expected_task_count=1,
        completed_task_count=1, failed_task_count=0, not_supported_task_count=0,
        raw_provider_response_artifacts=(), raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(run_artifact,), run_record_hashes={run.task_id: sha256_model(run)},
        score_artifacts=(), native_vs_extracted_field_summary={},
    )
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task, run=run, task_manifest=task_manifest, run_manifest=run_manifest,
        task_artifact=task_artifact, run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_hash,
        authenticated_run_manifest_sha256=sha256_model(run_manifest),
    )


def _oracle_run(task, *, answer_override=None, omit_historical_support=False):
    from mub.vnext.contracts.v3.runtime import (
        AnswerPredictionV3,
        MemoryEntryRecordV3,
        MemorySnapshotV3,
        ParsedManagerActionV3,
        ParserExtractorProvenanceV3,
        RetrievalTraceV3,
        TaskRunRecordV3,
    )

    replay = replay_task_v3(task)
    all_entries = tuple(
        MemoryEntryRecordV3(
            entry_id=f"v-{version.version_index}", content=str(version.value),
            object_key_candidate=version.object_key, value_candidate=version.value,
            version_index=version.version_index, source_event_ids=version.source_event_ids,
        )
        for ledger in replay.ledgers
        for version in ledger.versions
    )
    query = task.queries[0]
    evidence = task.gold_evidence[0]
    resolution = resolve_query_v3(query, replay, task.events)
    selected_indices = {version.version_index for version in resolution.selected_versions}
    retrieved = tuple(entry for entry in all_entries if entry.version_index in selected_indices)
    if omit_historical_support and query.query_type is not QueryTypeV3.CURRENT:
        retrieved = ()
    parsed_actions = tuple(
        ParsedManagerActionV3(
            action_id=action.action_id, event_id=action.event_id, operation=action.operation,
            observed_scope=action.scope, target_object_keys=action.target_object_keys,
            value=action.value, format_valid=True, execution_status="executed", fallback_used=False,
            raw_output="oracle",
        )
        for action in task.actions
    )
    prediction = AnswerPredictionV3(
        query_id=query.query_id, raw_output="oracle",
        parsed_answer=evidence.answer if answer_override is None else answer_override,
        cited_event_ids=evidence.supporting_event_ids, cited_object_keys=evidence.supporting_object_keys,
        cited_derivation_step_ids=(evidence.final_derivation_step_id,), format_valid=True,
    )
    current_entries = tuple(entry for entry in all_entries if any(
        entry.object_key_candidate.canonical_id == version.object_key.canonical_id
        and entry.version_index == version.version_index
        for version in replay.current_state.values()
    ))
    state = {key: version.value for key, version in replay.current_state.items()}
    return TaskRunRecordV3(
        task_id=task.task_id, adapter_id="family-f-oracle", run_id=f"run-{task.task_id}-{omit_historical_support}-{answer_override is not None}",
        parsed_actions=parsed_actions,
        memory_snapshots=(MemorySnapshotV3(
            after_event_id=task.events[-1].event_id, entries=current_entries,
            state_by_object=state, store_size=len(current_entries),
        ),),
        retrieval_traces=(RetrievalTraceV3(
            query_id=query.query_id, retrieved_entries=retrieved,
            ranks=tuple(range(1, len(retrieved) + 1)),
            gold_in_context=bool(retrieved), stale_in_context=False, distractor_in_context=False,
        ),),
        answer_predictions=(prediction,),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1", answer_parser_version="1", memory_entry_extractor_version="1",
            object_value_extractor_config_hash="a" * 64, redaction_policy_version="1",
        ),
        completion_status="completed",
    )


def test_family_f_reference_oracle_scores_exact_per_selector_applicability_over_all_84_tasks():
    from mub.vnext.scoring.registry_v3 import CORE_METRIC_REGISTRY_V3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    _, _, _, compile_micro = _api()
    compiled = compile_micro(_config(), code_revision="4d3f9a6")
    registered_principal = {path for path, descriptor in CORE_METRIC_REGISTRY_V3.items() if descriptor.principal}
    family_f_principal = COMMON_PRINCIPAL_PATHS | HISTORICAL_PRINCIPAL_PATHS
    assert family_f_principal <= registered_principal

    for task in compiled.tasks:
        query_type = task.queries[0].query_type
        expected_paths = EXPECTED_PRINCIPAL_BY_QUERY[query_type]
        run = _oracle_run(task)
        score = score_task_v3(task, run, _authenticated_score_context(task, run, FULL_CAPABILITIES))
        observed_paths = {path for path in family_f_principal if _metric_value(score, path) is not None}
        assert observed_paths == expected_paths, (task.task_id, observed_paths)
        for path in family_f_principal:
            value = _metric_value(score, path)
            if path in expected_paths:
                assert value == 1.0, (task.task_id, path, value)
            else:
                assert value is None
                assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE
        for path in registered_principal - family_f_principal:
            assert _metric_value(score, path) is None
            assert score.supported_metric_fields[path].reason is SupportReason.NOT_APPLICABLE


def test_family_f_corrupted_controls_detect_current_confusion_wrong_order_and_missing_support():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    _, _, _, compile_micro = _api()
    tasks = compile_micro(_config(), code_revision="4d3f9a6").tasks
    historical = next(task for task in tasks if task.queries[0].selector.kind == "previous")
    current_value = replay_task_v3(historical).current_state[historical.target_objects[0].canonical_id].value
    confused_run = _oracle_run(historical, answer_override=current_value)
    confused = score_task_v3(
        historical, confused_run,
        _authenticated_score_context(historical, confused_run, FULL_CAPABILITIES),
    )
    assert confused.historical_scores.previous_state_accuracy == 0.0
    assert confused.historical_scores.version_confusion_rate == 1.0

    history = next(task for task in tasks if task.queries[0].selector.kind == "ordered_history")
    wrong_order = list(reversed(history.gold_evidence[0].answer))
    wrong_order_run = _oracle_run(history, answer_override=wrong_order)
    wrong_order_score = score_task_v3(
        history, wrong_order_run,
        _authenticated_score_context(history, wrong_order_run, FULL_CAPABILITIES),
    )
    assert wrong_order_score.historical_scores.ordered_history_accuracy == 0.0

    missing_run = _oracle_run(history, omit_historical_support=True)
    missing_score = score_task_v3(
        history, missing_run,
        _authenticated_score_context(history, missing_run, FULL_CAPABILITIES),
    )
    assert missing_score.historical_scores.historical_support_recall == 0.0


def test_family_f_withheld_historical_capabilities_are_typed_nulls_never_fabricated_zeroes():
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    _, _, _, compile_micro = _api()
    representatives = {}
    for task in compile_micro(_config(), code_revision="4d3f9a6").tasks:
        representatives.setdefault(task.queries[0].query_type, task)
    assert set(representatives) == set(EXPECTED_PRINCIPAL_BY_QUERY)

    for query_type, task in representatives.items():
        run = _oracle_run(task)
        score = score_task_v3(
            task, run,
            _authenticated_score_context(task, run, AdapterCapabilitiesV3()),
        )
        expected = EXPECTED_PRINCIPAL_BY_QUERY[query_type] & HISTORICAL_PRINCIPAL_PATHS
        for path in HISTORICAL_PRINCIPAL_PATHS:
            assert _metric_value(score, path) is None
            reason = score.supported_metric_fields[path].reason
            assert reason is (SupportReason.NOT_SUPPORTED if path in expected else SupportReason.NOT_APPLICABLE)
