from __future__ import annotations

import builtins
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import mub.vnext.validation.pilot as pilot_module
from mub.vnext.contracts import EventRole, Operation, Split, TaskFamily
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.generation import (
    GenerationContext,
    generate_family_d_cores,
    load_pilot_config,
    render_core,
)
from mub.vnext.validation.pilot import validate_family_d_task


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


def _codes(report):
    return {issue.code for issue in report.issues}


def _payload(task):
    return task.model_dump(mode="json")


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
    assert {"malformed_gold_actions", "malformed_resolved_profile"} <= _codes(
        paths_report
    )


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
