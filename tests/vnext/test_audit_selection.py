from __future__ import annotations

import inspect
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import mub.vnext.audit.select as selector_module
from mub.vnext.audit import (
    AuditFamilySelectionReport,
    AuditSelection,
    AuditSelectionIssue,
    AuditSelectionResult,
    FAMILY_CONDITION_POLICY,
    audit_selection_id,
    select_pilot_audit_sample,
)
from mub.vnext.contracts import Difficulty, MemUpdateTask, Split, TaskFamily, TaskManifest
from mub.vnext.generation import (
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "audit-selection-test-revision"
PILOT_FAMILIES = (
    TaskFamily.REPEATED_SAME_SLOT,
    TaskFamily.INTERLEAVED_MULTI_SLOT,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
    TaskFamily.NOOP_WRITE_DISCIPLINE,
)


@pytest.fixture(scope="session")
def canonical_release() -> tuple[tuple[MemUpdateTask, ...], TaskManifest]:
    config = load_pilot_config(CONFIG_PATH)
    compiled = compile_pilot_tasks(config, code_revision=REVISION)
    bundle = build_pilot_artifact_bundle(compiled, config)
    return compiled.tasks, bundle.task_manifest


@pytest.fixture(scope="session")
def canonical_selection(canonical_release) -> AuditSelectionResult:
    tasks, manifest = canonical_release
    return select_pilot_audit_sample(tasks, manifest)


def _selected_tokens(result: AuditSelectionResult) -> dict[TaskFamily, set[str]]:
    covered: dict[TaskFamily, set[str]] = defaultdict(set)
    for selection in result.selections:
        covered[selection.family].update(selection.covered_conditions)
    return covered


def _replace_task(
    tasks: tuple[MemUpdateTask, ...], index: int, replacement: MemUpdateTask
) -> tuple[MemUpdateTask, ...]:
    changed = list(tasks)
    changed[index] = replacement
    return tuple(changed)


def _task_with_payload_change(task: MemUpdateTask, change) -> MemUpdateTask:
    payload = task.model_dump(mode="python")
    change(payload)
    return MemUpdateTask.model_validate(payload)


def test_canonical_release_selects_exact_balanced_valid_sample(
    canonical_release, canonical_selection
) -> None:
    tasks, _ = canonical_release

    result = canonical_selection

    assert result.valid
    assert result.issues == ()
    assert result.impossible_reasons == ()
    assert result.uncovered_required_conditions == ()
    assert len(result.selections) == 96
    assert Counter(item.family for item in result.selections) == {
        family: 24 for family in PILOT_FAMILIES
    }
    assert len({item.task_id for item in result.selections}) == 96
    assert len({item.audit_id for item in result.selections}) == 96
    assert len({item.task_id for item in result.selections}) == len(result.selections)
    selected_by_id = {task.task_id: task for task in tasks}
    assert all(item.task_id in selected_by_id for item in result.selections)
    for family in PILOT_FAMILIES:
        family_items = [item for item in result.selections if item.family is family]
        assert {item.split for item in family_items} == {Split.TRAIN, Split.DEV, Split.TEST}
        assert {item.difficulty for item in family_items} == {
            Difficulty.EASY,
            Difficulty.MEDIUM,
            Difficulty.HARD,
        }
        assert {
            selected_by_id[item.task_id].metadata.extra["surface_variant"]
            for item in family_items
        } == {0, 1, 2}
        assert len(
            {
                selected_by_id[item.task_id].metadata.split_key.semantic_core_id
                for item in family_items
            }
        ) == 24


def test_reviewed_condition_policy_and_all_values_are_covered(
    canonical_selection,
) -> None:
    result = canonical_selection
    covered = _selected_tokens(result)

    assert FAMILY_CONDITION_POLICY == {
        TaskFamily.REPEATED_SAME_SLOT: (
            ("profile", "update_depth"),
            ("stratification", "same_name_distractor_count"),
            ("stratification", "same_entity_other_attribute_count"),
            ("stratification", "noop_count"),
        ),
        TaskFamily.INTERLEAVED_MULTI_SLOT: (
            ("profile", "update_depth"),
            ("stratification", "active_object_count"),
            ("stratification", "cross_slot_distractor_density"),
            ("stratification", "interleaving_pattern"),
        ),
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: (
            ("stratification", "entity_condition"),
            ("stratification", "attribute_condition"),
        ),
        TaskFamily.NOOP_WRITE_DISCIPLINE: (
            ("stratification", "configured_noop_density"),
            ("stratification", "trap_type"),
        ),
    }
    universal = {
        'split="train"',
        'split="dev"',
        'split="test"',
        'difficulty="easy"',
        'difficulty="medium"',
        'difficulty="hard"',
        "surface_variant=0",
        "surface_variant=1",
        "surface_variant=2",
    }
    expected_family_tokens = {
        TaskFamily.REPEATED_SAME_SLOT: {
            "profile.update_depth=1",
            "profile.update_depth=4",
            "profile.update_depth=16",
            "stratification.same_name_distractor_count=0",
            "stratification.same_name_distractor_count=2",
            "stratification.same_name_distractor_count=4",
            "stratification.same_entity_other_attribute_count=0",
            "stratification.same_entity_other_attribute_count=1",
            "stratification.same_entity_other_attribute_count=2",
            "stratification.noop_count=0",
            "stratification.noop_count=2",
            "stratification.noop_count=4",
        },
        TaskFamily.INTERLEAVED_MULTI_SLOT: {
            "profile.update_depth=1",
            "profile.update_depth=4",
            "profile.update_depth=16",
            "stratification.active_object_count=2",
            "stratification.active_object_count=4",
            "stratification.active_object_count=8",
            "stratification.cross_slot_distractor_density=0.0",
            "stratification.cross_slot_distractor_density=0.25",
            "stratification.cross_slot_distractor_density=0.5",
            'stratification.interleaving_pattern="round_robin"',
            'stratification.interleaving_pattern="burst"',
            'stratification.interleaving_pattern="adversarial_adjacent"',
        },
        TaskFamily.ENTITY_ATTRIBUTE_GROUNDING: {
            'stratification.entity_condition="distinct"',
            'stratification.entity_condition="same_name"',
            'stratification.entity_condition="alias"',
            'stratification.entity_condition="namespace_collision"',
            'stratification.attribute_condition="exact"',
            'stratification.attribute_condition="paraphrase"',
            'stratification.attribute_condition="near_name"',
        },
        TaskFamily.NOOP_WRITE_DISCIPLINE: {
            "stratification.configured_noop_density=0.25",
            "stratification.configured_noop_density=0.5",
            "stratification.configured_noop_density=0.75",
            'stratification.trap_type="semantic_near_miss"',
            'stratification.trap_type="duplicate_current"',
            'stratification.trap_type="other_entity_correction"',
            'stratification.trap_type="other_attribute_correction"',
        },
    }
    for family in PILOT_FAMILIES:
        assert universal | expected_family_tokens[family] <= covered[family]
        family_report = next(item for item in result.family_reports if item.family is family)
        assert set(family_report.required_conditions) == universal | expected_family_tokens[family]
        assert family_report.uncovered_required_conditions == ()
        assert family_report.impossible_reasons == ()


def test_selection_is_permutation_invariant_and_canonically_stable(
    canonical_release, canonical_selection
) -> None:
    tasks, manifest = canonical_release
    shuffled = list(tasks)
    random.Random(404).shuffle(shuffled)

    original = canonical_selection
    reversed_result = select_pilot_audit_sample(reversed(tasks), manifest)
    shuffled_result = select_pilot_audit_sample(shuffled, manifest)

    assert set(inspect.signature(select_pilot_audit_sample).parameters) == {
        "tasks",
        "manifest",
    }
    assert original == reversed_result == shuffled_result
    assert canonical_json_bytes(original) == canonical_json_bytes(reversed_result)
    assert canonical_json_bytes(original) == canonical_json_bytes(shuffled_result)


def test_audit_id_binds_every_selection_contract_field(canonical_selection) -> None:
    selected = canonical_selection.selections[0]
    base = {
        "task_id": selected.task_id,
        "family": selected.family,
        "difficulty": selected.difficulty,
        "split": selected.split,
        "covered_conditions": selected.covered_conditions,
        "selection_reason": selected.selection_reason,
        "selection_algorithm": selector_module.SELECTION_ALGORITHM,
        "selection_version": selector_module.SELECTION_VERSION,
    }

    identifiers = {audit_selection_id(**base)}
    changes = (
        {"task_id": "task_changed"},
        {"family": TaskFamily.INTERLEAVED_MULTI_SLOT},
        {"difficulty": Difficulty.CHALLENGE},
        {"split": Split.EVALUATION_ONLY},
        {"covered_conditions": (*selected.covered_conditions, "changed=true")},
        {"selection_reason": "changed_reason"},
        {"selection_algorithm": "changed_algorithm"},
        {"selection_version": "changed_version"},
    )
    for change in changes:
        identifiers.add(audit_selection_id(**(base | change)))

    assert len(identifiers) == len(changes) + 1
    assert audit_selection_id(**base) == selected.audit_id


def test_invalid_release_conditions_return_typed_nonvalid_result(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    corruptions = (
        lambda payload: payload["metadata"]["extra"]["stratification"].pop(
            "same_name_distractor_count", None
        ),
        lambda payload: payload.update(
            {"task_family": TaskFamily.NOOP_WRITE_DISCIPLINE.value}
        ),
        lambda payload: payload["metadata"]["extra"].pop("surface_variant", None),
    )
    changed = list(tasks)
    for index, corrupt in enumerate(corruptions):
        changed[index] = _task_with_payload_change(tasks[index], corrupt)

    result = select_pilot_audit_sample(tuple(changed), manifest)

    assert isinstance(result, AuditSelectionResult)
    assert not result.valid
    assert result.selections == ()
    assert result.issues


def test_invalid_manifest_and_model_construct_inputs_fail_closed(canonical_release) -> None:
    tasks, manifest = canonical_release
    invalid_manifest = manifest.validated_replace(data_release_id="not-the-pilot")
    hostile_manifest = TaskManifest.model_construct(data_release_id="pilot")
    hostile_task = MemUpdateTask.model_construct(task_id="forged")

    duplicate_result = select_pilot_audit_sample(
        _replace_task(tasks, 1, tasks[0]), manifest
    )
    invalid_result = select_pilot_audit_sample(tasks, invalid_manifest)
    hostile_manifest_result = select_pilot_audit_sample(tasks, hostile_manifest)
    hostile_task_result = select_pilot_audit_sample(
        _replace_task(tasks, 0, hostile_task), manifest
    )

    assert not duplicate_result.valid
    assert any("duplicate" in issue.code for issue in duplicate_result.issues)
    assert not invalid_result.valid
    assert any("manifest" in issue.code for issue in invalid_result.issues)
    assert not hostile_manifest_result.valid
    assert hostile_manifest_result.issues
    assert not hostile_task_result.valid
    assert hostile_task_result.issues
    forged_result = AuditSelectionResult.model_construct(
        selection_algorithm=selector_module.SELECTION_ALGORITHM,
        selection_version=selector_module.SELECTION_VERSION,
        selections=(),
        family_reports=(),
        uncovered_required_conditions=(),
        impossible_reasons=(),
        issues=(),
    )
    assert forged_result.valid is False


def test_result_validated_replace_excludes_computed_valid_field() -> None:
    result = AuditSelectionResult(
        selection_algorithm=selector_module.SELECTION_ALGORITHM,
        selection_version=selector_module.SELECTION_VERSION,
    )
    issue = AuditSelectionIssue(code="changed", message="changed", path="selection")

    replaced = result.validated_replace(issues=(issue,))

    assert replaced.issues == (issue,)
    assert replaced.valid is False


def test_hostile_iterables_are_bounded_and_reported(canonical_release) -> None:
    tasks, manifest = canonical_release
    yielded = 0

    def endless_tasks():
        nonlocal yielded
        while True:
            yielded += 1
            yield tasks[0]

    def exploding_tasks():
        yield tasks[0]
        raise RuntimeError("hostile iterator")

    oversized = select_pilot_audit_sample(endless_tasks(), manifest)
    exploding = select_pilot_audit_sample(exploding_tasks(), manifest)
    wrong = select_pilot_audit_sample("tasks.jsonl", manifest)
    raw_bytes = select_pilot_audit_sample(b"{}\n", manifest)

    assert not oversized.valid
    assert yielded == 1441
    assert len(oversized.issues) <= 128
    assert any(issue.code == "audit_selection_input_size_limit" for issue in oversized.issues)
    assert not exploding.valid
    assert any(
        issue.code == "audit_selection_malformed_tasks_iterable"
        for issue in exploding.issues
    )
    assert not wrong.valid
    assert not raw_bytes.valid


def test_impossible_cover_is_not_silently_dropped(canonical_release, monkeypatch) -> None:
    tasks, manifest = canonical_release
    monkeypatch.setattr(selector_module, "_TASKS_PER_FAMILY", 1)

    result = select_pilot_audit_sample(tasks, manifest)

    assert not result.valid
    assert result.uncovered_required_conditions
    assert result.impossible_reasons
    assert any(issue.code == "audit_selection_impossible_cover" for issue in result.issues)


def test_set_cover_finds_a_feasible_cover_after_greedy_dead_end(monkeypatch) -> None:
    family = TaskFamily.REPEATED_SAME_SLOT

    def candidate(task_id: str, *conditions: str):
        return selector_module._Candidate(
            task_id=task_id,
            family=family,
            difficulty=Difficulty.EASY,
            split=Split.TRAIN,
            semantic_core_id=f"core-{task_id}",
            surface_variant=0,
            conditions=tuple(sorted(conditions)),
        )

    candidates = (
        candidate("a", "one", "two"),
        candidate("b", "one", "three"),
        candidate("c", "two", "four"),
    )
    monkeypatch.setattr(selector_module, "_TASKS_PER_FAMILY", 2)

    selections, report, issues = selector_module._select_family(family, candidates)

    assert issues == ()
    assert report.uncovered_required_conditions == ()
    assert {item.task_id for item in selections} == {"b", "c"}


def test_malformed_diagnostics_do_not_depend_on_input_indices(canonical_release) -> None:
    tasks, manifest = canonical_release
    malformed = MemUpdateTask.model_construct(task_id="forged")

    first = select_pilot_audit_sample((malformed, tasks[0]), manifest)
    second = select_pilot_audit_sample((tasks[0], malformed), manifest)

    assert first == second
    assert first.issues
    assert all("[" not in issue.path for issue in first.issues)


def test_valid_rejects_family_report_with_incomplete_condition_universe(
    canonical_selection,
) -> None:
    report = canonical_selection.family_reports[0]
    tampered = report.model_copy(
        update={"required_conditions": report.required_conditions[1:]}
    )
    forged = canonical_selection.validated_replace(
        family_reports=(tampered, *canonical_selection.family_reports[1:])
    )

    assert forged.valid is False


def test_typed_selection_result_round_trips_through_canonical_json() -> None:
    selection = AuditSelection(
        audit_id="audit_roundtrip",
        task_id="task_roundtrip",
        family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        split=Split.TRAIN,
        covered_conditions=("condition=true",),
        selection_reason="greedy_set_cover",
    )
    report = AuditFamilySelectionReport(
        family=TaskFamily.REPEATED_SAME_SLOT,
        required_conditions=("condition=true",),
        selected_task_ids=("task_roundtrip",),
    )
    issue = AuditSelectionIssue(code="typed", message="typed", path="selection")
    result = AuditSelectionResult(
        selection_algorithm=selector_module.SELECTION_ALGORITHM,
        selection_version=selector_module.SELECTION_VERSION,
        selections=(selection,),
        family_reports=(report,),
        issues=(issue,),
    )

    reconstructed = AuditSelectionResult.model_validate_json(
        canonical_json_bytes(result)
    )

    assert reconstructed == result
    assert reconstructed.valid is False
