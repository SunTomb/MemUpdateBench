from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from mub.vnext.contracts import (
    AnswerDisposition,
    ArtifactRef,
    MemUpdateTask,
    Split,
    TaskFamily,
    TaskManifest,
)
from mub.vnext.generation import (
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.validation import validate_pilot_release


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "pilot-release-validation-test-revision"


@pytest.fixture(scope="module")
def canonical_release():
    config = load_pilot_config(CONFIG_PATH)
    compiled = compile_pilot_tasks(config, code_revision=REVISION)
    bundle = build_pilot_artifact_bundle(compiled, config)
    return compiled.tasks, bundle.task_manifest


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _replace_manifest(manifest: TaskManifest, **updates) -> TaskManifest:
    payload = manifest.model_dump(mode="python")
    payload.update(updates)
    return TaskManifest.model_validate(payload)


def _replace_task(tasks, index: int, task: MemUpdateTask):
    changed = list(tasks)
    changed[index] = task
    return tuple(changed)


def _task_index(tasks, *, family: TaskFamily, split: Split | None = None) -> int:
    return next(
        index
        for index, task in enumerate(tasks)
        if task.task_family == family.value
        and (split is None or task.metadata.split is split)
    )


def test_canonical_pilot_release_passes(canonical_release) -> None:
    tasks, manifest = canonical_release

    report = validate_pilot_release(tasks, manifest)

    assert report.valid
    assert report.issues == ()
    assert any(
        canonical.disposition is AnswerDisposition.ABSTAINED
        for task in tasks
        if task.task_family == TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value
        for canonical in task.gold.canonical_answers.values()
    )


def test_release_validation_is_independent_of_task_input_order(canonical_release) -> None:
    tasks, manifest = canonical_release
    shuffled = list(tasks)
    random.Random(367).shuffle(shuffled)

    assert validate_pilot_release(shuffled, manifest) == validate_pilot_release(
        tasks, manifest
    )


def test_release_rejects_total_task_count(canonical_release) -> None:
    tasks, manifest = canonical_release

    report = validate_pilot_release(tasks[:-1], manifest)

    assert "pilot_release_task_count_mismatch" in _codes(report)


def test_release_rejects_family_task_count(canonical_release) -> None:
    tasks, manifest = canonical_release
    index = _task_index(tasks, family=TaskFamily.REPEATED_SAME_SLOT)
    changed = tasks[index].model_copy(
        update={"task_family": TaskFamily.INTERLEAVED_MULTI_SLOT.value}
    )

    report = validate_pilot_release(_replace_task(tasks, index, changed), manifest)

    assert "pilot_release_family_task_count_mismatch" in _codes(report)


def test_release_rejects_split_task_count(canonical_release) -> None:
    tasks, manifest = canonical_release
    core_id = tasks[0].metadata.split_key.semantic_core_id
    changed = tuple(
        task.model_copy(
            update={
                "metadata": task.metadata.model_copy(update={"split": Split.DEV})
            }
        )
        if task.metadata.split_key.semantic_core_id == core_id
        else task
        for task in tasks
    )

    report = validate_pilot_release(changed, manifest)

    assert "pilot_release_split_task_count_mismatch" in _codes(report)
    assert "pilot_release_family_split_count_mismatch" in _codes(report)


def test_release_rejects_duplicate_and_missing_task_ids(canonical_release) -> None:
    tasks, manifest = canonical_release
    duplicate = tasks[1].model_copy(update={"task_id": tasks[0].task_id})
    duplicate_report = validate_pilot_release(
        _replace_task(tasks, 1, duplicate), manifest
    )
    raw = dict(tasks[0].__dict__)
    raw["task_id"] = ""
    missing = MemUpdateTask.model_construct(**raw)
    missing_report = validate_pilot_release(_replace_task(tasks, 0, missing), manifest)

    assert "pilot_release_duplicate_task_id" in _codes(duplicate_report)
    assert "pilot_release_missing_task_id" in _codes(missing_report)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    (
        ("sha256", "0" * 64, "pilot_release_task_file_hash_mismatch"),
        ("record_count", 1439, "pilot_release_task_file_count_mismatch"),
    ),
)
def test_release_rejects_changed_task_manifest_binding(
    canonical_release, field, replacement, expected_code
) -> None:
    tasks, manifest = canonical_release
    original = manifest.task_file_paths_and_hashes[0]
    payload = original.model_dump(mode="python")
    payload[field] = replacement
    changed_ref = ArtifactRef.model_validate(payload)
    changed_manifest = _replace_manifest(
        manifest, task_file_paths_and_hashes=(changed_ref,)
    )

    report = validate_pilot_release(tasks, changed_manifest)

    assert expected_code in _codes(report)


def test_release_rejects_noncanonical_generation_config_hash(canonical_release) -> None:
    tasks, manifest = canonical_release
    original = manifest.generation_configs_and_hashes[0]
    payload = original.model_dump(mode="python")
    payload["sha256"] = "0" * 64
    changed_ref = ArtifactRef.model_validate(payload)
    changed_manifest = _replace_manifest(
        manifest, generation_configs_and_hashes=(changed_ref,)
    )

    report = validate_pilot_release(tasks, changed_manifest)

    assert "pilot_release_generation_config_hash_mismatch" in _codes(report)


def test_release_rejects_semantic_core_split_leakage(canonical_release) -> None:
    tasks, manifest = canonical_release
    changed = tasks[0].model_copy(
        update={
            "metadata": tasks[0].metadata.model_copy(update={"split": Split.DEV})
        }
    )

    report = validate_pilot_release(_replace_task(tasks, 0, changed), manifest)

    assert "pilot_release_semantic_core_split_leakage" in _codes(report)


def test_release_rejects_source_group_split_leakage(canonical_release) -> None:
    tasks, manifest = canonical_release
    train_index = _task_index(
        tasks, family=TaskFamily.REPEATED_SAME_SLOT, split=Split.TRAIN
    )
    test_index = _task_index(
        tasks, family=TaskFamily.REPEATED_SAME_SLOT, split=Split.TEST
    )
    test_group = tasks[test_index].metadata.split_key.source_group_id
    split_key = tasks[train_index].metadata.split_key.model_copy(
        update={"source_group_id": test_group}
    )
    changed = tasks[train_index].model_copy(
        update={
            "metadata": tasks[train_index].metadata.model_copy(
                update={"split_key": split_key}
            )
        }
    )

    report = validate_pilot_release(_replace_task(tasks, train_index, changed), manifest)

    assert "pilot_release_source_group_split_leakage" in _codes(report)


def test_release_rejects_surface_and_core_cardinality_corruption(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    extra = dict(tasks[1].metadata.extra)
    extra["surface_variant"] = tasks[0].metadata.extra["surface_variant"]
    changed = tasks[1].model_copy(
        update={
            "metadata": tasks[1].metadata.model_copy(update={"extra": extra})
        }
    )

    report = validate_pilot_release(_replace_task(tasks, 1, changed), manifest)

    assert "pilot_release_duplicate_surface_id" in _codes(report)
    assert "pilot_release_core_surface_cardinality_mismatch" in _codes(report)


def test_release_rejects_semantic_hash_mismatch_within_core(canonical_release) -> None:
    tasks, manifest = canonical_release
    task = tasks[0]
    source = task.source.model_copy(update={"normalized_hash": "0" * 64})
    changed = task.model_copy(update={"source": source})

    report = validate_pilot_release(_replace_task(tasks, 0, changed), manifest)

    assert "pilot_release_semantic_hash_mismatch" in _codes(report)


def test_release_rejects_declared_stratum_mismatch(canonical_release) -> None:
    tasks, manifest = canonical_release
    summary = manifest.model_dump(mode="python")["leakage_check_summary"]
    summary["required_minimum_strata"] = summary["required_minimum_strata"][:-1]
    changed_manifest = _replace_manifest(manifest, leakage_check_summary=summary)

    report = validate_pilot_release(tasks, changed_manifest)

    assert "required_minimum_strata_mismatch" in _codes(report)


def test_release_rejects_noncanonical_generation_core_index_strata(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    core_id = tasks[0].metadata.split_key.semantic_core_id
    changed_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != core_id:
            changed_tasks.append(task)
            continue
        extra = dict(task.metadata.extra)
        extra["core_index"] = 120
        changed_tasks.append(
            task.model_copy(
                update={
                    "metadata": task.metadata.model_copy(update={"extra": extra})
                }
            )
        )

    report = validate_pilot_release(changed_tasks, manifest)

    assert "pilot_release_generation_strata_mismatch" in _codes(report)


def test_release_rejects_canonical_stratum_substitution(canonical_release) -> None:
    tasks, manifest = canonical_release
    family = TaskFamily.INTERLEAVED_MULTI_SLOT.value
    victim = next(
        task
        for task in tasks
        if task.task_family == family and task.difficulty.value == "easy"
    )
    donor = next(
        task
        for task in tasks
        if task.task_family == family and task.difficulty.value == "hard"
    )
    victim_core = victim.metadata.split_key.semantic_core_id
    changed_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != victim_core:
            changed_tasks.append(task)
            continue
        extra = dict(task.metadata.extra)
        extra["stratification"] = dict(donor.metadata.extra["stratification"])
        changed_tasks.append(
            task.model_copy(
                update={
                    "difficulty": donor.difficulty,
                    "metadata": task.metadata.model_copy(
                        update={
                            "profile_name": donor.metadata.profile_name,
                            "resolved_profile": donor.metadata.resolved_profile,
                            "extra": extra,
                        }
                    ),
                }
            )
        )

    report = validate_pilot_release(changed_tasks, manifest)

    assert "pilot_release_generation_strata_mismatch" in _codes(report)
    assert "pilot_release_generation_cell_count_mismatch" in _codes(report)


def test_release_aggregates_strict_per_task_semantic_corruption(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    index = _task_index(tasks, family=TaskFamily.REPEATED_SAME_SLOT)
    provenance = dict(tasks[index].source.provenance)
    provenance["release_id"] = "wrong-release"
    changed = tasks[index].model_copy(
        update={
            "source": tasks[index].source.model_copy(
                update={"provenance": provenance}
            )
        }
    )

    report = validate_pilot_release(_replace_task(tasks, index, changed), manifest)

    assert "family_a_release_provenance_mismatch" in _codes(report)


def test_release_rejects_nonunique_answer_support(canonical_release) -> None:
    tasks, manifest = canonical_release
    index = _task_index(tasks, family=TaskFamily.REPEATED_SAME_SLOT)
    task = tasks[index]
    query_id = task.queries[0].query_id
    acceptable = dict(task.gold.acceptable_answers)
    acceptable[query_id] = [task.gold.gold_answers[query_id], "another answer"]
    changed = task.model_copy(
        update={"gold": task.gold.model_copy(update={"acceptable_answers": acceptable})}
    )

    report = validate_pilot_release(_replace_task(tasks, index, changed), manifest)

    assert "family_a_multiple_current_answers" in _codes(report)


def test_release_handles_hostile_constructed_and_wrong_type_inputs_stably(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    hostile_task = MemUpdateTask.model_construct(task_family=TaskFamily.REPEATED_SAME_SLOT.value)
    hostile_manifest = TaskManifest.model_construct(data_release_id="pilot")
    malformed_tasks = [
        MemUpdateTask.model_construct(task_id=f"malformed-{index}")
        for index in range(200)
    ]
    inputs = [hostile_task, object(), *malformed_tasks]

    first = validate_pilot_release(inputs, hostile_manifest)
    second = validate_pilot_release(reversed(inputs), hostile_manifest)
    wrong = validate_pilot_release(42, object())

    exemplar = tasks[0]
    split_key_payload = dict(exemplar.metadata.split_key.__dict__)
    split_key_payload["source_group_id"] = []
    hostile_split_key = type(exemplar.metadata.split_key).model_construct(
        **split_key_payload
    )
    hostile_metadata = exemplar.metadata.model_copy(
        update={"split_key": hostile_split_key}
    )
    nested_payload = dict(exemplar.__dict__)
    nested_payload["metadata"] = hostile_metadata
    hostile_nested = MemUpdateTask.model_construct(**nested_payload)
    nested = validate_pilot_release([hostile_nested], manifest)
    oversized = validate_pilot_release(itertools.repeat(object()), manifest)

    assert first == second
    assert not first.valid
    assert len(first.issues) <= 128
    assert "pilot_release_issue_limit_reached" in _codes(first)
    assert not wrong.valid
    assert "pilot_release_malformed_tasks_iterable" in _codes(wrong)
    assert "pilot_release_malformed_manifest" in _codes(wrong)
    assert not nested.valid
    assert len(nested.issues) <= 128
    assert "pilot_release_input_size_limit" in _codes(oversized)
