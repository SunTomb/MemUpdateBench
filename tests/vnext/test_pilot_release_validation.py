from __future__ import annotations

import hashlib
import importlib.resources
import itertools
import random
from pathlib import Path

import pytest
import yaml

import mub.vnext.validation.pilot as pilot_module
from mub.vnext.contracts import (
    AnswerDisposition,
    ArtifactRef,
    MemUpdateTask,
    Split,
    TaskFamily,
    TaskManifest,
)
from mub.vnext.generation import (
    PilotConfig,
    build_pilot_artifact_bundle,
    compile_pilot_tasks,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes, sha256_model
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


def _coordinated_manifest_for_tasks(
    tasks: tuple[MemUpdateTask, ...], manifest: TaskManifest
) -> TaskManifest:
    split_order = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
    family_order = {
        family.value: index
        for index, family in enumerate(
            (
                TaskFamily.REPEATED_SAME_SLOT,
                TaskFamily.INTERLEAVED_MULTI_SLOT,
                TaskFamily.ENTITY_ATTRIBUTE_GROUNDING,
                TaskFamily.NOOP_WRITE_DISCIPLINE,
            )
        )
    }
    ordered = sorted(
        tasks,
        key=lambda task: (
            split_order[task.metadata.split],
            family_order[task.task_family],
            task.metadata.split_key.semantic_core_id,
            task.metadata.extra["surface_variant"],
        ),
    )
    task_bytes = b"".join(canonical_json_bytes(task) + b"\n" for task in ordered)
    task_ref_payload = manifest.task_file_paths_and_hashes[0].model_dump(mode="python")
    task_ref_payload["sha256"] = hashlib.sha256(task_bytes).hexdigest()
    task_ref_payload["record_count"] = len(tasks)
    task_ref = ArtifactRef.model_validate(task_ref_payload)
    summary = manifest.model_dump(mode="python")["leakage_check_summary"]
    summary["task_hashes"] = {
        task.task_id: sha256_model(task) for task in sorted(tasks, key=lambda item: item.task_id)
    }
    return _replace_manifest(
        manifest,
        task_file_paths_and_hashes=(task_ref,),
        leakage_check_summary=summary,
    )


def _replace_strings(value, old: str, new: str):
    if type(value) is str:
        return value.replace(old, new)
    if type(value) is list:
        return [_replace_strings(item, old, new) for item in value]
    if type(value) is tuple:
        return tuple(_replace_strings(item, old, new) for item in value)
    if type(value) is dict:
        return {
            _replace_strings(key, old, new): _replace_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def _replace_object_type(value, replacement: str):
    if type(value) is list:
        return [_replace_object_type(item, replacement) for item in value]
    if type(value) is tuple:
        return tuple(_replace_object_type(item, replacement) for item in value)
    if type(value) is dict:
        return {
            key: (
                replacement
                if key == "object_type"
                else _replace_object_type(item, replacement)
            )
            for key, item in value.items()
        }
    return value


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


def test_canonical_pilot_config_is_packaged_and_digest_bound() -> None:
    resource = importlib.resources.files("mub.vnext.resources").joinpath(
        "pilot.yaml"
    )
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    config = PilotConfig.model_validate(payload)

    assert resource.read_text(encoding="utf-8") == CONFIG_PATH.read_text(
        encoding="utf-8"
    )
    assert hashlib.sha256(canonical_json_bytes(config)).hexdigest() == (
        "685759627773beba18f431a53c43f7077d9639596ee1a78fe970265a0d0626bf"
    )


def test_release_snapshots_tasks_and_manifest_before_validation(
    canonical_release, monkeypatch
) -> None:
    tasks, manifest = canonical_release
    original_validate = pilot_module.validate_pilot_task
    original_provenance = dict(tasks[0].source.provenance)
    original_revision = manifest.code_revision
    mutated = False

    def mutate_callers_then_validate(task):
        nonlocal mutated
        if not mutated:
            mutated = True
            tasks[0].source.provenance["release_id"] = "mutated-after-ingress"
            object.__setattr__(manifest, "code_revision", "mutated-after-ingress")
        return original_validate(task)

    monkeypatch.setattr(pilot_module, "validate_pilot_task", mutate_callers_then_validate)
    try:
        report = validate_pilot_release(tasks, manifest)
    finally:
        tasks[0].source.provenance.clear()
        tasks[0].source.provenance.update(original_provenance)
        object.__setattr__(manifest, "code_revision", original_revision)

    assert mutated
    assert report.valid
    assert report.issues == ()


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


def test_release_rejects_coordinated_core_identity_relabel(canonical_release) -> None:
    tasks, manifest = canonical_release
    original_core_id = tasks[0].metadata.split_key.semantic_core_id
    relabeled_core_id = "core_ffffffffffffffff"
    changed_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != original_core_id:
            changed_tasks.append(task)
            continue
        split_key = task.metadata.split_key.model_copy(
            update={"semantic_core_id": relabeled_core_id}
        )
        extra = dict(task.metadata.extra)
        extra["semantic_core_id"] = relabeled_core_id
        provenance = dict(task.source.provenance)
        provenance["semantic_core_id"] = relabeled_core_id
        changed_tasks.append(
            task.model_copy(
                update={
                    "source": task.source.model_copy(
                        update={"provenance": provenance}
                    ),
                    "metadata": task.metadata.model_copy(
                        update={"split_key": split_key, "extra": extra}
                    ),
                }
            )
        )
    changed = tuple(changed_tasks)
    coordinated_manifest = _coordinated_manifest_for_tasks(changed, manifest)

    report = validate_pilot_release(changed, coordinated_manifest)

    assert "pilot_release_canonical_identity_mismatch" in _codes(report)


def test_release_rejects_coordinated_four_part_identity_relabel(
    canonical_release,
) -> None:
    tasks, manifest = canonical_release
    exemplar = next(
        task
        for task in tasks
        if task.task_family == TaskFamily.REPEATED_SAME_SLOT.value
        and task.metadata.extra["core_index"] == 0
    )
    core_id = exemplar.metadata.split_key.semantic_core_id
    original_entity = exemplar.target_objects[0].entity
    replacement_entity = "Caller Chosen Entity"
    changed = tuple(
        MemUpdateTask.model_validate(
            _replace_strings(
                task.model_dump(mode="python"),
                original_entity,
                replacement_entity,
            )
        )
        if task.metadata.split_key.semantic_core_id == core_id
        else task
        for task in tasks
    )
    coordinated_manifest = _coordinated_manifest_for_tasks(changed, manifest)

    report = validate_pilot_release(changed, coordinated_manifest)

    assert "pilot_release_semantic_core_hash_mismatch" in _codes(report)


def test_release_allows_object_type_and_admin_only_mutation(canonical_release) -> None:
    tasks, manifest = canonical_release
    exemplar = tasks[0]
    core_id = exemplar.metadata.split_key.semantic_core_id
    changed_tasks = []
    for task in tasks:
        if task.metadata.split_key.semantic_core_id != core_id:
            changed_tasks.append(task)
            continue
        payload = _replace_object_type(
            task.model_dump(mode="python"),
            "caller_classification",
        )
        payload["metadata"]["extra"]["caller_admin_note"] = "permitted"
        changed_tasks.append(MemUpdateTask.model_validate(payload))
    changed = tuple(changed_tasks)
    coordinated_manifest = _coordinated_manifest_for_tasks(changed, manifest)

    report = validate_pilot_release(changed, coordinated_manifest)

    assert report.valid
    assert report.issues == ()


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

    cyclic_extra = {}
    cyclic_extra["self"] = cyclic_extra
    cyclic_metadata_payload = dict(exemplar.metadata.__dict__)
    cyclic_metadata_payload["extra"] = cyclic_extra
    cyclic_metadata = type(exemplar.metadata).model_construct(
        **cyclic_metadata_payload
    )
    cyclic_payload = dict(exemplar.__dict__)
    cyclic_payload["metadata"] = cyclic_metadata
    cyclic_task = MemUpdateTask.model_construct(**cyclic_payload)
    cyclic = validate_pilot_release([cyclic_task], manifest)
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
    assert not cyclic.valid
    assert len(cyclic.issues) <= 128
    assert "family_a_cyclic_json" in _codes(cyclic)
    assert "pilot_release_input_size_limit" in _codes(oversized)
