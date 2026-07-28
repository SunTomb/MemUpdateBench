from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import mub.vnext.generation.build as build_module
from mub.vnext.contracts import MemUpdateTask, Split, TaskFamily
from mub.vnext.generation import (
    CompiledPilotTasks,
    PilotConfig,
    compile_pilot_tasks,
    generate_family_a_cores,
    generate_family_b_cores,
    generate_family_c_cores,
    generate_family_d_cores,
    load_pilot_config,
)
from mub.vnext.io import canonical_json_bytes, semantic_task_hash
from mub.vnext.validation import (
    ValidationIssue,
    build_report,
    validate_gold_replay,
    validate_task,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"
REVISION = "generation-build-test-revision"
SPLIT_ORDER = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}
FAMILY_ORDER = {
    TaskFamily.REPEATED_SAME_SLOT.value: 0,
    TaskFamily.INTERLEAVED_MULTI_SLOT.value: 1,
    TaskFamily.ENTITY_ATTRIBUTE_GROUNDING.value: 2,
    TaskFamily.NOOP_WRITE_DISCIPLINE.value: 3,
}


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def compiled(config) -> CompiledPilotTasks:
    return compile_pilot_tasks(config, code_revision=REVISION)


@pytest.fixture(scope="module")
def cores_by_family(config):
    return (
        generate_family_a_cores(config),
        generate_family_b_cores(config),
        generate_family_c_cores(config),
        generate_family_d_cores(config),
    )


def _sort_key(task: MemUpdateTask):
    return (
        SPLIT_ORDER[task.metadata.split],
        FAMILY_ORDER[task.task_family],
        task.metadata.split_key.semantic_core_id,
        task.metadata.extra["surface_variant"],
    )


def _all_linked_ids(tasks: tuple[MemUpdateTask, ...]) -> list[str]:
    linked_ids = []
    for task in tasks:
        linked_ids.extend((task.task_id, task.source.source_id))
        linked_ids.extend(event.event_id for event in task.events)
        linked_ids.extend(action.action_id for action in task.gold.actions)
        linked_ids.extend(query.query_id for query in task.queries)
    return linked_ids


def test_compile_pilot_tasks_has_exact_counts_order_linkage_and_validation(
    compiled,
) -> None:
    tasks = compiled.tasks
    assert isinstance(tasks, tuple)
    assert len(tasks) == 1440
    assert tasks == tuple(sorted(tasks, key=_sort_key))
    assert Counter(task.metadata.split for task in tasks) == {
        Split.TRAIN: 1008,
        Split.DEV: 144,
        Split.TEST: 288,
    }
    assert Counter(task.task_family for task in tasks) == {
        family: 360 for family in FAMILY_ORDER
    }

    variants_by_core = defaultdict(set)
    hashes_by_core = defaultdict(set)
    for task in tasks:
        core_id = task.metadata.split_key.semantic_core_id
        variants_by_core[core_id].add(task.metadata.extra["surface_variant"])
        hashes_by_core[core_id].add(semantic_task_hash(task))
        assert task.source.generator is not None
        assert task.source.generator.config_sha256 == compiled.config_sha256
        assert task.source.generator.code_revision == compiled.code_revision
        assert task.source.generator.compiler_version == compiled.compiler_version
        assert task.source.generator.generator_name == compiled.generator_name
        assert validate_task(task).valid
        assert validate_gold_replay(task).valid
    assert len(variants_by_core) == 480
    assert all(variants == {0, 1, 2} for variants in variants_by_core.values())
    assert all(len(hashes) == 1 for hashes in hashes_by_core.values())
    linked_ids = _all_linked_ids(tasks)
    assert len(linked_ids) == len(set(linked_ids))

    rows = compiled.tasks_jsonl.splitlines(keepends=True)
    assert len(rows) == 1440
    for task, row in zip(tasks, rows, strict=True):
        assert row.endswith(b"\n") and not row.endswith(b"\r\n")
        reparsed = MemUpdateTask.model_validate_json(row[:-1])
        assert reparsed == task
        assert canonical_json_bytes(reparsed) + b"\n" == row


def test_same_inputs_are_byte_identical_config_unchanged_and_create_no_files(
    config,
    compiled,
    tmp_path,
    monkeypatch,
) -> None:
    before = canonical_json_bytes(config)
    monkeypatch.chdir(tmp_path)

    repeated = compile_pilot_tasks(config, code_revision=REVISION)

    assert repeated == compiled
    assert repeated.tasks_jsonl == compiled.tasks_jsonl
    assert canonical_json_bytes(config) == before
    assert list(tmp_path.iterdir()) == []
    with pytest.raises((AttributeError, TypeError)):
        repeated.tasks = ()
    with pytest.raises(ValidationError):
        repeated.split_assignment.assignments = ()


def test_compiled_tasks_property_returns_fresh_deep_snapshot(compiled) -> None:
    original_jsonl = compiled.tasks_jsonl
    first_access = compiled.tasks
    original_task_id = first_access[0].task_id
    original_text = first_access[0].events[0].raw_text

    object.__setattr__(first_access[0], "task_id", "task_mutated")
    first_access[0].events[0].raw_text = "mutated text"
    second_access = compiled.tasks

    assert second_access is not first_access
    assert second_access[0] is not first_access[0]
    assert second_access[0].task_id == original_task_id
    assert second_access[0].events[0].raw_text == original_text
    assert compiled.tasks_jsonl == original_jsonl


def test_compiled_snapshot_rejects_noncanonical_framing_and_row_replacement(
    compiled,
) -> None:
    with pytest.raises(ValueError, match="final LF"):
        replace(compiled, tasks_jsonl=compiled.tasks_jsonl[:-1])
    with pytest.raises(ValueError, match="LF-only"):
        replace(compiled, tasks_jsonl=compiled.tasks_jsonl.replace(b"\n", b"\r\n", 1))
    with pytest.raises(ValueError, match="UTF-8 BOM"):
        replace(compiled, tasks_jsonl=b"\xef\xbb\xbf" + compiled.tasks_jsonl)
    with pytest.raises(ValueError, match="blank rows"):
        replace(compiled, tasks_jsonl=b"\n" + compiled.tasks_jsonl)

    rows = compiled.tasks_jsonl.split(b"\n")
    rows[0] = rows[1]
    with pytest.raises(ValueError):
        replace(compiled, tasks_jsonl=b"\n".join(rows))
    with pytest.raises(ValueError, match="generator provenance"):
        replace(compiled, config_sha256="0" * 64)


def test_public_snapshot_gate_validates_all_rows_after_gold_tamper(
    compiled,
    monkeypatch,
) -> None:
    tasks = compiled.tasks
    tampered_core = tasks[0].metadata.split_key.semantic_core_id
    tampered = [
        task
        for task in tasks
        if task.metadata.split_key.semantic_core_id == tampered_core
    ]
    assert len(tampered) == 3
    for task in tampered:
        object.__setattr__(task.gold, "final_state", {"tampered": "state"})
    tampered_jsonl = b"".join(
        canonical_json_bytes(task) + b"\n" for task in tasks
    )

    calls = Counter()
    original_task_validator = build_module.validate_task
    original_replay_validator = build_module.validate_gold_replay

    def task_validator(task):
        calls["task"] += 1
        return original_task_validator(task)

    def replay_validator(task):
        calls["gold_replay"] += 1
        return original_replay_validator(task)

    monkeypatch.setattr(build_module, "validate_task", task_validator)
    monkeypatch.setattr(build_module, "validate_gold_replay", replay_validator)
    with pytest.raises(ValueError) as exc_info:
        replace(compiled, tasks_jsonl=tampered_jsonl)

    assert calls == {"task": 1440, "gold_replay": 1440}
    message = str(exc_info.value)
    assert "validation stage=gold_replay" in message
    assert "final_state" in message


def test_reversed_generator_inputs_still_produce_canonical_order(
    config,
    compiled,
    cores_by_family,
    monkeypatch,
) -> None:
    generator_names = (
        "generate_family_a_cores",
        "generate_family_b_cores",
        "generate_family_c_cores",
        "generate_family_d_cores",
    )
    for name, cores in zip(generator_names, cores_by_family, strict=True):
        monkeypatch.setattr(
            build_module,
            name,
            lambda _config, records=cores: list(reversed(records)),
        )
    original_assign = build_module.assign_splits
    calls = 0

    def counted_assign(cores, seed):
        nonlocal calls
        calls += 1
        return original_assign(cores, seed)

    monkeypatch.setattr(build_module, "assign_splits", counted_assign)
    reversed_result = compile_pilot_tasks(config, code_revision=REVISION)

    assert calls == 1
    assert reversed_result == compiled
    assert reversed_result.tasks_jsonl == compiled.tasks_jsonl


def test_revision_changes_only_surface_artifact_provenance(
    config,
    compiled,
) -> None:
    changed = compile_pilot_tasks(config, code_revision="different-revision")
    changed_tasks = changed.tasks
    original_tasks = compiled.tasks

    assert changed.tasks_jsonl != compiled.tasks_jsonl
    assert changed.config_sha256 == compiled.config_sha256
    assert changed.code_revision == "different-revision"
    assert changed.compiler_version == compiled.compiler_version
    assert changed.generator_name == compiled.generator_name
    assert changed.split_assignment == compiled.split_assignment
    assert all(
        task.source.generator is not None
        and task.source.generator.code_revision == "different-revision"
        for task in changed_tasks
    )
    assert [semantic_task_hash(task) for task in changed_tasks] == [
        semantic_task_hash(task) for task in original_tasks
    ]

    original_rows = compiled.tasks_jsonl.splitlines(keepends=True)
    changed_rows = changed.tasks_jsonl.splitlines(keepends=True)
    mixed_revision = b"".join((changed_rows[0], *original_rows[1:]))
    with pytest.raises(ValueError, match="generator provenance"):
        replace(compiled, tasks_jsonl=mixed_revision)
    with pytest.raises(ValueError, match="generator provenance"):
        replace(compiled, code_revision="different-revision")


def test_repeated_variant_renderer_is_rejected(config, monkeypatch) -> None:
    original_render = build_module.render_core

    def repeated_variant(core, *, split, surface_variant, context):
        return original_render(
            core,
            split=split,
            surface_variant=0,
            context=context,
        )

    monkeypatch.setattr(build_module, "render_core", repeated_variant)
    with pytest.raises(ValueError) as exc_info:
        compile_pilot_tasks(config, code_revision=REVISION)

    message = str(exc_info.value)
    assert "field=surface_variant expected=1 observed=0" in message
    assert "must have exactly surface variants [0, 1, 2]" in message
    assert "duplicate linked ID" in message


def test_all_validators_run_and_failures_are_stably_aggregated(
    config,
    compiled,
    monkeypatch,
) -> None:
    tasks = compiled.tasks
    first_task_id = tasks[0].task_id
    last_task_id = tasks[-1].task_id
    calls = Counter()
    render_calls = 0
    original_render = build_module.render_core

    def corrupted_render(core, *, split, surface_variant, context):
        nonlocal render_calls
        task = original_render(
            core,
            split=split,
            surface_variant=surface_variant,
            context=context,
        )
        if render_calls == 0:
            object.__setattr__(task, "task_family", "corrupted_family")
        elif render_calls == 1:
            object.__setattr__(task.metadata, "split", Split.EVALUATION_ONLY)
        elif render_calls == 2:
            extra = dict(task.metadata.extra)
            extra["surface_variant"] = "corrupted_variant"
            object.__setattr__(task.metadata, "extra", extra)
        render_calls += 1
        return task

    def task_validator(task):
        calls["task"] += 1
        issues = []
        if task.task_id == first_task_id:
            issues.append(
                ValidationIssue(
                    code="early_structural_failure",
                    path="task_id",
                    message="early task failed",
                    severity="error",
                )
            )
        return build_report(issues)

    def replay_validator(task):
        calls["gold_replay"] += 1
        issues = []
        if task.task_id == last_task_id:
            issues.append(
                ValidationIssue(
                    code="late_replay_failure",
                    path="gold",
                    message="late task failed",
                    severity="error",
                )
            )
        return build_report(issues)

    monkeypatch.setattr(build_module, "render_core", corrupted_render)
    monkeypatch.setattr(build_module, "validate_task", task_validator)
    monkeypatch.setattr(build_module, "validate_gold_replay", replay_validator)
    with pytest.raises(ValueError) as exc_info:
        compile_pilot_tasks(config, code_revision=REVISION)

    assert render_calls == 1440
    assert calls == {"task": 1440, "gold_replay": 1440}
    message = str(exc_info.value)
    early = (
        f"validation stage=task task={first_task_id!r} "
        "issue=early_structural_failure@task_id: early task failed"
    )
    late = (
        f"validation stage=gold_replay task={last_task_id!r} "
        "issue=late_replay_failure@gold: late task failed"
    )
    assert early in message
    assert late in message
    assert "field=task_family" in message
    assert "field=split" in message
    assert "field=surface_variant" in message
    assert "family counts expected=" in message
    assert "split counts expected=" in message
    assert "canonical order exception=KeyError" in message
    assert message.index(early) < message.index("field=task_family")
    assert message.index(early) < message.index(late)


def test_noncanonical_surface_variant_count_rejected_before_render(
    config,
    monkeypatch,
) -> None:
    payload = config.model_dump(mode="python")
    payload["surface_variants_per_core"] = 4
    changed = PilotConfig.model_validate(payload)
    render_calls = 0

    def unexpected_render(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        raise AssertionError("render must not be called")

    monkeypatch.setattr(build_module, "render_core", unexpected_render)
    with pytest.raises(
        ValueError,
        match="surface_variants_per_core == 3",
    ):
        compile_pilot_tasks(changed, code_revision=REVISION)
    assert render_calls == 0


def test_noncanonical_split_counts_rejected_before_generation(
    config,
    monkeypatch,
) -> None:
    payload = config.model_dump(mode="python")
    payload["splits"] = {"train": 0.6, "dev": 0.2, "test": 0.2}
    changed = PilotConfig.model_validate(payload)
    generation_calls = 0
    render_calls = 0

    def unexpected_generation(_config):
        nonlocal generation_calls
        generation_calls += 1
        raise AssertionError("generation must not be called")

    def unexpected_render(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        raise AssertionError("render must not be called")

    monkeypatch.setattr(
        build_module,
        "generate_family_a_cores",
        unexpected_generation,
    )
    monkeypatch.setattr(build_module, "render_core", unexpected_render)
    with pytest.raises(ValueError, match="split task counts"):
        compile_pilot_tasks(changed, code_revision=REVISION)
    assert generation_calls == 0
    assert render_calls == 0


@pytest.mark.parametrize("revision", ["", " ", "\t\n"])
def test_code_revision_must_be_explicit_and_nonblank(config, revision) -> None:
    with pytest.raises(ValueError, match="code_revision must not be blank"):
        compile_pilot_tasks(config, code_revision=revision)
