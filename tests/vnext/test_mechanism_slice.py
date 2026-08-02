from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mub.vnext.contracts import MemoryObjectKey, Split
from mub.vnext.generation.config import load_pilot_config
from mub.vnext.generation.core import GenerationContext
from mub.vnext.generation.family_a import generate_family_a_cores
from mub.vnext.generation.render import render_core
from mub.vnext.io import canonical_json_bytes
from mub.vnext.mechanisms.context import ContextEntry, entries_from_task, render_context
from mub.vnext.mechanisms.matrix import build_mechanism_slice

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "vnext" / "pilot.yaml"


def _config():
    return load_pilot_config(CONFIG_PATH)


def _tasks():
    config = _config()
    generation = GenerationContext(config=config, code_revision="test-revision")
    cores = generate_family_a_cores(config)
    selected = [
        next(
            core
            for core in cores
            if core.profile["stale_count"] == stale_count
            and core.profile["active_object_count"] > 1
        )
        for stale_count in (1, 16)
    ]
    return tuple(
        render_core(core, split=Split.TEST, surface_variant=0, context=generation)
        for core in selected[:2]
    )


def test_render_context_pairs_preserve_entries_and_gold_without_answer_labels():
    task = _tasks()[0]
    entries = entries_from_task(task)
    outputs = {
        condition: render_context(entries, *condition)
        for condition in (
            ("chronological", "none"),
            ("reverse_chronological", "none"),
            ("reverse_chronological", "latest_outdated_label"),
        )
    }
    assert {frozenset(item.entry_ids) for item in outputs.values()} == {frozenset(outputs["chronological", "none"].entry_ids)}
    assert {item.value for item in entries} == {item.value for item in entries}
    assert outputs["chronological", "none"].labels == {}
    labeled = outputs["reverse_chronological", "latest_outdated_label"]
    assert set(labeled.labels.values()) == {"latest", "outdated"}
    assert all("Qingdao" not in label for label in labeled.labels.values())
    assert any("[latest]" in line for line in labeled.rendered_context.splitlines())


@pytest.mark.parametrize("annotation", ["none", "latest_outdated_label"])
def test_reverse_order_places_reversed_stale_target_versions_before_current(annotation):
    tasks = _tasks()
    assert {task.metadata.resolved_profile["stale_count"] for task in tasks} == {1, 16}
    for task in tasks:
        entries = entries_from_task(task)
        target_key = task.queries[0].target_object_keys[0]
        target_entries = [entry for entry in entries if entry.object_key == target_key]
        current = max(
            target_entries,
            key=lambda entry: (entry.event_index, entry.version_index, entry.entry_id),
        )
        stale = [entry for entry in target_entries if entry.entry_id != current.entry_id]
        expected_stale_ids = [
            entry.entry_id
            for entry in sorted(
                stale,
                key=lambda entry: (entry.event_index, entry.version_index, entry.entry_id),
                reverse=True,
            )
        ]
        auxiliary = [entry for entry in entries if entry.object_key != target_key]
        expected_auxiliary_ids = [
            entry.entry_id
            for entry in sorted(
                auxiliary,
                key=lambda entry: (entry.event_index, entry.version_index, entry.entry_id),
                reverse=True,
            )
        ]

        rendered = render_context(entries, "reverse_chronological", annotation)
        rendered_target_ids = [entry_id for entry_id in rendered.entry_ids if entry_id in {entry.entry_id for entry in target_entries}]
        rendered_auxiliary_ids = [entry_id for entry_id in rendered.entry_ids if entry_id in {entry.entry_id for entry in auxiliary}]

        assert rendered_target_ids == [*expected_stale_ids, current.entry_id]
        assert all(rendered.entry_ids.index(entry_id) < rendered.entry_ids.index(current.entry_id) for entry_id in expected_stale_ids)
        assert rendered_auxiliary_ids == expected_auxiliary_ids
        if annotation == "latest_outdated_label":
            assert rendered.labels[current.entry_id] == "latest"
            assert all(rendered.labels[entry_id] == "outdated" for entry_id in expected_stale_ids)


def test_render_context_rejects_unsupported_cells_and_malformed_entries():
    entry = ContextEntry(
        entry_id="entry-0",
        event_id="event-0",
        event_index=0,
        version_index=0,
        object_key=MemoryObjectKey(object_type="slot", entity="e", attribute="a"),
        value="old",
    )
    with pytest.raises(ValueError, match="unsupported"):
        render_context([entry], "chronological", "latest_outdated_label")
    with pytest.raises(ValueError, match="duplicate"):
        render_context([entry, entry], "chronological", "none")
    with pytest.raises(ValueError, match="unsupported"):
        render_context([entry], "random", "none")


def test_build_slice_selects_exact_stale_counts_and_manifest_contract():
    result = build_mechanism_slice(_tasks(), _config())
    assert set(result.manifest["stale_counts"]) == {1, 16}
    assert result.manifest["answer_model"] == "deterministic_reference_smoke"
    assert result.manifest["smoke_only"] is True
    assert result.manifest["not_model_result"] is True
    assert len(result.records) == len(_tasks()) * 3
    assert {record.stale_count for record in result.records} == {1, 16}
    assert all(record.condition_id for record in result.records)
    assert all(record.task_id for record in result.records)
    assert all(record.semantic_core_id for record in result.records)
    assert all(record.retrieval_composition == "identical_entry_multiset" for record in result.records)
    for semantic_core_id in result.manifest["semantic_core_ids"]:
        paired = [record for record in result.records if record.semantic_core_id == semantic_core_id]
        assert len({record.gold_value for record in paired}) == 1
        assert len({frozenset(record.entry_ids) for record in paired}) == 1


def test_build_slice_is_permutation_invariant_and_rejects_invalid_inputs():
    tasks = _tasks()
    config = _config()
    left = build_mechanism_slice(tasks, config)
    right = build_mechanism_slice(tuple(reversed(tasks)), config)
    assert [record.model_dump(mode="json") for record in left.records] == [record.model_dump(mode="json") for record in right.records]
    with pytest.raises((TypeError, ValueError)):
        build_mechanism_slice(tasks, object())
    with pytest.raises(ValueError, match="test"):
        build_mechanism_slice((tasks[0].model_copy(update={"metadata": tasks[0].metadata.model_copy(update={"split": "dev"})}),), config)


def test_cli_writes_stable_canonical_outputs_without_network(tmp_path):
    tasks = _tasks()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_bytes(b"".join(canonical_json_bytes(task) + b"\n" for task in tasks))
    output_dir = tmp_path / "slice"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "vnext_build_mechanism_slice.py"),
        "--tasks", str(tasks_path),
        "--config", str(CONFIG_PATH),
        "--output-dir", str(output_dir),
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    first_context = (output_dir / "contexts.jsonl").read_bytes()
    first_manifest = (output_dir / "condition_manifest.json").read_bytes()
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert first.returncode == second.returncode == 0
    assert first_context == (output_dir / "contexts.jsonl").read_bytes()
    assert first_manifest == (output_dir / "condition_manifest.json").read_bytes()
    manifest = json.loads(first_manifest)
    assert manifest["answer_model"] == "deterministic_reference_smoke"
    assert manifest["not_model_result"] is True
