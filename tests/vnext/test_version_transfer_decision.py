from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mub.vnext.external.visibility import visible_event_input
from mub.vnext.validation.replay_v3 import replay_task_v3, resolve_query_v3
from scripts import vnext_prepare_version_transfer_decision as prep


def test_tasks_are_canonical_paired_and_gold_replay_exact():
    tasks = prep.build_tasks(code_revision="test-revision")
    assert len(tasks) == 48
    assert Counter(t.metadata.split.value for t in tasks) == {"dev": 24, "test": 24}
    cores = {}
    for task in tasks:
        assert len(task.events) == 17
        replay = replay_task_v3(task)
        assert not replay.issues
        for query, gold in zip(task.queries, task.gold_evidence):
            resolution = resolve_query_v3(query, replay, task.events)
            assert not resolution.issues
            assert resolution.answer == gold.answer
            assert resolution.selected_event_ids == gold.supporting_event_ids
        cores.setdefault(task.metadata.split_key.semantic_core_id, []).append(task)
    assert len(cores) == 16
    assert len({t.target_objects[0].attribute for t in tasks}) == 16
    for group in cores.values():
        assert {t.metadata.extra["update_count"] for t in group} == {1, 4, 16}
        assert len({t.gold_evidence[0].answer for t in group}) == 1
        assert len({t.events[-1].timestamp for t in group}) == 1
        assert len({t.queries[1].text for t in group}) == 1


def test_provider_inputs_do_not_depend_on_gold_or_normalized_actions():
    task = prep.build_tasks(code_revision="test-revision")[0]
    event = task.events[0]
    original = visible_event_input(event, runtime_namespace="smoke")
    changed = event.model_copy(update={"gold_action_ids": (), "normalized_text": "GOLD MUST NOT REACH PROVIDER"})
    assert visible_event_input(changed, runtime_namespace="smoke") == original
    assert event.timestamp in original.raw_text
    assert "latest" not in original.raw_text.lower()
    assert "outdated" not in original.raw_text.lower()


def test_prepare_is_no_replace_and_integrity_checked(tmp_path):
    out = tmp_path / "packet"
    receipt = prep.publish(out, code_revision="test-revision")
    reopened = prep.read_packet(out, receipt["index_sha256"])
    assert len(reopened) == 48
    with pytest.raises(FileExistsError):
        prep.publish(out, code_revision="test-revision")
    (out / "tasks.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact"):
        prep.read_packet(out, receipt["index_sha256"])


def test_smoke_uses_four_small_canonical_tasks():
    cases = prep.build_smoke(code_revision="test-revision")
    assert len(cases) == 4
    assert {t.metadata.extra["smoke_case"] for t in cases} == {
        "replacement", "repeated_value", "history", "similar_entity"
    }
    for task in cases:
        assert len(task.events) <= 5
        assert not replay_task_v3(task).issues


def test_config_and_outputs_are_not_frozen_release_inputs():
    assert prep.CONFIG_PATH.name == "version_transfer_decision_v1.json"
    with pytest.raises(ValueError, match="protected"):
        prep.publish(prep.ROOT / "data/vnext/core/v3/forbidden", code_revision="test-revision")


def test_existing_frozen_result_roots_are_protected(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='protected'):
        prep.publish(tmp_path / 'results/vnext/pilot/addition', code_revision='test-revision')


def test_config_is_authoritative_not_silently_ignored(monkeypatch, tmp_path):
    config = json.loads(prep.CONFIG_PATH.read_bytes())
    config['depths'] = [2, 8]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(prep, 'CONFIG_PATH', path)
    with pytest.raises(ValueError, match='config'):
        prep.build_tasks(code_revision='test-revision')


def test_packet_binds_generator_and_contract_sources(tmp_path):
    out = tmp_path / 'source-bound'
    receipt = prep.publish(out, code_revision='test-revision')
    manifest = json.loads((out / 'manifest.json').read_bytes())
    assert manifest['source_hash_policy'] == 'python_lf_normalized'
    assert manifest['source_hashes']['scripts/vnext_prepare_version_transfer_decision.py']
    assert manifest['source_hashes']['mub/vnext/contracts/v3/task.py']
    assert manifest['independent_sample_claim'] == '16_declared_synthetic_trajectories_not_population_independence'


def test_depth_conditions_use_one_matched_form_background_entity():
    tasks = prep.build_tasks(code_revision='test-revision')
    for task in tasks:
        target = task.queries[0].target_object_keys[0]
        others = [key for key in task.target_objects if key.canonical_id != target.canonical_id]
        depth = task.metadata.extra['update_count']
        assert len(others) == (0 if depth == 16 else 1)
        if others:
            assert others[0].entity.split('-')[0] == target.entity.split('-')[0]
            assert len(others[0].entity) == len(target.entity)


def test_value_labels_do_not_encode_event_time_or_fixed_final_version():
    tasks = prep.build_tasks(code_revision='test-revision')
    high = [t for t in tasks if t.metadata.extra['update_count'] == 16]
    finals = []
    for task in high:
        values = [a.value for a in task.actions]
        assert values != sorted(values) and values != sorted(values, reverse=True)
        assert task.gold_evidence[0].answer not in (min(values), max(values))
        finals.append(task.gold_evidence[0].answer.rsplit('-', 1)[1])
    assert len(set(finals)) >= 6
    for split in ('dev', 'test'):
        low = [t for t in tasks if t.metadata.extra['update_count'] == 1 and t.metadata.split.value == split]
        assert sum(t.gold_evidence[0].answer > t.actions[0].value for t in low) == 4


def test_value_schedule_is_deterministic_and_keeps_smoke_surface_unchanged():
    assert prep._value_schedule(0) == prep._value_schedule(0)
    assert all(sorted(prep._value_schedule(i)) == list(range(17)) for i in range(16))
    cases = prep.build_smoke(code_revision='test-revision')
    for task in cases:
        repeated = task.metadata.extra['smoke_case'] == 'repeated_value'
        target = task.queries[0].target_object_keys[0].canonical_id
        for position, action in enumerate(task.actions):
            expected = 0 if repeated and action.target_object_keys[0].canonical_id == target else position
            assert action.value.endswith(f'-{expected:02d}')


def test_manifest_declares_surface_and_reports_simple_shortcut_baselines(tmp_path):
    out = tmp_path / 'v3-candidate'
    prep.publish(out, code_revision='test-revision')
    manifest = json.loads((out / 'manifest.json').read_bytes())
    assert manifest['surface_revision'] == 'matched-entity-counterbalanced-values-v3'
    audit = manifest['value_shortcut_audit']
    assert audit['evidence_class'] == 'deterministic_input_control_not_model_results'
    assert audit['rows']['dev/u1']['maximum_value_correct'] == 4
    assert audit['rows']['test/u16']['maximum_value_correct'] == 0
    assert audit['rows']['test/u16']['minimum_value_correct'] == 0
    assert manifest['tokenizer_length_validation'] == 'PENDING_MODEL_QUALIFICATION'
