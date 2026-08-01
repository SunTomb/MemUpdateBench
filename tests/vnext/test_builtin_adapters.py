from __future__ import annotations

import json

import pytest
from pydantic import RootModel

from mub.vnext.adapters import (
    ExactCrudAdapter,
    HeuristicCrudAdapter,
    RawAppendAdapter,
    ReferenceAdapter,
)
from mub.vnext.contracts.enums import AnswerDisposition, Operation
from mub.vnext.io.canonical import canonical_json_bytes


def _direct_task(make_task):
    task = make_task()
    task.events[0].raw_text = 'ADD friend_alex.location = Dalian'
    task.events[0].normalized_text = task.events[0].raw_text
    task.events[1].raw_text = 'UPDATE friend_alex.location = Qingdao'
    task.events[1].normalized_text = task.events[1].raw_text
    return task


@pytest.mark.parametrize('adapter_type', [RawAppendAdapter, ExactCrudAdapter])
def test_builtin_adapters_reset_link_source_events_and_canonical_entries(make_task, adapter_type):
    task = _direct_task(make_task)
    adapter = adapter_type()
    assert adapter.reset('session-a', {}) .success
    logs = [adapter.ingest_event(event) for event in task.events]
    assert [log.event_id for log in logs] == [event.event_id for event in task.events]
    assert all(log.error is None for log in logs)
    entries = adapter.export_entries()
    assert entries
    assert all(entry.source_event_ids for entry in entries)
    values = [entry.value_candidate for entry in entries]
    if adapter_type is RawAppendAdapter:
        assert values == ['Dalian', 'Qingdao']
    else:
        assert values == ['Qingdao']
    assert canonical_json_bytes(entries[0]).decode('utf-8') == canonical_json_bytes(entries[0]).decode('utf-8')
    assert adapter.capabilities().supports_isolated_reset
    assert isinstance(adapter.capability_bitset(), int)
    assert adapter.reset('session-b', {}).success
    assert adapter.export_entries() == []
    adapter.close()
    adapter.close()


def test_reference_is_gold_oracle_and_isolated(make_task):
    task = _direct_task(make_task)
    adapter = ReferenceAdapter(task)
    assert adapter.adapter_info().system_name == 'oracle_smoke_only'
    assert adapter.adapter_info().adapter_id == 'reference'
    assert adapter.reset('oracle', {}).success
    for event in task.events:
        adapter.ingest_event(event)
    assert adapter.export_raw_state()['state_by_object'] == task.gold.final_state
    assert adapter.export_raw_state()['history'] == task.gold.version_history
    assert len(adapter.export_raw_state()['action_trace']) == 2
    answer = adapter.answer(task.queries[0], 'slot_direct')
    assert answer.disposition is AnswerDisposition.ANSWERED
    assert answer.value == 'Qingdao'


def test_exact_identity_excludes_object_type(make_task):
    task = _direct_task(make_task)
    key = task.target_objects[0]
    first = key.model_copy(update={'object_type': 'slot'})
    second = key.model_copy(update={'object_type': 'profile'})
    assert first.canonical_id == second.canonical_id
    assert first == second


def test_heuristic_requires_verified_nonzero_encoder(make_task):
    task = _direct_task(make_task)

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    adapter = HeuristicCrudAdapter(encoder=FakeEncoder(), encoder_model='fake/model', encoder_revision='r1', backend='fake')
    assert adapter.reset('heuristic', {}).success
    assert adapter.adapter_info().system_version == 'fake/model'
    assert adapter.adapter_info().sdk_version == 'r1'
    assert adapter.adapter_info().extractor_id == 'fake'

    unavailable = HeuristicCrudAdapter()
    reset = unavailable.reset('unavailable', {})
    assert not reset.success
    assert reset.error['code'] == 'not_supported'
    log = unavailable.ingest_event(task.events[0])
    assert log.error['code'] == 'not_supported'

    class ZeroEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.0, 0.0] for _ in texts]

    zero = HeuristicCrudAdapter(encoder=ZeroEncoder())
    assert zero.reset('zero', {}).error['code'] == 'not_supported'


def test_reference_answer_is_deterministically_serialized(make_task):
    task = _direct_task(make_task)
    left = ReferenceAdapter(task)
    right = ReferenceAdapter(task)
    for adapter in (left, right):
        assert adapter.reset('x', {}).success
        for event in task.events:
            adapter.ingest_event(event)
    left_model = RootModel[list[dict]](root=[entry.model_dump(mode='python') for entry in left.export_entries()])
    right_model = RootModel[list[dict]](root=[entry.model_dump(mode='python') for entry in right.export_entries()])
    assert canonical_json_bytes(left_model) == canonical_json_bytes(right_model)
