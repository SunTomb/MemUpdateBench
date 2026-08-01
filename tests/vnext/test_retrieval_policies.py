from __future__ import annotations

from mub.vnext.adapters.retrieval import latest_per_object, normal_topk
from mub.vnext.contracts import AnswerSchema, EvaluationMode, MemoryObjectKey, MemoryQuery, QueryType
from mub.vnext.contracts.runtime import MemoryEntryRecord


def _query(key):
    return MemoryQuery(
        query_id='q1',
        query_type=QueryType.CURRENT_STATE,
        text='where is alex now',
        target_object_keys=[key],
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )


def _entry(entry_id, key, value, index):
    return MemoryEntryRecord(
        entry_id=entry_id,
        content=f'{key.entity}.{key.attribute}={value}',
        object_key_candidate=key,
        value_candidate=value,
        source_event_ids=[f'e{index}'],
        version_index=index,
        raw_metadata={'sequence_index': index},
    )


def test_normal_topk_ranks_without_rewriting():
    key = MemoryObjectKey(object_type='slot', entity='alex', attribute='location')
    entries = [_entry('old', key, 'Dalian', 0), _entry('new', key, 'Qingdao', 1)]
    result = normal_topk(entries, _query(key), 1)
    assert [entry.entry_id for entry in result.entries] == ['new']
    assert result.raw_result['retrieval_rewrite'] is False
    assert result.raw_result['not_original_topk_filter'] is False
    assert result.raw_result['full_store_scan'] is False


def test_latest_per_object_rewrites_after_full_scan_and_uses_exact_identity():
    key = MemoryObjectKey(object_type='slot', namespace='ns', entity='alex', attribute='location', subkey='home')
    other_type = key.model_copy(update={'object_type': 'profile'})
    entries = [
        _entry('old', key, 'Dalian', 0),
        _entry('new', key, 'Qingdao', 1),
        _entry('other-type', other_type, 'Berlin', 99),
    ]
    result = latest_per_object(entries, _query(key), 2)
    assert [entry.entry_id for entry in result.entries] == ['other-type']
    assert result.raw_result['retrieval_rewrite'] is True
    assert result.raw_result['not_original_topk_filter'] is True
    assert result.raw_result['full_store_scan'] is True
    assert result.raw_result['metadata']['policy'] == 'latest_per_object'
