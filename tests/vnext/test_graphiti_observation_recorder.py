import hashlib
import json

import pytest

from mub.vnext.external.providers.graphiti_observation_recorder import ObservationRecorder


def args():
    return dict(role='graphiti', schema_name='EdgeDuplicate', call_id='study-attempt-7',
        usage={'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
        finish_reason='stop', raw_output='{"duplicate_facts":[],"contradicted_facts":[]}')


def record(recorder=None, **overrides):
    data = args()
    data.update(overrides)
    return (recorder or ObservationRecorder()).record(**data)


def response(**overrides):
    data = {'id': 'study-attempt-7', 'object': 'chat.completion', 'model': 'pinned-local',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{}'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
    data.update(overrides)
    return data


def test_recorder_keeps_exact_redacted_metadata_without_raw_output():
    recorder = ObservationRecorder()
    row = record(recorder, timestamp='2025-01-01T00:00:00Z')
    assert row['schema_name'] == 'EdgeDuplicate' and row['role'] == 'graphiti'
    assert row['call_id'] == 'study-attempt-7'
    assert row['observation_id'] == 1
    assert row['output_sha256'] == hashlib.sha256(args()['raw_output'].encode()).hexdigest()
    assert row['top_level_keys'] == ['contradicted_facts', 'duplicate_facts']
    assert row['parsed_top_level_type'] == 'object' and row['status'] == 'OBSERVED'
    assert row['finish_reason'] == 'stop' and row['usage'] == args()['usage']
    assert row['timestamp'] == '2025-01-01T00:00:00Z'
    assert not ({'raw_output', 'prompt', 'response', 'output'} & row.keys())
    row['usage']['total_tokens'] = 0
    assert recorder.rows[0]['usage']['total_tokens'] == 14


@pytest.mark.parametrize('raw,reason', [
    ('not-json', 'malformed_json'), ('[]', 'nonobject_json'), ('null', 'nonobject_json'),
    ('"secret scalar"', 'nonobject_json'), ('{"n":NaN}', 'malformed_json'),
])
def test_bad_output_retains_redacted_failure_hash_before_raising(raw, reason):
    recorder = ObservationRecorder()
    with pytest.raises(ValueError, match=reason):
        record(recorder, raw_output=raw)
    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert row['status'] == 'BLOCKED' and reason in row['errors']
    assert row['output_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    assert not ({'raw_output', 'response', 'output'} & row.keys())
    if raw not in ('[]', 'null'):
        assert raw not in json.dumps(row)


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2},
    {'prompt_tokens': 1.0, 'completion_tokens': 1, 'total_tokens': 2},
    {'prompt_tokens': '1', 'completion_tokens': 1, 'total_tokens': 2},
    {'prompt_tokens': 1, 'completion_tokens': -1, 'total_tokens': 0},
    {'prompt_tokens': 1}, {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 4}])
def test_rejects_bad_usage_without_losing_the_attempt(usage):
    recorder = ObservationRecorder()
    with pytest.raises(ValueError, match='usage'):
        record(recorder, usage=usage)
    assert recorder.rows[0]['status'] == 'BLOCKED'
    assert recorder.rows[0]['output_sha256']
    assert recorder.rows[0]['usage'] is None


@pytest.mark.parametrize('field,value', [('role', ''), ('schema_name', ''), ('call_id', None),
    ('call_id', ''), ('finish_reason', None), ('finish_reason', 'length'), ('finish_reason', 'tool_calls')])
def test_missing_metadata_or_incomplete_finish_is_not_empty_success(field, value):
    recorder = ObservationRecorder()
    with pytest.raises(ValueError, match=field):
        record(recorder, **{field: value})
    assert recorder.rows[0]['status'] == 'BLOCKED'


def test_duplicate_exact_call_id_is_recorded_as_failure_with_monotonic_observation_id():
    recorder = ObservationRecorder()
    record(recorder)
    with pytest.raises(ValueError, match='duplicate_call_id'):
        record(recorder)
    record(recorder, call_id='study-attempt-9')
    assert [row['observation_id'] for row in recorder.rows] == [1, 2, 3]
    assert [row['call_id'] for row in recorder.rows] == ['study-attempt-7', 'study-attempt-7', 'study-attempt-9']
    assert 'timestamp' not in recorder.rows[0]


def test_attempt_ids_can_be_injected_but_observation_ids_stay_monotonic():
    recorder = ObservationRecorder(attempt_id_factory=iter([10, 11]).__next__)
    assert record(recorder)['attempt_id'] == 10
    assert record(recorder, call_id='study-attempt-8')['attempt_id'] == 11
    assert [row['observation_id'] for row in recorder.rows] == [1, 2]


def test_transport_callback_preserves_local_boundary_response_id_and_usage():
    private = []
    recorder = ObservationRecorder(private_sink=private.append)
    envelope = response()
    row = recorder.record_response(envelope, role='graphiti', schema_name='Entities',
                                   private_material={'prompt': 'private prompt'})
    assert row['call_id'] == envelope['id']
    assert row['usage'] == envelope['usage']
    assert row['output_sha256'] == hashlib.sha256(b'{}').hexdigest()
    assert row['response_sha256']
    assert private[0]['response'] == envelope
    assert private[0]['private_material'] == {'prompt': 'private prompt'}
    assert 'private prompt' not in json.dumps(row)


@pytest.mark.parametrize('envelope', [response(id=None), response(usage=None),
    response(choices=[]), response(choices=[{'message': {'content': '{}'}}]),
    response(choices=[{'finish_reason': 'stop', 'message': {'content': 'broken'}}]),
    response(choices=[{'finish_reason': 'length', 'message': {'content': '{}'}}]),
])
def test_transport_failures_keep_an_envelope_hash_and_block(envelope):
    recorder = ObservationRecorder()
    with pytest.raises(ValueError):
        recorder.record_response(envelope, role='graphiti', schema_name='Entities')
    assert len(recorder.rows) == 1
    assert recorder.rows[0]['response_sha256']
    assert recorder.rows[0]['status'] == 'BLOCKED'


def test_raw_material_is_never_in_compact_rows_even_on_failure():
    private = []
    recorder = ObservationRecorder(private_sink=private.append)
    with pytest.raises(ValueError):
        record(recorder, raw_output='private broken output', private_material={'prompt': 'private prompt'})
    assert private[0]['raw_output'] == 'private broken output'
    assert private[0]['private_material']['prompt'] == 'private prompt'
    assert 'private broken output' not in json.dumps(recorder.rows)
    assert 'private prompt' not in json.dumps(recorder.rows)
