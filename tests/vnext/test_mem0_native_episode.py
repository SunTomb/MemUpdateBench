from types import SimpleNamespace

import pytest

from mub.vnext.external.providers.mem0_native_episode import Mem0NativeEpisode, NativeHistoryUnsupported
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1


class Memory:
    def __init__(self):
        self.calls = []

    def add(self, *args, **kwargs):
        self.calls.append(('add', args, kwargs))
        return {'results': [{'id': 'native-id', 'memory': 'native text', 'event': 'ADD'}]}

    def search(self, *args, **kwargs):
        self.calls.append(('search', args, kwargs))
        return {'results': [{'id': 'old', 'score': 0.4}, {'id': 'new', 'score': 0.2}]}

    def get_all(self, **kwargs):
        self.calls.append(('get_all', (), kwargs))
        return {'results': []}

    def delete_all(self, **kwargs):
        self.calls.append(('delete_all', (), kwargs))
        return {'message': 'deleted'}


def test_mem0_ingestion_is_native_inference_without_gold_actions():
    memory = Memory()
    bridge = Mem0NativeEpisode(memory, namespace='owned')
    event = ProviderEventInputV1(event_id='e1',sequence_index=0,logical_time='2025-01-01T00:00:00Z',
                                raw_text='Public update.',runtime_namespace='owned')
    result = bridge.ingest(event)
    assert result['results'][0]['id'] == 'native-id'
    method, args, kwargs = memory.calls[0]
    assert method == 'add' and args == ([{'role':'user','content':'Public update.'}],)
    assert kwargs['infer'] is True and kwargs['user_id'] == 'owned'
    assert 'timestamp' not in kwargs
    assert kwargs['metadata'] == {'mub_source_event_id':'e1','mub_sequence_index':0,
                                  'mub_logical_time':'2025-01-01T00:00:00Z'}


def test_mem0_preserves_native_result_order_and_refuses_unsupported_history():
    memory = Memory()
    bridge = Mem0NativeEpisode(memory, namespace='owned')
    query = ProviderQueryInputV1(query_id='q',query_text='Public question',k=8,runtime_namespace='owned')
    with pytest.raises(NativeHistoryUnsupported):
        bridge.retrieve(query, query_kind='point_in_time')
    assert not memory.calls
    result = bridge.retrieve(query, query_kind='current')
    assert [x['id'] for x in result['results']] == ['old','new']
    assert memory.calls[0][2] == {'top_k':8,'filters':{'user_id':'owned'},'threshold':0.0,'rerank':False}


def test_mem0_namespace_and_shape_checks_do_not_repair_provider_outputs():
    memory = Memory()
    shapes = []
    bridge = Mem0NativeEpisode(memory, namespace='owned', response_shapes=shapes)
    event = ProviderEventInputV1(event_id='e1',sequence_index=0,raw_text='Public update.',runtime_namespace='other')
    with pytest.raises(ValueError, match='namespace'):
        bridge.ingest(event)
    assert not memory.calls
    shapes.append({'is_extraction':True,'valid_memory_list':False})
    event = event.model_copy(update={'runtime_namespace':'owned'})
    original = memory.add
    def invalid(*args, **kwargs):
        shapes.append({'is_extraction':True,'valid_memory_list':False})
        return original(*args, **kwargs)
    memory.add = invalid
    with pytest.raises(ValueError, match='format'):
        bridge.ingest(event)
    assert len(memory.calls) == 1


def test_response_monitor_retains_only_shape_and_hash_not_raw_model_content():
    events = []
    monitor = Mem0NativeEpisode.response_monitor(events, extraction_system_prompt='native-extraction')
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"memory":[]}'))])
    monitor(None,response,{'messages':[{'role':'system','content':'native-extraction'}]})
    assert events[0]['valid_memory_list'] is True
    response.choices[0].message.content = '{"properties":{}}'
    monitor(None,response,{'messages':[{'role':'system','content':'native-extraction'}]})
    assert events[1]['valid_memory_list'] is False
    assert 'content' not in events[0] and 'raw_output' not in events[0]
    for content, expected in (
        ('{"memory":[{"text":"native fact"}]}', True),
        ('{"memory":[{"properties":{}}]}', False),
        ('{"memory":["wrong item type"]}', False),
    ):
        response.choices[0].message.content = content
        monitor(None,response,{'messages':[{'role':'system','content':'native-extraction'}]})
        assert events[-1]['valid_memory_list'] is expected
