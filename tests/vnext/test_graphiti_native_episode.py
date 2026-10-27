from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mub.vnext.external.providers.graphiti_native_episode import GraphitiNativeEpisode
from mub.vnext.external.visibility import ProviderEventInputV1, ProviderQueryInputV1


class Client:
    def __init__(self):
        self.writes = []
        self.searches = []
        self.closed = False

    async def add_episode(self, **kwargs):
        self.writes.append(kwargs)
        return SimpleNamespace(edges=[])

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return [SimpleNamespace(uuid='native-old', fact='fact', valid_at=None, invalid_at=None),
                SimpleNamespace(uuid='native-new', fact='fact2', valid_at=None, invalid_at=None)]

    async def close(self):
        self.closed = True


def run(awaitable):
    import asyncio
    return asyncio.run(awaitable)


def test_public_episode_only_and_native_order_preserved():
    client = Client()
    bridge = GraphitiNativeEpisode(client, namespace='owned-study-1',
                                  source_type='text', filter_factory=lambda when: {'as_of': when})
    event = ProviderEventInputV1(event_id='event-1', sequence_index=0,
                                logical_time='2025-01-01T00:00:00Z',
                                raw_text='Effective 2025-01-01T00:00:00Z, Pat lives in Oslo.',
                                runtime_namespace='owned-study-1')
    run(bridge.ingest(event))
    write = client.writes[0]
    assert write['episode_body'] == event.raw_text
    assert write['reference_time'] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert set(write) == {'name', 'episode_body', 'source_description', 'reference_time', 'source', 'group_id'}
    query = ProviderQueryInputV1(query_id='q', query_text='As of 2025-01-02T00:00:00Z, where does Pat live?',
                                 k=8, runtime_namespace='owned-study-1')
    edges = run(bridge.retrieve(query))
    assert [x.uuid for x in edges] == ['native-old', 'native-new']
    assert client.searches[0]['group_ids'] == ['owned-study-1']
    assert client.searches[0]['query'] == query.query_text
    assert bridge.counts == {'episode_attempts': 1, 'retrieval_attempts': 1}
    run(bridge.close())
    assert client.closed


def test_missing_public_time_and_foreign_namespace_fail_before_client():
    client = Client()
    bridge = GraphitiNativeEpisode(client, namespace='owned', source_type='text', filter_factory=lambda x: x)
    query = ProviderQueryInputV1(query_id='q', query_text='Where is Pat?', k=8, runtime_namespace='owned')
    with pytest.raises(ValueError, match='public as-of'):
        run(bridge.retrieve(query))
    query = query.model_copy(update={'runtime_namespace': 'someone-else'})
    with pytest.raises(ValueError, match='namespace'):
        run(bridge.retrieve(query))
    assert not client.searches


def test_native_filter_factory_uses_validity_interval_not_gold_version():
    class Filter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    factory = GraphitiNativeEpisode.validity_filter_factory(Filter, Filter, SimpleNamespace(
        less_than_equal='<=', greater_than='>', is_null='IS NULL'))
    when = datetime(2025, 1, 2, tzinfo=timezone.utc)
    filt = factory(when)
    assert filt.valid_at[0][0].date == when
    assert filt.valid_at[0][0].comparison_operator == '<='
    assert filt.invalid_at[0][0].comparison_operator == 'IS NULL'
    assert filt.invalid_at[1][0].comparison_operator == '>'
    assert filt.invalid_at[1][0].date == when
