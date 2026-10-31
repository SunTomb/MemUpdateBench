import asyncio
import hashlib
import json
from datetime import datetime, timezone

import pytest

from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness


def run(coro):
    return asyncio.run(coro)


class FakeDriver:
    """The actual Graphiti async (records, summary, keys) driver boundary."""

    def __init__(self, rows=(), *, nodes=0, edges=0, anomalies=0):
        self.rows = list(rows)
        self.nodes, self.edges, self.anomalies = nodes, edges, anomalies
        self.calls = []

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if 'RETURN node_count, count(r) AS edge_count' in query:
            rows = [{'node_count': self.nodes, 'edge_count': self.edges}]
        elif 'RETURN anomaly_count' in query:
            rows = [{'anomaly_count': self.anomalies}]
        else:
            rows = self.rows
        return rows, object(), ['unused-key']


def witness(driver, namespace='case-a', **kwargs):
    return GraphitiStateWitness(driver, owned_namespaces={namespace}, **kwargs)


def entity(uuid='e1', **properties):
    return {'kind': 'Entity', 'properties': {'uuid': uuid, 'name': 'Ada',
            'group_id': 'case-a', **properties}, 'linked_episode_uuids': ['p1'],
            'linked_entity_uuids': []}


def edge(**properties):
    return {'kind': 'RELATES_TO', 'properties': {'uuid': 'r1', 'group_id': 'case-a',
            **properties}, 'source_uuid': 'e1', 'target_uuid': 'e2',
            'source_name': 'Ada', 'target_name': 'Lab',
            'source_group_id': 'case-a', 'target_group_id': 'case-a'}


def test_snapshot_awaits_real_tuple_contract_and_preserves_native_fields():
    driver = FakeDriver([
        entity(created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
               name_embedding=[1.0, 2.0], summary='private node prose'),
        {'kind': 'Episodic', 'properties': {'uuid': 'p1', 'name': 'event-1',
         'group_id': 'case-a', 'valid_at': None, 'content': 'private episode'},
         'linked_episode_uuids': [], 'linked_entity_uuids': ['e1']},
        edge(fact='Ada works at Lab', fact_embedding=[0.1, 0.2],
             valid_at='2025-01-01T00:00:00Z', invalid_at=None,
             created_at='2025-01-01T00:00:01Z', expired_at=None, episodes=['p1']),
    ])
    snapshot = run(witness(driver).snapshot('case-a'))
    query, params = driver.calls[0]
    assert params == {'group': 'case-a'} and '$group' in query
    assert 'properties(n)' in query and 'properties(r)' in query
    assert 'MENTIONS' in query
    row = next(row for row in snapshot.rows if row['kind'] == 'RELATES_TO')
    assert row['valid_at'] == '2025-01-01T00:00:00Z'
    assert row['invalid_at'] is None and row['expired_at'] is None
    assert row['source_uuid'] == 'e1' and row['target_name'] == 'Lab'
    assert row['source_group_id'] == row['target_group_id'] == 'case-a'
    assert row['episodes'] == ['p1']
    assert row['fact_sha256'] == hashlib.sha256(b'Ada works at Lab').hexdigest()
    assert row['fact_embedding_sha256']
    assert snapshot.rows[0]['name_embedding_sha256']
    assert snapshot.rows[0]['linked_episode_uuids'] == ['p1']
    serialized = json.dumps(snapshot.rows)
    for secret in ('Ada works at Lab', 'private node prose', 'private episode', '[0.1, 0.2]', '[1.0, 2.0]'):
        assert secret not in serialized
    assert snapshot.anomalies == ()


def test_private_sink_gets_raw_facts_properties_but_never_numeric_vectors():
    private = []
    snapshot = run(witness(FakeDriver([entity(name_embedding=[1.0]),
        edge(fact='private fact', fact_embedding=[0.2])]), private_sink=private.append).snapshot('case-a'))
    assert len(private) == 1 and private[0]['namespace'] == 'case-a'
    serialized = json.dumps(private)
    assert 'private fact' in serialized
    assert '[1.0]' not in serialized and '[0.2]' not in serialized
    assert 'private fact' not in json.dumps(snapshot.rows)
    assert private[0]['sha256'] == snapshot.sha256


def test_snapshot_hash_is_deterministic_and_distinguishes_missing_from_null():
    rows = [entity('b', name_embedding=None), entity('a')]
    rows[0]['linked_episode_uuids'] = ['p2', 'p1']
    first = run(witness(FakeDriver(rows)).snapshot('case-a'))
    rows.reverse()
    rows[1]['linked_episode_uuids'].reverse()
    second = run(witness(FakeDriver(rows)).snapshot('case-a'))
    assert first.rows == second.rows and first.sha256 == second.sha256
    assert 'created_at' not in first.rows[0]  # Missing is not manufactured provider null.
    assert 'name_embedding_sha256' not in first.rows[0]
    assert first.rows[1]['name_embedding_sha256'] is None
    changed = run(witness(FakeDriver([entity('b')])).snapshot('case-a'))
    explicit_null = run(witness(FakeDriver([entity('b', name_embedding=None)])).snapshot('case-a'))
    assert changed.sha256 != explicit_null.sha256


def test_all_owned_edges_including_foreign_endpoints_are_exported_and_flagged():
    foreign = edge(fact='cross-group fact')
    foreign['target_group_id'] = 'foreign'
    driver = FakeDriver([foreign])
    snapshot = run(witness(driver).snapshot('case-a'))
    assert len(snapshot.rows) == 1
    assert snapshot.rows[0]['target_group_id'] == 'foreign'
    assert snapshot.anomalies
    query = driver.calls[0][0]
    assert 'r.group_id = $group' in query
    assert 'OR s.group_id = $group' in query


@pytest.mark.parametrize('method', ['snapshot', 'delete_namespace', 'verify_empty'])
@pytest.mark.parametrize('namespace', ['', '  ', 'x\ny', None, 7, 'foreign'])
def test_rejects_unadmitted_namespace_before_any_driver_call(method, namespace):
    driver = FakeDriver()
    with pytest.raises((TypeError, ValueError)):
        run(getattr(witness(driver), method)(namespace))
    assert not driver.calls


def test_explicit_namespace_admission_is_required_and_can_be_single_string():
    driver = FakeDriver()
    with pytest.raises((TypeError, ValueError)):
        GraphitiStateWitness(driver)
    with pytest.raises(ValueError):
        GraphitiStateWitness(driver, owned_namespaces=set())
    run(GraphitiStateWitness(driver, owned_namespaces='case-a').snapshot('case-a'))


def test_delete_guards_all_relationships_before_non_detach_delete_and_verifies_empty():
    driver = FakeDriver()
    assert run(witness(driver).delete_namespace('case-a')) is None
    query, params = driver.calls[0]
    assert params == {'group': 'case-a'}
    assert 'DETACH' not in query.upper()
    assert '(s)-[r]->(t)' in query
    assert 'anomaly_count = 0' in query
    assert query.index('anomaly_count') < query.index('DELETE')
    assert len(driver.calls) == 2
    assert all(params == {'group': 'case-a'} for _, params in driver.calls)


def test_cleanup_cross_namespace_anomaly_blocks_without_verification_success():
    driver = FakeDriver(anomalies=1)
    with pytest.raises(ValueError, match='cross.namespace'):
        run(witness(driver).delete_namespace('case-a'))
    assert len(driver.calls) == 1


@pytest.mark.parametrize('nodes,edges', [(1, 0), (0, 1), (1, 1)])
def test_verify_empty_detects_both_owned_nodes_and_owned_edges(nodes, edges):
    driver = FakeDriver(nodes=nodes, edges=edges)
    assert run(witness(driver).verify_empty('case-a')) is False
    with pytest.raises(ValueError, match='not empty'):
        run(witness(driver).delete_namespace('case-a'))


def test_verify_empty_zero_counts_are_required():
    assert run(witness(FakeDriver()).verify_empty('case-a')) is True


def test_private_properties_are_reopenably_hash_bound_and_sink_cannot_mutate_compact():
    private = []

    def sink(payload):
        private.append(payload)
        payload['rows'][0]['linked_episode_uuids'].append('sink-only')

    snapshot = run(witness(FakeDriver([entity(name_embedding=[0.1])]),
                           private_sink=sink).snapshot('case-a'))
    assert snapshot.rows[0]['linked_episode_uuids'] == ['p1']
    properties = private[0]['rows'][0]['properties']
    encoded = json.dumps(properties, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    assert snapshot.rows[0]['properties_sha256'] == hashlib.sha256(encoded).hexdigest()


def test_native_neo4j_temporal_objects_use_iso_format_without_replacement():
    class NativeTime:
        def iso_format(self):
            return '2025-01-01T12:00:00.123456789+00:00'

    snapshot = run(witness(FakeDriver([entity(created_at=NativeTime())])).snapshot('case-a'))
    assert snapshot.rows[0]['created_at'] == '2025-01-01T12:00:00.123456789+00:00'


def test_native_episode_entity_edge_ids_are_not_lost_in_compact_evidence():
    episodic = {'kind': 'Episodic', 'properties': {'uuid': 'p1', 'group_id': 'case-a',
                'entity_edges': ['r1', 'r2'], 'valid_at': None}}
    snapshot = run(witness(FakeDriver([episodic])).snapshot('case-a'))
    assert snapshot.rows[0]['entity_edges'] == ['r1', 'r2']
