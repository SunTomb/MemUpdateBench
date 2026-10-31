"""Offline synthetic runner-bound evidence, never genuine live observations."""
import asyncio
import json

import pytest

from scripts import vnext_finalize_historical_mechanism as finalizer
from scripts import vnext_run_historical_mechanism as runner
from tests.vnext.test_historical_mechanism_runner import cases, environment, envelope


def fixture(tmp_path, *, cross=False, history='correct'):
    runtime, graph, driver, raw, _ = environment(tmp_path)
    original = graph.add_episode
    async def add(**kwargs):
        result = await original(**kwargs)
        group = kwargs['group_id']
        position = int(kwargs['name'][-1])
        is_cross = group.endswith('cross_entity_same_relation')
        values = ['target-alpha'] if is_cross else ['target-alpha', 'target-beta', 'target-gamma'][:position+1]
        episodes = [r for r in driver.rows[group] if r['kind'] == 'Episodic']
        nodes = [{'kind': 'Entity', 'properties': {'uuid': 'node-'+name, 'name': name, 'group_id': group},
                  'linked_episode_uuids': [], 'linked_entity_uuids': []} for name in ['Target-00', *values]]
        edges = []
        for i, value in enumerate(values):
            end = f'2025-01-0{i+2}T00:00:00Z' if i < len(values)-1 else None
            if is_cross and cross and position >= 1:
                end = '2025-01-02T00:00:00Z'
            edges.append({'kind': 'RELATES_TO', 'properties': {'uuid': 'edge-'+value,
                'group_id': group, 'name': 'RELATION', 'fact': f"Target-00's relation is {value}.",
                'valid_at': f'2025-01-0{i+1}T00:00:00Z', 'invalid_at': end,
                'created_at': None, 'expired_at': None}, 'source_uuid': 'node-Target-00',
                'target_uuid': 'node-'+value, 'source_name': 'Target-00', 'target_name': value,
                'source_group_id': group, 'target_group_id': group})
        driver.rows[group] = episodes+nodes+edges
        return result
    graph.add_episode = add
    async def search(**kwargs):
        group = kwargs['group_ids'][0]
        when = kwargs['search_filter'][1].isoformat().replace('+00:00', 'Z')
        historical = when == '2025-01-02T00:00:00Z'
        picked = [r for r in driver.rows[group] if r['kind'] == 'RELATES_TO'
                  and r['properties']['valid_at'] <= when
                  and (r['properties']['invalid_at'] is None or r['properties']['invalid_at'] > when)]
        edges = [{**r['properties'], 'source_node_uuid': r['source_uuid'],
                  'target_node_uuid': r['target_uuid']} for r in picked]
        if history == 'retrieval' and group.endswith('historical_serving') and historical:
            edges = []
        from tests.vnext.test_historical_mechanism_runner import Result
        return Result({'edges': edges, 'edge_reranker_scores': [1.0]*len(edges)})
    graph.search_ = search
    async def complete(request, role):
        query = request.query.text
        answer = 'target-beta' if '2025-01-02' in query else ('target-alpha' if 'cross_entity' in request.query.query_id else 'target-gamma')
        if history == 'readout' and 'historical_serving-as_of' in request.query.query_id:
            answer = 'wrong'
        response = envelope(len(runtime.recorder.rows)+1, json.dumps({'disposition': 'answered', 'answer': answer}))
        runtime.recorder.record_response(response, role=role, schema_name='VersionTransferAnswer')
        return response
    runtime.complete = complete
    report = asyncio.run(runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=10))
    assert report['status'] == 'OBSERVATIONS_COMPLETE'
    artifacts = []
    for item in raw:
        if isinstance(item, tuple):
            artifacts.append(item)
        elif 'namespace' in item:
            artifacts.append(('state', item))
    report.update(owned_process_cleanup=[True, True], ports_closed={'17687': True, '18043': True},
                  owned_gpu_process_absent=True, elapsed_seconds=1.5, output_bytes=1234,
                  accounting={'llm': {'all_settled': True, 'charged': {'attempts': len(runtime.recorder.rows),
                  'input_tokens': len(runtime.recorder.rows)*20, 'output_tokens': len(runtime.recorder.rows)*2}},
                  'embeddings': {'attempts': 0, 'items': 0, 'input_tokens': 0}, 'actual_http_attempts': len(runtime.recorder.rows)})
    return report, artifacts, runtime.recorder.rows


@pytest.mark.parametrize('cross,history,label', [
    (False, 'correct', 'NO_DISTINGUISHING_SIGNAL'),
    (True, 'correct', 'CROSS_ENTITY_INVALIDATION_LEAD'),
    (False, 'retrieval', 'HISTORICAL_SERVING_LEAD'),
    (False, 'readout', 'HISTORICAL_SERVING_LEAD'),
    (True, 'retrieval', 'CROSS_ENTITY_AND_HISTORICAL_LEAD')])
def test_actual_runner_shapes_all_positive_labels(tmp_path, cross, history, label):
    report, raw, obs = fixture(tmp_path, cross=cross, history=history)
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == label
    assert result['cross_entity_transitions'] == int(cross)
    assert result['historical_subtype'] == (history if history != 'correct' else None)
    assert len(result['cases']) == 3
    assert result['counts'] == report['counts']
    assert result['semantic_input_patterns'] == 2
    assert result['independent_semantic_cores'] is None
    assert result['scientific_release_allowed'] is False
    assert result['independent_signer'] is False
    assert result['validation_tasks'] == 0
    assert all(row['state_accuracy'] is None for row in result['cases'])
    assert all(row['layers']['state'] == 'INTERPRETABLE' for row in result['cases'])


def test_missing_final_state_blocks_with_null_not_zero(tmp_path):
    report, raw, obs = fixture(tmp_path)
    pin = report['cases'][2]['snapshots'][-1]['sha256']
    raw = [(kind, value) for kind, value in raw if not (kind == 'state' and value['sha256'] == pin)]
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == 'BLOCKED'
    assert result['historical_subtype'] is None
    assert result['cross_entity_transitions'] is None
    assert result['cases'][2]['reason'] == 'DIRECT_STATE_UNAVAILABLE'


@pytest.mark.parametrize('field,value', [('invalid_at', 'not-time'), ('target_name', 'ambiguous')])
def test_ambiguous_or_malformed_state_blocks(tmp_path, field, value):
    report, raw, obs = fixture(tmp_path)
    state = next(v for k, v in raw if k == 'state' and any(r['kind'] == 'RELATES_TO' for r in v['rows']))
    edge = next(r for r in state['rows'] if r['kind'] == 'RELATES_TO')
    edge[field] = value
    assert finalizer.classify(report, cases(), raw, obs)['label'] == 'BLOCKED'


def replace_state(report, artifacts, select, mutate):
    """Re-sign test-only state as the actual witness would, preserving receipt bindings."""
    from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness as W
    changes = {}
    for kind, snapshot in artifacts:
        if kind != 'state' or not select(snapshot):
            continue
        old = snapshot['sha256']
        mutate(snapshot['rows'])
        for row in snapshot['rows']:
            row['properties_sha256'] = W._value_hash(row['properties'])
            if 'fact' in row['properties']:
                row['fact_sha256'] = W._value_hash(row['properties']['fact'])
        compact = [{k: v for k, v in row.items() if k != 'properties'} for row in snapshot['rows']]
        compact.sort(key=lambda row: (row['kind'], str(row.get('uuid')), W._encoded(row)))
        snapshot['sha256'] = runner.digest({'namespace': snapshot['namespace'], 'rows': compact, 'anomalies': []})
        changes[old] = (snapshot['sha256'], len(snapshot['rows']))
    for row in report['cases']:
        for snapshot in row['snapshots']:
            if snapshot['sha256'] in changes:
                snapshot['sha256'], snapshot['rows'] = changes[snapshot['sha256']]
        for query in row['queries']:
            if query['final_snapshot_sha256'] in changes:
                query['final_snapshot_sha256'] = changes[query['final_snapshot_sha256']][0]


def test_missing_stored_historical_edge_is_absent_witness_not_deletion_claim(tmp_path):
    report, raw, obs = fixture(tmp_path)
    pin = report['cases'][2]['snapshots'][-1]['sha256']
    replace_state(report, raw, lambda s: s['sha256'] == pin,
                  lambda rows: rows.__setitem__(slice(None), [r for r in rows if r.get('uuid') != 'edge-target-beta']))
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == 'NO_DISTINGUISHING_SIGNAL'
    assert result['cases'][2]['queries'][1]['status'] == 'ABSENT_TARGET_ALIGNED_WITNESS'


def test_first_seen_invalid_is_unknown_not_transition(tmp_path):
    report, raw, obs = fixture(tmp_path, cross=True)
    def invalidate(rows):
        for row in rows:
            if row['kind'] == 'RELATES_TO':
                row['invalid_at'] = row['properties']['invalid_at'] = '2025-01-02T00:00:00Z'
    first = report['cases'][1]['snapshots'][0]['sha256']
    replace_state(report, raw, lambda s: s['sha256'] == first, invalidate)
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == 'BLOCKED'
    assert result['cross_entity_transitions'] is None
    assert result['reason'] == 'CROSS_INITIAL_TARGET_UNKNOWN'


@pytest.mark.parametrize('field,value,reason', [('invalid_at', 'bad-time', 'MALFORMED_CASE_EVIDENCE'),
    ('target_name', 'not-exact', 'TARGET_ALIGNMENT_UNKNOWN')])
def test_resigned_malformed_or_ambiguous_state_not_gold_filled(tmp_path, field, value, reason):
    report, raw, obs = fixture(tmp_path)
    first = report['cases'][0]['snapshots'][0]['sha256']
    def mutate(rows):
        for row in rows:
            if row['kind'] == 'RELATES_TO':
                row[field] = value
                if field in row['properties']:
                    row['properties'][field] = value
    replace_state(report, raw, lambda s: s['sha256'] == first, mutate)
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == 'BLOCKED'
    assert result['cases'][0]['reason'] == reason


def test_partial_preserves_all_rows_and_counts(tmp_path):
    report, raw, obs = fixture(tmp_path)
    report['status'] = 'BLOCKED'
    report['cases'][1]['status'] = 'BLOCKED'
    report['cases'][2]['status'] = 'NOT_RUN'
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['label'] == 'BLOCKED'
    assert [r['status'] for r in result['cases']] == ['OBSERVED', 'BLOCKED', 'NOT_RUN']
    assert result['cases'][2]['state_accuracy'] is None


@pytest.mark.parametrize('failure', ['extraction', 'cleanup', 'accounting', 'projection', 'membership'])
def test_evidence_qualification_gaps_block(tmp_path, failure):
    report, raw, obs = fixture(tmp_path)
    if failure == 'extraction':
        obs[0]['status'] = 'BLOCKED'
    elif failure == 'cleanup':
        report['cases'][0]['namespace_empty_after_cleanup'] = False
    elif failure == 'accounting':
        report['accounting'] = None
    elif failure == 'projection':
        report['cases'][0]['readout_projection']['projected_task_sha256'] = '0'*64
    else:
        report['cases'][0]['task_hash'] = '0'*64
    assert finalizer.classify(report, cases(), raw, obs)['label'] == 'BLOCKED'


def test_malformed_created_time_blocks_even_when_world_interval_is_valid(tmp_path):
    report, raw, obs = fixture(tmp_path)
    first = report['cases'][0]['snapshots'][0]['sha256']
    def mutate(rows):
        for row in rows:
            if row['kind'] == 'RELATES_TO':
                row['created_at'] = row['properties']['created_at'] = 'not-a-time'
    replace_state(report, raw, lambda s: s['sha256'] == first, mutate)
    assert finalizer.classify(report, cases(), raw, obs)['label'] == 'BLOCKED'


def test_public_rows_preserve_cleanup_projection_and_unavailable_internal_counts(tmp_path):
    report, raw, obs = fixture(tmp_path)
    report['cases'][1]['namespace_empty_after_cleanup'] = False
    result = finalizer.classify(report, cases(), raw, obs)
    assert result['cases'][1]['namespace_empty_after_cleanup'] is False
    assert result['cases'][0]['readout_projection'] == report['cases'][0]['readout_projection']
    assert result['cases'][0]['internal_accounting']['llm_calls'] is None


def test_wrong_observed_usage_total_blocks(tmp_path):
    report, raw, obs = fixture(tmp_path)
    report['accounting']['llm']['charged']['input_tokens'] += 1
    assert finalizer.classify(report, cases(), raw, obs)['label'] == 'BLOCKED'
