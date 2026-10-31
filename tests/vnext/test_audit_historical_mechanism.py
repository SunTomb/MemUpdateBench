"""Offline injected Task 3 evidence only; no genuine observations or services."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import vnext_finalize_historical_mechanism as finalizer
from scripts import vnext_run_historical_mechanism as runner
from tests.vnext import test_historical_mechanism_analysis as analysis
from tests.vnext.test_historical_mechanism_decision import write
from tests.vnext.test_historical_mechanism_runner import environment, cases, PACKET
from mub.vnext.runtime.version_transfer_budget import StudyBudget
from mub.vnext.runtime.version_transfer_embedding import LocalEmbeddingBoundary


def auditor():
    from scripts import vnext_audit_historical_mechanism
    return vnext_audit_historical_mechanism


def immediate(coro):
    """Passive hooks do not suspend for these already-buffered responses."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError('unexpected I/O in injected transport')


def archive(tmp_path, monkeypatch, *, failure=None, embeddings=False, negative=False, two_edges=False, sdk_defaults=False):
    """Execute existing fixtures with actual observer/budget/private boundaries.

    Older analysis fixtures omit full episode fields and transport. Supply those
    at the injected runtime boundary, never by filling an old run's artifacts.
    """
    import httpx
    holder, prompt = {}, {}
    for name in ('build_readout', 'oracle_request'):
        original = getattr(runner, name)
        def capture(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            prompt['request'] = result
            return result
        monkeypatch.setattr(runner, name, capture)

    def instrument(path, **kwargs):
        runtime, graph, driver, raw, readouts = environment(path, **kwargs)
        holder.update(runtime=runtime, raw=raw)
        budget = StudyBudget(path/'llm-calls.jsonl', **runner.LIMITS)
        holder['budget'] = budget
        backend = SimpleNamespace(dimension=2, max_tokens=10, token_count=lambda text: 3,
                                  encode=lambda texts: [[0.1, 0.2] for _ in texts])
        embedding = LocalEmbeddingBoundary(backend, path/'embedding-calls.jsonl', model_id=runner.EMBEDDING_ID)
        holder['embedding'] = embedding
        runtime.recorder._private_sink = lambda value: raw.append(('observation', value))
        observer = runner.TransportObserver(runtime.recorder, private_sink=runtime.private_sink)
        runtime.observer = observer
        original_record = runtime.recorder.record_response
        original_add = graph.add_episode
        async def add(**kw):
            result = await original_add(**kw)
            event = {'uuid': 'provider-'+kw['name'], 'name': kw['name'], 'group_id': kw['group_id'],
                     'valid_at': kw['reference_time'].isoformat().replace('+00:00', 'Z'),
                     'content': kw['episode_body']}
            result.value['episode'] = event
            for row in driver.rows[kw['group_id']]:
                if row['kind'] == 'Episodic' and row['properties']['name'] == kw['name']:
                    row['properties'].update(event)
            return result
        graph.add_episode = add
        def record(response, *, role, schema_name, **kw):
            op = observer.operation
            if embeddings and not observer.embedding_attempts:
                body = {'model': runner.EMBEDDING_ID, 'input': ['injected text']}
                request = httpx.Request('POST', 'http://127.0.0.1:18043/v1/embeddings',
                                        json=body, headers={'X-MUB-Role': 'graphiti'})
                immediate(observer.on_request(request))
                payload = embedding.create(body, role='graphiti')
                immediate(observer.on_response(httpx.Response(200, json=payload, request=request)))
            if op['kind'] == 'readout':
                text = prompt['request'].rendered_prompt
            elif op['kind'] == 'event':
                text = cases()[op['case']].events[op['position']].public_payload['raw_text']
            else:
                text = cases()[op['case']].queries[op['position']].public_payload['query_text']
            body = {'model': runner.MODEL_ID, 'messages': [{'role': 'user', 'content': text}],
                    'response_format': {'type': 'json_schema', 'json_schema': {
                        'name': schema_name, 'schema': runner.ANSWER_SCHEMA if role != 'graphiti' else {'type': 'object'}}}}
            ticket = budget.reserve(role=role, prompt_tokens=response['usage']['prompt_tokens'], max_output_tokens=128)
            assert response['id'] == f'study-attempt-{ticket}'
            request = httpx.Request('POST', 'http://127.0.0.1:18043/v1/chat/completions',
                                    json=body, headers={'X-MUB-Role': role})
            immediate(observer.on_request(request))
            budget.settle(ticket, completion_tokens=response['usage']['completion_tokens'], finish_reason='stop')
            runtime.recorder.record_response = original_record
            try:
                immediate(observer.on_response(httpx.Response(200, json=response, request=request)))
            finally:
                runtime.recorder.record_response = record
            if sdk_defaults:
                # SDK model_dump adds null defaults; the passive hook retains wire JSON.
                response['service_tier'] = None
                response['usage']['completion_tokens_details'] = None
                response['choices'][0]['message']['refusal'] = None
            return runtime.recorder.rows[-1]
        runtime.recorder.record_response = record
        return runtime, graph, driver, raw, readouts

    original_execute = runner.execute_cases
    async def execute(bound_cases, runtime, **kw):
        original_add, original_search = runtime.graph.add_episode, runtime.graph.search_
        async def add(**args):
            if failure == 'pending' and args['name'].endswith('1'):
                ticket = holder['budget'].reserve(role='graphiti', prompt_tokens=20, max_output_tokens=128)
                import httpx
                request = httpx.Request('POST', 'http://127.0.0.1:18043/v1/chat/completions',
                    json={'model':runner.MODEL_ID, 'messages':[], 'response_format':{'type':'json_schema',
                        'json_schema':{'name':'ExtractedEntities', 'schema':{'type':'object'}}}},
                    headers={'X-MUB-Role':'graphiti'})
                await runtime.observer.on_request(request)
                assert ticket == 2
                raise TimeoutError('injected pending response')
            result = await original_add(**args)
            if negative:
                for row in runtime.driver.rows[args['group_id']]:
                    if row['properties'].get('uuid') == 'edge-target-beta':
                        row['properties']['fact'] = "Target-00's relation is not target-beta."
            return result
        async def search(**args):
            result = await original_search(**args)
            if two_edges and result.value['edges']:
                other = deepcopy(result.value['edges'][0])
                other.update(uuid='unrelated-native-edge', fact='an authentic unrelated native fact')
                result.value['edges'].append(other)
                result.value['edge_reranker_scores'].append(0.25)
            return result
        runtime.graph.add_episode, runtime.graph.search_ = add, search
        return await original_execute(bound_cases, runtime, **kw)
    monkeypatch.setattr(runner, 'execute_cases', execute)
    monkeypatch.setattr(analysis, 'environment', instrument)
    if failure:
        runtime, _, _, raw, _ = instrument(tmp_path, fail_write=2 if failure == 'midwrite' else 0,
                                          fail_snapshot=failure == 'startup')
        report = asyncio.run(runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=10))
        artifacts = [item if isinstance(item, tuple) else ('state', item) for item in raw]
        observations = runtime.recorder.rows
    else:
        report, artifacts, observations = analysis.fixture(tmp_path)
    runtime = holder['runtime']
    holder['budget'].close()
    holder['embedding'].close()
    report.update(owned_process_cleanup=[True, True], ports_closed={'17687': True, '18043': True},
                  owned_gpu_process_absent=True, elapsed_seconds=1.5, output_bytes=1234,
                  llm_ledger_audit=runner.audit_llm_ledger(tmp_path/'llm-calls.jsonl'))
    try:
        report['accounting'] = runner.audit_current(tmp_path, runtime)
    except ValueError as exc:
        report['accounting'] = None
        runner.blocked(report, exc, field='accounting_error_type')
    root = tmp_path/'run'
    index = {}
    for i, (kind, value) in enumerate(artifacts, 1):
        name = f'{i:05d}_{kind}.json'
        index[name] = write(root/'private'/name, value)
    index_pin = write(root/'private/index.json', index)
    obs_pin = write(root/'compact/observations.json', observations)
    trans_pin = write(root/'compact/transport.json', runtime.observer.receipts)
    for name in ('llm-calls.jsonl', 'embedding-calls.jsonl'):
        (root/name).write_bytes((tmp_path/name).read_bytes())
    source_root = tmp_path/'source'
    files = {}
    for name in runner.REQUIRED_SOURCE_FILES:
        target = source_root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'injected offline source archive\n')
        files[name] = runner.sha(target)
    manifest = tmp_path/'source-manifest.json'
    source_pin = write(manifest, {'schema': runner.SOURCE_SCHEMA, 'runtime_revision': 'a'*40, 'files': files})
    report.update(runtime_revision='a'*40, runtime_source_manifest_sha256=source_pin,
        runtime_source_map_sha256=runner.digest(files), packet_index_sha256=runner.PACKET_PIN,
        packet_source_revision=runner.PACKET_SOURCE, private_artifacts={'count': len(index), 'index_sha256': index_pin},
        observations_sha256=obs_pin, transport_sha256=trans_pin)
    terminal_pin = write(root/'terminal.json', report)
    kwargs = dict(run_root=root, terminal_sha256=terminal_pin, packet_root=PACKET,
                  source_manifest=manifest, source_manifest_sha256=source_pin, runtime_revision='a'*40,
                  source_root=source_root, output_path=tmp_path/'audit.json')
    return kwargs, report


def rebind(kwargs, report):
    root = kwargs['run_root']
    index = json.loads((root/'private/index.json').read_bytes())
    for name in index:
        index[name] = runner.sha(root/'private'/name)
    report['private_artifacts']['index_sha256'] = write(root/'private/index.json', index)
    for kind in ('observations', 'transport'):
        if report.get(kind+'_sha256') is not None:
            report[kind+'_sha256'] = runner.sha(root/f'compact/{kind}.json')
    kwargs['terminal_sha256'] = write(root/'terminal.json', report)


def private(kwargs, kind):
    return sorted(kwargs['run_root'].glob(f'private/*_{kind}.json'))


def test_actual_success_receipt_is_exact_and_finalizer_accepts(tmp_path, monkeypatch):
    kwargs, report = archive(tmp_path, monkeypatch, embeddings=True)
    pin = auditor().audit(**kwargs)
    receipt = json.loads(kwargs['output_path'].read_bytes())
    assert runner.sha(kwargs['output_path']) == pin
    assert set(receipt) == {'schema', 'status', 'bindings', 'checks', 'cases'}
    assert receipt['cases'] == finalizer.audit_projection(report)
    finalizer.verify_audit(kwargs['output_path'], pin, receipt['bindings'], report)
    final_args = {k: v for k, v in kwargs.items() if k != 'output_path'}
    result = finalizer.reopen_run(**final_args, audit_receipt=kwargs['output_path'], audit_sha256=pin)
    assert result['label'] == 'NO_DISTINGUISHING_SIGNAL'
    assert all(row['state_accuracy'] is None for row in result['cases'])
    text = kwargs['output_path'].read_text()
    for value in (str(tmp_path), 'Target-00', 'target-alpha', 'rendered_prompt', 'raw_output', 'study-attempt-'):
        assert value not in text
    with pytest.raises(auditor().AuditError, match='OUTPUT'):
        auditor().audit(**kwargs)


@pytest.mark.parametrize('failure', ['midwrite', 'startup', 'pending'])
def test_actual_partial_and_startup_are_auditable_not_success(tmp_path, monkeypatch, failure):
    kwargs, report = archive(tmp_path, monkeypatch, failure=failure)
    pin = auditor().audit(**kwargs)
    assert report['status'] == 'BLOCKED'
    receipt = json.loads(kwargs['output_path'].read_bytes())
    assert receipt['cases'] == finalizer.audit_projection(report)
    result = finalizer.reopen_run(**{k:v for k,v in kwargs.items() if k != 'output_path'},
                                 audit_receipt=kwargs['output_path'], audit_sha256=pin)
    assert result['label'] == 'BLOCKED'
    assert result['cases'][2]['status'] == 'NOT_RUN'


@pytest.mark.parametrize('kind,field', [
    ('mutation', 'event'), ('mutation', 'time'), ('mutation', 'namespace'),
    ('state', 'hash'), ('state', 'properties'), ('state', 'vector'),
    ('retrieval', 'order'), ('retrieval', 'score'),
    ('readout', 'prompt'), ('readout', 'prediction'), ('readout', 'call_id'),
    ('observation', 'usage'), ('transport', 'role'), ('canonical_run', 'prediction')])
def test_resigned_artifact_inconsistencies_fail_closed(tmp_path, monkeypatch, kind, field):
    kwargs, report = archive(tmp_path, monkeypatch)
    paths = private(kwargs, kind)
    path = next((p for p in paths if kind != 'state' or json.loads(p.read_bytes())['rows']), paths[0])
    payload = json.loads(path.read_bytes())
    if kind == 'mutation':
        payload['episode'][{'event':'name', 'time':'valid_at', 'namespace':'group_id'}[field]] = 'tampered'
    elif kind == 'state':
        if field == 'hash':
            payload['sha256'] = '0'*64
        elif field == 'properties':
            payload['rows'][0]['properties']['name'] = 'tampered'
        else:
            payload['rows'][0]['properties']['name_embedding_sha256'] = '0'*64
    elif kind == 'retrieval':
        if field == 'order':
            payload['edges'].append(deepcopy(payload['edges'][0]))
            payload['edge_reranker_scores'].append(1.0)
        else:
            payload['edge_reranker_scores'][0] = 0.5
    elif kind == 'readout':
        if field == 'prompt':
            payload['request']['rendered_prompt'] += 'tampered'
        elif field == 'prediction':
            payload['prediction']['parsed_answer'] = 'tampered'
        else:
            payload['response']['id'] = 'study-attempt-999'
    elif kind == 'observation':
        payload['response']['usage']['completion_tokens'] += 1
    elif kind == 'transport':
        payload['operation']['case'] = 2
    else:
        payload['answer_predictions'][0]['parsed_answer'] = 'tampered'
    write(path, payload)
    rebind(kwargs, report)
    with pytest.raises(auditor().AuditError):
        auditor().audit(**kwargs)
    assert not kwargs['output_path'].exists()


@pytest.mark.parametrize('failure', ['usage', 'unsettled', 'cleanup', 'missing', 'unexpected',
    'source_pin', 'source_bytes', 'packet', 'terminal', 'projection', 'counts', 'absent_compact'])
def test_ledger_cleanup_membership_and_source_gates(tmp_path, monkeypatch, failure):
    kwargs, report = archive(tmp_path, monkeypatch)
    root = kwargs['run_root']
    if failure in ('usage', 'unsettled'):
        path = root/'llm-calls.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if failure == 'usage':
            rows[0]['input_tokens'] += 1
        else:
            rows.pop(1)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif failure == 'cleanup':
        report['cases'][0]['namespace_empty_after_cleanup'] = False
    elif failure == 'missing':
        private(kwargs, 'state')[0].unlink()
    elif failure == 'unexpected':
        write(root/'private/99999_credentials.json', {'raw': 'private'})
    elif failure == 'source_pin':
        kwargs['source_manifest_sha256'] = '0'*64
    elif failure == 'source_bytes':
        (kwargs['source_root']/runner.REQUIRED_SOURCE_FILES[0]).write_bytes(b'changed')
    elif failure == 'packet':
        kwargs['packet_root'] = tmp_path/'missing-packet'
    elif failure == 'terminal':
        kwargs['terminal_sha256'] = '0'*64
    elif failure == 'projection':
        report['cases'][0]['readout_projection']['policy'] = 'tampered'
    elif failure == 'counts':
        report['cases'][0]['event_attempts'] = 0
    else:
        report['observations_sha256'] = None
    if failure in ('cleanup', 'projection', 'counts', 'absent_compact'):
        kwargs['terminal_sha256'] = write(root/'terminal.json', report)
    with pytest.raises(auditor().AuditError):
        auditor().audit(**kwargs)
    assert not kwargs['output_path'].exists()


def test_cannot_write_inside_old_run(tmp_path, monkeypatch):
    kwargs, _ = archive(tmp_path, monkeypatch)
    kwargs['output_path'] = kwargs['run_root']/'audit.json'
    with pytest.raises(auditor().AuditError, match='OUTPUT'):
        auditor().audit(**kwargs)


def test_authentic_negative_fact_is_integrity_pass_but_classifier_blocked(tmp_path, monkeypatch):
    kwargs, report = archive(tmp_path, monkeypatch, negative=True)
    pin = auditor().audit(**kwargs)
    result = finalizer.reopen_run(**{k:v for k,v in kwargs.items() if k != 'output_path'},
                                 audit_receipt=kwargs['output_path'], audit_sha256=pin)
    assert result['label'] == 'BLOCKED'
    assert result['cases'][2]['reason'] == 'TARGET_ASSERTION_UNKNOWN'
    assert 'not target-beta' not in kwargs['output_path'].read_text()


def test_true_two_uuid_native_order_reversal_is_not_silently_accepted(tmp_path, monkeypatch):
    kwargs, report = archive(tmp_path, monkeypatch, two_edges=True)
    path = private(kwargs, 'retrieval')[0]
    native = json.loads(path.read_bytes())
    assert len(native['edges']) == 2
    native['edges'].reverse()
    native['edge_reranker_scores'].reverse()
    old_hash = runner.digest(json.loads(path.read_bytes()))
    write(path, native)
    report['cases'][0]['queries'][0]['retrieval_sha256'] = runner.digest(native)
    assert old_hash != runner.digest(native)
    rebind(kwargs, report)
    with pytest.raises(auditor().AuditError):
        auditor().audit(**kwargs)


@pytest.mark.parametrize('failure', ['extra_state', 'missing_query_snapshot', 'missing_client', 'missing_error', 'extra_compact'])
def test_no_unaccounted_state_or_cleanup_or_compact_payloads(tmp_path, monkeypatch, failure):
    kwargs, report = archive(tmp_path, monkeypatch, failure='midwrite' if failure == 'missing_error' else None)
    root = kwargs['run_root']
    if failure == 'extra_state':
        index = json.loads((root/'private/index.json').read_bytes())
        name = f'{len(index)+1:05d}_state.json'
        payload = json.loads(private(kwargs, 'state')[-1].read_bytes())
        index[name] = write(root/'private'/name, payload)
        report['private_artifacts'] = {'count':len(index), 'index_sha256':write(root/'private/index.json', index)}
    elif failure == 'missing_query_snapshot':
        # Remove one duplicated final-state witness and renumber the actual stream.
        path = private(kwargs, 'retrieval')[0]
        state = next(p for p in private(kwargs, 'state') if p.name > path.name)
        state.unlink()
        items = sorted(p for p in (root/'private').iterdir() if p.name != 'index.json')
        index = {}
        for i, path in enumerate(items, 1):
            name = f'{i:05d}_'+path.name.split('_', 1)[1]
            payload = json.loads(path.read_bytes())
            path.unlink()
            index[name] = write(root/'private'/name, payload)
        report['private_artifacts'] = {'count':len(index), 'index_sha256':write(root/'private/index.json', index)}
    elif failure == 'missing_client':
        report.pop('client_cleanup_errors')
    elif failure == 'missing_error':
        report['cases'][0]['namespace_empty_after_cleanup'] = False
        report.pop('client_cleanup_errors')
    else:
        write(root/'compact/credentials.json', {'private':'payload'})
    kwargs['terminal_sha256'] = write(root/'terminal.json', report)
    with pytest.raises(auditor().AuditError):
        auditor().audit(**kwargs)


def rewrite_episode_snapshots(kwargs, report, *, field, value, final_only=False, first_only=False):
    paths = sorted(p for p in (kwargs['run_root']/'private').iterdir() if p.name != 'index.json')
    artifacts = [(path.name.split('_', 1)[1][:-5], json.loads(path.read_bytes())) for path in paths]
    case = cases()[0]
    final_hash = report['cases'][0]['snapshots'][-1]['sha256']
    first_id = case.events[0].public_payload['event_id']
    def mutate(rows):
        for row in rows:
            if row['kind'] == 'Episodic' and (not first_only or row['name'] == first_id):
                row['properties'][field] = value
                if field == 'valid_at':
                    row[field] = value
    analysis.replace_state(report, artifacts,
        lambda snapshot: snapshot['namespace'] == case.runtime_namespace
            and (not final_only or snapshot['sha256'] == final_hash), mutate)
    for path, (_, payload) in zip(paths, artifacts):
        write(path, payload)
    rebind(kwargs, report)


@pytest.mark.parametrize('field,value', [('valid_at', '2030-01-01T00:00:00Z'), ('content', 'tampered persisted episode text')])
@pytest.mark.parametrize('final_only', [False, True])
def test_rehashed_episode_time_and_content_cannot_diverge_from_mutation_and_public_event(tmp_path, monkeypatch, field, value, final_only):
    kwargs, report = archive(tmp_path, monkeypatch)
    rewrite_episode_snapshots(kwargs, report, field=field, value=value,
                              final_only=final_only, first_only=final_only)
    with pytest.raises(auditor().AuditError, match='SNAPSHOT_EPISODE_BINDING'):
        auditor().audit(**kwargs)
    assert not kwargs['output_path'].exists()


def test_matching_episode_snapshot_time_accepts_equivalent_sdk_iso_format(tmp_path, monkeypatch):
    kwargs, report = archive(tmp_path, monkeypatch)
    rewrite_episode_snapshots(kwargs, report, field='valid_at', value='2025-01-01T00:00:00+00:00', first_only=True)
    pin = auditor().audit(**kwargs)
    receipt = json.loads(kwargs['output_path'].read_bytes())
    finalizer.verify_audit(kwargs['output_path'], pin, receipt['bindings'], report)
    assert receipt['cases'] == finalizer.audit_projection(report)


def test_sdk_null_defaults_do_not_change_wire_response_identity(tmp_path, monkeypatch):
    kwargs, _ = archive(tmp_path, monkeypatch, sdk_defaults=True)
    pin = auditor().audit(**kwargs)
    assert pin == runner.sha(kwargs['output_path'])


@pytest.mark.parametrize('failure', ['embedding_usage', 'embedding_duplicate', 'packet_binding', 'time_property', 'transport_call', 'unknown_indexed_kind'])
def test_additional_substantive_bindings(tmp_path, monkeypatch, failure):
    kwargs, report = archive(tmp_path, monkeypatch, embeddings=True)
    root = kwargs['run_root']
    if failure.startswith('embedding_'):
        path = root/'embedding-calls.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if failure == 'embedding_usage':
            rows[0]['input_tokens'] += 1
        else:
            rows.append(rows[-1])
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif failure == 'packet_binding':
        report['packet_index_sha256'] = '0'*64
    elif failure == 'time_property':
        path = next(p for p in private(kwargs, 'state') if any(r['kind'] == 'RELATES_TO' for r in json.loads(p.read_bytes())['rows']))
        state = json.loads(path.read_bytes())
        next(r for r in state['rows'] if r['kind'] == 'RELATES_TO')['valid_at'] = 'not-time'
        write(path, state)
    elif failure == 'transport_call':
        path = root/'compact/transport.json'
        rows = json.loads(path.read_bytes())
        next(r for r in rows if r['kind'] == 'completions')['call_id'] = 'study-attempt-999'
        write(path, rows)
    else:
        index = json.loads((root/'private/index.json').read_bytes())
        name = f'{len(index)+1:05d}_credentials.json'
        index[name] = write(root/'private'/name, {'raw':'private'})
        report['private_artifacts']['count'] = len(index)
        write(root/'private/index.json', index)
    rebind(kwargs, report)
    with pytest.raises(auditor().AuditError):
        auditor().audit(**kwargs)


def test_before_runtime_initial_report_preserves_absent_observations_and_unknown_cleanup(tmp_path, monkeypatch):
    kwargs, old = archive(tmp_path, monkeypatch, failure='startup')
    root = kwargs['run_root']
    report = runner.initial_report(cases())
    runner.blocked(report, RuntimeError())
    for key in ('runtime_revision', 'runtime_source_manifest_sha256', 'runtime_source_map_sha256',
                'packet_index_sha256', 'packet_source_revision', 'elapsed_seconds', 'output_bytes'):
        report[key] = old[key]
    for path in (root/'private').iterdir():
        path.unlink()
    report['private_artifacts'] = {'count':0, 'index_sha256':write(root/'private/index.json', {})}
    for path in (root/'compact').iterdir():
        path.unlink()
    for name in ('llm-calls.jsonl', 'embedding-calls.jsonl'):
        (root/name).unlink()
    report.update(owned_process_cleanup=[False], ports_closed={'17687':False, '18043':True},
                  owned_gpu_process_absent=None, llm_ledger_audit=None)
    kwargs['terminal_sha256'] = write(root/'terminal.json', report)
    pin = auditor().audit(**kwargs)
    receipt = json.loads(kwargs['output_path'].read_bytes())
    assert receipt['bindings']['observations_sha256'] is None
    assert receipt['bindings']['transport_sha256'] is None
    assert receipt['cases'][2]['status'] == 'NOT_RUN'
    assert receipt['cases'][0]['namespace_empty_after_cleanup'] is None
    result = finalizer.reopen_run(**{k:v for k,v in kwargs.items() if k != 'output_path'},
                                 audit_receipt=kwargs['output_path'], audit_sha256=pin)
    assert result['label'] == 'BLOCKED'
    assert result['resources']['finish_reasons'] is None
