"""Offline historical Task 3 contracts; no SDK/model/driver services are loaded."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import vnext_prepare_historical_mechanism as preparation
from scripts import vnext_run_historical_mechanism as runner
from mub.vnext.external.providers.graphiti_observation_recorder import ObservationRecorder
from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / 'results/vnext/version_transfer_historical_mechanism_preparation_20261030_v2'


def cases():
    return preparation.build_cases(code_revision=runner.PACKET_SOURCE)


def envelope(n=1, text='{}'):
    return {'id': f'study-attempt-{n}', 'model': runner.MODEL_ID, 'choices': [{'finish_reason': 'stop',
            'message': {'content': text}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 2, 'total_tokens': 22}}


class Result:
    def __init__(self, value):
        self.value = value
    def model_dump(self, **_):
        return self.value


class FakeDriver:
    def __init__(self):
        self.rows, self.calls = {}, []
        self._init_task = None
        self.fail_snapshot = False
    async def execute_query(self, query, **kwargs):
        group = kwargs['group']
        self.calls.append((query, group))
        if 'RETURN node_count' in query:
            return ([{'node_count': len(self.rows.get(group, [])), 'edge_count': 0}], None, None)
        if 'RETURN anomaly_count' in query:
            self.rows[group] = []
            return ([{'anomaly_count': 0}], None, None)
        if self.fail_snapshot:
            raise ValueError('private state backend text must not leak')
        return (self.rows.get(group, []), None, None)


class FakeGraph:
    def __init__(self, driver, recorder, *, fail_write=0, observe=True):
        self.driver, self.recorder = driver, recorder
        self.fail_write, self.observe = fail_write, observe
        self.writes, self.searches = [], []
        self.closed = False
    async def build_indices_and_constraints(self):
        pass
    async def add_episode(self, **kwargs):
        self.writes.append(kwargs)
        if self.fail_write == len(self.writes):
            raise RuntimeError('raw private prompt /NAS/yesh/secret')
        if self.observe:
            self.recorder.record_response(envelope(len(self.recorder.rows)+1), role='graphiti',
                                          schema_name='ExtractedEntities')
        group = kwargs['group_id']
        episode = {'kind': 'Episodic', 'properties': {'uuid': 'provider-'+kwargs['name'], 'name': kwargs['name'], 'group_id': group},
                   'linked_episode_uuids': [], 'linked_entity_uuids': []}
        self.driver.rows.setdefault(group, []).append(episode)
        result = Result({'episode': {'group_id': group}, 'nodes': [], 'edges': []})
        result.episode = SimpleNamespace(group_id=group, valid_at=kwargs['reference_time'])
        result.nodes, result.edges = [], []
        return result
    async def search_(self, **kwargs):
        self.searches.append(kwargs)
        if self.observe:
            self.recorder.record_response(envelope(len(self.recorder.rows)+1), role='graphiti',
                                          schema_name='QueryExpansion')
        group = kwargs['group_ids'][0]
        return Result({'edges': [{'uuid': key, 'group_id': group, 'fact': fact,
                                 'valid_at': '2026-01-01T00:00:00Z', 'invalid_at': None,
                                 'created_at': None, 'expired_at': None}
                                for key, fact in [('z', 'returned first'), ('a', 'returned second')]],
                       'edge_reranker_scores': [0.6, 0.4]})
    async def close(self):
        self.closed = True


class SearchConfig:
    def model_copy(self, **kwargs):
        return kwargs['update']


def environment(tmp_path, *, fail_write=0, observe=True, fail_snapshot=False):
    raw = []
    recorder = ObservationRecorder(private_sink=lambda p: raw.append(p))
    driver = FakeDriver()
    driver.fail_snapshot = fail_snapshot
    graph = FakeGraph(driver, recorder, fail_write=fail_write, observe=observe)
    witness = GraphitiStateWitness(driver, owned_namespaces=[c.runtime_namespace for c in cases()],
                                  private_sink=lambda p: raw.append(p))
    readouts = []
    async def complete(request, role):
        readouts.append((request, role))
        response = envelope(len(recorder.rows)+1, '{"disposition":"answered","answer":"test"}')
        recorder.record_response(response, role=role, schema_name='VersionTransferAnswer')
        return response
    runtime = runner.CaseRuntime(graph=graph, driver=driver, witness=witness, recorder=recorder,
        complete=complete, source_type='text', filter_factory=lambda when: ('public_time', when),
        search_config=SearchConfig(), private_sink=lambda name, payload: raw.append((name, payload)))
    return runtime, graph, driver, raw, readouts


def run_core(tmp_path, **kwargs):
    runtime, graph, driver, raw, readouts = environment(tmp_path, **kwargs)
    report = asyncio.run(runner.execute_cases(cases(), runtime,
        runtime_revision='a'*40, seconds=10))
    return report, runtime, graph, driver, raw, readouts


def test_readout_projection_changes_only_evaluation_mode_and_revalidates():
    from mub.vnext.contracts.v3.task import MemUpdateTaskV3
    for case in cases():
        original = case.task.model_dump(mode='json')
        projected = runner.project_readout_task(case.task)
        assert type(projected) is MemUpdateTaskV3
        assert MemUpdateTaskV3.model_validate_json(projected.model_dump_json()) == projected
        wanted = json.loads(json.dumps(original))
        for query in wanted['queries']:
            assert query['evaluation_mode'] == 'native_system'
            query['evaluation_mode'] = 'retrieved_prompt'
        assert projected.model_dump(mode='json') == wanted
        assert original == case.task.model_dump(mode='json')
        assert projected.semantic_hash != case.semantic_hash


def test_success_actual_task1_cases_through_native_boundary(tmp_path):
    report, runtime, graph, driver, raw, readouts = run_core(tmp_path)
    assert report['status'] == 'OBSERVATIONS_COMPLETE'
    assert [c['status'] for c in report['cases']] == ['OBSERVED']*3
    assert report['counts'] == {'event_attempts': 9, 'event_completions': 9,
                                'retrieval_attempts': 5, 'retrieval_completions': 5,
                                'readout_attempts': 15, 'readout_completions': 15}
    expected = [e.public_payload for c in cases() for e in c.events]
    assert [w['episode_body'] for w in graph.writes] == [e['raw_text'] for e in expected]
    assert [w['name'] for w in graph.writes] == [e['event_id'] for e in expected]
    public_queries = [q.public_payload for c in cases() for q in c.queries]
    assert [s['query'] for s in graph.searches] == [q['query_text'] for q in public_queries]
    assert all(s['search_filter'][0] == 'public_time' for s in graph.searches)
    assert len(readouts) == 15
    for offset in range(0, 15, 3):
        native, reverse, oracle = readouts[offset:offset+3]
        assert tuple(e.entry_id for e in native[0].retrieval_trace.retrieved_entries) == ('z', 'a')
        assert tuple(e.entry_id for e in reverse[0].retrieval_trace.retrieved_entries) == ('a', 'z')
        assert oracle[1] == 'oracle_readout'
    assert len(report['canonical_scores']) == 2
    for order in ('native', 'reverse'):
        score_rows = next(value for item in raw if isinstance(item, tuple)
                          for name, value in [item] if name == f'scoring_{order}_scores.jsonl')
        assert len(score_rows) == 3
        assert all(score['answer_scores']['exact_match'] is not None for score in score_rows)
        assert all(all(value is None for value in score['state_scores'].values()) for score in score_rows)
        assert report['canonical_scores'][order]['projection_policy'] == runner.PROJECTION_POLICY
    assert all(c['namespace_empty_after_cleanup'] for c in report['cases'])
    assert len(report['empty_state_preflight']) == 3
    assert graph.closed
    assert 'raw_text' not in json.dumps(report)
    assert '/NAS/' not in json.dumps(report)
    assert all(c['state_accuracy'] is None for c in report['cases'])


@pytest.mark.parametrize('options', [{'observe': False}, {'fail_snapshot': True}, {'fail_write': 2}])
def test_failure_blocks_active_and_never_retries_later_cases(tmp_path, options):
    report, _, graph, driver, _, _ = run_core(tmp_path, **options)
    assert report['status'] == 'BLOCKED'
    assert [c['status'] for c in report['cases']] == ['BLOCKED', 'NOT_RUN', 'NOT_RUN']
    assert any('RETURN anomaly_count' in q for q, _ in driver.calls)
    assert graph.closed
    assert 'secret' not in json.dumps(report)
    if options.get('fail_write'):
        assert report['cases'][0]['event_attempts'] == 2
        assert report['cases'][0]['event_completions'] == 1


def test_timeout_is_terminal_and_cleanup_runs(tmp_path):
    runtime, graph, driver, _, _ = environment(tmp_path)
    async def hung(**kwargs):
        await asyncio.Future()
    graph.add_episode = hung
    report = asyncio.run(runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=.01))
    assert report['cases'][0]['status'] == 'BLOCKED'
    assert report['cases'][0]['event_attempts'] == 1
    assert report['cases'][0]['event_completions'] == 0
    assert report['cases'][1]['status'] == 'NOT_RUN'
    assert graph.closed
    assert all(not driver.rows.get(c.runtime_namespace) for c in cases())


def test_source_manifest_exact_map_and_raw_bytes(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    files = {}
    for name in runner.REQUIRED_SOURCE_FILES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'qualified bytes\n')
        files[name] = runner.sha(path)
    manifest = tmp_path/'source.json'
    manifest.write_text(json.dumps({'schema': runner.SOURCE_SCHEMA, 'runtime_revision': 'a'*40,
                                    'files': files}))
    pin = runner.sha(manifest)
    verified = runner.verify_runtime_source(manifest, pin, 'a'*40, root=root)
    assert verified['files'] == files
    (root/runner.REQUIRED_SOURCE_FILES[0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='source'):
        runner.verify_runtime_source(manifest, pin, 'a'*40, root=root)
    with pytest.raises(ValueError):
        runner.verify_runtime_source(manifest, pin, 'b'*40, root=root)


def test_packet_exact_v2_reopen_and_separate_runtime_source():
    packet, bound = runner.load_cases(PACKET, runner.PACKET_PIN)
    assert packet['code_revision'] == runner.PACKET_SOURCE
    assert len(bound) == 3
    assert [len(c.queries) for c in bound] == [2, 1, 2]
    with pytest.raises(ValueError):
        runner.load_cases(PACKET, '0'*64)


def test_one_shot_reservation_never_removed(tmp_path):
    path = tmp_path/'reservation'
    runner.reserve_once(path, {'attempts': 3})
    assert path.is_dir()
    with pytest.raises(FileExistsError):
        runner.reserve_once(path, {'attempts': 3})
    assert path.exists()


def test_budget_and_time_rejected_before_runtime():
    assert runner.verify_budget(runner.PRIOR_TOTALS)['attempts'] == 2023
    with pytest.raises(ValueError):
        runner.verify_budget({**runner.PRIOR_TOTALS, 'attempts': 9999})
    with pytest.raises(ValueError):
        runner.verify_budget(runner.PRIOR_TOTALS, limits={**runner.LIMITS, 'seconds': 20000})


def test_transport_observer_preserves_requests_and_accounts_embeddings(tmp_path):
    import httpx
    recorder = ObservationRecorder()
    raw = []
    observer = runner.TransportObserver(recorder, private_sink=lambda name, value: raw.append(value))
    observer.operation = {'case': 0, 'kind': 'event', 'position': 1}
    request = httpx.Request('POST', 'http://127.0.0.1:18043/v1/chat/completions',
        headers={'X-MUB-Role': 'graphiti'}, json={'messages': [{'role':'user','content':'private prompt'}],
        'response_format': {'json_schema': {'name': 'ActualSchema', 'schema': {'type': 'object'}}}})
    response = httpx.Response(200, request=request, json=envelope())
    original = request.content
    asyncio.run(observer.on_request(request))
    asyncio.run(observer.on_response(response))
    assert request.content == original
    assert recorder.rows[0]['schema_name'] == 'ActualSchema'
    assert observer.receipts[0]['operation'] == observer.operation
    assert observer.receipts[0]['call_id'] == 'study-attempt-1'
    embedding = httpx.Request('POST', 'http://127.0.0.1:18043/v1/embeddings', json={'input':['private']})
    asyncio.run(observer.on_request(embedding))
    asyncio.run(observer.on_response(httpx.Response(200, request=embedding, json={'data':[]})))
    assert observer.embedding_attempts == 1
    assert len(recorder.rows) == 1
    assert len(raw) == 2
    assert 'private prompt' not in json.dumps(observer.receipts)
    assert observer.assert_complete() is None


def test_prior_development_terminal_and_ledger_are_both_authenticated(tmp_path, monkeypatch):
    prior = {'attempts': 84, 'input_tokens': 128445, 'output_tokens': 6286}
    monkeypatch.setattr(runner, 'verify_prior_budget', lambda: prior)
    monkeypatch.setattr(runner, 'DEV_ROOT', tmp_path)
    charged = {'attempts': 1747, 'input_tokens': 2457599, 'output_tokens': 118502}
    audit = {'charged': charged, 'all_settled': True, 'unsettled_attempts': [],
             'roles': {'graphiti':1603,'readout':96,'oracle_readout':48}}
    ledger = tmp_path/'llm-calls.jsonl'
    ledger.write_bytes(b'authenticated fake ledger bytes')
    monkeypatch.setattr(runner, 'audit_llm_ledger', lambda path: audit)
    terminal = tmp_path/'terminal.json'
    terminal.write_text(json.dumps({'index': {'llm-calls.jsonl': runner.sha(ledger)},
                                    'llm_accounting': audit}))
    monkeypatch.setattr(runner, 'DEV_TERMINAL_PIN', runner.sha(terminal))
    assert runner.verify_prior_consumption() == runner.PRIOR_TOTALS
    ledger.write_bytes(b'changed')
    with pytest.raises(ValueError, match='ledger hash'):
        runner.verify_prior_consumption()
    terminal.write_bytes(b'changed')
    with pytest.raises(ValueError, match='terminal hash'):
        runner.verify_prior_consumption()


def test_default_admission_never_reserves_or_loads_sdk(tmp_path, monkeypatch, capsys):
    manifest = {'files': {'bound.py': 'a'*64}}
    monkeypatch.setattr(runner, 'verify_runtime_source', lambda *args: manifest)
    packet = {'code_revision': runner.PACKET_SOURCE}
    monkeypatch.setattr(runner, 'load_cases', lambda *args: (packet, cases()))
    monkeypatch.setattr(runner, 'verify_qualification', lambda: {'elapsed_seconds': 1})
    monkeypatch.setattr(runner, 'verify_prior_consumption', lambda: runner.PRIOR_TOTALS)
    monkeypatch.setattr(runner, 'admit_output', lambda path: tmp_path/'never-created')
    def forbidden(*args):
        pytest.fail('default admission must not load SDK, reserve or execute')
    for name in ('verify_sdk_python_sources', 'reserve_once', 'run_owned'):
        monkeypatch.setattr(runner, name, forbidden)
    flags = ['--output-root', 'not-created', '--packet-root', str(PACKET),
             '--packet-index-sha256', runner.PACKET_PIN, '--runtime-revision', 'a'*40,
             '--source-manifest', 'source.json', '--source-manifest-sha256', 'b'*64]
    assert runner.main(flags) == 0
    admission = json.loads(capsys.readouterr().out)
    assert admission['status'] == 'NONEXECUTING'
    assert admission['reserved_smoke_slots'] == 0
    assert admission['packet_source_revision'] != admission['runtime_revision']
    assert not (tmp_path/'never-created').exists()
    def changed(*args):
        raise ValueError('runtime source changed')
    monkeypatch.setattr(runner, 'verify_runtime_source', changed)
    with pytest.raises(ValueError, match='source changed'):
        runner.main([*flags, '--execute'])


def test_service_command_constrains_actual_budget_and_pinned_inputs(tmp_path):
    command = runner.service_command(tmp_path, 'GPU-123', 123)
    for name, value in (('--attempts',192), ('--input-tokens',512000),
                        ('--output-tokens',65536), ('--seconds',123)):
        assert command[command.index(name)+1] == str(value)
    assert '--require-json-schema' in command
    assert 'a09a35458c702b33eeacc393d103063234e8bc28' in command[command.index('--snapshot')+1]
    assert command[command.index('--model-id')+1] == runner.MODEL_ID


def test_cleanup_failure_blocks_active_and_leaves_later_not_run(tmp_path):
    runtime, graph, driver, _, _ = environment(tmp_path, fail_write=1)
    original = driver.execute_query
    async def fail_cleanup(query, **kwargs):
        if 'RETURN anomaly_count' in query and graph.writes:
            return ([{'anomaly_count':1}], None, None)
        return await original(query, **kwargs)
    driver.execute_query = fail_cleanup
    report = asyncio.run(runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=10))
    assert report['status'] == 'BLOCKED'
    assert report['cases'][0]['namespace_empty_after_cleanup'] is False
    assert [c['status'] for c in report['cases']] == ['BLOCKED','NOT_RUN','NOT_RUN']
    assert graph.closed


def test_driver_initialization_once_and_empty_preflight_before_ingest(tmp_path):
    runtime, graph, driver, _, _ = environment(tmp_path)
    async def scenario():
        initialized = []
        async def init():
            initialized.append(True)
        driver._init_task = asyncio.create_task(init())
        await runner.initialize_empty(runtime, cases())
        assert len(initialized) == 1
        assert graph.writes == []
        assert len(runtime.preflight) == 3
        report = await runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=10)
        assert len(initialized) == 1
        return report
    assert asyncio.run(scenario())['status'] == 'OBSERVATIONS_COMPLETE'


def test_core_periodic_limit_guard_stops_a_hung_native_write(tmp_path):
    runtime, graph, driver, _, _ = environment(tmp_path)
    async def hung(**kwargs):
        await asyncio.Future()
    graph.add_episode = hung
    calls = []
    def guard():
        calls.append(1)
        if len(calls) > 1:
            raise ValueError('storage limit')
    runtime.check_limits = guard
    report = asyncio.run(runner.execute_cases(cases(), runtime, runtime_revision='a'*40, seconds=1))
    assert len(calls) > 1
    assert report['status'] == 'BLOCKED'
    assert report['cases'][1]['status'] == 'NOT_RUN'


@pytest.mark.parametrize('failure', ['database_ready', 'empty_snapshot'])
def test_primary_startup_failure_emits_terminal_and_never_loads_model(tmp_path, monkeypatch, failure):
    run = tmp_path/'run'
    reservation = tmp_path/'reservation'
    monkeypatch.setattr(runner, 'BASE', tmp_path)
    monkeypatch.setattr(runner, 'RESERVATION', reservation)
    monkeypatch.setattr(runner.os, 'environ', dict(runner.os.environ))
    monkeypatch.setattr(runner, 'fresh_gpu', lambda uuid: {'uuid':uuid,'free_mib':40000})
    monkeypatch.setattr(runner, 'available', lambda port: None)
    def config(path, neo):
        for name in ('home','cache','tmp'):
            (path/name).mkdir()
        return path/'conf'
    monkeypatch.setattr(runner, 'database_config', config)
    launched, stopped = [], []
    def popen(command, **kwargs):
        process = SimpleNamespace(pid=99999)
        launched.append(command)
        return process
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(runner, 'stop_owned', lambda p: stopped.append(p.pid) or True)
    def ready(*args):
        if failure == 'database_ready':
            raise RuntimeError('private startup failure /NAS/yesh/path')
    monkeypatch.setattr(runner, 'wait_port', ready)
    runtime, graph, driver, _, _ = environment(tmp_path, fail_snapshot=True)
    runtime.observer = runner.TransportObserver(runtime.recorder, private_sink=lambda *args: None)
    async def factory(*args):
        return runtime
    monkeypatch.setattr(runner, 'make_runtime', factory)
    class ClosedSocket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def connect_ex(self, address):
            return 1
    real_socket = runner.socket.socket
    monkeypatch.setattr(runner.socket, 'socket',
                        lambda *a, **kw: real_socket(*a, **kw) if a or kw else ClosedSocket())
    monkeypatch.setattr(runner.signal, 'SIGALRM', 14, raising=False)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(runner.signal, 'alarm', lambda *args: None, raising=False)
    args = SimpleNamespace(gpu_uuid='GPU-123', runtime_revision='a'*40)
    admission = {'status':'NONEXECUTING','packet_source_revision':runner.PACKET_SOURCE,
                 'runtime_revision':'a'*40,'scientific_release_allowed':False,'validation_tasks':0}
    assert runner.run_owned(args, cases(), run, admission) == 1
    terminal = json.loads((run/'terminal.json').read_bytes())
    assert terminal['status'] == 'BLOCKED'
    assert [c['status'] for c in terminal['cases']] == ['BLOCKED','NOT_RUN','NOT_RUN']
    assert len(launched) == 1
    assert 'neo4j' in ' '.join(launched[0])
    assert stopped == [99999]
    assert terminal['counts']['event_attempts'] == 0
    assert '/NAS/' not in json.dumps(terminal)
    assert (reservation/'terminal.json').is_file()
    assert reservation.exists()
    if failure == 'empty_snapshot':
        assert graph.closed
        assert any('RETURN anomaly_count' in q for q, _ in driver.calls)


def test_actual_service_ledger_reconciles_call_ids_usage_and_embeddings(tmp_path):
    from mub.vnext.runtime.version_transfer_budget import StudyBudget
    import httpx
    budget = StudyBudget(tmp_path/'llm-calls.jsonl', **runner.LIMITS)
    ticket = budget.reserve(role='graphiti', prompt_tokens=20, max_output_tokens=128)
    budget.settle(ticket, completion_tokens=2, finish_reason='stop')
    budget.close()
    recorder = ObservationRecorder()
    observer = runner.TransportObserver(recorder, private_sink=lambda *args: None)
    request = httpx.Request('POST','http://127.0.0.1:18043/v1/chat/completions',
        headers={'X-MUB-Role':'graphiti'}, json={'response_format':{'json_schema':{'name':'X','schema':{}}}})
    asyncio.run(observer.on_request(request))
    asyncio.run(observer.on_response(httpx.Response(200, request=request, json=envelope())))
    (tmp_path/'embedding-calls.jsonl').write_bytes(b'')
    runtime = SimpleNamespace(observer=observer, recorder=recorder)
    audit = runner.audit_current(tmp_path, runtime)
    assert audit['llm']['charged'] == {'attempts':1, 'input_tokens':20, 'output_tokens':2}
    assert audit['embeddings']['attempts'] == 0
    assert audit['native_sdk_retry_policy'] != 0
    assert audit['openai_transport_retries'] == 0
    # Recompute rather than trust a terminal-supplied total or nominal call count.
    path = tmp_path/'llm-calls.jsonl'
    path.write_text(path.read_text().replace('"completion_tokens": 2', '"completion_tokens": 3'))
    with pytest.raises(ValueError, match='binding mismatch'):
        runner.audit_current(tmp_path, runtime)


def test_owned_group_tracks_model_child_not_just_timeout_wrapper(tmp_path, monkeypatch):
    proc = tmp_path/'proc'
    for pid in (100, 101, 200):
        (proc/str(pid)).mkdir(parents=True)
    monkeypatch.setattr(runner.os, 'getpgid', lambda pid: 100 if pid in (100,101) else 200, raising=False)
    assert runner.owned_group_pids([SimpleNamespace(pid=100)], proc_root=proc) == {100,101}


def test_missing_response_or_invalid_json_observation_blocks():
    import httpx
    recorder = ObservationRecorder()
    observer = runner.TransportObserver(recorder, private_sink=lambda *args: None)
    request = httpx.Request('POST','http://127.0.0.1:18043/v1/chat/completions',
        headers={'X-MUB-Role':'graphiti'}, json={'response_format':{'json_schema':{'name':'X','schema':{}}}})
    asyncio.run(observer.on_request(request))
    with pytest.raises(ValueError):
        observer.assert_complete()
    # A passive observer must not replace native JSON/status errors and retries.
    asyncio.run(observer.on_response(httpx.Response(200, request=request, json=envelope(text='[]'))))
    assert recorder.rows[-1]['status'] == 'BLOCKED'
    with pytest.raises(ValueError):
        observer.assert_complete()
