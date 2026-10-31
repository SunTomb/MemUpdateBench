"""Historical serving Task 3: source-bound admission and one-shot observations.

Default is NONEXECUTING. Neither --help nor admission imports native SDK/model
packages. A runtime revision is a label; the separately pinned raw-byte source
manifest authenticates deployed project code. No scientific decision is made.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.contracts.v3.task import MemUpdateTaskV3
from mub.vnext.external.artifacts import assert_no_reparse_components
from mub.vnext.external.providers.graphiti_native_episode import GraphitiNativeEpisode
from mub.vnext.external.providers.graphiti_observation_recorder import (
    ObservationRecorder, ObservationQualificationError,
)
from mub.vnext.external.providers.graphiti_state_witness import GraphitiStateWitness
from mub.vnext.external.visibility import visible_event_input, visible_query_input
from mub.vnext.io import canonical_json_bytes
from mub.vnext.runtime.version_transfer_readout import build_readout, parse_readout_response
from mub.vnext.runtime.version_transfer_scoring import build_readout_run, build_scoring_artifacts
from scripts.vnext_prepare_historical_mechanism import build_cases, read_packet, SOURCE_FILES
from scripts.vnext_run_version_transfer_decision import (
    MODEL_ID, EMBEDDING_ID, available, database_config, native_graph_clients,
    sha, stop_owned, wait_port, write_new,
)
from scripts.vnext_run_version_transfer_dev import (
    BASE, ANSWER_SCHEMA, QUALIFICATION_PIN, QUALIFICATION_REL, oracle_request,
    audit_llm_ledger, verify_qualification, verify_sdk_python_sources, verify_prior_budget,
)

PACKET_SOURCE = 'db8c8585d90b312db7d5f692882b2a52bc988f76'
PACKET_PIN = 'b98b815b500a6e7594be39b4be78b233550c1f78f9acbee701ac1a106c027155'
DEV_TERMINAL_PIN = '84e75a7352e48ecff9cadd8f01fb81da303ecc89253b15f88e1084e95c34b1c7'
DEV_ROOT = BASE / '_local/graphiti_version_transfer_dev_1218c76_run_v1'
RESERVATION = BASE / '_local/historical_mechanism_reservation_v1'
SOURCE_SCHEMA = 'memupdatebench.historical-runtime-source.v1'
LIMITS = {'attempts': 192, 'input_tokens': 512000, 'output_tokens': 65536, 'seconds': 3600}
PRIOR_TOTALS = {'attempts': 1831, 'input_tokens': 2586044, 'output_tokens': 124788}
STORAGE_BYTES = 1024**3
CLEANUP_SECONDS = 240
REQUIRED_SOURCE_FILES = tuple(sorted(set(SOURCE_FILES) | {
    'scripts/__init__.py', 'scripts/vnext_run_historical_mechanism.py',
    'scripts/vnext_run_version_transfer_decision.py', 'scripts/vnext_run_version_transfer_dev.py',
    'scripts/vnext_prepare_version_transfer_decision.py', 'scripts/vnext_serve_version_transfer_local.py',
    'mub/vnext/external/providers/graphiti_native_episode.py',
    'mub/vnext/external/providers/graphiti_state_witness.py',
    'mub/vnext/external/providers/graphiti_observation_recorder.py',
    'mub/vnext/runtime/version_transfer_readout.py', 'mub/vnext/runtime/version_transfer_scoring.py',
    'mub/vnext/runtime/version_transfer_budget.py', 'mub/vnext/runtime/version_transfer_local.py',
    'mub/vnext/runtime/version_transfer_embedding.py',
    'configs/vnext/version_transfer_decision_v1.json',
}))
_COUNT_KEYS = ('event_attempts', 'event_completions', 'retrieval_attempts',
               'retrieval_completions', 'readout_attempts', 'readout_completions')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_runtime_source(path, expected_sha256, runtime_revision, *, root=ROOT):
    """Exact raw bytes, safe relative names, and all project package sources.

    Manifest schema: {schema, runtime_revision, files:{relative_posix:sha256}}.
    The expected SHA-256 must be supplied out of band with the clean archive.
    Extra listed files are verified too; unlisted mub Python sources are refused.
    """
    if not re.fullmatch('[0-9a-f]{40}', runtime_revision):
        raise ValueError('runtime source revision must be 40 lowercase hex digits')
    path, root = Path(path), Path(root).resolve(strict=True)
    assert_no_reparse_components(path)
    if sha(path) != expected_sha256:
        raise ValueError('runtime source manifest hash mismatch')
    manifest = json.loads(path.read_bytes())
    if (set(manifest) != {'schema', 'runtime_revision', 'files'}
            or manifest['schema'] != SOURCE_SCHEMA or manifest['runtime_revision'] != runtime_revision
            or type(manifest['files']) is not dict):
        raise ValueError('runtime source manifest shape/revision mismatch')
    files = manifest['files']
    required = set(REQUIRED_SOURCE_FILES) | {
        p.relative_to(root).as_posix() for p in (root/'mub').rglob('*.py')}
    if not required.issubset(files):
        raise ValueError('runtime source manifest missing required modules')
    for name, pin in files.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or relative.as_posix() != name or not re.fullmatch('[0-9a-f]{64}', pin)):
            raise ValueError('runtime source map invalid')
        target = root/name
        assert_no_reparse_components(target)
        if not target.resolve(strict=True).is_relative_to(root) or sha(target) != pin:
            raise ValueError('runtime source file hash mismatch')
    return manifest


def load_cases(packet_root, pin):
    if pin != PACKET_PIN:
        raise ValueError('only the frozen v2 preparation packet is executable')
    manifest = read_packet(packet_root, pin)
    if manifest['code_revision'] != PACKET_SOURCE:
        raise ValueError('packet preparation source mismatch')
    cases = build_cases(code_revision=manifest['code_revision'])
    if (len(cases) != 3 or len({c.runtime_namespace for c in cases}) != 3
            or [len(c.events) for c in cases] != [3, 3, 3]
            or [len(c.queries) for c in cases] != [2, 1, 2]):
        raise ValueError('exact three-case protocol required')
    return manifest, cases


PROJECTION_POLICY = 'native-system-to-retrieved-prompt-readout-only-v1'


def project_readout_task(task):
    """Revalidate a reader-only diagnostic; never supply this copy to Graphiti."""
    payload = task.model_dump(mode='json')
    for query in payload['queries']:
        if query['evaluation_mode'] != 'native_system':
            raise ValueError('original task must retain native-system evaluation mode')
        query['evaluation_mode'] = 'retrieved_prompt'
    return MemUpdateTaskV3.model_validate_json(json.dumps(payload))


def projection_binding(case, projected):
    return {'policy': PROJECTION_POLICY, 'original_task_sha256': case.task_hash,
        'original_semantic_hash': case.semantic_hash,
        'projected_task_sha256': hashlib.sha256(canonical_json_bytes(projected)).hexdigest(),
        'projected_semantic_hash': projected.semantic_hash,
        'evidence_class': 'native_evidence_separate_prompted_reader_not_native_system_answer'}


def verify_prior_consumption():
    prerequisite = verify_prior_budget()
    terminal = DEV_ROOT/'terminal.json'
    if sha(terminal) != DEV_TERMINAL_PIN:
        raise ValueError('development terminal hash mismatch')
    row = json.loads(terminal.read_bytes())
    ledger = DEV_ROOT/'llm-calls.jsonl'
    if sha(ledger) != row['index']['llm-calls.jsonl']:
        raise ValueError('development ledger hash mismatch')
    audit = audit_llm_ledger(ledger)
    if not audit['all_settled'] or audit != row['llm_accounting']:
        raise ValueError('development accounting mismatch')
    totals = {key: prerequisite[key]+audit['charged'][key] for key in prerequisite}
    if totals != PRIOR_TOTALS or audit['charged']['attempts'] != 1747:
        raise ValueError('unexpected prior consumption')
    return totals


def verify_budget(prior, *, limits=None):
    limits = LIMITS if limits is None else limits
    if set(limits) != set(LIMITS) or any(type(limits[k]) is not int or not 0 < limits[k] <= LIMITS[k]
                                        for k in LIMITS):
        raise ValueError('historical sub-budget exceeded')
    ceiling = {'attempts': 10000, 'input_tokens': 24000000, 'output_tokens': 2000000}
    if set(prior) != set(ceiling) or any(type(prior[k]) is not int or prior[k] < 0 for k in prior):
        raise ValueError('invalid prior accounting')
    aggregate = {k: prior[k]+limits[k] for k in prior}
    if any(aggregate[k] > ceiling[k] for k in aggregate) or 7200+21600+limits['seconds'] > 43200:
        raise ValueError('whole-study budget exceeded')
    return aggregate


def reserve_once(path, admission):
    path = Path(path)
    assert_no_reparse_components(path)
    path.mkdir(exist_ok=False)
    write_new(path/'reservation.json', admission)


class TransportObserver:
    """Passive local httpx hooks; never changes native request/response bodies.

    A failed response is retained even if native tenacity subsequently retries.
    Missing responses remain pending and fail qualification. Embeddings are
    counted separately, never misclassified as extraction JSON.
    """
    def __init__(self, recorder, *, private_sink):
        self.recorder, self.private_sink = recorder, private_sink
        self.operation = {'kind': 'startup'}
        self.receipts = []
        self.embedding_attempts = 0
        self._pending = {}

    def attach(self, api):
        client = api._client
        client.event_hooks.setdefault('request', []).append(self.on_request)
        client.event_hooks.setdefault('response', []).append(self.on_response)

    async def on_request(self, request):
        if request.url.host != '127.0.0.1' or request.url.port != 18043:
            raise ValueError('observer requires declared local transport')
        kind = request.url.path
        if kind not in ('/v1/chat/completions', '/v1/embeddings'):
            raise ValueError('unqualified local transport operation')
        body = json.loads(request.content)
        role = request.headers.get('X-MUB-Role')
        schema = body.get('response_format', {}).get('json_schema', {})
        row = {'transport_attempt': len(self.receipts)+1, 'kind': kind.rsplit('/', 1)[-1],
               'role': role, 'operation': dict(self.operation), 'request_sha256': digest(body),
               'schema_name': schema.get('name'),
               'schema_sha256': digest(schema.get('schema')) if 'schema' in schema else None,
               'status': 'PENDING'}
        self.receipts.append(row)
        self._pending[id(request)] = (row, body)
        if kind.endswith('embeddings'):
            self.embedding_attempts += 1

    async def on_response(self, response):
        row, body = self._pending.pop(id(response.request))
        await response.aread()
        row['http_status'] = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {'non_json_response_sha256': hashlib.sha256(response.content).hexdigest()}
        row['response_sha256'] = digest(payload)
        self.private_sink('transport', {'operation': row['operation'], 'request': body, 'response': payload})
        if row['kind'] == 'embeddings':
            row['status'] = 'OBSERVED' if response.status_code == 200 else 'BLOCKED'
            if row['status'] == 'BLOCKED':
                raise ValueError('embedding transport failed')
            return
        try:
            observed = self.recorder.record_response(payload, role=row['role'],
                schema_name=row['schema_name'], timestamp=datetime.now(timezone.utc).isoformat(),
                private_material={'operation': row['operation'], 'request': body})
            row.update(call_id=observed['call_id'], observation_id=observed['observation_id'],
                       usage=observed['usage'], status=observed['status'])
            if (response.status_code != 200 or row['schema_sha256'] is None
                    or not re.fullmatch(r'study-attempt-[1-9][0-9]*', observed['call_id'])):
                raise ValueError('local completion identity/schema/http mismatch')
        except ObservationQualificationError as exc:
            row.update(status='BLOCKED', observation_id=exc.observation['observation_id'])
            # Leave the original HTTP/JSON response for native SDK handling.
            # Its qualified tenacity retries remain observable and unmodified;
            # qualification still stops at the next operation boundary.
        except BaseException:
            row['status'] = 'BLOCKED'
            raise

    def assert_complete(self):
        if self._pending or any(r['status'] != 'OBSERVED' for r in self.receipts):
            raise ValueError('transport observations incomplete or blocked')


@dataclass
class CaseRuntime:
    graph: Any
    driver: Any
    witness: GraphitiStateWitness
    recorder: ObservationRecorder
    complete: Callable
    source_type: Any
    filter_factory: Callable
    search_config: Any
    private_sink: Callable
    observer: TransportObserver | None = None
    close_clients: Callable | None = None
    check_limits: Callable = lambda: None
    initialized: bool = False
    preflight: tuple = ()


def initial_report(cases):
    return {'status': 'BLOCKED', 'scientific_release_allowed': False, 'validation_tasks': 0,
        'scientific_decision': 'NOT_EVALUATED', 'cases': [
            {'role': c.role, 'task_hash': c.task_hash, 'status': 'NOT_RUN',
             **dict.fromkeys(_COUNT_KEYS, 0), 'state_accuracy': None,
             'snapshots': [], 'queries': [], 'namespace_empty_after_cleanup': None} for c in cases],
        'counts': dict.fromkeys(_COUNT_KEYS, 0), 'empty_state_preflight': [], 'canonical_scores': {}}


def blocked(report, exc, active=0, *, field='error_type'):
    report['status'] = 'BLOCKED'
    report[field] = type(exc).__name__
    if report['cases']:
        report['cases'][active].update(status='BLOCKED', **{field: type(exc).__name__})


def snapshot_receipt(snapshot):
    if snapshot is None or snapshot.anomalies:
        raise ValueError('missing or anomalous direct state snapshot')
    return {'sha256': snapshot.sha256, 'rows': len(snapshot.rows)}


async def initialize_empty(runtime, cases):
    if runtime.initialized:
        return
    if runtime.driver._init_task is not None:
        await runtime.driver._init_task
    else:
        await runtime.graph.build_indices_and_constraints()
    runtime.initialized = True
    receipts = []
    for case in cases:
        group = case.runtime_namespace
        snapshot = await runtime.witness.snapshot(group)
        receipt = snapshot_receipt(snapshot)
        if snapshot.rows or not await runtime.witness.verify_empty(group):
            raise ValueError('fresh owned namespace is not empty')
        await runtime.witness.delete_namespace(group)
        if not await runtime.witness.verify_empty(group):
            raise ValueError('empty-state cleanup preflight failed')
        receipts.append({'role': case.role, **receipt, 'empty': True,
                         'evidence_class': 'empty_state_runtime_preflight_not_native_state'})
    runtime.preflight = tuple(receipts)


async def close_runtime(runtime):
    errors = []
    for callback in (runtime.graph.close, runtime.close_clients):
        if callback is not None:
            try:
                await asyncio.wait_for(callback(), timeout=20)
            except BaseException as exc:
                errors.append(type(exc).__name__)
    return errors


async def execute_cases(cases, runtime, *, runtime_revision, seconds, report=None, close=True):
    """Inject an actual async Graphiti/tuple-driver boundary; no service startup.

    Receipts are compact; the mandatory private_sink exclusively owns raw data.
    All three rows exist before the first await. No retries/resume are provided.
    """
    report = initial_report(cases) if report is None else report
    active = 0
    records = {'native': [], 'reverse': []}
    projected_tasks = tuple(project_readout_task(case.task) for case in cases)
    projection_bindings = [projection_binding(c, p) for c, p in zip(cases, projected_tasks)]
    for row, binding in zip(report['cases'], projection_bindings):
        row['readout_projection'] = binding

    def operation(kind, position, **extra):
        runtime.check_limits()
        if runtime.observer:
            runtime.observer.operation = {'case': active, 'kind': kind, 'position': position, **extra}

    def observed_since(start, *, required=False):
        rows = runtime.recorder.rows
        if (required and len(rows) == start) or any(r['status'] != 'OBSERVED' for r in rows):
            raise ValueError('missing or failed extraction observation')
        if runtime.observer:
            runtime.observer.assert_complete()

    async def body():
        nonlocal active
        await initialize_empty(runtime, cases)
        report['empty_state_preflight'] = list(runtime.preflight)
        for active, case in enumerate(cases):
            row, group, task = report['cases'][active], case.runtime_namespace, case.task
            bridge = GraphitiNativeEpisode(runtime.graph, namespace=group,
                source_type=runtime.source_type, filter_factory=runtime.filter_factory,
                search_config=runtime.search_config)
            requests, predictions = {o: [] for o in records}, {o: [] for o in records}
            for position, event in enumerate(task.events):
                operation('event', position)
                public = visible_event_input(event, runtime_namespace=group)
                if public.model_dump(mode='json') != case.events[position].public_payload:
                    raise ValueError('public episode packet binding mismatch')
                start = len(runtime.recorder.rows)
                row['event_attempts'] += 1
                result = await bridge.ingest(public)
                row['event_completions'] += 1
                runtime.private_sink('mutation', result.model_dump(mode='json'))
                if (result.episode.group_id != group
                        or result.episode.valid_at.isoformat().replace('+00:00', 'Z') != event.timestamp
                        or any(n.group_id != group for n in result.nodes)
                        or any(e.group_id != group for e in result.edges)):
                    raise ValueError('native mutation provenance mismatch')
                snapshot = await runtime.witness.snapshot(group)
                receipt = snapshot_receipt(snapshot)
                expected_episodes = {e.event_id for e in task.events[:position+1]}
                observed_episodes = {r.get('name') for r in snapshot.rows if r['kind'] == 'Episodic'}
                if not snapshot.rows or not expected_episodes.issubset(observed_episodes):
                    raise ValueError('missing native episode in direct state')
                row['snapshots'].append({'position': position, **receipt,
                    'mutation_sha256': digest(result.model_dump(mode='json'))})
                observed_since(start, required=True)
            for position, query in enumerate(task.queries):
                operation('query', position)
                public = visible_query_input(query, k=8, runtime_namespace=group)
                if public.model_dump(mode='json') != case.queries[position].public_payload:
                    raise ValueError('public query packet binding mismatch')
                before = await runtime.witness.snapshot(group)
                binding = snapshot_receipt(before)
                if not before.rows or before.sha256 != row['snapshots'][-1]['sha256']:
                    raise ValueError('final provider state missing or changed before retrieval')
                start = len(runtime.recorder.rows)
                row['retrieval_attempts'] += 1
                native = await bridge.retrieve(public)
                row['retrieval_completions'] += 1
                result = native.model_dump(mode='json')
                runtime.private_sink('retrieval', result)
                observed_since(start)
                after = await runtime.witness.snapshot(group)
                if snapshot_receipt(after) != binding:
                    raise ValueError('retrieval changed provider state')
                row['queries'].append({'position': position, 'final_snapshot_sha256': before.sha256,
                    'retrieval_sha256': digest(result)})
                for order in ('native', 'reverse', 'oracle'):
                    operation('readout', position, order=order)
                    # Privileged oracle sees gold only here, after native search and state binding.
                    readout_task = projected_tasks[active]
                    readout_query = readout_task.queries[position]
                    request = (oracle_request(readout_task, readout_query) if order == 'oracle' else
                               build_readout(readout_query, result, namespace=group, order=order))
                    role = 'oracle_readout' if order == 'oracle' else 'readout'
                    start = len(runtime.recorder.rows)
                    row['readout_attempts'] += 1
                    response = await runtime.complete(request, role)
                    prediction = parse_readout_response(request, response, model_id=MODEL_ID)
                    runtime.private_sink('readout', {'order': order, 'request': request.model_dump(mode='json'),
                        'response': response, 'prediction': prediction.model_dump(mode='json')})
                    observed_since(start, required=True)
                    if response['choices'][0]['finish_reason'] != 'stop' or not prediction.format_valid:
                        raise ValueError('readout technical/format failure')
                    row['readout_completions'] += 1
                    if order != 'oracle':
                        requests[order].append(request)
                        predictions[order].append(prediction)
            for order in records:
                record = build_readout_run(projected_tasks[active], requests[order], predictions[order],
                    run_id=f'historical-mechanism-{order}', order=order)
                records[order].append(record)
                runtime.private_sink('canonical_run', record.model_dump(mode='json'))
            await runtime.witness.delete_namespace(group)
            row['namespace_empty_after_cleanup'] = await runtime.witness.verify_empty(group)
            if not row['namespace_empty_after_cleanup']:
                raise ValueError('namespace cleanup failed')
            row['status'] = 'OBSERVED'
        for order in records:
            bundle = build_scoring_artifacts(projected_tasks, records[order],
                code_revision=runtime_revision, qualification_sha256=QUALIFICATION_PIN,
                historical_supported=True)
            for name, raw in bundle['payloads'].items():
                runtime.private_sink('scoring_'+order+'_'+name, json.loads(raw) if not name.endswith('.jsonl')
                    else [json.loads(line) for line in raw.splitlines()])
            runtime.private_sink('scoring_'+order+'_projection', projection_bindings)
            report['canonical_scores'][order] = {
                'projection_bindings_sha256': digest(projection_bindings),
                'projection_policy': PROJECTION_POLICY,
                'score_sha256': hashlib.sha256(bundle['payloads']['scores.jsonl']).hexdigest(),
                'task_manifest_sha256': bundle['task_manifest_sha256'],
                'run_manifest_sha256': bundle['run_manifest_sha256'], 'tasks': len(cases),
                'state_alignment': 'UNAVAILABLE_NOT_INFERRED_FROM_GOLD'}
        runtime.check_limits()
        report['status'] = 'OBSERVATIONS_COMPLETE'

    async def guarded_body():
        async def watch_limits():
            while True:
                await asyncio.sleep(.25)
                runtime.check_limits()
        worker = asyncio.create_task(body())
        watchdog = asyncio.create_task(watch_limits())
        try:
            done, _ = await asyncio.wait((worker, watchdog), return_when=asyncio.FIRST_COMPLETED)
            if watchdog in done:
                await watchdog
            await worker
        finally:
            for task in (worker, watchdog):
                if not task.done():
                    task.cancel()
            await asyncio.gather(worker, watchdog, return_exceptions=True)

    try:
        if type(seconds) not in (int, float) or not 0 < seconds <= LIMITS['seconds']:
            raise ValueError('invalid execution timeout')
        await asyncio.wait_for(guarded_body(), timeout=seconds)
    except BaseException as exc:
        blocked(report, exc, active)
        row = report['cases'][active]
        if row['event_attempts'] > len(row['snapshots']):
            try:
                snapshot = await asyncio.wait_for(runtime.witness.snapshot(cases[active].runtime_namespace), 15)
                row['interrupted_event_snapshot'] = snapshot_receipt(snapshot)
            except BaseException as state_exc:
                row['interrupted_state_error_type'] = type(state_exc).__name__
    finally:
        for index, case in enumerate(cases):
            try:
                await asyncio.wait_for(runtime.witness.delete_namespace(case.runtime_namespace), 15)
                empty = await asyncio.wait_for(runtime.witness.verify_empty(case.runtime_namespace), 20)
                report['cases'][index]['namespace_empty_after_cleanup'] = empty
                if not empty:
                    raise ValueError('namespace cleanup incomplete')
            except BaseException as exc:
                report['cases'][index]['namespace_empty_after_cleanup'] = False
                blocked(report, exc, active, field='cleanup_error_type')
        if close:
            errors = await close_runtime(runtime)
            report['client_cleanup_errors'] = errors
            if errors:
                blocked(report, RuntimeError(), active, field='client_cleanup_error_type')
        report['empty_state_preflight'] = list(runtime.preflight)
        report['counts'] = {key: sum(r[key] for r in report['cases']) for key in _COUNT_KEYS}
        report['extraction_observations'] = {'count': len(runtime.recorder.rows),
            'sha256': digest(runtime.recorder.rows),
            'blocked': sum(r['status'] == 'BLOCKED' for r in runtime.recorder.rows)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--packet-root', required=True, type=Path)
    parser.add_argument('--packet-index-sha256', required=True)
    parser.add_argument('--runtime-revision', required=True)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    source = verify_runtime_source(args.source_manifest, args.source_manifest_sha256, args.runtime_revision)
    packet, cases = load_cases(args.packet_root, args.packet_index_sha256)
    qualification = verify_qualification()
    prior = verify_prior_consumption()
    aggregate = verify_budget(prior)
    run = admit_output(args.output_root)
    admission = {'status': 'NONEXECUTING', 'scientific_release_allowed': False, 'validation_tasks': 0,
        'runtime_revision': args.runtime_revision, 'runtime_source_manifest_sha256': args.source_manifest_sha256,
        'runtime_source_map_sha256': digest(source['files']), 'packet_source_revision': packet['code_revision'],
        'packet_index_sha256': args.packet_index_sha256, 'qualification_sha256': QUALIFICATION_PIN,
        'development_terminal_sha256': DEV_TERMINAL_PIN, 'prior_usage': prior, 'ceilings': LIMITS,
        'aggregate_reserved_usage': aggregate, 'prior_time_allocation_seconds': 28800,
        'new_storage_ceiling_bytes': STORAGE_BYTES, 'smoke_slots_previously_consumed': 6,
        'reserved_smoke_slots': 0 if not args.execute else 3, 'paid_api_calls': 0,
        'planned_writes': 9, 'planned_retrievals': 5, 'planned_readouts': 15,
        'qualified_elapsed_seconds': qualification['elapsed_seconds']}
    if not args.execute:
        print(json.dumps(admission, sort_keys=True))
        return 0
    # Only execution imports installed SDK metadata or launches owned processes.
    sdk = verify_sdk_python_sources()
    admission['sdk_identity'] = sdk
    return run_owned(args, cases, run, admission)


def admit_output(path):
    path = Path(path).absolute()
    if ('..' in path.parts or not path.is_relative_to(BASE/'_local')
            or path.exists() or path.is_symlink()):
        raise ValueError('fresh project-local NAS output required')
    assert_no_reparse_components(path)
    return path


def output_bytes(run):
    return sum(p.stat().st_size for p in run.rglob('*') if p.is_file())


def fresh_gpu(uuid):
    if not uuid or not re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid):
        raise ValueError('explicit GPU UUID required')
    rows = subprocess.check_output(['nvidia-smi', '--id='+uuid, '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
    if len(rows) != 1:
        raise ValueError('GPU identity is not unique')
    identity, free = [x.strip() for x in rows[0].split(',')]
    if identity != uuid or int(free) < 24576:
        raise ValueError('fresh GPU identity/headroom admission blocked')
    return {'uuid': identity, 'free_mib': int(free)}


def service_command(run, uuid, seconds):
    cache = BASE.parent/'hf_cache/hub'
    return [str(BASE/'external/version_transfer_constraints_py310_v1/venv/bin/python'), '-B', '-I',
        str(ROOT/'scripts/vnext_serve_version_transfer_local.py'),
        '--snapshot', str(cache/'models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'),
        '--gpu-uuid', uuid, '--model-id', MODEL_ID, '--require-json-schema',
        '--ledger', str(run/'llm-calls.jsonl'), '--attempts', str(LIMITS['attempts']),
        '--input-tokens', str(LIMITS['input_tokens']), '--output-tokens', str(LIMITS['output_tokens']),
        '--seconds', str(seconds), '--embedding-snapshot',
        str(cache/'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'),
        '--embedding-model-id', EMBEDDING_ID, '--embedding-ledger', str(run/'embedding-calls.jsonl')]


async def make_runtime(cases, sink, check):
    from openai import AsyncOpenAI
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
    graph, driver, graph_api = native_graph_clients(structured_output_mode='json_schema')
    clients = [graph_api]
    try:
        for role in ('readout', 'oracle_readout'):
            clients.append(AsyncOpenAI(api_key='local-only-placeholder', base_url='http://127.0.0.1:18043/v1',
                default_headers={'X-MUB-Role': role}, max_retries=0, timeout=120))
        recorder = ObservationRecorder(private_sink=lambda value: sink('observation', value))
        observer = TransportObserver(recorder, private_sink=sink)
        for client in clients:
            observer.attach(client)
        async def complete(request, role):
            api = clients[2] if role == 'oracle_readout' else clients[1]
            response = await api.chat.completions.create(model=MODEL_ID,
                messages=[{'role': 'user', 'content': request.rendered_prompt}], temperature=0, max_tokens=128,
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'VersionTransferAnswer', 'schema': ANSWER_SCHEMA}})
            return response.model_dump(mode='json')
        async def close():
            failures = []
            for client in clients:
                try:
                    await client.close()
                except BaseException as exc:
                    failures.append(type(exc).__name__)
            if failures:
                raise RuntimeError('owned client close failure')
        return CaseRuntime(graph, driver, GraphitiStateWitness(driver,
            owned_namespaces=[c.runtime_namespace for c in cases],
            private_sink=lambda value: sink('state', value)), recorder, complete,
            EpisodeType.text, GraphitiNativeEpisode.validity_filter_factory(SearchFilters, DateFilter, ComparisonOperator),
            EDGE_HYBRID_SEARCH_RRF, sink, observer=observer, close_clients=close, check_limits=check)
    except BaseException:
        try:
            await graph.close()
        finally:
            for client in clients:
                await client.close()
        raise


def audit_current(run, runtime):
    audit = audit_llm_ledger(run/'llm-calls.jsonl')
    if not audit['all_settled'] or any(audit['charged'][k] > LIMITS[k] for k in PRIOR_TOTALS):
        raise ValueError('current LLM accounting incomplete or over limit')
    if runtime is None or runtime.observer is None:
        raise ValueError('missing transport observations')
    runtime.observer.assert_complete()
    rows = runtime.recorder.rows
    reserved = [json.loads(line) for line in (run/'llm-calls.jsonl').read_bytes().splitlines()
                if json.loads(line)['event'] == 'reserved']
    settled = {r['attempt']: r for r in map(json.loads, (run/'llm-calls.jsonl').read_bytes().splitlines())
               if r['event'] == 'settled'}
    if len(rows) != len(reserved):
        raise ValueError('actual completion observations and service ledger differ')
    for observation, reservation in zip(rows, reserved):
        ticket = reservation['attempt']
        if (observation['status'] != 'OBSERVED' or observation['call_id'] != f'study-attempt-{ticket}'
                or observation['role'] != reservation['role']
                or observation['usage']['prompt_tokens'] != reservation['input_tokens']
                or observation['usage']['completion_tokens'] != settled[ticket]['completion_tokens']
                or observation['finish_reason'] != settled[ticket]['finish_reason']):
            raise ValueError('completion usage/call binding mismatch')
    embeddings = list(map(json.loads, (run/'embedding-calls.jsonl').read_bytes().splitlines()))
    a = [r for r in embeddings if r['event'] == 'embedding_reserved']
    b = [r for r in embeddings if r['event'] == 'embedding_settled']
    expected = list(range(1, len(a)+1))
    if ([r['attempt'] for r in a] != expected or [r['attempt'] for r in b] != expected
            or len(a) != runtime.observer.embedding_attempts):
        raise ValueError('embedding accounting incomplete or differs from transport')
    return {'llm': audit, 'aggregate_charged_llm': {
        k: PRIOR_TOTALS[k]+audit['charged'][k] for k in PRIOR_TOTALS},
        'embeddings': {'attempts': len(a), 'items': sum(r['items'] for r in a),
                       'input_tokens': sum(r['input_tokens'] for r in a)},
        'native_sdk_retry_policy': 'qualified_native_tenacity_up_to_4_preserved',
        'openai_transport_retries': 0, 'actual_http_attempts': len(runtime.observer.receipts)}


def owned_group_pids(processes, *, proc_root=Path('/proc')):
    """Track actual model/Java children of our start_new_session groups."""
    groups = {p.pid for p in processes}
    members = set()
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            pid = int(entry.name)
            try:
                if os.getpgid(pid) in groups:
                    members.add(pid)
            except ProcessLookupError:
                pass
    return members


def run_owned(args, cases, run, admission):
    """Local Linux/NAS orchestration only; never called during offline tests."""
    for port in (17687, 18043):
        available(port)
    admission['gpu_admission'] = fresh_gpu(args.gpu_uuid)
    reserve_once(RESERVATION, admission)
    report = initial_report(cases)
    report.update({k: v for k, v in admission.items() if k != 'status'})
    created = False
    try:
        run.mkdir(mode=0o700)
        created = True
        (run/'private').mkdir(mode=0o700)
        (run/'compact').mkdir()
        write_new(run/'admission.json', admission)
    except BaseException as exc:
        blocked(report, exc)
        if created:
            write_new(run/'terminal.json', report)
        write_new(RESERVATION/'terminal.json', report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1
    started = time.monotonic()
    processes, streams, runtime_box = [], [], []
    counter = 0
    private_index = {}

    def check():
        if time.monotonic()-started >= LIMITS['seconds']-CLEANUP_SECONDS:
            raise TimeoutError('historical runtime ceiling')
        if output_bytes(run) > STORAGE_BYTES:
            raise ValueError('historical storage ceiling')

    def sink(kind, value):
        nonlocal counter
        check()
        counter += 1
        path = run/'private'/f'{counter:05d}_{kind}.json'
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if output_bytes(run)+len(payload) > STORAGE_BYTES:
            raise ValueError('private artifact would exceed storage ceiling')
        with path.open('xb') as stream:
            stream.write(payload)
        private_index[path.name] = hashlib.sha256(payload).hexdigest()

    def start(command, env, log_name):
        check()
        log = (run/log_name).open('xb')
        streams.append(log)
        # timeout owns its child group, so even a lost parent has a hard lifetime.
        seconds = max(1, int(LIMITS['seconds']-(time.monotonic()-started)-60))
        process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=20', str(seconds), *command],
            cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    def terminate(signum, frame):
        raise SystemExit(128+signum)
    old_handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
    old_alarm = None
    watchdog_stop = threading.Event()
    watchdog_errors = []
    def watchdog():
        while not watchdog_stop.wait(.5):
            try:
                check()
            except BaseException as exc:
                watchdog_errors.append(type(exc).__name__)
                os.kill(os.getpid(), signal.SIGALRM)
                return
    try:
        neo_base = BASE/'external/version_transfer_neo4j_5260_v1'
        neo = neo_base/'database_runtime/neo4j-community-5.26.0'
        java = neo_base/'java_runtime/zulu21.54.107-ca-jre21.0.13-linux_x64'
        conf = database_config(run, neo)
        env = dict(os.environ, HOME=str(run/'home'), TMPDIR=str(run/'tmp'), XDG_CACHE_HOME=str(run/'cache'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_CACHE=str(BASE.parent/'hf_cache/hub'),
            GRAPHITI_TELEMETRY_ENABLED='false', EMBEDDING_DIM='384', PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env[key] = ''
        env.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        for key in ('GIT_DIR', 'GIT_WORK_TREE'):
            env.pop(key, None)
        for key in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                    'GRAPHITI_TELEMETRY_ENABLED', 'EMBEDDING_DIM', 'NO_PROXY', 'no_proxy',
                    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            os.environ[key] = env[key]
        # Parent watchdog also covers synchronous startup/readiness and hung generation.
        old_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('wall ceiling')))
        signal.alarm(LIMITS['seconds']-CLEANUP_SECONDS)
        threading.Thread(target=watchdog, daemon=True).start()
        db_env = dict(env, JAVA_HOME=str(java), NEO4J_HOME=str(neo), NEO4J_CONF=str(conf),
            JAVA_TOOL_OPTIONS=f'-Djava.io.tmpdir={run/"tmp"} -XX:-UsePerfData')
        database = start(['bash', str(neo/'bin/neo4j'), 'console'], db_env, 'database-console.log')
        wait_port(database, 17687, 120)

        async def trajectory():
            runtime = await make_runtime(cases, sink, check)
            runtime_box.append(runtime)
            try:
                # Direct tuple-driver/Cypher + guarded EMPTY cleanup before any model load.
                await asyncio.wait_for(initialize_empty(runtime, cases), timeout=120)
                report['empty_state_preflight'] = list(runtime.preflight)
                check()
                report['gpu_before_load'] = fresh_gpu(args.gpu_uuid)
                available(18043)
                model = start(service_command(run, args.gpu_uuid,
                    max(1, int(LIMITS['seconds']-(time.monotonic()-started)-CLEANUP_SECONDS))), env, 'model-console.log')
                wait_port(model, 18043, 600)
                from urllib.request import urlopen
                with urlopen('http://127.0.0.1:18043/health', timeout=5) as response:
                    health = json.load(response)
                if health.get('model') != MODEL_ID or not health.get('supports_json_schema'):
                    raise ValueError('model/schema health mismatch')
                remaining = LIMITS['seconds']-(time.monotonic()-started)-CLEANUP_SECONDS
                await execute_cases(cases, runtime, runtime_revision=args.runtime_revision,
                    seconds=remaining, report=report, close=False)
            finally:
                # Startup failures still attempt guarded cleanup for every owned namespace.
                for index, case in enumerate(cases):
                    try:
                        await asyncio.wait_for(runtime.witness.delete_namespace(case.runtime_namespace), 15)
                        empty = await asyncio.wait_for(runtime.witness.verify_empty(case.runtime_namespace), 5)
                        report['cases'][index]['namespace_empty_after_cleanup'] = empty
                        if not empty:
                            raise ValueError('startup/final cleanup incomplete')
                    except BaseException as exc:
                        report['cases'][index]['namespace_empty_after_cleanup'] = False
                        blocked(report, exc, next((i for i, c in enumerate(report['cases'])
                            if c['status'] != 'OBSERVED'), 2), field='startup_cleanup_error_type')
                errors = await close_runtime(runtime)
                report['client_cleanup_errors'] = errors
                if errors:
                    blocked(report, RuntimeError(), next((i for i, c in enumerate(report['cases'])
                        if c['status'] != 'OBSERVED'), 2), field='client_cleanup_error_type')
        asyncio.run(trajectory())
    except BaseException as exc:
        blocked(report, exc, next((i for i, c in enumerate(report['cases']) if c['status'] != 'OBSERVED'), 2))
    finally:
        watchdog_stop.set()
        report['limit_watchdog_errors'] = watchdog_errors
        if old_alarm is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_alarm)
        cleanup = []
        try:
            owned_pids = owned_group_pids(processes)
            report['owned_child_pid_count'] = len(owned_pids)
        except BaseException as exc:
            owned_pids = None
            report['owned_child_inspection_error_type'] = type(exc).__name__
        for process in reversed(processes):
            try:
                stopped = stop_owned(process)
                if owned_group_pids([process]):
                    # The timeout wrapper may already have exited; its remaining
                    # session group is still ours, never another user's job.
                    os.killpg(process.pid, signal.SIGTERM)
                    deadline = time.monotonic()+5
                    while owned_group_pids([process]) and time.monotonic() < deadline:
                        time.sleep(.1)
                    if owned_group_pids([process]):
                        os.killpg(process.pid, signal.SIGKILL)
                cleanup.append(stopped)
            except BaseException as exc:
                cleanup.append(False)
                report['process_cleanup_error_type'] = type(exc).__name__
        report['owned_process_cleanup'] = cleanup
        for stream in streams:
            stream.close()
        report['ports_closed'] = {}
        for port in (17687, 18043):
            with socket.socket() as sock:
                report['ports_closed'][str(port)] = sock.connect_ex(('127.0.0.1', port)) != 0
        try:
            pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, timeout=10).split()
            report['owned_gpu_process_absent'] = (owned_pids is not None
                and not any(str(pid) in pids for pid in owned_pids)
                and not owned_group_pids(processes))
        except BaseException:
            report['owned_gpu_process_absent'] = None
        runtime = runtime_box[0] if runtime_box else None
        report['llm_ledger_audit'] = None
        try:
            if (run/'llm-calls.jsonl').exists():
                report['llm_ledger_audit'] = audit_llm_ledger(run/'llm-calls.jsonl')
            report['accounting'] = audit_current(run, runtime)
        except BaseException as exc:
            blocked(report, exc, next((i for i, c in enumerate(report['cases']) if c['status'] != 'OBSERVED'), 2),
                    field='accounting_error_type')
        if runtime is not None:
            write_new(run/'compact/observations.json', runtime.recorder.rows)
            write_new(run/'compact/transport.json', runtime.observer.receipts)
            report['observations_sha256'] = sha(run/'compact/observations.json')
            report['transport_sha256'] = sha(run/'compact/transport.json')
        write_new(run/'private/index.json', private_index)
        report['private_artifacts'] = {'count': len(private_index), 'index_sha256': sha(run/'private/index.json')}
        report['counts'] = {key: sum(c[key] for c in report['cases']) for key in _COUNT_KEYS}
        report['elapsed_seconds'] = time.monotonic()-started
        report['output_bytes'] = output_bytes(run)
        if (not all(cleanup) or not all(report['ports_closed'].values())
                or report['owned_gpu_process_absent'] is not True or report['output_bytes'] > STORAGE_BYTES
                or report['elapsed_seconds'] > LIMITS['seconds']):
            blocked(report, RuntimeError(), next((i for i, c in enumerate(report['cases']) if c['status'] != 'OBSERVED'), 2),
                    field='terminal_cleanup_or_limit_error_type')
        write_new(run/'terminal.json', report)
        write_new(RESERVATION/'terminal.json', {'status': report['status'], 'terminal_sha256': sha(run/'terminal.json')})
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report['status'] == 'OBSERVATIONS_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
