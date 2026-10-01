"""Test the frozen answer-replay lifecycle; production remains a no-execution gate."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mub.vnext.external.security import scan_for_secrets
from mub.vnext.io.atomic import publish_files_atomically
from scripts import vnext_prepare_cross_domain_answer_replay as preparation

PREPARATION = ROOT / 'results/vnext/cross_domain_answer_replay_20261002_v2'
PREPARATION_INDEX = '1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687'
SCHEMA = 'memupdatebench.cross-domain.answer-execution-engineering.v1'
CONTEXT_CAP = 4096
MAX_NEW_TOKENS = 64
DECODE = {'max_new_tokens': MAX_NEW_TOKENS, 'seed': 0, 'do_sample': False, 'num_beams': 1}
BUDGETS = {'prepare': 30.0, 'load': 120.0, 'generate': 60.0, 'close': 30.0}
BLOCKERS = ['current_device_authorization', 'replay_specific_runtime_qualification',
            'independently_bound_tokenizer_preflight', 'qualified_production_backend',
            'owned_process_hard_watchdog']
ERROR_TYPES = {'ValueError', 'TypeError', 'RuntimeError', 'MemoryError', 'TimeoutError',
               'OSError', 'FileNotFoundError', 'PermissionError', 'KeyboardInterrupt', 'SystemExit',
               'AssertionError'}
canonical, sha, require = preparation.canonical, preparation.sha, preparation.require


@dataclass(frozen=True)
class PublicRequest:
    request_id: str
    visible_prompt: str


@dataclass(frozen=True)
class PreparedInput:
    request_id: str
    rendered_text: str
    template_sha256: str
    input_tokens: int


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    input_tokens: int
    generated_tokens: int


class ReplayBackend(Protocol):
    def prepare(self, request: PublicRequest) -> PreparedInput: ...
    def load(self) -> None: ...
    def generate(self, prepared: PreparedInput, *, max_new_tokens: int, seed: int,
                 do_sample: bool, num_beams: int) -> GenerationOutput: ...
    def close(self) -> None: ...


def _error_type(exc):
    name = type(exc).__name__
    return name if name in ERROR_TYPES else 'OtherError'


def _hex(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None and value != '0' * 64


def _time_call(callback, clock, seconds, on_enter=lambda: None):
    started = clock()
    require(type(started) in (int, float) and math.isfinite(started), 'invalid clock observation')
    on_enter()
    value = callback()
    elapsed = clock() - started
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'invalid elapsed time')
    if elapsed > seconds:
        raise TimeoutError('cooperative callback budget exceeded')
    return value, float(elapsed)


def _prepared_metadata(request, prepared):
    require(type(prepared) is PreparedInput and prepared.request_id == request.request_id,
            'prepared request identity mismatch')
    require(type(prepared.rendered_text) is str and prepared.rendered_text
            and len(prepared.rendered_text.encode('utf-8')) <= 1024**2, 'invalid rendered prompt size')
    require(_hex(prepared.template_sha256), 'invalid template hash')
    require(type(prepared.input_tokens) is int and 0 < prepared.input_tokens
            and prepared.input_tokens + MAX_NEW_TOKENS <= CONTEXT_CAP, 'context token budget exceeded')
    return {'request_id': request.request_id, 'visible_prompt_sha256': sha(request.visible_prompt.encode('utf-8')),
            'rendered_prompt_sha256': sha(prepared.rendered_text.encode('utf-8')),
            'template_sha256': prepared.template_sha256, 'input_tokens': prepared.input_tokens}


def _output_metadata(output, expected_tokens):
    require(type(output) is GenerationOutput, 'generation output contract mismatch')
    require(type(output.text) is str and len(output.text.encode('utf-8')) <= 16384, 'generation text exceeds byte limit')
    require(type(output.input_tokens) is int and output.input_tokens == expected_tokens, 'input token usage drift')
    require(type(output.generated_tokens) is int and 0 <= output.generated_tokens <= MAX_NEW_TOKENS,
            'invalid generated token usage')
    return {'output_sha256': sha(output.text.encode('utf-8')), 'input_tokens': output.input_tokens,
            'generated_tokens': output.generated_tokens, 'output_cap_reached': output.generated_tokens == MAX_NEW_TOKENS}


class _Journal:
    def __init__(self, root):
        self.root = root
        self.root_id = (root.stat().st_dev, root.stat().st_ino)
        self.path = root / 'journal.jsonl'
        self.stream = self.path.open('xb')
        self.file_id = (self.path.stat().st_dev, self.path.stat().st_ino)
        self.records = []

    def append(self, event):
        import os
        preparation.safe_path(self.path)
        require((self.root.stat().st_dev, self.root.stat().st_ino) == self.root_id
                and (self.path.stat().st_dev, self.path.stat().st_ino) == self.file_id, 'journal ownership changed')
        require(not scan_for_secrets(event), 'journal metadata failed secret scan')
        record = {'sequence': len(self.records), 'previous_sha256': sha(self.records[-1]) if self.records else None,
                  'event': event}
        raw = canonical(record)
        self.stream.write(raw)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.append(raw)

    def close(self):
        self.stream.close()

    def assert_unchanged(self):
        preparation.safe_path(self.path)
        require(self.path.read_bytes() == b''.join(self.records), 'journal contents changed')


def _rows(plans):
    return [{'request_id': p['request_id'], 'task_id': p['task_id'], 'query_id': p['query_id'],
             'release_name': p['release_name'], 'repetition_index': p['repetition_index'],
             'visible_prompt_sha256': p['visible_prompt_sha256'], 'retrieval_sha256': p['retrieval_sha256'],
             'status': 'NOT_RUN', 'reason_code': 'not_attempted', 'attempts': 0, 'attempt_intents': 0,
             'input_tokens': None, 'generated_tokens': None, 'output_sha256': None,
             'output_cap_reached': None, 'generation_seconds': None, 'score': None}
            for p in plans]


def run_replay(bundle, preparation_root, output_root, *, mode='production',
               backend_factory: Callable[[], ReplayBackend] | None = None, clock=time.monotonic):
    require(mode in ('production', 'injected_test_only'), 'execution mode is not allowed')
    prepared_root = preparation.safe_path(preparation_root)
    preparation.read_preparation(prepared_root, PREPARATION_INDEX, bundle)
    plans = preparation.build_plans(bundle)
    preparation_bytes = preparation.snapshot(prepared_root)
    source_sha = sha(Path(__file__).read_bytes())
    output = preparation._output_root(output_root, bundle)
    require(output != prepared_root and prepared_root not in output.parents and output not in prepared_root.parents,
            'output overlaps frozen preparation')
    if mode == 'injected_test_only':
        require(callable(backend_factory), 'test execution requires an explicit injected backend')
    requests = [PublicRequest(p['request_id'], preparation.render_plan(bundle, p)) for p in plans]
    output.parent.mkdir(parents=True, exist_ok=True)
    preparation.safe_path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    journal = _Journal(output)
    rows = _rows(plans)
    life = {'constructor_attempts': 0, 'constructor_status': 'NOT_RUN', 'prepare_calls': 0,
            'constructor_intents': 0, 'prepare_intents': 0, 'load_intents': 0, 'generation_intents': 0,
            'load_attempts': 0, 'load_status': 'NOT_RUN', 'generation_attempts': 0,
            'close_attempts': 0, 'close_status': 'NOT_RUN', 'error_phase': None, 'error_type': None}
    backend = None
    phase, active, stopped = 'initialization', None, mode == 'production'
    fatal_journal_error = None
    preflight = []
    bindings_valid = True
    scoring_tasks = dict(bundle.tasks)

    def entered(counter):
        life[counter] += 1

    def revalidate():
        bundle.assert_unchanged()
        require(preparation.snapshot(prepared_root) == preparation_bytes, 'frozen preparation changed')
        require(sha(Path(__file__).read_bytes()) == source_sha, 'execution source changed')

    try:
        journal.append({'phase': 'planned', 'mode': mode, 'preparation_index_sha256': PREPARATION_INDEX,
                        'request_ids': [r.request_id for r in requests], 'maximum_attempts': 4, 'retries': 0})
        if mode == 'production':
            for row in rows:
                row['reason_code'] = 'production_qualification_and_authorization_missing'
        else:
            phase = 'construction'
            life['constructor_intents'] = 1
            journal.append({'phase': 'before_construction'})
            life['constructor_attempts'] = 1
            backend = backend_factory()
            require(backend is not None, 'backend factory returned no handle')
            life['constructor_status'] = 'RETURNED'
            revalidate()
            phase = 'template_preflight'
            for request in requests:
                life['prepare_intents'] += 1
                journal.append({'phase': 'before_template_preflight', 'request_id': request.request_id})
                rendered, elapsed = _time_call(lambda: backend.prepare(request), clock, BUDGETS['prepare'],
                                               lambda: entered('prepare_calls'))
                revalidate()
                metadata = _prepared_metadata(request, rendered)
                preflight.append(metadata)
                journal.append({'phase': 'template_preflight_complete', 'binding': metadata, 'seconds': elapsed})
            require(len({p['template_sha256'] for p in preflight}) == 1, 'template changed between requests')
            revalidate()
            phase = 'load'
            life['load_intents'] = 1
            journal.append({'phase': 'before_load', 'intent': 1})
            def load_entered():
                life['load_attempts'] += 1
                life['load_status'] = 'ATTEMPTED'
            _, elapsed = _time_call(backend.load, clock, BUDGETS['load'], load_entered)
            revalidate()
            life['load_status'] = 'RETURNED'
            life['load_seconds'] = elapsed
            journal.append({'phase': 'load_returned', 'seconds': elapsed})
            for plan, request, expected, row in zip(plans, requests, preflight, rows, strict=True):
                active = row
                phase = 'input_revalidation'
                revalidate()
                phase = 'template_recheck'
                life['prepare_intents'] += 1
                rendered, elapsed = _time_call(lambda: backend.prepare(request), clock, BUDGETS['prepare'],
                                               lambda: entered('prepare_calls'))
                require(_prepared_metadata(request, rendered) == expected, 'loaded template/token binding differs from preflight')
                revalidate()
                row['input_tokens'] = expected['input_tokens']
                phase = 'generation'
                row['attempt_intents'] = 1
                life['generation_intents'] += 1
                journal.append({'phase': 'before_generation', 'row': row})
                def generation_entered():
                    row.update(status='ATTEMPTED', reason_code=None, attempts=1)
                    life['generation_attempts'] += 1
                generated, elapsed = _time_call(lambda: backend.generate(rendered, **DECODE), clock, BUDGETS['generate'],
                                                generation_entered)
                row.update(_output_metadata(generated, expected['input_tokens']))
                row['generation_seconds'] = elapsed
                journal.append({'phase': 'generation_returned', 'row': row})
                phase = 'parsing_scoring'
                revalidate()
                row['score'] = preparation.score_readout(scoring_tasks[plan['release_name']],
                                                        status='COMPLETED', raw_output=generated.text)
                row['status'] = 'COMPLETED'
                journal.append({'phase': 'readout_completed', 'row': row})
                active = None
    except BaseException as exc:
        stopped = True
        life['error_phase'], life['error_type'] = phase, _error_type(exc)
        if phase == 'load':
            life['load_status'] = 'FAILED' if life['load_attempts'] else 'NOT_ENTERED'
        if phase == 'construction':
            life['constructor_status'] = 'FAILED' if life['constructor_attempts'] else 'NOT_ENTERED'
        if active is not None:
            active.update(status='TECHNICAL_FAILURE', reason_code=phase + '_failed', score=None)
        for row in rows:
            if row['status'] == 'NOT_RUN':
                row['reason_code'] = 'previous_technical_failure'
        try:
            journal.append({'phase': 'technical_failure', 'error_phase': phase, 'error_type': _error_type(exc)})
        except BaseException as write_exc:
            fatal_journal_error = write_exc
    finally:
        if backend is not None:
            life['close_attempts'] = 1
            life['close_status'] = 'ATTEMPTED'
            try:
                journal.append({'phase': 'before_close'})
            except BaseException as write_exc:
                fatal_journal_error = write_exc
            close_started = None
            try:
                close_started = clock()
                require(type(close_started) in (int, float) and math.isfinite(close_started), 'invalid cleanup clock')
            except BaseException as exc:
                stopped = True
                close_started = None
                life['close_timing_error_type'] = _error_type(exc)
            try:
                backend.close()
                life['close_status'] = 'RETURNED'
                if close_started is not None:
                    elapsed = clock() - close_started
                    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                            'invalid cleanup duration')
                    life['close_seconds'] = float(elapsed)
                    if elapsed > BUDGETS['close']:
                        raise TimeoutError('cooperative cleanup budget exceeded')
            except BaseException as exc:
                stopped = True
                life['close_status'] = 'FAILED'
                life['close_error_type'] = _error_type(exc)
        elif life['constructor_attempts']:
            life['close_status'] = 'UNAVAILABLE_NO_HANDLE'
        try:
            revalidate()
        except BaseException as exc:
            stopped = True
            bindings_valid = False
            life['revalidation_error_type'] = _error_type(exc)
        try:
            journal.append({'phase': 'terminal', 'backend_lifecycle': life, 'rows': rows,
                            'input_bindings_unchanged': bindings_valid})
        except BaseException as write_exc:
            fatal_journal_error = write_exc
        journal.close()
    if fatal_journal_error is not None:
        raise RuntimeError('terminal journal publication failed after cleanup') from fatal_journal_error
    journal.assert_unchanged()
    complete = (mode == 'injected_test_only' and not stopped
                and life['close_status'] == 'RETURNED' and all(r['status'] == 'COMPLETED' for r in rows))
    completed = [r for r in rows if r['status'] == 'COMPLETED']
    attempted = [r for r in rows if r['attempts']]
    known = [r['generated_tokens'] for r in attempted if r['generated_tokens'] is not None]
    unknown = len(attempted) - len(known)
    unmeasured = None if life['constructor_attempts'] else 0
    result = {'schema': SCHEMA, 'status': ('BLOCKED' if mode == 'production' else
                                         'TEST_ONLY_COMPLETE' if complete else 'TEST_ONLY_BLOCKED'),
        'mode': mode, 'evidence_class': 'answer_executor_engineering_only',
        'preparation_index_sha256': PREPARATION_INDEX, 'runner_source_sha256': source_sha,
        'backend_lifecycle': life, 'rows': rows, 'template_preflight': preflight,
        'template_qualification_scope': 'same_injected_backend_comparison_not_independent_runtime_qualification',
        'resource_accounting': {name: unmeasured for name in ('model_loads', 'generations', 'tokenizer_loads',
                                    'network_calls', 'gpu_calls', 'provider_calls', 'child_process_launches')},
        'usage': {'observed_generated_tokens': sum(known), 'unknown_generation_attempts': unknown,
                  'generated_tokens_total': None if unknown else sum(known),
                  'input_tokens_total_observed': sum(r['input_tokens'] or 0 for r in attempted)},
        'retries': 0, 'maximum_requests': 4, 'maximum_generated_tokens': 256,
        'decoding': dict(DECODE), 'context_token_cap': CONTEXT_CAP,
        'test_control_summary': ({'completed': len(completed),
            'format_valid': sum(r['score']['format_valid'] is True for r in completed),
            'typed_exact_matches': sum(r['score']['typed_exact_match'] is True for r in completed)} if complete else None),
        'answer_metrics': None, 'scientific_evidence': False, 'scientific_release_allowed': False,
        'benchmark_accuracy_claimed': False, 'execution_authorized': False,
        'cluster_n': 2, 'cluster_unit': 'source_trajectory', 'input_bindings_unchanged': bindings_valid,
        'deadline_enforcement': 'COOPERATIVE_TEST_ONLY_NOT_HARD_WATCHDOG', 'budgets_seconds': BUDGETS,
        'blockers': BLOCKERS, 'canonical_score_record': False,
        'claim_boundary': 'Injected-callback lifecycle controls only; no actual answer-model capability or accuracy evidence.'}
    require(not scan_for_secrets(result), 'execution metadata failed secret scan')
    row_bytes = b''.join(canonical(row) for row in rows)
    payload = {'summary.json': canonical({k:v for k,v in result.items() if k != 'rows'}), 'rows.jsonl': row_bytes}
    index = {'schema': SCHEMA + '.index', 'status': result['status'], 'scientific_release_allowed': False,
             'artifacts': [{'path': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(payload.items())] +
                          [{'path': 'journal.jsonl', 'bytes': journal.path.stat().st_size, 'sha256': sha(journal.path.read_bytes())}]}
    payload['index.json'] = canonical(index)
    def before_publish():
        journal.assert_unchanged()
        require((output.stat().st_dev, output.stat().st_ino) == journal.root_id, 'output ownership changed')
        if complete or mode == 'production':
            revalidate()
    publish_files_atomically({output / name: raw for name, raw in payload.items()}, overwrite=False,
                              source_paths=(journal.path,), pre_publish=before_publish)
    result['index_sha256'] = sha(payload['index.json'])
    verify_execution_output(output, result['index_sha256'])
    return result


def _same(left, right, message):
    require(canonical(left) == canonical(right), message)


def _plan_contract():
    raw = preparation.read_file(PREPARATION / 'index.json')
    require(sha(raw) == PREPARATION_INDEX, 'frozen preparation index drift')
    index = preparation.strict_json(raw)
    expected = next(r for r in index['artifacts'] if r['path'] == 'plans.jsonl')
    raw = preparation.read_file(PREPARATION / 'plans.jsonl')
    require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], 'frozen plan hash drift')
    return [preparation.strict_json(line) for line in raw.splitlines()]


def _validate_execution_semantics(summary, rows, events):
    require(summary['schema'] == SCHEMA and summary['mode'] in ('production', 'injected_test_only'),
            'execution schema/mode mismatch')
    _same({k: summary[k] for k in ('answer_metrics', 'scientific_evidence', 'scientific_release_allowed',
        'benchmark_accuracy_claimed', 'execution_authorized', 'cluster_n', 'cluster_unit', 'canonical_score_record',
        'retries', 'maximum_requests', 'maximum_generated_tokens', 'preparation_index_sha256')},
        {'answer_metrics': None, 'scientific_evidence': False, 'scientific_release_allowed': False,
         'benchmark_accuracy_claimed': False, 'execution_authorized': False, 'cluster_n': 2,
         'cluster_unit': 'source_trajectory', 'canonical_score_record': False, 'retries': 0,
         'maximum_requests': 4, 'maximum_generated_tokens': 256, 'preparation_index_sha256': PREPARATION_INDEX},
        'execution contract boundary mismatch')
    require(summary['deadline_enforcement'] == 'COOPERATIVE_TEST_ONLY_NOT_HARD_WATCHDOG', 'deadline scope drift')
    _same(summary['budgets_seconds'], BUDGETS, 'execution budget drift')
    _same(summary['decoding'], DECODE, 'decoding settings drift')
    _same(summary['context_token_cap'], CONTEXT_CAP, 'context budget drift')
    _same(summary['blockers'], BLOCKERS, 'production blocker drift')
    life = summary['backend_lifecycle']
    _same(events[-1]['input_bindings_unchanged'], summary['input_bindings_unchanged'], 'terminal binding mismatch')
    plans = _plan_contract()
    require(len(rows) == len(plans) == 4, 'four unique planned rows required')
    for row, plan in zip(rows, plans, strict=True):
        for key in ('request_id', 'task_id', 'query_id', 'release_name', 'repetition_index',
                    'visible_prompt_sha256', 'retrieval_sha256'):
            _same(row[key], plan[key], 'execution row identity/order mismatch')
        require(row['status'] in ('NOT_RUN', 'TECHNICAL_FAILURE', 'COMPLETED'), 'row is not terminal')
        require(type(row['attempts']) is int and type(row['attempt_intents']) is int
                and 0 <= row['attempts'] <= row['attempt_intents'] <= 1, 'invalid one-attempt row accounting')
        if row['input_tokens'] is not None:
            require(type(row['input_tokens']) is int and 0 < row['input_tokens'] <= CONTEXT_CAP - MAX_NEW_TOKENS,
                    'input token budget mismatch')
        if row['generated_tokens'] is not None:
            require(row['attempts'] == 1 and type(row['generated_tokens']) is int
                    and 0 <= row['generated_tokens'] <= MAX_NEW_TOKENS and _hex(row['output_sha256']),
                    'generated token budget/identity mismatch')
            _same(row['output_cap_reached'], row['generated_tokens'] == MAX_NEW_TOKENS, 'output cap flag mismatch')
        else:
            require(row['output_cap_reached'] is None and row['output_sha256'] is None,
                    'unknown usage carries a completed output binding')
        if row['status'] == 'COMPLETED':
            require(row['attempts'] == 1 and row['generated_tokens'] is not None and row['reason_code'] is None,
                    'completed row did not finish its one attempt')
            score = row['score']
            require(type(score) is dict and score['status'] == 'COMPLETED'
                    and type(score['format_valid']) is bool and type(score['typed_exact_match']) is bool
                    and score['included_in_denominator'] is True and score['canonical_score_record'] is False
                    and score['output_sha256'] == row['output_sha256'], 'completed score boundary mismatch')
        else:
            require(row['score'] is None and type(row['reason_code']) is str and row['reason_code'],
                    'technical/not-run row has a score or lacks reason')
        if row['status'] == 'NOT_RUN':
            require(row['attempts'] == 0, 'not-run row contains an attempt')
    for counter, ceiling in (('constructor_attempts', 1), ('constructor_intents', 1), ('load_attempts', 1),
                             ('load_intents', 1), ('close_attempts', 1), ('prepare_calls', 8),
                             ('prepare_intents', 8), ('generation_attempts', 4), ('generation_intents', 4)):
        require(type(life[counter]) is int and 0 <= life[counter] <= ceiling, 'invalid lifecycle count')
    require(life['constructor_attempts'] <= life['constructor_intents']
            and life['load_attempts'] <= life['load_intents']
            and life['prepare_calls'] <= life['prepare_intents'], 'callback count exceeds recorded intent')
    _same(life['generation_attempts'], sum(r['attempts'] for r in rows), 'generation attempt count mismatch')
    _same(life['generation_intents'], sum(r['attempt_intents'] for r in rows), 'generation intent count mismatch')
    attempted = [row for row in rows if row['attempts']]
    known = [r['generated_tokens'] for r in attempted if r['generated_tokens'] is not None]
    unknown = len(attempted) - len(known)
    _same(summary['usage'], {'observed_generated_tokens': sum(known), 'unknown_generation_attempts': unknown,
        'generated_tokens_total': None if unknown else sum(known),
        'input_tokens_total_observed': sum(r['input_tokens'] or 0 for r in attempted)}, 'token usage summary mismatch')
    unmeasured = None if life['constructor_attempts'] else 0
    _same(summary['resource_accounting'], {name: unmeasured for name in ('model_loads', 'generations',
        'tokenizer_loads', 'network_calls', 'gpu_calls', 'provider_calls', 'child_process_launches')},
        'unmeasured callback resources cannot be zero')
    if summary['mode'] == 'production':
        require(summary['status'] == 'BLOCKED' and life['constructor_attempts'] == life['load_attempts']
                == life['generation_attempts'] == life['close_attempts'] == 0
                and all(r['status'] == 'NOT_RUN' for r in rows) and summary['test_control_summary'] is None,
                'production branch executed or claimed completion')
    else:
        complete = (all(r['status'] == 'COMPLETED' for r in rows) and life['constructor_status'] == 'RETURNED'
                    and life['load_status'] == 'RETURNED' and life['close_status'] == 'RETURNED'
                    and life['error_type'] is None and 'close_timing_error_type' not in life
                    and 'close_error_type' not in life and summary['input_bindings_unchanged'] is True)
        _same(summary['status'], 'TEST_ONLY_COMPLETE' if complete else 'TEST_ONLY_BLOCKED', 'test completion mismatch')
        controls = {'completed': 4, 'format_valid': sum(r['score']['format_valid'] is True for r in rows),
                    'typed_exact_matches': sum(r['score']['typed_exact_match'] is True for r in rows)} if complete else None
        _same(summary['test_control_summary'], controls, 'test control summary mismatch')
    require(not scan_for_secrets(summary) and not scan_for_secrets(rows), 'public metadata secret scan failed')


def verify_execution_output(root, expected_sha):
    root = preparation.safe_path(root)
    require(root.is_dir() and {p.name for p in root.iterdir()} == {'index.json', 'summary.json', 'rows.jsonl', 'journal.jsonl'},
            'execution output membership mismatch')
    raw = preparation.read_file(root / 'index.json')
    require(sha(raw) == expected_sha, 'execution index hash mismatch')
    index = preparation.strict_json(raw)
    require(index['schema'] == SCHEMA + '.index' and index['scientific_release_allowed'] is False
            and len(index['artifacts']) == 3 and canonical(index) == raw, 'execution index schema/membership mismatch')
    require({r['path'] for r in index['artifacts']} == {'summary.json', 'rows.jsonl', 'journal.jsonl'}, 'execution index members mismatch')
    for record in index['artifacts']:
        content = preparation.read_file(root / record['path'])
        require(len(content) == record['bytes'] and sha(content) == record['sha256'], 'execution artifact hash mismatch')
    previous = None
    events = []
    for number, line in enumerate(preparation.read_file(root / 'journal.jsonl').splitlines()):
        record = preparation.strict_json(line)
        require(record['sequence'] == number and record['previous_sha256'] == previous,
                'journal hash chain mismatch')
        require(canonical(record) == line + b'\n', 'noncanonical journal')
        previous = sha(line + b'\n')
        events.append(record['event'])
    require(events and events[-1]['phase'] == 'terminal', 'terminal journal record missing')
    rows = [preparation.strict_json(line) for line in preparation.read_file(root / 'rows.jsonl').splitlines()]
    summary = preparation.strict_json(preparation.read_file(root / 'summary.json'))
    require(events[-1]['rows'] == rows and len(rows) == 4, 'terminal row coverage mismatch')
    require(summary['status'] == index['status'] and summary['scientific_evidence'] is False
            and summary['scientific_release_allowed'] is False and summary['answer_metrics'] is None,
            'execution scientific boundary mismatch')
    require(events[-1]['backend_lifecycle'] == summary['backend_lifecycle'], 'lifecycle/journal mismatch')
    _validate_execution_semantics(summary, rows, events)
    return {**summary, 'rows': rows, 'index_sha256': expected_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = preparation.load_inputs(preparation.DEFAULT_EVIDENCE, preparation.DEFAULT_BEA, preparation.DEFAULT_NOAA)
        result = run_replay(bundle, PREPARATION, args.output_root)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error_type': _error_type(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'template_preflight')}, sort_keys=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
