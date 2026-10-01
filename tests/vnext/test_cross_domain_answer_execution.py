from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import vnext_run_cross_domain_answer_replay as execution
from scripts import vnext_prepare_cross_domain_answer_replay as preparation

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / 'results/vnext/cross_domain_answer_replay_20261002_v2'


@pytest.fixture(scope='module')
def admitted_bundle():
    return preparation.load_inputs(preparation.DEFAULT_EVIDENCE, preparation.DEFAULT_BEA, preparation.DEFAULT_NOAA)


@pytest.fixture
def bundle(admitted_bundle):
    return deepcopy(admitted_bundle)


class FakeBackend:
    def __init__(self, outputs=None, fail=None, drift=False):
        self.outputs = outputs or ['{"disposition":"answered","answer":2.4}'] * 2 + ['{"disposition":"answered","answer":75}'] * 2
        self.fail = fail
        self.drift = drift
        self.loaded = False
        self.closed = False
        self.load_calls = self.generation_calls = self.prepare_calls = 0
        self.requests = []

    def prepare(self, request):
        assert {field.name for field in fields(request)} == {'request_id', 'visible_prompt'}
        self.requests.append(request)
        self.prepare_calls += 1
        rendered = 'TEST_CHAT\n' + request.visible_prompt
        if self.loaded and self.drift:
            rendered += '\nCHANGED'
        return execution.PreparedInput(request.request_id, rendered, '1' * 64, 100)

    def load(self):
        self.load_calls += 1
        if self.fail == 'load':
            raise MemoryError('private load message /home/person/secret')
        self.loaded = True

    def generate(self, prepared, *, max_new_tokens, seed, do_sample, num_beams):
        assert self.loaded and not self.closed
        assert (max_new_tokens, seed, do_sample, num_beams) == (64, 0, False, 1)
        self.generation_calls += 1
        if self.fail == 'generate' and self.generation_calls == 2:
            raise TimeoutError('secret output should never be persisted')
        return execution.GenerationOutput(self.outputs[self.generation_calls - 1], prepared.input_tokens, 12)

    def close(self):
        self.closed = True
        if self.fail == 'close':
            raise RuntimeError('raw unload secret')


def run(bundle, tmp_path, backend, **kwargs):
    return execution.run_replay(bundle, PREPARED, tmp_path / 'run',
        mode='injected_test_only', backend_factory=lambda: backend, **kwargs)


def test_default_production_gate_never_constructs_a_backend(bundle, tmp_path):
    def forbidden():
        raise AssertionError('backend must not be constructed')
    result = execution.run_replay(bundle, PREPARED, tmp_path / 'gate', backend_factory=forbidden)
    assert result['status'] == 'BLOCKED'
    assert result['mode'] == 'production'
    assert result['backend_lifecycle']['constructor_attempts'] == 0
    assert result['answer_metrics'] is None
    assert result['resource_accounting']['model_loads'] == 0
    assert len(result['rows']) == 4 and all(r['status'] == 'NOT_RUN' for r in result['rows'])
    assert all(r['score'] is None for r in result['rows'])
    assert 'current_device_authorization' in result['blockers']
    assert 'qualified_production_backend' in result['blockers']


def test_four_test_readouts_are_bound_and_not_scientific_results(bundle, tmp_path):
    backend = FakeBackend()
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_COMPLETE'
    assert backend.prepare_calls == 8 and backend.load_calls == 1 and backend.generation_calls == 4
    assert backend.closed
    assert result['scientific_evidence'] is False and result['answer_metrics'] is None
    assert result['benchmark_accuracy_claimed'] is False
    assert result['test_control_summary'] == {'completed': 4, 'format_valid': 4, 'typed_exact_matches': 4}
    assert result['resource_accounting']['model_loads'] is None
    assert result['resource_accounting']['network_calls'] is None
    assert result['usage']['observed_generated_tokens'] == 48
    assert result['usage']['unknown_generation_attempts'] == 0
    assert result['usage']['generated_tokens_total'] == 48
    assert result['retries'] == 0
    assert result['cluster_n'] == 2
    assert all(r['attempts'] == 1 for r in result['rows'])
    assert execution.verify_execution_output(tmp_path / 'run', result['index_sha256'])['status'] == result['status']
    with pytest.raises(FileExistsError):
        run(bundle, tmp_path, FakeBackend())


def test_each_attempt_is_durable_before_callback(bundle, tmp_path):
    backend = FakeBackend()
    original_load = backend.load
    original_generate = backend.generate
    def read_events():
        return [json.loads(l)['event'] for l in (tmp_path / 'run/journal.jsonl').read_bytes().splitlines()]
    def load():
        assert read_events()[-1]['phase'] == 'before_load'
        return original_load()
    def generate(*args, **kwargs):
        assert read_events()[-1]['phase'] == 'before_generation'
        return original_generate(*args, **kwargs)
    backend.load = load
    backend.generate = generate
    assert run(bundle, tmp_path, backend)['status'] == 'TEST_ONLY_COMPLETE'


@pytest.mark.parametrize('failure', ['load', 'generate', 'close'])
def test_failure_cleanup_and_partial_usage(bundle, tmp_path, failure):
    backend = FakeBackend(fail=failure)
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_BLOCKED'
    assert backend.closed
    assert result['test_control_summary'] is None and result['answer_metrics'] is None
    assert result['retries'] == 0
    if failure == 'load':
        assert backend.generation_calls == 0
        assert all(r['status'] == 'NOT_RUN' for r in result['rows'])
    elif failure == 'generate':
        assert backend.generation_calls == 2
        assert [r['status'] for r in result['rows']] == ['COMPLETED', 'TECHNICAL_FAILURE', 'NOT_RUN', 'NOT_RUN']
        assert result['rows'][1]['score'] is None
        assert result['usage']['observed_generated_tokens'] == 12
        assert result['usage']['unknown_generation_attempts'] == 1
        assert result['usage']['generated_tokens_total'] is None
    else:
        assert all(r['status'] == 'COMPLETED' for r in result['rows'])
        assert result['backend_lifecycle']['close_status'] == 'FAILED'


def test_constructor_failure_records_all_rows_without_fabricated_cleanup(bundle, tmp_path):
    def fail():
        raise OSError('constructor secret')
    result = execution.run_replay(bundle, PREPARED, tmp_path / 'run',
                                  mode='injected_test_only', backend_factory=fail)
    assert result['status'] == 'TEST_ONLY_BLOCKED'
    assert len(result['rows']) == 4
    assert result['backend_lifecycle']['close_status'] == 'UNAVAILABLE_NO_HANDLE'
    assert result['resource_accounting']['model_loads'] is None


def test_loaded_tokenizer_drift_prevents_generation(bundle, tmp_path):
    backend = FakeBackend(drift=True)
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_BLOCKED'
    assert backend.generation_calls == 0 and backend.closed
    assert result['backend_lifecycle']['error_phase'] == 'template_recheck'


def test_context_overflow_prevents_model_load(bundle, tmp_path):
    backend = FakeBackend()
    prepare = backend.prepare
    backend.prepare = lambda r: replace(prepare(r), input_tokens=4096)
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_BLOCKED' and backend.load_calls == 0
    assert backend.closed and backend.generation_calls == 0


@pytest.mark.parametrize('output', [
    '{"disposition":"answered","answer":"2.4"}',
    '```json\n{"disposition":"answered","answer":2.4}\n```',
    '{"disposition":"answered","answer":true}',
    '{"disposition":"abstained"}',
])
def test_completed_bad_answers_are_wrong_controls_not_technical_failures(bundle, tmp_path, output):
    result = run(bundle, tmp_path, FakeBackend(outputs=[output] * 4))
    assert result['status'] == 'TEST_ONLY_COMPLETE'
    assert all(r['status'] == 'COMPLETED' for r in result['rows'])
    assert result['test_control_summary']['typed_exact_matches'] == 0
    assert result['answer_metrics'] is None


def test_output_cap_and_usage_are_observed_without_retry(bundle, tmp_path):
    backend = FakeBackend()
    generate = backend.generate
    backend.generate = lambda *a, **kw: replace(generate(*a, **kw), generated_tokens=64)
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_COMPLETE'
    assert result['usage']['generated_tokens_total'] == 256
    assert all(r['output_cap_reached'] for r in result['rows'])
    assert backend.generation_calls == 4


@pytest.mark.parametrize('tokens', [65, -1, True, None])
def test_invalid_token_usage_blocks_following_attempts(bundle, tmp_path, tokens):
    backend = FakeBackend()
    generate = backend.generate
    backend.generate = lambda *a, **kw: replace(generate(*a, **kw), generated_tokens=tokens)
    result = run(bundle, tmp_path, backend)
    assert result['status'] == 'TEST_ONLY_BLOCKED'
    assert backend.generation_calls == 1 and backend.closed
    assert result['usage']['generated_tokens_total'] is None


def test_callback_overrun_is_labeled_cooperative_not_hard_watchdog(bundle, tmp_path):
    now = [0.0]
    backend = FakeBackend()
    generate = backend.generate
    def slow(*a, **kw):
        now[0] += 61
        return generate(*a, **kw)
    backend.generate = slow
    result = run(bundle, tmp_path, backend, clock=lambda: now[0])
    assert result['status'] == 'TEST_ONLY_BLOCKED'
    assert backend.generation_calls == 1 and backend.closed
    assert result['deadline_enforcement'] == 'COOPERATIVE_TEST_ONLY_NOT_HARD_WATCHDOG'
    assert result['backend_lifecycle']['error_type'] == 'TimeoutError'


def test_no_prompt_raw_output_or_private_exception_text_is_published(bundle, tmp_path):
    result = run(bundle, tmp_path, FakeBackend(fail='generate'))
    for path in (tmp_path / 'run').iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'TEST_CHAT' not in text and 'Use only the retrieved memory entries' not in text
        assert 'secret output should never' not in text
        assert '"raw_output"' not in text and '"parsed_answer"' not in text
        assert str(ROOT) not in text and '/NAS/' not in text
    assert result['status'] == 'TEST_ONLY_BLOCKED'


def test_bad_mode_or_preparation_never_constructs_backend(bundle, tmp_path):
    calls = []
    with pytest.raises(ValueError):
        execution.run_replay(bundle, PREPARED, tmp_path / 'invalid', mode='live', backend_factory=lambda: calls.append(1))
    import shutil
    copied = tmp_path / 'preparation'
    shutil.copytree(PREPARED, copied)
    (copied / 'plans.jsonl').write_bytes((copied / 'plans.jsonl').read_bytes() + b'\n')
    with pytest.raises(ValueError):
        execution.run_replay(bundle, copied, tmp_path / 'invalid2', mode='injected_test_only', backend_factory=lambda: calls.append(1))
    assert calls == []


def test_invalid_clock_never_prevents_cleanup_or_counts_unentered_load(bundle, tmp_path):
    backend = FakeBackend()
    clock_broken = [False]
    original_prepare = backend.prepare
    def prepare(request):
        result = original_prepare(request)
        if backend.prepare_calls == 4:
            clock_broken[0] = True
        return result
    backend.prepare = prepare
    def clock():
        if clock_broken[0]:
            raise RuntimeError('clock unavailable')
        return 0.0
    result = run(bundle, tmp_path, backend, clock=clock)
    assert backend.closed
    assert result['backend_lifecycle']['load_attempts'] == 0
    assert result['status'] == 'TEST_ONLY_BLOCKED'


def test_no_generation_after_prepare_callback_changes_admitted_inputs(bundle, tmp_path):
    backend = FakeBackend()
    original_prepare = backend.prepare
    def prepare(request):
        result = original_prepare(request)
        if backend.loaded:
            bundle.rows[0]['retrieval']['entries'][0]['value'] = 999
        return result
    backend.prepare = prepare
    result = run(bundle, tmp_path, backend)
    assert backend.generation_calls == 0 and backend.closed
    assert result['rows'][0]['score'] is None
    assert result['status'] == 'TEST_ONLY_BLOCKED'


def test_no_scoring_against_inputs_mutated_by_generation_callback(bundle, tmp_path):
    backend = FakeBackend()
    generate = backend.generate
    def mutate(*args, **kwargs):
        result = generate(*args, **kwargs)
        bundle.rows[0]['retrieval']['entries'][0]['value'] = 999
        return result
    backend.generate = mutate
    result = run(bundle, tmp_path, backend)
    assert backend.generation_calls == 1 and backend.closed
    assert result['rows'][0]['score'] is None
    assert result['rows'][0]['status'] == 'TECHNICAL_FAILURE'
    assert result['test_control_summary'] is None


def test_clock_failure_before_callback_records_intent_not_generation(bundle, tmp_path):
    backend = FakeBackend()
    def clock():
        path = tmp_path / 'run/journal.jsonl'
        events = [json.loads(l)['event'] for l in path.read_bytes().splitlines()] if path.exists() else []
        if events and events[-1]['phase'] == 'before_generation':
            raise RuntimeError('clock unavailable')
        return 0.0
    result = run(bundle, tmp_path, backend, clock=clock)
    assert backend.generation_calls == 0 and backend.closed
    assert result['rows'][0]['attempts'] == 0
    assert result['backend_lifecycle']['generation_attempts'] == 0
    assert result['usage']['unknown_generation_attempts'] == 0


def test_journal_failure_before_generation_closes_backend(bundle, tmp_path, monkeypatch):
    backend = FakeBackend()
    append = execution._Journal.append
    def fail_before_attempt(self, event):
        if event['phase'] == 'before_generation':
            raise OSError('disk failed before intent persistence')
        return append(self, event)
    monkeypatch.setattr(execution._Journal, 'append', fail_before_attempt)
    result = run(bundle, tmp_path, backend)
    assert backend.generation_calls == 0 and backend.closed
    assert result['rows'][0]['attempts'] == 0
    assert result['usage']['unknown_generation_attempts'] == 0


@pytest.fixture(scope='module')
def completed_run(admitted_bundle, tmp_path_factory):
    folder = tmp_path_factory.mktemp('answer-engine-recorded-test')
    result = run(deepcopy(admitted_bundle), folder, FakeBackend())
    return folder / 'run', result


@pytest.mark.parametrize('mutation', ['duplicate_request', 'bool_attempt', 'token_limit', 'usage', 'control_summary',
                                     'production_status', 'cleanup', 'terminal_binding', 'duplicate_artifact',
                                     'authorization', 'cluster_count'])
def test_reopen_rejects_semantic_contradictions_even_with_rehashed_files(completed_run, tmp_path, mutation):
    import shutil
    original, result = completed_run
    root = tmp_path / 'tampered'
    shutil.copytree(original, root)
    summary = json.loads((root / 'summary.json').read_bytes())
    rows = [json.loads(line) for line in (root / 'rows.jsonl').read_bytes().splitlines()]
    journal = [json.loads(line) for line in (root / 'journal.jsonl').read_bytes().splitlines()]
    if mutation == 'duplicate_request': rows[1]['request_id'] = rows[0]['request_id']
    elif mutation == 'bool_attempt': rows[0]['attempts'] = True
    elif mutation == 'token_limit': rows[0]['generated_tokens'] = 65
    elif mutation == 'usage': summary['usage']['generated_tokens_total'] = 999
    elif mutation == 'control_summary': summary['test_control_summary']['typed_exact_matches'] = 999
    elif mutation == 'production_status': summary['mode'] = 'production'
    elif mutation == 'cleanup': summary['backend_lifecycle']['close_status'] = 'FAILED'
    elif mutation == 'terminal_binding': journal[-1]['event']['input_bindings_unchanged'] = False
    elif mutation == 'authorization': summary['execution_authorized'] = True
    elif mutation == 'cluster_count': summary['cluster_n'] = 4
    journal[-1]['event']['rows'] = rows
    journal[-1]['event']['backend_lifecycle'] = summary['backend_lifecycle']
    previous = None
    raw_journal = []
    for record in journal:
        record['previous_sha256'] = previous
        raw = preparation.canonical(record)
        previous = preparation.sha(raw)
        raw_journal.append(raw)
    (root / 'journal.jsonl').write_bytes(b''.join(raw_journal))
    (root / 'rows.jsonl').write_bytes(b''.join(preparation.canonical(r) for r in rows))
    (root / 'summary.json').write_bytes(preparation.canonical(summary))
    index = json.loads((root / 'index.json').read_bytes())
    for row in index['artifacts']:
        raw = (root / row['path']).read_bytes()
        row.update(bytes=len(raw), sha256=preparation.sha(raw))
    if mutation == 'duplicate_artifact': index['artifacts'].append(dict(index['artifacts'][0]))
    raw = preparation.canonical(index)
    (root / 'index.json').write_bytes(raw)
    with pytest.raises(ValueError):
        execution.verify_execution_output(root, preparation.sha(raw))


def test_journal_failure_during_cleanup_does_not_skip_close(bundle, tmp_path, monkeypatch):
    backend = FakeBackend()
    original = execution._Journal.append
    def fail_before_close(self, event):
        if event['phase'] == 'before_close':
            raise OSError('test disk failure')
        return original(self, event)
    monkeypatch.setattr(execution._Journal, 'append', fail_before_close)
    with pytest.raises(RuntimeError, match='journal'):
        run(bundle, tmp_path, backend)
    assert backend.closed
    assert not (tmp_path / 'run/index.json').exists()


def test_backend_receives_public_request_without_task_or_gold(bundle, tmp_path):
    backend = FakeBackend()
    result = run(bundle, tmp_path, backend)
    for request in backend.requests:
        payload = json.loads(request.visible_prompt.split('\n', 1)[1])
        assert set(payload) == {'query', 'retrieved_entries'}
        assert set(payload['query']) == {'text', 'answer_schema'}
        assert not hasattr(request, 'task') and not hasattr(request, 'gold')
        assert not hasattr(request, 'target_object_keys') and not hasattr(request, 'expected_answer')
    assert result['status'] == 'TEST_ONLY_COMPLETE'


def test_direct_cli_only_reports_blocked_production_gate_without_pythonpath(tmp_path):
    env = {k:v for k,v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/vnext_run_cross_domain_answer_replay.py'),
                             '--output-root', str(tmp_path / 'gate')],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 2, result.stderr
    summary = json.loads(result.stdout)
    assert summary['status'] == 'BLOCKED'
    assert summary['answer_metrics'] is None
    assert summary['backend_lifecycle']['constructor_attempts'] == 0
