from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import vnext_prepare_cross_domain_answer_replay as replay

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/vnext/fixtures/cross_domain_v13'
BEA = ROOT / 'data/vnext/family_h_cross_domain_bea_gdp/v1'
NOAA = ROOT / 'data/vnext/family_h_cross_domain_noaa/v1'
CONFIG = ROOT / 'configs/vnext/post_core/cross_domain_answer_replay_v1.json'


def inputs():
    return replay.load_inputs(EVIDENCE, BEA, NOAA, CONFIG)


def test_real_inputs_prepare_four_repetition_bound_prompts_without_execution(tmp_path):
    bundle = inputs()
    plans = replay.build_plans(bundle)
    assert [(p['release_name'], p['repetition_index']) for p in plans] == [
        ('bea', 0), ('bea', 1), ('noaa', 0), ('noaa', 1)]
    assert len({p['request_id'] for p in plans}) == 4
    for plan in plans:
        prompt = replay.render_plan(bundle, plan)
        assert hashlib.sha256(prompt.encode()).hexdigest() == plan['visible_prompt_sha256']
        payload = json.loads(prompt.split('\n', 1)[1])
        assert payload['query']['answer_schema'] == 'number'
        assert len(payload['retrieved_entries']) == 1
        assert set(payload) == {'query', 'retrieved_entries'}
        assert not {'gold', 'target_object_keys', 'selector', 'state_match'} & set(payload['query'])
    output = tmp_path / 'prepared'
    result = replay.publish_preparation(bundle, output)
    assert result['status'] == 'PREPARED_NOT_EXECUTED'
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert manifest['prospective_requests'] == 4
    assert manifest['cluster_n'] == 2 and manifest['cluster_unit'] == 'source_trajectory'
    assert manifest['execution_authorized'] is False
    assert manifest['scientific_release_allowed'] is False
    assert manifest['answer_metrics'] is None
    assert set(manifest['execution_boundary'].values()) == {0}
    assert manifest['model_runtime_status'] == 'NOT_QUALIFIED_FOR_THIS_REPLAY'
    assert manifest['native_answer_status'] == 'UNSUPPORTED'
    assert manifest['canonical_score_record'] is False
    for path in output.iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'Use only the retrieved memory entries' not in text
        assert '"raw_output"' not in text and '"expected_answer"' not in text
        assert str(ROOT) not in text and '/NAS/' not in text
    assert replay.read_preparation(output, result['index_sha256'], bundle)['status'] == 'PREPARED_NOT_EXECUTED'
    with pytest.raises(FileExistsError):
        replay.publish_preparation(bundle, output)


def test_public_projection_ignores_oracle_and_exported_state():
    bundle = inputs()
    task = bundle.tasks['bea']
    row = bundle.rows[0]
    prompt = replay.project_prompt(task.queries[0], row['retrieval'])
    class PublicOnlyQuery:
        query_id = task.queries[0].query_id
        text = task.queries[0].text
        answer_schema = task.queries[0].answer_schema
        @property
        def target_object_keys(self):
            raise AssertionError('hidden target read')
        @property
        def selector(self):
            raise AssertionError('hidden selector read')
    query = PublicOnlyQuery()
    assert replay.project_prompt(query, row['retrieval']) == prompt
    poisoned_row = {**row, 'steps': 'POISON', 'final_state_match': False, 'gold_answer': 'POISON'}
    assert replay.project_prompt(query, poisoned_row['retrieval']) == prompt
    visible = json.loads(prompt.split('\n', 1)[1])
    assert visible['retrieved_entries'][0]['value_candidate'] == row['retrieval']['entries'][0]['value']
    assert visible['retrieved_entries'][0]['source_event_ids'] == row['retrieval']['entries'][0]['source_event_ids']


def test_projection_preserves_provider_order_and_content_without_gold_fallback():
    bundle = inputs()
    query = bundle.tasks['bea'].queries[0]
    raw = deepcopy(bundle.rows[0]['retrieval'])
    one = raw['entries'][0]
    two = {**deepcopy(one), 'entry_id': 'distractor', 'content': '2.3', 'value': 2.3,
           'rank': 2, 'score': -0.9}
    raw['entries'].append(two)
    raw['entries_sha256'] = replay.digest(raw['entries'])
    prompt = replay.project_prompt(query, raw)
    assert [e['content'] for e in json.loads(prompt.split('\n', 1)[1])['retrieved_entries']] == ['2.4', '2.3']
    raw['entries'].reverse()
    raw['entries_sha256'] = replay.digest(raw['entries'])
    assert replay.project_prompt(query, raw) != prompt
    raw['entries'] = []
    raw['entries_sha256'] = replay.digest([])
    assert json.loads(replay.project_prompt(query, raw).split('\n', 1)[1])['retrieved_entries'] == []


@pytest.mark.parametrize('key', ['gold', 'selector', 'expected_answer', 'raw_prompt', 'api_key'])
def test_projection_rejects_hidden_or_unknown_provider_metadata(key):
    bundle = inputs()
    raw = deepcopy(bundle.rows[0]['retrieval'])
    raw['entries'][0]['version_metadata'][key] = 'forbidden'
    raw['entries_sha256'] = replay.digest(raw['entries'])
    with pytest.raises(ValueError, match='metadata'):
        replay.project_prompt(bundle.tasks['bea'].queries[0], raw)


@pytest.mark.parametrize('raw,expected', [
    ('{"disposition":"answered","answer":2.4}', True),
    ('{"disposition":"answered","answer":2.3}', False),
    ('{"disposition":"abstained"}', False),
    ('2.4', False), ('{"disposition":"answered","answer":"2.4"}', False),
    ('{"disposition":"answered","answer":true}', False),
    ('{"disposition":"answered","answer":NaN}', False),
    ('{"disposition":"answered","answer":1e999}', False),
    ('```json\n{"disposition":"answered","answer":2.4}\n```', False),
    ('{"disposition":"answered","answer":2.4,"answer":2.3}', False),
    ('{"disposition":"answered","answer":2.4,"unit":"percent"}', False),
])
def test_number_parser_and_typed_exact_controls(raw, expected):
    task = inputs().tasks['bea']
    score = replay.score_readout(task, status='COMPLETED', raw_output=raw)
    assert score['typed_exact_match'] is expected
    assert 'raw_output' not in score
    assert score['canonical_score_record'] is False


def test_number_equality_preserves_int_float_boundary():
    task = inputs().tasks['noaa']
    assert replay.score_readout(task, status='COMPLETED', raw_output='{"disposition":"answered","answer":75}')['typed_exact_match'] is True
    assert replay.score_readout(task, status='COMPLETED', raw_output='{"disposition":"answered","answer":75.0}')['typed_exact_match'] is False


@pytest.mark.parametrize('status', ['NOT_RUN', 'TECHNICAL_FAILURE', 'UNSUPPORTED'])
def test_nonexecuted_results_are_null_not_accuracy_zero(status):
    score = replay.score_readout(inputs().tasks['noaa'], status=status, raw_output=None)
    assert score['typed_exact_match'] is None and score['format_valid'] is None
    assert score['included_in_denominator'] is False
    with pytest.raises(ValueError):
        replay.score_readout(inputs().tasks['noaa'], status=status, raw_output='{"answer":75}')


def test_bogus_stored_success_flag_cannot_pass_recomputed_state_admission():
    bundle = inputs()
    rows = deepcopy(bundle.rows)
    rows[0]['steps'][0]['entries'][0]['value'] = 999
    rows[0]['steps'][0]['entries_sha256'] = replay.digest(rows[0]['steps'][0]['entries'])
    with pytest.raises(ValueError, match='state'):
        replay.recompute_evidence(bundle.tasks, rows, bundle.summary)


def test_bogus_retrieval_success_flag_cannot_pass_recomputed_admission():
    bundle = inputs()
    rows = deepcopy(bundle.rows)
    rows[0]['retrieval']['entries'][0]['value'] = 999
    rows[0]['retrieval']['entries_sha256'] = replay.digest(rows[0]['retrieval']['entries'])
    with pytest.raises(ValueError, match='retrieval'):
        replay.recompute_evidence(bundle.tasks, rows, bundle.summary)


def test_strict_root_authentication_and_no_replace(tmp_path):
    copied = tmp_path / 'evidence'
    shutil.copytree(EVIDENCE, copied)
    (copied / 'results/rows.jsonl').write_bytes((copied / 'results/rows.jsonl').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash'):
        replay.load_inputs(copied, BEA, NOAA, CONFIG)
    bundle = inputs()
    with pytest.raises(ValueError, match='protected|overlap|indexed'):
        replay.publish_preparation(bundle, BEA / 'nested')
    assert not (BEA / 'nested').exists()


def test_input_drift_between_preparation_and_publication_is_rejected(tmp_path):
    copied = tmp_path / 'evidence'
    shutil.copytree(EVIDENCE, copied)
    bundle = replay.load_inputs(copied, BEA, NOAA, CONFIG)
    (copied / 'supervisor.exitcode').write_bytes(b'2\n')
    with pytest.raises(ValueError, match='changed'):
        replay.publish_preparation(bundle, tmp_path / 'prepared')


def test_projection_rejects_unknown_trace_metadata():
    bundle = inputs()
    raw = deepcopy(bundle.rows[0]['retrieval'])
    raw['version_metadata']['raw_prompt'] = 'must not be admitted'
    with pytest.raises(ValueError, match='metadata'):
        replay.project_prompt(bundle.tasks['bea'].queries[0], raw)


def test_recomputed_admission_rejects_false_sequence_types():
    bundle = inputs()
    rows = deepcopy(bundle.rows)
    rows[0]['steps'][0]['sequence_index'] = False
    with pytest.raises(ValueError, match='state event'):
        replay.recompute_evidence(bundle.tasks, rows, bundle.summary)


def test_configuration_cannot_silently_authorize_execution(tmp_path):
    config = json.loads(CONFIG.read_bytes())
    config['prospective_model_intent']['execution_authorized'] = True
    path = tmp_path / 'policy.json'
    path.write_bytes(replay.canonical(config))
    with pytest.raises(ValueError, match='policy hash'):
        replay.load_inputs(EVIDENCE, BEA, NOAA, path)


def test_relocated_package_retains_exact_bytes_and_no_paths(tmp_path):
    original = tmp_path / 'one'
    second = tmp_path / 'two'
    original.mkdir()
    second.mkdir()
    for base in (original, second):
        shutil.copytree(EVIDENCE, base / 'evidence')
        shutil.copytree(BEA, base / 'bea')
        shutil.copytree(NOAA, base / 'noaa')
    first_bundle = replay.load_inputs(original / 'evidence', original / 'bea', original / 'noaa', CONFIG)
    second_bundle = replay.load_inputs(second / 'evidence', second / 'bea', second / 'noaa', CONFIG)
    a = replay.publish_preparation(first_bundle, tmp_path / 'published-one')
    b = replay.publish_preparation(second_bundle, tmp_path / 'published-two')
    assert a['index_sha256'] == b['index_sha256']


def test_publication_rechecks_inputs_at_commit_boundary(tmp_path, monkeypatch):
    copied = tmp_path / 'evidence'
    shutil.copytree(EVIDENCE, copied)
    bundle = replay.load_inputs(copied, BEA, NOAA, CONFIG)
    publish = replay.publish_files_atomically
    def mutate_at_boundary(payloads, **kwargs):
        (copied / 'supervisor.exitcode').write_bytes(b'2\n')
        return publish(payloads, **kwargs)
    monkeypatch.setattr(replay, 'publish_files_atomically', mutate_at_boundary)
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='changed'):
        replay.publish_preparation(bundle, output)
    assert not (output / 'index.json').exists()


def test_input_with_symlink_parent_is_rejected(tmp_path):
    link = tmp_path / 'redirect'
    try:
        link.symlink_to(EVIDENCE, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation unavailable')
    with pytest.raises(ValueError, match='symlink|reparse'):
        replay.load_inputs(link, BEA, NOAA, CONFIG)


def test_output_reopen_authenticates_all_members(tmp_path):
    bundle = inputs()
    output = tmp_path / 'prepared'
    result = replay.publish_preparation(bundle, output)
    (output / 'plans.jsonl').write_bytes((output / 'plans.jsonl').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash|bytes'):
        replay.read_preparation(output, result['index_sha256'], bundle)


def test_offline_preparation_never_loads_models_or_uses_network(monkeypatch):
    import socket
    from mub.vnext.runtime.answer_model_v3 import OfflinePromptedAnswerModelV3
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected execution')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(OfflinePromptedAnswerModelV3, 'load', forbidden)
    assert len(replay.build_plans(inputs())) == 4


def test_in_memory_future_value_cannot_masquerade_as_authenticated_prompt():
    bundle = inputs()
    plans = replay.build_plans(bundle)
    raw = bundle.rows[0]['retrieval']
    raw['entries'][0]['value'] = 999
    raw['entries'][0]['content'] = '999'
    raw['entries_sha256'] = replay.digest(raw['entries'])
    with pytest.raises(ValueError, match='in-memory'):
        replay.build_plans(bundle)
    with pytest.raises(ValueError, match='in-memory'):
        replay.render_plan(bundle, plans[0])


@pytest.mark.parametrize('verification_hash', ['', 'INVALID', 'A' * 64])
def test_invalid_verification_mode_never_publishes(tmp_path, verification_hash):
    output = tmp_path / 'must-not-exist'
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/vnext_prepare_cross_domain_answer_replay.py'),
        '--output-root', str(output), '--verify-index-sha256', verification_hash],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert not output.exists()


def test_direct_cli_help_does_not_require_pythonpath(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/vnext_prepare_cross_domain_answer_replay.py'), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
