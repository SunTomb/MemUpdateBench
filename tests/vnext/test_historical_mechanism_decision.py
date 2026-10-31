"""Publication contract tests use synthetic receipts, never an independent live audit."""
import json

import pytest

from scripts import vnext_finalize_historical_mechanism as finalizer
from scripts import vnext_run_historical_mechanism as runner
from tests.vnext.test_historical_mechanism_analysis import fixture, cases
from tests.vnext.test_historical_mechanism_runner import PACKET


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(finalizer.encoded(value))
    return runner.sha(path)


def publication_fixture(tmp_path):
    report, artifacts, observations = fixture(tmp_path)
    root = tmp_path/'run'
    index = {}
    for i, (kind, value) in enumerate(artifacts, 1):
        name = f'{i:05d}_{kind}.json'
        index[name] = write(root/'private'/name, value)
    index_pin = write(root/'private/index.json', index)
    obs_pin = write(root/'compact/observations.json', observations)
    transport_pin = write(root/'compact/transport.json', [])
    source_root = tmp_path/'source'
    files = {}
    for name in runner.REQUIRED_SOURCE_FILES:
        path = source_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic offline archive bytes\n')
        files[name] = runner.sha(path)
    source_manifest = tmp_path/'source-manifest.json'
    source_pin = write(source_manifest, {'schema': runner.SOURCE_SCHEMA, 'runtime_revision': 'a'*40, 'files': files})
    report.update(runtime_revision='a'*40, runtime_source_manifest_sha256=source_pin,
        runtime_source_map_sha256=runner.digest(files), packet_index_sha256=runner.PACKET_PIN,
        packet_source_revision=runner.PACKET_SOURCE, private_artifacts={'count': len(index), 'index_sha256': index_pin},
        observations_sha256=obs_pin, transport_sha256=transport_pin)
    terminal_pin = write(root/'terminal.json', report)
    bindings = dict(zip(finalizer.BINDING_KEYS, (terminal_pin, runner.PACKET_PIN, source_pin, index_pin, obs_pin, transport_pin)))
    audit = {'schema': finalizer.AUDIT_SCHEMA, 'status': 'PASS', 'bindings': bindings,
        'checks': dict.fromkeys(finalizer.AUDIT_CHECKS, True), 'cases': finalizer.audit_projection(report)}
    audit_path = tmp_path/'audit.json'
    audit_pin = write(audit_path, audit)
    parent = tmp_path/'results/vnext'
    parent.mkdir(parents=True)
    kwargs = dict(run_root=root, terminal_sha256=terminal_pin, packet_root=PACKET,
        source_manifest=source_manifest, source_manifest_sha256=source_pin, runtime_revision='a'*40,
        source_root=source_root, audit_receipt=audit_path, audit_sha256=audit_pin,
        output_root=parent/'version_transfer_historical_mechanism_20261030_v1', results_parent=parent)
    return kwargs, report, audit


def test_synthetic_hashbound_publication_redacts_and_no_replace(tmp_path):
    kwargs, report, audit = publication_fixture(tmp_path)
    result = finalizer.finalize(**kwargs)
    assert result['label'] == 'NO_DISTINGUISHING_SIGNAL'
    output = kwargs['output_root']
    decision = (output/'decision.json').read_bytes()
    index = json.loads((output/'index.json').read_bytes())
    assert index['artifacts']['decision.json'] == finalizer.sha_bytes(decision)
    assert result['bindings']['audit_sha256'] == kwargs['audit_sha256']
    assert result['resources']['llm']['attempts'] > 0
    for forbidden in ('target-alpha', 'target-beta', 'target-gamma', 'Target-00', 'rendered_prompt',
                      'raw_output', 'parsed_answer', '/NAS/', str(tmp_path), 'edge-target'):
        assert forbidden not in decision.decode()
    assert set(p.name for p in output.iterdir()) == {'decision.json', 'index.json'}
    with pytest.raises(ValueError, match='OUTPUT_ALREADY_EXISTS'):
        finalizer.finalize(**kwargs)


@pytest.mark.parametrize('failure', ['missing_audit', 'audit_pin', 'audit_binding', 'audit_check',
    'audit_coverage', 'terminal_pin', 'source_pin', 'source_bytes', 'packet', 'private_bytes', 'membership', 'projection', 'index_membership'])
def test_bad_binding_or_missing_audit_refuses_publication(tmp_path, failure):
    kwargs, report, audit = publication_fixture(tmp_path)
    root = kwargs['run_root']
    if failure == 'missing_audit':
        kwargs['audit_receipt'].unlink()
    elif failure == 'audit_pin':
        kwargs['audit_sha256'] = '0'*64
    elif failure in ('audit_binding', 'audit_check', 'audit_coverage'):
        if failure == 'audit_binding':
            audit['bindings']['terminal_sha256'] = '0'*64
        elif failure == 'audit_check':
            audit['checks']['cleanup_reopened'] = False
        else:
            audit['cases'].pop()
        kwargs['audit_sha256'] = write(kwargs['audit_receipt'], audit)
    elif failure == 'terminal_pin':
        kwargs['terminal_sha256'] = '0'*64
    elif failure == 'source_pin':
        kwargs['source_manifest_sha256'] = '0'*64
    elif failure == 'source_bytes':
        (kwargs['source_root']/runner.REQUIRED_SOURCE_FILES[0]).write_bytes(b'changed')
    elif failure == 'packet':
        kwargs['packet_root'] = tmp_path/'no-packet'
    elif failure == 'private_bytes':
        next(p for p in (root/'private').iterdir() if p.name != 'index.json').write_bytes(b'changed')
    elif failure == 'index_membership':
        write(root/'private/99999_state.json', {'unindexed': True})
    else:
        if failure == 'membership':
            report['cases'][0]['task_hash'] = '0'*64
        else:
            report['cases'][0]['readout_projection']['projected_task_sha256'] = '0'*64
        kwargs['terminal_sha256'] = write(root/'terminal.json', report)
        audit['bindings']['terminal_sha256'] = kwargs['terminal_sha256']
        audit['cases'] = finalizer.audit_projection(report)
        kwargs['audit_sha256'] = write(kwargs['audit_receipt'], audit)
    with pytest.raises((ValueError, OSError)):
        finalizer.finalize(**kwargs)
    assert not kwargs['output_root'].exists()


def test_well_audited_partial_remains_blocked_all_rows_present(tmp_path):
    kwargs, report, audit = publication_fixture(tmp_path)
    report['status'] = 'BLOCKED'
    report['cases'][1]['status'] = 'BLOCKED'
    report['cases'][2]['status'] = 'NOT_RUN'
    report['accounting'] = None
    kwargs['terminal_sha256'] = write(kwargs['run_root']/'terminal.json', report)
    audit['bindings']['terminal_sha256'] = kwargs['terminal_sha256']
    audit['cases'] = finalizer.audit_projection(report)
    kwargs['audit_sha256'] = write(kwargs['audit_receipt'], audit)
    result = finalizer.finalize(**kwargs)
    assert result['label'] == 'BLOCKED'
    assert [r['status'] for r in result['cases']] == ['OBSERVED', 'BLOCKED', 'NOT_RUN']
    assert result['resources']['llm']['attempts'] is None
    assert result['scientific_release_allowed'] is False


@pytest.mark.parametrize('name', ['core', 'pilot', 'version_transfer_historical_mechanism_preparation_20261030_v2',
    'version_transfer_historical_mechanism_20261030_v0'])
def test_unrelated_or_frozen_output_rejected(tmp_path, name):
    kwargs, _, _ = publication_fixture(tmp_path)
    kwargs['output_root'] = kwargs['results_parent']/name
    with pytest.raises(ValueError, match='OUTPUT_OUTSIDE_PROTOCOL'):
        finalizer.finalize(**kwargs)


def test_numeric_resource_allowlist_drops_provider_metadata_and_exception_text():
    report = {'elapsed_seconds': 'private-text', 'output_bytes': 1,
              'accounting': {'llm': {'charged': {'attempts': 2, 'input_tokens': 'secret', 'private': 'secret'}},
                             'arbitrary': 'password'}}
    resources = finalizer.resource_projection(report)
    assert resources['elapsed_seconds'] is None
    assert resources['llm']['input_tokens'] is None
    assert 'secret' not in json.dumps(resources)
    assert 'password' not in json.dumps(resources)


def test_resource_totals_keep_role_finish_and_retry_unknowns():
    report = {'accounting': {'llm': {'roles': {'graphiti': 7, 'readout': 2, 'oracle_readout': 1}},
                            'openai_transport_retries': 0}}
    observations = [{'finish_reason': 'stop'}, {'finish_reason': 'length'}, {'finish_reason': None}]
    resources = finalizer.resource_projection(report, observations)
    assert resources['roles'] == {'graphiti': 7, 'readout': 2, 'oracle_readout': 1}
    assert resources['finish_reasons'] == {'stop': 1, 'length': 1, 'other': 0, 'unknown': 1}
    assert resources['openai_transport_retries'] == 0
    assert resources['native_retry_attempts'] is None


def test_unavailable_compact_observations_have_null_finish_totals(tmp_path):
    kwargs, report, audit = publication_fixture(tmp_path)
    report['status'] = 'BLOCKED'
    report['accounting'] = None
    report.pop('observations_sha256')
    report.pop('transport_sha256')
    for name in ('observations.json', 'transport.json'):
        (kwargs['run_root']/'compact'/name).unlink()
    for row in report['cases']:
        row['status'] = 'NOT_RUN'
    kwargs['terminal_sha256'] = write(kwargs['run_root']/'terminal.json', report)
    audit['bindings'].update(terminal_sha256=kwargs['terminal_sha256'], observations_sha256=None, transport_sha256=None)
    audit['cases'] = finalizer.audit_projection(report)
    kwargs['audit_sha256'] = write(kwargs['audit_receipt'], audit)
    result = finalizer.finalize(**kwargs)
    assert result['label'] == 'BLOCKED'
    assert result['resources']['finish_reasons'] is None
