from types import SimpleNamespace

import pytest

from scripts import vnext_run_version_transfer_decision as runner


def test_database_configuration_is_loopback_owned_and_does_not_patch_vendor(tmp_path):
    neo = tmp_path / 'vendor'
    (neo / 'conf').mkdir(parents=True)
    (neo / 'conf/server-logs.xml').write_bytes(b'<Configuration/>')
    run = tmp_path / 'owned'
    run.mkdir()
    conf = runner.database_config(run, neo)
    text = (conf / 'neo4j.conf').read_text()
    assert 'server.bolt.listen_address=127.0.0.1:17687' in text
    assert 'server.http.enabled=false' in text
    assert 'server.https.enabled=false' in text
    assert 'dbms.usage_report.enabled=false' in text
    assert str(run / 'tmp') in text and str(run / 'database-data') in text
    assert (neo / 'conf/server-logs.xml').read_bytes() == b'<Configuration/>'
    assert not (neo / 'conf/neo4j.conf').exists()


def test_runner_refuses_unowned_output_before_subprocess_or_provider(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['smoke', '--output-root', str(tmp_path),
        '--packet-root', str(tmp_path / 'packet'), '--packet-index-sha256', 'a' * 64,
        '--gpu-uuid', 'GPU-test'])
    def forbidden(*args, **kwargs):
        raise AssertionError('subprocess invoked before output admission')
    monkeypatch.setattr(runner.subprocess, 'check_output', forbidden)
    with pytest.raises(ValueError, match='owned'):
        runner.main()


def test_owned_stop_targets_only_the_supplied_process_group(monkeypatch):
    state = {'running': True}
    sent = []
    def wait(timeout):
        state['running'] = False
        return 0
    process = SimpleNamespace(pid=12345, poll=lambda: None if state['running'] else 0, wait=wait)
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: sent.append((pid, sig)), raising=False)
    assert runner.stop_owned(process) is True
    assert len(sent) == 1 and sent[0][0] == 12345
    assert runner.stop_owned(process) is True
    assert len(sent) == 1


def test_native_client_factory_is_reusable_without_importing_provider_packages():
    import inspect
    assert callable(runner.native_graph_clients)
    signature = inspect.signature(runner.native_graph_clients)
    assert signature.parameters['structured_output_mode'].kind is inspect.Parameter.KEYWORD_ONLY
