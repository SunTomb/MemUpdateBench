from __future__ import annotations

from pathlib import Path

import pytest

from scripts.vnext_run_cross_domain_qdrant_state import (
    build_server_config,
    build_source_manifest,
)


def test_runtime_tree_revalidation_preserves_manifest_order(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import verify_runtime_snapshot, canonical_json_bytes
    import hashlib
    root = tmp_path / 'runtime'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg/a.py').write_bytes(b'one')
    (root / 'pkg-1.dist-info').mkdir()
    (root / 'pkg-1.dist-info/RECORD').write_bytes(b'two')
    rows = [{'path': 'pkg/a.py', 'bytes': 3, 'sha256': hashlib.sha256(b'one').hexdigest()},
            {'path': 'pkg-1.dist-info/RECORD', 'bytes': 3, 'sha256': hashlib.sha256(b'two').hexdigest()}]
    manifest = {'file_count': 2, 'bytes': 6, 'files': rows,
                'tree_sha256': hashlib.sha256(canonical_json_bytes(rows)).hexdigest(), 'scope': 'test'}
    assert verify_runtime_snapshot(root, manifest) == manifest


def test_local_runtime_snapshot_is_rehashed_before_and_after(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import verify_runtime_snapshot, canonical_json_bytes
    import hashlib
    root = tmp_path / 'runtime'
    root.mkdir()
    path = root / 'module.py'
    path.write_bytes(b'unchanged')
    rows = [{'path': 'module.py', 'bytes': 9, 'sha256': hashlib.sha256(b'unchanged').hexdigest()}]
    manifest = {'file_count': 1, 'files': rows, 'tree_sha256': hashlib.sha256(canonical_json_bytes(rows)).hexdigest()}
    first = verify_runtime_snapshot(root, manifest)
    assert first == verify_runtime_snapshot(root, manifest)
    path.write_bytes(b'modified!')
    with pytest.raises(ValueError, match='runtime file'):
        verify_runtime_snapshot(root, manifest)
    path.write_bytes(b'unchanged')
    (root / 'injected.py').write_bytes(b'added')
    with pytest.raises(ValueError, match='membership'):
        verify_runtime_snapshot(root, manifest)


def test_runtime_manifest_uses_local_runtime_copy_without_distribution_rescan(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import build_local_runtime_manifest
    executable = tmp_path / 'python'
    executable.write_bytes(b'python')
    venv = tmp_path / 'pyvenv.cfg'
    venv.write_bytes(b'version=3.10')
    report = build_local_runtime_manifest(
        {'file_count': 2, 'bytes': 10, 'tree_sha256': 'a' * 64},
        executable, venv, {'qdrant-client': '1.19.0', 'pydantic': '2.13.5', 'PyYAML': '6.0.3'},
    )
    assert report['runtime_copy']['file_count'] == 2
    assert report['versions']['qdrant-client'] == '1.19.0'
    assert report['scope'] == 'local_record_verified_runtime_copy_plus_minimal_version_identity'


def test_anchored_reader_bounds_members_and_closes(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import AnchoredRuntimeReader
    root = tmp_path / 'root'
    (root / 'package').mkdir(parents=True)
    (root / 'package/a.py').write_bytes(b'abc')
    with AnchoredRuntimeReader(root) as reader:
        assert reader.read('package/a.py', limit=3) == b'abc'
        assert reader.read('package/a.py', limit=3) == b'abc'
        with pytest.raises(ValueError):
            reader.read('../outside', limit=10)
        with pytest.raises(ValueError):
            reader.read('package/a.py', limit=2)
    assert reader.closed is True
    with pytest.raises(ValueError):
        reader.read('package/a.py', limit=3)


def test_runtime_materialization_uses_record_hashes_and_keeps_script_layout(tmp_path):
    import base64
    import hashlib
    from scripts.vnext_run_cross_domain_qdrant_state import materialize_cpu_runtime
    original = tmp_path / 'original'
    site = original / 'lib/python3.10/site-packages'
    metadata = site / 'example-1.dist-info'
    metadata.mkdir(parents=True)
    (original / 'bin').mkdir()
    (site / 'example.py').write_bytes(b'example')
    (original / 'bin/helper').write_bytes(b'helper')
    rows = []
    for path, raw in [('example.py', b'example'), ('../../../bin/helper', b'helper')]:
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        rows.append(f'{path},sha256={digest},{len(raw)}')
    rows.append('example-1.dist-info/RECORD,,')
    (metadata / 'RECORD').write_text('\n'.join(rows), encoding='utf-8')
    out = tmp_path / 'snapshot'
    report = materialize_cpu_runtime(original, out)
    assert report['file_count'] == 3
    assert (out / 'lib/python3.10/site-packages/example.py').read_bytes() == b'example'
    assert (out / 'bin/helper').read_bytes() == b'helper'
    (site / 'example.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='RECORD'):
        materialize_cpu_runtime(original, tmp_path / 'bad')


def test_client_construction_preflight_never_requests_network(monkeypatch):
    from types import SimpleNamespace
    import sys
    from scripts.vnext_run_cross_domain_qdrant_state import preflight_client
    calls = []
    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def close(self):
            calls.append('closed')
    monkeypatch.setitem(sys.modules, 'qdrant_client', SimpleNamespace(QdrantClient=Client))
    assert preflight_client()['provider_calls'] == 0
    assert calls[-1] == 'closed'
    assert calls[0]['check_compatibility'] is False and calls[0]['trust_env'] is False


def test_yaml_dependency_is_materialized_without_mutating_source(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import materialize_yaml_dependency
    site = tmp_path / 'existing-environment'
    for name in ('yaml', '_yaml', 'pyyaml-6.0.3.dist-info'):
        (site / name).mkdir(parents=True)
        (site / name / 'file.py').write_bytes(name.encode())
    before = {p.relative_to(site).as_posix(): p.read_bytes() for p in site.rglob('*') if p.is_file()}
    destination = tmp_path / 'runtime_extra'
    result = materialize_yaml_dependency(site, destination)
    assert result['version'] == '6.0.3' and result['file_count'] == 3
    assert before == {p.relative_to(destination).as_posix(): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    assert before == {p.relative_to(site).as_posix(): p.read_bytes() for p in site.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        materialize_yaml_dependency(site, destination)


def test_source_manifest_is_deterministic_and_hash_bound(tmp_path: Path):
    first = tmp_path / "one.py"
    second = tmp_path / "nested" / "two.py"
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    manifest = build_source_manifest(tmp_path, (first, second))

    assert manifest["schema"] == "memupdatebench.cross-domain.source-manifest.v1"
    assert [row["path"] for row in manifest["files"]] == ["nested/two.py", "one.py"]
    assert manifest["bundle_sha256"]
    assert len(manifest["bundle_sha256"]) == 64
    assert manifest == build_source_manifest(tmp_path, (second, first))


def test_source_manifest_rejects_symlink(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_bytes(b"source")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(ValueError, match="symlink"):
        build_source_manifest(tmp_path, (link,))


def test_server_config_contains_no_secret(tmp_path: Path):
    config = build_server_config(tmp_path)

    assert config["service"] == {"host": "127.0.0.1", "http_port": 16333,
                                  "grpc_port": None, "enable_cors": False, "max_workers": 2}
    assert config["storage"]["storage_path"] == str(tmp_path / "storage")
    assert "api" not in " ".join(str(value).casefold() for value in config.values())


def test_exclusive_text_write_encodes_utf8_and_cannot_replace(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import _write_text_exclusive
    path = tmp_path / "server.yaml"
    _write_text_exclusive(path, "配置\n")
    assert path.read_bytes() == "配置\n".encode()
    with pytest.raises(FileExistsError):
        _write_text_exclusive(path, "replacement")


def test_bound_source_rejects_changed_or_unlisted_python(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import verify_source_manifest, canonical_json_bytes
    import hashlib
    path = tmp_path / "worker.py"
    path.write_bytes(b"value = 1\n")
    manifest = tmp_path / "source_manifest.json"
    raw = canonical_json_bytes(build_source_manifest(tmp_path, (path,)))
    manifest.write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    assert verify_source_manifest(tmp_path, manifest, pin)["bundle_sha256"]
    (tmp_path / "unlisted.py").write_bytes(b"extra = 1\n")
    with pytest.raises(ValueError, match="membership"):
        verify_source_manifest(tmp_path, manifest, pin)
    (tmp_path / "unlisted.py").unlink()
    path.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="hash"):
        verify_source_manifest(tmp_path, manifest, pin)


def test_owned_directory_cleanup_refuses_changed_marker(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import claim_owned_directory, remove_owned_directory
    path = tmp_path / "owned"
    identity = claim_owned_directory(path)
    (path / "ownership.marker").write_bytes(b"foreign-marker")
    with pytest.raises(ValueError, match="ownership"):
        remove_owned_directory(path, identity)
    assert path.exists()


def test_owned_directory_cleanup_and_no_replace(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import claim_owned_directory, remove_owned_directory
    path = tmp_path / "owned"
    identity = claim_owned_directory(path)
    with pytest.raises(FileExistsError):
        claim_owned_directory(path)
    (path / "mine.txt").write_bytes(b"mine")
    remove_owned_directory(path, identity)
    assert not path.exists()


def test_success_requires_postrun_binding_and_cleanup():
    from scripts.vnext_run_cross_domain_qdrant_state import finalize_result
    row = {"status": "COMPLETE", "rows": [], "state_step_matches": 20,
           "final_state_matches": 4, "retrieval_typed_object_matches": 4,
           "repetitions_equal": True}
    cleanup = {"processes_stopped": True, "directory_removed": True, "port_closed": True}
    assert finalize_result(row, cleanup, bindings_unchanged=True, error=None)["status"] == "COMPLETE"
    for field in cleanup:
        result = finalize_result(row, {**cleanup, field: False}, bindings_unchanged=True, error=None)
        assert result["status"] == "BLOCKED"
        assert result["state_step_matches"] is None
        assert result["scientific_evidence"] is False
    result = finalize_result(row, cleanup, bindings_unchanged=False, error=None)
    assert result["status"] == "BLOCKED" and result["state_step_matches"] is None


def test_output_boundary_excludes_inputs_and_foreign_remote_roots(tmp_path):
    from scripts.vnext_run_cross_domain_qdrant_state import validate_run_paths
    from pathlib import PurePosixPath
    run = PurePosixPath("/NAS/yesh/MemUpdateBench/external/cross_domain_state_20260930_v1")
    temp = PurePosixPath("/tmp/mub-qdrant-cross-domain-20260930-v1")
    validate_run_paths(run, temp)
    for bad in ("/tmp/foreign", "/tmp/mub-qdrant-cross-domain-20260930-v1/../other"):
        with pytest.raises(ValueError):
            validate_run_paths(run, PurePosixPath(bad))
    with pytest.raises(ValueError):
        validate_run_paths(PurePosixPath("/NAS/yesh/MemUpdateBench/data/vnext/core/v3"), temp)


@pytest.mark.parametrize('failure', [None, 'preflight', 'readiness', 'worker', 'source_drift', 'unowned_listener', 'log_hash', 'python38'])
def test_supervisor_publishes_only_after_cleanup_and_keeps_failure_null(tmp_path, monkeypatch, failure):
    import hashlib
    import json
    import shutil
    from types import SimpleNamespace
    from scripts import vnext_run_cross_domain_qdrant_state as runner
    root = tmp_path / 'run'
    source = root / 'source'
    (source / 'mub/vnext/io').mkdir(parents=True)
    original_atomic = Path(runner.__file__).resolve().parents[1] / 'mub/vnext/io/atomic.py'
    shutil.copyfile(original_atomic, source / 'mub/vnext/io/atomic.py')
    (source / 'worker.py').write_bytes(b'pass\n')
    if failure == 'python38':
        (source / 'mub/vnext/io/atomic.py').write_bytes(b'raise TypeError("python38 generic alias unsupported")\n')
        monkeypatch.setattr(runner.sys, 'version_info', (3, 8, 10))
    manifest = runner.build_source_manifest(source, tuple(source.rglob('*.py')))
    manifest_raw = runner.canonical_json_bytes(manifest)
    (source / 'source_manifest.json').write_bytes(manifest_raw)
    pin = hashlib.sha256(manifest_raw).hexdigest()
    pins = {}
    for name in ('bea', 'noaa'):
        folder = root / 'input_release' / name
        folder.mkdir(parents=True)
        raw = json.dumps({'name': name}).encode()
        (folder / 'release_index.json').write_bytes(raw)
        pins[name] = hashlib.sha256(raw).hexdigest()
    binary = root / 'binary'
    binary.write_bytes(b'pinned-binary')
    envfile = root / 'env.sh'
    envfile.write_bytes(b'# pinned environment')
    temp = tmp_path / 'owned-temp'
    monkeypatch.setattr(runner, 'PROJECT_ROOT', source)
    monkeypatch.setattr(runner, 'RELEASE_PINS', pins)
    monkeypatch.setattr(runner, 'BINARY', binary)
    monkeypatch.setattr(runner, 'ENVFILE', envfile)
    monkeypatch.setattr(runner, 'QDRANT_BINARY_SHA256', runner._sha_file(binary))
    monkeypatch.setattr(runner, 'QDRANT_ENVFILE_SHA256', runner._sha_file(envfile))
    monkeypatch.setattr(runner.sys, 'platform', 'linux')
    monkeypatch.setattr(runner.os, 'getuid', lambda: temp.stat().st_uid, raising=False)
    monkeypatch.setattr(runner.stat, 'S_IMODE', lambda _: 0o700)
    monkeypatch.setattr(runner, 'validate_run_paths', lambda *args: None)
    monkeypatch.setattr(runner, '_filesystem_info', lambda _: {'fstype': 'ext4', 'block_device': True})
    monkeypatch.setattr(runner, 'materialize_yaml_dependency', lambda *args: {'name': 'fake-test-only', 'files': []})
    monkeypatch.setattr(runner, 'materialize_cpu_runtime', lambda *args, **kwargs: {'scope': 'fake-test-only'})
    monkeypatch.setattr(runner, '_listener_owned', lambda _: failure != 'unowned_listener')
    monkeypatch.setattr(runner, '_port_closed', lambda: True, raising=False)
    monkeypatch.setattr(runner, '_group_members', lambda _: [], raising=False)
    if failure == 'log_hash':
        original_sha = runner._sha_file
        def fail_log_hash(path):
            if Path(path).suffix == '.log':
                raise OSError('diagnostic collection failed')
            return original_sha(path)
        monkeypatch.setattr(runner, '_sha_file', fail_log_hash)
    monkeypatch.setattr(runner, 'signal', SimpleNamespace(SIGALRM=14, SIGTERM=15,
        signal=lambda *args: None, alarm=lambda *args: None))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=b'PATH=/bin\0'))
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    phases = []

    class Process:
        def __init__(self, argv, **kwargs):
            self.pid = 123
            self.returncode = None
            if '--config-path' in argv:
                self.phase = 'server'
            else:
                self.phase = argv[argv.index('--mode') + 1]
                self.returncode = 2 if self.phase == failure else 0
                if self.phase == 'preflight' and not self.returncode:
                    (temp / 'preflight.json').write_bytes(runner.canonical_json_bytes({
                        'cells': 2, 'events': 10, 'runtime_sha256': '1'*64,
                        'runtime': {'python_version': 'test', 'distribution_count': 1}}))
                elif self.phase == 'worker' and not self.returncode:
                    (temp / 'observed_rows.json').write_bytes(runner.canonical_json_bytes({
                        'status': 'COMPLETE', 'rows': [], 'state_step_matches': 20,
                        'final_state_matches': 4, 'retrieval_typed_object_matches': 4}))
                    (temp / 'worker_complete.json').write_bytes(runner.canonical_json_bytes({
                        'runtime_revalidated': True, 'source_revalidated': True, 'release_revalidated': True}))
                    (temp / 'http_calls.json').write_bytes(b'[]')
                    if failure == 'source_drift':
                        (source / 'worker.py').write_bytes(b'changed\n')
                        (source / 'mub/vnext/io/atomic.py').write_bytes(b'raise RuntimeError("untrusted changed source executed")\n')
            phases.append(self)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner, '_stop_process', lambda p: setattr(p, 'returncode', -15) if p else None)

    def http(path, key):
        assert failure != 'unowned_listener', 'authenticated request reached an unverified listener'
        if failure == 'readiness':
            raise TimeoutError()
        return ((200, json.dumps({'version': runner.QDRANT_VERSION, 'commit': runner.QDRANT_RUNTIME_REVISION}).encode())
                if key else (403, b'forbidden'))

    monkeypatch.setattr(runner, '_http_get', http)

    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
        def settimeout(self, *args): pass
        def connect_ex(self, *args): return 1

    monkeypatch.setattr(runner.socket, 'socket', Socket)
    result = runner.run_live(run_root=root, temp_root=temp, source_sha256=pin)
    summary = json.loads((root / 'results/summary.json').read_bytes())
    assert not temp.exists()
    assert all(p.poll() is not None for p in phases)
    assert all(summary['cleanup'][k] for k in ('processes_stopped', 'directory_removed', 'port_closed'))
    succeeded = failure in (None, 'python38')
    assert result['status'] == ('COMPLETE' if succeeded else 'BLOCKED')
    if not succeeded:
        assert summary['state_step_matches'] is None
        assert summary['scientific_evidence'] is False
        if failure == 'worker':
            assert summary['execution_boundary']['worker_http_calls'] is None
        if failure == 'unowned_listener':
            assert summary['execution_boundary']['setup_http_calls'] == 0
    else:
        assert summary['state_step_matches'] == 20
    with pytest.raises(FileExistsError):
        runner.run_live(run_root=root, temp_root=temp, source_sha256=pin)


def test_cleanup_stops_owned_group_after_leader_exit(monkeypatch):
    from types import SimpleNamespace
    from scripts import vnext_run_cross_domain_qdrant_state as runner
    members = [{'pid': 223, 'session': 123, 'uid': 7, 'start': 456}]
    sent = []
    monkeypatch.setattr(runner, '_group_members', lambda pid: list(members), raising=False)
    monkeypatch.setattr(runner.os, 'getuid', lambda: 7, raising=False)
    monkeypatch.setattr(runner, 'signal', SimpleNamespace(SIGTERM=15, SIGKILL=9))
    def killpg(pid, signum):
        sent.append(pid)
        members.clear()
    monkeypatch.setattr(runner.os, 'killpg', killpg, raising=False)
    process = SimpleNamespace(pid=123, poll=lambda: 0)
    runner._stop_process(process)
    assert sent == [123] and not members
