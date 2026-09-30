"""Bounded, source-pinned NOAA/BEA Qdrant canary with post-cleanup publication."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_BINARY_SHA256 = "abfe97e1d0225111dec2f048790428f151846c8a049eefc85328b6c9eccaf419"
QDRANT_ENVFILE_SHA256 = "c27be94ba7ed50e4d23d78a5e4a350202f20ac74e2a40bf40f77d6069d3696fc"
QDRANT_RUNTIME_REVISION = "74f3e85b9473c62560006c043e13737ce6b48412"
QDRANT_VERSION = "1.19.0"
SOURCE_MANIFEST_SCHEMA = "memupdatebench.cross-domain.source-manifest.v1"
REMOTE_PARENT = PurePosixPath('/NAS/yesh/MemUpdateBench/external')
BINARY = REMOTE_PARENT / 'qdrant_server_artifacts_1_19_0_20260905_v1/qdrant'
ENVFILE = REMOTE_PARENT / 'qdrant_qwen_environment_20260905_v1/environment.sh'
PYTHON = REMOTE_PARENT / 'qdrant_py310_v1/bin/python'
RELEASE_PINS = {
    'bea': '5525357436510b0c4e89efa53cdb6caf551c59234171c71bb97563e4c641790a',
    'noaa': '738f46932d8e2e8c26da0a2820dc67c7b0bad9aa43659f4be266f22f0fc8ee3f',
}
QUALITY_FIELDS = ('state_step_matches', 'final_state_matches', 'retrieval_typed_object_matches')
DEADLINE_SECONDS = 720
STORAGE_LIMIT = 1024**3 - 64 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(',', ':')) + '\n').encode('utf-8')


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _no_links(path):
    path = Path(path).absolute()
    _require('..' not in path.parts, 'path traversal forbidden')
    for part in (path, *path.parents):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(metadata.st_mode)
                 and not getattr(metadata, 'st_file_attributes', 0) & 0x400,
                 'symlink or reparse path forbidden')
    return path


def _safe_file(path, label):
    path = _no_links(path)
    _require(path.is_file(), label + ' must be a regular file')
    return path


def build_source_manifest(root: Path, paths: Iterable[Path]):
    root = _no_links(root)
    _require(root.is_dir(), 'source root must be a directory')
    rows = []
    for candidate in paths:
        path = _safe_file(candidate, 'source')
        rows.append({'path': path.relative_to(root).as_posix(),
                     'bytes': path.stat().st_size, 'sha256': _sha_file(path)})
    rows.sort(key=lambda row: row['path'])
    _require(rows and len({r['path'] for r in rows}) == len(rows), 'empty or duplicate source manifest')
    return {'schema': SOURCE_MANIFEST_SCHEMA, 'scope': 'explicit_source_bundle_not_clean_git_revision',
            'files': rows, 'bundle_sha256': _sha_bytes(canonical_json_bytes(rows))}


def verify_source_manifest(root, manifest_path, expected_sha):
    root = _no_links(root)
    raw = _safe_file(manifest_path, 'source manifest').read_bytes()
    _require(_sha_bytes(raw) == expected_sha, 'source manifest hash mismatch')
    manifest = json.loads(raw)
    _require(manifest.get('schema') == SOURCE_MANIFEST_SCHEMA, 'source manifest schema mismatch')
    rows = manifest['files']
    names = [r['path'] for r in rows]
    for name in names:
        value = PurePosixPath(name)
        _require(not value.is_absolute() and '..' not in value.parts and str(value) == name,
                 'unsafe source manifest member')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != Path(manifest_path)}
    _require(actual == set(names), 'source membership mismatch')
    measured = build_source_manifest(root, (root / name for name in names))
    _require(measured == manifest, 'source hash mismatch')
    return manifest


def claim_owned_directory(path):
    path = _no_links(path)
    path.mkdir(mode=0o700, exist_ok=False)
    metadata = path.lstat()
    marker = secrets.token_bytes(32)
    with (path / 'ownership.marker').open('xb') as stream:
        stream.write(marker)
    return {'device': metadata.st_dev, 'inode': metadata.st_ino, 'uid': metadata.st_uid,
            'marker_sha256': _sha_bytes(marker)}


def remove_owned_directory(path, identity):
    path = _no_links(path)
    metadata = path.lstat()
    _require((metadata.st_dev, metadata.st_ino, metadata.st_uid) ==
             (identity['device'], identity['inode'], identity['uid']), 'directory ownership changed')
    _require(_sha_file(_safe_file(path / 'ownership.marker', 'ownership marker')) == identity['marker_sha256'],
             'directory ownership marker changed')
    if os.name == 'posix':
        _require(metadata.st_uid == os.getuid() and shutil.rmtree.avoids_symlink_attacks,
                 'safe owned directory deletion unavailable')
    shutil.rmtree(path)


def validate_run_paths(run_root, temp_root):
    run, temp = PurePosixPath(run_root), PurePosixPath(temp_root)
    _require(run.parent == REMOTE_PARENT
             and re.fullmatch(r'cross_domain_state_20260930_v[1-9][0-9]*', run.name), 'unauthorized run root')
    version = run.name.rsplit('_', 1)[-1]
    _require(temp == PurePosixPath('/tmp') / ('mub-qdrant-cross-domain-20260930-' + version),
             'unauthorized temporary root')


class AnchoredRuntimeReader:
    def __init__(self, root):
        self.root = _no_links(root)
        self.closed = False
        self.fds = {}
        if os.name == 'posix' and os.open in os.supports_dir_fd:
            self.fds[()] = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def read(self, name, *, limit):
        relative = PurePosixPath(name)
        _require(not self.closed and not relative.is_absolute() and '..' not in relative.parts,
                 'runtime reader is closed or member escapes anchor')
        _require(relative.parts and type(limit) is int and limit >= 0, 'invalid bounded runtime read')
        if not self.fds:
            path = _safe_file(self.root / name, 'runtime dependency')
            _require(path.stat().st_size <= limit, 'runtime file exceeds read bound')
            return path.read_bytes()
        parent = ()
        for part in relative.parts[:-1]:
            current = (*parent, part)
            if current not in self.fds:
                _require(len(self.fds) < 512, 'runtime directory descriptor budget exceeded')
                self.fds[current] = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=self.fds[parent])
            parent = current
        fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fds[parent])
        with os.fdopen(fd, 'rb') as stream:
            metadata = os.fstat(stream.fileno())
            _require(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= limit,
                     'runtime file is not regular or exceeds read bound')
            raw = stream.read(metadata.st_size + 1)
            _require(len(raw) == metadata.st_size, 'runtime file changed size during read')
            return raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        self.closed = True


def _canonical_dist_name(value):
    return re.sub(r'[-_.]+', '-', value.casefold())


def materialize_cpu_runtime(original, destination, package_roots=None):
    import base64
    import csv
    import io
    original, destination = _no_links(original), _no_links(destination)
    site = original / 'lib/python3.10/site-packages'
    records = {}
    distribution_dirs = sorted(site.glob('*.dist-info'))
    distribution_names = {}
    requirements = {}
    for dist_dir in distribution_dirs:
        metadata = dist_dir / 'METADATA'
        dist_name = None
        requires = []
        if metadata.is_file():
            for line in metadata.read_text(errors='strict').splitlines():
                if line.startswith('Name: '):
                    dist_name = line[6:].strip()
                elif line.startswith('Requires-Dist: '):
                    requirement = line[15:]
                    requirement_parts = requirement.split(';', 1)
                    if len(requirement_parts) == 2:
                        marker = requirement_parts[1].casefold()
                        if 'extra' in marker or 'sys_platform' in marker and 'win32' in marker or 'platform_system' in marker and 'windows' in marker:
                            continue
                    match = re.match(r'\s*([A-Za-z0-9_.-]+)', requirement_parts[0])
                    if match:
                        requires.append(match.group(1))
        if dist_name is None:
            dist_name = dist_dir.name.split('-')[0]
        canonical_name = _canonical_dist_name(dist_name)
        distribution_names[canonical_name] = dist_dir
        requirements[canonical_name] = requires
    if package_roots is None:
        selected_names = set(distribution_names)
    else:
        pending = [_canonical_dist_name(name) for name in package_roots]
        selected_names = set()
        while pending:
            name = pending.pop()
            if name in selected_names:
                continue
            _require(name in distribution_names, 'requested runtime distribution missing: ' + name)
            selected_names.add(name)
            pending.extend(_canonical_dist_name(dep) for dep in requirements.get(name, ()))
    selected_dirs = {distribution_names[name] for name in selected_names}
    for record_file in sorted(dist_dir / 'RECORD' for dist_dir in selected_dirs):
        for name, expected_hash, expected_size in csv.reader(io.StringIO(record_file.read_text())):
            if name.endswith('.pyc') or '__pycache__' in PurePosixPath(name).parts:
                continue
            source = Path(os.path.normpath(str(site / name)))
            _require(original in source.parents, 'runtime RECORD path escapes environment')
            relative = source.relative_to(original)
            row = (expected_hash, expected_size)
            _require(relative not in records or records[relative] == row, 'conflicting runtime RECORD entries')
            records[relative] = row
    _require(records and sum(int(size) for _, size in records.values() if size) < 512 * 1024**2,
             'runtime copy exceeds budget or has no RECORD inventory')
    destination.mkdir(mode=0o700, exist_ok=False)
    rows, total = [], 0
    with AnchoredRuntimeReader(original) as reader:
        for relative, (expected_hash, expected_size) in sorted(records.items()):
            raw = reader.read(relative.as_posix(), limit=512 * 1024**2 - total)
            digest = hashlib.sha256(raw)
            _require(not expected_size or len(raw) == int(expected_size), 'runtime RECORD size mismatch')
            _require(not expected_hash or expected_hash == 'sha256=' + base64.urlsafe_b64encode(digest.digest()).decode().rstrip('='),
                     'runtime RECORD hash mismatch')
            total += len(raw)
            _require(total < 512 * 1024**2, 'runtime copy exceeds byte budget')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
            _require(_sha_file(target) == digest.hexdigest(), 'local runtime copy differs')
            rows.append({'path': relative.as_posix(), 'bytes': len(raw), 'sha256': digest.hexdigest()})
            if len(rows) % 250 == 0:
                print(json.dumps({'stage': 'runtime_materialization', 'files_completed': len(rows),
                                  'file_count': len(records), 'bytes_completed': total}), flush=True)
    return {'file_count': len(rows), 'bytes': total, 'files': rows,
            'tree_sha256': _sha_bytes(canonical_json_bytes(rows)),
            'scope': 'RECORD_verified_copy_of_existing_CPU_environment_no_installation'}


def verify_runtime_snapshot(root, manifest):
    root = _no_links(root)
    _require(isinstance(manifest, dict) and isinstance(manifest.get('files'), list),
             'runtime snapshot manifest is malformed')
    expected = {row['path']: row for row in manifest['files']}
    _require(len(expected) == len(manifest['files']) == manifest['file_count'], 'runtime file count or duplicate member mismatch')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    _require(actual == set(expected), 'runtime file membership changed')
    rows = []
    with AnchoredRuntimeReader(root) as reader:
        for row in manifest['files']:
            name = row['path']
            raw = reader.read(name, limit=int(row['bytes']))
            digest = _sha_bytes(raw)
            _require(len(raw) == row['bytes'] and digest == row['sha256'],
                     'runtime file hash or size changed: ' + name)
            rows.append({'path': name, 'bytes': len(raw), 'sha256': digest})
    _require(_sha_bytes(canonical_json_bytes(rows)) == manifest['tree_sha256'],
             'runtime snapshot tree hash changed')
    return {'file_count': len(rows), 'bytes': sum(row['bytes'] for row in rows),
            'files': rows, 'tree_sha256': manifest['tree_sha256'], 'scope': manifest.get('scope')}


def materialize_yaml_dependency(site, destination):
    site, destination = _no_links(site), _no_links(destination)
    _require((site / 'yaml').is_dir() and (site / 'pyyaml-6.0.3.dist-info').is_dir(),
             'existing PyYAML 6.0.3 source unavailable')
    destination.mkdir(mode=0o700, exist_ok=False)
    rows = []
    for name in ('yaml', '_yaml', 'pyyaml-6.0.3.dist-info'):
        folder = site / name
        for source in sorted(folder.rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
                continue
            raw = _safe_file(source, 'existing dependency').read_bytes()
            relative = source.relative_to(site)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
            _require(source.read_bytes() == raw and target.read_bytes() == raw, 'dependency copy changed')
            rows.append({'path': relative.as_posix(), 'bytes': len(raw), 'sha256': _sha_bytes(raw)})
    _require(rows, 'empty dependency copy')
    return {'name': 'PyYAML', 'version': '6.0.3', 'file_count': len(rows),
            'files': rows, 'tree_sha256': _sha_bytes(canonical_json_bytes(rows)),
            'origin': 'byte_copy_from_existing_project_environment_no_installation'}


def build_server_config(temp_root: Path):
    return {'log_level': 'INFO', 'telemetry_disabled': True,
            'service': {'host': '127.0.0.1', 'http_port': 16333, 'grpc_port': None,
                        'enable_cors': False, 'max_workers': 2},
            'cluster': {'enabled': False},
            'storage': {'storage_path': str(temp_root / 'storage'),
                        'snapshots_path': str(temp_root / 'snapshots'), 'temp_path': str(temp_root / 'temp'),
                        'performance': {'max_search_threads': 2},
                        'optimizers': {'max_optimization_threads': 1}, 'wal': {'wal_capacity_mb': 16}}}


def _write_text_exclusive(path, text):
    raw = text.encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def finalize_result(result, cleanup, *, bindings_unchanged, error):
    result = dict(result or {'rows': []})
    complete = (result.get('status') == 'COMPLETE' and error is None and bindings_unchanged
                and all(cleanup.get(k) is True for k in ('processes_stopped', 'directory_removed', 'port_closed')))
    result['status'] = 'COMPLETE' if complete else 'BLOCKED'
    result['cleanup'] = dict(cleanup)
    result['bindings_unchanged'] = bindings_unchanged
    result['error'] = error
    result['scientific_evidence'] = False
    result['scientific_release_allowed'] = False
    result['benchmark_accuracy_claimed'] = False
    result['evidence_class'] = ('bounded_cross_domain_qdrant_state_retrieval' if complete
                                else 'cross_domain_runtime_failure_diagnostic')
    result['answer_metrics'] = None
    if not complete:
        for name in QUALITY_FIELDS:
            result[name] = None
        result['repetitions_equal'] = None
    return result


def _tree_snapshot(root):
    root = _no_links(root)
    rows = []
    for path in sorted(root.rglob('*')):
        _no_links(path)
        if path.is_file():
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                         'sha256': _sha_file(path)})
    return rows


def _filesystem_info(path):
    completed = subprocess.run(['findmnt', '-J', '-T', str(path), '-o', 'SOURCE,FSTYPE'],
                               capture_output=True, text=True, check=True, timeout=10)
    rows = json.loads(completed.stdout)['filesystems']
    _require(len(rows) == 1 and rows[0]['fstype'] == 'ext4'
             and rows[0]['source'].startswith('/dev/'), 'local ext4 block storage required')
    return {'fstype': 'ext4', 'block_device': True}


def _storage_sample(path):
    apparent = allocated = 0
    for parent, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            child = Path(parent) / name
            metadata = child.lstat()
            _require(not stat.S_ISLNK(metadata.st_mode), 'symlink in owned storage')
            apparent += metadata.st_size
            allocated += getattr(metadata, 'st_blocks', 0) * 512
    _require(max(apparent, allocated) < STORAGE_LIMIT, 'storage budget exceeded')
    _require(shutil.disk_usage(path).free >= 64 * 1024**2, 'local free-space floor reached')
    return max(apparent, allocated)


def _group_members(pgid):
    members = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append({'pid': int(directory.name), 'session': int(fields[3]),
                                'uid': directory.stat().st_uid, 'start': fields[19]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def _stop_process(process):
    if process is None:
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        members = _group_members(process.pid)
        if not members:
            if process.poll() is None:
                process.wait(timeout=2)
            return
        _require(all(m['session'] == process.pid and m['uid'] == os.getuid() for m in members),
                 'process group ownership changed')
        os.killpg(process.pid, signum)
        until = time.monotonic() + 10
        while time.monotonic() < until:
            process.poll()
            if not _group_members(process.pid):
                return
            time.sleep(0.1)
    raise TimeoutError('owned process group remains after termination')


def _port_closed():
    for name in ('tcp', 'tcp6'):
        for row in Path('/proc/net', name).read_text().splitlines()[1:]:
            fields = row.split()
            if fields[1].rsplit(':', 1)[1] == '3FCD' and fields[3] == '0A':
                return False
    return True


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))


def _listener_owned(process):
    _require(process.poll() is None, 'owned server exited')
    links = set()
    for path in Path('/proc', str(process.pid), 'fd').iterdir():
        try:
            links.add(os.readlink(path))
        except FileNotFoundError:
            pass
    listeners = [r.split() for r in Path('/proc/net/tcp').read_text().splitlines()[1:]
                 if r.split()[1] == '0100007F:3FCD' and r.split()[3] == '0A']
    if not listeners:
        return False
    _require(len(listeners) == 1 and 'socket:[' + listeners[0][9] + ']' in links,
             'listener is not owned by this server')
    return True


def _http_get(path, api_key):
    import http.client
    connection = http.client.HTTPConnection('127.0.0.1', 16333, timeout=2)
    try:
        connection.request('GET', path, headers={'api-key': api_key} if api_key else {})
        response = connection.getresponse()
        raw = response.read()
        return response.status, raw
    finally:
        connection.close()


def _check_import_origins(source):
    for name, module in tuple(sys.modules.items()):
        if name == 'mub' or name.startswith('mub.') or name == 'scripts' or name.startswith('scripts.'):
            filename = getattr(module, '__file__', None)
            if filename:
                _require(source in Path(filename).resolve().parents, 'project import escaped source bundle')


def build_local_runtime_manifest(runtime_copy, executable, venv_config, versions):
    executable, venv_config = _safe_file(executable, 'runtime executable'), _safe_file(venv_config, 'venv config')
    _require(type(runtime_copy) is dict and type(runtime_copy.get('file_count')) is int
             and type(runtime_copy.get('tree_sha256')) is str and len(runtime_copy['tree_sha256']) == 64,
             'runtime copy manifest is malformed')
    _require(isinstance(versions, dict) and versions.get('qdrant-client') == QDRANT_VERSION,
             'minimal runtime versions are not qualified')
    return {'python_version': sys.version.split()[0], 'python_executable_sha256': _sha_file(executable),
            'python_venv_config_sha256': _sha_file(venv_config), 'runtime_copy': dict(runtime_copy),
            'versions': dict(versions), 'distribution_count': len(versions),
            'scope': 'local_record_verified_runtime_copy_plus_minimal_version_identity'}


def _runtime_manifest(runtime_copy=None):
    if runtime_copy is not None:
        import importlib.metadata as metadata
        versions = {name: metadata.version(name) for name in ('qdrant-client', 'pydantic', 'PyYAML')}
        return build_local_runtime_manifest(runtime_copy, Path(sys.executable).resolve(strict=True),
                                            Path(sys.prefix).resolve(strict=True) / 'pyvenv.cfg', versions)
    import importlib.metadata as metadata
    print(json.dumps({'stage': 'dependency_measurement_started'}), flush=True)
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    pending = ['qdrant-client', 'pydantic', 'pip', 'pyyaml']
    seen, records = set(), []
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        dist = metadata.distribution(name)
        if name == 'qdrant-client':
            _require(dist.version == QDRANT_VERSION, 'qdrant-client version mismatch')
        hashes = []
        for entry in sorted(dist.files or (), key=str):
            if str(entry).endswith('.pyc'):
                continue
            path = Path(dist.locate_file(entry))
            _require(path.is_file(), 'declared runtime distribution file missing')
            hashes.append({'member': str(entry), 'bytes': path.stat().st_size, 'sha256': _sha_file(path)})
        _require(hashes, 'runtime distribution content unavailable')
        records.append({'name': name, 'version': dist.version, 'files': hashes})
        print(json.dumps({'stage': 'dependency_measured', 'name': name, 'file_count': len(hashes)}), flush=True)
        for text in dist.requires or ():
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({'extra': ''}):
                pending.append(req.name)
    records.sort(key=lambda r: r['name'])
    return {'python_version': sys.version.split()[0], 'python_executable_sha256': _sha_file(Path(sys.executable).resolve()),
            'python_venv_config_sha256': _sha_file(Path(sys.prefix) / 'pyvenv.cfg'),
            'distribution_count': len(records), 'distributions': records,
            'scope': 'interpreter_bytes_and_declared_dependency_files_not_OS_shared_library_closure'}


def preflight_client():
    import httpx
    from qdrant_client import QdrantClient
    calls = []
    def reject(request):
        calls.append(request.method)
        raise ValueError('preflight must not issue provider requests')
    with httpx.MockTransport(reject) as transport:
        client = QdrantClient(url='http://127.0.0.1:16333', prefer_grpc=False,
            check_compatibility=False, timeout=10, trust_env=False, transport=transport)
        client.close()
    _require(not calls, 'preflight issued a provider request')
    return {'status': 'CLIENT_CONSTRUCTION_PASS', 'provider_calls': 0, 'model_loads': 0}


def worker(args):
    print(json.dumps({'stage': 'worker_python_started', 'mode': args.mode}), flush=True)
    source = Path(args.source).resolve()
    verify_source_manifest(source, source / 'source_manifest.json', args.source_sha256)
    sys.path.insert(0, str(source))
    from scripts.vnext_run_cross_domain_state import load_public_cells, run_cross_domain_cells, build_qdrant_manager_factory
    temp = Path(args.temp_root)
    releases = {name: (temp / 'input_release' / name, pin) for name, pin in RELEASE_PINS.items()}
    cells = load_public_cells(releases)
    _check_import_origins(source)
    if args.mode == 'preflight':
        client_check = preflight_client()
        runtime_copy = json.loads((temp / 'runtime_copy_manifest.json').read_bytes())
        runtime = _runtime_manifest(runtime_copy)
        report = {'client_preflight': client_check, 'cells_sha256': _sha_bytes(canonical_json_bytes(cells)), 'cells': len(cells),
                  'events': sum(len(c['events']) for c in cells), 'runtime': runtime,
                  'runtime_sha256': _sha_bytes(canonical_json_bytes(runtime))}
        _write_text_exclusive(temp / 'preflight.json', canonical_json_bytes(report).decode())
        return 0
    preflight_raw = (temp / 'preflight.json').read_bytes()
    _require(_sha_bytes(preflight_raw) == os.environ['MUB_PREFLIGHT_SHA256'], 'preflight changed')
    preflight = json.loads(preflight_raw)
    _require(preflight['cells_sha256'] == _sha_bytes(canonical_json_bytes(cells)), 'public cells changed')
    import httpx
    calls = []

    class Transport(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(retries=0, trust_env=False)

        def handle_request(self, request):
            _require(request.url.host == '127.0.0.1' and request.url.port == 16333
                     and request.url.scheme == 'http', 'non-loopback provider request')
            record = {'method': request.method, 'path': request.url.raw_path.decode(),
                      'authenticated': bool(request.headers.get('api-key')),
                      'request_sha256': _sha_bytes(request.method.encode() + request.url.raw_path + request.content)}
            calls.append(record)
            try:
                response = self.inner.handle_request(request)
                record.update(status=response.status_code, response_sha256=_sha_bytes(response.read()))
                return response
            except Exception as exc:
                record['error_type'] = type(exc).__name__
                raise

        def close(self):
            self.inner.close()

    factory = build_qdrant_manager_factory(endpoint='http://127.0.0.1:16333',
        qdrant_base_path=temp / 'UNUSED_REMOTE_CLIENT_PATH', source_revision=args.source_sha256[:40],
        runtime_revision=preflight['runtime_sha256'][:40], source_hash=args.source_sha256,
        runtime_hash=preflight['runtime_sha256'], transport_factory=Transport,
        run_scope=Path(args.run_root).name.replace('_', '-'))
    result = run_cross_domain_cells(cells, manager_factory=factory, repetitions=2)
    _write_text_exclusive(temp / 'observed_rows.json', canonical_json_bytes(result).decode())
    _write_text_exclusive(temp / 'http_calls.json', canonical_json_bytes(calls).decode())
    runtime_copy = preflight['runtime']['runtime_copy']
    _require(verify_runtime_snapshot(temp / 'cpu_runtime', runtime_copy) == runtime_copy,
             'runtime files changed during execution')
    runtime_after = _runtime_manifest(runtime_copy)
    _require(runtime_after == preflight['runtime'], 'runtime identity changed during execution')
    verify_source_manifest(source, source / 'source_manifest.json', args.source_sha256)
    _require(load_public_cells(releases) == cells, 'release changed during execution')
    _check_import_origins(source)
    _write_text_exclusive(temp / 'worker_complete.json', canonical_json_bytes({'status': result['status'],
        'runtime_revalidated': True, 'source_revalidated': True, 'release_revalidated': True}).decode())
    return 0 if result['status'] == 'COMPLETE' else 2


def run_live(*, run_root, temp_root, source_sha256):
    _require(sys.platform == 'linux', 'live canary requires Linux')
    run_root, temp_root = Path(run_root), Path(temp_root)
    validate_run_paths(run_root, temp_root)
    source = run_root / 'source'
    _require(PROJECT_ROOT == source, 'entrypoint must be in its frozen source bundle')
    _no_links(run_root)
    output = run_root / 'results'
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    owned, server, child = None, None, None
    files, setup_http, processes = [], [], []
    worker_started = False
    cleanup = {'processes_stopped': False, 'directory_removed': False, 'port_closed': False}
    before, runtime, observed, atomic = None, None, None, None
    unchanged, error = False, None
    phase = 'input_validation'
    accounting = {'model_loads': 0, 'generations': 0, 'provider_model_calls': 0, 'gpu_calls': 0,
                  'server_instances_started': 0, 'retries': 0, 'peak_owned_storage_bytes': 0}

    def alarm_handler(signum, frame):
        raise TimeoutError('supervisor deadline')

    def sample():
        if owned is not None:
            accounting['peak_owned_storage_bytes'] = max(accounting['peak_owned_storage_bytes'], _storage_sample(temp_root))
        if server is not None:
            _require(server.poll() is None, 'owned server exited during canary')

    def launch(argv, env, log_name, timeout, limits=True):
        nonlocal child
        log = (temp_root / log_name).open('xb')
        files.append(log)
        child = subprocess.Popen(argv, cwd=temp_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=_limits if limits else None)
        processes.append(child)
        until = time.monotonic() + timeout
        while child.poll() is None:
            sample()
            if time.monotonic() >= until:
                _stop_process(child)
                raise TimeoutError('child deadline')
            time.sleep(0.5)
        _require(child.returncode == 0, 'child execution failed')

    old_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    old_term = signal.signal(signal.SIGTERM, alarm_handler)
    signal.alarm(DEADLINE_SECONDS)
    try:
        manifest = verify_source_manifest(source, source / 'source_manifest.json', source_sha256)
        if sys.version_info >= (3, 9):
            spec = importlib.util.spec_from_file_location('cross_domain_atomic', source / 'mub/vnext/io/atomic.py')
            publisher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(publisher)
            atomic = publisher
        before = {'source': _tree_snapshot(source), 'release': _tree_snapshot(run_root / 'input_release')}
        for name, pin in RELEASE_PINS.items():
            _require(_sha_file(run_root / 'input_release' / name / 'release_index.json') == pin, 'release hash mismatch')
        phase = 'runtime_storage_preflight'
        filesystem = _filesystem_info(temp_root.parent)
        _require(shutil.disk_usage(temp_root.parent).free >= 2 * 1024**3, 'insufficient local space')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 16333))
        _require(_sha_file(_safe_file(ENVFILE, 'environment')) == QDRANT_ENVFILE_SHA256, 'environment hash mismatch')
        _require(_sha_file(_safe_file(BINARY, 'binary')) == QDRANT_BINARY_SHA256, 'binary hash mismatch')
        owned = claim_owned_directory(temp_root)
        _require(temp_root.stat().st_uid == os.getuid() and stat.S_IMODE(temp_root.stat().st_mode) == 0o700,
                 'temporary ownership or mode mismatch')
        for name in ('home', 'temp', 'storage', 'snapshots', 'config'):
            (temp_root / name).mkdir(mode=0o700)
        env_result = subprocess.run(['/bin/bash', '-c', 'source "$1"; env -0', 'mub', str(ENVFILE)],
            env={'PATH': '/usr/bin:/bin', 'HOME': str(temp_root / 'home')}, capture_output=True, check=True, timeout=15)
        env = dict(item.decode().split('=', 1) for item in env_result.stdout.split(b'\0') if b'=' in item)
        for key in list(env):
            if key.startswith('QDRANT__'):
                del env[key]
        env.update(HOME=str(temp_root / 'home'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONNOUSERSITE='1', PYTHONPATH=str(source), LC_ALL='C.UTF-8',
                   NO_PROXY='127.0.0.1,localhost', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR', 'PYTHONPYCACHEPREFIX'):
            env[key] = str(temp_root / 'temp')
        phase = 'local_source_materialization'
        shutil.copytree(source, temp_root / 'source')
        shutil.copytree(run_root / 'input_release', temp_root / 'input_release')
        verify_source_manifest(temp_root / 'source', temp_root / 'source/source_manifest.json', source_sha256)
        _require(_tree_snapshot(temp_root / 'input_release') == before['release'], 'local input copy mismatch')
        base = [str(PYTHON), '-B', '-u', str(temp_root / 'source/scripts/vnext_run_cross_domain_qdrant_state.py'),
                '--run-root', str(run_root), '--temp-root', str(temp_root), '--source-sha256', source_sha256]
        extra = materialize_yaml_dependency(
            Path('/NAS/yesh/MemUpdateBench/.venv-factorial-song1-clean/lib/python3.10/site-packages'),
            temp_root / 'runtime_extra')
        accounting['isolated_dependency'] = {k: v for k, v in extra.items() if k != 'files'}
        phase = 'runtime_materialization'
        accounting['runtime_copy'] = materialize_cpu_runtime(
            Path('/NAS/yesh/MemUpdateBench/external/qdrant_py310_v1'),
            temp_root / 'cpu_runtime',
            package_roots=('qdrant-client', 'pydantic', 'httpx', 'httpcore', 'pip'))
        _write_text_exclusive(temp_root / 'runtime_copy_manifest.json',
                              canonical_json_bytes(accounting['runtime_copy']).decode())
        env['PYTHONPATH'] = os.pathsep.join((str(temp_root / 'source'), str(temp_root / 'runtime_extra'),
            str(temp_root / 'cpu_runtime/lib/python3.10/site-packages')))
        phase = 'python_dependency_preflight'
        launch(base + ['--mode', 'preflight'], env, 'preflight.log', 240)
        runtime = json.loads((temp_root / 'preflight.json').read_bytes())
        _require(runtime['cells'] == 2 and runtime['events'] == 10, 'preflight cardinality mismatch')
        phase = 'server_launch'
        shutil.copyfile(BINARY, temp_root / 'qdrant')
        _require(_sha_file(temp_root / 'qdrant') == QDRANT_BINARY_SHA256, 'copied binary hash mismatch')
        (temp_root / 'qdrant').chmod(0o700)
        config = build_server_config(temp_root)
        _write_text_exclusive(temp_root / 'server.yaml', canonical_json_bytes(config).decode())
        api_key = secrets.token_hex(32)
        server_env = {'PATH': '/usr/bin:/bin', 'HOME': env['HOME'], 'TMPDIR': env['TMPDIR'],
                      'XDG_CONFIG_HOME': str(temp_root / 'config'), 'CUDA_VISIBLE_DEVICES': '',
                      'QDRANT__SERVICE__API_KEY': api_key, 'RUST_BACKTRACE': '0'}
        log = (temp_root / 'server.log').open('xb')
        files.append(log)
        server = subprocess.Popen([str(temp_root / 'qdrant'), '--config-path', str(temp_root / 'server.yaml')],
            cwd=temp_root, env=server_env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=_limits)
        processes.append(server)
        accounting['server_instances_started'] = 1
        ready = False
        for attempt in range(40):
            sample()
            if not _listener_owned(server):
                time.sleep(0.5)
                continue
            call = {'method': 'GET', 'path': '/', 'authenticated': True}
            setup_http.append(call)
            try:
                code, raw = _http_get('/', api_key)
                call.update(status=code, response_sha256=_sha_bytes(raw))
                if code == 200:
                    info = json.loads(raw)
                    _require(info['version'] == QDRANT_VERSION and info['commit'] == QDRANT_RUNTIME_REVISION,
                             'server runtime identity mismatch')
                    ready = True
                    break
            except (OSError, TimeoutError) as exc:
                call['error_type'] = type(exc).__name__
            time.sleep(0.5)
        _require(ready, 'server readiness timeout')
        _listener_owned(server)
        code, raw = _http_get('/collections', None)
        setup_http.append({'method': 'GET', 'path': '/collections', 'authenticated': False,
                           'status': code, 'response_sha256': _sha_bytes(raw)})
        _require(code in (401, 403), 'unauthenticated service access permitted')
        accounting['listener_owned'] = True
        phase = 'state_retrieval_worker'
        env['MUB_DIAGNOSTIC_API_KEY'] = api_key
        env['MUB_PREFLIGHT_SHA256'] = _sha_file(temp_root / 'preflight.json')
        worker_started = True
        launch(base + ['--mode', 'worker'], env, 'worker.log', 300)
        observed = json.loads((temp_root / 'observed_rows.json').read_bytes())
        finish = json.loads((temp_root / 'worker_complete.json').read_bytes())
        _require(all(finish.get(k) is True for k in ('runtime_revalidated', 'source_revalidated', 'release_revalidated')),
                 'worker did not revalidate inputs')
        phase = 'postrun_binding'
        after = {'source': _tree_snapshot(source), 'release': _tree_snapshot(run_root / 'input_release')}
        _require(before == after and _sha_file(BINARY) == QDRANT_BINARY_SHA256
                 and _sha_file(ENVFILE) == QDRANT_ENVFILE_SHA256, 'bound input changed')
        unchanged = True
    except BaseException as exc:
        error = {'phase': phase, 'error_type': type(exc).__name__}
    finally:
        signal.alarm(90)
        diagnostics = {}
        try:
            for process in reversed(processes):
                try:
                    _stop_process(process)
                except BaseException as exc:
                    cleanup['process_error_type'] = type(exc).__name__
            cleanup['processes_stopped'] = all(p.poll() is not None and not _group_members(p.pid) for p in processes)
            for stream in files:
                try:
                    stream.close()
                except OSError as exc:
                    error = error or {'phase': 'log_close', 'error_type': type(exc).__name__}
            try:
                if owned is not None:
                    for name in ('preflight.log', 'server.log', 'worker.log'):
                        path = temp_root / name
                        if path.is_file():
                            diagnostics[name] = {'bytes': path.stat().st_size, 'sha256': _sha_file(path)}
                    for name in ('observed_rows.json', 'http_calls.json', 'worker_error.json'):
                        path = temp_root / name
                        if path.is_file():
                            value = json.loads(path.read_bytes())
                            if name == 'observed_rows.json':
                                observed = value
                            elif name == 'http_calls.json':
                                diagnostics['worker_http_calls'] = value
                            else:
                                diagnostics['worker_error'] = value
            except BaseException as exc:
                error = error or {'phase': 'diagnostic_collection', 'error_type': type(exc).__name__}
            finally:
                if owned is not None and cleanup['processes_stopped']:
                    try:
                        remove_owned_directory(temp_root, owned)
                        cleanup['directory_removed'] = True
                    except BaseException as exc:
                        cleanup['directory_error_type'] = type(exc).__name__
                elif owned is None:
                    cleanup['directory_removed'] = not temp_root.exists()
                try:
                    cleanup['port_closed'] = _port_closed()
                except OSError as exc:
                    cleanup['port_check_error_type'] = type(exc).__name__
        except BaseException as exc:
            error = error or {'phase': 'cleanup', 'error_type': type(exc).__name__}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_alarm)
            signal.signal(signal.SIGTERM, old_term)
    result = finalize_result(observed, cleanup, bindings_unchanged=unchanged, error=error)
    result['execution_boundary'] = {**accounting, 'child_process_launches': len(processes),
        'setup_http_calls': len(setup_http),
        'worker_http_calls': (len(diagnostics['worker_http_calls']) if 'worker_http_calls' in diagnostics
                              else None if worker_started else 0),
        'elapsed_seconds': time.monotonic() - started,
        'accounting_scope': 'owned_runtime_children_and_loopback_requests_not_launcher_ssh_calls'}
    result.update(model_loads=0, generations=0, provider_model_calls=0)
    result['claim_boundary'] = ('two scalar single-slot source trajectories; repeated runs are not independent cores; '
        'generated structured event surface; direct Qdrant CRUD and hash-vector retrieval only; '
        'no semantic retrieval, native answer, prompted answer or broad benchmark claim')
    result['source_manifest_sha256'] = source_sha256
    result['release_bindings'] = RELEASE_PINS
    result['runtime_binding'] = None if runtime is None else {'runtime_sha256': runtime['runtime_sha256'],
        'python_version': runtime['runtime']['python_version'], 'distribution_count': runtime['runtime']['distribution_count'],
        'binary_sha256': QDRANT_BINARY_SHA256, 'server_revision': QDRANT_RUNTIME_REVISION}
    rows = result.pop('rows', [])
    artifacts = {'summary.json': canonical_json_bytes(result),
                 'rows.jsonl': b''.join(canonical_json_bytes(row) for row in rows),
                 'runtime.json': canonical_json_bytes(runtime),
                 'http_accounting.json': canonical_json_bytes({'setup': setup_http, **diagnostics}),
                 'input_binding.json': canonical_json_bytes({'release_pins': RELEASE_PINS,
                     'source_manifest_sha256': source_sha256, 'before': before, 'unchanged': unchanged})}
    index = {'schema': 'memupdatebench.cross-domain.qdrant-state-index.v1', 'status': result['status'],
             'scientific_release_allowed': False, 'benchmark_accuracy_claimed': False,
             'artifacts': [{'path': n, 'bytes': len(raw), 'sha256': _sha_bytes(raw)} for n, raw in sorted(artifacts.items())]}
    artifacts['artifact_index.json'] = canonical_json_bytes(index)
    if atomic is not None:
        atomic.publish_files_atomically({output / n: raw for n, raw in artifacts.items()}, overwrite=False)
    else:
        # Readers admit the root only after the fsynced index commit marker exists.
        for name, raw in artifacts.items():
            _write_text_exclusive(output / name, raw.decode('utf-8'))
    return {'status': result['status'], 'error': error, 'cleanup': cleanup,
            'artifact_index_sha256': _sha_bytes(artifacts['artifact_index.json'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=('supervisor', 'preflight', 'worker'), default='supervisor')
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--temp-root', required=True)
    parser.add_argument('--source', default=str(PROJECT_ROOT))
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args(argv)
    if args.mode != 'supervisor':
        try:
            return worker(args)
        except Exception as exc:
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                frames.append({'file': Path(tb.tb_frame.f_code.co_filename).name, 'line': tb.tb_lineno,
                               'function': tb.tb_frame.f_code.co_name})
                tb = tb.tb_next
            error = {'status': 'BLOCKED', 'error_type': type(exc).__name__, 'frames': frames}
            _write_text_exclusive(Path(args.temp_root) / 'worker_error.json', canonical_json_bytes(error).decode())
            print(json.dumps(error), flush=True)
            return 2
    result = run_live(run_root=args.run_root, temp_root=args.temp_root, source_sha256=args.source_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
