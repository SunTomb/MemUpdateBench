"""Build a no-replace cross-domain canary package without remote operations."""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vnext_run_cross_domain_qdrant_state import (
    _no_links, _require, _sha_bytes, _sha_file, build_source_manifest,
    canonical_json_bytes, verify_source_manifest,
)
from scripts.vnext_run_cross_domain_state import BEA_INDEX_SHA256, NOAA_INDEX_SHA256, load_public_cells

SCRIPT_NAMES = (
    'vnext_run_cross_domain_state.py', 'vnext_run_cross_domain_qdrant_state.py',
    'vnext_promote_family_h_cross_domain_bea.py', 'vnext_promote_family_h_cross_domain_noaa.py',
    'vnext_stage_cross_domain_state.py',
)


def stage_package(project_root, output_root):
    project, output = _no_links(project_root), _no_links(output_root)
    _require(output.is_absolute(), 'absolute output required')
    for parent in (output, *output.parents):
        _require(not any((parent / name).is_file() for name in ('release_index.json', 'index.json', 'artifact_index.json', 'task_manifest.json')),
                 'output overlaps indexed artifact root')
    for name in ('mub', 'scripts', 'data'):
        protected = project / name
        _require(output != protected and protected not in output.parents and output not in protected.parents,
                 'output overlaps source or frozen data')
    archive_path = output.with_suffix('.tar.gz')
    if output.exists() or archive_path.exists():
        raise FileExistsError('package or archive already exists')
    releases = {'bea': (project / 'data/vnext/family_h_cross_domain_bea_gdp/v1', BEA_INDEX_SHA256),
                'noaa': (project / 'data/vnext/family_h_cross_domain_noaa/v1', NOAA_INDEX_SHA256)}
    cells = load_public_cells(releases)
    paths = sorted((project / 'mub').rglob('*.py')) + [project / 'scripts' / name for name in SCRIPT_NAMES]
    paths.append(project / 'configs/vnext/main_track_family_h_realistic_source_v1.json')
    script_init = project / 'scripts/__init__.py'
    if script_init.is_file():
        paths.append(script_init)
    sources = {}
    payload = {}
    for path in paths:
        _no_links(path)
        raw = path.read_bytes()
        sources[path] = raw
        payload['source/' + path.relative_to(project).as_posix()] = raw
    for name, (root, _) in releases.items():
        for path in root.iterdir():
            _no_links(path)
            _require(path.is_file(), 'release contains non-file member')
            raw = path.read_bytes()
            sources[path] = raw
            payload[f'input_release/{name}/{path.name}'] = raw
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    for name, raw in payload.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    source = output / 'source'
    manifest = build_source_manifest(source, tuple(p for p in source.rglob('*') if p.is_file()))
    manifest_raw = canonical_json_bytes(manifest)
    (source / 'source_manifest.json').write_bytes(manifest_raw)
    payload['source/source_manifest.json'] = manifest_raw
    pin = _sha_bytes(manifest_raw)
    verify_source_manifest(source, source / 'source_manifest.json', pin)
    copied = {name: (output / 'input_release' / name, digest) for name, (_, digest) in releases.items()}
    _require(load_public_cells(copied) == cells, 'copied release projection differs')
    for path, raw in sources.items():
        _require(path.read_bytes() == raw, 'source changed during staging')
    with archive_path.open('xb') as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, raw in sorted(payload.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(raw), 0o600, 0
                    archive.addfile(info, io.BytesIO(raw))
    return {'status': 'PREPARED_NOT_EXECUTED', 'source_manifest_sha256': pin,
            'archive_sha256': _sha_file(archive_path), 'archive_bytes': archive_path.stat().st_size,
            'archive_members': len(payload), 'source_python_files': sum(r['path'].endswith('.py') for r in manifest['files']),
            'source_asset_files': sum(not r['path'].endswith('.py') for r in manifest['files']),
            'release_pins': {name: digest for name, (_, digest) in releases.items()},
            'cells': len(cells), 'events': sum(len(cell['events']) for cell in cells),
            'model_loads': 0, 'generations': 0, 'network_calls': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    result = stage_package(ROOT, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
