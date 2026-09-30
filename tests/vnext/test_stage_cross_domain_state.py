from pathlib import Path
import hashlib
import json
import tarfile

import pytest


def test_package_revalidates_sources_and_release_copies(tmp_path):
    from scripts.vnext_stage_cross_domain_state import stage_package
    from scripts.vnext_run_cross_domain_qdrant_state import verify_source_manifest
    root = Path(__file__).resolve().parents[2]
    package = tmp_path / 'package'
    result = stage_package(root, package)
    raw = (package / 'source/source_manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result['source_manifest_sha256']
    verify_source_manifest(package / 'source', package / 'source/source_manifest.json', result['source_manifest_sha256'])
    for name, relative in [('bea', 'family_h_cross_domain_bea_gdp'), ('noaa', 'family_h_cross_domain_noaa')]:
        original = root / 'data/vnext' / relative / 'v1'
        assert {p.name: p.read_bytes() for p in original.iterdir()} == {
            p.name: p.read_bytes() for p in (package / 'input_release' / name).iterdir()}
    with tarfile.open(package.with_suffix('.tar.gz'), 'r:gz') as archive:
        names = [member.name for member in archive.getmembers()]
        assert all(not name.startswith('/') and '..' not in Path(name).parts for name in names)
        assert 'source/scripts/vnext_run_cross_domain_qdrant_state.py' in names
        assert all(member.isfile() for member in archive.getmembers())
    assert (package / 'source/configs/vnext/main_track_family_h_realistic_source_v1.json').read_bytes() == (
        root / 'configs/vnext/main_track_family_h_realistic_source_v1.json').read_bytes()
    import os
    import subprocess
    import sys
    program = """
import sys
from pathlib import Path
from scripts.vnext_run_cross_domain_state import load_public_cells, BEA_INDEX_SHA256, NOAA_INDEX_SHA256
r = Path(sys.argv[1])
cells = load_public_cells({'bea': (r/'input_release/bea', BEA_INDEX_SHA256),
                          'noaa': (r/'input_release/noaa', NOAA_INDEX_SHA256)})
assert len(cells)==2 and sum(len(c['events']) for c in cells)==10
assert all(r/'source' in Path(m.__file__).resolve().parents for n,m in sys.modules.items()
           if (n=='mub' or n.startswith('mub.') or n.startswith('scripts.')) and getattr(m,'__file__',None))
print('ISOLATED_INPUT_PASS')
"""
    completed = subprocess.run([sys.executable, '-B', '-c', program, str(package)],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(package / 'source'), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    assert 'ISOLATED_INPUT_PASS' in completed.stdout
    with pytest.raises(FileExistsError):
        stage_package(root, package)


def test_package_rejects_frozen_input_output(tmp_path):
    from scripts.vnext_stage_cross_domain_state import stage_package
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        stage_package(root, root / 'data/vnext/family_h_cross_domain_bea_gdp/v1/nested')
