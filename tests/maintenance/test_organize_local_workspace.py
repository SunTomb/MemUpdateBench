from pathlib import Path

import pytest

from scripts.maintenance.organize_local_workspace import archive_paths, restore_archive, tree_fingerprint


def test_archive_and_restore_preserve_bytes(tmp_path):
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    source = workspace / 'old_source_bundle'
    source.mkdir()
    (source / 'index.json').write_bytes(b'{"fixed":true}\n')
    (source / 'nested').mkdir()
    (source / 'nested/code.py').write_bytes(b'print(1)\n')
    before = tree_fingerprint(source)
    root = project / '_local/archive/test'
    receipt = archive_paths(workspace, project, root, [(source, 'workspace_staging/old_source_bundle')])
    assert not source.exists()
    assert tree_fingerprint(root / 'workspace_staging/old_source_bundle') == before
    assert receipt['status'] == 'ARCHIVED'
    restore_archive(root / 'archive_receipt.json')
    assert tree_fingerprint(source) == before
    assert not (root / 'workspace_staging/old_source_bundle').exists()


def test_archive_refuses_existing_destination_and_git_worktree(tmp_path):
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    source = workspace / 'source'
    source.mkdir()
    (source / '.git').write_text('gitdir: protected', encoding='utf-8')
    with pytest.raises(ValueError, match='Git'):
        archive_paths(workspace, project, project / '_local/archive/no-git', [(source, 'source')])
    assert source.exists()


def test_archive_cannot_touch_protected_or_escape_workspace(tmp_path):
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    for relative in ('MemUpdateBench_releases', 'MemUpdateBench/data/vnext/core/v3', 'G-MSRA'):
        source = workspace / relative
        source.mkdir(parents=True)
        with pytest.raises(ValueError, match='protected'):
            archive_paths(workspace, project, project / '_local/archive/protected', [(source, 'source')])
    source = tmp_path / 'unrelated'
    source.mkdir()
    with pytest.raises(ValueError):
        archive_paths(workspace, project, project / '_local/archive/escape', [(source, 'outside')])


def test_restore_does_not_overwrite_new_source(tmp_path):
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    source = workspace / 'old.py'
    source.write_bytes(b'original')
    root = project / '_local/archive/test'
    archive_paths(workspace, project, root, [(source, 'old.py')])
    source.write_bytes(b'new work')
    with pytest.raises(FileExistsError):
        restore_archive(root / 'archive_receipt.json')
    assert source.read_bytes() == b'new work'
    assert (root / 'old.py').read_bytes() == b'original'


def test_restore_rejects_rebound_project_scope(tmp_path):
    import json
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    source = workspace / 'old.py'
    source.write_bytes(b'original')
    root = project / '_local/archive/test'
    archive_paths(workspace, project, root, [(source, 'old.py')])
    receipt = root / 'archive_receipt.json'
    value = json.loads(receipt.read_bytes())
    value['project'] = str(workspace / 'different-project')
    receipt.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='relationship'):
        restore_archive(receipt)
    assert not source.exists()


def test_source_change_is_refused(tmp_path, monkeypatch):
    import scripts.maintenance.organize_local_workspace as organizer
    workspace = tmp_path / 'workspace'
    project = workspace / 'MemUpdateBench'
    project.mkdir(parents=True)
    source = workspace / 'source.py'
    source.write_bytes(b'original')
    original = organizer.tree_fingerprint
    calls = 0
    def changing(path):
        nonlocal calls
        result = original(path)
        calls += 1
        if calls == 1:
            source.write_bytes(b'changed')
        return result
    monkeypatch.setattr(organizer, 'tree_fingerprint', changing)
    with pytest.raises(ValueError, match='changed'):
        archive_paths(workspace, project, project / '_local/archive/test', [(source, 'source.py')])
    assert source.exists()
