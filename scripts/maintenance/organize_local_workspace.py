"""Inventory-preserving local archival; never delete contents or follow links."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


SCHEMA = 'memupdatebench.workspace-archive.v1'


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def no_links(path):
    path = Path(path).absolute()
    require('..' not in path.parts, 'path traversal is forbidden')
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'symlink or reparse component is forbidden')
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_fingerprint(path):
    path = no_links(path)
    require(path.exists(), 'source missing')
    require(path.name != '.git', 'Git metadata is protected')
    rows = []
    if path.is_file():
        rows.append({'path': '.', 'kind': 'file', 'bytes': path.stat().st_size, 'sha256': digest_file(path)})
    else:
        require(path.is_dir(), 'special files are unsupported')
        for parent, dirs, files in os.walk(path, followlinks=False):
            for name in sorted(dirs + files):
                child = no_links(Path(parent) / name)
                require(name != '.git', 'Git checkout/worktree is protected')
                relative = child.relative_to(path).as_posix()
                if child.is_dir():
                    rows.append({'path': relative, 'kind': 'directory'})
                else:
                    require(child.is_file(), 'special files are unsupported')
                    rows.append({'path': relative, 'kind': 'file', 'bytes': child.stat().st_size,
                                 'sha256': digest_file(child)})
    rows.sort(key=lambda row: row['path'])
    return {'kind': 'directory' if path.is_dir() else 'file', 'members': rows,
            'file_count': sum(row['kind'] == 'file' for row in rows),
            'bytes': sum(row.get('bytes', 0) for row in rows),
            'sha256': hashlib.sha256(canonical(rows)).hexdigest()}


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _guard_source(workspace, project, archive_root, source):
    require(workspace in source.parents, 'source is outside the workspace')
    protected = [workspace / name for name in ('G-MSRA', 'MemUpdateBench_releases', 'MemUpdateBench_qualification_inputs')]
    protected += [project / name for name in ('data', 'results', 'checkpoints', '_local', '.git')]
    require(not any(_overlap(source, target) for target in protected), 'protected source cannot be archived')
    require(not _overlap(source, archive_root), 'archive overlaps source')
    require('.git' not in source.relative_to(workspace).parts, 'Git metadata is protected')
    for parts in (('data', 'vnext'), ('results', 'vnext')):
        relative = source.relative_to(workspace).parts
        require(not any(relative[i:i + len(parts)] == parts for i in range(len(relative))),
                'protected evidence root cannot be archived')
    if source.is_dir():
        require(not (source / '.git').exists(), 'Git checkout/worktree is protected')


def _write_new(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _journal(path, value):
    with path.open('ab') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def archive_paths(workspace_root, project_root, archive_root, moves):
    workspace, project, archive = map(no_links, (workspace_root, project_root, archive_root))
    require(workspace.is_dir() and project.is_dir() and project.parent == workspace,
            'workspace/project relationship mismatch')
    require(project / '_local/archive' in archive.parents, 'archive must be inside project/_local/archive')
    if archive.exists():
        raise FileExistsError(archive)
    planned = []
    for supplied, destination in moves:
        source = no_links(supplied)
        relative = PurePosixPath(destination)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == destination,
                'unsafe archive destination')
        _guard_source(workspace, project, archive, source)
        planned.append({'source': str(source), 'destination': destination, 'before': tree_fingerprint(source)})
    require(planned and len({row['source'] for row in planned}) == len(planned)
            and len({row['destination'] for row in planned}) == len(planned), 'empty or duplicate move plan')
    for index, row in enumerate(planned):
        for other in planned[index + 1:]:
            require(not _overlap(Path(row['source']), Path(other['source'])), 'nested sources in move plan')
            require(not _overlap(Path(row['destination']), Path(other['destination'])), 'nested archive destinations')
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.mkdir(mode=0o700)
    journal = archive / 'move_journal.jsonl'
    _write_new(archive / 'archive_plan.json', {'schema': SCHEMA, 'workspace': str(workspace),
        'project': str(project), 'archive': str(archive), 'moves': planned, 'policy': 'move_whole_roots_no_content_deletion'})
    completed = []
    for row in planned:
        source, target = no_links(row['source']), no_links(archive / row['destination'])
        require(tree_fingerprint(source) == row['before'], 'source changed after inventory')
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        _journal(journal, {'event': 'MOVE_INTENT', 'source': row['source'], 'destination': row['destination']})
        os.rename(source, target)
        after = tree_fingerprint(target)
        require(after == row['before'], 'archive bytes changed during move')
        completed.append(row)
        _journal(journal, {'event': 'MOVED_VERIFIED', 'source': row['source'],
                           'destination': row['destination'], 'sha256': after['sha256']})
    receipt = {'schema': SCHEMA, 'status': 'ARCHIVED', 'workspace': str(workspace), 'project': str(project),
               'archive': str(archive), 'moves': completed, 'moved_roots': len(completed),
               'file_count': sum(row['before']['file_count'] for row in completed),
               'bytes': sum(row['before']['bytes'] for row in completed),
               'deleted_files': 0, 'git_worktrees_removed': 0}
    _write_new(archive / 'archive_receipt.json', receipt)
    return receipt


def restore_archive(receipt_path):
    receipt_path = no_links(receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'ARCHIVED', 'invalid archive receipt')
    archive = no_links(receipt['archive'])
    require(receipt_path.parent == archive, 'receipt is not in its archive')
    workspace, project = no_links(receipt['workspace']), no_links(receipt['project'])
    require(project.parent == workspace and project.is_dir()
            and project / '_local/archive' in archive.parents, 'receipt workspace/project/archive relationship mismatch')
    pending = []
    for row in receipt['moves']:
        source = no_links(row['source'])
        target = no_links(archive / row['destination'])
        require(archive in target.parents, 'restore target escapes archive')
        _guard_source(workspace, project, archive, source)
        if source.exists():
            raise FileExistsError(source)
        require(tree_fingerprint(target) == row['before'], 'archived bytes changed; restore refused')
        pending.append((source, target, row))
    for source, target, row in reversed(pending):
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            raise FileExistsError(source)
        os.rename(target, source)
        require(tree_fingerprint(source) == row['before'], 'restored bytes changed')
        _journal(archive / 'restore_journal.jsonl', {'event': 'RESTORED_VERIFIED', 'source': str(source),
                                                   'sha256': row['before']['sha256']})
    return {'status': 'RESTORED', 'roots': len(pending)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    apply = sub.add_parser('archive')
    apply.add_argument('--plan', type=Path, required=True)
    restore = sub.add_parser('restore')
    restore.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'archive':
        plan = json.loads(args.plan.read_bytes())
        result = archive_paths(plan['workspace'], plan['project'], plan['archive'],
                               [(row['source'], row['destination']) for row in plan['moves']])
        result = {key: value for key, value in result.items() if key != 'moves'}
    else:
        result = restore_archive(args.receipt)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
