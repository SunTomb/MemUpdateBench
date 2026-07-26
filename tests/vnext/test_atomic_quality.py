from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def _crash_prepared(output_dir: Path, *, overwrite: bool) -> dict[str, object]:
    if overwrite:
        (output_dir / "first.json").write_bytes(b"O1")
        (output_dir / "second.json").write_bytes(b"O2")
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
atomic._transaction_fault_point=lambda point: os._exit(81) if point == 'publish:0' else None
out=Path(sys.argv[1]); overwrite=sys.argv[2]=='true'
atomic.publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'},overwrite=overwrite)
'''
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output_dir), str(overwrite).lower()],
        cwd=PROJECT_ROOT,
        env=_env(),
        check=False,
    )
    assert crashed.returncode == 81
    return json.loads(
        (output_dir / ".mub-vnext-transaction.json").read_text(encoding="utf-8")
    )


def _foreign(path: Path, kind: str, root: Path) -> None:
    path.unlink(missing_ok=True)
    source = root / f"foreign-{kind}.bin"
    source.write_bytes(b"EVIL")
    if kind == "regular":
        path.write_bytes(b"EVIL")
    elif kind == "hardlink":
        os.link(source, path)
    elif kind == "symlink":
        try:
            path.symlink_to(source)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("overwrite", (False, True))
@pytest.mark.parametrize("position", (0, 1))
@pytest.mark.parametrize("kind", ("regular", "hardlink", "symlink"))
def test_prepared_recovery_rejects_foreign_destination(
    tmp_path: Path, overwrite: bool, position: int, kind: str
) -> None:
    output_dir = tmp_path / f"foreign-{overwrite}-{position}-{kind}"
    output_dir.mkdir()
    journal = _crash_prepared(output_dir, overwrite=overwrite)
    destination = output_dir / journal["entries"][position]["destination"]
    _foreign(destination, kind, tmp_path)
    marker = tmp_path / f"callback-{overwrite}-{position}-{kind}"
    script = r'''
import sys
from pathlib import Path
from mub.vnext.io.atomic import publish_files_atomically
out=Path(sys.argv[1]); marker=Path(sys.argv[2])
publish_files_atomically({out/'first.json':b'R1',out/'second.json':b'R2'},overwrite=True,pre_publish=lambda:marker.write_text('called'))
'''
    recovered = subprocess.run(
        [sys.executable, "-c", script, str(output_dir), str(marker)],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert recovered.returncode != 0
    assert not marker.exists()
    assert (output_dir / ".mub-vnext-transaction.json").exists()
    assert destination.exists() or destination.is_symlink()


@pytest.mark.parametrize("overwrite", (False, True))
@pytest.mark.parametrize("position", (0, 1))
def test_pre_publish_stage_mutation_never_reaches_destinations(
    tmp_path: Path, overwrite: bool, position: int
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    if overwrite:
        first.write_bytes(b"O1")
        second.write_bytes(b"O2")

    def mutate() -> None:
        stages = sorted(tmp_path.glob("*.tmp.*"))
        stages[position].write_bytes(b"EVIL")

    with pytest.raises(RuntimeError, match="stage|content|transaction"):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"},
            overwrite=overwrite,
            pre_publish=mutate,
        )
    if overwrite:
        assert (first.read_bytes(), second.read_bytes()) == (b"O1", b"O2")
    else:
        assert not first.exists() and not second.exists()


@pytest.mark.parametrize("overwrite", (False, True))
def test_publish_time_stage_mutation_rolls_back_coherently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    if overwrite:
        first.write_bytes(b"O1")
        second.write_bytes(b"O2")
    real_replace, real_link = atomic.os.replace, atomic.os.link
    changed = False

    def replace(source, destination):
        nonlocal changed
        if ".tmp." in Path(source).name and not changed:
            Path(source).write_bytes(b"EVIL")
            changed = True
        return real_replace(source, destination)

    def link(source, destination, *args, **kwargs):
        nonlocal changed
        if not changed:
            Path(source).write_bytes(b"EVIL")
            changed = True
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.os, "link", link)
    with pytest.raises(RuntimeError):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"}, overwrite=overwrite
        )
    if overwrite:
        assert (first.read_bytes(), second.read_bytes()) == (b"O1", b"O2")
    else:
        assert not first.exists() and not second.exists()


def test_directory_fsync_propagates_real_error_and_tolerates_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mub.vnext.io.atomic as atomic

    descriptor = os.open(tmp_path / "descriptor.bin", os.O_CREAT | os.O_RDWR)
    monkeypatch.setattr(atomic.os, "open", lambda *args, **kwargs: os.dup(descriptor))
    monkeypatch.setattr(
        atomic.os,
        "fsync",
        lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "injected")),
    )
    with pytest.raises(OSError) as exc_info:
        atomic._fsync_directory(tmp_path)
    assert exc_info.value.errno == errno.EIO

    monkeypatch.setattr(
        atomic.os,
        "fsync",
        lambda fd: (_ for _ in ()).throw(OSError(errno.EINVAL, "unsupported")),
    )
    atomic._fsync_directory(tmp_path)
    os.close(descriptor)


@pytest.mark.parametrize("committed", (False, True))
def test_publication_reports_directory_fsync_failure_and_stays_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_bytes(b"O1")
    second.write_bytes(b"O2")
    real_sync = atomic._fsync_directory
    failed = False

    def failing(directory: Path) -> None:
        nonlocal failed
        journal = directory / ".mub-vnext-transaction.json"
        state = None
        if journal.exists():
            state = json.loads(journal.read_text(encoding="utf-8"))["state"]
        if not failed and ((committed and state == "committed") or (not committed and state == "prepared")):
            failed = True
            raise OSError(errno.EIO, "injected directory fsync failure")
        real_sync(directory)

    monkeypatch.setattr(atomic, "_fsync_directory", failing)
    with pytest.raises(OSError, match="directory fsync"):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"}, overwrite=True
        )
    assert failed
    if committed:
        assert (tmp_path / ".mub-vnext-transaction.json").exists()
        assert (first.read_bytes(), second.read_bytes()) == (b"N1", b"N2")


@pytest.mark.parametrize("phase", ("stage", "journal", "backup", "publish", "commit", "cleanup"))
def test_directory_fsync_failure_at_each_transaction_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_bytes(b"O1"); second.write_bytes(b"O2")
    real_sync = atomic._fsync_directory
    failed = False

    def classify(directory: Path) -> str:
        journal = directory / ".mub-vnext-transaction.json"
        if not journal.exists():
            return "stage"
        payload = json.loads(journal.read_text(encoding="utf-8"))
        if payload["state"] == "committed":
            backups = list(directory.glob("*.bak.*"))
            return "commit" if len(backups) == 2 else "cleanup"
        backups = list(directory.glob("*.bak.*"))
        published = any(
            path.exists() and path.read_bytes() in {b"N1", b"N2"}
            for path in (first, second)
        )
        if published:
            return "publish"
        if backups:
            return "backup"
        return "journal"

    def failing(directory: Path) -> None:
        nonlocal failed
        if not failed and classify(directory) == phase:
            failed = True
            raise OSError(errno.EIO, f"injected {phase} directory fsync failure")
        real_sync(directory)

    monkeypatch.setattr(atomic, "_fsync_directory", failing)
    with pytest.raises(OSError, match=phase):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"}, overwrite=True
        )
    assert failed
    if phase in {"commit", "cleanup"}:
        assert (first.read_bytes(), second.read_bytes()) == (b"N1", b"N2")
        assert (tmp_path / ".mub-vnext-transaction.json").exists()
    else:
        assert (first.read_bytes(), second.read_bytes()) == (b"O1", b"O2")




_RESERVED_TRANSACTION_NAMES = (
    ".mub-vnext-transaction.json",
    ".mub-vnext-transaction.json.new",
    ".mub-vnext-transaction.committed.json",
    ".mub-vnext-transaction.committed.witness.json",
)


@pytest.mark.parametrize("name", _RESERVED_TRANSACTION_NAMES)
@pytest.mark.parametrize("kind", ("symlink", "junction"))
def test_broken_reserved_reparse_object_blocks_publication(
    tmp_path: Path, name: str, kind: str
) -> None:
    output_dir = tmp_path / f"broken-{kind}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    output_dir.mkdir()
    reserved = output_dir / name
    missing = tmp_path / f"missing-{kind}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    if kind == "symlink":
        try:
            reserved.symlink_to(missing, target_is_directory=False)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
    else:
        if os.name != "nt":
            pytest.skip("Windows junction coverage")
        target = tmp_path / f"junction-target-{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        target.mkdir()
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(reserved), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip("junction creation unavailable")
        target.rmdir()
    assert os.path.lexists(reserved)
    marker = tmp_path / f"callback-{kind}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    script = r'''
import sys
from pathlib import Path
from mub.vnext.io.atomic import publish_files_atomically
out=Path(sys.argv[1]); marker=Path(sys.argv[2])
publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'},overwrite=True,pre_publish=lambda:marker.write_text('called'))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(output_dir), str(marker)],
        cwd=PROJECT_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert os.path.lexists(reserved)
    assert not (output_dir / "first.json").exists()
    assert not (output_dir / "second.json").exists()


@pytest.mark.parametrize("overwrite", (False, True))
@pytest.mark.parametrize("double_failure", (False, True))
def test_reconstructed_witness_directory_durability_is_reported_precisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    double_failure: bool,
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    if overwrite:
        first.write_bytes(b"O1"); second.write_bytes(b"O2")
    real_sync = atomic._fsync_directory
    failures = 0

    def failing(directory: Path) -> None:
        nonlocal failures
        active = directory / ".mub-vnext-transaction.json"
        evidence = list(directory.glob(".mub-vnext-transaction.committed*"))
        committed = (
            first.exists()
            and second.exists()
            and first.read_bytes() == b"N1"
            and second.read_bytes() == b"N2"
        )
        if failures == 0 and committed and not active.exists() and not evidence:
            failures = 1
            raise OSError(errno.EIO, "final retirement barrier")
        if double_failure and failures == 1 and evidence:
            failures = 2
            raise OSError(errno.EIO, "reconstructed witness barrier")
        real_sync(directory)

    monkeypatch.setattr(atomic, "_fsync_directory", failing)
    pattern = "durability unconfirmed" if double_failure else "evidence retained"
    with pytest.raises(RuntimeError, match=pattern):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"}, overwrite=overwrite
        )
    assert failures == (2 if double_failure else 1)
    assert (first.read_bytes(), second.read_bytes()) == (b"N1", b"N2")
    assert list(tmp_path.glob(".mub-vnext-transaction.committed*"))
    atomic.publish_files_atomically(
        {first: b"R1", second: b"R2"}, overwrite=True
    )
    assert (first.read_bytes(), second.read_bytes()) == (b"R1", b"R2")
    assert not list(tmp_path.glob(".mub-vnext-transaction*"))


@pytest.mark.parametrize("overwrite", (False, True))
def test_committed_retirement_fsync_failure_retains_truthful_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    import mub.vnext.io.atomic as atomic

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    if overwrite:
        first.write_bytes(b"O1"); second.write_bytes(b"O2")
    real_sync = atomic._fsync_directory
    failed = False

    def failing(directory: Path) -> None:
        nonlocal failed
        active = directory / ".mub-vnext-transaction.json"
        tombstones = list(directory.glob(".mub-vnext-transaction.committed*"))
        if not failed and not active.exists() and tombstones:
            failed = True
            raise OSError(errno.EIO, "retirement barrier failure")
        real_sync(directory)

    monkeypatch.setattr(atomic, "_fsync_directory", failing)
    with pytest.raises(RuntimeError, match="committed.*retirement|retirement.*committed"):
        atomic.publish_files_atomically(
            {first: b"N1", second: b"N2"}, overwrite=overwrite
        )
    assert failed
    assert (first.read_bytes(), second.read_bytes()) == (b"N1", b"N2")
    assert list(tmp_path.glob(".mub-vnext-transaction.committed*"))


@pytest.mark.parametrize("overwrite", (False, True))
@pytest.mark.parametrize("stage", ("retirement_renamed", "retirement_fsynced", "retirement_unlinked", "retirement_final_fsynced"))
def test_committed_retirement_crash_is_reconciled_before_next_publish(
    tmp_path: Path, overwrite: bool, stage: str
) -> None:
    output_dir = tmp_path / f"retirement-{overwrite}-{stage}"
    output_dir.mkdir()
    if overwrite:
        (output_dir / "first.json").write_bytes(b"O1")
        (output_dir / "second.json").write_bytes(b"O2")
    script = r'''
import os, sys
from pathlib import Path
import mub.vnext.io.atomic as atomic
stage=sys.argv[2]; atomic._transaction_fault_point=lambda point: os._exit(82) if point==stage else None
out=Path(sys.argv[1]); overwrite=sys.argv[3]=='true'
atomic.publish_files_atomically({out/'first.json':b'N1',out/'second.json':b'N2'},overwrite=overwrite)
'''
    crashed = subprocess.run([sys.executable, "-c", script, str(output_dir), stage, str(overwrite).lower()], cwd=PROJECT_ROOT, env=_env(), check=False)
    assert crashed.returncode == 82
    marker = tmp_path / f"retirement-observed-{overwrite}-{stage}"
    recovery = r'''
import sys
from pathlib import Path
from mub.vnext.io.atomic import publish_files_atomically
out=Path(sys.argv[1]); marker=Path(sys.argv[2])
def inspect():
    pair=((out/'first.json').read_bytes(),(out/'second.json').read_bytes())
    if pair != (b'N1',b'N2'): raise RuntimeError(repr(pair))
    marker.write_text('committed',encoding='utf-8')
publish_files_atomically({out/'first.json':b'R1',out/'second.json':b'R2'},overwrite=True,pre_publish=inspect)
'''
    recovered = subprocess.run([sys.executable, "-c", recovery, str(output_dir), str(marker)], cwd=PROJECT_ROOT, env=_env(), capture_output=True, text=True, check=False)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert marker.read_text(encoding="utf-8") == "committed"
    assert ((output_dir / "first.json").read_bytes(), (output_dir / "second.json").read_bytes()) == (b"R1", b"R2")
    assert not list(output_dir.glob(".mub-vnext-transaction*"))
