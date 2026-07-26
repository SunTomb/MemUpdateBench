from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

StageValidator = Callable[[Path], None]
FileIdentity = tuple[int, int]
_JOURNAL_NAME = ".mub-vnext-transaction.json"
_JOURNAL_TEMP_NAME = ".mub-vnext-transaction.json.new"
_RETIREMENT_NAME = ".mub-vnext-transaction.committed.json"
_RETIREMENT_WITNESS_NAME = ".mub-vnext-transaction.committed.witness.json"
_LOCK_NAME = ".mub-vnext-publish.lock"
_TRANSACTION_VERSION = 1


class _StageIntegrityError(RuntimeError):
    pass


class _CommittedRetirementError(RuntimeError):
    pass


def _transaction_fault_point(stage: str) -> None:
    del stage


def publish_files_atomically(payloads: Mapping[Path, bytes], *, overwrite: bool, source_paths: Sequence[Path] = (), validators: Mapping[Path, StageValidator] | None = None, pre_publish: Callable[[], None] | None = None) -> None:
    if not payloads:
        raise ValueError("atomic publication requires at least one output")
    destinations = tuple(payloads)
    if len(destinations) != len(set(destinations)):
        raise ValueError("atomic publication destinations must be unique")
    parents = {path.parent for path in destinations}
    if len(parents) != 1:
        raise ValueError("transactional outputs must share one directory")
    output_dir = next(iter(parents))
    _validate_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _directory_lock(
        output_dir,
        protected_paths=(
            *source_paths,
            *destinations,
            output_dir / _JOURNAL_NAME,
            output_dir / _JOURNAL_TEMP_NAME,
            output_dir / _RETIREMENT_NAME,
            output_dir / _RETIREMENT_WITNESS_NAME,
        ),
    ):
        _recover_pending_transaction(output_dir)
        _validate_output_directory(output_dir)
        _validate_destinations(destinations, output_dir=output_dir, overwrite=overwrite, source_paths=source_paths)
        _publish_locked(payloads, destinations=destinations, output_dir=output_dir, overwrite=overwrite, source_paths=source_paths, validators=validators or {}, pre_publish=pre_publish)


def _publish_locked(payloads: Mapping[Path, bytes], *, destinations: tuple[Path, ...], output_dir: Path, overwrite: bool, source_paths: Sequence[Path], validators: Mapping[Path, StageValidator], pre_publish: Callable[[], None] | None) -> None:
    invocation = uuid.uuid4().hex
    staged = {destination: destination.with_name(f"{destination.name}.tmp.{invocation}") for destination in destinations}
    backups = {destination: destination.with_name(f"{destination.name}.bak.{invocation}") for destination in destinations}
    journal_path = output_dir / _JOURNAL_NAME
    journal_written = False
    published_effects: set[int] = set()
    try:
        entries: list[dict[str, object]] = []
        for destination, temporary in staged.items():
            backup = backups[destination]
            if _path_present(temporary) or _path_present(backup):
                raise FileExistsError("atomic staging collision")
            old_present = _path_present(destination)
            old_identity = list(_file_identity(destination)) if old_present else None
            old_sha256 = _sha256_file(destination) if old_present else None
            with temporary.open("xb") as handle:
                handle.write(payloads[destination]); handle.flush(); os.fsync(handle.fileno())
            entries.append({"destination": destination.name, "temporary": temporary.name, "backup": backup.name, "new_identity": list(_file_identity(temporary)), "new_sha256": _sha256_file(temporary), "old_present": old_present, "old_identity": old_identity, "old_sha256": old_sha256})
        _fsync_directory(output_dir)
        journal: dict[str, object] = {
            "version": _TRANSACTION_VERSION,
            "state": "prepared",
            "overwrite": overwrite,
            "protected_source_identities": [
                list(_file_identity(path)) for path in source_paths if _path_present(path)
            ],
            "entries": entries,
        }
        for destination, temporary in staged.items():
            validator = validators.get(destination)
            if validator is not None:
                validator(temporary)
        _verify_staged_entries(output_dir, journal)
        if pre_publish is not None:
            pre_publish()
        _verify_staged_entries(output_dir, journal)
        _validate_destinations(destinations, output_dir=output_dir, overwrite=overwrite, source_paths=source_paths)
        _write_journal(output_dir, journal)
        journal_written = True
        _transaction_fault_point("journal_prepared")
        if overwrite:
            for index, destination in enumerate(destinations):
                if _path_present(destination):
                    os.replace(destination, backups[destination]); _fsync_directory(output_dir)
                _transaction_fault_point(f"backup:{index}")
            for index, destination in enumerate(destinations):
                _verify_staged_entries(
                    output_dir, journal, remaining_indices=range(index, len(destinations))
                )
                os.replace(staged[destination], destination)
                published_effects.add(index)
                _fsync_directory(output_dir)
                entry = entries[index]
                if not _owned_matches(
                    destination,
                    _journal_identity(entry["new_identity"]),
                    _journal_hash(entry["new_sha256"]),
                ):
                    raise _StageIntegrityError("published stage content changed")
                _transaction_fault_point(f"publish:{index}")
        else:
            for index, destination in enumerate(destinations):
                _verify_staged_entries(
                    output_dir, journal, remaining_indices=range(index, len(destinations))
                )
                os.link(staged[destination], destination)
                published_effects.add(index)
                _fsync_directory(output_dir)
                entry = entries[index]
                if not _owned_matches(
                    destination,
                    _journal_identity(entry["new_identity"]),
                    _journal_hash(entry["new_sha256"]),
                ):
                    raise _StageIntegrityError("published stage content changed")
                _transaction_fault_point(f"publish:{index}")
        journal["state"] = "committed"
        _write_journal(output_dir, journal, replace=True)
        _transaction_fault_point("commit_marked")
        _finish_committed_transaction(output_dir, journal, fault_points=True)
    except _CommittedRetirementError:
        raise
    except _StageIntegrityError as publication_error:
        if journal_written or _path_present(journal_path):
            try:
                recovery_journal = _load_journal(journal_path, output_dir)
                _rollback_active_integrity_failure(
                    output_dir, recovery_journal, published_effects=published_effects
                )
            except BaseException:
                raise RuntimeError(
                    "stage integrity rollback failed; recovery artifacts preserved"
                ) from publication_error
        else:
            for entry in entries:
                temporary = output_dir / str(entry["temporary"])
                identity = _journal_identity(entry["new_identity"])
                if _identity_matches(temporary, identity):
                    temporary.unlink()
        raise RuntimeError("staged transaction integrity check failed") from publication_error
    except BaseException as publication_error:
        if (
            isinstance(publication_error, FileExistsError)
            and not overwrite
            and _path_present(journal_path)
        ):
            try:
                pending = _load_journal(journal_path, output_dir)
                _rollback_active_no_clobber_race(output_dir, pending)
            except BaseException:
                try:
                    pending["preserve_staged"] = True
                    _write_journal(output_dir, pending, replace=True)
                except BaseException:
                    pass
                raise RuntimeError(
                    "atomic rollback failed; recovery artifacts preserved"
                ) from publication_error
            raise
        if journal_written or _path_present(journal_path):
            try:
                pending = _load_journal(journal_path, output_dir)
                if pending["state"] == "committed":
                    raise publication_error
                _recover_pending_transaction(output_dir)
            except BaseException as recovery_error:
                if recovery_error is publication_error:
                    raise
                try:
                    recovery_journal = _load_journal(journal_path, output_dir)
                    recovery_journal["preserve_staged"] = True
                    _write_journal(output_dir, recovery_journal, replace=True)
                except BaseException:
                    pass
                raise RuntimeError("atomic rollback failed; recovery artifacts preserved") from publication_error
        else:
            _cleanup_temporary_paths(tuple(staged.values()))
        raise


def _write_journal(output_dir: Path, journal: dict[str, object], *, replace: bool = False) -> None:
    path, temporary = output_dir / _JOURNAL_NAME, output_dir / _JOURNAL_TEMP_NAME
    if _is_reparse_point(path) or _is_reparse_point(temporary):
        raise ValueError("transaction journal may not be a reparse point")
    if _path_present(temporary):
        if not stat.S_ISREG(temporary.stat(follow_symlinks=False).st_mode):
            raise ValueError("transaction journal temporary is not a regular file")
        temporary.unlink()
    payload = json.dumps(journal, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    if not replace:
        _transaction_fault_point("journal_new_fsynced")
    if _path_present(path) and not replace:
        raise FileExistsError("pending transaction journal already exists")
    os.replace(temporary, path); _fsync_directory(output_dir)


def _recover_pending_transaction(output_dir: Path) -> None:
    path, temporary = output_dir / _JOURNAL_NAME, output_dir / _JOURNAL_TEMP_NAME
    retirement_paths = (
        output_dir / _RETIREMENT_NAME,
        output_dir / _RETIREMENT_WITNESS_NAME,
    )
    if any(_path_present(item) for item in retirement_paths):
        if _path_present(path) or _path_present(temporary):
            raise RuntimeError("active and retired transaction evidence conflict")
        journal = _load_retirement_journal(output_dir)
        _finish_committed_transaction(
            output_dir, journal, fault_points=False, retire=False
        )
        _cleanup_retirement_evidence(output_dir, journal)
        return
    if _path_present(temporary):
        if _is_reparse_point(temporary) or not stat.S_ISREG(temporary.stat(follow_symlinks=False).st_mode):
            raise ValueError("unexpected transaction journal temporary")
        if _path_present(path):
            temporary.unlink(); _fsync_directory(output_dir)
        else:
            journal = _load_journal(temporary, output_dir)
            if journal["state"] != "prepared":
                raise RuntimeError("ambiguous pre-journal transaction state")
            _discard_prejournal_transaction(output_dir, journal)
            return
    if not _path_present(path):
        return
    journal = _load_journal(path, output_dir)
    if journal["state"] == "committed":
        _finish_committed_transaction(output_dir, journal, fault_points=False)
    else:
        _rollback_prepared_transaction(output_dir, journal)


def _load_journal(path: Path, output_dir: Path) -> dict[str, object]:
    if _is_reparse_point(path) or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError("transaction journal is not a regular file")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid transaction journal") from exc
    if type(payload) is not dict or payload.get("version") != _TRANSACTION_VERSION or payload.get("state") not in {"prepared", "committed"} or type(payload.get("overwrite")) is not bool or type(payload.get("entries")) is not list or not payload["entries"]:
        raise ValueError("unsupported transaction journal")
    protected_sources = payload.get("protected_source_identities")
    if type(protected_sources) is not list:
        raise ValueError("invalid transaction protected-source identities")
    for identity in protected_sources:
        _journal_identity(identity)
    if "preserve_staged" in payload and type(payload["preserve_staged"]) is not bool:
        raise ValueError("invalid transaction recovery policy")
    seen: set[str] = set(); invocation: str | None = None
    for entry in payload["entries"]:
        if type(entry) is not dict:
            raise ValueError("invalid transaction journal entry")
        destination = _journal_member(output_dir, entry.get("destination"), "destination")
        temporary = _journal_member(output_dir, entry.get("temporary"), "temporary")
        backup = _journal_member(output_dir, entry.get("backup"), "backup")
        if destination.name in seen:
            raise ValueError("duplicate transaction destination")
        seen.add(destination.name)
        prefix = f"{destination.name}.tmp."
        if not temporary.name.startswith(prefix):
            raise ValueError("invalid transaction temporary name")
        current = temporary.name[len(prefix):]
        if len(current) != 32 or any(c not in "0123456789abcdef" for c in current):
            raise ValueError("invalid transaction identifier")
        if invocation is None: invocation = current
        if invocation != current or backup.name != f"{destination.name}.bak.{current}":
            raise ValueError("inconsistent transaction member names")
        _journal_identity(entry.get("new_identity")); _journal_hash(entry.get("new_sha256"))
        if entry.get("old_present") is True:
            _journal_identity(entry.get("old_identity")); _journal_hash(entry.get("old_sha256"))
        elif entry.get("old_present") is not False or entry.get("old_identity") is not None or entry.get("old_sha256") is not None:
            raise ValueError("invalid original transaction identity")
    return payload


def _journal_member(output_dir: Path, value: object, label: str) -> Path:
    if type(value) is not str or not value or Path(value).name != value:
        raise ValueError(f"invalid transaction {label} name")
    path = output_dir / value
    if _is_reparse_point(path):
        raise ValueError(f"transaction {label} may not be a reparse point")
    return path


def _journal_identity(value: object) -> FileIdentity:
    if type(value) is not list or len(value) != 2 or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("invalid transaction file identity")
    return value[0], value[1]


def _verify_staged_entries(
    output_dir: Path,
    journal: dict[str, object],
    *,
    remaining_indices: Sequence[int] | None = None,
) -> None:
    entries = journal["entries"]
    assert isinstance(entries, list)
    selected = set(range(len(entries))) if remaining_indices is None else set(remaining_indices)
    protected_sources = {
        _journal_identity(identity)
        for identity in journal["protected_source_identities"]
    }
    selected_paths: list[Path] = []
    for index, entry in enumerate(entries):
        if index not in selected:
            continue
        assert isinstance(entry, dict)
        temporary = output_dir / str(entry["temporary"])
        identity = _journal_identity(entry["new_identity"])
        digest = _journal_hash(entry["new_sha256"])
        if not _path_present(temporary):
            raise _StageIntegrityError("staged transaction file is missing")
        result = temporary.stat(follow_symlinks=False)
        if (
            _is_reparse_point(temporary)
            or not stat.S_ISREG(result.st_mode)
            or getattr(result, "st_nlink", 1) != 1
            or identity in protected_sources
            or not _owned_matches(temporary, identity, digest)
        ):
            raise _StageIntegrityError("staged transaction identity or content changed")
        destination = output_dir / str(entry["destination"])
        backup = output_dir / str(entry["backup"])
        for member in (destination, backup, output_dir / _JOURNAL_NAME, output_dir / _JOURNAL_TEMP_NAME):
            if _path_present(member) and _paths_alias(temporary, member):
                raise _StageIntegrityError("staged transaction file aliases another member")
        selected_paths.append(temporary)
    for index, left in enumerate(selected_paths):
        for right in selected_paths[index + 1 :]:
            if _paths_alias(left, right):
                raise _StageIntegrityError("staged transaction files alias each other")


def _journal_hash(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid transaction file hash")
    return value


def _discard_prejournal_transaction(output_dir: Path, journal: dict[str, object]) -> None:
    entries = journal["entries"]
    assert isinstance(entries, list)
    intent = output_dir / _JOURNAL_TEMP_NAME
    final_journal = output_dir / _JOURNAL_NAME
    if _path_present(final_journal):
        raise RuntimeError("pre-journal recovery found an installed journal")
    intent_stat = intent.stat(follow_symlinks=False)
    if (
        _is_reparse_point(intent)
        or not stat.S_ISREG(intent_stat.st_mode)
        or getattr(intent_stat, "st_nlink", 1) != 1
    ):
        raise RuntimeError("pre-journal intent object is unsafe")
    protected_sources = {
        _journal_identity(identity)
        for identity in journal["protected_source_identities"]
    }
    stages: list[tuple[Path, FileIdentity, str]] = []
    protected_members: list[Path] = [intent]
    for entry in entries:
        assert isinstance(entry, dict)
        destination, temporary, backup = (
            output_dir / str(entry[key])
            for key in ("destination", "temporary", "backup")
        )
        new_identity = _journal_identity(entry["new_identity"])
        new_hash = _journal_hash(entry["new_sha256"])
        old_present = entry["old_present"] is True
        if old_present:
            old_identity = _journal_identity(entry["old_identity"])
            old_hash = _journal_hash(entry["old_sha256"])
            if not _owned_matches(destination, old_identity, old_hash):
                raise RuntimeError("pre-journal destination differs from recorded old state")
        elif _path_present(destination):
            raise RuntimeError("pre-journal destination effect is not allowed")
        if _path_present(backup):
            raise RuntimeError("pre-journal backup effect is not allowed")
        if not _path_present(temporary):
            raise RuntimeError("pre-journal stage is missing")
        temporary_stat = temporary.stat(follow_symlinks=False)
        if (
            _is_reparse_point(temporary)
            or not stat.S_ISREG(temporary_stat.st_mode)
            or getattr(temporary_stat, "st_nlink", 1) != 1
            or not _owned_matches(temporary, new_identity, new_hash)
            or new_identity in protected_sources
        ):
            raise RuntimeError("pre-journal stage identity, content, or link state is unsafe")
        stages.append((temporary, new_identity, new_hash))
        protected_members.extend((destination, backup))
    stage_paths = [item[0] for item in stages]
    for index, stage in enumerate(stage_paths):
        for other in (*stage_paths[index + 1 :], *protected_members):
            if _path_present(other) and _paths_alias(stage, other):
                raise RuntimeError("pre-journal transaction members alias")
    for temporary, identity, digest in stages:
        _unlink_owned(temporary, identity, digest)
    intent.unlink()
    _fsync_directory(output_dir)


def _verify_recovery_contents(output_dir: Path, journal: dict[str, object]) -> None:
    entries = journal["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        destination, temporary, backup = (
            output_dir / str(entry[key])
            for key in ("destination", "temporary", "backup")
        )
        new_identity = _journal_identity(entry["new_identity"])
        new_hash = _journal_hash(entry["new_sha256"])
        old_present = entry["old_present"] is True
        old_identity = _journal_identity(entry["old_identity"]) if old_present else None
        old_hash = _journal_hash(entry["old_sha256"]) if old_present else None
        if _path_present(temporary) and not _owned_matches(temporary, new_identity, new_hash):
            raise RuntimeError("staged transaction content mismatch")
        if _path_present(backup) and (
            old_identity is None
            or old_hash is None
            or not _owned_matches(backup, old_identity, old_hash)
        ):
            raise RuntimeError("backup transaction content mismatch")
        if _path_present(destination):
            if _identity_matches(destination, new_identity):
                if not _owned_matches(destination, new_identity, new_hash):
                    raise RuntimeError("published transaction content mismatch")
            elif old_identity is not None and _identity_matches(destination, old_identity):
                if old_hash is None or not _owned_matches(destination, old_identity, old_hash):
                    raise RuntimeError("original transaction content mismatch")
            else:
                raise RuntimeError("unexpected destination blocks transaction recovery")


def _rollback_active_no_clobber_race(
    output_dir: Path, journal: dict[str, object]
) -> None:
    if journal["state"] != "prepared" or journal["overwrite"] is not False:
        raise RuntimeError("invalid no-clobber race recovery state")
    entries = journal["entries"]
    assert isinstance(entries, list)
    for entry in reversed(entries):
        assert isinstance(entry, dict)
        destination = output_dir / str(entry["destination"])
        temporary = output_dir / str(entry["temporary"])
        identity = _journal_identity(entry["new_identity"])
        digest = _journal_hash(entry["new_sha256"])
        if _identity_matches(destination, identity):
            _unlink_owned(destination, identity, digest)
        if _path_present(temporary):
            _unlink_owned(temporary, identity, digest)
    _remove_journal(output_dir)


def _rollback_active_integrity_failure(
    output_dir: Path,
    journal: dict[str, object],
    *,
    published_effects: set[int],
) -> None:
    if journal["state"] != "prepared":
        raise RuntimeError("active integrity rollback requires prepared state")
    entries = journal["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        backup = output_dir / str(entry["backup"])
        if _path_present(backup):
            if entry["old_present"] is not True or not _owned_matches(
                backup,
                _journal_identity(entry["old_identity"]),
                _journal_hash(entry["old_sha256"]),
            ):
                raise RuntimeError("active rollback backup is not authentic")
    for index in reversed(range(len(entries))):
        entry = entries[index]
        assert isinstance(entry, dict)
        destination = output_dir / str(entry["destination"])
        temporary = output_dir / str(entry["temporary"])
        backup = output_dir / str(entry["backup"])
        new_identity = _journal_identity(entry["new_identity"])
        if _identity_matches(destination, new_identity):
            result = destination.stat(follow_symlinks=False)
            if _is_reparse_point(destination) or not stat.S_ISREG(result.st_mode):
                raise RuntimeError("active rollback destination changed type")
            destination.unlink()
        elif _path_present(destination) and index in published_effects:
            result = destination.stat(follow_symlinks=False)
            if _is_reparse_point(destination) or not stat.S_ISREG(result.st_mode):
                raise RuntimeError("published substituted destination changed type")
            if journal["overwrite"] is True:
                if _path_present(temporary):
                    raise RuntimeError("cannot preserve substituted stage evidence")
                os.replace(destination, temporary)
            else:
                if _path_present(backup):
                    raise RuntimeError("cannot preserve substituted no-clobber evidence")
                os.replace(destination, backup)
        elif _path_present(destination):
            old_identity = (
                _journal_identity(entry["old_identity"])
                if entry["old_present"] is True
                else None
            )
            old_hash = (
                _journal_hash(entry["old_sha256"])
                if entry["old_present"] is True
                else None
            )
            if old_identity is None or old_hash is None or not _owned_matches(
                destination, old_identity, old_hash
            ):
                raise RuntimeError("active rollback destination is unexpected")
        if entry["old_present"] is True and not _path_present(destination):
            os.replace(backup, destination)
            if not _owned_matches(
                destination,
                _journal_identity(entry["old_identity"]),
                _journal_hash(entry["old_sha256"]),
            ):
                raise RuntimeError("active rollback restoration failed")
        _fsync_directory(output_dir)


def _rollback_prepared_transaction(output_dir: Path, journal: dict[str, object]) -> None:
    _verify_recovery_contents(output_dir, journal)
    entries = journal["entries"]
    assert isinstance(entries, list)
    for entry in reversed(entries):
        assert isinstance(entry, dict)
        destination, temporary, backup = (output_dir / str(entry[key]) for key in ("destination", "temporary", "backup"))
        new_identity = _journal_identity(entry["new_identity"])
        new_hash = _journal_hash(entry["new_sha256"])
        old_present = entry["old_present"] is True
        old_identity = _journal_identity(entry["old_identity"]) if old_present else None
        old_hash = _journal_hash(entry["old_sha256"]) if old_present else None
        if _identity_matches(destination, new_identity):
            _unlink_owned(destination, new_identity, new_hash)
        elif _path_present(destination) and old_identity is not None and not _identity_matches(destination, old_identity):
            raise RuntimeError("unexpected destination blocks transaction recovery")
        if old_present and not _owned_matches(destination, old_identity, old_hash):
            if not _owned_matches(backup, old_identity, old_hash):
                raise RuntimeError("recoverable backup identity or content mismatch")
            os.replace(backup, destination); _fsync_directory(output_dir)
        if _path_present(temporary) and journal.get("preserve_staged") is not True:
            _unlink_owned(temporary, new_identity, new_hash)
        if _path_present(backup):
            if old_identity is None or old_hash is None:
                raise RuntimeError("unexpected backup blocks transaction recovery")
            _unlink_owned(backup, old_identity, old_hash)
    _remove_journal(output_dir)


def _finish_committed_transaction(
    output_dir: Path,
    journal: dict[str, object],
    *,
    fault_points: bool,
    retire: bool = True,
) -> None:
    _verify_recovery_contents(output_dir, journal)
    entries = journal["entries"]
    assert isinstance(entries, list)
    for index, entry in enumerate(entries):
        assert isinstance(entry, dict)
        destination, temporary, backup = (output_dir / str(entry[key]) for key in ("destination", "temporary", "backup"))
        new_identity = _journal_identity(entry["new_identity"])
        new_hash = _journal_hash(entry["new_sha256"])
        if not _owned_matches(destination, new_identity, new_hash):
            raise RuntimeError("committed destination identity or content mismatch")
        if _path_present(temporary): _unlink_owned(temporary, new_identity, new_hash)
        if _path_present(backup):
            _unlink_owned(
                backup,
                _journal_identity(entry["old_identity"]),
                _journal_hash(entry["old_sha256"]),
            )
        _fsync_directory(output_dir)
        if fault_points: _transaction_fault_point(f"cleanup:{index}")
    if retire:
        _retire_committed_journal(output_dir, journal)


def _journal_bytes(journal: dict[str, object]) -> bytes:
    return json.dumps(
        journal,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _ensure_retirement_evidence(
    output_dir: Path, journal: dict[str, object]
) -> None:
    tombstone = output_dir / _RETIREMENT_NAME
    witness = output_dir / _RETIREMENT_WITNESS_NAME
    payload = _journal_bytes(journal)
    if not _path_present(tombstone) and not _path_present(witness):
        with witness.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    retained = _load_retirement_journal(output_dir)
    if _journal_bytes(retained) != payload:
        raise RuntimeError("committed retirement evidence identifies another transaction")
    _fsync_directory(output_dir)


def _retire_committed_journal(
    output_dir: Path, journal: dict[str, object]
) -> None:
    active = output_dir / _JOURNAL_NAME
    tombstone = output_dir / _RETIREMENT_NAME
    witness = output_dir / _RETIREMENT_WITNESS_NAME
    try:
        if _path_present(tombstone) or _path_present(witness):
            raise RuntimeError("committed retirement evidence already exists")
        os.replace(active, tombstone)
        _transaction_fault_point("retirement_renamed")
        _fsync_directory(output_dir)
        _transaction_fault_point("retirement_fsynced")
        os.link(tombstone, witness)
        _fsync_directory(output_dir)
        tombstone.unlink()
        _transaction_fault_point("retirement_unlinked")
        _fsync_directory(output_dir)
        _transaction_fault_point("retirement_final_fsynced")
        witness.unlink()
        _fsync_directory(output_dir)
    except BaseException as exc:
        try:
            _ensure_retirement_evidence(output_dir, journal)
        except BaseException as evidence_error:
            raise _CommittedRetirementError(
                "committed generation installed; retirement durability failed; "
                "committed-evidence durability unconfirmed; recovery artifacts preserved"
            ) from evidence_error
        raise _CommittedRetirementError(
            "committed generation installed; retirement durability failed; "
            "authenticated committed evidence retained"
        ) from exc


def _load_retirement_journal(output_dir: Path) -> dict[str, object]:
    paths = [
        path
        for path in (
            output_dir / _RETIREMENT_NAME,
            output_dir / _RETIREMENT_WITNESS_NAME,
        )
        if _path_present(path)
    ]
    if not paths:
        raise RuntimeError("committed retirement evidence is missing")
    journals: list[dict[str, object]] = []
    payloads: list[bytes] = []
    for path in paths:
        result = path.stat(follow_symlinks=False)
        if _is_reparse_point(path) or not stat.S_ISREG(result.st_mode):
            raise RuntimeError("committed retirement evidence has unsafe type")
        journal = _load_journal(path, output_dir)
        if journal["state"] != "committed":
            raise RuntimeError("retirement evidence is not committed")
        raw = path.read_bytes()
        if raw != _journal_bytes(journal):
            raise RuntimeError("retirement evidence is not canonical")
        journals.append(journal)
        payloads.append(raw)
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise RuntimeError("committed retirement evidence disagrees")
    journal = journals[0]
    entries = journal["entries"]
    assert isinstance(entries, list)
    for path in paths:
        for entry in entries:
            assert isinstance(entry, dict)
            for key in ("destination", "temporary", "backup"):
                member = output_dir / str(entry[key])
                if _path_present(member) and _paths_alias(path, member):
                    raise RuntimeError("retirement evidence aliases transaction data")
    return journal


def _cleanup_retirement_evidence(
    output_dir: Path, journal: dict[str, object]
) -> None:
    tombstone = output_dir / _RETIREMENT_NAME
    witness = output_dir / _RETIREMENT_WITNESS_NAME
    try:
        if not _path_present(witness):
            if not _path_present(tombstone):
                raise RuntimeError("retirement cleanup lacks committed evidence")
            os.link(tombstone, witness)
            _fsync_directory(output_dir)
        if _path_present(tombstone):
            tombstone.unlink()
            _fsync_directory(output_dir)
        witness.unlink()
        _fsync_directory(output_dir)
    except BaseException as exc:
        try:
            _ensure_retirement_evidence(output_dir, journal)
        except BaseException as evidence_error:
            raise _CommittedRetirementError(
                "committed generation verified; retirement cleanup durability failed; "
                "committed-evidence durability unconfirmed; recovery artifacts preserved"
            ) from evidence_error
        raise _CommittedRetirementError(
            "committed generation verified; retirement cleanup durability failed; "
            "authenticated committed evidence retained"
        ) from exc


def _remove_journal(output_dir: Path) -> None:
    for name in (_JOURNAL_NAME, _JOURNAL_TEMP_NAME):
        path = output_dir / name
        if _path_present(path):
            if _is_reparse_point(path) or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise ValueError("transaction journal changed type")
            path.unlink()
    _fsync_directory(output_dir)


def _owned_matches(path: Path, identity: FileIdentity, expected_hash: str) -> bool:
    return (
        _identity_matches(path, identity)
        and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
        and _sha256_file(path) == expected_hash
    )


def _unlink_owned(path: Path, identity: FileIdentity, expected_hash: str) -> None:
    if _identity_matches(path, identity) and not _owned_matches(path, identity, expected_hash):
        raise RuntimeError("transaction-owned file content mismatch")
    for _ in range(2):
        if not _identity_matches(path, identity): return
        try: path.unlink()
        except OSError: continue
    if _identity_matches(path, identity):
        raise OSError(f"could not remove transaction-owned file: {path.name}")


def _validate_destinations(destinations: tuple[Path, ...], *, output_dir: Path, overwrite: bool, source_paths: Sequence[Path]) -> None:
    reserved = {
        _LOCK_NAME,
        _JOURNAL_NAME,
        _JOURNAL_TEMP_NAME,
        _RETIREMENT_NAME,
        _RETIREMENT_WITNESS_NAME,
    }
    for destination in destinations:
        if destination.name in {"", ".", ".."} or destination.name in reserved or destination.parent != output_dir:
            raise ValueError(f"unsafe output destination: {destination}")
        if _is_reparse_point(destination): raise ValueError(f"output destination may not be a reparse point: {destination}")
        if _path_present(destination) and not stat.S_ISREG(destination.stat().st_mode): raise ValueError(f"output destination must be a regular file: {destination}")
        if _path_present(destination) and not overwrite: raise FileExistsError(f"output destination already exists: {destination}")
        for source in source_paths:
            if _paths_alias(source, destination): raise ValueError(f"source and destination may not identify the same file: {source}")
    for index, left in enumerate(destinations):
        for right in destinations[index + 1:]:
            if _paths_alias(left, right): raise ValueError(f"output destinations may not alias the same file: {left}, {right}")


@contextmanager
def _directory_lock(output_dir: Path, *, protected_paths: Sequence[Path] = ()):
    canonical = os.path.normcase(str(output_dir.resolve(strict=True)))
    lock_name = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lock_path = output_dir.parent / f".mub-vnext-publish-{lock_name}.lock"
    if _is_reparse_point(lock_path):
        raise ValueError("publication lock may not be a reparse point")
    if _path_present(lock_path):
        lock_stat = lock_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(lock_stat.st_mode) or getattr(lock_stat, "st_nlink", 1) != 1:
            raise ValueError("publication lock has unsafe type or link count")
        if any(_path_present(path) and _paths_alias(path, lock_path) for path in protected_paths):
            raise ValueError("publication lock aliases a protected artifact")
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(lock_path, flags)
    handle = None
    try:
        handle = os.fdopen(descriptor, "r+b", closefd=False)
        result = os.fstat(descriptor)
        if (
            not stat.S_ISREG(result.st_mode)
            or getattr(result, "st_nlink", 1) != 1
            or _file_identity(lock_path) != (result.st_dev, result.st_ino)
        ):
            raise ValueError("publication lock identity is unsafe")
        if created:
            handle.write(b"\0"); handle.flush(); os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                if os.fstat(descriptor).st_size != 1:
                    raise ValueError("publication lock content is invalid")
                yield
            finally:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                if os.fstat(descriptor).st_size != 1:
                    raise ValueError("publication lock content is invalid")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if handle is not None: handle.close()
        os.close(descriptor)


def _cleanup_temporary_paths(paths: tuple[Path, ...]) -> None:
    for path in paths:
        try: path.unlink(missing_ok=True)
        except OSError: pass


def _validate_output_directory(output_dir: Path) -> None:
    if _path_present(output_dir) and not output_dir.is_dir(): raise NotADirectoryError(f"output directory is not a directory: {output_dir}")
    current = output_dir if _path_present(output_dir) else output_dir.parent
    while not _path_present(current) and current != current.parent: current = current.parent
    if not _path_present(current) or not current.is_dir(): raise NotADirectoryError(f"output parent does not exist: {output_dir.parent}")
    probe = output_dir
    while probe != probe.parent:
        if _path_present(probe) and _is_reparse_point(probe): raise ValueError(f"output path contains reparse-point component: {probe}")
        probe = probe.parent


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse_point(path: Path) -> bool:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(result.st_mode) or bool(
        getattr(result, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> FileIdentity:
    result = path.stat(follow_symlinks=False); return result.st_dev, result.st_ino


def _identity_matches(path: Path, identity: FileIdentity) -> bool:
    try: return _file_identity(path) == identity
    except OSError: return False


def _paths_alias(source: Path, destination: Path) -> bool:
    try:
        if _path_present(source) and _path_present(destination): return os.path.samefile(source, destination)
    except OSError: pass
    try: return source.resolve(strict=False) == destination.resolve(strict=False)
    except (OSError, RuntimeError): return os.path.abspath(source) == os.path.abspath(destination)


def _fsync_directory(directory: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    unsupported = {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    if os.name == "nt":
        unsupported.add(errno.EACCES)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in unsupported:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in unsupported:
            raise
    finally:
        os.close(descriptor)


__all__ = ["publish_files_atomically"]
