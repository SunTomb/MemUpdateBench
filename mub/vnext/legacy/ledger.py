from __future__ import annotations

import difflib
import hashlib
import re
import shlex
import stat
from pathlib import Path, PurePosixPath

from pydantic import Field

from mub.vnext.contracts.common import ImmutableContractModel, SHA256_PATTERN
from mub.vnext.legacy.loaders import _decode_utf8, _load_stable


_CODE_REFERENCE_RE = re.compile(r"`([^`\r\n]+)`")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")
_GLOB_CHARACTERS = frozenset("*?[]{}")
_ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".csv",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".parquet",
        ".py",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_MAX_LEDGER_BYTES = 8 * 1024 * 1024


class ResolvedLedgerReference(ImmutableContractModel):
    reference: str
    line_number: int = Field(gt=0, strict=True)
    resolved_path: str


class UnresolvedLedgerReference(ImmutableContractModel):
    reference: str
    line_number: int = Field(gt=0, strict=True)
    reason: str
    candidate_aliases: tuple[str, ...] = ()


class LedgerReferenceAudit(ImmutableContractModel):
    ledger_path: str
    ledger_sha256: str = Field(pattern=SHA256_PATTERN)
    project_root: str
    resolved: tuple[ResolvedLedgerReference, ...]
    unresolved: tuple[UnresolvedLedgerReference, ...]


def _signature(path: Path) -> tuple[int, int, int, int]:
    try:
        result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ledger does not exist: {path}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise IsADirectoryError(f"Ledger is not a regular file: {path}")
    if result.st_size > _MAX_LEDGER_BYTES:
        raise ValueError(f"Ledger exceeds byte cap {_MAX_LEDGER_BYTES}: {path}")
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ledger(raw: bytes, path: Path) -> str:
    return _decode_utf8(raw, path)


def _load_ledger(path: Path) -> tuple[str, str]:
    before = _signature(path)
    before_hash = _sha256_file(path)
    text = _load_stable(path, _parse_ledger)
    after_hash = _sha256_file(path)
    after = _signature(path)
    if before != after or before_hash != after_hash:
        raise RuntimeError(f"Ledger changed during audit: {path}")
    return text, before_hash


def _looks_like_artifact_reference(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    if "://" in value:
        return False
    if value.endswith("/"):
        return True
    return PurePosixPath(value).suffix.lower() in _ARTIFACT_SUFFIXES


def _fenced_candidates(line: str, language: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    if language in {"bash", "sh", "shell", "zsh"}:
        shell_line = stripped[:-1].rstrip() if stripped.endswith("\\") else stripped
        try:
            tokens = shlex.split(shell_line, comments=True, posix=True)
        except ValueError:
            return []
        candidates: list[str] = []
        for token in tokens:
            if token.startswith("-"):
                if "=" not in token:
                    continue
                token = token.split("=", 1)[1]
            elif "=" in token and not token.startswith(("./", "../")):
                _, token = token.split("=", 1)
            if _looks_like_artifact_reference(token):
                candidates.append(token)
        return candidates
    if any(character.isspace() for character in stripped):
        return []
    return [stripped] if _looks_like_artifact_reference(stripped) else []


def _extract_references(text: str) -> list[tuple[str, int]]:
    references: list[tuple[str, int]] = []
    seen: set[str] = set()
    in_fence = False
    fence_language = ""

    def add(reference: str, line_number: int) -> None:
        if reference in seen or not _looks_like_artifact_reference(reference):
            return
        seen.add(reference)
        references.append((reference, line_number))

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                fence_language = ""
            else:
                in_fence = True
                language_text = stripped[3:].strip().lower()
                fence_language = (
                    language_text.split(None, 1)[0] if language_text else ""
                )
            continue
        if in_fence:
            for reference in _fenced_candidates(line, fence_language):
                add(reference, line_number)
            continue
        for match in _CODE_REFERENCE_RE.finditer(line):
            add(match.group(1), line_number)
    return references


def _unsafe_reason(reference: str) -> str | None:
    if "\x00" in reference or "\\" in reference:
        return "unsafe_path"
    if _WINDOWS_DRIVE_RE.match(reference) or reference.startswith("/"):
        return "unsafe_path"
    if any(character in reference for character in _GLOB_CHARACTERS):
        return "unsafe_path"
    path = PurePosixPath(reference)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return "unsafe_path"
    if any("\x00" in part or ":" in part for part in path.parts):
        return "unsafe_path"
    return None


def _has_symlink_component(root: Path, parts: tuple[str, ...]) -> bool:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            return False
    return False


def _exact_case_exists(root: Path, parts: tuple[str, ...]) -> bool:
    current = root
    for part in parts:
        try:
            names = {child.name for child in current.iterdir()}
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return False
        if part not in names:
            return False
        current = current / part
    return current.exists()


def _candidate_aliases(root: Path, reference: str) -> tuple[str, ...]:
    pure = PurePosixPath(reference)
    parent_parts = pure.parts[:-1]
    parent = root.joinpath(*parent_parts)
    if not parent.is_dir() or _has_symlink_component(root, parent_parts):
        return ()
    try:
        names = sorted(child.name for child in parent.iterdir())
    except (PermissionError, OSError):
        return ()
    close = difflib.get_close_matches(pure.name, names, n=3, cutoff=0.72)
    prefix = PurePosixPath(*parent_parts)
    return tuple(
        (prefix / name).as_posix() if parent_parts else name
        for name in close
    )


def audit_ledger_references(
    ledger_path: Path,
    project_root: Path,
) -> LedgerReferenceAudit:
    """Audit exact repository-relative references without reading target payloads."""

    if type(ledger_path) is not type(Path()) or type(project_root) is not type(Path()):
        raise TypeError("ledger_path and project_root must be exact concrete pathlib.Path values")
    ledger = Path(ledger_path)
    root = Path(project_root)
    try:
        root_stat = root.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Project root does not exist: {root}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    root_resolved = root.resolve(strict=True)
    text, ledger_hash = _load_ledger(ledger)

    resolved: list[ResolvedLedgerReference] = []
    unresolved: list[UnresolvedLedgerReference] = []
    for reference, line_number in _extract_references(text):
        unsafe = _unsafe_reason(reference)
        if unsafe is not None:
            unresolved.append(
                UnresolvedLedgerReference(
                    reference=reference,
                    line_number=line_number,
                    reason=unsafe,
                    candidate_aliases=(),
                )
            )
            continue
        parts = PurePosixPath(reference).parts
        target = root_resolved.joinpath(*parts)
        try:
            target_resolved = target.resolve(strict=False)
            inside_root = target_resolved == root_resolved or target_resolved.is_relative_to(root_resolved)
        except (OSError, RuntimeError):
            inside_root = False
        if not inside_root or _has_symlink_component(root_resolved, parts):
            unresolved.append(
                UnresolvedLedgerReference(
                    reference=reference,
                    line_number=line_number,
                    reason="unsafe_path",
                    candidate_aliases=(),
                )
            )
            continue
        if _exact_case_exists(root_resolved, parts):
            resolved.append(
                ResolvedLedgerReference(
                    reference=reference,
                    line_number=line_number,
                    resolved_path=str(target_resolved),
                )
            )
            continue
        unresolved.append(
            UnresolvedLedgerReference(
                reference=reference,
                line_number=line_number,
                reason="not_found",
                candidate_aliases=_candidate_aliases(root_resolved, reference),
            )
        )

    return LedgerReferenceAudit(
        ledger_path=str(ledger),
        ledger_sha256=ledger_hash,
        project_root=str(root_resolved),
        resolved=tuple(resolved),
        unresolved=tuple(unresolved),
    )


__all__ = [
    "LedgerReferenceAudit",
    "ResolvedLedgerReference",
    "UnresolvedLedgerReference",
    "audit_ledger_references",
]
