from __future__ import annotations

import hashlib
from pathlib import Path

from mub.vnext.legacy import ledger as ledger_module
from mub.vnext.legacy.ledger import audit_ledger_references


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "legacy"
PROJECT_ROOT = Path(__file__).parents[2]


def test_ledger_audit_is_public_from_ledger_module() -> None:
    assert "audit_ledger_references" in ledger_module.__all__


def test_ledger_audit_resolves_exact_paths_without_alias_substitution() -> None:
    ledger = FIXTURE_DIR / "ledger_references.md"

    audit = audit_ledger_references(ledger, PROJECT_ROOT)

    assert [record.reference for record in audit.resolved] == [
        "tests/vnext/fixtures/legacy/p83_conflict_rows.csv"
    ]
    assert audit.resolved[0].resolved_path == str(
        (PROJECT_ROOT / "tests/vnext/fixtures/legacy/p83_conflict_rows.csv").resolve()
    )
    assert [record.reference for record in audit.unresolved] == [
        "tests/vnext/fixtures/legacy/p83_conflict_row.csv"
    ]
    assert audit.unresolved[0].reason == "not_found"
    assert not hasattr(audit.unresolved[0], "resolved_path")
    assert audit.ledger_sha256 == hashlib.sha256(ledger.read_bytes()).hexdigest()


def test_candidate_aliases_are_suggestions_and_never_change_resolution() -> None:
    audit = audit_ledger_references(
        FIXTURE_DIR / "ledger_references.md", PROJECT_ROOT
    )
    unresolved = audit.unresolved[0]

    assert "tests/vnext/fixtures/legacy/p83_conflict_rows.csv" in unresolved.candidate_aliases
    assert len(audit.resolved) == 1
    assert len(audit.unresolved) == 1


def test_ledger_audit_rejects_traversal_absolute_glob_and_out_of_root_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    safe = project / "safe.txt"
    safe.write_text("safe", encoding="utf-8")
    ledger = project / "ledger.md"
    ledger.write_text(
        "`safe.txt`\n"
        "`../outside.txt`\n"
        "`/absolute/path.txt`\n"
        "`results/**/*.json`\n"
        "`C:/private/raw.json`\n",
        encoding="utf-8",
    )

    audit = audit_ledger_references(ledger, project)

    assert [record.reference for record in audit.resolved] == ["safe.txt"]
    assert [(record.reference, record.reason) for record in audit.unresolved] == [
        ("../outside.txt", "unsafe_path"),
        ("/absolute/path.txt", "unsafe_path"),
        ("results/**/*.json", "unsafe_path"),
        ("C:/private/raw.json", "unsafe_path"),
    ]
    assert all(not record.candidate_aliases for record in audit.unresolved)


def test_ledger_audit_does_not_read_target_payloads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "artifact.bin"
    target.write_bytes(b"payload that must not be audited")
    ledger = project / "ledger.md"
    ledger.write_text("`artifact.bin`\n", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == target.resolve():
            raise AssertionError("target payload was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    audit = audit_ledger_references(ledger, project)

    assert [record.reference for record in audit.resolved] == ["artifact.bin"]


def test_fenced_markdown_extracts_text_and_shell_paths_without_prose_or_options() -> None:
    audit = audit_ledger_references(
        FIXTURE_DIR / "ledger_fenced_references.md", PROJECT_ROOT
    )

    assert [record.reference for record in audit.resolved] == [
        "tests/vnext/fixtures/legacy/p83_conflict_rows.csv"
    ]
    assert [(record.reference, record.reason) for record in audit.unresolved] == [
        ("results/**/*.json", "unsafe_path"),
        ("scripts/missing_audit.py", "not_found"),
        ("results/missing_fenced_summary.json", "not_found"),
    ]
    all_references = [
        *(record.reference for record in audit.resolved),
        *(record.reference for record in audit.unresolved),
    ]
    assert "--input" not in all_references
    assert "--output" not in all_references
    assert "results/mentioned-but-not-code.json" not in all_references
    assert all_references.count(
        "tests/vnext/fixtures/legacy/p83_conflict_rows.csv"
    ) == 1


def test_duplicate_ledger_reference_is_reported_once(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("{}", encoding="utf-8")
    ledger = tmp_path / "ledger.md"
    ledger.write_text("`artifact.json` and `artifact.json`\n", encoding="utf-8")

    audit = audit_ledger_references(ledger, tmp_path)

    assert [record.reference for record in audit.resolved] == ["artifact.json"]
    assert audit.unresolved == ()
