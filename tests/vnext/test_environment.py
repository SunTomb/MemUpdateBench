import ast
import inspect
from pathlib import Path

from pydantic import BaseModel, VERSION


def _python310_incompatible_self_references(source: str) -> list[str]:
    tree = ast.parse(source)
    typing_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "typing"
    }
    references = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            if any(alias.name == "Self" for alias in node.names):
                references.append("typing.Self")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "Self"
            and isinstance(node.value, ast.Name)
            and node.value.id in typing_aliases
        ):
            references.append("typing.Self")
    return references


def test_pydantic_version_supports_canonical_dump_api() -> None:
    version = tuple(int(part) for part in VERSION.split(".")[:2])
    assert version >= (2, 12)
    assert "exclude_computed_fields" in inspect.signature(BaseModel.model_dump).parameters


def test_hash_sensitive_artifacts_are_checked_out_with_lf() -> None:
    project_root = Path(__file__).resolve().parents[2]
    rules = {
        line.strip()
        for line in (project_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {
        "schemas/vnext/*.schema.json text eol=lf",
        "schemas/legacy/*.schema.json text eol=lf",
        "tests/vnext/fixtures/legacy/*.md text eol=lf",
    } <= rules


def test_vnext_self_annotations_support_declared_python310() -> None:
    project_root = Path(__file__).resolve().parents[2]
    incompatible_imports = []
    for path in (project_root / "mub" / "vnext").rglob("*.py"):
        references = _python310_incompatible_self_references(
            path.read_text(encoding="utf-8")
        )
        if references:
            incompatible_imports.append(path.relative_to(project_root).as_posix())

    assert incompatible_imports == []


def test_python310_guard_detects_qualified_typing_self() -> None:
    assert _python310_incompatible_self_references(
        "import typing as t\n\ndef method() -> t.Self:\n    ...\n"
    ) == ["typing.Self"]
