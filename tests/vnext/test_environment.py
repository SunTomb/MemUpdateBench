import inspect
from pathlib import Path

from pydantic import BaseModel, VERSION


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
