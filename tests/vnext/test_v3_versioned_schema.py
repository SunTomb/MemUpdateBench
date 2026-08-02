from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.io.versioned import parse_versioned_task
from mub.vnext.schema_export import export_schemas


def test_version_dispatch_fails_closed() -> None:
    for payload in ({}, {"schema_version": "9.0.0"}, {"schema_version": 3}):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            parse_versioned_task(payload)


def test_v3_schema_export_is_deterministic_and_separate(tmp_path: Path) -> None:
    first = export_schemas(tmp_path / "first", version="3.0.0")
    second = export_schemas(tmp_path / "second", version="3.0.0")
    assert len(first) == 5
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert all(b'"$schema":"https://json-schema.org/draft/2020-12/schema"' in path.read_bytes() for path in first)
