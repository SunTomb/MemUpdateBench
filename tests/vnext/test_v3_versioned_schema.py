from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from mub.vnext.io.versioned import parse_versioned_task
from mub.vnext.schema_export import SCHEMA_MODEL_REGISTRIES, export_schemas


def test_version_dispatch_fails_closed() -> None:
    for payload in ({}, {"schema_version": "9.0.0"}, {"schema_version": 3}):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            parse_versioned_task(payload)


def test_schema_registry_is_immutable_and_unknown_versions_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        SCHEMA_MODEL_REGISTRIES["9.0.0"] = SCHEMA_MODEL_REGISTRIES["2.0.0"]  # type: ignore[index]
    assert tuple(SCHEMA_MODEL_REGISTRIES) == ("2.0.0", "3.0.0")
    with pytest.raises(ValueError, match="unsupported schema registry version"):
        export_schemas(tmp_path / "hostile", version="9.0.0")
    assert not (tmp_path / "hostile").exists()


def test_v3_schema_export_is_deterministic_and_separate(tmp_path: Path) -> None:
    first = export_schemas(tmp_path / "first", version="3.0.0")
    second = export_schemas(tmp_path / "second", version="3.0.0")
    assert len(first) == 5
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert all(b'"$schema":"https://json-schema.org/draft/2020-12/schema"' in path.read_bytes() for path in first)


def test_v3_manifest_record_hash_schema_rejects_blank_property_names(tmp_path: Path) -> None:
    exported = {path.name: path for path in export_schemas(tmp_path / "v3", version="3.0.0")}
    for filename, field in (("task_manifest.schema.json", "task_record_hashes"), ("run_manifest.schema.json", "run_record_hashes")):
        schema = json.loads(exported[filename].read_text(encoding="utf-8"))
        assert schema["properties"][field]["propertyNames"]["pattern"] == r"\S"
