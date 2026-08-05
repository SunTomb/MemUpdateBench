from pathlib import Path
import json

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.v3.runtime import (
    MemoryEntryRecordV3,
    ParserExtractorProvenanceV3,
    RetrievalTraceV3,
    TaskRunRecordV3,
)
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


def test_checked_in_v3_task_schema_exports_reference_resolution_discriminator() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "vnext" / "v3" / "mem_update_task.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    selector = schema["$defs"]["MemoryQueryV3"]["properties"]["selector"]
    assert selector["discriminator"]["propertyName"] == "kind"
    assert selector["discriminator"]["mapping"]["reference_resolution"] == "#/$defs/ReferenceResolutionSelector"
    assert {branch["$ref"] for branch in selector["oneOf"]} >= {
        "#/$defs/ReferenceResolutionSelector"
    }
    reference_kind = schema["$defs"]["ReferenceResolutionSelector"]["properties"]["kind"]
    assert reference_kind["const"] == "reference_resolution"


def test_v3_task_run_schema_requires_action_id(tmp_path: Path) -> None:
    exported = {path.name: path for path in export_schemas(tmp_path / "v3", version="3.0.0")}
    schema = json.loads(exported["task_run_record.schema.json"].read_text(encoding="utf-8"))
    parsed_action_schema = schema["$defs"]["ParsedManagerActionV3"]
    assert "action_id" in parsed_action_schema["properties"]
    assert "action_id" in parsed_action_schema["required"]


def test_v3_task_run_schema_enforces_positive_integer_ranks(tmp_path: Path) -> None:
    exported = {path.name: path for path in export_schemas(tmp_path / "v3", version="3.0.0")}
    schema = json.loads(exported["task_run_record.schema.json"].read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = TaskRunRecordV3(
        task_id="task",
        adapter_id="adapter",
        run_id="run",
        retrieval_traces=(
            RetrievalTraceV3(
                query_id="query",
                retrieved_entries=(MemoryEntryRecordV3(entry_id="entry", content="value"),),
                ranks=(1,),
            ),
        ),
        parser_extractor_provenance=ParserExtractorProvenanceV3(
            action_parser_version="1",
            answer_parser_version="1",
            memory_entry_extractor_version="1",
            redaction_policy_version="1",
        ),
        completion_status=CompletionStatus.COMPLETED,
    ).model_dump(mode="json")

    validator.validate(payload)
    for invalid_rank in (0, -1):
        payload["retrieval_traces"][0]["ranks"] = [invalid_rank]
        with pytest.raises(JsonSchemaValidationError):
            validator.validate(payload)


def test_v3_manifest_record_hash_schema_rejects_blank_property_names(tmp_path: Path) -> None:
    exported = {path.name: path for path in export_schemas(tmp_path / "v3", version="3.0.0")}
    for filename, field in (("task_manifest.schema.json", "task_record_hashes"), ("run_manifest.schema.json", "run_record_hashes")):
        schema = json.loads(exported[filename].read_text(encoding="utf-8"))
        assert schema["properties"][field]["propertyNames"]["pattern"] == r"\S"
