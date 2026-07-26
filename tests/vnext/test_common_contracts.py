import pytest
from pydantic import ValidationError

import mub.vnext as vnext
import mub.vnext.contracts as contracts
import mub.vnext.contracts.common as common_contracts
from mub.vnext.contracts.common import (
    GeneratorProvenance,
    MemoryObjectKey,
    MetricFieldSupport,
    SourceAnchor,
    SourceRecord,
)
from mub.vnext.contracts.enums import (
    AnswerSchema,
    Difficulty,
    EvaluationMode,
    EventRole,
    Operation,
    QueryType,
    SourceType,
    Split,
    SupportReason,
    TaskFamily,
)
from mub.vnext.version import (
    COMPILER_VERSION,
    METRIC_REGISTRY_VERSION,
    PRIMARY_FAILURE_PRECEDENCE_VERSION,
    PROFILE_VERSION,
    RUN_MANIFEST_VERSION,
    RUNTIME_RECORD_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TASK_MANIFEST_VERSION,
)


RAW_HASH = "a" * 64
NORMALIZED_HASH = "b" * 64
GENERATOR_CONFIG_HASH = "c" * 64


def synthetic_generator_data() -> dict[str, object]:
    return {
        "generator_name": "unit_generator",
        "seed": 7,
        "config_sha256": GENERATOR_CONFIG_HASH,
        "code_revision": "test-revision",
        "compiler_version": "1.0.0",
    }


def test_version_constants_are_exported_and_pinned() -> None:
    expected_versions = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "SCORER_VERSION": SCORER_VERSION,
        "METRIC_REGISTRY_VERSION": METRIC_REGISTRY_VERSION,
        "COMPILER_VERSION": COMPILER_VERSION,
        "PROFILE_VERSION": PROFILE_VERSION,
        "RUNTIME_RECORD_VERSION": RUNTIME_RECORD_VERSION,
        "TASK_MANIFEST_VERSION": TASK_MANIFEST_VERSION,
        "RUN_MANIFEST_VERSION": RUN_MANIFEST_VERSION,
        "PRIMARY_FAILURE_PRECEDENCE_VERSION": PRIMARY_FAILURE_PRECEDENCE_VERSION,
    }
    assert set(expected_versions.values()) == {"1.0.0"}
    for name, value in expected_versions.items():
        assert getattr(vnext, name) == value
        assert name in vnext.__all__


def test_exact_object_key_includes_namespace_and_subkey() -> None:
    base = MemoryObjectKey(
        object_type="slot",
        namespace="n1",
        entity="friend:alex",
        attribute="location",
    )
    other_namespace = base.model_copy(update={"namespace": "n2"})
    other_subkey = base.model_copy(update={"subkey": "office"})
    other_object_type = base.model_copy(update={"object_type": "profile"})
    assert base != other_namespace
    assert base != other_subkey
    assert base == other_object_type
    assert base.canonical_id == "n1|friend:alex|location|"
    assert other_object_type.canonical_id == base.canonical_id


def test_object_key_equality_uses_only_exact_four_part_identity() -> None:
    base = MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="friend:alex",
        attribute="location",
        subkey="office",
    )
    other_classification = base.model_copy(update={"object_type": "profile"})
    other_identity = base.model_copy(update={"attribute": "timezone"})
    assert base == other_classification
    assert other_classification == base
    assert base != other_identity
    with pytest.raises(TypeError):
        hash(base)


def test_object_key_requires_object_type() -> None:
    with pytest.raises(ValidationError):
        MemoryObjectKey(namespace="n1", entity="friend:alex", attribute="location")


def test_object_key_strips_and_normalizes_blank_subkey() -> None:
    key = MemoryObjectKey(
        object_type=" slot ",
        namespace=" default ",
        entity=" friend:alex ",
        attribute=" location ",
        subkey="   ",
    )
    assert key.object_type == "slot"
    assert key.namespace == "default"
    assert key.entity == "friend:alex"
    assert key.attribute == "location"
    assert key.subkey is None
    assert key.canonical_id == "default|friend:alex|location|"


def test_object_key_rejects_blank_identity_parts() -> None:
    for field_name in ("object_type", "namespace", "entity", "attribute"):
        data = {
            "object_type": "slot",
            "namespace": "default",
            "entity": "friend:alex",
            "attribute": "location",
        }
        data[field_name] = " "
        with pytest.raises(ValidationError):
            MemoryObjectKey(**data)


def test_object_key_escapes_delimiter_without_rejecting_it() -> None:
    key = MemoryObjectKey(
        object_type="slot",
        namespace="default",
        entity="friend|alex",
        attribute="location",
    )
    assert key.canonical_id == "default|friend%7Calex|location|"


def test_object_key_escapes_percent_before_delimiter() -> None:
    key = MemoryObjectKey(
        object_type="slot",
        namespace="team%west",
        entity="friend%|alex",
        attribute="loc%ation",
        subkey="desk|50%",
    )
    assert key.canonical_id == "team%25west|friend%25%7Calex|loc%25ation|desk%7C50%25"


def test_object_key_dump_round_trips_without_computed_canonical_id() -> None:
    key = MemoryObjectKey(
        object_type="slot",
        namespace="n1",
        entity="friend:alex",
        attribute="location",
    )
    dumped = key.model_dump()
    assert "canonical_id" not in dumped
    assert MemoryObjectKey.model_validate(dumped) == key


def test_controlled_vocabularies_match_vnext_design() -> None:
    assert [item.value for item in SourceType] == [
        "synthetic",
        "dialogue",
        "changelog",
        "calendar",
        "issue",
        "report_revision",
        "other",
    ]
    assert [item.value for item in Split] == ["train", "dev", "test", "evaluation_only"]
    assert [item.value for item in Difficulty] == ["easy", "medium", "hard", "challenge"]
    assert [item.value for item in QueryType] == [
        "current_state",
        "historical_state",
        "transition",
        "multi_object",
        "deletion_compliance",
    ]
    assert [item.value for item in EvaluationMode] == [
        "state_direct",
        "retrieved_prompt",
        "native_system",
    ]
    assert [item.value for item in AnswerSchema] == ["string", "number", "boolean", "list", "object"]
    assert [item.value for item in SupportReason] == [
        "not_applicable",
        "not_supported",
        "runtime_failed",
        "missing_artifact",
    ]
    assert Operation.DELETE.value == "DELETE"
    assert TaskFamily.REPEATED_SAME_SLOT.value == "repeated_same_slot_update"
    assert EventRole.DUPLICATE_CURRENT.value == "duplicate_current"
    assert SCHEMA_VERSION == "1.0.0"
    assert contracts.AnswerSchema is AnswerSchema
    assert not hasattr(contracts, "ExecutionStatus")


def test_task_family_remains_available_as_helper_enum() -> None:
    assert TaskFamily.REPEATED_SAME_SLOT.value == "repeated_same_slot_update"
    assert contracts.TaskFamily is TaskFamily


def test_metric_field_support_uses_canonical_fields() -> None:
    support = MetricFieldSupport(
        reason=SupportReason.NOT_SUPPORTED,
        null_policy="emit_null",
        detail="metric not implemented",
    )
    assert support.model_dump() == {
        "reason": SupportReason.NOT_SUPPORTED,
        "null_policy": "emit_null",
        "detail": "metric not implemented",
    }
    with pytest.raises(ValidationError):
        MetricFieldSupport(reason=SupportReason.NOT_SUPPORTED)
    assert contracts.MetricFieldSupport is MetricFieldSupport
    assert "MetricFieldSupport" in contracts.__all__
    assert "MetricSupport" not in common_contracts.__all__
    assert "MetricSupport" not in contracts.__all__
    assert not hasattr(common_contracts, "MetricSupport")
    assert not hasattr(contracts, "MetricSupport")


def test_source_record_uses_canonical_fields_and_round_trips() -> None:
    record = SourceRecord(
        source_id="src-1",
        source_type=SourceType.DIALOGUE,
        source_uri=None,
        license_or_privacy="private_eval_only",
        raw_hash=None,
        normalized_hash=NORMALIZED_HASH,
        normalization_version="norm-v1",
        provenance={"turns": 3, "speaker": "alex"},
        generator=None,
    )
    dumped = record.model_dump()
    assert dumped == {
        "source_id": "src-1",
        "source_type": SourceType.DIALOGUE,
        "source_uri": None,
        "license_or_privacy": "private_eval_only",
        "raw_hash": None,
        "normalized_hash": NORMALIZED_HASH,
        "normalization_version": "norm-v1",
        "provenance": {"turns": 3, "speaker": "alex"},
        "generator": None,
    }
    assert SourceRecord.model_validate(dumped) == record


def test_source_record_rejects_legacy_fields_and_requires_canonical_fields() -> None:
    valid_data = {
        "source_id": "src-1",
        "source_type": "synthetic",
        "source_uri": "memory://src-1",
        "license_or_privacy": "synthetic",
        "raw_hash": RAW_HASH,
        "normalized_hash": NORMALIZED_HASH,
        "normalization_version": "norm-v1",
        "provenance": {},
        "generator": synthetic_generator_data(),
    }
    with pytest.raises(ValidationError):
        SourceRecord(**{**valid_data, "license_id": "legacy"})
    for required_field in valid_data:
        missing = dict(valid_data)
        missing.pop(required_field)
        with pytest.raises(ValidationError):
            SourceRecord(**missing)


def test_synthetic_source_requires_typed_generator_provenance() -> None:
    source_data = {
        "source_id": "src-synthetic",
        "source_type": SourceType.SYNTHETIC,
        "source_uri": None,
        "license_or_privacy": "synthetic",
        "raw_hash": RAW_HASH,
        "normalized_hash": NORMALIZED_HASH,
        "normalization_version": "norm-v1",
        "provenance": {},
    }
    with pytest.raises(ValidationError):
        SourceRecord(**source_data, generator=None)
    with pytest.raises(ValidationError):
        SourceRecord(**source_data, generator={"generator_name": "incomplete"})
    with pytest.raises(ValidationError):
        SourceRecord(**source_data, generator={**synthetic_generator_data(), "extra": True})

    record = SourceRecord(**source_data, generator=synthetic_generator_data())
    assert isinstance(record.generator, GeneratorProvenance)
    assert record.generator.compiler_version == "1.0.0"


def test_non_synthetic_source_allows_omitted_generator() -> None:
    record = SourceRecord(
        source_id="src-dialogue",
        source_type=SourceType.DIALOGUE,
        source_uri=None,
        license_or_privacy="private_eval_only",
        raw_hash=None,
        normalized_hash=NORMALIZED_HASH,
        normalization_version="norm-v1",
        provenance={},
    )
    assert record.generator is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("seed", "7"),
        ("seed", True),
        ("seed", 7.0),
        ("generator_name", b"unit_generator"),
        ("config_sha256", b"c" * 64),
        ("code_revision", b"test-revision"),
        ("compiler_version", b"1.0.0"),
    ],
)
def test_generator_provenance_rejects_scalar_coercions(
    field_name: str,
    invalid_value: object,
) -> None:
    data = synthetic_generator_data()
    data[field_name] = invalid_value
    with pytest.raises(ValidationError):
        GeneratorProvenance(**data)


def test_generator_provenance_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GeneratorProvenance(**synthetic_generator_data(), extra="forbidden")


def test_source_record_rejects_invalid_sha256_hashes() -> None:
    valid_data = {
        "source_id": "src-1",
        "source_type": "synthetic",
        "source_uri": "memory://src-1",
        "license_or_privacy": "synthetic",
        "raw_hash": RAW_HASH,
        "normalized_hash": NORMALIZED_HASH,
        "normalization_version": "norm-v1",
        "provenance": {},
        "generator": synthetic_generator_data(),
    }
    invalid_hashes = ("short", "A" * 64, "g" * 64)
    for field_name in ("raw_hash", "normalized_hash"):
        for invalid_hash in invalid_hashes:
            with pytest.raises(ValidationError):
                SourceRecord(**{**valid_data, field_name: invalid_hash})
    assert SourceRecord(**{**valid_data, "raw_hash": None}).raw_hash is None


def test_source_anchor_accepts_absent_or_ordered_character_span() -> None:
    absent = SourceAnchor(document_id="doc1", section_id="sec1")
    present = SourceAnchor(document_id="doc1", section_id="sec1", start_char=3, end_char=7)
    assert absent.start_char is None
    assert absent.end_char is None
    assert present.start_char == 3
    assert present.end_char == 7


def test_source_anchor_rejects_partial_or_reversed_character_span() -> None:
    invalid_spans = (
        {"start_char": 3},
        {"end_char": 7},
        {"start_char": 8, "end_char": 7},
    )
    for span in invalid_spans:
        with pytest.raises(ValidationError):
            SourceAnchor(document_id="doc1", section_id="sec1", **span)
