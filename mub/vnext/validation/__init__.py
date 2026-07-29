from mub.vnext.contracts import (
    GeneratorProvenance,
    LegacyProvenance,
    MemUpdateTask,
    SourceRecord,
    SourceType,
    TaskFamily,
    TaskMetadata,
)
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
    merge_reports,
)
from mub.vnext.validation.replay import (
    ReplayResult,
    replay_actions,
    validate_distractors,
    validate_gold_replay,
)
from mub.vnext.validation.split import (
    FAMILY_STRATIFICATION_AXES,
    SliceDefinition,
    SplitException,
    validate_splits,
)
from mub.vnext.validation.task import validate_task
from mub.vnext.validation.pilot import validate_family_a_task, validate_family_d_task


_PILOT_GENERATOR = "memupdatebench_vnext_pilot"
_LEGACY_P63_GENERATOR = "legacy_p63_episode_compiler"
_LEGACY_CODE_REVISION = "legacy-compatibility-import"
_PHASE0_GENERIC_GENERATORS = frozenset(
    {
        ("vnext_phase0_factory", "fixed-test-revision"),
        ("vnext_smoke", "fixed-smoke-revision"),
    }
)


def _raw_model(model, expected_type):
    if type(model) is not expected_type:
        return None
    raw = object.__getattribute__(model, "__dict__")
    return raw if type(raw) is dict else None


def _recognizes_legacy_family_a(raw) -> bool:
    source_raw = _raw_model(raw.get("source"), SourceRecord)
    metadata_raw = _raw_model(raw.get("metadata"), TaskMetadata)
    if source_raw is None or metadata_raw is None:
        return False

    generator_raw = _raw_model(
        source_raw.get("generator"),
        GeneratorProvenance,
    )
    legacy_raw = _raw_model(
        metadata_raw.get("legacy_provenance"),
        LegacyProvenance,
    )
    if generator_raw is None or legacy_raw is None:
        return False
    generator_name = generator_raw.get("generator_name")
    if type(generator_name) is not str:
        return False
    if str.__eq__(generator_name, _PILOT_GENERATOR) is True:
        return False

    tags = metadata_raw.get("tags")
    source_provenance = source_raw.get("provenance")
    extra = metadata_raw.get("extra")
    if (
        type(tags) is not list
        or len(tags) > 64
        or any(type(tag) is not str for tag in tags)
        or type(source_provenance) is not dict
        or len(source_provenance) > 64
        or type(extra) is not dict
        or len(extra) > 64
    ):
        return False

    source_hash = source_raw.get("raw_hash")
    legacy_hash = legacy_raw.get("source_artifact_hash")
    generator_config_hash = generator_raw.get("config_sha256")
    metadata_config_hash = metadata_raw.get("generation_config_hash")
    generator_version = generator_raw.get("compiler_version")
    metadata_version = metadata_raw.get("compiler_version")
    return all(
        (
            str.__eq__(generator_name, _LEGACY_P63_GENERATOR) is True,
            type(generator_raw.get("code_revision")) is str
            and str.__eq__(
                generator_raw["code_revision"],
                _LEGACY_CODE_REVISION,
            )
            is True,
            type(source_raw.get("source_type")) is SourceType
            and source_raw["source_type"] is SourceType.SYNTHETIC,
            type(source_raw.get("license_or_privacy")) is str
            and str.__eq__(
                source_raw["license_or_privacy"],
                "compatibility_only",
            )
            is True,
            "compatibility" in tags,
            "vnext_pilot" not in tags,
            "release_id" not in source_provenance,
            not any(
                marker in extra
                for marker in (
                    "core_index",
                    "semantic_core_id",
                    "stratification",
                    "surface_variant",
                )
            ),
            type(source_provenance.get("normalization_version")) is str
            and str.__eq__(
                source_provenance["normalization_version"],
                "semantic-source-v1",
            )
            is True,
            type(extra.get("legacy_parser_dependency")) is str
            and str.__eq__(
                extra["legacy_parser_dependency"],
                "scripts.eval_evomemory.parse_event_slot",
            )
            is True,
            type(legacy_raw.get("legacy_family_id")) is str
            and str.__eq__(
                legacy_raw["legacy_family_id"],
                "evomemory_update_frequency",
            )
            is True,
            type(source_hash) is str
            and type(legacy_hash) is str
            and str.__eq__(source_hash, legacy_hash) is True,
            type(generator_config_hash) is str
            and type(metadata_config_hash) is str
            and str.__eq__(generator_config_hash, metadata_config_hash) is True,
            type(generator_version) is str
            and type(metadata_version) is str
            and str.__eq__(generator_version, metadata_version) is True,
        )
    )


def _recognizes_phase0_generic_family_a(raw) -> bool:
    source_raw = _raw_model(raw.get("source"), SourceRecord)
    metadata_raw = _raw_model(raw.get("metadata"), TaskMetadata)
    if source_raw is None or metadata_raw is None:
        return False
    generator_raw = _raw_model(
        source_raw.get("generator"),
        GeneratorProvenance,
    )
    if generator_raw is None:
        return False

    generator_name = generator_raw.get("generator_name")
    code_revision = generator_raw.get("code_revision")
    if type(generator_name) is not str or type(code_revision) is not str:
        return False
    tags = metadata_raw.get("tags")
    extra = metadata_raw.get("extra")
    source_provenance = source_raw.get("provenance")
    generator_config_hash = generator_raw.get("config_sha256")
    metadata_config_hash = metadata_raw.get("generation_config_hash")
    generator_version = generator_raw.get("compiler_version")
    metadata_version = metadata_raw.get("compiler_version")
    return all(
        (
            type(generator_name) is str,
            type(code_revision) is str,
            (generator_name, code_revision) in _PHASE0_GENERIC_GENERATORS,
            type(source_raw.get("source_type")) is SourceType
            and source_raw["source_type"] is SourceType.SYNTHETIC,
            type(source_raw.get("license_or_privacy")) is str
            and str.__eq__(
                source_raw["license_or_privacy"],
                "synthetic_redistributable",
            )
            is True,
            metadata_raw.get("legacy_provenance") is None,
            type(tags) is list and not tags,
            type(extra) is dict and not extra,
            type(source_provenance) is dict
            and len(source_provenance) <= 64
            and "release_id" not in source_provenance,
            type(generator_config_hash) is str
            and type(metadata_config_hash) is str
            and str.__eq__(generator_config_hash, metadata_config_hash) is True,
            type(generator_version) is str
            and type(metadata_version) is str
            and str.__eq__(generator_version, "1.0.0") is True
            and str.__eq__(generator_version, metadata_version) is True,
        )
    )


def validate_task_semantics(task) -> ValidationReport:
    identifies_family_a = False
    recognizes_generic_family_a = False
    identifies_family_d = False
    if type(task) is MemUpdateTask:
        raw = object.__getattribute__(task, "__dict__")
        if type(raw) is dict:
            candidate = raw.get("task_family")
            identifies_family_a = isinstance(candidate, str) and str.__eq__(
                candidate,
                TaskFamily.REPEATED_SAME_SLOT.value,
            ) is True
            identifies_family_d = isinstance(candidate, str) and str.__eq__(
                candidate,
                TaskFamily.NOOP_WRITE_DISCIPLINE.value,
            ) is True
            recognizes_generic_family_a = identifies_family_a and (
                _recognizes_legacy_family_a(raw)
                or _recognizes_phase0_generic_family_a(raw)
            )
    if identifies_family_a and not recognizes_generic_family_a:
        return validate_family_a_task(task)
    if identifies_family_d:
        return validate_family_d_task(task)
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(task),
    )


__all__ = [
    "FAMILY_STRATIFICATION_AXES",
    "ReplayResult",
    "SliceDefinition",
    "SplitException",
    "ValidationIssue",
    "ValidationReport",
    "build_report",
    "merge_reports",
    "replay_actions",
    "validate_distractors",
    "validate_family_a_task",
    "validate_family_d_task",
    "validate_gold_replay",
    "validate_splits",
    "validate_task",
    "validate_task_semantics",
]
