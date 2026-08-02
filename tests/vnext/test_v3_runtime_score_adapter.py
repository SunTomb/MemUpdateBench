import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey, MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, ParsedManagerActionV3
from mub.vnext.contracts.v3.score import (
    CORE_METRIC_FIELD_PATHS,
    ScoreRecordV3,
    ScorerConfigV3,
    V3_FAILURE_FLAGS,
    V3_PRIMARY_FAILURE_PRECEDENCE,
)


def key() -> MemoryObjectKey:
    return MemoryObjectKey(object_type="slot", namespace="n", entity="e", attribute="a")


def test_runtime_v3_validates_multi_target_scope_and_citations() -> None:
    action = ParsedManagerActionV3(event_id="ev", operation="DELETE", observed_scope="entity", target_object_keys=(key(),), format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
    assert action.target_object_keys == (key(),)
    with pytest.raises(ValidationError):
        ParsedManagerActionV3(event_id="ev", operation="DELETE", observed_scope="entity", target_object_keys=(), format_valid=True, execution_status="executed", fallback_used=False, raw_output="ok")
    with pytest.raises(ValidationError):
        AnswerPredictionV3(query_id="q", raw_output="x", parsed_answer="x", format_valid=True, cited_object_keys=(key(),), cited_derivation_step_ids=("",))


def test_v3_metric_nulls_require_complete_typed_support_map() -> None:
    with pytest.raises(ValidationError):
        ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields={})
    support = {path: MetricFieldSupport(reason=SupportReason.NOT_SUPPORTED, null_policy="emit_null") for path in CORE_METRIC_FIELD_PATHS}
    score = ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields=support)
    assert set(score.supported_metric_fields) == CORE_METRIC_FIELD_PATHS


def test_v3_failure_flags_are_deduplicated_and_use_unified_precedence() -> None:
    support = {path: MetricFieldSupport(reason=SupportReason.NOT_SUPPORTED, null_policy="emit_null") for path in CORE_METRIC_FIELD_PATHS}
    score = ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields=support, failure_flags=("stale_copied", "wrong_delete_scope", "system_exception", "stale_copied"))
    assert len(score.failure_flags) == 3
    assert score.primary_failure == "system_exception"
    assert len(V3_PRIMARY_FAILURE_PRECEDENCE) == len(set(V3_PRIMARY_FAILURE_PRECEDENCE))
    assert set(V3_PRIMARY_FAILURE_PRECEDENCE) == set(V3_FAILURE_FLAGS)
    layered = ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields=support, failure_flags=("stale_propagation", "current_state_missing"))
    assert layered.primary_failure == "current_state_missing"
    empty = ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields=support)
    assert empty.primary_failure == "correct"
    with pytest.raises(ValidationError):
        ScoreRecordV3.empty(task_id="t", run_id="r", adapter_id="a", task_family="f", difficulty="easy", completion_status="completed", supported_metric_fields=support, failure_flags=("stale_copied", "wrong_delete_scope"), primary_failure="stale_copied")


def test_scorer_config_v3_accepts_every_core_metric_and_rejects_old_versions() -> None:
    config = ScorerConfigV3(requested_metric_fields=tuple(reversed(sorted(CORE_METRIC_FIELD_PATHS))))
    assert config.requested_metric_fields == tuple(sorted(CORE_METRIC_FIELD_PATHS))
    assert len(config.configuration_hash) == 64
    assert {path.split(".", 1)[0] for path in config.requested_metric_fields} >= {"deletion_scores", "historical_scores", "synthesis_scores"}
    with pytest.raises(ValidationError):
        ScorerConfigV3(scorer_version="2.0.0")
    with pytest.raises(ValidationError):
        ScorerConfigV3(requested_metric_fields=("unknown_scores.metric",))


def test_v3_capabilities_preserve_levels_and_add_core_features() -> None:
    caps = AdapterCapabilitiesV3(supports_native_answer=True, supports_scoped_delete=True, supports_historical_query=True, exports_version_history=True, supports_multi_object_query=True, exports_evidence_linkage=True)
    assert caps.presentation_level() == 0
    assert caps.core_capability_requirements()["historical_query"] is True
