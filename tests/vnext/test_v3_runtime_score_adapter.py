import pytest
from pydantic import ValidationError

from mub.vnext.contracts.common import MemoryObjectKey, MetricFieldSupport
from mub.vnext.contracts.enums import SupportReason
from mub.vnext.contracts.v3.adapter import AdapterCapabilitiesV3
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, ParsedManagerActionV3
from mub.vnext.contracts.v3.score import CORE_METRIC_FIELD_PATHS, ScoreRecordV3


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


def test_v3_capabilities_preserve_levels_and_add_core_features() -> None:
    caps = AdapterCapabilitiesV3(supports_native_answer=True, supports_scoped_delete=True, supports_historical_query=True, exports_version_history=True, supports_multi_object_query=True, exports_evidence_linkage=True)
    assert caps.presentation_level() == 0
    assert caps.core_capability_requirements()["historical_query"] is True
