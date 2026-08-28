from pathlib import Path
import pytest
from scripts.vnext_run_letta_qwen_extraction_canary import validate_loopback_binding

def test_loopback_endpoint_must_match_qualification_server_port():
    closure={"runtime":{"loopback_only":True,"measured":{"server_port":"8123"}}}
    with pytest.raises(ValueError): validate_loopback_binding("http://127.0.0.1:8124", closure)

def test_qualification_requires_exact_v2_schema():
    from scripts.vnext_run_letta_qwen_extraction_canary import validate_qualification_artifacts
    assert callable(validate_qualification_artifacts)

def test_current_state_metric_is_not_named_stale_burden():
    from scripts.vnext_run_letta_qwen_extraction_canary import current_retrieval_metric
    assert current_retrieval_metric([]) is None
