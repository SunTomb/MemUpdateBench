from pathlib import Path
import pytest
from scripts.vnext_run_letta_qwen_extraction_canary import validate_qualification_artifacts, validate_output_root, validate_extraction, canonical_json_bytes

def test_missing_qualification_validator_is_callable(tmp_path):
    with pytest.raises(ValueError):
        validate_qualification_artifacts(tmp_path)

def test_qualification_requires_schema_and_affirmative_gates(tmp_path):
    for name in ('letta_runtime_qualification.json','letta_runtime_preflight.json','letta_runtime_admission.json'):
        (tmp_path/name).write_text('{}')
    with pytest.raises(ValueError):
        validate_qualification_artifacts(tmp_path)

def test_output_root_rejects_frozen_overlap(tmp_path):
    frozen=tmp_path/'frozen'; frozen.mkdir()
    with pytest.raises(ValueError): validate_output_root(frozen/'out', frozen_roots=(frozen,))

def test_qualification_hashes_are_mandatory(tmp_path):
    closure = {"schema_version":"memupdatebench.external.letta.runtime_qualification.v1","candidate_id":"letta_0_16_8_song1_local_linux","outcome":"PASS","identity":{"package_name":"letta","package_version":"0.16.8","source_commit":"1131535716e8a31c9a437f8695e25ac98f203a24"},"source":{},"project_source":{},"runner_source_sha256":"a"*64,"runtime":{"loopback_only":True},"boundary":{"llm_used":False,"api_used":False,"gpu_used":False},"cleanup":{"status":"PASS"},"preflight":{},"admission":{}}
    preflight = {"schema_version":"memupdatebench.external.letta.preflight.v2","candidate_id":"letta_0_16_8_profile","mode":"profile_single_record_runtime","outcome":"pass","passed":True,"identity":{},"official_health":{},"runtime":{},"namespace_reset_probe":{},"lifecycle":{},"clean_close":{},"security":{},"boundary":{},"unsupported":{}}
    admission = {"schema_version":"memupdatebench.external.letta.admission.v2","candidate_id":"letta_0_16_8_profile","admission_scope":"profile_single_record_runtime","outcome":"pass","admitted":True,"gates":{},"reasons":[]}
    for name, value in (("letta_runtime_qualification.json",closure),("letta_runtime_preflight.json",preflight),("letta_runtime_admission.json",admission)):
        (tmp_path/name).write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="hash"):
        validate_qualification_artifacts(tmp_path)
