from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema, Split
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from scripts import vnext_run_main_track_muse_answer_baseline as module


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
AUDIT_ATTESTATION = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


def test_loopback_url_rejects_credentials_and_remote_hosts() -> None:
    assert module.validate_loopback_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    for url in ("https://127.0.0.1:8080", "http://localhost:8080?token=x", "http://user:pass@127.0.0.1:8080", "http://example.com:8080"):
        with pytest.raises(ValueError):
            module.validate_loopback_url(url)


def test_muse_http_adapter_parses_final_content_and_redacts_reasoning(monkeypatch) -> None:
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": module.MUSE_MODEL_ID,
                "choices": [{
                    "message": {"role": "assistant", "content": '{"disposition":"answered","answer":"Paris"}', "reasoning_content": "private chain"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }).encode()

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    model = module.MuseGlimmerAnswerModel("http://127.0.0.1:8080", model_name="muse")
    request = SimpleNamespace(query=SimpleNamespace(query_id="q1", answer_schema=AnswerSchema.STRING), rendered_prompt="prompt")
    prediction = model.answer(request)

    assert prediction.parsed_answer == "Paris"
    assert prediction.raw_output == '{"disposition":"answered","answer":"Paris"}'
    assert "reasoning_content" not in model.last_answer_metadata
    assert "raw_output" not in model.last_answer_metadata
    assert model.last_answer_metadata["reasoning_sha256"]
    assert model.last_answer_metadata["content_sha256"]
    payload = json.loads(requests[0][0].data)
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["seed"] == 0
    assert payload["max_tokens"] == 2048
    assert payload["stream"] is False
    assert requests[0][0].full_url == "http://127.0.0.1:8080/v1/chat/completions"


def test_canary_selection_is_deterministic_and_stratified() -> None:
    candidate = module.load_candidate(CANDIDATE_ROOT)
    selected = module.select_tasks(candidate, module.CANARY_SCOPE)
    assert len(selected) == 32
    assert tuple(task.task_id for task in selected) == tuple(task.task_id for task in module.select_tasks(candidate, module.CANARY_SCOPE))
    assert tuple(task.task_id for task in selected) != tuple(task.task_id for task in candidate.test_tasks[:32])
    assert {task.task_family for task in selected} >= {"interleaved_multi_slot_update", "entity_attribute_grounding", "noop_write_discipline"}
    assert {task.metadata.extra["language"] for task in selected} >= {"en", "es", "ja"}
    c_tasks = [task for task in selected if task.task_family == "entity_attribute_grounding"]
    assert any(task.gold_evidence[0].disposition is AnswerDisposition.ANSWERED for task in c_tasks)
    assert any(task.gold_evidence[0].disposition is AnswerDisposition.ABSTAINED for task in c_tasks)
    assert all(task.metadata.split is Split.TEST for task in selected)


def test_c_abstention_scoring_is_typed() -> None:
    gold = SimpleNamespace(disposition=AnswerDisposition.ABSTAINED, answer=None)
    query = SimpleNamespace(query_id="q1")
    prediction = AnswerPredictionV3(query_id="q1", raw_output='{"disposition":"abstained"}', disposition=AnswerDisposition.ABSTAINED, format_valid=True)
    scored = module.score_prediction(query, prediction, gold)
    assert scored["answer_outcome"] == "CORRECT_ABSTENTION"
    assert scored["typed_match"] is True


def test_attestation_binds_candidate_and_tampering_fails(tmp_path: Path) -> None:
    candidate = module.load_candidate(CANDIDATE_ROOT)
    attestation = module.validate_audit_attestation(AUDIT_ATTESTATION, candidate)
    assert attestation.review_status == "PASS"
    raw = json.loads(AUDIT_ATTESTATION.read_bytes())
    raw["candidate_artifact_hashes"] = dict(raw["candidate_artifact_hashes"])
    raw["candidate_artifact_hashes"]["tasks.jsonl"] = "0" * 64
    forged = tmp_path / "forged.json"
    forged.write_bytes(module.canonical_bytes(raw))
    with pytest.raises(ValueError, match="candidate"):
        module.validate_audit_attestation(forged, candidate)


def test_publication_is_no_replace_and_tamper_checked(tmp_path: Path) -> None:
    candidate = module.load_candidate(CANDIDATE_ROOT)
    out = tmp_path / "out"
    row = {"task_id": candidate.test_tasks[0].task_id}
    summary = {"scope": module.CANARY_SCOPE, "candidate_artifact_hashes": candidate.artifact_hashes, "model_binding": module.muse_model_binding()}
    module.publish_muse_answer_baseline(candidate, [row], summary, out)
    assert {path.name for path in out.iterdir()} == {"rows.jsonl", "summary.json", "artifact_index.json"}
    with pytest.raises(FileExistsError):
        module.publish_muse_answer_baseline(candidate, [row], summary, out)

    tampered = tmp_path / "tampered"
    original = (candidate.root / "tasks.jsonl").read_bytes()
    try:
        (candidate.root / "tasks.jsonl").write_bytes(original + b"\\n")
        with pytest.raises(ValueError, match="candidate"):
            module.publish_muse_answer_baseline(candidate, [row], summary, tampered)
        assert not tampered.exists() or not any(tampered.iterdir())
    finally:
        (candidate.root / "tasks.jsonl").write_bytes(original)


def test_muse_identity_and_qualification_fields_are_bound(tmp_path: Path) -> None:
    binding = module.muse_model_binding()
    assert binding["model_id"] == module.MUSE_MODEL_ID
    assert binding["revision"] == "70bf1b61ac09f91b24d39038091b41c582bc5d7a"
    assert module.MUSE_MODEL_FILE in binding["model_file"]
    assert binding["speculative_decoding"] is False
    assert binding["reasoning_mode"] == "off"
    assert binding["reasoning_storage"] == "sha256_only"
    assert binding["old_32_token_smoke_status"] == "BLOCKED"
    assert binding["max_tokens"] == 2048

    candidate = module.load_candidate(CANDIDATE_ROOT)
    task = candidate.test_tasks[0]
    row = {
        "task_id": task.task_id, "family": task.task_family, "domain": task.metadata.extra["domain"],
        "attribute": task.metadata.extra["attribute"], "language": task.metadata.extra["language"],
        "expected_disposition": "answered", "answer_outcome": "CORRECT", "exact_match": True,
        "normalized_match": True, "typed_match": True, "typed_exact_match": True, "answer_f1": 1.0,
    }
    summary = module.build_summary([row], scope=module.CANARY_SCOPE, candidate=candidate, model_binding=binding)
    assert summary["model_binding"]["model_id"] == module.MUSE_MODEL_ID
    assert summary["qualification"]["old_32_token_smoke_status"] == "BLOCKED"
    output = tmp_path / "identity"
    module.publish_muse_answer_baseline(candidate, [row], summary, output)
    index = json.loads((output / "artifact_index.json").read_bytes())
    assert index["model_binding"]["model_id"] == module.MUSE_MODEL_ID
    assert index["evidence_class"] == "answer_layer_reference_state"
    assert index["qualification"]["reasoning_storage"] == "sha256_only"
