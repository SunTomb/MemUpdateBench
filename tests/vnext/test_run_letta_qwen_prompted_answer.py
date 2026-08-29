from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema, EvaluationMode
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3, MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryQueryV3
from mub.vnext.contracts.common import MemoryObjectKey


def _query(query_id: str = "q-1") -> MemoryQueryV3:
    return MemoryQueryV3(
        query_id=query_id,
        query_type="current",
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city"),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )


def _trace(query: MemoryQueryV3) -> RetrievalTraceV3:
    return RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=(
            MemoryEntryRecordV3(entry_id="entry-1", content="Alice lives in Paris.", value_candidate="Paris"),
        ),
    )


def test_prompt_request_uses_only_actual_retrieval_entries_and_binds_hash() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_prompted_answer_request

    query = _query()
    trace = _trace(query)
    request = build_prompted_answer_request(query, trace)

    assert request.retrieval_trace.retrieved_entries == trace.retrieved_entries
    assert "Alice lives in Paris." in request.rendered_prompt
    assert "gold" not in request.rendered_prompt.lower()
    assert request.prompt_hash == hashlib.sha256(request.rendered_prompt.encode()).hexdigest()


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (AnswerPredictionV3(query_id="q-1", raw_output='{"answer":"Rome"}', parsed_answer="Rome", format_valid=True), "WRONG"),
        (AnswerPredictionV3(query_id="q-1", raw_output="not json", format_valid=False, error_flags=("answer_json_invalid",)), "FORMAT_INVALID"),
        (AnswerPredictionV3(query_id="q-1", raw_output="", disposition=AnswerDisposition.UNAVAILABLE, format_valid=False, error_flags=("model_unavailable",)), "UNAVAILABLE"),
        (AnswerPredictionV3(query_id="q-1", raw_output='{"disposition":"abstained"}', disposition=AnswerDisposition.ABSTAINED, format_valid=True), "UNAVAILABLE"),
    ],
)
def test_answer_outcome_distinguishes_wrong_format_invalid_and_unavailable(prediction, expected) -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import classify_answer_prediction

    assert classify_answer_prediction(prediction, "Paris") == expected


def test_summary_has_separate_denominators_and_joint_pipeline_label() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_summary

    rows = [
        {"task_id": "a", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "answer_outcome": "CORRECT", "final_memory_size": 1},
        {"task_id": "b", "status": "PASS", "state_accuracy": False, "gold_retrieved_k16": True, "answer_outcome": "WRONG", "final_memory_size": 2},
        {"task_id": "c", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": False, "answer_outcome": "FORMAT_INVALID", "final_memory_size": 1},
        {"task_id": "d", "status": "NOT_SUPPORTED", "state_accuracy": None, "gold_retrieved_k16": None, "answer_outcome": None, "final_memory_size": None},
    ]

    summary = build_summary(rows, scope="full-family-a80", requested=4, rows_sha256="a" * 64,
                            qualification_hashes={}, qualification_identity={}, letta_binding={},
                            endpoint="http://127.0.0.1:8000", model_provenance={})

    assert summary["supported"] == 3
    assert summary["unsupported"] == 1
    assert summary["state_accuracy_denominator"] == 3
    assert summary["gold_retrieval_denominator"] == 3
    assert summary["answer_metrics_denominator"] == 3
    assert summary["answer_outcome_counts"] == {"CORRECT": 1, "WRONG": 1, "FORMAT_INVALID": 1, "UNAVAILABLE": 0}
    assert summary["evidence_class"] == "joint_pipeline"
    assert summary["provider_calls"] == summary["api_calls"] == summary["retries"] == 0


def test_finalize_rows_requires_exact_ordered_one_row_per_task() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import finalize_rows

    tasks = [SimpleNamespace(task_id="task-a"), SimpleNamespace(task_id="task-b")]
    assert finalize_rows(tasks, [{"task_id": "task-a"}, {"task_id": "task-b"}]) == [
        {"task_id": "task-a"}, {"task_id": "task-b"}
    ]
    with pytest.raises(ValueError, match="order"):
        finalize_rows(tasks, [{"task_id": "task-b"}, {"task_id": "task-a"}])
    with pytest.raises(ValueError, match="exactly one"):
        finalize_rows(tasks, [{"task_id": "task-a"}, {"task_id": "task-a"}])


def test_adapter_construction_converts_target_to_frozen_key(monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module
    from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey

    class Config:
        def model_dump(self, mode="json"):
            return {"run_id": "test"}

    captured = {}
    monkeypatch.setattr(module, "build_letta_adapter_configuration", lambda run_id: Config())
    monkeypatch.setattr(module, "build_worker_command", lambda *args: ("python", "worker"))
    monkeypatch.setattr(module, "safe_worker_environment", lambda *args: {})
    monkeypatch.setattr(module, "JsonlSubprocessBridge", lambda **kwargs: object())

    class Adapter:
        def __init__(self, *, target_objects, **kwargs):
            captured["target_objects"] = target_objects

    monkeypatch.setattr(module, "LettaExternalAdapterV3", Adapter)
    task = SimpleNamespace(
        task_id="task-a",
        target_objects=(MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city"),),
    )
    args = SimpleNamespace(letta_python_executable="python", letta_project_root="project")
    adapter = module._adapter_for_task(args, task, {"project_root": "project"}, "http://127.0.0.1:8000")
    assert isinstance(captured["target_objects"][0], FrozenMemoryObjectKey)
    assert adapter is not None


def test_answer_comparison_is_typed_and_counter_based() -> None:
    from mub.vnext.contracts.enums import AnswerDisposition
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from scripts.vnext_run_letta_qwen_prompted_answer import _answer_f1, classify_answer_prediction

    typed = AnswerPredictionV3(
        query_id="q-1", raw_output="1", parsed_answer=1, format_valid=True,
        disposition=AnswerDisposition.ANSWERED,
    )
    assert classify_answer_prediction(typed, True) == "WRONG"
    repeated = AnswerPredictionV3(
        query_id="q-1", raw_output="red red blue", parsed_answer="red red blue",
        format_valid=True, disposition=AnswerDisposition.ANSWERED,
    )
    assert _answer_f1(repeated, "red blue blue") == pytest.approx(2 / 3)
    invalid = AnswerPredictionV3(
        query_id="q-1", raw_output="bad", format_valid=False,
        disposition=AnswerDisposition.ANSWERED,
    )
    unavailable = AnswerPredictionV3(
        query_id="q-1", raw_output="", format_valid=False,
        disposition=AnswerDisposition.UNAVAILABLE,
    )
    assert _answer_f1(invalid, "Paris") == 0.0
    assert _answer_f1(unavailable, "Paris") == 0.0


def test_summary_exposes_attempted_and_evaluable_answer_denominators() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_summary

    rows = [
        {"task_id": "a", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "answer_outcome": "CORRECT", "answer_f1": 1.0, "final_memory_size": 1},
        {"task_id": "b", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "answer_outcome": "WRONG", "answer_f1": 0.0, "final_memory_size": 1},
        {"task_id": "c", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "answer_outcome": "FORMAT_INVALID", "answer_f1": 0.0, "final_memory_size": 1},
        {"task_id": "d", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True, "answer_outcome": "UNAVAILABLE", "answer_f1": 0.0, "final_memory_size": 1},
    ]
    summary = build_summary(rows, scope="canary32", requested=4, rows_sha256="a" * 64,
                            qualification_hashes={}, qualification_identity={}, letta_binding={},
                            endpoint="http://127.0.0.1:8000", model_provenance={})
    assert summary["answer_attempted_denominator"] == 4
    assert summary["answer_evaluable_denominator"] == 2
    assert summary["prompted_answer_em"] == pytest.approx(0.25)
    assert summary["prompted_answer_f1"] == pytest.approx(0.25)
    assert summary["llm_roles"] == ["visible_event_crud_extraction", "retrieved_context_prompted_answer"]


def test_qwen_answer_metadata_binds_chat_prompt_and_raw_output_without_raw_text() -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Tokenizer:
        chat_template = "template-v1"

        def apply_chat_template(self, messages, **kwargs):
            return "CHAT:" + messages[0]["content"]

    session = module.QwenSession.__new__(module.QwenSession)
    session.tokenizer = Tokenizer()
    session._generate = lambda rendered, max_new_tokens: ('{"disposition":"answered","answer":"Paris"}', 7, 12.5)
    request = module.PromptedAnswerRequestV3(query=_query(), retrieval_trace=_trace(_query()).model_copy(update={"prompt_hash": hashlib.sha256(module.render_visible_prompt_v3(query=_query(), retrieval_trace=_trace(_query())).encode()).hexdigest()}), rendered_prompt=module.render_visible_prompt_v3(query=_query(), retrieval_trace=_trace(_query())), prompt_hash=hashlib.sha256(module.render_visible_prompt_v3(query=_query(), retrieval_trace=_trace(_query())).encode()).hexdigest())
    prediction = session.answer(request)
    metadata = session.last_answer_metadata
    assert metadata["rendered_chat_prompt_sha256"] == hashlib.sha256(("CHAT:" + request.rendered_prompt).encode()).hexdigest()
    assert metadata["raw_output_sha256"] == hashlib.sha256(b'{"disposition":"answered","answer":"Paris"}').hexdigest()
    assert metadata["generated_tokens"] == 7
    assert metadata["latency_ms"] == 12.5
    assert metadata["chat_template_sha256"] == hashlib.sha256(b"template-v1").hexdigest()
    assert "raw_output" not in metadata and "rendered_prompt" not in metadata
    assert prediction.parsed_answer == "Paris"
