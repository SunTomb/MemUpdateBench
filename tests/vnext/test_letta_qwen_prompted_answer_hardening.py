from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest

from mub.vnext.contracts.enums import AnswerSchema, EvaluationMode
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryQueryV3
from mub.vnext.contracts.common import MemoryObjectKey


def test_prompt_request_rejects_trace_query_mismatch() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_prompted_answer_request

    query = MemoryQueryV3(
        query_id="query-a", query_type="current", text="Question", selector=CurrentSelector(),
        target_object_keys=(MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city"),), answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    with pytest.raises(ValueError):
        build_prompted_answer_request(query, RetrievalTraceV3(query_id="query-b"))


def test_summary_rejects_missing_terminal_rows() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import validate_terminal_rows

    with pytest.raises(ValueError, match="exactly 80"):
        validate_terminal_rows(
            [{"task_id": f"task-{index}", "status": "PASS"} for index in range(79)],
            scope="full-family-a80",
        )


def test_summary_does_not_treat_unsupported_as_failed_or_answer_denominator() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_summary

    rows = [
        *({"task_id": f"supported-{i}", "status": "PASS", "state_accuracy": True,
           "gold_retrieved_k16": True, "answer_outcome": "CORRECT", "final_memory_size": 1} for i in range(52)),
        *({"task_id": f"unsupported-{i}", "status": "NOT_SUPPORTED", "state_accuracy": None,
           "gold_retrieved_k16": None, "answer_outcome": None, "final_memory_size": None} for i in range(28)),
    ]
    summary = build_summary(list(rows), scope="full-family-a80", requested=80, rows_sha256="b" * 64,
                            qualification_hashes={}, qualification_identity={}, letta_binding={},
                            endpoint="http://127.0.0.1:8000", model_provenance={})
    assert summary["supported"] == 52
    assert summary["unsupported"] == 28
    assert summary["state_accuracy_denominator"] == 52
    assert summary["gold_retrieval_denominator"] == 52
    assert summary["answer_metrics_denominator"] == 52


def test_secret_scan_and_no_replace_artifact_writer_are_required(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import write_artifact_no_replace

    output = tmp_path / "artifact.json"
    digest = write_artifact_no_replace(output, {"safe": True})
    assert output.exists()
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_artifact_no_replace(output, {"safe": False})
    with pytest.raises(ValueError, match="secret"):
        write_artifact_no_replace(tmp_path / "secret.json", {"api_key": "sk-test-secret"})




def test_support_requires_one_target_and_one_retrieved_prompt_query() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import task_support_reason

    query = MemoryQueryV3(
        query_id="q", query_type="current", text="Where?", selector=CurrentSelector(),
        target_object_keys=(MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city"),),
        answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    one = type("Task", (), {"target_objects": (query.target_object_keys[0],), "queries": (query,)})()
    assert task_support_reason(one) is None
    no_query = type("Task", (), {"target_objects": (query.target_object_keys[0],), "queries": ()})()
    assert task_support_reason(no_query) is not None
    wrong_mode = query.model_copy(update={"evaluation_mode": EvaluationMode.STATE_DIRECT})
    wrong = type("Task", (), {"target_objects": (query.target_object_keys[0],), "queries": (wrong_mode,)})()
    assert task_support_reason(wrong) is not None
    multi = type("Task", (), {"target_objects": (query.target_object_keys[0], query.target_object_keys[0].model_copy(update={"entity": "bob"})), "queries": (query,)})()
    assert task_support_reason(multi) is not None


def test_offline_environment_overwrites_bad_values(monkeypatch, tmp_path: Path) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    module.safe_offline_environment()
    assert module.os.environ["HF_HUB_OFFLINE"] == "1"
    assert module.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert module.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_summary_has_no_non_verifiable_self_hash() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import build_summary, canonical_json_bytes, sha256_bytes

    summary = build_summary([], scope="canary32", requested=0, rows_sha256="a" * 64,
                            qualification_hashes={}, qualification_identity={}, letta_binding={},
                            endpoint="http://127.0.0.1:8000", model_provenance={})
    assert "payload_sha256" not in summary
    assert sha256_bytes(canonical_json_bytes(summary)) != summary.get("payload_sha256")


def test_cli_defaults_to_full_scope(monkeypatch, capsys) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    monkeypatch.setattr(module, "run", lambda args: {"outcome": "PASS", "scope": args.scope})
    assert module.main([
        "--tasks", "tasks.jsonl", "--output-root", "out", "--qualification-root", "qual",
        "--letta-base-url", "http://127.0.0.1:8000", "--letta-python-executable", "python",
        "--letta-project-root", "project", "--model-snapshot", "snapshot",
        "--model-runtime-receipt", "runtime.json", "--model-snapshot-binding", "binding.json",
    ]) == 0
    assert '"scope": "full-family-a80"' in capsys.readouterr().out



def test_injected_model_pair_lifecycle_closes_model_when_extractor_load_fails() -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Model:
        def __init__(self): self.loaded = 0; self.closed = 0
        def load(self): self.loaded += 1
        def close(self): self.closed += 1
    class Extractor:
        def __init__(self): self.loaded = 0
        def load(self): self.loaded += 1; raise RuntimeError("extractor blocked")

    model, extractor = Model(), Extractor()
    with pytest.raises(RuntimeError, match="extractor blocked"):
        module._load_model_pair(model, extractor)
    assert (model.loaded, model.closed, extractor.loaded) == (1, 1, 1)


def test_full_scope_run_binds_selection_order_support_and_index(monkeypatch, tmp_path: Path) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module
    from mub.vnext.contracts.enums import AnswerDisposition, Operation
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

    key = MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city")
    query = MemoryQueryV3(
        query_id="q", query_type="current", text="Where does Alice live?", selector=CurrentSelector(),
        target_object_keys=(key,), answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    class Event:
        def __init__(self, event_id): self.event_id = event_id; self.raw_text = "Alice lives in Paris."; self.timestamp = None
        def model_copy(self, update): return self
    def task(task_id, supported):
        return type("Task", (), {
            "task_id": task_id, "target_objects": (key,) if supported else (key, key.model_copy(update={"entity": "bob"})),
            "queries": (query,), "events": (Event("event-1"), Event("event-2")), "gold_evidence": (type("Gold", (), {"answer": "Paris"})(),),
            "metadata": type("Meta", (), {"extra": {"semantic_core_id": task_id}})(),
        })()
    tasks = [task(f"task-{i:02d}", i < 52) for i in range(80)]
    raw = b"authenticated-full-scope"
    monkeypatch.setattr(module.extraction, "TASK_SHA256", module.sha256_bytes(raw))
    monkeypatch.setattr(module, "TASK_SHA256", module.sha256_bytes(raw))
    monkeypatch.setattr(module.extraction, "_parse_authenticated_tasks", lambda payload: list(reversed(tasks)))
    monkeypatch.setattr(module, "validate_output_root", lambda path, frozen_roots=(): tmp_path / "run")
    monkeypatch.setattr(module, "verify_model_provenance", lambda *args, **kwargs: {
        "snapshot": "/snapshot", "tree_sha256": "b" * 64, "snapshot_binding": {"receipt_file_sha256": "c" * 64, "receipt_payload_sha256": "d" * 64},
        "runtime_receipt_sha256": "e" * 64, "runtime_identity": "runtime.v1",
    })
    qualification = {"closure": {"identity": {}, "source": {}, "project_source": {}, "runtime": {}}, "hashes": {"closure": "f" * 64}}
    monkeypatch.setattr(module, "validate_qualification_artifacts", lambda root: qualification)
    worker = {"project_root": "project", "worker_source_sha256": "1" * 64, "runner_source_sha256": "2" * 64}
    monkeypatch.setattr(module, "validate_worker_runtime_binding", lambda *args, **kwargs: worker)
    monkeypatch.setattr(module, "validate_loopback_binding", lambda url, closure: url)
    monkeypatch.setattr(module, "build_letta_adapter_configuration", lambda run_id: type("Config", (), {"model_dump": lambda self, mode="json": {"run_id": run_id}})())
    monkeypatch.setattr(module, "compute_letta_configuration_hash", lambda config: "3" * 64)

    class Extractor:
        def load(self): pass
        def extract(self, raw_text, attribute): return {"operation": "update", "value": "Paris"}, "{}", 1, 1.0
        def close(self): pass
    class Answer:
        def load(self): pass
        def answer(self, request):
            self.last_answer_metadata = {
                "rendered_chat_prompt_sha256": "4" * 64,
                "raw_output_sha256": "5" * 64,
                "generated_tokens": 3,
                "latency_ms": 2.0,
                "chat_template_sha256": "6" * 64,
            }
            return AnswerPredictionV3(query_id=request.query.query_id, raw_output='{"disposition":"answered","answer":"Paris"}', parsed_answer="Paris", format_valid=True, disposition=AnswerDisposition.ANSWERED)
        def close(self): pass
    class Adapter:
        export_calls = 0
        def __init__(self, task): self.task = task
        def reset(self, request): pass
        def ingest_event(self, event):
            return type("Result", (), {"effective_action": type("Action", (), {"operation": Operation.ADD})(), "affected_entry_ids": ("entry",)})()
        def export_entries(self):
            type(self).export_calls += 1
            return type("Export", (), {"entries": (MemoryEntryRecordV3(entry_id="entry", content="Paris", value_candidate="Paris"),)})()
        def retrieve(self, request):
            return type("Retrieval", (), {"trace": RetrievalTraceV3(query_id="q", retrieved_entries=(MemoryEntryRecordV3(entry_id="entry", content="Paris", value_candidate="Paris"),))})()
        def close(self): pass

    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(module.canonical_json_bytes({}))
    args = type("Args", (), {"tasks": str(tmp_path / "tasks"), "output_root": str(tmp_path / "out"), "qualification_root": str(tmp_path / "qual"), "letta_base_url": "http://127.0.0.1:8000", "letta_python_executable": "python", "letta_project_root": "project", "model_snapshot": "snapshot", "model_runtime_receipt": "runtime.json", "model_snapshot_binding": str(binding_path), "scope": module.FULL_SCOPE})()
    Path(args.tasks).write_bytes(raw)
    result = module.run(args, extractor_factory=Extractor, adapter_factory=lambda t: Adapter(t), answer_model_factory=Answer)
    assert result["requested"] == 80 and result["supported"] == 52 and result["unsupported"] == 28
    assert Adapter.export_calls == 2 * 52
    assert result["llm_roles"] == ["visible_event_crud_extraction", "retrieved_context_prompted_answer"]
    rows = [module.json.loads(line) for line in (tmp_path / "run" / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["retrieval_trace_sha256"] == module._canonical_model_sha256(RetrievalTraceV3(query_id="q", retrieved_entries=(MemoryEntryRecordV3(entry_id="entry", content="Paris", value_candidate="Paris"),), prompt_hash=rows[0]["visible_prompt_sha256"]))
    assert rows[0]["visible_prompt_sha256"]
    assert rows[0]["answer_metadata"] == {"rendered_chat_prompt_sha256": "4" * 64, "raw_output_sha256": "5" * 64, "generated_tokens": 3, "latency_ms": 2.0, "chat_template_sha256": "6" * 64}
    unsupported_rows = [row for row in rows if row["status"] == "NOT_SUPPORTED"]
    assert len(unsupported_rows) == 28
    assert all(row[field] is None for row in unsupported_rows for field in ("state_accuracy", "gold_retrieved_k16", "answer_outcome", "final_memory_size"))
    index = module.json.loads((tmp_path / "run" / "artifact_index.json").read_text())
    assert index["task_view_sha256"] == module.TASK_SHA256
    assert index["model"]["tree_sha256"] == "b" * 64
    assert index["model"]["runtime_receipt_sha256"] == "e" * 64
    assert index["qualification_hashes"] == qualification["hashes"]
    assert index["qualification_identity"] == {"package": {}, "source": {}, "project_source": {}, "runtime": {}}
    assert index["worker_binding"] == worker
    assert index["runner_source_sha256"] == module.sha256_bytes(module.Path(module.__file__).read_bytes())
    assert index["llm_roles"] == ["visible_event_crud_extraction", "retrieved_context_prompted_answer"]
    assert index["provider_calls"] == index["api_calls"] == index["retries"] == 0


def test_answer_role_must_consume_prompt_request_not_exported_state() -> None:
    from mub.vnext.contracts.enums import AnswerDisposition
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from scripts.vnext_run_letta_qwen_prompted_answer import answer_from_retrieval

    class FakeAnswerModel:
        def answer(self, request):
            assert request.retrieval_trace.retrieved_entries[0].value_candidate == "Paris"
            assert "Paris" in request.rendered_prompt
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output='{"disposition":"answered","answer":"Paris"}',
                parsed_answer="Paris", format_valid=True,
                disposition=AnswerDisposition.ANSWERED,
            )

    query = MemoryQueryV3(
        query_id="q", query_type="current", text="Where?", selector=CurrentSelector(),
        target_object_keys=(MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city"),), answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    trace = RetrievalTraceV3(
        query_id="q",
        retrieved_entries=(MemoryEntryRecordV3(entry_id="entry", value_candidate="Paris", content="Paris"),),
    )
    prediction = answer_from_retrieval(FakeAnswerModel(), query, trace)
    assert prediction.parsed_answer == "Paris"


def test_model_pair_closes_model_when_model_load_raises() -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Model:
        def __init__(self): self.closed = 0
        def load(self): raise RuntimeError("model blocked")
        def close(self): self.closed += 1

    class Extractor:
        def load(self): raise AssertionError("extractor must not load")
        def close(self): pass

    model = Model()
    with pytest.raises(RuntimeError, match="model blocked"):
        module._load_model_pair(model, Extractor())
    assert model.closed == 1


def test_qwen_load_closes_partial_resources_when_model_load_raises(monkeypatch, tmp_path: Path) -> None:
    import sys
    import types
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Cuda:
        def __init__(self): self.empty = 0; self.sync = 0
        def is_available(self): return True
        def empty_cache(self): self.empty += 1
        def synchronize(self): self.sync += 1

    fake_torch = types.SimpleNamespace(
        cuda=Cuda(), bfloat16=object(), manual_seed=lambda value: None,
        use_deterministic_algorithms=lambda value: None,
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True, deterministic=False)),
    )

    class Tokenizer:
        def __init__(self): self.closed = 0
        def close(self): self.closed += 1

    tokenizer = Tokenizer()
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs): return tokenizer
    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs): raise RuntimeError("partial model load")
    fake_transformers = types.SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    session = module.QwenSession.__new__(module.QwenSession)
    session.snapshot = tmp_path
    session.model = None
    session.tokenizer = None
    session.torch = None
    session.last_answer_metadata = {}
    with pytest.raises(RuntimeError, match="partial model load"):
        session.load()
    assert tokenizer.closed == 1
    assert fake_torch.cuda.empty == 1
    assert fake_torch.cuda.sync == 1
    assert session.model is None and session.tokenizer is None and session.torch is None


def test_qwen_load_closes_model_when_eval_raises(monkeypatch, tmp_path: Path) -> None:
    import sys
    import types
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Cuda:
        def is_available(self): return False
    fake_torch = types.SimpleNamespace(
        cuda=Cuda(), bfloat16=object(), manual_seed=lambda value: None,
        use_deterministic_algorithms=lambda value: None,
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True, deterministic=False)),
    )
    class Resource:
        def __init__(self): self.closed = 0
        def close(self): self.closed += 1
    tokenizer = Resource()
    model = Resource()
    class ModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs): return model
    model.eval = lambda: (_ for _ in ()).throw(RuntimeError("eval blocked"))
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForCausalLM=ModelLoader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    session = module.QwenSession.__new__(module.QwenSession)
    session.snapshot = tmp_path
    session.model = None
    session.tokenizer = None
    session.torch = None
    session.last_answer_metadata = {}
    with pytest.raises(RuntimeError, match="eval blocked"):
        session.load()
    assert model.closed == 1 and tokenizer.closed == 1 and session.model is None


def test_rows_claim_is_exclusive_and_does_not_append_to_existing_file(tmp_path: Path) -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import claim_rows_file

    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"preexisting\n")
    with pytest.raises(FileExistsError):
        claim_rows_file(rows)
    assert rows.read_bytes() == b"preexisting\n"


def test_prompt_metadata_hashes_plain_and_chat_renderings_separately() -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    class Tokenizer:
        chat_template = "template-v1"
        def apply_chat_template(self, messages, **kwargs):
            return "CHAT PREFIX\\n" + messages[0]["content"]

    session = module.QwenSession.__new__(module.QwenSession)
    session.tokenizer = Tokenizer()
    session._generate = lambda rendered, max_new_tokens: ('{"answer":"Paris"}', 1, 1.0)
    key = MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city")
    query = MemoryQueryV3(
        query_id="q-metadata", query_type="current", text="Where does Alice live?", selector=CurrentSelector(),
        target_object_keys=(key,), answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    trace = RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=(MemoryEntryRecordV3(entry_id="entry", content="Alice lives in Paris.", value_candidate="Paris"),),
    )
    request = module.build_prompted_answer_request(query, trace)
    session.answer(request)
    metadata = session.last_answer_metadata
    assert metadata["rendered_prompt_sha256"] == hashlib.sha256(request.rendered_prompt.encode()).hexdigest()
    assert metadata["rendered_chat_prompt_sha256"] == hashlib.sha256(("CHAT PREFIX\\n" + request.rendered_prompt).encode()).hexdigest()
    assert metadata["rendered_prompt_sha256"] != metadata["rendered_chat_prompt_sha256"]


def _valid_pass_row(module):
    row = {
        "row_schema_version": module.ROW_SCHEMA_VERSION,
        "task_id": "task-1",
        "semantic_core_id": "core-1",
        "status": "PASS",
        "reason": None,
        "support_status": None,
        "error_class": None,
        "stage": None,
        "error_detail": None,
        **{field: None for field in module.ROW_NULLABLE_FIELDS},
    }
    row.update(
        {
            "parsed_final_value": "Paris",
            "stable_entry_id": True,
            "stale_retrieved_k16": 0,
            "retrieval_trace_sha256": "a" * 64,
            "visible_prompt_sha256": "b" * 64,
            "prompted_answer": "Paris",
            "prompted_exact_match": True,
            "answer_outcome": "CORRECT",
            "answer_f1": 1.0,
            "answer_format_valid": True,
            "answer_disposition": "answered",
            "answer_error_flags": (),
            "answer_output_sha256": "c" * 64,
            "answer_metadata": {},
            "final_memory_size": 1,
            "state_accuracy": True,
            "gold_retrieved_k16": True,
            "gold_sha256": "d" * 64,
            "extractions": [],
            "reconciliation_count": 0,
            "affected_entry_ids": ("entry-1",),
            "latency_ms": 1.0,
        }
    )
    return row


@pytest.mark.parametrize(
    "field",
    [
        "state_accuracy",
        "final_memory_size",
        "stable_entry_id",
        "gold_retrieved_k16",
        "stale_retrieved_k16",
        "retrieval_trace_sha256",
        "visible_prompt_sha256",
        "prompted_exact_match",
        "answer_outcome",
        "answer_f1",
        "answer_format_valid",
        "answer_disposition",
        "answer_output_sha256",
        "answer_metadata",
        "gold_sha256",
        "extractions",
        "reconciliation_count",
        "affected_entry_ids",
        "latency_ms",
    ],
)
def test_pass_rows_require_non_null_completed_evidence(field: str) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    row = _valid_pass_row(module)
    row[field] = None
    with pytest.raises(ValueError, match="PASS rows require non-null"):
        module.validate_row_shape(row, execution_mode="injected_test_only")


@pytest.mark.parametrize(
    "field",
    ["retrieval_trace_sha256", "visible_prompt_sha256", "answer_output_sha256", "gold_sha256"],
)
def test_pass_rows_require_lowercase_sha256_evidence_hashes(field: str) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    row = _valid_pass_row(module)
    row[field] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.validate_row_shape(row, execution_mode="injected_test_only")


@pytest.mark.parametrize(
    "updates",
    [
        {"answer_outcome": "CORRECT", "prompted_exact_match": False},
        {"answer_outcome": "WRONG", "prompted_exact_match": True},
        {"answer_outcome": "CORRECT", "answer_format_valid": False},
        {"answer_outcome": "FORMAT_INVALID", "answer_format_valid": True},
        {"answer_outcome": "UNAVAILABLE", "answer_disposition": "answered"},
    ],
)
def test_pass_rows_require_basic_answer_outcome_coherence(updates: dict) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    row = _valid_pass_row(module)
    row.update(updates)
    with pytest.raises(ValueError, match="answer outcome"):
        module.validate_row_shape(row, execution_mode="injected_test_only")


@pytest.mark.parametrize(
    ("extractor_factory", "adapter_factory", "answer_model_factory", "expected"),
    [
        (None, None, None, "production"),
        (object(), None, None, "injected_test_only"),
        (None, object(), None, "injected_test_only"),
        (None, None, object(), "injected_test_only"),
    ],
)
def test_execution_mode_is_production_only_without_injected_components(
    extractor_factory, adapter_factory, answer_model_factory, expected: str,
) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    assert module._execution_mode(
        extractor_factory=extractor_factory,
        adapter_factory=adapter_factory,
        answer_model_factory=answer_model_factory,
    ) == expected


def test_injected_extractor_does_not_require_or_publish_letta_configuration_hash(
    monkeypatch, tmp_path: Path,
) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module
    from mub.vnext.contracts.enums import AnswerDisposition, Operation
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

    key = MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city")
    query = MemoryQueryV3(
        query_id="q", query_type="current", text="Where does Alice live?", selector=CurrentSelector(),
        target_object_keys=(key,), answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )

    class Event:
        event_id = "event-1"
        raw_text = "Alice lives in Paris."
        timestamp = None

        def model_copy(self, update):
            return self

    task = SimpleNamespace(
        task_id="task-1", target_objects=(key,), queries=(query,), events=(Event(),),
        gold_evidence=(SimpleNamespace(answer="Paris"),),
    )
    monkeypatch.setattr(module.extraction, "task_core", lambda task: "core-1")
    monkeypatch.setattr(module, "select_tasks", lambda *args, **kwargs: [task])
    monkeypatch.setattr(module, "validate_terminal_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_output_root", lambda *args, **kwargs: tmp_path / "run")
    monkeypatch.setattr(module, "verify_model_provenance", lambda *args, **kwargs: {
        "snapshot": "/snapshot", "tree_sha256": "a" * 64, "snapshot_binding": {},
        "runtime_receipt_sha256": "b" * 64, "runtime_identity": "runtime.v1",
    })
    monkeypatch.setattr(module, "validate_qualification_artifacts", lambda *args, **kwargs: {"closure": {}, "hashes": {}})
    monkeypatch.setattr(module, "validate_worker_runtime_binding", lambda *args, **kwargs: {"project_root": "project"})
    monkeypatch.setattr(module, "validate_loopback_binding", lambda *args, **kwargs: "http://127.0.0.1:8000")
    monkeypatch.setattr(module, "_task_letta_configuration_hash", lambda task_id: (_ for _ in ()).throw(AssertionError("production Letta hash requested")))

    class Extractor:
        def load(self):
            pass

        def extract(self, raw_text, attribute):
            return {"operation": "update", "value": "Paris"}, "{}", 1, 1.0

        def close(self):
            pass

    class Answer:
        def load(self):
            pass

        def answer(self, request):
            self.last_answer_metadata = {}
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output='{"disposition":"answered","answer":"Paris"}',
                parsed_answer="Paris", format_valid=True, disposition=AnswerDisposition.ANSWERED,
            )

        def close(self):
            pass

    class Adapter:
        def reset(self, request):
            pass

        def ingest_event(self, event):
            return SimpleNamespace(
                effective_action=SimpleNamespace(operation=Operation.ADD),
                affected_entry_ids=("entry-1",),
            )

        def export_entries(self):
            return SimpleNamespace(entries=(MemoryEntryRecordV3(entry_id="entry-1", content="Paris", value_candidate="Paris"),))

        def retrieve(self, request):
            return SimpleNamespace(trace=RetrievalTraceV3(
                query_id="q",
                retrieved_entries=(MemoryEntryRecordV3(entry_id="entry-1", content="Paris", value_candidate="Paris"),),
            ))

        def close(self):
            pass

    monkeypatch.setattr(module, "QwenSession", lambda snapshot: Answer())
    monkeypatch.setattr(module, "_adapter_for_task", lambda *args, **kwargs: Adapter())
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(module.canonical_json_bytes({}))
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_bytes(b"tasks")
    args = SimpleNamespace(
        tasks=str(tasks_path), output_root=str(tmp_path / "out"), qualification_root="qual",
        letta_base_url="http://127.0.0.1:8000", letta_python_executable="python",
        letta_project_root="project", model_snapshot="snapshot", model_runtime_receipt="runtime.json",
        model_snapshot_binding=str(binding_path), scope=module.CANARY_SCOPE,
    )

    result = module.run(args, extractor_factory=Extractor)

    assert result["execution_mode"] == "injected_test_only"
    assert result["letta_configuration_hashes_by_task"] == {}
    row = module.json.loads((tmp_path / "run" / "rows.jsonl").read_text().splitlines()[0])
    assert row["letta_configuration_hash"] is None


def test_unsupported_row_has_explicit_nulls_for_all_state_retrieval_and_answer_fields(monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module
    monkeypatch.setattr(module.extraction, "task_core", lambda task: "core-1")
    task = SimpleNamespace(task_id="task-1")
    row = module._row(task, "NOT_SUPPORTED", reason="unsupported", support_status="NOT_SUPPORTED")
    fields = (
        "parsed_final_value", "stable_entry_id", "stale_retrieved_k16", "retrieval_trace_sha256",
        "visible_prompt_sha256", "prompted_answer", "prompted_exact_match", "answer_outcome", "answer_f1",
        "answer_format_valid", "answer_disposition", "answer_error_flags", "answer_output_sha256",
        "answer_metadata", "letta_configuration_hash",
    )
    assert all(field in row and row[field] is None for field in fields)


def test_status_shape_validation_rejects_non_null_unsupported_state(monkeypatch) -> None:
    import scripts.vnext_run_letta_qwen_prompted_answer as module
    task = SimpleNamespace(task_id="task-1")
    monkeypatch.setattr(module.extraction, "task_core", lambda task: "core-1")
    row = module._row(task, "NOT_SUPPORTED", reason="unsupported", support_status="NOT_SUPPORTED")
    row["state_accuracy"] = False
    with pytest.raises(ValueError, match="NOT_SUPPORTED rows require null"):
        module.validate_row_shape(row, execution_mode="injected_test_only")


    from scripts.vnext_run_letta_qwen_prompted_answer import build_summary, canonical_json_bytes, sha256_bytes

    rows = [
        {"task_id": "a", "status": "PASS", "state_accuracy": True, "gold_retrieved_k16": True,
         "answer_outcome": "CORRECT", "answer_f1": 1.0, "final_memory_size": 1, "letta_configuration_hash": "a" * 64},
        {"task_id": "b", "status": "FAIL", "state_accuracy": False, "gold_retrieved_k16": False,
         "answer_outcome": "WRONG", "answer_f1": 0.0, "final_memory_size": 1, "letta_configuration_hash": "b" * 64},
    ]
    mapping = {"a": "a" * 64, "b": "b" * 64}
    summary = build_summary(rows, scope="canary32", requested=2, rows_sha256="c" * 64,
                            qualification_hashes={}, qualification_identity={}, letta_binding={},
                            endpoint="http://127.0.0.1:8000", model_provenance={},
                            execution_mode="production")
    assert summary["letta_configuration_hashes_by_task"] == mapping
    assert summary["letta_configuration_hashes_sha256"] == sha256_bytes(canonical_json_bytes(mapping))
    assert "letta_configuration_hash" not in summary


def _setup_injected_run(monkeypatch, tmp_path: Path):
    import scripts.vnext_run_letta_qwen_prompted_answer as module

    output = tmp_path / "run"
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(module.canonical_json_bytes({}))
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_bytes(b"tasks")
    monkeypatch.setattr(module, "validate_output_root", lambda *args, **kwargs: output)
    monkeypatch.setattr(module, "verify_model_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "validate_qualification_artifacts", lambda *args, **kwargs: {"closure": {}, "hashes": {}})
    monkeypatch.setattr(module, "validate_worker_runtime_binding", lambda *args, **kwargs: {"project_root": "project"})
    monkeypatch.setattr(module, "validate_loopback_binding", lambda *args, **kwargs: "http://127.0.0.1:8000")
    monkeypatch.setattr(module, "select_tasks", lambda *args, **kwargs: [])
    args = SimpleNamespace(
        tasks=str(tasks_path), output_root=str(tmp_path / "out"), qualification_root="qual",
        letta_base_url="http://127.0.0.1:8000", letta_python_executable="python",
        letta_project_root="project", model_snapshot="snapshot", model_runtime_receipt="runtime.json",
        model_snapshot_binding=str(binding_path), scope=module.CANARY_SCOPE,
    )
    return module, output, args


def test_output_root_creation_failure_closes_loaded_model_and_extractor_once(monkeypatch, tmp_path: Path) -> None:
    module, output, args = _setup_injected_run(monkeypatch, tmp_path)

    class Resource:
        def __init__(self): self.loads = 0; self.closes = 0
        def load(self): self.loads += 1
        def close(self): self.closes += 1

    model, extractor = Resource(), Resource()
    original_mkdir = module.Path.mkdir
    def fail_output_mkdir(path, *args, **kwargs):
        if path == output:
            raise OSError("output root blocked")
        return original_mkdir(path, *args, **kwargs)
    monkeypatch.setattr(module.Path, "mkdir", fail_output_mkdir)
    with pytest.raises(OSError, match="output root blocked"):
        module.run(args, extractor_factory=lambda: extractor, answer_model_factory=lambda: model)
    assert (model.loads, extractor.loads, model.closes, extractor.closes) == (1, 1, 1, 1)


def test_rows_claim_failure_closes_loaded_model_and_extractor_once(monkeypatch, tmp_path: Path) -> None:
    module, output, args = _setup_injected_run(monkeypatch, tmp_path)

    class Resource:
        def __init__(self): self.loads = 0; self.closes = 0
        def load(self): self.loads += 1
        def close(self): self.closes += 1

    model, extractor = Resource(), Resource()
    monkeypatch.setattr(module, "claim_rows_file", lambda path: (_ for _ in ()).throw(FileExistsError(path)))
    with pytest.raises(FileExistsError):
        module.run(args, extractor_factory=lambda: extractor, answer_model_factory=lambda: model)
    assert (model.loads, extractor.loads, model.closes, extractor.closes) == (1, 1, 1, 1)


@pytest.mark.parametrize("timestamp", (None, ""))
def test_absent_delete_timestamp_uses_fallback_compatible_metadata(timestamp) -> None:
    from mub.vnext.contracts.enums import EventRole
    from mub.vnext.contracts.v3.task import MemoryEventV3
    from scripts.vnext_run_letta_qwen_prompted_answer import _visible_action
    from mub.vnext.external.providers.letta_adapter import LettaExternalAdapterV3
    from mub.vnext.contracts.v3.common import FrozenMemoryObjectKey

    key = FrozenMemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city")
    event = MemoryEventV3(
        event_id="event-delete", sequence_index=0, timestamp=timestamp,
        raw_text="delete", normalized_text="delete", role=EventRole.DELETION,
    )
    rendered = _visible_action("delete", None, key.canonical_id, event)
    adapter = LettaExternalAdapterV3.__new__(LettaExternalAdapterV3)
    adapter._target_objects = (key,)
    adapter._target_by_id = {key.canonical_id: key}
    action = adapter._requested_action(event.model_copy(update={"raw_text": rendered, "normalized_text": rendered}))
    assert action.operation.value == "DELETE"


def test_support_rejects_non_current_declared_query_limits() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import task_support_reason
    from mub.vnext.contracts.v3.task import PreviousSelector
    from mub.vnext.contracts.v3.enums import QueryTypeV3

    key = MemoryObjectKey(object_type="profile", namespace="default", entity="alice", attribute="city")
    query = MemoryQueryV3(
        query_id="q-prev", query_type=QueryTypeV3.PREVIOUS, text="Where?", selector=PreviousSelector(),
        target_object_keys=(key,), answer_schema=AnswerSchema.STRING, evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    task = SimpleNamespace(target_objects=(key,), queries=(query,))
    assert task_support_reason(task) == "current_single_object_retrieved_prompt_required"


def test_canary_validation_requires_24_supported_and_8_unsupported() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import CANARY_SCOPE, validate_terminal_rows

    rows = [{"task_id": f"task-{i}", "status": "PASS"} for i in range(32)]
    with pytest.raises(ValueError, match="24 supported and 8"):
        validate_terminal_rows(rows, scope=CANARY_SCOPE)


def test_answer_f1_short_circuits_exact_typed_json_answers() -> None:
    from mub.vnext.contracts.enums import AnswerDisposition
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from scripts.vnext_run_letta_qwen_prompted_answer import _answer_f1

    answer = {"city": "Paris", "metadata": {"year": 2026}}
    prediction = AnswerPredictionV3(
        query_id="q", raw_output="{}", parsed_answer=answer,
        format_valid=True, disposition=AnswerDisposition.ANSWERED,
    )
    assert _answer_f1(prediction, {"city": "Paris", "metadata": {"year": 2026}}) == 1.0


def test_production_row_accepts_required_letta_configuration_hash() -> None:
    from scripts.vnext_run_letta_qwen_prompted_answer import _row

    task = SimpleNamespace(
        task_id="task-production",
        metadata=SimpleNamespace(extra={"semantic_core_id": "core-production"}),
    )
    row = _row(
        task,
        "FAIL",
        execution_mode="production",
        letta_configuration_hash="a" * 64,
        error_class="RuntimeError",
        stage="joint_pipeline",
    )
    assert row["letta_configuration_hash"] == "a" * 64
    assert "execution_mode" not in row
