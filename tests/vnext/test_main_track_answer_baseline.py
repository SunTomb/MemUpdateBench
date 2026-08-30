from __future__ import annotations

import json
from pathlib import Path

import pytest

from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
from mub.vnext.contracts.enums import AnswerDisposition, CompletionStatus, Operation
from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_tasks_v3
from scripts.vnext_run_main_track_answer_baseline import (
    CANARY_SCOPE,
    TEST_SCOPE,
    build_row,
    build_summary,
    execute_reference_task,
    load_candidate,
    publish_answer_baseline,
    run,
    score_prediction,
    select_tasks,
    validate_audit_attestation,
)


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
AUDIT_ATTESTATION = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


class GoldAnswerModel:
    def answer(self, request):
        return AnswerPredictionV3(
            query_id=request.query.query_id,
            raw_output='{"disposition":"answered","answer":"hidden"}',
            parsed_answer="hidden",
            format_valid=True,
        )

    def close(self):
        pass


def _prediction(query_id: str, *, disposition: AnswerDisposition, answer=None):
    raw = '{"disposition":"abstained"}' if disposition is AnswerDisposition.ABSTAINED else '{"disposition":"answered","answer":"x"}'
    return AnswerPredictionV3(
        query_id=query_id,
        raw_output=raw,
        disposition=disposition,
        parsed_answer=answer,
        format_valid=True,
    )


def test_select_tasks_has_exact_720_test_rows_and_first_32_canary() -> None:
    candidate = load_candidate(CANDIDATE_ROOT)
    test_tasks = select_tasks(candidate.tasks, TEST_SCOPE)
    canary = select_tasks(candidate.tasks, CANARY_SCOPE)

    assert len(test_tasks) == 720
    assert len(canary) == 32
    assert tuple(task.task_id for task in canary) == tuple(
        task.task_id for task in test_tasks[:32]
    )
    assert all(task.metadata.split.value == "test" for task in canary)


def test_c_abstention_and_answer_are_scored_as_typed_outcomes() -> None:
    candidate = load_candidate(CANDIDATE_ROOT)
    task = next(
        task
        for task in candidate.test_tasks
        if task.task_family == "entity_attribute_grounding"
        and task.gold_evidence[0].disposition is AnswerDisposition.ABSTAINED
    )
    gold = task.gold_evidence[0]
    abstained = score_prediction(
        task.queries[0],
        _prediction(task.queries[0].query_id, disposition=AnswerDisposition.ABSTAINED),
        gold,
    )
    answered = score_prediction(
        task.queries[0],
        _prediction(task.queries[0].query_id, disposition=AnswerDisposition.ANSWERED, answer="wrong"),
        gold,
    )

    assert abstained["answer_outcome"] == "CORRECT_ABSTENTION"
    assert abstained["exact_match"] is True
    assert abstained["typed_match"] is True
    assert answered["answer_outcome"] == "WRONG_ABSTENTION"
    assert answered["exact_match"] is False
    assert answered["typed_match"] is False


def test_reference_answer_task_preserves_d_noop_and_retrieval_trace() -> None:
    candidate = load_candidate(CANDIDATE_ROOT)
    task = next(task for task in candidate.test_tasks if task.task_family == "noop_write_discipline")

    class Model:
        def answer(self, request):
            gold = task.gold_evidence[0].answer
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output='{"disposition":"answered","answer":null}',
                parsed_answer=gold,
                format_valid=True,
            )

    record = execute_reference_task(task, Model(), run_id="answer-baseline-test")

    assert record.completion_status is CompletionStatus.COMPLETED
    assert record.retrieval_traces[0].prompt_hash
    assert any(action.operation is Operation.NOOP for action in record.parsed_actions)


def test_rows_never_store_raw_prompt_or_model_output() -> None:
    candidate = load_candidate(CANDIDATE_ROOT)
    task = candidate.test_tasks[0]

    class Model:
        last_answer_metadata = {
            "rendered_chat_prompt_sha256": "a" * 64,
            "rendered_prompt_sha256": "b" * 64,
            "raw_output_sha256": "c" * 64,
            "generated_tokens": 4,
        }

        def answer(self, request):
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output='{"disposition":"answered","answer":null}',
                parsed_answer=task.gold_evidence[0].answer,
                format_valid=True,
            )

    record = execute_reference_task(task, Model(), run_id="redaction-test")
    row = build_row(task, record, model=Model(), candidate=candidate)
    encoded = json.dumps(row, ensure_ascii=False)
    assert '"raw_output"' not in encoded
    assert '"rendered_prompt"' not in encoded
    assert "Use only the retrieved" not in encoded
    assert "Where" not in encoded
    assert row["retrieval_trace_sha256"]
    assert row["visible_prompt_sha256"]
    assert row["qwen_metadata_hashes"]["raw_output_sha256"] == "c" * 64


def test_publication_is_hash_bound_and_no_replace(tmp_path: Path) -> None:
    candidate = load_candidate(CANDIDATE_ROOT)
    task = candidate.test_tasks[0]

    class Model:
        last_answer_metadata = {}

        def answer(self, request):
            return AnswerPredictionV3(
                query_id=request.query.query_id,
                raw_output="{}",
                parsed_answer=task.gold_evidence[0].answer,
                format_valid=True,
            )

    record = execute_reference_task(task, Model(), run_id="publish-test")
    row = build_row(task, record, model=Model(), candidate=candidate)
    summary = build_summary([row], scope=CANARY_SCOPE, candidate=candidate)
    output = tmp_path / "published"
    publish_answer_baseline(candidate, [row], summary, output)
    assert {path.name for path in output.iterdir()} == {"rows.jsonl", "summary.json", "artifact_index.json"}
    with pytest.raises(FileExistsError):
        publish_answer_baseline(candidate, [row], summary, output)

    tampered = tmp_path / "tampered"
    original = (candidate.root / "tasks.jsonl").read_bytes()
    try:
        (candidate.root / "tasks.jsonl").write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="candidate"):
            publish_answer_baseline(candidate, [row], summary, tampered)
        assert not tampered.exists() or not any(tampered.iterdir())
    finally:
        (candidate.root / "tasks.jsonl").write_bytes(original)


def test_production_run_rejects_missing_audit_attestation_before_model(tmp_path: Path) -> None:
    called = []

    def factory():
        called.append(True)
        return GoldAnswerModel()

    with pytest.raises(FileNotFoundError):
        run(
            CANDIDATE_ROOT,
            tmp_path / "missing-output",
            scope=CANARY_SCOPE,
            audit_attestation_path=tmp_path / "missing-attestation.json",
            model_factory=factory,
        )
    assert called == []


def test_audit_attestation_requires_pass_eligibility_and_candidate_binding(tmp_path: Path) -> None:
    original = json.loads(AUDIT_ATTESTATION.read_bytes())
    for mutation in ({"review_status": "FAIL"}, {"benchmark_release_eligible": False}):
        mutated = dict(original)
        mutated.update(mutation)
        path = tmp_path / ("attestation-" + next(iter(mutation)) + ".json")
        path.write_bytes(json.dumps(mutated, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(ValueError, match="attestation"):
            validate_audit_attestation(path, load_candidate(CANDIDATE_ROOT))

    mutated = dict(original)
    mutated["candidate_artifact_hashes"] = dict(original["candidate_artifact_hashes"])
    mutated["candidate_artifact_hashes"]["tasks.jsonl"] = "0" * 64
    path = tmp_path / "attestation-candidate.json"
    path.write_bytes(json.dumps(mutated, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="candidate"):
        validate_audit_attestation(path, load_candidate(CANDIDATE_ROOT))
