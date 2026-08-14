from __future__ import annotations

import hashlib

import pytest

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema, EvaluationMode
from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3, RetrievalTraceV3
from mub.vnext.contracts.v3.task import CurrentSelector, MemoryQueryV3


def _target_key() -> MemoryObjectKey:
    return MemoryObjectKey(
        object_type="profile",
        namespace="default",
        entity="alice",
        attribute="city",
    )


def test_answer_json_parser_accepts_typed_answer() -> None:
    from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3

    prediction = parse_answer_prediction_v3(
        query_id="task11-query",
        answer_schema=AnswerSchema.STRING,
        raw_output='{"disposition":"answered","answer":"Paris"}',
    )

    assert prediction.query_id == "task11-query"
    assert prediction.disposition is AnswerDisposition.ANSWERED
    assert prediction.parsed_answer == "Paris"
    assert prediction.format_valid is True
    assert prediction.error_flags == ()


def test_deterministic_decode_contract_rejects_sampling_and_beams() -> None:
    from pydantic import ValidationError

    from mub.vnext.runtime.answer_model_v3 import DeterministicDecodeConfigV3

    with pytest.raises(ValidationError):
        DeterministicDecodeConfigV3(do_sample=True)
    with pytest.raises(ValidationError):
        DeterministicDecodeConfigV3(num_beams=2)

    decoding = DeterministicDecodeConfigV3(max_new_tokens=32, seed=7)
    assert decoding.do_sample is False
    assert decoding.num_beams == 1
    assert decoding.max_new_tokens == 32
    assert decoding.seed == 7


def test_answer_model_slot_requires_a_fixed_revision() -> None:
    from pydantic import ValidationError

    from mub.vnext.runtime.answer_model_v3 import AnswerModelSlotV3

    slot = AnswerModelSlotV3(
        slot_id="answer_model_a",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        snapshot_path="/approved/snapshot",
        license_id="apache-2.0",
        tree_manifest_sha256="a" * 64,
    )
    assert slot.slot_id == "answer_model_a"

    with pytest.raises(ValidationError):
        AnswerModelSlotV3(
            slot_id="answer_model_b",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            revision="main",
            snapshot_path="/approved/snapshot",
            license_id="apache-2.0",
            tree_manifest_sha256="b" * 64,
        )


def test_snapshot_tree_verification_binds_model_revision_and_bytes(tmp_path) -> None:
    from mub.vnext.runtime.answer_model_v3 import (
        AnswerModelSlotV3,
        snapshot_tree_sha256_v3,
        verify_snapshot_tree_v3,
    )

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    expected_hash = snapshot_tree_sha256_v3(
        snapshot_path=snapshot,
        model_id="example/Test-Instruct",
        revision="a" * 40,
    )
    (snapshot / "snapshot-tree.json").write_text("private audit", encoding="utf-8")
    assert snapshot_tree_sha256_v3(
        snapshot_path=snapshot,
        model_id="example/Test-Instruct",
        revision="a" * 40,
    ) == expected_hash
    slot = AnswerModelSlotV3(
        slot_id="answer_model_a",
        model_id="example/Test-Instruct",
        revision="a" * 40,
        snapshot_path=str(snapshot),
        license_id="apache-2.0",
        tree_manifest_sha256=expected_hash,
    )

    verify_snapshot_tree_v3(slot)

    with pytest.raises(ValueError, match="tree manifest"):
        verify_snapshot_tree_v3(
            slot.model_copy(update={"tree_manifest_sha256": "b" * 64})
        )


def test_offline_model_loader_checks_environment_before_importing_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mub.vnext.runtime.answer_model_v3 import (
        AnswerModelSlotV3,
        OfflinePromptedAnswerModelV3,
    )

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    model = OfflinePromptedAnswerModelV3(
        slot=AnswerModelSlotV3(
            slot_id="answer_model_a",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            snapshot_path="/approved/snapshot",
            license_id="apache-2.0",
            tree_manifest_sha256="a" * 64,
        ),
    )

    with pytest.raises(RuntimeError, match="offline"):
        model.load()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with pytest.raises(ValueError, match="snapshot"):
        model.load()


def test_offline_environment_gate_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from mub.vnext.runtime.answer_model_v3 import require_offline_environment_v3

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="offline"):
        require_offline_environment_v3()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    require_offline_environment_v3()


def test_visible_prompt_renderer_preserves_retrieval_order() -> None:
    from mub.vnext.runtime.answer_model_v3 import render_visible_prompt_v3

    query = MemoryQueryV3(
        query_id="task11-render-query",
        query_type="current",
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_target_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    trace = RetrievalTraceV3(
        query_id=query.query_id,
        retrieved_entries=(
            MemoryEntryRecordV3(entry_id="entry-first", content="Alice lives in Prague."),
            MemoryEntryRecordV3(entry_id="entry-second", content="Alice lives in Paris."),
        ),
    )

    prompt = render_visible_prompt_v3(query=query, retrieval_trace=trace)

    assert "Where does Alice live?" in prompt
    assert prompt.index("entry-first") < prompt.index("entry-second")
    assert prompt.index("Prague") < prompt.index("Paris")
    assert render_visible_prompt_v3(query=query, retrieval_trace=trace) == prompt


def test_prompted_answer_request_binds_query_trace_and_rendered_prompt() -> None:
    from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3

    query = MemoryQueryV3(
        query_id="task11-prompted-query",
        query_type="current",
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_target_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    rendered_prompt = "Question: Where does Alice live?"
    prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    request = PromptedAnswerRequestV3(
        query=query,
        retrieval_trace=RetrievalTraceV3(
            query_id=query.query_id,
            prompt_hash=prompt_hash,
        ),
        rendered_prompt=rendered_prompt,
        prompt_hash=prompt_hash,
    )

    assert request.query.query_id == request.retrieval_trace.query_id
    assert request.prompt_hash == hashlib.sha256(
        request.rendered_prompt.encode("utf-8")
    ).hexdigest()


def test_prompted_answer_request_rejects_trace_without_bound_prompt_hash() -> None:
    from pydantic import ValidationError

    from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3

    query = MemoryQueryV3(
        query_id="task11-prompted-hash-query",
        query_type="current",
        text="Where does Alice live?",
        selector=CurrentSelector(),
        target_object_keys=(_target_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    rendered_prompt = "Question: Where does Alice live?"
    with pytest.raises(ValidationError):
        PromptedAnswerRequestV3(
            query=query,
            retrieval_trace=RetrievalTraceV3(
                query_id=query.query_id,
                prompt_hash="a" * 64,
            ),
            rendered_prompt=rendered_prompt,
            prompt_hash=hashlib.sha256(
                rendered_prompt.encode("utf-8")
            ).hexdigest(),
        )


@pytest.mark.parametrize(
    ("query_id", "trace_query_id", "prompt_hash"),
    (
        ("q", "other-q", "a" * 64),
        ("q", "q", "a" * 64),
    ),
)
def test_prompted_answer_request_rejects_unbound_inputs(
    query_id: str,
    trace_query_id: str,
    prompt_hash: str,
) -> None:
    from pydantic import ValidationError

    from mub.vnext.contracts.v3.adapter import PromptedAnswerRequestV3

    query = MemoryQueryV3(
        query_id=query_id,
        query_type="current",
        text="Question",
        selector=CurrentSelector(),
        target_object_keys=(_target_key(),),
        answer_schema=AnswerSchema.STRING,
        evaluation_mode=EvaluationMode.RETRIEVED_PROMPT,
    )
    with pytest.raises(ValidationError):
        PromptedAnswerRequestV3(
            query=query,
            retrieval_trace=RetrievalTraceV3(query_id=trace_query_id),
            rendered_prompt="Question",
            prompt_hash=prompt_hash,
        )

@pytest.mark.parametrize(
    ("answer_schema", "raw_output", "error_flag"),
    (
        (
            AnswerSchema.STRING,
            "not json",
            "answer_json_invalid",
        ),
        (
            AnswerSchema.NUMBER,
            '{"disposition":"answered","answer":true}',
            "answer_schema_mismatch",
        ),
        (
            AnswerSchema.BOOLEAN,
            '{"disposition":"answered"}',
            "answer_envelope_invalid",
        ),
        (
            AnswerSchema.STRING,
            '{"disposition":"unavailable"}',
            "answer_disposition_invalid",
        ),
    ),
)
def test_answer_json_parser_preserves_format_invalid_prediction(
    answer_schema: AnswerSchema,
    raw_output: str,
    error_flag: str,
) -> None:
    from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3

    prediction = parse_answer_prediction_v3(
        query_id="task11-invalid-query",
        answer_schema=answer_schema,
        raw_output=raw_output,
    )

    assert prediction.disposition is AnswerDisposition.ANSWERED
    assert prediction.parsed_answer is None
    assert prediction.format_valid is False
    assert prediction.error_flags == (error_flag,)


def test_answer_json_parser_accepts_explicit_abstention() -> None:
    from mub.vnext.runtime.answer_model_v3 import parse_answer_prediction_v3

    prediction = parse_answer_prediction_v3(
        query_id="task11-abstain-query",
        answer_schema=AnswerSchema.OBJECT,
        raw_output='{"disposition":"abstained"}',
    )

    assert prediction.disposition is AnswerDisposition.ABSTAINED
    assert prediction.parsed_answer is None
    assert prediction.format_valid is True
    assert prediction.error_flags == ()
