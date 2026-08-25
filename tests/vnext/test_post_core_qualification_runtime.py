from __future__ import annotations

import pytest

from mub.vnext.post_core.qualification_receipts_v1 import GateStatus
from tests.vnext.qualification_fixtures import open_runtime_receipts


def _replace(rows, index: int, **changes):
    replacement = rows[index].model_copy(update=changes)
    return (*rows[:index], replacement, *rows[index + 1 :])


def test_open_runtime_receipts_accept_load_only_qwen_and_blocked_bf16() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = validate_runtime_receipts_v1(open_runtime_receipts())

    assert tuple(row.registry_key for row in rows) == (
        "qwen35_9b_bf16",
        "meta_muse_glimmer_30b_int4",
        "meta_muse_glimmer_30b_bf16",
    )
    assert rows[0].generation_status is GateStatus.NOT_RUN
    assert rows[0].load_status is GateStatus.PASS
    assert rows[2].load_status is GateStatus.BLOCKED
    assert tuple(row.snapshot_tree_sha256 for row in rows) == (
        "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db",
        "55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f",
        "7a90420d22f8c98737f15bc31473bbe8a3579ee95f9bf2237172679709877782",
    )
    assert tuple(row.source_binding_ids for row in rows) == (
        ("open_snapshot_closure_receipt", "qwen_load_receipt", "runtime_receipts"),
        ("open_snapshot_closure_receipt", "runtime_receipts"),
        ("open_snapshot_closure_receipt", "runtime_receipts"),
    )
    assert rows[2].blocked_reasons == ("resource/runtime unavailable",)
    assert tuple(
        (
            row.runtime.trust_remote_code,
            row.runtime.compute_dtype,
            row.runtime.attention_implementation,
            row.runtime.seed,
            row.runtime.sampling_mode,
            row.runtime.timeout_seconds,
            row.runtime.engine_args_sha256,
        )
        for row in rows
    ) == (
        (False, "bf16", "sdpa", 0, "greedy", 60, "a" * 64),
        (False, "int4", "llama-cuda", 0, "greedy", 60, "b" * 64),
        (False, "bf16", "eager", 0, "greedy", 60, "c" * 64),
    )


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"trust_remote_code": True}),
        (0, {"compute_dtype": "fp16"}),
        (0, {"attention_implementation": "flash_attention_2"}),
        (0, {"seed": 1}),
        (0, {"sampling_mode": "sample"}),
        (0, {"timeout_seconds": 30}),
        (1, {"compute_dtype": "bf16"}),
        (1, {"attention_implementation": "eager"}),
        (2, {"attention_implementation": "sdpa"}),
    ],
)
def test_runtime_role_policy_rejects_noncanonical_reproducibility_settings(index: int, changes: dict[str, object]) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    with pytest.raises(ValueError):
        runtime = rows[index].runtime.model_copy(update=changes)
        validate_runtime_receipts_v1(_replace(rows, index, runtime=runtime))


def test_muse_gguf_requires_llama_commit_binary_build_and_speculative_off() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    missing_commit = rows[1].runtime.model_copy(update={"engine_commit": None})
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(rows, 1, runtime=missing_commit))

    malformed_commit = rows[1].runtime.model_copy(update={"engine_commit": "not-a-sha"})
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(rows, 1, runtime=malformed_commit))

    payload = rows[1].model_dump(mode="python")
    payload["runtime"] = rows[1].runtime
    payload["speculative_decoding"] = "on"
    speculative_on = type(rows[1]).model_construct(**payload)
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1((rows[0], speculative_on, rows[2]))


def test_bf16_blocked_receipt_keeps_measurements_and_generation_evidence_null() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    row = open_runtime_receipts()[2]
    assert row.generated_token_count is None
    assert row.peak_memory_bytes is None
    assert row.prompt_fixture_sha256 is None
    assert row.parser_sha256 is None
    assert row.chat_template_sha256 is None
    assert row.output_projection_sha256 is None
    validate_runtime_receipts_v1(open_runtime_receipts())


@pytest.mark.parametrize(
    "changes",
    [
        {
            "generation_status": GateStatus.PASS,
            "determinism_status": GateStatus.NOT_RUN,
            "load_status": GateStatus.PASS,
            "unload_status": GateStatus.PASS,
            "prompt_fixture_sha256": "a" * 64,
            "parser_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "output_projection_sha256": "d" * 64,
            "generated_token_count": 8,
            "peak_memory_bytes": 1,
        },
        {
            "generation_status": GateStatus.PASS,
            "determinism_status": GateStatus.BLOCKED,
            "load_status": GateStatus.PASS,
            "unload_status": GateStatus.PASS,
            "prompt_fixture_sha256": "a" * 64,
            "parser_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "output_projection_sha256": "d" * 64,
            "generated_token_count": 8,
            "peak_memory_bytes": 1,
            "blocked_reasons": ("determinism was not run",),
        },
        {
            "generation_status": GateStatus.NOT_RUN,
            "generated_token_count": 8,
        },
        {
            "generation_status": GateStatus.BLOCKED,
            "prompt_fixture_sha256": "a" * 64,
        },
        {"determinism_status": GateStatus.PASS},
    ],
)
def test_runtime_gate_dependencies_and_evidence_are_required(changes) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(open_runtime_receipts(), 0, **changes))


def test_blocked_or_not_run_gates_reject_fabricated_measurements() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(open_runtime_receipts(), 2, peak_memory_bytes=1))
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(open_runtime_receipts(), 0, generated_token_count=1))


def test_runtime_identity_order_revision_and_source_bindings_are_exact() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(tuple(reversed(rows)))
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(rows, 0, revision="wrong"))
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(rows, 0, source_binding_ids=()))
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(
            _replace(rows, 0, source_binding_ids=("open_snapshot_closure_receipt",))
        )
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(
            _replace(rows, 0, source_binding_ids=("qwen_snapshot_closure", "qwen_snapshot_closure"))
        )
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(rows, 0, registry_key="unexpected"))


def test_bf16_resource_block_cannot_be_reported_as_unsupported() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(
            _replace(open_runtime_receipts(), 2, load_status=GateStatus.UNSUPPORTED)
        )


def test_qwen_requires_transformers_engine() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    wrong_engine = open_runtime_receipts()[0].runtime.model_copy(update={"engine": "llama.cpp"})
    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(_replace(open_runtime_receipts(), 0, runtime=wrong_engine))


def test_fail_gate_requires_nonempty_blocked_reasons() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    with pytest.raises(ValueError):
        validate_runtime_receipts_v1(
            _replace(open_runtime_receipts(), 0, load_status=GateStatus.FAIL)
        )


def test_unsupported_runtime_gates_require_future_typed_proof() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    for index in range(3):
        with pytest.raises(ValueError, match="UNSUPPORTED"):
            validate_runtime_receipts_v1(
                _replace(rows, index, load_status=GateStatus.UNSUPPORTED, blocked_reasons=("GPU unavailable",))
            )

def test_runtime_generation_pass_requires_positive_peak_memory_evidence() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    qwen_index = next(index for index, row in enumerate(rows) if row.registry_key == "qwen35_9b_bf16")

    with pytest.raises(ValueError, match="peak memory"):
        validate_runtime_receipts_v1(
            _replace(
                rows,
                qwen_index,
                generation_status=GateStatus.PASS,
                determinism_status=GateStatus.PASS,
                prompt_fixture_sha256="a" * 64,
                parser_sha256="b" * 64,
                chat_template_sha256="c" * 64,
                output_projection_sha256="d" * 64,
                generated_token_count=1,
                peak_memory_bytes=None,
            )
        )


def test_runtime_validation_exports_exactly_four_public_functions() -> None:
    import mub.vnext.post_core.qualification_validation_v1 as validation

    assert validation.__all__ == [
        "load_capability_anomaly_receipt_v1",
        "load_canonical_jsonl_v1",
        "load_execution_authorization_v1",
        "validate_capability_attempt_receipts_v1",
        "validate_canonical_capability_smoke_plan_v1",
        "validate_escalation_anomaly_evidence_v1",
        "validate_escalation_anomaly_receipt_v1",
        "validate_provider_attestations_v1",
        "validate_qualification_secret_free",
        "validate_runtime_receipts_v1",
    ]


def test_generation_determinism_pass_requires_matching_repeat_projection_hash() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import validate_runtime_receipts_v1

    rows = open_runtime_receipts()
    complete = {
        "generation_status": GateStatus.PASS,
        "determinism_status": GateStatus.PASS,
        "load_status": GateStatus.PASS,
        "unload_status": GateStatus.PASS,
        "prompt_fixture_sha256": "a" * 64,
        "parser_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "output_projection_sha256": "d" * 64,
        "repeat_output_projection_sha256": "d" * 64,
        "generated_token_count": 1,
        "peak_memory_bytes": 1,
    }
    assert validate_runtime_receipts_v1(_replace(rows, 0, **complete))[0].determinism_status is GateStatus.PASS
    for repeat_hash in (None, "e" * 64):
        with pytest.raises(ValueError, match="repeat projection"):
            validate_runtime_receipts_v1(
                _replace(rows, 0, **{**complete, "repeat_output_projection_sha256": repeat_hash})
            )
