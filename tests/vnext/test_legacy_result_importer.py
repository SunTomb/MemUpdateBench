from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from mub.vnext import legacy
from mub.vnext.contracts.enums import EvaluationMode, SupportReason
from mub.vnext.io.canonical import canonical_json_bytes
from mub.vnext.legacy import import_evomemory_results, parse_legacy_run_name
from mub.vnext.legacy.caveats import legacy_namespace
from mub.vnext.legacy.loaders import load_evomemory_results


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "legacy"


def _phase(payload: dict[str, Any]) -> str:
    metadata = payload["summary"].get("legacy_analysis_metadata")
    if type(metadata) is dict and type(metadata.get("legacy_phase")) is str:
        return metadata["legacy_phase"]
    return "P6.3"


def _write_source(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tasks(
    make_task,
    count: int = 2,
    *,
    phase: str = "P6.3",
    dataset_id: str = "evomemory_update_frequency_hard_p63",
    split_id: str = "test",
    gold_answer: str = "Qingdao",
):
    base_task = make_task()
    task_type = type(base_task)
    base_data = base_task.model_dump(mode="python")
    query_id = base_data["queries"][0]["query_id"]
    base_data["gold"]["gold_answers"][query_id] = gold_answer
    base_data["gold"]["acceptable_answers"][query_id] = [gold_answer]
    base_data["metadata"]["legacy_provenance"] = {
        "legacy_family_id": "evomemory",
        "legacy_phase": phase,
        "legacy_dataset_id": dataset_id,
        "legacy_split_id": split_id,
        "legacy_metric_namespace": legacy_namespace(phase),
        "legacy_run_condition_id": None,
        "checkpoint_family": None,
        "training_seed": None,
        "answer_mode": None,
        "memory_trajectory_id": None,
        "source_artifact_path": "legacy/dataset.json",
        "source_artifact_hash": "c" * 64,
        "known_caveats": [],
    }
    return {
        index: task_type.model_validate(
            {**base_data, "task_id": f"task_legacy_{index}"}
        )
        for index in range(count)
    }


def _tasks_for_indices(make_task, indices: list[int], **kwargs):
    sequential = _tasks(make_task, len(indices), **kwargs)
    tasks = {}
    for offset, legacy_index in enumerate(indices):
        task = sequential[offset]
        tasks[legacy_index] = _replace_task(
            task, task_id=f"task_legacy_{legacy_index}"
        )
    return tasks


def _payload(*, count: int = 2, phase: str = "P6.3") -> dict[str, Any]:
    return {
        "summary": {
            "benchmark": "evomemory",
            "mode": "raw_add",
            "answer_mode": "slot_prompt",
            "num_examples": count,
            "legacy_analysis_metadata": {
                "legacy_phase": phase,
                "fixture_origin": "handwritten_synthetic",
            },
        },
        "results": [
            {
                "example_id": index,
                "shard_local_example_id": index,
                "gold_answer": "Qingdao",
                "predicted": "Qingdao" if index else "Wuxi",
                "em": 1.0 if index else 0.0,
                "f1": 1.0 if index else 0.0,
                "state_value_em": True,
            }
            for index in range(count)
        ],
    }


def _import(
    payload: dict[str, Any],
    make_task,
    *,
    source_payload: dict[str, Any] | None = None,
    **overrides,
):
    phase = _phase(payload)
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "results.json"
        source_sha256 = _write_source(
            source_path, payload if source_payload is None else source_payload
        )
        options = {
            "source_path": source_path,
            "source_sha256": source_sha256,
            "run_name": None,
            "task_by_legacy_index": _tasks(
                make_task,
                len(payload.get("results", [])),
                phase=phase,
                dataset_id=f"evomemory_{phase.lower().replace('.', '')}",
            ),
        }
        options.update(overrides)
        return import_evomemory_results(payload, **options)


def _legacy_event(record) -> dict[str, Any]:
    events = [event for event in record.system_events if event.get("type") == "legacy_evomemory_result"]
    assert len(events) == 1
    return events[0]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("raw_add_slot_prompt_k16", {"mode": "raw_add", "answer_mode": "slot_prompt", "update_depth": 16, "warnings": ("legacy_directory_name_inference",)}),
        ("long25_slot_direct_k8", {"mode": "long25", "answer_mode": "slot_direct", "update_depth": 8, "warnings": ("legacy_directory_name_inference",)}),
        ("oracle_slot_direct_k1", {"mode": "oracle", "answer_mode": "slot_direct", "update_depth": 1, "warnings": ("legacy_directory_name_inference",)}),
    ],
)
def test_parse_documented_legacy_run_names(name: str, expected: dict[str, Any]) -> None:
    assert parse_legacy_run_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "unknown_slot_prompt_k16", "oracle_slot_prompt_k1", "raw_add_slot_prompt_k3",
        "raw_add_prompt_k16", "raw_add_slot_prompt_k16_extra", "RAW_ADD_SLOT_PROMPT_K16",
        " raw_add_slot_prompt_k16", "raw_add_slot_prompt_k16 ", "",
    ],
)
def test_unknown_or_malformed_run_names_infer_nothing(name: str) -> None:
    assert parse_legacy_run_name(name) is None


def test_run_name_parser_rejects_string_subclasses_without_hooks() -> None:
    class HostileString(str):
        calls = 0
        def __str__(self) -> str:
            self.calls += 1
            raise AssertionError("coercion hook must not run")

    value = HostileString("raw_add_slot_prompt_k16")
    with pytest.raises(TypeError, match="exact built-in string"):
        parse_legacy_run_name(value)
    assert value.calls == 0


def test_explicit_summary_identity_wins_and_conflict_is_audited(make_task) -> None:
    payload = _payload()
    payload["summary"]["mode"] = "long25"
    manifest, _, _, warnings = _import(payload, make_task, run_name="raw_add_slot_prompt_k16")
    identity = manifest.prompt_config["legacy_result_import"]["run_identity"]
    assert identity["mode"] == "long25"
    assert identity["answer_mode"] == "slot_prompt"
    assert identity["update_depth"] == 16
    assert "legacy_directory_name_inference" in warnings
    assert "legacy_directory_name_conflict:mode" in warnings


@pytest.mark.parametrize("run_name", ["mystery", ""])
def test_unknown_or_malformed_name_does_not_emit_inference_warning(make_task, run_name: str) -> None:
    _, _, _, warnings = _import(_payload(), make_task, run_name=run_name)
    assert "legacy_directory_name_inference" not in warnings


def test_old_and_traced_dialects_preserve_raw_fields_and_optional_none(make_task) -> None:
    old_path = FIXTURE_DIR / "evomemory_results_old.json"
    traced_path = FIXTURE_DIR / "evomemory_results_traced.json"
    old = load_evomemory_results(old_path)
    traced = load_evomemory_results(traced_path)
    old_import = import_evomemory_results(old, source_path=old_path, source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(), run_name=None, task_by_legacy_index=_tasks(make_task, gold_answer="Suzhou"))
    traced_import = import_evomemory_results(traced, source_path=traced_path, source_sha256=hashlib.sha256(traced_path.read_bytes()).hexdigest(), run_name=None, task_by_legacy_index=_tasks(make_task, phase="P6.5", dataset_id="evomemory_p65", gold_answer="Suzhou"))
    old_manifest, old_runs, _, _ = old_import
    traced_manifest, traced_runs, _, _ = traced_import
    old_event = _legacy_event(old_runs[0])
    traced_event = _legacy_event(traced_runs[0])
    assert old_manifest.prompt_config["legacy_result_import"]["dialect"] == "old"
    assert traced_manifest.prompt_config["legacy_result_import"]["dialect"] == "traced"
    assert old_event["raw_row"] == old["results"][0]
    assert traced_event["raw_row"] == traced["results"][0]
    assert old_event["legacy_optional"]["answer_trace"] is None
    assert old_event["legacy_optional"]["answer_topk"] is None
    assert traced_event["legacy_optional"]["answer_trace"] == traced["results"][0]["answer_trace"]
    assert traced_runs[1].retrieval_traces == []


def test_gold_retrieval_dialect_fields_remain_separate(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["gold_retrieved"] = False
    payload["results"][0]["answer_trace"] = {"gold_value_in_retrieved": True}
    _, runs, _, _ = _import(payload, make_task)
    optional = _legacy_event(runs[0])["legacy_optional"]
    assert optional["gold_retrieved"] is False
    assert optional["gold_value_in_retrieved"] is True
    assert "gold_retrieved_normalized" not in optional


def test_nonidentical_p63_metrics_remain_legacy_only(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0].update({
        "answer_exact_match": 1.0, "answer_token_f1": 0.75,
        "final_state_accuracy": 1.0, "stale_same_slot_count": 2,
        "unknown_metric": 0.25,
    })
    _, _, scores, _ = _import(payload, make_task)
    score = scores[0]
    assert score.answer_scores.exact_match is None
    assert score.answer_scores.token_f1 is None
    assert score.state_scores.final_state_accuracy is None
    assert score.store_scores.stale_conflicting_value_count is None
    expected_reasons = {
        "answer_scores.exact_match": SupportReason.MISSING_ARTIFACT,
        "answer_scores.token_f1": SupportReason.MISSING_ARTIFACT,
        "state_scores.final_state_accuracy": SupportReason.NOT_SUPPORTED,
        "store_scores.stale_conflicting_value_count": SupportReason.NOT_SUPPORTED,
    }
    for path, expected_reason in expected_reasons.items():
        assert score.supported_metric_fields[path].reason is expected_reason
    legacy = score.legacy_metrics["legacy_p63"]
    assert legacy["answer_exact_match"] == 1.0
    assert legacy["answer_token_f1"] == 0.75
    assert legacy["final_state_accuracy"] == 1.0
    assert legacy["stale_same_slot_count"] == 2
    assert legacy["unknown_metric"] == 0.25


def test_p84_metric_names_are_not_canonical_em_aliases(make_task) -> None:
    payload = _payload(count=1, phase="P8.4")
    payload["results"][0]["answer_exact_match"] = 1.0
    _, _, scores, _ = _import(payload, make_task)
    assert scores[0].answer_scores.exact_match is None
    assert scores[0].legacy_metrics["legacy_p84"]["answer_exact_match"] == 1.0


def _append_duplicate_row(payload: dict[str, Any]) -> None:
    payload["results"].append(dict(payload["results"][0]))
    payload["summary"]["num_examples"] += 1


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["results"][0].pop("example_id"), "example_id"),
        (_append_duplicate_row, "duplicate"),
        (lambda payload: payload["results"][0].__setitem__("example_id", True), "integer"),
    ],
)
def test_rejects_missing_duplicate_or_malformed_global_row_indices(make_task, mutator, message: str) -> None:
    payload = _payload()
    mutator(payload)
    with pytest.raises(ValueError, match=message) as exc_info:
        _import(payload, make_task)
    assert "results.json" in str(exc_info.value)


def test_rejects_missing_extra_or_noninteger_task_indices(make_task) -> None:
    payload = _payload()
    task_map = _tasks(make_task)
    bad_maps = [
        ({0: task_map[0]}, "unconsumed|missing"),
        ({**task_map, 2: _tasks(make_task, 3)[2]}, "unconsumed|extra"),
        ({False: task_map[0], 1: task_map[1]}, "integer"),
    ]
    for bad_map, message in bad_maps:
        with pytest.raises(ValueError, match=message):
            _import(payload, make_task, task_by_legacy_index=bad_map)


@pytest.mark.parametrize(
    ("where", "field", "value"),
    [
        ("summary", "status", "failed"), ("summary", "run_status", "pending"),
        ("summary", "completion_status", "partial"), ("summary", "status", "incomplete"),
        ("summary", "status", "capacity-exhausted"), ("summary", "status", "FAILED"),
        ("summary", "status", " pending "), ("summary", "status", "mystery"),
        ("row", "row_status", "pending"), ("row", "capacity_failed", True),
        ("row", "capacity_failed", 1.0), ("row", "capacity_failed", "1.0"),
        ("row", "supported", False), ("row", "supported", 0.0),
    ],
)
def test_rejects_incomplete_failed_pending_or_capacity_statuses(make_task, where: str, field: str, value: Any) -> None:
    payload = _payload()
    target = payload["summary"] if where == "summary" else payload["results"][0]
    target[field] = value
    with pytest.raises(ValueError, match="completed canonical run"):
        _import(payload, make_task)


def test_answer_mode_mapping_requires_explicit_verified_evidence(make_task) -> None:
    payload = _payload(count=1)
    manifest, _, _, warnings = _import(payload, make_task)
    metadata = manifest.prompt_config["legacy_result_import"]
    assert metadata["run_identity"]["answer_mode"] == "slot_prompt"
    assert metadata["canonical_evaluation_mode"] is None
    assert "legacy_answer_mode_unverified" in warnings
    payload["summary"]["semantic_compatibility"] = {
        "verifier": "source-authored-label-is-not-proof",
        "legacy_answer_mode": "slot_prompt",
        "canonical_evaluation_mode": "retrieved_prompt",
    }
    with pytest.raises(ValueError, match="semantic_compatibility"):
        _import(payload, make_task)


def test_state_direct_mapping_requires_exact_linked_state_evidence(make_task) -> None:
    payload = _payload(count=1)
    tasks = _tasks(make_task, 1)
    task = tasks[0]
    query = task.queries[0]
    gold_answer = task.gold.gold_answers[query.query_id]
    payload["summary"]["answer_mode"] = "slot_direct"
    payload["results"][0]["gold_answer"] = gold_answer
    payload["results"][0]["predicted"] = gold_answer
    payload["results"][0]["state_direct_trace"] = {
        "query_id": query.query_id,
        "object_key": query.target_object_keys[0].canonical_id,
        "value": gold_answer,
    }
    payload["summary"]["semantic_compatibility"] = {
        "verifier": "evomemory_slot_direct_state_readout_v1",
        "legacy_answer_mode": "slot_direct",
        "canonical_evaluation_mode": "state_direct",
    }

    with pytest.raises(ValueError, match="semantic_compatibility|evaluation_mode|state_direct_trace"):
        _import(payload, make_task, task_by_legacy_index=tasks)


@pytest.mark.parametrize(
    "evidence",
    [
        {"verified": True, "legacy_answer_mode": "slot_prompt", "canonical_evaluation_mode": "retrieved_prompt", "evidence": "self-attested prose"},
        {"verifier": "evomemory_traced_slot_prompt_v1", "legacy_answer_mode": "slot_prompt", "canonical_evaluation_mode": "state_direct"},
        {"verifier": "unknown", "legacy_answer_mode": "slot_prompt", "canonical_evaluation_mode": "retrieved_prompt"},
        {"verifier": "evomemory_slot_direct_state_readout_v1", "legacy_answer_mode": "slot_direct", "canonical_evaluation_mode": "state_direct"},
    ],
)
def test_incompatible_semantic_evidence_fails_closed(make_task, evidence) -> None:
    payload = _payload(count=1)
    payload["summary"]["semantic_compatibility"] = evidence
    with pytest.raises(ValueError, match="semantic_compatibility"):
        _import(payload, make_task)


def test_identity_fields_require_exact_scalar_types(make_task) -> None:
    cases = [
        ("mode", []),
        ("answer_mode", True),
        ("checkpoint_family", ""),
        ("seed", True),
        ("update_depth", 1.0),
        ("answer_topk", -1),
    ]
    for field, value in cases:
        payload = _payload()
        payload["summary"][field] = value
        with pytest.raises(ValueError, match=field):
            _import(payload, make_task)


def test_bool_and_integer_run_identity_values_do_not_compare_equal(make_task) -> None:
    payload = _payload()
    payload["summary"]["seed"] = 1
    payload["results"][0]["seed"] = True
    payload["results"][1]["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        _import(payload, make_task)


def test_present_malformed_runtime_fields_are_rejected(make_task) -> None:
    for field, value in [("answer_trace", []), ("predicted", 3)]:
        payload = _payload()
        payload["results"][0][field] = value
        with pytest.raises(ValueError, match=field):
            _import(payload, make_task)


@pytest.mark.parametrize("field", ["memory_trajectory_id", "checkpoint_family", "run_condition_id"])
def test_rejects_mixed_row_run_identity(field: str, make_task) -> None:
    payload = _payload()
    payload["summary"][field] = "summary-value"
    payload["results"][1][field] = "different-row-value"
    with pytest.raises(ValueError, match=field):
        _import(payload, make_task)


def test_source_provenance_determinism_and_input_immutability(make_task, tmp_path: Path) -> None:
    payload = _payload()
    task_map = _tasks(make_task)
    payload_before = copy.deepcopy(payload)
    task_ids_before = {index: task.task_id for index, task in task_map.items()}
    source_path = tmp_path / "deterministic-results.json"
    source_sha256 = _write_source(source_path, payload)
    first = import_evomemory_results(payload, source_path=source_path, source_sha256=source_sha256, run_name="raw_add_slot_prompt_k16", task_by_legacy_index=task_map)
    second = import_evomemory_results(dict(reversed(payload.items())), source_path=source_path, source_sha256=source_sha256, run_name="raw_add_slot_prompt_k16", task_by_legacy_index=dict(reversed(task_map.items())))
    first_manifest, first_runs, first_scores, first_warnings = first
    second_manifest, second_runs, second_scores, second_warnings = second
    metadata = first_manifest.prompt_config["legacy_result_import"]
    assert metadata["source_path"] == str(source_path)
    assert metadata["source_sha256"] == source_sha256
    assert first_manifest.raw_provider_response_artifacts[0].path == str(source_path)
    assert first_manifest.raw_provider_response_artifacts[0].sha256 == source_sha256
    assert first_manifest.run_id == second_manifest.run_id
    assert [record.task_id for record in first_runs] == ["task_legacy_0", "task_legacy_1"]
    assert [record.run_id for record in first_runs] == [first_manifest.run_id] * 2
    assert [record.run_id for record in first_scores] == [first_manifest.run_id] * 2
    assert canonical_json_bytes(first_manifest) == canonical_json_bytes(second_manifest)
    assert [canonical_json_bytes(item) for item in first_runs] == [canonical_json_bytes(item) for item in second_runs]
    assert [canonical_json_bytes(item) for item in first_scores] == [canonical_json_bytes(item) for item in second_scores]
    assert first_warnings == second_warnings
    assert payload == payload_before
    assert {index: task.task_id for index, task in task_map.items()} == task_ids_before


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "g" * 64, "", "a" * 65])
def test_rejects_noncanonical_source_sha256(make_task, bad_hash: str) -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        _import(_payload(), make_task, source_sha256=bad_hash)


def test_rejects_hostile_containers_nonfinite_surrogates_cycles_shared_dag_and_secrets(make_task) -> None:
    class HostileDict(dict):
        calls = 0
        def items(self):
            self.calls += 1
            raise AssertionError("hostile items hook must not run")

    hostile = HostileDict(_payload())
    with pytest.raises(TypeError, match="exact built-in JSON object"):
        _import(hostile, make_task, source_payload=_payload())
    assert hostile.calls == 0
    cases: list[tuple[dict[str, Any], str]] = []
    nonfinite = _payload(); nonfinite["summary"]["extra"] = float("nan"); cases.append((nonfinite, "finite"))
    surrogate = _payload(); surrogate["results"][0]["extra"] = "\ud800"; cases.append((surrogate, "Unicode scalar"))
    cycle = _payload(); cycle["summary"]["cycle"] = cycle["summary"]; cases.append((cycle, "cycle"))
    shared = _payload(); child: list[Any] = []; shared["summary"]["left"] = child; shared["summary"]["right"] = child; cases.append((shared, "shared"))
    secret = _payload(); secret["summary"]["api_key"] = "do-not-copy"; cases.append((secret, "sensitive"))
    budget = _payload(); budget["summary"]["many"] = [None] * 1_000_001; cases.append((budget, "budget"))
    for case, message in cases:
        with pytest.raises(ValueError, match=message):
            _import(case, make_task, source_payload=_payload())


def _replace_task(
    task,
    *,
    task_id: str | None = None,
    query_mode: EvaluationMode | None = None,
    provenance: dict[str, Any] | None | object = ...,
):
    data = task.model_dump(mode="python")
    if task_id is not None:
        data["task_id"] = task_id
    if query_mode is not None:
        data["queries"][0]["evaluation_mode"] = query_mode
    if provenance is not ...:
        data["metadata"]["legacy_provenance"] = provenance
    return type(task).model_validate(data)


def test_source_artifact_must_exist_match_hash_and_payload(make_task, tmp_path: Path) -> None:
    payload = _payload(count=1)
    tasks = _tasks(make_task, 1)
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="missing.json"):
        import_evomemory_results(payload, source_path=missing, source_sha256="a" * 64, run_name=None, task_by_legacy_index=tasks)

    source = tmp_path / "results.json"
    digest = _write_source(source, payload)
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    import_evomemory_results(payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    with pytest.raises((IsADirectoryError, PermissionError, OSError)):
        import_evomemory_results(payload, source_path=tmp_path, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)
    with pytest.raises(ValueError, match="source_sha256"):
        import_evomemory_results(payload, source_path=source, source_sha256="b" * 64, run_name=None, task_by_legacy_index=tasks)

    changed_payload = copy.deepcopy(payload)
    changed_payload["results"][0]["em"] = 1.0
    with pytest.raises(ValueError, match="payload.*source|source.*payload"):
        import_evomemory_results(changed_payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)


def test_source_artifact_change_during_import_is_rejected(make_task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload(count=1)
    source = tmp_path / "changing.json"
    digest = _write_source(source, payload)
    tasks = _tasks(make_task, 1)
    calls = 0
    real_hash = hashlib.sha256

    def changing_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        value = real_hash(path.read_bytes()).hexdigest()
        return value if calls == 1 else "f" * 64

    import mub.vnext.legacy.results as result_module
    monkeypatch.setattr(result_module, "_sha256_file", changing_hash)
    with pytest.raises(RuntimeError, match="changed"):
        import_evomemory_results(payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)


@pytest.mark.parametrize("field", ["failed", "pending", "partial", "incomplete", "unsupported", "not_supported", "capacity_failed", "capacity_exhausted"])
def test_completion_boolean_indicators_fail_closed(make_task, field: str) -> None:
    payload = _payload(count=1)
    payload["summary"][field] = "1.0"
    with pytest.raises(ValueError, match="completed canonical run"):
        _import(payload, make_task)


def test_row_completion_indicator_also_fails_closed(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["pending"] = True
    with pytest.raises(ValueError, match="completed canonical run"):
        _import(payload, make_task)


def test_missing_optional_prediction_is_allowed_with_substantive_metrics(make_task) -> None:
    payload = _payload(count=1)
    del payload["results"][0]["predicted"]
    _, runs, scores, _ = _import(payload, make_task)
    assert runs[0].answer_predictions == []
    assert scores[0].legacy_metrics["legacy_p63"]["em"] == 0.0


@pytest.mark.parametrize("field", ["error", "exception", "error_message"])
def test_nonnull_summary_or_row_errors_fail_closed(make_task, field: str) -> None:
    for target_name in ("summary", "row"):
        payload = _payload(count=1)
        target = payload["summary"] if target_name == "summary" else payload["results"][0]
        target[field] = "legacy failure"
        with pytest.raises(ValueError, match=field):
            _import(payload, make_task)


@pytest.mark.parametrize(
    "counts",
    [
        {"completed_task_count": 0},
        {"completed_task_count": 2},
        {"failed_task_count": 1},
        {"not_supported_task_count": 1},
        {"completed_task_count": 1, "failed_task_count": 1},
    ],
)
def test_completion_counts_must_describe_exactly_complete_rows(make_task, counts: dict[str, int]) -> None:
    payload = _payload(count=1)
    payload["summary"].update(counts)
    with pytest.raises(ValueError, match="task_count|completed canonical run"):
        _import(payload, make_task)


_ZERO_REQUIRED_COMPLETION_TASK_COUNT_FIELDS = (
    "pending_task_count",
    "partial_task_count",
    "incomplete_task_count",
    "unsupported_task_count",
    "capacity_failed_task_count",
    "capacity_exhausted_task_count",
)


@pytest.mark.parametrize(
    "field",
    _ZERO_REQUIRED_COMPLETION_TASK_COUNT_FIELDS,
)
def test_noncompleted_task_counts_must_be_zero_for_completed_import(
    make_task,
    field: str,
) -> None:
    payload = _payload(count=1)
    payload["summary"][field] = 1
    with pytest.raises(ValueError, match=f"{field}|task_count|completed"):
        _import(payload, make_task)


@pytest.mark.parametrize(
    "field",
    _ZERO_REQUIRED_COMPLETION_TASK_COUNT_FIELDS,
)
@pytest.mark.parametrize("malformed", [True, -1, 0.0, "0", None])
def test_noncompleted_task_counts_require_exact_nonnegative_integers(
    make_task,
    field: str,
    malformed: Any,
) -> None:
    payload = _payload(count=1)
    payload["summary"][field] = malformed
    with pytest.raises(ValueError, match=f"{field}|task_count|integer"):
        _import(payload, make_task)


def test_zero_noncompleted_task_counts_are_valid_and_raw_preserved(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        field: 0 for field in _ZERO_REQUIRED_COMPLETION_TASK_COUNT_FIELDS
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw_summary = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert all(
        raw_summary[field] == 0
        for field in _ZERO_REQUIRED_COMPLETION_TASK_COUNT_FIELDS
    )


def test_operation_exec_failures_and_generic_counts_are_not_task_count_aliases(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "exec_stats": {"failed": 3},
        "pending_count": 2,
        "capacity_exhausted_count": 1,
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw_summary = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw_summary["exec_stats"]["failed"] == 3
    assert raw_summary["pending_count"] == 2
    assert raw_summary["capacity_exhausted_count"] == 1


def test_valid_explicit_completion_counts_are_preserved(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({"completed_task_count": 1, "failed_task_count": 0, "not_supported_task_count": 0})
    manifest, _, _, _ = _import(payload, make_task)
    assert manifest.completed_task_count == 1


@pytest.mark.parametrize("field", ["predicted", "gold_answer"])
def test_explicit_null_answer_fields_are_rejected(make_task, field: str) -> None:
    payload = _payload(count=1)
    payload["results"][0][field] = None
    with pytest.raises(ValueError, match=field):
        _import(payload, make_task)


def test_index_only_row_is_not_substantive_result_evidence(make_task) -> None:
    payload = _payload(count=1)
    payload["results"] = [{"example_id": 0, "shard_local_example_id": 0}]
    with pytest.raises(ValueError, match="substantive"):
        _import(payload, make_task)


def test_metadata_only_row_is_not_substantive_result_evidence(make_task) -> None:
    payload = _payload(count=1)
    payload["results"] = [{
        "example_id": 0,
        "shard_local_example_id": 0,
        "legacy_note": "metadata only",
    }]
    with pytest.raises(ValueError, match="substantive"):
        _import(payload, make_task)


def test_task_ids_must_be_unique_and_provenance_required(make_task) -> None:
    payload = _payload()
    tasks = _tasks(make_task)
    tasks[1] = _replace_task(tasks[1], task_id=tasks[0].task_id)
    with pytest.raises(ValueError, match="duplicate task_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    tasks = _tasks(make_task)
    tasks[0] = _replace_task(tasks[0], provenance=None)
    with pytest.raises(ValueError, match="LegacyProvenance"):
        _import(payload, make_task, task_by_legacy_index=tasks)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("legacy_family_id", "other_family"),
        ("legacy_phase", "P6.5"),
        ("legacy_dataset_id", "other_dataset"),
        ("legacy_split_id", "dev"),
        ("legacy_metric_namespace", "legacy_p65"),
    ],
)
def test_mixed_or_explicitly_mismatched_task_lineage_is_rejected(make_task, field: str, value: str) -> None:
    payload = _payload()
    payload["summary"]["legacy_analysis_metadata"].update({
        "legacy_family_id": "evomemory",
        "legacy_dataset_id": "evomemory_p63",
        "legacy_split_id": "test",
    })
    tasks = _tasks(make_task, dataset_id="evomemory_p63")
    provenance = tasks[1].metadata.legacy_provenance.model_dump(mode="python")
    provenance[field] = value
    tasks[1] = _replace_task(tasks[1], provenance=provenance)
    with pytest.raises(ValueError, match=field):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_explicit_source_lineage_mismatch_is_rejected(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["legacy_analysis_metadata"]["legacy_dataset_id"] = "explicit-other-dataset"
    tasks = _tasks(make_task, 1, dataset_id="evomemory_p63")
    with pytest.raises(ValueError, match="legacy_dataset_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_top_level_explicit_source_lineage_mismatch_is_rejected(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["legacy_dataset_id"] = "explicit-other-dataset"
    payload["summary"]["legacy_split_id"] = "test"
    tasks = _tasks(make_task, 1, dataset_id="evomemory_p63")
    with pytest.raises(ValueError, match="legacy_dataset_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)


_NORMALIZED_IDENTITY_KEYS = {
    "mode", "answer_mode", "retrieval_policy", "context_order",
    "context_annotation", "checkpoint_family", "training_seed",
    "memory_trajectory_id", "legacy_run_condition_id", "update_depth", "answer_topk",
    "slot_prompt_variant",
}


def test_old_payload_materializes_every_identity_axis_as_none_or_value(make_task) -> None:
    payload = _payload(count=1)
    manifest, _, _, _ = _import(payload, make_task)
    identity = manifest.prompt_config["legacy_result_import"]["run_identity"]
    assert set(identity) == _NORMALIZED_IDENTITY_KEYS
    assert identity == {
        "mode": "raw_add", "answer_mode": "slot_prompt", "retrieval_policy": None,
        "context_order": None, "context_annotation": None, "checkpoint_family": None,
        "training_seed": None, "memory_trajectory_id": None,
        "legacy_run_condition_id": None, "update_depth": None, "answer_topk": None,
        "slot_prompt_variant": None,
    }


def test_explicit_null_identity_beats_fallback_and_conflict_is_audited(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["mode"] = None
    manifest, _, _, warnings = _import(
        payload, make_task, run_name="raw_add_slot_prompt_k16"
    )
    identity = manifest.prompt_config["legacy_result_import"]["run_identity"]
    assert identity["mode"] is None
    assert "legacy_directory_name_inference" in warnings
    assert "legacy_directory_name_conflict:mode" in warnings


@pytest.mark.parametrize(
    ("raw_field", "canonical_field", "value"),
    [
        ("retrieval_policy", "retrieval_policy", "top_k"),
        ("context_order", "context_order", "ranked"),
        ("annotation", "context_annotation", "latest_outdated"),
        ("checkpoint", "checkpoint_family", "long25/best"),
        ("seed", "training_seed", 7),
        ("memory_trajectory_id", "memory_trajectory_id", "trajectory-a"),
        ("run_condition_id", "legacy_run_condition_id", "condition-a"),
        ("update_depth", "update_depth", 8),
        ("answer_topk", "answer_topk", 4),
    ],
)
def test_each_run_identity_axis_is_normalized_and_changes_run_id(make_task, raw_field: str, canonical_field: str, value: Any) -> None:
    baseline, _, _, _ = _import(_payload(count=1), make_task)
    payload = _payload(count=1)
    payload["summary"][raw_field] = value
    overrides = {}
    if canonical_field == "legacy_run_condition_id":
        tasks = _tasks(make_task, 1)
        provenance = tasks[0].metadata.legacy_provenance.model_dump(mode="python")
        provenance["legacy_run_condition_id"] = value
        tasks[0] = _replace_task(tasks[0], provenance=provenance)
        overrides["task_by_legacy_index"] = tasks
    changed, _, _, _ = _import(payload, make_task, **overrides)
    assert changed.prompt_config["legacy_result_import"]["run_identity"][canonical_field] == value
    assert changed.run_id != baseline.run_id


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("context_annotation", "annotation"),
        ("checkpoint_family", "checkpoint"),
        ("training_seed", "seed"),
        ("legacy_run_condition_id", "run_condition_id"),
    ],
)
def test_run_identity_alias_conflicts_are_rejected(make_task, left: str, right: str) -> None:
    payload = _payload(count=1)
    payload["summary"][left] = "left" if "seed" not in left else 1
    payload["summary"][right] = "right" if "seed" not in right else 2
    with pytest.raises(ValueError, match="conflict"):
        _import(payload, make_task)


def test_documented_name_parser_returns_exact_warning_and_importer_propagates(make_task) -> None:
    parsed = parse_legacy_run_name("raw_add_slot_prompt_k16")
    assert parsed == {
        "mode": "raw_add", "answer_mode": "slot_prompt", "update_depth": 16,
        "warnings": ("legacy_directory_name_inference",),
    }
    _, _, _, warnings = _import(_payload(count=1), make_task, run_name="raw_add_slot_prompt_k16")
    assert warnings.count("legacy_directory_name_inference") == 1


def test_bare_legacy_prediction_does_not_claim_parse_validity(make_task) -> None:
    _, runs, _, _ = _import(_payload(count=1), make_task)
    assert runs[0].answer_predictions == []
    assert runs[0].parser_extractor_provenance.answer_parser_version == "legacy-unavailable"


def test_exact_parse_validity_and_parser_provenance_emit_prediction(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0].update({
        "answer_parse_valid": True,
        "answer_parser_version": "legacy-answer-parser-v1",
        "raw_output": "Qingdao",
    })
    manifest, runs, _, _ = _import(payload, make_task)
    assert len(runs[0].answer_predictions) == 1
    assert runs[0].answer_predictions[0].format_valid is True
    assert runs[0].parser_extractor_provenance.answer_parser_version == "legacy-answer-parser-v1"
    assert manifest.answer_parser_version == "legacy-answer-parser-v1"


def test_mixed_verified_answer_parser_versions_are_rejected(make_task) -> None:
    payload = _payload()
    payload["results"][0].update({"answer_parse_valid": True, "answer_parser_version": "parser-v1"})
    payload["results"][1].update({"answer_parse_valid": True, "answer_parser_version": "parser-v2"})
    with pytest.raises(ValueError, match="answer_parser_version"):
        _import(payload, make_task)


def _claimed_mode(payload: dict[str, Any], legacy_mode: str, canonical_mode: str) -> None:
    payload["summary"]["answer_mode"] = legacy_mode
    payload["summary"]["semantic_compatibility"] = {
        "legacy_answer_mode": legacy_mode,
        "canonical_evaluation_mode": canonical_mode,
    }


def test_state_direct_requires_query_mode_and_canonical_snapshot(make_task) -> None:
    payload = _payload(count=1)
    _claimed_mode(payload, "slot_direct", "state_direct")
    tasks = _tasks(make_task, 1)
    with pytest.raises(ValueError, match="evaluation_mode"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    tasks[0] = _replace_task(tasks[0], query_mode=EvaluationMode.STATE_DIRECT)
    with pytest.raises(ValueError, match="state_direct_trace|snapshot"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    task = tasks[0]
    query = task.queries[0]
    value = payload["results"][0]["predicted"]
    payload["results"][0]["state_direct_trace"] = {
        "query_id": query.query_id,
        "after_event_id": task.events[-1].event_id,
        "object_key": query.target_object_keys[0].canonical_id,
        "value": value,
        "state_by_object": {query.target_object_keys[0].canonical_id: value},
        "store_size": 1,
    }
    manifest, runs, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.prompt_config["legacy_result_import"]["canonical_evaluation_mode"] == "state_direct"
    assert runs[0].memory_snapshots[0].state_by_object[query.target_object_keys[0].canonical_id] == value


def _valid_prompt_trace(payload: dict[str, Any], task) -> None:
    event_id = task.events[-1].event_id
    payload["summary"]["save_answer_traces"] = True
    payload["results"][0]["answer_trace"] = {
        "retrieved_entries": [{
            "id": "entry-current", "content": "current value", "rank": 1,
            "score": 0.9, "source_event_id": event_id,
        }],
        "source_event_ids": [event_id],
        "predicted_answer": payload["results"][0]["predicted"],
        "gold_answer": payload["results"][0]["gold_answer"],
    }


def test_retrieved_prompt_requires_query_mode_and_canonical_entries(make_task) -> None:
    payload = _payload(count=1)
    _claimed_mode(payload, "slot_prompt", "retrieved_prompt")
    tasks = _tasks(make_task, 1)
    tasks[0] = _replace_task(tasks[0], query_mode=EvaluationMode.STATE_DIRECT)
    _valid_prompt_trace(payload, tasks[0])
    with pytest.raises(ValueError, match="evaluation_mode"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    tasks[0] = _replace_task(tasks[0], query_mode=EvaluationMode.RETRIEVED_PROMPT)
    payload["results"][0]["answer_trace"]["retrieved_entries"][0]["source_event_id"] = "missing-event"
    payload["results"][0]["answer_trace"]["source_event_ids"] = ["missing-event"]
    with pytest.raises(ValueError, match="source_event"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    _valid_prompt_trace(payload, tasks[0])
    manifest, runs, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    trace = runs[0].retrieval_traces[0]
    assert manifest.prompt_config["legacy_result_import"]["canonical_evaluation_mode"] == "retrieved_prompt"
    assert [entry.entry_id for entry in trace.retrieved_entries] == ["entry-current"]
    assert trace.scores == [0.9]
    assert trace.ranks == [1]


def test_retrieved_prompt_rejects_empty_or_inconsistent_trace(make_task) -> None:
    payload = _payload(count=1)
    _claimed_mode(payload, "slot_prompt", "retrieved_prompt")
    tasks = _tasks(make_task, 1)
    payload["summary"]["save_answer_traces"] = True
    payload["results"][0]["answer_trace"] = {
        "retrieved_entries": [], "source_event_ids": [],
        "predicted_answer": payload["results"][0]["predicted"],
        "gold_answer": payload["results"][0]["gold_answer"],
    }
    with pytest.raises(ValueError, match="non-empty|retrieved"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_metric_support_distinguishes_inapplicable_and_unsupported(make_task) -> None:
    _, _, scores, _ = _import(_payload(count=1), make_task)
    support = scores[0].supported_metric_fields
    assert support["state_scores.expected_absence_accuracy"].reason is SupportReason.NOT_APPLICABLE
    assert support["state_scores.expected_absence_accuracy"].null_policy == "exclude_from_aggregation"
    assert support["answer_scores.exact_match"].reason is SupportReason.MISSING_ARTIFACT
    assert support["answer_scores.exact_match"].null_policy == "exclude_from_aggregation"


@pytest.mark.parametrize("field", ["answer_exact_match", "stale_same_slot_count"])
def test_explicit_null_nonalias_remains_legacy_and_unsupported(make_task, field: str) -> None:
    payload = _payload(count=1)
    payload["results"][0][field] = None
    _, _, scores, _ = _import(payload, make_task)
    path = {
        "answer_exact_match": "answer_scores.exact_match",
        "stale_same_slot_count": "store_scores.stale_conflicting_value_count",
    }[field]
    assert scores[0].legacy_metrics["legacy_p63"][field] is None
    expected_reason = (
        SupportReason.MISSING_ARTIFACT
        if field == "answer_exact_match"
        else SupportReason.NOT_SUPPORTED
    )
    assert scores[0].supported_metric_fields[path].reason is expected_reason
    assert scores[0].supported_metric_fields[path].null_policy == "exclude_from_aggregation"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer_trace", None),
        ("answer_exact_match", None),
        ("memory", None),
        ("answer_trace", {}),
    ],
)
def test_null_or_empty_only_substantive_fields_cannot_complete(make_task, field: str, value: Any) -> None:
    payload = _payload(count=1)
    payload["results"] = [{
        "example_id": 0,
        "shard_local_example_id": 0,
        field: value,
    }]
    with pytest.raises(ValueError, match="substantive"):
        _import(payload, make_task)


def test_explicit_null_nonalias_can_coexist_with_nonnull_substantive_metric(make_task) -> None:
    payload = _payload(count=1)
    payload["results"] = [{
        "example_id": 0,
        "shard_local_example_id": 0,
        "em": 0.0,
        "answer_exact_match": None,
    }]
    _, _, scores, _ = _import(payload, make_task)
    score = scores[0]
    assert score.supported_metric_fields[
        "answer_scores.exact_match"
    ].reason is SupportReason.MISSING_ARTIFACT
    assert score.legacy_metrics["legacy_p63"]["answer_exact_match"] is None


@pytest.mark.parametrize(
    ("source_value", "supplied_value"),
    [
        (1, 1.0),
        (1, True),
        (1.0, True),
        ({"nested": [1]}, {"nested": [1.0]}),
        ({"nested": [1.0]}, {"nested": [True]}),
    ],
)
def test_source_payload_authentication_distinguishes_exact_json_types(make_task, source_value: Any, supplied_value: Any) -> None:
    source_payload = _payload(count=1)
    source_payload["summary"]["type_probe"] = source_value
    supplied_payload = copy.deepcopy(source_payload)
    supplied_payload["summary"]["type_probe"] = supplied_value
    with pytest.raises(ValueError, match="payload.*source|source.*payload"):
        _import(supplied_payload, make_task, source_payload=source_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer_mode", "slot_prompt"),
        ("checkpoint_family", "long25"),
        ("training_seed", 7),
    ],
)
def test_task_provenance_must_not_embed_result_run_identity(make_task, field: str, value: Any) -> None:
    payload = _payload(count=1)
    tasks = _tasks(make_task, 1)
    original_id = tasks[0].task_id
    provenance = tasks[0].metadata.legacy_provenance.model_dump(mode="python")
    provenance[field] = value
    tasks[0] = _replace_task(tasks[0], provenance=provenance)
    with pytest.raises(ValueError, match=field):
        _import(payload, make_task, task_by_legacy_index=tasks)
    assert tasks[0].task_id == original_id


def test_task_condition_requires_explicit_run_condition_equality_without_inference(make_task) -> None:
    payload = _payload(count=1)
    tasks = _tasks(make_task, 1)
    provenance = tasks[0].metadata.legacy_provenance.model_dump(mode="python")
    provenance["legacy_run_condition_id"] = "condition-a"
    provenance["memory_trajectory_id"] = "task-lineage-trajectory"
    tasks[0] = _replace_task(tasks[0], provenance=provenance)

    manifest, _, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    identity = manifest.prompt_config["legacy_result_import"]["run_identity"]
    assert identity["legacy_run_condition_id"] is None
    assert identity["memory_trajectory_id"] is None

    payload["summary"]["legacy_run_condition_id"] = "condition-b"
    with pytest.raises(ValueError, match="legacy_run_condition_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    payload["summary"]["legacy_run_condition_id"] = "condition-a"
    payload["summary"]["memory_trajectory_id"] = "run-memory-trajectory"
    manifest, _, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.prompt_config["legacy_result_import"]["run_identity"]["memory_trajectory_id"] == "run-memory-trajectory"


def test_fallback_warning_is_emitted_only_when_identity_is_adopted(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["update_depth"] = 16
    _, _, _, warnings = _import(payload, make_task, run_name="raw_add_slot_prompt_k16")
    assert warnings == ["legacy_answer_mode_unverified"]

    payload["summary"]["mode"] = "long25"
    _, _, _, warnings = _import(payload, make_task, run_name="raw_add_slot_prompt_k16")
    assert warnings == ["legacy_directory_name_conflict:mode", "legacy_answer_mode_unverified"]

    del payload["summary"]["update_depth"]
    _, _, _, warnings = _import(payload, make_task, run_name="raw_add_slot_prompt_k16")
    assert warnings == [
        "legacy_directory_name_conflict:mode",
        "legacy_directory_name_inference",
        "legacy_answer_mode_unverified",
    ]


def _state_direct_payload_and_tasks(make_task):
    payload = _payload(count=1)
    _claimed_mode(payload, "slot_direct", "state_direct")
    tasks = _tasks(make_task, 1)
    tasks[0] = _replace_task(tasks[0], query_mode=EvaluationMode.STATE_DIRECT)
    task = tasks[0]
    query = task.queries[0]
    value = payload["results"][0]["predicted"]
    payload["results"][0]["state_direct_trace"] = {
        "query_id": query.query_id,
        "after_event_id": task.events[-1].event_id,
        "object_key": query.target_object_keys[0].canonical_id,
        "value": value,
        "state_by_object": {query.target_object_keys[0].canonical_id: value},
        "store_size": 1,
    }
    return payload, tasks


def test_state_direct_snapshot_must_be_terminal_but_prediction_need_not_equal_gold(make_task) -> None:
    payload, tasks = _state_direct_payload_and_tasks(make_task)
    payload["results"][0]["state_direct_trace"]["after_event_id"] = tasks[0].events[0].event_id
    with pytest.raises(ValueError, match="terminal|after_event_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    payload["results"][0]["state_direct_trace"]["after_event_id"] = max(
        tasks[0].events, key=lambda event: event.sequence_index
    ).event_id
    assert payload["results"][0]["predicted"] != payload["results"][0]["gold_answer"]
    manifest, runs, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.prompt_config["legacy_result_import"]["canonical_evaluation_mode"] == "state_direct"
    assert runs[0].memory_snapshots[0].after_event_id == payload["results"][0]["state_direct_trace"]["after_event_id"]


def test_old_row_gold_must_match_linked_task_gold(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["gold_answer"] = "wrong-gold"
    with pytest.raises(ValueError, match="gold_answer"):
        _import(payload, make_task)


def test_state_direct_and_prompt_gold_must_match_linked_task_gold(make_task) -> None:
    payload, tasks = _state_direct_payload_and_tasks(make_task)
    payload["results"][0]["gold_answer"] = "wrong-gold"
    with pytest.raises(ValueError, match="gold_answer"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    payload = _payload(count=1)
    _claimed_mode(payload, "slot_prompt", "retrieved_prompt")
    tasks = _tasks(make_task, 1)
    _valid_prompt_trace(payload, tasks[0])
    payload["results"][0]["gold_answer"] = "wrong-gold"
    payload["results"][0]["answer_trace"]["gold_answer"] = "wrong-gold"
    with pytest.raises(ValueError, match="gold_answer"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_missing_row_gold_is_allowed_with_linkage_and_other_evidence(make_task) -> None:
    payload = _payload(count=1)
    del payload["results"][0]["gold_answer"]
    manifest, _, _, _ = _import(payload, make_task)
    assert manifest.completed_task_count == 1


def test_authentic_producer_summary_derives_phase_from_task_provenance(make_task) -> None:
    payload = _payload(count=1)
    del payload["summary"]["legacy_analysis_metadata"]
    manifest, runs, _, _ = _import(payload, make_task)
    metadata = manifest.prompt_config["legacy_result_import"]
    assert metadata["namespace"] == "legacy_p63"
    assert runs[0].task_id == "task_legacy_0"


def test_nonzero_shard_and_merged_global_indices_are_supported(make_task) -> None:
    payload = _payload(count=2)
    payload["summary"].update({"start_idx": 10, "end_idx": 12})
    for local_index, row in enumerate(payload["results"]):
        row["example_id"] = 10 + local_index
        row["shard_local_example_id"] = local_index
    tasks = _tasks_for_indices(make_task, [10, 11])
    manifest, runs, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.expected_task_count == 2
    assert [run.task_id for run in runs] == ["task_legacy_10", "task_legacy_11"]

    merged = _payload(count=4)
    merged["summary"]["merged_shards"] = 2
    global_ids = [8, 9, 2, 3]
    local_ids = [0, 1, 0, 1]
    for row, global_id, local_id in zip(merged["results"], global_ids, local_ids, strict=True):
        row["example_id"] = global_id
        row["shard_local_example_id"] = local_id
    tasks = _tasks_for_indices(make_task, [2, 3, 8, 9])
    _, runs, _, _ = _import(merged, make_task, task_by_legacy_index=tasks)
    assert [run.task_id for run in runs] == [
        "task_legacy_2", "task_legacy_3", "task_legacy_8", "task_legacy_9"
    ]


def test_shard_completeness_rejects_contradictory_width_and_missing_global_id(make_task) -> None:
    width_mismatch = _payload(count=2)
    width_mismatch["summary"].update({"start_idx": 10, "end_idx": 13})
    for local_index, row in enumerate(width_mismatch["results"]):
        row["example_id"] = 10 + local_index
        row["shard_local_example_id"] = local_index
    with pytest.raises(ValueError, match="width|num_examples|start_idx|end_idx"):
        _import(
            width_mismatch,
            make_task,
            task_by_legacy_index=_tasks_for_indices(make_task, [10, 11]),
        )

    missing_global = _payload(count=2)
    missing_global["summary"].update({"start_idx": 10, "end_idx": 12})
    missing_global["results"][0]["example_id"] = 10
    missing_global["results"][1]["example_id"] = 12
    with pytest.raises(ValueError, match="missing|range|bounds|start_idx|end_idx"):
        _import(
            missing_global,
            make_task,
            task_by_legacy_index=_tasks_for_indices(make_task, [10, 12]),
        )


def test_shard_completeness_validates_total_and_single_shard_local_indices(make_task) -> None:
    impossible_total = _payload(count=2)
    impossible_total["summary"].update({
        "start_idx": 10,
        "end_idx": 12,
        "total_examples": 11,
    })
    for local_index, row in enumerate(impossible_total["results"]):
        row["example_id"] = 10 + local_index
        row["shard_local_example_id"] = local_index
    tasks = _tasks_for_indices(make_task, [10, 11])
    with pytest.raises(ValueError, match="total_examples|total"):
        _import(impossible_total, make_task, task_by_legacy_index=tasks)

    malformed_total = copy.deepcopy(impossible_total)
    malformed_total["summary"]["total_examples"] = True
    with pytest.raises(ValueError, match="total_examples|integer"):
        _import(malformed_total, make_task, task_by_legacy_index=tasks)

    bad_local = copy.deepcopy(impossible_total)
    bad_local["summary"]["total_examples"] = 20
    bad_local["results"][1]["shard_local_example_id"] = 2
    with pytest.raises(ValueError, match="shard_local_example_id|contiguous"):
        _import(bad_local, make_task, task_by_legacy_index=tasks)


def test_merged_shard_marker_is_strict_and_allows_restarted_local_indices(make_task) -> None:
    payload = _payload(count=4)
    payload["summary"]["merged_shards"] = 2
    global_ids = [2, 3, 8, 9]
    local_ids = [0, 1, 0, 1]
    for row, global_id, local_id in zip(
        payload["results"], global_ids, local_ids, strict=True
    ):
        row["example_id"] = global_id
        row["shard_local_example_id"] = local_id
    tasks = _tasks_for_indices(make_task, global_ids)
    manifest, _, _, _ = _import(
        payload, make_task, task_by_legacy_index=tasks
    )
    assert manifest.completed_task_count == 4

    for malformed in (True, 0, -1):
        bad = copy.deepcopy(payload)
        bad["summary"]["merged_shards"] = malformed
        with pytest.raises(ValueError, match="merged_shards|integer"):
            _import(bad, make_task, task_by_legacy_index=tasks)


def test_total_examples_constrains_every_global_id_without_explicit_interval(make_task) -> None:
    out_of_domain = _payload(count=2)
    out_of_domain["summary"].update({
        "total_examples": 10,
        "merged_shards": 2,
    })
    out_of_domain["results"][0].update({
        "example_id": 0,
        "shard_local_example_id": 0,
    })
    out_of_domain["results"][1].update({
        "example_id": 10,
        "shard_local_example_id": 0,
    })
    with pytest.raises(ValueError, match="total_examples|global|domain"):
        _import(
            out_of_domain,
            make_task,
            task_by_legacy_index=_tasks_for_indices(make_task, [0, 10]),
        )

    zero_total = _payload(count=1)
    zero_total["summary"]["total_examples"] = 0
    with pytest.raises(ValueError, match="total_examples|global|domain"):
        _import(zero_total, make_task)


def test_total_examples_allows_upper_boundary_and_noncontiguous_in_domain_globals(make_task) -> None:
    upper_boundary = _payload(count=1)
    upper_boundary["summary"]["total_examples"] = 10
    upper_boundary["results"][0]["example_id"] = 9
    manifest, _, _, _ = _import(
        upper_boundary,
        make_task,
        task_by_legacy_index=_tasks_for_indices(make_task, [9]),
    )
    assert manifest.completed_task_count == 1

    noncontiguous = _payload(count=3)
    noncontiguous["summary"]["total_examples"] = 10
    for local_index, (row, global_id) in enumerate(
        zip(noncontiguous["results"], [0, 5, 9], strict=True)
    ):
        row["example_id"] = global_id
        row["shard_local_example_id"] = local_index
    manifest, runs, _, _ = _import(
        noncontiguous,
        make_task,
        task_by_legacy_index=_tasks_for_indices(make_task, [0, 5, 9]),
    )
    assert manifest.completed_task_count == 3
    assert [run.task_id for run in runs] == [
        "task_legacy_0",
        "task_legacy_5",
        "task_legacy_9",
    ]


def test_merged_local_indices_encode_exact_contiguous_shard_sequences(make_task) -> None:
    valid = _payload(count=5)
    valid["summary"]["merged_shards"] = 2
    valid_locals = [0, 1, 2, 0, 1]
    for row, local_index in zip(
        valid["results"], valid_locals, strict=True
    ):
        row["shard_local_example_id"] = local_index
    manifest, _, _, _ = _import(valid, make_task)
    assert manifest.completed_task_count == 5

    for invalid_locals in (
        [99, 99, 99, 99, 99],
        [0, 2, 0, 2, 2],
        [0, 1, 0, 1, 1],
    ):
        invalid = copy.deepcopy(valid)
        for row, local_index in zip(
            invalid["results"], invalid_locals, strict=True
        ):
            row["shard_local_example_id"] = local_index
        with pytest.raises(
            ValueError,
            match="shard_local_example_id|merged|contiguous|multiplicit",
        ):
            _import(invalid, make_task)


def test_global_indices_respect_explicit_shard_bounds(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({"start_idx": 10, "end_idx": 11})
    payload["results"][0]["example_id"] = 12
    tasks = _tasks_for_indices(make_task, [12])
    with pytest.raises(ValueError, match="start_idx|end_idx|bounds"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_calibrated_budget_accepts_production_sized_k16_run(make_task) -> None:
    payload = _payload(count=100)
    for row in payload["results"]:
        row["slot_actions"] = [
            {"operation": "UPDATE", "entity": "friend_alex", "attribute": "location", "value": f"city-{index}"}
            for index in range(16)
        ]
    manifest, runs, _, _ = _import(payload, make_task)
    assert manifest.completed_task_count == 100
    assert len(runs) == 100


def test_calibrated_per_row_budget_rejects_oversized_payload(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["slot_actions"] = [None] * 100_001
    with pytest.raises(ValueError, match="per-row|node budget"):
        _import(payload, make_task)


@pytest.mark.parametrize(
    "credential_key",
    [
        "x-api-key", "x_api_key", "api-key", "access-token",
        "bearer_token", "id_token", "set-cookie", "MiXeD-X_Api-Key",
    ],
)
def test_credential_key_variants_are_rejected_without_canary_leak(make_task, credential_key: str) -> None:
    payload = _payload(count=1)
    payload["results"][0]["answer_trace"] = {
        "nested": {credential_key: "CANARY-SECRET-VALUE"}
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "CANARY-SECRET-VALUE" not in str(exc_info.value)


@pytest.mark.parametrize(
    "credential_key",
    [
        "auth_token",
        "API-TOKEN",
        "authentication.token",
        "api-key-backup",
        "private_key",
        "PrIvAtE-Key-Backup",
    ],
)
def test_credential_stems_and_aliases_are_rejected_recursively(make_task, credential_key: str) -> None:
    payload = _payload(count=1)
    payload["results"][0]["answer_trace"] = {
        "nested": {credential_key: "SECOND-CANARY-SECRET"}
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "SECOND-CANARY-SECRET" not in str(exc_info.value)


@pytest.mark.parametrize(
    "telemetry_key",
    [
        "token_count",
        "prompt_tokens",
        "completion_tokens",
        "input_token_count",
        "output_token_count",
        "total_token_count",
    ],
)
def test_benign_token_telemetry_names_remain_importable(make_task, telemetry_key: str) -> None:
    payload = _payload(count=1)
    payload["results"][0][telemetry_key] = 17
    _, runs, _, _ = _import(payload, make_task)
    assert _legacy_event(runs[0])["raw_row"][telemetry_key] == 17


def _put_typed_token_field(
    payload: dict[str, Any],
    location: str,
    key: str,
    value: Any,
) -> None:
    if location == "summary":
        payload["summary"][key] = value
    elif location == "row":
        payload["results"][0][key] = value
    else:
        payload["results"][0]["answer_trace"] = {
            "runtime_config": {key: value}
        }


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "max_new_tokens",
        "min_tokens",
        "min_new_tokens",
        "token_budget",
    ],
)
@pytest.mark.parametrize("value", [0, 17])
def test_typed_token_integer_fields_are_allowed_at_runtime_locations(
    make_task,
    location: str,
    field: str,
    value: int,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, location, field, value)
    manifest, runs, _, _ = _import(payload, make_task)
    assert manifest.completed_task_count == 1
    assert runs[0].completion_status.value == "completed"


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
@pytest.mark.parametrize("value", [0, 7, 3.5])
def test_tokens_per_second_accepts_finite_nonnegative_numbers(
    make_task,
    location: str,
    value: int | float,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, location, "tokens_per_second", value)
    manifest, _, _, _ = _import(payload, make_task)
    assert manifest.completed_task_count == 1


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "max_new_tokens",
        "min_tokens",
        "min_new_tokens",
        "token_budget",
    ],
)
@pytest.mark.parametrize(
    "malformed",
    [True, -1, 1.5, "TYPED-TOKEN-CANARY", {"payload": "TYPED-TOKEN-CANARY"}],
)
def test_typed_token_integer_fields_reject_wrong_types_without_value_echo(
    make_task,
    field: str,
    malformed: Any,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, "nested_runtime", field, malformed)
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "TYPED-TOKEN-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    "malformed",
    [
        True,
        -1,
        -0.5,
        float("inf"),
        float("nan"),
        "THROUGHPUT-CANARY",
        {"payload": "THROUGHPUT-CANARY"},
    ],
)
def test_tokens_per_second_rejects_invalid_values_without_value_echo(
    make_task,
    malformed: Any,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(
        payload,
        "nested_runtime",
        "tokens_per_second",
        malformed,
    )
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "THROUGHPUT-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    "credential_key",
    [
        "token",
        "session_token",
        "github-token",
        "provider.token",
        "AUTHENTICATION_TOKEN",
        "Api_Token",
    ],
)
def test_generic_token_credentials_are_rejected_after_safe_allowlist(make_task, credential_key: str) -> None:
    payload = _payload(count=1)
    payload["results"][0]["answer_trace"] = {
        "nested": {credential_key: "GENERIC-TOKEN-CANARY"}
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "GENERIC-TOKEN-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    "metadata_key",
    ["tokenizer", "tokenizer_name", "tokenizer_version", "tokenization"],
)
def test_nonsecret_tokenizer_metadata_remains_importable(make_task, metadata_key: str) -> None:
    payload = _payload(count=1)
    payload["results"][0][metadata_key] = "legacy-tokenizer"
    _, runs, _, _ = _import(payload, make_task)
    assert (
        _legacy_event(runs[0])["raw_row"][metadata_key]
        == "legacy-tokenizer"
    )


@pytest.mark.parametrize(
    "credential_key",
    [
        "ｘ－ａｐｉ－ｋｅｙ",
        "ａｃｃｅｓｓ＿ｔｏｋｅｎ",
        "ｉｄ＿ｔｏｋｅｎ",
        "ＡＰＩ＿ＫＥＹ",
        "ｓｅｓｓｉｏｎ＿ｔｏｋｅｎ",
    ],
)
def test_unicode_compatibility_credential_keys_are_rejected_recursively(
    make_task,
    credential_key: str,
) -> None:
    payload = _payload(count=1)
    payload["results"][0]["answer_trace"] = {
        "nested": {credential_key: "UNICODE-CREDENTIAL-CANARY"}
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "UNICODE-CREDENTIAL-CANARY" not in str(exc_info.value)


def _raw_telemetry_value(manifest, run, location: str, key: str) -> Any:
    if location == "summary":
        return manifest.prompt_config["legacy_result_import"]["raw_summary"][key]
    raw_row = _legacy_event(run)["raw_row"]
    if location == "row":
        return raw_row[key]
    return raw_row["answer_trace"]["runtime_config"][key]


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
def test_scalar_token_telemetry_is_typed_and_raw_preserved(make_task, location: str) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, location, "input_tokens", 17)
    manifest, runs, _, _ = _import(payload, make_task)
    assert _raw_telemetry_value(manifest, runs[0], location, "input_tokens") == 17


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
@pytest.mark.parametrize("aggregate_key", ["token_usage", "token_counts"])
def test_aggregate_token_telemetry_is_typed_and_raw_preserved(
    make_task,
    location: str,
    aggregate_key: str,
) -> None:
    payload = _payload(count=1)
    value = {"input_tokens": 11, "output_token_count": 6}
    _put_typed_token_field(payload, location, aggregate_key, value)
    manifest, runs, _, _ = _import(payload, make_task)
    assert _raw_telemetry_value(
        manifest, runs[0], location, aggregate_key
    ) == value


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
@pytest.mark.parametrize(
    "malformed",
    [
        "SCALAR-TELEMETRY-CANARY",
        ["SCALAR-TELEMETRY-CANARY"],
        {"payload": "SCALAR-TELEMETRY-CANARY"},
        True,
        -1,
    ],
)
def test_scalar_token_telemetry_rejects_malformed_values_without_echo(
    make_task,
    location: str,
    malformed: Any,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, location, "input_tokens", malformed)
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "SCALAR-TELEMETRY-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row", "nested_runtime"])
@pytest.mark.parametrize("aggregate_key", ["token_usage", "token_counts"])
@pytest.mark.parametrize(
    "malformed",
    [
        "AGGREGATE-TELEMETRY-CANARY",
        ["AGGREGATE-TELEMETRY-CANARY"],
        {"input_tokens": {"nested": "AGGREGATE-TELEMETRY-CANARY"}},
        {"input_tokens": ["AGGREGATE-TELEMETRY-CANARY"]},
        {"input_tokens": "AGGREGATE-TELEMETRY-CANARY"},
        {"input_tokens": True},
        {"input_tokens": -1},
        {"requests": 1},
    ],
)
def test_aggregate_token_telemetry_rejects_malformed_values_without_echo(
    make_task,
    location: str,
    aggregate_key: str,
    malformed: Any,
) -> None:
    payload = _payload(count=1)
    _put_typed_token_field(payload, location, aggregate_key, malformed)
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "AGGREGATE-TELEMETRY-CANARY" not in str(exc_info.value)


def test_token_telemetry_rejects_hostile_container_and_leaf_subclasses(make_task) -> None:
    class HostileDict(dict):
        calls = 0

        def items(self):
            self.calls += 1
            raise AssertionError("hostile telemetry hook must not run")

    class HostileInt(int):
        pass

    for hostile in (
        HostileDict({"input_tokens": 1}),
        {"input_tokens": HostileInt(1)},
    ):
        payload = _payload(count=1)
        payload["results"][0]["token_usage"] = hostile
        with pytest.raises((TypeError, ValueError)):
            _import(payload, make_task, source_payload=_payload(count=1))
    assert HostileDict.calls == 0


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    ("container_key", "leaf_key"),
    [
        ("credentials", "key"),
        ("auth", "token"),
        ("secrets", "value"),
        ("credential_store", "value"),
        ("provider_auth", "key"),
        ("ＣＲＥＤＥＮＴＩＡＬＳ", "ＫＥＹ"),
        ("ＡＵＴＨ", "ＴＯＫＥＮ"),
        ("ＳＥＣＲＥＴＳ", "ＶＡＬＵＥ"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_sensitive_ancestry_rejects_generic_secret_leaves_without_echo(
    make_task,
    location: str,
    container_key: str,
    leaf_key: str,
    nested: bool,
) -> None:
    payload = _payload(count=1)
    private_payload = {
        container_key: {leaf_key: "sk-test-secret-value"}
    }
    value = {"runtime": private_payload} if nested else private_payload
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target.update(value)
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "sk-test-secret-value" not in str(exc_info.value)


def test_sensitive_container_screening_preserves_unrelated_benign_keys(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "author": {"key": "Ada", "value": "writer"},
        "authorship": {"value": "human"},
        "secretary": {"value": "assistant"},
        "authentication_method": "none",
    })
    payload["results"][0].update({
        "object_key": "friend_alex.location",
        "token_usage": {"input_tokens": 11, "output_tokens": 4},
    })
    manifest, runs, _, _ = _import(payload, make_task)
    raw_summary = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    raw_row = _legacy_event(runs[0])["raw_row"]
    assert raw_summary["author"]["key"] == "Ada"
    assert raw_summary["authorship"]["value"] == "human"
    assert raw_summary["secretary"]["value"] == "assistant"
    assert raw_summary["authentication_method"] == "none"
    assert raw_row["object_key"] == "friend_alex.location"
    assert raw_row["token_usage"] == {
        "input_tokens": 11,
        "output_tokens": 4,
    }


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    "credential_key",
    [
        "аpi_key",
        "api_κey",
        "pаssword",
        "passwοrd",
        "ѕecret_access_key",
    ],
)
def test_mixed_script_credential_confusables_are_rejected_without_echo(
    make_task,
    location: str,
    credential_key: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target[credential_key] = "MIXED-SCRIPT-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "MIXED-SCRIPT-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    ("container_key", "structured_value"),
    [
        (
            "headers",
            [{"name": "X-Api-Key", "value": "STRUCTURED-CANARY"}],
        ),
        (
            "headers",
            [["Authorization", "STRUCTURED-CANARY"]],
        ),
        (
            "config",
            [{"name": "AWS_SECRET_ACCESS_KEY", "value": "STRUCTURED-CANARY"}],
        ),
        (
            "env",
            [["SECRET_ACCESS_KEY", "STRUCTURED-CANARY"]],
        ),
        (
            "environment",
            {"name": "api_key", "value": "STRUCTURED-CANARY"},
        ),
    ],
)
def test_structured_credential_records_and_pairs_are_rejected_without_echo(
    make_task,
    location: str,
    container_key: str,
    structured_value: Any,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target[container_key] = structured_value
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "STRUCTURED-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    "credential_key",
    ["aws_secret_access_key", "secret_access_key"],
)
def test_exact_aws_secret_aliases_are_rejected_without_echo(
    make_task,
    location: str,
    credential_key: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target[credential_key] = "AWS-CREDENTIAL-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "AWS-CREDENTIAL-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
def test_sensitive_key_names_and_status_values_are_redacted_from_errors(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["credentials-RAW-KEY-CANARY"] = {"key": "SECRET-VALUE-CANARY"}
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "RAW-KEY-CANARY" not in message
    assert "SECRET-VALUE-CANARY" not in message

    status_payload = _payload(count=1)
    status_target = (
        status_payload["summary"]
        if location == "summary"
        else status_payload["results"][0]
    )
    status_target["status"] = "STATUS-VALUE-CANARY"
    with pytest.raises(ValueError) as status_exc_info:
        _import(status_payload, make_task)
    assert "STATUS-VALUE-CANARY" not in str(status_exc_info.value)


def test_oversized_keys_and_deep_paths_fail_with_bounded_locations(make_task) -> None:
    oversized = _payload(count=1)
    oversized["summary"]["K" * 20_000] = "value"
    with pytest.raises(ValueError) as oversized_exc:
        _import(oversized, make_task)
    assert len(str(oversized_exc.value)) < 1_000

    deep = _payload(count=1)
    current = deep["summary"]
    for depth in range(70):
        child: dict[str, Any] = {}
        current[f"segment-{depth}-" + "x" * 80] = child
        current = child
    with pytest.raises(ValueError) as deep_exc:
        _import(deep, make_task)
    assert len(str(deep_exc.value)) < 1_000


def test_source_authentication_precedes_caller_payload_traversal(
    make_task,
    tmp_path: Path,
) -> None:
    class HostileDict(dict):
        calls = 0

        def items(self):
            self.calls += 1
            raise AssertionError("caller payload hook must not run")

    source_payload = _payload(count=1)
    source = tmp_path / "authenticated-first.json"
    _write_source(source, source_payload)
    caller_payload = _payload(count=1)
    caller_payload["summary"]["hostile"] = HostileDict()
    with pytest.raises(ValueError, match="source_sha256"):
        import_evomemory_results(
            caller_payload,
            source_path=source,
            source_sha256="b" * 64,
            run_name=None,
            task_by_legacy_index=_tasks(make_task, 1),
        )
    assert HostileDict.calls == 0


def test_benign_tokenizer_and_precise_password_cookie_names_are_preserved(
    make_task,
) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "tokenizer_config": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "special_tokens_map": {"pad_token": "<pad>"},
        },
        "passwordless": True,
        "cookiecutter": {"template": "benchmark"},
        "config": {"key": "model-key", "value": "benign"},
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["tokenizer_config"]["bos_token"] == "<s>"
    assert raw["tokenizer_config"]["special_tokens_map"]["pad_token"] == "<pad>"
    assert raw["passwordless"] is True
    assert raw["cookiecutter"]["template"] == "benchmark"
    assert raw["config"]["key"] == "model-key"


def test_tokenizer_config_still_rejects_auth_token_without_echo(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["tokenizer_config"] = {
        "auth_token": "TOKENIZER-AUTH-CANARY"
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "TOKENIZER-AUTH-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    "credential_key",
    ["priνate_key", "рrivate_key", "privаte_key"],
)
def test_private_key_confusables_are_rejected_without_multilingual_false_positives(
    make_task,
    credential_key: str,
) -> None:
    payload = _payload(count=1)
    payload["summary"][credential_key] = "PRIVATE-KEY-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "PRIVATE-KEY-CANARY" not in str(exc_info.value)

    benign = _payload(count=1)
    benign["summary"].update({
        "model_νersion": "multilingual-metadata",
        "greek_label_α": "alpha",
    })
    manifest, _, _, _ = _import(benign, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["model_νersion"] == "multilingual-metadata"
    assert raw["greek_label_α"] == "alpha"


@pytest.mark.parametrize(
    ("container_key", "record"),
    [
        ("config", {"key": "api_key", "value": "ALT-NAME-CANARY"}),
        (
            "environment",
            {"variable": "AWS_SECRET_ACCESS_KEY", "value": "ALT-NAME-CANARY"},
        ),
        ("env", {"key": "secret_access_key", "value": "ALT-NAME-CANARY"}),
    ],
)
def test_structured_alternate_identifier_fields_are_rejected_without_echo(
    make_task,
    container_key: str,
    record: dict[str, Any],
) -> None:
    payload = _payload(count=1)
    payload["summary"][container_key] = record
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "ALT-NAME-CANARY" not in str(exc_info.value)


def test_structured_alternate_identifiers_preserve_benign_record_shapes(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "config": {"key": "model_name", "value": "legacy-model"},
        "environment": {"variable": "HOME", "value": "/tmp/legacy"},
        "headers": [["Accept", "application/json"]],
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["config"]["key"] == "model_name"
    assert raw["environment"]["variable"] == "HOME"
    assert raw["headers"][0][0] == "Accept"


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    "headers",
    [
        [{"key": "X-Api-Key", "value": "HEADER-KEY-CANARY"}],
        {"key": "Authorization", "value": "Bearer HEADER-KEY-CANARY"},
    ],
)
def test_header_key_value_records_are_rejected_without_echo(
    make_task,
    location: str,
    headers: Any,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["headers"] = headers
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "HEADER-KEY-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    ("container_key", "record", "structural_key"),
    [
        ("config", {"kеy": "api_key", "value": "STRUCTURAL-CANARY"}, "kеy"),
        ("config", {"key": "api_key", "vаlue": "STRUCTURAL-CANARY"}, "vаlue"),
        (
            "environment",
            {"variаble": "AWS_SECRET_ACCESS_KEY", "value": "STRUCTURAL-CANARY"},
            "variаble",
        ),
        (
            "headers",
            {"nаme": "Authorization", "value": "Bearer STRUCTURAL-CANARY"},
            "nаme",
        ),
        (
            "headers",
            {"kеy": "X-Api-Key", "vаlue": "STRUCTURAL-CANARY"},
            "kеy",
        ),
    ],
)
def test_confusable_structural_fields_fail_closed_without_echo(
    make_task,
    location: str,
    container_key: str,
    record: dict[str, Any],
    structural_key: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target[container_key] = record
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "STRUCTURAL-CANARY" not in message
    assert structural_key not in message


@pytest.mark.parametrize("location", ["summary", "row"])
def test_ambiguous_structural_field_aliases_fail_closed_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["headers"] = {
        "k-e-y": "Authorization",
        "key": "Accept",
        "value": "COLLISION-CANARY",
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "COLLISION-CANARY" not in message
    assert "k-e-y" not in message


def test_structured_field_screening_preserves_benign_header_and_multilingual_metadata(
    make_task,
) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "headers": [
            {"key": "Accept", "value": "application/json"},
            {"name": "Content-Type", "value": "application/json"},
            ["Accept-Language", "en"],
        ],
        "config": {
            "display_κey": "model-name",
            "greek_label_α": "alpha",
        },
        "unstructured_metadata": {
            "kеy": "model_name",
            "vаlue": "legacy-model",
        },
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["headers"][0]["key"] == "Accept"
    assert raw["headers"][1]["name"] == "Content-Type"
    assert raw["config"]["display_κey"] == "model-name"
    assert raw["unstructured_metadata"]["kеy"] == "model_name"


@pytest.mark.parametrize("location", ["summary", "row"])
@pytest.mark.parametrize(
    "credential_key",
    ["ɑpi_key", "apı_key"],
)
def test_latin_credential_confusables_are_rejected_without_echo(
    make_task,
    location: str,
    credential_key: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target[credential_key] = "LATIN-CONFUSABLE-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "LATIN-CONFUSABLE-CANARY" not in message
    assert credential_key not in message


@pytest.mark.parametrize("location", ["summary", "row"])
def test_unmapped_greek_credential_confusable_is_rejected_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["seϲret"] = "CONFUSABLE-SECRET-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "CONFUSABLE-SECRET-CANARY" not in message
    assert "seϲret" not in message


@pytest.mark.parametrize("location", ["summary", "row"])
def test_unmapped_cyrillic_structural_confusable_is_rejected_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["environment"] = {
        "varӏable": "AWS_SECRET_ACCESS_KEY",
        "value": "STRUCTURAL-CANARY",
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "STRUCTURAL-CANARY" not in message
    assert "varӏable" not in message


def test_unmapped_mixed_script_controls_remain_benign_inside_and_outside_ancestry(
    make_task,
) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "model_ϲonfig": "multilingual-metadata",
        "environment": {
            "variable": "MODEL_NAME",
            "value": "mӏetadata",
            "ϲustom": "benign-field",
        },
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["model_ϲonfig"] == "multilingual-metadata"
    assert raw["environment"]["value"] == "mӏetadata"
    assert raw["environment"]["ϲustom"] == "benign-field"


@pytest.mark.parametrize("location", ["summary", "row"])
def test_latin_confusable_structured_variable_is_rejected_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["environment"] = {
        "varıable": "AWS_SECRET_ACCESS_KEY",
        "value": "LATIN-STRUCTURAL-CANARY",
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "LATIN-STRUCTURAL-CANARY" not in message
    assert "varıable" not in message


def test_latin_confusable_controls_remain_benign(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "model_ɑlpha": "multilingual-metadata",
        "access_count": 3,
        "environment": {
            "variable": "MODEL_NAME",
            "value": "benign-metadata",
        },
    })
    manifest, _, _, _ = _import(payload, make_task)
    raw = manifest.prompt_config["legacy_result_import"]["raw_summary"]
    assert raw["model_ɑlpha"] == "multilingual-metadata"
    assert raw["access_count"] == 3
    assert raw["environment"]["variable"] == "MODEL_NAME"


@pytest.mark.parametrize("location", ["summary", "row"])
def test_structured_tokenizer_name_metadata_is_accepted_with_exact_string(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["environment"] = {
        "variable": "TOKENIZER_NAME",
        "value": "benign-metadata",
    }
    manifest, runs, _, _ = _import(payload, make_task)
    raw = (
        manifest.prompt_config["legacy_result_import"]["raw_summary"]
        if location == "summary"
        else _legacy_event(runs[0])["raw_row"]
    )
    assert raw["environment"]["value"] == "benign-metadata"


@pytest.mark.parametrize("identifier", ["AUTH_TOKEN", "API_TOKEN"])
def test_structured_tokenizer_like_credentials_are_rejected_without_echo(
    make_task,
    identifier: str,
) -> None:
    payload = _payload(count=1)
    payload["summary"]["environment"] = {
        "variable": identifier,
        "value": "STRUCTURED-TOKEN-CANARY",
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "STRUCTURED-TOKEN-CANARY" not in str(exc_info.value)


def test_structured_tokenizer_name_requires_exact_string_value(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["environment"] = {
        "variable": "TOKENIZER_NAME",
        "value": 123,
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "123" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
def test_aws_access_key_id_is_rejected_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["aws_access_key_id"] = "AWS-ACCESS-ID-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "AWS-ACCESS-ID-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("location", ["summary", "row"])
def test_structured_aws_access_key_id_is_rejected_without_echo(
    make_task,
    location: str,
) -> None:
    payload = _payload(count=1)
    target = (
        payload["summary"]
        if location == "summary"
        else payload["results"][0]
    )
    target["environment"] = {
        "variable": "AWS_ACCESS_KEY_ID",
        "value": "STRUCTURED-AWS-ID-CANARY",
    }
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "STRUCTURED-AWS-ID-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("field", ["access_count", "access_latency_ms"])
def test_access_telemetry_remains_benign(make_task, field: str) -> None:
    payload = _payload(count=1)
    payload["summary"][field] = 3
    manifest, _, _, _ = _import(payload, make_task)
    assert manifest.prompt_config["legacy_result_import"]["raw_summary"][field] == 3


@pytest.mark.parametrize("field", ["tokenized_text", "tokenized"])
def test_free_form_tokenized_fields_are_not_benign_metadata(
    make_task,
    field: str,
) -> None:
    payload = _payload(count=1)
    payload["summary"][field] = "sk-test-TOKENIZED-CANARY"
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    assert "TOKENIZED-CANARY" not in str(exc_info.value)


def test_importer_redacts_duplicate_sensitive_source_keys(
    make_task,
    tmp_path: Path,
) -> None:
    raw_key = "authorization-Bearer-DUPLICATE-CANARY"
    source = tmp_path / "duplicate-sensitive-results.json"
    source.write_text(
        '{"summary":{"'
        + raw_key
        + '":1,"'
        + raw_key
        + '":2},"results":[]}',
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate JSON key") as exc_info:
        import_evomemory_results(
            _payload(count=1),
            source_path=source,
            source_sha256=digest,
            run_name=None,
            task_by_legacy_index=_tasks(make_task, 1),
        )
    message = str(exc_info.value)
    assert raw_key not in message
    assert "DUPLICATE-CANARY" not in message
    assert len(message) < 800


def test_token_usage_is_not_treated_as_credential(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["token_usage"] = {
        "input_tokens": 10,
        "output_tokens": 7,
    }
    _, runs, _, _ = _import(payload, make_task)
    assert _legacy_event(runs[0])["raw_row"]["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
    }


def test_historical_model_checkpoint_and_prompt_variant_identity(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "model_name": "legacy-model",
        "lora_checkpoint": "checkpoints/long25/best",
        "slot_prompt_variant": "minimal",
    })
    manifest, _, _, _ = _import(payload, make_task)
    identity = manifest.prompt_config["legacy_result_import"]["run_identity"]
    assert manifest.model_name == "legacy-model"
    assert identity["checkpoint_family"] == "checkpoints/long25/best"
    assert identity["slot_prompt_variant"] == "minimal"

    conflict = copy.deepcopy(payload)
    conflict["summary"]["checkpoint_family"] = "other"
    with pytest.raises(ValueError, match="conflict"):
        _import(conflict, make_task)


def test_generation_identity_populates_manifest_and_binds_run_and_config_ids(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "model_name": "legacy-model",
        "provider": "legacy-provider",
        "model_revision": "revision-a",
    })
    baseline, _, _, _ = _import(payload, make_task)
    assert baseline.model_name == "legacy-model"
    assert baseline.provider == "legacy-provider"
    assert baseline.model_revision == "revision-a"

    for field, changed_value in (
        ("model_name", "other-model"),
        ("provider", "other-provider"),
        ("model_revision", "revision-b"),
    ):
        changed_payload = copy.deepcopy(payload)
        changed_payload["summary"][field] = changed_value
        changed, _, _, _ = _import(changed_payload, make_task)
        assert changed.run_id != baseline.run_id
        assert (
            changed.adapter_info.configuration_hash
            != baseline.adapter_info.configuration_hash
        )


def test_generation_identity_is_strict_optional_and_model_aliases_must_agree(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"].update({
        "model_name": None,
        "provider": None,
        "model_revision": None,
    })
    manifest, _, _, _ = _import(payload, make_task)
    assert manifest.model_name is None
    assert manifest.provider is None
    assert manifest.model_revision is None

    for field, malformed in (
        ("model_name", ""),
        ("provider", True),
        ("model_revision", "  "),
    ):
        bad = _payload(count=1)
        bad["summary"][field] = malformed
        with pytest.raises(ValueError, match=field):
            _import(bad, make_task)

    conflict = _payload(count=1)
    conflict["summary"].update({"model_name": "model-a", "model": "model-b"})
    with pytest.raises(ValueError, match="model_name|conflict"):
        _import(conflict, make_task)


def test_slot_prompt_variant_is_materialized_as_none_when_absent(make_task) -> None:
    manifest, _, _, _ = _import(_payload(count=1), make_task)
    assert manifest.prompt_config["legacy_result_import"]["run_identity"]["slot_prompt_variant"] is None


def test_final_source_toctou_check_rejects_late_mutation(make_task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mub.vnext.legacy.results as result_module

    payload = _payload(count=1)
    source = tmp_path / "late-change.json"
    digest = _write_source(source, payload)
    tasks = _tasks(make_task, 1)
    original_score_record = result_module._score_record
    mutated = False

    def mutate_after_early_verification(*args, **kwargs):
        nonlocal mutated
        result = original_score_record(*args, **kwargs)
        if not mutated:
            source.write_text("{}", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(result_module, "_score_record", mutate_after_early_verification)
    with pytest.raises(RuntimeError, match="changed"):
        import_evomemory_results(payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)


def test_final_source_toctou_sandwich_rejects_mutation_after_digest(make_task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mub.vnext.legacy.results as result_module

    payload = _payload(count=1)
    source = tmp_path / "post-digest-change.json"
    digest = _write_source(source, payload)
    tasks = _tasks(make_task, 1)
    real_hash = result_module._sha256_file
    hash_calls = 0

    def mutate_immediately_after_final_digest(path: Path) -> str:
        nonlocal hash_calls
        value = real_hash(path)
        hash_calls += 1
        if hash_calls == 3:
            path.write_text("{}", encoding="utf-8")
        return value

    monkeypatch.setattr(
        result_module,
        "_sha256_file",
        mutate_immediately_after_final_digest,
    )
    with pytest.raises(RuntimeError, match="changed"):
        import_evomemory_results(
            payload,
            source_path=source,
            source_sha256=digest,
            run_name=None,
            task_by_legacy_index=tasks,
        )


def test_large_declared_count_error_is_bounded(make_task) -> None:
    payload = _payload(count=1)
    payload["summary"]["num_examples"] = 1_000_000
    with pytest.raises(ValueError) as exc_info:
        _import(payload, make_task)
    message = str(exc_info.value)
    assert "1000000" in message
    assert len(message) < 800


def test_task_map_hash_and_run_id_bind_canonical_task_content(make_task, tmp_path: Path) -> None:
    payload = _payload(count=1)
    source = tmp_path / "task-binding.json"
    digest = _write_source(source, payload)
    tasks = _tasks(make_task, 1)
    first, _, _, _ = import_evomemory_results(payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index=tasks)

    task_data = tasks[0].model_dump(mode="python")
    task_data["metadata"]["legacy_provenance"]["known_caveats"] = ["content-change"]
    changed_task = type(tasks[0]).model_validate(task_data)
    assert changed_task.task_id == tasks[0].task_id
    second, _, _, _ = import_evomemory_results(payload, source_path=source, source_sha256=digest, run_name=None, task_by_legacy_index={0: changed_task})
    assert first.task_manifest.sha256 != second.task_manifest.sha256
    assert first.run_id != second.run_id


def _with_second_query(task):
    data = task.model_dump(mode="python")
    second = copy.deepcopy(data["queries"][0])
    second["query_id"] = "query_1"
    second["text"] = "Second query"
    data["queries"].append(second)
    data["gold"]["gold_answers"]["query_1"] = data["gold"]["gold_answers"]["query_0"]
    data["gold"]["acceptable_answers"]["query_1"] = list(data["gold"]["acceptable_answers"]["query_0"])
    return type(task).model_validate(data)


def test_shared_query_linkage_requires_query_id_for_multi_query(make_task) -> None:
    payload = _payload(count=1)
    tasks = _tasks(make_task, 1)
    tasks[0] = _with_second_query(tasks[0])
    with pytest.raises(ValueError, match="query_id|query"):
        _import(payload, make_task, task_by_legacy_index=tasks)

    payload["results"][0]["query_id"] = "query_1"
    manifest, runs, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.completed_task_count == 1
    assert runs[0].task_id == tasks[0].task_id

    payload["results"][0]["query_id"] = "missing"
    with pytest.raises(ValueError, match="query_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_zero_query_task_is_rejected_by_shared_linkage(make_task) -> None:
    payload = _payload(count=1)
    task = _tasks(make_task, 1)[0]
    zero_query = task.model_copy(update={"queries": []})
    with pytest.raises(ValueError, match="query"):
        _import(payload, make_task, task_by_legacy_index={0: zero_query})


def test_answer_prediction_requires_explicit_authenticated_raw_output(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0].update({
        "answer_parse_valid": True,
        "answer_parser_version": "parser-v1",
    })
    _, runs, _, _ = _import(payload, make_task)
    assert runs[0].answer_predictions == []

    payload["results"][0]["raw_output"] = "raw provider response"
    _, runs, _, _ = _import(payload, make_task)
    prediction = runs[0].answer_predictions[0]
    assert prediction.raw_output == "raw provider response"
    assert prediction.parsed_answer == payload["results"][0]["predicted"]


def test_trace_query_and_config_must_match_linked_identity(make_task) -> None:
    payload = _payload(count=1)
    _claimed_mode(payload, "slot_prompt", "retrieved_prompt")
    payload["summary"].update({
        "answer_topk": 1,
        "retrieval_policy": "top_k",
        "context_order": "ranked",
        "context_annotation": "none",
        "slot_prompt_variant": "minimal",
    })
    tasks = _tasks(make_task, 1)
    _valid_prompt_trace(payload, tasks[0])
    trace = payload["results"][0]["answer_trace"]
    trace.update({
        "query_id": tasks[0].queries[0].query_id,
        "answer_topk": 1,
        "retrieval_policy": "top_k",
        "context_order": "ranked",
        "annotation": "none",
        "slot_prompt_variant": "minimal",
    })
    _import(payload, make_task, task_by_legacy_index=tasks)

    trace["context_order"] = "reverse"
    with pytest.raises(ValueError, match="context_order"):
        _import(payload, make_task, task_by_legacy_index=tasks)
    trace["context_order"] = "ranked"
    trace["query_id"] = "missing"
    with pytest.raises(ValueError, match="query_id"):
        _import(payload, make_task, task_by_legacy_index=tasks)


@pytest.mark.parametrize(
    ("trace_field", "canonical_axis", "value"),
    [
        ("retrieval_policy", "retrieval_policy", "trace-policy"),
        ("context_order", "context_order", "ranked"),
        ("annotation", "context_annotation", "latest-outdated"),
        ("slot_prompt_variant", "slot_prompt_variant", "minimal"),
        ("answer_topk", "answer_topk", 4),
    ],
)
def test_trace_identity_inference_requires_trace_on_every_result_row(
    make_task,
    trace_field: str,
    canonical_axis: str,
    value: Any,
) -> None:
    payload = _payload(count=2)
    payload["results"][0]["answer_trace"] = {trace_field: value}
    with pytest.raises(ValueError, match=f"{canonical_axis}|trace|every"):
        _import(payload, make_task)


def test_trace_config_adoption_requires_complete_consensus(make_task) -> None:
    payload = _payload(count=2)
    _claimed_mode(payload, "slot_prompt", "retrieved_prompt")
    tasks = _tasks(make_task, 2)
    for index in range(2):
        event_id = tasks[index].events[-1].event_id
        payload["results"][index]["answer_trace"] = {
            "retrieved_entries": [{"id": f"entry-{index}", "content": "value", "rank": 1, "score": 0.9, "source_event_id": event_id}],
            "source_event_ids": [event_id],
            "predicted_answer": payload["results"][index]["predicted"],
            "gold_answer": payload["results"][index]["gold_answer"],
            "retrieval_policy": "trace-policy",
        }
    payload["summary"]["save_answer_traces"] = True
    manifest, _, _, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert manifest.prompt_config["legacy_result_import"]["run_identity"]["retrieval_policy"] == "trace-policy"

    del payload["results"][1]["answer_trace"]["retrieval_policy"]
    with pytest.raises(ValueError, match="retrieval_policy|partial"):
        _import(payload, make_task, task_by_legacy_index=tasks)


def test_parser_version_required_for_true_and_false_validity(make_task) -> None:
    payload = _payload(count=2)
    payload["results"][0]["answer_parse_valid"] = False
    payload["results"][1]["answer_parse_valid"] = False
    with pytest.raises(ValueError, match="answer_parser_version"):
        _import(payload, make_task)

    for row in payload["results"]:
        row["answer_parser_version"] = "parser-v1"
    manifest, runs, _, _ = _import(payload, make_task)
    assert manifest.answer_parser_version == "parser-v1"
    assert all(run.parser_extractor_provenance.answer_parser_version == "parser-v1" for run in runs)
    assert all(run.answer_predictions == [] for run in runs)

    payload["results"][1]["answer_parser_version"] = "parser-v2"
    with pytest.raises(ValueError, match="answer_parser_version"):
        _import(payload, make_task)


def test_parser_declarations_require_complete_run_wide_pairs(make_task) -> None:
    partial_pair = _payload(count=2)
    partial_pair["results"][0].update({
        "answer_parse_valid": False,
        "answer_parser_version": "parser-v1",
    })
    with pytest.raises(ValueError, match="answer_parse_valid|answer_parser_version|every row"):
        _import(partial_pair, make_task)

    versions_without_flags = _payload(count=2)
    for row in versions_without_flags["results"]:
        row["answer_parser_version"] = "parser-v1"
    with pytest.raises(ValueError, match="answer_parse_valid|answer_parser_version"):
        _import(versions_without_flags, make_task)

    flag_missing_from_one_row = _payload(count=2)
    for row in flag_missing_from_one_row["results"]:
        row["answer_parser_version"] = "parser-v1"
    flag_missing_from_one_row["results"][0]["answer_parse_valid"] = True
    flag_missing_from_one_row["results"][0]["raw_output"] = "raw-output"
    with pytest.raises(ValueError, match="answer_parse_valid|answer_parser_version|every row"):
        _import(flag_missing_from_one_row, make_task)


def test_mixed_parse_outcomes_share_one_run_parser_provenance(make_task) -> None:
    payload = _payload(count=2)
    payload["results"][0].update({
        "answer_parse_valid": True,
        "answer_parser_version": "parser-v1",
        "raw_output": "raw-output",
    })
    payload["results"][1].update({
        "answer_parse_valid": False,
        "answer_parser_version": "parser-v1",
    })
    manifest, runs, _, _ = _import(payload, make_task)
    assert manifest.answer_parser_version == "parser-v1"
    assert [len(run.answer_predictions) for run in runs] == [1, 0]
    assert all(
        run.parser_extractor_provenance.answer_parser_version == "parser-v1"
        for run in runs
    )


def test_partial_row_identity_is_rejected_and_full_consensus_promoted(make_task) -> None:
    for field, value in [
        ("checkpoint", "checkpoint-a"),
        ("memory_trajectory_id", "trajectory-a"),
        ("context_order", "ranked"),
    ]:
        payload = _payload(count=2)
        payload["results"][0][field] = value
        with pytest.raises(ValueError, match=field):
            _import(payload, make_task)
        payload["results"][1][field] = value
        manifest, _, _, _ = _import(payload, make_task)
        canonical = {
            "checkpoint": "checkpoint_family",
            "memory_trajectory_id": "memory_trajectory_id",
            "context_order": "context_order",
        }[field]
        assert manifest.prompt_config["legacy_result_import"]["run_identity"][canonical] == value
        payload["results"][1][field] = value + "-mixed"
        with pytest.raises(ValueError, match=field):
            _import(payload, make_task)


def test_run_id_is_independent_of_raw_row_order_and_source_path(make_task, tmp_path: Path) -> None:
    payload = _payload(count=2)
    tasks = _tasks(make_task, 2)
    source_a = tmp_path / "a.json"
    hash_a = _write_source(source_a, payload)
    first = import_evomemory_results(payload, source_path=source_a, source_sha256=hash_a, run_name=None, task_by_legacy_index=tasks)

    permuted = copy.deepcopy(payload)
    permuted["results"] = list(reversed(permuted["results"]))
    source_b = tmp_path / "b.json"
    hash_b = _write_source(source_b, permuted)
    second = import_evomemory_results(permuted, source_path=source_b, source_sha256=hash_b, run_name=None, task_by_legacy_index=dict(reversed(tasks.items())))
    assert hash_a != hash_b
    assert first[0].run_id == second[0].run_id
    assert first[0].task_manifest.sha256 == second[0].task_manifest.sha256
    assert [score.model_dump(mode="json") for score in first[2]] == [score.model_dump(mode="json") for score in second[2]]


def _normalized_runtime_semantic_bytes(record) -> bytes:
    data = record.model_dump(mode="json")
    provenance = data["parser_extractor_provenance"]
    provenance["raw_provider_artifact_path"] = None
    provenance["raw_provider_artifact_hash"] = None
    for event in data["system_events"]:
        if event.get("type") != "legacy_evomemory_result":
            continue
        raw_row = dict(event["raw_row"])
        raw_row.pop("example_id", None)
        raw_row.pop("shard_local_example_id", None)
        event["raw_row"] = raw_row
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_run_identity_ignores_shard_packaging_but_retains_row_semantics(
    make_task,
    tmp_path: Path,
) -> None:
    single = _payload(count=4)
    merged = copy.deepcopy(single)
    merged["summary"]["merged_shards"] = 2
    for row, local_index in zip(
        merged["results"], [0, 1, 0, 1], strict=True
    ):
        row["shard_local_example_id"] = local_index
    tasks = _tasks(make_task, 4)

    single_source = tmp_path / "single-shard.json"
    merged_source = tmp_path / "merged-shards.json"
    single_hash = _write_source(single_source, single)
    merged_hash = _write_source(merged_source, merged)
    assert single_hash != merged_hash

    single_import = import_evomemory_results(
        single,
        source_path=single_source,
        source_sha256=single_hash,
        run_name=None,
        task_by_legacy_index=tasks,
    )
    merged_import = import_evomemory_results(
        merged,
        source_path=merged_source,
        source_sha256=merged_hash,
        run_name=None,
        task_by_legacy_index=tasks,
    )
    single_manifest, single_runs, single_scores, _ = single_import
    merged_manifest, merged_runs, merged_scores, _ = merged_import

    assert single_manifest.run_id == merged_manifest.run_id
    assert single_manifest.adapter_info.adapter_id == merged_manifest.adapter_info.adapter_id
    assert (
        single_manifest.task_manifest.sha256
        == merged_manifest.task_manifest.sha256
    )
    assert [
        _normalized_runtime_semantic_bytes(record) for record in single_runs
    ] == [
        _normalized_runtime_semantic_bytes(record) for record in merged_runs
    ]
    assert [canonical_json_bytes(record) for record in single_scores] == [
        canonical_json_bytes(record) for record in merged_scores
    ]

    assert (
        single_manifest.raw_provider_response_artifacts[0].sha256
        != merged_manifest.raw_provider_response_artifacts[0].sha256
    )
    assert canonical_json_bytes(single_runs[0]) != canonical_json_bytes(merged_runs[0])
    assert (
        _legacy_event(single_runs[2])["raw_row"]["shard_local_example_id"]
        != _legacy_event(merged_runs[2])["raw_row"]["shard_local_example_id"]
    )

    for suffix, mutate in (
        (
            "substantive",
            lambda payload: payload["results"][0].__setitem__(
                "predicted", "different-answer"
            ),
        ),
        (
            "config",
            lambda payload: payload["results"][0].__setitem__(
                "answer_trace", {"runtime_config": {"max_tokens": 8}}
            ),
        ),
    ):
        changed = copy.deepcopy(single)
        mutate(changed)
        changed_source = tmp_path / f"changed-{suffix}.json"
        changed_hash = _write_source(changed_source, changed)
        changed_manifest, _, _, _ = import_evomemory_results(
            changed,
            source_path=changed_source,
            source_sha256=changed_hash,
            run_name=None,
            task_by_legacy_index=tasks,
        )
        assert changed_manifest.run_id != single_manifest.run_id


def test_nonidentical_metrics_are_retained_in_legacy_metrics(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0].update({
        "answer_exact_match": 1.0,
        "unknown_metric": 0.25,
    })
    _, _, scores, _ = _import(payload, make_task)
    legacy_metrics = scores[0].legacy_metrics["legacy_p63"]
    assert legacy_metrics["answer_exact_match"] == 1.0
    assert legacy_metrics["unknown_metric"] == 0.25

    payload["results"][0]["answer_exact_match"] = None
    _, _, scores, _ = _import(payload, make_task)
    assert scores[0].legacy_metrics["legacy_p63"]["answer_exact_match"] is None


def test_inapplicable_registered_alias_remains_legacy(make_task) -> None:
    payload = _payload(count=1)
    payload["results"][0]["stale_same_slot_count"] = 2
    tasks = _tasks(make_task, 1)
    data = tasks[0].model_dump(mode="python")
    data["task_family"] = "future_unrelated_family"
    tasks[0] = type(tasks[0]).model_validate(data)
    _, _, scores, _ = _import(payload, make_task, task_by_legacy_index=tasks)
    assert scores[0].store_scores.stale_conflicting_value_count is None
    assert scores[0].supported_metric_fields["store_scores.stale_conflicting_value_count"].reason is SupportReason.NOT_APPLICABLE
    assert scores[0].legacy_metrics["legacy_p63"]["stale_same_slot_count"] == 2


def test_public_exports_retain_task10_and_task11_and_add_task12() -> None:
    assert {"import_evomemory_results", "parse_legacy_run_name"} <= set(legacy.__all__)
    assert get_type_hints(parse_legacy_run_name) == {
        "name": str,
        "return": dict[str, str | int | tuple[str, ...]] | None,
    }
    hints = get_type_hints(import_evomemory_results)
    assert hints["payload"] == dict[str, Any]
    assert hints["source_path"] is Path
    assert hints["source_sha256"] is str
    assert hints["run_name"] == str | None
    assert str(hints["return"]).startswith("tuple[")
