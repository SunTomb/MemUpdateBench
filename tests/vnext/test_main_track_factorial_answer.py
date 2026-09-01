from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import vnext_run_main_track_factorial_answer as runner


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "vnext" / "main_track_v1_factorial_plan_v2" / "factorial_manifest.json"
CANDIDATE = ROOT / "data" / "vnext" / "main_track_v1_audit_fix_v1"
AUDIT = ROOT / "results" / "vnext" / "main_track_v1_audit_completion_attestation_v1" / "review_attestation.json"


def test_module_exposes_immutable_replay_entrypoint() -> None:
    assert callable(runner.run)
    assert callable(runner.reconstruct_retrieval_trace)
    assert runner.EVIDENCE_CLASS == "manager_fixture_answer_replay"


def test_missing_fixture_is_rejected_before_answer_model(tmp_path: Path) -> None:
    calls: list[str] = []

    with pytest.raises((FileNotFoundError, ValueError)):
        runner.run(
            manager_fixture_root=tmp_path / "missing",
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "output",
            cell_id="reference__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: calls.append("model"),
        )
    assert calls == []
def test_public_rows_do_not_contain_raw_material() -> None:
    with pytest.raises(ValueError, match="raw prompt/output/reasoning"):
        runner.validate_public_payload({"raw_output": "secret"})
    with pytest.raises(ValueError, match="raw prompt/output/reasoning"):
        runner.validate_public_payload({"rendered_prompt": "private"})


@pytest.fixture(scope="module")
def full_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.vnext.test_main_track_factorial_manager import _run as build_manager_fixture

    root = tmp_path_factory.mktemp("manager-fixture")
    build_manager_fixture(root)
    fixture = root / "out"
    rows = [json.loads(line) for line in (fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    rows_raw = b"".join(runner.canonical_bytes(row) + b"\n" for row in rows)
    (fixture / "manager_rows.jsonl").write_bytes(rows_raw)
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary.update({"scientific_evidence": True, "rows_sha256": runner.sha256_bytes(rows_raw)})
    summary_bytes = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_bytes)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["scientific_evidence"] = True
    index["manager_kind"] = summary["manager_kind"]
    index["manifest_sha256"] = summary["manifest_sha256"]
    index["artifacts"]["manager_rows.jsonl"] = {
        "sha256": runner.sha256_bytes(rows_raw), "bytes": len(rows_raw), "record_count": len(rows)
    }
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_bytes), "bytes": len(summary_bytes), "record_count": 1
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    return fixture


class GoldReplayModel:
    def __init__(self, candidate_root: Path, *, identity: dict[str, object]) -> None:
        from scripts import vnext_plan_main_track_factorial as planner

        self.identity = identity
        self.last_answer_metadata = {"generated_tokens": 4, "latency_ms": 1.25}
        self.answers = {
            task.queries[0].query_id: task.gold_evidence[0]
            for task in planner.select_test_tasks(candidate_root)
        }
        self.requests = []
        self.closed = False

    def answer(self, request):
        self.requests.append(request)
        gold = self.answers[request.query.query_id]
        if gold.disposition and gold.disposition.value == "abstained":
            return '{"disposition":"abstained"}'
        return json.dumps({"disposition": "answered", "answer": gold.answer}, ensure_ascii=False, separators=(",", ":"))

    def close(self):
        self.closed = True


def _qwen_identity() -> dict[str, object]:
    import scripts.vnext_plan_main_track_factorial as planner

    return dict(planner._answer_spec("qwen")["identity"])


def test_full_replay_preserves_order_and_excludes_unsupported_from_answer_denominator(
    tmp_path: Path, full_fixture: Path
) -> None:
    fixture = _make_test_only_full_fixture(full_fixture, tmp_path / "full-test-only")
    model = GoldReplayModel(CANDIDATE, identity=_qwen_identity())
    summary = runner.run(
        manager_fixture_root=fixture,
        manifest=MANIFEST,
        candidate_root=CANDIDATE,
        audit_attestation=AUDIT,
        output_root=tmp_path / "answer",
        cell_id="letta_profile__qwen35_answer",
        execution_mode="injected_test_only",
        answer_model_factory=lambda: model,
    )
    rows = [json.loads(line) for line in (tmp_path / "answer" / "answer_rows.jsonl").read_bytes().splitlines()]
    assert summary["status"] == "PASS"
    assert summary["evidence_class"] == runner.TEST_ONLY_EVIDENCE_CLASS
    assert summary["scientific_evidence"] is False
    assert len(rows) == 720 and len(model.requests) == 240
    assert [row["task_id"] for row in rows] == [task.task_id for task in __import__("scripts.vnext_plan_main_track_factorial", fromlist=["select_test_tasks"]).select_test_tasks(CANDIDATE)]
    unsupported = [row for row in rows if row["status"] == "UNSUPPORTED"]
    assert len(unsupported) == 480
    assert all(row["parsed_answer"] is None and row["answer_outcome"] is None and row["answer_f1"] is None for row in unsupported)
    assert summary["attempted_answer_denominator"] == 240
    assert summary["answer_em"] == 1.0
    assert all("Use only the retrieved" in request.rendered_prompt for request in model.requests)



def test_reconstruct_binds_source_event_ids_to_current_task_events(full_fixture: Path) -> None:
    rows = [json.loads(line) for line in (full_fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    import scripts.vnext_plan_main_track_factorial as planner

    task = next(item for item in planner.select_test_tasks(CANDIDATE) if item.task_id == row["task_id"])
    trace = dict(row["retrieval"]["trace"])
    entries = [dict(entry) for entry in trace["entries"]]
    entries[0]["source_event_ids"] = ["not-a-current-task-event"]
    trace["entries"] = entries
    row["retrieval"]["trace"] = trace
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    with pytest.raises(ValueError, match="source event"):
        runner.reconstruct_retrieval_trace(
            row,
            task.queries[0],
            task_events=task.events,
        )


def test_reconstruct_rejects_duplicate_entry_ids(full_fixture: Path) -> None:
    rows = [json.loads(line) for line in (full_fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    import scripts.vnext_plan_main_track_factorial as planner

    task = next(item for item in planner.select_test_tasks(CANDIDATE) if item.task_id == row["task_id"])
    trace = dict(row["retrieval"]["trace"])
    duplicate = dict(trace["entries"][0])
    trace["entries"] = [trace["entries"][0], duplicate]
    trace["retrieved_count"] = len(trace["entries"])
    row["retrieval"]["trace"] = trace
    row["retrieval"]["retrieved_count"] = trace["retrieved_count"]
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    with pytest.raises(ValueError, match="duplicate entry_id"):
        runner.reconstruct_retrieval_trace(row, task.queries[0])


def test_fixture_validation_binds_source_event_ids_to_current_task_events(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "source-event-canary")
    rows = [json.loads(line) for line in (fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    trace = dict(row["retrieval"]["trace"])
    entries = [dict(entry) for entry in trace["entries"]]
    entries[0]["source_event_ids"] = ["not-a-current-task-event"]
    trace["entries"] = entries
    row["retrieval"]["trace"] = trace
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    row["retrieval_trace"] = trace
    row["retrieval_trace_sha256"] = row["retrieval"]["trace_sha256"]
    rows_raw = b"".join(runner.canonical_bytes(item) + b"\n" for item in rows)
    (fixture / "manager_rows.jsonl").write_bytes(rows_raw)
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["rows_sha256"] = runner.sha256_bytes(rows_raw)
    summary_raw = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["artifacts"]["manager_rows.jsonl"] = {
        "sha256": runner.sha256_bytes(rows_raw), "bytes": len(rows_raw), "record_count": len(rows)
    }
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    with pytest.raises(ValueError, match="source event"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "answer",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_tamper_and_fail_status_are_rejected_before_model(tmp_path: Path, full_fixture: Path) -> None:
    fixture = tmp_path / "tampered"
    import shutil

    shutil.copytree(full_fixture, fixture)
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["status"] = "FAIL"
    summary_bytes = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_bytes)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["status"] = "FAIL"
    index["artifacts"]["manager_summary.json"]["sha256"] = runner.sha256_bytes(summary_bytes)
    index["artifacts"]["manager_summary.json"]["bytes"] = len(summary_bytes)
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    calls: list[str] = []
    with pytest.raises(ValueError, match="status"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: calls.append("model"),
        )
    assert calls == []


def test_model_identity_mismatch_is_fail_closed(full_fixture: Path, tmp_path: Path) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "identity-canary")
    model = GoldReplayModel(CANDIDATE, identity={**_qwen_identity(), "revision": "0" * 40})
    with pytest.raises(ValueError, match="identity"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only", answer_model_factory=lambda: model,
        )
    assert model.requests == []
    assert model.closed is True




def _make_canary_fixture(source: Path, destination: Path, count: int = 8) -> Path:
    import shutil

    shutil.copytree(source, destination)
    source_rows = [json.loads(line) for line in (destination / "manager_rows.jsonl").read_bytes().splitlines()]
    supported = [row for row in source_rows if row["status"] == "SUPPORTED"][:count]
    selected_ids = {row["task_id"] for row in supported}
    rows = [row for row in source_rows if row["status"] == "UNSUPPORTED" or row["task_id"] in selected_ids]
    rows_raw = b"".join(runner.canonical_bytes(row) + b"\n" for row in rows)
    (destination / "manager_rows.jsonl").write_bytes(rows_raw)
    summary = json.loads((destination / "manager_summary.json").read_bytes())
    summary.update({
        "scope": f"canary{count}", "scientific_evidence": False, "requested_task_count": len(rows),
        "terminal_rows": len(rows), "supported": count, "unsupported": len(rows) - count,
        "executed_supported_count": count, "not_requested_supported_count": 240 - count,
        "state_accuracy_denominator": count, "state_accuracy": 1.0,
        "retrieval_denominator": count, "gold_retrieval_rate": 1.0,
        "selected_supported_task_ids": [row["task_id"] for row in supported],
        "selected_supported_task_ids_sha256": runner.sha256_bytes(runner.canonical_bytes([row["task_id"] for row in supported])),
        "evidence_class": "manager_state_retrieval_fixture_test_only_canary", "rows_sha256": runner.sha256_bytes(rows_raw),
    })
    summary_raw = runner.canonical_bytes(summary)
    (destination / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((destination / "artifact_index.json").read_bytes())
    index.update({"scope": f"canary{count}", "scientific_evidence": False, "evidence_class": "manager_state_retrieval_fixture_test_only_canary", "requested_task_count": len(rows), "executed_supported_count": count, "not_requested_supported_count": 240 - count, "selected_supported_task_ids": summary["selected_supported_task_ids"], "selected_supported_task_ids_sha256": summary["selected_supported_task_ids_sha256"], "manager_kind": summary["manager_kind"], "manifest_sha256": summary["manifest_sha256"]})
    index["artifacts"]["manager_rows.jsonl"] = {"sha256": runner.sha256_bytes(rows_raw), "bytes": len(rows_raw), "record_count": len(rows)}
    index["artifacts"]["manager_summary.json"] = {"sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1}
    (destination / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    return destination


def _make_test_only_full_fixture(source: Path, destination: Path) -> Path:
    import shutil

    shutil.copytree(source, destination)
    summary = json.loads((destination / "manager_summary.json").read_bytes())
    summary.update({"scientific_evidence": False, "evidence_class": "manager_state_retrieval_fixture_test_only"})
    summary_raw = runner.canonical_bytes(summary)
    (destination / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((destination / "artifact_index.json").read_bytes())
    index.update({"scientific_evidence": False, "evidence_class": "manager_state_retrieval_fixture_test_only", "manager_kind": summary["manager_kind"], "manifest_sha256": summary["manifest_sha256"]})
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1
    }
    (destination / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    return destination


def test_canary_fixture_is_non_scientific_and_keeps_unsupported_rows(tmp_path: Path, full_fixture: Path) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "canary")
    model = GoldReplayModel(CANDIDATE, identity=_qwen_identity())
    summary = runner.run(
        manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
        audit_attestation=AUDIT, output_root=tmp_path / "answer",
        cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only", answer_model_factory=lambda: model,
    )
    assert summary["scope"] == "canary8"
    assert summary["evidence_class"] == runner.CANARY_EVIDENCE_CLASS
    assert summary["scientific_evidence"] is False
    assert summary["attempted_answer_denominator"] == 8
    assert len(model.requests) == 8


def test_publication_is_no_replace_and_artifacts_are_hash_bound(full_fixture: Path, tmp_path: Path) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "publication-canary")
    model = GoldReplayModel(CANDIDATE, identity=_qwen_identity())
    output = tmp_path / "out"
    runner.run(
        manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
        audit_attestation=AUDIT, output_root=output,
        cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only", answer_model_factory=lambda: model,
    )
    index = json.loads((output / "artifact_index.json").read_bytes())
    for name in ("answer_rows.jsonl", "answer_summary.json"):
        assert index["artifacts"][name]["sha256"] == runner.sha256_bytes((output / name).read_bytes())
    with pytest.raises(FileExistsError):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=output,
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only", answer_model_factory=lambda: model,
        )


def test_publication_failure_cleans_created_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    fixture_hashes = {name: "b" * 64 for name in runner._ARTIFACTS}
    summary = runner.build_summary(
        [],
        {
            "executed_supported_count": 0,
            "eligible_supported_count": 0,
            "selected_supported_task_ids": [],
        },
        scope="canary1",
        scientific_evidence=False,
        model_identity={"model_id": "test"},
        candidate_hashes={},
        audit_sha="c" * 64,
        fixture_hashes=fixture_hashes,
        execution_mode="injected_test_only",
        manifest_sha256="a" * 64,
        eligible_supported_count=0,
        selected_supported_task_ids=[],
        unsupported_task_ids=[],
    )

    def fail_publish(*args, **kwargs):
        output.mkdir()
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(runner, "publish_files_atomically", fail_publish)
    with pytest.raises(OSError, match="synthetic publication failure"):
        runner._publish(output, [], summary, source_paths=())
    assert not output.exists()


def test_explicit_execution_mode_rejects_test_factory_on_production_path(
    full_fixture: Path, tmp_path: Path
) -> None:
    model = GoldReplayModel(CANDIDATE, identity=_qwen_identity())
    with pytest.raises(ValueError, match="production-bound|injected|execution mode|scientific"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="production",
            answer_model_factory=lambda: model,
        )
    assert model.requests == []


def test_test_only_fixture_promotion_is_rejected_before_factory(
    full_fixture: Path, tmp_path: Path
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="scientific|test-only|authenticated"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: calls.append("model"),
        )
    assert calls == []


def test_output_root_is_validated_before_model_factory(
    full_fixture: Path, tmp_path: Path
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="output_root"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=CANDIDATE / "new-output",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: calls.append("model"),
        )
    assert calls == []


def test_retrieval_trace_binds_target_identity_and_policy(
    full_fixture: Path,
) -> None:
    rows = [json.loads(line) for line in (full_fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    import scripts.vnext_plan_main_track_factorial as planner

    task = next(item for item in planner.select_test_tasks(CANDIDATE) if item.task_id == row["task_id"])
    trace = dict(row["retrieval"]["trace"])
    trace["query_target_object_identities"] = ["wrong-target"]
    trace["retrieval_policy"] = "normal_topk"
    trace["retrieval_k"] = 16
    trace["retrieved_count"] = len(trace["entries"])
    row["retrieval"]["trace"] = trace
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    with pytest.raises(ValueError, match="target|query binding"):
        runner.reconstruct_retrieval_trace(row, task.queries[0])


def test_retrieval_trace_requires_frozen_k_and_order_metadata(
    full_fixture: Path,
) -> None:
    rows = [json.loads(line) for line in (full_fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    import scripts.vnext_plan_main_track_factorial as planner

    task = next(item for item in planner.select_test_tasks(CANDIDATE) if item.task_id == row["task_id"])
    trace = dict(row["retrieval"]["trace"])
    for key in ("retrieval_policy", "retrieval_k", "retrieved_count"):
        trace.pop(key, None)
    row["retrieval"]["trace"] = trace
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    row["retrieval"].pop("retrieved_count", None)
    with pytest.raises(ValueError, match="retrieval policy|retrieval_k|retrieved_count"):
        runner.reconstruct_retrieval_trace(row, task.queries[0])


def test_fixture_manifest_hash_and_membership_are_bound(
    full_fixture: Path, tmp_path: Path
) -> None:
    import shutil

    fixture = _make_canary_fixture(full_fixture, tmp_path / "fixture")
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["manifest_sha256"] = "0" * 64
    summary_raw = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    with pytest.raises(ValueError, match="manifest|membership"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_secret_in_parsed_answer_is_scanned_before_publication(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "secret-canary")
    class SecretModel(GoldReplayModel):
        def answer(self, request):
            self.requests.append(request)
            return '{"disposition":"answered","answer":"api_key=sk-abcdefghijklmnopqrstuvwxyz"}'

    model = SecretModel(CANDIDATE, identity=_qwen_identity())
    with pytest.raises(ValueError, match="security|secret"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: model,
        )
    assert not (tmp_path / "out").exists()


def test_load_failure_still_closes_allocated_model(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "load-canary")
    class AllocatingLoadFailure(GoldReplayModel):
        def __init__(self):
            super().__init__(CANDIDATE, identity=_qwen_identity())
            self.allocated = False
            self.closed = False

        def load(self):
            self.allocated = True
            raise RuntimeError("load failed after allocation")

        def close(self):
            self.closed = True
            self.allocated = False

    model = AllocatingLoadFailure()
    with pytest.raises(RuntimeError, match="load failed"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: model,
        )
    assert model.closed is True and model.allocated is False


def test_factorial_score_uses_canonical_v3_structured_value_semantics() -> None:
    from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3
    from types import SimpleNamespace

    query = SimpleNamespace(query_id="q", answer_schema=AnswerSchema.LIST)
    gold = SimpleNamespace(disposition=AnswerDisposition.ANSWERED, answer=["alpha", "beta"])
    prediction = AnswerPredictionV3(
        query_id="q", raw_output='{"disposition":"answered","answer":["alpha","gamma"]}',
        disposition=AnswerDisposition.ANSWERED, parsed_answer=["alpha", "gamma"], format_valid=True,
    )
    scored = runner.score_prediction(query, prediction, gold)
    assert scored["typed_match"] is False
    assert scored["normalized_match"] is False
    assert scored["answer_f1"] > 0
    assert scored["scorer_version"] == "3.0.0"


def test_public_relocation_metadata_contains_no_host_paths() -> None:
    relocation = runner.build_relocation_metadata(
        manifest_sha256="a" * 64,
        candidate_release_index_sha256="b" * 64,
        audit_attestation_sha256="c" * 64,
        allow_relocated_authenticated_inputs=True,
    )
    rendered = json.dumps(relocation)
    assert "D:\\\\" not in rendered and "/NAS/" not in rendered and "\\\\Users\\\\" not in rendered



def test_execution_mode_is_explicit_and_injected_factory_cannot_be_production(tmp_path: Path, full_fixture: Path) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="execution_mode"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            answer_model_factory=lambda: calls.append("factory"),
        )
    assert calls == []

    with pytest.raises(ValueError, match="injected.*production|execution mode|scientific"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "production",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="production",
            answer_model_factory=lambda: calls.append("factory"),
        )
    assert calls == []


def test_fixture_manifest_and_selected_membership_are_bound(tmp_path: Path, full_fixture: Path) -> None:
    import shutil

    fixture = _make_canary_fixture(full_fixture, tmp_path / "fixture")
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["manifest_sha256"] = "0" * 64
    summary_raw = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw),
        "bytes": len(summary_raw),
        "record_count": 1,
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    with pytest.raises(ValueError, match="manifest SHA|manifest binding"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: None,
        )


def test_retrieval_trace_requires_full_query_and_policy_binding(full_fixture: Path) -> None:
    rows = [json.loads(line) for line in (full_fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    row = next(item for item in rows if item["status"] == "SUPPORTED")
    import scripts.vnext_plan_main_track_factorial as planner

    task = next(item for item in planner.select_test_tasks(CANDIDATE) if item.task_id == row["task_id"])
    trace = dict(row["retrieval"]["trace"])
    trace["query_target_object_identities"] = [["post_core", "wrong", "role", None]]
    row["retrieval"]["trace"] = trace
    row["retrieval"]["trace_sha256"] = runner.sha256_bytes(runner.canonical_bytes(trace))
    with pytest.raises(ValueError, match="target|retrieval policy|retrieval_k"):
        runner.reconstruct_retrieval_trace(row, task.queries[0])


def test_load_failure_still_closes_owned_model(tmp_path: Path, full_fixture: Path) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "owned-load-canary")
    class LoadFailureModel:
        identity = _qwen_identity()
        last_answer_metadata = {}

        def __init__(self):
            self.allocated = False
            self.closed = False

        def load(self):
            self.allocated = True
            raise RuntimeError("synthetic load failure")

        def close(self):
            self.closed = True
            self.allocated = False

    model = LoadFailureModel()
    with pytest.raises(RuntimeError, match="load"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: model,
        )
    assert model.closed is True
    assert model.allocated is False


def test_public_rows_redact_relocation_paths_and_bind_scoring_contract() -> None:
    row = {"parsed_answer": "ok", "input_relocation": {"manifest": "D:\\\\private\\\\manifest.json"}}
    with pytest.raises(ValueError, match="absolute path"):
        runner.validate_public_payload(row)
    assert hasattr(runner, "SCORING_BINDING")
    assert runner.SCORING_BINDING["scorer_version"]
    assert runner.SCORING_BINDING["metric_registry_version"]


def test_attempted_answer_failure_publishes_fail_summary_and_non_scientific_artifact(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "failure-canary")

    class FailingModel(GoldReplayModel):
        def answer(self, request):
            raise RuntimeError("synthetic answer failure")

    model = FailingModel(CANDIDATE, identity=_qwen_identity())
    summary = runner.run(
        manager_fixture_root=fixture,
        manifest=MANIFEST,
        candidate_root=CANDIDATE,
        audit_attestation=AUDIT,
        output_root=tmp_path / "out",
        cell_id="letta_profile__qwen35_answer",
        execution_mode="injected_test_only",
        answer_model_factory=lambda: model,
    )
    assert summary["status"] == "FAIL"
    assert summary["failed"] == 8
    assert summary["scientific_evidence"] is False
    index = json.loads((tmp_path / "out" / "artifact_index.json").read_bytes())
    assert index["status"] == "FAIL"


def test_fixture_trace_validation_happens_before_answer_factory(
    full_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "trace-before-factory")
    calls: list[str] = []

    def reject_trace(row, query):
        raise ValueError("synthetic trace validation failure")

    monkeypatch.setattr(runner, "reconstruct_retrieval_trace", reject_trace)
    with pytest.raises(ValueError, match="trace validation"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: calls.append("factory") or GoldReplayModel(
                CANDIDATE, identity=_qwen_identity()
            ),
        )
    assert calls == []


def test_safe_answer_metadata_omits_arbitrary_finish_reason() -> None:
    class MetadataModel:
        last_answer_metadata = {"finish_reason": "arbitrary-provider-text"}

    safe = runner._safe_model_metadata(MetadataModel(), "output")
    assert "finish_reason" not in safe


def test_safe_answer_metadata_keeps_small_finish_reason_enum() -> None:
    class MetadataModel:
        last_answer_metadata = {"finish_reason": "stop"}

    safe = runner._safe_model_metadata(MetadataModel(), "output")
    assert safe["finish_reason"] == "stop"


def test_run_returns_rows_hash_bound_to_serialized_rows(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "rows-hash-canary")
    output = tmp_path / "out"
    summary = runner.run(
        manager_fixture_root=fixture,
        manifest=MANIFEST,
        candidate_root=CANDIDATE,
        audit_attestation=AUDIT,
        output_root=output,
        cell_id="letta_profile__qwen35_answer",
        execution_mode="injected_test_only",
        answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
    )
    assert summary["rows_sha256"] == runner.sha256_bytes((output / "answer_rows.jsonl").read_bytes())
    published = json.loads((output / "answer_summary.json").read_bytes())
    assert published["rows_sha256"] == summary["rows_sha256"]


def test_answer_summary_and_index_repeat_membership_and_evidence_bindings(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "membership-canary")
    output = tmp_path / "out"
    summary = runner.run(
        manager_fixture_root=fixture,
        manifest=MANIFEST,
        candidate_root=CANDIDATE,
        audit_attestation=AUDIT,
        output_root=output,
        cell_id="letta_profile__qwen35_answer",
        execution_mode="injected_test_only",
        answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
    )
    published = json.loads((output / "answer_summary.json").read_bytes())
    index = json.loads((output / "artifact_index.json").read_bytes())
    for artifact in (published, index):
        assert artifact["manifest_sha256"] == summary["manifest_sha256"]
        assert artifact["eligible_supported_count"] == 240
        assert artifact["executed_supported_count"] == 8
        assert artifact["selected_supported_task_ids"] == summary["selected_supported_task_ids"]
        assert artifact["selected_supported_task_ids_sha256"] == summary["selected_supported_task_ids_sha256"]
        assert artifact["unsupported_count"] == 480
        assert artifact["unsupported_task_ids_sha256"] == summary["unsupported_task_ids_sha256"]
        assert artifact["execution_mode"] == summary["execution_mode"]
        assert artifact["evidence_class"] == summary["evidence_class"]
        assert artifact["scientific_evidence"] == summary["scientific_evidence"]


def test_load_fixture_rejects_reparse_root_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    calls: list[Path] = []

    def reject_reparse(path):
        calls.append(Path(path))
        raise ValueError("reparse fixture root")

    monkeypatch.setattr(runner, "assert_no_reparse_components", reject_reparse)
    with pytest.raises(ValueError, match="reparse fixture root"):
        runner._load_fixture(root)
    assert calls == [root]




def test_production_canary_uses_canary_evidence_class() -> None:
    fixture_summary = {
        "executed_supported_count": 1,
        "eligible_supported_count": 240,
        "selected_supported_task_ids": ["task_x"],
    }
    row = {"task_id": "task_x", "status": "PASS", "answer_outcome": "CORRECT", "exact_match": True, "normalized_match": True, "typed_match": True, "typed_exact_match": True, "answer_f1": 1.0}
    summary = runner.build_summary(
        [row],
        fixture_summary,
        scope="canary1",
        scientific_evidence=False,
        model_identity={"model_id": "test"},
        candidate_hashes={},
        audit_sha="a" * 64,
        fixture_hashes={name: "b" * 64 for name in runner._ARTIFACTS},
        execution_mode="production",
        manifest_sha256="c" * 64,
        eligible_supported_count=240,
        selected_supported_task_ids=["task_x"],
        unsupported_task_ids=[],
    )
    assert summary["evidence_class"] == runner.CANARY_EVIDENCE_CLASS
    assert summary["scientific_evidence"] is False


def test_cli_parser_exposes_manager_fixture_attestation_digest() -> None:
    parser = runner.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--manager-fixture-root", "fixture",
            "--candidate-root", "candidate",
            "--audit-attestation", "audit.json",
            "--output-root", "out",
            "--cell-id", "reference__qwen35_answer",
            "--execution-mode", "production",
            "--manager-fixture-attestation", "receipt.json",
            "--manager-fixture-attestation-sha256", "a" * 64,
        ]
    )
    assert parsed.manager_fixture_attestation_sha256 == "a" * 64


def test_answer_prediction_rejects_a_mismatched_query_id() -> None:
    from types import SimpleNamespace
    from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

    request = SimpleNamespace(
        query=SimpleNamespace(query_id="query-expected", answer_schema=AnswerSchema.STRING),
    )

    class WrongQueryModel:
        def answer(self, _request):
            return AnswerPredictionV3(
                query_id="query-other",
                raw_output='{"disposition":"answered","answer":"ok"}',
                disposition=AnswerDisposition.ANSWERED,
                parsed_answer="ok",
                format_valid=True,
            )

    with pytest.raises(ValueError, match="query_id"):
        runner._answer_prediction(WrongQueryModel(), request)


def test_answer_prediction_rejects_raw_object_field_mismatch() -> None:
    from types import SimpleNamespace
    from mub.vnext.contracts.enums import AnswerDisposition, AnswerSchema
    from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

    request = SimpleNamespace(
        query=SimpleNamespace(query_id="query-expected", answer_schema=AnswerSchema.STRING),
    )

    class InconsistentModel:
        def answer(self, _request):
            return AnswerPredictionV3(
                query_id="query-expected",
                raw_output='{"disposition":"answered","answer":"raw-value"}',
                disposition=AnswerDisposition.ANSWERED,
                parsed_answer="object-value",
                format_valid=True,
            )

    with pytest.raises(ValueError, match="raw|consisten|prediction"):
        runner._answer_prediction(InconsistentModel(), request)


def test_requested_manifest_cell_matches_deterministic_planner(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_test_only_full_fixture(full_fixture, tmp_path / "cell-fixture")
    manifest = json.loads(MANIFEST.read_bytes())
    cell = next(item for item in manifest["cells"] if item["cell_id"] == "letta_profile__qwen35_answer")
    cell["extractor"]["identity"]["extractor_version"] = "forged-extractor"
    payload = dict(manifest)
    payload.pop("payload_sha256")
    manifest["payload_sha256"] = runner.sha256_bytes(runner.canonical_bytes(payload))
    manifest_path = tmp_path / "forged-manifest.json"
    manifest_path.write_bytes(runner.canonical_bytes(manifest))
    manifest_sha = runner.sha256_bytes(manifest_path.read_bytes())
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["manifest_sha256"] = manifest_sha
    summary_raw = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["manifest_sha256"] = manifest_sha
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    with pytest.raises(ValueError, match="cell|extractor|planner"):
        runner.run(
            manager_fixture_root=fixture,
            manifest=manifest_path,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_output_root_overlap_sources_include_production_qwen_inputs(
    full_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def capture(_output, source_paths):
        seen.extend(source_paths)
        raise ValueError("stop before validation")

    monkeypatch.setattr(runner, "_validate_output_root", capture)
    receipt = tmp_path / "fixture-receipt.json"
    snapshot = tmp_path / "qwen-snapshot"
    runtime = tmp_path / "qwen-runtime.json"
    binding = tmp_path / "qwen-binding.json"
    with pytest.raises(ValueError, match="stop"):
        runner.run(
            manager_fixture_root=full_fixture,
            manifest=MANIFEST,
            candidate_root=CANDIDATE,
            audit_attestation=AUDIT,
            output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer",
            execution_mode="production",
            model_snapshot=snapshot,
            model_runtime_receipt=runtime,
            model_snapshot_binding=binding,
            manager_fixture_attestation=receipt,
            manager_fixture_attestation_sha256="a" * 64,
        )
    assert {receipt, snapshot, runtime, binding}.issubset(set(seen))
    assert Path(runner.__file__).with_name("vnext_run_letta_qwen_prompted_answer.py") in seen


def test_fixture_unsupported_reason_code_is_bound_to_manifest(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-unsupported-reason")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "UNSUPPORTED")
        row["reason_code"] = "forged_reason"

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="reason|unsupported"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


    with pytest.raises(ValueError, match="absolute path"):
        runner.validate_public_payload({"path": "/etc/passwd"})
    runner.validate_public_payload({"artifact_sha256": "a" * 64})


def _rewrite_fixture_rows(fixture: Path, mutate) -> None:
    rows = [json.loads(line) for line in (fixture / "manager_rows.jsonl").read_bytes().splitlines()]
    mutate(rows)
    rows_raw = b"".join(runner.canonical_bytes(row) + b"\n" for row in rows)
    (fixture / "manager_rows.jsonl").write_bytes(rows_raw)
    summary = json.loads((fixture / "manager_summary.json").read_bytes())
    summary["rows_sha256"] = runner.sha256_bytes(rows_raw)
    summary_raw = runner.canonical_bytes(summary)
    (fixture / "manager_summary.json").write_bytes(summary_raw)
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["artifacts"]["manager_rows.jsonl"] = {
        "sha256": runner.sha256_bytes(rows_raw), "bytes": len(rows_raw), "record_count": len(rows)
    }
    index["artifacts"]["manager_summary.json"] = {
        "sha256": runner.sha256_bytes(summary_raw), "bytes": len(summary_raw), "record_count": 1
    }
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))


@pytest.mark.parametrize("field", ["state_accuracy", "gold_retrieved"])
def test_fixture_rejects_non_boolean_accuracy_flags(
    full_fixture: Path, tmp_path: Path, field: str
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / f"bad-{field}")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "SUPPORTED")
        nested = "state" if field == "state_accuracy" else "retrieval"
        row[field] = "true"
        row[nested][field] = "true"

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="boolean|bool"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_rejects_recomputed_accuracy_flag_mismatch(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-state-accuracy")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "SUPPORTED")
        row["state_accuracy"] = False
        row["state"]["state_accuracy"] = False

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="state accuracy|state fields|recomputed"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_rejects_recomputed_gold_retrieved_flag_mismatch(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-gold-retrieved")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "SUPPORTED")
        row["gold_retrieved"] = False
        row["retrieval"]["gold_retrieved"] = False

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="gold retrieved|retrieval"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


@pytest.mark.parametrize("field", ["final_memory_size", "stable_entry_id"])
def test_fixture_rejects_invalid_state_field_types(
    full_fixture: Path, tmp_path: Path, field: str
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / f"bad-{field}")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "SUPPORTED")
        row[field] = "1"
        row["state"][field] = "1"

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="final_memory_size|stable_entry_id|state"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_unsupported_reason_detail_is_bound_to_manifest(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-unsupported-detail")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "UNSUPPORTED")
        row["detail"] = "forged detail"

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="reason|detail|unsupported"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_index_cell_binding_must_match_summary(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-index-cell")
    index = json.loads((fixture / "artifact_index.json").read_bytes())
    index["cell_id"] = "wrong-cell"
    (fixture / "artifact_index.json").write_bytes(runner.canonical_bytes(index))
    with pytest.raises(ValueError, match="summary/index|cell"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )


def test_fixture_unsupported_answer_metrics_are_required_null(
    full_fixture: Path, tmp_path: Path
) -> None:
    fixture = _make_canary_fixture(full_fixture, tmp_path / "bad-unsupported-metric")

    def tamper(rows):
        row = next(item for item in rows if item["status"] == "UNSUPPORTED")
        row["exact_match"] = False

    _rewrite_fixture_rows(fixture, tamper)
    with pytest.raises(ValueError, match="unsupported|metric|null"):
        runner.run(
            manager_fixture_root=fixture, manifest=MANIFEST, candidate_root=CANDIDATE,
            audit_attestation=AUDIT, output_root=tmp_path / "out",
            cell_id="letta_profile__qwen35_answer", execution_mode="injected_test_only",
            answer_model_factory=lambda: GoldReplayModel(CANDIDATE, identity=_qwen_identity()),
        )
