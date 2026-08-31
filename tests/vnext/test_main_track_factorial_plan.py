from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.vnext_plan_main_track_factorial as factorial_plan
from mub.vnext.contracts.enums import AnswerSchema
from mub.vnext.contracts.v3.enums import QueryTypeV3
from mub.vnext.contracts.v3.task import (
    MemoryQueryV3,
    MultiObjectCurrentSelector,
    PreviousSelector,
)
from scripts.vnext_plan_main_track_factorial import (
    AUDIT_ATTESTATION,
    CANDIDATE_ROOT,
    CELL_IDS,
    build_factorial_manifest,
    publish_factorial_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def test_factorial_plan_has_six_cells_and_contract_support_counts() -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)

    assert manifest["schema_version"] == "memupdatebench.main-track.factorial-plan.v2"
    assert manifest["planning_status"] == "PLANNING_ONLY_NO_EXECUTION"
    assert manifest["task_view"]["count"] == 720
    assert len(manifest["task_view"]["task_ids"]) == 720
    assert tuple(cell["cell_id"] for cell in manifest["cells"]) == CELL_IDS

    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    assert by_id["reference__qwen35_answer"]["supported_count"] == 720
    assert by_id["reference__muse_answer"]["supported_count"] == 720
    assert by_id["letta_profile__qwen35_answer"]["supported_count"] == 240
    assert by_id["letta_profile__muse_answer"]["supported_count"] == 240
    assert by_id["langgraph_store_custom_adapter__qwen35_answer"]["supported_count"] == 240
    assert by_id["langgraph_store_custom_adapter__muse_answer"]["supported_count"] == 240

    for cell in manifest["cells"]:
        assert cell["supported_count"] + cell["unsupported_count"] == 720
        assert len(cell["supported_task_ids"]) == cell["supported_count"]
        assert len(cell["unsupported_tasks"]) == cell["unsupported_count"]
        assert all(item["reason_code"] for item in cell["unsupported_tasks"])
        assert cell["extractor"]["role"] == (
            "none" if cell["manager_id"] == "reference" else "visible_event_crud_extraction"
        )


def test_langgraph_cell_uses_exact_custom_adapter_identity_and_boundary() -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    cell = next(
        cell for cell in manifest["cells"]
        if cell["cell_id"] == "langgraph_store_custom_adapter__qwen35_answer"
    )

    assert cell["manager"] == {
        "manager_id": "langgraph_store_custom_adapter",
        "adapter_id": "langgraph_store_custom_adapter",
        "adapter_version": "memupdatebench-langmem-adapter-v1",
        "system_name": "langgraph_in_memory_store",
        "system_version": "0.0.30",
        "backend": "langgraph_in_memory_store",
        "implementation_boundary": factorial_plan.LANGGRAPH_IMPLEMENTATION_BOUNDARY,
    }
    assert cell["manager_id"] == "langgraph_store_custom_adapter"
    assert cell["manager"]["implementation_boundary"] == (
        "LangGraph InMemoryStore + custom MemUpdateBench adapter; "
        "not native LangMem CRUD/retrieval evidence"
    )
    assert "langmem_0_0_30_profile" not in json.dumps(cell, sort_keys=True)


def test_manifest_shape_rejects_langgraph_boundary_drift() -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    manifest["cells"][4]["manager"]["implementation_boundary"] = "drifted"
    payload = dict(manifest)
    payload.pop("payload_sha256")
    manifest["payload_sha256"] = factorial_plan.sha256_bytes(
        factorial_plan.canonical_json_bytes(payload)
    )

    with pytest.raises(ValueError, match="manager specification"):
        factorial_plan._validate_manifest_shape(manifest)


def test_profile_cells_use_typed_reasons_not_unsupported_zero() -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    profile = next(cell for cell in manifest["cells"] if cell["manager_id"] == "letta_0_16_8_block_profile")
    reasons = {item["reason_code"] for item in profile["unsupported_tasks"]}
    assert reasons
    assert "unsupported" not in reasons
    assert "single_target_object_required" in reasons or "current_single_object_retrieved_prompt_required" in reasons


def test_capability_reason_validates_query_count_before_target_shape() -> None:
    task = next(
        task for task in factorial_plan.select_test_tasks(CANDIDATE_ROOT)
        if len(task.target_objects) > 1
    )
    query = task.queries[0]
    malformed = SimpleNamespace(
        target_objects=task.target_objects,
        queries=(query, query),
        actions=task.actions,
    )

    reason = factorial_plan._capability_reason(
        malformed,
        factorial_plan._PROFILE_CAPABILITIES,
    )

    assert reason is not None
    assert reason["reason_code"] == "single_query_required"


def test_historical_query_uses_declared_historical_capability() -> None:
    task = next(
        task for task in factorial_plan.select_test_tasks(CANDIDATE_ROOT)
        if any(len(version.entries) > 1 for version in task.version_history)
    )
    query = task.queries[0]
    historical_query = MemoryQueryV3.model_validate({
        **query.model_dump(mode="python"),
        "query_type": QueryTypeV3.PREVIOUS,
        "selector": PreviousSelector(),
    })
    historical_task = SimpleNamespace(
        target_objects=(task.target_objects[0],),
        queries=(historical_query,),
        actions=task.actions,
    )

    assert factorial_plan._capability_reason(
        historical_task,
        factorial_plan._REFERENCE_CAPABILITIES,
    ) is None


def test_multi_object_query_uses_declared_multi_object_capability() -> None:
    task = next(
        task for task in factorial_plan.select_test_tasks(CANDIDATE_ROOT)
        if len(task.target_objects) > 1
    )
    targets = task.target_objects[:2]
    query = task.queries[0]
    multi_query = MemoryQueryV3.model_validate({
        **query.model_dump(mode="python"),
        "query_type": QueryTypeV3.MULTI_OBJECT_CURRENT,
        "selector": MultiObjectCurrentSelector(object_keys=targets),
        "target_object_keys": targets,
        "answer_schema": AnswerSchema.LIST,
    })
    multi_task = SimpleNamespace(
        target_objects=targets,
        queries=(multi_query,),
        actions=task.actions,
    )

    assert factorial_plan._capability_reason(
        multi_task,
        factorial_plan._REFERENCE_CAPABILITIES,
    ) is None


def test_candidate_release_index_rejects_duplicate_artifact_paths(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    release_index = {
        "artifacts": [
            {"path": "tasks.jsonl", "sha256": "a" * 64},
            {"path": "tasks.jsonl", "sha256": "b" * 64},
        ]
    }

    with pytest.raises(ValueError, match="duplicate artifact path"):
        factorial_plan._candidate_hashes(candidate, release_index)


def test_manifest_rejects_nested_manager_id_mismatch() -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    manifest["cells"][0]["manager"]["manager_id"] = "wrong-manager"
    payload = dict(manifest)
    payload.pop("payload_sha256")
    manifest["payload_sha256"] = factorial_plan.sha256_bytes(
        factorial_plan.canonical_json_bytes(payload)
    )

    with pytest.raises(ValueError, match="manager_id"):
        factorial_plan._validate_manifest_shape(manifest)


@pytest.mark.parametrize(
    ("spec_factory", "unknown_kind"),
    [
        (factorial_plan._extractor_spec, "future_manager"),
        (factorial_plan._answer_spec, "future_model"),
        (factorial_plan._cell_spec, "future_manager__future_answer"),
    ],
)
def test_factorial_specs_reject_unknown_kinds(spec_factory, unknown_kind: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        spec_factory(unknown_kind)


def test_attestation_tamper_is_rejected(tmp_path: Path) -> None:
    source = AUDIT_ATTESTATION.read_bytes()
    tampered = tmp_path / "review_attestation.json"
    tampered.write_bytes(source.replace(b'"review_status":"PASS"', b'"review_status":"BLOCK"'))
    with pytest.raises(ValueError, match="attestation"):
        build_factorial_manifest(CANDIDATE_ROOT, tampered)


def test_candidate_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    target = (CANDIDATE_ROOT / "catalog_manifest.json").resolve()
    original_read_bytes = Path.read_bytes

    def tampered_read_bytes(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if path.resolve() == target:
            return raw + b" "
        return raw

    monkeypatch.setattr(Path, "read_bytes", tampered_read_bytes)
    with pytest.raises(ValueError, match="candidate artifact hash mismatch"):
        build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)


def test_build_rejects_candidate_mutation_after_task_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    target = (CANDIDATE_ROOT / "tasks.jsonl").resolve()
    original_read_bytes = Path.read_bytes
    original_select = factorial_plan.select_test_tasks
    state = {"mutated": False}

    def conditional_read_bytes(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if state["mutated"] and path.resolve() == target:
            return raw + b"\n"
        return raw

    def select_then_mutate(candidate_root: Path | str):
        tasks = original_select(candidate_root)
        state["mutated"] = True
        return tasks

    monkeypatch.setattr(Path, "read_bytes", conditional_read_bytes)
    monkeypatch.setattr(factorial_plan, "select_test_tasks", select_then_mutate)
    with pytest.raises(ValueError, match="candidate artifact hash mismatch"):
        build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)


def test_support_partition_is_cached_per_manager_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}
    original_reason = factorial_plan.task_support_reason

    def counting_reason(task, manager_kind):
        calls[manager_kind] = calls.get(manager_kind, 0) + 1
        return original_reason(task, manager_kind)

    monkeypatch.setattr(factorial_plan, "task_support_reason", counting_reason)
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)

    assert calls == {"letta": 720, "langgraph": 720}
    assert manifest["task_view"]["count"] == 720


def test_publication_interruption_cleans_temp_and_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    output = tmp_path / "factorial_manifest.json"

    def interrupted_publish(*args, **kwargs) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(factorial_plan, "publish_files_atomically", interrupted_publish)
    with pytest.raises(OSError, match="simulated interruption"):
        publish_factorial_manifest(manifest, output)

    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.*")) == []


def test_publication_race_never_clobbers_competing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    output = tmp_path / "factorial_manifest.json"
    original_publish = factorial_plan.publish_files_atomically

    def competing_publish(payloads, **kwargs):
        output.write_bytes(b"competing-writer")
        return original_publish(payloads, **kwargs)

    monkeypatch.setattr(factorial_plan, "publish_files_atomically", competing_publish)
    with pytest.raises(FileExistsError):
        publish_factorial_manifest(manifest, output)
    assert output.read_bytes() == b"competing-writer"


def test_cli_rejects_existing_output_before_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "factorial_manifest.json"
    output.write_text("occupied", encoding="utf-8")
    called = False

    def unexpected_build(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("build should not run for an existing output")

    monkeypatch.setattr(factorial_plan, "build_factorial_manifest", unexpected_build)
    with pytest.raises(FileExistsError):
        factorial_plan.main(["--output", str(output)])
    assert not called


def test_publication_is_no_replace(tmp_path: Path) -> None:
    manifest = build_factorial_manifest(CANDIDATE_ROOT, AUDIT_ATTESTATION)
    output = tmp_path / "factorial_manifest.json"
    digest = publish_factorial_manifest(manifest, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        publish_factorial_manifest(manifest, output)
    assert json.loads(output.read_text(encoding="utf-8"))["payload_sha256"]
