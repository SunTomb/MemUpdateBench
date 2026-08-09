import pytest

from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.v3.manifest import RunManifestV3, TaskManifestV3
from mub.vnext.contracts.v3.score import ScorerConfigV3
from mub.vnext.generation.core_config import load_core_config
from mub.vnext.io import sha256_model


H = "a" * 64


@pytest.fixture(scope="module")
def core_control_tasks():
    from pathlib import Path

    from mub.vnext.generation.family_e import compile_family_e_micro_pilot
    from mub.vnext.generation.family_f import compile_family_f_micro_pilot
    from mub.vnext.generation.family_g import compile_family_g_micro_pilot

    root = Path(__file__).resolve().parents[2]
    config = load_core_config(root / "configs" / "vnext" / "core.yaml")
    tasks = (
        *compile_family_e_micro_pilot(
            config, code_revision="core-corrupted-controls-red"
        ).tasks,
        *compile_family_f_micro_pilot(
            config, code_revision="core-corrupted-controls-red"
        ).tasks,
        *compile_family_g_micro_pilot(
            config, code_revision="core-corrupted-controls-red"
        ).tasks,
    )
    return tuple(
        task for task in tasks if task.metadata.extra["surface_variant"] == 0
    )


def _task(tasks, family, condition):
    return next(
        task
        for task in tasks
        if task.task_family == family and condition(task)
    )


def _authenticated_context(task, run, info, capabilities):
    from mub.vnext.scoring.scorer_v3 import VerifiedScoringContextV3

    scorer_config = ScorerConfigV3()
    task_artifact = {
        "path": "tasks.jsonl",
        "sha256": "b" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    run_artifact = {
        "path": "runs.jsonl",
        "sha256": "c" * 64,
        "media_type": "application/jsonl",
        "record_count": 1,
    }
    task_manifest = TaskManifestV3(
        data_release_id="core-corrupted-controls-smoke",
        split_policy_version=task.metadata.split_key.split_policy_version,
        compiler_versions={"core": task.metadata.compiler_version},
        source_manifest_paths_and_hashes=(),
        generation_configs_and_hashes=(),
        split_counts={task.metadata.split.value: 1},
        family_difficulty_counts={
            f"{task.task_family}.{task.difficulty.value}": 1
        },
        semantic_core_counts={task.metadata.split_key.semantic_core_id: 1},
        task_file_paths_and_hashes=(task_artifact,),
        task_record_hashes={task.task_id: sha256_model(task)},
        leakage_check_summary={},
        human_audit_artifacts=(),
        created_at="2026-08-09T00:00:00Z",
        code_revision="core-corrupted-controls-red",
    )
    task_manifest_hash = sha256_model(task_manifest)
    run_manifest = RunManifestV3(
        run_id=run.run_id,
        timestamp="2026-08-09T00:00:00Z",
        code_revision="core-corrupted-controls-red",
        dirty_state=False,
        task_manifest={
            "path": "task-manifest.json",
            "sha256": task_manifest_hash,
            "media_type": "application/json",
        },
        scorer_config=scorer_config,
        adapter_info=info,
        adapter_capabilities=capabilities,
        capability_verification_artifact={
            "path": "capabilities.json",
            "sha256": "e" * 64,
            "media_type": "application/json",
        },
        model_name=None,
        provider=None,
        model_revision=None,
        prompt_config={},
        decoding_config={},
        seed_information={},
        action_parser_version=(
            run.parser_extractor_provenance.action_parser_version
        ),
        answer_parser_version=(
            run.parser_extractor_provenance.answer_parser_version
        ),
        memory_entry_extractor_version=(
            run.parser_extractor_provenance.memory_entry_extractor_version
        ),
        object_value_extractor_config_hash=(
            run.parser_extractor_provenance.object_value_extractor_config_hash
        ),
        redaction_policy_version=(
            run.parser_extractor_provenance.redaction_policy_version
        ),
        environment_summary={"smoke_only": True},
        package_summary={"leaderboard_eligible": False},
        expected_task_count=1,
        completed_task_count=int(
            run.completion_status is CompletionStatus.COMPLETED
        ),
        failed_task_count=int(
            run.completion_status in {
                CompletionStatus.FAILED,
                CompletionStatus.PARTIAL,
            }
        ),
        not_supported_task_count=int(
            run.completion_status is CompletionStatus.NOT_SUPPORTED
        ),
        raw_provider_response_artifacts=(),
        raw_adapter_state_artifacts=(),
        normalized_runtime_artifacts=(run_artifact,),
        run_record_hashes={run.task_id: sha256_model(run)},
        score_artifacts=(),
        native_vs_extracted_field_summary={},
    )
    return VerifiedScoringContextV3.from_authenticated_manifests(
        task=task,
        run=run,
        task_manifest=task_manifest,
        run_manifest=run_manifest,
        task_artifact=task_artifact,
        run_artifact=run_artifact,
        authenticated_task_manifest_sha256=task_manifest_hash,
        authenticated_run_manifest_sha256=sha256_model(run_manifest),
    )


def _control_cases(tasks):
    family_e_attribute = _task(
        tasks,
        "deletion_forgetting",
        lambda task: task.metadata.extra["stratification"]["deletion_scope"]
        == "attribute",
    )
    family_e_ttl = _task(
        tasks,
        "deletion_forgetting",
        lambda task: task.metadata.extra["stratification"]["lifecycle_cell"]
        == "logical_ttl_expiry",
    )
    family_e_collateral = _task(
        tasks,
        "deletion_forgetting",
        lambda task: task.metadata.extra["stratification"]["lifecycle_cell"]
        == "scoped_delete_protected_collateral",
    )
    family_e_forgotten = _task(
        tasks,
        "deletion_forgetting",
        lambda task: task.metadata.extra["stratification"]["lifecycle_cell"]
        == "explicit_object_or_attribute_deletion"
        and task.metadata.extra["stratification"]["deletion_scope"] == "object",
    )
    family_f_previous = _task(
        tasks,
        "current_historical_query",
        lambda task: task.queries[0].selector.kind == "previous",
    )
    family_f_ordered = _task(
        tasks,
        "current_historical_query",
        lambda task: task.queries[0].selector.kind == "ordered_history",
    )
    family_g = _task(
        tasks,
        "long_horizon_memory_synthesis",
        lambda task: task.gold_evidence[0].stale_alternative is not None,
    )
    return (
        ("wrong_delete_scope", family_e_attribute, "wrong_delete_scope"),
        ("missed_ttl", family_e_ttl, "ttl_violation"),
        ("collateral_deletion", family_e_collateral, "collateral_corruption"),
        ("retained_forgotten_value", family_e_forgotten, "deletion_failure"),
        ("wrong_historical_version", family_f_previous, "version_confusion"),
        ("wrong_history_order", family_f_ordered, "version_confusion"),
        ("stale_g_propagation", family_g, "stale_propagation"),
        ("fabricated_evidence", family_g, "evidence_linkage_error"),
    )


def test_core_v3_corrupted_control_registry_is_exact_and_smoke_only() -> None:
    from mub.vnext.adapters.corrupted_v3 import (
        CORRUPTED_CONTROL_IDS_V3,
        corrupted_control_metadata_v3,
    )

    assert CORRUPTED_CONTROL_IDS_V3 == (
        "wrong_delete_scope",
        "missed_ttl",
        "collateral_deletion",
        "retained_forgotten_value",
        "wrong_historical_version",
        "wrong_history_order",
        "stale_g_propagation",
        "fabricated_evidence",
    )
    for control_id in CORRUPTED_CONTROL_IDS_V3:
        metadata = corrupted_control_metadata_v3(control_id)
        assert metadata.smoke_only is True
        assert metadata.leaderboard_eligible is False
        assert metadata.expected_failure_flags


def test_core_v3_corrupted_controls_trigger_canonical_scorer_flags(
    core_control_tasks,
) -> None:


    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.adapters.corrupted_v3 import (
        apply_corrupted_control_v3,
        corrupted_control_adapter_info_v3,
        corrupted_control_metadata_v3,
    )
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3
    from mub.vnext.scoring.scorer_v3 import score_task_v3

    for control_id, task, expected_flag in _control_cases(core_control_tasks):
        reference = ReferenceAdapterV3(task)
        baseline = execute_task_v3(
            task,
            reference,
            RuntimeConfigV3(run_id=f"control-{control_id}"),
        )
        assert baseline.completion_status is CompletionStatus.COMPLETED
        baseline_score = score_task_v3(
            task,
            baseline,
            _authenticated_context(
                task,
                baseline,
                reference.adapter_info(),
                reference.capabilities(),
            ),
        )
        baseline_flags = {
            flag.value if hasattr(flag, "value") else flag
            for flag in baseline_score.failure_flags
        }
        assert expected_flag not in baseline_flags

        corrupted = apply_corrupted_control_v3(task, baseline, control_id)
        metadata = corrupted_control_metadata_v3(control_id)
        info = corrupted_control_adapter_info_v3(control_id)
        score = score_task_v3(
            task,
            corrupted,
            _authenticated_context(
                task,
                corrupted,
                info,
                reference.capabilities(),
            ),
        )
        flags = {
            flag.value if hasattr(flag, "value") else flag
            for flag in score.failure_flags
        }

        assert corrupted.adapter_id == info.adapter_id
        assert metadata.control_id == control_id
        assert metadata.smoke_only is True
        assert metadata.leaderboard_eligible is False
        assert expected_flag in metadata.expected_failure_flags
        assert expected_flag in flags, (control_id, flags, score.model_dump(mode="json"))


def test_fabricated_evidence_changes_only_citations_and_not_answer(
    core_control_tasks,
) -> None:
    from mub.vnext.adapters.core_v3 import ReferenceAdapterV3
    from mub.vnext.adapters.corrupted_v3 import apply_corrupted_control_v3
    from mub.vnext.runtime.engine_v3 import RuntimeConfigV3, execute_task_v3

    task = _task(
        core_control_tasks,
        "long_horizon_memory_synthesis",
        lambda task: task.gold_evidence[0].stale_alternative is not None,
    )
    baseline = execute_task_v3(
        task,
        ReferenceAdapterV3(task),
        RuntimeConfigV3(run_id="fabricated-evidence-layer-check"),
    )
    corrupted = apply_corrupted_control_v3(
        task, baseline, "fabricated_evidence"
    )

    assert (
        corrupted.answer_predictions[0].parsed_answer
        == baseline.answer_predictions[0].parsed_answer
    )
    assert (
        corrupted.answer_predictions[0].raw_output
        == baseline.answer_predictions[0].raw_output
    )
    assert (
        corrupted.answer_predictions[0].cited_event_ids
        != baseline.answer_predictions[0].cited_event_ids
    )
