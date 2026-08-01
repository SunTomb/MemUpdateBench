from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mub.vnext.contracts.enums import CompletionStatus
from mub.vnext.contracts.manifest import RunManifest
from mub.vnext.contracts.score import ScoreRecord
from mub.vnext.contracts.task import MemUpdateTask
from mub.vnext.scoring.registry import METRIC_REGISTRY

_MAX_DIAGNOSTIC_CASES = 32
_ADMIN_PROFILE_KEYS = {"task_family", "difficulty", "profile_name", "profile_version", "update_depth_bucket"}


def _value(score: ScoreRecord, path: str):
    layer, field = path.split(".", 1)
    return getattr(score, layer).model_dump(mode="python")[field]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _condition_pairs(task: MemUpdateTask) -> tuple[tuple[str, str], ...]:
    profile = task.metadata.resolved_profile
    pairs = []
    for key in sorted(profile):
        if key not in _ADMIN_PROFILE_KEYS:
            pairs.append((key, _canonical(profile[key])))
    stratification = task.metadata.extra.get("stratification")
    if isinstance(stratification, Mapping):
        for key in sorted(stratification):
            pairs.append((str(key), _canonical(stratification[key])))
    return tuple(pairs or (("condition", "none"),))


def _eligible(manifest: RunManifest) -> bool:
    system_name = manifest.adapter_info.system_name.casefold()
    metadata = manifest.native_vs_extracted_field_summary
    if metadata.get("oracle_smoke_only") is True or metadata.get("smoke_control") is True:
        return False
    if metadata.get("leaderboard_eligible") is False:
        return False
    return system_name != "oracle_smoke_only" and "smoke_control" not in system_name and "corrupted_control" not in system_name


def _metric_summary(scores: Sequence[ScoreRecord], tasks: Mapping[str, MemUpdateTask], path: str) -> dict[str, dict[str, float | None]]:
    values = [(score, _value(score, path)) for score in scores]
    values = [(score, float(value)) for score, value in values if value is not None]
    if not values:
        empty = {"numerator": 0.0, "denominator": 0.0, "value": None}
        return {"micro": empty, "macro": dict(empty)}
    micro_numerator = sum(value for _, value in values)
    micro_denominator = float(len(values))
    by_core: dict[str, list[float]] = defaultdict(list)
    for score, value in values:
        core = tasks[score.task_id].metadata.split_key.semantic_core_id
        by_core[core].append(value)
    core_means = [sum(core_values) / len(core_values) for core_values in by_core.values()]
    macro_numerator = sum(core_means)
    macro_denominator = float(len(core_means))
    return {
        "micro": {"numerator": micro_numerator, "denominator": micro_denominator, "value": micro_numerator / micro_denominator},
        "macro": {"numerator": macro_numerator, "denominator": macro_denominator, "value": macro_numerator / macro_denominator},
    }


def _group_rows(scores: Sequence[ScoreRecord], tasks: Mapping[str, MemUpdateTask]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[ScoreRecord]] = defaultdict(list)
    groups[("overall", "overall")].extend(scores)
    for score in scores:
        task = tasks[score.task_id]
        groups[("family", task.task_family)].append(score)
        groups[("difficulty", getattr(task.difficulty, "value", task.difficulty))].append(score)
        groups[("split", task.metadata.split.value)].append(score)
        for key, value in _condition_pairs(task):
            groups[("condition", f"{key}={value}")].append(score)

    rows = []
    for (group_by, group), group_scores in sorted(groups.items()):
        metrics: dict[str, dict[str, dict[str, float | None]]] = {}
        for path in sorted(METRIC_REGISTRY):
            metric = _metric_summary(group_scores, tasks, path)
            metrics[path] = {"micro": metric["micro"], "macro": metric["macro"]}
        rows.append({
            "group_by": group_by,
            "group": group,
            "task_count": len(group_scores),
            "micro": {path: item["micro"] for path, item in metrics.items()},
            "macro": {path: item["macro"] for path, item in metrics.items()},
            "failure_flags": dict(sorted(Counter(flag for score in group_scores for flag in score.failure_flags).items())),
            "primary_failures": dict(sorted(Counter(score.primary_failure for score in group_scores if score.primary_failure).items())),
        })
    return rows


def aggregate_scores(
    scores: Iterable[ScoreRecord],
    tasks: Iterable[MemUpdateTask],
    run_manifest: RunManifest,
    *,
    max_diagnostic_cases: int = _MAX_DIAGNOSTIC_CASES,
) -> dict[str, Any]:
    score_list = tuple(scores)
    task_list = tuple(tasks)
    if any(not isinstance(score, ScoreRecord) for score in score_list):
        raise TypeError("aggregate_scores accepts only ScoreRecord rows")
    if any(not isinstance(task, MemUpdateTask) for task in task_list):
        raise TypeError("aggregate_scores accepts only MemUpdateTask rows")
    task_by_id = {task.task_id: task for task in task_list}
    if len(task_by_id) != len(task_list) or len({score.task_id for score in score_list}) != len(score_list):
        raise ValueError("duplicate task IDs are not aggregable")
    if set(task_by_id) != {score.task_id for score in score_list}:
        raise ValueError("score/task IDs do not match")
    if any(score.run_id != run_manifest.run_id for score in score_list):
        raise ValueError("score rows belong to a different run manifest")
    if run_manifest.expected_task_count != len(task_list):
        raise ValueError("run manifest expected count does not match scores")
    completed = sum(score.completion_status is CompletionStatus.COMPLETED for score in score_list)
    failed = sum(score.completion_status in {CompletionStatus.FAILED, CompletionStatus.PARTIAL} for score in score_list)
    unsupported = sum(score.completion_status is CompletionStatus.NOT_SUPPORTED for score in score_list)
    if (completed, failed, unsupported) != (
        run_manifest.completed_task_count,
        run_manifest.failed_task_count,
        run_manifest.not_supported_task_count,
    ):
        raise ValueError("score completion counts do not match run manifest")
    eligible = _eligible(run_manifest)
    leaderboard_scores = score_list if eligible else ()
    rows = _group_rows(leaderboard_scores, task_by_id)
    failure_examples = [
        {"task_id": score.task_id, "failure_flags": list(score.failure_flags), "primary_failure": score.primary_failure}
        for score in sorted(score_list, key=lambda item: item.task_id)
        if score.failure_flags
    ][:max(0, max_diagnostic_cases)]
    answer_dispositions: Counter[str] = Counter()
    for score in score_list:
        payload = score.legacy_metrics.get("pilot_answer_dispositions", {})
        if isinstance(payload, Mapping):
            for disposition, count in payload.items():
                if isinstance(disposition, str) and type(count) is int and count >= 0:
                    answer_dispositions[disposition] += count
    return {
        "schema_version": run_manifest.schema_version,
        "run_id": run_manifest.run_id,
        "task_manifest": run_manifest.task_manifest.model_dump(mode="json"),
        "counts": {
            "expected": len(task_list),
            "scored": len(score_list),
            "completed": completed,
            "failed": failed,
            "partial": sum(score.completion_status is CompletionStatus.PARTIAL for score in score_list),
            "not_supported": unsupported,
            "not_supported_rate": unsupported / len(task_list) if task_list else 0.0,
            "runtime_failure_rate": failed / len(task_list) if task_list else 0.0,
        },
        "leaderboard": {
            "eligible": eligible,
            "excluded_reason": None if eligible else "oracle_smoke_only_or_smoke_control",
            "rows": rows,
        },
        "diagnostics": {
            "failure_flags": dict(sorted(Counter(flag for score in score_list for flag in score.failure_flags).items())),
            "primary_failures": dict(sorted(Counter(score.primary_failure for score in score_list if score.primary_failure).items())),
            "answer_dispositions": dict(sorted(answer_dispositions.items())),
            "failure_examples": failure_examples,
            "capability_coverage": {
                "adapter_id": run_manifest.adapter_info.adapter_id,
                "system_name": run_manifest.adapter_info.system_name,
                "presentation_level": run_manifest.adapter_capabilities.presentation_level(),
            },
        },
    }


__all__ = ["aggregate_scores"]
