from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from mub.vnext.io import canonical_json_bytes
from mub.vnext.statistics.contracts_v3 import (
    CaseCategory,
    Task13AnswerProjectionV1,
    Task13CaseRecordV1,
    Task13RunCaseCoverageV1,
    Task13RunProjectionV1,
    Task13ScoreProjectionV1,
    Task13TaskProjectionV1,
    Task13TimelineProjectionV1,
    Task13RetrievalProjectionV1,
)


_CASE_CATEGORIES: tuple[CaseCategory, ...] = (
    "correct",
    "stale_copied",
    "answer_parse_invalid",
    "other_wrong",
)
_REQUIRED_INPUT_HASHES = (
    "core_tasks",
    "core_task_manifest",
    "task12_matrix_summary",
)


@dataclass(frozen=True, slots=True)
class Task13CasesResultV1:
    cases: tuple[Task13CaseRecordV1, ...]
    coverage: tuple[Task13RunCaseCoverageV1, ...]


def _json(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _input_hashes(input_hashes: Mapping[str, str]) -> tuple[str, str, str]:
    if not isinstance(input_hashes, Mapping):
        raise TypeError("input_hashes must be a mapping")
    values: list[str] = []
    for key in _REQUIRED_INPUT_HASHES:
        value = input_hashes.get(key)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"input_hashes[{key!r}] must be a lowercase SHA-256 digest")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def classify_task13_case_v1(observation: Any) -> CaseCategory:
    score = observation.score
    if score.answer_scores.exact_match == 1:
        return "correct"
    if score.answer_scores.stale_copied == 1:
        return "stale_copied"
    if any(not prediction.format_valid for prediction in observation.run.answer_predictions):
        return "answer_parse_invalid"
    return "other_wrong"


def _validate_observation(observation: Any) -> None:
    task = observation.task
    run = observation.run
    score = observation.score
    source = observation.source
    core_id = task.metadata.split_key.semantic_core_id
    if observation.semantic_core_id != core_id:
        raise ValueError("case observation semantic-core ID differs from task metadata")
    if run.task_id != task.task_id or score.task_id != task.task_id:
        raise ValueError("case task ID differs across authenticated evidence")
    if run.run_id != score.run_id or source.run_id != run.run_id:
        raise ValueError("case run ID differs across authenticated evidence")
    if run.adapter_id != score.adapter_id:
        raise ValueError("case adapter ID differs across authenticated evidence")
    if score.task_family != task.task_family or score.difficulty != task.difficulty:
        raise ValueError("case task metadata differs from authenticated score")


def _case_id(run_id: str, task_id: str, category: CaseCategory) -> str:
    raw = json.dumps(
        {"category": category, "run_id": run_id, "task_id": task_id},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"case-{hashlib.sha256(raw).hexdigest()}"


def project_task13_case_v1(
    observation: Any,
    *,
    input_hashes: Mapping[str, str],
) -> Task13CaseRecordV1:
    """Copy one case from immutable authenticated task, run, and score evidence."""

    _validate_observation(observation)
    task_hash, task_manifest_hash, summary_hash = _input_hashes(input_hashes)
    task = observation.task
    run = observation.run
    score = observation.score
    category = classify_task13_case_v1(observation)
    private = task.source.provenance.get("redistributable") is False
    source_payload = _json(task.source)
    if private:
        source_payload["source_uri"] = None
    timeline: list[dict[str, Any]] = []
    for event in task.events:
        item = _json(event)
        if private:
            item.pop("raw_text", None)
            item.pop("normalized_text", None)
        timeline.append(item)
    metric_layers = {
        field_name: _json(getattr(score, field_name))
        for field_name in type(score).model_fields
        if field_name.endswith("_scores")
    }
    support = {
        path: _json(record)
        for path, record in score.supported_metric_fields.items()
    }
    flags = tuple(
        flag.value if hasattr(flag, "value") else flag for flag in score.failure_flags
    )
    return Task13CaseRecordV1(
        case_id=_case_id(run.run_id, task.task_id, category),
        category=category,
        run_id=run.run_id,
        task_id=task.task_id,
        semantic_core_id=observation.semantic_core_id,
        answer_model_slot=observation.slot,
        k=observation.k,
        task_artifact_sha256=task_hash,
        task_manifest_sha256=task_manifest_hash,
        run_manifest_sha256=observation.source.run_manifest_sha256,
        score_artifact_sha256=observation.source.score_artifact_sha256,
        matrix_summary_sha256=summary_hash,
        task=Task13TaskProjectionV1(
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            family=task.task_family,
            difficulty=task.difficulty.value,
            metadata=_json(task.metadata),
            source=source_payload,
            target_objects=tuple(_json(item) for item in task.target_objects),
            queries=tuple(_json(item) for item in task.queries),
            gold_actions=tuple(_json(item) for item in task.actions),
        ),
        timeline=Task13TimelineProjectionV1(
            redacted=private,
            items=tuple(timeline),
        ),
        run=Task13RunProjectionV1(
            run_id=run.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            completion_status=run.completion_status.value,
            parsed_actions=tuple(_json(item) for item in run.parsed_actions),
            memory_snapshots=tuple(_json(item) for item in run.memory_snapshots),
            final_state=(
                _json(run.memory_snapshots[-1])["state_by_object"]
                if run.memory_snapshots
                else None
            ),
            system_events=tuple(dict(item) for item in run.system_events),
            provenance=_json(run.parser_extractor_provenance),
            exceptions=tuple(dict(item) for item in run.exceptions),
        ),
        score=Task13ScoreProjectionV1(
            run_id=run.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            metric_layers=metric_layers,
            support=support,
            failure_flags=flags,
            primary_failure=score.primary_failure,
        ),
        retrieval=Task13RetrievalProjectionV1(
            run_id=run.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            available=bool(run.retrieval_traces),
            items=tuple(_json(item) for item in run.retrieval_traces),
        ),
        answer=Task13AnswerProjectionV1(
            run_id=run.run_id,
            task_id=task.task_id,
            semantic_core_id=observation.semantic_core_id,
            category=category,
            available=bool(run.answer_predictions),
            items=tuple(_json(item) for item in run.answer_predictions),
        ),
    )


def select_task13_cases_for_run_v1(
    run: Any,
    *,
    input_hashes: Mapping[str, str],
) -> tuple[Task13CaseRecordV1, ...]:
    observations = tuple(run.observations)
    if not observations:
        raise ValueError("each Task 13 run must contain authenticated observations")
    if any(observation.source != run.source for observation in observations):
        raise ValueError("case observations must share the authenticated run source")
    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.semantic_core_id.encode("utf-8"),
            observation.task.task_id.encode("utf-8"),
        ),
    )
    selected: list[Task13CaseRecordV1] = []
    selected_tasks: set[str] = set()
    for category in _CASE_CATEGORIES:
        candidate = next(
            (
                observation
                for observation in ordered
                if observation.task.task_id not in selected_tasks
                and classify_task13_case_v1(observation) == category
            ),
            None,
        )
        if candidate is None:
            continue
        case = project_task13_case_v1(candidate, input_hashes=input_hashes)
        selected.append(case)
        selected_tasks.add(case.task_id)
    if not selected:
        raise ValueError("each Task 13 run must select at least one case")
    return tuple(selected)


def _coverage(run_id: str, cases: Sequence[Task13CaseRecordV1]) -> Task13RunCaseCoverageV1:
    by_category = {case.category: case.case_id for case in cases}
    return Task13RunCaseCoverageV1(
        run_id=run_id,
        correct_case_id=by_category.get("correct"),
        stale_copied_case_id=by_category.get("stale_copied"),
        answer_parse_invalid_case_id=by_category.get("answer_parse_invalid"),
        other_wrong_case_id=by_category.get("other_wrong"),
    )


def build_task13_cases_v1(matrix_input: Any) -> Task13CasesResultV1:
    runs = tuple(matrix_input.runs)
    if len(runs) != 18:
        raise ValueError("Task 13 case export requires exactly 18 authenticated runs")
    run_ids = tuple(run.source.run_id for run in runs)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Task 13 case export requires unique run IDs")
    cases: list[Task13CaseRecordV1] = []
    coverage: list[Task13RunCaseCoverageV1] = []
    for run in runs:
        run_cases = select_task13_cases_for_run_v1(
            run,
            input_hashes=matrix_input.input_hashes,
        )
        cases.extend(run_cases)
        coverage.append(_coverage(run.source.run_id, run_cases))
    if not 18 <= len(cases) <= 72:
        raise AssertionError("Task 13 case export must contain between 18 and 72 cases")
    return Task13CasesResultV1(tuple(cases), tuple(coverage))


def verify_task13_cases_v1(
    cases: Sequence[Task13CaseRecordV1],
    matrix_input: Any,
) -> None:
    supplied = tuple(cases)
    expected = build_task13_cases_v1(matrix_input).cases
    if len(supplied) != len(expected):
        raise ValueError("case export does not equal authenticated source evidence")
    if tuple(case.case_id for case in supplied) != tuple(case.case_id for case in expected):
        raise ValueError("case export does not equal authenticated source evidence")
    for candidate, source in zip(supplied, expected):
        if canonical_json_bytes(candidate) != canonical_json_bytes(source):
            raise ValueError("case export does not equal authenticated source evidence")


__all__ = [
    "Task13CasesResultV1",
    "build_task13_cases_v1",
    "classify_task13_case_v1",
    "project_task13_case_v1",
    "select_task13_cases_for_run_v1",
    "verify_task13_cases_v1",
]
