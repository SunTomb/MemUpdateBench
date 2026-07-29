from mub.vnext.contracts import MemUpdateTask, TaskFamily
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
    merge_reports,
)
from mub.vnext.validation.replay import (
    ReplayResult,
    replay_actions,
    validate_distractors,
    validate_gold_replay,
)
from mub.vnext.validation.split import (
    FAMILY_STRATIFICATION_AXES,
    SliceDefinition,
    SplitException,
    validate_splits,
)
from mub.vnext.validation.task import validate_task
from mub.vnext.validation.pilot import validate_family_d_task


def validate_task_semantics(task) -> ValidationReport:
    if (
        type(task) is MemUpdateTask
        and getattr(task, "task_family", None)
        == TaskFamily.NOOP_WRITE_DISCIPLINE.value
    ):
        return validate_family_d_task(task)
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(task),
    )


__all__ = [
    "FAMILY_STRATIFICATION_AXES",
    "ReplayResult",
    "SliceDefinition",
    "SplitException",
    "ValidationIssue",
    "ValidationReport",
    "build_report",
    "merge_reports",
    "replay_actions",
    "validate_distractors",
    "validate_family_d_task",
    "validate_gold_replay",
    "validate_splits",
    "validate_task",
    "validate_task_semantics",
]
