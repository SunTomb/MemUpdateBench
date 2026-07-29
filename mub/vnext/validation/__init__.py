from mub.vnext.contracts import MemUpdateTask
from mub.vnext.validation.issues import (
    ValidationIssue,
    ValidationReport,
    build_report,
    merge_reports,
)
from mub.vnext.validation.replay import (
    DistractorValidationPolicy,
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
from mub.vnext.validation.pilot import (
    validate_family_a_task,
    validate_family_b_task,
    validate_family_d_task,
    validate_pilot_task,
)


def validate_legacy_task_semantics(task: MemUpdateTask) -> ValidationReport:
    """Validate a task under the explicit audited legacy compatibility boundary.

    Caller selection of this API is the authorization. The legacy distractor policy
    recognizes only the compiler's exact historical neutral-NOOP ambiguity shape;
    mutable provenance or task metadata alone never selects this context.
    """
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(
            task,
            policy=DistractorValidationPolicy.LEGACY_AUDITED_AMBIGUITY,
        ),
    )


def validate_task_semantics(task: MemUpdateTask) -> ValidationReport:
    """Run historical family-agnostic structural, replay, and distractor checks.

    This API does not claim Pilot family-semantic completeness. Call
    ``validate_pilot_task`` when the caller explicitly intends Pilot semantics.
    """
    return merge_reports(
        validate_task(task),
        validate_gold_replay(task),
        validate_distractors(task),
    )


__all__ = [
    "DistractorValidationPolicy",
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
    "validate_family_a_task",
    "validate_family_b_task",
    "validate_family_d_task",
    "validate_gold_replay",
    "validate_legacy_task_semantics",
    "validate_pilot_task",
    "validate_splits",
    "validate_task",
    "validate_task_semantics",
]
