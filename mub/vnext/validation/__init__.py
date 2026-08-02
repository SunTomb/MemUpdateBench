from mub.vnext.contracts import MemUpdateTask
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
from mub.vnext.validation.replay_v3 import (
    EvidenceEvaluationV3,
    QueryResolutionV3,
    ReplayLedgerV3,
    ReplayResultV3,
    ReplayVersionV3,
    evaluate_evidence_v3,
    replay_actions_v3,
    replay_task_v3,
    resolve_query_v3,
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
    validate_family_c_task,
    validate_family_d_task,
    validate_pilot_release,
    validate_pilot_task,
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
    "FAMILY_STRATIFICATION_AXES",
    "EvidenceEvaluationV3",
    "QueryResolutionV3",
    "ReplayLedgerV3",
    "ReplayResult",
    "ReplayResultV3",
    "ReplayVersionV3",
    "SliceDefinition",
    "SplitException",
    "ValidationIssue",
    "ValidationReport",
    "build_report",
    "evaluate_evidence_v3",
    "merge_reports",
    "replay_actions",
    "replay_actions_v3",
    "replay_task_v3",
    "resolve_query_v3",
    "validate_distractors",
    "validate_family_a_task",
    "validate_family_b_task",
    "validate_family_c_task",
    "validate_family_d_task",
    "validate_gold_replay",
    "validate_pilot_release",
    "validate_pilot_task",
    "validate_splits",
    "validate_task",
    "validate_task_semantics",
]
