from enum import Enum


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Operation(StringEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    NOOP = "NOOP"
    DELETE = "DELETE"


class ActionScope(StringEnum):
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    ENTITY = "entity"
    NAMESPACE = "namespace"
    TTL = "ttl"


class EventRole(StringEnum):
    LATEST_GOLD = "latest_gold"
    STALE_SAME_SLOT = "stale_same_slot"
    DUPLICATE_CURRENT = "duplicate_current"
    SAME_ENTITY_OTHER_ATTRIBUTE = "same_entity_other_attribute"
    SAME_NAME_OTHER_ENTITY = "same_name_other_entity"
    NOOP_NEAR_MISS = "noop_near_miss"
    NEUTRAL = "neutral"
    DELETION = "deletion"
    HISTORICAL_SUPPORT = "historical_support"


class TaskFamily(StringEnum):
    REPEATED_SAME_SLOT = "repeated_same_slot_update"
    INTERLEAVED_MULTI_SLOT = "interleaved_multi_slot_update"
    ENTITY_ATTRIBUTE_GROUNDING = "entity_attribute_grounding"
    NOOP_WRITE_DISCIPLINE = "noop_write_discipline"
    DELETION_FORGETTING = "deletion_forgetting"
    CURRENT_HISTORICAL_QUERY = "current_historical_query"
    LONG_HORIZON_MEMORY_SYNTHESIS = "long_horizon_memory_synthesis"
    REALISTIC_SOURCE_UPDATE = "realistic_source_update"


class SourceType(StringEnum):
    SYNTHETIC = "synthetic"
    DIALOGUE = "dialogue"
    CHANGELOG = "changelog"
    CALENDAR = "calendar"
    ISSUE = "issue"
    REPORT_REVISION = "report_revision"
    OTHER = "other"


class Split(StringEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    EVALUATION_ONLY = "evaluation_only"


class Difficulty(StringEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHALLENGE = "challenge"


class QueryType(StringEnum):
    CURRENT_STATE = "current_state"
    HISTORICAL_STATE = "historical_state"
    TRANSITION = "transition"
    MULTI_OBJECT = "multi_object"
    DELETION_COMPLIANCE = "deletion_compliance"


class EvaluationMode(StringEnum):
    STATE_DIRECT = "state_direct"
    RETRIEVED_PROMPT = "retrieved_prompt"
    NATIVE_SYSTEM = "native_system"


class AnswerSchema(StringEnum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"


class CompletionStatus(StringEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"


class SupportReason(StringEnum):
    NOT_APPLICABLE = "not_applicable"
    NOT_SUPPORTED = "not_supported"
    RUNTIME_FAILED = "runtime_failed"
    MISSING_ARTIFACT = "missing_artifact"


__all__ = [
    "ActionScope",
    "AnswerSchema",
    "CompletionStatus",
    "Difficulty",
    "EvaluationMode",
    "EventRole",
    "Operation",
    "QueryType",
    "SourceType",
    "Split",
    "StringEnum",
    "SupportReason",
    "TaskFamily",
]
