from mub.vnext.contracts.enums import StringEnum


class QueryTypeV3(StringEnum):
    CURRENT = "current"
    PREVIOUS = "previous"
    POINT_IN_TIME = "point_in_time"
    TRANSITION = "transition"
    ORDERED_HISTORY = "ordered_history"
    MULTI_OBJECT_CURRENT = "multi_object_current"
    UPDATE_SENSITIVE_MULTI_HOP = "update_sensitive_multi_hop"
    MULTI_OBJECT_CURRENT_CONSISTENCY = "multi_object_current_consistency"


class SynthesisKindV3(StringEnum):
    UPDATE_SENSITIVE_MULTI_HOP = "update_sensitive_multi_hop"
    MULTI_OBJECT_CURRENT_CONSISTENCY = "multi_object_current_consistency"


class LedgerEntryStatus(StringEnum):
    PRESENT = "present"
    TOMBSTONE = "tombstone"


class ExecutionStatusV3(StringEnum):
    EXECUTED = "executed"
    NO_EFFECT = "no_effect"
    REJECTED = "rejected"
    NOT_SUPPORTED = "not_supported"
    FAILED = "failed"


class FailureFlagV3(StringEnum):
    WRONG_DELETE_SCOPE = "wrong_delete_scope"
    COLLATERAL_MUTATION = "collateral_mutation"
    TTL_VIOLATION = "ttl_violation"
    FORGOTTEN_VALUE_EXPOSED = "forgotten_value_exposed"
    VERSION_CONFUSION = "version_confusion"
    EVIDENCE_LINKAGE_ERROR = "evidence_linkage_error"
    STALE_PROPAGATION = "stale_propagation"


__all__ = ["ExecutionStatusV3", "FailureFlagV3", "LedgerEntryStatus", "QueryTypeV3", "SynthesisKindV3"]
