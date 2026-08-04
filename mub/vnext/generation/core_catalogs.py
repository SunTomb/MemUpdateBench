from __future__ import annotations

from mub.vnext.generation.surface_catalog import SurfaceCatalog


CORE_SURFACE_CATALOG_VERSION = "vnext-core-surfaces-v1"
CORE_SURFACE_IDS = (
    "explicit_canonical",
    "concise_natural",
    "short_dialogue_lifecycle_intent",
    "controlled_adversarial_paraphrase",
)

CORE_SURFACE_TEMPLATE_SETS = (
    (
        "explicit_canonical",
        "ADD $targets = $value.",
        "UPDATE $targets = $value.",
        "DELETE $targets.",
        "$statement NOOP.",
        "QUERY CURRENT $targets.",
        "QUERY ALL_ABSENT $targets.",
        "QUERY ABSENT_COUNT $targets.",
        "QUERY ABSENCE_BY_TARGET $targets (true if absent, false if present; listed order or keyed by target).",
        "QUERY AGGREGATE_STATUS $targets (absent, present, or mixed).",
    ),
    (
        "concise_natural",
        "Store $value for $targets.",
        "Set the latest value of $targets to $value.",
        "Remove $targets from memory.",
        "$statement Leave memory unchanged.",
        "What is the latest value of $targets?",
        "Are all of $targets absent from memory?",
        "How many of $targets are absent from memory?",
        "Give each target's absence status for $targets (true if absent, false if present), in listed order or keyed by target.",
        "Is the status of $targets absent, present, or mixed?",
    ),
    (
        "short_dialogue_lifecycle_intent",
        "I learned a new fact: $targets has value $value. Add it to memory.",
        "That fact changed: $targets now has value $value. Replace the earlier value.",
        "That fact is no longer valid: remove $targets from memory.",
        "$statement This is context only, so do not change memory.",
        "What does memory currently say for $targets?",
        "Have all of $targets been removed from memory?",
        "How many of $targets have been removed from memory?",
        "For $targets, tell me which are removed (true if absent, false if present), in listed order or keyed by target.",
        "For $targets, is the overall status absent, present, or mixed?",
    ),
    (
        "controlled_adversarial_paraphrase",
        "Without altering any other record, commit $value as the stored value for $targets.",
        "Supersede any earlier value for $targets; the current value is $value.",
        "Disregard any lingering value for $targets; it must no longer be stored.",
        "Although this may resemble an update, $statement It does not authorize a memory write.",
        "Ignore superseded values and return the current value attached to $targets.",
        "After accounting for all writes and removals, are all of $targets absent?",
        "After accounting for all writes and removals, how many of $targets are absent?",
        "After accounting for all writes and removals, report each target's status for $targets (true if absent, false if present), in listed order or keyed by target.",
        "After accounting for all writes and removals, classify $targets as absent, present, or mixed.",
    ),
)

CORE_REFERENCE_QUERY_TEMPLATE_SETS = (
    (
        "explicit_canonical",
        "CANDIDATES $candidates REFERENCES $references $resolution_instruction $abstention_instruction",
        "Resolve exactly one candidate and return its current value.",
        "If resolution is ambiguous or missing, return ABSTAIN.",
    ),
    (
        "concise_natural",
        "Candidates: $candidates Reference: $references $resolution_instruction $abstention_instruction",
        "Choose the one matching candidate and give its latest value.",
        "If none or more than one matches, answer ABSTAIN.",
    ),
    (
        "short_dialogue_lifecycle_intent",
        "Here are the memory candidates: $candidates I mean: $references $resolution_instruction $abstention_instruction",
        "Use the reference to identify one current record and report its value.",
        "If you cannot identify exactly one record, say ABSTAIN rather than guess.",
    ),
    (
        "controlled_adversarial_paraphrase",
        "Possible records: $candidates Indirect reference: $references $resolution_instruction $abstention_instruction",
        "Resolve the indirect wording to exactly one record before returning the current value.",
        "Ambiguity or no match requires ABSTAIN, even if a candidate seems plausible.",
    ),
)

CORE_SURFACE_CATALOG_V1 = SurfaceCatalog(
    catalog_version=CORE_SURFACE_CATALOG_VERSION,
    template_sets=CORE_SURFACE_TEMPLATE_SETS,
    reference_query_template_sets=CORE_REFERENCE_QUERY_TEMPLATE_SETS,
    speakers=("System", "User", "User", "User"),
    source_namespace="vnext_core",
    task_tag="vnext_core",
    normalization_version="vnext-core-semantic-v1",
    split_policy_version="vnext-core-group-first-v1",
)


__all__ = [
    "CORE_REFERENCE_QUERY_TEMPLATE_SETS",
    "CORE_SURFACE_CATALOG_V1",
    "CORE_SURFACE_CATALOG_VERSION",
    "CORE_SURFACE_IDS",
    "CORE_SURFACE_TEMPLATE_SETS",
]
