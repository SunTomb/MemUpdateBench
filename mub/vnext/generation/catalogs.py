from __future__ import annotations

import hashlib

from pydantic import RootModel

from mub.vnext.generation.identity import _validate_strict_json
from mub.vnext.io import canonical_json_bytes


NAMESPACES = (
    "personal",
    "work",
    "community",
    "services",
)

RELATION_QUALIFIED_ENTITIES = (
    "friend_alex",
    "manager_alex",
    "neighbor_alex",
    "friend_jordan",
    "colleague_jordan",
    "trainer_jordan",
    "cousin_morgan",
    "client_morgan",
    "landlord_morgan",
    "doctor_riley",
    "teammate_riley",
    "vendor_riley",
)

SAME_NAME_ENTITIES = (
    ("friend_alex", "manager_alex", "neighbor_alex"),
    ("friend_jordan", "colleague_jordan", "trainer_jordan"),
    ("cousin_morgan", "client_morgan", "landlord_morgan"),
    ("doctor_riley", "teammate_riley", "vendor_riley"),
)

ALIAS_MAPPINGS = (
    ("alex_from_book_club", "friend_alex"),
    ("alex_at_work", "manager_alex"),
    ("alex_next_door", "neighbor_alex"),
    ("jordan_from_school", "friend_jordan"),
    ("jordan_on_my_team", "colleague_jordan"),
    ("coach_jordan", "trainer_jordan"),
    ("cousin_mo", "cousin_morgan"),
    ("morgan_the_client", "client_morgan"),
    ("my_landlord", "landlord_morgan"),
    ("dr_riley", "doctor_riley"),
    ("riley_from_the_team", "teammate_riley"),
    ("riley_the_supplier", "vendor_riley"),
)

CANONICAL_ATTRIBUTES = (
    "city",
    "employer",
    "favorite_color",
    "phone_number",
    "preferred_cafe",
    "project_code",
    "shipping_address",
    "timezone",
)

VALUES = (
    "amber",
    "blue",
    "coral",
    "green",
    "indigo",
    "red",
    "Berlin",
    "Lisbon",
    "Osaka",
    "Quito",
    "Toronto",
    "Zurich",
    "Aster Labs",
    "Beacon Works",
    "Cedar Group",
    "Delta Systems",
    "Elm Studio",
    "Fjord Analytics",
    "+1-202-555-0101",
    "+1-202-555-0114",
    "+1-202-555-0127",
    "Cafe North",
    "Juniper Corner",
    "Willow House",
)

# The legacy flat ``VALUES`` catalog remains exported for compatibility with
# older consumers.  Pilot generation must use this attribute-aware catalog so
# that rendered facts read like plausible facts rather than arbitrary labels.
ATTRIBUTE_VALUES = {
    "city": (
        "Berlin", "Lisbon", "Osaka", "Quito", "Toronto", "Zurich", "Nairobi", "Dublin",
        "Seoul", "Helsinki", "Prague", "Melbourne", "Montreal", "Kyoto", "Accra", "Tallinn",
        "Vienna", "Lima", "Taipei", "Denver", "Rabat", "Bergen", "Manila", "Valencia",
    ),
    "employer": (
        "Aster Labs", "Beacon Works", "Cedar Group", "Delta Systems", "Elm Studio", "Fjord Analytics",
        "Harbor Partners", "Ivory Health", "Juniper Media", "Keystone Robotics", "Lumen Foods", "Meridian Design",
        "Northstar Energy", "Orchid Logistics", "Pinecrest Legal", "Quartz Finance", "Riverline Transit", "Summit Health",
        "Tandem Learning", "Umber Textiles", "Verdant Foods", "Westbridge Labs", "Yarrow Books", "Zenith Software",
    ),
    "favorite_color": (
        "amber", "azure", "beige", "black", "blue", "bronze", "coral", "crimson",
        "emerald", "gold", "green", "indigo", "ivory", "lilac", "maroon", "navy",
        "olive", "orange", "rose", "silver", "teal", "violet", "white", "yellow",
    ),
    "phone_number": tuple(f"+1-202-555-{number:04d}" for number in range(101, 125)),
    "preferred_cafe": (
        "Cafe North", "Juniper Corner", "Willow House", "Harbor Coffee", "Maple Roastery", "Orchid Espresso",
        "Pine & Press", "Quartz Cafe", "Riverside Coffee", "Summit Beans", "Tandem Cafe", "Union Roasters",
        "Violet Cup", "West End Cafe", "Yellow Finch", "Acorn Coffee", "Birch House", "Copper Kettle",
        "Daybreak Cafe", "Ember Roasters", "Fern & Foam", "Garden Gate Cafe", "Hearth Coffee", "Ivy Corner",
    ),
    "project_code": tuple(f"{prefix}-{number:02d}" for prefix in ("ALPHA", "BETA", "GAMMA") for number in range(1, 9)),
    "shipping_address": (
        "101 Cedar Avenue", "22 Harbor Street", "8 Juniper Lane", "47 Willow Road", "315 Maple Drive", "9 Orchard Way",
        "64 Pine Street", "18 River Road", "203 Summit Avenue", "71 Valley Lane", "56 Meadow Street", "12 Lantern Road",
        "88 Garden Avenue", "34 Station Road", "190 Lake Street", "7 Market Lane", "42 Bridge Road", "109 Hill Street",
        "26 Forest Drive", "73 Park Avenue", "15 Clover Lane", "61 College Road", "28 Sunrise Street", "94 Westgate Road",
    ),
    "timezone": (
        "UTC-12:00", "UTC-11:00", "UTC-10:00", "UTC-09:00", "UTC-08:00", "UTC-07:00",
        "UTC-06:00", "UTC-05:00", "UTC-04:00", "UTC-03:00", "UTC-02:00", "UTC-01:00",
        "UTC+00:00", "UTC+01:00", "UTC+02:00", "UTC+03:00", "UTC+04:00", "UTC+05:00",
        "UTC+06:00", "UTC+07:00", "UTC+08:00", "UTC+09:00", "UTC+10:00", "UTC+11:00",
    ),
}


def values_for_attribute(attribute: str) -> tuple[str, ...]:
    """Return the reviewed value catalog for one canonical attribute."""
    if type(attribute) is not str:
        raise TypeError("attribute must be an exact string")
    try:
        return ATTRIBUTE_VALUES[attribute]
    except KeyError as exc:
        raise ValueError(f"unsupported canonical attribute: {attribute}") from exc

REFERENCE_CONDITION_LABELS = (
    ("alias", "alias"),
    ("same_name", "same-name"),
    ("namespace_collision", "namespace collision"),
    ("attribute_paraphrase", "attribute paraphrase"),
)

REFERENCE_QUERY_TEMPLATE_SETS = (
    (
        "direct",
        "Candidate memory entries: $candidates Reference: $references "
        "$resolution_instruction $abstention_instruction",
        "Resolve the reference against the entries and return the current value for the one matching entry.",
        "If it is ambiguous or has no match, respond with ABSTAIN rather than guessing.",
    ),
    (
        "conversational",
        "Here are the candidate memory entries: $candidates Please resolve this reference: $references "
        "$resolution_instruction $abstention_instruction",
        "Please identify the one matching entry and report its current value.",
        "If more than one entry or no entry matches, please ABSTAIN rather than guess a value.",
    ),
    (
        "correction",
        "Check these candidate memory entries: $candidates Reference: $references "
        "$resolution_instruction $abstention_instruction",
        "Use the reference to select exactly one entry before returning its current value.",
        "When it leaves ambiguity or no match, output ABSTAIN; never guess a value.",
    ),
)


SURFACE_TEMPLATE_SETS = (
    (
        "direct",
        "Remember $targets with value $value.",
        "Change $targets to $value.",
        "Forget $targets.",
        "$statement No memory change is required.",
        "What is the current value of $targets?",
        "Are all queried targets absent from memory: $targets?",
        "How many queried targets are absent from memory: $targets?",
        "Report each target's absence status (true if absent, false if present), in listed order or keyed by target for $targets.",
        "What is the aggregate status of $targets: absent, present, or mixed?",
    ),
    (
        "conversational",
        "Please add $targets as $value to memory.",
        "Please revise the stored value for $targets to $value.",
        "Please remove $targets from memory.",
        "$statement Keep memory unchanged.",
        "Can you report the latest value for $targets?",
        "Can you confirm whether all of $targets are absent from memory?",
        "Can you report how many of $targets are absent from memory?",
        "Can you report each target's absence status for $targets (true if absent, false if present), in listed order or keyed by target?",
        "Can you report whether the status of $targets is absent, present, or mixed?",
    ),
    (
        "correction",
        "Create a memory entry for $targets: $value.",
        "Correct the stored value for $targets to $value.",
        "Erase the stored entry for $targets.",
        "$statement Do not write anything to memory.",
        "According to the record, what value belongs to $targets?",
        "According to the record, are all of $targets absent?",
        "According to the record, how many of $targets are absent?",
        "According to the record, report per-target absence statuses for $targets (true if absent, false if present), in listed order or keyed by target.",
        "According to the record, is the status of $targets absent, present, or mixed?",
    ),
)


class _SelectionPayload(RootModel[object]):
    pass


def _selection_key(seed_payload: object, value: str) -> tuple[bytes, bytes]:
    candidate_payload = _SelectionPayload(
        root={"seed_payload": seed_payload, "candidate_value": value}
    )
    digest = hashlib.sha256(canonical_json_bytes(candidate_payload)).digest()
    value_bytes = canonical_json_bytes(_SelectionPayload(root=value))
    return digest, value_bytes


def select_conflicting_values(
    values: list[str] | tuple[str, ...],
    current_value: str,
    count: int,
    seed_payload: object,
) -> tuple[str, ...]:
    """Select distinct non-current values using canonical hash ordering."""
    if type(values) not in {list, tuple}:
        raise TypeError("values must be an exact list or tuple")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if type(current_value) is not str:
        raise TypeError("current_value must be an exact string")
    _validate_strict_json(seed_payload, "seed_payload")

    distinct_candidates = set()
    for value in values:
        if type(value) is not str:
            raise TypeError("values must contain only exact strings")
        if value != current_value:
            distinct_candidates.add(value)

    if count > len(distinct_candidates):
        raise ValueError(
            "insufficient distinct conflicting values after excluding current_value"
        )
    if count == 0:
        return ()

    ordered = sorted(
        distinct_candidates,
        key=lambda value: _selection_key(seed_payload, value),
    )
    return tuple(ordered[:count])


__all__ = [
    "ALIAS_MAPPINGS",
    "ATTRIBUTE_VALUES",
    "CANONICAL_ATTRIBUTES",
    "NAMESPACES",
    "REFERENCE_CONDITION_LABELS",
    "REFERENCE_QUERY_TEMPLATE_SETS",
    "RELATION_QUALIFIED_ENTITIES",
    "SAME_NAME_ENTITIES",
    "SURFACE_TEMPLATE_SETS",
    "VALUES",
    "values_for_attribute",
    "select_conflicting_values",
]
