from __future__ import annotations

import hashlib
from collections.abc import Sequence

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

SURFACE_TEMPLATE_SETS = (
    (
        "direct",
        "{entity}'s {attribute} is now {value}.",
        "What is {entity}'s current {attribute}?",
    ),
    (
        "conversational",
        "Please remember that {entity} has {attribute} set to {value}.",
        "Can you tell me the latest {attribute} for {entity}?",
    ),
    (
        "correction",
        "Update the record: the {attribute} for {entity} should be {value}.",
        "According to the updates, what is {entity}'s {attribute}?",
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
    values: Sequence[str],
    current_value: str,
    count: int,
    seed_payload: object,
) -> tuple[str, ...]:
    """Select distinct non-current values using canonical hash ordering."""
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
    "CANONICAL_ATTRIBUTES",
    "NAMESPACES",
    "RELATION_QUALIFIED_ENTITIES",
    "SAME_NAME_ENTITIES",
    "SURFACE_TEMPLATE_SETS",
    "VALUES",
    "select_conflicting_values",
]
