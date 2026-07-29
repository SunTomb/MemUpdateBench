from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


INTERLEAVING_PATTERNS = ("round_robin", "burst", "adversarial_adjacent")


def canonical_cross_slot_update_count(base_event_count: int, density: float) -> int:
    if type(base_event_count) is not int or base_event_count <= 0:
        raise TypeError("base_event_count must be a positive exact integer")
    if type(density) is not float or not 0.0 <= density <= 1.0:
        raise TypeError("density must be an exact float in [0, 1]")
    scaled = Decimal(base_event_count) * Decimal(str(density))
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def canonical_interleaving_schedule(
    trajectory_lengths: tuple[int, ...],
    pattern: str,
) -> tuple[tuple[int, int], ...]:
    """Return the canonical ``(slot_index, version_index)`` emission schedule."""
    if type(trajectory_lengths) is not tuple:
        raise TypeError("trajectory_lengths must be an exact tuple")
    if not trajectory_lengths or any(
        type(length) is not int or length <= 0 for length in trajectory_lengths
    ):
        raise ValueError("trajectory_lengths must contain positive exact integers")
    if type(pattern) is not str or pattern not in INTERLEAVING_PATTERNS:
        raise ValueError(f"unsupported interleaving pattern: {pattern}")

    if pattern == "burst":
        return tuple(
            (slot_index, version_index)
            for slot_index, length in enumerate(trajectory_lengths)
            for version_index in range(length)
        )
    if pattern == "round_robin":
        ordered_slots = (*range(1, len(trajectory_lengths)), 0)
        return tuple(
            (slot_index, version_index)
            for version_index in range(max(trajectory_lengths))
            for slot_index in ordered_slots
            if version_index < trajectory_lengths[slot_index]
        )
    return (
        *((0, version_index) for version_index in range(trajectory_lengths[0] - 1)),
        *(
            (slot_index, version_index)
            for slot_index in range(1, len(trajectory_lengths))
            for version_index in range(trajectory_lengths[slot_index])
        ),
        (0, trajectory_lengths[0] - 1),
    )


__all__ = [
    "INTERLEAVING_PATTERNS",
    "canonical_cross_slot_update_count",
    "canonical_interleaving_schedule",
]
