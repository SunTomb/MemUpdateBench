from __future__ import annotations

import pytest

from mub.vnext.preparation.task12 import RawAppendTrajectoryV1


def test_raw_trajectory_receipt_records_order_versions_and_full_latest_truth() -> None:
    receipt = RawAppendTrajectoryV1(
        task_id="a-000",
        entry_ids=("event-0:key:0", "event-1:key:1"),
        object_ids=("key", "key"),
        event_indices=(0, 1),
        version_indices=(0, 1),
        latest_entry_ids=("event-1:key:1",),
        trajectory_sha256="a" * 64,
    )

    assert receipt.entry_ids == ("event-0:key:0", "event-1:key:1")
    assert receipt.event_indices == (0, 1)
    assert receipt.version_indices == (0, 1)


@pytest.mark.parametrize(
    "change",
    (
        {"entry_ids": ()},
        {"event_indices": (0,)},
        {"version_indices": (0,)},
        {"event_indices": (1, 0)},
        {"latest_entry_ids": ("missing:key:0",)},
        {"latest_entry_ids": ()},
        {"latest_entry_ids": ("event-0:key:0",)},
    ),
)
def test_raw_trajectory_receipt_rejects_malformed_provenance(change) -> None:
    payload = {
        "task_id": "a-000",
        "entry_ids": ("event-0:key:0", "event-1:key:1"),
        "object_ids": ("key", "key"),
        "event_indices": (0, 1),
        "version_indices": (0, 1),
        "latest_entry_ids": ("event-1:key:1",),
        "trajectory_sha256": "a" * 64,
    }
    payload.update(change)

    with pytest.raises(ValueError):
        RawAppendTrajectoryV1(**payload)
