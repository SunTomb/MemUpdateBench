from __future__ import annotations

from mub.vnext.contracts.v3.runtime import ParsedManagerActionV3, TaskRunRecordV3
from mub.vnext.contracts.v3.task import GoldActionV3, MemUpdateTaskV3


def bind_action_pairs_v3(
    task: MemUpdateTaskV3,
    run: TaskRunRecordV3,
) -> tuple[tuple[GoldActionV3, ParsedManagerActionV3 | None], ...]:
    gold_by_id = {action.action_id: action for action in task.actions}
    observed_by_id = {action.action_id: action for action in run.parsed_actions}
    unknown_action_ids = [action_id for action_id in observed_by_id if action_id not in gold_by_id]
    if unknown_action_ids:
        raise ValueError(f"runtime contains unknown observed action_id values: {sorted(unknown_action_ids)}")
    for action_id, observed in observed_by_id.items():
        gold = gold_by_id[action_id]
        if observed.event_id != gold.event_id:
            raise ValueError(
                f"observed action event_id mismatch for action_id {action_id!r}: "
                f"expected {gold.event_id!r}, got {observed.event_id!r}"
            )
    return tuple((gold, observed_by_id.get(gold.action_id)) for gold in task.actions)


__all__ = ["bind_action_pairs_v3"]
