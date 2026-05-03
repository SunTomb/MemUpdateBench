# P6.3 Method Definition Table

| Paper label | Internal mode/name | Role in paper | Expected strength | Main limitation |
| --- | --- | --- | --- | --- |
| Constrained CRUD | `constrained_slot_crud` | Oracle-like diagnostic upper bound | Exact slot resolution and zero stale same-slot entries | Not a natural deployable external-memory system |
| Raw append | `raw_add` | Append-only baseline | Preserves final value under oracle slot lookup | Accumulates stale same-slot evidence and large memory size |
| Heuristic CRUD | `heuristic_crud` | Partial-compaction baseline | Reduces memory size relative to raw append | Still retains stale same-slot entries and collapses under slot prompting |
| Long25 | `learned_constrained_slot` checkpoint `long25` | Learned compact manager | Lower stale burden and smaller memory size | Misses some final updates under hard distractors |

## Usage in paper

Use this table or its content when introducing methods. It prevents the main result from being misread as a generic leaderboard: each method occupies a different diagnostic role in the reliability--compactness--robustness tradeoff.
