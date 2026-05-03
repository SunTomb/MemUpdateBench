# Experimental Section Draft: Repeated Same-Slot Updates

## Setup

We evaluate how external memory systems behave when the same `(entity, attribute)` slot is updated repeatedly. Each example contains a target memory slot, same-name multi-entity distractors, and semantic near-miss NOOP events. The update frequency is controlled by `k \in {1, 2, 4, 8, 16}`, where larger `k` means more repeated updates to the same slot.

We report four complementary diagnostics:

1. final-state reliability under oracle-like `slot_direct` evaluation,
2. answer robustness under realistic `slot_prompt` evaluation,
3. stale same-slot burden,
4. final memory size.

The compared methods are constrained slot CRUD, raw append, heuristic CRUD, and the learned Long25 manager. Constrained slot CRUD is an upper-bound diagnostic because it has exact slot resolution and removes stale same-slot values by construction.

## Main result

The update-frequency stress test reveals that final-state recoverability alone is not enough to characterize memory quality. Under `slot_direct`, raw append and heuristic CRUD keep state accuracy at 1.00 even at k=16. However, this oracle-like metric hides the stale values retained in memory. Under `slot_prompt`, where the answer layer must condition on the memory contents, stale same-slot burden causes severe answer collapse.

At k=16, raw append retains 14.25 stale same-slot entries and grows to final memory size 52.00. Its slot-prompt EM/F1 falls to 0.07 / 0.10 despite slot-direct state accuracy remaining 1.00. Heuristic CRUD is smaller but still retains 7.44 stale same-slot entries and reaches only 0.10 / 0.13 EM/F1.

Long25 shows the opposite side of the tradeoff. It is much more compact at k=16, with stale same-slot burden 1.13 and final memory size 9.43, and its slot-prompt EM/F1 is higher than append-only and heuristic baselines at 0.48 / 0.53. But this compactness comes with missed final updates: its slot-direct state accuracy is 0.91 rather than 1.00.

Constrained slot CRUD provides the clean diagnostic upper bound. At k=16 it reaches slot-direct state accuracy 1.00, has zero stale same-slot entries, and obtains slot-prompt EM/F1 0.70 / 0.70. State-level analysis finds no state errors, so its remaining slot-prompt failures indicate that even a clean final memory state does not fully remove answer-layer failures under hard distractors.

## Interpretation

The central finding is a tradeoff curve, not a simple method ranking. Append-only memory preserves final values under oracle lookup but accumulates stale same-slot evidence that breaks slot-conditioned answering. Learned compact managers reduce stale burden and memory size, but may miss final updates or remain incompletely compact. A realistic memory benchmark should therefore report final-state reliability, stale burden, compactness, and answer robustness together.

## Figure reference

Use `paper/p63_update_frequency_tradeoff.pdf` as the main figure. The figure shows:

- slot-direct state accuracy,
- slot-prompt exact match,
- stale same-slot entries,
- final memory size.

Suggested in-text reference:

> Figure~\\ref{fig:p63-update-frequency-tradeoff} shows that slot-direct state accuracy remains high for append-only and heuristic managers, but their slot-prompt accuracy collapses as stale same-slot entries accumulate. Long25 occupies a different point on the curve: substantially more compact, but less reliable in final-state tracking.

## Table reference

Use the k=16 table from `paper/p63_update_frequency_latex_snippets.tex` as the main result table.

Suggested in-text reference:

> Table~\\ref{tab:p63-k16-tradeoff} summarizes the high-frequency endpoint. Raw append and heuristic CRUD have perfect slot-direct state accuracy but poor slot-prompt EM/F1, while Long25 reduces stale burden and memory size at the cost of final-state reliability.

## Recommended framing decisions

- Describe constrained slot CRUD as an oracle-like diagnostic upper bound, not as the main deployable external-memory method. Its role is to show what exact slot resolution and perfect stale-value deletion can achieve.
- Present Long25 as a learned compact manager rather than a winning baseline. Its value is that it moves along the compactness/stale-burden axis, while its k=16 state accuracy of 0.91 shows the unresolved reliability cost.
- Treat raw append as the canonical final-value-recoverable but stale-burdened baseline. It makes the slot-direct versus slot-prompt gap clearest.
- Treat heuristic CRUD as a partial-compaction baseline. It reduces memory size relative to raw append but does not solve stale same-slot interference.
- Defer external Mem0 feasibility until this figure/table is placed in the draft. The current result already supports the main diagnostic claim; an external baseline should be added only if the paper needs ecosystem grounding.

## Next writing decisions

- Decide the final paper terminology for `slot_direct` and `slot_prompt`.
- Decide whether to include constrained slot CRUD in the main plot legend or describe it separately as an upper-bound reference.
- Decide whether the final paper needs an external Mem0 row after the current tradeoff narrative is integrated.
