# Main Results Draft

## Repeated updates separate recoverability from answer robustness

Figure~\\ref{fig:p63-update-frequency-tradeoff} summarizes the P6.3 hard update-frequency results. The first panel shows that Constrained CRUD, Raw append, and Heuristic CRUD all maintain slot-direct state accuracy 1.00 through k=16. Under oracle-like slot-state evaluation, the final value remains recoverable for these methods even after many repeated updates.

The slot-prompt panel tells a different story. Raw append falls from 0.90 / 0.91 EM/F1 at k=1 to 0.07 / 0.10 at k=16. Heuristic CRUD follows the same qualitative pattern, reaching 0.10 / 0.13 at k=16. These methods can preserve the final value under exact slot lookup, but stale same-slot evidence remains in memory and causes answer-layer collapse.

## Stale same-slot burden explains the collapse

The stale-burden and memory-size panels show why append-only methods degrade under slot-conditioned answering. Raw append retains 14.25 stale same-slot entries at k=16 and grows to final memory size 52.00. Heuristic CRUD is more compact but still leaves 7.44 stale same-slot entries and final memory size 26.73. The answer layer is therefore exposed to many obsolete values for the same entity and attribute.

This pattern shows why final-state recoverability alone is insufficient. A memory system can contain the correct final value while still exposing enough stale same-slot evidence to make answer generation unreliable.

## Compact learned memory trades stale burden for reliability

Long25 occupies a different point on the curve. At k=16, it reduces stale same-slot burden to 1.13 and final memory size to 9.43. Its slot-prompt EM/F1, 0.48 / 0.53, is substantially above Raw append and Heuristic CRUD at the same update frequency. However, compactness comes with imperfect state tracking: Long25 slot-direct state accuracy is 0.91 rather than 1.00.

Thus Long25 should not be presented as simply better or worse. It demonstrates that learned compact managers can reduce stale burden, but may miss final updates under hard distractors.

## Oracle-like constrained CRUD is an upper-bound diagnostic

Constrained CRUD has no stale same-slot entries and reaches slot-direct state accuracy 1.00 at every k. At k=16, it still obtains only 0.70 / 0.70 under slot-prompt answering. State-level analysis finds 100 / 100 correct state predictions and no state errors, so the remaining gap should be interpreted as answer-layer or prompt-conditioned failure rather than stale-state retention.

Long25's residual k=16 failures look different. The canonical local analysis finds 9 `wrong_value` state errors and no wrong-entity or wrong-attribute state errors. A remote Sui-3-Wu rerun provides close confirmation with state accuracy 0.92, slot-prompt EM/F1 0.49 / 0.5467, and 8 `wrong_value` errors concentrated in `company` slots. This supports a compactness/reliability interpretation rather than a simple learned-manager win.

Table~\\ref{tab:p63-k16-tradeoff} highlights the high-frequency endpoint. The main result is not a single best method, but a tradeoff: append-only methods preserve recoverability but accumulate stale burden, heuristic compaction only partially helps, and learned compact memory reduces stale burden at the cost of final-state reliability.
