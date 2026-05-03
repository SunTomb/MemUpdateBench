# P6.3 Update-Frequency Paper Assets

## Main figure

- `p63_update_frequency_tradeoff.png`
- `p63_update_frequency_tradeoff.pdf`
- `p63_update_frequency_latex_snippets.tex`
- `p63_gap_slot_direct_vs_prompt.png` / `.pdf`
- `p63_stale_vs_prompt_em_k16.png` / `.pdf`
- `p63_memory_size_vs_prompt_em_k16.png` / `.pdf`

Suggested caption: Update-frequency stress test on P6.3 hard splits. Oracle-like slot-direct state evaluation hides stale-memory burden for append-only and heuristic methods, while slot-prompt answering reveals answer collapse as repeated same-slot updates accumulate stale entries. Long25 is substantially more compact but sacrifices final-state reliability under hard distractors.

## Main body table: k=16 tradeoff

| method | k=16 slot_direct state_acc | k=16 slot_prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| raw_add | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| heuristic_crud | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

Suggested caption: At k=16, append-only and heuristic managers preserve final-state recoverability under slot-direct evaluation but collapse under slot-prompt answering because stale same-slot entries remain in memory. Long25 reduces stale burden and memory size, but its final-state accuracy is below the constrained CRUD upper bound.

## Experimental narrative bullets

- The constrained slot CRUD upper-bound reaches k=16 slot-direct state accuracy 1.00 with no stale same-slot burden (0.00) and slot-prompt EM/F1 0.70 / 0.70.
- Raw append preserves k=16 slot-direct recoverability but retains 14.25 stale same-slot entries, grows to final memory size 52.00, and drops to slot-prompt EM/F1 0.07 / 0.10.
- Heuristic CRUD partially limits memory growth but still leaves 7.44 stale same-slot entries at k=16 and reaches only slot-prompt EM/F1 0.10 / 0.13.
- Long25 is much more compact at k=16, with stale same-slot 1.13 and final memory size 9.43, but its slot-direct state accuracy is 0.91 rather than 1.00.
- The central result is therefore a tradeoff curve: recoverability, stale burden, compactness, and answer robustness move differently as repeated updates increase.
- These results support the P6.7 manuscript package; external baseline feasibility should be considered only if the integrated draft still needs ecosystem grounding.

## Appendix table: slot-direct state accuracy

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| raw_add | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| heuristic_crud | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| long25 | 1.00 | 0.89 | 0.92 | 0.85 | 0.91 |

## Appendix table: slot-prompt EM/F1

| method | k=1 EM/F1 | k=2 EM/F1 | k=4 EM/F1 | k=8 EM/F1 | k=16 EM/F1 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 0.90 / 0.91 | 0.90 / 0.94 | 0.86 / 0.86 | 0.98 / 0.98 | 0.70 / 0.70 |
| raw_add | 0.90 / 0.91 | 0.78 / 0.79 | 0.24 / 0.27 | 0.06 / 0.10 | 0.07 / 0.10 |
| heuristic_crud | 0.89 / 0.91 | 0.80 / 0.81 | 0.28 / 0.28 | 0.19 / 0.22 | 0.10 / 0.13 |
| long25 | 1.00 / 1.00 | 0.79 / 0.79 | 0.76 / 0.77 | 0.73 / 0.74 | 0.48 / 0.53 |

## Appendix table: stale same-slot entries

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| raw_add | 0.00 | 1.00 | 3.00 | 6.75 | 14.25 |
| heuristic_crud | 0.00 | 1.00 | 2.55 | 5.23 | 7.44 |
| long25 | 0.00 | 0.23 | 0.46 | 0.74 | 1.13 |

## Appendix table: final memory size

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 2.00 | 4.00 | 7.00 | 12.00 | 23.00 |
| raw_add | 2.00 | 5.00 | 12.00 | 25.00 | 52.00 |
| heuristic_crud | 1.50 | 4.97 | 10.51 | 18.25 | 26.73 |
| long25 | 1.46 | 3.45 | 4.34 | 6.18 | 9.43 |
