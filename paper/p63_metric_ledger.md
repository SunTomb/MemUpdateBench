# P6.3 Canonical Metric Ledger

## Source of truth

All paper-facing P6.3 headline numbers should trace back to:

```text
results/update_frequency_p63_summary/update_frequency_summary.json
results/update_frequency_p63_summary/update_frequency_summary.csv
results/update_frequency_p63_summary/update_frequency_tables.md
```

These summary artifacts are regenerated from:

```text
results/update_frequency_p63/*/evomemory_results.json
```

Canonical regeneration command:

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
python scripts/package_update_frequency_paper_assets.py \
  --summary_json results/update_frequency_p63_summary/update_frequency_summary.json \
  --paper_dir paper
```

## Expected P6.3 matrix

The P6.3 hard update-frequency matrix is complete for the main paper cells:

- Methods: `constrained_slot_crud`, `raw_add`, `heuristic_crud`, `long25`
- Answer modes: `slot_direct`, `slot_prompt`
- k values: `1, 2, 4, 8, 16`

The summary loader reports 45 total result files because it includes additional result cells beyond the 4 x 2 x 5 main matrix.

## k=16 thesis table

| Method | slot-direct state acc. | slot-prompt EM/F1 | stale same-slot | final memory size | Paper role |
| --- | ---: | ---: | ---: | ---: | --- |
| Constrained CRUD | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 | Oracle-like diagnostic upper bound |
| Raw append | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 | Final-value recoverable but stale-burdened append-only baseline |
| Heuristic CRUD | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 | Partial-compaction baseline |
| Long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 | Learned compact manager with imperfect reliability |

## Slot-direct state accuracy sweep

| Method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Raw append | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Heuristic CRUD | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Long25 | 1.00 | 0.89 | 0.92 | 0.85 | 0.91 |

## Slot-prompt EM/F1 sweep

| Method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | 0.90 / 0.91 | 0.90 / 0.94 | 0.86 / 0.86 | 0.98 / 0.98 | 0.70 / 0.70 |
| Raw append | 0.90 / 0.91 | 0.78 / 0.79 | 0.24 / 0.27 | 0.06 / 0.10 | 0.07 / 0.10 |
| Heuristic CRUD | 0.89 / 0.91 | 0.80 / 0.81 | 0.28 / 0.28 | 0.19 / 0.22 | 0.10 / 0.13 |
| Long25 | 1.00 / 1.00 | 0.79 / 0.79 | 0.76 / 0.77 | 0.73 / 0.74 | 0.48 / 0.53 |

## Stale same-slot burden sweep

| Method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Raw append | 0.00 | 1.00 | 3.00 | 6.75 | 14.25 |
| Heuristic CRUD | 0.00 | 1.00 | 2.55 | 5.23 | 7.44 |
| Long25 | 0.00 | 0.23 | 0.46 | 0.74 | 1.13 |

## Final memory size sweep

| Method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | 2.00 | 4.00 | 7.00 | 12.00 | 23.00 |
| Raw append | 2.00 | 5.00 | 12.00 | 25.00 | 52.00 |
| Heuristic CRUD | 1.50 | 4.97 | 10.51 | 18.25 | 26.73 |
| Long25 | 1.46 | 3.45 | 4.34 | 6.18 | 9.43 |

## Citation rules

- Cite `slot_direct` only as oracle-like slot-state evaluation or exact slot lookup.
- Cite `slot_prompt` as the realistic answer robustness metric.
- Do not describe Long25 as winning overall; describe it as compact but imperfectly reliable.
- Do not describe constrained CRUD as a deployable external-memory system; it is a diagnostic upper bound.
- Do not cite Mem0/Letta/MemGPT as evaluated baselines unless a future isolated feasibility run is actually performed.
