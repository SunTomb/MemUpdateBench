# P6.7 Consistency Audit

## Audit scope

Files audited for the P6.3 canonical results and P6.7 manuscript package:

```text
README.md
WORKFLOW.md
results/update_frequency_p63_summary/update_frequency_tables.md
paper/p63_metric_ledger.md
paper/p63_update_frequency_assets.md
paper/p63_update_frequency_latex_snippets.tex
paper/p63_experimental_section_draft.md
paper/p63_handoff_summary.md
paper/external_baseline_feasibility_note.md
paper/p63_claim_evidence_matrix.md
paper/p63_error_analysis_k16.md
paper/remote_verification_log.md
paper/manuscript_draft.md
paper/manuscript_sections/*.md
```

## Numeric consistency

Canonical k=16 values are consistent across the paper-facing assets:

| Method | State acc. | EM/F1 | Stale | Mem. size | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Constrained CRUD | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 | OK |
| Raw append | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 | OK |
| Heuristic CRUD | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 | OK |
| Long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 | OK |

## Terminology audit

| Concept | Preferred paper term | Internal name | Status |
| --- | --- | --- | --- |
| Oracle state evaluation | oracle-like slot-state evaluation / exact slot lookup | `slot_direct` | OK |
| Prompted answer evaluation | slot-conditioned answering | `slot_prompt` | OK |
| Upper-bound CRUD | Constrained CRUD | `constrained_slot_crud` | OK |
| Append-only baseline | Raw append | `raw_add` | OK |
| Partial compaction baseline | Heuristic CRUD | `heuristic_crud` | OK |
| Learned compact manager | Long25 | checkpoint `long25` | OK |

## Claim audit

| Claim risk | Current status | Required wording guard |
| --- | --- | --- |
| Constrained CRUD misread as deployable method | Controlled | Always call it oracle-like diagnostic upper bound. |
| Long25 misread as winner | Controlled | Describe as compact but imperfectly reliable. |
| `slot_direct` misread as realistic answer mode | Controlled | Describe as diagnostic control. |
| Mem0/Letta/MemGPT misread as evaluated | Controlled | Say no external row yet; isolated feasibility only if needed. |
| k=32 absence seen as missing | Controlled | Say k=16 already gives decisive separation; revisit only if needed. |
| Repair training implied | Controlled | State repair training is future method work. |

## Source-of-truth chain

```text
results/update_frequency_p63/*/evomemory_results.json
  -> scripts/summarize_update_frequency.py
  -> results/update_frequency_p63_summary/update_frequency_summary.{csv,json}
  -> scripts/package_update_frequency_paper_assets.py
  -> paper/*
```

## Open issues

No unresolved numeric mismatches are known after P6.7 manuscript and support-note alignment.

Remaining manual review items:

1. Check final paper template fit for the main 2x2 figure.
2. Decide whether appendix derived figures are all needed or whether some should remain internal.
3. Confirm final venue terminology for `slot_direct` and `slot_prompt`.
4. Ensure the remote verification note stays framed as close confirmation rather than alternate headline numbers.
5. Keep the external-baseline trigger explicit: defer by default, isolated Mem0 only if manuscript positioning later demands it.
