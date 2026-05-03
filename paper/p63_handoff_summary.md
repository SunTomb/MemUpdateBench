# P6.7 Handoff Summary

## Main claim

- Repeated same-slot updates expose a tradeoff among final-state reliability, stale same-slot burden, memory compactness, and slot-conditioned answer robustness.
- Append-only and heuristic methods can preserve the final value under oracle slot-state evaluation, but they retain stale same-slot entries.
- Under slot-conditioned answering, stale same-slot burden causes severe answer collapse for raw append and heuristic CRUD.
- Long25 is much more compact and retains fewer stale same-slot entries, but it misses some final updates under hard distractors.
- Constrained CRUD should be read as an oracle-like diagnostic upper bound, not as a deployable external-memory system.

## Files to inspect

```text
paper/manuscript_draft.md
paper/p63_update_frequency_tradeoff.pdf
paper/p63_update_frequency_latex_snippets.tex
paper/p63_metric_ledger.md
paper/p63_error_analysis_k16.md
paper/remote_verification_log.md
paper/external_baseline_feasibility_note.md
```

## k=16 headline numbers

| method | slot-direct state acc. | slot-prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| Constrained CRUD | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| Raw append | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| Heuristic CRUD | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| Long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

## How to regenerate

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
python scripts/package_update_frequency_paper_assets.py \
  --summary_json results/update_frequency_p63_summary/update_frequency_summary.json \
  --paper_dir paper
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py scripts/package_update_frequency_paper_assets.py
python scripts/smoke_test.py
```

## What this does not claim

- It does not claim Long25 is a universal winner; it is a compactness/reliability tradeoff point.
- It does not claim oracle slot-state evaluation is sufficient; the gap to slot-conditioned answering is the core result.
- It does not add k=32; k=16 already shows decisive separation.
- It does not start repair training.
- It does not integrate Mem0/Letta/MemGPT into the main environment.

## External baseline recommendation

Default decision: defer external baseline implementation.

Add an isolated Mem0 feasibility run only if the manuscript needs a recognizable external-memory framework for positioning. If that becomes necessary, keep it isolated, start with P6.3 hard k=16 only, and require inspectable memory entries so stale same-slot burden can be measured.

## Remote verification note

P6.7 added cluster-backed support for the manuscript package:

- `Tang-2-Wu` passed the constrained CRUD k=16 remote slot-direct sanity check.
- `Sui-3-Wu` completed Long25 k=16 slot-direct and slot-prompt reruns.
- The remote Long25 rerun is a close confirmation of the canonical ledger, not a replacement for it: the rerun slightly shifts to state accuracy 0.92 and slot-prompt EM/F1 0.49 / 0.5467 while preserving the same paper-level conclusion.
- Remote state errors remain dominated by `wrong_value` failures concentrated in `company` slots.

## Recommended next action

Use `paper/manuscript_draft.md` as the primary reading target, with `paper/p63_metric_ledger.md`, `paper/p63_error_analysis_k16.md`, and `paper/remote_verification_log.md` as supporting evidence for results, diagnostics, and reproducibility.
