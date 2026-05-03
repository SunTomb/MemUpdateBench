# P6.7 Artifact Release Checklist

## Data files

Main P6.3 hard update-frequency files:

```text
data/evomemory_update_frequency_hard_k1_p63_dev.json
data/evomemory_update_frequency_hard_k1_p63_test.json
data/evomemory_update_frequency_hard_k2_p63_dev.json
data/evomemory_update_frequency_hard_k2_p63_test.json
data/evomemory_update_frequency_hard_k4_p63_dev.json
data/evomemory_update_frequency_hard_k4_p63_test.json
data/evomemory_update_frequency_hard_k8_p63_dev.json
data/evomemory_update_frequency_hard_k8_p63_test.json
data/evomemory_update_frequency_hard_k16_p63_dev.json
data/evomemory_update_frequency_hard_k16_p63_test.json
data/evomemory_update_frequency_hard_p63_train.json
data/evomemory_update_frequency_hard_p63_dev.json
data/evomemory_update_frequency_hard_p63_test.json
```

## Result roots

```text
results/update_frequency_p63/
results/update_frequency_p63_summary/
```

## Core scripts

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/summarize_update_frequency.py
scripts/package_update_frequency_paper_assets.py
scripts/smoke_test.py
scripts/generate_constrained_sft.py
scripts/train_constrained_sft.py
```

## Paper artifacts

```text
paper/p63_update_frequency_tradeoff.png
paper/p63_update_frequency_tradeoff.pdf
paper/p63_gap_slot_direct_vs_prompt.png
paper/p63_gap_slot_direct_vs_prompt.pdf
paper/p63_stale_vs_prompt_em_k16.png
paper/p63_stale_vs_prompt_em_k16.pdf
paper/p63_memory_size_vs_prompt_em_k16.png
paper/p63_memory_size_vs_prompt_em_k16.pdf
paper/p63_update_frequency_assets.md
paper/p63_update_frequency_latex_snippets.tex
paper/p63_metric_ledger.md
paper/p63_claim_evidence_matrix.md
paper/p63_consistency_audit.md
paper/p63_diagnostic_ablation_plan.md
paper/p63_reviewer_risk_matrix.md
paper/p63_artifact_release_checklist.md
paper/p63_handoff_summary.md
paper/p63_error_analysis_k16.md
paper/remote_verification_log.md
paper/figure_table_placement_plan.md
paper/manuscript_draft.md
paper/external_baseline_feasibility_note.md
paper/mem0_isolated_feasibility_plan.md
paper/manuscript_sections/
```

## Regeneration commands

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

## Checkpoint assumptions

The learned Long25 checkpoint should live at:

```text
checkpoints/long25/best
```

Local checkout may not include the checkpoint. Cluster checkout is expected to have or receive it under:

```text
/NAS/yesh/MemUpdateBench/checkpoints/long25/best
```

Do not delete archived G-MSRA paths until checkpoint migration has been verified.

## Release boundaries

This release scope is:

- repeated same-slot update-frequency benchmark,
- deterministic slot diagnostics,
- P6.3 hard k=1/2/4/8/16 result summaries,
- P6.5/P6.6 paper-facing assets,
- P6.7 manuscript assembly, cluster-backed verification, k=16 error-analysis support notes, and advisor-ready handoff package.

This release scope is not:

- old G-MSRA Phase 1-5 training stack,
- repair training,
- k=32 expansion,
- Mem0/Letta/MemGPT integration,
- a broad agent-memory leaderboard.

## Pre-release checks

- [ ] `python scripts/smoke_test.py` passes.
- [ ] `scripts/package_update_frequency_paper_assets.py` regenerates all paper artifacts.
- [ ] Main k=16 numbers match `paper/p63_metric_ledger.md`.
- [ ] Remote verification is cited only as close confirmation, not as replacement headline numbers.
- [ ] README commands match actual script names.
- [ ] WORKFLOW.md records latest generated artifacts.
- [ ] `paper/manuscript_draft.md`, `paper/p63_claim_evidence_matrix.md`, and `paper/p63_consistency_audit.md` agree on terminology and claim boundaries.
- [ ] No external-baseline claims appear unless an actual isolated run is performed.
- [ ] No old G-MSRA Phase 1-5 modules are restored into mainline.
- [ ] Repair training remains outside the paper-package release scope.
