# P8.0 Release Candidate Checklist

## Purpose

This document records the current release-readiness state for MemUpdateBench as a controlled repeated same-slot update benchmark plus analysis package. It is intended for final paper/release preparation, not as a new experiment note.

## Current claim scope

Recommended paper framing:

> MemUpdateBench is a controlled diagnostic benchmark and empirical analysis of stale context contamination under repeated same-slot updates. It separates final-state reliability, stale same-slot burden, memory compactness, and answer robustness.

Avoid claiming:

- a broad all-purpose memory-agent benchmark;
- a fair leaderboard over external memory SDKs;
- a new deployable retrieval method;
- pure Long25 seed sensitivity.

## Core data files

### P6.3 hard split

```text
data/evomemory_update_frequency_hard_k1_p63_{dev,test}.json
data/evomemory_update_frequency_hard_k2_p63_{dev,test}.json
data/evomemory_update_frequency_hard_k4_p63_{dev,test}.json
data/evomemory_update_frequency_hard_k8_p63_{dev,test}.json
data/evomemory_update_frequency_hard_k16_p63_{dev,test}.json
```

### Expanded split

```text
data/evomemory_update_frequency_expanded_k1_p68_{dev,test}.json
data/evomemory_update_frequency_expanded_k2_p68_{dev,test}.json
data/evomemory_update_frequency_expanded_k4_p68_{dev,test}.json
data/evomemory_update_frequency_expanded_k8_p68_{dev,test}.json
data/evomemory_update_frequency_expanded_k16_p68_{dev,test}.json
```

## Core scripts

### Evaluation and smoke tests

```text
scripts/eval_evomemory.py
scripts/smoke_test.py
```

Important evaluation options:

```text
--answer_mode slot_direct|slot_prompt
--retrieval_policy normal|latest_per_slot
--context_order normal|chronological|reverse_chronological|current_first|current_last
--context_annotation none|timestamp|latest_outdated_label
--model_name <base answer model>
```

### P8.0 analysis scripts

```text
scripts/build_evidence_manifest.py
scripts/analyze_stale_dose_response.py
scripts/analyze_attribute_failures.py
scripts/summarize_context_mechanisms.py
scripts/run_synthetic_same_slot_probe.py
scripts/summarize_synthetic_same_slot_probe.py
scripts/run_lost_in_middle_probe.py
scripts/summarize_lost_in_middle_probe.py
scripts/package_p80_paper_tables.py
scripts/package_p80_figures.py
```

### Cluster runners

```text
scripts/run_p80_mechanism_probe_batch_sui3.sh
scripts/run_p80_llama_replication_sui3.sh
scripts/run_p80_llama_test_confirmation_sui3.sh
scripts/run_p80_third_model_replication_sui3.sh
scripts/run_p80_expanded_latest_per_slot_sui3.sh
```

## Core result artifacts

### Main P6.3 tradeoff results

```text
results/update_frequency_p63/
results/update_frequency_p63_summary/
```

### Stale intervention / latest-per-slot

```text
results/p70_stale_filter_intervention/
results/p70_stale_filter_intervention_summary/
results/p80_expanded_latest_per_slot_allk/
results/p80_expanded_latest_per_slot_summary/
```

### P8.0 mechanism analysis

```text
results/p80_stale_dose_response/
results/p80_mechanism_probes/
results/p80_mechanism_probe_summary/
results/p80_synthetic_same_slot_probe/
results/p80_synthetic_same_slot_probe_analysis/
results/p80_lost_in_middle_probe/
results/p80_lost_in_middle_probe_summary/
results/p80_attribute_error_analysis/
results/p81_synthetic_same_slot_probe_expanded/
results/p81_synthetic_same_slot_probe_expanded_analysis/
results/p81_same_as_current_failure_analysis/
results/p81_heuristic_threshold_k16_rigor/
results/p81_heuristic_threshold_k16_rigor_summary/
```

### Multi-model replication

```text
results/p80_multimodel_stale_susceptibility/llama31_8b/
results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.{json,csv,md}
results/p80_multimodel_stale_susceptibility/llama31_8b_test/
results/p80_multimodel_stale_susceptibility_summary/llama31_8b_test_context_summary.{json,csv,md}
results/p80_multimodel_stale_susceptibility/mistral7b/
results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.{json,csv,md}
results/p81_llama_constrained_zero_stale/
results/p82_mistral_constrained_zero_stale/
```

### Simple external pipeline baseline

```text
results/p80_simple_external_pipeline/
results/p80_simple_external_pipeline_summary/simple_external_pipeline_summary.{json,csv,md}
```

### Provenance index

```text
results/p80_evidence_manifest/evidence_manifest.{json,csv,md}
```

Current refreshed manifest count after P8/P8.1/P8.2 outputs: 402 result rows.

## Paper-facing notes

```text
paper/p70_stale_filter_intervention_note.md
paper/p70_stale_filter_allk_note.md
paper/p70_stale_filter_extension_note.md
paper/p70_long25_reproducibility_note.md
paper/p80_stale_dose_response_note.md
paper/p80_context_mechanism_probe_note.md
paper/p80_synthetic_same_slot_probe_note.md
paper/p80_lost_in_middle_probe_note.md
paper/p80_attribute_error_case_study.md
paper/p80_multimodel_stale_susceptibility_note.md
paper/p80_third_model_mistral_note.md
paper/p80_simple_external_pipeline_note.md
paper/p80_expanded_latest_per_slot_note.md
paper/p80_long25_training_provenance_audit.md
paper/p80_paper_tables.md
paper/p80_claim_evidence_matrix.md
paper/p80_canonical_main_number_ledger.md
paper/p80_remaining_work_summary.md
paper/manuscript_sections/p80_results_section_draft.md
```

## Generated figures

```text
paper/figures/p80_stale_dose_response.{png,pdf}
paper/figures/p80_synthetic_same_slot_matrix.{png,pdf}
paper/figures/p80_expanded_latest_per_slot.{png,pdf}
paper/figures/p80_llama_stale_susceptibility.{png,pdf}
paper/figures/p80_figure_manifest.json
```

Regenerate with:

```bash
PYTHONPATH=. python scripts/package_p80_figures.py
```

## Core regeneration commands

### Smoke and compile

```bash
PYTHONPATH=. python -m py_compile \
  scripts/eval_evomemory.py \
  scripts/smoke_test.py \
  scripts/build_evidence_manifest.py \
  scripts/analyze_stale_dose_response.py \
  scripts/analyze_attribute_failures.py \
  scripts/summarize_context_mechanisms.py \
  scripts/run_synthetic_same_slot_probe.py \
  scripts/summarize_synthetic_same_slot_probe.py \
  scripts/analyze_same_as_current_failures.py \
  scripts/summarize_heuristic_threshold.py \
  scripts/package_p80_paper_tables.py \
  scripts/package_p80_figures.py
PYTHONPATH=. python scripts/smoke_test.py
```

### Evidence manifest

```bash
PYTHONPATH=. python scripts/build_evidence_manifest.py \
  --result_root results \
  --output_dir results/p80_evidence_manifest
```

### P8 tables and figures

```bash
PYTHONPATH=. python scripts/package_p80_paper_tables.py
PYTHONPATH=. python scripts/package_p80_figures.py
```

### Dose-response analysis

```bash
PYTHONPATH=. python scripts/analyze_stale_dose_response.py
```

### Synthetic same-slot summary

```bash
PYTHONPATH=. python scripts/summarize_synthetic_same_slot_probe.py
```

## Release blockers

### Still open

1. A fair external SDK row is still missing, but a transparent simple extract-then-store external pipeline baseline is now complete.
2. README/release instructions still need final outside-user polishing and a result-artifact packaging decision.
3. A compilable LaTeX production draft now exists at `paper/manuscript_production_draft.tex` with generated PDF `paper/manuscript_production_draft.pdf`; remaining production work is replacing citation placeholders with formal BibTeX, venue-template conversion, and final advisor-facing polishing.

### Closed since the previous checklist

1. Llama3.1-8B test-split confirmation is complete and integrated into the P8 notes, ledger, remaining-work summary, and evidence manifest.
2. Mistral-7B third-model dev replication is complete and integrated into the multi-model note, Mistral note, ledger, table pack, and evidence manifest.
3. Simple extract-then-store external pipeline baseline is complete and integrated into the baseline note, table pack, and evidence manifest.
4. Company/language gold-retrieved attribute error case study is complete and integrated into the attribute note.
5. Lost-in-the-Middle gold-position probe is complete and integrated into the probe note, ledger, table pack, and evidence manifest.
6. P8.1 expanded synthetic diagnostic subset is complete at 64 examples per selected cell and integrated into the synthetic probe note and ledger.
7. P8.1 same_as_current EM-failure / answer-value-present inspection is complete and integrated into the synthetic probe note.
8. P8.1 Llama constrained CRUD k=16 zero-stale baseline is complete and integrated into the multi-model note and ledger.
9. P8.1 fixed-k heuristic threshold dose-response sweep is complete and integrated into the dose-response note and ledger.
10. P8.2 Mistral constrained CRUD k=16 zero-stale baseline is complete and closes the three-model ceiling-recovery story: Mistral latest_per_slot EM/F1 0.720/0.735 exactly matches constrained zero-stale EM/F1 0.720/0.735.

### Not blockers for controlled diagnostic paper

1. Mem0 text-backend failure, if reported honestly as an external-baseline limitation.
2. k=32 remaining appendix-only.
3. No repair training.
4. No implicit/negative/conditional update split in the main benchmark.

## Final validation status

Last local validation after the manuscript production pass:

```text
PYTHONPATH=. python scripts/package_p80_paper_tables.py
PYTHONPATH=. python scripts/package_p80_figures.py
PYTHONPATH=. python -m py_compile ...
PYTHONPATH=. python scripts/smoke_test.py
SMOKE TEST: 26/26 passed
PYTHONPATH=. python scripts/build_evidence_manifest.py --result_root results --output_dir results/p80_evidence_manifest
num_rows: 402
```
