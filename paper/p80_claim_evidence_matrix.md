# P8.0 Claim-Evidence Matrix

## Purpose

This matrix maps the current benchmark + analysis paper claims to the concrete scripts, result artifacts, notes, and caveats that support them.

## Claim 1: State accuracy does not imply answer robustness

**Evidence:**

- P6.3 hard k=16 Raw append and Constrained CRUD retain state accuracy 1.00 under slot-direct but are lower under slot-prompt.
- Constrained CRUD k=16 has zero stale same-slot burden but slot-prompt EM/F1 around 0.70/0.70.

**Artifacts:**

```text
results/update_frequency_p63/
results/update_frequency_p63_summary/
paper/p63_error_analysis_k16.md
paper/manuscript_draft.md
```

**Caveat:** `slot_direct` is oracle-like and should be presented as a diagnostic, not as end-user answering.

## Claim 2: Stale same-slot context is a major causal driver of raw append collapse

**Evidence:**

- Raw append k=16 dev normal top-k5: EM/F1 0.140/0.173.
- Latest-per-slot retrieval rewrite: EM/F1 0.690/0.703 while writes, memory size, and stored stale burden remain unchanged.
- Expanded latest-per-slot k=16 dev: EM/F1 0.750/0.764 with stale retrieved 0.000.

**Artifacts:**

```text
scripts/eval_evomemory.py
scripts/summarize_stale_filter_intervention.py
scripts/run_p80_expanded_latest_per_slot_sui3.sh
results/p70_stale_filter_intervention/
results/p70_stale_filter_intervention_summary/
results/p80_expanded_latest_per_slot_allk/
results/p80_expanded_latest_per_slot_summary/
paper/p70_stale_filter_intervention_note.md
paper/p80_expanded_latest_per_slot_note.md
```

**Caveat:** latest_per_slot is an answer-time retrieval rewrite over the full store, not a pure filter of the original top-k context and not a proposed deployable method.

## Claim 3: Stale burden has a dose-response relationship with answer collapse

**Evidence:**

- Stored stale count 0 -> EM 0.967.
- Stored stale count 1 -> EM 0.743.
- Stored stale count 3 -> EM 0.290.
- Logistic ED50: stored stale 3.18; retrieved stale 1.89.

**Artifacts:**

```text
scripts/analyze_stale_dose_response.py
results/p80_stale_dose_response/
paper/p80_stale_dose_response_note.md
paper/figures/p80_stale_dose_response.{png,pdf}
```

**Caveat:** first-pass lightweight fit, pooled across existing runs; use as mechanism evidence, not final statistical modeling.

## Claim 4: Collapse is not a simple majority-vote-only effect

**Evidence:**

- Synthetic conflict + chronological order remains robust even when many stale values precede the current value.
- Synthetic conflict + reverse chronological order collapses immediately without labels.
- Latest/outdated labels nearly repair conflict-driven collapse.
- same_as_current repetition still hurts exact EM while answer-value-present remains high.

**Artifacts:**

```text
scripts/run_synthetic_same_slot_probe.py
scripts/summarize_synthetic_same_slot_probe.py
results/p80_synthetic_same_slot_probe/
results/p80_synthetic_same_slot_probe_analysis/
paper/p80_synthetic_same_slot_probe_note.md
paper/figures/p80_synthetic_same_slot_matrix.{png,pdf}
```

**Caveat:** synthetic examples are cleaner than real benchmark retrieval contexts; use as mechanism isolation rather than benchmark performance.

## Claim 5: Context presentation affects answer-layer behavior even with retrieval composition fixed

**Evidence:**

- Real-context raw_add k=16 dev has fixed gold retrieval 0.360 and stale retrieved avg 4.040 across formal conditions.
- EM changes from normal 0.110 to chronological 0.230, reverse chronological 0.010, latest/outdated label 0.260.

**Artifacts:**

```text
scripts/run_p80_mechanism_probe_batch_sui3.sh
scripts/summarize_context_mechanisms.py
results/p80_mechanism_probes/
results/p80_mechanism_probe_summary/
paper/p80_context_mechanism_probe_note.md
```

**Caveat:** gold retrieval remains low in real contexts; labels help answer-layer use of retrieved context but do not solve latest/gold retrieval bottlenecks.

## Claim 6: Stale filtering recovers each tested model to its own zero-stale ceiling

**Evidence:**

- Qwen k=16 dev raw_add normal EM 0.110, latest_per_slot EM 0.690, constrained CRUD zero-stale ceiling around 0.700.
- Llama3.1-8B k=16 dev raw_add normal EM/F1 0.060/0.062, latest_per_slot EM/F1 0.290/0.341, constrained CRUD zero-stale EM/F1 0.270/0.321.
- Mistral-7B k=16 dev raw_add normal EM/F1 0.080/0.177, latest_per_slot EM/F1 0.720/0.735, constrained CRUD zero-stale EM/F1 0.720/0.735.
- In all three models, latest_per_slot removes stale retrieved entries and recovers to approximately the model's own zero-stale slot-prompt ceiling.

**Artifacts:**

```text
scripts/run_p80_llama_replication_sui3.sh
scripts/run_p80_third_model_replication_sui3.sh
scripts/run_p81_llama_constrained_zero_stale_sui3.sh
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p80_multimodel_stale_susceptibility/llama31_8b/
results/p80_multimodel_stale_susceptibility/mistral7b/
results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.{json,csv,md}
results/p80_multimodel_stale_susceptibility_summary/mistral7b_context_summary.{json,csv,md}
results/p81_llama_constrained_zero_stale/
results/p82_mistral_constrained_zero_stale/
paper/p80_multimodel_stale_susceptibility_note.md
paper/p80_third_model_mistral_note.md
paper/p80_canonical_main_number_ledger.md
```

**Caveat:** Qwen and Mistral ceiling-recovery evidence is dev-only; Llama has dev and test stale-susceptibility confirmation, but the zero-stale control is dev. Present this as diagnostic multi-model evidence, not a full leaderboard.

## Claim 7: The P6.3 latest_per_slot k=8 anomaly should not be overclaimed

**Evidence:**

- P6.3 latest_per_slot k=8 dev EM 0.990.
- Expanded latest_per_slot k=8 dev EM 0.925.
- Expanded k=16 remains lower at EM 0.750 with stale retrieved 0.000 and gold retrieved 0.860.

**Artifacts:**

```text
scripts/run_p80_expanded_latest_per_slot_sui3.sh
results/p80_expanded_latest_per_slot_allk/
results/p80_expanded_latest_per_slot_summary/
paper/p80_expanded_latest_per_slot_note.md
paper/figures/p80_expanded_latest_per_slot.{png,pdf}
```

**Caveat:** expanded dev only; use to attenuate overclaiming, not as a final test-set table.

## Claim 8: Long25 should be reported with checkpoint provenance, not as pure seed sensitivity

**Evidence:**

- P6.3 original checkpoint and P6.5 seed family use same k=16 test split but different checkpoint paths.
- Existing artifacts do not prove matched training commands with only seed changed.

**Artifacts:**

```text
results/p70_long25_reproducibility/long25_provenance.{json,csv}
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
paper/p70_long25_reproducibility_note.md
paper/p80_long25_training_provenance_audit.md
```

**Caveat:** call this checkpoint/training-provenance sensitivity unless a future matched retraining sweep fixes all variables except seed.

## Claim 9: External baseline remains the main external-validity gap

**Evidence:**

- Mem0 Qwen2.5-VL dev20 is runnable but unfair/qualitative.
- Qwen2.5-7B text backend is queryable and CPU MiniLM works.
- Mem0 structured extraction parser fails before dev3 completes.

**Artifacts:**

```text
scripts/serve_openai_compatible_transformers.py
paper/p70_external_baseline_fairness_note.md
paper/p70_external_baseline_text_backend_probe.md
```

**Caveat:** do not put Mem0 in main leaderboard unless extraction compatibility is fixed and a fair text-backed run completes.

## Claim 10: Order- and metadata-sensitive version arbitration is answer-model dependent

**Evidence:**

- The completed Qwen3.5-9B frozen Raw Append full matrix has 1,440/1,440 format-valid rows, 1,437/1,440 exact matches, EM 0.9979167, and three stale copies.
- Eight of nine condition-by-k cells are perfect. The lone non-perfect cell, chronological/no-label k=4, has EM 0.98125 and clustered semantic-core bootstrap CI [0.94375, 1.00000]. Reverse/no-label is perfect at k=4/8/16, as are all reverse-labeled cells.
- The three errors occur in one semantic core (`core_c40d565fabd02e01`), with core EM 0.625. In two surface tasks the model copies `ALPHA-01` rather than gold `CORE-04` although the retrieved sequence ends in the gold value (`GAMMA-01`, `CORE-15`, `ALPHA-01`, `CORE-04`).
- The paired core contrast, reverse/no-label minus chronological/no-label at k=4, is +0.01875 with CI [0.00000, 0.05625]. This does not reproduce the earlier reverse-order high-k collapse.

**Artifacts:**

```text
results/post_core_qwen35_full_matrix_analysis/analysis.json
results/post_core_qwen35_full_matrix_analysis/analysis.sha256
results/post_core_qwen35_full_matrix_analysis/qwen35_full_matrix_paper_table.md
paper/p80_qwen35_full_matrix_note.md
paper/p80_canonical_main_number_ledger.md
```

**Caveat:** this is bounded answer-model evidence over frozen Raw Append Family-A contexts, not evidence that an external-memory system is broadly robust. Qwen3.5-9B's result defines a model-dependent boundary on the version-arbitration mechanism; it does not invalidate the earlier controlled probes or establish universality for either pattern.

## Claim 11: Capability smoke establishes interface scope, not benchmark performance

**Evidence:**

- The corrected formal BASE smoke has 56 attempts: Qwen3.5-9B 8/8, GPT-5.5 route 8/8, Grok-4.5 route 8/8, Gemini 3.6 Flash 7/8, Claude Sonnet 4.6 4/8, Claude Opus 4.8 0/8 (HTTP 400), and Muse Glimmer GGUF 0/8 (parser mismatch).
- Sixty-four wrong-provider diagnostic HTTP 404 calls are separately accounted for and excluded from the formal outcome; the campaign has 40 formal provider calls and 104 total real provider calls.

**Artifacts:**

```text
D:/USTC/2026Winter/MemUpdateBench_qualification_inputs/phase12_0981a38_v1/post_core_capability_smoke_base_20260826_v2.json
paper/p80_qwen35_full_matrix_note.md
```

**Caveat:** the smoke contains zero canary and zero benchmark generations, so it is operational qualification rather than scientific evidence. Route success does not authenticate mutable upstream identities: GPT-5.5 and Grok-4.5 retain their identity caveats; document-verified Claude identities are not promoted by a partial or failed route result.
