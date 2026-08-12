# MemUpdateBench Workflow

This project starts from the G-MSRA P6.x line. Historical details are in `docs/PROJECT_WORKFLOW6.*.md`.

Current mainline: repeated same-slot update-frequency stress tests for external memory systems.

Current strategic direction after `docs/critical_review.md`: stop treating paper packaging as the default path. The project must now prioritize external validity and diagnostic depth: real external baselines, deeper answer-layer failure analysis, same-method-family tradeoff curves, larger/more diverse data, and serious related-work positioning. The honest near-term target is to turn a workshop-level controlled diagnostic into a stronger Findings-level empirical paper.

## P6.5 Paper Asset Packaging

### Motivation

P6.3/P6.4 established the main update-frequency tradeoff, but the existing outputs were diagnostic summaries under `results/update_frequency_p63_summary/`. P6.5 packages those results into paper-facing assets before starting any repair training or external-baseline work.

### Files generated

```text
scripts/package_update_frequency_paper_assets.py
paper/p63_update_frequency_tradeoff.png
paper/p63_update_frequency_tradeoff.pdf
paper/p63_update_frequency_assets.md
paper/p63_update_frequency_latex_snippets.tex
paper/p63_experimental_section_draft.md
paper/external_baseline_feasibility_note.md
paper/p63_handoff_summary.md
```

### Commands run

```bash
python scripts/package_update_frequency_paper_assets.py \
  --summary_json results/update_frequency_p63_summary/update_frequency_summary.json \
  --paper_dir paper
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py scripts/package_update_frequency_paper_assets.py
python scripts/smoke_test.py
```

### Validation

```text
Loaded 45 summary rows
Wrote paper/p63_update_frequency_tradeoff.png
Wrote paper/p63_update_frequency_tradeoff.pdf
Wrote paper/p63_update_frequency_assets.md
Wrote paper/p63_update_frequency_latex_snippets.tex
SMOKE TEST: 18/18 passed
```

### Paper-facing assets

`paper/p63_update_frequency_tradeoff.{png,pdf}` is a 2x2 figure covering:

1. slot-direct state accuracy,
2. slot-prompt exact match,
3. stale same-slot entries,
4. final memory size.

`paper/p63_update_frequency_assets.md` contains:

- the k=16 thesis table,
- appendix-ready k-sweep tables,
- figure/table caption drafts,
- experimental narrative bullets,
- LaTeX-ready main figure/table snippets and appendix k-sweep tables,
- a paper-facing experimental section draft with recommended baseline framing,
- an external-baseline feasibility note recommending no immediate external row unless the draft needs ecosystem grounding,
- a concise handoff summary for advisor/reviewer review.

### Conclusion

The packaged assets support the paper framing: repeated same-slot updates reveal a tradeoff among final-state recoverability, stale burden, compactness, and slot-conditioned answer robustness. Append-only and heuristic methods remain recoverable under `slot_direct` but collapse under `slot_prompt`; long25 is compact but not fully reliable; constrained slot CRUD serves as an upper-bound diagnostic.

### P6.6 paper-integration expansion

Additional paper-integration artifacts were added after the initial P6.5 packaging:

```text
paper/p63_metric_ledger.md
paper/p63_claim_evidence_matrix.md
paper/p63_method_definition_table.md
paper/p63_consistency_audit.md
paper/p63_diagnostic_ablation_plan.md
paper/p63_reviewer_risk_matrix.md
paper/p63_artifact_release_checklist.md
paper/mem0_isolated_feasibility_plan.md
paper/manuscript_sections/abstract_draft.md
paper/manuscript_sections/introduction_draft.md
paper/manuscript_sections/benchmark_setup_draft.md
paper/manuscript_sections/main_results_draft.md
paper/manuscript_sections/limitations_draft.md
paper/p63_gap_slot_direct_vs_prompt.png
paper/p63_gap_slot_direct_vs_prompt.pdf
paper/p63_stale_vs_prompt_em_k16.png
paper/p63_stale_vs_prompt_em_k16.pdf
paper/p63_memory_size_vs_prompt_em_k16.png
paper/p63_memory_size_vs_prompt_em_k16.pdf
```

The packaging script now also generates derived diagnostic figures for:

1. the gap between oracle slot-state accuracy and slot-prompt EM,
2. k=16 stale same-slot burden vs slot-prompt EM,
3. k=16 final memory size vs slot-prompt EM.

Historical note: this P6.6-era external-baseline no-go decision is superseded by P6.8. External-baseline feasibility, especially Mem0 first, is now a top priority if it can be isolated and inspected for stale same-slot burden.

### P6.7 cluster-backed verification and manuscript assembly

P6.7 moves from local paper packaging to cluster-backed verification and a complete manuscript skeleton.

Additional files:

```text
paper/cluster_resource_snapshot.md
paper/remote_verification_log.md
paper/figure_table_placement_plan.md
paper/manuscript_draft.md
paper/p63_error_analysis_k16.md
paper/error_analysis_k16_local.json
```

Remote verification status:

- `Tang-2-Wu` remote smoke test passed: `SMOKE TEST: 18/18 passed`.
- `Tang-2-Wu` constrained CRUD slot-direct k=16 sanity passed with state accuracy 1.00 and stale same-slot 0.00.
- `Song-1-Wu` learned Long25 rerun failed before model load because that node cannot use the expected NAS `gmsra` environment cleanly and hit a Python/transformers static TLS import issue.
- Learned Long25 k=16 slot-direct and slot-prompt spot-checks were rerouted to `Sui-3-Wu` and completed under the expected NAS `gmsra` environment.
- Sui-3 Long25 observed: slot_direct state_acc=0.92, slot_prompt EM/F1=0.49/0.5467, stale_same_slot=1.13, final_memory_size=9.28.

Local k=16 error analysis found:

- Constrained CRUD slot-prompt k=16 has 100/100 correct state predictions; its EM/F1 gap is therefore answer-layer/prompt-conditioned, not stale-state retention.
- Long25 slot-prompt k=16 has 91/100 correct state predictions and 9 `wrong_value` errors, mostly on `company` slots.

### Next steps superseded

The earlier paper-packaging next steps are superseded by the stricter reviewer-risk direction below. Existing paper assets remain useful evidence records, but the default path is no longer prose polish or figure packaging.

## P6.8 Reviewer-Risk Reorientation

### Motivation

`docs/critical_review.md` gives a strict simulated ACL/EMNLP/NeurIPS-style review and rates the current version as a likely reject. The most important criticisms are not cosmetic: the project lacks external baselines, data scale/diversity, related work, and deep mechanism analysis. This reorients the next phase toward evidence that can change the paper's level rather than better packaging of the same results.

### Reviewer-risk diagnosis

Current strengths:

- The repeated same-slot update stressor is clear and controlled.
- Existing P6.3/P6.5 results separate final-state reliability, stale burden, compactness, and prompted answer robustness.
- P6.5 prompt-robustness results show the k=16 pattern is not a single-template artifact.
- Answer traces expose separate state, retrieval-context, stale-contamination, distractor, and answer-generation failure layers.

Current blockers for a strong paper:

- No real external memory baseline has been evaluated.
- The current paper-facing data scale and attribute diversity are too small for a broad benchmark claim.
- Related work is missing and must cover AMemGym, Ledger-QA/UMA, Memory-R1, Mem0, MemGPT/Letta, LoCoMo/LongMemEval, dialogue state tracking, and knowledge editing.
- The tradeoff curve is currently mostly cross-method; reviewers can reasonably ask for same-method-family parameter sweeps.
- The Constrained CRUD k=16 answer-layer gap is interesting but still needs deeper controlled diagnosis.

### New default priority order

1. **External baselines first.** Run an isolated Mem0 feasibility probe, then expand if it exposes memory state and stale same-slot burden. Investigate Memory-R1 only after confirming code/checkpoint availability. MemGPT/Letta are optional if they can be isolated and inspected.
2. **Answer-layer mechanism analysis.** Extend P6.5 diagnostics with oracle retrieval, retrieval top-k/context-length sensitivity, and case studies for Constrained CRUD failures.
3. **Stale-burden interventions.** On Raw append k=16, remove/filter stale same-slot entries at retrieval time and measure EM/F1 recovery to test mechanism rather than only correlation.
4. **Same-method-family tradeoff.** Sweep heuristic CRUD thresholds, e.g. 0.70/0.80/0.85/0.90/0.95, and plot stale burden vs state accuracy vs slot-prompt EM.
5. **Data expansion.** Add a separate opt-in split with more examples, more attributes, and paraphrased explicit update templates. Keep exact `(entity, attribute)` semantics. Do not add implicit updates to the main split until their gold semantics are unambiguous.
6. **Related work and positioning.** Rewrite the manuscript claim around a narrow controlled diagnostic benchmark for repeated same-slot updates, not a broad memory benchmark.
7. **Long25 stability and repair.** Finish current seed stability checks. If repair is pursued, target operation selection and NOOP discrimination, since P6.5 action pathology shows invalid action rate is near zero.
8. **k=32 only after core risks.** Use k=32 as an extrapolation stress test after external baseline feasibility and mechanism diagnostics are in place.

### Recent P6.5/P6.8 evidence now available

```text
results/p65_prompt_robustness/
results/p65_prompt_robustness_summary/
results/p65_diagnostics/k16_prompt_diagnostics.json
results/p65_diagnostics/long25_action_pathology_by_k.{json,csv,md}
results/p65_stability_sharded/
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
results/p68_expanded_oracle/
results/p68_expanded_baselines/
results/p68_expanded_baselines_summary/
data/evomemory_update_frequency_expanded_p68_{train,dev,test}.json
data/evomemory_update_frequency_expanded_k{1,2,4,8,16}_p68_{dev,test}.json
results/p68_answer_layer_diagnostics/
results/p68_stale_intervention/
results/p68_heuristic_threshold_summary/
results/p68_mem0_feasibility/
results/p69_external_baselines/
results/p69_expanded_slot_prompt/
results/p69_expanded_slot_prompt_summary/
results/p69_expanded_slot_prompt_allk/
results/p69_expanded_slot_prompt_allk_summary/
results/p69_k32_oracle/
results/p69_k32_slot_direct/
results/p69_k32_slot_direct_summary/
results/p69_k32_slot_prompt/
results/p69_k32_slot_prompt_summary/
data/evomemory_update_frequency_expanded_k32_p69k32_{dev,test}.json
paper/p65_prompt_robustness_note.md
paper/p65_diagnostic_findings.md
paper/p65_long25_stability_note.md
paper/p68_expanded_split_note.md
paper/p68_expanded_baseline_note.md
paper/manuscript_sections/related_work_positioning_draft.md
paper/p68_answer_layer_mechanism_note.md
paper/p68_stale_intervention_note.md
paper/p68_heuristic_tradeoff_note.md
paper/p68_external_baseline_feasibility.md
paper/p69_external_baseline_result_note.md
paper/p69_expanded_slot_prompt_note.md
paper/p69_k32_extrapolation_note.md
scripts/summarize_prompt_robustness.py
scripts/analyze_action_pathology.py
scripts/analyze_answer_layer_mechanism.py
scripts/analyze_stale_intervention.py
scripts/summarize_heuristic_threshold.py
scripts/eval_mem0_baseline.py
scripts/merge_evomemory_shards.py
scripts/run_p65_long25_sharded_tang3.sh
```

Key findings:

- Prompt variants do not rescue Raw append at k=16: EM remains around 0.09-0.11 despite perfect final-state availability.
- Constrained CRUD stays at perfect state accuracy and zero stale burden, but slot-prompt EM remains around 0.68-0.69 at k=16; this confirms a clean-state answer-layer gap.
- Long25 remains between Constrained CRUD and Raw append at k=16 in the earlier prompt-robustness sweep, with state accuracy around 0.92 and prompt EM around 0.42-0.48 across variants.
- Answer traces show Raw append failures are dominated by gold-not-retrieved and stale contamination; Constrained CRUD failures include gold-not-retrieved and gold-retrieved-wrong-answer; Long25 mixes state errors with stale/distractor answer-context failures.
- Long25 action pathology shows invalid action rate is near zero, so future repair should target operation selection and NOOP discrimination rather than output-format cleanup.
- P7.0 Long25 reproducibility audit resolves the apparent conflict between the original P6.3 Long25 row and the later P6.5 stability row. The P6.3 row (`results/update_frequency_p63/long25_slot_prompt_k16/evomemory_results.json`) uses checkpoint `outputs/constrained_sft_curriculum_long25/best` and gives EM/F1 0.48/0.53, state accuracy 0.91, stale same-slot 1.13, and final memory size 9.43. The P6.5 stability row uses a different checkpoint family, `outputs/p65_long25_seed{11,22,33}/best`, on the same P6.3 hard k=16 test split and gives EM mean/std/range 0.880/0.008/0.870-0.890, F1 mean/std/range 0.908/0.004/0.903-0.913, state accuracy mean/std/range 0.967/0.021/0.940-0.990, stale same-slot around 0.07, and final memory size around 2.04. These must be reported as separate checkpoint families, not as reproductions of the exact same Long25 run.
- Long25 provenance artifacts are saved in `results/p70_long25_reproducibility/long25_provenance.{json,csv}` and the paper-facing decision is documented in `paper/p70_long25_reproducibility_note.md`. Future paper tables should use one explicit canonical family or show both families with checkpoint provenance.
- Learned Long25 stability evaluation should use sharded execution for future runs. The original Sui-3 serial jobs were too slow because the eval path makes thousands of batch-size-1 generations; Tang-3 sharding completed 60 shards and merged all three P6.5 seed checkpoints in roughly one hour wall-clock.
- P6.8 expanded split adds an opt-in scale/diversity stressor without overwriting P6.3: 2500 train / 1000 dev / 1000 test examples, 200 dev/test examples per k, and eight explicit attributes (`location`, `company`, `preference`, `language`, `timezone`, `hobby`, `instrument`, `project`). Deterministic constrained CRUD slot-direct sanity is 1.00 EM/F1/state accuracy for k=1/2/4/8/16 dev.
- Expanded split deterministic baselines confirm the state/stale/memory invariant at larger scale: constrained CRUD keeps stale same-slot at 0.00, while Raw append grows to 14.12 stale same-slot entries and 51.00 final memories at k=16 under slot-direct. Local heuristic CRUD matches Raw append because the local zero-vector encoder fallback makes semantic thresholding non-informative.
- Related-work positioning has been reframed around a narrow repeated same-slot update diagnostic, explicitly distinguishing MemUpdateBench from broad long-term memory benchmarks, external memory systems, memory editing, and dialogue state tracking. External frameworks are now treated as an external-validity gap rather than optional prose-only grounding.
- P6.9 expanded split model-backed slot-prompt evaluation on Tang-2 confirms the scale/diversity story. At k=16 dev, Constrained CRUD has state accuracy 1.00, stale burden 0.00, and EM/F1 0.675/0.688; Raw append has state accuracy 1.00, stale burden 14.12, and EM/F1 0.140/0.163. The all-k expanded dev sweep shows Raw append EM falling from 1.000 at k=1 to 0.725/0.315/0.095/0.140 at k=2/4/8/16, while stale retrieved is 1.00 for all k>1.
- P6.9 k=32 extrapolation was added as an opt-in appendix-style stress point, not a new main axis. Deterministic slot-direct sanity passes at k=32; Raw append keeps state accuracy 1.00 but grows to 28.50 stale same-slot entries and memory size 103.00. Slot-prompt k=32 stays near saturated collapse for Raw append (EM/F1 0.155/0.172) and shows a stable clean-state answer-layer gap for Constrained CRUD (EM/F1 0.655/0.655).
- P6.8 answer-layer mechanism diagnostics on full k=16 dev show Constrained CRUD improves from top-k5 EM 0.67 to gold-context EM 0.92 while keeping state accuracy 1.00, so its clean-state gap is largely retrieval/context selection rather than state failure.
- Raw append k=16 dev improves from top-k5 EM 0.14 to gold-context EM 0.92; stale retrieved falls from 1.00 to 0.00, supporting stale-burden contamination as a mechanism.
- P7.0 adds a slot-aware answer-time retrieval rewrite requested by the second-round review. `--retrieval_policy latest_per_slot` leaves raw_add writes unchanged, retrieves from the full store, then deduplicates the answer context by `(entity, attribute)` and keeps only the latest entry per slot. On raw_add k=16 dev, EM/F1 improves from the P6.8 normal top-k5 baseline 0.140/0.173 to 0.690/0.703, while memory size remains 52.00 and stale same-slot burden remains 14.25. This is strong intervention evidence that stale same-slot context contamination and failure to surface the latest slot entry drive much of raw append's slot-prompt collapse, but it is not a pure filter over the original top-k context.
- The all-k filtered dev sweep is now complete after migrating the remaining runs from Tang to Sui-3, which shares the same NAS-backed checkout and environment. Latest-per-slot filtered EM/F1 is 1.000/1.000 at k=1, 0.910/0.930 at k=2, 0.850/0.857 at k=4, 0.990/0.997 at k=8, and 0.690/0.703 at k=16. This shows that the slot-aware retrieval rewrite remains highly effective across the full k sweep, even though stale same-slot burden and memory size still grow with k. See `results/p70_stale_filter_intervention_summary/stale_filter_summary.{json,md}`, `results/p70_stale_filter_intervention_summary/stale_filter_allk_filtered.md`, `paper/p70_stale_filter_intervention_note.md`, `paper/p70_stale_filter_extension_note.md`, and `paper/p70_stale_filter_allk_note.md`.
- Raw append top-k10 increases gold retrieval relative to top-k5 but decreases EM, showing that simply retrieving more context does not solve stale competition.
- Heuristic CRUD threshold sweep gives a same-method-family tradeoff curve: state accuracy remains 1.00 across thresholds, while stale burden and memory size increase with threshold and slot-prompt EM is best at the lowest tested threshold.
- Mem0 external baseline feasibility reached isolated package installation and import under `external/mem0_vendor`. P69 discovered a usable local OpenAI-compatible vLLM endpoint on Tang-2 (`Qwen2.5-VL` at port 8011), a cached local MiniLM embedder, and a local Qdrant path, so Mem0 can now run end-to-end without external API keys. The resulting off-the-shelf Mem0 dev20 row remains badly misaligned with exact repeated-slot tracking: improved value extraction reaches EM 0.00 / F1 0.05, memory is inspectable, and retrieved values are mostly stale or wrong. P7.0 fairness audit therefore treats this as a qualitative runnable probe, not a fair main-table external baseline. A follow-up text-backend probe found local Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct weights on NAS. A minimal project-local OpenAI-compatible transformers server was launched on Sui-3 at `http://127.0.0.1:8013/v1` with Qwen2.5-7B-Instruct, and MiniLM embeddings were forced to CPU via `configs/mem0_qwen25_text_qdrant_minilm384_cpu.json`, which fixed the earlier CUDA CUBLAS embedding failure. However, Mem0's structured extraction parser repeatedly rejected the lightweight server's JSON-like responses, and both k=16 dev20 and k=16 dev3 stopped before the first completed example with zero `Progress:` lines. The remaining blocker is adapter/extraction compatibility, not model-weight availability. Original Memory-R1 is also not currently available under `/NAS/yesh`; the repository only contains a project-local `baselines/memory_r1_agent.py` approximation, which must not be reported as original Memory-R1. See `paper/p70_external_baseline_fairness_note.md` and `paper/p70_external_baseline_text_backend_probe.md`.
- P8.0 begins the v3 long-horizon benchmark+analysis plan. `scripts/build_evidence_manifest.py` now scans existing `results/**/evomemory_results.json` files into `results/p80_evidence_manifest/evidence_manifest.{json,csv,md}`; the refreshed manifest contains 370 result rows after adding P8 Llama and expanded latest-per-slot outputs and is meant as provenance infrastructure rather than a paper table. `scripts/analyze_stale_dose_response.py` then pools existing raw_add slot-prompt P6.3 and expanded all-k results into `results/p80_stale_dose_response/`. The first-pass dose-response analysis covers 1500 examples and shows that EM drops from 0.967 at stored stale count 0 to 0.743 at stale count 1 and 0.290 at stale count 3. Lightweight logistic fits estimate ED50 ≈ 3.18 for stored stale count and ED50 ≈ 1.89 for retrieved stale count. This supports the v3 review's claim that stale contamination should be analyzed as a dose-response mechanism, and suggests retrieved stale exposure is closer to the answer-time failure mechanism than mere memory-store pollution. See `paper/p80_stale_dose_response_note.md`.
- P8.0 also adds `scripts/analyze_attribute_failures.py` and `results/p80_attribute_error_analysis/` to inspect expanded-split attribute sensitivity. The first-pass k=16 table shows that low Constrained CRUD EM for `company` and `language` is not purely gold-retrieved-wrong-answer: `company` has EM 0.28 and gold retrieved 0.28, while `language` has EM 0.12 and gold retrieved 0.60 despite state accuracy 1.00. This reframes part of the residual gap as clean-state retrieval/context selection failure. Attributes with gold retrieval 1.00 but lower EM, such as `hobby` (0.68), `project` (0.60), and `timezone` (0.80), remain useful for studying true gold-present answer-layer failures. See `paper/p80_attribute_error_case_study.md`.
- P8.0 mechanism probes now support `--context_order` and `--context_annotation` in `scripts/eval_evomemory.py`, with smoke coverage in `scripts/smoke_test.py`. The first raw_add k=16 dev batch on Sui-3 held retrieval composition fixed across context presentations: gold retrieved 0.360, stale retrieved 1.000, and retrieved stale count 4.040 in every formal condition. Under those fixed retrieved entries, normal order/no annotation gives EM/F1 0.110/0.136; chronological order improves to 0.230/0.275; reverse chronological drops to 0.010/0.050; timestamp annotation improves to 0.150/0.200; and explicit `[latest]`/`[outdated]` labels improve to 0.260/0.298. This supports semantic/version disambiguation and order sensitivity as answer-layer mechanisms, while also showing that labels do not solve the larger latest-retrieval bottleneck. See `results/p80_mechanism_probe_summary/context_mechanism_summary.{json,csv,md}` and `paper/p80_context_mechanism_probe_note.md`.
- P8.0 adds a controlled synthetic same-slot probe in `scripts/run_synthetic_same_slot_probe.py` plus `scripts/summarize_synthetic_same_slot_probe.py`. The first Sui-3 batch covers 768 examples across stale counts 0/1/2/4/8/16, conflict vs same_as_current values, chronological vs reverse chronological order, and no-label vs `[latest]`/`[outdated]` labels. In conflict contexts without labels, chronological order remains relatively robust as the current value appears last (EM 0.750 at stale=16), but reverse chronological order collapses immediately (EM 0.188 at stale=1, near 0 by stale=2+). Latest/outdated labels almost completely repair this conflict-driven collapse. In same_as_current contexts, EM still drops at higher repetition even though answer-value-present stays 1.000, suggesting a separate attention/formatting dilution effect. This rejects a simple majority-vote-only explanation and supports interacting mechanisms: value conflict, order sensitivity, version ambiguity, and repetition/format dilution. See `results/p80_synthetic_same_slot_probe/`, `results/p80_synthetic_same_slot_probe_analysis/`, and `paper/p80_synthetic_same_slot_probe_note.md`.
- P8.0 Long25 provenance audit now distinguishes a verified checkpoint-family mismatch from an unverified pure seed effect. The original P6.3 Long25 checkpoint (`outputs/constrained_sft_curriculum_long25/best`) and the later P6.5 family (`outputs/p65_long25_seed{11,22,33}/best`) are evaluated on the same P6.3 hard k=16 test split, but available local artifacts do not prove matched training commands with only `--seed` changed. The safe paper wording is therefore training/checkpoint-provenance sensitivity, not Long25 seed sensitivity. See `paper/p80_long25_training_provenance_audit.md`.
- P8.0 expanded latest-per-slot all-k is complete on expanded dev. Raw append with `retrieval_policy=latest_per_slot` gives EM/F1 0.955/0.970 at k=1, 0.940/0.954 at k=2, 0.855/0.855 at k=4, 0.925/0.929 at k=8, and 0.750/0.764 at k=16. Stale retrieved is 0.000 for every k, while gold retrieval falls to 0.860 at k=16. The earlier P6.3 k=8 near-perfect EM=0.990 therefore attenuates on the larger expanded split, supporting the interpretation that it was partly sample/attribute composition rather than a robust retrieval sweet spot. See `results/p80_expanded_latest_per_slot_summary/expanded_latest_per_slot_summary.{json,csv,md}` and `paper/p80_expanded_latest_per_slot_note.md`.
- P8.0 Llama3.1-8B multi-model replication is complete on both P6.3 hard k=16 dev and test. On dev, Llama raw_add normal top-k5 collapses to EM/F1 0.060/0.062 with stale retrieved rate 1.000, while latest_per_slot removes stale retrieval and improves to 0.290/0.341. On test, the same pattern remains: normal top-k5 is 0.040/0.042, latest_per_slot is 0.290/0.345, latest/outdated labels are 0.100/0.125, chronological is 0.050/0.057, and reverse chronological is 0.040/0.040. This supports model-agnostic stale-context susceptibility, but the recovery is much weaker than Qwen's k=16 dev latest_per_slot EM 0.690. The paper should state that stale collapse generalizes across answer models, but mitigation magnitude and context-presentation response are model-dependent. See `results/p80_multimodel_stale_susceptibility_summary/llama31_8b_context_summary.{json,csv,md}`, `results/p80_multimodel_stale_susceptibility_summary/llama31_8b_test_context_summary.{json,csv,md}`, and `paper/p80_multimodel_stale_susceptibility_note.md`.
- P8.0 manuscript integration has started. `paper/manuscript_draft.md` now includes the dose-response, real-context mechanism, synthetic same-slot, Llama replication, expanded latest-per-slot, and Long25 provenance conclusions in the main narrative. `scripts/package_p80_paper_tables.py` generates `paper/p80_paper_tables.md`, a manuscript/appendix table pack for dose-response, real-context mechanisms, Llama replication, and expanded latest-per-slot all-k results. `scripts/package_p80_figures.py` generates paper figures under `paper/figures/`: stale dose-response, synthetic same-slot mechanism matrix, expanded latest-per-slot curve, and Llama stale-susceptibility chart. `paper/manuscript_sections/p80_results_section_draft.md` further converts these results into LaTeX-style table drafts, figure-caption drafts, and suggested main-text paragraphs for the mechanism-analysis section. `paper/p80_claim_evidence_matrix.md` maps final paper claims to scripts, result artifacts, and caveats for release/manuscript audit, while `paper/p80_canonical_main_number_ledger.md` lists main-paper candidate numbers with exact source paths. `paper/p80_release_candidate_checklist.md` and `paper/p80_remaining_work_summary.md` summarize release readiness, true blockers, and optional external-validity extensions.
- V4 closing-phase final-lock pass on 2026-05-05 updated the manuscript and P8 section draft to include Llama3.1-8B test confirmation explicitly, and cleaned stale release wording that still described the Llama test run as pending. The current remaining work is manuscript LaTeX/citation/figure placement plus release packaging, not additional stale-mechanism experimentation. Validation passed with `PYTHONPATH=. python -m py_compile ...`, `PYTHONPATH=. python scripts/smoke_test.py` (`SMOKE TEST: 26/26 passed`), refreshed evidence manifest (`num_rows: 375`), regenerated `paper/p80_paper_tables.md`, and regenerated `paper/figures/p80_figure_manifest.json` with 8 figure entries.
- Advisor-requested credibility upgrades began after the v4 closing plan. Third-model resources were checked on Sui-3: `/NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1` and `/NAS/HuggingFaceModels/Phi-3-mini-4k-instruct` are available. Mistral-7B-Instruct was selected as the third 7B-scale model, and `scripts/run_p80_third_model_replication_sui3.sh` now runs the same five-condition stale-susceptibility matrix as the Llama replication. The full P6.3 k=16 dev Mistral matrix completed: normal top-k5 EM/F1 0.080/0.177 with stale retrieved 1.000, latest_per_slot EM/F1 0.720/0.735 with stale retrieved 0.000, latest/outdated labels 0.300/0.332, chronological 0.150/0.182, and reverse chronological 0.040/0.117. This addresses the third-model requirement and shows Mistral recovers strongly like Qwen, unlike Llama.
- A simple external extract-then-store pipeline baseline was added in `scripts/eval_simple_external_pipeline.py` with summary script `scripts/summarize_simple_external_pipeline.py`. It uses the project parser to extract inspectable `(entity, attribute, value)` memory records, then compares append versus slot-update storage. On P6.3 k=16 dev, append parsed-only keeps state accuracy 1.000 but stale same-slot 14.250 and slot-prompt EM/F1 0.140/0.177; slot-update parsed-only has state accuracy 1.000, stale 0.000, memory size 2.000, and slot-prompt EM/F1 0.910/0.926. This gives the paper a transparent external-pipeline diagnostic row without depending on Mem0. Results are documented in `paper/p80_simple_external_pipeline_note.md`, `paper/p80_third_model_mistral_note.md`, and `paper/p80_multimodel_stale_susceptibility_note.md`. The refreshed evidence manifest now contains 386 result rows, and `paper/p80_paper_tables.md` includes both Mistral and simple external pipeline sections.
- The advisor-requested attribute-sensitive case study was deepened in `scripts/analyze_attribute_failures.py`. New outputs `results/p80_attribute_error_analysis/company_language_error_type_summary.csv` and `gold_retrieved_wrong_cases.{csv,md}` isolate company/language failures where gold is retrieved but the answer is still wrong. The k=16 result clarifies that `company` is mostly retrieval/context-selection failure under clean Constrained CRUD (18/25 gold-not-retrieved), while `language` has a true gold-retrieved answer-layer failure mode (12/25 gold-retrieved but wrong), often caused by near-miss language distractors such as workshop/discussion statements. The paper-facing interpretation is updated in `paper/p80_attribute_error_case_study.md`.
- The advisor-requested Lost-in-the-Middle comparison was added as a strict gold-position probe in `scripts/run_lost_in_middle_probe.py` with `scripts/summarize_lost_in_middle_probe.py`. Unlike chronological/reverse probes, it fixes the context set and moves only the gold entry to beginning/middle/end. On Qwen2.5-7B-Instruct P6.3 k=16 dev with 8 distractors, gold-at-end reaches EM/F1 0.630/0.654, gold-in-middle drops to 0.090/0.183, and gold-at-beginning drops to 0.010/0.073. This directly links MemUpdateBench stale-context failures to a Lost-in-the-Middle-style position effect with a strong final-position/recency advantage. Results are documented in `paper/p80_lost_in_middle_probe_note.md`, `results/p80_lost_in_middle_probe_summary/`, the canonical ledger, and the P8 table pack.

### Validation discipline

Continue ending substantial code/data phases with:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py scripts/analyze_answer_layer_mechanism.py scripts/analyze_stale_intervention.py scripts/eval_mem0_baseline.py scripts/merge_evomemory_shards.py scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py scripts/summarize_heuristic_threshold.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py scripts/package_update_frequency_paper_assets.py
PYTHONPATH=. python scripts/eval_evomemory.py --mode constrained_slot_crud --answer_mode slot_direct --data_file data/evomemory_update_frequency_expanded_k16_p68_dev.json --output_dir results/p68_expanded_oracle/k16_dev
python scripts/smoke_test.py
```

For any new split or baseline:

1. keep exact `(entity, attribute)` slot semantics where applicable,
2. run deterministic oracle sanity before learned/black-box interpretation,
3. compute stale same-slot burden whenever memory state is inspectable,
4. document commands, outputs, metrics, errors, and conclusions here.

## P8.1 methodological rigor pass after critical review v4

### Motivation

`docs/critical_review_v4.md` identified four remaining one-week methodological holes after the larger v4 credibility upgrades were complete: larger-n synthetic diagnostic cells, same_as_current exact-match failure inspection, Llama zero-stale control, and a k-controlled heuristic dose-response curve. This pass closes those holes without reopening broad benchmark scope.

### Files changed/generated

```text
scripts/run_synthetic_same_slot_probe.py
scripts/analyze_same_as_current_failures.py
scripts/summarize_heuristic_threshold.py
scripts/run_p81_synthetic_same_slot_expanded_sui3.sh
scripts/run_p81_llama_constrained_zero_stale_sui3.sh
scripts/run_p81_heuristic_threshold_k16_sui3.sh
results/p81_synthetic_same_slot_probe_expanded/
results/p81_synthetic_same_slot_probe_expanded_analysis/
results/p81_same_as_current_failure_analysis/
results/p81_llama_constrained_zero_stale/
results/p81_heuristic_threshold_k16_rigor/
results/p81_heuristic_threshold_k16_rigor_summary/
```

### Commands run

Remote jobs ran under `/NAS/yesh/MemUpdateBench` on Sui-3-Wu and Tang-1-Wu:

```bash
bash scripts/run_p81_synthetic_same_slot_expanded_sui3.sh
bash scripts/run_p81_llama_constrained_zero_stale_sui3.sh
bash scripts/run_p81_heuristic_threshold_k16_sui3.sh
python scripts/summarize_synthetic_same_slot_probe.py \n  --input_csv results/p81_synthetic_same_slot_probe_expanded/synthetic_same_slot_examples.csv \n  --output_dir results/p81_synthetic_same_slot_probe_expanded_analysis
python scripts/summarize_heuristic_threshold.py \n  --result_root results/p81_heuristic_threshold_k16_rigor \n  --output_dir results/p81_heuristic_threshold_k16_rigor_summary \n  --prefix heuristic_threshold_k16_summary
python scripts/analyze_same_as_current_failures.py \n  --input_csv results/p80_synthetic_same_slot_probe/synthetic_same_slot_examples.csv \n  --output_dir results/p81_same_as_current_failure_analysis
```

### Results

Expanded synthetic selected cells now have 64 examples each. The main pattern is stable: `conflict + reverse_chronological + none` collapses quickly (stale=1/2/8/16 EM 0.234/0.094/0.000/0.031), `latest_outdated_label` repairs the conflict setting (stale=1/2/8/16 EM 0.969/0.969/1.000/1.000), and `conflict + chronological + none` remains much more robust at stale=16 EM 0.797. Same-as-current cells keep answer-value-present 1.000 while exact EM falls, supporting answer-surface dilution rather than stale-value selection.

The same_as_current failure analysis found 76 existing-pilot cases with EM failure but answer-value-present=1, now classified under `results/p81_same_as_current_failure_analysis/`.

The Llama constrained CRUD zero-stale control is complete. Slot-prompt EM/F1 is 0.270/0.321 with value EM 0.660, answer-value-present 0.730, state accuracy 1.000, and stale same-slot burden 0.000. Slot-direct remains 1.000. This confirms that Llama's weak latest_per_slot recovery includes an answer-layer / instruction-following weakness even when stale context is removed.

The fixed k=16 heuristic threshold sweep is complete. As the threshold rises from 0.70 to 0.95, stale same-slot burden rises from 4.43 to 13.04 and memory size from 11.57 to 42.20. Slot-direct remains 1.000 throughout, while slot-prompt EM decreases overall from 0.220 to 0.100, closing the k-confounding caveat in the dose-response story.

### Conclusion

The four v4 one-week methodological holes are now closed. Experimental scope should freeze unless the advisor explicitly requests a new direction; the next priority is final paper production, numerical consistency checking, README/release polish, and result packaging decisions.

## P8.2 Mistral zero-stale ceiling-recovery lock

### Motivation

`docs/critical_review_v5.md` identified one final low-cost experiment needed to complete the multi-model story: Mistral constrained CRUD k=16 slot-prompt under zero-stale memory. P8.1 had shown that Llama latest_per_slot recovery matched its own zero-stale ceiling; v5 argued that if Mistral showed the same pattern, the paper should frame multi-model evidence as ceiling recovery rather than model-dependent mitigation magnitude.

### Files changed/generated

```text
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
paper/p80_multimodel_stale_susceptibility_note.md
paper/p80_third_model_mistral_note.md
paper/p80_canonical_main_number_ledger.md
paper/p80_claim_evidence_matrix.md
paper/p80_remaining_work_summary.md
paper/p80_release_candidate_checklist.md
paper/manuscript_draft.md
paper/manuscript_sections/p80_results_section_draft.md
```

### Command run

```bash
cd /NAS/yesh/MemUpdateBench
source activate.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/run_p82_mistral_constrained_zero_stale_sui3.sh > logs_p82_mistral_ctrl.txt 2>&1
```

### Results

Mistral constrained CRUD k=16 dev zero-stale results:

| Answer mode | EM | F1 | value EM | answer value present | state acc. | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Mistral latest_per_slot was already EM/F1 0.720/0.735, so the zero-stale ceiling exactly matches stale-filtered recovery. Together with Qwen and Llama, this supports the revised headline: retrieval-time stale filtering recovers each tested model to approximately its own zero-stale slot-prompt ceiling. Absolute differences across models reflect clean-context / instruction-following ceilings rather than differential stale-mechanism susceptibility.

### Conclusion

The final v5-requested experiment is complete. Experiments should now freeze. Remaining work should be manuscript production, citation cleanup, figure/table placement, numerical consistency checking, README polishing, and result packaging.

## P8.3 stale same-slot conflict mechanism package

### Motivation

P8.3 sharpens the paper's novelty by testing whether stale same-slot entries are a distinct high-similarity version-conflict mechanism rather than generic retrieval noise.

### Files changed/generated

```text
scripts/run_conflict_type_probe.py
scripts/summarize_conflict_type_probe.py
scripts/run_synthetic_same_slot_probe.py
scripts/analyze_stale_specific_removal.py
scripts/run_p83_conflict_type_probe_sui3.sh
scripts/run_p83_stale_conflict_dose_sui3.sh
results/p83_conflict_type_probe/
results/p83_conflict_type_probe_summary/
results/p83_stale_conflict_dose/
results/p83_stale_conflict_dose_summary/
results/p83_raw_add_k16_trace/
results/p83_stale_specific_removal_trace/
paper/p83_stale_same_slot_conflict_plan_note.md
```

### Commands run

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m py_compile scripts/run_conflict_type_probe.py scripts/summarize_conflict_type_probe.py scripts/run_synthetic_same_slot_probe.py scripts/analyze_stale_specific_removal.py scripts/smoke_test.py
PYTHONPATH=. python scripts/summarize_conflict_type_probe.py --input_csv results/tmp_conflict_summary_fixture/conflict_type_examples.csv --output_dir results/tmp_conflict_summary_fixture/summary
bash -n scripts/run_p83_conflict_type_probe_sui3.sh
bash -n scripts/run_p83_stale_conflict_dose_sui3.sh
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/run_conflict_type_probe.py --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct --examples_per_condition 128 --distractor_count 4 --conditions final_only,unrelated_distractors,same_entity_different_attribute,different_entity_same_attribute,stale_same_slot --output_dir results/p83_conflict_type_probe/qwen25_7b_d4 --no_qlora
PYTHONPATH=. python scripts/summarize_conflict_type_probe.py --input_csv results/p83_conflict_type_probe/qwen25_7b_d4/conflict_type_examples.csv --output_dir results/p83_conflict_type_probe_summary/qwen25_7b_d4
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/run_synthetic_same_slot_probe.py --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct --examples_per_condition 64 --stale_counts 0,1,2,4,8,16 --value_policies conflict --context_orders chronological,reverse_chronological,middle,random --context_annotations none,latest_outdated_label --output_dir results/p83_stale_conflict_dose/qwen25_7b --no_qlora
PYTHONPATH=. python scripts/summarize_synthetic_same_slot_probe.py --input_csv results/p83_stale_conflict_dose/qwen25_7b/synthetic_same_slot_examples.csv --output_dir results/p83_stale_conflict_dose_summary/qwen25_7b
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/eval_evomemory.py --mode raw_add --answer_mode slot_prompt --data_file data/evomemory_update_frequency_hard_k16_p63_dev.json --output_dir results/p83_raw_add_k16_trace --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct --no_qlora --save_answer_traces
PYTHONPATH=. python scripts/analyze_stale_specific_removal.py --input_json results/p83_raw_add_k16_trace/evomemory_results.json --output_dir results/p83_stale_specific_removal_trace
```

Remote execution used Tang-1-Wu GPU7 for the conflict-type and synthetic dose-response probes, and Tang-2-Wu GPU2 for the raw_add k16 trace rerun.

### Results

Conflict-type decomposition on Qwen2.5-7B-Instruct produced 640 examples. With four distractors, final-only EM/F1 was 0.875/0.888. Same-entity/different-attribute distractors were easiest (1.000/1.000), different-entity/same-attribute was comparable to final-only (0.891/0.944), stale same-slot was also comparable (0.867/0.899, stale copied 0.133), while unrelated distractors were worse (0.750/0.887). This first construction therefore does not support the strongest claim that stale same-slot entries are always more harmful than generic distractors; it suggests that the prompt/context template and distractor naturalness matter and that the paper should not overclaim from this table alone.

Version-conflict dose-response produced 3072 examples. The refined synthetic probe strongly supports order-sensitive version conflict: without labels, reverse-chronological EM collapsed from 0.234 at stale=1 to 0.094 at stale=2, 0.000 at stale=8, and 0.031 at stale=16. Random order without labels also degraded with stale count, reaching EM 0.047 at stale=16. Middle placement without labels was unstable but much lower than chronological placement for most stale counts. Chronological/no-label remained comparatively robust, with EM 0.812 at stale=16, because the current value appears last. Latest/outdated labels largely repaired all non-chronological conflict settings, with reverse-chronological EM 1.000 at stale=8 and stale=16.

The raw_add k=16 dev trace rerun reproduced low slot-prompt performance: EM/F1 0.130/0.163 with final memory size 52. Trace-level stale-specific removal analysis over 100 examples showed normal retrieved contexts had gold-in-context rate 0.320 and average stale same-slot count 4.040. Removing stale same-slot entries reduced stale count to 0.000 while preserving gold-in-context rate 0.320 and leaving 0.960 entries on average. Random non-gold removal removed the same average number of entries and left stale count 0.270, while unrelated and near-slot removal barely changed stale exposure. Latest-per-slot reduced context to 1.500 entries on average but had lower gold-in-context rate 0.310 and residual stale count 0.690 in this trace-level analyzer, indicating that event-index availability and retrieved-entry schema should be audited before treating this proxy as equivalent to the full answer-time latest-per-slot intervention.

### Conclusion

P8.3 strengthens the mechanism-first framing mainly through the refined synthetic dose-response and trace-level removal diagnostics. The cleanest supported claim is that stale same-slot conflict creates severe answer-layer failures when version order is ambiguous or the current value is not presented last, and that explicit latest/outdated metadata can largely repair the synthetic conflict. The conflict-type decomposition is a useful negative/nuanced result: in its current surface form, stale same-slot distractors are not uniformly worse than all generic distractors, so the manuscript should frame stale conflict as an order- and metadata-sensitive version arbitration mechanism rather than a universally largest distractor category.

## P8.4 latest API answer-model probe

### Motivation

Advisor feedback after the group meeting raised a model-recency concern: the existing Qwen2.5/Llama3.1/Mistral evidence might look dated to future readers or reviewers. P8.4 therefore adds a latest GPT/Gemini API answer-layer probe to test whether the stale same-slot version-arbitration mechanism persists beyond the earlier open-weight answer models.

### Files changed/generated

```text
scripts/probe_api_answer_model.py
scripts/summarize_api_latest_model_probe.py
scripts/run_p84_api_latest_models_tang2.sh
results/p84_api_latest_model_probe_summary/api_latest_model_summary.{json,csv,md}
paper/p84_latest_api_model_probe_note.md
docs/superpowers/plans/2026-06-02-latest-api-answer-models.md
```

Remote per-model outputs were generated under:

```text
/NAS/yesh/MemUpdateBench/results/p84_api_latest_model_probe/
```

### Commands run

Local validation:

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m py_compile scripts/probe_api_answer_model.py scripts/summarize_api_latest_model_probe.py scripts/smoke_test.py
```

Server API feasibility and runs used environment variables for the API base URL/key and did not write secrets to repository files. The main remote pattern was:

```bash
source activate.sh
MUB_API_MODEL=<model> PYTHONPATH=. python scripts/probe_api_answer_model.py \
  --connectivity \
  --synthetic-dose-probe \
  --stale-counts 0,1,2,4,8,16 \
  --examples-per-condition 16 \
  --output-dir results/p84_api_latest_model_probe \
  --timeout 120
PYTHONPATH=. python scripts/summarize_api_latest_model_probe.py \
  --result-root results/p84_api_latest_model_probe \
  --output-dir results/p84_api_latest_model_probe_summary
```

### Model availability

Completed synthetic-dose summaries:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gemini-2.5-flash
gemini-2.5-pro
gemini-3-flash-preview
gemini-3.1-flash-lite-preview
```

Excluded or non-interpretable:

```text
gemini-3-pro-preview        # minimal OK probe passed, but synthetic run returned empty content
gemini-3.1-pro-preview      # empty chat response
gpt-5.3-codex-spark        # account/model support error
gpt-5.3-codex              # account/model support error
gpt-5.2                    # account/model support error
```

### Results

The clean, format-stable latest-model subset strongly reproduces the version-arbitration mechanism. For `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, and `gemini-3.1-flash-lite-preview`:

- chronological/no-label at stale=16: EM 1.000, stale copied 0.000;
- reverse/no-label at stale=1: EM 0.000, stale copied 1.000;
- reverse/no-label at stale=16: EM 0.000, stale copied 1.000;
- reverse+latest/outdated label at stale=16: EM 1.000, stale copied 0.000.

This is a direct latest-model replication of the P8.3 synthetic mechanism: when the current value is present but appears before stale same-slot versions and lacks explicit version metadata, even very recent GPT-family answer models copy stale values; explicit latest/outdated metadata repairs the conflict.

Gemini results require caution. `gemini-2.5-flash` and `gemini-2.5-pro` show stale copying in reverse/no-label contexts, but also many empty or truncated outputs under this prompt format. `gemini-3-flash-preview` produced mostly empty outputs, so it is not interpretable for the mechanism claim despite completing the run. These rows should be kept as API/prompt-format diagnostics rather than central evidence.

### Conclusion

P8.4 addresses the model-recency criticism without changing the paper's narrow claim. The strongest defensible statement is that the order- and metadata-sensitive stale same-slot version-arbitration failure is reproduced by current GPT API models and one current Gemini flash-lite model in a controlled answer-layer probe. Do not present these API runs as external memory baselines; they are latest-model robustness checks for the answer-layer mechanism.

## vNext Phase 0 Task 15: no-network smoke and legacy bridge handoff

### Motivation

Task 15 closes Phase 0 by exercising the accepted vNext contracts through the normal no-network smoke entry point and documenting the legacy compatibility boundary. The work intentionally does not start the Pilot, generate model results, or turn compatibility rows into benchmark or paper claims.

### Files changed

```text
scripts/smoke_test.py
tests/vnext/test_smoke_vnext.py
tests/vnext/test_environment.py
.gitattributes
docs/vnext/legacy_bridge.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-phase0-contract-legacy-bridge.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md
WORKFLOW.md
```

No generated acceptance artifact was written under `data/` or `results/`; all schema exports and compiled fixture outputs used temporary directories.

### TDD and commands run

The focused test first ran RED against the existing smoke executable: the script returned 0 with its existing `29/29` checks, but the required vNext pass line was absent. A direct import attempt also exposed an existing Windows pytest-capture interaction caused by the smoke script's stdout wrapper; Task 15 did not alter that unrelated behavior. The focused test was changed to execute the smoke script in an isolated subprocess and then passed GREEN.

```bash
PYTHONPATH=<worktree> python -m pytest tests/vnext/test_smoke_vnext.py -q
# RED: 1 failed; required vNext pass line absent from an otherwise green 29/29 smoke run
# GREEN: 1 passed in 9.82s

PYTHONPATH=<worktree> python scripts/smoke_test.py
# SMOKE TEST: 30/30 passed = 29 existing + 1 vNext

PYTHONPATH=<worktree> python -m pytest tests/vnext \
  --ignore=tests/vnext/test_phase0_cli.py -q
# 1476 passed in 53.39s

PYTHONPATH=<worktree> python -m pytest tests/vnext/test_phase0_cli.py -q -k 'atomic'
# 11 passed, 28 deselected in 10.35s

PYTHONPATH=<worktree> python -m pytest tests/vnext/test_phase0_cli.py -q \
  -k '(validator or source_recompilation) and not atomic'
# 10 passed, 29 deselected in 827.79s

PYTHONPATH=<worktree> python -m pytest tests/vnext/test_phase0_cli.py -q \
  -k 'not atomic and not validator and not source_recompilation'
# 18 passed, 21 deselected in 184.87s
```

After the review corrections, the final focused smoke rerun passed `1` test in `19.43s`, the normal smoke executable passed `30/30`, and `python -m py_compile scripts/smoke_test.py scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py scripts/vnext_validate_artifacts.py` exited 0 with no output.

The required unpartitioned command was first run exactly from the repository root before the final LF-attribute regression test was added:

```bash
python -m pytest tests/vnext -q
# historical snapshot: 1515 passed in 1144.50s (0:19:04)
```

That 1,515-test result and the earlier nonoverlapping partition run (`1476` non-Task 14 plus `11 + 10 + 18 = 39` Task 14 tests) are retained as historical/reviewer evidence for the pre-attribute-lock snapshot.

Final controller verification then ran the same exact command on the latest 1,516-test snapshot:

```bash
python -m pytest tests/vnext -q
# final authoritative result: 1516 passed in 1298.05s (0:21:38)
```

The compile command covered every Python module under `mub/vnext/`, including `failure.py`, `io/atomic.py`, and `legacy/artifacts.py`, plus all three vNext CLIs and `scripts/smoke_test.py`:

```bash
PYTHONPATH=<worktree> python -m py_compile \
  mub/vnext/__init__.py mub/vnext/version.py mub/vnext/profiles.py \
  mub/vnext/schema_export.py mub/vnext/failure.py \
  mub/vnext/contracts/{__init__,enums,common,task,runtime,score,manifest,adapter}.py \
  mub/vnext/io/{__init__,canonical,jsonl,atomic}.py \
  mub/vnext/validation/{__init__,issues,task,replay,split}.py \
  mub/vnext/scoring/{__init__,registry,failures,scorer}.py \
  mub/vnext/legacy/{__init__,artifacts,caveats,names,loaders,dataset,results,mechanisms,ledger}.py \
  scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py \
  scripts/vnext_validate_artifacts.py scripts/smoke_test.py
# exit 0, no output
```

Schema acceptance exported the five canonical schemas twice to distinct `TemporaryDirectory` roots, compared the two exact SHA-256 maps, and compared them with `schemas/vnext/`. The compatibility schema was independently regenerated from `LegacyAnalysisManifest.model_json_schema(mode="serialization")` and compared byte-for-byte with `schemas/legacy/legacy_analysis_manifest.schema.json`.

After review, representative commands were copied from `docs/vnext/legacy_bridge.md` and run from the repository root with `PYTHONPATH=.`. They exported five schemas, verified the canonical legacy schema at 1,829 bytes, compiled dataset/results/conflict/ledger fixtures in a temporary root, and produced `valid: true` for `tasks`, `task-manifest`, `task-runs`, `scores`, and `run-manifest`. Observed compatibility warnings were `legacy_answer_mode_unverified` for the intentionally incomplete result fixture and `unresolved_ledger_references` for the ledger fixture. The temporary command-probe root was removed afterward.

Final quality review corrected the bridge's namespace wording: imported P6.x slot keys use `MemoryObjectKey(namespace="default", subkey=None, object_type="slot")`, while `legacy_p63` and related values belong only to `LegacyProvenance.legacy_metric_namespace` and compatibility metric identity. Three focused compiler tests covering exact slot identity, provenance-only legacy identity, and the immutable legacy namespace registry passed as part of a 14-test environment/schema/compiler run in `10.31s`.

The already-approved Pilot implementation plan was present only in the original main workspace. It was inspected read-only, confirmed to contain planning/tasks rather than Pilot implementation artifacts, and copied byte-for-byte to `docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md`. Both copies are 53,678 bytes with SHA-256 `6be5401a758074c69048728c249bfc711f1007f046ca9f47f69026be27c3bb99`; the original was not modified.

A focused TDD check locked LF checkout policy for hash-sensitive artifacts. RED failed with exactly the three missing rules; after adding `text eol=lf` rules for `schemas/vnext/*.schema.json`, `schemas/legacy/*.schema.json`, and `tests/vnext/fixtures/legacy/*.md`, GREEN passed `1` test in `0.07s`. `git check-attr text eol --` reported `text: set` and `eol: lf` for representatives of all three classes. All six schema files and `ledger_references.md` contain LF-only bytes, retain the hash values below, and are unchanged by both direct LF normalization simulation and Git clean-filter simulation. Final focused verification passed 14 environment/schema/compiler tests in `10.31s`, the smoke integration test in `12.39s`, normal smoke `30/30`, and changed-file py_compile with no output. The final controller subsequently verified the complete current 1,516-test snapshot: `1516 passed in 1298.05s (0:21:38)`. This is the authoritative Phase 0 test result; the earlier 1,515-test run is a superseded snapshot retained only as historical/reviewer evidence.

Fixture acceptance compiled the dataset, results, all four mechanism kinds, and ledger fixture to one temporary destination, validated all five canonical artifact kinds, recompiled to the same authenticated destination with `--overwrite`, revalidated, and compared all 15 output hashes. The source fixtures were hashed before and after. An initial cross-directory comparison correctly found that manifests differ when the authenticated absolute output root differs; payloads were stable. The final determinism check therefore used the same destination, matching the manifest authentication contract rather than removing path evidence.

Read-only real-artifact checks found no P6.3 files under `data/` and no P8.3/P8.4 trees under `results/` in this worktree. Paper ledgers/notes are present, but were not substituted for authenticated source/result artifacts and were not converted into claims.

Final repository checks used:

```bash
git diff --check
git status --short
git diff --cached --name-only
git diff -- requirements.txt pytest.ini mub/vnext \
  scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py \
  scripts/vnext_validate_artifacts.py scripts/smoke_test.py schemas/vnext \
  tests/vnext docs/vnext/legacy_bridge.md WORKFLOW.md
```

The exact plan-scoped diff command showed the tracked `WORKFLOW.md`, `requirements.txt`, and `scripts/smoke_test.py` changes. As expected, Git omitted untracked Phase 0 paths. Supplementary scoped `git status --short` identified the expected untracked `.gitattributes`, both plans, bridge guide, `mub/vnext/`, `pytest.ini`, `schemas/{vnext,legacy}/`, three vNext CLIs, and `tests/vnext/`. Explicit no-index stats reviewed the 5-line `.gitattributes`, 25-line environment test, 24-line focused smoke test, 304-line bridge guide, 1,385-line corrected Phase 0 plan, and 1,167-line exact Pilot plan copy. A recursive inventory reviewed 85 scoped files and found no trailing whitespace. `git diff --check` passed apart from Git's LF-to-CRLF warnings for both `.gitignore` and `WORKFLOW.md`; neither file was normalized merely to silence warnings. No file was staged.

### Deterministic schema hash ledger

| Schema | SHA-256 |
| --- | --- |
| `schemas/vnext/mem_update_task.schema.json` | `d02f587b7856c6f970b374fda07051ec4e2d0d7f1243be7e6ade9678ca49f2fc` |
| `schemas/vnext/run_manifest.schema.json` | `c09278d4c82ba7167abb27c37870038b69d13747b2cb30093115b390ee0576fa` |
| `schemas/vnext/score_record.schema.json` | `fddf41b727e699ed5a72b866b22f696be52727f6b35b5caa4d15e08eaddb0891` |
| `schemas/vnext/task_manifest.schema.json` | `e952f446cfd574dc82c2f4a072c6c3dd7a51bc5062d62a2b043c90238a152207` |
| `schemas/vnext/task_run_record.schema.json` | `5427ebc86482b5bfd4f6e2d41ad053552a1415b641ed384c3836f39d2275de2a` |
| `schemas/legacy/legacy_analysis_manifest.schema.json` | `ee5193147b4df6d052c4cfa2df6e71e1ac2115fffcf6fc07ba697c7a69bd5877` |

### Deterministic fixture output hash ledger

These hashes are from the observed temporary destination. Manifest hashes intentionally authenticate that absolute destination and therefore differ across different output roots while remaining exact across overwrite recompilation to the same root.

| Temporary relative output | SHA-256 |
| --- | --- |
| `api/legacy_analysis.jsonl` | `aa7a1193709204e7dc9de24d7788e8e86b4bb510396f2052f6003d5752728026` |
| `api/legacy_analysis_manifest.json` | `9f4af23677cae4bd03dd5cd3a0ddc7e4c369a0cf08c6b1a974e45be90215e2fc` |
| `conflict/legacy_analysis.jsonl` | `df2a5dcef527e608e68b4c141ee2dc00decbf2c833574c37747409d7dfee692b` |
| `conflict/legacy_analysis_manifest.json` | `d1037ab9752d4235354e4fa9e012365568a6f206465578f63cdc807d14811441` |
| `dose/legacy_analysis.jsonl` | `2279272935e22b1f117ddf2a045c3e8814aef49bad97657499bb67b6f9dc96af` |
| `dose/legacy_analysis_manifest.json` | `b9e5b1799ea2cc23727a958c2e2d9a25beec9a7e9f35075ff36ff78b18dcc82e` |
| `ledger/ledger_audit.json` | `e40f77ee558792ac72e1b2f6d843f71cf4e5f786f34190598596ee07e55025f5` |
| `ledger/legacy_analysis_manifest.json` | `c09ac6dbcbbdf51b5c4dff8bf987a77222f8994fb3ebd5ee34f48c32be18be57` |
| `results/run_manifest.json` | `721cc3cd496a7a8289e2642551366f5c6600a78012fbe828192d319792a23bb9` |
| `results/scores.jsonl` | `bb9db7c8211934a652d8faec002adeaab04de7abae1d0c935f306664dc01257a` |
| `results/task_runs.jsonl` | `05730374613ffb6843e9a59d8255fa62a0d60d50633a2bec136574515a570587` |
| `stale-removal/legacy_analysis.jsonl` | `9b4007859a7ca4747df2e086f3f2299f72328f3515d451266ed5f3f993cabc70` |
| `stale-removal/legacy_analysis_manifest.json` | `13aabe9bc840de1c7265634467552afe4e7c4240baf5defeaa5d3dccadcf8231` |
| `tasks/task_manifest.json` | `fc38d4aaa221db6061b3004a3b1d45ba7217344c9bd2f3aa2341e7c67cc3a86c` |
| `tasks/tasks.jsonl` | `a93b5613bab7214d4eb687c6e79232814031058b5adb38a4aeb919d32784b617` |

Fixture record counts were: two tasks, two runtime records, two score records, conflict/dose/stale-removal two rows each, API three rows, and one ledger audit.

### Immutable source hash ledger

| Fixture | SHA-256 before and after |
| --- | --- |
| `p63_dataset_minimal.json` | `18b4c1346da3c4ee723be89882c764df3fa3d99a45d0ec121009244909caf47f` |
| `evomemory_results_old.json` | `0070edab3bb643680b23a2de0f760364b16e5d7f14dc10ec12760a91b0bdf960` |
| `p83_conflict_rows.csv` | `88395320807df32898345511906e8116e3b5c4f59e09b6f3fe95bfcbf9b8e4e8` |
| `p83_synthetic_dose_rows.csv` | `97a2528eeb01208710bfeb0ebdaacde69799bd4b8d3f9c92899b6d2de3da36d9` |
| `p83_stale_removal_rows.csv` | `f0990d9cc72ea48669ba1b0092814732af92a5f7a06fae057d8d5d5e676b4026` |
| `p84_api_rows.csv` | `fad6fe12443580ef1e94f39c86d5d2cfcb4bb94eafe8a45124e6fd4dfaf0210e` |
| `ledger_references.md` | `66e59c4ed7bf0340e1bea0b4e0d2d85a08ead75f9cc366f4341717687033b752` |

### Warnings, conclusion, and next steps

- Smoke emitted the expected FAISS-unavailable numpy-fallback logs; all checks passed.
- The fixture ledger intentionally reports `unresolved_ledger_references`; that warning remains visible in its compatibility manifest.
- Real P6.3/P8.3/P8.4 source/result artifacts are absent from this worktree, so read-only real-artifact authentication is unavailable here.
- The authoritative Phase 0 plan acceptance line 1361 was corrected to match the approved design and current code: identity is exactly `(namespace, entity, attribute, subkey)`, while `object_type` is classification metadata excluded from identity.
- Phase 0 creates compatibility/regression artifacts only. It reports no Pilot/model metric and makes no new paper claim.
- No commit or push was made. The corrected Phase 0 plan and implementation remain in the isolated `vnext-phase0` worktree. Read-only inspection of the original main workspace found overlapping uncommitted changes in `WORKFLOW.md` and `scripts/smoke_test.py` plus an untracked Phase 0 plan, so later synchronization requires manual/three-way reconciliation rather than overwriting either workspace. The original main workspace was not modified by Task 15.
- The next process step is independent spec/quality review of this Task 15 handoff. Do not start the Pilot until Phase 0 is reviewed and accepted.

## vNext Phase 0 Task 118: final integration revalidation

### Motivation and gate status

Task 118 regenerates and revalidates the integrated Phase 0 snapshot after Task 116 and Task 117 both passed their required gates: `SPEC_COMPLIANT` and `QUALITY_APPROVED`. This is integration/regeneration evidence only. It does not declare `FINAL_APPROVED`, does not declare Phase 0 complete, and does not start the Pilot. The final whole-phase reviewer remains required.

Task 118 intentionally changed only:

```text
schemas/vnext/mem_update_task.schema.json
WORKFLOW.md
```

The other four checked vNext schemas regenerated byte-identically. No source fixture, legacy result, paper artifact, Pilot implementation file, or original-main-workspace file was changed.

### Fresh commands and authoritative results

Baseline status and immutable inputs were captured before generation:

```bash
git status --short
python scripts/vnext_export_schemas.py --help
python <Task118 SHA-256 baseline script>
```

The baseline contained five checked schemas, eleven files under `tests/vnext/fixtures/legacy/`, and the pre-Task118 WORKFLOW hash. The complete immutable source hash ledger is below.

Checked schemas were regenerated transactionally through the official interface, not edited by hand:

```bash
PYTHONPATH=. python scripts/vnext_export_schemas.py --output-dir schemas/vnext
```

Result:

```text
Exported 5 vNext schemas to schemas\vnext
schema_debris=0
```

Only `mem_update_task.schema.json` changed. Its old stale hash was `d02f587b7856c6f970b374fda07051ec4e2d0d7f1243be7e6ade9678ca49f2fc`; its regenerated hash is `d377d386296aac94731df81cc5de4fd21a43433538f2bc62ec36c3d945e8b4ec`. The regenerated schema contains typed `GeneratorProvenance` under `$defs`, and `SourceRecord.generator` references it.

A second complete official export used a clean `TemporaryDirectory`, matched all five checked filenames and bytes, and was removed automatically:

```text
deterministic_schema_files=5
byte_exact=PASS
temp_cleaned=PASS
```

Full schema tests ran without deselection:

```bash
python -m pytest tests/vnext/test_schema_export.py -q
# 17 passed in 15.02s
```

The exact full suite then ran fresh from the repository root:

```bash
python -m pytest tests/vnext -q
# 1743 passed, 8 skipped in 1331.98s (0:22:11)
```

This `1743 passed, 8 skipped` result supersedes the earlier 1,516-era acceptance snapshot as the authoritative Task 118 integration result. The eight skips are explicitly limited to Windows symlink-privilege cases (`WinError 1314`) in `tests/vnext/test_atomic_quality.py`: four foreign-symlink prepared-recovery cases and four broken reserved-reparse transaction-marker cases. They are not hidden failures.

Fresh smoke:

```bash
python scripts/smoke_test.py
# SMOKE TEST: 30/30 passed
```

Fresh compilation used an explicit Python inventory script equivalent to:

```bash
python -m py_compile <all mub/vnext/**/*.py> \
  scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py \
  scripts/vnext_validate_artifacts.py scripts/smoke_test.py \
  scripts/prepare_data.py scripts/eval_evomemory.py \
  scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py \
  scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py \
  scripts/generate_constrained_sft.py scripts/train_constrained_sft.py
# py_compile_files=47, exit=0
```

### Focused deterministic and artifact validation

A same-destination, overwrite-enabled temporary fixture pipeline compiled dataset, results, all four mechanism kinds, and ledger twice. It ran the official validator for `tasks`, `task-manifest`, `task-runs`, `scores`, and `run-manifest`; parsed canonical task/run/score manifests and rows with their public models; and parsed all mechanism/ledger compatibility manifests with `LegacyAnalysisManifest`.

```text
artifact_files=15
deterministic_same_destination=PASS
validators=5
typed_reconstruction=PASS
```

An initial separate-root comparison was rejected during the acceptance script because authenticated manifests intentionally encode absolute output paths. The corrected same-destination check passed; contract authentication was not weakened. Temporary transaction lock files existed only inside the task-local `TemporaryDirectory` and were removed with it.

Focused authenticated-artifact and atomic-publication tests ran separately:

```bash
python -m pytest \
  tests/vnext/test_phase0_cli.py::test_dataset_cli_writes_canonical_tasks_and_exact_manifest \
  tests/vnext/test_phase0_cli.py::test_results_cli_writes_canonical_records_and_authenticated_manifest \
  tests/vnext/test_phase0_cli.py::test_mechanism_and_ledger_subcommands_emit_typed_compatibility_manifests \
  tests/vnext/test_phase0_cli.py::test_atomic_publication_rolls_back_staged_set_on_prepublish_failure \
  tests/vnext/test_phase0_cli.py::test_atomic_rejects_existing_destination_aliases -q
# 5 passed in 61.23s
```

Fixture record counts were two tasks, two runtime records, two score records, two conflict rows, two dose rows, two stale-removal rows, three API rows, and one ledger audit.

### Fresh schema hash ledger

| Checked schema | Task 118 SHA-256 |
| --- | --- |
| `mem_update_task.schema.json` | `d377d386296aac94731df81cc5de4fd21a43433538f2bc62ec36c3d945e8b4ec` |
| `run_manifest.schema.json` | `c09278d4c82ba7167abb27c37870038b69d13747b2cb30093115b390ee0576fa` |
| `score_record.schema.json` | `fddf41b727e699ed5a72b866b22f696be52727f6b35b5caa4d15e08eaddb0891` |
| `task_manifest.schema.json` | `e952f446cfd574dc82c2f4a072c6c3dd7a51bc5062d62a2b043c90238a152207` |
| `task_run_record.schema.json` | `5427ebc86482b5bfd4f6e2d41ad053552a1415b641ed384c3836f39d2275de2a` |

### Fresh fixture artifact hash ledger

These hashes are from the successful same-destination temporary compilation. Manifest hashes authenticate that temporary absolute path; payload hashes are path-independent.

| Relative artifact | SHA-256 |
| --- | --- |
| `tasks/tasks.jsonl` | `697e0808d13ef0ca0fa5deba8805d74088cf6a7ebfc488dfb654d1b147783f38` |
| `tasks/task_manifest.json` | `1b8bf07d2a4cf5799cf3f647a6ebfa8a33639f0e05055d2e1f4a9fc92ab7d9fb` |
| `results/task_runs.jsonl` | `afa94d67c1f22b991a08652fd0e136d1935e69fe4bea7d22bfed9f1209c816f3` |
| `results/scores.jsonl` | `96bb6d2017167a366b876bcbc58573814f85dc8fc16020f6b5301f0dc194c611` |
| `results/run_manifest.json` | `05b135ba99bdf9142caee0b5e664f6ed4a314661f7fd12b1f471069fb740cca1` |
| `conflict/legacy_analysis.jsonl` | `df2a5dcef527e608e68b4c141ee2dc00decbf2c833574c37747409d7dfee692b` |
| `conflict/legacy_analysis_manifest.json` | `cc574c982c17054af4f3574ef3318905ee27defcf8708357d6a57f612d36bc8d` |
| `dose/legacy_analysis.jsonl` | `2279272935e22b1f117ddf2a045c3e8814aef49bad97657499bb67b6f9dc96af` |
| `dose/legacy_analysis_manifest.json` | `0578dbee974de59851c278f5e06e236e651427390d65287c912852ca5db2f0e3` |
| `stale-removal/legacy_analysis.jsonl` | `9b4007859a7ca4747df2e086f3f2299f72328f3515d451266ed5f3f993cabc70` |
| `stale-removal/legacy_analysis_manifest.json` | `491505acd3836955a74c359f991dfdb397f292082137abe439be037df27b6344` |
| `api/legacy_analysis.jsonl` | `aa7a1193709204e7dc9de24d7788e8e86b4bb510396f2052f6003d5752728026` |
| `api/legacy_analysis_manifest.json` | `4a3d319de6eed37fe4c53b552ce380b0d0af8ecb8099934406efe9d587eaf964` |
| `ledger/ledger_audit.json` | `e40f77ee558792ac72e1b2f6d843f71cf4e5f786f34190598596ee07e55025f5` |
| `ledger/legacy_analysis_manifest.json` | `18d2a2d79f3edddb514bf2fe4807079ba75f9c2ba910cb85f187eca6c602af98` |

### Immutable source before/after ledger

All eleven source fixture hashes after every generation, validation, test, and smoke command exactly matched the pre-generation baseline:

| Immutable source | SHA-256 before and after |
| --- | --- |
| `evomemory_results_old.json` | `0070edab3bb643680b23a2de0f760364b16e5d7f14dc10ec12760a91b0bdf960` |
| `evomemory_results_traced.json` | `cdb3de2b4b01c7279817942ccc49b0e3776f37d2076173482a4e43f7880f1c74` |
| `ledger_fenced_references.md` | `5c82b215d3ccd400303de06c3ea37f3939fea908f5e18f1dc87f7d5f5bb46206` |
| `ledger_references.md` | `66e59c4ed7bf0340e1bea0b4e0d2d85a08ead75f9cc366f4341717687033b752` |
| `p63_dataset_minimal.json` | `18b4c1346da3c4ee723be89882c764df3fa3d99a45d0ec121009244909caf47f` |
| `p65_prompt_summary_minimal.json` | `b31dbf21ab53871cf4d814ed37748f1982b321b9cdb729df7fd887f4bee7f660` |
| `p83_conflict_rows.csv` | `88395320807df32898345511906e8116e3b5c4f59e09b6f3fe95bfcbf9b8e4e8` |
| `p83_stale_removal_rows.csv` | `f0990d9cc72ea48669ba1b0092814732af92a5f7a06fae057d8d5d5e676b4026` |
| `p83_synthetic_dose_rows.csv` | `97a2528eeb01208710bfeb0ebdaacde69799bd4b8d3f9c92899b6d2de3da36d9` |
| `p84_api_rows.csv` | `fad6fe12443580ef1e94f39c86d5d2cfcb4bb94eafe8a45124e6fd4dfaf0210e` |
| `p84_api_state_rows.csv` | `175f48206a69c6d43d43d32645d70ddf31ecb91faa8e8191deb6a238863dc45d` |

### Final repository and scope checks

```bash
git diff --check
git diff --cached --name-only
git status --short
python <Task118 final-newline/whitespace/debris/scope script>
```

Fresh result:

```text
git diff --check: exit 0
staged files: none
fixture_immutability=11 PASS
schema_changed=['mem_update_task.schema.json']
schema_newlines=PASS
whitespace=PASS
prohibited_debris=0
coordination_locks=1
no_pilot=PASS
```

Git emitted the existing LF-to-CRLF advisories for `.gitignore` and `WORKFLOW.md`; no unrelated file was normalized to hide them. The single `schemas/.mub-vnext-publish-<hash>.lock` is the current atomic publisher's intentional one-byte persistent coordination lock, verified against `_directory_lock`; it is not journal/tombstone/witness/stage/backup debris and was not deleted. No tracked `data/`, `results/`, or `paper/` file changed. No transaction journal, tombstone, witness, stage, temporary, or backup file remains. No file is staged, and no commit or push was made.

### Conclusion, limitation, and next step

Task 118 found and corrected one integration artifact: the checked `MemUpdateTask` schema was stale relative to the current typed `GeneratorProvenance` contract. Official regeneration, deterministic re-export, schema tests, the exact full suite, smoke, py_compile, focused artifact reconstruction, atomic tests, and source immutability all passed.

This is not a Phase 0 completion or acceptance declaration. The Pilot remains blocked. The next step is the coordinator-dispatched final whole-Phase-0 reviewer, which must decide whether the integrated Phase 0 snapshot is accepted.

## vNext Phase 0 final whole-phase gate

### Decision

The independent whole-Phase-0 reviewer returned:

```text
FINAL_APPROVED
```

This decision supersedes only Task 118's pending-review status. Task 118's fresh integration evidence remains authoritative:

```text
python -m pytest tests/vnext -q
# 1743 passed, 8 skipped in 1331.98s (0:22:11)

python scripts/smoke_test.py
# SMOKE TEST: 30/30 passed
```

The eight skips are the documented Windows symlink-privilege (`WinError 1314`) cases, not hidden failures. Available Windows junction/reparse and hardlink equivalents passed. Schema regeneration and byte-exact re-export, 47-file compilation, focused artifact validation, and all eleven immutable fixture before/after hashes also passed as recorded above.

### Accepted scope and publication boundary

Phase 0 is accepted as contract, validation, scoring, provenance, legacy-compatibility, and transactional-publication infrastructure. It produced no Pilot dataset, model result, external-validity result, benchmark metric, or new paper claim.

The approved Pilot plan now satisfies its Phase 0 prerequisite, but Pilot status remains `NOT_STARTED`. Beginning Pilot implementation requires a separate explicit instruction.

The accepted snapshot is published from the isolated `worktree-vnext-phase0` worktree. This publication does not merge or overwrite the independently dirty main workspace, whose newer P8.5/API content requires later manual or three-way reconciliation.

## vNext Pilot v2 reference-resolution contract gate

### Motivation

Family C requires explicit abstention gold for ambiguous and no-match entity/attribute references. Phase 0 v1 could not represent that distinction because `parsed_answer=None` meant unavailable evidence and replay required ordinary state-derived answers. The Pilot therefore added a generic v2 contract before resuming Family C instead of introducing Family-C-only sentinels or metadata dictionaries.

### Implemented boundary

- Added typed `AnswerDisposition`, `ReferenceResolutionStatus`, `ReferenceCandidate`, `SurfaceReference`, and `CanonicalAnswer` records plus `QueryType.UNRESOLVED_REFERENCE`.
- Preserved exact identity as `(namespace, entity, attribute, subkey)` and excluded `object_type`.
- Added strict structural/replay validation, semantic hashing of resolution graphs/outcomes, deterministic renderer support, explicit runtime disposition normalization, `reference_resolution_accuracy`, `wrong_reference_guess`, and `unjustified_abstention`.
- Kept `UNAVAILABLE`, missing rows, raw `None`, ordinary absent/deleted targets, and NOOP separate from abstention. No adapter capability bit was added.
- Regenerated all five vNext schemas with exact `2.0.0` top-level version constants. Published v1 artifacts remain immutable and are not silently migrated.

### Files changed

```text
mub/vnext/contracts/
mub/vnext/validation/
mub/vnext/io/canonical.py
mub/vnext/generation/{core,render,catalogs}.py
mub/vnext/runtime/
mub/vnext/scoring/
mub/vnext/failure.py
configs/vnext/pilot.yaml
schemas/vnext/*.schema.json
tests/vnext/
docs/specs/memupdatebench_vnext_benchmark_design.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md
WORKFLOW.md
```

No Pilot task data, result artifact, external API call, model result, or Family C core was generated by this gate.

### Verification evidence

Fresh focused controller runs completed at each reviewed boundary:

```text
Task 228 contracts/config/legacy regressions: 449 passed
Task 229 validation/replay/hash regressions: 495 passed
Task 230 renderer/Family A/Family B regressions: 484 passed
Task 231 runtime/scoring/failure regressions: 337 passed
Task 232 schema/compatibility focused suite: 149 passed
```

Each implementation task passed a specification review and a code-quality review before the next task began. Changed-module `py_compile` and `git diff --check` passed at the corresponding gates. Broader `tests/vnext` attempts exceeded the 600-second command window during this intermediate gate, so no new full-suite pass is claimed here.

### Conclusion and next step

The generic v2 contract can represent and score explicit unresolved-reference abstention without changing ordinary query semantics or rewriting v1 evidence. Family C remains not yet implemented at this checkpoint; the next step is the reviewed 120-core Family C grid, followed by its own structural, replay, hash, runtime, scoring, and audit gates.

## vNext Families A–D Pilot release and authenticated built-in runs

### Motivation and fixed scope

The approved Pilot implements Families A–D only: 480 semantic cores, three deterministic surfaces per core, and 1,440 tasks split 1,008/144/288 across train/dev/test. It reuses the Phase 0 v2 contracts and preserves exact object identity as `(namespace, entity, attribute, subkey)`; `object_type` remains classification metadata only. This phase uses no live API, external memory system, SFT, RLVR, LLM judge, or Families E–H.

The released task bundle is immutable at:

```text
/NAS/yesh/MemUpdateBench/data/vnext/pilot
```

Its current authenticated task, generation-manifest, and evidence-bound manifest hashes are:

```text
tasks.jsonl                  7573b635d8f72481e5f71630dc6101fef58a379fbda73ac81fef6788ce48bd2a
generation_task_manifest    2fe08afc00a7c6c7e23dc2905a6639c1d56d6cc28f0692898d6a5560c865fed6
evidence-bound task manifest b7d7f4169295df5fbcbda0b4be1d2cdc05ad436acb03b6c302137af2a7b59f27
```

The evidence-bound manifest is stored in the result checkpoint rather than replacing the immutable generation manifest. It authenticates the exact task bytes, generation config, generation manifest, split balance, both validation reports, audit sample, audit decisions, and audit gate report.

Two independent local release copies matched across all five generation artifacts. The user then supplied a detailed manual remediation review of the original 96-case audit, leading to revised surface rendering while preserving semantic cores, task identities, split assignments, and the four-part object key. Human reviewer Ye Shenghao subsequently reviewed all 96 regenerated audit IDs and supplied one terminal decision per ID. All 96 decisions have `verdict=pass`, all four checks true, no malformed, missing, duplicate, or foreign IDs, and `AuditGateReport.release_ready=true`. The human decisions SHA-256 is `eb4f4a1e74ec0fd2e4635664c2776b0875afa3fa9aeab8c80d9aff7aaf04a2df`; the gate report is `2183c579c82a88f70d15f5494b1917f4ce45a60607e983457d3b3b63d428de21`. The task, run, score, summary, release, and root provenance chains were rebound transactionally without changing task bytes, runtime rows, score rows, or metric values.

### Runtime-integrity corrections

The first authenticated execution exposed defects that smaller tests had not reached. Runtime outputs produced at `0a7d72d` and `2ab4e93` are diagnostic artifacts only and must not be interpreted as benchmark results. The task release's `code_revision=0a7d72d` remains valid generation provenance; it is distinct from the invalidated runtime behavior. The corrections were committed in this order:

```text
d676f8c  fix: unblock authenticated Pilot runs
2ab4e93  test: stabilize audit replacement regression
99be0f2  fix: bind mechanism slice to query target
ca47df7  fix: preserve observed Pilot actions
```

The final `ca47df7a6401fabfc25dd4d2151a392439e6c379` correction:

- parses values only from the suffix after the canonical `object(...)` span, preventing `to`, `as`, or `:` substrings inside entity names and prose from being mistaken for values;
- supports every released atomic-write surface, including `so each value is ...`;
- records structured observed operation/key/value/format evidence in `AdapterActionLog.raw_action`;
- removes runtime fallback to gold key/value during action normalization;
- retains correct typed Family C abstention as a completed answer rather than an adapter error;
- binds run identity and manifest provenance to an exact clean Git revision;
- loads one reviewed offline MiniLM encoder for heuristic CRUD; and
- binds the Family A mechanism slice to the query target rather than rejecting legitimate distractor objects in `task.target_objects`.

Focused verification on the final source included:

```text
194 focused adapter/runtime/resume/CLI/scoring tests passed
SMOKE TEST: 31/31 passed
7 mechanism-slice tests passed
py_compile passed
git diff --check passed
```

The full released-event audit over all 1,440 tasks reported:

```json
{"operation":0,"key":0,"value":0,"format":0}
```

The first complete Tang-2 suite after `ca47df7` reported `2699 passed, 6 skipped, 2 failed`. Both failures were stale test assumptions rather than production defects: one hard-coded the old profile version, and one expected an accidental Family B answer overlap that the audited data remediation intentionally removed. The profile test now uses the canonical `PROFILE_VERSION`; the Family B test constructs its overlap adversarially instead of requiring released data to contain a leak. The two corrected tests passed, followed by `133 passed` across both affected files. A fresh complete post-correction Windows suite then reported:

```text
2697 passed, 10 skipped in 3860.87s (1:04:20)
```

The ten skips are documented Windows symlink/junction privilege cases; there were zero failures. Together with the Linux suite's coverage of the corresponding platform cases, this closes the complete `tests/vnext` gate without modifying runtime code or formal result bytes.

### Formal execution and artifacts

The exact clean source revision was deployed to:

```text
/NAS/yesh/MemUpdateBench/.vnext-experiment-source
```

The new revision-qualified result root is:

```text
/NAS/yesh/MemUpdateBench/results/vnext/pilot_ca47df7_evidence_bound
```

The built-in matrix contains four methods under both `normal_topk` and `latest_per_object`:

```text
reference
raw_add
exact_crud
heuristic_crud
```

The verified heuristic encoder is the local `all-MiniLM-L6-v2` snapshot at reviewed revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Its capability probe returned dimension 384 with finite, nonzero embeddings. Song-2's shared project path entered an uninterruptible NFS wait, so the authenticated jobs were executed from a healthy client of the same approved NAS project path; this was a storage-path issue, not a model or GPU-memory substitution.

Every run contains 1,440 unique `TaskRunRecord` rows, an authenticated final run manifest, exact revision `ca47df7...`, and `dirty_state=false`. Status counts are:

| method | normal completed / unsupported / failed | latest completed / unsupported / failed |
| --- | ---: | ---: |
| reference | 1440 / 0 / 0 | 1440 / 0 / 0 |
| raw_add | 1080 / 360 / 0 | 1080 / 360 / 0 |
| exact_crud | 1080 / 360 / 0 | 1080 / 360 / 0 |
| heuristic_crud | 1080 / 360 / 0 | 1080 / 360 / 0 |

All 360 unsupported rows for each non-reference method are Family C `answer/multi_object_answer` capability exclusions. There are no parser-driven partial rows and no encoder-related unsupported rows.

Each of the eight run cells was authenticated, scored, and summarized against the human-rebound evidence manifest. Every cell contains 1,440 score rows and a six-file summary bundle whose artifact index matches the written bytes. Rebinding reproduced all eight `scores.jsonl` files byte-for-byte and preserved every aggregate metric. The root `artifact_index.json` binds the release, mechanism, corrupted-control, trace-review, and eight run/score/summary manifest chains; its final SHA-256 is `d9ef2cebc74a5445863de0ef047c9528cc01eab89354ca93b51917a5f2d0322b`. The release index SHA-256 is `2d2adf8f16d058e5b4d9346829a4f6a48c71aa3140f634a69586c3644951cc3a`. The principal built-in diagnostic metrics are:

| policy | method | current recall@k | stale exposure | final state accuracy | answer EM | final memory size | obsolete versions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | raw_add | 0.975 | 1.000 | 1.000 | 1.000 | 10.6000 | 8.6375 |
| latest per object | raw_add | 1.000 | 0.000 | 1.000 | 1.000 | 10.6000 | 8.6375 |
| normal | exact_crud | 1.000 | 0.000 | 1.000 | 1.000 | 4.6750 | 0.0000 |
| latest per object | exact_crud | 1.000 | 0.000 | 1.000 | 1.000 | 4.6750 | 0.0000 |
| normal | heuristic_crud | 1.000 | 0.000 | 1.000 | 1.000 | 4.6750 | 0.0000 |
| latest per object | heuristic_crud | 1.000 | 0.000 | 1.000 | 1.000 | 4.6750 | 0.0000 |

These values use deterministic `slot_direct`, not a prompted answer model. The `reference` adapter is intentionally excluded from leaderboard aggregation as `oracle_smoke_only`; its underlying score records independently verify action exact match 1.000 over 1,440 tasks, final-state accuracy 1.000 over 1,440, answer EM 1.000 over 1,080 ordinary-answer tasks, and reference-resolution accuracy 1.000 over all 360 Family C tasks.

Representative trace inspection found 27 Raw-append `normal_topk` current-retrieval misses: 12 medium and 15 hard Family B tasks. In each inspected miss, normal retrieval returned stale versions of the queried object while omitting the current value; `latest_per_object` preserved the query target, inserted the current value, and removed stale exposure. A Family C reference trace showed typed `ABSTAINED`, `format_valid=true`, and `completion_status=completed`; the corresponding non-reference trace showed the declared `multi_object_answer` capability exclusion. Family D traces preserved natural NOOPs as observed `NOOP` actions with null key/value and correctly parsed later writes.

The retained `trace_review/trace_review.json` expands this check to all 48 method × family × difficulty cells under normal retrieval, with the same task's latest-per-object counterpart attached whenever a failure-flag example exists. It contains outcome-correct examples for 39 cells and failure-flag examples for 42 cells. The nine missing correct cells are exactly the three Family C capability-exclusion cells for each non-reference method; missing failure examples occur only when no failure-flag row exists in that cell. The review artifact and its task input are authenticated by `trace_review/artifact_index.json`.

### Scorer controls and mechanism slice

Eight corrupted controls were run and retained on the deterministic 12-task, four-core no-network smoke release under:

```text
results/vnext/pilot_ca47df7_evidence_bound/corrupted_controls
```

All eight expected scorer failures were detected from observed output rather than gold-filled actions. Each control retains its own run manifest, score manifest, and summary artifact index; `corrupted_controls/artifact_index.json` binds those manifest chains and the consolidated check report:

```text
false_write
missed_update
current_not_retrieved
gold_retrieved_wrong_answer
invalid_action_format
stale_copied
wrong_attribute
wrong_entity
```

The corrected mechanism slice is stored under:

```text
results/vnext/pilot_ca47df7_evidence_bound/mechanism_slice
```

It contains 48 deterministic smoke contexts: six conditions, eight examples per condition, stale counts 1 and 16, chronological/reverse order, and optional latest/outdated labels. It is explicitly marked `smoke_only=true`, `not_model_result=true`, and `answer_model=deterministic_reference_smoke`. `mechanism_slice/artifact_index.json` binds both `contexts.jsonl` and `condition_manifest.json`.

### Conclusions and limitations

The Pilot release and built-in execution path now provide authenticated evidence that the task, runtime, scoring, capability, summary, control, and mechanism contracts compose correctly. The Raw-append retrieval comparison demonstrates the intended diagnostic separation: latest-per-object changes retrieval exposure without compacting the underlying append-only store.

This is not external-validity or prompted-answer evidence. Exact and heuristic CRUD are identical on this controlled release because the released atomic object surfaces expose exact keys; MiniLM readiness is verified, but this matrix does not establish a learned semantic-resolution advantage. Raw append's `slot_direct` answer remains correct even in the 27 normal-retrieval misses, so these rows must not be used to claim answer-model robustness. Family C support remains reference-only for the current built-ins and must be reported as capability coverage rather than silently dropped or treated as failure.

An independent final-gate review initially returned `NOT_APPROVED` because the retained audit was not human-attributed, release-level evidence was not fully bound, documentation contained stale pre-regeneration hashes, the complete trace matrix was absent, and the final full suite had not been rerun. The evidence-bound rerun, corrected hashes, nested artifact indices, retained 48-cell trace review, human-attributed 96-case audit, and green complete post-correction suite resolved every blocker. A final independent review returned `CONDITIONAL` with exactly one condition—the fresh complete suite must finish with zero failures—and explicitly found no other release blocker. The `2697 passed, 10 skipped` result satisfied that condition.

The Families A–D Pilot is therefore `FINAL_APPROVED` as a bounded benchmark-engineering release. This approval authenticates the released tasks and the built-in deterministic engineering evidence; it does not convert `slot_direct` diagnostics into external-system, prompted-answer, API, SFT, RLVR, or broad benchmark evidence. Any later Core expansion, external systems, prompted answer models, APIs, Families E–H, SFT, or RLVR require a separate approved design/implementation cycle and new result roots. The pre-human-rebind server metadata backup remains retained for rollback until a later explicit cleanup decision.

## vNext Core immutable task release

### Motivation

Core Task 8 closes the pre-runtime data boundary for the exact 3,000-core/12,000-task A–G release. The release required a full 224-task human audit, independent dual review for E–G, adjudication of every non-pass item, generator/template remediation rather than decision rebinding, full regeneration, and a fresh audit over the regenerated candidate before immutable publication.

### Human audit and remediation

The first complete audit contained 320 decisions: 224 primary rows and 96 independent E–G secondary rows. The authenticated gate required 104 adjudications. A third independent reviewer confirmed 90 remediation items: 32 Family D controlled-adversarial NOOP sentence splices, 26 Family E surface-naturalness failures from the same template root cause, and 32 Family G lowercase post-period query instructions. The other 14 adjudications passed.

The generator remediation was committed at:

```text
ba8444bd6db5d4a15eeb0062096d715c77016c86
fix: remediate Core surface audit findings
```

The Core controlled NOOP template now renders the reviewed statement as a complete sentence before the lifecycle caveat. Family G explicit and controlled query templates now use grammatical comma-linked instructions rather than inserting lowercase text after a period. Two focused tests were observed RED (`2 failed`) before the fix and GREEN afterward (`2 passed in 14.28s`). The five relevant Core modules then passed `96` tests in `941.66s`; `py_compile` and `git diff --check` passed.

The remediated full candidate passed standalone validation with exactly 3,000 semantic cores, 12,000 tasks, 8,400/1,200/2,400 train/dev/test tasks, and a 140-core/560-task `core-hard-v1`. An exhaustive surface scan checked all 900 Family D/E controlled surfaces and all 600 Family G explicit/controlled queries, finding zero instances of either confirmed defect.

Because the generation revision changed, every task hash and audit ID was freshly bound even though deterministic reselection chose the same 224 semantic cores. No prior decision was rewritten or mechanically rebound. The regenerated audit collected a new 320 decisions with non-overlapping observation fingerprints. All 224 primary decisions and all 96 independent E–G secondary decisions passed.

### Final gate and publication

The authoritative final gate returned:

```text
status: RELEASE_READY
full_candidate_at_verification: true
release_ready_at_verification: true
terminal_pass_count: 224
required_adjudication_count: 0
unresolved_adjudication_count: 0
remediation_count: 0
raw_agreement: 1.0
```

`Cohen's kappa` is null in the all-pass E–G re-audit because both reviewer marginals are constant; raw agreement is 1.0 and adjudicated terminal pass, rather than kappa alone, is the release criterion.

The authenticated task release was published by staging the complete tree beside the destination, verifying the final gate attestation and current candidate receipt, copying and hashing every artifact, and atomically renaming the staged directory to:

```text
data/vnext/core/v3
```

The immutable tree contains the exact seven candidate artifacts under `candidate/`, final audit selection/context/decisions/gate evidence under `audit/`, five strict-v3 schemas under `schemas/`, and the root `task_release_manifest.json`. The root manifest binds 18 artifacts and records:

```text
candidate generation revision  ba8444bd6db5d4a15eeb0062096d715c77016c86
candidate root digest           71a6beb3ac8a28dabc753c969e96a47a59f92031d217bebf0fa63d6061012af1
source task manifest hash       38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3
selection hash                  d2d5260c164ec72c826d6c692c13518269db05fc1f5dbae0ff057f78dad796da
gate attestation hash           45461659ab3f65a0a559897e50340a470f27cdecf55b999a1431988567cf00c2
release root digest             458d169a4732139f45361d90ea528f5ed0133f126a32bc5a16de23da6f8a2aba
release manifest hash           f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d
```

### Conclusion and next step

The Core **task release** is `FINAL_APPROVED`: its task bytes, strict-v3 schemas, hard-suite view, human evidence, candidate provenance, and publication root are authenticated. This is not overall Core `FINAL_APPROVED` and contains no built-in manager result, external-system result, prompted answer-model result, confidence interval, claim ledger, SFT, RLVR, API evidence, Family H, hidden test, or leaderboard evidence. Task 9 may now consume this immutable task release to extend built-ins, support resolution, and corrupted controls without regenerating or rebinding the task boundary.

## vNext Core built-ins and corrupted controls

### Motivation and fixed boundary

Core Task 9 extends the deterministic engineering checkpoint from Families A–D to the immutable strict-v3 A–G task contract. The goal is truthful capability coverage and diagnostic controls, not a benchmark result: every requested task must produce one ordered terminal row; unsupported work must remain explicit; runtime decisions must use visible task surfaces rather than gold action, history, or evidence fields; and state, lifecycle/history, retrieval/evidence, and answer layers must remain distinguishable.

This work consumes but does not modify, regenerate, rebind, or add files beneath `data/vnext/core/v3`. It adds no external adapter, prompted answer model, statistics, claim ledger, API call, SFT, RLVR, or overall Core release decision.

### Implemented runtime and support boundary

The strict-v3 runtime now includes one central support resolver and built-in adapters for:

```text
reference
raw_append
exact_crud
heuristic_crud
```

The resolver derives foundational runtime, operation, query-selector, and metric-export support from `MemUpdateTaskV3` requirements plus `AdapterCapabilitiesV3`. It checks isolated reset and event ingest, native-answer capability when requested, ADD/UPDATE/NOOP/DELETE, TTL and scoped deletion, historical selector semantics including selectors nested in Family G synthesis queries, multi-object answering, and truthful export capabilities. Unknown answer modes fail configuration validation. Requested retrieval policy is bound to adapter-declared or trace-observed effective policy; mismatched, blank, or unbound policies fail closed as `NOT_SUPPORTED` before answering.

`execute_tasks_v3` materializes the request sequence once and preserves exactly one `TaskRunRecordV3` per requested task in request order, including adapter-factory, support, reset, ingest, answer, and close outcomes. Snapshot hashing uses canonical JSON-compatible data. Snapshot capture respects export capabilities: full snapshots require entry and raw-state exports, entries-only adapters produce truthful entry snapshots, and adapters without entry export skip snapshots rather than fabricating an empty store or failing otherwise supported execution.

All non-reference built-ins parse the four frozen visible surfaces and do not use `task.actions`, `gold_action_ids`, `version_history`, or `gold_evidence` for runtime decisions. Canonical identity remains exactly `(namespace, entity, attribute, subkey)`; `object_type` remains excluded.

### Built-in semantics

- Reference supports Core A–G semantics and typed unique/ambiguous/no-match reference-resolution scoring. Gold access remains isolated to this explicit oracle adapter.
- Raw append retains ADD/UPDATE versions and separate DELETE/TTL instruction or logical tombstone evidence. Physical state is not deleted: requested DELETE maps to effective NOOP with `NO_EFFECT` and reason `append_only_no_physical_delete`. Retained forgotten and stale values remain observable for diagnostics, and version/history exports use logical time and valid source anchors.
- Exact CRUD implements visible exact-key current state, logical-time TTL, scoped multi-target deletion, protected collateral, and current multi-object Family G answers.
- Verified MiniLM Heuristic CRUD requires the reviewed model/backend/revision and a finite nonzero encoder probe. On the frozen exact-key Core surfaces it may behave identically to Exact CRUD; this gate establishes readiness and provenance, not a semantic-resolution advantage.
- Exact and Heuristic CRUD do not claim a historical version ledger. Historical selectors and dependent fields are therefore typed unsupported, while direct current-state Family F accuracy remains measurable.
- Evidence citations are empty when adapters do not export real evidence linkage; no gold citations are fabricated.

The metric registry now gates capability requirements by exact metric path where layer-wide defaults would be false. In particular, current-state accuracy does not require historical-query support, genuine historical fields still do, and Raw-append forgotten exposure/leakage diagnostics depend on observable retrieval/value/native-answer artifacts rather than physical-delete capability.

### Corrupted controls

Eight strict-v3 controls transform real completed reference rows and reuse the existing strict-v3 scorer and failure taxonomy:

```text
wrong_delete_scope          -> wrong_delete_scope
missed_ttl                  -> ttl_violation
collateral_deletion         -> collateral_corruption, collateral_mutation
retained_forgotten_value    -> deletion_failure, stale_retained
wrong_historical_version    -> version_confusion
wrong_history_order         -> version_confusion
stale_g_propagation         -> stale_propagation
fabricated_evidence         -> evidence_linkage_error
```

Every control has a stable ID and target layer, is `smoke_only=true`, and is `leaderboard_eligible=false`. The fabricated-evidence control changes citations without changing the parsed answer or raw output.

### Files changed

```text
mub/vnext/adapters/core_v3.py
mub/vnext/adapters/corrupted_v3.py
mub/vnext/runtime/engine_v3.py
mub/vnext/runtime/support_v3.py
mub/vnext/scoring/registry_v3.py
mub/vnext/scoring/scorer_v3.py
tests/vnext/test_core_corrupted_controls_v3.py
tests/vnext/test_core_generation_family_f.py
tests/vnext/test_core_runtime_v3.py
tests/vnext/test_v3_replay_scoring.py
WORKFLOW.md
```

The implementation commit is:

```text
9118d491fb3f13a2b4278f131fd2520f9c4fe809
feat: extend Core built-ins and controls
```

### Verification and review evidence

Focused parent-worktree gates after the final fixes reported:

```text
Core runtime/control/strict-v3 scoring and Pilot-v2 compatibility: 408 passed
Family C/E/F/G plus strict-v3 focused gate:                    484 passed
Additional Pilot/v2 adapter/runtime compatibility gate:         78 passed
py_compile:                                                    passed
git diff --check:                                              passed
```

The complete `tests/vnext` inventory was run in bounded shards because the Windows suite exceeds the command window. Aggregating every test file after the final code commit produced:

```text
3368 passed
13 skipped
6 warnings
0 failures
```

The 13 skips are Windows symlink/junction privilege cases. The six warnings are the existing Pydantic serializer warnings in generation-common tests. Before the implementation commit, 16 Core staging/release tests intentionally stopped at `tracked Core source differs from the anchored HEAD revision`; this demonstrated their clean-tree guard rather than a product regression. After commit `9118d491...`, that exact shard passed `110 passed, 3 skipped`, closing the clean-tree gate.

Implementation used observed RED/GREEN cycles for iterator/factory completeness, raw DELETE coherence, visible-surface fail-closed parsing, MiniLM revision verification, reset/close preservation, canonical snapshots, TTL history anchors, all eight controls, typed Family C resolution scoring, nested historical selectors, exact metric capability gates, foundational runtime capabilities, optional snapshots, answer-mode validation, and declared/observed retrieval-policy binding.

The required ordered reviews completed as follows:

1. Specification review returned `NOT_APPROVED` for null Family C reference-resolution scoring, missed nested G historical selectors, incorrect current-state historical gating, and Raw forgotten diagnostics gated on physical delete. All four were fixed and the same specification reviewer returned `APPROVED`.
2. Only after specification approval, code-quality review found missing foundational capability gates, unconditional optional exports, and unbound retrieval policy. These were fixed with regressions. A final direct code-quality re-review returned `APPROVED`.

Several reviewer streams terminated with API `unexpected EOF`; none was accepted as a completed review. Each was resumed or replaced until a definitive report was returned.

### Conclusion and next step

Core Task 9 is complete as a deterministic built-in/runtime/scorer-control engineering gate. It establishes truthful A–G support resolution, explicit unsupported coverage, exact terminal-row completeness, logical lifecycle semantics, diagnostic Raw retention, current-state CRUD behavior, and scorer sensitivity controls.

This is not an authenticated 12,000-task result matrix and does not make overall Core `FINAL_APPROVED`. It provides no external-validity or prompted-answer evidence. The next authorized work is Task 10: capability-first qualification of one genuine external adapter, beginning with the fixed Mem0 OSS feasibility gates and using the explicitly labeled LangGraph Store fallback only if those gates fail.

## vNext Core external-admission boundary

### Motivation and fixed scope

Core Task 10 Phase 1 establishes the provider-neutral trust boundary that must exist before any genuine external system is installed, invoked, or interpreted. It fixes the candidate vocabulary, evidence gates, fallback state machine, report identity, and portable provenance rules independently of Mem0, LangGraph, Qdrant, Transformers, or SentenceTransformers. The base package imports none of those optional providers.

The only primary candidate is `mem0_oss`. The only conditional candidate is `langgraph_store_extract_then_store`, and it may participate only after a current-manifest/current-evaluation Mem0 report records a candidate-specific eligible `FAIL`. Shared resource `BLOCKED` or `NOT_RUN` outcomes do not count as Mem0 failure and do not authorize fallback. Memory-R1, `baselines/memory_r1_agent.py`, project-local approximations, and the historical `scripts/eval_mem0_baseline.py` path are explicitly inadmissible as external-system evidence.

This phase does not run either provider, create a scientific result, compare accuracy, start the answer-model harness, or change the immutable Core task release.

### Contracts, gates, and selection semantics

The independent immutable admission contract fixes fourteen ordered gates:

```text
source_authentication
official_provenance_license
offline_model_prerequisite
candidate_environment
visible_only_fairness
namespace_reset
capability_truthfulness
raw_normalized_export
field_provenance
terminal_completeness
retrieval_policy
presentation_level
security_redaction
repetition_rule
```

Each gate is exactly one of `PASS`, `FAIL`, `BLOCKED`, or `NOT_RUN`, with status-coherent evidence and reasons. Reports bind the authenticated source task-manifest hash and reference, shared evaluation-configuration hash and reference, candidate-specific adapter configuration, probe, canary, package/model provenance, exact adapter identity, capabilities, state-transition linkage, fixed gate vector, aggregate outcome, and canonical reason.

Admission requires all gates to pass, a recomputed presentation level of 2 or 3, and real event-ingest/add/update support. Extractor-dependent candidates must bind a coherent extractor ID/version rather than silently inheriting benchmark gold. Public evaluation, fallback, and exact-one selection APIs revalidate exact stored fields and nested contract types so `model_construct`, subclass overrides, raw enum/string substitution, malformed strict values, and forged nested references fail closed.

Fallback eligibility and invariant gates are disjoint and exhaustive. Invariant provenance, license, offline-model, security, and repetition gates must pass; eligible gates must be terminal `PASS`/`FAIL`; at least one eligible Mem0 gate must fail; and any `BLOCKED` or `NOT_RUN` prevents fallback. Selection emits typed report references in canonical Mem0-then-LangGraph order and returns either exactly one admitted candidate or a canonical release-stopped decision. A `langgraph_fallback_not_authorized` reason is invalid unless a LangGraph report actually participated.

### Portable provenance boundary

Artifact references must use NFC-normalized relative forward-slash paths. The validator rejects absolute, drive, UNC, backslash, traversal, empty, trailing-dot/space, colon, Win32 device, DOS 8.3 alias, control/surrogate, default-ignorable, and unsafe punctuation forms. Prior-proxy matching uses NFKC/casefolded component tokens, iterative extension stripping, and contiguous multi-component windows, including a denied path embedded as an ancestor directory. This closes direct, prefixed, suffixed, compound-extension, Unicode-format, and ancestor-path evidence aliases without treating broad substrings such as `memory_r10` as the denied `memory_r1` token.

### Files and implementation checkpoint

```text
mub/vnext/external/__init__.py
mub/vnext/external/admission.py
mub/vnext/external/contracts.py
mub/vnext/external/registry.py
tests/vnext/test_external_admission_contracts.py
tests/vnext/test_external_task10_boundaries.py
```

The implementation and review-remediation commits are:

```text
c74d20d2b8aa63a5ad35182199f9ba49b5df2a9f
feat: establish Core external admission boundary

694526a25b10513933a9fc1c9a11d4754c125f79
fix: enforce Mem0-first external admission
```

### Verification and review evidence

Observed TDD cycles covered fixed gate/order semantics, configuration and artifact identity binding, Level 2/3 admission, extractor coherence, conditional fallback, duplicate/report-order rejection, typed decisions, exact nested revalidation, provider-neutral imports, immutable schema snapshots, portable Windows/Unicode paths, and prior-proxy aliases. The final two RED cases reproduced a denied script path used as an ancestor directory and a release-stop rationale without LangGraph participation (`2 failed, 32 deselected`); both passed after the minimal production fixes (`2 passed, 32 deselected`).

Post-commit parent-worktree verification reported:

```text
external admission contracts and boundaries:  38 passed
external/Core runtime compatibility matrix:   361 passed
py_compile:                                    passed
git diff --check:                              passed
immutable Core/legacy/schema diffs:            empty
```

The complete `tests/vnext` inventory was started after the commit but intentionally stopped after its Windows runtime made it unsuitable as a single command-window gate. Task 9 had already completed the prior bounded full inventory at `3368 passed, 13 skipped, 0 failures`; this Phase 1 checkpoint therefore uses the 361-test affected compatibility matrix, with the complete suite reserved for the final Task 10 batch gate after the remaining provider-neutral and provider-specific layers are integrated.

Specification review and security/provenance review both returned `APPROVED`. The first completed code-quality review, performed by `gpt-5.6-sol`, returned `NOT_APPROVED` for the ancestor-directory denial bypass and the inconsistent release-stop rationale described above. Both findings were independently reproduced, fixed test-first, and closed by the post-commit regression matrix.

A fresh review of the exact implementation commit then found two additional public-boundary inconsistencies: direct `evaluate_candidate_admission()` could return true for LangGraph without Mem0 fallback context, and a directly constructed decision could admit Mem0 while also carrying a LangGraph report. Each finding was reproduced by an isolated failing regression, fixed without changing the valid fallback selector path, and committed in `694526a2...`. The final matrix increased to 361 passing tests.

Two minimal delegated re-review attempts returned `APPROVED`, but their recorded response models were Opus and Fable rather than the selected budget routes, so those verdicts were not adopted. The parent `gpt-5.6-sol` session independently re-read the exact remediation diff, confirmed both failing scenarios against the pre-fix parent, ran the two-case GREEN regression and the complete 361-test affected matrix, and found no remaining actionable issue in the remediation scope.

### Conclusion and next step

Task 10 Phase 1 establishes an admission boundary, not external-validity evidence. No candidate is admitted yet, and overall Core remains in progress. The next bounded step is to derive two authenticated, deterministic, mutually independent 64-task dev canaries from the immutable Core release without writing beneath `data/vnext/core/v3`; provider probes and Mem0 implementation remain gated on that and the remaining visibility, security, persistence, and model-provenance work.

## vNext Core external capability canaries

### Motivation and release boundary

Core Task 10 Phase 2 derives two authenticated 64-task dev canaries for capability, reset, visibility, and genuine-provider qualification. These are deterministic views over the immutable Core task release, not new semantic cores or a replacement task release. No file beneath `data/vnext/core/v3` was regenerated, rebound, overwritten, or added. The derived view provides no external-system, manager-quality, prompted-answer, statistics, or overall Core `FINAL_APPROVED` evidence.

The implementation is provider-neutral. Importing `mub.vnext.external` does not import Mem0, LangGraph, Qdrant, Transformers, or SentenceTransformers.

### Authenticated source and selection policy

The derivation authenticates the exact approved release tree, including canonical manifest bytes, self-hash, exact file and directory sets, regular single-link files, every declared artifact size/hash, the release-root digest, strict `TaskManifestV3`, all 12,000 canonical task rows, task order, record hashes, and split counts. Only the 1,200 dev rows are retained after global validation.

```text
source release manifest hash  f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d
source task manifest hash     38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3
source tasks artifact hash    5c4fd518542b0665d7313d68f1a339de38502c376aa93fbda228196587cdd2c6
selection policy hash         23238afe7c34d9acd01da8c3ea6b81ba556aa8aaf27c48f811899d122d762737
selection version             core_task10_phase2_canary_selection_v1:23238afe7c34d9acd01da8c3ea6b81ba556aa8aaf27c48f811899d122d762737
```

Canary A and Canary B each contain exactly 64 dev tasks with fixed A/B/C/D/E/F/G quotas `8/8/8/8/12/12/8`. Selection is coverage-first, then domain-separated SHA-256 fill; no rank uses answer values. Each canary covers ADD, UPDATE, NOOP, Family E object/attribute/entity/namespace/TTL deletion, Family F current and historical selectors, multi-object queries, and Family G query/synthesis behavior.

The canaries are disjoint across task ID, semantic core, trajectory, version group, source group, source document, and paraphrase group. The fixed Core dev release has only six Family F trajectory/version groups, so those groups are deterministically partitioned three-and-three before task-level selection; reuse within one canary is allowed only where required by that authenticated capacity boundary.

### Derived artifacts

The canonical derived view is under the ignored result root:

```text
results/vnext/core_task10_canaries_v1/
  canary_a/tasks.jsonl
  canary_a/canary_manifest.json
  canary_b/tasks.jsonl
  canary_b/canary_manifest.json
  canary_set_manifest.json
```

```text
canary set manifest SHA-256  3c822b014af2b1026056f81b9284bbb6a4ed52d9072ac5524c7aa2fb6c8f95a8
Canary A manifest SHA-256    95a1242d7c4f49019feaa540fee9763fd0157fc98249ba1c5bb125e612a71086
Canary A tasks SHA-256       d62cfda6d0790658ac39057e70658e0319fb8f7a2d395ec9f8cb46a6299aad39
Canary B manifest SHA-256    6e538beea5b25e5e41d5e23e6d2fefca444e784c85d3030799548cf65c5b9de5
Canary B tasks SHA-256       13c97507bef40127e54d1aaa110f1f05b208bbf5c48ff1b10d27ac654a9f0308
```

A fresh authenticated rebuild matched all five existing artifact files byte-for-byte. Each selected manifest row binds the source record hash, and each derived JSONL line is the exact original canonical source line.

### Publication and trust boundaries

Public build, validation, and publication entry points reauthenticate the supplied release root and rebuild the expected canonical selection; caller-constructed snapshots and alternate authenticated-but-noncanonical selections cannot substitute records. Nested mappings are read-only, policy sources have no mutable dictionary backing reference, and manifest byte fields require exact immutable `bytes`.

Publication requires a new caller-owned destination outside immutable Core. It rejects existing and dangling destinations, static symlink/reparse components, changed parent identity, unanchored staging, hard-linked staged files, extra or missing staged entries, and any staged byte mismatch. Files and directories are fsynced; installation uses platform-native no-replace rename on Windows, Linux, and macOS and fails closed elsewhere. Staging identity and the complete staged tree are revalidated immediately before installation, and failed publication cleans its staging directory.

### Implementation commits

```text
a342273bf9e6ebba91e07a984ccae8abecb53271
feat: derive authenticated Core external canaries

dbb058d7f474dcbcfad911b85403f6e5e6d748e4
fix: harden Core canary publication
```

### Validation and review evidence

Observed RED regressions reproduced mutable policy backing, mutable `bytearray` manifest payload acceptance, and staged-tree tampering after fsync. All three failed before remediation and passed afterward. The final bounded Windows gate reported:

```text
canary tests across bounded shards:          17 passed, 1 skipped
external/Core runtime compatibility matrix: 361 passed
py_compile:                                  passed
git diff --check:                            passed
immutable Core/legacy/schema diffs:          empty
canonical artifact rebuild:                  byte-identical
```

The one skip is the Windows symlink/reparse construction case because the current account lacks symbolic-link privilege (`WinError 1314`); static reparse rejection remains implemented and is exercised where the OS permits link creation.

A direct `gpt-5.6-sol`/medium review first approved the four target files, then corrected its verdict after delayed review notifications exposed four reproducible issues: mutable policy backing, staged-directory identity drift, missing pre-install staged-tree verification, and mutable manifest bytes. The parent `gpt-5.6-sol` session independently reproduced all four mechanisms, fixed them test-first, re-read the complete remediation diff, and ran the final gates above. A routed continuation produced only a failed Fable response with no verdict and was not adopted. No actionable issue remains in the scoped remediation after parent review.

### Conclusion and next step

Task 10 Phase 2 is complete. The two canaries are authenticated deterministic diagnostic views only; no genuine external candidate is admitted yet. The next bounded step is Task 10 Phase 3: provider-neutral visible-only DTOs, subprocess bridge, 20-run namespace/reset probes, capability truthfulness, deterministic classification, private/raw versus redistributable/normalized artifact separation, and generalized secret/reparse/root-containment controls.

## vNext Core external capability and security gates

### Motivation and scope

Core Task 10 Phase 3 establishes the provider-neutral execution boundary required before a real Mem0 or conditional LangGraph worker can receive benchmark-visible text. It does not install a provider, run the canaries, admit a candidate, start prompted answering, or change strict-v3 schemas. The base external facade remains importable without Mem0, LangGraph, Qdrant, Transformers, or SentenceTransformers.

### Visible-only worker inputs

`ProviderEventInputV1` contains only event ID, sequence index, visible logical time, raw visible text, and the run-derived namespace. `ProviderQueryInputV1` contains only query ID, visible query text, native top-k, and runtime namespace. Conversion revalidates exact `MemoryEventV3`/`MemoryQueryV3` instances and returns rebuilt trusted models rather than forwarding benchmark task objects.

A recursive fairness guard rejects action, gold, target-object, selector, answer, current/ordered/version history, stale alternative, evidence, expected-effect, derivation, and stratification keys across nested mappings and lists, including NFKC/case/separator variants. Worker requests therefore cannot carry gold actions, gold object IDs, benchmark selectors, history ledgers, gold evidence/answers, or derivation plans.

### Subprocess bridge

The persistent JSONL bridge uses exact immutable request/response contracts, canonical UTF-8 JSON lines, explicit request IDs, a fixed operation vocabulary, no shell, an absolute executable, a real non-reparse working directory, and an explicit environment allowlist. Credentials in command arguments are rejected; approved credentials may be supplied only through explicitly allowed environment names and are never rendered in errors.

Responses are bounded before full-line allocation, canonical-JSON checked, request-ID bound, and fail closed on malformed data, unexpected extra output, EOF, timeout, broken pipes, nonzero/terminal process state, or mismatched IDs. Stderr is drained but never copied into normalized errors. Provider parsing failures suppress raw exception causes, and close/timeout paths terminate the worker, close pipes, and join reader threads.

### Reset, determinism, and capability truthfulness

The namespace probe runs exactly 20 independent target/control trials. Each trial resets both fresh namespaces, writes unique sentinels, verifies bidirectional isolation, resets only the target, verifies that the target is empty while the control remains unchanged, and then cleans only namespaces under the current run prefix. Every trial receives a terminal typed row even when the backend raises or returns an invalid type; raw provider exceptions are reduced to fixed error codes. Admission requires 20/20.

Determinism classification uses exactly three fresh normalized snapshots of state hash, retrieval ID order, and action-trace hash. Equal snapshots are `deterministic`; any semantic/order difference is `nondeterministic`; missing snapshots are `inconclusive`. The fixed canary repetition rule is one run for deterministic candidates and three runs for nondeterministic or inconclusive candidates.

Capability verification exact-revalidates adapter info and declared/observed `AdapterCapabilitiesV3`, reports overclaims and conservative underclaims separately, checks frozen extractor ID/version coherence, and recomputes presentation level from observed exports and state-transition linkage. A candidate passes this gate only with no overclaims and observed Level 2 or 3 behavior.

### Artifact, license, and redaction boundary

Private raw references contain hash, size, media type, storage class, and license status, but deliberately contain no public path or copied raw payload. Redistributable normalized references require a portable artifact path, at least one unique private-raw hash, and a fixed redaction version.

Redistributable payload validation requires an exact explicit `REDISTRIBUTABLE` license status and rejects private, license-uncertain, missing, or string-substituted statuses. Recursive scanning detects private-key markers, bearer/API/cloud/GitHub credentials, sensitive mapping fields, and compound assignments such as `client_secret`, `access_token`, `refresh_token`, `auth_token`, and `id_token`; findings contain only rule/location, never the secret value. Error redaction uses the same rules.

Private and normalized roots must be real, non-overlapping, non-reparse directories. A root is rejected both when it is inside immutable Core and when it contains immutable Core, preventing a broad parent root from addressing the task release through a relative path. Returned root identities are pinned for downstream persistence.

### Files and implementation checkpoint

```text
mub/vnext/external/__init__.py
mub/vnext/external/artifacts.py
mub/vnext/external/bridge.py
mub/vnext/external/probe_v3.py
mub/vnext/external/security.py
mub/vnext/external/visibility.py
tests/vnext/test_external_artifact_security.py
tests/vnext/test_external_capability_probe_v3.py
tests/vnext/test_external_visibility_bridge_v3.py
```

```text
a4d07e3585cc0535e01e53fb0207d060cd3d2678
feat: add external capability security gates
```

### Validation and review evidence

Observed RED regressions covered missing modules/facade exports, privileged nested payloads, constructed/subclassed contract attacks, command credentials, malformed/mismatched/terminal worker responses, timeout stderr leakage, short and compound credential redaction, license fail-closed behavior, empty private-raw provenance, overlapping/overbroad roots, invalid reset backend types, forged determinism snapshots, capability overclaims, and extractor mismatch.

```text
provider-neutral capability/security tests: 39 passed, 1 skipped
external admission contracts/boundaries:     38 passed
external/Core runtime compatibility matrix: 361 passed
py_compile:                                  passed
git diff --check:                            passed
immutable Core/legacy/schema diffs:          empty
```

The one skip is the Windows reparse-construction test because this account lacks symbolic-link privilege (`WinError 1314`). Static component checks and fail-closed reparse logic remain active.

The routed review returned only Fable/Haiku model usage and was not accepted as an authoritative routed verdict. Its five concrete claims were nevertheless treated as untrusted suggestions and independently reproduced in the parent `gpt-5.6-sol` session: missing license enforcement, compound credential assignment leakage, generic history leakage, successful response followed by terminal worker exit, and empty private-raw provenance. All five received RED regressions, minimal fixes, focused GREEN checks, and the complete final gates above. Parent review found no remaining scoped P0/P1/P2 issue.

### Conclusion and next step

Task 10 Phase 3 is complete as provider-neutral infrastructure. It still provides no genuine external-system evidence and admits no candidate. The next bounded step is Task 10 Phase 4: minimal strict-v3 append/flush/resume/finalize persistence that binds authenticated canaries, adapter/probe configuration, normalized terminal rows, and hash-only private raw references without changing the v3 schemas.

## vNext Core strict-v3 external run persistence

### Motivation and contract boundary

Core Task 10 Phase 4 adds the minimal crash-aware persistence layer needed for genuine external canary runs. It reuses `TaskRunRecordV3`, `RunManifestV3`, `ScorerConfigV3`, `AdapterInfoV3`, and `AdapterCapabilitiesV3` without changing any strict-v3 schema. It does not execute a provider, create a score, admit a candidate, or modify immutable Core data.

`ExternalRunConfigV1` binds the source task manifest, authenticated canary/task view, adapter configuration, capability verification, model provenance, package provenance, environment lock, runtime/evaluation configuration hashes, parser/extractor/redaction versions, explicit redistributable normalized license, retrieval/answer policy, repetition identity, exact task order, and every source task-record hash. The SHA-256 of its canonical bytes is the output-location-independent run identity.

### Append, resume, and public/private separation

Runs are created only through authenticated `create()` or `resume()` factories; direct writer construction is rejected. Each `TaskRunRecordV3` is exact-revalidated, matched to the next expected task ID, checked against run/adapter/parser/extractor/redaction identity, security-scanned, appended as one canonical JSONL row, flushed, and file-fsynced before canonical progress is atomically replaced.

Public rows reject private raw paths and embedded raw adapter state. Per-row `raw_provider_artifact_hash` and `raw_adapter_state_hash` remain hash-only provenance; the final manifest aggregates their unique hashes while leaving `raw_provider_response_artifacts` and `raw_adapter_state_artifacts` empty, so private paths or payloads do not enter the redistributable root. The normalized output license is part of run identity and must be the exact `REDISTRIBUTABLE` enum.

Resume reauthenticates canonical `run_identity.json`, the exact incomplete tree, real single-link files, output directory identity, ordered canonical row prefix, row hashes, statuses, and progress. Exact stale progress prefixes are repaired from the fsynced JSONL source of truth. A complete precommit `finalized=true` progress file without a manifest is recognized as an interrupted commit window and safely returned to unfinalized state. Replaced identities, extra files, hardlinks, duplicate/reordered/extra rows, changed configuration, invalid nested types, and incoherent progress fail closed.

The writer exposes rows as an immutable tuple; its mutable append state remains private.

### Finalization and manifest binding

Finalization requires exact complete ordered coverage and rejects any `FAILED` or `PARTIAL` row. It rereads and exact-validates the JSONL, reauthenticates `run_identity.json` immediately before publication, computes every run-record hash, and creates `RunManifestV3` with exact status counts. The manifest binds both `task_runs.jsonl` and `run_identity.json` as normalized artifacts and records task-view/runtime/evaluation/model/package/lock/repetition/license/private-raw hash evidence in the typed summary.

Final progress is written before the manifest; the manifest is the final commit marker. Manifest installation uses same-directory hardlink-based atomic no-replace publication and directory fsync. Link, write, progress, and post-link fsync failures remove temporary files and roll back any installed manifest by verified file identity, preventing a failed finalize from appearing committed. A published manifest causes all later resume/finalize attempts to fail as already finalized.

### Files and implementation checkpoint

```text
mub/vnext/runtime/__init__.py
mub/vnext/runtime/run_v3.py
tests/vnext/test_run_core_v3_persistence.py
```

```text
c9c2fc249359609ffdc4e718db2022efae0ec63f
feat: add strict v3 external run persistence
```

### Validation and review evidence

RED regressions covered missing persistence/facade APIs, output-independent identity, exact ordered append, constructed contracts, changed identity, malformed rows, stale/tampered progress, extra/hardlinked artifacts, private paths/raw state/secrets, missing/failed/partial finalize, license binding, mutable writer rows, direct-construction bypass, identity replacement before finalize, precommit progress recovery, no-replace link failure, final-progress failure, and post-link directory-fsync rollback.

```text
strict-v3 persistence tests:             18 passed
persistence/Core compatibility matrix:  379 passed
py_compile:                              passed
git diff --check:                        passed
immutable Core/legacy/schema diffs:      empty
```

The routed review again recorded only Fable/Haiku usage and was not adopted as an authoritative verdict. Its two concrete claims—manifest remaining after a post-publication progress/fsync failure and identity replacement before finalize—were independently reproduced RED in the parent `gpt-5.6-sol` session. Final progress was moved before the commit marker, post-link rollback was added, and finalize now reauthenticates canonical identity bytes. Additional parent review added strict tree/progress crash recovery, factory-only construction, immutable row exposure, explicit normalized-license identity, and manifest binding of the complete run-identity artifact. All final gates passed.

### Conclusion and next step

Task 10 Phase 4 is complete as persistence infrastructure only. No external run or admission result exists yet. Before Mem0 preflight, Task 9 must freeze the already verified offline Qwen/MiniLM snapshots into a formal model-provenance artifact bound to the current task manifest and evaluation configuration; then Task 10 Phase 6 may build the isolated genuine Mem0 OSS worker.

## vNext Core external offline-model provenance freeze

### Motivation and authenticated context

Core Task 10 model discovery is now frozen as typed, content-addressed provenance rather than an informal cache observation. The preparation CLI authenticates the complete immutable Core task release through the existing release authenticator, requires the exact task-manifest hash, strict-parses the canonical Canary A/B set manifest, checks its fixed hash and release binding, and verifies three single-link private evidence files before publication. Caller-supplied hashes are not trust roots.

The canonical evaluation configuration binds:

```text
source task manifest: 38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3
canary set manifest:  3c822b014af2b1026056f81b9284bbb6a4ed52d9072ac5524c7aa2fb6c8f95a8
canaries:             canary_a, canary_b
reset trials:         20
determinism namespaces: 3
repetitions:          deterministic=1, nondeterministic/inconclusive=3
retrieval policy:     normal_topk
answer mode:          slot_direct
```

Its canonical SHA-256 is:

```text
543881d8a6a1d16e5e4c5e5a3db655c4dda557d618736b00aee6e828d3003c7e
```

### Frozen snapshots and probe evidence

The role-specific trust boundary requires the complete exact lock for both snapshots, including ID, official URI, revision, license, architecture, tree-manifest algorithm/hash, file count, total size, and local-files-only status:

```text
Qwen/Qwen2.5-7B-Instruct
a09a35458c702b33eeacc393d103063234e8bc28
d2d9ab0fbeed7ab74ff3dc433209aec9b01952ccc4d88eec16c0d9aaf1fef9c8
Apache-2.0; 14 files; 15,242,807,270 bytes

sentence-transformers/all-MiniLM-L6-v2
c9745ed1d9f207416be6d2e6f8de32d1f16199bf
d17624986b02a007e8de99a086d7541ae0119b3f5840890ff196e687b846925b
Apache-2.0; 11 files; 91,578,367 bytes
```

The normalized facts bind the exact deterministic Qwen prompt/decoding contract and response hash, the exact two MiniLM inputs, CPU device, `(2, 384)` shape, finite/nonzero/repeatable checks, and measured package versions. Private raw evidence is retained outside the redistributable root and represented publicly only by hash, size, media type, storage class, and private license status:

```text
snapshot tree evidence: c2e3084b0239a62031c02573b4b0b65f0c538feb44474ad5a757f0f82321032e / 3574 bytes
offline model probe:    024a24eea20c0188d0a7666a093b2adf59992c7556129965b057d5bffba24655 / 1886 bytes
package-version probe:  51e44dc22ac808f7df563aed8b5771989334443fec2d277095255a244d1f777c / 1617 bytes
```

The final public artifact is under the ignored normalized result root `results/vnext/core_task10_model_provenance_v3`; its canonical model-provenance SHA-256 is:

```text
8cf12307c7d421ae46623f0428e626e7b99a9cbf5e31444a83729b929acdec8e
```

No local/NAS path or private payload is serialized into that artifact.

### Publication and validation

Publication exact-revalidates every nested contract, rejects secrets and any model/evidence/context drift, pins and repeatedly rechecks the output-parent identity and immutable-Core containment, validates the staged two-file tree byte-for-byte, uses atomic no-replace rename, fsyncs files/directories, and rolls back an owned installation after post-rename fsync failure. Contradictory evaluation/source/canary contexts are rejected.

```text
mub/vnext/external/model_provenance.py
mub/vnext/external/__init__.py
scripts/vnext_prepare_external_model_provenance.py
tests/vnext/test_external_model_provenance.py
```

```text
6f47dc5cbd885e3cd767825e3e3eb6e600a57952
feat: freeze Core external model provenance
```

Final validation:

```text
model-provenance focused tests:       14 passed
external infrastructure compatibility: 109 passed, 1 skipped
py_compile:                            passed
git diff --check:                      passed
immutable Core/legacy/schema diffs:    empty
```

The skip is the existing Windows symbolic-link construction case (`WinError 1314`); static reparse checks remain active.

The routed review used a mixed Fable/Haiku/Sol execution and could not be verified as the exact requested Decision profile, so its aggregate verdict was not adopted. The parent `gpt-5.6-sol` session independently reproduced the concrete issues: caller-controlled manifest hashes, unfrozen model tuples, contradictory evaluation context, unpinned publication parent, and provenance that did not require retained raw probe evidence. Each received a RED regression and a bounded fix. The final CLI additionally authenticates the immutable release, strict canonical canary manifest, and all three retained evidence files.

### Conclusion and next step

Task 9 is complete. This is model-prerequisite evidence only: no Mem0 package, Qdrant environment, external adapter run, capability report, or admission decision exists yet. Read-only preflight found neither `mem0` nor `qdrant-client` in the local environment or the approved cluster environment. The next bounded step is the isolated genuine Mem0 OSS package/lock and worker preflight; dependency installation must remain isolated and explicitly authorized, and a resource/setup `BLOCKED` state must not be relabeled as a Mem0 candidate `FAIL` or used to unlock LangGraph fallback.

## vNext Core Mem0 OSS configuration and host/worker preflight

### Motivation and fixed provider identity

Task 10 now has a dependency-free, fake-backend-tested host/worker path for the first fixed candidate, `mem0_oss`. The package contract pins the genuine official release rather than a project-local approximation:

```text
package:        mem0ai
version/tag:    2.0.17 / v2.0.17
release commit: 12c47f524935692e27ad48d829f35fa1e4417181
wheel SHA-256:  1521209f0ab4c77b7e5777aa1b0b5f0104efa06ca5b9eddb804cdd091c17726a
wheel size:     343876 bytes
license:        Apache-2.0
```

The public configuration binds the authenticated model-provenance artifact, run-derived non-default Qdrant collection, `user_id` namespace isolation, `delete_all_user_id` reset, normal native top-k retrieval with no reranking, inference enabled, the local Qwen provider, frozen 384-dimensional MiniLM embedder, embedded Qdrant, disabled telemetry, and fixed extractor/redaction versions. Local Qwen/MiniLM/Qdrant/history paths remain in a separately validated private worker configuration and cannot enter the public configuration hash.

### Host, worker, and trust boundaries

The implementation adds:

```text
mub/vnext/external/providers/mem0_protocol.py
mub/vnext/external/providers/mem0_adapter.py
mub/vnext/external/workers/mem0_worker.py
mub/vnext/external/workers/__init__.py
```

The host adapter authenticates an exact typed worker health identity before use. It sends only allowlisted visible event/query payloads; requested actions are parsed only on the host, while effective mutations and affected native entry IDs come from worker-observed state. The frozen evaluation extractor only recognizes exact visible canonical Add/Update memory text against declared object identities. It does not fill object/value candidates from gold fields. Native retrieval IDs, order, finite scores, source linkage, timestamps, and the fixed `normal_topk` policy are retained. `slot_direct` reads only normalized observable entries and uses visible sequence metadata, then native timestamps/order as bounded fallbacks; no Task 11 answer model is invoked.

Capabilities remain conservative: Mem0 declares isolated reset, event ingest, ADD, normalized entry export, native retrieval IDs/scores, and host action trace, but does not claim native UPDATE/DELETE/TTL/history/multi-object answer, raw-state export, native object/value fields, or evidence linkage. The resulting presentation level is 3 when the probed state-transition linkage is available and 2 otherwise.

The worker imports no optional Mem0/Qdrant/Torch/Transformers dependency at module import time. Its formal canonical JSONL loop validates exact requests, emits typed canonical responses, limits request size, closes cleanly with verified process exit status, and sanitizes backend failures. The CLI accepts only an absolute, real, canonical private worker-configuration file. Telemetry is disabled before the Mem0 SDK import. The official backend uses namespace-filtered `delete_all`, `add`, `get_all`, and native `search` calls; only currently exportable affected IDs and a small metadata allowlist cross into normalized responses.

Security regressions verify that benchmark gold/action/evidence metadata never crosses the worker boundary, untrusted worker error strings and native metadata are not echoed publicly, source event IDs are restricted to host-observed visible events, optional SDKs are not loaded by host imports, and CLOSE is idempotent and process-clean. The 20-trial reset protocol passes against the in-process fake worker, and an end-to-end fake worker round trip passes through the real `JsonlSubprocessBridge` framing and shutdown path.

### Validation and boundary

```text
complete external-infrastructure gate:        159 passed, 2 skipped
post-review Mem0 focused gate:                  37 passed
post-review bridge security/protocol gate:      25 passed
Core runtime compatibility gate:               323 passed
py_compile:                                     passed
git diff --cached --check:                      passed
immutable Core/legacy/strict-v3 schema diffs:  empty
```

The two complete-gate skips are the existing Windows symbolic-link construction cases (`WinError 1314`); static reparse checks remain active. The complete external snapshot passed before the final bounded source-linkage, metadata-redaction, and worker-exit diagnostics were added; every suite touched by those final changes was rerun independently and passed as shown above.

The previously attempted exact routed Decision review could not be verified as the required model/effort profile, so its aggregate verdict was not reused and no further mismatched delegated verdict was requested. The parent `gpt-5.6-sol` session completed the specification, security, and code-quality inspection directly, with RED regressions for request-ID and health drift, untrusted error/metadata/source leakage, non-exportable and multi-entry effects, telemetry ordering, normalized-state ordering, and clean/nonzero worker termination.

This is a code-path preflight only. No `mem0ai` or Qdrant package has been installed, no wheel or transitive lock has been downloaded, no local Qwen/MiniLM/Qdrant integration has run, and no authenticated candidate report exists. Therefore this work is neither Mem0 `PASS` nor candidate `FAIL`; it does not authorize LangGraph fallback and does not change overall Core status. The next step is the separately authorized isolated dependency/real-backend preflight, followed by 20/20 real reset trials and only then Canary A/B admission execution.

## vNext Core Mem0 OSS isolated real-backend preflight

### Authorized environment and dependency freeze

The bounded real preflight installed only the approved Mem0 overlay inside:

```text
/NAS/yesh/MemUpdateBench/external/mem0_2_0_17/venv
```

The venv uses `--system-site-packages` solely to reuse the Task 9 authenticated offline Qwen/MiniLM runtime. It does not modify `gmsra`, download a model, use an API key, or write outside `/NAS/yesh/MemUpdateBench`. The frozen Linux CPython 3.10 overlay contains 16 exact wheel hashes in:

```text
requirements/external/mem0-2.0.17-linux-py310.lock
requirements/external/mem0-2.0.17-linux-py310.wheels.sha256
```

`mem0ai==2.0.17` retains the official wheel hash `1521209f0ab4c77b7e5777aa1b0b5f0104efa06ca5b9eddb804cdd091c17726a`; `pip check` reports no broken requirements. The worker environment is rebuilt from an allowlist, forces Hugging Face and Transformers offline mode, disables Mem0 telemetry, and excludes unrelated credentials and secrets.

### Real integration diagnosis and bounded repair

The first real run established that the official backend and 20/20 namespace resets worked, but Qwen paraphrased the canonical memory sentence. A second instruction asked for exact text but returned the wrong official Mem0 JSON shape (`memory` contained a string instead of an object), so Mem0 2.0.17 failed inside its official extraction parser. Phase-localized and direct-backend diagnostics identified the exact boundary; resource contention encountered during one diagnostic was rerun on an available A40 and was not treated as a candidate failure.

The bounded v3 extraction instruction now follows Mem0's own additive schema exactly:

```json
{"memory":[{"id":"0","text":"<entire input sentence>"}]}
```

It simultaneously requires byte-identical visible input text and explicitly forbids a string-valued memory array. The public configuration authenticates this instruction as:

```text
version: mem0-exact-visible-memory-v3
SHA-256: 6ef4304659c0cde2c826165015064d632d91f674457280befd0e91a2f26e3913
```

Mem0 2.0.17 also exposes custom-provider registration through `LlmFactory` while its `LlmConfig` validator hard-codes built-in provider names. The worker therefore validates the complete official `MemoryConfig` using a temporary built-in provider name, restores the already registered `mub_local_qwen_v1` provider and its exact config, and constructs the genuine official `Memory` object. It does not relabel the provider as OpenAI, vLLM, or LangChain at runtime.

### Successful real preflight evidence

The isolated v3 run completed with status zero and recorded:

```text
20/20 namespace reset trials: PASS
canonical ADD action:          EXECUTED
normalized entry count:        1
stored content:                byte-identical canonical sentence
object/value extraction:       default|alice|city| / "Paris"
native top-k retrieval:        entry ID/order/finite score retained
slot_direct answer:            "Paris"
secret scan findings:          0
pip check:                     passed
```

Post-review rerun retained the same successful semantics while additionally enforcing exact stored content, retrieval of the exported native entry ID, a project-only worker `PYTHONPATH`, finite timeouts, immutable-Core output exclusion, wheel-to-install content authentication, and package-name/hash pairing in the lock regression. Private retained evidence and configuration hashes are:

```text
real-preflight-v3-reviewed.json
  SHA-256 e77c56b9115883b721eaf5e395605e8d24ad9b03f5aaa6be02149b8584b33312
  size 13292 bytes
worker-configuration-v3.json
  SHA-256 da3adbef8cf6ffe0b78f854dc9f4c1fb6c243b674870cbe5f29040c65b9cfd92
  size 1965 bytes
public configuration hash
  d72bd37887267e390484fa1b04cadba204108584a761fb5ee0ded30e9de26662
```

The real worker preserved a native Mem0/Qdrant entry ID, timestamps, native retrieval score, visible source event linkage, and host-derived action trace. Object/value fields remain explicitly marked `evaluation_extractor`; they are not native Mem0 structured fields and were not filled from gold state.

### Local validation and remaining admission boundary

```text
Mem0 focused tests and py_compile: 41 passed
post-review focused gate:         54 passed
complete external admission regression: 175 passed, 2 skipped
```

The two skips are the existing Windows symbolic-link privilege cases (`WinError 1314`); static reparse checks remain active. The immutable Core release, legacy fixtures, and strict-v3 schemas were not modified.

This closes the isolated real-backend preflight, not external admission. No Canary A/B terminal run, determinism classification, authenticated `ExternalAdmissionReportV1`, or `AdmissionDecisionV1` has yet been produced. Mem0 is therefore not yet admitted, and this preflight must not be used to authorize LangGraph fallback or to claim overall Core `FINAL_APPROVED`. The next bounded step is the authenticated Mem0 determinism probe and Canary A/B admission run against the immutable derived views.
