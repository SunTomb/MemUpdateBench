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

Admission requires all gates to pass, a recomputed presentation level of 2 or 3, and real event-ingest/add support. Fine-grained mutation/query capability bits remain authoritative for task support and terminal-row coverage; admission does not impose native in-place UPDATE as a global requirement on append/extract-then-store systems. Extractor-dependent candidates must bind a coherent extractor ID/version rather than silently inheriting benchmark gold. Public evaluation, fallback, and exact-one selection APIs revalidate exact stored fields and nested contract types so `model_construct`, subclass overrides, raw enum/string substitution, malformed strict values, and forged nested references fail closed.

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

## vNext Core Task 12 scientific freeze and admission-only preparation

### Motivation and fixed boundary

Task 12 freezes the approved Raw append version-arbitration mechanism study without executing it. The nine scientific cells are Family A only: chronological/no-label, reverse/no-label, and reverse/latest-outdated-label crossed with `normal_topk` retrieval `k=4/8/16`. The two Task 11 answer-model slots are replication/provenance bindings, producing 18 answer-run receipts rather than additional scientific cells. `reference`, `raw_add`, `exact_crud`, `heuristic_crud`, and `mem0_oss` remain an independent full 2,400-task Core test-split main-manager policy and are excluded from the intervention matrix.

The fixed transformation order is:

```text
frozen raw trajectory -> normal_topk -> presentation order -> full-trajectory version labels
```

The trajectory artifact now records authenticated entry IDs, event order, per-object version indices, full-trajectory latest-entry IDs, and a canonical trajectory digest. Admission recomputes every receipt from the immutable Core task artifact and rejects missing, reordered, malformed, or content-mismatched provenance rather than trusting task IDs alone.

### Scope, artifacts, and boundary

The dry-run distinguishes three authenticated scopes: 240 A/F/G hard-source tasks, 80 Family A matrix tasks, and 2,400 immutable Core test tasks. It binds one canonical scientific-design authority, one 3x3 semantic matrix, two Task 11 answer bindings, nine cell receipts, and 18 `(cell, answer_model_slot)` run receipts. `execution_authorized=false`; no provider, answer model, result root, score, statistics, claim ledger, or publication artifact is created.

Changed files include the Task 12 contracts/admission and admission-only CLI, mechanism context/matrix semantics, canonical scientific-design JSON, fixture provenance, and focused contract tests. Immutable `data/vnext/core/v3` and legacy fixtures remain untouched.

### Verification evidence

```text
Task 12 preparation/admission suite:                           25 passed
Task 12 compatibility, mechanism, and Task 10/11/runtime gate: 171 passed
py_compile and git diff --check:                                passed
```

The blocker remediation adds three provenance controls: `mem0_oss` is accepted only with canonical, authenticated Task 10 `AdmissionDecisionV1` and `ExternalAdmissionReportV1` evidence bound to the Core task-manifest hash, with decision/report evaluation-configuration hashes cross-checked; Task 11 qualification reports must match canonical JSON bytes after typed validation; and Raw append receipts record object identity and reject receipts whose latest-entry IDs do not cover each object’s actual final `(event_index, version_index, entry_id)` version. Focused negative tests cover non-admitted decisions, mismatched report/evaluation hashes, and noncanonical qualification evidence.

This is a preparation checkpoint only. Task 12 answer execution, Task 13 statistics/claim ledgers, and overall Core `FINAL_APPROVED` remain explicitly not started.

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

## vNext Core Mem0 capability admission verdict

### Determinism and mutation-capability probes

Three fresh isolated namespaces produced byte-identical normalized state, action, and retrieval-content snapshots:

```text
normalized semantic hash: 06915acc43090f4c013f2e422321a0747db108ed81c673be6d114a4e3d096615
determinism status:       deterministic
required repetitions:     1
```

The fixed real reset probe remained 20/20 PASS. A separate canonical `ADD → UPDATE → NOOP` probe established the candidate-specific boundary: ADD executed and created a native entry, UPDATE returned `no_effect` with `provider_no_effect`, and NOOP executed without a write. Final exported state retained only the original Paris entry. This supports the conservative declaration `supports_update=false`; changing it to true would overclaim observed behavior.

### Complete canary capability rows

The authenticated A/B views were rebuilt from the immutable Core release and compared canonically. For Mem0's declared capability bitset, every one of the 128 selected tasks requires at least one unsupported operation or query capability. The retained capability matrix therefore contains exactly one ordered terminal row per requested task:

```text
Canary A: 64 NOT_SUPPORTED
Canary B: 64 NOT_SUPPORTED
total:    128 ordered terminal rows
FAILED:   0
PARTIAL:  0
```

Rows preserve task ID, source record hash, runtime/operation/query support maps, and explicit missing-capability lists. They are not represented as zero scores or omitted work.

### Corrected admission policy and authenticated Mem0 PASS

A policy consistency review found that the initial verdict incorrectly treated native in-place UPDATE as a global admission prerequisite. That condition conflicts with the Level 2/3 presentation contract and would also reject truthful append/extract-then-store systems. Admission now requires all fixed gates to pass, presentation level 2 or 3, event ingest, and ADD. Fine-grained capability bits—including `supports_update=false`—remain authoritative for task support and the complete terminal-row matrix.

The prior `core_task10_mem0_admission_v1` FAIL and its fallback authorization were based on the invalid policy and have been removed from the active evidence tree; they must not authorize LangGraph. Rebuilding the same authenticated evidence under the corrected policy produced:

```text
candidate:                    mem0_oss
fixed gates:                  14/14 PASS
presentation level:           3
supports_update:              false (truthful observed boundary)
canary terminal rows:         128 ordered NOT_SUPPORTED rows
report outcome:               PASS
admission decision:           ADMITTED
fallback authorized:          false
secret scan findings:         0
```

Canonical artifacts:

```text
results/vnext/core_task10_mem0_admission_v2/external_admission_report.json
  SHA-256 2a00a350c750fc02f727af188a8f3d63f68df474e55a53a0710b5b62c6b43fae
results/vnext/core_task10_mem0_admission_v2/admission_decision.json
  SHA-256 c4355fdd1149325306eecf3242eeaf4e3e47a0d9ee616b0f9777058529e04f1c
  status admitted
  reason admitted_mem0_primary
```

The report remains bound by SHA-256 to the immutable local Core release and to the previously generated canary/model-provenance prerequisites; those roots remain intentionally outside this corrective commit and are authenticated in place rather than copied or rebound. The admission builder parses the UPDATE and NOOP probe as two bound typed `AdapterActionResultV3` records and verifies the frozen UPDATE object/value, so unrelated or wrong-target log fragments cannot satisfy capability truthfulness. It reparses all 20 reset trials through `NamespaceResetProbeV1`, reparses the public configuration as the exact frozen `Mem0AdapterConfigurationV1`, and structurally secret-scans both public configuration and adapter identity. All inputs, gate evidence, report, fallback check, and decision are closed in memory before a complete fsynced sibling staging tree is atomically installed; injected decision failure leaves no final root and a clean rerun succeeds. Independent local reconstruction revalidated the strict report and decision, reproduced the exact-one selection, confirmed `evaluate_candidate_admission(...) == true`, `authorize_fallback(...) == false`, 128 complete rows, and zero secret findings. The completed private LangGraph package/Store/extractor checks remain non-admission diagnostics only; because Mem0 is admitted, the fixed candidate order forbids executing or admitting the LangGraph fallback.

This closes Core Task 10 as a bounded external-adapter qualification. It does not supply prompted-answer evidence, manager accuracy results, Task 13 statistics, or overall Core `FINAL_APPROVED`. Task 11 remains the next separate gate.

## vNext Core Task 11 canonical Mistral provenance binding

The previously retained Mistral transfer snapshot was revalidated directly with `snapshot_tree_sha256_v3` over all 15 local snapshot files. The observed tree hash exactly matched the transfer manifest:

```text
model:              mistralai/Mistral-7B-Instruct-v0.3
revision:           c170c708c41dac9275d15a8fff4eca08d52bab71
file count:         15
size:               28,995,471,365 bytes
tree SHA-256:       31a92a122692365f74cc64939cc948fb21f1efa1d500afd3d92332ad319db015
manifest SHA-256:   579eb20207419eb53cfb0c8352487ec07276d55a9ded3b81137f53be4e53fe3d
recomputed match:   true
```

Two new canonical, redistributable metadata artifacts preserve the strict Task 11 contract without copying model weights or private raw payloads:

```text
results/vnext/core_task11_answer_harness/qualification_report.json
  SHA-256 00699e0d7a027d9bb63dca52753d53fe06bcdd0f7c87535aff6f25a7cb496672
results/vnext/core_task11_answer_harness/mistral_snapshot_provenance.json
  SHA-256 0fc48730152bafa005e3f18b12861bec295db02d9ff221ff7b0871cb9bf409da
```

Task 12 preparation schema is now explicitly `memupdatebench.core-task12-preparation.v2` because the required canonical Mistral provenance binding changes the manifest contract. Admission pins both Task 11 metadata artifact hashes above, so coordinated rebinding of the provenance record, qualification evidence, and answer-model binding cannot pass by self-consistency alone. The focused Task 10–12 regression gate completed with 112 passing tests; py_compile and `git diff --check` also passed. This closes the local snapshot-integrity binding only; the original download log records an unauthenticated Hugging Face request, so no upstream signature or provider attestation is claimed. No model inference, Task 12 result, score, statistics, or overall Core approval is produced by this step.

## vNext Core Task 12 authenticated admission-only dry run

A real preparation bundle was derived read-only from the authenticated Core v3 release and the approved Task 10/11 metadata artifacts. The immutable tree under `data/vnext/core/v3` was not modified. The bundle is retained at:

```text
results/vnext/core_task12_preparation_v2/task12_preparation_manifest.json
  SHA-256 7ab4af67e3cf84e2fcba9baa9b7ea6ee9a768cf4c3defcdc36dea78c0278e542
results/vnext/core_task12_preparation_v2/evidence/raw_add/trajectories.jsonl
  SHA-256 c615ee14b556faab566dd9b902c56b5b3cf793f0e4c0426ef3ddd94398245d0a
  records 80
results/vnext/core_task12_preparation_v2/dry_run_plan.json
  SHA-256 73725e8d2718449bf3438aa7e99c99783dab21bc74f9cef5cb1c533ec50a00bd
```

The raw-append trajectory receipt was recomputed from the authenticated Family A tasks and includes ordered entry IDs, canonical object IDs, event/version indices, current-version entry IDs, and per-task trajectory hashes. Static Raw append adapter identity, capability verification, and canonical `normal_topk` retrieval contracts for `k=4/8/16` are retained beside it. Approved Task 10 admission and Task 11 qualification/provenance artifacts were copied byte-for-byte into the dedicated evidence root and remain bound by their previously approved hashes.

The admission CLI was executed from a clean detached worktree at revision `ae3b55c187effd17da73899432359af07fc608c8`. It admitted exactly:

```text
A/F/G authenticated hard-source tasks: 240
Family A matrix tasks:                80
immutable Core test tasks:         2,400
scientific cells:                      9
answer-run bindings:                  18
execution_authorized:              false
```

The plan fingerprints are:

```text
hard-source selection: bca5f60b931d71d6cb78c1bd01b78867df7e2243ae6fca8d1d09bb4cf8f85aac
matrix selection:      e7def5d6ed8313f3c15bac723c019ca4f3607c4fd5d51c800f4c983187bb93c7
main-test selection:   4f63450b2fd54925ceb2f97544bda1e8051456cd4877385ec762464b990620f0
scientific design:     0e12f532f05976d650ea01c2ae62b8dbacb1e70a579b4e56b46d1e07721b0d12
semantic matrix:       3a73ffb5555e4049800ce247bcbfe39596cfc2d8bc804d6a1bd6983d3d27e686
plan fingerprint:      b17306163b932c8a228211f0f28e0a206b304ae8a98853b781731987aac323cf
```

## Post-Core Qwen3.5-9B 320-generation canary (2026-08-26)

The user authorized Tang-1-Wu GPU3 for the separate Qwen canary only. The canary reuses the authenticated Core Task 12 Family-A 80-task view and raw-append/retrieval semantics without relabeling the historical Task 12 answer-model slot. It freezes `normal_topk`, `k=16`, seeds `0` and `1` as greedy deterministic repetitions, and two conditions: `reverse-none` and `reverse-labeled` (`reverse_chronological` with and without `latest_outdated_label`).

The source-bound plan passed deterministic preflight with 80 tasks, 20 semantic cores, four tasks per core, 320 unique call IDs, identical retrieved semantic-entry multisets across paired conditions, and no-label/label metadata checks. The canary root is:

```text
/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_canary_13983f4_v1
```

Execution used the authenticated Qwen3.5-9B snapshot on Tang-1-Wu physical GPU3 with offline flags, BF16, eager attention, `trust_remote_code=false`, `enable_thinking=false`, greedy decoding, `num_beams=1`, and `max_new_tokens=64`. Each seed ran 160 calls with model load once, incremental terminal-row persistence, and process-exit GPU cleanup. No other process was terminated or altered.

```text
seed 0: 160/160 PASS, 160/160 exact match, stale copied 0
seed 1: 160/160 PASS, 160/160 exact match, stale copied 0
reverse-none:    160/160 PASS, exact match mean 1.000
reverse-labeled: 160/160 PASS, exact match mean 1.000
format-valid: 320/320
retries: 0
failures: 0
```

The finalized canary artifacts were reopened and checked: 320 terminal rows match the 320-call plan, the artifact index excludes itself and includes progress, and GPU3 returned to 3 MiB with no compute processes after each seed.

```text
canary receipt payload SHA-256: d763661bc3e4e65a1652dbd732bf7a4f7929ec8c8f479f076ea19ff1a304480
seed 0 attempts SHA-256:        8f7808f2fb8e5c60f99b18534d6fefb6eca70748d34ff1967c5043f4810c684c
seed 1 attempts SHA-256:        7f7d405b23f8dbfb95f2f4bebfb51e8cd8589ee9215f733c2c35b8445010f39c
```

This is `CANARY_ONLY` evidence. It does not execute or admit the three-condition confirmatory subset, full benchmark matrix, closed-provider calls, or scientific publication claims. The canary's perfect result is a bounded answer-model check on frozen Raw Append contexts, not evidence that every external memory system succeeds.

## Post-Core Qwen3.5-9B three-condition confirmatory hard subset (2026-08-26)

The user separately authorized the preregistered three-condition Qwen confirmatory subset on Tang-1-Wu GPU3. It reuses the authenticated Family-A 80-task view, `normal_topk` k=16, Qwen3.5 BF16/eager offline runtime, and greedy deterministic repetitions 0/1. The conditions are exactly:

```text
chronological / no label / k=16
reverse / no label / k=16
reverse / latest-outdated label / k=16
```

The independent confirmatory runner validated 20 semantic cores ×4 tasks, identical retrieved semantic-entry multisets across all three presentation conditions, correct ascending/descending order, no label leakage in the two no-label conditions, complete labels in the third condition, and 480 unique call IDs before model inference.

The no-replace result root is:

```text
/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_confirmatory_13983f4_v1
```

Each seed loaded Qwen once, emitted one append-only terminal row for each of 240 coordinates, and unloaded without terminating the pre-existing external GPU3 process. The runner used offline flags, `trust_remote_code=false`, BF16, eager attention, `enable_thinking=false`, greedy decoding, `num_beams=1`, `max_new_tokens=64`, and zero retries.

```text
seed 0: 240/240 PASS, exact match mean 1.000
seed 1: 240/240 PASS, exact match mean 1.000
chronological-none: 160/160 PASS, exact match 1.000, stale copied 0.000
reverse-none:       160/160 PASS, exact match 1.000, stale copied 0.000
reverse-labeled:    160/160 PASS, exact match 1.000, stale copied 0.000
format-valid: 480/480
failures: 0
retries: 0
```

Final verification reopened the 480-row plan and attempts, confirmed 480 unique call IDs, and confirmed the artifact index excludes itself.

```text
confirmatory receipt payload SHA-256: 9551a992c7b47505878bb6df71d4625714c5985219acc1ddedef30e5d43b7ae5
confirmatory receipt file SHA-256:    1ee0ce9210158426c598471747428c7889150bb03a72a47d0e57164e7f6f96fa
seed 0 attempts SHA-256:              280ff51de38931b828b802ee3efb3f66e0864661df24e30868ea60696c31811c
seed 1 attempts SHA-256:              23ee4b5905da41b35c859ea43d985ab485b09153fc04a287d199031f95d358c3
artifact index SHA-256:               be4c0baadf8a3652b8104d8f8a4db963d0d98000ec30434bba8ff36685d86bdc
```

This result is bounded to Qwen3.5 answering over frozen Raw Append Family-A contexts. It shows no order- or metadata-sensitive degradation for this qualified model and prompt/parser configuration. It does not establish external-memory-system breadth or authorize the full benchmark matrix.

## Post-Core Qwen3.5-9B full 18-cell matrix (2026-08-26)

The user authorized the complete Task12-shaped Qwen matrix after the canary and confirmatory gates. The source-bound runner expanded the same authenticated 80-task Family-A view over three context conditions, retrieval k values 4/8/16, and deterministic repetitions 0/1:

```text
80 tasks × 3 conditions × 3 k values × 2 repetitions = 1,440 generations
```

The no-replace result root is:

```text
/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_full_matrix_46e1e1f_v1
```

The preflight validated 1,440 unique call IDs and same-k retrieved semantic-entry multiset equality across all three presentation conditions. Both repetitions loaded Qwen once and persisted 720 terminal rows incrementally. GPU3 was shared with one pre-existing ~16 GiB process; no external process was altered or terminated. Qwen exited cleanly after each repetition.

Final overall result:

```text
terminal rows: 1,440/1,440
format-valid: 1,440/1,440
exact matches: 1,437/1,440
EM: 0.9979167
stale copied: 3/1,440
stale copied mean: 0.0020833
failures: 0
retries: 0
```

Eight of nine condition×k cells achieved 160/160 exact match. The only non-perfect cell was chronological/no-label at k=4:

```text
chronological-none-k04: EM 0.98125, stale copied 0.01875
chronological-none-k08: EM 1.00000, stale copied 0.00000
chronological-none-k16: EM 1.00000, stale copied 0.00000
reverse-none-k04/k08/k16: all EM 1.00000
reverse-labeled-k04/k08/k16: all EM 1.00000
```

Repetition 0 had one stale-copy error in chronological-none/k4; repetition 1 had two. This is a small but repeat-observed low-k retrieval/context failure, not the previously hypothesized reverse-order high-k collapse. Reverse/no-label remained perfect at every k.

```text
full-matrix receipt payload SHA-256: c9108a1aa8f06b8da98bc8b9d1f87cda7ebae1b06ab5b6752635ca242f604f6c
full-matrix receipt file SHA-256:    529ec8511e574ab6b92f369bd4b4d9007013b426868fd6fe99c7856405d9b5f6
seed 0 attempts SHA-256:             47c292e4d6d0e317b971e025e6cd55fa4a1b0304811da8ddf9ea4e35fe4611f2
seed 1 attempts SHA-256:             130c28fa3e8b5cf8aca9cacdb7125a767b710a7122edefc1d10a40eaed0b1722
artifact index SHA-256:              6d01cab144caff7532ff5e39f42ea012542ec0f0416a4eac58764ff0a95c29c3
```

This is a full Qwen answer-model matrix on frozen Raw Append Family-A contexts. It does not add manager diversity or external-system evidence. The dominant conclusion remains model-sensitive: Qwen3.5 is highly robust to reverse order and absent version labels in this matrix, with only a small chronological k=4 stale-copy tail.

The external output leaf remained absent after both admission invocations. Therefore this is authenticated preparation evidence only: no model/provider was loaded, no prompted answer was generated, no Task 12 execution/result/score was created, and Task 13 statistics, claim ledgers, and overall Core `FINAL_APPROVED` remain not started.

A subsequent reproducibility and evidence-integrity audit rebuilt the bundle from the same immutable Core and approved Task 10/11 inputs. The manifest hash remained `7ab4af67e3cf84e2fcba9baa9b7ea6ee9a768cf4c3defcdc36dea78c0278e542`, the trajectory hash remained `c615ee14b556faab566dd9b902c56b5b3cf793f0e4c0426ef3ddd94398245d0a`, and the dry-run receipt hash remained `73725e8d2718449bf3438aa7e99c99783dab21bc74f9cef5cb1c533ec50a00bd`. A clean-worktree admission rerun again produced 240/80/2,400 scopes, 9 cells, 18 answer-run bindings, `execution_authorized=false`, and no output leaf. All 15 uniquely bound Core/evidence artifacts matched their declared SHA-256 values, NDJSON record counts matched, and the evidence-root secret scan found zero findings. This audit remains pre-execution preparation and does not authorize Task 12 answer runs.

## vNext Core Task 12 cluster offline answer-model load preflight

The first real cluster preflight ran on Tang-2 (`Tang2`) using an NVIDIA A40 with 46,068 MiB visible memory. The Qwen snapshot was already present in the cluster HF cache and was independently rehashed with `snapshot_tree_sha256_v3`:

```text
slot:       answer_model_a
model:      Qwen/Qwen2.5-7B-Instruct
revision:   a09a35458c702b33eeacc393d103063234e8bc28
tree hash:  5c5fc08ade3cfa718521bbb2206deb1f0249527b8f210c95a4db9140460154ca
path:       /NAS/yesh/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
preflight:  PASS
```

The Mistral snapshot was found in the existing cluster Task 11 answer-model package and independently rehashed:

```text
slot:       answer_model_b
model:      mistralai/Mistral-7B-Instruct-v0.3
revision:   c170c708c41dac9275d15a8fff4eca08d52bab71
tree hash:  31a92a122692365f74cc64939cc948fb21f1efa1d500afd3d92332ad319db015
path:       /NAS/yesh/MemUpdateBench/external/task11_answer_models/mistral_7b_instruct_v03_snapshot
file count: 16
total size: 28,995,473,642 bytes
preflight:  PASS
```

Both slots were loaded and released through the existing offline runner with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`, and `trust_remote_code=False`. The Mistral tokenizer required `sentencepiece==0.2.0` and `protobuf==4.25.3`; these were installed into the isolated project directory `external/task11_answer_models/dependencies` from offline wheels only, without modifying the shared `gmsra` environment. The wheel hashes are recorded in the cluster evidence artifact:

```text
/NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_preflight_v1/answer_model_offline_preflight.json
SHA-256 7d5c7e5713bf92f548687a48d995dad514b22e374bdfaf260f6583bf40e68e4d
```

The first Qwen attempt correctly failed before model loading because `activate.sh` lacked `TRANSFORMERS_OFFLINE=1`; the project-local cluster environment was repaired and the clean rerun passed. The first Mistral attempt correctly rejected an incorrectly copied tree-hash argument before model loading; the exact canonical hash was then supplied. The corrected Mistral load passed after the isolated tokenizer wheels were installed. Successful preflight means only that both frozen slots can load offline on the A40; it does not mean answer generation or Task 12 execution has started. No prompts, answers, scores, or result root were created.

## vNext Core Task 12 single-run CLI and fake offline e2e gate

### Motivation

Task 12 now has a bounded execution layer for exactly one authorized `(cell_id, answer_model_slot)` run at a time. The admission-only dry-run remains immutable and keeps `execution_authorized=false`; execution requires the separate `Task12ExecutionAuthorizationV1`, matching the authenticated preparation manifest hash, plan fingerprint, cell binding, answer-model slot, and a one-component output leaf.

### Implementation

Added a Task 12 execution module and CLI that reuse the existing strict-v3 runtime, public run writer, and authenticated scorer instead of creating a parallel result format:

```text
mub/vnext/runtime/task12_execution_v3.py
scripts/vnext_run_core_task12.py
tests/vnext/test_core_task12_execution.py
tests/vnext/test_core_task12_cli_e2e.py
```

The runner executes only Raw append Family A cells with `slot_prompt` and `normal_topk(k)`. It applies the frozen presentation transform after retrieval: chronological/no-label, reverse/no-label, or reverse/latest-outdated-label. Latest/outdated labels are derived from the full raw trajectory, not from the retrieved subset. Public rows are persisted through `ExternalRunWriterV1`, finalized as `RunManifestV3`, reloaded, scored through `VerifiedScoringContextV3.from_authenticated_manifests`, and written as canonical `scores/scores.jsonl` plus `scores/score_receipt.json`.

The CLI also contains a hidden `--fake-offline-answer` test path. That path exercises the same CLI orchestration, persistence, manifest reload, authenticated scoring, and score-receipt writing without loading Qwen or Mistral. It is regression infrastructure only and is not prompted-answer scientific evidence.

### Verification

Fresh local verification after cleanup:

```text
python -m py_compile scripts/vnext_run_core_task12.py mub/vnext/runtime/task12_execution_v3.py
  PASS
python -m pytest tests/vnext/test_core_task12_cli_e2e.py tests/vnext/test_core_task12_execution.py tests/vnext/test_core_task12_cli.py tests/vnext/test_core_task12_run_contract.py tests/vnext/test_core_task12_matrix_contract.py tests/vnext/test_run_core_v3_persistence.py tests/vnext/test_v3_runtime_score_adapter.py -q
  54 passed in 16.10s
git diff --check
  PASS
```

### Boundary and next steps

This closes the local fake-offline end-to-end runner gate only. It does not run Qwen or Mistral, does not create a real Task 12 answer-matrix result root, does not start Task 13 statistics or claim ledgers, and does not declare overall Core `FINAL_APPROVED`. Before any real 18-run matrix, the production single-cell `ExternalRunConfigV1` binding should be reviewed against the authenticated preparation artifacts, then one real offline slot/cell run should be executed and inspected before expansion.

## vNext Core Task 12 single-run bundle builder

### Motivation

Task 12 execution now needs a production-facing preparation step that derives exactly one authorized `(cell_id, answer_model_slot)` execution bundle from the authenticated preparation manifest and admitted dry-run plan. This keeps execution configuration out of ad hoc shell arguments and preserves the separation between admission-only planning and execution authorization.

### Implementation

Added a bundle builder and CLI:

```text
mub/vnext/runtime/task12_bundle_v3.py
scripts/vnext_prepare_core_task12_run.py
tests/vnext/test_core_task12_run_bundle.py
tests/vnext/test_core_task12_run_bundle_cli.py
```

The builder reads canonical `Task12PreparationManifestV1` and `Task12DryRunPlanV1`, checks that the plan binds the preparation manifest's Core task-manifest, hard-suite, and task artifact hashes, selects exactly one admitted answer run, and writes an isolated bundle containing `tasks.jsonl`, `task_manifest.json`, `run_config.json`, and `authorization.json`. The generated `ExternalRunConfigV1` points to the 80-task view manifest rather than the full Core task manifest so authenticated scoring coverage matches the run rows. Output-root guards reject existing roots and roots inside the immutable Core or evidence roots.

The execution CLI now consumes the preparation manifest explicitly through `--preparation-manifest`, validates authorization against `sha256_model(Task12PreparationManifestV1)`, checks the run config's task-manifest/task-view/task-record hashes against the files supplied on the command line, requires `offline_hf`, and uses the deterministic decoding values embedded in `ExternalRunConfigV1`.

### Verification

Fresh local verification:

```text
python -m pytest tests/vnext/test_core_task12_cli_e2e.py -q
  1 passed in 288.79s (0:04:48)
python -m pytest tests/vnext/test_core_task12_run_bundle.py tests/vnext/test_core_task12_run_bundle_cli.py -q
  5 passed in 1132.39s (0:18:52)
python -m pytest tests/vnext/test_core_task12_execution.py -q
  5 passed in 1.18s
python -m py_compile scripts/vnext_run_core_task12.py scripts/vnext_prepare_core_task12_run.py mub/vnext/runtime/task12_execution_v3.py mub/vnext/runtime/task12_bundle_v3.py
  PASS
```

### Boundary

The bundle builder prepares one cell/slot run but does not execute models, does not launch the full 18-run matrix, does not write into `data/vnext/core/v3`, does not start Task 13, and does not declare overall Core `FINAL_APPROVED`.

## vNext Core Task 12 full 18-run matrix execution expansion

### Motivation

Task 12 already had one-cell execution and one-cell bundle preparation. The remaining expansion was to preserve that single-run contract while adding production-facing orchestration for the complete 9-cell × 2-answer-slot matrix. This must remain separate from Task 13 statistics and from any overall Core approval claim.

### Files changed/generated

```text
mub/vnext/runtime/task12_execution_v3.py
mub/vnext/runtime/task12_matrix_v3.py
scripts/vnext_run_core_task12_matrix.py
tests/vnext/test_core_task12_matrix_bundle.py
```

### Implementation

- Added `Task12MatrixRunRecordV1`, `Task12MatrixRunSummaryV1`, and `Task12MatrixRunResultV1` for a canonical 18-run execution summary.
- Added `execute_task12_matrix_bundles_v3(...)`, which validates a prepared matrix bundle manifest, loads both prompted answer-model slots once, executes every bound `(cell_id, answer_model_slot)` via `run_task12_cell_v3(...)`, requires complete task and score row coverage, and writes `matrix_run_summary.json`.
- Added `scripts/vnext_run_core_task12_matrix.py` as the production-facing matrix execution CLI. It requires `--execute`, constructs offline answer-model slots from the prepared run configs and caller-supplied snapshot/tree hashes, and exposes no provider/token/API configuration path.
- Fixed the Task 12 presentation transform to accept the built-in raw append adapter's `sequence_index` metadata as the event-order source, while preserving the existing `event_index` fixture path.
- Extended matrix tests with fake-offline 18-run execution, canonical summary reload checks, per-run manifest/score artifact checks, and matrix runner CLI help constraints.

### Commands run

```bash
python -m py_compile scripts/vnext_run_core_task12.py scripts/vnext_prepare_core_task12_run.py scripts/vnext_prepare_core_task12_matrix.py scripts/vnext_run_core_task12_matrix.py mub/vnext/runtime/task12_execution_v3.py mub/vnext/runtime/task12_bundle_v3.py mub/vnext/runtime/task12_matrix_v3.py
python -m pytest tests/vnext/test_core_task12_execution.py tests/vnext/test_core_task12_run_bundle.py tests/vnext/test_core_task12_run_bundle_cli.py -q
python -m pytest tests/vnext/test_core_task12_matrix_bundle.py::test_task12_matrix_runner_executes_all_18_fake_offline_runs -q
python -m pytest tests/vnext/test_core_task12_matrix_bundle.py -q
python -m pytest tests/vnext/test_core_task12_cli_e2e.py -q
git diff --check
```

### Current validation evidence

```text
py_compile: passed
Task 12 execution + bundle regression tests: 10 passed in 1148.28s (0:19:08)
Task 12 fake-offline 18-run matrix execution test: 1 passed in 705.70s (0:11:45)
Task 12 full matrix bundle test suite: 6 passed in 1918.85s (0:31:58)
Task 12 CLI fake-offline e2e test: 1 passed in 288.83s (0:04:48)
Consolidated focused Task 12 regression suite: 17 passed in 3297.04s (0:54:57)
git diff --check: passed
```

Earlier full matrix test runs exposed two root causes: adapter entries emitted `sequence_index` rather than `event_index`, and the prompted public-row/scorer path requires canonical gold action IDs rather than built-in observed action IDs. The fixes were to accept both metadata keys in `_event_index(...)` and wrap matrix-run adapters so `raw_result.parsed_action_id` is rebound to each task's gold action IDs before public row validation and authenticated scoring.

### Boundary

This expands the local authenticated execution layer and fake-offline regression coverage only. It does not launch real Qwen/Mistral scientific matrix jobs, does not modify `data/vnext/core/v3`, does not start Task 13, and does not declare overall Core `FINAL_APPROVED`.

## vNext Core Task 12 execution-layer final hardening and commit gate

### Motivation

A final adversarial review of the uncommitted Task 12 execution layer found that internal bundle consistency was not yet sufficient to prove closure against the admitted plan, immutable inputs, execution code, frozen answer-model slots, and exact matrix outputs. The implementation was therefore held back from commit and from any real model run until each confirmed boundary defect had a failing regression or a direct authenticated check.

### Files changed

The hardening remains confined to the existing Task 12 engineering unit:

```text
mub/vnext/runtime/task12_execution_v3.py
mub/vnext/runtime/task12_bundle_v3.py
mub/vnext/runtime/task12_matrix_v3.py
scripts/vnext_run_core_task12.py
scripts/vnext_prepare_core_task12_run.py
scripts/vnext_prepare_core_task12_matrix.py
scripts/vnext_run_core_task12_matrix.py
tests/vnext/test_core_task12_execution.py
tests/vnext/test_core_task12_cli_e2e.py
tests/vnext/test_core_task12_run_bundle.py
tests/vnext/test_core_task12_run_bundle_cli.py
tests/vnext/test_core_task12_matrix_bundle.py
```

Immutable Core and legacy fixtures were not modified.

### Confirmed findings and fixes

- Closed execution authorization over the preparation-manifest hash, plan fingerprint, cell/model/run bindings, exact 80-task view, task manifest, run config, output leaf, and a separately authenticated execution-code revision/tree. Preparation provenance and execution provenance remain distinct by design.
- Replaced free single-run task/config/output/hash arguments with one validated bundle root. Output is derived only from the authorization leaf; production CLIs expose no fake answer, provider, token, model-ID, revision, or tree-hash override.
- Bound both real answer-model objects to their frozen Task 11 slots and derive model ID, revision, license, and tree hash from the preparation manifest before offline loading.
- Parse immutable Core tasks only from bytes already verified by the Core `ArtifactRef`; all used evidence artifacts are digest-checked, canonical, regular, single-link files. Bundle children, finalized rows/manifests, and score artifacts reject symlinks/reparse points and hard links.
- Recompute and verify the complete manifest/plan binding, including scientific design, semantic matrix, manager policy, answer-model bindings, admitted cells, and all 18 admitted answer runs.
- Require the authenticated frozen raw-trajectory receipt for every presentation transform. Runtime retrieval `k`, context order, annotation, run ID, and `capture_snapshots=False` are rehashed and compared with the frozen run configuration before any task executes.
- Restrict action-ID rebinding to `observed_action:<current_event_id>` and the event's unique gold action ID; malformed observed IDs remain failures instead of being overwritten.
- Enforce unique ordered task/run coverage for scoring, exact persisted-prefix equality for resume, public-row validation on finalized reload, manifest counts/scorer/config/artifact checks, and atomic score publication with a hash-complete receipt.
- Make single bundles transactionally publish their four artifacts. Build the complete 18-bundle matrix under an owned staging root and rename it only after every child and the matrix manifest are complete; failed preparation removes its staging tree without exposing a partial final root.
- Resume now reloads and verifies finalized runs and scores without re-running inference. A fully finalized matrix resumes without loading either answer model; model-load failure closes every slot that began loading.

### Review and verification

The final specification review concluded `SPEC_COMPLIANT`; the final blocker-only code-quality review concluded `FINAL_CODE_QUALITY_APPROVED`.

Fresh final gate:

```text
python -m py_compile \
  scripts/vnext_run_core_task12.py \
  scripts/vnext_prepare_core_task12_run.py \
  scripts/vnext_prepare_core_task12_matrix.py \
  scripts/vnext_run_core_task12_matrix.py \
  mub/vnext/runtime/task12_execution_v3.py \
  mub/vnext/runtime/task12_bundle_v3.py \
  mub/vnext/runtime/task12_matrix_v3.py
  PASS

python -m pytest \
  tests/vnext/test_core_task12_execution.py \
  tests/vnext/test_core_task12_cli_e2e.py \
  tests/vnext/test_core_task12_run_bundle.py \
  tests/vnext/test_core_task12_run_bundle_cli.py \
  tests/vnext/test_core_task12_matrix_bundle.py -q
  20 passed in 2690.39s (0:44:50)
```

The complete fake-offline matrix again produced 18 × 80 = 1,440 terminal task rows and 1,440 score rows, then authenticated a no-inference resume. This remains orchestration evidence only and must not be interpreted as Qwen/Mistral prompted-answer scientific evidence.

### Error analysis and conclusion

The first final-suite attempt had 19 passing tests and one stale error-message assertion after manifest/plan validation was moved earlier; the implementation correctly rejected the mismatch. Updating the assertion and rerunning the entire gate produced the clean 20-test result above. No runtime defect remained.

This closes the local Task 12 execution-layer engineering gate only. It does not modify the immutable Core release, start Task 13, create real answer-model results, or declare overall Core `FINAL_APPROVED`.

### Next steps

After preserving this coherent engineering unit in a clean commit, rebuild the execution bundles so their runtime code revision/tree binds that commit. Then verify the canonical Qwen and Mistral snapshots, offline environment, devices, and an output root outside the repository/Core/evidence roots. Run one authenticated real single-cell smoke first; only after its typed parsing, 80 terminal rows, public-row privacy, run/score hashes, and authenticated scoring pass may the complete 18-run real matrix begin.

### Post-commit blocker correction

A final blocker-only review corrected its earlier approval after the first engineering commit and identified two library-boundary defects that the production CLI surface alone did not close:

- a same-slot fake object could call the public matrix executor and create run/score artifacts attributed to the frozen offline model;
- `matrix_run_summary.json` used a direct write, so interruption could leave a partial summary that blocked authenticated resume after all 18 child runs had completed.

The follow-up hardening requires exact `OfflinePromptedAnswerModelV3` values on production single-run and matrix paths, verifies the full frozen slot/model/revision/license/tree/decoding binding before model load, and moves fake models behind a private test-only executor/token. Matrix summary publication now uses the Phase 0 atomic publisher. No production CLI exposes the test-only path.

Fresh post-review gate:

```text
python -m py_compile \
  scripts/vnext_run_core_task12.py \
  scripts/vnext_prepare_core_task12_run.py \
  scripts/vnext_prepare_core_task12_matrix.py \
  scripts/vnext_run_core_task12_matrix.py \
  mub/vnext/runtime/task12_execution_v3.py \
  mub/vnext/runtime/task12_bundle_v3.py \
  mub/vnext/runtime/task12_matrix_v3.py
  PASS

python -m pytest \
  tests/vnext/test_core_task12_execution.py \
  tests/vnext/test_core_task12_cli_e2e.py \
  tests/vnext/test_core_task12_run_bundle.py \
  tests/vnext/test_core_task12_run_bundle_cli.py \
  tests/vnext/test_core_task12_matrix_bundle.py -q
  20 passed in 2530.85s (0:42:10)
```

This correction supersedes the earlier code-quality approval for the affected boundary. The corrected code again has no known Critical or Important Task 12 execution finding. The fake-offline result remains regression evidence only; no real prompted-answer run or Task 13 work was started.

### Authenticated dry-run plan loader compatibility

The first clean-cluster matrix-bundle preparation attempt failed before creating its output root because the authenticated `dry_run_plan.json` was one byte longer than `canonical_json_bytes(Task12DryRunPlanV1)`. Byte-level comparison proved that all 9,736 canonical content bytes matched and the only difference was one terminal LF at byte 9,736:

```text
authenticated plan SHA-256: 73725e8d2718449bf3438aa7e99c99783dab21bc74f9cef5cb1c533ec50a00bd
canonical content SHA-256:   aa21d5c711f86a5de17f5165a72c5d5a4739a1d03a358994221518c197c1722d
raw length / canonical:      9737 / 9736
first difference:            terminal LF only
```

The authenticated plan was not rewritten or rebound. A shared Task 12 control-file loader now remains strict by default and permits exactly `canonical_json_bytes(model) + b"\n"` only when callers explicitly enable `allow_trailing_lf=True`; all four production CLIs enable it only for the authoritative dry-run plan. Two trailing LFs, a trailing LF on other control artifacts, and all other noncanonical bytes remain rejected. The previous four duplicated CLI loaders were removed.

Fresh loader-compatible gate:

```text
python -m py_compile <the seven Task 12 runtime/CLI files>
  PASS
python -m pytest \
  tests/vnext/test_core_task12_execution.py \
  tests/vnext/test_core_task12_cli_e2e.py \
  tests/vnext/test_core_task12_run_bundle.py \
  tests/vnext/test_core_task12_run_bundle_cli.py \
  tests/vnext/test_core_task12_matrix_bundle.py -q
  21 passed in 2503.21s (0:41:43)
```

The failed cluster preparation created no matrix/output root and loaded no model. At this checkpoint, real Task 12 execution had not started.

## vNext Core Task 12 first real offline smoke and launcher correction

### Motivation and authenticated setup

The final local loader-compatible code was transferred to Tang-2 as a Git bundle and checked out in a clean detached runtime worktree. The dirty 445-entry remote main worktree was not modified or cleaned. Because the shared NAS ownership differs from the login UID, Task 12 uses an isolated `HOME` under the authorized project root with exact `safe.directory` entries; no user-global Git configuration was changed.

The canonical preparation/evidence root was absent remotely, so only the authenticated `task12_preparation_manifest.json`, `dry_run_plan.json`, and `evidence/` tree were transactionally synchronized. Immutable Core was not copied or modified. Remote hashes matched:

```text
runtime revision:              f88588b64ebc896c1cda6951adc464d9f000904a
runtime tree SHA-256:          108fe06093480bfc653d14bea4e35ad4d117bf00ba283e6dee147121399aa838
preparation manifest SHA-256:  7ab4af67e3cf84e2fcba9baa9b7ea6ee9a768cf4c3defcdc36dea78c0278e542
dry-run plan SHA-256:          73725e8d2718449bf3438aa7e99c99783dab21bc74f9cef5cb1c533ec50a00bd
Core task manifest SHA-256:    38e623e6888c8f692e6aeb4d7f8c593e72c8fab655d52aca96de954339a439d3
Core tasks SHA-256:            5c4fd518542b0665d7313d68f1a339de38502c376aa93fbda228196587cdd2c6
```

Both snapshots were freshly rehashed under the final runtime code with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`:

```text
Qwen tree:    5c5fc08ade3cfa718521bbb2206deb1f0249527b8f210c95a4db9140460154ca
Mistral tree: 31a92a122692365f74cc64939cc948fb21f1efa1d500afd3d92332ad319db015
```

### Bundle preparation and real Qwen smoke

The first production matrix root was prepared outside the clean runtime worktree, immutable Core, and evidence roots:

```text
/NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_matrix_f88588b_v1
matrix manifest SHA-256: 99e7fb0245fc524f9013cccf68dca3fcc8135111d8164c88dd1d48223e0f0e8f
bundle count: 18
rows per bundle: 80
run outputs before smoke: 0
```

A real Qwen `raw-add-chronological-none-k04 / answer_model_a` smoke ran on an A40 and completed 80/80 terminal rows plus 80/80 authenticated score rows:

```text
run manifest SHA-256: 24bb78bfb8657b830382c9c30ec4396d6d27eecfab96e154acc6306b315c3096
score artifact SHA-256: 87c676fad0bab06537ba723423a17198ce09a0180cbfb3284261383945f21649
completion statuses: completed=80
answer dispositions: answered=80
exact-match mean: 0.3875
format-valid: 58/80
```

The 22 invalid-format predictions were explicit parser outcomes, not silent coercions: 17 `answer_json_invalid` and five `answer_schema_mismatch`, all with JSON-object-shaped raw outputs. Snapshot paths had zero public-row hits. The apparent `raw_provider`/`raw_adapter_state` text hits were schema field names only, not private paths or payloads.

### Full-matrix launcher failure and boundary

The first `--resume` full-matrix launch failed before loading either model or creating a second run. Deduplicating the four CLI control-file loaders had left two stale `_load_canonical(...)` calls inside the matrix runner's `_model_for_slot(...)`; Python raised `NameError`. The root therefore contains exactly the authenticated Qwen smoke run, 17 absent run roots, and no matrix summary.

The two calls were changed to the shared strict control loader, and the matrix production-helper test now constructs both frozen slot models without loading them. Targeted verification passed:

```text
python -m py_compile scripts/vnext_run_core_task12_matrix.py
python -m pytest \
  tests/vnext/test_core_task12_matrix_bundle.py::test_task12_matrix_bundle_prepares_all_18_cell_slot_bundles -q
  1 passed in 435.79s
```

The `f88588b` matrix root is retained only as authenticated single-run smoke evidence and an incomplete launcher diagnostic. It must not be resumed under changed code or interpreted as the complete 18-run matrix. The corrected launcher requires a new clean runtime commit, freshly bound bundles, and a separate final matrix root.

## vNext Core Task 12 complete real offline answer matrix

### Final authenticated execution

The corrected launcher was committed and transferred to a new clean detached Tang-2 runtime worktree. Fresh bundles were built under a separate empty result root; no row or artifact from the incomplete `f88588b` root was reused:

```text
runtime revision:     9c798df5bb66e853466831ddae8ede3a1f2c01f4
runtime tree SHA-256: c6584f1d6db3241d6d80d9648d7b349483cf1e07301a212f9f92323d9af4d296
result root:          /NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_matrix_9c798df_v1
matrix manifest:      85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2
```

The real matrix ran in tmux with revision-pinned offline Qwen and Mistral snapshots on separate A40 devices. It completed all 18 frozen runs:

```text
run count:          18
terminal task rows: 1,440
score rows:         1,440
FAILED/PARTIAL:     0
matrix summary:     a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15
```

### Integrity and privacy audit

Every authorization, task view, run config, task row, run manifest, score row, and score receipt was reloaded through the final contracts. Summary run/score hashes matched all 18 artifacts. Public rows contained no Qwen/Mistral snapshot path. The audit artifact is outside the matrix root, under the sibling logs root:

```text
/NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_matrix_9c798df_v1_logs/matrix_integrity_audit.json
SHA-256 bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60
```

A production `--resume --execute` check then reloaded all 18 finalized runs, 1,440 rows, 1,440 scores, and the existing summary without any checkpoint load or inference. Its log is:

```text
/NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_matrix_9c798df_v1_logs/resume_check.log
```

The first audit compared adapter `entry_id` values across runs and correctly failed because those IDs include the isolated run namespace. This was not a retrieval-content mismatch. Re-auditing the frozen multiset by semantic entry identity `(object, source event, version, value, event index)` produced:

```text
(slot, k, task, query) groups: 480
incomplete groups:              0
semantic multiset mismatches:   0
snapshot-path hits:             0
```

Thus all three presentation conditions at a fixed `(slot, k, task, query)` preserve exactly the same retrieved semantic-entry multiset.

### Direct per-cell results

These are direct 80-task cell means only. They are Task 12 execution outputs, not Task 13 confidence intervals, claim-ledger entries, or final paper claims. `Fmt` is the fraction of typed answer outputs accepted as format-valid; `Stale` is the scored stale-copy rate. Retrieval stale exposure/count remain null under the frozen capability contract and must not be read as zero.

| context | k | Qwen EM | Qwen Fmt | Qwen Stale | Mistral EM | Mistral Fmt | Mistral Stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chronological / no label | 4 | 0.3875 | 0.7250 | 0.3125 | 0.7000 | 0.9250 | 0.2125 |
| chronological / no label | 8 | 0.5750 | 0.8125 | 0.2000 | 0.4250 | 0.8125 | 0.3125 |
| chronological / no label | 16 | 0.3875 | 0.9000 | 0.4125 | 0.2375 | 0.7625 | 0.5125 |
| reverse / no label | 4 | 0.7250 | 0.7625 | 0.0375 | 0.9500 | 0.9750 | 0.0125 |
| reverse / no label | 8 | 0.7625 | 0.8125 | 0.0250 | 0.9000 | 0.9375 | 0.0125 |
| reverse / no label | 16 | 0.8000 | 0.9000 | 0.0750 | 0.9000 | 0.9000 | 0.0000 |
| reverse / latest-outdated label | 4 | 0.7250 | 0.7375 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| reverse / latest-outdated label | 8 | 0.8500 | 0.8625 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| reverse / latest-outdated label | 16 | 0.9375 | 0.9375 | 0.0000 | 0.9750 | 0.9750 | 0.0000 |

The typed parser explicitly recorded invalid outputs rather than silently coercing them. Across 720 predictions per model, Qwen had 124 invalid outputs and Mistral had 57; per-run `answer_json_invalid` and `answer_schema_mismatch` counts are preserved in the audit artifact.

### Bounded conclusion and next step

Task 12 real offline execution and its row/hash completeness audit are complete. The direct matrix shows strong sensitivity to presentation order and explicit version labels, but its ordering differs from the earlier P8.3 surface construction: reverse/no-label is stronger than chronological/no-label here. This is a boundary result, not evidence that one order is universally best or that stale same-slot conflict is universally the strongest distractor. The latest/outdated intervention removes scored stale copying in both slots across all three k values, while format failures remain a separate answer-layer mechanism.

Do not start Task 13 implicitly from these means. The next task is the explicitly separate Task 13 statistics/claim-ledger/case-verification gate, which must treat semantic cores as independent units and bind every reported cell to the authenticated artifacts above. Overall Core is still not `FINAL_APPROVED`.

## vNext Core Task 13 clustered statistics, claims, and verified cases

### Motivation and contract

Task 12 established 18 authenticated prompted-answer runs but reported only direct 80-task cell means. Task 13 converts those rows into claim-ready evidence without rerunning a model. The independent unit is the semantic core: each cell has 20 cores and exactly four surface tasks per core. Cell estimates first average the four tasks inside a core and then average 20 core values. Paired contrasts use the same 20 core identities on both sides. The frozen bootstrap uses 10,000 replicates, 20 draws with replacement, Decimal precision 50 with `ROUND_HALF_EVEN`, Hyndman-Fan Type 1 endpoints 250/9750, seed `9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542`, and binary SHA-256 `0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53`.

The implementation authenticates the exact loader-issued matrix, exact capture-issued source snapshot, immutable publication bytes, Core/evidence/matrix membership, clean Git runtime, canonical Task 12 provenance, typed unsupported/null state, case score copies, and all receipt/index/claim hashes. It publishes exactly:

```text
bootstrap_indices.bin
cell_statistics.jsonl
paired_contrasts.jsonl
statistics_receipt.json
cases.jsonl
case_index.json
claim_ledger.jsonl
task13_artifact_index.json
```

The final index binds the first seven artifacts and does not self-hash. The production CLI has no model, provider, token, API, fake, metric, or override option.

### Implementation and validation

The coherent Task 13 implementation spans the contracts, frozen bootstrap, authenticated input loader, statistics, case selection, receipt/ledger builders, atomic publisher, and production CLI. Final integration fixes discovered by the real matrix added strict support for the authenticated Task 12 matrix-integrity-audit v1 schema, Python 3.10 weak-reference compatibility, exactly one trailing LF on the tracked canonical statistics config, canonical typed-config hashing, and Boolean-to-Decimal conversion for `protocol_scores.answer_parse_valid`.

Fresh gates included:

```text
Task 13 contracts/bootstrap/statistics/cases/ledger: 136 passed
Task 13 input authentication after audit adapter:    24 passed, 1 skipped
Task 13 CLI publication/safety review run:            42 passed, 3 skipped
Task 13 statistics after Boolean metric fix:           7 passed
Task 12 execution/matrix compatibility:               14 passed
```

The input skip and two CLI platform skips are not counted as passes: Windows lacked symlink privilege (`WinError 1314`), and POSIX-only renameat2/directory-fsync cases were skipped on Windows. Several earlier unbounded CLI attempts were stopped or killed and are not passing evidence. Independent reviews returned `SPEC_COMPLIANT` and, after closing the reported status/config/index/immutability/Python-floor findings, `CODE_QUALITY_APPROVED`.

The authenticated Task 12 controls were rechecked before execution:

```text
matrix manifest: 85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2
matrix summary:  a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15
integrity audit: bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60
```

The final computation ran from clean detached runtime revision `bc82566dd888c3993e826626bb13c8c057846266`, tree SHA-256 `2f5cd73f2bb4677532951fc1bb594a20589e9456303f20938c6ae77b2f68d125`. It completed and fully verified an owned remote staging root. The Tang-2 project filesystem is NFSv3 and rejected `renameat2(RENAME_NOREPLACE)` with `EINVAL`; the implementation correctly left the remote final root absent rather than falling back to `os.replace` or copy/delete. The exact verified bytes were transferred to the local NTFS project root and atomically committed with the same no-replace `MoveFileExW` primitive:

```text
local final root:
  results/vnext/core_task13_bc82566_v1

verified remote staging evidence:
  /NAS/yesh/MemUpdateBench/results/vnext/.mub-task13-stage-1a791f4cbfdd471aa6a8bd45ab6432d4

independent audit:
  results/vnext/core_task13_bc82566_v1_audit.json
  SHA-256 c60c49d917c582506e262534a6c48bb68668027e428ba0c06557ae8381982145
```

The remote staging path is evidence, not a published remote final root. Task 14 must preserve this platform boundary rather than relabeling the absent NFS final root as published.

### Artifact closure

```text
bootstrap_indices.bin      0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53
cell_statistics.jsonl      e4f25e3a7fc9795a93e8007acb1131dc84bb24fcdaf4867ac65042683bf0036b
paired_contrasts.jsonl     517d426b86e415467ab72e4655d9fe7972ca1218d3765141bc210d3b28120e47
statistics_receipt.json    398914d52b22c9c2bb71fc548e1f4239cf15cbf99d7cae2cd53e86b4fdcf9451
cases.jsonl                af863aa24f90851a6b7149b5cefbceafdbb8c3987bd7be439768386a4bdfdb80
case_index.json            8c97243db3265cb39f7048ea4e825d49aead50da94e122fd9c8e638360f2ed36
claim_ledger.jsonl         9f486dd90361dd8b70ed8cc2fa0c5a552dbf37f88b55addde71456347a4d0273
task13_artifact_index.json da02787276dd171cce716258ec071947ae99fb047a607df983f52125a20937aa
```

Cardinality and rejoin checks:

```text
cell statistics:       126 = 18 cells x 7 metrics
paired contrasts:       84 = 12 directed slot/k pairs x 7 metrics
claim rows:             210
verified cases:          57 across all 18 runs
case categories:         18 correct, 11 stale_copied,
                         16 answer_parse_invalid, 12 other_wrong
matrix case rejoin:      18 runs, 1,440 observations, 57/57 cases
bootstrap binary:        200,000 bytes
```

Four metrics are numeric in all 18 cells: exact match, stale copied, token F1, and answer-parse valid. Three remain typed unsupported/null in all 18 cells and were never converted to zero: gold-retrieved-wrong-answer, stale-count-in-context, and stale-exposure-rate.

### Core-clustered exact-match intervals

| slot | context | k=4 estimate [95% CI] | k=8 estimate [95% CI] | k=16 estimate [95% CI] |
| --- | --- | --- | --- | --- |
| Qwen | chronological / no label | 0.3875 [0.2500, 0.5250] | 0.5750 [0.4125, 0.7375] | 0.3875 [0.2250, 0.5625] |
| Qwen | reverse / no label | 0.7250 [0.6125, 0.8375] | 0.7625 [0.6250, 0.8875] | 0.8000 [0.7000, 0.8875] |
| Qwen | reverse / latest-outdated label | 0.7250 [0.6250, 0.8250] | 0.8500 [0.7875, 0.9125] | 0.9375 [0.8875, 0.9750] |
| Mistral | chronological / no label | 0.7000 [0.5375, 0.8500] | 0.4250 [0.2625, 0.6000] | 0.2375 [0.0875, 0.4125] |
| Mistral | reverse / no label | 0.9500 [0.8875, 1.0000] | 0.9000 [0.8250, 0.9625] | 0.9000 [0.8125, 0.9750] |
| Mistral | reverse / latest-outdated label | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9750 [0.9250, 1.0000] |

### Paired exact-match contrasts

Each value is left minus right over the same 20 semantic cores.

| slot | k | reverse/no-label minus chronological/no-label | labeled-reverse minus reverse/no-label |
| --- | ---: | --- | --- |
| Qwen | 4 | +0.3375 [0.1500, 0.5375] | 0.0000 [-0.1250, 0.1125] |
| Qwen | 8 | +0.1875 [0.0250, 0.3625] | +0.0875 [-0.0250, 0.2250] |
| Qwen | 16 | +0.4125 [0.2375, 0.5875] | +0.1375 [0.0625, 0.2375] |
| Mistral | 4 | +0.2500 [0.0875, 0.4125] | +0.0500 [0.0000, 0.1125] |
| Mistral | 8 | +0.4750 [0.3250, 0.6250] | +0.1000 [0.0375, 0.1750] |
| Mistral | 16 | +0.6625 [0.5000, 0.8125] | +0.0750 [0.0125, 0.1500] |

### Bounded conclusion

The core-clustered evidence confirms the Task 12 boundary result rather than the earlier P8.3 order surface: in this frozen Core construction, reverse/no-label is consistently better than chronological/no-label, with all six paired intervals excluding zero. Explicit latest/outdated labels add a positive high-k benefit in both slots; the k=16 label contrasts are +0.1375 [0.0625, 0.2375] for Qwen and +0.0750 [0.0125, 0.1500] for Mistral. At smaller k the label effect is model-sensitive and some intervals include zero. This supports the narrow claim that version arbitration is order- and metadata-sensitive; it does not support a universal claim that reverse order is best or that stale same-slot conflict is always the strongest distractor. Answer-format invalidity remains a separate mechanism, represented by 16 verified parse-invalid cases.

Task 13 is complete; Task 14 and overall Core `FINAL_APPROVED` remain not started.

## vNext Core Task 14 final release review

### Motivation and fixed boundary

Task 14 is the only gate permitted to decide the bounded overall Core release. It does not regenerate the immutable Core task release, rerun a manager or answer model, create a new statistic, expand a claim, or convert engineering/admission evidence into scientific accuracy. It freshly authenticates the immutable task release and Tasks 9–13 as separate evidence classes, constructs an acyclic evidence graph, publishes a five-file final-review root, and reopens that root against the current source snapshots and clean runtime before returning a verified release object.

The final review preserves the evidence boundaries established earlier:

- Pilot `slot_direct` and Task 9 built-ins/controls are deterministic engineering evidence.
- Task 10 Mem0 is a genuine external-system admission/capability result; its 128 canary rows remain explicit `NOT_SUPPORTED` and are not accuracy scores.
- Task 11 is model provenance and qualification evidence.
- Task 12 is the real 18-run offline prompted-answer matrix; fake-offline results are excluded.
- Task 13 is semantic-core clustered statistics, claims, and verified cases over Task 12; three retrieval/answer diagnostic metrics remain typed unsupported/null.
- The Task 13 `/NAS/.../.mub-task13-stage-*` root is verified remote NFS staging evidence only, not a published remote final root.

No file beneath `data/vnext/core/v3` or any existing Task 9–13 artifact root was modified.

### Design, implementation, and checks

Task 14 was first frozen in:

```text
docs/superpowers/specs/2026-08-20-memupdatebench-vnext-core-task14-design.md
docs/superpowers/plans/2026-08-20-memupdatebench-vnext-core-task14.md
```

The implementation adds strict contracts, source inventory/current-root snapshots, evidence-graph and review construction, atomic publication/reopen verification, and a no-model production CLI:

```text
mub/vnext/release/task14_contracts.py
mub/vnext/release/task14_sources.py
mub/vnext/release/task14_review.py
mub/vnext/release/task14_publish.py
scripts/vnext_review_core_task14.py
```

The final upstream graph contains 22 exact nodes and 22 exact edges. It binds the Core candidate receipt/root, human-audit attestation, Task 9 implementation/provenance, Task 10 report/decision, Task 11 model qualification/provenance, Task 12 manifest/summary/audit, and separate Task 13 statistic/contrast/case/ledger/audit roots. Task 14 output objects are not graph nodes, avoiding a self-hash cycle; they are bound in the acyclic order report/graph → attestation → manifest → non-self-hashing index.

The persisted structural report can only say `READY_FOR_VERIFICATION` or `NOT_APPROVED`. It cannot persist `FINAL_APPROVED`. Only the source-bound reopen verifier can construct the immutable verified wrapper whose attestation records approval at verification time.

Focused final gate:

```text
Task 14 contracts/sources/graph/approval/atomic/CLI: 36 passed, 2 skipped
py_compile:                                             passed
git diff --check:                                       passed
specification review:                                   SPEC_COMPLIANT
code-quality review:                                    APPROVED
```

The two skips are Windows symlink-construction cases (`WinError 1314`); lexical component and reparse checks remain active. They are not counted as passes. A later combined Core-audit/atomic/Task12/Task13 compatibility command was stopped after remaining at 56% without a terminal result; it is not passing evidence. The prior separate Task 12 and Task 13 compatibility gates remain the authoritative completed evidence, and Task 14 changed none of those modules.

Review findings closed before the final execution included incomplete evidence topology, missing Task 10/11/12/13 semantic bindings, caller-controlled runtime labels, source aliases, unbound NFS staging paths, source/runtime frontier revalidation, final-root link acceptance, minimal self-consistent report forgery, mutable loaded payload/snapshot splices, index metadata closure, parent/staging/final identity checks, parent-directory fsync, unsafe cleanup, CLI exit-code drift, and direct verified-wrapper construction.

### Real final review execution

The authoritative execution used a clean detached worktree at:

```text
runtime revision:
  84beabb62f5cd2cee97b294022db25c8261ab698
runtime tree SHA-256:
  6ab74200f2748a65ee29e8aedf17b4f19dd30e5f1cf856f282764fd3f6bf5133
```

Because the repository itself and its ignored evidence roots are protected inputs, the published output is deliberately outside the repository:

```text
D:/USTC/2026Winter/MemUpdateBench_releases/core_task14_84beabb_v1
```

The CLI performed no model, provider, API, token, GPU, fake, metric override, or approval override operation. The terminal decision was:

```text
decision=FINAL_APPROVED
core_final_root_index_sha256=2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035
core_final_attestation_sha256=a63008b5b5a60507fcb7c2f99b05373f6f10e7a366383a8d2cde20818208d005
```

The structural report contains 13 required passing checks, zero findings, six explicit exclusions, 22 evidence nodes, and 22 evidence edges. Its status is `READY_FOR_VERIFICATION`; overall approval comes from the separately persisted historical attestation and current-root verified wrapper.

### Final artifact closure

```text
core_final_review_report.json
  ec79c9b32b53016b5247f49eb82e90d6e0200214a196c190b62c68aae3f381ef
core_final_evidence_graph.json
  3a0d4d334c2dcf02ed1c5ddf713ca57402cf5adf97b60803a2fa9f1bc27c6417
core_final_verification_attestation.json
  21b995b946d33e4906e59f862f2f92324e9a2c83d9ebe34f67a6850c52e8db30
core_final_root_manifest.json
  a44464122414182e17af00e189f001ef6d69360c867f6d63bd4a6a68f5e2debb
core_final_root_index.json
  2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035
```

Attestation bindings:

```text
attestation self-hash:
  a63008b5b5a60507fcb7c2f99b05373f6f10e7a366383a8d2cde20818208d005
source snapshot SHA-256:
  f8605da478214d0ad717125cb57b3855a3d38539a5a5d6c55b478dc1b53ca068
final approval at verification:
  true
```

A second fresh detached worktree independently loaded every current Core/Task 9–13 source, rederived the exact report/graph, reopened all five final files, verified the complete artifact-ref metadata/hash chain, compared the source snapshot and runtime revision/tree, and returned `FINAL_APPROVED`. Its Git status remained clean after verification.

Earlier Task 14 roots at revisions `03256f9`, `c4fbb63`, and `ee8a3e6` are superseded diagnostics and are not authoritative release roots.

### Final conclusion and next phase

The bounded MemUpdateBench vNext Core release is `FINAL_APPROVED`.

This approval means that the immutable Core task release, deterministic engineering gates, genuine Mem0 admission boundary, offline answer-model provenance, real Task 12 prompted-answer matrix, and Task 13 statistics/claims/cases are mutually authenticated and correctly scoped. It does not turn Mem0 admission into accuracy, broaden the two-model Raw-append Family-A matrix into a universal external-memory benchmark, or support universal reverse-order/strongest-distractor claims.

The next project phase is separate from this frozen release: design the main-track external-validity expansion with genuine external-memory prompted-answer results, more independent semantic cores and domains/languages, and a layered reproducible-open/frontier-closed answer-model panel. Do not modify or rebind the `FINAL_APPROVED` Core release while designing that expansion.

## Post-Core model-expansion Phase 0 metadata release

### Motivation and boundary

The first post-Core unit turns the agreed modern model panel into a separate, authenticated execution plan without extending the literal two-slot Task 11/12/13 contracts or modifying any frozen Core artifact. It is Phase 0 metadata only: it performs no network request, provider call, model download/load, credential-value read, generation, scoring, or scientific comparison.

The frozen candidate intents are Qwen3.5-9B BF16, Meta Muse Glimmer 30B fixed int4 plus a BF16 k=16 control, Claude Sonnet 4.6, Claude Opus 4.8, Gemini 3.6 Flash, Grok 4.5, and GPT-5.5. Unverified model facts remain null/pending rather than inferred. The Qwen canary budget records 320 future requested generations, but every Phase 0 executable count is zero.

The design and plan are frozen in:

```text
docs/superpowers/specs/2026-08-20-memupdatebench-post-core-model-expansion-design.md
docs/superpowers/plans/2026-08-20-memupdatebench-post-core-model-expansion.md
```

### Implementation

A separate versioned namespace now owns the contracts, frozen registry, secret-free provenance, metadata qualification, call planning, transactional release, and no-network CLIs:

```text
mub/vnext/post_core/contracts_v1.py
mub/vnext/post_core/model_registry_v1.py
mub/vnext/post_core/provenance_v1.py
mub/vnext/post_core/qualification_v1.py
mub/vnext/post_core/planning_v1.py
mub/vnext/post_core/release_v1.py
configs/vnext/post_core/release_v1.json
scripts/vnext_prepare_post_core_release.py
scripts/vnext_qualify_post_core_models.py
```

The release publishes exactly seven files. Its final index binds the preceding six in frozen order and does not bind itself:

```text
post_core_release_manifest.json
model_registry.json
provenance.jsonl
qualification_report.json
capability_probe_report.json
execution_plan.json
post_core_artifact_index.json
```

The source boundary independently pins the approved Core task-release manifest and Task 14 index, semantically reopens the exact Task 14 sibling chain, snapshots/revalidates external config/registry/provenance inputs, requires the exact eight-candidate pending registry, and requires provided provenance to equal deterministic pending-intent provenance. Output publication uses a same-filesystem owned staging directory, no-replace directory commit, source/staging/output identity and digest checks, safe quarantine of tampered staging, parent durability, and a typed committed-but-unverified outcome.

Security hardening forbids environment credential-value reads, provider/model/network imports and calls, secret-bearing command flags and headers, and secret-bearing diagnostics. Credential environment-variable names are allowlisted and values are never persisted. Repository-relative authenticated Core and Task 14 fixtures keep the publication tests fail-closed in clean checkouts; the fixture hashes equal the immutable sources.

### Review, failures, and verification

Development followed failing-regression-first cycles. The first parent-worktree integration run exposed three direct-CLI import failures hidden by an inherited `PYTHONPATH`. Independent specification review then found caller-controlled source hashes, shallow Task 14 validation, mutable external metadata inputs, credential-value reads, staging cleanup gaps, and unbound zero network accounting. Subsequent reviews found public-config construction bypass, source snapshot races, secret-bearing exception echo, Task 14 sibling and staging/output TOCTOU windows, POSIX syscall fallback gaps, header-scanner bypasses, and tests that silently depended on local absolute roots. Every Critical/Important finding received a regression and was re-reviewed. An accidental Task 14 hardening edit was byte-restored immediately to preserve the frozen Task 14 implementation boundary.

Final independent decisions:

```text
specification review:  SPEC_COMPLIANT
code-quality review:   CODE_QUALITY_APPROVED
```

Fresh clean-checkout gates at revision `0745fc9dce33a1ace5efdf966d3b1f8b90b9e07b`:

```text
python -m pytest tests/vnext/test_post_core_*.py -q
  58 passed

python -m pytest tests/vnext/test_core_task14_atomic.py \
  tests/vnext/test_core_task14_sources.py -q
  14 passed, 2 skipped

python -m py_compile <post-Core modules, CLIs, and changed tests>
  PASS

git diff --check
  PASS
```

The two Task 14 skips are Windows symlink-privilege cases (`WinError 1314`) and are not counted as passes. The authenticated test fixtures have SHA-256 values:

```text
Core task-release manifest:
  dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b
Task 14 root index:
  2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035
```

### Strict-clean publication and artifact closure

The authoritative no-replace local root was produced with `PYTHONDONTWRITEBYTECODE=1` from a detached repository that passed the strict clean-runtime checker both immediately before publication and after semantic reopen:

```text
runtime revision:
  0745fc9dce33a1ace5efdf966d3b1f8b90b9e07b
runtime tree SHA-256:
  916a9cbc1c832270ccc1a9c57b4ac2a5404000da77d03bd35f986cacfc7ec84c
authoritative root:
  D:/USTC/2026Winter/MemUpdateBench_releases/post_core_phase0_0745fc9_clean_v1
```

A second independent strict-clean detached worktree reopened the seven files, rebuilt their exact semantics from the authenticated Core/Task 14/config sources, verified the six-entry non-self-hashing index, and returned `VERIFIED_PENDING_ONLY`.

```text
post_core_release_manifest.json
  6b01cf32ec76299e38c6bf3e4250994f7356717ba1108e2ec1914d73dd9c41b5
model_registry.json
  1cb669f3188b5d145338feab19412309cf774337c0b85cded412cd09e85fba12
provenance.jsonl
  01e4b65ae9c1948005a32533e76f1f26a5b0f5875f1abd8fd38a66cb5cc61965
qualification_report.json
  b6b49063a44623120f31ca613843b0b934bfc016dcb9fc5fc64f7fe35491078c
capability_probe_report.json
  7ed5ed5ba44e32eed60fb724870f2509951c686b0b95e72929abe8fb901c2de7
execution_plan.json
  d1657289003738dd8dd02de2afabc690b12e2f32271ef1bae43cadb33629ffca
post_core_artifact_index.json
  e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd
```

The earlier root `D:/USTC/2026Winter/MemUpdateBench_releases/post_core_phase0_0745fc9_v1` has identical deterministic artifact hashes but was created after tests had generated ignored cache directories in its detached worktree. It is retained only as a superseded diagnostic; the `_clean_v1` root above is authoritative.

Final metadata counts are:

```text
candidate rows:          8
qualification PENDING:   8
provenance rows:         8
provider calls:          0
model loads:             0
network calls:           0
executable calls:        0
Qwen future requested: 320
```

Post-publication hashes for the immutable Core manifest, Task 13 index, and all five Task 14 files exactly matched their pre-publication baselines. Therefore Phase 0 closes only the safe planning/publication gate. It is not model capability, accuracy, external-system, prompted-answer, or main-track scientific evidence. Phase 1 remains blocked until exact official open-model identities/revisions/licenses/snapshot manifests are authenticated and local model load/download execution is separately authorized; closed-provider preflight remains a later explicit network/credential/budget gate.

## Post-Core official model-identity document preflight

### Purpose and evidence boundary

The first post-Phase-0 step authenticated public model identities without modifying the frozen Phase 0 registry or release root. This work used public official documentation and pinned Hugging Face repository metadata only. It did not download a weight, create a local model snapshot, load a model, read a credential, call a provider inference API, generate an answer, or make any candidate executable.

The canonical evidence file is:

```text
configs/vnext/post_core/official_identity_evidence_v1.json
SHA-256:
  9e3780ed3d4303bda7bbd27865df89fcb384041da64af56107c8c5b7abf0a4f0
bound Phase 0 index:
  e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd
```

The strict contract and no-execution validator are:

```text
mub/vnext/post_core/identity_v1.py
scripts/vnext_validate_post_core_identities.py
tests/vnext/test_post_core_identity_evidence.py
```

The contract freezes the complete evidence bytes, exact candidate order, official URLs, repository revisions, selected artifact hashes, response identity fields, and conservative state transitions. It rejects links/reparse points, hard-linked inputs, noncanonical JSON, substituted evidence bytes, mutable-alias promotion, invented GPT-5.5 identity, closed-model architecture/license/parameter claims, and any nonzero provider/model/network/executable count.

### Open-model findings

`Qwen/Qwen3.5-9B` is an official Qwen repository, not a provisional display-name guess:

```text
repository:   Qwen/Qwen3.5-9B
revision:     c202236235762e1c871ad0ccb60c8ee5ba337b9a
created:      2026-02-27
license:      Apache-2.0
architecture: Qwen3_5ForConditionalGeneration
model type:   qwen3_5
parameters:   9,653,104,368
native max positions in config: 262,144
```

The selected revision provides four BF16 safetensors shards and explicit tokenizer/chat-template files. Their exact LFS SHA-256 values are bound in the evidence. The repository card calls the model 9B while the official API reports 9,653,104,368 parameters; the exact API total is preserved rather than rounding it into an architectural claim. No official static Qwen int4 variant is needed for the selected BF16 role. State advances only to `PENDING_LOCAL_SNAPSHOT`.

`meta-models/Muse-Glimmer-30B` and its GGUF repository are also official releases under Hugging Face's verified Meta Inc. organization:

```text
BF16 repository: meta-models/Muse-Glimmer-30B
BF16 revision:   a4e59da52a7bc87ae7251dd5545c0dd437c44b68
GGUF repository: meta-models/Muse-Glimmer-30B-GGUF
GGUF revision:   70bf1b61ac09f91b24d39038091b41c582bc5d7a
license:         Apache-2.0
architecture:    MuseGlimmerForConditionalGeneration
parameters:      29,776,626,688
```

The official repository resolves the earlier feasibility uncertainty. It publishes two static 4-bit builds plus a vision projector and an optional DFlash speculative drafter. The planned primary quantized target is the higher-fidelity Dynamic build:

```text
Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
bytes:  19,653,960,832
SHA-256:
  ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38

mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
SHA-256:
  f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6

dflash-Muse-Glimmer-30B-Q4_K_M.gguf
SHA-256:
  b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef
```

The official card reports approximately 20 GB for the Dynamic text-only build, approximately 22 GB with vision, and approximately 23 GB with vision plus drafter. These published estimates support a later A40/A100 resource preflight but are not local load evidence. Speculative decoding remains OFF for the scientific baseline and cannot enter results until a paired parity receipt passes. The BF16 and int4 roles both remain `PENDING_LOCAL_SNAPSHOT`.

### Closed-provider findings

Official provider documentation verifies three requested identities sufficiently to prepare—but not execute—a provider preflight:

```text
claude-sonnet-4-6  READY_FOR_PROVIDER_PREFLIGHT
claude-opus-4-8    READY_FOR_PROVIDER_PREFLIGHT
gemini-3.6-flash   READY_FOR_PROVIDER_PREFLIGHT
```

Anthropic explicitly documents `claude-sonnet-4-6` and `claude-opus-4-8` as dateless but pinned snapshots, not evergreen aliases. The Messages response schema exposes `model`, which must equal the requested pinned identity. Google lists `gemini-3.6-flash` as a stable specific model ID rather than a `latest` alias; `GenerateContentResponse.modelVersion` exposes the version used and must remain constant across accepted calls. These state changes authorize only a future identity/format preflight after separate network, credential, and hard-budget approval.

Two candidates remain blocked from preflight promotion:

```text
grok-4.5  PENDING_PROVIDER_QUALIFICATION
gpt-5.5   PENDING_OFFICIAL_IDENTITY
```

xAI officially lists `grok-4.5`, but its versioning documentation says undated IDs are mutable aliases for the newest stable release. No public dated `grok-4.5-YYYYMMDD` identity was verified, and the response `model` field is not documented to expose a resolved dated version. OpenAI's official model catalog retrieved on 2026-08-21 did not list `gpt-5.5`; an internal transfer-station alias therefore remains insufficient evidence.

Current pricing/context facts were inspected only for feasibility and were deliberately not frozen into the identity evidence because provider pricing is time-varying. A provider preflight must refresh and version its own price/budget table.

### Validation and review status

The implementation followed RED/GREEN development. Initial RED failed at module import; subsequent RED runs exposed the Phase 0 index computed-field serialization rule, exact-message test drift, evidence substitution, and lexical link handling. Final verification:

```text
python -m py_compile \
  mub/vnext/post_core/identity_v1.py \
  scripts/vnext_validate_post_core_identities.py \
  tests/vnext/test_post_core_identity_evidence.py
  PASS

python -m pytest tests/vnext/test_post_core_*.py -q
  65 passed, 1 skipped

git diff --check
  PASS
```

The one skip is the Windows symlink-construction case (`WinError 1314`) and is not counted as a pass; lexical reparse rejection remains implemented. The validator receipt contains eight candidates, three `PENDING_LOCAL_SNAPSHOT`, one `PENDING_OFFICIAL_IDENTITY`, one `PENDING_PROVIDER_QUALIFICATION`, three `READY_FOR_PROVIDER_PREFLIGHT`, and zero provider calls, model loads, network calls, or executable calls.

Multiple independent review attempts were made with separate local, inherited, remote, and alternate-profile agents. Every attempt terminated before reading the files because the subagent router selected unavailable `grok-4.6`. No independent review verdict exists for this unit; it must not be labeled `SPEC_COMPLIANT` or `CODE_QUALITY_APPROVED`. The completed evidence consists of official-source inspection, hash-frozen contracts, automated verification, and an inline adversarial review that closed evidence-substitution and lexical-link bypasses.

### Next hard gate

The document identity preflight is complete, but Phase 1 model execution has not started. Downloading Qwen or Muse files, allocating approved storage, authenticating a complete local snapshot tree, installing or changing Transformers/llama.cpp runtimes, or loading either model requires a separate explicit storage/download/execution authorization. Provider probes likewise require explicit network, credential-environment, and hard-budget authorization. Until those approvals exist, all execution counts remain zero and this document evidence is not scientific model evidence.

## Post-Core shared Hugging Face cache cleanup

### Motivation and authorization

The shared NAS reported only approximately 135 GB available while the next open-model panel requires Qwen3.5-9B BF16 and potentially Muse Glimmer int4/BF16 artifacts. The user explicitly selected the recommended deletion set after a read-only inventory. The cleanup was limited to `/NAS/yesh/hf_cache/hub`; no project artifact, frozen Core root, Task 11/12 model snapshot, result root, checkpoint, or another user's model path was targeted.

Before deletion, the inventory checked every cache root's exact `refs/main`, snapshot set, logical and allocated bytes. Project dependencies established two required cache roots:

```text
KEEP Qwen/Qwen2.5-7B-Instruct
  revision a09a35458c702b33eeacc393d103063234e8bc28
  reason: current Task 11 Qwen weights and long25 LoRA base model

KEEP sentence-transformers/all-MiniLM-L6-v2
  revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf
  reason: Pilot and heuristic CRUD encoder
```

The Task 11 Mistral v0.3 weights are an independent 28 GB project snapshot under `external/task11_answer_models`, not an HF-cache root. No running MemUpdateBench process was found.

### Exact deletion set

After reconfirming each target was a real direct child of the exact cache root, not a link, and still had the inventoried revision/snapshot set, the cleanup removed:

```text
models--NousResearch--Llama-2-7b-hf
  revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
  allocated 40,434,376,704 bytes

models--Qwen--Qwen2.5-7B
  revision d149729398750b98c0af14eb82c78cfe92750796
  allocated 15,242,833,920 bytes

models--facebook--contriever-msmarco
  revision abe8c1493371369031bcb1e02acb754cf4e162fa
  allocated 438,734,848 bytes

models--facebook--wav2vec2-base
  revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
  allocated 380,301,312 bytes

models--meta-llama--Llama-3.1-8B
  revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
  allocated 32,132,624,384 bytes

models--selfrag--selfrag_llama2_7b
  revision 190261383b0779ff66d2f95a73c7ad267d94b820
  allocated 13,477,765,120 bytes

models--meta-llama--Llama-2-7b-hf
  incomplete ref-only directory, no snapshot
  allocated 4,096 bytes
```

Total measured deletion:

```text
allocated bytes: 102,106,640,384
logical bytes:   102,106,422,168
approximate GiB: 95.1
```

NAS available space rose from `134,597,705,728` bytes immediately before deletion to `232,908,390,400` bytes in the cleanup script; the following independent `df` observation reported `232,998,699,008` bytes, reflecting concurrent filesystem activity.

### Receipt and independent verification

The operation wrote an atomic project-local receipt:

```text
/NAS/yesh/MemUpdateBench/external/post_core_storage_cleanup_20260821_v1.json
SHA-256:
  7ca169060d061852635872b1cfe13b068fa0a252f01af39de44d85593f3ba71e
payload self-hash:
  fa925cfccf742213bbeea3abda181ff87469ffe24fa6586022a7bd9701b13f23
```

A separate read-only verification confirmed all seven targets absent and exactly two HF model roots remaining. It also rechecked both required revisions, the long25 adapter file, and the independent Mistral snapshot. The final cache model roots are:

```text
models--Qwen--Qwen2.5-7B-Instruct
models--sentence-transformers--all-MiniLM-L6-v2
```

This cleanup only creates storage headroom. It does not authorize or perform Qwen3.5/Muse downloads, runtime installation, model loading, provider calls, or benchmark execution.

## Post-Core public open-model snapshots

### Scope and transfer boundary

The user subsequently authorized downloading the document-verified open models and requested that final weights live in the shared `/NAS/HuggingFaceModels` library rather than a personal cache. The cluster could not resolve or reach Hugging Face directly, so fixed-revision files were streamed from the local authenticated public connection through SSH into a resumable NAS staging root. The workstation never retained a complete weight file. No runtime package was installed or changed, and no model was loaded or invoked.

The exact source set remained:

```text
Qwen/Qwen3.5-9B
  revision c202236235762e1c871ad0ccb60c8ee5ba337b9a

meta-models/Muse-Glimmer-30B-GGUF
  revision 70bf1b61ac09f91b24d39038091b41c582bc5d7a
  selected Dynamic Q4_K_XL + mmproj + DFlash

meta-models/Muse-Glimmer-30B
  revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68
```

Transfers used remote-size-based range resumes. Multiple CDN TLS close events, one Tang-2 SSH disconnect, and several transient NAS `No space left on device` failures were handled by stopping the process, preserving exact partial bytes, and resuming only when capacity exceeded remaining bytes plus a safety reserve. No failed transfer exposed a final model root.

### Shared Qwen reuse

A read-only NAS search found several existing Qwen3.5-9B copies. `/NAS/HuggingFaceModels/Qwen3.5-9B` matched all 16 official files, four LFS shard hashes, tokenizer hash, ordinary Git blob OIDs, sizes, and the pinned revision manifest. It was adopted directly rather than downloading another final copy. The duplicated personal staging copy was removed only after both copies were fully rehashed.

```text
public path:
  /NAS/HuggingFaceModels/Qwen3.5-9B
file count:
  16
total bytes:
  19,329,393,661
tree SHA-256:
  e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db
binding receipt:
  /NAS/yesh/MemUpdateBench/external/post_core_shared_qwen_binding_20260822_v1.json
receipt SHA-256:
  924cb994248cf56d1d41df0da3bca06ce7abe1184cde5f0af489d4d364a1d9c2
```

The broader search found size-compatible copies under other users but did not rely on them. No reusable Muse Glimmer copy was found in the central library or during a time-bounded unique-filename NAS search.

### Muse download and public migration

The complete Muse GGUF package was downloaded first and verified against every frozen source digest:

```text
public path:
  /NAS/HuggingFaceModels/Muse-Glimmer-30B-GGUF
file count:
  5
total bytes:
  22,685,535,060
tree SHA-256:
  55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f
```

The BF16 snapshot then completed both official shards:

```text
model-00001-of-00002.safetensors
  49,950,112,952 bytes
  8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694

model-00002-of-00002.safetensors
  9,603,322,320 bytes
  b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381
```

All 13 BF16 files were first verified in the HF cache snapshot. Public flat snapshots were built through same-filesystem hardlinks in absent temporary roots and atomically renamed only after full membership, size, source digest, and SHA-256 validation. The public BF16 root was then reopened and rehashed before the personal cache duplicates were unlinked.

```text
public path:
  /NAS/HuggingFaceModels/Muse-Glimmer-30B
file count:
  13
total bytes:
  59,581,829,216
tree SHA-256:
  7a90420d22f8c98737f15bc31473bbe8a3579ee95f9bf2237172679709877782
download receipt:
  /NAS/yesh/MemUpdateBench/external/post_core_open_snapshot_download_20260821_v1.json
receipt SHA-256:
  c8a16cda9dbb5646305d29a3c2e97d4ea7a92c10ae7fd3a43673c4a159f4f0a6
```

A finalize-script Git blob header bug (`\\0` rather than NUL) initially rejected a correct GGUF license file after download. Independent Git blob calculation proved the file and hardlink were exact. The verifier was corrected; no model data was rewritten. A separate cache-snapshot symlink handling bug was also closed by strictly resolving HF snapshot links to authenticated blobs before creating public hardlinks.

### Final receipt and independent audit

The source-bound public closure receipt is:

```text
/NAS/yesh/MemUpdateBench/external/post_core_public_open_models_20260822_v1.json
SHA-256:
  77a69e02a8b092b7e1bf5e89ff9a5f69b449c89a1c2cd319f9c48edd3e2f4645
```

A second independent remote program then reopened and rehashed all 34 public files—approximately 101.6 GB—without reusing the finalizer result. It verified the three exact tree hashes, four source receipts, preserved dependencies, absence of personal Muse cache/staging duplicates, and available disk space.

```text
independent audit:
  /NAS/yesh/MemUpdateBench/external/post_core_public_open_models_20260822_audit.json
audit SHA-256:
  0b146bd8dc04e3343d899801f4746bee0ae69635f1ace3f4c92ada8f32819940
audit payload self-hash:
  392a4d11205803cf2559dff8843a87529ee93daefe1a27c152ca3b5363d588d6
dependencies preserved:
  true
personal duplicates absent:
  true
model loads:
  0
provider calls:
  0
available bytes at audit:
  102,711,689,216
```

Post-audit structural checks again confirmed the three public roots are real directories, both personal Muse cache roots and the download staging root are absent, Qwen2.5-Instruct and MiniLM retain their exact revisions, and long25 plus the frozen Task 11 Mistral snapshot remain present. These snapshots are authenticated storage/preflight inputs only; they are not load, capability, accuracy, prompted-answer, or benchmark evidence.

## Tang-3 GPU6 Qwen offline load preflight

### Authorization and shared-GPU boundary

The user explicitly designated `Tang-3-Wu` GPU6 for testing and authorized co-use with existing low-memory jobs. The two existing `tuzx` processes were not terminated or altered. At preflight start, GPU6 was an NVIDIA A40 with 46,068 MiB total and approximately 44.5 GiB free; the other processes each occupied approximately 454 MiB. The test did not start a benchmark, generate a prompt answer, or call a provider.

The preflight ran from an already installed `routertc` environment, without modifying the shared `gmsra` environment or installing packages. The runtime was:

```text
Python:       3.10
Transformers: 5.9.0
PyTorch:      2.5.1+cu121
Accelerate:   1.13.0
CUDA device:  NVIDIA A40, visible as CUDA device 6
```

The first attempt stopped before weight loading because `accelerate` was absent from `gmsra`; it was not treated as a passing run. A second isolated script-edit attempt initially failed with a local `NameError` (`model.to(cuda)`), also before GPU model loading. After correcting the isolated preflight script, the real load succeeded.

### Qwen result

```text
model:
  Qwen/Qwen3.5-9B
revision:
  c202236235762e1c871ad0ccb60c8ee5ba337b9a
public snapshot:
  /NAS/HuggingFaceModels/Qwen3.5-9B
tree SHA-256:
  e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db
class:
  Qwen3_5ForConditionalGeneration
parameters loaded:
  9,409,813,744
```

The model loaded completely from local files with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`, and `trust_remote_code=False`, then was moved to CUDA device 6, set to eval mode, and unloaded. No prompt or generation was performed.

```text
free before load:       46,373,076,992 bytes
memory allocated:       18,831,103,488 bytes
memory reserved:        18,884,853,760 bytes
free after load:        27,488,223,232 bytes
free after unload:      46,373,076,992 bytes
model loads:            1
generations:            0
provider calls:         0
network calls:          0
```

The optional fast-path packages `flash-linear-attention` and `causal-conv1d` were absent, so Transformers reported a torch fallback. This is a runtime/throughput caveat and must be fixed or explicitly frozen before any benchmark throughput claim.

The receipt is:

```text
/NAS/yesh/MemUpdateBench/external/post_core_qwen35_gpu6_load_preflight_20260823.json
SHA-256:
  fd4e47d75d86efdbe9add3cc469017b9aef23bb05bc4d03b74877bfbe289f6b7
payload self-hash:
  f8cf70557a713a910a75e37adfe47b08e1758003fcd63146d999db5f40ac579a
```

### Muse boundary

Muse BF16 is approximately 59.6 GB and cannot fit on this A40 alongside the existing shared processes; it was not loaded. Muse GGUF is public and fully hash-audited, but `llama.cpp` executables were not installed on Tang-3 and no GGUF load was attempted. The next Muse preflight requires an approved llama.cpp build/runtime and explicit device/context limits; it remains separate from Qwen's successful load-only check.

### Conclusion

Qwen3.5-9B passes an offline load/unload preflight on shared Tang-3 GPU6. This proves only local snapshot readability, architecture/runtime recognition, and one co-use load boundary. It is not a prompted-answer result, capability score, throughput result, or benchmark evidence. No Phase 1 benchmark execution has started.

## Post-Core closed-provider local and Tang-2 capability preflight (2026-08-23)

### Motivation and boundary

After the no-execution Phase 0 registry, frozen official-document identity gate, authenticated open snapshots, and Qwen load-only preflight, the user explicitly authorized bounded connectivity/interface checks for five selected API-Transfer-Station routes. These probes tested only whether an Anthropic Messages-compatible request could traverse the provider and return a short parseable response. They did not execute a MemUpdateBench task, measure accuracy, authenticate every upstream model identity, or authorize a broader provider matrix.

The credential was read from the selected CC Switch provider record into process memory. For Tang-2 it was passed only through SSH standard input to the remote in-memory process. No credential value, credential-bearing command, raw response, or provider configuration was written to a local or remote artifact or printed in probe output.

### Model names and provenance

The exact transfer-station request names were:

```text
claude-sonnet-4-6
claude-opus-4-8
Gemini 3.6 Flash (Low)
grok-4.5
gpt-5.5
```

Gemini must retain both provenance layers:

```text
canonical model identity:          gemini-3.6-flash
transfer-station request name:     Gemini 3.6 Flash (Low)
transfer-station reasoning tier:   Low
```

The Low route distinguishes this request from transfer-station Medium/High variants that force greater reasoning effort. It must not be silently collapsed into the canonical identity in raw call provenance.

### Local checks

Each route received one minimal request with prompt `Reply only: OK`, `max_tokens=8`, and zero retries. All five returned HTTP 200, exact `OK`, and a response `model` matching the requested transfer name. Sonnet, Opus, Gemini, and Grok returned standard Anthropic Messages JSON.

GPT-5.5 returned a valid complete SSE stream even when non-streaming behavior was expected. One additional local diagnostic explicitly set `stream=false`; GPT-5.5 again returned `text/event-stream`, but the six-event Anthropic stream parsed without error, reported `model=gpt-5.5`, ended with `end_turn`, and produced exact `OK`. This established a response-format defect, not a failed model route.

### Tang-2 checks and GPT fix verification

The cluster check ran from `Tang-2-Wu` (hostname `Tang2`) against the same endpoint and request names. An initial SSH command-quoting attempt failed before the remote Python probe executed and therefore generated zero provider calls. The corrected in-memory script then issued one request per model with explicit `stream=false`, `max_tokens=8`, and zero retries.

All five routes returned HTTP 200, exact `OK`, and matching response model names. Sonnet, Opus, Gemini, and Grok returned standard JSON. GPT-5.5 still returned a complete valid SSE response at that point. After the transfer-station implementation was fixed, one bounded Tang-2 GPT-5.5 retest returned:

```text
HTTP status:       200
Content-Type:      application/json
response type:     message
response model:    gpt-5.5
stop reason:       end_turn
text:              OK
usage:             present
stream requested:  false
retries:           0
```

The known GPT-5.5 `stream=false` format blocker is closed for this endpoint and Tang-2 path.

### Exact call accounting

```text
claude-sonnet-4-6:          2
claude-opus-4-8:            2
Gemini 3.6 Flash (Low):     2
grok-4.5:                   2
gpt-5.5:                    4
real provider calls:       12
retries:                    0
benchmark generations:      0
```

The failed SSH quoting attempt is excluded because it stopped before provider execution.

### Conclusions and next steps

The five transfer routes pass bounded local and Tang-2 connectivity/interface checks. Claude Sonnet 4.6 and Opus 4.8 retain their document-verified identities; Gemini retains the canonical/request-name/Low-tier mapping above. Matching transfer request and response fields do not authenticate Grok 4.5 or GPT-5.5 as immutable official upstream identities, so their identity blockers remain explicit.

The next formal unit is a separate Post-Core Phase 1–2 qualification release. It should publish no-replace redacted provider receipts, freeze an approved llama.cpp runtime for Muse GGUF with speculative decoding off, advance Qwen from load-only to bounded generation/template/parser/determinism checks, run the same small smoke across qualified roles, and emit typed `READY`, `BLOCKED`, or `UNSUPPORTED` decisions. Only then should the planned 320-generation canary and three-condition confirmatory hard subset be authorized. None of this evidence may be retrofitted into immutable Core, Task 9–14, Phase 0, or `official_identity_evidence_v1.json`.

## Post-Core Phase 1–2 qualification release implementation (2026-08-24)

### Motivation and boundary

Task 10 registers the offline qualification contracts in the normal smoke path without modifying frozen Core or Phase 0 evidence. The implementation is offline, source-bound, and no-replace. It authorizes no provider, adapter, model, credential, network request, benchmark generation, canary, confirmatory subset, or full matrix.

The prior 12 calls are imported as source-bound aggregate connectivity/interface attestations in this qualification release. They are not benchmark/scientific accuracy or release-`READY` evidence.

### Files changed/generated

```text
configs/vnext/post_core/qualification_release_v1.json
mub/vnext/post_core/qualification_receipts_v1.py
mub/vnext/post_core/qualification_validation_v1.py
mub/vnext/post_core/qualification_planning_v1.py
mub/vnext/post_core/qualification_decisions_v1.py
mub/vnext/post_core/qualification_release_v1.py
scripts/vnext_prepare_post_core_qualification_release.py
scripts/vnext_run_post_core_capability_smoke.py
scripts/smoke_test.py
tests/vnext/qualification_fixtures.py
tests/vnext/test_post_core_qualification_receipts.py
tests/vnext/test_post_core_qualification_validation.py
tests/vnext/test_post_core_qualification_runtime.py
tests/vnext/test_post_core_qualification_planning.py
tests/vnext/test_post_core_qualification_decisions.py
tests/vnext/test_post_core_qualification_release.py
tests/vnext/test_post_core_qualification_release_cli.py
tests/vnext/test_post_core_capability_smoke_cli.py
docs/superpowers/specs/2026-08-24-memupdatebench-post-core-phase12-qualification-design.md
docs/superpowers/plans/2026-08-24-memupdatebench-post-core-phase12-qualification.md
WORKFLOW.md
```

The smoke executable runs these exact isolated offline test files:

```text
tests/vnext/test_post_core_qualification_receipts.py
tests/vnext/test_post_core_qualification_validation.py
tests/vnext/test_post_core_qualification_runtime.py
tests/vnext/test_post_core_qualification_planning.py
tests/vnext/test_post_core_qualification_decisions.py
tests/vnext/test_post_core_qualification_release.py
tests/vnext/test_post_core_qualification_release_cli.py
tests/vnext/test_post_core_capability_smoke_cli.py
```

No real adapter is invoked.

### Fresh offline verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/prepare_data.py \
  scripts/eval_evomemory.py \
  scripts/generate_constrained_sft.py \
  scripts/train_constrained_sft.py \
  scripts/analyze_ood_errors.py \
  scripts/summarize_update_frequency.py \
  scripts/smoke_test.py \
  mub/vnext/post_core/qualification_v1.py \
  mub/vnext/post_core/qualification_receipts_v1.py \
  mub/vnext/post_core/qualification_validation_v1.py \
  mub/vnext/post_core/qualification_planning_v1.py \
  mub/vnext/post_core/qualification_decisions_v1.py \
  mub/vnext/post_core/qualification_release_v1.py \
  scripts/vnext_prepare_post_core_qualification_release.py \
  scripts/vnext_run_post_core_capability_smoke.py
# PASS: exit 0, no output

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest \
  tests/vnext/test_post_core_qualification_receipts.py \
  tests/vnext/test_post_core_qualification_validation.py \
  tests/vnext/test_post_core_qualification_runtime.py \
  tests/vnext/test_post_core_qualification_planning.py \
  tests/vnext/test_post_core_qualification_decisions.py \
  tests/vnext/test_post_core_qualification_release.py \
  tests/vnext/test_post_core_qualification_release_cli.py \
  tests/vnext/test_post_core_capability_smoke_cli.py -q
# 305 passed, 6 skipped, 0 failed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest tests/vnext/test_post_core_*.py -q
# 370 passed, 7 skipped, 0 failed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/smoke_test.py
# SMOKE TEST: 32/32 passed

git diff --check
# PASS
```

The targeted-test skips are documented Windows host limitations: unavailable symlink creation, Windows descriptor sharing for the concurrent replacement probe, unavailable directory `fsync`, and a controlled output-parent replacement that Windows denies while the reserved file handle is open; they are not counted as passes. The full Post-Core regression suite adds one unavailable identity-evidence symlink-creation skip.

The planned combined Post-Core plus frozen Task 14 regression command was also attempted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest \
  tests/vnext/test_post_core_*.py \
  tests/vnext/test_core_task14_atomic.py \
  tests/vnext/test_core_task14_sources.py -q
# 304 passed, 7 skipped, 7 failed, 6 errors
```

The 13 Task 14 failures/errors are pre-existing worktree-fixture unavailability, not qualification-code failures: `data/vnext/core/v3` is absent, so strict Task 14 source loading raises `FileNotFoundError [WinError 3]`. The focused Post-Core suite passes independently. Frozen Task 14 code and sources were not modified.

The authoritative open-snapshot audit receipt was reread on Tang-2 from `/NAS/yesh/MemUpdateBench/external/post_core_public_open_models_20260822_audit.json`. A fresh remote `sha256sum` recovered the exact 64-character SHA-256 `0b146bd8dc04e3343d899801f4746bee0ae69635f1ace3f4c92ada8f32819940`; the checked-in qualification configuration had omitted the seventeenth character (`3`) and is now corrected. No hash was guessed or padded. This removes the malformed-hash configuration blocker but does not authorize publication, model/provider execution, or capability smoke.

Validation performed zero model loads, provider calls, network calls, benchmark generations, and credential reads. The capability adapter transport now carries the canonical prompt payload and parser contract in each hash-bound request envelope, and the provider source-bound JSONL can preserve the typed zero-call pre-provider setup failure alongside the five capability attestations. Live short-generation and 8–16 smoke are `NOT_RUN` and require explicit authorization. All eight current `CAPABILITY_SMOKE` decisions remain `BLOCKED`; provider connectivity and open-model short-generation readiness cannot promote an unexecuted smoke to `READY`. The frozen zero-cost/PENDING budget also blocks all closed API attempts until an authorized price-version and nonzero hard cost ceiling are published. The 320 canary, confirmatory, and full matrix are unauthorized.

`EXIT10` remains reserved. A `SUCCESS_WITH_BLOCKERS` metadata release returns `0`.

### Conclusion

The offline/source-bound/no-replace qualification-release implementation is complete and registered in the smoke path. The malformed audit-hash configuration blocker is resolved from an authoritative remote rehash, but no publication or execution is authorized by that correction. Live short-generation, capability smoke, canary, confirmatory, and benchmark evidence remain `NOT_RUN`/`BLOCKED` under their separate gates.

## Post-Core Phase 1–2 qualification publication and Tang-1 runtime gates (2026-08-26)

### Boundary

PR #1 was merged into `master` at `0981a384262bdfbd2c52d5aa8aed64861a6d342b`. The user authorized only Qwen/Muse runtime qualification on Tang-1-Wu GPU3. This did not authorize closed-provider API calls, capability smoke, the 320-generation canary, confirmatory conditions, or benchmark execution. No other user's process was terminated or altered; every model command had a hard timeout and Muse speculative decoding stayed off.

### Source recovery

The immutable input bundle is `D:/USTC/2026Winter/MemUpdateBench_qualification_inputs/phase12_0981a38_v1`. Frozen inputs were recovered without regeneration: the Core manifest came from the authenticated NAS Core release; the exact workflow blob came from `a568574:WORKFLOW.md` with its original CRLF bytes; the handoff source came from `docs/handoffs/2026-08-23-post-core-provider-probes-next-window-prompt.md`; and the open-snapshot closure/audit plus Qwen load-only receipts were recopied from their authenticated NAS paths and rehashed.

The provider JSONL retains one typed zero-call setup failure and exactly five aggregate connectivity/interface attestations: 12 historical calls, zero retries, zero benchmark generations, Gemini's canonical/request/tier triplet, and GPT-5.5's SSE-to-JSON sequence. It contains no credentials, raw provider responses, invented request IDs, or benchmark metrics.

### Qwen3.5-9B

```text
model: Qwen/Qwen3.5-9B
revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
snapshot tree: e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db
runtime: Python 3.10.20 / Torch 2.5.1+cu121 / Transformers 5.9.0 / Accelerate 1.13.0
attention: eager
trust_remote_code: false
device: Tang-1-Wu GPU3 / NVIDIA A40
```

The canonical exact-output prompt produced `READY` twice under greedy seed-0 generation. Each repetition generated four tokens. Peak allocated memory was `17,963,988,992` bytes. A process-exit wrapper verified unload closure at 3 MiB and zero compute processes; the earlier in-process CUDA-context reading was not used as unload evidence.

```text
load: PASS
generation: PASS
determinism: PASS
unload: PASS
provider calls: 0
network calls: 0
```

Source receipt: `/NAS/yesh/MemUpdateBench/external/post_core_qwen35_tang1_gpu3_runtime_gate_20260826_v2.json`, SHA-256 `5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f`.

### Muse Glimmer GGUF

Official llama.cpp was frozen at release `v0.3.0`, commit `c1d0e7a004015f23bc0233470b747b596f29b264`. A project-local CMake 3.31.6 toolchain built a CUDA 12.0/A40 architecture-8.6 Release binary with no curl, tests, or speculative drafter:

```text
/NAS/yesh/MemUpdateBench/external/llama_cpp_v0.3.0_c1d0e7a/build-cuda-tang1/bin/llama
SHA-256 feab512b206b6a5e1714f8099dfaca62ad62a92c5353f2fd522a1890f6a3f3d2
```

The authenticated snapshot contains the 19GB `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf`; the README-mentioned 17GB Q4_K_M file was not present and was not substituted. Two 128-token greedy seed-0 repetitions produced identical thinking traces, and the parser extracted exact `READY` after `[End thinking]`. A third identical run measured peak memory.

```text
load: PASS
generation: PASS
determinism: PASS
unload: PASS
parsed projection: READY
generated token count: 106
peak memory: 18,803 MiB / 19,716,374,528 bytes
post-process GPU memory: 3 MiB
post-process compute processes: 0
speculative decoding: off
provider calls: 0
network calls: 0
```

Source receipt: `/NAS/yesh/MemUpdateBench/external/post_core_muse_glimmer_tang1_gpu3_runtime_gate_20260826_v1.json`, SHA-256 `3bae8741362aafe9d3ff11e2535c898c55ce1bcadf5112efde49de470e81acfe`.

Muse BF16 remains typed `BLOCKED`: its 59.6 GB snapshot exceeds the A40 capacity, so no load was attempted and no missing value was converted to zero or `UNSUPPORTED`.

### Qualification release

The no-replace local root is `D:/USTC/2026Winter/MemUpdateBench_releases/post_core_qualification_0981a38_v1`.

```text
status: SUCCESS_WITH_BLOCKERS
qualification index SHA-256: 872b3e55cb5322883b5dfacf2e5962bc6b1561f94bf23c8704b43e16436302bd
release tree SHA-256: fa2df961831ca045d6a402908238c2301a5a20facfe2ae34e8061538ba2aa833
READY/BLOCKED/UNSUPPORTED: 8 / 24 / 0
publication model loads/provider calls/network calls/credential reads/benchmark generations: 0 / 0 / 0 / 0 / 0
```

The release was rebuilt from the frozen source bundle and compared byte-for-byte after publication. Closure receipt: `/NAS/yesh/MemUpdateBench/external/post_core_phase12_qualification_closure_20260826_v1.json`, SHA-256 `616789469753b2fb1585706fb8ec02c0e6bc365e03b9c18d39628db5bf640880`, payload self-hash `d579b7da0c0b163037548aa7718084b89c9d26c5eade36ae44d54e51fcc9069a`.

### Conclusion

Qwen3.5-9B and Muse GGUF are READY only for the uniform capability-smoke gate. Muse BF16 remains resource-blocked. All capability-smoke decisions remain `BLOCKED/NOT_RUN`; closed-provider smoke requires explicit API/budget authorization and open-model smoke requires separate GPU authorization. Canary, confirmatory, and benchmark stages remain `NOT_RUN`.

## Post-Core uniform BASE capability smoke (2026-08-26)

### Scope and explicit exclusions

The user authorized Tang-1-Wu GPU3 for Qwen/Muse open-model smoke and five closed transfer routes for one complete eight-call BASE batch each. The campaign used the frozen four qualification fixtures with two repetitions per fixture:

```text
7 roles × 4 fixtures × 2 repetitions = 56 attempts
open-model generations: 16
provider calls: 40
retries: 0
escalations: 0
canary generations: 0
benchmark generations: 0
```

No pricing or cost-accounting subsystem was added to MemUpdateBench. External API spending was monitored by the user and remains outside benchmark semantics. No raw provider response, endpoint, token, Authorization value, or provider configuration was persisted.

### Qwen3.5-9B

Qwen ran on Tang-1-Wu GPU3 from the authenticated snapshot and the already-qualified private Transformers 5.9 runtime. The adapter loaded the model once, executed the exact eight BASE attempts, and unloaded.

```text
attempts: 8
PASS: 8
FAIL: 0
retries: 0
status: READY at capability-smoke scope
```

A first controller invocation failed before model loading because the runner intentionally strips `PYTHONPATH` and `CUDA_VISIBLE_DEVICES`; it produced zero model generations and no result receipts. A replacement authorization bound a corrected adapter that explicitly selected physical GPU3 and its private dependency directory. This technical pre-dispatch event is retained in the closure receipt and is not counted as a model attempt or retry.

### Muse Glimmer GGUF

Muse used the pinned CUDA llama.cpp runtime and authenticated Dynamic Q4_K_XL snapshot on Tang-1-Wu GPU3, with speculative decoding off. It executed the frozen fixture limit of 32 generated tokens per attempt without extension or retry.

```text
attempts: 8
PASS: 0
FAIL: 8
error class: PARSER_MISMATCH × 8
retries: 0
status: BLOCKED at capability-smoke scope
```

The earlier 128-token runtime gate showed that Muse can deterministically produce a final `READY`, but the frozen 32-token smoke fixture ends during the model's thinking trace. The smoke result is therefore preserved as a parser/capability mismatch rather than repaired by increasing the token limit.

### Closed transfer routes

Each route received exactly eight Anthropic Messages-compatible BASE requests with `stream=false`, `max_tokens=32`, and zero retries. Credentials and base URL were read from the current CC Switch provider record into process memory only. Receipts retain typed status, request/projection hashes, returned model, response format, usage counts when present, latency, and HTTP status; raw responses are absent.

```text
claude-sonnet-4-6:      0 PASS / 8 FAIL / HTTP 404 × 8
claude-opus-4-8:        0 PASS / 8 FAIL / HTTP 404 × 8
Gemini 3.6 Flash Low:   0 PASS / 8 FAIL / HTTP 404 × 8
grok-4.5:               0 PASS / 8 FAIL / HTTP 404 × 8
gpt-5.5:                8 PASS / 0 FAIL / HTTP 200 × 8
```

The four 404 route batches are retained as current transfer-route capability failures. They were not retried or replaced with aliases. GPT-5.5 returned standard Anthropic Message JSON, exact route-name binding, normal stop reason, and valid fixture projections for all eight calls. This establishes only the mutable transfer route's current capability; it does not authenticate GPT-5.5 as an immutable official upstream identity or unblock benchmark admission.

### Closure receipt

```text
/NAS/yesh/MemUpdateBench/external/post_core_capability_smoke_base_20260826_v1.json
SHA-256 8943b7a734dc2b38381818cad9b53821512e0e5db3bd7f1bb3a6e3efd3b2bcb9
payload self-hash 8e9d117de0660b9c81a4a6e92c2a7c63db52bcaf727dedfb7747e21eb5104d5e
```

Final campaign counts are 56 attempts, 16 passes, 40 failures, 40 provider calls, 16 open-model generations, zero retries, zero escalation, zero canary generation, and zero benchmark generation.

### Consequence for the next phase

Only Qwen3.5-9B and the GPT-5.5 transfer route pass this exact uniform BASE smoke. Muse GGUF and the other four closed routes remain typed `BLOCKED`. The planned 320-generation Qwen canary may proceed only under a separate GPU authorization; no failed or identity-blocked route is admitted to that canary.

## Post-Core LangMem 0.0.30 profile admission (2026-08-26)

### Scope and verified boundary

LangMem is the package identity gate for the first independent external-memory pipeline admitted after Core. This is a bounded capability/admission result only, not a benchmark run or accuracy claim. The implementation uses LangGraph's `InMemoryStore` through a custom MemUpdateBench adapter and the visible-only JSONL bridge; native LangMem runtime APIs are not used for CRUD or retrieval. It runs no LLM, API, GPU, network credential, or model input.

The admitted surface is explicitly **profile/single-record only**: one declared target object per adapter and one custom-adapter record per runtime namespace. Collection mode, multi-object queries, scoped delete, historical query, native answering, LLM enrichment, and semantic embedding search are all typed unsupported. The adapter performs deterministic local profile reconciliation/export and token-overlap retrieval over the LangGraph store.

### Identity and installation checks

The frozen identity is `langchain-ai/langmem` package `0.0.30`, source commit `29cbe41e58528f92e9efa773c12e15c47be3808c`, MIT. The isolated project-local installation reported package version `0.0.30`, `Requires-Python >=3.10`, and MIT license text; its installed LangMem content digest was `3e6d6cf4d81a1cd77ebc2b98b68b17043841b34239a2499c3a91a3cdc601847a` over 27 tracked files. The pinned wheel SHA-256 is `142f040014493eebd67e1055c0642f9ab38868b5b1fde5c8f2d39add57f4ba5b`. The public source commit was separately checked to carry version `0.0.30`; the installed wheel does not itself carry a VCS direct-URL record, so the receipt preserves the explicit pinned source lock rather than claiming an embedded wheel commit.

Any mismatch in package version, installed content, source lock, or MIT license is fail-closed as `BLOCKED`.

### Commands and results

```text
python -m pytest tests/vnext/test_langmem_configuration_v1.py tests/vnext/test_langmem_worker_protocol.py tests/vnext/test_langmem_adapter_v3.py tests/vnext/test_langmem_preflight_cli.py tests/vnext/test_langmem_admission_receipt.py -q
13 passed

python scripts/vnext_preflight_langmem.py --python-executable <isolated-python> --output <no-replace-preflight> --run-prefix langmem-0-0-30-admission --timeout-seconds 30.0
python scripts/vnext_admit_langmem_profile.py --preflight <no-replace-preflight> --output <no-replace-receipt>
```

The real isolated-package preflight passed the exact `ADD → UPDATE → NOOP → DELETE` lifecycle. The profile record ID was stable across add/update, exported the native current value `Lyon`, returned the same value under `slot_direct`, and was absent after delete. The 20-trial fresh-namespace reset probe passed completely, including target/control isolation and post-reset control preservation. No private filesystem path appears in either published artifact.

### No-replace evidence

```text
results/vnext/post_core_langmem_0_0_30_profile_admission_20260826/preflight.json
SHA-256 3d9b4628e0a1f8226eedda8d1f65abdbf76e71e0b95ce5b6594d5b25b374c1fa

results/vnext/post_core_langmem_0_0_30_profile_admission_20260826/admission_receipt.json
SHA-256 6fcf89b8c69ba68f65103a16a4402b293c33dff488062be9dfd022dd1480fe8f
```

The receipt outcome is `pass` for `profile_single_record_only`. It must not be interpreted as collection-mode admission, prompted-answer evidence, a full benchmark result, or a modification of Mem0/Core evidence.

## LangMem 0.0.30 profile Family-A canary (2026-08-26)

After profile-mode admission, a manager-only canary requested 32 authenticated Family-A tasks covering eight semantic cores and four surfaces per core. No LLM, API, GPU, embedding service, or prompted-answer model was used.

The admitted LangMem configuration is explicitly `profile_single_record_only`. Eight requested tasks from one multi-object semantic core were therefore emitted as `NOT_SUPPORTED` rather than zero-valued failures. The remaining 24 single-object tasks reached the visible external boundary, but all natural-language `raw_text` events were rejected before state creation because the bounded no-LLM worker accepts only explicit CRUD action surfaces. The benchmark's compiled `normalized_text` was not substituted because doing so would expose structured action information that a genuine external manager must infer from raw text.

```text
requested tasks: 32
supported-scope tasks: 24
NOT_SUPPORTED multi-object tasks: 8
PASS: 0
FAIL: 24
failure class: VISIBLE_SURFACE_UNSUPPORTED × 24
prompted answering: BLOCKED_BY_INGEST
state/retrieval metrics: typed null, not zero
LLM/API/GPU use: false / false / false
```

Evidence:

```text
results/vnext/post_core_langmem_family_a_canary_20260826/rows.jsonl
  SHA-256 1c1eded0602260c1f09be2ecfde8f9427288ac13e7da18198c186c2027cf0c36
results/vnext/post_core_langmem_family_a_canary_20260826/canary_receipt.json
  payload SHA-256 60cdf6adfda75d45ca083dab4454dd9ff507cf3a7c4826fb904448b57ae6f842
```

This is a genuine external-system pipeline result: the LangMem package identity gate is paired with LangGraph `InMemoryStore` and custom adapter logic that passes explicit profile CRUD admission but cannot consume the benchmark's natural update surface without an extraction/enrichment model. A manager-quality row requires a separately qualified LLM extraction configuration; the direct profile result must not be reported as accuracy zero or silently repaired with normalized gold-derived actions.

## LangMem 0.0.30 + Qwen3.5 extraction Family-A canary (2026-08-27)

The direct no-LLM profile configuration could not ingest natural benchmark `raw_text`, so a separate manager-quality configuration was qualified with Qwen3.5 as a visible-event CRUD extractor over the LangGraph `InMemoryStore` custom adapter. The extractor sees only the raw event and the single allowed profile attribute; it returns strict JSON with `operation` and scalar `value`. The controller binds the one admitted object ID. The custom adapter performs profile create/update/noop/delete, state export and deterministic retrieval; the LangMem package remains an identity gate only. Empty-store UPDATE is reconciled to CREATE as part of the explicitly versioned manager configuration.

A four-step admission sequence passed:

```text
Alice moved to Paris.        -> ADD scalar "Paris"
Alice now lives in Lyon.     -> UPDATE scalar "Lyon"
No memory object changes.    -> NOOP, value remains "Lyon"
Delete Alice city.           -> DELETE, store empty
```

The 32-task Family-A canary ran on Song-1-Wu GPU3 using the authenticated Qwen3.5 snapshot. Eight multi-object tasks remained explicit `NOT_SUPPORTED` under profile-single-record scope. All 24 supported single-object tasks completed successfully.

```text
requested tasks: 32
supported tasks: 24
NOT_SUPPORTED: 8
PASS: 24
FAIL: 0
state accuracy: 1.000
gold retrieval rate k16: 1.000
average final memory size: 1.000
provider/API calls: 0
retries: 0
```

Evidence:

```text
results/vnext/post_core_langmem_qwen_extraction_canary_20260827_v3/rows.jsonl
  SHA-256 7e266b6926954c9ce9b6c537a3ee70093bce4ae8bef673ed4669b6152ceb3b67
results/vnext/post_core_langmem_qwen_extraction_canary_20260827_v3/canary_receipt.json
  payload SHA-256 a582c0e42e73a3d83d4a5e44279dd3651aba6a5c3b086650d78e24a2c6dbd4a5
```

Two earlier execution roots are superseded diagnostics: Tang-1 lacked sufficient free memory at model load, and Song-1 initially exposed an incompatible optional sklearn/pyarrow import from the shared environment. Both stopped before task extraction. The final Song-1 runtime used a source-bound optional sklearn stub only to prevent importing the unused incompatible pyarrow path; no sklearn functionality was invoked.

This is the first genuine external manager-quality state/retrieval row for the LangGraph-store/custom-adapter pipeline. Prompted answering and the complete 80-task Family-A matrix remain separate later gates. Unsupported multi-object work is not pooled as zero.

## Letta 0.16.8 direct-block candidate gate (2026-08-27)

Letta 0.16.8 is frozen as an Apache-2.0 external-system candidate, but remains `BLOCKED`. No Letta package/server/model was started and no runtime capability is claimed. The metadata-only preflight reports `letta_package_not_installed`; official direct-block bootstrap is intentionally unimplemented.

The checked-in protocol fake exists only for adapter/CRUD contract tests. It uses a non-Letta health identity and cannot authenticate through the production adapter. The profile backend rejects a second distinct block in one namespace. Preflight and receipt writers reject immutable-Core destinations, use create-exclusive writes, fsync files and POSIX parent directories, and scan evidence for secrets. Admission v1 is fail-closed even if a hand-authored object claims all runtime booleans passed: official bootstrap remains a blocker.

```text
results/vnext/post_core_letta_0_16_8_block_profile_admission_20260827_v3/preflight.json
results/vnext/post_core_letta_0_16_8_block_profile_admission_20260827_v3/admission_receipt.json
outcome: BLOCKED
runtime/accuracy/prompted-answer evidence: NOT_RUN
```

The next valid step is an isolated Python 3.11–3.13 Letta installation and authenticated source/content check, followed by a real CPU-only direct-block client bootstrap. Until then Letta must not appear as an admitted capability or accuracy row.

## LangMem 0.0.30 + Qwen3.5 full Family-A supported matrix (2026-08-27)

The manager-quality LangGraph-store/custom-adapter configuration was extended from the 32-task canary to the complete authenticated 80-task Family-A view. Qwen3.5 performs visible raw-event CRUD extraction and the fixed prompted-answer layer; the custom adapter performs profile create/update/noop/delete, state export and deterministic search over LangGraph's `InMemoryStore`. The controller binds the single admitted object ID and requests a scalar attribute value, never compiled `normalized_text` or gold actions.

```text
requested tasks: 80
supported single-object tasks: 52
NOT_SUPPORTED multi-object tasks: 28
PASS on supported tasks: 52/52
state accuracy: 1.000
k16 gold retrieval rate: 1.000
average final memory size: 1.000
prompted-answer EM: 1.000
provider/API calls: 0
retries: 0
```

Evidence:

```text
results/vnext/post_core_langmem_qwen_full_family_a_20260827_v1/rows.jsonl
  SHA-256 0560fd3915cce125d0039d40a87898369eb90f15a10cd6abfb91e3027108ef07
results/vnext/post_core_langmem_qwen_full_family_a_20260827_v1/canary_receipt.json
  file SHA-256 c7e2e89e5cb518b7d101cf5146be6c77f898cd7581b07d888960a49ad75da59b
  payload SHA-256 5f4551783c816f4325463903803c7f1116d400d792ab02e7160667ba4b2953ea
```

This establishes a genuine external manager-quality row within the LangGraph-store/custom-adapter pipeline with a verified LangMem package dependency. Coverage is 65% of the Family-A view; the remaining 35% is explicit multi-object unsupported work and is not scored as zero. The result still shares Qwen3.5 for extraction and answering, so manager comparison must separate state/retrieval behavior from answer-model effects.

## Letta 0.16.8 real-package/server preflight update (2026-08-27)

The pinned Letta package was installed into an isolated Python 3.13 project overlay. Package metadata matched version `0.16.8`, Apache license metadata and the declared Python range; 568 installed files produced content SHA-256 `ff3b8f431407d886088ee53e61a80c8ad03f581597627f5112e0c691b55e675f`. The installed content is not yet cryptographically bound to the pinned source commit, so source identity remains incomplete.

A localhost-only server bootstrap on a verified free port imported the package successfully but failed before listening because Letta selected PostgreSQL and authentication for the default Letta database user failed. No agent model, LLM, external API, GPU or provider credential was used. An earlier probe on port 8899 contacted an unrelated pre-existing HTTP server and is discarded as a port-collision diagnostic.

```text
results/vnext/post_core_letta_0_16_8_block_profile_admission_20260827_v4/preflight.json
  SHA-256 89cc8ce890929a3b3710f50b1238b40865676288fdd3aff1257a896724778fbf
results/vnext/post_core_letta_0_16_8_block_profile_admission_20260827_v4/admission_receipt.json
  SHA-256 fbf9f7cdf2e80549f93f519a360beda3a076aab0ee54fcbd724f678aef935a9e
```

Letta remains `BLOCKED`; its exact blockers are now installed-content/source binding and unavailable or unauthorized PostgreSQL runtime. No runtime lifecycle or accuracy row is claimed.

## Letta 0.16.8 authenticated runtime qualification (2026-08-28)

A new post-Core qualification run closes the two historical v4 blockers without rewriting that earlier evidence. Letta was installed from the pinned official source commit `1131535716e8a31c9a437f8695e25ac98f203a24`; two consecutive no-dependency reinstalls produced the same installed-content SHA-256 `12fd32f25e188754184584c182db6be788403f545c196216956acace93bd0ca9` over 569 authenticated files. The qualification runner and project code were executed from detached, user-owned git clones at project revision `7293f94482aeae89b1966fe24ba741a03308b55a`.

The runtime used a temporary user-owned PostgreSQL 18.6 cluster with pgvector 0.8.6, a dedicated random-password superuser/database, private password-authenticated socket directory, random loopback ports and no shared database state. Alembic migrations ran before Letta startup. Letta 0.16.8 then served native block REST endpoints on loopback; the production JSONL worker authenticated exact package/source/content/configuration identity and completed the 20-trial namespace-reset probe plus add/update/noop/export/retrieve/slot-direct/delete/empty-export lifecycle.

```text
preflight outcome: pass
admission outcome: pass
namespace reset: 20/20 trials passed
lifecycle: PASS
stable add/update/delete entry ID: verified
post-delete export: empty
post-lifecycle marker count in the dedicated database: 0
PostgreSQL: 18.6
pgvector: 0.8.6
LLM/API/GPU calls: 0/0/0
cleanup: PASS
```

Authoritative evidence:

```text
results/vnext/post_core_letta_runtime_qualification_20260828_v1/letta_runtime_preflight.json
  SHA-256 620a30c8c3fe05788e17aae52e7b015eb8cda0db549d5d9e47092b1d8e436e41
results/vnext/post_core_letta_runtime_qualification_20260828_v1/letta_runtime_admission.json
  SHA-256 196b5d8bde95a3d35445aea143e4bae0a702de2fd5a7515fec970bf71e5dffc5
results/vnext/post_core_letta_runtime_qualification_20260828_v1/letta_runtime_qualification.json
  SHA-256 fb3ce1ae6829aeb61353190806360a5aad17c4b4387aeab715248aad49501305
```

This is authenticated external-system runtime capability and lifecycle evidence, not benchmark accuracy or prompted-answer evidence. The earlier v1/v2 remote roots are retained as blocked diagnostics and are not authoritative. The next gate is a bounded supported-scope canary that keeps natural-language extraction, Letta native block state/retrieval and answer-model behavior as separate evidence layers.

## Letta 0.16.8 + Qwen3.5 bounded extraction canary (2026-08-29)

A source-bound 32-task Family-A canary was run on Song-1-Wu GPU0 using target-conditioned Qwen3.5 extraction, an explicit controller policy and an authenticated Letta 0.16.8 native block runtime. The task-view SHA-256 is `ef352d6eb719389bcab39d4746ad97fe7f1b0489f4fa402f15e039e33c5c2ac6`; the selected scope is eight semantic cores and 32 released tasks. Twenty-four single-object tasks are supported and eight multi-object tasks remain explicit `NOT_SUPPORTED`.

The extraction prompt receives visible raw text plus the admitted attribute only. The controller binds the sole target object ID. Qwen returns operation/value JSON; mutation values must be finite scalars, and noop/delete values must be null. The controller's declared empty-store policy reconciled four extracted UPDATE operations to ADD. This reconciliation is preserved in the event traces and prevents interpreting the result as pure Qwen action accuracy. Letta performed native block create/update/export/retrieval/reset through its authenticated JSONL worker; task namespaces were removed after each terminal row. Prompted answering was not run in this gate.

```text
requested tasks: 32
supported single-object tasks: 24
NOT_SUPPORTED multi-object tasks: 8
PASS on supported tasks: 24/24
FAIL on supported tasks: 0
state accuracy on evaluated supported tasks: 1.000
k=16 gold retrieval rate: 1.000
average final memory size: 1.000
stable native entry ID: 24/24
empty-store UPDATE-to-ADD reconciliations: 4 events
provider/API calls: 0/0
prompted-answer metrics: NOT_RUN / null
```

Qwen provenance is bound to `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, authenticated shared-snapshot tree `e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db`, shared binding receipt SHA-256 `924cb994248cf56d1d41df0da3bca06ce7abe1184cde5f0af489d4d364a1d9c2`, and runtime receipt SHA-256 `5d06cb1cbacd43beb0b0a2aaafd1bd7a5b75e8f6d283f5dbbd899b8429ff202f`. Letta qualification was regenerated at project revision `ab434fd24718a84241c8af768de84e32c58ee689` with distinct qualification-runner and worker-source hashes, PostgreSQL 18.6, pgvector 0.8.6 and complete cleanup.

Authoritative artifacts:

```text
results/vnext/post_core_letta_runtime_qualification_20260829_v2/letta_runtime_preflight.json
  SHA-256 10e0f17047d62653bfff0e7a2c9a7a47624964c90963b1dbbe3ad0b905ec156c
results/vnext/post_core_letta_runtime_qualification_20260829_v2/letta_runtime_admission.json
  SHA-256 196b5d8bde95a3d35445aea143e4bae0a702de2fd5a7515fec970bf71e5dffc5
results/vnext/post_core_letta_runtime_qualification_20260829_v2/letta_runtime_qualification.json
  SHA-256 34f6de8a3f3a9634c0f8f785ec1393c5c892009eb52da737fccdfb4eec772086
results/vnext/post_core_letta_qwen_canary_20260829_v1/rows.jsonl
  SHA-256 95107ccaf39a880fb2d953cdee0cc8cf8969223bdcad952df0645d3d2fba35db
results/vnext/post_core_letta_qwen_canary_20260829_v1/canary_receipt.json
  SHA-256 885857451e27682c2e016059ee6380f2074bd0c2f8ec3e1f93cc70597817f164
```

Evidence boundary: this is a supported-scope joint pipeline result for target-conditioned Qwen extraction + controller reconciliation + Letta native block state/retrieval. It is not Letta-only accuracy, Qwen extraction-action accuracy, prompted-answer evidence or a full external-system leaderboard row. The next valid step is either a repeated deterministic canary or the full 80-task Family-A supported-scope matrix under the same frozen layer definitions.

## Letta 0.16.8 + Qwen3.5 full Family-A supported-scope matrix (2026-08-29)

The full authenticated matrix extends the same layer-separated pipeline to all 80 tasks in the frozen Family-A view. It completed 80/80 terminal rows: 52 single-object tasks were supported and 28 multi-object tasks were explicit `NOT_SUPPORTED`. All 52 supported tasks passed state execution and retrieval checks. The run used Qwen3.5 only for visible event extraction, Letta native blocks for state/retrieval, and the same controller-bound target identity and empty-store UPDATE-to-ADD reconciliation policy. Prompted answering remained unrun.

```text
requested tasks: 80
supported single-object tasks: 52
NOT_SUPPORTED multi-object tasks: 28
PASS on supported tasks: 52/52
FAIL on supported tasks: 0
state accuracy: 1.000 (denominator 52)
k=16 gold retrieval rate: 1.000 (denominator 52)
average final memory size: 1.000 (denominator 52)
stable entry ID rate: 1.000 (denominator 52)
requested operations: ADD 43, UPDATE 1193, NOOP 0, DELETE 0
effective operations: ADD 52, UPDATE 1184, NOOP 0, DELETE 0
empty-store UPDATE-to-ADD reconciliations: 9
total provider/API calls: 0/0
prompted-answer metrics: NOT_RUN / null
runtime cleanup: PASS
```

The nine reconciliations are disclosed because the result is not pure extraction-action accuracy: it is a joint target-conditioned Qwen + controller policy + Letta native block pipeline. Unsupported tasks remain outside all performance denominators.

Authoritative artifacts:

```text
results/vnext/post_core_letta_runtime_qualification_20260829_v3/letta_runtime_preflight.json
  SHA-256 6a8bf0bbc804af0a91ba12bb7ed6789d75a23dea3581dcf8ad20a2fbddc9ebdb
results/vnext/post_core_letta_runtime_qualification_20260829_v3/letta_runtime_admission.json
  SHA-256 196b5d8bde95a3d35445aea143e4bae0a702de2fd5a7515fec970bf71e5dffc5
results/vnext/post_core_letta_runtime_qualification_20260829_v3/letta_runtime_qualification.json
  SHA-256 16ef025b05648eb875e2569d61183b922b84e49db8bdfaf82aca30eeecfb9ec2
results/vnext/post_core_letta_qwen_full_family_a_20260829_v2/rows.jsonl
  SHA-256 3aee23cae9148ba14214d4afe10e3a0ff3d6821d9ab89f660f3f80bfcd544c16
results/vnext/post_core_letta_qwen_full_family_a_20260829_v2/full_family_a_receipt.json
  SHA-256 6ef2a66639de6d7da97b3f3345d19fcce86297670b011ffcba89f8f716740007
```

The full matrix is supported-scope external-system evidence, not a broad leaderboard, not prompted-answer evidence, and not an isolated Letta or Qwen accuracy claim. It should be analyzed alongside the 24/24 bounded canary and the earlier LangMem supported-scope result.

## Letta 0.16.8 + Qwen3.5 prompted-answer canary (2026-08-29)

### Motivation and evidence boundary

The earlier Letta Family-A runs established supported-scope state and retrieval evidence but intentionally left prompted-answer metrics null. This gate adds a separate answer layer without reinterpreting the extraction matrix: Qwen3.5 first performs visible-event CRUD extraction, the controller binds the admitted object and discloses empty-store UPDATE-to-ADD reconciliation, Letta performs native block state/export/retrieval, and the same source-bound Qwen snapshot answers only from the actual Letta k=16 retrieval trace. It is joint-pipeline evidence, not Letta-only accuracy, pure Qwen extraction accuracy, native Letta answering or a broad leaderboard result.

The final source revision is `1e21674c0e4d648a8b1b8b7e5aaed2c87a341bfb`. Tang-3-Wu GPU0 was explicitly authorized and recorded as physical UUID `GPU-b4046e43-32f6-8e18-1a7d-4cfd3461a717`; logical `cuda:0` refers to that isolated visible device. PostgreSQL 18.6 + pgvector 0.8.6 was reconstructed from the previously authenticated local conda package cache with `--offline`, then Letta qualification passed on Tang-3 with loopback-only services and complete cleanup.

### Runtime correction and diagnostics

Two pre-result diagnostics are excluded from accuracy evidence:

1. The first Tang-3 launcher stopped before Qwen load because it omitted four environment bindings used by the passing qualification (`PYTHONPATH`, `PYTHONIOENCODING`, `HF_HUB_OFFLINE` and `LETTA_NATIVE_API_BASE_URL`). No output root or task rows were created.
2. Revision `930e038` loaded Qwen but produced 24 supported `FAIL` rows before first extraction because PyTorch 2.5.1 has no strict deterministic CUDA implementation for the Qwen3.5 `cumsum` kernel. This root is diagnostic only. Revision `1e21674` retains seed 0, greedy decoding, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CUDA/CuDNN deterministic settings and switches PyTorch to the disclosed `warn_only=True` mode. Two complete repetitions were then required and compared empirically.

The missing strict CUDA kernel remains a runtime caveat. It did not change any semantic output across the two complete repetitions.

### Commands and execution

Representative commands were:

```bash
# Source-bound Letta qualification on Tang-3.
python scripts/vnext_run_letta_runtime_qualification.py \
  --python-executable /NAS/yesh/MemUpdateBench/external/conda_py313_v1/bin/python3.13 \
  --postgres-bin /data/wujcan/mub_pg_env_tang3_prompted_v1/bin \
  --letta-source /data/wujcan/letta_source_prompted_1131535716 \
  --project-root /data/wujcan/mub_letta_project_1e21674 \
  --project-revision 1e21674c0e4d648a8b1b8b7e5aaed2c87a341bfb \
  --expected-postgres-version 18.6 \
  --expected-pgvector-version 0.8.6

# Each no-replace canary repetition used a fresh private PostgreSQL/Letta runtime.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/vnext_run_letta_qwen_prompted_answer.py \
  --scope canary32 \
  --tasks <authenticated Family-A task view> \
  --qualification-root <source-bound Tang-3 qualification> \
  --letta-base-url http://127.0.0.1:58047 \
  --model-snapshot /NAS/HuggingFaceModels/Qwen3.5-9B \
  --model-snapshot-binding <authenticated binding receipt> \
  --model-runtime-receipt <authenticated runtime receipt> \
  --expected-letta-project-revision 1e21674c0e4d648a8b1b8b7e5aaed2c87a341bfb
```

### Results

Both complete repetitions produced the same bounded result:

```text
requested tasks: 32
supported single-object tasks: 24
NOT_SUPPORTED multi-object tasks: 8
PASS on supported tasks: 24/24
FAIL on supported tasks: 0
state accuracy: 1.000 (denominator 24)
k=16 gold retrieval rate: 1.000 (denominator 24)
average final memory size: 1.000
prompted-answer EM: 1.000 (denominator 24)
prompted-answer token F1: 1.000 (denominator 24)
answer outcomes: CORRECT 24, WRONG 0, FORMAT_INVALID 0, UNAVAILABLE 0
requested operations: ADD 20, UPDATE 600
Letta effective operations: ADD 24, UPDATE 596
empty-store UPDATE-to-ADD reconciliations: 4
provider/API calls: 0/0
retries: 0
runtime cleanup: PASS
```

The repeated-run semantic projection SHA-256 is `229c70edc22b5fef5dc87f12e71e589b9ae08450ec484af41d59103f644684c9`. All 32 ordered terminal rows matched across repetitions for extraction output hashes, requested/effective operations, affected entry identities, final state, retrieval-trace hashes, visible-prompt hashes, answer raw-output hashes, prompted answers and scored outcomes. Latency fields were excluded from the determinism projection.

### Authoritative artifacts

```text
results/vnext/post_core_letta_runtime_prompted_tang3_1e21674_v1/letta_runtime_qualification.json
  SHA-256 df6aeac1be6397c41be1a4965fba33c5141a9d93c8e73c4456b1669fc8992d0a
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_v1/rows.jsonl
  SHA-256 b526d7dee333c4a00bdf60b6da3f989ccc6daa03e67c9e892ce552015e5de638
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_v1/canary_receipt.json
  SHA-256 6aa56558377c59b72e06e037e22fc348e6438e781c2f33325934ae236c729ff8
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_v1/execution_environment_receipt.json
  SHA-256 322964c13f311d7631115fded1595d5563ca5c66f60384899ab9a4d230dd7aac
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_repeat_v1/rows.jsonl
  SHA-256 2fb899a98dfbc2e368c7296e1deaffc9de3408a1464db66aa1e25f4d33a53e00
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_repeat_v1/execution_environment_receipt.json
  SHA-256 028b85342adb7be3af323bacbf35872653fc7e22732e1ec43fd0813aaa2cde02
results/vnext/post_core_letta_qwen_prompted_canary_tang3_1e21674_determinism_v1/determinism_receipt.json
  SHA-256 b32c61637d6832154468ac6a95ca5d236f1378952f33037a19b62882aeece769
```

Each successful run also contains a canonical artifact index and integrity audit. Unsupported tasks remain outside every quality denominator. The valid conclusion is narrow: within Letta's admitted single-record Family-A surface, the source-bound Qwen extraction + controller + Letta retrieval + Qwen prompted-answer pipeline was fully correct and empirically stable across two repetitions. The separate 80-task full-family prompted-answer matrix remains a downstream gate rather than part of this canary claim.

## Letta 0.16.8 + Qwen3.5 full Family-A prompted-answer matrix (2026-08-29)

After the repeated canary closed the prompted-answer and empirical determinism gates, the same source-bound pipeline was extended without changing the model, prompt renderer, parser, retrieval policy, controller reconciliation policy or Letta profile. The full run used revision `1e21674c0e4d648a8b1b8b7e5aaed2c87a341bfb`, the same Tang-3 qualification SHA-256 `df6aeac1be6397c41be1a4965fba33c5141a9d93c8e73c4456b1669fc8992d0a`, physical Tang-3 GPU0 and a fresh private PostgreSQL/Letta runtime.

```text
requested tasks: 80
supported single-object tasks: 52
NOT_SUPPORTED multi-object tasks: 28
PASS on supported tasks: 52/52
FAIL on supported tasks: 0
state accuracy: 1.000 (denominator 52)
k=16 gold retrieval rate: 1.000 (denominator 52)
average final memory size: 1.000
prompted-answer EM: 1.000 (denominator 52)
prompted-answer token F1: 1.000 (denominator 52)
answer outcomes: CORRECT 52, WRONG 0, FORMAT_INVALID 0, UNAVAILABLE 0
requested operations: ADD 43, UPDATE 1193
Letta effective operations: ADD 52, UPDATE 1184
empty-store UPDATE-to-ADD reconciliations: 9
provider/API calls: 0/0
retries: 0
runtime cleanup: PASS
```

The integrity audit verified exactly 80 ordered unique terminal rows, canonical JSON, source/model/qualification/worker bindings, secret-free artifacts, status-specific metric/null policies and complete runtime cleanup. All 32 tasks from the authoritative prompted-answer canary are present in the full matrix; their latency-excluded semantic projections are identical 32/32 for extraction outputs, operations, state, retrieval trace, visible prompt, answer output and score. This provides a direct consistency bridge from the repeated canary to the expanded 80-task surface.

Authoritative artifacts:

```text
results/vnext/post_core_letta_qwen_prompted_full_family_a_tang3_1e21674_v1/rows.jsonl
  SHA-256 2589a216a38ee97b91fee7a0e9e0a318be43d192e7a78e5600f27a6e8c35c44a
results/vnext/post_core_letta_qwen_prompted_full_family_a_tang3_1e21674_v1/full_family_a_receipt.json
  SHA-256 cda35ec576ab493ce2ac1ab016e1a8a9c64766c7dd4794339c0857a3de750fbb
results/vnext/post_core_letta_qwen_prompted_full_family_a_tang3_1e21674_v1/artifact_index.json
  SHA-256 2d6e20174c2cb75cee7f189f06dedffd128b8a80eff36bea0412fe0f10b9f3a2
results/vnext/post_core_letta_qwen_prompted_full_family_a_tang3_1e21674_v1/execution_environment_receipt.json
  SHA-256 e6c51ebbe568b4d768685917e7c3203bdf86fabb142bd9b463d65c7e95dda1b0
results/vnext/post_core_letta_qwen_prompted_full_family_a_tang3_1e21674_v1/integrity_audit.json
  SHA-256 2f3c517ff5707610aa93e7e6470498a30c91035ed810470dbb7f7d609423a835
```

Evidence boundary: this is complete Family-A supported-scope joint-pipeline evidence for Qwen visible-event extraction + controller reconciliation + Letta native block state/retrieval + Qwen retrieved-context answering. It is still not Letta-only accuracy, pure Qwen accuracy, native Letta answering or a broad cross-system leaderboard claim. The 28 multi-object tasks remain explicit unsupported work rather than zero-valued failures.

## Main-track v1 multilingual B/C/D data candidate (2026-08-29)

### Motivation and boundary

The current Family-A external rows are a useful authenticated starting point but cover only 20 semantic cores and leave all multi-object tasks outside the supported profile envelope. To move toward a main-track benchmark, an add-only sibling data release was generated without modifying `data/vnext/core/v3`, Core manifests, Pilot files or Phase-0 metadata. This candidate expands independent semantic cores, controlled domains, attributes, task families and language/surface variation. It is a dataset candidate, not a human-audited or `FINAL_APPROVED` benchmark release.

### Candidate design

```text
release_id: main_track_v1
families: B interleaved multi-slot, C entity/attribute grounding, D NOOP/write discipline
semantic cores: 900 (300 per family)
renders: 4 per core
strict-v3 tasks: 3,600
split: group-first 630 train / 90 dev / 180 test cores
split tasks: 2,520 train / 360 dev / 720 test
surfaces: en-US explicit canonical, en-US concise natural, es-ES concise natural, ja-JP concise natural
domains: 12
attributes: 12
difficulty per family: 150 easy / 90 medium / 60 hard
review status: NOT_STARTED
```

Family B covers active object counts 2/4/8/12 and round-robin, burst and adversarial-adjacent interleavings. Family C covers exact/paraphrase/near-name attribute references, distinct/alias/same-name/namespace-collision entities and typed unique/ambiguous/no-match outcomes. Family D covers transient, hypothetical, negated, uncertain, semantic-near-miss, duplicate-current and unsupported-inference traps at NOOP densities 25/50/75 percent. The four surface variants preserve one semantic task hash per core while varying rendered/source/task identities and text hashes.

### Automated validation and audit selection

All 3,600 tasks pass strict-v3 construction, deterministic replay and normative evidence evaluation. The candidate validator checks exact counts, group-first split assignment, four-surface semantic equivalence, cross-split identity leakage, canonical bytes, manifest/per-task hashes and frozen-root exclusion. A separate deterministic audit-selection artifact selects 60 cores and all 240 corresponding surface tasks, covering all three families, all 12 domains and attributes, all four surfaces, all splits, all difficulty levels and every declared B/C/D diagnostic axis. Human review remains pending; no model or external-system evaluation has been run on this release.

The first candidate at `data/vnext/main_track_v1/` is superseded and must not be used: an independent semantic-hash audit found 900 cores but only 711 unique rendered semantic hashes. The renderer was corrected to bind deterministic expansion identity into actual object/value payloads, and the corrected no-replace candidate is now `data/vnext/main_track_v1_independence_v1/`. The corrected candidate has 900/900 unique semantic hashes, with four surface variants equivalent within each core; its audit selection is `results/vnext/main_track_v1_independence_audit_selection/selection.json`. The corrected candidate and selector were committed and pushed in `a55cffc`. It remains `review_status=NOT_STARTED`; only a passed stratified human audit and post-audit release validation can authorize downstream baseline/model runs.

### Deterministic held-out oracle gate (2026-08-30)

The corrected candidate's 720 held-out test tasks were reopened and run through the v3 `ReferenceAdapterV3` with `slot_direct`. All 720 tasks completed with zero replay/evidence/query coverage/typed-answer issues, including Family C typed abstentions and Family D NOOP nonmutation cases. This is dataset/reference-oracle diagnostic evidence only: it uses the reference adapter and no model, external system, GPU, network, or provider calls. It does not establish benchmark accuracy or human-audit approval.

```text
test tasks: 720
reference-oracle pass: 720/720
failures: 0
families: B 240, C 240, D 240
languages: en 360, es 180, ja 180
review status: NOT_STARTED
```

Artifact:

```text
results/vnext/main_track_v1_oracle_diagnostic_v1/oracle_diagnostic.json
  SHA-256 dbee00842013342f527ebba5a3342b26bef43453793982172ee2fbb3362b18d3
```

### Human-audit packet preparation (2026-08-30)

A separate no-replace packet was materialized from the corrected candidate and authenticated 60-core selection. It contains exactly 240 selected surface-task records in selection order, including visible event text, normalized text, roles/metadata, declared actions/effects, target objects, query selectors, gold evidence, version history and blank human decision fields. It contains no model outputs and does not represent human decisions.

```text
packet tasks: 240
selected semantic cores: 60
artifacts: 3 (audit_packet.jsonl, audit_manifest.json, review_instructions.md)
review status: NOT_STARTED
policy: post-core-data-audit-v1
audit_packet.jsonl SHA-256: ecdb7269db41309e2c8694dba9ecbb12a5a4d28fd4f11cbc2d99b77ce719cd65
audit_manifest.json SHA-256: 6f9c0431b8a0ce639f428a7c346f3a95e4225a8baafe4a8d8faf14d00339fcdb
review_instructions.md SHA-256: 323a2121a28ce56cc30e4f39fc9876939b355967ce7f1f2e026e1c92eebb4bcc
```

The packet is bound to candidate `data/vnext/main_track_v1_independence_v1` and selection hash `b6e9056792d8be212873ad1559487b2d6c66d833e1f9fb1cb6a1d2d2fc11e935`. Human review remains the release blocker; any requested semantic correction must regenerate a new candidate and packet rather than patching either artifact in place.

### First human audit outcome and regenerated packet (2026-08-30)

The completed audit package `main_track_v1_independence_audit_completed_person1_person2` was independently checked against its bound candidate. It contains 240 rows: 214 `pass`, 26 `needs_revision`, 0 `block`, and 26 unresolved findings. The review therefore correctly reports `review_status=FAIL` and `benchmark_release_eligible=false`.

Findings were concentrated in Family C: six alias cases used distinct `primary_entity` and `alternate_entity` canonical keys without establishing an alias relation, and twenty non-English C rows retained an English query sentence. The renderer was corrected so alias cases use one canonical target and explicitly state the alias relation, while Spanish and Japanese C queries are localized. The corrected no-replace candidate is `data/vnext/main_track_v1_audit_fix_v1/`; it again has 900 cores/3600 tasks, 900/900 unique semantic hashes, and passes the full automatic validation gate. A fresh selection and audit packet are bound to this corrected candidate:

```text
selection: results/vnext/main_track_v1_audit_fix_selection/selection.json
selection SHA-256: dbee5a66234bdfd5bd8cf2111b9c7de51dbccb04c08ffec3ad92b96307c332ef
packet: results/vnext/main_track_v1_audit_fix_packet_v1/
audit_packet.jsonl SHA-256: 828cb18dbc7ae4718fcd62eb0c87044cb99049cabf3682cf018d10bef362a9f9
audit_manifest.json SHA-256: c1c3fef54a5d46a7d72425c979e88d31f21132daef89878f847b408f8ad765d6
review_instructions.md SHA-256: 323a2121a28ce56cc30e4f39fc9876939b355967ce7f1f2e026e1c92eebb4bcc
review status: NOT_STARTED
```

The old failed audit remains historical evidence and is not rebound to the corrected candidate. It blocked downstream model and external-system evaluation until the corrected packet received a complete human `PASS`; that gate is now closed by the completion recorded below.

### Corrected candidate human-audit completion (2026-08-30)

The completed audit directory `D:/USTC/2026Winter/MemUpdateBench/results/vnext/main_track_v1_audit_completed_person1_person2` now binds the corrected `main_track_v1_audit_fix_v1` candidate and the fresh audit packet. The completion validator checked the 240-row packet, decision fields, source packet hash, selection hash, all candidate artifact hashes and the automatic candidate validation report. It reports a complete PASS and eligibility for final release promotion.

```text
review status: PASS
rows: 240
pass: 240
needs_revision: 0
block: 0
unresolved: 0
benchmark release eligible: true
reviewers: Person 1, Person 2
completed packet SHA-256: 4acc399c051b973cf283ca7d8055e1b8bccc2a7d8b86fb0794ce2a1394cc4987
review completion attestation SHA-256: 0f1d13f23e85f74c345642a63ca23efe89b1e8ebc47c6930f2fb79ad3437743c
```

The attestation records that the review metadata JSON was formatted rather than compact canonical (`review_metadata_canonical=false`); packet row bytes, candidate artifacts, selection bindings and semantic review decisions remain hash-checked. The original candidate and first failed audit remain immutable historical artifacts. This closes the data human-audit gate, but it does not itself establish model or external-system performance.

### Main-track fixed-reference Qwen answer-layer baseline (2026-08-30)

The first layered benchmark on the audited main-track candidate fixes memory state and retrieval to `ReferenceAdapterV3` and varies only the prompted answer layer. This deliberately isolates answer-model behavior from extraction and external-manager behavior. Qwen3.5-9B receives only the rendered retrieval prompt; no gold actions, hidden metadata, or raw model output is published. The full held-out test scope contains 720 tasks across Families B/C/D and en/es/ja.

```text
requested tasks: 720
terminal PASS rows: 720
answerable queries: 568
expected abstentions: 152
exact/normalized/typed answer match: 0.7708333 / 0.7708333 / 0.7708333
token F1: 0.7708333
answer outcomes: CORRECT 555, WRONG 6, FORMAT_INVALID 7, UNAVAILABLE 0, WRONG_ABSTENTION 152
family EM: B 0.9958, C 0.3208, D 0.9958
language EM: en 0.7806, es 0.7722, ja 0.7500
provider/API/network calls: 0/0/0
```

The 152 wrong abstentions are concentrated in Family C: Qwen frequently answers when the gold disposition is `ABSTAINED`, rather than failing to retrieve the memory. Family B and D are near-ceiling, with one wrong answer in each; seven format errors occur in the Japanese subset. This is a reference-state answer-layer result, not external-memory manager evidence and not an extraction result.

Authoritative artifacts:

```text
results/vnext/main_track_qwen35_answer_test720_tang1_ee793b1_v1/rows.jsonl
  SHA-256 b84eb34b1c5401716ffd26efa6d63b2c74dbab4a1d9e70ada3aa540c4295cdd0
results/vnext/main_track_qwen35_answer_test720_tang1_ee793b1_v1/summary.json
  SHA-256 38db939504405ee152ee64b45dc6c32b3c325f7b3e2f62dcbab4e2d683fd92bd
results/vnext/main_track_qwen35_answer_test720_tang1_ee793b1_v1/artifact_index.json
  SHA-256 53ecc96d355e1510aa9314258e6cc6a97cede3f205ddfa8eb9b83021ec4e9794
```

Qwen model identity is `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with authenticated tree `e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db`; the run is bound to human-audit attestation SHA-256 `0f1d13f23e85f74c345642a63ca23efe89b1e8ebc47c6930f2fb79ad3437743c`. Tang-1 GPU1 and the model process cleaned up successfully. The next layered gate is to repeat this answer-only panel with an independently qualified answer model, then evaluate fixed extractor/manager combinations separately.

### Main-track Qwen/Muse fixed-reference answer-model panel (2026-08-31)

Muse Glimmer 30B GGUF was added as the second independently frozen open answer model while keeping `ReferenceAdapterV3` state/retrieval, the audited 720-task test set, retrieval k=16 and typed scoring fixed. The old 32-token Muse capability smoke remains `BLOCKED`; main-track qualification separately established that long reasoning traces require a larger output budget. The final full run used the authenticated Dynamic Q4_K_XL GGUF, llama.cpp commit `c1d0e7a004015f23bc0233470b747b596f29b264`, loopback HTTP, speculative decoding off, reasoning mode off, seed 0, temperature 0, request max 2048, server context 8192 and server n-predict 4096.

```text
Qwen3.5-9B: EM/F1 0.7708333, CORRECT 555, WRONG 6, FORMAT_INVALID 7, WRONG_ABSTENTION 152
Muse GGUF:   EM/F1 0.8625000, CORRECT 491, CORRECT_ABSTENTION 130, FORMAT_INVALID 8, UNAVAILABLE 69, WRONG_ABSTENTION 22
Family B EM: Qwen 0.9958, Muse 1.0000
Family C EM: Qwen 0.3208, Muse 0.5875
Family D EM: Qwen 0.9958, Muse 1.0000
Language EM (Qwen/Muse): en 0.7806/0.8528, es 0.7722/0.8611, ja 0.7500/0.8833
paired agreement/disagreement: 504 / 216
paired disagreement categories: abstention 137, answer 2, format-or-unavailable 77
```

Muse's main gain is calibrated abstention: 130/152 expected-abstention tasks are correct versus 0/152 for Qwen. Muse also over-abstains on 69 expected-answer tasks, while Qwen mainly over-answers ambiguous/no-match tasks. Both models are near-ceiling on Families B and D, so the panel isolates Family C reference resolution and abstention as the dominant answer-layer discriminator. Eight Muse format-invalid rows reached the 2048 request cap; these remain explicit format/truncation outcomes rather than being silently retried or excluded.

Both 720-task roots pass the strict post-publication answer-artifact validator: ordered task identity, candidate and human-audit bindings, task hashes, row score recomputation, summary aggregates, redaction and model/runtime provenance were independently checked. Muse server/process/GPU cleanup passed, returning GPU6 to 5 MiB.

```text
Muse rows: results/vnext/main_track_muse_answer_test720_tang1_133adce_v1/rows.jsonl
  SHA-256 f3d966bf694b8f5238bf15ebccda834d8a5cac4dfa739aae5f1a54f62f4ded09
Muse summary: results/vnext/main_track_muse_answer_test720_tang1_133adce_v1/summary.json
  SHA-256 7ae37c93689812d62d320d8b533e996ec913755bac7e02130a267ab6d0c3007d
Muse artifact audit: results/vnext/main_track_muse_answer_test720_audit_v1/answer_artifact_audit.json
  SHA-256 e9a2de5068fcb7a78683bdb17aec54facbc0bf7e28e2ef6187a0939410c03dba
Panel summary: results/vnext/main_track_qwen_muse_answer_panel_v1/panel_summary.json
  SHA-256 873001f89888a3a2c117b43c4c9ca57529e0a5336d70c33e74dae8b03cbee1b7
Panel index: results/vnext/main_track_qwen_muse_answer_panel_v1/panel_index.json
  SHA-256 ae937ea2a1e70a4f54ec88ea852f1c94eba3feaf652a0281c5e681c7eb158a06
```

Evidence boundary: this is a paired fixed-reference answer-layer comparison. It is not external-manager evidence, extraction evidence, a significance test or a broad leaderboard. The next main-track layer is fixed-extractor/manager factorial evaluation on the same audited task universe.

### Main-track fixed-extractor/manager factorial: LangGraph custom adapter + Qwen (2026-09-02)

#### Motivation

The main-track expansion requires a fixed visible-event extractor, an external memory manager, and a separately replayable prompted-answer layer on one audited task universe. This stage keeps those layers separate: Qwen3.5-9B converts visible events to the explicit CRUD/NOOP surface; a LangGraph `InMemoryStore` plus the custom MemUpdateBench adapter performs mutation and retrieval; a later answer runner consumes only the frozen retrieval fixture. The runtime is not native LangMem CRUD/retrieval evidence.

#### Contract and qualification

The corrected `main_track_v1` candidate remains immutable and contains 900 semantic cores / 3,600 tasks, with 720 TEST tasks. The v2 factorial manifest binds six cells: Reference×Qwen/Muse (720 supported each), Letta profile×Qwen/Muse (240 supported + 480 typed unsupported each), and LangGraph custom adapter×Qwen/Muse (240 supported + 480 typed unsupported each). The LangGraph identity is explicitly `langgraph_store_custom_adapter`, system/backend `langgraph_in_memory_store`, with the implementation boundary `LangGraph InMemoryStore + custom MemUpdateBench adapter; not native LangMem CRUD/retrieval evidence`.

The local-only qualification loaded `langmem==0.0.30` and `langgraph==1.2.11`, exercised reset/add/update/NOOP/retrieve/export/reset in memory, and recorded zero provider/network/database/remote calls. The receipt is a hash-bound release attestation, not a cryptographic signature or independent signer assertion. The tested clean runtime used Torch 2.5.1+cu121, Transformers 5.9.0, Accelerate 1.12.0, and the shared Qwen3.5-9B snapshot.

#### Engineering gates

- Factorial planner and manager-fixture contract tests: 49 manager + 20 planner tests passed; support partitions remain 240/480 for both external profiles.
- Qwen fenced-output hardening: exact pure JSON and exact JSON fences are accepted; commentary, malformed fences, and invalid schemas are rejected. The existing extraction suite passed 48 tests with one expected Windows symlink skip.
- Manager canary8 on Song-2 GPU3 passed: 488 ordered rows (8 supported + 480 unsupported), state accuracy 1.0, gold retrieval 1.0, zero failures, one model load and 24 GPU generations. Rows SHA-256 `de55a0361a454615ca42a781fbc5af77559b7d0eaa533075fb79e6cdeddd57ba`.
- Authoritative full-240 manager fixture on Tang-1 GPU1 passed: 720 ordered rows (240 supported + 480 unsupported), state accuracy 1.0, gold retrieval 1.0, zero failures, one model load and 720 GPU generations. Rows SHA-256 `a768d58764a1082ff015a29e1e9b587ea0fa667f8a971a4e76f4fbc8bc97a31f`; summary SHA-256 `0d7eacaa15fc456291779a367bb4af66616f6c42450da9877e3ec8c0adc9ed60`; index SHA-256 `9cac739bd4ce164987c1d57fa02a6c9cd0ac9de2bc5d46f196e987f83fe9808a`.
- The source-pinned manager fixture attestation passed with receipt SHA-256 `0bd4bf9753db2698a8583f69f5e41fbd0401d45afd7fc355dee9c8e685809115`. It binds the manifest SHA-256 `6cd81f3c5eae9d4ea950c562d54ecc5e4f390ab4994b8ccb09286e45a272ee0b`, candidate/audit hashes, fixture root digest, manager source identity, production runtime accounting, and Tang-1 producer identity.
- Authoritative Qwen answer replay passed on the same frozen fixture: 720 answer rows, 240 attempted answers, 480 typed unsupported rows, zero answer execution failures, inherited state/retrieval 1.0/1.0, answer EM/F1 `0.9916667`, outcomes CORRECT 238 and WRONG 2. Answer rows SHA-256 `ff56a6502216eef39eae77426f468e6657c4789d5bd6d6f98a8ea1385f982b72`; answer summary SHA-256 `678c02d15e33bfcb36503c4b02b38c06a171dc4dcef55c7ad2be3da91ee82193`; answer index SHA-256 `a94e82782e8f202937366e313e05a636a7dae92770b8d23181d0b10ee093846c`.

#### Error analysis and conclusions

The first Song-1 full run was retained as an invalidated diagnostic: 20 supported tasks failed because Qwen emitted exact fenced JSON for some NOOP events while the extractor expected only raw JSON. No rows from that run are interpreted scientifically. Strict fence parsing fixed the root cause without adding retries. A later Song-1 rerun was stopped when another user's GPU jobs occupied the requested cards; the task was moved to Tang-1. Tang-1 completed the source-pinned fixture and Qwen replay with clean state/retrieval and answer-layer results. All model, LangGraph, and manager processes released their GPUs after completion.

These results are joint-pipeline evidence for one qualified LangGraph custom-adapter profile plus Qwen extraction and Qwen prompted answering. They do not establish native LangMem performance, a Letta factorial result, Muse production answer accuracy, statistical significance, or a broad external-memory leaderboard. The answer panel should not be interpreted as isolating manager effects from answer-model effects without the planned paired replay.

#### Authoritative local copies and next steps

Downloaded local copies are under:

```text
results/vnext/main_track_factorial_langgraph_qwen_full240_tang1_gpu1_v2/
results/vnext/main_track_langgraph_manager_attestation_tang1_gpu1_full240_v3/
results/vnext/main_track_factorial_langgraph_qwen_answer_full240_tang1_gpu1_v1/
```

The next steps are to audit/publish the downloaded roots as a separate evidence release, run the same frozen retrieval fixture through a second answer model only after its runtime identity is independently qualified, qualify the new Letta production manager seam against a fresh clean runtime, and then extend the matrix to independent Family H realistic-source revisions. Preserve Core/Pilot and all prior factorial v1/v2 diagnostic roots without overwrite.

### Letta factorial manager gate (preparation complete; runtime BLOCKED)

A production `LettaBlockProfileManager` seam is now implemented for the factorial runner. It reuses the v3 Letta adapter, `JsonlSubprocessBridge`, exact visible Add/Update/NOOP/Delete action translation, per-task namespace reset, export/retrieval provenance, and reset-before-close cleanup. The CLI now requires explicit Letta qualification root, loopback base URL, Python executable, clean project root, and optional expected project revision; manager kind dispatch fails closed rather than routing Letta requests through the LangGraph adapter. Worker subprocess activity is represented in the execution accounting instead of silently claiming an all-zero production boundary.

Local verification passed: 53 factorial manager tests, 20 planner tests, 22 attestor tests, 61 answer tests, and the relevant Letta adapter/protocol tests; py_compile and diff-check passed. No Letta server, database, model, or GPU was started for this preparation step.

The runtime gate remains explicitly `BLOCKED`. The historical Letta v3 closure binds project revision `28aa4f8e13ad4c3c99a37939310d4aa05761b6b5`, project tree `51a80a2c06e2dcac048a730e242a2b918ad0118f8d931c0c2cf6e070ff1c4518`, and worker source hash `ad554e10ac03f7385f34679ea190f7220cbcb0b62815b3507cb313522aa576fa`; the visible Tang-1 clone is at a different revision, so it cannot be used as a qualified worker project. The historical closure also reports a measured server port that does not match the embedded preflight server identity. A fresh qualification must provision PostgreSQL 18.6 + pgvector 0.8.6, a dedicated loopback Letta server, migrations, private runtime directories, exact clean source clone, and mutually consistent endpoint/worker/source receipts before any factorial canary.

Do not interpret existing Family-A Letta joint-pipeline results as this factorial manager evidence. Once the gate is closed, run an 8-task Family-D canary first; only a clean canary permits the matched full-240 Letta manager-only comparison against the completed LangGraph fixture.
