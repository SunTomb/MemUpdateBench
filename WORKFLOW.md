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
