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
- P7.0 adds the true retrieval-time stale filter requested by the second-round review. `--retrieval_policy latest_per_slot` leaves raw_add writes unchanged but deduplicates the answer context by `(entity, attribute)` and keeps only the latest entry per slot. On raw_add k=16 dev, EM/F1 improves from the P6.8 normal top-k5 baseline 0.140/0.173 to 0.690/0.703, while memory size remains 52.00 and stale same-slot burden remains 14.25. This is the strongest causal evidence so far that stale same-slot context contamination drives much of raw append's slot-prompt collapse. See `results/p70_stale_filter_intervention_summary/stale_filter_summary.{json,md}` and `paper/p70_stale_filter_intervention_note.md`.
- Raw append top-k10 increases gold retrieval relative to top-k5 but decreases EM, showing that simply retrieving more context does not solve stale competition.
- Heuristic CRUD threshold sweep gives a same-method-family tradeoff curve: state accuracy remains 1.00 across thresholds, while stale burden and memory size increase with threshold and slot-prompt EM is best at the lowest tested threshold.
- Mem0 external baseline feasibility reached isolated package installation and import under `external/mem0_vendor`. P69 discovered a usable local OpenAI-compatible vLLM endpoint on Tang-2 (`Qwen2.5-VL` at port 8011), a cached local MiniLM embedder, and a local Qdrant path, so Mem0 can now run end-to-end without external API keys. The resulting off-the-shelf Mem0 dev20 row remains badly misaligned with exact repeated-slot tracking: improved value extraction reaches EM 0.00 / F1 0.05, memory is inspectable, and retrieved values are mostly stale or wrong. P7.0 fairness audit checked local vLLM ports 8000/8001/8002/8010/8011/8012/8020 and found only Qwen2.5-VL at 8011, so the current Mem0 result must be treated as a qualitative runnable probe, not a fair main-table external baseline. Original Memory-R1 is also not currently available under `/NAS/yesh`; the repository only contains a project-local `baselines/memory_r1_agent.py` approximation, which must not be reported as original Memory-R1. See `paper/p70_external_baseline_fairness_note.md`.

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
