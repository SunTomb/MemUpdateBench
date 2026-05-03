# MemUpdateBench: Diagnosing Repeated Same-Slot Updates in External Memory

## Abstract

External memory systems are increasingly used to maintain long-lived user and world state, but their behavior under repeated updates to the same fact is poorly characterized. We introduce MemUpdateBench, a controlled diagnostic benchmark for repeated same-slot updates, where the same `(entity, attribute)` memory slot is updated across increasing frequencies while same-name distractors and semantic near-miss NOOP events test grounding and parsing robustness. The benchmark separates four quantities that are often conflated: final-state reliability, stale same-slot burden, memory compactness, and slot-conditioned answer robustness.

On the P6.3 hard update-frequency suite, append-only and heuristic memory managers preserve final-state recoverability under oracle slot-state evaluation, reaching state accuracy 1.00 even at k=16. However, they retain stale same-slot entries and collapse under slot-conditioned answering: raw append reaches only 0.07 / 0.10 EM/F1 at k=16 with 14.25 stale same-slot entries. The original P6.3 Long25 checkpoint improves over append-only answering but remains imperfect, reaching 0.48 / 0.53 EM/F1 with state accuracy 0.91. A later three-seed Long25 retraining family is substantially stronger and more compact on the same k=16 test split, so the paper reports these checkpoint families with explicit provenance rather than treating them as the same run. These results show that repeated same-slot updates induce a tradeoff curve rather than a single-method win.

## 1. Introduction

External memory gives language-model agents a way to carry user preferences, facts, and task state across long interactions. Most evaluation settings ask whether a system can retrieve a relevant fact after it has been written. In realistic use, however, memory is not only accumulated; it is repeatedly revised. A user may move cities multiple times, change a preferred name, update a project deadline, or correct an earlier statement. These cases stress the same memory slot repeatedly, and the central question becomes not simply whether the final value is present, but whether obsolete values remain and interfere with later answers.

MemUpdateBench focuses on this repeated same-slot update regime. The benchmark isolates updates of the form `(entity, attribute) -> value`, with actions represented as `ADD`, `UPDATE`, and `NOOP`. This exact slot structure lets us diagnose failure modes that broad memory-agent benchmarks often merge together: stale value retention, missed final updates, wrong entity grounding, wrong attribute parsing, and answer-layer failures. We treat update frequency `k` as the main independent variable because it directly controls how many stale same-slot values a non-compacting memory system can accumulate.

This diagnostic setup reveals an important gap between oracle-like memory-state evaluation and answer robustness. Under exact slot lookup, append-only memory can still expose the final value even when many obsolete values remain in the memory bank. Under slot-conditioned answering, the answer layer must condition on the memory contents as a whole, and stale same-slot entries can dominate or confuse generation. The result is a tradeoff among final-state reliability, stale burden, memory compactness, and answer robustness.

Our contribution is a controlled benchmark and analysis package for this update regime. We provide update-frequency splits, deterministic slot-based diagnostics, paper-facing summary assets, cluster-backed verification of the main k=16 anchors, and a reproducible workflow for analyzing how memory managers move along the reliability--compactness--robustness curve.

## 2. Benchmark

MemUpdateBench represents memory updates with three actions:

```text
ADD <entity>.<attribute> = <value>
UPDATE <entity>.<attribute> = <value>
NOOP
```

The central invariant is exact slot resolution by `(entity, attribute)`. Each example defines a target slot, a final gold value, distractor entities or attributes, and events that should either update the slot, add other facts, or be ignored as NOOPs. This structure lets the benchmark distinguish state-management failures from answer-generation failures.

The main P6.3 hard split varies update frequency with `k in {1, 2, 4, 8, 16}`. Larger `k` means more repeated updates to the same target slot. This controlled axis stresses whether a memory system overwrites stale values, appends new evidence without compacting old evidence, or loses track of the final value. The hard split also includes same-name multi-entity distractors and semantic near-miss NOOP events so that success requires resolving the intended entity and attribute rather than reacting to surface lexical overlap.

The benchmark is intentionally narrow. It does not attempt to cover every memory-agent behavior; instead, it makes one failure pressure measurable. Because each example has an exact target slot and final gold value, we can compute both whether the final state is recoverable and how many obsolete same-slot values remain in memory.

## 3. Methods and Evaluation

We report two complementary answer modes. `slot_direct` is oracle-like slot-state evaluation: it checks whether the final value can be resolved for the target `(entity, attribute)` slot. `slot_prompt` is slot-conditioned answering: it asks the model to answer under a slot-conditioned prompt using the memory contents. The gap between these modes is central to the paper.

We report four metrics together:

1. state accuracy under `slot_direct`, measuring final-state reliability;
2. EM/F1 under `slot_prompt`, measuring answer robustness;
3. stale same-slot entry count, measuring obsolete same-slot burden;
4. final memory size, measuring compactness.

The main comparison includes Constrained CRUD, Raw append, Heuristic CRUD, and Long25. Constrained CRUD is an oracle-like diagnostic upper bound: it performs exact slot updates and removes stale same-slot values by construction, so it should not be read as a deployable external-memory framework. Raw append is an append-only baseline that keeps all observed evidence, making stale burden visible. Heuristic CRUD is a partial-compaction baseline with rule-based update behavior. Long25 is a learned compact manager checkpoint from the constrained-slot curriculum; it is included as a compactness/reliability tradeoff point rather than as a final repaired method.

No single metric is sufficient for this setting. State accuracy alone can hide stale evidence, memory size alone can reward overly aggressive deletion, and EM/F1 alone can conflate state-management and answer-layer failures. The paper therefore reports all four metrics together.

## 4. Main Results

Figure~\\ref{fig:p63-update-frequency-tradeoff} summarizes the P6.3 hard update-frequency results. The first panel shows that Constrained CRUD, Raw append, and Heuristic CRUD all maintain slot-direct state accuracy 1.00 through k=16. Under oracle-like slot-state evaluation, the final value remains recoverable for these methods even after many repeated updates.

The slot-prompt panel tells a different story. Raw append falls from 0.90 / 0.91 EM/F1 at k=1 to 0.07 / 0.10 at k=16 on the P6.3 test split, and the P6.8 dev diagnostic gives the same collapse pattern at 0.14 / 0.17 under normal top-k5 retrieval. Heuristic CRUD follows the same qualitative pattern, reaching 0.10 / 0.13 at k=16. These methods can preserve the final value under exact slot lookup, but stale same-slot evidence remains in memory and causes answer-layer collapse. The stale-burden and memory-size panels explain this divergence: at k=16, Raw append retains 14.25 stale same-slot entries and grows to final memory size 52.00, while Heuristic CRUD still leaves 7.44 stale same-slot entries and final memory size 26.73. A P7.0 retrieval-time intervention confirms the mechanism: keeping raw-add writes unchanged but filtering the answer context to the latest entry per slot raises k=16 dev EM/F1 from 0.14 / 0.17 to 0.69 / 0.70 while memory size and stale-burden metrics remain unchanged.

The original P6.3 Long25 checkpoint occupies a different point on the curve. At k=16, it reduces stale same-slot burden to 1.13 and final memory size to 9.43. Its slot-prompt EM/F1, 0.48 / 0.53, is substantially above Raw append and Heuristic CRUD at the same update frequency. However, compactness comes with imperfect state tracking: Long25 slot-direct state accuracy is 0.91 rather than 1.00. A later three-seed Long25 retraining family reaches EM 0.87-0.89 with final memory size around 2.04 on the same k=16 test split; because it uses different checkpoints, it is reported separately as a reseeded learned-manager result rather than silently replacing the original P6.3 row.

Constrained CRUD serves as an oracle-like diagnostic upper bound rather than a deployable memory manager. It has zero stale same-slot entries by construction and reaches slot-direct state accuracy 1.00 at every k, but at k=16 it still obtains only 0.70 / 0.70 under slot-prompt answering. Even a clean final memory state therefore does not remove all answer-layer stress under the hard split.

Table~\\ref{tab:p63-k16-tradeoff} highlights the high-frequency endpoint. The main result is not a single best method, but a tradeoff: append-only methods preserve recoverability but accumulate stale burden, heuristic compaction only partially helps, and learned compact memory reduces stale burden at the cost of final-state reliability.

## 5. Error Analysis

The completed k=16 analysis sharpens two points. First, Constrained CRUD reaches perfect slot-direct state accuracy and zero stale same-slot burden at k=16, yet its slot-prompt EM/F1 remains 0.70 / 0.70. State-level analysis finds 100 / 100 correct state predictions and no state errors, so this residual gap should be interpreted as answer-layer or prompt-conditioned failure rather than stale-state retention.

Second, Long25's k=16 failures are primarily wrong-value final-state errors rather than wrong-entity or wrong-attribute errors. The canonical local analysis finds 91 / 100 correct state predictions for slot-prompt evaluation, with all 9 state errors classified as `wrong_value`. The remote Sui-3-Wu rerun provides a close confirmation rather than a replacement ledger: it reaches state accuracy 0.92 and slot-prompt EM/F1 0.49 / 0.5467, with 8 `wrong_value` state errors and the same stale same-slot burden 1.13. In both local and remote analysis, these errors are concentrated in `company` slots.

This pattern supports a compactness/reliability interpretation rather than a simple learned-manager win. Long25 substantially reduces stale burden and memory size, but its remaining k=16 failures are mostly failures to preserve the final value under hard distractors. Detailed local and remote summaries are recorded in `paper/p63_error_analysis_k16.md` and `paper/remote_verification_log.md`.

## 6. Discussion

The central finding is a tradeoff curve, not a simple method ranking. Append-only memory preserves final values under oracle lookup but accumulates stale same-slot evidence that breaks slot-conditioned answering. Heuristic compaction reduces some memory growth but still leaves enough obsolete same-slot evidence to cause severe prompted-answer degradation. Learned compact managers reduce stale burden and memory size, but may miss final updates or remain incompletely compact.

This separation matters for memory-system evaluation. A benchmark that only asks whether the final value is present can miss the practical failure mode where the correct value coexists with many obsolete alternatives. Conversely, a benchmark that only reports answer EM/F1 cannot distinguish whether a failure came from state corruption, stale interference, wrong entity grounding, wrong attribute parsing, or answer-layer behavior. MemUpdateBench is designed to keep these axes separate, so a realistic memory benchmark should report final-state reliability, stale burden, compactness, and answer robustness together.

The current results also motivate, but do not yet perform, method-improvement work. Long25 shows that learning can move a system toward lower stale burden and smaller memory, but its wrong-value failures show that compactness alone is not sufficient. The retrieval-time stale-filter intervention should likewise be read as a mechanism ablation rather than a proposed memory manager: it shows that stale context contamination causes much of raw append's prompted-answer collapse, while the remaining gap to gold-context answering indicates that retrieval and answer-layer behavior still matter. A future repair-training phase should therefore be evaluated against the same four axes rather than optimized only for prompted answer accuracy.

## 7. Limitations

MemUpdateBench is intentionally controlled. The benchmark uses synthetic slot-structured examples rather than unconstrained real user histories, because the goal is to isolate repeated same-slot update behavior and measure stale same-slot burden exactly. This control is also a limitation: real memory systems may face noisier language, implicit updates, uncertain entity boundaries, and facts whose attributes are not explicitly named.

The `slot_direct` metric is oracle-like and should not be interpreted as an end-user answering setting. Its role is to separate memory-state reliability from answer-layer robustness. The practical result should be read primarily through `slot_prompt`, where stale same-slot entries remain visible to the answer process.

The current main result stops at k=16 because k=16 already separates oracle recoverability from slot-conditioned answer robustness decisively. We do not add k=32 by default; a higher-frequency stress point should be added only if figure design, advisor feedback, or reviewer concerns require a longer tail.

The current comparison does not yet include a fair full external-framework row such as Mem0, Letta, MemGPT, or original Memory-R1. This is no longer treated as irrelevant; it is an external-validity gap. The Mem0 probe shows that a real SDK can be run with inspectable memory state, but the only discovered local endpoint is Qwen2.5-VL and the completed run is dev20, so it is reported only as qualitative feasibility rather than a main-table baseline. The repository also contains only a project-local Memory-R1-style approximation, not the original Memory-R1 code/checkpoint.

We also do not start repair training in this paper phase. The present contribution is the benchmark, diagnostic decomposition, and manuscript-ready evidence chain rather than a repaired method result. Long25 should therefore be read as a learned compact manager checkpoint, not as evidence that repair training has solved repeated same-slot updates.

## 8. Related Work Positioning

MemUpdateBench is complementary to broad long-term memory and agent-memory benchmarks such as LoCoMo, LongMemEval, AMemGym, Ledger-QA/UMA, and related multi-session memory evaluations. Those settings emphasize realistic memory use across long histories and downstream tasks. MemUpdateBench instead isolates one stressor: repeated revisions to the same `(entity, attribute)` slot. This narrow design makes stale same-slot burden directly measurable and separates it from final-state reliability and answer-layer behavior.

External memory systems such as Mem0 and MemGPT/Letta motivate the benchmark because they operationalize persistent memory in agent workflows. They are important external-validity targets, but they should be included only when the integration exposes enough memory state to compute stale same-slot burden or when the limitation is explicitly reported. Memory editing and dynamic-fact work are also related because repeated updates create conflicts between old and current values, but MemUpdateBench studies inspectable external memory rather than parametric model edits.

The exact slot format also connects to dialogue state tracking, where systems maintain slot-value states across turns. MemUpdateBench borrows this diagnostic clarity while shifting the focus to external memory persistence: the question is not only whether the current state is right, but whether obsolete values remain and later interfere with answers.

## 9. Reproducibility

The source-of-truth chain is:

```text
results/update_frequency_p63/*/evomemory_results.json
  -> scripts/summarize_update_frequency.py
  -> results/update_frequency_p63_summary/update_frequency_summary.{csv,json}
  -> scripts/package_update_frequency_paper_assets.py
  -> paper/*
```

Canonical regeneration commands are listed in `paper/p63_artifact_release_checklist.md`. The manuscript package also includes `paper/p63_metric_ledger.md` for frozen headline numbers, `paper/p63_claim_evidence_matrix.md` for claim-to-artifact mapping, `paper/p63_error_analysis_k16.md` for state-level error interpretation, and `paper/remote_verification_log.md` for the P6.7 cluster-backed sanity and Long25 spot-check record.
