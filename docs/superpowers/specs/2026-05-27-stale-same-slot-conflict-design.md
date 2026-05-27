# Stale Same-Slot Conflict Mechanism Design

## Purpose

This design upgrades MemUpdateBench from a narrow repeated-update benchmark into a mechanism-analysis paper about stale same-slot context contamination. The priority is novelty: show that obsolete values for the same `(entity, attribute)` slot are not generic retrieval noise, but a special high-similarity version-conflict mechanism that directly competes with the current value during answer generation.

The paper should still use MemUpdateBench as the controlled diagnostic tool, but the headline claim should be about the mechanism:

> Stale same-slot conflict is a measurable, causal answer-layer failure mode in memory-augmented LLMs. It is more damaging than generic distractors because stale entries remain semantically valid for the same entity and attribute, forcing the model to arbitrate between historical versions.

## Scope

### In scope

- Controlled context-construction experiments that separate stale same-slot conflict from generic distractors.
- Dose-response experiments over stale same-slot count, context order, final-value position, and version annotations.
- Causal removal/filtering interventions that compare removing stale same-slot entries against removing other retrieved entries.
- Reuse of existing P8 multi-model ceiling-recovery results as supporting evidence.
- A small retrieval/reranker appendix if time permits.

### Out of scope for this design

- Broad new benchmark claims.
- Large external SDK leaderboard.
- Training a new memory manager as the main contribution.
- Letta/MemGPT/Memory-R1 engineering unless explicitly requested later.
- Implicit, conditional, partial, or negative update semantics.

## Research questions

1. Are stale same-slot entries more damaging than equally long generic or near-slot distractor contexts?
2. Does answer performance degrade as a systematic function of same-slot conflict strength?
3. Does selectively removing stale same-slot entries recover more accuracy than removing other context entries?
4. Do existing multi-model results support the claim that stale filtering recovers each model to its own zero-stale ceiling?

## Main experiment 1: conflict-type decomposition

### Question

Is stale same-slot conflict qualitatively different from ordinary RAG noise?

### Data construction

For each target example, construct controlled answer contexts with the same target question and final gold value. Keep context length matched across conditions.

Conditions:

1. `final_only`: one current/gold memory entry.
2. `unrelated_distractors`: gold entry plus memories about unrelated entities and unrelated attributes.
3. `same_entity_different_attribute`: gold entry plus entries about the same entity but other attributes.
4. `different_entity_same_attribute`: gold entry plus entries about other entities with the same attribute.
5. `stale_same_slot`: gold entry plus obsolete values for the same entity and same attribute.

The critical comparison is condition 5 versus conditions 2-4 under matched context length.

### Metrics

- Exact match and F1.
- Value EM and answer-value-present.
- Stale-value-copied rate.
- Gold-value-present rate, fixed at 1.0 by construction.
- Condition-level degradation from `final_only`.

### Expected contribution

If stale same-slot context causes much larger degradation than other distractor types, the paper can claim that repeated-update failure is not reducible to generic retrieval noise or longer context.

## Main experiment 2: version-conflict dose-response

### Question

Does stale same-slot contamination produce a systematic dose-response curve, and what interventions reduce it?

### Factors

- Stale count: `0, 1, 2, 4, 8, 16`.
- Final-value position: `beginning`, `middle`, `end`.
- Context order: `chronological`, `reverse_chronological`, `random`.
- Version annotation: `none`, `timestamp`, `latest_outdated_label`.

The full Cartesian grid can be large, so the recommended main set is:

- stale count sweep with no annotation and final-at-end;
- final-position sweep at stale counts `2, 8, 16`;
- annotation comparison at stale counts `2, 8, 16`.

### Metrics

- EM/F1/value EM.
- Stale-value-copied rate.
- Answer-value-present rate.
- Label recovery: improvement from `none` to `latest_outdated_label`.
- Position sensitivity: gap between final-at-end and final-at-beginning/middle.

### Expected contribution

This turns stale burden into a mechanism variable rather than an aggregate metric. A strong result would show monotonic degradation with stale count, large sensitivity to final position, and partial repair from explicit version labels.

## Main experiment 3: stale-specific causal removal

### Question

Is stale same-slot content the causal driver of answer collapse, or is it merely correlated with larger memory size?

### Interventions

Start from retrieved contexts produced by `raw_add` or controlled retrieval. For each example, compare:

1. `normal`: original retrieved context.
2. `remove_random_non_gold`: remove the same number of non-gold entries at random.
3. `remove_unrelated`: remove unrelated distractors first.
4. `remove_near_slot`: remove same-entity/different-attribute or different-entity/same-attribute distractors.
5. `remove_stale_same_slot`: remove obsolete same-slot entries.
6. `latest_per_slot`: keep only the latest entry per `(entity, attribute)`.

### Metrics

- EM/F1/value EM recovery over normal.
- Gold-value-present after intervention.
- Stale-retrieved rate after intervention.
- Recovery per removed token or per removed entry.

### Expected contribution

If `remove_stale_same_slot` recovers substantially more than random or unrelated removal, the paper gets causal evidence that stale same-slot content, not generic context size, drives the failure.

## Supporting experiment 4: multi-model ceiling recovery

Reuse and foreground existing Qwen, Llama, and Mistral results.

Main table:

| Model | normal raw_add | latest_per_slot | constrained zero-stale ceiling |
| --- | ---: | ---: | ---: |
| Qwen | existing P8/P7 number | existing number | existing number |
| Llama | existing number | existing number | existing P8.1 number |
| Mistral | existing number | existing number | existing P8.2 number |

Target interpretation:

> Retrieval-time stale filtering recovers each answer model to approximately its own zero-stale ceiling. Absolute differences across models mostly reflect clean-context instruction-following capacity, not a different stale-specific mechanism.

This should support the mechanism story without becoming the main experiment.

## Supporting experiment 5: retrieval/reranker layer

This is optional and should be appendix-first unless results are especially strong.

Compare retrieval configurations:

- current MiniLM/dense fallback used by existing experiments;
- BM25 lexical retrieval;
- E5 or BGE embeddings;
- BGE reranker;
- oracle latest-per-slot filter.

Metrics:

- Final-value top-k recall.
- Stale same-slot retrieved rate.
- Final-over-stale ranking accuracy.
- Slot-prompt EM/F1.

Purpose:

Show whether stale conflict enters mainly through memory state, retrieval ordering, or answer generation. This can also explain residual clean-state failures.

## Paper narrative

### Opening angle

Start from the phenomenon rather than the benchmark:

> External memory systems often retrieve several plausible memories for a user. When these memories contain different historical values of the same slot, the answer model must decide which version is current. We show that this stale same-slot conflict is far more damaging than generic retrieval noise.

### Contribution framing

1. Define stale same-slot conflict as a controlled high-similarity conflict type in memory-augmented answering.
2. Introduce MemUpdateBench as a diagnostic construction for measuring this conflict under exact slot semantics.
3. Show conflict-type decomposition: stale same-slot entries are more harmful than generic or near-slot distractors.
4. Show dose-response and intervention evidence that stale same-slot content causally drives answer collapse.
5. Show multi-model ceiling recovery as supporting robustness evidence.

### Claims to avoid

- Do not claim MemUpdateBench is a broad long-term memory benchmark.
- Do not claim latest-per-slot is a deployable method; present it as a causal intervention.
- Do not claim external SDKs fail unless fair, documented runs exist.
- Do not overstate synthetic realism; emphasize diagnostic control.

## Implementation implications

This design likely requires new analysis/probe scripts rather than changing the core memory store.

Likely scripts:

- `scripts/run_conflict_type_probe.py`
- `scripts/summarize_conflict_type_probe.py`
- `scripts/run_stale_conflict_dose_probe.py` or extend the existing synthetic same-slot probe.
- `scripts/analyze_stale_specific_removal.py`

Existing useful code paths:

- `scripts/eval_evomemory.py` for answer prompting, context order, annotations, and latest-per-slot retrieval.
- `scripts/run_synthetic_same_slot_probe.py` for controlled conflict construction.
- `scripts/analyze_stale_dose_response.py` for dose-response summarization patterns.
- `scripts/package_p80_paper_tables.py` and `scripts/package_p80_figures.py` for paper asset integration.

## Validation plan

For new scripts:

1. Add small deterministic unit/smoke coverage for context construction.
2. Run `PYTHONPATH=. python -m py_compile` on modified/new scripts.
3. Run `PYTHONPATH=. python scripts/smoke_test.py` after integrating smoke coverage.
4. Start with a tiny local or cluster shard to verify output schema.
5. Only then launch full A40/A100 jobs.

For results:

- Every result directory should include raw JSON/CSV plus a Markdown summary.
- Update the evidence manifest only after result schema is stable.
- Record commands, outputs, metrics, and conclusions in `WORKFLOW.md`.

## Expected risks and fallback interpretations

### Risk 1: generic distractors hurt almost as much as stale same-slot entries

Fallback: frame the result as high-similarity conflict being one member of a broader retrieval-conflict family. Analyze whether same-entity or same-attribute distractors are the true drivers.

### Risk 2: labels fully solve synthetic stale conflict but not real retrieved contexts

Fallback: this is still useful. It means the answer model can arbitrate versions when version labels are explicit, but normal retrieval fails to provide reliable version metadata or surface the current value.

### Risk 3: multi-model patterns diverge

Fallback: keep the mechanism claim at the context-construction level and discuss model-specific instruction-following ceilings. Do not claim universal recovery.

### Risk 4: reranker results are weak or inconsistent

Fallback: move retrieval/reranker experiments to appendix or future work. The main novelty remains conflict-type decomposition plus causal stale removal.

## Recommended next step

Convert this design into an implementation plan with three phases:

1. Minimal novelty package: conflict-type decomposition, stale dose-response refinement, stale-specific removal.
2. Paper integration: figures, tables, and revised manuscript framing.
3. Optional robustness package: retrieval/reranker appendix and one additional model or split if the first phase reveals a compelling gap.
