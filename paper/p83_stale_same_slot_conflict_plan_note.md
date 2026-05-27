# P8.3 Stale Same-Slot Conflict Mechanism Plan

## Motivation

The next novelty-focused step is to sharpen MemUpdateBench from a repeated-update diagnostic into a mechanism analysis of stale same-slot context contamination. The target claim is that stale same-slot entries are not generic retrieval noise: they are high-similarity version conflicts that remain valid for the same entity and attribute while competing with the current value.

## Minimal experiment package

1. **Conflict-type decomposition** compares matched contexts with final-only evidence, unrelated distractors, same-entity/different-attribute distractors, different-entity/same-attribute distractors, and stale same-slot distractors.
2. **Version-conflict dose-response** extends the synthetic same-slot probe with middle/random placement and selected stale-count/order/annotation conditions.
3. **Stale-specific removal analysis** estimates whether removing stale same-slot entries is more targeted than removing unrelated or near-slot entries.

## Main expected paper claim

If the planned results match the hypothesis, the paper should claim that stale same-slot conflict is a distinct answer-layer failure mode in memory-augmented LLMs. Generic distractors may degrade performance, but obsolete same-slot values should cause disproportionate collapse because the answer model must arbitrate between historical versions of the same slot.

## Integration points

- Use `results/p83_conflict_type_probe_summary/` for the main novelty table.
- Use `results/p83_stale_conflict_dose_summary/` for the dose-response and order/annotation appendix table.
- Use `results/p83_stale_specific_removal_trace/` as a cheap trace-level guide before running full answer-model removal interventions.
- Keep existing Qwen/Llama/Mistral ceiling-recovery results as supporting evidence, not the main novelty.

## Guardrails

- Do not reframe MemUpdateBench as a broad memory benchmark.
- Do not present latest-per-slot as a deployed method.
- Do not claim external SDK failure without fair adapter runs.
- Treat synthetic probes as controlled mechanism evidence, not ecological realism.
