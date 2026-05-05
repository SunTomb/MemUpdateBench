# MemUpdateBench: How Stale Context Contaminates Memory-Augmented LLM Answering

**Anonymous Authors**

---

## Abstract

When a user tells an LLM agent "I moved to Tokyo," the agent should update its external memory—but what happens to the old entry "lives in Paris"? We show that lingering obsolete entries silently contaminate the answer-time context and cause catastrophic accuracy loss. On MemUpdateBench, a controlled diagnostic benchmark that stresses the same (entity, attribute) slot with up to 16 repeated updates, append-only memory preserves the correct final value yet collapses to 0.07 EM under prompted answering, because 14 stale same-slot entries overwhelm the context. A single retrieved stale entry halves accuracy; two push it below 20%. Through synthetic probes we identify the mechanism as an interaction of *version ambiguity* and *positional bias* rather than simple majority voting: explicit version labels almost fully repair the failure. Across three model families (Qwen, Llama, Mistral), retrieval-time stale filtering recovers each model to its own zero-stale ceiling, confirming stale context as the dominant answer-layer obstacle. These findings imply that memory systems must actively manage obsolete entries, not merely accumulate evidence.

---

## 1. Introduction

External memory gives language-model agents a way to carry user preferences, facts, and task state across sessions. Most evaluation settings ask whether a system can *retrieve* a relevant fact after it has been stored. In realistic use, however, the same fact is revised repeatedly: a user moves cities, switches jobs, updates a project deadline, or corrects an earlier statement. Each revision overwrites the intended slot value, but many memory implementations simply *append* the new evidence without removing the old. The result is a growing pool of obsolete entries that coexist with the current value.

Does this matter? Under oracle slot lookup the answer is no: the correct value is still there. Under prompted answering—where the model must select the right value from a retrieved context that may contain both current and obsolete entries—the answer is a decisive yes. On our controlled benchmark, append-only memory maintains perfect state recoverability at k=16 updates yet collapses to 0.07 EM under slot-conditioned answering. A retrieval-time stale filter that exposes only the latest entry per slot recovers EM from 0.14 to 0.69—matching the zero-stale oracle ceiling.

This paper asks three questions:

1. **Dominance.** Is stale same-slot context the dominant answer-layer obstacle, or do other factors contribute comparably?
2. **Mechanism.** What drives stale-induced answer collapse—majority voting, positional bias, version ambiguity, or their interaction?
3. **Generality.** Are these findings specific to one model, or do they hold across architectures?

Our findings are: (1) Stale context is the dominant answer-layer obstacle: filtering it recovers each of three tested models to its own zero-stale ceiling (Table 2). (2) The mechanism is an interaction of version ambiguity and positional bias; explicit version labels almost completely repair the failure even with 16 conflicting stale entries. (3) The collapse generalizes across Qwen, Llama, and Mistral, while the absolute recovery level reflects each model's prompt-following capacity rather than differential stale susceptibility.

MemUpdateBench is complementary to broad agent-memory benchmarks such as LoCoMo and LongMemEval, which emphasize ecological validity. Our benchmark is narrower by design: it makes one failure pressure—stale same-slot contamination—precisely measurable.

---

## 2. MemUpdateBench

**Slot structure.** Each example defines a target slot (entity, attribute) with a gold final value. Events take one of three actions: ADD introduces a new slot value, UPDATE revises an existing value, and NOOP mentions the entity or attribute without changing the slot. This exact structure lets us compute both whether the final value is recoverable and how many obsolete same-slot values remain.

**Update frequency as independent variable.** The main split varies k ∈ {1, 2, 4, 8, 16}, where k is the number of times the target slot is written. Larger k means more stale same-slot entries for non-compacting memory.

**Hard distractors.** The hard split includes same-name multi-entity distractors and semantic near-miss NOOP events. Success requires resolving the intended entity *and* attribute rather than reacting to surface lexical overlap.

**Scope.** The benchmark is intentionally narrow. It covers explicit key-value slot updates and does not attempt to model implicit updates, negations, or conditional revisions.

---

## 3. Experimental Setup

**Memory managers.** We compare four systems: **Constrained CRUD** (oracle-like exact slot updates, diagnostic upper bound), **Raw append** (keeps all events), **Heuristic CRUD** (rule-based partial compaction), and **Long25** (learned compact manager).

**Answer modes.** `slot_direct` checks whether the final value can be resolved by exact slot lookup. `slot_prompt` asks the model to answer using the memory contents as context. The gap between these two modes is central to the paper.

**Metrics.** We report four quantities together: (1) state accuracy under `slot_direct`, (2) EM/F1 under `slot_prompt`, (3) stale same-slot entry count, and (4) final memory size.

**Answer models.** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct.

---

## 4. Stale Contamination and Recovery

### 4.1 State accuracy masks answer collapse

One might expect that a memory system which never loses the final value would also answer correctly. Under oracle-like `slot_direct` evaluation, this expectation holds: Constrained CRUD, Raw append, and Heuristic CRUD all maintain state accuracy 1.00 through k=16.

But under `slot_prompt` answering, append-style memory collapses catastrophically. Raw append falls from 0.90 EM at k=1 to **0.07 EM** at k=16 on the hard test split, while retaining 14.25 stale same-slot entries and a final memory size of 52. The correct value is still in memory—the model simply cannot find it amid the obsolete alternatives.

| Method | State | EM / F1 | Stale | Mem. |
|--------|-------|---------|-------|------|
| Constrained CRUD | 1.00 | .70 / .70 | 0.0 | 23 |
| Raw append | 1.00 | .07 / .10 | 14.3 | 52 |
| Heuristic CRUD | 1.00 | .10 / .13 | 7.4 | 27 |
| Long25 | 0.91 | .48 / .53 | 1.1 | 9 |

*Table 1: High-frequency k=16 tradeoff (hard test).*

### 4.2 Stale filtering as diagnostic intervention

If stale same-slot entries are truly the dominant obstacle, removing them at retrieval time should recover answer accuracy—without changing anything about the stored memory.

We test this with a *latest-per-slot* intervention: raw-append writes are unchanged, but at answer time we expose only the most recent entry for each slot. On k=16 dev, this raises EM from 0.14 to **0.69**—a gain of +0.55—while stale retrieved entries drop from 4.04 to 0.

Crucially, the recovered EM (0.69) closely matches the zero-stale Constrained CRUD ceiling (0.70). This means the stale filter accounts for nearly *all* of the stale-specific answer loss.

### 4.3 Three-model ceiling recovery

| Model | Normal | Filtered | Ceiling | Gap |
|-------|--------|----------|---------|-----|
| Qwen2.5-7B | .110 | .690 | .700 | −.01 |
| Llama3.1-8B | .060 | .290 | .270 | +.02 |
| Mistral-7B | .080 | .720 | .720 | .00 |

*Table 2: Ceiling recovery on k=16 dev. All three models recover to within 0.02 EM of their own zero-stale ceiling.*

This is the paper's central result: **stale same-slot contamination is the dominant answer-layer obstacle for all tested models**, and retrieval-time stale filtering is sufficient to recover each model to its clean-memory ceiling.

---

## 5. Mechanism Analysis

### 5.1 Context presentation sensitivity

In real raw-append k=16 dev contexts, gold retrieval rate (0.360) and retrieved stale count (4.04) are identical across all conditions; only presentation changes.

| Presentation | EM | F1 |
|-------------|----|----|
| Normal (default) | .110 | .136 |
| Chronological order | .230 | .275 |
| Reverse chronological | .010 | .050 |
| Timestamp labels | .150 | .200 |
| Latest / outdated labels | .260 | .298 |

*Table 3: Real-context presentation probe. Retrieval composition is fixed.*

### 5.2 Isolating the mechanism: conflict, order, and labels

Synthetic same-slot probes with exact control over conflict type, order, and version labels (64 examples per cell):

- **Value conflict drives the sharpest collapse.** Conflicting stale values in reverse chronological order: stale=1 EM 0.234, stale=2 EM 0.094, stale=8–16 EM ≈0.
- **Majority voting is not the full story.** Chronological order (current value last): EM remains 0.797 even at stale=16.
- **Version labels repair the failure.** Adding [latest]/[outdated] labels to reverse-chrono conflict: EM 0.969–1.000 across all stale counts.
- **Same-value repetition has a separate effect.** answer-value-present stays 1.000 but exact EM drops (0.688 at stale=4, 0.531 at stale=16).

### 5.3 Position sensitivity

Lost-in-the-Middle-style gold-position probe (k=16 dev, Qwen):

| Gold position | EM | F1 | Ans. value |
|--------------|----|----|------------|
| Beginning | .010 | .073 | .040 |
| Middle | .090 | .183 | .190 |
| End | .630 | .654 | .720 |

*Table 4: Monotonic recency advantage—no U-shape recovery.*

### 5.4 Dose–response

EM drops from 0.967 with zero stored stale entries to 0.743 with one and 0.290 with three. ED50: 3.18 stored stale entries vs. 1.89 retrieved stale entries. Answer-context exposure is the proximate mechanism.

---

## 6. Error Analysis and Residual Limits

Constrained CRUD reaches state accuracy 1.00 and zero stale entries at k=16, yet its `slot_prompt` EM remains 0.70. The 0.30 residual reflects clean-context answer-layer limits rather than stale contamination.

Llama's clean-context ceiling (0.27) is far below Qwen's (0.70) and Mistral's (0.72), even though all three have identical memory state. These differences are instruction-following gaps.

Long25 greatly reduces stale burden (1.13 vs. 14.25) and memory size (9 vs. 52), but its state accuracy is 0.91: primarily wrong final values under hard distractors.

---

## 7. Related Work

**Long-term memory benchmarks.** LoCoMo and LongMemEval test memory *existence*—can the system remember that a fact was stated? MemUpdateBench tests memory *hygiene*—do obsolete versions persist and interfere? Broad benchmarks typically use each fact only once and can miss this failure mode.

**External memory systems.** MemGPT and Mem0 are *systems* that must solve many problems simultaneously; MemUpdateBench is a *diagnostic tool* that isolates one stressor.

**Knowledge editing.** Editing suffers from *ripple effects*; external-memory update suffers from *stale contamination*. Different failure modes require different evaluation.

**Dialogue state tracking.** MemUpdateBench borrows DST's slot-value diagnostic clarity while shifting focus to external memory persistence.

**Long-context position bias.** Lost in the Middle shows U-shaped accuracy in long contexts. Our probe shows *monotonic* recency advantage in short stale-contaminated contexts—a stronger interference signal.

---

## 8. Discussion and Limitations

**Design implications.** Memory systems that accumulate evidence without version management will fail under repeated updates, regardless of the answer model's capability. Retrieval-time stale removal is the most effective single intervention.

**Limitations.** (1) Synthetic slot-structured examples, not unconstrained real user histories. (2) No fair evaluation of external memory SDKs. (3) Stops at k=16. (4) Contributes benchmark and analysis, not a repaired method.

**Reproducibility.** Code, data, and canonical regeneration commands will be released upon acceptance. All manuscript numbers trace to versioned result artifacts through a claim–evidence matrix.
