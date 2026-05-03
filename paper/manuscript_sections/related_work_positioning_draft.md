# Related Work and Positioning Draft

## Recommended manuscript framing

MemUpdateBench should be positioned as a controlled diagnostic benchmark for repeated updates to the same `(entity, attribute)` slot. It is not a broad long-term memory benchmark, not a general agent leaderboard, and not a replacement for realistic dialogue-history evaluations. Its value is that it makes final-state reliability, stale same-slot burden, memory compactness, and slot-conditioned answer robustness jointly measurable under an exact update-frequency axis.

A safe contribution statement is:

> We study a narrow but under-isolated failure mode in external memory systems: repeated updates to the same memory slot. This setting lets us separate whether the final value is recoverable, whether obsolete same-slot values remain in memory, how compact the resulting memory is, and whether a downstream answer model can robustly answer from the retrieved memory context.

## Long-term and agent-memory benchmarks

Long-term memory benchmarks such as LoCoMo and LongMemEval evaluate whether systems can maintain useful information across long contexts, long histories, or multi-session interactions. These benchmarks are valuable because they are closer to realistic user-facing memory scenarios, where facts are distributed across many turns and answers require selecting relevant evidence from a large history. AMemGym and related agent-memory benchmarks further broaden the setting by testing how memory affects downstream agent behavior, decisions, or task success.

MemUpdateBench is complementary rather than competing with these benchmarks. Its stressor is deliberately narrower: the same `(entity, attribute)` slot is updated repeatedly while distractors and near-miss facts are controlled. This design sacrifices scenario breadth to make stale same-slot burden directly measurable. In broad long-term memory benchmarks, an incorrect answer may reflect retrieval failure, reasoning failure, ambiguity, or missing memory. In MemUpdateBench, the repeated-slot structure makes it possible to count obsolete retained values and compare them with final-state recoverability and prompted answer robustness.

This narrower design also explains why the benchmark should not be described as a comprehensive memory benchmark. Its claim is diagnostic: repeated same-slot updates reveal whether a memory system can preserve the current value without carrying forward obsolete values that later contaminate retrieval and answer generation.

## External memory systems

Systems such as Mem0 and MemGPT/Letta operationalize persistent memory for LLM agents through memory extraction, storage, retrieval, summarization, and updating. They motivate MemUpdateBench because they are exactly the kinds of systems that may face repeated user fact updates in deployment. However, these systems are often optimized for naturalistic memory usefulness rather than exact final-value tracking for a controlled slot. Their public interfaces may also expose answer behavior more readily than inspectable memory state, which limits whether stale same-slot burden can be computed.

For this reason, external systems should be framed as external-validity targets, not as already-settled comparisons. The current Mem0 probe shows that a real SDK can be run end-to-end with inspectable memory, but the only discovered local backend is Qwen2.5-VL and the run is dev20. This is enough to show feasibility and misalignment under one configuration, but it is not a fair main-table baseline. A fair external-system row should use a text-only backend, a larger split, and an adapter that gives the system a reasonable chance to preserve exact final slot values.

The manuscript should therefore avoid claiming that Mem0, MemGPT, or Letta broadly fail on MemUpdateBench unless such systems are evaluated under matched and documented configurations. The safer claim is that MemUpdateBench exposes a concrete evaluation target for these systems: exact repeated-slot final-value tracking with inspectable stale-burden metrics.

## Memory-R1, AgeMem, and learned memory-management methods

Memory-R1, ReMemR1, AgeMem, and related learned memory-management methods are closer methodological neighbors than SDK-style memory systems. They study how an agent or memory manager should decide whether to add, update, delete, or keep memories, often using SFT or RL-style training signals. This overlaps with MemUpdateBench because both settings care about CRUD-like memory operations and downstream answer quality.

The distinction is contribution type. MemUpdateBench should not be framed primarily as a new memory-management method. Its main contribution is a diagnostic evaluation setting and metric decomposition. A learned Long25 manager can be included as one method point, but the paper's central claim should not be that Long25 is superior to Memory-R1 or AgeMem. Instead, Memory-R1-style methods motivate the need for a benchmark that separates operation quality from retrieval-time stale contamination and answer-layer failures.

The current repository contains only a project-local `baselines/memory_r1_agent.py` approximation. Unless the original Memory-R1 code and checkpoint are run, this local approximation must not be reported as an original Memory-R1 external baseline. The paper can mention Memory-R1 in related work and state that original-code evaluation is an important future comparison.

## Dynamic knowledge, memory editing, and stale facts

Knowledge editing and dynamic-fact benchmarks study how systems should update facts when the world changes. They are conceptually related because repeated same-slot updates also create conflicts between old and new values. In model editing, the old value may remain implicitly in model parameters or be triggered by paraphrases. In external memory, the old value is often explicitly retained as a memory entry.

MemUpdateBench focuses on this external-memory version of the problem. Because memory entries are inspectable, stale same-slot values can be counted rather than inferred only from output errors. This makes the benchmark useful as a controlled complement to dynamic knowledge work: it asks not whether a parametric model has internalized a new fact, but whether an external memory system can keep the current value accessible while preventing obsolete values from dominating retrieval and prompted answering.

## Dialogue state tracking and slot-value evaluation

Dialogue state tracking (DST) represents dialogue progress as slot-value states and evaluates whether the current state is correct. MemUpdateBench borrows the clarity of slot-value evaluation from this tradition: exact `(entity, attribute)` slots make it possible to define a current value and count stale alternatives. This is not merely a simplifying assumption; it is the mechanism that makes the stale-burden metric well-defined.

The task setting is different from classical DST. DST usually tracks a dialogue ontology for task completion, whereas MemUpdateBench studies persistent external memory after a sequence of updates and distractors. The benchmark evaluates not only whether the current slot value can be recovered, but also how many obsolete same-slot values remain in memory, how compact the memory is, and whether a downstream LLM answer layer can use retrieved entries robustly.

## Positioning summary

MemUpdateBench should be presented as a controlled diagnostic that complements broader memory benchmarks and memory-agent systems. Its scope is intentionally narrow: repeated explicit updates to the same slot. Within that scope, it provides a clean way to separate four quantities that are often conflated:

1. final-state reliability;
2. stale same-slot burden;
3. memory compactness;
4. slot-conditioned answer robustness.

The manuscript should avoid broad claims that are not supported by the current evidence:

- Do not call MemUpdateBench a comprehensive long-term memory benchmark.
- Do not claim coverage of implicit updates, conditional updates, partial updates, or real user histories.
- Do not list Mem0, MemGPT/Letta, or original Memory-R1 as main evaluated baselines unless a fair, documented run exists.
- Do not present Constrained CRUD as a deployable method; it is an upper-bound diagnostic.
- Do not present retrieval-time stale filtering as a proposed system unless it is clearly labeled as an intervention/ablation.

The strongest paper claim after P6.8/P6.9/P7.0 should be:

> Repeated same-slot updates expose a failure mode that is invisible under final-state lookup alone: append-only memories can retain the correct final value while accumulating stale same-slot entries that collapse prompted answering. Controlled gold-context and retrieval-time stale-filter interventions isolate this mechanism, while clean-state CRUD results reveal a separate retrieval/answer-layer gap.
