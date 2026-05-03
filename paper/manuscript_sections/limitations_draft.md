# Limitations Draft

MemUpdateBench is intentionally controlled. The benchmark uses synthetic slot-structured examples rather than unconstrained real user histories, because the goal is to isolate repeated same-slot update behavior and measure stale same-slot burden exactly. This control is also a limitation: real memory systems may face noisier language, implicit updates, and facts whose entity or attribute boundaries are not explicit.

The `slot_direct` metric is oracle-like. It should not be interpreted as an end-user answering setting. Its role is to separate memory-state reliability from answer-layer robustness. The practical result should be read primarily through `slot_prompt`, where stale same-slot entries remain visible to the answer process.

The current main result stops at k=16. We do not add k=32 by default because k=16 already separates oracle recoverability from slot-conditioned answer robustness decisively. A higher-frequency stress point should be added only if the figure draft, advisor feedback, or reviewer concerns require a longer tail.

The current comparison does not include a full external framework such as Mem0, Letta, or MemGPT. This is a deliberate scope decision for the current paper phase. The controlled baselines already expose the tradeoff curve, while external frameworks may require heavier integration and may not expose memory entries in a way that supports stale same-slot diagnostics. If ecosystem grounding becomes necessary, an isolated Mem0 feasibility run should be conducted separately.

Finally, Long25 is a learned compact manager checkpoint, not a final repair-training result. We do not claim that repair training has solved repeated same-slot updates. Repair training is a future method direction that should follow, not precede, the benchmark and diagnostic framing.
