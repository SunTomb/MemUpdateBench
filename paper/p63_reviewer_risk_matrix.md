# P6.7 Reviewer-Risk Matrix

| Risk | Likely reviewer concern | Response | Evidence | Mitigation / action |
| --- | --- | --- | --- | --- |
| Oracle metric realism | `slot_direct` is unrealistic. | It is a diagnostic control for final-state reliability, not the practical answer metric. | Main collapse is shown under `slot_prompt`, especially raw append k=16 EM/F1 0.07 / 0.10. | Always call it oracle-like slot-state evaluation. |
| Synthetic benchmark | Synthetic slots may not reflect real memory. | The benchmark is controlled to isolate repeated same-slot overwrites, stale burden, same-name distractors, and NOOP near misses. | Exact `(entity, attribute)` invariant enables stale burden and final-state diagnostics. | Add limitation language and position as diagnostic benchmark. |
| No external baseline | Why no Mem0/Letta/MemGPT row? | Current controlled baselines already expose the key tradeoff; external systems may not expose memory entries needed for stale-burden metrics. | `paper/external_baseline_feasibility_note.md` gives isolated Mem0 criteria. | Keep optional isolated Mem0 feasibility plan ready; do not overclaim. |
| No k=32 | Why stop at k=16? | k=16 already decisively separates oracle recoverability from slot-conditioned answer robustness. | Raw append k=16 state acc 1.00 but EM/F1 0.07 / 0.10. | Add k=32 only if figure/advisor/reviewer needs a longer tail. |
| No repair training | Why not solve the failure? | This phase establishes the benchmark and diagnostic result; repair training is a later method contribution. | Long25 already shows compactness/reliability tradeoff, not a complete solution. | Keep repair training out of current claims. |
| Long25 checkpoint reliability | Is Long25 overfit or undertrained? | Treat Long25 as a learned compact manager checkpoint, not a final method. | k=16 state acc 0.91, stale 1.13, memory size 9.43. | Avoid winner language; include diagnostic error analysis if needed. |
| Prompt artifact | Is slot-prompt collapse just prompt wording? | Constrained CRUD still reaches 0.70 / 0.70 under slot-prompt despite clean state, and state-level analysis finds 100 / 100 correct state predictions with no state errors, so answer-layer stress exists; raw/heuristic collapse is much worse and tracks stale burden. | `paper/p63_error_analysis_k16.md`, comparison across constrained, raw, heuristic, Long25 under same prompt. | Optionally add prompt robustness follow-up after main paper draft. |
| Novelty vs existing memory benchmarks | How is this different from broad memory benchmarks? | MemUpdateBench isolates repeated same-slot update frequency and distinguishes stale same-slot burden from final-state recovery. | k-sweep and stale-entry metrics are the core contribution. | Frame as diagnostic benchmark, not broad agent suite. |
| Constrained CRUD confusion | Is constrained CRUD the proposed method? | No; it is an oracle-like upper-bound diagnostic. | It has exact slot resolution and zero stale same-slot entries by construction. | State this in setup, captions, and limitations. |
| Metric overinterpretation | Does EM/F1 fully measure memory quality? | No; the paper reports EM/F1 together with state accuracy, stale burden, and memory size. | Four-panel tradeoff figure. | Keep multi-metric framing in all claims. |

## Risk-triggered optional work

External baseline implementation remains no-go for the current phase unless the external-baseline risk becomes high after manuscript integration. If that happens, use `paper/mem0_isolated_feasibility_plan.md` and do not install into the main environment.

## Short response drafts

### Why use `slot_direct`?

`slot_direct` is not intended to simulate an end-user query. It is an oracle-like diagnostic that asks whether the final value is recoverable for the exact `(entity, attribute)` slot. The key result is that this diagnostic can remain high while slot-conditioned answering collapses, showing that final-state recoverability and practical answer robustness are separable.

### Why no k=32?

The current k=16 endpoint already shows the phenomenon sharply: raw append and heuristic CRUD retain slot-direct state accuracy 1.00, but slot-prompt EM/F1 falls to 0.07 / 0.10 and 0.10 / 0.13. Adding k=32 would increase compute and complexity without changing the present claim unless a reviewer specifically asks for a longer tail.

### Why no external baseline yet?

The paper's primary contribution is a controlled diagnostic stress test, not a broad systems leaderboard. External frameworks are valuable for ecosystem grounding, but only if they expose memory state sufficiently to measure stale same-slot burden. The current plan defers them unless the manuscript needs that positioning, in which case only an isolated Mem0 feasibility side branch should be considered.

### Why trust the current Long25 result?

The canonical manuscript numbers come from the local summary ledger, and the cluster rerun is used only as a close confirmation. The remote Sui-3-Wu rerun preserves the same paper-level conclusion with slight drift, reaching state accuracy 0.92 and slot-prompt EM/F1 0.49 / 0.5467 while retaining the same low-stale pattern. The remaining errors are still dominated by `wrong_value` failures, especially on `company` slots.

### Why not start repair training now?

The current phase is about stabilizing the benchmark, evidence chain, and manuscript package. Starting repair training now would change the paper from a diagnostic benchmark story into a method-improvement branch before the current contribution is cleanly packaged. The right order is benchmark first, then optional follow-on method work.
