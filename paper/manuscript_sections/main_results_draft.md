# Main Results Draft

## Repeated updates separate recoverability from answer robustness

To test whether repeated same-slot updates affect state and answering in the same way, we compare the P6.3 hard update-frequency conditions under exact slot lookup and slot-conditioned answering. Constrained CRUD, Raw append and Heuristic CRUD all retain `slot_direct` state accuracy 1.00 through `k=16`, so the final value remains recoverable under the state diagnostic. The corresponding prompted-answer results diverge sharply: Raw append falls from 0.90/0.91 EM/F1 at `k=1` to 0.07/0.10 at `k=16`, while Heuristic CRUD reaches 0.10/0.13.

## Stale burden and compactness define a tradeoff curve

The memory traces explain part of this divergence. At `k=16`, Raw append retains 14.25 stale same-slot entries and final memory size 52.00. Heuristic CRUD reduces final memory size to 26.73 but retains 7.44 stale same-slot entries. These values show why final-state recoverability alone is insufficient: the final value can remain present while obsolete values remain available to the answer layer. The result is a tested tradeoff, not a universal law about all memory systems.

Long25 occupies a different point on the curve. It reduces stale burden to 1.13 and final memory size to 9.43, while reaching 0.48/0.53 EM/F1 under prompted answering. Its `slot_direct` state accuracy is 0.91, however. Compactness therefore improves the stale-context condition in this panel while introducing missed-final-update errors; Long25 is not a uniformly superior method.

## Clean state does not guarantee a correct prompted answer

Constrained CRUD provides a useful diagnostic upper bound: it has no stale same-slot entries and reaches state accuracy 1.00 at every tested `k`. Nevertheless, its `k=16` prompted-answer EM/F1 is 0.70/0.70. Within this controlled setup, the residual gap is consistent with reader serialization, prompt conditioning or answer-model behavior rather than stale-state retention. We report this as a layer separation, not as a claim that the state diagnostic is a deployable memory interface.

## Mechanism diagnostics and answer-model boundaries

The later P8 diagnostics examine why stale evidence can affect answering. Earlier answer-model controls show dose-response and order/metadata effects in tested stale-context conditions, while stale filtering recovers those controls toward their own zero-stale ceilings. The qualified Qwen3.5-9B matrix supplies an important boundary: it remains nearly perfect on the frozen Raw Append contexts (overall EM 0.9979167), including reverse/no-label high-`k` cells. Thus version arbitration and stale-context sensitivity are answer-model-dependent; the evidence does not support a universal answer-layer failure claim.

## External systems and native-system evidence are separate layers

External-system probes are reported with explicit support envelopes. Letta and the LangGraph Store/custom-adapter pipeline each support 52/80 Family-A tasks in their disclosed joint extraction/storage/reader configurations, with 28 multi-object tasks typed `NOT_SUPPORTED`; their supported rows are not native Letta-only or native LangMem-only accuracy. Mem0 is a real but preliminary off-the-shelf probe with inspectable memory and near-zero exact-slot performance on the small dev20 configuration; it is not a matched main-table baseline.

The native Graphiti development screen did not observe the prespecified current-value degradation signal and was formally `INCONCLUSIVE`; held-out validation was not consumed. A separately authorized historical-serving retry completed all three runtime cases, but the frozen evaluator rejected the native Chinese relation facts under its conservative exact-English assertion grammar. The resulting compact decision is `BLOCKED` with null mechanism labels. This technical/evaluator boundary supplies no additional accuracy or mechanism result.
