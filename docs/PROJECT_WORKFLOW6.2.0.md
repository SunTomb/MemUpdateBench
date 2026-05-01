# PROJECT_WORKFLOW6.2.0 — Rich Diagnostics and Retrieval Robustness for Update-Frequency Stress

## Context

`PROJECT_WORKFLOW6.1.0.md` established that the controlled `update_frequency` EvoMemory variant has a real signal:

- deterministic oracle ceiling is 1.0 for k = 1, 2, 4, 8, 16;
- `outputs/constrained_sft_curriculum_long25/best` drops from k=1 to k=16 by 13 percentage points under `slot_direct`;
- errors are `wrong_value`, concentrated in company updates;
- `raw_add` and `heuristic_crud` look perfect under `slot_direct`, exposing an evaluation pitfall.

P6.2 adds richer diagnostics and non-oracle answer modes to decide whether this direction is worth continued investment.

## Code Changes

Modified locally and synced to Tang-2-Wu:

```text
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
```

Added per-example diagnostics:

```text
same_slot_entry_count
stale_same_slot_entry_count
gold_same_slot_entry_count
write_amplification
target_slot_write_count
```

Added summary diagnostics:

```text
same_slot_entry_count_avg
stale_same_slot_entry_count_avg
gold_same_slot_entry_count_avg
write_amplification_avg
target_slot_write_count_avg
final_memory_size_avg
```

Local validation:

```bash
python -m py_compile scripts/eval_evomemory.py scripts/analyze_ood_errors.py
python scripts/smoke_test.py
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_k2_pilot2_test.json \
  --output_dir results/update_frequency_pilot2_rich/local_oracle_k2
```

Result:

```text
SMOKE TEST: 31/31 passed
state_accuracy = 1.0
same_slot_entry_count_avg = 1.0
stale_same_slot_entry_count_avg = 0.0
gold_same_slot_entry_count_avg = 1.0
write_amplification_avg = 0.75
target_slot_write_count_avg = 2.0
```

Remote validation:

```bash
scp scripts/eval_evomemory.py scripts/analyze_ood_errors.py Tang-2-Wu:/NAS/yesh/G-MSRA/scripts/
ssh Tang-2-Wu 'cd /NAS/yesh/G-MSRA && source /NAS/yesh/G-MSRA/activate.sh && PYTHONPATH=. python -m py_compile scripts/eval_evomemory.py scripts/analyze_ood_errors.py'
```

## Rich Slot-Direct Curves

Remote outputs:

```text
results/update_frequency_pilot2_rich/
```

### constrained_slot_crud

| k | state_acc | EM | final_mem | same_slot | stale_same_slot | gold_same_slot | write_amp | target_writes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.000 | 1.00 |
| 2 | 1.00 | 1.00 | 3.00 | 1.00 | 0.00 | 1.00 | 0.750 | 2.00 |
| 4 | 1.00 | 1.00 | 5.00 | 1.00 | 0.00 | 1.00 | 0.500 | 4.00 |
| 8 | 1.00 | 1.00 | 8.00 | 1.00 | 0.00 | 1.00 | 0.381 | 8.00 |
| 16 | 1.00 | 1.00 | 15.00 | 1.00 | 0.00 | 1.00 | 0.341 | 16.00 |

### raw_add

| k | state_acc | EM | final_mem | same_slot | stale_same_slot | gold_same_slot | write_amp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.000 |
| 2 | 1.00 | 1.00 | 4.00 | 2.00 | 1.00 | 1.00 | 1.000 |
| 4 | 1.00 | 1.00 | 10.00 | 4.00 | 3.00 | 1.00 | 1.000 |
| 8 | 1.00 | 1.00 | 21.00 | 8.00 | 6.75 | 1.25 | 1.000 |
| 16 | 1.00 | 1.00 | 44.00 | 16.00 | 14.25 | 1.75 | 1.000 |

### heuristic_crud

| k | state_acc | EM | final_mem | same_slot | stale_same_slot | gold_same_slot | write_amp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.000 |
| 2 | 1.00 | 1.00 | 3.93 | 1.93 | 0.93 | 1.00 | 0.983 |
| 4 | 1.00 | 1.00 | 8.80 | 3.89 | 2.89 | 1.00 | 0.880 |
| 8 | 1.00 | 1.00 | 14.24 | 7.28 | 6.08 | 1.20 | 0.678 |
| 16 | 1.00 | 1.00 | 19.81 | 11.98 | 10.78 | 1.20 | 0.450 |

### learned long25

Checkpoint:

```text
outputs/constrained_sft_curriculum_long25/best
```

| k | state_acc | resolve | EM/F1 | final_mem | same_slot | stale_same_slot | gold_same_slot | write_amp | target_writes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.000 | 1.00 |
| 2 | 0.85 | 1.00 | 0.85 | 2.00 | 1.00 | 0.15 | 0.85 | 0.500 | 1.85 |
| 4 | 0.94 | 1.00 | 0.94 | 2.00 | 1.00 | 0.06 | 0.94 | 0.200 | 3.67 |
| 8 | 0.88 | 1.00 | 0.88 | 1.92 | 1.00 | 0.12 | 0.88 | 0.091 | 7.60 |
| 16 | 0.87 | 1.00 | 0.87 | 2.00 | 1.00 | 0.13 | 0.87 | 0.045 | 14.12 |

Interpretation:

- `raw_add` is not genuinely perfect. It is perfect only under oracle-like slot lookup.
- `raw_add` accumulates stale same-slot entries roughly proportional to k.
- `heuristic_crud` reduces total memory growth versus raw append, but still accumulates stale same-slot burden.
- `long25` is compact and keeps one same-slot entry, but misses final updates in 6–15% of cases depending on k.

This establishes a useful tradeoff axis:

```text
append-only: high final-state recoverability under slot_direct, high stale burden
learned compact manager: low stale burden, but missed-update final-state errors
```

## Retrieval Robustness

Remote outputs:

```text
results/update_frequency_pilot2_retrieval/
```

Evaluated k = 1, 8, 16 for:

```text
constrained_slot_crud
raw_add
learned_constrained_slot long25
```

with answer modes:

```text
slot_prompt
rag
```

### slot_prompt

| method | k=1 EM/F1 | k=8 EM/F1 | k=16 EM/F1 | interpretation |
|---|---:|---:|---:|---|
| constrained_slot_crud | 0.89 / 0.90 | 0.99 / 0.99 | 0.97 / 0.97 | compact oracle remains strong |
| raw_add | 0.89 / 0.90 | 0.18 / 0.20 | 0.08 / 0.08 | append-only collapses under stale entries |
| long25 | 1.00 / 1.00 | 0.88 / 0.88 | 0.87 / 0.87 | mirrors state accuracy; compact but misses writes |

Key diagnostic values:

```text
raw_add k8:  stale_same_slot=6.75, final_mem=21.00, answer_present=0.23
raw_add k16: stale_same_slot=14.25, final_mem=44.00, answer_present=0.12
```

This is the strongest result from P6.2. It shows that append-only memory can score 1.0 under `slot_direct` while becoming unusable under a more realistic slot-conditioned prompt.

### rag

| method | k=1 EM/F1 | k=8 EM/F1 | k=16 EM/F1 | interpretation |
|---|---:|---:|---:|---|
| constrained_slot_crud | 0.66 / 0.72 | 0.25 / 0.29 | 0.13 / 0.16 | generic RAG answer layer degrades even with clean state |
| raw_add | 0.66 / 0.72 | 0.08 / 0.08 | 0.06 / 0.06 | stale burden plus retrieval/prompt weakness |
| long25 | 0.42 / 0.47 | 0.26 / 0.28 | 0.08 / 0.10 | answer-layer instability dominates |

Interpretation:

- `rag` is useful as a harsh answer-layer stress test.
- It should not be the main state-management metric because even deterministic compact state collapses with k.
- `slot_prompt` is the better non-oracle comparison: it preserves compact oracle performance while penalizing stale appended memories.

## External Baseline Feasibility

Cluster import check on Tang-2-Wu:

```text
mem0   not_installed
mem0ai not_installed
memgpt not_installed
letta  not_installed
```

Package-index visibility check:

```text
mem0ai available, latest observed 2.0.1
letta available, latest observed 0.10.0
memgpt available, latest observed 0.2.0
```

Repository-local baselines exist:

```text
baselines/memory_r1_agent.py
baselines/mem0_memoryr1_agent.py
baselines/reflexion_agent.py
baselines/evolver_agent.py
baselines/self_consolidation_agent.py
gmsra/baselines.py
```

But these are project-local approximations or aligned internal baselines, not original external implementations. They are useful ablations but not enough as a paper's only external comparison.

Feasibility matrix:

| baseline | current status | credibility | next action |
|---|---|---|---|
| constrained_slot_crud | implemented | oracle/internal ceiling | keep as ceiling |
| raw_add | implemented | simple internal baseline | keep; now has meaningful stale-burden weakness |
| heuristic_crud | implemented | internal heuristic | keep as intermediate ablation |
| learned long25 | implemented | internal learned manager | keep as compact learned baseline |
| project-local Memory-R1 | implemented | weak as external claim | label as approximation only |
| Mem0 / mem0ai | package available, not installed | credible if adapted cleanly | test in isolated venv/conda env |
| Letta / MemGPT | package available, not installed | credible but likely heavier | feasibility spike only after Mem0 |
| original Memory-R1 | not locally available | unknown | locate official code/checkpoint before claiming |

No shared environment was modified for external packages.

## Go / No-Go Decision

Decision: **Go, with revised framing.**

The update-frequency direction is now stronger than in P6.1 because P6.2 established two separable failure modes:

1. **Append-only stale burden**
   - `raw_add` has perfect slot-direct accuracy but collapses under `slot_prompt` at high k.
   - This directly validates the critique that slot-direct evaluation hides stale memory burden.

2. **Compact learned missed updates**
   - `long25` maintains compact same-slot state but misses late/final updates.
   - Its state curve remains k=1 to k=16: 1.00 → 0.87.

This gives a plausible paper story:

```text
Repeated same-slot updates expose a reliability tradeoff between recoverability, stale burden, and compact state maintenance.
```

The direction should continue, but the main evaluation stack should be:

1. deterministic oracle ceiling;
2. slot-direct state accuracy;
3. stale same-slot burden and final memory size;
4. `slot_prompt` answer robustness;
5. `rag` only as a secondary answer-layer stress test.

## Remaining Risks

1. External baseline still missing.
   - Mem0/Letta/MemGPT are package-visible but not installed.
   - Must use an isolated environment before any claim.

2. Current `update_frequency` split is still synthetic and limited.
   - Company-specific learned failures may indicate attribute imbalance or model shortcut.
   - Need a harder variant only after preserving oracle ceiling.

3. `rag` answer mode is too unstable to serve as the main metric.
   - It conflates memory-state failure with answer-layer retrieval/prompt failure.

4. `slot_prompt` is non-oracle but still slot-conditioned.
   - This is acceptable for a state-management paper if framed honestly.

## Next Recommended Work

1. Build a harder update-frequency variant with:
   - same-name multi-entity distractors;
   - near-miss NOOP events;
   - balanced company cases;
   - optional k=32.
2. Run deterministic oracle first and require `state_accuracy >= 0.98`.
3. Test `slot_prompt` on raw_add, heuristic_crud, constrained_slot_crud, and long25.
4. Create an isolated environment for Mem0 and test whether it can process the EvoMemory event stream with stable cost/runtime.
5. Only after these checks, decide whether to train a targeted repair for long25 missed final updates.
