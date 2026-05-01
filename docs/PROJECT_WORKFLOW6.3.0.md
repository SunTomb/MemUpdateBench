# PROJECT_WORKFLOW6.3.0 — Hard Update-Frequency Benchmark

## Context

P6.2 established that the update-frequency direction is viable but needed a harder benchmark:

- `raw_add` is perfect under `slot_direct` but collapses under `slot_prompt` when stale same-slot entries accumulate.
- `long25` is compact but misses some late/final updates.
- `rag` is too unstable to serve as the main state-management metric.

P6.3 therefore introduced a harder update-frequency split with same-name multi-entity distractors and semantically near-miss NOOP events.

## Code Changes

Modified:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
```

### Generator

Added a new EvoMemory variant:

```bash
python scripts/prepare_data.py \
  --dataset evomemory \
  --output_dir data \
  --evomemory_variant update_frequency_hard \
  --seed 53 \
  --output_suffix p63
```

Generated:

```text
data/evomemory_update_frequency_hard_k1_p63_dev.json
data/evomemory_update_frequency_hard_k2_p63_dev.json
data/evomemory_update_frequency_hard_k4_p63_dev.json
data/evomemory_update_frequency_hard_k8_p63_dev.json
data/evomemory_update_frequency_hard_k16_p63_dev.json
data/evomemory_update_frequency_hard_k1_p63_test.json
data/evomemory_update_frequency_hard_k2_p63_test.json
data/evomemory_update_frequency_hard_k4_p63_test.json
data/evomemory_update_frequency_hard_k8_p63_test.json
data/evomemory_update_frequency_hard_k16_p63_test.json
data/evomemory_update_frequency_hard_p63_train.json
data/evomemory_update_frequency_hard_p63_dev.json
data/evomemory_update_frequency_hard_p63_test.json
```

Design:

- k values: `1, 2, 4, 8, 16`
- 100 train/dev/test examples per k
- attributes: `location`, `company`, `preference`, `language`
- same-name multi-entity distractors, e.g. `friend_alex` vs `coworker_alex`
- relation-qualified mentions to preserve deterministic oracle resolution
- near-miss NOOP events:
  - considering/visiting locations;
  - interviewing/reading company news;
  - trying/buying drinks;
  - discussing/reading language tutorials;
- metadata:
  - `stress_type = update_frequency_hard`
  - `distractor_level = same_name_multi_entity`
  - `noop_level = semantic_near_miss`
  - `k_updates`
  - `num_events`
  - `num_target_updates`

### Eval / analysis metadata

`eval_evomemory.py` now passes through:

```text
stress_type
k_updates
distractor_level
noop_level
num_target_updates
```

`analyze_ood_errors.py` now groups by:

```text
by_k_updates
by_stress_type
by_distractor_level
by_noop_level
```

## Validation

Local validation:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py
python scripts/smoke_test.py
```

Result:

```text
SMOKE TEST: 31/31 passed
```

## Oracle Gate

Deterministic oracle on local machine:

```bash
for k in 1 2 4 8 16; do
  PYTHONPATH=. python scripts/eval_evomemory.py \
    --mode constrained_slot_crud \
    --answer_mode slot_direct \
    --data_file data/evomemory_update_frequency_hard_k${k}_p63_test.json \
    --output_dir results/update_frequency_p63/oracle_slot_direct_k${k}
done
```

Result:

| k | state_accuracy | state_resolve | EM | F1 | stale_same_slot | final_mem |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 2.00 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 4.00 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 7.00 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 12.00 |
| 16 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 23.00 |

The hard split passes the oracle requirement.

## Operational Notes

Remote runs on Tang-2-Wu hit two infrastructure issues:

1. `slot_direct` on remote initially used sentence-transformer on GPU0 and failed when GPU0 was full.
2. `/NAS` intermittently reached full/near-full state, causing some result files to be written as 0-byte JSON files.

Mitigation:

- non-model `slot_direct` was rerun locally;
- remote successful `slot_prompt` and learned results were downloaded locally;
- empty P6.3 result files and failed logs were cleaned from the remote project directory.

This did not affect the final reported metrics below.

## Non-Model Results

### slot_direct

| method | k=1 | k=2 | k=4 | k=8 | k=16 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| constrained_slot_crud state_acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | oracle ceiling |
| raw_add state_acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | slot_direct hides stale burden |
| heuristic_crud state_acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | slot_direct hides stale burden |

Stale same-slot burden:

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| constrained_slot_crud | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| raw_add | 0.00 | 1.00 | 3.00 | 6.75 | 14.25 |
| heuristic_crud | 0.00 | 1.00 | 3.00 | 6.75 | 14.25 |

Final memory size:

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| constrained_slot_crud | 2.00 | 4.00 | 7.00 | 12.00 | 23.00 |
| raw_add | 2.00 | 5.00 | 12.00 | 25.00 | 52.00 |
| heuristic_crud | 2.00 | 5.00 | 12.00 | 25.00 | 52.00 |

Interpretation: P6.3 strengthens the P6.2 metric-pitfall finding. Exact slot lookup makes append-only methods look perfect while stale burden grows nearly linearly with k.

### slot_prompt

| method | k=1 EM/F1 | k=2 EM/F1 | k=4 EM/F1 | k=8 EM/F1 | k=16 EM/F1 |
|---|---:|---:|---:|---:|---:|
| constrained_slot_crud | 0.90 / 0.91 | 0.90 / 0.94 | 0.86 / 0.86 | 0.98 / 0.98 | 0.70 / 0.70 |
| raw_add | 0.90 / 0.91 | 0.78 / 0.79 | 0.24 / 0.27 | 0.06 / 0.10 | 0.07 / 0.10 |
| heuristic_crud | 0.89 / 0.91 | 0.80 / 0.81 | 0.28 / 0.28 | 0.19 / 0.22 | 0.10 / 0.13 |

Key observations:

- `raw_add` collapses from k=1 EM=0.90 to k=16 EM=0.07.
- `heuristic_crud` also collapses, though slightly less severely at k=8/16.
- `constrained_slot_crud` remains much stronger, but k=16 EM=0.70 shows that the hard split also stresses the answer layer even with clean compact state.

## Learned Long25 Results

Checkpoint:

```text
outputs/constrained_sft_curriculum_long25/best
```

### slot_direct

| k | state_accuracy | EM | F1 | answer_present | stale_same_slot | final_mem |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.46 |
| 2 | 0.89 | 0.89 | 0.89 | 0.89 | 0.23 | 3.45 |
| 4 | 0.92 | 0.92 | 0.92 | 0.92 | 0.46 | 4.34 |
| 8 | 0.85 | 0.85 | 0.86 | 0.88 | 0.74 | 6.18 |
| 16 | 0.91 | 0.91 | 0.91 | 0.92 | 1.13 | 9.43 |

### slot_prompt

| k | state_accuracy | EM | F1 | answer_present | stale_same_slot | final_mem |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.46 |
| 2 | 0.89 | 0.79 | 0.79 | 0.80 | 0.23 | 3.45 |
| 4 | 0.92 | 0.76 | 0.77 | 0.78 | 0.46 | 4.34 |
| 8 | 0.85 | 0.73 | 0.74 | 0.74 | 0.74 | 6.18 |
| 16 | 0.91 | 0.48 | 0.53 | 0.51 | 1.13 | 9.43 |

Interpretation:

- long25 is substantially more robust than raw_add / heuristic_crud under `slot_prompt` at high k.
- It is no longer perfectly compact on this hard split; final memory grows to 9.43 at k=16 and stale same-slot count reaches 1.13.
- The state curve is not monotonic: k=16 state accuracy recovers to 0.91, but `slot_prompt` EM drops to 0.48 because answer robustness is affected by accumulated distractor/near-miss state.

## Main Findings

P6.3 strengthens the core thesis:

```text
Repeated same-slot updates expose a reliability tradeoff between recoverability, stale burden, and compact state maintenance.
```

The hard split reveals three distinct behaviors:

1. **Oracle constrained slot CRUD**
   - perfect state under slot-direct;
   - no stale same-slot burden;
   - still shows answer-layer stress at hard k=16 under slot-prompt.

2. **Append-only / heuristic methods**
   - perfect slot-direct state accuracy;
   - rapidly growing stale same-slot burden;
   - catastrophic slot-prompt collapse at high k.

3. **learned long25**
   - much better slot-prompt robustness than append-only methods;
   - not fully compact on hard distractors;
   - still vulnerable to accumulated stale/distractor burden at k=16.

## Go / No-Go Decision

Decision: **Go.**

The P6.3 hard benchmark is more useful than the P6.2 pilot because it:

- preserves deterministic oracle ceiling;
- makes append-only weakness much clearer;
- differentiates learned long25 from raw/heuristic baselines;
- surfaces a harder second-order issue: even compact oracle memory can face answer-layer stress under many same-name distractors.

## Recommended P6.4 Direction

P6.4 should not immediately train another SFT checkpoint. The next best step is:

1. Create a compact table/plot script for P6.2/P6.3 curves.
2. Add `k=32` only if a stronger high-frequency tail is needed.
3. Run a lightweight Mem0 isolated-env feasibility spike once NAS space is stable.
4. After external-baseline feasibility is clear, decide whether to train a targeted repair for long25 hard-update failures.

For the paper narrative, the current strongest contribution is now:

```text
A controlled repeated-update stress benchmark showing that conventional append-only memory can preserve recoverability under oracle slot lookup while becoming unusable under slot-conditioned answering due to stale same-slot burden.
```
