# PROJECT_WORKFLOW6.1.0 — Update-Frequency Stress Test Pilot

## Context

`LITERATURE_REVIEW_P601_20260501.md` concluded that the P6.0.1 repeated-update direction is promising but only if a real decay signal exists. It also recommended two immediate stop-loss checks before investing further:

1. Use existing long-horizon results to see whether a rough update-count decay trend already appears.
2. Check whether a genuine external baseline such as Mem0/MemGPT is immediately available.

This workflow records those checks and the first controlled update-frequency pilot.

## Literature-Critique Constraints

The P6.0.1 critique says this direction is viable only if:

- update count is an isolated independent variable;
- deterministic oracle ceiling is near 1.0;
- k=1 to k=16 shows a meaningful effect size, ideally >= 10 percentage points;
- failure analysis goes beyond a shallow error log;
- at least one credible external baseline is eventually included;
- the work is positioned against DST as memory-state management, not slot extraction.

## Stop-Loss Check 1: Existing Long-Horizon Data

I grouped existing parser-fix long-horizon results by actual target-slot update count.

Combined result across existing long-horizon dev/test/seed101/seed202:

```text
k=1: acc=0.0000, n=1
k=3: acc=0.2500, n=4
k=4: acc=0.8750, n=8
k=5: acc=0.9474, n=38
k=6: acc=1.0000, n=188
k=7: acc=1.0000, n=1
```

This does **not** show update-frequency decay. The existing mixed long-horizon split is not suitable for this thesis because update count is confounded with generator distribution and sample imbalance.

Conclusion: a controlled generator is necessary.

## Stop-Loss Check 2: External Baseline Feasibility

Checked imports locally and on Tang-2-Wu:

```text
mem0   -> not installed
mem0ai -> not installed
memgpt -> not installed
```

I did not install new dependencies automatically. This remains a major risk: the paper will need at least one genuine external memory-system baseline, or the claim must stay modest.

## Implementation: update_frequency EvoMemory Variant

Modified:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
```

Added:

```bash
python scripts/prepare_data.py \
  --dataset evomemory \
  --output_dir data \
  --evomemory_variant update_frequency \
  --seed 53 \
  --output_suffix pilot2
```

Generated controlled k splits:

```text
data/evomemory_update_frequency_k1_pilot2_test.json
data/evomemory_update_frequency_k2_pilot2_test.json
data/evomemory_update_frequency_k4_pilot2_test.json
data/evomemory_update_frequency_k8_pilot2_test.json
data/evomemory_update_frequency_k16_pilot2_test.json
```

Also generated combined train/dev/test files:

```text
data/evomemory_update_frequency_pilot2_train.json
data/evomemory_update_frequency_pilot2_dev.json
data/evomemory_update_frequency_pilot2_test.json
```

Design:

- k values: `1, 2, 4, 8, 16`
- 100 examples per k per split
- balanced attributes:
  - location
  - company
  - preference
  - language
- same-attribute single distractor entity
- metadata:
  - `stress_type = update_frequency`
  - `k_updates`
  - `distractor_level`
  - `latest_event_idx`

Parser support added for language initialization:

```text
User says: my mentor Bob's programming language is Python.
```

## Oracle Gate

Deterministic oracle on Tang-2-Wu:

```text
k=1:  state_accuracy=1.0000, state_resolve=1.0000
k=2:  state_accuracy=1.0000, state_resolve=1.0000
k=4:  state_accuracy=1.0000, state_resolve=1.0000
k=8:  state_accuracy=1.0000, state_resolve=1.0000
k=16: state_accuracy=1.0000, state_resolve=1.0000
```

The controlled benchmark passes the oracle ceiling requirement.

## Learned Long25 Curve

Checkpoint:

```text
outputs/constrained_sft_curriculum_long25/best
```

Evaluation command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_direct \
  --no_qlora \
  --lora_checkpoint outputs/constrained_sft_curriculum_long25/best \
  --data_file data/evomemory_update_frequency_k<k>_pilot2_test.json \
  --output_dir results/update_frequency_pilot2/long25_k<k>
```

Results:

| k updates | state_accuracy | state_resolve | EM | F1 |
|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.8500 | 1.0000 | 0.8500 | 0.8500 |
| 4 | 0.9400 | 1.0000 | 0.9400 | 0.9400 |
| 8 | 0.8800 | 1.0000 | 0.8800 | 0.8800 |
| 16 | 0.8700 | 1.0000 | 0.8700 | 0.8700 |

Effect size:

```text
k=1 -> k=16 drop = 13 percentage points
```

This passes the P6.0.1 minimum effect-size gate.

## Error Analysis

Long25 errors by k:

```text
k=1:  correct=100
k=2:  correct=85, wrong_value=15
k=4:  correct=94, wrong_value=6
k=8:  correct=88, wrong_value=12
k=16: correct=87, wrong_value=13
```

All errors are `wrong_value`; `state_resolve` remains 1.0.

Attribute concentration:

```text
k=2:  company wrong_value=15 / 25 company cases
k=4:  company wrong_value=6  / 25 company cases
k=8:  company wrong_value=12 / 25 company cases
k=16: company wrong_value=13 / 25 company cases
```

Location, preference, and language were all at ceiling in this pilot.

Typical failure:

- gold latest company update is late in the episode;
- learned manager stops updating after an earlier company value;
- `stale_value_present_same_slot = True`;
- `gold_value_present_anywhere = False`, meaning the final value was not stored at all.

This is a genuine stale-value / missed-update pattern, not a parser artifact.

## Internal Baseline Curves

I also ran:

```text
raw_add
heuristic_crud
```

Both returned 1.0 across all k under the current `slot_direct` state evaluator:

```text
raw_add:        k=1/2/4/8/16 all 1.0000
heuristic_crud: k=1/2/4/8/16 all 1.0000
```

This is an important evaluation pitfall: because `slot_direct` retrieves the latest same-slot entry, a pure append-only system can look perfect if every update is parsed and appended with slot metadata. This means the current diagnostic setup measures whether the final value is recoverable as latest slot state, but it does not penalize memory bloat or stale entries unless the method overwrites incorrectly.

Implication: future diagnostic comparison must report additional metrics:

- final memory size;
- number of stale same-slot entries;
- update efficiency / write amplification;
- whether stale entries coexist with correct final value;
- retrieval robustness when not using oracle slot lookup.

## Interpretation

The repeated-update direction has a real signal, but the signal is more nuanced than a simple monotonic decay curve.

Findings:

1. The controlled benchmark has perfect oracle ceiling.
2. Long25 shows a >10pt drop from k=1 to k=16.
3. Failures are all `wrong_value` with `state_resolve=1.0`, matching the stale-value thesis.
4. The effect is concentrated in company updates.
5. Append-only and heuristic baselines are artificially favored by slot-direct evaluation, revealing a metric pitfall that must be addressed.

The direction remains worth pursuing, but the next step must strengthen evaluation beyond `slot_direct`.

## Next Steps

1. Add diagnostic metrics for:
   - stale same-slot entry count;
   - write count;
   - final memory size;
   - write amplification per k.
2. Evaluate with a non-oracle answer mode or retrieval mode to test whether stale appended values hurt real retrieval.
3. Add a harder update-frequency variant with:
   - same-name multi-entity distractors;
   - semantically near-miss NOOP events;
   - balanced company cases that avoid overfitting to one attribute.
4. Investigate why company updates fail while language/preference/location stay at ceiling.
5. Only after this, decide whether to train a targeted repair.
