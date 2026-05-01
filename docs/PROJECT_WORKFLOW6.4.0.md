# PROJECT_WORKFLOW6.4.0

## Motivation

P6.3 confirmed the main paper direction: repeated same-slot update frequency exposes the tradeoff among final-state recoverability, stale same-slot burden, memory compactness, and slot-conditioned answer robustness.

The goal of P6.4 is presentation-first rather than training-first:

1. turn P6.3 results into reusable tables/curves,
2. check whether older P6.2 artifacts can be summarized automatically,
3. perform lightweight external-baseline feasibility checks,
4. decide whether k=32 or immediate repair training is necessary.

## Files created

```text
scripts/summarize_update_frequency.py
results/update_frequency_p63_summary/update_frequency_summary.csv
results/update_frequency_p63_summary/update_frequency_summary.json
results/update_frequency_p63_summary/update_frequency_tables.md
results/update_frequency_p63_summary/slot_direct_state_accuracy.png
results/update_frequency_p63_summary/slot_prompt_em.png
results/update_frequency_p63_summary/stale_same_slot.png
results/update_frequency_p63_summary/final_memory_size.png
results/update_frequency_pilot_summary/update_frequency_summary.csv
results/update_frequency_pilot_summary/update_frequency_summary.json
results/update_frequency_pilot_summary/update_frequency_tables.md
```

## Commands run

### Compile, smoke, and P6.3 summary

```bash
python -m py_compile scripts/summarize_update_frequency.py scripts/analyze_ood_errors.py scripts/eval_evomemory.py scripts/prepare_data.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py
python scripts/smoke_test.py
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
```

Result:

```text
SMOKE TEST: 31/31 passed
Loaded 45 result files
Wrote results/update_frequency_p63_summary/update_frequency_summary.csv
Wrote results/update_frequency_p63_summary/update_frequency_summary.json
Wrote results/update_frequency_p63_summary/update_frequency_tables.md
Wrote results/update_frequency_p63_summary/slot_direct_state_accuracy.png
Wrote results/update_frequency_p63_summary/slot_prompt_em.png
Wrote results/update_frequency_p63_summary/stale_same_slot.png
Wrote results/update_frequency_p63_summary/final_memory_size.png
```

### Optional P6.2/pilot summary attempt

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_pilot results/update_frequency_pilot2 results/update_frequency_pilot2_rich \
  --output_dir results/update_frequency_pilot_summary
```

Result:

```text
Loaded 11 result files
```

The old pilot roots are not named as the full P6.3 matrix, so they are not suitable as the main automated table source. They remain useful as pilot evidence in text.

### External baseline package metadata

```bash
python -m pip index versions mem0ai
python -m pip index versions letta
python -m pip index versions memgpt
```

Observed package versions:

```text
mem0ai latest: 2.0.1
letta latest: 0.6.7
memgpt latest: 0.2.0
```

### Cluster import and space check

```bash
ssh Tang-2-Wu "cd /NAS/yesh/G-MSRA && source /NAS/yesh/G-MSRA/activate.sh && python - <<'PY'
import importlib.util
for name in ['mem0', 'mem0ai', 'letta', 'memgpt']:
    print(name, 'installed' if importlib.util.find_spec(name) else 'not_installed')
PY
df -h /NAS/yesh/G-MSRA"
```

Result:

```text
G-MSRA environment ready ✅
mem0 not_installed
mem0ai not_installed
letta not_installed
memgpt not_installed
Filesystem                  Size  Used Avail Use% Mounted on
192.168.1.200:volume1/work  175T  174T  831G 100% /NAS
```

## Main P6.3 summary tables

Source:

```text
results/update_frequency_p63_summary/update_frequency_tables.md
```

### Slot-direct state accuracy

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| raw_add | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| heuristic_crud | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| long25 | 1.00 | 0.89 | 0.92 | 0.85 | 0.91 |

### Slot-prompt EM/F1

| method | k=1 EM/F1 | k=2 EM/F1 | k=4 EM/F1 | k=8 EM/F1 | k=16 EM/F1 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 0.90 / 0.91 | 0.90 / 0.94 | 0.86 / 0.86 | 0.98 / 0.98 | 0.70 / 0.70 |
| raw_add | 0.90 / 0.91 | 0.78 / 0.79 | 0.24 / 0.27 | 0.06 / 0.10 | 0.07 / 0.10 |
| heuristic_crud | 0.89 / 0.91 | 0.80 / 0.81 | 0.28 / 0.28 | 0.19 / 0.22 | 0.10 / 0.13 |
| long25 | 1.00 / 1.00 | 0.79 / 0.79 | 0.76 / 0.77 | 0.73 / 0.74 | 0.48 / 0.53 |

### Stale same-slot entries

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| raw_add | 0.00 | 1.00 | 3.00 | 6.75 | 14.25 |
| heuristic_crud | 0.00 | 1.00 | 2.55 | 5.23 | 7.44 |
| long25 | 0.00 | 0.23 | 0.46 | 0.74 | 1.13 |

### Final memory size

| method | k=1 | k=2 | k=4 | k=8 | k=16 |
| --- | --- | --- | --- | --- | --- |
| constrained_slot_crud | 2.00 | 4.00 | 7.00 | 12.00 | 23.00 |
| raw_add | 2.00 | 5.00 | 12.00 | 25.00 | 52.00 |
| heuristic_crud | 1.50 | 4.97 | 10.51 | 18.25 | 26.73 |
| long25 | 1.46 | 3.45 | 4.34 | 6.18 | 9.43 |

### k=16 thesis table

| method | k=16 slot_direct state_acc | k=16 slot_prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| raw_add | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| heuristic_crud | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

## Interpretation

The P6.3 artifacts support the paper thesis cleanly:

- Under oracle-like `slot_direct`, append-only and heuristic baselines keep final-state recoverability at 1.00 even at k=16.
- Under realistic `slot_prompt`, stale burden causes severe answer collapse:
  - raw_add k16 EM/F1 = 0.07 / 0.10,
  - heuristic_crud k16 EM/F1 = 0.10 / 0.13.
- The learned long25 checkpoint is much more compact:
  - k16 final memory size = 9.43 vs raw_add 52.00,
  - k16 stale same-slot = 1.13 vs raw_add 14.25.
- But long25 is not fully reliable under hard distractors:
  - k16 slot_direct state accuracy = 0.91,
  - k16 slot_prompt EM/F1 = 0.48 / 0.53.

This is a stronger and cleaner framing than treating learned long25 as simply better or worse. The main result is a tradeoff curve.

## k=32 decision

Do not add k=32 yet.

Reason:

- k=16 already gives a decisive separation between oracle recoverability and slot-conditioned answer robustness.
- raw_add and heuristic already collapse under `slot_prompt` by k=8/k=16.
- Adding k=32 would increase compute and workflow complexity without changing the current paper claim.

Revisit k=32 only if a figure draft looks too short-tailed or an external reviewer/advisor asks for a higher-frequency stress point.

## External baseline feasibility decision

Mem0/Letta/MemGPT are not installed in the current cluster environment. Package metadata is visible from local pip, and /NAS reports 831G available despite showing 100% usage.

Recommendation:

- Do not install external baselines into the main G-MSRA environment.
- If external baseline comparison becomes necessary, create a separate environment such as:

```text
/NAS/yesh/G-MSRA/.venv_mem0_p63
```

- Start with Mem0/mem0ai feasibility only; Letta/MemGPT likely require heavier agent-server integration and are less aligned with a controlled slot-update benchmark.

## Next recommendation

Use the P6.3 summary artifacts to draft the main experimental figure/table first. The next useful phase should be P6.5 paper-figure/text packaging or a small Mem0 isolated-environment feasibility branch, not immediate repair training.

Only start targeted repair training after deciding whether the paper needs an external baseline row and after the current P6.3 tradeoff plot is incorporated into the draft narrative.
