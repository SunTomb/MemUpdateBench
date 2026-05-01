# MemUpdateBench

MemUpdateBench is a focused benchmark and diagnostic toolkit for repeated same-slot memory updates. It isolates the P6.x direction from the original G-MSRA research repo: how external memory systems behave when the same `(entity, attribute)` slot is updated many times.

## Core claim

Append-only memory can preserve final-state recoverability under oracle slot lookup, but repeated same-slot updates create stale memory burden. Under slot-conditioned answering, that stale burden causes answer collapse. Compact learned managers reduce stale burden, but can miss final updates or remain incompletely compact.

## Main benchmark

The main split family is `update_frequency_hard` with k values `1, 2, 4, 8, 16`, same-name multi-entity distractors, and semantic near-miss NOOPs.

Important files:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/summarize_update_frequency.py
scripts/smoke_test.py
scripts/generate_constrained_sft.py
scripts/train_constrained_sft.py
```

Core package:

```text
mub/
```

## Reproduce P6.3 summary

```bash
python scripts/summarize_update_frequency.py \n  --result_root results/update_frequency_p63 \n  --output_dir results/update_frequency_p63_summary
```

## Smoke test

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py
python scripts/smoke_test.py
```

## Learned baseline checkpoint

The migrated learned checkpoint path should be:

```text
checkpoints/long25/best
```

On the cluster, move it from the archived G-MSRA path before deleting old remote files.
