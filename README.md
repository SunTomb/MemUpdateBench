# MemUpdateBench

MemUpdateBench is a focused diagnostic toolkit for repeated same-slot memory updates. It isolates the P6.x direction from the original G-MSRA research repo: how external memory systems behave when the same `(entity, attribute)` slot is updated many times.

The current project direction is reviewer-risk driven. A strict review in `docs/critical_review.md` identified external validity, data diversity, related work, and diagnostic depth as the main blockers. The near-term goal is therefore not more paper packaging, but stronger evidence: real external baselines, deeper answer-layer analysis, same-method-family tradeoff curves, and expanded data.

## Core claim

Append-only memory can preserve final-state recoverability under oracle slot lookup, but repeated same-slot updates create stale memory burden. Under slot-conditioned answering, that stale burden causes answer collapse. Compact learned managers reduce stale burden, but can miss final updates or remain incompletely compact. P6.5 diagnostics further show that even perfect clean state can fail under prompted answering, so state, retrieval context, and answer generation must be analyzed as separate failure layers.

## Main benchmark

The main split family is `update_frequency_hard` with k values `1, 2, 4, 8, 16`, same-name multi-entity distractors, and semantic near-miss NOOPs.

Important files:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/analyze_action_pathology.py
scripts/summarize_update_frequency.py
scripts/summarize_prompt_robustness.py
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
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
```

## Current priority experiments

The default next work should address `docs/critical_review.md`:

1. isolated Mem0 feasibility and, if viable, full k=1/2/4/8/16 external-baseline evaluation;
2. answer-layer diagnostics for Constrained CRUD, including oracle retrieval and top-k/context-length sensitivity;
3. stale-burden interventions for Raw append;
4. heuristic CRUD threshold sweeps for same-method-family tradeoff curves;
5. expanded data with more examples, attributes, and paraphrased explicit update templates;
6. serious related-work positioning against AMemGym, Ledger-QA/UMA, Memory-R1, Mem0, MemGPT/Letta, LoCoMo/LongMemEval, dialogue state tracking, and knowledge editing.

## Generate paper assets

```bash
python scripts/package_update_frequency_paper_assets.py \
  --summary_json results/update_frequency_p63_summary/update_frequency_summary.json \
  --paper_dir paper
```

This writes paper-facing tradeoff figures, derived diagnostic figures, LaTeX snippets, appendix tables, and narrative notes under `paper/`. Treat this as support work after the evidence gaps above, not the default main path.

## Smoke test

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py scripts/package_update_frequency_paper_assets.py
python scripts/smoke_test.py
```

## Learned baseline checkpoint

The migrated learned checkpoint path should be:

```text
checkpoints/long25/best
```

On the cluster, move it from the archived G-MSRA path before deleting old remote files.
