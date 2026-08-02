# MemUpdateBench

MemUpdateBench is a focused diagnostic toolkit for repeated same-slot memory updates. It isolates the P6.x direction from the original G-MSRA research repo: how external memory systems behave when the same `(entity, attribute)` slot is updated many times.

The current project direction is reviewer-risk driven. Strict reviews in `docs/critical_review.md` and `docs/critical_review_v3.md` narrowed the target to a controlled benchmark plus empirical analysis paper: final-state reliability, stale same-slot burden, memory compactness, and answer robustness should be reported separately.

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

## vNext benchmark engineering

vNext extends the legacy `(entity, attribute)` diagnostic into a strict four-part memory-object identity:

```text
(namespace, entity, attribute, subkey)
```

`object_type` is classification metadata and is excluded from identity. Phase 0 and the bounded Families A–D Pilot release are both `FINAL_APPROVED`. The Pilot contains 480 semantic cores and 1,440 rendered tasks across repeated same-slot updates, interleaved multi-slot updates, entity/attribute grounding with explicit abstention, and NOOP/write discipline.

The authenticated built-in execution matrix is bound to revision `ca47df7a6401fabfc25dd4d2151a392439e6c379`; its human-rebound root index is `d9ef2cebc74a5445863de0ef047c9528cc01eab89354ca93b51917a5f2d0322b`. It covers reference, raw append, exact CRUD, and verified MiniLM heuristic CRUD under `normal_topk` and `latest_per_object`. These are deterministic `slot_direct` engineering diagnostics, not prompted answer-model or external-system results.

Key vNext entry points:

```text
docs/specs/memupdatebench_vnext_benchmark_design.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md
data/vnext/pilot/README.md
results/vnext/pilot/README.md
scripts/vnext_generate_pilot.py
scripts/vnext_validate_pilot.py
scripts/vnext_run_pilot.py
scripts/vnext_score_pilot.py
scripts/vnext_summarize_pilot.py
scripts/vnext_build_mechanism_slice.py
```

Exact release hashes, formal status counts, diagnostic metrics, scorer-control evidence, trace review, and limitations are recorded in `WORKFLOW.md`.

## Reproduce P6.3 summary

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
```

## Current analysis status

The P8.0 analysis layer adds:

1. stale dose-response analysis;
2. real-context order and annotation probes;
3. synthetic same-slot mechanism probes;
4. expanded latest-per-slot all-k validation;
5. Llama3.1-8B multi-model replication;
6. Long25 checkpoint/training-provenance audit.

Core release checklist:

```text
paper/p80_release_candidate_checklist.md
```

## Generate paper assets

P6.3 tradeoff figures:

```bash
python scripts/package_update_frequency_paper_assets.py \
  --summary_json results/update_frequency_p63_summary/update_frequency_summary.json \
  --paper_dir paper
```

P8.0 tables and figures:

```bash
PYTHONPATH=. python scripts/package_p80_paper_tables.py
PYTHONPATH=. python scripts/package_p80_figures.py
```

Generated P8 artifacts include:

```text
paper/p80_paper_tables.md
paper/figures/p80_figure_manifest.json
paper/manuscript_sections/p80_results_section_draft.md
```

## Smoke test

```bash
PYTHONPATH=. python -m py_compile \
  scripts/prepare_data.py \
  scripts/eval_evomemory.py \
  scripts/analyze_ood_errors.py \
  scripts/analyze_action_pathology.py \
  scripts/summarize_update_frequency.py \
  scripts/summarize_prompt_robustness.py \
  scripts/generate_constrained_sft.py \
  scripts/train_constrained_sft.py \
  scripts/smoke_test.py \
  scripts/package_update_frequency_paper_assets.py \
  scripts/build_evidence_manifest.py \
  scripts/package_p80_paper_tables.py \
  scripts/package_p80_figures.py
PYTHONPATH=. python scripts/smoke_test.py
```

## Learned baseline checkpoint

The migrated learned checkpoint path should be:

```text
checkpoints/long25/best
```

On the cluster, move it from the archived G-MSRA path before deleting old remote files.
