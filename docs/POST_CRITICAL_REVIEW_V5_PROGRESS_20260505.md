# Post-Critical-Review-v5 Progress Report

## Purpose

本文档总结在接受 `docs/critical_review_v5.md` 之后，项目围绕 v5 指出的核心叙事问题又推进了什么、最后一个实验 cell 是否补齐、以及现在是否应该彻底冻结实验并进入论文 production。

v5 的关键判断是：

> 项目的瓶颈已经不再是“缺什么实验”，而是“论文怎么写”。

但 v5 同时指出一个低成本、高价值的缺口：为了完成三模型 ceiling-recovery story，需要补 Mistral 的 constrained CRUD k=16 zero-stale baseline。

## What changed after `critical_review_v5.md`

### 1. The final Mistral zero-stale cell is complete

Key artifacts:

```text
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
paper/p80_third_model_mistral_note.md
paper/p80_multimodel_stale_susceptibility_note.md
```

Run setup:

```text
node: Tang-1-Wu
model: /NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1
data: data/evomemory_update_frequency_hard_k16_p63_dev.json
mode: constrained_slot_crud
answer modes: slot_prompt, slot_direct
```

Results:

| Condition | EM | F1 | value EM | answer value present | state | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| constrained_slot_crud slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

This exactly matches the existing Mistral latest_per_slot result:

```text
Mistral latest_per_slot EM/F1 = 0.720 / 0.735
Mistral constrained zero-stale EM/F1 = 0.720 / 0.735
```

### 2. The multi-model story has been corrected to ceiling recovery

Before v5, the paper-facing language said:

```text
stale susceptibility is shared, but mitigation magnitude is model-dependent
```

After P8.2, the stronger and cleaner statement is:

```text
retrieval-time stale filtering recovers each model to approximately its own zero-stale ceiling.
```

Current comparison:

| Model | normal raw_add EM | latest_per_slot EM | zero-stale constrained EM | Gap |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 0.110 | 0.690 | 0.700 | -0.010 |
| Llama3.1-8B | 0.060 | 0.290 | 0.270 | +0.020 |
| Mistral-7B | 0.080 | 0.720 | 0.720 | 0.000 |

Interpretation:

- stale-contaminated normal top-k retrieval collapses for all three models;
- latest_per_slot removes stale retrieved entries;
- after stale filtering, each model reaches its own clean-memory slot-prompt ceiling;
- the absolute EM differences are therefore mostly model-specific slot-prompt / instruction-following ceilings, not evidence that stale filtering solves fundamentally different fractions of stale-specific loss.

This is stronger than the previous model-dependent mitigation framing.

### 3. Paper-facing artifacts have been refreshed

Updated files include:

```text
paper/p80_multimodel_stale_susceptibility_note.md
paper/p80_third_model_mistral_note.md
paper/p80_canonical_main_number_ledger.md
paper/p80_claim_evidence_matrix.md
paper/p80_paper_tables.md
paper/p80_release_candidate_checklist.md
paper/p80_remaining_work_summary.md
paper/manuscript_draft.md
paper/manuscript_sections/p80_results_section_draft.md
WORKFLOW.md
```

The table pack now includes:

```text
Mistral-7B constrained zero-stale control
Ceiling-recovery comparison
```

### 4. Manuscript narrative has been partially rewritten

`paper/manuscript_draft.md` and `paper/manuscript_sections/p80_results_section_draft.md` now foreground ceiling recovery:

- abstract updated to state the three-model ceiling pattern;
- multi-model section updated from weaker Llama recovery to ceiling recovery;
- discussion updated to say stale same-slot context is the dominant stale-specific answer-layer obstacle;
- limitations updated to avoid overclaiming external SDK comparison while acknowledging the transparent extract-then-store diagnostic pipeline.

This is not yet final LaTeX, but the manuscript is no longer telling the wrong multi-model story.

## Validation

Final local validation after P8.2:

```text
PYTHONPATH=. python -m py_compile ...
PYTHONPATH=. python scripts/package_p80_paper_tables.py
PYTHONPATH=. python scripts/smoke_test.py
SMOKE TEST: 26/26 passed
PYTHONPATH=. python scripts/build_evidence_manifest.py --result_root results --output_dir results/p80_evidence_manifest
num_rows: 402
PYTHONPATH=. python scripts/package_p80_figures.py
figure manifest count: 8
```

## Current status after v5

The evidence package now contains:

- stale filter intervention;
- three-model ceiling recovery;
- dose-response;
- fixed-k heuristic threshold sweep;
- synthetic conflict / order / label mechanism probe;
- same_as_current failure classification;
- Lost-in-the-Middle gold-position probe;
- attribute-level failure analysis;
- transparent external extract-then-store diagnostic pipeline;
- refreshed evidence manifest, ledger, table pack, checklist, and workflow.

The remaining bottleneck is writing, not evidence.

## Recommended claim level now

The strongest defensible framing is:

> An empirical analysis of stale context contamination in memory-augmented LLM answering, supported by a controlled repeated same-slot update benchmark.

The headline finding should be:

> Retrieval-time stale filtering recovers Qwen, Llama, and Mistral to approximately each model's own zero-stale ceiling, showing that stale same-slot context is the dominant stale-specific answer-layer obstacle.

The paper should still avoid claiming:

- broad comprehensive memory benchmark;
- fair external SDK leaderboard;
- deployable memory method;
- repair training result.

## Remaining work

Only paper/release production remains:

1. final LaTeX conversion;
2. citation cleanup;
3. figure/table placement;
4. final numerical consistency pass using `paper/p80_canonical_main_number_ledger.md`;
5. README/release reproduction polish;
6. result-artifact packaging decision.

## What not to reopen

Unless explicitly requested by the advisor, do not reopen:

- Mistral test confirmation;
- Llama/Mistral LitM probes;
- Llama/Mistral synthetic probes;
- new external SDK baseline engineering;
- new splits;
- repair training;
- all-k multi-model sweeps;
- additional prompt variants.

## Bottom line

After v5, the last missing experiment cell is complete and the central multi-model story has been corrected.

The project should now freeze experiments completely and move into final paper production.
