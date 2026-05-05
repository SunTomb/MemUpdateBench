# Post-Critical-Review-v5 Production Progress Report

## Purpose

本文档接续 `docs/POST_CRITICAL_REVIEW_V5_PROGRESS_20260505.md`，总结在接受 `docs/critical_review_v5.md` 之后，项目不仅补齐了 v5 指出的最后一个 evidence cell，也进一步完成了论文 production 初稿。

v5 的核心判断是：

> 项目的瓶颈已经不再是“缺什么实验”，而是“论文怎么写”。

前一份 v5 progress report 已经记录了 P8.2 Mistral zero-stale cell 与 ceiling-recovery story 的完成情况。本文档记录后续 production 层面的推进：manuscript narrative 是否已经改对、LaTeX 初稿是否已经生成、以及现在还剩哪些真正的 paper/release 工作。

## What changed after `critical_review_v5.md`

### 1. The final evidence gap is closed

v5 指出的唯一低成本、高价值实验缺口已经补齐：

```text
Mistral-7B-Instruct constrained_slot_crud k=16 zero-stale baseline
```

Key artifacts:

```text
scripts/run_p82_mistral_constrained_zero_stale_sui3.sh
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_prompt_k16_dev/evomemory_results.json
results/p82_mistral_constrained_zero_stale/constrained_slot_crud_slot_direct_k16_dev/evomemory_results.json
paper/p80_third_model_mistral_note.md
paper/p80_multimodel_stale_susceptibility_note.md
```

Results:

| Condition | EM | F1 | value EM | answer value present | state | stale same-slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained_slot_crud slot_prompt | 0.720 | 0.735 | 0.750 | 0.750 | 1.000 | 0.000 |
| constrained_slot_crud slot_direct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

This exactly matches the existing Mistral latest_per_slot row:

```text
Mistral latest_per_slot EM/F1 = 0.720 / 0.735
Mistral constrained zero-stale EM/F1 = 0.720 / 0.735
```

This closes the three-model ceiling-recovery comparison.

### 2. The central story is now ceiling recovery, not model-dependent mitigation magnitude

Before v5, the multi-model language was weaker:

```text
stale susceptibility is shared, but mitigation magnitude is model-dependent
```

After P8.2, the cleaner and stronger framing is:

```text
retrieval-time stale filtering recovers each tested model to approximately its own zero-stale ceiling.
```

Current headline comparison:

| Model | normal raw_add EM | latest_per_slot EM | zero-stale constrained EM | Gap |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | 0.110 | 0.690 | 0.700 | -0.010 |
| Llama3.1-8B | 0.060 | 0.290 | 0.270 | +0.020 |
| Mistral-7B | 0.080 | 0.720 | 0.720 | 0.000 |

Interpretation:

- stale-contaminated normal top-k retrieval collapses across Qwen, Llama, and Mistral;
- latest_per_slot removes stale retrieved entries;
- after stale filtering, each model reaches approximately its own clean-memory slot-prompt ceiling;
- absolute EM differences are mostly model-specific slot-prompt / instruction-following ceilings, not evidence that stale filtering has fundamentally different stale-specific effectiveness across models.

This is now the central paper claim.

### 3. Paper-facing source-of-truth artifacts have been refreshed

Updated or generated artifacts include:

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

The release checklist and remaining-work summary now treat experiments as frozen and paper production as the active bottleneck.

### 4. The manuscript narrative has been rewritten around the correct story

`paper/manuscript_draft.md` now foregrounds the ceiling-recovery story rather than method ranking or model-dependent mitigation magnitude.

Main manuscript changes:

- abstract now states the three-model ceiling-recovery pattern;
- introduction starts from stale same-slot contamination rather than a broad benchmark claim;
- main results separate:
  - state recoverability vs answer robustness;
  - latest-per-slot as a diagnostic stale-filtering intervention;
  - multi-axis reporting rather than single method ranking;
  - three-model ceiling recovery;
- mechanism section is compressed around:
  - real-context order / annotation sensitivity;
  - synthetic conflict × order × label probes;
  - Lost-in-the-Middle gold-position probe;
  - stale dose response;
  - residual clean-context answer-layer limits;
- limitations avoid claiming a fair external SDK leaderboard while retaining the transparent extract-then-store diagnostic pipeline.

The manuscript is no longer telling the wrong multi-model story.

### 5. A compilable LaTeX production draft now exists

A new production-stage LaTeX draft has been created:

```text
paper/manuscript_production_draft.tex
paper/manuscript_production_draft.pdf
paper/references_todo.bib
```

Current status:

```text
pdflatex + bibtex + pdflatex + pdflatex: success
PDF length: 9 pages
```

The LaTeX draft includes:

- title and abstract;
- Introduction;
- Benchmark;
- Methods and Evaluation;
- Main Results;
- Mechanism and Error Analysis;
- Discussion;
- Limitations;
- Related Work Positioning;
- Reproducibility;
- main P6.3 update-frequency figure;
- k=16 tradeoff table;
- three-model ceiling-recovery table;
- real-context mechanism table;
- synthetic same-slot figure;
- stale dose-response figure;
- bibliography scaffold.

Important caveat:

```text
paper/references_todo.bib
```

currently contains placeholder citation entries. These are intentionally scaffolding only and must be replaced with canonical BibTeX entries during formal citation cleanup.

## Validation

Final local validation after the manuscript production pass:

```text
PYTHONPATH=. python scripts/package_p80_paper_tables.py
output: paper/p80_paper_tables.md

PYTHONPATH=. python scripts/package_p80_figures.py
figure manifest count: 8

PYTHONPATH=. python -m py_compile ...
passed

PYTHONPATH=. python scripts/smoke_test.py
SMOKE TEST: 26/26 passed

PYTHONPATH=. python scripts/build_evidence_manifest.py --result_root results --output_dir results/p80_evidence_manifest
num_rows: 402

pdflatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/manuscript_production_draft.tex
bibtex paper/manuscript_production_draft
pdflatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/manuscript_production_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/manuscript_production_draft.tex
Output: paper/manuscript_production_draft.pdf
```

Known non-scientific caveats from LaTeX production:

- citations are placeholder entries and need formal replacement;
- current draft uses generic `article` format, not a venue template;
- MiKTeX reports a local update warning unrelated to manuscript correctness.

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
- refreshed evidence manifest, ledger, table pack, checklist, and workflow;
- a compilable LaTeX production draft.

The remaining bottleneck is no longer evidence collection and no longer basic manuscript structure. It is final paper polishing.

## Recommended claim level now

The strongest defensible framing remains:

> An empirical analysis of stale context contamination in memory-augmented LLM answering, supported by a controlled repeated same-slot update benchmark.

The headline finding should be:

> Retrieval-time stale filtering recovers Qwen, Llama, and Mistral to approximately each model's own zero-stale ceiling, showing that stale same-slot context is the dominant stale-specific answer-layer obstacle.

The paper should still avoid claiming:

- a broad comprehensive memory benchmark;
- a fair external SDK leaderboard;
- a deployable memory method;
- a repair training result;
- general coverage of implicit, negative, conditional, or partially specified updates.

## Remaining work

Only final paper/release production remains:

1. replace `paper/references_todo.bib` placeholders with canonical BibTeX entries;
2. convert `paper/manuscript_production_draft.tex` from generic `article` format to the eventual venue template;
3. do advisor-facing prose polish and length control;
4. tune final figure/table placement after venue-template conversion;
5. run one final numerical consistency pass using `paper/p80_canonical_main_number_ledger.md`;
6. polish README/release reproduction instructions;
7. decide result-artifact packaging strategy.

## What not to reopen

Unless explicitly requested by the advisor or target venue, do not reopen:

- Mistral test confirmation;
- Llama/Mistral LitM probes;
- Llama/Mistral synthetic probes;
- new external SDK baseline engineering;
- new splits;
- repair training;
- all-k multi-model sweeps;
- additional prompt variants;
- k=32 main-paper stress testing.

## Bottom line

After `critical_review_v5.md`, the project has completed both the final missing evidence cell and the first real paper-production pass.

The current state is:

```text
Experiments: frozen.
Evidence chain: refreshed and validated.
Central story: ceiling recovery.
Manuscript: rewritten around the correct story.
LaTeX production draft: compilable, 9 pages.
Remaining work: citation cleanup, venue-template conversion, final polishing, release packaging.
```

The project should now stay in paper production mode unless the advisor explicitly asks for a new experiment.
