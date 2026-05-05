# P8.0 Remaining Work Summary

## Status summary

The core benchmark + analysis evidence package is now complete for a controlled diagnostic paper after the P8.1 methodological rigor pass and P8.2 Mistral ceiling-control cell. The remaining tasks fall into three categories:

1. true blockers for a public benchmark release;
2. optional external-validity strengthening;
3. paper-writing and formatting work.

## Completed core P8.0 items

### Evidence and provenance

- Evidence manifest built and refreshed.
- Current manifest: `results/p80_evidence_manifest/evidence_manifest.{json,csv,md}`.
- Current local count after P8/P8.1/P8.2 outputs: 402 result rows.

### Mechanism analysis

- Dose-response analysis complete.
- P8.1 fixed-k heuristic threshold sweep complete: at k=16, stale burden rises from 4.43 to 13.04 while slot_prompt EM falls from 0.220 to 0.100 and slot_direct stays 1.000.
- Real-context order/annotation probe complete.
- Synthetic same-slot probe complete.
- P8.1 expanded synthetic diagnostic subset complete at 64 examples per selected cell; reverse-chronological conflict collapse and label repair both replicate.
- P8.1 same_as_current EM-failure inspection complete; answer-value-present remains 1.000 in the expanded same_as_current cells, so these failures are treated as non-exact answer-surface degradation rather than stale-value selection.
- Lost-in-the-Middle gold-position probe complete: gold-at-end EM/F1 0.630/0.654, middle 0.090/0.183, beginning 0.010/0.073.
- Attribute failure analysis complete, including company/language gold-retrieved wrong-answer case study.

### Mitigation/anomaly checks

- P7.0 latest_per_slot all-k complete on P6.3 dev.
- P8.0 expanded latest_per_slot all-k complete.
- k=8 near-perfect anomaly attenuates on expanded dev.

### Multi-model replication

- Llama3.1-8B dev replication complete.
- Llama3.1-8B test-split confirmation is also complete and preserves the same qualitative conclusion.
- P8.1 Llama constrained CRUD zero-stale control complete: slot_prompt EM/F1 0.270/0.321 despite state accuracy 1.000 and stale burden 0.000, confirming that Llama's weak latest_per_slot recovery includes a model-specific answer-layer weakness.
- Mistral-7B-Instruct third-model dev replication is complete and confirms stale collapse plus strong latest_per_slot recovery.
- P8.2 Mistral constrained CRUD zero-stale control complete: slot_prompt EM/F1 0.720/0.735, exactly matching Mistral latest_per_slot EM/F1 0.720/0.735; this closes the three-model ceiling-recovery story.

### Long25 provenance

- P6.3 vs P6.5 checkpoint-family mismatch resolved.
- Safe wording: training/checkpoint-provenance sensitivity, not pure seed sensitivity.

### Manuscript assets

- `paper/manuscript_draft.md` updated with P8 narrative.
- `paper/p80_paper_tables.md` generated.
- `paper/figures/p80_figure_manifest.json` generated.
- `paper/manuscript_sections/p80_results_section_draft.md` created.
- `paper/p80_claim_evidence_matrix.md` created.

## True release blockers

### 1. README-level reproduction polish

The README now points to P8 assets, but a public release still needs a step-by-step path from fresh checkout to core tables. This should include environment assumptions and expected runtime.

### 2. Result artifact size and packaging decision

The repository contains many generated results. Before public release, decide whether to:

- include all `results/` artifacts;
- include only summaries and scripts;
- publish full results separately.

### 3. Final canonical result ledger

The P8 evidence manifest is broad. A smaller final ledger should identify only main-paper numbers and their exact source paths.

### 4. License/citation cleanup

The related work draft exists, but formal citations and release license checks remain to be done.

## Optional external-validity strengthening

### 1. Fair external baseline

Current status:

- Mem0 Qwen2.5-VL dev20 is runnable but unfair/qualitative.
- Qwen2.5-7B text server is queryable.
- CPU MiniLM embedding works.
- Mem0 structured extraction fails before dev3 completes.
- A simple project-local extract-then-store pipeline is now complete: append parsed-only reproduces stale collapse at EM/F1 0.140/0.177, while slot-update parsed-only reaches EM/F1 0.910/0.926 with zero stale same-slot burden.

Recommendation:

- Use the simple extract-then-store row as the transparent external-pipeline diagnostic baseline.
- Do not block the controlled diagnostic paper on Mem0 unless the target venue specifically requires an external SDK row.
- If Mem0 is pursued later, use a standard vLLM/OpenAI server or patch extraction to enforce valid JSON.

### 2. Third answer model

Mistral-7B-Instruct dev replication and constrained zero-stale control are now complete. The one-model-family concern is addressed by Qwen + Llama + Mistral, and the main multi-model story should be framed as ceiling recovery rather than absolute mitigation magnitude. Only Llama currently has matching test-split confirmation; Mistral test confirmation remains optional and unnecessary unless it becomes a headline test-split claim.

### 3. More realistic update types

Implicit, partial, negative, and conditional updates would improve realism but require deterministic oracle semantics. Treat as future work unless the paper target changes.

## Paper-writing work remaining

Completed in the current manuscript draft:

- abstract and introduction now lead with stale same-slot contamination and three-model ceiling recovery;
- main results now separate state recoverability, stale-filtering recovery, multi-axis reporting, and the ceiling-recovery table;
- mechanism analysis has been compressed around context presentation, synthetic probes, Lost-in-the-Middle, dose response, and residual clean-context limits;
- limitations now avoid claiming a fair external SDK leaderboard while retaining the transparent extract-then-store diagnostic pipeline.

Still remaining for paper production:

1. Replace placeholder citations in `paper/references_todo.bib` with canonical BibTeX entries.
2. Convert the generic article draft to the eventual venue template.
3. Do final advisor-facing polishing of figure/table placement and prose length.
4. Polish README/release instructions and decide result-artifact packaging.

## Recommended next action order

1. Freeze the experimental scope around the completed P8/P8.1/P8.2 evidence package.
2. Replace `paper/references_todo.bib` placeholders with canonical citations and then port `paper/manuscript_production_draft.tex` to the target venue template.
3. Use the canonical main-number ledger and claim-evidence matrix for the final advisor-facing numerical consistency pass.
4. Polish README/release instructions and decide result-artifact packaging.
5. Only revisit external baseline engineering if the advisor or target venue requires an external SDK row.
