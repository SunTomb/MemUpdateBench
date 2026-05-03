# Figure and Table Placement Plan

## Main body

### Main Figure 1: update-frequency tradeoff

File:

```text
paper/p63_update_frequency_tradeoff.pdf
```

Use in main results section as Figure 1.

Purpose:

- Shows the full four-metric tradeoff across k.
- Establishes that slot-direct recoverability and slot-prompt robustness diverge.
- Shows stale burden and final memory size as explanatory axes.

Caption source:

```text
paper/p63_update_frequency_latex_snippets.tex
```

### Main Table 1: k=16 endpoint

Source:

```text
paper/p63_update_frequency_latex_snippets.tex
```

Use table label:

```text
tab:p63-k16-tradeoff
```

Purpose:

- Gives the clean high-frequency endpoint.
- Supports the central thesis in one compact table.
- Makes Long25 framing clear: compact but not fully reliable.

## Appendix figures

### Appendix Figure A: direct-vs-prompt gap

File:

```text
paper/p63_gap_slot_direct_vs_prompt.pdf
```

Purpose:

- Shows how oracle recoverability can hide practical answer degradation.
- Useful for reviewer concern that `slot_direct` is unrealistic.

### Appendix Figure B: stale burden vs slot-prompt EM at k=16

File:

```text
paper/p63_stale_vs_prompt_em_k16.pdf
```

Purpose:

- Visualizes stale burden as an explanatory variable for answer collapse.
- Useful in discussion/error analysis.

### Appendix Figure C: memory size vs slot-prompt EM at k=16

File:

```text
paper/p63_memory_size_vs_prompt_em_k16.pdf
```

Purpose:

- Shows compactness/robustness frontier.
- Helps position Long25 as compact but not fully reliable.

## Appendix tables

Use appendix k-sweep tables from:

```text
paper/p63_update_frequency_latex_snippets.tex
```

Tables:

- `tab:p63-slot-direct-sweep`
- `tab:p63-slot-prompt-sweep`
- `tab:p63-stale-sweep`
- `tab:p63-memory-size-sweep`

## Method-definition table

File:

```text
paper/p63_method_definition_table.md
```

Recommendation:

Use in benchmark/setup section if reviewers may confuse diagnostic roles with deployable method claims. Otherwise keep as appendix or author-facing reference.

## Current placement decision

- Main body: one 2x2 figure + one k=16 table.
- Appendix: derived figures + k-sweep tables.
- Keep the main body lean; use appendix to defend reviewer questions.
- Use `paper/p63_error_analysis_k16.md` and `paper/remote_verification_log.md` as author-facing support or appendix-source material rather than adding extra main-text figures.
- Keep remote rerun numbers out of the main table; mention them only as close confirmation in prose or reproducibility notes.
