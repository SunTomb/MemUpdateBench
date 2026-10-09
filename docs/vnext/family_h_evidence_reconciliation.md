# Family-H Evidence Reconciliation

## Purpose

This package reconciles existing Family-H evidence without rerunning a manager, Qdrant, answer model, provider, GPU or SSH job. It is a new diagnostic sibling; it does not rewrite or promote any source, state, answer or frozen release root.

The package joins four existing artifacts:

- audited source release: `data/vnext/family_h_independent/v1_published/release_manifest.json` and its authenticated `release_index.json`;
- external-manager state/retrieval receipt: `results/vnext/family_h_state_completion_receipt_20260925_v2.json`;
- correction audit: `results/vnext/family_h_state_receipt_correction_20260927_v1.json`;
- inherited retrieved-context Qwen answer receipt: `results/vnext/family_h_answer_results_receipt_20260928_v6.json`.

## Reconciliation rules

The validator requires the source release to remain `FINAL_APPROVED` with `scientific_release_allowed=false`, three semantic cores/tasks/sources, 27 events and 33 decisions, with no source model backend run. It independently checks the state/retrieval receipt for 27/27 state steps and source links, 3/3 final states and typed retrievals, stale count zero, cleanup success and zero new model/provider execution.

The answer receipt is retained as inherited evidence, not requalified runtime evidence. Its three-task retrieved-context Qwen replay is checked for three model loads, three generations, three format-valid outputs, three typed-object matches, zero retries/provider calls and no raw generation text. It remains `scientific_release_allowed=false`.

The correction audit is binding: `STALE_COPIED_PROVENANCE_HASH` and `UNRESOLVED_ARCHIVE_IDENTITY_MISMATCH` remain unresolved discrepancies. The package therefore reports `DIAGNOSTIC_ONLY`, not `READY`, and never converts the discrepancy into an accuracy zero.

Evidence classes remain separate:

```text
source          audited task release
state_retrieval external-manager state/retrieval
answer          inherited retrieved-context Qwen replay
runtime         inherited receipt, not requalified by this package
```

## Output

The no-replace diagnostic package was generated at:

```text
results/vnext/family_h_evidence_reconciliation_20261010_v1
```

Its status is `DIAGNOSTIC_ONLY`; `scientific_release_allowed=false`; and its execution counters are all zero because this package performs no new execution. The output contains only a compact reconciliation and artifact index, not raw source captures, prompts, answer text or credentials.

Package index SHA-256:

```text
7bb87ee08d1d45cd1659bcad33825ded95577468415818df3643fa2af44c971a
```

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B -m pytest -q -p no:cacheprovider tests/vnext/test_reconcile_family_h_evidence.py
python -B scripts/vnext_reconcile_family_h_evidence.py \n  --output-root results/vnext/family_h_evidence_reconciliation_20261010_v1
```

The CLI uses exclusive directory creation. Re-running with the same output path fails rather than replacing the package. The package is evidence reconciliation, not a human audit, cryptographic signature or scientific approval.
