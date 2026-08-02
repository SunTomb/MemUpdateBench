# vNext Pilot result artifacts

This directory is reserved for manifest-backed vNext Pilot summaries and audit cases.

## Inputs and authentication

`vnext_summarize_pilot.py` consumes canonical vNext JSONL for `MemUpdateTask`, `TaskRunRecord`, and `ScoreRecord`. The task manifest authenticates task IDs and canonical task hashes. The run manifest authenticates the task-manifest hash, runtime-record file, adapter identity/capabilities, parser provenance, expected task set, and status counts. The score JSONL is authenticated through the run manifest's score artifact reference and is never re-scored by the summary command.

Raw legacy result JSON, untyped metric dictionaries, missing task/run rows, duplicate IDs, and artifact hash mismatches are rejected before any output is published.

## Case policies

- `all` selects the deterministic task-ID order, capped at 256 cases.
- `failures` selects only rows carrying one or more scorer failure flags.
- `stratified` (the default) selects at least one correct and one failing row when each is available in every family/difficulty/method cell. Selection is task-ID deterministic and capped at 256 cases.

A case is a projection of authenticated task, run, score, task-manifest, and run-manifest records. It contains event roles/timeline, gold and predicted actions, snapshots/final state, retrieved entries and IDs, answer output, metric values and support reasons, all failure flags, primary failure, capability declarations, artifact hashes, and source anchors. It does not recompute metrics. When a capability is unavailable, the corresponding trace is `null` rather than an invented empty trace. Private or non-redistributable source text is removed while hashes and anchors remain.

## Reading summaries

`summary.json` contains aggregate counts and metric numerators/denominators. Unsupported metric fields remain null and are described in `capability_coverage.json`; they are excluded according to each support reason rather than treated as negative scores. `failure_breakdown.json` counts scorer flags and primary failures. `cases.jsonl` is the bounded audit bundle and `artifact_index.json` authenticates every published output and input artifact.

Oracle/reference smoke runs and corrupted/smoke controls are retained for diagnostics but are not presented as leaderboard rows when the run manifest marks them ineligible. Latest-per-object retrieval is a trace-level projection (a retrieval rewrite), while answer-level interventions operate on answer context/output; these are not interchangeable metrics.

## Authenticated built-in release

The bounded Families A–D Pilot release is `FINAL_APPROVED`. Its formal built-in matrix is bound to clean revision `ca47df7a6401fabfc25dd4d2151a392439e6c379`. Cluster artifacts are stored in the revision-qualified directory:

```text
/NAS/yesh/MemUpdateBench/results/vnext/pilot_ca47df7_evidence_bound
```

It contains an evidence-bound release snapshot, reference, raw append, exact CRUD, and verified MiniLM heuristic CRUD under both `normal_topk` and `latest_per_object`, plus score bundles, summaries/cases, eight retained corrupted-control checks, and the deterministic mechanism-smoke slice. The human-rebound task manifest authenticates task/generation/validation/audit bytes, and every formal cell has 1,440 unique runtime rows and 1,440 score rows with authenticated manifests and summary indices. Human reviewer Ye Shenghao supplied 96/96 release-ready decisions; provenance rebinding preserved every score row and aggregate metric. The final root index SHA-256 is `d9ef2cebc74a5445863de0ef047c9528cc01eab89354ca93b51917a5f2d0322b`.

Reference completed all 1,440 tasks. Each other built-in completed 1,080 tasks and declared the 360 Family C tasks `NOT_SUPPORTED` at `answer/multi_object_answer`; these are explicit capability exclusions, not failures. No parser-driven partial row or encoder-related unsupported row remains.

These are deterministic `slot_direct` engineering diagnostics. They do not establish prompted-answer robustness, external-system validity, or learned semantic-resolution performance. Exact metrics, trace findings, commands, provenance, and limitations are recorded in `WORKFLOW.md`.
