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
