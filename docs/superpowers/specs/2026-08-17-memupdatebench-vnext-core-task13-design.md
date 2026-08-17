# MemUpdateBench vNext Core Task 13 Design

## Status and scope

Task 13 consumes the authenticated Core Task 12 matrix and produces clustered statistics, a generated claim ledger, and verified case exports. It does not regenerate tasks, rerun models, modify `data/vnext/core/v3`, start the final release review, or declare overall Core `FINAL_APPROVED`.

Authoritative Task 12 input:

```text
runtime revision: 9c798df5bb66e853466831ddae8ede3a1f2c01f4
runtime tree: c6584f1d6db3241d6d80d9648d7b349483cf1e07301a212f9f92323d9af4d296
matrix root: /NAS/yesh/MemUpdateBench/results/vnext/core_task12_answer_matrix_9c798df_v1
matrix manifest: 85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2
matrix summary: a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15
integrity audit: bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60
runs / task rows / score rows: 18 / 1,440 / 1,440
```

The incomplete `f88588b` smoke root, fake-offline runs, deterministic `slot_direct` runs, and invalidated Pilot roots are not Task 13 inputs.

## Statistical contract v1

### Independent unit and estimands

The independent unit is exactly `task.metadata.split_key.semantic_core_id`. Every cell contains 80 authenticated task/surface rows grouped into exactly 20 shared semantic cores, with exactly four distinct task IDs per core. Surface tasks are not independent bootstrap units.

For cell `c`, metric `m`, and semantic core `i`, let the four canonical task-level metric values copied from `ScoreRecordV3` be `y[i,j,c,m]`, where `j` is ordered by ascending UTF-8 `task_id`. Define the per-core observation as the Decimal arithmetic mean:

```text
z[i,c,m] = (y[i,0,c,m] + y[i,1,c,m] + y[i,2,c,m] + y[i,3,c,m]) / 4
```

The cell estimand is the equal-core mean over 20 independent observations:

```text
theta[c,m] = sum_i z[i,c,m] / 20
```

Because every core has exactly four task rows, this point estimate equals the 80-task mean, but all statistical language, interval construction, and inference operate on the 20 core means.

For a directed contrast `A - B`, the paired estimand first computes per-core differences on the exact same four task IDs:

```text
d[i,A-B,m] = z[i,A,m] - z[i,B,m]
delta[A-B,m] = sum_i d[i,A-B,m] / 20
```

A contrast is never computed by subtracting endpoints from two independent intervals. Both cells use the same resampled core indices in every replicate.

Core IDs are ordered by ascending UTF-8 bytes. Input file order, mapping iteration order, and run completion order cannot affect outputs.

### Frozen bootstrap parameters

```text
method: nonparametric clustered percentile bootstrap
confidence level: two-sided 95%
replicates: 10,000
draws per replicate: 20 semantic cores with replacement
quantile definition: Hyndman-Fan Type 1 / inverted_cdf
lower endpoint: sorted replicate value 250 (1-based)
upper endpoint: sorted replicate value 9,750 (1-based)
decimal precision: 50
rounding: ROUND_HALF_EVEN
```

All cells, metrics, and paired contrasts reuse one canonical `10,000 x 20` index matrix.

### Deterministic random stream

The seed is the fixed 256-bit value:

```text
9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542
```

For zero-based `(replicate_id, draw_id, rejection_attempt)`, compute:

```text
SHA-256(
  b"MUB-Core-Task13-bootstrap-v1\x00"
  || seed_bytes
  || replicate_id as unsigned u32 big-endian
  || draw_id as unsigned u32 big-endian
  || rejection_attempt as unsigned u32 big-endian
)
```

Take the first 64 digest bits as an unsigned integer. Map it to `[0,19]` by rejection sampling against `2^64 - (2^64 mod 20)` and then `% 20`. Increment `rejection_attempt` only after rejection.

`bootstrap_indices.bin` stores exactly `10,000 x 20 = 200,000` selected indices as one byte in row-major replicate/draw order. The canonical matrix SHA-256 is `0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53`. The statistics receipt records its SHA-256, the seed, and the canonical core-ID-list SHA-256.

### Decimal arithmetic

Metric values are extracted from `canonical_json_bytes(ScoreRecordV3)` using `json.loads(..., parse_float=Decimal, parse_int=Decimal)`. Point estimates and replicate means use `Decimal` precision 50 with `ROUND_HALF_EVEN` and canonical core order.

Evidence artifacts retain canonical, unrounded decimal strings. Display rounding is not part of the statistics contract:

- rate metrics may later be shown as percentages with one decimal point;
- non-rate metrics may later be shown with three decimals;
- claim decisions and ordering always use raw decimals.

### Metric set and support semantics

Task 13 v1 reports these Task 12 metrics:

```text
protocol_scores.answer_parse_valid
answer_scores.exact_match
answer_scores.token_f1
answer_scores.stale_copied
answer_scores.gold_retrieved_wrong_answer
retrieval_scores.stale_exposure_rate
retrieval_scores.stale_count_in_context
```

`answer_scores.exact_match` is the principal Task 12 metric. The others are answer-layer or retrieval diagnostics.

For one `(cell, metric)`, support classification inspects all 80 authenticated task rows; no row may be dropped or silently collapsed:

1. all 80 values non-null and finite, with exactly four task rows per core: aggregate the four values into each of 20 `z[i,c,m]` core means and compute the estimate and interval;
2. all 80 values null with exactly the same typed support record: emit `unsupported` with null estimate and interval;
3. mixed null/non-null, differing support records, missing/duplicate task or core IDs, a core with anything other than four task rows, non-finite value, or wrong count: validation failure; publish nothing.

Unsupported never becomes zero and unsupported rows are never dropped. A paired contrast is numeric only when both cells are numeric on the same 20 cores and the same four task IDs per core. Two all-unsupported cells produce a typed-null contrast; mixed support states are invalid.

### Predeclared paired contrasts

For each answer-model slot, `k in {4,8,16}`, and reported metric:

```text
reverse_no_label - chronological_no_label
reverse_labeled - reverse_no_label
```

Directions are immutable and included in every record. This covers the order intervention and the incremental version-label intervention without generating an exhaustive post-hoc contrast family.

## Authenticated input loading

Task 13 revalidates, rather than trusts, Task 12 outputs:

- preparation manifest, dry-run plan, Core task manifest and tasks;
- matrix bundle manifest and matrix summary;
- 18 authorization, task-view, run-config, run-manifest, task-run, score, and score-receipt artifacts;
- exact 18 ordered `(cell_id, answer_model_slot)` pairs;
- exact 80 ordered task IDs per run, grouped into exactly 20 ordered semantic-core IDs with four task IDs per core;
- runtime revision/tree, run/score hashes, 1,440-row totals, zero FAILED/PARTIAL;
- integrity-audit artifact hash.

Downstream validation uses each Task 12 authorization's recorded runtime binding. It does not require Task 13 code to have the Task 12 runtime revision.

The loader returns immutable in-memory evidence objects. Statistics code cannot open arbitrary alternate task/run/score paths.

## Output contracts

### Cell statistics

Each `Task13CellStatisticV1` binds:

- cell and answer-model slot;
- metric descriptor and support status;
- exactly `task_count=80` source task rows, `core_count=20` ordered core means, and canonical core-list hash;
- point estimate and percentile endpoints as canonical decimal strings;
- source run-manifest and score-artifact hashes;
- bootstrap config and index-matrix hashes.

### Paired contrasts

Each `Task13PairedContrastV1` binds the directed left/right cells, common slot and k, metric, 20 paired core means projected from the same 80 task IDs (four per core), estimate/interval or typed unsupported state, both source run/score hashes, and bootstrap hashes.

### Statistics receipt

`Task13StatisticsReceiptV1` binds all Task 12 input hashes, the statistics-config hash, runtime revision for Task 13, `semantic_core_count=20`, `task_count=1,440`, core and bootstrap-index hashes, exactly 126 cell-statistic records (`18 x 7`), exactly 84 paired-contrast records (`12 x 7`), and hashes of `cell_statistics.jsonl` and `paired_contrasts.jsonl`.

### Generated claim ledger

The claim ledger is generated from statistics records, not manually maintained prose. Every `Task13ClaimLedgerRecordV1` includes:

- stable claim/table-cell ID;
- `direct_cell` or `paired_contrast` kind;
- metric path, direction, slice, and the frozen typed denominator `{task_count: 80, semantic_core_count: 20, tasks_per_core: 4}`;
- raw estimate/interval copied from statistics;
- exact run IDs and run/score artifact hashes;
- statistics receipt hash;
- deterministic case IDs and case-index hash.

The ledger does not label a contrast "significant", assign stars, or create a universal mechanism claim.

### Case verification

For each of the 18 runs, deterministic selection considers four categories in order:

```text
correct
stale_copied
answer_parse_invalid
other_wrong
```

For each available category, select the lowest `(semantic_core_id UTF-8 bytes, task_id UTF-8 bytes)`, with no duplicate task within a run. This yields at most 72 cases and guarantees every run has at least one case when it has any task row.

A `Task13CaseRecordV1` is a projection of authenticated task/run/score artifacts. It copies, and never recomputes:

- task metadata/provenance and timeline;
- gold and predicted actions;
- snapshots/final state where available;
- retrieval context;
- answer raw output, parsed output, format flags, and parser errors;
- complete metric layers, support metadata, failure flags, and primary failure;
- task/run/score/manifest/summary hashes.

The verifier rejoins each case to source artifacts and requires exact equality for copied metrics, support records, failures, and relevant runtime traces.

## Atomic publication

Task 13 publishes a new result root, outside the repository, immutable Core, Task 12 matrix, and evidence roots. It stages and atomically publishes:

```text
bootstrap_indices.bin
cell_statistics.jsonl
paired_contrasts.jsonl
statistics_receipt.json
cases.jsonl
case_index.json
claim_ledger.jsonl
task13_artifact_index.json
```

Publication order resolves hashes without cycles:

1. bootstrap indices, cell statistics, and contrasts;
2. statistics receipt binding those artifacts;
3. cases and case index;
4. claim ledger binding receipt and case index;
5. final artifact index binding the preceding seven outputs. The index never self-hashes; its own SHA-256 is emitted by the CLI and recorded in `WORKFLOW.md` and the later Task 14 root index.

Any validation, statistics, case, or ledger failure leaves no final output root.

## Required validation

Tests must cover:

- exact 20-core coverage, exact 80-task/20-core x4 grouping, and duplicate/missing/foreign task/core rejection;
- input-order invariance;
- golden index-matrix digest and Type-1 endpoint selection;
- all-zero/all-one degenerate intervals;
- paired identity and constant-difference properties;
- independent resampling rejection;
- all-unsupported typed null and mixed-support failure;
- manifest/run/score/summary hash tampering;
- deterministic claim-ledger IDs and completeness;
- case metric-copy equality and source redaction;
- atomic failure cleanup and no-clobber publication;
- full Task 12 evidence integration on the cluster.

## Release boundary

Task 13 completion means authenticated intervals, generated ledger, and verified cases exist. It does not by itself declare overall Core `FINAL_APPROVED`; Task 14 remains the only final release-review gate.
