# vNext Phase 0 legacy bridge

## Scope and claim boundary

Phase 0 is a compatibility and regression bridge. It defines canonical contracts, imports authenticated legacy fixtures, and records caveats without changing legacy inputs. It produces no Pilot dataset, benchmark metric, model result, external-validity result, or paper claim. Imported compatibility rows must not be presented as new held-out evidence. In particular, do not claim that stale same-slot conflict is always the strongest distractor or that this bridge establishes broad external validity.

The immutable legacy input roots for Phase 0 are `data/`, `results/`, and `paper/`. The command-line tools accept explicit paths rather than enforcing a root allowlist, so preserving those roots is an execution requirement: treat every input as read-only and publish only to a separate output directory.

## Canonical contracts

The five public artifact models are available from `mub.vnext.contracts`:

- `MemUpdateTask`
- `TaskRunRecord`
- `ScoreRecord`
- `TaskManifest`
- `RunManifest`

Exact memory-object identity is the four-part tuple

```text
(namespace, entity, attribute, subkey)
```

`MemoryObjectKey.object_type` is classification metadata, not an identity component. It is preserved in artifacts but excluded from `canonical_id`, semantic task hashes, replay keys, and exact slot resolution. Imported P6.x `(entity, attribute)` slots map to `MemoryObjectKey(namespace="default", subkey=null, object_type="slot")`; the bridge does not invent subkeys. Legacy phase labels such as `legacy_p63` never populate `MemoryObjectKey.namespace`.

`task_family` is a string field so future family names are not rejected. Canonical `SourceType` values are exactly:

```text
synthetic
dialogue
changelog
calendar
issue
report_revision
other
```

Legacy phase, dataset, metric, run, checkpoint, and trajectory identities are not source types. They are retained only in `LegacyProvenance`.

### Current versions

The current Pilot contract matrix is `2.0.0`:

| Constant | Value |
| --- | --- |
| `SCHEMA_VERSION` | `2.0.0` |
| `RUNTIME_RECORD_VERSION` | `2.0.0` |
| `SCORER_VERSION` | `2.0.0` |
| `METRIC_REGISTRY_VERSION` | `2.0.0` |
| `COMPILER_VERSION` | `2.0.0` |
| `PROFILE_VERSION` | `2.0.0` |
| `TASK_MANIFEST_VERSION` | `2.0.0` |
| `RUN_MANIFEST_VERSION` | `2.0.0` |
| `PRIMARY_FAILURE_PRECEDENCE_VERSION` | `2.0.0` |

The published Phase 0 v1 release remains immutable and is not silently upgraded. Its task/runtime/score/manifest records retain v1 semantics, including that missing or `None` answers are not abstentions. Compatibility artifacts continue to use `LEGACY_ANALYSIS_MANIFEST_VERSION=1.0.0` and `LEGACY_CLI_COMPILER_VERSION=vnext-phase0-cli-1.0.0`. Imported EvoMemory runs use adapter version `legacy-import-v1`; their unavailable original system version is represented as `legacy-unknown`, not guessed.

### Canonical serialization

Canonical model JSON is compact UTF-8 with lexicographically sorted keys, `ensure_ascii=false`, `allow_nan=false`, no omitted `null` fields, no computed fields, and no trailing newline. Sets, unordered containers, and non-finite numbers are rejected. Canonical JSONL is one canonical object followed by one LF per record. Exported JSON Schemas use deterministic sorted compact JSON and one final LF.

The checked-in canonical schemas are under `schemas/vnext/`:

```text
mem_update_task.schema.json
task_run_record.schema.json
score_record.schema.json
task_manifest.schema.json
run_manifest.schema.json
```

The compatibility-only `LegacyAnalysisManifest` schema is maintained separately as `schemas/legacy/legacy_analysis_manifest.schema.json`; the five-schema exporter does not generate it.

## Legacy provenance and namespaces

`LegacyProvenance` has exactly these fields:

```text
legacy_family_id
legacy_phase
legacy_dataset_id
legacy_split_id
legacy_metric_namespace
legacy_run_condition_id
checkpoint_family
training_seed
answer_mode
memory_trajectory_id
source_artifact_path
source_artifact_hash
known_caveats
```

The first five identity fields plus `source_artifact_path` and its lowercase SHA-256 `source_artifact_hash` are required. The run/checkpoint/seed/answer/trajectory fields are nullable, and `known_caveats` is a list. This is the sole home for legacy identity because putting legacy phases into `SourceType`, canonical family names, or object keys would make compatibility metadata alter benchmark semantics.

Current exact legacy metric/provenance namespace mapping is:

| Legacy family | `LegacyProvenance.legacy_metric_namespace` |
| --- | --- |
| `p63` | `legacy_p63` |
| `p65` | `legacy_p65` |
| `p68_p70` | `legacy_p68_p70` |
| `p80_p82` | `legacy_p80_p82` |
| `p83` | `legacy_p83` |
| `p84` | `legacy_p84` |
| `p85_api_replacement` | `legacy_p85_api_replacement` |

Lookup trims, lowercases, and removes periods before exact matching; unsupported phases fail rather than silently creating a provenance/metric namespace. These values identify imported compatibility families and metrics only. They are never canonical `MemoryObjectKey.namespace` values; imported P6.x slot keys remain in the canonical `default` object namespace.

### Event-count terminology

- `num_events` is the total number of event records and equals `len(events)`.
- `num_target_updates` is the number of exact target-object mutations and agrees with `k_updates` when that field exists.
- Legacy `num_updates` is a historical total-event count. It is preserved as `legacy_num_updates`; it must not be reinterpreted as target updates.

For example, a legacy episode can have `num_events=3`, `num_target_updates=1`, and `legacy_num_updates=3`.

## Source authentication and transactional publication

Dataset compilation records the source path/hash in every task and in the task manifest. Validation verifies a single coherent source path/hash, split, and phase; hashes the actual source; recompiles canonical tasks from it; and compares canonical bytes. Results compilation authenticates the full task artifact and task manifest before importing rows. Manifests bind paths, media types, row counts, and SHA-256 values. Immediately before publication, compilers rehash all inputs and abort if any changed.

One compiler invocation publishes one transaction. All destinations share one output directory; source/destination aliases, hardlinks, symlinks, and Windows reparse-point aliases are rejected. Outputs are staged, flushed, validated, inputs are rechecked, and then the set is linked or replaced. A failed transaction rolls back the set and does not intentionally leave a mixed generation. Atomicity does not span separate output directories or separate CLI invocations. Existing outputs require explicit `--overwrite`; omission is the safer default.

Warnings are evidence, not cosmetic output. Preserve manifest warnings and `known_caveats`. Directory-name parsing is fallback-only identity inference and emits `legacy_directory_name_inference`; callers should provide structured identity whenever available.

## Evaluation trajectory boundary

Direct and prompted answer trajectories are distinct run identities. Checkpoint family, training seed, answer mode, memory trajectory, retrieval policy, context order, and context annotation must remain separate axes.

Legacy task compilation supplies a canonical `state_direct` query but does not infer result-run identity. A result row maps `slot_direct` to `state_direct` only after the summary declares matching semantic compatibility and the row has a matching terminal state snapshot. It maps `slot_prompt` to `retrieved_prompt` only after the same declared compatibility plus saved answer traces, ranked entries tied to canonical events, and consistent row/trace answers. Otherwise canonical evaluation mode remains null and the importer emits `legacy_answer_mode_unverified`. Never infer this mapping from a directory name alone.

### `latest_per_slot`

Legacy `latest_per_slot` is an answer-time retrieval rewrite, not a manager update, state compaction, or filter over only the original top-k. It retrieves up to the full store size, deduplicates scoped legacy entries by `(entity, attribute)`, retains the greatest `(event_idx, updated_at, created_at)`, preserves unscoped entries, sorts survivors by original retrieval score, and then applies `answer_topk`. Stored memory is unchanged. Its gain mixes stale suppression with recall expansion. The legacy rewrite has no namespace/subkey/object-type information, so it must not be generalized into canonical identity semantics.

## Mechanism compatibility boundaries

### P8.3 conflict and dose rows

P8.3 supports a narrow mechanism statement: stale same-slot conflict is an order- and metadata-sensitive version-arbitration failure. `value_policy`, `context_order`, `context_annotation`, and `stale_count` remain separate cell-identity axes. Metadata such as explicit latest/outdated labels can repair adverse ordering. Conflict-type decomposition is a boundary result: unrelated distractors can be harder in a particular surface construction, so stale same-slot conflict is not universally the strongest distractor.

### P8.3 stale-specific removal

The stale-removal artifact is trace composition, not an answer-model rerun. It removes/recomposes retrieved entries while copying the source row's original EM/F1. Imported rows carry `trace_composition_not_answer_rerun`. They may diagnose context composition, but they cannot establish the counterfactual answer accuracy of a fresh rerun.

### P8.4 API answer-layer rows

P8.4 probes only the answer layer over constructed contexts; it is not a full memory-system baseline. Preserve model, condition, order, metadata, stale dose, response-format, truncation, availability, and capacity evidence.

For a present row, status precedence is capacity failure, pending, unavailable model, empty/truncated/format evidence, missing EM or stale-copy evidence, then completed. `is_completed` is true only for `completed`. Completed rows require EM and stale-copy metrics and, when a raw-response field exists, a nonblank response.

The non-destructive clean view includes only completed metric-bearing rows for the exact allowlist:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gemini-3.1-flash-lite-preview
```

and excludes explicit blank responses. Empty/truncated Gemini outputs are API/prompt-format caveats, not mechanism counterevidence. Capacity failures are not accuracy failures. A missing expected matrix cell is not automatically materialized by the importer, aggregate rows without raw-response fields can be completed from metrics, and noncompleted rows may still carry preserved metrics; clean filtering excludes them but does not delete them.

### Missing-cell policy

- **Missing:** no source row exists; do not synthesize a zero, denominator member, or successful observation.
- **Pending:** a row exists but required evidence is incomplete; retain it with pending status and exclude it from completed-cell denominators.
- **Unavailable:** the requested model/service was unavailable; retain the status and exclude it from accuracy denominators.
- **Capacity failed:** the request exceeded a model/provider capacity boundary; retain it separately from wrong answers and exclude it from accuracy denominators.
- **Format/empty/truncated:** retain the row and caveat; do not silently coerce it into an answer score.

For canonical scores, every null metric has exactly one `supported_metric_fields` entry. `not_applicable`, `not_supported`, `runtime_failed`, and `missing_artifact` state why it is null and the null policy excludes it from aggregation. A null is never an implicit zero.

## Current commands

Run these from the repository root. Replace temporary paths as needed; do not target `data/`, `results/`, or `paper/`. Examples use Git Bash/POSIX syntax. In PowerShell, set `$env:PYTHONPATH='.'` before invoking the corresponding `python` command.

### Export and compare schemas

```bash
PYTHONPATH=. python scripts/vnext_export_schemas.py --output-dir /tmp/vnext-schemas-a
PYTHONPATH=. python scripts/vnext_export_schemas.py --output-dir /tmp/vnext-schemas-b
```

The exporter writes only the five canonical schemas. Compare both generated maps byte-for-byte and compare `/tmp/vnext-schemas-a` with `schemas/vnext/`. The compatibility schema has no exporter CLI; regenerate and verify its exact canonical bytes with the current public model and schema constant:

```bash
PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path

from mub.vnext.legacy.artifacts import LegacyAnalysisManifest
from mub.vnext.schema_export import DRAFT_2020_12_URI

schema = LegacyAnalysisManifest.model_json_schema(mode="serialization")
schema["$schema"] = DRAFT_2020_12_URI
schema["title"] = "LegacyAnalysisManifest"
payload = (
    json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    + b"\n"
)
checked = Path("schemas/legacy/legacy_analysis_manifest.schema.json")
assert checked.read_bytes() == payload
print(f"verified {checked} ({len(payload)} bytes)")
PY
```

### Compile the deterministic legacy fixtures

```bash
PYTHONPATH=. python scripts/vnext_compile_legacy.py dataset \
  --input tests/vnext/fixtures/legacy/p63_dataset_minimal.json \
  --split test --legacy-phase P6.3 \
  --output-dir /tmp/vnext-fixture/tasks

PYTHONPATH=. python scripts/vnext_compile_legacy.py results \
  --input tests/vnext/fixtures/legacy/evomemory_results_old.json \
  --tasks /tmp/vnext-fixture/tasks/tasks.jsonl \
  --output-dir /tmp/vnext-fixture/results

PYTHONPATH=. python scripts/vnext_compile_legacy.py mechanism --kind conflict \
  --input tests/vnext/fixtures/legacy/p83_conflict_rows.csv \
  --output-dir /tmp/vnext-fixture/conflict
PYTHONPATH=. python scripts/vnext_compile_legacy.py mechanism --kind dose \
  --input tests/vnext/fixtures/legacy/p83_synthetic_dose_rows.csv \
  --output-dir /tmp/vnext-fixture/dose
PYTHONPATH=. python scripts/vnext_compile_legacy.py mechanism --kind stale-removal \
  --input tests/vnext/fixtures/legacy/p83_stale_removal_rows.csv \
  --output-dir /tmp/vnext-fixture/stale-removal
PYTHONPATH=. python scripts/vnext_compile_legacy.py mechanism --kind api \
  --input tests/vnext/fixtures/legacy/p84_api_rows.csv \
  --output-dir /tmp/vnext-fixture/api

PYTHONPATH=. python scripts/vnext_compile_legacy.py ledger \
  --input tests/vnext/fixtures/legacy/ledger_references.md \
  --project-root . \
  --output-dir /tmp/vnext-fixture/ledger
```

Dataset output is `tasks.jsonl` plus `task_manifest.json`. Results output is `task_runs.jsonl`, `scores.jsonl`, and `run_manifest.json`. Each mechanism output is `legacy_analysis.jsonl` plus `legacy_analysis_manifest.json`. Ledger output is `ledger_audit.json` plus `legacy_analysis_manifest.json`. Add `--overwrite` only when deliberately replacing a complete existing output set.

### Validate all five canonical artifact kinds

```bash
PYTHONPATH=. python scripts/vnext_validate_artifacts.py --kind tasks \
  --input /tmp/vnext-fixture/tasks/tasks.jsonl \
  --manifest /tmp/vnext-fixture/tasks/task_manifest.json
PYTHONPATH=. python scripts/vnext_validate_artifacts.py --kind task-manifest \
  --input /tmp/vnext-fixture/tasks/task_manifest.json \
  --manifest /tmp/vnext-fixture/tasks/task_manifest.json
PYTHONPATH=. python scripts/vnext_validate_artifacts.py --kind task-runs \
  --input /tmp/vnext-fixture/results/task_runs.jsonl \
  --manifest /tmp/vnext-fixture/results/run_manifest.json
PYTHONPATH=. python scripts/vnext_validate_artifacts.py --kind scores \
  --input /tmp/vnext-fixture/results/scores.jsonl \
  --manifest /tmp/vnext-fixture/results/run_manifest.json
PYTHONPATH=. python scripts/vnext_validate_artifacts.py --kind run-manifest \
  --input /tmp/vnext-fixture/results/run_manifest.json \
  --manifest /tmp/vnext-fixture/results/run_manifest.json
```

For manifest kinds, `--input` and `--manifest` must identify the same manifest file. Mechanism and ledger outputs use their typed compatibility manifest and are not accepted as one of the five canonical validator kinds.

### Smoke, tests, and compile checks

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m pytest tests/vnext -q
PYTHONPATH=. python -m py_compile \
  mub/vnext/version.py mub/vnext/profiles.py mub/vnext/schema_export.py \
  mub/vnext/failure.py \
  mub/vnext/contracts/enums.py mub/vnext/contracts/common.py \
  mub/vnext/contracts/task.py mub/vnext/contracts/runtime.py \
  mub/vnext/contracts/score.py mub/vnext/contracts/manifest.py \
  mub/vnext/contracts/adapter.py \
  mub/vnext/io/canonical.py mub/vnext/io/jsonl.py mub/vnext/io/atomic.py \
  mub/vnext/validation/issues.py mub/vnext/validation/task.py \
  mub/vnext/validation/replay.py mub/vnext/validation/split.py \
  mub/vnext/scoring/registry.py mub/vnext/scoring/failures.py \
  mub/vnext/scoring/scorer.py \
  mub/vnext/legacy/artifacts.py mub/vnext/legacy/caveats.py \
  mub/vnext/legacy/names.py mub/vnext/legacy/loaders.py \
  mub/vnext/legacy/dataset.py mub/vnext/legacy/results.py \
  mub/vnext/legacy/mechanisms.py mub/vnext/legacy/ledger.py \
  scripts/vnext_export_schemas.py scripts/vnext_compile_legacy.py \
  scripts/vnext_validate_artifacts.py scripts/smoke_test.py
```

## Known limitations

- Phase 0 imports compatibility evidence; it does not repair P6.3 split leakage or authorize new held-out-generalization claims.
- Legacy directory parsing is fallback-only and warning-bearing.
- The schema exporter covers five canonical schemas; the legacy analysis schema is checked separately.
- `latest_per_slot` uses legacy two-part slot metadata and is not a canonical four-part identity implementation.
- Stale-removal rows are not answer reruns.
- P8.4 clean-view filtering is intentionally narrow and cannot recover wholly absent matrix cells.
- Private/raw provider payloads remain in place and must not be copied into compatibility outputs or documentation.
- If real `data/` or `results/` artifacts are absent, report the read-only check as unavailable rather than substituting fixtures or paper tables.
