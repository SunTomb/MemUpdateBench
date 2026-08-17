# Core Task 13 Statistics, Ledger, and Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an authenticated Task 13 pipeline that computes deterministic semantic-core clustered intervals and paired contrasts from the final Task 12 matrix, generates a canonical claim ledger, exports verified cases, and publishes all outputs atomically.

**Architecture:** A new `mub.vnext.statistics` package separates immutable contracts, deterministic bootstrap arithmetic, authenticated Task 12 evidence loading, statistics projection, case export, ledger generation, and orchestration. One CLI loads the final matrix and a tracked statistical config, validates everything before calculation, then stages eight mutually bound artifacts and atomically renames the result root. Task 13 never recomputes Task 12 score fields and never modifies Core or Task 12 artifacts.

**Tech Stack:** Python 3.10+, Pydantic v2 immutable contracts, `Decimal`, SHA-256 counter stream, canonical JSON/JSONL, existing v3 task/run/score manifests, `publish_files_atomically`, pytest.

---

## File structure

**Create:**

```text
mub/vnext/statistics/__init__.py
mub/vnext/statistics/contracts_v3.py
mub/vnext/statistics/bootstrap_v3.py
mub/vnext/statistics/input_v3.py
mub/vnext/statistics/statistics_v3.py
mub/vnext/statistics/cases_v3.py
mub/vnext/statistics/ledger_v3.py
mub/vnext/statistics/task13_v3.py
configs/vnext/core_task13_statistics_v1.json
scripts/vnext_run_core_task13.py
tests/vnext/test_core_task13_contracts.py
tests/vnext/test_core_task13_bootstrap.py
tests/vnext/test_core_task13_input.py
tests/vnext/test_core_task13_statistics.py
tests/vnext/test_core_task13_cases_ledger.py
tests/vnext/test_core_task13_cli.py
```

**Modify:**

```text
WORKFLOW.md
```

Each module owns one responsibility. Do not place bootstrap, case, and ledger logic in one large orchestration file.

---

### Task 1: Freeze Task 13 contracts and tracked config

**Files:**
- Create: `mub/vnext/statistics/__init__.py`
- Create: `mub/vnext/statistics/contracts_v3.py`
- Create: `configs/vnext/core_task13_statistics_v1.json`
- Test: `tests/vnext/test_core_task13_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from decimal import Decimal
import pytest
from mub.vnext.statistics.contracts_v3 import (
    Task13BootstrapConfigV1,
    Task13IntervalV1,
    Task13StatisticStatus,
)


def test_task13_bootstrap_contract_is_exact():
    config = Task13BootstrapConfigV1.model_validate_json(
        Path("configs/vnext/core_task13_statistics_v1.json").read_bytes()
    )
    assert config.cluster_key == "semantic_core_id"
    assert config.expected_cluster_count == 20
    assert config.replicates == 10_000
    assert config.confidence_level == "0.95"
    assert config.quantile_method == "inverted_cdf"
    assert config.lower_order_statistic == 250
    assert config.upper_order_statistic == 9_750


def test_numeric_and_unsupported_intervals_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Task13IntervalV1(
            status=Task13StatisticStatus.UNSUPPORTED,
            estimate="0.0",
            lower=None,
            upper=None,
            support=None,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/vnext/test_core_task13_contracts.py -q
```

Expected: import failure because `mub.vnext.statistics.contracts_v3` does not exist.

- [ ] **Step 3: Implement exact immutable contracts**

Define at minimum:

```python
class Task13StatisticStatus(str, Enum):
    NUMERIC = "numeric"
    UNSUPPORTED = "unsupported"


class Task13BootstrapConfigV1(ImmutableContractModel):
    schema_version: Literal[
        "memupdatebench.core-task13-statistics-config.v1"
    ] = "memupdatebench.core-task13-statistics-config.v1"
    cluster_key: Literal["semantic_core_id"]
    expected_cluster_count: Literal[20]
    seed_hex: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    replicates: Literal[10000]
    draws_per_replicate: Literal[20]
    confidence_level: Literal["0.95"]
    interval_method: Literal["clustered_percentile"]
    quantile_method: Literal["inverted_cdf"]
    lower_order_statistic: Literal[250]
    upper_order_statistic: Literal[9750]
    decimal_precision: Literal[50]
    decimal_rounding: Literal["ROUND_HALF_EVEN"]
    support_policy: Literal["all_supported_or_all_unsupported"]
    metric_paths: tuple[str, ...]
```

Also define exact models for:

```text
Task13ArtifactBindingV1
Task13IntervalV1
Task13CellStatisticV1
Task13PairedContrastV1
Task13StatisticsReceiptV1
Task13CaseSelectorV1
Task13CaseRecordV1
Task13CaseIndexV1
Task13ClaimLedgerRecordV1
Task13ArtifactIndexV1
```

Use canonical decimal-string validation instead of JSON floats for estimates and endpoints. Require null numeric fields for `unsupported`, exact typed support metadata, positive task/core counts, unique ordered IDs, and lowercase SHA-256 fields.

- [ ] **Step 4: Add the canonical config**

Write `configs/vnext/core_task13_statistics_v1.json` with:

```json
{"cluster_key":"semantic_core_id","confidence_level":"0.95","decimal_precision":50,"decimal_rounding":"ROUND_HALF_EVEN","draws_per_replicate":20,"expected_cluster_count":20,"interval_method":"clustered_percentile","lower_order_statistic":250,"metric_paths":["answer_scores.exact_match","answer_scores.gold_retrieved_wrong_answer","answer_scores.stale_copied","answer_scores.token_f1","protocol_scores.answer_parse_valid","retrieval_scores.stale_count_in_context","retrieval_scores.stale_exposure_rate"],"quantile_method":"inverted_cdf","replicates":10000,"schema_version":"memupdatebench.core-task13-statistics-config.v1","seed_hex":"9e3779b97f4a7c15d1b54a32d192ed03e47b8a31f5c6d2098374ab10ce69d542","support_policy":"all_supported_or_all_unsupported","upper_order_statistic":9750}
```

- [ ] **Step 5: Run contract tests and compile**

```bash
python -m py_compile mub/vnext/statistics/contracts_v3.py
python -m pytest tests/vnext/test_core_task13_contracts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the contract unit**

```bash
git add mub/vnext/statistics/__init__.py mub/vnext/statistics/contracts_v3.py configs/vnext/core_task13_statistics_v1.json tests/vnext/test_core_task13_contracts.py
git commit -m "feat: freeze Core Task 13 statistics contracts"
```

---

### Task 2: Implement the deterministic clustered bootstrap

**Files:**
- Create: `mub/vnext/statistics/bootstrap_v3.py`
- Test: `tests/vnext/test_core_task13_bootstrap.py`

- [ ] **Step 1: Write failing golden-stream and interval tests**

```python
from decimal import Decimal
from mub.vnext.statistics.bootstrap_v3 import (
    build_bootstrap_indices_v1,
    clustered_percentile_interval_v1,
    paired_percentile_interval_v1,
)


def test_bootstrap_index_stream_is_golden_and_order_invariant():
    core_ids = tuple(f"core-{index:03d}" for index in range(20))
    matrix = build_bootstrap_indices_v1(core_ids)
    assert len(matrix.raw) == 200_000
    assert matrix.sha256 == "0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53"
    assert matrix.rows[0][:8] == (13, 18, 1, 16, 4, 8, 4, 9)
    assert build_bootstrap_indices_v1(tuple(reversed(core_ids))).sha256 == matrix.sha256


def test_paired_identity_is_exact_zero():
    values = {f"core-{index:03d}": Decimal(index) / Decimal(19) for index in range(20)}
    result = paired_percentile_interval_v1(values, values)
    assert result.estimate == result.lower == result.upper == "0"
```

Before replacing golden placeholders, compute the first row independently in the test using the frozen SHA-256 formula, not by calling the production mapping helper.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/vnext/test_core_task13_bootstrap.py -q
```

Expected: missing module/functions.

- [ ] **Step 3: Implement the counter stream and binary matrix**

```python
_DOMAIN = b"MUB-Core-Task13-bootstrap-v1\x00"
_LIMIT = 1 << 64


def _draw(seed: bytes, replicate: int, draw: int, cluster_count: int) -> int:
    threshold = _LIMIT - (_LIMIT % cluster_count)
    attempt = 0
    while True:
        digest = hashlib.sha256(
            _DOMAIN
            + seed
            + replicate.to_bytes(4, "big")
            + draw.to_bytes(4, "big")
            + attempt.to_bytes(4, "big")
        ).digest()
        value = int.from_bytes(digest[:8], "big")
        if value < threshold:
            return value % cluster_count
        attempt += 1
```

Sort core IDs by UTF-8 bytes before creating 10,000 rows. Serialize every index as one byte. Reject non-20, duplicate, blank, or non-string core IDs.

- [ ] **Step 4: Implement Decimal cell and paired intervals**

Use `localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN))`. For each replicate, sum in draw order and divide by `Decimal(20)`. Sort replicate `Decimal` values and choose offsets 249 and 9,749. Canonicalize Decimal strings by removing redundant trailing zeros and normalizing negative zero to `"0"`.

- [ ] **Step 5: Add all-zero/all-one, constant contrast, Type-1 quantile, and shuffle tests**

Tests must prove:

```text
all zero -> 0 [0,0]
all one -> 1 [1,1]
A=B+k -> every paired replicate equals k
input row shuffling -> byte-identical result
10,000 increasing values -> exact 250th/9,750th values
```

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/vnext/test_core_task13_bootstrap.py -q
git add mub/vnext/statistics/bootstrap_v3.py tests/vnext/test_core_task13_bootstrap.py
git commit -m "feat: add deterministic semantic-core bootstrap"
```

---

### Task 3: Load and authenticate the complete Task 12 matrix

**Files:**
- Create: `mub/vnext/statistics/input_v3.py`
- Test: `tests/vnext/test_core_task13_input.py`

- [ ] **Step 1: Build a compact 18-run fixture and failing loader tests**

Reuse Task 12 fixture builders, but keep the loader tests fake-model-free. Tests must cover:

```python
def test_task13_loader_returns_18_exact_runs_and_shared_20_cores_x4_tasks(...): ...
def test_task13_loader_rejects_shape_preserving_task_substitution_or_core_reassignment(...): ...
def test_task13_loader_rejects_summary_hash_tampering(...): ...
def test_task13_loader_rejects_missing_score_row(...): ...
def test_task13_loader_rejects_foreign_runtime_binding(...): ...
def test_task13_loader_rejects_incomplete_or_reordered_matrix(...): ...
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/vnext/test_core_task13_input.py -q
```

- [ ] **Step 3: Implement `Task13AuthenticatedMatrixV1` loader**

The loader accepts explicit paths for Core, preparation/evidence, matrix root, matrix manifest, matrix summary, and integrity audit. It must:

1. load the one-LF authenticated dry-run plan with the Task 12 control loader;
2. run `validate_task12_manifest_plan_v3`;
3. authenticate Core task-manifest/tasks bytes;
4. verify matrix manifest and summary hashes supplied by the caller;
5. verify exact ordered admitted pairs;
6. read each authorization and use its recorded runtime binding when calling the internal Task 12 bundle validator;
7. call `load_finalized_task12_run_v3` and `verify_task12_score_artifact_v3`;
8. compare every run/score hash with the matrix summary;
9. require 18 x 80 task rows, and require all 18 runs to expose one exact canonical 80-task-ID sequence: 20 semantic-core IDs ordered by UTF-8 bytes, followed within each core by four task IDs ordered by UTF-8 bytes, with identical task-to-core assignment; reject any run that substitutes a task ID or reassigns a task to a different core even if it preserves the 80-row/20-core-x4 shape; require zero FAILED/PARTIAL;
10. verify the integrity-audit artifact hash and matching runtime/matrix/summary identifiers.

Return immutable run objects containing task, runtime row, score, source hashes, typed cell metadata, and `semantic_core_id`.

- [ ] **Step 4: Run input tests and commit**

```bash
python -m pytest tests/vnext/test_core_task13_input.py -q
git add mub/vnext/statistics/input_v3.py tests/vnext/test_core_task13_input.py
git commit -m "feat: authenticate Core Task 13 matrix inputs"
```

---

### Task 4: Project core observations and compute statistics

**Files:**
- Create: `mub/vnext/statistics/statistics_v3.py`
- Test: `tests/vnext/test_core_task13_statistics.py`

- [ ] **Step 1: Write failing projection/support tests**

```python
def test_projection_reads_decimal_from_canonical_score_json(): ...
def test_all_unsupported_emits_typed_null(): ...
def test_mixed_supported_and_unsupported_fails(): ...
def test_duplicate_or_missing_core_fails(): ...
def test_cell_and_paired_outputs_are_order_invariant(): ...
```

- [ ] **Step 2: Implement canonical metric extraction**

```python
def decimal_metric(score: ScoreRecordV3, path: str) -> Decimal | None:
    payload = json.loads(
        canonical_json_bytes(score),
        parse_float=Decimal,
        parse_int=Decimal,
    )
    layer, field = path.split(".", 1)
    return payload[layer][field]
```

Require the metric path to exist in `CORE_METRIC_REGISTRY_V3` and the tracked config.

- [ ] **Step 3: Implement support classification**

For each cell/metric, validate all 80 task rows and their exact 20-core x4 grouping. If all values are numeric and finite, order the four task IDs within each UTF-8 ordered core, compute one Decimal per-core projection `z[i,c,m] = mean(task_values[i,0:4])`, and pass the resulting 20-core mapping to the interval code. If all values are null, require the same exact `MetricFieldSupport` model on all 80 rows and emit typed unsupported. Reject mixed state, non-finite values, missing/duplicate task or core IDs, wrong rows-per-core, or wrong total count.

- [ ] **Step 4: Implement cell statistics and predeclared contrasts**

Generate cells in matrix manifest order and metrics in config order. Generate contrasts in this exact order for each slot and k:

```text
reverse_no_label - chronological_no_label
reverse_labeled - reverse_no_label
```

For paired contrasts, align the exact same canonical four task IDs within each of the 20 cores across both compared cells before subtracting per-core projections; the task IDs and task-to-core assignment must be identical on both sides. Report `core_count=20` and keep the source task-row cardinality explicit as 80 per side where represented. Use the one shared bootstrap matrix from Task 2. Bind each output to run/score hashes and core/bootstrap hashes.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/vnext/test_core_task13_statistics.py -q
git add mub/vnext/statistics/statistics_v3.py tests/vnext/test_core_task13_statistics.py
git commit -m "feat: compute Core Task 13 clustered intervals"
```

---

### Task 5: Export and verify v3 cases

**Files:**
- Create: `mub/vnext/statistics/cases_v3.py`
- Test: `tests/vnext/test_core_task13_cases_ledger.py`

- [ ] **Step 1: Write failing deterministic selection and copy-verification tests**

```python
def test_case_selection_is_stratified_and_order_invariant(): ...
def test_case_metrics_are_copied_not_recomputed(): ...
def test_case_verifier_rejects_changed_score_or_trace(): ...
def test_private_source_text_is_redacted(): ...
```

- [ ] **Step 2: Implement category selection**

Classify each source row:

```python
if score.answer_scores.exact_match == 1:
    category = "correct"
elif score.answer_scores.stale_copied == 1:
    category = "stale_copied"
elif any(not prediction.format_valid for prediction in run.answer_predictions):
    category = "answer_parse_invalid"
else:
    category = "other_wrong"
```

For every run and available category, select the lowest `(semantic_core_id.encode(), task_id.encode())`, without selecting one task twice in a run.

- [ ] **Step 3: Implement projection and verifier**

Export task metadata, timeline, actions, snapshots, retrieval, answers, all score layers, support metadata, failures, and artifact hashes. Verify the case by rejoining its exact task/run/score rows and comparing copied fields byte-for-byte after canonical serialization. Redact private source text using the existing v2 case-export policy as the reference.

- [ ] **Step 4: Run case tests**

```bash
python -m pytest tests/vnext/test_core_task13_cases_ledger.py -k case -q
```

Expected: PASS.

---

### Task 6: Generate the statistics receipt and canonical claim ledger

**Files:**
- Create: `mub/vnext/statistics/ledger_v3.py`
- Continue test: `tests/vnext/test_core_task13_cases_ledger.py`

- [ ] **Step 1: Write failing ledger completeness tests**

```python
def test_ledger_has_one_row_per_statistic_and_contrast(): ...
def test_ledger_ids_are_stable_under_input_shuffle(): ...
def test_ledger_binds_receipt_case_index_runs_and_scores(): ...
def test_ledger_preserves_unsupported_as_null(): ...
```

- [ ] **Step 2: Build `Task13StatisticsReceiptV1`**

Bind Task 12 preparation/plan/matrix/summary/audit hashes, Task 13 config hash, Task 13 runtime revision/tree, `semantic_core_count=20`, `task_count=1,440`, 20-core hash, bootstrap binary hash, cell/contrast artifact hashes and counts. Claim rows use the typed denominator `{task_count: 80, semantic_core_count: 20, tasks_per_core: 4}` so observation and independent-cluster counts cannot be conflated.

- [ ] **Step 3: Build the case index**

The case index binds ordered case IDs, category/run coverage, `cases.jsonl` SHA-256, record count, and all source run/score hashes.

- [ ] **Step 4: Generate ledger rows**

Stable IDs are SHA-256-derived identifiers over this canonical payload:

```python
{
    "kind": kind,
    "slot": slot,
    "cell_or_contrast": identifier,
    "metric_path": metric_path,
    "slice": slice_payload,
}
```

Ledger values are copied from statistics records. Include all exact source run IDs, run-manifest hashes, score-artifact hashes, statistics-receipt hash, case-index hash, and case IDs.

- [ ] **Step 5: Run ledger tests and commit Tasks 5–6**

```bash
python -m pytest tests/vnext/test_core_task13_cases_ledger.py -q
git add mub/vnext/statistics/cases_v3.py mub/vnext/statistics/ledger_v3.py tests/vnext/test_core_task13_cases_ledger.py
git commit -m "feat: generate Task 13 ledger and verified cases"
```

---

### Task 7: Implement atomic Task 13 orchestration and CLI

**Files:**
- Create: `mub/vnext/statistics/task13_v3.py`
- Create: `scripts/vnext_run_core_task13.py`
- Test: `tests/vnext/test_core_task13_cli.py`

- [ ] **Step 1: Write failing output-root, atomicity, and CLI-surface tests**

```python
def test_task13_pipeline_publishes_all_eight_artifacts_atomically(): ...
def test_task13_pipeline_refuses_existing_or_overlapping_root(): ...
def test_task13_fault_leaves_no_final_root(monkeypatch): ...
def test_task13_cli_has_no_model_provider_token_or_fake_flags(): ...
def test_task13_cli_requires_execute(): ...
```

- [ ] **Step 2: Implement orchestration**

The pipeline performs all input validation and all in-memory computations before creating staging output. Serialize:

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

The final index binds the preceding seven artifacts exactly; it does not self-hash. The CLI prints the index file SHA-256 for `WORKFLOW.md` and Task 14.

Use canonical JSON lines and `publish_files_atomically` within an owned staging root. Verify every staged model/hash, then rename staging to the absent final root. On failure remove only the owned staging root.

- [ ] **Step 3: Implement the production CLI**

Required flags:

```text
--manifest
--plan
--core-root
--evidence-root
--matrix-root
--matrix-bundle-manifest
--matrix-summary
--matrix-integrity-audit
--statistics-config
--output-root
--execute
```

No model, provider, token, API, fake, or metric override flag is permitted.

- [ ] **Step 4: Run CLI tests and commit**

```bash
python -m pytest tests/vnext/test_core_task13_cli.py -q
git add mub/vnext/statistics/task13_v3.py scripts/vnext_run_core_task13.py tests/vnext/test_core_task13_cli.py
git commit -m "feat: publish authenticated Core Task 13 artifacts"
```

---

### Task 8: Run the focused local gate and independent reviews

**Files:**
- All Task 13 files

- [ ] **Step 1: Run syntax and focused tests**

```bash
python -m py_compile \
  mub/vnext/statistics/contracts_v3.py \
  mub/vnext/statistics/bootstrap_v3.py \
  mub/vnext/statistics/input_v3.py \
  mub/vnext/statistics/statistics_v3.py \
  mub/vnext/statistics/cases_v3.py \
  mub/vnext/statistics/ledger_v3.py \
  mub/vnext/statistics/task13_v3.py \
  scripts/vnext_run_core_task13.py
python -m pytest \
  tests/vnext/test_core_task13_contracts.py \
  tests/vnext/test_core_task13_bootstrap.py \
  tests/vnext/test_core_task13_input.py \
  tests/vnext/test_core_task13_statistics.py \
  tests/vnext/test_core_task13_cases_ledger.py \
  tests/vnext/test_core_task13_cli.py -q
```

- [ ] **Step 2: Run Task 12 regression compatibility**

```bash
python -m pytest \
  tests/vnext/test_core_task12_execution.py \
  tests/vnext/test_core_task12_matrix_bundle.py -q
```

- [ ] **Step 3: Request spec and code-quality reviews**

Review must explicitly attempt to refute:

```text
semantic-core unit and paired resampling
PRNG/index golden determinism
unsupported/null policy
Task 12 input authentication
receipt/ledger/case hash closure
case score-copy equality
atomic publication/no-clobber
Task 13 vs Task 14 boundary
```

Fix confirmed findings and rerun their exact regressions.

- [ ] **Step 4: Check diff and commit the coherent local gate**

```bash
git diff --check
git status --short
git commit -m "feat: complete Core Task 13 analysis gate"
```

---

### Task 9: Execute Task 13 on the authenticated real matrix

**Files:**
- Modify: `WORKFLOW.md`
- Generate remotely: a new Task 13 result root outside repository/Core/evidence/Task 12 roots

- [ ] **Step 1: Transfer the clean Task 13 commit to Tang-2**

Use the same self-contained Git-bundle and clean detached-worktree procedure established for Task 12. Use an isolated project-local `HOME` for exact `safe.directory` entries. Do not modify or clean the dirty remote main worktree.

- [ ] **Step 2: Reverify Task 12 input hashes**

Require exact matches:

```text
Task 12 matrix manifest: 85145a8a460ee6cec3785926f9aaa85c8bee8cd41d4ad0582d2b0333b8cf10d2
Task 12 matrix summary:  a1c4f89af2b9f39de9791ce9c6348c24b4c81474abf3da865f22e5dfe68f1f15
Task 12 integrity audit: bfc85922c36dcc87deca983ce39ff395b10da00c2ee91c8aba7a6c02c3f04f60
```

- [ ] **Step 3: Run Task 13 without any model/GPU path**

```bash
repo=/NAS/yesh/MemUpdateBench
prep=$repo/results/vnext/core_task12_preparation_v2
matrix=$repo/results/vnext/core_task12_answer_matrix_9c798df_v1
logs=$repo/results/vnext/core_task12_answer_matrix_9c798df_v1_logs
revision=$(git rev-parse HEAD)
python scripts/vnext_run_core_task13.py \
  --manifest $prep/task12_preparation_manifest.json \
  --plan $prep/dry_run_plan.json \
  --core-root $repo/data/vnext/core/v3 \
  --evidence-root $prep/evidence \
  --matrix-root $matrix \
  --matrix-bundle-manifest $matrix/matrix_bundle_manifest.json \
  --matrix-summary $matrix/matrix_run_summary.json \
  --matrix-integrity-audit $logs/matrix_integrity_audit.json \
  --statistics-config configs/vnext/core_task13_statistics_v1.json \
  --output-root $repo/results/vnext/core_task13_${revision:0:7}_v1 \
  --execute
```

- [ ] **Step 4: Independently recompute and audit outputs**

Verify:

```text
18 cell-statistic groups x 7 metrics
12 directed slot/k contrast pairs x 7 metrics
10,000 x 20 bootstrap indices (`bootstrap_indices.bin` is exactly 200,000 bytes; SHA-256 `0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53`)
all ledger rows accounted for
all case records rejoin and reproduce scores
unsupported retrieval metrics remain typed null
all output hashes match receipt and artifact index
```

- [ ] **Step 5: Record results and boundaries in WORKFLOW.md**

Include motivation, commands, input/output hashes, config, core/task counts, intervals, contrasts, unsupported metrics, case counts, errors, conclusions, and the explicit statement:

```text
Task 13 is complete; Task 14 and overall Core FINAL_APPROVED remain not started.
```

- [ ] **Step 6: Commit evidence documentation**

```bash
git add WORKFLOW.md
git commit -m "docs: record complete Core Task 13 evidence"
```
