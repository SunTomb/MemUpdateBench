# Bounded Cross-Domain Qwen Readout

## Outcome

The authorized four-request NOAA/BEA answer trial reached a real Qwen3.5-9B load on a shared Tang-1 GPU1, then stopped on the first generation with a technical `RuntimeError`. The model unloaded cleanly. The first request is a `TECHNICAL_FAILURE`; the other three are `NOT_RUN`; answer metrics remain **null**. This is not an accuracy zero and no answer was scored.

CPU-only validation also passed the tokenizer/template and bounded Torch-proxy checks. The implementation is a separate offline Qwen answer reader of frozen Qdrant retrieval records. It does not implement native Qdrant answering, semantic-vector retrieval, another manager experiment, a broader benchmark release, or scientific approval. Four prospective readouts belong to **two source trajectories**, not four independent examples.

## Frozen inputs and authorization

The input packet is the unchanged `cross_domain_answer_replay_20261002_v2` preparation:

```text
1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687
```

The user explicitly authorized one model load, at most four frozen retrieval-context readouts, 64 new tokens per request and 256 total, zero generation retries, no paid API calls, no Qdrant rerun, and no termination of another user's processes. Authorization is not inferred from historical GPU records, hash-bound profiles, successful CPU tests, or another agent's report. The historical preparation's `execution_authorized=false` field is not rewritten by this later authorization.

The user subsequently authorized use of Tang-1, Tang-2 and Tang-3 and explicitly authorized sharing Tang-1 GPU1 with another process. The final run therefore binds Tang1/GPU1 UUID `GPU-479aa50d-c1ce-0bc5-e91e-770c97243178`, records one foreign process without terminating it, and retains the 30,720 MiB pre-load floor. Shared use is an observation and permission boundary, not a reservation.

The answer reader uses pinned `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, model tree `e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db`, and template SHA-256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Seed 0, greedy decoding, one beam, BF16 and eager attention are required settings. The final Tang-1 run measured a successful BF16 model load and cleanup but no successful generation. The 4,096-token admission ceiling is not a claim about model capacity.

Visible model input contains only the canonical public query/schema and projected provider-returned entries. It excludes gold answers, hidden target selectors, final-state substitutions, actions and correctness annotations. Provider order is preserved; scores/ranks are not added to the prompt. Decoded suffix text is parsed using the frozen number parser without rounding, tolerance, unit stripping, fence removal or regex extraction. Typed exact comparison distinguishes an integer from its floating-point representation.

## Implementation and enforcement

- `mub/vnext/runtime/cross_domain_qwen_backend.py` loads the local tokenizer, checks template/token bindings, permits one model-load callback and four distinct generation attempts, rejects retries, and releases partial/full load resources.
- `scripts/vnext_run_cross_domain_qwen_readout.py` measures source/snapshot/runtime bindings, performs CPU-only qualification, then supervises an owned GPU worker with phase and total deadlines. It uses owned-process helpers, not a Qdrant execution.
- Admission binds hostname, A40 index and UUID, checks current occupancy and at least 30,720 MiB free before loading, and permits foreign processes only for the explicitly authorized shared Tang-1 GPU1. Only after a successful load does the memory floor change to 1,024 MiB for the same owned worker; no foreign process is terminated or included in owned cleanup.
- A CPU qualifier resolves the model class without constructing a model. It binds four tokenizer/template observations. The GPU worker verifies these before generating and revalidates source, profile, snapshot and measured runtime before publishing completion.
- A terminal `COMPLETE` requires worker exit, an empty owned process group, absence of the owned GPU PID, removal of the owned temporary root, and unchanged bindings. The Tang-1 run exercised a real load and CUDA cleanup but did not produce a successful generation or a complete-answer result; fake tests and the real cleanup receipt do not establish successful-generation watchdog coverage.
- Runtime measurement covers loaded framework files and selected distribution metadata, not the complete operating-system dependency closure.
- Only exact `torch.ops`/`torch.classes` synthetic proxy identities are admitted with their expected type, owner, null module spec and placeholder filename. Their real `torch._ops`/`torch._classes` implementation files are still checked for distribution containment and hashed. Ordinary escaped file imports remain rejected.
- Generation intent, backend-callback entry and actual backend generation counts are separate. Counts lost with a terminated worker remain unknown/null rather than fabricated zero. Completed malformed answers would be incorrect completed readouts; technical failures and not-run work retain null aggregate quality metrics.

The driver has an exact one-use run-root/profile boundary, not an arbitrary deployment CLI. These roots already exist, so replaying their launch is forbidden. CPU receipt verification does not authorize or trigger a future model run.

## Preserved diagnostics

The original records and later shared-device attempts are retained without rewriting their status or source bindings. Public regression copies live in `tests/vnext/fixtures/cross_domain_qwen_blocked/`.

| Record | Observed stop | Qualification processes | Load workers | Generation attempts |
| --- | --- | ---: | ---: | ---: |
| v2 | GPU4 admission rejected; separately observed occupied device | 0 | 0 | 0 |
| v3 | CPU runtime-origin check misclassified Torch synthetic proxies | 1 | 0 | 0 |
| v4 | GPU6 admission rejected; separately observed occupied device | 0 | 0 | 0 |
| tang1gpu1b | Shared Tang-1 pre-load supervision check stopped before a child | 0 | 0 | 0 |
| tang1gpu1c | Shared Tang-1 pre-load supervision check stopped before a child | 0 | 0 | 0 |
| tang1gpu1d | Real Qwen load succeeded; first generation raised `RuntimeError`; remaining requests stopped | 1 | 1 | 1 |

The v2/v3/v4/tang1gpu1b/tang1gpu1c records have null answer metrics and four true cleanup checks. The tang1gpu1d record also has null answer metrics and four true cleanup checks, but it records one actual model load and one failed generation attempt. Its first row is `TECHNICAL_FAILURE`; its other three rows are `NOT_RUN`. No answer is treated as incorrect completed output. Generic supervisor `ValueError` records do not replace the more specific worker `RuntimeError` boundary in tang1gpu1d. No foreign process was touched.

Result-index SHA-256 pins:

```text
v2 89aebaec527853d37142bc81cfa07ab9b3d3c89ac7371607b72253a6c1192349
v3 4061638a7652a9be5a0a72b53295b47fee40876dd9f38fb6f4348ac99668732e
v4 cb49d495991c0591db4e1f395f04ece3f4b8c30febdba890d4228140467816bc
```

The v4 deployment source/profile/archive pins are:

```text
source manifest 3de4969ab6e85972586b4da2a542ce467b9c21af9585063866b011cdc2027730
profile         5ff6208774e25b81add85c106a1c1ebf5165f0ef1111738cf97c0b47a58b4e6e
archive         d1822b9a9531d2ed20b8e6dcdbef616763dcabff30a3f37a4c49922ead3ad832
```

The final shared Tang-1 run is `cross_domain_qwen_readout_20261009_tang1gpu1d`. Its authenticated pins are:

```text
result index    5aaf45dcba2092826c251f38e3f8b518aa86d09718e80fb9fb985f31f3f4705b
local audit     cb4e851eafd4c5d04974c6857a684b5948debecd71f256975cdbc9c01d07e12f
source manifest 70dc375720a5ee45349220063bcce03dc15618deca66b5c06e3633ba8a10e4a2
profile         88a3a28683dd6fe81e128c5b6a0c1c7d59e7132435efe9285e597d2768f0952b
archive         9247236c5a80f1b88e7c5b086595a0d449d126822b5a6a1c6ab36eb729fc1ac1
collection tar  e583316736f4c01d66410c4525fae5237dd4ba5750c38b09fb4a07fdb859072e
```

The real backend observed one load attempt and one generation callback/attempt. It allocated 17,941,189,632 bytes and reserved 18,001,952,768 bytes before close; both allocated and reserved bytes were zero afterward. The backend used the guarded process-local cuBLAS workspace cleanup and returned `CLOSED`. No generated token count or parsed answer exists because the first generation raised `RuntimeError` before producing an output. The remaining three requests were not attempted. Because the authorized one-load/four-request trial has entered real model execution and consumed its single load/first attempt, no further model retry is made in this work package.

Commit `109d8b2` preserves the exact driver/backend used in the earlier v4 frozen bundle; the later Tang-1 source/profile/run bindings are separate no-replace diagnostics. The subsequent accounting and shared-device corrections are not retrofitted into earlier records. A whole-bundle comparison attempt was blocked by the host's worktree command-classification gate, so the two-file blob check for v4 is not presented as verification of all 244 source/input members.


## Separate CPU validation

After the proxy correction, a separate CPU-only check reopened the frozen preparation, prepared four requests, resolved the Qwen class, measured 1,423 framework/metadata records and revalidated those records. Input-token counts were 258, 261, 246 and 247. It loaded one tokenizer, no model, and generated no answers. It recorded no provider calls or GPU operations.

Receipt SHA-256:

```text
62a127595d423157ba7d454a9a3c3a5626bbb5ba304ecf4bda8b485ac11075d4
```

This check used the exact v4 source; it did not perform full snapshot requalification, model loading, GPU runtime validation or answer scoring. Its public receipt contains relative runtime members, hashes/counts and no raw prompt/output text. It is engineering validation only, not an independent human audit, a signature or scientific approval.

## Validation and limits

Before deploying v4, 63 backend/driver tests passed. Two further regression tests reproduced the intent-versus-generation accounting defect before its correction. The prior targeted backend, driver, execution, preparation, parser and evidence gate passed **167 tests with two Windows symlink-permission skips**; the prior immutable-receipt gate passed **175 tests with the same two skips**. The Tang-1 shared-binding and bounded-headroom additions passed 69 backend/driver tests, and the final targeted gate including the Tang-1 real-result fixture is recorded in the workflow entry for this run. A full `tests/vnext tests/maintenance` attempt was stopped at the 600-second limit while still below 3% completion; it did not complete and is not reported as passing.

Offline targeted command:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/vnext/test_cross_domain_qwen_readout.py \
  tests/vnext/test_cross_domain_qwen_backend.py \
  tests/vnext/test_cross_domain_qwen_blocked_evidence.py \
  tests/vnext/test_cross_domain_answer_execution.py \
  tests/vnext/test_cross_domain_answer_replay.py \
  tests/vnext/test_core_answer_model_v3.py \
  tests/vnext/test_cross_domain_qdrant_state_evidence.py
```

The eight fixture-only tests passed against 12 exact copied files (243,491 bytes), including the CPU-only metadata receipt. An isolated export of staged tree `fe6aa8c98e4cfe2ba68fcc0c64c39c72de2a0a82` passed all **73 new backend/driver/receipt tests** and confirmed that every imported `mub`/`scripts` module came from that export. `PYTHONPATH` was unset; installed test dependencies were explicitly added without processing user-site path hooks. The initial isolated invocation could not find pytest with user-site packages disabled; no dependency was installed or silently substituted. These clean-checkout tests ran no real model.

The final run used shared Tang-1 GPU1 without terminating its foreign process. The four-request prompted-answer result remains unavailable as an accuracy row: one real load occurred, the first generation failed technically, and three requests were not run. Existing NOAA/BEA task releases, Qdrant v13 evidence, frozen preparation, Core/Pilot/Phase 0 roots and the original test-only production gate remain unchanged.
