# Bounded Cross-Domain Qwen Readout

## Outcome

The authorized four-request NOAA/BEA answer trial remains **BLOCKED before model loading**. CPU-only validation passed the tokenizer/template and bounded Torch-proxy checks. There are **zero model loads and zero answer generations**, and answer metrics remain **null**, not zero accuracy.

The implementation is a separate offline Qwen answer reader of frozen Qdrant retrieval records. It does not implement native Qdrant answering, semantic-vector retrieval, another manager experiment, a broader benchmark release, or scientific approval. Four prospective readouts belong to **two source trajectories**, not four independent examples.

## Frozen inputs and authorization

The input packet is the unchanged `cross_domain_answer_replay_20261002_v2` preparation:

```text
1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687
```

The user explicitly authorized one model load, at most four frozen retrieval-context readouts, 64 new tokens per request and 256 total, zero generation retries, no paid API calls, no Qdrant rerun, and no termination of another user's processes. Authorization is not inferred from historical GPU records, hash-bound profiles, successful CPU tests, or another agent's report. The historical preparation's `execution_authorized=false` field is not rewritten by this later authorization.

The answer reader uses pinned `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, model tree `e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db`, and template SHA-256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Seed 0, greedy decoding, one beam, BF16 and eager attention are required settings, **not measured successful GPU execution in this work package**. The 4,096-token admission ceiling is not a claim about model capacity.

Visible model input contains only the canonical public query/schema and projected provider-returned entries. It excludes gold answers, hidden target selectors, final-state substitutions, actions and correctness annotations. Provider order is preserved; scores/ranks are not added to the prompt. Decoded suffix text is parsed using the frozen number parser without rounding, tolerance, unit stripping, fence removal or regex extraction. Typed exact comparison distinguishes an integer from its floating-point representation.

## Implementation and enforcement

- `mub/vnext/runtime/cross_domain_qwen_backend.py` loads the local tokenizer, checks template/token bindings, permits one model-load callback and four distinct generation attempts, rejects retries, and releases partial/full load resources.
- `scripts/vnext_run_cross_domain_qwen_readout.py` measures source/snapshot/runtime bindings, performs CPU-only qualification, then supervises an owned GPU worker with phase and total deadlines. It uses owned-process helpers, not a Qdrant execution.
- Admission binds hostname, A40 index and UUID, checks current occupancy and at least 30,720 MiB free before loading, and rejects other selected-device compute processes. Only after a successful load does the memory floor change to 1,024 MiB for the same owned worker; foreign-process rejection remains active.
- A CPU qualifier resolves the model class without constructing a model. It binds four tokenizer/template observations. The GPU worker verifies these before generating and revalidates source, profile, snapshot and measured runtime before publishing completion.
- A terminal `COMPLETE` requires worker exit, an empty owned process group, absence of the owned GPU PID, removal of the owned temporary root, and unchanged bindings. Fake tests inspect this policy; these diagnostics did **not** exercise a real GPU load/unload or generation-time watchdog interruption.
- Runtime measurement covers loaded framework files and selected distribution metadata, not the complete operating-system dependency closure.
- Only exact `torch.ops`/`torch.classes` synthetic proxy identities are admitted with their expected type, owner, null module spec and placeholder filename. Their real `torch._ops`/`torch._classes` implementation files are still checked for distribution containment and hashed. Ordinary escaped file imports remain rejected.
- Generation intent, backend-callback entry and actual backend generation counts are separate. Counts lost with a terminated worker remain unknown/null rather than fabricated zero. Completed malformed answers would be incorrect completed readouts; technical failures and not-run work retain null aggregate quality metrics.

The driver has an exact one-use run-root/profile boundary, not an arbitrary deployment CLI. These roots already exist, so replaying their launch is forbidden. CPU receipt verification does not authorize or trigger a future model run.

## Preserved diagnostics

The original records are retained without rewriting their status or source bindings. Public regression copies live in `tests/vnext/fixtures/cross_domain_qwen_blocked/`.

| Record | Observed stop | Qualification processes | Load workers | Generations |
| --- | --- | ---: | ---: | ---: |
| v2 | GPU4 admission rejected; separately observed occupied device | 0 | 0 | 0 |
| v3 | CPU runtime-origin check misclassified Torch synthetic proxies | 1 | 0 | 0 |
| v4 | GPU6 admission rejected; separately observed occupied device | 0 | 0 | 0 |

All three terminal records have `BLOCKED`, null answer metrics, zero paid-provider/Qdrant calls and all four cleanup checks true. Their generic supervisor `ValueError` records do not themselves identify device occupancy; that diagnosis comes from separate read-only device observations. The v1 archive-order deployment rejection preceded these terminal records and launched no worker.

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

Commit `109d8b2` preserves the exact driver/backend used in the v4 frozen bundle; both blob identities were compared against the staged execution files. The subsequent accounting correction is **not** retrofitted into that source manifest or any diagnostic record. A whole-bundle comparison attempt was blocked by the host's worktree command-classification gate, so the two-file blob check is not presented as verification of all 244 source/input members.

## Separate CPU validation

After the proxy correction, a separate CPU-only check reopened the frozen preparation, prepared four requests, resolved the Qwen class, measured 1,423 framework/metadata records and revalidated those records. Input-token counts were 258, 261, 246 and 247. It loaded one tokenizer, no model, and generated no answers. It recorded no provider calls or GPU operations.

Receipt SHA-256:

```text
62a127595d423157ba7d454a9a3c3a5626bbb5ba304ecf4bda8b485ac11075d4
```

This check used the exact v4 source; it did not perform full snapshot requalification, model loading, GPU runtime validation or answer scoring. Its public receipt contains relative runtime members, hashes/counts and no raw prompt/output text. It is engineering validation only, not an independent human audit, a signature or scientific approval.

## Validation and limits

Before deploying v4, 63 backend/driver tests passed. Two further regression tests reproduced the intent-versus-generation accounting defect before its correction. The initial targeted backend, driver, execution, preparation, parser and evidence gate passed **167 tests with two Windows symlink-permission skips**. The final gate, including the immutable receipt tests, passed **175 tests with the same two skips**. A full `tests/vnext tests/maintenance` attempt was stopped at the 600-second limit while still below 3% completion; it did not complete and is not reported as passing.

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

At the last occupancy observation all eight Tang-2 GPUs had other compute jobs. No device was reserved, switched automatically inside a worker, or shared by weakening admission. The real four-readout result remains unavailable. Existing NOAA/BEA task releases, Qdrant v13 evidence, frozen preparation, Core/Pilot/Phase 0 roots and the original test-only production gate remain unchanged.
