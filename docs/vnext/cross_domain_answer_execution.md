# Answer Replay Execution

## Current boundary

This module implements **execution-layer engineering tests**, not a live answer-model runner. It consumes the immutable preparation documented in [Cross-Domain Answer Replay](cross_domain_answer_replay.md): four prospective readouts of frozen NOAA/BEA Qdrant retrieval records and two source-trajectory clusters.

- The command-line path always records `BLOCKED` production admission and constructs no backend.
- The Python test API requires `mode="injected_test_only"` and an explicitly supplied backend factory.
- Test completion is labeled `TEST_ONLY_COMPLETE`, never model `READY` or benchmark approval.
- `answer_metrics` is always null; synthetic controls are separately named `test_control_summary`.
- After invoking any injected constructor, model/tokenizer/network/GPU/provider/child-process resource totals are unknown (`null`). A method returning does not prove a real model was loaded or GPU memory was released.
- The frozen preparation is unchanged. This engineering result is a new sibling, not a promotion of its `execution_authorized=false` state.

No PyTorch/Transformers import, tokenizer loading, snapshot access, Qdrant mutation/retrieval, provider request, SSH or GPU command is implemented or invoked by the default production admission path.

## Backend interface

`PublicRequest` contains only `request_id` and `visible_prompt`. It has no task object, target selector, gold data, expected value or scoring annotation. `PreparedInput` carries rendered text and template/token observations only in memory. `GenerationOutput` carries output text and token usage only until hashing/parsing; neither raw prompt nor raw output is written to artifacts.

The protocol has `prepare`, `load`, `generate` and `close`. The injected test sequence is:

1. Authenticate the exact preparation index and reconstruct all four prompts.
2. Create a new output root and durable, hash-chained journal.
3. Record constructor intent before creating the backend.
4. Prepare all four chat/template/token bindings before a load call.
5. Record one load intent and permit at most one actual load callback.
6. For each request, revalidate inputs, prepare again, compare the post-load template/token binding, and revalidate once more before generation.
7. Record generation intent durably, then count actual callback entry separately. Permit one attempt per request, at most four total, with no retry.
8. Bind output hash and observed usage, revalidate inputs again, then apply the frozen strict number parser and descriptive typed-exact scorer.
9. On a technical failure, stop subsequent requests and always attempt close for any returned backend handle, even if timing or journaling failed.
10. Publish terminal artifacts only after cleanup attempts and final input checks.

The current interface specifies greedy decoding, seed 0, one beam and at most 64 generated tokens per request (256 across four attempts). A 4,096-token **test context-admission ceiling** is not a measured model capacity. Context overflow is rejected before load/generation, not truncated.

## Deadlines and failure accounting

The injected test harness detects exceptions/timeouts and callback-return overruns with budgets of 30 seconds for prepare, 120 for load, 60 for generation and 30 for close. This is explicitly **cooperative test-only timing**, not a hard watchdog: an arbitrary callback may block until it returns. A live backend must be supervised in an owned process with enforced deadlines before production can open.

- Constructor failures leave cleanup `UNAVAILABLE_NO_HANDLE`, not fabricated success.
- Partial load failures still call close on the returned handle.
- Unknown generation usage remains null; observed tokens and unknown-attempt counts are reported separately.
- A failure before entering generation has a recorded intent but zero generation attempts.
- Close failure blocks the whole run even if all individual test readouts completed.
- Completed malformed answers are incorrect completed controls, not technical failures.
- Failure/not-run row scores are null. Earlier complete rows can remain as partial diagnostics, but aggregate test scores are withheld when execution/cleanup is incomplete.
- Exception text is not persisted. Only bounded phase/reason codes and allowlisted exception class names are retained.

## Artifact verification

Each output root has `journal.jsonl`, `rows.jsonl`, `summary.json` and `index.json`. Publication uses exclusive directory creation and no-replace atomic payload publication. Inputs, the frozen preparation, current executor source, journal contents and owned output-directory identity are rechecked.

The reopening verifier checks artifact membership/hashes, canonical journal links and terminal-row agreement. It also checks the four expected request identities/order, attempt/token limits, intent-versus-callback accounting, lifecycle completion, null resource policy, cluster size, blocked production status, and recomputed usage/test-control summaries. Matching file hashes alone do not confer semantic admission or execution authorization.

All referenced source tasks, manager results and the v2 preparation remain immutable. No raw model output, full prompt, credential, private path or unrestricted backend exception appears in this public engineering report.

## Commands

Offline test gate:

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_cross_domain_answer_execution.py
```

Record the current fail-closed production gate in a new root:

```bash
python -B scripts/vnext_run_cross_domain_answer_replay.py \
  --output-root results/vnext/cross_domain_answer_admission_20261002_v1
```

The nonzero exit code **2** is intentional for a blocked gate; it does not mean model accuracy zero. Existing roots are never overwritten.

## Verification

The execution, frozen-preparation, canonical answer-parser and evidence regression gate passed **102 tests with two Windows symlink-permission skips**. Test backends are inspected fakes; their outputs are engineering controls, not benchmark answers.

Static review found clock-dependent cleanup, input drift during callbacks, intent/entry accounting ambiguity and incomplete semantic reopening checks. Sixteen new regression cases reproduced those gaps before the fixes. Cleanup no longer depends on clock availability; inputs are checked after callbacks before subsequent calls or scoring; durable intents are counted separately from callback entry; and hash-valid but contradictory output packages are rejected. Later cleanup-log-failure and public-input-only tests also passed. These are local regression results, not a second human audit or a model runtime qualification.

An isolated export of the staged source also reopened the unchanged preparation and the blocked gate with identical hashes, confirmed all project imports stayed in the export, and passed 15 selected execution/verifier tests (25 deliberately deselected). `PYTHONPATH` was unset. That export ran no real backend and did not depend on the development directory or archived source copies.

## Recorded production admission

The no-replace gate receipt is `results/vnext/cross_domain_answer_admission_20261002_v1`, with index SHA-256:

```text
4caae501ffcf2d40c49311f4b5adcea5bc00e5533177c5b54150b092b12f515b
```

It was reopened through the artifact and semantic verifier. All four rows are `NOT_RUN`, with zero callback attempts, zero model/tokenizer/provider/GPU/network calls and null scores. No fake successful answers are included in this production admission record.

## Live boundary when this engineering gate was recorded

At this gate, a real four-request Qwen3.5-9B readout still required explicit current device/operation authorization, replay-specific source/model/runtime qualification, an independently bound tokenizer/chat-template preflight, a qualified production backend and an owned-process hard watchdog/unload check. Historical Qwen runs and the earlier CPU-only Qdrant authorization did not satisfy these gates. This test-only implementation provides no free-form flag or receipt string that can manufacture live authorization.

A separately authorized concrete backend and CPU validation were subsequently implemented in [Bounded Cross-Domain Qwen Readout](cross_domain_qwen_readout.md). The live trial remains resource-blocked with zero model loads/generations. This does not rewrite the historical gate, enable this module's production path, or establish answer accuracy.
