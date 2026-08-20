# MemUpdateBench Post-Core Model Expansion Design

**Status:** FROZEN FOR PHASE 0 IMPLEMENTATION  
**Release family:** `memupdatebench.post-core.*.v1`  
**Network/model execution:** forbidden in Phase 0

## Purpose

The bounded vNext Core release is `FINAL_APPROVED` and immutable. This new release family prepares the broader model panel needed for a main-track benchmark without changing Core Task 11–14 contracts, matrices, statistics, or roots.

Phase 0 publishes only a validated model-intent registry, provenance policy, offline qualification report, capability report, and explicit call/budget plan. It performs no download, model load, API call, credential read, or benchmark inference. Unknown identities remain typed pending and cannot enter executable plans.

## Candidate roles

| registry key | role | initial state | future scope |
| --- | --- | --- | --- |
| `qwen35_9b_bf16` | modern open anchor | `PENDING_OFFICIAL_IDENTITY` | BF16 full matrix after qualification |
| `meta_muse_glimmer_30b_int4` | larger open anchor | `PENDING_OFFICIAL_IDENTITY` | fixed static int4 full matrix |
| `meta_muse_glimmer_30b_bf16` | quantization control | `PENDING_OFFICIAL_IDENTITY` | k=16 subset only |
| `claude_sonnet_4_6` | closed full | `PENDING_PROVIDER_QUALIFICATION` | full matrix |
| `claude_opus_4_8` | expensive closed | `PENDING_PROVIDER_QUALIFICATION` | hard subset only |
| `gemini_3_6_flash` | closed full | `PENDING_PROVIDER_QUALIFICATION` | full matrix |
| `grok_4_5` | closed full | `PENDING_PROVIDER_QUALIFICATION` | full matrix |
| `gpt_5_5` | proposed closed | `PENDING_OFFICIAL_IDENTITY` | no calls until identity verified |

Display names are not official identities. `official_model_id`, revision, license, architecture, weights URI, tokenizer identity, endpoint, and resolved upstream identity remain null until authenticated evidence is supplied.

## Contracts

All contracts are immutable Pydantic v2 models with `extra=forbid`, Python 3.10 support, exact schema literals, lowercase SHA-256, finite Decimal strings, canonical JSON/JSONL, explicit typed status, and no caller-controlled readiness.

Required contract families:

- model candidate and model identity;
- license and artifact refs;
- open snapshot/API provenance;
- qualification and capability probes;
- quantization and speculative-decoding factors;
- matrix cells and stable call IDs;
- call/token/cost budgets;
- gate results and blocked reasons;
- execution plan;
- release manifest, validation receipt, and non-self-hashing artifact index.

A candidate may be `PENDING_OFFICIAL_IDENTITY`, `PENDING_LOCAL_SNAPSHOT`, `PENDING_PROVIDER_QUALIFICATION`, `READY_FOR_OFFLINE_PREFLIGHT`, `READY_FOR_PROVIDER_PREFLIGHT`, `QUALIFIED`, or `BLOCKED`. Pending/blocked candidates cannot appear in an executable call list.

## Provenance and secrets

Open provenance requires official URI/revision/license evidence, config/tokenizer/chat-template hashes, complete tree hash/counts, architecture/context/dtype, runtime/package/CUDA/engine identity, trust-remote-code policy, and storage/redistribution class.

Glimmer int4 additionally requires base identity, static quantized checkpoint hash, quantizer method/version, bits/group-size/zero-point/symmetry/compute dtype, calibration IDs/hash, and excluded modules. Dynamic unrecorded quantization is not acceptable.

Closed provenance requires provider, requested/resolved ID, stable/preview status, pinned endpoint/API version, SDK, request ID/time, request/prompt/template hash, redacted raw-response hash, stop/parser/usage/cost metadata, and explicit credential environment-variable name.

Credential values, Authorization headers, bearer tokens, API-key-shaped fields, private keys, or secrets are rejected recursively. Phase 0 imports no provider SDK, opens no network socket, and has no network/API/download CLI flag.

## Offline qualification

Phase 0 validates metadata and policy only. It checks Core/Task14 source anchors, registry completeness, pending-field discipline, provenance shape, no-network policy, parser fixture contracts, executable-plan exclusion, and typed unsupported/pending reasons.

Phase 1 later authenticates Qwen3.5-9B BF16 and Glimmer snapshots and performs local load/VRAM/format/determinism preflights. Phase 2 later permits one counted provider identity/format probe only after explicit network and budget approval.

## Experiment design

Primary confirmatory hard subset:

```text
chronological / no label / k=16
reverse / no label / k=16
reverse / latest-outdated label / k=16
```

After qualification:

- Qwen3.5-9B BF16: full 18 cells;
- Glimmer fixed int4: full 18 cells if resource/qualification gates pass;
- Glimmer BF16: three k=16 cells only;
- Claude Sonnet, Gemini Flash, Grok, and a verified GPT model: full matrix subject to budget;
- Claude Opus: hard subset only.

Closed/nondeterministic models use at least three explicit repetitions; increase to five under a frozen variation threshold. Every repetition/seed/precision/quantization/speculative mode is a separate coordinate and receipt.

Glimmer scientific baseline is speculative OFF. Speculative ON requires a paired parity receipt using the same target checkpoint/tokenizer/prompt/seed/precision/parser and predeclared output/score/format/stale-copy equivalence plus latency/error tolerance. BF16/int4 sensitivity is a separate paired k=16 factor, never pooled into one estimate.

## Calls and budgets

Every planned call has a deterministic ID over release, experiment, model key, task/core/cell/split, repetition, seed, precision, quantization, speculative mode, and prompt hash. Budgets include qualification calls, provider probes, benchmark calls, stability/parity calls, maximum retries, prompt/output token caps, timeout, price version, estimated cost, and hard maximum cost.

Unknown identity or price makes a closed-model plan non-executable. Failed/retried attempts count against the budget. Phase 0 contains zero executable calls.

The first future execution after Qwen qualification is a local canary plan, proposed as:

```text
20 semantic cores × 4 tasks/core × 2 conditions × 2 seeds = 320 generations
```

It is not executable until the exact snapshot and design are frozen.

## Phase 0 artifacts

Publish exactly:

```text
post_core_release_manifest.json
model_registry.json
provenance.jsonl
qualification_report.json
capability_probe_report.json
execution_plan.json
post_core_artifact_index.json
```

The index binds the preceding six artifacts and does not self-hash. Publication is no-clobber, source-bound, atomic, fsynced, and reopened before success. Pending candidates are a valid Phase 0 outcome but never executable.

## CLI boundary

`vnext_prepare_post_core_release.py` and `vnext_qualify_post_core_models.py` accept config/registry/Core/Task14/provenance/output/`--execute`. In Phase 0, `--execute` means validate and publish metadata only. No `--allow-network`, endpoint, API key, model override, download, or inference flag exists.

## Later phases

Phase 1 authenticates open snapshots. Phase 2 authenticates closed provider identities/formats. Phase 3 runs the confirmatory hard subset. Phase 4 runs qualified full matrices. Phase 5 runs stability/quantization/speculative parity. Phase 6 publishes new semantic-core clustered statistics, claims, cases, and a separate post-Core final release.

Core and Task13/14 remain read-only inputs throughout.
