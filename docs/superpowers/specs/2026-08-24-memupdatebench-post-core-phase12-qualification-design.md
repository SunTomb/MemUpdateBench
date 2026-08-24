# MemUpdateBench Post-Core Phase 1–2 Qualification Release Design

**Status:** PROPOSED FOR IMPLEMENTATION
**Release family:** `memupdatebench.post-core.qualification-release.*`
**Base commit:** `a56857431023d2af1a392c75c5575316a916c174`
**Scientific execution:** forbidden by this design; model/provider execution requires a separate explicit authorization

## 1. Purpose

This design defines a separate Post-Core Phase 1–2 qualification release for the expanded answer-model panel. It converts already-completed storage, load-only, and provider connectivity evidence into source-bound qualification inputs; adds reproducible open-model runtime and short-generation gates; and defines a uniform 8–16 request capability smoke that emits typed readiness decisions.

The release is operational evidence only. It does not run MemUpdateBench tasks, compute benchmark metrics, establish prompted-answer accuracy, or repair the benchmark's broader external-validity gap. The existing Core, Task 9–14, Post-Core Phase 0, and official identity evidence remain immutable inputs.

## 2. Non-negotiable boundaries

The implementation must not modify, regenerate, overwrite, rebind, or reinterpret:

```text
data/vnext/core/v3
Task 9–14 roots
Task 14 final root or index
Post-Core Phase 0 root or index
configs/vnext/post_core/official_identity_evidence_v1.json
configs/vnext/post_core/release_v1.json
mub/vnext/post_core/qualification_v1.py semantics
```

`qualification_v1.py` remains the no-network Phase 0 metadata gate with zero provider calls, zero network calls, and zero model loads. Phase 1–2 uses a separately named contract and publication path rather than changing Phase 0 fields from zero to executable values.

The implementation must never read, print, write, commit, hash, or persist API key values, bearer tokens, Authorization values, private keys, credential-bearing commands, or raw provider configuration. Credential environment-variable names may be recorded only from a strict allowlist; values remain inaccessible to publication code.

Unknown, unavailable, unsupported, or unmeasured values remain typed null/status values. They must never be silently encoded as numeric zero.

## 3. Evidence already available

### 3.1 Authenticated open snapshots

The following storage roots are authenticated inputs, not runtime evidence:

- Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- Muse Glimmer 30B GGUF, revision `70bf1b61ac09f91b24d39038091b41c582bc5d7a`;
- Muse Glimmer 30B BF16, revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.

The qualification release binds the existing closure and independent-audit receipts by SHA-256. It does not copy, redownload, or rehash the model trees during ordinary publication.

### 3.2 Qwen load-only evidence

Qwen3.5-9B loaded and unloaded offline on Tang-3 GPU6. The receipt establishes runtime load compatibility and memory recovery only. It contains zero generation and cannot satisfy the short-generation, chat-template, parser, or determinism gates.

### 3.3 Closed-provider connectivity evidence

Five API-Transfer-Station request routes completed bounded local and Tang-2 connectivity/interface probes:

```text
claude-sonnet-4-6          2 calls
claude-opus-4-8            2 calls
Gemini 3.6 Flash (Low)     2 calls
grok-4.5                   2 calls
gpt-5.5                    4 calls
                            --------
total                      12 calls
```

All calls used `Reply only: OK`, `max_tokens=8`, and zero retries. They produced zero benchmark generations. GPT-5.5 first returned complete SSE despite `stream=false`; after the transfer-station fix, the bounded Tang-2 retest returned standard `application/json` Anthropic Message format.

Because raw responses, request IDs, and credential-bearing configuration were intentionally not persisted, these calls can produce only source-bound aggregate attestations. The release must not invent per-call raw-response hashes or request identifiers.

Gemini provenance always keeps these separate fields:

```text
canonical_model_identity:        gemini-3.6-flash
transfer_station_request_name:   Gemini 3.6 Flash (Low)
transfer_station_reasoning_tier: Low
```

Matching transfer-station request and response model names do not establish immutable upstream identities for Grok 4.5 or GPT-5.5.

## 4. Architecture

The implementation adds four isolated components.

### 4.1 Qualification contracts

A new module, `mub/vnext/post_core/qualification_receipts_v1.py`, owns immutable Pydantic v2 contracts for:

- source bindings;
- aggregate provider capability attestations;
- open-runtime manifests and receipts;
- individual capability attempts;
- capability smoke plans and budgets;
- scoped readiness decisions;
- validation and release-index records.

Contracts use `extra=forbid`, exact schema literals, finite Decimal values, lowercase SHA-256, canonical JSON/JSONL, explicit nullability, and stable enum values.

### 4.2 Runtime qualification

A new module, `mub/vnext/post_core/qualification_runtime_v1.py`, validates imported runtime receipts and deterministic fixture outputs. It does not load models itself. Provider/model runner processes remain separate execution adapters so the release builder can be tested and run without importing provider SDKs, opening sockets, reading credentials, or initializing CUDA.

### 4.3 Decision engine

A new module, `mub/vnext/post_core/qualification_decisions_v1.py`, derives readiness from validated evidence. Callers cannot set READY directly. The engine emits exactly:

```text
READY
BLOCKED
UNSUPPORTED
```

Every decision also carries a mandatory scope:

```text
STORAGE_INPUT
SHORT_GENERATION_GATE
CAPABILITY_SMOKE
BENCHMARK_ADMISSION
```

A route may be READY for `CAPABILITY_SMOKE` while remaining BLOCKED for `BENCHMARK_ADMISSION`. This is required for Grok 4.5 and GPT-5.5 unless separate immutable-identity policy evidence is later supplied. No Phase 1–2 decision mutates the Phase 0 `CandidateIdentityState` or silently upgrades a candidate to `QUALIFIED`.

### 4.4 Atomic release publisher

A new module, `mub/vnext/post_core/qualification_release_v1.py`, validates all source hashes, builds canonical artifacts in memory, writes them to a same-filesystem temporary sibling, fsyncs files and directory metadata, atomically renames into an absent caller-owned output root, and reopens every published artifact before reporting success.

Publication refuses an existing output root, symlinked/unsafe path, frozen-source path, stale source, inconsistent call count, secret-like content, or incomplete index. The index binds all preceding artifacts and never hashes itself.

## 5. Proposed files

```text
configs/vnext/post_core/qualification_release_v1.json
mub/vnext/post_core/qualification_receipts_v1.py
mub/vnext/post_core/qualification_runtime_v1.py
mub/vnext/post_core/qualification_decisions_v1.py
mub/vnext/post_core/qualification_release_v1.py
scripts/vnext_prepare_post_core_qualification_release.py
scripts/vnext_run_post_core_capability_smoke.py
tests/vnext/test_post_core_qualification_receipts.py
tests/vnext/test_post_core_qualification_runtime.py
tests/vnext/test_post_core_qualification_decisions.py
tests/vnext/test_post_core_qualification_release.py
tests/vnext/test_post_core_capability_smoke_cli.py
```

The preparation CLI imports already-produced, redacted receipts and publishes the qualification release. The smoke CLI validates a frozen plan and dispatches only through explicit runner adapters after separate execution authorization. Merely passing `--execute` to the publisher must not authorize provider or model execution.

## 6. Release artifacts

The qualification release publishes exactly:

```text
qualification_release_manifest.json
source_bindings.json
provider_capability_attestations.jsonl
open_runtime_receipts.jsonl
capability_smoke_plan.json
qualification_decisions.json
qualification_validation_receipt.json
qualification_artifact_index.json
```

The index binds the preceding seven artifacts in fixed order and excludes itself. Every artifact carries the release ID and schema version. `source_bindings.json` binds at minimum:

- Core v3 release/index;
- Task 14 final root/index;
- Post-Core Phase 0 root/index;
- `official_identity_evidence_v1.json`;
- authenticated open-snapshot closure and audit receipts;
- Qwen load-only receipt;
- base Git commit and the exact WORKFLOW/handoff source blobs;
- qualification config, prompt fixtures, parser, and runner hashes.

External cluster receipts are imported only through a caller-supplied source bundle with declared SHA-256. The publisher never reaches into credential-bearing home-directory configuration or raw provider logs.

## 7. Provider attestation contract

Each aggregate provider attestation records:

- registry key and provider class;
- exact transfer-station request name;
- canonical identity, when authenticated independently;
- response model-match status;
- local/Tang-2 location class;
- exact call count and retry count;
- HTTP/response-format class;
- exact-output parse status;
- stop-reason and usage-presence status when retained;
- benchmark-generation count, fixed at zero for the prior probes;
- evidence class and source binding;
- identity caveat and scientific-status fields;
- `raw_response_persisted=false` for the prior 12-call evidence.

The aggregate set must validate the exact per-route counts and total of 12. The failed SSH quoting attempt is represented as a pre-provider transport/setup event with `provider_call_count=0`; it is not added to the provider-call total.

GPT-5.5 requires separate pre-fix and post-fix format observations. Combining them into one undifferentiated success row is invalid. Gemini requires all three canonical/request/tier fields; omission or silent normalization is invalid.

For future capability-smoke calls, a runner may compute a SHA-256 over a strictly redacted, canonical response projection in memory. The projection must exclude headers, credentials, endpoint secrets, raw configuration, and unbounded response text. The original raw response is discarded.

## 8. Open-model runtime contracts

### 8.1 Qwen3.5-9B

The generation gate binds:

- official revision and snapshot tree hash;
- Python, Transformers, PyTorch, Accelerate, CUDA, and driver identities;
- device identity and memory boundary;
- tokenizer and chat-template hashes;
- trust-remote-code policy;
- dtype, attention implementation, context limit, and maximum generated tokens;
- deterministic generation parameters and seed;
- prompt-fixture and parser hashes;
- per-attempt output projection hash and parse result;
- load, generation, determinism, unload, and memory-recovery statuses.

The prior load-only receipt remains a source and is not overwritten.

### 8.2 Muse Glimmer GGUF

Before loading, the runtime manifest freezes:

- llama.cpp repository commit and source-tree hash;
- compiler, CUDA toolkit, build flags, and binary SHA-256;
- GGUF revision, target checkpoint hash, and quantization identity;
- device, driver, VRAM, GPU-layer count, context, batch/ubatch, thread count, flash-attention, mmap, and mlock settings;
- seed, sampling, token cap, timeout, template, and parser hashes;
- `speculative_decoding=off`.

The presence of a DFlash model does not authorize speculative decoding. A future speculative-ON experiment requires an independent paired parity receipt.

### 8.3 Muse Glimmer BF16

The BF16 role is a k=16 quantization sensitivity control. If the approved hardware cannot load it, the decision is `BLOCKED` with a resource/runtime reason, not `UNSUPPORTED` and not a row of zero measurements. `UNSUPPORTED` is reserved for a demonstrated contract/backend incompatibility that cannot satisfy the declared interface.

## 9. Short-generation gate

After separate model-execution authorization, Qwen and Muse GGUF each pass this sequence:

1. authenticate the frozen snapshot and runtime;
2. load offline;
3. render and hash the declared chat template;
4. run fixed, non-benchmark short-generation fixtures;
5. parse outputs into exact typed results;
6. repeat identical fixtures under identical deterministic settings;
7. compare output projections and parser outcomes;
8. unload and verify process/device memory recovery;
9. emit a secret-free receipt regardless of pass or failure.

Generation uses a short output cap, fixed deterministic parameters, explicit timeout, and zero retries. A failed attempt is retained and counted; it is never silently rerun or omitted.

## 10. Uniform 8–16 request capability smoke

The default smoke contains eight preregistered attempts per candidate role:

```text
2 exact-output/interface fixtures × 2 repetitions = 4
2 chat-template/parser fixtures  × 2 repetitions = 4
                                              total = 8
```

Fixtures contain no MemUpdateBench task item and produce no EM, F1, state accuracy, stale copied, or scientific claim. They test only request transport, template application, short response generation, parser stability, response identity fields, and clean teardown where applicable.

A second preregistered eight-attempt batch may run only when the first batch produces a typed parser, format, or stability anomaly. It raises the per-role maximum to 16. This is an escalation batch, not a retry. All attempts count against the frozen budget; `max_retries=0` remains unchanged.

Each attempt records a deterministic call ID over release, candidate, fixture, repetition, runtime/endpoint class, prompt hash, parser hash, and generation parameters. Closed calls additionally record requested and returned model names, response-format class, stop reason, usage presence, latency, and normalized redacted-response hash when available.

## 11. Readiness semantics

`READY` means every required gate for the named `decision_scope` passed. It never means benchmark accuracy is known.

`BLOCKED` means required evidence or resources are missing, identity policy is unresolved, a prior gate failed, or execution lacks authorization. A blocker is potentially removable.

`UNSUPPORTED` means the candidate/backend demonstrably cannot satisfy the declared capability contract. It is not a synonym for missing evidence, quota exhaustion, unavailable hardware, null data, or an unattempted run.

Expected boundary examples:

- Qwen may become READY for capability smoke after short-generation/determinism/unload gates pass.
- Muse GGUF may become READY only after the frozen llama.cpp runtime and equivalent gates pass.
- Muse BF16 may remain BLOCKED on insufficient hardware.
- Claude Sonnet/Opus and Gemini may become READY for capability smoke while scientific status remains NOT_RUN.
- Grok/GPT route interfaces may be READY for capability smoke, while benchmark admission remains BLOCKED by immutable-identity policy.

## 12. Error handling and security

The release publisher and CLIs use typed exit codes for usage/contract failure, stale source, blocked qualification, unsafe path, publication failure, and untrusted runtime failure. User-controlled exception text and raw provider errors are not printed verbatim.

A recursive secret validator rejects:

- key names containing credential/token/authorization/private-key patterns;
- bearer/API-key-shaped values;
- raw HTTP Authorization headers;
- private-key blocks;
- unallowlisted credential environment-variable names;
- endpoint URLs containing userinfo or secret query parameters.

The validator operates on every imported payload and every final artifact. Tests use synthetic canary secrets and never real credentials.

Publication cannot target the repository's immutable source roots, any ancestor/descendant that would overlap them, or an existing path. A failed publication leaves no visible final root.

## 13. Testing and verification

Required offline tests include:

- strict/frozen/canonical contract behavior;
- exact artifact order and non-self-hashing index;
- no-replace and atomic publication;
- stale/missing source rejection;
- frozen-root and symlink/path-overlap rejection;
- recursive secret rejection;
- prior provider counts sum to exactly 12 with GPT equal to four;
- failed SSH setup contributes zero provider calls;
- Gemini three-field preservation;
- GPT pre-fix SSE and post-fix JSON separation;
- absent raw evidence remains explicit rather than fabricated;
- null/unsupported values never become zero;
- failed/escalated attempts count against budgets;
- caller-controlled READY rejection;
- scope-specific READY does not imply benchmark admission;
- no benchmark metric fields in capability receipts;
- smoke plan has eight base attempts and at most eight explicit escalation attempts per role;
- publisher and fixture tests import no provider SDK, open no socket, read no credentials, initialize no CUDA, and load no model.

The implementation phase may run compile checks, targeted tests, and the full existing Post-Core/vNext offline test subset. It must not run the smoke CLI against a real adapter without separate authorization.

## 14. Future execution sequence

The frozen order is:

1. implement and offline-validate this qualification release;
2. obtain explicit model/provider execution authorization;
3. run Qwen and Muse short-generation gates;
4. run the uniform 8–16 capability smoke;
5. publish typed readiness decisions;
6. obtain separate authorization for the 320-generation canary;
7. run the canary;
8. run the three confirmatory hard conditions;
9. consider qualified full matrices only afterward.

The 320-generation Qwen canary remains:

```text
20 semantic cores × 4 tasks/core × 2 conditions × 2 seeds = 320 generations
```

The primary confirmatory hard subset remains:

```text
chronological / no label / k=16
reverse / no label / k=16
reverse / latest-outdated label / k=16
```

Neither the canary nor the confirmatory subset is authorized by implementing this release.

## 15. Acceptance criteria

The implementation unit is complete only when:

- all new contracts and CLIs are source-bound and tested;
- Phase 0 v1 artifacts and semantics are byte-for-byte untouched;
- the publisher creates only absent, atomic, reopened releases;
- the 12-call provider evidence is represented honestly as aggregate attestation;
- Qwen/Muse runtime gates can consume synthetic receipts without loading a model;
- capability planning produces bounded 8–16 attempt plans with zero retries;
- decision scopes prevent capability READY from becoming benchmark READY;
- secret, null, identity, and evidence-class boundaries have dedicated regression tests;
- existing relevant offline tests and new tests pass;
- validation reports zero model loads, zero provider calls, zero benchmark generations, and zero credential reads during implementation verification.
