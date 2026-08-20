# MemUpdateBench Post-Core Model Expansion Phase 0 Plan

**Goal:** Publish a no-network, no-model-execution post-Core registry/provenance/qualification/budget release with all uncertain model identities explicitly pending.

## Task 1: Freeze design and plan

Create and commit the post-Core design and this plan. Self-review for placeholders, invented model identities, hidden network paths, credential leakage, Core mutation, and unbounded calls.

## Task 2: Strict contracts

Create `mub/vnext/post_core/contracts_v1.py` and contract tests. Cover identity state, pending-field discipline, quantization/speculative constraints, calls/budgets, exact artifact order, immutable manifests/receipts/index, duplicate IDs, finite cost strings, and status derivation.

## Task 3: Frozen intent registry

Create `model_registry_v1.py`, static config, and tests. Register exactly eight candidate keys/roles/states. Do not fill unverified IDs. Enforce Qwen BF16/full, Glimmer int4/full + BF16/k16, Opus/hard-only, GPT/no-execution, and no blocked/pending executable calls.

## Task 4: Provenance and secret boundary

Create `provenance_v1.py` and tests. Add canonical file/tree hashes, runtime metadata, redacted command handling, credential-env-name-only records, recursive secret scanning, malicious nested fixtures, and no-network/provider-import guards.

## Task 5: Offline qualification

Create `qualification_v1.py` and tests. Generate deterministic PASS/PENDING/BLOCKED/NOT_RUN gate rows from registry/provenance without network/model load. Missing identity/license/architecture/revision stays pending or blocked. Unsupported is explicit, never zero.

## Task 6: Matrix/call/budget planning

Create `planning_v1.py` and tests. Encode blocked hard/full scopes, repetitions/seeds, precision/quantization/speculative factors, deterministic call IDs, exact token/cost arithmetic, retry accounting, and non-executable plans for pending candidates. Publish a zero-executable-call Phase 0 plan plus future call formulas.

## Task 7: Atomic Phase 0 release

Create `release_v1.py`, CLIs, and release/CLI tests. Publish exact seven artifacts under an absent post-Core root, with Core Task14/source hashes, no-clobber atomic staging/rename, fsync, source revalidation, exact index, reopen verification, safe flags, and failure cleanup.

## Task 8: Gates and reviews

Run py_compile, all post-Core tests, no-network/socket/provider SDK guards, secret scan, deterministic/hash tests, Core/Task13/Task14 before/after hashes, diff-check, and independent spec/code-quality reviews. Fix findings and rerun exact tests.

## Task 9: Real Phase 0 publication

From a clean detached worktree, publish only metadata/provenance/pending qualification/execution plan. Independently reopen and verify. Record all pending official-ID/license/snapshot/provider/budget blockers. Do not start Phase 1 automatically.

## Task 10: Documentation

Append Phase 0 commands/counts/hashes/pending blockers to WORKFLOW and update CLAUDE/memory. State explicitly that no model/API call or new scientific result exists.

## Verification

```bash
python -m py_compile mub/vnext/post_core/*.py scripts/vnext_*post_core*.py
python -m pytest tests/vnext/test_post_core_*.py -q
git diff --check
git status --short
```
