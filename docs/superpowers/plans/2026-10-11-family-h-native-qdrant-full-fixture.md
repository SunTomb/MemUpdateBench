# Family-H Native Qdrant Full-Fixture Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with tests first. Do not modify frozen releases or run a model; the only live runtime is the explicitly authorized CPU-only direct-Qdrant canary.

**Goal:** Run and independently attest the existing three-task Family-H release through the native Qdrant multi-object surface as a separate no-replace state/retrieval evidence package.

**Architecture:** Reuse existing Qdrant 1.19.0 lifecycle and native provider. Add a source-bound Family-H projection, an owned full-fixture worker, a trace-based attestor, and a compact release package. Keep answer metrics null and scientific release false.

**Tech Stack:** Python 3.10/3.14, Pydantic v3 contracts, Qdrant 1.19.0, pytest, SHA-256 manifests, transactional no-replace publication.

---

### Task 1: Family-H public projection

**Files:**
- Create: `scripts/vnext_prepare_family_h_native_qdrant.py`
- Create: `tests/vnext/test_prepare_family_h_native_qdrant.py`

- [ ] Write tests that load the existing release index, assert exactly 3 tasks/27 events, project every event to public native mutation requests, reject gold/answer fields, and reject source-index drift.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement the immutable source-bound projection and no-replace preparation packet with `source_release_index_sha256`, task hashes, public requests, and null answer metrics.
- [ ] Run focused tests; expected result: all pass.

### Task 2: Family-H direct-Qdrant worker

**Files:**
- Create: `scripts/vnext_run_family_h_native_qdrant.py`
- Create: `tests/vnext/test_run_family_h_native_qdrant.py`

- [ ] Add tests with a fake provider for 3 task ordering, 27 mutation rows, guard namespace isolation, cleanup failure, typed blocked/not-run rows, and zero model/provider-model accounting.
- [ ] Run focused tests to observe the intended failures.
- [ ] Implement worker input binding, per-task collection/runtime namespace, direct provider calls, source-event trace capture, final export, retrieval trace, cleanup and sanitized technical errors.
- [ ] Run focused tests; expected result: all pass.

### Task 3: Independent trace attestor

**Files:**
- Create: `scripts/vnext_attest_family_h_native_qdrant.py`
- Create: `tests/vnext/test_attest_family_h_native_qdrant.py`

- [ ] Write tests for exact artifact hashes, independently recomputed 27/27 state/link metrics, 3/3 final/retrieval metrics, stale count, provider witness counts, cleanup, tampered snapshot rejection, and scientific flag refusal.
- [ ] Run focused tests and confirm red state.
- [ ] Implement attestation from source release + worker trace only; do not trust stored aggregate metrics without recomputation.
- [ ] Run focused tests; expected result: all pass.

### Task 4: Offline integration gate and source package

**Files:**
- Modify: `WORKFLOW.md`
- Modify: `docs/vnext/native_multi_object_canary.md`
- Create: `docs/vnext/family_h_native_qdrant_full_fixture.md`

- [ ] Run source projection, worker fake, attestor, native provider and existing Qdrant contract tests.
- [ ] Stage a clean source packet with exact source/runtime/preparation pins and no raw private paths.
- [ ] Confirm no Core/Pilot/Family-H release files change.

### Task 5: Authorized Tang-3 run and package

**Files:**
- Create only under new no-replace roots: `results/vnext/family_h_native_qdrant_full_20261011_v1`

- [ ] Recheck Tang-3 runtime/binary/environment hashes and loopback port without starting a server.
- [ ] Run one owned Qdrant server and one Family-H full fixture; no model/provider API outside local Qdrant.
- [ ] Collect result root only after terminal cleanup; preserve BLOCKED/NOT_RUN if technical failure occurs.
- [ ] Run independent attestor, reopen index hashes, and publish compact package with answer metrics null and scientific release false.

### Task 6: Final verification and commit

- [ ] Run the final targeted gate plus clean-export import check.
- [ ] Inspect diff and ensure `docs/WORKSPACE_LAYOUT.md` remains unstaged.
- [ ] Commit with the required attribution and push the branch.
