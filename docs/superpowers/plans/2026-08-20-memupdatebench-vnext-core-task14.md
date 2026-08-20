# MemUpdateBench vNext Core Task 14 Implementation Plan

> **For agentic workers:** execute task-by-task with TDD and independent specification/code-quality reviews. Do not modify immutable Core or existing Task 9–13 artifacts.

**Goal:** Implement and execute the only gate permitted to decide bounded overall Core `FINAL_APPROVED`.

**Architecture:** A read-only evidence loader builds a typed evidence graph and structural report. A fresh current-root verifier derives a historical attestation. Five canonical artifacts are atomically published in acyclic hash order and reopened into an immutable verified wrapper.

**Tech stack:** Python 3.10+, Pydantic v2, existing strict-v3 validators, SHA-256/canonical JSON, `publish_files_atomically`, pytest.

---

## Task 1: Freeze design and plan

**Files:**
- Create `docs/superpowers/specs/2026-08-20-memupdatebench-vnext-core-task14-design.md`
- Create `docs/superpowers/plans/2026-08-20-memupdatebench-vnext-core-task14.md`

- [ ] Check for placeholders, circular hashes, mutable caller approval, NFS/local ambiguity, new-experiment scope, and Python 3.10 compatibility.
- [ ] Commit the two documents before implementation.

## Task 2: Strict contracts

**Files:**
- Create `mub/vnext/release/__init__.py`
- Create `mub/vnext/release/task14_contracts.py`
- Create `tests/vnext/test_core_task14_contracts.py`

- [ ] Write RED tests for strict lowercase hashes, immutable models, exact artifact ordering, duplicate/foreign graph nodes, invalid edge types, forbidden claim evidence classes, status derivation, attestation hash, manifest ordering, and non-self-hashing index.
- [ ] Implement artifact refs, root snapshots, evidence nodes/edges/graph, checks/findings/exclusions, structural report, verification attestation, manifest/index, and immutable verified wrapper.
- [ ] Run focused tests and commit.

## Task 3: Authenticated source inventory

**Files:**
- Create `mub/vnext/release/task14_sources.py`
- Create `tests/vnext/test_core_task14_sources.py`

- [ ] Write RED tests for missing/extra files, same-size replacement, symlink/reparse, duplicate aliases, wrong root kind, stale root snapshot, dirty/untrusted revision, and NFS staging mislabeled final.
- [ ] Implement bounded canonical JSON/JSONL loaders, regular-single-link checks, recursive sorted tree snapshots, trusted revision checks, and exact frozen Task 9–13 hash anchors.
- [ ] Reuse Core candidate/audit and Task 13 validators rather than cloning them.
- [ ] Run focused tests and commit.

## Task 4: Evidence graph and structural report

**Files:**
- Create `mub/vnext/release/task14_review.py`
- Create `tests/vnext/test_core_task14_graph.py`
- Create `tests/vnext/test_core_task14_approval.py`

- [ ] Write RED tests for all required nodes/edges and one tamper for each Core/Task 9–13 layer.
- [ ] Add explicit exclusion tests for fake-offline, `slot_direct`, Pilot deterministic outputs, Mem0 admission-as-accuracy, API probes, unsupported-as-zero, and NFS staging-as-final.
- [ ] Implement deterministic evidence graph, check matrix, findings/exclusions, and derived `READY_FOR_VERIFICATION`/`NOT_APPROVED` report.
- [ ] Verify Task 12 completeness and Task 13 index/receipt/statistic/ledger/case closure without rerunning science.
- [ ] Run focused tests and commit.

## Task 5: Attestation and atomic publication

**Files:**
- Create `mub/vnext/release/task14_publish.py`
- Create `tests/vnext/test_core_task14_atomic.py`

- [ ] Write RED tests for acyclic hash construction, stale source before publication, destination race, existing root, path overlap, fault at every staged file, crash recovery, manifest/index tamper, and reopen failure.
- [ ] Build report/graph → attestation → manifest → non-self-hashing index.
- [ ] Publish exact five files with `publish_files_atomically`, overwrite disabled, staged validators, and source rechecks in `pre_publish`.
- [ ] Reopen and return only an immutable `VerifiedCoreFinalRelease`; no persisted report may approve itself.
- [ ] Run focused tests and commit.

## Task 6: Production CLI

**Files:**
- Create `scripts/vnext_review_core_task14.py`
- Create `tests/vnext/test_core_task14_cli.py`

- [ ] Write RED tests for exact safe flag surface, `allow_abbrev=False`, execute gate, prohibited model/provider/token/API/fake/override flags, exit codes, no-overwrite, and no model invocation.
- [ ] Implement explicit evidence paths, expected Task 13 index hash, trusted revision, review ID, output root, and `--execute`.
- [ ] Print decision/index/attestation/output only after reopen verification.
- [ ] Run focused tests and commit.

## Task 7: Full local and compatibility gate

- [ ] Run `py_compile` for every Task 14 module/CLI.
- [ ] Run all Task 14 tests; split long suites into disjoint nodes and report skips separately.
- [ ] Run targeted Core candidate/audit/stage, atomic publisher, Task 12 matrix, and Task 13 final-root regressions.
- [ ] Run `git diff --check` and verify unrelated untracked files and immutable roots are untouched.
- [ ] Obtain independent `SPEC_COMPLIANT` and `CODE_QUALITY_APPROVED`; fix confirmed findings and rerun exact tests.

## Task 8: Real final review execution

- [ ] Inventory and hash the minimum authoritative local/remote evidence; copy remote evidence read-only without rebinding hashes.
- [ ] Create a clean detached runtime and reverify all fixed anchors.
- [ ] Execute Task 14 with no model/provider/GPU path into an absent local output root.
- [ ] Independently reopen the five-file root, rehash every file, reconstruct the graph, verify attestation/current roots, and compare producer and independent decisions.
- [ ] If any check fails, retain an honest `NOT_APPROVED`; never force approval.

## Task 9: Final documentation and memory

**Files:**
- Modify `WORKFLOW.md`
- Modify `CLAUDE.md`

- [ ] Record motivation, commands, inputs, hashes, checks, findings, exclusions, final root/index/attestation, platform boundaries, limitations, and decision.
- [ ] Update `CLAUDE.md` only to the verified Task 14 state.
- [ ] Commit documentation separately and update persistent memory.

## Verification commands

```bash
python -m py_compile \
  mub/vnext/release/task14_contracts.py \
  mub/vnext/release/task14_sources.py \
  mub/vnext/release/task14_review.py \
  mub/vnext/release/task14_publish.py \
  scripts/vnext_review_core_task14.py

python -m pytest \
  tests/vnext/test_core_task14_contracts.py \
  tests/vnext/test_core_task14_sources.py \
  tests/vnext/test_core_task14_graph.py \
  tests/vnext/test_core_task14_approval.py \
  tests/vnext/test_core_task14_atomic.py \
  tests/vnext/test_core_task14_cli.py -q

git diff --check
git status --short
```
