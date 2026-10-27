# Version Transfer Decision Implementation Plan

> **For agentic workers:** Execute inline with the `executing-plans` skill in the existing worktree. The approved design is `docs/superpowers/specs/2026-10-26-version-transfer-decision-design.md`. No new worktree or per-file review agents.

**Goal:** Produce a bounded, provenance-backed GO/PIVOT/INCONCLUSIVE/BLOCKED investment decision, not a predetermined Graphiti failure.

**Architecture:** Canonical v3 synthetic tasks and replay stay evaluator-only. Native Graphiti and Mem0 consume public episodes; raw temporal evidence survives retrieval and readout. Separate preparation, runtime qualification and scientific decision artifacts.

**Tech Stack:** Python/Pydantic v3 contracts, pytest, pinned Graphiti/native graph database, native Mem0 inference, existing offline cluster models.

---

## Batch 1 — Offline design, preparation and screening

**Files:**
- `configs/vnext/version_transfer_decision_v1.json`
- `scripts/vnext_prepare_version_transfer_decision.py`
- `scripts/vnext_analyze_version_transfer_decision.py`
- `tests/vnext/test_version_transfer_decision.py`
- `tests/vnext/test_version_transfer_analysis.py`

- [x] Write tests for 48 tasks, 16 paired semantic cores, 17 public events per task, 8/8 split and four small smoke tasks.
- [x] Run red: absent preparation module raises ImportError.
- [x] Implement `build_tasks(code_revision=...)`, `build_smoke(code_revision=...)`, `publish(...)`, and `read_packet(...)` using `MemUpdateTaskV3`, `replay_task_v3` and `resolve_query_v3`.
- [x] Implement pure `decide(...)` screening of declared observations; label it unverified-input screening, not authenticated execution evidence.
- [ ] Validate packet source/config binding, exact task/smoke membership and no-replace reopen; persist only a `PREPARED_NOT_EXECUTED` packet.

Run:
```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_version_transfer_decision.py tests/vnext/test_version_transfer_analysis.py
```
Expected: all focused tests pass; zero model/provider calls.

## Batch 2 — Public native boundaries

**Files:**
- `mub/vnext/external/providers/graphiti_native_episode.py`
- `tests/vnext/test_graphiti_native_episode.py`
- `scripts/vnext_run_version_transfer_decision.py`

- [x] Write fake-client tests: exact visible episode fields, namespace isolation, query-text-derived UTC filters and unchanged native result order.
- [x] Observe missing-module failure; implement `GraphitiNativeEpisode.ingest`, `retrieve`, `close`, and `validity_filter_factory`.
- [ ] Bind real pinned SDK types and native DB, without importing dependencies for offline preparation.
- [ ] Add shared runtime accounting of every internal model attempt, retry, token, timeout and owned cleanup. Reject a run that cannot account for SDK calls.
- [ ] Implement full-time native-order versus reversed-order evidence readout with identical item bytes and item set. No gold version selector.
- [ ] Implement native Mem0 `infer=True` public episode bridge; declare history unsupported when unavailable.

Run:
```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_graphiti_native_episode.py
```
Expected: fake-client tests pass, without scientific capability claims.

## Batch 3 — Cluster qualification

- [ ] Build an isolated pinned Graphiti environment and graph database under `/NAS/yesh`; log installation and native versions. No changes to frozen roots or other users' services.
- [ ] Verify model role identities, token limits, telemetry disabled, local-only networking and native schema parsing. Do not recycle old single-use Qwen generation permissions.
- [ ] Run four public capability trajectories per native system within the approved smoke ceiling. Gate on faithful observable execution, not perfect accuracy.
- [ ] Publish READY/BLOCKED/UNSUPPORTED qualification receipts. If blocked, finish offline deliverables and write the exact blocker; do not manufacture a scientific decision.

## Batch 4 — Bounded pilot and decision

- [ ] Execute dev first, using complete paired blocks; do not inspect frozen validation outcomes while tuning.
- [ ] Enter validation only if dev supports a meaningful native full-time hypothesis; obey total 10,000-attempt, token, GPU-hour, wall-time and storage budgets.
- [ ] Authenticate actual task/query IDs and outputs; preserve null and technical outcomes, current/history distinction, native timestamps and original order.
- [ ] Add fresh-ingestion replication within the approved allowance; identifiers alone are not replication evidence.
- [ ] Evaluate depth and depth-by-order contrasts, complete-pair thresholds, domain coverage and attribution before assigning a decision.
- [ ] Write `decision.json`, individual paired cases, unsupported/error ledger, resource usage and a concise research conclusion. Update `WORKFLOW.md` and manuscript positioning without fabricated Results.

## Completion boundary

The initial pure screening function does not yet implement the full authenticated decision pipeline or depth-by-order branch. A passing unit test is not model evidence. A full study completion requires all executed components above or an explicit technically verified BLOCKED gate, with all feasible offline deliverables completed. Preserve `docs/WORKSPACE_LAYOUT.md` outside staged changes.
