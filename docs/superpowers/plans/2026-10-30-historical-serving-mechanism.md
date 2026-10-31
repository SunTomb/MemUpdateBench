# Historical Serving Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a three-case, source-bound Graphiti protocol that distinguishes cross-entity invalidation, provider-state retention, native temporal retrieval, and readout failure using only the three remaining smoke trajectory slots.

**Architecture:** Keep the existing v3 task packet, Graphiti SDK, Neo4j runtime and readout contracts unchanged. Add a thin direct-state witness layer around the owned Neo4j database, a private extraction-observation recorder, and a no-gold mechanism runner. Publish only redacted/hash-bound case receipts and a typed `BLOCKED`/`NO_DISTINGUISHING_SIGNAL`/mechanism-lead decision; never promote a lead to validation or scientific GO.

**Tech Stack:** Python 3.13 Graphiti 0.31.2, Neo4j Community 5.26.0, Qwen2.5-7B-Instruct with `lm-format-enforcer` 0.11.3, MiniLM-L6-v2, Pydantic-backed v3 contracts, pytest, `/NAS/yesh`-local tmux and owned loopback services.

---

## Task 1: Freeze mechanism protocol and source-bound case packet

**Files:**
- Create: `configs/vnext/version_transfer_historical_mechanism_v1.json`
- Create: `scripts/vnext_prepare_historical_mechanism.py`
- Create: `tests/vnext/test_historical_mechanism_preparation.py`
- Modify: none of the frozen Core/Pilot/data roots

- [x] **Step 1: Write the failing tests**

Add tests for:

```python
def test_builds_exactly_three_cases_with_distinct_roles():
    cases = build_cases(code_revision="fixture-revision")
    assert [case.role for case in cases] == [
        "target_only_control", "cross_entity_same_relation", "historical_serving"
    ]
    assert all(case.model_calls == 0 for case in cases)


def test_packet_rejects_rebinding_or_replacement(tmp_path):
    receipt = publish_packet(tmp_path / "packet", code_revision="fixture-revision")
    assert read_packet(tmp_path / "packet", receipt["index_sha256"])["status"] == "PREPARED_NOT_EXECUTED"
    with pytest.raises(FileExistsError):
        publish_packet(tmp_path / "packet", code_revision="fixture-revision")


def test_packet_does_not_include_gold_actions_in_public_episode_payload():
    case = build_cases(code_revision="fixture-revision")[1]
    assert all("gold" not in json.dumps(event.public_payload).lower() for event in case.events)
    assert all("selector" not in json.dumps(event.public_payload).lower() for event in case.events)
```

Run:

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_historical_mechanism_preparation.py
```

Expected: collection fails because the new preparation module is absent.

- [x] **Step 2: Implement the smallest deterministic preparation packet**

Reuse `MemUpdateTaskV3`, `MemoryEventV3`, `MemoryQueryV3`, `replay_task_v3`, `resolve_query_v3`, `visible_event_input`, and `visible_query_input`. Do not add another task dictionary. Define exactly three cases:

```text
target_only_control:
  initial target fact -> target replacement -> target replacement
cross_entity_same_relation:
  initial target fact -> different entity, same relation -> different entity, same relation
historical_serving:
  initial target fact -> target replacement -> target replacement; current + as-of query
```

Every case must carry public UTC event times, an owned namespace template, source revision, task hash, and `PREPARED_NOT_EXECUTED` status. The packet must include an exact artifact index, SHA-256 bytes, no-replace output, path/reparse checks, zero execution counters, and `scientific_release_allowed=false`. Reject outputs inside Core/Pilot or unrelated result roots.

Exercise every generated public query through the existing `GraphitiNativeEpisode.retrieve` boundary with an injected fake client before accepting the packet. Its required `As of <UTC>, ` prefix must be in the public query itself, never added from a hidden selector at runtime. Anchor current queries to the final public event time, including the cross-entity case; target-only and historical-serving cases each include current and historical queries, giving exact query counts of 2, 1 and 2.

- [x] **Step 3: Run the focused tests**

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_historical_mechanism_preparation.py
```

Expected: all preparation tests pass; no model, database or provider process starts.

- [x] **Step 4: Run deterministic replay and source audit**

```bash
PYTHONPATH=. python -B scripts/vnext_prepare_historical_mechanism.py \
  --output-root results/vnext/version_transfer_historical_mechanism_preparation_20261030_v2 \
  --code-revision <committed-revision>
PYTHONPATH=. python -B scripts/vnext_prepare_historical_mechanism.py \
  --output-root results/vnext/version_transfer_historical_mechanism_preparation_20261030_v2 \
  --verify-index-sha256 <printed-index-sha256>
```

Expected: exactly three cases, no execution, and independent task/query replay passes.

Completed at source `db8c8585d90b312db7d5f692882b2a52bc988f76`: 63 preparation/native-adapter tests passed; v2 reopened with index SHA-256 `b98b815b500a6e7594be39b4be78b233550c1f78f9acbee701ac1a106c027155`. The unchanged v1 packet is a superseded integration diagnostic and must not be executed.

- [x] **Step 5: Commit the offline preparation unit**

```bash
git add configs/vnext/version_transfer_historical_mechanism_v1.json scripts/vnext_prepare_historical_mechanism.py tests/vnext/test_historical_mechanism_preparation.py
git commit -m "Prepare historical serving mechanism cases" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not stage `docs/WORKSPACE_LAYOUT.md` or the unrelated deleted test file.

## Task 2: Add direct Neo4j state witnesses and private extraction observations

**Files:**
- Create: `mub/vnext/external/providers/graphiti_state_witness.py`
- Create: `mub/vnext/external/providers/graphiti_observation_recorder.py`
- Create: `tests/vnext/test_graphiti_state_witness.py`
- Create: `tests/vnext/test_graphiti_observation_recorder.py`

- [ ] **Step 1: Write state-witness tests first**

Use a fake `Neo4jDriver.execute_query` and assert the generated query is namespace-bound:

```python
def test_snapshot_is_group_scoped_and_preserves_temporal_fields():
    snapshot = witness.snapshot("owned-case")
    assert "e.group_id = $group" in fake.last_query
    assert snapshot.edges[0].valid_at == "2025-01-01T00:00:00Z"
    assert snapshot.edges[0].invalid_at is None
    assert snapshot.edges[0].fact_embedding_sha256 is not None


def test_cleanup_verifies_zero_nodes_and_edges():
    witness.delete_namespace("owned-case")
    assert witness.verify_empty("owned-case") is True
```

- [ ] **Step 2: Implement the direct witness query**

`GraphitiStateWitness.snapshot(namespace)` must query only owned `Entity`, `Episodic`, and `RELATES_TO` records and return canonical redacted rows containing UUIDs, endpoint UUIDs/names, fact text hash, `valid_at`, `invalid_at`, `created_at`, `expired_at`, group ID, episode IDs, and vector hashes. Never export raw embeddings or private provider payloads. `delete_namespace` must issue a parameterized group-scoped deletion and `verify_empty` must query the same namespace after deletion.

- [ ] **Step 3: Write extraction recorder tests**

```python
def test_recorder_keeps_schema_and_usage_but_not_prompt_or_raw_output():
    recorder = ObservationRecorder()
    recorder.record(role="graphiti", schema_name="EdgeDuplicate",
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    raw_output='{"duplicate_facts":[],"contradicted_facts":[]}')
    row = recorder.rows[0]
    assert row["schema_name"] == "EdgeDuplicate"
    assert row["output_sha256"]
    assert "raw_output" not in row and "prompt" not in row
```

- [ ] **Step 4: Implement the private recorder**

Wrap only the local constrained LLM transport callback. Record monotonic attempt ID, role, schema class, finish reason, exact usage, output hash and parsed top-level keys. Keep raw prompts/responses only in a private `/NAS/yesh` run root if the runtime requires them; compact receipts must contain hashes and counts only. A missing or malformed recorder event is a qualification error, not an empty successful extraction.

- [ ] **Step 5: Run focused tests and commit**

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_graphiti_state_witness.py tests/vnext/test_graphiti_observation_recorder.py
```

Expected: all fake-boundary tests pass without importing a live Graphiti DB. Commit only these files after the tests pass.

## Task 3: Implement the three-case source-bound smoke runner

**Files:**
- Create: `scripts/vnext_run_historical_mechanism.py`
- Create: `tests/vnext/test_historical_mechanism_runner.py`
- Reuse without changing: `mub/vnext/external/providers/graphiti_native_episode.py`, `scripts/vnext_run_version_transfer_decision.py`, `scripts/vnext_serve_version_transfer_local.py`, `mub/vnext/runtime/version_transfer_budget.py`, `mub/vnext/runtime/version_transfer_embedding.py`

- [ ] **Step 1: Write runner tests before implementation**

Test a fake Graphiti client/driver with exactly three cases:

```python
def test_non_target_event_does_not_get_gold_target_selector():
    runner.run_case(fake_runtime, case)
    assert all("target_object_keys" not in call.provider_payload for call in fake_runtime.calls)
    assert all("gold" not in call.provider_payload.lower() for call in fake_runtime.calls)


def test_case_writes_not_run_rows_after_failure():
    result = runner.run_case(failing_runtime, case)
    assert result.status == "BLOCKED"
    assert result.event_attempts > result.event_completions
    assert result.retrieval_attempts == 0
    assert result.namespace_empty_after_cleanup is True


def test_runner_never_retries_a_consumed_case():
    with pytest.raises(BudgetBlocked):
        runner.run_case(budget_exhausted_runtime, case)
```

- [ ] **Step 2: Implement the one-shot runner**

The runner must:

1. Reopen and verify the preparation index and committed source hash.
2. Admit exactly three cases and fresh owned namespaces.
3. Start one owned Neo4j process and one owned constrained local model/embedding process group only after current GPU admission.
4. Ingest only `visible_event_input` data through native `GraphitiNativeEpisode`.
5. After every event, write provider return and `GraphitiStateWitness.snapshot(namespace)`.
6. Run public-time-filtered native retrieval for current/history queries.
7. Build native and reversed readouts from the exact same returned edges; build oracle evidence only as separately labeled diagnostic data.
8. Save per-call ledgers, timestamps, source hashes, request/result hashes and cleanup receipts.
9. On any case failure, mark later cases `NOT_RUN`, finish cleanup, and never retry the consumed case.
10. Publish only a compact no-replace terminal package after the run; raw files remain private on `/NAS/yesh`.

Use only the three remaining smoke slots. Do not run the Mem0 comparator or validation set in this plan.

- [ ] **Step 3: Run fake runner tests and the static CLI check**

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_historical_mechanism_runner.py
action='python -B scripts/vnext_run_historical_mechanism.py --help'; eval "$action" >/dev/null
```

Expected: all fake-boundary tests pass; no live provider/model starts.

- [ ] **Step 4: Commit the runner only after fake tests pass**

```bash
git add scripts/vnext_run_historical_mechanism.py tests/vnext/test_historical_mechanism_runner.py
git commit -m "Add bounded historical serving smoke runner" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

## Task 4: Add offline mechanism classification and typed decision output

**Files:**
- Create or reuse: `scripts/analyze_version_transfer_mechanism.py`
- Create: `scripts/vnext_finalize_historical_mechanism.py`
- Create: `tests/vnext/test_historical_mechanism_analysis.py`
- Create: `tests/vnext/test_historical_mechanism_decision.py`
- Output: `results/vnext/version_transfer_historical_mechanism_20261030_v1/`

- [ ] **Step 1: Write classification tests**

```python
def test_cross_entity_lead_requires_direct_state_transition():
    assert classify(case0, case1, case2).decision == "CROSS_ENTITY_INVALIDATION_LEAD"


def test_missing_final_state_is_blocked_not_storage_failure():
    assert classify_without_state_witness(case).decision == "BLOCKED"


def test_historical_retrieval_miss_is_distinct_from_readout_error():
    assert classify_history(state_valid=True, native_returned=False, readout=None).decision == "HISTORICAL_SERVING_LEAD"
    assert classify_history(state_valid=True, native_returned=True, readout_correct=False).decision == "HISTORICAL_SERVING_LEAD"
    assert classify_history(state_valid=True, native_returned=True, readout_correct=False).subtype == "readout"
```

- [ ] **Step 2: Implement typed classification**

Use only direct snapshot/retrieval/readout records. Emit exactly one decision:

```text
BLOCKED
NO_DISTINGUISHING_SIGNAL
CROSS_ENTITY_INVALIDATION_LEAD
HISTORICAL_SERVING_LEAD
CROSS_ENTITY_AND_HISTORICAL_LEAD
```

The output must state `scientific_release_allowed=false`, `validation_tasks=0`, `independent_signer=false`, sample/case denominators, all NOT_RUN rows, resource totals, source/packet/terminal hashes and a short non-causal interpretation. It must never write raw prompts, raw model output or private absolute paths.

- [ ] **Step 3: Run the analysis tests**

```bash
python -B -m pytest -q -p no:cacheprovider tests/vnext/test_historical_mechanism_analysis.py tests/vnext/test_historical_mechanism_decision.py
```

Expected: typed leads and blockers pass; no live execution.

## Task 5: Execute the one-shot cluster protocol and independently audit it

**Files:**
- Create: `scripts/vnext_audit_historical_mechanism.py`
- Create: `tests/vnext/test_audit_historical_mechanism.py`
- Modify: `WORKFLOW.md` only after terminal evidence exists

- [ ] **Step 1: Run the exact source-bound CLI admission on Tang-2**

Run only under `/NAS/yesh` with a new NAS-local tmux socket and output root. Verify current GPU headroom immediately before model load. Preserve all six consumed native smoke trajectory attempts (five Graphiti, one Mem0); reserve only the remaining three of nine for this protocol. Do not run Mem0 or validation.

- [ ] **Step 2: Reopen terminal evidence before interpreting it**

Verify exact counts for each case, event attempts/completions, retrieval attempts/completions, direct-state snapshots, native retrieval IDs/scores/temporal fields, extraction ledgers, readout usage and cleanup. If the run exits without a terminal receipt, stop with `BLOCKED`.

- [ ] **Step 3: Run the independent audit without opening the DB or model**

The audit must verify every artifact hash, source and packet binding, case order, direct-state namespace, event/time binding, edge transition logic, public-time interval calculation, readout binding and cleanup. It must not infer missing final state or repair provider data.

- [ ] **Step 4: Publish compact no-replace decision metadata**

Write only redacted `decision.json` and `index.json` under `results/vnext/version_transfer_historical_mechanism_20261030_v1`. Report the exact decision and never upgrade a mechanism lead to `scientific_release_allowed=true`.

- [ ] **Step 5: Update findings and commit the completed protocol unit**

Add a dated `WORKFLOW.md` entry and, if the result is complete, append a concise case-bound section to `paper/manuscript_sections/version_transfer_dev_findings_draft.md`. Commit the new protocol, audit, decision metadata and documentation while excluding `docs/WORKSPACE_LAYOUT.md` and the unrelated deleted test file.

## Verification and stopping rules

- Before any live run: all focused tests pass; task/packet hash pins reopen; Graphiti qualification receipt and source map match; model/embedding inputs are under `/NAS/yesh`; no owned port/process is active; the three smoke slots are reserved exactly.
- A provider failure, schema failure, missing direct state, missing cleanup or budget exhaustion is `BLOCKED`, not zero accuracy.
- The three cases can create a historical mechanism lead, but cannot create a paper-level GO or justify running the held-out validation set. Any broader confirmation requires a new design and explicit approval.
- If the result is `NO_DISTINGUISHING_SIGNAL`, stop the historical mechanism branch. If it is a lead, write the next hypothesis but do not expand the Cartesian product automatically.
- Preserve all frozen releases and previous smoke/qualification receipts; never overwrite or rebind them.
