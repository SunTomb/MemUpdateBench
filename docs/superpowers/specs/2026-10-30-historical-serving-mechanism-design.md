# Historical Serving Mechanism Design

**Status:** Approved bounded design; offline implementation is in progress. No database/model execution or smoke trajectory has started under this protocol.

## Context and research boundary

The Graphiti-only development screen is formally `INCONCLUSIVE`: current answers were 8/8 at update counts 1/4/16 in both native and reversed evidence order, while historical native answers were 2/8, 3/8 and 0/8. A read-only diagnosis found seven observed target-edge invalidation transitions on non-target events, but the prior run did not capture a complete final-state export. Mem0 is blocked and is not a comparator in this protocol.

This design does not revive the stopped current-value degradation claim. It tests one narrower hypothesis:

> When an unrelated entity receives an update with the same relation type, does native Graphiti invalidate a target entity's edge, and can a historical query distinguish a fact that remains valid in provider state from a fact omitted by native temporal retrieval?

This is a mechanism qualification, not a benchmark release, leaderboard, validation run, or paper-acceptance claim.

## Scope and budget

Use exactly the three unused Graphiti smoke trajectory slots. Each case is one fresh ingestion trajectory; a technical failure consumes its slot and is not retried. Do not run Mem0, held-out validation, paid APIs, additional models, or a broader factorial. Keep all code, temporary files, database data, model ledgers, logs and sockets under `/NAS/yesh`.

Use the pinned Graphiti commit `5c27edfa8d3e0f49b2571bcebb4261bcc89ee037`, Neo4j 5.26.0, Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, MiniLM revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, constrained schema decoding, and native `EDGE_HYBRID_SEARCH_RRF`. Reuse the already qualified environment and keep the aggregate study limits unchanged.

## Three cases

### Case 0 — target-only control

In one owned namespace, ingest an initial target fact followed by two same-object replacements. No other entity uses the target relation. After every episode, capture the provider return and a direct Neo4j snapshot. Run a current query and an as-of historical query.

Purpose: establish the target's normal native temporal trajectory without cross-entity interference.

### Case 1 — cross-entity same-relation control

In a fresh namespace, ingest an initial target fact, then update a different entity using the same relation type while never updating the target. After each episode capture the provider return and direct Neo4j state.

Purpose: test whether the target edge's `invalid_at`, `expired_at`, fact or UUID changes after a non-target event. A transition counts only when the same provider-owned edge UUID was valid in one snapshot and invalidated in a later snapshot. A repeated observation is not a new transition.

### Case 2 — historical serving control

In a fresh namespace, ingest an initial target fact and later target updates. At a public as-of time for an earlier version, capture:

1. direct Neo4j state, including all owned `RELATES_TO` edges and endpoint names;
2. Graphiti native time-filtered search result, preserving UUID, score, rank, order and all temporal fields;
3. the same evidence readout using the canonical v3 answer parser.

Purpose: distinguish (a) the historical edge absent from provider state, (b) the edge present and time-valid in provider state but omitted by native search, and (c) the edge returned but misread by the readout model.

## Evidence layers and no-leakage rules

Every case records the following private, hash-bound layers:

- **Public event:** event ID, visible text, public reference time and owned namespace.
- **Extraction observation:** Graphiti schema name, call ID, response hash, parsed-shape summary, exact token/finish accounting; raw prompt/response remains private and is not copied to compact artifacts.
- **Conflict transition:** previous/current edge UUID, endpoint UUID/name witnesses, validity fields and event positions. It does not assert semantic correctness from fact text alone.
- **Direct state witness:** every owned Entity/Episodic node and `RELATES_TO` edge after each event, including `valid_at`, `invalid_at`, `created_at`, `expired_at`, `episodes`, endpoint IDs and redacted/hash-only vector information.
- **Native retrieval:** query text, public as-of time, native `SearchFilters`, UUIDs, scores, ranks, order and temporal fields.
- **Readout:** prompt hash, exact model identity, typed parsed answer, finish status and usage.

Provider inputs are constructed only from visible event/query data. No gold action, gold object key, gold answer, hidden selector or local latest-value filter may enter the Graphiti write/search path. Direct Neo4j queries are provider-owned state witnesses, not local replacement state. Native results are never locally sorted, deduplicated or repaired.

## Attribution and result labels

- `BLOCKED`: database, extraction accounting, direct state query, time parsing, budget or owned cleanup fails. Do not score as zero.
- `NO_DISTINGUISHING_SIGNAL`: target-only and cross-entity cases show no complete contrast, and historical case does not separate provider state from native retrieval.
- `CROSS_ENTITY_INVALIDATION_LEAD`: Case 1 shows a target edge transition after a non-target event while Case 0 does not, with complete endpoint and time witnesses.
- `HISTORICAL_SERVING_LEAD`: Case 2 shows a time-valid historical edge in the direct provider snapshot but not in native search, or shows the edge returned but a readout error.
- `CROSS_ENTITY_AND_HISTORICAL_LEAD`: both leads satisfy their evidence requirements.

None of the lead labels is a scientific GO or general Graphiti claim. A lead only authorizes a separately designed confirmation protocol; no validation data is consumed automatically. A blocked or no-signal outcome stops this historical mechanism branch unless the user approves a new design.

## Publication and cleanup

Publish a sibling no-replace root:

```text
results/vnext/version_transfer_historical_mechanism_20261030_v1
```

It contains only redacted/hash-bound manifests, per-case receipts, typed errors, counts and the decision. Raw provider payloads, prompts, credentials, private absolute paths and unrestricted model output remain on the private `/NAS/yesh` run root. Each case uses a unique namespace and the runner verifies final empty state, client close, database shutdown, GPU process absence and port release. No frozen Core/Pilot/Family-H release or earlier Graphiti evidence is changed.

The case ledger must report all attempted/completed event writes, retrievals, internal LLM/embedding calls, retries, tokens, finish reasons, elapsed time and storage. The runner writes explicit `NOT_RUN` rows for cases after a terminal failure and never resumes a consumed case.

## Completion criteria

The protocol is complete only when all three case terminals, direct-state witnesses, native search traces, readout records, cleanup checks and the compact decision index are independently reopened. If any of those are unavailable, the result is `BLOCKED`. Completion does not imply a paper result; it supplies a decision about whether a separate historical-serving confirmation is worth designing.
