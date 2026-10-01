# Provisional Native Multi-Object Contract

This document records the bounded, additive foundation for a future native multi-object manager evaluation. It is a contract and fake qualification seam only; it is not a Qdrant result, benchmark release, or scientific finding.

The contract uses the canonical object identity `(namespace, entity, attribute, subkey)`. `object_type` remains classification metadata and is never part of identity, replay linkage, or scoring. Ordered multi-object mutations carry event IDs, per-object outcomes, provider entry IDs, source-event linkage, explicit no-effect statuses, atomicity, and provider-versus-extracted provenance. Native retrieval traces preserve provider-owned payloads, IDs, order, rank, and scores without local reranking, deduplication, filtering, target injection, or an authoritative local store.

The in-process fake provider used by temporary tests is deliberately non-scientific and cannot establish native admission, external-system capability, accuracy, or answer quality. Qualification is fail-closed: package import, a fake provider, missing direct observations, declared-versus-observed capability overclaims, failed visible-only checks, missing traces, failed isolation/cleanup, nondeterministic repeats, or provider calls during qualification produce typed `BLOCKED`/`NOT_REQUESTED` outcomes.

Direct Qdrant source/runtime/API evidence, a native canary, a full run, answer replay, and a release package do not yet exist. No Qdrant dependency, executable, network call, model call, GPU run, or provider call is introduced by this foundation.
