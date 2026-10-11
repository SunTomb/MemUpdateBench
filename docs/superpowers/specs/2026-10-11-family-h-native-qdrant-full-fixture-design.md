# Family-H Native Qdrant Full-Fixture Design

> Design record for the bounded Family-H native multi-object expansion. This document preserves the requirement that source, state/retrieval, answer, runtime and release evidence remain separate.

## Scope

Run the existing published Family-H independent release only: three audited software tasks (Rust, CPython, Kubernetes), 27 ordered events, no Qwen answer replay, no new source intake and no modification of the published release. The result is a sibling diagnostic/qualification package, not a Core/Pilot update and not a broad native-manager release.

## Architecture

A Family-H adapter reads `data/vnext/family_h_independent/v1_published/release_index.json` and `tasks.jsonl`, validates the release artifact index and task/source hashes, and projects each event into the existing `NativeMultiObjectMutationRequestV3` contract. It never passes gold answers, hidden selectors or expected state to Qdrant.

The runner reuses the verified direct-Qdrant lifecycle: pinned server/client/runtime identity, loopback API, owned collection and temporary roots, process-group cleanup, per-task namespace isolation, guard namespace collateral checks, direct Qdrant mutation/retrieval witnesses, and no local authoritative state. Each task uses one isolated collection and runs all 27 events in source order, then a public retrieval query derived from the task query. Answer execution is `NOT_RUN`.

An independent attestor reopens the source release, run inputs, provider observations and result rows. It recomputes state-step matches, source-event linkage, final-state matches, retrieval object/value/source linkage, stale same-slot burden, collection isolation and cleanup. It does not trust runner metric fields without trace recomputation.

## Evidence and release states

The package has explicit states:

- `BLOCKED`: technical/runtime/source/cleanup failure or incomplete fixture;
- `DIAGNOSTIC_ONLY`: observations exist but a provenance or release-level condition blocks promotion;
- `READY_FOR_REVIEW`: all bounded gates pass, but no scientific release is implied;
- `FINAL_APPROVED`: reserved for a later human/release decision and never emitted by the runner.

The first implementation may emit `DIAGNOSTIC_ONLY` or `READY_FOR_REVIEW`, but always sets `scientific_release_allowed=false` and `answer_metrics=null`. It binds the source release index, source manifest, runtime identity, preparation contract, direct provider witness counts, and every artifact hash. Raw server logs and private absolute paths are excluded from the compact public package.

## Failure and cleanup policy

Unsupported or not-requested answer work remains typed and null. Technical runtime failures are `BLOCKED`, not accuracy zero. If any task fails, later tasks are `NOT_RUN`; the owned Qdrant server, worker, collection and temporary root are cleaned before publication. Ambiguous collection ownership never authorizes deletion. Existing Family-H, NOAA/BEA, Core/Pilot, Qwen and Qdrant v13 roots are immutable.

## Verification

Offline tests cover source projection, event/request binding, public-input redaction, exact four-part identity, provider-witness recomputation, cleanup/ownership rules, and tampered artifact rejection. A bounded Tang-3 run is executed only after these tests pass. The final package is independently reopened from its own index and source bindings.
