# Cross-Domain Qdrant v13 Evidence

**Date:** 2026-10-01
**Evidence class:** direct external-manager state/retrieval evidence
**Status:** `COMPLETE`

## Scope

This bounded run uses the immutable formal task releases:

- NOAA index SHA-256: `738f46932d8e2e8c26da0a2820dc67c7b0bad9aa43659f4be266f22f0fc8ee3f`
- BEA index SHA-256: `5525357436510b0c4e89efa53cdb6caf551c59234171c71bb97563e4c641790a`

It evaluates two source trajectories/tasks, one NOAA and one BEA, with two repetitions per task. The public manager input contains only structured events and visible query ID/text. Gold actions, hidden selectors, version history and gold evidence are not passed to the manager.

## Run binding

```text
remote root:          /NAS/yesh/MemUpdateBench/external/cross_domain_state_20260930_v13
local package:        results/vnext/cross_domain_state_package_20260930_v13
package source SHA:   145a19b059aa3373de39e09a2581865e9910816739c530da09c20e3deb9a8c01
package archive SHA:  30e2b561575b7ea65616c0ca2f5c0b13adc5f72fc4076bb9c445e8503caff745
artifact index SHA:   2b2b9dbfd7703f0a91cc66a9267344837f4a892bfac89e5e2db25a00b836a60a
collection audit SHA: 5188cbbb7fbb3aacbf3a25994a75273060c9a2ab2857c180f3eacf7be3326eca
```

Runtime binding:

```text
Qdrant version:       1.19.0
server revision:      74f3e85b9473c62560006c043e13737ce6b48412
binary SHA-256:       abfe97e1d0225111dec2f048790428f151846c8a049eefc85328b6c9eccaf419
Python:               3.10.20
storage:              owned local ext4 temporary root
```

## Observed results

| Metric | Result |
|---|---:|
| base tasks | 2 |
| base events | 10 |
| executed mutations | 20 |
| retrieval requests | 4 |
| state step matches | 20/20 |
| source-event links | 20/20 |
| stale same-slot count | 0 |
| final state matches | 4/4 |
| typed-object retrieval matches | 4/4 |
| repetition equality | `true` |
| Qdrant server instances | 1 |
| worker HTTP calls | 100 |
| model loads/generations | 0/0 |
| provider-model calls/GPU calls | 0/0 |
| peak owned storage | 198,729,728 bytes |
| process/port/directory cleanup | all passed |

## Claim boundary

This is state/retrieval evidence for direct Qdrant CRUD and provider-returned retrieval over two realistic-source trajectories. It is not prompted-answer accuracy, answer-model capability, semantic embedding quality, native answer quality, statistical independence beyond the two trajectories, or broad benchmark closure. The formal task releases remain immutable and retain `scientific_release_allowed=false`; this result is a separate downstream evidence root.

Earlier v1–v12 attempts are retained as typed technical diagnostics. They are not accuracy zeros and are not merged into this result.

## Logical evidence package

The collected v13 result has a separate no-replace logical package:

```text
results/vnext/cross_domain_qdrant_evidence_package_20261001_v2
package manifest SHA-256:
3a4cf8b251b49cf19b2626b52cf4d7ee0471d50cd6c577bdb505bde73750267c
```

The package retains the authenticated result rows, summary, runtime, input bindings, HTTP accounting, source artifact index, collection audit and launcher/exit status. It does not copy raw NOAA/BEA source captures or task-release trees. Its metrics are descriptive counts over `cluster_unit=source_trajectory`, `cluster_n=2`; no confidence interval, broader independence claim, answer accuracy or scientific-release promotion is introduced.

A pure validator rechecks the exact artifact/index/audit/source pins, both immutable formal-release bindings, public-task projection, normalized state/retrieval traces, cleanup, repeat equality and no-model boundary. The validator and package integration gate passed 95 tests with one Windows symlink-permission skip.
