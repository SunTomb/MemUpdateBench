# Workspace Layout

## One source checkout, separate local archives

The source tree and the local research workspace have different lifetimes. Keep executable source and small authenticated regression inputs in Git. Keep downloads, deployment copies, private diagnostic files and unfinished experiments out of commits, but preserve their provenance and recovery paths.

| Location | Purpose |
|---|---|
| `mub/` | Reusable benchmark contracts, adapters, validation and scoring |
| `scripts/` | Supported benchmark and release command-line entrypoints |
| `scripts/maintenance/` | Local inventory, reversible archival and checkout checks |
| `configs/` | Versioned configurations required by those entrypoints |
| `tests/` | Unit/regression tests and small, explicitly selected fixtures |
| `data/vnext/family_h_cross_domain_*/v1/` | Existing reviewed NOAA/BEA release bytes; immutable |
| `results/` | Run artifacts; ignored except individually reviewed committed packages |
| `external/` | Private source captures and third-party inputs; not application source |
| `_local/archive/` | Recoverable working-copy backups, deferred research and one-off helpers |
| `.claude/worktrees/` | Git-managed checkouts, not disposable cache directories |

Do not scatter new source bundles or staging directories beside `MemUpdateBench`. Use a named local output directory within the project instead. Do not commit virtual environments, credentials, raw provider payloads, source captures or model weights.

## Cleanup boundary, 2026-10-01

The cleanup operates in the existing development worktree. The primary checkout is still on an older `master`; the active research branch is `correct-langmem-evidence-boundary`. A successful commit on this branch does not automatically update the other checkout.

The host rejected writes from this isolated session to the primary checkout. Therefore the parent-workspace migration has been **planned but not applied**. No alternate shell, peer session or Git index was used to bypass that boundary. Perform the parent migration from a session opened in the primary checkout.

### Already preserved and organized in the development worktree

- A binary Git diff, baseline commit ID and working-file inventory were saved before restoration or movement.
- 283 visible working files were copied and SHA-256 checked in `_local/archive/20261001-cleanup-backup/working_files/`.
- The deferred research and scratch allowlist was archived at `_local/archive/worktree_cleanup_20261001_v1/`: 127 moved roots, 551 files, 16,660,048 bytes, **zero deleted files and zero removed Git worktrees**.
- The archive includes experiments not selected for the supported source snapshot. They are preserved work, not declared redundant or invalid evidence.
- `archive_receipt.json` records every original/destination path and file digest. `move_journal.jsonl` records completed moves; restoration refuses to overwrite newer work.
- The previously deleted Task 12 test was restored from Git because no corresponding replacement or rename was found. The original deletion remains represented in the backup patch.

Some dependency files initially archived during curation may subsequently be copied back into the supported source tree when a regression test establishes that they are required. Their archive copies remain unchanged. A bulk restore correctly refuses occupied original paths; reconcile those paths rather than overwrite them.

### Pending parent-workspace migration

The private plan is `_local/workspace_staging_plan_20261001.json`. Its intended destination is the primary project's `_local/archive/workspace_staging_20261001_v1/`.

The 16 proposed whole-root moves are:

- `.matched_depth_source_bundle_v2`, `_v3`, `_v4`;
- `MemUpdateBench_qdrant_regression_20260919_source_bundle_v2`, `_v3`, `_v4`, `_v15`;
- `family_h_answer_source_bundle_20260925_v1`;
- `.matched_depth_staging_v2`, `_v3`, `_v4`;
- `.mub_md_bundle_v4`;
- `MemUpdateBench_qdrant_v16_staging`;
- `mub-stage0-cli-rxy7bkmj`, `task12-audit-output`, `task12-dry-run-output` (empty at inventory time).

These are archive candidates, not deletion candidates. The archive tool re-inventories and checks bytes immediately before and after movement. It refuses Git repositories/worktrees, symlinks/reparse points, protected release/data roots, input changes and existing destinations.

Keep the following in place:

- `MemUpdateBench_releases`: contains the authoritative Core Task 14 and post-Core Phase 0 roots, plus qualification evidence;
- `MemUpdateBench_qualification_inputs`: retained qualification and canary provenance;
- `G-MSRA`, the parent `.git`, literature directories, reports and personal documents;
- every registered worktree. The inventory found 187 registered worktrees; none of the 16 proposed sibling moves is one of them.

A currently idle session is not proof that a checkout is disposable. No worktree or branch removal is authorized by this cleanup receipt.

## Reversible archive command

Run only after opening a session with write access to the intended source and destination roots:

```bash
python scripts/maintenance/organize_local_workspace.py archive --plan <reviewed-plan.json>
python scripts/maintenance/organize_local_workspace.py restore --receipt <archive-root>/archive_receipt.json
```

A plan lists exact paths. There is no wildcard deletion, `git clean`, branch deletion or force-overwrite operation. The output catalog contains local paths and is deliberately excluded from Git.

## Reproducibility and test data

Required native-Qdrant contracts/providers, both cross-domain release readers and the import-time configuration belong in the source commit, not in an untracked working directory. The reviewed NOAA/BEA release files are added to Git without changing their bytes. Explicit `.gitattributes` rules prevent checkout newline conversion from invalidating authenticated hashes.

`tests/vnext/fixtures/cross_domain_v13/` is an exact historical regression fixture copied from the collected v13 result. It contains normalized observations and hash-only HTTP/runtime metadata, not credentials, request/response bodies, model prompts or restricted source documents. It was screened before inclusion. Running tests over it does not start Qdrant or constitute a new benchmark run. Original data/results roots and their evidence classes remain unchanged.

Other Core/Pilot, model and cluster integration tests may require separately provisioned immutable data or runtimes. A clean source checkout is not a claim that all historical model experiments are available or rerun.

## Verified cleanup snapshot

The staged source tree was exported without untracked files, imported in a separate process with `PYTHONPATH` unset, and checked to reject module origins outside that export. The selected regression gate reported:

- **Clean export:** 249 passed, 34 skipped. The skips are 32 explicit original-capture/review integration cases plus two Windows symlink-permission cases.
- **Working copy with retained original capture inputs:** 281 passed, two Windows symlink-permission skips.
- Both formal task release readers passed their fixed index hashes. All nine v13 fixture files remain byte-identical to their retained originals.
- Every one of the 127 archived roots was rehashed successfully after cleanup.

These are maintenance and reproducibility checks. No model, Qdrant service, remote experiment or new scientific result was started during cleanup.
