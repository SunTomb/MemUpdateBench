# Core Task 561 Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and validate an exact staged 3,000-core/12,000-task Core v3 candidate twice and create the authenticated 560-task `core-hard-v1` manifest view without immutable publication.

**Architecture:** Extend the current parameterized Core generation kernel rather than copying Pilot or E/F/G micro-pilot code. Full E/F/G schedules feed the existing four-surface renderer and authenticated group-first compiler; a separate staged-candidate layer serializes canonical v3 artifacts, validates them, and derives a manifest-only hard-suite view. Pilot/v2 and all micro-pilot public entry points remain unchanged.

**Tech Stack:** Python 3.10+, Pydantic v2, canonical UTF-8 JSON/JSONL, SHA-256, pytest, Git.

---

## File structure

- Modify `configs/vnext/core.yaml`: declare exact A–G release counts, split quotas, and E/F/G schedule axes.
- Modify `mub/vnext/generation/core_config.py`: strict typed E/F/G schedules and exact total validation.
- Modify `mub/vnext/generation/family_e.py`: parameterized lifecycle-cell generator preserving micro defaults.
- Modify `mub/vnext/generation/family_f.py`: parameterized trajectory generator preserving micro defaults.
- Modify `mub/vnext/generation/family_g.py`: parameterized synthesis schedules and evidence fingerprints preserving micro defaults.
- Modify `mub/vnext/generation/core_build.py`: unified A–G group-first assignment, exact quotas, overlaps, profile marginals, replay, and semantic-equivalence gates.
- Create `mub/vnext/generation/core_artifacts.py`: canonical staged candidate bundle and v3 task manifest construction.
- Create `mub/vnext/generation/core_hard_suite.py`: deterministic test-core selection and authenticated manifest-view model.
- Create `mub/vnext/generation/core_orchestrate.py`: transactional candidate staging without immutable publication.
- Create `mub/vnext/validation/core_release.py`: disk-load authentication and complete candidate/hard-suite validation report.
- Create `scripts/vnext_generate_core.py`: clean-revision-bound candidate generator to arbitrary output directories.
- Create `scripts/vnext_validate_core.py`: standalone candidate validator.
- Modify `mub/vnext/generation/__init__.py`: export new Core APIs without changing existing exports.
- Modify/add focused tests under `tests/vnext/` for schedules, compiler, artifacts, hard suite, CLIs, and compatibility.

### Task 1: Exact E/F/G full schedules

**Files:**
- Modify: `configs/vnext/core.yaml`
- Modify: `mub/vnext/generation/core_config.py`
- Modify: `mub/vnext/generation/family_e.py`
- Modify: `mub/vnext/generation/family_f.py`
- Modify: `mub/vnext/generation/family_g.py`
- Test: `tests/vnext/test_core_generation_config.py`
- Test: `tests/vnext/test_core_generation_family_e.py`
- Test: `tests/vnext/test_core_generation_family_f.py`
- Test: `tests/vnext/test_core_generation_family_g.py`

- [ ] **Step 1: Write failing exact-schedule tests**

Assert `CoreConfig` totals of 3,000 cores and 12,000 tasks; A/B/E quotas 336/48/96; C/D/F quotas 294/42/84; G quotas 210/30/60. Assert E has eight cells × 60, balanced difficulty/deletion position; F has 60 trajectories × seven selectors, each trajectory with at least four versions and one version group; G has hop counts 2/3/4 × 60 and object counts 3/5/8 × 40 with exact answer-type halves.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python -m pytest tests/vnext/test_core_generation_config.py tests/vnext/test_core_generation_family_e.py tests/vnext/test_core_generation_family_f.py tests/vnext/test_core_generation_family_g.py -q
```

Expected: failures because Core config is A–D-only and E/F/G expose only micro schedules.

- [ ] **Step 3: Implement minimal parameterized schedules**

Add typed schedule declarations and optional generator parameters/default profiles. Full Core calls consume YAML counts; `compile_family_{e,f,g}_micro_pilot` continues passing its existing fixed micro profile. Keep strict lifecycle/selector/evidence validators and canonical identity unchanged.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the same command. Expected: all existing micro tests plus new full-schedule tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/vnext/core.yaml mub/vnext/generation/core_config.py mub/vnext/generation/family_e.py mub/vnext/generation/family_f.py mub/vnext/generation/family_g.py tests/vnext/test_core_generation_config.py tests/vnext/test_core_generation_family_e.py tests/vnext/test_core_generation_family_f.py tests/vnext/test_core_generation_family_g.py
git commit -m "feat: add full Core E-G schedules\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 2: Unified authenticated A–G compiler

**Files:**
- Modify: `mub/vnext/generation/core_build.py`
- Modify: `mub/vnext/generation/__init__.py`
- Test: `tests/vnext/test_core_build.py`
- Test: `tests/vnext/test_core_generation_families_ab.py`
- Test: `tests/vnext/test_core_generation_families_cd.py`

- [ ] **Step 1: Write failing bounded and exact compiler tests**

Add a bounded fixture profile that renders few cores quickly while exercising all seven families. Add one schedule-only/full-count test proving exact family/split/profile marginals without repeatedly rendering 12,000 tasks. Add tests that mutate F version grouping, G evidence grouping/fingerprint, normalized source hash, and object type.

- [ ] **Step 2: Run tests and confirm RED**

```bash
python -m pytest tests/vnext/test_core_build.py tests/vnext/test_core_generation_families_ab.py tests/vnext/test_core_generation_families_cd.py -q
```

Expected: A–G fixture and E/F/G grouping assertions fail because `_CORE_FAMILIES` and compiler gates are A–D-only.

- [ ] **Step 3: Generalize compiler minimally**

Route all families through existing generators and `render_core_v3`; assign groups deterministically to exact per-family split quotas; retain four surfaces together. Validate zero cross-split overlap for semantic core, source, trajectory, paraphrase, version group, normalized source hash, and G evidence fingerprint. Run generic task validation, family validators, v3 replay/evidence, semantic equivalence, object-type exclusion, and exact profile marginals.

- [ ] **Step 4: Run focused plus Pilot compatibility tests**

```bash
python -m pytest tests/vnext/test_core_build.py tests/vnext/test_core_generation_families_ab.py tests/vnext/test_core_generation_families_cd.py tests/vnext/test_generation_build.py tests/vnext/test_pilot_release_validation.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/generation/core_build.py mub/vnext/generation/__init__.py tests/vnext/test_core_build.py tests/vnext/test_core_generation_families_ab.py tests/vnext/test_core_generation_families_cd.py
git commit -m "feat: compile authenticated A-G Core snapshots\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 3: Staged canonical candidate artifacts and validator

**Files:**
- Create: `mub/vnext/generation/core_artifacts.py`
- Create: `mub/vnext/generation/core_orchestrate.py`
- Create: `mub/vnext/validation/core_release.py`
- Create: `scripts/vnext_generate_core.py`
- Create: `scripts/vnext_validate_core.py`
- Modify: `mub/vnext/generation/__init__.py`
- Test: `tests/vnext/test_core_artifacts.py`
- Test: `tests/vnext/test_core_release_validation.py`
- Test: `tests/vnext/test_core_cli.py`

- [ ] **Step 1: Write failing bounded artifact/CLI tests**

Require canonical `tasks.jsonl`, `semantic_cores.jsonl`, `generation_config.json`, `split_balance.json`, `task_manifest.json`, and `validation_report.json`; authenticate source config/code revision; reject partial files and hashes; stage into arbitrary roots via temp-directory replacement; never create immutable markers or use `data/vnext/core/v3` implicitly.

- [ ] **Step 2: Run tests and confirm RED**

```bash
python -m pytest tests/vnext/test_core_artifacts.py tests/vnext/test_core_release_validation.py tests/vnext/test_core_cli.py -q
```

Expected: import/entry-point failures.

- [ ] **Step 3: Implement staged bundle, v3 manifest, validator, and CLIs**

Use existing canonical serialization/hash helpers and `TaskManifestV3`. Candidate generation writes to a sibling temporary directory, validates staged bytes and graph consistency, then renames into the requested output directory. Validation reconstructs the authenticated snapshot and proves counts, grouping, overlaps, semantic equivalence, replay, and exact marginals. It emits a canonical report but performs no audit or immutable publication.

- [ ] **Step 4: Run artifact tests and compatibility regressions**

```bash
python -m pytest tests/vnext/test_core_artifacts.py tests/vnext/test_core_release_validation.py tests/vnext/test_core_cli.py tests/vnext/test_generation_artifacts.py tests/vnext/test_generation_publish.py tests/vnext/test_generation_cli.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/generation/core_artifacts.py mub/vnext/generation/core_orchestrate.py mub/vnext/validation/core_release.py scripts/vnext_generate_core.py scripts/vnext_validate_core.py mub/vnext/generation/__init__.py tests/vnext/test_core_artifacts.py tests/vnext/test_core_release_validation.py tests/vnext/test_core_cli.py
git commit -m "feat: stage and validate Core candidates\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 4: Authenticated `core-hard-v1` manifest view

**Files:**
- Create: `mub/vnext/generation/core_hard_suite.py`
- Modify: `mub/vnext/generation/core_artifacts.py`
- Modify: `mub/vnext/validation/core_release.py`
- Test: `tests/vnext/test_core_hard_suite.py`

- [ ] **Step 1: Write failing hard-suite tests**

Require test-only deterministic selection of 20 semantic cores/family, four surfaces/core, 80 tasks/family, 560 total, and A/F/G 240 total. Assert explicit per-family hard-condition coverage, source task-manifest hash binding, selection policy `core-hard-v1`, sorted task IDs, no copied task records, and recomputable suite hash.

- [ ] **Step 2: Run test and confirm RED**

```bash
python -m pytest tests/vnext/test_core_hard_suite.py -q
```

Expected: module/manifest absent.

- [ ] **Step 3: Implement deterministic selection and authentication**

Define a strict manifest model and versioned condition policy. Rank eligible hard test cores by canonical hash within required condition strata, select exactly 20 per family, expand to existing four task IDs, and hash the canonical suite payload excluding only its own hash field. Add bundle serialization and validator authentication.

- [ ] **Step 4: Run hard-suite and candidate tests**

```bash
python -m pytest tests/vnext/test_core_hard_suite.py tests/vnext/test_core_artifacts.py tests/vnext/test_core_release_validation.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/generation/core_hard_suite.py mub/vnext/generation/core_artifacts.py mub/vnext/validation/core_release.py tests/vnext/test_core_hard_suite.py
git commit -m "feat: add authenticated Core hard-suite view\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 5: Full candidates, byte comparison, and final regression gate

**Files:**
- Candidate outputs only under task-local temporary/staging roots outside immutable release paths.
- Modify tests/code only if the full run exposes a defect; use a new failing regression before each fix.

- [ ] **Step 1: Run focused and compatibility suites**

```bash
python -m pytest tests/vnext/test_core_generation_config.py tests/vnext/test_core_generation_families_ab.py tests/vnext/test_core_generation_families_cd.py tests/vnext/test_core_generation_family_e.py tests/vnext/test_core_generation_family_f.py tests/vnext/test_core_generation_family_g.py tests/vnext/test_core_build.py tests/vnext/test_core_artifacts.py tests/vnext/test_core_hard_suite.py tests/vnext/test_core_release_validation.py tests/vnext/test_core_cli.py -q
python -m pytest tests/vnext/test_generation_build.py tests/vnext/test_generation_artifacts.py tests/vnext/test_generation_publish.py tests/vnext/test_generation_orchestrate.py tests/vnext/test_generation_cli.py tests/vnext/test_pilot_release_validation.py tests/vnext/test_pilot_validation_cli.py tests/vnext/test_v3_versioned_schema.py tests/vnext/test_v3_replay_scoring.py -q
```

Expected: all pass.

- [ ] **Step 2: Generate two complete candidates**

```bash
python scripts/vnext_generate_core.py --config configs/vnext/core.yaml --output-dir .task-561-staging/candidate-a
python scripts/vnext_generate_core.py --config configs/vnext/core.yaml --output-dir .task-561-staging/candidate-b
```

Expected: each reports 3,000 cores, 12,000 tasks, split tasks 8,400/1,200/2,400, and hard suite 560.

- [ ] **Step 3: Validate both candidates**

```bash
python scripts/vnext_validate_core.py --release-dir .task-561-staging/candidate-a
python scripts/vnext_validate_core.py --release-dir .task-561-staging/candidate-b
```

Expected: both reports pass counts, grouping, zero overlaps, semantic equivalence, replay, provenance, and hard-suite authentication.

- [ ] **Step 4: Byte-compare required canonical artifacts**

Compare `tasks.jsonl`, `semantic_cores.jsonl`, `generation_config.json`, `split_balance.json`, `task_manifest.json`, `core-hard-v1.json`, and `validation_report.json` byte-for-byte. Expected: no differences and equal SHA-256 for every file.

- [ ] **Step 5: Run syntax/diff sanity checks**

```bash
python -m py_compile mub/vnext/generation/core_config.py mub/vnext/generation/family_e.py mub/vnext/generation/family_f.py mub/vnext/generation/family_g.py mub/vnext/generation/core_build.py mub/vnext/generation/core_artifacts.py mub/vnext/generation/core_hard_suite.py mub/vnext/generation/core_orchestrate.py mub/vnext/validation/core_release.py scripts/vnext_generate_core.py scripts/vnext_validate_core.py
git diff --check
git status --short
```

Expected: compilation and diff checks pass; only intentional code/tests plus task-local ignored/untracked staging roots are present; no immutable Core release, Pilot data, legacy fixtures, or existing release-check directories changed.

- [ ] **Step 6: Commit any final verified fixes**

```bash
git add <intentional-files>
git commit -m "fix: close Core candidate validation gaps\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
```
