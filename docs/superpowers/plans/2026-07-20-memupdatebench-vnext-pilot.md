# MemUpdateBench vNext Pilot Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a validated 1,440-task synthetic vNext Pilot spanning Families A-D, with leakage-safe splits, deterministic gold replay, required built-in baselines, capability-aware scoring, incremental resumable runs, manifests, summaries, and auditable cases.

**Architecture:** Consume the Phase 0 `mub.vnext` contracts as the only task/runtime/score/manifest schema. Generate semantic cores first, render three surface variants per core, assign whole semantic-core groups to splits, validate before evaluation, and run every method through the adapter/runtime path. Store one immutable JSONL record per task, derive summaries only from `ScoreRecord`, and publish release artifacts atomically after automated and human-audit gates pass.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, PyYAML, JSONL/JSON, existing `MemoryStore` and constrained-slot parser through adapters, deterministic random generators, existing EM/F1 utilities where definitions match.

---

## 0. Prerequisites, scope, and release gate

### Prerequisite

Every acceptance item in `docs/superpowers/plans/2026-07-20-memupdatebench-vnext-phase0-contract-legacy-bridge.md` must pass. Pilot code must not introduce alternate task, runtime, score, capability, or manifest dictionaries.

### Required Pilot scope

- Exactly 1,440 tasks: 360 each for Families A-D.
- Exactly 120 semantic cores per family and three deterministic surface variants per core.
- Split by semantic-core group before stratification:
  - train: 84 cores/family × 3 variants = 1,008 tasks;
  - dev: 12 cores/family × 3 variants = 144 tasks;
  - test: 24 cores/family × 3 variants = 288 tasks.
- Families:
  - A: repeated same-slot update;
  - B: interleaved multi-slot update;
  - C: entity and attribute grounding;
  - D: NOOP and write discipline.
- Required methods:
  - deterministic reference oracle;
  - raw append;
  - heuristic CRUD with verified embedding backend;
  - deterministic exact-object CRUD.
- Required corrupted controls:
  - always ADD;
  - always NOOP;
  - stale-value copier;
  - wrong-entity writer;
  - wrong-attribute writer;
  - invalid action formatter;
  - current-not-retrieved;
  - gold-retrieved wrong-answer.
- Selected mechanism slice only, not a full Cartesian experiment:
  - chronological/no-label;
  - reverse/no-label;
  - reverse/latest-outdated label;
  - stale counts 1 and 16 on matched Family A semantic cores.

### Explicitly deferred

- Families E-H as benchmark releases;
- Core 10K-20K and Full 50K+ scaling;
- public leaderboard or hidden test server;
- mandatory external-system adapter;
- live API calls or multi-answer-model claims;
- SFT, distillation, RLVR, and online LLM-judge reward;
- public/live case-viewer UI;
- single composite score as the primary scientific result.

### Release gate

The Pilot is not published until:

1. all 1,440 tasks validate structurally and by gold replay;
2. all cross-split semantic-core and group overlaps are zero;
3. generation and split manifests reproduce byte-identically from the fixed config and revision;
4. the reference oracle is perfect on every applicable metric;
5. Families C/D pass family-specific grounding and non-mutation checks;
6. every corrupted control activates its intended flags;
7. every required method emits complete/failed/not-supported rows without silent loss;
8. summaries use only `ScoreRecord` and exclude unsupported/runtime-failed values according to the registry;
9. the deterministic 96-case human-audit sample is reviewed with no unresolved blocking issue;
10. representative aggregate trends are trace-explainable at action, state, store, retrieval, and answer layers.

## 1. File structure map

### Configuration and generation

- Create `configs/vnext/pilot.yaml`: fixed Pilot counts, seed, profile grid, split ratios, mechanism cells, and output layout.
- Create `mub/vnext/generation/__init__.py`: public generation entry points.
- Create `mub/vnext/generation/config.py`: strict YAML configuration models.
- Create `mub/vnext/generation/identity.py`: stable task/core/action/event IDs.
- Create `mub/vnext/generation/catalogs.py`: deterministic entities, attributes, values, aliases, and surface templates.
- Create `mub/vnext/generation/core.py`: semantic-core and generation-context records.
- Create `mub/vnext/generation/render.py`: surface rendering without semantic mutation.
- Create `mub/vnext/generation/family_a.py`: repeated same-slot compiler.
- Create `mub/vnext/generation/family_b.py`: interleaved multi-slot compiler.
- Create `mub/vnext/generation/family_c.py`: grounding compiler.
- Create `mub/vnext/generation/family_d.py`: NOOP/write-discipline compiler.
- Create `mub/vnext/generation/splits.py`: group-first stratified split assignment.
- Create `mub/vnext/generation/build.py`: deterministic build, validation, and atomic release staging.

### Pilot validation and audit

- Create `mub/vnext/validation/pilot.py`: family-specific semantic checks.
- Create `mub/vnext/audit/__init__.py`: audit exports.
- Create `mub/vnext/audit/sample.py`: deterministic audit-sample selection.
- Create `mub/vnext/audit/cases.py`: canonical case export.

### Built-in adapters and runtime

- Create `mub/vnext/adapters/__init__.py`: adapter registry.
- Create `mub/vnext/adapters/reference.py`: gold reference oracle.
- Create `mub/vnext/adapters/raw_append.py`: append-only wrapper.
- Create `mub/vnext/adapters/exact_crud.py`: deterministic exact-object CRUD wrapper.
- Create `mub/vnext/adapters/heuristic_crud.py`: verified semantic heuristic wrapper.
- Create `mub/vnext/adapters/corrupted.py`: scorer sanity controls.
- Create `mub/vnext/adapters/retrieval.py`: normal and latest-per-object retrieval policies.
- Create `mub/vnext/runtime/__init__.py`: runtime exports.
- Create `mub/vnext/runtime/engine.py`: one-task execution.
- Create `mub/vnext/runtime/resume.py`: cache identity and resume decisions.
- Create `mub/vnext/runtime/run.py`: incremental run orchestration and manifests.

### Scoring, aggregation, and mechanism matrix

- Create `mub/vnext/scoring/pilot.py`: concrete Pilot metric computations.
- Create `mub/vnext/scoring/aggregate.py`: registry-driven aggregation from scores only.
- Create `mub/vnext/mechanisms/__init__.py`: mechanism exports.
- Create `mub/vnext/mechanisms/context.py`: order/label context rendering.
- Create `mub/vnext/mechanisms/matrix.py`: selected paired condition manifest.

### CLIs, generated artifacts, and tests

- Create `scripts/vnext_generate_pilot.py`: build CLI.
- Create `scripts/vnext_validate_pilot.py`: release validation CLI.
- Create `scripts/vnext_run_pilot.py`: adapter/runtime CLI.
- Create `scripts/vnext_score_pilot.py`: score JSONL CLI.
- Create `scripts/vnext_summarize_pilot.py`: summary/case CLI.
- Create `scripts/vnext_build_mechanism_slice.py`: selected condition builder.
- Create `data/vnext/pilot/README.md`: generated-artifact and redistribution policy.
- Create `results/vnext/pilot/README.md`: run-artifact layout and claim boundary.
- Create focused `tests/vnext/test_generation_*.py`, adapter, runtime, scoring, audit, and CLI tests below.
- Modify `scripts/smoke_test.py`: add one small deterministic Pilot smoke after focused tests pass.
- Modify `README.md`: document vNext Pilot entry points without replacing historical commands.
- Modify `WORKFLOW.md`: append only actual implementation commands, counts, hashes, validations, and observed baseline results.

Existing legacy generator/evaluator/store/manager files remain unchanged; adapters may import their stable public behavior.

## 2. Fixed Pilot configuration

Create `configs/vnext/pilot.yaml` with the following normative content:

```yaml
schema_version: "1.0.0"
profile_version: "1.0.0"
release_id: "vnext-pilot-2026-07"
seed: 20260720
surface_variants_per_core: 3
cores_per_family: 120
splits:
  train: 0.70
  dev: 0.10
  test: 0.20
families:
  repeated_same_slot_update:
    enabled: true
    update_depths: [1, 4, 16]
    difficulties: [easy, medium, hard]
    same_name_distractors: {easy: 0, medium: 2, hard: 4}
    same_entity_other_attribute: {easy: 0, medium: 1, hard: 2}
    noop_near_miss: {easy: 0, medium: 2, hard: 4}
  interleaved_multi_slot_update:
    enabled: true
    update_depths: [1, 4, 16]
    difficulties: [easy, medium, hard]
    active_object_counts: {easy: 2, medium: 4, hard: 8}
    interleaving_patterns: [round_robin, burst, adversarial_adjacent]
    cross_slot_distractor_density: {easy: 0.0, medium: 0.25, hard: 0.50}
  entity_attribute_grounding:
    enabled: true
    difficulties: [easy, medium, hard]
    entity_conditions: [distinct, same_name, alias, namespace_collision]
    attribute_conditions: [exact, paraphrase, near_name]
  noop_write_discipline:
    enabled: true
    difficulties: [easy, medium, hard]
    noop_densities: [0.25, 0.50, 0.75]
    trap_types: [semantic_near_miss, duplicate_current, other_entity_correction, other_attribute_correction]
mechanism_slice:
  stale_counts: [1, 16]
  conditions:
    - {context_order: chronological, context_annotation: none}
    - {context_order: reverse_chronological, context_annotation: none}
    - {context_order: reverse_chronological, context_annotation: latest_outdated_label}
output:
  staging_dir: "data/vnext/.pilot-staging"
  release_dir: "data/vnext/pilot"
```

The builder must verify the config computes exactly 480 semantic cores and 1,440 tasks. Family-specific small-cell deviations caused by 120-core allocation must be recorded in the task manifest; they must not be hidden by dropping tasks.

## 3. Task breakdown

### Task 1: Add strict Pilot configuration loading

**Files:**
- Create: `configs/vnext/pilot.yaml`
- Create: `mub/vnext/generation/__init__.py`
- Create: `mub/vnext/generation/config.py`
- Test: `tests/vnext/test_generation_config.py`

- [ ] **Step 1: Write failing config tests**

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from mub.vnext.generation.config import PilotConfig, load_pilot_config


def test_fixed_config_computes_release_size() -> None:
    config = load_pilot_config(Path("configs/vnext/pilot.yaml"))
    assert config.total_semantic_cores == 480
    assert config.total_tasks == 1440
    assert config.expected_split_tasks == {"train": 1008, "dev": 144, "test": 288}


def test_unknown_family_key_is_rejected() -> None:
    payload = load_pilot_config(Path("configs/vnext/pilot.yaml")).model_dump(mode="json")
    payload["families"]["repeated_same_slot_update"]["unreviewed_axis"] = 7
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(payload)
```

- [ ] **Step 2: Run and verify imports fail**

```bash
python -m pytest tests/vnext/test_generation_config.py -v
```

Expected: missing generation config module.

- [ ] **Step 3: Implement strict Pydantic config models**

Define one model per family configuration, `MechanismCondition`, `MechanismSliceConfig`, `OutputConfig`, and `PilotConfig`. All inherit `ContractModel`; all numeric values have positive/range constraints; family keys are a fixed typed model rather than an open dictionary.

Computed properties:

```python
@property
def total_semantic_cores(self) -> int:
    return self.cores_per_family * 4

@property
def total_tasks(self) -> int:
    return self.total_semantic_cores * self.surface_variants_per_core

@property
def expected_split_tasks(self) -> dict[str, int]:
    per_family = {"train": 84, "dev": 12, "test": 24}
    return {name: core_count * 4 * self.surface_variants_per_core for name, core_count in per_family.items()}
```

`load_pilot_config` uses `yaml.safe_load` and `PilotConfig.model_validate`.

- [ ] **Step 4: Run config tests**

```bash
python -m pytest tests/vnext/test_generation_config.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add configs/vnext/pilot.yaml mub/vnext/generation tests/vnext/test_generation_config.py
git commit -m "feat: define fixed vnext pilot configuration"
```

Commit only when execution-time permission is active; otherwise record the scoped diff.

### Task 2: Add deterministic identities, catalogs, semantic cores, and rendering

**Files:**
- Create: `mub/vnext/generation/identity.py`
- Create: `mub/vnext/generation/catalogs.py`
- Create: `mub/vnext/generation/core.py`
- Create: `mub/vnext/generation/render.py`
- Test: `tests/vnext/test_generation_common.py`

- [ ] **Step 1: Write failing determinism and semantic-invariance tests**

Assert the same `GenerationContext` and semantic core produce identical bytes, surface variants have distinct linked surface IDs but one semantic-core ID, and changing relation wording, speaker labels, and linked IDs preserves replay state/history/answers plus normalized gold semantics.

```python
from pydantic import RootModel

from mub.vnext.contracts.common import MemoryObjectKey
from mub.vnext.contracts.enums import Difficulty, EventRole, Operation, Split, TaskFamily
from mub.vnext.generation.core import CoreEvent, SemanticCore
from mub.vnext.generation.identity import core_id, trajectory_id
from mub.vnext.generation.render import render_core
from mub.vnext.io.canonical import canonical_json_bytes, semantic_task_hash


class _GoldProjection(RootModel[object]):
    pass


def _normalized_gold_bytes(task: object) -> bytes:
    payload = task.gold.model_dump(mode="json")
    event_indices = {event.event_id: index for index, event in enumerate(task.events)}
    action_indices = {
        action_identifier: index
        for index, action_identifier in enumerate(task.gold.action_sequence)
    }
    query_indices = {query.query_id: index for index, query in enumerate(task.queries)}

    for action in payload["actions"]:
        action["action_id"] = f"action[{action_indices[action['action_id']]}]"
        action["event_id"] = f"event[{event_indices[action['event_id']]}]"
    payload["action_sequence"] = [
        f"action[{action_indices[action_identifier]}]"
        for action_identifier in payload["action_sequence"]
    ]
    payload["gold_source_event_ids"] = [
        f"event[{event_indices[event_identifier]}]"
        for event_identifier in payload["gold_source_event_ids"]
    ]
    payload["gold_answers"] = {
        f"query[{query_indices[query_identifier]}]": answer
        for query_identifier, answer in payload["gold_answers"].items()
    }
    payload["acceptable_answers"] = {
        f"query[{query_indices[query_identifier]}]": answers
        for query_identifier, answers in payload["acceptable_answers"].items()
    }
    return canonical_json_bytes(_GoldProjection(root=payload))


def make_test_core() -> SemanticCore:
    key = MemoryObjectKey(namespace="default", entity="friend:alex", attribute="location")
    semantic_core_id = core_id(TaskFamily.REPEATED_SAME_SLOT.value, {"fixture": "common"})
    return SemanticCore(
        core_id=semantic_core_id,
        task_family=TaskFamily.REPEATED_SAME_SLOT,
        difficulty=Difficulty.EASY,
        core_index=0,
        trajectory_id=trajectory_id(semantic_core_id, 0),
        events=[
            CoreEvent(
                operation=Operation.ADD,
                object_keys=[key],
                value="Dalian",
                role=EventRole.STALE_SAME_SLOT,
            ),
            CoreEvent(
                operation=Operation.UPDATE,
                object_keys=[key],
                value="Qingdao",
                role=EventRole.LATEST_GOLD,
            ),
        ],
        query_targets=[key],
        expected_answer="Qingdao",
        profile={"update_depth": 1},
        stratification={"update_depth": 1, "difficulty": "easy"},
    )


def test_surface_variants_share_semantics() -> None:
    core = make_test_core()
    tasks = [render_core(core, split=Split.TEST, surface_variant=index) for index in range(3)]
    assert len({task.task_id for task in tasks}) == 3
    assert len({task.metadata.split_key.semantic_core_id for task in tasks}) == 1
    assert len({_normalized_gold_bytes(task) for task in tasks}) == 1
    assert len({semantic_task_hash(task) for task in tasks}) == 1
```

Here `_normalized_gold_bytes` replaces linked event/action/query IDs and answer-map keys with their event/action/query sequence indices before canonical comparison. Raw `task.gold` bytes may differ only through those required linked IDs.

- [ ] **Step 2: Implement stable identity helpers**

`stable_id(prefix, payload)` hashes canonical JSON and returns `f"{prefix}_{digest[:16]}"`. Separate helpers create core, task, event, action, query, source, trajectory, and paraphrase-group IDs. No identity includes absolute file paths, process IDs, current timestamps, or dictionary insertion order.

- [ ] **Step 3: Add finite reviewed catalogs**

`catalogs.py` contains fixed tuples for namespaces, relation-qualified entities, same-name entities, aliases, attributes, values, and three reviewed surface-template sets. A helper rejects reuse of the final value as a conflicting stale value; duplicate-current events are generated explicitly by Family D instead.

- [ ] **Step 4: Implement semantic-core and rendering records**

Use:

```python
class CoreEvent(ContractModel):
    operation: Operation
    object_keys: list[MemoryObjectKey]
    value: JsonValue | None
    role: EventRole
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SemanticCore(ContractModel):
    core_id: str
    task_family: TaskFamily
    difficulty: Difficulty
    core_index: int
    trajectory_id: str
    events: list[CoreEvent]
    query_targets: list[MemoryObjectKey]
    expected_answer: JsonValue | None
    profile: dict[str, JsonValue]
    stratification: dict[str, str | int | float | bool]
```

`render_core(core, *, split: Split, surface_variant: int, context: GenerationContext) -> MemUpdateTask` converts core events into natural-language `MemoryEvent`s and ordered `GoldAction`s, attaches generator provenance from the required caller-supplied context, and validates replay before returning a task. The context is built from the fixed Pilot config plus the explicit code revision, generator name, and compiler version; release, schema, and profile versions remain recorded as artifact provenance but excluded from semantic task hashes. Surface variants may change wording, speaker labels, and deterministic linked task/source/event/action/query IDs. Event/action and query/gold references remain exact; semantic equivalence is established by identical semantic task hashes, replay state/history/answers, and the linked-ID-normalized gold projection above.

- [ ] **Step 5: Run common generation tests**

```bash
python -m pytest tests/vnext/test_generation_common.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Record an isolated checkpoint**

```bash
git add mub/vnext/generation tests/vnext/test_generation_common.py
git commit -m "feat: add deterministic pilot generation core"
```

### Task 3: Implement Family A repeated same-slot generation

**Files:**
- Create: `mub/vnext/generation/family_a.py`
- Test: `tests/vnext/test_generation_family_a.py`

- [ ] **Step 1: Write failing Family A invariant tests**

For update depths 1, 4, and 16 across all difficulties, assert:

- one exact target object key;
- one initial ADD followed by exactly `update_depth` UPDATE actions;
- final answer equals the latest target value;
- stale conflicting values never equal the final value;
- same-name and other-attribute distractors use different object keys;
- near-miss events compile to NOOP;
- `stale_same_slot` and `duplicate_current` roles remain distinct;
- replay is perfect.

- [ ] **Step 2: Implement the Family A core generator**

Expose:

```python
def generate_family_a_cores(config: PilotConfig) -> list[SemanticCore]:
```

Generate exactly 120 cores. Deterministically cycle the 3 update depths and 3 difficulties, then fill family axes using seeded shuffled product order. `update_depth` is the number of target-slot UPDATE actions after one initial ADD, so a depth-16 task has 17 target versions and 16 conflicting stale versions before the final state is selected. It is never the total event count. Store `num_events`, `num_target_updates`, stale count, distractor counts, and NOOP count as separate resolved-profile fields.

- [ ] **Step 3: Run Family A tests**

```bash
python -m pytest tests/vnext/test_generation_family_a.py -v
```

Expected: all tests pass and exactly 120 cores are produced.

- [ ] **Step 4: Record an isolated checkpoint**

```bash
git add mub/vnext/generation/family_a.py tests/vnext/test_generation_family_a.py
git commit -m "feat: generate repeated same-slot pilot tasks"
```

### Task 4: Implement Family B interleaved multi-slot generation

**Files:**
- Create: `mub/vnext/generation/family_b.py`
- Test: `tests/vnext/test_generation_family_b.py`

- [ ] **Step 1: Write failing interleaving and preservation tests**

For every pattern, assert target actions preserve their order, active non-target objects retain their own latest values, object keys never collapse across slots, and an update to one slot does not mutate another. Verify `round_robin`, `burst`, and `adversarial_adjacent` produce measurably different event-index patterns while sharing valid replay semantics.

- [ ] **Step 2: Implement the Family B core generator**

Expose:

```python
def generate_family_b_cores(config: PilotConfig) -> list[SemanticCore]:
```

Generate exactly 120 cores, balancing update depth, difficulty, active-object count, interleaving pattern, and cross-slot distractor density with deterministic small-cell deviations recorded in `stratification`. The target trajectory must be identical before interleaving; only event order and non-target trajectories vary.

- [ ] **Step 3: Run Family B tests**

```bash
python -m pytest tests/vnext/test_generation_family_b.py -v
```

Expected: all tests pass and final replay preserves every active object.

- [ ] **Step 4: Record an isolated checkpoint**

```bash
git add mub/vnext/generation/family_b.py tests/vnext/test_generation_family_b.py
git commit -m "feat: generate interleaved multi-slot pilot tasks"
```

### Task 5: Implement Family C entity and attribute grounding

**Files:**
- Create: `mub/vnext/generation/family_c.py`
- Test: `tests/vnext/test_generation_family_c.py`

- [ ] **Step 1: Write failing grounding-condition tests**

Cover the 4×3 entity/attribute condition grid. Assert same-name entities differ by relation-qualified entity identity, aliases carry an explicit alias map, namespace collisions differ by namespace, attribute paraphrases resolve to one canonical attribute, near-name attributes remain distinct, and ambiguous cases have an explicit abstention gold rather than a guessed value.

- [ ] **Step 2: Implement the Family C core generator**

Expose:

```python
def generate_family_c_cores(config: PilotConfig) -> list[SemanticCore]:
```

Generate exactly 120 cores: ten per `(entity_condition, attribute_condition)` cell. Difficulty is derived from reviewed combinations but remains stored explicitly. Every alias/namespace decision appears in `resolved_profile`; no evaluator reconstructs it from surface text.

- [ ] **Step 3: Run Family C tests**

```bash
python -m pytest tests/vnext/test_generation_family_c.py -v
```

Expected: all tests pass and every grid cell has ten semantic cores.

- [ ] **Step 4: Record an isolated checkpoint**

```bash
git add mub/vnext/generation/family_c.py tests/vnext/test_generation_family_c.py
git commit -m "feat: generate grounding pilot tasks"
```

### Task 6: Implement Family D NOOP and write-discipline generation

**Files:**
- Create: `mub/vnext/generation/family_d.py`
- Test: `tests/vnext/test_generation_family_d.py`

- [ ] **Step 1: Write failing write-discipline tests**

Cover every `(noop_density, trap_type)` cell. Assert NOOP replay never mutates state, duplicate-current is distinguishable from stale conflict, corrections about other entities/attributes do not write the target slot, and mutation count equals the number of non-NOOP gold actions.

- [ ] **Step 2: Implement the Family D core generator**

Expose:

```python
def generate_family_d_cores(config: PilotConfig) -> list[SemanticCore]:
```

Generate exactly 120 cores: ten per 3×4 grid cell. Store total event count, true write count, NOOP count, duplicate-current count, and trap type separately. Do not use `num_updates` as a canonical name.

- [ ] **Step 3: Run Family D tests**

```bash
python -m pytest tests/vnext/test_generation_family_d.py -v
```

Expected: all tests pass and NOOP-only perturbations preserve final state.

- [ ] **Step 4: Record an isolated checkpoint**

```bash
git add mub/vnext/generation/family_d.py tests/vnext/test_generation_family_d.py
git commit -m "feat: generate noop discipline pilot tasks"
```

### Task 7: Build leakage-safe grouped splits and atomic Pilot artifacts

**Files:**
- Create: `mub/vnext/generation/splits.py`
- Create: `mub/vnext/generation/build.py`
- Create: `scripts/vnext_generate_pilot.py`
- Create: `data/vnext/pilot/README.md`
- Test: `tests/vnext/test_generation_splits.py`
- Test: `tests/vnext/test_generation_build.py`

- [ ] **Step 1: Write failing split tests**

Assert all three variants of a core share one split; per-family core counts are exactly 84/12/24; task counts are exactly 1,008/144/288; semantic-core, trajectory, paraphrase, source-group, and exact-task hashes do not overlap across splits; and identical seed/config produces identical assignments.

- [ ] **Step 2: Implement group-first deterministic assignment**

`assign_splits(cores, seed)` groups by semantic-core ID, sorts a stable seeded hash of `(family, difficulty, family-specific strata, core_id)`, and allocates 84/12/24 cores per family while minimizing family-specific stratum deviation. It emits a `split_balance` table containing expected, observed, and deviation counts for every stratum.

- [ ] **Step 3: Write failing build/atomicity tests**

In a temporary directory, assert a successful build emits tasks, task manifest, validation report, split-balance report, generation config snapshot, and hashes. Inject one invalid task and assert the final release directory is absent while the staging failure report remains.

- [ ] **Step 4: Implement the deterministic builder**

`build_pilot(config, output_root)`:

1. snapshots the resolved config and hash;
2. creates 480 semantic cores;
3. assigns core groups to splits;
4. renders three variants/core;
5. validates tasks and replay;
6. writes canonical `tasks.jsonl` ordered by `(split, family, core_id, variant)`;
7. writes manifests/reports to a staging directory;
8. re-reads and validates every emitted artifact;
9. atomically promotes staging to the release directory only when valid.

- [ ] **Step 5: Implement the generation CLI**

`scripts/vnext_generate_pilot.py` accepts `--config`, `--output-dir`, and `--overwrite`. Default output is never silently overwritten. It prints machine-readable counts and artifact hashes.

- [ ] **Step 6: Run generation tests**

```bash
python -m pytest tests/vnext/test_generation_splits.py tests/vnext/test_generation_build.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Record an isolated checkpoint**

```bash
git add mub/vnext/generation scripts/vnext_generate_pilot.py data/vnext/pilot/README.md tests/vnext/test_generation_splits.py tests/vnext/test_generation_build.py
git commit -m "feat: build leakage-safe vnext pilot release"
```

### Task 8: Add Pilot semantic validation and human-audit artifacts

**Files:**
- Create: `mub/vnext/validation/pilot.py`
- Create: `mub/vnext/audit/__init__.py`
- Create: `mub/vnext/audit/sample.py`
- Create: `scripts/vnext_validate_pilot.py`
- Test: `tests/vnext/test_pilot_validation.py`
- Test: `tests/vnext/test_audit_sample.py`

- [ ] **Step 1: Write failing family-semantic tests**

Inject one error at a time: stale value equal to final value but labeled conflicting stale; distractor independently establishing current gold; Family B non-target corruption; Family C alias without alias map; Family D NOOP mutation; query with multiple valid current answers. Assert distinct issue codes.

- [ ] **Step 2: Implement family-specific validation**

`validate_pilot_task(task)` dispatches by `TaskFamily`, combines structural and replay reports, and adds family-specific issues. `validate_pilot_release(tasks, manifest)` verifies counts, strata, hashes, split isolation, unique-answer rules, and manifest consistency.

- [ ] **Step 3: Write failing audit-selection tests**

Assert exactly 96 tasks are selected deterministically: 24/family, covering each difficulty, split, and every declared family-specific condition where mathematically possible. Selection must use task metadata, not filenames.

- [ ] **Step 4: Implement audit sample and decision contracts**

Define:

```python
class AuditSelection(ContractModel):
    audit_id: str
    task_id: str
    family: TaskFamily
    difficulty: Difficulty
    split: Split
    covered_conditions: list[str]
    selection_reason: str


class AuditDecision(ContractModel):
    audit_id: str
    reviewer: str
    verdict: Literal["pass", "block", "needs_revision"]
    answer_unique: bool
    actions_correct: bool
    roles_correct: bool
    surface_natural: bool
    notes: str
```

The release gate requires one terminal decision per selected audit ID and zero `block`/`needs_revision` decisions. Automation creates selections and a blank decision template but never fabricates reviewer decisions.

- [ ] **Step 5: Implement validation CLI**

`scripts/vnext_validate_pilot.py` accepts tasks, manifest, optional audit decisions, and output report paths. Without completed human decisions it may report automated validation success but must return a non-release-ready status.

- [ ] **Step 6: Run validation/audit tests**

```bash
python -m pytest tests/vnext/test_pilot_validation.py tests/vnext/test_audit_sample.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Record an isolated checkpoint**

```bash
git add mub/vnext/validation/pilot.py mub/vnext/audit scripts/vnext_validate_pilot.py tests/vnext/test_pilot_validation.py tests/vnext/test_audit_sample.py
git commit -m "feat: validate and audit vnext pilot tasks"
```

### Task 9: Implement required built-in adapters

**Files:**
- Create: `mub/vnext/adapters/__init__.py`
- Create: `mub/vnext/adapters/reference.py`
- Create: `mub/vnext/adapters/raw_append.py`
- Create: `mub/vnext/adapters/exact_crud.py`
- Create: `mub/vnext/adapters/heuristic_crud.py`
- Create: `mub/vnext/adapters/retrieval.py`
- Test: `tests/vnext/test_builtin_adapters.py`
- Test: `tests/vnext/test_retrieval_policies.py`

- [ ] **Step 1: Write failing adapter conformance tests**

For each adapter, assert isolated reset, event ingestion, exported capability bitset, normalized entries, source-event linkage, and deterministic close behavior. The reference oracle must be perfect; raw append must retain stale same-slot entries; exact CRUD must keep one current entry/object; heuristic CRUD must refuse to run when its embedding backend cannot be verified.

- [ ] **Step 2: Implement the reference adapter**

`ReferenceAdapter` consumes the task only through a constructor-scoped immutable gold reference, applies gold actions in event order, exports exact state/history/action trace, retrieves the current target, and returns the expected answer. Adapter info labels it `oracle_smoke_only`; summaries must not place it in model leaderboards.

- [ ] **Step 3: Implement raw append and exact CRUD wrappers**

`RawAppendAdapter` writes one entry for every non-NOOP parsed event and never rewrites prior entries. `ExactCrudAdapter` uses the existing constrained-slot parser on event text and updates only the exact object key. Both translate `MemoryStore` entries into canonical `MemoryEntryRecord`; neither exposes native `MemoryEntry` as an artifact contract.

- [ ] **Step 4: Implement verified heuristic CRUD**

`HeuristicCrudAdapter` wraps current semantic similarity behavior but performs an explicit startup capability check. It records encoder model/revision/backend in `AdapterInfo`; missing encoder support returns typed `not_supported` before task execution. It must never silently accept all-zero fallback embeddings as a valid heuristic run.

- [ ] **Step 5: Implement retrieval policies**

Provide `normal_topk` and `latest_per_object`. The latter scans exported entries, groups by exact object key, selects latest by canonical order metadata, then ranks selected entries. Its adapter/run metadata must include:

```text
retrieval_rewrite=true
not_original_topk_filter=true
full_store_scan=true
```

- [ ] **Step 6: Run adapter/retrieval tests**

```bash
python -m pytest tests/vnext/test_builtin_adapters.py tests/vnext/test_retrieval_policies.py -v
```

Expected: all tests pass; heuristic test uses a fake verified encoder and a separate unavailable-backend case, not a network download.

- [ ] **Step 7: Record an isolated checkpoint**

```bash
git add mub/vnext/adapters tests/vnext/test_builtin_adapters.py tests/vnext/test_retrieval_policies.py
git commit -m "feat: add vnext pilot baseline adapters"
```

### Task 10: Add corrupted scorer controls

**Files:**
- Create: `mub/vnext/adapters/corrupted.py`
- Test: `tests/vnext/test_corrupted_controls.py`

- [ ] **Step 1: Write one expected-failure test per control**

Use fixed tasks to assert:

| Control | Required activated signal |
|---|---|
| always ADD | false write and stale/duplicate burden |
| always NOOP | missed write and final-state error |
| stale-value copier | stale value copied |
| wrong-entity writer | wrong entity grounding |
| wrong-attribute writer | wrong attribute grounding |
| invalid formatter | protocol/action parse failure |
| current-not-retrieved | current not retrieved |
| gold-retrieved wrong-answer | wrong answer with gold retrieved |

Also assert unrelated high-precedence flags do not appear.

- [ ] **Step 2: Implement deterministic corrupted adapters**

Each adapter changes exactly one layer and inherits all unaffected behavior from a small shared deterministic base. They are registered under `control/*`, marked `smoke_control=true`, and excluded from method leaderboards.

- [ ] **Step 3: Run control tests**

```bash
python -m pytest tests/vnext/test_corrupted_controls.py -v
```

Expected: eight controls activate the expected independent and primary failure labels.

- [ ] **Step 4: Record an isolated checkpoint**

```bash
git add mub/vnext/adapters/corrupted.py tests/vnext/test_corrupted_controls.py
git commit -m "test: add vnext scorer corruption controls"
```

### Task 11: Implement incremental runtime and resume semantics

**Files:**
- Create: `mub/vnext/runtime/__init__.py`
- Create: `mub/vnext/runtime/engine.py`
- Create: `mub/vnext/runtime/resume.py`
- Create: `mub/vnext/runtime/run.py`
- Create: `scripts/vnext_run_pilot.py`
- Test: `tests/vnext/test_runtime_engine.py`
- Test: `tests/vnext/test_runtime_resume.py`

- [ ] **Step 1: Write failing one-task runtime tests**

Assert each task produces one `TaskRunRecord` even on adapter exception, reset failure, unsupported call, or answer failure. Prior action/snapshot/retrieval records remain in a partial row.

- [ ] **Step 2: Implement `execute_task`**

`execute_task(task, adapter, run_config)` resets an isolated namespace derived from run/task IDs, ingests events sequentially, captures normalized actions and optional snapshots, retrieves/answers each query, records timing only when declared, and catches task-level exceptions into typed terminal rows. It closes the adapter in `finally` without replacing the primary error.

- [ ] **Step 3: Write failing resume/cache tests**

Assert matching completed rows skip, matching failed rows retry only with `--retry-failed`, partial/not-supported behavior follows explicit flags, hash mismatch forces execution, duplicate task IDs fail, and a missing expected task ID blocks summary generation.

- [ ] **Step 4: Implement stable run identity and resume decisions**

Run identity hashes:

```text
task manifest hash
adapter info and capability hash
runtime config
retrieval policy
answer mode
prompt/decoding config
schema/compiler/profile versions
```

It does not depend on output directory. `ResumeIndex` reads existing JSONL once, validates unique task IDs and identity hashes, and returns `skip`, `retry`, or `reject` per task.

- [ ] **Step 5: Implement incremental runner and manifest finalization**

Write each row, newline, and flush immediately. Maintain a sidecar progress JSON with expected/completed/failed/partial/not-supported IDs. Finalize `RunManifest` only after counts and file hashes validate. An interrupted run keeps resumable rows but no false completed manifest.

- [ ] **Step 6: Implement the runtime CLI**

`scripts/vnext_run_pilot.py` accepts:

```text
--tasks
--task-manifest
--adapter {reference,raw_add,heuristic_crud,exact_crud,control/*}
--retrieval-policy {normal_topk,latest_per_object}
--answer-mode {slot_direct,slot_prompt,native_answer}
--output-dir
--resume
--retry-failed
```

It makes no API calls. Unsupported adapter/mode combinations produce typed not-supported rows.

- [ ] **Step 7: Run runtime/resume tests**

```bash
python -m pytest tests/vnext/test_runtime_engine.py tests/vnext/test_runtime_resume.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Record an isolated checkpoint**

```bash
git add mub/vnext/runtime scripts/vnext_run_pilot.py tests/vnext/test_runtime_engine.py tests/vnext/test_runtime_resume.py
git commit -m "feat: run vnext pilot incrementally"
```

### Task 12: Implement concrete Pilot scoring and aggregation

**Files:**
- Create: `mub/vnext/scoring/pilot.py`
- Create: `mub/vnext/scoring/aggregate.py`
- Create: `scripts/vnext_score_pilot.py`
- Test: `tests/vnext/test_pilot_scoring.py`
- Test: `tests/vnext/test_score_aggregation.py`

- [ ] **Step 1: Write exact metric tests**

Create small task/run pairs with hand-computed expected values for protocol, action, state, store, retrieval, answer, system, and audit fields. Include conflicting stale versus duplicate-current entries, non-target preservation, expected absence, write amplification, current rank, stale copied, and gold-retrieved wrong-answer.

- [ ] **Step 2: Implement Pilot metric helpers**

Functions operate only on canonical records:

```python
score_actions(task, run) -> ActionScores
score_state(task, run) -> StateScores
score_store(task, run) -> StoreScores
score_retrieval(task, run) -> RetrievalScores
score_answers(task, run) -> AnswerScores
score_system(run, capabilities) -> SystemScores
score_audit(run, capabilities) -> AuditScores
```

Definitions come from the Phase 0 registry. `write_amplification` is actual mutating writes divided by required mutating gold actions; a zero-required-write task reports not-applicable rather than division by zero.

- [ ] **Step 3: Implement score CLI**

`scripts/vnext_score_pilot.py` reads canonical tasks, one run manifest, and task runs; writes one `ScoreRecord` per expected task incrementally; validates task/run IDs; and finalizes score hash in a new run manifest derivation. It never edits the runtime JSONL.

- [ ] **Step 4: Write failing aggregation tests**

Assert unsupported and runtime-failed values are excluded according to registry policy, group counts/denominators are explicit, and aggregation is available by family, difficulty, split, adapter, update depth, active object count, grounding condition, trap type, retrieval policy, and mechanism condition. No grouping uses path names.

- [ ] **Step 5: Implement registry-driven aggregation**

`aggregate_scores(scores, tasks, run_manifest)` returns:

```text
run identity
expected/completed/failed/not-supported counts
metric numerator/denominator/value for every group
failure-flag counts
primary-failure counts
capability coverage
artifact references
```

It accepts only `ScoreRecord`; raw answer text and legacy result JSON are not inputs.

- [ ] **Step 6: Run scoring/aggregation tests**

```bash
python -m pytest tests/vnext/test_pilot_scoring.py tests/vnext/test_score_aggregation.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Record an isolated checkpoint**

```bash
git add mub/vnext/scoring scripts/vnext_score_pilot.py tests/vnext/test_pilot_scoring.py tests/vnext/test_score_aggregation.py
git commit -m "feat: score and aggregate vnext pilot runs"
```

### Task 13: Build the selected mechanism slice

**Files:**
- Create: `mub/vnext/mechanisms/__init__.py`
- Create: `mub/vnext/mechanisms/context.py`
- Create: `mub/vnext/mechanisms/matrix.py`
- Create: `scripts/vnext_build_mechanism_slice.py`
- Test: `tests/vnext/test_mechanism_slice.py`

- [ ] **Step 1: Write paired-condition tests**

For matched Family A cores, assert all conditions preserve the same gold value and multiset of entries; only presentation order/annotation changes. Reverse/no-label must place stale values before the current value under the declared prompt order, while latest/outdated labels correctly mark every version. Stale counts are exactly 1 or 16.

- [ ] **Step 2: Implement context rendering**

`render_context(entries, order, annotation)` supports only the three approved cells. Version labels derive from canonical object key and order metadata, never from answer text. It returns rendered context plus entry IDs/order/labels for audit.

- [ ] **Step 3: Implement condition manifest builder**

`build_mechanism_slice(tasks, config)` selects matched test semantic cores that support stale counts 1 and 16 and emits a manifest containing condition ID, task IDs, semantic-core IDs, n, seed, retrieval composition, answer-model identity, and expected comparison. For the required smoke build, `answer_model` is `deterministic_reference_smoke`, explicitly not a model result.

- [ ] **Step 4: Implement CLI and run tests**

```bash
python -m pytest tests/vnext/test_mechanism_slice.py -v
```

`scripts/vnext_build_mechanism_slice.py` writes context JSONL and a condition manifest. It does not call local or remote language models.

- [ ] **Step 5: Record an isolated checkpoint**

```bash
git add mub/vnext/mechanisms scripts/vnext_build_mechanism_slice.py tests/vnext/test_mechanism_slice.py
git commit -m "feat: build paired pilot mechanism slice"
```

### Task 14: Add canonical summaries and auditable case exports

**Files:**
- Create: `mub/vnext/audit/cases.py`
- Create: `scripts/vnext_summarize_pilot.py`
- Create: `results/vnext/pilot/README.md`
- Test: `tests/vnext/test_case_export.py`
- Test: `tests/vnext/test_summary_cli.py`

- [ ] **Step 1: Write failing case-export tests**

A case must include task metadata, event timeline and roles, gold/predicted actions, snapshots/final state, retrieved context and IDs, answer output, per-layer metrics, support reasons, all failure flags, primary failure, run/task artifact hashes, and source linkage. Missing capabilities remain explicit rather than fabricating empty traces.

- [ ] **Step 2: Implement `export_case`**

The exporter joins canonical task, run, score, task manifest, and run manifest by IDs. It does not recompute metrics. Private/non-redistributable source text is redacted according to `SourceRecord.redistributable`, while hashes and anchors remain.

- [ ] **Step 3: Write and implement summary CLI tests**

In temporary directories, assert the CLI writes `summary.json`, `summary.csv`, `failure_breakdown.json`, `capability_coverage.json`, `cases.jsonl`, and `artifact_index.json`; refuses incomplete expected task sets; and does not accept raw legacy result JSON as score input.

- [ ] **Step 4: Implement `scripts/vnext_summarize_pilot.py`**

Arguments:

```text
--tasks
--task-manifest
--task-runs
--scores
--run-manifest
--output-dir
--case-policy {all,failures,stratified}
```

The default case policy is `stratified`: at least one correct and one failure per available family/difficulty/method cell, bounded by a documented maximum. No model/API execution occurs.

- [ ] **Step 5: Document result semantics**

`results/vnext/pilot/README.md` distinguishes oracle/control smoke from real method results; explains unsupported denominators, latest-per-object retrieval rewrite, trace-level versus answer-level interventions, and required manifest references.

- [ ] **Step 6: Run case/summary tests**

```bash
python -m pytest tests/vnext/test_case_export.py tests/vnext/test_summary_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Record an isolated checkpoint**

```bash
git add mub/vnext/audit/cases.py scripts/vnext_summarize_pilot.py results/vnext/pilot/README.md tests/vnext/test_case_export.py tests/vnext/test_summary_cli.py
git commit -m "feat: export vnext pilot summaries and cases"
```

### Task 15: Run the end-to-end Pilot release gate

**Files:**
- Modify: `scripts/smoke_test.py`
- Modify: `README.md`
- Modify: `WORKFLOW.md`
- Generate after approval: `data/vnext/pilot/tasks.jsonl`
- Generate after approval: `data/vnext/pilot/task_manifest.json`
- Generate after approval: `data/vnext/pilot/validation_report.json`
- Generate after approval: `data/vnext/pilot/split_balance.json`
- Generate after approval: `data/vnext/pilot/audit_sample.jsonl`
- Generate after human review: `data/vnext/pilot/audit_decisions.jsonl`
- Generate after approval: `data/vnext/pilot/mechanism_slice/`

- [ ] **Step 1: Add one no-network Pilot smoke path**

Extend `scripts/smoke_test.py` with a temporary 4-core smoke build, one per family, three variants/core. Run reference, raw append, exact CRUD, and corrupted controls; assert validation, scoring, and summary artifacts are produced. Do not use the fixed release directory, external APIs, model downloads, or real result roots.

- [ ] **Step 2: Run focused and full tests**

```bash
python -m pytest tests/vnext -q
python scripts/smoke_test.py
```

Expected: all vNext tests pass and the historical smoke suite remains green.

- [ ] **Step 3: Run compile checks**

```bash
python -m py_compile mub/vnext/generation/config.py mub/vnext/generation/identity.py mub/vnext/generation/catalogs.py mub/vnext/generation/core.py mub/vnext/generation/render.py mub/vnext/generation/family_a.py mub/vnext/generation/family_b.py mub/vnext/generation/family_c.py mub/vnext/generation/family_d.py mub/vnext/generation/splits.py mub/vnext/generation/build.py mub/vnext/validation/pilot.py mub/vnext/audit/sample.py mub/vnext/audit/cases.py mub/vnext/adapters/reference.py mub/vnext/adapters/raw_append.py mub/vnext/adapters/exact_crud.py mub/vnext/adapters/heuristic_crud.py mub/vnext/adapters/corrupted.py mub/vnext/adapters/retrieval.py mub/vnext/runtime/engine.py mub/vnext/runtime/resume.py mub/vnext/runtime/run.py mub/vnext/scoring/pilot.py mub/vnext/scoring/aggregate.py mub/vnext/mechanisms/context.py mub/vnext/mechanisms/matrix.py scripts/vnext_generate_pilot.py scripts/vnext_validate_pilot.py scripts/vnext_run_pilot.py scripts/vnext_score_pilot.py scripts/vnext_summarize_pilot.py scripts/vnext_build_mechanism_slice.py
```

Expected: command exits 0 with no output.

- [ ] **Step 4: Generate the fixed Pilot twice into temporary roots**

```bash
python scripts/vnext_generate_pilot.py --config configs/vnext/pilot.yaml --output-dir .tmp/vnext-pilot-a
python scripts/vnext_generate_pilot.py --config configs/vnext/pilot.yaml --output-dir .tmp/vnext-pilot-b
```

Expected from each build:

```text
semantic_cores=480
tasks=1440
train=1008
dev=144
test=288
validation_errors=0
cross_split_semantic_core_overlap=0
```

Compare canonical artifact SHA-256 maps. They must match exactly.

- [ ] **Step 5: Run automated release validation**

```bash
python scripts/vnext_validate_pilot.py --tasks .tmp/vnext-pilot-a/tasks.jsonl --manifest .tmp/vnext-pilot-a/task_manifest.json --output .tmp/vnext-pilot-a/release_validation.json
```

Expected: automated gates pass; release readiness remains false until human audit decisions are complete.

- [ ] **Step 6: Complete the 96-case human audit**

Review every `audit_sample.jsonl` row and fill one `AuditDecision` per audit ID. Re-run validation with `--audit-decisions`. Any blocking or needs-revision decision requires generator correction, full regeneration, and re-audit of affected strata. Do not alter task JSON manually.

- [ ] **Step 7: Publish the reviewed Pilot atomically**

After all automated and human gates pass, rerun generation with `--output-dir data/vnext/pilot --overwrite` only after inspecting the target and confirming it contains no contradictory user-created artifacts. Revalidate published hashes against the reviewed temporary build.

- [ ] **Step 8: Run required methods incrementally**

Run reference, raw append, exact CRUD, and heuristic CRUD using unique manifest-backed output directories. If the heuristic backend cannot be verified, record a complete not-supported run rather than substituting zero embeddings.

Representative commands:

```bash
python scripts/vnext_run_pilot.py --tasks data/vnext/pilot/tasks.jsonl --task-manifest data/vnext/pilot/task_manifest.json --adapter reference --retrieval-policy normal_topk --answer-mode slot_direct --output-dir results/vnext/pilot/reference
python scripts/vnext_run_pilot.py --tasks data/vnext/pilot/tasks.jsonl --task-manifest data/vnext/pilot/task_manifest.json --adapter raw_add --retrieval-policy normal_topk --answer-mode slot_direct --output-dir results/vnext/pilot/raw_add
python scripts/vnext_run_pilot.py --tasks data/vnext/pilot/tasks.jsonl --task-manifest data/vnext/pilot/task_manifest.json --adapter exact_crud --retrieval-policy normal_topk --answer-mode slot_direct --output-dir results/vnext/pilot/exact_crud
python scripts/vnext_run_pilot.py --tasks data/vnext/pilot/tasks.jsonl --task-manifest data/vnext/pilot/task_manifest.json --adapter heuristic_crud --retrieval-policy normal_topk --answer-mode slot_direct --output-dir results/vnext/pilot/heuristic_crud
```

Do not interpret metrics until every expected task ID has a terminal row.

- [ ] **Step 9: Run corrupted controls and scorer checks**

Run the eight controls on the deterministic smoke slice, score them, and compare activated flags against the expected control matrix. These are scorer sanity checks, not model results.

- [ ] **Step 10: Score and summarize required runs**

For each run, execute `vnext_score_pilot.py` then `vnext_summarize_pilot.py`. Verify summaries read score JSONL only and report explicit numerator/denominator/support counts.

- [ ] **Step 11: Build the selected mechanism slice**

```bash
python scripts/vnext_build_mechanism_slice.py --tasks data/vnext/pilot/tasks.jsonl --config configs/vnext/pilot.yaml --output-dir data/vnext/pilot/mechanism_slice
```

Verify matched semantic-core IDs, stale counts, order/annotation conditions, and deterministic-reference-smoke labeling. Do not describe this build as a new answer-model experiment.

- [ ] **Step 12: Inspect representative traces before reporting trends**

For each required method × family × difficulty, inspect at least one correct and one failing case when available. Confirm aggregate trends can be traced to the intended layer and record unexplained patterns as unresolved rather than assigning a mechanism post hoc.

- [ ] **Step 13: Update `README.md` and `WORKFLOW.md` with actual evidence**

Document exact commands, artifact hashes, task/split counts, test output, audit coverage, capability coverage, run completion counts, real observed metrics, failure cases, and limitations. Do not include projected numbers, unavailable external baselines, or pending answer-model cells.

- [ ] **Step 14: Run final repository checks**

```bash
python -m pytest tests/vnext -q
python scripts/smoke_test.py
git diff --check
git status --short
```

Inspect the scoped diff and confirm no existing legacy `data/`, `results/`, `paper/`, manuscript, API recovery script, or user presentation artifact was overwritten or accidentally staged.

- [ ] **Step 15: Record the Pilot checkpoint**

If execution-time commit permission is active, stage only reviewed Pilot code/config/docs and explicitly approved release artifacts. Keep raw caches, private payloads, API responses, and unrelated dirty files untracked/unstaged.

```bash
git add configs/vnext mub/vnext/generation mub/vnext/validation/pilot.py mub/vnext/audit mub/vnext/adapters mub/vnext/runtime mub/vnext/scoring/pilot.py mub/vnext/scoring/aggregate.py mub/vnext/mechanisms scripts/vnext_generate_pilot.py scripts/vnext_validate_pilot.py scripts/vnext_run_pilot.py scripts/vnext_score_pilot.py scripts/vnext_summarize_pilot.py scripts/vnext_build_mechanism_slice.py tests/vnext data/vnext/pilot/README.md results/vnext/pilot/README.md README.md WORKFLOW.md
git commit -m "feat: deliver validated vnext pilot benchmark"
```

Otherwise leave the complete validated diff uncommitted and report exact artifact/test status.

## 4. Pilot acceptance checklist

### Dataset and split integrity

- [ ] Exactly 480 semantic cores and 1,440 tasks exist.
- [ ] Each family has 120 cores and 360 tasks.
- [ ] Every core has exactly three surface variants with identical gold semantics.
- [ ] Train/dev/test counts are 1,008/144/288.
- [ ] Semantic-core, trajectory, paraphrase, source-group, source-document, version-group, and exact-task-hash cross-split overlaps are zero.
- [ ] Family-specific stratum deviations are machine-readable and not hidden.
- [ ] Two clean builds from the same config/revision have identical hashes.

### Semantic correctness

- [ ] All tasks pass structural, replay, unique-answer, role, distractor, and family-specific validation.
- [ ] Family A separates target update depth, total events, distractors, NOOPs, conflicting stale, and duplicate-current records.
- [ ] Family B preserves every non-target object and makes interleaving explicit.
- [ ] Family C tests entity/attribute grounding without collapsing aliases or namespaces.
- [ ] Family D proves NOOP non-mutation and distinguishes unnecessary/duplicate writes.
- [ ] The 96-case human audit has complete terminal decisions and zero unresolved blockers.

### Runtime, scoring, and auditability

- [ ] Reference, raw append, exact CRUD, and heuristic CRUD have manifest-backed terminal rows for every expected task or explicit not-supported status.
- [ ] Reference oracle is perfect on every applicable metric.
- [ ] Eight corrupted controls activate their intended flags.
- [ ] Unsupported/runtime-failed metrics remain null with reasons and correct denominator policies.
- [ ] Summaries use only canonical score records and never infer run identity from paths.
- [ ] Case exports join task/run/score/manifest provenance and preserve missing-capability visibility.
- [ ] Latest-per-object is labeled as a full-store retrieval rewrite, not a pure original-top-k deletion.
- [ ] Mechanism-slice contexts are paired and deterministic; smoke outputs are not reported as model results.

### Operational and claim safety

- [ ] Incremental JSONL writes survive interruption and resume only under matching hashes.
- [ ] Every release/run/score/summary artifact has a manifest and SHA-256.
- [ ] No API key, transcript-extracted secret, private source payload, or non-redistributable text enters release artifacts.
- [ ] No legacy source/result/paper artifact is overwritten.
- [ ] No P6/P8 compatibility result is described as leakage-free vNext held-out performance.
- [ ] No external adapter, API answer model, SFT, or RLVR claim is implied by the Pilot.
- [ ] Tests, compile checks, deterministic rebuild, smoke suite, and `git diff --check` pass.

## 5. Post-Pilot decision gate

After the Pilot is implemented, validated, audited, and reviewed, create a separate Core design/implementation cycle. Core planning may consider Families E/F, larger hard slices, a verified Level 2 external adapter, repeated answer-model runs, confidence intervals, and richer case tooling. Families G/H, public leaderboard infrastructure, Full-scale data, SFT, and RLVR remain separate later approvals rather than automatic continuation.
