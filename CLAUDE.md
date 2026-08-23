# CLAUDE.md

This file gives Claude Code the project context and operating rules needed to continue MemUpdateBench work from a new window.

## Project Overview

MemUpdateBench is the clean project extracted from G-MSRA P6.x. It is no longer a broad RL memory-manager project. The paper direction should now be a controlled diagnostic study of repeated same-slot updates in external memory systems, not a broad general-purpose memory benchmark.

A strict simulated reviewer review in `docs/critical_review.md` changed the project priorities. The current work is useful infrastructure and a plausible workshop-level diagnostic, but it is not yet strong enough for an ACL/EMNLP/NeurIPS/ICLR main-track benchmark paper. Future work should prioritize evidence that addresses external validity, diagnostic depth, related work positioning, and data diversity rather than continuing prose-only packaging.

Core question:

```text
What happens when the same (entity, attribute) memory slot is updated repeatedly?
```

The benchmark evaluates tradeoffs among:

- final-state reliability,
- stale same-slot burden,
- memory compactness,
- answer robustness under realistic slot-conditioned prompting.

The key action/state format remains:

```text
ADD <entity>.<attribute> = <value>
UPDATE <entity>.<attribute> = <value>
NOOP
```

Central invariant for the existing P6/P8 experimental line:

```text
exact legacy slot resolution by (entity, attribute)
```

The vNext canonical identity extends this to the four-part object key defined below.

Most analysis should distinguish:

- stale value retention,
- missed final updates,
- wrong entity grounding,
- wrong attribute parsing,
- answer-layer retrieval/prompt failures.

## Current Thesis and Reviewer-Risk Position

Repeated same-slot update frequency remains the main benchmark axis, but the paper claim must be narrower and sharper than "append more causes stale entries, compact more can lose updates." That framing is too close to a truism unless backed by external systems, method-family sensitivity curves, and deeper failure-mechanism analysis.

The strongest P6.3 result:

- append-only methods can keep final value recoverable under oracle `slot_direct`,
- but they accumulate stale same-slot entries,
- under `slot_prompt`, stale burden causes severe answer collapse,
- learned compact managers reduce stale burden but can miss final updates or remain incompletely compact,
- even perfect clean state does not guarantee prompted answer correctness, which exposes a distinct retrieval/answer-layer failure mode.

At k=16 on P6.3 hard:

| method | slot_direct state_acc | slot_prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| raw_add | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| heuristic_crud | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

P6.5 prompt-robustness diagnostics show that mild prompt variants do not remove the high-k ordering: Constrained CRUD remains around 0.68-0.69 EM at k=16, Raw append remains around 0.09-0.11 EM, and Long25 remains between them. Answer traces show different mechanisms: Raw append often fails because gold values are not retrieved under stale burden; Constrained CRUD still has gold-not-retrieved and gold-retrieved-wrong-answer cases despite zero stale same-slot burden; Long25 mixes state errors with stale/distractor answer-context failures.

Current honest positioning: MemUpdateBench is a promising controlled diagnostic benchmark. The project has now strengthened the mechanism story with P8.x evidence, including latest API answer-model probes, but manuscript work should still avoid broad benchmark overclaiming and should foreground definitions, method meanings, and the order/metadata-sensitive version-arbitration mechanism.

Recent mechanism lock:

- P8.3 shows the claim must be nuanced: stale same-slot conflict is an **order- and metadata-sensitive version arbitration failure**, not a universally strongest distractor. The conflict-type decomposition found `unrelated_distractors` harder than `stale_same_slot` in one surface construction, so do not claim stale same-slot is always the strongest distractor.
- P8.3 synthetic dose-response is the strongest mechanism evidence: reverse/no-label stale=8 EM 0.000 and stale=16 EM 0.031, while reverse+latest/outdated label reaches EM 1.000 at stale=8 and stale=16. Gold is in context; missing current-version signal causes stale copying.
- P8.4 latest API answer-model probes address model-recency concerns. On the clean, format-stable subset (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gemini-3.1-flash-lite-preview`), chronological/no-label stale=16 has EM 1.000, reverse/no-label stale=1/16 has EM 0.000 with stale copied 1.000, and reverse+latest/outdated stale=16 recovers to EM 1.000. Treat other Gemini rows with empty/truncated outputs as API/prompt-format caveats, not central mechanism evidence.

## vNext Status

MemUpdateBench vNext Phase 0 is `FINAL_APPROVED`. It establishes the reusable contract, validation, scoring, provenance, legacy-compatibility, and transactional-publication foundation.

The 1,440-task Families A–D Pilot is `FINAL_APPROVED` as a bounded engineering release, not as a final broad benchmark publication. Task release, built-in runtime, scoring, summaries, corrupted controls, mechanism smoke, trace inspection, the complete vNext test gate, and the 96-case human audit are complete. Human reviewer Ye Shenghao supplied one release-ready decision per regenerated audit ID; the human-rebound task manifest is `b7d7f4169295df5fbcbda0b4be1d2cdc05ad436acb03b6c302137af2a7b59f27`, and the authenticated root index is `d9ef2cebc74a5445863de0ef047c9528cc01eab89354ca93b51917a5f2d0322b`. The immutable task release is under `data/vnext/pilot`; the authenticated cluster result root is `/NAS/yesh/MemUpdateBench/results/vnext/pilot_ca47df7_evidence_bound`, bound to clean runtime revision `ca47df7a6401fabfc25dd4d2151a392439e6c379`. Runtime outputs at `0a7d72d` and `2ab4e93` are invalidated diagnostics and must not be interpreted; `0a7d72d` remains valid task-generation provenance.

The Core **task release** is `FINAL_APPROVED` at generation revision `ba8444bd6db5d4a15eeb0062096d715c77016c86`. The immutable local root is `data/vnext/core/v3`, containing exactly 3,000 semantic cores, 12,000 strict-v3 tasks, the 560-task `core-hard-v1` view, five strict-v3 schemas, and the final 224-task human-audit evidence. The root manifest hash is `f953283a10dd45d3f9d1de066570a9c09b9d132ed458f8dea3c948641b89e99d`, the candidate root digest is `71a6beb3ac8a28dabc753c969e96a47a59f92031d217bebf0fa63d6061012af1`, and the release-ready audit attestation hash is `45461659ab3f65a0a559897e50340a470f27cdecf55b999a1431988567cf00c2`. Initial human review found D/E controlled-surface sentence splices and Family G query capitalization defects; these were fixed in the generator, regenerated, and reviewed again using fresh non-rebound decisions.

The bounded overall Core release is now `FINAL_APPROVED` through Task 14. The authoritative final-review runtime is revision `84beabb62f5cd2cee97b294022db25c8261ab698`, tree SHA-256 `6ab74200f2748a65ee29e8aedf17b4f19dd30e5f1cf856f282764fd3f6bf5133`. The local no-replace final root is `D:/USTC/2026Winter/MemUpdateBench_releases/core_task14_84beabb_v1`, with root-index SHA-256 `2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035` and attestation self-hash `a63008b5b5a60507fcb7c2f99b05373f6f10e7a366383a8d2cde20818208d005`. It binds Core Tasks 9–13 while preserving their evidence classes: Task 9/Pilot engineering evidence, Task 10 Mem0 capability/admission rather than accuracy, Task 11 provenance, Task 12 real prompted-answer evidence, and Task 13 clustered statistics/claims/cases. The remote Task 13 NFS staging path remains evidence only, not a published remote final root. Earlier Task 14 roots are superseded diagnostics.

The separate post-Core model-expansion **Phase 0 metadata release** is complete at clean detached revision `0745fc9dce33a1ace5efdf966d3b1f8b90b9e07b`, tree SHA-256 `916a9cbc1c832270ccc1a9c57b4ac2a5404000da77d03bd35f986cacfc7ec84c`. Its authoritative no-replace root is `D:/USTC/2026Winter/MemUpdateBench_releases/post_core_phase0_0745fc9_clean_v1`, with artifact-index SHA-256 `e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd`. It contains eight explicitly pending model intents, 320 future requested Qwen canary generations, and exactly zero provider calls, model loads, network calls, or executable calls. This is authenticated planning/qualification metadata only—not model capability, accuracy, external-system, prompted-answer, or scientific evidence. The earlier non-strict-clean root without `_clean_` is a superseded diagnostic. Phase 1 has not started.

The follow-up official-document identity preflight is frozen in `configs/vnext/post_core/official_identity_evidence_v1.json` at SHA-256 `9e3780ed3d4303bda7bbd27865df89fcb384041da64af56107c8c5b7abf0a4f0`, bound to the Phase 0 index above. Official pinned repositories now establish Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Meta Muse Glimmer BF16/GGUF revisions `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` / `70bf1b61ac09f91b24d39038091b41c582bc5d7a`; all three roles remain `PENDING_LOCAL_SNAPSHOT`. Official docs establish pinned `claude-sonnet-4-6`, pinned `claude-opus-4-8`, and stable `gemini-3.6-flash` as `READY_FOR_PROVIDER_PREFLIGHT`, but this does not authorize calls. `grok-4.5` remains pending because the public undated ID is mutable and no dated fixed identity was verified; `gpt-5.5` remains unverified in OpenAI's official catalog. The no-execution identity gate has 65 passing tests and one Windows symlink-permission skip. Independent review is not complete because every review agent was routed to unavailable `grok-4.6`; do not claim an independent approval. No model download/load or provider probe has occurred.

The user-approved shared-cache cleanup removed six obsolete complete models plus one empty ref directory from `/NAS/yesh/hf_cache/hub`, releasing `102,106,640,384` allocated bytes (about 95.1 GiB). The atomic receipt is `/NAS/yesh/MemUpdateBench/external/post_core_storage_cleanup_20260821_v1.json`, SHA-256 `7ca169060d061852635872b1cfe13b068fa0a252f01af39de44d85593f3ba71e`. The cache now retains exactly `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` for Task 11/long25 and `sentence-transformers/all-MiniLM-L6-v2@c9745ed1d9f207416be6d2e6f8de32d1f16199bf` for Pilot/heuristic CRUD. Long25 and the independent Task 11 Mistral snapshot were rechecked.

The open-model snapshots are now complete and independently audited in the shared library: `/NAS/HuggingFaceModels/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, tree `e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db`; `/NAS/HuggingFaceModels/Muse-Glimmer-30B-GGUF@70bf1b61ac09f91b24d39038091b41c582bc5d7a`, tree `55357aa0a0a9dfe738725f864eb4183e9aa2a0a84da1245b13c47bd85ce9f90f`; and `/NAS/HuggingFaceModels/Muse-Glimmer-30B@a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, tree `7a90420d22f8c98737f15bc31473bbe8a3579ee95f9bf2237172679709877782`. The public closure receipt is `/NAS/yesh/MemUpdateBench/external/post_core_public_open_models_20260822_v1.json`, SHA-256 `77a69e02a8b092b7e1bf5e89ff9a5f69b449c89a1c2cd319f9c48edd3e2f4645`; independent audit SHA-256 is `0b146bd8dc04e3343d899801f4746bee0ae69635f1ace3f4c92ada8f32819940`. All personal Muse cache/staging duplicates are absent, required old dependencies remain, and audit-time NAS availability was `102,711,689,216` bytes. These are authenticated storage inputs only: model loads, runtime compatibility, capability, and benchmark evidence remain not started.

Canonical vNext memory-object identity is exactly:

```text
(namespace, entity, attribute, subkey)
```

`object_type` is classification metadata and is excluded from identity, semantic hashes, replay keys, and exact object resolution. Imported P6.x `(entity, attribute)` slots use `namespace="default"` and `subkey=null`; legacy phase and metric names remain provenance rather than canonical namespaces.

Authoritative vNext references:

```text
docs/specs/memupdatebench_vnext_benchmark_design.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-phase0-contract-legacy-bridge.md
docs/superpowers/plans/2026-07-20-memupdatebench-vnext-pilot.md
docs/vnext/legacy_bridge.md
WORKFLOW.md
```

The Pilot consumes the Phase 0 contracts without introducing parallel task, runtime, score, capability, or manifest dictionaries. Its built-in checkpoint uses deterministic `slot_direct`; it is not external-system or prompted-answer evidence. Reference covers all 1,440 tasks; raw append, exact CRUD, and heuristic CRUD explicitly mark the 360 Family C multi-object-answer tasks unsupported. Files under `tests/vnext/fixtures/legacy/` remain immutable authenticated regression inputs.

## Important Local Files

Core package:

```text
mub/config.py
mub/utils.py
mub/memory/entry.py
mub/memory/store.py
mub/manager/memory_manager.py
```

Core scripts:

```text
scripts/prepare_data.py
scripts/eval_evomemory.py
scripts/analyze_ood_errors.py
scripts/analyze_action_pathology.py
scripts/summarize_update_frequency.py
scripts/summarize_prompt_robustness.py
scripts/smoke_test.py
scripts/generate_constrained_sft.py
scripts/train_constrained_sft.py
scripts/probe_api_answer_model.py
scripts/summarize_api_latest_model_probe.py
scripts/run_p84_api_latest_models_tang2.sh
```

Main data:

```text
data/evomemory_update_frequency_hard_k1_p63_dev.json
data/evomemory_update_frequency_hard_k1_p63_test.json
data/evomemory_update_frequency_hard_k2_p63_dev.json
data/evomemory_update_frequency_hard_k2_p63_test.json
data/evomemory_update_frequency_hard_k4_p63_dev.json
data/evomemory_update_frequency_hard_k4_p63_test.json
data/evomemory_update_frequency_hard_k8_p63_dev.json
data/evomemory_update_frequency_hard_k8_p63_test.json
data/evomemory_update_frequency_hard_k16_p63_dev.json
data/evomemory_update_frequency_hard_k16_p63_test.json
data/evomemory_update_frequency_hard_p63_train.json
data/evomemory_update_frequency_hard_p63_dev.json
data/evomemory_update_frequency_hard_p63_test.json
```

Main results:

```text
results/update_frequency_p63/
results/update_frequency_p63_summary/
results/p65_prompt_robustness/
results/p65_prompt_robustness_summary/
results/p65_diagnostics/
results/p65_stability/
results/p83_conflict_type_probe_summary/
results/p83_stale_conflict_dose_summary/
results/p83_stale_specific_removal_trace/
results/p84_api_latest_model_probe_summary/
```

Historical context:

```text
docs/HISTORY.md
docs/PROJECT_WORKFLOW6.0.0.md
docs/PROJECT_WORKFLOW6.0.1.md
docs/PROJECT_WORKFLOW6.1.0.md
docs/PROJECT_WORKFLOW6.2.0.md
docs/PROJECT_WORKFLOW6.3.0.md
docs/PROJECT_WORKFLOW6.4.0.md
docs/MIGRATION_PLAN.md
```

## Main Commands

Compile and smoke test:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/analyze_action_pathology.py scripts/summarize_update_frequency.py scripts/summarize_prompt_robustness.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py
python scripts/smoke_test.py
```

Rebuild P6.3 summary artifacts:

```bash
python scripts/summarize_update_frequency.py \
  --result_root results/update_frequency_p63 \
  --output_dir results/update_frequency_p63_summary
```

Run deterministic oracle on a hard split:

```bash
PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode constrained_slot_crud \
  --answer_mode slot_direct \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/sanity_oracle_k16
```

Run learned long25 if checkpoint is available:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/eval_evomemory.py \
  --mode learned_constrained_slot \
  --answer_mode slot_prompt \
  --no_qlora \
  --lora_checkpoint checkpoints/long25/best \
  --data_file data/evomemory_update_frequency_hard_k16_p63_test.json \
  --output_dir results/long25_slot_prompt_k16_rerun
```

## Cluster Usage

Primary cluster target:

```text
Tang-2-Wu
```

New remote project path:

```text
/NAS/yesh/MemUpdateBench
```

Activate environment:

```bash
source /NAS/yesh/MemUpdateBench/activate.sh
```

Continue using the existing `gmsra` conda environment; only `PYTHONPATH` changes.

Remote `activate.sh` should contain:

```bash
cd /NAS/yesh/MemUpdateBench
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONPATH=/NAS/yesh/MemUpdateBench
echo "MemUpdateBench environment ready ✅"
```

Only modify files inside:

```text
/NAS/yesh/MemUpdateBench
```

Use `tmux` for long-running cluster jobs.

Typical pattern:

```bash
tmux new-session -d -s <session_name> \
  "cd /NAS/yesh/MemUpdateBench && source activate.sh && CUDA_VISIBLE_DEVICES=<gpu> python <script.py> <args> > <log_file> 2>&1"
```

Use `--no_qlora` for learned constrained evals if bitsandbytes 4-bit loading fails.

## Checkpoint Status

The learned long25 baseline should live at:

```text
checkpoints/long25/best
```

Migration note: before deleting old remote `/NAS/yesh/G-MSRA`, move the checkpoint from:

```text
/NAS/yesh/G-MSRA/outputs/constrained_sft_curriculum_long25
```

to:

```text
/NAS/yesh/MemUpdateBench/checkpoints/long25
```

Do not delete `/NAS/yesh/G-MSRA` unless the user explicitly asks and checkpoint migration has been verified.

## Coding Rules

Prefer minimal, targeted changes. Do not restore old Phase 1-5 G-MSRA components unless explicitly needed.

Do not reintroduce these old modules into the mainline:

```text
agent.py
reward/
consolidation/
train_phase*.py
eval_locomo.py
run_ablations.py
run_baselines.py
```

For parser/data-generator changes, run at least:

```bash
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/smoke_test.py
python scripts/smoke_test.py
```

For any new split:

1. generate/verify data,
2. run deterministic oracle first,
3. analyze errors if oracle is imperfect,
4. only then evaluate learned managers.

Avoid adding comments unless they explain a non-obvious invariant.

Commits are allowed when they are useful for preserving a coherent completed unit of work. Before committing, make sure the staged diff is intentional, validation has been run or the reason for skipping validation is recorded, and the commit message accurately describes the change.

## Workflow Documentation

This new project uses a single `WORKFLOW.md` plus historical docs under `docs/`.

When completing a substantial phase, append to `WORKFLOW.md` rather than creating many new numbered project files, unless the user asks for versioned workflow files.

Workflow entries should include:

- motivation,
- commands run,
- files changed/generated,
- metrics,
- error analysis,
- conclusions,
- next steps.

## Recommended Next Work

The Core task-release and bounded overall Core gates are closed. Do not regenerate, overwrite, rebind, or add files inside `data/vnext/core/v3`; preserve all Task 9–14 root hashes and evidence classifications.

Default priority order:

1. **Obtain explicit runtime/model-load authorization for the open-model preflight.** Exact Qwen/Muse public snapshots are authenticated and ready as storage inputs. The next executable unit must freeze runtime/package/engine revisions, approved A40/A100 devices, context/token caps, trust-remote-code policy, load/VRAM/time limits, visible prompt/parse smoke, and a no-benchmark-claim boundary. A generic continuation must not implicitly load a model.
2. **Keep closed-provider preflight separately blocked.** Claude Sonnet 4.6, Claude Opus 4.8, and Gemini 3.6 Flash have document-verified IDs but still require explicit network, credential-environment, one-probe, timeout/token, and hard-cost authorization. Grok 4.5 and GPT-5.5 are not ready for provider preflight.
3. **Design the broader main-track external-validity expansion around genuine systems and data.** Keep it separate from both the `FINAL_APPROVED` Core release and the Phase 0 model registry. Add genuine external-memory prompted-answer systems, more independent semantic cores, broader families/domains/languages, and a blocked answer-model panel rather than an uncontrolled Cartesian product.
4. **Integrate the manuscript around the frozen evidence.** Define methods, selectors, metrics, context order, version metadata, semantic-core bootstrap, unsupported/null policy, and evidence classes before numbers. Retain the narrow order- and metadata-sensitive version-arbitration claim and the P8.3/Core surface boundary.
5. **Preserve reproducibility anchors.** Keep the two revision-pinned offline models and exact Task 12/13/14/post-Core Phase 0/identity hashes as frozen anchors. Any future closed-provider model must use its verified public provider/model identity, not an internal routing alias.
6. **Do not retrofit future experiments into Core, Phase 0, or the identity evidence.** New external systems, data, models, calls, or claims require a new scoped execution/evidence release rather than modifying Task 14 outputs, immutable Core roots, the Phase 0 metadata root, or the frozen identity file.

For every downstream run, distinguish state, lifecycle/history, retrieval/evidence, and answer layers; run deterministic reference sanity before model interpretation; record unsupported work explicitly rather than as zero or omission; and document commands, counts, metrics, errors, and conclusions in `WORKFLOW.md`.

Avoid spending major time on prose-only polishing until Core external validity, manager coverage, prompted answering, and evidence-ledger gates are complete. Commits are allowed for coherent completed units after checking the diff and recording validation status.
