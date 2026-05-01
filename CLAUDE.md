# CLAUDE.md

This file gives Claude Code the project context and operating rules needed to continue MemUpdateBench work from a new window.

## Project Overview

MemUpdateBench is the clean project extracted from G-MSRA P6.x. It is no longer a broad RL memory-manager project. The main paper direction is a benchmark/diagnostic study of repeated same-slot updates in external memory systems.

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

Central invariant:

```text
exact slot resolution by (entity, attribute)
```

Most analysis should distinguish:

- stale value retention,
- missed final updates,
- wrong entity grounding,
- wrong attribute parsing,
- answer-layer retrieval/prompt failures.

## Current Thesis

Repeated same-slot update frequency is the main paper axis.

The strongest P6.3 result:

- append-only methods can keep final value recoverable under oracle `slot_direct`,
- but they accumulate stale same-slot entries,
- under `slot_prompt`, stale burden causes severe answer collapse,
- learned compact managers reduce stale burden but can miss final updates or remain incompletely compact.

At k=16 on P6.3 hard:

| method | slot_direct state_acc | slot_prompt EM/F1 | stale same-slot | final memory size |
| --- | --- | --- | --- | --- |
| constrained_slot_crud | 1.00 | 0.70 / 0.70 | 0.00 | 23.00 |
| raw_add | 1.00 | 0.07 / 0.10 | 14.25 | 52.00 |
| heuristic_crud | 1.00 | 0.10 / 0.13 | 7.44 | 26.73 |
| long25 | 0.91 | 0.48 / 0.53 | 1.13 | 9.43 |

Conclusion: this is a tradeoff curve, not a simple method-win story.

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
scripts/summarize_update_frequency.py
scripts/smoke_test.py
scripts/generate_constrained_sft.py
scripts/train_constrained_sft.py
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
python -m py_compile scripts/prepare_data.py scripts/eval_evomemory.py scripts/analyze_ood_errors.py scripts/summarize_update_frequency.py scripts/generate_constrained_sft.py scripts/train_constrained_sft.py scripts/smoke_test.py
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

Do not commit unless the user explicitly asks.

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

Start P6.5 in MemUpdateBench:

1. turn `results/update_frequency_p63_summary/` into paper-ready figure/table assets,
2. draft the experimental narrative around the tradeoff curve,
3. optionally do isolated Mem0 feasibility in a separate environment,
4. do not start repair training until external-baseline needs and paper presentation are settled.

Current k=32 decision: do not add k=32 yet; k=16 already shows decisive separation.
