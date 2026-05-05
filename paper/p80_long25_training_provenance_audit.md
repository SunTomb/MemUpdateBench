# P8.0 Long25 Training Provenance Audit

## Purpose

The P8.0 plan asks whether the large gap between the original P6.3 Long25 row and the later P6.5 stability row should be interpreted as seed sensitivity or as a broader training-recipe/provenance difference. This note records what can be concluded from the currently available local artifacts.

## Artifacts inspected

```text
paper/p70_long25_reproducibility_note.md
paper/p65_long25_stability_note.md
results/p70_long25_reproducibility/long25_provenance.{json,csv}
results/p65_stability/long25_seed_stability_summary.{json,csv,md}
scripts/run_p65_long25_sharded_tang3.sh
scripts/train_constrained_sft.py
docs/PROJECT_WORKFLOW6.1.0.md
docs/PROJECT_WORKFLOW6.2.0.md
WORKFLOW.md
```

## Known result families

| Family | Checkpoint | Evaluation split | EM | F1 | State acc. | Stale same-slot | Final memory size |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| P6.3 original Long25 | `outputs/constrained_sft_curriculum_long25/best` | P6.3 hard k=16 test | 0.480 | 0.530 | 0.910 | 1.13 | 9.43 |
| P6.5 seed 11 | `outputs/p65_long25_seed11/best` | P6.3 hard k=16 test | 0.870 | 0.903 | 0.940 | 0.10 | 2.06 |
| P6.5 seed 22 | `outputs/p65_long25_seed22/best` | P6.3 hard k=16 test | 0.880 | 0.913 | 0.990 | 0.07 | 2.04 |
| P6.5 seed 33 | `outputs/p65_long25_seed33/best` | P6.3 hard k=16 test | 0.890 | 0.910 | 0.970 | 0.05 | 2.02 |

The evaluation side is aligned for the comparison above: all rows use the same P6.3 hard k=16 test split and `answer_mode=slot_prompt`. The contradiction is therefore not caused by comparing different test files or answer modes.

## What is verified

1. The original P6.3 row and the P6.5 rows are not the same checkpoint family.
2. The P6.5 family is internally stable across seeds 11/22/33: EM range is only 0.870-0.890 and final memory size is 2.02-2.06.
3. The P6.5 family is much more compact than the P6.3 original checkpoint and nearly eliminates stale same-slot retention.
4. Existing local records do not contain enough training-command provenance to prove that the P6.3-vs-P6.5 gap is a pure random-seed effect.

## Training-recipe evidence currently available

`scripts/train_constrained_sft.py` exposes the relevant training knobs:

```text
--train_file
--dev_file
--output_dir
--model_name / --checkpoint
--num_epochs
--max_steps
--train_limit / --eval_limit
--max_length
--learning_rate
--grad_accum_steps
--lora_r / --lora_alpha
--use_qlora / --load_in_4bit
--seed
```

The available P6.5 evaluation runner records only evaluation-time parameters and checkpoint paths:

```text
scripts/run_p65_long25_sharded_tang3.sh
```

It does not record the original training commands used to produce `outputs/p65_long25_seed{11,22,33}/best`. The historical workflow notes identify `outputs/constrained_sft_curriculum_long25/best` as the older Long25 checkpoint, but they likewise do not provide a complete matched training command for the original checkpoint in this project checkout.

## Conclusion

The safe conclusion is **checkpoint/training-provenance sensitivity**, not pure seed sensitivity.

The evidence supports saying that a later independently trained/reseeded Long25 family is stable and substantially stronger than the original P6.3 checkpoint. It does **not** support saying that changing only the random seed explains the whole 0.48 -> 0.87-0.89 EM jump, because the local artifacts do not prove that dataset, checkpoint initialization, max steps, epoch count, LoRA settings, or curriculum recipe were identical.

## Paper wording

Use wording like:

> The original Long25 checkpoint and the later three-seed Long25 family are reported separately because they are different checkpoint families. The later family is stable across seeds and substantially more compact, but available provenance does not establish a pure seed-only comparison; we therefore treat this as training/checkpoint provenance sensitivity rather than as a clean random-seed effect.

Avoid wording like:

> Long25 has high seed sensitivity.

unless a future matched retraining sweep records and fixes every training variable except `--seed`.

## Recommended next verification

If this issue becomes paper-critical, run a matched three-seed retraining sweep with an explicit manifest per seed:

```text
train_file
dev_file
base model/checkpoint
num_epochs
max_steps
train_limit/eval_limit
max_length
learning_rate
grad_accum_steps
lora_r/lora_alpha
qlora/load_in_4bit flags
seed
best-checkpoint selection metric
```

Until then, report P6.3 original and P6.5 stable family as separate Long25 checkpoint families.
