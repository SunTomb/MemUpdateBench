#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh

PYTHONPATH=. python scripts/run_synthetic_same_slot_probe.py \
  --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct \
  --examples_per_condition 64 \
  --stale_counts 0,1,2,4,8,16 \
  --value_policies conflict \
  --context_orders chronological,reverse_chronological,middle,random \
  --context_annotations none,latest_outdated_label \
  --output_dir results/p83_stale_conflict_dose/qwen25_7b \
  --no_qlora

PYTHONPATH=. python scripts/summarize_synthetic_same_slot_probe.py \
  --input_csv results/p83_stale_conflict_dose/qwen25_7b/synthetic_same_slot_examples.csv \
  --output_dir results/p83_stale_conflict_dose_summary/qwen25_7b
