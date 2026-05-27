#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh

PYTHONPATH=. python scripts/run_conflict_type_probe.py \
  --model_name /NAS/HuggingFaceModels/Qwen2.5-7B-Instruct \
  --examples_per_condition 128 \
  --distractor_count 4 \
  --conditions final_only,unrelated_distractors,same_entity_different_attribute,different_entity_same_attribute,stale_same_slot \
  --output_dir results/p83_conflict_type_probe/qwen25_7b_d4 \
  --no_qlora

PYTHONPATH=. python scripts/summarize_conflict_type_probe.py \
  --input_csv results/p83_conflict_type_probe/qwen25_7b_d4/conflict_type_examples.csv \
  --output_dir results/p83_conflict_type_probe_summary/qwen25_7b_d4
