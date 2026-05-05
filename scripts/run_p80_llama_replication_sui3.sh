#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_NAME=${MODEL_NAME:-/NAS/HuggingFaceModels/Llama3.1-8B-Instruct}
OUT_ROOT=results/p80_multimodel_stale_susceptibility/llama31_8b
DATA=data/evomemory_update_frequency_hard_k16_p63_dev.json
mkdir -p "${OUT_ROOT}"
run_eval() {
  local name="$1"
  shift
  echo "START ${name} $(date)"
  python scripts/eval_evomemory.py \
    --model_name "${MODEL_NAME}" \
    --mode raw_add \
    --answer_mode slot_prompt \
    --data_file "${DATA}" \
    --output_dir "${OUT_ROOT}/${name}" \
    --answer_topk 5 \
    --save_answer_traces \
    --no_qlora \
    --lora_checkpoint "" \
    "$@"
  echo "DONE ${name} $(date)"
}
run_eval raw_add_normal_topk5 --retrieval_policy normal --context_order normal --context_annotation none
run_eval raw_add_latest_per_slot_topk5 --retrieval_policy latest_per_slot --context_order normal --context_annotation none
run_eval raw_add_latest_label_topk5 --retrieval_policy normal --context_order normal --context_annotation latest_outdated_label
run_eval raw_add_chronological_topk5 --retrieval_policy normal --context_order chronological --context_annotation none
run_eval raw_add_reverse_chronological_topk5 --retrieval_policy normal --context_order reverse_chronological --context_annotation none

echo "ALL_DONE $(date)"
