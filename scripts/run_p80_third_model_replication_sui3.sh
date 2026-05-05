#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_KEY=${MODEL_KEY:-mistral7b}
MODEL_NAME=${MODEL_NAME:-/NAS/HuggingFaceModels/Mistral-7B-Instruct-v0.1}
OUT_ROOT=${OUT_ROOT:-results/p80_multimodel_stale_susceptibility/${MODEL_KEY}}
DATA=${DATA:-data/evomemory_update_frequency_hard_k16_p63_dev.json}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-}
mkdir -p "${OUT_ROOT}"
run_eval() {
  local name="$1"
  shift
  echo "START ${name} $(date)"
  local range_args=(--start_idx "${START_IDX}")
  if [[ -n "${END_IDX}" ]]; then
    range_args+=(--end_idx "${END_IDX}")
  fi
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
    "${range_args[@]}" \
    "$@"
  echo "DONE ${name} $(date)"
}
run_eval raw_add_normal_topk5 --retrieval_policy normal --context_order normal --context_annotation none
run_eval raw_add_latest_per_slot_topk5 --retrieval_policy latest_per_slot --context_order normal --context_annotation none
run_eval raw_add_latest_label_topk5 --retrieval_policy normal --context_order normal --context_annotation latest_outdated_label
run_eval raw_add_chronological_topk5 --retrieval_policy normal --context_order chronological --context_annotation none
run_eval raw_add_reverse_chronological_topk5 --retrieval_policy normal --context_order reverse_chronological --context_annotation none

echo "ALL_DONE $(date)"
