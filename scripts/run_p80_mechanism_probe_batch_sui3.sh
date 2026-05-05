#!/usr/bin/env bash
set -euo pipefail
cd /NAS/yesh/MemUpdateBench
source activate.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p results/p80_mechanism_probes
run_probe() {
  local name="$1"
  shift
  echo "START ${name} $(date)"
  python scripts/eval_evomemory.py \
    --mode raw_add \
    --answer_mode slot_prompt \
    --data_file data/evomemory_update_frequency_hard_k16_p63_dev.json \
    --output_dir "results/p80_mechanism_probes/${name}" \
    --save_answer_traces \
    --no_qlora \
    "$@"
  echo "DONE ${name} $(date)"
}
run_probe normal_order_none --context_order normal --context_annotation none
run_probe current_first_none --context_order current_first --context_annotation none
run_probe current_last_none --context_order current_last --context_annotation none
run_probe chronological_none --context_order chronological --context_annotation none
run_probe reverse_chronological_none --context_order reverse_chronological --context_annotation none
run_probe normal_timestamp --context_order normal --context_annotation timestamp
run_probe normal_latest_outdated_label --context_order normal --context_annotation latest_outdated_label
