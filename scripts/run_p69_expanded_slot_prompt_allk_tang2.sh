#!/usr/bin/env bash
set -euo pipefail

ROOT=/NAS/yesh/MemUpdateBench
cd "$ROOT"
source activate.sh >/dev/null 2>&1
export PYTHONPATH="$ROOT"
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/NAS/yesh/miniconda3/envs/gmsra/bin/python
GPUS=(4 5 7)
SLOTS_PER_GPU=1
TOTAL_WORKERS=$((${#GPUS[@]} * SLOTS_PER_GPU))
RESULT_ROOT=results/p69_expanded_slot_prompt_allk
mkdir -p "$RESULT_ROOT/logs"

shard_size_for_k() {
  local k=$1
  if [[ "$k" == "16" ]]; then echo 10; else echo 20; fi
}

run_shard() {
  local gpu=$1
  local mode=$2
  local k=$3
  local start=$4
  local end=$5
  local data="data/evomemory_update_frequency_expanded_k${k}_p68_dev.json"
  local out_dir="$RESULT_ROOT/${mode}_prompt_k${k}_dev/shard_${start}_${end}"
  local log_file="$RESULT_ROOT/logs/${mode}_k${k}_shard_${start}_${end}_gpu${gpu}.log"
  if [[ -f "$out_dir/evomemory_results.json" ]]; then
    echo "$(date '+%F %T') skip mode=${mode} k=${k} range=[${start},${end}) gpu=${gpu}"
    return 0
  fi
  mkdir -p "$out_dir"
  echo "$(date '+%F %T') start mode=${mode} k=${k} range=[${start},${end}) gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_evomemory.py \
    --mode "$mode" \
    --answer_mode slot_prompt \
    --slot_prompt_variant v0_current \
    --save_answer_traces \
    --no_qlora \
    --data_file "$data" \
    --start_idx "$start" \
    --end_idx "$end" \
    --output_dir "$out_dir" \
    > "$log_file" 2>&1
  echo "$(date '+%F %T') done mode=${mode} k=${k} range=[${start},${end}) gpu=${gpu}"
}

run_worker() {
  local worker_id=$1
  local gpu=$2
  local task_id=0
  for mode in constrained_slot_crud raw_add; do
    for k in 1 2 4 8 16; do
      local shard_size=$(shard_size_for_k "$k")
      for start in $(seq 0 "$shard_size" 199); do
        local end=$((start + shard_size))
        if (( end > 200 )); then end=200; fi
        if (( task_id % TOTAL_WORKERS == worker_id )); then
          run_shard "$gpu" "$mode" "$k" "$start" "$end"
        fi
        task_id=$((task_id + 1))
      done
    done
  done
  echo "$(date '+%F %T') worker ${worker_id} finished on gpu=${gpu}"
}

worker_id=0
for gpu in "${GPUS[@]}"; do
  for slot in $(seq 1 "$SLOTS_PER_GPU"); do
    run_worker "$worker_id" "$gpu" > "$RESULT_ROOT/logs/worker_${worker_id}_gpu${gpu}.log" 2>&1 &
    echo "launched worker=${worker_id} gpu=${gpu} slot=${slot}"
    worker_id=$((worker_id + 1))
  done
done

wait
for mode in constrained_slot_crud raw_add; do
  for k in 1 2 4 8 16; do
    "$PY" scripts/merge_evomemory_shards.py --inputs "$RESULT_ROOT/${mode}_prompt_k${k}_dev" --output_dir "$RESULT_ROOT/${mode}_prompt_k${k}_dev_merged"
  done
done
"$PY" scripts/summarize_update_frequency.py --result_root "$RESULT_ROOT" --output_dir results/p69_expanded_slot_prompt_allk_summary
echo "$(date '+%F %T') expanded slot_prompt all-k jobs finished"
