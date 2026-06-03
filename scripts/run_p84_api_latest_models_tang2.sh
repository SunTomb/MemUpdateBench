#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh >/dev/null

if [[ -z "${MUB_API_BASE_URL:-}" ]]; then
  echo "MUB_API_BASE_URL is required" >&2
  exit 1
fi
if [[ -z "${MUB_API_KEY:-}" ]]; then
  echo "MUB_API_KEY is required" >&2
  exit 1
fi

MODELS=(
  gpt-5.5
  gpt-5.4
  gpt-5.4-mini
  gemini-2.5-flash
  gemini-2.5-pro
  gemini-3-flash-preview
  gemini-3-pro-preview
  gemini-3.1-flash-lite-preview
)

mkdir -p logs results/p84_api_latest_model_probe

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  echo "=== Running ${model} ==="
  MUB_API_MODEL="${model}" PYTHONPATH=. python scripts/probe_api_answer_model.py \
    --connectivity \
    --synthetic-dose-probe \
    --stale-counts 0,1,2,4,8,16 \
    --examples-per-condition 16 \
    --output-dir results/p84_api_latest_model_probe \
    --timeout 120 \
    > "logs/p84_api_${safe_model}.log" 2>&1
  echo "=== Completed ${model} ==="
done
