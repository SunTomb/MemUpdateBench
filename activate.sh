cd /NAS/yesh/MemUpdateBench
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONPATH=/NAS/yesh/MemUpdateBench
echo "MemUpdateBench environment ready ✅"
