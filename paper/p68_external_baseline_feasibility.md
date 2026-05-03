# P6.8 External Baseline Feasibility

## Motivation

`docs/critical_review.md` identifies the lack of real external baselines as the largest reviewer-risk item. This note records the first Mem0 feasibility probe. The goal is not to claim a result before the adapter is validated, but to determine whether a real external memory SDK can be run on MemUpdateBench while exposing enough state to measure stale same-slot burden.

## Scope

Initial target:

```text
data/evomemory_update_frequency_hard_k16_p63_dev.json
limit: 20 examples first
```

Required feasibility checks:

1. ingest all episode events into Mem0;
2. answer the benchmark question;
3. inspect stored memory entries or enough state to estimate memory size and stale same-slot burden;
4. write an `evomemory_results.json`-like payload for downstream summaries.

## Adapter

```text
scripts/eval_mem0_baseline.py
```

The adapter is intentionally separate from the old `mub/baselines.py` project-local reproductions, because those are not genuine external-system integrations and reference older G-MSRA modules. The new adapter attempts to import a real Mem0 package, runs a small subset, and writes an explicit unavailable report if Mem0 is not installed.

## Current environment status

Local environment:

```text
mem0: unavailable
mem0ai: unavailable
```

Sui-3-Wu `gmsra` environment:

```text
mem0: unavailable
mem0ai: unavailable
```

Attempted isolation:

- Python `venv` under `/NAS/yesh/MemUpdateBench/.venv_mem0` failed because system Python lacks `ensurepip`.
- Conda prefix environment creation under `/NAS/yesh/MemUpdateBench/.conda_mem0` failed because shared conda requires Anaconda channel Terms of Service acceptance; this should not be modified silently.
- `mem0ai` was successfully installed into project-local target directory `/NAS/yesh/MemUpdateBench/external/mem0_vendor` using `pip --target` from the existing `gmsra` Python. This avoids modifying the shared environment.
- Import check succeeded with `PYTHONPATH=/NAS/yesh/MemUpdateBench/external/mem0_vendor:/NAS/yesh/MemUpdateBench`; `mem0.Memory` is available.
- Default Mem0 initialization fails because it requires `OPENAI_API_KEY`.
- Non-OpenAI config probing with `huggingface` embedder and `faiss` vector store currently fails in the offline cluster because Mem0 tries to fetch `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace and DNS/network resolution fails. This is an environment/configuration blocker, not a benchmark result.

## Current outputs

Unavailable-path reports:

```text
results/p68_mem0_feasibility/local_unavailable_k16_dev20/
results/p68_mem0_feasibility/gmsra_unavailable_k16_dev20/
```

These are environment feasibility records, not external-baseline results.

## Next checks

To continue Mem0 feasibility, provide one of:

1. an `OPENAI_API_KEY` or other configured API endpoint acceptable for Mem0's LLM/embedder providers;
2. a confirmed local HuggingFace embedding model path usable by Mem0's `huggingface` embedder;
3. an Ollama/vLLM/lmstudio endpoint plus local embedder config that initializes without internet access.

Then run:

```bash
PYTHONPATH=/NAS/yesh/MemUpdateBench/external/mem0_vendor:/NAS/yesh/MemUpdateBench \
  /NAS/yesh/miniconda3/envs/gmsra/bin/python scripts/eval_mem0_baseline.py \
  --data_file data/evomemory_update_frequency_hard_k16_p63_dev.json \
  --output_dir results/p68_mem0_feasibility/mem0_k16_dev20 \
  --limit 20 \
  --run_id k16_dev20
```

Interpretation gate:

- If Mem0 can answer but memory inspection is unavailable, report answer metrics only and do not compare stale burden.
- If memory inspection is available but parseable same-slot values are not, report memory size and inspectability limits.
- If both answer and inspectable memory work, expand to k=1/2/4/8/16 dev before test.
