# P7.0 External Baseline Fairness Note

## Motivation

`docs/critical_review_v2.md` argues that the current Mem0 result should not be used as a main-table external baseline because it used a Qwen2.5-VL backend, only ran dev20, and may not give Mem0 a fair text-only configuration. It also argues that original Memory-R1 remains the more relevant external comparison.

This note records the Day-2 feasibility check.

## Mem0 status

The currently runnable Mem0 configuration on Tang-2 is:

```text
config: /NAS/yesh/MemUpdateBench/configs/mem0_vllm_qdrant_minilm384.json
llm.provider: vllm
model: Qwen2.5-VL
vllm_base_url: http://127.0.0.1:8011/v1
embedder: local all-MiniLM-L6-v2 snapshot
vector_store: local Qdrant, 384-dim collection
```

Endpoint probing on Tang-2 found only the following live OpenAI-compatible vLLM model among the checked local ports:

```text
port 8011: Qwen2.5-VL
ports 8000, 8001, 8002, 8010, 8012, 8020: no response
```

A follow-up project-local endpoint was then launched on Sui-3 using `scripts/serve_openai_compatible_transformers.py` with `/NAS/HuggingFaceModels/Qwen2.5-7B-Instruct` at `http://127.0.0.1:8013/v1`. The endpoint passed `/v1/models` and simple chat-completion checks, and Mem0 was rerun with MiniLM forced to CPU via `configs/mem0_qwen25_text_qdrant_minilm384_cpu.json`.

That text-backend Mem0 path did not complete: k=16 dev20 and k=16 dev3 both stalled before the first completed example with repeated Mem0 extraction JSON parse errors. Thus, the remaining blocker is no longer just endpoint availability; it is adapter/extraction compatibility between Mem0's structured extraction prompts and the lightweight transformers server.

Existing completed Mem0 dev20 results remain:

| Variant | N | EM | F1 | Inspectable | Stale same-slot | Same-slot | Gold same-slot | Memory size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first returned memory text | 20 | 0.00 | 0.00 | 1.00 | 1.15 | 1.20 | 0.05 | 20.00 |
| value extraction | 20 | 0.00 | 0.05 | 1.00 | 1.00 | 1.05 | 0.05 | 20.00 |

## Fairness decision

The current Mem0 result should **not** be placed in the main comparison table.

It is useful only as a qualitative runnable-probe result:

- Mem0 can be installed and run end-to-end on the cluster with the Qwen2.5-VL endpoint.
- Its memory state is inspectable in that completed probe.
- Under Qwen2.5-VL dev20, it is badly misaligned with exact repeated same-slot final-value tracking.
- A fairer Qwen2.5-7B-Instruct text endpoint can be launched, but the current lightweight endpoint plus Mem0 adapter fails structured extraction before completing even dev3.

A fair Mem0 main-table row requires at least one of:

1. a text-only OpenAI-compatible endpoint whose outputs satisfy Mem0's structured extraction parser, ideally Qwen2.5-7B-Instruct or comparable;
2. a larger run, at least k=16 dev100/test200 after the backend is fixed;
3. a documented adapter/prompt that gives Mem0 a reasonable chance to preserve exact final slot values.

Until then, report Mem0 in qualitative/appendix feasibility only.

## Memory-R1 status

Local repository search found only:

```text
baselines/memory_r1_agent.py
```

That file is a project-local aligned approximation using this repository's `MemoryStore` and `MemoryManager`; it is not the original Memory-R1 code/checkpoint. Tang-2 `/NAS/yesh` probing did not find an obvious official Memory-R1/ReMemR1 directory or checkpoint.

Therefore, original Memory-R1 is **not available in the current three-day revision window** unless the user/advisor provides the official repository/checkpoint path or permits a fresh external download/setup.

## Paper implication

The manuscript should state:

> We do not currently include a main-table Memory-R1 or fair Mem0 external baseline. We include Mem0 only as a runnable qualitative probe because the only discovered local backend is Qwen2.5-VL and the result is dev20. The project-local Memory-R1-style agent is not the original Memory-R1 implementation and is not reported as an external baseline.

This is weaker than a true external baseline, but safer than presenting an unfair Mem0 row or a project-local approximation as if it were original Memory-R1.
