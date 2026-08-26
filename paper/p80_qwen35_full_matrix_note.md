# P8.0 Qwen3.5 Full-Matrix Boundary Note

## Paper-facing conclusion

The Qwen3.5-9B answer-model result narrows the mechanism claim. Across frozen Raw Append Family-A contexts, Qwen3.5 does **not** reproduce the earlier reverse-order, no-label high-k collapse. The appropriate claim is therefore an **order- and metadata-sensitive version-arbitration failure that is model- and prompt/parser-configuration dependent**, not a universal memory-system response to stale context.

This is bounded answer-model evidence over a fixed memory-manager/context construction. It does not establish broad external-memory-system robustness, manager diversity, or natural-history validity.

## Evidence chain

| Stage | Scope and result | Evidence |
| --- | --- | --- |
| Canary | 320/320 exact matches; 0 stale copies | `/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_canary_13983f4_v1`; receipt payload SHA-256 `d763661bc3e4e65a1652dbd732bf7a4f7929ec8c8f479f076ea19ff1a304480` |
| Confirmatory subset | 480/480 exact matches; 0 stale copies across chronological/no-label, reverse/no-label, and reverse/labeled k=16 | `/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_confirmatory_13983f4_v1`; receipt file SHA-256 `1ee0ce9210158426c598471747428c7889150bb03a72a47d0e57164e7f6f96fa` |
| Full matrix | 1,440 rows; 1,437 exact matches; EM 0.9979167; 3 stale copies; 8/9 cells perfect | `/NAS/yesh/MemUpdateBench/results/vnext/post_core_qwen35_full_matrix_46e1e1f_v1`; source receipt SHA-256 `529ec8511e574ab6b92f369bd4b4d9007013b426868fd6fe99c7856405d9b5f6` |
| Local analysis | semantic-core cluster bootstrap and paired contrasts | `results/post_core_qwen35_full_matrix_analysis/analysis.json`; SHA-256 `9b1dd2066c6d2300a9f0af305a24d32c510c04186f8612fca78e51a813de474f` |

## Full-matrix result and error mechanism

The only non-perfect cell is chronological/no-label k=4 (EM 0.98125; 3/160 stale copies). Its clustered semantic-core bootstrap CI is [0.94375, 1.00000], with one affected core at EM 0.625. Every chronological/no-label k=8/16, reverse/no-label k=4/8/16, and reverse-labeled k=4/8/16 cell is perfect. The paired semantic-core contrast, reverse/no-label minus chronological/no-label at k=4, is +0.01875 with CI [0.00000, 0.05625].

All three failures are in `core_c40d565fabd02e01`. Across two surface tasks, Qwen predicts `ALPHA-01` rather than gold `CORE-04`; retrieved values are `GAMMA-01`, `CORE-15`, `ALPHA-01`, `CORE-04`. The gold is retrieved and final in the sequence, so the observed tail is positional/retrieval-sensitive stale copying rather than a gold-not-retrieved failure.

## Capability-smoke boundary

The corrected formal BASE smoke is operational qualification, not paper performance evidence. Its formal outcomes are Qwen3.5 8/8, GPT-5.5 route 8/8, Grok-4.5 route 8/8, Gemini 3.6 Flash 7/8, Claude Sonnet 4.6 4/8, Claude Opus 4.8 0/8 (HTTP 400), and Muse Glimmer GGUF 0/8 (parser mismatch). It has 56 formal attempts, 35 passes, 21 failures, 40 formal provider calls, and zero canary or benchmark generations. A separate wrong-provider diagnostic produced 64 HTTP 404 calls; these are excluded from formal outcomes but retained in total accounting (104 real provider calls).

Source: `D:/USTC/2026Winter/MemUpdateBench_qualification_inputs/phase12_0981a38_v1/post_core_capability_smoke_base_20260826_v2.json`; payload SHA-256 `1006bd7da2cded78a19e9d42a1a7a3cffc406a4fc6fa9e3e1f312b62bced7cfe`.

Route success does not establish immutable upstream identity. GPT-5.5 and Grok-4.5 retain their mutable/unverified official-identity caveats; document-verified Claude identities are not promoted by their partial or failed transfer-route outcomes.
