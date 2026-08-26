# Qwen3.5 Full-Matrix Paper Table

Source analysis: `analysis.json` (SHA-256 `9b1dd2066c6d2300a9f0af305a24d32c510c04186f8612fca78e51a813de474f`). Source receipt SHA-256: `529ec8511e574ab6b92f369bd4b4d9007013b426868fd6fe99c7856405d9b5f6`.

| Context condition | k | Rows | EM | Stale copies | Clustered core bootstrap CI | Affected cores |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Chronological, no label | 4 | 160 | 0.98125 | 3 | [0.94375, 1.00000] | 1 |
| Chronological, no label | 8 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Chronological, no label | 16 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, no label | 4 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, no label | 8 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, no label | 16 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, latest/outdated label | 4 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, latest/outdated label | 8 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| Reverse, latest/outdated label | 16 | 160 | 1.00000 | 0 | [1.00000, 1.00000] | 0 |
| **All cells** | **4/8/16** | **1,440** | **0.9979167** | **3** | — | **1** |

The paired reverse/no-label minus chronological/no-label contrast at k=4 is +0.01875, with clustered core bootstrap CI [0.00000, 0.05625]. This is a bounded Qwen3.5 answer-model result over frozen Raw Append Family-A contexts; it is not an external-memory-system result.
