# P8.0 Lost-in-the-Middle Gold-Position Probe Note

## Purpose

导师建议显式比较 MemUpdateBench 的 context-position sensitivity 与 “Lost in the Middle” 现象。此前 real-context probe 已经比较了 chronological / reverse chronological / current_first / current_last，但这些条件同时改变了多个真实检索上下文因素。

本 probe 更严格：固定同一组 synthetic answer-time context entries，只移动 gold entry 的位置：beginning / middle / end。这样可以直接测试 slot-conditioned answering 是否对 gold 在 context 中的位置敏感。

## Artifacts

```text
scripts/run_lost_in_middle_probe.py
scripts/summarize_lost_in_middle_probe.py
results/p80_lost_in_middle_probe/
results/p80_lost_in_middle_probe_summary/lost_in_middle_summary.{json,csv,md}
```

## Design

For each P6.3 hard k=16 dev example:

1. identify the latest target-slot event as the gold entry;
2. construct a fixed context set with the gold entry plus stale same-slot and near-miss distractors;
3. run the same slot-answer prompt three times:
   - gold at beginning;
   - gold in middle;
   - gold at end;
4. keep context size and distractor content fixed across positions.

Default setting:

```text
data: data/evomemory_update_frequency_hard_k16_p63_dev.json
model: Qwen2.5-7B-Instruct
num examples: 100
num distractors: 8
positions: beginning / middle / end
```

## Results

| Gold position | N | EM | F1 | Answer value present |
| --- | ---: | ---: | ---: | ---: |
| beginning | 100 | 0.010 | 0.073 | 0.040 |
| middle | 100 | 0.090 | 0.183 | 0.190 |
| end | 100 | 0.630 | 0.654 | 0.720 |

## Interpretation

The probe shows a strong context-position effect under fixed distractor content. Moving the gold entry from the end to the middle drops EM from 0.630 to 0.090, and moving it to the beginning drops EM further to 0.010. Answer-value-present follows the same pattern: 0.720 at the end, 0.190 in the middle, and 0.040 at the beginning.

This is a direct Lost-in-the-Middle-style result for MemUpdateBench: even when the gold entry is always present in the context and the distractor set is held fixed, the answer model is much more likely to recover the final slot value when it appears at the end of the context. In this benchmark, the effect looks more like a strong final-position/recency advantage than a symmetric beginning-and-end advantage.

Paper implication:

> Stale-context collapse interacts with a Lost-in-the-Middle-style position effect. Holding context contents fixed, Qwen2.5-7B-Instruct answers correctly far more often when the gold slot entry is placed at the end of the context than when it is placed in the middle or beginning. This helps explain why chronological contexts, where current updates tend to appear later, outperform reverse chronological or normal stale-contaminated contexts.
