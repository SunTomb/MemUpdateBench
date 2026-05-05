# P8.0 Simple External Pipeline Baseline Note

## Purpose

导师建议补一个简单外部 pipeline baseline，不必继续卡在 Mem0。这个 note 记录一个 deliberately simple 的 extract-then-store diagnostic baseline：它不使用项目内 oracle manager 作为被比较方法，但复用项目 parser 将事件抽取为 inspectable `(entity, attribute, value)` memory records，再报告与主 benchmark 相同的 state/stale/compactness/answer metrics。

它的角色不是替代 Mem0 或构成外部 SDK leaderboard，而是证明 MemUpdateBench 可以评价一个非 learned、可检查的 extract-then-store pipeline。

## Artifacts

```text
scripts/eval_simple_external_pipeline.py
scripts/summarize_simple_external_pipeline.py
results/p80_simple_external_pipeline/
results/p80_simple_external_pipeline_summary/simple_external_pipeline_summary.{json,csv,md}
```

## Pipeline definition

Two update policies are evaluated:

| Policy | Behavior |
| --- | --- |
| `append` | parse events and append every parsed memory record; this is the extract-then-store analogue of raw append |
| `slot_update` | parse events, then update the latest memory record with the same `(entity, attribute)` slot; this is the simplest structured external-memory pipeline |

The default store policy is `parsed_only`: events that the parser cannot map to a structured slot are treated as external-pipeline NOOPs rather than stored as unstructured memories. The resulting memory records are inspectable and saved when `--save_memory_records` is used.

## Current results

| run | split | update policy | answer mode | EM | F1 | state | stale same-slot | memory size | answer value present |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| append parsed-only | P6.3 k=16 dev | append | slot_direct | 1.000 | 1.000 | 1.000 | 14.250 | 31.000 | 1.000 |
| slot-update parsed-only | P6.3 k=16 dev | slot_update | slot_direct | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 1.000 |
| slot-update parsed-only | P6.3 k=16 test | slot_update | slot_direct | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 1.000 |
| append parsed-only | P6.3 k=16 dev | append | slot_prompt | 0.140 | 0.177 | 1.000 | 14.250 | 31.000 | 0.170 |
| slot-update parsed-only | P6.3 k=16 dev | slot_update | slot_prompt | 0.910 | 0.926 | 1.000 | 0.000 | 2.000 | 1.000 |

## Interpretation

The slot-update pipeline provides a clean external-pipeline row: with explicit slot extraction and overwrite semantics, it preserves final state and removes stale same-slot burden at k=16. Under Qwen2.5-7B-Instruct slot-prompt answering, this simple pipeline reaches EM/F1 0.910/0.926 on P6.3 k=16 dev, much higher than the parsed append counterpart at 0.140/0.177.

The append variant confirms that the same extractor plus append-only storage reproduces the stale-burden problem: final state is still recoverable under slot-direct lookup, but stale same-slot burden remains high and prompted answering collapses. This makes the row useful as a transparent external-pipeline diagnostic baseline rather than an oracle-only sanity check.

This does not prove that realistic external SDKs solve MemUpdateBench. It gives the paper a simple, transparent external-pipeline baseline and separates the question of structured extraction/overwrite semantics from the harder Mem0 compatibility issue.
