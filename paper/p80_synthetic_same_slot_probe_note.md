# P8.0 Synthetic Same-Slot Mechanism Probe Note

## Purpose

`docs/critical_review_v3.md` asked for controlled experiments to distinguish why stale same-slot contexts cause answer collapse. The previous context mechanism probe used real retrieved contexts, so retrieval composition was fixed but gold retrieval was only 0.36. This synthetic probe holds gold presence fixed and directly varies stale count, value conflict, order, and explicit labels.

## Artifacts

```text
scripts/run_synthetic_same_slot_probe.py
scripts/summarize_synthetic_same_slot_probe.py
results/p80_synthetic_same_slot_probe/synthetic_same_slot_examples.csv
results/p80_synthetic_same_slot_probe/synthetic_same_slot_summary.{json,csv,md}
results/p80_synthetic_same_slot_probe_analysis/synthetic_same_slot_grouped_summary.{json,csv,md}
```

Remote execution:

```text
node: Sui-3-Wu
model: Qwen/Qwen2.5-7B-Instruct
examples_per_condition: 16
attributes: location, company, language, preference
stale_counts: 0, 1, 2, 4, 8, 16
value_policies: conflict, same_as_current
context_orders: chronological, reverse_chronological
context_annotations: none, latest_outdated_label
```

Total examples: 768.

## Probe design

Each synthetic example constructs a same-slot context with one current final value and `n` stale same-slot entries.

- `value_policy=conflict`: stale entries have values different from the current value.
- `value_policy=same_as_current`: stale entries repeat the current value, removing semantic conflict while preserving repetition and context length.
- `chronological`: stale entries appear first and current appears last.
- `reverse_chronological`: current appears first and stale entries follow.
- `latest_outdated_label`: entries are explicitly marked `[latest]` or `[outdated]`.

This separates:

1. value conflict / majority-like competition;
2. order sensitivity;
3. semantic version disambiguation;
4. attention or formatting dilution from repeated same-value entries.

## Main results

### Conflict values: reverse chronological without labels collapses

| Order | Annotation | stale=0 | stale=1 | stale=2 | stale=4 | stale=8 | stale=16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chronological | none | 0.938 | 1.000 | 0.875 | 0.938 | 0.812 | 0.750 |
| chronological | latest/outdated | 0.938 | 1.000 | 0.938 | 0.938 | 0.938 | 0.938 |
| reverse chronological | none | 0.938 | 0.188 | 0.062 | 0.062 | 0.000 | 0.000 |
| reverse chronological | latest/outdated | 0.938 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |

Interpretation:

- Without labels, putting the current value first and then listing conflicting stale values is catastrophic.
- Even one conflicting stale entry after the current value drops EM from 0.938 to 0.188.
- With two or more conflicting stale entries after the current value, EM is near zero.
- Explicit `[latest]` / `[outdated]` labels almost completely repair this collapse.

This is strong evidence for semantic/version disambiguation plus order sensitivity. The model is not simply using the first occurrence; when conflicting later entries appear, it often follows or is confused by them unless version labels are explicit.

### Conflict values: chronological order is relatively robust

When stale entries appear first and the current value appears last, EM remains high even without labels, although it declines at larger stale counts:

```text
chronological none: stale 0/1/2/4/8/16 -> EM 0.938/1.000/0.875/0.938/0.812/0.750
```

This suggests a recency/final-position advantage in the synthetic context. The result differs from real retrieval contexts, where retrieval order is not cleanly chronological and gold is often missing.

### Same-value repetition still hurts

When stale entries repeat the current value, there is no semantic value conflict. Nevertheless, chronological no-label EM drops as repetition grows:

```text
same_as_current chronological none: stale 0/1/2/4/8/16 -> EM 0.938/0.875/0.875/0.562/0.500/0.562
```

Because answer-value-present remains 1.000 across these conditions, the model often includes or formats the right value in a way that fails exact EM, or produces extra text/value combinations. This supports an attention/formatting dilution effect distinct from value conflict.

## Mechanism conclusions

This probe supports three separate mechanisms:

1. **Semantic version ambiguity:** explicit `[latest]` / `[outdated]` labels can rescue conflict-heavy contexts.
2. **Order sensitivity:** whether stale entries appear before or after the current value drastically changes behavior.
3. **Repetition/attention-format dilution:** even non-conflicting repeated current values reduce exact EM at higher counts.

It does not support a simple majority-vote-only story. In the conflict + chronological condition, many stale values precede one current value, yet the model often answers correctly because the current value is last. In the conflict + reverse-chronological condition, one current value precedes stale values and the model collapses. Order and version semantics dominate raw count.

## Relationship to real-context probes

The real-context k=16 probe found:

```text
normal none EM=0.110
chronological none EM=0.230
latest/outdated label EM=0.260
```

The synthetic probe shows that when gold is always present and context is cleanly controlled, labels can be much more effective. Therefore, the limited gain in the real-context label probe is likely due to retrieval/context composition problems: gold is retrieved only 0.36 of the time, and real contexts include same-name/entity distractors, not just same-slot update chains.

## Caveats

- Synthetic examples are cleaner than benchmark retrieval contexts.
- The original pilot used 16 examples per condition; a focused P8.1 rigor rerun now expands the most diagnostic 21 cells to 64 examples each under `results/p81_synthetic_same_slot_probe_expanded/`.
- The expanded rerun preserves the headline qualitative picture: reverse-chronological conflict without labels still collapses (stale=1/2/8/16 EM 0.234/0.094/0.000/0.031), latest/outdated labels still repair it strongly, chronological conflict remains much more robust (stale=16 EM 0.797), and same_as_current retains answer-value-present 1.000 despite lower exact EM.
- EM may understate correctness in same-as-current conditions because answer-value-present is 1.000 while exact EM drops; the dedicated P8.1 same-as-current failure analysis now classifies these cases separately.
- This probe should be treated as mechanism evidence, not benchmark performance.

## P8.1 expanded-subset confirmation

The P8.1 rigor pass reran the most diagnostic subset with 64 examples per cell:

- `conflict + reverse_chronological + none`, stale `1,2,4,8,16`
- `conflict + reverse_chronological + latest_outdated_label`, stale `1,2,4,8,16`
- `conflict + chronological + none`, stale `1,2,4,8,16`
- `same_as_current + chronological + none`, stale `4,8,16`
- `same_as_current + reverse_chronological + none`, stale `4,8,16`

Expanded results:

| Condition | stale=1 | stale=2 | stale=4 | stale=8 | stale=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| conflict + chronological + none | 1.000 | 0.938 | 0.906 | 0.875 | 0.797 |
| conflict + reverse_chronological + none | 0.234 | 0.094 | 0.156 | 0.000 | 0.031 |
| conflict + reverse_chronological + latest/outdated | 0.969 | 0.969 | 0.891 | 1.000 | 1.000 |
| same_as_current + chronological + none | — | — | 0.688 | 0.547 | 0.531 |
| same_as_current + reverse_chronological + none | — | — | 0.906 | 0.812 | 0.781 |

These larger-n results strengthen the original mechanism claim rather than changing it. Conflict-heavy reverse ordering still collapses rapidly without labels, label semantics still largely repair the conflict cases, and same-as-current conditions still look like non-exact answer degradation rather than semantic value loss because answer-value-present remains 1.000 throughout.

## P8.1 same-as-current error classification

A dedicated follow-up analysis at `results/p81_same_as_current_failure_analysis/` classifies `same_as_current` rows where `EM=0` but `answer_value_present=1`. The main implication is that these failures should not be interpreted as stale-value selection errors: they are predominantly non-exact answer realizations once the correct value is already present in the output, which supports the earlier interpretation of formatting / answer-surface dilution rather than semantic confusion.

The expanded reverse-chronological same-as-current rows also show that this degradation is order-sensitive: repeating the same current value after the answer target is less harmful than repeating it before the final slot answer position.

## Paper implication update

After the P8.1 rerun, the synthetic probe is no longer just a 16-example pilot. The most diagnostic cells now have 64 examples each and still support the same mechanism story, which makes the conflict-versus-repetition distinction safer to report in the main text.

## Paper implication

The paper can now say:

> Stale contamination is not a single phenomenon. Controlled same-slot probes show that conflicting stale values interact strongly with context order and version ambiguity, while repeated non-conflicting entries still degrade exact-answer behavior. Explicit latest/outdated labels largely remove conflict-driven collapse, but real retrieval contexts remain hard because the latest value is often not retrieved and distractors are less cleanly structured.

## Next steps

1. Add a small formatting-normalized metric for same_as_current conditions to distinguish true semantic errors from exact-match formatting failures.
2. Repeat the most diagnostic subset with Llama3.1-8B to test model-specific stale susceptibility.
3. Use the probe results to design a smaller paper figure: conflict vs same_as_current, chronological vs reverse, label vs no label.
