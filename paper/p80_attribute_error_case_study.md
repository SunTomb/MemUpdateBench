# P8.0 Attribute Error Case Study

## Purpose

`docs/critical_review_v3.md` points out that expanded-split attribute differences may reveal why answer-layer failures remain even when memory state is correct. This note records the first-pass attribute-level analysis over existing expanded split slot-prompt results.

## Artifacts

```text
scripts/analyze_attribute_failures.py
results/p80_attribute_error_analysis/attribute_summary.csv
results/p80_attribute_error_analysis/attribute_error_cases.csv
results/p80_attribute_error_analysis/attribute_failure_analysis.md
results/p80_attribute_error_analysis/company_language_error_type_summary.csv
results/p80_attribute_error_analysis/gold_retrieved_wrong_cases.{csv,md}
```

Inputs:

```text
results/p69_expanded_slot_prompt_allk/constrained_slot_crud_prompt_k16_dev_merged/evomemory_results.json
results/p69_expanded_slot_prompt_allk/raw_add_prompt_k16_dev_merged/evomemory_results.json
results/p69_expanded_slot_prompt_allk/constrained_slot_crud_prompt_k8_dev_merged/evomemory_results.json
results/p69_expanded_slot_prompt_allk/raw_add_prompt_k8_dev_merged/evomemory_results.json
```

Total examples analyzed: 800.

## k=16 per-attribute summary

| Method | Attribute | N | EM | F1 | Gold retrieved | State acc. | Retrieved stale avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constrained CRUD | company | 25 | 0.280 | 0.280 | 0.280 | 1.000 | 0.000 |
| Constrained CRUD | hobby | 25 | 0.680 | 0.780 | 1.000 | 1.000 | 0.000 |
| Constrained CRUD | instrument | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Constrained CRUD | language | 25 | 0.120 | 0.120 | 0.600 | 1.000 | 0.000 |
| Constrained CRUD | location | 25 | 0.960 | 0.960 | 1.000 | 1.000 | 0.000 |
| Constrained CRUD | preference | 25 | 0.960 | 0.960 | 0.960 | 1.000 | 0.000 |
| Constrained CRUD | project | 25 | 0.600 | 0.600 | 1.000 | 1.000 | 0.000 |
| Constrained CRUD | timezone | 25 | 0.800 | 0.800 | 1.000 | 1.000 | 0.000 |
| Raw append | company | 25 | 0.120 | 0.120 | 0.280 | 1.000 | 3.160 |
| Raw append | hobby | 25 | 0.080 | 0.080 | 0.440 | 1.000 | 4.280 |
| Raw append | instrument | 25 | 0.080 | 0.080 | 0.600 | 1.000 | 4.200 |
| Raw append | language | 25 | 0.160 | 0.160 | 0.160 | 1.000 | 2.920 |
| Raw append | location | 25 | 0.080 | 0.080 | 0.240 | 1.000 | 4.760 |
| Raw append | preference | 25 | 0.000 | 0.187 | 0.080 | 1.000 | 4.920 |
| Raw append | project | 25 | 0.280 | 0.280 | 0.480 | 1.000 | 4.320 |
| Raw append | timezone | 25 | 0.320 | 0.320 | 0.480 | 1.000 | 4.440 |

## Main finding

The first-pass analysis partially revises the initial v3-review hypothesis. The low `company` and `language` EM under Constrained CRUD is not purely a gold-retrieved-wrong-answer phenomenon:

- `company`: EM 0.28, gold retrieved 0.28, state acc. 1.00.
- `language`: EM 0.12, gold retrieved 0.60, state acc. 1.00.

Thus, a large part of the residual gap for these attributes comes from answer-time retrieval/context selection even when the memory state is clean. The answer layer often sees semantically tempting distractors such as interview/no-change sentences for company or language exposure statements, and then returns a distractor value.

For attributes where gold retrieval is consistently 1.00, residual failures still exist:

- `hobby`: EM 0.68 despite gold retrieved 1.00.
- `project`: EM 0.60 despite gold retrieved 1.00.
- `timezone`: EM 0.80 despite gold retrieved 1.00.

These cases are closer to true gold-retrieved-wrong-answer failures.

## Company/language gold-retrieved wrong-answer deep dive

The expanded analyzer now isolates failures where the gold value is present in the retrieved context but the model still answers incorrectly. For `company` and `language`, the k=16 error-type summary is:

| Method | Attribute | Error type | N |
| --- | --- | --- | ---: |
| Constrained CRUD | company | correct | 7 |
| Constrained CRUD | company | gold_not_retrieved | 18 |
| Constrained CRUD | language | correct | 3 |
| Constrained CRUD | language | gold_not_retrieved | 10 |
| Constrained CRUD | language | gold_retrieved_answer_layer_failure | 12 |
| Raw append | company | correct | 3 |
| Raw append | company | gold_not_retrieved | 18 |
| Raw append | company | gold_retrieved_stale_competition | 4 |
| Raw append | language | correct | 4 |
| Raw append | language | gold_not_retrieved | 21 |

This clarifies the advisor's question. For `company`, the dominant k=16 failure under clean Constrained CRUD is still retrieval/context selection: 18/25 examples do not retrieve the gold company at all, even though state accuracy is 1.00. For `language`, the failure is more truly answer-layer: 12/25 Constrained CRUD k=16 examples retrieve the gold language but still answer incorrectly.

The gold-retrieved language cases show a consistent pattern: the model copies a semantically salient near-miss distractor from retrieved context, often repeated workshop/discussion statements about another language. Example: when the gold is `Python`, the retrieved context contains `switched to Python` plus repeated `discussed TypeScript in a workshop` entries, and the model predicts `TypeScript`. These are not stale same-slot entries in the memory state; they are same-entity same-attribute-looking non-update distractors that answer-time retrieval makes salient.

Raw append adds a different failure mode. For `company`, the few gold-retrieved wrong cases are classified as stale competition: the gold company is retrieved, but stale same-slot alternatives are retrieved too. For `language`, raw append k=16 mostly fails before this stage because gold retrieval itself drops to 0.16.

## Representative patterns

The extracted cases show several recurring patterns:

1. **Negated or non-update distractors outrank final updates.**
   - Example: company contexts include “interviewed with X but has not changed jobs” entries, and the model often predicts X.
2. **Same-entity same-attribute distractors remain dangerous even with clean memory.**
   - Hobby/project examples often include multiple same-entity mentions about related activities.
3. **Gold present is not sufficient when the prompt asks for current slot value but the context contains plausible near-miss facts.**
   - The model sometimes selects a distractor that is semantically salient but not the final state.
4. **Raw append compounds retrieval failure with stale exposure.**
   - Raw append k=16 has low gold retrieval and 3-5 retrieved stale same-slot entries on average across attributes.

## Paper implication

The paper should separate two answer-layer failure modes:

1. **Clean-state retrieval/context failure:** the memory store contains the right final slot value, but top-k context fails to surface it or surfaces strong near-miss distractors.
2. **Gold-retrieved answer failure:** the gold value is present in context, but the answer model still chooses a stale/distractor value.

This helps explain why `latest_per_slot` recovers raw_add from EM 0.14 to 0.69 but does not reach the gold-context upper bound of 0.92: stale suppression helps, but retrieval/context composition and attribute-specific distractor semantics remain independent failure sources.

## Paper implication

The paper should separate two answer-layer failure modes:

1. **Clean-state retrieval/context failure:** the memory store contains the right final slot value, but top-k context fails to surface it or surfaces strong near-miss distractors.
2. **Gold-retrieved answer failure:** the gold value is present in context, but the answer model still chooses a stale/distractor value.

This helps explain why `latest_per_slot` recovers raw_add from EM 0.14 to 0.69 but does not reach the gold-context upper bound of 0.92: stale suppression helps, but retrieval/context composition and attribute-specific distractor semantics remain independent failure sources.

The refined company/language result should be stated carefully:

- `company` is primarily a retrieval/context-selection problem at k=16, not a gold-retrieved answer-layer problem.
- `language` is a stronger gold-retrieved answer-layer problem: repeated near-miss language mentions can override a retrieved final update even in clean Constrained CRUD memory.
