# P8.0 Mechanism Probe Setup Note

## Purpose

`docs/critical_review_v3.md` asks for controlled experiments that distinguish why stale same-slot context causes answer collapse. This note records the initial evaluation-harness changes that enable those probes.

## Implemented hooks

`scripts/eval_evomemory.py` now supports answer-time context controls:

```text
--context_order normal | chronological | reverse_chronological | current_first | current_last
--context_annotation none | timestamp | latest_outdated_label
```

The implementation reuses the existing answer path:

```text
answer_question(...)
filter_latest_per_slot(...)
retrieved_trace(...)
build_slot_answer_prompt(...)
```

Default behavior is unchanged:

```text
--context_order normal
--context_annotation none
```

## Intended mechanism probes

### Position bias

Compare the same retrieved entries under:

```text
current_first
current_last
chronological
reverse_chronological
```

If answer accuracy changes strongly with order, this supports a positional-bias mechanism.

### Semantic disambiguation

Compare:

```text
context_annotation=none
context_annotation=timestamp
context_annotation=latest_outdated_label
```

If labels recover accuracy, this supports a semantic/version-disambiguation mechanism.

### Interaction with latest_per_slot

The controls can be combined with:

```text
--retrieval_policy normal
--retrieval_policy latest_per_slot
```

This lets us test whether ordering/labels mainly help when stale entries remain in context, or whether they also affect slot-aware retrieval.

## Initial validation

Local validation:

```text
python -m py_compile scripts/eval_evomemory.py scripts/smoke_test.py
python scripts/smoke_test.py
```

Smoke result:

```text
SMOKE TEST: 26/26 passed
```

The smoke test now checks that `current_first` places the latest same-slot entry first and that `latest_outdated_label` inserts both `[latest]` and `[outdated]` labels.

A remote 3-example model-backed smoke has been launched on Sui-3:

```text
results/p80_mechanism_probes/smoke_current_first_label_k2_dev3/
```

This smoke should verify that result traces include:

```text
answer_trace.context_order
answer_trace.context_annotation
answer_trace.retrieved_entries
```

## Next experiment batch after smoke passes

Minimal P8.0 mechanism batch:

```text
raw_add k=16 P6.3 hard dev slot_prompt
normal order + no annotation
current_first + no annotation
current_last + no annotation
chronological + no annotation
reverse_chronological + no annotation
normal order + timestamp
normal order + latest_outdated_label
```

The main comparison should report EM/F1, answer value present, gold retrieved, stale retrieved, and representative traces.
