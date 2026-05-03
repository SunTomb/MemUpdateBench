# Optional Mem0 Isolated Feasibility Plan

## Current status

Default decision: no-go for implementation in the current P6.7 manuscript block.

This plan exists only so that, if advisor/reviewer pressure requires an external memory-framework row, the work can be scoped without polluting the main MemUpdateBench environment or reviving old G-MSRA agent infrastructure.

## Go criteria

Proceed only if all conditions are true:

1. The manuscript still needs ecosystem grounding after the current P6.3 tradeoff figure and k=16 table are integrated.
2. Mem0 can be tested in an isolated environment without modifying the shared `gmsra` environment.
3. Mem0 exposes memory entries or enough state to measure stale same-slot burden.
4. A k=16-only feasibility row is sufficient for positioning.

## No-go criteria

Abort if any condition is true:

1. Mem0 requires heavy server or agent infrastructure for a minimal test.
2. Memory state is opaque, making stale same-slot burden unmeasurable.
3. Integration requires changing MemUpdateBench task semantics.
4. The setup would delay paper integration or reproducibility work.
5. It requires installing packages into the main cluster environment.

## Isolation boundary

Suggested isolated path if needed:

```text
/NAS/yesh/MemUpdateBench/.venv_mem0_p63
```

Do not install into:

```text
gmsra
```

Do not modify files outside:

```text
/NAS/yesh/MemUpdateBench
```

## Minimal target

Run only:

```text
data/evomemory_update_frequency_hard_k16_p63_test.json
```

Use deterministic oracle/constrained CRUD as the sanity anchor before interpreting any Mem0 result.

## Required output schema

A feasibility run must report:

```json
{
  "system": "mem0",
  "split": "evomemory_update_frequency_hard_k16_p63_test",
  "num_examples": 0,
  "state_accuracy": null,
  "slot_prompt_em": null,
  "slot_prompt_f1": null,
  "stale_same_slot_entry_count_avg": null,
  "final_memory_size_avg": null,
  "inspectable_memory": false,
  "notes": ""
}
```

If `inspectable_memory` is false, the run is not comparable for the main benchmark table and should be described only qualitatively.

## Minimal feasibility questions

1. Can Mem0 represent an exact `(entity, attribute)` update, or does it append natural-language memories?
2. Can obsolete same-slot values be removed or suppressed?
3. Can the memory bank be inspected after each episode?
4. Can stale same-slot burden be computed reliably?
5. Can the same slot-conditioned answer prompt be applied without changing the benchmark semantics?

## Commands, if approved later

Environment creation should be explicit and isolated. Do not run these unless the go criteria are met.

```bash
cd /NAS/yesh/MemUpdateBench
python -m venv .venv_mem0_p63
source .venv_mem0_p63/bin/activate
python -m pip install --upgrade pip
python -m pip install mem0ai
```

Then create a minimal feasibility script under a clearly isolated path, such as:

```text
paper/feasibility/mem0_k16_probe.py
```

The script should read the existing k=16 test file and write a small JSON report under:

```text
paper/feasibility/mem0_k16_report.json
```

## Recommendation

Do not execute this plan unless the paper draft cannot be positioned without an external baseline. The current controlled P6.3 results and P6.7 manuscript package already support the core diagnostic claim.
