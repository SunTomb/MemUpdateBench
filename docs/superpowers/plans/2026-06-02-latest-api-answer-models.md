# Latest API Answer Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a server-side latest GPT/Gemini answer-model replication for MemUpdateBench's stale same-slot mechanism story without storing API secrets.

**Architecture:** Keep this as an isolated API-answer-model probe rather than changing the main transformers evaluation path. Extend `scripts/probe_api_answer_model.py` with a reusable synthetic dose-response runner, launch approved API models from a server Bash script, summarize per-model results, then update paper/workflow notes with cautious interpretation.

**Tech Stack:** Python standard library (`urllib`, `argparse`, `csv`, `json`), existing MemUpdateBench smoke-test pattern, Bash/tmux on `/NAS/yesh/MemUpdateBench`, OpenAI-compatible `/v1/chat/completions` API.

---

## File Structure

- Modify `scripts/probe_api_answer_model.py`: add reusable synthetic stale-conflict API runner and output writer.
- Modify `scripts/smoke_test.py`: add helper-only tests; no external API call in smoke tests.
- Create `scripts/run_p84_api_latest_models_tang2.sh`: server batch runner for approved GPT/Gemini models; reads API config from environment only.
- Create `scripts/summarize_api_latest_model_probe.py`: aggregate model summaries into JSON/CSV/Markdown.
- Create `paper/p84_latest_api_model_probe_note.md`: paper-facing interpretation and caveats.
- Modify `WORKFLOW.md`: append P8.4 commands, results, conclusions after runs complete.

Approved model list:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gemini-2.5-flash
gemini-2.5-pro
gemini-3-flash-preview
gemini-3-pro-preview
gemini-3.1-flash-lite-preview
```

Excluded model list:

```text
gemini-3.1-pro-preview  # empty chat response
gpt-5.3-codex-spark     # account/model support error
gpt-5.3-codex           # account/model support error
gpt-5.2                 # account/model support error
```

Secrets rule: no API key or base URL may be written to repository files, result artifacts, workflow docs, or paper notes. Use `MUB_API_BASE_URL`, `MUB_API_KEY`, and `MUB_API_MODEL` only.

---

### Task 1: Add synthetic stale-conflict example builder

**Files:**
- Modify: `scripts/probe_api_answer_model.py`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Write failing smoke test**

Add this inside `test_api_probe_helpers` in `scripts/smoke_test.py`:

```python
        from scripts.probe_api_answer_model import build_synthetic_dose_examples

        synthetic = build_synthetic_dose_examples(stale_counts=[0, 1, 4], examples_per_condition=2)
        assert synthetic
        conditions = {example["condition"] for example in synthetic}
        assert "chronological_none" in conditions
        assert "reverse_chronological_none" in conditions
        assert "reverse_chronological_latest_outdated_label" in conditions
        reverse = [example for example in synthetic if example["condition"] == "reverse_chronological_none"]
        assert {example["stale_count"] for example in reverse} == {0, 1, 4}
        assert all(example["gold"] in example["prompt"] for example in synthetic)
        assert all("api_key" not in example for example in synthetic)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected: `API probe helpers` fails because `build_synthetic_dose_examples` cannot be imported.

- [ ] **Step 3: Implement builder**

Add to `scripts/probe_api_answer_model.py`:

```python
CITY_POOL = [
    "Shanghai", "Beijing", "Wuhan", "Nanjing", "Hangzhou", "Suzhou", "Shenzhen", "Guangzhou",
    "Xi'an", "Chongqing", "Tianjin", "Qingdao", "Dalian", "Xiamen", "Fuzhou", "Ningbo",
]
GOLD_POOL = ["Chengdu", "Kunming", "Harbin", "Lanzhou", "Hefei", "Changsha", "Urumqi", "Haikou"]


def format_update_line(value: str, label: str = "") -> str:
    prefix = f"{label} " if label else ""
    return f"- {prefix}User says: Alex's location is {value}."


def build_synthetic_prompt(context_lines: list[str]) -> str:
    return (
        "Question: Where does Alex currently live?\n"
        "Target entity: friend_Alex\n"
        "Target attribute: location\n"
        "Answer with only the current location value.\n\n"
        "Memory context:\n"
        + "\n".join(context_lines)
    )


def build_synthetic_dose_examples(stale_counts: list[int], examples_per_condition: int) -> list[dict[str, Any]]:
    examples = []
    for stale_count in stale_counts:
        for replicate in range(examples_per_condition):
            gold = GOLD_POOL[replicate % len(GOLD_POOL)]
            stale_values = [city for city in CITY_POOL if city != gold][:stale_count]
            condition_lines = {
                "chronological_none": [format_update_line(value) for value in stale_values] + [format_update_line(gold)],
                "reverse_chronological_none": [format_update_line(gold)] + [format_update_line(value) for value in stale_values],
                "reverse_chronological_latest_outdated_label": [format_update_line(gold, "[latest]")]
                + [format_update_line(value, "[outdated]") for value in stale_values],
            }
            for condition, lines in condition_lines.items():
                examples.append(
                    {
                        "example_id": f"{condition}_stale{stale_count}_rep{replicate}",
                        "condition": condition,
                        "stale_count": stale_count,
                        "gold": gold,
                        "stale_values": stale_values,
                        "prompt": build_synthetic_prompt(lines),
                    }
                )
    return examples
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected: all smoke tests pass.

---

### Task 2: Add synthetic-dose API runner mode

**Files:**
- Modify: `scripts/probe_api_answer_model.py`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Write failing summarizer-helper smoke test**

Add inside `test_api_probe_helpers`:

```python
        from scripts.probe_api_answer_model import summarize_rows

        rows = [
            {"condition": "reverse_chronological_none", "stale_count": 1, "em": 0.0, "stale_copied": 1.0},
            {"condition": "reverse_chronological_none", "stale_count": 1, "em": 1.0, "stale_copied": 0.0},
            {"condition": "chronological_none", "stale_count": 1, "em": 1.0, "stale_copied": 0.0},
        ]
        summary = summarize_rows(rows)
        assert summary["reverse_chronological_none"]["1"]["n"] == 2
        assert summary["reverse_chronological_none"]["1"]["em"] == 0.5
        assert summary["reverse_chronological_none"]["1"]["stale_copied"] == 0.5
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected: `summarize_rows` import failure.

- [ ] **Step 3: Implement summarizer and runner**

Add to `scripts/probe_api_answer_model.py`:

```python
def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        condition = str(row["condition"])
        stale_count = str(row["stale_count"])
        grouped.setdefault(condition, {}).setdefault(stale_count, []).append(row)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for condition, by_stale in grouped.items():
        summary[condition] = {}
        for stale_count, subset in by_stale.items():
            n = len(subset)
            summary[condition][stale_count] = {
                "n": n,
                "em": sum(float(row["em"]) for row in subset) / n,
                "stale_copied": sum(float(row["stale_copied"]) for row in subset) / n,
            }
    return summary


def run_synthetic_dose_probe(
    base_url: str,
    api_key: str,
    model: str,
    output_dir: Path,
    stale_counts: list[int],
    examples_per_condition: int,
    timeout: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in build_synthetic_dose_examples(stale_counts, examples_per_condition):
        started = time.time()
        raw = chat_completion(base_url, api_key, model, example["prompt"], timeout=timeout)
        prediction = exact_value_prediction(raw)
        stale_copied = prediction in set(example["stale_values"])
        rows.append(
            {
                "example_id": example["example_id"],
                "condition": example["condition"],
                "stale_count": example["stale_count"],
                "model": model,
                "prompt_sha256": hashlib.sha256(example["prompt"].encode("utf-8")).hexdigest()[:16],
                "gold": example["gold"],
                "prediction": prediction,
                "raw_response": raw.strip(),
                "em": 1.0 if prediction == example["gold"] else 0.0,
                "stale_copied": 1.0 if stale_copied else 0.0,
                "latency_seconds": round(time.time() - started, 3),
            }
        )
    csv_path = output_dir / "api_synthetic_dose_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": model, "num_examples": len(rows), "by_condition_and_stale_count": summarize_rows(rows)}
    (output_dir / "api_synthetic_dose_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
```

- [ ] **Step 4: Add CLI action**

In `main()` add:

```python
    parser.add_argument("--synthetic-dose-probe", action="store_true")
    parser.add_argument("--stale-counts", default="0,1,2,4,8,16")
    parser.add_argument("--examples-per-condition", type=int, default=16)
```

Before final no-action check add:

```python
    if args.synthetic_dose_probe:
        stale_counts = [int(item) for item in args.stale_counts.split(",") if item]
        model_dir = Path(args.output_dir) / model.replace("/", "_")
        summary = run_synthetic_dose_probe(
            base_url,
            api_key,
            model,
            model_dir,
            stale_counts=stale_counts,
            examples_per_condition=args.examples_per_condition,
            timeout=args.timeout,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
```

Update no-action check to include `args.synthetic_dose_probe`.

- [ ] **Step 5: Verify tests and compile**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m py_compile scripts/probe_api_answer_model.py scripts/smoke_test.py
```

Expected: all pass.

---

### Task 3: Add server batch runner

**Files:**
- Create: `scripts/run_p84_api_latest_models_tang2.sh`

- [ ] **Step 1: Create runner**

Create exact file:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/MemUpdateBench
source activate.sh >/dev/null

if [[ -z "${MUB_API_BASE_URL:-}" ]]; then
  echo "MUB_API_BASE_URL is required" >&2
  exit 1
fi
if [[ -z "${MUB_API_KEY:-}" ]]; then
  echo "MUB_API_KEY is required" >&2
  exit 1
fi

MODELS=(
  gpt-5.5
  gpt-5.4
  gpt-5.4-mini
  gemini-2.5-flash
  gemini-2.5-pro
  gemini-3-flash-preview
  gemini-3-pro-preview
  gemini-3.1-flash-lite-preview
)

mkdir -p logs results/p84_api_latest_model_probe

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  echo "=== Running ${model} ==="
  MUB_API_MODEL="${model}" PYTHONPATH=. python scripts/probe_api_answer_model.py \
    --connectivity \
    --synthetic-dose-probe \
    --stale-counts 0,1,2,4,8,16 \
    --examples-per-condition 16 \
    --output-dir results/p84_api_latest_model_probe \
    --timeout 120 \
    > "logs/p84_api_${safe_model}.log" 2>&1
  echo "=== Completed ${model} ==="
done
```

- [ ] **Step 2: Verify shell syntax**

Run:

```bash
bash -n scripts/run_p84_api_latest_models_tang2.sh
```

Expected: no output.

- [ ] **Step 3: Verify no secrets in scripts**

Run:

```bash
python - <<'PY'
import os
from pathlib import Path
real_key = os.environ.get('MUB_API_KEY', '')
real_base = os.environ.get('MUB_API_BASE_URL', '')
for path in [Path('scripts/run_p84_api_latest_models_tang2.sh'), Path('scripts/probe_api_answer_model.py')]:
    text = path.read_text(encoding='utf-8')
    if real_key:
        assert real_key not in text
    if real_base:
        assert real_base not in text
print('secret scan passed')
PY
```

Expected: `secret scan passed`.

---

### Task 4: Run one-model pilot on the server

**Files:**
- Remote result: `/NAS/yesh/MemUpdateBench/results/p84_api_latest_model_probe/gpt-5.5/`
- Remote log: `/NAS/yesh/MemUpdateBench/logs/p84_api_gpt-5.5_pilot.log`

- [ ] **Step 1: Sync scripts to server checkout**

Copy only changed scripts to `/NAS/yesh/MemUpdateBench/scripts/` if the server checkout does not already contain them. Do not copy untracked PPT/LaTeX files.

- [ ] **Step 2: Run pilot command in remote shell**

Run via SSH with environment variables supplied at invocation time, not written into files:

```bash
ssh Tang-2-Wu "cd /NAS/yesh/MemUpdateBench && source activate.sh >/dev/null && MUB_API_BASE_URL=\"$MUB_API_BASE_URL\" MUB_API_KEY=\"$MUB_API_KEY\" MUB_API_MODEL=gpt-5.5 PYTHONPATH=. python scripts/probe_api_answer_model.py --connectivity --synthetic-dose-probe --stale-counts 0,1,2,4,8,16 --examples-per-condition 4 --output-dir results/p84_api_latest_model_probe --timeout 120 > logs/p84_api_gpt-5.5_pilot.log 2>&1"
```

Expected: command exits 0 and creates `results/p84_api_latest_model_probe/gpt-5.5/api_synthetic_dose_summary.json`.

- [ ] **Step 3: Inspect pilot summary**

Run:

```bash
ssh Tang-2-Wu "cd /NAS/yesh/MemUpdateBench && python - <<'PY'
import json
p='results/p84_api_latest_model_probe/gpt-5.5/api_synthetic_dose_summary.json'
print(json.dumps(json.load(open(p, encoding='utf-8'))['by_condition_and_stale_count'], indent=2, ensure_ascii=False))
PY"
```

Expected: readable condition/stale-count table. Decide whether the prompt is too easy before full batch.

---

### Task 5: Run full approved-model batch

**Files:**
- Remote outputs under `/NAS/yesh/MemUpdateBench/results/p84_api_latest_model_probe/`
- Remote logs under `/NAS/yesh/MemUpdateBench/logs/`

- [ ] **Step 1: Start tmux batch**

Run:

```bash
ssh Tang-2-Wu "cd /NAS/yesh/MemUpdateBench && tmux new-session -d -s p84_api_latest_models 'export MUB_API_BASE_URL=\"$MUB_API_BASE_URL\"; export MUB_API_KEY=\"$MUB_API_KEY\"; bash scripts/run_p84_api_latest_models_tang2.sh'"
```

Expected: tmux session starts.

- [ ] **Step 2: Monitor completion**

Run periodically:

```bash
ssh Tang-2-Wu "tmux has-session -t p84_api_latest_models && echo RUNNING || echo DONE"
```

Expected: eventually prints `DONE`.

- [ ] **Step 3: Check all model outputs**

Run:

```bash
ssh Tang-2-Wu "cd /NAS/yesh/MemUpdateBench && for m in gpt-5.5 gpt-5.4 gpt-5.4-mini gemini-2.5-flash gemini-2.5-pro gemini-3-flash-preview gemini-3-pro-preview gemini-3.1-flash-lite-preview; do test -f results/p84_api_latest_model_probe/\$m/api_synthetic_dose_summary.json && echo OK \$m || echo MISSING \$m; done"
```

Expected: `OK` for all approved models.

---

### Task 6: Add result summarizer

**Files:**
- Create: `scripts/summarize_api_latest_model_probe.py`
- Modify: `scripts/smoke_test.py`

- [ ] **Step 1: Add import smoke test**

Add `"scripts.summarize_api_latest_model_probe"` to the `modules` list in `test_imports`.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Expected: import failure for `scripts.summarize_api_latest_model_probe`.

- [ ] **Step 3: Create summarizer script**

Create `scripts/summarize_api_latest_model_probe.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.glob("*/api_synthetic_dose_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary["model"]
        for condition, by_stale in summary["by_condition_and_stale_count"].items():
            for stale_count, stats in by_stale.items():
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "stale_count": int(stale_count),
                        "n": int(stats["n"]),
                        "em": float(stats["em"]),
                        "stale_copied": float(stats["stale_copied"]),
                    }
                )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "api_latest_model_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "api_latest_model_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "stale_count", "n", "em", "stale_copied"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# P8.4 API latest-model summary", "", "| Model | Condition | Stale count | n | EM | Stale copied |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['condition']} | {row['stale_count']} | {row['n']} | {row['em']:.3f} | {row['stale_copied']:.3f} |")
    (output_dir / "api_latest_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize P8.4 API latest-model stale-conflict probes")
    parser.add_argument("--result-root", default="results/p84_api_latest_model_probe")
    parser.add_argument("--output-dir", default="results/p84_api_latest_model_probe_summary")
    args = parser.parse_args()
    rows = load_rows(Path(args.result_root))
    write_outputs(rows, Path(args.output_dir))
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m py_compile scripts/summarize_api_latest_model_probe.py
```

Expected: all pass.

- [ ] **Step 5: Run summarizer after remote batch**

Run on server:

```bash
ssh Tang-2-Wu "cd /NAS/yesh/MemUpdateBench && source activate.sh >/dev/null && PYTHONPATH=. python scripts/summarize_api_latest_model_probe.py --result-root results/p84_api_latest_model_probe --output-dir results/p84_api_latest_model_probe_summary"
```

Expected: summary JSON/CSV/MD generated.

---

### Task 7: Write paper and workflow notes

**Files:**
- Create: `paper/p84_latest_api_model_probe_note.md`
- Modify: `WORKFLOW.md`

- [ ] **Step 1: Create paper-facing note**

Write `paper/p84_latest_api_model_probe_note.md` with these sections:

```markdown
# P8.4 Latest API answer-model probe

## Motivation

Advisor feedback raised a model-recency concern: existing Qwen2.5/Llama3.1/Mistral results may look dated to future readers. P8.4 tests whether the stale same-slot mechanism remains informative under current GPT/Gemini API answer models.

## Model availability

Use only models that passed a minimal `OK` chat-completions probe. Exclude models that returned empty content or account/model support errors.

## Interpretation rule

These API runs are answer-layer probes, not new memory managers. They should be reported as latest-model robustness checks for the version-arbitration mechanism.

## Result summary

Summarize the generated `results/p84_api_latest_model_probe_summary/api_latest_model_summary.md` table after the batch completes.

## Caveats

Do not overclaim from synthetic prompts alone. If latest models solve the easy synthetic matrix, harden the prompt or run a sampled real-context trace before using the result as paper evidence.
```

- [ ] **Step 2: Append `WORKFLOW.md` entry after results complete**

Append a P8.4 section with:

```markdown
## P8.4 latest API answer-model probe

### Motivation

Latest-model replication addresses feedback that older answer models may make the stale-conflict result look model-era specific.

### Commands run

List the server pilot, full batch, summarizer, smoke test, and compile commands.

### Results

Report per-model EM/stale-copied trends by condition and stale count.

### Conclusion

State whether latest GPT/Gemini models still show order/metadata-sensitive version arbitration, or whether they solve the current synthetic matrix and require a harder real-context trace.
```

- [ ] **Step 3: Verify no secrets and validation status**

Run locally:

```bash
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python -m py_compile scripts/probe_api_answer_model.py scripts/summarize_api_latest_model_probe.py scripts/smoke_test.py
python - <<'PY'
import os
from pathlib import Path
real_key = os.environ.get('MUB_API_KEY', '')
real_base = os.environ.get('MUB_API_BASE_URL', '')
for path in Path('.').rglob('*'):
    if path.is_file() and any(part in {'.git', '__pycache__'} for part in path.parts):
        continue
    if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.json', '.csv'}:
        text = path.read_text(encoding='utf-8', errors='ignore')
        if real_key:
            assert real_key not in text
        if real_base:
            assert real_base not in text
print('secret scan passed')
PY
```

Expected: smoke tests pass, compile passes, secret scan passes.

---

## Execution Recommendation

Run Tasks 1-4 first, then inspect the `gpt-5.5` pilot. If `gpt-5.5` stays perfect across all stale counts and conditions, pause before Task 5 and harden the prompt matrix instead of spending API budget on all models. If the pilot shows a useful order/metadata pattern, run Tasks 5-7.

## Self-Review

- Spec coverage: covers API runner, server execution, summarization, paper note, workflow note, and secret handling.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: functions use `dict[str, Any]`, `Path`, and existing helper names consistently.
- Scope check: one focused subsystem: latest API answer-model probing. Full integration into `eval_evomemory.py` is intentionally excluded until the pilot proves useful.
