# P8.4 Latest API answer-model probe

## Motivation

Advisor feedback raised a model-recency concern: existing Qwen2.5, Llama3.1, and Mistral results may look dated to future readers. P8.4 tests whether the stale same-slot mechanism remains informative under current GPT/Gemini API answer models.

## Model availability

The API gateway exposes both GPT and Gemini families. We use only models that passed a minimal `OK` chat-completions probe. Models that returned empty content or account/model support errors are excluded from the experimental batch.

Included models with completed synthetic-dose outputs:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gemini-2.5-flash
gemini-2.5-pro
gemini-3-flash-preview
gemini-3.1-flash-lite-preview
```

Unavailable or excluded models:

```text
gemini-3-pro-preview        # minimal OK probe passed, but synthetic run returned empty content
gemini-3.1-pro-preview      # empty chat response
gpt-5.3-codex-spark        # account/model support error
gpt-5.3-codex              # account/model support error
gpt-5.2                    # account/model support error
```

## Interpretation rule

These API runs are answer-layer probes, not new memory managers. They should be reported as latest-model robustness checks for the version-arbitration mechanism: given a controlled context containing current and stale same-slot values, does the answer model choose the current value, copy a stale value, or use explicit latest/outdated metadata?

## Probe design

The synthetic dose-response probe varies:

- context order: chronological vs reverse chronological;
- metadata: no label vs explicit `[latest]` / `[outdated]` labels;
- stale count: 0, 1, 2, 4, 8, 16.

Each model is evaluated on the same prompt matrix, and outputs are summarized by exact match and stale-copy rate.

## Result summary

Generated summary table:

```text
results/p84_api_latest_model_probe_summary/api_latest_model_summary.md
```

Seven models produced completed synthetic-dose summaries. The three GPT models (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`) show a clean and consistent version-arbitration pattern: chronological/no-label contexts remain correct at stale=16 (EM 1.000), reverse/no-label contexts collapse from stale=1 onward (EM 0.000, stale copied 1.000), and reverse contexts with explicit `[latest]` / `[outdated]` labels recover to EM 1.000 at stale=16.

`gemini-3.1-flash-lite-preview` shows the same clean pattern as the GPT models. `gemini-2.5-flash` and `gemini-2.5-pro` show stale copying under reverse/no-label contexts, but also produce many empty or truncated outputs under this prompt format, so their low EM should be treated as an API/prompt-format caveat rather than direct evidence of worse memory reasoning. `gemini-3-flash-preview` also produced mostly empty outputs in the synthetic run and is not interpretable for the mechanism claim.

The strongest P8.4 takeaway is therefore based on the robust subset of completed, format-stable models: latest GPT models and `gemini-3.1-flash-lite-preview` reproduce the order- and metadata-sensitive version-arbitration failure found with earlier open models.

## Caveats

This probe is synthetic and answer-layer-only. It should not be framed as a full external-memory benchmark row. If all latest models show the same clean mechanism pattern, the result strengthens the paper's version-arbitration claim; if some models solve the synthetic matrix, the next step is a harder real-context trace rather than weakening the core benchmark evidence.
