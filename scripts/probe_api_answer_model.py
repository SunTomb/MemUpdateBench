from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "p84_api_latest_model_probe"
VALUE_RE = re.compile(r"\b[A-Z][A-Za-z0-9+_.-]*\b")
CITY_POOL = [
    "Shanghai", "Beijing", "Wuhan", "Nanjing", "Hangzhou", "Suzhou", "Shenzhen", "Guangzhou",
    "Xi'an", "Chongqing", "Tianjin", "Qingdao", "Dalian", "Xiamen", "Fuzhou", "Ningbo",
]
GOLD_POOL = ["Chengdu", "Kunming", "Harbin", "Lanzhou", "Hefei", "Changsha", "Urumqi", "Haikou"]


def redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-4:]}"


def build_headers(api_key: str, include_json: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Claude-Code MemUpdateBench API Probe/1.0",
        "Accept": "application/json",
    }
    if include_json:
        headers["Content-Type"] = "application/json"
    return headers


def build_chat_payload(model: str, prompt: str, max_tokens: int = 32, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer with the exact requested value only. Do not add explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def exact_value_prediction(text: str) -> str:
    allowed_values = set(CITY_POOL) | set(GOLD_POOL) | {"OK"}
    stripped = text.strip().strip('"\'`。.!?，,;；:：')
    if stripped in allowed_values:
        return stripped
    for value in sorted(allowed_values, key=len, reverse=True):
        if re.search(rf"\b{re.escape(value)}\b", text):
            return value
    for match in VALUE_RE.findall(text):
        if match in allowed_values:
            return match
    return stripped.splitlines()[0].strip() if stripped else ""


def build_probe_examples() -> list[dict[str, Any]]:
    base_instruction = (
        "Question: Where does Alex currently live?\n"
        "Target entity: friend_Alex\n"
        "Target attribute: location\n"
        "Answer with only the current location value.\n\n"
        "Memory context:\n"
    )
    return [
        {
            "example_id": "final_only_0",
            "condition": "final_only",
            "gold": "Chengdu",
            "stale_values": [],
            "prompt": base_instruction + "- User says: Alex currently lives in Chengdu.",
        },
        {
            "example_id": "generic_unrelated_distractor_0",
            "condition": "generic_unrelated_distractor",
            "gold": "Chengdu",
            "stale_values": [],
            "prompt": base_instruction
            + "- User says: Bob likes coffee.\n"
            + "- User says: Alex currently lives in Chengdu.\n"
            + "- User says: Nora works at Tencent.",
        },
        {
            "example_id": "stale_reverse_none_0",
            "condition": "stale_same_slot_reverse_no_label",
            "gold": "Chengdu",
            "stale_values": ["Beijing", "Shanghai"],
            "prompt": base_instruction
            + "- User says: Alex currently lives in Chengdu.\n"
            + "- User says: Alex previously lived in Beijing.\n"
            + "- User says: Alex previously lived in Shanghai.",
        },
        {
            "example_id": "stale_reverse_label_0",
            "condition": "stale_same_slot_reverse_with_label",
            "gold": "Chengdu",
            "stale_values": ["Beijing", "Shanghai"],
            "prompt": base_instruction
            + "- [latest] User says: Alex currently lives in Chengdu.\n"
            + "- [outdated] User says: Alex previously lived in Beijing.\n"
            + "- [outdated] User says: Alex previously lived in Shanghai.",
        },
        {
            "example_id": "stale_reverse_none_1",
            "condition": "stale_same_slot_reverse_no_label",
            "gold": "Chengdu",
            "stale_values": ["Wuhan", "Nanjing", "Beijing", "Shanghai"],
            "prompt": base_instruction
            + "- User says: Alex currently lives in Chengdu.\n"
            + "- User says: Alex previously lived in Wuhan.\n"
            + "- User says: Alex previously lived in Nanjing.\n"
            + "- User says: Alex previously lived in Beijing.\n"
            + "- User says: Alex previously lived in Shanghai.",
        },
        {
            "example_id": "stale_reverse_label_1",
            "condition": "stale_same_slot_reverse_with_label",
            "gold": "Chengdu",
            "stale_values": ["Wuhan", "Nanjing", "Beijing", "Shanghai"],
            "prompt": base_instruction
            + "- [latest] User says: Alex currently lives in Chengdu.\n"
            + "- [outdated] User says: Alex previously lived in Wuhan.\n"
            + "- [outdated] User says: Alex previously lived in Nanjing.\n"
            + "- [outdated] User says: Alex previously lived in Beijing.\n"
            + "- [outdated] User says: Alex previously lived in Shanghai.",
        },
    ]


def parse_sse_chat_body(body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    chunks = []
    usage = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        parsed = json.loads(payload)
        choice = parsed.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        if isinstance(content, str):
            chunks.append(content)
        if "usage" in parsed:
            usage = parsed["usage"]
    result: dict[str, Any] = {"choices": [{"message": {"content": "".join(chunks)}}]}
    if usage is not None:
        result["usage"] = usage
    return result


def parse_json_body(body: bytes, path: str) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    if text.lstrip().startswith("data:"):
        return parse_sse_chat_body(body)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {path}: {text[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON response from {path}: {text[:500]}")
    return parsed


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


def api_request(base_url: str, api_key: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = build_headers(api_key, include_json=payload is not None)
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return parse_json_body(response.read(), path)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body[:500]}") from exc


def chat_completion(base_url: str, api_key: str, model: str, prompt: str, timeout: int = 60) -> str:
    payload = build_chat_payload(model, prompt)
    response = api_request(base_url, api_key, "/chat/completions", payload=payload, timeout=timeout)
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response schema: {json.dumps(response, ensure_ascii=False)[:500]}") from exc


def should_retry_api_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        isinstance(exc, TimeoutError)
        or "timed out" in message.lower()
        or "Non-JSON response from /chat/completions" in message
        or "HTTP 429 from /chat/completions" in message
        or "HTTP 500 from /chat/completions" in message
        or "HTTP 502 from /chat/completions" in message
        or "HTTP 503 from /chat/completions" in message
        or "HTTP 504 from /chat/completions" in message
    )


def chat_completion_with_retries(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return chat_completion(base_url, api_key, model, prompt, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not should_retry_api_error(exc):
                raise
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def list_models(base_url: str, api_key: str, timeout: int = 60) -> list[str]:
    response = api_request(base_url, api_key, "/models", timeout=timeout)
    models = []
    for item in response.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    return models


def run_connectivity(base_url: str, api_key: str, model: str, timeout: int = 60) -> dict[str, Any]:
    started = time.time()
    raw = chat_completion_with_retries(base_url, api_key, model, "Please answer with exactly: OK", timeout=timeout)
    parsed = exact_value_prediction(raw)
    return {
        "model": model,
        "raw_response": raw,
        "parsed_response": parsed,
        "ok": parsed == "OK",
        "latency_seconds": round(time.time() - started, 3),
    }


def run_probe(base_url: str, api_key: str, model: str, output_dir: Path, timeout: int = 60) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in build_probe_examples():
        started = time.time()
        raw = chat_completion_with_retries(base_url, api_key, model, example["prompt"], timeout=timeout)
        prediction = exact_value_prediction(raw)
        stale_copied = prediction in set(example["stale_values"])
        row = {
            "example_id": example["example_id"],
            "condition": example["condition"],
            "model": model,
            "prompt_sha256": hashlib.sha256(example["prompt"].encode("utf-8")).hexdigest()[:16],
            "gold": example["gold"],
            "prediction": prediction,
            "raw_response": raw.strip(),
            "em": 1.0 if prediction == example["gold"] else 0.0,
            "stale_copied": 1.0 if stale_copied else 0.0,
            "latency_seconds": round(time.time() - started, 3),
        }
        rows.append(row)

    csv_path = output_dir / "api_probe_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "n": len(subset),
            "em": sum(row["em"] for row in subset) / len(subset),
            "stale_copied": sum(row["stale_copied"] for row in subset) / len(subset),
        }
    summary = {"model": model, "num_examples": len(rows), "by_condition": by_condition}
    (output_dir / "api_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# API latest-model probe", "", f"Model: `{model}`", "", "| Condition | n | EM | Stale copied |", "| --- | ---: | ---: | ---: |"]
    for condition, stats in by_condition.items():
        lines.append(f"| {condition} | {stats['n']} | {stats['em']:.3f} | {stats['stale_copied']:.3f} |")
    (output_dir / "api_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


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
        raw = chat_completion_with_retries(base_url, api_key, model, example["prompt"], timeout=timeout)
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


def env_or_arg(value: str | None, env_name: str) -> str:
    resolved = value or os.environ.get(env_name, "")
    if not resolved:
        raise SystemExit(f"Missing {env_name}. Pass the argument or set the environment variable.")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible API answer models for MemUpdateBench mechanism checks")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, or MUB_API_BASE_URL")
    parser.add_argument("--api-key", default="", help="API key, or MUB_API_KEY. Prefer environment variable to avoid shell history.")
    parser.add_argument("--model", default="", help="Model name, or MUB_API_MODEL")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--connectivity", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--synthetic-dose-probe", action="store_true")
    parser.add_argument("--stale-counts", default="0,1,2,4,8,16")
    parser.add_argument("--examples-per-condition", type=int, default=16)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    base_url = env_or_arg(args.base_url, "MUB_API_BASE_URL")
    api_key = env_or_arg(args.api_key, "MUB_API_KEY")
    print(f"Using API base URL: {base_url}")
    print(f"Using API key: {redact_secret(api_key)}")

    if args.list_models:
        for model_id in list_models(base_url, api_key, timeout=args.timeout):
            print(model_id)
        return 0

    model = env_or_arg(args.model, "MUB_API_MODEL")
    if args.connectivity:
        result = run_connectivity(base_url, api_key, model, timeout=args.timeout)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["ok"]:
            return 1

    if args.probe:
        summary = run_probe(base_url, api_key, model, Path(args.output_dir), timeout=args.timeout)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

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

    if not (args.list_models or args.connectivity or args.probe or args.synthetic_dose_probe):
        parser.error("Choose at least one action: --list-models, --connectivity, --probe, or --synthetic-dose-probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
