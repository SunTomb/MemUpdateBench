from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SmokeTestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name: str, msg: str = "") -> None:
        self.passed.append(name)
        print(f"  [PASS] {name}" + (f" -- {msg}" if msg else ""))

    def fail(self, name: str, error: Exception | str) -> None:
        self.failed.append((name, str(error)))
        print(f"  [FAIL] {name} -- {error}")

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 50}")
        print(f"SMOKE TEST: {len(self.passed)}/{total} passed")
        if self.failed:
            print("FAILED:")
            for name, err in self.failed:
                print(f"  [FAIL] {name}: {err}")
        print(f"{'=' * 50}")
        return not self.failed


def test_imports(results: SmokeTestResult) -> None:
    print("\n[1/6] Testing imports...")
    modules = [
        "mub.config",
        "mub.utils",
        "mub.memory.entry",
        "mub.memory.store",
        "mub.manager.memory_manager",
        "scripts.prepare_data",
        "scripts.eval_evomemory",
        "scripts.analyze_ood_errors",
        "scripts.summarize_update_frequency",
        "scripts.summarize_prompt_robustness",
        "scripts.analyze_action_pathology",
        "scripts.eval_mem0_baseline",
        "scripts.analyze_answer_layer_mechanism",
        "scripts.analyze_stale_intervention",
        "scripts.summarize_heuristic_threshold",
        "scripts.merge_evomemory_shards",
        "scripts.generate_constrained_sft",
        "scripts.train_constrained_sft",
        "scripts.probe_api_answer_model",
        "scripts.summarize_api_latest_model_probe",
    ]
    for module in modules:
        try:
            __import__(module)
            results.ok(f"import {module}")
        except Exception as exc:
            results.fail(f"import {module}", exc)


def test_config(results: SmokeTestResult) -> None:
    print("\n[2/6] Testing configuration...")
    try:
        from mub.config import MUBConfig

        config = MUBConfig()
        assert config.model.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.memory.retrieval_topk > 0
        assert config.wandb_project == "mem_update_bench"
        results.ok("MUBConfig defaults")

        import yaml

        tmp_path = os.path.join(tempfile.gettempdir(), "mub_test_config.yaml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump({"model": {"model_name": "test-model"}, "seed": 123}, f)
        loaded = MUBConfig.from_yaml(tmp_path)
        assert loaded.model.model_name == "test-model"
        assert loaded.seed == 123
        os.unlink(tmp_path)
        results.ok("MUBConfig from YAML")
    except ImportError:
        results.ok("MUBConfig from YAML skipped")
    except Exception as exc:
        results.fail("MUBConfig", exc)


def test_memory_system(results: SmokeTestResult) -> None:
    print("\n[3/6] Testing memory system...")
    try:
        import numpy as np
        from mub.memory.entry import MemoryEntry
        from mub.memory.store import MemoryStore

        entry = MemoryEntry(
            content="User says: My friend Alex lives in Shanghai.",
            keywords=["friend", "alex"],
            tags=["fact"],
            slot={"entity": "friend_alex", "attribute": "location", "value": "Shanghai", "event_idx": 0},
        )
        assert entry.id
        assert MemoryEntry.from_dict(entry.to_dict()).slot == entry.slot
        results.ok("MemoryEntry serialization")

        store = MemoryStore()
        store._encode = lambda text: np.zeros(store.config.embedding_dim, dtype="float32")
        stored = store.add(entry.content, slot_meta=entry.slot)
        store.update(
            stored.id,
            "User says: Alex relocated to Chengdu.",
            slot_meta={"entity": "friend_alex", "attribute": "location", "value": "Chengdu", "event_idx": 1},
        )
        latest = store.get_latest_by_slot("friend_alex", "location")
        assert latest is not None
        assert latest.slot["value"] == "Chengdu"
        results.ok("MemoryStore slot update")
    except Exception as exc:
        results.fail("memory system", exc)


def test_utils(results: SmokeTestResult) -> None:
    print("\n[4/6] Testing utilities...")
    try:
        from mub.utils import compute_exact_match, compute_f1, compute_kendall_tau

        assert compute_f1("the cat sat", "the cat") > 0.5
        assert compute_exact_match("hello", "hello") == 1.0
        assert compute_kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
        results.ok("metrics utilities")
    except Exception as exc:
        results.fail("metrics utilities", exc)


def test_update_frequency_data(results: SmokeTestResult) -> None:
    print("\n[5/6] Testing update-frequency data generation...")
    try:
        from scripts.prepare_data import prepare_evomemory

        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_evomemory(tmpdir, variant="update_frequency_hard", seed=53, output_suffix="smoke")
            path = Path(tmpdir) / "evomemory_update_frequency_hard_k16_smoke_test.json"
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data
            assert data[0]["stress_type"] == "update_frequency_hard"
            assert data[0]["k_updates"] == 16
            assert data[0]["distractor_level"] == "same_name_multi_entity"
            results.ok("update_frequency_hard split generation")

            prepare_evomemory(tmpdir, variant="update_frequency_expanded", seed=83, output_suffix="smoke")
            path = Path(tmpdir) / "evomemory_update_frequency_expanded_k16_smoke_test.json"
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data
            assert data[0]["stress_type"] == "update_frequency_expanded"
            assert data[0]["k_updates"] == 16
            assert {item["attribute"] for item in data} >= {"timezone", "hobby", "instrument", "project"}
            results.ok("update_frequency_expanded split generation")
    except Exception as exc:
        results.fail("update_frequency_hard split generation", exc)


def test_constrained_slots(results: SmokeTestResult) -> None:
    print("\n[6/6] Testing constrained slot logic...")
    try:
        import numpy as np
        from mub.manager.memory_manager import MemoryManager
        from mub.memory.store import MemoryStore
        from scripts.eval_evomemory import (
            EpisodeEntityResolver,
            build_slot_answer_prompt,
            filter_latest_per_slot,
            format_memory_context,
            parse_event_slot,
            retrieved_trace,
            run_constrained_slot_crud,
        )

        resolver = EpisodeEntityResolver()
        parse_event_slot("User says: My manager Tom works at Tencent.", 0, resolver=resolver)
        slot = parse_event_slot("User says: Tom joined JD.", 1, resolver=resolver)
        assert slot["entity"] == "manager_tom"
        assert slot["attribute"] == "company"
        assert slot["value"] == "JD"
        assert parse_event_slot("User says: my advisor Nora timezone is UTC+8.", 2)["attribute"] == "timezone"
        assert parse_event_slot("User says: my teammate Leo hobby is climbing.", 3)["attribute"] == "hobby"
        assert parse_event_slot("User says: my cousin Mia instrument is erhu.", 4)["attribute"] == "instrument"
        assert parse_event_slot("User says: my neighbor Hank now works on project Atlas.", 5)["attribute"] == "project"

        parsed = MemoryManager.parse_constrained_slot_operation("UPDATE brother_chen.language = Kotlin NOOP")
        assert parsed["operation"] == "UPDATE"
        assert parsed["value"] == "Kotlin"

        store = MemoryStore()
        store._encode = lambda text: np.zeros(store.config.embedding_dim, dtype="float32")
        stats, actions = run_constrained_slot_crud(store, [
            "User says: My friend Alex lives in Shanghai.",
            "User says: Alex relocated to Chengdu.",
        ])
        assert actions[0]["operation"] == "ADD"
        assert actions[1]["operation"] == "UPDATE"
        assert stats["requested"]["UPDATE"] == 1
        assert store.get_latest_by_slot("friend_alex", "location").slot["value"] == "Chengdu"

        prompt = build_slot_answer_prompt(
            "Where does Alex live?",
            "- User says: Alex relocated to Chengdu.",
            {"entity": "friend_alex", "attribute": "location"},
            "v2_ignore_distractors",
        )
        assert "Ignore all memories" in prompt
        relevant = store.retrieve("Where does Alex live?", topk=5)
        trace = retrieved_trace(
            {"entity": "friend_alex", "attribute": "location", "answer": "Chengdu"},
            relevant,
            answer_topk=5,
        )
        assert trace["gold_value_in_retrieved"]
        assert "retrieved_entries" in trace

        raw_store = MemoryStore()
        raw_store._encode = lambda text: np.zeros(raw_store.config.embedding_dim, dtype="float32")
        first = raw_store.add(
            "User says: My friend Alex lives in Shanghai.",
            slot_meta={"entity": "friend_alex", "attribute": "location", "value": "Shanghai", "event_idx": 0},
        )
        second = raw_store.add(
            "User says: Alex relocated to Chengdu.",
            slot_meta={"entity": "friend_alex", "attribute": "location", "value": "Chengdu", "event_idx": 1},
        )
        filtered = filter_latest_per_slot([(first, 0.9), (second, 0.8)], raw_store, topk=5)
        assert len(filtered) == 1
        assert filtered[0][0].slot["value"] == "Chengdu"

        context, ordered = format_memory_context(
            [(first, 0.9), (second, 0.8)],
            {"entity": "friend_alex", "attribute": "location"},
            context_order="current_first",
            context_annotation="latest_outdated_label",
        )
        assert ordered[0][0].slot["value"] == "Chengdu"
        assert "[latest]" in context
        assert "[outdated]" in context

        import mub.utils as mub_utils
        from scripts.eval_evomemory import answer_question
        original_generate_text = mub_utils.generate_text
        mub_utils.generate_text = lambda model, tokenizer, prompt, **kwargs: "Chengdu"
        try:
            answer, policy_trace = answer_question(
                None,
                None,
                "Where does Alex live?",
                raw_store,
                slot_prompt={"entity": "friend_alex", "attribute": "location", "answer": "Chengdu"},
                answer_topk=1,
                retrieval_policy="latest_per_slot",
                return_trace=True,
            )
        finally:
            mub_utils.generate_text = original_generate_text
        assert answer == "Chengdu"
        assert policy_trace["retrieval_policy"] == "latest_per_slot"
        assert policy_trace["retrieved_entries"][0]["slot"]["value"] == "Chengdu"

        from scripts.run_conflict_type_probe import make_example as make_conflict_example
        conflict_example = make_conflict_example(
            example_id=0,
            attribute="location",
            condition="stale_same_slot",
            distractor_count=2,
        )
        assert "Target entity: friend_alice" in conflict_example["prompt"]
        assert "Target attribute: location" in conflict_example["prompt"]
        assert conflict_example["prompt"].count("User says:") == 3

        from scripts.run_synthetic_same_slot_probe import build_context as build_synthetic_context
        middle_context = build_synthetic_context(
            "friend",
            "Alex",
            "location",
            "Chengdu",
            ["Shanghai", "Beijing", "Wuhan", "Nanjing"],
            "middle",
            "latest_outdated_label",
        )
        assert middle_context.count("User says:") == 5
        assert "[latest]" in middle_context
        random_context = build_synthetic_context(
            "friend",
            "Alex",
            "location",
            "Chengdu",
            ["Shanghai", "Beijing"],
            "random",
            "none",
        )
        assert random_context.count("User says:") == 3

        results.ok("parser, constrained CRUD, answer traces, and latest-slot retrieval policy")
    except Exception as exc:
        results.fail("parser and constrained CRUD", exc)


def test_api_probe_helpers(results: SmokeTestResult) -> None:
    print("\n[7/7] Testing API probe helpers...")
    try:
        from scripts.probe_api_answer_model import (
            build_chat_payload,
            build_headers,
            build_probe_examples,
            exact_value_prediction,
            parse_json_body,
            parse_sse_chat_body,
            redact_secret,
        )

        sse = b'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}\n\ndata: [DONE]'
        assert parse_sse_chat_body(sse) == {"choices": [{"message": {"content": "OK"}}]}
        assert parse_json_body(b'{"ok": true}', "/models") == {"ok": True}
        try:
            parse_json_body(b"not json", "/chat/completions")
            raise AssertionError("parse_json_body should reject non-JSON responses")
        except RuntimeError as exc:
            assert "Non-JSON response from /chat/completions" in str(exc)
        headers = build_headers("testkey_abcdef1234567890")
        assert headers["Authorization"] == "Bearer testkey_abcdef1234567890"
        assert "Claude-Code" in headers["User-Agent"]
        assert redact_secret("testkey_abcdef1234567890") == "tes...7890"
        payload = build_chat_payload("gpt-test", "Answer OK", max_tokens=8, temperature=0.0)
        assert payload["model"] == "gpt-test"
        assert payload["messages"][-1]["content"] == "Answer OK"
        assert payload["max_tokens"] == 8
        assert payload["temperature"] == 0.0

        examples = build_probe_examples()
        names = {example["condition"] for example in examples}
        assert "final_only" in names
        assert "stale_same_slot_reverse_no_label" in names
        assert "stale_same_slot_reverse_with_label" in names
        reverse = next(example for example in examples if example["condition"] == "stale_same_slot_reverse_no_label")
        assert reverse["gold"] == "Chengdu"
        assert reverse["stale_values"] == ["Beijing", "Shanghai"]
        assert exact_value_prediction("Chengdu.") == "Chengdu"
        assert exact_value_prediction("The answer is Chengdu") == "Chengdu"
        assert exact_value_prediction("Kunming") == "Kunming"
        assert exact_value_prediction("The answer is Urumqi.") == "Urumqi"

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

        from scripts.probe_api_answer_model import should_retry_api_error

        assert should_retry_api_error(RuntimeError("Non-JSON response from /chat/completions: <!doctype html>"))
        assert should_retry_api_error(RuntimeError("HTTP 429 from /chat/completions: rate limit"))
        assert should_retry_api_error(RuntimeError("HTTP 500 from /chat/completions: upstream"))
        assert should_retry_api_error(TimeoutError("The read operation timed out"))
        assert not should_retry_api_error(RuntimeError("Unexpected chat response schema: {}"))
        results.ok("API probe helpers")
    except Exception as exc:
        results.fail("API probe helpers", exc)


def test_vnext_contracts(results: SmokeTestResult) -> None:
    print("\n[8/8] Testing vNext contracts...")
    name = "vNext contracts, replay, serialization, and capability gating"
    try:
        from mub.vnext.contracts import (
            ActionScope,
            AnswerSchema,
            CompletionStatus,
            Difficulty,
            EvaluationMode,
            EventRole,
            MemUpdateTask,
            MemoryObjectKey,
            MetricFieldSupport,
            Operation,
            QueryType,
            RunManifest,
            ScoreRecord,
            SourceType,
            Split,
            SupportReason,
            TaskFamily,
            TaskManifest,
            TaskRunRecord,
        )
        from mub.vnext.contracts.score import SCORE_LAYER_TYPES
        from mub.vnext.io.canonical import canonical_json_bytes, sha256_model
        from mub.vnext.io.jsonl import read_models, write_models
        from mub.vnext.validation import replay_actions, validate_task_semantics

        assert all(
            model.model_fields
            for model in (
                MemUpdateTask,
                TaskRunRecord,
                ScoreRecord,
                TaskManifest,
                RunManifest,
            )
        )

        object_key = MemoryObjectKey(
            object_type="slot",
            namespace="default",
            entity="friend:alex",
            attribute="location",
            subkey=None,
        )
        metadata_variant = MemoryObjectKey(
            object_type="profile_field",
            namespace="default",
            entity="friend:alex",
            attribute="location",
            subkey=None,
        )
        assert object_key.canonical_id == metadata_variant.canonical_id
        assert object_key.canonical_id == "default|friend:alex|location|"

        task = MemUpdateTask.model_validate(
            {
                "task_id": "task_vnext_smoke_add_update_0001",
                "task_family": TaskFamily.REPEATED_SAME_SLOT.value,
                "difficulty": Difficulty.EASY,
                "source": {
                    "source_id": "source_vnext_smoke_0001",
                    "source_type": SourceType.SYNTHETIC,
                    "source_uri": "memory://vnext-smoke/source-0001",
                    "license_or_privacy": "synthetic_redistributable",
                    "raw_hash": "a" * 64,
                    "normalized_hash": "b" * 64,
                    "normalization_version": "1.0.0",
                    "provenance": {
                        "source_group_id": "source_group_vnext_smoke_0001",
                        "redistributable": True,
                    },
                    "generator": {
                        "generator_name": "vnext_smoke",
                        "seed": 0,
                        "config_sha256": "c" * 64,
                        "code_revision": "fixed-smoke-revision",
                        "compiler_version": "1.0.0",
                    },
                },
                "events": [
                    {
                        "event_id": "event_0",
                        "sequence_index": 0,
                        "timestamp": None,
                        "raw_text": "My friend Alex lives in Dalian.",
                        "normalized_text": "My friend Alex lives in Dalian.",
                        "speaker": None,
                        "gold_action_ids": ["action_0"],
                        "role": EventRole.STALE_SAME_SLOT,
                    },
                    {
                        "event_id": "event_1",
                        "sequence_index": 1,
                        "timestamp": None,
                        "raw_text": "My friend Alex relocated to Qingdao.",
                        "normalized_text": "My friend Alex relocated to Qingdao.",
                        "speaker": None,
                        "gold_action_ids": ["action_1"],
                        "role": EventRole.LATEST_GOLD,
                    },
                ],
                "target_objects": [object_key],
                "queries": [
                    {
                        "query_id": "query_0",
                        "query_type": QueryType.CURRENT_STATE,
                        "text": "Where does my friend Alex live now?",
                        "target_object_keys": [object_key],
                        "answer_schema": AnswerSchema.STRING,
                        "evaluation_mode": EvaluationMode.RETRIEVED_PROMPT,
                    }
                ],
                "gold": {
                    "actions": [
                        {
                            "action_id": "action_0",
                            "event_id": "event_0",
                            "operation": Operation.ADD,
                            "scope": ActionScope.ATTRIBUTE,
                            "target_object_keys": [object_key],
                            "value": "Dalian",
                            "effective_at": None,
                        },
                        {
                            "action_id": "action_1",
                            "event_id": "event_1",
                            "operation": Operation.UPDATE,
                            "scope": ActionScope.ATTRIBUTE,
                            "target_object_keys": [object_key],
                            "value": "Qingdao",
                            "effective_at": None,
                        },
                    ],
                    "action_sequence": ["action_0", "action_1"],
                    "final_state": {object_key.canonical_id: "Qingdao"},
                    "version_history": {
                        object_key.canonical_id: ["Dalian", "Qingdao"]
                    },
                    "expected_present_objects": [object_key],
                    "expected_absent_objects": [],
                    "gold_source_event_ids": ["event_1"],
                    "gold_answers": {"query_0": "Qingdao"},
                    "acceptable_answers": {"query_0": ["Qingdao"]},
                },
                "metadata": {
                    "split": Split.TEST,
                    "split_key": {
                        "semantic_core_id": "semantic_core_vnext_smoke_0001",
                        "source_group_id": "source_group_vnext_smoke_0001",
                        "trajectory_id": "trajectory_vnext_smoke_0001",
                        "split_policy_version": "1.0.0",
                    },
                    "profile_name": Difficulty.EASY,
                    "resolved_profile": {"update_depth": 1},
                    "generation_config_hash": "c" * 64,
                    "compiler_version": "1.0.0",
                },
            }
        )

        validation = validate_task_semantics(task)
        assert validation.valid, validation.issues
        actions_by_id = {action.action_id: action for action in task.gold.actions}
        replay = replay_actions(
            actions_by_id[action_id] for action_id in task.gold.action_sequence
        )
        assert dict(replay.final_state) == dict(task.gold.final_state)
        assert dict(replay.version_history) == {
            object_key.canonical_id: ("Dalian", "Qingdao")
        }
        assert replay.mutation_count == 2

        canonical_payload = canonical_json_bytes(task)
        original_hash = sha256_model(task)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "task.json"
            json_path.write_bytes(canonical_payload)
            json_restored = MemUpdateTask.model_validate_json(json_path.read_bytes())
            assert json_restored == task
            assert sha256_model(json_restored) == original_hash

            jsonl_path = Path(tmpdir) / "tasks.jsonl"
            write_models(jsonl_path, [task], id_field="task_id")
            assert jsonl_path.read_bytes() == canonical_payload + b"\n"
            restored_rows = list(
                read_models(jsonl_path, MemUpdateTask, id_field="task_id")
            )
            assert restored_rows == [task]
            assert sha256_model(restored_rows[0]) == original_hash

        score_layers = {
            layer_name: ({"action_parse_valid": True} if layer_name == "protocol_scores" else {})
            for layer_name in SCORE_LAYER_TYPES
        }
        support_map = {}
        for layer_name, layer_type in SCORE_LAYER_TYPES.items():
            for field_name in layer_type.model_fields:
                if layer_name == "protocol_scores" and field_name == "action_parse_valid":
                    continue
                path = f"{layer_name}.{field_name}"
                support_map[path] = MetricFieldSupport(
                    reason=SupportReason.NOT_APPLICABLE,
                    null_policy="exclude_from_aggregation",
                    detail="not requested by this focused smoke row",
                )
        reason_cases = {
            "answer_scores.structured_field_accuracy": (
                SupportReason.NOT_APPLICABLE,
                "string-answer query has no structured fields",
            ),
            "retrieval_scores.current_mrr": (
                SupportReason.NOT_SUPPORTED,
                "adapter capability does not export ranked retrieval scores",
            ),
            "action_scores.operation_accuracy": (
                SupportReason.RUNTIME_FAILED,
                "partial runtime did not complete the expected action trace",
            ),
            "state_scores.final_state_accuracy": (
                SupportReason.MISSING_ARTIFACT,
                "final state snapshot artifact is absent",
            ),
        }
        for path, (reason, detail) in reason_cases.items():
            support_map[path] = MetricFieldSupport(
                reason=reason,
                null_policy="exclude_from_aggregation",
                detail=detail,
            )

        score = ScoreRecord(
            task_id=task.task_id,
            run_id="run_vnext_smoke_0001",
            adapter_id="adapter_vnext_smoke",
            task_family=task.task_family,
            difficulty=task.difficulty,
            completion_status=CompletionStatus.PARTIAL,
            supported_metric_fields=support_map,
            **score_layers,
        )
        score = ScoreRecord.model_validate(score.model_dump(mode="json"))
        observed = {
            f"{layer_name}.{field_name}": value
            for layer_name in SCORE_LAYER_TYPES
            for field_name, value in getattr(score, layer_name)
        }
        null_paths = {path for path, value in observed.items() if value is None}
        denominator_paths = {path for path, value in observed.items() if value is not None}
        assert null_paths == set(score.supported_metric_fields)
        assert denominator_paths == {"protocol_scores.action_parse_valid"}
        assert denominator_paths.isdisjoint(score.supported_metric_fields)
        assert {
            score.supported_metric_fields[path].reason for path in reason_cases
        } == {
            SupportReason.NOT_APPLICABLE,
            SupportReason.NOT_SUPPORTED,
            SupportReason.RUNTIME_FAILED,
            SupportReason.MISSING_ARTIFACT,
        }

        results.ok(name)
    except Exception as exc:
        results.fail(name, exc)


def main() -> int:
    print("=" * 50)
    print("MemUpdateBench SMOKE TEST")
    print("=" * 50)

    results = SmokeTestResult()
    test_imports(results)
    test_config(results)
    test_memory_system(results)
    test_utils(results)
    test_update_frequency_data(results)
    test_constrained_slots(results)
    test_api_probe_helpers(results)
    test_vnext_contracts(results)
    return 0 if results.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
