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
        results.ok("parser, constrained CRUD, answer traces, and latest-slot retrieval filter")
    except Exception as exc:
        results.fail("parser and constrained CRUD", exc)


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
    return 0 if results.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
