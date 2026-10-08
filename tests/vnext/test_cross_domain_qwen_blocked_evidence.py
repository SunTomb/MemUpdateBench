from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import pytest


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cross_domain_qwen_blocked"
RESULT_INDEX_PINS = {
    "v2": {
        "index_sha256": "89aebaec527853d37142bc81cfa07ab9b3d3c89ac7371607b72253a6c1192349",
        "qualification_processes": 0,
        "files": {
            "authorization_profile.json": (1223, "69958f6558b55418e2890bb73b3c73f39af9f3c3be7caec0421eb1331801df0e"),
            "summary.json": (1325, "858c1a1a65f38aa5f003a69ca6a0c94c2fbc1caca429cae5261ab8ebbd313c5e"),
        },
    },
    "v3": {
        "index_sha256": "4061638a7652a9be5a0a72b53295b47fee40876dd9f38fb6f4348ac99668732e",
        "qualification_processes": 1,
        "files": {
            "authorization_profile.json": (1223, "69ff07e590095cdb3c3c52c57276d990cf0d106aa4c73d2b15e267ca79111421"),
            "summary.json": (1438, "16871b7eaea05a6522cf9670e009ce1de786ddb3bc1edb2555be926413d80b6b"),
            "worker_error.json": (310, "e008fca940d8532a8ce6560ffcd2350d40ec9d9da3262171f15ffed849fb3b38"),
            "worker_journal.jsonl": (83, "5684a5d86be4747d593067428d0b81964c4533a5126b84ca7588f13a453cd15b"),
        },
    },
    "v4": {
        "index_sha256": "cb49d495991c0591db4e1f395f04ece3f4b8c30febdba890d4228140467816bc",
        "qualification_processes": 0,
        "files": {
            "authorization_profile.json": (1223, "5ff6208774e25b81add85c106a1c1ebf5165f0ef1111738cf97c0b47a58b4e6e"),
            "summary.json": (1324, "70adb89d944aeaa77c327c7d46a84695673b2bca18773115d1c1df24bd00ab22"),
        },
    },
}

PREPARATION_INDEX_SHA256 = "1112a371f07326130634715cfc15fe656aa11a48539f2d16b06f39c400f9e687"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_TREE_SHA256 = "e4e43ba06e1da35da5b24b13a3d41ee4354c8c23592dd7ef8d57ea81dc6628db"
CPU_PROXY_RECEIPT_SHA256 = "62a127595d423157ba7d454a9a3c3a5626bbb5ba304ecf4bda8b485ac11075d4"
V4_SOURCE_MANIFEST_SHA256 = "3de4969ab6e85972586b4da2a542ce467b9c21af9585063866b011cdc2027730"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", ("v2", "v3", "v4"))
def test_blocked_result_index_and_indexed_artifacts_are_pinned(version):
    expected = RESULT_INDEX_PINS[version]
    root = FIXTURE_ROOT / version
    index_raw = (root / "index.json").read_bytes()
    assert sha256(index_raw) == expected["index_sha256"]

    index = json.loads(index_raw)
    assert index["schema"] == "memupdatebench.cross-domain.qwen-readout-index.v1"
    assert index["status"] == "BLOCKED"
    assert index["scientific_release_allowed"] is False
    artifacts = {artifact["path"]: artifact for artifact in index["artifacts"]}
    assert set(artifacts) == set(expected["files"])

    for name, (expected_bytes, expected_sha256) in expected["files"].items():
        artifact = artifacts[name]
        raw = (root / name).read_bytes()
        assert artifact["bytes"] == expected_bytes == len(raw)
        assert artifact["sha256"] == expected_sha256 == sha256(raw)


@pytest.mark.parametrize("version", ("v2", "v3", "v4"))
def test_blocked_summary_and_authorization_profile_remain_bounded(version):
    expected = RESULT_INDEX_PINS[version]
    root = FIXTURE_ROOT / version
    summary = load_json(root / "summary.json")
    profile_raw = (root / "authorization_profile.json").read_bytes()
    profile = json.loads(profile_raw)

    assert summary["status"] == "BLOCKED"
    assert summary["scientific_release_allowed"] is False
    assert summary["benchmark_accuracy_claimed"] is False
    assert summary["answer_metrics"] is None
    assert summary["scores"] == []
    assert summary["profile_sha256"] == sha256(profile_raw)
    assert summary["generation_attempts"] == 0
    assert summary["model_loads_completed"] == 0
    assert summary["prospective_requests"] == 4
    assert all(value is True for value in summary["cleanup"].values())

    accounting = summary["execution_accounting"]
    for key in ("load_worker_launches", "paid_provider_calls", "qdrant_calls", "retries"):
        assert accounting[key] == 0
    assert accounting["qualification_processes"] == expected["qualification_processes"]

    run_id = f"cross_domain_qwen_readout_20261002_{version}"
    assert profile["run_id"] == run_id
    assert profile["model_id"] == summary["model_id"] == MODEL_ID
    assert profile["revision"] == summary["model_revision"] == MODEL_REVISION
    assert profile["model_tree_sha256"] == summary["model_tree_sha256"] == MODEL_TREE_SHA256
    assert profile["preparation_index_sha256"] == summary["preparation_index_sha256"] == PREPARATION_INDEX_SHA256
    assert profile["source_manifest_sha256"] == summary["source_manifest_sha256"]
    assert profile["max_new_tokens"] == 64
    assert profile["maximum_generated_tokens"] == 256
    assert profile["maximum_generations"] == 4
    assert profile["maximum_load_attempts"] == 1
    assert profile["context_token_cap"] == 4096
    assert profile["retries"] == 0
    assert profile["total_seconds"] == 3600
    assert profile["paid_provider_calls_allowed"] is False
    assert profile["qdrant_calls_allowed"] is False
    assert profile["scientific_release_allowed"] is False


def test_tang1_real_readout_preserves_load_failure_and_cleanup():
    root = FIXTURE_ROOT / "tang1gpu1d"
    index_raw = (root / "index.json").read_bytes()
    assert sha256(index_raw) == "5aaf45dcba2092826c251f38e3f8b518aa86d09718e80fb9fb985f31f3f4705b"
    index = json.loads(index_raw)
    expected = {
        "authorization_profile.json": (1291, "88a3a28683dd6fe81e128c5b6a0c1c7d59e7132435efe9285e597d2768f0952b"),
        "qualification.json": (235862, "94158fe96db43143ead4d651a7300e62f0cba76279efa855b82ef2621a712eab"),
        "qualification_journal.jsonl": (124, "7a1ba6d0585e34ce039de500e726b91302230e0d27e5f8986bd9bf689c1aa0d6"),
        "readouts.json": (1889, "9eab4d98fcefc04f62ba230f842615e4957a1329930e3e6f06ca67101576a588"),
        "summary.json": (2210, "9556f27bd386c3f74aee94d2a404fda8a3c718b9211439ad3021fa0c409cf451"),
        "worker_journal.jsonl": (2315, "3608a9524aaae36cd2174ef4835d72148d9fb7803d854653653852c194852132"),
        "worker_result.json": (2710, "8fa74566e0b1c96ec9a259cc0d96360ec26261ea4c5712767b9db054734b5ce8"),
    }
    assert index["status"] == "BLOCKED" and index["scientific_release_allowed"] is False
    assert {item["path"] for item in index["artifacts"]} == set(expected)
    for name, (size, digest) in expected.items():
        raw = (root / name).read_bytes()
        item = next(item for item in index["artifacts"] if item["path"] == name)
        assert len(raw) == size == item["bytes"] and sha256(raw) == digest == item["sha256"]

    summary = load_json(root / "summary.json")
    assert summary["status"] == "BLOCKED" and summary["answer_metrics"] is None
    assert summary["device"] == {
        "foreign_process_count": 1, "free_mib": 38158, "index": 1,
        "name": "NVIDIA A40", "node": "Tang1",
        "uuid": "GPU-479aa50d-c1ce-0bc5-e91e-770c97243178",
    }
    assert summary["execution_accounting"] == {
        "load_worker_launches": 1, "paid_provider_calls": 0,
        "peak_temporary_bytes": 6115328, "qdrant_calls": 0,
        "qualification_processes": 1, "retries": 0,
    }
    assert summary["model_loads_completed"] == 1
    assert summary["generation_attempts"] == 1
    assert all(value is True for value in summary["cleanup"].values())

    worker = load_json(root / "worker_result.json")
    assert worker["postvalidation"] is True
    assert worker["lifecycle"]["load_completed"] is True
    assert worker["lifecycle"]["error_phase"] == "before_generation"
    assert worker["lifecycle"]["error_type"] == "RuntimeError"
    assert worker["lifecycle"]["close"]["allocated_bytes_after"] == 0
    assert worker["backend_observations"]["load_attempts"] == 1
    assert worker["backend_observations"]["generation_attempts"] == 1
    assert [row["status"] for row in worker["rows"]] == [
        "TECHNICAL_FAILURE", "NOT_RUN", "NOT_RUN", "NOT_RUN"
    ]
    assert worker["rows"][0]["attempt_intent"] == worker["rows"][0]["callback_entries"] == 1
    assert worker["rows"][0]["attempts"] == 1
    assert all(row["prediction"] is None for row in worker["rows"])

    qualification = load_json(root / "qualification.json")
    assert qualification["status"] == "QUALIFIED_NO_MODEL_LOAD"
    assert qualification["model_loads"] == qualification["generations"] == qualification["provider_calls"] == 0
    assert [binding["input_tokens"] for binding in qualification["bindings"]] == [258, 261, 246, 247]
    assert qualification["model_class"] == "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM"
    for path in root.iterdir():
        if path.suffix in (".json", ".jsonl"):
            raw = path.read_bytes()
            assert b'"raw_output"' not in raw
            assert b'Use only the retrieved memory entries' not in raw



    root = FIXTURE_ROOT / "v3"
    error = load_json(root / "worker_error.json")
    assert error == {
        "error_type": "ValueError",
        "frames": [
            {"file": "vnext_run_cross_domain_qwen_readout.py", "function": "worker", "line": 326},
            {"file": "vnext_run_cross_domain_qwen_readout.py", "function": "runtime_identity", "line": 261},
            {"file": "vnext_run_cross_domain_qwen_readout.py", "function": "require", "line": 51},
        ],
        "phase": "qualify",
    }
    journal = [json.loads(line) for line in (root / "worker_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert journal == [
        {"phase": "qualification", "sequence": 0},
        {"phase": "template_preflight", "sequence": 1},
    ]


def test_cpu_proxy_receipt_is_pinned_and_contains_only_safe_bindings():
    raw = (FIXTURE_ROOT / "cpu_proxy_validation.json").read_bytes()
    assert sha256(raw) == CPU_PROXY_RECEIPT_SHA256
    receipt = json.loads(raw)

    assert receipt["schema"] == "memupdatebench.cross-domain.cpu-proxy-validation.v1"
    assert receipt["status"] == "PASS_CPU_ONLY"
    assert receipt["scope"] == "CPU proxy fix and four template bindings; not full snapshot qualification or model evidence"
    assert receipt["source_manifest_sha256"] == V4_SOURCE_MANIFEST_SHA256
    assert receipt["preparation_index_sha256"] == PREPARATION_INDEX_SHA256
    assert receipt["model_loads"] == 0
    assert receipt["generations"] == 0
    assert receipt["provider_calls"] == 0
    assert receipt["gpu_operations"] == 0
    assert receipt["tokenizer_loads"] == 1
    assert receipt["close"]["load_attempts"] == 0
    assert receipt["close"]["generation_attempts"] == 0
    assert receipt["close"]["tokenizer_loads"] == 1
    assert receipt["close"]["close"]["status"] == "CLOSED"
    assert "dtype" not in receipt and "model_dtype" not in receipt

    framework_proxies = receipt["runtime"]["framework_proxies"]
    assert framework_proxies == [
        {"implementation": "torch._classes", "module": "torch.classes"},
        {"implementation": "torch._ops", "module": "torch.ops"},
    ]
    files = receipt["runtime"]["files"]
    members = [entry["member"] for entry in files]
    assert len(files) == 1423
    assert len(set(members)) == len(members)
    assert {"torch/_classes.py", "torch/_ops.py"}.issubset(members)
    for entry in files:
        assert set(entry) == {"bytes", "distribution", "member", "sha256"}
        member = PurePosixPath(entry["member"])
        assert not member.is_absolute()
        assert all(part not in ("", ".", "..") for part in member.parts)
        assert ":" not in member.parts[0]
        assert "\\" not in entry["member"]
        assert entry["bytes"] >= 0
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])

    bindings = receipt["bindings"]
    assert [binding["request_id"] for binding in bindings] == [
        "bea-repeat-0", "bea-repeat-1", "noaa-repeat-0", "noaa-repeat-1"
    ]
    assert [binding["input_tokens"] for binding in bindings] == [258, 261, 246, 247]
    for binding in bindings:
        assert set(binding) == {
            "input_tokens", "rendered_prompt_sha256", "request_id", "template_sha256", "visible_prompt_sha256"
        }
        for field in ("rendered_prompt_sha256", "template_sha256", "visible_prompt_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", binding[field])
