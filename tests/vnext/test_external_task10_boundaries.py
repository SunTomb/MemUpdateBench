from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mub.vnext.contracts.v3.version import (
    ADAPTER_CONTRACT_VERSION_V3,
    SCHEMA_VERSION_V3,
    TASK_MANIFEST_VERSION_V3,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fresh_external_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_external_import_is_provider_neutral_and_leaves_schema_registry_untouched() -> None:
    result = _fresh_external_probe(
        """
import json
import sys
from mub.vnext.schema_export import SCHEMA_MODEL_REGISTRIES
before_modules = set(sys.modules)
before_registry = tuple(SCHEMA_MODEL_REGISTRIES)
import mub.vnext.external as external
new_modules = set(sys.modules) - before_modules
provider_modules = sorted(
    name for name in new_modules
    if name == "mem0" or name.startswith("mem0.")
    or name == "langgraph" or name.startswith("langgraph.")
)
print(json.dumps({
    "version": external.EXTERNAL_ADMISSION_CONTRACT_VERSION,
    "provider_modules": provider_modules,
    "registry_before": before_registry,
    "registry_after": tuple(SCHEMA_MODEL_REGISTRIES),
}))
"""
    )
    assert result["version"] == "1.0.0"
    assert result["provider_modules"] == []
    assert result["registry_after"] == result["registry_before"]
    assert (SCHEMA_VERSION_V3, TASK_MANIFEST_VERSION_V3, ADAPTER_CONTRACT_VERSION_V3) == (
        "3.0.0",
        "3.0.0",
        "3.0.0",
    )


def test_external_import_does_not_change_checked_in_strict_v3_schemas() -> None:
    result = _fresh_external_probe(
        """
import json
from pathlib import Path
schema_dir = Path.cwd() / "schemas" / "vnext" / "v3"
before = sorted(path.name for path in schema_dir.glob("*.schema.json"))
import mub.vnext.external
after = sorted(path.name for path in schema_dir.glob("*.schema.json"))
print(json.dumps({"before": before, "after": after}))
"""
    )
    assert result["after"] == result["before"]


def test_registry_accepts_only_fixed_candidate_ids_and_rejects_prior_proxy_evidence() -> None:
    from mub.vnext.external import (
        CANDIDATE_LABELS,
        DENIED_EXTERNAL_EVIDENCE_LABELS,
        ExternalCandidateId,
        reject_denied_evidence,
        resolve_candidate_id,
    )

    assert tuple(CANDIDATE_LABELS) == (
        ExternalCandidateId.MEM0_OSS,
        ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE,
    )
    assert resolve_candidate_id("mem0_oss") is ExternalCandidateId.MEM0_OSS
    assert (
        resolve_candidate_id("langgraph_store_extract_then_store")
        is ExternalCandidateId.LANGGRAPH_STORE_EXTRACT_THEN_STORE
    )
    assert DENIED_EXTERNAL_EVIDENCE_LABELS == frozenset(
        {
            "memory_r1",
            "mem0_memory_r1",
            "baselines/memory_r1_agent.py",
            "scripts/eval_mem0_baseline.py",
        }
    )

    for denied in DENIED_EXTERNAL_EVIDENCE_LABELS | {
        "evidence/local_approximation/probe.json",
        "evidence/local-approximation/probe.json",
        "evidence/local approximation/probe.json",
        "evidence/local--approximation/probe.json",
        "evidence/local  approximation/probe.json",
        "evidence/local__approximation/probe.json",
        "evidence/local_-approximation/probe.json",
        "evidence/local_ - approximation/probe.json",
        "evidence/local approximation/probe.json",
        "evidence/local‑approximation/probe.json",
        "evidence/memory_-r1/result.json",
        "evidence/memory_ - r1/result.json",
        "evidence/memory__r1/result.json",
        "evidence/memory r1/result.json",
        "evidence/Memory‑R1/result.json",
        "evidence/local.approximation/probe.json",
        "evidence/local_approximation.json",
        "evidence/memory.r1/result.json",
        "evidence/memory_r1.json",
        "evidence/baselines/memory_r1_agent.py",
        "prefix/evidence/scripts/eval_mem0_baseline.py",
        "_local_approximation",
        "local_approximation_",
        "evidence/!local-approximation!/probe.json",
        "evidence/memory_r1.json.bak",
        "evidence/local_approximation.tar.gz",
        r"evidence\memory_r1\result.json",
        "scripts/eval_mem0_baseline.py.bak",
        "scripts/eval_mem0_baseline.py.tar.gz",
        "baselines/memory_r1_agent.py.bak",
        "prefix/evidence/scripts/eval_mem0_baseline.py.tar.gz",
        "prefix/evidence/baselines/memory_r1_agent.py.bak",
        "evidence/Memory-R1/result.json",
    }:
        with pytest.raises(ValueError, match="denied prior-system evidence"):
            reject_denied_evidence((denied,))
        with pytest.raises(ValueError, match="denied prior-system evidence"):
            resolve_candidate_id(denied)
    portable_blocked_paths = (
        "scripts/EVAL_M~1.PY",
        "prefix/scripts/eval_m~1.py",
        "PREFIX/SCRIPTS/EvAl_M~1.Py",
        "CONIN$.json",
        "conout$.txt",
        "evidence/memory_​r1/result.json",
        "evidence/memory_⁠r1/result.json",
        "evidence/memory_­r1/result.json",
        "evidence/memory_͏r1/result.json",
        "evidence/memory_️r1/result.json",
        "evidence/memory_⁥r1/result.json",
        "evidence/memory_￰r1/result.json",
        "evidence/memory_￸r1/result.json",
    )
    for alias in portable_blocked_paths:
        with pytest.raises(ValueError, match="portable canonical relative path"):
            reject_denied_evidence((alias,))

    with pytest.raises(ValueError, match="collection"):
        reject_denied_evidence("evidence/memory_r1/result.json")
    allowed = (
        "evidence/memory_r10/result.json",
        "evidence/memory_r10.json.bak",
        "evidence/memory_r1_rejection_analysis/result.json",
        "evidence/memory_r1_rejection_analysis.json",
        "evidence/memory_r1_rejection_analysis.json.bak",
        "evidence/unrelated_memory_r1_rejection_analysis/result.json",
        "evidence/ordinary.json",
        "scripts/ordinary-file.py",
        "evidence/ordinary_file.json",
        "evidence/café.json",
        "scripts/eval_mem0_baseline_rejection_analysis.py.bak",
        "prefix/scripts/eval_mem0_baseline_rejection_analysis.tar.gz",
    )
    assert reject_denied_evidence(allowed) == allowed

    with pytest.raises(ValueError, match="unknown external candidate"):
        resolve_candidate_id("mem0")
    with pytest.raises(ValueError, match="unknown external candidate"):
        resolve_candidate_id("Mem0 OSS")
