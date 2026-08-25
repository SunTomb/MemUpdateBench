# MemUpdateBench Post-Core Phase 1–2 Qualification Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate, source-bound Phase 1–2 qualification release that imports honest provider/open-runtime evidence, plans a bounded 8–16 capability smoke, derives scoped readiness, and publishes atomically without model loads, provider calls, credential reads, or benchmark execution.

**Architecture:** Add focused immutable-contract, validation, planning, decision, and release modules under `mub/vnext/post_core/`; keep the frozen Phase 0 v1 code and artifacts unchanged. The offline publisher consumes canonical redacted inputs and authenticated source hashes, while a separate smoke CLI accepts only an explicit authorization receipt and an out-of-process adapter protocol, so provider SDKs and credentials never enter the release package.

**Tech Stack:** Python 3.10, Pydantic v2 immutable models, canonical JSON/JSONL, SHA-256 source binding, pytest, argparse, subprocess JSONL adapter protocol, existing vNext Post-Core publication conventions.

---

## File structure

Create these focused files:

- `configs/vnext/post_core/qualification_release_v1.json` — frozen production configuration and authoritative source hashes.
- `mub/vnext/post_core/qualification_receipts_v1.py` — enums and immutable receipt/release contracts only.
- `mub/vnext/post_core/qualification_validation_v1.py` — recursive secret scan, provider attestation validation, runtime receipt validation, canonical JSONL loading.
- `mub/vnext/post_core/qualification_planning_v1.py` — deterministic eight-base/eight-escalation attempt plan and call IDs.
- `mub/vnext/post_core/qualification_decisions_v1.py` — derives scoped READY/BLOCKED/UNSUPPORTED decisions; callers never select READY.
- `mub/vnext/post_core/qualification_release_v1.py` — source snapshots, deterministic artifact build, verify, no-replace atomic publication.
- `scripts/vnext_prepare_post_core_qualification_release.py` — offline publication CLI.
- `scripts/vnext_run_post_core_capability_smoke.py` — authorization-gated, out-of-process adapter dispatcher.
- `tests/vnext/qualification_fixtures.py` — reusable synthetic source/receipt builders for tests only.
- Six targeted test files matching the components above.

Modify only:

- `mub/vnext/post_core/__init__.py` — export no symbols; retain package marker unless a test proves an export is required.
- `scripts/smoke_test.py` — add the new offline test files to the existing vNext smoke list.
- `WORKFLOW.md` — append an implementation/validation entry after all code and tests pass.

Do not modify `contracts_v1.py`, `qualification_v1.py`, `release_v1.py`, `release_v1.json`, `official_identity_evidence_v1.json`, Core data, Task 9–14 roots, or existing Phase 0 fixtures.

### Task 1: Qualification contract backbone and frozen config

**Files:**
- Create: `mub/vnext/post_core/qualification_receipts_v1.py`
- Create: `configs/vnext/post_core/qualification_release_v1.json`
- Create: `tests/vnext/test_post_core_qualification_receipts.py`

- [ ] **Step 1: Write failing strict-contract tests**

Create `tests/vnext/test_post_core_qualification_receipts.py` with imports that do not yet exist and tests for frozen models, null preservation, exact artifact order, decision scope, and non-self-hashing index:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from mub.vnext.post_core.contracts_v1 import canonical_hash
from mub.vnext.post_core.qualification_receipts_v1 import (
    QUALIFICATION_ARTIFACT_ORDER,
    CapabilityAttemptPlanV1,
    CapabilityBudgetV1,
    DecisionScope,
    QualificationArtifactIndexV1,
    QualificationStatus,
    SourceBindingV1,
)

SHA = "a" * 64


def test_source_binding_preserves_unmeasured_values_as_null() -> None:
    row = SourceBindingV1(
        source_id="qwen_load_receipt",
        evidence_class="load_only_receipt",
        sha256=SHA,
        required=True,
        byte_count=None,
    )
    assert row.byte_count is None
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        row.byte_count = 0


def test_attempt_plan_requires_zero_retries_and_non_executable_default() -> None:
    budget = CapabilityBudgetV1(
        max_calls=1,
        max_prompt_tokens=128,
        max_output_tokens=16,
        estimated_cost=Decimal("0"),
        hard_max_cost=Decimal("1"),
        price_version="qualification-v1",
        max_retries=0,
        timeout_seconds=60,
    )
    attempt = CapabilityAttemptPlanV1(
        release_id="memupdatebench.post-core.qualification.v1",
        registry_key="qwen35_9b_bf16",
        fixture_id="exact_ok_1",
        phase="BASE",
        repetition=1,
        prompt_sha256=SHA,
        parser_sha256=SHA,
        runtime_or_endpoint_class="transformers_offline",
        budget=budget,
        authorized=False,
        executable=False,
    )
    assert attempt.call_id == canonical_hash(attempt, exclude={"call_id"})
    with pytest.raises(ValidationError, match="authorization"):
        attempt.model_copy(update={"executable": True})


def test_decision_status_and_scope_are_distinct() -> None:
    assert QualificationStatus.READY.value == "READY"
    assert DecisionScope.CAPABILITY_SMOKE.value == "CAPABILITY_SMOKE"
    assert DecisionScope.BENCHMARK_ADMISSION.value == "BENCHMARK_ADMISSION"


def test_index_binds_exact_preceding_artifacts_and_not_itself() -> None:
    index = QualificationArtifactIndexV1(
        release_id="memupdatebench.post-core.qualification.v1",
        artifacts=tuple({"path": name, "sha256": SHA} for name in QUALIFICATION_ARTIFACT_ORDER),
    )
    assert tuple(item.path for item in index.artifacts) == QUALIFICATION_ARTIFACT_ORDER
    assert "qualification_artifact_index.json" not in QUALIFICATION_ARTIFACT_ORDER
    assert index.canonical_hash == canonical_hash(index, exclude={"canonical_hash"})
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
python -m pytest tests/vnext/test_post_core_qualification_receipts.py -q
```

Expected: collection fails with `ModuleNotFoundError: mub.vnext.post_core.qualification_receipts_v1`.

- [ ] **Step 3: Implement the contract module**

Create the immutable enums/models below. Keep validators in this module limited to local structural invariants; cross-row validation belongs to Task 2.

```python
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal, Mapping

from pydantic import Field, StrictBool, StrictInt, StrictStr, computed_field, model_validator

from mub.vnext.contracts.common import ImmutableContractModel
from mub.vnext.post_core.contracts_v1 import SHA256_PATTERN, canonical_hash

QUALIFICATION_ARTIFACT_ORDER = (
    "qualification_release_manifest.json",
    "source_bindings.json",
    "provider_capability_attestations.jsonl",
    "open_runtime_receipts.jsonl",
    "capability_smoke_plan.json",
    "qualification_decisions.json",
    "qualification_validation_receipt.json",
)
QUALIFICATION_INDEX_PATH = "qualification_artifact_index.json"
QUALIFICATION_ARTIFACTS = (*QUALIFICATION_ARTIFACT_ORDER, QUALIFICATION_INDEX_PATH)

class QualificationStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"

class DecisionScope(str, Enum):
    STORAGE_INPUT = "STORAGE_INPUT"
    SHORT_GENERATION_GATE = "SHORT_GENERATION_GATE"
    CAPABILITY_SMOKE = "CAPABILITY_SMOKE"
    BENCHMARK_ADMISSION = "BENCHMARK_ADMISSION"

class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"

class AttemptPhase(str, Enum):
    BASE = "BASE"
    ESCALATION = "ESCALATION"

class SourceBindingV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-source.v1"] = "memupdatebench.post-core.qualification-source.v1"
    source_id: StrictStr
    evidence_class: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    required: StrictBool
    byte_count: StrictInt | None = Field(default=None, ge=0)

class SourceBindingBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-sources.v1"] = "memupdatebench.post-core.qualification-sources.v1"
    release_id: StrictStr
    sources: tuple[SourceBindingV1, ...]

class CapabilityBudgetV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-budget.v1"] = "memupdatebench.post-core.qualification-budget.v1"
    max_calls: Literal[1] = 1
    max_prompt_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)
    estimated_cost: Decimal = Field(ge=Decimal("0"))
    hard_max_cost: Decimal = Field(ge=Decimal("0"))
    price_version: StrictStr
    max_retries: Literal[0] = 0
    timeout_seconds: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _finite_cost_bound(self):
        if not self.estimated_cost.is_finite() or not self.hard_max_cost.is_finite():
            raise ValueError("costs must be finite")
        if self.estimated_cost > self.hard_max_cost:
            raise ValueError("estimated cost exceeds hard maximum")
        return self

class CapabilityAttemptPlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-attempt-plan.v1"] = "memupdatebench.post-core.capability-attempt-plan.v1"
    release_id: StrictStr
    registry_key: StrictStr
    fixture_id: StrictStr
    phase: AttemptPhase
    repetition: StrictInt = Field(ge=1, le=2)
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    parser_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    runtime_or_endpoint_class: StrictStr
    budget: CapabilityBudgetV1
    authorized: StrictBool = False
    executable: StrictBool = False

    @computed_field(return_type=str)
    @property
    def call_id(self) -> str:
        return canonical_hash(self, exclude={"call_id"})

    @model_validator(mode="after")
    def _execution_boundary(self):
        if self.executable and not self.authorized:
            raise ValueError("executable capability attempt requires authorization")
        return self

class ArtifactBindingV1(ImmutableContractModel):
    path: StrictStr
    sha256: StrictStr = Field(pattern=SHA256_PATTERN)

class QualificationArtifactIndexV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-index.v1"] = "memupdatebench.post-core.qualification-index.v1"
    release_id: StrictStr
    artifacts: tuple[ArtifactBindingV1, ...]

    @model_validator(mode="after")
    def _exact_order(self):
        if tuple(row.path for row in self.artifacts) != QUALIFICATION_ARTIFACT_ORDER:
            raise ValueError("qualification index artifact order mismatch")
        return self

    @computed_field(return_type=str)
    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self, exclude={"canonical_hash"})
```

Define the remaining signatures in the same module exactly once so later tasks do not invent fields independently:

```python
class ProviderObservationV1(ImmutableContractModel):
    location: Literal["LOCAL", "TANG2"]
    observation_id: StrictStr
    provider_call_count: Literal[1] = 1
    retry_count: Literal[0] = 0
    http_status: Literal[200] = 200
    response_format: Literal["ANTHROPIC_MESSAGE_JSON", "SSE"]
    response_model: StrictStr
    exact_ok: Literal[True] = True
    stop_reason: Literal["end_turn"] | None = None
    usage_present: StrictBool | None = None

class ProviderCapabilityAttestationV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-attestation.v1"] = "memupdatebench.post-core.provider-attestation.v1"
    registry_key: StrictStr
    evidence_class: Literal["connectivity_interface_attestation"] = "connectivity_interface_attestation"
    request_name: StrictStr
    canonical_model_identity: StrictStr | None = None
    reasoning_tier: StrictStr | None = None
    identity_caveat: StrictStr | None = None
    observations: tuple[ProviderObservationV1, ...]
    provider_call_count: StrictInt = Field(ge=0)
    retry_count: Literal[0] = 0
    benchmark_generation_count: Literal[0] = 0
    raw_response_persisted: Literal[False] = False
    source_binding_ids: tuple[StrictStr, ...]

class ProviderSetupEventV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.provider-setup-event.v1"] = "memupdatebench.post-core.provider-setup-event.v1"
    event_id: StrictStr
    stage: Literal["PRE_PROVIDER_SETUP"] = "PRE_PROVIDER_SETUP"
    status: Literal["FAILED"] = "FAILED"
    provider_call_count: Literal[0] = 0
    reason_class: StrictStr
    source_binding_ids: tuple[StrictStr, ...]

class RuntimeManifestV1(ImmutableContractModel):
    engine: Literal["transformers", "llama.cpp"]
    engine_version: StrictStr
    engine_commit: StrictStr | None = None
    binary_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    python_version: StrictStr | None = None
    torch_version: StrictStr | None = None
    transformers_version: StrictStr | None = None
    accelerate_version: StrictStr | None = None
    cuda_version: StrictStr | None = None
    driver_version: StrictStr | None = None
    device_name: StrictStr
    context_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)
    build_options_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)

class OpenRuntimeReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.open-runtime-receipt.v1"] = "memupdatebench.post-core.open-runtime-receipt.v1"
    registry_key: StrictStr
    revision: StrictStr
    snapshot_tree_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    runtime: RuntimeManifestV1
    speculative_decoding: Literal["off"] = "off"
    load_status: GateStatus
    generation_status: GateStatus
    determinism_status: GateStatus
    unload_status: GateStatus
    prompt_fixture_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    parser_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    chat_template_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    output_projection_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    generated_token_count: StrictInt | None = Field(default=None, ge=0)
    peak_memory_bytes: StrictInt | None = Field(default=None, ge=0)
    blocked_reasons: tuple[StrictStr, ...] = ()
    source_binding_ids: tuple[StrictStr, ...]

class CapabilityFixtureV1(ImmutableContractModel):
    fixture_id: StrictStr
    category: Literal["EXACT_OUTPUT", "CHAT_TEMPLATE_PARSER"]
    prompt_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    parser_sha256: StrictStr = Field(pattern=SHA256_PATTERN)
    max_prompt_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)

class CapabilitySmokePlanV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-smoke-plan.v1"] = "memupdatebench.post-core.capability-smoke-plan.v1"
    release_id: StrictStr
    registry_keys: tuple[StrictStr, ...]
    base_attempts_per_role: Literal[8] = 8
    escalation_attempts_per_role: Literal[8] = 8
    max_retries: Literal[0] = 0
    authorized: Literal[False] = False
    attempts: tuple[CapabilityAttemptPlanV1, ...]

class CapabilityAttemptReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.capability-attempt-receipt.v1"] = "memupdatebench.post-core.capability-attempt-receipt.v1"
    call_id: StrictStr = Field(pattern=SHA256_PATTERN)
    registry_key: StrictStr
    status: GateStatus
    retry_count: Literal[0] = 0
    response_model: StrictStr | None = None
    response_format: Literal["ANTHROPIC_MESSAGE_JSON", "SSE", "LOCAL_TEXT"] | None = None
    stop_reason: StrictStr | None = None
    usage_present: StrictBool | None = None
    latency_ms: StrictInt | None = Field(default=None, ge=0)
    redacted_response_sha256: StrictStr | None = Field(default=None, pattern=SHA256_PATTERN)
    error_class: StrictStr | None = None

class QualificationDecisionV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-decision.v1"] = "memupdatebench.post-core.qualification-decision.v1"
    registry_key: StrictStr
    scope: DecisionScope
    status: QualificationStatus
    reasons: tuple[StrictStr, ...]
    evidence_binding_ids: tuple[StrictStr, ...]
    scientific_status: Literal["NOT_RUN"] = "NOT_RUN"

class QualificationDecisionBundleV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-decisions.v1"] = "memupdatebench.post-core.qualification-decisions.v1"
    release_id: StrictStr
    decisions: tuple[QualificationDecisionV1, ...]

class QualificationReleaseManifestV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-release.v1"] = "memupdatebench.post-core.qualification-release.v1"
    release_id: StrictStr
    base_commit: StrictStr
    artifact_order: tuple[StrictStr, ...]
    source_hashes: Mapping[StrictStr, StrictStr]

    @model_validator(mode="after")
    def _exact_artifacts(self):
        if self.artifact_order != QUALIFICATION_ARTIFACT_ORDER:
            raise ValueError("qualification manifest artifact order mismatch")
        return self

class QualificationValidationReceiptV1(ImmutableContractModel):
    schema_version: Literal["memupdatebench.post-core.qualification-validation.v1"] = "memupdatebench.post-core.qualification-validation.v1"
    release_id: StrictStr
    status: Literal["SUCCESS_WITH_BLOCKERS", "SUCCESS"]
    source_count: StrictInt = Field(gt=0)
    decision_counts: Mapping[StrictStr, StrictInt]
    provider_calls_during_publication: Literal[0] = 0
    model_loads_during_publication: Literal[0] = 0
    network_calls_during_publication: Literal[0] = 0
    credential_reads_during_publication: Literal[0] = 0
    benchmark_generations: Literal[0] = 0
```

All unmeasured numeric/string fields above are optional and default to `None`, never zero. Add `__all__` for only the public enums, constants, and models listed in this task.

- [ ] **Step 4: Add the canonical production config**

Write `configs/vnext/post_core/qualification_release_v1.json` as one-line canonical JSON with this exact logical content and key ordering supplied by `canonical_bytes`:

```json
{
  "base_attempts_per_role": 8,
  "base_commit": "a56857431023d2af1a392c75c5575316a916c174",
  "escalation_attempts_per_role": 8,
  "max_retries": 0,
  "publisher_network_allowed": false,
  "registry_keys": [
    "qwen35_9b_bf16",
    "meta_muse_glimmer_30b_int4",
    "meta_muse_glimmer_30b_bf16",
    "claude_sonnet_4_6",
    "claude_opus_4_8",
    "gemini_3_6_flash",
    "grok_4_5",
    "gpt_5_5"
  ],
  "release_id": "memupdatebench.post-core.qualification.v1",
  "required_source_sha256": {
    "core_manifest": "dd5ea033fd1bb7353f4c7f443c6a1e14ed44fb9e8641f8e05838b4147d3ec13b",
    "handoff_source": "4c1424bd2da72e9ed1042f091256fc55484c2f04cfdc0f6a0b4cf731eb5519a2",
    "identity_evidence": "9e3780ed3d4303bda7bbd27865df89fcb384041da64af56107c8c5b7abf0a4f0",
    "open_snapshot_audit_receipt": "0b146bd8dc04e3343d899801f4746bee0ae69635f1ace3f4c92ada8f32819940",
    "open_snapshot_closure_receipt": "77a69e02a8b092b7e1bf5e89ff9a5f69b449c89a1c2cd319f9c48edd3e2f4645",
    "phase0_index": "e0b08cf0752798b55388c16f176af88a7a6a25a6facf29d6fa4100348ac199fd",
    "qwen_load_receipt": "fd4e47d75d86efdbe9add3cc469017b9aef23bb05bc4d03b74877bfbe289f6b7",
    "task14_index": "2ccc737dffb04bc377b123edee2ac1ca04ed338651d0bd19f9c112430bc04035",
    "workflow_source": "b2dc80c6dc30b74aff597cdeb83044056fb24efe7a260b39a004aa5d2f4905cb"
  },
  "schema_version": "memupdatebench.post-core.qualification-config.v1",
  "scientific_execution_allowed": false
}
```

Generate the committed bytes from the literal mapping in the test using:

```python
CONFIG.write_bytes(canonical_bytes(expected_config_payload()))
assert CONFIG.read_bytes() == canonical_bytes(expected_config_payload())
```

The test must also assert that `required_source_sha256` has exactly the nine keys shown above; no caller-selected source can be omitted or added.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/vnext/test_post_core_qualification_receipts.py -q
python -m pytest tests/vnext/test_post_core_contracts.py tests/vnext/test_post_core_identity_evidence.py -q
git diff --check
```

Expected: all tests pass and the Phase 0 regression files remain unchanged.

Commit:

```bash
git add configs/vnext/post_core/qualification_release_v1.json mub/vnext/post_core/qualification_receipts_v1.py tests/vnext/test_post_core_qualification_receipts.py
git commit -m "Add qualification release contracts"
```

### Task 2: Secret-free provider attestations and canonical JSONL

**Files:**
- Create: `mub/vnext/post_core/qualification_validation_v1.py`
- Create: `tests/vnext/qualification_fixtures.py`
- Create: `tests/vnext/test_post_core_qualification_validation.py`

- [ ] **Step 1: Write failing provider/security tests**

Create fixture builders for five aggregate rows. Encode GPT as four one-call observations (`LOCAL_INITIAL_SSE`, `LOCAL_EXPLICIT_FALSE_SSE`, `TANG2_PREFIX_SSE`, `TANG2_POSTFIX_JSON`); each other route has one local and one Tang-2 observation. Then add these tests:

```python
def test_prior_provider_attestations_account_exactly_twelve_calls() -> None:
    rows = provider_attestations()
    validated = validate_provider_attestations_v1(rows)
    counts = {row.registry_key: row.provider_call_count for row in validated}
    assert counts == {
        "claude_sonnet_4_6": 2,
        "claude_opus_4_8": 2,
        "gemini_3_6_flash": 2,
        "grok_4_5": 2,
        "gpt_5_5": 4,
    }
    assert sum(counts.values()) == 12
    assert sum(row.retry_count for row in validated) == 0
    assert sum(row.benchmark_generation_count for row in validated) == 0


def test_gemini_requires_canonical_request_and_low_tier() -> None:
    row = next(row for row in provider_attestations() if row.registry_key == "gemini_3_6_flash")
    with pytest.raises(ValueError, match="Gemini provenance"):
        validate_provider_attestations_v1((row.model_copy(update={"reasoning_tier": None}),))


def test_gpt_preserves_sse_before_json_after_fix() -> None:
    gpt = next(row for row in validate_provider_attestations_v1(provider_attestations()) if row.registry_key == "gpt_5_5")
    assert tuple(obs.response_format for obs in gpt.observations) == (
        "SSE", "SSE", "SSE", "ANTHROPIC_MESSAGE_JSON"
    )


def test_failed_ssh_setup_is_not_a_provider_call() -> None:
    event = failed_ssh_setup_event()
    assert event.provider_call_count == 0
    assert event.stage == "PRE_PROVIDER_SETUP"

@pytest.mark.parametrize("payload", [
    {"Authorization": "Bearer synthetic-secret-value"},
    {"endpoint": "https://user:pass@example.invalid/v1/messages"},
    {"endpoint": "https://example.invalid/v1/messages?token=synthetic"},
    {"raw": "-----BEGIN PRIVATE KEY-----"},
])
def test_qualification_secret_scan_rejects_credentials(payload) -> None:
    with pytest.raises(ValueError, match="secret|credential|endpoint"):
        validate_qualification_secret_free(payload)
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/vnext/test_post_core_qualification_validation.py -q
```

Expected: collection fails because `qualification_validation_v1` and fixture builders do not exist.

- [ ] **Step 3: Implement validation and canonical JSONL loading**

Implement:

```python
def validate_qualification_secret_free(value: Any) -> None:
    validate_secret_free(value, read_environment=False)
    _scan_endpoint_values(value)


def load_canonical_jsonl_v1(path: Path, model_type, *, label: str):
    raw = _read_regular_single_link(path, label)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be nonempty LF-terminated JSONL")
    rows = []
    for line in raw[:-1].split(b"\n"):
        if not line:
            raise ValueError(f"{label} contains an empty row")
        payload = json.loads(line)
        validate_qualification_secret_free(payload)
        row = model_type.model_validate(payload)
        if canonical_bytes(row) != line:
            raise ValueError(f"{label} row is not canonical")
        rows.append(row)
    return tuple(rows), raw
```

Use `urllib.parse.urlsplit` to reject userinfo and secret-like query keys in fields named `endpoint`, `endpoint_url`, or `source_url`. Never inspect environment variables.

Implement `validate_provider_attestations_v1` with exact key order, exact counts `(2,2,2,2,4)`, zero retries, zero benchmark generations, all `exact_ok=True`, all prior rows `raw_response_persisted=False`, mandatory Gemini three-field mapping, and mandatory GPT format sequence. Return an immutable tuple only after a final secret scan.

- [ ] **Step 4: Run validation and Phase 0 secret regressions**

Run:

```bash
python -m pytest tests/vnext/test_post_core_qualification_validation.py -q
python -m pytest tests/vnext/test_post_core_provenance.py tests/vnext/test_external_artifact_security.py -q
```

Expected: all pass; no environment reads or network calls occur.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_validation_v1.py tests/vnext/qualification_fixtures.py tests/vnext/test_post_core_qualification_validation.py
git commit -m "Validate redacted qualification evidence"
```

### Task 3: Open-model runtime receipt validation

**Files:**
- Modify: `mub/vnext/post_core/qualification_validation_v1.py`
- Modify: `tests/vnext/qualification_fixtures.py`
- Create: `tests/vnext/test_post_core_qualification_runtime.py`

- [ ] **Step 1: Write failing runtime-boundary tests**

Add synthetic receipts representing: Qwen load PASS/generation NOT_RUN/unload PASS; Muse GGUF all NOT_RUN with speculative off; Muse BF16 BLOCKED with null measurements.

```python
def test_qwen_load_only_receipt_cannot_pass_generation_gate() -> None:
    rows = validate_runtime_receipts_v1(open_runtime_receipts())
    qwen = next(row for row in rows if row.registry_key == "qwen35_9b_bf16")
    assert qwen.load_status is GateStatus.PASS
    assert qwen.generation_status is GateStatus.NOT_RUN
    assert qwen.determinism_status is GateStatus.NOT_RUN
    assert qwen.unload_status is GateStatus.PASS


def test_muse_gguf_requires_frozen_llama_runtime_and_speculative_off() -> None:
    muse = next(row for row in open_runtime_receipts() if row.registry_key == "meta_muse_glimmer_30b_int4")
    with pytest.raises(ValueError, match="llama.cpp|speculative"):
        validate_runtime_receipts_v1((muse.model_copy(update={"speculative_decoding": "on"}),))


def test_muse_bf16_resource_absence_is_blocked_not_unsupported_or_zero() -> None:
    bf16 = next(row for row in validate_runtime_receipts_v1(open_runtime_receipts()) if row.registry_key == "meta_muse_glimmer_30b_bf16")
    assert bf16.load_status is GateStatus.BLOCKED
    assert bf16.peak_memory_bytes is None
    assert bf16.generated_token_count is None
    assert "resource" in " ".join(bf16.blocked_reasons).lower()
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_runtime.py -q
```

Expected: import or assertion failure because runtime validation is absent.

- [ ] **Step 3: Implement `validate_runtime_receipts_v1`**

Validate exact registry order for the three open roles, source revision/tree hashes, runtime-engine requirements, and gate consistency:

```python
if row.generation_status is GateStatus.PASS:
    required = (
        row.prompt_fixture_sha256,
        row.parser_sha256,
        row.chat_template_sha256,
        row.output_projection_sha256,
        row.generated_token_count,
    )
    if any(value is None for value in required):
        raise ValueError("generation PASS requires complete bounded generation evidence")
if row.determinism_status is GateStatus.PASS and row.generation_status is not GateStatus.PASS:
    raise ValueError("determinism PASS requires generation PASS")
if row.registry_key == "meta_muse_glimmer_30b_int4":
    if row.runtime.engine != "llama.cpp" or not row.runtime.engine_commit:
        raise ValueError("Muse GGUF requires a frozen llama.cpp commit")
    if row.speculative_decoding != "off":
        raise ValueError("Muse GGUF qualification requires speculative decoding off")
```

For NOT_RUN/BLOCKED/UNSUPPORTED gates, reject fabricated measurement values that would imply an executed generation. Preserve nulls.

- [ ] **Step 4: Run runtime and identity tests**

```bash
python -m pytest tests/vnext/test_post_core_qualification_runtime.py -q
python -m pytest tests/vnext/test_post_core_identity_evidence.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_validation_v1.py tests/vnext/qualification_fixtures.py tests/vnext/test_post_core_qualification_runtime.py
git commit -m "Validate open model qualification receipts"
```

### Task 4: Deterministic 8–16 capability-smoke planning

**Files:**
- Create: `mub/vnext/post_core/qualification_planning_v1.py`
- Create: `tests/vnext/test_post_core_qualification_planning.py`

- [ ] **Step 1: Write failing plan tests**

```python
def test_plan_has_eight_base_and_eight_escalation_attempts_per_role() -> None:
    plan = build_capability_smoke_plan_v1(qualification_config(), capability_fixtures())
    for key in plan.registry_keys:
        rows = [row for row in plan.attempts if row.registry_key == key]
        assert sum(row.phase is AttemptPhase.BASE for row in rows) == 8
        assert sum(row.phase is AttemptPhase.ESCALATION for row in rows) == 8
    assert all(row.authorized is False and row.executable is False for row in plan.attempts)
    assert all(row.budget.max_retries == 0 for row in plan.attempts)


def test_plan_contains_no_benchmark_metrics_or_task_payloads() -> None:
    plan = build_capability_smoke_plan_v1(qualification_config(), capability_fixtures())
    payload = plan.model_dump(mode="json")
    forbidden = {"em", "f1", "state_accuracy", "stale_copied", "task_id", "gold"}
    assert not forbidden.intersection(json.dumps(payload).lower().replace('"', '').split("\n"))


def test_call_ids_are_unique_and_deterministic() -> None:
    left = build_capability_smoke_plan_v1(qualification_config(), capability_fixtures())
    right = build_capability_smoke_plan_v1(qualification_config(), capability_fixtures())
    assert left == right
    assert len({row.call_id for row in left.attempts}) == len(left.attempts)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_planning.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement fixture expansion and stable IDs**

Define four fixtures (`exact_ok_1`, `exact_ok_2`, `parser_city_1`, `parser_city_2`) with canonical prompt/parser hashes and category `EXACT_OUTPUT` or `CHAT_TEMPLATE_PARSER`. Expand each fixture over repetitions 1–2 for BASE, then generate a distinct ESCALATION phase with repetitions 1–2 and the same fixture set. Do not duplicate by copying BASE call IDs; phase is part of the computed ID.

```python
def build_capability_smoke_plan_v1(config, fixtures):
    if len(fixtures) != 4 or len({row.fixture_id for row in fixtures}) != 4:
        raise ValueError("capability smoke requires exactly four unique fixtures")
    attempts = tuple(
        _attempt(config, key, fixture, phase, repetition)
        for key in config.registry_keys
        for phase in (AttemptPhase.BASE, AttemptPhase.ESCALATION)
        for fixture in fixtures
        for repetition in (1, 2)
    )
    return CapabilitySmokePlanV1(
        release_id=config.release_id,
        registry_keys=config.registry_keys,
        base_attempts_per_role=8,
        escalation_attempts_per_role=8,
        max_retries=0,
        authorized=False,
        attempts=attempts,
    )
```

The planner does not accept model output, authorization, endpoint overrides, or caller-provided call IDs.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/vnext/test_post_core_qualification_planning.py -q
python -m pytest tests/vnext/test_post_core_planning.py -q
```

Expected: all pass, including unchanged Phase 0 320-call intent planning.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_planning_v1.py tests/vnext/test_post_core_qualification_planning.py
git commit -m "Plan bounded capability smoke attempts"
```

### Task 5: Scope-aware readiness decision engine

**Files:**
- Create: `mub/vnext/post_core/qualification_decisions_v1.py`
- Create: `tests/vnext/test_post_core_qualification_decisions.py`

- [ ] **Step 1: Write failing decision tests**

```python
def test_capability_ready_does_not_imply_benchmark_ready() -> None:
    decisions = derive_qualification_decisions_v1(identity_bundle(), provider_attestations(), open_runtime_receipts())
    by_key_scope = {(row.registry_key, row.scope): row for row in decisions}
    assert by_key_scope[("claude_sonnet_4_6", DecisionScope.CAPABILITY_SMOKE)].status is QualificationStatus.READY
    assert by_key_scope[("claude_sonnet_4_6", DecisionScope.BENCHMARK_ADMISSION)].status is QualificationStatus.BLOCKED


def test_grok_and_gpt_identity_caveats_block_benchmark_admission() -> None:
    rows = derive_qualification_decisions_v1(identity_bundle(), provider_attestations(), open_runtime_receipts())
    selected = {(row.registry_key, row.scope): row for row in rows}
    for key in ("grok_4_5", "gpt_5_5"):
        assert selected[(key, DecisionScope.CAPABILITY_SMOKE)].status is QualificationStatus.READY
        assert selected[(key, DecisionScope.BENCHMARK_ADMISSION)].status is QualificationStatus.BLOCKED
        assert "identity" in " ".join(selected[(key, DecisionScope.BENCHMARK_ADMISSION)].reasons).lower()


def test_qwen_load_only_is_blocked_for_short_generation_and_smoke() -> None:
    rows = derive_qualification_decisions_v1(identity_bundle(), provider_attestations(), open_runtime_receipts())
    selected = {(row.registry_key, row.scope): row for row in rows}
    assert selected[("qwen35_9b_bf16", DecisionScope.STORAGE_INPUT)].status is QualificationStatus.READY
    assert selected[("qwen35_9b_bf16", DecisionScope.SHORT_GENERATION_GATE)].status is QualificationStatus.BLOCKED
    assert selected[("qwen35_9b_bf16", DecisionScope.CAPABILITY_SMOKE)].status is QualificationStatus.BLOCKED
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_decisions.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement derived-only decisions**

Expose one public function. It receives validated typed rows, never a requested status:

```python
def derive_qualification_decisions_v1(identity, provider_rows, runtime_rows):
    provider = {row.registry_key: row for row in validate_provider_attestations_v1(provider_rows)}
    runtime = {row.registry_key: row for row in validate_runtime_receipts_v1(runtime_rows)}
    decisions = []
    for record in identity.records:
        decisions.extend(_derive_candidate(record, provider.get(record.registry_key), runtime.get(record.registry_key)))
    return tuple(decisions)
```

Rules:

- authenticated snapshot receipt makes `STORAGE_INPUT=READY` for the three open roles;
- Qwen/Muse `SHORT_GENERATION_GATE=READY` only when load/generation/determinism/unload are all PASS;
- open `CAPABILITY_SMOKE=READY` only after short-generation gate PASS;
- closed `CAPABILITY_SMOKE=READY` when provider attestation passes exact counts/format/identity-field checks;
- all `BENCHMARK_ADMISSION=BLOCKED` in this operational release because scientific execution is NOT_RUN;
- missing evidence/resources produce BLOCKED;
- only a demonstrated backend incompatibility produces UNSUPPORTED.

Sort decisions by frozen registry order, then scope order.

- [ ] **Step 4: Run decision and identity tests**

```bash
python -m pytest tests/vnext/test_post_core_qualification_decisions.py -q
python -m pytest tests/vnext/test_post_core_identity_evidence.py -q
```

Expected: all pass and GPT/Grok frozen identity states remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_decisions_v1.py tests/vnext/test_post_core_qualification_decisions.py
git commit -m "Derive scoped qualification readiness"
```

### Task 6: Deterministic source-bound release builder

**Files:**
- Create: `mub/vnext/post_core/qualification_release_v1.py`
- Create: `tests/vnext/test_post_core_qualification_release.py`

- [ ] **Step 1: Write failing build/verification tests**

```python
def test_build_has_exact_eight_artifacts_and_zero_execution(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    publication = build_qualification_release_v1(**inputs)
    assert tuple(publication.artifact_bytes) == QUALIFICATION_ARTIFACTS
    receipt = QualificationValidationReceiptV1.model_validate_json(
        publication.artifact_bytes["qualification_validation_receipt.json"]
    )
    assert receipt.provider_calls_during_publication == 0
    assert receipt.model_loads_during_publication == 0
    assert receipt.network_calls_during_publication == 0
    assert receipt.credential_reads_during_publication == 0
    assert receipt.benchmark_generations == 0


def test_build_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    left = build_qualification_release_v1(**inputs)
    right = build_qualification_release_v1(**inputs)
    assert left.artifact_bytes == right.artifact_bytes
    manifest = QualificationReleaseManifestV1.model_validate_json(
        left.artifact_bytes["qualification_release_manifest.json"]
    )
    assert manifest.base_commit == "a56857431023d2af1a392c75c5575316a916c174"
    assert manifest.source_hashes["phase0_index"] == EXPECTED_PHASE0_INDEX_SHA256


def test_source_byte_substitution_is_rejected(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    inputs["workflow_source_path"].write_bytes(b"substituted")
    with pytest.raises(StaleQualificationSourceError):
        build_qualification_release_v1(**inputs)
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement config/source snapshots and artifact build**

Define `QualificationReleaseConfigV1`, `_SourceSnapshot`, `QualificationPublicationV1`, and typed errors. Load every source as a regular single-link file, capture `(st_dev, st_ino, byte_count, sha256, raw)`, compare against the frozen config, and revalidate before returning.

Build artifacts in this order:

```python
artifacts = {
    "qualification_release_manifest.json": canonical_bytes(manifest),
    "source_bindings.json": canonical_bytes(source_bundle),
    "provider_capability_attestations.jsonl": provider_raw,
    "open_runtime_receipts.jsonl": runtime_raw,
    "capability_smoke_plan.json": canonical_bytes(plan),
    "qualification_decisions.json": canonical_bytes(decision_bundle),
    "qualification_validation_receipt.json": canonical_bytes(validation_receipt),
}
index = QualificationArtifactIndexV1(
    release_id=config.release_id,
    artifacts=tuple({"path": name, "sha256": sha256(artifacts[name])} for name in QUALIFICATION_ARTIFACT_ORDER),
)
artifacts[QUALIFICATION_INDEX_PATH] = canonical_bytes(index, exclude={"canonical_hash"})
```

Run `validate_qualification_secret_free` on every decoded artifact before exposing bytes. The builder imports no provider SDK and accepts no authorization or endpoint override.

- [ ] **Step 4: Implement `verify_qualification_release_v1` for an existing root**

Require exactly eight regular single-link artifacts, parse every model, validate canonical bytes/JSONL, recompute index hashes, rebuild expected bytes from the same source snapshots, compare exact bytes, and revalidate all sources after comparison.

- [ ] **Step 5: Run build tests and commit**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release.py -q
python -m pytest tests/vnext/test_post_core_release_cli.py::test_release_has_exact_seven_artifacts_and_zero_calls -q
git diff --check
```

Expected: all pass; Phase 0 still publishes exactly its original seven artifacts.

```bash
git add mub/vnext/post_core/qualification_release_v1.py tests/vnext/test_post_core_qualification_release.py
git commit -m "Build source-bound qualification releases"
```

### Task 7: No-replace atomic publication and race defenses

**Files:**
- Modify: `mub/vnext/post_core/qualification_release_v1.py`
- Modify: `tests/vnext/test_post_core_qualification_release.py`

- [ ] **Step 1: Add failing no-clobber/race/path tests**

```python
def test_publish_reopens_and_refuses_existing_root(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    output = tmp_path / "published"
    result = publish_qualification_release_v1(output_root=output, **inputs)
    assert result.output_root == output.resolve()
    reopened = verify_qualification_release_v1(root=output, **inputs)
    assert reopened.index_sha256 == result.index_sha256
    with pytest.raises(FileExistsError):
        publish_qualification_release_v1(output_root=output, **inputs)


def test_output_cannot_overlap_any_source_or_frozen_root(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    with pytest.raises(UnsafeQualificationPathError):
        publish_qualification_release_v1(output_root=inputs["phase0_index_path"].parent, **inputs)


def test_source_mutation_before_commit_leaves_no_final_root(tmp_path: Path) -> None:
    inputs = qualification_sources(tmp_path)
    output = tmp_path / "published"
    def mutate():
        inputs["provider_attestations_path"].write_bytes(b"changed")
    with pytest.raises(StaleQualificationSourceError):
        publish_qualification_release_v1(output_root=output, before_commit=mutate, **inputs)
    assert not output.exists()
```

Also port Phase 0's staging-member mutation and post-rename verification tests with qualification artifact names.

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release.py -q
```

Expected: atomic-publication tests fail because `publish_qualification_release_v1` is absent.

- [ ] **Step 3: Implement qualification-specific atomic publication**

Use a same-filesystem sibling stage named `.mub-post-core-qualification-stage-<uuid>`, exclusive `xb` file writes, file fsync, directory fsync, owned-member identity/hash capture, final source revalidation, Windows `MoveFileExW(..., MOVEFILE_WRITE_THROUGH)` without replace, and Linux `renameat2(..., RENAME_NOREPLACE)`/syscall fallback. Do not use copy/replace or `os.replace`.

The cleanup routine deletes a stage only when its directory identity, member set, member identities, byte counts, and hashes still match publisher ownership. Preserve tampered stages for quarantine.

- [ ] **Step 4: Run publication and Phase 0 race tests**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release.py -q
python -m pytest tests/vnext/test_post_core_release_cli.py -q
```

Expected: all pass on the current platform, with only the repository's pre-existing Windows symlink-permission skip if applicable.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_release_v1.py tests/vnext/test_post_core_qualification_release.py
git commit -m "Publish qualification releases atomically"
```

### Task 8: Offline preparation CLI

**Files:**
- Create: `scripts/vnext_prepare_post_core_qualification_release.py`
- Create: `tests/vnext/test_post_core_qualification_release_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test `--help` for forbidden network/credential flags, successful fixture publication, existing-root failure, stale-source exit, and secret-free stderr:

```python
def test_prepare_cli_has_no_network_or_credential_surface() -> None:
    run = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert run.returncode == 0
    lowered = run.stdout.lower()
    for forbidden in ("--api-key", "--token", "--authorization", "--endpoint", "--allow-network", "--provider"):
        assert forbidden not in lowered


def test_prepare_cli_publishes_fixture_release(tmp_path: Path) -> None:
    args = qualification_cli_args(tmp_path)
    run = subprocess.run([sys.executable, str(SCRIPT), *args, "--execute"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary["status"] == "SUCCESS_WITH_BLOCKERS"
    assert summary["provider_calls_during_publication"] == 0
    assert summary["model_loads_during_publication"] == 0
    assert summary["benchmark_generations"] == 0
```

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release_cli.py -q
```

Expected: script missing.

- [ ] **Step 3: Implement the preparation CLI**

Required arguments:

```text
--config
--core-manifest
--task14-index
--phase0-index
--identity-evidence
--workflow-source
--handoff-source
--open-snapshot-closure-receipt
--open-snapshot-audit-receipt
--qwen-load-receipt
--provider-attestations
--runtime-receipts
--output-root
--execute
```

`--execute` means validate/build/publish metadata only. It does not permit network/model execution. Map typed exceptions to stable codes: success 0, blocked 10, usage 11, stale source 12, publication 13, untrusted runtime 14. Print one canonical JSON summary and never arbitrary exception text.

- [ ] **Step 4: Run CLI tests**

```bash
python -m pytest tests/vnext/test_post_core_qualification_release_cli.py -q
python scripts/vnext_prepare_post_core_qualification_release.py --help
```

Expected: tests pass; help exposes no network, credential, provider, or endpoint option.

- [ ] **Step 5: Commit**

```bash
git add scripts/vnext_prepare_post_core_qualification_release.py tests/vnext/test_post_core_qualification_release_cli.py
git commit -m "Add offline qualification release CLI"
```

### Task 9: Authorization-gated external adapter smoke CLI

**Files:**
- Modify: `mub/vnext/post_core/qualification_receipts_v1.py`
- Modify: `mub/vnext/post_core/qualification_validation_v1.py`
- Create: `scripts/vnext_run_post_core_capability_smoke.py`
- Create: `tests/vnext/test_post_core_capability_smoke_cli.py`

- [ ] **Step 1: Write failing authorization and fake-adapter tests**

Add `ExecutionAuthorizationV1` with release ID, plan SHA-256, scope `CAPABILITY_SMOKE`, authorized call IDs, max calls, issued-at string, issuer, and signature/attestation SHA-256. It contains no credential fields.

```python
def test_smoke_cli_refuses_without_authorization(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan_path(tmp_path)), "--adapter-executable", str(fake_adapter(tmp_path)), "--execute"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 10
    assert "authorization" in run.stderr.lower()


def test_smoke_cli_runs_only_authorized_base_calls_through_fake_adapter(tmp_path: Path) -> None:
    plan = plan_path(tmp_path)
    auth = authorization_path(tmp_path, plan, selected_role="qwen35_9b_bf16", phase="BASE")
    output = tmp_path / "attempts.jsonl"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan), "--authorization-receipt", str(auth), "--adapter-executable", str(fake_adapter(tmp_path)), "--output", str(output), "--execute"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    rows, _ = load_canonical_jsonl_v1(output, CapabilityAttemptReceiptV1, label="attempt receipts")
    assert len(rows) == 8
    assert all(row.registry_key == "qwen35_9b_bf16" for row in rows)
    assert all(row.retry_count == 0 for row in rows)
```

The fake adapter reads one canonical attempt JSON object per stdin line and writes one canonical receipt per stdout line. It never imports a provider SDK or reads environment credentials.

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/vnext/test_post_core_capability_smoke_cli.py -q
```

Expected: script and authorization contract are absent.

- [ ] **Step 3: Implement authorization validation and adapter protocol**

Validate that authorization source is a regular single-link canonical JSON file; its plan hash matches exact plan bytes; every call ID exists in the plan; selected calls do not exceed `max_calls`; scope is exactly `CAPABILITY_SMOKE`; and every selected call is BASE unless authorization explicitly lists ESCALATION call IDs.

Launch only the exact `--adapter-executable` path plus fixed `--jsonl-protocol-v1`; do not accept arbitrary adapter arguments, shell strings, environment overrides, provider IDs, endpoints, or credentials. Use `subprocess.run([...], shell=False, input=canonical_jsonl, capture_output=True, timeout=...)` with a minimal inherited environment policy documented in code. Validate every returned receipt, exact call-ID coverage, zero retries, and secret freedom before writing the absent output file with `xb`.

The CLI itself performs no provider-specific logic. Real adapters remain separate, untracked execution components until independently authorized.

- [ ] **Step 4: Run CLI and security tests**

```bash
python -m pytest tests/vnext/test_post_core_capability_smoke_cli.py -q
python -m pytest tests/vnext/test_post_core_qualification_validation.py -q
```

Expected: all pass using only the fake adapter.

- [ ] **Step 5: Commit**

```bash
git add mub/vnext/post_core/qualification_receipts_v1.py mub/vnext/post_core/qualification_validation_v1.py scripts/vnext_run_post_core_capability_smoke.py tests/vnext/test_post_core_capability_smoke_cli.py
git commit -m "Gate capability smoke adapter execution"
```

### Task 10: Full offline regression, smoke integration, and workflow record

**Files:**
- Modify: `scripts/smoke_test.py`
- Modify: `WORKFLOW.md`
- Test: all files created in Tasks 1–9

- [ ] **Step 1: Add the qualification suite to the project smoke test**

Follow the existing `scripts/smoke_test.py` test-list pattern and add exactly:

```text
tests/vnext/test_post_core_qualification_receipts.py
tests/vnext/test_post_core_qualification_validation.py
tests/vnext/test_post_core_qualification_runtime.py
tests/vnext/test_post_core_qualification_planning.py
tests/vnext/test_post_core_qualification_decisions.py
tests/vnext/test_post_core_qualification_release.py
tests/vnext/test_post_core_qualification_release_cli.py
tests/vnext/test_post_core_capability_smoke_cli.py
```

Do not add a real adapter invocation.

- [ ] **Step 2: Run compile and targeted qualification tests**

```bash
python -m py_compile \
  mub/vnext/post_core/qualification_receipts_v1.py \
  mub/vnext/post_core/qualification_validation_v1.py \
  mub/vnext/post_core/qualification_planning_v1.py \
  mub/vnext/post_core/qualification_decisions_v1.py \
  mub/vnext/post_core/qualification_release_v1.py \
  scripts/vnext_prepare_post_core_qualification_release.py \
  scripts/vnext_run_post_core_capability_smoke.py
python -m pytest \
  tests/vnext/test_post_core_qualification_receipts.py \
  tests/vnext/test_post_core_qualification_validation.py \
  tests/vnext/test_post_core_qualification_runtime.py \
  tests/vnext/test_post_core_qualification_planning.py \
  tests/vnext/test_post_core_qualification_decisions.py \
  tests/vnext/test_post_core_qualification_release.py \
  tests/vnext/test_post_core_qualification_release_cli.py \
  tests/vnext/test_post_core_capability_smoke_cli.py -q
```

Expected: compile succeeds; all targeted tests pass with zero live calls.

- [ ] **Step 3: Run Phase 0/Post-Core regressions and the project smoke test**

```bash
python -m pytest tests/vnext/test_post_core_contracts.py tests/vnext/test_post_core_planning.py tests/vnext/test_post_core_qualification.py tests/vnext/test_post_core_release_cli.py tests/vnext/test_post_core_identity_evidence.py tests/vnext/test_post_core_provenance.py -q
python scripts/smoke_test.py
```

Expected: all pass except the already-documented Windows symlink-permission skip if the host cannot create symlinks. Confirm summaries report zero model loads, zero provider calls, zero network calls, zero benchmark generations, and zero credential reads during this implementation validation.

- [ ] **Step 4: Append the implementation result to `WORKFLOW.md`**

Add one final section named:

```markdown
## Post-Core Phase 1–2 qualification release implementation (2026-08-24)
```

Record motivation, files changed, exact commands, test counts, any Windows skip, and these conclusions:

- release implementation is offline and source-bound;
- prior 12 provider calls remain aggregate connectivity/interface attestations;
- no model/provider execution occurred during implementation;
- open short-generation and live 8–16 smoke remain NOT_RUN/BLOCKED until explicit authorization;
- 320 canary and confirmatory hard subset remain unauthorized.

Do not include cluster command transcripts, GPU scheduling detail, endpoint URLs, raw responses, or credential information.

- [ ] **Step 5: Inspect final diff and commit**

```bash
git status --short
git diff --check
git diff --stat
git diff -- configs/vnext/post_core/release_v1.json configs/vnext/post_core/official_identity_evidence_v1.json mub/vnext/post_core/qualification_v1.py mub/vnext/post_core/release_v1.py
```

Expected: the final command prints no diff for frozen Phase 0 inputs/code.

Commit:

```bash
git add scripts/smoke_test.py WORKFLOW.md
git commit -m "Document qualification release validation"
```

- [ ] **Step 6: Final verification checkpoint**

```bash
git status --short --branch
git log --oneline --decorate -12
```

Expected: clean worktree on `worktree-post-core-phase12-qualification`; commits correspond to Tasks 1–10; no push, merge, provider call, model load, canary, or benchmark run has occurred.
