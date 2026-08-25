from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from mub.vnext.post_core.contracts_v1 import canonical_bytes, canonical_hash
from pydantic import ValidationError

from mub.vnext.post_core.qualification_receipts_v1 import (
    DecisionScope,
    ExecutionAuthorizationV1,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _authorization(**overrides: object) -> ExecutionAuthorizationV1:
    payload: dict[str, object] = {
        "release_id": "release",
        "plan_sha256": HASH_A,
        "scope": DecisionScope.CAPABILITY_SMOKE,
        "authorized_call_ids": (HASH_B,),
        "max_calls": 1,
        "issued_at": "2026-08-24T00:00:00Z",
        "issuer": "offline-test",
        "authorization_attestation_sha256": HASH_A,
        "adapter_sha256": HASH_A,
        "escalation_anomaly_receipt_sha256": None,
    }
    payload.update(overrides)
    return ExecutionAuthorizationV1(**payload)


def test_execution_authorization_contract_is_strict_and_immutable() -> None:
    authorization = _authorization()

    assert authorization.schema_version == "qualification-execution-authorization.v1"
    assert authorization.scope is DecisionScope.CAPABILITY_SMOKE
    with pytest.raises(ValidationError):
        authorization.issuer = "changed"
    with pytest.raises(ValidationError):
        _authorization(max_calls=True)
    with pytest.raises(ValidationError):
        _authorization(authorized_call_ids=(HASH_B, HASH_B))



def test_capability_anomaly_receipt_contract_is_immutable_and_rejects_invalid_shape() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAnomalyReceiptV1

    receipt = CapabilityAnomalyReceiptV1(
        release_id="release",
        plan_sha256=HASH_A,
        base_receipts_sha256=HASH_B,
        base_call_ids=("c" * 64,),
        anomalous_call_ids=("c" * 64,),
        anomaly_types=("PARSER",),
        summary_class="base-parser-anomaly",
    )
    assert receipt.schema_version == "qualification-capability-anomaly-receipt.v1"
    with pytest.raises(ValidationError):
        receipt.summary_class = "changed"
    with pytest.raises(ValidationError):
        CapabilityAnomalyReceiptV1(
            release_id="release", plan_sha256=HASH_A, base_receipts_sha256=HASH_B,
            base_call_ids=("c" * 64,), anomalous_call_ids=("d" * 64,),
            anomaly_types=("PARSER",), summary_class="bad-subset",
        )


def _plan():
    from mub.vnext.post_core.qualification_planning_v1 import (
        CapabilitySmokePlanConfigV1,
        build_capability_budget_v1,
        build_capability_fixtures_v1,
        build_capability_smoke_plan_v1,
    )

    keys = (
        "qwen35_9b_bf16",
        "meta_muse_glimmer_30b_int4",
        "meta_muse_glimmer_30b_bf16",
        "claude_sonnet_4_6",
        "claude_opus_4_8",
        "gemini_3_6_flash",
        "grok_4_5",
        "gpt_5_5",
    )
    return build_capability_smoke_plan_v1(
        CapabilitySmokePlanConfigV1(
            release_id="memupdatebench.post-core.qualification.v1", registry_keys=keys, budget=build_capability_budget_v1()
        ),
        build_capability_fixtures_v1(),
    )


def test_load_execution_authorization_binds_canonical_plan_and_selected_base_calls(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_execution_authorization_v1

    plan = _plan()
    selected = tuple(row.call_id for row in plan.attempts[:8])
    authorization = _authorization(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        authorized_call_ids=selected,
        max_calls=len(selected),
    )
    path = tmp_path / "authorization.json"
    path.write_bytes(canonical_bytes(authorization))

    loaded = load_execution_authorization_v1(path, plan)

    assert loaded == authorization
CLI = Path(__file__).resolve().parents[2] / "scripts" / "vnext_run_post_core_capability_smoke.py"


def _write_authorization(path: Path, plan, call_ids: tuple[str, ...], adapter: Path) -> None:
    path.write_bytes(
        canonical_bytes(
            _authorization(
                release_id=plan.release_id,
                plan_sha256=canonical_hash(plan),
                authorized_call_ids=call_ids,
                max_calls=len(call_ids),
                adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
            )
        )
    )


def _write_local_adapter(path: Path, env_record: Path) -> None:
    path.write_text(
        """import hashlib, json, os, sys

assert sys.argv[1:] == ['--jsonl-protocol-v1']
open(%r, 'w', encoding='utf-8').write(json.dumps(sorted(os.environ)))
for line in sys.stdin.buffer:
""" % str(env_record) + """    attempt = json.loads(line)
    result = {
        'schema_version': 'memupdatebench.post-core.capability-adapter-result.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'request_sha256': hashlib.sha256(line.rstrip(b'\\n')).hexdigest(),
        'provider_call_count': 1,
        'retry_count': 0,
        'response_projection': {'exact_ok_1': 'READY', 'exact_ok_2': 'ACK'}.get(attempt['fixture_id'], 'Paris'),
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'error_class': None,
    }
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\\n')
    sys.stdout.buffer.flush()
""",
        encoding="utf-8",
    )


def test_cli_executes_only_authorized_qwen_base_calls_and_sanitizes_adapter_environment(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    selected = tuple(row.call_id for row in plan.attempts[:8])
    adapter = tmp_path / "adapter.py"
    environment_record = tmp_path / "adapter-environment.json"
    _write_local_adapter(adapter, environment_record)
    _write_authorization(authorization_path, plan, selected, adapter)
    output = tmp_path / "receipts.jsonl"
    environment = dict(os.environ)
    environment.update({"OPENAI_API_KEY": "test-secret", "CUSTOM_AUTH_TOKEN": "test-secret"})

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--plan", str(plan_path),
            "--authorization-receipt", str(authorization_path),
            "--adapter-executable", str(adapter),
            "--output", str(output),
            "--execute",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == {
        "base_count": 8,
        "call_count": 8,
        "escalation_count": 0,
        "fail_count": 0,
        "output": str(output),
        "pass_count": 8,
        "retries": 0,
        "status": "SUCCESS",
    }
    rows = output.read_bytes().splitlines()
    assert len(rows) == 8
    assert all(json.loads(row)["retry_count"] == 0 for row in rows)
    assert all(json.loads(row)["status"] == "PASS" for row in rows)
    assert all("response_projection" not in json.loads(row) for row in rows)
    for attempt, row in zip(plan.attempts[:8], rows):
        projection = {"exact_ok_1": "READY", "exact_ok_2": "ACK"}.get(attempt.fixture_id, "Paris")
        assert json.loads(row)["redacted_response_sha256"] == hashlib.sha256(
            canonical_bytes({"fixture_id": attempt.fixture_id, "projection": projection})
        ).hexdigest()
    adapter_environment = json.loads(environment_record.read_text(encoding="utf-8"))
    assert not any("api" in key.lower() or "token" in key.lower() or "auth" in key.lower() for key in adapter_environment)


def _cli_args(plan_path: Path, authorization_path: Path | None, adapter: Path, output: Path) -> list[str]:
    args = [
        sys.executable, str(CLI), "--plan", str(plan_path), "--adapter-executable", str(adapter),
        "--output", str(output),
    ]
    if authorization_path is not None:
        args.extend(["--authorization-receipt", str(authorization_path)])
    return args


def test_cli_blocks_missing_authorization_and_requires_execute(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "ignored")

    missing = subprocess.run(
        _cli_args(plan_path, None, adapter, tmp_path / "missing.jsonl"),
        capture_output=True, text=True, timeout=10,
    )
    assert missing.returncode == 10
    assert missing.stderr == "capability smoke blocked: missing execution authorization\n"

    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    unacknowledged = subprocess.run(
        _cli_args(plan_path, authorization_path, adapter, tmp_path / "unacknowledged.jsonl"),
        capture_output=True, text=True, timeout=10,
    )
    assert unacknowledged.returncode == 11
    assert not (tmp_path / "unacknowledged.jsonl").exists()


@pytest.mark.parametrize("mode", ["nonzero", "stderr", "malformed", "noncanonical", "missing", "extra", "duplicate", "wrong_registry", "secret"])
def test_cli_rejects_untrusted_adapter_protocol_without_output(tmp_path: Path, mode: str) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """import json, sys
rows = [json.loads(line) for line in sys.stdin]
mode = %r
if mode == 'nonzero': raise SystemExit(3)
if mode == 'stderr': sys.stderr.write('untrusted adapter detail')
if mode == 'malformed': sys.stdout.buffer.write(b'not-json\\n')
else:
    rows = rows[:-1] if mode == 'missing' else rows + rows[:1] if mode == 'extra' else rows
    for index, attempt in enumerate(rows):
        receipt = {'schema_version':'memupdatebench.post-core.capability-attempt-receipt.v1','call_id':attempt['call_id'],'registry_key':attempt['registry_key'],'status':'PASS','retry_count':0,'response_model':None,'response_format':'LOCAL_TEXT','stop_reason':None,'usage_present':None,'latency_ms':1,'redacted_response_sha256':'a'*64,'error_class':None}
        if mode == 'noncanonical': sys.stdout.write(json.dumps(receipt) + '\\n')
        elif mode == 'duplicate' and index: receipt['call_id'] = rows[0]['call_id']; sys.stdout.buffer.write(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()+b'\\n')
        elif mode == 'wrong_registry': receipt['registry_key'] = 'wrong'; sys.stdout.buffer.write(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()+b'\\n')
        elif mode == 'secret': receipt['error_class'] = 'api_key=leak'; sys.stdout.buffer.write(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()+b'\\n')
        elif mode not in {'noncanonical'}: sys.stdout.buffer.write(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()+b'\\n')
""" % mode,
        encoding="utf-8",
    )
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    output = tmp_path / "receipts.jsonl"

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, output), "--execute"],
        capture_output=True, text=True, timeout=20,
    )

    assert completed.returncode == 14
    assert completed.stdout == ""
    assert completed.stderr == "capability smoke adapter/runtime/protocol rejected\n"
    assert not output.exists()


def test_cli_rejects_existing_or_linked_output_before_adapter_start(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise AssertionError('adapter must not start')", encoding="utf-8")
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    existing = tmp_path / "existing.jsonl"
    existing.write_text("reserved", encoding="utf-8")

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, existing), "--execute"],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 13
    assert existing.read_text(encoding="utf-8") == "reserved"

    target = tmp_path / "target.jsonl"
    target.write_text("reserved", encoding="utf-8")
    linked = tmp_path / "linked.jsonl"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    linked_result = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, linked), "--execute"],
        capture_output=True, text=True, timeout=10,
    )
    assert linked_result.returncode == 13
    assert target.read_text(encoding="utf-8") == "reserved"


def test_authorization_loader_rejects_scope_unknown_ids_noncanonical_and_bad_escalation(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_execution_authorization_v1

    plan = _plan()
    base_ids = tuple(row.call_id for row in plan.attempts[:8])
    path = tmp_path / "authorization.json"
    invalid_payloads = (
        _authorization(release_id=plan.release_id, plan_sha256=canonical_hash(plan), authorized_call_ids=("f" * 64,), max_calls=1),
        _authorization(release_id=plan.release_id, plan_sha256=canonical_hash(plan), authorized_call_ids=(plan.attempts[8].call_id,), max_calls=1),
    )
    for authorization in invalid_payloads:
        path.write_bytes(canonical_bytes(authorization))
        with pytest.raises(ValueError):
            load_execution_authorization_v1(path, plan)
    path.write_bytes(canonical_bytes(_authorization(release_id=plan.release_id, plan_sha256=canonical_hash(plan), authorized_call_ids=base_ids, max_calls=8)) + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        load_execution_authorization_v1(path, plan)


def test_closed_receipt_requirements_and_public_exports_are_exact() -> None:
    import mub.vnext.post_core.qualification_receipts_v1 as receipts_module
    import mub.vnext.post_core.qualification_validation_v1 as validation_module
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_capability_attempt_receipts_v1

    plan = _plan()
    selected = next(row for row in plan.attempts if row.registry_key == "claude_sonnet_4_6" and row.phase.value == "BASE")
    receipt = CapabilityAttemptReceiptV1(
        call_id=selected.call_id,
        registry_key=selected.registry_key,
        status="PASS",
        response_model="claude-sonnet-4-6",
        response_format="SSE",
        stop_reason="end_turn",
        usage_present=True,
        redacted_response_sha256=HASH_A,
    )

    assert validate_capability_attempt_receipts_v1((selected,), (receipt,)) == (receipt,)
    with pytest.raises(ValueError, match="response format"):
        validate_capability_attempt_receipts_v1(
            (selected,), (receipt.model_copy(update={"response_format": "LOCAL_TEXT"}),)
        )
    with pytest.raises(ValueError, match="response model"):
        validate_capability_attempt_receipts_v1(
            (selected,), (receipt.model_copy(update={"response_model": ""}),)
        )
    assert "ExecutionAuthorizationV1" in receipts_module.__all__
    assert {"load_execution_authorization_v1", "validate_capability_attempt_receipts_v1"}.issubset(validation_module.__all__)
    with pytest.raises(ValidationError):
        _authorization(issued_at="not-a-utc-time")




def test_adapter_replacement_after_capture_cannot_change_executed_bytes(tmp_path: Path) -> None:
    import runpy

    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    module = runpy.run_path(str(CLI))

    def replace_original(original: Path, pinned: Path) -> None:
        original.write_text("raise AssertionError('replacement must not execute')", encoding="utf-8")

    module["main"].__globals__["_after_adapter_pinned"] = replace_original
    result = module["main"]([
        "--plan", str(plan_path), "--authorization-receipt", str(auth_path),
        "--adapter-executable", str(adapter), "--output", str(tmp_path / "receipts.jsonl"), "--execute",
    ])

    assert result == 0
    assert (tmp_path / "receipts.jsonl").exists()



def test_adapter_replacement_before_capture_is_rejected_by_authorized_hash(tmp_path: Path) -> None:
    import runpy

    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    adapter.write_text("raise AssertionError('replacement must not execute')", encoding="utf-8")
    module = runpy.run_path(str(CLI))

    result = module["main"]([
        "--plan", str(plan_path), "--authorization-receipt", str(auth_path),
        "--adapter-executable", str(adapter), "--output", str(tmp_path / "receipts.jsonl"), "--execute",
    ])

    assert result == 14
    assert not (tmp_path / "receipts.jsonl").exists()

def test_adapter_failure_removes_stably_reserved_output(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise SystemExit(3)", encoding="utf-8")
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    output = tmp_path / "receipts.jsonl"

    result = subprocess.run([*_cli_args(plan_path, auth_path, adapter, output), "--execute"], capture_output=True, text=True, timeout=20)

    assert result.returncode == 14
    assert not output.exists()

def test_output_parent_replacement_cannot_redirect_receipt_bytes(tmp_path: Path) -> None:
    import runpy

    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    parent = tmp_path / "requested-parent"
    replacement = tmp_path / "replacement-parent"
    old_parent = tmp_path / "old-parent"
    parent.mkdir()
    replacement.mkdir()
    output = parent / "receipts.jsonl"
    module = runpy.run_path(str(CLI))

    def replace_parent(reserved_output: Path) -> None:
        try:
            os.replace(parent, old_parent)
            os.replace(replacement, parent)
        except OSError as exc:
            pytest.skip(f"host prevents controlled parent replacement: {exc}")

    module["main"].__globals__["_before_adapter_run"] = replace_parent
    result = module["main"]([
        "--plan", str(plan_path), "--authorization-receipt", str(auth_path),
        "--adapter-executable", str(adapter), "--output", str(output), "--execute",
    ])

    assert result == 14
    assert not (parent / "receipts.jsonl").exists()


def _write_escalation_authorization(
    path: Path, plan, call_ids: tuple[str, ...], anomaly_raw: bytes, adapter: Path
) -> None:
    path.write_bytes(canonical_bytes(_authorization(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        authorized_call_ids=call_ids,
        max_calls=len(call_ids),
        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        escalation_anomaly_receipt_sha256=__import__("hashlib").sha256(anomaly_raw).hexdigest(),
    )))


def _anomaly(plan, base_ids: tuple[str, ...], base_receipts_sha256: str = HASH_A, **changes):
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAnomalyReceiptV1
    return CapabilityAnomalyReceiptV1(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        base_receipts_sha256=base_receipts_sha256,
        base_call_ids=base_ids,
        anomalous_call_ids=(base_ids[0],),
        anomaly_types=("PARSER",),
        summary_class="parser-anomaly",
    )



def _base_receipt_raw(plan, base_attempts, anomalous_ids: tuple[str, ...]) -> bytes:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1

    rows = tuple(
        CapabilityAttemptReceiptV1(
            call_id=attempt.call_id,
            registry_key=attempt.registry_key,
            status="FAIL",
            error_class="PARSER_MISMATCH" if attempt.call_id in anomalous_ids else "BASE_FAILURE",
        )
        for attempt in base_attempts
    )
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def test_escalation_requires_hash_bound_complete_base_receipt_evidence(tmp_path: Path) -> None:
    plan = _plan()
    escalation = next(row for row in plan.attempts if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION")
    base_attempts = tuple(row for row in plan.attempts if row.registry_key == escalation.registry_key and row.phase.value == "BASE")
    base_raw = _base_receipt_raw(plan, base_attempts, (base_attempts[0].call_id,))
    base_path = tmp_path / "base.jsonl"
    base_path.write_bytes(base_raw)
    anomaly_raw = canonical_bytes(_anomaly(plan, tuple(row.call_id for row in base_attempts), __import__("hashlib").sha256(base_raw).hexdigest()))
    anomaly_path = tmp_path / "anomaly.json"
    anomaly_path.write_bytes(anomaly_raw)
    auth_path = tmp_path / "authorization.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    _write_escalation_authorization(
        auth_path,
        plan,
        tuple(
            row.call_id for row in plan.attempts
            if row.registry_key == escalation.registry_key and row.phase.value == "ESCALATION"
        ),
        anomaly_raw,
        adapter,
    )
    output = tmp_path / "receipts.jsonl"

    missing_base = subprocess.run([*_cli_args(plan_path, auth_path, adapter, output), "--escalation-anomaly-receipt", str(anomaly_path), "--execute"], capture_output=True, text=True, timeout=20)
    assert missing_base.returncode == 10
    completed = subprocess.run([*_cli_args(plan_path, auth_path, adapter, output), "--escalation-anomaly-receipt", str(anomaly_path), "--base-receipts", str(base_path), "--execute"], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0






def test_anomaly_evidence_rejects_pass_and_type_error_mismatch() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    escalation = next(row for row in plan.attempts if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION")
    base_attempts = tuple(row for row in plan.attempts if row.registry_key == escalation.registry_key and row.phase.value == "BASE")
    raw = _base_receipt_raw(plan, base_attempts, (base_attempts[0].call_id,))
    rows = tuple(CapabilityAttemptReceiptV1.model_validate_json(line) for line in raw.splitlines())
    anomaly = _anomaly(plan, tuple(row.call_id for row in base_attempts), __import__("hashlib").sha256(raw).hexdigest())

    with pytest.raises(ValueError, match="anomaly type"):
        validate_escalation_anomaly_evidence_v1(
            anomaly.model_copy(update={"anomaly_types": ("FORMAT",)}), rows, raw, plan, (escalation,)
        )
    passed = list(rows)
    passed[0] = CapabilityAttemptReceiptV1(
        call_id=base_attempts[0].call_id, registry_key=base_attempts[0].registry_key,
        status="PASS", response_format="LOCAL_TEXT", latency_ms=1, redacted_response_sha256=HASH_A,
    )
    passed_raw = b"".join(canonical_bytes(row) + b"\n" for row in passed)
    passed_anomaly = _anomaly(plan, tuple(row.call_id for row in base_attempts), __import__("hashlib").sha256(passed_raw).hexdigest())
    with pytest.raises(ValueError, match="must be FAIL"):
        validate_escalation_anomaly_evidence_v1(passed_anomaly, tuple(passed), passed_raw, plan, (escalation,))

def test_escalation_rejects_base_receipt_hash_mismatch(tmp_path: Path) -> None:
    plan = _plan()
    escalation = next(row for row in plan.attempts if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION")
    base_attempts = tuple(row for row in plan.attempts if row.registry_key == escalation.registry_key and row.phase.value == "BASE")
    base_raw = _base_receipt_raw(plan, base_attempts, (base_attempts[0].call_id,))
    base_path = tmp_path / "base.jsonl"
    base_path.write_bytes(base_raw)
    anomaly_raw = canonical_bytes(_anomaly(plan, tuple(row.call_id for row in base_attempts), HASH_A))
    anomaly_path = tmp_path / "anomaly.json"
    anomaly_path.write_bytes(anomaly_raw)
    auth_path = tmp_path / "authorization.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    _write_escalation_authorization(
        auth_path,
        plan,
        tuple(
            row.call_id for row in plan.attempts
            if row.registry_key == escalation.registry_key and row.phase.value == "ESCALATION"
        ),
        anomaly_raw,
        adapter,
    )

    completed = subprocess.run([*_cli_args(plan_path, auth_path, adapter, tmp_path / "output.jsonl"), "--escalation-anomaly-receipt", str(anomaly_path), "--base-receipts", str(base_path), "--execute"], capture_output=True, text=True, timeout=20)

    assert completed.returncode == 12


def test_cli_help_exposes_no_provider_or_credential_flags() -> None:
    completed = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True, timeout=10)

    assert completed.returncode == 0
    assert {"--plan", "--authorization-receipt", "--escalation-anomaly-receipt", "--base-receipts", "--adapter-executable", "--output", "--execute"}.issubset(completed.stdout.split())
    assert not any(term in completed.stdout.lower() for term in ("provider", "endpoint", "credential", "token", "api-key"))


def test_cli_removes_requested_output_when_adapter_adds_hardlink(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    output = tmp_path / "receipts.jsonl"
    alias = tmp_path / "receipt-alias.jsonl"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """import json, os, sys

for line in sys.stdin:
    attempt = json.loads(line)
    if not os.path.exists(%r):
        os.link(%r, %r)
    result = {
        'schema_version': 'memupdatebench.post-core.capability-adapter-result.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'response_projection': {'exact_ok_1': 'READY', 'exact_ok_2': 'ACK'}.get(attempt['fixture_id'], 'Paris'),
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'error_class': None,
    }
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode() + b'\\n')
""" % (str(alias), str(output), str(alias)),
        encoding="utf-8",
    )
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, output), "--execute"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    if not alias.exists() and completed.returncode == 14:
        pytest.skip("host does not permit hardlink creation during adapter execution")
    assert completed.returncode == 14
    assert not output.exists()
    assert alias.read_bytes() == b""


def test_cli_persists_adapter_error_and_parser_mismatch_as_typed_failures(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    selected = tuple(row.call_id for row in plan.attempts[:8])
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """import hashlib, json, sys

for line in sys.stdin:
    attempt = json.loads(line)
    result = {
        'schema_version': 'memupdatebench.post-core.capability-adapter-result.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'request_sha256': hashlib.sha256(line.rstrip('\\n').encode('utf-8')).hexdigest(),
        'provider_call_count': 1,
        'retry_count': 0,
        'response_projection': None if attempt['fixture_id'] == 'exact_ok_1' else 'WRONG' if attempt['fixture_id'].startswith('exact_') else 'Paris',
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'error_class': 'ADAPTER_TIMEOUT' if attempt['fixture_id'] == 'exact_ok_1' else None,
    }
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode() + b'\\n')
""",
        encoding="utf-8",
    )
    _write_authorization(authorization_path, plan, selected, adapter)
    output = tmp_path / "receipts.jsonl"

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, output), "--execute"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 10
    assert json.loads(completed.stdout) == {
        "base_count": 8,
        "call_count": 8,
        "escalation_count": 0,
        "fail_count": 4,
        "output": str(output),
        "pass_count": 4,
        "retries": 0,
        "status": "BLOCKED",
    }
    persisted = tuple(json.loads(line) for line in output.read_bytes().splitlines())
    assert all("response_projection" not in row for row in persisted)
    for attempt, row in zip(plan.attempts[:8], persisted):
        if attempt.fixture_id == "exact_ok_1":
            assert row["status"] == "FAIL"
            assert row["error_class"] == "ADAPTER_TIMEOUT"
            assert row["redacted_response_sha256"] is None
        elif attempt.fixture_id == "exact_ok_2":
            assert row["status"] == "FAIL"
            assert row["error_class"] == "PARSER_MISMATCH"
            assert row["redacted_response_sha256"] is None
        else:
            assert row["status"] == "PASS"



def test_anomaly_evidence_requires_type_matched_failure_per_selected_role() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    roles = ("qwen35_9b_bf16", "meta_muse_glimmer_30b_int4")
    base_attempts = tuple(row for row in plan.attempts if row.registry_key in roles and row.phase.value == "BASE")
    selected = tuple(next(row for row in plan.attempts if row.registry_key == role and row.phase.value == "ESCALATION") for role in roles)
    anomalous_ids = (base_attempts[0].call_id, next(row.call_id for row in base_attempts if row.registry_key == roles[1]))
    rows = tuple(
        CapabilityAttemptReceiptV1(
            call_id=attempt.call_id, registry_key=attempt.registry_key, status="FAIL",
            error_class="PARSER_MISMATCH" if attempt.call_id == anomalous_ids[0] else "NETWORK_ERROR" if attempt.call_id == anomalous_ids[1] else "BASE_FAILURE",
        ) for attempt in base_attempts
    )
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAnomalyReceiptV1
    anomaly = CapabilityAnomalyReceiptV1(
        release_id=plan.release_id, plan_sha256=canonical_hash(plan),
        base_receipts_sha256=__import__("hashlib").sha256(raw).hexdigest(),
        base_call_ids=tuple(row.call_id for row in base_attempts), anomalous_call_ids=anomalous_ids,
        anomaly_types=("PARSER",), summary_class="two-role",
    )

    with pytest.raises(ValueError, match="declared anomaly type"):
        validate_escalation_anomaly_evidence_v1(anomaly, rows, raw, plan, selected)
    format_rows = tuple(row.model_copy(update={"error_class": "FORMAT_MISMATCH"}) if row.call_id == anomalous_ids[1] else row for row in rows)
    format_raw = b"".join(canonical_bytes(row) + b"\n" for row in format_rows)
    format_anomaly = anomaly.model_copy(update={
        "base_receipts_sha256": __import__("hashlib").sha256(format_raw).hexdigest(),
        "anomaly_types": ("PARSER", "FORMAT"),
    })
    assert validate_escalation_anomaly_evidence_v1(format_anomaly, format_rows, format_raw, plan, selected) == format_anomaly

def test_exact_fixtures_reject_leading_or_trailing_projection_whitespace() -> None:
    import runpy

    module = runpy.run_path(str(CLI))
    plan = _plan()
    exact_one = next(row for row in plan.attempts if row.fixture_id == "exact_ok_1")
    exact_two = next(row for row in plan.attempts if row.fixture_id == "exact_ok_2")

    assert module["_projection_matches_fixture"](exact_one, "READY") is True
    assert module["_projection_matches_fixture"](exact_two, "ACK") is True
    for attempt, projection in ((exact_one, " READY"), (exact_one, "READY "), (exact_two, " ACK"), (exact_two, "ACK ")):
        assert module["_projection_matches_fixture"](attempt, projection) is False


def test_execution_authorization_requires_adapter_sha256() -> None:
    authorization = _authorization(adapter_sha256=HASH_A)

    assert authorization.adapter_sha256 == HASH_A
    missing = authorization.model_dump(mode="json")
    missing.pop("adapter_sha256")
    with pytest.raises(ValidationError):
        ExecutionAuthorizationV1.model_validate(missing)
    with pytest.raises(ValidationError):
        _authorization(adapter_sha256="A" * 64)


def test_canonical_plan_validator_rejects_self_consistent_mutated_plan() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import (
        validate_canonical_capability_smoke_plan_v1,
    )

    plan = _plan()
    attempt_payload = plan.attempts[0].model_dump(mode="python", exclude={"call_id"})
    altered_attempt = type(plan.attempts[0]).model_validate(
        {**attempt_payload, "prompt_sha256": "0" * 64}
    )
    altered_plan = type(plan)(
        **plan.model_dump(mode="python", exclude={"attempts"}),
        attempts=(altered_attempt, *plan.attempts[1:]),
    )

    assert validate_canonical_capability_smoke_plan_v1(plan) == plan
    with pytest.raises(ValueError, match="canonical planner"):
        validate_canonical_capability_smoke_plan_v1(altered_plan)


def test_canonical_plan_validator_rejects_all_self_consistent_plan_mutations() -> None:
    from mub.vnext.post_core.qualification_validation_v1 import (
        validate_canonical_capability_smoke_plan_v1,
    )

    plan = _plan()

    def replace_attempt(attempt, **changes):
        return type(attempt).model_validate(
            {**attempt.model_dump(mode="python", exclude={"call_id"}), **changes}
        )

    def replace_plan(attempts, registry_keys=plan.registry_keys):
        payload = plan.model_dump(mode="python", exclude={"attempts"})
        payload["registry_keys"] = registry_keys
        return type(plan)(**payload, attempts=attempts)

    first = plan.attempts[0]
    changed_budget = first.budget.model_copy(update={"timeout_seconds": 59})
    changed = (
        replace_attempt(first, parser_sha256="1" * 64),
        replace_attempt(first, runtime_or_endpoint_class="forged_runtime"),
        replace_attempt(first, budget=changed_budget),
        replace_attempt(first, fixture_id="forged_fixture"),
    )
    for changed_attempt in changed:
        with pytest.raises(ValueError, match="canonical planner"):
            validate_canonical_capability_smoke_plan_v1(
                replace_plan((changed_attempt, *plan.attempts[1:]))
            )

    forged_registry = "forged_registry"
    forged_attempts = tuple(
        replace_attempt(attempt, registry_key=forged_registry)
        if attempt.registry_key == plan.registry_keys[0]
        else attempt
        for attempt in plan.attempts
    )
    with pytest.raises(ValueError, match="canonical planner"):
        validate_canonical_capability_smoke_plan_v1(
            replace_plan(forged_attempts, (forged_registry, *plan.registry_keys[1:]))
        )


def test_cli_rejects_mutated_plan_before_missing_authorization(tmp_path: Path) -> None:
    plan = _plan()
    attempt = type(plan.attempts[0]).model_validate({
        **plan.attempts[0].model_dump(mode="python", exclude={"call_id"}),
        "parser_sha256": "1" * 64,
    })
    altered = type(plan)(
        **plan.model_dump(mode="python", exclude={"attempts"}),
        attempts=(attempt, *plan.attempts[1:]),
    )
    plan_path = tmp_path / "altered-plan.json"
    plan_path.write_bytes(canonical_bytes(altered))
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise AssertionError('adapter must not start')", encoding="utf-8")

    completed = subprocess.run(
        [*_cli_args(plan_path, None, adapter, tmp_path / "receipts.jsonl"), "--execute"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 11
    assert completed.stderr == "capability smoke contract/usage rejected\n"


def test_adapter_result_binds_closed_registry_response_model() -> None:
    import runpy
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAdapterResultV1

    plan = _plan()
    attempt = next(
        row
        for row in plan.attempts
        if row.registry_key == "claude_sonnet_4_6" and row.fixture_id == "exact_ok_1"
    )
    result = CapabilityAdapterResultV1(
        call_id=attempt.call_id,
        registry_key=attempt.registry_key,
        request_sha256=hashlib.sha256(canonical_bytes(attempt)).hexdigest(),
        provider_call_count=1,
        retry_count=0,
        response_projection="READY",
        response_model="gpt-5.5",
        response_format="SSE",
        stop_reason="end_turn",
        usage_present=True,
        latency_ms=None,
    )

    module = runpy.run_path(str(CLI))
    with pytest.raises(Exception):
        module["_adapter_results_to_receipts"]((attempt,), (result,))

    local_attempt = next(
        row for row in plan.attempts
        if row.registry_key == "qwen35_9b_bf16" and row.fixture_id == "exact_ok_1"
    )
    local_result = CapabilityAdapterResultV1(
        call_id=local_attempt.call_id,
        registry_key=local_attempt.registry_key,
        request_sha256=hashlib.sha256(canonical_bytes(local_attempt)).hexdigest(),
        provider_call_count=1,
        retry_count=0,
        response_projection="READY",
        response_model="forged-local-model",
        response_format="LOCAL_TEXT",
        latency_ms=1,
    )
    with pytest.raises(Exception):
        module["_adapter_results_to_receipts"]((local_attempt,), (local_result,))


def test_adapter_result_rejects_unredacted_error_prose() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAdapterResultV1

    with pytest.raises(ValidationError):
        CapabilityAdapterResultV1(
            call_id=HASH_A,
            registry_key="qwen35_9b_bf16",
            error_class="provider returned raw error prose",
        )


def test_base_anomaly_evidence_rejects_incomplete_non_anomalous_pass() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    escalation = next(
        row for row in plan.attempts
        if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION"
    )
    base_attempts = tuple(
        row for row in plan.attempts
        if row.registry_key == escalation.registry_key and row.phase.value == "BASE"
    )
    rows = tuple(
        CapabilityAttemptReceiptV1(
            call_id=attempt.call_id,
            registry_key=attempt.registry_key,
            status="FAIL" if index == 0 else "PASS",
            response_format=None if index == 0 else "LOCAL_TEXT",
            latency_ms=None if index == 0 else 1,
            error_class="PARSER_MISMATCH" if index == 0 else None,
        )
        for index, attempt in enumerate(base_attempts)
    )
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    anomaly = _anomaly(
        plan,
        tuple(row.call_id for row in base_attempts),
        hashlib.sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="response hash"):
        validate_escalation_anomaly_evidence_v1(anomaly, rows, raw, plan, (escalation,))


def test_reserved_output_reparse_check_value_error_cleans_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy

    module = runpy.run_path(str(CLI))
    output, descriptor, parent_identity, output_identity = module["_reserve_output"](
        tmp_path / "receipts.jsonl"
    )
    monkeypatch.setitem(
        module["_reserved_output_stable"].__globals__,
        "_reject_reparse_components",
        lambda _path: (_ for _ in ()).throw(ValueError("unsafe")),
    )

    assert module["_reserved_output_stable"](output, descriptor, parent_identity, output_identity, 0) is False
    module["_discard_reserved_output"](output, descriptor, parent_identity, output_identity)
    assert output.exists()


def test_cli_rejects_oversized_adapter_output_without_receipt(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import sys\nsys.stdout.buffer.write(b'x' * (4 * 1024 * 1024 + 1))\nsys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    authorization_path = tmp_path / "authorization.json"
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    output = tmp_path / "receipts.jsonl"

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, output), "--execute"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 14
    assert completed.stdout == ""
    assert not output.exists()


def test_direct_anomaly_validator_binds_receipt_plan_and_selected_calls() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    escalation = next(
        attempt for attempt in plan.attempts
        if attempt.registry_key == "qwen35_9b_bf16" and attempt.phase.value == "ESCALATION"
    )
    base_attempts = tuple(
        attempt for attempt in plan.attempts
        if attempt.registry_key == escalation.registry_key and attempt.phase.value == "BASE"
    )
    raw = _base_receipt_raw(plan, base_attempts, (base_attempts[0].call_id,))
    rows = tuple(CapabilityAttemptReceiptV1.model_validate_json(line) for line in raw.splitlines())
    anomaly = _anomaly(
        plan, tuple(attempt.call_id for attempt in base_attempts), hashlib.sha256(raw).hexdigest()
    )

    with pytest.raises(ValueError, match="release ID"):
        validate_escalation_anomaly_evidence_v1(
            anomaly.model_copy(update={"release_id": "forged-release"}),
            rows,
            raw,
            plan,
            (escalation,),
        )
    with pytest.raises(ValueError, match="plan hash"):
        validate_escalation_anomaly_evidence_v1(
            anomaly.model_copy(update={"plan_sha256": "0" * 64}),
            rows,
            raw,
            plan,
            (escalation,),
        )

    fake_escalation = type(escalation).model_construct(**{
        **escalation.model_dump(mode="python", exclude={"call_id"}),
        "budget": escalation.budget,
        "prompt_sha256": "1" * 64,
    })
    with pytest.raises(ValueError, match="selected escalation"):
        validate_escalation_anomaly_evidence_v1(anomaly, rows, raw, plan, (fake_escalation,))


def test_direct_anomaly_validator_binds_base_receipt_rows_to_raw_bytes() -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    escalation = next(
        attempt for attempt in plan.attempts
        if attempt.registry_key == "qwen35_9b_bf16" and attempt.phase.value == "ESCALATION"
    )
    base_attempts = tuple(
        attempt for attempt in plan.attempts
        if attempt.registry_key == escalation.registry_key and attempt.phase.value == "BASE"
    )
    raw = _base_receipt_raw(plan, base_attempts, (base_attempts[0].call_id,))
    rows = tuple(CapabilityAttemptReceiptV1.model_validate_json(line) for line in raw.splitlines())
    altered_rows = (
        rows[0],
        rows[1].model_copy(update={"error_class": "DIFFERENT_FAILURE"}),
        *rows[2:],
    )
    anomaly = _anomaly(
        plan, tuple(attempt.call_id for attempt in base_attempts), hashlib.sha256(raw).hexdigest()
    )

    with pytest.raises(ValueError, match="do not match supplied receipt rows"):
        validate_escalation_anomaly_evidence_v1(anomaly, altered_rows, raw, plan, (escalation,))


def test_summary_blocks_when_persisted_receipts_include_failures() -> None:
    import runpy
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1

    plan = _plan()
    selected = plan.attempts[:2]
    receipts = (
        CapabilityAttemptReceiptV1(
            call_id=selected[0].call_id,
            registry_key=selected[0].registry_key,
            status="PASS",
            response_format="LOCAL_TEXT",
            latency_ms=1,
            redacted_response_sha256=HASH_A,
        ),
        CapabilityAttemptReceiptV1(
            call_id=selected[1].call_id,
            registry_key=selected[1].registry_key,
            status="FAIL",
            error_class="ADAPTER_TIMEOUT",
        ),
    )

    summary = json.loads(
        runpy.run_path(str(CLI))["_summary"](Path("receipts.jsonl"), selected, receipts)
    )
    assert summary["status"] == "BLOCKED"
    assert (summary["pass_count"], summary["fail_count"]) == (1, 1)


@pytest.mark.parametrize("issued_at", ["2026-08-24Z", "2026-08-24 00:00:00Z"])
def test_execution_authorization_rejects_non_rfc3339_utc_timestamp(issued_at: str) -> None:
    with pytest.raises(ValidationError):
        _authorization(issued_at=issued_at)


def test_bounded_adapter_reader_rejects_child_that_holds_output_pipes_open() -> None:
    import runpy

    module = runpy.run_path(str(CLI))
    source = (
        "import os, sys, time\n"
        "child = os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "time.sleep(30)\n"
    )

    started = time.monotonic()
    with pytest.raises(module["_AdapterProtocolError"]):
        module["_run_python_adapter_bounded"](source, b"", 1)

    assert time.monotonic() - started < 6


def test_adapter_batch_timeout_sums_attempt_budgets_and_caps() -> None:
    import runpy
    from types import SimpleNamespace

    module = runpy.run_path(str(CLI))
    attempts = (
        SimpleNamespace(budget=SimpleNamespace(timeout_seconds=20)),
        SimpleNamespace(budget=SimpleNamespace(timeout_seconds=21)),
        SimpleNamespace(budget=SimpleNamespace(timeout_seconds=22)),
    )

    assert module["_aggregate_adapter_timeout_seconds"](attempts) == 63
    assert module["_aggregate_adapter_timeout_seconds"](
        (SimpleNamespace(budget=SimpleNamespace(timeout_seconds=3601)),)
    ) == 3600
    with pytest.raises(ValueError, match="positive"):
        module["_aggregate_adapter_timeout_seconds"](
            (SimpleNamespace(budget=SimpleNamespace(timeout_seconds=0)),)
        )


@pytest.mark.parametrize(
    ("anomaly_type", "error_class"),
    [
        ("FORMAT", "NOT_FORMAT"),
        ("FORMAT", "MALFORMATTED"),
        ("STABILITY", "UNSTABLE_STABILITY"),
    ],
)
def test_anomaly_evidence_rejects_substring_matched_error_classes(
    anomaly_type: str, error_class: str
) -> None:
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAttemptReceiptV1
    from mub.vnext.post_core.qualification_validation_v1 import validate_escalation_anomaly_evidence_v1

    plan = _plan()
    escalation = next(
        row for row in plan.attempts
        if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION"
    )
    base_attempts = tuple(
        row for row in plan.attempts
        if row.registry_key == escalation.registry_key and row.phase.value == "BASE"
    )
    rows = tuple(
        CapabilityAttemptReceiptV1(
            call_id=attempt.call_id,
            registry_key=attempt.registry_key,
            status="FAIL",
            error_class=error_class if index == 0 else "BASE_FAILURE",
        )
        for index, attempt in enumerate(base_attempts)
    )
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    anomaly = _anomaly(
        plan,
        tuple(row.call_id for row in base_attempts),
        hashlib.sha256(raw).hexdigest(),
    ).model_copy(update={"anomaly_types": (anomaly_type,)})

    with pytest.raises(ValueError, match="declared anomaly type"):
        validate_escalation_anomaly_evidence_v1(anomaly, rows, raw, plan, (escalation,))


@pytest.mark.parametrize("injected_field", ["status", "redacted_response_sha256"])
def test_cli_rejects_adapter_verdict_or_hash_injection(
    tmp_path: Path, injected_field: str
) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    authorization_path = tmp_path / "authorization.json"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """import hashlib, json, sys

for line in sys.stdin:
    attempt = json.loads(line)
    result = {
        'schema_version': 'memupdatebench.post-core.capability-adapter-result.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'request_sha256': hashlib.sha256(line.rstrip('\\n').encode('utf-8')).hexdigest(),
        'provider_call_count': 1,
        'retry_count': 0,
        'response_projection': {'exact_ok_1': 'READY', 'exact_ok_2': 'ACK'}.get(attempt['fixture_id'], 'Paris'),
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'error_class': None,
        %r: %r,
    }
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode() + b'\\n')
""" % (injected_field, "PASS" if injected_field == "status" else "a" * 64),
        encoding="utf-8",
    )
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]), adapter)
    output = tmp_path / "receipts.jsonl"

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization_path, adapter, output), "--execute"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 14
    assert not output.exists()


def test_authorization_requires_complete_role_phase_batches(tmp_path: Path) -> None:
    from mub.vnext.post_core.qualification_validation_v1 import load_execution_authorization_v1

    plan = _plan()
    qwen_base = tuple(
        attempt.call_id for attempt in plan.attempts
        if attempt.registry_key == "qwen35_9b_bf16" and attempt.phase.value == "BASE"
    )
    claude_base = tuple(
        attempt.call_id for attempt in plan.attempts
        if attempt.registry_key == "claude_opus_4_8" and attempt.phase.value == "BASE"
    )
    path = tmp_path / "authorization.json"
    for selected in (qwen_base[:1], qwen_base[:7]):
        authorization = _authorization(
            release_id=plan.release_id,
            plan_sha256=canonical_hash(plan),
            authorized_call_ids=selected,
            max_calls=len(selected),
        )
        path.write_bytes(canonical_bytes(authorization))
        with pytest.raises(ValueError, match="complete role-phase batch"):
            load_execution_authorization_v1(path, plan)

    selected = (*qwen_base, *claude_base)
    authorization = _authorization(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        authorized_call_ids=selected,
        max_calls=len(selected),
    )
    path.write_bytes(canonical_bytes(authorization))
    assert load_execution_authorization_v1(path, plan) == authorization


def _write_attested_local_adapter(path: Path, marker: Path | None = None, *, unstable: bool = False) -> None:
    path.write_text(
        """import hashlib, json, sys

for line in sys.stdin.buffer:
    attempt = json.loads(line)
    %s
    projection = {'exact_ok_1': 'READY', 'exact_ok_2': 'ACK'}.get(attempt['fixture_id'], 'London' if %s and attempt['repetition'] == 2 else 'Paris')
    result = {
        'schema_version': 'memupdatebench.post-core.capability-adapter-result.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'request_sha256': hashlib.sha256(line.rstrip(b'\\n')).hexdigest(),
        'provider_call_count': 1,
        'retry_count': 0,
        'response_projection': projection,
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'error_class': None,
    }
    sys.stdout.buffer.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode() + b'\\n')
""" % (
            "open(%r, 'w', encoding='utf-8').write('executed')" % str(marker) if marker else "pass",
            repr(unstable),
        ),
        encoding="utf-8",
    )


def test_closed_api_budget_gate_blocks_complete_claude_batch_before_adapter(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    selected = tuple(
        attempt.call_id for attempt in plan.attempts
        if attempt.registry_key == "claude_opus_4_8" and attempt.phase.value == "BASE"
    )
    adapter = tmp_path / "adapter.py"
    marker = tmp_path / "adapter-executed"
    _write_attested_local_adapter(adapter, marker)
    authorization = tmp_path / "authorization.json"
    _write_authorization(authorization, plan, selected, adapter)

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization, adapter, tmp_path / "receipts.jsonl"), "--execute"],
        capture_output=True, text=True, timeout=20,
    )

    assert completed.returncode == 10
    assert completed.stderr == "capability smoke blocked: closed API budget is unpriced or zero\n"
    assert not marker.exists()


def test_cli_rejects_secret_bearing_adapter_source_before_execution(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    marker = tmp_path / "adapter-executed"
    adapter.write_text("# sk-SECRET_TEST_VALUE\nopen(%r, 'w').write('executed')\n" % str(marker), encoding="utf-8")
    authorization = tmp_path / "authorization.json"
    qwen_base = tuple(attempt.call_id for attempt in plan.attempts[:8])
    _write_authorization(authorization, plan, qwen_base, adapter)

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization, adapter, tmp_path / "receipts.jsonl"), "--execute"],
        capture_output=True, text=True, timeout=20,
    )

    assert completed.returncode == 14
    assert not marker.exists()


@pytest.mark.parametrize(("unstable", "expected_status"), [(False, "PASS"), (True, "FAIL")])
def test_cli_requires_repetition_stability_before_fixture_verdict(
    tmp_path: Path, unstable: bool, expected_status: str
) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_attested_local_adapter(adapter, unstable=unstable)
    authorization = tmp_path / "authorization.json"
    selected = tuple(attempt.call_id for attempt in plan.attempts[:8])
    _write_authorization(authorization, plan, selected, adapter)
    output = tmp_path / "receipts.jsonl"

    completed = subprocess.run(
        [*_cli_args(plan_path, authorization, adapter, output), "--execute"],
        capture_output=True, text=True, timeout=20,
    )

    assert completed.returncode == (10 if unstable else 0)
    rows = tuple(json.loads(line) for line in output.read_bytes().splitlines())
    if unstable:
        unstable_rows = tuple(
            row for attempt, row in zip(plan.attempts[:8], rows)
            if attempt.fixture_id.startswith("parser_city")
        )
        assert {row["status"] for row in unstable_rows} == {"FAIL"}
        assert {row["error_class"] for row in unstable_rows} == {"STABILITY_MISMATCH"}
        assert all(row["redacted_response_sha256"] is None for row in unstable_rows)
    else:
        assert {row["status"] for row in rows} == {expected_status}


def test_bounded_adapter_writer_times_out_when_child_never_reads_large_payload() -> None:
    import runpy

    module = runpy.run_path(str(CLI))
    source = "import time; time.sleep(30)\n"
    payload = b"x" * (128 * 1024)

    started = time.monotonic()
    with pytest.raises(module["_AdapterProtocolError"]):
        module["_run_python_adapter_bounded"](source, payload, 1)

    assert time.monotonic() - started < 6


def test_adapter_result_request_hash_mismatch_is_rejected_before_receipt_persistence() -> None:
    import runpy
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAdapterResultV1

    plan = _plan()
    selected = tuple(plan.attempts[:8])
    results = tuple(
        CapabilityAdapterResultV1(
            call_id=attempt.call_id,
            registry_key=attempt.registry_key,
            request_sha256="0" * 64,
            provider_call_count=1,
            retry_count=0,
            response_projection={"exact_ok_1": "READY", "exact_ok_2": "ACK"}.get(attempt.fixture_id, "Paris"),
            response_format="LOCAL_TEXT",
            latency_ms=1,
        )
        for attempt in selected
    )

    with pytest.raises(Exception):
        runpy.run_path(str(CLI))["_adapter_results_to_receipts"](selected, results)
