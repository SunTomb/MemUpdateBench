from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

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
            release_id="release-v1", registry_keys=keys, budget=build_capability_budget_v1()
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


def _write_authorization(path: Path, plan, call_ids: tuple[str, ...]) -> None:
    path.write_bytes(
        canonical_bytes(
            _authorization(
                release_id=plan.release_id,
                plan_sha256=canonical_hash(plan),
                authorized_call_ids=call_ids,
                max_calls=len(call_ids),
            )
        )
    )


def _write_local_adapter(path: Path, env_record: Path) -> None:
    path.write_text(
        """import hashlib, json, os, sys

assert sys.argv[1:] == ['--jsonl-protocol-v1']
open(%r, 'w', encoding='utf-8').write(json.dumps(sorted(os.environ)))
for line in sys.stdin:
""" % str(env_record) + """    attempt = json.loads(line)
    receipt = {
        'schema_version': 'memupdatebench.post-core.capability-attempt-receipt.v1',
        'call_id': attempt['call_id'],
        'registry_key': attempt['registry_key'],
        'status': 'PASS',
        'retry_count': 0,
        'response_model': None,
        'response_format': 'LOCAL_TEXT',
        'stop_reason': None,
        'usage_present': None,
        'latency_ms': 1,
        'redacted_response_sha256': hashlib.sha256(attempt['call_id'].encode()).hexdigest(),
        'error_class': None,
    }
    sys.stdout.buffer.write(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\\n')
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
    _write_authorization(authorization_path, plan, selected)
    adapter = tmp_path / "adapter.py"
    environment_record = tmp_path / "adapter-environment.json"
    _write_local_adapter(adapter, environment_record)
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
        "output": str(output),
        "retries": 0,
        "status": "SUCCESS",
    }
    rows = output.read_bytes().splitlines()
    assert len(rows) == 8
    assert all(json.loads(row)["retry_count"] == 0 for row in rows)
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
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
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
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
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
    _write_authorization(authorization_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise AssertionError('adapter must not start')", encoding="utf-8")
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
    with pytest.raises(ValueError, match="nonblank response model"):
        validate_capability_attempt_receipts_v1(
            (selected,), (receipt.model_copy(update={"response_model": ""}),)
        )
    assert "ExecutionAuthorizationV1" in receipts_module.__all__
    assert {"load_execution_authorization_v1", "validate_capability_attempt_receipts_v1"}.issubset(validation_module.__all__)
    with pytest.raises(ValidationError):
        _authorization(issued_at="not-a-utc-time")




def test_adapter_replacement_after_pin_cannot_change_executed_bytes(tmp_path: Path) -> None:
    import runpy

    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
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


def test_adapter_failure_removes_stably_reserved_output(tmp_path: Path) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    auth_path = tmp_path / "authorization.json"
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise SystemExit(3)", encoding="utf-8")
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
    _write_authorization(auth_path, plan, tuple(row.call_id for row in plan.attempts[:8]))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
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


def _write_escalation_authorization(path: Path, plan, call_ids: tuple[str, ...], anomaly_raw: bytes) -> None:
    path.write_bytes(canonical_bytes(_authorization(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        authorized_call_ids=call_ids,
        max_calls=len(call_ids),
        escalation_anomaly_receipt_sha256=__import__("hashlib").sha256(anomaly_raw).hexdigest(),
    )))


def _anomaly(plan, base_ids: tuple[str, ...]):
    from mub.vnext.post_core.qualification_receipts_v1 import CapabilityAnomalyReceiptV1
    return CapabilityAnomalyReceiptV1(
        release_id=plan.release_id,
        plan_sha256=canonical_hash(plan),
        base_receipts_sha256=HASH_A,
        base_call_ids=base_ids,
        anomalous_call_ids=(base_ids[0],),
        anomaly_types=("PARSER",),
        summary_class="parser-anomaly",
    )


def test_escalation_requires_hash_bound_typed_base_anomaly_evidence(tmp_path: Path) -> None:
    plan = _plan()
    escalation = next(row for row in plan.attempts if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION")
    base_ids = tuple(row.call_id for row in plan.attempts if row.registry_key == escalation.registry_key and row.phase.value == "BASE")
    anomaly_path = tmp_path / "anomaly.json"
    anomaly_raw = canonical_bytes(_anomaly(plan, base_ids))
    anomaly_path.write_bytes(anomaly_raw)
    auth_path = tmp_path / "authorization.json"
    _write_escalation_authorization(auth_path, plan, (escalation.call_id,), anomaly_raw)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    output = tmp_path / "receipts.jsonl"

    missing = subprocess.run([*_cli_args(plan_path, auth_path, adapter, output), "--execute"], capture_output=True, text=True, timeout=20)
    assert missing.returncode == 10
    success = subprocess.run([*_cli_args(plan_path, auth_path, adapter, output), "--escalation-anomaly-receipt", str(anomaly_path), "--execute"], capture_output=True, text=True, timeout=20)
    assert success.returncode == 0
    assert len(output.read_bytes().splitlines()) == 1


def test_escalation_rejects_bad_anomaly_hash_and_missing_base_role_coverage(tmp_path: Path) -> None:
    plan = _plan()
    escalation = next(row for row in plan.attempts if row.registry_key == "qwen35_9b_bf16" and row.phase.value == "ESCALATION")
    base_ids = tuple(row.call_id for row in plan.attempts if row.registry_key == escalation.registry_key and row.phase.value == "BASE")
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_bytes(plan))
    adapter = tmp_path / "adapter.py"
    _write_local_adapter(adapter, tmp_path / "environment.json")
    anomaly_path = tmp_path / "anomaly.json"
    short_raw = canonical_bytes(_anomaly(plan, base_ids[:1]))
    anomaly_path.write_bytes(short_raw)
    auth_path = tmp_path / "authorization.json"
    _write_escalation_authorization(auth_path, plan, (escalation.call_id,), short_raw)
    incomplete = subprocess.run([*_cli_args(plan_path, auth_path, adapter, tmp_path / "incomplete.jsonl"), "--escalation-anomaly-receipt", str(anomaly_path), "--execute"], capture_output=True, text=True, timeout=20)
    assert incomplete.returncode == 11
    _write_escalation_authorization(auth_path, plan, (escalation.call_id,), b"wrong")
    mismatch = subprocess.run([*_cli_args(plan_path, auth_path, adapter, tmp_path / "mismatch.jsonl"), "--escalation-anomaly-receipt", str(anomaly_path), "--execute"], capture_output=True, text=True, timeout=20)
    assert mismatch.returncode == 12
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"], capture_output=True, text=True, timeout=10
    )

    assert completed.returncode == 0
    assert {"--plan", "--authorization-receipt", "--adapter-executable", "--output", "--execute"}.issubset(
        completed.stdout.split()
    )
    assert not any(term in completed.stdout.lower() for term in ("provider", "endpoint", "credential", "token", "api-key"))
