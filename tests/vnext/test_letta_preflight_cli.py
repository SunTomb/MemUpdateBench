from __future__ import annotations

from scripts.vnext_preflight_letta import run_preflight


def test_letta_preflight_is_metadata_only_and_fails_closed_without_verified_runtime() -> None:
    payload = run_preflight(run_prefix="letta-test-preflight")

    assert payload["schema_version"] == "memupdatebench.external.letta.preflight.v1"
    assert payload["identity"]["package_version"] == "0.16.8"
    assert payload["identity"]["license_id"] == "Apache-2.0"
    assert payload["execution_boundary"] == {
        "llm_used": False,
        "api_used": False,
        "gpu_used": False,
        "network_credential_inputs": False,
    }
    assert payload["unsupported"]["passage_memory"] is True
    assert payload["unsupported"]["agent_mode"] is True
    assert payload["unsupported"]["native_answer"] is True
    assert payload["package_preflight"]["identity_verified"] is False
    assert payload["outcome"] == "blocked"
    assert payload["passed"] is False


def test_runtime_preflight_reports_not_run_when_factory_is_unavailable() -> None:
    from scripts.vnext_preflight_letta_runtime import run_preflight

    payload = run_preflight(
        python_executable="C:/Python/python.exe",
        worker_command=("-m", "missing.letta.worker"),
        server_identity="http://127.0.0.1:8283",
        database_identity="sqlite:///letta-preflight",
        run_prefix="letta-runtime-test",
        bridge_factory=lambda **_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert payload["schema_version"] == "memupdatebench.external.letta.preflight.v2"
    assert payload["outcome"] == "blocked"
    assert payload["passed"] is False
    assert payload["lifecycle"]["status"] == "NOT_RUN"
    assert "runtime_bridge_unavailable" in payload["blockers"]
    assert payload["metrics"] is None


class _SuccessfulFakeAdapter:
    def __init__(self, *, bridge):
        from scripts.vnext_preflight_letta_runtime import _key

        self.bridge = bridge
        self.key = _key()
        self.values = {}
        self.sentinels = {}
        self.closed = False

    def reset(self, request):
        from mub.vnext.contracts.v3.adapter import ResetResultV3

        self.reset_namespace(request.namespace)
        return ResetResultV3(success=True, namespace=request.namespace)

    def reset_namespace(self, namespace):
        self.values.clear()
        self.sentinels.pop(namespace, None)

    def write_sentinel(self, namespace, sentinel_id, sentinel_text):
        self.sentinels.setdefault(namespace, {})[sentinel_id] = sentinel_text

    def sentinel_visible(self, namespace, sentinel_text):
        return sentinel_text in self.sentinels.get(namespace, {}).values()

    def ingest_event(self, event):
        from mub.vnext.contracts.enums import ActionScope, Operation
        from mub.vnext.contracts.v3.adapter import (
            AdapterActionPayloadV3,
            AdapterActionResultV3,
        )
        from mub.vnext.contracts.v3.enums import ExecutionStatusV3

        if "No memory" in event.normalized_text:
            op = Operation.NOOP
            effective = op
        elif event.normalized_text.startswith("Add"):
            op = Operation.ADD
            effective = op
            self.values["id"] = "Paris"
        elif event.normalized_text.startswith("Update"):
            op = Operation.UPDATE
            effective = op
            self.values["id"] = "Lyon"
        else:
            op = Operation.DELETE
            effective = op
            self.values.clear()
        kwargs = (
            {}
            if op is Operation.NOOP
            else {
                "scope": ActionScope.OBJECT,
                "target_object_keys": (self.key,),
            }
        )
        if op in (Operation.ADD, Operation.UPDATE):
            kwargs["value"] = self.values["id"]
        return AdapterActionResultV3(
            event_id=event.event_id,
            requested_action=AdapterActionPayloadV3(operation=op, **kwargs),
            effective_action=AdapterActionPayloadV3(operation=effective, **kwargs),
            execution_status=ExecutionStatusV3.EXECUTED,
            affected_entry_ids=("stable-id",) if op is not Operation.NOOP else (),
        )

    def export_entries(self):
        from mub.vnext.contracts.v3.adapter import ExportEntriesResultV3
        from mub.vnext.contracts.v3.runtime import MemoryEntryRecordV3

        return ExportEntriesResultV3(
            entries=()
            if not self.values
            else (
                MemoryEntryRecordV3(
                    entry_id="stable-id",
                    content="Lyon",
                    object_key_candidate=self.key,
                    value_candidate="Lyon",
                    source_event_ids=("update",),
                ),
            )
        )

    def retrieve(self, request):
        from mub.vnext.contracts.v3.adapter import RetrievalResultV3
        from mub.vnext.contracts.v3.runtime import RetrievalTraceV3

        entries = self.export_entries().entries
        return RetrievalResultV3(
            request=request,
            trace=RetrievalTraceV3(
                query_id=request.query.query_id,
                retrieved_entries=entries,
                scores=(1.0,),
                ranks=(1,),
                retrieval_policy="fake",
                context_order="fake",
            ),
        )

    def answer(self, query, mode):
        from mub.vnext.contracts.v3.adapter import AdapterAnswerResultV3
        from mub.vnext.contracts.v3.runtime import AnswerPredictionV3

        return AdapterAnswerResultV3(
            prediction=AnswerPredictionV3(
                query_id=query.query_id,
                raw_output="Lyon",
                parsed_answer="Lyon",
                cited_entry_ids=("stable-id",),
                format_valid=True,
            )
        )

    def close(self):
        self.closed = True
        self.bridge.close()


def test_runtime_preflight_lifecycle_success_uses_stable_id_and_slot_direct() -> None:
    from scripts.vnext_preflight_letta_runtime import _run_lifecycle

    result = _run_lifecycle(
        _SuccessfulFakeAdapter(bridge=None), "letta-runtime-lifecycle"
    )
    assert result["passed"] is True
    assert result["stable_entry_id"] == "stable-id"
    assert result["slot_direct"]["passed"] is True
    assert result["export_after_delete_empty"] is True


def test_runtime_preflight_runs_successful_fake_path_before_official_gate() -> None:
    from scripts.vnext_preflight_letta_runtime import run_preflight

    class FakeBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def close(self):
            self.closed = True

    bridges = []

    def bridge_factory(**kwargs):
        bridge = FakeBridge(**kwargs)
        bridges.append(bridge)
        return bridge

    adapters = []

    def adapter_factory(**kwargs):
        adapter = _SuccessfulFakeAdapter(bridge=kwargs["bridge"])
        adapters.append(adapter)
        return adapter

    payload = run_preflight(
        python_executable="C:/Python/python.exe",
        worker_command=("-m", "missing.letta.worker"),
        server_identity="http://127.0.0.1:8283",
        database_identity="sqlite:///letta-preflight",
        run_prefix="letta-runtime-successful-fake",
        bridge_factory=bridge_factory,
        adapter_factory=adapter_factory,
    )

    assert payload["candidate_id"] == "letta_0_16_8_profile"
    assert payload["namespace_reset_probe"]["passed"] is True
    assert payload["lifecycle"]["passed"] is True
    assert payload["clean_close"] == {"status": "PASS", "passed": True}
    assert payload["official_health"] == {"passed": False, "source_binding": "blocked"}
    assert payload["passed"] is False
    assert "official_adapter_boundary_unverified" in payload["blockers"]
    assert len(adapters) == 1 and adapters[0].closed is True
    assert len(bridges) == 1 and bridges[0].closed is True
