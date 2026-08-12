from __future__ import annotations

from pathlib import Path
import os
import sys
import textwrap

import pytest

from mub.vnext.contracts.v3.task import MemoryEventV3, MemoryQueryV3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "data" / "vnext" / "core" / "v3"


@pytest.fixture(scope="session")
def authenticated_release():
    from mub.vnext.external.canaries_v3 import authenticate_core_release

    return authenticate_core_release(RELEASE_ROOT)


def test_external_facade_exports_provider_neutral_gate_contracts():
    from mub.vnext.external import (
        CapabilityVerificationV1,
        JsonlSubprocessBridge,
        NamespaceResetProbeV1,
        NormalizedArtifactRefV1,
        PrivateRawArtifactRefV1,
        ProviderEventInputV1,
        ProviderQueryInputV1,
        WorkerRequestV1,
    )

    assert ProviderEventInputV1.__name__ == "ProviderEventInputV1"
    assert ProviderQueryInputV1.__name__ == "ProviderQueryInputV1"
    assert WorkerRequestV1.__name__ == "WorkerRequestV1"
    assert JsonlSubprocessBridge.__name__ == "JsonlSubprocessBridge"
    assert NamespaceResetProbeV1.__name__ == "NamespaceResetProbeV1"
    assert CapabilityVerificationV1.__name__ == "CapabilityVerificationV1"
    assert PrivateRawArtifactRefV1.__name__ == "PrivateRawArtifactRefV1"
    assert NormalizedArtifactRefV1.__name__ == "NormalizedArtifactRefV1"


def test_visible_event_and_query_inputs_expose_only_provider_fields(
    authenticated_release,
):
    from mub.vnext.external.visibility import (
        ProviderEventInputV1,
        ProviderQueryInputV1,
        visible_event_input,
        visible_query_input,
    )

    task = authenticated_release.dev_tasks[0]
    event = task.events[0]
    query = task.queries[0]
    visible_event = visible_event_input(
        event,
        runtime_namespace="task10-visible-test",
    )
    visible_query = visible_query_input(
        query,
        k=5,
        runtime_namespace="task10-visible-test",
    )

    assert type(visible_event) is ProviderEventInputV1
    assert visible_event.model_dump(mode="python") == {
        "event_id": event.event_id,
        "sequence_index": event.sequence_index,
        "logical_time": event.timestamp,
        "raw_text": event.raw_text,
        "runtime_namespace": "task10-visible-test",
    }
    assert type(visible_query) is ProviderQueryInputV1
    assert visible_query.model_dump(mode="python") == {
        "query_id": query.query_id,
        "query_text": query.text,
        "k": 5,
        "runtime_namespace": "task10-visible-test",
    }


def test_visible_converters_reject_constructed_or_subclassed_inputs(
    authenticated_release,
):
    from mub.vnext.external.visibility import (
        visible_event_input,
        visible_query_input,
    )

    task = authenticated_release.dev_tasks[0]
    forged_event = MemoryEventV3.model_construct(
        **{
            **task.events[0].model_dump(mode="python"),
            "sequence_index": "0",
        }
    )
    with pytest.raises(ValueError, match="trust-boundary"):
        visible_event_input(
            forged_event,
            runtime_namespace="task10-visible-test",
        )

    class QuerySubclass(MemoryQueryV3):
        pass

    forged_query = QuerySubclass.model_validate(
        task.queries[0].model_dump(mode="python")
    )
    with pytest.raises(ValueError, match="exact MemoryQueryV3"):
        visible_query_input(
            forged_query,
            k=5,
            runtime_namespace="task10-visible-test",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"gold_action_ids": ["action-1"]},
        {"nested": {"versionHistory": []}},
        {"history": [{"old_value": "secret"}]},
        {"nested": {"Historical-Values": ["old"]}},
        {"items": [{"target-object-keys": []}]},
        {"metadata": {"Gold Answer": "secret"}},
        {"derivation": {"steps": []}},
        {"stratification_label": "hard"},
        {"staleAlternative": "old"},
        {"selector": {"kind": "current"}},
        {"evidence": ["gold-event"]},
    ],
)
def test_recursive_visible_payload_guard_rejects_benchmark_privileged_keys(
    payload,
):
    from mub.vnext.external.visibility import validate_visible_payload

    with pytest.raises(ValueError, match="provider-visible payload"):
        validate_visible_payload(payload)


def test_recursive_visible_payload_guard_accepts_visible_protocol_fields():
    from mub.vnext.external.visibility import validate_visible_payload

    payload = {
        "request_id": "request-1",
        "operation": "retrieve",
        "payload": {
            "query_id": "query-1",
            "query_text": "What is remembered?",
            "k": 5,
            "runtime_namespace": "task10-visible-test",
        },
    }
    assert validate_visible_payload(payload) is payload


def _worker_environment(*, include_api_key: bool = False):
    from mub.vnext.external.security import build_worker_environment

    source = dict(os.environ)
    source["PYTHONIOENCODING"] = "utf-8"
    source["UNRELATED_SECRET"] = "must-not-pass"
    if include_api_key:
        source["OPENAI_API_KEY"] = "environment-secret"
    allowed = tuple(
        name
        for name in (
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "PYTHONIOENCODING",
            "OPENAI_API_KEY",
        )
        if name in source
    )
    required = ("OPENAI_API_KEY",) if include_api_key else ()
    return build_worker_environment(
        source,
        allowed_names=allowed,
        required_names=required,
    )


def test_jsonl_bridge_rejects_credentials_in_command_arguments(tmp_path):
    from mub.vnext.external.bridge import JsonlSubprocessBridge

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    with pytest.raises(ValueError, match="command security") as exc_info:
        JsonlSubprocessBridge(
            command=(sys.executable, "-c", "pass", secret),
            cwd=tmp_path,
            environment=_worker_environment(),
            timeout_seconds=5.0,
        )
    assert secret not in str(exc_info.value)


def test_jsonl_bridge_round_trip_uses_explicit_environment(tmp_path):
    from mub.vnext.external.bridge import (
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "echo_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            for line in sys.stdin.buffer:
                request = json.loads(line)
                response = {
                    "schema_version": "memupdatebench.external.worker_response.v1",
                    "request_id": request["request_id"],
                    "status": "ok",
                    "payload": {
                        "operation": request["operation"],
                        "seen_api_key": "OPENAI_API_KEY" in os.environ,
                        "seen_unrelated_secret": "UNRELATED_SECRET" in os.environ,
                    },
                    "error_code": None,
                }
                raw = json.dumps(
                    response,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\\n"
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="request-1",
        operation=WorkerOperation.RETRIEVE,
        payload={
            "query_id": "query-1",
            "query_text": "What is remembered?",
            "k": 5,
            "runtime_namespace": "task10-bridge-test",
        },
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(include_api_key=True),
        timeout_seconds=5.0,
    ) as bridge:
        response = bridge.request(request)

    assert response.request_id == request.request_id
    assert response.payload["operation"] == "retrieve"
    assert response.payload["seen_api_key"] is True
    assert response.payload["seen_unrelated_secret"] is False


def test_worker_request_rejects_privileged_nested_payload():
    from mub.vnext.external.bridge import WorkerOperation, WorkerRequestV1

    with pytest.raises(ValueError, match="provider-visible payload"):
        WorkerRequestV1(
            request_id="request-1",
            operation=WorkerOperation.INGEST_EVENT,
            payload={
                "event_id": "event-1",
                "raw_text": "Visible text",
                "gold_action_ids": ["gold-1"],
            },
        )


@pytest.mark.parametrize(
    ("response_source", "error_match"),
    [
        (
            (
                "response = {"
                "\"schema_version\": "
                "\"memupdatebench.external.worker_response.v1\", "
                "\"request_id\": \"wrong-id\", "
                "\"status\": \"ok\", \"payload\": {}, "
                "\"error_code\": None}"
            ),
            "request ID",
        ),
        (
            (
                "sys.stdout.buffer.write(b'{not-json}\\n'); "
                "sys.stdout.buffer.flush(); continue"
            ),
            "canonical response",
        ),
    ],
)
def test_jsonl_bridge_rejects_untrusted_worker_responses(
    tmp_path,
    response_source,
    error_match,
):
    from mub.vnext.external.bridge import (
        BridgeProtocolError,
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "bad_worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            for line in sys.stdin.buffer:
                request = json.loads(line)
                {response_source}
                raw = json.dumps(
                    response,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\\n"
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="request-1",
        operation=WorkerOperation.HEALTH,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=5.0,
    ) as bridge:
        with pytest.raises(
            BridgeProtocolError,
            match=error_match,
        ) as exc_info:
            bridge.request(request)
    assert exc_info.value.__cause__ is None


def test_jsonl_bridge_reports_exit_code_before_response(tmp_path):
    from mub.vnext.external.bridge import (
        BridgeProcessError,
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "exit_before_response_worker.py"
    worker.write_text(
        "import os, sys\n"
        "sys.stdin.buffer.readline()\n"
        "os.close(sys.stdout.fileno())\n"
        "os._exit(7)\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="exit-before-response",
        operation=WorkerOperation.HEALTH,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=5.0,
    ) as bridge:
        with pytest.raises(BridgeProcessError, match="exit_code=7"):
            bridge.request(request)


def test_jsonl_bridge_rejects_response_followed_by_worker_failure(tmp_path):
    from mub.vnext.external.bridge import (
        BridgeProcessError,
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "exit_after_response_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            request = json.loads(sys.stdin.buffer.readline())
            response = {
                "schema_version": "memupdatebench.external.worker_response.v1",
                "request_id": request["request_id"],
                "status": "ok",
                "payload": {},
                "error_code": None,
            }
            raw = json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\\n"
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            sys.stderr.write("sk-secret-must-not-leak")
            sys.stderr.flush()
            os.close(sys.stdout.fileno())
            os._exit(7)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="request-1",
        operation=WorkerOperation.HEALTH,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=5.0,
    ) as bridge:
        with pytest.raises(BridgeProcessError) as exc_info:
            bridge.request(request)
    assert "sk-secret" not in str(exc_info.value)


def test_jsonl_bridge_accepts_worker_exit_after_close_response(tmp_path):
    from mub.vnext.external.bridge import (
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "exit_after_close_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            request = json.loads(sys.stdin.buffer.readline())
            response = {
                "schema_version": "memupdatebench.external.worker_response.v1",
                "request_id": request["request_id"],
                "status": "ok",
                "payload": {"closed": True},
                "error_code": None,
            }
            raw = json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\\n"
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            os.close(sys.stdout.fileno())
            os._exit(0)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="close-1",
        operation=WorkerOperation.CLOSE,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=5.0,
    ) as bridge:
        response = bridge.request(request)
    assert response.request_id == "close-1"
    assert response.payload["closed"] is True


def test_jsonl_bridge_rejects_nonzero_exit_after_close_response(tmp_path):
    from mub.vnext.external.bridge import (
        BridgeProcessError,
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "fail_after_close_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            request = json.loads(sys.stdin.buffer.readline())
            response = {
                "schema_version": "memupdatebench.external.worker_response.v1",
                "request_id": request["request_id"],
                "status": "ok",
                "payload": {"closed": True},
                "error_code": None,
            }
            raw = json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\\n"
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            os.close(sys.stdout.fileno())
            os._exit(7)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="close-failed",
        operation=WorkerOperation.CLOSE,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=5.0,
    ) as bridge:
        with pytest.raises(BridgeProcessError, match="exit_code=7"):
            bridge.request(request)


def test_jsonl_bridge_rejects_nonfinite_timeout(tmp_path):
    from mub.vnext.external.bridge import JsonlSubprocessBridge

    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite positive"):
            JsonlSubprocessBridge(
                command=(sys.executable, "-c", "pass"),
                cwd=tmp_path,
                environment={},
                timeout_seconds=value,
            )


def test_jsonl_bridge_timeout_terminates_worker_without_raw_stderr(tmp_path):
    from mub.vnext.external.bridge import (
        BridgeTimeoutError,
        JsonlSubprocessBridge,
        WorkerOperation,
        WorkerRequestV1,
    )

    worker = tmp_path / "timeout_worker.py"
    worker.write_text(
        "import sys, time\n"
        "for line in sys.stdin.buffer:\n"
        "    sys.stderr.write('sk-secret-must-not-leak')\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(10)\n",
        encoding="utf-8",
    )
    request = WorkerRequestV1(
        request_id="request-1",
        operation=WorkerOperation.HEALTH,
        payload={},
    )
    with JsonlSubprocessBridge(
        command=(sys.executable, str(worker)),
        cwd=tmp_path,
        environment=_worker_environment(),
        timeout_seconds=0.1,
    ) as bridge:
        with pytest.raises(BridgeTimeoutError) as exc_info:
            bridge.request(request)
    assert "sk-secret" not in str(exc_info.value)
