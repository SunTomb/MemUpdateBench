"""Redacted local extraction accounting, independent of any HTTP runtime."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any


class ObservationQualificationError(ValueError):
    """The failed attempt is already retained in the recorder's compact rows."""

    def __init__(self, observation: dict[str, Any]):
        self.observation = deepcopy(observation)
        super().__init__('extraction observation blocked: ' + ', '.join(observation['errors']))


def _hash(value: Any) -> str:
    encoded = value if type(value) is str else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError('nonstandard JSON constant')


class ObservationRecorder:
    """Record a decoded output or attach ``record_response`` to a transport.

    The callback takes the *returned* LocalCompletionBoundary envelope, plus
    caller-supplied role/schema metadata. Only ``stop`` with an object JSON
    output and exact usage qualifies. Failures are appended before raising
    ObservationQualificationError; there are no retries or synthesized usage.
    An explicitly injected synchronous private_sink receives raw material in a
    separate payload, never in compact rows. This recorder is per serial run.
    """

    def __init__(self, *, attempt_id_factory: Callable[[], Any] | None = None,
                 private_sink: Callable[[dict[str, Any]], None] | None = None):
        if private_sink is not None and not callable(private_sink):
            raise TypeError('private_sink must be callable')
        self._attempt_id_factory = attempt_id_factory
        self._private_sink = private_sink
        self._rows: list[dict[str, Any]] = []
        self._call_ids: set[str] = set()

    @property
    def rows(self) -> list[dict[str, Any]]:
        return deepcopy(self._rows)

    @staticmethod
    def _text(value: Any) -> bool:
        return type(value) is str and bool(value.strip())

    @staticmethod
    def _usage(value: Any) -> dict[str, int] | None:
        fields = {'prompt_tokens', 'completion_tokens', 'total_tokens'}
        if not isinstance(value, Mapping) or set(value) != fields:
            return None
        if any(type(count) is not int or count < 0 for count in value.values()):
            return None
        if value['total_tokens'] != value['prompt_tokens'] + value['completion_tokens']:
            return None
        return dict(value)

    def record_response(self, response: Any, *, role: str | None = None,
                        schema_name: str | None = None, timestamp: Any = None,
                        private_material: Any = None) -> dict[str, Any]:
        """Transport hook for the exact id/choices/usage completion envelope.

        It is called after each response, including malformed/error envelopes.
        Role and schema_name come from the caller, not from model-generated text.
        """
        envelope = response if isinstance(response, Mapping) else {}
        choices = envelope.get('choices')
        valid_choice = (isinstance(choices, list) and len(choices) == 1
                        and isinstance(choices[0], Mapping))
        choice = choices[0] if valid_choice else {}
        message = choice.get('message')
        message = message if isinstance(message, Mapping) else {}
        return self._record(role=role, schema_name=schema_name,
            usage=envelope.get('usage'), finish_reason=choice.get('finish_reason'),
            raw_output=message.get('content'), call_id=envelope.get('id'), timestamp=timestamp,
            private_material=private_material, response=response,
            initial_errors=[] if valid_choice else ['response_choices'])

    def record(self, *, role: str | None = None, schema_name: str | None = None,
               usage: Mapping[str, Any] | None = None, finish_reason: str | None = None,
               raw_output: str | None = None, call_id: str | None = None,
               timestamp: Any = None, private_material: Any = None) -> dict[str, Any]:
        """Record transport metadata when the caller already decoded the envelope."""
        return self._record(role=role, schema_name=schema_name, usage=usage,
            finish_reason=finish_reason, raw_output=raw_output, call_id=call_id,
            timestamp=timestamp, private_material=private_material)

    def _record(self, *, role, schema_name, usage, finish_reason, raw_output,
                call_id, timestamp, private_material, response=None,
                initial_errors=()) -> dict[str, Any]:
        errors = list(initial_errors)
        observation_id = len(self._rows) + 1
        row = {'observation_id': observation_id, 'attempt_id': observation_id}
        if self._attempt_id_factory is not None:
            try:
                attempt_id = self._attempt_id_factory()
                if type(attempt_id) is not int or attempt_id < 1 or (
                    self._rows and attempt_id <= self._rows[-1]['attempt_id']
                ):
                    raise ValueError('nonmonotonic attempt identifier')
                row['attempt_id'] = attempt_id
            except Exception:
                errors.append('attempt_id')
        for field, value in (('role', role), ('schema_name', schema_name),
                             ('call_id', call_id), ('finish_reason', finish_reason)):
            row[field] = value if type(value) is str else None
            if not self._text(value):
                errors.append(field + '_missing')
        if self._text(call_id):
            if call_id in self._call_ids:
                errors.append('duplicate_call_id')
            self._call_ids.add(call_id)
        if self._text(finish_reason) and finish_reason != 'stop':
            errors.append('finish_reason_incomplete')
        row['usage'] = self._usage(usage)
        if row['usage'] is None:
            errors.append('usage_invalid')
        row.update(output_sha256=None, response_sha256=None,
                   parsed_top_level_type=None, top_level_keys=[])
        if type(raw_output) is not str:
            errors.append('raw_output_missing')
        else:
            row['output_sha256'] = _hash(raw_output)
            try:
                parsed = json.loads(raw_output, parse_constant=_reject_constant)
                shape = ('object' if isinstance(parsed, dict) else
                         'array' if isinstance(parsed, list) else
                         'null' if parsed is None else
                         'boolean' if type(parsed) is bool else
                         'number' if type(parsed) in (int, float) else 'string')
                row['parsed_top_level_type'] = shape
                if shape != 'object':
                    errors.append('nonobject_json')
                else:
                    row['top_level_keys'] = sorted(parsed)
            except (ValueError, TypeError):
                errors.append('malformed_json')
        if response is not None:
            try:
                row['response_sha256'] = _hash(response)
            except (ValueError, TypeError):
                errors.append('response_not_json')
        if timestamp is not None:
            if type(timestamp) not in (str, int, float) or (
                type(timestamp) is float and not math.isfinite(timestamp)
            ):
                errors.append('timestamp_invalid')
            else:
                row['timestamp'] = timestamp
        row['errors'] = errors
        row['status'] = 'BLOCKED' if errors else 'OBSERVED'
        self._rows.append(row)
        if self._private_sink is not None:
            try:
                self._private_sink(deepcopy({'observation_id': observation_id,
                    'raw_output': raw_output, 'response': response,
                    'private_material': private_material}))
            except Exception:
                errors.append('private_sink_failed')
                row['status'] = 'BLOCKED'
        if errors:
            raise ObservationQualificationError(row)
        return deepcopy(row)
