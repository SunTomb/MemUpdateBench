from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json

from mub.vnext.io import sha256_model
from mub.vnext.statistics.contracts_v3 import (
    TASK13_METRIC_PATHS,
    TASK13_SEED_HEX,
    Task13BootstrapConfigV1,
    Task13IntervalV1,
    Task13StatisticStatus,
    canonical_decimal_string,
)


_DOMAIN = b"MUB-Core-Task13-bootstrap-v1\x00"
_U64_LIMIT = 1 << 64
_EXPECTED_CORE_COUNT = 20
_EXPECTED_REPLICATES = 10_000
_EXPECTED_DRAWS = 20
FROZEN_BOOTSTRAP_INDEX_SHA256 = (
    "0d8faf77bc7e4d138f0f9dd3db85ab136f99884906298984202c8dc38c0bbd53"
)


def _tracked_default_config() -> Task13BootstrapConfigV1:
    """Return the tracked config, not an unseeded or implicit PRNG policy."""

    return Task13BootstrapConfigV1(
        cluster_key="semantic_core_id",
        expected_cluster_count=_EXPECTED_CORE_COUNT,
        seed_hex=TASK13_SEED_HEX,
        replicates=_EXPECTED_REPLICATES,
        draws_per_replicate=_EXPECTED_DRAWS,
        confidence_level="0.95",
        interval_method="clustered_percentile",
        quantile_method="inverted_cdf",
        lower_order_statistic=250,
        upper_order_statistic=9_750,
        decimal_precision=50,
        decimal_rounding="ROUND_HALF_EVEN",
        support_policy="all_supported_or_all_unsupported",
        metric_paths=TASK13_METRIC_PATHS,
    )


DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1 = _tracked_default_config()


def _resolve_config(config: Task13BootstrapConfigV1 | None) -> Task13BootstrapConfigV1:
    if config is None:
        return DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1
    if not isinstance(config, Task13BootstrapConfigV1):
        raise TypeError("config must be Task13BootstrapConfigV1")
    return config


def _require_frozen_config(config: Task13BootstrapConfigV1) -> None:
    if config != DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1:
        raise ValueError("bootstrap config must equal the frozen Task 13 config")


def _ordered_core_ids(core_ids: Iterable[str], config: Task13BootstrapConfigV1) -> tuple[str, ...]:
    try:
        candidate_ids = tuple(core_ids)
    except TypeError as exc:
        raise ValueError("core IDs must be an iterable of strings") from exc
    if len(candidate_ids) != config.expected_cluster_count:
        raise ValueError("core IDs must contain exactly 20 entries")
    if any(type(core_id) is not str for core_id in candidate_ids):
        raise ValueError("core IDs must contain only built-in strings")
    if any(not core_id.strip() for core_id in candidate_ids):
        raise ValueError("core IDs must not contain blank strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("core IDs must be unique")
    return tuple(sorted(candidate_ids, key=lambda core_id: core_id.encode("utf-8")))


def _core_ids_sha256(ordered_core_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        list(ordered_core_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _draw_index_v1(
    seed: bytes,
    replicate: int,
    draw: int,
    cluster_count: int,
) -> int:
    threshold = _U64_LIMIT - (_U64_LIMIT % cluster_count)
    attempt = 0
    while True:
        if attempt >= _U64_LIMIT:
            raise RuntimeError("counter stream exhausted its attempt space")
        digest = hashlib.sha256(
            _DOMAIN
            + seed
            + replicate.to_bytes(4, "big")
            + draw.to_bytes(4, "big")
            + attempt.to_bytes(4, "big")
        ).digest()
        value = int.from_bytes(digest[:8], "big")
        if value < threshold:
            return value % cluster_count
        attempt += 1


@dataclass(frozen=True, slots=True)
class BootstrapIndicesV1:
    """Authenticated deterministic semantic-core bootstrap index matrix."""

    ordered_core_ids: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    raw: bytes
    sha256: str
    config: Task13BootstrapConfigV1
    config_sha256: str
    core_ids_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, Task13BootstrapConfigV1):
            raise TypeError("bootstrap config must be Task13BootstrapConfigV1")
        _require_frozen_config(self.config)
        expected_config_hash = sha256_model(self.config)
        if self.config_sha256 != expected_config_hash:
            raise ValueError("bootstrap config hash does not match config")
        ordered_ids = _ordered_core_ids(self.ordered_core_ids, self.config)
        if ordered_ids != self.ordered_core_ids:
            raise ValueError("ordered_core_ids must be sorted by UTF-8 bytes")
        expected_core_hash = _core_ids_sha256(ordered_ids)
        if self.core_ids_sha256 != expected_core_hash:
            raise ValueError("bootstrap core ID hash does not match ordered_core_ids")
        if len(self.rows) != self.config.replicates:
            raise ValueError("bootstrap rows must contain exactly 10,000 replicates")
        if any(len(row) != self.config.draws_per_replicate for row in self.rows):
            raise ValueError("each bootstrap row must contain exactly 20 draws")
        if any(
            type(index) is not int
            or index < 0
            or index >= self.config.expected_cluster_count
            for row in self.rows
            for index in row
        ):
            raise ValueError("bootstrap indices must be integers in [0, 20)")
        if type(self.raw) is not bytes:
            raise ValueError("bootstrap raw matrix must be bytes")
        if len(self.raw) != self.config.replicates * self.config.draws_per_replicate:
            raise ValueError("bootstrap raw matrix must contain exactly 200000 bytes")
        expected_raw = bytes(index for row in self.rows for index in row)
        if self.raw != expected_raw:
            raise ValueError("bootstrap raw matrix does not match rows")
        if self.sha256 != FROZEN_BOOTSTRAP_INDEX_SHA256:
            raise ValueError("bootstrap SHA-256 must equal the frozen canonical matrix digest")
        expected_sha256 = hashlib.sha256(self.raw).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError("bootstrap SHA-256 does not match raw matrix")


# Descriptive aliases keep the return type discoverable without introducing a
# second implementation or a second contract.
BootstrapIndexMatrixV1 = BootstrapIndicesV1
BootstrapMatrixV1 = BootstrapIndicesV1


def build_bootstrap_indices_v1(
    core_ids: Iterable[str],
    config: Task13BootstrapConfigV1 | None = None,
) -> BootstrapIndicesV1:
    """Build the frozen 10,000-by-20 counter-stream index matrix."""

    resolved_config = _resolve_config(config)
    _require_frozen_config(resolved_config)
    ordered_ids = _ordered_core_ids(core_ids, resolved_config)
    seed = bytes.fromhex(resolved_config.seed_hex)
    rows: list[tuple[int, ...]] = []
    raw = bytearray()
    for replicate in range(resolved_config.replicates):
        row = tuple(
            _draw_index_v1(
                seed,
                replicate,
                draw,
                resolved_config.expected_cluster_count,
            )
            for draw in range(resolved_config.draws_per_replicate)
        )
        rows.append(row)
        raw.extend(row)
    raw_bytes = bytes(raw)
    return BootstrapIndicesV1(
        ordered_core_ids=ordered_ids,
        rows=tuple(rows),
        raw=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        config=resolved_config,
        config_sha256=sha256_model(resolved_config),
        core_ids_sha256=_core_ids_sha256(ordered_ids),
    )


def _validate_matrix(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    if not isinstance(matrix, BootstrapIndicesV1):
        raise TypeError("matrix must be BootstrapIndicesV1")
    _require_frozen_config(config)
    if matrix.config != config:
        raise ValueError("bootstrap matrix config does not match supplied config")
    if matrix.config_sha256 != sha256_model(config):
        raise ValueError("bootstrap matrix config hash does not match config")
    if matrix.core_ids_sha256 != _core_ids_sha256(matrix.ordered_core_ids):
        raise ValueError("bootstrap matrix core ID hash does not match ordered cores")
    if matrix.sha256 != FROZEN_BOOTSTRAP_INDEX_SHA256:
        raise ValueError("bootstrap matrix SHA-256 is not the frozen canonical digest")
    if matrix.sha256 != hashlib.sha256(matrix.raw).hexdigest():
        raise ValueError("bootstrap matrix SHA-256 does not match raw bytes")


def _validate_values(
    values: Mapping[str, Decimal], matrix: BootstrapIndicesV1
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping from core ID to Decimal")
    expected = set(matrix.ordered_core_ids)
    if len(values) != len(expected) or set(values) != expected:
        raise ValueError("values must contain exactly the matrix core IDs")
    for core_id in matrix.ordered_core_ids:
        value = values[core_id]
        if type(value) is not Decimal:
            raise TypeError("bootstrap values must be Decimal instances")
        if not value.is_finite():
            raise ValueError("bootstrap values must be finite")


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("statistics must be finite")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered == "-0" or rendered == "":
        rendered = "0"
    return canonical_decimal_string(rendered)


def _replicate_means(
    values: Mapping[str, Decimal],
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> tuple[Decimal, ...]:
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        denominator = Decimal(config.draws_per_replicate)
        means: list[Decimal] = []
        for row in matrix.rows:
            total = Decimal(0)
            for index in row:
                total += values[matrix.ordered_core_ids[index]]
            means.append(total / denominator)
        return tuple(means)


def _point_mean(
    values: Mapping[str, Decimal],
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> Decimal:
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        total = Decimal(0)
        for core_id in matrix.ordered_core_ids:
            total += values[core_id]
        return total / Decimal(config.expected_cluster_count)


def type1_percentile_endpoints_v1(
    replicate_values: Sequence[Decimal] | Iterable[Decimal],
    config: Task13BootstrapConfigV1 | None = None,
) -> tuple[Decimal, Decimal]:
    """Return exact Type-1 inverted-CDF endpoints at the tracked offsets."""

    resolved_config = _resolve_config(config)
    _require_frozen_config(resolved_config)
    try:
        values = tuple(replicate_values)
    except TypeError as exc:
        raise ValueError("replicate values must be an iterable") from exc
    if len(values) != resolved_config.replicates:
        raise ValueError("replicate values must contain exactly 10,000 entries")
    if any(type(value) is not Decimal for value in values):
        raise TypeError("replicate values must be Decimal instances")
    if any(not value.is_finite() for value in values):
        raise ValueError("replicate values must be finite")
    ordered = sorted(values)
    lower_index = resolved_config.lower_order_statistic - 1
    upper_index = resolved_config.upper_order_statistic - 1
    if not 0 <= lower_index < upper_index < len(ordered):
        raise ValueError("Type-1 order statistics are outside the replicate range")
    return ordered[lower_index], ordered[upper_index]


def _numeric_interval(
    point: Decimal,
    replicate_values: Sequence[Decimal],
    config: Task13BootstrapConfigV1,
) -> Task13IntervalV1:
    lower, upper = type1_percentile_endpoints_v1(replicate_values, config=config)
    return Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC,
        estimate=_decimal_string(point),
        lower=_decimal_string(lower),
        upper=_decimal_string(upper),
    )


def clustered_percentile_interval_v1(
    values: Mapping[str, Decimal],
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1 | None = None,
) -> Task13IntervalV1:
    """Compute a deterministic semantic-core clustered percentile interval."""

    resolved_config = matrix.config if config is None else _resolve_config(config)
    _validate_matrix(matrix, resolved_config)
    _validate_values(values, matrix)
    replicate_means = _replicate_means(values, matrix, resolved_config)
    point = _point_mean(values, matrix, resolved_config)
    return _numeric_interval(point, replicate_means, resolved_config)


def paired_percentile_interval_v1(
    left_values: Mapping[str, Decimal],
    right_values: Mapping[str, Decimal],
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1 | None = None,
) -> Task13IntervalV1:
    """Compute a paired interval using left-minus-right on identical rows."""

    resolved_config = matrix.config if config is None else _resolve_config(config)
    _validate_matrix(matrix, resolved_config)
    _validate_values(left_values, matrix)
    _validate_values(right_values, matrix)
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        denominator = Decimal(resolved_config.draws_per_replicate)
        replicate_means: list[Decimal] = []
        for row in matrix.rows:
            total = Decimal(0)
            for index in row:
                core_id = matrix.ordered_core_ids[index]
                total += left_values[core_id] - right_values[core_id]
            replicate_means.append(total / denominator)
        point_total = Decimal(0)
        for core_id in matrix.ordered_core_ids:
            point_total += left_values[core_id] - right_values[core_id]
        point = point_total / Decimal(resolved_config.expected_cluster_count)
    return _numeric_interval(point, tuple(replicate_means), resolved_config)


__all__ = [
    "BootstrapIndexMatrixV1",
    "BootstrapIndicesV1",
    "BootstrapMatrixV1",
    "DEFAULT_TASK13_BOOTSTRAP_CONFIG_V1",
    "FROZEN_BOOTSTRAP_INDEX_SHA256",
    "build_bootstrap_indices_v1",
    "clustered_percentile_interval_v1",
    "paired_percentile_interval_v1",
    "type1_percentile_endpoints_v1",
]
