from dataclasses import replace
from decimal import Context, Decimal, Inexact, getcontext, setcontext
import hashlib
from typing import Mapping

import pytest

from mub.vnext.statistics.bootstrap_v3 import (
    FROZEN_BOOTSTRAP_INDEX_SHA256,
    BootstrapIndicesV1,
    build_bootstrap_indices_v1,
    clustered_percentile_interval_v1,
    paired_percentile_interval_v1,
    type1_percentile_endpoints_v1,
)
from mub.vnext.statistics.contracts_v3 import (
    TASK13_METRIC_PATHS,
    TASK13_SEED_HEX,
    Task13BootstrapConfigV1,
    Task13IntervalV1,
    Task13StatisticStatus,
)


DOMAIN = b"MUB-Core-Task13-bootstrap-v1\x00"
CORE_IDS = tuple(f"core-{index:03d}" for index in range(80))


def _config(seed_hex: str = TASK13_SEED_HEX) -> Task13BootstrapConfigV1:
    return Task13BootstrapConfigV1(
        cluster_key="semantic_core_id",
        expected_cluster_count=80,
        seed_hex=seed_hex,
        replicates=10_000,
        draws_per_replicate=80,
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


@pytest.fixture(scope="module")
def config() -> Task13BootstrapConfigV1:
    return _config()


@pytest.fixture(scope="module")
def matrix(config: Task13BootstrapConfigV1) -> BootstrapIndicesV1:
    return build_bootstrap_indices_v1(CORE_IDS, config=config)


def _independent_draw(seed: bytes, replicate: int, draw: int, count: int) -> int:
    threshold = (1 << 64) - ((1 << 64) % count)
    attempt = 0
    while True:
        digest = hashlib.sha256(
            DOMAIN
            + seed
            + replicate.to_bytes(4, "big")
            + draw.to_bytes(4, "big")
            + attempt.to_bytes(4, "big")
        ).digest()
        value = int.from_bytes(digest[:8], "big")
        if value < threshold:
            return value % count
        attempt += 1


def test_bootstrap_golden_is_independent_and_order_invariant(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    expected_first = tuple(
        _independent_draw(bytes.fromhex(config.seed_hex), 0, draw, 80)
        for draw in range(8)
    )
    assert matrix.rows[0][:8] == expected_first == (13, 78, 21, 16, 64, 28, 64, 69)
    assert len(matrix.raw) == 800_000
    assert FROZEN_BOOTSTRAP_INDEX_SHA256 == (
        "2b68e56c70cfbbda4777240b9fa8ed61b8c9d006e7201fed536dae50c07c6dee"
    )
    assert matrix.sha256 == FROZEN_BOOTSTRAP_INDEX_SHA256
    assert build_bootstrap_indices_v1(tuple(reversed(CORE_IDS)), config=config) == matrix


def test_self_consistent_alternative_matrix_is_rejected(
    matrix: BootstrapIndicesV1,
) -> None:
    zero_rows = tuple((0,) * 80 for _ in range(10_000))
    zero_raw = b"\x00" * 800_000
    zero_sha256 = hashlib.sha256(zero_raw).hexdigest()
    with pytest.raises(ValueError, match="frozen|canonical|golden"):
        replace(matrix, rows=zero_rows, raw=zero_raw, sha256=zero_sha256)


def test_bootstrap_rejects_wrong_count_duplicate_blank_and_non_string(
    config: Task13BootstrapConfigV1,
) -> None:
    with pytest.raises(ValueError, match="exactly 80"):
        build_bootstrap_indices_v1(CORE_IDS[:-1], config=config)
    with pytest.raises(ValueError, match="unique"):
        build_bootstrap_indices_v1(CORE_IDS[:-1] + (CORE_IDS[0],), config=config)
    with pytest.raises(ValueError, match="blank"):
        build_bootstrap_indices_v1(CORE_IDS[:-1] + ("",), config=config)
    with pytest.raises(ValueError, match="strings"):
        build_bootstrap_indices_v1(CORE_IDS[:-1] + (None,), config=config)  # type: ignore[arg-type]


def test_bootstrap_rows_are_binary_bytes_and_indices_are_in_range(
    matrix: BootstrapIndicesV1,
) -> None:
    assert len(matrix.rows) == 10_000
    assert all(len(row) == 80 for row in matrix.rows)
    assert len(matrix.raw) == 10_000 * 80
    assert set(matrix.raw) <= set(range(80))
    assert all(0 <= index < 80 for row in matrix.rows for index in row)
    assert matrix.raw == bytes(index for row in matrix.rows for index in row)


def test_all_zero_and_all_one_intervals_are_exact(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    zeros = {core_id: Decimal(0) for core_id in CORE_IDS}
    ones = {core_id: Decimal(1) for core_id in CORE_IDS}
    zero = clustered_percentile_interval_v1(zeros, matrix, config=config)
    one = clustered_percentile_interval_v1(ones, matrix, config=config)
    assert zero == Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC, estimate="0", lower="0", upper="0"
    )
    assert one == Task13IntervalV1(
        status=Task13StatisticStatus.NUMERIC, estimate="1", lower="1", upper="1"
    )


def test_paired_identity_is_zero_and_constant_contrast_is_constant(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    left = {core_id: Decimal(index) for index, core_id in enumerate(CORE_IDS)}
    same = paired_percentile_interval_v1(left, left, matrix, config=config)
    assert same.estimate == same.lower == same.upper == "0"

    right = {core_id: Decimal(index) - Decimal("2.5") for index, core_id in enumerate(CORE_IDS)}
    constant = paired_percentile_interval_v1(left, right, matrix, config=config)
    assert constant.estimate == constant.lower == constant.upper == "2.5"


def test_intervals_are_mapping_order_invariant_and_do_not_resample(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {core_id: Decimal(index) for index, core_id in enumerate(CORE_IDS)}
    shuffled = dict(reversed(tuple(values.items())))
    assert clustered_percentile_interval_v1(values, matrix, config=config) == clustered_percentile_interval_v1(
        shuffled, matrix, config=config
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("interval calculation must use the supplied matrix")

    monkeypatch.setattr(
        "mub.vnext.statistics.bootstrap_v3.build_bootstrap_indices_v1", fail_if_called
    )
    clustered_percentile_interval_v1(values, matrix, config=config)


def test_interval_type_and_exact_type1_endpoints() -> None:
    increasing = tuple(Decimal(index) for index in range(10_000))
    lower, upper = type1_percentile_endpoints_v1(increasing)
    assert lower == Decimal(249)
    assert upper == Decimal(9749)
    assert lower < upper


def test_intervals_require_exact_finite_core_mapping(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    values = {core_id: Decimal(1) for core_id in CORE_IDS}
    with pytest.raises(ValueError, match="exactly the matrix core IDs"):
        clustered_percentile_interval_v1(dict(list(values.items())[:-1]), matrix, config=config)
    with pytest.raises(ValueError, match="exactly the matrix core IDs"):
        foreign = dict(values)
        foreign.pop(CORE_IDS[0])
        foreign["foreign-core"] = Decimal(1)
        clustered_percentile_interval_v1(foreign, matrix, config=config)
    with pytest.raises(ValueError, match="finite"):
        nonfinite = dict(values)
        nonfinite[CORE_IDS[0]] = Decimal("NaN")
        clustered_percentile_interval_v1(nonfinite, matrix, config=config)
    with pytest.raises((TypeError, ValueError), match="Decimal"):
        wrong_type = dict(values)
        wrong_type[CORE_IDS[0]] = 1  # type: ignore[assignment]
        clustered_percentile_interval_v1(wrong_type, matrix, config=config)


def test_interval_rejects_matrix_config_and_core_hash_mismatch(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    values = {core_id: Decimal(1) for core_id in CORE_IDS}
    with pytest.raises(ValueError, match="config"):
        clustered_percentile_interval_v1(
            values,
            replace(matrix, config_sha256="0" * 64),
            config=config,
        )
    with pytest.raises(ValueError, match="core"):
        clustered_percentile_interval_v1(
            values,
            replace(matrix, core_ids_sha256="0" * 64),
            config=config,
        )
    with pytest.raises(ValueError, match="raw|sha256"):
        clustered_percentile_interval_v1(
            values,
            replace(matrix, raw=b"\x00" * len(matrix.raw)),
            config=config,
        )




def test_config_is_explicitly_supported_without_implicit_prng() -> None:
    first = build_bootstrap_indices_v1(CORE_IDS, config=_config())
    second = build_bootstrap_indices_v1(CORE_IDS, config=_config())
    assert first.raw == second.raw
    assert first.config.seed_hex == TASK13_SEED_HEX


def test_decimal_results_use_an_independent_context(
    matrix: BootstrapIndicesV1,
    config: Task13BootstrapConfigV1,
) -> None:
    long_decimal = Decimal("1." + "1234567890" * 6)
    short_decimal = Decimal("0." + "9876543210" * 6)
    values = {core_id: long_decimal for core_id in CORE_IDS}
    right_values = {core_id: short_decimal for core_id in CORE_IDS}
    expected_cluster = clustered_percentile_interval_v1(values, matrix, config=config)
    expected_paired = paired_percentile_interval_v1(
        values, right_values, matrix, config=config
    )
    previous = getcontext().copy()
    try:
        caller_context = Context(
            prec=7,
            rounding="ROUND_FLOOR",
            Emin=-3,
            Emax=3,
        )
        caller_context.traps[Inexact] = True
        setcontext(caller_context)
        actual_cluster = clustered_percentile_interval_v1(values, matrix, config=config)
        actual_paired = paired_percentile_interval_v1(
            values, right_values, matrix, config=config
        )
        assert actual_cluster.model_dump_json() == expected_cluster.model_dump_json()
        assert actual_paired.model_dump_json() == expected_paired.model_dump_json()
    finally:
        setcontext(previous)
    assert getcontext() == previous
