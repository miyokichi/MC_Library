"""Tests for the polars_mc Monte Carlo engine."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars_mc import (
    Bernoulli,
    Constant,
    Exponential,
    LogNormal,
    Normal,
    Simulation,
    Triangular,
    Uniform,
)
from polars_mc.aggregate import Moments, parse_outputs
from polars_mc.chunk import classify_trial
from polars_mc.rng import spawn_generators


def rectangle_trial(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        area=pl.col("w") * pl.col("h"),
    ).with_columns(
        passed=pl.col("area") >= 48.0,
    )


def rectangle_array_trial(w: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    area = w * h
    return {"area": area, "passed": area >= 48.0}


def make_sim(**kwargs: object) -> Simulation:
    return Simulation(
        inputs={"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)},
        trial=rectangle_trial,
        outputs={
            "area": ["mean", "std", "min", "max", "count", ("q", 0.95)],
            "passed": ["mean"],
        },
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_is_identical() -> None:
    sim = make_sim()
    a = sim.run(200_000, chunk_size=50_000, seed=7)
    b = sim.run(200_000, chunk_size=50_000, seed=7)
    assert a.to_dict() == b.to_dict()


def test_different_seed_differs() -> None:
    sim = make_sim()
    a = sim.run(200_000, seed=1)
    b = sim.run(200_000, seed=2)
    assert a.value("area", "mean") != b.value("area", "mean")


def test_chunk_size_does_not_change_seed_streams() -> None:
    # Same seed + same chunk size => identical, regardless of call.
    sim = make_sim()
    a = sim.run(120_000, chunk_size=40_000, seed=3)
    b = sim.run(120_000, chunk_size=40_000, seed=3)
    assert a.value("area", "std") == b.value("area", "std")


# --------------------------------------------------------------------------- #
# Correctness against analytic values
# --------------------------------------------------------------------------- #


def test_matches_analytic_moments() -> None:
    # E[w*h] = 50; Var(w*h) for independent normals.
    sim = make_sim()
    r = sim.run(2_000_000, seed=42)

    expected_mean = 10.0 * 5.0
    expected_var = (100.0 + 0.2**2) * (25.0 + 0.1**2) - expected_mean**2
    expected_std = math.sqrt(expected_var)

    assert r.value("area", "mean") == pytest.approx(expected_mean, abs=5e-3)
    assert r.value("area", "std") == pytest.approx(expected_std, rel=5e-3)
    assert r.value("area", "count") == 2_000_000


def test_bernoulli_rate_matches_p() -> None:
    sim = Simulation(
        inputs={"x": Bernoulli(0.3)},
        trial=lambda df: df,
        outputs={"x": ["mean"]},
    )
    r = sim.run(1_000_000, seed=0)
    assert r.value("x", "mean") == pytest.approx(0.3, abs=2e-3)


# --------------------------------------------------------------------------- #
# Moment merge correctness (the heart of chunk aggregation)
# --------------------------------------------------------------------------- #


def _moments_of(values: np.ndarray) -> Moments:
    mean = float(values.mean())
    return Moments(
        count=values.size,
        mean=mean,
        m2=float(((values - mean) ** 2).sum()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def test_moment_merge_equals_global() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 5.0, size=100_003)

    merged = Moments()
    for part in np.array_split(values, 17):
        merged = merged.merge(_moments_of(part))

    assert merged.count == values.size
    assert merged.mean == pytest.approx(float(values.mean()), rel=1e-12)
    assert merged.variance == pytest.approx(float(values.var(ddof=1)), rel=1e-9)
    assert merged.minimum == float(values.min())
    assert merged.maximum == float(values.max())


def test_moment_merge_identity_with_empty() -> None:
    m = _moments_of(np.arange(10.0))
    assert Moments().merge(m) == m
    assert m.merge(Moments()) == m


# --------------------------------------------------------------------------- #
# Quantiles
# --------------------------------------------------------------------------- #


def test_quantile_exact_when_within_budget() -> None:
    # n_trials <= budget => not flagged approximate.
    sim = make_sim(quantile_sample_size=500_000)
    r = sim.run(100_000, seed=5)
    assert "area" not in r.approximate_columns


def test_quantile_flagged_approximate_when_exceeding_budget() -> None:
    sim = make_sim(quantile_sample_size=50_000)
    r = sim.run(500_000, seed=5)
    assert "area" in r.approximate_columns


def test_quantile_estimate_is_reasonable() -> None:
    sim = make_sim(quantile_sample_size=200_000)
    r = sim.run(1_000_000, seed=9)
    q95 = r.value("area", "q0.95")
    # 95th percentile of area should sit well above the mean (~50).
    assert 51.0 < q95 < 54.0


# --------------------------------------------------------------------------- #
# RNG layer
# --------------------------------------------------------------------------- #


def test_spawn_generators_are_independent_and_reproducible() -> None:
    a = spawn_generators(42, 4)
    b = spawn_generators(42, 4)
    samples_a = [g.random(5) for g in a]
    samples_b = [g.random(5) for g in b]
    for sa, sb in zip(samples_a, samples_b):
        assert np.array_equal(sa, sb)
    # Distinct streams across chunks.
    assert not np.array_equal(samples_a[0], samples_a[1])


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dist",
    [
        Normal(0.0, 1.0),
        Uniform(0.0, 1.0),
        LogNormal(0.0, 0.5),
        Triangular(0.0, 0.5, 1.0),
        Exponential(2.0),
        Bernoulli(0.5),
        Constant(3.0),
    ],
)
def test_distribution_sample_shape(dist: object) -> None:
    g = np.random.default_rng(0)
    out = dist.sample(g, 100)  # type: ignore[attr-defined]
    assert out.shape == (100,)


def test_constant_is_constant() -> None:
    g = np.random.default_rng(0)
    out = Constant(7.5).sample(g, 50)
    assert np.all(out == 7.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Normal(0.0, -1.0),
        lambda: Uniform(1.0, 0.0),
        lambda: Bernoulli(1.5),
        lambda: Exponential(0.0),
        lambda: Triangular(1.0, 0.0, 2.0),
    ],
)
def test_invalid_distribution_params_raise(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Validation / errors
# --------------------------------------------------------------------------- #


def test_missing_output_column_raises() -> None:
    sim = Simulation(
        inputs={"w": Normal(1.0, 1.0)},
        trial=lambda df: df,  # never creates "area"
        outputs={"area": ["mean"]},
    )
    with pytest.raises(KeyError):
        sim.run(1000, seed=0)


def test_unknown_statistic_raises() -> None:
    with pytest.raises(ValueError):
        parse_outputs({"x": ["nonsense"]})


def test_empty_inputs_raises() -> None:
    with pytest.raises(ValueError):
        Simulation(inputs={}, trial=lambda df: df, outputs={"x": ["mean"]})


def test_invalid_n_trials_raises() -> None:
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.run(0, seed=0)


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Array-style ("normal function") trials
# --------------------------------------------------------------------------- #


def test_classify_trial() -> None:
    assert classify_trial(rectangle_array_trial, {"w", "h"}) == "array"
    assert classify_trial(rectangle_trial, {"w", "h"}) == "frame"
    assert classify_trial(lambda df: df, {"w", "h"}) == "frame"
    assert classify_trial(lambda **kw: kw, {"w", "h"}) == "array"


def test_array_and_frame_styles_agree() -> None:
    inputs = {"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)}
    outputs = {"area": ["mean", "std", "min", "max"], "passed": ["mean"]}

    frame_sim = Simulation(inputs=inputs, trial=rectangle_trial, outputs=outputs)
    array_sim = Simulation(inputs=inputs, trial=rectangle_array_trial, outputs=outputs)

    # Same seed + same chunking => identical sampled inputs => identical stats,
    # regardless of how the trial is expressed.
    a = frame_sim.run(200_000, chunk_size=50_000, seed=11)
    b = array_sim.run(200_000, chunk_size=50_000, seed=11)
    for col, stats in a.to_dict().items():
        for stat, value in stats.items():
            assert value == pytest.approx(b.value(col, stat), rel=1e-9)


def test_array_trial_subset_of_inputs() -> None:
    # A trial may take only the inputs it needs.
    def trial(w: np.ndarray) -> dict[str, np.ndarray]:
        return {"double": w * 2.0}

    sim = Simulation(
        inputs={"w": Normal(3.0, 1.0), "unused": Normal(0.0, 1.0)},
        trial=trial,
        outputs={"double": ["mean"]},
    )
    r = sim.run(100_000, seed=0)
    assert r.value("double", "mean") == pytest.approx(6.0, abs=0.05)


def test_array_trial_scalar_output_broadcasts() -> None:
    def trial(w: np.ndarray) -> dict[str, object]:
        return {"const": 7.0, "passthrough": w}

    sim = Simulation(
        inputs={"w": Normal(0.0, 1.0)},
        trial=trial,
        outputs={"const": ["mean", "min", "max"]},
    )
    r = sim.run(10_000, seed=0)
    assert r.value("const", "mean") == 7.0
    assert r.value("const", "min") == 7.0
    assert r.value("const", "max") == 7.0


def test_array_trial_wrong_length_raises() -> None:
    def trial(w: np.ndarray) -> dict[str, np.ndarray]:
        return {"bad": w[:-1]}  # wrong length

    sim = Simulation(
        inputs={"w": Normal(0.0, 1.0)},
        trial=trial,
        outputs={"bad": ["mean"]},
    )
    with pytest.raises(ValueError):
        sim.run(1000, seed=0)


def test_array_trial_must_return_dict() -> None:
    sim = Simulation(
        inputs={"w": Normal(0.0, 1.0)},
        trial=lambda w: w * 2,  # returns array, not dict
        outputs={"w": ["mean"]},
        trial_style="array",
    )
    with pytest.raises(TypeError):
        sim.run(1000, seed=0)


# --------------------------------------------------------------------------- #
# Process backend
# --------------------------------------------------------------------------- #


def test_processes_backend_matches_sequential() -> None:
    from mp_helpers import rectangle_array_trial as mp_trial

    sim = Simulation(
        inputs={"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)},
        trial=mp_trial,
        outputs={"area": ["mean", "std", "min", "max"], "passed": ["mean"]},
    )
    seq = sim.run(200_000, chunk_size=25_000, seed=4, backend="sequential")
    par = sim.run(200_000, chunk_size=25_000, seed=4, backend="processes", n_workers=2)
    # Per-chunk seeding makes the parallel result identical, bit for bit.
    assert seq.to_dict() == par.to_dict()


def test_processes_single_chunk_runs() -> None:
    from mp_helpers import rectangle_array_trial as mp_trial

    sim = Simulation(
        inputs={"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)},
        trial=mp_trial,
        outputs={"area": ["mean"]},
    )
    # One chunk -> short-circuits to local execution, still valid.
    r = sim.run(10_000, chunk_size=10_000, seed=4, backend="processes")
    assert r.value("area", "mean") == pytest.approx(50.0, abs=0.05)


def test_processes_rejects_unpicklable_trial() -> None:
    sim = Simulation(
        inputs={"w": Normal(0.0, 1.0)},
        trial=lambda w: {"y": w},  # lambda is not picklable
        outputs={"y": ["mean"]},
        trial_style="array",
    )
    with pytest.raises(ValueError, match="picklable"):
        sim.run(100_000, chunk_size=10_000, seed=0, backend="processes")


def test_unknown_backend_raises() -> None:
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.run(100_000, seed=0, backend="threads")  # type: ignore[arg-type]


def test_to_polars_long_form() -> None:
    sim = Simulation(
        inputs={"x": Normal(0.0, 1.0)},
        trial=lambda df: df,
        outputs={"x": ["mean", "std"]},
    )
    r = sim.run(10_000, seed=0)
    table = r.to_polars()
    assert table.columns == ["column", "statistic", "value"]
    assert table.height == 2
    assert set(table["statistic"].to_list()) == {"mean", "std"}
