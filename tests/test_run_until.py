"""Tests for adaptive, convergence-driven simulation (Simulation.run_until)."""

from __future__ import annotations

import numpy as np
import pytest

from mc_lib import Normal, Simulation, Uniform


def area_trial(w: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    area = w * h
    return {"area": area, "passed": area >= 48.0}


def make_sim() -> Simulation:
    return Simulation(
        inputs={"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)},
        trial=area_trial,
        outputs={"area": ["mean", "std", "count"], "passed": ["mean"]},
    )


def test_run_until_reaches_target_se() -> None:
    r = make_sim().run_until("area", target_se=1e-3, seed=1, chunk_size=20_000)
    assert r.converged
    assert r.standard_error is not None and r.standard_error <= 1e-3
    assert r.target == "area"
    assert r.value("area", "mean") == pytest.approx(50.0, abs=0.02)


def test_run_until_matches_batch_prefix() -> None:
    # Same seed + chunk size => the adaptive run's first N samples are exactly a
    # batch run of N, so the means must agree bit-for-bit.
    sim = make_sim()
    r = sim.run_until("area", target_se=2e-3, seed=7, chunk_size=25_000)
    b = sim.run(r.n_trials, chunk_size=25_000, seed=7)
    assert r.value("area", "mean") == pytest.approx(b.value("area", "mean"), rel=1e-12)
    assert r.value("area", "std") == pytest.approx(b.value("area", "std"), rel=1e-12)


def test_run_until_within_criterion() -> None:
    r = make_sim().run_until(
        "area", within=1e-3, seed=2, chunk_size=20_000, max_trials=2_000_000
    )
    assert r.converged
    assert r.n_trials > 0


def test_run_until_max_trials_caps() -> None:
    r = make_sim().run_until(
        "area", target_se=1e-12, seed=3, chunk_size=10_000, max_trials=30_000
    )
    assert not r.converged
    assert r.n_trials <= 30_000


def test_run_until_invalid_target_raises() -> None:
    with pytest.raises(ValueError):
        make_sim().run_until("nope", target_se=1e-2)


def test_run_until_reproducible() -> None:
    sim = make_sim()
    a = sim.run_until("area", target_se=3e-3, seed=5, chunk_size=20_000)
    b = sim.run_until("area", target_se=3e-3, seed=5, chunk_size=20_000)
    assert a.to_dict() == b.to_dict()
    assert a.n_trials == b.n_trials


def test_run_until_quantile_stays_sound() -> None:
    sim = Simulation(
        inputs={"x": Uniform(0.0, 1.0)},
        trial=lambda x: {"x": x},
        outputs={"x": [("q", 0.5), "mean"]},
        quantile_sample_size=50_000,
    )
    r = sim.run_until("x", target_se=1e-3, seed=1, chunk_size=20_000, max_trials=2_000_000)
    assert r.value("x", "q0.5") == pytest.approx(0.5, abs=0.02)
    if r.n_trials > 50_000:
        assert "x" in r.approximate_columns
