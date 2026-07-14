"""Tests for the composable Monte Carlo estimators."""

from __future__ import annotations

import math

import numpy as np

from polars_mc.distributions import Normal
from polars_mc.estimate import (
    Estimate,
    estimate_pi,
    expectation,
    integrate,
    probability,
)


def test_estimate_pi_hits_target_se() -> None:
    e = estimate_pi(target_se=2e-3, seed=1)
    assert e.converged
    assert e.standard_error <= 2e-3
    # The truth should sit within a few standard errors of the estimate.
    assert abs(e.value - math.pi) <= 5 * e.standard_error


def test_integrate_sin_over_0_pi() -> None:
    e = integrate(lambda x: np.sin(x), 0.0, math.pi, target_se=1e-3, seed=2)
    assert e.converged
    assert abs(e.value - 2.0) <= 5 * e.standard_error


def test_expectation_of_x_squared_is_variance() -> None:
    e = expectation(lambda x: x**2, Normal(0.0, 1.0), target_se=2e-3, seed=3)
    assert abs(e.value - 1.0) <= 5 * e.standard_error


def test_probability_of_positive_normal() -> None:
    e = probability(lambda x: x > 0, Normal(0.0, 1.0), target_se=1e-3, seed=4)
    assert abs(e.value - 0.5) <= 5 * e.standard_error


def test_estimate_is_reproducible() -> None:
    a = estimate_pi(target_se=5e-3, seed=11)
    b = estimate_pi(target_se=5e-3, seed=11)
    assert a == b
    assert isinstance(a, Estimate)


def test_max_trials_caps_when_target_unreachable() -> None:
    # An impossibly tight target inside a tiny budget must stop, not converge.
    e = expectation(
        lambda x: x,
        Normal(0.0, 1.0),
        target_se=1e-12,
        max_trials=20_000,
        chunk=10_000,
        seed=5,
    )
    assert not e.converged
    assert e.n_samples <= 20_000


def test_within_convergence_criterion() -> None:
    e = expectation(
        lambda x: x**2,
        Normal(0.0, 1.0),
        within=1e-2,
        chunk=50_000,
        seed=6,
    )
    assert e.converged
    assert e.n_samples > 0
