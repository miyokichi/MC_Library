"""Monte Carlo estimators, composed from streams and the Moments monoid.

This is the article's骨格 applied to Monte Carlo: take an infinite lazy stream
of random samples, ``map`` each chunk to the quantity of interest, ``scan`` the
running :class:`~polars_mc.aggregate.Moments` (Chan's stable combine), and stop
as soon as the estimate is good enough (a standard-error target, or the
``within`` / ``relative`` convergence tests from the article).  Nothing here is
new machinery -- every estimator is just a different way of gluing the same
parts together::

    running = sample_stream(dist, seed).map(g).map(Moments.from_array) \\
                                       .scan(Moments.merge, Moments())

``expectation`` / ``probability`` / ``integrate`` / ``estimate_pi`` are thin
compositions over that skeleton.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .aggregate import Moments
from .distributions import Distribution, Uniform
from .sampling import DEFAULT_STREAM_CHUNK, input_stream, sample_stream
from .stream import Stream

__all__ = [
    "Estimate",
    "estimate_mean",
    "expectation",
    "probability",
    "integrate",
    "estimate_pi",
]

# A per-sample transform: an array chunk in, an array chunk out (NumPy broadcasts
# the "one trial" code over the whole chunk).  The estimand array may hold any
# numeric dtype, so it is typed as ``NDArray[Any]``.
EstimandChunk = NDArray[Any]
SampleFn = Callable[[NDArray[np.generic]], "EstimandChunk | float"]

DEFAULT_MAX_TRIALS = 5_000_000


@dataclass(frozen=True)
class Estimate:
    """The outcome of an adaptive Monte Carlo estimate.

    Attributes
    ----------
    value:
        The estimated quantity (the running mean of the estimand).
    standard_error:
        Standard error of the mean at the stopping point.
    n_samples:
        Number of samples actually drawn before stopping.
    converged:
        ``True`` if a convergence criterion was met, ``False`` if the run
        stopped because it hit ``max_trials`` first.
    """

    value: float
    standard_error: float
    n_samples: int
    converged: bool

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max_trials reached"
        return (
            f"Estimate(value={self.value:.6g}, "
            f"standard_error={self.standard_error:.3g}, "
            f"n_samples={self.n_samples:,}, {status})"
        )


def running_moments(value_chunks: Stream[EstimandChunk]) -> Stream[Moments]:
    """Lazy stream of cumulative :class:`Moments`, one per consumed chunk.

    ``scan`` seeds the fold with an empty ``Moments``; we ``drop`` that seed so
    the stream starts at the first real (non-empty) accumulation.
    """
    return (
        value_chunks.map(Moments.from_array)
        .scan(Moments.merge, Moments())
        .drop(1)
    )


def _converge(
    moments: Stream[Moments],
    *,
    target_se: float | None,
    within: float | None,
    relative: float | None,
    max_trials: int,
    chunk: int,
) -> Estimate:
    """Drive the lazy ``moments`` stream until a stop condition fires.

    Consumes just enough of the (infinite) stream to satisfy the first of:
    ``standard_error <= target_se``; successive means within absolute ``within``;
    successive means within ``relative`` of the latest mean; or ``max_trials``
    samples drawn.
    """
    max_chunks = max(1, math.ceil(max_trials / chunk))
    prev_mean: float | None = None
    last = Moments()
    converged = False

    for i, m in enumerate(moments, start=1):
        last = m
        if target_se is not None and m.count >= 2 and m.standard_error <= target_se:
            converged = True
            break
        if prev_mean is not None:
            delta = abs(m.mean - prev_mean)
            if within is not None and delta <= within:
                converged = True
                break
            if relative is not None and delta <= relative * abs(m.mean):
                converged = True
                break
        prev_mean = m.mean
        if i >= max_chunks:
            break

    return Estimate(
        value=last.mean,
        standard_error=last.standard_error,
        n_samples=last.count,
        converged=converged,
    )


def estimate_mean(
    value_chunks: Stream[EstimandChunk],
    *,
    target_se: float | None = None,
    within: float | None = None,
    relative: float | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Estimate:
    """Estimate the mean of an estimand given as a stream of sample chunks.

    This is the shared driver behind every estimator below; give it a stream
    whose chunks are the per-sample estimand values and one or more stop
    conditions.  With no convergence criterion it simply runs to ``max_trials``.
    """
    return _converge(
        running_moments(value_chunks),
        target_se=target_se,
        within=within,
        relative=relative,
        max_trials=max_trials,
        chunk=chunk,
    )


def expectation(
    g: SampleFn,
    dist: Distribution,
    *,
    seed: int = 0,
    target_se: float | None = None,
    within: float | None = None,
    relative: float | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Estimate:
    """Estimate ``E[g(X)]`` for ``X ~ dist`` by adaptive Monte Carlo.

    ``g`` is written as if for a single sample; NumPy broadcasts it over each
    chunk.
    """
    chunks = sample_stream(dist, seed, chunk=chunk).map(lambda a: np.asarray(g(a)))
    return estimate_mean(
        chunks,
        target_se=target_se,
        within=within,
        relative=relative,
        max_trials=max_trials,
        chunk=chunk,
    )


def probability(
    pred: SampleFn,
    dist: Distribution,
    *,
    seed: int = 0,
    target_se: float | None = None,
    within: float | None = None,
    relative: float | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Estimate:
    """Estimate ``P(pred(X))`` for ``X ~ dist`` (the mean of a boolean estimand)."""
    return expectation(
        pred,
        dist,
        seed=seed,
        target_se=target_se,
        within=within,
        relative=relative,
        max_trials=max_trials,
        chunk=chunk,
    )


def integrate(
    f: Callable[[NDArray[np.generic]], NDArray[np.generic] | float],
    a: float,
    b: float,
    *,
    seed: int = 0,
    target_se: float | None = None,
    within: float | None = None,
    relative: float | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Estimate:
    """Estimate ``∫_a^b f(x) dx`` by Monte Carlo: ``(b - a) * E[f(U)]``, ``U ~ U(a, b)``."""
    if b == a:
        return Estimate(value=0.0, standard_error=0.0, n_samples=0, converged=True)
    width = b - a
    return expectation(
        lambda x: width * np.asarray(f(x), dtype=np.float64),
        Uniform(a, b),
        seed=seed,
        target_se=target_se,
        within=within,
        relative=relative,
        max_trials=max_trials,
        chunk=chunk,
    )


def estimate_pi(
    *,
    seed: int = 0,
    target_se: float | None = 1e-3,
    max_trials: int = DEFAULT_MAX_TRIALS,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Estimate:
    """Estimate π by the classic unit-circle hit rate, composed from the parts.

    Draw ``(x, y)`` uniformly in the square ``[-1, 1]^2``; the estimand
    ``4 * 1[x^2 + y^2 <= 1]`` has expectation π.
    """
    inputs: dict[str, Distribution] = {
        "x": Uniform(-1.0, 1.0),
        "y": Uniform(-1.0, 1.0),
    }

    def hit_rate(d: dict[str, NDArray[np.generic]]) -> EstimandChunk:
        x = np.asarray(d["x"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.float64)
        return np.asarray(4.0 * (x * x + y * y <= 1.0))

    chunks = input_stream(inputs, seed, chunk=chunk).map(hit_rate)
    return estimate_mean(
        chunks,
        target_se=target_se,
        max_trials=max_trials,
        chunk=chunk,
    )
