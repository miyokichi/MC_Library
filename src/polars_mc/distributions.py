"""Input distributions for the Monte Carlo engine.

A :class:`Distribution` only needs to know how to draw ``n`` samples from a
NumPy :class:`~numpy.random.Generator`.  The engine handles seeding,
reproducibility and assembling the samples into a Polars DataFrame.

Polars itself has no rich distribution sampling, so random generation is
delegated to NumPy; the resulting columns are then fed into the user trial as a
Polars frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Distribution",
    "Normal",
    "Uniform",
    "LogNormal",
    "Triangular",
    "Exponential",
    "Bernoulli",
    "Constant",
]


class Distribution(ABC):
    """Base class for a per-trial input distribution.

    One sample corresponds to the input value for a single trial (one row).
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.generic]:
        """Draw ``n`` independent samples as a 1-D array of length ``n``."""


@dataclass(frozen=True)
class Normal(Distribution):
    """Normal (Gaussian) distribution with ``mean`` and standard deviation ``std``."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(f"Normal.std must be >= 0, got {self.std}")

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return rng.normal(self.mean, self.std, size=n)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Continuous uniform distribution on ``[low, high)``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Uniform requires high >= low, got [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return rng.uniform(self.low, self.high, size=n)


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal distribution.

    ``mu`` and ``sigma`` are the mean and standard deviation of the underlying
    normal distribution (i.e. ``log(X) ~ Normal(mu, sigma)``).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"LogNormal.sigma must be >= 0, got {self.sigma}")

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return rng.lognormal(self.mu, self.sigma, size=n)


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular distribution with ``left <= mode <= right``."""

    left: float
    mode: float
    right: float

    def __post_init__(self) -> None:
        if not (self.left <= self.mode <= self.right):
            raise ValueError(
                f"Triangular requires left <= mode <= right, "
                f"got ({self.left}, {self.mode}, {self.right})"
            )

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return rng.triangular(self.left, self.mode, self.right, size=n)


@dataclass(frozen=True)
class Exponential(Distribution):
    """Exponential distribution with the given ``scale`` (= 1 / rate)."""

    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"Exponential.scale must be > 0, got {self.scale}")

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return rng.exponential(self.scale, size=n)


@dataclass(frozen=True)
class Bernoulli(Distribution):
    """Bernoulli distribution: ``True`` with probability ``p``."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"Bernoulli.p must be in [0, 1], got {self.p}")

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.bool_]:
        return rng.random(size=n) < self.p


@dataclass(frozen=True)
class Constant(Distribution):
    """Degenerate distribution: every trial gets the same ``value``.

    Useful for fixed parameters that should still appear as an input column.
    """

    value: float

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return np.full(n, self.value, dtype=np.float64)
